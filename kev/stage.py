"""One stage of a Kev pipeline (kev.pipeline): a contiguous slice of the backbone's decoder layers, served over HTTP.

    uv run --extra serve python -m kev.stage --run jaredpalmer/kev-27b --layers 0:22 --port 9100
    uv run --extra serve python -m kev.stage --run jaredpalmer/kev-27b --stage 1 --of 3 --port 9100   # partition(64, 3)[1]

Loading: the backbone is built from the checkpoint's config on the meta device, cut to layers [lo, hi) (the first stage
keeps the token embeddings, the last the final norm; every other part is never materialized), and only the tensors of
those layers are read from the safetensors shards, which for a Hub checkpoint means only the shards holding them are
downloaded (the shard index is read first). A LoRA checkpoint's adapter is folded into the slice's weights the way every
served Kev merges it: delta = B @ A x alpha / r (x lora_scale) in fp32, added to the fp32 value of the weight and rounded
once to the load dtype (LoadOptions.merge; kev.mlx_model.merge_lora; scripts/merge_lora_checkpoint.py's formula). So a
stage holds exactly the bytes the one-host serving path would hold for its layers.

Running: a pass is the stage's layers over hidden states it receives (or, on the first stage, embeds), with the masks the
one-host path builds (kev.model.branch_mask_batch for the packed form, causal rows otherwise; a padded query keeps its
diagonal) and the same transformers cache objects, so each layer runs the code it runs on one host. A cached state is a
DynamicCache under the coordinator's id, holding only this stage's layers; a pass continuing it runs on a replica
(kev.model.replicate_cache) and leaves it untouched. The stage keeps at most KEV_STAGE_CACHES states (least recently used
dropped), a bound that only matters if a coordinator dies without releasing its ids (a live one releases them with its
next pass); a pass that names a state the stage no longer holds is a 409.

Precision: CUDA and MPS default to bf16 (a full checkpoint to its saved dtype), the CPU to fp32, as kev.serve does;
--dtype fp32 is the exact path. Attention is SDPA on CUDA and eager elsewhere (--attn). A plain causal pass with no
padding and no prefix hands SDPA no mask, so a long state runs its flash kernel, as on one host.
"""
import argparse
import hmac
import json
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoModel, DynamicCache

from .checkpoint import Checkpoint, is_hub_id, resolve_run
from .device import DEVICE_HELP, DEVICES, select
from .model import branch_mask_batch, is_hybrid, replicate_cache
from .pipeline import PROTOCOL, pack, partition, unpack
from .shared_prefix import _masks

STAGE_CACHES = int(os.environ.get("KEV_STAGE_CACHES", "16"))   # states a stage keeps at most (its coordinator releases what it drops)
STAGE_KEY = os.environ.get("KEV_STAGE_KEY")                      # unset = open stage; set = require Authorization: Bearer <key> (the coordinator's LoadOptions.stage_key)
ADAPTER_PREFIX = "base_model.model."                             # peft's prefix on every adapter tensor name


class UnknownPrefix(KeyError):
    """A pass continues a state this stage does not hold (released, evicted, or never cached here)."""


def shard_index(root):
    """-> (prefix, {tensor name: shard file name}) of the backbone's shards at `root` (a resolved checkpoint or base
    snapshot, its metadata files present): from model.safetensors.index.json, or the single model.safetensors' header.
    `prefix` is what the files put before the text model's own names (`model.language_model.` on the Qwen3.5 bases,
    `model.` on Qwen2.5 / Qwen3, nothing on a full Kev checkpoint), found from the embedding's name."""
    root = Path(root)
    index = root / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    elif (root / "model.safetensors").exists():
        with safe_open(root / "model.safetensors", "pt", device="cpu") as f: weight_map = {k: "model.safetensors" for k in f.keys()}
    else:
        raise FileNotFoundError(f"{root}: neither model.safetensors.index.json nor model.safetensors")
    embed = [k for k in weight_map if k.endswith("embed_tokens.weight")]
    if len(embed) != 1: raise ValueError(f"{root}: expected one embed_tokens.weight in the shard index, found {embed}")
    return embed[0][: -len("embed_tokens.weight")], weight_map


def fetch_shards(source, names):
    """-> {tensor name: shard path} for the backbone tensors `names` (the text model's own names) from `source` (a
    checkpoint directory, or a Hub id / `base@revision` whose metadata is fetched first and whose needed shards only
    are then downloaded)."""
    root = resolve_run(source, weights=False)
    prefix, weight_map = shard_index(root)
    missing = [n for n in names if prefix + n not in weight_map]
    if missing: raise ValueError(f"{source}: the shards hold no {missing[:3]} (prefix {prefix!r})")
    files = {weight_map[prefix + n] for n in names}
    if is_hub_id(source):
        from huggingface_hub import snapshot_download
        repo, _, revision = str(source).partition("@")
        root = snapshot_download(repo, revision=revision or None, allow_patterns=sorted(files))
    return prefix, {n: Path(root) / weight_map[prefix + n] for n in names}


def read_tensors(located, prefix, dtype):
    """-> {name: tensor} read from the shards `located` maps them to (fetch_shards), one shard open at a time, in `dtype`
    (from_pretrained casts every floating tensor to its dtype the same way: the few fp32 tensors of a bf16 base too)."""
    by_file = {}
    for name, path in located.items(): by_file.setdefault(path, []).append(name)
    out = {}
    for path, names in by_file.items():
        with safe_open(path, "pt", device="cpu") as f:
            for n in names:
                t = f.get_tensor(prefix + n)
                out[n] = t.to(dtype) if t.is_floating_point() else t
    return out


def lora_deltas(ck, lo, hi, scale=1.0):
    """-> {stem: delta fp32} for the adapted weights of layers [lo, hi) of the LoRA checkpoint `ck`, stem being the
    weight's name local to the slice (`layers.{i - lo}.mlp.up_proj`): B @ A x alpha / r x scale (peft's get_delta_weight
    of a plain LoRA Linear; rslora: alpha / sqrt(r)). Refuses what the served merge refuses: DoRA, LoRA biases,
    modules_to_save, trained token embeddings, per-layer rank or alpha patterns, an adapter on anything but the layers."""
    cfg = ck.adapter_config()
    if cfg.get("peft_type") != "LORA": raise ValueError(f"{ck.path}: a pipeline stage merges LoRA adapters, not {cfg.get('peft_type')}")
    for key, bad in (("use_dora", True), ("lora_bias", True), ("modules_to_save", lambda v: bool(v)), ("trainable_token_indices", lambda v: bool(v)),
                     ("rank_pattern", lambda v: bool(v)), ("alpha_pattern", lambda v: bool(v)), ("layers_to_transform", lambda v: v is not None), ("bias", lambda v: v not in (None, "none"))):
        value = cfg.get(key)
        if (bad(value) if callable(bad) else value == bad): raise ValueError(f"{ck.path}: adapter_config {key}={value!r} is not supported by the stage merge")
    s = cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"]) * scale
    pairs = {}
    with safe_open(ck.file("adapter_model.safetensors"), "pt", device="cpu") as f:
        for key in f.keys():
            if not key.startswith(ADAPTER_PREFIX): raise ValueError(f"{ck.path}: unexpected adapter tensor {key}")
            stem, _, part = key[len(ADAPTER_PREFIX):].rpartition(".lora_")   # "layers.3.mlp.up_proj", "A.weight"
            if not stem.startswith("layers.") or part not in ("A.weight", "B.weight"):
                raise ValueError(f"{ck.path}: adapter tensor {key} is not a LoRA weight on a decoder layer (special_embeddings checkpoints cannot be staged)")
            i = int(stem.split(".")[1])
            if lo <= i < hi: pairs.setdefault(f"layers.{i - lo}." + stem.split(".", 2)[2], {})[part[0]] = f.get_tensor(key).float()
    out = {}
    for stem, ab in pairs.items():
        if set(ab) != {"A", "B"}: raise ValueError(f"{ck.path}: adapter tensor {stem} lacks its lora_{'B' if 'A' in ab else 'A'}")
        delta = (ab["B"] @ ab["A"]) * s
        out[stem] = delta.T if cfg.get("fan_in_fan_out") else delta
    return out


def load_stage(ck, lo, hi, device, dtype=None, attn=None, lora_scale=1.0, log=print):
    """-> Stage holding layers [lo, hi) of checkpoint `ck` on `device`. dtype None: the checkpoint's saved dtype for full
    weights, else bf16 on an accelerator and fp32 on the CPU (kev.serve's rule); attn None: SDPA on CUDA, eager elsewhere."""
    config = ck.text_config()
    n = config.num_hidden_layers
    if not 0 <= lo < hi <= n: raise ValueError(f"layers {lo}:{hi} do not lie within the backbone's {n} layers")
    if ck.full and lora_scale != 1: raise ValueError("lora_scale interpolates an adapter; a full-weight checkpoint has none")
    if ck.meta.special_embeddings: raise ValueError("a pipeline does not carry trained token embeddings (special_embeddings checkpoints); use one host")
    if dtype is None:
        dtype = getattr(torch, ck.saved_dtype()) if ck.full else torch.bfloat16 if (str(device) != "cpu" or ck.meta.weights_dtype == "bf16") else torch.float32
    attn = attn or ("sdpa" if str(device).startswith("cuda") else "eager")
    from accelerate import init_empty_weights
    with init_empty_weights():
        lm = AutoModel.from_config(config, attn_implementation=attn)
    lm.layers = nn.ModuleList(list(lm.layers)[lo:hi])   # each layer keeps its layer_idx, the cache index of its mixer
    if lo > 0: lm.embed_tokens = None
    if hi < n: lm.norm = nn.Identity()
    wanted = {}   # local name -> name in the shards (the text model's own naming, layers renumbered from 0 in the slice)
    for local in lm.state_dict():
        if local.startswith("layers."):
            _, j, rest = local.split(".", 2)
            wanted[local] = f"layers.{lo + int(j)}.{rest}"
        else: wanted[local] = local
    t0 = time.time()
    prefix, located = fetch_shards(ck.weights_source(), set(wanted.values()))
    tensors = read_tensors(located, prefix, dtype)
    state = {local: tensors[src] for local, src in wanted.items()}
    merged = 0
    if not ck.full:
        for stem, delta in lora_deltas(ck, lo, hi, lora_scale).items():
            key = stem + ".weight"
            if key not in state: raise ValueError(f"adapter weight {stem} has no base weight in layers {lo}:{hi}")
            state[key] = (state[key].float() + delta).to(dtype); merged += 1
    lm.load_state_dict(state, strict=True, assign=True)
    lm.to(device).eval()
    log(f"stage layers {lo}:{hi} of {n} ({'with embeddings, ' if lo == 0 else ''}{'with the final norm, ' if hi == n else ''}{sum(p.numel() for p in lm.parameters()) / 1e9:.2f}B parameters, "
        f"{merged} adapter deltas merged) on {device} in {str(dtype).removeprefix('torch.')}, {attn} attention, {time.time() - t0:.1f} s")
    return Stage(lm, config, lo, hi, device, dtype, attn, ck, lora_scale)


def crop_cache(cache, tokens):
    """Drop the last `tokens` positions from the layers this cache holds (DynamicCache.crop would touch every layer,
    the ones another stage owns included)."""
    for layer in cache.layers:
        if getattr(layer, "is_initialized", False) and hasattr(layer, "crop"): layer.crop(-tokens)


class Stage:
    """The slice, its caches and the one lock every pass runs under."""

    def __init__(self, lm, config, lo, hi, device, dtype, attn, checkpoint=None, lora_scale=1.0):
        self.lm, self.config, self.lo, self.hi, self.device, self.dtype, self.attn = lm, config, lo, hi, device, dtype, attn
        self.hybrid = is_hybrid(config)
        self.first, self.last = lo == 0, hi == config.num_hidden_layers
        self.checkpoint, self.lora_scale = checkpoint, lora_scale
        self.fingerprint = checkpoint.fingerprint() if checkpoint is not None else {"head": None, "weights": None}
        self.caches, self.lock = OrderedDict(), threading.Lock()
        self.passes = 0

    def info(self):
        return {"protocol": PROTOCOL, "layers": [self.lo, self.hi], "num_layers": self.config.num_hidden_layers, "hidden_size": self.config.hidden_size,
                "hybrid": self.hybrid, "device": str(self.device), "dtype": str(self.dtype).removeprefix("torch."), "attention": self.attn,
                "fingerprint": self.fingerprint, "run": self.checkpoint.requested if self.checkpoint else None, "lora_scale": self.lora_scale,
                "cached_states": len(self.caches), "max_cached_states": STAGE_CACHES, "passes": self.passes}

    def reset(self):
        with self.lock: self.caches.clear()

    def _masks(self, form, att, seg, opt, prefix_len, offset):
        """{"full_attention": 4D mask (bool for SDPA, additive in the model dtype for eager; None = plain causal, SDPA's
        is_causal), "linear_attention": the 2D padding mask (None when nothing is padded)} for this pass, over
        prefix_len cached keys followed by the pass's L tokens."""
        real = att.bool()
        B, L = real.shape
        device = real.device
        if form == "rows":
            if prefix_len == 0 and self.attn == "sdpa" and bool(real.all()):
                full = None
            else:
                keys_real = torch.cat([torch.ones(B, prefix_len, dtype=torch.bool, device=device), real], 1)
                causal = torch.arange(prefix_len + L, device=device)[None, :] <= (prefix_len + torch.arange(L, device=device))[:, None]
                allow = causal[None] & keys_real[:, None, :]
                allow[:, torch.arange(L, device=device), prefix_len + torch.arange(L, device=device)] = True   # a padded query keeps its diagonal
                full = _masks(allow, self.dtype, self.attn)
        elif form == "packed":
            segs = [row[row >= 0].tolist() for row in seg]
            opts = [row[: len(s)].tolist() for row, s in zip(opt, segs)] if opt is not None else None
            additive = branch_mask_batch(segs, device, torch.float32, opts=opts, length=prefix_len + L)
            full = _masks((additive[:, 0] == 0)[:, offset:, :], self.dtype, self.attn)
        else:
            raise ValueError(f"unknown pass form {form!r}")
        return {"full_attention": full, "linear_attention": None if bool(real.all()) else real.to(self.dtype)}

    @torch.no_grad()
    def forward(self, meta, tensors):
        """One pass (kev.pipeline.PipelineDecisionModel._run's header and tensors) -> ({"h": hidden states}, timing)."""
        started = time.perf_counter()
        with self.lock:
            for id in meta.get("drop") or []: self.caches.pop(id, None)
            pos, att = tensors["pos"].to(self.device), tensors["att"].to(self.device)
            B, L = pos.shape
            h = self.lm.embed_tokens(tensors["ids"].to(self.device)) if self.first else tensors["h"].to(self.device)
            if h.dtype != self.dtype: raise ValueError(f"hidden states arrived as {h.dtype}, this stage runs {self.dtype}")
            prefix, prefix_len, keep = meta.get("prefix"), int(meta.get("prefix_len") or 0), meta.get("keep")
            cache = None
            if prefix is not None:
                held = self.caches.get(prefix)
                if held is None: raise UnknownPrefix(f"this stage holds no cached state {prefix}")
                if held["tokens"] != prefix_len: raise ValueError(f"cached state {prefix} holds {held['tokens']} tokens, the pass says {prefix_len}")
                self.caches.move_to_end(prefix)
                cache = replicate_cache(held["cache"], B, self.device)
            elif keep:
                if B != 1: raise ValueError("a pass that caches its state holds one sequence")
                cache = DynamicCache(config=self.config)
            masks = self._masks(meta.get("form", "rows"), att, tensors.get("seg"), tensors.get("opt"), prefix_len, int(meta.get("offset") or 0))
            rope = self.lm.rotary_emb(h, pos[None].expand(3, -1, -1) if self.hybrid else pos)
            for layer in self.lm.layers:
                h = layer(h, position_embeddings=rope, attention_mask=masks[getattr(layer, "block_type", "full_attention")], position_ids=pos, past_key_values=cache)
            if self.last: h = self.lm.norm(h)
            if keep:
                crop = meta.get("crop")
                if crop is not None and crop < L: crop_cache(cache, L - crop)
                self.caches[keep] = {"cache": cache, "tokens": L if crop is None else min(crop, L)}
                while len(self.caches) > STAGE_CACHES: self.caches.popitem(last=False)
            self.passes += 1
            out = h.to("cpu")
        return {"h": out}, {"seconds": round(time.perf_counter() - started, 4), "layers": [self.lo, self.hi]}


# --- the server --------------------------------------------------------------------------------------------------------

def build_app(stage):
    from fastapi import FastAPI, Request
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import JSONResponse, Response
    app = FastAPI(title="kev-stage")

    @app.middleware("http")
    async def auth(request, call_next):
        if STAGE_KEY and not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {STAGE_KEY}"):
            return JSONResponse({"detail": "missing or invalid stage key; send Authorization: Bearer <KEV_STAGE_KEY>"}, 401, {"www-authenticate": "Bearer"})
        return await call_next(request)

    @app.get("/v1/stage")
    def info():
        return stage.info()

    @app.post("/v1/stage/reset")
    def reset():
        stage.reset()
        return {"cached_states": 0}

    @app.post("/v1/stage/forward")
    async def forward(request: Request):
        try:
            meta, tensors = unpack(await request.body())
            out, timing = await run_in_threadpool(stage.forward, meta, tensors)
        except UnknownPrefix as e:
            return JSONResponse({"detail": str(e)}, 409)
        except (ValueError, KeyError) as e:
            return JSONResponse({"detail": f"{type(e).__name__}: {e}"}, 422)
        except Exception as e:   # the pass failed (out of memory, a kernel): the coordinator fails this request, the stage lives on
            return JSONResponse({"detail": f"{type(e).__name__}: {e}"}, 500)
        return Response(pack(timing, out), media_type="application/octet-stream")

    app.state.stage = stage
    return app


def parse_layers(text):
    lo, _, hi = text.partition(":")
    try: lo, hi = int(lo), int(hi)
    except ValueError: raise ValueError(f"--layers takes lo:hi (layers lo to hi - 1), got {text!r}") from None
    return lo, hi


def main(argv=None):
    ap = argparse.ArgumentParser(description="one stage of a Kev pipeline: a slice of the backbone's layers over HTTP")
    ap.add_argument("--run", required=True, help="checkpoint: run directory or Hub id[@rev]; every stage and the coordinator use the same one")
    ap.add_argument("--layers", help="lo:hi, the decoder layers this stage holds (0:22 = layers 0 to 21)")
    ap.add_argument("--stage", type=int, help="with --of: this stage's index in kev.pipeline.partition(num_layers, of)")
    ap.add_argument("--of", type=int, help="how many equal slices the backbone is cut into")
    ap.add_argument("--host", default="127.0.0.1", help="interface to bind; 0.0.0.0 to serve the coordinator on another host")
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--device", choices=("auto", *DEVICES), default="auto", help=DEVICE_HELP)
    ap.add_argument("--dtype", choices=("auto", "bf16", "fp32"), default="auto", help="auto: a full checkpoint's saved dtype, else bf16 on an accelerator, fp32 on the CPU")
    ap.add_argument("--attn", choices=("sdpa", "eager"), default=None, help="default: sdpa on CUDA, eager elsewhere")
    ap.add_argument("--lora_scale", type=float, default=1.0, help="scale the adapter's delta before merging (LoadOptions.lora_scale); the coordinator must ask for the same")
    a = ap.parse_args(argv)
    if (a.layers is None) == (a.stage is None): raise SystemExit("give --layers lo:hi, or --stage i --of n")
    ck = Checkpoint(a.run, weights=False)
    n = ck.text_config().num_hidden_layers
    if a.layers: lo, hi = parse_layers(a.layers)
    else:
        if a.of is None: raise SystemExit("--stage needs --of")
        lo, hi = partition(n, a.of)[a.stage]
    dtype = {"auto": None, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    device = select(a.device, dtype=dtype or torch.bfloat16)
    stage = load_stage(ck, lo, hi, device, dtype=dtype, attn=a.attn, lora_scale=a.lora_scale)
    app = build_app(stage)
    print(f"stage {lo}:{hi} of {ck.requested} on {device} at {a.host}:{a.port}; {'bearer key required (KEV_STAGE_KEY)' if STAGE_KEY else 'open stage: set KEV_STAGE_KEY beyond a private network'}")
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
