"""Kev's backbone spread over several hosts: the decoder layers in contiguous slices, one slice per stage server, driven
by a coordinator that holds no backbone at all.

A Kev pass is prefill only: embeddings, the decoder stack, the final norm, then the pointer head reads a few hidden
states. The stack is the only part that needs the GPU memory, and each layer reads nothing but the hidden states the layer
before it wrote plus what it cached itself (an attention layer its keys and values of the state, a Gated DeltaNet layer its
convolution window and recurrent state). So the layers cut cleanly along the depth: stage k holds layers [lo_k, hi_k), the
first stage also the token embeddings, the last one the final norm, and only the residual stream [rows, tokens, d] crosses
the network, in the backbone's own dtype. Every per-layer cache stays on the stage that owns the layer: a cached state is
one id on the coordinator (RemotePrefix) and, under that id, each stage's own cache of its layers.

Topology: a star. The coordinator sends each pass to stage 0, takes its hidden states, sends them to stage 1, and so on
(kev.stage runs the layers, kev.pipeline only moves tensors and reads the head), so a pass costs 2 x stages network
transfers of the residual stream (a 64k-token state on Kev-27B: 64k x 5120 x 2 bytes = 0.6 GB per hop) and the stages
work one after another, not at once. That is the simplest thing that is exact, and it is what a single request costs on
one host too: the same layers in the same order, with the same per-layer caches, in the same dtype, so the answers equal
the one-host serving path up to floating-point reassociation from other padding and kernel shapes (tests/test_pipeline.py
checks the tiny hybrid and attention-only bases to ~1e-5 in fp32 against DecisionModel: rows, packed, prefix miss and hit,
option isolation). Overlapping stages across requests (stage 0 on the next batch while stage 1 runs this one) and a chain
topology (stage k forwarding to k + 1) are the two throughput steps this leaves open.

What this is for: a checkpoint larger than any one host (several 24-48 GB consumer GPUs on different machines, or a
laptop plus a workstation), or a host whose GPU is too small for the whole model but fine for a third of it. It is not
faster than one host that fits the model: every request pays the network hops.

Wire format: an HTTP POST per stage pass, body = a 4-byte big-endian header length, a JSON header (the pass: form, prefix
id and length, keep id, crop, mask offset, ids to drop) and the tensors as one safetensors blob (`pack` / `unpack`). The
response is the same shape with the hidden states. Hidden states of your documents cross the network between the
coordinator and the stages: put them on a private network or behind TLS, and start the stages with KEV_STAGE_KEY so only
the coordinator holding it can drive them (`LoadOptions.stage_key`).

    # on each host (the slices must tile the backbone; kev.stage fetches only the shards its layers need)
    uv run --extra serve python -m kev.stage --run jaredpalmer/kev-27b --layers 0:22 --port 9100
    uv run --extra serve python -m kev.stage --run jaredpalmer/kev-27b --layers 22:43 --port 9100
    uv run --extra serve python -m kev.stage --run jaredpalmer/kev-27b --layers 43:64 --port 9100
    # anywhere (a CPU box will do: the coordinator holds the tokenizer and the pointer head)
    KEV_STAGES=http://a:9100,http://b:9100,http://c:9100 uv run --extra serve python -m kev.serve --run jaredpalmer/kev-27b --port 8009
    # kev.benchmark --run <ckpt> reads through the same KEV_STAGES (LoadOptions.from_env)
"""
import json
import os
import struct
import threading
import uuid
import weakref

import torch
import torch.nn.functional as F
from safetensors.torch import load as load_tensors, save as save_tensors

from .model import OPT_NONE, ROW_PASS_TOKENS, PointerHead, encode, is_hybrid, probs_one, rows_of, rows_per_pass

STAGE_TIMEOUT = float(os.environ.get("KEV_STAGE_TIMEOUT", "600"))   # seconds one stage pass may take (a 64k-token state on a slow stage)
PROTOCOL = "kev-pipeline/1"                                           # in every stage's info; a coordinator refuses another


def pack(meta, tensors):
    """-> bytes: [4-byte header length][JSON header][safetensors blob of `tensors` (empty when there are none)]."""
    head = json.dumps(meta).encode("utf-8")
    body = save_tensors({k: v.detach().contiguous() for k, v in tensors.items()}) if tensors else b""
    return struct.pack(">I", len(head)) + head + body


def unpack(blob):
    """pack's inverse -> (meta, tensors on the CPU)."""
    if len(blob) < 4: raise ValueError("truncated pipeline message")
    n = struct.unpack(">I", blob[:4])[0]
    meta = json.loads(blob[4:4 + n].decode("utf-8"))
    return meta, load_tensors(blob[4 + n:]) if len(blob) > 4 + n else {}


def partition(num_layers, stages):
    """The decoder layers cut into `stages` contiguous slices as equal as possible: [(lo, hi), ...]."""
    if not 1 <= stages <= num_layers: raise ValueError(f"{stages} stages for {num_layers} layers")
    return [(num_layers * i // stages, num_layers * (i + 1) // stages) for i in range(stages)]


def pad_rows(ids_rows, pos_rows, pad_id):
    """Right-pad token rows into [N, L] ids / positions and a [N, L] mask (1 = real token), on the CPU: the same layout
    DecisionModel._pad_rows builds on its device."""
    L = max(len(r) for r in ids_rows)
    ids = torch.full((len(ids_rows), L), pad_id, dtype=torch.long)
    pos = torch.zeros((len(ids_rows), L), dtype=torch.long)
    att = torch.zeros((len(ids_rows), L), dtype=torch.long)
    for i, (rid, rpos) in enumerate(zip(ids_rows, pos_rows)):
        ids[i, : len(rid)] = torch.tensor(rid); pos[i, : len(rpos)] = torch.tensor(rpos); att[i, : len(rid)] = 1
    return ids, pos, att


def pad_lists(rows, fill, length=None):
    """Integer lists right-padded with `fill` into one [N, L] tensor."""
    L = max(max(len(r) for r in rows), length or 0)
    out = torch.full((len(rows), L), fill, dtype=torch.long)
    for i, r in enumerate(rows): out[i, : len(r)] = torch.tensor(r)
    return out


class StageError(RuntimeError):
    """A stage answered with an error (its HTTP status and detail), or could not be reached."""


class StageClient:
    """One stage server. `client` is any httpx-style client (httpx2 by default; a FastAPI TestClient in tests)."""

    def __init__(self, url, key=None, timeout=STAGE_TIMEOUT, client=None):
        self.url, self.key, self.timeout = url.rstrip("/"), key, timeout
        self._client = client
        self.info = None

    @property
    def client(self):
        if self._client is None:
            import httpx2
            self._client = httpx2.Client(timeout=self.timeout)
        return self._client

    def _headers(self):
        return {"authorization": f"Bearer {self.key}"} if self.key else {}

    def _check(self, r):
        if r.status_code == 200: return r
        try: detail = r.json().get("detail", r.text)
        except ValueError: detail = r.text
        raise StageError(f"{self.url}: HTTP {r.status_code}: {detail}")

    def _request(self, method, path, **kw):
        try: return self._check(self.client.request(method, self.url + path, headers=self._headers(), timeout=self.timeout, **kw))
        except StageError: raise
        except Exception as e: raise StageError(f"{self.url}: {type(e).__name__}: {e}") from e

    def fetch_info(self):
        self.info = self._request("GET", "/v1/stage").json()
        return self.info

    def forward(self, meta, tensors):
        """-> (tensors, meta) of the stage's answer."""
        return unpack(self._request("POST", "/v1/stage/forward", content=pack(meta, tensors)).content)[::-1]

    def reset(self):
        return self._request("POST", "/v1/stage/reset").json()

    def close(self):
        if self._client is not None and hasattr(self._client, "close"): self._client.close()


def connect(urls, config, fingerprint, key=None, lora_scale=1.0, client=None):
    """-> the stage clients in layer order, checked: every stage speaks PROTOCOL, serves the checkpoint `fingerprint` names
    (head.pt's sha256; the adapter's or shard index's when both sides know it), with the same hidden size and layer count
    as `config`, in one dtype, at this lora_scale, and together they tile [0, num_hidden_layers) exactly once. Every
    stage's caches are reset: a coordinator starts with no state cached anywhere."""
    if not urls: raise ValueError("a pipeline needs at least one stage URL")
    stages = [StageClient(u, key=key, client=client) for u in urls]
    for s in stages: s.fetch_info()
    stages.sort(key=lambda s: s.info["layers"][0])
    n = config.num_hidden_layers
    for s in stages:
        i = s.info
        if i.get("protocol") != PROTOCOL: raise StageError(f"{s.url} speaks {i.get('protocol')!r}, this coordinator {PROTOCOL!r}")
        if i["fingerprint"]["head"] != fingerprint["head"]:
            raise StageError(f"{s.url} serves another checkpoint (head.pt {i['fingerprint']['head'][:12]}, here {fingerprint['head'][:12]}): every stage and the coordinator must be started from the same --run")
        if None not in (i["fingerprint"]["weights"], fingerprint["weights"]) and i["fingerprint"]["weights"] != fingerprint["weights"]:
            raise StageError(f"{s.url} holds other weights than this checkpoint ({i['fingerprint']['weights'][:12]} vs {fingerprint['weights'][:12]})")
        if (i["hidden_size"], i["num_layers"]) != (config.hidden_size, n):
            raise StageError(f"{s.url} serves a backbone of {i['num_layers']} layers x {i['hidden_size']}; this checkpoint has {n} x {config.hidden_size}")
        if abs(float(i.get("lora_scale", 1.0)) - lora_scale) > 1e-9:
            raise StageError(f"{s.url} merged its adapter at lora_scale {i.get('lora_scale')}, the coordinator asks for {lora_scale}")
    weights = {s.info["fingerprint"]["weights"] for s in stages} - {None}
    if len(weights) > 1: raise StageError("the stages hold different weights: " + ", ".join(f"{s.url} {s.info['fingerprint']['weights'][:12]}" for s in stages))
    dtypes = {s.info["dtype"] for s in stages}
    if len(dtypes) > 1: raise StageError("the stages run in different dtypes: " + ", ".join(f"{s.url} {s.info['dtype']}" for s in stages))
    expect = 0
    for s in stages:
        lo, hi = s.info["layers"]
        if lo != expect: raise StageError(f"the stages do not tile the {n} layers: {s.url} starts at layer {lo}, layer {expect} is " + ("served twice" if lo < expect else "missing"))
        expect = hi
    if expect != n: raise StageError(f"the stages do not tile the {n} layers: layers {expect}-{n - 1} are missing")
    for s in stages: s.reset()
    return stages


class RemotePrefix:
    """A state cached on every stage under one id (kev.serve's PrefixCache holds it inside the prefix tuple). When the
    coordinator drops it, the id is queued and released on the stages with the next pass (PipelineDecisionModel._drops)."""

    def __init__(self, model, id, tokens):
        self.id, self.tokens = id, tokens
        self._finalizer = weakref.finalize(self, model._release, id)
        self._finalizer.atexit = False   # the stages may be gone at interpreter exit; they bound their caches themselves


class PipelineDecisionModel:
    """The scoring interface (kev.model.SCORING_INTERFACE) over stage servers. Encoding, the row / packed forms and the
    readout are DecisionModel's; every backbone pass goes through `_run`. The pointer head (fp32, the checkpoint's
    temperature) runs here on `device`."""

    backend = "pipeline"
    graphs = None

    def __init__(self, stages, config, pad_id, device="cpu", head_dim=256, option_isolation=False):
        self.stages, self.config, self.pad_id, self.device = stages, config, pad_id, device
        self.hybrid = is_hybrid(config)
        if self.hybrid and option_isolation: raise ValueError("option_isolation needs the packed mask; not available on hybrid backbones")
        self.option_isolation = option_isolation
        self.head = PointerHead(config.hidden_size, dp=head_dim).to(device)
        self.training = False
        self.session = uuid.uuid4().hex[:12]
        self._counter, self._released, self._lock = 0, [], threading.Lock()

    def eval(self):
        self.head.eval(); self.training = False
        return self

    @property
    def dtype(self):
        return self.stages[0].info["dtype"]

    @property
    def prefix_min_tokens(self):
        return 0   # a miss would run the state once per question row on every stage, however short the state

    def stages_info(self):
        return [{"url": s.url, "layers": s.info["layers"], "device": s.info["device"], "dtype": s.info["dtype"], "attention": s.info["attention"]} for s in self.stages]

    def encode(self, tok, rec, **kw):
        return encode(tok, rec, option_isolation=self.option_isolation, **kw)

    def close(self):
        """Release every state cached on the stages and the connections."""
        for s in self.stages:
            try: s.reset()
            except StageError: pass
            s.close()

    # --- ids of cached states and their release
    def _new_id(self):
        with self._lock:
            self._counter += 1
            return f"{self.session}-{self._counter}"

    def _release(self, id):
        with self._lock: self._released.append(id)

    def _drops(self):
        with self._lock:
            out, self._released = self._released, []
            return out

    # --- the one backbone pass
    def _run(self, ids_rows, pos_rows, form="rows", prefix=None, keep=None, crop=None, segs=None, opts=None):
        """Right-padded rows through every stage in order -> hidden states [N, L, d] fp32 on self.device, after the final
        norm. form "rows": causal rows, continuing the cached state `prefix` (a RemotePrefix) if given. form "packed": the
        block-causal mask built from `segs` (and `opts`, option isolation) on the stages, the mask's query rows starting
        after the prefix. keep: cache the pass's state under this id on every stage (one row), cropped to its first `crop`
        tokens if given (the packed form keeps the state only)."""
        ids, pos, att = pad_rows(ids_rows, pos_rows, self.pad_id)
        offset = prefix.tokens if prefix is not None else 0
        meta = {"form": form, "prefix": prefix.id if prefix is not None else None, "prefix_len": offset, "keep": keep, "crop": crop,
                "offset": offset, "drop": self._drops()}
        tensors = {"ids": ids, "pos": pos, "att": att}
        if segs is not None:
            tensors["seg"] = pad_lists(segs, -1, offset + ids.shape[1])
            if opts is not None: tensors["opt"] = pad_lists(opts, OPT_NONE, offset + ids.shape[1])
        h = None
        for k, stage in enumerate(self.stages):
            t = dict(tensors)
            if k > 0: del t["ids"]; t["h"] = h
            out, _ = stage.forward(meta, t)
            h = out["h"]
        return h.float().to(self.device)

    def _readout(self, h, enc):
        return [self.head(h[d], h[torch.tensor(oi, device=self.device)]) for d, oi in zip(enc["decide_idx"], enc["opt_idx"])]

    def rows_form(self, encs):
        """As DecisionModel.rows_form: hybrid backbones always, attention-only ones past ROW_PASS_TOKENS packed."""
        return self.hybrid or any(len(e["ids"]) > ROW_PASS_TOKENS for e in encs)

    # --- row form: the state once into every stage's cache, the question rows continuing from it (the prefix form, what
    # the MLX backend's forward runs too; the row form with the state recomputed per row is forward_rows, for checks)
    @torch.no_grad()
    def prefix(self, enc):
        """The state tokens through the stages, cached there. -> (n_state_tokens, RemotePrefix, state hidden states [Ls, d]
        fp32 for an attention-only backbone (the packed continuation reads them), None for a hybrid one)."""
        Ls = enc["seg"].count(0)
        id = self._new_id()
        h = self._run([enc["ids"][:Ls]], [enc["pos"][:Ls]], keep=id)
        return Ls, RemotePrefix(self, id, Ls), None if self.hybrid else h[0, :Ls].clone()

    def _branch_logits(self, enc, prefix):
        """Logits per question: its branch as a causal row continuing the cached state, rows_per_pass rows per pass."""
        _, _, rows = rows_of(enc)
        chunk, out = rows_per_pass([r["ids"] for r in rows], prefix.tokens), []
        for start in range(0, len(rows), chunk):
            part = rows[start:start + chunk]
            h = self._run([r["ids"] for r in part], [r["pos"] for r in part], prefix=prefix)
            out += [self.head(h[i, r["decide"]], h[i, torch.tensor(r["opts"], device=self.device)]) for i, r in enumerate(part)]
        return out

    @torch.no_grad()
    def forward_rows(self, enc):
        """The row form as DecisionModel.forward_rows_batch runs it: every question one causal row of state + branch, no
        cache. The reference the prefix form is checked against (tests/test_pipeline.py)."""
        S, Sp, rows = rows_of(enc)
        chunk, out = rows_per_pass([S + r["ids"] for r in rows]), []
        for start in range(0, len(rows), chunk):
            part = rows[start:start + chunk]
            h = self._run([S + r["ids"] for r in part], [Sp + r["pos"] for r in part])
            out += [self.head(h[i, len(S) + r["decide"]], h[i, torch.tensor([len(S) + o for o in r["opts"]], device=self.device)]) for i, r in enumerate(part)]
        return out

    # --- packed form (attention-only backbones): the block-causal mask, as DecisionModel.hidden_batch
    def _packed_hidden(self, encs, keep=None, crop=None):
        isolate = any(e.get("option_isolation") for e in encs)
        if isolate and not all(e.get("option_isolation") for e in encs): raise ValueError("cannot mix option-isolated and plain encodings in one batch")
        return self._run([e["ids"] for e in encs], [e["pos"] for e in encs], form="packed", keep=keep, crop=crop,
                         segs=[e["seg"] for e in encs], opts=[e["opt"] for e in encs] if isolate else None)

    @torch.no_grad()
    def forward_batch(self, encs, shared_prefix=False):
        """List (per record) of lists (per question) of logits; shared_prefix is what this backend always does."""
        if self.rows_form(encs): return [self._branch_logits(e, self.prefix(e)[1]) for e in encs]
        hs = self._packed_hidden(encs)
        return [self._readout(hs[b], e) for b, e in enumerate(encs)]

    def forward(self, enc):
        return self.forward_batch([enc])[0]

    @torch.no_grad()
    def probs(self, enc):
        return [F.softmax(z, -1).cpu() for z in self.forward(enc)]

    @torch.no_grad()
    def probs_and_prefix(self, enc):
        Ls = enc["seg"].count(0)
        if self.rows_form([enc]):
            prefix = self.prefix(enc)
            return [F.softmax(z, -1).cpu() for z in self._branch_logits(enc, prefix[1])], prefix
        id = self._new_id()
        h = self._packed_hidden([enc], keep=id, crop=Ls)[0]
        return [F.softmax(z, -1).cpu() for z in self._readout(h, enc)], (Ls, RemotePrefix(self, id, Ls), h[:Ls].clone())

    @torch.no_grad()
    def probs_with_prefix(self, enc, prefix):
        Ls, remote, h_state = prefix
        if enc["seg"].count(0) != Ls: raise ValueError("prefix does not match this record's state")
        if self.rows_form([enc]): return [F.softmax(z, -1).cpu() for z in self._branch_logits(enc, remote)]
        h = self._run([enc["ids"][Ls:]], [enc["pos"][Ls:]], form="packed", prefix=remote, segs=[enc["seg"]],
                      opts=[enc["opt"]] if enc.get("option_isolation") else None)[0, : len(enc["ids"]) - Ls]
        return [F.softmax(z, -1).cpu() for z in self._readout(torch.cat([h_state, h], 0), enc)]

    @torch.no_grad()
    def probs_batch(self, encs, prefixes, keep):
        """kev.serve's batch call: one request at a time through the stages."""
        out = [probs_one(self, e, p, k) for e, p, k in zip(encs, prefixes, keep)]
        return [o[0] for o in out], [o[1] for o in out]
