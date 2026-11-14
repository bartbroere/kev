"""kev.pipeline + kev.stage: the backbone's layers spread over stage servers answer what one process answers.

Every test runs the stages in-process (FastAPI TestClient over kev.stage.build_app) on the tiny hybrid base of
tests/test_unit.py and a tiny attention-only (Qwen2) base built here, in fp32 on the CPU, and compares against the
one-process DecisionModel the same checkpoint loads into: the row and packed forms, a prefix miss and hit, option
isolation, a merged LoRA adapter. No downloads, no network."""
import json
import struct
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient
from transformers import AutoTokenizer

from kev import pipeline, stage as stage_module
from kev.api import SystemOneRequest, to_record
from kev.checkpoint import Checkpoint, LoadOptions, Meta, write_meta
from kev.model import SPECIAL, DecisionModel
from kev.pipeline import PipelineDecisionModel, StageError, connect, pack, partition, unpack
from kev.stage import build_app, fetch_shards, load_stage
from test_unit import tiny_base  # noqa: F401  (the fixture: a 2-layer Qwen3.5 base with a word-level tokenizer)


def request(state="the customer is charged twice it it", isolation=None):
    body = {"state": state, "model": "kev-latest",
            "questions": {"team": {"type": "choice", "instructions": "which team", "criteria": {"billing": None, "shipping": None, "refund": None}},
                          "angry": {"type": "noul", "instructions": "is the customer angry"},
                          "urgent": {"type": "score", "instructions": "how urgent", "criteria": ["low", "mid", "high"]}}}
    return to_record(SystemOneRequest(**body))[0]


class Router:
    """An httpx-style client routing each stage URL to its in-process app."""

    def __init__(self, apps):
        self.apps, self.clients = apps, {}

    def request(self, method, url, **kw):
        base = url.split("/v1")[0]
        if base not in self.clients: self.clients[base] = TestClient(self.apps[base]).__enter__()
        return self.clients[base].request(method, url.replace(base, ""), **kw)

    def close(self):
        for c in self.clients.values(): c.__exit__(None, None, None)


def full_checkpoint(base_dir, out, dtype=torch.bfloat16, option_isolation=False, temperature=1.5):
    """The base's text model saved as a full-weight Kev checkpoint (as kev.train --full_ft saves one) with a random head."""
    tok = AutoTokenizer.from_pretrained(base_dir)
    torch.manual_seed(1)
    m = DecisionModel(str(base_dir), tok, "cpu", dtype=dtype, option_isolation=option_isolation).eval()
    out.mkdir(); m.lm.save_pretrained(out); tok.save_pretrained(out)
    write_meta(out, Meta(base=str(base_dir), head=m.head.state_dict(), lora=0, weights="full", weights_dtype={torch.bfloat16: "bf16", torch.float32: "fp32"}[dtype],
                         temperature=temperature, option_isolation=option_isolation))
    return Checkpoint(out)


def staged(ck, cuts, dtype=torch.float32, key=None, lora_scale=1.0):
    """-> (PipelineDecisionModel over in-process stages cut at `cuts`, the Stage objects). Loaded as Checkpoint.load
    would through LoadOptions.stages, but with the in-process client."""
    stages = [load_stage(ck, lo, hi, "cpu", dtype=dtype, lora_scale=lora_scale, log=lambda *a: None) for lo, hi in cuts]
    apps = {f"http://stage{i}": build_app(s) for i, s in enumerate(stages)}
    router = Router(apps)
    clients = connect(list(apps), ck.text_config(), ck.fingerprint(), key=key, client=router, lora_scale=lora_scale)
    tok = AutoTokenizer.from_pretrained(ck.meta.base)
    model = PipelineDecisionModel(clients, ck.text_config(), tok.pad_token_id, "cpu", head_dim=ck.meta.head_dim, option_isolation=ck.meta.option_isolation)
    model.head.load_state_dict(ck.meta.head); model.eval(); model.head.temperature = ck.meta.temperature
    return tok, model, stages


def close_to(a, b, tol=1e-5):
    return max(float((x.detach() - y.detach()).abs().max()) for x, y in zip(a, b)) < tol


def test_pack_unpack_and_partition():
    meta = {"form": "rows", "prefix": None, "keep": "s-1", "drop": ["s-0"]}
    tensors = {"ids": torch.tensor([[1, 2, 3]]), "h": torch.randn(1, 3, 4, dtype=torch.bfloat16)}
    blob = pack(meta, tensors)
    assert struct.unpack(">I", blob[:4])[0] == len(json.dumps(meta).encode())
    got_meta, got = unpack(blob)
    assert got_meta == meta and torch.equal(got["ids"], tensors["ids"]) and torch.equal(got["h"], tensors["h"]) and got["h"].dtype == torch.bfloat16
    assert unpack(pack({"x": 1}, {})) == ({"x": 1}, {})
    with pytest.raises(ValueError): unpack(b"\x00")
    assert partition(64, 3) == [(0, 21), (21, 42), (42, 64)] and partition(2, 2) == [(0, 1), (1, 2)] and partition(5, 1) == [(0, 5)]
    with pytest.raises(ValueError): partition(2, 3)


def test_hybrid_pipeline_answers_what_one_process_answers(tiny_base, tmp_path):
    """A full-weight checkpoint on the hybrid base, one layer per stage, in fp32: the row form with the state recomputed
    (forward_rows) and the prefix form (forward / probs: the state once into every stage's cache, the question rows
    continuing it), a prefix hit, and the cache ids released on the stages with the next pass."""
    ck = full_checkpoint(tiny_base / "base", tmp_path / "full")
    tok, ref = ck.load("cpu", LoadOptions(dtype=torch.float32))
    _, pipe, stages = staged(ck, [(0, 1), (1, 2)])
    assert pipe.hybrid and pipe.backend == "pipeline" and pipe.dtype == "float32" and pipe.prefix_min_tokens == 0
    assert [s.info()["layers"] for s in stages] == [[0, 1], [1, 2]] and stages[0].first and not stages[0].last and stages[1].last
    for rec in (request(), request("it is"), request(" ".join(["the customer"] * 40))):
        enc = ref.encode(tok, rec)
        want = ref.probs(enc)
        assert close_to(want, [torch.softmax(z, -1) for z in pipe.forward_rows(enc)])
        assert close_to(want, pipe.probs(enc))
        assert close_to(want, [torch.softmax(z, -1) for z in ref.forward(enc)])   # the reference's two forms agree too
        probs, prefix = pipe.probs_and_prefix(enc)
        assert close_to(want, probs) and prefix[0] == enc["seg"].count(0) and prefix[2] is None
        assert close_to(want, pipe.probs_with_prefix(enc, prefix))
        with pytest.raises(ValueError, match="prefix"): pipe.probs_with_prefix(ref.encode(tok, request("it")), prefix)
        held = prefix[1].id
        assert all(held in s.caches for s in stages)
        del probs, prefix
        assert pipe._released == [held]
        pipe.forward_rows(enc)   # the next pass carries the release (and, caching nothing, leaves none of its own)
        assert not any(held in s.caches for s in stages) and pipe._released == []
    # a hit's answers do not depend on what the stage holds besides: other states cached, then the first one read
    encs = [ref.encode(tok, request(s)) for s in ("it it", "the customer is angry", "charged twice")]
    prefixes = [pipe.probs_and_prefix(e)[1] for e in encs]
    assert all(len(s.caches) == 3 for s in stages)
    assert close_to(ref.probs(encs[0]), pipe.probs_with_prefix(encs[0], prefixes[0]))
    pipe.close()
    assert all(len(s.caches) == 0 for s in stages)


def test_pipeline_serves_through_kev_serve(tiny_base, tmp_path, monkeypatch):
    """kev.serve.Server runs the pipeline model like any backend: the prefix cache holds RemotePrefix handles, a hit runs
    only the question rows, /v1/models names the stages, and stopping the server releases the states on the stages."""
    from kev import serve
    ck = full_checkpoint(tiny_base / "base", tmp_path / "full")
    tok, ref = ck.load("cpu", LoadOptions(dtype=torch.float32))
    _, pipe, stages = staged(ck, [(0, 2)])
    server = serve.Server(SimpleNamespace(release_date=lambda: "2026-01-01", requested="tiny", meta=ck.meta), tok, pipe, "cpu")
    try:
        body = {"state": "the customer is charged twice", "model": "kev-latest",
                "questions": {"team": {"type": "choice", "instructions": "which team", "criteria": {"billing": None, "shipping": None, "refund": None}}}}
        first = server.answer(SystemOneRequest(**body)); second = server.answer(SystemOneRequest(**body))
        assert first["answers"]["team"]["probabilities"] == pytest.approx(second["answers"]["team"]["probabilities"], abs=1e-5)
        want = ref.probs(ref.encode(tok, to_record(SystemOneRequest(**body))[0]))[0]
        assert list(first["answers"]["team"]["probabilities"].values()) == pytest.approx(want.tolist(), abs=1e-4)   # the API rounds to 4 decimals
        assert (server.prefix_cache.misses, server.prefix_cache.hits, len(stages[0].caches)) == (1, 1, 1)
        monkeypatch.setattr(serve, "server", lambda: server)
        with TestClient(serve.app) as client:
            card = client.get("/v1/models").json()["models"][0]
        assert card["backend"] == "pipeline" and card["stages"] == [{"url": "http://stage0", "layers": [0, 2], "device": "cpu", "dtype": "float32", "attention": "eager"}]
    finally:
        server.close()
    server.prefix_cache.clear(); pipe.forward_rows(ref.encode(tok, request("it")))   # the dropped handle's release rides the next pass
    assert len(stages[0].caches) == 0


def test_lora_pipeline_merges_the_adapter_like_the_torch_path(tiny_base, tmp_path):
    """A LoRA checkpoint: each stage folds the adapter's deltas of its own layers into the slice (fp32 B @ A x alpha / r,
    one rounding), the same weights the one-process merge produces, so the answers agree; a lora_scale is honoured and
    must match between the stages and the coordinator."""
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(tiny_base / "base")
    torch.manual_seed(2)
    m = DecisionModel(str(tiny_base / "base"), tok, "cpu")
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
    peft_model = get_peft_model(m.lm, LoraConfig(task_type="FEATURE_EXTRACTION", r=4, lora_alpha=8, target_modules=targets))
    for name, p in peft_model.named_parameters():
        if "lora_B" in name: torch.nn.init.normal_(p, std=0.2)   # peft starts B at zero: a zero delta would test nothing
    out = tmp_path / "lora"; peft_model.save_pretrained(out); tok.save_pretrained(out)
    write_meta(out, Meta(base=str(tiny_base / "base"), head=m.head.state_dict(), lora=4, weights="lora", temperature=1.0))
    ck = Checkpoint(out)
    assert not ck.full and ck.fingerprint()["weights"] is not None
    tok, ref = ck.load("cpu")   # fp32, merged (LoadOptions.merge)
    _, pipe, stages = staged(ck, [(0, 1), (1, 2)])
    merged = [name for name in stages[0].lm.state_dict() if any(t in name for t in targets)]
    assert merged, "the first stage holds no adapted weight"
    for rec in (request(), request("it is angry")):
        enc = ref.encode(tok, rec)
        assert close_to(ref.probs(enc), pipe.probs(enc))
    # the merged slice equals the one-process merged weights bit for bit
    one = ref.lm.state_dict()
    for local, t in stages[1].lm.state_dict().items():
        assert torch.equal(t, one[local.replace("layers.0.", "layers.1.")]), local
    # lora_scale: the stage merges at its scale; a coordinator asking for another is refused
    _, half, _ = staged(ck, [(0, 2)], lora_scale=0.5)
    tok, ref_half = ck.load("cpu", LoadOptions(lora_scale=0.5))
    enc = ref.encode(tok, request())
    assert close_to(ref_half.probs(enc), half.probs(enc)) and not close_to(ref.probs(enc), half.probs(enc), 1e-3)
    scaled = load_stage(ck, 0, 2, "cpu", dtype=torch.float32, lora_scale=0.5, log=lambda *a: None)
    with pytest.raises(StageError, match="lora_scale"):
        connect(["http://s"], ck.text_config(), ck.fingerprint(), client=Router({"http://s": build_app(scaled)}), lora_scale=1.0)
    pipe.close()


def attention_only_base(root):
    """A tiny attention-only base (Qwen2, 3 layers) with the same word-level tokenizer as tiny_base."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    words = "it is charged twice which team billing shipping refund angry the customer".split()
    vocab = {t: i for i, t in enumerate(["<unk>", "<pad>", *SPECIAL, *words])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
    config = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1, pad_token_id=1)
    torch.manual_seed(0)
    Qwen2ForCausalLM(config).to(torch.bfloat16).save_pretrained(root / "base")
    PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>", additional_special_tokens=SPECIAL).save_pretrained(root / "base")
    return root / "base"


@pytest.mark.parametrize("isolation", [False, True])
def test_attention_only_pipeline_runs_the_packed_forms(tmp_path, monkeypatch, isolation):
    """An attention-only backbone runs packed (the block-causal mask built on the stages from the segments), caches the
    state by cropping the packed pass's cache, and continues a packed pass from it; with option isolation too. Past
    ROW_PASS_TOKENS it switches to rows like DecisionModel."""
    base = attention_only_base(tmp_path)
    ck = full_checkpoint(base, tmp_path / "full", option_isolation=isolation)
    tok, ref = ck.load("cpu", LoadOptions(dtype=torch.float32))
    assert not ref.hybrid and ref.option_isolation == isolation
    _, pipe, stages = staged(ck, [(0, 1), (1, 2), (2, 3)])
    assert not pipe.hybrid and pipe.option_isolation == isolation
    for rec in (request(), request("it is"), request(" ".join(["the customer"] * 30))):
        enc = ref.encode(tok, rec)
        assert enc["option_isolation"] == isolation and not pipe.rows_form([enc])
        want = ref.probs(enc)
        assert close_to(want, pipe.probs(enc))
        if not isolation: assert close_to(want, [torch.softmax(z, -1) for z in pipe.forward_rows(enc)])   # rows cannot isolate options (DecisionModel too)
        probs, prefix = pipe.probs_and_prefix(enc)
        Ls = enc["seg"].count(0)
        assert close_to(want, probs) and prefix[0] == Ls and prefix[2].shape == (Ls, 32)
        assert all(s.caches[prefix[1].id]["tokens"] == Ls for s in stages)
        assert close_to(want, pipe.probs_with_prefix(enc, prefix))
        # the same state under one question only hits the same prefix
        more = ref.encode(tok, to_record(SystemOneRequest(state=rec["state"], model="kev-latest", questions={"angry": {"type": "noul", "instructions": "is the customer angry"}}))[0])
        assert close_to(ref.probs(more), pipe.probs_with_prefix(more, prefix))
    # a batch of packed records, as kev.benchmark's forward_batch runs several
    encs = [ref.encode(tok, request(s)) for s in ("it", "the customer is angry twice", "charged")]
    for got, e in zip(pipe.forward_batch(encs), encs):
        assert close_to(ref.probs(e), [torch.softmax(z, -1) for z in got])
    # long packed records run as rows (the prefix form here) and still agree
    monkeypatch.setattr(pipeline, "ROW_PASS_TOKENS", 8)
    import kev.model as M
    monkeypatch.setattr(M, "ROW_PASS_TOKENS", 8)
    enc = ref.encode(tok, request())
    assert pipe.rows_form([enc]) and ref.rows_form([enc])
    if not isolation: assert close_to(ref.probs(enc), pipe.probs(enc))
    pipe.close()


def test_connect_refuses_what_does_not_fit(tiny_base, tmp_path):
    """connect checks the stages tile the layers once, serve this checkpoint (head.pt's hash), share a dtype and speak
    the protocol; a stage that lost a cached state answers 409 and the coordinator raises StageError."""
    ck = full_checkpoint(tiny_base / "base", tmp_path / "full")
    other = full_checkpoint(tiny_base / "base", tmp_path / "other", temperature=2.0)
    config = ck.text_config()
    s0, s1 = (load_stage(ck, lo, hi, "cpu", dtype=torch.float32, log=lambda *a: None) for lo, hi in [(0, 1), (1, 2)])
    s1_bf16 = load_stage(ck, 1, 2, "cpu", dtype=torch.bfloat16, log=lambda *a: None)
    s_other = load_stage(other, 1, 2, "cpu", dtype=torch.float32, log=lambda *a: None)
    apps = {"http://a": build_app(s0), "http://b": build_app(s1), "http://b16": build_app(s1_bf16), "http://other": build_app(s_other)}
    router = Router(apps)
    with pytest.raises(StageError, match="missing"): connect(["http://a"], config, ck.fingerprint(), client=router)
    with pytest.raises(StageError, match="served twice"): connect(["http://a", "http://b", "http://b"], config, ck.fingerprint(), client=router)
    with pytest.raises(StageError, match="another checkpoint"): connect(["http://a", "http://other"], config, ck.fingerprint(), client=router)
    with pytest.raises(StageError, match="dtypes"): connect(["http://a", "http://b16"], config, ck.fingerprint(), client=router)
    with pytest.raises(ValueError, match="at least one"): connect([], config, ck.fingerprint(), client=router)
    stages = connect(["http://b", "http://a"], config, ck.fingerprint(), client=router)   # any order on the command line
    assert [s.url for s in stages] == ["http://a", "http://b"]
    s0.info = lambda: {**stage_module.Stage.info(s0), "protocol": "kev-pipeline/0"}
    with pytest.raises(StageError, match="speaks"): connect(["http://a", "http://b"], config, ck.fingerprint(), client=router)
    del s0.info
    # a coordinator without the weight files compares head.pt only; one with them also the shard index / adapter
    light = Checkpoint(tmp_path / "full", weights=False)
    assert light.fingerprint()["head"] == ck.fingerprint()["head"] and light.fingerprint()["weights"] == ck.fingerprint()["weights"]   # local files: both present
    with pytest.raises(ValueError, match="without its weights"): light.load("cpu")
    # an unknown prefix
    pipe = PipelineDecisionModel(stages, config, 1, "cpu")
    pipe.head.load_state_dict(ck.meta.head); pipe.eval()
    tok = AutoTokenizer.from_pretrained(tiny_base / "base")
    enc = pipe.encode(tok, request())
    ghost = pipeline.RemotePrefix(pipe, "nobody-1", enc["seg"].count(0))
    with pytest.raises(StageError, match="409"): pipe.probs_with_prefix(enc, (enc["seg"].count(0), ghost, None))
    # the stage bounds what it holds when a coordinator never releases
    import kev.stage as S
    stages[0].reset()
    try:
        S.STAGE_CACHES = 2
        for _ in range(4): pipe.prefix(enc)[1]._finalizer.detach()   # never released
        pipe.probs(enc)
        assert len(s0.caches) <= 3   # the bound plus the pass's own state, released next
    finally:
        S.STAGE_CACHES = 16
    router.close()


def test_stage_key_gates_every_route(tiny_base, tmp_path, monkeypatch):
    import kev.stage as S
    ck = full_checkpoint(tiny_base / "base", tmp_path / "full")
    monkeypatch.setattr(S, "STAGE_KEY", "secret")
    s = load_stage(ck, 0, 2, "cpu", dtype=torch.float32, log=lambda *a: None)
    with TestClient(build_app(s)) as client:
        assert client.get("/v1/stage").status_code == 401
        assert client.get("/v1/stage", headers={"authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/v1/stage", headers={"authorization": "Bearer secret"}).json()["layers"] == [0, 2]
        assert client.post("/v1/stage/forward", content=b"").status_code == 401
    router = Router({"http://s": build_app(s)})
    with pytest.raises(StageError, match="401"): connect(["http://s"], ck.text_config(), ck.fingerprint(), client=router)
    assert connect(["http://s"], ck.text_config(), ck.fingerprint(), key="secret", client=router)[0].info["layers"] == [0, 2]


def test_stage_fetches_only_the_shards_its_layers_need(tiny_base, tmp_path, monkeypatch):
    """fetch_shards reads the shard index and, for a Hub checkpoint, downloads the shards holding this stage's tensors
    and nothing else; the Qwen3.5 bases' `model.language_model.` prefix is found from the embedding's name."""
    from safetensors.torch import load_file, save_file
    import huggingface_hub
    src = load_file(tiny_base / "base/model.safetensors")   # Qwen3_5ForCausalLM's names: model.layers.N..., lm_head...
    repo = tmp_path / "repo"; repo.mkdir()
    shards = {"model-00001-of-00002.safetensors": {k: v for k, v in src.items() if "layers.1" not in k},
              "model-00002-of-00002.safetensors": {k: v for k, v in src.items() if "layers.1" in k}}
    for name, part in shards.items(): save_file(part, repo / name)
    (repo / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: name for name, part in shards.items() for k in part}}), encoding="utf-8")
    calls = []
    def fake_download(repo_id, revision=None, allow_patterns=None, **kw):
        calls.append((repo_id, revision, sorted(allow_patterns)))
        return str(repo)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    prefix, located = fetch_shards("someone/tiny@abc", {"layers.1.mlp.up_proj.weight", "norm.weight"})
    assert prefix == "model." and {p.name for p in located.values()} == {"model-00002-of-00002.safetensors", "model-00001-of-00002.safetensors"}
    assert calls[-1] == ("someone/tiny", "abc", ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])
    prefix, located = fetch_shards("someone/tiny@abc", {"layers.1.mlp.up_proj.weight"})
    assert calls[-1][2] == ["model-00002-of-00002.safetensors"] and set(located) == {"layers.1.mlp.up_proj.weight"}
    with pytest.raises(ValueError, match="hold no"): fetch_shards(str(repo), {"layers.9.mlp.up_proj.weight"})
    # kev.checkpoint.resolve_run(weights=False) fetched the metadata only
    assert calls[0][2] == sorted(["*.json", "*.pt", "*.txt", "*.jinja"])
