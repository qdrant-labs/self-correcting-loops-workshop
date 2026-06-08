"""Environment check for the Self-Correcting Agentic Retrieval Loops workshop.

Verifies the VM is at the "Ready" state for the lab:
  - core libraries import at the expected versions
  - FastEmbed exposes the dense / sparse models the lab uses
  - Qdrant is reachable and healthy
  - both lab collections (musique + musique_colbert) are built and populated
  - (with --llm) the Claude agent path answers

Usage:
  python scripts/check_env.py            # libs + models + qdrant
  python scripts/check_env.py --llm      # also ping the Claude agent (via LiteLLM)
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as meta
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import config  # noqa: E402  single source of truth for ids + constants

QDRANT_URL = config.QDRANT_URL

# (import name, distribution name) for the libraries the workshop depends on.
CORE = [
    ("qdrant_client", "qdrant-client"),
    ("fastembed", "fastembed"),
    ("litellm", "litellm"),
    ("anthropic", "anthropic"),
    ("sklearn", "scikit-learn"),
    ("datasets", "datasets"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("onnxruntime", "onnxruntime"),  # transitive via fastembed
]

# The exact models the lab uses (from config). Matched against the FastEmbed
# supported-model lists so a vendor rename surfaces here, not at index time.
DENSE_MODEL = config.DENSE_MODEL
MINICOIL_MODEL = config.MINICOIL_MODEL


def _ok(msg: str) -> None:
    print(f"  ok   {msg}")


def _err(msg: str) -> None:
    print(f"  FAIL {msg}")


def check_imports() -> bool:
    print("[libraries]")
    all_ok = True
    for mod, dist in CORE:
        try:
            importlib.import_module(mod)
            try:
                version = meta.version(dist)
            except Exception:
                version = "?"
            _ok(f"{mod:<16} {version}")
        except Exception as exc:  # noqa: BLE001 - report every failure, don't abort
            _err(f"{mod:<16} {type(exc).__name__}: {exc}")
            all_ok = False
    return all_ok


def _model_names(cls) -> list[str]:
    names = []
    for spec in cls.list_supported_models():
        if isinstance(spec, dict):
            names.append(spec.get("model") or spec.get("model_name") or str(spec))
        else:
            names.append(getattr(spec, "model", None) or getattr(spec, "model_name", str(spec)))
    return names


def check_fastembed_models() -> bool:
    print("[fastembed models]")
    try:
        from fastembed import SparseTextEmbedding, TextEmbedding
    except Exception as exc:  # noqa: BLE001
        _err(f"import fastembed classes: {exc}")
        return False

    ok = True
    dense = _model_names(TextEmbedding)
    sparse = _model_names(SparseTextEmbedding)

    if DENSE_MODEL in dense:
        _ok(f"dense    {DENSE_MODEL}")
    else:
        _err(f"dense    {DENSE_MODEL} NOT in supported list")
        ok = False

    if MINICOIL_MODEL in sparse:
        _ok(f"sparse   {MINICOIL_MODEL}")
    else:
        _err(f"sparse   {MINICOIL_MODEL} NOT in supported list")
        ok = False

    return ok


def check_qdrant() -> bool:
    print("[qdrant]")
    try:
        with urllib.request.urlopen(f"{QDRANT_URL}/healthz", timeout=5) as resp:
            body = resp.read().decode().strip()
        with urllib.request.urlopen(f"{QDRANT_URL}/", timeout=5) as resp:
            info = json.loads(resp.read().decode())
        _ok(f"{QDRANT_URL} healthz={body!r} version={info.get('version')}")
        return True
    except Exception as exc:  # noqa: BLE001
        _err(f"{QDRANT_URL} unreachable: {type(exc).__name__}: {exc}")
        return False


def check_collections() -> bool:
    """Both lab collections exist and are fully populated. This is what the README
    build step (setup_collections.py + setup_colbert.py) produces; without it the
    notebook's Setup cell fails even though Qdrant itself is healthy."""
    print("[collections]")
    try:
        from qdrant_client import QdrantClient
    except Exception as exc:  # noqa: BLE001
        _err(f"import qdrant_client: {exc}")
        return False

    expected = None
    meta_path = config.DATA_DIR / "dataset_meta.json"
    if meta_path.exists():
        try:
            expected = json.loads(meta_path.read_text()).get("n_corpus_docs")
        except Exception:  # noqa: BLE001
            expected = None

    ok = True
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=15)
        for name in (config.COLLECTION, config.COLBERT_COLLECTION):
            if not client.collection_exists(name):
                _err(f"{name} missing - run scripts/setup_collections.py then setup_colbert.py")
                ok = False
                continue
            count = client.count(name, exact=True).count
            if count == 0:
                _err(f"{name} empty - run the setup scripts")
                ok = False
            elif expected and count != expected:
                _err(f"{name} has {count} points, expected {expected} (partial build?)")
                ok = False
            else:
                _ok(f"{name} {count} points")
    except Exception as exc:  # noqa: BLE001
        _err(f"collection check failed: {type(exc).__name__}: {exc}")
        ok = False
    return ok


def check_llms() -> bool:
    """Ping the model path the lab depends on: the Claude agent, via LiteLLM."""
    print("[llms]")
    try:
        from dotenv import load_dotenv

        load_dotenv(config.REPO_ROOT / ".env")
    except Exception as exc:  # noqa: BLE001
        _err(f"load .env: {exc}")
        return False

    ok = True
    if not os.environ.get("ANTHROPIC_API_KEY"):
        _err("ANTHROPIC_API_KEY not set")
        return False

    os.environ.setdefault("LITELLM_LOG", "ERROR")
    import litellm

    litellm.suppress_debug_info = True

    for label, model, max_tokens in (
        ("agent", config.AGENT_MODEL, 16),
    ):
        try:
            resp = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": "Reply with exactly: PONG"}],
                max_tokens=max_tokens,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                _ok(f"{label:<6} {model} -> {text!r}")
            else:
                _err(f"{label:<6} {model} returned empty content (raise max_tokens)")
                ok = False
        except Exception as exc:  # noqa: BLE001
            _err(f"{label:<6} {model}: {type(exc).__name__}: {str(exc)[:200]}")
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="also ping the Claude agent")
    args = parser.parse_args()

    results = {
        "libraries": check_imports(),
        "fastembed models": check_fastembed_models(),
        "qdrant": check_qdrant(),
        "collections": check_collections(),
    }
    if args.llm:
        results["llms"] = check_llms()

    print("\n[summary]")
    for name, ok in results.items():
        print(f"  {'READY' if ok else 'NOT READY':<10} {name}")
    all_ok = all(results.values())
    print(f"\n{'Ready' if all_ok else 'NOT ready - see FAIL lines above'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
