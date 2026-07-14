# DiGiT-DIM

See [ARCHITECTURE.md](ARCHITECTURE.md) for the current state of the codebase,
what changed in the ongoing legacy-architecture refactor, and the roadmap.

## `harmonic_llm/` — the productized model package

`harmonic_llm/` is a self-contained, packaged transformer LM (config / builder /
CLI / pytest suite over a large `_core.py`). It shares lineage with the
top-level `model.py` but is the surface where the active model work happens:
generation, tokenization, checkpointing, and the MLA fix. See
[harmonic_llm/README.md](harmonic_llm/README.md) for its own docs.

```
pip install -e "harmonic_llm[dev]"
python3 -m pytest harmonic_llm/tests -q
python3 -m harmonic_llm.cli demo
```

## Development

```
pip install -r requirements.txt
python3 -m unittest discover -s tests -v
```
