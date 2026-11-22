"""LoopBridge for Inkling-2-Mini (26.4B Sparse-MoE Looped Transformer)."""

__all__ = [
    "InklingKVMapAdapter",
    "InklingLoopBridgeSystem",
    "InklingRunner",
    "concat_caches",
    "load_inkling_model",
    "select_cache_rows",
]


def __getattr__(name: str):
    if name in __all__:
        from inkling import model as _model
        return getattr(_model, name)
    raise AttributeError(f"module 'inkling' has no attribute {name!r}")
