from aiter.ops.flydsl.moe_kernels import _graph_safe_stage2_mode


def _mode(capturing, **overrides):
    kwargs = dict(
        gfx="gfx950",
        capturing=capturing,
        a_dtype="fp8",
        b_dtype="fp8",
        num_experts=256,
        model_dim=6144,
        inter_dim=768,
        topk=5,
    )
    kwargs.update(overrides)
    return _graph_safe_stage2_mode("atomic", **kwargs)


def test_m3_mxfp8_capture_uses_reduce():
    assert _mode(True) == "reduce"


def test_m3_mxfp8_eager_keeps_atomic():
    assert _mode(False) == "atomic"


def test_other_shapes_keep_atomic_during_capture():
    assert _mode(True, topk=9) == "atomic"
    assert _mode(True, b_dtype="fp4") == "atomic"
    assert _mode(True, gfx="gfx942") == "atomic"
