"""Tests for --device auto resolution and the devices-per-node visibility check."""

import pytest

import biom3.backend.device as device_mod
from biom3.backend.device import check_devices_per_node, resolve_device


@pytest.mark.parametrize("explicit", ["cpu", "cuda", "xpu"])
def test_explicit_device_passes_through(monkeypatch, explicit):
    monkeypatch.setattr(device_mod, "BACKEND_NAME", "cpu")
    assert resolve_device(explicit, allow_cpu=False) == explicit


@pytest.mark.parametrize("requested", ["auto", None])
@pytest.mark.parametrize("backend", ["cuda", "xpu"])
def test_auto_resolves_to_detected_backend(monkeypatch, requested, backend):
    monkeypatch.setattr(device_mod, "BACKEND_NAME", backend)
    assert resolve_device(requested) == backend
    assert resolve_device(requested, allow_cpu=False) == backend


def test_auto_falls_back_to_cpu_only_when_allowed(monkeypatch):
    monkeypatch.setattr(device_mod, "BACKEND_NAME", "cpu")
    assert resolve_device("auto") == "cpu"
    with pytest.raises(RuntimeError, match="--device cpu"):
        resolve_device("auto", allow_cpu=False)


def test_cpu_is_never_checked(monkeypatch):
    monkeypatch.setattr(device_mod, "_visible_device_count", lambda d: 0)
    check_devices_per_node("cpu", 64)


@pytest.mark.parametrize("requested", [1, 4, None])
def test_request_within_visible_devices_passes(monkeypatch, requested):
    monkeypatch.setattr(device_mod, "_visible_device_count", lambda d: 4)
    check_devices_per_node("cuda", requested)


def test_cuda_request_beyond_visible_devices_fails(monkeypatch):
    monkeypatch.setattr(device_mod, "_visible_device_count", lambda d: 1)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match=r"Requested 4 cuda .* sees 1\. CUDA_VISIBLE_DEVICES=0"):
        check_devices_per_node("cuda", 4)


def test_xpu_composite_hierarchy_is_reported(monkeypatch):
    monkeypatch.setattr(device_mod, "_visible_device_count", lambda d: 6)
    monkeypatch.delenv("ZE_AFFINITY_MASK", raising=False)
    monkeypatch.setenv("ZE_FLAT_DEVICE_HIERARCHY", "COMPOSITE")
    with pytest.raises(RuntimeError, match=r"sees 6\. ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE"):
        check_devices_per_node("xpu", 12)


def test_xpu_affinity_mask_skips_the_check(monkeypatch):
    monkeypatch.setattr(device_mod, "_visible_device_count", lambda d: 1)
    monkeypatch.setenv("ZE_AFFINITY_MASK", "0")
    check_devices_per_node("xpu", 12)
