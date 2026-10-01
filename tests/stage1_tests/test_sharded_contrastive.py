"""Row-sharded contrastive losses must equal the dense ones exactly.

The dense implementations build the full M x M similarity matrix on every rank
(O(W^2)); the sharded ones build only [2B, M] per rank (O(W)). They must agree,
otherwise switching to sharded silently changes the objective.

Runs on CPU with tiny tensors -- no distributed, no XPU.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from biom3.Stage1.model import pfam_PEN_CL

pytestmark = pytest.mark.usefixtures("no_process_group")


class _Stub(pfam_PEN_CL):
    """Bare instance exposing only the loss methods (skips encoder construction)."""
    def __init__(self, temperature):
        nn.Module.__init__(self)
        self.temperature = temperature


def _dense_inter(m, z_p, z_t, N):
    """Verbatim re-implementation of compute_inter_loss, per-row (no .mean())."""
    M = 2 * N
    mask = torch.zeros((M, M))
    mask[N:, :N] = torch.eye(N)
    mask[:N, N:] = torch.eye(N)
    mask = mask.bool()
    logits = (z_t @ z_p.T) / m.temperature
    mp = m.set_inf(z_p @ z_p.T, mask)
    mt = m.set_inf(z_t @ z_t.T, mask)
    ml = m.set_inf(logits, mask)
    targets = torch.softmax((mp + mt) / (2 * m.temperature), dim=-1)
    text = (-targets * torch.log_softmax(ml, dim=-1)).sum(1)
    prot = (-targets.T * torch.log_softmax(ml.T, dim=-1)).sum(1)
    return ((prot + text) / 2)


def _dense_intra(m, z_p):
    M = z_p.shape[0]
    cs = (z_p @ z_p.T) / m.temperature
    eye = torch.eye(M, dtype=torch.bool)
    cs = m.set_inf(cs, eye)
    pos_mask = eye.roll(shifts=M // 2, dims=0)
    return -cs[pos_mask] + torch.logsumexp(cs, dim=-1)


# float32, not float64: set_inf() whitelists float32/float16 only
# (RUN1_KNOWN_ISSUES item E). Tolerance set for fp32 accumulation.
TOL = 2e-5


def _run(W, B, D=16, tau=0.8, seed=0):
    torch.manual_seed(seed)
    N, M = W * B, 2 * W * B
    z_p = torch.randn(M, D, dtype=torch.float32)
    z_t = torch.randn(M, D, dtype=torch.float32)
    m = _Stub(tau)

    dense_inter = _dense_inter(m, z_p, z_t, N)
    dense_intra = _dense_intra(m, z_p)

    # rank r owns swiss rows [r*B,(r+1)*B) and pfam rows [N+r*B, N+(r+1)*B)
    rows = [torch.cat([torch.arange(r * B, (r + 1) * B),
                       torch.arange(N + r * B, N + (r + 1) * B)]) for r in range(W)]

    # the all_gather the real code performs: per-row logsumexp for every row
    logZ = torch.empty(M, dtype=torch.float32)
    for ri in rows:
        logZ[ri] = m.inter_row_logsumexp(z_p, z_t, ri)

    max_inter = max_intra = 0.0
    for ri in rows:
        s_inter, _ = m.compute_inter_loss_sharded(z_p, z_t, N, ri, logZ)
        s_intra, _ = m.compute_intra_loss_sharded(z_p, N, ri)
        max_inter = max(max_inter, abs(s_inter.item() - dense_inter[ri].mean().item()))
        max_intra = max(max_intra, abs(s_intra.item() - dense_intra[ri].mean().item()))

    # the mean over ranks must also equal the global mean (DDP averages these)
    glob_inter = sum(m.compute_inter_loss_sharded(z_p, z_t, N, ri, logZ)[0] for ri in rows) / W
    glob_intra = sum(m.compute_intra_loss_sharded(z_p, N, ri)[0] for ri in rows) / W
    return (max_inter, max_intra,
            abs(glob_inter.item() - dense_inter.mean().item()),
            abs(glob_intra.item() - dense_intra.mean().item()))


def _grads(fn, *leaves):
    for l in leaves:
        if l.grad is not None:
            l.grad = None
    fn().backward()
    return [l.grad.clone() for l in leaves]


def _run_grad(W, B, D=16, tau=0.8, seed=0):
    """Values matching is not enough: row_logZ is differentiable in the dense
    path (targets is a softmax), so detaching it would match forward andchange the
    backward silently. Compare gradients too."""
    torch.manual_seed(seed)
    N, M = W * B, 2 * W * B
    z_p = torch.randn(M, D, requires_grad=True)
    z_t = torch.randn(M, D, requires_grad=True)
    m = _Stub(tau)
    rows = [torch.cat([torch.arange(r * B, (r + 1) * B),
                       torch.arange(N + r * B, N + (r + 1) * B)]) for r in range(W)]

    g_dense = _grads(lambda: _dense_inter(m, z_p, z_t, N).mean(), z_p, z_t)

    def sharded_total():
        logZ = torch.cat([m.inter_row_logsumexp(z_p, z_t, ri) for ri in rows])
        order = torch.cat(rows)
        full = torch.empty(M, dtype=logZ.dtype)
        full = full.index_copy(0, order, logZ)
        return sum(m.compute_inter_loss_sharded(z_p, z_t, N, ri, full)[0] for ri in rows) / W

    g_sh = _grads(sharded_total, z_p, z_t)
    return max((a - b).abs().max().item() for a, b in zip(g_dense, g_sh))


def test_sharded_grads_match_dense():
    for W, B in ((2, 3), (4, 2), (3, 5)):
        d = _run_grad(W, B)
        assert d < 1e-4, f"W={W} B={B}: gradient mismatch {d}"


def test_sharded_matches_dense():
    for W, B in ((1, 4), (2, 3), (4, 2), (8, 2), (3, 5)):
        pr_i, pr_a, gl_i, gl_a = _run(W, B)
        assert pr_i < TOL, f"W={W} B={B}: per-rank inter mismatch {pr_i}"
        assert pr_a < TOL, f"W={W} B={B}: per-rank intra mismatch {pr_a}"
        assert gl_i < TOL, f"W={W} B={B}: global inter mismatch {gl_i}"
        assert gl_a < TOL, f"W={W} B={B}: global intra mismatch {gl_a}"


def _naive_unif(z, N, t):
    """log mean exp(-t ||x_i - x_j||^2) over i != j, homolog pairs (i, i+-N) excluded."""
    x = F.normalize(z, dim=1)
    M = x.shape[0]
    d2 = torch.cdist(x, x).pow(2)
    valid = ~torch.eye(M, dtype=torch.bool)
    i = torch.arange(M)
    valid[i, (i + N) % M] = False
    return torch.log(torch.exp(-t * d2[valid]).mean())


def _rank_rows(W, B):
    N = W * B
    return [torch.cat([torch.arange(r * B, (r + 1) * B),
                       torch.arange(N + r * B, N + (r + 1) * B)]) for r in range(W)]


def test_uniformity_dense_matches_naive():
    torch.manual_seed(0)
    m = _Stub(0.8)
    for W, B, t in ((1, 4, 2.0), (3, 5, 2.0), (4, 2, 0.5)):
        N = W * B
        z = torch.randn(2 * N, 16) * 7 + 3   # unnormalized, offset, like LayerNorm output
        d = abs(m.compute_uniformity_loss(z, t).item() - _naive_unif(z, N, t).item())
        assert d < 1e-5, f"W={W} B={B} t={t}: {d}"


def test_uniformity_sharded_matches_dense():
    m = _Stub(0.8)
    for W, B in ((1, 4), (2, 3), (4, 2), (3, 5)):
        torch.manual_seed(W * 10 + B)
        z = torch.randn(2 * W * B, 16, requires_grad=True)

        def dense():
            return m.compute_uniformity_loss(z, 2.0)

        def sharded():
            lse = torch.cat([m.uniformity_row_logsumexp(z, ri, 2.0)
                             for ri in _rank_rows(W, B)])
            return m.uniformity_from_row_logsumexp(lse, 2.0)

        assert abs(dense().item() - sharded().item()) < 1e-5
        g_d, = _grads(dense, z)
        g_s, = _grads(sharded, z)
        assert (g_d - g_s).abs().max().item() < 1e-5, f"W={W} B={B}: gradient mismatch"


def test_uniformity_orders_collapsed_below_spread():
    torch.manual_seed(0)
    m = _Stub(0.8)
    spread = torch.randn(64, 256)
    collapsed = torch.randn(1, 256) + 0.05 * torch.randn(64, 256)
    l_spread = m.compute_uniformity_loss(spread).item()
    assert abs(l_spread - (-4.0)) < 0.1          # near the -2t floor in high dim
    assert m.compute_uniformity_loss(collapsed).item() > l_spread + 3.0


def test_uniformity_wrapper_helper_single_process():
    from argparse import Namespace
    from biom3.Stage1.PL_wrapper import _uniformity_losses, _gather_with_grad
    torch.manual_seed(0)
    m = _Stub(0.8)
    B = 4
    z_p, z_t = torch.randn(2 * B, 16), torch.randn(2 * B, 16)
    args = Namespace(uniformity_t=2.0, uniformity_on='both')
    dense = _uniformity_losses(m, args, z_p, z_t, B, 'dense', _gather_with_grad)
    sharded = _uniformity_losses(m, args, z_p, z_t, B, 'sharded', _gather_with_grad)
    assert set(dense) == {'protein', 'text'}
    for k in dense:
        assert abs(dense[k].item() - sharded[k].item()) < 1e-5
    args.uniformity_on = 'protein'
    assert set(_uniformity_losses(m, args, z_p, z_t, B, 'dense', _gather_with_grad)) == {'protein'}


def test_uniformity_skipped_for_a_single_pair():
    """M = 2 leaves no pair after the self/homolog exclusion, so L_unif is NaN;
    the term must be skipped, not added to the loss."""
    from argparse import Namespace
    from types import SimpleNamespace
    from biom3.Stage1.PL_wrapper import pfam_PL_PEN_CL
    host = SimpleNamespace(model=_Stub(0.8), script_args=Namespace(
        uniformity_weight=0.5, uniformity_t=2.0, uniformity_on='both'))
    loss = torch.tensor(1.0)
    for impl in ('dense', 'sharded'):
        out, logged = pfam_PL_PEN_CL._add_uniformity(
            host, loss, torch.randn(2, 16), torch.randn(2, 16), 1, impl, 'train')
        assert out is loss and logged == {}, impl
        out, logged = pfam_PL_PEN_CL._add_uniformity(
            host, loss, torch.randn(4, 16), torch.randn(4, 16), 2, impl, 'train')
        assert torch.isfinite(out) and len(logged) == 2, impl


if __name__ == "__main__":
    for W, B in ((1, 4), (2, 3), (4, 2), (8, 2), (3, 5)):
        pr_i, pr_a, gl_i, gl_a = _run(W, B)
        ok = max(pr_i, pr_a, gl_i, gl_a) < TOL
        print(f"  W={W:2} B={B}  M={2*W*B:3}  per-rank inter {pr_i:.2e} intra {pr_a:.2e} "
              f" global inter {gl_i:.2e} intra {gl_a:.2e}   {'PASS' if ok else 'FAIL'}")
    print()
    for W, B in ((2, 3), (4, 2), (3, 5)):
        d = _run_grad(W, B)
        print(f"  W={W:2} B={B}  max |grad_dense - grad_sharded| = {d:.3e}   "
              f"{'PASS' if d < 1e-4 else 'FAIL'}")
