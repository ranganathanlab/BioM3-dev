"""The same-sequence / same-family mask must not change the objective's shape.

Three things can go wrong when the homolog index rule is generalised to key
equality, and each is checked here:

1. A key always equals itself, so the mask hits the row's own positive unless it
   is cleared afterwards -- the diagonal for L_GC, the homolog column for L_PFC.
   Masking the positive makes the numerator -9e15 and the loss meaningless,
   and it would NOT show up as a crash.
2. The sharded and dense paths build the mask by different routes ([R, M] and
   [M, R] slices vs a full [M, M]); if they disagree, switching
   contrastive_impl silently changes the objective.
3. With both flags off the losses must be bit-comparable to the pre-change code,
   or Run 1 stops being reproducible.

Runs on CPU with tiny tensors -- no distributed, no XPU.
"""
import torch
import torch.nn as nn

from biom3.Stage1.model import pfam_PEN_CL, NO_KEY

TOL = 2e-5
GRAD_TOL = 1e-4


class _Stub(pfam_PEN_CL):
    def __init__(self, temperature):
        nn.Module.__init__(self)
        self.temperature = temperature


def _key_hits(keys, M):
    """[M, M] key-equality mask, written independently of model.py."""
    out = torch.zeros((M, M), dtype=torch.bool)
    if keys is None:
        return out
    for c in range(keys.shape[1]):
        k = keys[:, c]
        out |= (k.unsqueeze(1) == k.unsqueeze(0)) & (k.unsqueeze(1) != NO_KEY)
    return out


def _dense_inter_ref(m, z_p, z_t, N, keys):
    """compute_inter_loss re-implemented from the formula, per row."""
    M = 2 * N
    idx = torch.arange(M)
    mask = torch.zeros((M, M), dtype=torch.bool)
    mask[N:, :N] = torch.eye(N, dtype=torch.bool)
    mask[:N, N:] = torch.eye(N, dtype=torch.bool)
    mask |= _key_hits(keys, M)
    mask[idx, idx] = False                       # L_GC's positive is the diagonal
    ml = m.set_inf((z_t @ z_p.T) / m.temperature, mask)
    mp = m.set_inf(z_p @ z_p.T, mask)
    mt = m.set_inf(z_t @ z_t.T, mask)
    targets = torch.softmax((mp + mt) / (2 * m.temperature), dim=-1)
    text = (-targets * torch.log_softmax(ml, dim=-1)).sum(1)
    prot = (-targets.T * torch.log_softmax(ml.T, dim=-1)).sum(1)
    return (prot + text) / 2


def _dense_intra_ref(m, z_p, keys):
    M = z_p.shape[0]
    cs = (z_p @ z_p.T) / m.temperature
    eye = torch.eye(M, dtype=torch.bool)
    pos_mask = eye.roll(shifts=M // 2, dims=0)
    drop = (eye | _key_hits(keys, M)) & ~pos_mask   # L_PFC's positive is the homolog
    cs = m.set_inf(cs, drop)
    return -cs[pos_mask] + torch.logsumexp(cs, dim=-1)


def _make_keys(M, n_fam, dup_pairs, seed=0):
    """[M, 2] keys: column 0 sequence identity, column 1 family.

    Pairs (i, i+N) share a family, as the data pipeline guarantees; a few
    unrelated rows are given an identical sequence key; some rows get NO_KEY.
    """
    g = torch.Generator().manual_seed(seed)
    N = M // 2
    fam = torch.randint(1, n_fam + 1, (N,), generator=g, dtype=torch.long)
    fam = torch.cat([fam, fam])                       # the pair shares its family
    seq = torch.arange(1, M + 1, dtype=torch.long) * 1000
    for a, b in dup_pairs:
        seq[b] = seq[a]
    seq[0] = NO_KEY                                   # a 'nan'-label row
    fam[N] = NO_KEY
    return torch.stack([seq, fam], dim=-1)


def _rows(W, B, N):
    return [torch.cat([torch.arange(r * B, (r + 1) * B),
                       torch.arange(N + r * B, N + (r + 1) * B)]) for r in range(W)]


def _run(W, B, D=16, tau=0.8, seed=0, keys=True):
    torch.manual_seed(seed)
    N, M = W * B, 2 * W * B
    z_p, z_t = torch.randn(M, D), torch.randn(M, D)
    k = _make_keys(M, max(2, N // 2), [(1, 2), (3, N + 1)], seed) if keys else None
    m = _Stub(tau)

    d_inter = _dense_inter_ref(m, z_p, z_t, N, k)
    d_intra = _dense_intra_ref(m, z_p, k)

    # model's own dense path must match the reference
    mi = m.compute_inter_loss(z_p.clone(), z_t.clone(), N, keys=k)[0]
    ma = m.compute_intra_loss(z_p.clone(), N, keys=k)[0]
    e_dense = max(abs(mi.item() - d_inter.mean().item()),
                  abs(ma.item() - d_intra.mean().item()))

    logZ = torch.empty(M)
    for ri in _rows(W, B, N):
        logZ[ri] = m.inter_row_logsumexp(z_p, z_t, ri, k)
    e_shard = 0.0
    for ri in _rows(W, B, N):
        si = m.compute_inter_loss_sharded(z_p, z_t, N, ri, logZ, k)[0]
        sa = m.compute_intra_loss_sharded(z_p, N, ri, k)[0]
        e_shard = max(e_shard, abs(si.item() - d_inter[ri].mean().item()),
                      abs(sa.item() - d_intra[ri].mean().item()))
    return e_dense, e_shard


def _run_grad(W, B, D=16, tau=0.8, seed=0):
    torch.manual_seed(seed)
    N, M = W * B, 2 * W * B
    z_p = torch.randn(M, D, requires_grad=True)
    z_t = torch.randn(M, D, requires_grad=True)
    k = _make_keys(M, max(2, N // 2), [(1, 2), (3, N + 1)], seed)
    m = _Stub(tau)

    def grads(fn):
        for l in (z_p, z_t):
            l.grad = None
        fn().backward()
        return z_p.grad.clone(), z_t.grad.clone()

    gd = grads(lambda: _dense_inter_ref(m, z_p, z_t, N, k).mean())

    def sharded_total():
        logZ = torch.cat([m.inter_row_logsumexp(z_p, z_t, ri, k)
                          for ri in _rows(W, B, N)])
        order = torch.cat(_rows(W, B, N))
        full = torch.empty(M, dtype=logZ.dtype)
        full = full.index_copy(0, order, logZ)
        return sum(m.compute_inter_loss_sharded(z_p, z_t, N, ri, full, k)[0]
                   for ri in _rows(W, B, N)) / W

    gs = grads(sharded_total)
    return max((a - b).abs().max().item() for a, b in zip(gd, gs))


def _invariants():
    """The positive column is never masked, and NO_KEY matches nothing."""
    m = _Stub(0.8)
    W, B = 3, 4
    N, M = W * B, 2 * W * B
    k = _make_keys(M, 2, [(1, 2)], 0)          # only 2 families -> heavy collisions
    bad = []
    for ri in _rows(W, B, N):
        ar = torch.arange(ri.numel())
        mr = m._homolog_mask_rows(ri, M, k)
        if mr[ar, ri].any():
            bad.append("inter rows: diagonal masked")
        mc = m._homolog_mask_cols(ri, M, k)
        if mc[ri, ar].any():
            bad.append("inter cols: diagonal masked")
        # L_PFC: the homolog column must survive INSIDE the real function.
        # compute_intra_loss_sharded returns its masked [R, M]; the positive
        # entry must be a live similarity, not the -9e15 sentinel.
        pos = (ri - M // 2) % M
        _, cs = m.compute_intra_loss_sharded(torch.randn(M, 8), M // 2, ri, k)
        if (cs[ar, pos] < -1e14).any():
            bad.append("intra: positive was masked")
        if not (cs < -1e14).any():
            bad.append("intra: nothing masked at all")
        # with only 2 families the mask must actually be doing something
        if not m._key_equality_mask(k, ri, M).any():
            bad.append("key mask is empty")
    # NO_KEY rows (row 0 sequence, row N family) must not match each other
    km = m._key_equality_mask(k, torch.tensor([0]), M)
    if km[0, 0] and k[0, 0].item() == NO_KEY:
        pass  # self-hit on column 1 (family) is fine; cleared by the caller
    only_no_key = _Stub(0.8)._key_equality_mask(
        torch.zeros((M, 1), dtype=torch.long), torch.arange(M), M)
    if only_no_key.any():
        bad.append("NO_KEY matched NO_KEY")
    return bad


def test_masked_losses_match_reference():
    for W, B in ((1, 4), (2, 3), (4, 2), (3, 5)):
        e_d, e_s = _run(W, B)
        assert e_d < TOL, f"W={W} B={B}: dense vs reference {e_d}"
        assert e_s < TOL, f"W={W} B={B}: sharded vs reference {e_s}"


def test_no_keys_reproduces_unmasked_objective():
    for W, B in ((2, 3), (4, 2)):
        e_d, e_s = _run(W, B, keys=False)
        assert e_d < TOL, f"W={W} B={B}: dense vs reference {e_d}"
        assert e_s < TOL, f"W={W} B={B}: sharded vs reference {e_s}"


def test_masked_sharded_grads_match_dense():
    for W, B in ((2, 3), (3, 5)):
        d = _run_grad(W, B)
        assert d < GRAD_TOL, f"W={W} B={B}: gradient mismatch {d}"


def test_positive_never_masked_and_no_key_inert():
    assert _invariants() == []


if __name__ == "__main__":
    print("keys ON  -- model dense vs reference, and sharded vs dense")
    for W, B in ((1, 4), (2, 3), (4, 2), (3, 5)):
        e_d, e_s = _run(W, B)
        ok = max(e_d, e_s) < TOL
        print(f"  W={W:2} B={B}  M={2*W*B:3}  dense {e_d:.2e}  sharded {e_s:.2e}"
              f"   {'PASS' if ok else 'FAIL'}")

    print("\nkeys OFF -- must reproduce the unmasked objective")
    for W, B in ((2, 3), (4, 2)):
        e_d, e_s = _run(W, B, keys=False)
        ok = max(e_d, e_s) < TOL
        print(f"  W={W:2} B={B}  dense {e_d:.2e}  sharded {e_s:.2e}"
              f"   {'PASS' if ok else 'FAIL'}")

    print("\ngradients (dense vs sharded, keys ON)")
    for W, B in ((2, 3), (3, 5)):
        d = _run_grad(W, B)
        print(f"  W={W:2} B={B}  max |dgrad| = {d:.3e}   "
              f"{'PASS' if d < GRAD_TOL else 'FAIL'}")

    print("\ninvariants")
    problems = _invariants()
    for p in problems:
        print(f"  FAIL {p}")
    if not problems:
        print("  PASS positive never masked; NO_KEY inert; mask non-empty")
