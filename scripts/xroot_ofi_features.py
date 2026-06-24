#!/usr/bin/env python3
"""Stage 1 of the cross-root integrated-OFI lead-lag study.

Per (root, day): compute the standard Cont 10-level OFI per LOB update (matching
run_mm_full.py::MicroSkewQuoter._update_ofi), accumulate into fixed clock bins, emit one row
per bin with the 10-level summed OFI vector + causal L1 mid at bin close. The 10 levels are kept
(not yet PCA-integrated) so PCA + standardization fit on the IS window only downstream.

RAM-SAFE + FAST: weekday files are 5-7 M rows. We never materialise the whole file. We stream
by Arrow row-group/batch, flatten list<struct<px,sz>> with pyarrow compute (C-speed, no Python
dict loop), and compute OFI vectorised in numpy over each (chunk x 10) matrix. Peak RAM per file
~tens of MB. A hard RLIMIT_AS ceiling (6 GB) is set as a backstop.

Output: runs/xroot_ofi/feat/<root>_<date>_<binms>.parquet
        cols: bin_ts_ns, ofi0..ofi9, mid_close, n_upd
"""
import sys, os, time, resource
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc

# hard memory backstop: 6 GB address space
try:
    resource.setrlimit(resource.RLIMIT_AS, (6 * 1024**3, 6 * 1024**3))
except Exception:
    pass

OUT = os.environ.get("XROOT_OFI_FEAT", "runs/xroot_ofi/feat")
MANIFEST = os.environ.get("FUT_UNIVERSE_MANIFEST", "data/fut_universe/manifest_all.csv")
L = 10
BATCH = 300_000


def _pad(arr):
    """Arrow ListArray<struct<px,sz>> -> (px_mat[N,10], sz_mat[N,10]); missing -> nan px / 0 sz."""
    n = len(arr)
    lengths = pc.list_value_length(arr).to_numpy(zero_copy_only=False).astype(np.int64)
    flat = pc.list_flatten(arr)
    px = flat.field('px').to_numpy(zero_copy_only=False)
    sz = flat.field('sz').to_numpy(zero_copy_only=False)
    total = px.shape[0]
    if total == 0:
        return np.full((n, L), np.nan), np.zeros((n, L))
    starts = np.zeros(n, dtype=np.int64)
    np.cumsum(lengths[:-1], out=starts[1:])
    elem_row = np.repeat(np.arange(n), lengths)
    elem_pos = np.arange(total) - starts[elem_row]
    px_mat = np.full((n, L), np.nan)
    sz_mat = np.zeros((n, L))
    m = elem_pos < L
    px_mat[elem_row[m], elem_pos[m]] = px[m]
    sz_mat[elem_row[m], elem_pos[m]] = sz[m]
    return px_mat, sz_mat


def _cont_ofi(px_b, sz_b, px_a, sz_a):
    """Vectorised Cont 10-level OFI. Row i compared to row i-1 (by rank). Returns (M,10).
    Row 0 has no predecessor -> 0. nan price = level absent."""
    def shift(a):
        out = np.empty_like(a); out[0] = np.nan; out[1:] = a[:-1]; return out
    ppb, pqb = shift(px_b), shift(sz_b)
    ppa, pqa = shift(px_a), shift(sz_a)
    # bid contribution
    with np.errstate(invalid='ignore'):
        bidc = np.where(np.isnan(ppb) & ~np.isnan(px_b), sz_b,
               np.where(np.isnan(px_b) & ~np.isnan(ppb), -pqb,
               np.where(px_b > ppb, sz_b,
               np.where(px_b == ppb, sz_b - pqb, -pqb))))
        bidc = np.where(np.isnan(px_b) & np.isnan(ppb), 0.0, bidc)
        # ask contribution (signs mirrored: aggressive sell lowers ask -> negative)
        askc = np.where(np.isnan(ppa) & ~np.isnan(px_a), -sz_a,
               np.where(np.isnan(px_a) & ~np.isnan(ppa), pqa,
               np.where(px_a < ppa, -sz_a,
               np.where(px_a == ppa, -(sz_a - pqa), pqa))))
        askc = np.where(np.isnan(px_a) & np.isnan(ppa), 0.0, askc)
    ofi = bidc + askc
    ofi[0] = 0.0
    return ofi


def extract(snap_path, bin_ms=10_000):
    bin_ns = bin_ms * 1_000_000
    pf = pq.ParquetFile(snap_path)
    carry = None  # (px_b_last, sz_b_last, px_a_last, sz_a_last)
    parts = []
    ofi_cols = [f'ofi{i}' for i in range(L)]
    for batch in pf.iter_batches(batch_size=BATCH, columns=['ts_ns', 'bids', 'asks']):
        ts = batch.column('ts_ns').to_numpy(zero_copy_only=False)
        px_b, sz_b = _pad(batch.column('bids'))
        px_a, sz_a = _pad(batch.column('asks'))
        n = ts.shape[0]
        if n == 0:
            continue
        if carry is not None:
            PXB = np.vstack([carry[0], px_b]); SZB = np.vstack([carry[1], sz_b])
            PXA = np.vstack([carry[2], px_a]); SZA = np.vstack([carry[3], sz_a])
            ofi = _cont_ofi(PXB, SZB, PXA, SZA)[1:]   # drop carry row -> n rows aligned to batch
        else:
            ofi = _cont_ofi(px_b, sz_b, px_a, sz_a)   # row0 -> 0
        with np.errstate(invalid='ignore'):
            mid = 0.5 * (px_b[:, 0] + px_a[:, 0])
            spr = px_a[:, 0] - px_b[:, 0]   # best-level spread (price units)
        binidx = ts // bin_ns
        df = pd.DataFrame(ofi, columns=ofi_cols)
        df['binidx'] = binidx
        df['mid'] = mid
        df['spr'] = spr
        g = df.groupby('binidx', sort=True)
        agg = g[ofi_cols].sum()
        agg['mid_close'] = g['mid'].last()
        agg['spr_close'] = g['spr'].last()
        agg['n_upd'] = g.size()
        parts.append(agg.reset_index())
        carry = (px_b[-1:], sz_b[-1:], px_a[-1:], sz_a[-1:])
    if not parts:
        return pd.DataFrame(columns=['bin_ts_ns'] + ofi_cols + ['mid_close', 'spr_close', 'n_upd'])
    allp = pd.concat(parts, ignore_index=True)
    gg = allp.groupby('binidx', sort=True)
    out = gg[ofi_cols].sum()
    out['mid_close'] = gg['mid_close'].last()
    out['spr_close'] = gg['spr_close'].last()
    out['n_upd'] = gg['n_upd'].sum()
    out = out.reset_index()
    out.insert(0, 'bin_ts_ns', (out['binidx'].astype(np.int64) * bin_ns))
    out = out.drop(columns=['binidx'])
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    m = pd.read_csv(MANIFEST, dtype={'date': str})
    bin_ms = int(os.environ.get('BIN_MS', '10000'))
    days = os.environ.get('DAYS', '20230709,20230710,20230711,20230712,20230713,20230719,20230730,20230716,20230723').split(',')
    only_root = os.environ.get('ONLY_ROOT')
    sel = m[m['date'].isin(days)]
    if only_root:
        sel = sel[sel['root'] == only_root]
    print(f"[xroot-ofi feat] {len(sel)} root-days, bin_ms={bin_ms}", flush=True)
    for _, r in sel.iterrows():
        outp = f"{OUT}/{r['root']}_{r['date']}_{bin_ms}.parquet"
        if os.path.exists(outp) and not os.environ.get('FORCE'):
            continue
        t0 = time.time()
        feat = extract(r['snap'], bin_ms)
        feat.to_parquet(outp, index=False)
        print(f"  {r['root']} {r['date']}: {len(feat)} bins in {time.time()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
