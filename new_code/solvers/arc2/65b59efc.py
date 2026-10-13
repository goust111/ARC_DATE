from __future__ import annotations
from collections import Counter
import numpy as np
from arc_utils import STRUCT8, label_mask

def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a "stamp map" — place recolored template tiles onto an output grid.

    Input structure:
    - Background (bg) and separator color (sep) forming thick divider lines.
    - Top band: K template tiles (N×N each), separated by sep columns.
    - Bottom band: K palette swatches giving the output color for each template.
    - Middle band: marker pixels whose color identifies which template to stamp.

    Rule:
    - Each marker pixel at (r,c) of template-color j stamps template[j] recolored by palette[j] at output cell (r mod N, c mod N).
    - Output is (N²)×(N²), tiling N×N stamps.

    Key insight: Marker color encodes template index; marker position encodes stamp location via modular arithmetic.
    """
    g0 = np.array(grid, dtype=int)
    H, W = g0.shape
    counts = Counter(map(int, g0.flatten()))
    bg = int(counts.most_common(1)[0][0])
    # separator = most common non-bg
    sep = int(max((cnt, col) for col, cnt in counts.items() if int(col) != int(bg))[1])

    # STEP 1) Detect thick separator columns/rows.
    sep_cols = [int(c) for c in range(W) if int((g0[:, int(c)] == int(sep)).sum()) >= H // 2]
    sep_rows = [int(r) for r in range(H) if int((g0[int(r), :] == int(sep)).sum()) >= W // 2]
    assert sep_rows
    rm_row = int(sep_rows[0])

    # Tile size N is the gap before the first separator column.
    assert sep_cols
    N = int(sep_cols[0])
    starts = [0] + [int(c) + 1 for c in sep_cols]  # left edge of each template column
    K = int(len(starts))

    # STEP 2) Read K templates from the top band.
    tcols: list[int] = []
    tmasks: list[np.ndarray] = []
    for st in starts:
        sub = g0[0:int(N), int(st) : int(st + N)]
        vals = [int(v) for v in sub.flatten() if int(v) not in (int(bg), int(sep))]
        assert vals
        col = int(Counter(vals).most_common(1)[0][0])
        tcols.append(int(col))
        tmasks.append((sub == int(col)).astype(int))

    # STEP 3) Palette swatches live in the bottom band (below the last thick sep row).
    assert sep_rows
    pal_r0 = int(sep_rows[-1] + 1)
    pal_region = g0[int(pal_r0) : int(H), :]
    m = (pal_region != int(bg)) & (pal_region != int(sep))
    lab, n = label_mask(m, connectivity=8)
    pals: list[tuple[int, int]] = []  # (min_col, color)
    for kk in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(kk))
        if coords.size == 0:
            continue
        _r0, c0 = coords.min(axis=0)
        vals = [int(pal_region[int(r), int(c)]) for r, c in coords]
        col = int(Counter(vals).most_common(1)[0][0])
        pals.append((int(c0), int(col)))
    pals.sort()
    pcols = [int(col) for _c, col in pals]

    # Ensure we have K palette colors.
    if len(pcols) < K:
        extra = [int(c) for c in sorted(set(map(int, pal_region.flatten())) - {int(bg), int(sep)})]
        for c in extra:
            if int(c) not in pcols:
                pcols.append(int(c))
            if len(pcols) >= K:
                break
        if len(pcols) < K:
            pcols = (pcols + pcols + pcols)[:K]
    else:
        pcols = pcols[:K]

    # STEP 4) Build a compressed grid for marker reading:
    # remove vertical separator cols and the first thick separator row; map sep->bg.
    g = g0
    if rm_row is not None:
        g = np.delete(g, [int(rm_row)], axis=0)
    if sep_cols:
        g = np.delete(g, [int(c) for c in sep_cols], axis=1)
    g = np.where(g == int(sep), int(bg), g)

    # Palette band start row in g (rows >= pal_r0 are palette definition, not markers).
    pal_r0_g = int(pal_r0)
    if rm_row is not None and int(rm_row) < int(pal_r0_g):
        pal_r0_g -= 1

    # STEP 5) Placements: for each template color, collect marker pixels in the middle band.
    place = [np.zeros((N, N), dtype=bool) for _ in range(K)]
    for j, col in enumerate(tcols):
        if int(col) == int(bg):
            continue
        coords = np.argwhere(g == int(col))
        for r, c in coords:
            r = int(r)
            c = int(c)
            if r < int(N):
                continue  # skip top template band
            if r >= int(pal_r0_g):
                continue  # skip palette-definition band
            place[int(j)][int(r % N), int(c % N)] = True

    # STEP 6) Render output: N x N stamp grid, each stamp is N x N.
    out = np.full((int(N * N), int(N * N)), int(bg), dtype=int)
    for j in range(K):
        tile = np.where(tmasks[int(j)] == 1, int(pcols[int(j)]), int(bg))
        for rr, cc in np.argwhere(place[int(j)]):
            rr = int(rr)
            cc = int(cc)
            out[int(rr * N) : int((rr + 1) * N), int(cc * N) : int((cc + 1) * N)] = tile

    return out

