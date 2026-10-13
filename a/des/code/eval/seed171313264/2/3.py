import random

import numpy as np


BG = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _pipe_groups(g: np.ndarray) -> tuple[list[int], list[int], list[int], list[int]]:
    pipeish = g != BG
    h, w = g.shape
    row_scores = [int(pipeish[r].sum()) for r in range(h)]
    col_scores = [int(pipeish[:, c].sum()) for c in range(w)]
    hrows = [r for r, score in enumerate(row_scores) if score >= max(5, w // 2)]
    vcols = [c for c, score in enumerate(col_scores) if score >= max(5, h // 2)]
    assert hrows and vcols
    hcols = [c for c in range(w) if any(pipeish[r, c] for r in hrows)]
    vrows = [r for r in range(h) if any(pipeish[r, c] for c in vcols)]
    return hrows, hcols, vrows, vcols


def _seed_for(g: np.ndarray, hrows: list[int], vcols: list[int], orientation: str) -> tuple[int, int, int]:
    for r, c in np.argwhere((g != BG) & (g != GREY)):
        rr, cc = int(r), int(c)
        color = int(g[rr, cc])
        if orientation == "h" and rr in hrows and cc not in vcols:
            return rr, cc, color
        if orientation == "v" and cc in vcols and rr not in hrows:
            return rr, cc, color
    raise AssertionError(f"missing {orientation} seed")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a crossing pipe network from two dead-end seeds. The wider pipe owns the
    crossing and continues beyond it; equal-width pipes both stop before the
    crossing, leaving the crossing grey.
    """
    g = np.asarray(grid, dtype=int)
    hrows, hcols, vrows, vcols = _pipe_groups(g)
    hr0, hr1 = min(hrows), max(hrows)
    vc0, vc1 = min(vcols), max(vcols)
    _sr_h, sc_h, color_h = _seed_for(g, hrows, vcols, "h")
    sr_v, _sc_v, color_v = _seed_for(g, hrows, vcols, "v")

    out = g.copy()
    pipeish = g != BG
    out[pipeish] = GREY

    h_from_left = sc_h < vc0
    v_from_top = sr_v < hr0
    h_width = len(hrows)
    v_width = len(vcols)

    def h_before(c: int) -> bool:
        return c < vc0 if h_from_left else c > vc1

    def v_before(r: int) -> bool:
        return r < hr0 if v_from_top else r > hr1

    for r in hrows:
        for c in hcols:
            if h_before(c):
                out[r, c] = color_h
    for r in vrows:
        for c in vcols:
            if v_before(r):
                out[r, c] = color_v

    if h_width > v_width:
        for r in hrows:
            for c in hcols:
                out[r, c] = color_h
    elif v_width > h_width:
        for r in vrows:
            for c in vcols:
                out[r, c] = color_v

    return out


def generate_input() -> np.ndarray:
    h = random.randint(12, 18)
    w = random.randint(12, 18)
    h_width = random.choice((1, 2))
    v_width = random.choice((1, 2))
    hr0 = random.randint(4, h - 6)
    vc0 = random.randint(4, w - 6)
    hrows = list(range(hr0, hr0 + h_width))
    vcols = list(range(vc0, vc0 + v_width))
    hcols = list(range(1, w - 1))
    vrows = list(range(1, h - 1))

    g = np.full((h, w), BG, dtype=int)
    for r in hrows:
        for c in hcols:
            g[r, c] = GREY
    for r in vrows:
        for c in vcols:
            g[r, c] = GREY

    color_h, color_v = random.sample(SEED_COLORS, 2)
    g[random.choice(hrows), 1] = color_h
    g[1, random.choice(vcols)] = color_v
    return g
