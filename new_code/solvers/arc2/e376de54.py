import numpy as np

from arc_utils import get_bg as _bg_color

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag \, diag /


def _pick_main_dir(g: np.ndarray, bg: int) -> tuple[int, int] | None:
    best = None
    best_cnt = -1
    h, w = g.shape
    for dr, dc in _DIR_CANDIDATES:
        cnt = 0
        for r in range(h):
            for c in range(w):
                v = int(g[r, c])
                if v == bg:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == v:
                    cnt += 1
        if cnt > best_cnt:
            best_cnt = cnt
            best = (dr, dc)
    if best_cnt <= 0:
        return None
    return best


def _dot(p: tuple[int, int], d: tuple[int, int]) -> int:
    return p[0] * d[0] + p[1] * d[1]


def _perp_dot(p: tuple[int, int], d: tuple[int, int]) -> int:
    # perp to (dr,dc) can be (dc,-dr); dot(p, perp) = r*dc - c*dr
    return p[0] * d[1] - p[1] * d[0]


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Align parallel colored line segments so they all share the same start and end positions as a reference line.

    Input structure:
    - bg = most common color.
    - Several parallel line segments (diagonal, horizontal, or vertical), each a different color, running in the same direction but starting/ending at different positions.

    Rule:
    - Detect the dominant direction shared by all segments.
    - Pick the median segment (by perpendicular offset) as the reference.
    - Shift every segment along the shared direction so its start aligns with the reference's start, then trim or extend it to match the reference's length.
    - Grid size stays the same; isolated single pixels are untouched.

    Key insight: The reference is the median line, not the longest; all segments are snapped to its extent regardless of their original length.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    d = _pick_main_dir(g, bg)
    if d is None:
        return g.copy()
    dr, dc = d
    delta_t = dr * dr + dc * dc

    segments = []

    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    for col in colors:
        mask = g == col
        for r in range(h):
            for c in range(w):
                if not mask[r, c]:
                    continue
                pr, pc = r - dr, c - dc
                if 0 <= pr < h and 0 <= pc < w and mask[pr, pc]:
                    continue  # not a segment start
                cells = []
                rr, cc = r, c
                while 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
                    cells.append((rr, cc))
                    rr += dr
                    cc += dc
                if len(cells) < 2:
                    continue
                start = cells[0]
                end = cells[-1]
                segments.append(
                    {
                        "color": col,
                        "cells": cells,
                        "start": start,
                        "end": end,
                        "t_start": _dot(start, d),
                        "t_end": _dot(end, d),
                        "s": _perp_dot(start, d),
                        "len": len(cells),
                    }
                )

    if not segments:
        return g.copy()

    # Pick the reference segment as the one whose line coordinate (perpendicular dot) is the median.
    s_vals = sorted(s["s"] for s in segments)
    s_ref = s_vals[len(s_vals) // 2]
    ref_cands = [s for s in segments if s["s"] == s_ref]
    ref = sorted(ref_cands, key=lambda x: (-x["len"], x["color"], x["t_start"], x["t_end"]))[0]
    ref_t0, ref_t1 = ref["t_start"], ref["t_end"]
    if ref_t1 < ref_t0:
        ref_t0, ref_t1 = ref_t1, ref_t0
    if (ref_t1 - ref_t0) % delta_t != 0:
        return g.copy()
    target_len = (ref_t1 - ref_t0) // delta_t + 1

    out = g.copy()

    # Clear only the original segment pixels (leave isolated pixels unchanged).
    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg

    # Deterministic paint order.
    for s in sorted(segments, key=lambda x: (x["color"], x["s"], x["t_start"], x["t_end"])):
        col = s["color"]
        start = s["start"]
        # Shift the segment along the main direction so its start aligns to ref_t0,
        # then draw exactly target_len cells.
        dt = ref_t0 - s["t_start"]
        if dt % delta_t != 0:
            continue
        k = dt // delta_t
        r0 = start[0] + k * dr
        c0 = start[1] + k * dc
        rr, cc = r0, c0
        for _ in range(target_len):
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = col
            rr += dr
            cc += dc

    return out
