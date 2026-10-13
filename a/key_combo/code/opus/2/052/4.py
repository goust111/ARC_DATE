from __future__ import annotations

from collections import deque
from math import gcd
from typing import Iterable

import numpy as np

BLACK = 0
RED = 2
TEAL = 8


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    vis = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            q = deque([(r, c)])
            vis[r, c] = True
            comp = [(r, c)]
            while q:
                x, y = q.pop()
                for nx, ny in _neighbors4(x, y):
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not vis[nx, ny]:
                        vis[nx, ny] = True
                        q.append((nx, ny))
                        comp.append((nx, ny))
            comps.append(comp)
    return comps


def _crop_mask(mask: np.ndarray, coords: list[tuple[int, int]]) -> tuple[np.ndarray, int, int]:
    rs, cs = zip(*coords)
    r0, c0 = min(rs), min(cs)
    r1, c1 = max(rs), max(cs)
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        local[int(r - r0), int(c - c0)] = True
    return local, int(r0), int(c0)


def _infer_lattice(values: list[int], upper_bound: int) -> list[int]:
    vals = sorted(set(values))
    if not vals:
        return []
    if len(vals) == 1:
        # pick two points only; one gap impossible to infer well
        a = vals[0]
        if a + 1 < upper_bound:
            return [a, a + 1]
        return [a]
    diffs = [b - a for a, b in zip(vals[:-1], vals[1:]) if b > a]
    if not diffs:
        return vals
    step = diffs[0]
    for d in diffs[1:]:
        step = gcd(step, d)
    if step <= 0:
        return vals
    start = vals[0]
    end = vals[-1]
    return list(range(start, end + 1, step))


def _paste(out: np.ndarray, frame_r: int, frame_c: int, tr: int, tc: int, mask: np.ndarray, color: int) -> None:
    h, w = out.shape
    mh, mw = mask.shape
    for r in range(mh):
        rr = tr + r
        if rr < 0 or rr >= h:
            continue
        for c in range(mw):
            cc = tc + c
            if cc < 0 or cc >= w:
                continue
            if mask[r, c] and rr != frame_r and cc != frame_c:
                out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True).astype(int)
    red_rows = np.where((out == RED).sum(axis=1) == (out.shape[1]))[0]
    red_cols = np.where((out == RED).sum(axis=0) == (out.shape[0]))[0]
    if red_rows.size == 0 or red_cols.size == 0:
        return out
    frame_row = int(red_rows[0])
    frame_col = int(red_cols[0])

    # TL sprites (exclude frame)
    r_idx, c_idx = np.where((out != BLACK) & (out != RED))
    tl_mask = (r_idx < frame_row) & (c_idx < frame_col)
    r_idx = r_idx[tl_mask]
    c_idx = c_idx[tl_mask]
    if r_idx.size == 0:
        out[frame_row, :] = RED
        out[:, frame_col] = RED
        return out

    tl_mask2 = np.zeros(out.shape, dtype=bool)
    for r, c in zip(r_idx, c_idx):
        tl_mask2[int(r), int(c)] = True
    comps = _components(tl_mask2)
    if not comps:
        out[frame_row, :] = RED
        out[:, frame_col] = RED
        return out

    # infer sprite mask by largest area (fallback for generated inputs)
    areas = []
    local_info = []
    for comp in comps:
        local, tr, tc = _crop_mask(tl_mask2, comp)
        color = int(out[comp[0][0], comp[0][1]])
        local_info.append((local, tr, tc, color))
        areas.append(local.sum())

    local_info.sort(key=lambda x: int(x[0].sum()), reverse=True)
    template, _, _, _ = local_info[0]
    tl_positions: list[tuple[int, int]] = []
    tl_color: dict[tuple[int, int], int] = {}
    for _, tr, tc, color in local_info:
        tl_positions.append((tr, tc))
        tl_color[(tr, tc)] = color

    rows = [r for r, _ in tl_positions]
    cols = [c for _, c in tl_positions]
    full_rows = _infer_lattice(rows, frame_row)
    full_cols = _infer_lattice(cols, frame_col)
    if not full_rows:
        full_rows = sorted(set(rows))
    if not full_cols:
        full_cols = sorted(set(cols))

    # complete TL with teal
    all_tl = [(r, c) for r in full_rows for c in full_cols]
    for r0, c0 in all_tl:
        color = tl_color.get((r0, c0), TEAL)
        _paste(out, frame_row, frame_col, r0, c0, template, color)

    # mirror to all quadrants (mirrors use colors via opposite-axis original TL lookup)
    for r0, c0 in all_tl:
        # TR from TL mirrored horizontally
        tr, tc = r0, 2 * frame_col - c0
        color_tr = tl_color.get((2 * frame_row - r0, c0), TEAL)
        _paste(out, frame_row, frame_col, tr, tc, template, color_tr)

        # BL mirrored vertically
        tr, tc = 2 * frame_row - r0, c0
        color_bl = tl_color.get((r0, 2 * frame_col - c0), TEAL)
        _paste(out, frame_row, frame_col, tr, tc, template, color_bl)

        # BR mirrored by both axes
        tr, tc = 2 * frame_row - r0, 2 * frame_col - c0
        color_br = tl_color.get((2 * frame_row - r0, 2 * frame_col - c0), TEAL)
        _paste(out, frame_row, frame_col, tr, tc, template, color_br)

    out[frame_row, :] = RED
    out[:, frame_col] = RED
    return out


def _random_template(rng: np.random.Generator) -> np.ndarray:
    templates = [
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=bool),
        np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1], [1, 1]], dtype=bool),
    ]
    return templates[int(rng.integers(0, len(templates)))]


def _place_sprite(out: np.ndarray, tr: int, tc: int, m: np.ndarray, color: int, frame_r: int, frame_c: int) -> None:
    h, w = out.shape
    mh, mw = m.shape
    for r in range(mh):
        rr = tr + r
        if rr < 0 or rr >= h:
            continue
        for c in range(mw):
            cc = tc + c
            if cc < 0 or cc >= w:
                continue
            if m[r, c] and rr != frame_r and cc != frame_c:
                out[rr, cc] = color


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    size = int(rng.choice([17, 19, 21]))
    mid = size // 2
    out = np.full((size, size), BLACK, dtype=int)
    out[mid, :] = RED
    out[:, mid] = RED

    template = _random_template(rng)
    sprite_colors = [1, 3, 4]
    n_rows = int(rng.integers(2, 5))
    n_cols = int(rng.integers(2, 5))
    row_step = int(rng.integers(2, 5))
    col_step = int(rng.integers(2, 5))

    max_row0 = max(1, mid - 2 - (n_rows - 1) * row_step - template.shape[0])
    max_col0 = max(1, mid - 2 - (n_cols - 1) * col_step - template.shape[1])
    row0 = int(rng.integers(1, max_row0 + 1))
    col0 = int(rng.integers(1, max_col0 + 1))

    all_rows = [row0 + i * row_step for i in range(n_rows)]
    all_cols = [col0 + j * col_step for j in range(n_cols)]
    full_positions = [(r, c) for r in all_rows for c in all_cols if c < mid and r < mid]

    keep = np.ones(len(full_positions), dtype=bool)

    # keep at least one full row and one full col
    if full_positions:
        keep = np.zeros(len(full_positions), dtype=bool)
        row_idx = int(rng.integers(0, n_rows))
        col_idx = int(rng.integers(0, n_cols))
        for k, (r, c) in enumerate(full_positions):
            if r == all_rows[row_idx] or c == all_cols[col_idx]:
                keep[k] = True
        # optionally drop some others
        n_drop = int(len(full_positions) * rng.uniform(0.2, 0.45))
        order = np.array(list(range(len(full_positions))))
        rng.shuffle(order)
        for k in order[:n_drop]:
            keep[k] = False
        # ensure required row/col stay present
        for k, (r, c) in enumerate(full_positions):
            if r == all_rows[row_idx] or c == all_cols[col_idx]:
                keep[k] = True

    # paint TL
    color_counts = {1: 0, 3: 0, 4: 0}
    for k, (r, c) in enumerate(full_positions):
        if not keep[k]:
            continue
        color = int(rng.choice(sprite_colors))
        color_counts[color] += 1
        _place_sprite(out, r, c, template, color, mid, mid)

    for color in sprite_colors:
        if color_counts[color] < 2:
            for k, (r, c) in enumerate(full_positions):
                if keep[k]:
                    continue
                _place_sprite(out, r, c, template, color, mid, mid)
                color_counts[color] += 1
                keep[k] = True
                if color_counts[color] >= 2:
                    break

    return out

