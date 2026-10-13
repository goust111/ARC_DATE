from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
MARKER = 4  # yellow
OBJECT_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
SIDES = ("left", "right", "up", "down")

GRID_MIN = 14
GRID_MAX = 24
K_MIN = 2
K_MAX = 5
OBJECT_COUNT_MIN = 1
OBJECT_COUNT_MAX = 4
POLYOMINO_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 180
GEN_MAX_ATTEMPTS = 400
CLIP_PROB = 0.35
MAX_TEMPLATE_DIM = 5


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _trim(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _objects(grid: np.ndarray) -> list[tuple[int, int, int, int, int, list[tuple[int, int]]]]:
    comps = _components((grid != BG) & (grid != MARKER))
    out: list[tuple[int, int, int, int, int, list[tuple[int, int]]]] = []
    for comp in comps:
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1
        color = colors.pop()
        assert color not in {BG, MARKER}
        assert 2 <= len(comp) <= 6
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        out.append((min(rs), min(cs), max(rs), max(cs), color, comp))
    out.sort(key=lambda item: (item[0], item[1], item[4]))
    assert out
    return out


def _candidate_gap(
    grid: np.ndarray,
    marker: tuple[int, int],
    obj: tuple[int, int, int, int, int, list[tuple[int, int]]],
) -> int | None:
    mr, mc = marker
    r0, c0, r1, c1, _color, _cells = obj

    if r0 <= mr <= r1 and mc < c0 and np.all(grid[mr, mc + 1 : c0] == BG):
        return c0 - mc - 1
    if r0 <= mr <= r1 and mc > c1 and np.all(grid[mr, c1 + 1 : mc] == BG):
        return mc - c1 - 1
    if c0 <= mc <= c1 and mr < r0 and np.all(grid[mr + 1 : r0, mc] == BG):
        return r0 - mr - 1
    if c0 <= mc <= c1 and mr > r1 and np.all(grid[r1 + 1 : mr, mc] == BG):
        return mr - r1 - 1
    return None


def _template_from_cells(
    cells: list[tuple[int, int]], r0: int, c0: int, r1: int, c1: int
) -> np.ndarray:
    template = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        template[r - r0, c - c0] = True
    return template


def _paint_scaled(
    out: np.ndarray, template: np.ndarray, color: int, r0: int, c0: int, k: int
) -> None:
    scaled = np.repeat(np.repeat(template, k, axis=0), k, axis=1)
    h, w = out.shape
    rr1 = min(h, r0 + scaled.shape[0])
    cc1 = min(w, c0 + scaled.shape[1])
    patch = scaled[: rr1 - r0, : cc1 - c0]
    target = out[r0:rr1, c0:cc1]
    target[patch] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each small object by a scaled copy of its bbox template, with the
    scale encoded by the distance from the object's yellow marker to the bbox.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objs = _objects(g)
    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == MARKER)]
    assert len(markers) == len(objs)

    assigned: dict[int, int] = {}
    for marker in markers:
        matches: list[tuple[int, int]] = []
        for idx, obj in enumerate(objs):
            gap = _candidate_gap(g, marker, obj)
            if gap is not None:
                matches.append((gap, idx))
        assert matches
        best_gap = min(gap for gap, _ in matches)
        best = [idx for gap, idx in matches if gap == best_gap]
        assert len(best) == 1
        obj_idx = best[0]
        assert obj_idx not in assigned
        assigned[obj_idx] = best_gap + 2

    assert len(assigned) == len(objs)

    out = np.full(g.shape, BG, dtype=int)
    for idx, obj in enumerate(objs):
        r0, c0, r1, c1, color, cells = obj
        template = _template_from_cells(cells, r0, c0, r1, c1)
        _paint_scaled(out, template, color, r0, c0, assigned[idx])
    return out


def _random_polyomino() -> np.ndarray:
    for _ in range(POLYOMINO_ATTEMPTS):
        area = random.randint(2, 6)
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = {
                nbr
                for r, c in cells
                for nbr in _neighbors4(r, c)
                if nbr not in cells
            }
            cells.add(random.choice(tuple(frontier)))

        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        norm = {(r - min_r, c - min_c) for r, c in cells}
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        mask = np.zeros((h, w), dtype=bool)
        for r, c in norm:
            mask[r, c] = True

        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, k=random.randint(0, 3))
        mask = _trim(mask)
        if max(mask.shape) <= MAX_TEMPLATE_DIM:
            return mask
    raise RuntimeError("failed to generate a small connected template")


def _alignment_options(mask: np.ndarray, side: str, gap: int) -> list[int]:
    h, w = mask.shape
    if side == "left":
        if gap == 0:
            return [r for r in range(h) if not bool(mask[r, 0])]
        return list(range(h))
    if side == "right":
        if gap == 0:
            return [r for r in range(h) if not bool(mask[r, w - 1])]
        return list(range(h))
    if side == "up":
        if gap == 0:
            return [c for c in range(w) if not bool(mask[0, c])]
        return list(range(w))
    assert side == "down"
    if gap == 0:
        return [c for c in range(w) if not bool(mask[h - 1, c])]
    return list(range(w))


def _marker_coord(
    r0: int, c0: int, h: int, w: int, side: str, align: int, gap: int
) -> tuple[int, int]:
    if side == "left":
        return r0 + align, c0 - gap - 1
    if side == "right":
        return r0 + align, c0 + w + gap
    if side == "up":
        return r0 - gap - 1, c0 + align
    return r0 + h + gap, c0 + align


def _expanded_rect(
    r0: int, c0: int, r1: int, c1: int, h: int, w: int, pad: int = 1
) -> tuple[int, int, int, int]:
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _rect_free(mask: np.ndarray, rect: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = rect
    return not bool(mask[r0 : r1 + 1, c0 : c1 + 1].any())


def _fill_rect(mask: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = rect
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _choose_anchor(
    h_grid: int,
    w_grid: int,
    h_obj: int,
    w_obj: int,
    side: str,
    k: int,
    want_clip: bool,
) -> tuple[int, int] | None:
    gap = k - 2
    r_min, r_max = 0, h_grid - h_obj
    c_min, c_max = 0, w_grid - w_obj
    if side == "left":
        c_min = gap + 1
    elif side == "right":
        c_max = w_grid - w_obj - gap - 1
    elif side == "up":
        r_min = gap + 1
    else:
        r_max = h_grid - h_obj - gap - 1
    if r_min > r_max or c_min > c_max:
        return None

    row_safe_hi = min(r_max, h_grid - h_obj * k)
    col_safe_hi = min(c_max, w_grid - w_obj * k)
    row_clip_lo = max(r_min, h_grid - h_obj * k + 1)
    col_clip_lo = max(c_min, w_grid - w_obj * k + 1)

    safe_possible = row_safe_hi >= r_min and col_safe_hi >= c_min
    row_clip_possible = row_clip_lo <= r_max and col_safe_hi >= c_min
    col_clip_possible = col_clip_lo <= c_max and row_safe_hi >= r_min
    both_clip_possible = row_clip_lo <= r_max and col_clip_lo <= c_max

    if want_clip:
        modes: list[str] = []
        if row_clip_possible:
            modes.append("row")
        if col_clip_possible:
            modes.append("col")
        if both_clip_possible:
            modes.append("both")
        if not modes:
            return None
        mode = random.choice(modes)
        if mode in {"row", "both"}:
            r0 = random.randint(row_clip_lo, r_max)
        else:
            r0 = random.randint(r_min, row_safe_hi)
        if mode in {"col", "both"}:
            c0 = random.randint(col_clip_lo, c_max)
        else:
            c0 = random.randint(c_min, col_safe_hi)
        return r0, c0

    if not safe_possible:
        return None
    return random.randint(r_min, row_safe_hi), random.randint(c_min, col_safe_hi)


def _render_input(
    h: int, w: int, specs: list[tuple[int, np.ndarray, int, int, int, tuple[int, int]]]
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for color, mask, r0, c0, _k, marker in specs:
        rr, cc = np.argwhere(mask).T
        grid[r0 + rr, c0 + cc] = color
        grid[marker] = MARKER
    return grid


def _render_output(
    h: int, w: int, specs: list[tuple[int, np.ndarray, int, int, int, tuple[int, int]]]
) -> np.ndarray:
    out = np.full((h, w), BG, dtype=int)
    for color, mask, r0, c0, k, _marker in specs:
        _paint_scaled(out, mask, color, r0, c0, k)
    return out


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    n_objects = random.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX)
    colors = random.sample(OBJECT_COLORS, k=n_objects)

    input_reserved = np.zeros((h, w), dtype=bool)
    output_reserved = np.zeros((h, w), dtype=bool)
    specs: list[tuple[int, np.ndarray, int, int, int, tuple[int, int]]] = []

    for color in colors:
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            mask = _random_polyomino()
            h_obj, w_obj = mask.shape
            k = random.randint(K_MIN, K_MAX)
            gap = k - 2
            side = random.choice(SIDES)
            alignments = _alignment_options(mask, side, gap)
            if not alignments:
                continue
            align = random.choice(alignments)
            anchor = _choose_anchor(
                h,
                w,
                h_obj,
                w_obj,
                side,
                k,
                want_clip=random.random() < CLIP_PROB,
            )
            if anchor is None:
                continue
            r0, c0 = anchor
            marker = _marker_coord(r0, c0, h_obj, w_obj, side, align, gap)
            mr, mc = marker
            if not (0 <= mr < h and 0 <= mc < w):
                continue

            raw_input_rect = (
                min(r0, mr),
                min(c0, mc),
                max(r0 + h_obj - 1, mr),
                max(c0 + w_obj - 1, mc),
            )
            input_rect = _expanded_rect(*raw_input_rect, h, w, pad=1)
            clipped_r1 = min(h - 1, r0 + h_obj * k - 1)
            clipped_c1 = min(w - 1, c0 + w_obj * k - 1)
            output_rect = _expanded_rect(r0, c0, clipped_r1, clipped_c1, h, w, pad=1)

            if not _rect_free(input_reserved, input_rect):
                continue
            if not _rect_free(output_reserved, output_rect):
                continue

            _fill_rect(input_reserved, input_rect)
            _fill_rect(output_reserved, output_rect)
            specs.append((color, mask, r0, c0, k, marker))
            placed = True
            break
        if not placed:
            return None

    inp = _render_input(h, w, specs)
    expected = _render_output(h, w, specs)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
