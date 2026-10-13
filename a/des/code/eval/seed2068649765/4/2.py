from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
ALL_COLORS = tuple(range(1, 10))
GEN_RETRIES = 500
RING_PATHS = {
    "2x3": [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)],
    "3x3": [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)],
}


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                for nr, nc in neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _color_components(g: np.ndarray, blocked: np.ndarray | None = None) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    seen = np.array(blocked, dtype=bool) if blocked is not None else np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if bool(seen[r, c]) or int(g[r, c]) == BG:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                for nr, nc in neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not bool(seen[nr, nc]) and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _cycle_order(coords: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    cells = set(coords)
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in cells:
        adj = [n for n in neighbors4(*cell) if n in cells]
        if len(adj) != 2:
            return None
        neighbors[cell] = adj

    start = min(cells)
    order = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [n for n in neighbors[cur] if n != prev]
        if not nxts:
            return None
        nxt = nxts[0]
        if nxt == start:
            break
        order.append(nxt)
        prev, cur = cur, nxt
        if len(order) > len(cells):
            return None

    if len(order) != len(cells):
        return None
    return order


def _find_legend(g: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    h, w = g.shape
    candidates: list[tuple[tuple[int, int, int, int], list[tuple[int, int]], list[int]]] = []
    for comp in _components_mask(g != BG):
        colors = [int(g[r, c]) for r, c in comp]
        if len(comp) < 4 or len(comp) % 2 != 0:
            continue
        if len(set(colors)) != len(colors):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        if bh < 2 or bw < 2 or bh > 3 or bw > 3:
            continue
        order = _cycle_order(comp)
        if order is None:
            continue
        bottom_gap = h - 1 - r1
        left_gap = c0
        score = (left_gap + bottom_gap, left_gap, bottom_gap, len(comp))
        candidates.append((score, order, [int(g[r, c]) for r, c in order]))

    assert candidates, "legend cycle not found"
    _, order, colors = min(candidates, key=lambda item: item[0])
    n = len(colors)
    assert n % 2 == 0, "legend size must be even"

    legend_mask = np.zeros_like(g, dtype=bool)
    for r, c in order:
        legend_mask[r, c] = True

    complement = {colors[i]: colors[(i + n // 2) % n] for i in range(n)}
    return legend_mask, complement


def _local_enclosed_regions(border_mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = border_mask.shape
    regions: list[list[tuple[int, int]]] = []
    for comp in _components_mask(~border_mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        regions.append(comp)
    return regions


def _component_interior_regions(coords: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = _bbox(coords)
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        local[r - r0, c - c0] = True
    interiors = _local_enclosed_regions(local)
    return [[(r0 + r, c0 + c) for r, c in comp] for comp in interiors]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    legend_mask, complement = _find_legend(g)
    out = g.copy()

    jobs: list[tuple[list[tuple[int, int]], list[tuple[int, int]], int, int]] = []
    for comp in _color_components(g, blocked=legend_mask):
        border_color = int(g[comp[0]])
        if len(comp) < 4:
            continue
        interiors = _component_interior_regions(comp)
        if len(interiors) != 1:
            continue
        interior = interiors[0]
        non_bg = [(r, c) for r, c in interior if int(g[r, c]) != BG]
        if len(non_bg) != 1:
            continue
        seed_r, seed_c = non_bg[0]
        seed_color = int(g[seed_r, seed_c])
        if seed_color == border_color or seed_color not in complement:
            continue
        if any(int(g[r, c]) != BG for r, c in interior if (r, c) != (seed_r, seed_c)):
            continue
        jobs.append((comp, interior, seed_color, int(complement[seed_color])))

    for border, interior, seed_color, fill_color in jobs:
        for r, c in border:
            out[r, c] = seed_color
        for r, c in interior:
            out[r, c] = fill_color

    return out


def _shape_boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    boundary = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]):
                continue
            if r in (0, h - 1) or c in (0, w - 1):
                boundary[r, c] = True
                continue
            if any(not bool(mask[nr, nc]) for nr, nc in neighbors4(r, c)):
                boundary[r, c] = True
    return boundary


def _random_filled_shape(rng: random.Random) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    for _ in range(100):
        h = rng.randint(5, 8)
        w = rng.randint(5, 8)
        mask = np.ones((h, w), dtype=bool)

        for _ in range(rng.randint(0, 4)):
            side = rng.choice(["top", "bottom", "left", "right"])
            if side in ("top", "bottom"):
                if h <= 4 or w <= 4:
                    continue
                depth = rng.randint(1, min(2, h - 3))
                span = rng.randint(2, w - 2)
                c0 = rng.randint(1, w - span - 1)
                rows = slice(0, depth) if side == "top" else slice(h - depth, h)
                mask[rows, c0 : c0 + span] = False
            else:
                if h <= 4 or w <= 4:
                    continue
                depth = rng.randint(1, min(2, w - 3))
                span = rng.randint(2, h - 2)
                r0 = rng.randint(1, h - span - 1)
                cols = slice(0, depth) if side == "left" else slice(w - depth, w)
                mask[r0 : r0 + span, cols] = False

        if len(_components_mask(mask)) != 1:
            continue
        boundary = _shape_boundary(mask)
        if len(_components_mask(boundary)) != 1:
            continue
        interiors = _local_enclosed_regions(boundary)
        if len(interiors) != 1:
            continue
        interior = interiors[0]
        if len(interior) < 2:
            continue
        return boundary, [(int(r), int(c)) for r, c in interior]
    return None


def _can_place(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    oh, ow = occ.shape
    rs = max(0, r0 - pad)
    re = min(oh, r0 + h + pad)
    cs = max(0, c0 - pad)
    ce = min(ow, c0 + w + pad)
    return not bool(occ[rs:re, cs:ce].any())


def _mark_occ(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    oh, ow = occ.shape
    rs = max(0, r0 - pad)
    re = min(oh, r0 + h + pad)
    cs = max(0, c0 - pad)
    ce = min(ow, c0 + w + pad)
    occ[rs:re, cs:ce] = True


def _place_legend(
    inp: np.ndarray,
    out: np.ndarray,
    occ: np.ndarray,
    rng: random.Random,
) -> tuple[list[int], dict[int, int]]:
    ring_kind = rng.choice(["2x3", "3x3"])
    path = list(RING_PATHS[ring_kind])
    n = len(path)
    colors = rng.sample(list(ALL_COLORS), n)
    shift = rng.randrange(n)
    if rng.random() < 0.5:
        colors.reverse()
    colors = colors[shift:] + colors[:shift]

    h = max(r for r, _ in path) + 1
    w = max(c for _, c in path) + 1
    row_gap = rng.choice([0, 1])
    col_gap = rng.choice([0, 1])
    r0 = inp.shape[0] - h - row_gap
    c0 = col_gap

    for (dr, dc), color in zip(path, colors, strict=False):
        inp[r0 + dr, c0 + dc] = color
        out[r0 + dr, c0 + dc] = color

    _mark_occ(occ, r0, c0, h, w, pad=2)
    complement = {colors[i]: colors[(i + n // 2) % n] for i in range(n)}
    return colors, complement


def _place_fence(
    inp: np.ndarray,
    out: np.ndarray,
    occ: np.ndarray,
    boundary: np.ndarray,
    interior: list[tuple[int, int]],
    border_color: int,
    seed_color: int,
    fill_color: int,
    rng: random.Random,
) -> bool:
    h, w = boundary.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(1, inp.shape[0] - h):
        for c0 in range(1, inp.shape[1] - w):
            if _can_place(occ, r0, c0, h, w, pad=1):
                candidates.append((r0, c0))
    if not candidates:
        return False

    r0, c0 = rng.choice(candidates)
    for r, c in zip(*np.where(boundary), strict=False):
        inp[r0 + int(r), c0 + int(c)] = border_color
        out[r0 + int(r), c0 + int(c)] = seed_color

    seed_local = rng.choice(interior)
    inp[r0 + seed_local[0], c0 + seed_local[1]] = seed_color
    for rr, cc in interior:
        out[r0 + rr, c0 + cc] = fill_color

    _mark_occ(occ, r0, c0, h, w, pad=1)
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_fences = rng.choice([1, 2, 3])
    h = rng.randint(18, 28)
    w = rng.randint(18, 30)
    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    legend_colors, complement = _place_legend(inp, out, occ, rng)
    used_borders: set[int] = set()

    for _ in range(num_fences):
        shape = _random_filled_shape(rng)
        if shape is None:
            return None
        boundary, interior = shape

        seed_color = rng.choice(legend_colors)
        banned = {seed_color, int(complement[seed_color])}
        border_choices = [c for c in ALL_COLORS if c not in used_borders and c not in banned]
        if not border_choices:
            border_choices = [c for c in ALL_COLORS if c not in banned]
        if not border_choices:
            return None
        border_color = rng.choice(border_choices)
        used_borders.add(border_color)

        if not _place_fence(
            inp,
            out,
            occ,
            boundary,
            interior,
            border_color,
            seed_color,
            int(complement[seed_color]),
            rng,
        ):
            return None

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
