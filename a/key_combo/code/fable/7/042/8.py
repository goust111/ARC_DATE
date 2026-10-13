from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
PALETTE_MIN_LEN = 2
PALETTE_MAX_LEN = 5
MAX_GENERATION_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    runs = []
    start = end = int(indices[0])
    for value in map(int, indices[1:]):
        if value == end + 1:
            end = value
        else:
            runs.append((start, end))
            start = end = value
    runs.append((start, end))
    return runs


def _parse_palette(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[str, tuple[int, int, int, int]]] = []

    if h >= 2:
        cols = np.flatnonzero((g[0] != bg) & (g[1] != bg))
        for c0, c1 in _find_runs(cols):
            if c1 - c0 + 1 >= PALETTE_MIN_LEN and (h == 2 or np.all(g[2, c0 : c1 + 1] == bg)):
                candidates.append(("top", (0, c0, 1, c1)))
        cols = np.flatnonzero((g[h - 1] != bg) & (g[h - 2] != bg))
        for c0, c1 in _find_runs(cols):
            if c1 - c0 + 1 >= PALETTE_MIN_LEN and (h == 2 or np.all(g[h - 3, c0 : c1 + 1] == bg)):
                candidates.append(("bottom", (h - 2, c0, h - 1, c1)))

    if w >= 2:
        rows = np.flatnonzero((g[:, 0] != bg) & (g[:, 1] != bg))
        for r0, r1 in _find_runs(rows):
            if r1 - r0 + 1 >= PALETTE_MIN_LEN and (w == 2 or np.all(g[r0 : r1 + 1, 2] == bg)):
                candidates.append(("left", (r0, 0, r1, 1)))
        rows = np.flatnonzero((g[:, w - 1] != bg) & (g[:, w - 2] != bg))
        for r0, r1 in _find_runs(rows):
            if r1 - r0 + 1 >= PALETTE_MIN_LEN and (w == 2 or np.all(g[r0 : r1 + 1, w - 3] == bg)):
                candidates.append(("right", (r0, w - 2, r1, w - 1)))

    assert len(candidates) == 1, "expected exactly one edge palette"
    side, (r0, c0, r1, c1) = candidates[0]
    mask = np.zeros_like(g, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True

    if side == "top":
        olds = list(map(int, g[0, c0 : c1 + 1]))
        news = list(map(int, g[1, c0 : c1 + 1]))
    elif side == "bottom":
        olds = list(map(int, g[h - 1, c0 : c1 + 1]))
        news = list(map(int, g[h - 2, c0 : c1 + 1]))
    elif side == "left":
        olds = list(map(int, g[r0 : r1 + 1, 0]))
        news = list(map(int, g[r0 : r1 + 1, 1]))
    else:
        olds = list(map(int, g[r0 : r1 + 1, w - 1]))
        news = list(map(int, g[r0 : r1 + 1, w - 2]))

    assert len(olds) == len(set(olds))
    assert all(color != bg for color in olds + news)
    return dict(zip(olds, news)), mask


def _count_holes(grid: np.ndarray, comp_mask: np.ndarray, bg: int) -> int:
    coords = [(int(r), int(c)) for r, c in np.argwhere(comp_mask)]
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    local_bg = region == bg
    h, w = local_bg.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0

    for r in range(h):
        for c in range(w):
            if not local_bg[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_border = False
            while q:
                cr, cc = q.popleft()
                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and local_bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _advance_color(color: int, steps: int, mapping: dict[int, int]) -> int:
    current = int(color)
    for _ in range(int(steps)):
        if current not in mapping:
            break
        current = int(mapping[current])
    return current


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each monochrome component by walking along the palette chain a number of times equal to its hole count.

    Input structure:
    - A dominant background contains several separated single-color components, some with zero holes and some with one or more enclosed background holes.
    - A 2xN or Nx2 legend lies on one grid edge; the outer strip lists source colors and the inner strip lists their one-step destination colors.
    - Legend entries may chain, so a destination color can also appear as another source color.

    Rule:
    - Locate the edge legend and read it as an old->new mapping, with the edge-adjacent row or column as the source side.
    - Erase the whole legend to background in the output.
    - For every non-legend 4-connected monochrome component, count how many enclosed background regions lie inside its bbox.
    - Advance the component color along the mapping exactly that many steps, stopping early if the current color is not a source color; holes remain background.

    Key insight:
    - The legend gives only single-step substitutions; each object's own number of enclosed background holes tells you how far to walk along that visible mapping chain.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    mapping, palette_mask = _parse_palette(g, bg)

    out = g.copy()
    out[palette_mask] = bg

    for color in sorted(int(x) for x in np.unique(g)):
        if color == bg:
            continue
        color_mask = (g == color) & ~palette_mask
        for coords in _mask_components(color_mask):
            comp_mask = np.zeros_like(g, dtype=bool)
            for r, c in coords:
                comp_mask[r, c] = True
            hole_count = _count_holes(g, comp_mask, bg)
            out_color = _advance_color(color, hole_count, mapping)
            out[comp_mask] = out_color

    return out


def _solid_patch(color: int, rng: random.Random) -> np.ndarray:
    h = rng.randint(2, 4)
    w = rng.randint(3, 5)
    patch = np.full((h, w), color, dtype=int)
    if h >= 3 and w >= 4 and rng.random() < 0.5:
        corner_h = rng.randint(1, h - 1)
        corner_w = rng.randint(1, w - 2)
        if rng.random() < 0.5:
            patch[:corner_h, :corner_w] = BG
        else:
            patch[:corner_h, -corner_w:] = BG
    return patch


def _holey_patch(color: int, holes: int, rng: random.Random) -> np.ndarray:
    assert holes >= 1
    hole_h = rng.randint(1, 2)
    hole_w = rng.randint(1, 2)
    horizontal = rng.random() < 0.5

    if horizontal:
        interior_h = hole_h
        interior_w = holes * hole_w + (holes - 1)
        patch = np.full((interior_h + 2, interior_w + 2), color, dtype=int)
        patch[1 : 1 + interior_h, 1 : 1 + interior_w] = BG
        for i in range(1, holes):
            divider_c = 1 + i * hole_w + (i - 1)
            patch[1 : 1 + interior_h, divider_c] = color
    else:
        interior_h = holes * hole_h + (holes - 1)
        interior_w = hole_w
        patch = np.full((interior_h + 2, interior_w + 2), color, dtype=int)
        patch[1 : 1 + interior_h, 1 : 1 + interior_w] = BG
        for i in range(1, holes):
            divider_r = 1 + i * hole_h + (i - 1)
            patch[divider_r, 1 : 1 + interior_w] = color

    turns = rng.randint(0, 3)
    return np.rot90(patch, turns).copy()


def _build_patch(color: int, holes: int, rng: random.Random) -> np.ndarray:
    if holes == 0:
        return _solid_patch(color, rng)
    return _holey_patch(color, holes, rng)


def _reserve(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    hr, wc = occupied.shape
    occupied[max(0, r0 - 1) : min(hr, r0 + h + 1), max(0, c0 - 1) : min(wc, c0 + w + 1)] = True


def _place_palette(
    grid: np.ndarray, occupied: np.ndarray, olds: list[int], news: list[int], side: str, rng: random.Random
) -> None:
    h, w = grid.shape
    n = len(olds)
    if side == "top":
        c0 = rng.randint(1, w - n - 1)
        grid[0, c0 : c0 + n] = olds
        grid[1, c0 : c0 + n] = news
        _reserve(occupied, 0, c0, 2, n)
    elif side == "bottom":
        c0 = rng.randint(1, w - n - 1)
        grid[h - 1, c0 : c0 + n] = olds
        grid[h - 2, c0 : c0 + n] = news
        _reserve(occupied, h - 2, c0, 2, n)
    elif side == "left":
        r0 = rng.randint(1, h - n - 1)
        grid[r0 : r0 + n, 0] = olds
        grid[r0 : r0 + n, 1] = news
        _reserve(occupied, r0, 0, n, 2)
    else:
        r0 = rng.randint(1, h - n - 1)
        grid[r0 : r0 + n, w - 1] = olds
        grid[r0 : r0 + n, w - 2] = news
        _reserve(occupied, r0, w - 2, n, 2)


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != BG
    grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]


def _expected_patch(patch: np.ndarray, in_color: int, out_color: int) -> np.ndarray:
    out = patch.copy()
    out[out == in_color] = out_color
    return out


def _choose_chain(rng: random.Random) -> tuple[list[int], dict[int, int]]:
    edge_count = rng.randint(PALETTE_MIN_LEN, PALETTE_MAX_LEN)
    chain = rng.sample(list(range(1, 10)), edge_count + 1)
    mapping = {chain[i]: chain[i + 1] for i in range(edge_count)}
    return chain, mapping


def _component_specs(chain: list[int], rng: random.Random) -> list[tuple[int, int]]:
    edge_count = len(chain) - 1
    repeated_index = rng.randint(0, edge_count - 1)
    repeated_color = chain[repeated_index]
    max_steps = edge_count - repeated_index
    pair_steps = sorted(rng.sample(list(range(max_steps + 1)), 2))
    specs = [(repeated_color, pair_steps[0]), (repeated_color, pair_steps[1])]

    total = rng.randint(4, 6)
    free_colors = [color for color in range(1, 10) if color not in chain]
    while len(specs) < total:
        if free_colors and rng.random() < 0.25:
            color = rng.choice(free_colors)
        else:
            color = rng.choice(chain)
        hole_count = rng.randint(0, min(4, edge_count + 1))
        specs.append((color, hole_count))

    if all(holes > 0 for _, holes in specs):
        color, _ = specs[-1]
        specs[-1] = (color, 0)
    if all(holes == 0 for _, holes in specs):
        color, _ = specs[-1]
        specs[-1] = (color, 1)

    rng.shuffle(specs)
    return specs


def _place_components(
    inp: np.ndarray,
    out: np.ndarray,
    occupied: np.ndarray,
    specs: list[tuple[int, int]],
    mapping: dict[int, int],
    rng: random.Random,
) -> bool:
    h, w = inp.shape
    order = list(range(len(specs)))
    order.sort(key=lambda i: specs[i][1], reverse=True)

    for idx in order:
        color, holes = specs[idx]
        patch = _build_patch(color, holes, rng)
        ph, pw = patch.shape
        positions = []
        for r0 in range(1, h - ph):
            for c0 in range(1, w - pw):
                if occupied[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1].any():
                    continue
                positions.append((r0, c0))
        if not positions:
            return False
        r0, c0 = rng.choice(positions)
        _paste_patch(inp, patch, r0, c0)
        out_color = _advance_color(color, holes, mapping)
        _paste_patch(out, _expected_patch(patch, color, out_color), r0, c0)
        _reserve(occupied, r0, c0, ph, pw)
    return True


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    chain, mapping = _choose_chain(rng)
    specs = _component_specs(chain, rng)
    side = rng.choice(["top", "bottom", "left", "right"])

    h = rng.randint(18, 28)
    w = rng.randint(18, 28)
    if side in ("top", "bottom") and w < len(chain) + 3:
        w = len(chain) + 3
    if side in ("left", "right") and h < len(chain) + 3:
        h = len(chain) + 3

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    _place_palette(inp, occupied, chain[:-1], chain[1:], side, rng)
    if not _place_components(inp, out, occupied, specs, mapping, rng):
        return None

    if _get_bg(inp) != BG:
        return None
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        seed = random.randrange(1 << 30)
        rng = random.Random(seed)
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
