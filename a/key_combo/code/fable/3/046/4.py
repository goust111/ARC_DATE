from __future__ import annotations

from collections import deque
import random

import numpy as np


NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _foreground_colors(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(int(v) for v in np.unique(grid) if int(v) != int(bg))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        pts: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for dr, dc in NEIGHBORS4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _as_cells(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    return frozenset((int(r), int(c)) for r, c in coords)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < h and 0 <= int(c) < w


def _aligned_offsets(
    body: frozenset[tuple[int, int]],
    ghost: frozenset[tuple[int, int]],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    offsets: set[tuple[int, int]] = set()
    for gr, gc in ghost:
        for br, bc in body:
            dr, dc = int(gr - br), int(gc - bc)
            if dr == 0 and dc == 0:
                continue
            if all((r - dr, c - dc) in body for r, c in ghost):
                visible = {
                    (r + dr, c + dc)
                    for r, c in body
                    if _in_bounds(shape, r + dr, c + dc)
                }
                if not ghost < visible:
                    continue
                if visible & body:
                    continue
                offsets.add((dr, dc))
    return sorted(offsets)


def _find_lattice_candidate(
    g: np.ndarray,
) -> tuple[int, frozenset[tuple[int, int]], tuple[int, int]]:
    bg = _get_bg(g)
    colors = _foreground_colors(g, bg)
    assert len(colors) == 1, "expected one sprite color besides the background"
    color = int(colors[0])

    fg = g == color
    foreground = _as_cells(np.argwhere(fg))
    components = _components(fg)
    assert components, "expected a complete sprite and a residual fragment"

    candidates: list[tuple[int, frozenset[tuple[int, int]], tuple[int, int]]] = []
    for comp in components:
        body = _as_cells(comp)
        ghost = foreground - body
        if not ghost:
            continue
        if len(body) <= len(ghost):
            continue
        for offset in _aligned_offsets(body, ghost, g.shape):
            candidates.append((color, body, offset))

    assert len(candidates) == 1, "expected one complete sprite with one unique residual alignment"
    return candidates[0]


def _paint_lattice(
    shape: tuple[int, int],
    bg: int,
    color: int,
    body: frozenset[tuple[int, int]],
    offset: tuple[int, int],
) -> np.ndarray:
    h, w = shape
    dr, dc = map(int, offset)
    assert (dr, dc) != (0, 0), "lattice vector must be nonzero"

    out = np.full((h, w), int(bg), dtype=int)
    limit = 2 * (h + w) + 1
    for k in range(-limit, limit + 1):
        any_visible = False
        for r, c in body:
            rr, cc = int(r + k * dr), int(c + k * dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(color)
                any_visible = True
        if not any_visible and abs(k) > h + w:
            continue
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the residual ghost sprite and tile the sprite along its inferred displacement vector.

    Input structure:
    - The background is the most common color.
    - Exactly one foreground color appears as one complete 4-connected sprite body plus a
      smaller set of same-color residual pixels from another translated copy of that sprite.
    - The residual pixels align to the complete sprite under exactly one nonzero translation.

    Rule:
    - Use shape alignment between the residual pixels and the complete sprite to infer the
      unique displacement vector from the body to the residual copy.
    - Starting from the complete body, stamp full copies at every positive and negative
      integer multiple of that vector.
    - Keep the original grid size and clip every stamped copy at the grid boundary.

    Key insight:
    - The small same-color pixels are not noise; they identify the two-dimensional lattice
      vector, and the full answer is the entire bidirectional lattice line through the body.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30 by 30"

    bg = _get_bg(g)
    color, body, offset = _find_lattice_candidate(g)
    return _paint_lattice(g.shape, bg, color, body, offset)


# ----------------------------- generator -----------------------------

GEN_MAX_ATTEMPTS = 800
GEN_SHAPE_ATTEMPTS = 200
GEN_FRAGMENT_ATTEMPTS = 200
GEN_MIN_SIZE = 24
GEN_MAX_SIZE = 30
GEN_COLORS = tuple(range(10))


def _normalize(cells: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _bbox_size(cells: frozenset[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _c in cells) + 1, max(c for _r, c in cells) + 1


def _random_sprite() -> frozenset[tuple[int, int]] | None:
    for _ in range(GEN_SHAPE_ATTEMPTS):
        height = random.randint(4, 6)
        width = random.randint(4, 6)
        area = random.randint(8, min(15, height * width - 2))
        start = (random.randrange(height), random.randrange(width))
        cells: set[tuple[int, int]] = {start}
        while len(cells) < area:
            r, c = random.choice(tuple(cells))
            dr, dc = random.choice(NEIGHBORS4)
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width:
                cells.add((rr, cc))

        norm = _normalize(cells)
        h, w = _bbox_size(norm)
        if h < 3 or w < 3:
            continue
        if len(norm) == h * w:
            continue
        row_counts = {r: 0 for r in range(h)}
        col_counts = {c: 0 for c in range(w)}
        for r, c in norm:
            row_counts[r] += 1
            col_counts[c] += 1
        if len(set(row_counts.values())) == 1 and len(set(col_counts.values())) == 1:
            continue
        return norm
    return None


def _shift(cells: frozenset[tuple[int, int]], dr: int, dc: int) -> frozenset[tuple[int, int]]:
    return frozenset((r + int(dr), c + int(dc)) for r, c in cells)


def _visible(cells: frozenset[tuple[int, int]], shape: tuple[int, int]) -> frozenset[tuple[int, int]]:
    return frozenset((r, c) for r, c in cells if _in_bounds(shape, r, c))


def _separate(a: frozenset[tuple[int, int]], b: frozenset[tuple[int, int]]) -> bool:
    for r, c in a:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in b:
                    return False
    return True


def _placement_options(
    rows: int,
    cols: int,
    sprite: frozenset[tuple[int, int]],
    vector: tuple[int, int],
) -> list[tuple[int, int]]:
    bh, bw = _bbox_size(sprite)
    dr, dc = vector
    options: list[tuple[int, int]] = []
    for r0 in range(rows - bh + 1):
        for c0 in range(cols - bw + 1):
            body = _shift(sprite, r0, c0)
            plus = _shift(body, dr, dc)
            minus = _shift(body, -dr, -dc)
            if len(_visible(plus, (rows, cols))) != len(body):
                continue
            if len(_visible(minus, (rows, cols))) != len(body):
                continue
            if not _separate(body, plus) or not _separate(body, minus):
                continue
            plus2 = _visible(_shift(body, 2 * dr, 2 * dc), (rows, cols))
            minus2 = _visible(_shift(body, -2 * dr, -2 * dc), (rows, cols))
            if not plus2 and not minus2:
                continue
            options.append((r0, c0))
    return options


def _fragment_for(
    body: frozenset[tuple[int, int]],
    vector: tuple[int, int],
    shape: tuple[int, int],
) -> frozenset[tuple[int, int]] | None:
    dr, dc = vector
    shifted_body = _shift(body, dr, dc)
    body_list = sorted(body)
    for _ in range(GEN_FRAGMENT_ATTEMPTS):
        size = random.randint(3, min(7, len(body_list) - 2))
        chosen = random.sample(body_list, size)
        local_rows = {r for r, _c in chosen}
        local_cols = {c for _r, c in chosen}
        if len(local_rows) < 2 or len(local_cols) < 2:
            continue
        fragment = frozenset((r + dr, c + dc) for r, c in chosen)
        if not fragment < shifted_body:
            continue
        if len(_aligned_offsets(body, fragment, shape)) != 1:
            continue
        return fragment
    return None


def _render_input(
    rows: int,
    cols: int,
    bg: int,
    color: int,
    body: frozenset[tuple[int, int]],
    fragment: frozenset[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((rows, cols), int(bg), dtype=int)
    for r, c in body | fragment:
        grid[r, c] = int(color)
    return grid


def _expected_output(
    rows: int,
    cols: int,
    bg: int,
    color: int,
    body: frozenset[tuple[int, int]],
    vector: tuple[int, int],
) -> np.ndarray:
    return _paint_lattice((rows, cols), bg, color, body, vector)


def _nondegenerate(
    inp: np.ndarray,
    out: np.ndarray,
    body: frozenset[tuple[int, int]],
    fragment: frozenset[tuple[int, int]],
    vector: tuple[int, int],
    color: int,
) -> bool:
    rows, cols = inp.shape
    dr, dc = vector
    full_ghost = _shift(body, dr, dc)
    missing_from_fragment = _visible(full_ghost - fragment, (rows, cols))
    if not missing_from_fragment:
        return False
    if not all(int(out[r, c]) == int(color) and int(inp[r, c]) != int(color) for r, c in missing_from_fragment):
        return False

    opposite = _visible(_shift(body, -dr, -dc), (rows, cols))
    if not opposite:
        return False
    if not any(int(out[r, c]) == int(color) and int(inp[r, c]) != int(color) for r, c in opposite):
        return False

    further = _visible(_shift(body, 2 * dr, 2 * dc), (rows, cols)) | _visible(
        _shift(body, -2 * dr, -2 * dc), (rows, cols)
    )
    if not further:
        return False
    if not any(int(out[r, c]) == int(color) and int(inp[r, c]) != int(color) for r, c in further):
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    return np.array_equal(solved, out)


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(GEN_MIN_SIZE, GEN_MAX_SIZE)
    cols = random.randint(GEN_MIN_SIZE, GEN_MAX_SIZE)
    bg = random.choice(GEN_COLORS)
    color = random.choice([c for c in GEN_COLORS if c != bg])

    sprite = _random_sprite()
    if sprite is None:
        return None
    bh, bw = _bbox_size(sprite)

    max_row_step = min(9, (rows - bh) // 2)
    max_col_step = min(9, (cols - bw) // 2)
    min_row_step = min(max_row_step, bh + 1)
    min_col_step = min(max_col_step, bw + 1)
    if min_row_step < 3 or min_col_step < 3:
        return None
    row_mag = random.randint(min_row_step, max_row_step)
    col_mag = random.randint(min_col_step, max_col_step)
    vector = (random.choice((-row_mag, row_mag)), random.choice((-col_mag, col_mag)))

    placements = _placement_options(rows, cols, sprite, vector)
    if not placements:
        return None
    r0, c0 = random.choice(placements)
    body = _shift(sprite, r0, c0)
    fragment = _fragment_for(body, vector, (rows, cols))
    if fragment is None:
        return None
    if not _separate(body, fragment):
        return None

    inp = _render_input(rows, cols, bg, color, body, fragment)
    out = _expected_output(rows, cols, bg, color, body, vector)
    bg_count = int(np.count_nonzero(inp == int(bg)))
    fg_count = int(np.count_nonzero(inp == int(color)))
    if bg_count <= fg_count:
        return None
    if not _nondegenerate(inp, out, body, fragment, vector, color):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid residual lattice input")
