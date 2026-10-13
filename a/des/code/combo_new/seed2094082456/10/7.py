import random

import numpy as np


BG = 0
FRAME = 2
INNER_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 400
PLACEMENT_RETRIES = 120


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _frame_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    if not coords:
        return None
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    expected = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    if set(coords) != expected:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _topmost_color_key(points: list[tuple[int, int]]) -> tuple[int, int]:
    return min((int(r), int(c)) for r, c in points)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each red rectangle by majority vote, then move the minority count onto the top border.

    Input structure:
    - Black background surrounds several non-touching, single-pixel-wide red rectangular frames.
    - Each frame encloses a black interior containing a small set of colored pixels.
    - Inside any one rectangle there are exactly two non-black, non-red colors, with at most six colored pixels total.

    Rule:
    - For each red rectangular frame, count the two interior colors.
    - Fill the entire interior with the color that appears more often.
    - If the counts tie, choose the color whose topmost interior pixel is higher in the grid; break any same-row tie by the leftmost such pixel.
    - Let N be the count of the losing color. Replace the first N pixels of the rectangle's top border, starting at the top-left corner and moving right, with that losing color. Leave the rest of the frame red.

    Key insight:
    - The two interior colors encode two different outputs at once: the winner paints the whole inside, while the loser survives only as a short top-border tally whose length is its count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for comp in _components4(g == FRAME):
        bbox = _frame_bbox(comp)
        if bbox is None:
            continue
        r0, c0, r1, c1 = bbox
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        pts = np.argwhere((inner != BG) & (inner != FRAME))
        if pts.size == 0:
            continue

        color_points: dict[int, list[tuple[int, int]]] = {}
        for rr, cc in pts:
            color = int(inner[int(rr), int(cc)])
            color_points.setdefault(color, []).append((int(r0 + 1 + rr), int(c0 + 1 + cc)))

        colors = list(color_points)
        assert len(colors) == 2, "each rectangle must contain exactly two interior colors"

        counts = {color: len(color_points[color]) for color in colors}
        if counts[colors[0]] != counts[colors[1]]:
            winner = max(colors, key=lambda color: counts[color])
            loser = min(colors, key=lambda color: counts[color])
        else:
            winner = min(colors, key=lambda color: _topmost_color_key(color_points[color]))
            loser = colors[0] if colors[1] == winner else colors[1]

        out[r0 + 1 : r1, c0 + 1 : c1] = int(winner)
        n = int(counts[loser])
        out[r0, c0 : c1 + 1] = FRAME
        out[r0, c0 : min(c1 + 1, c0 + n)] = int(loser)

    return out


def _placeable(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    h0, w0 = blocked.shape
    if r0 < 1 or c0 < 1 or r0 + h >= h0 or c0 + w >= w0:
        return False
    return not bool(blocked[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1].any())


def _sample_counts(rng: random.Random, interior_area: int, force_tie: bool) -> tuple[int, int] | None:
    if force_tie:
        ms = [m for m in (1, 2, 3) if 2 * m <= interior_area]
        if not ms:
            return None
        m = rng.choice(ms)
        return int(m), int(m)

    pairs = [
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 3),
        (2, 4),
    ]
    valid = [pair for pair in pairs if pair[0] + pair[1] <= interior_area]
    if not valid:
        return None
    loser, winner = rng.choice(valid)
    return int(winner), int(loser)


def _pick_cell_from_rows(
    rng: random.Random,
    cells: list[tuple[int, int]],
    row_predicate,
    used: set[tuple[int, int]],
) -> tuple[int, int] | None:
    pool = [cell for cell in cells if row_predicate(cell[0]) and cell not in used]
    if not pool:
        return None
    return rng.choice(pool)


def _draw_rectangle(
    grid: np.ndarray,
    out: np.ndarray,
    rng: random.Random,
    r0: int,
    c0: int,
    h: int,
    w: int,
    force_tie: bool,
) -> bool:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    inner_h = h - 2
    inner_w = w - 2
    interior_area = inner_h * inner_w
    counts = _sample_counts(rng, interior_area, force_tie)
    if counts is None:
        return False

    fill_count, mark_count = counts
    fill_color, mark_color = rng.sample(INNER_COLORS, 2)
    interior_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    used: set[tuple[int, int]] = set()

    fill_cells: list[tuple[int, int]] = []
    mark_cells: list[tuple[int, int]] = []
    if force_tie:
        fill_head = _pick_cell_from_rows(rng, interior_cells, lambda row: row == r0 + 1, used)
        mark_head = _pick_cell_from_rows(rng, interior_cells, lambda row: row >= r0 + 2, used)
        if fill_head is None or mark_head is None:
            return False
        fill_cells.append(fill_head)
        mark_cells.append(mark_head)
        used.update((fill_head, mark_head))

    if force_tie:
        mark_candidates = [cell for cell in interior_cells if cell not in used and cell[0] >= r0 + 2]
        rng.shuffle(mark_candidates)
        mark_need = mark_count - len(mark_cells)
        if len(mark_candidates) < mark_need:
            return False
        mark_cells.extend(mark_candidates[:mark_need])
        used.update(mark_cells)
        fill_candidates = [cell for cell in interior_cells if cell not in used]
        rng.shuffle(fill_candidates)
        fill_need = fill_count - len(fill_cells)
        if len(fill_candidates) < fill_need:
            return False
        fill_cells.extend(fill_candidates[:fill_need])
    else:
        remaining = [cell for cell in interior_cells if cell not in used]
        if len(remaining) < (fill_count - len(fill_cells)) + (mark_count - len(mark_cells)):
            return False
        rng.shuffle(remaining)
        fill_need = fill_count - len(fill_cells)
        mark_need = mark_count - len(mark_cells)
        fill_cells.extend(remaining[:fill_need])
        mark_cells.extend(remaining[fill_need : fill_need + mark_need])

    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    for r, c in fill_cells:
        grid[r, c] = int(fill_color)
    for r, c in mark_cells:
        grid[r, c] = int(mark_color)

    out[r0, c0 : c1 + 1] = FRAME
    out[r1, c0 : c1 + 1] = FRAME
    out[r0 : r1 + 1, c0] = FRAME
    out[r0 : r1 + 1, c1] = FRAME
    out[r0 + 1 : r1, c0 + 1 : c1] = int(fill_color)
    out[r0, c0 : c0 + mark_count] = int(mark_color)
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 30)
    w = rng.randint(14, 30)
    n_rects = rng.randint(1, 4)

    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    placed = 0
    branches = [bool(rng.randint(0, 1)) for _ in range(n_rects)]
    rng.shuffle(branches)

    for force_tie in branches:
        success = False
        for _ in range(PLACEMENT_RETRIES):
            rh = rng.randint(5, min(10, h - 2))
            rw = rng.randint(5, min(10, w - 2))
            r0 = rng.randint(1, h - rh - 1)
            c0 = rng.randint(1, w - rw - 1)
            if not _placeable(blocked, r0, c0, rh, rw):
                continue
            if not _draw_rectangle(inp, out, rng, r0, c0, rh, rw, force_tie):
                continue
            blocked[r0 - 1 : r0 + rh + 1, c0 - 1 : c0 + rw + 1] = True
            placed += 1
            success = True
            break
        if not success:
            return None

    if placed != n_rects:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
