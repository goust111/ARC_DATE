import random

import numpy as np


BG_COLOR = 0
MARKER_COLOR = 5
FRAME_COLOR = 9
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8]
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
LAYOUTS = {
    2: [(1, 2), (2, 1)],
    3: [(1, 3), (3, 1), (2, 2)],
    4: [(2, 2)],
    5: [(2, 3), (3, 2)],
}
SLOT_MIN = 6
SLOT_MAX = 8
GAP = 1
GENERATION_RETRIES = 300


def _components4(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: set[tuple[int, int]] = set()
            while stack:
                rr, cc = stack.pop()
                comp.add((rr, cc))
                for dr, dc in DIRECTIONS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_bbox(component: set[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in component]
    cols = [c for _, c in component]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    expected = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    if component != expected:
        return None
    return r0, c0, r1, c1


def _paint_triangle(
    out: np.ndarray,
    color: int,
    bbox: tuple[int, int, int, int],
    seed: tuple[int, int],
    direction: tuple[int, int],
) -> None:
    r0, c0, r1, c1 = bbox
    sr, sc = seed
    dr, dc = direction

    if dr == -1:
        max_steps = sr - (r0 + 1)
    elif dr == 1:
        max_steps = (r1 - 1) - sr
    elif dc == -1:
        max_steps = sc - (c0 + 1)
    else:
        max_steps = (c1 - 1) - sc

    for step in range(1, max_steps + 1):
        half = step - 1
        if dr != 0:
            rr = sr + dr * step
            cs = max(c0 + 1, sc - half)
            ce = min(c1 - 1, sc + half)
            out[rr, cs : ce + 1] = color
        else:
            cc = sc + dc * step
            rs = max(r0 + 1, sr - half)
            re = min(r1 - 1, sr + half)
            out[rs : re + 1, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Expand each seed inside its maroon frame into a direction-marked widening triangle.

    Input structure:
    - The background is black.
    - Maroon pixels form several disjoint 1-cell-thick hollow rectangles.
    - Each rectangle contains exactly one non-black, non-grey seed pixel and one grey pixel directly above, below, left, or right of that seed.

    Rule:
    - The grey pixel's position relative to the seed gives the growth direction.
    - Remove the grey marker and grow the seed in that direction with centered slices of widths 1, 3, 5, ...; the first slice is the adjacent cell in the marked direction.
    - Clip every slice to the rectangle interior and stop when the next slice would cross the inner wall.
    - Keep the seed pixel and the maroon border unchanged.

    Key insight:
    - The grey pixel encodes direction, while the maroon box limits a centered triangle whose width increases by 2 on each step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frame_components = _components4(g == FRAME_COLOR)
    frames: list[tuple[int, int, int, int]] = []
    for component in frame_components:
        bbox = _frame_bbox(component)
        assert bbox is not None, "all maroon components must be hollow rectangles"
        frames.append(bbox)
    assert frames, "expected at least one frame"

    out = g.copy()
    for r0, c0, r1, c1 in sorted(frames):
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_cells = np.argwhere(
            (interior != BG_COLOR)
            & (interior != MARKER_COLOR)
            & (interior != FRAME_COLOR)
        )
        marker_cells = np.argwhere(interior == MARKER_COLOR)
        assert len(seed_cells) == 1, "each frame must contain one seed"
        assert len(marker_cells) == 1, "each frame must contain one grey marker"

        sr, sc = seed_cells[0]
        mr, mc = marker_cells[0]
        sr += r0 + 1
        sc += c0 + 1
        mr += r0 + 1
        mc += c0 + 1

        dr = mr - sr
        dc = mc - sc
        assert abs(dr) + abs(dc) == 1, "marker must be cardinally adjacent to seed"

        out[mr, mc] = BG_COLOR
        _paint_triangle(
            out=out,
            color=int(g[sr, sc]),
            bbox=(r0, c0, r1, c1),
            seed=(sr, sc),
            direction=(dr, dc),
        )

    return out


def _sample_directions(rng: random.Random, count: int) -> list[tuple[int, int]]:
    directions: list[tuple[int, int]] = []
    while len(directions) < count:
        batch = DIRECTIONS[:]
        rng.shuffle(batch)
        directions.extend(batch)
    return directions[:count]


def _slot_origins(sizes: list[int]) -> list[int]:
    pos = GAP
    origins = []
    for size in sizes:
        origins.append(pos)
        pos += size + GAP
    return origins


def _choose_seed(
    rng: random.Random,
    inner_h: int,
    inner_w: int,
    direction: tuple[int, int],
) -> tuple[int, int]:
    dr, dc = direction
    if dr == -1:
        return rng.randint(1, inner_h - 1), rng.randint(0, inner_w - 1)
    if dr == 1:
        return rng.randint(0, inner_h - 2), rng.randint(0, inner_w - 1)
    if dc == -1:
        return rng.randint(0, inner_h - 1), rng.randint(1, inner_w - 1)
    return rng.randint(0, inner_h - 1), rng.randint(0, inner_w - 2)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    box_count = rng.randint(2, 5)
    layout = rng.choice(LAYOUTS[box_count])
    rows, cols = layout

    slot_heights = [rng.randint(SLOT_MIN, SLOT_MAX) for _ in range(rows)]
    slot_widths = [rng.randint(SLOT_MIN, SLOT_MAX) for _ in range(cols)]
    height = sum(slot_heights) + (rows + 1) * GAP
    width = sum(slot_widths) + (cols + 1) * GAP
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BG_COLOR, dtype=int)
    row_origins = _slot_origins(slot_heights)
    col_origins = _slot_origins(slot_widths)

    slots = [(rr, cc) for rr in range(rows) for cc in range(cols)]
    rng.shuffle(slots)
    chosen_slots = slots[:box_count]
    directions = _sample_directions(rng, box_count)
    colors = rng.sample(SEED_COLORS, box_count)

    for idx, (slot_r, slot_c) in enumerate(chosen_slots):
        slot_h = slot_heights[slot_r]
        slot_w = slot_widths[slot_c]
        box_h = rng.randint(5, slot_h)
        box_w = rng.randint(5, slot_w)
        top = row_origins[slot_r] + rng.randint(0, slot_h - box_h)
        left = col_origins[slot_c] + rng.randint(0, slot_w - box_w)

        bottom = top + box_h - 1
        right = left + box_w - 1
        grid[top, left : right + 1] = FRAME_COLOR
        grid[bottom, left : right + 1] = FRAME_COLOR
        grid[top : bottom + 1, left] = FRAME_COLOR
        grid[top : bottom + 1, right] = FRAME_COLOR

        inner_h = box_h - 2
        inner_w = box_w - 2
        seed_r, seed_c = _choose_seed(rng, inner_h, inner_w, directions[idx])
        sr = top + 1 + seed_r
        sc = left + 1 + seed_c
        dr, dc = directions[idx]
        mr = sr + dr
        mc = sc + dc

        grid[mr, mc] = MARKER_COLOR
        grid[sr, sc] = colors[idx]

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
