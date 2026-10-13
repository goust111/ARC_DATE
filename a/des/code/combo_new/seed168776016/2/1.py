import random

import numpy as np


BG = 0
FRAME = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("up", "down", "left", "right")
SIDE_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_HEIGHT_RANGE = (14, 30)
GRID_WIDTH_RANGE = (14, 30)
FRAME_COUNT_OPTIONS = (2, 3, 4)
FRAME_COUNT_WEIGHTS = (2, 2, 1)
OUTER_HEIGHT_RANGE = (4, 8)
OUTER_WIDTH_RANGE = (4, 8)
MAX_MARKERS = 7
PLACEMENT_ATTEMPTS = 150
GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    return border


def _gap_side(gap: tuple[int, int], r0: int, c0: int, r1: int, c1: int) -> str:
    gr, gc = gap
    if gr == r0 and c0 < gc < c1:
        return "up"
    if gr == r1 and c0 < gc < c1:
        return "down"
    if gc == c0 and r0 < gr < r1:
        return "left"
    if gc == c1 and r0 < gr < r1:
        return "right"
    raise AssertionError("gap must lie on exactly one non-corner frame side")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow outward rays from each framed region's border gap, using the interior marker count as the ray length.

    Input structure:
    - Black background surrounds several one-pixel-thick gray rectangular frames.
    - Each frame is missing exactly one non-corner border pixel, creating a unique gap on one side.
    - Inside each frame, one or more non-gray marker pixels of a single color are scattered away from the border.

    Rule:
    - Detect each gray rectangular frame and locate its unique border gap.
    - The gap side determines the outward direction: top/up, bottom/down, left/left, right/right.
    - Count the marker pixels inside the frame and draw a same-colored ray of that length outward from the gap, clipped by the grid edge.
    - Keep the frame and the interior markers unchanged.

    Key insight:
    - The missing border pixel encodes direction, while the number of enclosed marker pixels independently encodes ray length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    out = g.copy()
    frame_components = _components(g == FRAME)
    assert frame_components, "expected at least one gray frame"

    for comp in frame_components:
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose an interior"

        border = _perimeter(r0, c0, r1, c1)
        comp_set = set(comp)
        assert comp_set <= border, "frame pixels must lie on a rectangle perimeter"

        missing = list(border - comp_set)
        assert len(missing) == 1, "expected exactly one frame gap"
        gap = missing[0]
        side = _gap_side(gap, r0, c0, r1, c1)
        dr, dc = SIDE_TO_DELTA[side]

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_mask = (interior != BG) & (interior != FRAME)
        marker_count = int(marker_mask.sum())
        assert marker_count > 0, "each frame must contain markers"
        marker_colors = np.unique(interior[marker_mask])
        assert marker_colors.size == 1, "markers inside one frame must share a color"
        marker_color = int(marker_colors[0])

        for step in range(1, marker_count + 1):
            rr = gap[0] + dr * step
            cc = gap[1] + dc * step
            if not (0 <= rr < h and 0 <= cc < w):
                break
            out[rr, cc] = marker_color

    return out


def _sample_frame_count(rng: random.Random) -> int:
    return rng.choices(FRAME_COUNT_OPTIONS, weights=FRAME_COUNT_WEIGHTS, k=1)[0]


def _avail_from_position(side: str, r0: int, c0: int, outer_h: int, outer_w: int, h: int, w: int) -> int:
    if side == "up":
        return r0
    if side == "down":
        return h - (r0 + outer_h)
    if side == "left":
        return c0
    return w - (c0 + outer_w)


def _frame_support(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    ray_cells: list[tuple[int, int]],
    h: int,
    w: int,
) -> set[tuple[int, int]]:
    support = set()
    for rr in range(max(0, r0 - 1), min(h - 1, r1 + 1) + 1):
        for cc in range(max(0, c0 - 1), min(w - 1, c1 + 1) + 1):
            support.add((rr, cc))
    for rr, cc in ray_cells:
        for ar in range(max(0, rr - 1), min(h - 1, rr + 1) + 1):
            for ac in range(max(0, cc - 1), min(w - 1, cc + 1) + 1):
                support.add((ar, ac))
    return support


def _paint_frame(
    rng: random.Random,
    grid: np.ndarray,
    blocked: np.ndarray,
    r0: int,
    c0: int,
    outer_h: int,
    outer_w: int,
    side: str,
    color: int,
    marker_count: int,
) -> bool:
    h, w = grid.shape
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    border = _perimeter(r0, c0, r1, c1)

    if side in ("up", "down"):
        gap = (r0 if side == "up" else r1, rng.randint(c0 + 1, c1 - 1))
    else:
        gap = (rng.randint(r0 + 1, r1 - 1), c0 if side == "left" else c1)

    avail = _avail_from_position(side, r0, c0, outer_h, outer_w, h, w)
    ray_len = min(marker_count, avail)
    if ray_len <= 0:
        return False

    dr, dc = SIDE_TO_DELTA[side]
    ray_cells = [(gap[0] + dr * step, gap[1] + dc * step) for step in range(1, ray_len + 1)]
    support = _frame_support(r0, c0, r1, c1, ray_cells, h, w)
    if any(blocked[rr, cc] for rr, cc in support):
        return False

    interior_cells = [(rr, cc) for rr in range(r0 + 1, r1) for cc in range(c0 + 1, c1)]
    markers = rng.sample(interior_cells, marker_count)

    for rr, cc in border:
        if (rr, cc) != gap:
            grid[rr, cc] = FRAME
    for rr, cc in markers:
        grid[rr, cc] = color
    for rr, cc in support:
        blocked[rr, cc] = True
    return True


def _try_place_frame(
    rng: random.Random,
    grid: np.ndarray,
    blocked: np.ndarray,
    side: str,
    color: int,
    want_truncation: bool,
) -> bool:
    h, w = grid.shape
    for _ in range(PLACEMENT_ATTEMPTS):
        outer_h = rng.randint(*OUTER_HEIGHT_RANGE)
        outer_w = rng.randint(*OUTER_WIDTH_RANGE)
        if outer_h >= h - 1 or outer_w >= w - 1:
            continue

        interior_area = (outer_h - 2) * (outer_w - 2)
        if interior_area <= 0:
            continue

        if side in ("up", "down"):
            max_avail = h - outer_h - 1
            c_low = 1
            c_high = w - outer_w - 1
            if c_low > c_high:
                c_low = 0
                c_high = w - outer_w
            if c_low > c_high:
                continue
            c0 = rng.randint(c_low, c_high)
            if max_avail < 1:
                continue
            if want_truncation:
                avail_limit = min(3, max_avail)
                avail = rng.randint(1, avail_limit)
            else:
                avail = rng.randint(1, max_avail)
            r0 = avail if side == "up" else h - outer_h - avail
        else:
            max_avail = w - outer_w - 1
            r_low = 1
            r_high = h - outer_h - 1
            if r_low > r_high:
                r_low = 0
                r_high = h - outer_h
            if r_low > r_high:
                continue
            r0 = rng.randint(r_low, r_high)
            if max_avail < 1:
                continue
            if want_truncation:
                avail_limit = min(3, max_avail)
                avail = rng.randint(1, avail_limit)
            else:
                avail = rng.randint(1, max_avail)
            c0 = avail if side == "left" else w - outer_w - avail

        if want_truncation:
            low = avail + 1
            high = min(interior_area, MAX_MARKERS)
        else:
            low = 1
            high = min(interior_area, avail, MAX_MARKERS)
        if low > high:
            continue

        marker_count = rng.randint(low, high)
        if _paint_frame(rng, grid, blocked, r0, c0, outer_h, outer_w, side, color, marker_count):
            return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_HEIGHT_RANGE)
    w = rng.randint(*GRID_WIDTH_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    frame_count = _sample_frame_count(rng)
    colors = rng.sample(MARKER_COLORS, k=frame_count)
    sides = list(SIDES)
    rng.shuffle(sides)
    chosen_sides = sides[:frame_count]
    truncation_flags = [bool(rng.getrandbits(1)) for _ in range(frame_count)]

    order = list(range(frame_count))
    rng.shuffle(order)
    for idx in order:
        if not _try_place_frame(
            rng,
            grid,
            blocked,
            chosen_sides[idx],
            colors[idx],
            truncation_flags[idx],
        ):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
