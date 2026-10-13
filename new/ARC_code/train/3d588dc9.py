from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg
import random


CUTOUT_COLOR = 0
SOURCE_COLOR = 5
MARK_COLOR = 6


def _largest_solid_rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = mask.shape
    rects: list[tuple[int, int, int, int, int]] = []
    for r0 in range(h):
        for r1 in range(r0, h):
            for c0 in range(w):
                for c1 in range(c0, w):
                    if bool(mask[r0 : r1 + 1, c0 : c1 + 1].all()):
                        area = (r1 - r0 + 1) * (c1 - c0 + 1)
                        rects.append((area, r0, c0, r1, c1))

    assert rects, "cutout component must contain at least one solid rectangle"
    largest_area = max(area for area, *_ in rects)
    return [(r0, c0, r1, c1) for area, r0, c0, r1, c1 in rects if area == largest_area]


def _core_and_tail(g: np.ndarray, comp: dict) -> tuple[int, int, int, int, list[tuple[int, int]]] | None:
    r0, c0, r1, c1 = comp["bbox"]
    sub = g[r0 : r1 + 1, c0 : c1 + 1] == CUTOUT_COLOR
    candidates: list[tuple[int, int, int, int, list[tuple[int, int]]]] = []

    for rr0, cc0, rr1, cc1 in _largest_solid_rectangles(sub):
        core = np.zeros_like(sub, dtype=bool)
        core[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        tail_rel = np.argwhere(sub & ~core)
        if int(tail_rel.shape[0]) != 2:
            continue
        tail = [(int(r0 + r), int(c0 + c)) for r, c in tail_rel]
        candidates.append((int(r0 + rr0), int(c0 + cc0), int(r0 + rr1), int(c0 + cc1), tail))

    if not candidates:
        return None
    assert len(candidates) == 1, "rectangle-plus-tail cutout must have one maximal rectangular core"
    return candidates[0]


def _source_components(bg: int, comps: list[dict]) -> list[dict]:
    sources = [
        comp
        for comp in comps
        if int(comp["color"]) == SOURCE_COLOR
        and int(comp["area"]) > 1
    ]
    assert len(sources) == 1, "exactly one active gray source region must be present"
    return sources


def _distractor_source_components(bg: int, comps: list[dict]) -> list[dict]:
    distractor_sources = [
        comp
        for comp in comps
        if int(comp["color"]) not in {bg, CUTOUT_COLOR, SOURCE_COLOR, MARK_COLOR}
        and int(comp["area"]) > 1
    ]
    assert len(distractor_sources) == 1, "exactly one non-gray distractor source region must be present"
    return distractor_sources


def _visible_source_ids(
    g: np.ndarray,
    bg: int,
    source_ids: dict[tuple[int, int], int],
    rows: range,
    side_col: int,
    dc: int,
) -> list[int]:
    hits: list[int] = []
    for r in rows:
        c = side_col + dc
        while 0 <= c < g.shape[1]:
            if int(g[r, c]) == bg:
                c += dc
                continue
            source_id = source_ids.get((int(r), int(c)))
            if source_id is not None:
                hits.append(source_id)
            break
    return hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace the active black cutout's tail with background and mark the cutout side facing a colored source region.

    Input structure:
    - The dominant color is background, with black rectangle-plus-tail cutouts, one gray source region, and colored distractors.
    - Non-gray colored regions and single pixels are distractors/blockers, even when they are multi-pixel staircase shapes.

    Rule:
    - Find each black cutout whose largest solid rectangle leaves exactly a two-pixel tail.
    - The active gray source region is visible only by horizontal rays through background from a cutout's left or right side.
    - Select the unique cutout side with the most same-source visible rows, erase that cutout's tail to background, and recolor the full selected vertical side pink (6).

    Key insight:
    - The colored regions do not move; their horizontal line-of-sight identifies which rectangular cutout is being opened and which side receives the pink mark.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    _distractor_source_components(bg, comps)
    sources = _source_components(bg, comps)

    source_ids: dict[tuple[int, int], int] = {}
    for source_id, source in enumerate(sources):
        for r, c in source["coords"]:
            source_ids[(int(r), int(c))] = int(source_id)

    candidates: list[tuple[int, tuple[int, int, int, int, list[tuple[int, int]]], str]] = []
    for comp in comps:
        if int(comp["color"]) != CUTOUT_COLOR:
            continue
        core_tail = _core_and_tail(g, comp)
        if core_tail is None:
            continue

        r0, c0, r1, c1, _tail = core_tail
        side_hits: list[tuple[int, str]] = []
        for side, side_col, dc in (("left", c0, -1), ("right", c1, 1)):
            hits = _visible_source_ids(g, bg, source_ids, range(r0, r1 + 1), side_col, dc)
            if hits:
                assert len(set(hits)) == 1, "one side must see at most one source region"
                side_hits.append((len(hits), side))

        if side_hits:
            max_hits = max(hit_count for hit_count, _side in side_hits)
            winning_sides = [side for hit_count, side in side_hits if hit_count == max_hits]
            assert len(winning_sides) == 1, "active cutout side must be unique"
            candidates.append((max_hits, core_tail, winning_sides[0]))

    assert candidates, "one black cutout must be horizontally visible from a source"
    max_hits = max(hit_count for hit_count, _core_tail, _side in candidates)
    winners = [candidate for candidate in candidates if candidate[0] == max_hits]
    assert len(winners) == 1, "active cutout must be unique"

    _hit_count, core_tail, side = winners[0]
    r0, c0, r1, c1, tail = core_tail

    out = g.copy()
    for r, c in tail:
        out[r, c] = bg

    mark_col = c0 if side == "left" else c1
    out[r0 : r1 + 1, mark_col] = MARK_COLOR
    return out

# --------------- generator (refactored att1, Claude) ---------------

# Grid and colors.
BG = 7
BLACK = 0
GRID_SIZE = 16
GRID_SHAPE = (GRID_SIZE, GRID_SIZE)
SOURCE_COLOR = 5
DISTRACTOR_COLORS = [1, 2, 3, 4, 8, 9]
AVAILABLE_COLORS = [SOURCE_COLOR] + DISTRACTOR_COLORS

# Staircase geometry and edge placement.
STAIR_HEIGHT_RANGE = (3, 5)
STAIR_MAX_WIDTH = 8
STAIR_WIDTH_EXTRA = 4
EDGE_SIDES = ("left", "right")
LEFT_CORNERS = ("tl", "bl")
RIGHT_CORNERS = ("tr", "br")

# Cutout geometry and line-of-sight requirements.
CORE_SHAPES = ((3, 4), (4, 3))
TAIL_DIRECTIONS = ("top", "bottom", "left", "right")
FOUR_CELL_MIDDLE_OFFSETS = [1, 2]
MIN_SOURCE_HITS = 2
LOS_GAP = 1
OBJECT_PADDING = 1

# Retry counts and optional-object rates.
SECOND_CUTOUT_RETRIES = 50
DISTRACTOR_STAIR_RETRIES = 50
SINGLE_PIXEL_RETRIES = 30
EXTRA_BLACK_RETRIES = 20
GENERATE_RETRIES = 300
SINGLE_PIXEL_COUNT_RANGE = (1, 2)
EXTRA_BLACK_PROB = 0.3


def _random_stair_width(height: int) -> int:
    return random.randint(height, min(STAIR_MAX_WIDTH, height + STAIR_WIDTH_EXTRA))


def _edge_col(side: str, width: int) -> int:
    return 0 if side == "left" else GRID_SIZE - width


def _corner_choices(side: str) -> tuple[str, str]:
    return LEFT_CORNERS if side == "left" else RIGHT_CORNERS


def _make_staircase(height: int, width: int, corner: str, color: int) -> np.ndarray:
    arr = np.full((height, width), BG, dtype=int)
    for i in range(height):
        length = width - i
        row = i if corner in ("tl", "tr") else height - 1 - i
        if corner in ("tl", "bl"):
            arr[row, :length] = color
        else:
            arr[row, width - length :] = color
    return arr


def _paste_non_bg_patch(grid: np.ndarray, patch: np.ndarray, row: int, col: int) -> None:
    mask = patch != BG
    region = grid[row : row + patch.shape[0], col : col + patch.shape[1]]
    region[mask] = patch[mask]


def _can_paste_non_bg_patch(grid: np.ndarray, patch: np.ndarray, row: int, col: int) -> bool:
    mask = patch != BG
    region = grid[row : row + patch.shape[0], col : col + patch.shape[1]]
    return not bool(np.any(region[mask] != BG))


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _cells_in_bounds(cells: list[tuple[int, int]]) -> bool:
    return all(0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE for r, c in cells)


def _cells_are_bg(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    return all(int(grid[r, c]) == BG for r, c in cells)


def _place_source_staircase(grid: np.ndarray) -> tuple[str, int, int, int, int]:
    """Place the gray staircase and return side, height, width, row, and column."""
    height = random.randint(*STAIR_HEIGHT_RANGE)
    width = _random_stair_width(height)
    side = random.choice(EDGE_SIDES)
    row = random.randint(0, GRID_SIZE - 1 - height)
    corner = random.choice(_corner_choices(side))
    col = _edge_col(side, width)
    patch = _make_staircase(height, width, corner, SOURCE_COLOR)
    grid[row : row + height, col : col + width] = patch
    return side, height, width, row, col


def _count_ray_hits(grid: np.ndarray, rows: range, start_col: int, step: int) -> int:
    hits = 0
    for r in rows:
        c = start_col
        while 0 <= c < GRID_SIZE:
            value = int(grid[r, c])
            if value == BG:
                c += step
                continue
            if value == SOURCE_COLOR:
                hits += 1
            break
    return hits


def _count_cutout_side_hits(
    grid: np.ndarray,
    row: int,
    col: int,
    height: int,
    width: int,
    side: str,
) -> int:
    if side == "left":
        return _count_ray_hits(grid, range(row, row + height), col - 1, -1)
    return _count_ray_hits(grid, range(row, row + height), col + width, 1)


def _middle_coord(start: int, size: int) -> int:
    if size == 3:
        return start + 1
    if size == 4:
        return start + random.choice(FOUR_CELL_MIDDLE_OFFSETS)
    return start + size // 2


def _tail_cells(row: int, col: int, height: int, width: int, direction: str) -> list[tuple[int, int]]:
    mid_row = _middle_coord(row, height)
    mid_col = _middle_coord(col, width)
    if direction == "top":
        return [(row - 1, mid_col), (row - 2, mid_col)]
    if direction == "bottom":
        return [(row + height, mid_col), (row + height + 1, mid_col)]
    if direction == "left":
        return [(mid_row, col - 1), (mid_row, col - 2)]
    return [(mid_row, col + width), (mid_row, col + width + 1)]


def _choose_tail_cells(
    grid: np.ndarray,
    row: int,
    col: int,
    height: int,
    width: int,
    blocked_side: str | None = None,
) -> list[tuple[int, int]] | None:
    directions = list(TAIL_DIRECTIONS)
    random.shuffle(directions)
    for direction in directions:
        cells = _tail_cells(row, col, height, width, direction)
        valid = _cells_in_bounds(cells) and _cells_are_bg(grid, cells)
        if direction == blocked_side:
            continue
        if valid:
            return cells
    return None


def _place_winning_cutout(
    grid: np.ndarray,
    stair_side: str,
    stair_height: int,
    stair_width: int,
    stair_row: int,
    stair_col: int,
) -> int | None:
    """Place the source-facing cutout and return its line-of-sight hit count."""
    core_h, core_w = random.choice(CORE_SHAPES)
    r_lo = max(0, stair_row + MIN_SOURCE_HITS - core_h)
    r_hi = min(GRID_SIZE - core_h, stair_row + stair_height - MIN_SOURCE_HITS)
    if r_lo > r_hi:
        return None

    core_r = random.randint(r_lo, r_hi)
    if stair_side == "left":
        min_col = stair_width + LOS_GAP
        max_col = GRID_SIZE - core_w
        if min_col > max_col:
            return None
        core_c = random.randint(min_col, max_col)
    else:
        max_col = stair_col - core_w - LOS_GAP
        if max_col < 0:
            return None
        core_c = random.randint(0, max_col)

    winning_hits = _count_cutout_side_hits(grid, core_r, core_c, core_h, core_w, stair_side)
    if winning_hits < MIN_SOURCE_HITS:
        return None

    grid[core_r : core_r + core_h, core_c : core_c + core_w] = BLACK
    tail = _choose_tail_cells(grid, core_r, core_c, core_h, core_w, blocked_side=stair_side)
    if tail is None:
        return None
    _paint_cells(grid, tail, BLACK)
    return winning_hits


def _rect_is_bg(grid: np.ndarray, row: int, col: int, height: int, width: int) -> bool:
    return bool(np.all(grid[row : row + height, col : col + width] == BG))


def _padded_rect_is_bg(grid: np.ndarray, row: int, col: int, height: int, width: int) -> bool:
    r0 = max(0, row - OBJECT_PADDING)
    r1 = min(GRID_SIZE, row + height + OBJECT_PADDING)
    c0 = max(0, col - OBJECT_PADDING)
    c1 = min(GRID_SIZE, col + width + OBJECT_PADDING)
    return bool(np.all(grid[r0:r1, c0:c1] == BG))


def _place_losing_cutout(grid: np.ndarray, winning_hits: int) -> bool:
    """Place the second cutout where both horizontal sides score below the winner."""
    core_h, core_w = random.choice(CORE_SHAPES)
    for _ in range(SECOND_CUTOUT_RETRIES):
        row = random.randint(0, GRID_SIZE - core_h)
        col = random.randint(0, GRID_SIZE - core_w)
        if not _rect_is_bg(grid, row, col, core_h, core_w):
            continue
        if not _padded_rect_is_bg(grid, row, col, core_h, core_w):
            continue

        left_hits = _count_cutout_side_hits(grid, row, col, core_h, core_w, "left")
        right_hits = _count_cutout_side_hits(grid, row, col, core_h, core_w, "right")
        if max(left_hits, right_hits) >= winning_hits:
            continue

        grid[row : row + core_h, col : col + core_w] = BLACK
        tail = _choose_tail_cells(grid, row, col, core_h, core_w)
        if tail is not None:
            _paint_cells(grid, tail, BLACK)
            return True
        grid[row : row + core_h, col : col + core_w] = BG
    return False


def _place_distractor_staircase(grid: np.ndarray, colors_used: list[int]) -> bool:
    """Place the single non-gray staircase distractor."""
    color = random.choice(DISTRACTOR_COLORS)
    colors_used.append(color)
    for _ in range(DISTRACTOR_STAIR_RETRIES):
        height = random.randint(*STAIR_HEIGHT_RANGE)
        width = _random_stair_width(height)
        side = random.choice(EDGE_SIDES)
        row = random.randint(0, GRID_SIZE - 1 - height)
        corner = random.choice(_corner_choices(side))
        col = _edge_col(side, width)
        patch = _make_staircase(height, width, corner, color)
        if not _can_paste_non_bg_patch(grid, patch, row, col):
            continue
        _paste_non_bg_patch(grid, patch, row, col)
        return True
    return False


def _scatter_single_pixels(grid: np.ndarray, colors_used: list[int]) -> None:
    count = random.randint(*SINGLE_PIXEL_COUNT_RANGE)
    for _ in range(count):
        remaining = [color for color in AVAILABLE_COLORS if color not in colors_used]
        if not remaining:
            break
        color = random.choice(remaining)
        colors_used.append(color)
        for _ in range(SINGLE_PIXEL_RETRIES):
            row = random.randint(0, GRID_SIZE - 1)
            col = random.randint(0, GRID_SIZE - 1)
            if int(grid[row, col]) == BG:
                grid[row, col] = color
                break


def _maybe_place_extra_black(grid: np.ndarray) -> None:
    if random.random() >= EXTRA_BLACK_PROB:
        return
    for _ in range(EXTRA_BLACK_RETRIES):
        row = random.randint(0, GRID_SIZE - 1)
        col = random.randint(0, GRID_SIZE - 1)
        if int(grid[row, col]) == BG:
            grid[row, col] = BLACK
            break


def _oracle_checked_grid(grid: np.ndarray) -> np.ndarray | None:
    try:
        out = main(grid)
    except (AssertionError, Exception):
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    grid = np.full(GRID_SHAPE, BG, dtype=int)
    colors_used = [SOURCE_COLOR]

    stair_side, stair_height, stair_width, stair_row, stair_col = _place_source_staircase(grid)
    winning_hits = _place_winning_cutout(
        grid,
        stair_side,
        stair_height,
        stair_width,
        stair_row,
        stair_col,
    )
    if winning_hits is None:
        return None
    if not _place_losing_cutout(grid, winning_hits):
        return None
    if not _place_distractor_staircase(grid, colors_used):
        return None

    _scatter_single_pixels(grid, colors_used)
    _maybe_place_extra_black(grid)
    return _oracle_checked_grid(grid)


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GENERATE_RETRIES} attempts")
