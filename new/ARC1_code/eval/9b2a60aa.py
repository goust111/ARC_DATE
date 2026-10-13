from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg, is_contiguous, paste
import itertools
import random


def _template_mask(component: dict, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in component["coords"]:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _copy_starts(positions: list[int], anchor_index: int, anchor_start: int, sprite_len: int) -> list[int]:
    starts: list[int | None] = [None] * len(positions)
    starts[anchor_index] = int(anchor_start)

    for i in range(anchor_index + 1, len(positions)):
        gap = positions[i] - positions[i - 1] - 1
        assert gap >= 0, "control pixels must be in increasing order without duplicates"
        assert starts[i - 1] is not None
        starts[i] = starts[i - 1] + sprite_len + gap

    for i in range(anchor_index - 1, -1, -1):
        gap = positions[i + 1] - positions[i] - 1
        assert gap >= 0, "control pixels must be in increasing order without duplicates"
        assert starts[i + 1] is not None
        starts[i] = starts[i + 1] - sprite_len - gap

    assert all(start is not None for start in starts)
    return [int(start) for start in starts]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the red template sprite along the control-pixel line, preserving control gaps and recoloring each copy.

    Input structure
    The grid has a black background, one 8-connected red template sprite, and
    several isolated control pixels in one row or column.  Exactly one control
    pixel has the template color.

    Rule
    Read the control pixels left-to-right or top-to-bottom.  The copy matching
    the template-color control is the original sprite; each neighboring control
    becomes a same-shape copy in that control color.  The empty cells between
    adjacent control pixels become the empty columns or rows between adjacent
    sprite bboxes.

    Key insight
    The singleton control line is a compressed ordered layout: expanding each
    marker to the template bbox while keeping the inter-marker gaps fixes every
    copy position from the original sprite.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    components = find_components(g, bg=bg, connectivity=8, monochromatic=True)
    sprite_components = [component for component in components if component["area"] > 1]
    assert len(sprite_components) == 1, "expected one non-singleton template sprite"

    sprite = sprite_components[0]
    template_color = int(sprite["color"])
    sr0, sc0, sr1, sc1 = sprite["bbox"]
    template = _template_mask(sprite, sprite["bbox"])
    sprite_h, sprite_w = template.shape

    controls: list[tuple[int, int, int]] = []
    for component in components:
        if component is sprite:
            continue
        assert component["area"] == 1, "all non-template components must be singleton controls"
        r, c = map(int, component["coords"][0])
        controls.append((r, c, int(component["color"])))

    assert len(controls) >= 2, "expected an ordered control line"
    anchor_indices = [i for i, (_, _, color) in enumerate(controls) if color == template_color]
    assert len(anchor_indices) == 1, "expected one control pixel with the template color"

    control_rows = {r for r, _, _ in controls}
    control_cols = {c for _, c, _ in controls}
    if len(control_rows) == 1:
        controls.sort(key=lambda item: item[1])
        positions = [c for _, c, _ in controls]
        anchor_index = next(i for i, (_, _, color) in enumerate(controls) if color == template_color)
        starts = _copy_starts(positions, anchor_index, sc0, sprite_w)
        placements = [(sr0, start, color) for start, (_, _, color) in zip(starts, controls)]
    elif len(control_cols) == 1:
        controls.sort(key=lambda item: item[0])
        positions = [r for r, _, _ in controls]
        anchor_index = next(i for i, (_, _, color) in enumerate(controls) if color == template_color)
        starts = _copy_starts(positions, anchor_index, sr0, sprite_h)
        placements = [(start, sc0, color) for start, (_, _, color) in zip(starts, controls)]
    else:
        raise AssertionError("control pixels must share one row or one column")

    out = g.copy()
    H, W = g.shape
    for r0, c0, color in placements:
        assert 0 <= r0 and 0 <= c0 and r0 + sprite_h <= H and c0 + sprite_w <= W
        target = out[r0 : r0 + sprite_h, c0 : c0 + sprite_w]
        occupied = target[template]
        if color == template_color:
            assert np.all(occupied == template_color), "template-color copy must be the original sprite"
        else:
            assert np.all(occupied == bg), "new copies must land on background"
        target[template] = color

    return out

# --------------- generator (o5x) ---------------

# Canvas and axis bounds.
GRID_HEIGHT_BOUNDS = (13, 23)
GRID_WIDTH_BOUNDS = (14, 24)
HORIZONTAL = "horizontal"
VERTICAL = "vertical"
AXES = [HORIZONTAL, VERTICAL]

# Sprite geometry and colors.
SPRITE_SIZE = 3
SPRITE_AREAS = [6, 7]
BACKGROUND_COLOR = 0
SPRITE_COLOR = 2

# Control-line layout and palette.
CONTROL_COUNTS = [4, 5]
CONTROL_GAP_BOUNDS = (1, 4)
CONTROL_EDGE_COORDS = [1, 2]
CONTROL_TO_SPRITE_MIN_OFFSET = 5
CONTROL_PALETTE = [1, 3, 4, 5, 6, 7, 8, 9]
CONTROL_DISTINCT_BOUNDS = (2, 4)
MIN_COPY_SPAN_FRACTION = 0.75

# Retry limits.
GENERATION_RETRIES = 300

def _gen_sprite_mask() -> np.ndarray:
    area = random.choice(SPRITE_AREAS)
    masks: list[np.ndarray] = []
    for cells in itertools.combinations(range(SPRITE_SIZE * SPRITE_SIZE), area):
        mask = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
        for cell in cells:
            mask[cell // SPRITE_SIZE, cell % SPRITE_SIZE] = True
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if is_contiguous(mask.astype(int), background=BACKGROUND_COLOR, connectivity=8):
            masks.append(mask)
    return random.choice(masks)


def _gen_gaps(count: int, max_axis_len: int) -> list[int]:
    gap_min, gap_max = CONTROL_GAP_BOUNDS
    choices = [
        list(gaps)
        for gaps in itertools.product(range(gap_min, gap_max + 1), repeat=count - 1)
        if count * SPRITE_SIZE + sum(gaps) <= max_axis_len
    ]
    return random.choice(choices)


def _gen_control_positions(axis_len: int, gaps: list[int]) -> list[int] | None:
    control_span = len(gaps) + 1 + sum(gaps)
    if control_span > axis_len:
        return None
    if control_span <= axis_len - 2:
        start = random.randint(1, axis_len - control_span - 1)
    else:
        start = random.randint(0, axis_len - control_span)
    positions = [start]
    for gap in gaps:
        positions.append(positions[-1] + gap + 1)
    return positions


def _gen_control_colors(count: int, anchor_index: int) -> list[int]:
    distinct_min, distinct_max = CONTROL_DISTINCT_BOUNDS
    distinct_count = random.randint(distinct_min, min(distinct_max, count))
    selected = random.sample(CONTROL_PALETTE, distinct_count - 1)
    other_count = count - 1
    others = selected[:]
    while len(others) < other_count:
        others.append(random.choice(selected))
    random.shuffle(others)

    colors: list[int] = []
    other_i = 0
    for i in range(count):
        if i == anchor_index:
            colors.append(SPRITE_COLOR)
        else:
            colors.append(others[other_i])
            other_i += 1
    return colors


def _gen_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=int)
    patch[mask] = int(color)
    return patch


def _gen_copy_starts(axis_len: int, copy_span: int, gaps: list[int]) -> list[int]:
    starts = [random.randint(0, axis_len - copy_span)]
    for gap in gaps:
        starts.append(starts[-1] + SPRITE_SIZE + gap)
    return starts


def _origin(axis: str, start: int, fixed_coord: int) -> tuple[int, int]:
    return (fixed_coord, start) if axis == HORIZONTAL else (start, fixed_coord)


def _render_candidate(axis, height, width, mask, starts, control_positions, colors, anchor_index):
    out = np.zeros((height, width), dtype=int)
    fixed_limit = height if axis == HORIZONTAL else width
    control_fixed = random.choice(CONTROL_EDGE_COORDS)
    min_sprite_fixed = control_fixed + CONTROL_TO_SPRITE_MIN_OFFSET
    if min_sprite_fixed > fixed_limit - SPRITE_SIZE:
        return None
    sprite_fixed = random.randint(min_sprite_fixed, fixed_limit - SPRITE_SIZE)

    for start, color in zip(starts, colors):
        r, c = _origin(axis, start, sprite_fixed)
        paste(out, _gen_patch(mask, color), r, c, transparent=BACKGROUND_COLOR)

    for position, color in zip(control_positions, colors):
        r, c = _origin(axis, position, control_fixed)
        out[r, c] = color

    inp = out.copy()
    for i, start in enumerate(starts):
        if i == anchor_index:
            continue
        r, c = _origin(axis, start, sprite_fixed)
        inp[r : r + SPRITE_SIZE, c : c + SPRITE_SIZE][mask] = BACKGROUND_COLOR
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    axis = random.choice(AXES)
    count = random.choice(CONTROL_COUNTS)
    max_axis_len = GRID_WIDTH_BOUNDS[1] if axis == HORIZONTAL else GRID_HEIGHT_BOUNDS[1]
    gaps = _gen_gaps(count, max_axis_len)
    copy_span = count * SPRITE_SIZE + sum(gaps)
    max_len_for_span = min(max_axis_len, int(copy_span / MIN_COPY_SPAN_FRACTION))
    axis_len = random.randint(copy_span, max_len_for_span)
    if axis == HORIZONTAL:
        height, width = random.randint(*GRID_HEIGHT_BOUNDS), axis_len
    else:
        height, width = axis_len, random.randint(*GRID_WIDTH_BOUNDS)

    mask = _gen_sprite_mask()
    anchor_index = random.randrange(count)
    colors = _gen_control_colors(count, anchor_index)
    starts = _gen_copy_starts(axis_len, copy_span, gaps)
    control_positions = _gen_control_positions(axis_len, gaps)
    if control_positions is None:
        return None

    rendered = _render_candidate(axis, height, width, mask, starts, control_positions, colors, anchor_index)
    if rendered is None:
        return None
    inp, out = rendered
    try:
        solved = main(inp)
    except Exception:
        return None
    return inp if np.array_equal(solved, out) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 9b2a60aa input")
