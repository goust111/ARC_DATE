import random

import numpy as np


BG = 0
RED = 2
GREY = 5
CONTROL = 6
SHADOW = 8

GRID_SIZE_RANGE = (15, 23)
GEN_RETRIES = 400

ROW_PATTERNS = (
    (1, 3, 3),
    (3, 1, 3),
    (3, 3, 1),
)
COL_PATTERNS = ROW_PATTERNS
CORNER_SLOTS = ((0, 0), (0, 2), (2, 0), (2, 2))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    groups = []
    start = prev = indices[0]
    for value in indices[1:]:
        if value == prev + 1:
            prev = value
            continue
        groups.append((start, prev))
        start = prev = value
    groups.append((start, prev))
    return groups


def _band_ranges(start: int, sizes: tuple[int, int, int]) -> list[tuple[int, int]]:
    out = []
    cur = start
    for i, size in enumerate(sizes):
        out.append((cur, cur + size - 1))
        cur += size
        if i < 2:
            cur += 1
    return out


def _center_of(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return (r0 + r1) // 2, (c0 + c1) // 2


def _parse_grid(
    grid: np.ndarray,
) -> tuple[
    tuple[int, int],
    tuple[int, int, int, int],
    list[tuple[int, int]],
    list[tuple[int, int]],
    np.ndarray,
    tuple[int, int],
]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    controls = np.argwhere(g == CONTROL)
    assert controls.shape == (1, 2), "expected exactly one control pixel"
    control = tuple(map(int, controls[0]))

    region_mask = (g == GREY) | (g == RED)
    guide_bbox = _bbox(region_mask)
    assert guide_bbox is not None, "missing target region"
    r0, c0, r1, c1 = guide_bbox

    center = _center_of(guide_bbox)

    active_rows = [r for r in range(r0, r1 + 1) if np.any(region_mask[r, c0 : c1 + 1])]
    active_cols = [c for c in range(c0, c1 + 1) if np.any(region_mask[r0 : r1 + 1, c])]
    row_runs = _runs(active_rows)
    col_runs = _runs(active_cols)
    assert len(row_runs) == 3 and len(col_runs) == 3, "target region must be a 3x3 banded lattice"

    for rr0, rr1 in row_runs:
        for cc0, cc1 in col_runs:
            block = g[rr0 : rr1 + 1, cc0 : cc1 + 1]
            colors = set(map(int, np.unique(block)))
            assert colors in ({GREY}, {RED}), "each lattice cell must be a solid grey/red rectangle"

    red_slots = [
        (ri, ci)
        for ri, (rr0, rr1) in enumerate(row_runs)
        for ci, (cc0, cc1) in enumerate(col_runs)
        if int(g[rr0, cc0]) == RED
    ]
    assert len(red_slots) == 1, "expected exactly one red lattice cell"

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            in_row = any(rr0 <= r <= rr1 for rr0, rr1 in row_runs)
            in_col = any(cc0 <= c <= cc1 for cc0, cc1 in col_runs)
            if in_row and in_col:
                assert g[r, c] in (GREY, RED)
            else:
                assert g[r, c] == BG, "target region gaps must stay black"

    assert not (r0 <= control[0] <= r1 and c0 <= control[1] <= c1), "control must lie outside the region"
    return control, guide_bbox, row_runs, col_runs, region_mask, center


def _relative_index(value: int, center: int) -> int:
    if value < center:
        return 0
    if value > center:
        return 2
    return 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the control pixel's relative direction to choose a lattice cell, then use that cell's center offset to cast a teal shadow of the whole region.

    Input structure:
    - The background is black.
    - One target region is made from nine solid grey/red rectangles arranged as a 3x3 lattice with black gaps.
    - Exactly one rectangle is red, the other eight are grey, and exactly one magenta control pixel sits outside the region.

    Rule:
    - Locate the joint bbox of the grey/red lattice and the magenta control pixel.
    - Compare the control pixel to the target bbox center to get a 3x3 relative-direction slot.
    - Select the lattice rectangle at that slot, compute its center minus the whole-region center, recolor the control pixel red, and copy the entire lattice by that offset in teal.
    - Teal shadow pixels are only painted onto black cells; the original lattice stays unchanged.

    Key insight:
    - The control pixel does not point to a final output location directly; it first selects one lattice subshape, and that subshape's geometric offset becomes the shadow vector for the whole object.
    """
    g = np.asarray(grid, dtype=int)
    control, guide_bbox, row_runs, col_runs, region_mask, region_center = _parse_grid(g)
    pr, pc = control
    row_idx = _relative_index(pr, region_center[0])
    col_idx = _relative_index(pc, region_center[1])

    rr0, rr1 = row_runs[row_idx]
    cc0, cc1 = col_runs[col_idx]
    slot_center = ((rr0 + rr1) // 2, (cc0 + cc1) // 2)
    dr = slot_center[0] - region_center[0]
    dc = slot_center[1] - region_center[1]

    out = g.copy()
    out[pr, pc] = RED
    for r, c in np.argwhere(region_mask):
        rr = int(r) + dr
        cc = int(c) + dc
        if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and out[rr, cc] == BG:
            out[rr, cc] = SHADOW
    return out


def _build_region(
    grid: np.ndarray,
    top: int,
    left: int,
    row_sizes: tuple[int, int, int],
    col_sizes: tuple[int, int, int],
    red_slot: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], np.ndarray]:
    row_runs = _band_ranges(top, row_sizes)
    col_runs = _band_ranges(left, col_sizes)
    mask = np.zeros_like(grid, dtype=bool)
    for ri, (rr0, rr1) in enumerate(row_runs):
        for ci, (cc0, cc1) in enumerate(col_runs):
            color = RED if (ri, ci) == red_slot else GREY
            grid[rr0 : rr1 + 1, cc0 : cc1 + 1] = color
            mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return row_runs, col_runs, mask


def _explicit_output(
    inp: np.ndarray,
    region_mask: np.ndarray,
    control: tuple[int, int],
    offset: tuple[int, int],
) -> np.ndarray:
    out = inp.copy()
    out[control] = RED
    dr, dc = offset
    for r, c in np.argwhere(region_mask):
        rr = int(r) + dr
        cc = int(c) + dc
        if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and out[rr, cc] == BG:
            out[rr, cc] = SHADOW
    return out


def _choose_control(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    shadow_cells: set[tuple[int, int]],
    corner_slot: tuple[int, int],
) -> tuple[int, int] | None:
    h, w = shape
    r0, c0, r1, c1 = bbox
    ri, ci = corner_slot
    if ri == 0:
        rows = range(0, r0)
    else:
        rows = range(r1 + 1, h)
    if ci == 0:
        cols = range(0, c0)
    else:
        cols = range(c1 + 1, w)

    candidates = [
        (r, c)
        for r in rows
        for c in cols
        if (r, c) not in shadow_cells
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    row_sizes = random.choice(ROW_PATTERNS)
    col_sizes = random.choice(COL_PATTERNS)
    selected_slot = random.choice(CORNER_SLOTS)
    red_choices = [slot for slot in [(r, c) for r in range(3) for c in range(3)] if slot != selected_slot]
    red_slot = random.choice(red_choices)

    bbox_h = sum(row_sizes) + 2
    bbox_w = sum(col_sizes) + 2

    local_rows = _band_ranges(0, row_sizes)
    local_cols = _band_ranges(0, col_sizes)
    local_center = _center_of((0, 0, bbox_h - 1, bbox_w - 1))
    sr0, sr1 = local_rows[selected_slot[0]]
    sc0, sc1 = local_cols[selected_slot[1]]
    slot_center = ((sr0 + sr1) // 2, (sc0 + sc1) // 2)
    offset = (slot_center[0] - local_center[0], slot_center[1] - local_center[1])
    dr, dc = offset

    top_min = max(0, -dr)
    top_max = h - bbox_h - max(0, dr)
    left_min = max(0, -dc)
    left_max = w - bbox_w - max(0, dc)
    if top_min > top_max or left_min > left_max:
        return None

    top = random.randint(top_min, top_max)
    left = random.randint(left_min, left_max)

    inp = np.zeros((h, w), dtype=int)
    row_runs, col_runs, region_mask = _build_region(inp, top, left, row_sizes, col_sizes, red_slot)
    bbox = (top, left, top + bbox_h - 1, left + bbox_w - 1)

    shadow_cells = {
        (r + dr, c + dc)
        for r, c in np.argwhere(region_mask)
        if 0 <= r + dr < h and 0 <= c + dc < w and not region_mask[r + dr, c + dc]
    }
    if len(shadow_cells) < 10:
        return None

    control = _choose_control(inp.shape, bbox, shadow_cells, selected_slot)
    if control is None:
        return None
    inp[control] = CONTROL

    parsed_control, parsed_bbox, parsed_rows, parsed_cols, parsed_mask, parsed_center = _parse_grid(inp)
    if parsed_control != control or parsed_bbox != bbox or parsed_rows != row_runs or parsed_cols != col_runs:
        return None

    row_idx = _relative_index(control[0], parsed_center[0])
    col_idx = _relative_index(control[1], parsed_center[1])
    if (row_idx, col_idx) != selected_slot:
        return None

    rr0, rr1 = parsed_rows[row_idx]
    cc0, cc1 = parsed_cols[col_idx]
    parsed_slot_center = ((rr0 + rr1) // 2, (cc0 + cc1) // 2)
    parsed_offset = (parsed_slot_center[0] - parsed_center[0], parsed_slot_center[1] - parsed_center[1])
    if parsed_offset != offset:
        return None

    expected = _explicit_output(inp, parsed_mask, control, offset)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    shadow_mask = actual == SHADOW
    if int(shadow_mask.sum()) != len(shadow_cells):
        return None
    if actual[control] != RED:
        return None
    if np.any(actual[region_mask] == SHADOW):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
