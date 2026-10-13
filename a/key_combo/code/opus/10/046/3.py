import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

MAX_GENERATION_ATTEMPTS = 400
_ROTATION_BAG: list[int] = []


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rect_cycle(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        coords.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return coords


def _is_full_rectangle_border(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    border = np.zeros_like(mask, dtype=bool)
    border[r0, c0 : c1 + 1] = True
    border[r1, c0 : c1 + 1] = True
    border[r0 : r1 + 1, c0] = True
    border[r0 : r1 + 1, c1] = True
    return np.array_equal(mask, border)


def _blue_run_start(cycle_values: list[int]) -> int:
    starts = []
    for i, value in enumerate(cycle_values):
        if value == BLUE and cycle_values[i - 1] != BLUE:
            starts.append(i)
    assert len(starts) == 1, "expected exactly one blue run"
    return starts[0]


def _crop_nonblack(patch: np.ndarray) -> np.ndarray:
    bbox = _bbox(patch != BLACK)
    assert bbox is not None, "expected a non-black patch"
    return _crop_bbox(patch, bbox)


def _shape_key(patch: np.ndarray) -> tuple[tuple[int, int], ...]:
    cropped = _crop_nonblack(patch)
    coords = np.argwhere(cropped == RED)
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _rotate_template(patch: np.ndarray, k: int) -> np.ndarray:
    rotated = np.rot90(patch, k=-int(k))
    return _crop_nonblack(rotated)


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _random_connected_shape(height: int, width: int, area: int) -> np.ndarray | None:
    for _ in range(200):
        cells = {(random.randrange(height), random.randrange(width))}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue

        patch = np.zeros((height, width), dtype=int)
        for r, c in cells:
            patch[r, c] = RED
        patch = _crop_nonblack(patch)
        red_mask = patch == RED
        if len(_components4(red_mask)) != 1:
            continue
        if min(patch.shape) < 2:
            continue
        if len({_shape_key(np.rot90(patch, k=-k)) for k in range(4)}) != 4:
            continue
        return patch
    return None


def _generate_template(inner_h: int, inner_w: int) -> np.ndarray | None:
    max_h = min(4, inner_h - 2)
    max_w = min(4, inner_w - 2)
    if max_h < 2 or max_w < 2:
        return None

    for _ in range(200):
        patch_h = random.randint(2, max_h)
        patch_w = random.randint(2, max_w)
        max_area = min(6, patch_h * patch_w)
        if max_area < 4:
            continue
        area = random.randint(4, max_area)
        patch = _random_connected_shape(patch_h, patch_w, area)
        if patch is not None:
            return patch
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Decode the outer border segment and rotate the inner red template accordingly.

    Input structure:
    - The grid has a black background, one complete grey 1-pixel-thick rectangular frame, and a red 4-connected template strictly inside that frame.
    - The pixels immediately outside the grey frame form a rectangular perimeter path; exactly one contiguous quarter of that path is blue and the rest is black.
    - The blue quarter starts at one of the four quarter-phase offsets of the perimeter cycle when read clockwise from the outer path's top-left corner.

    Rule:
    - Find the grey frame and unwrap the adjacent outer perimeter clockwise starting from its top-left corner.
    - Read the start index of the blue run, divide it by one quarter of the perimeter length, and decode a rotation count k in {0,1,2,3}.
    - Crop the red template, rotate it clockwise by k * 90 degrees, and output only that rotated template on black, cropped to its own bounding box.

    Key insight:
    - The blue border segment is a cyclic position code rather than decoration: its phase on the outer ring specifies the rotation to apply to the enclosed template.
    """
    g = np.asarray(grid, dtype=int)

    grey_mask = g == GREY
    frame_bbox = _bbox(grey_mask)
    assert frame_bbox is not None, "expected a grey frame"
    assert _is_full_rectangle_border(grey_mask, frame_bbox), "grey pixels must form one full rectangular border"

    fr0, fc0, fr1, fc1 = frame_bbox
    assert fr0 >= 1 and fc0 >= 1 and fr1 + 1 < g.shape[0] and fc1 + 1 < g.shape[1], "frame must have an outer ring"

    outer_cycle = _rect_cycle(fr0 - 1, fc0 - 1, fr1 + 1, fc1 + 1)
    cycle_values = [int(g[r, c]) for r, c in outer_cycle]
    cycle_len = len(outer_cycle)
    assert cycle_len % 4 == 0, "outer ring perimeter must be divisible by 4"
    assert all(value in (BLACK, BLUE) for value in cycle_values), "outer ring may only contain black and blue"

    quarter = cycle_len // 4
    blue_count = sum(value == BLUE for value in cycle_values)
    assert blue_count == quarter, "blue run must occupy exactly one quarter of the perimeter"

    start = _blue_run_start(cycle_values)
    assert start % quarter == 0, "blue run must start on a quarter boundary"
    k = start // quarter
    assert 0 <= k <= 3

    red_mask = g == RED
    red_bbox = _bbox(red_mask)
    assert red_bbox is not None, "expected a red template"
    assert fr0 < red_bbox[0] and red_bbox[2] < fr1 and fc0 < red_bbox[1] and red_bbox[3] < fc1, "template must be strictly inside the frame"
    assert len(_components4(red_mask)) == 1, "template must be 4-connected"

    template = _crop_bbox(g, red_bbox)
    return _rotate_template(template, k)


def _expected_output_from_parts(template: np.ndarray, k: int) -> np.ndarray:
    return _rotate_template(template, k)


def _sample_rotation() -> int:
    global _ROTATION_BAG
    if not _ROTATION_BAG:
        _ROTATION_BAG = [0, 1, 2, 3]
        random.shuffle(_ROTATION_BAG)
    return _ROTATION_BAG.pop()


def _build_instance() -> tuple[np.ndarray, np.ndarray]:
    outer_choices = [(9, 11), (11, 9), (9, 9), (10, 10), (11, 11)]
    outer_h, outer_w = random.choice(outer_choices)
    inner_h, inner_w = outer_h - 4, outer_w - 4

    template = _generate_template(inner_h, inner_w)
    assert template is not None, "failed to sample a non-degenerate template"
    th, tw = template.shape

    vertical_margin = random.randint(2, 4)
    horizontal_margin = random.randint(2, 4)
    top = vertical_margin // 2
    bottom = vertical_margin - top
    left = horizontal_margin // 2
    right = horizontal_margin - left

    grid = np.zeros((outer_h + top + bottom, outer_w + left + right), dtype=int)

    or0, oc0 = top, left
    or1, oc1 = or0 + outer_h - 1, oc0 + outer_w - 1
    fr0, fc0 = or0 + 1, oc0 + 1
    fr1, fc1 = or1 - 1, oc1 - 1

    grid[fr0, fc0 : fc1 + 1] = GREY
    grid[fr1, fc0 : fc1 + 1] = GREY
    grid[fr0 : fr1 + 1, fc0] = GREY
    grid[fr0 : fr1 + 1, fc1] = GREY

    inner_r0, inner_c0 = fr0 + 1, fc0 + 1
    inner_r1, inner_c1 = fr1 - 1, fc1 - 1
    row_room = inner_h - th - 2
    col_room = inner_w - tw - 2
    assert row_room >= 0 and col_room >= 0
    place_r = inner_r0 + 1 + random.randint(0, row_room)
    place_c = inner_c0 + 1 + random.randint(0, col_room)
    patch_mask = template == RED
    grid[place_r : place_r + th, place_c : place_c + tw][patch_mask] = RED

    cycle = _rect_cycle(or0, oc0, or1, oc1)
    cycle_len = len(cycle)
    assert cycle_len % 4 == 0
    quarter = cycle_len // 4
    k = _sample_rotation()
    start = k * quarter
    for offset in range(quarter):
        r, c = cycle[start + offset]
        grid[r, c] = BLUE

    expected = _expected_output_from_parts(template, k)
    return grid, expected


def _validate_generated_instance(grid: np.ndarray, expected: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert isinstance(expected, np.ndarray) and expected.ndim == 2
    assert max(grid.shape) <= 30 and max(expected.shape) <= 30

    grey_mask = grid == GREY
    red_mask = grid == RED
    blue_mask = grid == BLUE

    assert _bbox(grey_mask) is not None
    assert len(_components4(red_mask)) == 1
    assert len(_components4(blue_mask)) == 1

    frame_bbox = _bbox(grey_mask)
    assert frame_bbox is not None
    assert _is_full_rectangle_border(grey_mask, frame_bbox)

    fr0, fc0, fr1, fc1 = frame_bbox
    outer_cycle = _rect_cycle(fr0 - 1, fc0 - 1, fr1 + 1, fc1 + 1)
    values = [int(grid[r, c]) for r, c in outer_cycle]
    quarter = len(outer_cycle) // 4
    start = _blue_run_start(values)
    assert sum(v == BLUE for v in values) == quarter
    assert start % quarter == 0

    solver_out = main(grid)
    assert np.array_equal(solver_out, expected), "solver output must match the intended rotated template"


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid, expected = _build_instance()
        try:
            _validate_generated_instance(grid, expected)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid ring-rotation ARC input")
