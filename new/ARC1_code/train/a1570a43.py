from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, bbox_h, bbox_w, colors_present, find_components, get_bg, paste
import random


def _corner_marker_bbox(
    g: np.ndarray, colors: list[int]
) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in colors:
        coords = np.argwhere(g == int(color))
        if int(coords.shape[0]) != 4:
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        actual = {tuple(map(int, rc)) for rc in coords}
        if actual == corners:
            candidates.append((int(color), (r0, c0, r1, c1)))

    assert len(candidates) == 1, "expected a unique four-corner marker rectangle"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Move the foreground object so its bounding box fits exactly inside the four corner markers.

    Input structure
    The grid has a dominant background, one foreground color forming exactly the
    four corners of a rectangle, and one 4-connected foreground object whose
    bounding box is the same size as that rectangle's interior.

    Rule
    Clear the moving object from its original position, then translate every one
    of its pixels so the object's bbox upper-left corner becomes the upper-left
    cell inside the marker rectangle.

    Key insight
    The corner markers are not copied or filled; they define a target interior
    window for the unchanged object mask.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected one marker color and one moving-object color"

    marker_color, marker_bbox = _corner_marker_bbox(g, colors)
    moving_colors = [color for color in colors if color != marker_color]
    assert len(moving_colors) == 1, "expected a unique moving-object color"
    moving_color = moving_colors[0]

    components = find_components(g, bg=bg, colors=[moving_color], connectivity=4)
    assert len(components) == 1, "expected one 4-connected moving object"
    source_bbox = components[0]["bbox"]
    assert int(components[0]["area"]) == int(np.sum(g == moving_color))

    fr0, fc0, fr1, fc1 = marker_bbox
    target_bbox = (fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1)
    assert bbox_h(source_bbox) == bbox_h(target_bbox)
    assert bbox_w(source_bbox) == bbox_w(target_bbox)

    sr0, sc0, _sr1, _sc1 = source_bbox
    dr = int(target_bbox[0] - sr0)
    dc = int(target_bbox[1] - sc0)

    out = g.copy()
    out[g == moving_color] = bg
    for r_raw, c_raw in np.argwhere(g == moving_color):
        r = int(r_raw + dr)
        c = int(c_raw + dc)
        assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1], "translated pixel must stay in bounds"
        assert int(out[r, c]) == bg, "translated object must land on empty cells"
        out[r, c] = moving_color

    return out

# --------------- generator (o5x) ---------------

# Canvas and frame geometry.
GRID_MIN, GRID_MAX = 7, 10
FRAME_SIZE, FRAME_MARGIN_MAX, SPRITE_SIZE = 7, 2, 5

# Fixed task colors.
BACKGROUND_COLOR, MARKER_COLOR, OBJECT_COLOR = 0, 3, 2

# Sprite shape sampling.
OBJECT_AREA_MIN, OBJECT_AREA_MAX = 11, 13
SPRITE_GROW_RETRIES = 30

# Source placement.
OFFSET_MIN_L1, OFFSET_MAX_L1 = 1, 2

# End-to-end generation retries.
GENERATION_RETRIES = 300


def _frame_origins_for_axis(axis_len: int) -> list[int]:
    return [pos for pos in range(axis_len - FRAME_SIZE + 1) if pos <= FRAME_MARGIN_MAX and axis_len - FRAME_SIZE - pos <= FRAME_MARGIN_MAX]


def _frame_corner_cells(frame_r: int, frame_c: int) -> set[tuple[int, int]]:
    last = FRAME_SIZE - 1
    return {(frame_r, frame_c), (frame_r, frame_c + last), (frame_r + last, frame_c), (frame_r + last, frame_c + last)}


def _is_asymmetric(mask: np.ndarray) -> bool:
    variants = (np.fliplr(mask), np.flipud(mask), np.rot90(mask, 2))
    return not any(np.array_equal(mask, variant) for variant in variants)


def _gen_path_sprite_mask() -> np.ndarray:
    mask = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
    start_c = random.randrange(SPRITE_SIZE)
    end_c = random.randrange(SPRITE_SIZE)
    dc = 1 if end_c > start_c else -1
    moves = [(1, 0)] * (SPRITE_SIZE - 1) + [(0, dc)] * abs(end_c - start_c)
    random.shuffle(moves)

    r, c = 0, start_c
    mask[r, c] = True
    for dr, step_c in moves:
        r += dr
        c += step_c
        mask[r, c] = True

    coords = np.argwhere(mask)
    min_c = int(coords[:, 1].min())
    max_c = int(coords[:, 1].max())

    left_roots = coords[coords[:, 1] == min_c]
    lr, _lc = map(int, random.choice(left_roots))
    mask[lr, : min_c + 1] = True

    right_roots = coords[coords[:, 1] == max_c]
    rr, _rc = map(int, random.choice(right_roots))
    mask[rr, max_c:] = True

    if random.choice([False, True]):
        mask = mask.T.copy()
    return mask


def _frontier_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    frontier = []
    last = SPRITE_SIZE - 1
    for r in range(SPRITE_SIZE):
        for c in range(SPRITE_SIZE):
            if mask[r, c]:
                continue
            if (
                (r > 0 and mask[r - 1, c])
                or (r < last and mask[r + 1, c])
                or (c > 0 and mask[r, c - 1])
                or (c < last and mask[r, c + 1])
            ):
                frontier.append((r, c))
    return frontier


def _gen_grow_sprite_mask(area: int) -> np.ndarray:
    for _ in range(SPRITE_GROW_RETRIES):
        mask = _gen_path_sprite_mask()
        while int(mask.sum()) < int(area):
            frontier = _frontier_cells(mask)
            if not frontier:
                break
            grow_r, grow_c = frontier[random.randrange(len(frontier))]
            mask[grow_r, grow_c] = True
        if int(mask.sum()) == int(area) and _is_asymmetric(mask):
            return mask

    return mask


def _gen_valid_offsets(
    mask: np.ndarray, shape: tuple[int, int], target_r: int, target_c: int, marker_cells: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    H, W = shape
    offsets = []
    mask_cells = [(int(rr), int(cc)) for rr, cc in np.argwhere(mask)]
    for dr in range(-OFFSET_MAX_L1, OFFSET_MAX_L1 + 1):
        for dc in range(-OFFSET_MAX_L1, OFFSET_MAX_L1 + 1):
            if not (OFFSET_MIN_L1 <= abs(dr) + abs(dc) <= OFFSET_MAX_L1):
                continue
            source_r = target_r + dr
            source_c = target_c + dc
            if source_r < 0 or source_c < 0 or source_r + SPRITE_SIZE > H or source_c + SPRITE_SIZE > W:
                continue
            if not any((source_r + rr, source_c + cc) in marker_cells for rr, cc in mask_cells):
                offsets.append((dr, dc))
    random.shuffle(offsets)
    return offsets


def _render_output(shape: tuple[int, int], marker_cells: set[tuple[int, int]], sprite: np.ndarray, target_r: int, target_c: int) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c in marker_cells:
        out[r, c] = MARKER_COLOR
    paste(out, sprite, target_r, target_c, transparent=BACKGROUND_COLOR)
    return out


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(GRID_MIN, GRID_MAX)
    W = random.randint(GRID_MIN, GRID_MAX)
    frame_r = random.choice(_frame_origins_for_axis(H))
    frame_c = random.choice(_frame_origins_for_axis(W))

    target_r = frame_r + 1
    target_c = frame_c + 1
    marker_cells = _frame_corner_cells(frame_r, frame_c)

    area = random.randint(OBJECT_AREA_MIN, OBJECT_AREA_MAX)
    mask = _gen_grow_sprite_mask(area)
    sprite = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=int)
    sprite[mask] = OBJECT_COLOR

    out = _render_output((H, W), marker_cells, sprite, target_r, target_c)

    offsets = _gen_valid_offsets(mask, (H, W), target_r, target_c, marker_cells)
    if not offsets:
        return None
    dr, dc = offsets[0]

    inp = out.copy()
    inp[inp == OBJECT_COLOR] = BACKGROUND_COLOR
    paste(inp, sprite, target_r + dr, target_c + dc, transparent=BACKGROUND_COLOR)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for a1570a43")
