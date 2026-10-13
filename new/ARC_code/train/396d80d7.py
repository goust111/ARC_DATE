from __future__ import annotations

import numpy as np

from arc_utils import bbox, colors_present, components_mask, get_bg, in_bounds
import random


DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
ORTHOGONALS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the inner foreground color into every background pixel that touches the enclosing frame only diagonally.

    Input structure:
    - The grid has one dominant background color and exactly two foreground colors.
    - One foreground color is an enclosing frame: its bbox strictly contains the bbox of the other, inner color.

    Rule:
    - Keep all existing pixels unchanged.
    - For each background pixel, if it is diagonally adjacent to the frame color and has no orthogonal adjacency to the frame color, recolor it to the inner color.

    Key insight: The inner color marks the color to use, while the frame's diagonal-only background neighbors define the cells to add.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2, f"expected exactly two foreground colors, got {colors}"

    boxes: dict[int, tuple[int, int, int, int]] = {}
    for color in colors:
        b = bbox(g == int(color))
        assert b is not None, f"foreground color {color} has no bbox"
        boxes[int(color)] = b

    role_pairs = [
        (outer, inner)
        for outer in colors
        for inner in colors
        if outer != inner and _strictly_contains(boxes[int(outer)], boxes[int(inner)])
    ]
    assert len(role_pairs) == 1, f"expected one enclosing frame color, got {role_pairs}"
    frame_color, inner_color = map(int, role_pairs[0])

    assert len(components_mask(g != bg, connectivity=8)) == 1, "all foreground must be one 8-connected component"

    out = g.copy()
    H, W = g.shape
    for r in range(H):
        for c in range(W):
            if int(g[r, c]) != bg:
                continue

            has_diagonal_frame = any(
                in_bounds(g.shape, r + dr, c + dc) and int(g[r + dr, c + dc]) == frame_color
                for dr, dc in DIAGONALS
            )
            if not has_diagonal_frame:
                continue

            has_orthogonal_frame = any(
                in_bounds(g.shape, r + dr, c + dc) and int(g[r + dr, c + dc]) == frame_color
                for dr, dc in ORTHOGONALS
            )
            if not has_orthogonal_frame:
                has_orthogonal_inner = any(
                    in_bounds(g.shape, r + dr, c + dc) and int(g[r + dr, c + dc]) == inner_color
                    for dr, dc in ORTHOGONALS
                )
                assert not has_orthogonal_inner, "new pixel must not be orthogonally adjacent to inner color"
                out[r, c] = inner_color

    return out

# --------------- generator (att1, Claude) ---------------

# Canvas and palette.
BG = 7
GRID_H, GRID_W = 16, 16
FOREGROUND_COLORS = [c for c in range(10) if c != BG]

# Local rendering colors and fill-rule geometry.
LOCAL_FRAME_COLOR, LOCAL_INNER_COLOR = 1, 2
PAD = 2
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHOGONALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Frame geometry and accepted output-fill size.
FRAME_H_RANGE = (4, 7)
FRAME_W_RANGE = (4, 7)
FRAME_PIXELS_MIN, FRAME_PIXELS_MAX = 4, 21
FRAME_DENSITY_MIN, FRAME_DENSITY_MAX = 0.25, 0.50
FILL_COUNT_MIN, FILL_COUNT_MAX = 12, 15

# Inner cluster choices.
INNER_COUNTS = [2, 3, 4]
INNER_SHAPES_BY_COUNT = {
    2: [[(0, 0), (0, 1)], [(0, 0), (1, 0)], [(0, 0), (1, 1)], [(0, 1), (1, 0)]],
    3: [[(0, 0), (1, 0), (2, 0)], [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (0, 1), (1, 0)], [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (1, 0), (1, 1)], [(0, 1), (1, 0), (1, 1)]],
    4: [[(0, 0), (0, 1), (1, 0), (1, 1)]],
}

# Placement margins and retry budget.
INPUT_MARGIN, OUTPUT_MARGIN = 2, 1
MAX_ATTEMPTS = 300


def _neighbor_mask(mask, offsets):
    H, W = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for dr, dc in offsets:
        dst_r0 = max(0, -dr)
        dst_r1 = min(H, H - dr)
        dst_c0 = max(0, -dc)
        dst_c1 = min(W, W - dc)
        out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[
            dst_r0 + dr:dst_r1 + dr,
            dst_c0 + dc:dst_c1 + dc,
        ]
    return out


def _fill_mask(local, frame_color):
    frame = local == frame_color
    return (local == BG) & _neighbor_mask(frame, DIAGONALS) & ~_neighbor_mask(frame, ORTHOGONALS)


def _count_fill(local, frame_color):
    return int(np.count_nonzero(_fill_mask(local, frame_color)))


def _build_padded(frame_coords, inner_coords, h, w):
    local = np.full((h + 2 * PAD, w + 2 * PAD), BG, dtype=int)
    for r, c in frame_coords:
        local[PAD + r, PAD + c] = LOCAL_FRAME_COLOR
    for r, c in inner_coords:
        local[PAD + r, PAD + c] = LOCAL_INNER_COLOR
    return local


def _edges_covered(coords, h, w):
    rows = {r for r, c in coords}
    cols = {c for r, c in coords}
    return 0 in rows and (h - 1) in rows and 0 in cols and (w - 1) in cols


def _choose_inner(frame_h, frame_w):
    n_inner = random.choice(INNER_COUNTS)
    inner_local = random.choice(INNER_SHAPES_BY_COUNT[n_inner])
    inner_h = max(r for r, _ in inner_local) + 1
    inner_w = max(c for _, c in inner_local) + 1

    max_ir = frame_h - 1 - inner_h
    max_ic = frame_w - 1 - inner_w
    if max_ir < 1 or max_ic < 1:
        return None

    ir = random.randint(1, max_ir)
    ic = random.randint(1, max_ic)
    return [(ir + r, ic + c) for r, c in inner_local]


def _without_index(coords, idx):
    return [q for j, q in enumerate(coords) if j != idx]


def _removable_edge_candidates(frame_coords, frame_h, frame_w):
    return [i for i in range(len(frame_coords)) if _edges_covered(_without_index(frame_coords, i), frame_h, frame_w)]


def _thin_frame(frame_coords, inner_coords, frame_h, frame_w, full_new):
    min_target = max(FRAME_PIXELS_MIN, int(FRAME_DENSITY_MIN * frame_h * frame_w))
    max_target = len(frame_coords)
    if min_target > max_target:
        return None

    target_px = random.randint(min_target, max_target)
    if len(frame_coords) <= target_px and full_new <= FILL_COUNT_MAX:
        return frame_coords

    random.shuffle(frame_coords)
    for _ in range(len(frame_coords)):
        if len(frame_coords) <= max(FRAME_PIXELS_MIN, target_px):
            break

        candidates = _removable_edge_candidates(frame_coords, frame_h, frame_w)
        if not candidates:
            break

        idx = random.choice(candidates)
        trial = _without_index(frame_coords, idx)
        trial_padded = _build_padded(trial, inner_coords, frame_h, frame_w)
        if _count_fill(trial_padded, LOCAL_FRAME_COLOR) >= FILL_COUNT_MIN:
            frame_coords = trial

    return frame_coords


def _build_frame(frame_h, frame_w, inner_coords):
    inner_set = set(inner_coords)
    offset = random.choice([0, 1])
    full_checker = [
        (r, c)
        for r in range(frame_h)
        for c in range(frame_w)
        if (r + c) % 2 == offset and (r, c) not in inner_set
    ]

    if len(full_checker) < FRAME_PIXELS_MIN:
        return None

    padded = _build_padded(full_checker, inner_coords, frame_h, frame_w)
    full_new = _count_fill(padded, LOCAL_FRAME_COLOR)
    if full_new < FILL_COUNT_MIN:
        return None

    return _thin_frame(list(full_checker), inner_coords, frame_h, frame_w, full_new)


def _validated_padded(frame_coords, inner_coords, frame_h, frame_w):
    frame_set = set(frame_coords)
    if any(c in frame_set for c in inner_coords):
        return None

    n_frame = len(frame_coords)
    if not (FRAME_PIXELS_MIN <= n_frame <= FRAME_PIXELS_MAX):
        return None

    density = n_frame / (frame_h * frame_w)
    if not (FRAME_DENSITY_MIN <= density <= FRAME_DENSITY_MAX):
        return None

    if not _edges_covered(frame_coords, frame_h, frame_w):
        return None

    padded = _build_padded(frame_coords, inner_coords, frame_h, frame_w)
    new_count = _count_fill(padded, LOCAL_FRAME_COLOR)
    if not (FILL_COUNT_MIN <= new_count <= FILL_COUNT_MAX):
        return None

    return padded


def _apply_local_fill(padded):
    out_padded = padded.copy()
    out_padded[_fill_mask(padded, LOCAL_FRAME_COLOR)] = LOCAL_INNER_COLOR
    return out_padded


def _placement_bounds(padded, frame_h, frame_w):
    out_box = bbox(_apply_local_fill(padded) != BG)
    if out_box is None:
        return None

    out_r0, out_c0, out_r1, out_c1 = out_box
    min_pr = max(0, OUTPUT_MARGIN - out_r0, INPUT_MARGIN - PAD)
    min_pc = max(0, OUTPUT_MARGIN - out_c0, INPUT_MARGIN - PAD)
    max_pr = min(GRID_H - OUTPUT_MARGIN - 1 - out_r1, GRID_H - INPUT_MARGIN - PAD - frame_h)
    max_pc = min(GRID_W - OUTPUT_MARGIN - 1 - out_c1, GRID_W - INPUT_MARGIN - PAD - frame_w)

    if min_pr > max_pr or min_pc > max_pc:
        return None

    return min_pr, max_pr, min_pc, max_pc


def _place_on_grid(frame_coords, inner_coords, frame_h, frame_w, padded, frame_color, inner_color):
    bounds = _placement_bounds(padded, frame_h, frame_w)
    if bounds is None:
        return None

    min_pr, max_pr, min_pc, max_pc = bounds
    pr = random.randint(min_pr, max_pr)
    pc = random.randint(min_pc, max_pc)

    grid = np.full((GRID_H, GRID_W), BG, dtype=int)
    for r, c in frame_coords:
        grid[pr + PAD + r, pc + PAD + c] = frame_color
    for r, c in inner_coords:
        grid[pr + PAD + r, pc + PAD + c] = inner_color
    return grid


def _gen_attempt():
    frame_color, inner_color = random.sample(FOREGROUND_COLORS, 2)
    frame_h = random.randint(*FRAME_H_RANGE)
    frame_w = random.randint(*FRAME_W_RANGE)

    inner_coords = _choose_inner(frame_h, frame_w)
    if inner_coords is None:
        return None

    frame_coords = _build_frame(frame_h, frame_w, inner_coords)
    if frame_coords is None:
        return None

    padded = _validated_padded(frame_coords, inner_coords, frame_h, frame_w)
    if padded is None:
        return None

    grid = _place_on_grid(frame_coords, inner_coords, frame_h, frame_w, padded, frame_color, inner_color)
    if grid is None:
        return None

    try:
        main(grid)
    except Exception:
        return None

    return grid


def generate_input():
    for _ in range(MAX_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
