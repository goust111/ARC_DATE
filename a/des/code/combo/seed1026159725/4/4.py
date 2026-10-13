from __future__ import annotations

import random

import numpy as np


DEPTH_COLORS = {
    1: 1,  # blue
    2: 3,  # green
    3: 2,  # red
}

GEN_COLOR_POOL = [0, 4, 5, 6, 7, 8, 9]
GRID_HEIGHT_RANGE = (15, 30)
GRID_WIDTH_RANGE = (15, 30)
GROUP_COUNT_CHOICES = [1, 2, 3, 4]
GROUP_COUNT_WEIGHTS = [1, 3, 4, 2]
GROUP_MAX_OUTER = 16
INNER_SIZE_RANGE = (3, 8)
FRAME_GAP_RANGE = (2, 4)
PLACEMENT_BORDER_RANGE = (1, 2)
PLACEMENT_PADDING = 2
PLACEMENT_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    winners = values[counts == counts.max()]
    assert winners.size == 1, "expected a unique dominant background color"
    return int(winners[0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _strictly_contains(
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expanded(box: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    return box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _parse_frames(grid: np.ndarray) -> tuple[int, int, list[tuple[np.ndarray, tuple[int, int, int, int]]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _dominant_color(g)
    fg_colors = [int(v) for v in np.unique(g) if int(v) != bg]
    assert len(fg_colors) == 1, "expected exactly one frame color in the input"
    frame_color = fg_colors[0]

    frames: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    for coords in _components(g == frame_color):
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= 3 and width >= 3, "each frame must enclose background"

        local = np.zeros((height, width), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        expected = np.zeros_like(local)
        expected[0, :] = True
        expected[-1, :] = True
        expected[:, 0] = True
        expected[:, -1] = True
        assert np.array_equal(local, expected), "foreground components must be one-pixel-thick rectangular frames"
        frames.append((coords, box))

    assert frames, "expected at least one frame"

    boxes = [box for _, box in frames]
    for i, a in enumerate(boxes):
        for j, b in enumerate(boxes):
            if i >= j:
                continue
            if _boxes_overlap(a, b):
                assert _strictly_contains(a, b) or _strictly_contains(b, a), "frames may only be nested or disjoint"
    return bg, frame_color, frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each one-pixel-thick rectangular frame by its nesting depth.

    Depth 1 frames become blue, depth 2 frames become green, and depth 3 frames
    become red. Background outside frames and inside the innermost frame stays
    unchanged.
    """
    g = np.asarray(grid, dtype=int)
    _bg, _frame_color, frames = _parse_frames(g)
    boxes = [box for _, box in frames]

    out = g.copy()
    for coords, box in frames:
        depth = 1 + sum(_strictly_contains(other, box) for other in boxes if other != box)
        assert depth in DEPTH_COLORS, f"unsupported nesting depth {depth}"
        out[coords[:, 0], coords[:, 1]] = DEPTH_COLORS[depth]
    return out


def _sample_group_depths() -> list[int]:
    n_groups = random.choices(GROUP_COUNT_CHOICES, weights=GROUP_COUNT_WEIGHTS, k=1)[0]
    if n_groups == 1:
        return [random.randint(1, 3)]
    if n_groups == 2:
        patterns = [
            [1, 2],
            [1, 3],
            [2, 3],
            [1, 1],
            [2, 2],
            [3, 3],
            [1, 2],
            [1, 3],
            [2, 3],
        ]
        depths = random.choice(patterns)[:]
        random.shuffle(depths)
        return depths

    if random.random() < 0.75:
        depths = [1, 2, 3]
        while len(depths) < n_groups:
            depths.append(random.randint(1, 3))
    else:
        depths = [random.randint(1, 3) for _ in range(n_groups)]
    random.shuffle(depths)
    return depths


def _build_group(depth: int, frame_color: int, bg: int) -> tuple[np.ndarray, list[tuple[tuple[int, int, int, int], int]]]:
    while True:
        inner_h = random.randint(*INNER_SIZE_RANGE)
        inner_w = random.randint(*INNER_SIZE_RANGE)
        gaps = [
            (
                random.randint(*FRAME_GAP_RANGE),
                random.randint(*FRAME_GAP_RANGE),
                random.randint(*FRAME_GAP_RANGE),
                random.randint(*FRAME_GAP_RANGE),
            )
            for _ in range(depth - 1)
        ]

        outer_h = inner_h + sum(top + bottom for top, bottom, _left, _right in gaps)
        outer_w = inner_w + sum(left + right for _top, _bottom, left, right in gaps)
        if outer_h <= GROUP_MAX_OUTER and outer_w <= GROUP_MAX_OUTER:
            break

    boxes: list[tuple[tuple[int, int, int, int], int]] = []
    r0 = 0
    c0 = 0
    r1 = outer_h - 1
    c1 = outer_w - 1
    boxes.append(((r0, c0, r1, c1), 1))
    for level, (top, bottom, left, right) in enumerate(gaps, start=2):
        r0 += top
        c0 += left
        r1 -= bottom
        c1 -= right
        boxes.append(((r0, c0, r1, c1), level))

    patch = np.full((outer_h, outer_w), bg, dtype=int)
    for box, _depth in boxes:
        _draw_frame(patch, box, frame_color)
    return patch, boxes


def _can_place(
    box: tuple[int, int, int, int],
    placed: list[tuple[int, int, int, int]],
    border: int,
    shape: tuple[int, int],
) -> bool:
    h, w = shape
    if box[0] < border or box[1] < border or box[2] >= h - border or box[3] >= w - border:
        return False
    padded = _expanded(box, PLACEMENT_PADDING)
    for other in placed:
        if _boxes_overlap(padded, _expanded(other, PLACEMENT_PADDING)):
            return False
    return True


def _place_groups(
    grid_shape: tuple[int, int],
    groups: list[tuple[np.ndarray, list[tuple[tuple[int, int, int, int], int]]]],
) -> tuple[np.ndarray, list[tuple[tuple[int, int, int, int], int]]] | None:
    h, w = grid_shape
    border = random.randint(*PLACEMENT_BORDER_RANGE)
    order = sorted(range(len(groups)), key=lambda i: groups[i][0].shape[0] * groups[i][0].shape[1], reverse=True)
    placements: dict[int, tuple[int, int]] = {}
    placed_boxes: list[tuple[int, int, int, int]] = []

    for idx in order:
        patch, _boxes = groups[idx]
        ph, pw = patch.shape
        if ph + 2 * border > h or pw + 2 * border > w:
            return None

        positions = [
            (r, c)
            for r in range(border, h - border - ph + 1)
            for c in range(border, w - border - pw + 1)
        ]
        random.shuffle(positions)

        placed = False
        for r, c in positions[:PLACEMENT_ATTEMPTS]:
            candidate = (r, c, r + ph - 1, c + pw - 1)
            if _can_place(candidate, placed_boxes, border, grid_shape):
                placements[idx] = (r, c)
                placed_boxes.append(candidate)
                placed = True
                break
        if not placed:
            return None

    return placements


def _render_input(
    grid_shape: tuple[int, int],
    bg: int,
    groups: list[tuple[np.ndarray, list[tuple[tuple[int, int, int, int], int]]]],
    placements: dict[int, tuple[int, int]],
) -> tuple[np.ndarray, list[tuple[tuple[int, int, int, int], int]]]:
    grid = np.full(grid_shape, bg, dtype=int)
    abs_boxes: list[tuple[tuple[int, int, int, int], int]] = []
    for idx, (patch, boxes) in enumerate(groups):
        r0, c0 = placements[idx]
        ph, pw = patch.shape
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch
        for (br0, bc0, br1, bc1), depth in boxes:
            abs_boxes.append(((r0 + br0, c0 + bc0, r0 + br1, c0 + bc1), depth))
    return grid, abs_boxes


def _expected_output(
    grid: np.ndarray,
    frame_specs: list[tuple[tuple[int, int, int, int], int]],
) -> np.ndarray:
    out = grid.copy()
    for box, depth in frame_specs:
        _draw_frame(out, box, DEPTH_COLORS[depth])
    return out


def _generate_attempt() -> np.ndarray | None:
    bg, frame_color = random.sample(GEN_COLOR_POOL, 2)
    group_depths = _sample_group_depths()
    groups = [_build_group(depth, frame_color, bg) for depth in group_depths]

    max_h = max(patch.shape[0] for patch, _ in groups)
    max_w = max(patch.shape[1] for patch, _ in groups)
    min_h = max(GRID_HEIGHT_RANGE[0], max_h + 2)
    min_w = max(GRID_WIDTH_RANGE[0], max_w + 2)
    if min_h > GRID_HEIGHT_RANGE[1] or min_w > GRID_WIDTH_RANGE[1]:
        return None

    height = random.randint(min_h, GRID_HEIGHT_RANGE[1])
    width = random.randint(min_w, GRID_WIDTH_RANGE[1])
    placements = _place_groups((height, width), groups)
    if placements is None:
        return None

    grid, frame_specs = _render_input((height, width), bg, groups, placements)
    if np.count_nonzero(grid == bg) <= np.count_nonzero(grid != bg):
        return None

    expected = _expected_output(grid, frame_specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame input")
