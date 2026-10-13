import numpy as np


BG = 0
DIVIDER_COLOR = 5
FRAME_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_H_MIN = 18
GRID_H_MAX = 26
LEFT_W_MIN = 14
LEFT_W_MAX = 18
RIGHT_W_MIN = 7
RIGHT_W_MAX = 11
FRAME_COUNT_MIN = 2
FRAME_COUNT_MAX = 4
FRAME_H_MIN = 4
FRAME_H_MAX = 8
FRAME_W_MIN = 4
FRAME_W_MAX = 8
PLACEMENT_MARGIN = 1
PLACEMENT_GAP = 1
PLACEMENT_TRIES = 300
GENERATE_MAX_RETRIES = 500


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _rectangle_border(height: int, width: int) -> np.ndarray:
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _find_divider(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    height, width = grid.shape
    for col in range(width):
        column = grid[:, col]
        color = int(column[0])
        if color == bg or not np.all(column == color):
            continue
        if col == 0 or col == width - 1:
            continue
        if not np.any(grid[:, :col] != bg):
            continue
        if not np.any(grid[:, col + 1 :] != bg):
            continue
        candidates.append((col, color))
    assert len(candidates) == 1, "expected one full-height divider column"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each right-side piece into the matching left-side frame gap and return only the repaired left region.

    Input structure:
    - The background is black, and one full-height grey column splits the grid into left and right halves.
    - The left half contains several colored hollow rectangles with one missing contiguous border segment of length 1-3.
    - The right half contains one connected piece per frame, using the same color as its matching frame.

    Rule:
    - Find the divider and keep only the area to its left for the output canvas.
    - For each non-divider color, pair the left connected component with the right connected component of that color.
    - Reconstruct the full rectangular border from the left component's bounding box, extract its missing border cells, and test the right piece under 0/90/180/270 degree rotations.
    - When a rotated piece matches the missing shape, fill that gap with the frame color and leave everything else unchanged.

    Key insight: The left object already determines the target rectangle; the right object only supplies the missing border segment under rotation, and the final output is cropped to the divider's left side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    divider_col, divider_color = _find_divider(g, bg)
    out = g[:, :divider_col].copy()

    colors = sorted(set(map(int, g.ravel())) - {bg, divider_color})
    assert colors, "expected at least one frame-piece color"

    for color in colors:
        comps = _components(g == color)
        assert len(comps) == 2, "each frame color must appear in exactly two components"

        left_parts = [coords for coords in comps if np.all(coords[:, 1] < divider_col)]
        right_parts = [coords for coords in comps if np.all(coords[:, 1] > divider_col)]
        assert len(left_parts) == 1 and len(right_parts) == 1

        frame_coords = left_parts[0]
        fr0, fc0, fr1, fc1 = _bbox_from_coords(frame_coords)
        frame_mask = np.zeros((fr1 - fr0 + 1, fc1 - fc0 + 1), dtype=bool)
        frame_mask[frame_coords[:, 0] - fr0, frame_coords[:, 1] - fc0] = True

        border = _rectangle_border(*frame_mask.shape)
        assert np.all(frame_mask <= border), "frame pixels must stay on their rectangle border"
        missing = border & ~frame_mask
        assert 1 <= int(missing.sum()) <= 3, "gap must be 1-3 pixels"
        assert len(_components(missing)) == 1, "gap must be one contiguous segment"

        piece_coords = right_parts[0]
        pr0, pc0, pr1, pc1 = _bbox_from_coords(piece_coords)
        piece_mask = np.zeros((pr1 - pr0 + 1, pc1 - pc0 + 1), dtype=bool)
        piece_mask[piece_coords[:, 0] - pr0, piece_coords[:, 1] - pc0] = True

        gap_shape = _tight_crop(missing)
        matches = [np.array_equal(np.rot90(piece_mask, k), gap_shape) for k in range(4)]
        assert any(matches), "piece must match the frame gap under rotation"

        patch = out[fr0 : fr1 + 1, fc0 : fc1 + 1]
        patch[missing] = color

    return out


def _border_cells(height: int, width: int) -> list[tuple[int, int]]:
    cells = [(0, c) for c in range(width)]
    cells += [(r, width - 1) for r in range(1, height - 1)]
    cells += [(height - 1, c) for c in range(width - 1, -1, -1)]
    cells += [(r, 0) for r in range(height - 2, 0, -1)]
    return cells


def _side_segment(height: int, width: int, side: int, length: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    if side == 0:
        start = int(rng.integers(0, width - length + 1))
        return [(0, start + i) for i in range(length)]
    if side == 1:
        start = int(rng.integers(0, height - length + 1))
        return [(start + i, width - 1) for i in range(length)]
    if side == 2:
        start = int(rng.integers(0, width - length + 1))
        return [(height - 1, start + i) for i in range(length)]
    start = int(rng.integers(0, height - length + 1))
    return [(start + i, 0) for i in range(length)]


def _sample_gap_mask(height: int, width: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    gap = np.zeros((height, width), dtype=bool)
    if mode == "single":
        cells = _border_cells(height, width)
        r, c = cells[int(rng.integers(0, len(cells)))]
        gap[r, c] = True
        return gap

    if mode == "double":
        coords = _side_segment(height, width, int(rng.integers(0, 4)), 2, rng)
        for r, c in coords:
            gap[r, c] = True
        return gap

    if mode == "triple":
        coords = _side_segment(height, width, int(rng.integers(0, 4)), 3, rng)
        for r, c in coords:
            gap[r, c] = True
        return gap

    corners = [
        [(0, 0), (0, 1), (1, 0)],
        [(0, width - 1), (0, width - 2), (1, width - 1)],
        [(height - 1, width - 1), (height - 1, width - 2), (height - 2, width - 1)],
        [(height - 1, 0), (height - 1, 1), (height - 2, 0)],
    ]
    for r, c in corners[int(rng.integers(0, 4))]:
        gap[r, c] = True
    return gap


def _boxes_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _sample_positions(
    shapes: list[tuple[int, int]],
    canvas_shape: tuple[int, int],
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    canvas_h, canvas_w = canvas_shape
    order = sorted(range(len(shapes)), key=lambda i: shapes[i][0] * shapes[i][1], reverse=True)
    positions: list[tuple[int, int] | None] = [None] * len(shapes)
    placed_boxes: list[tuple[int, int, int, int]] = []

    for idx in order:
        height, width = shapes[idx]
        min_r = PLACEMENT_MARGIN
        max_r = canvas_h - height - PLACEMENT_MARGIN
        min_c = PLACEMENT_MARGIN
        max_c = canvas_w - width - PLACEMENT_MARGIN
        if max_r < min_r or max_c < min_c:
            return None

        for _ in range(PLACEMENT_TRIES):
            top = int(rng.integers(min_r, max_r + 1))
            left = int(rng.integers(min_c, max_c + 1))
            box = (top, left, top + height - 1, left + width - 1)
            if any(_boxes_conflict(box, other, PLACEMENT_GAP) for other in placed_boxes):
                continue
            positions[idx] = (top, left)
            placed_boxes.append(box)
            break

        if positions[idx] is None:
            return None

    return [pos for pos in positions if pos is not None]


def _paint(canvas: np.ndarray, obj: np.ndarray, top: int, left: int) -> None:
    height, width = obj.shape
    patch = canvas[top : top + height, left : left + width]
    mask = obj != BG
    assert not np.any(patch[mask] != BG)
    patch[mask] = obj[mask]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    frame_count = int(rng.integers(FRAME_COUNT_MIN, FRAME_COUNT_MAX + 1))
    colors = [int(c) for c in rng.choice(FRAME_COLORS, size=frame_count, replace=False)]

    gap_modes = list(rng.permutation(np.array(["single", "double", "triple", "corner"], dtype=object)))[:frame_count]
    rotations = [int(k) for k in rng.permutation(4)[:frame_count]]

    specs = []
    for color, mode, rotation in zip(colors, gap_modes, rotations):
        frame_h = int(rng.integers(FRAME_H_MIN, FRAME_H_MAX + 1))
        frame_w = int(rng.integers(FRAME_W_MIN, FRAME_W_MAX + 1))
        gap = _sample_gap_mask(frame_h, frame_w, str(mode), rng)
        border = _rectangle_border(frame_h, frame_w)

        frame = np.full((frame_h, frame_w), BG, dtype=int)
        frame[border & ~gap] = color

        solved = np.full((frame_h, frame_w), BG, dtype=int)
        solved[border] = color

        piece_mask = _tight_crop(gap)
        piece = np.full(piece_mask.shape, BG, dtype=int)
        piece[piece_mask] = color
        piece = np.rot90(piece, rotation)

        specs.append({"frame": frame, "solved": solved, "piece": piece})

    max_frame_h = max(int(spec["frame"].shape[0]) for spec in specs)
    max_frame_w = max(int(spec["frame"].shape[1]) for spec in specs)
    max_piece_h = max(int(spec["piece"].shape[0]) for spec in specs)
    max_piece_w = max(int(spec["piece"].shape[1]) for spec in specs)

    height_low = max(GRID_H_MIN, max_frame_h + 2 * PLACEMENT_MARGIN, max_piece_h + 2 * PLACEMENT_MARGIN)
    if height_low > GRID_H_MAX:
        return None
    height = int(rng.integers(height_low, GRID_H_MAX + 1))

    right_low = max(RIGHT_W_MIN, max_piece_w + 2 * PLACEMENT_MARGIN)
    left_low = max(LEFT_W_MIN, max_frame_w + 2 * PLACEMENT_MARGIN)
    left_high = min(LEFT_W_MAX, 29 - right_low)
    if left_low > left_high:
        return None
    left_width = int(rng.integers(left_low, left_high + 1))

    right_high = min(RIGHT_W_MAX, 29 - left_width)
    if right_low > right_high:
        return None
    right_width = int(rng.integers(right_low, right_high + 1))

    left_positions = _sample_positions(
        [tuple(map(int, spec["frame"].shape)) for spec in specs],
        (height, left_width),
        rng,
    )
    if left_positions is None:
        return None

    right_positions = _sample_positions(
        [tuple(map(int, spec["piece"].shape)) for spec in specs],
        (height, right_width),
        rng,
    )
    if right_positions is None:
        return None

    left_input = np.full((height, left_width), BG, dtype=int)
    left_output = np.full((height, left_width), BG, dtype=int)
    right_input = np.full((height, right_width), BG, dtype=int)

    for spec, (top, left) in zip(specs, left_positions):
        _paint(left_input, spec["frame"], top, left)
        _paint(left_output, spec["solved"], top, left)

    for spec, (top, left) in zip(specs, right_positions):
        _paint(right_input, spec["piece"], top, left)

    grid = np.full((height, left_width + 1 + right_width), BG, dtype=int)
    grid[:, :left_width] = left_input
    grid[:, left_width] = DIVIDER_COLOR
    grid[:, left_width + 1 :] = right_input

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(out, left_output):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATE_MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to generate valid input")
