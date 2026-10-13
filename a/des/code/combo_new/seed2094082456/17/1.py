import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
MARKER_COLORS = (2, 3, 4, 6, 7, 8, 9)

FRAME_COUNT_CHOICES = (2, 3, 4)
OUTER_SIZE_CHOICES = (7, 9, 11)
MAX_SIDE = 30
MAX_GENERATION_ATTEMPTS = 400

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRECTIONS = ("top", "right", "bottom", "left")
ROTATION_K = {
    "top": 0,
    "right": 3,
    "bottom": 2,
    "left": 1,
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(cells)
    return parts


def _frame_gap_side(patch: np.ndarray) -> str:
    size = patch.shape[0]
    mid = size // 2
    side_cells = {
        "top": (0, mid),
        "right": (mid, size - 1),
        "bottom": (size - 1, mid),
        "left": (mid, 0),
    }
    gaps = []
    for side, (r, c) in side_cells.items():
        value = int(patch[r, c])
        if value == BLACK:
            gaps.append(side)
        else:
            assert value == GREY, "frame midpoint border pixels must be grey except the unique gap"
    assert len(gaps) == 1, "each frame must have exactly one midpoint gap"
    return gaps[0]


def _extract_frames(grid: np.ndarray) -> list[tuple[int, int, int, str]]:
    frames: list[tuple[int, int, int, str]] = []
    for cells in _components(grid == GREY):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        size = r1 - r0 + 1
        if size != c1 - c0 + 1 or size < 5 or size % 2 == 0:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        gap_side = _frame_gap_side(patch)
        gap_pos = {
            "top": (0, size // 2),
            "right": (size // 2, size - 1),
            "bottom": (size - 1, size // 2),
            "left": (size // 2, 0),
        }[gap_side]
        for rr in range(size):
            for cc in range(size):
                is_border = rr in (0, size - 1) or cc in (0, size - 1)
                value = int(patch[rr, cc])
                if is_border:
                    if (rr, cc) == gap_pos:
                        assert value == BLACK, "gap cell must be black"
                    else:
                        assert value == GREY, "frame border must be a one-pixel grey square"
                else:
                    assert value != GREY, "grey pixels may only appear on the frame border"
        frames.append((r0, c0, size, gap_side))
    frames.sort()
    assert frames, "expected at least one grey square frame"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each framed blue sprite from the border gap direction and recolor it from the interior marker.

    Input structure:
    - The background is black.
    - Several disconnected grey one-pixel square frames appear on the grid, and
      each frame has exactly one missing midpoint pixel on one side.
    - Every frame contains the same unrotated blue pattern plus exactly one
      single-pixel marker in some other color on an otherwise black interior cell.

    Rule:
    - Read the missing border midpoint to choose the rotation of that frame's
      interior blue pattern: top -> 0 degrees, right -> 90 degrees clockwise,
      bottom -> 180 degrees, left -> 270 degrees clockwise.
    - Replace every blue pixel of the rotated pattern with the frame's marker color.
    - Clear the original marker pixel to black and keep the grey frame and its gap unchanged.

    Key insight:
    - The border gap and the colored marker are two independent controls: one
      selects the rotation, and the other selects the output color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for r0, c0, size, gap_side in _extract_frames(g):
        inner = g[r0 + 1 : r0 + size - 1, c0 + 1 : c0 + size - 1]
        assert inner.shape[0] == inner.shape[1], "frame interior must be square"

        marker_cells = np.argwhere((inner != BLACK) & (inner != BLUE))
        assert len(marker_cells) == 1, "each frame must contain exactly one marker pixel"
        mr, mc = map(int, marker_cells[0])
        marker_color = int(inner[mr, mc])
        assert marker_color != GREY, "marker color must differ from grey"

        blue_mask = inner == BLUE
        rotated_mask = np.rot90(blue_mask, ROTATION_K[gap_side])

        painted = np.full(inner.shape, BLACK, dtype=int)
        painted[rotated_mask] = marker_color
        out[r0 + 1 : r0 + size - 1, c0 + 1 : c0 + size - 1] = painted

    return out


def _split_total(total: int, parts: int, rng: random.Random) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[rng.randrange(parts)] += 1
    return values


def _random_connected_mask(inner_size: int, frame_count: int, rng: random.Random) -> tuple[np.ndarray, list[tuple[int, int]]]:
    max_cells = min(7, inner_size + 2)
    for _ in range(600):
        target = rng.randint(4, max_cells)
        start = (rng.randrange(inner_size), rng.randrange(inner_size))
        cells = {start}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < inner_size and 0 <= nc < inner_size and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))
        if len(cells) != target:
            continue

        mask = np.zeros((inner_size, inner_size), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        rows, cols = np.where(mask)
        bbox_h = int(rows.max() - rows.min() + 1)
        bbox_w = int(cols.max() - cols.min() + 1)
        if bbox_h < 2 or bbox_w < 2:
            continue

        rotations = [np.rot90(mask, k) for k in range(4)]
        if any(np.array_equal(rotations[i], rotations[j]) for i in range(4) for j in range(i + 1, 4)):
            continue

        reserved = np.zeros_like(mask, dtype=bool)
        for rot in rotations:
            reserved |= rot
        available = [tuple(map(int, cell)) for cell in np.argwhere(~reserved)]
        if len(available) < frame_count:
            continue

        interior_available = [
            (r, c)
            for r, c in available
            if 0 < r < inner_size - 1 and 0 < c < inner_size - 1
        ]
        if len(interior_available) >= frame_count:
            available = interior_available
        return mask, available
    raise RuntimeError("failed to generate a non-symmetric sprite with safe marker cells")


def _layout_spec(frame_count: int, rng: random.Random) -> tuple[int, int, list[tuple[int, int]]]:
    if frame_count == 2:
        if rng.random() < 0.5:
            return 1, 2, [(0, 0), (0, 1)]
        return 2, 1, [(0, 0), (1, 0)]
    if frame_count == 3:
        slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
        return 2, 2, rng.sample(slots, 3)
    return 2, 2, [(0, 0), (0, 1), (1, 0), (1, 1)]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_count = rng.choice(FRAME_COUNT_CHOICES)
    band_rows, band_cols, slots = _layout_spec(frame_count, rng)

    feasible_sizes = [
        size
        for size in OUTER_SIZE_CHOICES
        if band_rows * (size + 2) <= MAX_SIDE and band_cols * (size + 2) <= MAX_SIDE
    ]
    if not feasible_sizes:
        return None
    outer_size = rng.choice(feasible_sizes)
    inner_size = outer_size - 2

    try:
        pattern_mask, marker_candidates = _random_connected_mask(inner_size, frame_count, rng)
    except RuntimeError:
        return None

    base_h = band_rows * (outer_size + 2)
    base_w = band_cols * (outer_size + 2)
    total_h = rng.randint(base_h, min(MAX_SIDE, base_h + 6))
    total_w = rng.randint(base_w, min(MAX_SIDE, base_w + 6))
    row_sizes = [outer_size + 2 + extra for extra in _split_total(total_h - base_h, band_rows, rng)]
    col_sizes = [outer_size + 2 + extra for extra in _split_total(total_w - base_w, band_cols, rng)]

    row_starts = []
    acc = 0
    for size in row_sizes:
        row_starts.append(acc)
        acc += size
    col_starts = []
    acc = 0
    for size in col_sizes:
        col_starts.append(acc)
        acc += size

    positions = []
    for band_r, band_c in slots:
        top_room = row_sizes[band_r] - outer_size - 1
        left_room = col_sizes[band_c] - outer_size - 1
        top = row_starts[band_r] + 1 + rng.randrange(top_room)
        left = col_starts[band_c] + 1 + rng.randrange(left_room)
        positions.append((top, left))

    directions = rng.sample(list(DIRECTIONS), frame_count)
    marker_colors = rng.sample(list(MARKER_COLORS), frame_count)
    marker_positions = rng.sample(marker_candidates, frame_count)

    grid = np.full((total_h, total_w), BLACK, dtype=int)
    mid = outer_size // 2
    for (top, left), direction, marker_color, marker_pos in zip(
        positions, directions, marker_colors, marker_positions
    ):
        grid[top, left : left + outer_size] = GREY
        grid[top + outer_size - 1, left : left + outer_size] = GREY
        grid[top : top + outer_size, left] = GREY
        grid[top : top + outer_size, left + outer_size - 1] = GREY

        if direction == "top":
            grid[top, left + mid] = BLACK
        elif direction == "right":
            grid[top + mid, left + outer_size - 1] = BLACK
        elif direction == "bottom":
            grid[top + outer_size - 1, left + mid] = BLACK
        else:
            grid[top + mid, left] = BLACK

        inner = grid[top + 1 : top + outer_size - 1, left + 1 : left + outer_size - 1]
        inner[pattern_mask] = BLUE
        mr, mc = marker_pos
        inner[mr, mc] = marker_color

    try:
        frames = _extract_frames(grid)
    except AssertionError:
        return None
    if len(frames) != frame_count:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
