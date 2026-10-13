import random

import numpy as np


BG = 0
PIPE = 5
ARROW = 7
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_COLORS = [1, 2, 3, 4, 6, 8, 9]
GEN_RETRIES = 500
BASE_CANVAS = 30


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)]


def _frame_cells(top: int, left: int) -> list[tuple[int, int]]:
    cells = []
    for dr in range(3):
        for dc in range(3):
            if (dr, dc) != (1, 1):
                cells.append((top + dr, left + dc))
    return cells


def _find_frames(grid: np.ndarray) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], tuple[int, int]], np.ndarray]:
    h, w = grid.shape
    centers: dict[tuple[int, int], int] = {}
    border_to_center: dict[tuple[int, int], tuple[int, int]] = {}
    frame_mask = np.zeros((h, w), dtype=bool)

    for r in range(h - 2):
        for c in range(w - 2):
            patch = grid[r : r + 3, c : c + 3]
            if int(patch[1, 1]) != BG:
                continue
            border = [
                int(patch[0, 0]),
                int(patch[0, 1]),
                int(patch[0, 2]),
                int(patch[1, 0]),
                int(patch[1, 2]),
                int(patch[2, 0]),
                int(patch[2, 1]),
                int(patch[2, 2]),
            ]
            color = border[0]
            if color in (BG, PIPE, ARROW):
                continue
            if any(v != color for v in border):
                continue
            center = (r + 1, c + 1)
            centers[center] = color
            for cell in _frame_cells(r, c):
                border_to_center[cell] = center
                frame_mask[cell] = True

    return centers, border_to_center, frame_mask


def _find_signals(grid: np.ndarray, frame_mask: np.ndarray) -> list[tuple[tuple[int, int], int, tuple[int, int]]]:
    h, w = grid.shape
    border_cells = []
    for c in range(w):
        border_cells.append((0, c))
        border_cells.append((h - 1, c))
    for r in range(1, h - 1):
        border_cells.append((r, 0))
        border_cells.append((r, w - 1))

    signals = []
    for r, c in border_cells:
        if frame_mask[r, c]:
            continue
        color = int(grid[r, c])
        if color in (BG, PIPE, ARROW):
            continue
        pipe_dirs = []
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == PIPE:
                pipe_dirs.append((dr, dc))
        if len(pipe_dirs) == 1:
            signals.append(((r, c), color, pipe_dirs[0]))
    return signals


def _adjacent_frame_center(pos: tuple[int, int], border_to_center: dict[tuple[int, int], tuple[int, int]]) -> tuple[int, int] | None:
    r, c = pos
    centers = {
        border_to_center[(nr, nc)]
        for nr, nc in _neighbors4(r, c)
        if (nr, nc) in border_to_center
    }
    if len(centers) == 1:
        return next(iter(centers))
    return None


def _left_right(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    idx = DIRS.index(direction)
    return DIRS[(idx - 1) % 4], DIRS[(idx + 1) % 4]


def _trace_signal(
    grid: np.ndarray,
    start: tuple[int, int],
    start_dir: tuple[int, int],
    border_to_center: dict[tuple[int, int], tuple[int, int]],
) -> tuple[int, int]:
    h, w = grid.shape
    prev = (start[0] - start_dir[0], start[1] - start_dir[1])
    current = start
    visited = {start}

    for _ in range(h * w * 4):
        exit_center = _adjacent_frame_center(current, border_to_center)
        next_cells = []
        for dr, dc in DIRS:
            nr, nc = current[0] + dr, current[1] + dc
            if (nr, nc) == prev:
                continue
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == PIPE:
                next_cells.append((nr, nc))

        if not next_cells:
            assert exit_center is not None, "pipe must terminate at a frame"
            return exit_center

        assert exit_center is None, "frame should only appear at a pipe terminal"

        if len(next_cells) == 1:
            nxt = next_cells[0]
        else:
            assert len(next_cells) == 2, "junctions should be T-shaped"
            direction = (current[0] - prev[0], current[1] - prev[1])
            left_dir, right_dir = _left_right(direction)
            left_cell = (current[0] + left_dir[0], current[1] + left_dir[1])
            right_cell = (current[0] + right_dir[0], current[1] + right_dir[1])
            assert set(next_cells) == {left_cell, right_cell}, "T junction must offer left/right branches"

            left_mark = (current[0] + direction[0] + left_dir[0], current[1] + direction[1] + left_dir[1])
            right_mark = (current[0] + direction[0] + right_dir[0], current[1] + direction[1] + right_dir[1])

            left_marked = 0 <= left_mark[0] < h and 0 <= left_mark[1] < w and int(grid[left_mark]) == ARROW
            right_marked = 0 <= right_mark[0] < h and 0 <= right_mark[1] < w and int(grid[right_mark]) == ARROW
            assert left_marked ^ right_marked, "exactly one branch marker is required"
            nxt = left_cell if left_marked else right_cell

        prev, current = current, nxt
        assert current not in visited, "path should not loop"
        visited.add(current)

    raise AssertionError("signal trace exceeded search budget")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each colored signal through the grey pipe network and fill the destination container.

    Input structure:
    - bg = black background.
    - Grey pixels form one-cell-wide horizontal/vertical pipes, with straight runs, L bends, and T junctions.
    - Colored signal pixels sit on pipe entrances on the grid edge, and each exit is a colored hollow 3x3 square with a black center.
    - Orange pixels sit just outside T junctions and indicate which branch should be taken.

    Rule:
    - Each signal travels along the pipe from its edge entrance.
    - At straight runs and L bends, the only available continuation is followed automatically.
    - At a T junction, interpret the orange marker relative to the signal's incoming direction: a marker ahead-left means turn left, and ahead-right means turn right.
    - When the signal reaches the pipe cell adjacent to an exit frame, fill that frame's center with the signal color while leaving everything else unchanged.

    Key insight:
    - The hard part is not path following itself but reading each orange marker in the moving signal's local frame of reference, so the same absolute marker position can mean different turns after rotation.
    """
    grid = np.asarray(grid, dtype=int)
    out = grid.copy()
    frames, border_to_center, frame_mask = _find_frames(grid)
    signals = _find_signals(grid, frame_mask)

    assert frames, "expected at least one exit frame"
    assert signals, "expected at least one edge signal"

    for start, color, start_dir in signals:
        center = _trace_signal(grid, start, start_dir, border_to_center)
        out[center] = color

    return out


def _draw_frame_left(grid: np.ndarray, row: int, left: int, color: int) -> None:
    grid[row - 1 : row + 2, left : left + 3] = color
    grid[row, left + 1] = BG


def _build_component(signal_color: int, frame_color: int) -> np.ndarray:
    track_count = random.randint(3, 4)
    track_rows = [2 + 3 * i for i in range(track_count)]
    height = track_rows[-1] + 3

    feature_count = random.randint(2, 3)
    feature_types = ["T"] + [random.choice(["T", "L"]) for _ in range(feature_count - 1)]
    horizontal_runs = [random.randint(2, 4) for _ in range(feature_count + 1)]

    start_track = random.randrange(track_count)
    current_track = start_track
    current_col = 0
    features: list[tuple[str, int, int, int]] = []
    for idx, kind in enumerate(feature_types):
        current_col += horizontal_runs[idx]
        next_track = random.choice([t for t in range(track_count) if t != current_track])
        features.append((kind, current_track, next_track, current_col))
        current_track = next_track

    end_col = current_col + horizontal_runs[-1]
    width = end_col + 5
    grid = np.full((height, width), BG, dtype=int)

    current_track = start_track
    current_row = track_rows[current_track]
    current_col = 0
    grid[current_row, 0] = signal_color

    for kind, from_track, to_track, junction_col in features:
        assert from_track == current_track
        for c in range(current_col + 1, junction_col + 1):
            grid[current_row, c] = PIPE

        step = 1 if to_track > from_track else -1
        target_row = track_rows[to_track]
        for r in range(current_row + step, target_row + step, step):
            grid[r, junction_col] = PIPE

        if kind == "T":
            stub_step = -step
            available = current_row - 1 if stub_step == -1 else height - 2 - current_row
            stub_len = random.randint(1, min(2, available))
            for r in range(current_row + stub_step, current_row + stub_step * (stub_len + 1), stub_step):
                grid[r, junction_col] = PIPE
            grid[current_row + step, junction_col + 1] = ARROW

        current_track = to_track
        current_row = target_row
        current_col = junction_col

    for c in range(current_col + 1, end_col + 1):
        grid[current_row, c] = PIPE
    _draw_frame_left(grid, current_row, end_col + 1, frame_color)
    return grid


def _rotate_to_side(component: np.ndarray, side: str) -> np.ndarray:
    rotations = {"left": 0, "bottom": 1, "right": 2, "top": 3}[side]
    return np.rot90(component, rotations).copy()


def _can_place(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int = 1) -> bool:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > canvas.shape[0] or c0 + w > canvas.shape[1]:
        return False

    for dr, dc in np.argwhere(patch != BG):
        r = int(r0 + dr)
        c = int(c0 + dc)
        r1 = max(0, r - padding)
        r2 = min(canvas.shape[0], r + padding + 1)
        c1 = max(0, c - padding)
        c2 = min(canvas.shape[1], c + padding + 1)
        if np.any(canvas[r1:r2, c1:c2] != BG):
            return False
    return True


def _place_patch(canvas: np.ndarray, patch: np.ndarray, side: str) -> bool:
    h, w = patch.shape
    candidates: list[tuple[int, int]] = []
    if side == "left":
        rows = list(range(canvas.shape[0] - h + 1))
        random.shuffle(rows)
        candidates = [(r, 0) for r in rows]
    elif side == "right":
        rows = list(range(canvas.shape[0] - h + 1))
        random.shuffle(rows)
        candidates = [(r, canvas.shape[1] - w) for r in rows]
    elif side == "top":
        cols = list(range(canvas.shape[1] - w + 1))
        random.shuffle(cols)
        candidates = [(0, c) for c in cols]
    else:
        cols = list(range(canvas.shape[1] - w + 1))
        random.shuffle(cols)
        candidates = [(canvas.shape[0] - h, c) for c in cols]

    for r0, c0 in candidates:
        if not _can_place(canvas, patch, r0, c0):
            continue
        mask = patch != BG
        canvas[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]
        return True
    return False


def _crop_canvas(canvas: np.ndarray, used_sides: set[str]) -> np.ndarray:
    points = np.argwhere(canvas != BG)
    r_min, c_min = points.min(axis=0)
    r_max, c_max = points.max(axis=0)

    top_blank = int(r_min)
    bottom_blank = int(canvas.shape[0] - 1 - r_max)
    left_blank = int(c_min)
    right_blank = int(canvas.shape[1] - 1 - c_max)

    top_crop = 0 if "top" in used_sides else random.randint(0, max(0, top_blank - 1))
    bottom_crop = 0 if "bottom" in used_sides else random.randint(0, max(0, bottom_blank - 1))
    left_crop = 0 if "left" in used_sides else random.randint(0, max(0, left_blank - 1))
    right_crop = 0 if "right" in used_sides else random.randint(0, max(0, right_blank - 1))

    return canvas[
        top_crop : canvas.shape[0] - bottom_crop,
        left_crop : canvas.shape[1] - right_crop,
    ]


def _valid_instance(grid: np.ndarray, signal_colors: list[int]) -> bool:
    frames, _, frame_mask = _find_frames(grid)
    signals = _find_signals(grid, frame_mask)
    if len(frames) != len(signal_colors) or len(signals) != len(signal_colors):
        return False
    if int(np.sum(grid == ARROW)) < len(signal_colors):
        return False

    out = main(grid)
    changed = {tuple(p) for p in np.argwhere(out != grid)}
    if changed != set(frames.keys()):
        return False

    filled = sorted(int(out[r, c]) for r, c in frames)
    return filled == sorted(signal_colors)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        count = random.choice([1, 2, 2, 3])
        colors = random.sample(GEN_COLORS, 2 * count)
        signal_colors = colors[:count]
        frame_colors = colors[count:]
        sides = random.sample(["left", "right", "top", "bottom"], count)

        canvas = np.full((BASE_CANVAS, BASE_CANVAS), BG, dtype=int)
        ok = True
        for side, signal_color, frame_color in zip(sides, signal_colors, frame_colors, strict=False):
            component = _build_component(signal_color, frame_color)
            component = _rotate_to_side(component, side)
            if not _place_patch(canvas, component, side):
                ok = False
                break
        if not ok:
            continue

        cropped = _crop_canvas(canvas, set(sides))
        if max(cropped.shape) > 30:
            continue
        if _valid_instance(cropped, signal_colors):
            return cropped

    raise RuntimeError("failed to generate a valid pipe-network instance")
