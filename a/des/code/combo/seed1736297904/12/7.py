import numpy as np


BLACK = 0
RED = 2
GREY = 5
OBJECT_COLORS = np.array([1, 3, 4, 6, 7, 8, 9], dtype=int)
SIDES = ("left", "right", "top", "bottom")
GENERATION_RETRIES = 400


def _detect_wall(grid: np.ndarray) -> str:
    candidates = []
    if np.all(grid[:, 0] == GREY):
        candidates.append("left")
    if np.all(grid[:, -1] == GREY):
        candidates.append("right")
    if np.all(grid[0, :] == GREY):
        candidates.append("top")
    if np.all(grid[-1, :] == GREY):
        candidates.append("bottom")
    assert len(candidates) == 1, f"expected exactly one grey wall edge, got {candidates}"
    return candidates[0]


def _foreground_components(grid: np.ndarray) -> list[dict[str, np.ndarray | int]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BLACK, GREY) or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "coords": np.asarray(coords, dtype=int)})

    return comps


def _wall_mask(shape: tuple[int, int], side: str) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if side == "left":
        mask[:, 0] = True
    elif side == "right":
        mask[:, -1] = True
    elif side == "top":
        mask[0, :] = True
    else:
        mask[-1, :] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each monochrome object orthogonally onto the shared grey wall.

    Objects activate wall cells by row for left/right walls and by column for
    top/bottom walls. A single activating object paints its own color; two or
    more activations collapse to red. Original objects are erased.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    side = _detect_wall(g)
    h, w = g.shape
    out = np.full((h, w), BLACK, dtype=int)
    wall = _wall_mask((h, w), side)
    out[wall] = GREY

    comps = _foreground_components(g)
    if side in ("left", "right"):
        wall_col = 0 if side == "left" else w - 1
        counts = np.zeros(h, dtype=int)
        single_color = np.zeros(h, dtype=int)
        for comp in comps:
            rows = np.unique(comp["coords"][:, 0])
            color = int(comp["color"])
            for row in rows:
                row = int(row)
                if counts[row] == 0:
                    single_color[row] = color
                counts[row] += 1
        for row in range(h):
            if counts[row] == 1:
                out[row, wall_col] = single_color[row]
            elif counts[row] >= 2:
                out[row, wall_col] = RED
    else:
        wall_row = 0 if side == "top" else h - 1
        counts = np.zeros(w, dtype=int)
        single_color = np.zeros(w, dtype=int)
        for comp in comps:
            cols = np.unique(comp["coords"][:, 1])
            color = int(comp["color"])
            for col in cols:
                col = int(col)
                if counts[col] == 0:
                    single_color[col] = color
                counts[col] += 1
        for col in range(w):
            if counts[col] == 1:
                out[wall_row, col] = single_color[col]
            elif counts[col] >= 2:
                out[wall_row, col] = RED

    return out


def _sample_widths(depth_len: int, count: int, rng: np.random.Generator) -> list[int]:
    widths = [2] * count
    slack = depth_len - (2 * count + (count - 1))
    assert slack >= 0
    order = rng.permutation(count)
    for idx in order:
        if slack <= 0:
            break
        extra = int(rng.integers(0, min(2, slack) + 1))
        widths[int(idx)] += extra
        slack -= extra
    return widths


def _sample_band_starts(depth_len: int, widths: list[int], rng: np.random.Generator) -> list[int]:
    mandatory = sum(widths) + (len(widths) - 1)
    slack = depth_len - mandatory
    extras = rng.multinomial(slack, np.full(len(widths) + 1, 1.0 / (len(widths) + 1)))

    starts = []
    pos = int(extras[0])
    for i, width in enumerate(widths):
        starts.append(pos)
        pos += int(width)
        if i < len(widths) - 1:
            pos += 1 + int(extras[i + 1])
    return starts


def _sample_intervals(axis_len: int, count: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    max_len = min(axis_len - 1, max(3, axis_len // 2 + 1))
    for _ in range(300):
        intervals = []
        for _ in range(count):
            length = int(rng.integers(2, max_len + 1))
            start = int(rng.integers(0, axis_len - length + 1))
            intervals.append((start, start + length - 1))

        counts = np.zeros(axis_len, dtype=int)
        for start, end in intervals:
            counts[start : end + 1] += 1

        if len(set(intervals)) < 2:
            continue
        if np.any(counts == 0) and np.any(counts == 1) and np.any(counts >= 2):
            return intervals
    return None


def _row_segment_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    prev_l = 0
    prev_r = width - 1

    for r in range(height):
        if r == 0:
            anchor = int(rng.integers(0, width))
        else:
            anchor = int(rng.integers(prev_l, prev_r + 1))
        left = anchor - int(rng.integers(0, min(2, anchor) + 1))
        right = anchor + int(rng.integers(0, min(2, width - 1 - anchor) + 1))
        mask[r, left : right + 1] = True
        prev_l, prev_r = left, right

    return mask


def _place_objects(
    grid: np.ndarray,
    side: str,
    intervals: list[tuple[int, int]],
    widths: list[int],
    starts: list[int],
    colors: np.ndarray,
    rng: np.random.Generator,
) -> None:
    for interval, band_width, band_start, color in zip(intervals, widths, starts, colors):
        start, end = interval
        span = end - start + 1
        if side in ("left", "right"):
            mask = _row_segment_mask(span, band_width, rng)
            top = start
            left = 2 + band_start if side == "left" else band_start
        else:
            mask = _row_segment_mask(span, band_width, rng).T
            top = 2 + band_start if side == "top" else band_start
            left = start

        patch = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
        assert patch.shape == mask.shape
        assert np.all(patch[mask] == BLACK)
        patch[mask] = int(color)


def _branch_values(output: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return output[:, 0]
    if side == "right":
        return output[:, -1]
    if side == "top":
        return output[0, :]
    return output[-1, :]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    side = str(rng.choice(SIDES))
    obj_count = int(rng.integers(2, 5))

    axis_len = int(rng.integers(8, 17))
    min_depth = 3 * obj_count - 1
    max_depth = max(min_depth + 1, 15)
    depth_len = int(rng.integers(min_depth, max_depth + 1))

    widths = _sample_widths(depth_len, obj_count, rng)
    starts = _sample_band_starts(depth_len, widths, rng)
    intervals = _sample_intervals(axis_len, obj_count, rng)
    if intervals is None:
        return None

    colors = rng.choice(OBJECT_COLORS, size=obj_count, replace=False)

    if side in ("left", "right"):
        grid = np.full((axis_len, depth_len + 2), BLACK, dtype=int)
        if side == "left":
            grid[:, 0] = GREY
        else:
            grid[:, -1] = GREY
    else:
        grid = np.full((depth_len + 2, axis_len), BLACK, dtype=int)
        if side == "top":
            grid[0, :] = GREY
        else:
            grid[-1, :] = GREY

    _place_objects(grid, side, intervals, widths, starts, colors, rng)

    comps = _foreground_components(grid)
    if len(comps) != obj_count:
        return None

    out = main(grid)
    wall_vals = _branch_values(out, side)
    if not np.any(wall_vals == GREY):
        return None
    if not np.any((wall_vals != GREY) & (wall_vals != RED)):
        return None
    if not np.any(wall_vals == RED):
        return None

    wall = _wall_mask(grid.shape, side)
    if np.any(out[~wall] != BLACK):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
