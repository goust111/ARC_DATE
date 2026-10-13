import numpy as np


BG = 0
RNG = np.random.default_rng()
MASTER_SIZE = 30
FRAME_COUNT_RANGE = (3, 5)
PLACEMENT_PAD = 1
PLACEMENT_ATTEMPTS = 500
GENERATION_ATTEMPTS = 300
COLOR_POOL = tuple(range(1, 10))
MIN_DIM_BY_PARITY = {0: 4, 1: 3}
MAX_DIM_BY_COUNT = {
    3: {0: 20, 1: 19},
    4: {0: 18, 1: 17},
    5: {0: 16, 1: 15},
}


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    blocked = set() if exclude is None else set(exclude)
    return [int(v) for v in np.unique(grid) if int(v) not in blocked]


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _draw_frame(canvas: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    bottom = top + height - 1
    right = left + width - 1
    canvas[top, left : right + 1] = color
    canvas[bottom, left : right + 1] = color
    canvas[top : bottom + 1, left] = color
    canvas[top : bottom + 1, right] = color


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    seen = np.zeros_like(mask, dtype=bool)
    stack = [tuple(int(v) for v in coords[0])]
    seen[stack[0]] = True
    count = 0
    height, width = mask.shape

    while stack:
        r, c = stack.pop()
        count += 1
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                stack.append((rr, cc))

    return count == int(mask.sum())


def _extract_frames(grid: np.ndarray) -> list[dict[str, int]]:
    frames: list[dict[str, int]] = []
    for color in _colors_present(grid, exclude={BG}):
        mask = grid == color
        assert _is_connected(mask), "each foreground color must form one 4-connected component"
        coords = np.argwhere(mask)
        (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        assert height >= 2 and width >= 2, "rectangle frames must have height and width at least 2"
        submask = mask[r0 : r1 + 1, c0 : c1 + 1]
        assert np.array_equal(submask, _border_mask(height, width)), "component must be a complete rectangle border"
        frames.append(
            {
                "color": int(color),
                "height": height,
                "width": width,
                "area": int(height * width),
            }
        )

    assert frames, "input must contain at least one colored rectangle frame"
    frames.sort(key=lambda item: (item["area"], item["height"], item["width"]))
    return frames


def _compose_output(frames: list[dict[str, int]], bg: int = BG) -> np.ndarray:
    out_h = max(frame["height"] for frame in frames)
    out_w = max(frame["width"] for frame in frames)
    out = np.full((out_h, out_w), bg, dtype=int)

    for frame in frames:
        top = (out_h - frame["height"]) // 2
        left = (out_w - frame["width"]) // 2
        _draw_frame(out, top, left, frame["height"], frame["width"], frame["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rearrange scattered colored rectangle frames into concentric frames ordered by area.

    Input structure:
    - A black background contains several non-overlapping foreground color groups.
    - Each foreground color appears in exactly one 4-connected component.
    - Every component is a one-pixel-thick complete rectangular border with its own width and height.

    Rule:
    - Detect each colored component and reconstruct its bounding rectangle.
    - Verify that the component exactly matches the border of that rectangle, then measure its width and height.
    - Sort the rectangle frames by area (width x height) from smallest to largest.
    - Create an output canvas just large enough for the largest frame and redraw all frames centered on the same canvas, from inner to outer, preserving their colors.

    Key insight:
    - The scattered input objects are already complete rectangle frames; the task is to identify their geometry, sort by size, and recenter them into a single concentric nesting.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    frames = _extract_frames(g)
    return _compose_output(frames, bg=BG)


def _sample_sizes(count: int, parity: int) -> list[int]:
    start = MIN_DIM_BY_PARITY[parity]
    stop = MAX_DIM_BY_COUNT[count][parity]
    candidates = np.arange(start, stop + 1, 2, dtype=int)
    assert len(candidates) >= count
    chosen = np.sort(RNG.choice(candidates, size=count, replace=False))
    return [int(v) for v in chosen]


def _random_frames() -> list[dict[str, int]]:
    count = int(RNG.integers(FRAME_COUNT_RANGE[0], FRAME_COUNT_RANGE[1] + 1))
    parity_h = int(RNG.integers(0, 2))
    parity_w = int(RNG.integers(0, 2))
    heights = _sample_sizes(count, parity_h)
    widths = _sample_sizes(count, parity_w)
    colors = RNG.choice(np.array(COLOR_POOL, dtype=int), size=count, replace=False)

    frames = []
    for idx in range(count):
        h = heights[idx]
        w = widths[idx]
        frames.append(
            {
                "color": int(colors[idx]),
                "height": int(h),
                "width": int(w),
                "area": int(h * w),
            }
        )

    frames.sort(key=lambda item: (item["area"], item["height"], item["width"]))
    return frames


def _place_frames(frames: list[dict[str, int]]) -> np.ndarray | None:
    grid = np.full((MASTER_SIZE, MASTER_SIZE), BG, dtype=int)
    blocked = np.zeros_like(grid, dtype=bool)

    for frame in sorted(frames, key=lambda item: item["area"], reverse=True):
        height = frame["height"]
        width = frame["width"]
        placed = False

        for _ in range(PLACEMENT_ATTEMPTS):
            top = int(RNG.integers(0, MASTER_SIZE - height + 1))
            left = int(RNG.integers(0, MASTER_SIZE - width + 1))

            r0 = max(0, top - PLACEMENT_PAD)
            c0 = max(0, left - PLACEMENT_PAD)
            r1 = min(MASTER_SIZE, top + height + PLACEMENT_PAD)
            c1 = min(MASTER_SIZE, left + width + PLACEMENT_PAD)
            if blocked[r0:r1, c0:c1].any():
                continue

            _draw_frame(grid, top, left, height, width, frame["color"])
            blocked[r0:r1, c0:c1] = True
            placed = True
            break

        if not placed:
            return None

    return grid


def _crop_randomly(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BG)
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    top = int(RNG.integers(0, int(r0) + 1))
    left = int(RNG.integers(0, int(c0) + 1))
    bottom = int(RNG.integers(int(r1) + 1, grid.shape[0] + 1))
    right = int(RNG.integers(int(c1) + 1, grid.shape[1] + 1))
    return grid[top:bottom, left:right]


def _frame_signature(frames: list[dict[str, int]]) -> list[tuple[int, int, int]]:
    return [(frame["color"], frame["height"], frame["width"]) for frame in frames]


def _validate_generated(grid: np.ndarray, frames: list[dict[str, int]]) -> None:
    parsed = _extract_frames(grid)
    assert _frame_signature(parsed) == _frame_signature(frames)
    expected = _compose_output(frames, bg=BG)
    actual = main(grid)
    assert np.array_equal(actual, expected)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert actual.ndim == 2 and max(actual.shape) <= 30


def _gen_attempt() -> np.ndarray | None:
    frames = _random_frames()
    placed = _place_frames(frames)
    if placed is None:
        return None

    grid = _crop_randomly(placed)
    _validate_generated(grid, frames)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
