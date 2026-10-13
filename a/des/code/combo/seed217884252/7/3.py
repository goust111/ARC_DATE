import numpy as np


BG_COLOR = 0
FRAME_COUNT_CHOICES = (3, 4, 5)
PALETTE = tuple(range(1, 10))
INSET_RANGE = (1, 4)  # sampled with exclusive high
HOLE_HEIGHT_RANGE = (1, 6)
HOLE_WIDTH_RANGE = (1, 8)
MARGIN_RANGE = (0, 5)
GRID_SIZE_LIMIT = 30
MAX_GENERATION_ATTEMPTS = 300


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _background_color(grid: np.ndarray) -> int:
    flat = np.asarray(grid, dtype=int).ravel()
    if np.any(flat == BG_COLOR):
        return BG_COLOR
    return int(np.bincount(flat).argmax())


def _frame_area(frame: dict) -> int:
    r0, c0, r1, c1 = frame["bbox"]
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _extract_structure(grid: np.ndarray) -> tuple[int, list[dict], tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    frames: list[dict] = []
    loose_pixels: list[tuple[int, int, int]] = []

    for color in map(int, np.unique(g)):
        if color == bg:
            continue

        coords = np.argwhere(g == color)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)

        is_frame = False
        if height >= 3 and width >= 3:
            border = _border_mask(height, width)
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            if coords.shape[0] == int(border.sum()) and np.array_equal(sub == color, border):
                frames.append(
                    {
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "color": color,
                    }
                )
                is_frame = True

        if not is_frame:
            loose_pixels.extend((int(r), int(c), color) for r, c in coords)

    assert 3 <= len(frames) <= 5, "expected 3 to 5 nested frames"
    frames.sort(key=_frame_area, reverse=True)

    for outer, inner in zip(frames, frames[1:]):
        or0, oc0, or1, oc1 = outer["bbox"]
        ir0, ic0, ir1, ic1 = inner["bbox"]
        assert or0 < ir0 < ir1 < or1, "frame rows must be strictly nested"
        assert oc0 < ic0 < ic1 < oc1, "frame cols must be strictly nested"

    assert len(loose_pixels) == 1, "expected exactly one control pixel"
    control_r, control_c, control_color = loose_pixels[0]
    ir0, ic0, ir1, ic1 = frames[-1]["bbox"]
    assert ir0 < control_r < ir1 and ic0 < control_c < ic1, "control pixel must be inside the innermost frame"
    return bg, frames, (control_r, control_c, control_color)


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift nested frame colors outward by one level, using the control pixel color
    for the innermost frame and erasing the control pixel afterward.
    """
    g = np.asarray(grid, dtype=int)
    bg, frames, (control_r, control_c, control_color) = _extract_structure(g)

    out = g.copy()
    replacement_colors = [frame["color"] for frame in frames[1:]] + [control_color]
    for frame, new_color in zip(frames, replacement_colors):
        _paint_frame(out, frame["bbox"], new_color)

    out[control_r, control_c] = bg
    return out


def _sample_geometry(
    rng: np.random.Generator, n_frames: int
) -> tuple[list[tuple[int, int, int, int]], tuple[int, int, int, int], tuple[int, int]]:
    insets = [int(rng.integers(*INSET_RANGE)) for _ in range(n_frames - 1)]
    hole_height = int(rng.integers(*HOLE_HEIGHT_RANGE))
    hole_width = int(rng.integers(*HOLE_WIDTH_RANGE))

    inner_height = hole_height + 2
    inner_width = hole_width + 2
    total_inset = sum(insets)
    outer_height = inner_height + 2 * total_inset
    outer_width = inner_width + 2 * total_inset

    top = int(rng.integers(*MARGIN_RANGE))
    bottom = int(rng.integers(*MARGIN_RANGE))
    left = int(rng.integers(*MARGIN_RANGE))
    right = int(rng.integers(*MARGIN_RANGE))
    grid_height = outer_height + top + bottom
    grid_width = outer_width + left + right

    if grid_height > GRID_SIZE_LIMIT or grid_width > GRID_SIZE_LIMIT:
        raise ValueError("sampled geometry is too large")

    bboxes: list[tuple[int, int, int, int]] = []
    r0 = top
    c0 = left
    r1 = top + outer_height - 1
    c1 = left + outer_width - 1
    bboxes.append((r0, c0, r1, c1))
    for inset in insets:
        r0 += inset
        c0 += inset
        r1 -= inset
        c1 -= inset
        bboxes.append((r0, c0, r1, c1))

    ir0, ic0, ir1, ic1 = bboxes[-1]
    control_r = int(rng.integers(ir0 + 1, ir1))
    control_c = int(rng.integers(ic0 + 1, ic1))
    return bboxes, (grid_height, grid_width), (control_r, control_c)


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_frames = int(rng.choice(FRAME_COUNT_CHOICES))
    try:
        bboxes, (grid_height, grid_width), (control_r, control_c) = _sample_geometry(rng, n_frames)
    except ValueError:
        return None

    colors = rng.choice(PALETTE, size=n_frames + 1, replace=False)
    frame_colors = [int(color) for color in colors[:-1]]
    control_color = int(colors[-1])

    grid = np.full((grid_height, grid_width), BG_COLOR, dtype=int)
    for bbox, color in zip(bboxes, frame_colors):
        _paint_frame(grid, bbox, color)
    grid[control_r, control_c] = control_color

    if not np.any(grid == BG_COLOR):
        return None

    try:
        bg, frames, control = _extract_structure(grid)
    except AssertionError:
        return None

    if bg != BG_COLOR:
        return None
    if len(frames) != n_frames:
        return None
    if control[:2] != (control_r, control_c) or control[2] != control_color:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame instance")
