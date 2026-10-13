import numpy as np


BACKGROUND = 0
YELLOW = 4
TRACK_COLORS = np.array([1, 2, 3, 5, 6, 7, 8, 9], dtype=int)
GRID_LIMIT = 30
GENERATION_RETRIES = 300
LAYOUT_RETRIES = 200
_RNG = np.random.default_rng()


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
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


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append(np.array(coords, dtype=int))

    return components


def _analyze_track(grid: np.ndarray, coords: np.ndarray) -> tuple[list[tuple[int, int]], list[int], int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert 3 <= height <= 8 and 3 <= width <= 8

    border = _border_coords(r0, c0, r1, c1)
    coord_set = {tuple(map(int, coord)) for coord in coords}
    assert len(coord_set) == len(border) == 2 * (height + width) - 4
    assert coord_set == set(border)

    track_colors = {int(grid[r, c]) for r, c in border if int(grid[r, c]) != YELLOW}
    assert len(track_colors) == 1
    track_color = int(next(iter(track_colors)))
    assert track_color not in (BACKGROUND, YELLOW)

    markers = [index for index, (r, c) in enumerate(border) if int(grid[r, c]) == YELLOW]
    assert 1 <= len(markers) <= 4
    assert all(int(grid[r, c]) in (track_color, YELLOW) for r, c in border)
    return border, markers, track_color


def main(grid: np.ndarray) -> np.ndarray:
    """Move each yellow marker clockwise along its rectangular ring by that ring's marker count.

    Input structure:
    - The background is black.
    - Each object is a one-pixel-wide rectangular border track made of a single non-yellow color.
    - Some border pixels are replaced by yellow markers; each ring has between one and four such markers.

    Rule:
    - Treat every 4-connected non-black component as one rectangular ring track.
    - Reconstruct that ring's clockwise border order: top edge left-to-right, right edge top-to-bottom, bottom edge right-to-left, left edge bottom-to-top.
    - If a ring has N yellow markers, move every yellow marker N steps forward in that clockwise order.
    - Restore the vacated border pixels to the ring color and paint the destination border pixels yellow.

    Key insight:
    - The yellow pixels do not move independently on the whole grid; each one belongs to a specific rectangular perimeter, and the step size is the number of yellow markers on that same perimeter.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for coords in _components(g != BACKGROUND):
        border, markers, track_color = _analyze_track(g, coords)
        step = len(markers)

        for r, c in border:
            out[r, c] = track_color
        for marker_index in markers:
            r, c = border[(marker_index + step) % len(border)]
            out[r, c] = YELLOW

    return out


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_rect(rect: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(GRID_LIMIT - 1, r1 + pad),
        min(GRID_LIMIT - 1, c1 + pad),
    )


def _sample_marker_counts(rng: np.random.Generator, count: int) -> list[int]:
    if count == 1:
        return [int(rng.integers(1, 5))]

    for _ in range(50):
        values = [int(rng.integers(1, 5)) for _ in range(count)]
        if len(set(values)) >= 2:
            return values

    return [1, 2] + [int(rng.integers(1, 5)) for _ in range(count - 2)]


def _sample_disjoint_rect(
    rng: np.random.Generator,
    reserved: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    for _ in range(LAYOUT_RETRIES):
        height = int(rng.integers(3, 9))
        width = int(rng.integers(3, 9))
        top = int(rng.integers(0, GRID_LIMIT - height + 1))
        left = int(rng.integers(0, GRID_LIMIT - width + 1))
        rect = (top, left, top + height - 1, left + width - 1)
        padded = _expand_rect(rect, 1)
        if all(not _rectangles_overlap(padded, other) for other in reserved):
            return rect
    return None


def _sample_nested_pair(
    rng: np.random.Generator,
    reserved: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    for _ in range(LAYOUT_RETRIES):
        outer_height = int(rng.integers(7, 9))
        outer_width = int(rng.integers(7, 9))
        top = int(rng.integers(0, GRID_LIMIT - outer_height + 1))
        left = int(rng.integers(0, GRID_LIMIT - outer_width + 1))
        outer = (top, left, top + outer_height - 1, left + outer_width - 1)
        outer_padded = _expand_rect(outer, 1)
        if any(_rectangles_overlap(outer_padded, other) for other in reserved):
            continue

        inner_height = int(rng.integers(3, outer_height - 3))
        inner_width = int(rng.integers(3, outer_width - 3))
        inner_top = int(rng.integers(top + 2, top + outer_height - inner_height - 1))
        inner_left = int(rng.integers(left + 2, left + outer_width - inner_width - 1))
        inner = (
            inner_top,
            inner_left,
            inner_top + inner_height - 1,
            inner_left + inner_width - 1,
        )
        return outer, inner
    return None


def _sample_layout(rng: np.random.Generator) -> list[tuple[int, int, int, int]] | None:
    nested_layout = bool(rng.integers(0, 2))
    total_tracks = int(rng.integers(2, 5))
    reserved: list[tuple[int, int, int, int]] = []
    rects: list[tuple[int, int, int, int]] = []

    if nested_layout:
        nested_pair = _sample_nested_pair(rng, reserved)
        if nested_pair is None:
            return None
        outer, inner = nested_pair
        rects.extend([outer, inner])
        reserved.append(_expand_rect(outer, 1))
        remaining = total_tracks - 2
    else:
        remaining = total_tracks

    for _ in range(remaining):
        rect = _sample_disjoint_rect(rng, reserved)
        if rect is None:
            return None
        rects.append(rect)
        reserved.append(_expand_rect(rect, 1))

    rng.shuffle(rects)
    return rects


def _draw_ring(grid: np.ndarray, rect: tuple[int, int, int, int], color: int, markers: np.ndarray) -> None:
    r0, c0, r1, c1 = rect
    border = _border_coords(r0, c0, r1, c1)
    for r, c in border:
        grid[r, c] = color
    for index in markers:
        r, c = border[int(index)]
        grid[r, c] = YELLOW


def _crop_with_random_margin(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    coords = np.argwhere(grid != BACKGROUND)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

    top_extra = int(rng.integers(0, r0 + 1))
    left_extra = int(rng.integers(0, c0 + 1))
    bottom_extra = int(rng.integers(0, GRID_LIMIT - r1))
    right_extra = int(rng.integers(0, GRID_LIMIT - c1))
    return grid[
        r0 - top_extra : r1 + bottom_extra + 1,
        c0 - left_extra : c1 + right_extra + 1,
    ]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    rects = _sample_layout(rng)
    if rects is None:
        return None

    colors = rng.choice(TRACK_COLORS, size=len(rects), replace=False)
    marker_counts = _sample_marker_counts(rng, len(rects))

    grid = np.zeros((GRID_LIMIT, GRID_LIMIT), dtype=int)
    for rect, color, marker_count in zip(rects, colors, marker_counts):
        border_len = len(_border_coords(*rect))
        marker_indices = rng.choice(border_len, size=marker_count, replace=False)
        _draw_ring(grid, rect, int(color), marker_indices)

    cropped = _crop_with_random_margin(grid, rng)
    try:
        output = main(cropped)
    except Exception:
        return None

    if np.array_equal(cropped, output):
        return None

    return cropped


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        sample = _gen_attempt(_RNG)
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid ring-shift sample")
