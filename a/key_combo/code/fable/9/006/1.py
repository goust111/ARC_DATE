from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 200

INVALID_FOUR_PATTERNS = [
    [(0, 0), (0, 4), (2, 2), (4, 0)],
    [(0, 0), (2, 2), (4, 0), (4, 4)],
    [(0, 0), (0, 4), (4, 2), (6, 0)],
]
OTHER_NOISE_PATTERNS = {
    1: [[(0, 0)]],
    2: [[(0, 0), (0, 2)], [(0, 0), (2, 0)], [(0, 0), (2, 2)]],
    3: [[(0, 0), (0, 2), (2, 0)], [(0, 0), (2, 0), (4, 0)], [(0, 0), (2, 2), (4, 0)]],
    5: [[(0, 0), (0, 4), (2, 2), (4, 0), (4, 4)]],
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(set(map(int, grid.ravel())) - {int(bg)})


def _markers_for_rect(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return {
        (r0 - 1, c0 - 1),
        (r0 - 1, c1 + 1),
        (r1 + 1, c0 - 1),
        (r1 + 1, c1 + 1),
    }


def _point_in_rect(point: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    r, c = point
    r0, c0, r1, c1 = rect
    return r0 <= r <= r1 and c0 <= c <= c1


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _chebyshev_clear(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for rr, cc in occupied:
            if max(abs(r - rr), abs(c - cc)) <= 1:
                return False
    return True


def _legal_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    rects: list[tuple[int, int, int, int, int]] = []

    for color in _colors_present(g, bg):
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        if len(coords) != 4:
            continue

        rows = sorted({r for r, _ in coords})
        cols = sorted({c for _, c in coords})
        if len(rows) != 2 or len(cols) != 2:
            continue

        if set(coords) != {(rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[0]), (rows[1], cols[1])}:
            continue
        if rows[1] - rows[0] < 2 or cols[1] - cols[0] < 2:
            continue

        rects.append((color, rows[0] + 1, cols[0] + 1, rows[1] - 1, cols[1] - 1))

    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """Recover invisible rectangles from diagonal corner markers, then cancel every overlapping fill back to background.

    Input structure:
    - The grid is mostly black background with scattered colored singleton pixels.
    - Some colors appear exactly four times, occupying two rows and two columns as the four outward diagonal markers of an unseen rectangle.
    - Other colors have the wrong pixel count or the wrong four-point layout and act as distractor groups.

    Rule:
    - For each legal four-marker color group, shift one step inward diagonally from the markers to obtain the rectangle corners, then fill that whole solid rectangle with the same color.
    - After all rectangles are reconstructed simultaneously, any cell covered by two or more rectangles becomes background again.
    - Keep the original corner-marker pixels and all distractor pixels unchanged.

    Key insight:
    - The visible pixels are not the rectangle corners themselves: each one sits one diagonal step outside the true corner, so using the visible bbox directly produces a rectangle that is too large.
    - Overlap cancellation depends on total coverage count across all reconstructed rectangles, so the fills must be combined before deciding final colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    rects = _legal_rectangles(g)

    cover_count = np.zeros(g.shape, dtype=int)
    single_color = np.zeros(g.shape, dtype=int)
    for color, r0, c0, r1, c1 in rects:
        cover_count[r0 : r1 + 1, c0 : c1 + 1] += 1
        single_color[r0 : r1 + 1, c0 : c1 + 1] = color

    out = g.copy()
    fillable = g == bg
    out[fillable & (cover_count == 1)] = single_color[fillable & (cover_count == 1)]
    out[fillable & (cover_count >= 2)] = bg
    return out


def _transform_pattern(pattern: list[tuple[int, int]], rng: np.random.Generator) -> list[tuple[int, int]]:
    arr = np.array(pattern, dtype=int)
    if bool(rng.integers(0, 2)):
        arr = arr[:, ::-1]
    if bool(rng.integers(0, 2)):
        arr[:, 0] = arr[:, 0].max() - arr[:, 0]
    if bool(rng.integers(0, 2)):
        arr[:, 1] = arr[:, 1].max() - arr[:, 1]
    arr -= arr.min(axis=0)
    return [tuple(map(int, rc)) for rc in arr]


def _paint_points(grid: np.ndarray, color: int, points: set[tuple[int, int]]) -> None:
    for r, c in points:
        grid[r, c] = color


def _cover_count(shape: tuple[int, int], rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    cover = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in rects:
        cover[r0 : r1 + 1, c0 : c1 + 1] += 1
    return cover


def _sample_nonoverlap_rect(
    rng: np.random.Generator,
    shape: tuple[int, int],
    rects: list[tuple[int, int, int, int]],
    occupied: set[tuple[int, int]],
    *,
    size_low: int,
    size_high: int,
) -> tuple[int, int, int, int] | None:
    H, W = shape
    for _ in range(PLACEMENT_RETRIES):
        h = int(rng.integers(size_low, size_high + 1))
        w = int(rng.integers(size_low, size_high + 1))
        if h > H - 2 or w > W - 2:
            continue

        r0 = int(rng.integers(1, H - h))
        c0 = int(rng.integers(1, W - w))
        rect = (r0, c0, r0 + h - 1, c0 + w - 1)
        markers = _markers_for_rect(rect)

        if not _chebyshev_clear(markers, occupied):
            continue
        if any(_point_in_rect(p, rect) for p in occupied):
            continue
        if any(_rectangles_overlap(rect, prev) for prev in rects):
            continue
        if any(any(_point_in_rect(m, prev) for m in markers) for prev in rects):
            continue
        return rect
    return None


def _sample_overlap_pair(
    rng: np.random.Generator, shape: tuple[int, int]
) -> tuple[list[tuple[int, int, int, int]], set[tuple[int, int]]] | None:
    H, W = shape
    for _ in range(PLACEMENT_RETRIES):
        axis = "horizontal" if bool(rng.integers(0, 2)) else "vertical"
        h = int(rng.integers(3, 8))
        w = int(rng.integers(3, 8))
        if h > H - 2 or w > W - 2:
            continue

        if axis == "horizontal":
            shifts = [s for s in range(2, w) if W - w - s >= 2]
            if not shifts:
                continue
            shift = int(rng.choice(shifts))
            r0 = int(rng.integers(1, H - h))
            c0 = int(rng.integers(1, W - w - shift))
            a = (r0, c0, r0 + h - 1, c0 + w - 1)
            b = (r0, c0 + shift, r0 + h - 1, c0 + shift + w - 1)
        else:
            shifts = [s for s in range(2, h) if H - h - s >= 2]
            if not shifts:
                continue
            shift = int(rng.choice(shifts))
            r0 = int(rng.integers(1, H - h - shift))
            c0 = int(rng.integers(1, W - w))
            a = (r0, c0, r0 + h - 1, c0 + w - 1)
            b = (r0 + shift, c0, r0 + shift + h - 1, c0 + w - 1)

        ma = _markers_for_rect(a)
        mb = _markers_for_rect(b)
        if not _chebyshev_clear(ma, mb):
            continue
        if any(_point_in_rect(p, a) for p in mb) or any(_point_in_rect(p, b) for p in ma):
            continue

        cover = _cover_count(shape, [a, b])
        overlap_area = int(np.count_nonzero(cover >= 2))
        if overlap_area == 0:
            continue
        if overlap_area > max(6, min(a[2] - a[0] + 1, a[3] - a[1] + 1)):
            continue
        if not np.any(cover[a[0] : a[2] + 1, a[1] : a[3] + 1] == 1):
            continue
        if not np.any(cover[b[0] : b[2] + 1, b[1] : b[3] + 1] == 1):
            continue
        return [a, b], ma | mb
    return None


def _cell_allowed(
    point: tuple[int, int],
    shape: tuple[int, int],
    rects: list[tuple[int, int, int, int]],
    occupied: set[tuple[int, int]],
) -> bool:
    H, W = shape
    r, c = point
    if not (0 <= r < H and 0 <= c < W):
        return False
    if any(_point_in_rect(point, rect) for rect in rects):
        return False
    return _chebyshev_clear({point}, occupied)


def _place_noise_pattern(
    rng: np.random.Generator,
    grid: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    occupied: set[tuple[int, int]],
    color: int,
    pattern: list[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    H, W = grid.shape
    coords = _transform_pattern(pattern, rng)
    pr = max(r for r, _ in coords)
    pc = max(c for _, c in coords)

    for _ in range(PLACEMENT_RETRIES):
        r0 = int(rng.integers(0, H - pr))
        c0 = int(rng.integers(0, W - pc))
        placed = {(r0 + r, c0 + c) for r, c in coords}
        if all(_cell_allowed(p, grid.shape, rects, occupied | (placed - {p})) for p in placed):
            _paint_points(grid, color, placed)
            return placed
    return None


def _place_groups(
    rng: np.random.Generator, shape: tuple[int, int], want_overlap: bool
) -> tuple[list[tuple[int, int, int, int, int]], set[tuple[int, int]]] | None:
    n_groups = int(rng.integers(3, 5)) if want_overlap else int(rng.integers(2, 5))
    colors = rng.choice(PALETTE, size=n_groups, replace=False).tolist()

    rects: list[tuple[int, int, int, int]] = []
    occupied: set[tuple[int, int]] = set()

    if want_overlap:
        pair = _sample_overlap_pair(rng, shape)
        if pair is None:
            return None
        base_rects, occupied = pair
        rects.extend(base_rects)

    while len(rects) < n_groups:
        rect = _sample_nonoverlap_rect(
            rng,
            shape,
            rects,
            occupied,
            size_low=1,
            size_high=6,
        )
        if rect is None:
            return None
        rects.append(rect)
        occupied |= _markers_for_rect(rect)

    groups = [(color, *rect) for color, rect in zip(colors, rects)]
    return groups, occupied


def _noise_specs(rng: np.random.Generator, n_colors: int) -> list[tuple[str, list[tuple[int, int]]]]:
    specs: list[tuple[str, list[tuple[int, int]]]] = []
    if bool(rng.random() < 0.75):
        specs.append(("invalid4", INVALID_FOUR_PATTERNS[int(rng.integers(0, len(INVALID_FOUR_PATTERNS)))]))

    extra_count = int(rng.integers(0, 3))
    extra_choices = [1, 2, 3, 5]
    for _ in range(extra_count):
        n = int(extra_choices[int(rng.integers(0, len(extra_choices)))])
        specs.append((f"noise{n}", OTHER_NOISE_PATTERNS[n][int(rng.integers(0, len(OTHER_NOISE_PATTERNS[n])))]))

    return specs[:n_colors]


def _attempt_generate() -> np.ndarray | None:
    rng = np.random.default_rng()
    want_overlap = bool(rng.random() < 0.6)

    if want_overlap:
        H = int(rng.integers(14, 20))
        W = int(rng.integers(14, 20))
    else:
        H = int(rng.integers(16, 27))
        W = int(rng.integers(16, 27))

    groups_result = _place_groups(rng, (H, W), want_overlap)
    if groups_result is None:
        return None
    groups, occupied = groups_result

    noise_color_count = int(rng.integers(1, 4))
    if len(groups) + noise_color_count > len(PALETTE):
        return None

    used_colors = {color for color, *_ in groups}
    free_colors = [c for c in PALETTE if c not in used_colors]
    noise_colors = rng.choice(free_colors, size=noise_color_count, replace=False).tolist()
    noise_specs = _noise_specs(rng, noise_color_count)
    if not noise_specs:
        noise_specs = [("noise1", [(0, 0)])]

    rects = [(r0, c0, r1, c1) for _, r0, c0, r1, c1 in groups]
    grid = np.zeros((H, W), dtype=int)

    for color, r0, c0, r1, c1 in groups:
        _paint_points(grid, color, _markers_for_rect((r0, c0, r1, c1)))

    occupied_pixels = set(occupied)
    for color, (_, pattern) in zip(noise_colors, noise_specs):
        placed = _place_noise_pattern(rng, grid, rects, occupied_pixels, color, pattern)
        if placed is None:
            return None
        occupied_pixels |= placed

    legal = _legal_rectangles(grid)
    if sorted(legal) != sorted(groups):
        return None

    cover = _cover_count(grid.shape, rects)
    overlap_area = int(np.count_nonzero(cover >= 2))
    if want_overlap != (overlap_area > 0):
        return None
    if want_overlap and overlap_area > max(6, int(np.count_nonzero(cover > 0) * 0.3)):
        return None

    if any(not np.any(cover[r0 : r1 + 1, c0 : c1 + 1] == 1) for _, r0, c0, r1, c1 in groups):
        return None

    invalid_four_present = any(len(np.argwhere(grid == color)) == 4 for color in noise_colors)
    if noise_specs and noise_specs[0][0] == "invalid4" and not invalid_four_present:
        return None

    out = main(grid)
    assert out.shape == grid.shape
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
