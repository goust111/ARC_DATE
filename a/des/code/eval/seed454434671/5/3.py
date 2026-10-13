import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5
ARC_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 400


def _mode_color(arr: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(arr, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _monochrome_components(grid: np.ndarray, skip_colors: set[int] | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    skip = set() if skip_colors is None else set(skip_colors)
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0]:
                continue
            color = int(g[r0, c0])
            if color in skip:
                seen[r0, c0] = True
                continue

            q = deque([(r0, c0)])
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for dr, dc in DIR4:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and int(g[rr, cc]) == color:
                        seen[rr, cc] = True
                        q.append((rr, cc))

            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "size": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    return components


def _find_separator(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, int, int]] = []

    for c in range(1, w - 1):
        col = g[:, c]
        if not np.all(col == col[0]):
            continue
        color = int(col[0])
        row_hits = 0
        for r in range(h):
            if np.all(g[r, :c] == color) and np.any(g[r, c + 1 :] != color):
                row_hits += 1
        if row_hits:
            candidates.append((c, color, row_hits))

    assert candidates, "expected a full-height separator column"
    candidates.sort(key=lambda item: (item[2], -abs(item[0] - w // 2)), reverse=True)
    sep_col, sep_color, _ = candidates[0]
    return int(sep_col), int(sep_color)


def _band_intervals(left: np.ndarray, separator_color: int) -> list[tuple[int, int]]:
    h = left.shape[0]
    sep_rows = [r for r in range(h) if np.all(left[r, :] == separator_color)]
    starts = [0] + [r + 1 for r in sep_rows]
    ends = [r - 1 for r in sep_rows] + [h - 1]
    bands = [(s, e) for s, e in zip(starts, ends) if s <= e]
    assert bands, "expected at least one content band"
    heights = {e - s + 1 for s, e in bands}
    assert len(heights) == 1, "bands must have equal height"
    return bands


def _stamp_assignments(right: np.ndarray, bands: list[tuple[int, int]]) -> dict[int, int]:
    bg = _mode_color(right)
    assignments: dict[int, int] = {}

    for comp in _monochrome_components(right, skip_colors={bg}):
        r0, c0, r1, c1 = comp["bbox"]
        assert comp["size"] == 4 and r1 - r0 == 1 and c1 - c0 == 1, "stamp must be a 2x2 square"
        center_row = sum(r for r, _ in comp["coords"]) / comp["size"]
        matches = [idx for idx, (b0, b1) in enumerate(bands) if b0 <= center_row <= b1]
        assert len(matches) == 1, "stamp center must match exactly one band"
        band_idx = matches[0]
        assert band_idx not in assignments, "each band can have at most one stamp"
        assignments[band_idx] = int(comp["color"])

    return assignments


def _largest_component_coords(grid: np.ndarray) -> list[tuple[int, int]]:
    components = _monochrome_components(grid)
    assert components, "band cannot be empty"
    max_size = max(comp["size"] for comp in components)
    largest = [comp for comp in components if comp["size"] == max_size]
    largest.sort(key=lambda comp: min(comp["coords"]))
    return largest[0]["coords"]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep_col, sep_color = _find_separator(g)
    left = g[:, :sep_col].copy()
    right = g[:, sep_col + 1 :]
    bands = _band_intervals(left, sep_color)
    assignments = _stamp_assignments(right, bands)

    out = left.copy()
    for band_idx, stamp_color in assignments.items():
        r0, r1 = bands[band_idx]
        band = out[r0 : r1 + 1, :]
        for rr, cc in _largest_component_coords(band):
            out[r0 + rr, cc] = stamp_color

    return out


def _partition_with_unique_largest(total: int, count: int, min_size: int) -> list[int] | None:
    spare = total - count * min_size
    if spare < 0:
        return None

    for _ in range(80):
        sizes = [min_size] * count
        for _ in range(spare):
            sizes[random.randrange(count)] += 1
        random.shuffle(sizes)
        if sizes.count(max(sizes)) == 1:
            return sizes
    return None


def _valid_boundaries(bounds: np.ndarray, width: int) -> bool:
    if bounds.size and np.any(np.abs(np.diff(bounds, axis=0)) > 1):
        return False
    for row in bounds:
        cuts = [0, *map(int, row), width]
        if any(cuts[i + 1] - cuts[i] < 1 for i in range(len(cuts) - 1)):
            return False
    return True


def _make_band(height: int, width: int, palette: list[int]) -> np.ndarray | None:
    max_components = min(4, len(palette), width // 2)
    if max_components < 3:
        return None

    for _ in range(120):
        comp_count = random.randint(3, max_components)
        widths = _partition_with_unique_largest(width, comp_count, 2)
        if widths is None:
            continue

        bounds = np.tile(np.cumsum(widths)[:-1], (height, 1)).astype(int)
        for _ in range(random.randint(1, 4)):
            if bounds.size == 0:
                break
            idx = random.randrange(bounds.shape[1])
            delta = random.choice((-1, 1))
            r0 = random.randint(0, height - 1)
            r1 = random.randint(r0, height - 1)
            trial = bounds.copy()
            trial[r0 : r1 + 1, idx] += delta
            if _valid_boundaries(trial, width):
                bounds = trial

        colors = random.sample(palette, comp_count)
        band = np.empty((height, width), dtype=int)
        for r in range(height):
            cuts = [0, *map(int, bounds[r]), width]
            for idx in range(comp_count):
                band[r, cuts[idx] : cuts[idx + 1]] = colors[idx]

        components = _monochrome_components(band)
        if len(components) != comp_count:
            continue
        sizes = [comp["size"] for comp in components]
        if sizes.count(max(sizes)) != 1:
            continue
        return band

    return None


def _build_left_half(band_count: int, band_height: int, left_width: int, palette: list[int]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    total_height = band_count * band_height + (band_count - 1)
    left = np.full((total_height, left_width), GREY, dtype=int)
    bands: list[tuple[int, int]] = []

    row = 0
    for band_idx in range(band_count):
        band = _make_band(band_height, left_width, palette)
        if band is None:
            raise RuntimeError("failed to build a valid left band")
        left[row : row + band_height, :] = band
        bands.append((row, row + band_height - 1))
        row += band_height
        if band_idx != band_count - 1:
            left[row, :] = GREY
            row += 1

    return left, bands


def _place_stamps(
    shape: tuple[int, int], bands: list[tuple[int, int]], band_indices: list[int], colors: list[int]
) -> np.ndarray:
    h, w = shape
    right = np.full((h, w), BLACK, dtype=int)
    valid_cols = list(range(1, w - 2)) if w >= 5 else list(range(0, w - 1))

    for band_idx, color in zip(band_indices, colors):
        r0, r1 = bands[band_idx]
        top = random.randint(r0, r1 - 1)
        col = random.choice(valid_cols)
        right[top : top + 2, col : col + 2] = color

    return right


def _generated_instance() -> np.ndarray | None:
    band_count = random.randint(2, 4)
    band_height = random.randint(3, 5)
    left_width = random.randint(8, 14)
    right_width = random.randint(6, 10)
    stamp_count = random.randint(1, band_count)

    colors = ARC_COLORS.copy()
    random.shuffle(colors)
    stamp_colors = colors[:stamp_count]
    remaining = colors[stamp_count:]
    palette_size = 4 if len(remaining) >= 4 and random.random() < 0.7 else 3
    left_palette = remaining[:palette_size]
    if len(left_palette) < 3:
        return None

    try:
        left, bands = _build_left_half(band_count, band_height, left_width, left_palette)
    except RuntimeError:
        return None

    stamped_bands = sorted(random.sample(range(band_count), stamp_count))
    right = _place_stamps((left.shape[0], right_width), bands, stamped_bands, stamp_colors)

    grid = np.full((left.shape[0], left_width + 1 + right_width), BLACK, dtype=int)
    grid[:, :left_width] = left
    grid[:, left_width] = GREY
    grid[:, left_width + 1 :] = right

    if np.any(left == BLACK):
        return None
    if np.any(np.isin(stamp_colors, left_palette)):
        return None

    out = main(grid)
    if out.shape != left.shape:
        return None
    if np.array_equal(out, left):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generated_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
