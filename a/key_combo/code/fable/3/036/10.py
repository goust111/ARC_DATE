from __future__ import annotations

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(grid[r0, c0]) == bg:
                continue
            color = int(grid[r0, c0])
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for rr, cc in _neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and int(grid[rr, cc]) == color:
                        seen[rr, cc] = True
                        stack.append((rr, cc))

            arr = np.array(coords, dtype=int)
            r_min, c_min = arr.min(axis=0)
            r_max, c_max = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r_min), int(c_min), int(r_max), int(c_max)),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _bbox_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def _bbox_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def _filled_rectangles(grid: np.ndarray, bg: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    rectangles: list[tuple[int, tuple[int, int, int, int]]] = []
    for comp in _find_components(grid, bg):
        box = comp["bbox"]
        height = _bbox_h(box)
        width = _bbox_w(box)
        color = int(comp["color"])
        if height < 3 or width < 3:
            continue
        if int(comp["area"]) != height * width:
            continue
        r0, c0, r1, c1 = box
        if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
            rectangles.append((color, box))
    return rectangles


def _overlap_groups(intervals: list[tuple[int, int, int]]) -> tuple[dict[int, int], int]:
    group_items: list[list[int]] = []
    common_los: list[int] = []
    common_his: list[int] = []
    center_sums: list[float] = []
    counts: list[int] = []

    for idx, lo, hi in sorted(intervals, key=lambda item: (item[1], item[2], item[0])):
        matches = [
            group_idx
            for group_idx in range(len(group_items))
            if max(common_los[group_idx], lo) <= min(common_his[group_idx], hi)
        ]
        assert len(matches) <= 1, "each interval must belong to at most one overlap band"
        if matches:
            group_idx = matches[0]
            group_items[group_idx].append(idx)
            common_los[group_idx] = max(common_los[group_idx], lo)
            common_his[group_idx] = min(common_his[group_idx], hi)
            center_sums[group_idx] += (lo + hi) / 2
            counts[group_idx] += 1
        else:
            group_items.append([idx])
            common_los.append(lo)
            common_his.append(hi)
            center_sums.append((lo + hi) / 2)
            counts.append(1)

    assert group_items, "expected at least one overlap band"
    ordered = sorted(
        range(len(group_items)),
        key=lambda group_idx: center_sums[group_idx] / counts[group_idx],
    )
    mapping: dict[int, int] = {}
    for rank, group_idx in enumerate(ordered):
        for idx in group_items[group_idx]:
            mapping[idx] = rank
    return mapping, len(group_items)


def _latin_solutions(partial: np.ndarray, colors: list[int], limit: int = 2) -> list[np.ndarray]:
    grid = np.array(partial, dtype=int, copy=True)
    n = int(grid.shape[0])
    color_list = [int(color) for color in colors]
    full_set = set(color_list)
    solutions: list[np.ndarray] = []

    for r in range(n):
        vals = [int(v) for v in grid[r] if int(v) >= 0]
        assert len(vals) == len(set(vals)), "observed row clues must not repeat colors"
    for c in range(n):
        vals = [int(v) for v in grid[:, c] if int(v) >= 0]
        assert len(vals) == len(set(vals)), "observed column clues must not repeat colors"

    def search() -> None:
        best_pos: tuple[int, int] | None = None
        best_cands: list[int] | None = None

        for r in range(n):
            row_vals = {int(v) for v in grid[r] if int(v) >= 0}
            for c in range(n):
                if int(grid[r, c]) >= 0:
                    continue
                col_vals = {int(v) for v in grid[:, c] if int(v) >= 0}
                cands = [color for color in color_list if color not in row_vals and color not in col_vals]
                if not cands:
                    return
                if best_cands is None or len(cands) < len(best_cands):
                    best_pos = (r, c)
                    best_cands = cands
                    if len(best_cands) == 1:
                        break
            if best_cands is not None and len(best_cands) == 1:
                break

        if best_pos is None:
            for r in range(n):
                if {int(v) for v in grid[r]} != full_set:
                    return
            for c in range(n):
                if {int(v) for v in grid[:, c]} != full_set:
                    return
            solutions.append(grid.copy())
            return

        r, c = best_pos
        assert best_cands is not None
        for color in best_cands:
            grid[r, c] = color
            search()
            if len(solutions) >= limit:
                break
            grid[r, c] = -1
        grid[r, c] = -1

    search()
    return solutions


def _solve_latin_unique(partial: np.ndarray, colors: list[int]) -> np.ndarray:
    solutions = _latin_solutions(partial, colors, limit=2)
    assert len(solutions) == 1, "compressed table must have a unique Latin completion"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the rectangle lattice and fill its missing entries by Latin row-and-column constraints.

    Input structure
    - A dominant background contains scattered clutter pixels or small non-rectangular blobs.
    - The meaningful objects are monochromatic filled rectangles, each at least 3x3, arranged on an n x n row-column lattice with a few empty lattice slots.
    - Rectangles in the same lattice row share overlapping vertical spans, and rectangles in the same lattice column share overlapping horizontal spans.

    Rule
    - Ignore every foreground component that is not a filled 3x3-or-larger rectangle.
    - Group rectangle bbox spans into lattice rows and columns, then collapse the observed rectangles into a compact color table with blanks where lattice slots are missing.
    - Fill the blanks so that every row and every column contains each rectangle color exactly once, and return that completed n x n table.

    Key insight
    - The large rectangles define a compressed partially observed Latin square; the missing colors are determined only by intersecting row and column exclusions after clutter is discarded.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    rectangles = _filled_rectangles(g, bg)
    assert rectangles, "expected at least one filled rectangle"

    colors = sorted({int(color) for color, _box in rectangles})
    n = len(colors)
    row_by_rect, row_count = _overlap_groups(
        [(idx, box[0], box[2]) for idx, (_color, box) in enumerate(rectangles)]
    )
    col_by_rect, col_count = _overlap_groups(
        [(idx, box[1], box[3]) for idx, (_color, box) in enumerate(rectangles)]
    )
    assert row_count == col_count == n, "rectangle bands and color count must agree on lattice size"

    partial = np.full((n, n), -1, dtype=int)
    for idx, (color, _box) in enumerate(rectangles):
        r = row_by_rect[idx]
        c = col_by_rect[idx]
        assert int(partial[r, c]) == -1, "each lattice slot can contain at most one rectangle"
        partial[r, c] = int(color)

    for r in range(n):
        vals = [int(v) for v in partial[r] if int(v) >= 0]
        assert len(vals) == len(set(vals)), "row clues must not repeat a color"
    for c in range(n):
        vals = [int(v) for v in partial[:, c] if int(v) >= 0]
        assert len(vals) == len(set(vals)), "column clues must not repeat a color"

    return _solve_latin_unique(partial, colors)


# --------------- generator ---------------

_PALETTE = tuple(range(10))
_CLUTTER_BASE_SHAPES = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
)
_GEN_RETRIES = 400


def _sample_band_extents(
    rng: np.random.Generator,
    count: int,
    total_size: int,
    min_len: int = 3,
    max_len: int = 6,
    min_gap: int = 1,
) -> list[tuple[int, int]]:
    available = total_size - min_gap * (count - 1)
    assert available >= count * min_len

    lengths: list[int] = []
    used = 0
    for idx in range(count):
        remaining = count - idx - 1
        hi = min(max_len, available - used - remaining * min_len)
        assert hi >= min_len
        length = int(rng.integers(min_len, hi + 1))
        lengths.append(length)
        used += length

    slack = available - sum(lengths)
    extras = rng.multinomial(slack, [1 / (count + 1)] * (count + 1)).tolist()

    pos = extras[0]
    extents: list[tuple[int, int]] = []
    for idx, length in enumerate(lengths):
        start = pos
        end = start + length - 1
        extents.append((start, end))
        pos = end + 1
        if idx < count - 1:
            pos += min_gap + extras[idx + 1]
    return extents


def _sample_intervals_in_extent(
    rng: np.random.Generator,
    extent: tuple[int, int],
    count: int,
    min_len: int = 3,
    max_len: int = 6,
) -> list[tuple[int, int]]:
    start, end = extent
    span = end - start + 1
    common = int(rng.integers(start, end + 1))
    intervals: list[tuple[int, int]] = []
    for _ in range(count):
        length = int(rng.integers(min_len, min(max_len, span) + 1))
        lo_min = max(start, common - length + 1)
        lo_max = min(common, end - length + 1)
        lo = int(rng.integers(lo_min, lo_max + 1))
        intervals.append((lo, lo + length - 1))
    return intervals


def _latin_square(rng: np.random.Generator, colors: list[int]) -> np.ndarray:
    n = len(colors)
    row_offsets = rng.permutation(n)
    col_offsets = rng.permutation(n)
    symbols = np.array(colors, dtype=int)
    out = np.empty((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            out[r, c] = int(symbols[(int(row_offsets[r]) + int(col_offsets[c])) % n])
    return out


def _carve_unique_holes(rng: np.random.Generator, solution: np.ndarray) -> np.ndarray | None:
    n = int(solution.shape[0])
    colors = sorted(int(color) for color in np.unique(solution))
    keep = np.ones((n, n), dtype=bool)
    row_counts = np.full(n, n, dtype=int)
    col_counts = np.full(n, n, dtype=int)
    color_counts = {color: n for color in colors}
    target_holes = int(rng.integers(2, min(n + 3, n * n) + 1))
    holes = 0

    while holes < target_holes:
        removable: list[tuple[int, int]] = []
        coords = rng.permutation(n * n)
        for flat in coords.tolist():
            r = int(flat // n)
            c = int(flat % n)
            if not keep[r, c]:
                continue
            color = int(solution[r, c])
            if row_counts[r] <= 1 or col_counts[c] <= 1 or color_counts[color] <= 1:
                continue
            trial = keep.copy()
            trial[r, c] = False
            partial = np.where(trial, solution, -1)
            if len(_latin_solutions(partial, colors, limit=2)) == 1:
                removable.append((r, c))
        if not removable:
            break
        r, c = removable[int(rng.integers(len(removable)))]
        color = int(solution[r, c])
        keep[r, c] = False
        row_counts[r] -= 1
        col_counts[c] -= 1
        color_counts[color] -= 1
        holes += 1

    if holes < 2:
        return None
    return keep


def _random_shape_variant(rng: np.random.Generator) -> np.ndarray:
    shape = _CLUTTER_BASE_SHAPES[int(rng.integers(len(_CLUTTER_BASE_SHAPES)))]
    variant = np.array(shape, copy=True)
    if bool(int(rng.integers(2))):
        variant = np.flipud(variant)
    if bool(int(rng.integers(2))):
        variant = np.fliplr(variant)
    return np.rot90(variant, int(rng.integers(4)))


def _can_place_shape(grid: np.ndarray, shape: np.ndarray, r0: int, c0: int, bg: int) -> bool:
    h, w = map(int, grid.shape)
    coords = {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(shape)}
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != bg:
            return False
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if (rr, cc) in coords:
                    continue
                if int(grid[rr, cc]) != bg:
                    return False
    return True


def _place_shape(
    rng: np.random.Generator,
    grid: np.ndarray,
    shape: np.ndarray,
    color: int,
    bg: int,
    attempts: int = 50,
) -> bool:
    gh, gw = map(int, grid.shape)
    sh, sw = map(int, shape.shape)
    if sh > gh or sw > gw:
        return False
    for _ in range(attempts):
        r0 = int(rng.integers(0, gh - sh + 1))
        c0 = int(rng.integers(0, gw - sw + 1))
        if _can_place_shape(grid, shape, r0, c0, bg):
            patch = grid[r0 : r0 + sh, c0 : c0 + sw]
            patch[shape] = int(color)
            return True
    return False


def _add_clutter(
    rng: np.random.Generator,
    grid: np.ndarray,
    bg: int,
    rect_colors: list[int],
) -> bool:
    clutter_palette = [color for color in _PALETTE if color != bg and color not in rect_colors]
    assert clutter_palette
    rng.shuffle(clutter_palette)
    clutter_colors = clutter_palette[: int(rng.integers(1, min(4, len(clutter_palette)) + 1))]
    target = int(rng.integers(6, 15))
    placed = 0
    for _ in range(target * 12):
        if placed >= target:
            break
        shape = _random_shape_variant(rng)
        color = int(clutter_colors[int(rng.integers(len(clutter_colors)))])
        if _place_shape(rng, grid, shape, color, bg):
            placed += 1
    return placed >= 4


def _background_is_dominant(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    count_by_color = {int(v): int(c) for v, c in zip(values.tolist(), counts.tolist(), strict=False)}
    bg_count = count_by_color.get(bg, 0)
    fg_counts = [count for color, count in count_by_color.items() if color != bg]
    return bool(fg_counts) and bg_count > max(fg_counts)


def _generate_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.integers(3, 6))
    bg = int(rng.integers(10))
    available = [color for color in _PALETTE if color != bg]
    rng.shuffle(available)
    rect_colors = [int(color) for color in available[:n]]

    solution = _latin_square(rng, rect_colors)
    keep = _carve_unique_holes(rng, solution)
    if keep is None:
        return None

    min_axis = n * 3 + (n - 1)
    height = int(rng.integers(min_axis + 2, 31))
    width = int(rng.integers(min_axis + 2, 31))
    row_bands = _sample_band_extents(rng, n, height)
    col_bands = _sample_band_extents(rng, n, width)
    row_intervals = [_sample_intervals_in_extent(rng, extent, n) for extent in row_bands]
    col_intervals = [_sample_intervals_in_extent(rng, extent, n) for extent in col_bands]

    grid = np.full((height, width), bg, dtype=int)
    for r in range(n):
        for c in range(n):
            if not keep[r, c]:
                continue
            r0, r1 = row_intervals[r][c]
            c0, c1 = col_intervals[c][r]
            grid[r0 : r1 + 1, c0 : c1 + 1] = int(solution[r, c])

    if not _add_clutter(rng, grid, bg, rect_colors):
        return None
    if not _background_is_dominant(grid, bg):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, solution):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid Latin-compression input")
