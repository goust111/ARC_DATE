from __future__ import annotations

from collections import Counter
import random

import numpy as np


EMPTY = -1
PALETTE = tuple(range(10))
CLUTTER_SHAPES = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 1]], dtype=bool),
)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def _bbox_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_monochrome_components(
    grid: np.ndarray, bg: int, min_area: int = 1
) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict[str, object]] = []
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
                for nr, nc in _neighbors4(r, c):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            if len(coords) < min_area:
                continue
            arr = np.array(coords, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _overlap_groups(
    intervals: list[tuple[int, int, int]]
) -> tuple[dict[int, int], int]:
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
        assert len(matches) <= 1
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

    assert group_items
    ordered = sorted(
        range(len(group_items)),
        key=lambda group_idx: center_sums[group_idx] / counts[group_idx],
    )
    index_to_rank: dict[int, int] = {}
    for rank, group_idx in enumerate(ordered):
        for idx in group_items[group_idx]:
            index_to_rank[idx] = rank
    return index_to_rank, len(group_items)


def _filled_rectangles(grid: np.ndarray) -> list[tuple[int, tuple[int, int, int, int]]]:
    bg = _get_bg(grid)
    rectangles: list[tuple[int, tuple[int, int, int, int]]] = []
    for comp in _find_monochrome_components(grid, bg=bg, min_area=9):
        box = comp["bbox"]
        height = _bbox_h(box)
        width = _bbox_w(box)
        if height >= 3 and width >= 3 and int(comp["area"]) == height * width:
            rectangles.append((int(comp["color"]), box))
    return rectangles


def _compress_wallpaper(grid: np.ndarray) -> np.ndarray:
    rectangles = _filled_rectangles(grid)
    assert rectangles

    row_by_rect, row_count = _overlap_groups(
        [(idx, box[0], box[2]) for idx, (_color, box) in enumerate(rectangles)]
    )
    col_by_rect, col_count = _overlap_groups(
        [(idx, box[1], box[3]) for idx, (_color, box) in enumerate(rectangles)]
    )

    partial = np.full((row_count, col_count), EMPTY, dtype=int)
    for idx, (color, _box) in enumerate(rectangles):
        row = row_by_rect[idx]
        col = col_by_rect[idx]
        assert partial[row, col] == EMPTY
        partial[row, col] = color
    return partial


def _periodic_fill(tile: np.ndarray, rows: int, cols: int) -> np.ndarray:
    rr = np.arange(rows)[:, None] % tile.shape[0]
    cc = np.arange(cols)[None, :] % tile.shape[1]
    return tile[rr, cc]


def _candidate_completions(
    partial: np.ndarray,
    allowed_shapes: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int, int, np.ndarray, np.ndarray]]:
    rows, cols = partial.shape
    known = partial != EMPTY
    known_coords = np.argwhere(known)
    candidates: list[tuple[int, int, int, np.ndarray, np.ndarray]] = []

    if allowed_shapes is None:
        allowed_shapes = [(th, tw) for th in range(1, rows + 1) for tw in range(1, cols + 1)]

    for th, tw in allowed_shapes:
        tile = np.full((th, tw), EMPTY, dtype=int)
        valid = True
        for r, c in known_coords:
            pr = int(r) % th
            pc = int(c) % tw
            value = int(partial[r, c])
            if tile[pr, pc] == EMPTY:
                tile[pr, pc] = value
            elif int(tile[pr, pc]) != value:
                valid = False
                break
        if not valid or np.any(tile == EMPTY):
            continue
        full = _periodic_fill(tile, rows, cols)
        candidates.append((th * tw, th, tw, tile, full))
    return candidates


def _infer_unique_completion(partial: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    candidates = _candidate_completions(partial)
    assert candidates
    best_area = min(area for area, _th, _tw, _tile, _full in candidates)
    best = [
        (th, tw, tile, full)
        for area, th, tw, tile, full in candidates
        if area == best_area
    ]
    assert len(best) == 1
    th, tw, _tile, full = best[0]
    return full, (th, tw)


def _minimal_period(grid: np.ndarray) -> tuple[int, int]:
    rows, cols = grid.shape
    for th in range(1, rows + 1):
        for tw in range(1, cols + 1):
            if np.array_equal(grid, _periodic_fill(grid[:th, :tw], rows, cols)):
                return th, tw
    return rows, cols


def _unique_axis_completion(partial: np.ndarray, axis: str) -> bool:
    rows, cols = partial.shape
    if axis == "row":
        shapes = [(rows, tw) for tw in range(1, cols)]
    else:
        shapes = [(th, cols) for th in range(1, rows)]
    candidates = _candidate_completions(partial, shapes)
    if not candidates:
        return False
    outputs: set[tuple[bytes, tuple[int, int]]] = set()
    for _area, _th, _tw, _tile, full in candidates:
        outputs.add((full.tobytes(), full.shape))
    return len(outputs) == 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the rectangle lattice into a small color table and complete its missing cells by the table's own minimal wallpaper period.

    Input structure:
    - The dominant background contains clutter plus several monochrome filled rectangles of size at least 3x3.
    - These large rectangles form a row-column lattice: rectangles in the same lattice row have overlapping vertical bbox intervals, and rectangles in the same lattice column have overlapping horizontal bbox intervals.
    - Some lattice positions are missing, so no large rectangle appears there.

    Rule:
    - Ignore every foreground component that is not a filled 3x3-or-larger rectangle.
    - Group the remaining rectangles into lattice rows and columns by overlapping bbox intervals, then collapse them into a partial color table with blanks where rectangles are missing.
    - Treat those blanks as missing data, infer the unique minimal repeating tile consistent with the visible table cells, and fill the whole table by that period.

    Key insight:
    - The visible rectangles only reveal part of the answer: first compress the big-object lattice, then solve the smaller compressed table itself as an incomplete periodic wallpaper.
    """
    g = np.asarray(grid, dtype=int)
    partial = _compress_wallpaper(g)
    full, _period = _infer_unique_completion(partial)
    return full


def _build_full_table(bg: int) -> tuple[np.ndarray, tuple[int, int]]:
    for _ in range(400):
        tile_h = random.choice((2, 3))
        tile_w = random.choice((2, 3))
        row_repeat = 2 if tile_h == 3 else random.choice((2, 3))
        col_repeat = 2 if tile_w == 3 else random.choice((2, 3))
        rows = tile_h * row_repeat
        cols = tile_w * col_repeat

        colors = [color for color in PALETTE if color != bg]
        random.shuffle(colors)
        color_count = random.randint(3, min(len(colors), tile_h * tile_w))
        flat = colors[:color_count]
        while len(flat) < tile_h * tile_w:
            flat.append(random.choice(flat))
        random.shuffle(flat)
        tile = np.array(flat[: tile_h * tile_w], dtype=int).reshape(tile_h, tile_w)
        if len(set(map(int, tile.ravel()))) < min(3, tile_h * tile_w):
            continue
        if _minimal_period(tile) != (tile_h, tile_w):
            continue
        full = _periodic_fill(tile, rows, cols)
        return full, (tile_h, tile_w)
    raise RuntimeError("failed to build a non-degenerate wallpaper table")


def _valid_missing_set(
    full: np.ndarray,
    period: tuple[int, int],
    missing: set[tuple[int, int]],
) -> bool:
    rows, cols = full.shape
    tile_h, tile_w = period
    if len(missing) < 4:
        return False

    partial = full.copy()
    for r, c in missing:
        partial[r, c] = EMPTY

    if np.any(np.all(partial == EMPTY, axis=1)) or np.any(np.all(partial == EMPTY, axis=0)):
        return False

    for pr in range(tile_h):
        for pc in range(tile_w):
            phase_seen = False
            for r in range(pr, rows, tile_h):
                for c in range(pc, cols, tile_w):
                    if partial[r, c] != EMPTY:
                        phase_seen = True
                        break
                if phase_seen:
                    break
            if not phase_seen:
                return False

    try:
        inferred, inferred_period = _infer_unique_completion(partial)
    except AssertionError:
        return False
    if inferred_period != period or not np.array_equal(inferred, full):
        return False

    if _unique_axis_completion(partial, "row"):
        return False
    if _unique_axis_completion(partial, "col"):
        return False

    missing_colors = {int(full[r, c]) for r, c in missing}
    return len(missing_colors) >= 2


def _choose_missing_partial(
    full: np.ndarray, period: tuple[int, int]
) -> np.ndarray:
    rows, cols = full.shape
    removable_limit = rows * cols - period[0] * period[1]
    all_positions = [(r, c) for r in range(rows) for c in range(cols)]
    target_lo = max(4, min(rows, cols))
    target_hi = min(removable_limit, rows + cols)

    for _ in range(800):
        target = random.randint(target_lo, target_hi)
        missing: set[tuple[int, int]] = set()

        row = random.randrange(rows)
        count_in_row = random.randint(2, min(cols - 1, max(2, target // 2)))
        for c in random.sample(range(cols), count_in_row):
            missing.add((row, c))

        col = random.randrange(cols)
        count_in_col = random.randint(2, min(rows - 1, max(2, target // 2)))
        for r in random.sample(range(rows), count_in_col):
            missing.add((r, col))

        pool = all_positions[:]
        random.shuffle(pool)
        for pos in pool:
            if len(missing) >= target:
                break
            if pos in missing:
                continue
            if random.random() < 0.65:
                missing.add(pos)

        if len(missing) > removable_limit:
            continue
        if not _valid_missing_set(full, period, missing):
            continue

        partial = full.copy()
        for r, c in missing:
            partial[r, c] = EMPTY
        return partial

    raise RuntimeError("failed to choose an informative missing-cell pattern")


def _sample_band_extents(count: int) -> tuple[list[tuple[int, int]], int]:
    for _ in range(400):
        lengths = [random.randint(3, 4) for _ in range(count)]
        base = sum(lengths) + (count - 1)
        if base > 30:
            continue
        slack = random.randint(0, 30 - base)
        extras = [0] * (count + 1)
        for _ in range(slack):
            extras[random.randrange(count + 1)] += 1

        extents: list[tuple[int, int]] = []
        pos = extras[0]
        for idx, length in enumerate(lengths):
            start = pos
            end = start + length - 1
            extents.append((start, end))
            pos = end + 1
            if idx < count - 1:
                pos += 1 + extras[idx + 1]
        size = extents[-1][1] + 1 + extras[-1]
        if size <= 30:
            return extents, size
    raise RuntimeError("failed to sample rectangle bands")


def _sample_intervals(
    extents: list[tuple[int, int]], samples_per_band: int
) -> list[list[tuple[int, int]]]:
    out: list[list[tuple[int, int]]] = []
    for lo, hi in extents:
        common = random.randint(lo, hi)
        span = hi - lo + 1
        band: list[tuple[int, int]] = []
        for _ in range(samples_per_band):
            length = random.randint(3, span)
            start_lo = max(lo, common - length + 1)
            start_hi = min(common, hi - length + 1)
            start = random.randint(start_lo, start_hi)
            band.append((start, start + length - 1))
        out.append(band)
    return out


def _mark_blocked(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    rows, cols = blocked.shape
    for r, c in coords:
        for nr, nc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < rows and 0 <= nc < cols:
                blocked[nr, nc] = True


def _add_clutter(grid: np.ndarray, rect_mask: np.ndarray, bg: int) -> int:
    rows, cols = grid.shape
    blocked = np.zeros((rows, cols), dtype=bool)
    rect_coords = [tuple(map(int, rc)) for rc in np.argwhere(rect_mask)]
    _mark_blocked(blocked, rect_coords)

    colors = [color for color in PALETTE if color != bg]
    target = random.randint(8, 14)
    placed = 0
    attempts = 0
    while attempts < target * 50 and placed < target:
        attempts += 1
        shape = random.choice(CLUTTER_SHAPES)
        h, w = shape.shape
        r0 = random.randint(0, rows - h)
        c0 = random.randint(0, cols - w)
        coords = [
            (r0 + dr, c0 + dc)
            for dr, dc in np.argwhere(shape)
        ]
        if any(blocked[r, c] or int(grid[r, c]) != bg for r, c in coords):
            continue
        color = random.choice(colors)
        for r, c in coords:
            grid[r, c] = color
        _mark_blocked(blocked, coords)
        placed += 1
    return placed


def _render_input(partial: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = partial.shape
    row_extents, grid_h = _sample_band_extents(rows)
    col_extents, grid_w = _sample_band_extents(cols)
    row_intervals = _sample_intervals(row_extents, cols)
    col_intervals = _sample_intervals(col_extents, rows)

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    rect_mask = np.zeros((grid_h, grid_w), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            color = int(partial[r, c])
            if color == EMPTY:
                continue
            r0, r1 = row_intervals[r][c]
            c0, c1 = col_intervals[c][r]
            grid[r0 : r1 + 1, c0 : c1 + 1] = color
            rect_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    clutter_pixels = _add_clutter(grid, rect_mask, bg)
    assert clutter_pixels >= 4
    return grid, rect_mask


def _validate_case(
    inp: np.ndarray,
    expected: np.ndarray,
    period: tuple[int, int],
) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if not (1 < period[0] < expected.shape[0] and 1 < period[1] < expected.shape[1]):
        return False

    try:
        partial = _compress_wallpaper(inp)
        actual = main(inp)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False
    if partial.shape != expected.shape:
        return False

    missing_mask = partial == EMPTY
    if np.count_nonzero(missing_mask) < 4:
        return False
    if np.any(np.all(partial == EMPTY, axis=1)) or np.any(np.all(partial == EMPTY, axis=0)):
        return False

    try:
        inferred, inferred_period = _infer_unique_completion(partial)
    except AssertionError:
        return False
    if inferred_period != period or not np.array_equal(inferred, expected):
        return False

    if _unique_axis_completion(partial, "row"):
        return False
    if _unique_axis_completion(partial, "col"):
        return False

    rectangles = _filled_rectangles(inp)
    if len(rectangles) != int(np.count_nonzero(~missing_mask)):
        return False

    bg = _get_bg(inp)
    rect_mask = np.zeros_like(inp, dtype=bool)
    for _color, box in rectangles:
        r0, c0, r1, c1 = box
        rect_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    if np.count_nonzero((inp != bg) & ~rect_mask) < 4:
        return False

    missing_colors = set(map(int, expected[missing_mask].ravel()))
    return len(missing_colors) >= 2


def generate_input() -> np.ndarray:
    for _ in range(400):
        bg = random.choice(PALETTE)
        full, period = _build_full_table(bg)
        partial = _choose_missing_partial(full, period)
        inp, _rect_mask = _render_input(partial, bg)
        if _validate_case(inp, full, period):
            return inp
    raise RuntimeError("failed to generate a valid wallpaper-rectangle ARC input")
