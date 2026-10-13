from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
FILTER_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
PATTERNS = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
]


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_filter_columns(grid: np.ndarray) -> list[tuple[int, int]]:
    cols = []
    for c in range(grid.shape[1]):
        values = set(map(int, grid[:, c]))
        if len(values) == 1:
            color = next(iter(values))
            if color not in {BG, GREY}:
                cols.append((c, color))
    assert cols, "expected at least one full monochrome filter column"
    return cols


def _find_filter_rows(grid: np.ndarray, filter_cols: list[tuple[int, int]]) -> list[tuple[int, int]]:
    col_idx = {c for c, _ in filter_cols}
    rows = []
    for r in range(grid.shape[0]):
        vals = [int(grid[r, c]) for c in range(grid.shape[1]) if c not in col_idx]
        if not vals:
            continue
        if len(set(vals)) == 1 and vals[0] not in {BG, GREY}:
            rows.append((r, vals[0]))
    assert rows, "expected at least one filter row"
    return rows


def _nearest_before(index: int, entries: list[tuple[int, int]]) -> tuple[int, int] | None:
    prior = [entry for entry in entries if entry[0] < index]
    if not prior:
        return None
    return max(prior, key=lambda item: item[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat full colored rows and full colored columns as filters. Each grey object inherits the
    row-filter above it and the column-filter to its left. Matching colors recolor the object;
    mismatching colors keep the object grey and add a dashed frame using the row color on the
    horizontal sides and the column color on the vertical sides.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    filter_cols = _find_filter_columns(g)
    filter_rows = _find_filter_rows(g, filter_cols)
    grey_components = _components_of_color(g, GREY)
    assert grey_components, "expected at least one grey foreground object"

    out = g.copy()
    for coords in grey_components:
        r0, c0, r1, c1 = _bbox(coords)
        row_filter = _nearest_before(r0, filter_rows)
        col_filter = _nearest_before(c0, filter_cols)
        assert row_filter is not None and col_filter is not None, "each object must have a filter row above and a filter column to the left"
        row_color = row_filter[1]
        col_color = col_filter[1]

        if row_color == col_color:
            out[coords[:, 0], coords[:, 1]] = row_color
            continue

        top = r0 - 1
        bottom = r1 + 1
        left = c0 - 1
        right = c1 + 1
        assert 0 <= top <= bottom < g.shape[0]
        assert 0 <= left <= right < g.shape[1]

        for c in range(left, right + 1, 2):
            out[top, c] = row_color
            out[bottom, c] = row_color
        for r in range(top + 1, bottom, 2):
            out[r, left] = col_color
            out[r, right] = col_color

    return out


def _sample_positions(length: int, count: int, min_gap: int, edge_margin: int) -> list[int] | None:
    positions = list(range(edge_margin, length - edge_margin))
    random.shuffle(positions)
    chosen: list[int] = []
    for pos in positions:
        if all(abs(pos - other) >= min_gap for other in chosen):
            chosen.append(pos)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _region_bounds(lines: list[int], limit: int) -> list[tuple[int, int, int]]:
    out = []
    for i, start in enumerate(lines):
        end = lines[i + 1] if i + 1 < len(lines) else limit
        out.append((i, start, end))
    return out


def _random_pattern(max_h: int, max_w: int) -> np.ndarray:
    candidates = [p for p in PATTERNS if p.shape[0] <= max_h and p.shape[1] <= max_w]
    assert candidates
    pat = random.choice(candidates).copy()
    if random.random() < 0.5:
        pat = np.rot90(pat, random.randrange(4))
    if random.random() < 0.5:
        pat = np.fliplr(pat)
    if random.random() < 0.5:
        pat = np.flipud(pat)
    return pat.astype(bool)


def _place_object(
    grid: np.ndarray,
    region_top: int,
    region_bottom: int,
    region_left: int,
    region_right: int,
) -> bool:
    usable_h = region_bottom - region_top - 3
    usable_w = region_right - region_left - 3
    if usable_h < 1 or usable_w < 1:
        return False

    pattern = _random_pattern(min(usable_h, 4), min(usable_w, 4))
    ph, pw = pattern.shape
    row_choices = list(range(region_top + 2, region_bottom - ph))
    col_choices = list(range(region_left + 2, region_right - pw))
    random.shuffle(row_choices)
    random.shuffle(col_choices)

    for r0 in row_choices:
        for c0 in col_choices:
            sub = grid[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1]
            if np.any(sub != BG):
                continue
            patch = np.where(pattern, GREY, BG)
            grid[r0 : r0 + ph, c0 : c0 + pw] = patch
            return True
    return False


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(18, 28)
    w = random.randint(18, 28)

    filter_colors = random.sample(FILTER_COLOR_POOL, 2)
    a, b = filter_colors

    row_count = random.randint(2, 4)
    col_count = random.randint(2, 4)
    row_lines = _sample_positions(h, row_count, min_gap=5, edge_margin=1)
    col_lines = _sample_positions(w, col_count, min_gap=5, edge_margin=1)
    if row_lines is None or col_lines is None:
        return None

    row_colors = [a, b]
    while len(row_colors) < row_count:
        row_colors.append(random.choice([a, b]))
    col_colors = [a, b]
    while len(col_colors) < col_count:
        col_colors.append(random.choice([a, b]))
    random.shuffle(row_colors)
    random.shuffle(col_colors)

    if len(set(row_colors)) < 2 or len(set(col_colors)) < 2:
        return None

    inp = np.full((h, w), BG, dtype=int)
    for r, color in zip(row_lines, row_colors):
        inp[r, :] = color
    for c, color in zip(col_lines, col_colors):
        inp[:, c] = color

    row_regions = _region_bounds(row_lines, h)
    col_regions = _region_bounds(col_lines, w)
    candidates = []
    for row_idx, row_line, row_end in row_regions:
        for col_idx, col_line, col_end in col_regions:
            if row_end - row_line >= 5 and col_end - col_line >= 5:
                same = row_colors[row_idx] == col_colors[col_idx]
                candidates.append((same, row_line, row_end, col_line, col_end))

    same_regions = [item for item in candidates if item[0]]
    diff_regions = [item for item in candidates if not item[0]]
    if not same_regions or not diff_regions:
        return None

    target_count = min(len(candidates), random.randint(3, 6))
    chosen = [random.choice(same_regions), random.choice(diff_regions)]
    remaining = [item for item in candidates if item not in chosen]
    random.shuffle(remaining)
    chosen.extend(remaining[: max(0, target_count - 2)])
    random.shuffle(chosen)

    placed = 0
    same_count = 0
    diff_count = 0
    for same, row_line, row_end, col_line, col_end in chosen:
        if _place_object(inp, row_line, row_end, col_line, col_end):
            placed += 1
            if same:
                same_count += 1
            else:
                diff_count += 1

    if placed < 2 or same_count == 0 or diff_count == 0:
        return None

    out = main(inp)
    if np.array_equal(out, inp):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        case = _build_case()
        if case is None:
            continue
        inp, out = case
        try:
            if np.array_equal(main(inp), out):
                return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
