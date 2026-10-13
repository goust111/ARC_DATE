import random

import numpy as np


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                if cr > 0 and mask[cr - 1, cc] and not seen[cr - 1, cc]:
                    seen[cr - 1, cc] = True
                    stack.append((cr - 1, cc))
                if cr + 1 < h and mask[cr + 1, cc] and not seen[cr + 1, cc]:
                    seen[cr + 1, cc] = True
                    stack.append((cr + 1, cc))
                if cc > 0 and mask[cr, cc - 1] and not seen[cr, cc - 1]:
                    seen[cr, cc - 1] = True
                    stack.append((cr, cc - 1))
                if cc + 1 < w and mask[cr, cc + 1] and not seen[cr, cc + 1]:
                    seen[cr, cc + 1] = True
                    stack.append((cr, cc + 1))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_separator_row(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates = [
        r
        for r in range(grid.shape[0] - 1)
        if int(grid[r, 0]) != bg and np.all(grid[r] == grid[r, 0])
    ]
    assert candidates, "expected at least one full-width non-background separator row"
    sep = max(candidates)
    return int(sep), int(grid[sep, 0])


def _legend_mapping(grid: np.ndarray, sep: int, bg: int) -> dict[int, np.ndarray]:
    legend = grid[sep + 1 :]
    assert legend.size > 0, "legend area must exist below the separator"

    mapping: dict[int, np.ndarray] = {}
    for comp in _components(legend != bg):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h in (2, 3), "legend patterns must be 2x2 or 3x3"
        assert w == h + 1, "legend block must be one marker column plus an sxs pattern"

        block = legend[r0 : r1 + 1, c0 : c1 + 1]
        marker_rows = np.flatnonzero(block[:, 0] != bg)
        assert len(marker_rows) == 1, "legend marker column must contain exactly one colored pixel"

        marker = int(block[marker_rows[0], 0])
        pattern = np.array(block[:, 1:], dtype=int)
        assert marker not in mapping, "legend marker colors must be unique"
        assert pattern.shape == (h, h)
        assert np.all(pattern != bg), "legend pattern pixels must be visible"
        mapping[marker] = pattern

    assert mapping, "legend must contain at least one marker-pattern mapping"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each seeded partition with the bottom legend pattern selected by the seed color.

    Input structure:
    - The background is the most common color, while grey divider lines split the upper part of the grid into closed rectangular regions.
    - A full-width grey separator row sits above a bottom legend strip.
    - Each legend entry is a connected block made of one colored marker pixel on the left and an adjacent 2x2 or 3x3 pattern on the right.
    - Some closed regions contain exactly one colored seed pixel whose color matches one legend marker.

    Rule:
    - Read the bottom legend to build a mapping from each marker color to its visible pattern.
    - Find every closed non-grey region above the separator row.
    - If a region contains a seed pixel, replace the entire region with the corresponding pattern tiled from the region's top-left corner; the seed pixel is overwritten by the tiling.
    - Regions with no seed stay unchanged, and the divider lines plus the legend remain as they are.

    Key insight:
    - The seed color is only a lookup key: the actual fill must first be decoded from the bottom legend, then tiled inside the seed's enclosing partition.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    sep, wall = _find_separator_row(g, bg)
    mapping = _legend_mapping(g, sep, bg)

    out = g.copy()
    upper = g[:sep, :]
    for comp in _components(upper != wall):
        seed_cells = [(int(r), int(c)) for r, c in comp if int(g[r, c]) != bg]
        if not seed_cells:
            continue

        assert len(seed_cells) == 1, "each filled region must contain exactly one seed pixel"
        seed_r, seed_c = seed_cells[0]
        seed_color = int(g[seed_r, seed_c])
        assert seed_color in mapping, "every seed color must appear in the legend"

        pattern = mapping[seed_color]
        tile_h, tile_w = pattern.shape
        r0, c0, _r1, _c1 = _bbox(comp)
        for r, c in comp:
            rr, cc = int(r), int(c)
            out[rr, cc] = int(pattern[(rr - r0) % tile_h, (cc - c0) % tile_w])

    return out


# --------------- generator ---------------

_BG = 0
_WALL = 5
_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
_MAX_RETRIES = 400

_PATTERN_BITS = {
    2: [
        np.array([[0, 1], [1, 0]], dtype=int),
        np.array([[0, 0], [1, 1]], dtype=int),
        np.array([[0, 1], [0, 1]], dtype=int),
    ],
    3: [
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
        np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=int),
        np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=int),
        np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=int),
    ],
}


def _strict_bg_is_largest(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    count_map = {int(v): int(c) for v, c in zip(values, counts)}
    return all(count_map.get(bg, 0) > c for color, c in count_map.items() if color != bg)


def _sample_entry_sizes(count: int) -> list[int]:
    sizes = [random.choice([2, 3]) for _ in range(count)]
    if count >= 2 and random.random() < 0.5:
        sizes[0] = 2
        sizes[1] = 3
        random.shuffle(sizes)
    return sizes


def _pick_patterns(markers: list[int], sizes: list[int]) -> dict[int, np.ndarray] | None:
    palette = [c for c in _COLORS if c not in markers]
    used: set[tuple[int, ...]] = set()
    result: dict[int, np.ndarray] = {}
    for marker, size in zip(markers, sizes):
        for _ in range(80):
            bits = random.choice(_PATTERN_BITS[size])
            a, b = random.sample(palette, 2)
            if random.random() < 0.5:
                a, b = b, a
            pattern = np.where(bits == 0, a, b).astype(int)
            sig = tuple(int(x) for x in pattern.ravel())
            if sig in used:
                continue
            used.add(sig)
            result[marker] = pattern
            break
        else:
            return None
    return result


def _render_expected(
    grid: np.ndarray,
    cells: list[tuple[int, int, int, int]],
    assignments: dict[int, int],
    patterns: dict[int, np.ndarray],
) -> np.ndarray:
    out = grid.copy()
    for idx, marker in assignments.items():
        r0, r1, c0, c1 = cells[idx]
        pattern = patterns[marker]
        size = pattern.shape[0]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                out[r, c] = int(pattern[(r - r0) % size, (c - c0) % size])
    return out


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(2, 3)
    n_cols = random.randint(3, 4)
    n_cells = n_rows * n_cols

    n_entries = random.randint(2, min(4, n_cells - 1))
    sizes = _sample_entry_sizes(n_entries)
    legend_h = max(sizes)
    gap = random.randint(1, 2)
    legend_width = sum(size + 1 for size in sizes) + gap * (n_entries - 1)

    row_heights = [random.randint(3, 6) for _ in range(n_rows)]
    col_widths = [random.randint(3, 6) for _ in range(n_cols)]
    max_width = n_cols * 6 + n_cols + 1
    if max_width < legend_width + 2:
        return None
    while sum(col_widths) + n_cols + 1 < legend_width + 2:
        growable = [i for i, width in enumerate(col_widths) if width < 6]
        if not growable:
            return None
        idx = random.choice(growable)
        col_widths[idx] += 1

    sep = sum(row_heights) + n_rows
    height = sep + 1 + legend_h
    width = sum(col_widths) + n_cols + 1
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), _BG, dtype=int)

    grid[0, :] = _WALL
    row_boxes: list[tuple[int, int]] = []
    r = 0
    for row_h in row_heights:
        r0 = r + 1
        r1 = r0 + row_h - 1
        row_boxes.append((r0, r1))
        r = r1 + 1
        grid[r, :] = _WALL

    col_boxes: list[tuple[int, int]] = []
    grid[: sep + 1, 0] = _WALL
    c = 0
    for col_w in col_widths:
        c0 = c + 1
        c1 = c0 + col_w - 1
        col_boxes.append((c0, c1))
        c = c1 + 1
        grid[: sep + 1, c] = _WALL

    cells = [(r0, r1, c0, c1) for r0, r1 in row_boxes for c0, c1 in col_boxes]

    markers = random.sample(_COLORS, n_entries)
    patterns = _pick_patterns(markers, sizes)
    if patterns is None:
        return None

    left = random.randint(1, width - legend_width - 1)
    x = left
    for marker, size in zip(markers, sizes):
        pattern = patterns[marker]
        y = sep + 1 + random.randint(0, legend_h - size)
        marker_row = random.randint(0, size - 1)
        grid[y + marker_row, x] = marker
        grid[y : y + size, x + 1 : x + 1 + size] = pattern
        x += size + 1 + gap

    cover_cells = random.sample(range(n_cells), n_entries)
    shuffled_markers = markers[:]
    random.shuffle(shuffled_markers)
    assignments = {cell_idx: marker for cell_idx, marker in zip(cover_cells, shuffled_markers)}

    remaining = [idx for idx in range(n_cells) if idx not in assignments]
    extra_count = random.randint(0, len(remaining) - 1)
    for idx in random.sample(remaining, extra_count):
        assignments[idx] = random.choice(markers)

    for idx, marker in assignments.items():
        r0, r1, c0, c1 = cells[idx]
        sr = random.randint(r0, r1)
        sc = random.randint(c0, c1)
        grid[sr, sc] = marker

    if not _strict_bg_is_largest(grid, _BG):
        return None

    expected = _render_expected(grid, cells, assignments, patterns)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected) and not np.array_equal(actual, grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {_MAX_RETRIES} attempts")
