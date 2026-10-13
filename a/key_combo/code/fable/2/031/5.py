from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
SPRITE_COLOR_POOL = [3, 4, 5, 6, 7, 8, 9]


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    """Minimal ARC-style connected components in canonical (r, c) coordinates."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
        ]

    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _corner_origin(shape: tuple[int, int], size: int, corner: str) -> Coord:
    h, w = shape
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - size
    if corner == "bl":
        return h - size, 0
    if corner == "br":
        return h - size, w - size
    raise ValueError(f"unknown corner: {corner}")


def _sprite_offsets(sprite_mask: np.ndarray) -> list[Coord]:
    size = sprite_mask.shape[0]
    k = size // 2
    coords = np.argwhere(sprite_mask)
    return [(int(r - k), int(c - k)) for r, c in coords]


def _stamp(
    out: np.ndarray,
    center: Coord,
    offsets: list[Coord],
    color: int,
    occupied: np.ndarray | None = None,
) -> None:
    h, w = out.shape
    cells: list[Coord] = []
    for dr, dc in offsets:
        r = center[0] + dr
        c = center[1] + dc
        assert 0 <= r < h and 0 <= c < w, "stamp leaves the grid"
        cells.append((r, c))
    if occupied is not None:
        for r, c in cells:
            assert not occupied[r, c], "stamp footprints overlap"
    for r, c in cells:
        out[r, c] = int(color)
        if occupied is not None:
            occupied[r, c] = True


def _mask_from_corner(sprite_mask: np.ndarray, shape: tuple[int, int], corner: str) -> np.ndarray:
    size = sprite_mask.shape[0]
    r0, c0 = _corner_origin(shape, size, corner)
    mask = np.zeros(shape, dtype=bool)
    coords = np.argwhere(sprite_mask)
    mask[r0 + coords[:, 0], c0 + coords[:, 1]] = True
    return mask


def _red_pairs(g: np.ndarray) -> dict[int, tuple[int, int]]:
    rows = np.flatnonzero(np.any(g == RED, axis=1))
    pairs: dict[int, tuple[int, int]] = {}
    for r in rows:
        cols = np.flatnonzero(g[r] == RED)
        assert len(cols) == 2, "each red row must contain exactly one pair"
        c0, c1 = map(int, sorted(cols.tolist()))
        assert c1 - c0 >= 2, "red pair must define a non-empty open interval"
        pairs[int(r)] = (c0, c1)
    assert pairs, "expected at least one red pair"
    return pairs


def _blue_pairs(g: np.ndarray) -> dict[int, tuple[int, int]]:
    cols = np.flatnonzero(np.any(g == BLUE, axis=0))
    pairs: dict[int, tuple[int, int]] = {}
    for c in cols:
        rows = np.flatnonzero(g[:, c] == BLUE)
        assert len(rows) == 2, "each blue column must contain exactly one pair"
        r0, r1 = map(int, sorted(rows.tolist()))
        assert r1 - r0 >= 2, "blue pair must define a non-empty open interval"
        pairs[int(c)] = (r0, r1)
    assert pairs, "expected at least one blue pair"
    return pairs


def _extract_sprite(g: np.ndarray) -> tuple[int, np.ndarray, str]:
    colors = sorted(set(map(int, g.ravel())) - {BG, RED, BLUE})
    assert len(colors) == 1, "expected one sprite color besides red and blue"
    color = colors[0]
    mask = g == color
    comps = _components(mask, connectivity=8)
    assert len(comps) == 1, "sprite color must form one object"
    bbox = _bbox_coords(comps[0])
    r0, c0, r1, c1 = bbox
    size = r1 - r0 + 1
    assert size == c1 - c0 + 1 and size % 2 == 1, "sprite bbox must be an odd square"
    center = (r0 + size // 2, c0 + size // 2)
    assert mask[center], "sprite center cell must be occupied"

    h, w = g.shape
    corner_map = {
        "tl": (0, 0),
        "tr": (0, w - size),
        "bl": (h - size, 0),
        "br": (h - size, w - size),
    }
    matching = [name for name, origin in corner_map.items() if origin == (r0, c0)]
    assert len(matching) == 1, "sprite must sit flush in exactly one corner"

    sprite = np.zeros((size, size), dtype=int)
    local = comps[0] - np.array([[r0, c0]])
    sprite[local[:, 0], local[:, 1]] = color
    return color, sprite, matching[0]


def _solve_from_parts(
    g: np.ndarray,
    red_pairs: dict[int, tuple[int, int]],
    blue_pairs: dict[int, tuple[int, int]],
    sprite_color: int,
    sprite_mask: np.ndarray,
) -> np.ndarray:
    out = g.copy()
    out[g == sprite_color] = BG
    offsets = _sprite_offsets(sprite_mask)
    occupied = np.zeros_like(g, dtype=bool)

    centers: list[Coord] = []
    for r, (c0, c1) in red_pairs.items():
        for c, (r0, r1) in blue_pairs.items():
            if c0 < c < c1 and r0 < r < r1:
                centers.append((r, c))
    assert centers, "expected at least one hidden intersection"

    for center in centers:
        _stamp(out, center, offsets, sprite_color, occupied=occupied)

    out[g == RED] = RED
    out[g == BLUE] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the marker pairs, erase the corner sprite, and stamp that sprite only at hidden row-column segment intersections.

    Input structure:
    - The background is black.
    - Red marker pixels appear in same-row pairs, with at most one red pair per row.
    - Blue marker pixels appear in same-column pairs, with at most one blue pair per column.
    - One odd-square sprite of a fourth color sits flush in a corner and is separate from the markers.

    Rule:
    - Each red pair defines a hidden horizontal open segment strictly between its two markers.
    - Each blue pair defines a hidden vertical open segment strictly between its two markers.
    - Whenever a hidden horizontal segment and a hidden vertical segment cross, stamp the full corner sprite centered on that intersection cell.
    - Do not draw the hidden segments themselves; erase the original corner sprite to background and keep all red and blue marker pixels unchanged.

    Key insight:
    - The markers are only geometric scaffolding: the output event is not a visible line, but a copied sprite placed exactly where unseen horizontal and vertical open intervals intersect.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert Counter(map(int, g.ravel()))[BG] > 0, "expected black background"

    red_pairs = _red_pairs(g)
    blue_pairs = _blue_pairs(g)
    sprite_color, sprite, _corner = _extract_sprite(g)
    sprite_mask = sprite == sprite_color

    return _solve_from_parts(g, red_pairs, blue_pairs, sprite_color, sprite_mask)


def _random_path(start: Coord, goal: Coord) -> list[Coord]:
    r, c = start
    out = [(r, c)]
    while (r, c) != goal:
        steps = []
        if r < goal[0]:
            steps.append((1, 0))
        elif r > goal[0]:
            steps.append((-1, 0))
        if c < goal[1]:
            steps.append((0, 1))
        elif c > goal[1]:
            steps.append((0, -1))
        dr, dc = random.choice(steps)
        r += dr
        c += dc
        out.append((r, c))
    return out


def _random_sprite_mask(size: int) -> np.ndarray:
    k = size // 2
    center = (k, k)
    mask = np.zeros((size, size), dtype=bool)
    mask[center] = True

    anchors = [
        (0, random.randrange(size)),
        (size - 1, random.randrange(size)),
        (random.randrange(size), 0),
        (random.randrange(size), size - 1),
    ]
    for anchor in anchors:
        for cell in _random_path(anchor, center):
            mask[cell] = True

    target_area = random.randint(max(size + 2, 5), size * size - 1)
    while int(mask.sum()) < target_area:
        frontier = []
        for r in range(size):
            for c in range(size):
                if mask[r, c]:
                    continue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size and mask[nr, nc]:
                        frontier.append((r, c))
                        break
        if not frontier:
            break
        r, c = random.choice(frontier)
        mask[r, c] = True

    if random.random() < 0.5:
        mask = np.rot90(mask, k=random.randrange(4))
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    if random.random() < 0.5:
        mask = np.flipud(mask)

    assert mask[k, k]
    assert len(_components(mask, connectivity=4)) == 1
    assert _bbox_coords(np.argwhere(mask)) == (0, 0, size - 1, size - 1)
    return mask


def _all_spans(n: int) -> list[tuple[int, int]]:
    return [(a, b) for a in range(n) for b in range(a, n)]


def _choose_interval_spec(n_rows: int, n_cols: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]], np.ndarray]:
    row_spans_all = _all_spans(n_cols)
    col_spans_all = _all_spans(n_rows)

    for _ in range(500):
        row_spans = [random.choice(row_spans_all) for _ in range(n_rows)]
        col_spans = [random.choice(col_spans_all) for _ in range(n_cols)]

        h_cover = np.zeros((n_rows, n_cols), dtype=bool)
        v_cover = np.zeros((n_rows, n_cols), dtype=bool)
        for i, (a, b) in enumerate(row_spans):
            h_cover[i, a : b + 1] = True
        for j, (a, b) in enumerate(col_spans):
            v_cover[a : b + 1, j] = True
        valid = h_cover & v_cover

        if np.any(valid.sum(axis=1) == 0) or np.any(valid.sum(axis=0) == 0):
            continue
        if n_rows > 1 and n_cols > 1 and np.all(valid):
            continue
        if n_rows > 1 and n_cols > 1:
            if not np.any(h_cover & ~v_cover):
                continue
            if not np.any(~h_cover & v_cover):
                continue
            if not any(0 < int(valid[i].sum()) < n_cols for i in range(n_rows)):
                continue
            if not any(0 < int(valid[:, j].sum()) < n_rows for j in range(n_cols)):
                continue
        return row_spans, col_spans, valid

    raise AssertionError("failed to sample a non-degenerate interval specification")


def _sample_positions(count: int, low: int, high: int, min_gap: int) -> list[int]:
    combos = [
        list(combo)
        for combo in itertools.combinations(range(low, high + 1), count)
        if all(combo[i + 1] - combo[i] >= min_gap for i in range(len(combo) - 1))
    ]
    assert combos, f"no valid positions for count={count}, range=({low}, {high})"
    return random.choice(combos)


def _marker_ranges_for_row(
    cols: list[int],
    span: tuple[int, int],
    k: int,
    width: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    a, b = span
    left_lo = 0 if a == 0 else cols[a - 1] + 1
    left_hi = cols[a] - k - 1
    right_lo = cols[b] + k + 1
    right_hi = width - 1 if b == len(cols) - 1 else cols[b + 1] - 1
    assert left_lo <= left_hi and right_lo <= right_hi and left_hi < right_lo
    return (left_lo, left_hi), (right_lo, right_hi)


def _marker_ranges_for_col(
    rows: list[int],
    span: tuple[int, int],
    k: int,
    height: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    a, b = span
    top_lo = 0 if a == 0 else rows[a - 1] + 1
    top_hi = rows[a] - k - 1
    bottom_lo = rows[b] + k + 1
    bottom_hi = height - 1 if b == len(rows) - 1 else rows[b + 1] - 1
    assert top_lo <= top_hi and bottom_lo <= bottom_hi and top_hi < bottom_lo
    return (top_lo, top_hi), (bottom_lo, bottom_hi)


def _build_expected(
    shape: tuple[int, int],
    row_positions: list[int],
    col_positions: list[int],
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    sprite_mask: np.ndarray,
    sprite_color: int,
    corner: str,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    size = sprite_mask.shape[0]
    k = size // 2
    grid = np.zeros(shape, dtype=int)

    row_markers: dict[int, tuple[int, int]] = {}
    for r, span in zip(row_positions, row_spans):
        left_range, right_range = _marker_ranges_for_row(col_positions, span, k, w)
        row_markers[r] = (
            random.randint(*left_range),
            random.randint(*right_range),
        )

    col_markers: dict[int, tuple[int, int]] = {}
    for c, span in zip(col_positions, col_spans):
        top_range, bottom_range = _marker_ranges_for_col(row_positions, span, k, h)
        col_markers[c] = (
            random.randint(*top_range),
            random.randint(*bottom_range),
        )

    for r, (c0, c1) in row_markers.items():
        grid[r, c0] = RED
        grid[r, c1] = RED
    for c, (r0, r1) in col_markers.items():
        assert grid[r0, c] == BG and grid[r1, c] == BG
        grid[r0, c] = BLUE
        grid[r1, c] = BLUE

    sprite_cells = _mask_from_corner(sprite_mask, shape, corner)
    assert not np.any((grid != BG) & sprite_cells), "sprite touches markers"
    grid[sprite_cells] = sprite_color

    expected = _solve_from_parts(grid, row_markers, col_markers, sprite_color, sprite_mask)

    stamp_occupied = np.zeros(shape, dtype=bool)
    offsets = _sprite_offsets(sprite_mask)
    for i, r in enumerate(row_positions):
        for j, c in enumerate(col_positions):
            if not (row_spans[i][0] <= j <= row_spans[i][1] and col_spans[j][0] <= i <= col_spans[j][1]):
                continue
            before = stamp_occupied.copy()
            _stamp(np.zeros(shape, dtype=int), (r, c), offsets, sprite_color, occupied=stamp_occupied)
            stamp_mask = stamp_occupied & ~before
            assert not np.any(stamp_mask & (grid == RED)), "stamp would cover a red marker"
            assert not np.any(stamp_mask & (grid == BLUE)), "stamp would cover a blue marker"

    return grid, expected


def _valid_geometry(
    grid: np.ndarray,
    expected: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    valid: np.ndarray,
    sprite_mask: np.ndarray,
    corner: str,
) -> bool:
    if np.array_equal(grid, expected):
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if len(set(map(int, grid.ravel())) - {BG}) != 3:
        return False

    sprite_cells = _mask_from_corner(sprite_mask, grid.shape, corner)
    if np.any(expected[sprite_cells] != BG):
        return False

    if np.any(valid.sum(axis=1) == 0) or np.any(valid.sum(axis=0) == 0):
        return False
    if valid.shape[0] > 1 and valid.shape[1] > 1:
        h_cover = np.zeros_like(valid, dtype=bool)
        v_cover = np.zeros_like(valid, dtype=bool)
        for i, (a, b) in enumerate(row_spans):
            h_cover[i, a : b + 1] = True
        for j, (a, b) in enumerate(col_spans):
            v_cover[a : b + 1, j] = True
        if not np.any(h_cover & ~v_cover):
            return False
        if not np.any(~h_cover & v_cover):
            return False
    return np.array_equal(main(grid), expected)


def generate_input() -> np.ndarray:
    for _ in range(500):
        sprite_size = random.choice([3, 5])
        sprite_mask = _random_sprite_mask(sprite_size)
        sprite_color = random.choice(SPRITE_COLOR_POOL)
        corner = random.choice(["tl", "tr", "bl", "br"])

        n_rows = random.choice([1, 2, 2, 3])
        n_cols = random.choice([1, 2, 2, 3])
        row_spans, col_spans, valid = _choose_interval_spec(n_rows, n_cols)

        margin = sprite_size + 2
        min_gap = sprite_size
        min_h = 2 * margin + 1 + (n_rows - 1) * min_gap
        min_w = 2 * margin + 1 + (n_cols - 1) * min_gap
        if min_h > 30 or min_w > 30:
            continue
        height = random.randint(min_h, 30)
        width = random.randint(min_w, 30)
        row_positions = _sample_positions(n_rows, margin, height - margin - 1, min_gap)
        col_positions = _sample_positions(n_cols, margin, width - margin - 1, min_gap)

        try:
            grid, expected = _build_expected(
                (height, width),
                row_positions,
                col_positions,
                row_spans,
                col_spans,
                sprite_mask,
                sprite_color,
                corner,
            )
        except AssertionError:
            continue

        if _valid_geometry(grid, expected, row_spans, col_spans, valid, sprite_mask, corner):
            return grid

    raise AssertionError("failed to generate a valid puzzle instance")
