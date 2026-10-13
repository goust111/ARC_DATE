from __future__ import annotations

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


STAMP_MASKS = [
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
]


GEN_RETRIES = 400


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = np.asarray(grid == color, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        comp = [(r, c)]
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            for dr, dc in DIR4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
                    comp.append((nr, nc))
        comps.append(comp)
    return comps


def _stamp_spec(coords: list[tuple[int, int]]) -> dict[str, object]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    r1 = max(rows)
    c0 = min(cols)
    c1 = max(cols)
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    offsets = [(r - center_r, c - center_c) for r, c in coords]
    return {
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "offsets": offsets,
        "center_int": (center_r, center_c),
        "center_float": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
    }


def _build_rulers(grid: np.ndarray, bg: int) -> tuple[dict[int, int], dict[int, int], list[int]]:
    h, w = grid.shape
    top_positions: dict[int, int] = {}
    left_positions: dict[int, int] = {}

    for c in range(1, w):
        color = int(grid[0, c])
        if color == bg:
            continue
        assert color not in top_positions, "top ruler colors must be unique"
        top_positions[color] = c

    for r in range(1, h):
        color = int(grid[r, 0])
        if color == bg:
            continue
        assert color not in left_positions, "left ruler colors must be unique"
        left_positions[color] = r

    assert top_positions and left_positions, "both rulers must exist"
    assert set(top_positions) == set(left_positions), "top and left rulers must use the same colors"

    top_order = [color for color, _ in sorted(top_positions.items(), key=lambda item: item[1])]
    left_order = [color for color, _ in sorted(left_positions.items(), key=lambda item: item[1])]
    assert top_order == left_order, "ruler order must agree on both edges"

    return top_positions, left_positions, top_order


def _build_markers(
    grid: np.ndarray,
    ruler_colors: list[int],
    top_positions: dict[int, int],
    left_positions: dict[int, int],
) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    interior = grid[1:, 1:]
    for color in ruler_colors:
        coords = np.argwhere(interior == color)
        if coords.size == 0:
            continue
        assert coords.shape == (1, 2), "each ruler color may define at most one internal marker"
        r = int(coords[0, 0]) + 1
        c = int(coords[0, 1]) + 1
        assert (r, c) == (left_positions[color], top_positions[color]), "marker must sit on its ruler intersection"
        markers.append((color, r, c))
    assert len(markers) >= 2, "need at least two markers to create crossings"
    return markers


def _collect_stamp_sprites(grid: np.ndarray, bg: int, ruler_colors: set[int]) -> list[dict[str, object]]:
    sprite_colors = sorted(int(color) for color in np.unique(grid) if int(color) not in ruler_colors and int(color) != bg)
    sprites: list[dict[str, object]] = []
    for color in sprite_colors:
        comps = _components_of_color(grid, color)
        assert len(comps) == 1, "each stamp color must form one component"
        spec = _stamp_spec(comps[0])
        r0, c0, r1, c1 = spec["bbox"]
        assert 1 <= r0 <= r1 < grid.shape[0]
        assert 1 <= c0 <= c1 < grid.shape[1]
        assert 2 <= r1 - r0 + 1 <= 3 and 2 <= c1 - c0 + 1 <= 3, "stamps must be 2x2 or 3x3"
        sprites.append({"color": color, **spec})
    assert sprites, "need at least one source stamp"
    return sprites


def _crossings(markers: list[tuple[int, int, int]]) -> list[tuple[int, int, int, int]]:
    result: list[tuple[int, int, int, int]] = []
    for vertical_color, _vr, vcol in markers:
        for horizontal_color, hrow, _hc in markers:
            if vertical_color == horizontal_color:
                continue
            result.append((hrow, vcol, vertical_color, horizontal_color))
    return result


def _nearest_stamp(stamps: list[dict[str, object]], row: int, col: int) -> dict[str, object]:
    dists = [
        ((float(stamp["center_float"][0]) - row) ** 2 + (float(stamp["center_float"][1]) - col) ** 2, idx)
        for idx, stamp in enumerate(stamps)
    ]
    best_dist = min(dist for dist, _ in dists)
    winners = [idx for dist, idx in dists if abs(dist - best_dist) < 1e-9]
    assert len(winners) == 1, "nearest stamp must be unique"
    return stamps[winners[0]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read matching color rulers on the top and left edges, extend each marker into
    a horizontal/vertical colored line, and stamp the nearest source sprite at
    every crossing between different markers using the earlier ruler color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _bg_color(g)

    top_positions, left_positions, ruler_order = _build_rulers(g, bg)
    ruler_colors = set(ruler_order)
    rank = {color: idx for idx, color in enumerate(ruler_order)}
    markers = _build_markers(g, ruler_order, top_positions, left_positions)
    stamps = _collect_stamp_sprites(g, bg, ruler_colors)

    out = g.copy()
    line_cells: dict[tuple[int, int], int] = {}

    for color, row, col in markers:
        for rr in range(1, h):
            line_cells[(rr, col)] = color
        for cc in range(1, w):
            line_cells[(row, cc)] = color

    crossings = _crossings(markers)
    assert crossings, "need at least one pairwise crossing"
    for row, col, vertical_color, horizontal_color in crossings:
        chosen = vertical_color if rank[vertical_color] < rank[horizontal_color] else horizontal_color
        line_cells[(row, col)] = chosen

    for (row, col), color in line_cells.items():
        base = int(g[row, col])
        assert base == bg or base in ruler_colors, "lines must only cover background or markers"
        out[row, col] = color

    for row, col, vertical_color, horizontal_color in crossings:
        chosen = vertical_color if rank[vertical_color] < rank[horizontal_color] else horizontal_color
        stamp = _nearest_stamp(stamps, row, col)
        for dr, dc in stamp["offsets"]:
            rr = row + int(dr)
            cc = col + int(dc)
            assert 0 < rr < h and 0 < cc < w, "stamped copy must stay inside the interior"
            base = int(g[rr, cc])
            assert base == bg or (rr, cc) in line_cells, "stamped copy must not overwrite source stamps or rulers"
            out[rr, cc] = chosen

    return out


def _sample_spaced_positions(length: int, count: int, margin: int, gap: int, rng: np.random.Generator) -> list[int] | None:
    slots = list(range(margin, length - margin))
    rng.shuffle(slots)
    chosen: list[int] = []
    for pos in slots:
        if all(abs(pos - prev) >= gap for prev in chosen):
            chosen.append(int(pos))
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _mask_offsets(mask: np.ndarray) -> list[tuple[int, int]]:
    center_r = mask.shape[0] // 2
    center_c = mask.shape[1] // 2
    return [
        (int(r) - center_r, int(c) - center_c)
        for r, c in np.argwhere(mask)
    ]


def _stamp_cells(center: tuple[int, int], mask: np.ndarray) -> set[tuple[int, int]]:
    row, col = center
    return {(row + dr, col + dc) for dr, dc in _mask_offsets(mask)}


def _crossing_boxes(crossings: list[tuple[int, int, int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for row, col, _vc, _hc in crossings:
        for rr in range(row - 1, row + 2):
            for cc in range(col - 1, col + 2):
                cells.add((rr, cc))
    return cells


def _line_cells_from_markers(height: int, width: int, markers: list[tuple[int, int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for _color, row, col in markers:
        for rr in range(1, height):
            cells.add((rr, col))
        for cc in range(1, width):
            cells.add((row, cc))
    return cells


def _find_stamp_centers(
    height: int,
    width: int,
    crossings: list[tuple[int, int, int, int]],
    forbidden: set[tuple[int, int]],
    count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    candidates = [
        (r, c)
        for r in range(2, height - 2)
        for c in range(2, width - 2)
        if all((rr, cc) not in forbidden for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2))
    ]
    if len(candidates) < count:
        return None

    for _ in range(120):
        rng.shuffle(candidates)
        chosen: list[tuple[int, int]] = []
        for center in candidates:
            if all(max(abs(center[0] - other[0]), abs(center[1] - other[1])) >= 4 for other in chosen):
                chosen.append(center)
                if len(chosen) == count:
                    break
        if len(chosen) != count:
            continue

        winners: list[int] = []
        for row, col, _vc, _hc in crossings:
            dists = [(center[0] - row) ** 2 + (center[1] - col) ** 2 for center in chosen]
            best = min(dists)
            if dists.count(best) != 1:
                break
            winners.append(int(np.argmin(dists)))
        else:
            if len(set(winners)) >= min(2, count):
                return chosen
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(16, 25))
    width = int(rng.integers(16, 25))

    bg = int(rng.integers(0, 10))
    palette = [color for color in range(10) if color != bg]

    max_rulers = min(5, (height - 4 + 2) // 3, (width - 4 + 2) // 3, len(palette) - 2)
    if max_rulers < 3:
        return None
    ruler_count = int(rng.integers(3, max_rulers + 1))

    row_positions = _sample_spaced_positions(height, ruler_count, margin=2, gap=3, rng=rng)
    col_positions = _sample_spaced_positions(width, ruler_count, margin=2, gap=3, rng=rng)
    if row_positions is None or col_positions is None:
        return None

    rng.shuffle(palette)
    ruler_colors = palette[:ruler_count]
    remaining = palette[ruler_count:]
    if len(remaining) < 2:
        return None

    grid = np.full((height, width), bg, dtype=int)
    top_positions = {color: col for color, col in zip(ruler_colors, col_positions)}
    left_positions = {color: row for color, row in zip(ruler_colors, row_positions)}

    for color in ruler_colors:
        grid[0, top_positions[color]] = color
        grid[left_positions[color], 0] = color

    marker_count = int(rng.integers(3, ruler_count + 1))
    marker_colors = list(rng.choice(ruler_colors, size=marker_count, replace=False))
    markers = [(color, left_positions[color], top_positions[color]) for color in marker_colors]
    for color, row, col in markers:
        grid[row, col] = color

    crossings = _crossings(markers)
    if not crossings:
        return None

    line_cells = _line_cells_from_markers(height, width, markers)
    crossing_boxes = _crossing_boxes(crossings)
    forbidden = set(line_cells) | crossing_boxes
    forbidden |= {(0, c) for c in range(width)}
    forbidden |= {(r, 0) for r in range(height)}

    stamp_count = int(rng.integers(2, min(4, len(remaining)) + 1))
    centers = _find_stamp_centers(height, width, crossings, forbidden, stamp_count, rng)
    if centers is None:
        return None

    rng.shuffle(remaining)
    stamp_colors = remaining[:stamp_count]
    mask_ids = list(range(len(STAMP_MASKS)))
    rng.shuffle(mask_ids)
    masks = [STAMP_MASKS[idx] for idx in mask_ids[:stamp_count]]

    occupied = {(r, c) for r, c in np.argwhere(grid != bg)}
    for color, center, mask in zip(stamp_colors, centers, masks):
        cells = _stamp_cells(center, mask)
        if any(not (0 < r < height and 0 < c < width) for r, c in cells):
            return None
        if any((r, c) in occupied for r, c in cells):
            return None
        for r, c in cells:
            grid[r, c] = color
            occupied.add((r, c))

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None

    added = int(np.count_nonzero(out != grid))
    if added < 10:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
