from __future__ import annotations

import random
from collections import deque

import numpy as np


_DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_LEGEND_GAP = 1
_MAX_RETRIES = 400

_PATTERNS_2 = (
    np.array([[0, 1], [1, 0]], dtype=int),
    np.array([[0, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [0, 1]], dtype=int),
    np.array([[0, 0], [0, 1]], dtype=int),
    np.array([[0, 1], [1, 1]], dtype=int),
)

_PATTERNS_3 = (
    np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=int),
    np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=int),
    np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=int),
)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                pts.append((cr, cc))
                for dr, dc in _DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rs = coords[:, 0]
    cs = coords[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _touches_border(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _most_common_non_bg(grid: np.ndarray, bg: int) -> int:
    vals, cnts = np.unique(grid[grid != bg], return_counts=True)
    return int(vals[cnts.argmax()])


def _extract_legend_tile(
    grid: np.ndarray, color: int, coords: np.ndarray
) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    h, w = grid.shape
    r0, c0, r1, c1 = _bbox(coords)
    bh = r1 - r0 + 1
    bw = c1 - c0 + 1
    if (bh, bw) not in ((4, 4), (5, 5)):
        return None
    if not (r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1):
        return None

    pts = {tuple(map(int, pt)) for pt in coords}
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            on_border = rr in (r0, r1) or cc in (c0, c1)
            if ((rr, cc) in pts) != on_border:
                return None

    tile = grid[r0 + 1 : r1, c0 + 1 : c1]
    if tile.shape[0] != tile.shape[1] or tile.shape[0] not in (2, 3):
        return None
    if np.any(tile == color):
        return None
    if len(np.unique(tile)) != 2:
        return None
    return (r0, c0, r1, c1), tile.copy()


def _detect_legends(grid: np.ndarray, bg: int, wall: int) -> tuple[dict[int, np.ndarray], np.ndarray]:
    legend_map: dict[int, np.ndarray] = {}
    legend_mask = np.zeros(grid.shape, dtype=bool)
    for color0 in np.unique(grid):
        color = int(color0)
        if color in (bg, wall):
            continue
        for coords in _components(grid == color):
            if not _touches_border(coords, *grid.shape):
                continue
            parsed = _extract_legend_tile(grid, color, coords)
            if parsed is None:
                continue
            bbox, tile = parsed
            if color in legend_map:
                raise AssertionError("duplicate legend color")
            legend_map[color] = tile
            r0, c0, r1, c1 = bbox
            legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return legend_map, legend_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region with the legend tile keyed by its seed color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    wall = _most_common_non_bg(g, bg)
    legend_map, legend_mask = _detect_legends(g, bg, wall)
    assert legend_map, "expected at least one legend block"

    out = g.copy()
    applied = 0
    passable = (g != wall) & ~legend_mask
    for coords in _components(passable):
        if _touches_border(coords, *g.shape):
            continue
        vals = g[coords[:, 0], coords[:, 1]]
        seeds = [int(v) for v in np.unique(vals) if int(v) != bg]
        if len(seeds) != 1:
            continue
        seed = seeds[0]
        if seed not in legend_map or int(np.count_nonzero(vals == seed)) != 1:
            continue
        tile = legend_map[seed]
        r0, c0, _, _ = _bbox(coords)
        th, tw = tile.shape
        for r, c in coords:
            out[r, c] = int(tile[(int(r) - r0) % th, (int(c) - c0) % tw])
        applied += 1

    assert applied > 0, "expected at least one enclosed seeded region"
    return out


def _partition_labels(rng: random.Random, rows: int, cols: int, n_regions: int) -> np.ndarray | None:
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    seeds = rng.sample(all_cells, n_regions)
    labels = np.full((rows, cols), -1, dtype=int)
    regions: dict[int, list[tuple[int, int]]] = {}
    for label, cell in enumerate(seeds):
        labels[cell] = label
        regions[label] = [cell]

    remaining = rows * cols - n_regions
    while remaining > 0:
        expandable: list[tuple[int, list[tuple[int, int]]]] = []
        for label, cells in regions.items():
            frontier: list[tuple[int, int]] = []
            seen: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in _DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and labels[nr, nc] == -1 and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        frontier.append((nr, nc))
            if frontier:
                expandable.append((label, frontier))
        if not expandable:
            return None
        label, frontier = rng.choice(expandable)
        nr, nc = rng.choice(frontier)
        labels[nr, nc] = label
        regions[label].append((nr, nc))
        remaining -= 1

    if max(len(cells) for cells in regions.values()) == 1:
        return None
    return labels


def _cell_origin(cell_r: int, cell_c: int, cell_h: int, cell_w: int) -> tuple[int, int]:
    return 1 + cell_r * (cell_h + 1), 1 + cell_c * (cell_w + 1)


def _build_field(labels: np.ndarray, bg: int, wall: int, cell_h: int, cell_w: int) -> np.ndarray:
    rows, cols = labels.shape
    field_h = rows * cell_h + rows + 1
    field_w = cols * cell_w + cols + 1
    field = np.full((field_h, field_w), wall, dtype=int)

    for r in range(rows):
        for c in range(cols):
            top, left = _cell_origin(r, c, cell_h, cell_w)
            field[top : top + cell_h, left : left + cell_w] = bg

    for r in range(rows):
        for c in range(cols - 1):
            if labels[r, c] != labels[r, c + 1]:
                continue
            top, left = _cell_origin(r, c, cell_h, cell_w)
            field[top : top + cell_h, left + cell_w] = bg

    for r in range(rows - 1):
        for c in range(cols):
            if labels[r, c] != labels[r + 1, c]:
                continue
            top, left = _cell_origin(r, c, cell_h, cell_w)
            field[top + cell_h, left : left + cell_w] = bg

    return field


def _random_pattern(rng: random.Random, size: int) -> np.ndarray:
    base = rng.choice(_PATTERNS_2 if size == 2 else _PATTERNS_3)
    pattern = np.rot90(base, rng.randrange(4))
    if rng.random() < 0.5:
        pattern = np.fliplr(pattern)
    if rng.random() < 0.5:
        pattern = np.flipud(pattern)
    return np.array(pattern, dtype=int, copy=True)


def _make_tile(rng: random.Random, size: int, palette: list[int]) -> np.ndarray:
    a, b = rng.sample(palette, 2)
    pattern = _random_pattern(rng, size)
    return np.where(pattern == 0, a, b).astype(int)


def _render_legend(frame_color: int, tile: np.ndarray) -> np.ndarray:
    block = np.full((tile.shape[0] + 2, tile.shape[1] + 2), frame_color, dtype=int)
    block[1:-1, 1:-1] = tile
    return block


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    dst[top : top + h, left : left + w] = patch


def _assemble_canvas(
    rng: random.Random,
    field_inp: np.ndarray,
    field_out: np.ndarray,
    legend_blocks: list[np.ndarray],
    bg: int,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    fh, fw = field_inp.shape
    margin = rng.randint(1, 3)
    field_gap = rng.randint(1, 3)
    extra = rng.randint(0, 4)

    if side in ("top", "bottom"):
        strip_h = max(block.shape[0] for block in legend_blocks)
        legend_span = sum(block.shape[1] for block in legend_blocks) + _LEGEND_GAP * (len(legend_blocks) - 1)
        height = fh + strip_h + field_gap + margin
        width = max(fw, legend_span) + 2 * margin + extra
        inp = np.full((height, width), bg, dtype=int)
        out = np.full((height, width), bg, dtype=int)

        if side == "top":
            field_top = strip_h + field_gap
            legend_row = 0
        else:
            field_top = margin
            legend_row = None
        field_left = rng.randint(margin, width - fw - margin)
        legend_left = rng.randint(margin, width - legend_span - margin)

        _paste(inp, field_inp, field_top, field_left)
        _paste(out, field_out, field_top, field_left)

        cursor = legend_left
        for block in legend_blocks:
            row = legend_row if legend_row is not None else height - block.shape[0]
            _paste(inp, block, row, cursor)
            _paste(out, block, row, cursor)
            cursor += block.shape[1] + _LEGEND_GAP
        return inp, out

    strip_w = max(block.shape[1] for block in legend_blocks)
    legend_span = sum(block.shape[0] for block in legend_blocks) + _LEGEND_GAP * (len(legend_blocks) - 1)
    height = max(fh, legend_span) + 2 * margin + extra
    width = fw + strip_w + field_gap + margin
    inp = np.full((height, width), bg, dtype=int)
    out = np.full((height, width), bg, dtype=int)

    if side == "left":
        field_left = strip_w + field_gap
        legend_col = 0
    else:
        field_left = margin
        legend_col = None
    field_top = rng.randint(margin, height - fh - margin)
    legend_top = rng.randint(margin, height - legend_span - margin)

    _paste(inp, field_inp, field_top, field_left)
    _paste(out, field_out, field_top, field_left)

    cursor = legend_top
    for block in legend_blocks:
        col = legend_col if legend_col is not None else width - block.shape[1]
        _paste(inp, block, cursor, col)
        _paste(out, block, cursor, col)
        cursor += block.shape[0] + _LEGEND_GAP
    return inp, out


def _closed_seeded_regions(grid: np.ndarray, bg: int, wall: int) -> int:
    legends, legend_mask = _detect_legends(grid, bg, wall)
    count = 0
    for coords in _components((grid != wall) & ~legend_mask):
        if _touches_border(coords, *grid.shape):
            continue
        vals = grid[coords[:, 0], coords[:, 1]]
        seeds = [int(v) for v in np.unique(vals) if int(v) != bg]
        if len(seeds) == 1 and seeds[0] in legends and int(np.count_nonzero(vals == seeds[0])) == 1:
            count += 1
    return count


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    coarse_h = rng.randint(2, 4)
    coarse_w = rng.randint(3, 5)
    max_regions = min(5, coarse_h * coarse_w - 1)
    if max_regions < 2:
        return None
    n_regions = rng.randint(2, max_regions)

    colors = list(range(10))
    rng.shuffle(colors)
    bg = colors[0]
    wall = colors[1]
    seed_colors = colors[2 : 2 + n_regions]
    texture_palette = colors[2 + n_regions :]
    if len(texture_palette) < 2:
        return None

    labels = _partition_labels(rng, coarse_h, coarse_w, n_regions)
    if labels is None:
        return None

    cell_h = rng.randint(3, 5)
    cell_w = rng.randint(3, 5)
    field = _build_field(labels, bg, wall, cell_h, cell_w)
    region_coords = _components(field == bg)
    if len(region_coords) != n_regions:
        return None

    field_inp = field.copy()
    field_out = field.copy()
    legend_blocks: list[np.ndarray] = []

    rng.shuffle(seed_colors)
    for seed_color, coords in zip(seed_colors, region_coords):
        size = rng.choice((2, 3))
        tile = _make_tile(rng, size, texture_palette)
        legend_blocks.append(_render_legend(seed_color, tile))

        seed_r, seed_c = map(int, coords[rng.randrange(len(coords))])
        field_inp[seed_r, seed_c] = seed_color

        r0, c0, _, _ = _bbox(coords)
        th, tw = tile.shape
        for r, c in coords:
            field_out[int(r), int(c)] = int(tile[(int(r) - r0) % th, (int(c) - c0) % tw])

    rng.shuffle(legend_blocks)
    side = rng.choice(("top", "bottom", "left", "right"))
    inp, out = _assemble_canvas(rng, field_inp, field_out, legend_blocks, bg, side)

    if _most_common_color(inp) != bg:
        return None
    if _most_common_non_bg(inp, bg) != wall:
        return None
    legends, _ = _detect_legends(inp, bg, wall)
    if set(legends) != set(seed_colors):
        return None
    if _closed_seeded_regions(inp, bg, wall) != n_regions:
        return None

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {_MAX_RETRIES} attempts")
