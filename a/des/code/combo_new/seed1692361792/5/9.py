from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GRID_COLOR = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
CELL_SIZE_CHOICES = (3, 4)
MAX_GENERATION_ATTEMPTS = 400


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected a nonempty grid"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant background color"
    return int(ranked[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    return r0, c0, r1, c1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _find_canvas(
    g: np.ndarray, bg: int
) -> tuple[int, tuple[int, int, int, int], list[int], list[int], int]:
    candidates = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            row_lines = [r for r in range(sub.shape[0]) if np.all(sub[r, :] == color)]
            col_lines = [c for c in range(sub.shape[1]) if np.all(sub[:, c] == color)]
            if len(row_lines) < 3 or len(col_lines) < 3:
                continue
            if row_lines[0] != 0 or row_lines[-1] != sub.shape[0] - 1:
                continue
            if col_lines[0] != 0 or col_lines[-1] != sub.shape[1] - 1:
                continue
            row_steps = np.diff(row_lines)
            col_steps = np.diff(col_lines)
            if len(set(map(int, row_steps))) != 1 or len(set(map(int, col_steps))) != 1:
                continue
            step = int(row_steps[0])
            if step < 2 or not np.all(col_steps == step):
                continue
            expected = np.zeros_like(sub, dtype=bool)
            expected[row_lines, :] = True
            expected[:, col_lines] = True
            if not np.array_equal(sub == color, expected):
                continue
            candidates.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "row_lines": [r0 + r for r in row_lines],
                    "col_lines": [c0 + c for c in col_lines],
                    "cell_size": step - 1,
                    "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                }
            )

    assert len(candidates) == 1, "expected exactly one grey canvas grid"
    cand = candidates[0]
    return cand["color"], cand["bbox"], cand["row_lines"], cand["col_lines"], cand["cell_size"]


def _quadrant(local_r: int, local_c: int, size: int) -> int:
    if size % 2 == 1:
        mid = size // 2
        assert local_r != mid and local_c != mid, "marker may not lie on a center line"
        top = local_r < mid
        left = local_c < mid
    else:
        split = size // 2
        top = local_r < split
        left = local_c < split
    if top and left:
        return 0
    if top and not left:
        return 1
    if not top and left:
        return 2
    return 3


def _transform(tile: np.ndarray, quadrant: int) -> np.ndarray:
    if quadrant == 0:
        return tile.copy()
    if quadrant == 1:
        return np.fliplr(tile)
    if quadrant == 2:
        return np.flipud(tile)
    if quadrant == 3:
        return np.rot90(tile, 2)
    raise AssertionError("unknown quadrant")


def _extract_markers(
    g: np.ndarray,
    bg: int,
    grid_color: int,
    row_lines: list[int],
    col_lines: list[int],
    cell_size: int,
) -> list[dict]:
    markers: list[dict] = []
    for rr in range(len(row_lines) - 1):
        for cc in range(len(col_lines) - 1):
            r0 = row_lines[rr] + 1
            r1 = row_lines[rr + 1]
            c0 = col_lines[cc] + 1
            c1 = col_lines[cc + 1]
            cell = g[r0:r1, c0:c1]
            assert cell.shape == (cell_size, cell_size)
            marker_coords = np.argwhere((cell != bg) & (cell != grid_color))
            assert marker_coords.shape[0] == 1, "each cell must contain exactly one marker"
            lr, lc = map(int, marker_coords[0])
            markers.append(
                {
                    "cell_row": rr,
                    "cell_col": cc,
                    "row0": r0,
                    "col0": c0,
                    "color": int(cell[lr, lc]),
                    "quadrant": _quadrant(lr, lc, cell_size),
                }
            )
    return markers


def _extract_templates(
    g: np.ndarray,
    bg: int,
    canvas_bbox: tuple[int, int, int, int],
    cell_size: int,
    template_colors: set[int],
) -> dict[int, np.ndarray]:
    r0, c0, r1, c1 = canvas_bbox
    outside = np.ones(g.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    templates: dict[int, np.ndarray] = {}
    for color in sorted(template_colors):
        pts = np.argwhere((g == color) & outside)
        assert pts.size > 0, f"expected an external template for color {color}"
        tr0, tc0, tr1, tc1 = _bbox(pts)
        tile = g[tr0 : tr1 + 1, tc0 : tc1 + 1]
        assert tile.shape == (cell_size, cell_size), "template bbox must match cell size"
        vals = {int(v) for v in np.unique(tile)}
        assert vals <= {bg, color}, "template must use only its own color and black"
        templates[color] = tile.copy()
    assert set(templates) == set(template_colors)
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each marked canvas cell with the matching external tile template, oriented by the marker's quadrant.

    Input structure:
    - The background is black, and one grey rectangular canvas in the grid interior is partitioned by equally spaced 1-pixel grey lines into equal square cells.
    - Every cell interior contains exactly one non-black, non-grey marker pixel whose color names a template and whose quadrant position encodes an orientation.
    - Outside the canvas, each template color appears as one square tile of the same size as a cell interior, drawn only with that color and black.

    Rule:
    - Detect the grey canvas grid and infer the cell size from the constant spacing between grey lines.
    - For each cell, use the marker color to select the external tile and use the marker quadrant to choose the transform: top-left = identity, top-right = horizontal flip, bottom-left = vertical flip, bottom-right = 180-degree rotation.
    - Fill every cell interior with the transformed tile, keep the grey grid lines unchanged, and erase all external templates back to black.

    Key insight:
    - The single marker pixel inside a cell encodes two independent pieces of information at once: its color chooses the tile template, and its quadrant chooses the tile orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    bg = _unique_mode(g)
    grid_color, canvas_bbox, row_lines, col_lines, cell_size = _find_canvas(g, bg)
    markers = _extract_markers(g, bg, grid_color, row_lines, col_lines, cell_size)
    template_colors = {marker["color"] for marker in markers}
    templates = _extract_templates(g, bg, canvas_bbox, cell_size, template_colors)

    out = np.full(g.shape, bg, dtype=int)
    r0, c0, r1, c1 = canvas_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = g[r0 : r1 + 1, c0 : c1 + 1]
    for marker in markers:
        tile = _transform(templates[marker["color"]], marker["quadrant"])
        rr = marker["row0"]
        cc = marker["col0"]
        out[rr : rr + cell_size, cc : cc + cell_size] = tile
    return out


def _rand_partition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in [*cuts, total]:
        values.append(cut - prev)
        prev = cut
    return values


def _quadrant_slots(size: int) -> dict[int, list[tuple[int, int]]]:
    if size % 2 == 1:
        mid = size // 2
        top_rows = list(range(0, mid))
        bottom_rows = list(range(mid + 1, size))
        left_cols = list(range(0, mid))
        right_cols = list(range(mid + 1, size))
    else:
        split = size // 2
        top_rows = list(range(0, split))
        bottom_rows = list(range(split, size))
        left_cols = list(range(0, split))
        right_cols = list(range(split, size))
    return {
        0: [(r, c) for r in top_rows for c in left_cols],
        1: [(r, c) for r in top_rows for c in right_cols],
        2: [(r, c) for r in bottom_rows for c in left_cols],
        3: [(r, c) for r in bottom_rows for c in right_cols],
    }


def _random_tile(size: int, color: int) -> np.ndarray | None:
    min_pixels = max(4, size + 1)
    max_pixels = size * size - 1
    if min_pixels > max_pixels:
        return None
    for _ in range(300):
        pixel_count = random.randint(min_pixels, max_pixels)
        cells = {(r, c) for r in range(size) for c in range(size)}
        mandatory = {
            (0, random.randrange(size)),
            (size - 1, random.randrange(size)),
            (random.randrange(size), 0),
            (random.randrange(size), size - 1),
        }
        if len(mandatory) > pixel_count:
            continue
        remaining = list(cells - mandatory)
        chosen = set(mandatory)
        if pixel_count > len(chosen):
            chosen.update(random.sample(remaining, pixel_count - len(chosen)))
        mask = np.zeros((size, size), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)) or np.array_equal(mask, np.rot90(mask, 2)):
            continue
        variants = {
            mask.tobytes(),
            np.fliplr(mask).tobytes(),
            np.flipud(mask).tobytes(),
            np.rot90(mask, 2).tobytes(),
        }
        if len(variants) != 4:
            continue
        tile = np.zeros((size, size), dtype=int)
        tile[mask] = color
        return tile
    return None


def _blank_canvas(rows: int, cols: int, cell_size: int) -> np.ndarray:
    h = rows * cell_size + rows + 1
    w = cols * cell_size + cols + 1
    canvas = np.full((h, w), BG, dtype=int)
    for r in range(rows + 1):
        canvas[r * (cell_size + 1), :] = GRID_COLOR
    for c in range(cols + 1):
        canvas[:, c * (cell_size + 1)] = GRID_COLOR
    return canvas


def _paste(dst: np.ndarray, src: np.ndarray, r0: int, c0: int) -> None:
    h, w = src.shape
    dst[r0 : r0 + h, c0 : c0 + w] = src


def _make_orientation_pool(cell_count: int) -> list[int]:
    counts = _rand_partition(cell_count, 4)
    pool: list[int] = []
    for quadrant, count in enumerate(counts):
        pool.extend([quadrant] * count)
    if not all(q in pool for q in range(4)):
        return _make_orientation_pool(cell_count)
    return pool


def _template_position_ranges(
    grid_h: int,
    grid_w: int,
    canvas_top: int,
    canvas_left: int,
    canvas_h: int,
    canvas_w: int,
    size: int,
) -> dict[str, tuple[range, range]]:
    top = (range(0, canvas_top - size), range(canvas_left, canvas_left + canvas_w - size + 1))
    bottom = (
        range(canvas_top + canvas_h + 1, grid_h - size + 1),
        range(canvas_left, canvas_left + canvas_w - size + 1),
    )
    left = (range(canvas_top, canvas_top + canvas_h - size + 1), range(0, canvas_left - size))
    right = (
        range(canvas_top, canvas_top + canvas_h - size + 1),
        range(canvas_left + canvas_w + 1, grid_w - size + 1),
    )
    return {"top": top, "bottom": bottom, "left": left, "right": right}


def _gen_attempt() -> np.ndarray | None:
    cell_size = random.choice(CELL_SIZE_CHOICES)
    if cell_size == 3:
        dim_options = [(2, 3), (3, 2), (2, 4), (4, 2), (3, 3), (3, 4), (4, 3), (4, 4)]
    else:
        dim_options = [(2, 3), (3, 2), (3, 3)]
    rows, cols = random.choice(dim_options)
    cell_count = rows * cols

    color_count = random.randint(2, min(4, cell_count - 3))
    colors = random.sample(COLOR_POOL, color_count)
    tiles: dict[int, np.ndarray] = {}
    for color in colors:
        tile = _random_tile(cell_size, color)
        if tile is None:
            return None
        tiles[color] = tile

    canvas = _blank_canvas(rows, cols, cell_size)
    canvas_h, canvas_w = map(int, canvas.shape)
    min_h = canvas_h + 2 * (cell_size + 1)
    min_w = canvas_w + 2 * (cell_size + 1)
    if min_h > 30 or min_w > 30:
        return None

    extra_h = random.randint(0, 30 - min_h)
    extra_w = random.randint(0, 30 - min_w)
    top_margin = cell_size + 1 + random.randint(0, extra_h)
    bottom_margin = min_h + extra_h - canvas_h - top_margin
    left_margin = cell_size + 1 + random.randint(0, extra_w)
    right_margin = min_w + extra_w - canvas_w - left_margin
    grid_h = top_margin + canvas_h + bottom_margin
    grid_w = left_margin + canvas_w + right_margin
    canvas_top = top_margin
    canvas_left = left_margin

    zone_ranges = _template_position_ranges(grid_h, grid_w, canvas_top, canvas_left, canvas_h, canvas_w, cell_size)
    color_sides = random.sample(["top", "bottom", "left", "right"], color_count)
    placements: dict[int, tuple[int, int]] = {}
    for color, side in zip(colors, color_sides, strict=False):
        row_range, col_range = zone_ranges[side]
        if len(row_range) == 0 or len(col_range) == 0:
            return None
        placements[color] = (random.choice(list(row_range)), random.choice(list(col_range)))

    positions = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(positions)
    focus_color = colors[0]
    orientation_pool = _make_orientation_pool(cell_count)
    for quadrant in range(4):
        orientation_pool.remove(quadrant)
    random.shuffle(orientation_pool)

    assignments: dict[tuple[int, int], tuple[int, int]] = {}
    for pos, quadrant in zip(positions[:4], range(4), strict=False):
        assignments[pos] = (focus_color, quadrant)
    remaining_positions = positions[4:]
    for pos, color in zip(remaining_positions[: color_count - 1], colors[1:], strict=False):
        assignments[pos] = (color, orientation_pool.pop())
    for pos in remaining_positions[color_count - 1 :]:
        assignments[pos] = (random.choice(colors), orientation_pool.pop())
    assert len(assignments) == cell_count

    quadrant_slots = _quadrant_slots(cell_size)
    inp = np.zeros((grid_h, grid_w), dtype=int)
    out = np.zeros_like(inp)
    _paste(inp, canvas, canvas_top, canvas_left)
    _paste(out, canvas, canvas_top, canvas_left)

    for color, (r0, c0) in placements.items():
        _paste(inp, tiles[color], r0, c0)

    for cell_r in range(rows):
        for cell_c in range(cols):
            color, quadrant = assignments[(cell_r, cell_c)]
            rr = canvas_top + 1 + cell_r * (cell_size + 1)
            cc = canvas_left + 1 + cell_c * (cell_size + 1)
            mr, mc = random.choice(quadrant_slots[quadrant])
            inp[rr + mr, cc + mc] = color
            out[rr : rr + cell_size, cc : cc + cell_size] = _transform(tiles[color], quadrant)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input")
