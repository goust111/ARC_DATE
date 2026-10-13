import random
from collections import Counter

import numpy as np


BG = 0
ARROW = 2
SEP = 9
BLOCK_COLORS = (1, 3, 4, 5, 6, 7, 8)

DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
DIR_ORDER = ("up", "right", "down", "left")

# The four rotated 2x2 L-trominoes that encode the four pointer directions.
ARROW_PATTERNS = {
    "up": ((0, 0), (0, 1), (1, 0)),
    "right": ((0, 0), (0, 1), (1, 1)),
    "down": ((0, 1), (1, 0), (1, 1)),
    "left": ((0, 0), (1, 0), (1, 1)),
}
PATTERN_TO_DIR = {frozenset(coords): name for name, coords in ARROW_PATTERNS.items()}

SHAPES = (
    np.array([[1, 1]], dtype=int),
    np.array([[1], [1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [1, 1]], dtype=int),
)

GRID_OPTIONS = ((2, 2), (2, 3), (3, 2))
MIN_CELL = 5
MAX_CELL = 8
MAX_GENERATION_ATTEMPTS = 400


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transport each regional sprite into the adjacent tile named by its red L-pointer, mirroring the landing position.

    Input structure
    - A maroon separator lattice, including the outer border, partitions the grid into equal-sized rectangular regions.
    - Each region contains exactly one red 3-cell L-shaped pointer in a 2x2 bbox, rotated to encode one of the four cardinal directions.
    - Each region also contains one non-background monochrome mini-sprite, and every pointer targets a valid neighboring region.

    Rule
    - For each region, read the pointer direction and identify the adjacent target region in that direction.
    - Measure the sprite coordinates relative to the source region, clear every non-separator cell, and move the sprite into the target region.
    - Mirror the moved sprite across the target region's axis parallel to the move: left/right moves flip columns, up/down moves flip rows.
    - Process source regions in reading order so later transfers overwrite earlier ones on any shared destination cells.

    Key insight
    - The red L marker is a transport control, not a foreground object to keep, and the destination pose is a mirrored version of the source sprite rather than a direct copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    sep_color, sep_rows, sep_cols = _separator_lines(g)
    row_spans = _spans_from_separators(sep_rows, g.shape[0])
    col_spans = _spans_from_separators(sep_cols, g.shape[1])

    tile_heights = {r1 - r0 + 1 for r0, r1 in row_spans}
    tile_widths = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(tile_heights) == 1 and len(tile_widths) == 1, "expected equal-sized regions"
    tile_h = tile_heights.pop()
    tile_w = tile_widths.pop()

    bg = _mode_color(g[g != sep_color])
    assert bg != sep_color

    transfers = []
    out = np.full_like(g, bg)
    out[sep_rows, :] = sep_color
    out[:, sep_cols] = sep_color

    region_rows = len(row_spans)
    region_cols = len(col_spans)
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            assert tile.shape == (tile_h, tile_w)

            arrow_cells = [tuple(map(int, rc)) for rc in np.argwhere(tile == ARROW)]
            assert len(arrow_cells) == 3, "each region must contain exactly one 3-cell arrow"
            direction = _decode_arrow(arrow_cells)

            dr, dc = DIRS[direction]
            tri = ri + dr
            tci = ci + dc
            assert 0 <= tri < region_rows and 0 <= tci < region_cols, "pointer must target an in-bounds neighbor"

            sprite_mask = (tile != bg) & (tile != sep_color) & (tile != ARROW)
            sprite_colors = {int(v) for v in np.unique(tile[sprite_mask])}
            assert len(sprite_colors) == 1, "expected one monochrome sprite per region"
            sprite_color = sprite_colors.pop()
            sprite_cells = [tuple(map(int, rc)) for rc in np.argwhere(tile == sprite_color)]
            assert sprite_cells, "sprite cannot be empty"

            transfers.append((tri, tci, direction, sprite_color, sprite_cells))

    for tri, tci, direction, sprite_color, sprite_cells in transfers:
        rr0, cc0 = row_spans[tri][0], col_spans[tci][0]
        mirrored = _mirror_cells(sprite_cells, (tile_h, tile_w), direction)
        for rr, cc in mirrored:
            out[rr0 + rr, cc0 + cc] = sprite_color

    return out


def _mode_color(values: np.ndarray) -> int:
    counts = np.bincount(np.asarray(values, dtype=int).ravel())
    return int(np.argmax(counts))


def _separator_lines(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    row_groups: dict[int, list[int]] = {}
    col_groups: dict[int, list[int]] = {}
    h, w = grid.shape

    for r in range(h):
        if np.all(grid[r] == grid[r, 0]):
            row_groups.setdefault(int(grid[r, 0]), []).append(r)
    for c in range(w):
        if np.all(grid[:, c] == grid[0, c]):
            col_groups.setdefault(int(grid[0, c]), []).append(c)

    candidates = []
    for color, rows in row_groups.items():
        cols = col_groups.get(color, [])
        if len(rows) < 3 or len(cols) < 3:
            continue
        if rows[0] != 0 or rows[-1] != h - 1 or cols[0] != 0 or cols[-1] != w - 1:
            continue
        row_gaps = [rows[i + 1] - rows[i] for i in range(len(rows) - 1)]
        col_gaps = [cols[i + 1] - cols[i] for i in range(len(cols) - 1)]
        if len(set(row_gaps)) == 1 and len(set(col_gaps)) == 1 and row_gaps[0] >= 2 and col_gaps[0] >= 2:
            candidates.append((color, rows, cols))

    assert len(candidates) == 1, "expected one separator lattice color"
    color, rows, cols = candidates[0]
    return int(color), rows, cols


def _spans_from_separators(lines: list[int], limit: int) -> list[tuple[int, int]]:
    assert lines[0] == 0 and lines[-1] == limit - 1
    spans = []
    for a, b in zip(lines, lines[1:]):
        assert b - a >= 2, "separator lines must enclose a nonempty region"
        spans.append((a + 1, b - 1))
    return spans


def _decode_arrow(cells: list[tuple[int, int]]) -> str:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    assert r1 - r0 == 1 and c1 - c0 == 1, "arrow must fit in a 2x2 bbox"
    normalized = frozenset((r - r0, c - c0) for r, c in cells)
    direction = PATTERN_TO_DIR.get(normalized)
    assert direction is not None, "red pixels must form a recognized L-arrow"
    return direction


def _mirror_cells(cells: list[tuple[int, int]], shape: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    h, w = shape
    mirrored = []
    if direction in ("left", "right"):
        for r, c in cells:
            mirrored.append((r, w - 1 - c))
    else:
        for r, c in cells:
            mirrored.append((h - 1 - r, c))
    return mirrored


def _cell_origins(region_rows: int, region_cols: int, cell_h: int, cell_w: int) -> dict[tuple[int, int], tuple[int, int]]:
    origins = {}
    for ri in range(region_rows):
        for ci in range(region_cols):
            origins[(ri, ci)] = (1 + ri * (cell_h + 1), 1 + ci * (cell_w + 1))
    return origins


def _valid_dirs(ri: int, ci: int, region_rows: int, region_cols: int) -> list[str]:
    valid = []
    for name, (dr, dc) in DIRS.items():
        nr, nc = ri + dr, ci + dc
        if 0 <= nr < region_rows and 0 <= nc < region_cols:
            valid.append(name)
    return valid


def _sample_directions(region_rows: int, region_cols: int, want_merge: bool) -> dict[tuple[int, int], str] | None:
    coords = [(ri, ci) for ri in range(region_rows) for ci in range(region_cols)]
    for _ in range(200):
        directions = {
            coord: random.choice(_valid_dirs(coord[0], coord[1], region_rows, region_cols))
            for coord in coords
        }
        used = set(directions.values())
        if len(coords) >= 4 and len(used) < 3:
            continue
        if not any(d in ("left", "right") for d in used):
            continue
        if not any(d in ("up", "down") for d in used):
            continue

        indegrees = Counter()
        for (ri, ci), direction in directions.items():
            dr, dc = DIRS[direction]
            indegrees[(ri + dr, ci + dc)] += 1
        has_merge = any(v >= 2 for v in indegrees.values())
        if want_merge and not has_merge:
            continue
        if not want_merge and has_merge and random.random() < 0.7:
            continue
        return directions
    return None


def _arrow_origin_candidates(cell_h: int, cell_w: int, direction: str) -> list[tuple[int, int]]:
    max_r = cell_h - 2
    max_c = cell_w - 2
    if direction == "up":
        row_choices = range(0, min(2, max_r) + 1)
        col_choices = range(0, max_c + 1)
    elif direction == "down":
        start = max(0, max_r - 2)
        row_choices = range(start, max_r + 1)
        col_choices = range(0, max_c + 1)
    elif direction == "left":
        row_choices = range(0, max_r + 1)
        col_choices = range(0, min(2, max_c) + 1)
    else:
        row_choices = range(0, max_r + 1)
        start = max(0, max_c - 2)
        col_choices = range(start, max_c + 1)
    return [(r, c) for r in row_choices for c in col_choices]


def _expanded_cells(cells: set[tuple[int, int]], cell_h: int, cell_w: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < cell_h and 0 <= nc < cell_w:
                    out.add((nr, nc))
    return out


def _shape_cells(shape: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    coords = np.argwhere(shape > 0)
    return [(int(r) + top, int(c) + left) for r, c in coords]


def _candidate_sprite_placements(
    cell_h: int,
    cell_w: int,
    direction: str,
    shape: np.ndarray,
    forbidden: set[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    sh, sw = shape.shape
    candidates = []
    for top in range(cell_h - sh + 1):
        for left in range(cell_w - sw + 1):
            cells = _shape_cells(shape, top, left)
            if any(cell in forbidden for cell in cells):
                continue
            if sorted(cells) == sorted(_mirror_cells(cells, (cell_h, cell_w), direction)):
                continue
            candidates.append(cells)
    random.shuffle(candidates)
    return candidates


def _gen_attempt() -> np.ndarray | None:
    region_rows, region_cols = random.choice(GRID_OPTIONS)
    max_h = min(MAX_CELL, (30 - (region_rows + 1)) // region_rows)
    max_w = min(MAX_CELL, (30 - (region_cols + 1)) // region_cols)
    if max_h < MIN_CELL or max_w < MIN_CELL:
        return None

    cell_h = random.randint(MIN_CELL, max_h)
    cell_w = random.randint(MIN_CELL, max_w)
    want_merge = random.random() < 0.5
    directions = _sample_directions(region_rows, region_cols, want_merge)
    if directions is None:
        return None

    region_count = region_rows * region_cols
    colors = random.sample(BLOCK_COLORS, region_count)
    origins = _cell_origins(region_rows, region_cols, cell_h, cell_w)

    regions: dict[tuple[int, int], dict[str, object]] = {}
    for idx, coord in enumerate((ri, ci) for ri in range(region_rows) for ci in range(region_cols)):
        direction = directions[coord]
        arrow_candidates = _arrow_origin_candidates(cell_h, cell_w, direction)
        random.shuffle(arrow_candidates)

        chosen = None
        shape_choices = list(SHAPES)
        random.shuffle(shape_choices)
        for arrow_top, arrow_left in arrow_candidates:
            arrow_cells = {
                (arrow_top + dr, arrow_left + dc)
                for dr, dc in ARROW_PATTERNS[direction]
            }
            forbidden = _expanded_cells(arrow_cells, cell_h, cell_w)
            for shape in shape_choices:
                placements = _candidate_sprite_placements(cell_h, cell_w, direction, shape, forbidden)
                if placements:
                    chosen = {
                        "direction": direction,
                        "arrow_cells": arrow_cells,
                        "sprite_cells": placements[0],
                        "color": colors[idx],
                    }
                    break
            if chosen is not None:
                break
        if chosen is None:
            return None
        regions[coord] = chosen

    arrivals: dict[tuple[int, int], list[tuple[int, list[tuple[int, int]]]]] = {}
    for ri in range(region_rows):
        for ci in range(region_cols):
            region = regions[(ri, ci)]
            direction = region["direction"]
            dr, dc = DIRS[direction]
            target = (ri + dr, ci + dc)
            mirrored = _mirror_cells(region["sprite_cells"], (cell_h, cell_w), direction)
            arrivals.setdefault(target, []).append((int(region["color"]), mirrored))

    merge_count = 0
    for inbound in arrivals.values():
        if len(inbound) >= 2:
            merge_count += 1
        occupied = {}
        for color, cells in inbound:
            for cell in cells:
                if cell in occupied and occupied[cell] != color:
                    return None
                occupied[cell] = color

    if want_merge and merge_count == 0:
        return None

    height = region_rows * cell_h + region_rows + 1
    width = region_cols * cell_w + region_cols + 1
    grid = np.full((height, width), BG, dtype=int)
    sep_rows = [i * (cell_h + 1) for i in range(region_rows + 1)]
    sep_cols = [i * (cell_w + 1) for i in range(region_cols + 1)]
    grid[sep_rows, :] = SEP
    grid[:, sep_cols] = SEP

    for ri in range(region_rows):
        for ci in range(region_cols):
            r0, c0 = origins[(ri, ci)]
            region = regions[(ri, ci)]
            for rr, cc in region["arrow_cells"]:
                grid[r0 + rr, c0 + cc] = ARROW
            for rr, cc in region["sprite_cells"]:
                grid[r0 + rr, c0 + cc] = int(region["color"])

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    if np.any(out == ARROW):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pointer-transport input")
