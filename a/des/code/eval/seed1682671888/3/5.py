from __future__ import annotations

import random

import numpy as np


GREEN = 3
RED = 2
ORANGE = 7
MAROON = 9
PATTERN_COLORS = (1, 4, 5, 6, 8)
BACKGROUND_COLORS = (GREEN, ORANGE)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _regular_intervals(separators: list[int], length: int) -> list[tuple[int, int]]:
    bounds = [-1, *separators, int(length)]
    intervals = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
    sizes = [end - start for start, end in intervals]
    assert min(sizes) > 0, "separators must leave non-empty cells"
    assert len(set(sizes)) == 1, "all macro-cells must have the same size"
    return intervals


def _find_cell_layout(g: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_seps = [r for r in range(g.shape[0]) if np.all(g[r, :] == MAROON)]
    col_seps = [c for c in range(g.shape[1]) if np.all(g[:, c] == MAROON)]
    rows = _regular_intervals(row_seps, g.shape[0])
    cols = _regular_intervals(col_seps, g.shape[1])
    return rows, cols


def _cell_bg(patch: np.ndarray) -> int:
    green_count = int(np.count_nonzero(patch == GREEN))
    orange_count = int(np.count_nonzero(patch == ORANGE))
    assert green_count != orange_count, "cell background must be structurally dominant"
    return GREEN if green_count > orange_count else ORANGE


def _detect_arrow(patch: np.ndarray) -> tuple[int, int] | None:
    red_pts = np.argwhere(patch == RED)
    if red_pts.size == 0:
        return None
    assert red_pts.shape[0] == 3, "each arrow cell must contain exactly three red pixels"

    rows = sorted(int(r) for r in red_pts[:, 0])
    cols = sorted(int(c) for c in red_pts[:, 1])
    h, w = patch.shape

    if len(set(rows)) == 1:
        row = rows[0]
        assert cols[1] == cols[0] + 1 and cols[2] == cols[1] + 1, "horizontal arrow must be consecutive"
        left_gap = cols[0]
        right_gap = (w - 1) - cols[2]
        assert left_gap != right_gap, "horizontal arrow must lean toward one side"
        return DIRS["left"] if left_gap < right_gap else DIRS["right"]

    if len(set(cols)) == 1:
        col = cols[0]
        assert rows[1] == rows[0] + 1 and rows[2] == rows[1] + 1, "vertical arrow must be consecutive"
        up_gap = rows[0]
        down_gap = (h - 1) - rows[2]
        assert up_gap != down_gap, "vertical arrow must lean toward one side"
        return DIRS["up"] if up_gap < down_gap else DIRS["down"]

    raise AssertionError("arrow must be a straight horizontal or vertical triad")


def _mirror_positions(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    if dc != 0:
        return np.fliplr(mask)
    return np.flipud(mask)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each arrow-marked cell's pattern into its neighboring cell, then refill
    the source with a mirrored copy of the target's original pattern.

    Input structure
    - The grid is partitioned into equal rectangular cells by one-pixel maroon
      separators.
    - Cell backgrounds alternate between green and orange in a checkerboard.
    - Some cells contain sparse pattern pixels in colors other than the local
      background, maroon, or red.
    - Some cells contain a three-pixel red arrow marker encoded as a short
      horizontal or vertical run that visibly leans toward one side of the cell.

    Rule
    - Read each red marker as an arrow direction.
    - For every valid arrow, copy that cell's non-background, non-arrow pattern
      into the adjacent target cell at the same local coordinates, but only
      where the target had no original pattern pixel.
    - Clear the pushed pattern from the source cell.
    - Refill the source using the target cell's original pattern mirrored across
      the axis perpendicular to the arrow direction.
    - Erase all red markers. Invalid arrows that point outside the macro-grid
      only get erased.
    """
    g = np.asarray(grid, dtype=int)
    row_cells, col_cells = _find_cell_layout(g)
    cell_h = row_cells[0][1] - row_cells[0][0]
    cell_w = col_cells[0][1] - col_cells[0][0]

    info: dict[tuple[int, int], dict[str, object]] = {}
    for i, (r0, r1) in enumerate(row_cells):
        for j, (c0, c1) in enumerate(col_cells):
            patch = g[r0:r1, c0:c1].copy()
            bg = _cell_bg(patch)
            arrow = _detect_arrow(patch)
            pattern_mask = (patch != bg) & (patch != RED)
            pattern = patch.copy()
            pattern[~pattern_mask] = bg
            info[(i, j)] = {
                "slice": (slice(r0, r1), slice(c0, c1)),
                "bg": int(bg),
                "arrow": arrow,
                "pattern": pattern,
                "mask": pattern_mask,
            }

    out = g.copy()
    for cell in info.values():
        sl_r, sl_c = cell["slice"]
        patch = out[sl_r, sl_c]
        bg = int(cell["bg"])
        patch[patch == RED] = bg

    valid_moves: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    for (i, j), cell in info.items():
        arrow = cell["arrow"]
        if arrow is None:
            continue
        dr, dc = arrow
        ni, nj = i + dr, j + dc
        if 0 <= ni < len(row_cells) and 0 <= nj < len(col_cells):
            valid_moves.append(((i, j), (ni, nj), (dr, dc)))

    for src, _dst, _dir in valid_moves:
        cell = info[src]
        sl_r, sl_c = cell["slice"]
        patch = out[sl_r, sl_c]
        patch[np.asarray(cell["mask"], dtype=bool)] = int(cell["bg"])

    for src, dst, (dr, dc) in valid_moves:
        src_cell = info[src]
        dst_cell = info[dst]
        sl_r, sl_c = src_cell["slice"]
        patch = out[sl_r, sl_c]

        dst_pattern = np.asarray(dst_cell["pattern"], dtype=int)
        dst_mask = np.asarray(dst_cell["mask"], dtype=bool)
        mirrored_values = _mirror_positions(dst_pattern, dr, dc)
        mirrored_mask = _mirror_positions(dst_mask, dr, dc)
        patch[mirrored_mask] = mirrored_values[mirrored_mask]

    for src, dst, _dir in valid_moves:
        src_cell = info[src]
        dst_cell = info[dst]
        src_pattern = np.asarray(src_cell["pattern"], dtype=int)
        src_mask = np.asarray(src_cell["mask"], dtype=bool)
        dst_orig_mask = np.asarray(dst_cell["mask"], dtype=bool)
        write_mask = src_mask & ~dst_orig_mask

        sl_r, sl_c = dst_cell["slice"]
        patch = out[sl_r, sl_c]
        patch[write_mask] = src_pattern[write_mask]

    assert out.shape == g.shape
    assert cell_h > 0 and cell_w > 0
    return out


GEN_RETRIES = 300
CELL_SIZE_CHOICES = (5, 6)
GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (2, 4)
ARROW_COUNT_RANGE = (1, 4)
PATTERN_AREA_RANGE = (2, 5)
SOURCE_PATTERN_PROB = 0.7
OTHER_PATTERN_PROB = 0.55
ACCENT_COLOR_PROB = 0.5


def _compose_canvas(mr: int, mc: int, size: int, top_left_bg: int) -> np.ndarray:
    h = mr * size + (mr - 1)
    w = mc * size + (mc - 1)
    g = np.full((h, w), MAROON, dtype=int)
    for i in range(mr):
        for j in range(mc):
            bg = top_left_bg if (i + j) % 2 == 0 else (GREEN if top_left_bg == ORANGE else ORANGE)
            r0 = i * (size + 1)
            c0 = j * (size + 1)
            g[r0 : r0 + size, c0 : c0 + size] = bg
    return g


def _cell_origin(i: int, j: int, size: int) -> tuple[int, int]:
    return i * (size + 1), j * (size + 1)


def _arrow_coords(size: int, direction: str) -> list[tuple[int, int]]:
    if direction == "left":
        row = random.randint(1, size - 2)
        return [(row, 0), (row, 1), (row, 2)]
    if direction == "right":
        row = random.randint(1, size - 2)
        return [(row, size - 3), (row, size - 2), (row, size - 1)]
    if direction == "up":
        col = random.randint(1, size - 2)
        return [(0, col), (1, col), (2, col)]
    if direction == "down":
        col = random.randint(1, size - 2)
        return [(size - 3, col), (size - 2, col), (size - 1, col)]
    raise ValueError(direction)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _grow_pattern(size: int, blocked: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    allowed = [(r, c) for r in range(size) for c in range(size) if (r, c) not in blocked]
    if len(allowed) < PATTERN_AREA_RANGE[0]:
        return None

    candidates = [pt for pt in allowed if 0 < pt[0] < size - 1 and 0 < pt[1] < size - 1]
    if not candidates:
        candidates = allowed
    target = random.randint(PATTERN_AREA_RANGE[0], min(PATTERN_AREA_RANGE[1], len(allowed)))
    blob = {random.choice(candidates)}
    frontier = {nb for nb in _neighbors4(*next(iter(blob))) if nb in allowed and nb not in blob}

    while len(blob) < target:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        blob.add(cell)
        for nb in _neighbors4(*cell):
            if nb in allowed and nb not in blob:
                frontier.add(nb)

    return sorted(blob)


def _paint_pattern(
    g: np.ndarray,
    origin: tuple[int, int],
    size: int,
    blocked: set[tuple[int, int]],
) -> bool:
    coords = _grow_pattern(size, blocked)
    if coords is None:
        return False

    colors = list(PATTERN_COLORS)
    base = random.choice(colors)
    accent_choices = [color for color in colors if color != base]
    accent = random.choice(accent_choices)
    accent_count = 1 if len(coords) >= 3 and random.random() < ACCENT_COLOR_PROB else 0
    accent_cells = set(random.sample(coords, accent_count)) if accent_count else set()

    r0, c0 = origin
    for r, c in coords:
        g[r0 + r, c0 + c] = accent if (r, c) in accent_cells else base
    return True


def _choose_moves(mr: int, mc: int) -> tuple[list[tuple[tuple[int, int], str]], bool]:
    all_cells = [(i, j) for i in range(mr) for j in range(mc)]
    random.shuffle(all_cells)
    max_arrows = min(ARROW_COUNT_RANGE[1], len(all_cells))
    arrow_total = random.randint(ARROW_COUNT_RANGE[0], max_arrows)

    chosen: list[tuple[tuple[int, int], str]] = []
    sources: set[tuple[int, int]] = set()
    reserved_targets: set[tuple[int, int]] = set()
    has_valid = False

    directions = list(DIRS)
    for _ in range(arrow_total * 4):
        if len(chosen) >= arrow_total:
            break
        src = random.choice(all_cells)
        if src in sources or src in reserved_targets:
            continue

        i, j = src
        random.shuffle(directions)
        for direction in directions:
            dr, dc = DIRS[direction]
            dst = (i + dr, j + dc)
            valid = 0 <= dst[0] < mr and 0 <= dst[1] < mc
            if valid:
                if dst in reserved_targets or dst in sources:
                    continue
            chosen.append((src, direction))
            sources.add(src)
            if valid:
                reserved_targets.add(dst)
                has_valid = True
            break

    return chosen, has_valid


def _make_attempt() -> np.ndarray | None:
    mr = random.randint(*GRID_ROWS_RANGE)
    mc = random.randint(*GRID_COLS_RANGE)
    size = random.choice(CELL_SIZE_CHOICES)
    top_left_bg = random.choice(BACKGROUND_COLORS)
    grid = _compose_canvas(mr, mc, size, top_left_bg)

    arrows, has_valid = _choose_moves(mr, mc)
    if not arrows or not has_valid:
        return None

    arrow_map = {src: direction for src, direction in arrows}
    arrow_pixels = {src: _arrow_coords(size, direction) for src, direction in arrows}
    valid_targets = {
        (src[0] + DIRS[direction][0], src[1] + DIRS[direction][1])
        for src, direction in arrows
        if 0 <= src[0] + DIRS[direction][0] < mr and 0 <= src[1] + DIRS[direction][1] < mc
    }

    has_source_pattern = False
    has_target_pattern = False

    for i in range(mr):
        for j in range(mc):
            origin = _cell_origin(i, j, size)
            bg = top_left_bg if (i + j) % 2 == 0 else (GREEN if top_left_bg == ORANGE else ORANGE)
            blocked: set[tuple[int, int]] = set()
            if (i, j) in arrow_pixels:
                blocked.update(arrow_pixels[(i, j)])

            want_pattern = random.random() < (SOURCE_PATTERN_PROB if (i, j) in arrow_map else OTHER_PATTERN_PROB)
            if (i, j) in valid_targets:
                want_pattern = want_pattern or random.random() < 0.5

            if want_pattern and _paint_pattern(grid, origin, size, blocked):
                if (i, j) in arrow_map:
                    has_source_pattern = True
                if (i, j) in valid_targets:
                    has_target_pattern = True

    for (i, j), direction in arrow_map.items():
        r0, c0 = _cell_origin(i, j, size)
        for r, c in arrow_pixels[(i, j)]:
            if grid[r0 + r, c0 + c] not in BACKGROUND_COLORS:
                return None
            grid[r0 + r, c0 + c] = RED

    if not has_source_pattern:
        src = random.choice(list(arrow_map))
        origin = _cell_origin(*src, size)
        bg = top_left_bg if sum(src) % 2 == 0 else (GREEN if top_left_bg == ORANGE else ORANGE)
        if not _paint_pattern(grid, origin, size, set(arrow_pixels[src])):
            return None
        has_source_pattern = True

    if valid_targets and not has_target_pattern:
        target = random.choice(list(valid_targets))
        origin = _cell_origin(*target, size)
        if not _paint_pattern(grid, origin, size, set()):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _make_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid puzzle instance")
