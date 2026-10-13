from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
BROWN = 9

FRAME_SIZE = 5
FRAME_INNER_SIZE = FRAME_SIZE - 2
CELL_SIZE = 3
TABLE_COLS = 4  # header column + count columns 1/2/3
MAX_GENERATION_ATTEMPTS = 1000
USABLE_COLORS = [BLUE, RED, GREEN, MAGENTA, ORANGE, AZURE, BROWN]


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_corner(
    bbox: tuple[int, int, int, int], height: int, width: int
) -> bool:
    r0, c0, r1, c1 = bbox
    return (
        (r0 == 0 and c0 == 0)
        or (r0 == 0 and c1 == width - 1)
        or (r1 == height - 1 and c0 == 0)
        or (r1 == height - 1 and c1 == width - 1)
    )


def _parse_lookup_patch(
    patch: np.ndarray, table_color: int, bg: int
) -> dict[tuple[int, int], int] | None:
    h, w = patch.shape
    row_lines = [r for r in range(h) if np.all(patch[r, :] == table_color)]
    col_lines = [c for c in range(w) if np.all(patch[:, c] == table_color)]
    if len(col_lines) != TABLE_COLS + 1 or len(row_lines) < 4:
        return None
    if row_lines[0] != 0 or row_lines[-1] != h - 1:
        return None
    if col_lines[0] != 0 or col_lines[-1] != w - 1:
        return None

    row_steps = [b - a for a, b in zip(row_lines, row_lines[1:])]
    col_steps = [b - a for a, b in zip(col_lines, col_lines[1:])]
    if len(set(row_steps)) != 1 or len(set(col_steps)) != 1:
        return None
    if row_steps[0] < 2 or col_steps[0] < 2 or row_steps[0] != col_steps[0]:
        return None

    def cell(i: int, j: int) -> np.ndarray:
        return patch[row_lines[i] + 1 : row_lines[i + 1], col_lines[j] + 1 : col_lines[j + 1]]

    if np.any(cell(0, 0) != bg):
        return None

    header_counts: list[int] = []
    marker_color = None
    for j in range(1, TABLE_COLS):
        header = cell(0, j)
        colors = {int(v) for v in np.unique(header) if int(v) != bg}
        if len(colors) != 1:
            return None
        col = colors.pop()
        if marker_color is None:
            marker_color = col
        if marker_color != col:
            return None
        header_counts.append(int(np.count_nonzero(header == marker_color)))
    if header_counts != [1, 2, 3]:
        return None

    mapping: dict[tuple[int, int], int] = {}
    seen_row_colors: set[int] = set()
    total_rows = len(row_lines) - 1
    for i in range(1, total_rows):
        row_header = cell(i, 0)
        row_colors = {int(v) for v in np.unique(row_header) if int(v) != bg}
        if len(row_colors) != 1:
            return None
        row_color = row_colors.pop()
        seen_row_colors.add(row_color)
        for j, count in enumerate((1, 2, 3), start=1):
            data = cell(i, j)
            data_colors = {int(v) for v in np.unique(data) if int(v) != bg}
            if len(data_colors) != 1:
                return None
            mapping[(row_color, count)] = data_colors.pop()
    if len(seen_row_colors) < 2:
        return None
    return mapping


def _find_lookup_table(
    grid: np.ndarray, bg: int
) -> tuple[int, tuple[int, int, int, int], dict[tuple[int, int], int]]:
    h, w = grid.shape
    candidates: list[tuple[int, tuple[int, int, int, int], dict[tuple[int, int], int]]] = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        for comp in _components_of_color(grid, color):
            box = _bbox(comp)
            r0, c0, r1, c1 = box
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            mapping = _parse_lookup_patch(patch, color, bg)
            if mapping is not None:
                candidates.append((color, box, mapping))

    corner_candidates = [item for item in candidates if _touches_corner(item[1], h, w)]
    if len(corner_candidates) == 1:
        return corner_candidates[0]
    assert len(candidates) == 1, f"expected one lookup table, got {len(candidates)}"
    return candidates[0]


def _boxes_intersect(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _find_frames(
    grid: np.ndarray,
    bg: int,
    table_bbox: tuple[int, int, int, int],
    table_color: int,
) -> list[dict[str, int | tuple[int, int, int, int]]]:
    frames: list[dict[str, int | tuple[int, int, int, int]]] = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg, table_color}):
        for comp in _components_of_color(grid, color):
            box = _bbox(comp)
            if _boxes_intersect(box, table_bbox):
                continue
            r0, c0, r1, c1 = box
            if r1 - r0 + 1 != c1 - c0 + 1 or r1 - r0 + 1 < 3:
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            expected = np.zeros_like(patch, dtype=bool)
            expected[0, :] = True
            expected[-1, :] = True
            expected[:, 0] = True
            expected[:, -1] = True
            if not np.array_equal(patch == color, expected):
                continue
            interior = patch[1:-1, 1:-1]
            non_bg = interior[interior != bg]
            if len(non_bg) not in (1, 2, 3):
                continue
            if len({int(v) for v in non_bg}) != 1:
                continue
            frames.append({"color": color, "count": int(len(non_bg)), "bbox": box})

    sizes = Counter(int(frame["bbox"][2]) - int(frame["bbox"][0]) + 1 for frame in frames)
    assert sizes, "expected at least one square frame"
    common_size, _ = sizes.most_common(1)[0]
    frames = [frame for frame in frames if int(frame["bbox"][2]) - int(frame["bbox"][0]) + 1 == common_size]
    assert frames, "expected at least one valid frame"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each frame interior with the color found at the intersection of its frame color row and yellow-count column in the lookup table.

    Input structure:
    - The background is black.
    - Several one-pixel-thick hollow square frames of the same size are scattered on the grid; each frame is monochrome and may share its color with other frames.
    - Each frame interior contains 1 to 3 yellow marker pixels on an otherwise black interior.
    - One corner contains a grey lookup table with internal grey separators: the first column holds frame-color headers, the top row holds 1/2/3 yellow-dot headers, and each remaining cell shows one fill color.

    Rule:
    - Detect the lookup table and read, for every frame-color row and marker-count column, the fill color stored in their intersection cell.
    - For each square frame, extract its border color and count the yellow markers inside it.
    - Replace the entire interior of that frame, including the yellow markers, with the looked-up fill color.
    - Keep every frame border and the lookup table unchanged.

    Key insight:
    - The output color is determined by crossing two independently extracted attributes: frame color and the number of yellow markers inside that frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    table_color, table_bbox, mapping = _find_lookup_table(g, bg)
    frames = _find_frames(g, bg, table_bbox, table_color)

    out = g.copy()
    for frame in frames:
        frame_color = int(frame["color"])
        count = int(frame["count"])
        assert (frame_color, count) in mapping, "missing lookup-table entry for frame"
        fill_color = int(mapping[(frame_color, count)])
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = fill_color
    return out


def _table_dimensions(row_count: int) -> tuple[int, int]:
    table_h = (row_count + 1) * CELL_SIZE + (row_count + 2)
    table_w = TABLE_COLS * CELL_SIZE + (TABLE_COLS + 1)
    return table_h, table_w


def _table_position(corner: str, height: int, width: int, table_h: int, table_w: int) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, width - table_w
    if corner == "bl":
        return height - table_h, 0
    if corner == "br":
        return height - table_h, width - table_w
    raise ValueError(f"unknown corner: {corner}")


def _draw_table(
    grid: np.ndarray,
    top: int,
    left: int,
    row_colors: list[int],
    mapping: dict[tuple[int, int], int],
) -> tuple[int, int, int, int]:
    row_count = len(row_colors)
    table_h, table_w = _table_dimensions(row_count)
    bottom = top + table_h - 1
    right = left + table_w - 1

    for i in range(row_count + 2):
        r = top + i * (CELL_SIZE + 1)
        grid[r, left : right + 1] = GREY
    for j in range(TABLE_COLS + 1):
        c = left + j * (CELL_SIZE + 1)
        grid[top : bottom + 1, c] = GREY

    marker_patterns = {
        1: [(1, 1)],
        2: [(0, 0), (2, 2)],
        3: [(0, 2), (1, 1), (2, 0)],
    }

    for count in (1, 2, 3):
        cell_top = top + 1
        cell_left = left + count * (CELL_SIZE + 1) + 1
        for dr, dc in marker_patterns[count]:
            grid[cell_top + dr, cell_left + dc] = YELLOW

    for row_idx, row_color in enumerate(row_colors, start=1):
        cell_top = top + row_idx * (CELL_SIZE + 1) + 1
        cell_left = left + 1
        grid[cell_top : cell_top + CELL_SIZE, cell_left : cell_left + CELL_SIZE] = row_color
        for count in (1, 2, 3):
            fill_color = mapping[(row_color, count)]
            data_top = top + row_idx * (CELL_SIZE + 1) + 1
            data_left = left + count * (CELL_SIZE + 1) + 1
            grid[data_top + 1, data_left + 1] = fill_color

    return top, left, bottom, right


def _marker_positions(rng: random.Random, count: int) -> list[tuple[int, int]]:
    return rng.sample([(r, c) for r in range(FRAME_INNER_SIZE) for c in range(FRAME_INNER_SIZE)], count)


def _mark_occupied(
    occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1
) -> None:
    h, w = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    occupied[r0:r1, c0:c1] = True


def _legal_frame_positions(occupied: np.ndarray) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h - FRAME_SIZE + 1):
        for c0 in range(w - FRAME_SIZE + 1):
            r1 = min(h, r0 + FRAME_SIZE + 1)
            c1 = min(w, c0 + FRAME_SIZE + 1)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            if not occupied[rr0:r1, cc0:c1].any():
                positions.append((r0, c0))
    return positions


def _build_mapping(
    rng: random.Random, row_colors: list[int]
) -> dict[tuple[int, int], int] | None:
    for _ in range(200):
        mapping: dict[tuple[int, int], int] = {}
        for row_color in row_colors:
            row_values = [rng.choice(USABLE_COLORS) for _ in range(3)]
            if len(set(row_values)) == 1:
                continue
            for count, fill_color in zip((1, 2, 3), row_values):
                mapping[(row_color, count)] = fill_color
        if len(mapping) != len(row_colors) * 3:
            continue
        cols_ok = True
        for count in (1, 2, 3):
            if len({mapping[(row_color, count)] for row_color in row_colors}) < 2:
                cols_ok = False
                break
        if cols_ok:
            return mapping
    return None


def _build_frame_specs(
    rng: random.Random, row_colors: list[int], mapping: dict[tuple[int, int], int]
) -> list[tuple[int, int]] | None:
    frame_count = rng.randint(max(4, len(row_colors) + 1), 7)
    for _ in range(200):
        specs = [(rng.choice(row_colors), rng.choice((1, 2, 3))) for _ in range(frame_count)]
        used_colors = {color for color, _ in specs}
        used_counts = {count for _, count in specs}
        by_color: defaultdict[int, set[int]] = defaultdict(set)
        by_count: defaultdict[int, set[int]] = defaultdict(set)
        fill_colors = set()
        for color, count in specs:
            by_color[color].add(count)
            by_count[count].add(color)
            fill_colors.add(mapping[(color, count)])
        if used_colors != set(row_colors):
            continue
        if used_counts != {1, 2, 3}:
            continue
        if max(len(v) for v in by_color.values()) < 2:
            continue
        if max(len(v) for v in by_count.values()) < 2:
            continue
        if len(fill_colors) < 3:
            continue
        return specs
    return None


def _draw_frame(
    grid: np.ndarray, top: int, left: int, frame_color: int, count: int, rng: random.Random
) -> None:
    bottom = top + FRAME_SIZE - 1
    right = left + FRAME_SIZE - 1
    grid[top, left : right + 1] = frame_color
    grid[bottom, left : right + 1] = frame_color
    grid[top : bottom + 1, left] = frame_color
    grid[top : bottom + 1, right] = frame_color
    for dr, dc in _marker_positions(rng, count):
        grid[top + 1 + dr, left + 1 + dc] = YELLOW


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    row_count = rng.randint(2, 4)
    row_colors = rng.sample(USABLE_COLORS, row_count)
    mapping = _build_mapping(rng, row_colors)
    if mapping is None:
        return None
    frame_specs = _build_frame_specs(rng, row_colors, mapping)
    if frame_specs is None:
        return None

    table_h, table_w = _table_dimensions(row_count)
    min_h = max(table_h, 22)
    min_w = max(table_w, 22)
    height = rng.randint(min_h, 30)
    width = rng.randint(min_w, 30)
    grid = np.full((height, width), BLACK, dtype=int)

    corner = rng.choice(["tl", "tr", "bl", "br"])
    table_top, table_left = _table_position(corner, height, width, table_h, table_w)
    table_bbox = _draw_table(grid, table_top, table_left, row_colors, mapping)

    occupied = np.zeros((height, width), dtype=bool)
    _mark_occupied(occupied, table_top, table_left, table_h, table_w, pad=1)

    for frame_color, count in frame_specs:
        positions = _legal_frame_positions(occupied)
        if not positions:
            return None
        top, left = rng.choice(positions)
        _draw_frame(grid, top, left, frame_color, count, rng)
        _mark_occupied(occupied, top, left, FRAME_SIZE, FRAME_SIZE, pad=1)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to create a valid puzzle")
