from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, colors_present, get_bg


ROW_BAND_RANGE = (2, 4)
COL_BAND_RANGE = (2, 5)
BAND_SIZE_RANGE = (3, 5)
SOURCE_COUNT_RANGE = (2, 5)
GENERATION_RETRIES = 400


def _regions_between(lines: list[int], limit: int) -> list[tuple[int, int]]:
    bounds = [-1, *sorted(int(x) for x in lines), int(limit)]
    return [(a + 1, b - 1) for a, b in zip(bounds, bounds[1:]) if a + 1 <= b - 1]


def _find_separator_layout(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    matches: list[tuple[int, list[int], list[int]]] = []
    for color in colors_present(grid, exclude={bg}):
        rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == color)]
        cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]
        if not rows or not cols:
            continue

        expected = np.zeros(grid.shape, dtype=bool)
        expected[rows, :] = True
        expected[:, cols] = True
        if np.array_equal(grid == color, expected):
            matches.append((int(color), rows, cols))

    assert len(matches) == 1, "expected exactly one separator-grid color"
    return matches[0]


def _cell_boxes(row_regions: list[tuple[int, int]], col_regions: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    boxes = []
    for r0, r1 in row_regions:
        for c0, c1 in col_regions:
            boxes.append((int(r0), int(c0), int(r1), int(c1)))
    return boxes


def _cell_payload(patch: np.ndarray, bg: int) -> tuple[int, int] | None:
    colors = sorted(set(map(int, patch.ravel())) - {int(bg)})
    assert len(colors) <= 1, "each cell may contain at most one source color"
    if not colors:
        return None
    color = int(colors[0])
    return color, int(np.count_nonzero(patch == color))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat each source cell's pixel count as a row-major address and deliver that color to the addressed table cell.

    Input structure
    - One separator color forms complete horizontal and vertical lines that split the canvas into rectangular table cells.
    - Some cells contain scattered pixels of exactly one non-background color; other cells are empty.
    - Within one input, the source-cell pixel counts are all distinct.

    Rule
    - Keep the separator lines and clear every scattered source pixel.
    - Number the table cells from 1 in row-major order.
    - For each cell containing n pixels of color c, fill cell n completely with color c.
    - All deliveries are based on the original input simultaneously, even when a target cell is itself also a source cell.

    Key insight
    - The pixels inside a cell are not the shape to preserve; their count is an address that points to another cell in the separator table.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    line_color, full_rows, full_cols = _find_separator_layout(g, bg)
    assert line_color != bg

    row_regions = _regions_between(full_rows, g.shape[0])
    col_regions = _regions_between(full_cols, g.shape[1])
    assert len(row_regions) >= 2 and len(col_regions) >= 2, "expected a table with at least 2x2 cells"

    boxes = _cell_boxes(row_regions, col_regions)
    out = np.full_like(g, bg)
    out[g == line_color] = line_color

    transfers: list[tuple[int, int]] = []
    seen_counts: set[int] = set()
    for r0, c0, r1, c1 in boxes:
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        payload = _cell_payload(patch, bg)
        if payload is None:
            continue

        color, count = payload
        assert color != line_color, "source color must differ from separator color"
        assert 1 <= count <= len(boxes), "pixel count must address an existing table cell"
        assert count <= patch.size, "pixel count must fit within its source cell"
        assert count not in seen_counts, "source-cell counts must be unique"

        seen_counts.add(count)
        transfers.append((count, color))

    assert transfers, "expected at least one source cell"

    for count, color in transfers:
        r0, c0, r1, c1 = boxes[count - 1]
        out[r0 : r1 + 1, c0 : c1 + 1] = color

    return out


def _sample_band_sizes(count_range: tuple[int, int]) -> list[int]:
    n = random.randint(*count_range)
    return [random.randint(*BAND_SIZE_RANGE) for _ in range(n)]


def _make_table(row_heights: list[int], col_widths: list[int], bg: int, line_color: int) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    height = sum(row_heights) + len(row_heights) - 1
    width = sum(col_widths) + len(col_widths) - 1
    grid = np.full((height, width), int(bg), dtype=int)

    row_lines: list[int] = []
    pos = row_heights[0]
    for h in row_heights[1:]:
        row_lines.append(int(pos))
        pos += 1 + h

    col_lines: list[int] = []
    pos = col_widths[0]
    for w in col_widths[1:]:
        col_lines.append(int(pos))
        pos += 1 + w

    grid[row_lines, :] = int(line_color)
    grid[:, col_lines] = int(line_color)

    boxes: list[tuple[int, int, int, int]] = []
    r0 = 0
    for h in row_heights:
        c0 = 0
        for w in col_widths:
            boxes.append((int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))
            c0 += w + 1
        r0 += h + 1

    assert max(grid.shape) <= 30
    return grid, boxes


def _choose_mode(n_sources_max: int, count_max: int) -> str:
    modes = ["plain"]
    if n_sources_max >= 2 and count_max >= 2:
        modes.append("cycle")
        modes.append("self")
    if n_sources_max >= 3 and count_max >= 3:
        modes.append("both")
        modes.append("both")
    return random.choice(modes)


def _pick_index(pool: list[int], min_value: int | None = None) -> int:
    candidates = [x for x in pool if min_value is None or x >= min_value]
    if candidates:
        return int(random.choice(candidates))
    return int(random.choice(pool))


def _choose_assignments(num_cells: int, count_max: int, col_count: int, n_sources: int, mode: str) -> list[tuple[int, int]]:
    all_sources = list(range(1, num_cells + 1))
    targetable_sources = list(range(1, count_max + 1))
    chosen_sources: list[int] = []
    count_by_source: dict[int, int] = {}

    want_late_target = count_max > col_count
    late_threshold = col_count + 1

    if mode in {"self", "both"}:
        s = _pick_index(targetable_sources, min_value=late_threshold if want_late_target else None)
        chosen_sources.append(s)
        count_by_source[s] = s

    if mode in {"cycle", "both"}:
        pool = [x for x in targetable_sources if x not in chosen_sources]
        a = _pick_index(pool, min_value=late_threshold if want_late_target and not any(v > col_count for v in count_by_source.values()) else None)
        pool.remove(a)
        b = _pick_index(pool)
        chosen_sources.extend([a, b])
        count_by_source[a] = b
        count_by_source[b] = a

    remaining_sources = n_sources - len(chosen_sources)
    extras = random.sample([x for x in all_sources if x not in chosen_sources], remaining_sources)
    chosen_sources.extend(extras)
    random.shuffle(chosen_sources)

    available_counts = [x for x in range(1, count_max + 1) if x not in count_by_source.values()]
    undecided_sources = [src for src in chosen_sources if src not in count_by_source]

    if undecided_sources and not any(v > col_count for v in count_by_source.values()) and want_late_target:
        late_counts = [x for x in available_counts if x > col_count]
        if late_counts:
            forced_source = undecided_sources.pop()
            forced_count = int(random.choice(late_counts))
            count_by_source[forced_source] = forced_count
            available_counts.remove(forced_count)

    random.shuffle(available_counts)
    for src in undecided_sources:
        count_by_source[src] = int(available_counts.pop())

    pairs = [(int(src), int(count_by_source[src])) for src in chosen_sources]
    assert len({src for src, _ in pairs}) == n_sources
    assert len({count for _, count in pairs}) == n_sources
    assert any(count > col_count for _, count in pairs), "need at least one non-first-row target"
    if mode == "self":
        assert any(src == count for src, count in pairs)
    if mode == "cycle":
        assert any(src2 == count1 and src1 == count2 for src1, count1 in pairs for src2, count2 in pairs if src1 != src2)
    if mode == "both":
        assert any(src == count for src, count in pairs)
        assert any(src2 == count1 and src1 == count2 for src1, count1 in pairs for src2, count2 in pairs if src1 != src2)
    return pairs


def _scatter_coords(box: tuple[int, int, int, int], count: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    coords = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    return random.sample(coords, int(count))


def _build_expected_output(inp: np.ndarray, boxes: list[tuple[int, int, int, int]], line_color: int, bg: int, assignments: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.full_like(inp, int(bg))
    out[inp == int(line_color)] = int(line_color)
    for _source_index, count, color in assignments:
        r0, c0, r1, c1 = boxes[count - 1]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def _validate_construction(inp: np.ndarray, boxes: list[tuple[int, int, int, int]], bg: int, line_color: int, assignments: list[tuple[int, int, int]]) -> bool:
    planned = {int(src): (int(count), int(color)) for src, count, color in assignments}
    for idx, (r0, c0, r1, c1) in enumerate(boxes, start=1):
        patch = inp[r0 : r1 + 1, c0 : c1 + 1]
        colors = sorted(set(map(int, patch.ravel())) - {int(bg)})
        if idx not in planned:
            if colors:
                return False
            continue
        if len(colors) != 1 or colors[0] != planned[idx][1]:
            return False
        if int(np.count_nonzero(patch == planned[idx][1])) != planned[idx][0]:
            return False
        if int(line_color) in colors:
            return False
    return True


def _attempt_generate() -> np.ndarray | None:
    row_heights = _sample_band_sizes(ROW_BAND_RANGE)
    col_widths = _sample_band_sizes(COL_BAND_RANGE)
    base_cell_area = min(h * w for h in row_heights for w in col_widths)

    num_cells = len(row_heights) * len(col_widths)
    count_max = min(num_cells, base_cell_area - 1)
    if count_max < 2 or count_max <= len(col_widths):
        return None

    palette = list(Color.ALL_COLORS)
    bg = int(random.choice(palette))
    palette.remove(bg)
    line_color = int(random.choice(palette))
    palette.remove(line_color)

    n_sources_max = min(SOURCE_COUNT_RANGE[1], len(palette), count_max, num_cells)
    if n_sources_max < SOURCE_COUNT_RANGE[0]:
        return None

    mode = _choose_mode(n_sources_max, count_max)
    n_sources_min = 3 if mode == "both" else 2
    n_sources = random.randint(n_sources_min, n_sources_max)

    base, boxes = _make_table(row_heights, col_widths, bg, line_color)
    pairs = _choose_assignments(num_cells, count_max, len(col_widths), n_sources, mode)
    colors = random.sample(palette, n_sources)

    inp = base.copy()
    assignments: list[tuple[int, int, int]] = []
    for color, (source_index, count) in zip(colors, pairs):
        box = boxes[source_index - 1]
        for r, c in _scatter_coords(box, count):
            inp[r, c] = int(color)
        assignments.append((int(source_index), int(count), int(color)))

    if not _validate_construction(inp, boxes, bg, line_color, assignments):
        return None

    expected = _build_expected_output(inp, boxes, line_color, bg, assignments)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid count-address table input")
