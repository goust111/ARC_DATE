from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _color_counts(grid: np.ndarray) -> Counter[int]:
    return Counter(map(int, grid.reshape(-1)))


def _dominant_color(grid: np.ndarray) -> int:
    counts = _color_counts(grid)
    return int(max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0])


def _find_wall_color(grid: np.ndarray, bg: int) -> int:
    counts = _color_counts(grid)
    best: tuple[int, int, int] | None = None
    best_color = -1
    for color in counts:
        color = int(color)
        if color == bg:
            continue
        full_rows = sum(bool(np.all(grid[r, :] == color)) for r in range(grid.shape[0]))
        full_cols = sum(bool(np.all(grid[:, c] == color)) for c in range(grid.shape[1]))
        if full_rows < 2 or full_cols < 2:
            continue
        score = (full_rows + full_cols, min(full_rows, full_cols), counts[color])
        if best is None or score > best:
            best = score
            best_color = color
    assert best is not None, "failed to identify wall color"
    return int(best_color)


def _intervals(lines: list[int]) -> list[tuple[int, int]]:
    lines = sorted(int(x) for x in lines)
    spans: list[tuple[int, int]] = []
    for a, b in zip(lines, lines[1:], strict=False):
        if b - a > 1:
            spans.append((a + 1, b - 1))
    assert spans, "expected at least one interior span"
    return spans


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _find_template_cell(
    grid: np.ndarray,
    wall: int,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> tuple[int, int]:
    corner_cells = [
        (0, 0),
        (0, len(col_spans) - 1),
        (len(row_spans) - 1, 0),
        (len(row_spans) - 1, len(col_spans) - 1),
    ]
    found: list[tuple[int, int]] = []
    for ri, ci in corner_cells:
        r0, r1 = row_spans[ri]
        c0, c1 = col_spans[ci]
        cell = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.any(cell == wall):
            continue
        comps = _components(cell != wall)
        if len(comps) >= 2:
            found.append((ri, ci))
    assert len(found) == 1, "expected exactly one corner template cell"
    return found[0]


def _extract_templates(cell: np.ndarray, wall: int) -> list[np.ndarray]:
    comps = _components(cell != wall)
    templates: list[tuple[int, np.ndarray]] = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        patch = cell[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch != wall), "template bbox must be wall-free"
        templates.append((c0, patch.copy()))
    templates.sort(key=lambda item: item[0])
    out = [patch for _, patch in templates]
    assert out, "expected at least one template"
    return out


def _tile(template: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    th, tw = template.shape
    rr = np.arange(h)[:, None] % th
    cc = np.arange(w)[None, :] % tw
    return template[rr, cc]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify the corner template area, count distinct marker colors in each room,
    select the corresponding template, and tile-fill the room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    wall = _find_wall_color(g, bg)

    full_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == wall)]
    full_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == wall)]
    row_spans = _intervals(full_rows)
    col_spans = _intervals(full_cols)

    template_ri, template_ci = _find_template_cell(g, wall, row_spans, col_spans)
    tr0, tr1 = row_spans[template_ri]
    tc0, tc1 = col_spans[template_ci]
    template_cell = g[tr0 : tr1 + 1, tc0 : tc1 + 1]
    templates = _extract_templates(template_cell, wall)

    out = g.copy()
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            if ri == template_ri and ci == template_ci:
                continue
            room = g[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(room == wall), "regular room must not contain wall pixels"
            markers = {int(v) for v in np.unique(room) if int(v) not in (bg, wall)}
            marker_count = len(markers)
            assert 1 <= marker_count <= len(templates), "room marker count must map to a template"
            out[r0 : r1 + 1, c0 : c1 + 1] = _tile(templates[marker_count - 1], room.shape)

    return out


COLOR_IDS = tuple(range(10))
TEMPLATE_COUNT_RANGE = (2, 4)
ROW_COUNT_RANGE = (2, 4)
COL_COUNT_RANGE = (2, 4)
ROOM_HEIGHT_RANGE = (2, 6)
ROOM_WIDTH_RANGE = (2, 7)
TEMPLATE_HEIGHT_RANGE = (2, 4)
TEMPLATE_WIDTH_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 400


def _random_shape(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1])


def _random_template(
    height: int,
    width: int,
    palette: list[int],
    existing: set[tuple[tuple[int, ...], tuple[int, int]]],
) -> np.ndarray:
    for _ in range(200):
        color_count = random.randint(2, min(len(palette), 4))
        colors = random.sample(palette, color_count)
        arr = np.array(
            [[random.choice(colors) for _ in range(width)] for _ in range(height)],
            dtype=int,
        )
        if len(set(map(int, arr.reshape(-1)))) < 2:
            continue
        if height > 1 and all(np.array_equal(arr[0], arr[r]) for r in range(1, height)):
            continue
        if width > 1 and all(np.array_equal(arr[:, 0], arr[:, c]) for c in range(1, width)):
            continue
        sig = (tuple(map(int, arr.reshape(-1))), (height, width))
        if sig in existing:
            continue
        existing.add(sig)
        return arr
    raise RuntimeError("failed to sample a non-degenerate template")


def _build_layout(template_count: int) -> tuple[int, int, int, int]:
    valid: list[tuple[int, int]] = []
    for rows in range(ROW_COUNT_RANGE[0], ROW_COUNT_RANGE[1] + 1):
        for cols in range(COL_COUNT_RANGE[0], COL_COUNT_RANGE[1] + 1):
            if rows * cols - 1 >= template_count:
                valid.append((rows, cols))
    rows, cols = random.choice(valid)
    corner = random.choice(("tl", "tr", "bl", "br"))
    tr = 0 if corner[0] == "t" else rows - 1
    tc = 0 if corner[1] == "l" else cols - 1
    return rows, cols, tr, tc


def _render_case() -> tuple[np.ndarray, np.ndarray]:
    bg, wall = random.sample(COLOR_IDS, 2)
    usable_colors = [c for c in COLOR_IDS if c not in (bg, wall)]
    template_count = random.randint(*TEMPLATE_COUNT_RANGE)
    coarse_rows, coarse_cols, template_row_idx, template_col_idx = _build_layout(template_count)

    template_height = _random_shape(TEMPLATE_HEIGHT_RANGE)
    template_widths = [_random_shape(TEMPLATE_WIDTH_RANGE) for _ in range(template_count)]
    template_total_width = sum(template_widths) + template_count - 1

    row_heights = [_random_shape(ROOM_HEIGHT_RANGE) for _ in range(coarse_rows)]
    col_widths = [_random_shape(ROOM_WIDTH_RANGE) for _ in range(coarse_cols)]
    row_heights[template_row_idx] = template_height
    col_widths[template_col_idx] = template_total_width

    height = sum(row_heights) + coarse_rows + 1
    width = sum(col_widths) + coarse_cols + 1
    grid = np.full((height, width), bg, dtype=int)

    wall_rows = [0]
    for h in row_heights:
        wall_rows.append(wall_rows[-1] + h + 1)
    wall_cols = [0]
    for w in col_widths:
        wall_cols.append(wall_cols[-1] + w + 1)

    for r in wall_rows:
        grid[r, :] = wall
    for c in wall_cols:
        grid[:, c] = wall

    row_spans = [(wall_rows[i] + 1, wall_rows[i + 1] - 1) for i in range(coarse_rows)]
    col_spans = [(wall_cols[i] + 1, wall_cols[i + 1] - 1) for i in range(coarse_cols)]

    tr0, tr1 = row_spans[template_row_idx]
    tc0, tc1 = col_spans[template_col_idx]
    grid[tr0 : tr1 + 1, tc0 : tc1 + 1] = wall

    seen_templates: set[tuple[tuple[int, ...], tuple[int, int]]] = set()
    templates: list[np.ndarray] = []
    cursor = tc0
    for tw in template_widths:
        template = _random_template(template_height, tw, usable_colors, seen_templates)
        templates.append(template)
        grid[tr0 : tr1 + 1, cursor : cursor + tw] = template
        cursor += tw + 1

    regular_rooms: list[tuple[int, int, int, int, int]] = []
    counts = list(range(1, template_count + 1))
    extra = coarse_rows * coarse_cols - 1 - len(counts)
    counts.extend(random.randint(1, template_count) for _ in range(extra))
    random.shuffle(counts)
    count_iter = iter(counts)

    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            if ri == template_row_idx and ci == template_col_idx:
                continue
            room_h = r1 - r0 + 1
            room_w = c1 - c0 + 1
            marker_count = next(count_iter)
            assert room_h * room_w >= marker_count
            marker_positions = random.sample(
                [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)],
                marker_count,
            )
            marker_colors = random.sample(usable_colors, marker_count)
            for (r, c), color in zip(marker_positions, marker_colors, strict=False):
                grid[r, c] = color
            regular_rooms.append((r0, r1, c0, c1, marker_count))

    output = grid.copy()
    for r0, r1, c0, c1, marker_count in regular_rooms:
        output[r0 : r1 + 1, c0 : c1 + 1] = _tile(templates[marker_count - 1], (r1 - r0 + 1, c1 - c0 + 1))

    counts_by_color = _color_counts(grid)
    sorted_counts = counts_by_color.most_common()
    assert len(sorted_counts) >= 2
    if sorted_counts[0][0] != bg or sorted_counts[0][1] <= sorted_counts[1][1]:
        raise RuntimeError("background is not uniquely dominant")

    wall_count = counts_by_color[wall]
    assert counts_by_color[bg] > wall_count, "wall color should not dominate background"
    return grid, output


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid, expected = _render_case()
        except Exception:
            continue
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
