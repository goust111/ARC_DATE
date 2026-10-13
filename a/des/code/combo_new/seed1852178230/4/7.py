import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
SEPARATOR_COLOR = 8
WALL_COLOR = 3
MARKER_COLOR = 2
TEMPLATE_COLORS = (1, 4, 5, 6, 7, 9)
PATTERN_KINDS = ("hstripe", "vstripe", "diag", "anti", "block", "skew")
SIDES = ("up", "right", "down", "left")
MAX_GENERATION_ATTEMPTS = 400


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _spans(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            spans.append((start, idx - 1))
            start = None
    if start is not None:
        spans.append((start, len(mask) - 1))
    return spans


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
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
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates = [
        c
        for c in range(grid.shape[1])
        if grid[0, c] != bg and np.all(grid[:, c] == grid[0, c])
    ]
    assert len(candidates) == 1, "expected exactly one full-height separator column"
    col = int(candidates[0])
    return col, int(grid[0, col])


def _extract_templates(left: np.ndarray, bg: int) -> list[np.ndarray]:
    row_spans = _spans(np.any(left != bg, axis=1))
    templates: list[np.ndarray] = []
    for r0, r1 in row_spans:
        band = left[r0 : r1 + 1]
        col_spans = _spans(np.any(band != bg, axis=0))
        assert len(col_spans) == 1, "each legend row band must contain one template rectangle"
        c0, c1 = col_spans[0]
        patch = band[:, c0 : c1 + 1]
        assert np.all(patch != bg), "template rectangles must be fully filled"
        templates.append(patch.copy())
    assert templates, "expected at least one template"
    return templates


def _identify_right_colors(right: np.ndarray, bg: int) -> tuple[int, int]:
    info = []
    for color in (int(x) for x in np.unique(right) if int(x) != bg):
        comps = _components(right == color)
        info.append((color, comps, sum(len(comp) for comp in comps)))
    assert len(info) >= 2, "expected wall and marker colors in the target zone"
    wall_color = max(info, key=lambda item: item[2])[0]
    marker_candidates = [
        color for color, comps, _ in info if color != wall_color and all(len(comp) == 1 for comp in comps)
    ]
    assert len(marker_candidates) == 1, "expected exactly one singleton marker color"
    return int(wall_color), int(marker_candidates[0])


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _line_span(flags: np.ndarray, anchor: int) -> tuple[int, int]:
    assert flags[anchor], "anchor must lie on the wall"
    start = int(anchor)
    gap = 0
    pos = int(anchor) - 1
    while pos >= 0:
        if flags[pos]:
            start = pos
            gap = 0
        else:
            gap += 1
            if gap >= 2:
                break
        pos -= 1

    end = int(anchor)
    gap = 0
    pos = int(anchor) + 1
    while pos < len(flags):
        if flags[pos]:
            end = pos
            gap = 0
        else:
            gap += 1
            if gap >= 2:
                break
        pos += 1
    return start, end


def _attached_side(zone: np.ndarray, marker: tuple[int, int], wall_color: int) -> str:
    r, c = marker
    h, w = map(int, zone.shape)
    options = []
    if r + 1 < h and int(zone[r + 1, c]) == wall_color:
        options.append("up")
    if c - 1 >= 0 and int(zone[r, c - 1]) == wall_color:
        options.append("right")
    if r - 1 >= 0 and int(zone[r - 1, c]) == wall_color:
        options.append("down")
    if c + 1 < w and int(zone[r, c + 1]) == wall_color:
        options.append("left")
    assert len(options) == 1, "marker must touch exactly one wall side"
    return options[0]


def _bbox_from_marker(zone: np.ndarray, marker: tuple[int, int], wall_color: int, side: str) -> tuple[int, int, int, int]:
    r, c = marker
    wall_mask = zone == wall_color
    if side == "up":
        top = r + 1
        c0, c1 = _line_span(wall_mask[top], c)
        a0, a1 = _line_span(wall_mask[:, c0], top)
        b0, b1 = _line_span(wall_mask[:, c1], top)
        return min(a0, b0), c0, max(a1, b1), c1
    if side == "down":
        bottom = r - 1
        c0, c1 = _line_span(wall_mask[bottom], c)
        a0, a1 = _line_span(wall_mask[:, c0], bottom)
        b0, b1 = _line_span(wall_mask[:, c1], bottom)
        return min(a0, b0), c0, max(a1, b1), c1
    if side == "left":
        left = c + 1
        r0, r1 = _line_span(wall_mask[:, left], r)
        a0, a1 = _line_span(wall_mask[r0], left)
        b0, b1 = _line_span(wall_mask[r1], left)
        return r0, min(a0, b0), r1, max(a1, b1)
    if side == "right":
        right = c - 1
        r0, r1 = _line_span(wall_mask[:, right], r)
        a0, a1 = _line_span(wall_mask[r0], right)
        b0, b1 = _line_span(wall_mask[r1], right)
        return r0, min(a0, b0), r1, max(a1, b1)
    raise ValueError(side)


def _count_windows(zone: np.ndarray, bbox: tuple[int, int, int, int], wall_color: int, bg: int) -> int:
    r0, c0, r1, c1 = bbox
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "walls must enclose a non-empty interior"
    windows = []
    for r, c in _perimeter_cells(r0, c0, r1, c1):
        if int(zone[r, c]) != wall_color:
            assert int(zone[r, c]) == bg, "window cells must be background"
            windows.append((r, c))
    assert windows, "each wall must contain at least one window"
    return len(windows)


def _rotation_k(side: str) -> int:
    return {"up": 0, "right": 3, "down": 2, "left": 1}[side]


def _tile_patch(patch: np.ndarray, height: int, width: int) -> np.ndarray:
    reps_r = (height + patch.shape[0] - 1) // patch.shape[0]
    reps_c = (width + patch.shape[1] - 1) // patch.shape[1]
    tiled = np.tile(patch, (reps_r, reps_c))
    return tiled[:height, :width].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each walled rectangle with the legend texture indexed by its window count and rotated by the outside marker side.

    Input structure:
    - The background is black and one full-height vertical separator column splits the grid into a left legend zone and a right target zone.
    - The legend zone contains stacked rectangular texture templates separated by black gaps; the first template corresponds to one window, the second to two windows, and so on.
    - The target zone contains rectangular green walls with one or more single-cell window gaps on the perimeter, plus one single-pixel outside marker beside one wall side.

    Rule:
    - Count the missing wall cells on each rectangular perimeter to get the region's window count N.
    - Select the N-th template from the legend, rotate it according to the marker side (top=no rotation, right=clockwise 90, bottom=180, left=counterclockwise 90), and tile it over the interior.
    - Keep the separator, legend, walls, and window gaps unchanged, and erase the outside markers to background.

    Key insight:
    - The wall gaps are an index into the legend, while the outside marker independently determines the template orientation, so both cues are required before filling the interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _mode_color(g)
    sep_col, _ = _find_separator(g, bg)

    templates = _extract_templates(g[:, :sep_col], bg)
    right = g[:, sep_col + 1 :]
    wall_color, marker_color = _identify_right_colors(right, bg)
    marker_components = _components(right == marker_color)
    markers = {tuple(comp[0]) for comp in marker_components}
    assert all(len(comp) == 1 for comp in marker_components)

    out = g.copy()
    for mr, mc in markers:
        out[mr, sep_col + 1 + mc] = bg

    for marker in markers:
        side = _attached_side(right, marker, wall_color)
        bbox = _bbox_from_marker(right, marker, wall_color, side)
        window_count = _count_windows(right, bbox, wall_color, bg)
        assert 1 <= window_count <= len(templates), "window count must map to an existing legend template"
        template = np.rot90(templates[window_count - 1], _rotation_k(side))
        r0, c0, r1, c1 = bbox
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        fill = _tile_patch(template, inner_h, inner_w)
        out[r0 + 1 : r1, sep_col + 1 + c0 + 1 : sep_col + 1 + c1] = fill

    return out


def _pattern_patch(height: int, width: int, color_a: int, color_b: int, kind: str) -> np.ndarray:
    patch = np.full((height, width), color_a, dtype=int)
    for r in range(height):
        for c in range(width):
            if kind == "hstripe":
                idx = r % 2
            elif kind == "vstripe":
                idx = c % 2
            elif kind == "diag":
                idx = (r + c) % 2
            elif kind == "anti":
                idx = (r + 2 * c) % 2
            elif kind == "block":
                idx = ((r // 1) + (c // 2)) % 2
            elif kind == "skew":
                idx = (2 * r + c) % 2
            else:
                raise ValueError(kind)
            patch[r, c] = color_b if idx else color_a
    assert set(map(int, np.unique(patch))) == {int(color_a), int(color_b)}
    return patch


def _noncorner_perimeter(height: int, width: int) -> list[tuple[int, int]]:
    cells = []
    for c in range(1, width - 1):
        cells.append((0, c))
        cells.append((height - 1, c))
    for r in range(1, height - 1):
        cells.append((r, 0))
        cells.append((r, width - 1))
    return cells


def _sample_windows(height: int, width: int, count: int, forbidden: tuple[int, int]) -> list[tuple[int, int]] | None:
    candidates = [cell for cell in _noncorner_perimeter(height, width) if cell != forbidden]
    for _ in range(80):
        shuffled = candidates[:]
        random.shuffle(shuffled)
        chosen: list[tuple[int, int]] = []
        for cell in shuffled:
            if all(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) > 1 for other in chosen):
                chosen.append(cell)
                if len(chosen) == count:
                    return chosen
    return None


def _sample_template_specs() -> tuple[list[np.ndarray], np.ndarray]:
    template_count = random.randint(3, 5)
    kinds = random.sample(PATTERN_KINDS, template_count)
    templates: list[np.ndarray] = []
    widths = []
    heights = []
    for kind in kinds:
        height = random.randint(2, 3)
        width = random.randint(3, 4)
        color_a, color_b = random.sample(TEMPLATE_COLORS, 2)
        patch = _pattern_patch(height, width, color_a, color_b, kind)
        templates.append(patch)
        heights.append(height)
        widths.append(width)

    legend_width = max(widths) + random.randint(1, 3)
    top_pad = random.randint(0, 2)
    bottom_pad = random.randint(0, 2)
    gaps = [random.randint(1, 2) for _ in range(template_count - 1)]
    legend_height = top_pad + bottom_pad + sum(heights) + sum(gaps)
    legend = np.full((legend_height, legend_width), BACKGROUND, dtype=int)

    row = top_pad
    for idx, patch in enumerate(templates):
        col = random.randint(0, legend_width - patch.shape[1])
        legend[row : row + patch.shape[0], col : col + patch.shape[1]] = patch
        row += patch.shape[0]
        if idx + 1 < len(templates):
            row += gaps[idx]

    return templates, legend


def _sample_region_specs(template_count: int) -> list[dict] | None:
    region_count = random.randint(2, 4)
    specs = []
    for idx in range(region_count):
        outer_h = random.randint(4, 6)
        outer_w = random.randint(5, 7)
        count = random.randint(1, template_count)
        side = random.choice(SIDES)
        if side in ("up", "down"):
            marker_rel = (-1 if side == "up" else outer_h, random.randint(1, outer_w - 2))
            forbidden = (0 if side == "up" else outer_h - 1, marker_rel[1])
        else:
            marker_rel = (random.randint(1, outer_h - 2), -1 if side == "left" else outer_w)
            forbidden = (marker_rel[0], 0 if side == "left" else outer_w - 1)
        windows = _sample_windows(outer_h, outer_w, count, forbidden)
        if windows is None:
            return None
        specs.append(
            {
                "id": idx,
                "count": count,
                "side": side,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "marker_rel": marker_rel,
                "windows": windows,
                "reserve_h": outer_h + 2,
                "reserve_w": outer_w + 2,
            }
        )
    return specs


def _pack_regions(specs: list[dict], legend_width: int, legend_height: int) -> tuple[int, int, dict[int, tuple[int, int]]] | None:
    order = specs[:]
    random.shuffle(order)
    for _ in range(120):
        row_count = 1 if len(order) == 1 else random.choice((1, 2))
        if row_count == 1:
            groups = [order]
        else:
            split = random.randint(1, len(order) - 1)
            groups = [order[:split], order[split:]]

        row_gap_values = [random.randint(1, 2) for _ in range(len(groups) - 1)]
        item_gap_values = [[random.randint(1, 2) for _ in range(len(group) - 1)] for group in groups]
        row_widths = []
        row_heights = []
        for group, gaps in zip(groups, item_gap_values):
            row_widths.append(sum(item["reserve_w"] for item in group) + sum(gaps))
            row_heights.append(max(item["reserve_h"] for item in group))

        content_w = max(row_widths)
        content_h = sum(row_heights) + sum(row_gap_values)
        left_pad = random.randint(0, 2)
        right_pad = random.randint(0, 2)
        top_pad = random.randint(0, 2)
        bottom_pad = random.randint(0, 2)
        right_w = left_pad + content_w + right_pad
        right_h = top_pad + content_h + bottom_pad
        global_h_base = max(legend_height, right_h)
        if legend_width + 1 + right_w > 30 or global_h_base > 30:
            continue

        placements: dict[int, tuple[int, int]] = {}
        row_top = top_pad
        for row_idx, group in enumerate(groups):
            row_height = row_heights[row_idx]
            row_width = row_widths[row_idx]
            x = left_pad + random.randint(0, content_w - row_width)
            for item_idx, item in enumerate(group):
                extra_y = row_height - item["reserve_h"]
                box_top = row_top + random.randint(0, extra_y)
                box_left = x
                placements[item["id"]] = (box_top + 1, box_left + 1)
                x += item["reserve_w"]
                if item_idx + 1 < len(group):
                    x += item_gap_values[row_idx][item_idx]
            row_top += row_height
            if row_idx + 1 < len(groups):
                row_top += row_gap_values[row_idx]

        return right_h, right_w, placements
    return None


def _draw_input(
    templates: list[np.ndarray],
    legend: np.ndarray,
    specs: list[dict],
    right_shape: tuple[int, int],
    local_positions: dict[int, tuple[int, int]],
) -> tuple[np.ndarray, list[dict]]:
    right_h, right_w = right_shape
    global_h = max(legend.shape[0], right_h) + random.randint(0, min(2, 30 - max(legend.shape[0], right_h)))
    legend_top = random.randint(0, global_h - legend.shape[0])
    right_top = random.randint(0, global_h - right_h)

    global_w = legend.shape[1] + 1 + right_w
    grid = np.full((global_h, global_w), BACKGROUND, dtype=int)
    grid[:, legend.shape[1]] = SEPARATOR_COLOR
    grid[legend_top : legend_top + legend.shape[0], : legend.shape[1]] = legend

    realized_specs: list[dict] = []
    right_left = legend.shape[1] + 1
    for spec in specs:
        top_local, left_local = local_positions[spec["id"]]
        top = right_top + top_local
        left = right_left + left_local
        outer_h = spec["outer_h"]
        outer_w = spec["outer_w"]
        bottom = top + outer_h - 1
        right = left + outer_w - 1

        grid[top, left : right + 1] = WALL_COLOR
        grid[bottom, left : right + 1] = WALL_COLOR
        grid[top : bottom + 1, left] = WALL_COLOR
        grid[top : bottom + 1, right] = WALL_COLOR
        for wr, wc in spec["windows"]:
            grid[top + wr, left + wc] = BACKGROUND

        mr, mc = spec["marker_rel"]
        marker = (top + mr, left + mc)
        grid[marker] = MARKER_COLOR

        realized = dict(spec)
        realized["top"] = top
        realized["left"] = left
        realized["marker"] = marker
        realized_specs.append(realized)

    return grid, realized_specs


def _expected_output(grid: np.ndarray, templates: list[np.ndarray], specs: Iterable[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        out[spec["marker"]] = BACKGROUND
        template = np.rot90(templates[spec["count"] - 1], _rotation_k(spec["side"]))
        fill = _tile_patch(template, spec["outer_h"] - 2, spec["outer_w"] - 2)
        top = spec["top"]
        left = spec["left"]
        out[top + 1 : top + spec["outer_h"] - 1, left + 1 : left + spec["outer_w"] - 1] = fill
    return out


def _gen_attempt() -> np.ndarray | None:
    templates, legend = _sample_template_specs()
    specs = _sample_region_specs(len(templates))
    if specs is None:
        return None
    packed = _pack_regions(specs, legend.shape[1], legend.shape[0])
    if packed is None:
        return None
    right_h, right_w, local_positions = packed
    grid, realized_specs = _draw_input(templates, legend, specs, (right_h, right_w), local_positions)
    expected = _expected_output(grid, templates, realized_specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
