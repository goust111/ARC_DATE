from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg, paste
import math
import random
from itertools import product


Cell = tuple[int, int, int]
Placement = tuple[int, int, list[Cell]]


def _component_cells(g: np.ndarray, coords: np.ndarray) -> list[Cell]:
    return [(int(r), int(c), int(g[int(r), int(c)])) for r, c in coords]


def _find_placements(
    g: np.ndarray,
    bg: int,
    template: list[Cell],
    all_template_coords: set[tuple[int, int]],
    target_cells: list[Cell],
) -> list[Placement]:
    h, w = g.shape
    shifts: set[tuple[int, int]] = set()
    for tr, tc, target_color in target_cells:
        for sr, sc, source_color in template:
            if target_color == source_color:
                shifts.add((tr - sr, tc - sc))

    candidates: list[tuple[int, int, int, list[Cell]]] = []
    for dr, dc in shifts:
        placed = [(r + dr, c + dc, color) for r, c, color in template]
        if not all(0 <= r < h and 0 <= c < w for r, c, _color in placed):
            continue
        if any((r, c) in all_template_coords for r, c, _color in placed):
            continue

        overlap: list[Cell] = []
        conflict = False
        for r, c, color in placed:
            existing = int(g[r, c])
            if existing == bg:
                continue
            if existing != color:
                conflict = True
                break
            overlap.append((r, c, color))

        if not conflict and overlap:
            candidates.append((len(overlap), dr, dc, overlap))

    assert candidates, "each source sprite must have at least one matching target mask"
    max_overlap = max(overlap_count for overlap_count, _dr, _dc, _overlap in candidates)
    selected = [
        (dr, dc, overlap)
        for overlap_count, dr, dc, overlap in candidates
        if overlap_count == max_overlap
    ]

    used_targets: set[tuple[int, int]] = set()
    for _dr, _dc, overlap in selected:
        for r, c, _color in overlap:
            assert (r, c) not in used_targets, "target masks for one sprite must be disjoint"
            used_targets.add((r, c))
    return selected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each partial marker mask by stamping the matching multicolor sprite over it.

    Input structure
    - A dominant background contains one or more 8-connected multicolor sprites.
    - Separate target masks use colors that already occur inside one of those sprites.
    - A target mask may contain only one marker pixel or several same-alignment marker pixels.

    Rule
    - For each multicolor sprite, test every translation that aligns a target pixel with a
      same-color sprite pixel.
    - Keep the translations with maximal exact overlap against the existing target pixels,
      rejecting translations that conflict with other foreground or overlap a source sprite.
    - Stamp the whole sprite at those translations; if a non-marker sprite color occurs as
      a single source pixel, consume that source-only marker back to background.

    Key insight
    The scattered target pixels are partial copies: their same-color overlap with a sprite
    determines where the full sprite must be completed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=8, monochromatic=False)
    templates: list[list[Cell]] = []
    for component in components:
        cells = _component_cells(g, component["coords"])
        colors = {color for _r, _c, color in cells}
        if len(colors) > 1:
            templates.append(cells)

    assert templates, "expected at least one multicolor source sprite"
    all_template_coords = {(r, c) for template in templates for r, c, _color in template}
    target_cells = [
        (int(r), int(c), int(g[int(r), int(c)]))
        for r, c in np.argwhere(g != bg)
        if (int(r), int(c)) not in all_template_coords
    ]
    assert target_cells, "expected separate target marker pixels"

    template_colors = {color for template in templates for _r, _c, color in template}
    assert {color for _r, _c, color in target_cells} <= template_colors

    out = g.copy()
    for template in templates:
        placements = _find_placements(g, bg, template, all_template_coords, target_cells)

        marker_colors = {color for _dr, _dc, overlap in placements for _r, _c, color in overlap}
        for dr, dc, _overlap in placements:
            for r, c, color in template:
                rr = r + dr
                cc = c + dc
                assert int(out[rr, cc]) in {bg, color}, "sprite placements must not collide"
                out[rr, cc] = color

        for color in {color for _r, _c, color in template} - marker_colors:
            source_cells = [(r, c) for r, c, cell_color in template if cell_color == color]
            if len(source_cells) == 1:
                r, c = source_cells[0]
                out[r, c] = bg

    return out

# --------------- generator (att2) ---------------

GRID_SIZE_RANGE = (13, 24)
PALETTE_SIZE = 10
FOREGROUND_RATE_RANGE = (0.04, 0.06)
TEMPLATE_COUNT_RANGE = (1, 3)
GROUPS_PER_TEMPLATE = (1, 2)
MARKER_CELLS_PER_GROUP = (1, 2)
TOTAL_GROUP_RANGE = (2, 4)
TEMPLATE_CELL_RANGE = (5, 10)
TEMPLATE_BBOX_RANGE = (3, 5)
TEMPLATE_COLOR_RANGE = (2, 4)
EXTRA_COLOR_STOP_PROB = 0.45
ERASURE_TEMPLATE_PROB = 0.12
MAX_GENERATION_ATTEMPTS = 300


def _gen_group_options(template_count: int) -> list[tuple[int, ...]]:
    return [
        counts
        for counts in product(GROUPS_PER_TEMPLATE, repeat=template_count)
        if TOTAL_GROUP_RANGE[0] <= sum(counts) <= TOTAL_GROUP_RANGE[1]
    ]


def _gen_skeleton_options(size: int) -> list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
    min_fg = math.ceil(FOREGROUND_RATE_RANGE[0] * size * size)
    max_fg = math.floor(FOREGROUND_RATE_RANGE[1] * size * size)
    options = []
    for template_count in range(TEMPLATE_COUNT_RANGE[0], TEMPLATE_COUNT_RANGE[1] + 1):
        for group_counts in _gen_group_options(template_count):
            for marker_counts in product(MARKER_CELLS_PER_GROUP, repeat=template_count):
                marker_fg = sum(g * m for g, m in zip(group_counts, marker_counts))
                cell_range = range(TEMPLATE_CELL_RANGE[0], TEMPLATE_CELL_RANGE[1] + 1)
                for cell_counts in product(cell_range, repeat=template_count):
                    foreground = marker_fg + sum(cell_counts)
                    if min_fg <= foreground <= max_fg:
                        options.append((group_counts, marker_counts, cell_counts))
    return options


def _gen_color_counts(
    cell_count: int,
    color_count: int,
    marker_count: int,
    erasure_index: int | None,
) -> list[int] | None:
    counts = [0] * color_count
    counts[1] = marker_count
    for idx in range(2, color_count):
        counts[idx] = 1 if idx == erasure_index else 2

    dominant = cell_count - sum(counts[1:])
    if dominant <= max(counts[1:]):
        return None
    counts[0] = dominant

    for _ in range(cell_count):
        valid = []
        for idx in range(2, color_count):
            new_dominant = counts[0] - 1
            new_minorities = counts[1:].copy()
            new_minorities[idx - 1] += 1
            if new_dominant > max(new_minorities):
                valid.append(idx)
        if not valid or random.random() < EXTRA_COLOR_STOP_PROB:
            break
        chosen = random.choice(valid)
        counts[0] -= 1
        counts[chosen] += 1
    return counts


def _gen_corner_path(height: int, width: int) -> list[tuple[int, int]]:
    start_r = random.choice((0, height - 1))
    start_c = random.choice((0, width - 1))
    end_r = height - 1 - start_r
    end_c = width - 1 - start_c
    steps = max(height - 1, width - 1)
    row_steps = set(random.sample(range(steps), height - 1))
    col_steps = set(random.sample(range(steps), width - 1))
    row_dir = 1 if end_r > start_r else -1
    col_dir = 1 if end_c > start_c else -1

    row, col = start_r, start_c
    coords = [(row, col)]
    for step in range(steps):
        if step in row_steps:
            row += row_dir
        if step in col_steps:
            col += col_dir
        coords.append((row, col))
    return coords


def _gen_shape_coords(height: int, width: int, cell_count: int) -> list[tuple[int, int]]:
    cells = set(_gen_corner_path(height, width))
    while len(cells) < cell_count:
        frontier = []
        for row, col in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    if len(cells) != cell_count:
        return []
    return list(cells)


def _gen_adjacent_pair(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pairs = []
    for idx, first in enumerate(coords):
        for second in coords[idx + 1 :]:
            if max(abs(first[0] - second[0]), abs(first[1] - second[1])) == 1:
                pairs.append([first, second])
    return random.choice(pairs) if pairs else []


def _gen_color_options(
    cell_count: int,
    marker_count: int,
    max_colors: int,
    want_erasure: bool,
) -> list[tuple[int, int | None, list[int]]]:
    options = []
    for color_count in range(TEMPLATE_COLOR_RANGE[0], max_colors + 1):
        erasure_choices: list[int | None] = [None]
        if want_erasure and color_count >= 3:
            erasure_choices = list(range(2, color_count))
        for erasure_index in erasure_choices:
            counts = _gen_color_counts(cell_count, color_count, marker_count, erasure_index)
            if counts is not None:
                options.append((color_count, erasure_index, counts))
    return options


def _gen_template(
    bg: int,
    palette: list[int],
    max_colors: int,
    cell_count: int,
    marker_count: int,
    want_erasure: bool,
) -> dict[str, object] | None:
    color_options = _gen_color_options(cell_count, marker_count, max_colors, want_erasure)
    if not color_options and want_erasure:
        return _gen_template(bg, palette, max_colors, cell_count, marker_count, False)
    if not color_options:
        return None

    color_count, _erasure_index, counts = random.choice(color_options)
    colors = palette[:color_count]
    dims = [
        (height, width)
        for height in range(TEMPLATE_BBOX_RANGE[0], TEMPLATE_BBOX_RANGE[1] + 1)
        for width in range(TEMPLATE_BBOX_RANGE[0], TEMPLATE_BBOX_RANGE[1] + 1)
        if max(height, width) <= cell_count <= height * width
    ]
    if not dims:
        return None
    height, width = random.choice(dims)
    coords = _gen_shape_coords(height, width, cell_count)
    if not coords:
        return None

    marker_offsets = [random.choice(coords)] if marker_count == 1 else _gen_adjacent_pair(coords)
    if len(marker_offsets) != marker_count:
        return None

    remaining = [coord for coord in coords if coord not in marker_offsets]
    random.shuffle(remaining)
    color_by_coord = {coord: colors[0] for coord in coords}
    for coord in marker_offsets:
        color_by_coord[coord] = colors[1]

    cursor = 0
    for color_idx in range(2, color_count):
        for coord in remaining[cursor : cursor + counts[color_idx]]:
            color_by_coord[coord] = colors[color_idx]
        cursor += counts[color_idx]

    patch = np.full((height, width), bg, dtype=int)
    for (row, col), color in color_by_coord.items():
        patch[row, col] = color

    color_sizes = {color: counts[idx] for idx, color in enumerate(colors)}
    erase_offsets = [
        coord
        for coord, color in color_by_coord.items()
        if color != colors[1] and color_sizes[color] == 1
    ]
    cells = [(row, col, int(patch[row, col])) for row, col in coords]
    return {
        "patch": patch,
        "cells": cells,
        "marker_offsets": marker_offsets,
        "marker_color": colors[1],
        "erase_offsets": erase_offsets,
        "color_count": color_count,
    }


def _gen_can_place(canvas: np.ndarray, cells: list[tuple[int, int, int]], row0: int, col0: int, bg: int) -> bool:
    height, width = canvas.shape
    for row, col, _color in cells:
        rr, cc = row0 + row, col0 + col
        if not (0 <= rr < height and 0 <= cc < width):
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < height and 0 <= nc < width and int(canvas[nr, nc]) != bg:
                    return False
    return True


def _gen_place_patch(canvas: np.ndarray, template: dict[str, object], bg: int) -> tuple[int, int] | None:
    patch = template["patch"]
    cells = template["cells"]
    assert isinstance(patch, np.ndarray)
    assert isinstance(cells, list)
    height, width = canvas.shape
    ph, pw = patch.shape
    choices = [
        (row, col)
        for row in range(height - ph + 1)
        for col in range(width - pw + 1)
        if _gen_can_place(canvas, cells, row, col, bg)
    ]
    if not choices:
        return None
    row, col = random.choice(choices)
    paste(canvas, patch, row, col, transparent=bg)
    return row, col


def _gen_templates(bg: int, cell_counts: tuple[int, ...], marker_counts: tuple[int, ...]) -> list[dict[str, object]] | None:
    available = [color for color in range(PALETTE_SIZE) if color != bg]
    random.shuffle(available)

    templates = []
    for idx, (cell_count, marker_count) in enumerate(zip(cell_counts, marker_counts)):
        remaining_templates = len(cell_counts) - idx - 1
        max_colors = min(TEMPLATE_COLOR_RANGE[1], len(available) - 2 * remaining_templates)
        if max_colors < TEMPLATE_COLOR_RANGE[0]:
            return None
        template = _gen_template(
            bg,
            available,
            max_colors,
            cell_count,
            marker_count,
            random.random() < ERASURE_TEMPLATE_PROB,
        )
        if template is None:
            return None
        templates.append(template)
        del available[: int(template["color_count"])]
    return templates


def _gen_render_expected(
    size: int,
    bg: int,
    templates: list[dict[str, object]],
    group_counts: tuple[int, ...],
) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    expected = np.full((size, size), bg, dtype=int)
    placements = [{"source": None, "stamps": []} for _template in templates]
    items = []
    for idx, template in enumerate(templates):
        area = len(template["cells"])
        items.append((-area, random.random(), idx, "source"))
        for _ in range(group_counts[idx]):
            items.append((-area, random.random(), idx, "stamp"))
    items.sort()

    for _area, _tie, idx, kind in items:
        placed = _gen_place_patch(expected, templates[idx], bg)
        if placed is None:
            return None
        if kind == "source":
            placements[idx]["source"] = placed
        else:
            placements[idx]["stamps"].append(placed)
    return expected, placements


def _gen_build_input(
    expected: np.ndarray,
    templates: list[dict[str, object]],
    placements: list[dict[str, object]],
    bg: int,
) -> np.ndarray | None:
    inp = expected.copy()
    for idx, template in enumerate(templates):
        source = placements[idx]["source"]
        if source is None:
            return None
        source_r, source_c = source
        for row, col in template["erase_offsets"]:
            expected[source_r + row, source_c + col] = bg

        for stamp_r, stamp_c in placements[idx]["stamps"]:
            for row, col, _color in template["cells"]:
                inp[stamp_r + row, stamp_c + col] = bg
            for row, col in template["marker_offsets"]:
                inp[stamp_r + row, stamp_c + col] = int(template["marker_color"])
    return inp


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(GRID_SIZE_RANGE[0], GRID_SIZE_RANGE[1])
    skeletons = _gen_skeleton_options(size)
    if not skeletons:
        return None

    group_counts, marker_counts, cell_counts = random.choice(skeletons)
    bg = random.randrange(PALETTE_SIZE)
    templates = _gen_templates(bg, cell_counts, marker_counts)
    if templates is None:
        return None

    rendered = _gen_render_expected(size, bg, templates, group_counts)
    if rendered is None:
        return None
    expected, placements = rendered

    inp = _gen_build_input(expected, templates, placements, bg)
    if inp is None:
        return None

    try:
        got = main(inp)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate d6542281 input")
