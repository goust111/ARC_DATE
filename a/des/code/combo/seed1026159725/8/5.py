from __future__ import annotations

import random
from collections import deque

import numpy as np


RED = 2
UP = "up"
DOWN = "down"
LEFT = "left"
RIGHT = "right"

DIR_TO_DELTA = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}

# The 3-cell L occupies a 2x2 box. The missing corner encodes the direction:
# missing top-left -> up, top-right -> right, bottom-right -> down, bottom-left -> left.
DIR_TO_ARROW = {
    UP: frozenset({(0, 1), (1, 0), (1, 1)}),
    RIGHT: frozenset({(0, 0), (1, 0), (1, 1)}),
    DOWN: frozenset({(0, 0), (0, 1), (1, 0)}),
    LEFT: frozenset({(0, 0), (0, 1), (1, 1)}),
}
ARROW_TO_DIR = {shape: direction for direction, shape in DIR_TO_ARROW.items()}

ORTHO_STEPS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_COLORS = tuple(range(10))
GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _segments(length: int, separators: list[int]) -> list[tuple[int, int]]:
    cuts = [-1] + sorted(int(x) for x in separators) + [int(length)]
    spans: list[tuple[int, int]] = []
    for a, b in zip(cuts, cuts[1:]):
        if b - a > 1:
            spans.append((a + 1, b))
    return spans


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        q = deque([(r, c)])
        while q:
            rr, cc = q.popleft()
            comp.append((rr, cc))
            for dr, dc in ORTHO_STEPS:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _normalize(cells: list[tuple[int, int]] | set[tuple[int, int]] | frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    rows = [int(r) for r, _c in cells]
    cols = [int(c) for _r, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    return frozenset((int(r - r0), int(c - c0)) for r, c in cells)


def _panel_structure(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)

    sep_rows = [
        r
        for r in range(h)
        if np.all(g[r] == int(g[r, 0])) and int(g[r, 0]) != bg
    ]
    sep_cols = [
        c
        for c in range(w)
        if np.all(g[:, c] == int(g[0, c])) and int(g[0, c]) != bg
    ]
    sep_colors = {int(g[r, 0]) for r in sep_rows} | {int(g[0, c]) for c in sep_cols}
    assert sep_colors, "expected at least one separator line"
    assert len(sep_colors) == 1, "separator lines must share one color"
    sep = sep_colors.pop()
    row_spans = _segments(h, sep_rows)
    col_spans = _segments(w, sep_cols)
    assert row_spans and col_spans, "expected at least one panel"
    return sep, row_spans, col_spans


def _read_arrow(panel: np.ndarray) -> str:
    comps = _components4(panel == RED)
    assert len(comps) == 1, "each panel must contain one red arrow"
    comp = comps[0]
    assert len(comp) == 3, "arrow must have three pixels"
    norm = _normalize(comp)
    assert norm in ARROW_TO_DIR, "unexpected arrow shape"
    return ARROW_TO_DIR[norm]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each panel's main object into the adjacent panel indicated by its red L-arrow.

    Input structure
    - The background is the dominant color.
    - Full separator rows/columns split the canvas into rectangular panels.
    - Every panel contains one 3-pixel red L-arrow; some panels also contain one main object.
    - The arrow direction is encoded by which corner is missing from its 2x2 L box.

    Rule
    - For every panel that contains a main object, read its arrow direction.
    - Copy the object into the adjacent panel in that direction, preserving panel-relative coordinates.
    - Copying is a stamp: pre-existing target pixels remain unless overwritten by copied object pixels.
    - Source objects remain in place and all arrows are erased.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    sep, row_spans, col_spans = _panel_structure(g)
    out = g.copy()
    out[g == RED] = bg

    sources: list[tuple[int, int, str, list[tuple[int, int, int]]]] = []
    for pr, (r0, r1) in enumerate(row_spans):
        for pc, (c0, c1) in enumerate(col_spans):
            panel = g[r0:r1, c0:c1]
            direction = _read_arrow(panel)
            obj_cells = [
                (int(r), int(c), int(panel[r, c]))
                for r, c in np.argwhere((panel != bg) & (panel != sep) & (panel != RED))
            ]
            if obj_cells:
                sources.append((pr, pc, direction, obj_cells))

    for pr, pc, direction, obj_cells in sources:
        dr, dc = DIR_TO_DELTA[direction]
        tr = pr + dr
        tc = pc + dc
        assert 0 <= tr < len(row_spans) and 0 <= tc < len(col_spans), "copy target must be an adjacent panel"
        tr0, tr1 = row_spans[tr]
        tc0, tc1 = col_spans[tc]
        target_h = tr1 - tr0
        target_w = tc1 - tc0
        for rr, cc, color in obj_cells:
            assert rr < target_h and cc < target_w, "copied object must fit target panel"
            out[tr0 + rr, tc0 + cc] = int(color)

    return out


def _panel_top_left(pr: int, pc: int, panel_h: int, panel_w: int) -> tuple[int, int]:
    return pr * (panel_h + 1), pc * (panel_w + 1)


def _panel_neighbors(pr: int, pc: int, panel_rows: int, panel_cols: int) -> list[tuple[str, tuple[int, int]]]:
    out: list[tuple[str, tuple[int, int]]] = []
    for direction, (dr, dc) in DIR_TO_DELTA.items():
        nr = pr + dr
        nc = pc + dc
        if 0 <= nr < panel_rows and 0 <= nc < panel_cols:
            out.append((direction, (nr, nc)))
    return out


def _grow_polyomino(height: int, width: int, area: int) -> frozenset[tuple[int, int]]:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier = set()
    for dr, dc in ORTHO_STEPS:
        nr = start[0] + dr
        nc = start[1] + dc
        if 0 <= nr < height and 0 <= nc < width:
            frontier.add((nr, nc))

    while len(cells) < area:
        if not frontier:
            return _grow_polyomino(height, width, area)
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for dr, dc in ORTHO_STEPS:
            nr = cell[0] + dr
            nc = cell[1] + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))
    return _normalize(cells)


def _sample_object_cells(panel_h: int, panel_w: int) -> frozenset[tuple[int, int]]:
    max_box_h = min(4, panel_h - 1)
    max_box_w = min(4, panel_w - 1)
    box_h = random.randint(2, max_box_h)
    box_w = random.randint(2, max_box_w)
    max_area = min(6, box_h * box_w - 1)
    area = random.randint(2, max_area)
    shape = _grow_polyomino(box_h, box_w, area)
    r0 = random.randint(0, panel_h - box_h)
    c0 = random.randint(0, panel_w - box_w)
    return frozenset((int(r + r0), int(c + c0)) for r, c in shape)


def _all_arrow_placements(panel_h: int, panel_w: int, obj_cells: frozenset[tuple[int, int]]) -> list[tuple[int, int]]:
    placements: list[tuple[int, int]] = []
    for r0 in range(panel_h - 1):
        for c0 in range(panel_w - 1):
            box = {(r0 + r, c0 + c) for r in range(2) for c in range(2)}
            if box & obj_cells:
                continue
            placements.append((r0, c0))
    random.shuffle(placements)
    return placements


def _paint_arrow(panel: np.ndarray, direction: str, top: int, left: int) -> None:
    for dr, dc in DIR_TO_ARROW[direction]:
        panel[top + dr, left + dc] = RED


def _choose_assignment(
    occupied: set[tuple[int, int]],
    panel_rows: int,
    panel_cols: int,
    mode: str,
) -> dict[tuple[int, int], tuple[int, int]] | None:
    neighbors = {
        src: [dst for _direction, dst in _panel_neighbors(src[0], src[1], panel_rows, panel_cols)]
        for src in occupied
    }
    ordered = sorted(occupied, key=lambda cell: len(neighbors[cell]))
    need_empty = mode in {"empty", "mixed"}
    need_occ = mode in {"overwrite", "mixed"}

    def rec(
        idx: int,
        used: set[tuple[int, int]],
        mapping: dict[tuple[int, int], tuple[int, int]],
        has_empty: bool,
        has_occ: bool,
    ) -> dict[tuple[int, int], tuple[int, int]] | None:
        if idx == len(ordered):
            if (not need_empty or has_empty) and (not need_occ or has_occ):
                return dict(mapping)
            return None

        src = ordered[idx]
        options = list(neighbors[src])
        random.shuffle(options)
        for dst in options:
            if dst in used:
                continue
            mapping[src] = dst
            out = rec(
                idx + 1,
                used | {dst},
                mapping,
                has_empty or dst not in occupied,
                has_occ or dst in occupied,
            )
            if out is not None:
                return out
            del mapping[src]
        return None

    return rec(0, set(), {}, False, False)


def _direction_between(src: tuple[int, int], dst: tuple[int, int]) -> str:
    dr = dst[0] - src[0]
    dc = dst[1] - src[1]
    for direction, delta in DIR_TO_DELTA.items():
        if delta == (dr, dc):
            return direction
    raise AssertionError("panels must be orthogonally adjacent")


def _sample_layout(mode: str) -> tuple[int, int, int, int, set[tuple[int, int]], dict[tuple[int, int], tuple[int, int]]] | None:
    panel_rows = random.randint(2, 3)
    panel_cols = random.randint(2, 3)
    panel_h = random.randint(5, 7)
    panel_w = random.randint(5, 7)
    panels = [(r, c) for r in range(panel_rows) for c in range(panel_cols)]
    total = len(panels)

    if mode == "empty":
        occ_min = 1
        occ_max = total - 1
    elif mode == "overwrite":
        occ_min = 2
        occ_max = total
    else:
        occ_min = 2
        occ_max = total - 1
    if occ_min > occ_max:
        return None

    for _ in range(80):
        occ_count = random.randint(occ_min, occ_max)
        occupied = set(random.sample(panels, occ_count))
        mapping = _choose_assignment(occupied, panel_rows, panel_cols, mode)
        if mapping is not None:
            return panel_rows, panel_cols, panel_h, panel_w, occupied, mapping
    return None


def _color_pool(bg: int, sep: int) -> list[int]:
    return [color for color in ALL_COLORS if color not in {bg, sep, RED}]


def _build_attempt(mode: str) -> tuple[np.ndarray, np.ndarray] | None:
    layout = _sample_layout(mode)
    if layout is None:
        return None
    panel_rows, panel_cols, panel_h, panel_w, occupied, mapping = layout

    special_free = [color for color in ALL_COLORS if color != RED]
    bg = random.choice(special_free)
    sep = random.choice([color for color in special_free if color != bg])
    palette = _color_pool(bg, sep)
    h = panel_rows * panel_h + panel_rows - 1
    w = panel_cols * panel_w + panel_cols - 1

    grid = np.full((h, w), bg, dtype=int)
    for r in range(1, panel_rows):
        grid[r * panel_h + (r - 1), :] = sep
    for c in range(1, panel_cols):
        grid[:, c * panel_w + (c - 1)] = sep

    panel_specs: dict[tuple[int, int], dict[str, object]] = {}
    sources_by_target: dict[tuple[int, int], tuple[int, int]] = {dst: src for src, dst in mapping.items()}
    overlap_targets = [dst for dst in mapping.values() if dst in occupied]
    if mode == "overwrite":
        assert overlap_targets
    if mode == "mixed":
        assert overlap_targets and any(dst not in occupied for dst in mapping.values())
    designated_overlap = random.choice(overlap_targets) if overlap_targets else None

    src_color_by_target: dict[tuple[int, int], int] = {}
    src_cells_by_target: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
    for panel in occupied:
        color = random.choice(palette)
        cells = _sample_object_cells(panel_h, panel_w)
        panel_specs[panel] = {"color": color, "cells": cells}

    # Force at least one visible overwrite case.
    if designated_overlap is not None:
        src = sources_by_target[designated_overlap]
        src_color = int(panel_specs[src]["color"])
        src_cells = frozenset(panel_specs[src]["cells"])
        new_color_choices = [color for color in palette if color != src_color]
        panel_specs[designated_overlap] = {
            "color": random.choice(new_color_choices),
            "cells": src_cells,
        }

    for panel in occupied:
        incoming = sources_by_target.get(panel)
        if incoming is not None and panel != designated_overlap and random.random() < 0.45:
            incoming_cells = frozenset(panel_specs[incoming]["cells"])
            incoming_color = int(panel_specs[incoming]["color"])
            color_choices = [color for color in palette if color != incoming_color]
            if color_choices:
                panel_specs[panel] = {
                    "color": random.choice(color_choices),
                    "cells": incoming_cells,
                }

    # Pick arrow directions for every panel and placements that do not overlap panel objects.
    arrows: dict[tuple[int, int], tuple[str, int, int]] = {}
    for pr in range(panel_rows):
        for pc in range(panel_cols):
            panel = (pr, pc)
            if panel in mapping:
                direction = _direction_between(panel, mapping[panel])
            else:
                direction = random.choice([item[0] for item in _panel_neighbors(pr, pc, panel_rows, panel_cols)])
            obj_cells = frozenset(panel_specs.get(panel, {}).get("cells", frozenset()))
            placements = _all_arrow_placements(panel_h, panel_w, obj_cells)
            if not placements:
                return None
            arrow_r, arrow_c = placements[0]
            arrows[panel] = (direction, arrow_r, arrow_c)

    for pr in range(panel_rows):
        for pc in range(panel_cols):
            panel = (pr, pc)
            r0, c0 = _panel_top_left(pr, pc, panel_h, panel_w)
            patch = grid[r0 : r0 + panel_h, c0 : c0 + panel_w]
            direction, arrow_r, arrow_c = arrows[panel]
            _paint_arrow(patch, direction, arrow_r, arrow_c)
            if panel in panel_specs:
                color = int(panel_specs[panel]["color"])
                for rr, cc in panel_specs[panel]["cells"]:
                    patch[int(rr), int(cc)] = color

    expected = grid.copy()
    expected[expected == RED] = bg
    for src, dst in mapping.items():
        src_color = int(panel_specs[src]["color"])
        src_cells = frozenset(panel_specs[src]["cells"])
        tr0, tc0 = _panel_top_left(dst[0], dst[1], panel_h, panel_w)
        for rr, cc in src_cells:
            expected[tr0 + int(rr), tc0 + int(cc)] = src_color

    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    modes = ["empty", "overwrite", "mixed"]
    for _ in range(GEN_ATTEMPTS):
        mode = random.choice(modes)
        built = _build_attempt(mode)
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid panel-copy ARC instance")
