from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS) not in sys.path:
    sys.path.append(str(_ARC_UTILS))

from arc_utils import bbox_coords, colors_present, crop_bbox, find_components, get_bg, paste


def _rect_border_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    cells |= {(r, c0) for r in range(r0, r1 + 1)}
    cells |= {(r, c1) for r in range(r0, r1 + 1)}
    return cells


def _extract_panels(grid: np.ndarray) -> tuple[int, list[np.ndarray]]:
    separator = int(get_bg(grid))
    panels = find_components(grid, bg=separator, connectivity=4, monochromatic=False, return_format="dict")
    assert len(panels) >= 3, "expected at least three separated panels"

    boxes = [tuple(map(int, panel["bbox"])) for panel in panels]
    shapes = {(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in boxes}
    assert len(shapes) == 1, "all panels must share the same rectangular size"

    ordered = []
    for box in sorted(boxes):
        ordered.append(crop_bbox(grid, box))
    return separator, ordered


def _rectangle_fragment(patch: np.ndarray, color: int) -> dict | None:
    coords = np.argwhere(patch == int(color))
    if coords.size == 0:
        return None

    box = bbox_coords(coords)
    r0, c0, r1, c1 = box
    if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
        return None

    full_border = _rect_border_cells(box)
    cells = {tuple(map(int, rc)) for rc in coords}
    if not cells <= full_border:
        return None

    if not any(r == r0 for r, _ in cells):
        return None
    if not any(r == r1 for r, _ in cells):
        return None
    if not any(c == c0 for _, c in cells):
        return None
    if not any(c == c1 for _, c in cells):
        return None

    return {"bbox": box, "cells": cells, "full_border": full_border}


def _unique_topological_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int] | None:
    outgoing = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}
    for lower, upper in edges:
        if upper not in outgoing[lower]:
            outgoing[lower].add(upper)
            indegree[upper] += 1

    remaining = set(colors)
    order: list[int] = []
    while remaining:
        available = sorted(color for color in remaining if indegree[color] == 0)
        if len(available) != 1:
            return None
        color = available[0]
        order.append(color)
        remaining.remove(color)
        for nxt in outgoing[color]:
            indegree[nxt] -= 1
    return order


def _all_pairs_comparable(order: list[int], edges: set[tuple[int, int]]) -> bool:
    reach = {color: set() for color in order}
    for lower, upper in edges:
        reach[lower].add(upper)

    changed = True
    while changed:
        changed = False
        for color in order:
            extra = set()
            for nxt in tuple(reach[color]):
                extra |= reach[nxt]
            if not extra <= reach[color]:
                reach[color] |= extra
                changed = True

    for i, a in enumerate(order):
        for b in order[i + 1 :]:
            if b not in reach[a]:
                return False
    return True


def _infer_key_order(patch: np.ndarray) -> dict | None:
    blank = int(get_bg(patch))
    key_colors = colors_present(patch, exclude={blank})
    if not (3 <= len(key_colors) <= 5):
        return None

    fragments: dict[int, dict] = {}
    for color in key_colors:
        fragment = _rectangle_fragment(patch, color)
        if fragment is None:
            return None
        fragments[int(color)] = fragment

    edges: set[tuple[int, int]] = set()
    overlap_pairs = 0
    for i, a in enumerate(key_colors):
        for b in key_colors[i + 1 :]:
            intersection = fragments[a]["full_border"] & fragments[b]["full_border"]
            if not intersection:
                continue
            visible = {int(patch[r, c]) for r, c in intersection}
            if blank in visible:
                return None
            if any(color not in (a, b) for color in visible):
                return None
            if len(visible) != 1:
                return None
            winner = next(iter(visible))
            loser = b if winner == a else a
            edges.add((int(loser), int(winner)))
            overlap_pairs += 1

    if overlap_pairs < len(key_colors) - 1:
        return None

    order = _unique_topological_order(list(map(int, key_colors)), edges)
    if order is None:
        return None
    if not _all_pairs_comparable(order, edges):
        return None

    return {"blank": blank, "order": order, "edges": edges, "fragments": fragments}


def _overlay_with_order(patches: list[np.ndarray], blank: int, order: list[int]) -> np.ndarray:
    ranks = {int(color): idx for idx, color in enumerate(order)}
    out = np.full_like(patches[0], blank)
    top_rank = np.full(patches[0].shape, -1, dtype=int)
    for patch in patches:
        coords = np.argwhere(patch != blank)
        for r_raw, c_raw in coords:
            r = int(r_raw)
            c = int(c_raw)
            color = int(patch[r, c])
            rank = ranks[color]
            if rank > top_rank[r, c]:
                top_rank[r, c] = rank
                out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recover a color occlusion order from one key panel and use it to resolve overlay conflicts in the others.

    Input structure:
    - The most common color is a separator splitting the canvas into several same-sized rectangular panels.
    - Exactly one panel is a key panel: it contains several distinct-color one-pixel-thick hollow rectangles whose borders overlap.
    - Every other panel uses one common blank color plus sparse colored marks placed at panel-relative positions.

    Rule:
    - Infer the unique bottom-to-top color order from the key panel: on every border-overlap cell, the visible color is above the hidden one.
    - Ignore the key panel itself, then transparently copy every non-blank mark from the remaining panels onto one blank output panel of the same size.
    - If several colors want the same output cell, keep the color that is highest in the recovered order.

    Key insight:
    - The overlapping rectangles are not the object to reconstruct; they encode a global layering priority table that decides all later overlay conflicts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _separator, panels = _extract_panels(g)
    candidates = [(idx, _infer_key_order(patch)) for idx, patch in enumerate(panels)]
    candidates = [(idx, info) for idx, info in candidates if info is not None]
    assert len(candidates) == 1, "expected exactly one key panel"

    key_idx, key_info = candidates[0]
    order = key_info["order"]
    order_set = set(order)
    non_key_panels = [patch for idx, patch in enumerate(panels) if idx != key_idx]

    blank_votes = [int(Counter(map(int, patch.ravel())).most_common(1)[0][0]) for patch in non_key_panels]
    assert len(set(blank_votes)) == 1, "non-key panels must share one blank color"
    blank = blank_votes[0]
    assert blank not in order_set, "blank color must not participate in the key order"

    for patch in non_key_panels:
        used = set(map(int, patch[patch != blank]))
        assert used, "each non-key panel must contain at least one mark"
        assert used <= order_set, "all mark colors must appear in the key panel"

    return _overlay_with_order(non_key_panels, blank, order)


ALL_COLORS = list(range(10))
LAYOUTS = [(2, 2), (2, 3)]
PANEL_GAP = 1
MAX_TRIES = 400


def _overlay_in_sequence(patches: list[np.ndarray], blank: int) -> np.ndarray:
    out = np.full_like(patches[0], blank)
    for patch in patches:
        mask = patch != blank
        out[mask] = patch[mask]
    return out


def _overlay_numeric_extreme(
    patches: list[np.ndarray], blank: int, *, use_max: bool
) -> np.ndarray:
    out = np.full_like(patches[0], blank)
    per_cell: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for patch in patches:
        for r_raw, c_raw in np.argwhere(patch != blank):
            r = int(r_raw)
            c = int(c_raw)
            per_cell[(r, c)].append(int(patch[r, c]))
    for (r, c), colors in per_cell.items():
        out[r, c] = max(colors) if use_max else min(colors)
    return out


def _panel_canvas_shape(rows: int, cols: int, ph: int, pw: int, margins: tuple[int, int, int, int]) -> tuple[int, int]:
    top, bottom, left, right = margins
    height = top + bottom + rows * ph + (rows - 1) * PANEL_GAP
    width = left + right + cols * pw + (cols - 1) * PANEL_GAP
    return height, width


def _choose_layout() -> tuple[int, int, int, int, tuple[int, int, int, int]] | None:
    rows, cols = random.choice(LAYOUTS)
    ph = random.randint(7, 9)
    pw = random.randint(7, 9)

    base_h = rows * ph + (rows - 1) * PANEL_GAP
    base_w = cols * pw + (cols - 1) * PANEL_GAP
    extra_h = 30 - base_h
    extra_w = 30 - base_w
    if extra_h < 4 or extra_w < 4:
        return None

    top = random.randint(2, min(6, extra_h - 2))
    bottom = random.randint(2, min(6, extra_h - top))
    left = random.randint(2, min(6, extra_w - 2))
    right = random.randint(2, min(6, extra_w - left))
    shape = _panel_canvas_shape(rows, cols, ph, pw, (top, bottom, left, right))
    if max(shape) > 30:
        return None
    return rows, cols, ph, pw, (top, bottom, left, right)


def _make_blank_patch(ph: int, pw: int, blank: int) -> np.ndarray:
    return np.full((ph, pw), blank, dtype=int)


def _draw_rectangles(
    ph: int, pw: int, blank: int, order: list[int], boxes: list[tuple[int, int, int, int]]
) -> np.ndarray:
    patch = np.full((ph, pw), blank, dtype=int)
    for color, box in zip(order, boxes):
        for r, c in _rect_border_cells(box):
            patch[r, c] = int(color)
    return patch


def _visible_side_counts(patch: np.ndarray, color: int, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    coords = {tuple(map(int, rc)) for rc in np.argwhere(patch == color)}
    return (
        sum((r0, c) in coords for c in range(c0, c1 + 1)),
        sum((r1, c) in coords for c in range(c0, c1 + 1)),
        sum((r, c0) in coords for r in range(r0, r1 + 1)),
        sum((r, c1) in coords for r in range(r0, r1 + 1)),
    )


def _make_key_panel(ph: int, pw: int, blank: int, order: list[int]) -> np.ndarray | None:
    all_boxes = [
        (r0, c0, r1, c1)
        for r0 in range(ph - 3)
        for c0 in range(pw - 3)
        for r1 in range(r0 + 3, ph)
        for c1 in range(c0 + 3, pw)
    ]

    for _ in range(250):
        boxes: list[tuple[int, int, int, int]] = []
        cell_counts: Counter[tuple[int, int]] = Counter()
        ok = True
        for idx in range(len(order)):
            random.shuffle(all_boxes)
            chosen = None
            for box in all_boxes:
                if box in boxes:
                    continue
                border = _rect_border_cells(box)
                if idx > 0 and not (border & _rect_border_cells(boxes[idx - 1])):
                    continue
                if idx > 1 and not any(border & _rect_border_cells(prev) for prev in boxes[:-1]):
                    continue
                if any(cell_counts[cell] >= 2 for cell in border):
                    continue
                chosen = box
                break
            if chosen is None:
                ok = False
                break
            boxes.append(chosen)
            cell_counts.update(_rect_border_cells(chosen))
        if not ok:
            continue

        patch = _draw_rectangles(ph, pw, blank, order, boxes)
        info = _infer_key_order(patch)
        if info is None or info["order"] != order:
            continue

        strong_visibility = True
        for color, box in zip(order, boxes):
            top, bottom, left, right = _visible_side_counts(patch, color, box)
            if min(top, bottom, left, right) < 1:
                strong_visibility = False
                break
            if sum((patch == color).ravel()) < 8:
                strong_visibility = False
                break
        if strong_visibility:
            return patch
    return None


def _place_marks(
    ph: int, pw: int, blank: int, order: list[int], n_panels: int
) -> tuple[list[np.ndarray], list[tuple[int, int]], dict[tuple[int, int], list[int]]] | None:
    patches = [_make_blank_patch(ph, pw, blank) for _ in range(n_panels)]
    used_positions: set[tuple[int, int]] = set()
    conflict_colors: dict[tuple[int, int], list[int]] = {}

    conflict_count = random.randint(2, min(4, ph * pw // 8))
    all_positions = [(r, c) for r in range(ph) for c in range(pw)]
    random.shuffle(all_positions)

    for pos in all_positions:
        if len(conflict_colors) >= conflict_count:
            break
        participants = random.randint(2, min(n_panels, len(order)))
        panel_ids = random.sample(range(n_panels), participants)
        colors = random.sample(order, participants)
        if max(order.index(color) for color in colors) == min(order.index(color) for color in colors):
            continue
        for panel_id, color in zip(panel_ids, colors):
            patches[panel_id][pos] = int(color)
        conflict_colors[pos] = colors
        used_positions.add(pos)

    if len(conflict_colors) < 2:
        return None

    for panel_id, patch in enumerate(patches):
        extra_count = random.randint(2, 5)
        free_positions = [pos for pos in all_positions if pos not in used_positions]
        random.shuffle(free_positions)
        added = 0
        for pos in free_positions:
            if patch[pos] != blank:
                continue
            patch[pos] = int(random.choice(order))
            used_positions.add(pos)
            added += 1
            if added == extra_count:
                break
        if np.count_nonzero(patch != blank) < 3:
            return None

    return patches, list(conflict_colors), conflict_colors


def _assemble_input(
    rows: int,
    cols: int,
    ph: int,
    pw: int,
    margins: tuple[int, int, int, int],
    separator: int,
    panels: list[np.ndarray],
) -> np.ndarray:
    top, bottom, left, right = margins
    H, W = _panel_canvas_shape(rows, cols, ph, pw, margins)
    grid = np.full((H, W), separator, dtype=int)
    for idx, patch in enumerate(panels):
        r = idx // cols
        c = idx % cols
        r0 = top + r * (ph + PANEL_GAP)
        c0 = left + c * (pw + PANEL_GAP)
        paste(grid, patch, r0, c0)
    return grid


def _semantic_checks(
    grid: np.ndarray,
    key_patch: np.ndarray,
    non_key_patches: list[np.ndarray],
    blank: int,
    order: list[int],
    expected: np.ndarray,
) -> bool:
    if not np.array_equal(main(grid), expected):
        return False

    _sep, panels = _extract_panels(grid)
    key_candidates = [idx for idx, patch in enumerate(panels) if _infer_key_order(patch) is not None]
    if len(key_candidates) != 1:
        return False

    if np.array_equal(_overlay_in_sequence(non_key_patches, blank), expected):
        return False
    if np.array_equal(_overlay_in_sequence(list(reversed(non_key_patches)), blank), expected):
        return False
    if np.array_equal(_overlay_numeric_extreme(non_key_patches, blank, use_max=True), expected):
        return False
    if np.array_equal(_overlay_numeric_extreme(non_key_patches, blank, use_max=False), expected):
        return False

    if np.array_equal(_overlay_in_sequence(non_key_patches + [key_patch], blank), expected):
        return False

    conflicts: dict[tuple[int, int], set[int]] = defaultdict(set)
    winners = set()
    ranks = {color: idx for idx, color in enumerate(order)}
    for patch in non_key_patches:
        for r_raw, c_raw in np.argwhere(patch != blank):
            r = int(r_raw)
            c = int(c_raw)
            conflicts[(r, c)].add(int(patch[r, c]))
    conflict_positions = [pos for pos, colors in conflicts.items() if len(colors) >= 2]
    if len(conflict_positions) < 2:
        return False
    for pos in conflict_positions:
        winners.add(max(conflicts[pos], key=lambda color: ranks[color]))
    if len(winners) < 2:
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    layout = _choose_layout()
    if layout is None:
        return None
    rows, cols, ph, pw, margins = layout
    total_panels = rows * cols

    key_color_count = random.choice([3, 3, 4, 4, 5])
    palette = random.sample(ALL_COLORS, key_color_count + 2)
    separator = int(palette[0])
    blank = int(palette[1])
    order = [int(color) for color in palette[2:]]
    random.shuffle(order)

    key_patch = _make_key_panel(ph, pw, blank, order)
    if key_patch is None:
        return None

    non_key = _place_marks(ph, pw, blank, order, total_panels - 1)
    if non_key is None:
        return None
    non_key_patches, _conflict_positions, _conflict_colors = non_key
    expected = _overlay_with_order(non_key_patches, blank, order)

    panels = non_key_patches + [key_patch]
    random.shuffle(panels)
    input_non_key_order = [patch for patch in panels if patch is not key_patch]
    grid = _assemble_input(rows, cols, ph, pw, margins, separator, panels)

    if separator != int(get_bg(grid)):
        return None
    if not _semantic_checks(grid, key_patch, input_non_key_order, blank, order, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_TRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid occlusion-order overlay puzzle")
