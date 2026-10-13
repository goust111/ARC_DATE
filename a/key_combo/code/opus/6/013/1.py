from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np


BG = 0
BASE_PALETTE = (1, 3, 4, 6)
MARKER_PALETTE = (2, 5, 7, 8, 9)
RECT_SIDE_OPTIONS = (5, 6)
RECT_COUNT_OPTIONS = (3, 4)
DIAGONAL_STEPS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
MARKER_THRESHOLD = 5
MAX_GENERATION_ATTEMPTS = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_shape(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return r1 - r0 + 1, c1 - c0 + 1


def _contains(box: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _iter_box(box: tuple[int, int, int, int]):
    r0, c0, r1, c1 = box
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            yield r, c


def _most_common_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    counts = Counter(map(int, grid.ravel()))
    bg = _most_common_bg(grid)
    base_colors = [color for color, count in counts.items() if color != bg and count >= MARKER_THRESHOLD]
    assert 3 <= len(base_colors) <= 4, "expected 3-4 rectangle base colors"

    rects = []
    for color in sorted(base_colors):
        comps = _components(grid == color)
        assert comps, "base color must be visible"
        largest = max(comps, key=len)
        box = _bbox(largest)
        rects.append(
            {
                "base": int(color),
                "bbox": box,
                "visible_count": int(counts[color]),
            }
        )

    shapes = {_bbox_shape(rect["bbox"]) for rect in rects}
    assert len(shapes) == 1, "all rectangles must share the same size"
    h, w = next(iter(shapes))
    assert h == w and h >= 5
    return rects


def _infer_order(grid: np.ndarray, rects: list[dict]) -> tuple[list[dict], dict[int, int]]:
    n = len(rects)
    edges: set[tuple[int, int]] = set()
    for i, j in combinations(range(n), 2):
        rect_i = rects[i]
        rect_j = rects[j]
        inter = _intersection(rect_i["bbox"], rect_j["bbox"])
        assert inter is not None, "every rectangle pair must overlap"
        r0, c0, r1, c1 = inter
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        sees_i = bool(np.any(patch == rect_i["base"]))
        sees_j = bool(np.any(patch == rect_j["base"]))
        assert sees_i != sees_j, "overlap must expose exactly one rectangle base as an ordering witness"
        if sees_i:
            edges.add((j, i))
        else:
            edges.add((i, j))

    indegree = {i: 0 for i in range(n)}
    for lower, upper in edges:
        indegree[upper] += 1

    ranks = sorted(indegree.values())
    assert ranks == list(range(n)), "overlap witnesses must define a unique total order"

    ordered = [None] * n
    rank_by_base: dict[int, int] = {}
    for i, rect in enumerate(rects):
        rank0 = indegree[i]
        ordered[rank0] = rect
        rank_by_base[rect["base"]] = rank0 + 1

    return ordered, rank_by_base


def _control_pixel(grid: np.ndarray, rects: list[dict]) -> tuple[int, int, int]:
    inside = np.zeros_like(grid, dtype=bool)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        inside[r0 : r1 + 1, c0 : c1 + 1] = True
    coords = np.argwhere((grid != BG) & ~inside)
    assert coords.shape == (1, 2), "expected exactly one external control pixel"
    r, c = map(int, coords[0])
    return int(grid[r, c]), r, c


def _merge_markers(grid: np.ndarray, rects: list[dict], rank_by_base: dict[int, int]) -> np.ndarray:
    side, _ = _bbox_shape(rects[0]["bbox"])
    counts = Counter(map(int, grid.ravel()))
    base_colors = [rect["base"] for rect in rects]
    base_fill = max(base_colors, key=lambda color: counts[color])
    assert sum(1 for color in base_colors if counts[color] == counts[base_fill]) == 1

    out = np.full((side, side), int(base_fill), dtype=int)
    base_set = set(base_colors)
    for r, c in np.argwhere(grid != BG):
        color = int(grid[r, c])
        if color in base_set:
            continue
        owners = [rect for rect in rects if _contains(rect["bbox"], int(r), int(c))]
        assert owners, "marker pixels must lie inside at least one rectangle"
        owner = max(owners, key=lambda rect: rank_by_base[rect["base"]])
        r0, c0, _r1, _c1 = owner["bbox"]
        lr = int(r) - r0
        lc = int(c) - c0
        assert out[lr, lc] in (base_fill, color), "merged marker projections must agree"
        out[lr, lc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer rectangle depth from visible overlaps, merge their marker pixels, and cyclically shift the merged canvas by the control rectangle's depth rank.

    Input structure:
    - A black background contains 3-4 same-sized filled rectangles with distinct base colors.
    - Rectangles partially overlap, and each rectangle also carries 1-3 non-base marker pixels inside its local coordinates.
    - Exactly one isolated control pixel sits outside all rectangles, and its color matches one rectangle base color.

    Rule:
    - Recover each rectangle bbox from its visible base-color region.
    - For every overlapping pair, the base color visible inside their shared area belongs to the upper rectangle; these pairwise witnesses determine a unique depth order from bottom rank 1 to top rank N.
    - Build one output canvas with the common rectangle size, filled by the uniquely most frequent base color in the input.
    - Copy every visible marker pixel to that canvas by its owning rectangle's local coordinates, then cyclically shift the whole canvas downward by the depth rank of the rectangle selected by the control pixel.

    Key insight:
    - The control pixel does not directly encode a shift amount: it only names a rectangle, so the required row-roll is available only after solving the overlap-ordering problem.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    rects = _extract_rectangles(g)
    ordered, rank_by_base = _infer_order(g, rects)
    control_color, _r, _c = _control_pixel(g, rects)
    assert control_color in rank_by_base, "control pixel must match a rectangle base color"
    merged = _merge_markers(g, ordered, rank_by_base)
    return np.roll(merged, shift=rank_by_base[control_color], axis=0)


def _paint_design(
    height: int,
    width: int,
    rects: list[dict],
    local_markers: dict[int, list[tuple[int, int]]],
    marker_colors: dict[int, int],
    control_pos: tuple[int, int],
    control_color: int,
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for idx, rect in enumerate(rects):
        r0, c0, r1, c1 = rect["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = rect["base"]
        for lr, lc in local_markers[idx]:
            grid[r0 + lr, c0 + lc] = marker_colors[idx]
    grid[control_pos] = control_color
    return grid


def _absolute_rect_mask(box: tuple[int, int, int, int], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = box
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _visible_masks(rects: list[dict], shape: tuple[int, int]) -> list[np.ndarray]:
    masks = []
    for i, rect in enumerate(rects):
        mask = _absolute_rect_mask(rect["bbox"], shape)
        for j in range(i + 1, len(rects)):
            mask &= ~_absolute_rect_mask(rects[j]["bbox"], shape)
        masks.append(mask)
    return masks


def _local_visible_interior(rect: dict, visible_mask: np.ndarray) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect["bbox"]
    side = r1 - r0 + 1
    cells = []
    for lr in range(1, side - 1):
        for lc in range(1, side - 1):
            if visible_mask[r0 + lr, c0 + lc]:
                cells.append((lr, lc))
    return cells


def _connected_and_full_bbox(mask: np.ndarray) -> bool:
    comps = _components(mask)
    if len(comps) != 1:
        return False
    box = _bbox(comps[0])
    return box == (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)


def _pick_marker_layout(rects: list[dict], visible_masks: list[np.ndarray], side: int) -> dict[int, list[tuple[int, int]]] | None:
    local_choices = {i: _local_visible_interior(rects[i], visible_masks[i]) for i in range(len(rects))}
    if any(not cells for cells in local_choices.values()):
        return None

    used: set[tuple[int, int]] = set()
    layout: dict[int, list[tuple[int, int]]] = {}
    order = list(range(len(rects)))
    random.shuffle(order)
    for i in order:
        allowed = [cell for cell in local_choices[i] if cell not in used]
        if not allowed:
            return None
        pick = random.choice(allowed)
        layout[i] = [pick]
        used.add(pick)

    extra_budget = random.randint(0, len(rects) + 1)
    extra_order = order[:]
    random.shuffle(extra_order)
    for i in extra_order:
        while extra_budget > 0 and len(layout[i]) < 3:
            allowed = [cell for cell in local_choices[i] if cell not in used]
            if not allowed:
                break
            pick = random.choice(allowed)
            layout[i].append(pick)
            used.add(pick)
            extra_budget -= 1

    for i, rect in enumerate(rects):
        base_mask = np.ones((side, side), dtype=bool)
        for lr, lc in layout[i]:
            base_mask[lr, lc] = False
        visible_local = visible_masks[i][rect["bbox"][0] : rect["bbox"][2] + 1, rect["bbox"][1] : rect["bbox"][3] + 1]
        base_visible = visible_local & base_mask
        if not _connected_and_full_bbox(base_visible):
            return None

    return layout


def _pairwise_witness_ok(grid: np.ndarray, rects: list[dict]) -> bool:
    for i, j in combinations(range(len(rects)), 2):
        inter = _intersection(rects[i]["bbox"], rects[j]["bbox"])
        if inter is None:
            return False
        r0, c0, r1, c1 = inter
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        sees_i = bool(np.any(patch == rects[i]["base"]))
        sees_j = bool(np.any(patch == rects[j]["base"]))
        if sees_i == sees_j:
            return False
    return True


def _choose_control_cell(grid: np.ndarray, rects: list[dict]) -> tuple[int, int] | None:
    h, w = grid.shape
    blocked = np.zeros_like(grid, dtype=bool)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    choices = []
    for r in range(h):
        for c in range(w):
            if blocked[r, c] or grid[r, c] != BG:
                continue
            if all(
                not (0 <= nr < h and 0 <= nc < w) or grid[nr, nc] == BG
                for nr, nc in _neighbors8(r, c)
            ):
                choices.append((r, c))
    if not choices:
        return None
    return random.choice(choices)


def _merged_from_design(rects: list[dict], local_markers: dict[int, list[tuple[int, int]]], marker_colors: dict[int, int], fill_color: int) -> np.ndarray:
    side, _ = _bbox_shape(rects[0]["bbox"])
    out = np.full((side, side), fill_color, dtype=int)
    for i in range(len(rects)):
        for lr, lc in local_markers[i]:
            out[lr, lc] = marker_colors[i]
    return out


def _nontrivial_rolls(canvas: np.ndarray) -> bool:
    for shift in range(1, canvas.shape[0]):
        if np.array_equal(np.roll(canvas, shift=shift, axis=0), canvas):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    count = random.choice(RECT_COUNT_OPTIONS)
    side = random.choice(RECT_SIDE_OPTIONS)
    dr, dc = random.choice(DIAGONAL_STEPS)

    min_off_r = min(i * dr for i in range(count))
    max_off_r = max(i * dr for i in range(count))
    min_off_c = min(i * dc for i in range(count))
    max_off_c = max(i * dc for i in range(count))
    cluster_h = side + (max_off_r - min_off_r)
    cluster_w = side + (max_off_c - min_off_c)

    height = random.randint(cluster_h + 4, cluster_h + 8)
    width = random.randint(cluster_w + 4, cluster_w + 8)
    if max(height, width) > 30:
        return None

    top = random.randint(1 - min_off_r, height - side - max_off_r - 1)
    left = random.randint(1 - min_off_c, width - side - max_off_c - 1)

    base_colors = random.sample(BASE_PALETTE, count)
    marker_palette = random.sample(MARKER_PALETTE, count)
    rects = []
    marker_colors: dict[int, int] = {}
    for i in range(count):
        r0 = top + i * dr
        c0 = left + i * dc
        rects.append({"base": base_colors[i], "bbox": (r0, c0, r0 + side - 1, c0 + side - 1)})
        marker_colors[i] = marker_palette[i]

    visible_masks = _visible_masks(rects, (height, width))
    local_markers = _pick_marker_layout(rects, visible_masks, side)
    if local_markers is None:
        return None

    control_idx = random.randrange(count)
    temp = _paint_design(height, width, rects, local_markers, marker_colors, (0, 0), base_colors[0])
    temp[0, 0] = BG
    control_pos = _choose_control_cell(temp, rects)
    if control_pos is None:
        return None
    grid = _paint_design(height, width, rects, local_markers, marker_colors, control_pos, rects[control_idx]["base"])

    inferred_rects = _extract_rectangles(grid)
    inferred_order, rank_by_base = _infer_order(grid, inferred_rects)
    if [rect["base"] for rect in inferred_order] != [rect["base"] for rect in rects]:
        return None
    if not _pairwise_witness_ok(grid, rects):
        return None

    control_color, _, _ = _control_pixel(grid, inferred_rects)
    if control_color != rects[control_idx]["base"]:
        return None

    counts = Counter(map(int, grid.ravel()))
    top_base = rects[-1]["base"]
    if sum(1 for rect in rects if counts[rect["base"]] == counts[top_base]) != 1:
        return None
    if counts[top_base] != max(counts[rect["base"]] for rect in rects):
        return None

    for i, rect in enumerate(rects):
        r0, c0, _r1, _c1 = rect["bbox"]
        for lr, lc in local_markers[i]:
            if int(grid[r0 + lr, c0 + lc]) != marker_colors[i]:
                return None

    merged = _merged_from_design(rects, local_markers, marker_colors, top_base)
    if not _nontrivial_rolls(merged):
        return None

    expected = np.roll(merged, shift=control_idx + 1, axis=0)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
