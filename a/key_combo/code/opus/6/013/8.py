from __future__ import annotations

from itertools import combinations
from pathlib import Path
import random
import sys

import numpy as np


ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import bbox_coords, get_bg, paste


BG = 0
BASE_COLORS = (1, 3, 4, 6)
MARKER_COLORS = (2, 7, 8, 9)
RECT_COUNT_OPTIONS = (3, 4)
RECT_H_OPTIONS = (5, 6, 7)
RECT_W_OPTIONS = (5, 6, 7)
GRID_H_RANGE = (13, 18)
GRID_W_RANGE = (13, 18)
MARKER_COUNT_RANGE = (2, 4)
GEN_ATTEMPTS = 800
LAYOUT_LIBRARY = {
    3: [
        {
            "shape": (6, 7),
            "offsets": [(-3, -2), (-2, 2), (-1, -3)],
            "order": (2, 0, 1),
        },
    ],
    4: [
        {
            "shape": (7, 6),
            "offsets": [(-1, 2), (-3, -2), (3, 1), (1, 3)],
            "order": (1, 2, 0, 3),
        },
        {
            "shape": (7, 7),
            "offsets": [(-2, -3), (3, 1), (2, -2), (-1, 3)],
            "order": (0, 3, 2, 1),
        },
    ],
}


def _all_pairings(colors: tuple[int, ...]) -> list[tuple[tuple[int, int], ...]]:
    if not colors:
        return [tuple()]
    first = colors[0]
    out = []
    for i in range(1, len(colors)):
        pair = tuple(sorted((first, colors[i])))
        rest = colors[1:i] + colors[i + 1 :]
        for suffix in _all_pairings(rest):
            out.append((pair, *suffix))
    return out


def _rect_dims(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _swapped_patch(rect: dict, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    patch = np.full((h, w), int(rect["marker"]), dtype=int)
    for rr, cc in rect["marker_local"]:
        patch[int(rr), int(cc)] = int(rect["base"])
    return patch


def _pairing_to_rects(grid: np.ndarray, pairing: tuple[tuple[int, int], ...]) -> tuple[list[dict], tuple[int, int]] | None:
    rects: list[dict] = []
    dims = set()
    local_markers: list[tuple[int, int]] = []

    for color_a, color_b in pairing:
        mask = (grid == int(color_a)) | (grid == int(color_b))
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        bbox = bbox_coords(coords)
        h, w = _rect_dims(bbox)
        dims.add((h, w))

        count_a = int(np.count_nonzero(grid == int(color_a)))
        count_b = int(np.count_nonzero(grid == int(color_b)))
        if count_a == count_b:
            return None
        base, marker = (int(color_a), int(color_b)) if count_a > count_b else (int(color_b), int(color_a))

        marker_coords = np.argwhere(grid == int(marker))
        if not (MARKER_COUNT_RANGE[0] <= len(marker_coords) <= MARKER_COUNT_RANGE[1]):
            return None

        r0, c0, _r1, _c1 = bbox
        marker_local = tuple(sorted((int(r - r0), int(c - c0)) for r, c in marker_coords))
        if any(rr in {0, h - 1} or cc in {0, w - 1} for rr, cc in marker_local):
            return None
        local_markers.extend(marker_local)

        rects.append(
            {
                "bbox": bbox,
                "base": base,
                "marker": marker,
                "marker_local": marker_local,
                "colors": frozenset({int(color_a), int(color_b)}),
            }
        )

    if len(dims) != 1:
        return None
    if len({rect["bbox"] for rect in rects}) != len(rects):
        return None
    if len(local_markers) != len(set(local_markers)):
        return None

    return rects, dims.pop()


def _infer_order(grid: np.ndarray, rects: list[dict]) -> list[int] | None:
    n = len(rects)
    edges: set[tuple[int, int]] = set()

    for i, j in combinations(range(n), 2):
        a = rects[i]
        b = rects[j]
        ar0, ac0, ar1, ac1 = a["bbox"]
        br0, bc0, br1, bc1 = b["bbox"]
        r0 = max(ar0, br0)
        c0 = max(ac0, bc0)
        r1 = min(ar1, br1)
        c1 = min(ac1, bc1)
        if r0 > r1 or c0 > c1:
            return None

        seen = set()
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                value = int(grid[r, c])
                if value in a["colors"]:
                    seen.add(i)
                elif value in b["colors"]:
                    seen.add(j)

        if len(seen) != 1:
            return None

        top = seen.pop()
        bottom = j if top == i else i
        edges.add((bottom, top))

    indegree = {i: 0 for i in range(n)}
    for _src, dst in edges:
        indegree[dst] += 1
    if sorted(indegree.values()) != list(range(n)):
        return None
    return [idx for idx, _rank in sorted(indegree.items(), key=lambda item: item[1])]


def _interpretations(grid: np.ndarray) -> list[tuple[list[dict], list[int], tuple[int, int]]]:
    bg = int(get_bg(grid))
    colors = tuple(sorted(int(v) for v in np.unique(grid) if int(v) != bg))
    if len(colors) < 6 or len(colors) % 2 != 0:
        return []

    valid = []
    for pairing in _all_pairings(colors):
        made = _pairing_to_rects(grid, pairing)
        if made is None:
            continue
        rects, shape = made
        order = _infer_order(grid, rects)
        if order is None:
            continue
        valid.append((rects, order, shape))
    return valid


def _canonical_solution(rects: list[dict], order: list[int]) -> tuple[tuple[int, int, tuple[int, int, int, int]], ...]:
    ranked = []
    for rank, idx in enumerate(order):
        rect = rects[idx]
        ranked.append((rank, int(rect["base"]), int(rect["marker"]), tuple(int(v) for v in rect["bbox"])))
    return tuple(ranked)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer overlapping filled rectangles, swap each rectangle's base and marker colors, then composite the swapped layers in the recovered depth order.

    Input structure:
    - The background is black and the foreground contains 3-4 same-sized filled rectangles at different positions.
    - Every rectangle has its own base color plus one unique marker color used by 2-4 interior marker pixels.
    - Rectangles partially overlap, so an overlap cell shows the color of whichever rectangle is above there.

    Rule:
    - Recover each rectangle by pairing its two unique colors and taking the full rectangle bbox spanned by its visible pixels.
    - For every overlapping rectangle pair, inspect their shared area: if the visible color belongs to rectangle A, then A is above the other rectangle.
    - Use those pairwise visibility relations to obtain the unique bottom-to-top depth order.
    - Swap colors inside every rectangle: base cells become the rectangle's marker color, and marker cells become its base color.
    - Create one output canvas with the shared rectangle size and draw the swapped rectangles from bottom to top at the same local coordinates.

    Key insight:
    - Occlusion must be solved in the original color space, but the final compositing happens only after each recovered rectangle has been color-swapped in its own local coordinate system.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    interpretations = _interpretations(g)
    canonical = {_canonical_solution(rects, order): (rects, order, shape) for rects, order, shape in interpretations}
    assert len(canonical) == 1, "input must admit a unique rectangle pairing and depth order"
    rects, order, shape = next(iter(canonical.values()))

    bg = int(get_bg(g))
    out = np.full(shape, bg, dtype=int)
    for idx in order:
        paste(out, _swapped_patch(rects[idx], shape), 0, 0)
    return out


def _make_patch(base: int, marker: int, marker_local: tuple[tuple[int, int], ...], shape: tuple[int, int]) -> np.ndarray:
    patch = np.full(shape, int(base), dtype=int)
    for rr, cc in marker_local:
        patch[int(rr), int(cc)] = int(marker)
    return patch


def _transform_layout(
    shape: tuple[int, int],
    offsets: list[tuple[int, int]],
    *,
    transpose: bool,
    flip_r: bool,
    flip_c: bool,
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    h, w = shape
    cur = list(offsets)
    if transpose:
        h, w = w, h
        cur = [(c, r) for r, c in cur]
    if flip_r:
        cur = [(-r, c) for r, c in cur]
    if flip_c:
        cur = [(r, -c) for r, c in cur]
    return (h, w), cur


def _sample_layout(rect_count: int) -> tuple[tuple[int, int], list[tuple[int, int]], list[int], tuple[int, int]] | None:
    base = random.choice(LAYOUT_LIBRARY[rect_count])
    transpose = random.choice([False, True])
    flip_r = random.choice([False, True])
    flip_c = random.choice([False, True])
    rect_shape, offsets = _transform_layout(
        base["shape"],
        base["offsets"],
        transpose=transpose,
        flip_r=flip_r,
        flip_c=flip_c,
    )

    rh, rw = rect_shape
    min_dr = min(dr for dr, _dc in offsets)
    max_dr = max(dr for dr, _dc in offsets)
    min_dc = min(dc for _dr, dc in offsets)
    max_dc = max(dc for _dr, dc in offsets)

    min_gh = max(GRID_H_RANGE[0], rh + (max_dr - min_dr) + 3)
    min_gw = max(GRID_W_RANGE[0], rw + (max_dc - min_dc) + 3)
    if min_gh > GRID_H_RANGE[1] or min_gw > GRID_W_RANGE[1]:
        return None

    gh = random.randint(min_gh, GRID_H_RANGE[1])
    gw = random.randint(min_gw, GRID_W_RANGE[1])

    max_r = gh - rh
    max_c = gw - rw
    lo_r = 1 - min_dr
    hi_r = max_r - 1 - max_dr
    lo_c = 1 - min_dc
    hi_c = max_c - 1 - max_dc
    if lo_r > hi_r or lo_c > hi_c:
        return None

    anchor_r = random.randint(lo_r, hi_r)
    anchor_c = random.randint(lo_c, hi_c)
    positions = [(anchor_r + dr, anchor_c + dc) for dr, dc in offsets]
    return rect_shape, positions, list(base["order"]), (gh, gw)


def _render_input(rects: list[dict], order: list[int], grid_shape: tuple[int, int], rect_shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(grid_shape, BG, dtype=int)
    for idx in order:
        rect = rects[idx]
        patch = _make_patch(rect["base"], rect["marker"], rect["marker_local"], rect_shape)
        r0, c0, _r1, _c1 = rect["bbox"]
        paste(grid, patch, r0, c0)
    return grid


def _expected_output(rects: list[dict], order: list[int], shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for idx in order:
        paste(out, _swapped_patch(rects[idx], shape), 0, 0)
    return out


def _visible_local_info(rects: list[dict], order: list[int], rect_shape: tuple[int, int]) -> tuple[list[list[tuple[int, int]]], list[int]]:
    h, w = rect_shape
    rank = {idx: depth for depth, idx in enumerate(order)}
    visible_locals: list[list[tuple[int, int]]] = []
    visible_area_counts: list[int] = []

    for idx, rect in enumerate(rects):
        mask = np.ones((h, w), dtype=bool)
        r0, c0, r1, c1 = rect["bbox"]
        for other_idx in order[rank[idx] + 1 :]:
            or0, oc0, or1, oc1 = rects[other_idx]["bbox"]
            ir0 = max(r0, or0)
            ic0 = max(c0, oc0)
            ir1 = min(r1, or1)
            ic1 = min(c1, oc1)
            if ir0 <= ir1 and ic0 <= ic1:
                mask[ir0 - r0 : ir1 - r0 + 1, ic0 - c0 : ic1 - c0 + 1] = False

        visible_area_counts.append(int(mask.sum()))
        visible_locals.append(
            [
                (rr, cc)
                for rr in range(1, h - 1)
                for cc in range(1, w - 1)
                if mask[rr, cc]
            ]
        )

    return visible_locals, visible_area_counts


def _assign_visible_markers(
    visible_locals: list[list[tuple[int, int]]],
    visible_area_counts: list[int],
) -> list[tuple[tuple[int, int], ...]] | None:
    rect_order = sorted(range(len(visible_locals)), key=lambda idx: len(visible_locals[idx]))
    for _ in range(200):
        used: set[tuple[int, int]] = set()
        assignments: list[tuple[tuple[int, int], ...] | None] = [None] * len(visible_locals)
        ok = True
        for idx in rect_order:
            candidates = [pos for pos in visible_locals[idx] if pos not in used]
            max_k = min(MARKER_COUNT_RANGE[1], len(candidates), (visible_area_counts[idx] - 1) // 2)
            if max_k < MARKER_COUNT_RANGE[0]:
                ok = False
                break
            k = random.randint(MARKER_COUNT_RANGE[0], max_k)
            chosen = tuple(sorted(random.sample(candidates, k)))
            assignments[idx] = chosen
            used.update(chosen)
        if ok and all(item is not None for item in assignments):
            return [item for item in assignments if item is not None]
    return None


def _visible_bbox_matches(grid: np.ndarray, rect: dict) -> bool:
    mask = (grid == int(rect["base"])) | (grid == int(rect["marker"]))
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    if bbox_coords(coords) != rect["bbox"]:
        return False

    r0, c0, _r1, _c1 = rect["bbox"]
    marker_positions = {
        (int(r - r0), int(c - c0))
        for r, c in np.argwhere(grid == int(rect["marker"]))
    }
    if marker_positions != set(rect["marker_local"]):
        return False
    if int(np.count_nonzero(grid == int(rect["base"]))) <= int(np.count_nonzero(grid == int(rect["marker"]))):
        return False
    return True


def _pairwise_visibility_matches(grid: np.ndarray, rects: list[dict], order: list[int]) -> bool:
    rank = {idx: depth for depth, idx in enumerate(order)}
    for i, j in combinations(range(len(rects)), 2):
        a = rects[i]
        b = rects[j]
        ar0, ac0, ar1, ac1 = a["bbox"]
        br0, bc0, br1, bc1 = b["bbox"]
        r0 = max(ar0, br0)
        c0 = max(ac0, bc0)
        r1 = min(ar1, br1)
        c1 = min(ac1, bc1)
        if r0 > r1 or c0 > c1:
            return False

        seen = set()
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                value = int(grid[r, c])
                if value in a["colors"]:
                    seen.add(i)
                elif value in b["colors"]:
                    seen.add(j)
        if len(seen) != 1:
            return False
        expected_top = i if rank[i] > rank[j] else j
        if seen != {expected_top}:
            return False
    return True


def _solver_matches_generated(grid: np.ndarray, rects: list[dict], order: list[int]) -> bool:
    interpretations = _interpretations(grid)
    canonical = {_canonical_solution(found_rects, found_order): (found_rects, found_order, found_shape) for found_rects, found_order, found_shape in interpretations}
    if len(canonical) != 1:
        return False

    found_rects, found_order, found_shape = next(iter(canonical.values()))
    gen_shape = _rect_dims(rects[0]["bbox"])
    if found_shape != gen_shape:
        return False

    found_rank = {frozenset(found_rects[idx]["colors"]): depth for depth, idx in enumerate(found_order)}
    gen_rank = {frozenset(rects[idx]["colors"]): depth for depth, idx in enumerate(order)}
    if found_rank != gen_rank:
        return False

    for found in found_rects:
        key = frozenset(found["colors"])
        original = next(rect for rect in rects if frozenset(rect["colors"]) == key)
        if found["bbox"] != original["bbox"]:
            return False
        if tuple(found["marker_local"]) != tuple(original["marker_local"]):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rect_count = random.choice(RECT_COUNT_OPTIONS)
    sampled = _sample_layout(rect_count)
    if sampled is None:
        return None
    rect_shape, positions, order, grid_shape = sampled

    base_colors = random.sample(BASE_COLORS, rect_count)
    marker_colors = random.sample(MARKER_COLORS, rect_count)

    rects = []
    for idx in range(rect_count):
        r0, c0 = positions[idx]
        h, w = rect_shape
        rects.append(
            {
                "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
                "base": int(base_colors[idx]),
                "marker": int(marker_colors[idx]),
                "marker_local": tuple(),
                "colors": frozenset({int(base_colors[idx]), int(marker_colors[idx])}),
            }
        )

    visible_locals, visible_area_counts = _visible_local_info(rects, order, rect_shape)
    marker_sets = _assign_visible_markers(visible_locals, visible_area_counts)
    if marker_sets is None:
        return None
    for idx, markers in enumerate(marker_sets):
        rects[idx]["marker_local"] = markers

    grid = _render_input(rects, order, grid_shape, rect_shape)
    expected = _expected_output(rects, order, rect_shape)

    if any(not _visible_bbox_matches(grid, rect) for rect in rects):
        return None
    if not _pairwise_visibility_matches(grid, rects, order):
        return None
    if not _solver_matches_generated(grid, rects, order):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    top_idx = order[-1]
    top_original = _make_patch(rects[top_idx]["base"], rects[top_idx]["marker"], rects[top_idx]["marker_local"], rect_shape)
    if np.array_equal(expected, top_original):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        made = _gen_attempt()
        if made is not None:
            return made
    raise RuntimeError("failed to generate a valid input for puzzle 8")
