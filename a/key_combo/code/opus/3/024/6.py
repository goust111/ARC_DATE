from __future__ import annotations

from collections import defaultdict
from importlib.machinery import SourceFileLoader
from itertools import groupby
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_024_6",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()
bbox_coords = _ARC_UTILS.bbox_coords
components_mask = _ARC_UTILS.components_mask
colors_present = _ARC_UTILS.colors_present
get_bg = _ARC_UTILS.get_bg
paste = _ARC_UTILS.paste


BG = 0
GREY = 5
ORANGE = 7
CYAN = 8

LEGEND_H = 9
LEGEND_W = 5
GEN_RETRIES = 400
COLOR_POOL = [1, 2, 3, 4, 6, 9]

Rect = tuple[int, int, int, int]
Bridge = tuple[int, Rect]

SYMBOL_BY_DEGREE = {
    1: np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),  # cross
    2: np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),  # T
    3: np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=int),  # L
}


def _bbox_h(rect: Rect) -> int:
    return int(rect[2] - rect[0] + 1)


def _bbox_w(rect: Rect) -> int:
    return int(rect[3] - rect[1] + 1)


def _rect_shape(rects: list[Rect]) -> tuple[int, int]:
    return max(r[2] for r in rects) + 1, max(r[3] for r in rects) + 1


def _normalize_rects(rects: list[Rect]) -> list[Rect]:
    r0 = min(r[0] for r in rects)
    c0 = min(r[1] for r in rects)
    return [(r[0] - r0, r[1] - c0, r[2] - r0, r[3] - c0) for r in rects]


def _shift_rects(rects: list[Rect], dr: int, dc: int) -> list[Rect]:
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in rects]


def _rotate_rects(rects: list[Rect], k: int) -> list[Rect]:
    k = int(k) % 4
    out = list(rects)
    for _ in range(k):
        h, w = _rect_shape(out)
        rotated = []
        for r0, c0, r1, c1 in out:
            rotated.append((c0, h - 1 - r1, c1, h - 1 - r0))
        out = _normalize_rects(rotated)
    return out


def _rect_center(rect: Rect) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    assert (r1 - r0 + 1) % 2 == 1 and (c1 - c0 + 1) % 2 == 1, "rectangle center must be integral"
    return (r0 + r1) // 2, (c0 + c1) // 2


def _paint_rect(grid: np.ndarray, rect: Rect, color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _bboxes_overlap(a: Rect, b: Rect, pad: int = 0) -> bool:
    pad = int(pad)
    return not (
        a[2] + pad < b[0] - pad
        or b[2] + pad < a[0] - pad
        or a[3] + pad < b[1] - pad
        or b[3] + pad < a[1] - pad
    )


def _legend_bbox(shape: tuple[int, int]) -> Rect:
    h, w = map(int, shape)
    return h - LEGEND_H, w - LEGEND_W, h - 1, w - 1


def _legend_panel() -> np.ndarray:
    panel = np.full((LEGEND_H, LEGEND_W), GREY, dtype=int)
    order = [1, 2, 3]
    random.shuffle(order)
    for band_index, degree in enumerate(order):
        band_r0 = 3 * band_index
        label_col = random.randint(0, 1)
        label_r0 = random.randint(0, 3 - degree)
        panel[band_r0 + label_r0 : band_r0 + label_r0 + degree, label_col] = ORANGE
        symbol = SYMBOL_BY_DEGREE[degree].astype(bool)
        panel[band_r0 : band_r0 + 3, 2:5][symbol] = CYAN
    return panel


def _read_legend(panel: np.ndarray) -> dict[int, np.ndarray]:
    assert panel.shape == (LEGEND_H, LEGEND_W), f"expected {LEGEND_H}x{LEGEND_W} legend panel"
    mapping: dict[int, np.ndarray] = {}
    for band_index in range(3):
        band = panel[3 * band_index : 3 * band_index + 3, :]
        degree = int(np.count_nonzero(band[:, :2] == ORANGE))
        symbol = (band[:, 2:5] == CYAN).astype(int)
        assert 1 <= degree <= 3, f"legend degree out of range: {degree}"
        assert degree not in mapping, "legend degrees must be unique"
        mapping[degree] = symbol
    assert set(mapping) == {1, 2, 3}, "legend must encode degrees 1, 2, and 3"
    return mapping


def _extract_rectangles(work: np.ndarray) -> list[dict]:
    rectangles: list[dict] = []
    for color in colors_present(work, exclude={BG}):
        for coords in components_mask(work == color, connectivity=4):
            rect = bbox_coords(coords)
            r0, c0, r1, c1 = rect
            assert coords.shape[0] == _bbox_h(rect) * _bbox_w(rect), "foreground objects must be solid rectangles"
            assert np.all(work[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle interior must be solid"
            rectangles.append({"color": int(color), "bbox": rect})
    assert rectangles, "expected at least one rectangle outside the legend"
    return rectangles


def _candidate_bridge(grid: np.ndarray, a: Rect, b: Rect) -> Bridge | None:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    candidates: list[Bridge] = []

    shared_r0 = max(ar0, br0)
    shared_r1 = min(ar1, br1)
    shared_c0 = max(ac0, bc0)
    shared_c1 = min(ac1, bc1)
    assert shared_r1 < shared_r0 or shared_c1 < shared_c0, "rectangles must not overlap"

    if shared_r1 - shared_r0 + 1 >= 3 and (ac1 < bc0 or bc1 < ac0):
        gap_c0, gap_c1 = (ac1 + 1, bc0 - 1) if ac1 < bc0 else (bc1 + 1, ac0 - 1)
        if gap_c0 <= gap_c1 and np.all(grid[shared_r0 : shared_r1 + 1, gap_c0 : gap_c1 + 1] == BG):
            candidates.append((gap_c1 - gap_c0 + 1, (shared_r0 + 1, gap_c0, shared_r1 - 1, gap_c1)))

    if shared_c1 - shared_c0 + 1 >= 3 and (ar1 < br0 or br1 < ar0):
        gap_r0, gap_r1 = (ar1 + 1, br0 - 1) if ar1 < br0 else (br1 + 1, ar0 - 1)
        if gap_r0 <= gap_r1 and np.all(grid[gap_r0 : gap_r1 + 1, shared_c0 : shared_c1 + 1] == BG):
            candidates.append((gap_r1 - gap_r0 + 1, (gap_r0, shared_c0 + 1, gap_r1, shared_c1 - 1)))

    assert len(candidates) <= 1, "a rectangle pair must face in at most one direction"
    return candidates[0] if candidates else None


def _find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: list[int], a: int, b: int) -> None:
    ra = _find(parent, a)
    rb = _find(parent, b)
    assert ra != rb, "union expects two distinct sets"
    parent[ra] = rb


def _assert_equal_weight_forest(parent: list[int], edges: list[tuple[int, int]]) -> None:
    if not edges:
        return
    reps = sorted({_find(parent, i) for i, _ in edges} | {_find(parent, j) for _, j in edges})
    rep_index = {rep: k for k, rep in enumerate(reps)}
    temp_parent = list(range(len(reps)))
    for i, j in edges:
        ri = rep_index[_find(parent, i)]
        rj = rep_index[_find(parent, j)]
        if _find(temp_parent, ri) == _find(temp_parent, rj):
            raise AssertionError("equal-length candidate bridges would make the topology ambiguous")
        _union(temp_parent, ri, rj)


def _topology_degrees(work: np.ndarray, rectangles: list[dict]) -> dict:
    degrees = [0] * len(rectangles)
    by_color: dict[int, list[int]] = defaultdict(list)
    for index, rect in enumerate(rectangles):
        by_color[int(rect["color"])].append(index)

    total_candidates = 0
    total_selected = 0
    skipped_edges = 0
    candidate_counts: list[int] = []
    degree_multisets: list[list[int]] = []

    for color_indices in by_color.values():
        parent = list(range(len(color_indices)))
        edges: list[tuple[int, int, int, Bridge]] = []
        for i, idx_a in enumerate(color_indices):
            for j in range(i + 1, len(color_indices)):
                idx_b = color_indices[j]
                bridge = _candidate_bridge(work, rectangles[idx_a]["bbox"], rectangles[idx_b]["bbox"])
                if bridge is not None:
                    distance, _paint = bridge
                    edges.append((distance, i, j, bridge))

        selected_in_group = 0
        for _distance, edge_iter in groupby(sorted(edges, key=lambda item: item[0]), key=lambda item: item[0]):
            edge_group = [(i, j, bridge) for _dist, i, j, bridge in edge_iter]
            open_edges = [(i, j, bridge) for i, j, bridge in edge_group if _find(parent, i) != _find(parent, j)]
            _assert_equal_weight_forest(parent, [(i, j) for i, j, _ in open_edges])
            for i, j, _bridge in open_edges:
                _union(parent, i, j)
                degrees[color_indices[i]] += 1
                degrees[color_indices[j]] += 1
                selected_in_group += 1

        if len(color_indices) > 1:
            assert selected_in_group == len(color_indices) - 1, "same-color rectangles must form a tree"
        total_candidates += len(edges)
        total_selected += selected_in_group
        skipped_edges += len(edges) - selected_in_group
        candidate_counts.append(len(edges))
        degree_multisets.append(sorted(degrees[idx] for idx in color_indices))

    return {
        "degrees": degrees,
        "candidate_counts": sorted(candidate_counts),
        "degree_multisets": sorted(degree_multisets),
        "total_candidates": int(total_candidates),
        "total_selected": int(total_selected),
        "skipped_edges": int(skipped_edges),
    }


def _analyse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert get_bg(g) == BG, "background must be black"
    assert g.shape[0] > LEGEND_H and g.shape[1] > LEGEND_W, "grid must leave room for the legend"

    lr0, lc0, lr1, lc1 = _legend_bbox(g.shape)
    panel = g[lr0 : lr1 + 1, lc0 : lc1 + 1].copy()
    legend_map = _read_legend(panel)

    work = g.copy()
    work[lr0 : lr1 + 1, lc0 : lc1 + 1] = BG
    rectangles = _extract_rectangles(work)
    topology = _topology_degrees(work, rectangles)

    return {
        "grid": g,
        "panel_bbox": (lr0, lc0, lr1, lc1),
        "panel": panel,
        "legend_map": legend_map,
        "work": work,
        "rectangles": rectangles,
        **topology,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each solid rectangle by a degree-coded symbol read from the legend after inferring hidden bridges.

    Input structure:
    - The background is black and the main field contains several solid axis-aligned rectangles in a small set of colors.
    - Same-color rectangles may face each other across empty horizontal or vertical gaps with at least three cells of overlap.
    - The lower-right corner holds a grey legend panel with three stacked cyan 3x3 symbols and orange labels whose pixel counts encode degrees.

    Rule:
    - For each color independently, enumerate same-color rectangle pairs that face across an all-black gap, then sort those candidate bridges by gap length.
    - Select only the shortest candidates that connect different same-color components, so each color class forms an acyclic bridge tree; the bridge count incident to a rectangle is its degree.
    - Read the legend as a lookup table from orange-count degree to cyan symbol shape, erase the rectangles, and place the corresponding 3x3 symbol in each rectangle's original color centered on its bbox.

    Key insight:
    - The bridges are never visible in the final output: they exist only as hidden topology used to compute each rectangle's degree, and the legend converts that degree into the replacement symbol.
    """
    info = _analyse_grid(np.asarray(grid, dtype=int))
    out = np.zeros_like(info["grid"])
    lr0, lc0, lr1, lc1 = info["panel_bbox"]
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = info["panel"]

    for rect_info, degree in zip(info["rectangles"], info["degrees"]):
        rect = rect_info["bbox"]
        center_r, center_c = _rect_center(rect)
        symbol = info["legend_map"][int(degree)]
        patch = np.where(symbol.astype(bool), int(rect_info["color"]), BG).astype(int)
        paste(out, patch, center_r - 1, center_c - 1, transparent=BG)

    return out


def _make_star_group() -> list[Rect]:
    size = random.choice([3, 5])
    gap_left = random.randint(1, 2)
    gap_right = random.randint(1, 2)
    gap_top = random.randint(1, 2)

    center = (size + gap_top, size + gap_left, size + gap_top + size - 1, size + gap_left + size - 1)
    left = (size + gap_top, 0, size + gap_top + size - 1, size - 1)
    right_c0 = center[3] + gap_right + 1
    right = (size + gap_top, right_c0, size + gap_top + size - 1, right_c0 + size - 1)
    top = (0, size + gap_left, size - 1, size + gap_left + size - 1)

    return _rotate_rects([center, left, right, top], random.randint(0, 3))


def _make_square_group() -> list[Rect]:
    size = 5
    gap_top, gap_left, gap_right = random.choice([(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)])

    top_left = (0, 0, size - 1, size - 1)
    top_right = (0, size + gap_top, size - 1, 2 * size + gap_top - 1)
    bottom_left = (size + gap_left, 0, 2 * size + gap_left - 1, size - 1)
    bottom_right = (
        size + gap_right,
        size + gap_top,
        2 * size + gap_right - 1,
        2 * size + gap_top - 1,
    )

    return _rotate_rects([top_left, top_right, bottom_left, bottom_right], random.randint(0, 3))


def _place_group(local_rects: list[Rect], shape: tuple[int, int], blocked: list[Rect]) -> list[Rect] | None:
    h, w = map(int, shape)
    gh, gw = _rect_shape(local_rects)
    if gh + 2 > h or gw + 2 > w:
        return None

    for _ in range(250):
        top = random.randint(1, h - gh - 1)
        left = random.randint(1, w - gw - 1)
        placed = _shift_rects(local_rects, top, left)
        bbox = (top, left, top + gh - 1, left + gw - 1)
        if any(_bboxes_overlap(bbox, other, pad=1) for other in blocked):
            continue
        return placed
    return None


def _validate_generated_grid(grid: np.ndarray) -> None:
    info = _analyse_grid(grid)
    assert len(info["rectangles"]) == 8, "generator should place exactly eight rectangles"
    assert info["candidate_counts"] == [3, 4], f"expected one star group and one cycle group, got {info['candidate_counts']}"
    assert info["degree_multisets"] == [[1, 1, 1, 3], [1, 1, 2, 2]], (
        f"unexpected degree patterns: {info['degree_multisets']}"
    )
    assert info["total_candidates"] == 7, f"expected seven candidate bridges, got {info['total_candidates']}"
    assert info["total_selected"] == 6, f"expected six selected bridges, got {info['total_selected']}"
    assert info["skipped_edges"] == 1, "generator must include a non-selected cycle-closing candidate"
    assert set(info["degrees"]) == {1, 2, 3}, "all output symbols must appear in each sample"

    out = main(grid)
    lr0, lc0, lr1, lc1 = info["panel_bbox"]
    assert np.array_equal(out[lr0 : lr1 + 1, lc0 : lc1 + 1], info["panel"]), "legend must stay unchanged"

    occupied = np.zeros_like(out, dtype=bool)
    occupied[lr0 : lr1 + 1, lc0 : lc1 + 1] = True
    for rect_info, degree in zip(info["rectangles"], info["degrees"]):
        rect = rect_info["bbox"]
        color = int(rect_info["color"])
        center_r, center_c = _rect_center(rect)
        patch = out[center_r - 1 : center_r + 2, center_c - 1 : center_c + 2]
        expected = np.where(info["legend_map"][int(degree)].astype(bool), color, BG)
        assert np.array_equal(patch, expected), "output symbol does not match legend lookup"

        r0, c0, r1, c1 = rect
        box = out[r0 : r1 + 1, c0 : c1 + 1].copy()
        box[center_r - 1 - r0 : center_r + 2 - r0, center_c - 1 - c0 : center_c + 2 - c0] = BG
        assert not box.any(), "rectangle area outside the centered symbol must be cleared"
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True

    out_without_known = out.copy()
    out_without_known[occupied] = BG
    assert not out_without_known.any(), "output should contain only legend pixels and replacement symbols"


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        h = random.randint(26, 29)
        w = random.randint(26, 29)
        grid = np.zeros((h, w), dtype=int)

        legend = _legend_panel()
        lr0, lc0, lr1, lc1 = _legend_bbox(grid.shape)
        grid[lr0 : lr1 + 1, lc0 : lc1 + 1] = legend

        colors = random.sample(COLOR_POOL, 2)
        groups = [
            {"rects": _make_star_group(), "color": colors[0]},
            {"rects": _make_square_group(), "color": colors[1]},
        ]
        groups.sort(key=lambda item: _rect_shape(item["rects"])[0] * _rect_shape(item["rects"])[1], reverse=True)

        blocked = [_legend_bbox(grid.shape)]
        placed_groups: list[dict] = []
        ok = True
        for group in groups:
            placed = _place_group(group["rects"], grid.shape, blocked)
            if placed is None:
                ok = False
                break
            bbox = (
                min(r[0] for r in placed),
                min(r[1] for r in placed),
                max(r[2] for r in placed),
                max(r[3] for r in placed),
            )
            blocked.append(bbox)
            placed_groups.append({"rects": placed, "color": group["color"]})

        if not ok:
            continue

        for group in placed_groups:
            for rect in group["rects"]:
                _paint_rect(grid, rect, int(group["color"]))

        try:
            _validate_generated_grid(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid bridge-topology sample")
