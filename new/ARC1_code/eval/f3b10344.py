from __future__ import annotations

from collections import defaultdict
from itertools import groupby

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, get_bg
import random


BRIDGE_COLOR = 8
Rect = tuple[int, int, int, int]
Bridge = tuple[int, Rect]


def _rectangles_by_color(g: np.ndarray, bg: int) -> dict[int, list[Rect]]:
    rectangles: dict[int, list[Rect]] = defaultdict(list)
    for color in colors_present(g, exclude={bg}):
        assert color != BRIDGE_COLOR, "bridge color should not appear in the input"
        for coords in components_mask(g == color, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(coords)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            assert coords.shape[0] == h * w, "foreground objects must be solid rectangles"
            assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle bbox must be solid"
            rectangles[int(color)].append((int(r0), int(c0), int(r1), int(c1)))

    assert rectangles, "expected at least one foreground rectangle"
    return dict(rectangles)


def _candidate_bridge(g: np.ndarray, bg: int, a: Rect, b: Rect) -> Bridge | None:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    candidates: list[Bridge] = []

    shared_r0 = max(ar0, br0)
    shared_r1 = min(ar1, br1)
    shared_c0 = max(ac0, bc0)
    shared_c1 = min(ac1, bc1)
    assert shared_r1 < shared_r0 or shared_c1 < shared_c0, "separate solid rectangles cannot overlap"

    if shared_r1 - shared_r0 + 1 >= 3 and (ac1 < bc0 or bc1 < ac0):
        if ac1 < bc0:
            gap_c0, gap_c1 = ac1 + 1, bc0 - 1
        else:
            gap_c0, gap_c1 = bc1 + 1, ac0 - 1

        if gap_c0 <= gap_c1 and np.all(g[shared_r0 : shared_r1 + 1, gap_c0 : gap_c1 + 1] == bg):
            paint = (shared_r0 + 1, gap_c0, shared_r1 - 1, gap_c1)
            candidates.append((gap_c1 - gap_c0 + 1, paint))

    if shared_c1 - shared_c0 + 1 >= 3 and (ar1 < br0 or br1 < ar0):
        if ar1 < br0:
            gap_r0, gap_r1 = ar1 + 1, br0 - 1
        else:
            gap_r0, gap_r1 = br1 + 1, ar0 - 1

        if gap_r0 <= gap_r1 and np.all(g[gap_r0 : gap_r1 + 1, shared_c0 : shared_c1 + 1] == bg):
            paint = (gap_r0, shared_c0 + 1, gap_r1, shared_c1 - 1)
            candidates.append((gap_r1 - gap_r0 + 1, paint))

    assert len(candidates) <= 1, "rectangle pair should face in at most one direction"
    return candidates[0] if candidates else None


def _find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: list[int], a: int, b: int) -> None:
    ra = _find(parent, a)
    rb = _find(parent, b)
    assert ra != rb, "union expects distinct sets"
    parent[ra] = rb


def _assert_equal_weight_forest(parent: list[int], group: list[tuple[int, int, Bridge]]) -> None:
    reps = sorted({_find(parent, i) for i, _, _ in group} | {_find(parent, j) for _, j, _ in group})
    index = {rep: k for k, rep in enumerate(reps)}
    temp_parent = list(range(len(reps)))

    for i, j, _bridge in group:
        ri = index[_find(parent, i)]
        rj = index[_find(parent, j)]
        if _find(temp_parent, ri) == _find(temp_parent, rj):
            raise AssertionError("equal-length bridge choices must be cycle-free")
        _union(temp_parent, ri, rj)


def _paint_shortest_connectors(g: np.ndarray, out: np.ndarray, bg: int, rectangles: list[Rect]) -> None:
    parent = list(range(len(rectangles)))
    edges: list[tuple[int, int, int, Bridge]] = []

    for i, rect_a in enumerate(rectangles):
        for j in range(i + 1, len(rectangles)):
            bridge = _candidate_bridge(g, bg, rect_a, rectangles[j])
            if bridge is not None:
                distance, _paint = bridge
                edges.append((distance, i, j, bridge))

    used = 0
    for _distance, distance_edges_iter in groupby(sorted(edges, key=lambda e: e[0]), key=lambda e: e[0]):
        distance_edges = [(i, j, bridge) for _dist, i, j, bridge in distance_edges_iter]
        open_edges = [(i, j, bridge) for i, j, bridge in distance_edges if _find(parent, i) != _find(parent, j)]
        if not open_edges:
            continue

        _assert_equal_weight_forest(parent, open_edges)
        for i, j, bridge in open_edges:
            assert _find(parent, i) != _find(parent, j), "equal-length bridge group should remain acyclic"
            _union(parent, i, j)
            used += 1

            _distance, (r0, c0, r1, c1) = bridge
            assert r0 <= r1 and c0 <= c1, "painted bridge must be non-empty"
            out[r0 : r1 + 1, c0 : c1 + 1] = BRIDGE_COLOR

    assert len(rectangles) == 1 or used == len(rectangles) - 1, "same-color rectangles must be connectable"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint teal bridges through the shortest empty gaps needed to connect each color's solid rectangles.

    Input structure
    - The background is the dominant color and every foreground object is a solid axis-aligned rectangle.
    - Rectangles of the same color may face each other horizontally or vertically across empty background.
    - Teal is absent from the input and is reserved for the added bridge pixels.

    Rule
    - Preserve the input and process each foreground color independently.
    - Candidate bridges join same-color rectangles whose row or column spans overlap by at least three cells and whose intervening gap is all background.
    - Paint the selected gap teal, using the shared span eroded by one pixel on both sides.

    Key insight
    - The selected bridges are the shortest facing gaps that connect each same-color rectangle group; longer gaps that would only create a cycle are left blank.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    assert bg != BRIDGE_COLOR, "background cannot be the bridge color"
    out = g.copy()

    for rectangles in _rectangles_by_color(g, bg).values():
        _paint_shortest_connectors(g, out, bg, rectangles)

    return out

# --------------- generator (o5x) ---------------

# Colors.
GEN_BRIDGE_COLOR = 8
GEN_BG = 0
GEN_FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

# Canvas and palette size.
GEN_HEIGHT_RANGE = (25, 30)
GEN_WIDTH_RANGE = (29, 30)
GEN_COLOR_COUNT_CHOICES = (2, 3)

# Rectangle counts and dimensions.
GEN_RECT_SIZE_RANGE = (3, 9)
GEN_TOTAL_RECTS_RANGE = (5, 11)
GEN_RECTS_PER_COLOR_MIN = 2
GEN_RECTS_PER_COLOR_MAX = 6

# Bridge geometry.
GEN_MIN_OVERLAP = 3
GEN_GAP_RANGE = (1, 10)
GEN_ORIENTATIONS = ("H", "V")
GEN_SIDES = (-1, 1)
GEN_MAX_EXTRA_EDGES = 1

# Rejection budgets and final density floor.
GEN_LAYOUT_TRIES = 120
GEN_EDGE_TRIES = 28
GEN_GROUP_TRIES = 80
GEN_ATTEMPTS = 300
GEN_MIN_OCCUPANCY_DIVISOR = 9

GEN_Rect = tuple[int, int, int, int]
GEN_Bridge = tuple[int, GEN_Rect]
GEN_EdgeMap = dict[tuple[int, int], GEN_Bridge]
GEN_Group = tuple[list[GEN_Rect], list[tuple[int, int]], GEN_EdgeMap]


def _gen_rects_overlap(a: GEN_Rect, b: GEN_Rect) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _gen_shift_rect(rect: GEN_Rect, dr: int, dc: int) -> GEN_Rect:
    r0, c0, r1, c1 = rect
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _gen_rects_shape(rects: list[GEN_Rect]) -> tuple[int, int]:
    return max(r[2] for r in rects) + 1, max(r[3] for r in rects) + 1


def _gen_normalize_rects(rects: list[GEN_Rect]) -> list[GEN_Rect]:
    r0 = min(r[0] for r in rects)
    c0 = min(r[1] for r in rects)
    return [_gen_shift_rect(rect, -r0, -c0) for rect in rects]


def _gen_candidate_bridge(g: np.ndarray, a: GEN_Rect, b: GEN_Rect) -> GEN_Bridge | None:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    candidates: list[GEN_Bridge] = []

    shared_r0 = max(ar0, br0)
    shared_r1 = min(ar1, br1)
    shared_c0 = max(ac0, bc0)
    shared_c1 = min(ac1, bc1)
    if shared_r1 >= shared_r0 and shared_c1 >= shared_c0:
        return None

    if shared_r1 - shared_r0 + 1 >= GEN_MIN_OVERLAP and (ac1 < bc0 or bc1 < ac0):
        gap_c0, gap_c1 = (ac1 + 1, bc0 - 1) if ac1 < bc0 else (bc1 + 1, ac0 - 1)
        if gap_c0 <= gap_c1 and np.all(g[shared_r0 : shared_r1 + 1, gap_c0 : gap_c1 + 1] == GEN_BG):
            candidates.append((gap_c1 - gap_c0 + 1, (shared_r0 + 1, gap_c0, shared_r1 - 1, gap_c1)))

    if shared_c1 - shared_c0 + 1 >= GEN_MIN_OVERLAP and (ar1 < br0 or br1 < ar0):
        gap_r0, gap_r1 = (ar1 + 1, br0 - 1) if ar1 < br0 else (br1 + 1, ar0 - 1)
        if gap_r0 <= gap_r1 and np.all(g[gap_r0 : gap_r1 + 1, shared_c0 : shared_c1 + 1] == GEN_BG):
            candidates.append((gap_r1 - gap_r0 + 1, (gap_r0, shared_c0 + 1, gap_r1, shared_c1 - 1)))

    return candidates[0] if len(candidates) == 1 else None


def _gen_paint_rect(g: np.ndarray, rect: GEN_Rect, value: int | bool) -> None:
    r0, c0, r1, c1 = rect
    g[r0 : r1 + 1, c0 : c1 + 1] = value


def _gen_candidate_edges(rects: list[GEN_Rect]) -> GEN_EdgeMap:
    r0 = min(r[0] for r in rects)
    c0 = min(r[1] for r in rects)
    shifted = [_gen_shift_rect(rect, -r0, -c0) for rect in rects]
    h, w = _gen_rects_shape(shifted)
    g = np.zeros((h, w), dtype=int)
    for rect in shifted:
        _gen_paint_rect(g, rect, 1)

    edges: GEN_EdgeMap = {}
    for i in range(len(shifted)):
        for j in range(i + 1, len(shifted)):
            bridge = _gen_candidate_bridge(g, shifted[i], shifted[j])
            if bridge is not None:
                dist, paint = bridge
                edges[(i, j)] = (dist, _gen_shift_rect(paint, r0, c0))
    return edges


def _gen_try_place_child(
    rects: list[GEN_Rect],
    dims: list[tuple[int, int]],
    child: int,
) -> tuple[GEN_Rect, tuple[int, int]] | None:
    """Sample one child rectangle attached to any existing parent."""
    child_h, child_w = dims[child]
    parents = list(range(child))
    random.shuffle(parents)
    for parent in parents:
        pr0, pc0, pr1, pc1 = rects[parent]
        parent_h, parent_w = pr1 - pr0 + 1, pc1 - pc0 + 1
        for _edge_try in range(GEN_EDGE_TRIES):
            gap = random.randint(*GEN_GAP_RANGE)
            orientation = random.choice(GEN_ORIENTATIONS)
            side = random.choice(GEN_SIDES)
            if orientation == "H":
                overlap = random.randint(GEN_MIN_OVERLAP, min(parent_h, child_h))
                shared_r0 = random.randint(pr0, pr1 - overlap + 1)
                child_r0 = shared_r0 - random.randint(0, child_h - overlap)
                child_c0 = pc1 + gap + 1 if side > 0 else pc0 - gap - child_w
            else:
                overlap = random.randint(GEN_MIN_OVERLAP, min(parent_w, child_w))
                shared_c0 = random.randint(pc0, pc1 - overlap + 1)
                child_c0 = shared_c0 - random.randint(0, child_w - overlap)
                child_r0 = pr1 + gap + 1 if side > 0 else pr0 - gap - child_h

            candidate = (child_r0, child_c0, child_r0 + child_h - 1, child_c0 + child_w - 1)
            if any(_gen_rects_overlap(candidate, existing) for existing in rects):
                continue
            return candidate, (min(parent, child), max(parent, child))
    return None


def _gen_candidate_edges_pass_mst_checks(tree_edges: list[tuple[int, int]], candidate_edges: GEN_EdgeMap) -> bool:
    """Keep groups where tree edges are the only shortest MST choices."""
    tree_set = set(tree_edges)
    if not tree_set.issubset(candidate_edges):
        return False

    tree_distances = [candidate_edges[e][0] for e in tree_edges]
    max_tree_distance = max(tree_distances)
    extra_edges = [e for e in candidate_edges if e not in tree_set]
    if len(extra_edges) > GEN_MAX_EXTRA_EDGES:
        return False
    return not any(candidate_edges[e][0] <= max_tree_distance for e in extra_edges)


def _gen_tree_bridges_are_disjoint(rects: list[GEN_Rect], tree_edges: list[tuple[int, int]], candidate_edges: GEN_EdgeMap) -> bool:
    """Reject layouts whose selected bridge paint regions overlap."""
    bridge_mask = np.zeros(_gen_rects_shape(rects), dtype=bool)
    for edge in tree_edges:
        _dist, paint = candidate_edges[edge]
        r0, c0, r1, c1 = paint
        if np.any(bridge_mask[r0 : r1 + 1, c0 : c1 + 1]):
            return False
        _gen_paint_rect(bridge_mask, paint, True)
    return True


def _gen_build_group(n_rects: int) -> GEN_Group | None:
    for _layout_try in range(GEN_LAYOUT_TRIES):
        dims = [(random.randint(*GEN_RECT_SIZE_RANGE), random.randint(*GEN_RECT_SIZE_RANGE)) for _ in range(n_rects)]
        rects: list[GEN_Rect] = [(0, 0, dims[0][0] - 1, dims[0][1] - 1)]
        tree_edges: list[tuple[int, int]] = []

        for child in range(1, n_rects):
            placed = _gen_try_place_child(rects, dims, child)
            if placed is None:
                break
            candidate, edge = placed
            rects.append(candidate)
            tree_edges.append(edge)

        if len(rects) != n_rects:
            continue

        rects = _gen_normalize_rects(rects)
        candidate_edges = _gen_candidate_edges(rects)
        if not _gen_candidate_edges_pass_mst_checks(tree_edges, candidate_edges):
            continue
        if not _gen_tree_bridges_are_disjoint(rects, tree_edges, candidate_edges):
            continue

        return rects, tree_edges, candidate_edges
    return None


def _gen_group_mask(rects: list[GEN_Rect], tree_edges: list[tuple[int, int]], candidate_edges: GEN_EdgeMap) -> np.ndarray:
    """Return occupied cells for a placed color group, including tree bridges."""
    group_mask = np.zeros(_gen_rects_shape(rects), dtype=bool)
    for rect in rects:
        _gen_paint_rect(group_mask, rect, True)
    for edge in tree_edges:
        _dist, paint = candidate_edges[edge]
        _gen_paint_rect(group_mask, paint, True)
    return group_mask


def _gen_place_group(
    shape: tuple[int, int],
    occupied: np.ndarray,
    n_rects: int,
) -> GEN_Group | None:
    h, w = shape
    for _group_try in range(GEN_GROUP_TRIES):
        built = _gen_build_group(n_rects)
        if built is None:
            continue
        rects, tree_edges, candidate_edges = built
        gh, gw = _gen_rects_shape(rects)
        if gh > h or gw > w:
            continue

        group_mask = _gen_group_mask(rects, tree_edges, candidate_edges)
        valid_offsets: list[tuple[int, int]] = []
        for dr in range(h - gh + 1):
            for dc in range(w - gw + 1):
                if not np.any(occupied[dr : dr + gh, dc : dc + gw] & group_mask):
                    valid_offsets.append((dr, dc))
        if not valid_offsets:
            continue

        dr, dc = random.choice(valid_offsets)
        shifted_rects = [_gen_shift_rect(rect, dr, dc) for rect in rects]
        shifted_edges = {
            edge: (dist, _gen_shift_rect(paint, dr, dc))
            for edge, (dist, paint) in candidate_edges.items()
        }
        return shifted_rects, tree_edges, shifted_edges
    return None


def _gen_counts(n_colors: int) -> list[int]:
    total = random.randint(
        max(GEN_TOTAL_RECTS_RANGE[0], GEN_RECTS_PER_COLOR_MIN * n_colors),
        min(GEN_TOTAL_RECTS_RANGE[1], GEN_RECTS_PER_COLOR_MAX * n_colors),
    )
    counts = [GEN_RECTS_PER_COLOR_MIN] * n_colors
    remaining = total - GEN_RECTS_PER_COLOR_MIN * n_colors
    while remaining:
        choices = [i for i, count in enumerate(counts) if count < GEN_RECTS_PER_COLOR_MAX]
        idx = random.choice(choices)
        counts[idx] += 1
        remaining -= 1
    random.shuffle(counts)
    return counts


def _gen_paint_group(out: np.ndarray, occupied: np.ndarray, color: int, group: GEN_Group) -> None:
    rects, tree_edges, candidate_edges = group
    for rect in rects:
        _gen_paint_rect(out, rect, color)
        _gen_paint_rect(occupied, rect, True)

    for edge in tree_edges:
        _dist, paint = candidate_edges[edge]
        _gen_paint_rect(out, paint, GEN_BRIDGE_COLOR)
        _gen_paint_rect(occupied, paint, True)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)
    n_colors = random.choice(GEN_COLOR_COUNT_CHOICES)
    colors = random.sample(GEN_FOREGROUND_COLORS, n_colors)
    counts = _gen_counts(n_colors)

    out = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    for color, n_rects in zip(colors, counts):
        placed = _gen_place_group((h, w), occupied, n_rects)
        if placed is None:
            return None
        _gen_paint_group(out, occupied, color, placed)

    inp = out.copy()
    inp[inp == GEN_BRIDGE_COLOR] = GEN_BG
    if int(np.count_nonzero(inp)) < (h * w) // GEN_MIN_OCCUPANCY_DIVISOR:
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _attempt in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid f3b10344 input")
