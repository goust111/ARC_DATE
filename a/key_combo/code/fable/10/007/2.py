from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
CYAN = 8
MAROON = 9

SEED_COLORS = np.array([BLUE, RED, GREEN, YELLOW, GREY, PINK, ORANGE, MAROON], dtype=int)

MAX_GEN_ATTEMPTS = 400
MAX_SHAPE_ATTEMPTS = 120
MAX_ORIENTATION_ATTEMPTS = 60


def _norm_rect(rect):
    r0, c0, r1, c1 = rect
    if r0 > r1:
        r0, r1 = r1, r0
    if c0 > c1:
        c0, c1 = c1, c0
    return int(r0), int(c0), int(r1), int(c1)


def _border_cells(rect):
    r0, c0, r1, c1 = rect
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _interior_cells(rect):
    r0, c0, r1, c1 = rect
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _closed_contains(rect, pt):
    r0, c0, r1, c1 = rect
    r, c = pt
    return r0 <= r <= r1 and c0 <= c <= c1


def _interior_contains(rect, pt):
    r0, c0, r1, c1 = rect
    r, c = pt
    return r0 < r < r1 and c0 < c < c1


def _strict_contains(outer, inner):
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _pair_relation(a, b):
    if a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]:
        return "disjoint"
    if _strict_contains(a, b):
        return "a_contains_b"
    if _strict_contains(b, a):
        return "b_contains_a"
    if a == b:
        return "same"
    return "cross"


def _extract_entities(grid):
    pts = [tuple(map(int, rc)) for rc in np.argwhere(grid == CYAN)]
    pts = sorted(pts)
    seeds = []
    for r, c in np.argwhere((grid != BLACK) & (grid != CYAN)):
        seeds.append((int(r), int(c), int(grid[r, c])))
    return pts, seeds


def _build_candidates(points):
    n = len(points)
    candidates_by_point = defaultdict(list)
    all_candidates = []
    for i in range(n):
        r0, c0 = points[i]
        for j in range(i + 1, n):
            r1, c1 = points[j]
            if r0 == r1 or c0 == c1:
                continue
            rect = _norm_rect((r0, c0, r1, c1))
            if rect[2] - rect[0] < 2 or rect[3] - rect[1] < 2:
                continue
            candidate = {
                "pair": (i, j),
                "rect": rect,
                "border": frozenset(_border_cells(rect)),
            }
            idx = len(all_candidates)
            all_candidates.append(candidate)
            candidates_by_point[i].append(idx)
            candidates_by_point[j].append(idx)
    return all_candidates, candidates_by_point


def _direct_seed_info(rects, seeds):
    descendants = [[] for _ in rects]
    for i, rect in enumerate(rects):
        for j, other in enumerate(rects):
            if i != j and _strict_contains(rect, other):
                descendants[i].append(j)

    counts = [0] * len(rects)
    colors = [None] * len(rects)
    for idx, rect in enumerate(rects):
        for r, c, color in seeds:
            if not _interior_contains(rect, (r, c)):
                continue
            if any(_closed_contains(rects[ch], (r, c)) for ch in descendants[idx]):
                continue
            counts[idx] += 1
            colors[idx] = int(color)
    return counts, colors


def _enumerate_pairings(points, seeds, require_border, require_seed_rule, limit=None):
    all_candidates, candidates_by_point = _build_candidates(points)
    n = len(points)
    used = [False] * n
    border_sets = []
    chosen = []
    solutions = []

    def rec():
        if limit is not None and len(solutions) >= limit:
            return
        try:
            first = next(i for i, flag in enumerate(used) if not flag)
        except StopIteration:
            rects = [all_candidates[idx]["rect"] for idx in chosen]
            if require_border:
                for i in range(len(rects)):
                    for j in range(i + 1, len(rects)):
                        if _pair_relation(rects[i], rects[j]) in {"cross", "same"}:
                            return
            colors = [None] * len(rects)
            if require_seed_rule:
                counts, colors = _direct_seed_info(rects, seeds)
                if any(count != 1 for count in counts):
                    return
            solutions.append({"rects": rects, "colors": colors})
            return

        used[first] = True
        for cand_idx in candidates_by_point[first]:
            i, j = all_candidates[cand_idx]["pair"]
            other = j if i == first else i
            if used[other]:
                continue
            border = all_candidates[cand_idx]["border"]
            if require_border and any(border & prev for prev in border_sets):
                continue
            used[other] = True
            chosen.append(cand_idx)
            if require_border:
                border_sets.append(border)
            rec()
            if require_border:
                border_sets.pop()
            chosen.pop()
            used[other] = False
        used[first] = False

    rec()
    return solutions


def _solve(grid):
    g = np.asarray(grid, dtype=int)
    points, seeds = _extract_entities(g)
    assert len(points) >= 2 and len(points) % 2 == 0, "expected an even number of cyan corner points"
    assert seeds, "expected colored seed pixels"

    solutions = _enumerate_pairings(points, seeds, require_border=True, require_seed_rule=True, limit=2)
    assert len(solutions) == 1, "expected a unique valid pairing"
    rects = solutions[0]["rects"]
    counts, colors = _direct_seed_info(rects, seeds)
    assert all(count == 1 for count in counts)
    assert all(color is not None for color in colors)
    return rects, [int(color) for color in colors]


def _render_output(shape, rects, colors):
    out = np.full(shape, BLACK, dtype=int)
    for rect in rects:
        for r, c in _border_cells(rect):
            out[r, c] = CYAN
    for idx, rect in enumerate(rects):
        mask = np.zeros(shape, dtype=bool)
        for r, c in _interior_cells(rect):
            mask[r, c] = True
        for j, other in enumerate(rects):
            if idx != j and _strict_contains(rect, other):
                r0, c0, r1, c1 = other
                mask[r0 : r1 + 1, c0 : c1 + 1] = False
        out[mask] = int(colors[idx])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild paired cyan-corner rectangles and fill each annulus with its direct seed color.

    Input structure:
    - A black background contains an even number of isolated cyan pixels; each hidden axis-aligned rectangle contributes exactly one diagonal pair of opposite corners.
    - Additional isolated non-black, non-cyan seed pixels lie inside the hidden rectangles.
    - The rectangles may be disjoint or nested, but only their diagonal corner generators are shown in the input.

    Rule:
    - Pair the cyan pixels into diagonal corner pairs that define axis-aligned rectangles.
    - The valid global pairing is the unique one whose rectangle borders are pairwise cell-disjoint and for which every rectangle has exactly one direct seed inside it, meaning a seed in its interior but not inside or on the border of any strictly inner rectangle.
    - Draw every recovered rectangle as a cyan hollow frame.
    - Fill each rectangle's interior with the color of its own direct seed, excluding cells occupied by any inner rectangle border or interior; inner rectangles determine those excluded cells themselves.

    Key insight:
    - Local corner pairing is ambiguous, so the rectangles must be recovered by a global laminar matching test that combines non-intersecting frames with the one-direct-seed-per-ring constraint.
    """
    rects, colors = _solve(grid)
    return _render_output(np.asarray(grid).shape, rects, colors)


def _make_nodes(parents):
    nodes = [{"id": i, "children": []} for i in range(len(parents))]
    roots = []
    for i, parent in enumerate(parents):
        if parent is None:
            roots.append(nodes[i])
        else:
            nodes[parent]["children"].append(nodes[i])
    return nodes, roots


def _tree_templates():
    return {
        3: [
            [None, 0, 1],
            [None, 0, 0],
            [None, None, 1],
        ],
        4: [
            [None, 0, 1, 2],
            [None, 0, 0, 0],
            [None, 0, 0, 1],
            [None, None, 1, 1],
            [None, None, 0, 1],
        ],
        5: [
            [None, 0, 1, 2, 3],
            [None, 0, 0, 1, 3],
            [None, 0, 0, 0, 2],
            [None, None, 0, 1, 1],
            [None, 0, 0, 1, 2],
        ],
    }


def _pack_children(boxes, rng):
    if len(boxes) == 1:
        (h, w) = boxes[0]
        return [(0, 0)], h, w

    orient = rng.choice(["h", "v"])
    positions = []
    if orient == "h":
        x = 0
        max_h = 0
        for idx, (h, w) in enumerate(boxes):
            y = int(rng.integers(0, 3))
            positions.append((y, x))
            max_h = max(max_h, y + h)
            x += w
            if idx + 1 < len(boxes):
                x += int(rng.integers(2, 5))
        return positions, max_h, x

    y = 0
    max_w = 0
    for idx, (h, w) in enumerate(boxes):
        x = int(rng.integers(0, 3))
        positions.append((y, x))
        max_w = max(max_w, x + w)
        y += h
        if idx + 1 < len(boxes):
            y += int(rng.integers(2, 5))
    return positions, y, max_w


def _shift_rects(rect_map, dr, dc):
    shifted = {}
    for key, rect in rect_map.items():
        r0, c0, r1, c1 = rect
        shifted[key] = (r0 + dr, c0 + dc, r1 + dr, c1 + dc)
    return shifted


def _build_subtree(node, rng):
    if not node["children"]:
        h = int(rng.integers(5, 9))
        w = int(rng.integers(5, 9))
        return {node["id"]: (0, 0, h - 1, w - 1)}, (h, w)

    child_payloads = [_build_subtree(child, rng) for child in node["children"]]
    child_sizes = [size for _rects, size in child_payloads]
    child_positions, inner_h, inner_w = _pack_children(child_sizes, rng)

    top = int(rng.integers(2, 5))
    bottom = int(rng.integers(2, 5))
    left = int(rng.integers(2, 5))
    right = int(rng.integers(2, 5))
    rects = {node["id"]: (0, 0, inner_h + top + bottom - 1, inner_w + left + right - 1)}

    for (child_rects, _size), (dr, dc) in zip(child_payloads, child_positions):
        rects.update(_shift_rects(child_rects, top + dr, left + dc))

    return rects, (inner_h + top + bottom, inner_w + left + right)


def _forest_layout(root_payloads, rng):
    sizes = [size for _rects, size in root_payloads]
    root_positions, block_h, block_w = _pack_children(sizes, rng)
    margin_top = int(rng.integers(1, 4))
    margin_bottom = int(rng.integers(1, 4))
    margin_left = int(rng.integers(1, 4))
    margin_right = int(rng.integers(1, 4))

    rects = {}
    for (payload_rects, _size), (dr, dc) in zip(root_payloads, root_positions):
        rects.update(_shift_rects(payload_rects, margin_top + dr, margin_left + dc))

    shape = (block_h + margin_top + margin_bottom, block_w + margin_left + margin_right)
    return rects, shape


def _exclusive_cells(rects_by_id, parents, node_id):
    rect = rects_by_id[node_id]
    cells = set(_interior_cells(rect))
    for other_id, parent in enumerate(parents):
        if parent is None:
            continue
        cur = other_id
        while parents[cur] is not None:
            if parents[cur] == node_id:
                r0, c0, r1, c1 = rects_by_id[other_id]
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        cells.discard((r, c))
                break
            cur = parents[cur]
    return sorted(cells)


def _render_input(shape, rects_by_id, chosen_diagonals, seeds):
    grid = np.full(shape, BLACK, dtype=int)
    for node_id, rect in rects_by_id.items():
        r0, c0, r1, c1 = rect
        if chosen_diagonals[node_id] == 0:
            pts = [(r0, c0), (r1, c1)]
        else:
            pts = [(r0, c1), (r1, c0)]
        for r, c in pts:
            grid[r, c] = CYAN
    for r, c, color in seeds:
        assert grid[r, c] == BLACK
        grid[r, c] = int(color)
    return grid


def _count_constraint_solutions(grid, require_border, require_seed_rule, limit=3):
    points, seeds = _extract_entities(grid)
    return len(_enumerate_pairings(points, seeds, require_border=require_border, require_seed_rule=require_seed_rule, limit=limit))


def _target_ambiguity_ok(grid, mode):
    both = _count_constraint_solutions(grid, True, True, limit=2)
    if both != 1:
        return False
    border_only = _count_constraint_solutions(grid, True, False, limit=3)
    seed_only = _count_constraint_solutions(grid, False, True, limit=3)
    if mode == "both_needed":
        return border_only > 1 and seed_only > 1
    if mode == "border_needed":
        return border_only > 1
    if mode == "seed_needed":
        return seed_only > 1
    return max(border_only, seed_only) > 1


def _build_attempt(rng):
    n_rects = int(rng.integers(3, 6))
    parents = list(rng.choice(_tree_templates()[n_rects]))
    nodes, roots = _make_nodes(parents)
    root_payloads = [_build_subtree(root, rng) for root in roots]
    rects_by_id, shape = _forest_layout(root_payloads, rng)
    if max(shape) > 30:
        return None

    seed_colors = rng.choice(SEED_COLORS, size=n_rects, replace=False)
    seeds = []
    for node_id in range(n_rects):
        cells = _exclusive_cells(rects_by_id, parents, node_id)
        if not cells:
            return None
        r, c = cells[int(rng.integers(0, len(cells)))]
        seeds.append((r, c, int(seed_colors[node_id])))

    target_mode = rng.choice(["border_needed", "seed_needed", "both_needed"], p=[0.35, 0.35, 0.30])
    for _ in range(MAX_ORIENTATION_ATTEMPTS):
        chosen_diagonals = {node_id: int(rng.integers(0, 2)) for node_id in range(n_rects)}
        grid = _render_input(shape, rects_by_id, chosen_diagonals, seeds)
        if not _target_ambiguity_ok(grid, target_mode):
            continue
        out = main(grid)
        ordered_rects = [rects_by_id[i] for i in range(n_rects)]
        expected = _render_output(shape, ordered_rects, seed_colors.tolist())
        if np.array_equal(out, expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
