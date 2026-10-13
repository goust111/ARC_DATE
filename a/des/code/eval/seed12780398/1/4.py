import random

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 300


def _bbox_from_mask(mask: np.ndarray):
    pts = np.argwhere(mask)
    assert pts.size > 0
    return int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()), int(pts[:, 1].max())


def _bbox_union(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def _rect_contains_bbox(rect, bbox):
    if bbox is None:
        return True
    return rect[0] <= bbox[0] and rect[1] <= bbox[1] and rect[2] >= bbox[2] and rect[3] >= bbox[3]


def _integral(mask: np.ndarray):
    return mask.astype(np.int32).cumsum(axis=0).cumsum(axis=1)


def _rect_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    total = int(ps[r1, c1])
    if r0 > 0:
        total -= int(ps[r0 - 1, c1])
    if c0 > 0:
        total -= int(ps[r1, c0 - 1])
    if r0 > 0 and c0 > 0:
        total += int(ps[r0 - 1, c0 - 1])
    return total


def _compose(rects_by_color, order, shape):
    out = np.full(shape, BG, dtype=int)
    for color in order:
        r0, c0, r1, c1 = rects_by_color[color]
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _topo_order(colors, edges):
    indeg = {color: 0 for color in colors}
    for src in colors:
        for dst in edges[src]:
            indeg[dst] += 1
    ready = sorted([color for color in colors if indeg[color] == 0])
    order = []
    while ready:
        color = ready.pop(0)
        order.append(color)
        for dst in sorted(edges[color]):
            indeg[dst] -= 1
            if indeg[dst] == 0:
                ready.append(dst)
                ready.sort()
    return order if len(order) == len(colors) else None


def _acyclic(edges, nodes):
    nodes = set(nodes)
    state = {}

    def dfs(node):
        state[node] = 1
        for nxt in edges[node]:
            if nxt not in nodes:
                continue
            if state.get(nxt, 0) == 1:
                return False
            if state.get(nxt, 0) == 0 and not dfs(nxt):
                return False
        state[node] = 2
        return True

    for node in nodes:
        if state.get(node, 0) == 0 and not dfs(node):
            return False
    return True


def _extract_prompt(grid: np.ndarray):
    grey_pts = np.argwhere(grid == GREY)
    assert grey_pts.size > 0, "expected a grey prompt strip"
    split_col = int(grey_pts[:, 1].min())
    main = np.asarray(grid[:, :split_col], dtype=int)

    pr0 = int(grey_pts[:, 0].min())
    pc0 = int(grey_pts[:, 1].min())
    pr1 = int(grey_pts[:, 0].max())
    pc1 = int(grey_pts[:, 1].max())
    prompt_area = grid[pr0 : pr1 + 1, pc0 : pc1 + 1]

    prompt_points = []
    for r in range(prompt_area.shape[0]):
        for c in range(prompt_area.shape[1]):
            value = int(prompt_area[r, c])
            if value not in (BG, GREY):
                prompt_points.append((pr0 + r, c, value))
    prompt_points.sort(key=lambda item: (item[0], item[1]))
    prompt_colors = [value for _r, _c, value in prompt_points]
    assert prompt_colors, "prompt strip must contain colored depth hints"
    return main, list(reversed(prompt_colors))


def _enumerate_candidates(grid: np.ndarray, color: int):
    h, w = grid.shape
    nonbg = grid != BG
    ps = _integral(nonbg)
    vis = grid == color
    rmin, cmin, rmax, cmax = _bbox_from_mask(vis)

    candidates = []
    for r0 in range(rmin + 1):
        for c0 in range(cmin + 1):
            for r1 in range(rmax, h):
                for c1 in range(cmax, w):
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if _rect_sum(ps, r0, c0, r1, c1) != area:
                        continue

                    cover_map = {}
                    window = grid[r0 : r1 + 1, c0 : c1 + 1]
                    for other in np.unique(window):
                        other = int(other)
                        if other in (BG, color):
                            continue
                        pts = np.argwhere(window == other)
                        bbox = (
                            r0 + int(pts[:, 0].min()),
                            c0 + int(pts[:, 1].min()),
                            r0 + int(pts[:, 0].max()),
                            c0 + int(pts[:, 1].max()),
                        )
                        cover_map[other] = bbox

                    candidates.append(
                        {
                            "rect": (r0, c0, r1, c1),
                            "cover_map": cover_map,
                            "area": area,
                        }
                    )

    candidates.sort(key=lambda cand: (cand["area"], cand["rect"]))
    assert candidates, f"no candidate rectangles for color {color}"
    return candidates


def _solve_main_area(grid: np.ndarray, max_solutions: int = 2):
    colors = sorted(int(v) for v in np.unique(grid) if int(v) != BG)
    assert colors, "expected at least one rectangle color"

    candidates = {color: _enumerate_candidates(grid, color) for color in colors}
    solutions = []

    def dfs(chosen, required, edges):
        if len(solutions) >= max_solutions:
            return
        if len(chosen) == len(colors):
            order = _topo_order(colors, edges)
            if order is None:
                return
            rects = {color: chosen[color]["rect"] for color in colors}
            if not np.array_equal(_compose(rects, order, grid.shape), grid):
                return
            solutions.append(rects)
            return

        next_color = None
        feasible = None
        for color in colors:
            if color in chosen:
                continue
            cur = [cand for cand in candidates[color] if _rect_contains_bbox(cand["rect"], required[color])]
            if not cur:
                return
            if feasible is None or len(cur) < len(feasible):
                next_color = color
                feasible = cur

        for cand in feasible:
            new_required = dict(required)
            new_edges = {color: set(targets) for color, targets in edges.items()}
            new_chosen = dict(chosen)
            new_chosen[next_color] = cand

            ok = True
            for target, bbox in cand["cover_map"].items():
                new_edges[next_color].add(target)
                new_required[target] = _bbox_union(new_required[target], bbox)

            if not _acyclic(new_edges, new_chosen.keys()):
                ok = False

            if ok:
                for color, chosen_cand in new_chosen.items():
                    if not _rect_contains_bbox(chosen_cand["rect"], new_required[color]):
                        ok = False
                        break

            if ok:
                for color in colors:
                    if color in new_chosen:
                        continue
                    if not any(_rect_contains_bbox(other["rect"], new_required[color]) for other in candidates[color]):
                        ok = False
                        break

            if ok:
                dfs(new_chosen, new_required, new_edges)

    dfs(
        chosen={},
        required={color: _bbox_from_mask(grid == color) for color in colors},
        edges={color: set() for color in colors},
    )
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each hidden rectangle from the left composite, then redraw the same
    rectangles using the reversed order encoded by the right grey prompt strip.
    """
    g = np.asarray(grid, dtype=int)
    main_area, draw_order = _extract_prompt(g)

    solutions = _solve_main_area(main_area, max_solutions=1)
    assert solutions, "failed to recover a consistent rectangle set"
    rects = solutions[0]

    colors = set(rects)
    assert colors == set(draw_order), "prompt colors must match the rectangle colors"
    return _compose(rects, draw_order, main_area.shape)


def _random_rectangles(h: int, w: int, colors, rng: random.Random):
    center_r = rng.randint(2, h - 3)
    center_c = rng.randint(2, w - 3)
    rects = {}
    used = set()
    for color in colors:
        for _ in range(80):
            top = rng.randint(max(0, center_r - 6), center_r - 1)
            bottom = rng.randint(center_r + 1, min(h - 1, center_r + 6))
            left = rng.randint(max(0, center_c - 6), center_c - 1)
            right = rng.randint(center_c + 1, min(w - 1, center_c + 6))
            if bottom - top + 1 < 3 or right - left + 1 < 3:
                continue
            rect = (top, left, bottom, right)
            if rect in used:
                continue
            used.add(rect)
            rects[color] = rect
            break
        if color not in rects:
            return None
    return rects


def _pairwise_overlap(rect_a, rect_b):
    r0 = max(rect_a[0], rect_b[0])
    c0 = max(rect_a[1], rect_b[1])
    r1 = min(rect_a[2], rect_b[2])
    c1 = min(rect_a[3], rect_b[3])
    return max(0, r1 - r0 + 1) * max(0, c1 - c0 + 1)


def _build_prompt_strip(grid: np.ndarray, main_w: int, colors, rng: random.Random):
    n = len(colors)
    strip_h = 4 * n + 1
    top = rng.randint(0, grid.shape[0] - strip_h)
    left = main_w

    grid[top : top + strip_h, left] = GREY
    grid[top : top + strip_h, left + 2] = GREY
    for i in range(n + 1):
        grid[top + 4 * i, left : left + 3] = GREY
    for i, color in enumerate(colors):
        grid[top + 2 + 4 * i, left + 1] = color


def _visible_stats(main_area: np.ndarray, rects_by_color):
    stats = {}
    for color, rect in rects_by_color.items():
        vis = main_area == color
        if not np.any(vis):
            return None
        stats[color] = {
            "visible": int(vis.sum()),
            "area": (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1),
        }
    return stats


def _gen_attempt(rng: random.Random):
    n = rng.randint(3, 5)
    main_h = rng.randint(4 * n + 1, 4 * n + 4)
    main_w = rng.randint(10, 17)
    colors = rng.sample(COLOR_POOL, n)

    rects = _random_rectangles(main_h, main_w, colors, rng)
    if rects is None:
        return None

    overlap_pairs = sum(
        1
        for i, ca in enumerate(colors)
        for cb in colors[i + 1 :]
        if _pairwise_overlap(rects[ca], rects[cb]) > 0
    )
    if overlap_pairs == 0:
        return None

    init_order = colors[:]
    rng.shuffle(init_order)
    main_area = _compose(rects, init_order, (main_h, main_w))

    stats = _visible_stats(main_area, rects)
    if stats is None:
        return None
    if not any(info["visible"] < info["area"] for info in stats.values()):
        return None

    solutions = _solve_main_area(main_area, max_solutions=2)
    if len(solutions) != 1 or solutions[0] != rects:
        return None

    target_order = colors[:]
    rng.shuffle(target_order)
    if target_order == init_order:
        return None
    output = _compose(rects, target_order, (main_h, main_w))
    if np.array_equal(output, main_area):
        return None

    full = np.full((main_h, main_w + 3), BG, dtype=int)
    full[:, :main_w] = main_area
    _build_prompt_strip(full, main_w, list(reversed(target_order)), rng)
    return full


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
