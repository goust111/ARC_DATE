from __future__ import annotations

import random
from collections import deque

import numpy as np


# ARC colors used by the puzzle description.
BG = 0
RECT = 5
SEED = 9
LEGEND_COLORS = (1, 2, 3, 4, 7)  # blue, red, green, yellow, orange


# Generator configuration.
GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 32)
GEN_RETRIES = 400
TREE_RETRIES = 120
ROOT_PLACEMENT_RETRIES = 120
CHILD_PLACEMENT_RETRIES = 120
MIN_LEAF_SIDE = 5
COUNT_OPTIONS = (1, 2, 3, 4, 5, 6, 7)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(coords)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _is_frame_component(mask: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    sub = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        sub[r - r0, c - c0] = True
    border = np.zeros_like(sub, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(sub, border))


def _legend_colors(grid: np.ndarray, bg: int) -> list[int]:
    row = grid[-1]
    runs: list[int] = []
    last = None
    for value in row:
        value = int(value)
        if value == bg:
            last = None
            continue
        if last != value:
            runs.append(value)
            last = value
    assert len(runs) >= 5, "expected a five-color legend on the bottom row"
    return runs[:5]


def _find_rectangle_color(body: np.ndarray, bg: int) -> int:
    candidates: list[tuple[int, int]] = []
    for color in sorted(set(map(int, body.flatten())) - {int(bg)}):
        comps = _components(body == color)
        if comps and all(_is_frame_component(body == color, comp) for comp in comps):
            candidates.append((len(comps), color))
    assert candidates, "expected at least one rectangular frame color"
    candidates.sort(reverse=True)
    return int(candidates[0][1])


def _parse_rectangles(grid: np.ndarray) -> tuple[list[dict[str, object]], int, list[int], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    legend = _legend_colors(g, bg)
    body = g[:-1]
    rect_color = _find_rectangle_color(body, bg)

    rectangles: list[dict[str, object]] = []
    for comp in _components(body == rect_color):
        assert _is_frame_component(body == rect_color, comp), "rectangle components must be axis-aligned frames"
        box = _bbox(comp)
        rectangles.append(
            {
                "bbox": box,
                "coords": comp,
                "children": [],
                "parent": None,
                "seed_count": 0,
            }
        )

    assert rectangles, "expected at least one rectangle"

    for i, rect in enumerate(rectangles):
        box = rect["bbox"]
        parent_idx = None
        parent_area = None
        for j, other in enumerate(rectangles):
            if i == j:
                continue
            r0, c0, r1, c1 = box
            or0, oc0, or1, oc1 = other["bbox"]
            if or0 < r0 and oc0 < c0 and or1 > r1 and oc1 > c1:
                area = _bbox_area(other["bbox"])
                if parent_area is None or area < parent_area:
                    parent_area = area
                    parent_idx = j
        rect["parent"] = parent_idx

    for i, rect in enumerate(rectangles):
        parent_idx = rect["parent"]
        if parent_idx is not None:
            rectangles[parent_idx]["children"].append(i)

    seed_mask = (body != bg) & (body != rect_color)
    for sr, sc in np.argwhere(seed_mask):
        containers: list[tuple[int, int]] = []
        for i, rect in enumerate(rectangles):
            r0, c0, r1, c1 = rect["bbox"]
            if r0 < int(sr) < r1 and c0 < int(sc) < c1:
                containers.append((_bbox_area(rect["bbox"]), i))
        assert containers, "every seed must lie inside some rectangle"
        containers.sort()
        rectangles[containers[0][1]]["seed_count"] += 1

    return rectangles, bg, legend, rect_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested rectangular containers by the number of directly visible elements.

    Directly visible elements are:
    - maroon singleton seeds inside the rectangle but not inside a deeper one
    - directly nested child rectangles

    After recoloring all frames using the bottom legend, remove the seeds.
    """
    g = np.asarray(grid, dtype=int)
    rectangles, bg, legend, rect_color = _parse_rectangles(g)
    out = g.copy()

    body = g[:-1]
    seed_mask = (body != bg) & (body != rect_color)
    out_body = out[:-1]
    out_body[seed_mask] = bg

    for rect in rectangles:
        visible_count = int(rect["seed_count"]) + len(rect["children"])
        assert visible_count >= 1, "every rectangle must contain at least one visible element"
        color = legend[min(visible_count, len(legend)) - 1]
        for r, c in rect["coords"]:
            out[r, c] = color
    return out


def _min_side_for_depth(depth: int) -> int:
    return MIN_LEAF_SIDE + 4 * (depth - 1)


def _boxes_too_close(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    margin: int,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _visible_cells(
    box: tuple[int, int, int, int],
    child_boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells: list[tuple[int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            inside_child = False
            for cr0, cc0, cr1, cc1 in child_boxes:
                if cr0 <= r <= cr1 and cc0 <= c <= cc1:
                    inside_child = True
                    break
            if not inside_child:
                cells.append((r, c))
    return cells


def _choose_target_count(rng: random.Random, n_children: int, capacity: int) -> int | None:
    options = [count for count in COUNT_OPTIONS if count >= max(1, n_children) and count - n_children <= capacity]
    if not options:
        return None
    rng.shuffle(options)
    return int(options[0])


def _place_child_boxes(
    box: tuple[int, int, int, int],
    child_depths: list[int],
    rng: random.Random,
) -> list[tuple[int, int, int, int]] | None:
    r0, c0, r1, c1 = box
    child_depths = sorted(child_depths, reverse=True)
    usable_r0 = r0 + 2
    usable_c0 = c0 + 2
    usable_r1 = r1 - 2
    usable_c1 = c1 - 2
    if usable_r0 > usable_r1 or usable_c0 > usable_c1:
        return None

    placed: list[tuple[int, int, int, int]] = []
    for depth in child_depths:
        min_side = _min_side_for_depth(depth)
        max_h = min(usable_r1 - usable_r0 + 1, min_side + 5)
        max_w = min(usable_c1 - usable_c0 + 1, min_side + 5)
        if max_h < min_side or max_w < min_side:
            return None

        success = None
        for _ in range(CHILD_PLACEMENT_RETRIES):
            h = rng.randint(min_side, max_h)
            w = rng.randint(min_side, max_w)
            rr = rng.randint(usable_r0, usable_r1 - h + 1)
            cc = rng.randint(usable_c0, usable_c1 - w + 1)
            candidate = (rr, cc, rr + h - 1, cc + w - 1)
            if any(_boxes_too_close(candidate, other, margin=1) for other in placed):
                continue
            success = candidate
            break
        if success is None:
            return None
        placed.append(success)
    return placed


def _build_tree(
    box: tuple[int, int, int, int],
    depth: int,
    rng: random.Random,
) -> dict[str, object] | None:
    for _ in range(TREE_RETRIES):
        children: list[dict[str, object]] = []
        if depth == 1:
            child_boxes: list[tuple[int, int, int, int]] = []
        else:
            r0, c0, r1, c1 = box
            area = (r1 - r0 - 1) * (c1 - c0 - 1)
            can_branch = area >= 170 and (r1 - r0 + 1) >= 12 and (c1 - c0 + 1) >= 12
            n_children = 1 + int(can_branch and rng.random() < 0.55)
            child_depths = [depth - 1] + [rng.randint(1, depth - 1) for _ in range(n_children - 1)]
            rng.shuffle(child_depths)
            child_depths = sorted(child_depths, reverse=True)
            child_boxes = _place_child_boxes(box, child_depths, rng)
            if child_boxes is None:
                continue
            ok = True
            for child_box, child_depth in zip(child_boxes, child_depths):
                child = _build_tree(child_box, child_depth, rng)
                if child is None:
                    ok = False
                    break
                children.append(child)
            if not ok:
                continue

        visible = _visible_cells(box, [child["bbox"] for child in children])
        target = _choose_target_count(rng, len(children), len(visible))
        if target is None:
            continue
        seed_count = target - len(children)
        seeds = rng.sample(visible, seed_count)
        return {"bbox": box, "children": children, "seeds": seeds, "target": target}
    return None


def _draw_box(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _render_tree(grid: np.ndarray, tree: dict[str, object]) -> None:
    _draw_box(grid, tree["bbox"], RECT)
    for child in tree["children"]:
        _render_tree(grid, child)
    for r, c in tree["seeds"]:
        grid[r, c] = SEED


def _tree_depth(tree: dict[str, object]) -> int:
    children = tree["children"]
    if not children:
        return 1
    return 1 + max(_tree_depth(child) for child in children)


def _flatten_trees(trees: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    stack = list(trees)
    while stack:
        node = stack.pop()
        out.append(node)
        stack.extend(node["children"])
    return out


def _add_legend(grid: np.ndarray, rng: random.Random) -> None:
    run_lengths = [rng.randint(1, 3) for _ in LEGEND_COLORS]
    total = sum(run_lengths)
    assert total <= grid.shape[1] - 2
    start = rng.randint(1, grid.shape[1] - total - 1)
    col = start
    for color, length in zip(LEGEND_COLORS, run_lengths):
        grid[-1, col : col + length] = color
        col += length


def _root_count_for_depth(depth: int, rng: random.Random) -> int:
    if depth == 3:
        return rng.choice((1, 2))
    if depth == 2:
        return rng.choice((1, 2, 3))
    return rng.choice((2, 3))


def _place_root_boxes(
    h: int,
    w: int,
    root_depths: list[int],
    rng: random.Random,
) -> list[tuple[int, int, int, int]] | None:
    usable_r0 = 1
    usable_c0 = 1
    usable_r1 = h - 3
    usable_c1 = w - 2
    if usable_r1 <= usable_r0 or usable_c1 <= usable_c0:
        return None

    placed: list[tuple[int, int, int, int] | None] = [None] * len(root_depths)
    order = sorted(range(len(root_depths)), key=lambda i: root_depths[i], reverse=True)
    for idx in order:
        depth = root_depths[idx]
        min_side = _min_side_for_depth(depth)
        max_h = min(usable_r1 - usable_r0 + 1, min_side + 7)
        max_w = min(usable_c1 - usable_c0 + 1, min_side + 7)
        if max_h < min_side or max_w < min_side:
            return None

        success = None
        for _ in range(ROOT_PLACEMENT_RETRIES):
            hh = rng.randint(min_side, max_h)
            ww = rng.randint(min_side, max_w)
            rr = rng.randint(usable_r0, usable_r1 - hh + 1)
            cc = rng.randint(usable_c0, usable_c1 - ww + 1)
            candidate = (rr, cc, rr + hh - 1, cc + ww - 1)
            others = [box for box in placed if box is not None]
            if any(_boxes_too_close(candidate, other, margin=2) for other in others):
                continue
            success = candidate
            break
        if success is None:
            return None
        placed[idx] = success

    return [box for box in placed if box is not None]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    max_depth = rng.choice((1, 2, 3))
    n_roots = _root_count_for_depth(max_depth, rng)
    root_depths = [max_depth] + [rng.randint(1, max_depth) for _ in range(n_roots - 1)]
    rng.shuffle(root_depths)
    root_boxes = _place_root_boxes(h, w, root_depths, rng)
    if root_boxes is None:
        return None

    trees: list[dict[str, object]] = []
    for box, depth in zip(root_boxes, root_depths):
        tree = _build_tree(box, depth, rng)
        if tree is None:
            return None
        trees.append(tree)

    for tree in trees:
        _render_tree(grid, tree)
    _add_legend(grid, rng)

    if _get_bg(grid) != BG:
        return None

    parsed, bg, legend, _ = _parse_rectangles(grid)
    if bg != BG or legend != list(LEGEND_COLORS):
        return None

    flat = _flatten_trees(trees)
    if len(parsed) != len(flat):
        return None
    if max(_tree_depth(tree) for tree in trees) != max_depth:
        return None

    counts = sorted(int(node["target"]) for node in flat)
    parsed_counts = sorted(int(rect["seed_count"]) + len(rect["children"]) for rect in parsed)
    if counts != parsed_counts:
        return None

    out = main(grid)
    if np.any(out[:-1] == RECT) or np.any(out[:-1] == SEED):
        return None
    if not np.array_equal(out[-1], grid[-1]):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
