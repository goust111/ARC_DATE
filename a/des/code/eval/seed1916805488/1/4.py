from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [p[0] for p in points]
    cs = [p[1] for p in points]
    return min(rs), min(cs), max(rs), max(cs)


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _ring_border_points(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    pts = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _detect_palette(grid: np.ndarray, bg: int) -> tuple[list[int], set[tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[tuple[int, list[int], set[tuple[int, int]]]] = []
    corner_specs = [
        (0, 0, 1),
        (0, w - 1, 1),
        (h - 1, 0, -1),
        (h - 1, w - 1, -1),
    ]
    for start_r, c, step in corner_specs:
        colors: list[int] = []
        cells: set[tuple[int, int]] = set()
        r = start_r
        while 0 <= r < h and int(grid[r, c]) != bg:
            colors.append(int(grid[r, c]))
            cells.add((r, c))
            r += step
        if not colors or len(set(colors)) != len(colors):
            continue
        ok = True
        for pr, pc in cells:
            side_c = pc + 1 if pc == 0 else pc - 1
            if 0 <= side_c < w and int(grid[pr, side_c]) != bg:
                ok = False
                break
        if ok:
            candidates.append((len(colors), colors, cells))
    assert candidates, "failed to detect corner palette"
    max_len = max(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == max_len]
    assert len(best) == 1, "palette corner must be unique"
    _, colors, cells = best[0]
    return colors, cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill nested rectangular rings using a depth palette placed in a grid corner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    palette, palette_cells = _detect_palette(g, bg)

    object_mask = (g != bg).copy()
    for r, c in palette_cells:
        object_mask[r, c] = False

    comps = _components(object_mask)
    rings: list[dict[str, object]] = []
    border_mask = np.zeros_like(g, dtype=bool)
    for comp in comps:
        box = _bbox(comp)
        r0, c0, r1, c1 = box
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "ring must be at least 3x3"
        expected = _ring_border_points(box)
        actual = set(comp)
        assert actual == expected, "component must be a 1-cell-thick rectangular ring"
        color = int(g[comp[0]])
        assert all(int(g[r, c]) == color for r, c in comp), "each ring must be single-colored"
        for r, c in comp:
            border_mask[r, c] = True
        rings.append({"box": box, "color": color})

    assert rings, "expected at least one ring"
    for ring in rings:
        depth = 1 + sum(_strictly_contains(other["box"], ring["box"]) for other in rings)
        ring["depth"] = depth
    max_depth = max(int(ring["depth"]) for ring in rings)
    assert len(palette) == max_depth, "palette length must match maximum nesting depth"

    out = g.copy()
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            if border_mask[r, c] or (r, c) in palette_cells:
                continue
            deepest = 0
            for ring in rings:
                r0, c0, r1, c1 = ring["box"]
                if r0 < r < r1 and c0 < c < c1:
                    deepest = max(deepest, int(ring["depth"]))
            if deepest > 0:
                out[r, c] = palette[deepest - 1]
    return out


MIN_SIZE = 18
MAX_SIZE = 30
MAX_GENERATION_ATTEMPTS = 400
LAYOUT_MODES = ("chain_only", "chain_plus_siblings", "multi_root", "mixed")


def _min_dim_for_depth(depth: int) -> int:
    return 4 * depth - 1


def _random_child_box(
    parent: tuple[int, int, int, int], remaining_depth: int, rng: random.Random
) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = parent
    ph = r1 - r0 + 1
    pw = c1 - c0 + 1
    min_dim = _min_dim_for_depth(remaining_depth)
    if ph < min_dim + 4 or pw < min_dim + 4:
        return None
    child_h = rng.randint(min_dim, ph - 4)
    child_w = rng.randint(min_dim, pw - 4)
    extra_h = ph - child_h
    extra_w = pw - child_w
    top = rng.randint(2, extra_h - 2)
    left = rng.randint(2, extra_w - 2)
    bottom = extra_h - top
    right = extra_w - left
    return (r0 + top, c0 + left, r1 - bottom, c1 - right)


def _build_chain(
    root: tuple[int, int, int, int], depth: int, rng: random.Random
) -> list[tuple[int, int, int, int]] | None:
    boxes = [root]
    cur = root
    for remaining_depth in range(depth - 1, 0, -1):
        nxt = _random_child_box(cur, remaining_depth, rng)
        if nxt is None:
            return None
        boxes.append(nxt)
        cur = nxt
    return boxes


def _boxes_interfere(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


def _collect_all_boxes(nodes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    return list(nodes)


def _sibling_candidates(
    parent: tuple[int, int, int, int], child: tuple[int, int, int, int]
) -> list[tuple[int, int, int, int]]:
    pr0, pc0, pr1, pc1 = parent
    cr0, cc0, cr1, cc1 = child
    candidates = [
        (pr0 + 2, pc0 + 2, cr0 - 2, pc1 - 2),
        (cr1 + 2, pc0 + 2, pr1 - 2, pc1 - 2),
        (pr0 + 2, pc0 + 2, pr1 - 2, cc0 - 2),
        (pr0 + 2, cc1 + 2, pr1 - 2, pc1 - 2),
    ]
    return [box for box in candidates if box[2] - box[0] >= 2 and box[3] - box[1] >= 2]


def _place_sibling_ring(
    boxes: list[tuple[int, int, int, int]],
    chain: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> bool:
    if len(chain) < 2:
        return False
    parent_idx = rng.randint(0, len(chain) - 2)
    parent = chain[parent_idx]
    child = chain[parent_idx + 1]
    candidates = _sibling_candidates(parent, child)
    rng.shuffle(candidates)
    for area in candidates:
        ar0, ac0, ar1, ac1 = area
        max_h = min(ar1 - ar0 + 1, 9)
        max_w = min(ac1 - ac0 + 1, 9)
        if max_h < 3 or max_w < 3:
            continue
        for _ in range(15):
            h = rng.randint(3, max_h)
            w = rng.randint(3, max_w)
            r0 = rng.randint(ar0, ar1 - h + 1)
            c0 = rng.randint(ac0, ac1 - w + 1)
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if all(not _boxes_interfere(box, other, pad=1) for other in boxes if not _strictly_contains(other, box)):
                boxes.append(box)
                return True
    return False


def _random_root_box(
    h: int,
    w: int,
    depth: int,
    corner: str,
    used: list[tuple[int, int, int, int]],
    rng: random.Random,
    extra_slack: int = 0,
) -> tuple[int, int, int, int] | None:
    min_h = _min_dim_for_depth(depth) + extra_slack
    min_w = _min_dim_for_depth(depth) + extra_slack
    col_lo = 1
    col_hi = w - 2
    if "left" in corner:
        col_lo = 2
    else:
        col_hi = w - 3
    row_span = h - 2
    col_span = col_hi - col_lo + 1
    if row_span < min_h or col_span < min_w:
        return None
    for _ in range(60):
        box_h = rng.randint(min_h, min(min_h + 8, row_span))
        box_w = rng.randint(min_w, min(min_w + 8, col_span))
        r0 = rng.randint(1, h - box_h - 1)
        c0 = rng.randint(col_lo, col_hi - box_w + 1)
        box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        if all(not _boxes_interfere(box, other, pad=1) for other in used):
            return box
    return None


def _draw_ring(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _paint_palette(grid: np.ndarray, colors: list[int], corner: str) -> None:
    h, w = grid.shape
    if corner == "top_left":
        for i, color in enumerate(colors):
            grid[i, 0] = color
    elif corner == "top_right":
        for i, color in enumerate(colors):
            grid[i, w - 1] = color
    elif corner == "bottom_left":
        for i, color in enumerate(colors):
            grid[h - 1 - i, 0] = color
    else:
        for i, color in enumerate(colors):
            grid[h - 1 - i, w - 1] = color


def _assign_ring_colors(
    boxes: list[tuple[int, int, int, int]], palette_colors: list[int], rng: random.Random
) -> list[int]:
    color_pool = [color for color in range(1, 10) if color not in palette_colors]
    ring_data = []
    for box in boxes:
        parent = None
        parent_area = None
        for other in boxes:
            if other == box or not _strictly_contains(other, box):
                continue
            area = (other[2] - other[0] + 1) * (other[3] - other[1] + 1)
            if parent_area is None or area < parent_area:
                parent = other
                parent_area = area
        ring_data.append((box, parent))

    color_by_box: dict[tuple[int, int, int, int], int] = {}
    for box, parent in sorted(ring_data, key=lambda item: (item[0][0], item[0][1], -(item[0][2] - item[0][0]))):
        choices = color_pool[:]
        if parent is not None and color_by_box[parent] in choices and len(choices) > 1:
            choices.remove(color_by_box[parent])
        color_by_box[box] = rng.choice(choices)
    return [color_by_box[box] for box in boxes]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    mode = rng.choice(LAYOUT_MODES)
    depth = rng.randint(1, 4)
    h = rng.randint(max(MIN_SIZE, depth + 10), MAX_SIZE)
    w = rng.randint(max(MIN_SIZE, depth + 10), MAX_SIZE)
    corner = rng.choice(("top_left", "top_right", "bottom_left", "bottom_right"))
    grid = np.zeros((h, w), dtype=int)

    roots: list[list[tuple[int, int, int, int]]] = []
    used_roots: list[tuple[int, int, int, int]] = []
    root = _random_root_box(
        h,
        w,
        depth,
        corner,
        used_roots,
        rng,
        extra_slack=3 if mode in ("chain_plus_siblings", "mixed") else 0,
    )
    if root is None:
        return None
    main_chain = _build_chain(root, depth, rng)
    if main_chain is None:
        return None
    roots.append(main_chain)
    used_roots.append(root)

    all_boxes = _collect_all_boxes(main_chain)
    if mode in ("chain_plus_siblings", "mixed"):
        sibling_count = rng.randint(1, 2)
        placed = 0
        for _ in range(sibling_count):
            if _place_sibling_ring(all_boxes, main_chain, rng):
                placed += 1
        if placed == 0:
            return None

    if mode in ("multi_root", "mixed"):
        extra_roots = rng.randint(1, 2)
        for _ in range(extra_roots):
            extra_depth = rng.randint(1, min(3, depth + 1))
            extra_root = _random_root_box(h, w, extra_depth, corner, used_roots, rng)
            if extra_root is None:
                continue
            chain = _build_chain(extra_root, extra_depth, rng)
            if chain is None:
                continue
            roots.append(chain)
            used_roots.append(extra_root)
            all_boxes.extend(chain)
        if len(roots) == 1:
            return None

    palette_colors = rng.sample(list(range(1, 10)), depth)
    ring_colors = _assign_ring_colors(all_boxes, palette_colors, rng)
    for box, color in zip(all_boxes, ring_colors):
        _draw_ring(grid, box, color)
    _paint_palette(grid, palette_colors, corner)

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    if len(np.unique(out)) < 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
