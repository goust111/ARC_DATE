from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GEN_TEMPLATE_COLOR = 5
GEN_SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _boundary(fill: np.ndarray) -> np.ndarray:
    h, w = fill.shape
    out = np.zeros_like(fill, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not fill[r, c]:
                continue
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                out[r, c] = True
                continue
            if not (fill[r - 1, c] and fill[r + 1, c] and fill[r, c - 1] and fill[r, c + 1]):
                out[r, c] = True
    return out


def _fill_from_outline(outline: np.ndarray) -> np.ndarray:
    h, w = outline.shape
    outside = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not outline[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not outline[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not outline[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                queue.append((nr, nc))

    return outline | ~outside


def _downscale_to_third(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    small_h = h // 3
    small_w = w // 3
    if small_h <= 0 or small_w <= 0:
        raise AssertionError("template is too small to shrink by 3")

    row_edges = np.linspace(0, h, small_h + 1, dtype=int)
    col_edges = np.linspace(0, w, small_w + 1, dtype=int)
    out = np.zeros((small_h, small_w), dtype=bool)
    for r in range(small_h):
        for c in range(small_w):
            out[r, c] = mask[row_edges[r] : row_edges[r + 1], col_edges[c] : col_edges[c + 1]].any()
    return out


def _template_color(grid: np.ndarray) -> int:
    colors = grid[grid != BLACK]
    vals, counts = np.unique(colors, return_counts=True)
    return int(vals[np.argmax(counts)])


def _recover_small_masks(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    template_color = _template_color(grid)
    template_mask = grid == template_color
    r0, c0, r1, c1 = _bbox(template_mask)
    outline = template_mask[r0 : r1 + 1, c0 : c1 + 1]
    large_fill = _fill_from_outline(outline)
    small_fill = _downscale_to_third(large_fill)
    small_outline = _boundary(small_fill)
    small_interior = small_fill & ~small_outline
    return template_color, small_outline, small_interior


def _contains_center(interior: np.ndarray, dr: int, dc: int) -> bool:
    h, w = interior.shape
    center_r = h // 2
    center_c = w // 2
    rr = center_r + dr
    cc = center_c + dc
    return 0 <= rr < h and 0 <= cc < w and bool(interior[rr, cc])


def _paint_order(centers: list[tuple[int, int]], interior: np.ndarray) -> list[int]:
    n = len(centers)
    above: dict[int, set[int]] = {i: set() for i in range(n)}
    indegree: dict[int, int] = {i: 0 for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            dr = centers[j][0] - centers[i][0]
            dc = centers[j][1] - centers[i][1]
            i_contains_j = _contains_center(interior, dr, dc)
            j_contains_i = _contains_center(interior, -dr, -dc)
            if i_contains_j and not j_contains_i:
                if j not in above[i]:
                    above[i].add(j)
                    indegree[j] += 1
            elif j_contains_i and not i_contains_j:
                if i not in above[j]:
                    above[j].add(i)
                    indegree[i] += 1

    order: list[int] = []
    remaining = set(range(n))
    while remaining:
        ready = sorted(
            [i for i in remaining if indegree[i] == 0],
            key=lambda i: (centers[i][1], centers[i][0]),
        )
        if not ready:
            return sorted(range(n), key=lambda i: (centers[i][1], centers[i][0]))
        cur = ready[0]
        remaining.remove(cur)
        order.append(cur)
        for nxt in above[cur]:
            indegree[nxt] -= 1
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow one reduced template copy around each seed, then resolve overlaps by containment
    and right/down priority.
    """
    g = np.asarray(grid, dtype=int)
    template_color, small_outline, small_interior = _recover_small_masks(g)

    seeds = [(int(r), int(c), int(g[r, c])) for r, c in np.argwhere((g != BLACK) & (g != template_color))]
    centers = [(r, c) for r, c, _color in seeds]

    out = g.copy()
    out[g == template_color] = BLACK

    sh, sw = small_outline.shape
    center_r = sh // 2
    center_c = sw // 2
    outline_pixels = np.argwhere(small_outline)

    for idx in _paint_order(centers, small_interior):
        seed_r, seed_c, color = seeds[idx]
        for rr, cc in outline_pixels:
            ar = seed_r + int(rr) - center_r
            ac = seed_c + int(cc) - center_c
            if 0 <= ar < out.shape[0] and 0 <= ac < out.shape[1]:
                out[ar, ac] = color

    return out


def _shape_rectangle() -> np.ndarray:
    h = random.choice([5, 7, 9])
    w = random.choice([5, 7, 9])
    return np.ones((h, w), dtype=bool)


def _shape_diamond() -> np.ndarray:
    radius = random.choice([2, 3, 4])
    size = 2 * radius + 1
    center = radius
    yy, xx = np.indices((size, size))
    return np.abs(yy - center) + np.abs(xx - center) <= radius


def _shape_plus() -> np.ndarray:
    size = random.choice([7, 9])
    center = size // 2
    fill = np.zeros((size, size), dtype=bool)
    fill[:, center - 1 : center + 2] = True
    fill[center - 1 : center + 2, :] = True
    return fill


def _shape_corner_notch() -> np.ndarray:
    h = random.choice([7, 9])
    w = random.choice([7, 9])
    fill = np.ones((h, w), dtype=bool)
    notch_h = random.choice([2, 3])
    notch_w = random.choice([2, 3])
    fill[:notch_h, w - notch_w :] = False

    transforms = random.randrange(8)
    if transforms & 1:
        fill = np.fliplr(fill)
    if transforms & 2:
        fill = np.flipud(fill)
    if transforms & 4:
        fill = np.rot90(fill)
    return np.ascontiguousarray(fill)


def _shape_side_bite() -> np.ndarray:
    h = random.choice([7, 9])
    w = random.choice([9, 11])
    fill = np.ones((h, w), dtype=bool)
    bite_h = 3
    top = h // 2 - bite_h // 2
    left = w // 2 + 1
    fill[top : top + bite_h, left:] = False

    transforms = random.randrange(8)
    if transforms & 1:
        fill = np.fliplr(fill)
    if transforms & 2:
        fill = np.flipud(fill)
    if transforms & 4:
        fill = np.rot90(fill)
    return np.ascontiguousarray(fill)


def _make_small_fill(scenario: str) -> np.ndarray:
    if scenario == "containment":
        return random.choice([_shape_corner_notch, _shape_side_bite])()
    return random.choice(
        [_shape_rectangle, _shape_diamond, _shape_plus, _shape_corner_notch, _shape_side_bite]
    )()


def _offset_sets(
    outline: np.ndarray,
    interior: np.ndarray,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = outline.shape
    outline_points = set(map(tuple, np.argwhere(outline)))

    overlap_offsets: set[tuple[int, int]] = set()
    contain_offsets: set[tuple[int, int]] = set()
    asymmetric_overlap_offsets: list[tuple[int, int]] = []
    overlap_only_offsets: list[tuple[int, int]] = []

    for dr in range(-(h - 1), h):
        for dc in range(-(w - 1), w):
            if dr == 0 and dc == 0:
                continue

            overlaps = any((r + dr, c + dc) in outline_points for r, c in outline_points)
            if overlaps:
                overlap_offsets.add((dr, dc))

            forward_contains = _contains_center(interior, dr, dc)
            reverse_contains = _contains_center(interior, -dr, -dc)
            if forward_contains:
                contain_offsets.add((dr, dc))

            if overlaps and forward_contains and not reverse_contains:
                asymmetric_overlap_offsets.append((dr, dc))
            if overlaps and not forward_contains and not reverse_contains:
                overlap_only_offsets.append((dr, dc))

    return overlap_offsets, contain_offsets, asymmetric_overlap_offsets, overlap_only_offsets


def _classify_seed_centers(
    h: int,
    w: int,
    large_fill_abs: np.ndarray,
    large_outline_abs: np.ndarray,
    small_shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    sh, sw = small_shape
    center_r = sh // 2
    center_c = sw // 2

    allowed: list[tuple[int, int]] = []
    inside: list[tuple[int, int]] = []
    outside: list[tuple[int, int]] = []

    for r in range(center_r, h - (sh - center_r - 1)):
        for c in range(center_c, w - (sw - center_c - 1)):
            if large_outline_abs[r, c]:
                continue
            pos = (r, c)
            allowed.append(pos)
            if large_fill_abs[r, c]:
                inside.append(pos)
            else:
                outside.append(pos)

    return allowed, inside, outside


def _position_relations(
    positions: list[tuple[int, int]],
    overlap_offsets: set[tuple[int, int]],
    contain_offsets: set[tuple[int, int]],
) -> tuple[int, list[tuple[int, int]]]:
    overlap_count = 0
    edges: list[tuple[int, int]] = []
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dr = positions[j][0] - positions[i][0]
            dc = positions[j][1] - positions[i][1]
            if (dr, dc) in overlap_offsets:
                overlap_count += 1
            if (dr, dc) in contain_offsets:
                edges.append((i, j))
            if (-dr, -dc) in contain_offsets:
                edges.append((j, i))
    return overlap_count, edges


def _has_cycle(n: int, edges: list[tuple[int, int]]) -> bool:
    above: dict[int, set[int]] = {i: set() for i in range(n)}
    indegree = [0] * n
    for src, dst in edges:
        if dst not in above[src]:
            above[src].add(dst)
            indegree[dst] += 1

    stack = [i for i in range(n) if indegree[i] == 0]
    seen = 0
    while stack:
        cur = stack.pop()
        seen += 1
        for nxt in above[cur]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                stack.append(nxt)
    return seen != n


def _try_add_separate_position(
    chosen: list[tuple[int, int]],
    candidates: list[tuple[int, int]],
    overlap_offsets: set[tuple[int, int]],
    contain_offsets: set[tuple[int, int]],
) -> tuple[int, int] | None:
    options = list(candidates)
    random.shuffle(options)
    for pos in options:
        if pos in chosen:
            continue
        ok = True
        for other in chosen:
            dr = pos[0] - other[0]
            dc = pos[1] - other[1]
            if (dr, dc) in overlap_offsets or (dr, dc) in contain_offsets or (-dr, -dc) in contain_offsets:
                ok = False
                break
        if ok:
            return pos
    return None


def _build_seed_positions(
    scenario: str,
    allowed: list[tuple[int, int]],
    inside: list[tuple[int, int]],
    outside: list[tuple[int, int]],
    overlap_offsets: set[tuple[int, int]],
    contain_offsets: set[tuple[int, int]],
    asymmetric_overlap_offsets: list[tuple[int, int]],
    overlap_only_offsets: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    allowed_set = set(allowed)
    inside_set = set(inside)
    outside_set = set(outside)
    seed_count = random.randint(2, 4)
    positions: list[tuple[int, int]] = []

    if scenario == "separate":
        pools: list[list[tuple[int, int]]] = []
        if inside and outside and seed_count >= 2:
            pools = [inside, outside]
        while len(pools) < seed_count:
            if inside and (not outside or random.random() < 0.5):
                pools.append(inside)
            elif outside:
                pools.append(outside)
            else:
                pools.append(allowed)
        for pool in pools:
            pos = _try_add_separate_position(positions, pool, overlap_offsets, contain_offsets)
            if pos is None:
                return None
            positions.append(pos)
    else:
        offsets = overlap_only_offsets if scenario == "overlap" else asymmetric_overlap_offsets
        if not offsets:
            return None

        offset = random.choice(offsets)
        anchors = list(allowed)
        random.shuffle(anchors)
        for anchor in anchors:
            other = (anchor[0] + offset[0], anchor[1] + offset[1])
            if other in allowed_set:
                positions.extend([anchor, other])
                break
        if len(positions) != 2:
            return None

        extra_count = seed_count - 2
        if all(pos in inside_set for pos in positions) and outside:
            preferred_pool = outside
        elif all(pos in outside_set for pos in positions) and inside:
            preferred_pool = inside
        else:
            preferred_pool = allowed

        for extra_idx in range(extra_count):
            pool = preferred_pool if extra_idx == 0 else allowed
            pos = _try_add_separate_position(positions, pool, overlap_offsets, contain_offsets)
            if pos is None:
                return None
            positions.append(pos)

    overlap_count, edges = _position_relations(positions, overlap_offsets, contain_offsets)
    if scenario == "separate" and overlap_count != 0:
        return None
    if scenario == "overlap" and not (overlap_count >= 1 and not edges):
        return None
    if scenario == "containment" and not (overlap_count >= 1 and edges and not _has_cycle(len(positions), edges)):
        return None
    return positions


def _generate_attempt() -> np.ndarray | None:
    scenario = random.choices(["separate", "overlap", "containment"], weights=[1, 1, 2])[0]
    small_fill = _make_small_fill(scenario)
    small_outline = _boundary(small_fill)
    small_interior = small_fill & ~small_outline
    if small_interior.sum() == 0:
        return None

    overlap_offsets, contain_offsets, asymmetric_overlap_offsets, overlap_only_offsets = _offset_sets(
        small_outline, small_interior
    )
    if scenario == "overlap" and not overlap_only_offsets:
        return None
    if scenario == "containment" and not asymmetric_overlap_offsets:
        return None

    large_fill = np.kron(small_fill, np.ones((3, 3), dtype=bool))
    large_outline = _boundary(large_fill)

    sh, sw = small_outline.shape
    center_r = sh // 2
    center_c = sw // 2
    top_margin = random.randint(center_r + 1, center_r + 5)
    bottom_margin = random.randint(center_r + 1, center_r + 5)
    left_margin = random.randint(center_c + 1, center_c + 5)
    right_margin = random.randint(center_c + 1, center_c + 5)

    h = top_margin + large_outline.shape[0] + bottom_margin
    w = left_margin + large_outline.shape[1] + right_margin
    if h > 36 or w > 36:
        return None

    grid = np.zeros((h, w), dtype=int)
    large_fill_abs = np.zeros((h, w), dtype=bool)
    large_outline_abs = np.zeros((h, w), dtype=bool)
    r0 = top_margin
    c0 = left_margin
    large_fill_abs[r0 : r0 + large_fill.shape[0], c0 : c0 + large_fill.shape[1]] = large_fill
    large_outline_abs[r0 : r0 + large_outline.shape[0], c0 : c0 + large_outline.shape[1]] = large_outline
    grid[large_outline_abs] = GEN_TEMPLATE_COLOR

    allowed, inside, outside = _classify_seed_centers(h, w, large_fill_abs, large_outline_abs, small_outline.shape)
    positions = _build_seed_positions(
        scenario,
        allowed,
        inside,
        outside,
        overlap_offsets,
        contain_offsets,
        asymmetric_overlap_offsets,
        overlap_only_offsets,
    )
    if positions is None:
        return None

    colors = random.sample(GEN_SEED_COLORS, len(positions))
    for (r, c), color in zip(positions, colors):
        grid[r, c] = color

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
