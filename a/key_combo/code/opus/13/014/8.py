from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
INPUT_FRAME = 5
DEPTH_COLORS = (2, 1, 3, 4)
MIN_SIDE = 4
MAX_FRAME_COUNT = 4
MIN_FRAME_COUNT = 3
GEN_MAX_ATTEMPTS = 500


def _rect_corners(rect: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = rect
    return ((r0, c0), (r0, c1), (r1, c0), (r1, c1))


def _rect_dims(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1, c1 - c0 + 1)


def _rect_border_coords(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    coords = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    return coords


def _point_on_strict_edge(point: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    r, c = point
    r0, c0, r1, c1 = rect
    on_top_or_bottom = r in (r0, r1) and c0 < c < c1
    on_left_or_right = c in (c0, c1) and r0 < r < r1
    return on_top_or_bottom or on_left_or_right


def _strict_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    ro0, co0, ro1, co1 = outer
    ri0, ci0, ri1, ci1 = inner
    return ro0 < ri0 and co0 < ci0 and ri1 < ro1 and ci1 < co1


def _all_border_present(mask: np.ndarray, rect: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = rect
    if r1 - r0 + 1 < MIN_SIDE or c1 - c0 + 1 < MIN_SIDE:
        return False
    if not (mask[r0, c0] and mask[r0, c1] and mask[r1, c0] and mask[r1, c1]):
        return False
    if not bool(np.all(mask[r0, c0 : c1 + 1])):
        return False
    if not bool(np.all(mask[r1, c0 : c1 + 1])):
        return False
    if not bool(np.all(mask[r0 : r1 + 1, c0])):
        return False
    if not bool(np.all(mask[r0 : r1 + 1, c1])):
        return False
    return True


def _enumerate_candidates(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    rows = sorted(set(int(r) for r in np.flatnonzero(mask.any(axis=1))))
    cols = sorted(set(int(c) for c in np.flatnonzero(mask.any(axis=0))))
    candidates = []
    for i, r0 in enumerate(rows):
        for r1 in rows[i + 1 :]:
            if r1 - r0 + 1 < MIN_SIDE:
                continue
            for j, c0 in enumerate(cols):
                for c1 in cols[j + 1 :]:
                    rect = (r0, c0, r1, c1)
                    if _all_border_present(mask, rect):
                        candidates.append(rect)
    candidates.sort(key=lambda rect: (-(2 * sum(_rect_dims(rect)) - 4), rect))
    return candidates


def _depth_order(rects: list[tuple[int, int, int, int]]) -> tuple[list[int], int] | None:
    n = len(rects)
    edges = [set() for _ in range(n)]
    for front_idx, front in enumerate(rects):
        for back_idx, back in enumerate(rects):
            if front_idx == back_idx:
                continue
            if any(_point_on_strict_edge(corner, front) for corner in _rect_corners(back)):
                edges[front_idx].add(back_idx)
    for i in range(n):
        for j in edges[i]:
            if i in edges[j]:
                return None

    indeg = [0] * n
    for src in range(n):
        for dst in edges[src]:
            indeg[dst] += 1

    order = []
    ready = [idx for idx, deg in enumerate(indeg) if deg == 0]
    edge_count = sum(len(out) for out in edges)
    while ready:
        if len(ready) != 1:
            return None
        node = ready.pop()
        order.append(node)
        for dst in sorted(edges[node]):
            indeg[dst] -= 1
            if indeg[dst] == 0:
                ready.append(dst)

    if len(order) != n:
        return None
    return order, edge_count


def _rect_bits(
    rect: tuple[int, int, int, int],
    bit_index: dict[tuple[int, int], int],
) -> int:
    bits = 0
    for coord in _rect_border_coords(rect):
        bits |= 1 << bit_index[coord]
    return bits


def _visible_corner_count(rect: tuple[int, int, int, int], others: list[tuple[int, int, int, int]]) -> int:
    count = 0
    for corner in _rect_corners(rect):
        if not any(_point_on_strict_edge(corner, other) for other in others):
            count += 1
    return count


def _solution_score(rects: list[tuple[int, int, int, int]], bits: list[int]) -> tuple[int, int, int]:
    topo = _depth_order(rects)
    assert topo is not None
    _order, edge_count = topo
    unique_pixels = 0
    for i, bitset in enumerate(bits):
        others = 0
        for j, other in enumerate(bits):
            if i != j:
                others |= other
        unique_pixels += (bitset & ~others).bit_count()
    visible_corners = sum(
        _visible_corner_count(rect, [other for other in rects if other != rect])
        for rect in rects
    )
    return (edge_count, unique_pixels, visible_corners)


def _valid_solution(rects: list[tuple[int, int, int, int]], bits: list[int]) -> bool:
    topo = _depth_order(rects)
    if topo is None:
        return False
    order, edge_count = topo
    if len(order) != len(rects) or edge_count < len(rects) - 1:
        return False

    dims = [_rect_dims(rect) for rect in rects]
    if len(set(dims)) != len(dims):
        return False

    for i, rect in enumerate(rects):
        if _visible_corner_count(rect, [other for j, other in enumerate(rects) if j != i]) < 1:
            return False
        others = 0
        for j, other_bits in enumerate(bits):
            if i != j:
                others |= other_bits
        if (bits[i] & ~others) == 0:
            return False

    for i, rect_a in enumerate(rects):
        for j, rect_b in enumerate(rects):
            if i != j and (_strict_contains(rect_a, rect_b) or _strict_contains(rect_b, rect_a)):
                return False
    return True


def _decompose_frames(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    points = [tuple(map(int, p)) for p in np.argwhere(mask)]
    assert points, "expected at least one frame pixel"
    bit_index = {coord: idx for idx, coord in enumerate(points)}
    all_bits = (1 << len(points)) - 1

    candidate_rects = _enumerate_candidates(mask)
    assert candidate_rects, "expected at least one rectangle candidate"

    candidate_bits = [_rect_bits(rect, bit_index) for rect in candidate_rects]
    dims = [_rect_dims(rect) for rect in candidate_rects]
    perims = [len(_rect_border_coords(rect)) for rect in candidate_rects]

    by_pixel: list[list[int]] = [[] for _ in range(len(points))]
    for idx, bits in enumerate(candidate_bits):
        for bit_pos in range(len(points)):
            if bits & (1 << bit_pos):
                by_pixel[bit_pos].append(idx)

    best_rects = None
    best_score = None

    def search(selected: list[int], covered: int, used_dims: set[tuple[int, int]]) -> None:
        nonlocal best_rects, best_score

        if len(selected) > MAX_FRAME_COUNT:
            return

        feasible = [
            idx
            for idx, dim in enumerate(dims)
            if dim not in used_dims and idx not in selected
        ]
        reachable = covered
        for idx in feasible:
            reachable |= candidate_bits[idx]
        if reachable != all_bits:
            return

        if covered == all_bits:
            if MIN_FRAME_COUNT <= len(selected) <= MAX_FRAME_COUNT:
                rects = [candidate_rects[idx] for idx in selected]
                bits = [candidate_bits[idx] for idx in selected]
                if _valid_solution(rects, bits):
                    score = _solution_score(rects, bits)
                    if best_score is None or score > best_score:
                        best_rects = rects
                        best_score = score
            return

        if len(selected) == MAX_FRAME_COUNT:
            return

        uncovered = all_bits & ~covered
        bit_pos = None
        choices = None
        for pos in range(len(points)):
            if not (uncovered & (1 << pos)):
                continue
            options = [
                idx
                for idx in by_pixel[pos]
                if idx not in selected
                and dims[idx] not in used_dims
                and (candidate_bits[idx] & uncovered) != 0
            ]
            if not options:
                return
            if choices is None or len(options) < len(choices):
                bit_pos = pos
                choices = options
                if len(choices) == 1:
                    break

        assert bit_pos is not None and choices is not None
        choices.sort(
            key=lambda idx: (
                -((candidate_bits[idx] & uncovered).bit_count()),
                -perims[idx],
                candidate_rects[idx],
            )
        )
        for idx in choices:
            search(selected + [idx], covered | candidate_bits[idx], used_dims | {dims[idx]})

    search([], 0, set())
    assert best_rects is not None, "failed to decompose frames"
    return best_rects


def _paint_rect(out: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    for r, c in _rect_border_coords(rect):
        out[r, c] = int(color)


def _render_output(shape: tuple[int, int], rects_front_to_back: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    for depth_idx in range(len(rects_front_to_back) - 1, -1, -1):
        _paint_rect(out, rects_front_to_back[depth_idx], DEPTH_COLORS[depth_idx])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Decompose overlapping same-color hollow rectangles, order them by T-junction depth, and recolor by depth.

    Input structure:
    - The grid has black background and 3-4 gray hollow axis-aligned rectangular frames.
    - Every frame is a 1-pixel-thick border, all input frame pixels share the same gray color, and different frames may overlap on edge pixels.
    - The frame size pairs are unique, no frame strictly contains another, and some corners land on other frames' edges to form T-like contacts.

    Rule:
    - Enumerate every axis-aligned rectangle whose full border is present in the gray pixel set, then choose the 3-4 unique-size frames whose border union exactly matches the input.
    - Infer depth from T-junctions: if one frame has a corner lying on another frame's non-corner edge, the crossed frame is in front and the corner-owning frame is behind.
    - Topologically order the frames from front to back, recolor depth 1/2/3/4 as red/blue/green/yellow, and assign each overlapping border pixel to the frontmost frame covering it.

    Key insight:
    - The input color carries no identity information, so the task is to reconstruct hidden frame instances from shared border pixels and use corner-on-edge contacts to recover a unique depth order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mask = g == INPUT_FRAME
    rects = _decompose_frames(mask)
    topo = _depth_order(rects)
    assert topo is not None, "expected a unique depth order"
    order, _edge_count = topo
    ordered_rects = [rects[idx] for idx in order]
    return _render_output(g.shape, ordered_rects)


def _in_bounds_rect(rect: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = rect
    h, w = shape
    return 0 <= r0 < r1 < h and 0 <= c0 < c1 < w


def _choose_dims(used: set[tuple[int, int]]) -> tuple[int, int]:
    options = [
        (h, w)
        for h in range(6, 9)
        for w in range(6, 9)
        if (h, w) not in used
    ]
    return random.choice(options)


def _choose_child_dims(
    used: set[tuple[int, int]],
    parent: tuple[int, int, int, int],
    side: str,
) -> tuple[int, int] | None:
    parent_h, parent_w = _rect_dims(parent)
    options = []
    for height in range(4, 9):
        for width in range(4, 9):
            if (height, width) in used:
                continue
            if side in ("top", "bottom") and width <= parent_w - 2:
                options.append((height, width))
            if side in ("left", "right") and height <= parent_h - 2:
                options.append((height, width))
    if not options:
        return None
    return random.choice(options)


def _attach_inside_edge(
    parent: tuple[int, int, int, int],
    side: str,
    height: int,
    width: int,
) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = parent
    if side == "bottom":
        if width > (c1 - c0 - 1):
            return None
        left = random.randint(c0 + 1, c1 - width)
        return (r1, left, r1 + height - 1, left + width - 1)
    if side == "top":
        if width > (c1 - c0 - 1):
            return None
        left = random.randint(c0 + 1, c1 - width)
        return (r0 - height + 1, left, r0, left + width - 1)
    if side == "right":
        if height > (r1 - r0 - 1):
            return None
        top = random.randint(r0 + 1, r1 - height)
        return (top, c1, top + height - 1, c1 + width - 1)
    if height > (r1 - r0 - 1):
        return None
    top = random.randint(r0 + 1, r1 - height)
    return (top, c0 - width + 1, top + height - 1, c0)


def _border_set(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    return set(_rect_border_coords(rect))


def _attempt_rectangles() -> list[tuple[int, int, int, int]] | None:
    frame_count = 4 if random.random() < 0.55 else 3

    used_dims: set[tuple[int, int]] = set()
    base_h, base_w = _choose_dims(used_dims)
    used_dims.add((base_h, base_w))
    rects = [(0, 0, base_h - 1, base_w - 1)]

    for _depth in range(1, frame_count):
        parent = rects[-1]
        placed = False
        choices = ["top", "bottom", "left", "right"]
        random.shuffle(choices)
        for side in choices:
            for _ in range(24):
                dims = _choose_child_dims(used_dims, parent, side)
                if dims is None:
                    break
                height, width = dims
                rect = _attach_inside_edge(parent, side, height, width)
                if rect is None:
                    continue
                if any(_strict_contains(rect, old) or _strict_contains(old, rect) for old in rects):
                    continue

                rect_border = _border_set(rect)
                ok = True
                for old_idx, old in enumerate(rects):
                    old_border = _border_set(old)
                    inter = rect_border & old_border
                    if old_idx == len(rects) - 1:
                        shared_len = width if side in ("top", "bottom") else height
                        if len(inter) != shared_len:
                            ok = False
                            break
                        if sum(_point_on_strict_edge(corner, old) for corner in _rect_corners(rect)) < 2:
                            ok = False
                            break
                        if any(_point_on_strict_edge(corner, rect) for corner in _rect_corners(old)):
                            ok = False
                            break
                    elif inter:
                        ok = False
                        break
                if not ok:
                    continue

                rects.append(rect)
                used_dims.add((height, width))
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    topo = _depth_order(rects)
    if topo is None:
        return None
    order, edge_count = topo
    if order != list(range(len(rects))) or edge_count < len(rects) - 1:
        return None
    if any(
        _visible_corner_count(rect, [other for other in rects if other != rect]) < 1
        for rect in rects
    ):
        return None

    min_r = min(rect[0] for rect in rects)
    min_c = min(rect[1] for rect in rects)
    max_r = max(rect[2] for rect in rects)
    max_c = max(rect[3] for rect in rects)
    union_h = max_r - min_r + 1
    union_w = max_c - min_c + 1
    if union_h > 18 or union_w > 18:
        return None

    rows = random.randint(max(union_h + 2, 12), min(union_h + 6, 20))
    cols = random.randint(max(union_w + 2, 12), min(union_w + 6, 20))
    shift_r = random.randint(1 - min_r, rows - 2 - max_r)
    shift_c = random.randint(1 - min_c, cols - 2 - max_c)
    shifted = [
        (r0 + shift_r, c0 + shift_c, r1 + shift_r, c1 + shift_c)
        for r0, c0, r1, c1 in rects
    ]
    if not all(_in_bounds_rect(rect, (rows, cols)) for rect in shifted):
        return None
    return shifted


def _render_input(shape: tuple[int, int], rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for rect in rects:
        _paint_rect(grid, rect, INPUT_FRAME)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        rects = _attempt_rectangles()
        if rects is None:
            continue
        rmax = max(rect[2] for rect in rects) + 1
        cmax = max(rect[3] for rect in rects) + 1
        shape = (rmax, cmax)
        grid = _render_input(shape, rects)
        if max(shape) > 30:
            continue
        try:
            recovered = _decompose_frames(grid == INPUT_FRAME)
            if set(recovered) != set(rects):
                continue
            expected = _render_output(shape, rects)
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid overlapping-frame input")
