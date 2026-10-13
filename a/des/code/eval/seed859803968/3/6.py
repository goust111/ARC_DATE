from __future__ import annotations

import random
from collections import Counter

import numpy as np


Rect = tuple[int, int, int, int]
Side = str


def _dominant_color(grid: np.ndarray) -> int:
    values = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(values))


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(int), ((1, 0), (1, 0)), constant_values=0)
    return padded.cumsum(axis=0).cumsum(axis=1)


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    return int(
        prefix[r1 + 1, c1 + 1]
        - prefix[r0, c1 + 1]
        - prefix[r1 + 1, c0]
        + prefix[r0, c0]
    )


def _rect_area(rect: Rect) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _rect_height(rect: Rect) -> int:
    return rect[2] - rect[0] + 1


def _rect_width(rect: Rect) -> int:
    return rect[3] - rect[1] + 1


def _draw_rect(canvas: np.ndarray, rect: Rect, color: int) -> None:
    r0, c0, r1, c1 = rect
    canvas[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _contains_any(rect: Rect, coords: np.ndarray) -> bool:
    if coords.size == 0:
        return False
    r0, c0, r1, c1 = rect
    rows = coords[:, 0]
    cols = coords[:, 1]
    inside = (r0 <= rows) & (rows <= r1) & (c0 <= cols) & (cols <= c1)
    return bool(np.any(inside))


def _isolated_border_markers(grid: np.ndarray, bg: int) -> tuple[dict[int, Side], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    markers: dict[int, Side] = {}
    mask = np.zeros_like(g, dtype=bool)

    def side_of(r: int, c: int) -> Side | None:
        if r == 0 and 0 < c < w - 1:
            return "top"
        if r == h - 1 and 0 < c < w - 1:
            return "bottom"
        if c == 0 and 0 < r < h - 1:
            return "left"
        if c == w - 1 and 0 < r < h - 1:
            return "right"
        return None

    border_cells = []
    for c in range(w):
        border_cells.append((0, c))
        border_cells.append((h - 1, c))
    for r in range(1, h - 1):
        border_cells.append((r, 0))
        border_cells.append((r, w - 1))

    for r, c in border_cells:
        color = int(g[r, c])
        if color == bg:
            continue
        side = side_of(r, c)
        if side is None:
            continue

        neighbors = []
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < h and 0 <= cc < w:
                neighbors.append((rr, cc))
        if any(int(g[rr, cc]) != bg for rr, cc in neighbors):
            continue

        markers[color] = side
        mask[r, c] = True

    return markers, mask


def _candidate_rectangles(core: np.ndarray, color: int, bg: int, bg_prefix: np.ndarray) -> list[Rect]:
    coords = np.argwhere(core == int(color))
    if coords.size == 0:
        return []

    h, w = core.shape
    vr0 = int(coords[:, 0].min())
    vr1 = int(coords[:, 0].max())
    vc0 = int(coords[:, 1].min())
    vc1 = int(coords[:, 1].max())
    candidates: set[Rect] = set()

    def maybe_add(r0: int, c0: int, r1: int, c1: int) -> None:
        if r0 > vr0 or c0 > vc0 or r1 < vr1 or c1 < vc1:
            return
        if _rect_sum(bg_prefix, r0, c0, r1, c1) != 0:
            return
        candidates.add((r0, c0, r1, c1))

    for r1 in range(vr1, h):
        for c1 in range(vc1, w):
            maybe_add(vr0, vc0, r1, c1)

    for r1 in range(vr1, h):
        for c0 in range(vc0 + 1):
            maybe_add(vr0, c0, r1, vc1)

    for r0 in range(vr0 + 1):
        for c1 in range(vc1, w):
            maybe_add(r0, vc0, vr1, c1)

    for r0 in range(vr0 + 1):
        for c0 in range(vc0 + 1):
            maybe_add(r0, c0, vr1, vc1)

    if not candidates:
        candidates.add((vr0, vc0, vr1, vc1))

    return sorted(
        candidates,
        key=lambda rect: (
            _rect_area(rect),
            rect[0],
            rect[1],
            rect[2],
            rect[3],
        ),
    )


def _has_cycle(nodes: list[int], edges: set[tuple[int, int]]) -> bool:
    indegree = {node: 0 for node in nodes}
    succ = {node: [] for node in nodes}
    for lower, upper in edges:
        succ[lower].append(upper)
        indegree[upper] += 1

    stack = [node for node in nodes if indegree[node] == 0]
    seen = 0
    while stack:
        node = stack.pop()
        seen += 1
        for nxt in succ[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                stack.append(nxt)
    return seen != len(nodes)


def _topological_orders(nodes: list[int], edges: set[tuple[int, int]]) -> list[list[int]]:
    preds = {node: set() for node in nodes}
    succ = {node: set() for node in nodes}
    for lower, upper in edges:
        preds[upper].add(lower)
        succ[lower].add(upper)

    orders: list[list[int]] = []

    def dfs(order: list[int], remaining: set[int], cur_preds: dict[int, set[int]]) -> None:
        if not remaining:
            orders.append(order.copy())
            return
        available = sorted(node for node in remaining if not cur_preds[node])
        for node in available:
            next_remaining = set(remaining)
            next_remaining.remove(node)
            next_preds = {k: set(v) for k, v in cur_preds.items()}
            for nxt in succ[node]:
                next_preds[nxt].discard(node)
            dfs(order + [node], next_remaining, next_preds)

    dfs([], set(nodes), preds)
    return orders


def _simulate_stack(shape: tuple[int, int], bg: int, rectangles: dict[int, Rect], order: list[int]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for color in order:
        _draw_rect(out, rectangles[color], color)
    return out


def _reconstruct_rectangles(core: np.ndarray, colors: list[int], bg: int) -> tuple[dict[int, Rect], list[int]]:
    bg_prefix = _prefix_sum(core == int(bg))
    visible = {color: np.argwhere(core == int(color)) for color in colors}
    candidates = {
        color: _candidate_rectangles(core, color, bg, bg_prefix)
        for color in colors
    }
    contains = {
        color: [
            {
                other: _contains_any(rect, visible[other])
                for other in colors
                if other != color
            }
            for rect in candidates[color]
        ]
        for color in colors
    }

    search_order = sorted(colors, key=lambda color: (len(candidates[color]), -len(visible[color])))
    chosen_idx: dict[int, int] = {}
    chosen_nodes: list[int] = []

    def dfs(pos: int, edges: set[tuple[int, int]]) -> tuple[dict[int, Rect], list[int]] | None:
        if pos == len(search_order):
            rectangles = {
                color: candidates[color][chosen_idx[color]]
                for color in colors
            }
            for order in _topological_orders(colors, edges):
                if np.array_equal(_simulate_stack(core.shape, bg, rectangles, order), core):
                    return rectangles, order
            return None

        color = search_order[pos]
        for idx, rect in enumerate(candidates[color]):
            new_edges = set(edges)
            ok = True
            for other in chosen_nodes:
                other_idx = chosen_idx[other]
                color_below_other = contains[color][idx].get(other, False)
                other_below_color = contains[other][other_idx].get(color, False)
                if color_below_other and other_below_color:
                    ok = False
                    break
                if color_below_other:
                    new_edges.add((color, other))
                if other_below_color:
                    new_edges.add((other, color))
            if not ok:
                continue

            trial_nodes = chosen_nodes + [color]
            if _has_cycle(trial_nodes, new_edges):
                continue

            chosen_idx[color] = idx
            chosen_nodes.append(color)
            result = dfs(pos + 1, new_edges)
            chosen_nodes.pop()
            chosen_idx.pop(color, None)
            if result is not None:
                return result

        return None

    result = dfs(0, set())
    if result is not None:
        return result

    fallback_rects = {}
    for color in colors:
        coords = visible[color]
        r0 = int(coords[:, 0].min())
        r1 = int(coords[:, 0].max())
        c0 = int(coords[:, 1].min())
        c1 = int(coords[:, 1].max())
        fallback_rects[color] = (r0, c0, r1, c1)

    fallback_edges: set[tuple[int, int]] = set()
    for a in colors:
        for b in colors:
            if a == b:
                continue
            if _contains_any(fallback_rects[a], visible[b]):
                fallback_edges.add((a, b))
    orders = _topological_orders(colors, fallback_edges)
    if orders:
        return fallback_rects, orders[0]
    return fallback_rects, colors[:]


def _translate_rect(rect: Rect, side: Side, shape: tuple[int, int]) -> Rect:
    h, w = shape
    r0, c0, r1, c1 = rect
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if side == "top":
        return (0, c0, height - 1, c1)
    if side == "bottom":
        return (h - height, c0, h - 1, c1)
    if side == "left":
        return (r0, 0, r1, width - 1)
    if side == "right":
        return (r0, w - width, r1, w - 1)
    raise ValueError(side)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct overlapped solid rectangles, infer stack order from occlusion,
    then slide each rectangle to the edge indicated by its same-color border marker.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    markers, marker_mask = _isolated_border_markers(g, bg)
    core = g.copy()
    core[marker_mask] = bg

    colors = sorted(color for color in markers if np.any(core == color))
    if not colors:
        return np.full_like(g, bg)

    rectangles, order = _reconstruct_rectangles(core, colors, bg)
    out = np.full_like(g, bg)
    for color in order:
        moved = _translate_rect(rectangles[color], markers[color], g.shape)
        _draw_rect(out, moved, color)
    return out


# --------------- generator ---------------

_SIDES = ("top", "bottom", "left", "right")
_EXPOSED_CORNERS = ("tl", "tr", "bl", "br")
_GEN_RETRIES = 400


def _flip_rect(rect: Rect, shape: tuple[int, int], vertical: bool, horizontal: bool) -> Rect:
    h, w = shape
    r0, c0, r1, c1 = rect
    if vertical:
        r0, r1 = h - 1 - r1, h - 1 - r0
    if horizontal:
        c0, c1 = w - 1 - c1, w - 1 - c0
    return (r0, c0, r1, c1)


def _sample_spaced_positions(limit: int, count: int) -> list[int] | None:
    values = list(range(1, limit - 1))
    random.shuffle(values)
    picked: list[int] = []
    for value in values:
        if all(abs(value - prev) >= 2 for prev in picked):
            picked.append(value)
            if len(picked) == count:
                return picked
    return None


def _build_nested_rectangles(n_rects: int, shape: tuple[int, int]) -> tuple[list[Rect], str] | None:
    h, w = shape
    exposed = random.choice(_EXPOSED_CORNERS)

    grow_steps = []
    grow_up = grow_left = grow_down = grow_right = 0
    for _ in range(n_rects - 1):
        step_up = random.randint(1, 2)
        step_left = random.randint(1, 2)
        step_down = random.randint(0, 1)
        step_right = random.randint(0, 1)
        grow_steps.append((step_up, step_left, step_down, step_right))
        grow_up += step_up
        grow_left += step_left
        grow_down += step_down
        grow_right += step_right

    top_h = random.randint(3, 6)
    top_w = random.randint(3, 6)
    min_h = top_h + grow_up + grow_down + 2
    min_w = top_w + grow_left + grow_right + 2
    if min_h > h or min_w > w:
        return None

    top_r0_lo = 1 + grow_up
    top_r0_hi = h - 2 - grow_down - top_h + 1
    top_c0_lo = 1 + grow_left
    top_c0_hi = w - 2 - grow_right - top_w + 1
    if top_r0_lo > top_r0_hi or top_c0_lo > top_c0_hi:
        return None

    top_r0 = random.randint(top_r0_lo, top_r0_hi)
    top_c0 = random.randint(top_c0_lo, top_c0_hi)
    current = (top_r0, top_c0, top_r0 + top_h - 1, top_c0 + top_w - 1)
    top_to_bottom = [current]
    for step_up, step_left, step_down, step_right in grow_steps:
        current = (
            current[0] - step_up,
            current[1] - step_left,
            current[2] + step_down,
            current[3] + step_right,
        )
        top_to_bottom.append(current)

    bottom_to_top = list(reversed(top_to_bottom))
    vertical = exposed in ("bl", "br")
    horizontal = exposed in ("tr", "br")
    transformed = [
        _flip_rect(rect, shape, vertical=vertical, horizontal=horizontal)
        for rect in bottom_to_top
    ]
    return transformed, exposed


def _render_input(shape: tuple[int, int], bg: int, rectangles: dict[int, Rect], order: list[int], markers: dict[int, Side]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), int(bg), dtype=int)
    for color in order:
        _draw_rect(grid, rectangles[color], color)

    for side in _SIDES:
        colors = [color for color in order if markers[color] == side]
        if not colors:
            continue
        positions = _sample_spaced_positions(w if side in ("top", "bottom") else h, len(colors))
        if positions is None:
            raise ValueError("not enough border positions for markers")
        for color, pos in zip(colors, positions):
            if side == "top":
                grid[0, pos] = color
            elif side == "bottom":
                grid[h - 1, pos] = color
            elif side == "left":
                grid[pos, 0] = color
            else:
                grid[pos, w - 1] = color
    return grid


def _render_output(shape: tuple[int, int], bg: int, rectangles: dict[int, Rect], order: list[int], markers: dict[int, Side]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), int(bg), dtype=int)
    for color in order:
        moved = _translate_rect(rectangles[color], markers[color], (h, w))
        _draw_rect(out, moved, color)
    return out


def _visible_edges_ok(inp: np.ndarray, rectangles: dict[int, Rect], order: list[int], corner: str) -> bool:
    visible_sides = {
        "tl": ("top", "left"),
        "tr": ("top", "right"),
        "bl": ("bottom", "left"),
        "br": ("bottom", "right"),
    }[corner]

    for color in order[:-1]:
        r0, c0, r1, c1 = rectangles[color]
        if visible_sides[0] == "top" and not np.any(inp[r0, c0 : c1 + 1] == color):
            return False
        if visible_sides[0] == "bottom" and not np.any(inp[r1, c0 : c1 + 1] == color):
            return False
        if visible_sides[1] == "left" and not np.any(inp[r0 : r1 + 1, c0] == color):
            return False
        if visible_sides[1] == "right" and not np.any(inp[r0 : r1 + 1, c1] == color):
            return False
    return True


def _sample_case_once() -> tuple[np.ndarray, np.ndarray]:
    bg = 0
    n_rects = random.randint(2, 4)
    colors = random.sample(list(range(1, 10)), n_rects)
    order = colors[:]

    min_h = 12
    min_w = 12
    max_h = 20
    max_w = 20
    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)

    built = _build_nested_rectangles(n_rects, (h, w))
    if built is None:
        raise ValueError("failed to place nested rectangles")
    rect_list, corner = built
    rectangles = {color: rect for color, rect in zip(order, rect_list)}

    markers = {color: random.choice(_SIDES) for color in order}
    inp = _render_input((h, w), bg, rectangles, order, markers)
    out = _render_output((h, w), bg, rectangles, order, markers)

    if np.array_equal(inp, out):
        raise ValueError("movement must change the scene")
    if not _visible_edges_ok(inp, rectangles, order, corner):
        raise ValueError("occluded rectangles must keep two adjacent visible edges")
    if not np.array_equal(main(inp), out):
        raise ValueError("solver mismatch on generated case")
    return inp, out


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(_GEN_RETRIES):
        try:
            return _sample_case_once()
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input/output pair")


def generate_input() -> np.ndarray:
    inp, _out = _generate_case()
    return inp
