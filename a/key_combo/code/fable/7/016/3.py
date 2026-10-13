"""ARC task: rotate sprite center colors along a self-keyed conveyor path."""

from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
PATH = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
N8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
)
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _in(shape: tuple[int, int], p: tuple[int, int]) -> bool:
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _path_order(g: np.ndarray) -> list[tuple[int, int]]:
    """Return the gray simple path in either endpoint order."""
    cells = {tuple(map(int, p)) for p in np.argwhere(g == PATH)}
    assert len(cells) >= 3, "expected a gray path"
    graph = {
        p: [q for dr, dc in N8 if (q := (p[0] + dr, p[1] + dc)) in cells]
        for p in cells
    }
    endpoints = [p for p, ns in graph.items() if len(ns) == 1]
    assert len(endpoints) == 2, "gray path must have exactly two endpoints"
    assert all(len(ns) in (1, 2) for ns in graph.values()), "gray path must not branch"

    order = [endpoints[0]]
    previous = None
    while len(order) < len(cells):
        nxt = [q for q in graph[order[-1]] if q != previous]
        assert len(nxt) == 1, "gray path must be one unambiguous stroke"
        previous, current = order[-1], nxt[0]
        assert current not in order, "gray path must not contain a cycle"
        order.append(current)
    return order


def _border_component(g: np.ndarray, seed: tuple[int, int], color: int) -> set[tuple[int, int]]:
    seen = {seed}
    todo = deque([seed])
    while todo:
        r, c = todo.popleft()
        for dr, dc in N4:
            q = (r + dr, c + dc)
            if _in(g.shape, q) and q not in seen and int(g[q]) == color:
                seen.add(q)
                todo.append(q)
    return seen


def _sprites(g: np.ndarray, bg: int) -> list[dict]:
    """Find 3x3 framed sprites and include their same-color port cells."""
    found: list[dict] = []
    used_centers: set[tuple[int, int]] = set()
    h, w = g.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center = (r, c)
            ring = {
                (r + dr, c + dc)
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0)
            }
            colors = {int(g[p]) for p in ring}
            if len(colors) != 1:
                continue
            border = colors.pop()
            if border in (bg, PATH) or center in used_centers:
                continue
            component = _border_component(g, next(iter(ring)), border)
            if not ring <= component:
                continue
            found.append(
                {
                    "center": center,
                    "border": border,
                    "center_color": int(g[center]),
                    "border_cells": component,
                }
            )
            used_centers.add(center)
    assert found, "expected framed sprites"
    borders = [s["border"] for s in found]
    assert len(borders) == len(set(borders)), "sprite border colors must be unique"
    return found


def _contact_index(sprite: dict, path_index: dict[tuple[int, int], int]) -> int | None:
    touched = set()
    for r, c in sprite["border_cells"]:
        for dr, dc in N4:
            q = (r + dr, c + dc)
            if q in path_index:
                touched.add(path_index[q])
    if not touched:
        return None
    # A one-pixel port has exactly one 4-neighbor on the generated one-cell path.
    assert len(touched) == 1, "each port must touch one path position"
    return touched.pop()


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate center colors one step forward along a gray conveyor and erase unconnected sprites.

    Input structure:
    - A dominant background contains one gray 8-connected, non-branching path with exactly two endpoints.
    - Distinct-color sprites are 3x3 monochrome frames with one center pixel and a same-frame-color port.
    - Some ports touch the path by a 4-edge; exactly one touching sprite has center color equal to its frame color,
      and its port touches a path endpoint. Other framed sprites do not touch the path and are distractors.

    Rule:
    - Start at the endpoint touched by the unique self-colored sprite and traverse the gray path to its other endpoint.
    - Sort touching sprites by their port positions on that traversal. Give each next sprite the preceding sprite's
      original center color, and give the first sprite the last sprite's original center color.
    - Preserve the path and every touching frame, and erase every non-touching sprite completely to background.

    Key insight:
    - The self-colored sprite is both the unique anchor and the direction key; its path-relative visit order defines
      one cyclic shift of center colors, while visually identical but disconnected sprites are irrelevant.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = Counter(map(int, g.ravel())).most_common(1)[0][0]
    order = _path_order(g)
    sprites = _sprites(g, bg)
    self_locked = [s for s in sprites if s["center_color"] == s["border"]]
    assert len(self_locked) == 1, "expected one self-colored sprite"

    def contacts(path: list[tuple[int, int]]) -> list[tuple[int, dict]]:
        index = {p: i for i, p in enumerate(path)}
        return [(i, s) for s in sprites if (i := _contact_index(s, index)) is not None]

    contacted = contacts(order)
    locked_item = next(((i, s) for i, s in contacted if s is self_locked[0]), None)
    assert locked_item is not None, "self-colored sprite must touch the path"
    if locked_item[0] == len(order) - 1:
        order.reverse()
        contacted = contacts(order)
        locked_item = next(x for x in contacted if x[1] is self_locked[0])
    assert locked_item[0] == 0, "self-colored sprite must touch an endpoint"
    contacted.sort(key=lambda x: x[0])
    active = [s for _, s in contacted]
    assert len(active) >= 3, "expected at least three path-connected sprites"
    assert len({i for i, _ in contacted}) == len(contacted), "one sprite per path port is required"

    out = g.copy()
    active_ids = {id(s) for s in active}
    for s in sprites:
        if id(s) not in active_ids:
            for p in s["border_cells"]:
                out[p] = bg
            out[s["center"]] = bg

    old = [s["center_color"] for s in active]
    for i, s in enumerate(active):
        out[s["center"]] = old[i - 1]
    return out


def _sprite_cells(center: tuple[int, int], port_dir: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    cells = {
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr, dc) != (0, 0)
    }
    cells.add((r + 2 * port_dir[0], c + 2 * port_dir[1]))
    cells.add(center)
    return cells


def _paint_sprite(
    g: np.ndarray,
    center: tuple[int, int],
    port_dir: tuple[int, int],
    border: int,
    center_color: int,
) -> set[tuple[int, int]]:
    r, c = center
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr, dc) != (0, 0):
                g[r + dr, c + dc] = border
    g[r + 2 * port_dir[0], c + 2 * port_dir[1]] = border
    g[r, c] = center_color
    return _sprite_cells(center, port_dir)


def _can_place(shape: tuple[int, int], cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    if not all(_in(shape, p) for p in cells):
        return False
    # A one-cell moat prevents two sprites from becoming one visual object.
    halo = {
        (r + dr, c + dc)
        for r, c in cells
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
    }
    return not (halo & occupied)


def _expected_from_metadata(
    inp: np.ndarray,
    active: list[tuple[tuple[int, int], int]],
    distractors: list[set[tuple[int, int]]],
    bg: int,
) -> np.ndarray:
    out = inp.copy()
    old = [color for _, color in active]
    for i, (center, _) in enumerate(active):
        out[center] = old[i - 1]
    for cells in distractors:
        for p in cells:
            out[p] = bg
    return out


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(22, 30), random.randint(22, 30)
    horizontal = random.choice((True, False))
    length_axis = w if horizontal else h
    k_max = min(5, 2 + (length_axis - 12) // 4)
    if k_max < 3:
        return None
    k = random.randint(3, k_max)
    d = random.randint(1, min(2, 8 - k))
    bg = random.choice((0, *PALETTE))
    usable_colors = [c for c in PALETTE if c != bg]
    colors = random.sample(usable_colors, k + d)
    g = np.full((h, w), bg, dtype=int)

    if horizontal:
        fixed = random.randint(6, h - 7)
        a0, a1 = 5, w - 6
        path = [(fixed, x) for x in range(a0, a1 + 1)]
    else:
        fixed = random.randint(6, w - 7)
        a0, a1 = 5, h - 6
        path = [(x, fixed) for x in range(a0, a1 + 1)]
    for p in path:
        g[p] = PATH
    path_set = set(path)

    reverse = random.choice((False, True))
    start_axis = a1 if reverse else a0
    if horizontal:
        self_center = (fixed, start_axis + (3 if reverse else -3))
        self_port = (0, -1 if reverse else 1)
    else:
        self_center = (start_axis + (3 if reverse else -3), fixed)
        self_port = (-1 if reverse else 1, 0)

    occupied = set(path)
    active_centers = [self_center]
    active_dirs = [self_port]
    occupied |= _sprite_cells(self_center, self_port)

    candidates = list(range(a0 + 3, a1 - 2))
    random.shuffle(candidates)
    chosen: list[int] = []
    for x in candidates:
        if all(abs(x - y) >= 4 for y in chosen):
            chosen.append(x)
            if len(chosen) == k - 1:
                break
    if len(chosen) != k - 1:
        return None
    chosen.sort(reverse=reverse)
    for x in chosen:
        side = random.choice((-1, 1))
        if horizontal:
            center, port_dir = (fixed + 3 * side, x), (-side, 0)
        else:
            center, port_dir = (x, fixed + 3 * side), (0, -side)
        cells = _sprite_cells(center, port_dir)
        if (cells & path_set) or not _can_place(g.shape, cells, occupied - path_set):
            return None
        active_centers.append(center)
        active_dirs.append(port_dir)
        occupied |= cells

    # Make all active center colors visibly distinct, and only the first self-colored.
    center_pool = list(usable_colors)
    random.shuffle(center_pool)
    active_center_colors = [colors[0]]
    for border in colors[1:k]:
        choices = [c for c in center_pool if c != border and c not in active_center_colors]
        if not choices:
            return None
        active_center_colors.append(random.choice(choices))

    active_meta = []
    for center, direction, border, center_color in zip(
        active_centers, active_dirs, colors[:k], active_center_colors
    ):
        _paint_sprite(g, center, direction, border, center_color)
        active_meta.append((center, center_color))

    distractor_sets: list[set[tuple[int, int]]] = []
    for j in range(d):
        placed = False
        for _ in range(300):
            center = (random.randint(2, h - 3), random.randint(2, w - 3))
            direction = random.choice(N4)
            cells = _sprite_cells(center, direction)
            if not _can_place(g.shape, cells, occupied):
                continue
            # Distractor ports must have no 4-edge contact with the path.
            if any((r + dr, c + dc) in path_set for r, c in cells for dr, dc in N4):
                continue
            border = colors[k + j]
            center_choices = [c for c in usable_colors if c != border and c != colors[0]]
            center_color = random.choice(center_choices)
            _paint_sprite(g, center, direction, border, center_color)
            distractor_sets.append(cells)
            occupied |= cells
            placed = True
            break
        if not placed:
            return None

    expected = _expected_from_metadata(g, active_meta, distractor_sets, bg)
    try:
        actual = main(g)
    except (AssertionError, StopIteration):
        return None
    if not np.array_equal(actual, expected):
        return None
    # Semantic/non-degeneracy checks: all three essential operations affect output.
    if np.array_equal(actual, g) or any(actual[p] != PATH for p in path):
        return None
    if not all(actual[c] != old for c, old in active_meta):
        return None
    if not distractor_sets or not all(actual[p] == bg for cells in distractor_sets for p in cells):
        return None
    # A pairwise-swap shortcut and the opposite cyclic direction must both disagree.
    pairwise = [x[1] for x in active_meta]
    for i in range(0, len(pairwise) - 1, 2):
        pairwise[i], pairwise[i + 1] = pairwise[i + 1], pairwise[i]
    opposite = [active_meta[(i + 1) % k][1] for i in range(k)]
    solved = [int(actual[c]) for c, _ in active_meta]
    if solved == pairwise or solved == opposite:
        return None
    return g


def generate_input() -> np.ndarray:
    """Generate diverse, independently oracle-checked conveyor instances."""
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid conveyor instance")
