from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]

BACKGROUND = 0
TRACK = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(coords)
    return parts


def _trace_cycle(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = {tuple(p) for p in coords}
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in cells:
        adjacent = []
        for dr, dc in DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in cells:
                adjacent.append(nxt)
        assert len(adjacent) == 2, "each track cell must have degree 2"
        neighbors[(r, c)] = adjacent

    start = min(cells)
    prev: tuple[int, int] | None = None
    cur = start
    order = [start]
    while True:
        a, b = neighbors[cur]
        nxt = a if a != prev else b
        if nxt == start:
            break
        order.append(nxt)
        prev, cur = cur, nxt
        assert len(order) <= len(cells), "cycle traversal exceeded component size"
    assert len(order) == len(cells), "cycle traversal must visit every track cell once"
    return order


def _signed_area(path: list[tuple[int, int]]) -> int:
    area = 0
    wrapped = path[1:] + path[:1]
    for (r1, c1), (r2, c2) in zip(path, wrapped):
        area += c1 * r2 - c2 * r1
    return int(area)


def _clockwise_cycle(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    order = _trace_cycle(coords)
    area = _signed_area(order)
    assert area != 0, "cycle area must be non-zero"
    if area < 0:
        order = order[::-1]
    return order


def _infer_track_and_passenger(
    grid: np.ndarray, bg: int, components: list[list[tuple[int, int]]]
) -> tuple[int, int | None]:
    presence = Counter()
    pixels = Counter()
    for comp in components:
        counts = Counter(int(grid[r, c]) for r, c in comp)
        for color, count in counts.items():
            if color == bg:
                continue
            presence[color] += 1
            pixels[color] += count

    fg_colors = [int(c) for c in presence]
    assert fg_colors, "expected at least one foreground color"
    track_color = max(fg_colors, key=lambda color: (presence[color], pixels[color]))

    passenger_candidates = [color for color in fg_colors if color != track_color]
    if not passenger_candidates:
        return track_color, None

    repeated = max(presence[color] for color in passenger_candidates)
    likely = [color for color in passenger_candidates if presence[color] == repeated]
    passenger = max(likely, key=lambda color: pixels[color])
    return track_color, int(passenger)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide passengers forward on each grey orbital loop by that loop's fuel count, then erase the fuel.

    Input structure:
    - A black background surrounds several disjoint single-pixel-wide closed loops.
    - Each loop is mostly grey track, with one shared passenger color and one loop-local fuel color painted on track cells.
    - Colored pixels lie directly on the loop, so every non-background connected component is still one 4-connected cycle.

    Rule:
    - For each loop, count its fuel pixels N.
    - Recover the loop order and keep the orientation whose traversal around the loop is clockwise.
    - Move every passenger pixel on that loop forward by N cells along the clockwise order.
    - Clear all fuel pixels back to grey track and leave the shifted passengers on top of the same loop.

    Key insight:
    - The fuel markers do not block motion; they only encode a distance, while the real hidden structure to recover is the clockwise ordering of each underlying cycle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(Counter(map(int, g.flat)).most_common(1)[0][0])
    components = _components(g != bg)
    if not components:
        return g.copy()

    track_color, passenger_color = _infer_track_and_passenger(g, bg, components)
    out = g.copy()

    for comp in components:
        order = _clockwise_cycle(comp)
        index_of = {pos: idx for idx, pos in enumerate(order)}
        passenger_cells = [(r, c) for r, c in comp if passenger_color is not None and int(g[r, c]) == passenger_color]
        fuel_cells = [
            (r, c)
            for r, c in comp
            if int(g[r, c]) != track_color and (passenger_color is None or int(g[r, c]) != passenger_color)
        ]

        for r, c in comp:
            out[r, c] = track_color

        if passenger_color is None or not passenger_cells:
            continue

        shift = len(fuel_cells) % len(order)
        for cell in passenger_cells:
            nr, nc = order[(index_of[cell] + shift) % len(order)]
            out[nr, nc] = passenger_color

    return out


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(int(r0), int(c)) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(int(r), int(c0)) for r in range(r0, r1 + step, step)]
    raise AssertionError("segments must be orthogonal")


def _path_from_vertices(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    for idx in range(len(vertices)):
        seg = _segment(vertices[idx], vertices[(idx + 1) % len(vertices)])
        if path:
            seg = seg[1:]
        path.extend(seg)
    if path and path[-1] == path[0]:
        path.pop()
    return path


def _transform_path(
    path: list[tuple[int, int]], height: int, width: int, flip_lr: bool, rot_k: int
) -> tuple[list[tuple[int, int]], int, int]:
    pts = list(path)
    h, w = int(height), int(width)
    if flip_lr:
        pts = [(r, w - 1 - c) for r, c in pts]
    for _ in range(rot_k % 4):
        pts = [(c, h - 1 - r) for r, c in pts]
        h, w = w, h
    return pts, h, w


def _sample_shape(kind: str) -> tuple[list[tuple[int, int]], int, int]:
    if kind == "rect":
        height = random.randint(4, 8)
        width = random.randint(4, 10)
        vertices = [(0, 0), (0, width - 1), (height - 1, width - 1), (height - 1, 0)]
    else:
        assert kind == "el"
        height = random.randint(7, 10)
        width = random.randint(7, 10)
        cut_row = random.randint(2, height - 3)
        cut_col = random.randint(2, width - 3)
        vertices = [
            (0, 0),
            (0, width - 1),
            (cut_row, width - 1),
            (cut_row, cut_col),
            (height - 1, cut_col),
            (height - 1, 0),
        ]

    path = _path_from_vertices(vertices)
    flip_lr = random.choice([False, True])
    rot_k = random.randint(0, 3)
    path, height, width = _transform_path(path, height, width, flip_lr=flip_lr, rot_k=rot_k)
    _trace_cycle(path)
    return path, height, width


def _boxes_separate(
    box: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], gap: int = 1
) -> bool:
    r0, c0, r1, c1 = box
    for or0, oc0, or1, oc1 in others:
        if not (r1 + gap < or0 or or1 + gap < r0 or c1 + gap < oc0 or oc1 + gap < c0):
            return False
    return True


def _sample_markers(cycle_len: int, fuel_count: int, passenger_count: int) -> tuple[set[int], set[int]]:
    all_idx = list(range(cycle_len))
    for _ in range(80):
        fuel_idx = set(random.sample(all_idx, fuel_count))
        remaining = [idx for idx in all_idx if idx not in fuel_idx]
        passenger_idx = set(random.sample(remaining, passenger_count))
        shifted = {(idx + fuel_count) % cycle_len for idx in passenger_idx}
        if shifted != passenger_idx:
            return fuel_idx, passenger_idx
    raise RuntimeError("failed to sample non-degenerate passenger positions")


def _track_count() -> int:
    return random.choices([2, 3, 4], weights=[3, 3, 2])[0]


def _build_case() -> np.ndarray | None:
    rows = random.randint(18, 30)
    cols = random.randint(18, 30)
    n_tracks = _track_count()

    passenger_color = random.choice(PALETTE)
    fuel_palette = [color for color in PALETTE if color != passenger_color]
    fuel_colors = random.sample(fuel_palette, n_tracks)

    kinds = ["rect", "el"]
    while len(kinds) < n_tracks:
        kinds.append(random.choice(["rect", "el"]))
    random.shuffle(kinds)

    specs = []
    for idx, kind in enumerate(kinds):
        path, height, width = _sample_shape(kind)
        cycle_len = len(path)
        fuel_count = random.randint(1, min(4, max(1, cycle_len // 6)))
        passenger_cap = min(4, max(1, cycle_len // 7))
        passenger_count = random.randint(1, passenger_cap)
        fuel_idx, passenger_idx = _sample_markers(cycle_len, fuel_count, passenger_count)
        specs.append(
            {
                "kind": kind,
                "path": path,
                "height": height,
                "width": width,
                "fuel_color": fuel_colors[idx],
                "fuel_idx": fuel_idx,
                "passenger_idx": passenger_idx,
                "area": height * width,
            }
        )

    specs.sort(key=lambda spec: (spec["area"], len(spec["path"])), reverse=True)

    boxes: list[tuple[int, int, int, int]] = []
    for spec in specs:
        height = int(spec["height"])
        width = int(spec["width"])
        placed = False
        for _ in range(150):
            r0 = random.randint(0, rows - height)
            c0 = random.randint(0, cols - width)
            box = (r0, c0, r0 + height - 1, c0 + width - 1)
            if _boxes_separate(box, boxes, gap=1):
                spec["origin"] = (r0, c0)
                boxes.append(box)
                placed = True
                break
        if not placed:
            return None

    grid = np.full((rows, cols), BACKGROUND, dtype=int)
    for spec in specs:
        r0, c0 = spec["origin"]
        abs_path = [(r0 + r, c0 + c) for r, c in spec["path"]]
        for rr, cc in abs_path:
            grid[rr, cc] = TRACK
        for idx in spec["fuel_idx"]:
            rr, cc = abs_path[idx]
            grid[rr, cc] = int(spec["fuel_color"])
        for idx in spec["passenger_idx"]:
            rr, cc = abs_path[idx]
            grid[rr, cc] = int(passenger_color)

    if len(_components(grid != BACKGROUND)) != n_tracks:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid orbital-loop input")
