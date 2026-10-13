from __future__ import annotations

import random
from collections import Counter

import numpy as np


GREY = 5
MAROON = 9
EDGE_ORDER = ("top", "right", "left", "bottom")
PALETTE = (0, 1, 2, 3, 4, 6, 7, 8)
GRID_H_RANGE = (12, 18)
GRID_W_RANGE = (12, 18)
OBJECT_COUNT_RANGE = (4, 5)
OBJECT_H_RANGE = (2, 5)
OBJECT_W_RANGE = (2, 5)
MAX_GENERATION_ATTEMPTS = 400

_GEN_COUNTER = 0
_CATEGORY_CYCLE = ("none", "top", "right", "left", "bottom", "multi")
_MULTI_PATTERNS = (
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _find_legend(grid: np.ndarray, bg: int) -> tuple[int, int]:
    coords = np.argwhere((grid != bg) & (grid != GREY))
    assert len(coords) == 4, "legend must be the only non-bg, non-grey structure"
    rows = sorted({int(r) for r, _ in coords})
    cols = sorted({int(c) for _, c in coords})
    assert len(rows) == 2 and len(cols) == 2, "legend must occupy a 2x2 footprint"
    assert rows[1] == rows[0] + 1 and cols[1] == cols[0] + 1, "legend must be contiguous"
    return rows[0], cols[0]


def _touches(coords: np.ndarray, h: int, w: int) -> tuple[str, ...]:
    edges: list[str] = []
    if np.any(coords[:, 0] == 0):
        edges.append("top")
    if np.any(coords[:, 1] == w - 1):
        edges.append("right")
    if np.any(coords[:, 1] == 0):
        edges.append("left")
    if np.any(coords[:, 0] == h - 1):
        edges.append("bottom")
    return tuple(edges)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey object according to which grid edge(s) it touches.

    The input contains grey 4-connected objects on a dominant background and a
    central 2x2 legend:
    - legend[0, 0] -> top edge
    - legend[0, 1] -> right edge
    - legend[1, 0] -> left edge
    - legend[1, 1] -> bottom edge

    Grey objects touching exactly one edge inherit that legend color.
    Grey objects touching two or more edges become maroon.
    Grey objects touching no edge stay grey.
    The legend itself is erased to the background color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _background_color(g)
    lr, lc = _find_legend(g, bg)

    legend = {
        "top": int(g[lr, lc]),
        "right": int(g[lr, lc + 1]),
        "left": int(g[lr + 1, lc]),
        "bottom": int(g[lr + 1, lc + 1]),
    }

    out = g.copy()
    for comp in _components(g == GREY):
        touched = _touches(comp, h, w)
        if len(touched) == 1:
            out[comp[:, 0], comp[:, 1]] = int(legend[touched[0]])
        elif len(touched) >= 2:
            out[comp[:, 0], comp[:, 1]] = int(MAROON)

    out[lr : lr + 2, lc : lc + 2] = int(bg)
    return out


def _mark_with_margin(mask: np.ndarray, cells: np.ndarray, margin: int = 1) -> None:
    h, w = mask.shape
    for r, c in cells:
        for rr in range(max(0, int(r) - margin), min(h, int(r) + margin + 1)):
            for cc in range(max(0, int(c) - margin), min(w, int(c) + margin + 1)):
                mask[rr, cc] = True


def _manhattan_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    path = [(r, c)]
    if random.random() < 0.5:
        while c != tc:
            c += 1 if tc > c else -1
            path.append((r, c))
        while r != tr:
            r += 1 if tr > r else -1
            path.append((r, c))
    else:
        while r != tr:
            r += 1 if tr > r else -1
            path.append((r, c))
        while c != tc:
            c += 1 if tc > c else -1
            path.append((r, c))
    return path


def _random_connected_shape(h: int, w: int, edges: tuple[str, ...]) -> np.ndarray:
    anchors: list[tuple[int, int]] = []
    if "top" in edges:
        anchors.append((0, random.randrange(w)))
    if "bottom" in edges:
        anchors.append((h - 1, random.randrange(w)))
    if "left" in edges:
        anchors.append((random.randrange(h), 0))
    if "right" in edges:
        anchors.append((random.randrange(h), w - 1))
    if not anchors:
        anchors.append((random.randrange(h), random.randrange(w)))

    hub_r_lo = 0 if "top" in edges else 1
    hub_r_hi = h - 1 if "bottom" in edges else h - 2
    hub_c_lo = 0 if "left" in edges else 1
    hub_c_hi = w - 1 if "right" in edges else w - 2
    hub_r_lo = min(hub_r_lo, h - 1)
    hub_r_hi = max(hub_r_lo, hub_r_hi)
    hub_c_lo = min(hub_c_lo, w - 1)
    hub_c_hi = max(hub_c_lo, hub_c_hi)
    hub = (random.randint(hub_r_lo, hub_r_hi), random.randint(hub_c_lo, hub_c_hi))

    cells = set()
    for anchor in anchors:
        for cell in _manhattan_path(anchor, hub):
            cells.add(cell)

    target = random.randint(max(3, len(cells)), min(h * w - 1, max(len(cells), random.randint(4, max(4, h * w - 1)))))
    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(tuple(frontier)))

    arr = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        arr[r, c] = True
    return arr


def _category_edges(category: str) -> tuple[str, ...]:
    if category == "none":
        return ()
    if category == "multi":
        return tuple(random.choice(_MULTI_PATTERNS))
    return (category,)


def _candidate_positions(h: int, w: int, obj_h: int, obj_w: int, edges: tuple[str, ...]) -> list[tuple[int, int]]:
    top = "top" in edges
    bottom = "bottom" in edges
    left = "left" in edges
    right = "right" in edges

    if top and bottom:
        return []
    if left and right:
        return []

    if top:
        rows = [0]
    elif bottom:
        rows = [h - obj_h]
    else:
        if h - obj_h - 1 < 1:
            return []
        rows = list(range(1, h - obj_h))

    if left:
        cols = [0]
    elif right:
        cols = [w - obj_w]
    else:
        if w - obj_w - 1 < 1:
            return []
        cols = list(range(1, w - obj_w))

    positions = [(r, c) for r in rows for c in cols]
    random.shuffle(positions)
    return positions


def _place_object(
    grid: np.ndarray,
    blocked: np.ndarray,
    category: str,
) -> tuple[np.ndarray, tuple[str, ...]] | None:
    h, w = grid.shape
    edges = _category_edges(category)

    for _ in range(120):
        obj_h = random.randint(*OBJECT_H_RANGE)
        obj_w = random.randint(*OBJECT_W_RANGE)
        positions = _candidate_positions(h, w, obj_h, obj_w, edges)
        if not positions:
            continue

        local = _random_connected_shape(obj_h, obj_w, edges)
        cells_local = np.argwhere(local)
        if len(cells_local) < 3:
            continue

        for r0, c0 in positions:
            cells = cells_local + np.array([r0, c0], dtype=int)
            if np.any(blocked[cells[:, 0], cells[:, 1]]):
                continue
            touched = _touches(cells, h, w)
            if touched != edges:
                continue
            return cells, edges
    return None


def _build_categories(counter: int) -> list[str]:
    count = random.randint(*OBJECT_COUNT_RANGE)
    singles = ["top", "right", "left", "bottom"]
    start = counter % len(singles)
    base = ["none", "multi", singles[start], singles[(start + 1) % len(singles)]]
    while len(base) < count:
        base.append(random.choice(singles))
    random.shuffle(base)
    return base


def _gen_attempt(counter: int) -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    if h < 6 or w < 6:
        return None

    bg = random.choice(PALETTE)
    legend_colors = random.sample([c for c in PALETTE if c != bg], 4)

    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    lr = h // 2 - 1
    lc = w // 2 - 1
    if lr <= 0 or lc <= 0 or lr + 1 >= h - 1 or lc + 1 >= w - 1:
        return None

    legend = np.array(
        [
            [legend_colors[0], legend_colors[1]],
            [legend_colors[2], legend_colors[3]],
        ],
        dtype=int,
    )
    grid[lr : lr + 2, lc : lc + 2] = legend
    legend_cells = np.argwhere(np.ones((2, 2), dtype=bool)) + np.array([lr, lc], dtype=int)
    _mark_with_margin(blocked, legend_cells, margin=1)

    categories = _build_categories(counter)
    placed: list[tuple[np.ndarray, tuple[str, ...]]] = []
    for category in categories:
        result = _place_object(grid, blocked, category)
        if result is None:
            return None
        cells, edges = result
        grid[cells[:, 0], cells[:, 1]] = GREY
        _mark_with_margin(blocked, cells, margin=1)
        placed.append((cells, edges))

    comps = _components(grid == GREY)
    if len(comps) != len(placed):
        return None

    classes = Counter(len(_touches(comp, h, w)) for comp in comps)
    if classes[0] == 0 or classes[1] == 0 or sum(v for k, v in classes.items() if k >= 2) == 0:
        return None

    legend_coords = np.argwhere((grid != bg) & (grid != GREY))
    if len(legend_coords) != 4:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(_GEN_COUNTER)
        if grid is not None:
            _GEN_COUNTER += 1
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
