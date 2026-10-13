from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

MIN_PAIRS = 3
MAX_PAIRS = 5
MIN_SIZE = 12
MAX_SIZE = 25

GEN_RETRIES = 400
PAIR_RETRIES = 160
WANT_OVERLAP_PROB = 0.6
MAX_OVERLAP_CELLS = 8
MAX_STACKED_PATHS = 3


Pos = tuple[int, int]


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[Pos] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _edge_of(pos: Pos, shape: tuple[int, int]) -> str:
    r, c = pos
    h, w = shape
    sides = []
    if r == 0:
        sides.append("top")
    if r == h - 1:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    if c == w - 1:
        sides.append("right")
    assert len(sides) == 1, "anchors must lie on exactly one border side"
    return sides[0]


def _segment(start: Pos, end: Pos) -> list[Pos]:
    sr, sc = start
    er, ec = end
    dr = er - sr
    dc = ec - sc
    assert dr == 0 or dc == 0 or abs(dr) == abs(dc)
    step_r = 0 if dr == 0 else dr // abs(dr)
    step_c = 0 if dc == 0 else dc // abs(dc)
    steps = max(abs(dr), abs(dc))
    return [(sr + k * step_r, sc + k * step_c) for k in range(steps + 1)]


def _ordered_for_l(a: Pos, b: Pos, shape: tuple[int, int]) -> tuple[Pos, Pos]:
    edge_a = _edge_of(a, shape)
    edge_b = _edge_of(b, shape)
    vertical_edges = {"left", "right"}
    horizontal_edges = {"top", "bottom"}

    if edge_a in vertical_edges and edge_b in horizontal_edges:
        return a, b
    if edge_b in vertical_edges and edge_a in horizontal_edges:
        return b, a
    if {edge_a, edge_b} == {"left", "right"}:
        return (a, b) if edge_a == "left" else (b, a)
    if {edge_a, edge_b} == {"top", "bottom"}:
        return (a, b) if edge_a == "top" else (b, a)
    return min(a, b), max(a, b)


def _trace_path(a: Pos, b: Pos, shape: tuple[int, int]) -> list[Pos]:
    if a[0] == b[0] or a[1] == b[1] or abs(a[0] - b[0]) == abs(a[1] - b[1]):
        path = _segment(a, b)
    else:
        start, end = _ordered_for_l(a, b, shape)
        corner = (start[0], end[1])
        first = _segment(start, corner)
        second = _segment(corner, end)
        path = first + second[1:]

    h, w = shape
    assert all(0 <= r < h and 0 <= c < w for r, c in path)
    return path


def _find_legend(g: np.ndarray) -> tuple[tuple[int, int, int, int], list[int]]:
    grey_comps = _components_mask(g == GREY)
    candidates: list[tuple[tuple[int, int, int, int], list[int]]] = []

    for coords in grey_comps:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        if width != 4 or not (MIN_PAIRS + 2 <= height <= MAX_PAIRS + 2):
            continue

        border_area = 2 * (height + width) - 4
        if int(coords.shape[0]) != border_area:
            continue

        local = np.zeros((height, width), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        border = np.ones((height, width), dtype=bool)
        border[1:-1, 1:-1] = False
        if not np.array_equal(local, border):
            continue

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.shape[1] != 2:
            continue

        colors: list[int] = []
        ok = True
        for row in inner:
            left = int(row[0])
            right = int(row[1])
            if left != right or left in {BLACK, GREY}:
                ok = False
                break
            colors.append(left)
        if not ok or len(set(colors)) != len(colors):
            continue
        candidates.append(((int(r0), int(c0), int(r1), int(c1)), colors))

    assert len(candidates) == 1, "expected one grey priority legend"
    return candidates[0]


def _find_border_anchors(g: np.ndarray, legend_colors: list[int]) -> dict[int, tuple[Pos, Pos]]:
    h, w = g.shape
    by_color: dict[int, list[Pos]] = {}

    for c in range(w):
        for r in (0, h - 1):
            color = int(g[r, c])
            if color not in {BLACK, GREY}:
                by_color.setdefault(color, []).append((r, c))
    for r in range(1, h - 1):
        for c in (0, w - 1):
            color = int(g[r, c])
            if color not in {BLACK, GREY}:
                by_color.setdefault(color, []).append((r, c))

    assert set(by_color) == set(legend_colors), "legend colors must match border anchor colors"

    anchors: dict[int, tuple[Pos, Pos]] = {}
    for color, pts in by_color.items():
        assert len(pts) == 2, "each color must have exactly two border anchors"
        assert _edge_of(pts[0], g.shape) != _edge_of(pts[1], g.shape)
        anchors[color] = (pts[0], pts[1])
    return anchors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color border-anchor pair, then resolve overlaps using the
    top-to-bottom order encoded by the interior grey legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    legend_bbox, priority = _find_legend(g)
    _ = legend_bbox
    anchors = _find_border_anchors(g, priority)

    out = np.full(g.shape, BLACK, dtype=int)
    for color in reversed(priority):
        a, b = anchors[color]
        for r, c in _trace_path(a, b, g.shape):
            out[r, c] = int(color)
    return out


def _sample_families(n_pairs: int, rng: random.Random) -> list[str]:
    base = ["h", "v", "d", "l"]
    rng.shuffle(base)
    families = base[: min(len(base), n_pairs)]
    while len(families) < n_pairs:
        families.append(rng.choice(base))
    rng.shuffle(families)
    return families


def _sample_pair(family: str, h: int, w: int, rng: random.Random) -> tuple[Pos, Pos]:
    if family == "h":
        row = rng.randint(1, h - 2)
        return (row, 0), (row, w - 1)

    if family == "v":
        col = rng.randint(1, w - 2)
        return (0, col), (h - 1, col)

    if family == "d":
        d = rng.randint(1, min(h, w) - 2)
        mode = rng.randint(0, 3)
        if mode == 0:
            return (d, 0), (0, d)
        if mode == 1:
            return (d, w - 1), (0, w - 1 - d)
        if mode == 2:
            return (h - 1 - d, 0), (h - 1, d)
        return (h - 1 - d, w - 1), (h - 1, w - 1 - d)

    assert family == "l"
    mode = rng.randint(0, 3)
    if mode == 0:
        while True:
            row = rng.randint(1, h - 2)
            col = rng.randint(1, w - 2)
            if row != col:
                return (row, 0), (0, col)
    if mode == 1:
        while True:
            row = rng.randint(1, h - 2)
            col = rng.randint(1, w - 2)
            if row != (w - 1 - col):
                return (row, w - 1), (0, col)
    if mode == 2:
        while True:
            row = rng.randint(1, h - 2)
            col = rng.randint(1, w - 2)
            if (h - 1 - row) != col:
                return (row, 0), (h - 1, col)
    while True:
        row = rng.randint(1, h - 2)
        col = rng.randint(1, w - 2)
        if (h - 1 - row) != (w - 1 - col):
            return (row, w - 1), (h - 1, col)


def _legend_candidates(
    h: int,
    w: int,
    n_pairs: int,
    forbidden: set[Pos],
) -> list[Pos]:
    legend_h = n_pairs + 2
    legend_w = 4
    candidates: list[Pos] = []
    for r0 in range(1, h - legend_h):
        for c0 in range(1, w - legend_w):
            pad_r0 = r0 - 1
            pad_c0 = c0 - 1
            pad_r1 = r0 + legend_h
            pad_c1 = c0 + legend_w
            blocked = False
            for r in range(pad_r0, pad_r1 + 1):
                for c in range(pad_c0, pad_c1 + 1):
                    if (r, c) in forbidden:
                        blocked = True
                        break
                if blocked:
                    break
            if not blocked:
                candidates.append((r0, c0))
    return candidates


def _render_input(
    h: int,
    w: int,
    pairs: list[tuple[int, Pos, Pos]],
    priority: list[int],
    legend_origin: Pos,
) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for color, a, b in pairs:
        grid[a] = int(color)
        grid[b] = int(color)

    r0, c0 = legend_origin
    legend_h = len(priority) + 2
    legend_w = 4
    grid[r0 : r0 + legend_h, c0 : c0 + legend_w] = GREY
    for i, color in enumerate(priority):
        grid[r0 + 1 + i, c0 + 1 : c0 + 3] = int(color)
    return grid


def _path_overlap_stats(paths: list[list[Pos]]) -> tuple[int, int]:
    counts: dict[Pos, int] = {}
    for path in paths:
        for cell in path:
            counts[cell] = counts.get(cell, 0) + 1
    overlap_cells = sum(v > 1 for v in counts.values())
    max_stack = max(counts.values())
    return overlap_cells, max_stack


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(MIN_SIZE, MAX_SIZE)
    w = rng.randint(MIN_SIZE, MAX_SIZE)
    n_pairs = rng.randint(MIN_PAIRS, MAX_PAIRS)
    colors = rng.sample(COLOR_POOL, n_pairs)
    priority = colors[:]
    rng.shuffle(priority)
    families = _sample_families(n_pairs, rng)
    want_overlap = rng.random() < WANT_OVERLAP_PROB

    used_anchors: set[Pos] = set()
    path_union: set[Pos] = set()
    pairs: list[tuple[int, Pos, Pos]] = []
    paths: list[list[Pos]] = []

    for color, family in zip(colors, families, strict=False):
        placed = False
        for _ in range(PAIR_RETRIES):
            a, b = _sample_pair(family, h, w, rng)
            if a in used_anchors or b in used_anchors:
                continue
            if a in path_union or b in path_union:
                continue

            path = _trace_path(a, b, (h, w))
            if any(cell in used_anchors and cell not in {a, b} for cell in path):
                continue

            used_anchors.update((a, b))
            path_union.update(path)
            pairs.append((color, a, b))
            paths.append(path)
            placed = True
            break
        if not placed:
            return None

    overlap_cells, max_stack = _path_overlap_stats(paths)
    if want_overlap and overlap_cells == 0:
        return None
    if overlap_cells > MAX_OVERLAP_CELLS or max_stack > MAX_STACKED_PATHS:
        return None

    forbidden = set(path_union) | used_anchors
    legend_spots = _legend_candidates(h, w, n_pairs, forbidden)
    if not legend_spots:
        return None

    legend_origin = rng.choice(legend_spots)
    grid = _render_input(h, w, pairs, priority, legend_origin)

    try:
        out = main(grid)
    except Exception:
        return None

    if GREY in np.unique(out):
        return None
    if not set(map(int, np.unique(out))).issubset({BLACK, *colors}):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate a valid input after {GEN_RETRIES} attempts")
