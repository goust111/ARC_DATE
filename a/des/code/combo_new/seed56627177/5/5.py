from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from functools import lru_cache
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
BROWN = 9

BACKGROUND_PAIR = (BLACK, AZURE)
ANCHOR_COLORS = (BLUE, RED, GREEN, YELLOW, GREY, MAGENTA, BROWN)

GEN_GRID_MIN = 12
GEN_GRID_MAX = 24
GEN_PAIR_COUNT = (1, 3)
GEN_COMPONENT_SIZE = (3, 5)
GEN_MAX_SIDE_OFFSET = 3
GEN_RETRIES = 400


@dataclass(frozen=True)
class Axis:
    color: int
    kind: str
    a: tuple[int, int]
    b: tuple[int, int]

    def side_value(self, cell: tuple[int, int]) -> int:
        r, c = cell
        if self.kind == "h":
            return r - self.a[0]
        if self.kind == "v":
            return c - self.a[1]
        if self.kind == "d1":
            return (r - c) - (self.a[0] - self.a[1])
        return (r + c) - (self.a[0] + self.a[1])

    def reflect(self, cell: tuple[int, int]) -> tuple[int, int]:
        r, c = cell
        if self.kind == "h":
            return 2 * self.a[0] - r, c
        if self.kind == "v":
            return r, 2 * self.a[1] - c
        if self.kind == "d1":
            k = self.a[0] - self.a[1]
            return c + k, r - k
        s = self.a[0] + self.a[1]
        return s - c, s - r

    def in_segment(self, cell: tuple[int, int]) -> bool:
        r, c = cell
        if self.kind == "h":
            return min(self.a[1], self.b[1]) <= c <= max(self.a[1], self.b[1])
        if self.kind == "v":
            return min(self.a[0], self.b[0]) <= r <= max(self.a[0], self.b[0])
        if self.kind == "d1":
            s = r + c
            lo = min(self.a[0] + self.a[1], self.b[0] + self.b[1])
            hi = max(self.a[0] + self.a[1], self.b[0] + self.b[1])
            return lo <= s <= hi
        d = r - c
        lo = min(self.a[0] - self.a[1], self.b[0] - self.b[1])
        hi = max(self.a[0] - self.a[1], self.b[0] - self.b[1])
        return lo <= d <= hi

    def axis_cells(self) -> list[tuple[int, int]]:
        if self.kind == "h":
            row = self.a[0]
            return [(row, c) for c in range(self.a[1], self.b[1] + 1)]
        if self.kind == "v":
            col = self.a[1]
            return [(r, col) for r in range(self.a[0], self.b[0] + 1)]
        if self.kind == "d1":
            length = self.b[0] - self.a[0]
            return [(self.a[0] + t, self.a[1] + t) for t in range(length + 1)]
        length = self.b[0] - self.a[0]
        return [(self.a[0] + t, self.a[1] - t) for t in range(length + 1)]


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(comp)
    return parts


def _infer_checkerboard_colors(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    parity = (np.indices((h, w)).sum(axis=0) & 1).astype(bool)
    colors = list(map(int, np.unique(grid)))
    best_score = -1
    best_pair = None
    for a in colors:
        for b in colors:
            if a == b:
                continue
            score = int(np.sum(grid[~parity] == a) + np.sum(grid[parity] == b))
            if score > best_score:
                best_score = score
                best_pair = (a, b)
    assert best_pair is not None
    return best_pair


def _make_axis(color: int, p0: tuple[int, int], p1: tuple[int, int]) -> Axis:
    r0, c0 = p0
    r1, c1 = p1
    if r0 == r1:
        a, b = sorted((p0, p1), key=lambda p: p[1])
        return Axis(color, "h", a, b)
    if c0 == c1:
        a, b = sorted((p0, p1), key=lambda p: p[0])
        return Axis(color, "v", a, b)
    assert abs(r0 - r1) == abs(c0 - c1)
    if (r1 - r0) * (c1 - c0) > 0:
        a, b = sorted((p0, p1), key=lambda p: p[0])
        return Axis(color, "d1", a, b)
    top, bottom = sorted((p0, p1), key=lambda p: p[0])
    if top[1] < bottom[1]:
        top, bottom = bottom, top
    return Axis(color, "d2", top, bottom)


def _component_score(
    grid: np.ndarray,
    component: list[tuple[int, int]],
    axis: Axis,
    bg_colors: set[int],
) -> float | None:
    h, w = grid.shape
    direction = None
    total_distance = 0
    reflected: set[tuple[int, int]] = set()
    for cell in component:
        side = axis.side_value(cell)
        if side == 0 or not axis.in_segment(cell):
            return None
        sign = 1 if side > 0 else -1
        if direction is None:
            direction = sign
        elif direction != sign:
            return None
        rr, cc = axis.reflect(cell)
        if not (0 <= rr < h and 0 <= cc < w):
            return None
        if not axis.in_segment((rr, cc)):
            return None
        if int(grid[rr, cc]) not in bg_colors:
            return None
        if (rr, cc) in reflected:
            return None
        reflected.add((rr, cc))
        total_distance += abs(side)
    return -float(total_distance) / float(len(component))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the orange motif across each anchor-defined axis and repaint the reflected copy with the anchor color.

    Input structure:
    - The whole grid sits on an alternating two-color checkerboard background.
    - Every non-background anchor color appears exactly twice, and each same-color pair lies on one row, one column, or one 45-degree diagonal.
    - Orange pixels form one or more small 4-connected motifs placed between anchor pairs on one side of their axis.

    Rule:
    - Infer the checkerboard background, then pair anchor pixels by color.
    - Treat each anchor pair's connecting segment as a mirror axis: horizontal, vertical, or diagonal.
    - For every orange motif, reflect it across its matching axis, keep the original orange pixels unchanged, and paint only the reflected copy with that pair's anchor color.

    Key insight:
    - The geometry comes from color pairing first and only then from the motif shape: the output color is determined by the anchor pair that defines the reflection axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg_even, bg_odd = _infer_checkerboard_colors(g)
    bg_colors = {bg_even, bg_odd}

    counts = Counter(map(int, g.flat))
    non_bg = [color for color in counts if color not in bg_colors]
    anchor_colors = sorted(color for color in non_bg if counts[color] == 2)
    motif_colors = [color for color in non_bg if counts[color] != 2]
    assert len(motif_colors) == 1
    motif_color = motif_colors[0]

    axes: list[Axis] = []
    for color in anchor_colors:
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2)
        p0 = tuple(map(int, coords[0]))
        p1 = tuple(map(int, coords[1]))
        axes.append(_make_axis(color, p0, p1))

    motif_components = _components(g == motif_color)
    assert motif_components

    scores = [
        [_component_score(g, component, axis, bg_colors) for axis in axes]
        for component in motif_components
    ]

    @lru_cache(maxsize=None)
    def _assign(idx: int, used_mask: int) -> tuple[float, tuple[int, ...]] | None:
        if idx == len(motif_components):
            return 0.0, ()
        best: tuple[float, tuple[int, ...]] | None = None
        for axis_idx, score in enumerate(scores[idx]):
            if used_mask & (1 << axis_idx) or score is None:
                continue
            tail = _assign(idx + 1, used_mask | (1 << axis_idx))
            if tail is None:
                continue
            total = score + tail[0]
            choice = (axis_idx,) + tail[1]
            if best is None or total > best[0]:
                best = (total, choice)
        return best

    assignment = _assign(0, 0)
    assert assignment is not None

    out = g.copy()
    for component, axis_idx in zip(motif_components, assignment[1]):
        axis = axes[axis_idx]
        for cell in component:
            rr, cc = axis.reflect(cell)
            out[rr, cc] = axis.color
    return out


def _checkerboard(h: int, w: int, even_color: int, odd_color: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    return np.where(((rows + cols) & 1) == 0, even_color, odd_color).astype(int)


def _normalize_shape(cells: list[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    mask = np.zeros((max(rows) - r0 + 1, max(cols) - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _is_asymmetric_shape(cells: list[tuple[int, int]]) -> bool:
    mask = _normalize_shape(cells)
    if np.array_equal(mask, mask[::-1, :]):
        return False
    if np.array_equal(mask, mask[:, ::-1]):
        return False
    if mask.shape[0] == mask.shape[1]:
        if np.array_equal(mask, mask.T):
            return False
        if np.array_equal(mask, np.fliplr(mask).T):
            return False
    return True


def _reserve(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    reserved: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    reserved.add((rr, cc))
    return reserved


def _candidate_cells(
    axis: Axis,
    side_sign: int,
    forbidden: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    anchor_cells = {axis.a, axis.b}
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            cell = (r, c)
            side = axis.side_value(cell)
            if side == 0 or (1 if side > 0 else -1) != side_sign:
                continue
            if abs(side) > GEN_MAX_SIDE_OFFSET or not axis.in_segment(cell):
                continue
            rr, cc = axis.reflect(cell)
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            refl = (rr, cc)
            if not axis.in_segment(refl):
                continue
            if cell in forbidden or refl in forbidden:
                continue
            if cell in anchor_cells or refl in anchor_cells:
                continue
            if min(abs(r - ar) + abs(c - ac) for ar, ac in anchor_cells) <= 1:
                continue
            if min(abs(rr - ar) + abs(cc - ac) for ar, ac in anchor_cells) <= 1:
                continue
            candidates.append(cell)
    return candidates


def _grow_component(
    axis: Axis,
    side_sign: int,
    forbidden: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int]] | None:
    candidate_set = set(_candidate_cells(axis, side_sign, forbidden, h, w))
    if len(candidate_set) < GEN_COMPONENT_SIZE[0]:
        return None
    target = random.randint(*GEN_COMPONENT_SIZE)
    for _ in range(80):
        seed = random.choice(tuple(candidate_set))
        comp = {seed}
        while len(comp) < target:
            frontier = []
            for r, c in comp:
                for nr, nc in _neighbors4(r, c, h, w):
                    if (nr, nc) in candidate_set and (nr, nc) not in comp:
                        frontier.append((nr, nc))
            if not frontier:
                break
            comp.add(random.choice(frontier))
        if len(comp) != target:
            continue
        ordered = sorted(comp)
        if not _is_asymmetric_shape(ordered):
            continue
        reflected = {axis.reflect(cell) for cell in ordered}
        if comp & reflected:
            continue
        return ordered
    return None


def _random_axis(kind: str, color: int, h: int, w: int) -> Axis | None:
    for _ in range(80):
        if kind == "h":
            span = random.randint(4, max(4, min(10, w - 3)))
            row = random.randint(2, h - 3)
            left = random.randint(1, w - span - 2)
            return Axis(color, "h", (row, left), (row, left + span))
        if kind == "v":
            span = random.randint(4, max(4, min(10, h - 3)))
            col = random.randint(2, w - 3)
            top = random.randint(1, h - span - 2)
            return Axis(color, "v", (top, col), (top + span, col))
        if kind == "d1":
            span = random.randint(4, max(4, min(h, w) - 3))
            r0 = random.randint(1, h - span - 2)
            c0 = random.randint(1, w - span - 2)
            return Axis(color, "d1", (r0, c0), (r0 + span, c0 + span))
        span = random.randint(4, max(4, min(h, w) - 3))
        r0 = random.randint(1, h - span - 2)
        c0 = random.randint(span + 1, w - 2)
        return Axis(color, "d2", (r0, c0), (r0 + span, c0 - span))
    return None


def _semantic_kinds(pair_count: int) -> list[str]:
    semantics = ["h", "v", "diag"]
    random.shuffle(semantics)
    chosen = semantics[:pair_count]
    kinds: list[str] = []
    for item in chosen:
        if item == "diag":
            kinds.append(random.choice(["d1", "d2"]))
        else:
            kinds.append(item)
    random.shuffle(kinds)
    return kinds


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    w = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    pair_count = random.randint(*GEN_PAIR_COUNT)
    if min(h, w) < 13 and pair_count == 3:
        return None

    even_color, odd_color = BACKGROUND_PAIR
    if random.random() < 0.5:
        even_color, odd_color = odd_color, even_color
    grid = _checkerboard(h, w, even_color, odd_color)

    colors = random.sample(ANCHOR_COLORS, pair_count)
    kinds = _semantic_kinds(pair_count)
    forbidden: set[tuple[int, int]] = set()
    expected_new = 0

    for color, kind in zip(colors, kinds):
        placed = False
        for _ in range(120):
            axis = _random_axis(kind, color, h, w)
            if axis is None:
                continue
            corridor = set(axis.axis_cells())
            if corridor & forbidden:
                continue
            side_sign = random.choice([-1, 1])
            component = _grow_component(axis, side_sign, forbidden | corridor, h, w)
            if component is None:
                continue
            reflected = {axis.reflect(cell) for cell in component}
            payload = corridor | {axis.a, axis.b} | set(component) | reflected
            if payload & forbidden:
                continue
            grid[axis.a] = color
            grid[axis.b] = color
            for cell in component:
                grid[cell] = ORANGE
            forbidden |= _reserve(payload, h, w)
            expected_new += len(component)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if int(np.count_nonzero(out != grid)) != expected_new:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
