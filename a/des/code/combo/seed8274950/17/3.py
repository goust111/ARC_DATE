import numpy as np
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Shape:
    color: int
    r0: int
    c0: int
    h: int
    w: int

    @property
    def r1(self) -> int:
        return self.r0 + self.h - 1

    @property
    def c1(self) -> int:
        return self.c0 + self.w - 1

    @property
    def area(self) -> int:
        return self.h * self.w


# --------------- solver ---------------


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _shape_from_visible_color(grid: np.ndarray, color: int) -> Shape:
    coords = np.argwhere(grid == color)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return Shape(int(color), int(r0), int(c0), int(r1 - r0 + 1), int(c1 - c0 + 1))


def _overlap_bbox(a: Shape, b: Shape) -> tuple[int, int, int, int] | None:
    r0 = max(a.r0, b.r0)
    c0 = max(a.c0, b.c0)
    r1 = min(a.r1, b.r1)
    c1 = min(a.c1, b.c1)
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _shape_sort_key(shape: Shape) -> tuple[int, int, int, int, int]:
    return (shape.c0, shape.r0, shape.w, shape.h, shape.color)


def _infer_layer_order(grid: np.ndarray, shapes: list[Shape]) -> list[Shape]:
    n = len(shapes)
    above = [set() for _ in range(n)]
    indeg = [0] * n

    for i in range(n):
        for j in range(i + 1, n):
            overlap = _overlap_bbox(shapes[i], shapes[j])
            if overlap is None:
                continue
            r0, c0, r1, c1 = overlap
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            sees_i = bool(np.any(patch == shapes[i].color))
            sees_j = bool(np.any(patch == shapes[j].color))
            if sees_i and sees_j:
                continue
            if sees_i:
                lower, upper = j, i
            elif sees_j:
                lower, upper = i, j
            else:
                continue
            if upper not in above[lower]:
                above[lower].add(upper)
                indeg[upper] += 1

    pending = set(range(n))
    order: list[int] = []
    while pending:
        ready = sorted(
            (idx for idx in pending if indeg[idx] == 0),
            key=lambda idx: _shape_sort_key(shapes[idx]),
        )
        if not ready:
            ready = [min(pending, key=lambda idx: _shape_sort_key(shapes[idx]))]
        idx = ready[0]
        pending.remove(idx)
        order.append(idx)
        for nxt in above[idx]:
            indeg[nxt] -= 1
    return [shapes[idx] for idx in order]


def _pack_shapes(shapes: Iterable[Shape], bg: int) -> np.ndarray:
    ordered = list(shapes)
    if not ordered:
        return np.full((1, 1), bg, dtype=int)
    out_h = max(shape.h for shape in ordered)
    out_w = sum(shape.w for shape in ordered) + len(ordered) - 1
    out = np.full((out_h, out_w), bg, dtype=int)
    c = 0
    for shape in ordered:
        out[: shape.h, c : c + shape.w] = shape.color
        c += shape.w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decompose an overpainted multicolor pattern into its base rectangles.

    Each color corresponds to one solid axis-aligned rectangle. Line segments are
    represented as rectangles with height 1 or width 1. Later rectangles overwrite
    earlier ones. Recover every rectangle from the bbox of its visible pixels, infer
    the partial order from which color remains visible on overlaps, and output the
    tight crops from bottom layer to top layer with one background column between them.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    shapes = [_shape_from_visible_color(g, color) for color in colors]
    ordered = _infer_layer_order(g, shapes)
    return _pack_shapes(ordered, bg)


# --------------- generator ---------------

KINDS = ("hline", "vline", "rect")
GRID_MIN = 12
GRID_MAX = 24
SHAPE_COUNT_CHOICES = (3, 4, 5)
LINE_LEN_RANGE = (4, 9)
RECT_H_RANGE = (3, 7)
RECT_W_RANGE = (3, 7)
PAD_RANGE = (1, 5)
MAX_CHAIN_SPAN = 18
MAX_ATTEMPTS = 400


def _sample_kind_sequence(rng: np.random.Generator, n: int) -> list[str]:
    seq = list(KINDS[: min(n, len(KINDS))])
    while len(seq) < n:
        seq.append(str(rng.choice(KINDS)))
    rng.shuffle(seq)
    return seq


def _sample_dims(rng: np.random.Generator, kind: str) -> tuple[int, int]:
    if kind == "hline":
        return 1, int(rng.integers(*LINE_LEN_RANGE))
    if kind == "vline":
        return int(rng.integers(*LINE_LEN_RANGE)), 1
    return int(rng.integers(*RECT_H_RANGE)), int(rng.integers(*RECT_W_RANGE))


def _safe_cells(shape: Shape) -> list[tuple[int, int]]:
    if shape.h == 1:
        return [(shape.r0, c) for c in range(shape.c0 + 1, shape.c1)]
    if shape.w == 1:
        return [(r, shape.c0) for r in range(shape.r0 + 1, shape.r1)]
    return [
        (r, c)
        for r in range(shape.r0 + 1, shape.r1)
        for c in range(shape.c0 + 1, shape.c1)
    ]


def _overlap_inside_safe(prev: Shape, nxt: Shape) -> bool:
    overlap = _overlap_bbox(prev, nxt)
    if overlap is None:
        return False
    r0, c0, r1, c1 = overlap
    if prev.h == 1:
        return r0 == r1 == prev.r0 and c0 >= prev.c0 + 1 and c1 <= prev.c1 - 1
    if prev.w == 1:
        return c0 == c1 == prev.c0 and r0 >= prev.r0 + 1 and r1 <= prev.r1 - 1
    return (
        r0 >= prev.r0 + 1
        and r1 <= prev.r1 - 1
        and c0 >= prev.c0 + 1
        and c1 <= prev.c1 - 1
    )


def _intersects(a: Shape, b: Shape) -> bool:
    return _overlap_bbox(a, b) is not None


def _anchored_candidate(
    color: int,
    anchor: tuple[int, int],
    dims: tuple[int, int],
    direction: str,
    rng: np.random.Generator,
) -> Shape:
    ar, ac = anchor
    h, w = dims
    if direction == "up":
        r0 = ar - h + 1
        c0 = int(rng.integers(ac - w + 1, ac + 1))
    elif direction == "down":
        r0 = ar
        c0 = int(rng.integers(ac - w + 1, ac + 1))
    elif direction == "left":
        r0 = int(rng.integers(ar - h + 1, ar + 1))
        c0 = ac - w + 1
    else:
        r0 = int(rng.integers(ar - h + 1, ar + 1))
        c0 = ac
    return Shape(color, int(r0), int(c0), int(h), int(w))


def _place_next_shape(
    rng: np.random.Generator,
    color: int,
    prev: Shape,
    older: list[Shape],
    kind: str,
) -> Shape | None:
    safe = _safe_cells(prev)
    if not safe:
        return None
    dims = _sample_dims(rng, kind)
    directions = ["up", "down", "left", "right"]
    rng.shuffle(directions)
    for _ in range(80):
        anchor = safe[int(rng.integers(len(safe)))]
        for direction in directions:
            cand = _anchored_candidate(color, anchor, dims, direction, rng)
            overlap = _overlap_bbox(prev, cand)
            if overlap is None or not _overlap_inside_safe(prev, cand):
                continue
            overlap_area = (overlap[2] - overlap[0] + 1) * (overlap[3] - overlap[1] + 1)
            if overlap_area >= cand.area:
                continue
            if any(_intersects(cand, old) for old in older):
                continue
            if not _safe_cells(cand):
                continue
            return cand
    return None


def _translate_shapes(
    rng: np.random.Generator, shapes: list[Shape], bg: int
) -> tuple[list[Shape], np.ndarray] | None:
    min_r = min(shape.r0 for shape in shapes)
    min_c = min(shape.c0 for shape in shapes)
    max_r = max(shape.r1 for shape in shapes)
    max_c = max(shape.c1 for shape in shapes)
    shape_h = max_r - min_r + 1
    shape_w = max_c - min_c + 1
    if shape_h > MAX_CHAIN_SPAN or shape_w > MAX_CHAIN_SPAN:
        return None

    top = int(rng.integers(*PAD_RANGE))
    left = int(rng.integers(*PAD_RANGE))
    bottom = int(rng.integers(*PAD_RANGE))
    right = int(rng.integers(*PAD_RANGE))
    grid_h = shape_h + top + bottom
    grid_w = shape_w + left + right
    if grid_h < GRID_MIN:
        bottom += GRID_MIN - grid_h
        grid_h = GRID_MIN
    if grid_w < GRID_MIN:
        right += GRID_MIN - grid_w
        grid_w = GRID_MIN
    if grid_h > GRID_MAX or grid_w > GRID_MAX:
        return None

    dr = top - min_r
    dc = left - min_c
    shifted = [
        Shape(shape.color, shape.r0 + dr, shape.c0 + dc, shape.h, shape.w)
        for shape in shapes
    ]
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    for shape in shifted:
        grid[shape.r0 : shape.r1 + 1, shape.c0 : shape.c1 + 1] = shape.color
    return shifted, grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_shapes = int(rng.choice(SHAPE_COUNT_CHOICES))
    colors = list(range(10))
    rng.shuffle(colors)
    bg = int(colors[0])
    shape_colors = [int(c) for c in colors[1 : 1 + n_shapes]]

    kinds = _sample_kind_sequence(rng, n_shapes)
    h0, w0 = _sample_dims(rng, kinds[0])
    shapes = [Shape(shape_colors[0], 0, 0, h0, w0)]

    for idx in range(1, n_shapes):
        nxt = _place_next_shape(
            rng,
            shape_colors[idx],
            shapes[-1],
            shapes[:-1],
            kinds[idx],
        )
        if nxt is None:
            return None
        shapes.append(nxt)

    translated = _translate_shapes(rng, shapes, bg)
    if translated is None:
        return None
    shifted, grid = translated
    if _background_color(grid) != bg:
        return None
    expected = _pack_shapes(shifted, bg)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid decomposition puzzle")
