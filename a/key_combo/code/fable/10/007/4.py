from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BLACK = 0
COLORS = tuple(range(1, 10))


@dataclass(frozen=True)
class RayConflict:
    axis: str
    side: str
    fixed: int
    color: int
    candidates: tuple[int, ...]


def _first_visible(scene: np.ndarray, start: tuple[int, int], direction: tuple[int, int]):
    h, w = scene.shape
    r, c = start
    dr, dc = direction
    while 0 <= r < h and 0 <= c < w:
        color = int(scene[r, c])
        if color != BLACK:
            return (r, c), color
        r += dr
        c += dc
    return None, BLACK


def _projection_ring(scene: np.ndarray) -> np.ndarray:
    h, w = scene.shape
    ring = np.zeros((h + 2, w + 2), dtype=int)

    for c in range(w):
        _coord, color = _first_visible(scene, (0, c), (1, 0))
        ring[0, c + 1] = color
        _coord, color = _first_visible(scene, (h - 1, c), (-1, 0))
        ring[h + 1, c + 1] = color

    for r in range(h):
        _coord, color = _first_visible(scene, (r, 0), (0, 1))
        ring[r + 1, 0] = color
        _coord, color = _first_visible(scene, (r, w - 1), (0, -1))
        ring[r + 1, w + 1] = color

    return ring


def _split_grid(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 3 and w >= 3, "expected an outer ring and an inner scene"
    assert int(g[0, 0]) == int(g[0, -1]) == int(g[-1, 0]) == int(g[-1, -1]) == BLACK, "corners must be black"
    return g, g[1:-1, 1:-1]


def _collect_conflicts(grid: np.ndarray) -> tuple[list[RayConflict], list[RayConflict]]:
    g, scene = _split_grid(grid)
    h, w = scene.shape
    horiz: list[RayConflict] = []
    vert: list[RayConflict] = []

    for r in range(h):
        expected = int(g[r + 1, 0])
        if expected != BLACK:
            coord, seen = _first_visible(scene, (r, 0), (0, 1))
            if coord is None or seen != expected:
                cols = []
                c = 0
                while c < w and int(scene[r, c]) == BLACK:
                    cols.append(c)
                    c += 1
                horiz.append(RayConflict("h", "left", r, expected, tuple(cols)))

        expected = int(g[r + 1, w + 1])
        if expected != BLACK:
            coord, seen = _first_visible(scene, (r, w - 1), (0, -1))
            if coord is None or seen != expected:
                cols = []
                c = w - 1
                while c >= 0 and int(scene[r, c]) == BLACK:
                    cols.append(c)
                    c -= 1
                horiz.append(RayConflict("h", "right", r, expected, tuple(cols)))

    for c in range(w):
        expected = int(g[0, c + 1])
        if expected != BLACK:
            coord, seen = _first_visible(scene, (0, c), (1, 0))
            if coord is None or seen != expected:
                rows = []
                r = 0
                while r < h and int(scene[r, c]) == BLACK:
                    rows.append(r)
                    r += 1
                vert.append(RayConflict("v", "top", c, expected, tuple(rows)))

        expected = int(g[h + 1, c + 1])
        if expected != BLACK:
            coord, seen = _first_visible(scene, (h - 1, c), (-1, 0))
            if coord is None or seen != expected:
                rows = []
                r = h - 1
                while r >= 0 and int(scene[r, c]) == BLACK:
                    rows.append(r)
                    r -= 1
                vert.append(RayConflict("v", "bottom", c, expected, tuple(rows)))

    return horiz, vert


def _ring_matches_archive(grid: np.ndarray) -> bool:
    g, scene = _split_grid(grid)
    return np.array_equal(g, _compose_output(scene))


def _locate_fillable_cells(grid: np.ndarray) -> dict[tuple[int, int], int]:
    _g, scene = _split_grid(grid)
    horiz, vert = _collect_conflicts(grid)
    if not horiz and not vert:
        return {}

    h_cells: dict[int, set[tuple[int, int]]] = {i: set() for i in range(len(horiz))}
    v_cells: dict[int, set[tuple[int, int]]] = {j: set() for j in range(len(vert))}
    pairs_by_cell: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for hi, h_conflict in enumerate(horiz):
        cols = set(h_conflict.candidates)
        if not cols:
            continue
        for vi, v_conflict in enumerate(vert):
            if h_conflict.color != v_conflict.color:
                continue
            if h_conflict.fixed not in v_conflict.candidates:
                continue
            if v_conflict.fixed not in cols:
                continue
            cell = (h_conflict.fixed, v_conflict.fixed)
            if int(scene[cell]) != BLACK:
                continue
            h_cells[hi].add(cell)
            v_cells[vi].add(cell)
            pairs_by_cell.setdefault(cell, []).append((hi, vi))

    fills: dict[tuple[int, int], int] = {}
    for cell, pairs in pairs_by_cell.items():
        if any(len(h_cells[hi]) == 1 and len(v_cells[vi]) == 1 for hi, vi in pairs):
            color = horiz[pairs[0][0]].color
            fills[cell] = color
    return fills


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore blacked-out interior pixels by cross-locating row and column conflicts against a first-seen projection ring.

    Input structure
    - The outermost border is a projection ring: each non-corner border cell stores the first non-black color visible when looking inward along its row or column; corners are always black.
    - Inside the ring is a black-background scene with solid occupied regions, but some occupied pixels have been erased to black.
    - The ring was computed from the complete scene before erasure, so some ring rays now disagree with the visible interior.

    Rule
    - For every border ray, compare the ring color with the current first visible non-black pixel on that inward ray.
    - A non-black ring entry that no longer matches marks a contradiction on that row or column; the missing pixel must lie on the leading black prefix of that ray.
    - Intersect a contradictory row-ray with a contradictory column-ray of the same recorded color, restore the uniquely supported black intersection pixel, and repeat until all ring entries agree again.

    Key insight
    - The border is not decorative: it is a frozen four-direction archive of the complete scene, so solving the puzzle means inverting those first-seen projections and using row-column contradiction crossings to recover the erased colors.
    """
    out = np.asarray(grid, dtype=int).copy()

    while True:
        fills = _locate_fillable_cells(out)
        if not fills:
            break
        for (r, c), color in fills.items():
            out[r + 1, c + 1] = color

    assert _ring_matches_archive(out), "input does not admit a unique full restoration"
    return out


GEN_INNER_RANGE = (8, 21)
GEN_OBJECT_COUNT = (3, 7)
GEN_RECT_H = (1, 6)
GEN_RECT_W = (1, 6)
GEN_BORDER_PAD = 1
GEN_OBJECT_GAP = 1
GEN_MISSING_COUNT = (1, 5)
GEN_ATTEMPTS = 500


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _paint_rectangle(scene: np.ndarray, rect: tuple[int, int, int, int], palette: list[int]) -> None:
    r0, c0, r1, c1 = rect
    patch = np.random.choice(palette, size=(r1 - r0 + 1, c1 - c0 + 1))
    scene[r0 : r1 + 1, c0 : c1 + 1] = patch


def _orientation_info(name: str) -> tuple[str, str]:
    if name == "tl":
        return "top", "left"
    if name == "tr":
        return "top", "right"
    if name == "bl":
        return "bottom", "left"
    if name == "br":
        return "bottom", "right"
    raise ValueError(name)


def _next_color_after(scene: np.ndarray, r: int, c: int, side: str) -> int:
    h, w = scene.shape
    if side == "left":
        cc = c + 1
        while cc < w:
            color = int(scene[r, cc])
            if color != BLACK:
                return color
            cc += 1
        return BLACK
    if side == "right":
        cc = c - 1
        while cc >= 0:
            color = int(scene[r, cc])
            if color != BLACK:
                return color
            cc -= 1
        return BLACK
    if side == "top":
        rr = r + 1
        while rr < h:
            color = int(scene[rr, c])
            if color != BLACK:
                return color
            rr += 1
        return BLACK
    if side == "bottom":
        rr = r - 1
        while rr >= 0:
            color = int(scene[rr, c])
            if color != BLACK:
                return color
            rr -= 1
        return BLACK
    raise ValueError(side)


def _is_first_from_side(scene: np.ndarray, r: int, c: int, side: str) -> bool:
    if side == "left":
        return not np.any(scene[r, :c] != BLACK)
    if side == "right":
        return not np.any(scene[r, c + 1 :] != BLACK)
    if side == "top":
        return not np.any(scene[:r, c] != BLACK)
    if side == "bottom":
        return not np.any(scene[r + 1 :, c] != BLACK)
    raise ValueError(side)


def _candidate_cells(scene: np.ndarray) -> list[tuple[int, int, str]]:
    h, w = scene.shape
    candidates: list[tuple[int, int, str]] = []
    for r in range(h):
        for c in range(w):
            color = int(scene[r, c])
            if color == BLACK:
                continue
            for name in ("tl", "tr", "bl", "br"):
                v_side, h_side = _orientation_info(name)
                if not _is_first_from_side(scene, r, c, v_side):
                    continue
                if not _is_first_from_side(scene, r, c, h_side):
                    continue
                if _next_color_after(scene, r, c, v_side) == color:
                    continue
                if _next_color_after(scene, r, c, h_side) == color:
                    continue
                candidates.append((r, c, name))
    return candidates


def _build_scene() -> np.ndarray | None:
    h = random.randint(*GEN_INNER_RANGE)
    w = random.randint(*GEN_INNER_RANGE)
    scene = np.zeros((h, w), dtype=int)
    rects: list[tuple[int, int, int, int]] = []

    n_objects = random.randint(*GEN_OBJECT_COUNT)
    for _ in range(n_objects):
        placed = False
        for _ in range(80):
            rh = random.randint(*GEN_RECT_H)
            rw = random.randint(*GEN_RECT_W)
            if rh > h - 2 * GEN_BORDER_PAD or rw > w - 2 * GEN_BORDER_PAD:
                continue
            r0 = random.randint(GEN_BORDER_PAD, h - GEN_BORDER_PAD - rh)
            c0 = random.randint(GEN_BORDER_PAD, w - GEN_BORDER_PAD - rw)
            rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if any(_rectangles_overlap(rect, other, pad=GEN_OBJECT_GAP) for other in rects):
                continue
            palette_size = random.randint(2, min(5, len(COLORS)))
            palette = random.sample(list(COLORS), palette_size)
            _paint_rectangle(scene, rect, palette)
            rects.append(rect)
            placed = True
            break
        if not placed:
            if len(rects) < 2:
                return None
            break

    if len(rects) < 2:
        return None
    return scene


def _select_missing(scene: np.ndarray) -> list[tuple[int, int]]:
    candidates = _candidate_cells(scene)
    random.shuffle(candidates)
    target = random.randint(*GEN_MISSING_COUNT)
    chosen: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    used_row_rays: set[tuple[str, int]] = set()
    used_col_rays: set[tuple[str, int]] = set()

    for r, c, name in candidates:
        v_side, h_side = _orientation_info(name)
        if r in used_rows or c in used_cols:
            continue
        if (h_side, r) in used_row_rays or (v_side, c) in used_col_rays:
            continue
        chosen.append((r, c))
        used_rows.add(r)
        used_cols.add(c)
        used_row_rays.add((h_side, r))
        used_col_rays.add((v_side, c))
        if len(chosen) >= target:
            break
    return chosen


def _compose_input(scene: np.ndarray, missing: list[tuple[int, int]]) -> np.ndarray:
    ring = _projection_ring(scene)
    inp = ring.copy()
    damaged = scene.copy()
    for r, c in missing:
        damaged[r, c] = BLACK
    inp[1:-1, 1:-1] = damaged
    return inp


def _compose_output(scene: np.ndarray) -> np.ndarray:
    ring = _projection_ring(scene)
    out = ring.copy()
    out[1:-1, 1:-1] = scene
    return out


def _validate_instance(scene: np.ndarray, missing: list[tuple[int, int]]) -> np.ndarray | None:
    if not missing:
        return None

    inp = _compose_input(scene, missing)
    expected = _compose_output(scene)
    fillable = _locate_fillable_cells(inp)
    if set(fillable) != set(missing):
        return None
    if any(int(scene[r, c]) != fillable[(r, c)] for r, c in missing):
        return None

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return None
    if not _ring_matches_archive(solved):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        scene = _build_scene()
        if scene is None:
            continue
        missing = _select_missing(scene)
        inp = _validate_instance(scene, missing)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid projection-ring restoration task")
