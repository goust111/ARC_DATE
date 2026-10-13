from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
GEN_ATTEMPTS = 500
CROSS_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def _bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _cross_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r + dr, c + dc) for dr, dc in CROSS_OFFSETS}


def _separator_col(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    assert len(candidates) == 1, "expected exactly one full-height grey separator"
    sep = int(candidates[0])
    assert 2 * sep + 1 == w, "separator must split equal-width halves"
    return sep


def _is_template_component(points: list[tuple[int, int]]) -> bool:
    if len(points) != 5:
        return False
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    if r1 - r0 != 2 or c1 - c0 != 2:
        return False
    normalized = {(r - r0, c - c0) for r, c in points}
    return normalized == {(1, 1), (0, 1), (2, 1), (1, 0), (1, 2)}


def _find_template(grid: np.ndarray) -> set[tuple[int, int]]:
    comps = _components(grid == RED)
    templates = [set(comp) for comp in comps if _is_template_component(comp)]
    assert len(templates) == 1, "expected exactly one red 3x3 cross template"
    return templates[0]


def _solve(
    grid: np.ndarray,
    *,
    remove_template: bool,
    stamp_mode: str,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int).copy()
    bg = _bg(g)
    sep = _separator_col(g)

    if remove_template:
        for r, c in _find_template(g):
            g[r, c] = bg

    left = g[:, :sep]
    right = np.fliplr(g[:, sep + 1 :])
    left_blue = left == BLUE
    right_blue = right == BLUE
    left_fg = left != bg
    right_fg = right != bg
    overlap = left_blue & right_blue

    out = np.full(left.shape, bg, dtype=int)
    out[left_fg] = left[left_fg]
    out[right_fg] = right[right_fg]

    if stamp_mode == "none":
        return out

    if stamp_mode == "overlap":
        centers = np.argwhere(overlap)
    elif stamp_mode == "union":
        centers = np.argwhere(out == BLUE)
    else:
        raise ValueError(f"unknown stamp mode: {stamp_mode}")

    h, w = out.shape
    for r, c in centers:
        for dr, dc in CROSS_OFFSETS:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Fold across the grey separator, find mirrored blue overlaps, and stamp a red cross on each overlap center.

    Input structure:
    - A black background contains equal-width left and right blue-pixel halves separated by one full-height grey column.
    - A single red 3x3 cross template appears away from the separator as an instruction sprite and is not part of the folded scene.
    - Valid overlap centers lie at least one cell inside the folded half so a radius-1 cross can fit.

    Rule:
    - Identify the full-height grey separator and remove it from the coordinate system.
    - Ignore the external red cross template, mirror the right half horizontally, and align it onto the left-half coordinates.
    - Keep every position that is blue on either side as blue, but record the positions that are blue on both aligned halves.
    - For each such overlap position, recolor the center and its four orthogonal neighbors red, overwriting any existing blue.

    Key insight:
    - The red stamp centers are not explicitly marked in the input; they are the blue pixels that coincide only after the right half is mirrored and folded onto the left half.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    return _solve(g, remove_template=True, stamp_mode="overlap")


def _template_cells(kind: str, shape: tuple[int, int], half_w: int) -> set[tuple[int, int]]:
    h, _ = shape
    sep = half_w
    if kind == "bottom_left":
        center = (h - 2, 1)
        return _cross_cells(center)
    if kind == "top_right":
        center = (1, 2 * sep - 1)
        return _cross_cells(center)
    raise ValueError(f"unknown template placement: {kind}")


def _template_reserved(kind: str, h: int, w: int) -> set[tuple[int, int]]:
    if kind == "bottom_left":
        return _cross_cells((h - 2, 1))
    if kind == "top_right":
        return {(1, 1), (0, 1), (2, 1), (1, 0), (1, 2)}
    raise ValueError(f"unknown template placement: {kind}")


def _pick_overlap_centers(
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1) if (r, c) not in blocked]
    random.shuffle(candidates)
    centers: list[tuple[int, int]] = []
    used_stamp_cells: set[tuple[int, int]] = set()
    for pos in candidates:
        stamp = _cross_cells(pos)
        if stamp & blocked:
            continue
        if stamp & used_stamp_cells:
            continue
        centers.append(pos)
        used_stamp_cells |= stamp
        if len(centers) == count:
            return centers
    return None


def _build_expected(
    h: int,
    w: int,
    overlap_centers: list[tuple[int, int]],
    left_only: list[tuple[int, int]],
    right_only: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full((h, w), BLACK, dtype=int)
    for r, c in overlap_centers + left_only + right_only:
        out[r, c] = BLUE
    for center in overlap_centers:
        for r, c in _cross_cells(center):
            out[r, c] = RED
    return out


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    overlap_centers: list[tuple[int, int]],
    template_kind: str,
) -> bool:
    if np.array_equal(inp[:, : inp.shape[1] // 2], inp[:, inp.shape[1] // 2 + 1 :]):
        return False

    sep = _separator_col(inp)
    if not np.all(inp[:, sep] == GREY):
        return False

    template = _find_template(inp)
    half_w = sep
    if template != _template_cells(template_kind, inp.shape, half_w):
        return False

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False
    if np.array_equal(expected, _solve(inp, remove_template=True, stamp_mode="none")):
        return False
    if np.array_equal(expected, _solve(inp, remove_template=False, stamp_mode="overlap")):
        return False
    if np.array_equal(expected, _solve(inp, remove_template=True, stamp_mode="union")):
        return False

    overlap_mask = solved == RED
    for r, c in overlap_centers:
        if not _cross_cells((r, c)).issubset(set(map(tuple, np.argwhere(overlap_mask)))):
            return False
    if len(overlap_centers) < 2 or len(overlap_centers) > 4:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(11, 15)
    w = random.randint(6, 8)
    full_w = 2 * w + 1
    template_kind = random.choice(["bottom_left", "top_right"])

    blocked = _template_reserved(template_kind, h, w)
    overlap_count = random.choice([2, 3, 4])
    overlap_centers = _pick_overlap_centers(h, w, blocked, overlap_count)
    if overlap_centers is None:
        return None

    forbidden = set(blocked)
    for center in overlap_centers:
        forbidden |= _cross_cells(center)

    pool = [(r, c) for r in range(1, h - 1) for c in range(w) if (r, c) not in forbidden]
    random.shuffle(pool)

    max_extra = min(7, len(pool) // 2)
    if max_extra < 3:
        return None
    left_only_count = random.randint(3, min(7, 13 - overlap_count))
    right_only_count = random.randint(3, min(7, 13 - overlap_count))
    if len(pool) < left_only_count + right_only_count:
        return None

    left_only = pool[:left_only_count]
    right_only = pool[left_only_count : left_only_count + right_only_count]

    left = np.full((h, w), BLACK, dtype=int)
    right_work = np.full((h, w), BLACK, dtype=int)
    for r, c in overlap_centers:
        left[r, c] = BLUE
        right_work[r, c] = BLUE
    for r, c in left_only:
        left[r, c] = BLUE
    for r, c in right_only:
        right_work[r, c] = BLUE

    grid = np.full((h, full_w), BLACK, dtype=int)
    grid[:, :w] = left
    grid[:, w] = GREY
    grid[:, w + 1 :] = np.fliplr(right_work)
    for r, c in _template_cells(template_kind, grid.shape, w):
        grid[r, c] = RED

    expected = _build_expected(h, w, overlap_centers, left_only, right_only)
    if np.array_equal(expected, left):
        return None
    if not _semantic_checks(grid, expected, overlap_centers, template_kind):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fold-and-stamp input")
