from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


Rect = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(grid[r0, c0]) == bg:
                continue

            color = int(grid[r0, c0])
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []

            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r1, c1 = arr.min(axis=0)
            r2, c2 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r1), int(c1), int(r2), int(c2)),
                    "area": int(arr.shape[0]),
                }
            )

    return components


def _rect_area(rect: Rect) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _in_rect(r: int, c: int, rect: Rect | None) -> bool:
    if rect is None:
        return False
    r0, c0, r1, c1 = rect
    return r0 <= r <= r1 and c0 <= c <= c1


def _rect_in_one_quadrant(rect: Rect, shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = rect
    h, w = shape
    mid_r = (h - 1) / 2.0
    mid_c = (w - 1) / 2.0
    row_side0 = np.sign(r0 - mid_r)
    row_side1 = np.sign(r1 - mid_r)
    col_side0 = np.sign(c0 - mid_c)
    col_side1 = np.sign(c1 - mid_c)
    return row_side0 != 0 and row_side0 == row_side1 and col_side0 != 0 and col_side0 == col_side1


def _add_mapping(mapping: dict[int, int], inverse: dict[int, int], src: int, dst: int, bg: int) -> bool:
    if (src == bg) != (dst == bg):
        return False
    prev = mapping.get(src)
    if prev is not None and prev != dst:
        return False
    prev_inv = inverse.get(dst)
    if prev_inv is not None and prev_inv != src:
        return False
    mapping[src] = dst
    inverse[dst] = src
    return True


def _infer_maps(
    grid: np.ndarray, bg: int, *, skip_rect: Rect | None = None
) -> tuple[dict[int, int], dict[int, int], dict[int, int], dict[int, int]] | None:
    h, w = grid.shape
    h_map: dict[int, int] = {}
    h_inv: dict[int, int] = {}
    v_map: dict[int, int] = {}
    v_inv: dict[int, int] = {}

    for r in range(h):
        for c in range((w + 1) // 2):
            mc = w - 1 - c
            if _in_rect(r, c, skip_rect) or _in_rect(r, mc, skip_rect):
                continue
            if not _add_mapping(h_map, h_inv, int(grid[r, c]), int(grid[r, mc]), bg):
                return None

    for r in range((h + 1) // 2):
        mr = h - 1 - r
        for c in range(w):
            if _in_rect(r, c, skip_rect) or _in_rect(mr, c, skip_rect):
                continue
            if not _add_mapping(v_map, v_inv, int(grid[r, c]), int(grid[mr, c]), bg):
                return None

    if h_map.get(bg, bg) != bg or h_inv.get(bg, bg) != bg:
        return None
    if v_map.get(bg, bg) != bg or v_inv.get(bg, bg) != bg:
        return None
    h_map[bg] = bg
    h_inv[bg] = bg
    v_map[bg] = bg
    v_inv[bg] = bg
    return h_map, h_inv, v_map, v_inv


def _horizontal_to_hidden(color: int, c: int, mirror_c: int, h_map: dict[int, int], h_inv: dict[int, int]) -> int | None:
    if c < mirror_c:
        return h_inv.get(color)
    if c > mirror_c:
        return h_map.get(color)
    return color


def _vertical_to_hidden(color: int, r: int, mirror_r: int, v_map: dict[int, int], v_inv: dict[int, int]) -> int | None:
    if r < mirror_r:
        return v_inv.get(color)
    if r > mirror_r:
        return v_map.get(color)
    return color


def _reconstruct_patch(
    grid: np.ndarray,
    rect: Rect,
    h_map: dict[int, int],
    h_inv: dict[int, int],
    v_map: dict[int, int],
    v_inv: dict[int, int],
) -> np.ndarray | None:
    h, w = grid.shape
    r0, c0, r1, c1 = rect
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            mr = h - 1 - r
            mc = w - 1 - c
            choices: list[int] = []

            h_val = _horizontal_to_hidden(int(grid[r, mc]), c, mc, h_map, h_inv)
            if h_val is not None:
                choices.append(h_val)

            v_val = _vertical_to_hidden(int(grid[mr, c]), r, mr, v_map, v_inv)
            if v_val is not None:
                choices.append(v_val)

            diag_color = int(grid[mr, mc])
            step = _horizontal_to_hidden(diag_color, c, mc, h_map, h_inv)
            if step is not None:
                step = _vertical_to_hidden(step, r, mr, v_map, v_inv)
                if step is not None:
                    choices.append(step)
            step = _vertical_to_hidden(diag_color, r, mr, v_map, v_inv)
            if step is not None:
                step = _horizontal_to_hidden(step, c, mc, h_map, h_inv)
                if step is not None:
                    choices.append(step)

            if not choices or len(set(choices)) != 1:
                return None
            patch[r - r0, c - c0] = choices[0]

    return patch


def _is_consistent(grid: np.ndarray, bg: int) -> bool:
    return _infer_maps(grid, bg) is not None


def _solve_candidate(grid: np.ndarray, bg: int, rect: Rect) -> np.ndarray | None:
    maps = _infer_maps(grid, bg, skip_rect=rect)
    if maps is None:
        return None

    patch = _reconstruct_patch(grid, rect, *maps)
    if patch is None:
        return None

    completed = grid.copy()
    r0, c0, r1, c1 = rect
    completed[r0 : r1 + 1, c0 : c1 + 1] = patch
    if not _is_consistent(completed, bg):
        return None
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the rectangle hidden under a monochrome occluder by undoing two independent mirror color maps.

    Input structure:
    - Aside from one solid rectangular occluder, the scene is shape-symmetric under
      both left-right and top-bottom mirroring.
    - Horizontal mirror pairs follow one fixed color permutation, and vertical
      mirror pairs follow another fixed permutation.
    - The occluder lies fully inside one quadrant, so every hidden cell has three
      visible mirror counterparts outside the block.

    Rule:
    - Find the occluding monochrome rectangle.
    - Reconstruct the horizontal and vertical color maps from visible mirror pairs
      that do not touch the occluder.
    - For each hidden cell, convert a visible horizontal mirror, vertical mirror,
      or diagonal mirror through the appropriate map or map composition, and output
      the recovered rectangle crop.

    Key insight:
    - The geometry is still doubly mirrored, but colors are not copied directly:
      each axis has its own substitution table, so the hidden patch is recovered by
      combining spatial mirroring with axis-specific color inversion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    components = _find_components(g, bg)
    candidates = []
    for comp in components:
        rect = comp["bbox"]
        if int(comp["area"]) != _rect_area(rect):
            continue
        if not _rect_in_one_quadrant(rect, g.shape):
            continue
        candidates.append(rect)

    candidates = sorted(set(candidates))
    solutions: list[tuple[Rect, np.ndarray]] = []
    for rect in candidates:
        patch = _solve_candidate(g, bg, rect)
        if patch is not None:
            solutions.append((rect, patch))

    assert len(solutions) == 1, "expected exactly one valid occluder rectangle"
    return solutions[0][1]


_ALL_COLORS = list(range(10))
_MAP_SHAPES = [(2, 2), (2, 3), (3, 2)]
_RECT_SHAPES = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1)]
_GEN_ATTEMPTS = 400


def _non_identity_permutations(n: int) -> list[tuple[int, ...]]:
    return [p for p in itertools.permutations(range(n)) if any(i != p[i] for i in range(n))]


def _choose_maps(bg: int, occluder: int) -> tuple[list[int], dict[int, int], dict[int, int]]:
    rows, cols = random.choice(_MAP_SHAPES)
    usable = [c for c in _ALL_COLORS if c not in {bg, occluder}]
    picked = random.sample(usable, rows * cols)
    random.shuffle(picked)
    table = np.asarray(picked, dtype=int).reshape(rows, cols)

    row_perm = random.choice(_non_identity_permutations(rows))
    col_perm = random.choice(_non_identity_permutations(cols))

    h_map = {bg: bg}
    v_map = {bg: bg}
    for r in range(rows):
        for c in range(cols):
            color = int(table[r, c])
            h_map[color] = int(table[row_perm[r], c])
            v_map[color] = int(table[r, col_perm[c]])
    return picked, h_map, v_map


def _place_anchor_cells(base: np.ndarray, bg: int, colors: list[int]) -> bool:
    spots = [(r, c) for r in range(base.shape[0]) for c in range(base.shape[1])]
    random.shuffle(spots)
    if len(spots) < len(colors):
        return False
    for color, (r, c) in zip(colors, spots):
        base[r, c] = color
    return True


def _fill_extra_rectangles(base: np.ndarray, bg: int, colors: list[int]) -> None:
    qh, qw = base.shape
    target_area = random.randint(len(colors) + 2, max(len(colors) + 3, int(round(0.45 * qh * qw))))
    current = int(np.count_nonzero(base != bg))

    for _ in range(80):
        if current >= target_area:
            return
        rh, rw = random.choice(_RECT_SHAPES)
        if rh > qh or rw > qw:
            continue
        r0 = random.randint(0, qh - rh)
        c0 = random.randint(0, qw - rw)
        if np.any(base[r0 : r0 + rh, c0 : c0 + rw] != bg):
            continue
        color = random.choice(colors)
        base[r0 : r0 + rh, c0 : c0 + rw] = color
        current += rh * rw


def _build_base_quadrant(qh: int, qw: int, bg: int, colors: list[int]) -> np.ndarray | None:
    base = np.full((qh, qw), bg, dtype=int)
    if not _place_anchor_cells(base, bg, colors):
        return None
    _fill_extra_rectangles(base, bg, colors)
    return base


def _apply_maps(base: np.ndarray, bg: int, h_map: dict[int, int], v_map: dict[int, int]) -> np.ndarray:
    qh, qw = base.shape
    full = np.full((2 * qh, 2 * qw), bg, dtype=int)
    for r in range(qh):
        for c in range(qw):
            color = int(base[r, c])
            mr = 2 * qh - 1 - r
            mc = 2 * qw - 1 - c
            full[r, c] = color
            full[r, mc] = h_map[color]
            full[mr, c] = v_map[color]
            full[mr, mc] = v_map[h_map[color]]
    return full


def _quadrant_rects(shape: tuple[int, int]) -> list[Rect]:
    h, w = shape
    qh, qw = h // 2, w // 2
    quadrants = [
        (0, 0, qh - 1, qw - 1),
        (0, qw, qh - 1, w - 1),
        (qh, 0, h - 1, qw - 1),
        (qh, qw, h - 1, w - 1),
    ]

    rects: list[Rect] = []
    for qr0, qc0, qr1, qc1 in quadrants:
        max_h = min(4, qr1 - qr0 + 1)
        max_w = min(4, qc1 - qc0 + 1)
        for rh in range(2, max_h + 1):
            for rw in range(2, max_w + 1):
                for r0 in range(qr0, qr1 - rh + 2):
                    for c0 in range(qc0, qc1 - rw + 2):
                        rects.append((r0, c0, r0 + rh - 1, c0 + rw - 1))
    random.shuffle(rects)
    return rects


def _patch_is_informative(full: np.ndarray, rect: Rect, bg: int) -> bool:
    r0, c0, r1, c1 = rect
    patch = full[r0 : r1 + 1, c0 : c1 + 1]
    uniq = set(map(int, np.unique(patch)))
    return len(uniq) >= 2 and bg in uniq and any(color != bg for color in uniq)


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    bg = random.choice(_ALL_COLORS)
    occluder = random.choice([c for c in _ALL_COLORS if c != bg])
    colors, h_map, v_map = _choose_maps(bg, occluder)

    qh = random.randint(5, 8)
    qw = random.randint(5, 8)
    base = _build_base_quadrant(qh, qw, bg, colors)
    if base is None:
        return None

    full = _apply_maps(base, bg, h_map, v_map)
    if _get_bg(full) != bg:
        return None

    for rect in _quadrant_rects(full.shape):
        if not _patch_is_informative(full, rect, bg):
            continue
        inp = full.copy()
        r0, c0, r1, c1 = rect
        inp[r0 : r1 + 1, c0 : c1 + 1] = occluder
        expected = full[r0 : r1 + 1, c0 : c1 + 1].copy()
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp, expected
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid double-mirror recoloring task")
