from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
MAROON = 9


def _find_axis(g: np.ndarray) -> tuple[str, int]:
    h, w = g.shape
    candidates: list[tuple[str, int]] = []

    for r in range(h):
        if np.all(g[r, :] == YELLOW):
            assert 2 * r == h - 1, "yellow row axis must be centered"
            candidates.append(("row", r))

    for c in range(w):
        if np.all(g[:, c] == YELLOW):
            assert 2 * c == w - 1, "yellow column axis must be centered"
            candidates.append(("col", c))

    assert len(candidates) == 1, "expected exactly one centered yellow row or column axis"
    return candidates[0]


def _mirror_cell(axis_kind: str, axis_index: int, r: int, c: int) -> tuple[int, int]:
    if axis_kind == "row":
        return 2 * axis_index - r, c
    assert axis_kind == "col"
    return r, 2 * axis_index - c


def _side_counts(g: np.ndarray, axis_kind: str, axis_index: int) -> tuple[int, int]:
    if axis_kind == "row":
        before = int(np.count_nonzero(g[:axis_index, :] != BLACK))
        after = int(np.count_nonzero(g[axis_index + 1 :, :] != BLACK))
    else:
        before = int(np.count_nonzero(g[:, :axis_index] != BLACK))
        after = int(np.count_nonzero(g[:, axis_index + 1 :] != BLACK))
    return before, after


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the smaller side of a yellow-axis mirror scene and mark its original wrong pixels maroon.

    Input structure:
    - Black background with one centered single-pixel-wide yellow row or column forming the mirror axis.
    - On the two sides of the axis are colored 4-connected solid objects that are nearly mirrored.
    - One side has strictly more non-background pixels than the other; this larger side is the reference side.

    Rule:
    - Count non-background pixels on the two sides of the yellow axis to identify the reference and deficient sides.
    - For each deficient-side position, compare it with the reference-side pixel reflected across the axis.
    - If the deficient cell is black and its reference mate is foreground, fill it with the reference color.
    - If the deficient cell is foreground but its reference mate is black or a different color, recolor that deficient cell maroon.
    - Keep the yellow axis and the reference side unchanged.

    Key insight:
    - The yellow line gives the reflection axis, but the reference direction must be inferred by side population; repairs and maroon error marking are both required.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    axis_kind, axis_index = _find_axis(g)
    before_count, after_count = _side_counts(g, axis_kind, axis_index)
    assert before_count != after_count, "one side must have strictly more foreground pixels"

    deficient_before = before_count < after_count
    out = g.copy()
    h, w = g.shape

    if axis_kind == "row":
        rows = range(axis_index) if deficient_before else range(axis_index + 1, h)
        for r in rows:
            for c in range(w):
                rr, cc = _mirror_cell(axis_kind, axis_index, r, c)
                ref = int(g[rr, cc])
                cur = int(g[r, c])
                if ref == BLACK:
                    if cur != BLACK:
                        out[r, c] = MAROON
                elif cur == BLACK:
                    out[r, c] = ref
                elif cur != ref:
                    out[r, c] = MAROON
    else:
        cols = range(axis_index) if deficient_before else range(axis_index + 1, w)
        for r in range(h):
            for c in cols:
                rr, cc = _mirror_cell(axis_kind, axis_index, r, c)
                ref = int(g[rr, cc])
                cur = int(g[r, c])
                if ref == BLACK:
                    if cur != BLACK:
                        out[r, c] = MAROON
                elif cur == BLACK:
                    out[r, c] = ref
                elif cur != ref:
                    out[r, c] = MAROON

    return out


# --------------- generator ---------------

_GEN_COLORS = (1, 2, 3, 5, 6, 7, 8)
_GEN_ATTEMPTS = 500
_PLACE_ATTEMPTS = 180
_DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _draw_axis(g: np.ndarray, axis_kind: str, axis_index: int) -> None:
    if axis_kind == "row":
        g[axis_index, :] = YELLOW
    else:
        g[:, axis_index] = YELLOW


def _side_mask(shape: tuple[int, int], axis_kind: str, axis_index: int, side: str) -> np.ndarray:
    h, w = shape
    rows, cols = np.indices((h, w))
    if axis_kind == "row":
        return rows < axis_index if side == "before" else rows > axis_index
    return cols < axis_index if side == "before" else cols > axis_index


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for dr, dc in _DIRS8:
        src_r0 = max(0, -dr)
        src_r1 = mask.shape[0] - max(0, dr)
        src_c0 = max(0, -dc)
        src_c1 = mask.shape[1] - max(0, dc)
        dst_r0 = max(0, dr)
        dst_r1 = mask.shape[0] - max(0, -dr)
        dst_c0 = max(0, dc)
        dst_c1 = mask.shape[1] - max(0, -dc)
        out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _random_shape() -> np.ndarray:
    kind = random.choice(("rect", "notch", "ell", "stair", "bar"))

    if kind == "rect":
        mask = np.ones((random.randint(2, 4), random.randint(2, 5)), dtype=bool)
    elif kind == "notch":
        mask = np.ones((random.randint(3, 5), random.randint(3, 5)), dtype=bool)
        corner = random.randrange(4)
        if corner == 0:
            mask[0, 0] = False
        elif corner == 1:
            mask[0, -1] = False
        elif corner == 2:
            mask[-1, 0] = False
        else:
            mask[-1, -1] = False
    elif kind == "ell":
        h, w = random.randint(3, 5), random.randint(3, 5)
        thick = random.randint(1, 2)
        mask = np.zeros((h, w), dtype=bool)
        mask[:, :thick] = True
        mask[-thick:, :] = True
    elif kind == "stair":
        h, w = random.randint(3, 5), random.randint(3, 5)
        mask = np.zeros((h, w), dtype=bool)
        for r in range(h):
            width = 1 + round(r * (w - 1) / max(1, h - 1))
            mask[r, :width] = True
    else:
        if random.random() < 0.5:
            mask = np.ones((random.randint(1, 2), random.randint(3, 6)), dtype=bool)
        else:
            mask = np.ones((random.randint(3, 6), random.randint(1, 2)), dtype=bool)

    mask = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return _trim_mask(mask)


def _can_place(
    shape_mask: np.ndarray,
    r0: int,
    c0: int,
    side_allowed: np.ndarray,
    occupied: np.ndarray,
) -> bool:
    h, w = shape_mask.shape
    region_allowed = side_allowed[r0 : r0 + h, c0 : c0 + w]
    if region_allowed.shape != shape_mask.shape or bool((shape_mask & ~region_allowed).any()):
        return False
    blocked = _dilate(occupied)
    return not bool((shape_mask & blocked[r0 : r0 + h, c0 : c0 + w]).any())


def _paint_mask(g: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> set[tuple[int, int]]:
    coords: set[tuple[int, int]] = set()
    for rr, cc in np.argwhere(mask):
        r = r0 + int(rr)
        c = c0 + int(cc)
        g[r, c] = color
        coords.add((r, c))
    return coords


def _place_objects(
    g: np.ndarray,
    axis_kind: str,
    axis_index: int,
    base_side: str,
    count: int,
) -> list[dict[str, object]] | None:
    side_allowed = _side_mask(g.shape, axis_kind, axis_index, base_side)
    occupied = g != BLACK
    colors = random.sample(_GEN_COLORS, count)
    objects: list[dict[str, object]] = []

    for color in colors:
        placed = False
        for _ in range(_PLACE_ATTEMPTS):
            mask = _random_shape()
            mh, mw = mask.shape
            if mh > g.shape[0] or mw > g.shape[1]:
                continue
            r0 = random.randint(0, g.shape[0] - mh)
            c0 = random.randint(0, g.shape[1] - mw)
            if not _can_place(mask, r0, c0, side_allowed, occupied):
                continue
            coords = _paint_mask(g, mask, r0, c0, color)
            for r, c in coords:
                occupied[r, c] = True
            objects.append({"color": color, "base": coords})
            placed = True
            break
        if not placed:
            return None
    return objects


def _connected_after_removal(cells: set[tuple[int, int]], removed: tuple[int, int]) -> bool:
    remaining = cells - {removed}
    if not remaining:
        return False
    start = next(iter(remaining))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in _DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in remaining and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(remaining)


def _choose_missing_cells(
    object_cells: list[set[tuple[int, int]]],
    target: int,
) -> list[tuple[int, int]] | None:
    missing: list[tuple[int, int]] = []
    for _ in range(target):
        choices: list[tuple[int, tuple[int, int]]] = []
        for idx, cells in enumerate(object_cells):
            if len(cells) <= 2:
                continue
            for cell in cells:
                if _connected_after_removal(cells, cell):
                    choices.append((idx, cell))
        if not choices:
            break
        idx, cell = random.choice(choices)
        object_cells[idx].remove(cell)
        missing.append(cell)
    return missing if missing else None


def _extra_candidates(
    g: np.ndarray,
    axis_kind: str,
    axis_index: int,
    deficient_side: str,
) -> list[tuple[int, int]]:
    allowed = _side_mask(g.shape, axis_kind, axis_index, deficient_side)
    candidates: list[tuple[int, int]] = []
    h, w = g.shape
    for r, c in np.argwhere(allowed & (g == BLACK)):
        rr, cc = _mirror_cell(axis_kind, axis_index, int(r), int(c))
        if int(g[rr, cc]) != BLACK:
            continue
        adjacent = False
        for dr, dc in _DIRS4:
            nr = int(r) + dr
            nc = int(c) + dc
            if 0 <= nr < h and 0 <= nc < w and allowed[nr, nc] and int(g[nr, nc]) not in {BLACK, YELLOW}:
                adjacent = True
                break
        if adjacent:
            candidates.append((int(r), int(c)))
    return candidates


def _sample_layout() -> tuple[int, int, str, int]:
    axis_kind = random.choice(("row", "col"))
    if axis_kind == "row":
        h = random.choice(range(15, 30, 2))
        w = random.randint(12, 29)
        axis_index = h // 2
    else:
        h = random.randint(12, 29)
        w = random.choice(range(15, 30, 2))
        axis_index = w // 2
    return h, w, axis_kind, axis_index


def _gen_attempt() -> np.ndarray | None:
    h, w, axis_kind, axis_index = _sample_layout()
    grid = np.full((h, w), BLACK, dtype=int)
    _draw_axis(grid, axis_kind, axis_index)

    base_side = random.choice(("before", "after"))
    deficient_side = "after" if base_side == "before" else "before"
    object_count = random.randint(2, 5)
    objects = _place_objects(grid, axis_kind, axis_index, base_side, object_count)
    if objects is None:
        return None

    mirrored_sets: list[set[tuple[int, int]]] = []
    for obj in objects:
        color = int(obj["color"])
        mirrored: set[tuple[int, int]] = set()
        for r, c in obj["base"]:
            rr, cc = _mirror_cell(axis_kind, axis_index, int(r), int(c))
            if grid[rr, cc] != BLACK:
                return None
            grid[rr, cc] = color
            mirrored.add((rr, cc))
        mirrored_sets.append(mirrored)

    base_count, _ = _side_counts(grid, axis_kind, axis_index)
    if base_side == "after":
        _, base_count = _side_counts(grid, axis_kind, axis_index)
    if base_count < 10:
        return None

    inp = grid.copy()
    max_missing = min(6, max(2, base_count // 3), base_count - 2)
    if max_missing < 2:
        return None
    missing_target = random.randint(2, max_missing)
    working_sets = [set(cells) for cells in mirrored_sets]
    missing = _choose_missing_cells(working_sets, missing_target)
    if missing is None:
        return None
    for r, c in missing:
        inp[r, c] = BLACK

    remaining = [cell for cells in working_sets for cell in cells]
    if not remaining:
        return None
    wrong_count = random.randint(1, min(3, len(remaining)))
    wrong_cells = random.sample(remaining, wrong_count)
    for r, c in wrong_cells:
        true_color = int(grid[r, c])
        choices = [color for color in _GEN_COLORS if color != true_color]
        inp[r, c] = random.choice(choices)

    extra_pool = _extra_candidates(inp, axis_kind, axis_index, deficient_side)
    max_extra = min(3, len(extra_pool), len(missing) - 1)
    extra_cells: list[tuple[int, int]] = []
    if max_extra > 0:
        extra_count = random.randint(0, max_extra)
        extra_cells = random.sample(extra_pool, extra_count)
    for r, c in extra_cells:
        inp[r, c] = random.choice(_GEN_COLORS)

    before_count, after_count = _side_counts(inp, axis_kind, axis_index)
    if base_side == "before":
        if before_count <= after_count:
            return None
    elif after_count <= before_count:
        return None

    try:
        solved = main(inp)
    except AssertionError:
        return None

    for r, c in missing:
        if int(solved[r, c]) != int(grid[r, c]):
            return None
    for r, c in wrong_cells + extra_cells:
        if int(solved[r, c]) != MAROON:
            return None
    if not np.any(solved == MAROON):
        return None
    if max(inp.shape) > 30 or max(solved.shape) > 30:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
