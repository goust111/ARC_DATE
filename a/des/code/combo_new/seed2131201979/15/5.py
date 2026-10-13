from __future__ import annotations

import random
from itertools import product

import numpy as np


BG = 0
MAX_GEN_ATTEMPTS = 600
PAIR_COUNT_RANGE = (1, 3)
LINE_LEN_RANGE = (2, 8)
GRID_SIZE_BY_PAIR_COUNT = {
    1: (10, 18),
    2: (14, 24),
    3: (20, 30),
}

# Small non-symmetric polyominoes with 4 distinct rotations.
SHAPE_TEMPLATES = (
    ((0, 0), (1, 0), (1, 1)),                            # L triomino
    ((0, 0), (1, 0), (2, 0), (2, 1)),                    # L tetromino
    ((0, 1), (1, 0), (1, 1), (1, 2)),                    # T tetromino
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),            # long L pentomino
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),            # bent pentomino
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_from_cells(cells: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    remaining = set(cells)
    comps: list[list[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        comp = [start]
        while stack:
            r, c = stack.pop()
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in remaining:
                    remaining.remove((nr, nc))
                    stack.append((nr, nc))
                    comp.append((nr, nc))
        comps.append(comp)
    return comps


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    cells = set(map(tuple, np.argwhere(mask)))
    return _components_from_cells(cells)


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]] | set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _coords_to_mask(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    arr = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        arr[r - r0, c - c0] = True
    return arr


def _rotate_norm(coords: tuple[tuple[int, int], ...], turns: int) -> tuple[tuple[int, int], ...]:
    arr = _coords_to_mask(coords)
    rot = np.rot90(arr, turns)
    pts = tuple(map(tuple, np.argwhere(rot)))
    return _normalize(pts)


def _rotation_keys(coords: tuple[tuple[int, int], ...]) -> set[tuple[tuple[int, int], ...]]:
    return {_rotate_norm(coords, turns) for turns in range(4)}


def _translate(coords: tuple[tuple[int, int], ...], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in coords}


def _split_candidate(
    cells: set[tuple[int, int]],
    axis: str,
    fixed: int,
    start: int,
    end: int,
) -> dict[str, object] | None:
    if axis == "h":
        line = {(fixed, c) for c in range(start, end + 1)}
        left_contact = (fixed, start - 1)
        right_contact = (fixed, end + 1)
    else:
        line = {(r, fixed) for r in range(start, end + 1)}
        left_contact = (start - 1, fixed)
        right_contact = (end + 1, fixed)

    if not line.issubset(cells):
        return None
    rem = cells - line
    if len(rem) < 6:
        return None

    comps = _components_from_cells(rem)
    if len(comps) != 2:
        return None
    if any(not (3 <= len(comp) <= 5) for comp in comps):
        return None

    comp_sets = [set(comp) for comp in comps]
    if left_contact not in rem or right_contact not in rem:
        return None
    left_owner = next((i for i, comp in enumerate(comp_sets) if left_contact in comp), None)
    right_owner = next((i for i, comp in enumerate(comp_sets) if right_contact in comp), None)
    if left_owner is None or right_owner is None or left_owner == right_owner:
        return None

    adjacency_counts = [0, 0]
    for lr, lc in line:
        for idx, comp in enumerate(comp_sets):
            for nr, nc in _neighbors4(lr, lc):
                if (nr, nc) in comp:
                    adjacency_counts[idx] += 1
    if adjacency_counts != [1, 1]:
        return None

    shape_a = _normalize(comps[left_owner])
    shape_b = _normalize(comps[right_owner])
    if shape_a == shape_b:
        return None
    if shape_b not in _rotation_keys(shape_a):
        return None

    return {
        "axis": axis,
        "line": line,
        "objects": [comps[left_owner], comps[right_owner]],
    }


def _decompose_pair(comp: list[tuple[int, int]]) -> dict[str, object]:
    cells = set(comp)
    candidates: list[dict[str, object]] = []

    rows = {}
    cols = {}
    for r, c in comp:
        rows.setdefault(r, []).append(c)
        cols.setdefault(c, []).append(r)

    for r, row_cols in rows.items():
        uniq = sorted(set(row_cols))
        for i, start in enumerate(uniq):
            end = start
            while i + 1 < len(uniq) and uniq[i + 1] == end + 1:
                i += 1
                end = uniq[i]
            for a in range(start, end + 1):
                for b in range(a, end + 1):
                    cand = _split_candidate(cells, "h", r, a, b)
                    if cand is not None:
                        candidates.append(cand)

    for c, col_rows in cols.items():
        uniq = sorted(set(col_rows))
        for i, start in enumerate(uniq):
            end = start
            while i + 1 < len(uniq) and uniq[i + 1] == end + 1:
                i += 1
                end = uniq[i]
            for a in range(start, end + 1):
                for b in range(a, end + 1):
                    cand = _split_candidate(cells, "v", c, a, b)
                    if cand is not None:
                        candidates.append(cand)

    assert candidates, "component must split into two rotated objects joined by one straight line"
    candidates.sort(key=lambda d: (-len(d["line"]), int(d["axis"] == "v")))
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each linked object pair's anchors while also swapping the orientations at the destination anchors.

    Input structure:
    - The background is black.
    - Each non-background color forms one connected structure made of two small same-color objects and a same-color one-pixel-wide straight line segment joining them.
    - Within a pair, the two endpoint objects have the same shape up to rotation, different orientations, and areas between 3 and 5 pixels.

    Rule:
    - For every connected color structure, identify the straight connector and split the structure into its two endpoint objects.
    - Exchange the two objects' anchor locations, where an anchor is the top-left corner of the object's bounding box.
    - After moving, redraw each object using the orientation that was originally present at its destination anchor; keep the straight connector fixed.

    Key insight:
    - The line reveals which two endpoint objects belong together, and the swap has two coupled parts: anchor positions exchange and the destination orientation is inherited together with the destination slot.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[counts.argmax()])
    out = np.full_like(g, bg)

    for color in sorted(int(v) for v in vals if int(v) != bg):
        comps = _components(g == color)
        for comp in comps:
            dec = _decompose_pair(comp)
            for r, c in dec["line"]:
                out[r, c] = color

            obj_a, obj_b = dec["objects"]
            anchor_a = _bbox(obj_a)[:2]
            anchor_b = _bbox(obj_b)[:2]
            mask_a = _normalize(obj_a)
            mask_b = _normalize(obj_b)

            # After the identity swap, each anchor keeps the geometry that originally
            # occupied that anchor because the arriving object inherits that anchor's
            # orientation.
            for r, c in _translate(mask_a, *anchor_a):
                out[r, c] = color
            for r, c in _translate(mask_b, *anchor_b):
                out[r, c] = color

    return out


def _build_pair_patch(
    template: tuple[tuple[int, int], ...],
    rot_a: int,
    rot_b: int,
    axis: str,
    line_len: int,
    color: int,
) -> np.ndarray | None:
    shape_a = _rotate_norm(template, rot_a)
    shape_b = _rotate_norm(template, rot_b)

    if axis == "h":
        side_a = [pt for pt in shape_a if pt[1] == max(c for _, c in shape_a)]
        side_b = [pt for pt in shape_b if pt[1] == min(c for _, c in shape_b)]
    else:
        side_a = [pt for pt in shape_a if pt[0] == max(r for r, _ in shape_a)]
        side_b = [pt for pt in shape_b if pt[0] == min(r for r, _ in shape_b)]

    contacts = list(product(side_a, side_b))
    random.shuffle(contacts)

    for contact_a, contact_b in contacts:
        if axis == "h":
            anchor_a = (0, 0)
            anchor_b = (contact_a[0] - contact_b[0], contact_a[1] + line_len + 1 - contact_b[1])
        else:
            anchor_a = (0, 0)
            anchor_b = (contact_a[0] + line_len + 1 - contact_b[0], contact_a[1] - contact_b[1])

        shift_r = -min(0, anchor_a[0], anchor_b[0])
        shift_c = -min(0, anchor_a[1], anchor_b[1])
        anchor_a = (anchor_a[0] + shift_r, anchor_a[1] + shift_c)
        anchor_b = (anchor_b[0] + shift_r, anchor_b[1] + shift_c)

        obj_a = _translate(shape_a, *anchor_a)
        obj_b = _translate(shape_b, *anchor_b)
        contact_a_abs = (anchor_a[0] + contact_a[0], anchor_a[1] + contact_a[1])
        contact_b_abs = (anchor_b[0] + contact_b[0], anchor_b[1] + contact_b[1])

        if axis == "h":
            if contact_a_abs[0] != contact_b_abs[0] or contact_a_abs[1] >= contact_b_abs[1] - 1:
                continue
            line = {(contact_a_abs[0], c) for c in range(contact_a_abs[1] + 1, contact_b_abs[1])}
        else:
            if contact_a_abs[1] != contact_b_abs[1] or contact_a_abs[0] >= contact_b_abs[0] - 1:
                continue
            line = {(r, contact_a_abs[1]) for r in range(contact_a_abs[0] + 1, contact_b_abs[0])}

        if len(line) != line_len:
            continue
        if obj_a & obj_b or obj_a & line or obj_b & line:
            continue

        # The line must touch each object exactly once and only at its endpoint.
        pair_cells = obj_a | obj_b | line
        if len(_components_from_cells(pair_cells)) != 1:
            continue
        adj_a = 0
        adj_b = 0
        for r, c in line:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in obj_a:
                    adj_a += 1
                if (nr, nc) in obj_b:
                    adj_b += 1
        if (adj_a, adj_b) != (1, 1):
            continue

        r0, c0, r1, c1 = _bbox(pair_cells)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
        for r, c in pair_cells:
            patch[r - r0, c - c0] = color

        comps = _components(patch == color)
        if len(comps) != 1:
            continue
        try:
            _decompose_pair(comps[0])
        except AssertionError:
            continue
        return patch

    return None


def _sample_pair_patch(color: int) -> np.ndarray:
    for _ in range(200):
        template = random.choice(SHAPE_TEMPLATES)
        rot_a, rot_b = random.sample(range(4), 2)
        axis = random.choice(("h", "v"))
        line_len = random.randint(*LINE_LEN_RANGE)
        patch = _build_pair_patch(template, rot_a, rot_b, axis, line_len, color)
        if patch is not None:
            return patch
    raise RuntimeError("failed to build a valid pair patch")


def _can_place(occupied: np.ndarray, top: int, left: int, patch: np.ndarray) -> bool:
    h, w = patch.shape
    H, W = occupied.shape
    if top + h > H or left + w > W:
        return False
    rr0 = max(0, top - 1)
    cc0 = max(0, left - 1)
    rr1 = min(H, top + h + 1)
    cc1 = min(W, left + w + 1)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> bool:
    H, W = grid.shape
    h, w = patch.shape
    top_choices = list(range(H - h + 1))
    left_choices = list(range(W - w + 1))
    random.shuffle(top_choices)
    random.shuffle(left_choices)
    for top in top_choices:
        for left in left_choices:
            if not _can_place(occupied, top, left, patch):
                continue
            mask = patch != BG
            grid[top : top + h, left : left + w][mask] = patch[mask]
            occupied[top : top + h, left : left + w] |= mask
            return True
    return False


def _generate_once() -> np.ndarray:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    side_lo, side_hi = GRID_SIZE_BY_PAIR_COUNT[pair_count]
    height = random.randint(side_lo, side_hi)
    width = random.randint(side_lo, side_hi)
    colors = random.sample(list(range(1, 10)), pair_count)

    patches = [_sample_pair_patch(color) for color in colors]
    random.shuffle(patches)

    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    for patch in sorted(patches, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
        if not _place_patch(grid, occupied, patch):
            raise RuntimeError("placement failed")

    out = main(grid)
    if not np.array_equal(out, grid):
        raise RuntimeError("solver/oracle mismatch")
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            return _generate_once()
        except Exception:
            continue
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
