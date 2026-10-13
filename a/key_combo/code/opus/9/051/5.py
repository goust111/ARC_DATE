from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GREY = 5
TEAL = 8

SHAPES = {
    "square": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "rect_h": ((0, 0), (0, 1), (0, 2)),
    "rect_v": ((0, 0), (1, 0), (2, 0)),
}

_GEN_CALL_COUNT = 0


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    box = _bbox_from_mask(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (int(r0), int(c0))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue = deque([start])
    h, w = mask.shape
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _find_cross_axes(grid: np.ndarray) -> tuple[int, int]:
    row_axes = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == RED)]
    col_axes = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == RED)]
    assert len(row_axes) == 1, "expected exactly one full red horizontal axis"
    assert len(col_axes) == 1, "expected exactly one full red vertical axis"
    return int(row_axes[0]), int(col_axes[0])


def _candidate_placements(
    mask: np.ndarray, covered: np.ndarray, anchor: tuple[int, int]
) -> list[dict]:
    h, w = mask.shape
    r, c = anchor
    placements: list[dict] = []

    options = [
        ("square", ((r, c), (r - 1, c), (r, c - 1), (r - 1, c - 1))),
        ("rect_h", ((r, c), (r, c - 1), (r, c - 2))),
        ("rect_v", ((r, c), (r - 1, c), (r - 2, c))),
    ]
    for kind, top_lefts in options:
        shape = SHAPES[kind]
        for r0, c0 in top_lefts:
            cells = []
            ok = True
            for dr, dc in shape:
                rr = int(r0 + dr)
                cc = int(c0 + dc)
                if not (0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not covered[rr, cc]):
                    ok = False
                    break
                cells.append((rr, cc))
            if ok:
                cell_tuple = tuple(sorted(cells))
                placements.append({"kind": kind, "cells": cell_tuple})

    dedup: dict[tuple[str, tuple[tuple[int, int], ...]], dict] = {}
    for placement in placements:
        key = (placement["kind"], placement["cells"])
        dedup[key] = placement
    return list(dedup.values())


def _decompose_unique(mask: np.ndarray, limit: int = 2) -> list[dict]:
    mask = np.asarray(mask, dtype=bool)
    assert mask.ndim == 2
    assert mask.any(), "foreground mask cannot be empty"
    solutions: list[list[dict]] = []
    covered = np.zeros_like(mask, dtype=bool)

    def dfs(current: list[dict]) -> None:
        if len(solutions) >= limit:
            return
        uncovered = np.argwhere(mask & ~covered)
        if uncovered.size == 0:
            solutions.append([dict(item) for item in current])
            return
        anchor = tuple(map(int, uncovered[0]))
        candidates = _candidate_placements(mask, covered, anchor)
        for placement in candidates:
            for rr, cc in placement["cells"]:
                covered[rr, cc] = True
            current.append(placement)
            dfs(current)
            current.pop()
            for rr, cc in placement["cells"]:
                covered[rr, cc] = False

    dfs([])
    assert len(solutions) == 1, "expected a unique primitive decomposition"
    return solutions[0]


def _paint_cells(out: np.ndarray, cells: tuple[tuple[int, int], ...], color: int) -> None:
    for r, c in cells:
        out[int(r), int(c)] = int(color)


def _quadrant_signs(coords: np.ndarray, axis_r: int, axis_c: int) -> tuple[int, int]:
    row_signs = {-1 if int(r) < axis_r else 1 if int(r) > axis_r else 0 for r, _ in coords}
    col_signs = {-1 if int(c) < axis_c else 1 if int(c) > axis_c else 0 for _, c in coords}
    assert row_signs in ({-1}, {1}), "shape must lie strictly above or below the horizontal axis"
    assert col_signs in ({-1}, {1}), "shape must lie strictly left or right of the vertical axis"
    return next(iter(row_signs)), next(iter(col_signs))


def _transform_cells(
    cells: tuple[tuple[int, int], ...],
    axis_r: int,
    axis_c: int,
    *,
    flip_r: bool,
    flip_c: bool,
) -> tuple[tuple[int, int], ...]:
    out = []
    for r, c in cells:
        rr = 2 * axis_r - int(r) if flip_r else int(r)
        cc = 2 * axis_c - int(c) if flip_c else int(c)
        out.append((rr, cc))
    return tuple(out)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decompose the lone gray composite shape, recolor its primitives, and mirror it across the red cross according to square-vs-rectangle counts.

    Input structure:
    - The background is black.
    - One full-width red row and one full-height red column form a cross that splits the grid into four quadrants.
    - Exactly one quadrant contains one gray 4-connected composite shape built only from non-overlapping 2x2 squares and 1x3 or 3x1 rectangles; the other quadrants are empty.

    Rule:
    - Recover the unique decomposition of the gray shape into 2x2 squares and 1x3 or 3x1 rectangles.
    - Recolor every square teal and every length-three rectangle yellow, keeping the recolored version in the original quadrant.
    - Let S be the number of squares and R the number of rectangles. If S > R, reflect the recolored shape across the horizontal red axis only. If R > S, reflect it across the vertical red axis only. If S = R, reflect it across both axes, producing the horizontal, vertical, and central opposite copies.
    - Leave the red cross unchanged.

    Key insight:
    - The mirror direction is not fixed by geometry alone: it is determined only after decomposing the gray blob into primitive shapes and comparing the two primitive counts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"

    axis_r, axis_c = _find_cross_axes(g)
    allowed = {BLACK, RED, GREY}
    assert set(map(int, np.unique(g))) <= allowed, "input may contain only black, red, and gray"

    shape_coords = np.argwhere(g == GREY)
    assert shape_coords.size > 0, "expected one gray composite shape"
    row_side, col_side = _quadrant_signs(shape_coords, axis_r, axis_c)
    assert np.all(g[(g != BLACK) & (g != RED)] == GREY), "only the gray shape may occupy a quadrant"

    mask = g == GREY
    assert _is_connected(mask), "expected one connected gray composite"
    box = _bbox_from_mask(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    patch_mask = mask[r0 : r1 + 1, c0 : c1 + 1]
    decomposition = _decompose_unique(patch_mask)

    out = g.copy()
    out[g == GREY] = BLACK

    absolute_primitives = []
    for placement in decomposition:
        abs_cells = tuple((r0 + rr, c0 + cc) for rr, cc in placement["cells"])
        color = TEAL if placement["kind"] == "square" else YELLOW
        _paint_cells(out, abs_cells, color)
        absolute_primitives.append((placement["kind"], abs_cells))

    counts = Counter(kind for kind, _ in absolute_primitives)
    squares = int(counts["square"])
    rectangles = int(counts["rect_h"] + counts["rect_v"])
    assert (squares > 0 or rectangles > 0) and squares + rectangles == len(absolute_primitives)

    transforms = [(False, False)]
    if squares > rectangles:
        transforms.append((True, False))
    elif rectangles > squares:
        transforms.append((False, True))
    else:
        transforms.extend([(True, False), (False, True), (True, True)])

    for flip_r, flip_c in transforms[1:]:
        for kind, abs_cells in absolute_primitives:
            color = TEAL if kind == "square" else YELLOW
            mirrored = _transform_cells(abs_cells, axis_r, axis_c, flip_r=flip_r, flip_c=flip_c)
            for rr, cc in mirrored:
                assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1], "reflection must stay in bounds"
                assert rr != axis_r and cc != axis_c, "reflected cells must stay off the separator lines"
            _paint_cells(out, mirrored, color)

    return out


def _piece_candidates(kind: str, occupied: np.ndarray, require_touch: bool) -> list[tuple[tuple[int, int], ...]]:
    shape = SHAPES[kind]
    canvas_h, canvas_w = occupied.shape
    candidates = []
    union = occupied.copy()
    for r0 in range(canvas_h):
        for c0 in range(canvas_w):
            cells = []
            ok = True
            touch = not require_touch
            for dr, dc in shape:
                rr = int(r0 + dr)
                cc = int(c0 + dc)
                if not (0 <= rr < canvas_h and 0 <= cc < canvas_w):
                    ok = False
                    break
                if union[rr, cc]:
                    ok = False
                    break
                cells.append((rr, cc))
            if not ok:
                continue
            if require_touch:
                for rr, cc in cells:
                    for nr, nc in _neighbors4(rr, cc):
                        if 0 <= nr < canvas_h and 0 <= nc < canvas_w and union[nr, nc]:
                            touch = True
                            break
                    if touch:
                        break
            if touch:
                candidates.append(tuple(cells))
    return candidates


def _primitive_contacts(primitives: list[dict]) -> list[int]:
    contacts = []
    cell_sets = [set(piece["cells"]) for piece in primitives]
    for i, cells_a in enumerate(cell_sets):
        for cells_b in cell_sets[i + 1 :]:
            shared_edges = 0
            for r, c in cells_a:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in cells_b:
                        shared_edges += 1
            if shared_edges:
                contacts.append(shared_edges)
    return contacts


def _branch_counts(branch: str) -> tuple[int, int]:
    if branch == "s_gt_r":
        return random.choice([(2, 1), (3, 1)])
    if branch == "r_gt_s":
        return random.choice([(1, 2), (1, 3), (1, 4)])
    assert branch == "equal"
    return 2, 2


def _branch_from_counter() -> str:
    global _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1
    return ("s_gt_r", "r_gt_s", "equal")[(_GEN_CALL_COUNT - 1) % 3]


def _random_piece_order(square_count: int, rect_count: int) -> list[str]:
    pieces = ["square"] * square_count
    rects = []
    if rect_count == 1:
        rects = [random.choice(["rect_h", "rect_v"])]
    elif rect_count >= 2:
        if random.random() < 0.7:
            rects = ["rect_h", "rect_v"] + [random.choice(["rect_h", "rect_v"]) for _ in range(rect_count - 2)]
        else:
            rects = [random.choice(["rect_h", "rect_v"]) for _ in range(rect_count)]
    pieces.extend(rects)
    random.shuffle(pieces)
    return pieces


def _generate_composite(branch: str) -> tuple[np.ndarray, list[dict]]:
    square_count, rect_count = _branch_counts(branch)
    for _ in range(500):
        canvas = np.zeros((7, 7), dtype=bool)
        primitives: list[dict] = []
        piece_order = _random_piece_order(square_count, rect_count)
        ok = True
        for index, kind in enumerate(piece_order):
            candidates = _piece_candidates(kind, canvas, require_touch=index > 0)
            random.shuffle(candidates)
            if not candidates:
                ok = False
                break
            chosen = candidates[0]
            for r, c in chosen:
                canvas[r, c] = True
            primitives.append({"kind": kind, "cells": chosen})
        if not ok:
            continue

        cropped_mask, (off_r, off_c) = _crop_mask(canvas)
        cropped_primitives = []
        for piece in primitives:
            cells = tuple((r - off_r, c - off_c) for r, c in piece["cells"])
            cropped_primitives.append({"kind": piece["kind"], "cells": tuple(sorted(cells))})

        if not _is_connected(cropped_mask):
            continue
        if _bbox_from_mask(cropped_mask) is None:
            continue
        if np.all(cropped_mask):
            continue
        if max(cropped_mask.shape) > 7:
            continue

        contacts = _primitive_contacts(cropped_primitives)
        if not contacts or max(contacts) < 2:
            continue

        try:
            solved = _decompose_unique(cropped_mask)
        except AssertionError:
            continue
        solved_counts = Counter(piece["kind"] for piece in solved)
        if int(solved_counts["square"]) != square_count:
            continue
        if int(solved_counts["rect_h"] + solved_counts["rect_v"]) != rect_count:
            continue

        return cropped_mask, cropped_primitives
    raise RuntimeError(f"failed to generate a valid composite for branch {branch}")


def _sample_grid_shape(patch_h: int, patch_w: int) -> tuple[int, int]:
    min_h = max(15, 2 * patch_h + 5)
    min_w = max(15, 2 * patch_w + 5)
    h_choices = [n for n in range(min_h, 24) if n % 2 == 1]
    w_choices = [n for n in range(min_w, 24) if n % 2 == 1]
    assert h_choices and w_choices
    return random.choice(h_choices), random.choice(w_choices)


def _quadrant_ranges(
    grid_shape: tuple[int, int], axis_r: int, axis_c: int, quadrant: str, patch_h: int, patch_w: int
) -> tuple[list[int], list[int]]:
    if quadrant == "tl":
        rows = list(range(1, axis_r - patch_h))
        cols = list(range(1, axis_c - patch_w))
    elif quadrant == "tr":
        rows = list(range(1, axis_r - patch_h))
        cols = list(range(axis_c + 2, grid_shape[1] - patch_w))
    elif quadrant == "bl":
        rows = list(range(axis_r + 2, grid_shape[0] - patch_h))
        cols = list(range(1, axis_c - patch_w))
    else:
        rows = list(range(axis_r + 2, grid_shape[0] - patch_h))
        cols = list(range(axis_c + 2, grid_shape[1] - patch_w))
    assert rows and cols
    return rows, cols


def _build_expected(
    grid_shape: tuple[int, int],
    axis_r: int,
    axis_c: int,
    base_r: int,
    base_c: int,
    primitives: list[dict],
) -> np.ndarray:
    expected = np.zeros(grid_shape, dtype=int)
    expected[axis_r, :] = RED
    expected[:, axis_c] = RED

    absolute_primitives = []
    for piece in primitives:
        abs_cells = tuple((base_r + r, base_c + c) for r, c in piece["cells"])
        absolute_primitives.append((piece["kind"], abs_cells))

    counts = Counter(kind for kind, _ in absolute_primitives)
    squares = int(counts["square"])
    rectangles = int(counts["rect_h"] + counts["rect_v"])

    transforms = [(False, False)]
    if squares > rectangles:
        transforms.append((True, False))
    elif rectangles > squares:
        transforms.append((False, True))
    else:
        transforms.extend([(True, False), (False, True), (True, True)])

    for flip_r, flip_c in transforms:
        for kind, abs_cells in absolute_primitives:
            color = TEAL if kind == "square" else YELLOW
            cells = _transform_cells(abs_cells, axis_r, axis_c, flip_r=flip_r, flip_c=flip_c)
            _paint_cells(expected, cells, color)

    return expected


def _paint_input_shape(
    grid: np.ndarray, patch_mask: np.ndarray, base_r: int, base_c: int
) -> None:
    coords = np.argwhere(patch_mask)
    for r, c in coords:
        grid[base_r + int(r), base_c + int(c)] = GREY


def _verify_generated_input(
    grid: np.ndarray,
    expected: np.ndarray,
    patch_mask: np.ndarray,
    primitives: list[dict],
    axis_r: int,
    axis_c: int,
    base_r: int,
    base_c: int,
) -> None:
    assert _is_connected(grid == GREY)

    solved = _decompose_unique(patch_mask)
    truth_counts = Counter(piece["kind"] for piece in primitives)
    solved_counts = Counter(piece["kind"] for piece in solved)
    assert int(truth_counts["square"]) == int(solved_counts["square"])
    assert int(truth_counts["rect_h"] + truth_counts["rect_v"]) == int(
        solved_counts["rect_h"] + solved_counts["rect_v"]
    )

    actual = main(grid)
    assert np.array_equal(actual, expected), "generated truth and solver output must match"

    out_quadrants = set()
    colored_coords = np.argwhere((expected == TEAL) | (expected == YELLOW))
    for r, c in colored_coords:
        rs = -1 if int(r) < axis_r else 1
        cs = -1 if int(c) < axis_c else 1
        out_quadrants.add((rs, cs))
    branch_size = len(out_quadrants)
    squares = int(truth_counts["square"])
    rectangles = int(truth_counts["rect_h"] + truth_counts["rect_v"])
    assert branch_size == (4 if squares == rectangles else 2)
    assert not np.array_equal(grid, expected), "input and output must differ"

    original_colors = expected[base_r : base_r + patch_mask.shape[0], base_c : base_c + patch_mask.shape[1]]
    assert np.any(original_colors == TEAL) or np.any(original_colors == YELLOW)


def generate_input() -> np.ndarray:
    for _ in range(400):
        branch = _branch_from_counter()
        patch_mask, primitives = _generate_composite(branch)
        patch_h, patch_w = patch_mask.shape
        grid_h, grid_w = _sample_grid_shape(patch_h, patch_w)
        axis_r, axis_c = grid_h // 2, grid_w // 2
        quadrant = random.choice(["tl", "tr", "bl", "br"])
        rows, cols = _quadrant_ranges((grid_h, grid_w), axis_r, axis_c, quadrant, patch_h, patch_w)
        base_r = random.choice(rows)
        base_c = random.choice(cols)

        grid = np.zeros((grid_h, grid_w), dtype=int)
        grid[axis_r, :] = RED
        grid[:, axis_c] = RED
        _paint_input_shape(grid, patch_mask, base_r, base_c)

        expected = _build_expected((grid_h, grid_w), axis_r, axis_c, base_r, base_c, primitives)
        try:
            _verify_generated_input(grid, expected, patch_mask, primitives, axis_r, axis_c, base_r, base_c)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
