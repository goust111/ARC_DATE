from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
RECT_COLOR = 2
PIVOT_COLOR = 3
GREY = 5
SQUARE_COLOR = 8

SQUARE = "square"
HBAR = "hbar"
VBAR = "vbar"
RECTS = {HBAR, VBAR}
PRIMITIVES = {
    SQUARE: ((0, 0), (0, 1), (1, 0), (1, 1)),
    HBAR: ((0, 0), (0, 1), (0, 2)),
    VBAR: ((0, 0), (1, 0), (2, 0)),
}
KIND_TO_COLOR = {SQUARE: SQUARE_COLOR, HBAR: RECT_COLOR, VBAR: RECT_COLOR}
MAX_DECOMP_SOLUTIONS = 2
MAX_GEN_ATTEMPTS = 500


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _component_masks(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    nbrs = _neighbors8 if connectivity == 8 else None
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            if connectivity == 8:
                it = nbrs(rr, cc)
            else:
                it = ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1))
            for nr, nc in it:
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        out.append(np.asarray(coords, dtype=int))
    return out


def _enumerate_placements(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    placements = []
    seen: set[tuple[str, tuple[tuple[int, int], ...]]] = set()
    for r in range(h):
        for c in range(w):
            for kind, offsets in PRIMITIVES.items():
                cells = []
                ok = True
                for dr, dc in offsets:
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < h and 0 <= cc < w and mask[rr, cc]):
                        ok = False
                        break
                    cells.append((rr, cc))
                if not ok:
                    continue
                key = (kind, tuple(sorted(cells)))
                if key in seen:
                    continue
                seen.add(key)
                arr = np.asarray(cells, dtype=int)
                placements.append(
                    {
                        "kind": kind,
                        "coords": key[1],
                        "center": (float(arr[:, 0].mean()), float(arr[:, 1].mean())),
                    }
                )
    return placements


def _decompose_unique(mask: np.ndarray) -> list[dict]:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    assert cells, "expected at least one grey cell"
    index = {cell: i for i, cell in enumerate(cells)}
    full = (1 << len(cells)) - 1

    placements = _enumerate_placements(mask)
    bitmasks = []
    by_cell: dict[int, list[int]] = defaultdict(list)
    for i, placement in enumerate(placements):
        bits = 0
        for cell in placement["coords"]:
            bits |= 1 << index[cell]
        bitmasks.append(bits)
        for cell in placement["coords"]:
            by_cell[index[cell]].append(i)

    for i in range(len(cells)):
        assert by_cell[i], "grey cells must be coverable by the primitive library"

    solutions: list[list[int]] = []

    def dfs(covered: int, chosen: list[int]) -> None:
        if len(solutions) >= MAX_DECOMP_SOLUTIONS:
            return
        if covered == full:
            solutions.append(chosen.copy())
            return

        uncovered = [i for i in range(len(cells)) if not (covered >> i) & 1]
        best_cell = None
        best_options = None
        for cell_idx in uncovered:
            opts = [p for p in by_cell[cell_idx] if bitmasks[p] & covered == 0]
            if best_options is None or len(opts) < len(best_options):
                best_cell = cell_idx
                best_options = opts
            if not opts:
                return
        assert best_cell is not None and best_options is not None
        for placement_idx in best_options:
            dfs(covered | bitmasks[placement_idx], chosen + [placement_idx])

    dfs(0, [])
    assert len(solutions) == 1, "grey object must have a unique primitive decomposition"
    return [placements[i] for i in solutions[0]]


def _assign_pivots(pieces: list[dict], pivots: np.ndarray) -> list[int]:
    assert len(pivots) >= 2
    assignments = []
    for piece in pieces:
        cr, cc = piece["center"]
        dists = ((pivots[:, 0] - cr) ** 2 + (pivots[:, 1] - cc) ** 2).astype(float)
        order = np.argsort(dists)
        assert len(order) >= 1
        if len(order) >= 2:
            assert dists[order[1]] - dists[order[0]] > 1e-9, "nearest pivot must be unique"
        assignments.append(int(order[0]))
    return assignments


def _apply_rule(grid: np.ndarray, pieces: list[dict], pivots: np.ndarray, assignments: list[int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[out == GREY] = BG
    occupied = {tuple(map(int, rc)) for rc in np.argwhere(out != BG)}
    occupied -= {tuple(map(int, p)) for p in pivots}

    for piece, pivot_idx in zip(pieces, assignments):
        pr, pc = map(int, pivots[pivot_idx])
        color = KIND_TO_COLOR[piece["kind"]]
        rotated = []
        for r, c in piece["coords"]:
            rr = 2 * pr - r
            cc = 2 * pc - c
            if not (0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]):
                raise AssertionError("rotated primitive left the grid")
            rotated.append((rr, cc))
        for rr, cc in rotated:
            if (rr, cc) in occupied:
                raise AssertionError("rotated primitives must not overlap")
        for rr, cc in rotated:
            if out[rr, cc] == PIVOT_COLOR:
                continue
            out[rr, cc] = color
            occupied.add((rr, cc))
    return out


def _piece_component_ids(mask: np.ndarray, pieces: list[dict]) -> list[int]:
    components = _component_masks(mask, connectivity=8)
    comp_index = {}
    for comp_id, comp in enumerate(components):
        for cell in map(tuple, comp):
            comp_index[(int(cell[0]), int(cell[1]))] = comp_id
    ids = []
    for piece in pieces:
        cell = piece["coords"][0]
        ids.append(comp_index[cell])
    return ids


def _component_centers(mask: np.ndarray) -> list[tuple[float, float]]:
    return [(float(comp[:, 0].mean()), float(comp[:, 1].mean())) for comp in _component_masks(mask, connectivity=8)]


def _shortcut_component_pivot_output(grid: np.ndarray, pieces: list[dict], pivots: np.ndarray) -> np.ndarray:
    grey_mask = np.asarray(grid == GREY)
    centers = _component_centers(grey_mask)
    comp_ids = _piece_component_ids(grey_mask, pieces)
    comp_assignments = []
    for cr, cc in centers:
        dists = ((pivots[:, 0] - cr) ** 2 + (pivots[:, 1] - cc) ** 2).astype(float)
        comp_assignments.append(int(np.argmin(dists)))
    piece_assignments = [comp_assignments[comp_id] for comp_id in comp_ids]
    return _apply_rule(grid, pieces, pivots, piece_assignments)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decompose grey composite shapes into primitives, rotate each primitive around its nearest pivot, then recolor by primitive type.

    Input structure:
    - Black background with one or two grey composite shapes built from non-overlapping 2x2 squares and 1x3 or 3x1 rectangles.
    - Two or three green single-pixel pivots lie in empty space, separate from the grey cells.
    - The grey decomposition is uniquely determined, and each primitive has a unique nearest pivot under Euclidean distance from its pixel-average center.

    Rule:
    - Uniquely decompose all grey cells into 2x2 squares and 1x3 or 3x1 rectangles.
    - For each primitive, compute its center as the arithmetic mean of its pixel coordinates and assign the nearest green pivot.
    - Rotate every primitive 180 degrees around its own assigned pivot using (r, c) -> (2*pr-r, 2*pc-c).
    - Clear the original grey cells, keep pivots green, and recolor rotated 2x2 squares to teal and rotated 1x3 or 3x1 rectangles to red.

    Key insight:
    - The critical step is primitive-wise pivot assignment: different pieces from the same visible grey composite can fly to different destinations because decomposition fixes each piece center and therefore its nearest pivot.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    grey_mask = g == GREY
    pivots = np.argwhere(g == PIVOT_COLOR).astype(int)
    assert grey_mask.any(), "expected grey composite shapes"
    assert 2 <= len(pivots) <= 3, "expected two or three pivots"

    pieces = _decompose_unique(grey_mask)
    assignments = _assign_pivots(pieces, pivots)
    return _apply_rule(g, pieces, pivots, assignments)


def _piece_cells(kind: str, anchor_r: int, anchor_c: int) -> set[tuple[int, int]]:
    return {(anchor_r + dr, anchor_c + dc) for dr, dc in PRIMITIVES[kind]}


def _touches_group(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in occupied:
                return True
    return False


def _build_group(kinds: list[str]) -> list[dict] | None:
    pieces: list[dict] = []
    occupied: set[tuple[int, int]] = set()

    first_kind = kinds[0]
    first_cells = _piece_cells(first_kind, 0, 0)
    pieces.append({"kind": first_kind, "cells": first_cells})
    occupied |= first_cells

    for kind in kinds[1:]:
        r0, c0, r1, c1 = _bbox(sorted(occupied))
        options = []
        for ar in range(r0 - 4, r1 + 5):
            for ac in range(c0 - 4, c1 + 5):
                cells = _piece_cells(kind, ar, ac)
                if cells & occupied:
                    continue
                if not _touches_group(cells, occupied):
                    continue
                rr0, cc0, rr1, cc1 = _bbox(sorted(occupied | cells))
                if rr1 - rr0 > 7 or cc1 - cc0 > 7:
                    continue
                options.append(cells)
        if not options:
            return None
        cells = random.choice(options)
        pieces.append({"kind": kind, "cells": cells})
        occupied |= cells

    rr0, cc0, rr1, cc1 = _bbox(sorted(occupied))
    rot_k = random.randrange(4)
    flip_h = bool(random.getrandbits(1))
    flip_v = bool(random.getrandbits(1))
    canvas_h = rr1 - rr0 + 1
    canvas_w = cc1 - cc0 + 1
    transformed_arrays = []
    for piece in pieces:
        arr = np.asarray(sorted(piece["cells"]), dtype=int) - np.array([rr0, cc0])
        pts = arr.copy()
        ch, cw = canvas_h, canvas_w
        for _ in range(rot_k % 4):
            pts = np.column_stack([pts[:, 1], ch - 1 - pts[:, 0]])
            ch, cw = cw, ch
        if flip_h:
            pts[:, 1] = cw - 1 - pts[:, 1]
        if flip_v:
            pts[:, 0] = ch - 1 - pts[:, 0]
        transformed_arrays.append((piece["kind"], pts))

    min_r = min(int(pts[:, 0].min()) for _, pts in transformed_arrays)
    min_c = min(int(pts[:, 1].min()) for _, pts in transformed_arrays)
    out = []
    for kind, pts in transformed_arrays:
        pts = pts - np.array([min_r, min_c])
        out.append({"kind": kind, "cells": {tuple(map(int, rc)) for rc in pts}})
    return out


def _piece_kind_multiset(total: int) -> list[str]:
    while True:
        n_squares = random.randint(1, total - 1)
        n_rects = total - n_squares
        rect_kinds = [random.choice([HBAR, VBAR]) for _ in range(n_rects)]
        kinds = [SQUARE] * n_squares + rect_kinds
        random.shuffle(kinds)
        if SQUARE in kinds and any(kind in RECTS for kind in kinds):
            return kinds


def _split_groups(total: int, n_groups: int) -> list[int]:
    if n_groups == 1:
        return [total]
    first = random.randint(1, total - 1)
    return [first, total - first]


def _place_groups(
    h: int, w: int, groups: list[list[dict]]
) -> list[dict] | None:
    placed: list[dict] = []
    occupied: set[tuple[int, int]] = set()

    for group in groups:
        group_cells = set().union(*(piece["cells"] for piece in group))
        r0, c0, r1, c1 = _bbox(sorted(group_cells))
        gh = r1 - r0 + 1
        gw = c1 - c0 + 1
        rel_group = [{**piece, "cells": {(r - r0, c - c0) for r, c in piece["cells"]}} for piece in group]
        options = []
        for ar in range(1, h - gh):
            for ac in range(1, w - gw):
                shifted = set()
                ok = True
                for rr, cc in group_cells:
                    cell = (ar + rr - r0, ac + cc - c0)
                    if cell in occupied:
                        ok = False
                        break
                    shifted.add(cell)
                if not ok:
                    continue
                for rr, cc in shifted:
                    for nr, nc in _neighbors8(rr, cc):
                        if (nr, nc) in occupied:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    options.append((ar, ac))
        if not options:
            return None
        ar, ac = random.choice(options)
        for piece in rel_group:
            cells = {(ar + rr, ac + cc) for rr, cc in piece["cells"]}
            placed.append({"kind": piece["kind"], "cells": cells})
            occupied |= cells
    return placed


def _render_input(h: int, w: int, pieces: list[dict], pivots: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for piece in pieces:
        for r, c in piece["cells"]:
            grid[r, c] = GREY
    for r, c in pivots:
        assert grid[r, c] == BG
        grid[r, c] = PIVOT_COLOR
    return grid


def _candidate_pivot_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    rows = list(range(1, h - 1))
    cols = list(range(1, w - 1))
    central_rows = [r for r in rows if h // 4 <= r < h - h // 4]
    central_cols = [c for c in cols if w // 4 <= c < w - w // 4]
    core = [(r, c) for r in central_rows for c in central_cols if grid[r, c] == BG]
    core_set = set(core)
    outer = [(r, c) for r in rows for c in cols if grid[r, c] == BG and (r, c) not in core_set]
    random.shuffle(core)
    random.shuffle(outer)
    return core + outer


def _validate_instance(inp: np.ndarray) -> bool:
    grey_mask = inp == GREY
    pivots = np.argwhere(inp == PIVOT_COLOR).astype(int)
    try:
        pieces = _decompose_unique(grey_mask)
        assignments = _assign_pivots(pieces, pivots)
        out = _apply_rule(inp, pieces, pivots, assignments)
    except Exception:
        return False

    if np.any(out == GREY):
        return False
    if len(set(assignments)) < 2:
        return False
    if not any(piece["kind"] == SQUARE for piece in pieces):
        return False
    if not any(piece["kind"] in RECTS for piece in pieces):
        return False

    comp_ids = _piece_component_ids(grey_mask, pieces)
    pivots_per_comp: dict[int, set[int]] = defaultdict(set)
    sizes_per_comp: dict[int, int] = defaultdict(int)
    for comp_id, pivot_id in zip(comp_ids, assignments):
        pivots_per_comp[comp_id].add(pivot_id)
        sizes_per_comp[comp_id] += 1
    if not any(sizes_per_comp[cid] >= 2 and len(pivots_per_comp[cid]) >= 2 for cid in sizes_per_comp):
        return False

    try:
        shortcut = _shortcut_component_pivot_output(inp, pieces, pivots)
    except Exception:
        return False
    if np.array_equal(shortcut, out):
        return False

    comps = _component_masks(grey_mask, connectivity=8)
    if not (1 <= len(comps) <= 2):
        return False
    if any(inp[tuple(p)] != PIVOT_COLOR for p in pivots):
        return False
    if np.any(out[np.where(inp == PIVOT_COLOR)] != PIVOT_COLOR):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(10, 16)
    w = random.randint(12, 18)
    total_pieces = random.randint(3, 5)
    n_groups = random.choice([1, 2])
    group_sizes = _split_groups(total_pieces, n_groups)
    if max(group_sizes) < 2:
        return None

    kinds = _piece_kind_multiset(total_pieces)
    groups = []
    idx = 0
    for size in group_sizes:
        group = _build_group(kinds[idx : idx + size])
        if group is None:
            return None
        groups.append(group)
        idx += size

    placed = _place_groups(h, w, groups)
    if placed is None:
        return None

    base_grid = _render_input(h, w, placed, [])
    pivot_count = random.choice([2, 3])
    candidates = _candidate_pivot_cells(base_grid)
    if len(candidates) < pivot_count:
        return None

    for _ in range(60):
        pivots = random.sample(candidates[: min(len(candidates), 60)], pivot_count)
        inp = _render_input(h, w, placed, pivots)
        if _validate_instance(inp):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input")
