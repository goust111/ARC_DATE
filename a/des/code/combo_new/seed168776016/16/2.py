from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GRAY = 5
L_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MISSING_CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))
AXES = ("h", "v")
GENERATE_RETRIES = 400
PLACE_RETRIES = 300
EXTRA_OBSTACLE_RETRIES = 60


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.flat)
    return max(counts, key=counts.get)


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if (
                        _in_bounds(grid.shape, nr, nc)
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _is_l_triomino(comp: dict[str, object]) -> bool:
    if int(comp["area"]) != 3:
        return False
    r0, c0, r1, c1 = comp["bbox"]
    return int(r1 - r0) == 1 and int(c1 - c0) == 1


def _missing_corner(comp: dict[str, object]) -> tuple[int, int]:
    r0, c0, _r1, _c1 = comp["bbox"]
    rel = {(int(r - r0), int(c - c0)) for r, c in comp["coords"]}
    missing = [corner for corner in MISSING_CORNERS if corner not in rel]
    assert len(missing) == 1, "each L must miss exactly one corner"
    return missing[0]


def _l_cells(r0: int, c0: int, missing: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (r0 + rr, c0 + cc)
        for rr in (0, 1)
        for cc in (0, 1)
        if (rr, cc) != missing
    )


def _candidate_spec(missing: tuple[int, int], axis: str) -> dict[str, object]:
    mr, mc = missing
    if axis == "h":
        dc = -1 if mc == 0 else 1
        source_rel = (1 - mr, mc)
        row_offsets = (0, 1) if source_rel[0] == 0 else (-1, 0)
        col_offsets = (0, 1) if dc == 1 else (-1, 0)
        direction = (0, dc)
    else:
        dr = -1 if mr == 0 else 1
        source_rel = (mr, 1 - mc)
        row_offsets = (0, 1) if dr == 1 else (-1, 0)
        col_offsets = (0, 1) if source_rel[1] == 0 else (-1, 0)
        direction = (dr, 0)

    return {
        "axis": axis,
        "direction": direction,
        "source_rel": source_rel,
        "row_offsets": row_offsets,
        "col_offsets": col_offsets,
    }


def _source_from_bbox(
    r0: int, c0: int, missing: tuple[int, int], axis: str
) -> tuple[int, int]:
    spec = _candidate_spec(missing, axis)
    sr, sc = spec["source_rel"]
    return int(r0 + sr), int(c0 + sc)


def _bbox_from_source(
    source: tuple[int, int], missing: tuple[int, int], axis: str
) -> tuple[int, int]:
    spec = _candidate_spec(missing, axis)
    sr, sc = spec["source_rel"]
    return int(source[0] - sr), int(source[1] - sc)


def _anchor_at_step(
    source: tuple[int, int], missing: tuple[int, int], axis: str, step: int
) -> tuple[int, int]:
    spec = _candidate_spec(missing, axis)
    dr, dc = spec["direction"]
    return int(source[0] + dr * step), int(source[1] + dc * step)


def _patch_cells(
    anchor: tuple[int, int], missing: tuple[int, int], axis: str
) -> tuple[tuple[int, int], ...]:
    spec = _candidate_spec(missing, axis)
    return tuple(
        (int(anchor[0] + ro), int(anchor[1] + co))
        for ro in spec["row_offsets"]
        for co in spec["col_offsets"]
    )


def _patch_in_bounds(
    shape: tuple[int, int], anchor: tuple[int, int], missing: tuple[int, int], axis: str
) -> bool:
    return all(_in_bounds(shape, r, c) for r, c in _patch_cells(anchor, missing, axis))


def _path_length(
    grid: np.ndarray,
    source: tuple[int, int],
    missing: tuple[int, int],
    axis: str,
    bg: int,
) -> int:
    step = 1
    while True:
        anchor = _anchor_at_step(source, missing, axis, step)
        patch = _patch_cells(anchor, missing, axis)
        if not all(_in_bounds(grid.shape, r, c) and int(grid[r, c]) == bg for r, c in patch):
            return step - 1
        step += 1


def _path_length_on_blocked(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    source: tuple[int, int],
    missing: tuple[int, int],
    axis: str,
) -> int:
    step = 1
    while True:
        anchor = _anchor_at_step(source, missing, axis, step)
        patch = _patch_cells(anchor, missing, axis)
        if not all(_in_bounds(shape, r, c) and (r, c) not in blocked for r, c in patch):
            return step - 1
        step += 1


def _path_cells(
    source: tuple[int, int], missing: tuple[int, int], axis: str, length: int
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for step in range(1, length + 1):
        cells.update(_patch_cells(_anchor_at_step(source, missing, axis, step), missing, axis))
    return cells


def _leading_edge_cells(
    source: tuple[int, int], missing: tuple[int, int], axis: str, step: int
) -> list[tuple[int, int]]:
    current = set(_patch_cells(_anchor_at_step(source, missing, axis, step), missing, axis))
    previous = (
        set()
        if step == 1
        else set(_patch_cells(_anchor_at_step(source, missing, axis, step - 1), missing, axis))
    )
    return sorted(current - previous)


def _expand(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if _in_bounds(shape, nr, nc):
                    out.add((nr, nc))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project a filled 2x2 patch from each three-cell L object along its uniquely longer arm.

    Input structure:
    - The background is black.
    - Several foreground objects are monochrome 3-pixel L triominoes, each occupying
      three corners of a 2x2 box and using a distinct color.
    - Additional grey single pixels or small rectangles may appear as blockers.

    Rule:
    - For each L triomino, locate the missing corner of its 2x2 box.
    - That missing corner defines one horizontal candidate arm and one vertical candidate arm.
      Starting from the source pixel adjacent to the missing-corner side, slide the implied
      2x2 filled patch forward along each candidate axis.
    - Use the candidate with the longer unobstructed run through black cells, stopping at the
      last position where the full 2x2 patch still fits before another object or the boundary.
    - Paint that final 2x2 patch in the L's color while leaving the original input unchanged.

    Key insight:
    - The missing corner fixes both the source-adjacent patch corner and the two possible launch
      axes; the surrounding open space breaks the tie and selects the unique projection path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(_get_bg(g))
    comps = _find_components(g, bg)
    l_comps = [comp for comp in comps if _is_l_triomino(comp)]
    assert l_comps, "expected at least one L emitter"

    out = g.copy()
    paint_cells: set[tuple[int, int]] = set()

    for comp in sorted(l_comps, key=lambda item: tuple(int(v) for v in item["bbox"])):
        missing = _missing_corner(comp)
        r0, c0, _r1, _c1 = comp["bbox"]

        lengths: dict[str, int] = {}
        sources: dict[str, tuple[int, int]] = {}
        for axis in AXES:
            source = _source_from_bbox(int(r0), int(c0), missing, axis)
            sources[axis] = source
            lengths[axis] = _path_length(g, source, missing, axis, bg)

        best_axis = "h" if lengths["h"] > lengths["v"] else "v"
        other_axis = "v" if best_axis == "h" else "h"
        assert lengths[best_axis] > lengths[other_axis], "expected a unique longer arm"
        assert lengths[best_axis] > 0, "patch must be able to move at least one step"

        source = sources[best_axis]
        stop = _anchor_at_step(source, missing, best_axis, lengths[best_axis])
        patch = set(_patch_cells(stop, missing, best_axis))
        assert all(int(g[r, c]) == bg for r, c in patch), "final patch must land on background"
        assert paint_cells.isdisjoint(patch), "projected patches must stay disjoint"
        paint_cells.update(patch)

        for r, c in patch:
            out[r, c] = int(comp["color"])

    return out


def _anchor_ok(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    missing: tuple[int, int],
    axis: str,
    stop_mode: str,
) -> bool:
    if not _patch_in_bounds(shape, anchor, missing, axis):
        return False
    spec = _candidate_spec(missing, axis)
    dr, dc = spec["direction"]
    next_anchor = (int(anchor[0] + dr), int(anchor[1] + dc))
    next_in_bounds = _patch_in_bounds(shape, next_anchor, missing, axis)
    if stop_mode == "boundary":
        return not next_in_bounds
    return next_in_bounds


def _place_emitter(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    output_cells: set[tuple[int, int]],
    axis: str,
    stop_mode: str,
    missing: tuple[int, int],
    color: int,
) -> bool:
    h, w = grid.shape

    for _ in range(PLACE_RETRIES):
        anchor = (random.randrange(h), random.randrange(w))
        if not _anchor_ok(grid.shape, anchor, missing, axis, stop_mode):
            continue

        max_len = 8
        desired = random.randint(2, max_len)
        spec = _candidate_spec(missing, axis)
        dr, dc = spec["direction"]
        source = (int(anchor[0] - dr * desired), int(anchor[1] - dc * desired))
        r0, c0 = _bbox_from_source(source, missing, axis)
        l_cells = set(_l_cells(r0, c0, missing))
        if not all(_in_bounds(grid.shape, r, c) for r, c in l_cells):
            continue
        if l_cells & reserved:
            continue

        path = _path_cells(source, missing, axis, desired)
        if path & reserved:
            continue
        if path & l_cells:
            continue
        final_patch = set(_patch_cells(anchor, missing, axis))
        if final_patch & output_cells:
            continue

        blocked = occupied | l_cells
        blockers: list[tuple[int, int]] = []

        if stop_mode == "obstacle":
            leading = _leading_edge_cells(source, missing, axis, desired + 1)
            random.shuffle(leading)
            placed = False
            for cell in leading:
                if not _in_bounds(grid.shape, *cell):
                    continue
                if cell in reserved or cell in blocked or cell in path:
                    continue
                blockers = [cell]
                blocked = blocked | {cell}
                placed = True
                break
            if not placed:
                continue

        if _path_length_on_blocked(grid.shape, blocked, source, missing, axis) != desired:
            continue

        other_axis = "v" if axis == "h" else "h"
        other_source = _source_from_bbox(r0, c0, missing, other_axis)
        other_blocker: tuple[int, int] | None = None
        other_len = _path_length_on_blocked(grid.shape, blocked, other_source, missing, other_axis)
        if other_len >= desired:
            leading = _leading_edge_cells(other_source, missing, other_axis, desired)
            random.shuffle(leading)
            found = False
            for cell in leading:
                if cell in reserved or cell in blocked or cell in path or cell in final_patch:
                    continue
                test_blocked = blocked | {cell}
                if _path_length_on_blocked(grid.shape, test_blocked, source, missing, axis) != desired:
                    continue
                if _path_length_on_blocked(grid.shape, test_blocked, other_source, missing, other_axis) >= desired:
                    continue
                other_blocker = cell
                blocked = test_blocked
                found = True
                break
            if not found:
                continue

        if _path_length_on_blocked(grid.shape, blocked, other_source, missing, other_axis) >= desired:
            continue

        all_input_cells = l_cells | set(blockers)
        if other_blocker is not None:
            all_input_cells.add(other_blocker)
        if all_input_cells & reserved:
            continue

        for r, c in l_cells:
            grid[r, c] = color
        for r, c in blockers:
            grid[r, c] = GRAY
        if other_blocker is not None:
            grid[other_blocker] = GRAY

        occupied.update(all_input_cells)
        output_cells.update(final_patch)
        reserved.update(_expand(all_input_cells, grid.shape))
        reserved.update(path)
        reserved.update(_expand(final_patch, grid.shape))
        return True

    return False


def _add_extra_obstacles(
    grid: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]
) -> None:
    shapes = ((1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1))
    extra_count = random.randint(0, 3)
    h, w = grid.shape

    for _ in range(extra_count):
        for _ in range(EXTRA_OBSTACLE_RETRIES):
            oh, ow = random.choice(shapes)
            if oh > h or ow > w:
                continue
            r0 = random.randint(0, h - oh)
            c0 = random.randint(0, w - ow)
            cells = {(r0 + dr, c0 + dc) for dr in range(oh) for dc in range(ow)}
            if cells & reserved:
                continue
            for r, c in cells:
                grid[r, c] = GRAY
            occupied.update(cells)
            reserved.update(_expand(cells, grid.shape))
            break


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 26)
    w = random.randint(12, 26)
    count = random.randint(2, min(5, len(L_COLORS)))

    grid = np.full((h, w), BG, dtype=int)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    output_cells: set[tuple[int, int]] = set()

    colors = random.sample(L_COLORS, count)
    axes = ["h"] * (count // 2) + ["v"] * (count - count // 2)
    stop_modes = ["boundary"] * (count // 2) + ["obstacle"] * (count - count // 2)
    random.shuffle(axes)
    random.shuffle(stop_modes)

    for idx in range(count):
        missing = random.choice(MISSING_CORNERS)
        if not _place_emitter(
            grid,
            occupied,
            reserved,
            output_cells,
            axes[idx],
            stop_modes[idx],
            missing,
            colors[idx],
        ):
            return None

    _add_extra_obstacles(grid, occupied, reserved)

    bg = int(_get_bg(grid))
    l_count = sum(1 for comp in _find_components(grid, bg) if _is_l_triomino(comp))
    if l_count != count:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
