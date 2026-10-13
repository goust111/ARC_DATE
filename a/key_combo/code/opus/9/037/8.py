import random
from collections import Counter, defaultdict

import numpy as np


BLACK = 0
GREY = 5
PINK = 6
SHAPE_COLORS = (1, 2, 3, 4, 7, 8, 9)
MAX_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(cells) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _translate(cells, dr: int, dc: int) -> frozenset[tuple[int, int]]:
    return frozenset((r + dr, c + dc) for r, c in cells)


def _components_by_color(grid: np.ndarray, ignored: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in ignored or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            cell_set = frozenset(cells)
            comps.append(
                {
                    "color": color,
                    "cells": cell_set,
                    "bbox": _bbox(cell_set),
                    "shape": _normalize(cell_set),
                }
            )
    return comps


def _find_occluder(grid: np.ndarray) -> frozenset[tuple[int, int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == PINK)]
    assert len(coords) == 4, "expected one 2x2 pink occluder"
    r0, c0, r1, c1 = _bbox(coords)
    assert r1 == r0 + 1 and c1 == c0 + 1, "pink cells must form a 2x2 block"
    box = frozenset((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))
    assert box == frozenset(coords), "pink occluder must be solid"
    return box


def _adjacent_to_occluder(comp_cells: frozenset[tuple[int, int]], occluder: frozenset[tuple[int, int]]) -> bool:
    for r, c in comp_cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occluder:
                return True
    return False


def _symmetry_axes(full_cells: frozenset[tuple[int, int]]) -> list[tuple[str, int]]:
    r0, c0, r1, c1 = _bbox(full_cells)
    candidates = [("v", c0 + c1), ("h", r0 + r1)]
    good = []
    for kind, axis_sum in candidates:
        ok = True
        for r, c in full_cells:
            mirror = (r, axis_sum - c) if kind == "v" else (axis_sum - r, c)
            if mirror not in full_cells:
                ok = False
                break
        if ok:
            good.append((kind, axis_sum))
    return good


def _axis_score(visible: frozenset[tuple[int, int]], kind: str, axis_sum: int) -> int:
    score = 0
    for r, c in visible:
        mirror = (r, axis_sum - c) if kind == "v" else (axis_sum - r, c)
        if mirror in visible:
            score += 1
    return score


def _candidate_full_shapes(
    template_shape: tuple[tuple[int, int], ...],
    visible: frozenset[tuple[int, int]],
    occluder: frozenset[tuple[int, int]],
) -> list[frozenset[tuple[int, int]]]:
    template_cells = list(template_shape)
    candidates: set[frozenset[tuple[int, int]]] = set()
    for vr, vc in visible:
        for sr, sc in template_cells:
            dr = vr - sr
            dc = vc - sc
            full = _translate(template_shape, dr, dc)
            if not visible.issubset(full):
                continue
            if not full - visible <= occluder:
                continue
            if not full & occluder:
                continue
            candidates.add(full)
    return sorted(candidates, key=lambda cells: sorted(cells))


def _recover_spec(grid: np.ndarray) -> tuple[dict, frozenset[tuple[int, int]], frozenset[tuple[int, int]], tuple[str, int]]:
    occluder = _find_occluder(grid)
    comps = _components_by_color(grid, ignored={BLACK, PINK})
    assert len(comps) >= 3, "expected at least three pattern components"

    repeated = [shape for shape, count in Counter(comp["shape"] for comp in comps).items() if count >= 2]
    assert len(repeated) == 1, "expected exactly one repeated complete-shape group"
    template_shape = repeated[0]

    target_candidates = [comp for comp in comps if _adjacent_to_occluder(comp["cells"], occluder)]
    assert len(target_candidates) == 1, "expected one visible component next to the occluder"
    target = target_candidates[0]

    full_candidates = _candidate_full_shapes(template_shape, target["cells"], occluder)
    assert len(full_candidates) == 1, "template position behind the occluder must be unique"
    full_cells = full_candidates[0]
    missing = full_cells & occluder
    assert 1 <= len(missing) <= 3, "occluder must hide one to three cells"

    actual_axes = _symmetry_axes(full_cells)
    assert len(actual_axes) == 1, "the hidden pattern must have a unique symmetry axis"

    r0, c0, r1, c1 = _bbox(full_cells)
    candidate_axes = [("v", c0 + c1), ("h", r0 + r1)]
    scored = []
    for kind, axis_sum in candidate_axes:
        recoverable = sum(_mirror_of(cell, kind, axis_sum) in target["cells"] for cell in missing)
        visible_score = _axis_score(target["cells"], kind, axis_sum)
        scored.append((recoverable, visible_score, kind, axis_sum))
    scored.sort(reverse=True)
    assert scored[0][:2] > scored[1][:2], "expected a unique best symmetry axis"
    axis = (scored[0][2], scored[0][3])
    assert axis == actual_axes[0], "best local symmetry axis must match the shape's true unique axis"
    return target, full_cells, missing, axis


def main(grid: np.ndarray) -> np.ndarray:
    """Locate the 2x2 occluded copy in a same-shape group, restore its hidden cells by local mirror symmetry, and mark the occluder with a grey frame.

    Input structure:
    - The grid has a black background plus 3 or 4 disconnected monochrome pattern components whose complete copies share one exact shape but use different colors.
    - Exactly one 2x2 pink block partially covers one copy, hiding 1 to 3 of its cells while also covering some background.
    - The covered copy still has a visible 4-connected remainder next to the pink block, and its full shape has exactly one mirror axis, horizontal or vertical.

    Rule:
    - Find the repeated complete-shape template from the unobscured components, then find the single visible component adjacent to the pink 2x2 block.
    - Slide the template over that visible component and keep the unique placement whose extra cells fall only inside the pink block; these are the hidden cells.
    - Use the covered copy's unique mirror axis to confirm each hidden cell from its visible mirrored partner, erase the pink block back to black, restore the hidden cells in the component color, and draw a one-cell-thick grey frame around the original 2x2 block without overwriting existing foreground pixels.

    Key insight:
    - The pink block only marks where damage happened; the missing cells come from matching the damaged copy against the repeated shape template, and the final restoration is validated locally by mirror symmetry.
    """
    g = np.asarray(grid, dtype=int)
    target, _full_cells, missing, axis = _recover_spec(g)
    kind, axis_sum = axis
    occluder = _find_occluder(g)

    out = g.copy()
    out[out == PINK] = BLACK
    for r, c in missing:
        mirror = (r, axis_sum - c) if kind == "v" else (axis_sum - r, c)
        assert mirror in target["cells"], "each hidden cell must have a visible mirrored counterpart"
        out[r, c] = target["color"]

    r0, c0, r1, c1 = _bbox(occluder)
    frame_cells = set()
    for r in range(r0 - 1, r1 + 2):
        frame_cells.add((r, c0 - 1))
        frame_cells.add((r, c1 + 1))
    for c in range(c0 - 1, c1 + 2):
        frame_cells.add((r0 - 1, c))
        frame_cells.add((r1 + 1, c))
    for r, c in frame_cells:
        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1] and int(out[r, c]) == BLACK:
            out[r, c] = GREY
    return out


def _mask_to_cells(rows: tuple[str, ...]) -> frozenset[tuple[int, int]]:
    cells = []
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "1":
                cells.append((r, c))
    return frozenset(cells)


def _transpose_cells(cells: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    return frozenset((c, r) for r, c in cells)


def _is_connected(cells: frozenset[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nb in _neighbors4(r, c):
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def _mirror_of(cell: tuple[int, int], kind: str, axis_sum: int) -> tuple[int, int]:
    r, c = cell
    return (r, axis_sum - c) if kind == "v" else (axis_sum - r, c)


def _enumerate_occluders(cells: frozenset[tuple[int, int]], axis_kind: str, axis_sum: int) -> list[dict]:
    r0, c0, r1, c1 = _bbox(cells)
    placements = []
    for br in range(r0 - 1, r1 + 1):
        for bc in range(c0 - 1, c1 + 1):
            box = frozenset((r, c) for r in range(br, br + 2) for c in range(bc, bc + 2))
            covered = cells & box
            if not (1 <= len(covered) <= 3):
                continue
            remaining = cells - covered
            if not _is_connected(remaining):
                continue
            if not any(nb in box for r, c in remaining for nb in _neighbors4(r, c)):
                continue
            if any(_mirror_of(cell, axis_kind, axis_sum) not in remaining for cell in covered):
                continue
            visible_bbox_shift = _bbox(remaining) != _bbox(cells)
            border_overlap = 0
            for r in range(br - 1, br + 3):
                for c in (bc - 1, bc + 2):
                    if (r, c) in remaining:
                        border_overlap += 1
            for c in range(bc - 1, bc + 3):
                for r in (br - 1, br + 2):
                    if (r, c) in remaining:
                        border_overlap += 1
            placements.append(
                {
                    "box_top_left": (br, bc),
                    "box": box,
                    "covered": covered,
                    "bbox_shift": visible_bbox_shift,
                    "border_overlap": border_overlap,
                }
            )
    return placements


def _build_variants() -> list[dict]:
    base_vertical = [
        _mask_to_cells(("010", "111")),
        _mask_to_cells(("101", "111")),
        _mask_to_cells(("010", "111", "101")),
        _mask_to_cells(("111", "010", "101")),
        _mask_to_cells(("1111", "1001")),
        _mask_to_cells(("1111", "0110")),
        _mask_to_cells(("1001", "1111")),
        _mask_to_cells(("0110", "1111")),
    ]
    variants = []
    seen_shapes = set()
    for orient in ("v", "h"):
        for base in base_vertical:
            cells = base if orient == "v" else _transpose_cells(base)
            shape = _normalize(cells)
            if (shape, orient) in seen_shapes:
                continue
            seen_shapes.add((shape, orient))
            axes = _symmetry_axes(cells)
            if len(axes) != 1 or axes[0][0] != orient:
                continue
            placements = _enumerate_occluders(cells, axes[0][0], axes[0][1])
            placements = [p for p in placements if p["border_overlap"] <= 3]
            if not placements:
                continue
            variants.append(
                {
                    "cells": cells,
                    "shape": shape,
                    "axis": axes[0],
                    "size": len(cells),
                    "bbox": _bbox(cells),
                    "placements": placements,
                }
            )
    return variants


VARIANTS = _build_variants()


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _expand_box(box: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    return box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad


def _union_box(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _candidate_positions(
    grid_shape: tuple[int, int],
    variant: dict,
    placement: dict | None,
) -> list[tuple[int, int]]:
    h, w = grid_shape
    r0, c0, r1, c1 = variant["bbox"]
    shape_h = r1 - r0 + 1
    shape_w = c1 - c0 + 1
    positions = []
    for top in range(0, h - shape_h + 1):
        for left in range(0, w - shape_w + 1):
            if placement is not None:
                br, bc = placement["box_top_left"]
                abs_br = top + br
                abs_bc = left + bc
                if not (1 <= abs_br <= h - 3 and 1 <= abs_bc <= w - 3):
                    continue
            positions.append((top, left))
    random.shuffle(positions)
    return positions


def _paint_shape(grid: np.ndarray, cells: frozenset[tuple[int, int]], top: int, left: int, color: int) -> None:
    for r, c in cells:
        grid[top + r, left + c] = color


def _absolute_box(local_box: tuple[int, int, int, int], top: int, left: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = local_box
    return top + r0, left + c0, top + r1, left + c1


def _frame_box_from_occ(top: int, left: int, placement: dict) -> tuple[int, int, int, int]:
    br, bc = placement["box_top_left"]
    return top + br - 1, left + bc - 1, top + br + 2, left + bc + 2


def _choose_positions(grid_shape: tuple[int, int], variant: dict, placement: dict, count: int) -> list[tuple[int, int]] | None:
    target_positions = _candidate_positions(grid_shape, variant, placement)
    complete_positions = _candidate_positions(grid_shape, variant, None)
    for target_top, target_left in target_positions:
        target_shape_box = _absolute_box(variant["bbox"], target_top, target_left)
        target_frame_box = _frame_box_from_occ(target_top, target_left, placement)
        target_reserve = _expand_box(_union_box(target_shape_box, target_frame_box), 1)
        chosen = [(target_top, target_left)]
        reserves = [target_reserve]
        for top, left in complete_positions:
            shape_box = _absolute_box(variant["bbox"], top, left)
            reserve = _expand_box(shape_box, 1)
            if any(_boxes_overlap(reserve, other) for other in reserves):
                continue
            chosen.append((top, left))
            reserves.append(reserve)
            if len(chosen) == count:
                return chosen
    return None


def _expected_output(clean: np.ndarray, target_top: int, target_left: int, placement: dict) -> np.ndarray:
    out = clean.copy()
    br, bc = placement["box_top_left"]
    for r in range(target_top + br - 1, target_top + br + 3):
        for c in (target_left + bc - 1, target_left + bc + 2):
            if 0 <= r < out.shape[0] and 0 <= c < out.shape[1] and int(out[r, c]) == BLACK:
                out[r, c] = GREY
    for c in range(target_left + bc - 1, target_left + bc + 3):
        for r in (target_top + br - 1, target_top + br + 2):
            if 0 <= r < out.shape[0] and 0 <= c < out.shape[1] and int(out[r, c]) == BLACK:
                out[r, c] = GREY
    return out


def _input_from_clean(clean: np.ndarray, target_top: int, target_left: int, placement: dict) -> np.ndarray:
    inp = clean.copy()
    br, bc = placement["box_top_left"]
    inp[target_top + br : target_top + br + 2, target_left + bc : target_left + bc + 2] = PINK
    return inp


def _semantic_ok(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        target, full_cells, missing, axis = _recover_spec(inp)
    except AssertionError:
        return False

    occluder = _find_occluder(inp)
    comps = _components_by_color(inp, ignored={BLACK, PINK})
    if len([comp for comp in comps if comp["shape"] == _normalize(full_cells)]) < 2:
        return False
    if len(missing) != len(full_cells & occluder):
        return False

    actual_axes = _symmetry_axes(full_cells)
    if actual_axes != [axis]:
        return False
    if not np.array_equal(main(inp), out):
        return False
    return True


def _choose_variant() -> tuple[dict, dict]:
    variants = []
    for variant in VARIANTS:
        for placement in variant["placements"]:
            weight = 2
            if placement["bbox_shift"]:
                weight += 4
            weight += len(placement["covered"])
            variants.extend([(variant, placement)] * weight)
    return random.choice(variants)


def _gen_attempt() -> np.ndarray | None:
    if not VARIANTS:
        return None

    grid_h = random.randint(14, 19)
    grid_w = random.randint(14, 19)
    variant, placement = _choose_variant()
    count = random.choice((3, 4))
    positions = _choose_positions((grid_h, grid_w), variant, placement, count)
    if positions is None:
        return None

    colors = random.sample(SHAPE_COLORS, count)
    clean = np.zeros((grid_h, grid_w), dtype=int)
    for (top, left), color in zip(positions, colors):
        _paint_shape(clean, variant["cells"], top, left, color)

    target_top, target_left = positions[0]
    inp = _input_from_clean(clean, target_top, target_left, placement)
    out = _expected_output(clean, target_top, target_left, placement)
    if not _semantic_ok(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
