from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
INPUT_COLORS = (1, 2, 4, 7, 9)

BASE_SHAPES = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (2, 1), (2, 2)),
)


def _normalize(coords: set[tuple[int, int]] | list[tuple[int, int]] | np.ndarray) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(list(coords), dtype=int)
    arr = arr - arr.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in arr))


def _bbox_from_sig(sig: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in sig]
    cols = [c for _, c in sig]
    return max(rows) + 1, max(cols) + 1


def _transform_signature(sig: tuple[tuple[int, int], ...], code: int) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(sig, dtype=int)
    rows = arr[:, 0]
    cols = arr[:, 1]
    if code == 0:
        out = np.stack([rows, cols], axis=1)
    elif code == 1:
        out = np.stack([rows, -cols], axis=1)
    elif code == 2:
        out = np.stack([-rows, cols], axis=1)
    elif code == 3:
        out = np.stack([-rows, -cols], axis=1)
    elif code == 4:
        out = np.stack([cols, rows], axis=1)
    elif code == 5:
        out = np.stack([cols, -rows], axis=1)
    elif code == 6:
        out = np.stack([-cols, rows], axis=1)
    else:
        out = np.stack([-cols, -rows], axis=1)
    return _normalize(out)


def _all_shape_variants() -> list[tuple[tuple[int, int], ...]]:
    variants: set[tuple[tuple[int, int], ...]] = set()
    for base in BASE_SHAPES:
        sig = tuple(base)
        for code in range(8):
            variants.add(_transform_signature(sig, code))
    return sorted(variants)


ALL_SHAPES = _all_shape_variants()


def _coords_at(sig: tuple[tuple[int, int], ...], r0: int, c0: int) -> set[tuple[int, int]]:
    return {(r0 + r, c0 + c) for r, c in sig}


def _bbox_of_cells(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _find_cross(g: np.ndarray) -> tuple[int, int]:
    rows = np.where(np.all(g == GREEN, axis=1))[0]
    cols = np.where(np.all(g == GREEN, axis=0))[0]
    assert len(rows) == 1 and len(cols) == 1
    return int(rows[0]), int(cols[0])


def _components(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if visited[r, c] or color in (BLACK, GREEN):
                continue
            stack = [(r, c)]
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            sig = _normalize(cells)
            bbox = _bbox_of_cells(cells)
            comps.append(
                {
                    "color": color,
                    "cells": set(cells),
                    "bbox": bbox,
                    "sig": sig,
                }
            )
    return comps


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _paint_if_empty(out: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        if out[r, c] == BLACK:
            out[r, c] = color


def _move_cells(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _reflect_cells(cells: set[tuple[int, int]], axis_r: int, axis_c: int) -> set[tuple[int, int]]:
    return {(2 * axis_r - r, 2 * axis_c - c) for r, c in cells}


def _pair_orientation(a: dict, b: dict) -> str:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]
    row_overlap = _intervals_overlap(ar0, ar1, br0, br1)
    col_overlap = _intervals_overlap(ac0, ac1, bc0, bc1)
    if row_overlap and not col_overlap:
        return "h"
    if col_overlap and not row_overlap:
        return "v"
    raise AssertionError("matched pair must share rows or columns but not both")


def _paint_matched_pair(out: np.ndarray, a: dict, b: dict) -> None:
    orient = _pair_orientation(a, b)
    ah, aw = _bbox_from_sig(a["sig"])
    bh, bw = _bbox_from_sig(b["sig"])
    assert (ah, aw) == (bh, bw)

    if orient == "h":
        left, right = (a, b) if a["bbox"][1] < b["bbox"][1] else (b, a)
        row0 = left["bbox"][0]
        gap = right["bbox"][1] - left["bbox"][3] - 1
        assert gap >= 1 and gap % 2 == 1
        steps = (gap - 1) // 2
        for t in range(steps + 1):
            _paint_if_empty(out, _move_cells(left["cells"], 0, t), left["color"])
            _paint_if_empty(out, _move_cells(right["cells"], 0, -t), right["color"])
        mid_c = left["bbox"][3] + steps + 1
        for r in range(row0, row0 + ah):
            if out[r, mid_c] == BLACK:
                out[r, mid_c] = GREY
    else:
        top, bottom = (a, b) if a["bbox"][0] < b["bbox"][0] else (b, a)
        col0 = top["bbox"][1]
        gap = bottom["bbox"][0] - top["bbox"][2] - 1
        assert gap >= 1 and gap % 2 == 1
        steps = (gap - 1) // 2
        for t in range(steps + 1):
            _paint_if_empty(out, _move_cells(top["cells"], t, 0), top["color"])
            _paint_if_empty(out, _move_cells(bottom["cells"], -t, 0), bottom["color"])
        mid_r = top["bbox"][2] + steps + 1
        for c in range(col0, col0 + aw):
            if out[mid_r, c] == BLACK:
                out[mid_r, c] = GREY


def _paint_unmatched(out: np.ndarray, comp: dict, axis_r: int, axis_c: int) -> None:
    r0, c0, r1, c1 = comp["bbox"]
    dr = 1 if r1 < axis_r else -1
    dc = 1 if c1 < axis_c else -1
    row_steps = axis_r - r1 - 1 if dr == 1 else r0 - axis_r - 1
    col_steps = axis_c - c1 - 1 if dc == 1 else c0 - axis_c - 1
    assert row_steps == col_steps and row_steps >= 1
    steps = row_steps

    reflected = _reflect_cells(comp["cells"], axis_r, axis_c)
    for t in range(steps + 1):
        _paint_if_empty(out, _move_cells(comp["cells"], dr * t, dc * t), comp["color"])
        _paint_if_empty(out, _move_cells(reflected, -dr * t, -dc * t), MAGENTA)
    if out[axis_r, axis_c] == BLACK:
        out[axis_r, axis_c] = GREY


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-shape sprites for inward sliding, and create a mirrored partner for the lone unmatched sprite before sliding.

    Input structure:
    - A black background contains one full green cross: exactly one complete row and one complete column.
    - Five non-green monochrome 4-connected sprites occupy the quadrants, each using a distinct input color.
    - Two normalized shapes appear twice each to form matched pairs, and one normalized shape appears once as the unmatched sprite.

    Rule:
    - Detect all non-black, non-green connected components and group them by translation-normalized shape.
    - For every matched pair, slide the two equal-shape sprites toward each other along their shared row or shared column, leaving trails in their own colors and placing a gray meeting line when their bounding boxes stop one cell apart.
    - For the lone unmatched sprite, generate its 180-degree point reflection around the cross center in magenta, then slide the original and the reflected copy diagonally toward the center while leaving trails.
    - Later paths never overwrite earlier colored trails or gray meeting marks, and the full green cross is restored unchanged at the end.

    Key insight:
    - Shape equality is only the branch selector: repeated shapes move directly toward their partner, while the unique shape must first be completed by point symmetry and only then follows the same inward-motion idea.
    """
    g = np.asarray(grid, dtype=int)
    axis_r, axis_c = _find_cross(g)
    comps = _components(g)
    by_sig: dict[tuple[tuple[int, int], ...], list[dict]] = defaultdict(list)
    for comp in comps:
        by_sig[comp["sig"]].append(comp)

    matched: list[tuple[dict, dict]] = []
    unmatched: list[dict] = []
    for group in by_sig.values():
        assert len(group) in (1, 2)
        if len(group) == 2:
            matched.append((group[0], group[1]))
        else:
            unmatched.append(group[0])
    assert len(unmatched) == 1

    out = g.copy()
    pair_order: list[tuple[float, float, dict, dict]] = []
    for a, b in matched:
        ar0, ac0, ar1, ac1 = a["bbox"]
        br0, bc0, br1, bc1 = b["bbox"]
        mid_r = (ar0 + ar1 + br0 + br1) / 4.0
        mid_c = (ac0 + ac1 + bc0 + bc1) / 4.0
        pair_order.append((mid_r, mid_c, a, b))
    for _, _, a, b in sorted(pair_order):
        _paint_matched_pair(out, a, b)
    for comp in sorted(unmatched, key=lambda item: (item["bbox"][0], item["bbox"][1])):
        _paint_unmatched(out, comp, axis_r, axis_c)

    out[axis_r, :] = GREEN
    out[:, axis_c] = GREEN
    return out


def _buffer_ok(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        if (r, c) in occupied:
            return False
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) in occupied:
                    return False
    return True


def _path_cells_pair(sig: tuple[tuple[int, int], ...], pos_a: tuple[int, int], pos_b: tuple[int, int], orient: str) -> set[tuple[int, int]]:
    h, w = _bbox_from_sig(sig)
    a_cells = _coords_at(sig, *pos_a)
    b_cells = _coords_at(sig, *pos_b)
    path: set[tuple[int, int]] = set()
    if orient == "h":
        gap = pos_b[1] - (pos_a[1] + w - 1) - 1
        if gap < 1 or gap % 2 == 0:
            return set()
        steps = (gap - 1) // 2
        for t in range(steps + 1):
            path |= _move_cells(a_cells, 0, t)
            path |= _move_cells(b_cells, 0, -t)
        mid_c = pos_a[1] + w - 1 + steps + 1
        for r in range(pos_a[0], pos_a[0] + h):
            path.add((r, mid_c))
    else:
        gap = pos_b[0] - (pos_a[0] + h - 1) - 1
        if gap < 1 or gap % 2 == 0:
            return set()
        steps = (gap - 1) // 2
        for t in range(steps + 1):
            path |= _move_cells(a_cells, t, 0)
            path |= _move_cells(b_cells, -t, 0)
        mid_r = pos_a[0] + h - 1 + steps + 1
        for c in range(pos_a[1], pos_a[1] + w):
            path.add((mid_r, c))
    return path


def _path_cells_unmatched(sig: tuple[tuple[int, int], ...], pos: tuple[int, int], axis_r: int, axis_c: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    cells = _coords_at(sig, *pos)
    mirrored = _reflect_cells(cells, axis_r, axis_c)
    r0, c0, r1, c1 = _bbox_of_cells(cells)
    dr = 1 if r1 < axis_r else -1
    dc = 1 if c1 < axis_c else -1
    row_steps = axis_r - r1 - 1 if dr == 1 else r0 - axis_r - 1
    col_steps = axis_c - c1 - 1 if dc == 1 else c0 - axis_c - 1
    if row_steps < 1 or row_steps != col_steps:
        return set(), set()
    path: set[tuple[int, int]] = set()
    for t in range(row_steps + 1):
        path |= _move_cells(cells, dr * t, dc * t)
        path |= _move_cells(mirrored, -dr * t, -dc * t)
    path.add((axis_r, axis_c))
    return mirrored, path


def _render_input(shape: tuple[int, int], specs: list[dict], axis_r: int, axis_c: int) -> np.ndarray:
    g = np.zeros(shape, dtype=int)
    g[axis_r, :] = GREEN
    g[:, axis_c] = GREEN
    for spec in specs:
        for r, c in spec["cells"]:
            g[r, c] = spec["color"]
    return g


def _sample_shapes() -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    chosen = random.sample(ALL_SHAPES, 3)
    return chosen[0], chosen[1], chosen[2]


def _try_place_horizontal_pair(
    h: int,
    w: int,
    axis_c: int,
    sig: tuple[tuple[int, int], ...],
    half: str,
    colors: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved_empty: set[tuple[int, int]],
    path_used: set[tuple[int, int]],
) -> dict | None:
    sh, sw = _bbox_from_sig(sig)
    row_min = 0 if half == "top" else h // 2 + 1
    row_max = h // 2 - sh if half == "top" else h - sh
    rows = list(range(row_min, row_max + 1))
    random.shuffle(rows)
    left_cols = list(range(0, axis_c - sw + 1))
    right_cols = list(range(axis_c + 1, w - sw + 1))
    pairs = [(lc, rc) for lc in left_cols for rc in right_cols if rc - lc - sw >= 3 and (rc - lc - sw) % 2 == 1]
    random.shuffle(pairs)

    for r0 in rows:
        for lc, rc in pairs:
            gap = rc - lc - sw
            steps = (gap - 1) // 2
            mid_c = lc + sw - 1 + steps + 1
            if mid_c == axis_c:
                continue
            left = _coords_at(sig, r0, lc)
            right = _coords_at(sig, r0, rc)
            actual = left | right
            if any(c == axis_c for _, c in actual):
                continue
            if actual & reserved_empty:
                continue
            if not _buffer_ok(actual, occupied):
                continue
            path = _path_cells_pair(sig, (r0, lc), (r0, rc), "h")
            if not path or path & path_used:
                continue
            return {
                "objects": [
                    {"color": colors[0], "cells": left, "sig": sig},
                    {"color": colors[1], "cells": right, "sig": sig},
                ],
                "path": path,
            }
    return None


def _try_place_vertical_pair(
    h: int,
    w: int,
    axis_r: int,
    sig: tuple[tuple[int, int], ...],
    side: str,
    colors: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved_empty: set[tuple[int, int]],
    path_used: set[tuple[int, int]],
) -> dict | None:
    sh, sw = _bbox_from_sig(sig)
    col_min = 0 if side == "left" else w // 2 + 1
    col_max = w // 2 - sw if side == "left" else w - sw
    cols = list(range(col_min, col_max + 1))
    random.shuffle(cols)
    top_rows = list(range(0, axis_r - sh + 1))
    bottom_rows = list(range(axis_r + 1, h - sh + 1))
    pairs = [(tr, br) for tr in top_rows for br in bottom_rows if br - tr - sh >= 3 and (br - tr - sh) % 2 == 1]
    random.shuffle(pairs)

    for c0 in cols:
        for tr, br in pairs:
            gap = br - tr - sh
            steps = (gap - 1) // 2
            mid_r = tr + sh - 1 + steps + 1
            if mid_r == axis_r:
                continue
            top = _coords_at(sig, tr, c0)
            bottom = _coords_at(sig, br, c0)
            actual = top | bottom
            if any(r == axis_r for r, _ in actual):
                continue
            if actual & reserved_empty:
                continue
            if not _buffer_ok(actual, occupied):
                continue
            path = _path_cells_pair(sig, (tr, c0), (br, c0), "v")
            if not path or path & path_used:
                continue
            return {
                "objects": [
                    {"color": colors[0], "cells": top, "sig": sig},
                    {"color": colors[1], "cells": bottom, "sig": sig},
                ],
                "path": path,
            }
    return None


def _try_place_unmatched(
    h: int,
    w: int,
    axis_r: int,
    axis_c: int,
    sig: tuple[tuple[int, int], ...],
    color: int,
    occupied: set[tuple[int, int]],
    reserved_empty: set[tuple[int, int]],
    path_used: set[tuple[int, int]],
) -> dict | None:
    sh, sw = _bbox_from_sig(sig)
    quadrants = ["TL", "TR", "BL", "BR"]
    random.shuffle(quadrants)
    for quad in quadrants:
        max_d = min(axis_r - sh, axis_c - sw) if quad == "TL" else None
        if quad == "TR":
            max_d = min(axis_r - sh, w - axis_c - 1 - sw)
        if quad == "BL":
            max_d = min(h - axis_r - 1 - sh, axis_c - sw)
        if quad == "BR":
            max_d = min(h - axis_r - 1 - sh, w - axis_c - 1 - sw)
        if max_d is None or max_d < 1:
            continue
        ds = list(range(1, max_d + 1))
        random.shuffle(ds)
        for d in ds:
            if quad == "TL":
                pos = (axis_r - d - sh, axis_c - d - sw)
            elif quad == "TR":
                pos = (axis_r - d - sh, axis_c + 1 + d)
            elif quad == "BL":
                pos = (axis_r + 1 + d, axis_c - d - sw)
            else:
                pos = (axis_r + 1 + d, axis_c + 1 + d)
            cells = _coords_at(sig, *pos)
            mirrored, path = _path_cells_unmatched(sig, pos, axis_r, axis_c)
            if not mirrored or not path:
                continue
            if cells & reserved_empty or mirrored & occupied:
                continue
            if cells & occupied or mirrored & reserved_empty:
                continue
            if not _buffer_ok(cells, occupied):
                continue
            if path & path_used:
                continue
            return {
                "object": {"color": color, "cells": cells, "sig": sig},
                "mirrored": mirrored,
                "path": path,
            }
    return None


def _quadrant_of_bbox(bbox: tuple[int, int, int, int], axis_r: int, axis_c: int) -> str:
    r0, c0, r1, c1 = bbox
    assert r1 < axis_r or r0 > axis_r
    assert c1 < axis_c or c0 > axis_c
    if r1 < axis_r and c1 < axis_c:
        return "TL"
    if r1 < axis_r and c0 > axis_c:
        return "TR"
    if r0 > axis_r and c1 < axis_c:
        return "BL"
    return "BR"


def _pair_path_from_components(a: dict, b: dict) -> set[tuple[int, int]]:
    orient = _pair_orientation(a, b)
    if orient == "h":
        left, right = (a, b) if a["bbox"][1] < b["bbox"][1] else (b, a)
        pos_a = (left["bbox"][0], left["bbox"][1])
        pos_b = (right["bbox"][0], right["bbox"][1])
    else:
        top, bottom = (a, b) if a["bbox"][0] < b["bbox"][0] else (b, a)
        pos_a = (top["bbox"][0], top["bbox"][1])
        pos_b = (bottom["bbox"][0], bottom["bbox"][1])
    return _path_cells_pair(a["sig"], pos_a, pos_b, orient)


def _validate_instance(g: np.ndarray) -> None:
    axis_r, axis_c = _find_cross(g)
    comps = _components(g)
    assert len(comps) == 5
    assert len({comp["color"] for comp in comps}) == 5

    by_sig: dict[tuple[tuple[int, int], ...], list[dict]] = defaultdict(list)
    for comp in comps:
        by_sig[comp["sig"]].append(comp)
        assert 3 <= len(comp["cells"]) <= 5
        assert _quadrant_of_bbox(comp["bbox"], axis_r, axis_c) in {"TL", "TR", "BL", "BR"}

    sizes = sorted(len(group) for group in by_sig.values())
    assert sizes == [1, 2, 2]

    all_paths: list[set[tuple[int, int]]] = []
    for group in by_sig.values():
        if len(group) == 2:
            a, b = group
            assert _quadrant_of_bbox(a["bbox"], axis_r, axis_c) != _quadrant_of_bbox(b["bbox"], axis_r, axis_c)
            path = _pair_path_from_components(a, b)
            assert path
            all_paths.append(path)
        else:
            comp = group[0]
            mirrored = _reflect_cells(comp["cells"], axis_r, axis_c)
            assert all(g[r, c] == BLACK for r, c in mirrored)
            _, path = _path_cells_unmatched(comp["sig"], (comp["bbox"][0], comp["bbox"][1]), axis_r, axis_c)
            assert path
            all_paths.append(path)

    for i in range(len(all_paths)):
        for j in range(i + 1, len(all_paths)):
            assert not (all_paths[i] & all_paths[j])


def _attempt_generate() -> np.ndarray | None:
    h = random.choice((19, 21, 23, 25, 27))
    w = random.choice((19, 21, 23, 25, 27))
    axis_r = h // 2
    axis_c = w // 2
    pair_a_sig, pair_b_sig, single_sig = _sample_shapes()
    colors = list(INPUT_COLORS)
    random.shuffle(colors)

    occupied: set[tuple[int, int]] = set()
    reserved_empty: set[tuple[int, int]] = set()
    path_used: set[tuple[int, int]] = set()
    specs: list[dict] = []

    layout = random.choice(("HH", "VV"))
    if layout == "HH":
        pair1 = _try_place_horizontal_pair(h, w, axis_c, pair_a_sig, "top", (colors[0], colors[1]), occupied, reserved_empty, path_used)
        if pair1 is None:
            return None
        for obj in pair1["objects"]:
            occupied |= obj["cells"]
            specs.append(obj)
        path_used |= pair1["path"]

        pair2 = _try_place_horizontal_pair(h, w, axis_c, pair_b_sig, "bottom", (colors[2], colors[3]), occupied, reserved_empty, path_used)
        if pair2 is None:
            return None
        for obj in pair2["objects"]:
            occupied |= obj["cells"]
            specs.append(obj)
        path_used |= pair2["path"]
    else:
        pair1 = _try_place_vertical_pair(h, w, axis_r, pair_a_sig, "left", (colors[0], colors[1]), occupied, reserved_empty, path_used)
        if pair1 is None:
            return None
        for obj in pair1["objects"]:
            occupied |= obj["cells"]
            specs.append(obj)
        path_used |= pair1["path"]

        pair2 = _try_place_vertical_pair(h, w, axis_r, pair_b_sig, "right", (colors[2], colors[3]), occupied, reserved_empty, path_used)
        if pair2 is None:
            return None
        for obj in pair2["objects"]:
            occupied |= obj["cells"]
            specs.append(obj)
        path_used |= pair2["path"]

    single = _try_place_unmatched(h, w, axis_r, axis_c, single_sig, colors[4], occupied, reserved_empty, path_used)
    if single is None:
        return None
    specs.append(single["object"])
    occupied |= single["object"]["cells"]
    reserved_empty |= single["mirrored"]
    path_used |= single["path"]

    g = _render_input((h, w), specs, axis_r, axis_c)
    _validate_instance(g)
    out = main(g)
    assert out.shape == g.shape
    assert np.any(out == MAGENTA)
    assert np.any(out == GREY)
    return g


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
