from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    for dr, dc in ORTHO_DIRS:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_nonzero(patch: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    coords = np.argwhere(patch != BLACK)
    if coords.size == 0:
        return patch.copy(), (0, 0)
    r0, c0, r1, c1 = _bbox(coords)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0)


def _hole_count_from_mask(mask: np.ndarray) -> int:
    holes = 0
    for comp in _components(~mask):
        if np.any(
            (comp[:, 0] == 0)
            | (comp[:, 1] == 0)
            | (comp[:, 0] == mask.shape[0] - 1)
            | (comp[:, 1] == mask.shape[1] - 1)
        ):
            continue
        holes += 1
    return holes


def _object_holes(grid: np.ndarray, object_coords: np.ndarray) -> list[np.ndarray]:
    r0, c0, r1, c1 = _bbox(object_coords)
    obj_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    obj_mask[object_coords[:, 0] - r0, object_coords[:, 1] - c0] = True
    holes: list[np.ndarray] = []
    for comp in _components(~obj_mask):
        if np.any(
            (comp[:, 0] == 0)
            | (comp[:, 1] == 0)
            | (comp[:, 0] == obj_mask.shape[0] - 1)
            | (comp[:, 1] == obj_mask.shape[1] - 1)
        ):
            continue
        holes.append(comp + np.array([r0, c0], dtype=int))
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each holed grey object by hole-count-selected concentric rings and erase the hole-free grey objects.

    Input structure:
    - The grid uses black background, grey 4-connected objects, and one red control pixel inside every enclosed hole.
    - Different grey objects are separated from one another; some have no enclosed holes, some have one, and some have multiple.
    - Each hole is a 4-connected non-grey interior region inside one object's bbox, containing exactly one red control pixel and otherwise black cells.

    Rule:
    - Find every grey 4-connected object and count its enclosed holes by examining non-grey regions inside that object's bbox that do not touch the bbox boundary.
    - If an object has no holes, erase all of its grey pixels to black.
    - If an object has holes, keep its grey boundary and red control pixels. For every black cell in each hole, compute Chebyshev distance to that hole's red control pixel.
    - Objects with exactly one hole use odd distance -> yellow and even distance (>0) -> blue; objects with two or more holes use odd distance -> orange and even distance (>0) -> green.

    Key insight:
    - The visible ring colors are not determined per hole in isolation; they are gated by the total number of enclosed holes in the surrounding grey object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    objects = _components(g == GREY)
    for obj in objects:
        holes = _object_holes(g, obj)
        hole_count = len(holes)
        if hole_count == 0:
            out[obj[:, 0], obj[:, 1]] = BLACK
            continue

        odd_color, even_color = (YELLOW, BLUE) if hole_count == 1 else (ORANGE, GREEN)
        for hole in holes:
            red_points = hole[g[hole[:, 0], hole[:, 1]] == RED]
            assert red_points.shape[0] == 1, "each hole must contain exactly one red control pixel"
            control_r, control_c = map(int, red_points[0])
            for r, c in hole:
                rr = int(r)
                cc = int(c)
                value = int(g[rr, cc])
                assert value in (BLACK, RED), "hole interiors may only contain black cells and one red control"
                if value != BLACK:
                    continue
                dist = max(abs(rr - control_r), abs(cc - control_c))
                assert dist > 0, "distance-zero hole cell must be the red control pixel"
                out[rr, cc] = odd_color if dist % 2 == 1 else even_color

    return out


def _choose_control(hole_h: int, hole_w: int) -> tuple[int, int]:
    return random.randint(2, hole_h - 3), random.randint(2, hole_w - 3)


def _add_tabs(patch: np.ndarray, core_bbox: tuple[int, int, int, int], min_tabs: int, max_tabs: int) -> None:
    r0, c0, r1, c1 = core_bbox
    for _ in range(random.randint(min_tabs, max_tabs)):
        side = random.choice(["top", "bottom", "left", "right"])
        if side in ("top", "bottom"):
            tab_h = random.randint(1, 3)
            tab_w = random.randint(1, max(1, c1 - c0))
            start = random.randint(c0, c1 - tab_w + 1)
            if side == "top":
                patch[r0 - tab_h : r0, start : start + tab_w] = GREY
            else:
                patch[r1 + 1 : r1 + 1 + tab_h, start : start + tab_w] = GREY
        else:
            tab_h = random.randint(1, max(1, r1 - r0))
            tab_w = random.randint(1, 3)
            start = random.randint(r0, r1 - tab_h + 1)
            if side == "left":
                patch[start : start + tab_h, c0 - tab_w : c0] = GREY
            else:
                patch[start : start + tab_h, c1 + 1 : c1 + 1 + tab_w] = GREY


def _ring_patch(hole_count: int, hole_h: int, hole_w: int, orientation: str | None = None):
    margin = 4
    holes_meta = []
    if hole_count == 1:
        outer_h = hole_h + 2
        outer_w = hole_w + 2
        patch = np.zeros((outer_h + 2 * margin, outer_w + 2 * margin), dtype=int)
        r0 = c0 = margin
        patch[r0 : r0 + outer_h, c0 : c0 + outer_w] = GREY
        hole_r0 = r0 + 1
        hole_c0 = c0 + 1
        patch[hole_r0 : hole_r0 + hole_h, hole_c0 : hole_c0 + hole_w] = BLACK
        ctrl_dr, ctrl_dc = _choose_control(hole_h, hole_w)
        control = (hole_r0 + ctrl_dr, hole_c0 + ctrl_dc)
        patch[control] = RED
        holes_meta.append(
            {
                "shape": (hole_h, hole_w),
                "control": control,
                "cells": [(hole_r0 + dr, hole_c0 + dc) for dr in range(hole_h) for dc in range(hole_w)],
            }
        )
        _add_tabs(patch, (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1), 0, 3)
    else:
        if orientation is None:
            orientation = random.choice(["horizontal", "vertical"])
        outer_h = hole_h + 2 if orientation == "horizontal" else 2 + hole_count * hole_h + (hole_count - 1)
        outer_w = 2 + hole_count * hole_w + (hole_count - 1) if orientation == "horizontal" else hole_w + 2
        patch = np.zeros((outer_h + 2 * margin, outer_w + 2 * margin), dtype=int)
        r0 = c0 = margin
        patch[r0 : r0 + outer_h, c0 : c0 + outer_w] = GREY
        for idx in range(hole_count):
            if orientation == "horizontal":
                hole_r0 = r0 + 1
                hole_c0 = c0 + 1 + idx * (hole_w + 1)
            else:
                hole_r0 = r0 + 1 + idx * (hole_h + 1)
                hole_c0 = c0 + 1
            patch[hole_r0 : hole_r0 + hole_h, hole_c0 : hole_c0 + hole_w] = BLACK
            ctrl_dr, ctrl_dc = _choose_control(hole_h, hole_w)
            control = (hole_r0 + ctrl_dr, hole_c0 + ctrl_dc)
            patch[control] = RED
            holes_meta.append(
                {
                    "shape": (hole_h, hole_w),
                    "control": control,
                    "cells": [(hole_r0 + dr, hole_c0 + dc) for dr in range(hole_h) for dc in range(hole_w)],
                }
            )
        _add_tabs(patch, (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1), 1, 3)

    cropped, offset = _crop_nonzero(patch)
    off_r, off_c = offset
    norm_holes = []
    for hole in holes_meta:
        norm_holes.append(
            {
                "shape": hole["shape"],
                "control": (hole["control"][0] - off_r, hole["control"][1] - off_c),
                "cells": [(r - off_r, c - off_c) for r, c in hole["cells"]],
            }
        )
    return cropped, {"hole_count": hole_count, "holes": norm_holes}


def _grow_nohole_blob(target_area: int) -> np.ndarray | None:
    mode = random.choice(["compact", "compact", "winding"])
    coords = {(0, 0)}
    max_span = random.randint(6, 8)
    while len(coords) < target_area:
        candidates = []
        weights = []
        for r, c in tuple(coords):
            for dr, dc in ORTHO_DIRS:
                nr, nc = r + dr, c + dc
                if (nr, nc) in coords:
                    continue
                trial = coords | {(nr, nc)}
                arr = np.array(list(trial), dtype=int)
                arr -= arr.min(axis=0)
                h, w = arr.max(axis=0) + 1
                if int(h) > max_span or int(w) > max_span:
                    continue
                mask = np.zeros((int(h), int(w)), dtype=bool)
                mask[arr[:, 0], arr[:, 1]] = True
                if _hole_count_from_mask(mask) != 0:
                    continue
                adj = sum((nr + adr, nc + adc) in coords for adr, adc in ORTHO_DIRS)
                if mode == "compact":
                    weight = 1.0 + 3.0 * adj + max(0, 6 - max(int(h), int(w)))
                else:
                    weight = 1.0 + (4 - adj) + max(0, int(h) + int(w) - 4) * 0.25
                candidates.append((nr, nc))
                weights.append(weight)
        if not candidates:
            return None
        coords.add(random.choices(candidates, weights=weights, k=1)[0])

    arr = np.array(list(coords), dtype=int)
    arr -= arr.min(axis=0)
    h, w = arr.max(axis=0) + 1
    patch = np.zeros((int(h), int(w)), dtype=int)
    patch[arr[:, 0], arr[:, 1]] = GREY
    return patch


def _nohole_patch(target_area: int):
    for _ in range(120):
        patch = _grow_nohole_blob(target_area)
        if patch is None:
            continue
        mask = patch == GREY
        if _hole_count_from_mask(mask) != 0:
            continue
        return patch, {"hole_count": 0, "holes": []}
    return None


def _materialize_expected(input_grid: np.ndarray, placed: list[dict]) -> np.ndarray:
    out = input_grid.copy()
    for item in placed:
        top = int(item["top"])
        left = int(item["left"])
        patch = item["patch"]
        meta = item["meta"]
        grey_coords = np.argwhere(patch == GREY)
        if meta["hole_count"] == 0:
            out[top + grey_coords[:, 0], left + grey_coords[:, 1]] = BLACK
            continue

        odd_color, even_color = (YELLOW, BLUE) if meta["hole_count"] == 1 else (ORANGE, GREEN)
        for hole in meta["holes"]:
            control_r = top + hole["control"][0]
            control_c = left + hole["control"][1]
            for local_r, local_c in hole["cells"]:
                rr = top + local_r
                cc = left + local_c
                if (rr, cc) == (control_r, control_c):
                    continue
                dist = max(abs(rr - control_r), abs(cc - control_c))
                out[rr, cc] = odd_color if dist % 2 == 1 else even_color
    return out


def _can_place(occupied: np.ndarray, patch: np.ndarray, top: int, left: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    ph, pw = patch.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    pts = np.argwhere(patch != BLACK)
    for pr, pc in pts:
        gr = top + int(pr)
        gc = left + int(pc)
        for rr in range(max(0, gr - pad), min(h, gr + pad + 1)):
            for cc in range(max(0, gc - pad), min(w, gc + pad + 1)):
                if occupied[rr, cc]:
                    return False
    return True


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    candidates = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    random.shuffle(candidates)
    for top, left in candidates[:250]:
        if not _can_place(occupied, patch, top, left, pad=1):
            continue
        rows, cols = np.where(patch != BLACK)
        grid[top + rows, left + cols] = patch[rows, cols]
        occupied[top + rows, left + cols] = True
        return top, left
    return None


def _validate_structure(grid: np.ndarray, placed: list[dict], expected: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30 or max(expected.shape) > 30:
        return False

    objects = _components(grid == GREY)
    if len(objects) != len(placed):
        return False

    hole_counts = sorted(int(len(_object_holes(grid, obj))) for obj in objects)
    if 0 not in hole_counts or 1 not in hole_counts or max(hole_counts) < 2:
        return False

    single_shapes = []
    multi_shapes = []
    nohole_area = None
    for item in placed:
        meta = item["meta"]
        grey_area = int(np.count_nonzero(item["patch"] == GREY))
        if meta["hole_count"] == 0:
            nohole_area = grey_area if nohole_area is None else max(nohole_area, grey_area)
        elif meta["hole_count"] == 1:
            single_shapes.extend(hole["shape"] for hole in meta["holes"])
        else:
            multi_shapes.extend(hole["shape"] for hole in meta["holes"])
        for hole in meta["holes"]:
            top = int(item["top"])
            left = int(item["left"])
            ctrl_r = top + hole["control"][0]
            ctrl_c = left + hole["control"][1]
            if int(grid[ctrl_r, ctrl_c]) != RED:
                return False
            max_dist = max(max(abs(top + r - ctrl_r), abs(left + c - ctrl_c)) for r, c in hole["cells"])
            if max_dist < 2:
                return False

    if nohole_area is None or nohole_area < 18:
        return False
    if not set(single_shapes) & set(multi_shapes):
        return False

    changed = expected != grid
    if not np.any(changed):
        return False
    changed_colors = set(map(int, expected[changed].ravel()))
    if not {BLUE, YELLOW, GREEN, ORANGE}.issubset(changed_colors | set(map(int, expected.ravel()))):
        return False

    return np.array_equal(main(grid), expected)


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    shared_h = random.randint(5, 6)
    shared_w = random.randint(5, 6)

    single_patch, single_meta = _ring_patch(1, shared_h, shared_w)
    multi_holes = random.choice([2, 2, 3])
    multi_patch, multi_meta = _ring_patch(multi_holes, shared_h, shared_w, random.choice(["horizontal", "vertical"]))

    single_area = int(np.count_nonzero(single_patch == GREY))
    multi_area = int(np.count_nonzero(multi_patch == GREY))
    min_nohole = max(18, min(single_area, multi_area) - 3)
    max_nohole = max(min_nohole, min(34, max(single_area, multi_area)))
    nohole_target = random.randint(min_nohole, max_nohole)
    nohole = _nohole_patch(nohole_target)
    if nohole is None:
        return None
    nohole_patch, nohole_meta = nohole

    pieces = [
        {"patch": nohole_patch, "meta": nohole_meta},
        {"patch": single_patch, "meta": single_meta},
        {"patch": multi_patch, "meta": multi_meta},
    ]

    if random.random() < 0.4:
        extra_type = random.choice(["nohole", "single"])
        if extra_type == "nohole":
            extra = _nohole_patch(random.randint(12, 24))
        else:
            eh = random.randint(5, 6)
            ew = random.randint(5, 6)
            extra = _ring_patch(1, eh, ew)
        if extra is not None:
            pieces.append({"patch": extra[0], "meta": extra[1]})

    max_h = max(piece["patch"].shape[0] for piece in pieces)
    max_w = max(piece["patch"].shape[1] for piece in pieces)
    total_area = sum((piece["patch"] != BLACK).sum() for piece in pieces)

    h_min = max(18, max_h + 2)
    if h_min > 28:
        return None
    h_max = min(28, max(h_min, 22 + random.randint(-2, 3)))
    h = random.randint(h_min, h_max)

    w_floor = max(max_w + 2, int(np.ceil(total_area / max(h - 4, 1))) + 6, 18)
    w_min = min(30, max(18, w_floor))
    if w_min > 30:
        return None
    w_max = min(30, max(w_min, 24 + random.randint(-2, 4)))
    w = random.randint(w_min, w_max)

    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    placed: list[dict] = []
    for piece in sorted(pieces, key=lambda item: int(np.count_nonzero(item["patch"])), reverse=True):
        loc = _place_patch(grid, occupied, piece["patch"])
        if loc is None:
            return None
        placed.append({"patch": piece["patch"], "meta": piece["meta"], "top": loc[0], "left": loc[1]})

    expected = _materialize_expected(grid, placed)
    if not _validate_structure(grid, placed, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_scene()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid topology-gated concentric-ring input")
