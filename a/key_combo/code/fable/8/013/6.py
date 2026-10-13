from __future__ import annotations

import itertools
import random
from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]
Shape = frozenset[Coord]

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400
SEARCH_LIMIT = 2


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(coords: set[Coord] | Shape) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: set[Coord] | Shape) -> Shape:
    r0, c0, _r1, _c1 = _bbox(coords)
    return frozenset((r - r0, c - c0) for r, c in coords)


def _shape_dims(shape: Shape) -> tuple[int, int]:
    _r0, _c0, r1, c1 = _bbox(set(shape))
    return r1 + 1, c1 + 1


def _translate(shape: Shape, dr: int, dc: int) -> Shape:
    return frozenset((r + dr, c + dc) for r, c in shape)


def _components_from_mask(mask: np.ndarray) -> list[set[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[set[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            comp: set[Coord] = set()
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.add((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _color_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        for coords in _components_from_mask(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            comps.append(
                {
                    "color": int(color),
                    "coords": frozenset(coords),
                    "bbox": (r0, c0, r1, c1),
                }
            )
    comps.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    return comps


def _find_bar_bbox(grid: np.ndarray, color: int, *, top: bool) -> tuple[int, int, int, int]:
    h, w = grid.shape
    candidates = []
    for comp in _components_from_mask(grid == color):
        r0, c0, r1, c1 = _bbox(comp)
        if c0 != 0 or c1 != w - 1:
            continue
        if top and r0 != 0:
            continue
        if not top and r1 != h - 1:
            continue
        candidates.append((len(comp), (r0, c0, r1, c1)))
    assert candidates, "missing border bar"
    candidates.sort(reverse=True)
    return candidates[0][1]


def _extract_notches(grid: np.ndarray, color: int, bar_bbox: tuple[int, int, int, int], side: str) -> list[dict]:
    r0, c0, r1, c1 = bar_bbox
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    notch_mask = patch != int(color)
    notches = []
    for local_coords in _components_from_mask(notch_mask):
        norm = _normalize(local_coords)
        global_coords = frozenset((r0 + r, c0 + c) for r, c in local_coords)
        rr0, cc0, rr1, cc1 = _bbox(global_coords)
        notches.append(
            {
                "shape": norm,
                "cells": global_coords,
                "side": side,
                "bbox": (rr0, cc0, rr1, cc1),
            }
        )
    notches.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return notches


def _parse_puzzle(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    top_color = int(g[0, 0])
    bottom_color = int(g[-1, 0])
    assert top_color != bottom_color != bg

    top_bbox = _find_bar_bbox(g, top_color, top=True)
    bottom_bbox = _find_bar_bbox(g, bottom_color, top=False)
    top_h = top_bbox[2] + 1
    bottom_r0 = bottom_bbox[0]
    assert top_h < bottom_r0, "bars must leave a middle field"

    notches = _extract_notches(g, top_color, top_bbox, "top") + _extract_notches(g, bottom_color, bottom_bbox, "bottom")
    shape_keys = [tuple(sorted(item["shape"])) for item in notches]
    assert len(shape_keys) == len(set(shape_keys)), "notch shapes must be distinct"

    field = g[top_h:bottom_r0, :]
    comps = _color_components(field, bg)
    assert len(comps) == len(notches), "number of visible objects must match notch count"

    for item in notches:
        item["shape_key"] = tuple(sorted(item["shape"]))
    notches.sort(key=lambda item: (item["side"], item["bbox"][1], item["bbox"][0]))
    return {
        "grid": g,
        "bg": int(bg),
        "top_color": int(top_color),
        "bottom_color": int(bottom_color),
        "top_h": int(top_h),
        "bottom_r0": int(bottom_r0),
        "field": field,
        "notches": notches,
        "components": comps,
    }


def _enumerate_candidates(comp: dict, field: np.ndarray, bg: int, shapes: list[dict]) -> list[dict]:
    fh, fw = field.shape
    coords = set(comp["coords"])
    color = int(comp["color"])
    out: dict[tuple[int, int, int], dict] = {}

    for shape_idx, notch in enumerate(shapes):
        shape = notch["shape"]
        for cr, cc in coords:
            for sr, sc in shape:
                dr = cr - sr
                dc = cc - sc
                placed = _translate(shape, dr, dc)
                if any(r < 0 or r >= fh or c < 0 or c >= fw for r, c in placed):
                    continue
                if not coords.issubset(placed):
                    continue
                hidden_ok = True
                for r, c in placed:
                    if (r, c) in coords:
                        continue
                    val = int(field[r, c])
                    if val == int(bg) or val == color:
                        hidden_ok = False
                        break
                if not hidden_ok:
                    continue
                key = (shape_idx, dr, dc)
                out[key] = {
                    "shape_idx": int(shape_idx),
                    "origin": (int(dr), int(dc)),
                    "cells": placed,
                    "color": color,
                }
    candidates = list(out.values())
    candidates.sort(key=lambda item: (item["shape_idx"], item["origin"][0], item["origin"][1]))
    return candidates


def _paint_field(shape_hw: tuple[int, int], bg: int, objects: list[dict], order: tuple[int, ...]) -> np.ndarray:
    field = np.full(shape_hw, int(bg), dtype=int)
    for idx in order:
        obj = objects[idx]
        for r, c in obj["cells"]:
            field[r, c] = int(obj["color"])
    return field


def _has_valid_overlay(field: np.ndarray, bg: int, objects: list[dict]) -> bool:
    for i, j in itertools.combinations(range(len(objects)), 2):
        if objects[i]["color"] == objects[j]["color"] and objects[i]["cells"] & objects[j]["cells"]:
            return False
    for order in itertools.permutations(range(len(objects))):
        painted = _paint_field(field.shape, bg, objects, order)
        if np.array_equal(painted, field):
            return True
    return False


def _recover_solutions(grid: np.ndarray, max_solutions: int = SEARCH_LIMIT) -> tuple[dict, list[list[dict]]]:
    parsed = _parse_puzzle(grid)
    field = parsed["field"]
    comps = parsed["components"]
    shapes = parsed["notches"]
    bg = parsed["bg"]

    candidate_lists = [_enumerate_candidates(comp, field, bg, shapes) for comp in comps]
    assert all(cands for cands in candidate_lists), "every visible component must match at least one notch shape"

    order = sorted(range(len(comps)), key=lambda idx: (len(candidate_lists[idx]), comps[idx]["bbox"][0], comps[idx]["bbox"][1]))
    assigned: list[dict | None] = [None] * len(comps)
    used_shapes: set[int] = set()
    solutions: list[list[dict]] = []

    def backtrack(pos: int) -> None:
        if len(solutions) >= max_solutions:
            return
        if pos == len(order):
            objects = [item for item in assigned if item is not None]
            if _has_valid_overlay(field, bg, objects):
                solutions.append(objects)
            return

        comp_idx = order[pos]
        for cand in candidate_lists[comp_idx]:
            shape_idx = int(cand["shape_idx"])
            if shape_idx in used_shapes:
                continue
            conflict = False
            for other in assigned:
                if other is None:
                    continue
                overlap = cand["cells"] & other["cells"]
                if overlap and cand["color"] == other["color"]:
                    conflict = True
                    break
            if conflict:
                continue

            assigned[comp_idx] = cand
            used_shapes.add(shape_idx)
            backtrack(pos + 1)
            used_shapes.remove(shape_idx)
            assigned[comp_idx] = None

    backtrack(0)
    return parsed, solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each occluded field object from the border-notch shape dictionary, then move every object into its matching notch.

    Input structure:
    - The top and bottom of the grid are full-width thick bars in two different colors.
    - Their inner edges contain connected background notches of distinct shapes.
    - The middle field contains the same number of colored objects, using only the two bar colors; one object may be partially hidden by another.

    Rule:
    - Treat the notch shapes as the complete-shape dictionary for the field objects.
    - For each visible field fragment, infer which translated notch shape could contain it, with every missing cell explained by overlap from a differently colored top object.
    - Choose the unique full set of objects whose layered repaint reproduces the observed field, then clear the field and paint each recovered object into the notch with the same shape, preserving the object's own color.

    Key insight:
    - Color does not determine the destination bar: the notches define the object library, and overlap constraints reveal which full shape each fragment must really be.
    """
    parsed, solutions = _recover_solutions(grid, max_solutions=2)
    assert len(solutions) == 1, "expected a unique reconstruction"
    objects = solutions[0]

    g = parsed["grid"]
    out = np.full_like(g, parsed["bg"])
    out[: parsed["top_h"], :] = parsed["top_color"]
    out[parsed["bottom_r0"] :, :] = parsed["bottom_color"]

    for obj in objects:
        notch = parsed["notches"][int(obj["shape_idx"])]
        for r, c in notch["cells"]:
            out[r, c] = int(obj["color"])
    return out


def _random_connected_shape() -> Shape | None:
    target_area = random.randint(4, 7)
    cells = {(0, 0)}
    while len(cells) < target_area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
        norm = _normalize(cells)
        h, w = _shape_dims(norm)
        if h > 4 or w > 5:
            return None
    norm = _normalize(cells)
    h, w = _shape_dims(norm)
    if h < 2 or w < 2:
        return None
    return norm


def _shapes_with_overlap(shape_a: Shape, shape_b: Shape, fh: int, fw: int) -> list[tuple[Coord, Coord, Shape]]:
    ha, wa = _shape_dims(shape_a)
    hb, wb = _shape_dims(shape_b)
    options = []
    for ar in range(0, fh - ha + 1):
        for ac in range(0, fw - wa + 1):
            placed_a = _translate(shape_a, ar, ac)
            for br in range(0, fh - hb + 1):
                for bc in range(0, fw - wb + 1):
                    placed_b = _translate(shape_b, br, bc)
                    overlap = placed_a & placed_b
                    if not overlap or overlap == placed_a:
                        continue
                    visible_a = placed_a - overlap
                    if len(visible_a) < 2:
                        continue
                    if len(_components_from_mask(_cells_to_mask(visible_a, fh, fw))) != 1:
                        continue
                    options.append(((ar, ac), (br, bc), frozenset(visible_a)))
    random.shuffle(options)
    return options


def _cells_to_mask(cells: set[Coord] | Shape, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _sample_nonoverlap_position(shape: Shape, fh: int, fw: int, occupied: set[Coord], blocked: set[Coord], margin: int = 1) -> Coord | None:
    h, w = _shape_dims(shape)
    choices = []
    for r in range(0, fh - h + 1):
        for c in range(0, fw - w + 1):
            placed = set(_translate(shape, r, c))
            if placed & occupied:
                continue
            expanded = set(placed)
            for rr, cc in list(placed):
                for dr in range(-margin, margin + 1):
                    for dc in range(-margin, margin + 1):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < fh and 0 <= nc < fw:
                            expanded.add((nr, nc))
            if expanded & blocked:
                continue
            choices.append((r, c))
    if not choices:
        return None
    return random.choice(choices)


def _build_observed_field(fh: int, fw: int, bg: int, objects: list[dict], order: list[int]) -> np.ndarray:
    field = np.full((fh, fw), int(bg), dtype=int)
    for idx in order:
        for r, c in objects[idx]["cells"]:
            field[r, c] = int(objects[idx]["color"])
    return field


def _make_shape_set(count: int) -> list[Shape] | None:
    for _ in range(100):
        shapes: list[Shape] = []
        keys: set[tuple[Coord, ...]] = set()
        while len(shapes) < count:
            shape = _random_connected_shape()
            if shape is None:
                break
            key = tuple(sorted(shape))
            if key in keys:
                continue
            shapes.append(shape)
            keys.add(key)
        if len(shapes) == count:
            return shapes
    return None


def _place_notches(shapes: list[Shape], h: int, w: int, top_h: int, bottom_h: int, top_color: int, bottom_color: int, bg: int) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.full((h, w), int(bg), dtype=int)
    grid[:top_h, :] = top_color
    grid[h - bottom_h :, :] = bottom_color

    n = len(shapes)
    top_count = random.randint(1, n - 1)
    sides = ["top"] * top_count + ["bottom"] * (n - top_count)
    random.shuffle(sides)

    notches: list[dict] = []
    for side in ("top", "bottom"):
        entries = [(idx, shapes[idx]) for idx in range(n) if sides[idx] == side]
        random.shuffle(entries)
        occupied_cols: list[tuple[int, int]] = []
        for idx, shape in entries:
            sh, sw = _shape_dims(shape)
            attempts = []
            for c in range(1, w - sw):
                c0 = c
                c1 = c + sw - 1
                if any(not (c1 + 1 < oc0 or oc1 + 1 < c0) for oc0, oc1 in occupied_cols):
                    continue
                attempts.append(c)
            if not attempts:
                return None
            c = random.choice(attempts)
            occupied_cols.append((c, c + sw - 1))
            if side == "top":
                r = top_h - sh
            else:
                r = h - bottom_h
            cells = _translate(shape, r, c)
            for rr, cc in cells:
                grid[rr, cc] = bg
            rr0, cc0, rr1, cc1 = _bbox(set(cells))
            notches.append(
                {
                    "shape_idx": idx,
                    "shape": shape,
                    "cells": cells,
                    "side": side,
                    "bbox": (rr0, cc0, rr1, cc1),
                }
            )
    notches.sort(key=lambda item: item["shape_idx"])
    return grid, notches


def _assign_object_colors(notches: list[dict], top_color: int, bottom_color: int) -> list[int]:
    colors = [random.choice([top_color, bottom_color]) for _ in notches]
    side_match = {
        "top": top_color,
        "bottom": bottom_color,
    }
    if all(colors[i] == side_match[notches[i]["side"]] for i in range(len(notches))):
        flip_idx = random.randrange(len(notches))
        colors[flip_idx] = bottom_color if side_match[notches[flip_idx]["side"]] == top_color else top_color
    return colors


def _generate_layout() -> np.ndarray | None:
    count = random.choice([3, 4])
    shapes = _make_shape_set(count)
    if shapes is None:
        return None
    max_h = max(_shape_dims(shape)[0] for shape in shapes)
    max_w = max(_shape_dims(shape)[1] for shape in shapes)

    top_h = random.randint(max_h + 1, min(6, max_h + 2))
    bottom_h = random.randint(max_h + 1, min(6, max_h + 2))
    field_h = random.randint(max(8, max_h + 4), 14)
    w_lo = max(16, max_w + 6)
    w = random.randint(w_lo, 28)
    h = top_h + bottom_h + field_h
    if h > 30:
        return None

    palette = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], 2)
    top_color, bottom_color = palette[0], palette[1]
    bg = 0

    placed = _place_notches(shapes, h, w, top_h, bottom_h, top_color, bottom_color, bg)
    if placed is None:
        return None
    grid, notches = placed
    colors = _assign_object_colors(notches, top_color, bottom_color)

    fh, fw = field_h, w
    hidden_idx = random.randrange(count)
    occluder_choices = [idx for idx in range(count) if idx != hidden_idx]
    random.shuffle(occluder_choices)
    occluder_idx = occluder_choices[0]
    colors[hidden_idx], colors[occluder_idx] = top_color, bottom_color
    if random.random() < 0.5:
        colors[hidden_idx], colors[occluder_idx] = colors[occluder_idx], colors[hidden_idx]
    side_match = {"top": top_color, "bottom": bottom_color}
    if all(colors[i] == side_match[notches[i]["side"]] for i in range(count)):
        flip_idx = random.randrange(count)
        colors[flip_idx] = bottom_color if colors[flip_idx] == top_color else top_color

    overlap_options = _shapes_with_overlap(shapes[hidden_idx], shapes[occluder_idx], fh, fw)
    if not overlap_options:
        return None

    occupied: set[Coord] = set()
    blocked: set[Coord] = set()
    placements: dict[int, Shape] = {}
    hidden_origin, occluder_origin, _hidden_visible = overlap_options[0]
    placements[hidden_idx] = _translate(shapes[hidden_idx], hidden_origin[0], hidden_origin[1])
    placements[occluder_idx] = _translate(shapes[occluder_idx], occluder_origin[0], occluder_origin[1])
    occupied |= set(placements[hidden_idx]) | set(placements[occluder_idx])
    for rr, cc in occupied:
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < fh and 0 <= nc < fw:
                    blocked.add((nr, nc))

    other_indices = [idx for idx in range(count) if idx not in (hidden_idx, occluder_idx)]
    random.shuffle(other_indices)
    for idx in other_indices:
        pos = _sample_nonoverlap_position(shapes[idx], fh, fw, occupied, blocked, margin=1)
        if pos is None:
            return None
        placements[idx] = _translate(shapes[idx], pos[0], pos[1])
        occupied |= set(placements[idx])
        for rr, cc in placements[idx]:
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < fh and 0 <= nc < fw:
                        blocked.add((nr, nc))

    objects = [{"shape_idx": idx, "cells": placements[idx], "color": colors[idx]} for idx in range(count)]
    order = [idx for idx in other_indices] + [hidden_idx, occluder_idx]
    field = _build_observed_field(fh, fw, bg, objects, order)

    comps = _color_components(field, bg)
    if len(comps) != count:
        return None
    if min(len(comp["coords"]) for comp in comps) < 2:
        return None

    grid[top_h : top_h + fh, :] = field
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_layout()
        if grid is None:
            continue
        try:
            parsed, solutions = _recover_solutions(grid, max_solutions=2)
            if len(solutions) != 1:
                continue
            out = main(grid)
        except AssertionError:
            continue

        if out.shape != grid.shape:
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        if len(np.unique(parsed["field"])) < 3:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle-6 input")
