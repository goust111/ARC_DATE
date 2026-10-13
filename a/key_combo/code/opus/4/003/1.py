from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, label_mask, paste


BACKGROUND = Color.BLACK
OBJECT_COLOR = Color.GREY
HOLE_TO_COLOR = {
    1: Color.BLUE,
    2: Color.RED,
    3: Color.GREEN,
}
MARKER_COLORS = tuple(HOLE_TO_COLOR.values())
SIDES = ("top", "bottom", "left", "right")

GEN_MIN_SIDE = 15
GEN_MAX_SIDE = 23
GEN_EDGE_MARGIN = 2
GEN_GAP = 2
GEN_MAX_ATTEMPTS = 500
GEN_COORD_TRIES = 24


def _hole_components(patch: np.ndarray) -> list[np.ndarray]:
    bg_labels, count = label_mask(patch == BACKGROUND, connectivity=4)
    holes: list[np.ndarray] = []
    h, w = patch.shape
    for idx in range(1, count + 1):
        coords = np.argwhere(bg_labels == idx)
        if coords.size == 0:
            continue
        touches_border = np.any(
            (coords[:, 0] == 0)
            | (coords[:, 0] == h - 1)
            | (coords[:, 1] == 0)
            | (coords[:, 1] == w - 1)
        )
        if not touches_border:
            holes.append(coords)
    return holes


def _analyze_objects(grid: np.ndarray) -> list[dict]:
    objects = find_components(
        grid,
        bg=BACKGROUND,
        colors=[OBJECT_COLOR],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    analyzed: list[dict] = []
    for obj in objects:
        r0, c0, r1, c1 = map(int, obj["bbox"])
        patch = crop_bbox(grid, (r0, c0, r1, c1))
        holes_local = _hole_components(patch)
        hole_count = len(holes_local)
        assert hole_count in HOLE_TO_COLOR, "each object must have 1-3 holes"
        holes_global = [
            [(r0 + int(rr), c0 + int(cc)) for rr, cc in coords]
            for coords in holes_local
        ]
        analyzed.append(
            {
                "bbox": (r0, c0, r1, c1),
                "coords": obj["coords"],
                "hole_count": hole_count,
                "fill_color": HOLE_TO_COLOR[hole_count],
                "holes_global": holes_global,
            }
        )
    assert analyzed, "expected at least one gray object"
    return analyzed


def _edge_markers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    coords: list[tuple[int, int]] = []
    coords.extend((0, c) for c in range(w))
    coords.extend((h - 1, c) for c in range(w))
    coords.extend((r, 0) for r in range(1, h - 1))
    coords.extend((r, w - 1) for r in range(1, h - 1))
    markers = []
    for r, c in coords:
        color = int(grid[r, c])
        if color in MARKER_COLORS:
            markers.append((r, c, color))
    return markers


def _marker_dirs(r: int, c: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    dirs: list[tuple[int, int]] = []
    if r == 0:
        dirs.append((1, 0))
    if r == h - 1:
        dirs.append((-1, 0))
    if c == 0:
        dirs.append((0, 1))
    if c == w - 1:
        dirs.append((0, -1))
    return dirs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each holed gray rectangle by its hole count, then fill only the holes reached by color-matching edge rays.

    Input structure:
    - The background is black, and every main object is a gray 4-connected rectangle-like component away from the border.
    - Each gray object contains 1-3 enclosed black hole regions inside its bounding box.
    - Border markers are single blue, red, or green pixels on the outer frame.

    Rule:
    - Count each gray object's 4-connected enclosed background holes inside its bounding box.
    - Recolor the gray body pixels by hole count: 1 hole -> blue, 2 holes -> red, 3 holes -> green, while leaving the holes black.
    - From every border marker, shoot an inward row or column ray, paint only background cells until the first non-background cell, and stop before that blocker.
    - If the blocker is a recolored object wall of the same color as the ray, fill all holes of that object with the object's color; otherwise leave its holes black.

    Key insight:
    - The ray can fill a hole only after the object's own color has been inferred from its hole count, so counting, recoloring, ray stopping, and conditional filling are all necessary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _analyze_objects(g)

    out = g.copy()
    for obj in objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = int(obj["fill_color"])

    base = out.copy()
    object_by_cell: dict[tuple[int, int], int] = {}
    for idx, obj in enumerate(objects):
        for r_raw, c_raw in obj["coords"]:
            object_by_cell[(int(r_raw), int(c_raw))] = idx

    planned: dict[tuple[int, int], int] = {}
    filled_objects: set[int] = set()

    for r, c, color in _edge_markers(g):
        for dr, dc in _marker_dirs(r, c, g.shape):
            rr = r + dr
            cc = c + dc
            while 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                value = int(base[rr, cc])
                if value == BACKGROUND:
                    prev = planned.get((rr, cc))
                    assert prev is None or prev == color, "conflicting ray colors"
                    planned[(rr, cc)] = color
                    rr += dr
                    cc += dc
                    continue

                obj_idx = object_by_cell.get((rr, cc))
                if obj_idx is not None and color == int(objects[obj_idx]["fill_color"]):
                    filled_objects.add(obj_idx)
                break

    for (r, c), color in planned.items():
        out[r, c] = color

    for idx in filled_objects:
        fill_color = int(objects[idx]["fill_color"])
        for hole in objects[idx]["holes_global"]:
            for r, c in hole:
                out[r, c] = fill_color

    return out


def _choose_hole_cells(
    h: int,
    w: int,
    hole_count: int,
    *,
    randomize: bool,
) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    if randomize:
        order = np.random.permutation(len(cells))
        ordered = [cells[int(i)] for i in order]
    else:
        ordered = cells

    def backtrack(start: int, chosen: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(chosen) == hole_count:
            return chosen.copy()
        need = hole_count - len(chosen)
        for idx in range(start, len(ordered)):
            if len(ordered) - idx < need:
                break
            r, c = ordered[idx]
            if any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in chosen):
                continue
            chosen.append((r, c))
            found = backtrack(idx + 1, chosen)
            if found is not None:
                return found
            chosen.pop()
        return None

    return backtrack(0, [])


def _feasible_dims_by_holes() -> dict[int, list[tuple[int, int]]]:
    dims: dict[int, list[tuple[int, int]]] = {k: [] for k in HOLE_TO_COLOR}
    for hole_count in HOLE_TO_COLOR:
        for h in range(3, 7):
            for w in range(3, 7):
                if _choose_hole_cells(h, w, hole_count, randomize=False) is not None:
                    dims[hole_count].append((h, w))
    return dims


GEN_DIMS_BY_HOLES = _feasible_dims_by_holes()


def _make_object(hole_count: int) -> dict:
    h, w = GEN_DIMS_BY_HOLES[hole_count][
        int(np.random.randint(len(GEN_DIMS_BY_HOLES[hole_count])))
    ]
    hole_cells = _choose_hole_cells(h, w, hole_count, randomize=True)
    assert hole_cells is not None

    patch = np.full((h, w), OBJECT_COLOR, dtype=int)
    for r, c in hole_cells:
        patch[r, c] = BACKGROUND

    body_local = [(r, c) for r in range(h) for c in range(w) if patch[r, c] == OBJECT_COLOR]
    return {
        "hole_count": hole_count,
        "color": HOLE_TO_COLOR[hole_count],
        "patch": patch,
        "h": h,
        "w": w,
        "hole_local": hole_cells,
        "body_local": body_local,
    }


def _sample_starts(lengths: list[int], total: int) -> list[int] | None:
    required = 2 * GEN_EDGE_MARGIN + sum(lengths) + GEN_GAP * (len(lengths) - 1)
    if required > total:
        return None

    extra = total - required
    slots = np.zeros(len(lengths) + 1, dtype=int)
    for _ in range(extra):
        slots[int(np.random.randint(len(slots)))] += 1

    starts: list[int] = []
    pos = GEN_EDGE_MARGIN + int(slots[0])
    starts.append(pos)
    for idx in range(1, len(lengths)):
        pos += lengths[idx - 1] + GEN_GAP + int(slots[idx])
        starts.append(pos)
    return starts


def _direct_ray_path(marker: dict, obj: dict, shape: tuple[int, int]) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    h, w = shape
    side = marker["side"]
    r0, c0, r1, c1 = obj["bbox"]
    mr, mc = marker["coord"]
    if side == "top":
        wall = (r0, mc)
        return [(r, mc) for r in range(1, r0)], wall
    if side == "bottom":
        wall = (r1, mc)
        return [(r, mc) for r in range(h - 2, r1, -1)], wall
    if side == "left":
        wall = (mr, c0)
        return [(mr, c) for c in range(1, c0)], wall
    assert side == "right"
    wall = (mr, c1)
    return [(mr, c) for c in range(w - 2, c1, -1)], wall


def _pick_marker_coord(
    side: str,
    obj: dict,
    used: dict[str, set[int]],
    shape: tuple[int, int],
) -> tuple[int, int] | None:
    h, w = shape
    r0, c0, r1, c1 = obj["bbox"]
    if side in ("top", "bottom"):
        candidates = [c for c in range(c0, c1 + 1) if c not in used[side]]
        if not candidates:
            return None
        c = int(candidates[int(np.random.randint(len(candidates)))])
        return (0, c) if side == "top" else (h - 1, c)

    candidates = [r for r in range(r0, r1 + 1) if r not in used[side]]
    if not candidates:
        return None
    r = int(candidates[int(np.random.randint(len(candidates)))])
    return (r, 0) if side == "left" else (r, w - 1)


def _solve_marker_colors(
    markers: list[dict],
    objects: list[dict],
    shape: tuple[int, int],
) -> list[dict] | None:
    mismatch_indices = [idx for idx, marker in enumerate(markers) if not marker["is_match"]]
    color_choices = [
        [color for color in MARKER_COLORS if color != int(objects[markers[idx]["obj_idx"]]["color"])]
        for idx in mismatch_indices
    ]

    for combo in itertools.product(*color_choices) if color_choices else [()]:
        colored: list[dict] = []
        for idx, marker in enumerate(markers):
            copied = dict(marker)
            if marker["is_match"]:
                copied["color"] = int(objects[marker["obj_idx"]]["color"])
            else:
                mismatch_slot = mismatch_indices.index(idx)
                copied["color"] = int(combo[mismatch_slot])
            colored.append(copied)

        planned: dict[tuple[int, int], int] = {}
        ok = True
        for marker in colored:
            path, _wall = _direct_ray_path(marker, objects[marker["obj_idx"]], shape)
            for cell in path:
                prev = planned.get(cell)
                if prev is not None and prev != int(marker["color"]):
                    ok = False
                    break
                planned[cell] = int(marker["color"])
            if not ok:
                break
        if ok:
            return colored
    return None


def _build_marker_plan(objects: list[dict], shape: tuple[int, int]) -> list[dict] | None:
    a, b, c = map(int, np.random.permutation(len(objects)))
    templates = [
        [("top", a, True), ("bottom", a, False), ("left", b, False), ("right", c, True)],
        [("top", a, False), ("bottom", b, True), ("left", b, False), ("right", c, True)],
        [("top", a, True), ("bottom", b, False), ("left", c, True), ("right", c, False)],
    ]
    assignments = list(templates[int(np.random.randint(len(templates)))])

    matched = {obj_idx for _side, obj_idx, is_match in assignments if is_match}
    unmatched = {idx for idx in range(len(objects)) if idx not in matched}
    side_counts = {side: 0 for side in SIDES}
    for side, _obj_idx, _is_match in assignments:
        side_counts[side] += 1

    extra_budget = 0
    if np.random.random() < 0.65:
        extra_budget += 1
    if np.random.random() < 0.35:
        extra_budget += 1

    for _ in range(extra_budget):
        candidates: list[tuple[str, int, bool]] = []
        for side in SIDES:
            if side_counts[side] >= 2:
                continue
            for obj_idx in range(len(objects)):
                candidates.append((side, obj_idx, False))
                if obj_idx not in unmatched:
                    candidates.append((side, obj_idx, True))
        if not candidates:
            break
        side, obj_idx, is_match = candidates[int(np.random.randint(len(candidates)))]
        assignments.append((side, obj_idx, is_match))
        side_counts[side] += 1

    for _ in range(GEN_COORD_TRIES):
        used = {side: set() for side in SIDES}
        markers: list[dict] = []
        ok = True
        for side, obj_idx, is_match in assignments:
            coord = _pick_marker_coord(side, objects[obj_idx], used, shape)
            if coord is None:
                ok = False
                break
            used[side].add(coord[1] if side in ("top", "bottom") else coord[0])
            markers.append(
                {
                    "side": side,
                    "coord": coord,
                    "obj_idx": obj_idx,
                    "is_match": is_match,
                }
            )
        if not ok:
            continue

        colored = _solve_marker_colors(markers, objects, shape)
        if colored is not None:
            return colored

    return None


def _expected_from_spec(inp: np.ndarray, objects: list[dict], markers: list[dict]) -> np.ndarray:
    out = inp.copy()
    matched_objects: set[int] = set()
    planned: dict[tuple[int, int], int] = {}

    for obj in objects:
        for r, c in obj["body_cells"]:
            out[r, c] = int(obj["color"])

    for marker in markers:
        obj = objects[marker["obj_idx"]]
        path, wall = _direct_ray_path(marker, obj, inp.shape)
        assert path, "every generated marker must have a visible ray segment"
        for r, c in path:
            assert inp[r, c] == BACKGROUND, "ray path must stay on empty background"
            prev = planned.get((r, c))
            assert prev is None or prev == int(marker["color"]), "ray colors must not conflict"
            planned[(r, c)] = int(marker["color"])
        assert inp[wall] == OBJECT_COLOR, "ray must stop on the intended gray object"
        if marker["is_match"]:
            matched_objects.add(int(marker["obj_idx"]))

    for (r, c), color in planned.items():
        out[r, c] = color

    for idx in matched_objects:
        fill_color = int(objects[idx]["color"])
        for r, c in objects[idx]["hole_cells"]:
            out[r, c] = fill_color

    return out


def _place_objects_on_grid(objects: list[dict], shape: tuple[int, int]) -> np.ndarray | None:
    h, w = shape
    grid = np.full((h, w), BACKGROUND, dtype=int)

    row_order = list(map(int, np.random.permutation(len(objects))))
    col_order = list(map(int, np.random.permutation(len(objects))))
    row_starts = _sample_starts([objects[idx]["h"] for idx in row_order], h)
    col_starts = _sample_starts([objects[idx]["w"] for idx in col_order], w)
    if row_starts is None or col_starts is None:
        return None

    for idx, r0 in zip(row_order, row_starts):
        objects[idx]["r0"] = int(r0)
    for idx, c0 in zip(col_order, col_starts):
        objects[idx]["c0"] = int(c0)

    for obj in objects:
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        paste(grid, obj["patch"], r0, c0)
        obj["bbox"] = (r0, c0, r0 + int(obj["h"]) - 1, c0 + int(obj["w"]) - 1)
        obj["body_cells"] = [(r0 + r, c0 + c) for r, c in obj["body_local"]]
        obj["hole_cells"] = [(r0 + r, c0 + c) for r, c in obj["hole_local"]]

    return grid


def _validate_generated_instance(inp: np.ndarray, out: np.ndarray, objects: list[dict], markers: list[dict]) -> None:
    side_counts = {side: 0 for side in SIDES}
    per_object_match = {idx: 0 for idx in range(len(objects))}
    per_object_mismatch = {idx: 0 for idx in range(len(objects))}

    for marker in markers:
        side_counts[marker["side"]] += 1
        if marker["is_match"]:
            per_object_match[int(marker["obj_idx"])] += 1
        else:
            per_object_mismatch[int(marker["obj_idx"])] += 1

    assert all(1 <= side_counts[side] <= 2 for side in SIDES), "each side must have 1-2 markers"
    assert len(markers) <= 6, "marker budget exceeded"
    assert sorted(obj["hole_count"] for obj in objects) == [1, 2, 3], "generator must cover all three hole counts"
    assert any(per_object_match[idx] > 0 for idx in per_object_match), "need at least one matched object"
    assert any(per_object_mismatch[idx] > 0 and per_object_match[idx] == 0 for idx in per_object_match), (
        "need at least one mismatch-only object"
    )
    assert any(
        per_object_match[idx] > 0 and per_object_mismatch[idx] > 0 for idx in per_object_match
    ), "need one object hit by both matching and mismatching rays"
    assert not np.array_equal(inp, out), "output must differ from input"


def _gen_attempt() -> np.ndarray | None:
    objects = [_make_object(hole_count) for hole_count in (1, 2, 3)]
    min_h = 2 * GEN_EDGE_MARGIN + sum(obj["h"] for obj in objects) + GEN_GAP * (len(objects) - 1)
    min_w = 2 * GEN_EDGE_MARGIN + sum(obj["w"] for obj in objects) + GEN_GAP * (len(objects) - 1)
    if min_h > GEN_MAX_SIDE or min_w > GEN_MAX_SIDE:
        return None

    h = int(np.random.randint(max(GEN_MIN_SIDE, min_h), GEN_MAX_SIDE + 1))
    w = int(np.random.randint(max(GEN_MIN_SIDE, min_w), GEN_MAX_SIDE + 1))
    inp = _place_objects_on_grid(objects, (h, w))
    if inp is None:
        return None

    markers = _build_marker_plan(objects, (h, w))
    if markers is None:
        return None

    for marker in markers:
        r, c = marker["coord"]
        inp[r, c] = int(marker["color"])

    expected = _expected_from_spec(inp, objects, markers)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    _validate_generated_instance(inp, expected, objects, markers)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
