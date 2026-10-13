from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, in_bounds, label_mask, neighbors4, neighbors8


PAIR_COLORS = [Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.MAROON]
HOLE_AREA_POOL = [1, 2, 3, 4, 6]
DISTANCE_POOL = [6, 10, 14, 18]
LANES = ("top", "bottom", "left", "right")
GEN_RETRIES = 400
OBJECT_RETRIES = 250
PLACEMENT_RETRIES = 400
MIN_OBJECT_GAP = 3


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _edge_component_ids(labels: np.ndarray, n_labels: int) -> set[int]:
    edge_ids: set[int] = set()
    if n_labels == 0:
        return edge_ids
    edge_ids.update(map(int, labels[0, :]))
    edge_ids.update(map(int, labels[-1, :]))
    edge_ids.update(map(int, labels[:, 0]))
    edge_ids.update(map(int, labels[:, -1]))
    edge_ids.discard(0)
    return edge_ids


def _mask_holes(mask: np.ndarray) -> list[np.ndarray]:
    labels, n_labels = label_mask(~mask, connectivity=4)
    edge_ids = _edge_component_ids(labels, n_labels)
    holes = []
    for comp_id in range(1, n_labels + 1):
        if comp_id in edge_ids:
            continue
        holes.append(np.argwhere(labels == comp_id))
    return holes


def _analyze_blue_objects(grid: np.ndarray) -> list[dict]:
    blue_labels, n_blue = label_mask(grid == Color.BLUE, connectivity=4)
    bg_labels, n_bg = label_mask(grid == Color.BLACK, connectivity=4)
    edge_bg_ids = _edge_component_ids(bg_labels, n_bg)
    holes_by_object: dict[int, list[np.ndarray]] = defaultdict(list)

    for bg_id in range(1, n_bg + 1):
        if bg_id in edge_bg_ids:
            continue
        coords = np.argwhere(bg_labels == bg_id)
        neighbor_objects: set[int] = set()
        for r, c in coords:
            for nr, nc in neighbors4(int(r), int(c)):
                if in_bounds(grid.shape, nr, nc) and blue_labels[nr, nc] != 0:
                    neighbor_objects.add(int(blue_labels[nr, nc]))
        if len(neighbor_objects) == 1:
            holes_by_object[next(iter(neighbor_objects))].append(coords)

    objects: list[dict] = []
    for obj_id in range(1, n_blue + 1):
        coords = np.argwhere(blue_labels == obj_id)
        hole_list = holes_by_object.get(obj_id, [])
        assert len(hole_list) <= 1, "valid inputs must have at most one hole per blue object"
        hole_coords = hole_list[0] if hole_list else np.empty((0, 2), dtype=int)
        objects.append(
            {
                "id": obj_id,
                "coords": coords,
                "bbox": _bbox(coords),
                "hole_coords": hole_coords,
                "hole_area": int(hole_coords.shape[0]),
            }
        )
    return objects


def _find_isolated_pairs(grid: np.ndarray) -> list[dict]:
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for comp in find_components(
        grid,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    ):
        color = int(comp["color"])
        if color in (Color.BLACK, Color.BLUE):
            continue
        assert int(comp["area"]) == 1, "all non-blue foreground markers must be isolated pixels"
        r, c = map(int, comp["coords"][0])
        groups[color].append((r, c))

    pairs = []
    for color, coords in groups.items():
        assert len(coords) == 2, "each marker color must appear exactly twice"
        p0, p1 = sorted(coords)
        if p0[0] == p1[0]:
            pair = {
                "color": color,
                "axis": "h",
                "p0": p0 if p0[1] <= p1[1] else p1,
                "p1": p1 if p0[1] <= p1[1] else p0,
            }
        else:
            assert p0[1] == p1[1], "pair pixels must be aligned horizontally or vertically"
            pair = {
                "color": color,
                "axis": "v",
                "p0": p0 if p0[0] <= p1[0] else p1,
                "p1": p1 if p0[0] <= p1[0] else p0,
            }
        pair["distance"] = int(abs(pair["p1"][0] - pair["p0"][0]) + abs(pair["p1"][1] - pair["p0"][1]))
        pairs.append(pair)
    return pairs


def _outline_mask(shape: tuple[int, int], obj_coords: np.ndarray, bg_mask: np.ndarray) -> np.ndarray:
    outline = np.zeros(shape, dtype=bool)
    for r, c in obj_coords:
        for nr, nc in neighbors8(int(r), int(c)):
            if in_bounds(shape, nr, nc) and bg_mask[nr, nc]:
                outline[nr, nc] = True
    return outline


def _paint_hollow_rect(out: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    out[r0, c0 : c0 + w] = color
    out[r0 + h - 1, c0 : c0 + w] = color
    out[r0 : r0 + h, c0] = color
    out[r0 : r0 + h, c0 + w - 1] = color


def _draw_bridge(out: np.ndarray, pair: dict, color: int) -> None:
    if pair["axis"] == "h":
        r = int(pair["p0"][0])
        c0 = int(pair["p0"][1])
        c1 = int(pair["p1"][1])
        distance = int(pair["distance"])
        rect_w = distance // 2
        assert distance % 2 == 0 and rect_w >= 3 and rect_w % 2 == 1
        center_c = (c0 + c1) // 2
        left = center_c - rect_w // 2
        _paint_hollow_rect(out, r - 1, left, 3, rect_w, color)
        out[r, c0 + 1 : left] = color
        out[r, left + rect_w : c1] = color
    else:
        c = int(pair["p0"][1])
        r0 = int(pair["p0"][0])
        r1 = int(pair["p1"][0])
        distance = int(pair["distance"])
        rect_h = distance // 2
        assert distance % 2 == 0 and rect_h >= 3 and rect_h % 2 == 1
        center_r = (r0 + r1) // 2
        top = center_r - rect_h // 2
        _paint_hollow_rect(out, top, c - 1, rect_h, 3, color)
        out[r0 + 1 : top, c] = color
        out[top + rect_h : r1, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair hole-area ranks with isolated-pixel-distance ranks to recolor holes and draw same-color bridges.

    Input structure:
    - The background is black, with several blue 4-connected objects and several isolated non-blue marker pixels.
    - Some blue objects enclose exactly one black hole; the remaining blue objects are solid distractors.
    - Each marker color appears in exactly one aligned two-pixel pair, and pair distances are all distinct.

    Rule:
    - Rank the hole-enclosing blue objects by enclosed hole area, smallest first.
    - Rank the aligned isolated-pixel pairs by Manhattan distance, shortest first.
    - Pair equal ranks: recolor that blue object teal, fill its hole with the pair color, and bridge the two marker pixels with a same-color hollow rectangle plus connecting lines.
    - For every blue object without a hole, keep the object blue and add a red 8-neighborhood exterior outline on background cells.

    Key insight:
    - The decisive correspondence is not local adjacency but a synchronized ranking across two unrelated domains: hole area and pair distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    objects = _analyze_blue_objects(g)
    pairs = _find_isolated_pairs(g)

    hole_objects = [obj for obj in objects if obj["hole_area"] > 0]
    solid_objects = [obj for obj in objects if obj["hole_area"] == 0]
    assert len(hole_objects) == len(pairs) >= 1

    ranked_objects = sorted(hole_objects, key=lambda obj: (obj["hole_area"], obj["bbox"]))
    ranked_pairs = sorted(pairs, key=lambda pair: (pair["distance"], pair["color"]))

    out = g.copy()
    for obj, pair in zip(ranked_objects, ranked_pairs):
        out[obj["coords"][:, 0], obj["coords"][:, 1]] = Color.TEAL
        out[obj["hole_coords"][:, 0], obj["hole_coords"][:, 1]] = int(pair["color"])
        _draw_bridge(out, pair, int(pair["color"]))

    bg_mask = g == Color.BLACK
    for obj in solid_objects:
        outline = _outline_mask(g.shape, obj["coords"], bg_mask)
        out[outline] = Color.RED

    return out


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    out = []
    for h in range(1, 5):
        if area % h == 0:
            w = area // h
            if w <= 4:
                out.append((h, w))
    return out


def _make_hole_patch(hole_area: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(OBJECT_RETRIES):
        hole_h, hole_w = random.choice(_factor_pairs(hole_area))
        top = random.randint(1, 2)
        bottom = random.randint(1, 2)
        left = random.randint(1, 2)
        right = random.randint(1, 2)
        outer_h = hole_h + top + bottom
        outer_w = hole_w + left + right

        mask = np.zeros((outer_h + 2, outer_w + 2), dtype=bool)
        mask[1 : 1 + outer_h, 1 : 1 + outer_w] = True
        hole_r0 = 1 + top
        hole_c0 = 1 + left
        mask[hole_r0 : hole_r0 + hole_h, hole_c0 : hole_c0 + hole_w] = False

        for _ in range(random.randint(0, 2)):
            side = random.choice(("top", "bottom", "left", "right"))
            span = random.randint(1, 2)
            if side in ("top", "bottom"):
                c0 = random.randint(1, outer_w - span + 1)
                rr = 0 if side == "top" else outer_h + 1
                mask[rr, c0 : c0 + span] = True
            else:
                r0 = random.randint(1, outer_h - span + 1)
                cc = 0 if side == "left" else outer_w + 1
                mask[r0 : r0 + span, cc] = True

        mask = _crop_mask(mask)
        labels, n_labels = label_mask(mask, connectivity=4)
        if n_labels != 1:
            continue
        holes = _mask_holes(mask)
        if len(holes) != 1 or int(holes[0].shape[0]) != hole_area:
            continue
        area = int(mask.sum())
        if not (8 <= area <= 20):
            continue
        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = Color.BLUE
        return patch, holes[0]
    raise RuntimeError(f"failed to build a hole object for area {hole_area}")


def _make_solid_patch() -> np.ndarray:
    for _ in range(OBJECT_RETRIES):
        target = random.randint(4, 9)
        canvas = np.zeros((7, 7), dtype=bool)
        start = (random.randint(2, 4), random.randint(2, 4))
        cells = {start}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < 7 and 0 <= nc < 7 and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            cells.add(random.choice(tuple(frontier)))
        for r, c in cells:
            canvas[r, c] = True
        mask = _crop_mask(canvas)
        labels, n_labels = label_mask(mask, connectivity=4)
        if n_labels != 1 or _mask_holes(mask):
            continue
        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = Color.BLUE
        return patch
    raise RuntimeError("failed to build a solid distractor")


def _coords_chebyshev_gap(a: np.ndarray, b: np.ndarray) -> int:
    best = 10**9
    for r0, c0 in a:
        for r1, c1 in b:
            best = min(best, max(abs(int(r0) - int(r1)), abs(int(c0) - int(c1))))
    return best


def _place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    occupied_coords: list[np.ndarray],
    row_range: tuple[int, int],
    col_range: tuple[int, int],
) -> tuple[int, int] | None:
    ph, pw = patch.shape
    min_r, max_r = row_range
    min_c, max_c = col_range
    if max_r - min_r + 1 < ph or max_c - min_c + 1 < pw:
        return None

    for _ in range(PLACEMENT_RETRIES):
        r0 = random.randint(min_r, max_r - ph + 1)
        c0 = random.randint(min_c, max_c - pw + 1)
        coords = np.argwhere(patch == Color.BLUE) + np.array([r0, c0])
        if any(_coords_chebyshev_gap(coords, other) < MIN_OBJECT_GAP for other in occupied_coords):
            continue
        grid[coords[:, 0], coords[:, 1]] = Color.BLUE
        occupied_coords.append(coords)
        return r0, c0
    return None


def _place_pair(grid: np.ndarray, lane: str, color: int, distance: int) -> dict | None:
    h, w = grid.shape
    if lane == "top":
        row = 2
        starts = list(range(1, w - distance - 1))
        if not starts:
            return None
        c0 = random.choice(starts)
        c1 = c0 + distance
        grid[row, c0] = color
        grid[row, c1] = color
        return {"lane": lane, "color": color, "axis": "h", "p0": (row, c0), "p1": (row, c1), "distance": distance}
    if lane == "bottom":
        row = h - 3
        starts = list(range(1, w - distance - 1))
        if not starts:
            return None
        c0 = random.choice(starts)
        c1 = c0 + distance
        grid[row, c0] = color
        grid[row, c1] = color
        return {"lane": lane, "color": color, "axis": "h", "p0": (row, c0), "p1": (row, c1), "distance": distance}
    if lane == "left":
        col = 2
        starts = list(range(4, h - distance - 4))
        if not starts:
            return None
        r0 = random.choice(starts)
        r1 = r0 + distance
        grid[r0, col] = color
        grid[r1, col] = color
        return {"lane": lane, "color": color, "axis": "v", "p0": (r0, col), "p1": (r1, col), "distance": distance}
    col = w - 3
    starts = list(range(4, h - distance - 4))
    if not starts:
        return None
    r0 = random.choice(starts)
    r1 = r0 + distance
    grid[r0, col] = color
    grid[r1, col] = color
    return {"lane": lane, "color": color, "axis": "v", "p0": (r0, col), "p1": (r1, col), "distance": distance}


def _build_expected(grid: np.ndarray, meta: dict) -> np.ndarray:
    out = grid.copy()
    ranked_objects = sorted(meta["hole_objects"], key=lambda obj: (obj["hole_area"], obj["bbox"]))
    ranked_pairs = sorted(meta["pairs"], key=lambda pair: (pair["distance"], pair["color"]))

    for obj, pair in zip(ranked_objects, ranked_pairs):
        out[obj["coords"][:, 0], obj["coords"][:, 1]] = Color.TEAL
        out[obj["hole_coords"][:, 0], obj["hole_coords"][:, 1]] = int(pair["color"])
        _draw_bridge(out, pair, int(pair["color"]))

    bg_mask = grid == Color.BLACK
    for obj in meta["solid_objects"]:
        outline = _outline_mask(grid.shape, obj["coords"], bg_mask)
        out[outline] = Color.RED

    return out


def _bridge_mask(shape: tuple[int, int], pair: dict) -> np.ndarray:
    tmp = np.zeros(shape, dtype=int)
    _draw_bridge(tmp, pair, 1)
    return tmp == 1


def _semantic_audit(grid: np.ndarray, expected: np.ndarray) -> None:
    analyzed = _analyze_blue_objects(grid)
    pairs = _find_isolated_pairs(grid)
    hole_objects = sorted((obj for obj in analyzed if obj["hole_area"] > 0), key=lambda obj: (obj["hole_area"], obj["bbox"]))
    solid_objects = [obj for obj in analyzed if obj["hole_area"] == 0]
    ranked_pairs = sorted(pairs, key=lambda pair: (pair["distance"], pair["color"]))

    assert len(hole_objects) == len(ranked_pairs) >= 2
    assert len({obj["hole_area"] for obj in hole_objects}) == len(hole_objects)
    assert len({pair["distance"] for pair in ranked_pairs}) == len(ranked_pairs)

    for obj, pair in zip(hole_objects, ranked_pairs):
        assert np.all(expected[obj["coords"][:, 0], obj["coords"][:, 1]] == Color.TEAL)
        assert np.all(expected[obj["hole_coords"][:, 0], obj["hole_coords"][:, 1]] == int(pair["color"]))
        bridge = _bridge_mask(grid.shape, pair)
        assert np.all(expected[bridge] == int(pair["color"]))

    bg_mask = grid == Color.BLACK
    for obj in solid_objects:
        outline = _outline_mask(grid.shape, obj["coords"], bg_mask)
        assert np.all(expected[obj["coords"][:, 0], obj["coords"][:, 1]] == Color.BLUE)
        assert np.all(expected[outline] == Color.RED)


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    h = random.randint(24, 28)
    w = random.randint(24, 28)
    grid = np.zeros((h, w), dtype=int)

    pair_count = random.randint(2, 4)
    hole_areas = random.sample(HOLE_AREA_POOL, pair_count)
    distances = random.sample(DISTANCE_POOL, pair_count)
    lanes = random.sample(LANES, pair_count)
    pair_colors = random.sample(PAIR_COLORS, pair_count)
    solid_count = random.choices([0, 1, 2], weights=[1, 3, 2], k=1)[0]

    hole_objects = []
    solid_objects = []
    occupied_coords: list[np.ndarray] = []
    row_range = (6, h - 7)
    col_range = (6, w - 7)

    hole_specs = []
    for hole_area in hole_areas:
        patch, hole_local = _make_hole_patch(hole_area)
        hole_specs.append((patch, hole_local, hole_area))
    random.shuffle(hole_specs)
    for patch, hole_local, hole_area in sorted(hole_specs, key=lambda item: -int(np.sum(item[0] == Color.BLUE))):
        placed = _place_patch(grid, patch, occupied_coords, row_range, col_range)
        if placed is None:
            raise RuntimeError("failed to place a hole object")
        r0, c0 = placed
        coords = np.argwhere(patch == Color.BLUE) + np.array([r0, c0])
        hole_coords = hole_local + np.array([r0, c0])
        hole_objects.append(
            {
                "coords": coords,
                "bbox": _bbox(coords),
                "hole_coords": hole_coords,
                "hole_area": hole_area,
            }
        )

    solid_specs = [_make_solid_patch() for _ in range(solid_count)]
    random.shuffle(solid_specs)
    for patch in sorted(solid_specs, key=lambda item: -int(np.sum(item == Color.BLUE))):
        placed = _place_patch(grid, patch, occupied_coords, row_range, col_range)
        if placed is None:
            raise RuntimeError("failed to place a solid object")
        r0, c0 = placed
        coords = np.argwhere(patch == Color.BLUE) + np.array([r0, c0])
        solid_objects.append({"coords": coords, "bbox": _bbox(coords)})

    pairs = []
    for lane, color, distance in zip(lanes, pair_colors, distances):
        pair = _place_pair(grid, lane, color, distance)
        if pair is None:
            raise RuntimeError("failed to place a marker pair")
        pairs.append(pair)

    meta = {"hole_objects": hole_objects, "solid_objects": solid_objects, "pairs": pairs}
    expected = _build_expected(grid, meta)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid, expected = _generate_case()
        except RuntimeError:
            continue
        if not np.array_equal(main(grid), expected):
            continue
        try:
            _semantic_audit(grid, expected)
        except AssertionError:
            continue
        return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
