from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 5)
OBJECT_COUNT_RANGE = (3, 5)
GRID_HEIGHT_RANGE = (12, 20)
GRID_WIDTH_RANGE = (18, 28)
BBOX_GAP = 2
MAX_GENERATION_ATTEMPTS = 600


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _c in coords]
    cs = [c for _r, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return objects


def _rotate_cell_90(cell: tuple[int, int], center2: tuple[int, int]) -> tuple[int, int]:
    r, c = cell
    cr2, cc2 = center2
    pr2 = 2 * r
    pc2 = 2 * c
    nr2 = cr2 + cc2 - pc2
    nc2 = cc2 + pr2 - cr2
    assert nr2 % 2 == 0 and nc2 % 2 == 0
    return nr2 // 2, nc2 // 2


def _rotate_coords(coords: set[tuple[int, int]], center2: tuple[int, int], turns: int) -> set[tuple[int, int]]:
    out = set(coords)
    for _ in range(turns % 4):
        out = {_rotate_cell_90(cell, center2) for cell in out}
    return out


def _complete_coords(coords: list[tuple[int, int]] | set[tuple[int, int]], center2: tuple[int, int]) -> set[tuple[int, int]]:
    visible = set(coords)
    completed = set(visible)
    for cell in list(visible):
        cur = cell
        for _ in range(3):
            cur = _rotate_cell_90(cur, center2)
            completed.add(cur)
    return completed


def _infer_center2(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> dict[str, object]:
    coords_list = list(coords)
    rs = np.array([r for r, _c in coords_list], dtype=float)
    cs = np.array([c for _r, c in coords_list], dtype=float)
    mean2_r = 2.0 * float(rs.mean())
    mean2_c = 2.0 * float(cs.mean())
    r0, c0, r1, c1 = _bbox(coords_list)

    candidates: list[tuple[float, int, int, int]] = []
    for cr2 in range(2 * r0 - 4, 2 * r1 + 5):
        for cc2 in range(2 * c0 - 4, 2 * c1 + 5):
            if (cr2 - cc2) & 1:
                continue
            dist2 = (cr2 - mean2_r) ** 2 + (cc2 - mean2_c) ** 2
            completed_area = len(_complete_coords(coords_list, (cr2, cc2)))
            candidates.append((dist2, completed_area, cr2, cc2))
    candidates.sort()
    best = candidates[0]
    unique_distance = len(candidates) == 1 or best[0] + 1e-9 < candidates[1][0]
    return {
        "center2": (best[2], best[3]),
        "distance2": best[0],
        "completed_area": best[1],
        "unique_distance": unique_distance,
    }


def _draw_nested_rings(colors: list[int]) -> np.ndarray:
    side = 2 * len(colors) - 1
    out = np.full((side, side), int(colors[0]), dtype=int)
    for layer, color in enumerate(colors[1:], start=1):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each rotationally symmetric fragment and redraw the objects as nested square rings ordered by completed area.

    Input structure:
    - The grid has a black background and 3-5 small monochromatic 4-connected objects in distinct colors.
    - Each object is a partial view of a 90-degree rotationally symmetric shape, with one or more rotated sectors missing.
    - Objects are separated from each other by background and do not touch.

    Rule:
    - Find every monochromatic 4-connected object.
    - For each object, compute the geometric center of its visible pixels and choose the nearest grid point or half-grid intersection that supports 90-degree integer-cell rotation.
    - Rotate the visible pixels around that center by 90, 180, and 270 degrees, filling any black target cells to complete the symmetric object.
    - Count the completed area of every object, sort the objects from largest completed area to smallest, and output a (2k-1)x(2k-1) square where layer i is filled with the color of the i-th largest object.

    Key insight:
    - Visible size is intentionally misleading; the correct ring order is determined only after rotational completion reconstructs each object's full area.
    """
    g = np.asarray(grid, dtype=int)
    objects = _find_components(g, bg=BG)
    assert 1 <= len(objects) <= 5

    analyzed = []
    for obj in objects:
        inferred = _infer_center2(obj["coords"])
        completed = _complete_coords(obj["coords"], inferred["center2"])
        analyzed.append(
            {
                "color": int(obj["color"]),
                "completed_area": len(completed),
                "bbox": obj["bbox"],
            }
        )

    analyzed.sort(key=lambda item: (-int(item["completed_area"]), item["bbox"]))
    return _draw_nested_rings([int(item["color"]) for item in analyzed])


def _connected(coords: set[tuple[int, int]]) -> bool:
    start = next(iter(coords))
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in coords and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(coords)


def _build_variant_catalog() -> list[dict[str, object]]:
    families = [
        (
            "integer",
            (0, 0),
            {(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)},
            [
                {(-1, 1)},
                {(-1, 1), (-2, 1)},
                {(-1, 1), (-2, 1), (-1, 2)},
            ],
        ),
        (
            "half",
            (1, 1),
            {(0, 0), (0, 1), (1, 0), (1, 1)},
            [
                {(-1, 1)},
                {(-1, 1), (-1, 2)},
                {(-1, 1), (-1, 2), (-2, 1)},
                {(-1, 1), (-1, 2), (-2, 1), (-2, 2)},
            ],
        ),
    ]

    variants: list[dict[str, object]] = []
    for family, center2, core, seeds in families:
        for seed in seeds:
            full = set(core)
            for turns in range(4):
                full |= _rotate_coords(seed, center2, turns)
            full_r0, full_c0, full_r1, full_c1 = _bbox(full)
            shape = (full_r1 - full_r0 + 1, full_c1 - full_c0 + 1)
            rel_center2 = (center2[0] - 2 * full_r0, center2[1] - 2 * full_c0)
            for kept_count in (1, 2, 3):
                for kept_turns in itertools.combinations(range(4), kept_count):
                    visible = set(core)
                    for turns in kept_turns:
                        visible |= _rotate_coords(seed, center2, turns)
                    if not (3 <= len(visible) <= 12):
                        continue
                    if not _connected(visible):
                        continue
                    inferred = _infer_center2(visible)
                    if tuple(inferred["center2"]) != center2:
                        continue
                    if not bool(inferred["unique_distance"]):
                        continue
                    variants.append(
                        {
                            "family": family,
                            "shape": shape,
                            "visible": frozenset((r - full_r0, c - full_c0) for r, c in visible),
                            "center2": rel_center2,
                            "visible_area": len(visible),
                            "full_area": len(full),
                            "kept_count": kept_count,
                        }
                    )
    return variants


VARIANT_CATALOG = _build_variant_catalog()


def _has_area_inversion(variants: list[dict[str, object]]) -> bool:
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            full_i = int(variants[i]["full_area"])
            full_j = int(variants[j]["full_area"])
            vis_i = int(variants[i]["visible_area"])
            vis_j = int(variants[j]["visible_area"])
            if (full_i - full_j) * (vis_i - vis_j) < 0:
                return True
    return False


def _pick_variants(k: int) -> list[dict[str, object]] | None:
    require_mixed_families = k >= 4 and random.random() < 0.6
    require_mixed_kept = random.random() < 0.75
    for _ in range(500):
        chosen = random.sample(VARIANT_CATALOG, k)
        full_areas = [int(item["full_area"]) for item in chosen]
        visible_areas = [int(item["visible_area"]) for item in chosen]
        if len(set(full_areas)) != k:
            continue
        if len(set(visible_areas)) != k:
            continue
        if not _has_area_inversion(chosen):
            continue
        if require_mixed_families and len({str(item["family"]) for item in chosen}) < 2:
            continue
        if require_mixed_kept and len({int(item["kept_count"]) for item in chosen}) < 2:
            continue
        return chosen
    return None


def _rect_far(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = BBOX_GAP) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _place_variants(variants: list[dict[str, object]]) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    ordered = sorted(
        variants,
        key=lambda item: (item["shape"][0] * item["shape"][1], item["shape"]),
        reverse=True,
    )
    for _ in range(200):
        h = random.randint(*GRID_HEIGHT_RANGE)
        w = random.randint(*GRID_WIDTH_RANGE)
        grid = np.full((h, w), BG, dtype=int)
        colors = random.sample(PALETTE, len(ordered))
        used_rects: list[tuple[int, int, int, int]] = []
        placed: list[dict[str, object]] = []
        ok = True
        for variant, color in zip(ordered, colors):
            box_h, box_w = variant["shape"]
            options: list[tuple[int, int]] = []
            for r0 in range(h - box_h + 1):
                for c0 in range(w - box_w + 1):
                    rect = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
                    if all(_rect_far(rect, old) for old in used_rects):
                        options.append((r0, c0))
            if not options:
                ok = False
                break
            r0, c0 = random.choice(options)
            for rr, cc in variant["visible"]:
                grid[r0 + rr, c0 + cc] = int(color)
            used_rects.append((r0, c0, r0 + box_h - 1, c0 + box_w - 1))
            placed.append(
                {
                    "color": int(color),
                    "visible_area": int(variant["visible_area"]),
                    "full_area": int(variant["full_area"]),
                    "bbox": used_rects[-1],
                    "center2": (int(variant["center2"][0]) + 2 * r0, int(variant["center2"][1]) + 2 * c0),
                    "family": str(variant["family"]),
                    "kept_count": int(variant["kept_count"]),
                }
            )
        if ok:
            return grid, placed
    return None


def _expected_output_from_metadata(placed: list[dict[str, object]]) -> np.ndarray:
    ordered = sorted(placed, key=lambda item: (-int(item["full_area"]), tuple(item["bbox"])))
    return _draw_nested_rings([int(item["color"]) for item in ordered])


def _semantic_check(grid: np.ndarray, placed: list[dict[str, object]]) -> bool:
    components = _find_components(grid, bg=BG)
    if len(components) != len(placed):
        return False

    meta_by_color = {int(item["color"]): item for item in placed}
    visible_areas = [int(item["visible_area"]) for item in placed]
    full_areas = [int(item["full_area"]) for item in placed]
    if len(set(visible_areas)) != len(visible_areas):
        return False
    if len(set(full_areas)) != len(full_areas):
        return False
    if not _has_area_inversion(placed):
        return False

    for comp in components:
        color = int(comp["color"])
        if color not in meta_by_color:
            return False
        meta = meta_by_color[color]
        if int(comp["area"]) != int(meta["visible_area"]):
            return False
        inferred = _infer_center2(comp["coords"])
        if tuple(inferred["center2"]) != tuple(meta["center2"]):
            return False
        if not bool(inferred["unique_distance"]):
            return False
        completed = _complete_coords(comp["coords"], inferred["center2"])
        if len(completed) != int(meta["full_area"]):
            return False

    expected = _expected_output_from_metadata(placed)
    actual = main(grid)
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        k = random.randint(*OBJECT_COUNT_RANGE)
        variants = _pick_variants(k)
        if variants is None:
            continue
        placed = _place_variants(variants)
        if placed is None:
            continue
        grid, metadata = placed
        if _semantic_check(grid, metadata):
            return grid
    raise RuntimeError("failed to generate a valid rotational-completion nesting input")
