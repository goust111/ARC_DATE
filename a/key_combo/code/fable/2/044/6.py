from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components, get_bg, paste, random_free_location


def _bbox_chebyshev_distance(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> int:
    r, c = point
    r0, c0, r1, c1 = bbox
    dr = r0 - r if r < r0 else r - r1 if r > r1 else 0
    dc = c0 - c if c < c0 else c - c1 if c > c1 else 0
    return int(max(dr, dc))


def _coords_chebyshev_distance(point: tuple[int, int], coords: np.ndarray) -> int:
    r, c = point
    return min(max(abs(r - int(rr)), abs(c - int(cc))) for rr, cc in coords)


def _paint_bbox_ring(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    distance: int,
    color: int,
    bg: int,
) -> None:
    if distance <= 0:
        return
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    rr0, cc0 = r0 - distance, c0 - distance
    rr1, cc1 = r1 + distance, c1 + distance

    def paint_cell(r: int, c: int) -> None:
        if 0 <= r < h and 0 <= c < w and int(grid[r, c]) == bg:
            grid[r, c] = color

    for c in range(cc0, cc1 + 1):
        paint_cell(rr0, c)
        paint_cell(rr1, c)
    for r in range(rr0 + 1, rr1):
        paint_cell(r, cc0)
        paint_cell(r, cc1)


def _parse_grid(grid: np.ndarray) -> tuple[int, dict[int, dict], dict[int, list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")

    objects_by_color: dict[int, list[dict]] = {}
    points_by_color: dict[int, list[tuple[int, int]]] = {}
    for comp in comps:
        color = int(comp["color"])
        if int(comp["area"]) == 1:
            rr, cc = comp["coords"][0]
            points_by_color.setdefault(color, []).append((int(rr), int(cc)))
        else:
            objects_by_color.setdefault(color, []).append(comp)

    owned_objects: dict[int, dict] = {}
    for color, points in points_by_color.items():
        owners = objects_by_color.get(color, [])
        assert len(owners) == 1, f"expected exactly one same-color object for points of color {color}"
        owned_objects[color] = owners[0]
        assert points, "point list should not be empty"
    return bg, owned_objects, points_by_color


def _solve(grid: np.ndarray, *, use_bbox_distance: bool, ring_mode: str) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, objects_by_color, points_by_color = _parse_grid(g)
    out = g.copy()

    for points in points_by_color.values():
        for r, c in points:
            out[r, c] = bg

    for color, obj in objects_by_color.items():
        points = points_by_color[color]
        bbox = tuple(map(int, obj["bbox"]))
        coords = obj["coords"]
        if use_bbox_distance:
            distances = [_bbox_chebyshev_distance(point, bbox) for point in points]
        else:
            distances = [_coords_chebyshev_distance(point, coords) for point in points]

        if ring_mode == "unique":
            targets = sorted(set(distances))
        elif ring_mode == "max_only":
            targets = [max(distances)]
        elif ring_mode == "count_layers":
            targets = list(range(1, len(points) + 1))
        else:
            raise ValueError(f"unknown ring mode: {ring_mode}")

        for distance in targets:
            _paint_bbox_ring(out, bbox, int(distance), int(color), bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn each isolated same-color point into a full ripple ring around its matching object.

    Input structure:
    - A dominant background contains several separated monochrome multi-cell objects and some
      isolated single-pixel points.
    - Every point shares its color with exactly one multi-cell object and is not adjacent to
      any object pixel.
    - Objects may be irregular inside their own tight bounding boxes.

    Rule:
    - Split non-background cells by monochrome 8-connectivity into multi-cell objects and
      single-pixel points.
    - Erase all isolated points, then for each point compute its Chebyshev distance j to the
      tight bounding box of the same-color object.
    - For every distinct distance j selected for an object, expand that object's bbox outward
      by j and paint only the outermost rectangular ring cells that are still background.

    Key insight:
    - The point chooses a ring by distance to the object's bbox, not by its exact direction,
      count, or distance to the object's filled pixels.
    """
    return _solve(grid, use_bbox_distance=True, ring_mode="unique")


_PALETTE = tuple(range(10))
_MAX_DISTANCE = 3
_GEN_RETRIES = 400
_SPEC_RETRIES = 200
_RING_DISTANCE_CHOICES = (1, 2, 3)


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, dtype=bool, copy=True)
    out = np.rot90(out, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _make_rect_mask() -> np.ndarray:
    h = random.randint(2, 4)
    w = random.randint(2, 5)
    return np.ones((h, w), dtype=bool)


def _make_l_mask() -> np.ndarray:
    h = random.randint(3, 5)
    w = random.randint(3, 5)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[:, 0] = True
    return mask


def _make_c_mask() -> np.ndarray:
    h = random.randint(3, 5)
    w = random.randint(3, 5)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    return mask


def _make_t_mask() -> np.ndarray:
    h = random.randint(3, 5)
    w = random.choice((3, 5))
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[:, w // 2] = True
    return mask


def _make_stair_mask() -> np.ndarray:
    h = random.randint(3, 5)
    w = random.randint(3, 5)
    mask = np.zeros((h, w), dtype=bool)
    span = min(h, w)
    for idx in range(span):
        mask[idx, idx] = True
        if idx + 1 < w:
            mask[idx, idx + 1] = True
    return mask


def _random_mask(kind: str) -> np.ndarray:
    if kind == "critical":
        builder = _make_l_mask
    elif kind == "duplicate":
        builder = random.choice((_make_rect_mask, _make_c_mask, _make_t_mask, _make_stair_mask))
    else:
        builder = random.choice((_make_rect_mask, _make_l_mask, _make_c_mask, _make_t_mask, _make_stair_mask))
    return _transform_mask(builder())


def _is_far_from_coords(point: tuple[int, int], used: list[tuple[int, int]], min_cheb: int = 2) -> bool:
    return all(max(abs(point[0] - r), abs(point[1] - c)) >= min_cheb for r, c in used)


def _ring_candidates(mask: np.ndarray, distance: int, *, require_bbox_only: bool = False) -> list[tuple[int, int]]:
    h, w = mask.shape
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    candidates: list[tuple[int, int]] = []
    for r in range(-distance, h + distance):
        for c in range(-distance, w + distance):
            if r not in (-distance, h - 1 + distance) and c not in (-distance, w - 1 + distance):
                continue
            obj_dist = min(max(abs(r - rr), abs(c - cc)) for rr, cc in coords)
            if obj_dist <= 1:
                continue
            if require_bbox_only and obj_dist <= distance:
                continue
            candidates.append((r, c))
    return candidates


def _pick_points(
    candidates: list[tuple[int, int]],
    count: int,
    used: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    used = [] if used is None else list(used)
    pool = list(candidates)
    random.shuffle(pool)
    chosen: list[tuple[int, int]] = []
    for point in pool:
        if _is_far_from_coords(point, used + chosen):
            chosen.append(point)
            if len(chosen) == count:
                return chosen
    return None


def _make_point_record(mask: np.ndarray, rel_point: tuple[int, int], distance: int) -> dict:
    coords = np.argwhere(mask)
    return {
        "rel": rel_point,
        "j": int(distance),
        "obj_dist": int(_coords_chebyshev_distance(rel_point, coords)),
    }


def _build_critical_spec(color: int) -> dict | None:
    for _ in range(_SPEC_RETRIES):
        mask = _random_mask("critical")
        critical = _ring_candidates(mask, 1, require_bbox_only=True)
        if not critical:
            continue
        farther_js = [j for j in (2, 3) if _ring_candidates(mask, j)]
        if not farther_js:
            continue
        first = random.choice(critical)
        second_j = random.choice(farther_js)
        second_choices = _pick_points(_ring_candidates(mask, second_j), 1, used=[first])
        if second_choices is None:
            continue
        points = [
            _make_point_record(mask, first, 1),
            _make_point_record(mask, second_choices[0], second_j),
        ]
        return {
            "color": int(color),
            "mask": mask,
            "points": points,
            "critical": True,
            "duplicate_ring": False,
        }
    return None


def _build_duplicate_spec(color: int) -> dict | None:
    for _ in range(_SPEC_RETRIES):
        mask = _random_mask("duplicate")
        good_choices = []
        for j in _RING_DISTANCE_CHOICES:
            chosen = _pick_points(_ring_candidates(mask, j), 2)
            if chosen is not None:
                good_choices.append((j, chosen))
        if not good_choices:
            continue
        distance, base_points = random.choice(good_choices)
        points = list(base_points)
        extra = _pick_points(_ring_candidates(mask, distance), 1, used=points)
        if extra is not None and random.random() < 0.5:
            points.extend(extra)
        return {
            "color": int(color),
            "mask": mask,
            "points": [_make_point_record(mask, point, distance) for point in points],
            "critical": False,
            "duplicate_ring": True,
        }
    return None


def _build_random_spec(color: int) -> dict | None:
    for _ in range(_SPEC_RETRIES):
        mask = _random_mask("random")
        available = {j: _ring_candidates(mask, j) for j in _RING_DISTANCE_CHOICES}
        available = {j: cells for j, cells in available.items() if cells}
        if not available:
            continue

        used: list[tuple[int, int]] = []
        point_records: list[dict] = []
        if len(available) >= 2 and random.random() < 0.6:
            distances = random.sample(sorted(available), 2)
        else:
            distances = [random.choice(sorted(available))]

        for distance in distances:
            chosen = _pick_points(available[distance], 1, used=used)
            if chosen is None:
                break
            used.extend(chosen)
            point_records.append(_make_point_record(mask, chosen[0], distance))
        else:
            if random.random() < 0.35:
                dup_distance = random.choice(distances)
                extra = _pick_points(available[dup_distance], 1, used=used)
                if extra is not None:
                    point_records.append(_make_point_record(mask, extra[0], dup_distance))
            return {
                "color": int(color),
                "mask": mask,
                "points": point_records,
                "critical": any(record["obj_dist"] > record["j"] for record in point_records),
                "duplicate_ring": len(point_records) > len({record["j"] for record in point_records}),
            }
    return None


def _object_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = int(color)
    return patch


def _reserve_patch(spec: dict) -> np.ndarray:
    max_j = max(point["j"] for point in spec["points"])
    h, w = spec["mask"].shape
    margin = max_j + 1
    return np.ones((h + 2 * margin, w + 2 * margin), dtype=int)


def _render_specs(
    shape: tuple[int, int],
    bg: int,
    specs: list[dict],
) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    h, w = shape
    occ = np.zeros((h, w), dtype=int)
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    placed: list[dict] = []

    for spec in specs:
        reserve = _reserve_patch(spec)
        try:
            rr, cc = random_free_location(occ, reserve, transparent=0, border=0, padding=0)
        except ValueError:
            return None
        paste(occ, reserve, rr, cc, transparent=0, clip=False)

        max_j = max(point["j"] for point in spec["points"])
        margin = max_j + 1
        r0 = rr + margin
        c0 = cc + margin
        patch = _object_patch(spec["mask"], spec["color"], bg)
        paste(inp, patch, r0, c0, transparent=bg, clip=False)
        paste(out, patch, r0, c0, transparent=bg, clip=False)

        bbox = (r0, c0, r0 + spec["mask"].shape[0] - 1, c0 + spec["mask"].shape[1] - 1)
        global_points = []
        for point in spec["points"]:
            pr, pc = point["rel"]
            gr = r0 + pr
            gc = c0 + pc
            inp[gr, gc] = int(spec["color"])
            global_points.append((gr, gc))

        for distance in sorted({int(point["j"]) for point in spec["points"]}):
            _paint_bbox_ring(out, bbox, distance, int(spec["color"]), bg)

        placed.append(
            {
                "color": int(spec["color"]),
                "bbox": bbox,
                "points": global_points,
                "point_info": spec["points"],
                "critical": bool(spec["critical"]),
                "duplicate_ring": bool(spec["duplicate_ring"]),
            }
        )
    return inp, out, placed


def _wrong_object_distance(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, use_bbox_distance=False, ring_mode="unique")


def _wrong_max_only(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, use_bbox_distance=True, ring_mode="max_only")


def _wrong_count_layers(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, use_bbox_distance=True, ring_mode="count_layers")


def _validate_instance(inp: np.ndarray, expected: np.ndarray, placed: list[dict], bg: int) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if int(get_bg(inp)) != bg:
        return False
    if not np.array_equal(main(inp), expected):
        return False

    has_bbox_critical = any(
        any(int(point["obj_dist"]) > int(point["j"]) for point in entry["point_info"]) for entry in placed
    )
    has_duplicate_ring = any(entry["duplicate_ring"] for entry in placed)
    has_multi_ring = any(len({int(point["j"]) for point in entry["point_info"]}) >= 2 for entry in placed)
    if not (has_bbox_critical and has_duplicate_ring and has_multi_ring):
        return False

    try:
        if np.array_equal(expected, _wrong_object_distance(inp)):
            return False
        if np.array_equal(expected, _wrong_max_only(inp)):
            return False
        if np.array_equal(expected, _wrong_count_layers(inp)):
            return False
    except AssertionError:
        return False
    return True


def _sample_specs(colors: list[int]) -> list[dict] | None:
    specs: list[dict] = []
    critical = _build_critical_spec(colors[0])
    duplicate = _build_duplicate_spec(colors[1])
    if critical is None or duplicate is None:
        return None
    specs.extend([critical, duplicate])

    for color in colors[2:]:
        extra = _build_random_spec(color)
        if extra is None:
            return None
        specs.append(extra)
    random.shuffle(specs)
    return specs


def _grid_shape_for_specs(specs: list[dict]) -> tuple[int, int]:
    reserve_heights = []
    reserve_widths = []
    reserve_area = 0
    for spec in specs:
        reserve = _reserve_patch(spec)
        reserve_heights.append(int(reserve.shape[0]))
        reserve_widths.append(int(reserve.shape[1]))
        reserve_area += int(reserve.shape[0] * reserve.shape[1])

    side = max(max(reserve_heights), max(reserve_widths), int(np.ceil(np.sqrt(reserve_area * 1.7))))
    min_side = max(18, side)
    max_side = 30
    if min_side > max_side:
        raise ValueError("specs do not fit into ARC size limits")
    h = random.randint(min_side, max_side)
    w = random.randint(min_side, max_side)
    return h, w


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_PALETTE)
    colors = [color for color in _PALETTE if color != bg]
    object_count = random.randint(2, 4)
    chosen_colors = random.sample(colors, object_count)
    specs = _sample_specs(chosen_colors)
    if specs is None:
        return None

    try:
        shape = _grid_shape_for_specs(specs)
    except ValueError:
        return None
    rendered = _render_specs(shape, bg, specs)
    if rendered is None:
        return None
    inp, expected, placed = rendered
    if not _validate_instance(inp, expected, placed, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ripple-freeze input")
