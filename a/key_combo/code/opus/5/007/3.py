from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import colors_present, find_components, get_bg, in_bounds


BG = 0
MARK_COLOR = 7
INPUT_TOP_MAX_ROW = 6
GRID_H_RANGE = (14, 16)
GRID_W_RANGE = (15, 18)
PALETTE = [1, 2, 3, 4, 5, 6, 8, 9]
MAX_GEN_ATTEMPTS = 800
SEAM_COUNT = 2

_BASE_SHAPES = [
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)),
]


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    arr = arr - arr.min(axis=0, keepdims=True)
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def _shape_variants(shape: tuple[tuple[int, int], ...]) -> list[np.ndarray]:
    pts = np.asarray(shape, dtype=int)
    seen: set[tuple[tuple[int, int], ...]] = set()
    variants: list[np.ndarray] = []
    for rot in range(4):
        q = pts.copy()
        for _ in range(rot):
            q = np.stack([q[:, 1], -q[:, 0]], axis=1)
        for flip in (False, True):
            t = q.copy()
            if flip:
                t[:, 1] *= -1
            norm = _normalize_coords(t)
            key = tuple(map(tuple, norm.tolist()))
            if key not in seen:
                seen.add(key)
                variants.append(norm)
    return variants


_SHAPES_BY_AREA: dict[int, list[np.ndarray]] = {}
for _shape in _BASE_SHAPES:
    for _variant in _shape_variants(_shape):
        _SHAPES_BY_AREA.setdefault(int(len(_variant)), []).append(_variant)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _drop_objects(grid: np.ndarray, mode: str = "area", *, strict: bool = True) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    out = np.full_like(g, bg)
    occupied = np.zeros((H, W), dtype=bool)

    for comp in sorted(comps, key=lambda d: (int(d["bbox"][0]), int(d["bbox"][1]), int(d["color"]))):
        coords = np.asarray(comp["coords"], dtype=int)
        if mode == "area":
            dr = int(comp["area"])
        elif mode == "bottom":
            dr = H - 1 - int(comp["bbox"][2])
        elif mode == "none":
            dr = 0
        else:
            raise ValueError(f"unsupported drop mode: {mode}")

        shifted = coords + np.array([dr, 0], dtype=int)
        keep = (0 <= shifted[:, 0]) & (shifted[:, 0] < H) & (0 <= shifted[:, 1]) & (shifted[:, 1] < W)
        kept = shifted[keep]
        if kept.size == 0:
            continue
        rr = kept[:, 0]
        cc = kept[:, 1]
        if strict:
            assert not bool(occupied[rr, cc].any()), "dropped objects must not overlap"
        occupied[rr, cc] = True
        out[rr, cc] = int(comp["color"])
    return out


def _mixed_diagonal_colors(grid: np.ndarray, r: int, c: int, bg: int) -> set[int]:
    colors: set[int] = set()
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        rr = r + dr
        cc = c + dc
        if in_bounds(grid.shape, rr, cc):
            color = int(grid[rr, cc])
            if color != bg:
                colors.add(color)
    return colors


def _has_orthogonal_foreground(grid: np.ndarray, r: int, c: int, bg: int) -> bool:
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) != bg:
            return True
    return False


def _mark_gaps(dropped: np.ndarray, *, require_empty_orth: bool = True) -> np.ndarray:
    g = np.asarray(dropped, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()
    H, W = g.shape
    for r in range(H):
        for c in range(W):
            if int(g[r, c]) != bg:
                continue
            colors = _mixed_diagonal_colors(g, r, c, bg)
            if len(colors) < 2:
                continue
            if require_empty_orth and _has_orthogonal_foreground(g, r, c, bg):
                continue
            out[r, c] = MARK_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each foreground object by its own area, then mark diagonal-only gaps between different colors.

    Input structure:
    - The grid has a black background and several separated monochrome 4-connected foreground objects.
    - Objects occupy the upper part of the grid, use a small set of non-black colors, and may have different areas.
    - Objects are independent: they move simultaneously and are clipped if part of a dropped copy passes below the grid.

    Rule:
    - For each foreground object, compute its area A and translate the whole object downward by A rows.
    - Clear the original positions, keep each object's original color, and discard any dropped pixels that fall outside the bottom edge.
    - On the post-drop grid, recolor any background cell to orange if its diagonal neighbors contain at least two different non-background colors and its four orthogonal neighbors are all background.

    Key insight:
    - The orange cells are not determined from the input arrangement directly; they appear only after the area-dependent drops create new pure diagonal seams between differently colored objects.
    """
    dropped = _drop_objects(np.asarray(grid, dtype=int), mode="area")
    return _mark_gaps(dropped, require_empty_orth=True)


def _quadrant_coords(base: np.ndarray, quadrant: str) -> np.ndarray:
    if quadrant == "NW":
        return np.column_stack((-1 - base[:, 0], -1 - base[:, 1]))
    if quadrant == "NE":
        return np.column_stack((-1 - base[:, 0], 1 + base[:, 1]))
    if quadrant == "SW":
        return np.column_stack((1 + base[:, 0], -1 - base[:, 1]))
    if quadrant == "SE":
        return np.column_stack((1 + base[:, 0], 1 + base[:, 1]))
    raise ValueError(f"bad quadrant: {quadrant}")


def _shift(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return np.asarray(coords, dtype=int) + np.array([dr, dc], dtype=int)


def _coords_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, p)) for p in np.asarray(coords, dtype=int).tolist()}


def _orth_touch(a: np.ndarray, b: np.ndarray) -> bool:
    bs = _coords_set(b)
    for r, c in _coords_set(a):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (r + dr, c + dc) in bs:
                return True
    return False


def _object_grid(shape: tuple[int, int], objects: list[dict], dropped: bool = False) -> np.ndarray:
    H, W = shape
    grid = np.full((H, W), BG, dtype=int)
    for obj in objects:
        coords = np.asarray(obj["dropped_coords"] if dropped else obj["input_coords"], dtype=int)
        if coords.size == 0:
            continue
        grid[coords[:, 0], coords[:, 1]] = int(obj["color"])
    return grid


def _input_boxes_are_separate(objects: list[dict]) -> bool:
    boxes = [obj["input_bbox"] for obj in objects]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a = boxes[i]
            b = boxes[j]
            if not (a[2] + 1 < b[0] or b[2] + 1 < a[0] or a[3] + 1 < b[1] or b[3] + 1 < a[1]):
                return False
    return True


def _choose_shape(area: int) -> np.ndarray:
    return np.asarray(random.choice(_SHAPES_BY_AREA[int(area)]), dtype=int)


def _make_object_from_final(final_coords: np.ndarray, color: int) -> dict | None:
    final_coords = np.asarray(final_coords, dtype=int)
    area = int(final_coords.shape[0])
    input_coords = _shift(final_coords, -area, 0)
    if input_coords.size == 0:
        return None
    if input_coords[:, 0].min() < 0 or input_coords[:, 0].max() > INPUT_TOP_MAX_ROW:
        return None
    if input_coords[:, 1].min() < 0:
        return None
    return {
        "color": int(color),
        "area": area,
        "input_coords": input_coords,
        "dropped_coords": final_coords,
        "input_bbox": _bbox(input_coords),
        "dropped_bbox": _bbox(final_coords),
    }


def _objects_compatible(existing: list[dict], new_objs: list[dict]) -> bool:
    trial = existing + new_objs
    if not _input_boxes_are_separate(trial):
        return False
    for i in range(len(trial)):
        for j in range(i + 1, len(trial)):
            if _coords_set(trial[i]["input_coords"]) & _coords_set(trial[j]["input_coords"]):
                return False
            if _coords_set(trial[i]["dropped_coords"]) & _coords_set(trial[j]["dropped_coords"]):
                return False
    return True


def _sample_seam_pair(
    H: int,
    W: int,
    color_a: int,
    color_b: int,
    existing: list[dict],
    col_band: tuple[int, int],
) -> tuple[list[dict], tuple[int, int]] | None:
    for _ in range(220):
        area_a = random.choice([3, 4, 5, 6, 7])
        area_b = random.choice([3, 4, 5, 6, 7])
        shape_a = _choose_shape(area_a)
        shape_b = _choose_shape(area_b)
        qa, qb = random.choice([("NW", "SE"), ("NE", "SW")])
        rel_a = _quadrant_coords(shape_a, qa)
        rel_b = _quadrant_coords(shape_b, qb)
        row_lo = max(-int(rel_a[:, 0].min()), -int(rel_b[:, 0].min()))
        row_hi = min(H - 1 - int(rel_a[:, 0].max()), H - 1 - int(rel_b[:, 0].max()))
        col_lo = max(-int(rel_a[:, 1].min()), -int(rel_b[:, 1].min()), int(col_band[0]))
        col_hi = min(W - 1 - int(rel_a[:, 1].max()), W - 1 - int(rel_b[:, 1].max()), int(col_band[1]))
        if row_lo > row_hi or col_lo > col_hi:
            continue
        seam_r = random.randint(row_lo, row_hi)
        seam_c = random.randint(col_lo, col_hi)
        obj_a = _make_object_from_final(_shift(rel_a, seam_r, seam_c), color_a)
        obj_b = _make_object_from_final(_shift(rel_b, seam_r, seam_c), color_b)
        if obj_a is None or obj_b is None:
            continue
        new_objs = [obj_a, obj_b]
        if _coords_set(obj_a["input_coords"]) & _coords_set(obj_b["input_coords"]):
            continue
        if _coords_set(obj_a["dropped_coords"]) & _coords_set(obj_b["dropped_coords"]):
            continue
        if not _objects_compatible(existing, new_objs):
            continue
        return new_objs, (int(seam_r), int(seam_c))
    return None


def _place_extras(H: int, W: int, objects: list[dict], extra_colors: list[int], require_clip: bool) -> list[dict] | None:
    placed = [dict(obj) for obj in objects]
    clipped = False
    for idx, color in enumerate(extra_colors):
        success = False
        for _ in range(180):
            area = random.choice([3, 4, 5, 6, 7])
            shape = _choose_shape(area)
            h = int(shape[:, 0].max()) + 1
            w = int(shape[:, 1].max()) + 1
            max_r0 = INPUT_TOP_MAX_ROW - h + 1
            if max_r0 < 0 or w > W:
                continue
            r0_choices = list(range(max_r0 + 1))
            if require_clip and idx == 0:
                r0_choices = [r for r in r0_choices if r + area + h - 1 >= H]
                if not r0_choices:
                    continue
            r0 = random.choice(r0_choices)
            c0 = random.randint(0, W - w)
            input_coords = _shift(shape, r0, c0)
            dropped_full = _shift(input_coords, area, 0)
            keep = (0 <= dropped_full[:, 0]) & (dropped_full[:, 0] < H)
            dropped_coords = dropped_full[keep]
            if dropped_coords.size == 0:
                continue
            cand = {
                "color": int(color),
                "area": area,
                "input_coords": input_coords,
                "dropped_coords": dropped_coords,
                "input_bbox": _bbox(input_coords),
                "dropped_bbox": _bbox(dropped_coords),
            }
            if any(_coords_set(cand["input_coords"]) & _coords_set(obj["input_coords"]) for obj in placed):
                continue
            if any(_coords_set(cand["dropped_coords"]) & _coords_set(obj["dropped_coords"]) for obj in placed):
                continue
            placed.append(cand)
            clipped = clipped or bool((~keep).any())
            success = True
            break
        if not success:
            return None
    if require_clip and not clipped:
        return None
    return placed


def _false_positive_gap_exists(dropped: np.ndarray) -> bool:
    bg = int(get_bg(dropped))
    H, W = dropped.shape
    for r in range(H):
        for c in range(W):
            if int(dropped[r, c]) != bg:
                continue
            colors = _mixed_diagonal_colors(dropped, r, c, bg)
            if len(colors) >= 2 and _has_orthogonal_foreground(dropped, r, c, bg):
                return True
    return False


def _validate_semantics(inp: np.ndarray, objects: list[dict], seam_cells: list[tuple[int, int]]) -> bool:
    H, W = inp.shape
    if not (inp.ndim == 2 and max(inp.shape) <= 30):
        return False
    if len(objects) < 4 or len(objects) > 6:
        return False
    if not _input_boxes_are_separate(objects):
        return False
    for obj in objects:
        rows = obj["input_coords"][:, 0]
        if int(rows.min()) < 0 or int(rows.max()) > INPUT_TOP_MAX_ROW:
            return False
        if obj["dropped_coords"].size == 0:
            return False

    by_color: dict[int, list[np.ndarray]] = {}
    for obj in objects:
        by_color.setdefault(int(obj["color"]), []).append(np.asarray(obj["dropped_coords"], dtype=int))
    if not (2 <= len(by_color) <= 3):
        return False
    for coords_list in by_color.values():
        if not (2 <= len(coords_list) <= 3):
            return False
        for i in range(len(coords_list)):
            for j in range(i + 1, len(coords_list)):
                if _orth_touch(coords_list[i], coords_list[j]):
                    return False

    expected_dropped = _object_grid((H, W), objects, dropped=True)
    if not np.array_equal(expected_dropped, _drop_objects(inp, mode="area", strict=True)):
        return False
    out = main(inp)
    orange = np.argwhere(out == MARK_COLOR)
    if orange.shape[0] < 2:
        return False
    for seam_r, seam_c in seam_cells:
        if not in_bounds(out.shape, seam_r, seam_c) or int(out[seam_r, seam_c]) != MARK_COLOR:
            return False

    no_drop = _mark_gaps(inp, require_empty_orth=True)
    bottom_drop = _mark_gaps(_drop_objects(inp, mode="bottom", strict=False), require_empty_orth=True)
    loose = _mark_gaps(expected_dropped, require_empty_orth=False)
    if np.array_equal(no_drop, out):
        return False
    if np.array_equal(bottom_drop, out):
        return False
    if np.array_equal(loose, out):
        return False
    if not _false_positive_gap_exists(expected_dropped):
        return False
    if len({int(obj["area"]) for obj in objects}) < 2:
        return False
    return True


def _build_candidate() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    color1, color2 = random.sample(PALETTE, 2)
    counts = random.choice([(2, 2), (3, 2), (2, 3), (3, 3)])
    extra_colors = [color1] * (counts[0] - 2) + [color2] * (counts[1] - 2)
    random.shuffle(extra_colors)
    require_clip = bool(extra_colors) and random.random() < 0.55

    pair1 = random.choice([(color1, color2), (color2, color1)])
    pair2 = random.choice([(color1, color2), (color2, color1)])
    left_band = (2, max(2, W // 2 - 1))
    right_band = (min(W // 2 + 1, W - 3), W - 3)
    first = _sample_seam_pair(H, W, pair1[0], pair1[1], [], left_band)
    if first is None:
        return None
    base_objects, seam1 = first
    second = _sample_seam_pair(H, W, pair2[0], pair2[1], base_objects, right_band)
    if second is None:
        return None
    extra_objects, seam2 = second
    if abs(seam1[0] - seam2[0]) + abs(seam1[1] - seam2[1]) < 6:
        return None
    objects = base_objects + extra_objects
    full_objects = _place_extras(H, W, objects, extra_colors, require_clip=require_clip)
    if full_objects is None:
        return None
    inp = _object_grid((H, W), full_objects, dropped=False)
    if _validate_semantics(inp, full_objects, [seam1, seam2]):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        candidate = _build_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError(f"failed to generate a valid sample after {MAX_GEN_ATTEMPTS} attempts")
