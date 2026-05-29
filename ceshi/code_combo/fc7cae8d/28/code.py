# Combo variant of fc7cae8d (index 28)
# combo_spec: D5/1+D7/0
# Name: 移动轨迹+多对多
#
# Design analysis for the fused variant
# a) Concrete behavior:
#    The original puzzle had one interior object and returned only the cropped core.
#    Here the border can expose 2-4 active corner controls. Each interior rectangle
#    has two colored guide edges; after the correct dihedral transform those guide
#    colors must align to one active corner's horizontal/vertical control pair.
#    The output keeps the full grid: the source object disappears, its cropped core
#    is moved into the matched corner, and the moving corner of the core leaves a
#    1-pixel L-shaped trail colored with that object's dominant fill color.
# b) Complexity calibration:
#    Four global side colors (top/bottom/left/right) are reused across all objects.
#    This keeps the palette small enough to read, while 2-4 objects still force a
#    real many-to-many assignment over multiple corners rather than a single match.
# c) Hidden assumptions broken from the original solver:
#    - "largest interior component" no longer works: now we must extract all objects.
#    - "two solid full-frame sides" no longer works: controls are local corner runs.
#    - "return the cropped object immediately" no longer works: output is full-grid.
#    - "one match is enough" no longer works: every object is solved independently.
# d) Placement legality:
#    The generator keeps all source objects away from the border and from each other,
#    and final corner cores are kept small enough that corner placements never overlap.
# e) Visual observability:
#    Across samples, people can see repeated side colors on the frame, repeated guide
#    colors on several objects, and repeated same-color trails from object to corner.
#    The motion is visible without relying on absolute color identities.
# f) Visual cleanliness:
#    Trails are 1-pixel wide and cores are small asymmetric rectangles with a few
#    internal holes, so the board stays readable even with multiple objects.
# g) Color inference:
#    Guide colors encode sides; trail colors encode object identity by reusing the
#    core fill color. The palette stays structured: 4 side-control colors + 2-4 fills.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import crop_bbox, find_components, get_bg, paste

CORNER_SIDES: dict[str, tuple[str, str]] = {
    "tl": ("top", "left"),
    "tr": ("top", "right"),
    "bl": ("bottom", "left"),
    "br": ("bottom", "right"),
}
SIDES_TO_CORNER = {v: k for k, v in CORNER_SIDES.items()}
TRANSFORMS = [(k, flip) for k in range(4) for flip in (False, True)]


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _run_from_start(vec: np.ndarray, bg: int, min_len: int = 3) -> tuple[int, int] | None:
    vals = list(map(int, vec))
    if not vals or vals[0] == int(bg):
        return None
    color = int(vals[0])
    count = 0
    for v in vals:
        if int(v) != color:
            break
        count += 1
    if count < int(min_len):
        return None
    return color, count


def _dominant_non_bg(arr: np.ndarray, bg: int) -> int:
    vals = [int(x) for x in arr.flatten() if int(x) != int(bg)]
    if not vals:
        return int(bg)
    return int(Counter(vals).most_common(1)[0][0])


def _detect_active_corners(arr: np.ndarray, bg: int) -> dict[str, dict[str, int]]:
    top = arr[0, :]
    bottom = arr[-1, :]
    left = arr[:, 0]
    right = arr[:, -1]

    scans = {
        "tl": (_run_from_start(top[1:], bg), _run_from_start(left[1:], bg)),
        "tr": (_run_from_start(top[-2::-1], bg), _run_from_start(right[1:], bg)),
        "bl": (_run_from_start(bottom[1:], bg), _run_from_start(left[-2::-1], bg)),
        "br": (_run_from_start(bottom[-2::-1], bg), _run_from_start(right[-2::-1], bg)),
    }

    out: dict[str, dict[str, int]] = {}
    for corner, (h_run, v_run) in scans.items():
        if h_run is None or v_run is None:
            continue
        h_side, v_side = CORNER_SIDES[corner]
        out[corner] = {h_side: int(h_run[0]), v_side: int(v_run[0])}
    return out


def _guide_edges(arr: np.ndarray, bg: int) -> dict[str, int]:
    fill = _dominant_non_bg(arr, bg)
    edges = {
        "top": arr[0, :],
        "bottom": arr[-1, :],
        "left": arr[:, 0],
        "right": arr[:, -1],
    }
    out: dict[str, int] = {}
    for side, edge in edges.items():
        vals = list(map(int, edge))
        others = [x for x in vals if x not in {int(fill), int(bg)}]
        if not others:
            continue
        color, count = Counter(others).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            out[side] = int(color)
    return out


def _crop_core_from_aligned(arr: np.ndarray, sides: tuple[str, str]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    top = 1 if "top" in sides else 0
    bottom = int(arr.shape[0]) - (1 if "bottom" in sides else 0)
    left = 1 if "left" in sides else 0
    right = int(arr.shape[1]) - (1 if "right" in sides else 0)
    return arr[int(top) : int(bottom), int(left) : int(right)].copy(), (int(top), int(left), int(bottom), int(right))


def _aligned_anchor(corner: str, crop_box: tuple[int, int, int, int]) -> tuple[int, int]:
    top, left, bottom, right = crop_box
    if corner == "tl":
        return int(top), int(left)
    if corner == "tr":
        return int(top), int(right - 1)
    if corner == "bl":
        return int(bottom - 1), int(left)
    return int(bottom - 1), int(right - 1)


def _inverse_coord(aligned_shape: tuple[int, int], coord: tuple[int, int], k: int, flip: bool) -> tuple[int, int]:
    marker = np.zeros(aligned_shape, dtype=int)
    marker[int(coord[0]), int(coord[1])] = 1
    inv = _inverse_transform(marker, k, flip)
    rr, cc = np.argwhere(inv == 1)[0]
    return int(rr), int(cc)


def _final_origin(shape: tuple[int, int], corner: str, core_shape: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, core_shape)
    H, W = map(int, shape)
    if corner == "tl":
        return 1, 1
    if corner == "tr":
        return 1, W - 1 - w
    if corner == "bl":
        return H - 1 - h, 1
    return H - 1 - h, W - 1 - w


def _final_anchor(origin: tuple[int, int], corner: str, core_shape: tuple[int, int]) -> tuple[int, int]:
    r0, c0 = map(int, origin)
    h, w = map(int, core_shape)
    if corner == "tl":
        return r0, c0
    if corner == "tr":
        return r0, c0 + w - 1
    if corner == "bl":
        return r0 + h - 1, c0
    return r0 + h - 1, c0 + w - 1


def _draw_trail(out: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int) -> None:
    sr, sc = map(int, start)
    er, ec = map(int, end)
    step_r = 1 if er >= sr else -1
    for rr in range(sr, er + step_r, step_r):
        out[rr, sc] = int(color)
    step_c = 1 if ec >= sc else -1
    for cc in range(sc, ec + step_c, step_c):
        out[er, cc] = int(color)


def _extract_objects(arr: np.ndarray, bg: int) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    H, W = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    objs: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1:
            continue
        if int(comp["area"]) < 8:
            continue
        objs.append(((r0, c0, r1, c1), crop_bbox(arr, (r0, c0, r1, c1))))
    objs.sort(key=lambda item: item[0])
    return objs


def _solve_object(obj: np.ndarray, active_corners: dict[str, dict[str, int]], bg: int) -> dict:
    matches: list[dict] = []
    for k, flip in TRANSFORMS:
        aligned = _transform(obj, k, flip)
        gedges = _guide_edges(aligned, bg)
        if len(gedges) != 2:
            continue
        horiz = next((s for s in ("top", "bottom") if s in gedges), None)
        vert = next((s for s in ("left", "right") if s in gedges), None)
        if horiz is None or vert is None:
            continue
        corner = SIDES_TO_CORNER.get((horiz, vert))
        if corner is None or corner not in active_corners:
            continue
        wanted = active_corners[corner]
        if int(gedges[horiz]) != int(wanted[horiz]) or int(gedges[vert]) != int(wanted[vert]):
            continue
        core, crop_box = _crop_core_from_aligned(aligned, (horiz, vert))
        anchor_aligned = _aligned_anchor(corner, crop_box)
        anchor_local = _inverse_coord(aligned.shape, anchor_aligned, k, flip)
        matches.append(
            {
                "corner": corner,
                "core": core,
                "anchor_local": anchor_local,
                "fill": _dominant_non_bg(core, bg),
            }
        )
    if not matches:
        raise ValueError("object has no valid corner match")
    return matches[0]


def _analyze_instance(grid: np.ndarray) -> tuple[int, dict[str, dict[str, int]], list[dict]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    active_corners = _detect_active_corners(g, bg)
    solved: list[dict] = []
    for bbox, obj in _extract_objects(g, bg):
        info = _solve_object(obj, active_corners, bg)
        origin = _final_origin(g.shape, info["corner"], info["core"].shape)
        target_anchor = _final_anchor(origin, info["corner"], info["core"].shape)
        source_anchor = (int(bbox[0]) + int(info["anchor_local"][0]), int(bbox[1]) + int(info["anchor_local"][1]))
        solved.append(
            {
                "bbox": bbox,
                "corner": info["corner"],
                "core": info["core"],
                "fill": int(info["fill"]),
                "source_anchor": source_anchor,
                "target_origin": origin,
                "target_anchor": target_anchor,
            }
        )
    return bg, active_corners, solved


def _render_output(shape: tuple[int, int], bg: int, border_template: np.ndarray, solved: list[dict]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    out[0, :] = border_template[0, :]
    out[-1, :] = border_template[-1, :]
    out[:, 0] = border_template[:, 0]
    out[:, -1] = border_template[:, -1]
    for info in solved:
        _draw_trail(out, info["source_anchor"], info["target_anchor"], info["fill"])
    for info in solved:
        r0, c0 = map(int, info["target_origin"])
        paste(out, info["core"], r0, c0, transparent=bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    bg, _active_corners, solved = _analyze_instance(grid)
    border_template = np.array(grid, dtype=int)
    border_only = np.full_like(border_template, bg)
    border_only[0, :] = border_template[0, :]
    border_only[-1, :] = border_template[-1, :]
    border_only[:, 0] = border_template[:, 0]
    border_only[:, -1] = border_template[:, -1]
    return _render_output(border_template.shape, bg, border_only, solved)


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    bg, active_corners, solved = _analyze_instance(grid)
    total_move = 0
    for info in solved:
        sr, sc = info["source_anchor"]
        tr, tc = info["target_anchor"]
        total_move += abs(int(sr) - int(tr)) + abs(int(sc) - int(tc))
    return {
        "bg": int(bg),
        "active_corners": int(len(active_corners)),
        "objects": int(len(solved)),
        "total_manhattan_move": int(total_move),
    }


def _is_asymmetric(core: np.ndarray) -> bool:
    views = set()
    for k, flip in TRANSFORMS:
        t = _transform(core, k, flip)
        views.add((tuple(t.shape), t.tobytes()))
    return len(views) >= 4


def _build_core(h: int, w: int, fill: int) -> np.ndarray:
    for _ in range(200):
        core = np.full((h, w), int(fill), dtype=int)
        interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        random.shuffle(interior)
        target_holes = random.randint(2, max(2, min(5, len(interior) // 2)))
        for r, c in interior[:target_holes]:
            core[r, c] = 0
        if _is_asymmetric(core):
            return core
    raise RuntimeError("failed to build asymmetric core")


def _attach_guides(core: np.ndarray, sides: tuple[str, str], colors: dict[str, int]) -> np.ndarray:
    h, w = core.shape
    has_top = "top" in sides
    has_bottom = "bottom" in sides
    has_left = "left" in sides
    has_right = "right" in sides
    out = np.zeros((h + has_top + has_bottom, w + has_left + has_right), dtype=int)
    r0 = 1 if has_top else 0
    c0 = 1 if has_left else 0
    paste(out, core, r0, c0)
    if has_top:
        out[0, c0 : c0 + w] = int(colors["top"])
    if has_bottom:
        out[-1, c0 : c0 + w] = int(colors["bottom"])
    if has_left:
        out[r0 : r0 + h, 0] = int(colors["left"])
    if has_right:
        out[r0 : r0 + h, -1] = int(colors["right"])
    if has_top and has_left:
        out[0, 0] = 0
    if has_top and has_right:
        out[0, -1] = 0
    if has_bottom and has_left:
        out[-1, 0] = 0
    if has_bottom and has_right:
        out[-1, -1] = 0
    return out


def _paint_corner_controls(grid: np.ndarray, corners: list[str], side_colors: dict[str, int]) -> np.ndarray:
    out = np.zeros_like(grid)
    n = int(grid.shape[0])
    seg_len = random.randint(4, min(7, max(4, n // 3)))
    for corner in corners:
        horiz, vert = CORNER_SIDES[corner]
        if corner == "tl":
            out[0, 1 : 1 + seg_len] = int(side_colors[horiz])
            out[1 : 1 + seg_len, 0] = int(side_colors[vert])
        elif corner == "tr":
            out[0, n - 1 - seg_len : n - 1] = int(side_colors[horiz])
            out[1 : 1 + seg_len, -1] = int(side_colors[vert])
        elif corner == "bl":
            out[-1, 1 : 1 + seg_len] = int(side_colors[horiz])
            out[n - 1 - seg_len : n - 1, 0] = int(side_colors[vert])
        else:
            out[-1, n - 1 - seg_len : n - 1] = int(side_colors[horiz])
            out[n - 1 - seg_len : n - 1, -1] = int(side_colors[vert])
    return out


def _dilated_occupancy(canvas: np.ndarray, bg: int, padding: int) -> np.ndarray:
    occ = (canvas != int(bg))
    if padding <= 0:
        return occ
    return ndimage.binary_dilation(occ, iterations=int(padding), structure=np.ones((3, 3), dtype=bool))


def _free_locations(canvas: np.ndarray, patch: np.ndarray, *, bg: int = 0, border: int = 2, padding: int = 1) -> list[tuple[int, int]]:
    H, W = canvas.shape
    h, w = patch.shape
    occ = _dilated_occupancy(canvas, bg, padding)
    mask = (patch != int(bg))
    out: list[tuple[int, int]] = []
    for r0 in range(int(border), H - h - int(border) + 1):
        for c0 in range(int(border), W - w - int(border) + 1):
            if not bool((occ[r0 : r0 + h, c0 : c0 + w] & mask).any()):
                out.append((int(r0), int(c0)))
    return out


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(25, 30)
    all_corners = list(CORNER_SIDES)
    n_obj = random.randint(2, 4)
    corners = random.sample(all_corners, n_obj)

    palette = list(range(1, 10))
    top_c, bottom_c, left_c, right_c = random.sample(palette, 4)
    side_colors = {"top": top_c, "bottom": bottom_c, "left": left_c, "right": right_c}
    fill_pool = [c for c in palette if c not in side_colors.values()]
    if len(fill_pool) < n_obj:
        return None
    fills = random.sample(fill_pool, n_obj)

    specs: list[dict] = []
    final_rects: list[tuple[int, int, int, int]] = []
    for corner, fill in zip(corners, fills):
        h = random.randint(4, 6)
        w = random.randint(4, 6)
        core = _build_core(h, w, fill)
        origin = _final_origin((n, n), corner, core.shape)
        rect = (origin[0], origin[1], origin[0] + h - 1, origin[1] + w - 1)
        if any(_rectangles_overlap(rect, other) for other in final_rects):
            return None
        final_rects.append(rect)
        aligned = _attach_guides(core, CORNER_SIDES[corner], side_colors)
        crop_box = _crop_core_from_aligned(aligned, CORNER_SIDES[corner])[1]
        anchor_aligned = _aligned_anchor(corner, crop_box)
        k = random.randrange(4)
        flip = bool(random.randrange(2))
        input_object = _inverse_transform(aligned, k, flip)
        anchor_local = _inverse_coord(aligned.shape, anchor_aligned, k, flip)
        specs.append(
            {
                "corner": corner,
                "fill": int(fill),
                "core": core,
                "input_object": input_object,
                "anchor_local": anchor_local,
                "target_origin": origin,
                "target_anchor": _final_anchor(origin, corner, core.shape),
            }
        )

    border_only = _paint_corner_controls(np.zeros((n, n), dtype=int), corners, side_colors)
    grid = border_only.copy()

    for info in specs:
        candidates = _free_locations(grid, info["input_object"], bg=0, border=2, padding=1)
        if not candidates:
            return None
        random.shuffle(candidates)
        chosen = None
        min_move = random.randint(8, 14)
        for r0, c0 in candidates:
            source_anchor = (int(r0) + int(info["anchor_local"][0]), int(c0) + int(info["anchor_local"][1]))
            dist = abs(int(source_anchor[0]) - int(info["target_anchor"][0])) + abs(int(source_anchor[1]) - int(info["target_anchor"][1]))
            if dist >= min_move:
                chosen = (int(r0), int(c0), source_anchor)
                break
        if chosen is None:
            return None
        info["source_anchor"] = chosen[2]
        paste(grid, info["input_object"], chosen[0], chosen[1], transparent=0)

    expected = _render_output((n, n), 0, border_only, specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    metrics = variant_metrics(grid)
    if metrics["objects"] < 2 or metrics["active_corners"] < 2 or metrics["total_manhattan_move"] < 18:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate combo input for fc7cae8d")
