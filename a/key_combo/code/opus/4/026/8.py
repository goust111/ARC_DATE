from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "arc_utils_for_opus_4_026_8",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()

bbox_coords = _ARC_UTILS.bbox_coords
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
paste = _ARC_UTILS.paste


SIDE_TO_VEC = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
ROTATIONS_FROM_TOP = {"up": 0, "right": 3, "down": 2, "left": 1}
SIDES = ("up", "right", "down", "left")
GEN_COLORS = [1, 2, 3, 4]
GEN_COUNTS = [6, 7, 8]
GEN_GRID_RANGE = (24, 28)
GEN_OBJECT_COUNT_CHOICES = (3,)
GEN_ATTEMPTS = 500

TOP_TEMPLATES = {
    6: np.array([[1, 0, 1], [1, 1, 1], [1, 0, 0]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
    8: np.array([[1, 0, 1], [1, 1, 1], [1, 1, 1]], dtype=int),
}


def _notch_indices(line: np.ndarray, fg: int, bg: int) -> list[int]:
    return [
        i
        for i in range(1, len(line) - 1)
        if int(line[i]) == bg and int(line[i - 1]) == fg and int(line[i + 1]) == fg
    ]


def _detect_notch_side(patch: np.ndarray, fg: int, bg: int) -> str:
    sides: list[str] = []
    if _notch_indices(patch[0, :], fg, bg):
        sides.append("up")
    if _notch_indices(patch[:, -1], fg, bg):
        sides.append("right")
    if _notch_indices(patch[-1, :], fg, bg):
        sides.append("down")
    if _notch_indices(patch[:, 0], fg, bg):
        sides.append("left")
    assert len(sides) == 1, "each object must have exactly one notch side"
    return sides[0]


def _tile_positions(bbox: tuple[int, int, int, int], patch_shape: tuple[int, int], side: str, count: int) -> list[tuple[int, int]]:
    r0, c0, _r1, _c1 = map(int, bbox)
    h, w = map(int, patch_shape)
    dr, dc = SIDE_TO_VEC[side]
    step = h if dr else w
    return [(r0 + dr * step * k, c0 + dc * step * k) for k in range(int(count))]


def _footprint_mask(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    patch: np.ndarray,
    side: str,
    count: int,
    bg: int,
) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    fg_coords = np.argwhere(patch != bg)
    for r0, c0 in _tile_positions(bbox, patch.shape, side, count):
        abs_coords = fg_coords + np.array([r0, c0], dtype=int)
        keep = (
            (abs_coords[:, 0] >= 0)
            & (abs_coords[:, 0] < shape[0])
            & (abs_coords[:, 1] >= 0)
            & (abs_coords[:, 1] < shape[1])
        )
        abs_coords = abs_coords[keep]
        out[abs_coords[:, 0], abs_coords[:, 1]] = True
    return out


def _render_objects(shape: tuple[int, int], objects: list[dict], bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    draw_order = sorted(
        objects,
        key=lambda obj: (int(obj["count"]), int(obj["bbox"][0]), int(obj["bbox"][1])),
        reverse=True,
    )
    for obj in draw_order:
        for r0, c0 in _tile_positions(obj["bbox"], obj["patch"].shape, obj["side"], obj["count"]):
            paste(out, obj["patch"], r0, c0, transparent=bg, clip=True)
    return out


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    objects: list[dict] = []
    for component in components:
        bbox = tuple(map(int, component["bbox"]))
        color = int(component["color"])
        patch = crop_bbox(g, bbox)
        side = _detect_notch_side(patch, color, bg)
        objects.append(
            {
                "bbox": bbox,
                "color": color,
                "count": int(component["area"]),
                "patch": patch,
                "side": side,
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """Tile each notched object along its notch direction as many times as its pixel count, with smaller objects overwriting larger ones on overlaps.

    Input structure:
    - The grid has a black background and 3-4 separated monochromatic 4-connected foreground objects in distinct colors.
    - Each object's bbox is at least 3x3, and exactly one bbox side contains a one-cell background notch between two same-color boundary pixels.

    Rule:
    - Detect each object's notch side from its own bbox boundary: top notch means tile upward, right notch means tile rightward, bottom notch means tile downward, and left notch means tile leftward.
    - Count the object's foreground pixels C, crop the object to its bbox, and place C tight copies in that direction, including the original copy.
    - Clip any out-of-bounds copy cells.
    - Draw all tiled regions in descending pixel-count order so that smaller-count objects are painted later and therefore appear on top; if counts tie, the earlier bbox in scan order is on top.

    Key insight:
    - Pixel count has two simultaneous roles: it sets both the tiling length and the occlusion priority, while the boundary notch converts a local missing cell into the global tiling direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    objects = _extract_objects(g)
    return _render_objects(g.shape, objects, bg)


def _template_patch(side: str, count: int, color: int, bg: int = 0) -> np.ndarray:
    mask = np.rot90(TOP_TEMPLATES[int(count)], ROTATIONS_FROM_TOP[side])
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask.astype(bool)] = int(color)
    return patch


def _anchor_bounds(side: str, count: int, h: int, w: int) -> tuple[tuple[int, int], tuple[int, int]]:
    size = 3
    if side == "up":
        row_range = (0, h - size * count)
        col_range = (0, w - size)
    elif side == "down":
        row_range = (size * (count - 1), h - size)
        col_range = (0, w - size)
    elif side == "left":
        row_range = (0, h - size)
        col_range = (0, w - size * count)
    else:
        row_range = (0, h - size)
        col_range = (size * (count - 1), w - size)
    return row_range, col_range


def _origin_from_anchor(side: str, count: int, anchor: tuple[int, int]) -> tuple[int, int]:
    r, c = map(int, anchor)
    delta = 3 * (int(count) - 1)
    if side == "up":
        return r + delta, c
    if side == "down":
        return r - delta, c
    if side == "left":
        return r, c + delta
    return r, c - delta


def _bbox_from_origin(origin: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0 = map(int, origin)
    return r0, c0, r0 + 2, c0 + 2


def _bboxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _orientation(side: str) -> str:
    return "vertical" if side in {"up", "down"} else "horizontal"


def _build_object_specs() -> list[dict]:
    n_objects = random.choice(GEN_OBJECT_COUNT_CHOICES)
    sides = random.sample(SIDES, n_objects)
    counts = random.sample(GEN_COUNTS, n_objects)
    colors = random.sample(GEN_COLORS, n_objects)
    random.shuffle(sides)
    return [
        {
            "side": side,
            "count": count,
            "color": color,
            "patch": _template_patch(side, count, color),
        }
        for side, count, color in zip(sides, counts, colors)
    ]


def _choose_cross_pair(specs: list[dict]) -> tuple[int, int]:
    vertical = [i for i, spec in enumerate(specs) if _orientation(spec["side"]) == "vertical"]
    horizontal = [i for i, spec in enumerate(specs) if _orientation(spec["side"]) == "horizontal"]
    return random.choice(vertical), random.choice(horizontal)


def _candidate_anchors(
    spec: dict,
    shape: tuple[int, int],
    fixed_row: int | None = None,
    fixed_col: int | None = None,
) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    (r0, r1), (c0, c1) = _anchor_bounds(spec["side"], spec["count"], h, w)
    if fixed_row is not None:
        if not (r0 <= fixed_row <= r1):
            return []
        rows = [int(fixed_row)]
    else:
        rows = list(range(int(r0), int(r1) + 1))
    if fixed_col is not None:
        if not (c0 <= fixed_col <= c1):
            return []
        cols = [int(fixed_col)]
    else:
        cols = list(range(int(c0), int(c1) + 1))
    out = [(r, c) for r in rows for c in cols]
    random.shuffle(out)
    return out


def _place_specs(shape: tuple[int, int], specs: list[dict]) -> list[dict] | None:
    placed = [dict(spec) for spec in specs]
    v_idx, h_idx = _choose_cross_pair(placed)
    cross_rows = _candidate_anchors(placed[v_idx], shape)
    if not cross_rows:
        return None

    pair_candidates = []
    for cross_r, cross_c in cross_rows:
        if _candidate_anchors(placed[h_idx], shape, fixed_row=cross_r, fixed_col=cross_c):
            pair_candidates.append((cross_r, cross_c))
    random.shuffle(pair_candidates)
    if not pair_candidates:
        return None

    cross_r, cross_c = pair_candidates[0]
    placed[v_idx]["anchor"] = (cross_r, cross_c)
    placed[h_idx]["anchor"] = (cross_r, cross_c)

    for idx in (v_idx, h_idx):
        placed[idx]["origin"] = _origin_from_anchor(placed[idx]["side"], placed[idx]["count"], placed[idx]["anchor"])
        placed[idx]["bbox"] = _bbox_from_origin(placed[idx]["origin"])

    used_vertical_cols = {cross_c} if _orientation(placed[v_idx]["side"]) == "vertical" else set()
    used_horizontal_rows = {cross_r} if _orientation(placed[h_idx]["side"]) == "horizontal" else set()

    for idx, spec in enumerate(placed):
        if idx in {v_idx, h_idx}:
            continue
        candidates = _candidate_anchors(spec, shape)
        chosen = None
        for anchor in candidates:
            if _orientation(spec["side"]) == "vertical" and anchor[1] in used_vertical_cols:
                continue
            if _orientation(spec["side"]) == "horizontal" and anchor[0] in used_horizontal_rows:
                continue
            origin = _origin_from_anchor(spec["side"], spec["count"], anchor)
            bbox = _bbox_from_origin(origin)
            if any(
                _bboxes_too_close(bbox, other["bbox"])
                for j, other in enumerate(placed)
                if j != idx and "bbox" in other
            ):
                continue
            chosen = anchor
            spec["anchor"] = anchor
            spec["origin"] = origin
            spec["bbox"] = bbox
            break
        if chosen is None:
            return None
        if _orientation(spec["side"]) == "vertical":
            used_vertical_cols.add(chosen[1])
        else:
            used_horizontal_rows.add(chosen[0])

    return placed


def _render_input(shape: tuple[int, int], specs: list[dict], bg: int = 0) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for spec in specs:
        r0, c0 = spec["origin"]
        paste(grid, spec["patch"], r0, c0, transparent=bg)
    return grid


def _semantic_checks(inp: np.ndarray, specs: list[dict], bg: int = 0) -> bool:
    out = _render_objects(inp.shape, specs, bg)
    if not np.array_equal(main(inp), out):
        return False

    objects = _extract_objects(inp)
    if len(objects) != len(specs):
        return False

    counts = [obj["count"] for obj in objects]
    if len(set(counts)) != len(counts):
        return False

    by_color = {int(spec["color"]): spec for spec in specs}
    for obj in objects:
        spec = by_color.get(int(obj["color"]))
        if spec is None:
            return False
        if obj["count"] != spec["count"] or obj["side"] != spec["side"]:
            return False

    footprints = {}
    for spec in specs:
        footprints[spec["color"]] = _footprint_mask(
            inp.shape,
            spec["bbox"],
            spec["patch"],
            spec["side"],
            spec["count"],
            bg,
        )

    overlaps = []
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            a = specs[i]
            b = specs[j]
            inter = footprints[a["color"]] & footprints[b["color"]]
            if inter.any():
                overlaps.append((a, b, inter))
    if not overlaps:
        return False

    if any(not np.any(out == spec["color"]) for spec in specs):
        return False

    for spec in specs:
        others = np.zeros(inp.shape, dtype=bool)
        for other in specs:
            if other["color"] == spec["color"]:
                continue
            others |= footprints[other["color"]]
        if not np.any(footprints[spec["color"]] & ~others):
            return False

    visible_overlap = False
    for a, b, inter in overlaps:
        top = a if a["count"] < b["count"] else b
        if np.any(out[inter] == int(top["color"])):
            visible_overlap = True
            break
    if not visible_overlap:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_GRID_RANGE)
    w = random.randint(*GEN_GRID_RANGE)
    specs = _build_object_specs()
    placed = _place_specs((h, w), specs)
    if placed is None:
        return None
    inp = _render_input((h, w), placed)
    if not _semantic_checks(inp, placed):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid notched-tiling ARC input")
