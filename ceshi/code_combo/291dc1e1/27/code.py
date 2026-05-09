# Combo variant of 291dc1e1 (index 27)
# combo_spec: D3/0+D6/0
# Name: L/T形对象+左对齐堆叠

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace
# - The border logic stays the same: a marked corner, one blue full edge line, and
#   one red full edge line define the ordering axes.
# - Original task objects were solid rectangles, so the old solver could safely
#   require "no background inside bbox" and then center every stacked object.
# - In this combo, each source object is a connected L or T polyomino. Its bbox
#   necessarily contains background cells, so rectangle-specific extraction fails.
# - Example in canonical orientation:
#   band0 contains [L(3x2), T(2x3)], band1 contains [T(4x3), L(2x4)].
#   Original rectangle logic would reject all four because each bbox contains bg.
#   The combo solver instead extracts each connected component into a bg-padded bbox,
#   sorts them by (distance to blue, distance to red, corner tie-breaks), rotates
#   them if blue is vertical, then stacks them with column 0 shared by every bbox.
# - The visible output difference is twofold at once:
#   irregular silhouettes survive the stack, and the stack is left-flush instead of
#   centered, so the right edge becomes jagged.
#
# b) Complexity calibration
# - Only two shape families are used: L and T. This keeps the hypothesis space small.
# - Each sample contains 4-6 objects split across two bands, preserving the old
#   geometric cue while adding enough repetition for a human to compare shapes.
# - Template sizes come from a small catalog with bbox widths/heights in {2,3,4}.
#   This is enough to make left alignment visible after rotation without creating
#   arbitrary free-form clutter.
# - The generator rejects samples unless both families appear and the centered stack
#   would differ from the left-aligned stack.
#
# c) Implicit assumption audit of the original main()
# - Corner detection and blue/red axis inference remain valid because the border
#   arrangement is unchanged.
# - `line_mask` still assumes interior objects do not touch the corner row/column.
#   The generator keeps at least one bg cell between every object bbox and the axes.
# - The original extraction step depended on "bbox has no bg" to identify rectangles.
#   That assumption is invalid here and is replaced by connected-component extraction
#   into bg-padded bboxes.
# - Distance sorting only depends on bbox extents. For L/T polyominoes, min/max row
#   and column of occupied cells are still exactly the bbox edges, so the same sort
#   key remains meaningful.
# - The old placement step used horizontal centering. D6/0 invalidates that: every
#   transformed object must now paste at `out_c = 0`.
#
# d) Placement legality
# - Canonical generation lays objects in two horizontal bands with 2-cell horizontal
#   gaps and a 2-row vertical gap between the bands. Bboxes never overlap.
# - Each object patch is pasted with background transparency, so L/T holes stay empty.
# - Whole-grid rotations/flips preserve non-overlap and keep all objects in bounds.
#
# e) Visual observability
# - The input still clearly advertises the old coordinate-system rule via the two edge
#   lines, but the objects are visibly non-rectangular.
# - The output keeps those irregular silhouettes and makes the shared left edge easy
#   to notice because widths differ after the required rotation branch.
# - Seeing 2-4 examples is enough to infer both "sort by edge distance" and "stack
#   on the left" without relying on hidden metadata.
#
# f) Visual cleanliness
# - Objects are monochrome and separated by explicit gaps, so the only visual elements
#   are the two axes, one corner marker, and the L/T blocks.
# - No minority pixels are added: they would make already-irregular shapes harder to
#   parse and would not contribute to the fused rule.
#
# g) Color inferability
# - Colors are identity-preserving only. There is no recoloring rule.
# - Distinct object colors help track which input object moved where; the rule still
#   depends on geometry and position, not on any color mapping.
#
# 9) New assumptions introduced by the combo solver
# - Every meaningful interior object is one 4-connected component. The generator
#   guarantees this by using connected L/T templates only.
# - Every object is separated from every other object and from the axes by background.
#   The generator enforces fixed gaps.
# - Left alignment should be behaviorally observable. The generator guarantees this by
#   rejecting samples where left-aligned and centered stacking are identical.
# - Every extracted component should classify as L or T for this variant. The generator
#   validates that post-transform, so the solver can safely treat all extracted
#   components as reorderable source objects.


BG = 8
BLUE = 1
RED = 2
MARKER = 0
OBJECT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
BAND_COUNTS = [(2, 2), (2, 3), (3, 2), (3, 3)]

BASE_L_SHAPES = [
    np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
    np.array([[1, 0], [1, 0], [1, 0], [1, 1]], dtype=int),
]
BASE_T_SHAPES = [
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=int),
]


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _shape_variants(base: np.ndarray) -> list[np.ndarray]:
    variants: dict[tuple[tuple[int, ...], ...], np.ndarray] = {}
    for flipped in (False, True):
        src = np.fliplr(base) if flipped else base
        for k in range(4):
            arr = np.rot90(src, k)
            variants[_mask_key(arr)] = arr.astype(bool)
    return [arr.copy() for arr in variants.values()]


def _build_shape_catalog() -> tuple[dict[str, list[np.ndarray]], dict[tuple[tuple[int, ...], ...], str]]:
    catalog = {"L": [], "T": []}
    family_by_key: dict[tuple[tuple[int, ...], ...], str] = {}
    for family, bases in (("L", BASE_L_SHAPES), ("T", BASE_T_SHAPES)):
        seen: dict[tuple[tuple[int, ...], ...], np.ndarray] = {}
        for base in bases:
            for arr in _shape_variants(base):
                key = _mask_key(arr)
                seen[key] = arr.copy()
                prev = family_by_key.get(key)
                if prev is None:
                    family_by_key[key] = family
                else:
                    assert prev == family
        catalog[family] = [arr.copy() for arr in seen.values()]
    return catalog, family_by_key


SHAPE_CATALOG, SHAPE_FAMILY_BY_KEY = _build_shape_catalog()


def _locate_axes(grid: np.ndarray, bg: int) -> tuple[int, int, int, str, str]:
    H, W = grid.shape
    corners = [(0, 0), (0, int(W - 1)), (int(H - 1), 0), (int(H - 1), int(W - 1))]
    corner = None
    for r, c in corners:
        if int(grid[r, c]) == int(bg):
            continue
        nbr = []
        if r == 0:
            nbr.append(int(grid[r + 1, c]))
        if r == H - 1:
            nbr.append(int(grid[r - 1, c]))
        if c == 0:
            nbr.append(int(grid[r, c + 1]))
        if c == W - 1:
            nbr.append(int(grid[r, c - 1]))
        if nbr and all(int(n) != int(grid[r, c]) for n in nbr):
            corner = (int(r), int(c))
            break
    assert corner is not None

    cr, cc = map(int, corner)
    marker = int(grid[cr, cc])
    hstep = 1 if cc == 0 else -1
    vstep = 1 if cr == 0 else -1
    hcol = int(grid[cr, cc + hstep])
    vcol = int(grid[cr + vstep, cc])

    if hcol == BLUE or vcol == BLUE:
        blue_col = BLUE
        red_col = int(vcol if hcol == BLUE else hcol)
    else:
        hlen = 0
        c = int(cc + hstep)
        while 0 <= c < W and int(grid[cr, c]) == int(hcol):
            hlen += 1
            c += int(hstep)
        vlen = 0
        r = int(cr + vstep)
        while 0 <= r < H and int(grid[r, cc]) == int(vcol):
            vlen += 1
            r += int(vstep)
        if vlen >= hlen:
            blue_col = int(vcol)
            red_col = int(hcol)
        else:
            blue_col = int(hcol)
            red_col = int(vcol)

    blue_orient = "h" if int(hcol) == int(blue_col) else "v"
    red_orient = "h" if int(hcol) == int(red_col) else "v"
    return int(cr), int(cc), int(marker), blue_orient, red_orient


def _dist_to_edge(box: tuple[int, int, int, int], orient: str, edge_idx: int) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        col = int(edge_idx)
        if int(c0) > col:
            return int(c0 - col - 1)
        if int(c1) < col:
            return int(col - c1 - 1)
        return 0
    row = int(edge_idx)
    if int(r0) > row:
        return int(r0 - row - 1)
    if int(r1) < row:
        return int(row - r1 - 1)
    return 0


def _along_from_corner(box: tuple[int, int, int, int], orient: str, cr: int, cc: int) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


def _extract_objects(grid: np.ndarray, bg: int, marker: int, cr: int, cc: int) -> list[dict]:
    H, W = grid.shape
    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (grid != int(bg)) & (~line_mask) & (grid != int(marker))
    objects: list[dict] = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = tuple(map(int, bbox_coords(coords)))
        r0, c0, r1, c1 = box
        arr = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
        for r, c in coords:
            arr[int(r - r0), int(c - c0)] = int(grid[int(r), int(c)])
        objects.append({"box": box, "arr": arr})

    assert objects
    return objects


def _order_objects(objects: list[dict], blue_orient: str, red_orient: str, cr: int, cc: int) -> list[dict]:
    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)
    ordered = []
    for obj in objects:
        box = obj["box"]
        key = (
            int(_dist_to_edge(box, blue_orient, blue_edge_idx)),
            int(_dist_to_edge(box, red_orient, red_edge_idx)),
            int(_along_from_corner(box, blue_orient, cr, cc)),
            int(_along_from_corner(box, red_orient, cr, cc)),
        )
        ordered.append({"box": box, "arr": obj["arr"].copy(), "sort_key": key})
    ordered.sort(key=lambda obj: obj["sort_key"])
    return ordered


def _transform_sequence(ordered: list[dict], blue_orient: str) -> list[np.ndarray]:
    seq = []
    for obj in ordered:
        arr = obj["arr"].copy()
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)
    return seq


def _solve(grid: np.ndarray, *, left_align: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    cr, cc, marker, blue_orient, red_orient = _locate_axes(g, bg)
    objects = _extract_objects(g, bg, marker, cr, cc)
    ordered = _order_objects(objects, blue_orient, red_orient, cr, cc)
    seq = _transform_sequence(ordered, blue_orient)

    out_w = int(max(int(arr.shape[1]) for arr in seq))
    out_h = int(sum(int(arr.shape[0]) for arr in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out_r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        out_c = 0 if left_align else int((out_w - w) // 2)
        paste(out, arr, int(out_r), int(out_c), transparent=int(bg))
        out_r += int(h)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder L/T connected components by the original edge-distance rule, rotate
    them when the blue axis is vertical, and stack the resulting bbox crops with
    shared left-edge alignment.
    """
    return _solve(grid, left_align=True)


def _transform(grid: np.ndarray, blue_orient: str, corner: str) -> np.ndarray:
    if blue_orient == "h":
        if corner == "tl":
            return grid
        if corner == "tr":
            return np.fliplr(grid)
        if corner == "bl":
            return np.flipud(grid)
        return np.rot90(grid, 2)

    base = np.rot90(grid, 1)
    if corner == "bl":
        return base
    if corner == "tl":
        return np.flipud(base)
    if corner == "tr":
        return np.rot90(grid, 3)
    return np.flipud(np.rot90(grid, 3))


def _classify_shape(arr: np.ndarray, bg: int) -> str | None:
    mask = arr != int(bg)
    return SHAPE_FAMILY_BY_KEY.get(_mask_key(mask))


def _make_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, int(BG), dtype=int)
    patch[mask.astype(bool)] = int(color)
    return patch


def _sample_families(total: int) -> list[str]:
    for _ in range(200):
        families = ["L", "T"]
        families.extend(str(np.random.choice(["L", "T"])) for _ in range(total - 2))
        np.random.shuffle(families)
        if "L" in families and "T" in families:
            return families
    raise RuntimeError("failed to sample shape families")


def _sample_shape_records(total: int) -> list[dict]:
    for _ in range(400):
        families = _sample_families(total)
        records = []
        heights = []
        widths = []
        for family in families:
            variants = SHAPE_CATALOG[family]
            mask = variants[int(np.random.randint(len(variants)))].copy()
            records.append({"family": family, "mask": mask})
            heights.append(int(mask.shape[0]))
            widths.append(int(mask.shape[1]))
        if len(set(heights)) < 2:
            continue
        if len(set(widths)) < 2:
            continue
        return records
    raise RuntimeError("failed to sample informative L/T shapes")


def _build_canonical() -> np.ndarray:
    n0, n1 = BAND_COUNTS[int(np.random.randint(len(BAND_COUNTS)))]
    total = int(n0 + n1)
    records = _sample_shape_records(total)
    colors = np.random.choice(OBJECT_COLORS, size=total, replace=False)

    top_records = records[:n0]
    bot_records = records[n0:]
    top_max_h = max(int(rec["mask"].shape[0]) for rec in top_records)
    bot_max_h = max(int(rec["mask"].shape[0]) for rec in bot_records)

    top_used = int(sum(int(rec["mask"].shape[1]) for rec in top_records) + 2 * (n0 - 1))
    bot_used = int(sum(int(rec["mask"].shape[1]) for rec in bot_records) + 2 * (n1 - 1))
    trailing_right = int(np.random.randint(1, 3))
    top_row = 2
    bottom_row = int(top_row + top_max_h + 2)
    trailing_bottom = int(np.random.randint(1, 3))

    H = int(bottom_row + bot_max_h + trailing_bottom)
    W = int(2 + max(top_used, bot_used) + trailing_right)
    grid = np.full((H, W), int(BG), dtype=int)
    grid[0, :] = int(BLUE)
    grid[:, 0] = int(RED)
    grid[0, 0] = int(MARKER)

    idx = 0
    for row, band_records in ((top_row, top_records), (bottom_row, bot_records)):
        c = 2
        for rec in band_records:
            patch = _make_patch(rec["mask"], int(colors[idx]))
            paste(grid, patch, int(row), int(c), transparent=int(BG))
            c += int(patch.shape[1]) + 2
            idx += 1

    return grid


def _variant_metrics(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    cr, cc, marker, blue_orient, red_orient = _locate_axes(g, bg)
    objects = _extract_objects(g, bg, marker, cr, cc)
    families = [_classify_shape(obj["arr"], bg) for obj in objects]
    hole_cells = [int(np.sum(obj["arr"] == int(bg))) for obj in objects]
    ordered = _order_objects(objects, blue_orient, red_orient, cr, cc)
    seq = _transform_sequence(ordered, blue_orient)
    out_w = int(max(int(arr.shape[1]) for arr in seq))
    center_offset_sum = int(sum((out_w - int(arr.shape[1])) // 2 for arr in seq))
    widths = [int(arr.shape[1]) for arr in seq]
    heights = [int(arr.shape[0]) for arr in seq]
    return {
        "count": int(len(objects)),
        "families": families,
        "hole_cells": hole_cells,
        "center_offset_sum": center_offset_sum,
        "widths": widths,
        "heights": heights,
        "blue_orient": blue_orient,
    }


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)

    metrics = _variant_metrics(grid)
    if not (4 <= int(metrics["count"]) <= 6):
        return None
    if any(fam not in {"L", "T"} for fam in metrics["families"]):
        return None
    if "L" not in metrics["families"] or "T" not in metrics["families"]:
        return None
    if not all(int(v) > 0 for v in metrics["hole_cells"]):
        return None
    if int(metrics["center_offset_sum"]) <= 0:
        return None

    left = _solve(grid, left_align=True)
    centered = _solve(grid, left_align=False)
    if np.array_equal(left, centered):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 combo D3/0+D6/0")
