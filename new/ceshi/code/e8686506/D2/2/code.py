# Variant of e8686506
# Dimension: D2 — 源对象：碎片的类型与结构 (src)
# Change: 散落的单色非背景像素碎片和精灵片段，每个碎片为单一颜色的连通区域... → 碎片为中空框架形
#
# Design analysis for the D2 variant
# ----------------------------------
# spec:
#   core_change:
#     original: output was a dense rectangle assembled from filled monochrome fragments.
#     variant: every source fragment is a hollow rectangular frame; the assembled output
#       is still a left-right symmetric rectangle by bbox, but its interior keeps visible
#       background holes.
#   concretization:
#     output uses one unique outer frame whose bbox is the full output size, plus 2-5
#       smaller hollow frames placed in a fixed symmetric lattice.
#     all pieces are 4-connected single-color border loops with bbox >= 3x3.
#     the input keeps the outer frame intact as a grouped anchor and scatters every other
#       hollow frame outside that bbox.
#   behavior_trace:
#     original small case:
#       total colored area = 25 -> infer 5x5 dense board -> pack all solid pieces.
#     variant small case:
#       the largest hollow frame has bbox 9x7 -> infer output size directly from that
#       frame, not from colored area. Two smaller hollow frames are then reinserted into
#       template slots, leaving their holes as background in the output.
#     difference:
#       total ink area is now much smaller than bbox area, so dense-packing/factorization
#       is invalid; shape roles come from frame geometry instead.
#   complexity_calibration:
#     using only one hollow ring would be visually obvious but too trivial.
#     this variant uses three layout families (9x7, 11x9, 13x11) with 3/5/6 total pieces.
#     roles are distinguished by repeated structural cues:
#       - unique largest outer frame
#       - at most one mirrored side-pair shape
#       - centered single frames with distinct sizes
#     this keeps the rule readable while preventing role collisions.
#   original_main_assumption_audit:
#     - extracting components by color still works: a hollow ring is one 4-connected
#       component, so `_piece_sig` remains valid.
#     - total_area -> factor-pair dimension inference no longer holds because holes make
#       bbox area larger than colored area.
#     - dense board initialization no longer holds; the solver must preserve background
#       cells inside the rectangle.
#     - singleton/bar/line anchors are removed because they are not hollow-frame objects
#       and would dilute the D2 change.
#     - the original "1-2 frame parts under 8-connectivity" logic is replaced by a
#       stronger guarantee: there is exactly one largest hollow rectangular frame.
#   placement_legality:
#     generator uses prevalidated templates, draws only non-overlapping rings, and blocks
#       whole bounding boxes while scattering so another piece cannot hide inside a hole.
#     solver assumes every non-outer frame sits strictly inside the outer frame and that
#       the set of remaining frame shapes matches exactly one template for that output size.
#   visual_observability:
#     the variant is visible without relying on color ids: every scattered object is a ring
#       with a hole, and the output keeps those holes. Compared to the original dense fill,
#       the hollow nature is obvious in 2-3 examples.
#   cleanliness:
#     there is no overdraw in the output and no input nesting of scattered pieces, so each
#       frame remains easy to isolate by eye.
#   color_inference:
#     colors are preserved piece-wise; no hidden color remapping rule is introduced.

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _ring_coords(h: int, w: int) -> tuple[tuple[int, int], ...]:
    coords = []
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                coords.append((r, c))
    return tuple(coords)


def _is_rect_frame_sig(sig: tuple[int, int, tuple[tuple[int, int], ...]]) -> bool:
    h, w, coords = sig
    return h >= 3 and w >= 3 and coords == _ring_coords(h, w)


def _frame_patch(h: int, w: int, col: int, bg: int) -> np.ndarray:
    patch = np.full((h, w), bg, dtype=int)
    patch[0, :] = col
    patch[h - 1, :] = col
    patch[:, 0] = col
    patch[:, w - 1] = col
    return patch


def _paint_frame(out: np.ndarray, r0: int, c0: int, h: int, w: int, col: int) -> None:
    out[r0, c0 : c0 + w] = col
    out[r0 + h - 1, c0 : c0 + w] = col
    out[r0 : r0 + h, c0] = col
    out[r0 : r0 + h, c0 + w - 1] = col


def _mirror_c0(W: int, w: int, c0: int) -> int:
    return W - w - c0


_TEMPLATE_LIBRARY = (
    {
        "dims": (9, 7),
        "roles": (
            {"shape": (3, 5), "count": 1, "r": 1, "c": 1},
            {"shape": (3, 3), "count": 1, "r": 5, "c": 2},
        ),
    },
    {
        "dims": (11, 9),
        "roles": (
            {"shape": (3, 5), "count": 1, "r": 1, "c": 2},
            {"shape": (3, 3), "count": 2, "r": 4, "c": 1},
            {"shape": (3, 7), "count": 1, "r": 7, "c": 1},
        ),
    },
    {
        "dims": (13, 11),
        "roles": (
            {"shape": (3, 7), "count": 1, "r": 1, "c": 2},
            {"shape": (5, 3), "count": 2, "r": 4, "c": 1},
            {"shape": (3, 3), "count": 1, "r": 5, "c": 4},
            {"shape": (3, 5), "count": 1, "r": 9, "c": 3},
        ),
    },
)
_TEMPLATE_BY_DIMS = {tpl["dims"]: tpl for tpl in _TEMPLATE_LIBRARY}
_TEMPLATE_WEIGHTS = [tpl["dims"][0] * tpl["dims"][1] for tpl in _TEMPLATE_LIBRARY]


def _extract_frame_pieces(g: np.ndarray, bg: int) -> list[dict]:
    pieces = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=4):
            h, w, coords = _piece_sig(pts)
            assert _is_rect_frame_sig((h, w, coords)), "every non-bg component must be a hollow rectangular frame"
            pieces.append(
                {
                    "col": int(col),
                    "h": int(h),
                    "w": int(w),
                    "coords": coords,
                    "bbox_area": int(h * w),
                }
            )
    assert pieces, "grid must contain at least one hollow frame piece"
    return pieces


def _find_outer_frame(pieces: list[dict]) -> dict:
    max_box = max(int(p["bbox_area"]) for p in pieces)
    outer = [p for p in pieces if int(p["bbox_area"]) == max_box]
    assert len(outer) == 1, "the unique largest hollow frame defines the output bbox"
    return outer[0]


def _group_inner_types(pieces: list[dict], outer: dict) -> list[dict]:
    ctr = Counter(
        (int(p["col"]), int(p["h"]), int(p["w"]), tuple(p["coords"]))
        for p in pieces
        if p is not outer
    )
    types = []
    for (col, h, w, coords), cnt in sorted(ctr.items(), key=lambda kv: (-kv[0][1] * kv[0][2], kv[0][0], kv[1])):
        assert cnt in (1, 2), "inner hollow frames must appear either once or as one mirrored pair"
        types.append({"col": int(col), "h": int(h), "w": int(w), "coords": coords, "cnt": int(cnt)})
    return types


def _match_template(H: int, W: int, types: list[dict]) -> tuple[dict, list[tuple[dict, dict]]]:
    tpl = _TEMPLATE_BY_DIMS.get((H, W))
    assert tpl is not None, "unsupported hollow-frame layout dimensions"
    remaining = types[:]
    assignments: list[tuple[dict, dict]] = []
    for role in tpl["roles"]:
        h, w = role["shape"]
        matches = [t for t in remaining if int(t["h"]) == h and int(t["w"]) == w and int(t["cnt"]) == int(role["count"])]
        assert len(matches) == 1, "remaining hollow frame shapes do not match the expected template roles"
        assignments.append((role, matches[0]))
        remaining.remove(matches[0])
    assert not remaining, "unused hollow frame types remain after template matching"
    return tpl, assignments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered hollow frames into a left-right symmetric rectangular lattice.

    Input structure:
    - bg is the most common color; every non-bg component is a single-color rectangular ring
      with a background hole in its middle.
    - The largest ring is the outer border frame; smaller rings are scattered elsewhere and
      appear either once on the center column or twice as a mirrored side pair.

    Rule:
    - Detect the unique largest hollow frame and use its bounding box as the output size.
    - Match the remaining ring sizes to the layout template for that output size.
    - Place each hollow frame into its template slot, preserving each piece's color and the
      background holes inside the assembled rectangle.

    Key insight:
    - In this variant the holes are part of the signal: the largest ring fixes the rectangle,
      and the size/orientation pattern of the smaller hollow frames uniquely identifies the
      symmetric slots they occupy.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    pieces = _extract_frame_pieces(g, bg)
    outer = _find_outer_frame(pieces)
    H, W = int(outer["h"]), int(outer["w"])
    types = _group_inner_types(pieces, outer)
    _tpl, assignments = _match_template(H, W, types)

    out = np.full((H, W), bg, dtype=int)
    _paint_frame(out, 0, 0, H, W, int(outer["col"]))

    for role, piece_type in assignments:
        h, w = int(piece_type["h"]), int(piece_type["w"])
        col = int(piece_type["col"])
        if int(role["count"]) == 1:
            _paint_frame(out, int(role["r"]), int(role["c"]), h, w, col)
        else:
            left_c = int(role["c"])
            right_c = _mirror_c0(W, w, left_c)
            _paint_frame(out, int(role["r"]), left_c, h, w, col)
            _paint_frame(out, int(role["r"]), right_c, h, w, col)

    assert np.array_equal(out, out[:, ::-1]), "assembled hollow-frame output must be LR-symmetric"
    return out.astype(int)


# --------------- generator (hollow-frame lattice, output-first) ---------------

_CANVAS_MIN = 18
_CANVAS_MAX = 22
_SCATTER_PADDING = 1
_GEN_MAX_RETRIES = 300


def _build_output() -> tuple[np.ndarray, int]:
    tpl = random.choices(_TEMPLATE_LIBRARY, weights=_TEMPLATE_WEIGHTS, k=1)[0]
    H, W = tpl["dims"]

    bg = random.randint(0, 9)
    needed_colors = 1 + len(tpl["roles"])
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    assert len(palette) >= needed_colors

    out = np.full((H, W), bg, dtype=int)
    _paint_frame(out, 0, 0, H, W, palette[0])

    for i, role in enumerate(tpl["roles"], start=1):
        col = palette[i]
        h, w = role["shape"]
        if int(role["count"]) == 1:
            _paint_frame(out, int(role["r"]), int(role["c"]), h, w, col)
        else:
            left_c = int(role["c"])
            right_c = _mirror_c0(W, w, left_c)
            _paint_frame(out, int(role["r"]), left_c, h, w, col)
            _paint_frame(out, int(role["r"]), right_c, h, w, col)

    return out, bg


def _extract_patches(out: np.ndarray, bg: int) -> list[np.ndarray]:
    patches = []
    for col in sorted(set(map(int, np.unique(out))) - {bg}):
        for pts in components_mask(out == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in pts:
                patch[int(r) - r0, int(c) - c0] = col
            patches.append(patch)
    patches.sort(key=lambda p: (-(p.shape[0] * p.shape[1]), -(p != bg).sum()))
    return patches


def _scatter(out: np.ndarray, bg: int) -> np.ndarray:
    patches = _extract_patches(out, bg)
    areas = [p.shape[0] * p.shape[1] for p in patches]
    outer_idx = int(np.argmax(areas))
    outer_patch = patches[outer_idx]
    inner_patches = [p for i, p in enumerate(patches) if i != outer_idx]

    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    occ = np.zeros((rows, cols), dtype=int)

    oh, ow = outer_patch.shape
    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("outer frame cannot fit on the scatter canvas")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    occ[off_r : off_r + oh, off_c : off_c + ow] = 1

    for patch in inner_patches:
        blocker = np.ones(patch.shape, dtype=int)
        r, c = random_free_location(
            occ,
            blocker,
            transparent=0,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(occ, blocker, r, c)
        paste(canvas, patch, r, c, transparent=bg)

    paste(canvas, outer_patch, off_r, off_c, transparent=bg)
    return canvas


def _component_hole_count(grid: np.ndarray, bg: int) -> int:
    holes = 0
    for col in sorted(set(map(int, np.unique(grid))) - {bg}):
        for pts in components_mask(grid == col, connectivity=4):
            h, w, coords = _piece_sig(pts)
            if int(h * w) > len(coords):
                holes += 1
    return holes


def _gen_attempt() -> np.ndarray | None:
    out, bg = _build_output()
    if not np.array_equal(out, out[:, ::-1]):
        return None
    if _component_hole_count(out, bg) == 0:
        return None
    try:
        inp = _scatter(out, bg)
        got = main(inp)
    except Exception:
        return None
    if np.array_equal(got, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hollow-frame e8686506 input")
