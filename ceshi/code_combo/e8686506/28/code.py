# Combo variant of e8686506 (index 28)
# combo_spec: D2/2+D6/0
# Name: 中空框架碎片+拼合后颜色统一
#
# Design analysis for the combo variant
# -------------------------------------
# spec:
#   core_change:
#     original: scattered solid fragments were packed into a dense LR-symmetric rectangle
#       and kept their own colors after assembly.
#     combo: every fragment is instead a hollow rectangular frame, and after reassembly
#       every foreground frame is recolored to the OUTER frame's color so the output shows
#       only one-color perforated structure.
#   concretization:
#     - the input contains 3 layout families with one unique largest hollow frame plus
#       2-4 inner hollow frames.
#     - inner roles are determined by a small template library keyed by the outer frame's
#       bbox `(H, W)`, exactly as in a puzzle board.
#     - the hidden assembled layout is multi-color: outer frame color + one distinct color
#       per inner role. The visible target is obtained by collapsing all non-bg cells to the
#       outer frame color while preserving holes.
#   behavior_trace:
#     original 8x5 example:
#       total solid area fixed the output dims, then solid pieces were packed and colors
#       stayed attached to each component.
#     combo 11x9 example:
#       the largest hollow ring immediately fixes the output dims; smaller hollow rings are
#       matched to template slots by size/count; then the finished multi-color lattice is
#       recolored to the outer ring color.
#     difference:
#       bbox geometry, not ink area, now determines the board; holes remain visible; color
#       becomes a temporary cue used only during recognition and disappears in the answer.
#   complexity_calibration:
#     - using only one ring would make the recolor step trivial, so every sample has one
#       outer ring plus at least two inner roles.
#     - only three board families are used: this keeps the rule learnable from 2-3 examples,
#       while the repeated ring sizes (single center vs mirrored side-pair roles) provide
#       enough structure to avoid ambiguous assemblies.
#     - the smallest family still has three foreground colors in the hidden assembled state,
#       so the color-collapse behavior is visually observable.
#   original_main_assumption_audit:
#     - `_piece_sig` still works because each hollow frame is one 4-connected component.
#     - total colored area can no longer infer output size; holes reduce ink area, so the
#       solver must use the largest frame bbox instead.
#     - the original dense fill assumption is invalid; output must preserve background holes.
#     - singleton/bar/line anchors are removed because they conflict with the "all pieces are
#       hollow frames" rule and are no longer needed once the outer ring anchors the board.
#   placement_legality:
#     - templates were chosen so every inner frame lies strictly inside the outer frame and
#       no two frames overlap or 4-touch.
#     - scattering blocks the outer frame bbox before placing inner pieces, so no patch can
#       hide inside another frame's hole or cross the anchor region.
#   visual_observability:
#     - in the input, every object is visibly a hollow ring and colors distinguish roles.
#     - in the output, the same holes remain but all foreground colors collapse to one; this
#       makes the D6 recolor visible without requiring any invisible palette ordering rule.
#   cleanliness:
#     - assembled frames never overdraw; each hole remains a plain background rectangle.
#     - the scatter canvas keeps at least one-cell Chebyshev gaps between pieces.
#   color_inference:
#     - the unified output color is not arbitrary: it is always the outer frame's input color.
#       This gives a stable, human-visible anchor for the recolor rule across examples.
#   new_main_assumption_audit:
#     - solver assumes every non-bg component is a rectangular hollow ring with bbox >= 3x3.
#       generator enforces this by drawing only border loops.
#     - solver assumes there is exactly one largest ring by bbox area; generator uses a unique
#       outer frame size per family, larger than every inner role.
#     - solver assumes the remaining `(shape, count)` multiset matches one template for that
#       `(H, W)`; generator draws only prevalidated templates from `_TEMPLATE_LIBRARY`.

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
    assert len(outer) == 1, "the unique largest hollow frame defines the output bbox and output color"
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


def _assemble_colored_layout(H: int, W: int, bg: int, outer: dict, assignments: list[tuple[dict, dict]]) -> np.ndarray:
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
    return out


def _collapse_to_outer_color(colored_layout: np.ndarray, bg: int, outer_col: int) -> np.ndarray:
    out = np.array(colored_layout, copy=True)
    out[out != bg] = int(outer_col)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered hollow frames into a left-right symmetric perforated rectangle,
    then collapse every frame to the outer border's color.

    Input structure:
    - `bg` is the most common color; every non-bg component is a single-color rectangular
      ring with a background hole.
    - The unique largest ring is the outer border. Smaller rings are scattered elsewhere
      and appear either once on the center line or twice as a mirrored side-pair.

    Rule:
    - Use the largest hollow frame's bbox as the output size and its color as the final
      recolor target.
    - Match the remaining ring sizes/counts to the template for that bbox and place them
      into their fixed slots, preserving holes.
    - After the full hollow lattice is reconstructed, recolor every non-background cell
      to the outer frame's color.

    Key insight:
    - The colors are only temporary assembly cues. Geometry tells you where the rings go,
      but the finished answer discards per-piece color and keeps only the hollow outline
      structure, all painted in the anchor color of the largest ring.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    pieces = _extract_frame_pieces(g, bg)
    outer = _find_outer_frame(pieces)
    H, W = int(outer["h"]), int(outer["w"])
    types = _group_inner_types(pieces, outer)
    _tpl, assignments = _match_template(H, W, types)

    colored_layout = _assemble_colored_layout(H, W, bg, outer, assignments)
    out = _collapse_to_outer_color(colored_layout, bg, int(outer["col"]))

    assert np.array_equal(out, out[:, ::-1]), "assembled hollow-frame output must be LR-symmetric"
    return out.astype(int)


# --------------- generator (hidden colored layout -> visible single-color target) ---------------

_CANVAS_MIN = 18
_CANVAS_MAX = 22
_SCATTER_PADDING = 1
_GEN_MAX_RETRIES = 300


def _build_layouts() -> tuple[np.ndarray, np.ndarray, int, int]:
    tpl = random.choices(_TEMPLATE_LIBRARY, weights=_TEMPLATE_WEIGHTS, k=1)[0]
    H, W = tpl["dims"]

    bg = random.randint(0, 9)
    needed_colors = 1 + len(tpl["roles"])
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    assert len(palette) >= needed_colors

    colored = np.full((H, W), bg, dtype=int)
    outer_col = int(palette[0])
    _paint_frame(colored, 0, 0, H, W, outer_col)

    for i, role in enumerate(tpl["roles"], start=1):
        col = int(palette[i])
        h, w = role["shape"]
        if int(role["count"]) == 1:
            _paint_frame(colored, int(role["r"]), int(role["c"]), h, w, col)
        else:
            left_c = int(role["c"])
            right_c = _mirror_c0(W, w, left_c)
            _paint_frame(colored, int(role["r"]), left_c, h, w, col)
            _paint_frame(colored, int(role["r"]), right_c, h, w, col)

    target = _collapse_to_outer_color(colored, bg, outer_col)
    return colored, target, bg, outer_col


def _extract_patches(colored_layout: np.ndarray, bg: int) -> list[np.ndarray]:
    patches = []
    for col in sorted(set(map(int, np.unique(colored_layout))) - {bg}):
        for pts in components_mask(colored_layout == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in pts:
                patch[int(r) - r0, int(c) - c0] = col
            patches.append(patch)
    patches.sort(key=lambda p: (-(p.shape[0] * p.shape[1]), -(p != bg).sum()))
    return patches


def _scatter(colored_layout: np.ndarray, bg: int) -> np.ndarray:
    patches = _extract_patches(colored_layout, bg)
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


def _hole_count(grid: np.ndarray, bg: int) -> int:
    holes = 0
    for col in sorted(set(map(int, np.unique(grid))) - {bg}):
        for pts in components_mask(grid == col, connectivity=4):
            h, w, coords = _piece_sig(pts)
            if int(h * w) > len(coords):
                holes += 1
    return holes


def _foreground_colors(grid: np.ndarray, bg: int) -> set[int]:
    return set(map(int, np.unique(grid))) - {int(bg)}


def _gen_attempt() -> np.ndarray | None:
    colored_layout, target, bg, outer_col = _build_layouts()
    if not np.array_equal(target, target[:, ::-1]):
        return None
    if len(_foreground_colors(colored_layout, bg)) < 3:
        return None
    if len(_foreground_colors(target, bg)) != 1:
        return None
    if int(next(iter(_foreground_colors(target, bg)))) != int(outer_col):
        return None
    if _hole_count(target, bg) == 0:
        return None
    try:
        inp = _scatter(colored_layout, bg)
        got = main(inp)
    except Exception:
        return None
    if np.array_equal(got, target):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hollow-frame recolor e8686506 input")
