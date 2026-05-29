# Combo variant of e8686506 (index 30)
# combo_spec: D4/1+D15/1
# Name: 双轴对称+对称轴标记

# Design analysis
# yaml:
#   concrete_behavior:
#     output:
#       - The solved rectangle is a dense 7x5 portrait board that is mirrored across both the vertical
#         and horizontal axes, so every non-axis cell belongs to a 4-way reflection orbit.
#       - The output keeps the original "frame + scattered fills" flavor, but the interior now has a
#         central plus-shaped component sitting exactly on both axes, plus mirrored side columns and
#         mirrored top/bottom center caps.
#       - The marker pixels are not copied into the output; they are pure input-side hints.
#     input:
#       - Two singleton pixels of one special color lie one cell above and below the rectangle center
#         column, marking the endpoints of the vertical symmetry axis.
#       - Two singleton pixels of another special color lie one cell left and right of the rectangle
#         center row, marking the endpoints of the horizontal symmetry axis.
#       - The frame cells still stay grouped together inside the input, while the non-frame pieces are
#         scattered away from that bbox exactly as independent jigsaw fragments.
#     trace_example:
#       - Original logic inferred output dimensions only from total area and only enforced left/right
#         reflection while packing pieces.
#       - In this fused variant the solver first strips the four marker pixels, reads height and width
#         from the marker distances, and fixes the exact center row/column before reasoning about pieces.
#       - The old single-axis logic would treat a top-side pair and its bottom-side counterpart as
#         interchangeable; the new dual-axis rule plus explicit axis markers removes that ambiguity.
#   complexity_calibration:
#     - The rectangle stays narrow and portrait like the source task, so the family still looks like
#       e8686506 instead of an unrelated synthetic puzzle.
#     - Only five output roles are used repeatedly: corner singletons, frame rails, a central plus,
#       four side pairs, and two center-cap pairs. This is varied enough to be visible but still easy
#       to parse from 2-4 ARC examples.
#     - Marker geometry is deliberately rigid: always two orthogonal singleton pairs exactly one cell
#       outside the frame bbox. That keeps the hint obvious rather than noisy.
#   implicit_assumption_review:
#     - Old main() assumed every non-background color belonged to a real puzzle piece. That breaks once
#       marker pixels are introduced, so the new solver must identify and erase marker colors before
#       counting components or computing area.
#     - Old dimension inference relied only on factorizing total piece area. Here the key assumption is
#       stronger and explicit: the two marker pairs must bracket the rectangle by one cell, and their
#       distances must therefore equal H+1 and W+1.
#     - Old symmetry checks only needed left/right consistency. The new construction must guarantee both
#       row and column midlines are respected, so the output template and validation both enforce LR and
#       UD symmetry.
#     - The source heuristics for horizontal edge bars and paired side lines are unsafe here because the
#       new fixed pattern intentionally contains center-column cap pairs. Those old heuristics are removed.
#   placement_legality:
#     - The generator reserves the full frame bbox plus the four marker cells before scattering any other
#       patches, so no overlap with markers or frame can occur.
#       Non-frame patches still use the same free-location helper with Chebyshev padding.
#     - All output components have area at most 5, stay inside a 7x5 bbox, and use only 4-connected
#       shapes for scatterable pieces.
#   observability:
#     - Human solvers can see the combo directly: the output examples show four-quadrant symmetry, and
#       the inputs show a colored cross of four isolated pixels hugging the frame bbox.
#     - Because marker colors are chosen from colors not used in the output, they read as annotations
#       rather than ordinary fragments, making the "axis endpoint" interpretation visually accessible.
#   cleanliness:
#     - The visible layers are separated: marker pixels sit outside the frame, the frame remains grouped,
#       and scattered fills stay away from both. No region has multiple symbolic roles piled onto it.
#   inferability:
#     - Color remapping is purely role-based. The same structural roles recur in every sample, so the
#       learner can infer that marker colors indicate axes by geometry alone rather than by fixed ids.

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


_OUT_H = 7
_OUT_W = 5
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1
_GEN_MAX_RETRIES = 300

_PLUS_COORDS = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))
_VERT2_COORDS = ((0, 0), (1, 0))


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _frame_cells(H: int, W: int) -> set[tuple[int, int]]:
    assert (H, W) == (_OUT_H, _OUT_W), "generator and solver share a single fixed output size"
    left = {(0, 1), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 1)}
    return left | {(r, W - 1 - c) for r, c in left}


def _side_pair_cells() -> set[tuple[int, int]]:
    return {(1, 1), (2, 1), (1, 3), (2, 3), (4, 1), (5, 1), (4, 3), (5, 3)}


def _cap_pair_cells() -> set[tuple[int, int]]:
    return {(0, 2), (1, 2), (5, 2), (6, 2)}


def _plus_cells() -> set[tuple[int, int]]:
    return {(2, 2), (3, 1), (3, 2), (3, 3), (4, 2)}


def _is_dual_symmetric(grid: np.ndarray) -> bool:
    return np.array_equal(grid, grid[:, ::-1]) and np.array_equal(grid, grid[::-1, :])


def _parse_axis_markers(g: np.ndarray, bg: int) -> dict:
    vertical = None
    horizontal = None
    marker_colors = set()
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        comps = components_mask(g == col, connectivity=4)
        if len(comps) != 2 or not all(len(pts) == 1 for pts in comps):
            continue
        pts = sorted((int(pts[0, 0]), int(pts[0, 1])) for pts in comps)
        (r0, c0), (r1, c1) = pts
        if c0 == c1 and r0 != r1:
            assert vertical is None, "expected a unique vertical-axis marker color"
            vertical = {"color": col, "top": (r0, c0), "bottom": (r1, c1)}
            marker_colors.add(col)
        elif r0 == r1 and c0 != c1:
            assert horizontal is None, "expected a unique horizontal-axis marker color"
            horizontal = {"color": col, "left": (r0, c0), "right": (r1, c1)}
            marker_colors.add(col)
    assert vertical is not None and horizontal is not None, "expected both axis marker pairs"

    vt_r, vt_c = vertical["top"]
    vb_r, vb_c = vertical["bottom"]
    hl_r, hl_c = horizontal["left"]
    hr_r, hr_c = horizontal["right"]

    assert vt_c == vb_c, "vertical markers must share a column"
    assert hl_r == hr_r, "horizontal markers must share a row"
    H = vb_r - vt_r - 1
    W = hr_c - hl_c - 1
    assert H > 0 and W > 0, "marker distances must define a positive rectangle"
    assert (vt_r + vb_r) % 2 == 0 and (hl_c + hr_c) % 2 == 0, "axis midpoints must land on cells"
    center_r = (vt_r + vb_r) // 2
    center_c = (hl_c + hr_c) // 2
    assert center_r == hl_r and center_c == vt_c, "marker pairs must agree on the same center"

    clean = g.copy()
    for col in marker_colors:
        clean[clean == col] = bg

    return {
        "clean": clean,
        "H": int(H),
        "W": int(W),
        "center_r": int(center_r),
        "center_c": int(center_c),
        "bbox": (vt_r + 1, hl_c + 1, vb_r - 1, hr_c - 1),
        "marker_colors": marker_colors,
    }


def _collect_components(g: np.ndarray, bg: int) -> dict[int, list[np.ndarray]]:
    out: dict[int, list[np.ndarray]] = {}
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        out[col] = components_mask(g == col, connectivity=4)
    return out


def _classify_roles(comps_by_color: dict[int, list[np.ndarray]]) -> dict[str, int]:
    corner_color = None
    plus_color = None
    side_color = None
    cap_color = None

    for col, comps in comps_by_color.items():
        sigs = [_piece_sig(pts) for pts in comps]
        if len(comps) == 4 and all(len(pts) == 1 for pts in comps):
            assert corner_color is None, "corner color must be unique"
            corner_color = col
        elif len(comps) == 1 and sigs[0] == (3, 3, _PLUS_COORDS):
            assert plus_color is None, "central plus color must be unique"
            plus_color = col
        elif len(comps) == 4 and all(sig == (2, 1, _VERT2_COORDS) for sig in sigs):
            assert side_color is None, "side-pair color must be unique"
            side_color = col
        elif len(comps) == 2 and all(sig == (2, 1, _VERT2_COORDS) for sig in sigs):
            assert cap_color is None, "center-cap color must be unique"
            cap_color = col

    assert corner_color is not None, "missing corner-singleton color"
    assert plus_color is not None, "missing center-plus color"
    assert side_color is not None, "missing side-pair color"
    assert cap_color is not None, "missing center-cap color"

    used = {corner_color, plus_color, side_color, cap_color}
    frame_colors = [col for col in comps_by_color if col not in used]
    assert len(frame_colors) == 1, "expected exactly one frame color"
    return {
        "corner": corner_color,
        "plus": plus_color,
        "side": side_color,
        "cap": cap_color,
        "frame": frame_colors[0],
    }


def _validate_frame(g: np.ndarray, frame_color: int, marker_info: dict) -> None:
    H = int(marker_info["H"])
    W = int(marker_info["W"])
    fr0, fc0, fr1, fc1 = marker_info["bbox"]
    comps8 = components_mask(g == frame_color, connectivity=8)
    assert len(comps8) == 2, "frame color must form two 8-connected rail groups"
    union = np.concatenate(comps8)
    assert bbox_coords(union) == (fr0, fc0, fr1, fc1), "frame bbox must match the axis markers"
    rel = {(int(r) - fr0, int(c) - fc0) for r, c in union}
    assert rel == _frame_cells(H, W), "frame bitmap must match the fused template"


def _render_output(H: int, W: int, roles: dict[str, int]) -> np.ndarray:
    out = np.full((H, W), -1, dtype=int)
    for r, c in ((0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)):
        out[r, c] = roles["corner"]
    for r, c in _frame_cells(H, W):
        out[r, c] = roles["frame"]
    for r, c in _plus_cells():
        out[r, c] = roles["plus"]
    for r, c in _side_pair_cells():
        out[r, c] = roles["side"]
    for r, c in _cap_pair_cells():
        out[r, c] = roles["cap"]
    assert np.all(out >= 0), "template must cover the whole rectangle"
    assert _is_dual_symmetric(out), "rendered output must satisfy both symmetry axes"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered fragments into a dense rectangle that is symmetric across both axes.

    Input structure:
    - The dominant color is background; all other colored cells are either puzzle fragments or four special marker pixels.
    - One marker color appears as a vertical singleton pair and another as a horizontal singleton pair; together they sit one cell outside the hidden rectangle and point to its two symmetry axes.
    - The frame still appears as a grouped 8-connected outline inside that marked bbox, while the remaining pieces are scattered elsewhere.

    Rule:
    - First identify and remove the two marker pairs, then read the output height, width, and center from their endpoint geometry.
    - Classify the remaining piece colors by their component signatures, verify that the grouped frame matches the marked bbox, and rebuild the canonical 7x5 arrangement.
    - The reconstructed rectangle must satisfy both left-right and up-down symmetry; marker pixels never appear in the output.

    Key insight:
    - The marker pixels turn the hidden axes into explicit visible guides. Once those axes are fixed, the remaining fragments fall into a small set of structural roles, and the central plus together with the mirrored side/cap pairs enforces full four-quadrant symmetry rather than the source task's single left-right reflection.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    marker_info = _parse_axis_markers(g, bg)
    clean = marker_info["clean"]
    H = int(marker_info["H"])
    W = int(marker_info["W"])
    assert (H, W) == (_OUT_H, _OUT_W), "this fused generator always uses a 7x5 dual-symmetric rectangle"

    comps_by_color = _collect_components(clean, bg)
    total_area = sum(len(pts) for comps in comps_by_color.values() for pts in comps)
    assert total_area == H * W, "marker-implied dimensions must match the total fragment area"

    roles = _classify_roles(comps_by_color)
    _validate_frame(clean, roles["frame"], marker_info)

    out = _render_output(H, W, roles)
    assert _is_dual_symmetric(out), "assembled board must be dual-symmetric"
    return out.astype(int)


def _make_output() -> tuple[np.ndarray, int, set[int], tuple[int, int]]:
    roles = {"corner": 0, "frame": 1, "plus": 2, "side": 3, "cap": 4}
    out = _render_output(_OUT_H, _OUT_W, roles)

    bg = random.randint(0, 9)
    usable = [c for c in range(10) if c != bg]
    random.shuffle(usable)
    output_colors = usable[:5]
    marker_colors = usable[5:7]
    assert len(marker_colors) == 2, "need two colors reserved for axis markers"

    cmap = {i: output_colors[i] for i in range(5)}
    remapped = np.vectorize(lambda x: cmap[int(x)])(out)
    return remapped.astype(int), bg, {cmap[roles["frame"]]}, (marker_colors[0], marker_colors[1])


def _extract_non_frame_patches(out: np.ndarray, bg: int, frame_colors: set[int]) -> list[np.ndarray]:
    patches = []
    for col in sorted(map(int, np.unique(out))):
        if col in frame_colors:
            continue
        for pts in components_mask(out == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in pts:
                patch[int(r) - r0, int(c) - c0] = col
            patches.append(patch)
    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), random.random()))
    return patches


def _scatter(out: np.ndarray, bg: int, frame_colors: set[int], marker_colors: tuple[int, int]) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = out.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("canvas too small for frame + markers")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1
    marker_cells = [
        (off_r - 1, off_c + ow // 2),
        (off_r + oh, off_c + ow // 2),
        (off_r + oh // 2, off_c - 1),
        (off_r + oh // 2, off_c + ow),
    ]
    for r, c in marker_cells:
        canvas[r, c] = -1

    for patch in _extract_non_frame_patches(out, bg, frame_colors):
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    frame_mask = np.zeros((oh, ow), dtype=bool)
    for col in frame_colors:
        frame_mask |= (out == col)
    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    bbox_region[:] = bg
    bbox_region[frame_mask] = out[frame_mask]

    vcol, hcol = marker_colors
    canvas[off_r - 1, off_c + ow // 2] = vcol
    canvas[off_r + oh, off_c + ow // 2] = vcol
    canvas[off_r + oh // 2, off_c - 1] = hcol
    canvas[off_r + oh // 2, off_c + ow] = hcol
    return canvas


def _gen_attempt() -> np.ndarray | None:
    try:
        out, bg, frame_colors, marker_colors = _make_output()
        inp = _scatter(out, bg, frame_colors, marker_colors)
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
    raise RuntimeError("failed to generate a valid e8686506 combo input")
