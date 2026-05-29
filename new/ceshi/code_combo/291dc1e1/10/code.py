# Combo variant of 291dc1e1 (index 10)
# combo_spec: D4/0+D5/2
# Name: 三band+band依赖旋转

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

# Design analysis
# a) Concrete behavior and trace
#    Canonical input keeps the original "corner marker + blue edge + red edge +
#    interior solid rectangles" vocabulary, but the rectangles now occupy three
#    parallel blue-distance bands instead of two. In canonical orientation the
#    blue line is on row 0, the red line is on col 0, and the three 2-cell-high
#    bands start at rows 2, 5, and 8, so their blue distances are db=1/4/7.
#    Original task on rectangles A/B/C in those rows would sort by (db, dr) and
#    then either rotate all of them together (if blue is vertical) or rotate none.
#    This combo keeps the same sort key but changes transform into a three-way
#    band rule: the nearest band keeps orientation, the middle band rotates 90 CCW,
#    and the farthest band rotates 180. Example: if three equal-width patches sit
#    at db=1,4,7 and dr=1,7,13, original blue-horizontal output would stack
#    [0°,0°,0°]; combo output stacks [0°,90°,180°]. If the whole input is then
#    rotated so blue becomes vertical, original output becomes [90°,90°,90°]
#    while combo still stays [0°,90°,180°]. The geometry cue and the transform
#    cue therefore have to be inferred together.
# b) Complexity calibration
#    Three bands are enough to make the fusion visibly richer than the two-band
#    source task, but still low-arity enough for humans to infer from 2-4 examples.
#    Each band contains 2-3 rectangles so every rotation class appears repeatedly.
#    Middle/far bands use widths >=4 and every patch carries an asymmetric minority
#    motif, ensuring 90-degree and 180-degree rotations are visibly distinct.
# c) Hidden-assumption audit against original main()
#    - Corner detection still assumes the corner marker differs from both edge
#      lines; generator preserves that exact frame structure.
#    - Line masking still assumes the active row and column span the whole grid;
#      generator preserves full-length edge lines.
#    - Component extraction still assumes each object bbox has no background holes;
#      generator only recolors cells inside a filled rectangle.
#    - Original transform assumed a single global rotation state controlled by
#      `blue_orient`. That premise is invalid here, so the solver must first rank
#      rectangles by their blue-distance bands and rotate per band.
#    - Original task often tolerated only two blue-distance values. This combo
#      needs three ranks. The solver maps sorted unique `db` values to band ranks;
#      generator guarantees exactly three ranks, each non-empty.
# d) Placement legality
#    Canonical short dimension is 11: one edge row, one gap, band0, one gap,
#    band1, one gap, band2, one trailing border row. Within each band rectangles
#    start at dr=1 and are separated by 2 background cells along the blue axis, so
#    they never touch each other, the frame lines, or the marker. Global
#    rotations/flips preserve non-overlap.
# e) Visual observability
#    A human sees three clearly separated source strips parallel to the blue line.
#    In the output the near group stays flat, the middle group becomes tall/narrow,
#    and the far group flips upside-down without swapping its overall width/height.
#    Because each class repeats within the same sample, the band->rotation mapping
#    is supported by multiple instances rather than a one-off anomaly.
# f) Visual cleanliness
#    The scene still contains only background, two full guide lines, one corner
#    marker, and disjoint rectangles. No extra symbols are added and no two object
#    types share cells, so the grid remains easy to parse.
# g) Color inference
#    Colors do not encode the rule. Rectangle dominant colors stay in the original
#    palette, and minority pixels only make orientation visible. The crucial cue is
#    spatial band membership, not a latent color mapping.
# h) Quantified fusion behavior used by the generator
#    A generated instance is accepted only if:
#    - it has exactly three distinct blue-distance bands with counts >=2 each,
#    - combo output differs from the original blue-orientation solver, and
#    - combo output also differs from a weaker two-band rule that would rotate all
#      non-nearest bands by only 90 degrees. This prevents silent collapse where
#      the third band or the 180-degree action is ineffective.
# 9) New solver assumptions after the edit, and generator support
#    - `_band_rotation_k` assumes blue-distance ranks are meaningful. Generator
#      enforces exactly three separated bands at fixed canonical db values.
#    - The sort uses `(db, dr, along_blue, along_red)` as before, so rectangles in
#      a band should have distinct `dr` values. Generator preserves left-to-right
#      sequential placement with positive gaps.
#    - The "far band rotates 180" branch assumes far-band shapes are not invariant
#      under 90/180-degree rotations; generator uses asymmetric motifs and validates
#      `main(grid) != two_band_output`.


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
CANON_BAND_ROWS = (2, 5, 8)
CANON_H = 11


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    corners = [(0, 0), (0, int(W - 1)), (int(H - 1), 0), (int(H - 1), int(W - 1))]
    corner = None
    for r, c in corners:
        if int(g[r, c]) == int(bg):
            continue
        nbr = []
        if r == 0:
            nbr.append(int(g[r + 1, c]))
        if r == H - 1:
            nbr.append(int(g[r - 1, c]))
        if c == 0:
            nbr.append(int(g[r, c + 1]))
        if c == W - 1:
            nbr.append(int(g[r, c - 1]))
        if nbr and all(int(n) != int(g[r, c]) for n in nbr):
            corner = (int(r), int(c))
            break
    assert corner is not None

    cr, cc = map(int, corner)
    marker = int(g[cr, cc])

    hstep = 1 if cc == 0 else -1
    vstep = 1 if cr == 0 else -1
    hcol = int(g[cr, cc + hstep])
    vcol = int(g[cr + vstep, cc])

    if hcol == BLUE or vcol == BLUE:
        blue_col = BLUE
        red_col = int(vcol if hcol == BLUE else hcol)
    else:
        hlen = 0
        c = int(cc + hstep)
        while 0 <= c < W and int(g[cr, c]) == int(hcol):
            hlen += 1
            c += int(hstep)
        vlen = 0
        r = int(cr + vstep)
        while 0 <= r < H and int(g[r, cc]) == int(vcol):
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

    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = bbox_coords(coords)
        sub = crop_bbox(g, box)
        if bool((sub == int(bg)).any()):
            continue
        rects.append({"box": tuple(map(int, box)), "arr": sub.copy()})

    assert rects

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    for rect in rects:
        box = tuple(map(int, rect["box"]))
        rect["db"] = int(_dist_to_edge(box, blue_orient, blue_edge_idx))
        rect["dr"] = int(_dist_to_edge(box, red_orient, red_edge_idx))

    rects.sort(
        key=lambda rect: (
            int(rect["db"]),
            int(rect["dr"]),
            _along_from_corner(tuple(map(int, rect["box"])), blue_orient, cr, cc),
            _along_from_corner(tuple(map(int, rect["box"])), red_orient, cr, cc),
        )
    )

    return {
        "grid": g,
        "bg": bg,
        "corner_r": int(cr),
        "corner_c": int(cc),
        "blue_orient": str(blue_orient),
        "red_orient": str(red_orient),
        "rects": rects,
    }


def _dist_to_edge(box: tuple[int, int, int, int], orient: str, edge_idx: int) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        if int(c0) > int(edge_idx):
            return int(c0 - edge_idx - 1)
        if int(c1) < int(edge_idx):
            return int(edge_idx - c1 - 1)
        return 0
    if int(r0) > int(edge_idx):
        return int(r0 - edge_idx - 1)
    if int(r1) < int(edge_idx):
        return int(edge_idx - r1 - 1)
    return 0


def _along_from_corner(
    box: tuple[int, int, int, int], orient: str, corner_r: int, corner_c: int
) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        return int(r0 if int(corner_r) == 0 else -r1)
    return int(c0 if int(corner_c) == 0 else -c1)


def _band_rotation_k(db: int, db_values: list[int], rotate_mode: str, blue_orient: str) -> int:
    if rotate_mode == "original":
        return 1 if blue_orient == "v" else 0

    band_rank = int(db_values.index(int(db))) if int(db) in db_values else len(db_values) - 1
    if rotate_mode == "two_band":
        return 0 if band_rank == 0 else 1
    if rotate_mode == "fusion":
        return min(int(band_rank), 2)
    raise ValueError(f"unknown rotate_mode: {rotate_mode}")


def _render_output(rects: list[dict], bg: int, blue_orient: str, rotate_mode: str) -> np.ndarray:
    db_values = sorted({int(rect["db"]) for rect in rects})
    seq = []
    for rect in rects:
        arr = rect["arr"]
        k = _band_rotation_k(int(rect["db"]), db_values, rotate_mode, blue_orient)
        if k:
            arr = np.rot90(arr, k=k)
        seq.append(arr)

    out_w = int(max(int(arr.shape[1]) for arr in seq))
    out_h = int(sum(int(arr.shape[0]) for arr in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out_r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        out_c = int((out_w - w) // 2)
        paste(out, arr, int(out_r), int(out_c))
        out_r += int(h)
    return out


def _solve(grid: np.ndarray, rotate_mode: str) -> np.ndarray:
    scene = _parse_scene(grid)
    return _render_output(
        scene["rects"],
        int(scene["bg"]),
        str(scene["blue_orient"]),
        rotate_mode,
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Sort rectangles by (distance to blue edge, distance to red edge).
    - Rank the blue-distance bands from nearest to farthest.
    - Rotate nearest-band rectangles 0 degrees, middle-band rectangles 90 degrees
      CCW, and farthest-band rectangles 180 degrees.
    - Stack the transformed rectangles top-to-bottom, centered horizontally.
    """
    return _solve(grid, rotate_mode="fusion")


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


def _band_used(widths: list[int]) -> int:
    return int(sum(int(w) for w in widths) + 2 * (len(widths) - 1))


def _distribute_widths(count: int, used_target: int, min_width: int, max_width: int) -> list[int]:
    base_used = int((min_width + 2) * count - 2)
    extra_units = int((used_target - base_used) // 2)
    per_rect_cap = int((max_width - min_width) // 2)

    order = np.random.permutation(count)
    units = np.zeros(count, dtype=int)
    remaining = int(extra_units)
    for i, idx in enumerate(order):
        slots_left = count - i - 1
        lo = max(0, remaining - per_rect_cap * slots_left)
        hi = min(per_rect_cap, remaining)
        take = int(np.random.randint(lo, hi + 1))
        units[int(idx)] = take
        remaining -= take

    return [int(min_width + 2 * int(unit)) for unit in units]


def _pick_other(color: int, excluded: set[int] | None = None) -> int:
    banned = {int(color)}
    if excluded is not None:
        banned |= {int(v) for v in excluded}
    choices = [int(c) for c in DOMINANT_COLORS if int(c) not in banned]
    return int(np.random.choice(choices))


def _make_patch(width: int, dominant: int) -> np.ndarray:
    patch = np.full((2, int(width)), int(dominant), dtype=int)

    # Fixed asymmetric motif; all colors stay in the rectangle palette so the
    # frame colors remain unique and rotations stay visually legible.
    c1 = _pick_other(dominant)
    c2 = _pick_other(dominant, {c1})
    c3 = _pick_other(dominant, {c1, c2})
    patch[0, 0] = int(c1)
    patch[1, min(int(width) - 1, 1)] = int(c2)
    patch[0, int(width) - 1] = int(c3)
    if int(width) >= 6:
        patch[1, int(width) - 2] = int(c1)
    return patch


def _sample_band_widths(max_used: int, count: int, band_idx: int) -> list[int] | None:
    min_width = 2 if band_idx == 0 else 4
    max_width = 6 if band_idx == 0 else 8
    min_used = int((min_width + 2) * count - 2)
    max_used_band = int((max_width + 2) * count - 2)
    if int(max_used) < int(min_used):
        return None
    used_cap = min(int(max_used), int(max_used_band))
    used_target = int(np.random.choice(np.arange(min_used, used_cap + 1, 2)))
    widths = _distribute_widths(count, used_target, min_width, max_width)
    if band_idx > 0 and max(widths) < 6:
        return None
    return widths


def _build_canonical() -> np.ndarray:
    for _ in range(600):
        counts = [int(np.random.randint(2, 4)) for _ in range(3)]
        if sum(counts) > 9:
            continue

        trailing = int(np.random.randint(1, 4))
        long_dim = int(np.random.randint(16, 31))
        max_used = int(long_dim - 2 - trailing)
        if max_used < 10:
            continue

        band_widths: list[list[int]] = []
        ok = True
        for band_idx, count in enumerate(counts):
            widths = _sample_band_widths(max_used, count, band_idx)
            if widths is None:
                ok = False
                break
            band_widths.append(widths)
        if not ok:
            continue

        grid = np.full((CANON_H, long_dim), BG, dtype=int)
        grid[0, :] = BLUE
        grid[:, 0] = RED
        grid[0, 0] = MARKER

        for r0, widths in zip(CANON_BAND_ROWS, band_widths):
            c = 2
            for width in widths:
                dominant = int(np.random.choice(DOMINANT_COLORS))
                paste(grid, _make_patch(width, dominant), int(r0), int(c))
                c += int(width) + 2

        return grid

    raise RuntimeError("failed to build canonical input")


def _fusion_metrics(grid: np.ndarray) -> dict:
    scene = _parse_scene(grid)
    rects = scene["rects"]
    db_values = sorted({int(rect["db"]) for rect in rects})
    band_counts = tuple(sum(1 for rect in rects if int(rect["db"]) == db) for db in db_values)

    visible_mid = 0
    visible_far = 0
    for rect in rects:
        band_rank = db_values.index(int(rect["db"]))
        arr = rect["arr"]
        if band_rank == 1 and not np.array_equal(arr, np.rot90(arr, 1)):
            visible_mid += 1
        if band_rank == 2 and not np.array_equal(np.rot90(arr, 1), np.rot90(arr, 2)):
            visible_far += 1

    combo_out = _render_output(rects, int(scene["bg"]), str(scene["blue_orient"]), "fusion")
    original_out = _render_output(rects, int(scene["bg"]), str(scene["blue_orient"]), "original")
    two_band_out = _render_output(rects, int(scene["bg"]), str(scene["blue_orient"]), "two_band")

    return {
        "db_count": int(len(db_values)),
        "band_counts": band_counts,
        "visible_mid": int(visible_mid),
        "visible_far": int(visible_far),
        "diff_vs_original": bool(not np.array_equal(combo_out, original_out)),
        "diff_vs_two_band": bool(not np.array_equal(combo_out, two_band_out)),
        "blue_orient": str(scene["blue_orient"]),
    }


def _is_valid_variant_instance(grid: np.ndarray) -> bool:
    metrics = _fusion_metrics(grid)
    if int(metrics["db_count"]) != 3:
        return False
    if min(metrics["band_counts"]) < 2:
        return False
    if int(metrics["visible_mid"]) < metrics["band_counts"][1]:
        return False
    if int(metrics["visible_far"]) < metrics["band_counts"][2]:
        return False
    if not bool(metrics["diff_vs_original"]):
        return False
    if not bool(metrics["diff_vs_two_band"]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)
    if not _is_valid_variant_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            main(grid)
            return grid
    raise RuntimeError("failed to generate combo variant input for 291dc1e1")
