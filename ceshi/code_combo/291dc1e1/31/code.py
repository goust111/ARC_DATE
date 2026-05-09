# Combo variant of 291dc1e1 (index 31)
# combo_spec: D4/0+D5/2+D10/0
# Name: 三band+band旋转+三步堆叠

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

# Design analysis
# a) Concrete behavior and trace:
#    - Start from the original 291dc1e1 scene: one marked corner, a full blue edge line, a full red edge line,
#      and interior solid rectangles that are parallel to the blue line.
#    - Original behavior on a canonical top-left instance with two bands was:
#      sort all rectangles by (db, dr), optionally rotate everything if blue is vertical, then stack in one pass.
#    - This combo changes all three touched stages together:
#      1. D4/0 expands the source layout from 2 blue-distance bands to 3 bands at db=1,4,7.
#      2. D5/2 is fused with that third band by making rotation depend on band rank, not on blue orientation:
#         band 0 keeps 0 quarter-turns, band 1 gets 1 CCW quarter-turn, band 2 gets 2 quarter-turns.
#      3. D10/0 then renders those three bands in three separate stacking phases, inserting one full bg row
#         between phases.
#    - Small trace: suppose the three bands contain A,B / C,D / E,F after sorting by dr.
#      Original output could be A,B,C,D,E,F or all-rotated A',B',C',D',E',F' depending on blue orientation.
#      New output is A,B,[bg row],C',D',[bg row],E'',F'' where ' = 90 CCW and '' = 180.
#      The solver therefore has to detect a 3-level distance structure and cannot rely on blue orientation alone.
# b) Complexity calibration:
#    - The combo keeps only three band levels and only three rotation states (0/90/180), so the rule is visible
#      but not combinatorially loose.
#    - The generator makes every band contain 2-3 rectangles, giving repeated evidence for each stage.
#    - Widths are restricted to {4,6,8} and each patch uses a fixed asymmetric motif, so 90-degree and 180-degree
#      rotations are visible to a human rather than hidden by symmetry.
# c) Hidden assumptions reviewed against the original solver:
#    - Corner detection still depends on a unique marker at a corner whose color differs from the adjacent line
#      colors; the generator preserves that exactly.
#    - Masking the whole corner row and column still assumes the two edge lines span those full axes; preserved.
#    - Rectangle extraction still assumes each connected component has a bbox with no bg holes; preserved because
#      the generator only recolors cells inside solid rectangles.
#    - The original single global `rects.sort(...)` assumed a total order was enough. That becomes insufficient:
#      sorting must now happen inside each db-group, and groups must be rendered separately.
#    - The original `if blue_orient == "v": rot90(...)` assumed one rotation state per instance. That assumption is
#      broken by the combo and replaced with a band-rank-driven rule.
#    - Output height can no longer be just sum(rect heights); it must add separator rows between the three phases.
# d) Placement legality:
#    - Canonical short dimension becomes 11:
#      1 edge + 1 gap + 2 band0 + 1 gap + 2 band1 + 1 gap + 2 band2 + 1 border.
#    - Bands start at rows 2, 5, 8 in the canonical orientation; within each band, rectangles start at dr=1 and
#      are separated by 2 bg cells along the blue direction, so they never touch or overlap.
#    - Whole-grid flips/rotations are applied only after legal canonical placement, preserving disjointness.
# e) Visual observability:
#    - Across examples, the output always has three segments separated by two full bg rows.
#    - Segment 1 keeps horizontal 2xh rectangles, segment 2 turns them vertical, and segment 3 keeps the width but
#      flips the asymmetric motif by 180 degrees. The banded input layout and the staged output reinforce each other.
# f) Cleanliness:
#    - No extra markers are introduced beyond the original corner marker and edge lines.
#    - Minority colors stay inside rectangles and never reuse bg; edge-line colors are also avoided inside patches,
#      so object boundaries remain clean.
# g) Color inferability:
#    - Colors do not encode the rule. Rotation and grouping are conveyed by band position and output segmentation.
#      The same motif is reused across bands so color variation does not compete with the geometric signal.
# new_main_assumptions:
#    - The combo solver assumes rectangles can be partitioned by distinct db values; if fewer than three bands are
#      present it still works, but the generator guarantees exactly three.
#    - Band order is inferred from sorted db values, so the generator must preserve strict spacing between bands.
#    - 180-degree rotation is only human-visible if the patches are asymmetric; the generator guarantees that.


BG = 8
BLUE = 1
RED = 2
MARKER = 0
SHORT_DIM = 11
BAND_OFFSETS = (2, 5, 8)
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
WIDTH_CHOICES = np.array([4, 6, 8], dtype=int)


def _locate_structure(g: np.ndarray) -> tuple[int, int, int, int, str, str]:
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
    return bg, cr, cc, marker, blue_orient, red_orient


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


def _extract_rects(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, cr, cc, marker, blue_orient, red_orient = _locate_structure(g)

    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects: list[dict] = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = tuple(map(int, bbox_coords(coords)))
        sub = crop_bbox(g, box)
        if bool((sub == int(bg)).any()):
            continue
        rects.append({"box": box, "arr": sub.copy()})

    assert rects

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)
    for rect in rects:
        rect["db"] = int(_dist_to_edge(rect["box"], blue_orient, blue_edge_idx))
        rect["dr"] = int(_dist_to_edge(rect["box"], red_orient, red_edge_idx))

    return {
        "grid": g,
        "bg": int(bg),
        "cr": int(cr),
        "cc": int(cc),
        "blue_orient": blue_orient,
        "red_orient": red_orient,
        "rects": rects,
    }


def _band_sort_key(rect: dict, red_orient: str, blue_orient: str, cr: int, cc: int) -> tuple[int, int, int]:
    box = rect["box"]
    return (
        int(rect["dr"]),
        _along_from_corner(box, red_orient, cr, cc),
        _along_from_corner(box, blue_orient, cr, cc),
    )


def _rotation_quarters(band_rank: int, rotate_mode: str, blue_orient: str) -> int:
    if rotate_mode == "original":
        return 1 if blue_orient == "v" else 0
    if rotate_mode == "binary_band":
        return 0 if band_rank == 0 else 1
    assert rotate_mode == "rank_band"
    return int(band_rank)


def _solve(grid: np.ndarray, rotate_mode: str, step_mode: str) -> np.ndarray:
    scene = _extract_rects(grid)
    bg = int(scene["bg"])
    blue_orient = str(scene["blue_orient"])
    red_orient = str(scene["red_orient"])
    cr = int(scene["cr"])
    cc = int(scene["cc"])
    rects = list(scene["rects"])

    bands: list[list[np.ndarray]] = []
    for band_rank, db in enumerate(sorted({int(rect["db"]) for rect in rects})):
        group = [rect for rect in rects if int(rect["db"]) == int(db)]
        group.sort(key=lambda rect: _band_sort_key(rect, red_orient, blue_orient, cr, cc))
        k = _rotation_quarters(int(band_rank), rotate_mode, blue_orient)
        band_arrays = [np.rot90(rect["arr"], k=k) if k else rect["arr"].copy() for rect in group]
        bands.append(band_arrays)

    assert bands
    seq = [arr for band in bands for arr in band]
    out_w = int(max(int(arr.shape[1]) for arr in seq))
    out_h = int(sum(int(arr.shape[0]) for arr in seq))
    if step_mode == "banded":
        out_h += max(0, len(bands) - 1)

    out = np.full((out_h, out_w), int(bg), dtype=int)
    out_r = 0
    for band_idx, band in enumerate(bands):
        if step_mode == "banded" and band_idx > 0:
            out_r += 1
        for arr in band:
            h, w = map(int, arr.shape)
            out_c = int((out_w - w) // 2)
            paste(out, arr, int(out_r), int(out_c))
            out_r += int(h)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder rectangles by three blue-distance bands and stack them in three phases.

    Combo rule:
    - Partition rectangles by distance to the blue edge.
    - Sort bands from near to far; sort within each band by red-edge distance.
    - Rotate band 0 by 0, band 1 by 90 CCW, band 2 by 180.
    - Stack each band top-to-bottom, centered, with one bg separator row between bands.
    """
    return _solve(grid, rotate_mode="rank_band", step_mode="banded")


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


def _pick_other(color: int) -> int:
    choices = [int(c) for c in DOMINANT_COLORS if int(c) != int(color)]
    return int(np.random.choice(choices))


def _make_patch(width: int, dominant: int) -> np.ndarray:
    patch = np.full((2, int(width)), int(dominant), dtype=int)

    # Fixed asymmetric motif so both 90-degree and 180-degree turns remain visible.
    patch[0, 0] = _pick_other(dominant)
    patch[1, 1] = _pick_other(dominant)
    patch[0, int(width) - 1] = _pick_other(dominant)
    patch[1, int(width) - 2] = _pick_other(dominant)
    return patch


def _build_canonical() -> np.ndarray:
    for _ in range(500):
        counts = [int(np.random.randint(2, 4)) for _ in range(3)]
        if sum(counts) > 9:
            continue

        band_widths = [
            [int(w) for w in np.random.choice(WIDTH_CHOICES, size=count, replace=True)]
            for count in counts
        ]
        used = [_band_used(widths) for widths in band_widths]
        long_dim = int(max(used) + 2 + np.random.randint(1, 4))
        if long_dim < 12 or long_dim > 30:
            continue

        grid = np.full((SHORT_DIM, long_dim), BG, dtype=int)
        grid[0, :] = BLUE
        grid[:, 0] = RED
        grid[0, 0] = MARKER

        for r0, widths in zip(BAND_OFFSETS, band_widths):
            c = 2
            for width in widths:
                dominant = int(np.random.choice(DOMINANT_COLORS))
                paste(grid, _make_patch(int(width), dominant), int(r0), int(c))
                c += int(width) + 2

        return grid

    raise RuntimeError("failed to build canonical input")


def _separator_rows(out: np.ndarray, bg: int) -> int:
    return sum(1 for r in range(1, int(out.shape[0]) - 1) if bool((out[r, :] == int(bg)).all()))


def _variant_metrics(grid: np.ndarray) -> dict:
    scene = _extract_rects(grid)
    rects = list(scene["rects"])
    db_values = sorted({int(rect["db"]) for rect in rects})
    counts = [sum(1 for rect in rects if int(rect["db"]) == db) for db in db_values]

    combo_out = _solve(grid, rotate_mode="rank_band", step_mode="banded")
    original_out = _solve(grid, rotate_mode="original", step_mode="global")
    binary_band_out = _solve(grid, rotate_mode="binary_band", step_mode="banded")

    return {
        "band_count": len(db_values),
        "band_counts": tuple(int(x) for x in counts),
        "separator_rows": int(_separator_rows(combo_out, int(scene["bg"]))),
        "diff_vs_original": not np.array_equal(combo_out, original_out),
        "diff_vs_binary_band": not np.array_equal(combo_out, binary_band_out),
    }


def _is_valid_variant_instance(grid: np.ndarray) -> bool:
    metrics = _variant_metrics(grid)
    if metrics["band_count"] != 3:
        return False
    if min(metrics["band_counts"]) < 2:
        return False
    if metrics["separator_rows"] != 2:
        return False
    if not metrics["diff_vs_original"]:
        return False
    if not metrics["diff_vs_binary_band"]:
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
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            main(grid)
            return grid
    raise RuntimeError("failed to generate input for 291dc1e1 combo variant")
