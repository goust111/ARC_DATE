# Combo variant of 291dc1e1 (index 14)
# combo_spec: D5/2+D10/0
# Name: band旋转+分步堆叠

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste

# Design analysis
# a) Concrete behavior and trace:
#    The input keeps the original visual grammar: a marked corner, one horizontal and one vertical edge line,
#    and solid interior rectangles. In the canonical generator, rectangles still live in two blue-distance
#    bands at db=1 and db=4. Original 291dc1e1 globally sorted every rectangle by (db, dr, ...) and then either
#    rotated all of them (blue vertical) or none of them (blue horizontal). In this combo, the grouping step
#    becomes first-class: solve band 0 completely, insert a background separator row, then solve band 1 below it.
#    Rotation is applied inside that same band loop: nearest band rectangles keep their observed orientation,
#    more distant bands rotate 90 degrees CCW. Small trace on a top-left / blue-horizontal example:
#    near band widths [4,6], far band widths [4,8]. Original output = 2x4, 2x6, 2x4, 2x8 with no separator.
#    D5-only = 2x4, 2x6, 4x2, 8x2 with no separator. D10-only = 2x4, 2x6, [bg row], 2x4, 2x8.
#    Combo = 2x4, 2x6, [bg row], 4x2, 8x2. The two modifications therefore interact on the same grouping axis.
# b) Complexity calibration:
#    The generator keeps exactly two db bands, each with 2-4 rectangles, so the human repeatedly sees both
#    phases and both rotation states. Near-band rectangles are forced to include at least one non-square object,
#    and far-band rectangles all have width >= 4 before any scene-wide transform, so the combo differs from
#    both single variants in a visually obvious way.
# c) Hidden assumptions reviewed against the original main():
#    - Corner detection assumes the corner marker color differs from both adjacent line colors; generator keeps it.
#    - Full-row/full-column masking assumes the edge lines span the whole corner row/column; generator keeps it.
#    - Rectangle extraction assumes each connected component fills its bbox without background holes; generator
#      uses solid 2xh strips with only non-background minority pixels, so this remains true.
#    - Original code assumed one global rotation condition `blue_orient == "v"`; this is invalid after D5/2, so
#      rotation is now decided per band from db.
#    - Original code assumed one global total order was enough; after D10/0 we must first partition by db and
#      only then sort inside each band by red-distance order.
# d) Placement legality:
#    Canonical inputs are 8 rows tall, matching the original spec: edge row + gap + near band (2 rows) + gap +
#    far band (2 rows) + border. Within each band, rectangles are separated by 2 background cells, start 1 cell
#    away from the red edge, and the total used width plus trailing margin is bounded so no rectangle overlaps or
#    exits the grid. Rotations/flips of the whole scene preserve disjointness.
# e) Visual observability:
#    From a few examples the solver can see two simultaneous regularities: all top-step rectangles come from the
#    band nearest the blue line, and only the later step changes aspect ratio. The separator row makes the step
#    boundary explicit; the asymmetric internal motifs make the 90-degree rotation visible.
# f) Visual cleanliness:
#    The scene still uses a single clean coordinate frame and two uncluttered rectangle rows/columns. The output
#    adds only one full background separator row between non-empty blocks, so there is no ambiguity about which
#    rectangles belong to which step.
# g) Color-change inferability:
#    Colors do not encode the rule. Each rectangle gets one dominant fill color plus a tiny asymmetric motif from
#    other non-edge colors purely to make rotation observable. Because the rotation cue is tied to band position
#    and separator structure, the human need not infer any hidden color mapping.
# New solver assumptions after the combo:
#    - There is at least one rectangle, so a nearest band exists.
#    - Grouping by sorted unique db values is meaningful; generator guarantees exactly two db values, but the
#      solver degrades gracefully to "first band unrotated, later bands rotated" if extra bands appear.
#    - Every band is non-empty before a separator is inserted; generator guarantees this by construction.
#    - Within a band, sorting by (dr, along red, along blue) is deterministic even if two rectangles share dr.


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)


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


def _along_from_corner(box: tuple[int, int, int, int], orient: str, cr: int, cc: int) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


def _extract_scene(grid: np.ndarray) -> dict:
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
        arr = crop_bbox(g, box)
        if bool((arr == int(bg)).any()):
            continue
        rects.append({"box": box, "arr": arr.copy()})

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


def _global_sort_key(rect: dict, blue_orient: str, red_orient: str, cr: int, cc: int) -> tuple[int, int, int, int]:
    return (
        int(rect["db"]),
        int(rect["dr"]),
        _along_from_corner(rect["box"], blue_orient, cr, cc),
        _along_from_corner(rect["box"], red_orient, cr, cc),
    )


def _band_sort_key(rect: dict, red_orient: str, blue_orient: str, cr: int, cc: int) -> tuple[int, int, int]:
    return (
        int(rect["dr"]),
        _along_from_corner(rect["box"], red_orient, cr, cc),
        _along_from_corner(rect["box"], blue_orient, cr, cc),
    )


def _transform_rect(arr: np.ndarray, rect: dict, blue_orient: str, near_db: int, rotate_mode: str) -> np.ndarray:
    out = np.array(arr, copy=True)
    if rotate_mode == "blue_orient":
        if blue_orient == "v":
            out = np.rot90(out, k=1)
    elif rotate_mode == "band":
        if int(rect["db"]) > int(near_db):
            out = np.rot90(out, k=1)
    else:
        raise ValueError(f"unknown rotate_mode: {rotate_mode}")
    return out


def _solve(grid: np.ndarray, rotate_mode: str, multi_step: bool) -> np.ndarray:
    scene = _extract_scene(grid)
    bg = int(scene["bg"])
    blue_orient = str(scene["blue_orient"])
    red_orient = str(scene["red_orient"])
    cr = int(scene["cr"])
    cc = int(scene["cc"])
    rects = list(scene["rects"])
    near_db = min(int(rect["db"]) for rect in rects)

    bands: list[list[np.ndarray]] = []
    if multi_step:
        db_values = sorted({int(rect["db"]) for rect in rects})
        for db in db_values:
            group = [rect for rect in rects if int(rect["db"]) == int(db)]
            group.sort(key=lambda rect: _band_sort_key(rect, red_orient, blue_orient, cr, cc))
            bands.append(
                [_transform_rect(rect["arr"], rect, blue_orient, near_db, rotate_mode) for rect in group]
            )
    else:
        rects.sort(key=lambda rect: _global_sort_key(rect, blue_orient, red_orient, cr, cc))
        bands.append([_transform_rect(rect["arr"], rect, blue_orient, near_db, rotate_mode) for rect in rects])

    out_w = int(max(int(arr.shape[1]) for band in bands for arr in band))
    out_h = int(sum(int(arr.shape[0]) for band in bands for arr in band) + max(0, len(bands) - 1 if multi_step else 0))
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out_r = 0
    for band_idx, band in enumerate(bands):
        if multi_step and band_idx > 0:
            out_r += 1
        for arr in band:
            h, w = map(int, arr.shape)
            out_c = int((out_w - w) // 2)
            paste(out, arr, int(out_r), int(out_c))
            out_r += int(h)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - partition rectangles by distance to the blue edge
    - output the nearest band first and later bands below it
    - insert one full background row between successive bands
    - keep the nearest band unrotated and rotate later bands 90 degrees CCW
    """
    return _solve(grid, rotate_mode="band", multi_step=True)


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

    # Fixed asymmetric motif so the 90-degree rotation is visible in output.
    patch[0, 0] = _pick_other(dominant)
    patch[1, min(int(width) - 1, 1)] = _pick_other(dominant)
    if int(width) >= 4:
        patch[0, int(width) - 1] = _pick_other(dominant)
    return patch


def _sample_widths(count: int, choices: list[int], require_nonsquare: bool) -> list[int]:
    widths = [int(w) for w in np.random.choice(choices, size=count, replace=True)]
    if require_nonsquare and all(int(w) == 2 for w in widths):
        widths[int(np.random.randint(count))] = int(np.random.choice([w for w in choices if int(w) > 2]))
    return widths


def _build_canonical() -> np.ndarray:
    for _ in range(500):
        near_count = int(np.random.randint(2, 5))
        far_count = int(np.random.randint(2, 5))
        if near_count + far_count > 8:
            continue

        near_widths = _sample_widths(near_count, [2, 4, 6], require_nonsquare=True)
        far_widths = _sample_widths(far_count, [4, 6, 8], require_nonsquare=True)
        used = max(_band_used(near_widths), _band_used(far_widths))
        trailing = int(np.random.randint(1, 3))
        long_dim = int(2 + used + trailing)
        if not (9 <= int(long_dim) <= 28):
            continue

        grid = np.full((8, long_dim), BG, dtype=int)
        grid[0, :] = BLUE
        grid[:, 0] = RED
        grid[0, 0] = MARKER

        for r0, widths in ((2, near_widths), (5, far_widths)):
            c = 2
            for width in widths:
                dominant = int(np.random.choice(DOMINANT_COLORS))
                paste(grid, _make_patch(width, dominant), int(r0), int(c))
                c += int(width) + 2

        return grid
    raise RuntimeError("failed to build canonical combo instance")


def _count_internal_bg_rows(grid: np.ndarray, bg: int) -> int:
    if int(grid.shape[0]) <= 2:
        return 0
    return sum(bool((grid[r, :] == int(bg)).all()) for r in range(1, int(grid.shape[0]) - 1))


def _combo_metrics(grid: np.ndarray) -> dict:
    scene = _extract_scene(grid)
    rects = list(scene["rects"])
    db_values = sorted({int(rect["db"]) for rect in rects})
    near_db = min(db_values)
    near_count = sum(1 for rect in rects if int(rect["db"]) == int(near_db))
    far_count = sum(1 for rect in rects if int(rect["db"]) > int(near_db))
    near_nonsquare = sum(1 for rect in rects if int(rect["db"]) == int(near_db) and int(rect["arr"].shape[0]) != int(rect["arr"].shape[1]))
    far_nonsquare = sum(1 for rect in rects if int(rect["db"]) > int(near_db) and int(rect["arr"].shape[0]) != int(rect["arr"].shape[1]))

    combo = _solve(grid, rotate_mode="band", multi_step=True)
    original = _solve(grid, rotate_mode="blue_orient", multi_step=False)
    d5_only = _solve(grid, rotate_mode="band", multi_step=False)
    d10_only = _solve(grid, rotate_mode="blue_orient", multi_step=True)

    return {
        "db_values": db_values,
        "near_count": int(near_count),
        "far_count": int(far_count),
        "near_nonsquare": int(near_nonsquare),
        "far_nonsquare": int(far_nonsquare),
        "separator_rows": int(_count_internal_bg_rows(combo, int(scene["bg"]))),
        "diff_vs_original": bool(not np.array_equal(combo, original)),
        "diff_vs_d5": bool(not np.array_equal(combo, d5_only)),
        "diff_vs_d10": bool(not np.array_equal(combo, d10_only)),
    }


def _is_valid_variant_instance(grid: np.ndarray) -> bool:
    try:
        metrics = _combo_metrics(grid)
    except Exception:
        return False

    if metrics["db_values"] != [1, 4]:
        return False
    if min(int(metrics["near_count"]), int(metrics["far_count"])) < 2:
        return False
    if int(metrics["separator_rows"]) != 1:
        return False
    if int(metrics["near_nonsquare"]) < 1 or int(metrics["far_nonsquare"]) < 1:
        return False
    if not metrics["diff_vs_original"]:
        return False
    if not metrics["diff_vs_d5"]:
        return False
    if not metrics["diff_vs_d10"]:
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
    raise RuntimeError("failed to generate input for 291dc1e1 combo variant")
