# Combo variant of 291dc1e1 (index 41)
# combo_spec: D3/0+D5/2+D6/0
# Name: L/T形+band旋转+左对齐

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis
# ---------------
# a) Concrete behavior trace
# The source task uses two edge lines meeting at a marked corner and several
# interior solid rectangles. The solver sorts objects by (distance to blue edge,
# distance to red edge), maybe rotates them, then stacks them.
#
# This combo keeps the same coordinate-system scaffold but changes all three
# stages together:
# - Source objects are no longer filled rectangles. They are L/T sprites, so the
#   extracted bbox now contains background holes around the shape.
# - Rotation no longer depends on blue orientation. Only the near-blue band keeps
#   its pose; the far-blue band rotates 90 degrees CCW.
# - Output stacking no longer centers each item. Every transformed sprite is
#   pasted with its left edge at column 0.
#
# Small trace in canonical orientation:
# - Blue line = top row, red line = left column.
# - Near band starts at row 2 (db=1), far band starts at row 6 (db=5).
# - Suppose sorted objects are:
#     near: L(2x4), T(3x3)
#     far : L(3x2), T(2x3)
# - Source task with horizontal blue would keep all four orientations and center
#   them in the output width.
# - This combo keeps the two near objects unchanged, rotates the two far objects,
#   then stacks all four with shared left edge. The visible difference is the
#   joint effect: complex silhouettes, split-by-band rotation, and a staircase
#   right edge caused by left alignment.
#
# b) Complexity calibration
# To keep the rule inferable but nontrivial:
# - Only two blue-distance bands are used, matching the original task's coarse
#   grouping and giving repeated evidence for the rotation rule.
# - Each instance has at least two objects per band, so "near stays / far turns"
#   is supported by multiple examples inside one grid.
# - The sprite library uses a small set of tight L/T templates with bbox widths
#   in {2,3,4} and heights in {2,3}. This is enough to make rotation and
#   left-alignment visible without introducing too many free degrees.
# - The generator requires at least one L and one T overall so the source-shape
#   variation is not a single-shape special case.
#
# c) Implicit assumption audit
# Relevant assumptions in the original main():
# - Corner detection assumes one corner marker differs from both adjacent edge
#   colors. Preserved.
# - Blue/red axis inference assumes one full row and one full column line exist.
#   Preserved.
# - `line_mask[cr, :]` and `line_mask[:, cc]` assume the structural lines occupy
#   the entire corner row/column. Preserved.
# - Original object extraction used `crop_bbox(...)` and rejected any bbox that
#   contained background. That encoded the "solid rectangle" assumption and is
#   invalid here because L/T sprites necessarily have background in their bbox.
#   The new solver instead reconstructs each component inside its tight bbox and
#   later pastes it with `transparent=bg`.
# - The original rotation branch assumed a single per-instance rotation state
#   (`if blue_orient == "v"`). D5/2 breaks that; the new solver decides rotation
#   per object from its `db` band.
# - The original placement branch assumed centered stacking
#   (`out_c = (outW - w) // 2`). D6/0 breaks that; the new solver uses `out_c=0`.
#
# d) Placement legality
# The canonical generator uses:
# - top edge line at row 0, left edge line at col 0,
# - near band row start 2,
# - far band row start 6,
# - horizontal gaps of 2 cells between sprites,
# - one empty separator row between bands.
# Because every sprite bbox is at most 3 rows tall and 4 columns wide, this
# guarantees no overlap and keeps all objects in bounds. Global flips/rotations
# preserve legality.
#
# e) Visual observability
# A human can infer the combo from a few samples because:
# - input objects are visibly L/T rather than rectangles,
# - output shows that all objects still sort into one vertical stack,
# - sprites nearer the blue axis keep their local pose while the farther band
#   turns 90 degrees,
# - the left edges of all output sprites line up while the right edges vary.
# The generator rejects samples where the band-rotation or left-alignment effect
# would be invisible.
#
# f) Visual cleanliness
# The scene still contains only three layers of meaning:
# - one marked corner,
# - two axis lines,
# - separated interior sprites.
# No object touches an axis or another object, and the output uses only one
# compact left-flush stack, so silhouettes remain easy to read.
#
# g) Color inference
# Colors do not encode the rule. They only distinguish sprites visually. To keep
# the task clean, sprites are monochrome and avoid the marker/axis colors.
#
# 9) New assumptions introduced by this combo
# - Object identity now depends on 4-connected components staying separated.
#   The generator guarantees at least a 2-cell horizontal gap and a 1-row band
#   gap after bbox placement.
# - The solver uses `min(db)` as the near band and `db > min(db)` as far bands.
#   The generator guarantees exactly two db values and both bands non-empty.
# - Because output pasting is transparent, each sprite array must be a tight bbox
#   with background elsewhere inside that bbox. `_component_to_patch` provides
#   that representation directly from component coordinates.


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)


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


def _component_to_patch(grid: np.ndarray, coords: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    box = bbox_coords(coords)
    r0, c0, r1, c1 = map(int, box)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    rr = coords[:, 0] - int(r0)
    cc = coords[:, 1] - int(c0)
    patch[rr, cc] = grid[coords[:, 0], coords[:, 1]]
    return tuple(map(int, box)), patch


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
    objects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box, arr = _component_to_patch(g, coords, int(bg))
        objects.append({"box": box, "arr": arr})
    assert objects

    def dist_to_edge(obj: dict, orient: str, edge_idx: int) -> int:
        r0, c0, r1, c1 = map(int, obj["box"])
        if orient == "v":
            if c0 > edge_idx:
                return int(c0 - edge_idx - 1)
            if c1 < edge_idx:
                return int(edge_idx - c1 - 1)
            return 0
        if r0 > edge_idx:
            return int(r0 - edge_idx - 1)
        if r1 < edge_idx:
            return int(edge_idx - r1 - 1)
        return 0

    def along_from_corner(obj: dict, orient: str) -> int:
        r0, c0, r1, c1 = map(int, obj["box"])
        if orient == "v":
            return int(r0 if cr == 0 else -r1)
        return int(c0 if cc == 0 else -c1)

    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    for obj in objects:
        obj["db"] = int(dist_to_edge(obj, blue_orient, blue_edge_idx))
        obj["dr"] = int(dist_to_edge(obj, red_orient, red_edge_idx))

    objects.sort(
        key=lambda obj: (
            int(obj["db"]),
            int(obj["dr"]),
            along_from_corner(obj, blue_orient),
            along_from_corner(obj, red_orient),
        )
    )

    return {"grid": g, "bg": bg, "blue_orient": blue_orient, "objects": objects}


def _transform_object(arr: np.ndarray, db: int, near_db: int, blue_orient: str, rotate_mode: str) -> np.ndarray:
    if rotate_mode == "band":
        return np.rot90(arr, k=1) if int(db) > int(near_db) else arr
    if rotate_mode == "blue_orient":
        return np.rot90(arr, k=1) if blue_orient == "v" else arr
    raise ValueError(f"unknown rotate_mode: {rotate_mode}")


def _render_output(objects: list[dict], bg: int, blue_orient: str, rotate_mode: str, left_align: bool) -> np.ndarray:
    near_db = min(int(obj["db"]) for obj in objects)
    seq = [_transform_object(obj["arr"], int(obj["db"]), int(near_db), blue_orient, rotate_mode) for obj in objects]

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


def _solve(grid: np.ndarray, *, rotate_mode: str, left_align: bool) -> np.ndarray:
    scene = _parse_scene(grid)
    return _render_output(scene["objects"], int(scene["bg"]), str(scene["blue_orient"]), rotate_mode, left_align)


def _array_diff_pixels(a: np.ndarray, b: np.ndarray, bg: int) -> int:
    ah, aw = map(int, a.shape)
    bh, bw = map(int, b.shape)
    h = max(ah, bh)
    w = max(aw, bw)
    aa = np.full((h, w), int(bg), dtype=int)
    bb = np.full((h, w), int(bg), dtype=int)
    aa[:ah, :aw] = a
    bb[:bh, :bw] = b
    return int(np.count_nonzero(aa != bb))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder L/T sprites by proximity to two colored edge lines.

    Combo rule:
    - Sort objects by (distance to blue edge, distance to red edge, tie breaks from the corner).
    - Objects in the nearest blue-distance band keep their orientation.
    - Objects in the farther blue-distance band rotate 90 degrees CCW.
    - Stack top-to-bottom with shared left-edge alignment.
    """
    return _solve(grid, rotate_mode="band", left_align=True)


def _sprite_catalog() -> list[tuple[str, np.ndarray]]:
    return [
        ("L", np.array([[1, 0, 0], [1, 1, 1]], dtype=int)),
        ("L", np.array([[1, 0, 0, 0], [1, 1, 1, 1]], dtype=int)),
        ("L", np.array([[1, 0], [1, 0], [1, 1]], dtype=int)),
        ("L", np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=int)),
        ("T", np.array([[1, 1, 1], [0, 1, 0]], dtype=int)),
        ("T", np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int)),
    ]


def _paint_sprite(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask.astype(bool)] = int(color)
    return patch


def _band_used_width(sprites: list[np.ndarray]) -> int:
    return int(sum(int(sprite.shape[1]) for sprite in sprites) + 2 * (len(sprites) - 1))


def _sample_band(count: int) -> tuple[list[np.ndarray], list[str]]:
    catalog = _sprite_catalog()
    idxs = np.random.choice(len(catalog), size=count, replace=True)
    sprites = [catalog[int(idx)][1].copy() for idx in idxs]
    kinds = [str(catalog[int(idx)][0]) for idx in idxs]
    return sprites, kinds


def _build_canonical() -> tuple[np.ndarray, dict]:
    for _ in range(400):
        n_near = int(np.random.randint(2, 5))
        n_far = int(np.random.randint(2, 5))
        if n_near + n_far > 7:
            continue

        near_sprites, near_kinds = _sample_band(n_near)
        far_sprites, far_kinds = _sample_band(n_far)
        all_kinds = near_kinds + far_kinds
        if "L" not in all_kinds or "T" not in all_kinds:
            continue

        used_near = _band_used_width(near_sprites)
        used_far = _band_used_width(far_sprites)
        trailing = int(np.random.randint(1, 4))
        width = int(max(used_near, used_far) + 2 + trailing)
        height = 10
        if width > 24:
            continue

        grid = np.full((height, width), BG, dtype=int)
        grid[0, :] = BLUE
        grid[:, 0] = RED
        grid[0, 0] = MARKER

        meta_objects = []
        for band_name, row_start, sprites, kinds in (
            ("near", 2, near_sprites, near_kinds),
            ("far", 6, far_sprites, far_kinds),
        ):
            c = 2
            for sprite, kind in zip(sprites, kinds):
                color = int(np.random.choice(DOMINANT_COLORS))
                patch = _paint_sprite(sprite, color)
                paste(grid, patch, int(row_start), int(c), transparent=BG)
                meta_objects.append(
                    {
                        "band": band_name,
                        "kind": kind,
                        "shape": tuple(map(int, sprite.shape)),
                        "row": int(row_start),
                        "col": int(c),
                    }
                )
                c += int(sprite.shape[1]) + 2

        return grid, {"objects": meta_objects}
    raise RuntimeError("failed to build canonical combo input")


def _variant_metrics(grid: np.ndarray) -> dict:
    scene = _parse_scene(grid)
    objects = scene["objects"]
    near_db = min(int(obj["db"]) for obj in objects)
    db_values = sorted({int(obj["db"]) for obj in objects})

    combo = _render_output(objects, int(scene["bg"]), str(scene["blue_orient"]), "band", True)
    orient_left = _render_output(objects, int(scene["bg"]), str(scene["blue_orient"]), "blue_orient", True)
    band_center = _render_output(objects, int(scene["bg"]), str(scene["blue_orient"]), "band", False)
    original_like = _render_output(objects, int(scene["bg"]), str(scene["blue_orient"]), "blue_orient", False)

    combo_seq = [_transform_object(obj["arr"], int(obj["db"]), int(near_db), str(scene["blue_orient"]), "band") for obj in objects]
    orient_seq = [_transform_object(obj["arr"], int(obj["db"]), int(near_db), str(scene["blue_orient"]), "blue_orient") for obj in objects]

    disagreement = int(
        sum(0 if np.array_equal(a, b) else 1 for a, b in zip(combo_seq, orient_seq))
    )
    out_w = int(max(int(arr.shape[1]) for arr in combo_seq))
    alignment_signal = int(sum((out_w - int(arr.shape[1])) // 2 for arr in combo_seq))

    return {
        "db_values": db_values,
        "band_counts": [sum(1 for obj in objects if int(obj["db"]) == db) for db in db_values],
        "disagreement_count": disagreement,
        "alignment_signal": alignment_signal,
        "band_diff_pixels": _array_diff_pixels(combo, orient_left, int(scene["bg"])),
        "align_diff_pixels": _array_diff_pixels(combo, band_center, int(scene["bg"])),
        "orig_diff_pixels": _array_diff_pixels(combo, original_like, int(scene["bg"])),
    }


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    canonical, meta = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    grid = _transform(canonical, blue_orient, corner)

    metrics = _variant_metrics(grid)
    if len(metrics["db_values"]) != 2:
        return None
    if min(metrics["band_counts"]) < 2:
        return None
    if metrics["disagreement_count"] < 2:
        return None
    if metrics["alignment_signal"] <= 0:
        return None
    if metrics["band_diff_pixels"] <= 0:
        return None
    if metrics["align_diff_pixels"] <= 0:
        return None
    if metrics["orig_diff_pixels"] <= 0:
        return None

    meta.update({"blue_orient": blue_orient, "corner": corner, "metrics": metrics})
    return grid, meta


def generate_input() -> np.ndarray:
    for _ in range(300):
        attempt = _gen_attempt()
        if attempt is None:
            continue
        grid, _meta = attempt
        main(grid)
        return grid
    raise RuntimeError("failed to generate input for 291dc1e1 combo variant 41")
