# Combo variant of 898e7135 (index 45)
# combo_spec: D1/1+D2/0+D15/0
# Name: 多色旋转消歧

from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np

from arc_utils import (
    bbox,
    bbox_coords,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
    random_free_location,
)


"""
Design analysis for combo D1/1 + D2/0 + D15/0
==============================================

a) Concrete behavior and trace
spec_variant:
  preserved_core:
    - The input still contains one large rectangular frame enclosing background holes.
    - Each outside piece is still a k-by-k block upscaling of exactly one hole shape.
    - The output is still the tight frame layout scaled by k, with the matched pieces pasted in.
  fused_changes:
    - D1/1: matching is rotation-only; mirror flips are illegal.
    - D2/0: every non-unit outside piece is multicolor at the block level, using 2-3 colors.
    - D15/0: at least one non-unit hole shape is duplicated, so shape alone is ambiguous.
  chosen visible encoding:
    - A duplicate-shape pair of holes is generated. Their pieces have the same geometry but different
      multicolor block patterns. Each of those two holes carries two touching wall markers. Each marker
      copies the color of a specific adjacent block cell from the correct piece in the correct rotation.
    - A separate chiral mirror pair of holes is also generated. Those two holes do not receive markers;
      they are distinguishable only because flips are forbidden.
  example_trace:
    - Duplicate pair:
      two identical T-like holes appear inside the frame. Outside, there are two T-like pieces with the
      same shape and the same 3-color palette, but different color layouts. Shape matching returns both
      pieces for either hole. The two wall markers next to hole A match only piece A when rotated into
      slot orientation, and similarly for hole B.
    - Chiral pair:
      one L hole and one mirrored-L hole also appear. Their outside pieces are rotated copies only.
      Because no flip is allowed, the left-handed piece cannot fill the right-handed hole.
    - Difference from the original task:
      the old task could ignore color layouts and could also flip pieces. This combo forces both a
      color-pattern tie-break and a chirality-sensitive geometric match.

b) Complexity calibration
  - The duplicate ambiguity is controlled: exactly one repeated non-unit rotation-class is forced.
  - The marker channel is sparse: only the duplicate pair gets markers, and each such hole gets exactly
    two marker pixels. This keeps the clue readable and prevents the frame from becoming noisy.
  - The chiral pair is separate from the duplicate pair, so the "no flips" rule is not masked by the
    marker rule. Each sample visibly demonstrates both aspects of the combo.
  - Color freedom is limited: each non-unit piece uses only 2-3 colors, and the duplicate pair shares a
    palette so the solver must use arrangement, not merely a single unique color id.

c) Implicit-assumption audit of the original main()
  - `find_components(..., monochromatic=False)`:
    assumption: the frame is still the largest non-background component even after adding colored markers.
    generator guarantee: frame area is checked to exceed every piece area, and all pieces remain separated.
  - `k` inferred from a solid square piece:
    assumption: at least one unit slot remains, and its piece is still a full k-by-k square.
    generator guarantee: every sample includes one unit slot.
  - matching by shape only:
    assumption broken by D15/0. Duplicate holes intentionally create >1 shape hit.
  - allowing flips:
    assumption broken by D1/1. Candidate generation must enumerate rotations only.
  - monochrome piece identity:
    assumption broken by D2/0. The solver must preserve full multicolor patches and inspect block colors
    against marker clues, not just compare masks.
  - first greedy exact hit:
    no longer safe in general. The solver now collects all rotation-shape hits for a slot and, when needed,
    filters them by marker constraints until exactly one candidate remains.
  - new solver-side assumption:
    every duplicate-hole marker pixel is 4-adjacent to exactly one hole cell, so a marker can be mapped to
    one specific block position inside the slot. The generator enforces this explicitly.

d) Placement legality
  - Layout holes are placed with 8-neighbor exclusion, so different holes never touch even diagonally.
  - Duplicate pair markers are placed on frame-wall cells only; they never overwrite holes or other markers.
  - Outside pieces use `random_free_location(..., padding=1, padding_connectivity=8)`, so they stay separate
    from the frame, from each other, and from distractors.
  - Every generation attempt is rejected if bounds, density, background-majority, frame-dominance, or oracle
    equivalence checks fail.

e) Visual observability
  - A human can see the repeated hole shape and the two small colored wall hints beside those holes.
  - The corresponding outside pieces are visibly multicolor, and the marker colors align with specific block
    positions only after the correct rotation is chosen.
  - The mirror pair is also visually obvious because the two hole silhouettes are opposite-handed, and there
    are no markers telling the solver to flip anything.

f) Visual cleanliness
  - Markers are confined to two pixels per duplicate hole; no overlays are drawn on top of holes or pieces.
  - The frame stays mostly monochrome, so markers stand out as localized clues rather than visual clutter.
  - Distractors remain isolated singleton pixels in one dedicated color and never touch meaningful objects.

g) Color inferability
  - Marker colors are copied directly from the intended piece's block colors; there is no hidden remapping.
  - The duplicate pair shares a palette, so the useful information is the arrangement of colors, which is
    visible from the piece itself and from the marker pair.
  - The number of colors per non-unit piece stays low enough that 2-3 examples are sufficient to infer the
    pattern-matching rule.

h) Post-change generator guarantees tied to main()
  - Every duplicate slot has exactly two marker constraints, and those constraints identify exactly one
    rotation-only candidate piece.
  - Every chiral pair contributes at least one dihedral-only false alternative, ensuring the no-flip rule is
    behaviorally active and not just syntactic.
  - All non-unit pieces are multicolor, so the D2 signal is present in every generated sample.
"""


ShapeKey = tuple[tuple[int, int], bytes]
MarkerSpec = tuple[int, int, int, int]


# --- shared helpers ----------------------------------------------------------

K_CHOICES = [2, 3]
LAYOUT_H_RANGE = (5, 12)
LAYOUT_W_RANGE = (5, 8)
N_SLOTS_RANGE = (5, 6)
N_UNITS = 1

NON_UNIT_SIZE_CHOICES = [3, 4, 5, 6, 7]
DUPLICATE_SIZE_CHOICES = [4, 5, 6]
CHIRAL_PAIR_SIZE_CHOICES = [4, 5, 6]

HOLE_DENSITY_RANGE = (0.20, 0.50)
LAYOUT_RETRIES = 140
SHAPE_RETRIES = 90
COLORING_RETRIES = 120

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 500


def _rotation_shape_key(mask: np.ndarray) -> ShapeKey:
    """Canonical key for a tight boolean mask under rotations only."""
    variants = []
    for kk in range(4):
        rot = np.rot90(mask, int(kk))
        variants.append((tuple(map(int, rot.shape)), rot.tobytes()))
    return min(variants)


def _dihedral_shape_key(mask: np.ndarray) -> ShapeKey:
    """Canonical key for a tight boolean mask under rotation plus flip."""
    variants = []
    for kk in range(4):
        rot = np.rot90(mask, int(kk))
        variants.append((tuple(map(int, rot.shape)), rot.tobytes()))
        flp = np.fliplr(rot)
        variants.append((tuple(map(int, flp.shape)), flp.tobytes()))
    return min(variants)


def _tight_mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, int(r0), int(c0)


def _slot_from_coords(coords: np.ndarray, role: str) -> dict:
    mask, r0, c0 = _tight_mask_from_coords(coords)
    return {
        "coords": coords.astype(int),
        "area": int(coords.shape[0]),
        "mask": mask,
        "r0": int(r0),
        "c0": int(c0),
        "rot_key": _rotation_shape_key(mask),
        "role": role,
        "marker_specs": [],
    }


def _rotation_orbit_size(mask: np.ndarray) -> int:
    seen: set[ShapeKey] = set()
    for kk in range(4):
        rot = np.rot90(mask, int(kk))
        seen.add((tuple(map(int, rot.shape)), rot.tobytes()))
    return int(len(seen))


def _is_chiral(mask: np.ndarray) -> bool:
    return _rotation_shape_key(mask) != _rotation_shape_key(np.fliplr(mask))


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[:-1, 1:] |= mask[1:, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    return out


def _forbid_coords(forbidden: np.ndarray, coords: np.ndarray) -> None:
    cell_mask = np.zeros_like(forbidden)
    cell_mask[coords[:, 0], coords[:, 1]] = True
    forbidden |= _dilate8(cell_mask)


def _place_mask_randomly(forbidden: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    h, w = map(int, mask.shape)
    H, W = map(int, forbidden.shape)
    if h > H or w > W:
        return None
    rel = np.argwhere(mask)
    positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(positions)
    for top, left in positions:
        coords = rel + np.array([top, left], dtype=int)
        if np.any(forbidden[coords[:, 0], coords[:, 1]]):
            continue
        _forbid_coords(forbidden, coords)
        return coords.astype(int)
    return None


def _random_shape_mask(
    target_size: int,
    *,
    require_chiral: bool = False,
    require_full_rotation_orbit: bool = False,
    used_rot_keys: set[ShapeKey] | None = None,
) -> np.ndarray | None:
    used_rot_keys = used_rot_keys or set()
    max_side = min(5, max(2, int(target_size)))

    for _ in range(SHAPE_RETRIES):
        box_h = random.randint(2, max_side)
        box_w = random.randint(2, max_side)
        if box_h * box_w < int(target_size):
            continue

        r0 = random.randint(0, box_h - 1)
        c0 = random.randint(0, box_w - 1)
        chosen = {(int(r0), int(c0))}
        frontier = {(int(r0 - 1), int(c0)), (int(r0 + 1), int(c0)), (int(r0), int(c0 - 1)), (int(r0), int(c0 + 1))}

        while len(chosen) < int(target_size):
            cands = [
                p for p in frontier
                if 0 <= p[0] < box_h and 0 <= p[1] < box_w and p not in chosen
            ]
            if not cands:
                break
            nr, nc = random.choice(cands)
            chosen.add((int(nr), int(nc)))
            frontier.discard((int(nr), int(nc)))
            frontier.update({(nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)})

        if len(chosen) != int(target_size):
            continue

        coords = np.array(sorted(chosen), dtype=int)
        mask, _r0, _c0 = _tight_mask_from_coords(coords)
        rot_key = _rotation_shape_key(mask)
        if rot_key in used_rot_keys:
            continue
        if require_chiral and not _is_chiral(mask):
            continue
        if require_full_rotation_orbit and _rotation_orbit_size(mask) < 4:
            continue
        if len(_unique_boundary_specs(mask)) < 2:
            continue
        return mask

    return None


def _layout_touches_all_borders(layout_mask: np.ndarray) -> bool:
    return bool(
        layout_mask[0].any()
        and layout_mask[-1].any()
        and layout_mask[:, 0].any()
        and layout_mask[:, -1].any()
    )


def _unique_boundary_specs(mask: np.ndarray) -> list[MarkerSpec]:
    """
    Return boundary descriptors `(rel_r, rel_c, dr, dc)` where the corresponding wall cell
    is 4-adjacent to exactly one occupied slot cell. These are safe places for color markers.
    """
    h, w = map(int, mask.shape)
    occupied = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    specs: list[MarkerSpec] = []
    for r, c in sorted(occupied):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (r + dr, c + dc) in occupied:
                continue
            wr = int(r + dr)
            wc = int(c + dc)
            adj = []
            for ddr, ddc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = int(wr + ddr)
                nc = int(wc + ddc)
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    adj.append((nr, nc))
            if len(adj) == 1:
                specs.append((int(r), int(c), int(dr), int(dc)))
    return specs


def _spec_wall_coord(slot: dict, spec: MarkerSpec) -> tuple[int, int]:
    rel_r, rel_c, dr, dc = map(int, spec)
    return int(slot["r0"] + rel_r + 1 + dr), int(slot["c0"] + rel_c + 1 + dc)


def _downsample_colored_blocks(patch: np.ndarray, mask: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray] | None:
    hh, ww = map(int, mask.shape)
    if hh % k != 0 or ww % k != 0:
        return None
    out_h = int(hh // k)
    out_w = int(ww // k)
    colors = np.zeros((out_h, out_w), dtype=int)
    out_mask = np.zeros((out_h, out_w), dtype=bool)
    for r in range(out_h):
        for c in range(out_w):
            block_mask = mask[r * k : (r + 1) * k, c * k : (c + 1) * k]
            block_patch = patch[r * k : (r + 1) * k, c * k : (c + 1) * k]
            if bool(block_mask.all()):
                vals = sorted(set(map(int, block_patch.flatten())))
                if len(vals) != 1 or vals[0] == 0:
                    return None
                out_mask[r, c] = True
                colors[r, c] = int(vals[0])
            elif bool(block_mask.any()):
                return None
            else:
                if np.any(block_patch != 0):
                    return None
    return colors, out_mask


def _rotation_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    uniq = []
    seen: set[tuple[ShapeKey, bytes]] = set()
    for kk in range(4):
        vp = np.rot90(patch, int(kk))
        vm = np.rot90(mask, int(kk))
        key = ((tuple(map(int, vm.shape)), vm.tobytes()), vp.tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((vp, vm))
    return uniq


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    uniq = []
    seen: set[tuple[ShapeKey, bytes]] = set()
    for kk in range(4):
        vp = np.rot90(patch, int(kk))
        vm = np.rot90(mask, int(kk))
        for p2, m2 in ((vp, vm), (np.fliplr(vp), np.fliplr(vm))):
            key = ((tuple(map(int, m2.shape)), m2.tobytes()), p2.tobytes())
            if key in seen:
                continue
            seen.add(key)
            uniq.append((p2, m2))
    return uniq


def _extract_structure(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain a frame component"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == int(bg)
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one bg hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slot_records = []
    for coords in components_mask(layout, connectivity=4):
        sm, rr0, cc0 = _tight_mask_from_coords(coords)
        rel_hole = {tuple(map(int, rc)) for rc in coords}
        marker_records = []
        for rr, cc in coords:
            sr = int(rr + lr0)
            sc = int(cc + lc0)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                wr = int(sr + dr)
                wc = int(sc + dc)
                if not (0 <= wr < sub.shape[0] and 0 <= wc < sub.shape[1]):
                    continue
                col = int(sub[wr, wc])
                if col == int(bg) or col == int(frame_color):
                    continue
                touched = []
                for ddr, ddc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = int(wr + ddr - lr0)
                    nc = int(wc + ddc - lc0)
                    if (nr, nc) in rel_hole:
                        touched.append((nr, nc))
                assert len(touched) == 1, "each marker must touch exactly one cell of its slot"
                tr, tc = touched[0]
                marker_records.append(
                    {
                        "wall_r": int(wr),
                        "wall_c": int(wc),
                        "rel_r": int(tr - rr0),
                        "rel_c": int(tc - cc0),
                        "color": int(col),
                    }
                )
        marker_records.sort(key=lambda d: (int(d["wall_r"]), int(d["wall_c"]), int(d["color"])))
        slot_records.append(
            {
                "area": int(sm.sum()),
                "r0": int(rr0),
                "c0": int(cc0),
                "mask": sm,
                "markers": marker_records,
            }
        )
    slot_records.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
    assert slot_records, "frame must contain at least one slot"

    pieces = []
    for comp in comps:
        if comp is frame_comp:
            continue
        coords = comp["coords"]
        if coords.size == 0 or int(coords.shape[0]) <= 1:
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        pm = np.zeros(patch.shape, dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append({"area": int(pm.sum()), "mask": pm, "patch": patch})
    assert pieces, "at least one non-frame piece must exist"

    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    k = int(k or 1)

    return {
        "grid": g,
        "bg": bg,
        "frame_color": frame_color,
        "layout": layout,
        "slots": slot_records,
        "pieces": pieces,
        "k": k,
    }


# --- solver ------------------------------------------------------------------

def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge the framed hole layout and fill each hole with its matching multicolor piece.

    Input structure:
    - The largest foreground component is a mostly monochrome rectangular frame enclosing background holes.
    - Outside the frame are k-by-k-block pieces; non-unit pieces use 2-3 colors, and some repeated-shape
      holes have small colored wall markers touching them.

    Rule:
    - Extract the hole components as slots, and compare outside pieces to slot shapes after block downsampling.
    - Legal matching uses rotations only; mirror flips are forbidden.
    - If several unused pieces share the same slot shape, resolve the tie by checking whether the candidate's
      block colors at the marked slot-adjacent positions match the wall-marker colors.

    Key insight:
    - Geometry still determines which rotated pieces can fit, but repeated hole shapes force a second check:
      the wall markers name specific block colors of the correctly rotated multicolor piece. Matching those
      local color probes breaks duplicate-shape ambiguity without ever allowing a flip.
    """
    data = _extract_structure(grid)
    frame_color = int(data["frame_color"])
    layout = data["layout"]
    slots = data["slots"]
    pieces = data["pieces"]
    k = int(data["k"])

    candidates = []
    for pi, piece in enumerate(pieces):
        for vp, vm in _rotation_variants(piece["patch"], piece["mask"]):
            down = _downsample_colored_blocks(vp, vm, k)
            if down is None:
                continue
            uc, um = down
            candidates.append(
                {
                    "pi": int(pi),
                    "area": int(piece["area"]),
                    "uc": uc,
                    "um": um,
                    "vp": vp,
                    "vm": vm,
                }
            )

    used_piece: set[int] = set()
    placements: list[tuple[dict, dict]] = []

    for slot in slots:
        need_area = int(slot["area"]) * int(k) * int(k)
        shape_hits = []
        for cand in candidates:
            if int(cand["pi"]) in used_piece:
                continue
            if int(cand["area"]) != int(need_area):
                continue
            if cand["um"].shape != slot["mask"].shape:
                continue
            if np.array_equal(cand["um"], slot["mask"]):
                shape_hits.append(cand)

        assert shape_hits, "every slot must match at least one piece under rotations only"

        marker_hits = shape_hits
        if slot["markers"]:
            marker_hits = []
            for cand in shape_hits:
                ok = True
                for marker in slot["markers"]:
                    if int(cand["uc"][int(marker["rel_r"]), int(marker["rel_c"])]) != int(marker["color"]):
                        ok = False
                        break
                if ok:
                    marker_hits.append(cand)
            assert marker_hits, "marked duplicate slots must be resolvable by color constraints"

        if len(marker_hits) > 1:
            assert not slot["markers"], "marker-constrained slots must become unique"
            assert len(shape_hits) == 1, "unmarked slots must be unique by rotation-shape alone"

        found = marker_hits[0]
        used_piece.add(int(found["pi"]))
        placements.append((slot, found))

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)

    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, cand["vp"], int(top), int(left), mask=cand["vm"], overwrite=True)

    return out


# --- generator ---------------------------------------------------------------

def _gen_layout(h: int, w: int, n_slots: int) -> tuple[list[dict], list[dict], np.ndarray] | None:
    """
    Build a layout with:
    - one duplicate non-unit shape pair (exact same rotation class),
    - one chiral mirror pair (different under rotation, same under flip),
    - optional extra unique non-unit shapes,
    - one unit slot for k inference.
    """
    n_non_unit = int(n_slots - N_UNITS)
    if n_non_unit < 4:
        return None

    for _ in range(LAYOUT_RETRIES):
        forbidden = np.zeros((h, w), dtype=bool)
        unit_slots: list[dict] = []
        non_unit_slots: list[dict] = []
        used_rot_keys: set[ShapeKey] = set()

        dup_mask = _random_shape_mask(
            random.choice(DUPLICATE_SIZE_CHOICES),
            require_full_rotation_orbit=True,
            used_rot_keys=used_rot_keys,
        )
        if dup_mask is None:
            continue
        dup_specs = _unique_boundary_specs(dup_mask)
        dup_cells = sorted({(int(r), int(c)) for r, c, _dr, _dc in dup_specs})
        if len(dup_cells) < 2:
            continue
        dup_key = _rotation_shape_key(dup_mask)

        coords_a = _place_mask_randomly(forbidden, dup_mask)
        if coords_a is None:
            continue
        coords_b = _place_mask_randomly(forbidden, dup_mask)
        if coords_b is None:
            continue
        non_unit_slots.append(_slot_from_coords(coords_a, role="dup_a"))
        non_unit_slots.append(_slot_from_coords(coords_b, role="dup_b"))
        used_rot_keys.add(dup_key)

        chiral_mask = _random_shape_mask(
            random.choice(CHIRAL_PAIR_SIZE_CHOICES),
            require_chiral=True,
            require_full_rotation_orbit=True,
            used_rot_keys=used_rot_keys,
        )
        if chiral_mask is None:
            continue
        mirror_mask = np.fliplr(chiral_mask)
        mirror_key = _rotation_shape_key(mirror_mask)
        if mirror_key == _rotation_shape_key(chiral_mask) or mirror_key in used_rot_keys:
            continue
        coords_c = _place_mask_randomly(forbidden, chiral_mask)
        if coords_c is None:
            continue
        coords_d = _place_mask_randomly(forbidden, mirror_mask)
        if coords_d is None:
            continue
        non_unit_slots.append(_slot_from_coords(coords_c, role="chiral_a"))
        non_unit_slots.append(_slot_from_coords(coords_d, role="chiral_b"))
        used_rot_keys.add(_rotation_shape_key(chiral_mask))
        used_rot_keys.add(mirror_key)

        extra_count = int(n_non_unit - 4)
        ok = True
        for _extra_idx in range(extra_count):
            mask = _random_shape_mask(
                random.choice(NON_UNIT_SIZE_CHOICES),
                require_full_rotation_orbit=False,
                used_rot_keys=used_rot_keys,
            )
            if mask is None:
                ok = False
                break
            coords = _place_mask_randomly(forbidden, mask)
            if coords is None:
                ok = False
                break
            slot = _slot_from_coords(coords, role="other")
            used_rot_keys.add(slot["rot_key"])
            non_unit_slots.append(slot)
        if not ok:
            continue

        unit_coords = _place_mask_randomly(forbidden, np.ones((1, 1), dtype=bool))
        if unit_coords is None:
            continue
        unit_slots.append(_slot_from_coords(unit_coords, role="unit"))

        layout_mask = np.zeros((h, w), dtype=bool)
        for slot in unit_slots + non_unit_slots:
            layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True

        density = float(layout_mask.sum()) / float(h * w)
        lo, hi = HOLE_DENSITY_RANGE
        if not (lo <= density <= hi):
            continue
        if not _layout_touches_all_borders(layout_mask):
            continue

        return unit_slots, non_unit_slots, layout_mask

    return None


def _random_color_grid(mask: np.ndarray, palette: list[int]) -> np.ndarray:
    coords = np.argwhere(mask)
    n_colors = random.randint(2, min(3, len(palette)))
    use_colors = random.sample(palette, n_colors)
    grid = np.zeros(mask.shape, dtype=int)
    while True:
        picks = [random.choice(use_colors) for _ in range(int(coords.shape[0]))]
        if len(set(map(int, picks))) >= 2:
            break
    for (r, c), col in zip(coords, picks):
        grid[int(r), int(c)] = int(col)
    return grid


def _rotation_color_key(color_grid: np.ndarray, mask: np.ndarray) -> tuple[ShapeKey, bytes]:
    best = None
    for kk in range(4):
        rot_mask = np.rot90(mask, int(kk))
        rot_grid = np.rot90(color_grid, int(kk))
        key = ((tuple(map(int, rot_mask.shape)), rot_mask.tobytes()), rot_grid.tobytes())
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def _choose_duplicate_markers(mask: np.ndarray, grid_a: np.ndarray, grid_b: np.ndarray) -> list[MarkerSpec] | None:
    specs = _unique_boundary_specs(mask)
    by_cell: dict[tuple[int, int], list[MarkerSpec]] = {}
    for spec in specs:
        cell = (int(spec[0]), int(spec[1]))
        by_cell.setdefault(cell, []).append(spec)
    cell_pairs = list(combinations(sorted(by_cell), 2))
    random.shuffle(cell_pairs)
    for cell_a, cell_b in cell_pairs:
        sig_a = (int(grid_a[cell_a]), int(grid_a[cell_b]))
        sig_b = (int(grid_b[cell_a]), int(grid_b[cell_b]))
        if sig_a != sig_b:
            return [by_cell[cell_a][0], by_cell[cell_b][0]]
    return None


def _make_piece(slot: dict, color_grid: np.ndarray, k: int, apply_rotation: bool) -> dict:
    out_patch = np.kron(color_grid, np.ones((k, k), dtype=int))
    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    if apply_rotation and int(slot["area"]) > 1:
        rot = random.randint(0, 3)
        in_patch = np.rot90(out_patch, int(rot))
        in_mask = np.rot90(out_mask, int(rot))
    else:
        in_patch = out_patch
        in_mask = out_mask
    return {
        "slot": slot,
        "out_colors": color_grid.astype(int),
        "out_patch": out_patch.astype(int),
        "out_mask": out_mask,
        "in_patch": in_patch.astype(int),
        "in_mask": in_mask,
    }


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for piece in pieces:
        coords = piece["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * int(k)
        left = (int(coords[:, 1].min()) + 1) * int(k)
        paste(out, piece["out_patch"], int(top), int(left), mask=piece["out_mask"], overwrite=True)
    return out


def _build_frame_image(
    layout_mask: np.ndarray,
    slots: list[dict],
    piece_by_slot: dict[int, dict],
    frame_color: int,
) -> np.ndarray | None:
    frame_img = np.full((layout_mask.shape[0] + 2, layout_mask.shape[1] + 2), int(frame_color), dtype=int)
    frame_img[1:-1, 1:-1] = np.where(layout_mask, 0, int(frame_color))

    used_wall_cells: set[tuple[int, int]] = set()
    for slot in slots:
        piece = piece_by_slot[id(slot)]
        for spec in slot["marker_specs"]:
            wr, wc = _spec_wall_coord(slot, spec)
            if not (0 <= wr < frame_img.shape[0] and 0 <= wc < frame_img.shape[1]):
                return None
            if frame_img[wr, wc] != int(frame_color):
                return None
            if (int(wr), int(wc)) in used_wall_cells:
                return None
            used_wall_cells.add((int(wr), int(wc)))
            frame_img[wr, wc] = int(piece["out_colors"][int(spec[0]), int(spec[1])])

    return frame_img


def _build_input_canvas(pieces: list[dict], frame_img: np.ndarray, distractor_color: int) -> np.ndarray | None:
    frame_h, frame_w = map(int, frame_img.shape)
    total_piece_area = sum(int(piece["in_mask"].sum()) for piece in pieces)
    needed_cells = frame_h * frame_w + AREA_BUFFER_MULT * total_piece_area + AREA_BUFFER_CONST

    min_h = max(CANVAS_DIM_RANGE[0], frame_h + 2)
    min_w = max(CANVAS_DIM_RANGE[0], frame_w + 2)
    canvas_h = random.randint(min_h, CANVAS_DIM_RANGE[1])
    canvas_w = random.randint(min_w, CANVAS_DIM_RANGE[1])
    while canvas_h * canvas_w < needed_cells and (
        canvas_h < CANVAS_DIM_RANGE[1] or canvas_w < CANVAS_DIM_RANGE[1]
    ):
        if canvas_h <= canvas_w and canvas_h < CANVAS_DIM_RANGE[1]:
            canvas_h += 1
        elif canvas_w < CANVAS_DIM_RANGE[1]:
            canvas_w += 1
        else:
            break
    if canvas_h * canvas_w < needed_cells:
        return None

    canvas = np.zeros((canvas_h, canvas_w), dtype=int)
    fr_max = canvas_h - frame_h - 1
    fc_max = canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

    for piece in sorted(pieces, key=lambda p: -int(p["in_mask"].sum())):
        try:
            pr, pc = random_free_location(
                canvas,
                piece["in_patch"],
                transparent=0,
                border=0,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, piece["in_patch"], int(pr), int(pc), mask=piece["in_mask"], overwrite=True)

    n_distractors = random.randint(*DISTRACTOR_RANGE)
    placed = 0
    for _ in range(n_distractors * DISTRACTOR_RETRY_MULT):
        if placed >= n_distractors:
            break
        rr = random.randint(0, canvas_h - 1)
        cc = random.randint(0, canvas_w - 1)
        if canvas[rr, cc] != 0:
            continue
        r_lo, r_hi = max(0, rr - 1), min(canvas_h, rr + 2)
        c_lo, c_hi = max(0, cc - 1), min(canvas_w, cc + 2)
        if np.any(canvas[r_lo:r_hi, c_lo:c_hi] != 0):
            continue
        canvas[rr, cc] = int(distractor_color)
        placed += 1
    if placed < MIN_DISTRACTORS:
        return None

    vals, counts = np.unique(canvas, return_counts=True)
    if int(vals[np.argmax(counts)]) != 0:
        return None
    return canvas


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = int(palette.pop())
    distractor_color = int(palette.pop())
    piece_palette = palette[:]
    if len(piece_palette) < 4:
        return None

    layout_res = _gen_layout(layout_h, layout_w, n_slots)
    if layout_res is None:
        return None
    unit_slots, non_unit_slots, layout_mask = layout_res

    dup_slots = [slot for slot in non_unit_slots if slot["role"].startswith("dup_")]
    if len(dup_slots) != 2:
        return None
    dup_mask = dup_slots[0]["mask"]

    dup_palette = random.sample(piece_palette, random.randint(2, min(3, len(piece_palette))))
    marker_specs = None
    dup_color_a = None
    dup_color_b = None
    for _ in range(COLORING_RETRIES):
        grid_a = _random_color_grid(dup_mask, dup_palette)
        grid_b = _random_color_grid(dup_mask, dup_palette)
        if _rotation_color_key(grid_a, dup_mask) == _rotation_color_key(grid_b, dup_mask):
            continue
        specs = _choose_duplicate_markers(dup_mask, grid_a, grid_b)
        if specs is None:
            continue
        marker_specs = specs
        dup_color_a = grid_a
        dup_color_b = grid_b
        break
    if marker_specs is None or dup_color_a is None or dup_color_b is None:
        return None

    dup_slots[0]["marker_specs"] = list(marker_specs)
    dup_slots[1]["marker_specs"] = list(marker_specs)

    piece_specs: list[tuple[dict, np.ndarray, bool]] = []
    piece_specs.append((dup_slots[0], dup_color_a, True))
    piece_specs.append((dup_slots[1], dup_color_b, True))

    for slot in non_unit_slots:
        if slot["role"].startswith("dup_"):
            continue
        grid = _random_color_grid(slot["mask"], piece_palette)
        piece_specs.append((slot, grid, True))

    unit_color = random.choice(piece_palette)
    for slot in unit_slots:
        unit_grid = np.full(slot["mask"].shape, int(unit_color), dtype=int)
        piece_specs.append((slot, unit_grid, False))

    pieces = [_make_piece(slot, color_grid, k, apply_rotation=apply_rotation) for slot, color_grid, apply_rotation in piece_specs]
    piece_by_slot = {id(piece["slot"]): piece for piece in pieces}

    frame_area = int((layout_h + 2) * (layout_w + 2) - layout_mask.sum())
    max_piece_area = int(max(int(piece["in_mask"].sum()) for piece in pieces))
    if frame_area <= max_piece_area:
        return None

    frame_img = _build_frame_image(layout_mask, unit_slots + non_unit_slots, piece_by_slot, frame_color)
    if frame_img is None:
        return None

    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
    canvas = _build_input_canvas(pieces, frame_img, distractor_color)
    if canvas is None:
        return None

    metrics = behavior_metrics(canvas)
    if int(metrics["duplicate_ambiguous_slots"]) < 2:
        return None
    if int(metrics["marker_resolved_slots"]) < 2:
        return None
    if int(metrics["flip_only_slots"]) < 1:
        return None
    if int(metrics["multicolor_nonunit_pieces"]) < len(non_unit_slots):
        return None

    try:
        produced = main(canvas)
    except Exception:
        return None
    if not np.array_equal(produced, expected_out):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError(f"generate_input: failed to generate after {GENERATE_MAX_RETRIES} attempts")


# --- behavior inspection for validation -------------------------------------

def behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify whether a generated sample actually exercises the combo.

    Metrics:
    - duplicate_ambiguous_slots: slots with >1 piece match under rotation-only shape.
    - marker_resolved_slots: ambiguous slots where markers leave exactly one piece.
    - flip_only_slots: slots that gain extra piece options if flips are allowed.
    - multicolor_nonunit_pieces: extracted pieces whose downsampled block palette has >=2 colors.
    """
    data = _extract_structure(grid)
    slots = data["slots"]
    pieces = data["pieces"]
    k = int(data["k"])

    rot_piece_hits: list[set[int]] = []
    mark_piece_hits: list[set[int]] = []
    flip_piece_hits: list[set[int]] = []
    multicolor_nonunit = 0

    rot_candidates_by_piece: dict[int, list[dict]] = {}
    dihedral_candidates_by_piece: dict[int, list[dict]] = {}

    for pi, piece in enumerate(pieces):
        rot_candidates = []
        for vp, vm in _rotation_variants(piece["patch"], piece["mask"]):
            down = _downsample_colored_blocks(vp, vm, k)
            if down is None:
                continue
            uc, um = down
            rot_candidates.append({"pi": int(pi), "area": int(piece["area"]), "uc": uc, "um": um})
        rot_candidates_by_piece[int(pi)] = rot_candidates

        if any(len(set(map(int, cand["uc"][cand["um"]].flatten()))) >= 2 for cand in rot_candidates):
            multicolor_nonunit += 1

        dihedral_candidates = []
        for vp, vm in _dihedral_variants(piece["patch"], piece["mask"]):
            down = _downsample_colored_blocks(vp, vm, k)
            if down is None:
                continue
            uc, um = down
            dihedral_candidates.append({"pi": int(pi), "area": int(piece["area"]), "uc": uc, "um": um})
        dihedral_candidates_by_piece[int(pi)] = dihedral_candidates

    for slot in slots:
        need_area = int(slot["area"]) * int(k) * int(k)
        rot_hits: set[int] = set()
        marker_hits: set[int] = set()
        flip_hits: set[int] = set()

        for pi, cands in rot_candidates_by_piece.items():
            hit = False
            marker_ok = False
            for cand in cands:
                if int(cand["area"]) != int(need_area):
                    continue
                if cand["um"].shape != slot["mask"].shape:
                    continue
                if not np.array_equal(cand["um"], slot["mask"]):
                    continue
                hit = True
                ok = True
                for marker in slot["markers"]:
                    if int(cand["uc"][int(marker["rel_r"]), int(marker["rel_c"])]) != int(marker["color"]):
                        ok = False
                        break
                if ok:
                    marker_ok = True
            if hit:
                rot_hits.add(int(pi))
            if marker_ok:
                marker_hits.add(int(pi))

        for pi, cands in dihedral_candidates_by_piece.items():
            for cand in cands:
                if int(cand["area"]) != int(need_area):
                    continue
                if cand["um"].shape != slot["mask"].shape:
                    continue
                if np.array_equal(cand["um"], slot["mask"]):
                    flip_hits.add(int(pi))
                    break

        rot_piece_hits.append(rot_hits)
        mark_piece_hits.append(marker_hits)
        flip_piece_hits.append(flip_hits)

    duplicate_ambiguous_slots = sum(1 for hits in rot_piece_hits if len(hits) > 1)
    marker_resolved_slots = sum(1 for rot_hits, mark_hits in zip(rot_piece_hits, mark_piece_hits) if len(rot_hits) > 1 and len(mark_hits) == 1)
    flip_only_slots = sum(1 for rot_hits, flip_hits in zip(rot_piece_hits, flip_piece_hits) if len(flip_hits - rot_hits) > 0)

    return {
        "duplicate_ambiguous_slots": int(duplicate_ambiguous_slots),
        "marker_resolved_slots": int(marker_resolved_slots),
        "flip_only_slots": int(flip_only_slots),
        "multicolor_nonunit_pieces": int(multicolor_nonunit),
    }
