# Combo variant of 898e7135 (index 22)
# combo_spec: D1/1+D14/1
# Name: 标记引导旋转

from __future__ import annotations

from collections import Counter
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
Design analysis for combo D1/1 + D14/1
======================================

spec_variant:
  preserved_backbone:
    - input is still a bg-majority canvas with one large rectangular frame and several scattered pieces
    - the frame still contains a hole-layout that is later scaled by k in the output
    - each outside piece is still a k-by-k block upscaling of one hole region
    - output is still the cropped frame layout enlarged by k and filled with the matched pieces

  fusion_change:
    - D1/1 removes flips: only 0/90/180/270 degree rotations are legal
    - D14/1 adds one red marker cell inside every slot and one red marker pixel inside every piece
    - the marker is not decorative: for same-shape slots, the marker orbit identifies which slot a piece belongs to,
      and for every matched pair the marker position determines which rotation must be undone

  concrete_behavior_trace:
    example:
      base slot S:
        shape: a chiral L-like polyomino
        duplicate markers: one slot marks the elbow cell, another marks the tip cell
      mirror slot M:
        shape: horizontal mirror of S, with its own marker
      pieces:
        each slot is expanded to k-by-k blocks; one pixel in the marked block becomes red
        then each piece is rotated randomly before being scattered outside the frame
    original task:
      a solver could compare each piece against every slot under the full dihedral group and stop at the first shape match
    fused variant:
      - S and M are no longer equivalent, because flips are forbidden
      - the two copies of S are still shape-identical, so shape alone cannot decide placement
      - after the solver groups piece and slot by rotation-class shape, it must read the marked block to recover
        the unique legal rotation and the correct duplicate target
    visible difference:
      humans can see both signals at once: chirality matters, and the red marker tells which turned copy goes where

  complexity_calibration:
    - if we only removed flips, many instances would still look like the original because all shapes could remain unique
    - if we only added markers without repeated shapes, the marker could be ignored whenever the silhouette already fixes the slot
    - the generator therefore forces:
      1. a duplicate pair of slots with the same rotation-class shape but different marker orbits
      2. a mirror partner of that same chiral shape, so "no flip" is also observable
      3. at least one non-zero piece rotation, so the marker actually needs to be interpreted
    - the duplicate markers are chosen from different rotation orbits, preventing two same-shape pieces from becoming
      interchangeable after rotation

  implicit_assumption_audit_for_main:
    - `find_components(..., monochromatic=False)`:
      assumption: the largest foreground component is the frame together with the in-frame markers.
      generator guarantee: frame area is checked to exceed every piece area, and objects are separated by background.
    - `hole_like = (sub == bg) | (sub == MARKER_COLOR)`:
      assumption: marker cells live inside slots and must count as hole territory.
      generator guarantee: each slot receives exactly one red cell and no wall cell does.
    - slot extraction:
      assumption: every slot has exactly one marker cell.
      generator guarantee: layout construction assigns one and only one marker per slot.
    - k inference:
      original assumption: unit slots reveal k via the smallest solid square piece.
      fusion status: invalid, because unit slots would make the rotation marker meaningless.
      new rule: try k in {2, 3} and keep only full one-to-one matchings.
    - matching:
      assumption: shape matching must respect chirality and marker orbit.
      solver action: it first filters by rotation-class shape, then finds the unique rotation whose transformed marker
      lands on the slot marker; flips are never generated.
    - duplicate handling:
      assumption: same-shape duplicates are not ambiguous.
      generator guarantee: duplicate slots use distinct marked-state keys, and the oracle rejects examples where a wrong k
      also yields a complete matching.

  placement_legality:
    - layout slots are placed with 8-neighbor exclusion, so slot markers never merge across slots
    - scattered pieces use `random_free_location(..., padding=1, padding_connectivity=8)`, so pieces stay separated from
      the frame and from each other
    - marker pixels are painted inside already-occupied piece blocks, so they never alter the piece silhouette
    - all candidates are rejected if density, border-touch, frame-dominance, or oracle checks fail

  visual_observability:
    - the red marker has one job only: indicate the alignment block
    - a repeated silhouette with two different red positions makes it obvious that color order is not the rule; the marker is
      the rule
    - the mirrored sibling of that silhouette shows that left-handed and right-handed copies are distinct, so flips are not allowed

  cleanliness:
    - only one extra overlay layer is introduced: a single red cell/pixel per object
    - markers are sparse and isolated inside their host objects, so the grid stays readable
    - distractors remain singleton noise in a separate color and never touch meaningful structures
"""


MARKER_COLOR = 2
BG_COLOR = 0
K_CHOICES = (2, 3)

LAYOUT_H_RANGE = (5, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
SLOT_SIZE_CHOICES = [3, 4, 5, 6]
CHIRAL_CORE_SIZE_CHOICES = [3, 4, 5, 6]

HOLE_DENSITY_RANGE = (0.20, 0.50)
SHAPE_RETRIES = 80
LAYOUT_RETRIES = 180

CANVAS_DIM_RANGE = (16, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 60

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 12
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 500

RotKey = tuple[tuple[int, int], bytes]
MarkedStateKey = tuple[tuple[int, int], bytes, int, int]


def _rotation_shape_key(mask: np.ndarray) -> RotKey:
    variants = []
    for rot in range(4):
        rm = np.rot90(mask, int(rot))
        variants.append((tuple(map(int, rm.shape)), rm.tobytes()))
    return min(variants)


def _rotate_marker(marker: tuple[int, int], shape: tuple[int, int], rot: int) -> tuple[int, int]:
    mr, mc = map(int, marker)
    h, w = map(int, shape)
    rot = int(rot) % 4
    if rot == 0:
        return mr, mc
    if rot == 1:
        return w - 1 - mc, mr
    if rot == 2:
        return h - 1 - mr, w - 1 - mc
    return mc, h - 1 - mr


def _marked_state_key(mask: np.ndarray, marker: tuple[int, int]) -> MarkedStateKey:
    states = []
    for rot in range(4):
        rm = np.rot90(mask, int(rot))
        rr, cc = _rotate_marker(marker, mask.shape, rot)
        states.append((tuple(map(int, rm.shape)), rm.tobytes(), int(rr), int(cc)))
    return min(states)


def _rotation_state_count(mask: np.ndarray, marker: tuple[int, int]) -> int:
    states = set()
    for rot in range(4):
        rm = np.rot90(mask, int(rot))
        rr, cc = _rotate_marker(marker, mask.shape, rot)
        states.add((tuple(map(int, rm.shape)), rm.tobytes(), int(rr), int(cc)))
    return len(states)


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = map(int, mask.shape)
    k = int(k)
    if hh % k != 0 or ww % k != 0:
        return None
    out = np.zeros((hh // k, ww // k), dtype=bool)
    for r in range(hh // k):
        for c in range(ww // k):
            block = mask[r * k : (r + 1) * k, c * k : (c + 1) * k]
            if bool(block.all()):
                out[r, c] = True
            elif bool(block.any()):
                return None
    return out


def _marker_block_from_patch(patch: np.ndarray, mask: np.ndarray, k: int) -> tuple[int, int] | None:
    coords = np.argwhere((patch == MARKER_COLOR) & mask)
    if coords.shape[0] != 1:
        return None
    rr, cc = map(int, coords[0])
    return rr // int(k), cc // int(k)


def _find_rotation_to_slot(
    piece_shape: np.ndarray,
    piece_marker: tuple[int, int],
    slot_shape: np.ndarray,
    slot_marker: tuple[int, int],
) -> int | None:
    for rot in range(4):
        rs = np.rot90(piece_shape, int(rot))
        if rs.shape != slot_shape.shape or not np.array_equal(rs, slot_shape):
            continue
        rm = _rotate_marker(piece_marker, piece_shape.shape, rot)
        if rm == slot_marker:
            return int(rot)
    return None


def _match_slots(slots: list[dict], pieces: list[dict], k: int) -> list[tuple[dict, dict]] | None:
    candidates_by_slot: list[list[dict]] = []
    for slot in slots:
        want_area = int(slot["area"]) * int(k) * int(k)
        cur: list[dict] = []
        for pi, piece in enumerate(pieces):
            if int(piece["area"]) != want_area:
                continue
            piece_shape = _downsample_blocks(piece["mask"], int(k))
            if piece_shape is None:
                continue
            if _rotation_shape_key(piece_shape) != _rotation_shape_key(slot["mask"]):
                continue
            piece_marker = _marker_block_from_patch(piece["patch"], piece["mask"], int(k))
            if piece_marker is None:
                continue
            rot = _find_rotation_to_slot(piece_shape, piece_marker, slot["mask"], slot["marker"])
            if rot is None:
                continue
            cur.append(
                {
                    "pi": int(pi),
                    "rot": int(rot),
                    "vp": np.rot90(piece["patch"], int(rot)),
                    "vm": np.rot90(piece["mask"], int(rot)),
                }
            )
        if not cur:
            return None
        candidates_by_slot.append(cur)

    order = sorted(
        range(len(slots)),
        key=lambda i: (
            len(candidates_by_slot[i]),
            -int(slots[i]["area"]),
            int(slots[i]["r0"]),
            int(slots[i]["c0"]),
        ),
    )

    used: set[int] = set()
    ans: list[tuple[dict, dict] | None] = [None] * len(slots)

    def dfs(idx: int) -> bool:
        if idx == len(order):
            return True
        si = order[idx]
        for cand in candidates_by_slot[si]:
            pi = int(cand["pi"])
            if pi in used:
                continue
            used.add(pi)
            ans[si] = (slots[si], cand)
            if dfs(idx + 1):
                return True
            ans[si] = None
            used.remove(pi)
        return False

    if not dfs(0):
        return None
    return [x for x in ans if x is not None]


def _extract_slots_from_frame(g: np.ndarray, bg: int, frame_comp: dict) -> tuple[int, np.ndarray, list[dict]]:
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])
    frame_cols = [
        int(g[int(r), int(c)])
        for r, c in frame_comp["coords"]
        if int(g[int(r), int(c)]) != MARKER_COLOR
    ]
    assert frame_cols, "frame must contain non-marker cells"
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole_like = (sub == int(bg)) | (sub == MARKER_COLOR)
    marker_mask = sub == MARKER_COLOR
    hole_box = bbox(hole_like)
    assert hole_box is not None, "frame must enclose at least one marked hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole_like[lr0 : lr1 + 1, lc0 : lc1 + 1]
    layout_markers = marker_mask[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        sm = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        marker_coords = [(int(r), int(c)) for r, c in coords if bool(layout_markers[int(r), int(c)])]
        assert len(marker_coords) == 1, "each slot must have exactly one marker cell"
        mr, mc = marker_coords[0]
        slots.append(
            {
                "area": int(sm.sum()),
                "r0": int(rr0),
                "c0": int(cc0),
                "mask": sm,
                "marker": (int(mr - rr0), int(mc - cc0)),
                "shape_key": _rotation_shape_key(sm),
                "marked_key": _marked_state_key(sm, (int(mr - rr0), int(mc - cc0))),
            }
        )
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
    assert slots, "frame must contain at least one slot"
    return frame_color, layout, slots


def _extract_pieces(g: np.ndarray, bg: int, frame_comp: dict) -> list[dict]:
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
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
        if int(np.sum((patch == MARKER_COLOR) & pm)) != 1:
            continue
        pieces.append({"area": int(pm.sum()), "mask": pm, "patch": patch})
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge a marked frame layout and refill its slots with the correctly rotated marked pieces.

    Input structure:
    - The most common color is background; the largest foreground component is a rectangular frame whose holes may contain one red marker cell each.
    - Outside the frame sit block-scaled polyomino pieces, each carrying exactly one red marker pixel inside one occupied block.

    Rule:
    - Treat both background cells and red marker cells inside the frame as slot territory, then extract each slot's shape and marked cell.
    - For each external piece, try k in {2,3}, recover its block-level shape, and use rotations only to align both shape and marker with one slot.
    - Paste the uniquely matched, re-rotated pieces into the k-scaled frame layout, preserving the red marker pixels in the output.

    Key insight:
    - The red marker tells how a piece must be rotated within its rotation-class shape, while mirror images remain illegal because flips are forbidden.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "expected a frame component"
    frame_comp = max(comps, key=lambda d: int(d["area"]))

    frame_color, layout, slots = _extract_slots_from_frame(g, bg, frame_comp)
    pieces = _extract_pieces(g, bg, frame_comp)
    assert len(pieces) == len(slots), "need one marked piece per slot"

    placements = None
    chosen_k = None
    for k in K_CHOICES:
        cand = _match_slots(slots, pieces, int(k))
        if cand is not None:
            placements = cand
            chosen_k = int(k)
            break
    assert placements is not None and chosen_k is not None, "no valid k / rotation-only matching found"

    out_h = (int(layout.shape[0]) + 2) * int(chosen_k)
    out_w = (int(layout.shape[1]) + 2) * int(chosen_k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)

    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * int(chosen_k)
        left = (int(slot["c0"]) + 1) * int(chosen_k)
        paste(out, cand["vp"], int(top), int(left), mask=cand["vm"], overwrite=True)

    return out


def _tight_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _slot_dict(coords: np.ndarray, marker_rel: tuple[int, int], tag: str) -> dict:
    mask = _tight_mask_from_coords(coords)
    r0, c0, _r1, _c1 = map(int, bbox_coords(coords))
    marker_rel = (int(marker_rel[0]), int(marker_rel[1]))
    return {
        "coords": coords.astype(int),
        "area": int(coords.shape[0]),
        "mask": mask,
        "r0": int(r0),
        "c0": int(c0),
        "marker_rel": marker_rel,
        "marker": marker_rel,
        "marker_abs": (int(r0 + marker_rel[0]), int(c0 + marker_rel[1])),
        "shape_key": _rotation_shape_key(mask),
        "marked_key": _marked_state_key(mask, marker_rel),
        "tag": tag,
    }


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


def _random_shape_mask(
    target_size: int,
    *,
    require_chiral: bool,
    banned_rot_keys: set[RotKey] | None = None,
) -> np.ndarray | None:
    banned_rot_keys = banned_rot_keys or set()
    max_side = min(5, max(2, int(target_size)))

    for _ in range(SHAPE_RETRIES):
        box_h = random.randint(2, max_side)
        box_w = random.randint(2, max_side)
        if box_h * box_w < int(target_size):
            continue

        start = (random.randint(0, box_h - 1), random.randint(0, box_w - 1))
        chosen = {start}
        frontier = {
            (start[0] - 1, start[1]),
            (start[0] + 1, start[1]),
            (start[0], start[1] - 1),
            (start[0], start[1] + 1),
        }
        while len(chosen) < int(target_size):
            cands = [p for p in frontier if 0 <= p[0] < box_h and 0 <= p[1] < box_w and p not in chosen]
            if not cands:
                break
            nr, nc = random.choice(cands)
            chosen.add((nr, nc))
            frontier.discard((nr, nc))
            frontier.update({(nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)})
        if len(chosen) != int(target_size):
            continue

        coords = np.array(sorted(chosen), dtype=int)
        mask = _tight_mask_from_coords(coords)
        key = _rotation_shape_key(mask)
        if key in banned_rot_keys:
            continue
        if require_chiral and not _is_chiral(mask):
            continue
        return mask
    return None


def _place_mask_randomly(forbidden: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    h, w = map(int, mask.shape)
    H, W = map(int, forbidden.shape)
    if h > H or w > W:
        return None

    coords_local = np.argwhere(mask)
    positions = [(r0, c0) for r0 in range(H - h + 1) for c0 in range(W - w + 1)]
    random.shuffle(positions)
    for top, left in positions:
        coords = coords_local + np.array([top, left], dtype=int)
        if bool(forbidden[coords[:, 0], coords[:, 1]].any()):
            continue
        _forbid_coords(forbidden, coords)
        return coords.astype(int)
    return None


def _marker_orbits(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    groups: dict[MarkedStateKey, list[tuple[int, int]]] = {}
    for r, c in np.argwhere(mask):
        marker = (int(r), int(c))
        groups.setdefault(_marked_state_key(mask, marker), []).append(marker)
    return list(groups.values())


def _choose_duplicate_markers(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]] | None:
    orbits = [orbit for orbit in _marker_orbits(mask) if orbit]
    if len(orbits) < 2:
        return None
    random.shuffle(orbits)
    first = random.choice(orbits[0])
    second = random.choice(orbits[1])
    return first, second


def _choose_any_marker(mask: np.ndarray) -> tuple[int, int]:
    pts = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return random.choice(pts)


def _layout_touches_all_borders(layout_mask: np.ndarray) -> bool:
    return bool(
        layout_mask[0].any()
        and layout_mask[-1].any()
        and layout_mask[:, 0].any()
        and layout_mask[:, -1].any()
    )


def _gen_layout(h: int, w: int, n_slots: int) -> tuple[list[dict], np.ndarray, np.ndarray] | None:
    for _ in range(LAYOUT_RETRIES):
        forbidden = np.zeros((h, w), dtype=bool)
        slots: list[dict] = []
        banned_rot_keys: set[RotKey] = set()
        used_marked_keys: set[MarkedStateKey] = set()

        base_mask = None
        dup_markers = None
        mirror_mask = None
        mirror_marker = None
        for _ in range(SHAPE_RETRIES):
            cand = _random_shape_mask(
                random.choice(CHIRAL_CORE_SIZE_CHOICES),
                require_chiral=True,
                banned_rot_keys=set(),
            )
            if cand is None:
                continue
            markers = _choose_duplicate_markers(cand)
            if markers is None:
                continue
            mirror = np.fliplr(cand)
            if _rotation_shape_key(cand) == _rotation_shape_key(mirror):
                continue
            m_marker = _choose_any_marker(mirror)
            base_mask = cand
            dup_markers = markers
            mirror_mask = mirror
            mirror_marker = m_marker
            break
        if base_mask is None or dup_markers is None or mirror_mask is None or mirror_marker is None:
            continue

        templates = [
            {"mask": base_mask, "marker_rel": dup_markers[0], "tag": "dup"},
            {"mask": base_mask, "marker_rel": dup_markers[1], "tag": "dup"},
            {"mask": mirror_mask, "marker_rel": mirror_marker, "tag": "mirror"},
        ]
        banned_rot_keys.add(_rotation_shape_key(base_mask))
        banned_rot_keys.add(_rotation_shape_key(mirror_mask))
        used_marked_keys.add(_marked_state_key(base_mask, dup_markers[0]))
        used_marked_keys.add(_marked_state_key(base_mask, dup_markers[1]))
        used_marked_keys.add(_marked_state_key(mirror_mask, mirror_marker))

        ok = True
        for _ in range(n_slots - 3):
            extra_mask = _random_shape_mask(
                random.choice(SLOT_SIZE_CHOICES),
                require_chiral=False,
                banned_rot_keys=banned_rot_keys,
            )
            if extra_mask is None:
                ok = False
                break
            orbit_reps = [random.choice(orbit) for orbit in _marker_orbits(extra_mask) if orbit]
            random.shuffle(orbit_reps)
            chosen_marker = None
            for marker in orbit_reps:
                marked_key = _marked_state_key(extra_mask, marker)
                if marked_key in used_marked_keys:
                    continue
                chosen_marker = marker
                used_marked_keys.add(marked_key)
                break
            if chosen_marker is None:
                ok = False
                break
            templates.append({"mask": extra_mask, "marker_rel": chosen_marker, "tag": "extra"})
            banned_rot_keys.add(_rotation_shape_key(extra_mask))
        if not ok:
            continue

        templates.sort(key=lambda t: -int(np.sum(t["mask"])))
        slots = []
        for template in templates:
            coords = _place_mask_randomly(forbidden, template["mask"])
            if coords is None:
                ok = False
                break
            slots.append(_slot_dict(coords, template["marker_rel"], template["tag"]))
        if not ok:
            continue

        layout_mask = np.zeros((h, w), dtype=bool)
        layout_markers = np.zeros((h, w), dtype=bool)
        for slot in slots:
            layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True
            mr, mc = slot["marker_abs"]
            layout_markers[int(mr), int(mc)] = True

        density = float(layout_mask.sum()) / float(h * w)
        lo, hi = HOLE_DENSITY_RANGE
        if not (lo <= density <= hi):
            continue
        if not _layout_touches_all_borders(layout_mask):
            continue

        dup_slots = [slot for slot in slots if slot["tag"] == "dup"]
        if len(dup_slots) != 2:
            continue
        if len({slot["marked_key"] for slot in dup_slots}) != 2:
            continue
        mirror_slots = [slot for slot in slots if slot["tag"] == "mirror"]
        if len(mirror_slots) != 1:
            continue
        if _rotation_shape_key(dup_slots[0]["mask"]) == _rotation_shape_key(mirror_slots[0]["mask"]):
            continue

        return slots, layout_mask, layout_markers

    return None


def _marker_anchor(_k: int) -> tuple[int, int]:
    return 0, 0


def _make_piece(slot: dict, color: int, k: int, rot: int) -> dict:
    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, int(color), BG_COLOR).astype(int)
    ar, ac = _marker_anchor(int(k))
    mr, mc = map(int, slot["marker_rel"])
    out_patch[mr * int(k) + ar, mc * int(k) + ac] = MARKER_COLOR

    in_patch = np.rot90(out_patch, int(rot))
    in_mask = np.rot90(out_mask, int(rot))
    return {
        "color": int(color),
        "slot": slot,
        "rotation": int(rot),
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
    }


def _build_expected_output(
    pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int
) -> np.ndarray:
    out = np.full(((layout_h + 2) * int(k), (layout_w + 2) * int(k)), int(frame_color), dtype=int)
    for piece in pieces:
        coords = piece["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * int(k)
        left = (int(coords[:, 1].min()) + 1) * int(k)
        paste(out, piece["out_patch"], int(top), int(left), mask=piece["out_mask"], overwrite=True)
    return out


def _build_input_canvas(
    pieces: list[dict],
    layout_mask: np.ndarray,
    layout_markers: np.ndarray,
    layout_h: int,
    layout_w: int,
    frame_color: int,
    distractor_color: int,
) -> np.ndarray | None:
    frame_h, frame_w = int(layout_h + 2), int(layout_w + 2)
    total_piece_area = sum(int(np.sum(piece["in_mask"])) for piece in pieces)
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

    inner = np.where(layout_mask, BG_COLOR, int(frame_color)).astype(int)
    inner[layout_markers] = MARKER_COLOR
    frame_img = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    frame_img[1:-1, 1:-1] = inner

    fr_max, fc_max = canvas_h - frame_h - 1, canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

    for piece in sorted(pieces, key=lambda item: -int(np.sum(item["in_mask"]))):
        try:
            pr, pc = random_free_location(
                canvas,
                piece["in_patch"],
                transparent=BG_COLOR,
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
        if canvas[rr, cc] != BG_COLOR:
            continue
        r0, r1 = max(0, rr - 1), min(canvas_h, rr + 2)
        c0, c1 = max(0, cc - 1), min(canvas_w, cc + 2)
        if np.any(canvas[r0:r1, c0:c1] != BG_COLOR):
            continue
        canvas[rr, cc] = int(distractor_color)
        placed += 1
    if placed < MIN_DISTRACTORS:
        return None

    vals, counts = np.unique(canvas, return_counts=True)
    if int(vals[np.argmax(counts)]) != BG_COLOR:
        return None
    return canvas


def _solver_view_from_generated_pieces(pieces: list[dict]) -> list[dict]:
    return [
        {
            "area": int(np.sum(piece["in_mask"])),
            "mask": piece["in_mask"].copy(),
            "patch": piece["in_patch"].copy(),
        }
        for piece in pieces
    ]


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)

    palette = [color for color in range(1, 10) if color != MARKER_COLOR]
    random.shuffle(palette)
    if len(palette) < n_slots + 2:
        return None
    frame_color = palette.pop()
    distractor_color = palette.pop()
    piece_colors = [palette.pop() for _ in range(n_slots)]

    layout_res = _gen_layout(layout_h, layout_w, n_slots)
    if layout_res is None:
        return None
    slots, layout_mask, layout_markers = layout_res

    rotations = [random.randint(0, 3) for _ in slots]
    if not any(rot != 0 for rot in rotations):
        rotations[random.randrange(len(rotations))] = random.randint(1, 3)

    pieces = [
        _make_piece(slot=slot, color=piece_colors[i], k=int(k), rot=int(rotations[i]))
        for i, slot in enumerate(slots)
    ]
    if not any(
        int(piece["rotation"]) != 0 and _rotation_state_count(piece["slot"]["mask"], piece["slot"]["marker_rel"]) >= 2
        for piece in pieces
    ):
        return None

    solver_pieces = _solver_view_from_generated_pieces(pieces)
    actual_match = _match_slots(slots, solver_pieces, int(k))
    if actual_match is None:
        return None
    other_k = 5 - int(k)
    if _match_slots(slots, solver_pieces, int(other_k)) is not None:
        return None

    frame_template = np.full((layout_h + 2, layout_w + 2), int(frame_color), dtype=int)
    frame_template[1:-1, 1:-1] = np.where(layout_mask, BG_COLOR, int(frame_color))
    frame_inner = frame_template[1:-1, 1:-1].copy()
    frame_inner[layout_markers] = MARKER_COLOR
    frame_template[1:-1, 1:-1] = frame_inner
    frame_area = int(np.count_nonzero(frame_template))
    max_piece_area = max(int(np.sum(piece["in_mask"])) for piece in pieces)
    if frame_area <= max_piece_area:
        return None

    dup_slots = [slot for slot in slots if slot["tag"] == "dup"]
    if len(dup_slots) != 2 or dup_slots[0]["shape_key"] != dup_slots[1]["shape_key"]:
        return None
    if dup_slots[0]["marked_key"] == dup_slots[1]["marked_key"]:
        return None
    if not any(slot["tag"] == "mirror" for slot in slots):
        return None

    expected_out = _build_expected_output(pieces, layout_h, layout_w, int(k), int(frame_color))
    canvas = _build_input_canvas(
        pieces=pieces,
        layout_mask=layout_mask,
        layout_markers=layout_markers,
        layout_h=layout_h,
        layout_w=layout_w,
        frame_color=int(frame_color),
        distractor_color=int(distractor_color),
    )
    if canvas is None:
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
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
