# Combo variant of 898e7135 (index 50)
# combo_spec: D1/1+D4/3+D14/1
# Name: 标记引导旋转缩放

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


# Design analysis
# ---------------
# a) Concrete behavior trace
#    Original 898e7135 treats every outside sprite as an already-scaled kxk-block patch.
#    A solver can rotate/flip that patch, then paste it directly into the enlarged frame.
#    In this combo, every slot still has a block-level hole shape, but one hole cell is
#    recolored to MARKER_COLOR. Every outside piece also has exactly one marker pixel.
#    The marked block identifies which block of the polyomino is special, while the
#    marker pixel's corner encodes a quarter-turn amount:
#      TL -> 0, BL -> 90 CCW, BR -> 180, TR -> 270.
#    Example: a 3-cell L slot has its top cell marked. If an input piece shows the same
#    L mirrored by rotation, and its red pixel sits in the bottom-left corner of the
#    marked kxk block, the solver must rotate the piece 90 CCW at block level. After
#    that rotation the block mask and marked block match the slot, and only then is the
#    piece re-rendered at scale k into the output. The output marker is always rendered
#    at the canonical top-left corner of the marked block, so "rotate then scale" is
#    visibly different from "scale then rotate the input patch back".
#
# b) Complexity calibration
#    A weak generator could accidentally regress to the original task if shape alone
#    always determines the slot and if all marker corners are zero. To prevent that:
#    - every sample contains a chiral shape A and its mirror B, so flips would solve
#      something the combo explicitly forbids;
#    - every sample also contains two copies of shape A with different marked cells,
#      so shape alone is insufficient;
#    - at least one non-unit piece uses a non-zero marker corner, so the re-rendered
#      output differs from direct patch rotation.
#
# c) Implicit assumption audit for main()
#    - find_components(..., monochromatic=False): assumes the largest foreground
#      component is the frame border/walls. Generator enforces frame area > any piece
#      area and keeps all pieces 8-separated from the frame.
#    - hole extraction: original code assumed holes were bg-only. Here hole territory is
#      bg or MARKER_COLOR, and each slot must contain exactly one marker cell.
#    - k inference from the smallest solid square piece: preserved by forcing 1-2 unit
#      slots. Their pieces are full kxk squares in mask terms even though one pixel is
#      recolored to MARKER_COLOR.
#    - downsample_blocks: assumes each piece is exactly composed of solid kxk blocks.
#      Generator renders every occupied block as full kxk body color except one marker
#      pixel, so the binary mask is still perfectly block-aligned.
#    - rotation decoding: new logic assumes the marker pixel lies on one of the four
#      block corners. Generator guarantees that exact encoding.
#    - matching: new logic assumes every piece canonicalizes to some slot under
#      rotations only. Generator guarantees this by construction and uses oracle checks.
#
# d) Placement legality
#    Slots are placed with 8-neighborhood exclusion inside the layout, pieces are placed
#    with random_free_location(..., padding=1, padding_connectivity=8), and distractors
#    are isolated singleton pixels. This keeps all meaningful objects disjoint.
#
# e) Visual observability
#    Humans can see three coupled signals:
#    - mirror chirality matters because A and mirrored-B both occur;
#    - same-shape different-marker slots show that the red marked block matters;
#    - the red marker corner varies across input pieces but becomes canonical in output,
#      revealing that the marker corner is an instruction for rotation before scaling.
#
# f) Cleanliness
#    The only overlay is one marker pixel per piece and one marker cell per slot. Those
#    are sparse, high-contrast, and never stacked with distractors or with each other.
#
# g) New solver assumptions that the generator must satisfy
#    - each slot has exactly one marker cell;
#    - each piece has exactly one marker pixel;
#    - at least one non-unit piece has a non-zero encoded rotation;
#    - at least one mirror pair of non-unit shapes exists;
#    - at least one duplicate rotation-class shape with different markers exists;
#    - the old "rotate scaled patch and paste" behavior disagrees with this combo on
#      most samples because the output re-renders markers canonically after rotation.


MARKER_COLOR = 2
BG_COLOR = 0
K_CHOICES = (2, 3)

LAYOUT_H_RANGE = (5, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (4, 6)
N_UNITS_MAX = 2
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6]
CHIRAL_BASE_SIZE_CHOICES = [4, 5, 6]

HOLE_DENSITY_RANGE = (0.20, 0.50)
SHAPE_RETRIES = 70
LAYOUT_RETRIES = 140

CANVAS_DIM_RANGE = (16, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 60

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 12
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 450


RotKey = tuple[tuple[int, int], bytes]


def _rotation_shape_key(mask: np.ndarray) -> RotKey:
    variants = []
    for rot in range(4):
        rm = np.rot90(mask, int(rot))
        variants.append((tuple(map(int, rm.shape)), rm.tobytes()))
    return min(variants)


def _rotate_coord(coord: tuple[int, int], shape: tuple[int, int], rot: int) -> tuple[int, int]:
    r, c = map(int, coord)
    h, w = map(int, shape)
    rot %= 4
    if rot == 0:
        return r, c
    if rot == 1:
        return w - 1 - c, r
    if rot == 2:
        return h - 1 - r, w - 1 - c
    return c, h - 1 - r


def _marker_anchor(k: int, rot_code: int) -> tuple[int, int]:
    rot_code %= 4
    if rot_code == 0:
        return 0, 0
    if rot_code == 1:
        return k - 1, 0
    if rot_code == 2:
        return k - 1, k - 1
    return 0, k - 1


def _rotation_from_marker_local(local_r: int, local_c: int, k: int) -> int | None:
    for rot in range(4):
        if (int(local_r), int(local_c)) == _marker_anchor(int(k), int(rot)):
            return int(rot)
    return None


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = map(int, mask.shape)
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


def _render_scaled_piece(
    block_mask: np.ndarray,
    marker_block: tuple[int, int],
    color: int,
    k: int,
    marker_rot_code: int,
) -> tuple[np.ndarray, np.ndarray]:
    out_mask = np.kron(block_mask, np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, int(color), BG_COLOR).astype(int)
    mr, mc = map(int, marker_block)
    ar, ac = _marker_anchor(int(k), int(marker_rot_code))
    out_patch[mr * k + ar, mc * k + ac] = MARKER_COLOR
    return out_patch, out_mask


def _tight_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _slot_from_coords(
    coords: np.ndarray,
    marker_rel: tuple[int, int],
    *,
    family: str,
) -> dict:
    mask = _tight_mask_from_coords(coords)
    r0, c0, _r1, _c1 = map(int, bbox_coords(coords))
    return {
        "coords": coords.astype(int),
        "area": int(coords.shape[0]),
        "mask": mask,
        "r0": int(r0),
        "c0": int(c0),
        "marker_rel": (int(marker_rel[0]), int(marker_rel[1])),
        "rot_key": _rotation_shape_key(mask),
        "family": family,
    }


def _coords_to_mask(coords: set[tuple[int, int]]) -> np.ndarray:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    r0, c0 = min(rows), min(cols)
    h = max(rows) - r0 + 1
    w = max(cols) - c0 + 1
    out = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        out[int(r - r0), int(c - c0)] = True
    return out


def _grow_polyomino(area: int) -> np.ndarray:
    chosen = {(0, 0)}
    frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
    while len(chosen) < int(area):
        nr, nc = random.choice(list(frontier))
        frontier.remove((nr, nc))
        chosen.add((nr, nc))
        for nxt in ((nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)):
            if nxt not in chosen:
                frontier.add(nxt)
    return _coords_to_mask(chosen)


def _marker_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def _is_chiral(mask: np.ndarray) -> bool:
    return _rotation_shape_key(mask) != _rotation_shape_key(np.fliplr(mask))


def _random_shape_mask(
    target_size: int,
    *,
    require_chiral: bool,
    used_rot_keys: set[RotKey] | None = None,
) -> np.ndarray | None:
    used_rot_keys = used_rot_keys or set()
    for _ in range(SHAPE_RETRIES):
        mask = _grow_polyomino(int(target_size))
        key = _rotation_shape_key(mask)
        if key in used_rot_keys:
            continue
        if require_chiral and not _is_chiral(mask):
            continue
        return mask
    return None


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
    base_coords = np.argwhere(mask)
    positions = [(int(r0), int(c0)) for r0 in range(H - h + 1) for c0 in range(W - w + 1)]
    random.shuffle(positions)
    for top, left in positions:
        coords = base_coords + np.array([top, left], dtype=int)
        if bool(forbidden[coords[:, 0], coords[:, 1]].any()):
            continue
        _forbid_coords(forbidden, coords)
        return coords.astype(int)
    return None


def _layout_touches_all_borders(layout_mask: np.ndarray) -> bool:
    return bool(
        layout_mask[0].any()
        and layout_mask[-1].any()
        and layout_mask[:, 0].any()
        and layout_mask[:, -1].any()
    )


def _count_mirror_pairs(non_unit_slots: list[dict]) -> int:
    count = 0
    for i in range(len(non_unit_slots)):
        mirror_key = _rotation_shape_key(np.fliplr(non_unit_slots[i]["mask"]))
        if mirror_key == non_unit_slots[i]["rot_key"]:
            continue
        for j in range(i + 1, len(non_unit_slots)):
            if non_unit_slots[j]["rot_key"] == mirror_key:
                count += 1
    return count


def _count_duplicate_marker_pairs(non_unit_slots: list[dict]) -> int:
    count = 0
    for i in range(len(non_unit_slots)):
        for j in range(i + 1, len(non_unit_slots)):
            if non_unit_slots[i]["rot_key"] != non_unit_slots[j]["rot_key"]:
                continue
            if non_unit_slots[i]["marker_rel"] == non_unit_slots[j]["marker_rel"]:
                continue
            count += 1
    return count


def _gen_layout(
    h: int, w: int, n_slots: int, n_units: int
) -> tuple[list[dict], list[dict], np.ndarray, np.ndarray] | None:
    n_non_unit = int(n_slots - n_units)
    if n_non_unit < 3:
        return None

    for _ in range(LAYOUT_RETRIES):
        forbidden = np.zeros((h, w), dtype=bool)
        used_rot_keys: set[RotKey] = set()
        templates: list[dict] = []

        base_mask = None
        base_markers = None
        for _ in range(SHAPE_RETRIES):
            cand = _random_shape_mask(
                random.choice(CHIRAL_BASE_SIZE_CHOICES),
                require_chiral=True,
                used_rot_keys=used_rot_keys,
            )
            if cand is None:
                continue
            cands = _marker_candidates(cand)
            if len(cands) < 2:
                continue
            base_mask = cand
            base_markers = random.sample(cands, 2)
            break
        if base_mask is None or base_markers is None:
            continue

        mirror_mask = np.fliplr(base_mask)
        if _rotation_shape_key(base_mask) == _rotation_shape_key(mirror_mask):
            continue

        used_rot_keys.add(_rotation_shape_key(base_mask))
        used_rot_keys.add(_rotation_shape_key(mirror_mask))
        templates.extend(
            [
                {"mask": base_mask, "marker_rel": base_markers[0], "family": "dup_a"},
                {"mask": base_mask, "marker_rel": base_markers[1], "family": "dup_b"},
                {
                    "mask": mirror_mask,
                    "marker_rel": (int(base_markers[0][0]), int(base_mask.shape[1] - 1 - base_markers[0][1])),
                    "family": "mirror",
                },
            ]
        )

        for idx in range(n_non_unit - 3):
            extra_mask = _random_shape_mask(
                random.choice(NON_UNIT_SIZE_CHOICES),
                require_chiral=False,
                used_rot_keys=used_rot_keys,
            )
            if extra_mask is None:
                break
            used_rot_keys.add(_rotation_shape_key(extra_mask))
            templates.append(
                {
                    "mask": extra_mask,
                    "marker_rel": random.choice(_marker_candidates(extra_mask)),
                    "family": f"extra_{idx}",
                }
            )
        else:
            templates.sort(key=lambda item: -int(np.sum(item["mask"])))
            non_unit_slots: list[dict] = []
            ok = True
            for tpl in templates:
                coords = _place_mask_randomly(forbidden, tpl["mask"])
                if coords is None:
                    ok = False
                    break
                non_unit_slots.append(
                    _slot_from_coords(coords, tpl["marker_rel"], family=str(tpl["family"]))
                )
            if not ok:
                continue

            unit_slots: list[dict] = []
            for ui in range(n_units):
                coords = _place_mask_randomly(forbidden, np.ones((1, 1), dtype=bool))
                if coords is None:
                    ok = False
                    break
                unit_slots.append(_slot_from_coords(coords, (0, 0), family=f"unit_{ui}"))
            if not ok:
                continue

            layout_mask = np.zeros((h, w), dtype=bool)
            layout_markers = np.zeros((h, w), dtype=bool)
            for slot in non_unit_slots + unit_slots:
                layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True
                rr0 = int(slot["coords"][:, 0].min())
                cc0 = int(slot["coords"][:, 1].min())
                mr, mc = slot["marker_rel"]
                layout_markers[rr0 + mr, cc0 + mc] = True

            density = float(layout_mask.sum()) / float(h * w)
            lo, hi = HOLE_DENSITY_RANGE
            if not (lo <= density <= hi):
                continue
            if not _layout_touches_all_borders(layout_mask):
                continue
            if _count_mirror_pairs(non_unit_slots) < 1:
                continue
            if _count_duplicate_marker_pairs(non_unit_slots) < 1:
                continue

            return unit_slots, non_unit_slots, layout_mask, layout_markers

    return None


def _build_piece(slot: dict, color: int, k: int, solve_rot: int) -> dict:
    out_patch, out_mask = _render_scaled_piece(
        slot["mask"], slot["marker_rel"], int(color), int(k), marker_rot_code=0
    )
    in_mask_blocks = np.rot90(slot["mask"], (4 - int(solve_rot)) % 4)
    in_marker = _rotate_coord(
        slot["marker_rel"], tuple(map(int, slot["mask"].shape)), (4 - int(solve_rot)) % 4
    )
    in_patch, in_mask = _render_scaled_piece(
        in_mask_blocks, in_marker, int(color), int(k), marker_rot_code=int(solve_rot)
    )
    return {
        "color": int(color),
        "slot": slot,
        "solve_rot": int(solve_rot),
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
        "in_marker_block": in_marker,
    }


def _match_slots(slots: list[dict], pieces: list[dict], k: int) -> list[tuple[dict, dict]] | None:
    candidates_by_slot: list[list[dict]] = []
    for slot in slots:
        want_area = int(slot["area"]) * int(k) * int(k)
        cur = []
        for pi, piece in enumerate(pieces):
            if int(piece["area"]) != want_area:
                continue
            canon_mask = np.rot90(piece["block_mask"], int(piece["solve_rot"]))
            canon_marker = _rotate_coord(
                piece["marker_block"], tuple(map(int, piece["block_mask"].shape)), int(piece["solve_rot"])
            )
            if canon_mask.shape != slot["mask"].shape:
                continue
            if not np.array_equal(canon_mask, slot["mask"]):
                continue
            if canon_marker != slot["marker"]:
                continue
            cur.append({"pi": int(pi), "color": int(piece["color"])})
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
    ans: list[tuple[dict, dict]] = [None] * len(slots)  # type: ignore

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
            used.remove(pi)
            ans[si] = None  # type: ignore
        return False

    return ans if dfs(0) else None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode marker-guided rotations, then rebuild the enlarged frame with canonically rendered pieces.

    Input structure:
    - bg is the most common color; the largest foreground component is the monochrome frame.
    - Each hole region inside the frame contains exactly one marker-colored cell.
    - Each outside piece is still made of exact kxk blocks, but it has one marker pixel whose
      block corner encodes the quarter-turn needed to align the piece to its slot.

    Rule:
    - Treat both bg and marker cells inside the frame as hole territory and extract slot masks plus
      their unique marked cells.
    - Infer k from the smallest solid square piece, downsample each piece to block level, read the
      needed rotation from the marker corner, and allow matching under rotations only.
    - After the block-level piece is rotated into its slot orientation, re-render it at scale k and
      place a canonical marker pixel in the slot's marked block.

    Key insight:
    The red marker is not decorative: its block position identifies the marked cell, and its corner
    identifies the rotation angle. Matching is purely rotational, and the final picture is rebuilt
    from the rotated block mask rather than by pasting the input patch back directly.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "expected at least one foreground component"
    frame_comp = max(comps, key=lambda item: int(item["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [
        int(g[int(r), int(c)])
        for r, c in frame_comp["coords"]
        if int(g[int(r), int(c)]) != MARKER_COLOR
    ]
    assert frame_cols, "frame must contain a non-marker color"
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole_like = (sub == bg) | (sub == MARKER_COLOR)
    marker_mask = sub == MARKER_COLOR
    hole_box = bbox(hole_like)
    assert hole_box is not None, "frame must enclose at least one hole-like cell"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole_like[lr0 : lr1 + 1, lc0 : lc1 + 1]
    layout_markers = marker_mask[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        sm = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        marker_coords = [(int(r), int(c)) for r, c in coords if bool(layout_markers[int(r), int(c)])]
        assert len(marker_coords) == 1, "each slot must contain exactly one marker cell"
        mr, mc = marker_coords[0]
        slots.append(
            {
                "area": int(sm.sum()),
                "r0": int(rr0),
                "c0": int(cc0),
                "mask": sm,
                "marker": (int(mr - rr0), int(mc - cc0)),
            }
        )
    slots.sort(key=lambda item: (-int(item["area"]), int(item["r0"]), int(item["c0"])))
    assert slots, "frame must contain at least one slot"

    raw_pieces = []
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
        marker_coords = np.argwhere((patch == MARKER_COLOR) & pm)
        if marker_coords.shape[0] != 1:
            continue
        body_colors = [int(v) for v in patch[pm] if int(v) != MARKER_COLOR]
        if not body_colors:
            continue
        raw_pieces.append(
            {
                "area": int(pm.sum()),
                "mask": pm,
                "patch": patch,
                "color": int(Counter(body_colors).most_common(1)[0][0]),
                "marker_pixel": (int(marker_coords[0][0]), int(marker_coords[0][1])),
            }
        )
    assert raw_pieces, "need at least one piece"

    k = None
    for piece in raw_pieces:
        h, w = map(int, piece["mask"].shape)
        if h == w and int(piece["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    assert k is not None and k in K_CHOICES, "failed to infer valid k from unit pieces"

    pieces = []
    for piece in raw_pieces:
        block_mask = _downsample_blocks(piece["mask"], int(k))
        assert block_mask is not None, "piece mask must be an exact kxk block upscaling"
        mr, mc = piece["marker_pixel"]
        solve_rot = _rotation_from_marker_local(int(mr % k), int(mc % k), int(k))
        assert solve_rot is not None, "marker pixel must sit on a block corner"
        pieces.append(
            {
                "area": int(piece["area"]),
                "color": int(piece["color"]),
                "block_mask": block_mask,
                "marker_block": (int(mr // k), int(mc // k)),
                "solve_rot": int(solve_rot),
            }
        )

    assert len(pieces) == len(slots), "need exactly one usable piece per slot"
    placements = _match_slots(slots, pieces, int(k))
    assert placements is not None, "no rotation-only matching satisfies all slots"

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)

    for slot, cand in placements:
        patch, mask = _render_scaled_piece(
            slot["mask"], slot["marker"], int(cand["color"]), int(k), marker_rot_code=0
        )
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, patch, top, left, mask=mask, overwrite=True)

    return out


def _build_expected_output(
    pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int
) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for piece in pieces:
        coords = piece["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * int(k)
        left = (int(coords[:, 1].min()) + 1) * int(k)
        paste(out, piece["out_patch"], top, left, mask=piece["out_mask"], overwrite=True)
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

    inner = np.where(layout_mask, BG_COLOR, int(frame_color)).astype(int)
    inner[layout_markers] = MARKER_COLOR
    frame_img = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    frame_img[1:-1, 1:-1] = inner

    fr_max, fc_max = canvas_h - frame_h - 1, canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

    for piece in sorted(pieces, key=lambda item: -int(item["in_mask"].sum())):
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
        paste(canvas, piece["in_patch"], pr, pc, mask=piece["in_mask"], overwrite=True)

    n_distractors = random.randint(*DISTRACTOR_RANGE)
    placed = 0
    for _ in range(n_distractors * DISTRACTOR_RETRY_MULT):
        if placed >= n_distractors:
            break
        rr = random.randint(0, canvas_h - 1)
        cc = random.randint(0, canvas_w - 1)
        if int(canvas[rr, cc]) != BG_COLOR:
            continue
        r0, r1 = max(0, rr - 1), min(canvas_h, rr + 2)
        c0, c1 = max(0, cc - 1), min(canvas_w, cc + 2)
        if bool(np.any(canvas[r0:r1, c0:c1] != BG_COLOR)):
            continue
        canvas[rr, cc] = int(distractor_color)
        placed += 1
    if placed < MIN_DISTRACTORS:
        return None

    vals, counts = np.unique(canvas, return_counts=True)
    if int(vals[np.argmax(counts)]) != BG_COLOR:
        return None
    return canvas


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)
    n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 3))

    palette = [color for color in range(1, 10) if color != MARKER_COLOR]
    random.shuffle(palette)
    needed_piece_colors = (n_slots - n_units) + 1
    if len(palette) < needed_piece_colors + 2:
        return None
    frame_color = palette.pop()
    distractor_color = palette.pop()
    non_unit_colors = [palette.pop() for _ in range(n_slots - n_units)]
    unit_color = palette.pop()

    layout_res = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if layout_res is None:
        return None
    unit_slots, non_unit_slots, layout_mask, layout_markers = layout_res

    frame_area = int((layout_h + 2) * (layout_w + 2) - np.sum(layout_mask))
    max_piece_area = int(
        max(slot["area"] * k * k for slot in non_unit_slots + unit_slots)
    )
    if frame_area <= max_piece_area:
        return None

    non_unit_rots = [random.randint(0, 3) for _ in non_unit_slots]
    if not any(rot in (1, 3) for rot in non_unit_rots):
        non_unit_rots[random.randrange(len(non_unit_rots))] = random.choice([1, 3])
    if len(set(non_unit_rots)) == 1 and len(non_unit_rots) >= 2:
        idx = random.randrange(len(non_unit_rots))
        choices = [rot for rot in range(4) if rot != non_unit_rots[idx]]
        non_unit_rots[idx] = random.choice(choices)
    unit_rots = [random.randint(0, 3) for _ in unit_slots]

    pieces = [
        _build_piece(slot, non_unit_colors[i], int(k), int(non_unit_rots[i]))
        for i, slot in enumerate(non_unit_slots)
    ]
    pieces += [
        _build_piece(slot, unit_color, int(k), int(unit_rots[i]))
        for i, slot in enumerate(unit_slots)
    ]

    if not any(
        piece["solve_rot"] in (1, 3) and int(piece["slot"]["area"]) > 1 for piece in pieces
    ):
        return None

    expected_out = _build_expected_output(pieces, layout_h, layout_w, int(k), int(frame_color))
    canvas = _build_input_canvas(
        pieces,
        layout_mask,
        layout_markers,
        layout_h,
        layout_w,
        int(frame_color),
        int(distractor_color),
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
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
