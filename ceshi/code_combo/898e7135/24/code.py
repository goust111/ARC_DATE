# Combo variant of 898e7135 (index 24)
# combo_spec: D3/1+D5/0
# Name: 多框架就地填充

from __future__ import annotations

# Design analysis
# - Original 898e7135 assumes one dominant frame component and emits a cropped,
#   k-scaled reconstruction of that single frame. This combo breaks both
#   assumptions at once: there are now 2-3 disjoint frames, and the output keeps
#   the full canvas instead of extracting one enlarged object.
# - Concrete behavior shift: if frame A contains an L-shaped hole and frame B a
#   unit hole, the old logic would keep only the largest frame and build a new
#   cropped board. The fused logic must detect both frames, match every external
#   piece globally, and recolor the hole cells in both frames at their original
#   locations while erasing all off-frame sprites and distractors.
# - Complexity is calibrated by making non-unit hole shapes globally unique under
#   D4 across all frames. Unit holes may repeat, but all unit pieces share one
#   color, so repeated 1-cell matches remain output-equivalent and do not create
#   ambiguous supervision. Each frame gets at least one non-unit slot, so the
#   multi-frame behavior is visibly active in every example.
# - Original implicit assumptions that no longer hold:
#   1. "largest foreground component is the frame" -> replaced by rectangle-frame
#      detection over all monochrome components.
#   2. "hole bbox must be tight-cropped" -> replaced by using each frame's full
#      interior, because the output is in-place rather than extracted.
#   3. "output size derives from one layout and k" -> replaced by preserving the
#      input canvas and repainting only frame locations.
# - New solver assumptions and generator guarantees:
#   1. A frame is a monochrome rectangular bbox whose outer border is solid and
#      whose interior contains only frame-color cells and bg holes. The generator
#      constructs frames exactly that way.
#   2. Every usable piece is a k-scaled monochrome sprite separated from all
#      frames. The generator reserves full frame rectangles during placement, so
#      no piece can overlap a frame hole and accidentally become "inside".
#   3. Matching is shape-based after block downsampling. The generator enforces
#      global D4-unique non-unit slot shapes and provides at least one unit slot
#      so k is recoverable from the smallest solid square piece.
# - Visual observability: every sample contains 2-3 frames, output size always
#   equals input size, and at least two frames receive filled slots. A human can
#   see that the task is "fill every frame in place" rather than "extract one
#   enlarged frame".

import random
import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location


K_CHOICES = [2, 3]
N_FRAMES_CHOICES = [2, 3]
TOTAL_SLOTS_MAX = 6
TOTAL_UNITS_MAX = 2
LAYOUT_H_RANGE = (4, 6)
LAYOUT_W_RANGE = (4, 6)
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5]
HOLE_DENSITY_RANGE = (0.18, 0.55)

FRAME_LAYOUT_RETRIES = 60
UNIT_SLOT_RETRIES = 25
NON_UNIT_SLOT_RETRIES = 35
GROW_SEED_TRIES = 10

CANVAS_DIM_RANGE = (22, 30)
FRAME_PLACEMENT_PADDING = 2
PIECE_PLACEMENT_PADDING = 1
AREA_BUFFER_CONST = 60

DISTRACTOR_RANGE = (6, 10)
DISTRACTOR_RETRY_MULT = 12
MIN_DISTRACTORS = 6

GENERATE_MAX_RETRIES = 400


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    variants = []
    for kk in range(4):
        rot = np.rot90(mask, kk)
        variants.append(rot.shape[0].to_bytes(1, "little") + rot.shape[1].to_bytes(1, "little") + rot.tobytes())
        flp = np.fliplr(rot)
        variants.append(flp.shape[0].to_bytes(1, "little") + flp.shape[1].to_bytes(1, "little") + flp.tobytes())
    return min(variants)


def _dihedral_masks(mask: np.ndarray) -> list[np.ndarray]:
    base = []
    for kk in range(4):
        rot = np.rot90(mask, kk)
        base.append(rot)
        base.append(np.fliplr(rot))
    uniq = []
    seen = set()
    for vm in base:
        key = vm.shape[0].to_bytes(1, "little") + vm.shape[1].to_bytes(1, "little") + vm.tobytes()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(vm)
    return uniq


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = mask.shape
    if hh % k != 0 or ww % k != 0:
        return None
    out = np.zeros((hh // k, ww // k), dtype=bool)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            block = mask[r * k : (r + 1) * k, c * k : (c + 1) * k]
            if bool(block.all()):
                out[r, c] = True
            elif bool(block.any()):
                return None
    return out


def _is_frame_component(grid: np.ndarray, bg: int, comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 5 or w < 5:
        return False
    color = int(comp["color"])
    sub = crop_bbox(grid, (r0, c0, r1, c1))
    if np.any(sub[0, :] != color) or np.any(sub[-1, :] != color):
        return False
    if np.any(sub[:, 0] != color) or np.any(sub[:, -1] != color):
        return False
    if np.any((sub != color) & (sub != bg)):
        return False
    interior = sub[1:-1, 1:-1]
    if interior.size == 0 or not np.any(interior == bg):
        return False
    return True


def _extract_frames_and_slots(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict], list[dict]]:
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    frames = []
    frame_ids: set[int] = set()
    for idx, comp in enumerate(comps):
        if not _is_frame_component(grid, bg, comp):
            continue
        frame_ids.add(int(idx))
        r0, c0, r1, c1 = map(int, comp["bbox"])
        sub = crop_bbox(grid, (r0, c0, r1, c1))
        interior_holes = sub[1:-1, 1:-1] == bg
        slots = []
        for coords in components_mask(interior_holes, connectivity=4):
            rr0, cc0, rr1, cc1 = bbox_coords(coords)
            hh = int(rr1 - rr0 + 1)
            ww = int(cc1 - cc0 + 1)
            sm = np.zeros((hh, ww), dtype=bool)
            sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
            abs_coords = coords + np.array([r0 + 1, c0 + 1], dtype=int)
            slots.append(
                {
                    "frame_bbox": (r0, c0, r1, c1),
                    "area": int(sm.sum()),
                    "abs_coords": abs_coords,
                    "abs_r0": int(r0 + 1 + rr0),
                    "abs_c0": int(c0 + 1 + cc0),
                    "mask": sm,
                }
            )
        slots.sort(key=lambda s: (-int(s["area"]), int(s["abs_r0"]), int(s["abs_c0"])))
        frames.append({"comp": comp, "color": int(comp["color"]), "bbox": (r0, c0, r1, c1), "slots": slots})
    frames.sort(key=lambda f: (int(f["bbox"][0]), int(f["bbox"][1])))

    slots = []
    for frame in frames:
        slots.extend(frame["slots"])
    slots.sort(key=lambda s: (-int(s["area"]), int(s["abs_r0"]), int(s["abs_c0"])))

    pieces = []
    for idx, comp in enumerate(comps):
        if int(idx) in frame_ids:
            continue
        coords = comp["coords"]
        if coords.size == 0 or int(coords.shape[0]) <= 1:
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        ph = int(pr1 - pr0 + 1)
        pw = int(pc1 - pc0 + 1)
        pm = np.zeros((ph, pw), dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append({"color": int(comp["color"]), "area": int(pm.sum()), "mask": pm, "bbox": (pr0, pc0, pr1, pc1)})
    pieces.sort(key=lambda p: (-int(p["area"]), int(p["bbox"][0]), int(p["bbox"][1])))
    return frames, slots, pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill every detected frame in place by matching all external block-scaled pieces.

    Input structure:
    - bg is the most common color. The canvas contains 2-3 separate monochrome rectangular frames.
    - Each frame has a 1-cell border and interior walls in the frame color, with bg hole regions inside.
    - Outside the frames sit monochrome pieces made of k x k solid blocks, plus singleton distractors.

    Rule:
    - Detect every valid frame, then collect all hole components from every frame interior.
    - Infer the global block scale k from the smallest solid square piece, and downsample each piece under all dihedral variants.
    - Match pieces to hole shapes globally, then repaint the matched hole cells inside their original frames.
    - Keep the full canvas size and frame positions unchanged; erase all off-frame pieces and distractors.

    Key insight:
    - The pieces still encode the holes at block scale, but the output no longer extracts or enlarges one frame.
    - The task is to solve a global assignment across multiple frames, then apply the result directly at the holes' original coordinates.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    frames, slots, pieces = _extract_frames_and_slots(g, bg)
    assert len(frames) >= 2, "variant expects multiple frames"
    assert slots, "at least one slot must exist"
    assert pieces, "at least one non-frame piece must exist"

    k = None
    for piece in pieces:
        h, w = map(int, piece["mask"].shape)
        if h == w and int(piece["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    assert k is not None and k >= 2, "k must be inferable from a solid square piece"

    candidates = []
    for pi, piece in enumerate(pieces):
        for vm in _dihedral_masks(piece["mask"]):
            um = _downsample_blocks(vm, k)
            if um is None:
                continue
            candidates.append({"pi": int(pi), "area": int(piece["area"]), "um": um, "color": int(piece["color"])})

    used_piece: set[int] = set()
    placements: list[tuple[dict, dict]] = []
    for slot in slots:
        need_area = int(slot["area"]) * int(k) * int(k)
        found = None
        for cand in candidates:
            if int(cand["pi"]) in used_piece:
                continue
            if int(cand["area"]) != int(need_area):
                continue
            if cand["um"].shape != slot["mask"].shape:
                continue
            if np.array_equal(cand["um"], slot["mask"]):
                found = cand
                break
        assert found is not None, "every slot must match a piece"
        used_piece.add(int(found["pi"]))
        placements.append((slot, found))

    out = np.full_like(g, bg)
    for frame in frames:
        coords = frame["comp"]["coords"]
        out[coords[:, 0], coords[:, 1]] = int(frame["color"])
    for slot, cand in placements:
        abs_coords = slot["abs_coords"]
        out[abs_coords[:, 0], abs_coords[:, 1]] = int(cand["color"])
    return out


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    h, w = forbidden.shape
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
        return None
    random.shuffle(free_cells)
    for start in free_cells[:GROW_SEED_TRIES]:
        r0, c0 = int(start[0]), int(start[1])
        chosen = {(r0, c0)}
        frontier = {(r0 - 1, c0), (r0 + 1, c0), (r0, c0 - 1), (r0, c0 + 1)}
        while len(chosen) < target_size and frontier:
            cands = [
                p
                for p in frontier
                if 0 <= p[0] < h and 0 <= p[1] < w and not forbidden[p[0], p[1]] and p not in chosen
            ]
            if not cands:
                break
            nr, nc = random.choice(cands)
            chosen.add((nr, nc))
            frontier.discard((nr, nc))
            frontier.update({(nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)})
        if len(chosen) == target_size:
            return np.array(sorted(chosen), dtype=int)
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


def _gen_layout(h: int, w: int, n_non_units: int, n_units: int, global_shape_keys: set[bytes]) -> tuple[list[dict], set[bytes]] | None:
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []
    new_keys: set[bytes] = set()
    used_here = set(global_shape_keys)

    for _ in range(n_units):
        for _attempt in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is None:
                continue
            slots.append({"coords": coords, "area": 1, "mask": np.ones((1, 1), dtype=bool)})
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    for _ in range(n_non_units):
        size = random.choice(NON_UNIT_SIZE_CHOICES)
        for _attempt in range(NON_UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, size)
            if coords is None:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            sm = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            sm[coords[:, 0] - r0, coords[:, 1] - c0] = True
            key = _canonical_shape_key(sm)
            if key in used_here:
                continue
            used_here.add(key)
            new_keys.add(key)
            slots.append({"coords": coords, "area": int(size), "mask": sm})
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    density = sum(int(s["area"]) for s in slots) / float(h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (lo <= density <= hi):
        return None
    return slots, new_keys


def _make_piece(slot: dict, color: int, k: int, apply_dihedral: bool) -> dict:
    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    in_mask = out_mask.copy()
    if apply_dihedral:
        rot = random.randint(0, 3)
        in_mask = np.rot90(in_mask, rot)
        if random.random() < 0.5:
            in_mask = np.fliplr(in_mask)
    in_patch = np.where(in_mask, color, 0).astype(int)
    return {"color": int(color), "slot": slot, "in_mask": in_mask, "in_patch": in_patch}


def _build_frame_patch(layout_h: int, layout_w: int, slots: list[dict], frame_color: int) -> np.ndarray:
    patch = np.full((layout_h + 2, layout_w + 2), int(frame_color), dtype=int)
    for slot in slots:
        coords = slot["coords"]
        patch[coords[:, 0] + 1, coords[:, 1] + 1] = 0
    return patch


def _choose_counts(n_frames: int) -> tuple[list[int], list[int]] | None:
    total_slots_lo = max(4, n_frames + 1)
    if total_slots_lo > TOTAL_SLOTS_MAX:
        return None
    total_slots = random.randint(total_slots_lo, TOTAL_SLOTS_MAX)
    max_units = min(TOTAL_UNITS_MAX, total_slots - n_frames)
    if max_units < 1:
        return None
    total_units = random.randint(1, max_units)
    total_non_units = total_slots - total_units
    if total_non_units < n_frames:
        return None

    frame_non_units = [1] * n_frames
    for _ in range(total_non_units - n_frames):
        frame_non_units[random.randrange(n_frames)] += 1

    frame_units = [0] * n_frames
    order = list(range(n_frames))
    random.shuffle(order)
    for idx in range(total_units):
        frame_units[order[idx]] += 1
    return frame_non_units, frame_units


def _choose_canvas_shape(frame_patches: list[np.ndarray], pieces: list[dict]) -> tuple[int, int] | None:
    frame_area = sum(int(p.shape[0] * p.shape[1]) for p in frame_patches)
    piece_area = sum(int(p["in_mask"].sum()) for p in pieces)
    needed = frame_area + piece_area + AREA_BUFFER_CONST
    max_frame_h = max(int(p.shape[0]) for p in frame_patches)
    max_frame_w = max(int(p.shape[1]) for p in frame_patches)
    lo_h = max(CANVAS_DIM_RANGE[0], max_frame_h + 2)
    lo_w = max(CANVAS_DIM_RANGE[0], max_frame_w + 2)
    if lo_h > CANVAS_DIM_RANGE[1] or lo_w > CANVAS_DIM_RANGE[1]:
        return None
    for _ in range(20):
        ch = random.randint(lo_h, CANVAS_DIM_RANGE[1])
        cw = random.randint(lo_w, CANVAS_DIM_RANGE[1])
        if ch * cw >= needed:
            return ch, cw
    ch = CANVAS_DIM_RANGE[1]
    cw = CANVAS_DIM_RANGE[1]
    if ch * cw < needed:
        return None
    return ch, cw


def _place_frames_and_pieces(frame_patches: list[np.ndarray], pieces: list[dict]) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray] | None:
    canvas_shape = _choose_canvas_shape(frame_patches, pieces)
    if canvas_shape is None:
        return None
    ch, cw = canvas_shape
    canvas = np.zeros((ch, cw), dtype=int)
    occ = np.zeros((ch, cw), dtype=int)
    frame_positions: list[tuple[int, int]] = [(-1, -1)] * len(frame_patches)

    frame_order = sorted(range(len(frame_patches)), key=lambda i: -(frame_patches[i].shape[0] * frame_patches[i].shape[1]))
    for fi in frame_order:
        patch = frame_patches[fi]
        solid = np.ones_like(patch, dtype=int)
        try:
            fr, fc = random_free_location(
                occ,
                solid,
                transparent=0,
                border=1,
                padding=FRAME_PLACEMENT_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, patch, int(fr), int(fc), overwrite=True)
        paste(occ, solid, int(fr), int(fc), overwrite=True)
        frame_positions[fi] = (int(fr), int(fc))

    for piece in sorted(pieces, key=lambda p: -int(p["in_mask"].sum())):
        try:
            pr, pc = random_free_location(
                occ,
                piece["in_patch"],
                transparent=0,
                border=0,
                padding=PIECE_PLACEMENT_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, piece["in_patch"], int(pr), int(pc), mask=piece["in_mask"], overwrite=True)
        occ_patch = np.where(piece["in_mask"], 1, 0).astype(int)
        paste(occ, occ_patch, int(pr), int(pc), transparent=0, overwrite=True)

    return canvas, frame_positions, occ


def _add_distractors(canvas: np.ndarray, occ: np.ndarray, distractor_color: int) -> bool:
    h, w = canvas.shape
    target = random.randint(*DISTRACTOR_RANGE)
    placed = 0
    for _ in range(target * DISTRACTOR_RETRY_MULT):
        if placed >= target:
            break
        rr = random.randint(0, h - 1)
        cc = random.randint(0, w - 1)
        if occ[rr, cc] != 0 or canvas[rr, cc] != 0:
            continue
        r0 = max(0, rr - 1)
        r1 = min(h, rr + 2)
        c0 = max(0, cc - 1)
        c1 = min(w, cc + 2)
        if np.any(canvas[r0:r1, c0:c1] != 0):
            continue
        canvas[rr, cc] = int(distractor_color)
        occ[rr, cc] = 1
        placed += 1
    return placed >= MIN_DISTRACTORS


def _build_expected_output(canvas_shape: tuple[int, int], frame_patches: list[np.ndarray], frame_positions: list[tuple[int, int]], frame_piece_groups: list[list[dict]]) -> np.ndarray:
    out = np.zeros(canvas_shape, dtype=int)
    for frame_patch, (fr, fc), pieces in zip(frame_patches, frame_positions, frame_piece_groups):
        paste(out, frame_patch, int(fr), int(fc), overwrite=True)
        for piece in pieces:
            coords = piece["slot"]["coords"]
            out[coords[:, 0] + int(fr) + 1, coords[:, 1] + int(fc) + 1] = int(piece["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    n_frames = random.choice(N_FRAMES_CHOICES)
    counts = _choose_counts(n_frames)
    if counts is None:
        return None
    frame_non_units, frame_units = counts

    total_non_units = sum(frame_non_units)
    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    if len(palette) < total_non_units + 1:
        return None
    non_unit_colors = [palette.pop() for _ in range(total_non_units)]
    unit_color = palette.pop()

    global_shape_keys: set[bytes] = set()
    frame_patches: list[np.ndarray] = []
    frame_piece_groups: list[list[dict]] = []
    all_pieces: list[dict] = []
    color_idx = 0

    for fi in range(n_frames):
        n_non = int(frame_non_units[fi])
        n_unit = int(frame_units[fi])
        built = None
        for _ in range(FRAME_LAYOUT_RETRIES):
            lh = random.randint(*LAYOUT_H_RANGE)
            lw = random.randint(*LAYOUT_W_RANGE)
            if n_non + n_unit > max(1, (lh * lw) // 4):
                continue
            res = _gen_layout(lh, lw, n_non, n_unit, global_shape_keys)
            if res is None:
                continue
            slots, new_keys = res
            built = (lh, lw, slots, new_keys)
            break
        if built is None:
            return None
        lh, lw, slots, new_keys = built
        global_shape_keys |= new_keys

        unit_slots = slots[:n_unit]
        non_unit_slots = slots[n_unit:]
        pieces_here = []
        for slot in non_unit_slots:
            piece = _make_piece(slot, non_unit_colors[color_idx], k, apply_dihedral=True)
            pieces_here.append(piece)
            all_pieces.append(piece)
            color_idx += 1
        for slot in unit_slots:
            piece = _make_piece(slot, unit_color, k, apply_dihedral=False)
            pieces_here.append(piece)
            all_pieces.append(piece)

        frame_patches.append(_build_frame_patch(lh, lw, slots, frame_color))
        frame_piece_groups.append(pieces_here)

    placed = _place_frames_and_pieces(frame_patches, all_pieces)
    if placed is None:
        return None
    canvas, frame_positions, occ = placed

    if not _add_distractors(canvas, occ, distractor_color):
        return None

    if int(get_bg(canvas)) != 0:
        return None

    expected = _build_expected_output(canvas.shape, frame_patches, frame_positions, frame_piece_groups)
    try:
        produced = main(canvas)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None

    frame_count = len(_extract_frames_and_slots(canvas, 0)[0])
    if frame_count < 2:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
