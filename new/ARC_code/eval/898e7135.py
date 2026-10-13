from __future__ import annotations
from collections import Counter
import numpy as np
from arc_utils import bbox, bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste

def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate a frame's hole-layout and fill each hole with its matching piece.

    Input structure:
    - bg = most common color. Largest component is a monochrome frame enclosing bg holes.
    - Outside pieces composed of k×k blocks; k = side of smallest solid square piece.

    Rule:
    - Extract connected hole regions inside the frame as slots.
    - Match each piece to a slot by comparing block-level shapes under rotation/flip.
    - Output the layout scaled by k, with pieces placed in slots; frame color as background.

    Key insight: Pieces are k×k-block-scaled versions of holes; downsampling reveals shape matches under all 8 orientations.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # --- find the largest connected component of foreground (multicolor) ---
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert not (not comps)
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    # The frame is monochrome; use its dominant color as output background.
    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    bgc = int(Counter(frame_cols).most_common(1)[0][0])

    # --- layout = bg holes inside the frame bbox, tight-cropped ---
    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == int(bg)
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one bg hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    # --- slot components in the layout ---
    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
        h = int(rr1 - rr0 + 1)
        w = int(cc1 - cc0 + 1)
        sm = np.zeros((h, w), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append({"area": int(sm.sum()), "r0": int(rr0), "c0": int(cc0), "mask": sm})
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))

    assert slots, "frame must contain at least one slot"

    # --- pieces = non-singleton foreground components excluding the frame component ---
    pieces = []
    for comp in comps:
        if comp is frame_comp:
            continue
        coords = comp["coords"]
        if coords.size == 0 or int(coords.shape[0]) <= 1:
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        ph, pw = map(int, patch.shape)
        pm = np.zeros((ph, pw), dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append({"area": int(pm.sum()), "mask": pm, "patch": patch})

    assert pieces, "at least one non-frame piece must exist"

    # --- infer k from the smallest solid square piece ---
    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    k = int(k or 1)

    def downsample_blocks(mask: np.ndarray) -> np.ndarray | None:
        hh, ww = mask.shape
        if hh % k != 0 or ww % k != 0:
            return None
        out_h, out_w = int(hh // k), int(ww // k)
        out = np.zeros((out_h, out_w), dtype=bool)
        for r in range(out_h):
            for c in range(out_w):
                block = mask[int(r * k) : int((r + 1) * k), int(c * k) : int((c + 1) * k)]
                if bool(block.all()):
                    out[r, c] = True
                elif bool(block.any()):
                    return None
        return out

    def dihedral_variants(
        patch: np.ndarray, mask: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        base = []
        for kk in range(4):
            p = np.rot90(patch, int(kk))
            m = np.rot90(mask, int(kk))
            base.append((p, m))
            base.append((np.fliplr(p), np.fliplr(m)))
        uniq = []
        seen = set()
        for p, m in base:
            key = m.tobytes()
            if key in seen:
                continue
            seen.add(key)
            uniq.append((p, m))
        return uniq

    # Precompute all usable (piece,variant) candidates with their block masks.
    candidates = []
    for pi, p in enumerate(pieces):
        for vp, vm in dihedral_variants(p["patch"], p["mask"]):
            um = downsample_blocks(vm)
            if um is None:
                continue
            candidates.append({"pi": int(pi), "area": int(p["area"]), "um": um, "vp": vp, "vm": vm})

    used_piece: set[int] = set()
    placements: list[tuple[dict, dict]] = []
    for s in slots:
        need_area = int(s["area"]) * int(k) * int(k)
        found = None
        for cand in candidates:
            if int(cand["pi"]) in used_piece:
                continue
            if int(cand["area"]) != int(need_area):
                continue
            if cand["um"].shape != s["mask"].shape:
                continue
            if np.array_equal(cand["um"], s["mask"]):
                found = cand
                break
        assert found is not None, "every slot must match a piece"
        used_piece.add(int(found["pi"]))
        placements.append((s, found))

    outH = (int(layout.shape[0]) + 2) * int(k)
    outW = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((outH, outW), int(bgc), dtype=int)

    for s, cand in placements:
        top = (int(s["r0"]) + 1) * int(k)
        left = (int(s["c0"]) + 1) * int(k)
        paste(out, cand["vp"], int(top), int(left), mask=cand["vm"], overwrite=True)

    return out

# --------------- generator (claude-att1) ---------------

"""Generator for ARC-AGI-2 task 898e7135.

Output-first strategy:
  1. Build a hole/wall layout of 4-conn slots (8-separated, includes unit slots).
  2. For each slot, pick a distinct color + dihedral transform; upscale slot mask
     by k to get the piece sprite (solid k*k blocks).
  3. Assemble the frame (layout + 1-cell border of frame_color).
  4. Scatter pieces outside the frame on a bg=0 canvas, plus single-pixel
     distractor noise in a fresh color.

The generator validates main(input) == expected_output before returning.
"""

import random
import numpy as np

from arc_utils import paste, random_free_location

# --- tunables ----------------------------------------------------------------

K_CHOICES = [2, 3]                  # block scale factor
LAYOUT_H_RANGE = (4, 12)           # inner layout rows
LAYOUT_W_RANGE = (4, 8)            # inner layout cols
N_SLOTS_RANGE = (3, 6)             # total slot count (incl. unit)
N_UNITS_MAX = 2                    # max unit-slot count per example
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]  # area of non-unit slots

HOLE_DENSITY_RANGE = (0.20, 0.50)  # valid hole fraction of layout area
UNIT_SLOT_RETRIES = 20             # attempts per unit slot placement
NON_UNIT_SLOT_RETRIES = 30         # attempts per non-unit slot placement
GROW_SEED_TRIES = 8                # random seed cells tried per _grow_slot call

CANVAS_DIM_RANGE = (15, 30)        # min/max for canvas height and width
AREA_BUFFER_MULT = 4               # multiplier on piece area for canvas sizing
AREA_BUFFER_CONST = 50             # additive constant for canvas area budget

DISTRACTOR_RANGE = (8, 13)         # singleton distractor pixel count
DISTRACTOR_RETRY_MULT = 10         # retry multiplier for distractor placement
MIN_DISTRACTORS = 8                # minimum placed distractors to accept

GENERATE_MAX_RETRIES = 300         # outer retry limit in generate_input


# --- layout generation -------------------------------------------------------

def _canonical_shape_key(mask: np.ndarray) -> bytes:
    """Canonical bytes representation under D4 for a bool mask (tight-cropped)."""
    variants = []
    for kk in range(4):
        r = np.rot90(mask, kk)
        variants.append(r.tobytes() + bytes([r.shape[0], r.shape[1]]))
        fr = np.fliplr(r)
        variants.append(fr.tobytes() + bytes([fr.shape[0], fr.shape[1]]))
    return min(variants)


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    """Grow a 4-conn slot of given size starting from a random free cell.
    `forbidden` marks cells that must not be used. Returns coords array or None.
    """
    H, W = forbidden.shape
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
        return None
    random.shuffle(free_cells)
    for start in free_cells[:GROW_SEED_TRIES]:
        r0, c0 = int(start[0]), int(start[1])
        chosen = {(r0, c0)}
        frontier = {(r0 - 1, c0), (r0 + 1, c0), (r0, c0 - 1), (r0, c0 + 1)}
        while len(chosen) < target_size and frontier:
            cands = [p for p in frontier if 0 <= p[0] < H and 0 <= p[1] < W
                     and not forbidden[p[0], p[1]] and p not in chosen]
            if not cands:
                break
            nr, nc = random.choice(cands)
            chosen.add((nr, nc))
            frontier.discard((nr, nc))
            frontier.update({(nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)})
        if len(chosen) == target_size:
            return np.array(sorted(chosen))
    return None


def _dilate8(mask: np.ndarray) -> np.ndarray:
    """Dilate a bool mask by 1 cell (8-connectivity)."""
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
    """Mark coords and their 8-neighbors as forbidden (in-place)."""
    cell_mask = np.zeros_like(forbidden)
    cell_mask[coords[:, 0], coords[:, 1]] = True
    forbidden |= _dilate8(cell_mask)


def _gen_layout(h: int, w: int, n_slots: int, n_units: int) -> list[dict] | None:
    """Generate a layout grid with n_slots 4-connected slots (including n_units unit slots).
    Returns list of {coords, area, mask, shape_key} or None on failure.
    Unit slots appear first in the returned list.
    """
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []
    used_shape_keys: set[bytes] = set()

    # Place unit slots first (trivially small, easy to fit)
    for _ in range(n_units):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is not None:
                slots.append({"coords": coords, "area": 1})
                _forbid_coords(forbidden, coords)
                break
        else:
            return None

    # Place non-unit slots with distinct D4 shapes
    for _ in range(n_slots - n_units):
        size = random.choice(NON_UNIT_SIZE_CHOICES)
        for _ in range(NON_UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, size)
            if coords is None:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            sm = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            sm[coords[:, 0] - r0, coords[:, 1] - c0] = True
            key = _canonical_shape_key(sm)
            if key in used_shape_keys:
                continue
            used_shape_keys.add(key)
            slots.append({"coords": coords, "area": size, "mask": sm, "shape_key": key})
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    # Validate hole density
    density = sum(s["area"] for s in slots) / (h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (lo <= density <= hi):
        return None

    return slots


# --- piece construction ------------------------------------------------------

def _make_piece(slot: dict, color: int, k: int, apply_dihedral: bool) -> dict:
    """Build output and input patches for one slot.
    For unit slots (area=1), the piece is a solid k*k square with no dihedral
    transform needed. For non-unit slots, a random dihedral is applied to the
    input-side patch.
    """
    if slot["area"] == 1:
        patch = np.full((k, k), color, dtype=int)
        mask = np.ones((k, k), dtype=bool)
        return {"color": color, "slot": slot,
                "out_patch": patch, "out_mask": mask,
                "in_patch": patch, "in_mask": mask}

    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, color, 0).astype(int)

    if apply_dihedral:
        rot = random.randint(0, 3)
        in_mask = np.rot90(out_mask, rot)
        in_patch = np.rot90(out_patch, rot)
        if random.random() < 0.5:
            in_mask = np.fliplr(in_mask)
            in_patch = np.fliplr(in_patch)
    else:
        in_mask, in_patch = out_mask, out_patch

    return {"color": color, "slot": slot,
            "out_patch": out_patch, "out_mask": out_mask,
            "in_patch": in_patch, "in_mask": in_mask}


# --- output assembly ---------------------------------------------------------

def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int,
                           k: int, frame_color: int) -> np.ndarray:
    """Assemble the expected output grid: frame_color background with pieces placed."""
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), frame_color, dtype=int)
    for p in pieces:
        coords = p["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * k
        left = (int(coords[:, 1].min()) + 1) * k
        paste(out, p["out_patch"], top, left, mask=p["out_mask"], overwrite=True)
    return out


# --- input canvas assembly ---------------------------------------------------

def _build_input_canvas(pieces: list[dict], layout_mask: np.ndarray,
                        layout_h: int, layout_w: int, k: int,
                        frame_color: int, distractor_color: int) -> np.ndarray | None:
    """Build the input grid: frame + scattered pieces + distractor noise.
    Returns canvas or None if placement fails.
    """
    frame_h, frame_w = layout_h + 2, layout_w + 2
    total_piece_area = sum(int(p["in_mask"].sum()) for p in pieces)
    needed_cells = frame_h * frame_w + AREA_BUFFER_MULT * total_piece_area + AREA_BUFFER_CONST

    # Canvas dimensions: start random within spec range, grow if area insufficient
    min_h = max(CANVAS_DIM_RANGE[0], frame_h + 2)
    min_w = max(CANVAS_DIM_RANGE[0], frame_w + 2)
    canvas_h = random.randint(min_h, CANVAS_DIM_RANGE[1])
    canvas_w = random.randint(min_w, CANVAS_DIM_RANGE[1])
    while canvas_h * canvas_w < needed_cells and (canvas_h < CANVAS_DIM_RANGE[1] or canvas_w < CANVAS_DIM_RANGE[1]):
        if canvas_h <= canvas_w and canvas_h < CANVAS_DIM_RANGE[1]:
            canvas_h += 1
        elif canvas_w < CANVAS_DIM_RANGE[1]:
            canvas_w += 1
        else:
            break
    if canvas_h * canvas_w < needed_cells:
        return None

    canvas = np.zeros((canvas_h, canvas_w), dtype=int)

    # Build and place the frame (1-cell border of frame_color around the layout)
    frame_img = np.full((frame_h, frame_w), frame_color, dtype=int)
    frame_img[1:-1, 1:-1] = np.where(layout_mask, 0, frame_color)
    fr_max, fc_max = canvas_h - frame_h - 1, canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

    # Place pieces outside the frame, largest-first for better packing
    for p in sorted(pieces, key=lambda p: -int(p["in_mask"].sum())):
        try:
            pr, pc = random_free_location(
                canvas, p["in_patch"],
                transparent=0, border=0, padding=1, padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, p["in_patch"], pr, pc, mask=p["in_mask"], overwrite=True)

    # Scatter distractor singletons (isolated single pixels in a dedicated color)
    n_distractors = random.randint(*DISTRACTOR_RANGE)
    placed = 0
    for _ in range(n_distractors * DISTRACTOR_RETRY_MULT):
        if placed >= n_distractors:
            break
        rr, cc = random.randint(0, canvas_h - 1), random.randint(0, canvas_w - 1)
        if canvas[rr, cc] != 0:
            continue
        r_lo, r_hi = max(0, rr - 1), min(canvas_h, rr + 2)
        c_lo, c_hi = max(0, cc - 1), min(canvas_w, cc + 2)
        if np.any(canvas[r_lo:r_hi, c_lo:c_hi] != 0):
            continue
        canvas[rr, cc] = distractor_color
        placed += 1
    if placed < MIN_DISTRACTORS:
        return None

    # bg=0 must remain the most common color for solver's get_bg
    vals, counts = np.unique(canvas, return_counts=True)
    if int(vals[np.argmax(counts)]) != 0:
        return None

    return canvas


# --- main generator attempt --------------------------------------------------

def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)
    n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 1))

    # Allocate colors: frame, distractor, then per-piece
    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    needed_piece_colors = (n_slots - n_units) + 1  # +1 shared unit color
    if len(palette) < needed_piece_colors:
        return None
    non_unit_colors = palette[:n_slots - n_units]
    unit_color = palette[n_slots - n_units]

    # Build layout
    slots = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if slots is None:
        return None
    unit_slots = slots[:n_units]
    non_unit_slots = slots[n_units:]

    # Validate that holes touch all 4 borders (so solver's tight-crop matches our layout)
    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for s in slots:
        layout_mask[s["coords"][:, 0], s["coords"][:, 1]] = True
    if not (layout_mask[0].any() and layout_mask[-1].any()
            and layout_mask[:, 0].any() and layout_mask[:, -1].any()):
        return None

    # Build pieces
    pieces = [_make_piece(s, non_unit_colors[i], k, apply_dihedral=True)
              for i, s in enumerate(non_unit_slots)]
    pieces += [_make_piece(s, unit_color, k, apply_dihedral=False)
               for s in unit_slots]

    # Build expected output
    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)

    # Build input canvas
    canvas = _build_input_canvas(pieces, layout_mask, layout_h, layout_w, k,
                                 frame_color, distractor_color)
    if canvas is None:
        return None

    # Oracle check
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
    raise RuntimeError("generate_input: failed to generate after %d attempts" % GENERATE_MAX_RETRIES)
