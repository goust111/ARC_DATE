# Combo variant of 898e7135 (index 53)
# combo_spec: D4/2+D8/2+D14/0
# Name: 双图例解码系统

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
Design analysis for the combo variant.

spec_variant:
  preserved_core:
    - One large frame still encloses a slot layout.
    - One outside sprite still corresponds to each slot.
    - Each sprite is still a k-scaled version of its slot shape up to D4 symmetry.
    - Output is still the cropped, scaled frame with sprites inserted.
  combo_change:
    - D14/0: slots are no longer bg holes; each slot is painted in its own marker color.
      A corner legend tells which piece color belongs to which marker color.
    - D8/2 + D4/2: scale is no longer inferred from a smallest square piece.
      A second corner legend shows a solid k-by-k color square; in this combo k is
      fixed to 2, but the solver still decodes it from the legend instead of from pieces.
  concretization:
    - Top-left corner: a vertical 2-column legend, one row per slot:
      (slot_marker_color, piece_color).
    - Top-right corner: a solid 2x2 square in a dedicated scale color.
    - Frame interior: frame_color walls plus three colored slot regions.
    - Outside frame: three monochrome sprites separated from each other and from both legends.
  behavior_trace:
    - Original: detect bg holes -> infer k from smallest solid square -> match pieces to holes by D4 shape.
    - Combo: read the scale square to get k=2 -> read the color-pair legend -> detect colored slots
      inside the frame -> restrict each slot to the legend-specified piece color -> use D4 shape
      matching only to orient the chosen sprite.
    - Difference: both the slot extraction rule and the k inference rule changed, and color now
      carries symbolic meaning instead of being a decorative sprite attribute.
  complexity_calibration:
    - Every sample has exactly three slots so the two legends remain visually simple and the ARC
      palette stays uncluttered.
    - Two slots deliberately share the same D4 shape but use different marker and piece colors.
      This makes the color legend necessary rather than optional.
    - No unit slots are generated, so the old "smallest solid square reveals k" shortcut is gone.
    - k only takes one value in this combo (2), but it is still visibly encoded as a separate symbol.
  implicit_assumption_audit:
    - Original `hole = sub == bg` is invalid because slots are colored; the new solver uses
      `layout != frame_color` to locate slots.
    - Original piece-only k inference is invalid because unit clues are removed; the new solver
      reads the top-right square and never consults piece sizes to decide k.
    - Original piece extraction would accidentally include both legends; the new solver masks out
      the top-left pair legend and the top-right scale square before collecting pieces.
    - Original color-agnostic matching is invalid because two slots have the same geometry; the new
      solver first applies the color legend, then uses D4 geometry to orient each chosen sprite.
  placement_legality:
    - Legends are pasted first into fixed corners.
    - Frame and sprites are then placed with `random_free_location(..., padding=1)`, which prevents
      touching, overlap, and accidental component merging.
    - The frame remains the largest foreground component because its bbox is much larger than either
      legend or any single sprite.
  visual_observability:
    - Humans can see two distinct instruction zones immediately: a 2-column color table and a 2x2
      confirmation square in the opposite corner.
    - The repeated slot shape creates a visible ambiguity that the color legend resolves across
      examples, so the symbolic rule is inferable from a few I/O pairs.
  visual_cleanliness:
    - The two legends live in different corners, so no region mixes multiple instruction systems.
    - Marker colors and piece colors are drawn from disjoint sets, making the pair legend easy to parse.
    - Distractors stay as isolated single pixels, so they do not obscure either legend or any sprite.
  generator_guarantees_for_new_solver_assumptions:
    - Each slot marker color appears exactly once in the color legend.
    - Each piece color is unique, so a legend row selects exactly one outside sprite.
    - Each sprite is a perfect 2x2 block-upscale of its slot shape, so downsampling by the decoded
      k succeeds without partial blocks.
    - The repeated-shape pair ensures that ignoring the color legend changes the placement logic.
"""


# --- tunables ----------------------------------------------------------------

K_FIXED = 2
N_SLOTS = 3

TEMP_LAYOUT_H_RANGE = (5, 10)
TEMP_LAYOUT_W_RANGE = (5, 8)
LAYOUT_H_RANGE = (4, 12)
LAYOUT_W_RANGE = (4, 8)

SHAPE_AREA_CHOICES = [2, 3, 4, 5]
UNIQUE_AREA_CHOICES = [3, 4, 5, 6]
SHAPE_GEN_RETRIES = 80
PLACEMENT_RETRIES = 80
LAYOUT_RETRIES = 120

HOLE_DENSITY_RANGE = (0.20, 0.50)
GROW_SEED_TRIES = 8

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 60

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 300


# --- solver helpers ----------------------------------------------------------

def _read_color_legend(g: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Read the top-left 2-column legend as (marker_color, piece_color) rows."""
    h, w = g.shape
    assert w >= 2, "grid too narrow for color legend"
    rows: list[tuple[int, int]] = []
    r = 0
    while r < h and int(g[r, 0]) != int(bg) and int(g[r, 1]) != int(bg):
        rows.append((int(g[r, 0]), int(g[r, 1])))
        r += 1
    assert len(rows) == N_SLOTS, "color legend must contain one row per slot"
    assert len({m for m, _ in rows}) == len(rows), "marker colors must be unique"
    assert len({p for _, p in rows}) == len(rows), "piece colors must be unique"
    legend_mask = np.zeros_like(g, dtype=bool)
    legend_mask[: len(rows), :2] = True
    return rows, legend_mask


def _read_scale_legend(g: np.ndarray, bg: int) -> tuple[int, np.ndarray]:
    """Read the solid square anchored at the top-right corner and return its side length."""
    h, w = g.shape
    scale_color = int(g[0, w - 1])
    assert scale_color != int(bg), "top-right scale legend must be present"

    kw = 0
    while kw < w and int(g[0, w - 1 - kw]) == scale_color:
        kw += 1
    kh = 0
    while kh < h and int(g[kh, w - 1]) == scale_color:
        kh += 1
    k = int(min(kw, kh))
    assert k >= 2, "scale legend must be at least 2x2"
    assert np.all(g[:k, w - k :] == scale_color), "scale legend must be a solid square"
    if k < h:
        assert not np.any(g[k, w - k :] == scale_color), "scale legend must stop after k rows"
    if w - k - 1 >= 0:
        assert not np.any(g[:k, w - k - 1] == scale_color), "scale legend must stop after k cols"

    legend_mask = np.zeros_like(g, dtype=bool)
    legend_mask[:k, w - k :] = True
    return k, legend_mask


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = mask.shape
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


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    base = []
    for kk in range(4):
        rp = np.rot90(patch, kk)
        rm = np.rot90(mask, kk)
        base.append((rp, rm))
        base.append((np.fliplr(rp), np.fliplr(rm)))
    uniq: list[tuple[np.ndarray, np.ndarray]] = []
    seen = set()
    for vp, vm in base:
        key = (vm.shape, vm.tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((vp, vm))
    return uniq


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    """Canonical bytes representation under D4 for a tight-cropped bool mask."""
    variants = []
    for kk in range(4):
        r = np.rot90(mask, kk)
        variants.append(r.tobytes() + bytes([r.shape[0], r.shape[1]]))
        fr = np.fliplr(r)
        variants.append(fr.tobytes() + bytes([fr.shape[0], fr.shape[1]]))
    return min(variants)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode two corner legends, enlarge the frame layout, and insert each sprite into its marked slot.

    Input structure:
    - bg is the most common color.
    - The top-left corner is a 2-column color legend: each row pairs a slot-marker color with a sprite color.
    - The top-right corner is a solid k×k square that encodes the scale factor; in this combo the square is 2×2.
    - The largest foreground component is a frame whose interior contains frame-color walls and colored slot regions.
    - Outside the frame sit monochrome sprites built from k×k blocks, possibly rotated or mirrored.

    Rule:
    - Read the scale square to recover k, then read the color-pair legend.
    - Extract each colored slot inside the frame and map its marker color to the required sprite color.
    - For that chosen sprite, test D4 variants and downsample by k until its block-level shape matches the slot.
    - Output the cropped frame layout scaled by k, with the oriented sprites pasted into their target slots.

    Key insight:
    - Geometry alone is insufficient because two slots share the same D4 shape; the color legend resolves which
      repeated-shape sprite goes where, and the separate square legend supplies the block scale without relying on
      the old smallest-square shortcut.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    legend_rows, color_legend_mask = _read_color_legend(g, bg)
    k, scale_legend_mask = _read_scale_legend(g, bg)
    legend_mask = color_legend_mask | scale_legend_mask
    marker_to_piece = {int(marker): int(piece) for marker, piece in legend_rows}

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain foreground components"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    slot_occ = sub != int(frame_color)
    slot_box = bbox(slot_occ)
    assert slot_box is not None, "frame must contain colored slots"
    lr0, lc0, lr1, lc1 = map(int, slot_box)
    layout = sub[lr0 : lr1 + 1, lc0 : lc1 + 1]
    layout_mask = layout != int(frame_color)

    slots = []
    for coords in components_mask(layout_mask, connectivity=4):
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
        sm = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        colors = {int(layout[int(r), int(c)]) for r, c in coords}
        assert len(colors) == 1, "each slot must be monochrome"
        marker = int(next(iter(colors)))
        assert marker in marker_to_piece, "every slot marker must appear in the legend"
        slots.append(
            {
                "area": int(sm.sum()),
                "r0": int(rr0),
                "c0": int(cc0),
                "mask": sm,
                "marker": marker,
            }
        )
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
    assert len(slots) == len(legend_rows), "slot count must match color legend rows"

    pieces = []
    for comp in comps:
        if comp is frame_comp:
            continue
        coords = comp["coords"]
        if coords.size == 0 or int(coords.shape[0]) <= 1:
            continue
        if bool(np.all(legend_mask[coords[:, 0], coords[:, 1]])):
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        pm = np.zeros((pr1 - pr0 + 1, pc1 - pc0 + 1), dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        colors = {int(g[int(r), int(c)]) for r, c in coords}
        assert len(colors) == 1, "each sprite must be monochrome"
        pieces.append(
            {
                "pi": len(pieces),
                "area": int(pm.sum()),
                "mask": pm,
                "patch": patch,
                "color": int(next(iter(colors))),
            }
        )

    assert len(pieces) == len(slots), "there must be one non-legend sprite per slot"

    candidates = []
    for piece in pieces:
        for vp, vm in _dihedral_variants(piece["patch"], piece["mask"]):
            um = _downsample_blocks(vm, k)
            if um is None:
                continue
            candidates.append(
                {
                    "pi": int(piece["pi"]),
                    "area": int(piece["area"]),
                    "um": um,
                    "vp": vp,
                    "vm": vm,
                    "color": int(piece["color"]),
                }
            )

    used_piece: set[int] = set()
    placements: list[tuple[dict, dict]] = []
    for slot in slots:
        need_color = int(marker_to_piece[int(slot["marker"])])
        need_area = int(slot["area"]) * int(k) * int(k)
        found = None
        for cand in candidates:
            if int(cand["pi"]) in used_piece:
                continue
            if int(cand["color"]) != int(need_color):
                continue
            if int(cand["area"]) != int(need_area):
                continue
            if cand["um"].shape != slot["mask"].shape:
                continue
            if np.array_equal(cand["um"], slot["mask"]):
                found = cand
                break
        assert found is not None, "every slot must match its legend-selected sprite"
        used_piece.add(int(found["pi"]))
        placements.append((slot, found))

    out = np.full(((layout.shape[0] + 2) * k, (layout.shape[1] + 2) * k), int(frame_color), dtype=int)
    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, cand["vp"], top, left, mask=cand["vm"], overwrite=True)
    return out


# --- generator helpers -------------------------------------------------------

def _tight_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _grow_shape(target_size: int) -> np.ndarray | None:
    """Grow a tight 4-connected shape mask of the given area."""
    box_h = max(3, min(5, target_size + 1))
    box_w = max(3, min(5, target_size + 1))
    forbidden = np.zeros((box_h, box_w), dtype=bool)
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
        return None
    order = list(range(len(free_cells)))
    random.shuffle(order)
    for idx in order[:GROW_SEED_TRIES]:
        r0, c0 = map(int, free_cells[idx])
        chosen = {(r0, c0)}
        frontier = {(r0 - 1, c0), (r0 + 1, c0), (r0, c0 - 1), (r0, c0 + 1)}
        while len(chosen) < target_size and frontier:
            cands = [
                p
                for p in frontier
                if 0 <= p[0] < box_h and 0 <= p[1] < box_w and p not in chosen
            ]
            if not cands:
                break
            nr, nc = random.choice(cands)
            chosen.add((nr, nc))
            frontier.discard((nr, nc))
            frontier.update({(nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)})
        if len(chosen) == target_size:
            coords = np.array(sorted(chosen), dtype=int)
            mask = _tight_mask_from_coords(coords)
            h, w = mask.shape
            if h == w and int(mask.sum()) == int(h * w):
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
    mask = np.zeros_like(forbidden)
    mask[coords[:, 0], coords[:, 1]] = True
    forbidden |= _dilate8(mask)


def _dihedral_masks(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    seen = set()
    for kk in range(4):
        r = np.rot90(mask, kk)
        for cand in (r, np.fliplr(r)):
            key = (cand.shape, cand.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(cand.copy())
    return variants


def _try_place_mask(forbidden: np.ndarray, variants: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    """Place one variant on free cells and return (coords, used_mask)."""
    h, w = forbidden.shape
    order = list(range(len(variants)))
    random.shuffle(order)
    for vidx in order:
        mask = variants[vidx]
        mh, mw = mask.shape
        if mh > h or mw > w:
            continue
        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(0, h - mh)
            left = random.randint(0, w - mw)
            coords = np.argwhere(mask) + np.array([top, left], dtype=int)
            if np.any(forbidden[coords[:, 0], coords[:, 1]]):
                continue
            _forbid_coords(forbidden, coords)
            return coords.astype(int), mask
    return None


def _generate_layout() -> tuple[list[dict], int, int] | None:
    """Generate three slots with one repeated D4 shape pair and one distinct shape."""
    for _ in range(LAYOUT_RETRIES):
        temp_h = random.randint(*TEMP_LAYOUT_H_RANGE)
        temp_w = random.randint(*TEMP_LAYOUT_W_RANGE)
        forbidden = np.zeros((temp_h, temp_w), dtype=bool)

        dup_mask = None
        for _ in range(SHAPE_GEN_RETRIES):
            candidate = _grow_shape(random.choice(SHAPE_AREA_CHOICES))
            if candidate is None:
                continue
            dup_mask = candidate
            break
        if dup_mask is None:
            continue
        dup_key = _canonical_shape_key(dup_mask)
        dup_variants = _dihedral_masks(dup_mask)

        repeated_slots = []
        ok = True
        for _ in range(2):
            placed = _try_place_mask(forbidden, dup_variants)
            if placed is None:
                ok = False
                break
            coords, used_mask = placed
            repeated_slots.append(
                {
                    "coords": coords,
                    "mask": used_mask.copy(),
                    "shape_key": dup_key,
                    "area": int(used_mask.sum()),
                }
            )
        if not ok:
            continue

        unique_mask = None
        unique_key = None
        for _ in range(SHAPE_GEN_RETRIES):
            candidate = _grow_shape(random.choice(UNIQUE_AREA_CHOICES))
            if candidate is None:
                continue
            key = _canonical_shape_key(candidate)
            if key == dup_key:
                continue
            unique_mask = candidate
            unique_key = key
            break
        if unique_mask is None or unique_key is None:
            continue

        placed = _try_place_mask(forbidden, _dihedral_masks(unique_mask))
        if placed is None:
            continue
        coords, used_mask = placed
        slots = repeated_slots + [
            {
                "coords": coords,
                "mask": used_mask.copy(),
                "shape_key": unique_key,
                "area": int(used_mask.sum()),
            }
        ]

        occ = np.zeros((temp_h, temp_w), dtype=bool)
        for slot in slots:
            occ[slot["coords"][:, 0], slot["coords"][:, 1]] = True
        box = bbox(occ)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        layout_h = int(r1 - r0 + 1)
        layout_w = int(c1 - c0 + 1)
        if not (LAYOUT_H_RANGE[0] <= layout_h <= LAYOUT_H_RANGE[1]):
            continue
        if not (LAYOUT_W_RANGE[0] <= layout_w <= LAYOUT_W_RANGE[1]):
            continue
        density = float(sum(int(slot["area"]) for slot in slots)) / float(layout_h * layout_w)
        if not (HOLE_DENSITY_RANGE[0] <= density <= HOLE_DENSITY_RANGE[1]):
            continue

        norm_slots = []
        for slot in slots:
            shifted = slot["coords"] - np.array([r0, c0], dtype=int)
            norm_slots.append(
                {
                    "coords": shifted.astype(int),
                    "mask": slot["mask"].copy(),
                    "shape_key": slot["shape_key"],
                    "area": int(slot["area"]),
                }
            )
        return norm_slots, layout_h, layout_w
    return None


def _make_piece(slot: dict, piece_color: int, k: int) -> dict:
    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, int(piece_color), 0).astype(int)

    rot = random.randint(0, 3)
    in_mask = np.rot90(out_mask, rot)
    in_patch = np.rot90(out_patch, rot)
    if random.random() < 0.5:
        in_mask = np.fliplr(in_mask)
        in_patch = np.fliplr(in_patch)

    return {
        "piece_color": int(piece_color),
        "slot": slot,
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
    }


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for piece in pieces:
        coords = piece["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * k
        left = (int(coords[:, 1].min()) + 1) * k
        paste(out, piece["out_patch"], top, left, mask=piece["out_mask"], overwrite=True)
    return out


def _build_layout_colors(slots: list[dict], layout_h: int, layout_w: int, frame_color: int) -> np.ndarray:
    layout = np.full((layout_h, layout_w), int(frame_color), dtype=int)
    for slot in slots:
        layout[slot["coords"][:, 0], slot["coords"][:, 1]] = int(slot["marker_color"])
    return layout


def _build_color_legend(entries: list[tuple[int, int]]) -> np.ndarray:
    legend = np.zeros((len(entries), 2), dtype=int)
    for r, (marker_color, piece_color) in enumerate(entries):
        legend[r, 0] = int(marker_color)
        legend[r, 1] = int(piece_color)
    return legend


def _build_input_canvas(
    pieces: list[dict],
    layout_colors: np.ndarray,
    layout_h: int,
    layout_w: int,
    frame_color: int,
    color_legend_entries: list[tuple[int, int]],
    scale_color: int,
    distractor_color: int,
) -> np.ndarray | None:
    frame_h, frame_w = layout_h + 2, layout_w + 2
    color_legend = _build_color_legend(color_legend_entries)
    scale_legend = np.full((K_FIXED, K_FIXED), int(scale_color), dtype=int)
    total_piece_area = sum(int(piece["in_mask"].sum()) for piece in pieces)
    needed_cells = (
        frame_h * frame_w
        + color_legend.size
        + scale_legend.size
        + AREA_BUFFER_MULT * total_piece_area
        + AREA_BUFFER_CONST
    )

    min_h = max(CANVAS_DIM_RANGE[0], frame_h + 3, color_legend.shape[0] + 3, scale_legend.shape[0] + 3)
    min_w = max(CANVAS_DIM_RANGE[0], frame_w + 3, color_legend.shape[1] + scale_legend.shape[1] + 4)
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
    paste(canvas, color_legend, 0, 0)
    paste(canvas, scale_legend, 0, canvas_w - K_FIXED)

    frame_img = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    frame_img[1:-1, 1:-1] = layout_colors
    try:
        fr, fc = random_free_location(canvas, frame_img, transparent=0, border=0, padding=1, padding_connectivity=8)
    except ValueError:
        return None
    paste(canvas, frame_img, fr, fc)

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
        paste(canvas, piece["in_patch"], pr, pc, mask=piece["in_mask"], overwrite=True)

    n_distractors = random.randint(*DISTRACTOR_RANGE)
    placed = 0
    for _ in range(n_distractors * DISTRACTOR_RETRY_MULT):
        if placed >= n_distractors:
            break
        rr = random.randint(0, canvas_h - 1)
        cc = random.randint(0, canvas_w - 1)
        if canvas[rr, cc] != 0:
            continue
        r0, r1 = max(0, rr - 1), min(canvas_h, rr + 2)
        c0, c1 = max(0, cc - 1), min(canvas_w, cc + 2)
        if np.any(canvas[r0:r1, c0:c1] != 0):
            continue
        canvas[rr, cc] = int(distractor_color)
        placed += 1
    if placed < MIN_DISTRACTORS:
        return None

    vals, counts = np.unique(canvas, return_counts=True)
    if int(vals[np.argmax(counts)]) != 0:
        return None

    return canvas


# --- generator attempt -------------------------------------------------------

def _gen_attempt() -> np.ndarray | None:
    layout_data = _generate_layout()
    if layout_data is None:
        return None
    slots, layout_h, layout_w = layout_data

    palette = list(range(1, 10))
    random.shuffle(palette)
    if len(palette) < 8:
        return None
    frame_color = palette[0]
    scale_color = palette[1]
    marker_colors = palette[2:5]
    piece_colors = palette[5:8]
    distractor_color = random.choice(piece_colors)

    random.shuffle(slots)
    random.shuffle(marker_colors)
    random.shuffle(piece_colors)

    for slot, marker_color in zip(slots, marker_colors):
        slot["marker_color"] = int(marker_color)

    pieces = []
    for slot, piece_color in zip(slots, piece_colors):
        pieces.append(_make_piece(slot, int(piece_color), K_FIXED))

    expected_out = _build_expected_output(pieces, layout_h, layout_w, K_FIXED, frame_color)

    color_legend_entries = [(int(piece["slot"]["marker_color"]), int(piece["piece_color"])) for piece in pieces]
    random.shuffle(color_legend_entries)
    layout_colors = _build_layout_colors(slots, layout_h, layout_w, frame_color)
    canvas = _build_input_canvas(
        pieces,
        layout_colors,
        layout_h,
        layout_w,
        frame_color,
        color_legend_entries,
        scale_color,
        distractor_color,
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
    raise RuntimeError(f"generate_input: failed to generate after {GENERATE_MAX_RETRIES} attempts")
