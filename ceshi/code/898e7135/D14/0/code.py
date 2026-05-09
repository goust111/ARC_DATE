# Variant of 898e7135
# Dimension: D14 — 符号编码与图例 (symbol_instructions)
# Change: 无符号编码 — 所有信息直接通过几何和颜色表达 → 颜色图例指示匹配对应

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
Design analysis for the D14 variant.

spec_variant:
  preserved_core:
    - There is still one large rectangular frame on bg=0.
    - Outside the frame there is still one piece per slot.
    - Each piece is still a k-by-k block-scaled version of its target slot shape,
      and may still be rotated/flipped in the input.
    - Output is still the k-scaled frame crop with the pieces inserted.
  d14_change:
    - The frame interior no longer exposes slots as bg holes.
    - Each slot is filled by a unique marker color.
    - A corner legend is added as a vertical 2-column strip at top-left.
      Each row is (slot_marker_color, piece_color).
    - The solver must read the legend first, then use it to decide which colored
      piece belongs to which colored slot, and finally use geometry only to orient it.
  concretization:
    - Input corner legend: rows 0..n-1, cols 0..1, one entry per slot.
    - Frame interior: frame_color cells are walls; non-frame cells are slot markers.
      Each slot stays 4-connected and all slots remain 8-separated.
    - Pieces: monochrome, outside the frame, separated by >=1 bg cell.
    - Distractors: isolated single pixels reusing one piece color; they are ignored
      because the solver only keeps non-singleton non-frame components.
  behavior_trace:
    - Original logic: detect bg holes inside frame -> extract slot masks -> match
      pieces to slot masks by D4-equivalent shape.
    - New logic: detect colored slot regions inside frame -> read marker->piece map
      from legend -> for each slot, restrict candidates to the legend-specified
      piece color -> use D4 shape matching only to orient/place that chosen piece.
    - Difference: geometry is no longer sufficient to determine the color placement.
      The input now explicitly contains symbolic instructions.
  complexity_calibration:
    - To make the legend necessary instead of decorative, every sample contains
      exactly two unit slots. These two slots have identical geometry, while their
      two pieces use different colors, so at least one color placement ambiguity
      must be resolved by the legend rather than by shape alone.
    - Non-unit slots still use distinct D4 shapes, keeping the original task's
      geometric flavor without exploding ambiguity.
    - Total slot count is reduced to 3-4 so that frame color + slot colors +
      piece colors all fit into ARC's 1..9 palette without introducing extra codes.
  implicit_assumption_audit:
    - Original `hole = sub == bg` assumed slots are bg. This is false now; the new
      solver instead treats `sub != frame_color` as slot occupancy.
    - Original piece matching ignored color. That would be wrong now because two
      same-shape unit pieces would be swappable. The new solver first filters by
      legend-specified piece color, then checks shape/orientation.
    - Original piece extraction treated every non-frame component with area>1 as a
      piece. That would accidentally include the legend, so the new solver masks
      out the fixed legend strip before collecting pieces.
    - Original output background inference used the dominant color of the frame
      component. That remains valid because the frame component still contains more
      frame-color cells than any slot-marker color, and the generator enforces it.
  placement_legality:
    - The legend is fixed at top-left and all other objects are placed with
      `random_free_location(..., padding=1)` after the legend is pasted, so no
      object touches or merges with the legend.
    - The frame remains the largest connected foreground component because its
      area is the whole frame bbox, while each piece is much smaller.
    - Layout border-touching is preserved so the solver's tight crop recovers the
      intended inner layout exactly.
  visual_observability:
    - Humans can see the new rule because there is now a dedicated symbol area in
      the corner, and the frame holes are visibly replaced by coded colored slots.
    - The two unit slots create repeated shape evidence across examples, so the
      legend's role is inferable from just a few I/O pairs.
  color_inference:
    - Colors are not used via numeric ordering or hidden ids.
    - Piece colors and slot marker colors are disjoint sets, making the legend's
      pairings visually explicit instead of allowing accidental direct matching.
"""


# --- tunables ----------------------------------------------------------------

K_CHOICES = [2, 3]
LAYOUT_H_RANGE = (4, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 4)  # capped so frame + slot colors + piece colors fit in 1..9
N_UNITS = 2             # always keep two identical unit shapes to force legend usage
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]

HOLE_DENSITY_RANGE = (0.20, 0.50)
UNIT_SLOT_RETRIES = 20
NON_UNIT_SLOT_RETRIES = 30
GROW_SEED_TRIES = 8

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 60

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 300


# --- solver helpers ----------------------------------------------------------

def _read_corner_legend(g: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Read the top-left 2-column legend as (slot_marker_color, piece_color) rows."""
    h, w = g.shape
    assert w >= 2, "grid too narrow for legend"
    rows: list[tuple[int, int]] = []
    r = 0
    while r < h and int(g[r, 0]) != int(bg) and int(g[r, 1]) != int(bg):
        rows.append((int(g[r, 0]), int(g[r, 1])))
        r += 1
    assert len(rows) >= 3, "legend must contain one row per slot"
    legend_mask = np.zeros_like(g, dtype=bool)
    legend_mask[:len(rows), :2] = True
    assert len({m for m, _ in rows}) == len(rows), "marker colors in legend must be unique"
    return rows, legend_mask


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
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


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    base = []
    for kk in range(4):
        p = np.rot90(patch, int(kk))
        m = np.rot90(mask, int(kk))
        base.append((p, m))
        base.append((np.fliplr(p), np.fliplr(m)))
    uniq = []
    seen = set()
    for p, m in base:
        key = (m.shape, m.tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((p, m))
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
    Inflate a frame layout and fill each colored slot with the piece indicated by the legend.

    Input structure:
    - bg = most common color.
    - Top-left corner contains a vertical legend: each row is (slot_marker_color, piece_color).
    - Largest component is a frame whose interior contains frame-color walls and colored slot regions.
    - Outside pieces are k×k-block-scaled slot shapes, possibly rotated/flipped.

    Rule:
    - Read the legend to map slot-marker colors to piece colors.
    - Extract colored slot regions inside the frame bbox.
    - Match each slot to the unique piece color specified by the legend, then orient
      that piece by block-level shape matching under rotation/flip.
    - Output the scaled frame with the matched pieces pasted into their slots.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    legend_rows, legend_mask = _read_corner_legend(g, bg)
    slot_to_piece = {int(marker): int(piece) for marker, piece in legend_rows}

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain foreground components"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    bgc = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    slot_occ = sub != int(bgc)
    slot_box = bbox(slot_occ)
    assert slot_box is not None, "frame must enclose at least one colored slot"
    lr0, lc0, lr1, lc1 = map(int, slot_box)
    layout = sub[lr0 : lr1 + 1, lc0 : lc1 + 1]
    layout_mask = layout != int(bgc)

    slots = []
    for coords in components_mask(layout_mask, connectivity=4):
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
        h = int(rr1 - rr0 + 1)
        w = int(cc1 - cc0 + 1)
        sm = np.zeros((h, w), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        colors = {int(layout[int(r), int(c)]) for r, c in coords}
        assert len(colors) == 1, "each slot must be monochrome"
        marker = int(next(iter(colors)))
        assert marker in slot_to_piece, "every slot marker must appear in the legend"
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

    assert slots and len(slots) == len(legend_rows), "legend rows must match slot count"

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
        ph, pw = map(int, patch.shape)
        pm = np.zeros((ph, pw), dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        colors = {int(g[int(r), int(c)]) for r, c in coords}
        assert len(colors) == 1, "each piece must be monochrome"
        pieces.append(
            {
                "area": int(pm.sum()),
                "mask": pm,
                "patch": patch,
                "color": int(next(iter(colors))),
            }
        )

    assert pieces and len(pieces) == len(slots), "there must be one non-legend piece per slot"

    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    k = int(k or 1)

    candidates = []
    for pi, p in enumerate(pieces):
        for vp, vm in _dihedral_variants(p["patch"], p["mask"]):
            um = _downsample_blocks(vm, k)
            if um is None:
                continue
            candidates.append(
                {
                    "pi": int(pi),
                    "area": int(p["area"]),
                    "um": um,
                    "vp": vp,
                    "vm": vm,
                    "color": int(p["color"]),
                }
            )

    used_piece: set[int] = set()
    placements: list[tuple[dict, dict]] = []
    for s in slots:
        need_color = int(slot_to_piece[int(s["marker"])])
        need_area = int(s["area"]) * int(k) * int(k)
        found = None
        for cand in candidates:
            if int(cand["pi"]) in used_piece:
                continue
            if int(cand["color"]) != int(need_color):
                continue
            if int(cand["area"]) != int(need_area):
                continue
            if cand["um"].shape != s["mask"].shape:
                continue
            if np.array_equal(cand["um"], s["mask"]):
                found = cand
                break
        assert found is not None, "every slot must match its legend-specified piece"
        used_piece.add(int(found["pi"]))
        placements.append((s, found))

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(bgc), dtype=int)

    for s, cand in placements:
        top = (int(s["r0"]) + 1) * int(k)
        left = (int(s["c0"]) + 1) * int(k)
        paste(out, cand["vp"], int(top), int(left), mask=cand["vm"], overwrite=True)

    return out


# --- layout generation -------------------------------------------------------

def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    """Grow a 4-connected slot of the requested area on free cells."""
    h, w = forbidden.shape
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
                if 0 <= p[0] < h and 0 <= p[1] < w and not forbidden[p[0], p[1]] and p not in chosen
            ]
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


def _gen_layout(h: int, w: int, n_slots: int) -> list[dict] | None:
    """Generate slots with two unit slots and D4-distinct non-unit shapes."""
    assert n_slots >= N_UNITS
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []
    used_shape_keys: set[bytes] = set()

    for _ in range(N_UNITS):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is not None:
                slots.append({"coords": coords, "area": 1})
                _forbid_coords(forbidden, coords)
                break
        else:
            return None

    for _ in range(n_slots - N_UNITS):
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

    density = sum(int(s["area"]) for s in slots) / float(h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (lo <= density <= hi):
        return None

    return slots


# --- piece / frame construction ---------------------------------------------

def _make_piece(slot: dict, piece_color: int, k: int, apply_dihedral: bool) -> dict:
    """Build one piece patch. Unit slots remain solid k×k squares."""
    if int(slot["area"]) == 1:
        patch = np.full((k, k), int(piece_color), dtype=int)
        mask = np.ones((k, k), dtype=bool)
        return {
            "piece_color": int(piece_color),
            "slot": slot,
            "out_patch": patch,
            "out_mask": mask,
            "in_patch": patch,
            "in_mask": mask,
        }

    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, int(piece_color), 0).astype(int)

    if apply_dihedral:
        rot = random.randint(0, 3)
        in_mask = np.rot90(out_mask, rot)
        in_patch = np.rot90(out_patch, rot)
        if random.random() < 0.5:
            in_mask = np.fliplr(in_mask)
            in_patch = np.fliplr(in_patch)
    else:
        in_mask, in_patch = out_mask, out_patch

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
    for p in pieces:
        coords = p["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * k
        left = (int(coords[:, 1].min()) + 1) * k
        paste(out, p["out_patch"], top, left, mask=p["out_mask"], overwrite=True)
    return out


def _build_layout_colors(slots: list[dict], layout_h: int, layout_w: int, frame_color: int) -> np.ndarray:
    layout = np.full((layout_h, layout_w), int(frame_color), dtype=int)
    for slot in slots:
        layout[slot["coords"][:, 0], slot["coords"][:, 1]] = int(slot["marker_color"])
    return layout


def _build_legend(entries: list[tuple[int, int]]) -> np.ndarray:
    legend = np.zeros((len(entries), 2), dtype=int)
    for r, (marker_color, piece_color) in enumerate(entries):
        legend[r, 0] = int(marker_color)
        legend[r, 1] = int(piece_color)
    return legend


# --- input canvas assembly ---------------------------------------------------

def _build_input_canvas(
    pieces: list[dict],
    layout_colors: np.ndarray,
    layout_h: int,
    layout_w: int,
    frame_color: int,
    legend_entries: list[tuple[int, int]],
    distractor_color: int,
) -> np.ndarray | None:
    """Assemble legend + frame + scattered pieces + singleton distractors on bg=0."""
    frame_h, frame_w = layout_h + 2, layout_w + 2
    legend = _build_legend(legend_entries)
    legend_h, legend_w = legend.shape
    total_piece_area = sum(int(p["in_mask"].sum()) for p in pieces)
    needed_cells = frame_h * frame_w + legend_h * legend_w + AREA_BUFFER_MULT * total_piece_area + AREA_BUFFER_CONST

    min_h = max(CANVAS_DIM_RANGE[0], frame_h + 2, legend_h + 4)
    min_w = max(CANVAS_DIM_RANGE[0], frame_w + 2, legend_w + 4)
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
    paste(canvas, legend, 0, 0)

    frame_img = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    frame_img[1:-1, 1:-1] = layout_colors
    try:
        fr, fc = random_free_location(canvas, frame_img, transparent=0, border=0, padding=1, padding_connectivity=8)
    except ValueError:
        return None
    paste(canvas, frame_img, fr, fc)

    for p in sorted(pieces, key=lambda x: -int(x["in_mask"].sum())):
        try:
            pr, pc = random_free_location(
                canvas,
                p["in_patch"],
                transparent=0,
                border=0,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, p["in_patch"], pr, pc, mask=p["in_mask"], overwrite=True)

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
    k = random.choice(K_CHOICES)
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)

    slots = _gen_layout(layout_h, layout_w, n_slots)
    if slots is None:
        return None

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for s in slots:
        layout_mask[s["coords"][:, 0], s["coords"][:, 1]] = True
    if not (layout_mask[0].any() and layout_mask[-1].any() and layout_mask[:, 0].any() and layout_mask[:, -1].any()):
        return None

    palette = list(range(1, 10))
    random.shuffle(palette)
    need_colors = 1 + 2 * n_slots
    if len(palette) < need_colors:
        return None
    frame_color = palette[0]
    marker_colors = palette[1 : 1 + n_slots]
    piece_colors = palette[1 + n_slots : 1 + 2 * n_slots]
    distractor_color = random.choice(piece_colors)

    random.shuffle(marker_colors)
    random.shuffle(piece_colors)
    random.shuffle(slots)
    for slot, marker_color in zip(slots, marker_colors):
        slot["marker_color"] = int(marker_color)

    pieces = []
    for slot, piece_color in zip(slots, piece_colors):
        apply_dihedral = int(slot["area"]) != 1
        pieces.append(_make_piece(slot, int(piece_color), k, apply_dihedral=apply_dihedral))

    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)

    legend_entries = [(int(p["slot"]["marker_color"]), int(p["piece_color"])) for p in pieces]
    random.shuffle(legend_entries)
    layout_colors = _build_layout_colors(slots, layout_h, layout_w, frame_color)
    canvas = _build_input_canvas(
        pieces,
        layout_colors,
        layout_h,
        layout_w,
        frame_color,
        legend_entries,
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
