# Combo variant of 898e7135 (index 48)
# combo_spec: D7/1+D10/1+D15/1
# Name: 颜色位置复制填充

from __future__ import annotations

"""
Design analysis for the fused variant.

spec_variant:
  preserved_core:
    - bg is still 0 and the largest foreground component is still the frame.
    - The output is still the frame's inner hole layout, scaled by k and surrounded
      by a 1-cell frame border in the frame color.
    - Outside objects are still monochrome k-scaled sprites; distractors are still
      isolated singleton pixels in a dedicated color.

  fusion_rule:
    - D15 makes every hole have exactly the same tight mask, so shape can no longer
      distinguish one target position from another.
    - D10 then promotes color from "arbitrary label" to "row code": occupied hole
      rows are ordered from top to bottom and mapped to a small fixed spectrum.
    - D7 finally removes 1:1 cardinality: there is only one physical sprite for each
      row color, and that sprite must be copied into every hole on that row.

  concrete_behavior_trace:
    - Example: suppose the frame contains five identical L-shaped holes arranged as
      row counts [2, 2, 1] from top to bottom.
    - Original task: five different sprites would exist outside the frame, and each
      hole would consume one unused sprite after shape matching.
    - Fused task: only three sprites exist outside the frame, colored red/orange/green
      for the top/middle/bottom occupied rows. The top red sprite is pasted into both
      top-row holes, the orange sprite into both middle-row holes, and the green sprite
      into the single bottom-row hole.
    - Difference summary:
      1. shape matching stops being hole-specific because every hole has the same mask;
      2. row position determines the expected color;
      3. a matched sprite is reused across all holes in the same row group.

  complexity_calibration:
    - To keep the visual pattern explicit, every sample uses exactly three occupied
      row groups and 4-6 total holes, with at least one row containing multiple holes.
    - The shared hole shape is chosen from a small library of connected non-unit masks
      whose bbox dimensions are coprime, so k remains inferable without reintroducing
      a special unit hole.
    - A small fixed row palette is used instead of many arbitrary colors, making the
      row encoding visible across 2-4 examples.

  implicit_assumption_review:
    - Original `used_piece` logic is invalid: it assumes a physical sprite can be
      consumed only once. This fusion explicitly requires repeated use.
    - Original per-slot shape lookup is underconstrained here because all slots share
      the same mask; color must be introduced before placement becomes determinate.
    - Original k-inference from the smallest solid square no longer applies because
      the generator no longer guarantees unit holes. The new solver instead infers k
      as the gcd of piece bbox dimensions, and the generator guarantees coprime base
      bbox dimensions so that gcd equals the true block scale.
    - Grouping by slot top row assumes same-row holes share the same vertical anchor.
      The generator constructs rows explicitly to satisfy that assumption.

  placement_legality:
    - Slots are assembled directly into a cropped layout and separated by at least one
      blank cell horizontally/vertically, which also avoids diagonal touching.
    - The frame, row-coded sprites, and distractor pixels are placed on the outer
      canvas with `random_free_location`, preserving non-overlap and in-bounds layout.
    - Frame color and distractor color are always chosen outside the row palette.

  visual_observability:
    - A human can see that all holes have the same silhouette, so shape cannot explain
      why different rows get different fill colors.
    - The same color appearing in multiple same-row holes, despite only one matching
      outside sprite, makes the copy rule directly observable.
    - The palette order is stable across examples and depends only on top-to-bottom
      row order, not on hidden numeric color ids.

  cleanliness:
    - There is no extra overlay or marker inside the frame; the only visible signals
      are shared hole shape, row grouping, and repeated colors.
    - Restricting the scene to three occupied rows and one sprite per row keeps the
      input uncluttered while still forcing the fused reasoning chain.

  new_solver_assumptions:
    - Every non-distractor piece is monochrome and downscales cleanly into k-sized blocks.
    - All true slots share one exact mask.
    - The set of non-singleton piece colors equals the expected row palette for the
      occupied rows. The generator enforces all three assumptions.
"""

import math
import random
import sys
from collections import Counter

import numpy as np

SOLVER_DIR = "/home/lixinyu/snap/new/solvers"
if SOLVER_DIR not in sys.path:
    sys.path.append(SOLVER_DIR)

from arc_utils import bbox, bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location


ROW_PALETTES = {
    2: [2, 3],
    3: [2, 7, 3],
    4: [2, 7, 4, 3],
}

K_CHOICES = [2, 3]
TOTAL_SLOT_CHOICES = [4, 5, 6]
ROW_COUNT = 3
ROW_SIZE_TEMPLATES = [
    [2, 1, 1],
    [2, 2, 1],
    [2, 2, 2],
]

HOLE_DENSITY_RANGE = (0.20, 0.50)
CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 400

BASE_SHAPES = [
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
]


def _row_palette(row_count: int) -> list[int]:
    assert int(row_count) in ROW_PALETTES
    return list(ROW_PALETTES[int(row_count)])


def _tight_mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, int, int]:
    rr0, cc0, rr1, cc1 = bbox_coords(coords)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
    tight = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
    tight[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
    return tight, int(rr0), int(cc0)


def _exact_shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return int(mask.shape[0]), int(mask.shape[1]), mask.tobytes()


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = map(int, mask.shape)
    if hh % int(k) != 0 or ww % int(k) != 0:
        return None
    out = np.zeros((hh // int(k), ww // int(k)), dtype=bool)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            block = mask[r * int(k) : (r + 1) * int(k), c * int(k) : (c + 1) * int(k)]
            if bool(block.all()):
                out[r, c] = True
            elif bool(block.any()):
                return None
    return out


def _infer_k(pieces: list[dict]) -> int:
    g = 0
    for piece in pieces:
        h, w = map(int, piece["mask"].shape)
        g = math.gcd(g, int(h))
        g = math.gcd(g, int(w))
    assert g in K_CHOICES, "piece bbox gcd must recover k"
    return int(g)


def _parse_grid(grid: np.ndarray) -> tuple[np.ndarray, int, np.ndarray, list[dict], list[dict], int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain foreground components"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == int(bg)
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots: list[dict] = []
    for coords in components_mask(layout, connectivity=4):
        tight_mask, rr0, cc0 = _tight_mask_from_coords(coords)
        slots.append(
            {
                "coords": coords,
                "area": int(tight_mask.sum()),
                "r0": int(rr0),
                "c0": int(cc0),
                "mask": tight_mask,
                "shape_key": _exact_shape_key(tight_mask),
            }
        )
    assert slots, "frame must contain at least one slot"
    slots.sort(key=lambda slot: (int(slot["r0"]), int(slot["c0"])))

    pieces: list[dict] = []
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
        colors = {int(v) for v in g[coords[:, 0], coords[:, 1]].tolist()}
        assert len(colors) == 1, "each piece must be monochrome"
        pieces.append(
            {
                "area": int(pm.sum()),
                "patch": patch,
                "mask": pm,
                "color": int(next(iter(colors))),
            }
        )
    assert pieces, "input must contain at least one non-frame piece"

    k = _infer_k(pieces)
    return g, int(frame_color), layout, slots, pieces, int(k)


def _make_piece(shape_mask: np.ndarray, color: int, k: int) -> dict:
    out_mask = np.kron(shape_mask, np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, int(color), 0).astype(int)
    return {
        "color": int(color),
        "in_patch": out_patch,
        "in_mask": out_mask,
        "out_patch": out_patch,
        "out_mask": out_mask,
    }


def _build_layout(shape_mask: np.ndarray, row_sizes: list[int], offsets: list[int], col_gap: int, row_gap: int) -> tuple[list[dict], np.ndarray]:
    mh, mw = map(int, shape_mask.shape)
    max_row_width = max(int(offsets[idx]) + int(row_sizes[idx]) * mw + (int(row_sizes[idx]) - 1) * int(col_gap) for idx in range(len(row_sizes)))
    layout_h = int(len(row_sizes) * mh + (len(row_sizes) - 1) * row_gap)
    layout_w = int(max_row_width)
    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    slots: list[dict] = []

    for row_idx, count in enumerate(row_sizes):
        top = int(row_idx * (mh + row_gap))
        left0 = int(offsets[row_idx])
        for j in range(int(count)):
            left = int(left0 + j * (mw + col_gap))
            coords = np.argwhere(shape_mask) + np.array([top, left], dtype=int)
            layout_mask[coords[:, 0], coords[:, 1]] = True
            tight_mask, rr0, cc0 = _tight_mask_from_coords(coords)
            slots.append(
                {
                    "coords": coords,
                    "area": int(tight_mask.sum()),
                    "r0": int(rr0),
                    "c0": int(cc0),
                    "mask": tight_mask,
                    "shape_key": _exact_shape_key(tight_mask),
                }
            )
    return slots, layout_mask


def _build_expected_output(slots: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int, color_by_row: dict[int, int]) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for slot in slots:
        color = int(color_by_row[int(slot["r0"])])
        patch = np.where(np.kron(slot["mask"], np.ones((k, k), dtype=bool)), color, 0).astype(int)
        mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, patch, int(top), int(left), mask=mask, overwrite=True)
    return out


def _build_input_canvas(pieces: list[dict], layout_mask: np.ndarray, frame_color: int, distractor_color: int) -> np.ndarray | None:
    layout_h, layout_w = map(int, layout_mask.shape)
    frame_h, frame_w = int(layout_h + 2), int(layout_w + 2)
    total_piece_area = sum(int(piece["in_mask"].sum()) for piece in pieces)
    needed_cells = int(frame_h * frame_w + AREA_BUFFER_MULT * total_piece_area + AREA_BUFFER_CONST)

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
    frame_img = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    frame_img[1:-1, 1:-1] = np.where(layout_mask, 0, int(frame_color))

    fr_max = canvas_h - frame_h - 1
    fc_max = canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

    for piece in pieces:
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


def _row_templates_for_shape(shape_mask: np.ndarray) -> list[list[int]]:
    mw = int(shape_mask.shape[1])
    if mw >= 3:
        return [tmpl for tmpl in ROW_SIZE_TEMPLATES if max(tmpl) <= 2]
    return list(ROW_SIZE_TEMPLATES)


def _slot_density_ok(layout_mask: np.ndarray, slots: list[dict]) -> bool:
    hole_area = sum(int(slot["area"]) for slot in slots)
    density = float(hole_area) / float(layout_mask.shape[0] * layout_mask.shape[1])
    lo, hi = HOLE_DENSITY_RANGE
    return bool(lo <= density <= hi)


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    shape_mask = np.array(random.choice(BASE_SHAPES), dtype=bool)
    row_palette = _row_palette(ROW_COUNT)
    row_sizes = list(random.choice(_row_templates_for_shape(shape_mask)))
    if sum(row_sizes) not in TOTAL_SLOT_CHOICES:
        return None

    mh, mw = map(int, shape_mask.shape)
    row_gap = 1
    col_gap = 1
    layout_h = int(ROW_COUNT * mh + (ROW_COUNT - 1) * row_gap)
    if not (4 <= layout_h <= 12):
        return None

    row_widths = [int(count * mw + (count - 1) * col_gap) for count in row_sizes]
    layout_w = int(max(row_widths))
    if not (4 <= layout_w <= 8):
        return None

    widest_rows = [idx for idx, width in enumerate(row_widths) if int(width) == int(layout_w)]
    anchor_row = random.choice(widest_rows)
    offsets = [0] * ROW_COUNT
    for idx, width in enumerate(row_widths):
        slack = int(layout_w - width)
        if idx == anchor_row:
            offsets[idx] = 0 if slack == 0 else random.randint(0, slack)
        else:
            offsets[idx] = 0 if slack == 0 else random.randint(0, slack)
    if all(offset == offsets[0] for offset in offsets):
        movable = [idx for idx, width in enumerate(row_widths) if layout_w - width > 0]
        if movable:
            idx = random.choice(movable)
            offsets[idx] = random.randint(0, int(layout_w - row_widths[idx]))

    slots, layout_mask = _build_layout(shape_mask, row_sizes, offsets, col_gap, row_gap)
    if not _slot_density_ok(layout_mask, slots):
        return None

    row_anchors = sorted({int(slot["r0"]) for slot in slots})
    if len(row_anchors) != ROW_COUNT:
        return None
    color_by_row = {int(row_anchors[idx]): int(row_palette[idx]) for idx in range(ROW_COUNT)}

    remaining_colors = [color for color in range(1, 10) if color not in set(row_palette)]
    random.shuffle(remaining_colors)
    if len(remaining_colors) < 2:
        return None
    frame_color = int(remaining_colors.pop())
    distractor_color = int(remaining_colors.pop())

    pieces = [_make_piece(shape_mask, int(color_by_row[row]), int(k)) for row in row_anchors]

    frame_area = int((layout_mask.shape[0] + 2) * (layout_mask.shape[1] + 2) - layout_mask.sum())
    if frame_area <= max(int(piece["in_mask"].sum()) for piece in pieces):
        return None

    expected = _build_expected_output(slots, int(layout_mask.shape[0]), int(layout_mask.shape[1]), int(k), int(frame_color), color_by_row)
    canvas = _build_input_canvas(pieces, layout_mask, int(frame_color), int(distractor_color))
    if canvas is None:
        return None

    metrics = _fusion_metrics(canvas)
    if int(metrics["unique_slot_shapes"]) != 1:
        return None
    if int(metrics["piece_count"]) != ROW_COUNT:
        return None
    if int(metrics["max_row_multiplicity"]) < 2:
        return None

    try:
        produced = main(canvas)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return canvas


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    _g, _frame_color, _layout, slots, pieces, _k = _parse_grid(grid)
    row_counts = Counter(int(slot["r0"]) for slot in slots)
    shape_counts = Counter(slot["shape_key"] for slot in slots)
    return {
        "slot_count": int(len(slots)),
        "piece_count": int(len(pieces)),
        "row_count": int(len(row_counts)),
        "reuse_gap": int(len(slots) - len(pieces)),
        "max_row_multiplicity": int(max(row_counts.values())),
        "unique_slot_shapes": int(len(shape_counts)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale the framed hole pattern and fill it by row-coded sprite reuse.

    Input structure:
    - The largest foreground component is a monochrome rectangular frame whose interior
      contains several background holes, and every hole has the same tight shape.
    - Outside the frame there is one monochrome k-scaled sprite for each occupied hole
      row, plus isolated 1-pixel distractors in another color.

    Rule:
    - Read the identical holes inside the frame, group them by their row position, and
      map those rows top-to-bottom onto the fixed row palette.
    - Match each row color to the outside sprite of the same color, then copy that
      sprite into every hole belonging to that row.
    - Return the frame layout enlarged by k, using the frame color as the output background.

    Key insight:
    - Because every hole has the same shape, shape cannot tell holes apart; the decisive
      information is the color-to-row code, and a single outside sprite must often be
      reused for multiple holes in the same row.
    """
    _g, frame_color, layout, slots, pieces, k = _parse_grid(grid)

    slot_keys = {slot["shape_key"] for slot in slots}
    assert len(slot_keys) == 1, "all slots must share one exact shape"
    ref_shape_key = next(iter(slot_keys))

    row_anchors = sorted({int(slot["r0"]) for slot in slots})
    row_palette = _row_palette(len(row_anchors))
    expected_color_by_row = {int(row_anchors[idx]): int(row_palette[idx]) for idx in range(len(row_anchors))}

    piece_by_color: dict[int, dict] = {}
    for piece in pieces:
        um = _downsample_blocks(piece["mask"], k)
        assert um is not None, "piece must be composed of full k blocks"
        assert _exact_shape_key(um) == ref_shape_key, "every piece must match the common slot shape"
        color = int(piece["color"])
        assert color not in piece_by_color, "there must be exactly one reusable sprite per row color"
        piece_by_color[color] = {"patch": piece["patch"], "mask": piece["mask"]}

    assert set(piece_by_color) == set(row_palette), "piece colors must equal the row palette"

    out = np.full(((layout.shape[0] + 2) * k, (layout.shape[1] + 2) * k), int(frame_color), dtype=int)
    for slot in slots:
        expected_color = int(expected_color_by_row[int(slot["r0"])])
        piece = piece_by_color[expected_color]
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, piece["patch"], int(top), int(left), mask=piece["mask"], overwrite=True)

    return out


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
