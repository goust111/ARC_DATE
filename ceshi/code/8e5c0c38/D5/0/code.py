# Variant of 8e5c0c38
# Dimension: D5 — 对称轴定位算法 (transform)
# Change: 遍历所有可能的垂直轴位置（整数轴和半整数轴），选择对称配对得分最高的轴... → 对称轴由外部标记显式指定

from __future__ import annotations

"""
Design analysis for the D5 variant.

variant_spec:
  summary:
    - Keep the original task's core denoising action: each sprite is repaired by
      deleting pixels that do not have a same-color mirror partner.
    - Change only how the axis is obtained: instead of searching all possible
      vertical axes, each sprite receives an explicit red marker (color 2).
    - The marker is external to the sprite: a red single pixel or short vertical
      segment is placed directly above or below the sprite, aligned to the
      sprite's symmetry column. The output removes both the noise and the marker.

  concrete_pixel_behavior:
    input:
      - 22x22 grid with one dominant background color.
      - Exactly 3 non-background sprite colors, each used by one sprite.
      - Every sprite is vertically symmetric around an integer-column axis.
      - Some extra same-color noise pixels are attached near the sprite and do
        not have mirrored partners.
      - A red marker component of height 1, 2, or 3 sits one cell above or below
        the sprite on the symmetry column. Height 1 is the "single-pixel marker";
        height >1 is the "vertical line segment" marker.
    output:
      - Same grid with the red markers removed.
      - For each sprite, only pixels with same-color mirrors across the marked
        axis remain.

  behavior_trace:
    - Original logic at one sprite:
      1. Collect all pixels of one color.
      2. Score every integer/half-integer vertical axis.
      3. Keep only pixels paired under the best-scoring axis.
    - Variant logic at one sprite:
      1. Find the red marker component aligned with that sprite.
      2. Read the axis column directly from the marker's column.
      3. Keep only pixels paired under that given axis.
    - Difference:
      - The solver no longer infers the axis from the noisy color pattern.
      - The input now contains an extra visible structure whose only role is to
        specify the axis.

  complexity_calibration:
    - Marker color is fixed and visible (red = 2), so the new mechanism is easy
      to notice from 2-4 examples.
    - Each grid still contains 3 sprites, giving repeated evidence that red
      markers correspond to axes, not payload.
    - Marker style varies between single-pixel and line-segment forms, but both
      are still just "one-column red markers", so the rule class stays narrow.
    - Half-integer axes are intentionally removed. This is a compatibility
      adaptation for the explicit-marker design: a visible one-column marker can
      point to an integer column unambiguously, whereas a half-column axis would
      require a second encoding convention and would change more than D5.

  implicit_assumption_review_of_original_main:
    - "colors = colors_present(g, exclude={bg})":
      original assumption: every non-background color is a payload color to be
      denoised.
      variant status: false, because red is now metadata. New solver must exclude
      the marker color from sprite processing and erase it in the output.
    - "for col in colors":
      original assumption: one independent axis search per color layer is the
      right decomposition.
      variant status: not sufficient, because the relevant grouping is now
      marker-to-sprite pairing, not blind per-color axis search.
    - "for k in all candidate axes":
      original assumption: the axis must be inferred from pixels.
      variant status: replaced entirely by reading the marker column.
    - "if mirror cell has same color, keep pixel":
      this remains valid and is the actual denoising rule we preserve.
    - "out starts as a copy of g":
      original assumption: copying all non-background pixels is harmless.
      variant status: false for markers; unconditional copy would preserve red
      metadata in the output. New solver clears markers first, then keeps only
      mirrored payload pixels.

  placement_legality:
    - Dedicated slot layout keeps the 3 sprite regions separated.
    - Marker cells and the one-cell gap between marker and sprite are reserved,
      so noise cannot accidentally merge with the marker.
    - Each marker's column is inside its sprite bbox, and the marker sits at bbox
      distance 1 from its sprite; other sprites are much farther away.
    - Because each sprite uses a unique non-red color and is generated as a
      connected component, marker-to-sprite association by nearest spanning
      component is stable.

  visual_observability:
    - The red marker is the only color whose output role is "disappear".
    - It is placed close to exactly one sprite and aligned with that sprite's
      center column, making the correspondence visually discoverable.
    - The cleaned output still shows the same kind of symmetry restoration as the
      original task, so the user can compare input noise against a clearly marked
      axis.

  visual_cleanliness:
    - Markers are external rather than embedded, so they do not overwrite sprite
      pixels or create mixed-color components.
    - The reserved gap cell preserves a clean visual separation between marker
      and sprite.
    - Slot spacing prevents cross-object ambiguity.

  color_change_inference:
    - No payload recoloring is introduced.
    - Red is not sorted against other colors or used numerically; it is simply a
      visibly special marker class that disappears in the output.

  new_main_assumptions_checked_against_generator:
    - Assumption: every marker component lies in one column.
      guaranteed by generator: marker is always a vertical 1x1 / 2x1 / 3x1 piece.
    - Assumption: each sprite has exactly one nearby marker whose column lies
      within that sprite bbox.
      guaranteed by generator: one marker is created for each slot-placed sprite.
    - Assumption: each sprite forms one 4-connected non-marker component.
      guaranteed by generator: clean sprite connectivity is checked, and noise is
      only added 4-adjacent to the clean sprite.
    - Assumption: no noise occupies the reserved marker/gap cells.
      guaranteed by generator: those cells are blocked during noise sampling.
"""

import math
import os
import random
import sys

import numpy as np

SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if SOLVER_UTILS not in sys.path:
    sys.path.append(SOLVER_UTILS)

from arc_utils import get_bg, label_mask, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove unmatched sprite pixels using explicitly marked vertical axes.

    Input structure:
    - bg is the dominant background color. Red (2) is a special marker color.
    - Three colored sprites appear with nearby red single-pixel or short vertical markers that indicate their symmetry columns.
    - Each sprite is almost vertically symmetric, except for same-color noise pixels without mirrored partners.

    Rule:
    - Pair each red marker with the nearby sprite whose bounding box spans that marker column.
    - Use the marker's column as the sprite's vertical reflection axis.
    - Delete every non-red sprite pixel that lacks a same-color mirror counterpart across its marked axis, and erase the red markers.

    Key insight: The challenge is no longer axis discovery; the marker already specifies the correct axis, so the task is to associate each marker with its sprite and apply mirror-based denoising around that given column.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    MARK = 2

    out = g.copy()
    out[g == int(MARK)] = int(bg)

    obj_lab, obj_n = label_mask((g != int(bg)) & (g != int(MARK)), connectivity=4)
    marker_lab, marker_n = label_mask(g == int(MARK), connectivity=4)

    obj_comps: list[dict] = []
    for lbl in range(1, int(obj_n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(obj_lab == int(lbl))]
        if not coords:
            continue
        colors = {int(g[r, c]) for r, c in coords}
        if len(colors) != 1:
            continue
        rs = [r for r, _c in coords]
        cs = [c for _r, c in coords]
        obj_comps.append(
            {
                "id": int(lbl),
                "coords": coords,
                "color": int(next(iter(colors))),
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )

    marker_comps: list[dict] = []
    for lbl in range(1, int(marker_n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(marker_lab == int(lbl))]
        if not coords:
            continue
        cols = {c for _r, c in coords}
        if len(cols) != 1:
            continue
        rs = [r for r, _c in coords]
        c_axis = int(next(iter(cols)))
        marker_comps.append(
            {
                "coords": coords,
                "axis_c": c_axis,
                "bbox": (int(min(rs)), int(c_axis), int(max(rs)), int(c_axis)),
            }
        )

    def bbox_gap(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> int:
        r0a, c0a, r1a, c1a = map(int, b1)
        r0b, c0b, r1b, c1b = map(int, b2)
        if r1a < r0b:
            dr = int(r0b - r1a - 1)
        elif r1b < r0a:
            dr = int(r0a - r1b - 1)
        else:
            dr = 0
        if c1a < c0b:
            dc = int(c0b - c1a - 1)
        elif c1b < c0a:
            dc = int(c0a - c1b - 1)
        else:
            dc = 0
        return int(dr + dc)

    used_obj_ids: set[int] = set()
    for marker in sorted(marker_comps, key=lambda rec: rec["bbox"][0]):
        axis_c = int(marker["axis_c"])
        candidates = [
            comp
            for comp in obj_comps
            if comp["id"] not in used_obj_ids and int(comp["bbox"][1]) <= axis_c <= int(comp["bbox"][3])
        ]
        if not candidates:
            continue

        comp = min(
            candidates,
            key=lambda rec: (
                bbox_gap(marker["bbox"], rec["bbox"]),
                abs(axis_c - ((int(rec["bbox"][1]) + int(rec["bbox"][3])) // 2)),
            ),
        )
        used_obj_ids.add(int(comp["id"]))

        k = 2 * axis_c
        color = int(comp["color"])
        for r, c in comp["coords"]:
            cc = int(k - c)
            if not (0 <= cc < W and int(g[int(r), cc]) == color):
                out[int(r), int(c)] = int(bg)

    return out


# --------------- generator (explicit marker-axis variant) ---------------

# ── CFG: canvas & layout ──
GRID_SIZE = 22
MARK = 2
NUM_SPRITES = 3
LAYOUTS = [
    [(0, 0, 9, 8), (0, 13, 9, 21), (12, 4, 21, 17)],
    [(0, 1, 9, 9), (0, 12, 9, 20), (12, 3, 21, 16)],
]

# ── CFG: clean sprite geometry ──
PIXEL_COUNT_MIN = 6
PIXEL_COUNT_MAX = 46
HEIGHT_MIN = 2
HEIGHT_MAX = 6
ODD_SIDE_W_RANGE = range(1, 6)  # widths 3..11
DENSITY_LO = 0.35
DENSITY_HI = 0.89

# ── CFG: noise ──
NOISE_FRAC_LO = 0.11
NOISE_FRAC_HI = 0.33
NOISE_MAX = 12

# ── CFG: marker placement ──
MARKER_GAP = 1
MARKER_LENGTHS = (1, 2, 3)

# ── CFG: retry limits ──
SPRITE_RETRIES = 200
GEN_RETRIES = 400

# ── CFG: augmentation ──
FLIP_PROB = 0.5


def _choose_shape_spec(max_h: int, max_w: int) -> tuple[int, int, int, int]:
    """Pick (h, w, target_count, center_count) for an odd-width symmetric sprite."""
    options: list[tuple[int, int, int, int]] = []
    for h in range(HEIGHT_MIN, min(HEIGHT_MAX, max_h) + 1):
        for side_w in ODD_SIDE_W_RANGE:
            w = 2 * side_w + 1
            if w > max_w:
                continue
            area = h * w
            lo = max(PIXEL_COUNT_MIN, math.ceil(DENSITY_LO * area))
            hi = min(PIXEL_COUNT_MAX, math.floor(DENSITY_HI * area))
            if lo > hi:
                continue
            for target_count in range(lo, hi + 1):
                for center_count in range(1, h + 1):
                    if (target_count - center_count) % 2 != 0:
                        continue
                    left_count = (target_count - center_count) // 2
                    if h <= left_count <= h * side_w:
                        options.append((h, w, target_count, center_count))
    if not options:
        raise ValueError("no feasible clean sprite spec")
    weights = [1.0 / (h * w) for h, w, _t, _c in options]
    return random.choices(options, weights=weights, k=1)[0]


def _left_cells(h: int, side_w: int, left_count: int) -> set[tuple[int, int]]:
    """Choose left-half cells, touching every row to keep the shape from becoming too sparse."""
    cells: set[tuple[int, int]] = set()
    inner_anchor_row = random.randrange(h)
    for r in range(h):
        if r == 0:
            c = 0
        elif side_w > 1 and r == inner_anchor_row:
            c = side_w - 1
        else:
            c = random.randrange(side_w)
        cells.add((r, c))

    remaining = left_count - len(cells)
    if remaining < 0:
        raise ValueError("left_count too small")
    pool = [(r, c) for r in range(h) for c in range(side_w) if (r, c) not in cells]
    cells.update(random.sample(pool, remaining))
    return cells


def _center_rows(h: int, center_count: int) -> np.ndarray:
    rows = np.zeros(h, dtype=bool)
    rows[random.sample(range(h), center_count)] = True
    return rows


def _crop_to_bbox(sprite: np.ndarray, color: int) -> np.ndarray:
    coords = np.argwhere(sprite == int(color))
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return sprite[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _build_clean_sprite(color: int, max_h: int, max_w: int) -> np.ndarray:
    """Build one connected odd-width symmetric sprite."""
    for _ in range(SPRITE_RETRIES):
        h, w, target_count, center_count = _choose_shape_spec(max_h, max_w)
        side_w = w // 2
        left_count = (target_count - center_count) // 2
        left = _left_cells(h, side_w, left_count)
        centers = _center_rows(h, center_count)

        sprite = np.zeros((h, w), dtype=int)
        for r, c in left:
            mirror_c = w - 1 - c
            sprite[r, c] = int(color)
            sprite[r, mirror_c] = int(color)
        sprite[centers, side_w] = int(color)
        sprite = _crop_to_bbox(sprite, color)

        if sprite.shape[0] > max_h or sprite.shape[1] > max_w or sprite.shape[1] % 2 == 0:
            continue
        count = int(np.sum(sprite == int(color)))
        area = int(sprite.shape[0] * sprite.shape[1])
        if not (PIXEL_COUNT_MIN <= count <= PIXEL_COUNT_MAX):
            continue
        density = count / area
        if not (DENSITY_LO <= density <= DENSITY_HI):
            continue
        _lab, n = label_mask(sprite == int(color), connectivity=4)
        if int(n) != 1:
            continue
        return sprite
    raise ValueError("failed to build clean sprite")


def _choose_marker_styles() -> tuple[list[int], list[str]]:
    """Force visible repetition: at least one single-pixel marker and one line marker."""
    lengths = [1, random.choice([2, 3]), random.choice(MARKER_LENGTHS)]
    random.shuffle(lengths)
    sides = ["above", "below", random.choice(["above", "below"])]
    random.shuffle(sides)
    return lengths, sides


def _slot_place_sprites(
    bg: int, sprite_specs: list[dict], slots: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, list[dict], np.ndarray] | None:
    """Place clean sprites and record external marker cells plus blocked cells for noise."""
    out = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    placed: list[dict] = []

    for spec, (r0s, c0s, r1s, c1s) in zip(sprite_specs, slots):
        spec = dict(spec)
        sprite = spec["sprite"]
        h, w = sprite.shape
        marker_len = int(spec["marker_len"])
        side = str(spec["marker_side"])
        slot_h = int(r1s - r0s + 1)
        slot_w = int(c1s - c0s + 1)
        bundle_h = int(h + MARKER_GAP + marker_len)
        if bundle_h > slot_h or w > slot_w:
            return None

        top = random.randint(int(r0s), int(r1s - bundle_h + 1))
        c0 = random.randint(int(c0s), int(c1s - w + 1))
        if side == "above":
            marker_r0 = int(top)
            sprite_r0 = int(top + marker_len + MARKER_GAP)
            gap_cells = [(int(sprite_r0 - 1), int(c0 + w // 2))]
        else:
            sprite_r0 = int(top)
            marker_r0 = int(sprite_r0 + h + MARKER_GAP)
            gap_cells = [(int(sprite_r0 + h), int(c0 + w // 2))]

        axis_c = int(c0 + w // 2)
        marker_cells = [(int(marker_r0 + i), int(axis_c)) for i in range(marker_len)]

        paste(out, sprite, int(sprite_r0), int(c0), transparent=0)
        for r, c in marker_cells + gap_cells:
            blocked[int(r), int(c)] = True

        spec["r0"] = int(sprite_r0)
        spec["c0"] = int(c0)
        spec["axis_c"] = int(axis_c)
        spec["true_k"] = int(2 * axis_c)
        spec["marker_cells"] = marker_cells
        spec["gap_cells"] = gap_cells
        placed.append(spec)

    return out, placed, blocked


def _noise_candidates(clean_mask: np.ndarray, true_k: int, blocked: np.ndarray) -> list[tuple[int, int]]:
    """Find 4-adjacent non-axis cells whose mirrors are not part of the clean shape."""
    H, W = clean_mask.shape
    adj = np.zeros_like(clean_mask)
    adj[1:, :] |= clean_mask[:-1, :]
    adj[:-1, :] |= clean_mask[1:, :]
    adj[:, 1:] |= clean_mask[:, :-1]
    adj[:, :-1] |= clean_mask[:, 1:]
    adj &= ~clean_mask
    adj &= ~blocked

    cols = np.arange(W)
    mirror_cols = int(true_k) - cols
    on_axis = mirror_cols == cols
    adj[:, on_axis] = False

    valid_mirror = (mirror_cols >= 0) & (mirror_cols < W) & ~on_axis
    for c in np.where(valid_mirror)[0]:
        adj[:, int(c)] &= ~clean_mask[:, int(mirror_cols[int(c)])]

    rows, cols_arr = np.where(adj)
    return list(zip(rows.tolist(), cols_arr.tolist()))


def _add_noise(inp: np.ndarray, out: np.ndarray, spec: dict, blocked: np.ndarray) -> bool:
    """Add non-mirrored adjacent noise to one placed sprite."""
    color = int(spec["color"])
    true_k = int(spec["true_k"])
    clean_mask = out == int(color)
    clean_coords = [tuple(x) for x in np.argwhere(clean_mask)]
    clean_count = len(clean_coords)

    lo = max(1, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count))
    if lo > hi:
        return False

    noise_target = random.randint(lo, hi)
    candidates = _noise_candidates(clean_mask, true_k, blocked)
    if len(candidates) < noise_target:
        return False

    chosen: list[tuple[int, int]] = []
    forbidden: set[tuple[int, int]] = set()
    for _ in range(noise_target):
        viable = [cell for cell in candidates if cell not in chosen and cell not in forbidden]
        if not viable:
            return False
        cell = random.choice(viable)
        chosen.append(cell)
        r, c = cell
        forbidden.add((int(r), int(c)))
        mirror = (int(r), int(true_k - c))
        if 0 <= mirror[1] < GRID_SIZE:
            forbidden.add(mirror)

    for r, c in chosen:
        inp[int(r), int(c)] = int(color)
    return True


def _gen_attempt() -> np.ndarray | None:
    """Output-first generation with explicit markers and oracle validation."""
    palette = [c for c in range(10) if c != MARK]
    bg, *obj_colors = random.sample(palette, NUM_SPRITES + 1)
    slots = random.choice(LAYOUTS)
    marker_lengths, marker_sides = _choose_marker_styles()

    sprite_specs: list[dict] = []
    try:
        for color, slot, marker_len, marker_side in zip(obj_colors, slots, marker_lengths, marker_sides):
            max_h = int(slot[2] - slot[0] + 1 - MARKER_GAP - marker_len)
            max_w = int(slot[3] - slot[1] + 1)
            sprite = _build_clean_sprite(int(color), max_h, max_w)
            sprite_specs.append(
                {
                    "color": int(color),
                    "sprite": sprite,
                    "marker_len": int(marker_len),
                    "marker_side": str(marker_side),
                }
            )
    except ValueError:
        return None

    result = _slot_place_sprites(int(bg), sprite_specs, slots)
    if result is None:
        return None
    out, placed_specs, blocked = result

    inp = out.copy()
    for spec in placed_specs:
        if not _add_noise(inp, out, spec, blocked):
            return None
        for r, c in spec["marker_cells"]:
            inp[int(r), int(c)] = int(MARK)

    if random.random() < FLIP_PROB:
        inp = np.flipud(inp)
        out = np.flipud(out)
    if random.random() < FLIP_PROB:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
