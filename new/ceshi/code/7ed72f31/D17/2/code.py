# Variant of 7ed72f31
# Dimension: D17 — 背景与干扰元素 (meta)
# Change: 背景为单色，覆盖率>=88%，无干扰元素... → 背景使用两色棋盘格纹理

from __future__ import annotations

import random
import sys

import numpy as np
from scipy import ndimage

_SOLVER_DIR = "/home/lixinyu/snap/new/solvers"
if _SOLVER_DIR not in sys.path:
    sys.path.append(_SOLVER_DIR)

from arc_utils import STRUCT8, label_mask

"""
Design analysis for the D17 checkerboard-background variant.

source_task:
  referenced_task: none

a_concretization_and_trace:
  original_scene:
    - One solid background color fills most of the canvas.
    - Red markers and colored payload sprites sit on top of that solid field.
    - Mirrored pixels are drawn onto empty background cells.
  variant_scene:
    - The empty field is now a perfect 2-color checkerboard over the entire grid.
    - Foreground still consists only of red markers plus monochrome payload sprites.
    - Mirror results still land only on checkerboard cells, so the texture is partially
      overwritten by new reflected pixels.
  concrete_trace:
    - Original solver: get_bg(grid) -> label all non-bg cells -> reflect them.
    - Variant solver: infer the two checkerboard colors and parity phase -> treat exactly
      those parity-matching cells as background -> label every other cell as foreground ->
      run the same reflection logic.
    - Difference: one background color is no longer safe to treat as foreground, so every
      place that used "v == bg" must instead use a background mask.

b_complexity_calibration:
  - The checkerboard uses exactly two colors, not an arbitrary texture family.
  - Sprite colors are forbidden from matching either checkerboard color.
  - Background still covers at least 88% of the grid, so the checkerboard is visually
    dominant and the inference problem is stable instead of noisy.
  - Marker types, payload sizes, and reflection rules stay unchanged so D17 remains the
    only conceptual novelty.

c_implicit_assumption_review_of_original_main:
  - "bg = get_bg(g)" assumed one global background color. Broken by two-color texture.
  - "label_mask(g != bg)" would absorb the second background color into foreground,
    merging unrelated objects and corrupting component detection.
  - "if v == bg or v == MARK" in the reflection scan assumed every empty cell has one
    value. In the variant, emptiness is a set of parity-constrained cells.
  - Neighbor filtering around isolated red markers used the same single-color background
    assumption and would wrongly treat one checkerboard color as payload.
  - The original unconditional write was only harmless because the generator guaranteed
    reflected pixels land on bg. The variant keeps that guarantee and also checks the
    inferred background mask before writing.

d_placement_legality:
  - Generator now keeps a separate boolean occupancy mask; checkerboard colors do not
    participate in overlap detection.
  - Placement still uses padding via 8-neighborhood dilation, so different sprites do not
    merge into one foreground component.
  - Output is built first on a full checkerboard canvas; input is recovered by restoring
    mirror cells back to their checkerboard colors.

e_visual_observability:
  - Across 2-4 examples, a human can see that the "empty" area is not solid anymore but
    an alternating texture.
  - The reflected pixels visibly replace checkerboard cells, making the extra foreground/
    background separation step necessary and observable.

f_visual_cleanliness:
  - Checkerboard colors are global and regular, so they read as one background layer.
  - Sprites remain separated by padding and never overlap each other or their mirrors.

g_color_inference:
  - No sprite color equals either checkerboard color or red.
  - This prevents ambiguous cases where a foreground payload pixel could masquerade as
    background texture.

h_new_solver_assumptions_and_generator_support:
  - Assumption: the background is a perfect 2-color checkerboard except where foreground
    overwrites it. Generator enforces this exactly.
  - Assumption: payload colors never equal checkerboard colors. Generator enforces this.
  - Assumption: reflected targets are background cells in the input. Generator enforces
    this during output-first construction, and solver writes only onto inferred bg cells.
"""

MARK = 2

# Canvas
GRID_MIN, GRID_MAX = 18, 28

# Marker counts
TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX = 3, 7
MAX_EMBEDDED, MAX_ISOLATED = 4, 4

# Embedded payload shape
PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX = 2, 6
PAYLOAD_PX_MIN, PAYLOAD_PX_MAX = 3, 11
PAYLOAD_DENSITY_LO, PAYLOAD_DENSITY_HI = 0.50, 0.80

# Isolated payload
ISOLATED_PX_MIN, ISOLATED_PX_MAX = 1, 8
ISOLATED_BBOX_MAX = 4

# Color palette
PALETTE_COLORS_MIN, PALETTE_COLORS_MAX = 3, 4

# Layout
PLACEMENT_PADDING = 1
BG_COVERAGE_MIN = 0.88

# Retries
MASK_RETRIES = 40
GEN_RETRIES = 300

_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _checkerboard_pattern(shape: tuple[int, int], c0: int, c1: int, phase: int) -> np.ndarray:
    rows, cols = np.indices(shape)
    parity = (rows + cols) & 1
    return np.where(parity == int(phase), int(c0), int(c1)).astype(int)


def _infer_checkerboard_background(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    g = np.array(grid, dtype=int)
    colors = [int(c) for c in np.unique(g)]
    best_score = -1
    best_pattern = None
    best_cfg = None
    rows, cols = np.indices(g.shape)
    parity = (rows + cols) & 1

    for c0 in colors:
        for c1 in colors:
            if int(c0) == int(c1):
                continue
            for phase in (0, 1):
                expected = np.where(parity == int(phase), int(c0), int(c1))
                score = int(np.sum(g == expected))
                if score > best_score:
                    best_score = score
                    best_pattern = expected.astype(int)
                    best_cfg = (int(c0), int(c1), int(phase))

    if best_pattern is None or best_cfg is None:
        raise ValueError("failed to infer checkerboard background")
    return best_pattern, g == best_pattern, best_cfg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror colored sprites using red markers as reflection axes, but first recover
    the two-color checkerboard background and treat only non-checkerboard cells as
    foreground.
    """
    g = np.array(grid, dtype=int)
    _bg_pattern, bg_mask, _bg_cfg = _infer_checkerboard_background(g)
    H, W = g.shape

    lab, n = label_mask(~bg_mask, connectivity=4)
    comps = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        rs = [r for r, _c in coords]
        cs = [c for _r, c in coords]
        comps.append(
            {
                "coords": coords,
                "vals": vals,
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )

    out = g.copy()

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    def refl_point(p: tuple[int, int], r: int, c: int) -> tuple[int, int]:
        pr, pc = map(int, p)
        return int(2 * pr - r), int(2 * pc - c)

    def refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
        return int(2 * int(r0) - int(r)), int(c)

    def refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
        return int(r), int(2 * int(c0) - int(c))

    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if int(MARK) not in set(vals):
            continue

        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        if not reds:
            continue

        rs = {int(r) for r, _c in reds}
        cs = {int(c) for _r, c in reds}
        if len(reds) == 1:
            axis = ("point", (int(reds[0][0]), int(reds[0][1])))
        elif len(rs) == 1:
            axis = ("row", int(next(iter(rs))))
        elif len(cs) == 1:
            axis = ("col", int(next(iter(cs))))
        else:
            continue

        r0, c0, r1, c1 = map(int, comp["bbox"])
        payload = {int(v) for v in vals if int(v) != int(MARK)}
        if not payload:
            continue

        if axis[0] == "row":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(c0) <= int(c) <= int(c1)

        elif axis[0] == "col":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(r0) <= int(r) <= int(r1)

        else:
            rr0 = max(0, int(r0 - 1))
            rr1 = min(int(H - 1), int(r1 + 1))
            cc0 = max(0, int(c0 - 1))
            cc1 = min(int(W - 1), int(c1 + 1))

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(rr0) <= int(r) <= int(rr1) and int(cc0) <= int(c) <= int(cc1)

        for r in range(int(H)):
            for c in range(int(W)):
                v = int(g[int(r), int(c)])
                if bool(bg_mask[int(r), int(c)]) or v == int(MARK):
                    continue
                if not eligible(int(r), int(c), int(v)):
                    continue

                if axis[0] == "row":
                    rr, cc = refl_row(int(axis[1]), int(r), int(c))
                elif axis[0] == "col":
                    rr, cc = refl_col(int(axis[1]), int(r), int(c))
                else:
                    rr, cc = refl_point(axis[1], int(r), int(c))

                if inb(int(rr), int(cc)) and bool(bg_mask[int(rr), int(cc)]):
                    out[int(rr), int(cc)] = int(v)

    marker_pts = []
    for comp in comps:
        if len(comp["coords"]) == 1 and int(comp["vals"][0]) == int(MARK):
            marker_pts.append(comp["coords"][0])

    for pr, pc in marker_pts:
        pr = int(pr)
        pc = int(pc)
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not inb(rr, cc):
                    continue
                v = int(g[rr, cc])
                if bool(bg_mask[rr, cc]) or v == int(MARK):
                    continue
                neigh_colors.add(int(v))
                neigh_pixels.append((int(rr), int(cc), int(v)))
        if not neigh_colors:
            continue

        for v in neigh_colors:
            labv, _nv = label_mask(g == int(v), connectivity=8)
            labs = {
                int(labv[int(rr), int(cc)])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(v) and int(labv[int(rr), int(cc)]) != 0
            }
            for lbl in labs:
                for r, c in np.argwhere(labv == int(lbl)):
                    rr, cc = refl_point((int(pr), int(pc)), int(r), int(c))
                    if inb(rr, cc) and bool(bg_mask[int(rr), int(cc)]):
                        out[int(rr), int(cc)] = int(v)

    return out


def _random_connected_mask(h: int, w: int, target_pixels: int, *, rng: random.Random) -> np.ndarray | None:
    if target_pixels < 1 or target_pixels > h * w:
        return None
    mask = np.zeros((h, w), dtype=bool)
    mask[rng.randrange(h), rng.randrange(w)] = True
    count = 1
    while count < target_pixels:
        frontier = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                for dr, dc in _DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                        frontier.append((r, c))
                        break
        if not frontier:
            return None
        r, c = rng.choice(frontier)
        mask[r, c] = True
        count += 1
    if not mask.any(axis=1).all() or not mask.any(axis=0).all():
        return None
    return mask


def _build_embedded_sprite(rng: random.Random) -> dict | None:
    ph = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
    pw = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
    max_px = min(PAYLOAD_PX_MAX, int(PAYLOAD_DENSITY_HI * ph * pw))
    min_px = max(PAYLOAD_PX_MIN, int(np.ceil(PAYLOAD_DENSITY_LO * ph * pw)))
    if min_px > max_px:
        return None
    pmask = _random_connected_mask(ph, pw, rng.randint(min_px, max_px), rng=rng)
    if pmask is None:
        return None

    is_row = rng.choice([True, False])
    side_flip = rng.random() < 0.5

    if is_row:
        height, width = 2 * ph + 1, pw
        payload = np.zeros((height, width), dtype=bool)
        mirror = np.zeros((height, width), dtype=bool)
        red_pixels = [(ph, c) for c in range(pw)]
        orig_slice, mirr_slice = (np.s_[:ph], np.s_[ph + 1 :]) if not side_flip else (np.s_[ph + 1 :], np.s_[:ph])
        payload[orig_slice] = pmask
        mirror[mirr_slice] = pmask[::-1]
    else:
        height, width = pw, 2 * ph + 1
        payload = np.zeros((height, width), dtype=bool)
        mirror = np.zeros((height, width), dtype=bool)
        red_pixels = [(r, ph) for r in range(pw)]
        orig_slice, mirr_slice = (np.s_[:, :ph], np.s_[:, ph + 1 :]) if not side_flip else (np.s_[:, ph + 1 :], np.s_[:, :ph])
        payload[orig_slice] = pmask.T
        mirror[mirr_slice] = pmask.T[:, ::-1]

    return {"payload": payload, "mirror": mirror, "red_pixels": red_pixels}


def _is_4connected_to(mask: np.ndarray, r: int, c: int) -> bool:
    height, width = mask.shape
    for dr, dc in _DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width and mask[nr, nc]:
            return True
    return False


def _build_isolated_sprite(rng: random.Random) -> dict | None:
    n_px = rng.randint(ISOLATED_PX_MIN, ISOLATED_PX_MAX)
    pmask = None
    for _ in range(MASK_RETRIES):
        ph = rng.randint(1, min(ISOLATED_BBOX_MAX, max(1, n_px)))
        pw = rng.randint(1, min(ISOLATED_BBOX_MAX, max(1, n_px)))
        if ph * pw < n_px or n_px < max(ph, pw):
            continue
        pmask = np.ones((ph, pw), dtype=bool) if n_px == ph * pw else _random_connected_mask(ph, pw, n_px, rng=rng)
        if pmask is not None:
            break
    if pmask is None:
        return None

    pcoords = list(zip(*np.where(pmask)))
    rng.shuffle(pcoords)

    for pr, pc in pcoords:
        dirs = list(_DIRS8)
        rng.shuffle(dirs)
        for dr, dc in dirs:
            red_r, red_c = pr + dr, pc + dc
            refl = [(2 * red_r - r, 2 * red_c - c) for r, c in pcoords]
            all_pts = pcoords + refl + [(red_r, red_c)]
            all_rs = [r for r, _ in all_pts]
            all_cs = [c for _, c in all_pts]
            minr, minc = min(all_rs), min(all_cs)
            height = max(all_rs) - minr + 1
            width = max(all_cs) - minc + 1

            payload = np.zeros((height, width), dtype=bool)
            mirror = np.zeros((height, width), dtype=bool)
            for r, c in pcoords:
                payload[r - minr, c - minc] = True
            for r, c in refl:
                mirror[r - minr, c - minc] = True
            red_pos = (red_r - minr, red_c - minc)

            if payload[red_pos] or mirror[red_pos]:
                continue
            if (payload & mirror).any():
                continue
            if _is_4connected_to(payload | mirror, *red_pos):
                continue
            return {"payload": payload, "mirror": mirror, "red_pixels": [red_pos]}
    return None


def _random_free_location(
    occupied: np.ndarray, patch_mask: np.ndarray, rng: random.Random, *, padding: int = PLACEMENT_PADDING
):
    height, width = occupied.shape
    patch_h, patch_w = patch_mask.shape
    if patch_h > height or patch_w > width:
        return None
    occ = occupied
    if padding > 0 and occ.any():
        occ = ndimage.binary_dilation(occ, structure=STRUCT8, iterations=padding)
    cands = []
    for r0 in range(height - patch_h + 1):
        for c0 in range(width - patch_w + 1):
            if not (occ[r0 : r0 + patch_h, c0 : c0 + patch_w] & patch_mask).any():
                cands.append((r0, c0))
    if not cands:
        return None
    return rng.choice(cands)


def _stamp_sprite(canvas: np.ndarray, spr: dict, r0: int, c0: int) -> None:
    height, width = spr["payload"].shape
    colored = spr["payload"] | spr["mirror"]
    canvas[r0 : r0 + height, c0 : c0 + width][colored] = spr["color"]
    for rr, cc in spr["red_pixels"]:
        canvas[r0 + rr, c0 + cc] = MARK


def _erase_mirror(canvas: np.ndarray, bg_pattern: np.ndarray, spr: dict) -> None:
    r0, c0 = spr["r0"], spr["c0"]
    height, width = spr["mirror"].shape
    patch = canvas[r0 : r0 + height, c0 : c0 + width]
    bg_patch = bg_pattern[r0 : r0 + height, c0 : c0 + width]
    patch[spr["mirror"]] = bg_patch[spr["mirror"]]


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    n = rng.randint(GRID_MIN, GRID_MAX)
    bg0, bg1 = rng.sample([c for c in range(10) if c != MARK], 2)
    palette = [c for c in range(10) if c not in {MARK, bg0, bg1}]

    total_markers = rng.randint(TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX)
    n_embedded = rng.randint(max(1, total_markers - MAX_ISOLATED), min(MAX_EMBEDDED, total_markers))
    n_isolated = total_markers - n_embedded

    n_colors = rng.randint(PALETTE_COLORS_MIN, min(PALETTE_COLORS_MAX, len(palette), total_markers))
    chosen_colors = rng.sample(palette, n_colors)

    sprites: list[dict] = []
    for builder, count in [(_build_embedded_sprite, n_embedded), (_build_isolated_sprite, n_isolated)]:
        for _ in range(count):
            spr = builder(rng)
            if spr is None:
                return None
            sprites.append(spr)

    assignment = list(chosen_colors) + [rng.choice(chosen_colors) for _ in range(len(sprites) - n_colors)]
    rng.shuffle(assignment)
    for spr, color in zip(sprites, assignment):
        spr["color"] = int(color)

    phase = rng.randint(0, 1)
    background = _checkerboard_pattern((n, n), bg0, bg1, phase)
    output = background.copy()
    occupied = np.zeros((n, n), dtype=bool)

    rng.shuffle(sprites)
    for spr in sprites:
        footprint = (spr["payload"] | spr["mirror"]).copy()
        for r, c in spr["red_pixels"]:
            footprint[r, c] = True
        loc = _random_free_location(occupied, footprint, rng)
        if loc is None:
            return None
        r0, c0 = loc
        spr["r0"], spr["c0"] = r0, c0
        _stamp_sprite(output, spr, r0, c0)
        occupied[r0 : r0 + footprint.shape[0], c0 : c0 + footprint.shape[1]] |= footprint

    inp = output.copy()
    for spr in sprites:
        _erase_mirror(inp, background, spr)

    if np.mean(inp == background) < BG_COVERAGE_MIN:
        return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
