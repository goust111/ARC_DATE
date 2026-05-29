# Combo variant of a32d8b75 (index 19)
# combo_spec: D6/0+D8/0+D18/1
# Name: 二进制编码+随机图例+标记嵌画布

import random
from collections import Counter

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, random_sprite


# Design analysis for the fused combo
# -----------------------------------
# a) Concretization + behavior trace:
#    - Original task:
#      legend = [unit sprite][frame][single-pixel 3x3 anchor][rotation glyph], all in fixed
#      vertical order. The solver could read "top component is unit", "bottom 8x5 block is
#      marker panel", then map the single anchor pixel to {start, middle, end}.
#    - Fused task:
#      side legends now keep only shape data (`unit` + `frame`), and those two components can
#      appear in either vertical order with variable gaps. Placement metadata leaves the legend
#      entirely and becomes a hidden 8x5 control panel inside the canvas background. The top
#      5x5 square still has a 6-border, but its 3x3 interior is no longer one hot pixel:
#        center = fixed color-4 sentinel,
#        left column bits = row offset bits 1/2/4,
#        right column bits = col offset bits 1/2/4.
#      The bottom 3x5 strip still carries the 2-pixel clock hand for rotation.
#    - Small trace:
#      suppose a left legend shows `frame` above `unit`, while the canvas corner contains
#      row bits on at rows 0 and 2 and col bit on at row 1. The old solver would fail because
#      it expects a marker under the legend frame and expects a unique anchor pixel. The fused
#      solver instead:
#        1. identifies `unit` by "filled 2-color square",
#        2. identifies `frame` by "single-color sparse 8-connected mask",
#        3. finds the hidden canvas panel,
#        4. decodes row_code = 1 + 4 = 5, col_code = 2,
#        5. rotates the tiled composite and places it at exact top-left (5, 2).
#
# b) Complexity calibration:
#    - The binary code uses only 3 bits per axis, so there are at most 8 row choices and 8
#      col choices. That is visibly richer than the original 3-way anchor, but still small
#      enough that repeated evidence across 2-legend samples can support human inference.
#    - The legend shuffle is constrained to only two visually distinct components after D18
#      moves markers away. This keeps the input readable rather than cluttering the canvas with
#      too many detached symbols.
#
# c) Hidden-assumption audit:
#    - Original "top-most component is the unit sprite" is invalid once legend order is random.
#    - Original "frame sits between unit and bottom marker square" is invalid because markers
#      are gone from legends entirely.
#    - Original "anchor is one interior pixel in the 3x3 square" is invalid because multiple
#      pixels now encode independent row/col bits.
#    - Original "start from raw canvas" is invalid because the input canvas now contains hidden
#      control panels that must be erased back to the banded background before drawing output.
#    - New solver assumptions are limited to:
#      each legend has exactly one filled 2-color square and one sparse 1-color frame component;
#      canvas background avoids marker colors {4,6,7}; hidden marker panels do not overlap the
#      final composite footprints; with two legends there is exactly one marker in each canvas
#      half so pairing by left/right legend remains unambiguous.
#
# d) Placement legality:
#    - The generator stays output-first. It chooses frame size and binary offsets only if the
#      rotated composite fits fully inside the canvas.
#    - For two legends, bounding boxes of the final composites must not overlap.
#    - Canvas marker panels are assigned only to corner slots that do not intersect any
#      composite bounding box, so the restored background and the final drawing remain clean.
#
# e) Visual observability:
#    - Humans can spot three changes directly from a few samples:
#      legends are simpler and can flip unit/frame order; control panels appear inside the
#      canvas corners; position panels contain several yellow pixels instead of one.
#    - Background colors deliberately exclude marker colors, so the control panels stand out and
#      the erased stripe pattern is inferable from surrounding rows or columns.
#
# f) Visual cleanliness:
#    - The hidden panels are confined to 4 corner slots with a 1-cell inset. They do not mix
#      with legend contents and do not overlap sprite footprints.
#    - Legend components keep at least one blank row between them, which makes the shuffled
#      ordering visible without causing component merges under 8-connectivity.
#
# g) Color inferability:
#    - No invisible ordering by numeric color value is used. The binary rule is spatial:
#      left bit-stack controls vertical offset, right bit-stack controls horizontal offset.
#      Color 4 acts only as a marker ink, and color 7 only marks rotation direction.


GRID_H_RANGE = (20, 30)
GRID_W = 30
LEGEND_COUNT_CHOICES = [1, 2]
UNIT_SIZE_RANGE = (2, 5)
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)
FRAME_W_RANGE = (3, 6)
FRAME_DENSITY = 0.5
BAND_WIDTH_MIN = 4
USABLE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
BACKGROUND_COLORS = [1, 2, 3, 5, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 1200
MIN_COMPONENT_GAP = 1
MARKER_PANEL_HEIGHT = 8
MARKER_PANEL_WIDTH = 5
MARKER_INSET = 1
MARKER_BG_COLORS = {4, 6, 7}


def _separator_runs(g: np.ndarray) -> list[tuple[int, int]]:
    cols6 = [c for c in range(g.shape[1]) if np.all(g[:, c] == 6)]
    if not cols6:
        return []
    runs: list[tuple[int, int]] = []
    s = e = cols6[0]
    for c in cols6[1:]:
        if c == e + 1:
            e = c
        else:
            runs.append((s, e))
            s = e = c
    runs.append((s, e))
    return runs


def _split_canvas(g: np.ndarray) -> tuple[tuple[int, int], tuple[int, int] | None]:
    runs = _separator_runs(g)
    assert runs, "must have at least one color-6 separator run"
    left = runs[0]
    right: tuple[int, int] | None = None
    for s, e in reversed(runs[1:]):
        if s >= g.shape[1] - 10 and s > left[1] + 1:
            right = (s, e)
            break
    return left, right


def _legend_panels(
    g: np.ndarray, left_sep: tuple[int, int], right_sep: tuple[int, int] | None
) -> list[np.ndarray]:
    out = [g[:, : left_sep[0]]]
    if right_sep is not None:
        out.append(g[:, right_sep[1] + 1 :])
    return out


def _swap_two_colors(sprite: np.ndarray) -> np.ndarray:
    cols = sorted({int(x) for x in np.unique(sprite) if int(x) not in (0, 6)})
    assert len(cols) == 2, "unit sprite must use exactly two colors"
    a, b = cols
    out = sprite.copy()
    out[sprite == a] = -1
    out[sprite == b] = a
    out[out == -1] = b
    return out


def _component_records(panel: np.ndarray) -> list[dict]:
    mask = (panel != 0) & (panel != 6)
    labels, n = _scipy_label(mask, structure=np.ones((3, 3), dtype=int))
    comps: list[dict] = []
    for idx in range(1, n + 1):
        pts = np.argwhere(labels == idx)
        r0, c0, r1, c1 = bbox_coords(pts)
        local_mask = labels[r0 : r1 + 1, c0 : c1 + 1] == idx
        colors = sorted({int(x) for x in panel[r0 : r1 + 1, c0 : c1 + 1][local_mask]})
        comps.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": local_mask.copy(),
                "colors": colors,
                "area": int(local_mask.sum()),
                "h": int(r1 - r0 + 1),
                "w": int(c1 - c0 + 1),
            }
        )
    return comps


def _find_unit_sprite(panel: np.ndarray) -> np.ndarray:
    candidates: list[dict] = []
    for comp in _component_records(panel):
        if comp["h"] != comp["w"]:
            continue
        if not (2 <= comp["h"] <= 5):
            continue
        if comp["area"] != comp["h"] * comp["w"]:
            continue
        if len(comp["colors"]) != 2:
            continue
        candidates.append(comp)
    assert len(candidates) == 1, f"expected exactly one unit sprite, got {len(candidates)}"
    r0, c0, r1, c1 = candidates[0]["bbox"]
    sprite = panel[r0 : r1 + 1, c0 : c1 + 1].copy()
    return _swap_two_colors(sprite)


def _find_frame_mask(panel: np.ndarray) -> np.ndarray:
    candidates: list[dict] = []
    for comp in _component_records(panel):
        h, w = comp["h"], comp["w"]
        if not (4 <= h <= 8 and 3 <= w <= 5):
            continue
        if len(comp["colors"]) != 1:
            continue
        if comp["area"] >= h * w:
            continue
        density = comp["area"] / float(h * w)
        if not (0.35 <= density <= 0.8):
            continue
        candidates.append(comp)
    assert len(candidates) == 1, f"expected exactly one frame, got {len(candidates)}"
    return candidates[0]["mask"].copy()


def _panel_square_ok(canvas: np.ndarray, r0: int, c0: int) -> bool:
    if r0 < 0 or c0 < 0 or r0 + MARKER_PANEL_HEIGHT > canvas.shape[0] or c0 + MARKER_PANEL_WIDTH > canvas.shape[1]:
        return False

    sq = canvas[r0 : r0 + 5, c0 : c0 + 5]
    if not (
        np.all(sq[0, :] == 6)
        and np.all(sq[4, :] == 6)
        and np.all(sq[:, 0] == 6)
        and np.all(sq[:, 4] == 6)
    ):
        return False

    inner = sq[1:4, 1:4]
    if int(inner[1, 1]) != 4:
        return False
    if int(inner[0, 1]) != 0 or int(inner[2, 1]) != 0:
        return False
    if np.any((inner[:, 0] != 0) & (inner[:, 0] != 4)):
        return False
    if np.any((inner[:, 2] != 0) & (inner[:, 2] != 4)):
        return False

    rot = canvas[r0 + 5 : r0 + 8, c0 : c0 + 5]
    if not (np.all(rot[:, 0] == 6) and np.all(rot[:, 4] == 6)):
        return False
    core = rot[:, 1:4]
    pts = np.argwhere(core == 7)
    if pts.shape[0] != 2:
        return False
    pts_set = {tuple(map(int, p)) for p in pts}
    if (1, 1) not in pts_set:
        return False
    return any((1 + dr, 1 + dc) in pts_set for dr, dc in ROTATION_DIRS)


def _all_canvas_marker_panels(canvas: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r0 in range(canvas.shape[0] - MARKER_PANEL_HEIGHT + 1):
        for c0 in range(canvas.shape[1] - MARKER_PANEL_WIDTH + 1):
            if _panel_square_ok(canvas, r0, c0):
                out.append((int(r0), int(c0)))
    out.sort()
    return out


def _decode_binary_offsets_from_border_square(canvas: np.ndarray, border6_r0: int, border6_c0: int) -> tuple[int, int]:
    inner = canvas[border6_r0 + 1 : border6_r0 + 4, border6_c0 + 1 : border6_c0 + 4]
    assert int(inner[1, 1]) == 4, "binary marker square must contain the center sentinel"
    row_code = 0
    col_code = 0
    for bit in range(3):
        if int(inner[bit, 0]) == 4:
            row_code |= 1 << bit
        if int(inner[bit, 2]) == 4:
            col_code |= 1 << bit
    return row_code, col_code


def _rotation_from_clockhand(canvas: np.ndarray, border6_r0: int, border6_c0: int) -> int:
    core = canvas[border6_r0 + 5 : border6_r0 + 8, border6_c0 + 1 : border6_c0 + 4]
    pts = np.argwhere(core == 7)
    assert pts.shape[0] == 2, "rotation marker must have center and one arm"
    center = np.array([1, 1], dtype=int)
    dists = np.abs(pts - center).sum(axis=1)
    p = pts[int(dists.argmax())]
    dr, dc = int(p[0] - center[0]), int(p[1] - center[1])
    if abs(dr) >= abs(dc):
        return 0 if dr < 0 else 2
    return 1 if dc > 0 else 3


def _build_composite(unit_sprite: np.ndarray, frame_mask: np.ndarray) -> np.ndarray:
    m_unit = (unit_sprite != 0) & (unit_sprite != 6)
    uh, uw = unit_sprite.shape
    fh, fw = frame_mask.shape
    comp = np.zeros((fh * uh, fw * uw), dtype=int)
    for i in range(fh):
        for j in range(fw):
            if not frame_mask[i, j]:
                continue
            rr0 = i * uh
            cc0 = j * uw
            comp[rr0 : rr0 + uh, cc0 : cc0 + uw][m_unit] = unit_sprite[m_unit]
    return comp


def _effective_canvas_shape(canvas: np.ndarray) -> tuple[int, int]:
    eff_h, eff_w = canvas.shape
    full6_cols = [c for c in range(eff_w) if np.all(canvas[:, c] == 6)]
    if full6_cols:
        eff_w = min(full6_cols)
    full6_rows = [r for r in range(eff_h) if np.all(canvas[r, :] == 6)]
    if full6_rows:
        eff_h = min(full6_rows)
    return eff_h, eff_w


def _offset_from_binary_code(code: int, size: int, place_size: int) -> int:
    max_offset = size - place_size
    assert max_offset >= 0, "composite larger than canvas"
    assert 0 <= code <= min(7, max_offset), "binary offset outside legal range"
    return code


def _majority_color_without_markers(values: np.ndarray) -> tuple[int, float]:
    kept = [int(v) for v in values if int(v) not in MARKER_BG_COLORS]
    assert kept, "line cannot be fully occupied by markers"
    ctr = Counter(kept)
    val, count = ctr.most_common(1)[0]
    return int(val), count / len(kept)


def _restore_banded_background(canvas: np.ndarray) -> np.ndarray:
    row_stats = [_majority_color_without_markers(canvas[r, :]) for r in range(canvas.shape[0])]
    col_stats = [_majority_color_without_markers(canvas[:, c]) for c in range(canvas.shape[1])]
    row_purity = sum(frac for _, frac in row_stats) / len(row_stats)
    col_purity = sum(frac for _, frac in col_stats) / len(col_stats)

    restored = np.zeros_like(canvas)
    if row_purity >= col_purity:
        for r, (val, _) in enumerate(row_stats):
            restored[r, :] = val
    else:
        for c, (val, _) in enumerate(col_stats):
            restored[:, c] = val
    return restored


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild enlarged sprites from shuffled legends and canvas-hidden binary control panels.

    Input structure:
    - Full-height color-6 separator columns still split the grid into a left legend, a main
      canvas, and optionally a right legend.
    - Each legend now contains only two shape components, a filled 2-color unit square and a
      sparse 1-color frame mask, and their vertical order is randomized.
    - Placement metadata is no longer in the legend: each hidden canvas panel is an 8x5 block
      made of a 5x5 border-6 square with binary row/col bits inside, plus a 3x5 rotation strip
      below it.

    Rule:
    - Identify the unit sprite and frame mask by geometry, swap the unit colors back, tile that
      unit into every active frame cell, and rotate the composite from the clock-hand strip.
    - Decode the hidden marker square by reading the left bit-stack as a 3-bit row offset and
      the right bit-stack as a 3-bit column offset, giving the exact composite top-left.
    - Restore the clean banded canvas underneath the hidden panels, then stamp each rotated
      composite at its decoded offset.

    Key insight:
    - The two sources of indirection now interact: legend order can no longer reveal roles, and
      the canvas marker is no longer a coarse anchor but a pair of binary counters. Solving the
      task requires shape-based legend parsing first, then searching the canvas for the hidden
      control panels and decoding their spatial bit pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape[1] >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else g.shape[1]
    canvas = g[:, canvas_c0:canvas_c1].copy()

    legends = _legend_panels(g, left_sep, right_sep)
    markers = _all_canvas_marker_panels(canvas)
    assert len(markers) == len(legends), "marker count must match legend count"

    if len(legends) == 2:
        left_markers = [rc for rc in markers if rc[1] < canvas.shape[1] // 2]
        right_markers = [rc for rc in markers if rc[1] >= canvas.shape[1] // 2]
        assert len(left_markers) == 1 and len(right_markers) == 1, "two legends need one left and one right marker"
        markers = [left_markers[0], right_markers[0]]

    out = _restore_banded_background(canvas)
    eff_h, eff_w = _effective_canvas_shape(out)
    composite_written = np.zeros(out.shape, dtype=bool)

    for panel, (marker_r0, marker_c0) in zip(legends, markers):
        unit_sprite = _find_unit_sprite(panel)
        frame_mask0 = _find_frame_mask(panel)

        dir_k = _rotation_from_clockhand(canvas, marker_r0, marker_c0)
        frame_mask = np.rot90(frame_mask0, (-dir_k) % 4)
        comp = _build_composite(unit_sprite, frame_mask)

        row_code, col_code = _decode_binary_offsets_from_border_square(canvas, marker_r0, marker_c0)
        top = _offset_from_binary_code(row_code, eff_h, comp.shape[0])
        left = _offset_from_binary_code(col_code, eff_w, comp.shape[1])

        ch, cw = comp.shape
        assert top + ch <= eff_h and left + cw <= eff_w, "binary placement exceeds effective canvas"
        for r in range(ch):
            rr = top + r
            for c in range(cw):
                cc = left + c
                val = int(comp[r, c])
                if val in (0, 6):
                    continue
                assert not composite_written[rr, cc], "composites must not overlap"
                composite_written[rr, cc] = True
                out[rr, cc] = val
    return out


def _allocate_colors(num_legends: int):
    pool = list(USABLE_COLORS)
    random.shuffle(pool)

    unit_color_sets = []
    used_unit = set()
    for _ in range(num_legends):
        if len(pool) < 2:
            return None
        c1, c2 = pool.pop(), pool.pop()
        unit_color_sets.append((c1, c2))
        used_unit.update([c1, c2])

    bg_cands = [c for c in BACKGROUND_COLORS if c not in used_unit]
    if len(bg_cands) < 2:
        return None
    random.shuffle(bg_cands)
    bg_colors = (bg_cands[0], bg_cands[1])

    frame_colors = []
    all_set = set(USABLE_COLORS)
    for uc in unit_color_sets:
        candidates = list(all_set - set(uc))
        frame_colors.append(random.choice(candidates))

    return unit_color_sets, bg_colors, frame_colors


def _color_swap(sprite: np.ndarray, c1: int, c2: int) -> np.ndarray:
    sa, sb = sorted([c1, c2])
    out = sprite.copy()
    out[sprite == sa] = sb
    out[sprite == sb] = sa
    return out


def _sample_frame_mask(frame_h: int, frame_w: int) -> np.ndarray:
    for _ in range(40):
        mask = random_sprite(
            frame_h,
            frame_w,
            density=FRAME_DENSITY,
            connectivity=8,
            color_palette=[1],
            symmetry="not_symmetric",
        ) != 0
        active = int(mask.sum())
        if active == 0 or active == frame_h * frame_w:
            continue
        density = active / float(frame_h * frame_w)
        if 0.35 <= density <= 0.8:
            return mask
    raise RuntimeError("failed to sample a suitable frame mask")


def _distribute_extra(total: int, slots: int) -> list[int]:
    parts = [0] * slots
    for _ in range(total):
        parts[random.randrange(slots)] += 1
    return parts


def _layout_legend_rows(leg: dict, grid_h: int) -> None:
    heights = {"unit": leg["N"], "frame": leg["frame_H"]}
    order = random.sample(["unit", "frame"], k=2)
    base_total = heights["unit"] + heights["frame"] + MIN_COMPONENT_GAP
    assert base_total <= grid_h, "legend components do not fit vertically"
    extras = _distribute_extra(grid_h - base_total, 3)

    rows = {}
    cursor = extras[0]
    rows[order[0]] = cursor
    cursor += heights[order[0]]
    cursor += MIN_COMPONENT_GAP + extras[1]
    rows[order[1]] = cursor

    leg.update(order=tuple(order), unit_r0=rows["unit"], frame_r0=rows["frame"])


def _design_legend(
    leg: dict,
    unit_colors: tuple[int, int],
    frame_color: int,
    grid_h: int,
    canvas_w: int,
    prefer_small: bool = False,
) -> bool:
    n = leg["N"]
    uc1, uc2 = unit_colors

    sprite = np.full((n, n), uc1, dtype=int)
    num_c2 = random.randint(1, n * n - 1)
    for p in random.sample(range(n * n), num_c2):
        sprite[p // n, p % n] = uc2
    legend_sprite = _color_swap(sprite, uc1, uc2)

    combos = []
    for frame_h in range(*FRAME_H_RANGE):
        for frame_w in range(*FRAME_W_RANGE):
            if frame_w > leg["legend_w"]:
                continue
            if frame_h + n + MIN_COMPONENT_GAP > grid_h:
                continue
            for rot_k in range(4):
                rot_h, rot_w = (frame_h, frame_w) if rot_k % 2 == 0 else (frame_w, frame_h)
                comp_h = rot_h * n
                comp_w = rot_w * n
                if comp_h <= grid_h and comp_w <= canvas_w:
                    combos.append((frame_h, frame_w, rot_k, comp_h * comp_w))
    if not combos:
        return False

    if prefer_small:
        combos.sort(key=lambda x: x[3])
        keep = max(1, len(combos) // 3)
        combos = combos[:keep]

    frame_h, frame_w, rot_k, _ = random.choice(combos)
    frame_mask = _sample_frame_mask(frame_h, frame_w)
    rotated_frame = np.rot90(frame_mask, (-rot_k) % 4)

    leg.update(
        sprite=sprite,
        legend_sprite=legend_sprite,
        frame_mask=frame_mask,
        frame_H=frame_h,
        frame_W=frame_w,
        frame_color=frame_color,
        rot_k=rot_k,
        rotated_frame=rotated_frame,
    )
    _layout_legend_rows(leg, grid_h)
    return True


def _comp_size(leg: dict) -> tuple[int, int]:
    rf = leg["rotated_frame"]
    n = leg["N"]
    return rf.shape[0] * n, rf.shape[1] * n


def _code_options(max_offset: int) -> list[int]:
    return list(range(min(7, max_offset) + 1))


def _code_complexity(code: int) -> int:
    return bin(code).count("1")


def _single_code_score(row_code: int, col_code: int) -> tuple[int, int]:
    multi = int(_code_complexity(row_code) >= 2) + int(_code_complexity(col_code) >= 2)
    nonzero = int(row_code > 0) + int(col_code > 0)
    return (multi, nonzero)


def _pair_code_score(pair: tuple[int, int, int, int]) -> tuple[int, int, int]:
    ar, ac, br, bc = pair
    same_one_axis = int((ar == br) ^ (ac == bc))
    multi = sum(int(_code_complexity(code) >= 2) for code in pair)
    spread = abs(ar - br) + abs(ac - bc)
    return (same_one_axis, multi, spread)


def _assign_binary_offsets(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
    if len(legends) == 1:
        leg = legends[0]
        ch, cw = _comp_size(leg)
        row_opts = _code_options(grid_h - ch)
        col_opts = _code_options(canvas_w - cw)
        if not row_opts or not col_opts:
            return False
        choices = [(r, c) for r in row_opts for c in col_opts if r > 0 or c > 0]
        if not choices:
            choices = [(r, c) for r in row_opts for c in col_opts]
        best = max(_single_code_score(r, c) for r, c in choices)
        preferred = [(r, c) for r, c in choices if _single_code_score(r, c) == best]
        row_code, col_code = random.choice(preferred)
        leg.update(row_code=row_code, col_code=col_code, comp_top=row_code, comp_left=col_code)
        return True

    a, b = legends
    a_ch, a_cw = _comp_size(a)
    b_ch, b_cw = _comp_size(b)
    a_rows = _code_options(grid_h - a_ch)
    a_cols = _code_options(canvas_w - a_cw)
    b_rows = _code_options(grid_h - b_ch)
    b_cols = _code_options(canvas_w - b_cw)
    if not a_rows or not a_cols or not b_rows or not b_cols:
        return False

    valid_pairs = []
    for ar in a_rows:
        for ac in a_cols:
            at, al = ar, ac
            for br in b_rows:
                for bc in b_cols:
                    bt, bl = br, bc
                    if at < bt + b_ch and bt < at + a_ch and al < bl + b_cw and bl < al + a_cw:
                        continue
                    valid_pairs.append((ar, ac, br, bc))
    if not valid_pairs:
        return False

    best = max(_pair_code_score(pair) for pair in valid_pairs)
    preferred = [pair for pair in valid_pairs if _pair_code_score(pair) == best]
    ar, ac, br, bc = random.choice(preferred)
    a.update(row_code=ar, col_code=ac, comp_top=ar, comp_left=ac)
    b.update(row_code=br, col_code=bc, comp_top=br, comp_left=bc)
    return True


def _build_banded_canvas(grid_h: int, canvas_w: int, bg1: int, bg2: int) -> np.ndarray:
    canvas = np.zeros((grid_h, canvas_w), dtype=int)
    band_dir = random.choice(["h", "v"])
    dim = grid_h if band_dir == "h" else canvas_w
    bw = random.randint(BAND_WIDTH_MIN, max(BAND_WIDTH_MIN, dim // 2))
    band_indices = np.arange(dim) // bw % 2
    colors = np.array([bg1, bg2])
    if band_dir == "h":
        canvas[:] = colors[band_indices][:, None]
    else:
        canvas[:] = colors[band_indices][None, :]
    return canvas


def _overlay_composites(canvas: np.ndarray, legends: list[dict]) -> np.ndarray:
    output = canvas.copy()
    for leg in legends:
        comp = _build_composite(leg["sprite"], leg["rotated_frame"])
        t, l = leg["comp_top"], leg["comp_left"]
        region = output[t : t + comp.shape[0], l : l + comp.shape[1]]
        write_mask = comp != 0
        region[write_mask] = comp[write_mask]
    return output


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _marker_corner_slots(grid_h: int, canvas_w: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (MARKER_INSET, MARKER_INSET),
        "bl": (grid_h - MARKER_PANEL_HEIGHT - MARKER_INSET, MARKER_INSET),
        "tr": (MARKER_INSET, canvas_w - MARKER_PANEL_WIDTH - MARKER_INSET),
        "br": (grid_h - MARKER_PANEL_HEIGHT - MARKER_INSET, canvas_w - MARKER_PANEL_WIDTH - MARKER_INSET),
    }


def _assign_canvas_markers(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
    slots = _marker_corner_slots(grid_h, canvas_w)

    def panel_free(name: str) -> bool:
        r0, c0 = slots[name]
        panel_box = (r0, c0, r0 + MARKER_PANEL_HEIGHT, c0 + MARKER_PANEL_WIDTH)
        for leg in legends:
            ch, cw = _comp_size(leg)
            comp_box = (leg["comp_top"], leg["comp_left"], leg["comp_top"] + ch, leg["comp_left"] + cw)
            if _boxes_intersect(panel_box, comp_box):
                return False
        return True

    if len(legends) == 1:
        valid = [name for name in slots if panel_free(name)]
        if not valid:
            return False
        name = random.choice(valid)
        legends[0].update(marker_r0=slots[name][0], marker_c0=slots[name][1], marker_name=name)
        return True

    valid_pairs = []
    for a_name, b_name in [("tl", "br"), ("bl", "tr")]:
        if panel_free(a_name) and panel_free(b_name):
            valid_pairs.append((a_name, b_name))
    if not valid_pairs:
        return False

    a_name, b_name = random.choice(valid_pairs)
    legends[0].update(marker_r0=slots[a_name][0], marker_c0=slots[a_name][1], marker_name=a_name)
    legends[1].update(marker_r0=slots[b_name][0], marker_c0=slots[b_name][1], marker_name=b_name)
    return True


def _draw_marker_panel(canvas: np.ndarray, leg: dict) -> None:
    r0, c0 = leg["marker_r0"], leg["marker_c0"]
    canvas[r0 : r0 + 5, c0 : c0 + 5] = 0
    canvas[r0, c0 : c0 + 5] = 6
    canvas[r0 + 4, c0 : c0 + 5] = 6
    canvas[r0 : r0 + 5, c0] = 6
    canvas[r0 : r0 + 5, c0 + 4] = 6
    canvas[r0 + 2, c0 + 2] = 4
    for bit in range(3):
        if leg["row_code"] & (1 << bit):
            canvas[r0 + 1 + bit, c0 + 1] = 4
        if leg["col_code"] & (1 << bit):
            canvas[r0 + 1 + bit, c0 + 3] = 4

    rr0 = r0 + 5
    canvas[rr0 : rr0 + 3, c0 : c0 + 5] = 0
    canvas[rr0 : rr0 + 3, c0] = 6
    canvas[rr0 : rr0 + 3, c0 + 4] = 6
    cx_r, cx_c = rr0 + 1, c0 + 2
    dr, dc = ROTATION_DIRS[leg["rot_k"]]
    canvas[cx_r, cx_c] = 7
    canvas[cx_r + dr, cx_c + dc] = 7


def _place_legend(grid: np.ndarray, leg: dict, col_start: int, sep_col: int) -> bool:
    grid[:, sep_col] = 6

    sprite_c0 = col_start + 1
    if sprite_c0 + leg["N"] > sep_col if col_start < sep_col else sprite_c0 + leg["N"] > grid.shape[1]:
        return False

    frame_max_left = col_start + leg["legend_w"] - leg["frame_W"]
    if frame_max_left < col_start:
        return False
    frame_c0 = random.randint(col_start, frame_max_left)

    ur0 = leg["unit_r0"]
    n = leg["N"]
    grid[ur0 : ur0 + n, sprite_c0 : sprite_c0 + n] = leg["legend_sprite"]

    fr0 = leg["frame_r0"]
    frame_region = grid[fr0 : fr0 + leg["frame_H"], frame_c0 : frame_c0 + leg["frame_W"]]
    frame_region[leg["frame_mask"]] = leg["frame_color"]
    return True


def _gen_attempt(num_legends: int | None = None) -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    if num_legends is None:
        num_legends = random.choice(LEGEND_COUNT_CHOICES)

    legends = []
    for _ in range(num_legends):
        n = random.choices([2, 3, 4, 5], weights=[6, 5, 2, 1])[0] if num_legends == 2 else random.randint(*UNIT_SIZE_RANGE)
        legends.append({"N": n, "legend_w": max(n + 2, 5)})

    left_lw = legends[0]["legend_w"]
    left_total = left_lw + 1
    right_total = (1 + legends[1]["legend_w"]) if num_legends == 2 else 0
    canvas_w = GRID_W - left_total - right_total
    if canvas_w < MIN_CANVAS_W or canvas_w <= MARKER_PANEL_WIDTH + 2 * MARKER_INSET:
        return None

    colors = _allocate_colors(num_legends)
    if colors is None:
        return None
    unit_color_sets, (bg1, bg2), frame_colors = colors

    for li in range(num_legends):
        if not _design_legend(
            legends[li],
            unit_color_sets[li],
            frame_colors[li],
            grid_h,
            canvas_w,
            prefer_small=(num_legends == 2),
        ):
            return None

    if not _assign_binary_offsets(legends, grid_h, canvas_w):
        return None

    if not _assign_canvas_markers(legends, grid_h, canvas_w):
        return None

    clean_canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2)
    input_canvas = clean_canvas.copy()
    for leg in legends:
        _draw_marker_panel(input_canvas, leg)
    output = _overlay_composites(clean_canvas, legends)

    grid = np.zeros((grid_h, GRID_W), dtype=int)
    canvas_c0 = left_total
    grid[:, canvas_c0 : canvas_c0 + canvas_w] = input_canvas

    if not _place_legend(grid, legends[0], col_start=0, sep_col=left_lw):
        return None
    if num_legends == 2:
        r_sep = canvas_c0 + canvas_w
        if not _place_legend(grid, legends[1], col_start=r_sep + 1, sep_col=r_sep):
            return None

    try:
        if np.array_equal(main(grid), output):
            return grid
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    # Balance the 1-legend and 2-legend branches at the accepted-sample level rather than
    # relying on raw attempt weights, because the two branches have different rejection rates.
    target_legends = random.choice(LEGEND_COUNT_CHOICES)
    for _ in range(GEN_RETRIES):
        result = _gen_attempt(target_legends)
        if result is not None:
            return result
    raise RuntimeError(
        f"Failed to generate valid input after {GEN_RETRIES} attempts for {target_legends} legends"
    )
