# Variant of a32d8b75
# Dimension: D18 — 标记面板在网格中的位置 (meta)
# Change: 标记面板（位置方块+旋转标记）位于图例列的固定底部位置（rows h-8 to h-1）... → 标记信息嵌入画布中而非图例中

import random
from collections import Counter

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4, random_sprite


# Design analysis for the D18 variant
# -----------------------------------
# a) Concrete behavior:
#    - The legend still teaches "what to stamp" (unit sprite + frame mask), but it no longer
#      stores "where/how to place it".
#    - Each placement instruction is now a full 8x5 marker panel hidden inside the canvas:
#      a 5x5 border-6 square with one yellow pixel for anchor, plus a 3x5 clock-hand strip
#      below it for rotation. Panels live near canvas corners with a 1-cell inset.
#    - Output keeps only the clean banded canvas plus the assembled composites. Therefore the
#      solver must first detect marker panels in the canvas, reconstruct the erased background
#      underneath them, and only then place the composites.
#    - Small trace:
#      old flow: legend -> read sprite/frame/marker -> build composite -> crop to canvas.
#      new flow: legend -> read sprite/frame, canvas -> search corner marker -> erase marker
#      back to bands -> build composite -> place on cleaned canvas.
#
# b) Complexity calibration:
#    - The marker syntax itself is unchanged, so the cognitive load stays close to the source
#      task. Only the location changes.
#    - To keep the search visible but not noisy, panels are limited to 4 corner slots. With
#      two legends we use opposite corners, so humans can still pair left legend with left-side
#      marker and right legend with right-side marker.
#
# c) Old assumptions that break:
#    - Old `_extract_frame_bbox(..., border6_r0)` assumed the frame ends before the bottom marker.
#      That fails once legends no longer contain markers. The new extractor simply uses the
#      remaining non-{0,6} pixels below the unit sprite.
#    - Old `main()` searched all marker panels and ignored those inside the canvas. In this
#      variant that would skip every real marker and draw nothing.
#    - Old output started from the raw canvas because metadata sat outside it. Now starting
#      from the raw canvas would incorrectly preserve the hidden marker panels.
#
# d) Placement legality:
#    - Generator assigns anchors first, then chooses marker corner slots that do not intersect
#      any composite bounding box. This avoids ambiguity between "marker removed" and
#      "composite drawn here".
#    - For two legends the markers are forced to opposite vertical corners, so no row/column is
#      overwhelmed by two panels at once and background recovery stays stable.
#
# e) Visual observability:
#    - The change is obvious in 2-3 examples because the canvas now contains small 6-bordered
#      control panels while the legends are shorter and simpler.
#    - Background colors deliberately avoid marker colors 4 and 7, so the marker panels remain
#      easy to spot and the erased background can be inferred from the surrounding stripe pattern.
#
# f) Cleanliness:
#    - Marker panels sit in isolated corner patches, not mixed into the sprite footprints.
#    - The canvas background is restricted to two clean horizontal or vertical bands, making the
#      "erase marker back to background" step visually legible.
#
# g) Color inferability:
#    - No hidden ordering by color id is introduced. Colors 4 and 7 are reserved only so the
#      marker syntax is visually stable; the actual placement rule still comes from position and
#      direction, not from numeric color ranks.


MARKER_PANEL_HEIGHT = 8
MARKER_PANEL_WIDTH = 5
MARKER_INSET = 1
MARKER_BG_COLORS = {4, 6, 7}
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


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


def _extract_unit_sprite(legend: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    mask = (legend != 0) & (legend != 6)
    pts_all = np.argwhere(mask)
    if pts_all.size == 0:
        return None
    rmin = int(pts_all[:, 0].min())
    cseed = int(np.argwhere(mask[rmin, :])[0, 0])
    h, w = legend.shape
    vis = np.zeros((h, w), dtype=bool)
    st = [(rmin, cseed)]
    vis[rmin, cseed] = True
    pts: list[tuple[int, int]] = []
    while st:
        r, c = st.pop()
        pts.append((r, c))
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not vis[rr, cc]:
                vis[rr, cc] = True
                st.append((rr, cc))
    r0, c0, r1, c1 = bbox_coords(np.asarray(pts))
    spr = legend[r0 : r1 + 1, c0 : c1 + 1].copy()
    cols = sorted({int(x) for x in np.unique(spr) if int(x) not in (0, 6)})
    if len(cols) == 2:
        a, b = cols
        spr2 = spr.copy()
        spr2[spr == a] = -1
        spr2[spr == b] = a
        spr2[spr2 == -1] = b
        spr = spr2
    return spr, (r0, c0, r1, c1)


def _extract_frame_bbox(legend: np.ndarray, unit_bb: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    r_start = unit_bb[2] + 1
    if r_start >= legend.shape[0]:
        return None
    sub = legend[r_start:, :]
    mask = (sub != 0) & (sub != 6)
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    pts[:, 0] += r_start
    r0 = int(pts[:, 0].min())
    r1 = int(pts[:, 0].max())
    c0 = int(pts[:, 1].min())
    c1 = int(pts[:, 1].max())
    return r0, c0, r1, c1


def _panel_square_ok(g: np.ndarray, r0: int, c0: int) -> bool:
    if r0 < 0 or c0 < 0 or r0 + 7 >= g.shape[0] or c0 + 4 >= g.shape[1]:
        return False
    sq = g[r0 : r0 + 5, c0 : c0 + 5]
    if not (
        np.all(sq[0, :] == 6)
        and np.all(sq[4, :] == 6)
        and np.all(sq[:, 0] == 6)
        and np.all(sq[:, 4] == 6)
    ):
        return False
    inner = sq[1:4, 1:4]
    if np.count_nonzero(inner == 4) != 1 or np.count_nonzero(inner) != 1:
        return False
    rot = g[r0 + 5 : r0 + 8, c0 : c0 + 5]
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


def _position_indices_from_border_square(g: np.ndarray, border6_r0: int, border6_c0: int) -> tuple[int, int]:
    inner = g[border6_r0 + 1 : border6_r0 + 4, border6_c0 + 1 : border6_c0 + 4]
    pts = np.argwhere(inner == 4)
    assert pts.shape[0] == 1, "anchor marker must be unique"
    return int(pts[0, 0]), int(pts[0, 1])


def _rotation_from_clockhand(g: np.ndarray, border6_r0: int, border6_c0: int) -> int:
    sub = g[border6_r0 + 5 : border6_r0 + 8, border6_c0 + 1 : border6_c0 + 4]
    pts = np.argwhere(sub == 7)
    assert pts.shape[0] == 2, "rotation marker must have center and one arm"
    center = np.array([1, 1], dtype=int)
    dists = np.abs(pts - center).sum(axis=1)
    p = pts[int(dists.argmax())]
    dr, dc = int(p[0] - center[0]), int(p[1] - center[1])
    if abs(dr) >= abs(dc):
        return 0 if dr < 0 else 2
    return 1 if dc > 0 else 3


def _anchor_to_top_left(anchor_idx: int, size: int, place_size: int) -> int:
    if anchor_idx <= 0:
        canvas_anchor = 0
        obj_anchor = 0
    elif anchor_idx >= 2:
        canvas_anchor = max(0, size - 1)
        obj_anchor = max(0, place_size - 1)
    else:
        canvas_anchor = (size - 1) // 2 if size > 1 else 0
        obj_anchor = (place_size - 1) // 2 if place_size > 1 else 0
    return canvas_anchor - obj_anchor


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


def _legend_panels(g: np.ndarray, left_sep: tuple[int, int], right_sep: tuple[int, int] | None) -> list[np.ndarray]:
    out = [g[:, : left_sep[0]]]
    if right_sep is not None:
        out.append(g[:, right_sep[1] + 1 :])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble enlarged sprites using legends for shape data and canvas-embedded panels for placement metadata.

    Input structure:
    - Color-6 separator columns still split the grid into left legend, main canvas, and optional right legend.
    - Each legend panel now contains only a color-swapped unit sprite near the top and a frame template lower down.
    - The 8x5 marker panel has moved into the canvas itself, tucked into a corner patch.

    Rule:
    - Read each legend's unit sprite and frame template, and read each canvas marker panel for anchor and rotation.
    - Reconstruct the clean striped canvas by removing the hidden marker panel, then tile the unit sprite through the frame, rotate it, and place it according to the decoded anchor.

    Key insight:
    - The control syntax is unchanged, but its location changed: the solver must search the canvas for the small border-6 panel and treat it as metadata rather than foreground content.
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
    composite_written = np.zeros(out.shape, dtype=bool)

    eff_h, eff_w = out.shape
    full6_cols = [c for c in range(eff_w) if np.all(out[:, c] == 6)]
    if full6_cols:
        eff_w = min(full6_cols)
    full6_rows = [r for r in range(eff_h) if np.all(out[r, :] == 6)]
    if full6_rows:
        eff_h = min(full6_rows)

    for panel, (marker_r0, marker_c0) in zip(legends, markers):
        unit = _extract_unit_sprite(panel)
        assert unit is not None, "legend must contain a unit sprite"
        unit_sprite, unit_bb = unit

        frame_bb = _extract_frame_bbox(panel, unit_bb)
        assert frame_bb is not None, "legend must contain a frame"
        fr0, fc0, fr1, fc1 = frame_bb
        frame = panel[fr0 : fr1 + 1, fc0 : fc1 + 1]
        frame_mask0 = (frame != 0) & (frame != 6)
        _, n8 = _scipy_label(frame_mask0, structure=np.ones((3, 3)))
        assert n8 == 1, f"frame must be 8-connected, got {n8} components"

        dir_k = _rotation_from_clockhand(canvas, marker_r0, marker_c0)
        frame_mask = np.rot90(frame_mask0, (-dir_k) % 4)

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

        pr, pc = _position_indices_from_border_square(canvas, marker_r0, marker_c0)
        top = _anchor_to_top_left(pr, eff_h, comp.shape[0])
        left = _anchor_to_top_left(pc, eff_w, comp.shape[1])

        for r in range(comp.shape[0]):
            rr = top + r
            assert 0 <= rr < out.shape[0], "sprite row out of canvas bounds"
            for c in range(comp.shape[1]):
                cc = left + c
                assert 0 <= cc < out.shape[1], "sprite col out of canvas bounds"
                val = int(comp[r, c])
                if val in (0, 6):
                    continue
                assert not composite_written[rr, cc], "composites must not overlap"
                composite_written[rr, cc] = True
                out[rr, cc] = val
    return out


# --------------- generator (canvas_embedded_marker_panels) ---------------

GRID_H_RANGE = (20, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [1, 4]
UNIT_SIZE_RANGE = (2, 5)
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)
FRAME_W_RANGE = (3, 6)
FRAME_DENSITY = 0.5
BAND_WIDTH_MIN = 4
USABLE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
BACKGROUND_COLORS = [1, 2, 3, 5, 8, 9]
GEN_RETRIES = 300


def _allocate_colors(num_legends: int):
    pool = list(USABLE_COLORS)
    random.shuffle(pool)

    unit_color_sets = []
    used_unit = set()
    for _ in range(num_legends):
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


def _design_legend(leg: dict, unit_colors: tuple[int, int], frame_color: int, grid_h: int, canvas_w: int) -> bool:
    n = leg["N"]
    uc1, uc2 = unit_colors

    sprite = np.full((n, n), uc1, dtype=int)
    num_c2 = random.randint(1, n * n - 1)
    for p in random.sample(range(n * n), num_c2):
        sprite[p // n, p % n] = uc2
    legend_sprite = _color_swap(sprite, uc1, uc2)

    combos = []
    max_frame_w = min(FRAME_W_RANGE[1] - 1, leg["legend_w"])
    for fH in range(FRAME_H_RANGE[0], FRAME_H_RANGE[1]):
        for fW in range(FRAME_W_RANGE[0], max_frame_w + 1):
            for rk in range(4):
                rh, rw = (fH, fW) if rk % 2 == 0 else (fW, fH)
                if rh * n <= grid_h and rw * n <= canvas_w:
                    if n + 2 + fH < grid_h:
                        combos.append((fH, fW, rk))
    if not combos:
        return False
    frame_H, frame_W, rot_k = random.choice(combos)

    frame_mask = random_sprite(
        frame_H,
        frame_W,
        density=FRAME_DENSITY,
        connectivity=8,
        color_palette=[1],
        symmetry="not_symmetric",
    ) != 0
    rot_frame = np.rot90(frame_mask, (-rot_k) % 4)

    leg.update(
        sprite=sprite,
        legend_sprite=legend_sprite,
        frame_mask=frame_mask,
        frame_H=frame_H,
        frame_W=frame_W,
        frame_color=frame_color,
        rot_k=rot_k,
        rotated_frame=rot_frame,
    )
    return True


def _comp_size(leg: dict) -> tuple[int, int]:
    rf = leg["rotated_frame"]
    n = leg["N"]
    return rf.shape[0] * n, rf.shape[1] * n


def _assign_anchors(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
    if len(legends) == 1:
        pr, pc = random.randint(0, 2), random.randint(0, 2)
        ch, cw = _comp_size(legends[0])
        legends[0].update(
            pr=pr,
            pc=pc,
            comp_top=_anchor_to_top_left(pr, grid_h, ch),
            comp_left=_anchor_to_top_left(pc, canvas_w, cw),
        )
        return True

    a, b = legends
    a_ch, a_cw = _comp_size(a)
    b_ch, b_cw = _comp_size(b)
    valid_pairs = []
    for apr in range(3):
        for apc in range(3):
            at = _anchor_to_top_left(apr, grid_h, a_ch)
            al = _anchor_to_top_left(apc, canvas_w, a_cw)
            if at < 0 or al < 0 or at + a_ch > grid_h or al + a_cw > canvas_w:
                continue
            for bpr in range(3):
                for bpc in range(3):
                    bt = _anchor_to_top_left(bpr, grid_h, b_ch)
                    bl = _anchor_to_top_left(bpc, canvas_w, b_cw)
                    if bt < 0 or bl < 0 or bt + b_ch > grid_h or bl + b_cw > canvas_w:
                        continue
                    if at < bt + b_ch and bt < at + a_ch and al < bl + b_cw and bl < al + a_cw:
                        continue
                    valid_pairs.append((apr, apc, at, al, bpr, bpc, bt, bl))
    if not valid_pairs:
        return False
    apr, apc, at, al, bpr, bpc, bt, bl = random.choice(valid_pairs)
    a.update(pr=apr, pc=apc, comp_top=at, comp_left=al)
    b.update(pr=bpr, pc=bpc, comp_top=bt, comp_left=bl)
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
        sp, rf = leg["sprite"], leg["rotated_frame"]
        n = leg["N"]
        t, l = leg["comp_top"], leg["comp_left"]
        fh, fw = rf.shape
        comp = np.zeros((fh * n, fw * n), dtype=int)
        sprite_mask = (sp != 0) & (sp != 6)
        for i in range(fh):
            for j in range(fw):
                if rf[i, j]:
                    r0, c0 = i * n, j * n
                    comp[r0 : r0 + n, c0 : c0 + n][sprite_mask] = sp[sprite_mask]
        region = output[t : t + fh * n, l : l + fw * n]
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
    canvas[r0 + 1 + leg["pr"], c0 + 1 + leg["pc"]] = 4

    rr0 = r0 + 5
    canvas[rr0 : rr0 + 3, c0 : c0 + 5] = 0
    canvas[rr0 : rr0 + 3, c0] = 6
    canvas[rr0 : rr0 + 3, c0 + 4] = 6
    cx_r, cx_c = rr0 + 1, c0 + 2
    dr, dc = ROTATION_DIRS[leg["rot_k"]]
    canvas[cx_r, cx_c] = 7
    canvas[cx_r + dr, cx_c + dc] = 7


def _place_legend(grid: np.ndarray, leg: dict, col_start: int, sep_col: int) -> bool:
    n = leg["N"]
    grid[:, sep_col] = 6
    grid[1 : 1 + n, col_start + 1 : col_start + 1 + n] = leg["legend_sprite"]

    min_fr_start = n + 2
    max_fr_start = grid.shape[0] - leg["frame_H"] - 1
    if max_fr_start < min_fr_start:
        return False
    fr_start = random.randint(min_fr_start, max_fr_start)
    max_f_col = leg["legend_w"] - leg["frame_W"]
    if max_f_col < 0:
        return False
    f_col = random.randint(0, max_f_col)
    frame_region = grid[fr_start : fr_start + leg["frame_H"], col_start + f_col : col_start + f_col + leg["frame_W"]]
    frame_region[leg["frame_mask"]] = leg["frame_color"]
    return True


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]

    legends = []
    for _ in range(num_legends):
        n = random.randint(*UNIT_SIZE_RANGE)
        legends.append({"N": n, "legend_w": n + 2})

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
        if not _design_legend(legends[li], unit_color_sets[li], frame_colors[li], grid_h, canvas_w):
            return None

    if not _assign_anchors(legends, grid_h, canvas_w):
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
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
