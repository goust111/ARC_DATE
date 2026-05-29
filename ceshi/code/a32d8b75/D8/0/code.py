# Variant of a32d8b75
# Dimension: D8 — 符号编码：图例面板的整体结构 (symbol_instructions)
# Change: 4层图例从上到下依次为：单元精灵、框架模板、位置标记方块、旋转标记，每层之间有间隔... → 图例排列顺序随机化

import random

import numpy as np
from arc_utils import bbox_coords, random_sprite
from scipy.ndimage import label as _scipy_label


# Design analysis for the D8 variant
# - Concrete change:
#   The four legend components keep their original shapes and meanings, but their
#   vertical order inside each legend is now a random permutation. The solver can
#   no longer assume "top nonzero block = unit sprite" or "5x5 border square has
#   the rotation strip immediately below it". Instead it must identify the parts by
#   geometry:
#   unit sprite = filled NxN square with exactly 2 colors,
#   frame = single-color sparse 8-connected mask,
#   position marker = 5x5 border-6 square with one interior pixel,
#   rotation marker = 3x5 strip with side borders and a 2-pixel clock hand.
# - Behavior trace:
#   Original flow used row order twice: it extracted the top-most non-{0,6}
#   component as the unit, then treated every non-{0,6} pixel between that unit and
#   the bordered square as frame, and finally read the rotation only from the rows
#   directly under that square. In the variant, a legend may be ordered as
#   [rotation, frame, unit, marker]. The old code would misread the 2-pixel rotation
#   glyph as the "top-most unit" and then either fail or build nonsense. The new
#   code first isolates each legend panel by the separator columns, then detects the
#   four roles by their local shape constraints, so the row permutation becomes
#   irrelevant.
# - Complexity calibration:
#   The variant changes only the instruction layout, not the canvas rule. To keep
#   the shuffled order visually obvious without introducing unrelated mechanisms, the
#   generator uses the same four symbols and the same placement rule, but enforces a
#   blank row gap between components so their identities remain readable.
# - Hidden assumptions audit:
#   Any logic that depended on fixed y-order was removed:
#   `_extract_unit_sprite(top-most)`, `_extract_frame_bbox(unit..square span)`, and
#   `_rotation_from_clockhand(square+5 rows)` were all order-sensitive in the
#   original code. The replacement parser only assumes that each legend contains one
#   instance of each visual component and that the generator separates them by at
#   least one zero row.
# - Placement legality:
#   The generator still works output-first. It picks frame size and rotation only if
#   the composite fits the canvas, and for 2 legends it rejects overlapping anchor
#   placements. Within a legend, the generator packs the four components with fixed
#   minimum gaps and rejects size combinations that would exceed the chosen grid
#   height.
# - Visual observability and neatness:
#   Randomizing among 24 possible orders creates a directly visible D8 difference.
#   The components themselves remain distinctive; the frame is single-color and
#   sparse, while the unit sprite is a solid two-color square, so a human can infer
#   that the symbols must be recognized by shape rather than row position.
# - Compatibility tweak:
#   Legends reserve at least 5 columns (`legend_w = max(N + 2, 5)`) so the bordered
#   square and rotation strip fit entirely inside the legend instead of borrowing the
#   separator column. This keeps separator detection stable after the order shuffle.


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
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 300
MIN_COMPONENT_GAP = 1
LEGEND_COMPONENTS = ("unit", "frame", "marker", "rotation")


def _separator_runs(g: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous runs of full-height columns that are all color 6."""
    h, w = g.shape
    cols6 = [c for c in range(w) if np.all(g[:, c] == 6)]
    if not cols6:
        return []
    cols6.sort()
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
    """
    Identify a left separator run (between left legend and canvas), and optionally a
    right separator run (between canvas and a right-side legend panel).
    """
    h, w = g.shape
    runs = _separator_runs(g)
    assert runs, "must have at least one color-6 separator run"
    left = runs[0]
    right: tuple[int, int] | None = None
    for s, e in reversed(runs[1:]):
        if s >= w - 10 and s > left[1] + 1:
            right = (s, e)
            break
    return left, right


def _legend_panels(
    g: np.ndarray, left_sep: tuple[int, int], right_sep: tuple[int, int] | None
) -> list[np.ndarray]:
    panels: list[np.ndarray] = []
    if left_sep[0] > 0:
        panels.append(g[:, : left_sep[0]])
    if right_sep is not None and right_sep[1] + 1 < g.shape[1]:
        panels.append(g[:, right_sep[1] + 1 :])
    return panels


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
                "mask": local_mask,
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


def _find_position_indices(panel: np.ndarray) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    h, w = panel.shape
    for r0 in range(h - 4):
        for c0 in range(w - 4):
            sub = panel[r0 : r0 + 5, c0 : c0 + 5]
            if not (
                np.all(sub[0, :] == 6)
                and np.all(sub[4, :] == 6)
                and np.all(sub[:, 0] == 6)
                and np.all(sub[:, 4] == 6)
            ):
                continue
            inner = sub[1:4, 1:4]
            pts = np.argwhere((inner != 0) & (inner != 6))
            if pts.shape[0] != 1:
                continue
            tmp = inner.copy()
            tmp[pts[0, 0], pts[0, 1]] = 0
            if np.any((tmp != 0) & (tmp != 6)):
                continue
            candidates.append((int(pts[0, 0]), int(pts[0, 1])))
    assert len(candidates) == 1, f"expected exactly one marker square, got {len(candidates)}"
    return candidates[0]


def _find_rotation_dir(panel: np.ndarray) -> int:
    candidates: list[int] = []
    h, w = panel.shape
    for r0 in range(h - 2):
        for c0 in range(w - 4):
            sub = panel[r0 : r0 + 3, c0 : c0 + 5]
            if not (np.all(sub[:, 0] == 6) and np.all(sub[:, 4] == 6)):
                continue
            inner = sub[:, 1:4]
            pts = np.argwhere((inner != 0) & (inner != 6))
            if pts.shape[0] != 2:
                continue
            pts_set = {tuple(map(int, p)) for p in pts}
            if (1, 1) not in pts_set:
                continue
            arm = next(p for p in pts_set if p != (1, 1))
            dr, dc = arm[0] - 1, arm[1] - 1
            if abs(dr) + abs(dc) != 1:
                continue
            candidates.append(ROTATION_DIRS.index((dr, dc)))
    assert len(candidates) == 1, f"expected exactly one rotation patch, got {len(candidates)}"
    return candidates[0]


def _anchor_to_top_left(anchor_idx: int, size: int, place_size: int) -> int:
    """
    anchor_idx in {0,1,2} selects position on a length:
      0 -> start, 1 -> middle, 2 -> end
    Returns the top-left coordinate so that the corresponding point of the placed object
    aligns with that point on the canvas.
    """
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild enlarged sprites from legends whose instruction symbols appear in shuffled vertical order.

    Input structure:
    - Full-height color-6 separator columns split the grid into a left legend, the output canvas, and optionally a right legend.
    - Every legend still contains the same four visual components, but their top-to-bottom order is no longer fixed: a filled two-color unit square, a sparse one-color frame mask, a 5x5 bordered anchor square, and a 3x5 rotation strip.

    Rule:
    - Identify each legend component by its shape, tile the color-swapped unit sprite into every active frame cell, rotate that composite according to the clock-hand strip, and place it on the canvas using the anchor encoded by the bordered square's 3x3 interior.

    Key insight:
    - The vertical arrangement is now a distractor; the semantic role of each legend piece comes from its geometry, so the solver must parse symbols by appearance rather than by row order.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    canvas = g[:, canvas_c0:canvas_c1].copy()

    eff_h, eff_w = canvas.shape
    full6_cols = [c for c in range(eff_w) if np.all(canvas[:, c] == 6)]
    if full6_cols:
        eff_w = min(full6_cols)
    full6_rows = [r for r in range(eff_h) if np.all(canvas[r, :] == 6)]
    if full6_rows:
        eff_h = min(full6_rows)

    out = canvas.copy()
    composite_written = np.zeros(out.shape, dtype=bool)

    panels = _legend_panels(g, left_sep, right_sep)
    assert panels, "no legend panels found"

    for panel in panels:
        unit_sprite = _find_unit_sprite(panel)
        frame_mask0 = _find_frame_mask(panel)
        pr, pc = _find_position_indices(panel)
        dir_k = _find_rotation_dir(panel)

        frame_mask = np.rot90(frame_mask0, (-dir_k) % 4)
        comp = _build_composite(unit_sprite, frame_mask)

        top = _anchor_to_top_left(pr, eff_h, comp.shape[0])
        left = _anchor_to_top_left(pc, eff_w, comp.shape[1])

        ch, cw = comp.shape
        for r in range(ch):
            rr = top + r
            assert 0 <= rr < out.shape[0], "sprite row out of canvas bounds"
            for c in range(cw):
                cc = left + c
                assert 0 <= cc < out.shape[1], "sprite col out of canvas bounds"
                if out[rr, cc] == 6:
                    continue
                val = int(comp[r, c])
                if val != 0 and val != 6:
                    assert not composite_written[rr, cc], "composites must not overlap"
                    composite_written[rr, cc] = True
                    out[rr, cc] = val
    return out


def _allocate_colors(num_legends):
    """Pick unit colors (2 per legend), 2 background colors, and 1 frame color per legend."""
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

    bg_cands = [c for c in pool if c not in used_unit]
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


def _color_swap(sprite, c1, c2):
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


def _layout_rows(leg: dict, grid_h: int) -> None:
    heights = {
        "unit": leg["N"],
        "frame": leg["frame_H"],
        "marker": 5,
        "rotation": 3,
    }
    base_total = sum(heights.values()) + MIN_COMPONENT_GAP * (len(LEGEND_COMPONENTS) - 1)
    assert base_total <= grid_h, "legend does not fit vertically"

    order = random.sample(list(LEGEND_COMPONENTS), k=len(LEGEND_COMPONENTS))
    extras = _distribute_extra(grid_h - base_total, len(LEGEND_COMPONENTS) + 1)

    rows = {}
    cursor = extras[0]
    for idx, name in enumerate(order):
        rows[name] = cursor
        cursor += heights[name]
        if idx < len(order) - 1:
            cursor += MIN_COMPONENT_GAP + extras[idx + 1]

    leg.update(
        order=tuple(order),
        unit_r0=rows["unit"],
        frame_r0=rows["frame"],
        marker_r0=rows["marker"],
        rot_r0=rows["rotation"],
    )


def _design_legend(leg, unit_colors, frame_color, grid_h, canvas_w):
    """Fill in sprite, frame, rotation, and shuffled vertical layout for one legend."""
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
            base_total = n + frame_h + 5 + 3 + MIN_COMPONENT_GAP * 3
            if base_total > grid_h:
                continue
            for rot_k in range(4):
                rh, rw = (frame_h, frame_w) if rot_k % 2 == 0 else (frame_w, frame_h)
                if rh * n <= grid_h and rw * n <= canvas_w:
                    combos.append((frame_h, frame_w, rot_k))
    if not combos:
        return False

    frame_h, frame_w, rot_k = random.choice(combos)
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
    _layout_rows(leg, grid_h)
    return True


def _comp_size(leg):
    rf = leg["rotated_frame"]
    n = leg["N"]
    return rf.shape[0] * n, rf.shape[1] * n


def _assign_anchors(legends, grid_h, canvas_w):
    """Assign anchor positions to legends, avoiding overlap for 2-legend case."""
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

    a, b = legends[0], legends[1]
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


def _build_banded_canvas(grid_h, canvas_w, bg1, bg2):
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


def _overlay_composites(canvas, legends):
    output = canvas.copy()
    for leg in legends:
        sp, rf = leg["sprite"], leg["rotated_frame"]
        n = leg["N"]
        t, l = leg["comp_top"], leg["comp_left"]
        comp = _build_composite(sp, rf)
        region = output[t : t + comp.shape[0], l : l + comp.shape[1]]
        write_mask = (comp != 0) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


def _place_legend(grid, leg, col_start, sep_col, is_right):
    """Draw one legend panel with shuffled vertical component order."""
    grid[:, sep_col] = 6

    legend_w = leg["legend_w"]
    instr_c0 = col_start if is_right else col_start + legend_w - 5
    sprite_c0 = col_start + 1
    frame_c0 = instr_c0 + random.randint(0, max(0, 5 - leg["frame_W"]))

    mr0 = leg["marker_r0"]
    grid[mr0 : mr0 + 5, instr_c0 : instr_c0 + 5] = 0
    grid[mr0, instr_c0 : instr_c0 + 5] = 6
    grid[mr0 + 4, instr_c0 : instr_c0 + 5] = 6
    grid[mr0 : mr0 + 5, instr_c0] = 6
    grid[mr0 : mr0 + 5, instr_c0 + 4] = 6
    grid[mr0 + 1 + leg["pr"], instr_c0 + 1 + leg["pc"]] = 4

    rr0 = leg["rot_r0"]
    grid[rr0 : rr0 + 3, instr_c0 : instr_c0 + 5] = 0
    grid[rr0 : rr0 + 3, instr_c0] = 6
    grid[rr0 : rr0 + 3, instr_c0 + 4] = 6
    cx_r, cx_c = rr0 + 1, instr_c0 + 2
    dr, dc = ROTATION_DIRS[leg["rot_k"]]
    grid[cx_r, cx_c] = 7
    grid[cx_r + dr, cx_c + dc] = 7

    ur0 = leg["unit_r0"]
    n = leg["N"]
    grid[ur0 : ur0 + n, sprite_c0 : sprite_c0 + n] = leg["legend_sprite"]

    fr0 = leg["frame_r0"]
    frame_region = grid[fr0 : fr0 + leg["frame_H"], frame_c0 : frame_c0 + leg["frame_W"]]
    frame_region[leg["frame_mask"]] = leg["frame_color"]


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]

    legends = []
    for _ in range(num_legends):
        n = random.randint(*UNIT_SIZE_RANGE)
        legends.append({"N": n, "legend_w": max(n + 2, 5)})

    left_lw = legends[0]["legend_w"]
    left_total = left_lw + 1
    right_total = (1 + legends[1]["legend_w"]) if num_legends == 2 else 0
    canvas_w = GRID_W - left_total - right_total
    if canvas_w < MIN_CANVAS_W:
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

    canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2)
    output = _overlay_composites(canvas, legends)

    grid = np.zeros((grid_h, GRID_W), dtype=int)
    canvas_c0 = left_total
    grid[:, canvas_c0 : canvas_c0 + canvas_w] = canvas

    _place_legend(grid, legends[0], col_start=0, sep_col=left_lw, is_right=False)
    if num_legends == 2:
        r_sep = canvas_c0 + canvas_w
        _place_legend(grid, legends[1], col_start=r_sep + 1, sep_col=r_sep, is_right=True)

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
