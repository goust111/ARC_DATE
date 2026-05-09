# Variant of a32d8b75
# Dimension: D1 — 匹配条件：精灵组装方式 (match_cond)
# Change: s1（单元精灵）替换 d1（框架模板）中每个活跃像素，形成分形式的放大复合精灵... → 计数匹配：框架中的连通空洞数量决定铺贴重复次数

import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4


# Design analysis for the D1 variant
# a) Concrete behavior:
# - The legend keeps the original four blocks: unit sprite, frame area, 5x5 anchor square, 3x3 rotation clock-hand.
# - The frame is no longer a sparse template that directly dictates where copies go. Instead it is a solid 5-column rectangle
#   with 2 or 3 disconnected zero-colored holes inside it. The solver must count those enclosed hole components.
# - If the frame has k holes, the output composite is a simple k x k repetition of the unit sprite. Example:
#   original rule on a 6x5 frame would tile into every colored frame cell (dozens of copies); new rule on the same frame
#   ignores that silhouette and places only a 2x2 or 3x3 repeated square, depending on the hole count.
# - Rotation and anchor placement still happen after assembly, so D1 is the only semantic change.
# b) Complexity calibration:
# - Hole counts are restricted to {2, 3}. This is visually obvious, avoids the degenerate k=1 case, and keeps composite sizes
#   small enough to fit while still being noticeably different.
# - Two legends are still favored; when two legends exist, the generator assigns different hole counts so one sample usually
#   shows both branches side by side.
# - The unit sprite is forced to be rotationally asymmetric so the preserved rotation marker remains observable even though
#   the repeated block itself is square.
# c) Old-logic assumptions that must be replaced:
# - Old main() assumed every nonzero frame cell is an output placement site. That premise is false here; colored frame area
#   is now mostly irrelevant except for enclosing holes.
# - Old main() rotated a boolean frame mask before assembly. New main() rotates the finished repeated block instead.
# - Old main() treated zeros inside the frame bbox as empty background. In the variant, those zeros are the signal, so the
#   solver must count their 4-connected components and must assume the frame border is solid enough to enclose them.
# d) Placement legality:
# - Composite size is now (k*N) x (k*N), independent of frame height. The generator computes this exact size first, chooses
#   anchors only when the composite fits, and rejects overlapping bounding boxes for the 2-legend case.
# - Frames keep a solid border, so hole counting is stable and does not leak to the outside of the bbox.
# e) Visual observability:
# - The repeated-output size differs sharply between 2-hole and 3-hole legends, while the frame rectangles themselves can
#   have similar outer size. That makes the “count holes, not colored cells” rule visible from a few examples.
# - Because unit sprites are asymmetric, the rotation marker still changes the visible orientation of the repeated block.
# f) Visual neatness:
# - Holes are small and isolated inside a plain rectangle, making them easy to count.
# - The output canvas still uses simple band backgrounds and non-overlapping composites, so there is no confusing stacking.
# g) Color inference:
# - No color id participates in ordering or counting. Colors only separate roles: separator=6, anchor marker=4,
#   rotation marker=7, frame color distinct from unit colors, and background colors distinct from unit colors.
# h) Variant-degeneration check used after implementation:
# - For generated examples, the old solver should usually disagree with the new solver because a solid 5-column frame has
#   far more colored cells than k^2 repeats. This is stronger than merely checking main(input) runs.
# i) New solver assumptions guaranteed by generate_input():
# - Each extracted frame bbox has a nonzero outer border.
# - Interior holes are 4-disconnected and lie strictly inside that border.
# - Hole count is always 2 or 3.
# - Unit sprites are fully filled and rotationally asymmetric.


STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


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


def _all_marker_panel_squares(g: np.ndarray) -> list[tuple[int, int]]:
    """List all 5x5 border-6 squares that look like the top of an 8x5 marker panel."""
    h, w = g.shape
    out: list[tuple[int, int]] = []
    for r0 in range(h - 7):
        for c0 in range(w - 4):
            sub = g[r0 : r0 + 5, c0 : c0 + 5]
            if not (np.all(sub[0, :] == 6) and np.all(sub[4, :] == 6) and np.all(sub[:, 0] == 6) and np.all(sub[:, 4] == 6)):
                continue
            inner = sub[1:4, 1:4]
            if not np.any((inner != 0) & (inner != 6)):
                continue
            rot = g[r0 + 5 : r0 + 8, c0 + 1 : c0 + 4]
            if rot.shape != (3, 3) or not np.any((rot != 0) & (rot != 6)):
                continue
            out.append((int(r0), int(c0)))
    out.sort()
    return out


def _extract_unit_sprite(legend: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    # Unit sprite is the top-most 4-connected component of pixels not in {0,6}.
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


def _extract_frame_bbox(legend: np.ndarray, unit_bb: tuple[int, int, int, int], border6_r0: int) -> tuple[int, int, int, int] | None:
    # Frame pixels live between the unit sprite and the border6 square.
    r_start = unit_bb[2] + 1
    r_end = border6_r0 - 1
    if r_start > r_end:
        return None
    sub = legend[r_start : r_end + 1, :]
    mask = (sub != 0) & (sub != 6)
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    pts[:, 0] += r_start
    r0 = int(pts[:, 0].min())
    r1 = int(pts[:, 0].max())
    c0 = int(pts[:, 1].min())
    c1 = int(pts[:, 1].max())
    return (r0, c0, r1, c1)


def _position_indices_from_border_square(legend: np.ndarray, border6_r0: int, border6_c0: int) -> tuple[int, int] | None:
    sq = legend[border6_r0 : border6_r0 + 5, border6_c0 : border6_c0 + 5]
    inner = sq[1:4, 1:4]
    pts = np.argwhere((inner != 0) & (inner != 6))
    if pts.size == 0:
        return None
    r, c = int(pts[0, 0]), int(pts[0, 1])
    return r, c


def _rotation_from_clockhand(legend: np.ndarray, border6_r0: int, border6_c0: int) -> int:
    """
    The rotation mark sits in the 3x3 patch below the bordered square. Use the endpoint farthest
    from the center to decode up/right/down/left into quarter-turns.
    """
    h, w = legend.shape
    r0 = border6_r0 + 5
    c0 = border6_c0 + 1
    assert r0 + 2 < h and c0 + 2 < w, "rotation marker patch out of bounds"
    sub = legend[r0 : r0 + 3, c0 : c0 + 3]
    pts = np.argwhere((sub != 0) & (sub != 6))
    assert pts.size > 0, "no rotation marker pixel found"
    center = np.array([1, 1], dtype=int)
    dists = np.abs(pts - center).sum(axis=1)
    p = pts[int(dists.argmax())]
    dr, dc = int(p[0] - center[0]), int(p[1] - center[1])
    if abs(dr) >= abs(dc):
        if dr < 0:
            return 0
        if dr > 0:
            return 2
    else:
        if dc > 0:
            return 1
        if dc < 0:
            return 3
    raise AssertionError("rotation marker pixel at center, direction ambiguous")


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


def _count_frame_holes(frame: np.ndarray) -> int:
    """Count 4-connected zero holes enclosed by a solid frame bbox."""
    assert frame.shape[0] >= 3 and frame.shape[1] >= 3, "frame too small for enclosed holes"
    assert np.all(frame[0, :] != 0) and np.all(frame[-1, :] != 0), "frame top/bottom border must be solid"
    assert np.all(frame[:, 0] != 0) and np.all(frame[:, -1] != 0), "frame left/right border must be solid"
    hole_mask = frame == 0
    _, num_holes = _scipy_label(hole_mask, structure=STRUCT4)
    assert num_holes > 0, "frame must contain at least one hole"
    return int(num_holes)


def _repeat_sprite(unit_sprite: np.ndarray, repeat_k: int) -> np.ndarray:
    """Repeat the unit sprite in a square k x k block."""
    assert repeat_k >= 1
    return np.tile(unit_sprite, (repeat_k, repeat_k))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble and place repeated sprites on a canvas using legend-panel instructions.

    Input structure:
    - Color-6 separator columns still split the grid into legend panels and the output canvas.
    - Each legend panel still contains, from top to bottom, a two-color unit sprite, a frame rectangle,
      a 5x5 bordered anchor square, and a 3x3 rotation marker.

    Rule:
    - Read the unit sprite, then count the number of disconnected zero holes enclosed inside the frame.
    - If the frame has k holes, build a k x k repeated block of the unit sprite, rotate that block by the
      clock-hand direction, and place it on the canvas using the anchor encoded in the bordered square.

    Key insight: The frame no longer contributes its colored silhouette to the output; only its enclosed
    hole count matters, and that count controls the repetition factor of the repeated sprite block.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    main = g[:, canvas_c0:canvas_c1].copy()

    marker_squares = _all_marker_panel_squares(g)
    assert marker_squares, "no marker panel found"

    eff_h, eff_w = main.shape
    full6_cols = [c for c in range(eff_w) if np.all(main[:, c] == 6)]
    if full6_cols:
        eff_w = min(full6_cols)
    full6_rows = [r for r in range(eff_h) if np.all(main[r, :] == 6)]
    if full6_rows:
        eff_h = min(full6_rows)

    out = main.copy()
    composite_written = np.zeros(out.shape, dtype=bool)

    for marker_r0, marker_c0 in marker_squares:
        if canvas_c0 <= marker_c0 < canvas_c1:
            continue
        if marker_c0 + 5 > w:
            continue

        panel = g[:, marker_c0 : marker_c0 + 5]
        unit = _extract_unit_sprite(panel)
        if unit is None:
            continue
        unit_sprite, unit_bb = unit

        frame_bb = _extract_frame_bbox(panel, unit_bb, marker_r0)
        if frame_bb is None:
            continue
        fr0, fc0, fr1, fc1 = frame_bb
        frame = panel[fr0 : fr1 + 1, fc0 : fc1 + 1]
        frame_mask = (frame != 0) & (frame != 6)
        _, n8 = _scipy_label(frame_mask, structure=np.ones((3, 3), dtype=int))
        assert n8 == 1, f"frame must be 8-connected, got {n8} components"

        repeat_k = _count_frame_holes(frame)
        comp0 = _repeat_sprite(unit_sprite, repeat_k)
        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        comp = np.rot90(comp0, (-dir_k) % 4)

        pos = _position_indices_from_border_square(g, marker_r0, marker_c0)
        if pos is None:
            continue
        pr, pc = pos

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


# --------------- generator (hole_count_repeat_block) ---------------

GRID_H_RANGE = (20, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [1, 4]
UNIT_SIZE_RANGE = (2, 5)
MIN_CANVAS_W = 8
FRAME_HEIGHT_CHOICES = [6, 7, 8]
FRAME_W = 5
HOLE_COUNT_CHOICES = [2, 3]
BAND_WIDTH_MIN = 4
MARKER_PANEL_HEIGHT = 8
USABLE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 300
SPRITE_RETRIES = 60
HOLE_RETRIES = 60


def _allocate_colors(num_legends: int):
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


def _color_swap(sprite: np.ndarray, c1: int, c2: int) -> np.ndarray:
    """Return a copy of sprite with the two colors swapped."""
    sa, sb = sorted([c1, c2])
    out = sprite.copy()
    out[sprite == sa] = sb
    out[sprite == sb] = sa
    return out


def _is_rotationally_asymmetric(sprite: np.ndarray) -> bool:
    return all(not np.array_equal(sprite, np.rot90(sprite, k)) for k in (1, 2, 3))


def _make_asymmetric_sprite(size: int, c1: int, c2: int) -> np.ndarray | None:
    """Generate a filled two-color square sprite whose rotations are all distinct."""
    for _ in range(SPRITE_RETRIES):
        sprite = np.full((size, size), c1, dtype=int)
        num_c2 = random.randint(1, size * size - 1)
        for p in random.sample(range(size * size), num_c2):
            sprite[p // size, p % size] = c2
        if _is_rotationally_asymmetric(sprite):
            return sprite
    return None


def _pick_isolated_cells(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    """Pick `count` interior cells with pairwise Manhattan distance > 1."""
    cells = [(r, c) for r in range(height) for c in range(width)]
    for _ in range(HOLE_RETRIES):
        random.shuffle(cells)
        chosen: list[tuple[int, int]] = []
        for r, c in cells:
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in chosen):
                chosen.append((r, c))
                if len(chosen) == count:
                    return chosen
    return None


def _make_holey_frame(frame_h: int, frame_color: int, hole_count: int) -> np.ndarray | None:
    """Build a solid 5-column frame rectangle with `hole_count` disconnected zero holes."""
    assert frame_h >= 4
    frame = np.full((frame_h, FRAME_W), frame_color, dtype=int)
    inner_h = frame_h - 2
    inner_w = FRAME_W - 2
    cells = _pick_isolated_cells(inner_h, inner_w, hole_count)
    if cells is None:
        return None
    for r, c in cells:
        frame[r + 1, c + 1] = 0
    if _count_frame_holes(frame) != hole_count:
        return None
    return frame


def _design_legend(leg: dict, unit_colors: tuple[int, int], frame_color: int, hole_count: int, marker_r0: int) -> bool:
    """Fill in sprite, frame, repetition count, and rotation for one legend."""
    n = leg["N"]
    uc1, uc2 = unit_colors
    sprite = _make_asymmetric_sprite(n, uc1, uc2)
    if sprite is None:
        return False
    legend_sprite = _color_swap(sprite, uc1, uc2)

    avail = marker_r0 - (n + 1)
    feasible_heights = [fh for fh in FRAME_HEIGHT_CHOICES if avail >= fh + 1]
    if not feasible_heights:
        return False
    frame_h = random.choice(feasible_heights)
    frame_patch = _make_holey_frame(frame_h, frame_color, hole_count)
    if frame_patch is None:
        return False

    rot_k = random.randint(0, 3)
    comp0 = _repeat_sprite(sprite, hole_count)
    rotated_comp = np.rot90(comp0, (-rot_k) % 4)

    leg.update(
        sprite=sprite,
        legend_sprite=legend_sprite,
        hole_count=hole_count,
        frame_color=frame_color,
        frame_h=frame_h,
        frame_w=FRAME_W,
        frame_patch=frame_patch,
        rot_k=rot_k,
        composite=rotated_comp,
    )
    return True


def _comp_size(leg: dict) -> tuple[int, int]:
    comp = leg["composite"]
    return int(comp.shape[0]), int(comp.shape[1])


def _assign_anchors(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
    """Assign anchor positions to legends, avoiding overlap for 2-legend case."""
    if len(legends) == 1:
        pr, pc = random.randint(0, 2), random.randint(0, 2)
        ch, cw = _comp_size(legends[0])
        top = _anchor_to_top_left(pr, grid_h, ch)
        left = _anchor_to_top_left(pc, canvas_w, cw)
        if top < 0 or left < 0 or top + ch > grid_h or left + cw > canvas_w:
            return False
        legends[0].update(pr=pr, pc=pc, comp_top=top, comp_left=left)
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


def _build_banded_canvas(grid_h: int, canvas_w: int, bg1: int, bg2: int) -> np.ndarray:
    """Build a canvas with alternating horizontal or vertical color bands."""
    canvas = np.zeros((grid_h, canvas_w), dtype=int)
    band_dir = random.choice(["h", "v"])
    dim = grid_h if band_dir == "h" else canvas_w
    bw = random.randint(BAND_WIDTH_MIN, max(BAND_WIDTH_MIN, dim // 2))
    band_indices = np.arange(dim) // bw % 2
    colors = np.array([bg1, bg2], dtype=int)
    if band_dir == "h":
        canvas[:] = colors[band_indices][:, None]
    else:
        canvas[:] = colors[band_indices][None, :]
    return canvas


def _overlay_composites(canvas: np.ndarray, legends: list[dict]) -> np.ndarray:
    """Stamp each legend's repeated composite onto the canvas."""
    output = canvas.copy()
    for leg in legends:
        comp = leg["composite"]
        t, l = leg["comp_top"], leg["comp_left"]
        ch, cw = comp.shape
        region = output[t : t + ch, l : l + cw]
        write_mask = (comp != 0) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


def _place_legend(grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int, is_right: bool) -> None:
    """Draw one legend panel (separator, marker square, rotation patch, sprite, frame)."""
    n = leg["N"]
    grid[:, sep_col] = 6

    mc0 = (col_start + max(0, n - 4)) if not is_right else (sep_col + max(0, n - 3))

    grid[marker_r0 : marker_r0 + 5, mc0 : mc0 + 5] = 0
    grid[marker_r0, mc0 : mc0 + 5] = 6
    grid[marker_r0 + 4, mc0 : mc0 + 5] = 6
    grid[marker_r0 : marker_r0 + 5, mc0] = 6
    grid[marker_r0 : marker_r0 + 5, mc0 + 4] = 6
    grid[marker_r0 + 1 + leg["pr"], mc0 + 1 + leg["pc"]] = 4

    rr0 = marker_r0 + 5
    grid[rr0 : rr0 + 3, mc0 : mc0 + 5] = 0
    grid[rr0 : rr0 + 3, mc0] = 6
    grid[rr0 : rr0 + 3, mc0 + 4] = 6
    cx_r, cx_c = rr0 + 1, mc0 + 2
    dr, dc = ROTATION_DIRS[leg["rot_k"]]
    grid[cx_r, cx_c] = 7
    grid[cx_r + dr, cx_c + dc] = 7

    sp_col = col_start + 1 - mc0
    grid[1 : 1 + n, mc0 + sp_col : mc0 + sp_col + n] = leg["legend_sprite"]

    f_h = leg["frame_h"]
    avail = marker_r0 - (n + 1)
    gap = random.randint(1, avail - f_h)
    fr_start = n + 1 + gap
    grid[fr_start : fr_start + f_h, mc0 : mc0 + FRAME_W] = leg["frame_patch"]


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
    if canvas_w < MIN_CANVAS_W:
        return None

    colors = _allocate_colors(num_legends)
    if colors is None:
        return None
    unit_color_sets, (bg1, bg2), frame_colors = colors

    marker_r0 = grid_h - MARKER_PANEL_HEIGHT

    if num_legends == 2:
        hole_counts = list(HOLE_COUNT_CHOICES)
        random.shuffle(hole_counts)
    else:
        hole_counts = [random.choice(HOLE_COUNT_CHOICES)]

    for li in range(num_legends):
        if not _design_legend(legends[li], unit_color_sets[li], frame_colors[li], hole_counts[li], marker_r0):
            return None

    if not _assign_anchors(legends, grid_h, canvas_w):
        return None

    canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2)
    output = _overlay_composites(canvas, legends)

    grid = np.zeros((grid_h, GRID_W), dtype=int)
    canvas_c0 = left_total
    grid[:, canvas_c0 : canvas_c0 + canvas_w] = canvas

    _place_legend(grid, legends[0], col_start=0, sep_col=left_lw, marker_r0=marker_r0, is_right=False)
    if num_legends == 2:
        r_sep = canvas_c0 + canvas_w
        _place_legend(grid, legends[1], col_start=r_sep + 1, sep_col=r_sep, marker_r0=marker_r0, is_right=True)

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
