# Variant of a32d8b75
# Dimension: D16 — 锚点对齐语义 (meta)
# Change: 锚点索引0/1/2分别表示起始/中间/结束，独立应用于行和列方向
#   → 锚点表示复合精灵的哪个角/中心对齐到画布对应位置

import numpy as np

from arc_utils import bbox_coords, neighbors4, random_sprite
from scipy.ndimage import label as _scipy_label
import random


# Design analysis for the D16 variant
# a) Concrete behavior trace
#    The legend layout stays the same: unit sprite, frame mask, 5x5 marker square, and
#    a 3x3 rotation patch. The visible change is only in how the marker square drives
#    final placement. The 3x3 marker still yields (pr, pc) in {0,1,2}^2, but these
#    indices now pick the composite's own anchor part:
#      row 0/1/2 -> top / vertical center / bottom of the composite
#      col 0/1/2 -> left / horizontal center / right of the composite
#    That selected composite point is aligned to one fixed canvas reference point:
#    the center cell of the effective canvas, computed independently on rows and cols.
#    Small trace:
#      effective canvas = 21x15, so the fixed reference point is (10,7)
#      composite size = 6x4, marker = (0,2)
#      original task: top = 0, left = 14 - 3 = 11, so the object sits at the top-right
#      this variant:  top = 10 - 0 = 10, left = 7 - 3 = 4, so the object's top-right
#      corner touches the center reference point instead
#    The input still looks like the original task, but the output objects cluster around
#    the same central reference rather than around matching canvas edges.
# b) Complexity calibration
#    The rule still uses only three anchor levels per axis, so there are only nine
#    placements per legend. This keeps the search space small and repeated across
#    examples. To prevent the generator from drifting into trivial cases, it rejects
#    samples where the new placement matches the legacy placement on every legend.
# c) Hidden-assumption review for main()
#    - Parsing separator runs, marker panels, unit sprites, frame bboxes, and rotation
#      markers does not rely on the old anchor meaning, so those routines remain valid.
#    - The original _anchor_to_top_left assumed the marker chose both the canvas anchor
#      and the object anchor simultaneously. That assumption breaks under D16 and must
#      be replaced with a fixed canvas-center reference plus a variable object anchor.
#    - The write loop assumes the chosen top-left already keeps the full composite
#      in bounds. The new generator must therefore pre-filter anchor choices more
#      carefully, because top/left can now shift toward negative values or toward the
#      lower/right half even when the composite would fit under the legacy semantics.
# d) Placement legality
#    The generator enumerates all nine anchor choices per legend under the new mapping,
#    keeps only in-bounds placements, and for two legends rejects overlapping bounding
#    boxes. The solver still asserts on overlap and bounds, so generator-side filtering
#    is mandatory.
# e) Visual observability
#    The change is visible because the same marker cell now causes a markedly different
#    position: non-center anchors make one composite corner or edge touch the canvas
#    center instead of the corresponding canvas side. The generator requires at least
#    one legend whose new placement differs from the original semantics, so samples do
#    not silently collapse back to the seed task.
# f) Visual cleanliness
#    No new marks are added to the grid. Cleanliness comes from keeping the background
#    unchanged and enforcing non-overlap of composites. With at most two legends, the
#    output remains legible even though objects are pulled toward the center.
# g) Color inferability
#    Colors keep their original roles: two colors inside the unit sprite, one frame
#    color, striped canvas colors, and fixed separator/marker colors. The variant does
#    not add any color-dependent branching, so there is no hidden color code to infer.
# h) New assumptions introduced by the variant
#    main() now assumes the effective canvas center is the universal reference point.
#    generate_input() guarantees this is workable by choosing only anchor pairs whose
#    composites fit fully and by requiring at least one placement to differ from the
#    legacy edge/corner interpretation.


def _find_border6_square_start(legend: np.ndarray) -> tuple[int, int] | None:
    h, w = legend.shape
    if h < 5 or w < 5:
        return None
    for r0 in range(h - 4):
        for c0 in range(w - 4):
            sub = legend[r0 : r0 + 5, c0 : c0 + 5]
            if np.all(sub[0, :] == 6) and np.all(sub[4, :] == 6) and np.all(sub[:, 0] == 6) and np.all(sub[:, 4] == 6):
                return int(r0), int(c0)
    return None


def _find_marker_panel_square(g: np.ndarray, sep_c1: int) -> tuple[int, int] | None:
    """
    Find the 5x5 border-6 position-marker square that belongs to the 8x5 marker panel.

    In some instances the marker panel can appear inside the main canvas as well as in the
    left legend region. Prefer a candidate that lies in the main canvas (col > sep_c1) if any.
    """
    h, w = g.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(h - 7):
        for c0 in range(w - 4):
            sub = g[r0 : r0 + 5, c0 : c0 + 5]
            if not (np.all(sub[0, :] == 6) and np.all(sub[4, :] == 6) and np.all(sub[:, 0] == 6) and np.all(sub[:, 4] == 6)):
                continue
            # Must contain a position marker inside (non-{0,6}) and a rotation mark below.
            inner = sub[1:4, 1:4]
            if not np.any((inner != 0) & (inner != 6)):
                continue
            rot = g[r0 + 5 : r0 + 8, c0 + 1 : c0 + 4]
            if rot.shape != (3, 3) or not np.any((rot != 0) & (rot != 6)):
                continue
            candidates.append((r0, c0))
    if not candidates:
        return None
    main_cands = [rc for rc in candidates if rc[1] > sep_c1]
    return min(main_cands) if main_cands else min(candidates)


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
    # Prefer a right separator run very near the right edge (a right legend panel).
    right: tuple[int, int] | None = None
    for s, e in reversed(runs[1:]):
        if s >= w - 10 and s > left[1] + 1:
            right = (s, e)
            break
    return left, right


def _find_marker_square_with_preferences(
    g: np.ndarray, left_sep: tuple[int, int], right_sep: tuple[int, int] | None
) -> tuple[int, int] | None:
    """
    Find the marker panel's 5x5 border-6 square.

    Preference order:
      1) left legend region (at/before left separator)
      2) right legend region (at/after right separator), if present
      3) anywhere (including inside the output canvas)
    """
    h, w = g.shape
    candidates: list[tuple[int, int]] = []
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
            candidates.append((r0, c0))
    if not candidates:
        return None
    left_end = left_sep[1]
    left_cands = [rc for rc in candidates if rc[1] + 2 <= left_end]
    if left_cands:
        return min(left_cands)
    if right_sep is not None:
        right_start = right_sep[0]
        right_cands = [rc for rc in candidates if rc[1] + 2 >= right_start]
        if right_cands:
            return min(right_cands)
    return min(candidates)


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
    # choose left-most on that row to seed
    cseed = int(np.argwhere(mask[rmin, :])[0, 0])
    # BFS from seed to get component
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
    # Unit sprite uses two colors; the rendered sprite is the color-swapped version.
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
    # If multiple, take the first (should be unique)
    r, c = int(pts[0, 0]), int(pts[0, 1])
    return r, c  # 0..2


def _rotation_from_clockhand(legend: np.ndarray, border6_r0: int, border6_c0: int) -> int:
    """
    Rotation marker is in the bottom 3x3 at rows (border6_r0+5 .. border6_r0+7) and cols (1..3),
    drawn with non-{0,6} pixels (typically 7). Use the furthest pixel from the center (1,1)
    to determine direction:
      up -> 0, right -> 1, down -> 2, left -> 3.
    """
    h, w = legend.shape
    r0 = border6_r0 + 5
    c0 = border6_c0 + 1
    assert r0 + 2 < h and c0 + 2 < w, "rotation marker patch out of bounds"
    sub = legend[r0 : r0 + 3, c0 : c0 + 3]
    pts = np.argwhere((sub != 0) & (sub != 6))
    assert pts.size > 0, "no rotation marker pixel found"
    center = np.array([1, 1], dtype=int)
    # pick furthest (Manhattan) from center
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
    anchor_idx in {0,1,2} selects which point of the placed object is aligned to the
    fixed center reference point of the usable canvas length:
      0 -> object's start, 1 -> object's middle, 2 -> object's end
    """
    canvas_anchor = (size - 1) // 2 if size > 1 else 0
    if anchor_idx <= 0:
        obj_anchor = 0
    elif anchor_idx >= 2:
        obj_anchor = max(0, place_size - 1)
    else:
        obj_anchor = (place_size - 1) // 2 if place_size > 1 else 0
    return canvas_anchor - obj_anchor


def _legacy_anchor_to_top_left(anchor_idx: int, size: int, place_size: int) -> int:
    """Original task's anchor mapping, kept only to reject degenerate generator samples."""
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build enlarged composite sprites from legend panels and place them onto the canvas.

    Input structure:
    - Full-height color-6 separator columns split the grid into one or two legend panels
      and the canvas that will become the output.
    - Each legend panel contains a color-swapped unit sprite, a frame mask, a 5x5
      border-6 marker square with one inner pixel, and a 3x3 rotation clock-hand.

    Rule:
    - Recover the unit sprite, tile it into every active frame cell, and rotate the
      resulting composite according to the clock-hand direction.
    - Read the marker pixel's row and column inside the 3x3 inner square.
    - Use those indices to choose which vertical part and horizontal part of the
      composite must touch the fixed center reference point of the effective canvas.

    Key insight: The 3x3 marker no longer selects a canvas edge/corner. It selects
    the composite's own top/center/bottom and left/center/right anchor, which must be
    aligned to the canvas center point.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    # Output canvas is the region between the left separator run and (optionally) a
    # right-side separator run near the far edge (right legend panel).
    left_sep, right_sep = _split_canvas(g)
    left_sep_c1 = left_sep[1]
    canvas_c0 = left_sep_c1 + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    main = g[:, canvas_c0:canvas_c1].copy()

    # Paint one constructed sprite per marker panel that lies outside the output canvas.
    marker_squares = _all_marker_panel_squares(g)
    assert marker_squares, "no marker panel found"

    # Anchor alignment (ignore internal full-height/full-width 6 separators inside main).
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
        # Ignore marker panels inside the output canvas; only legends should drive drawing.
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
        frame_mask0 = (frame != 0) & (frame != 6)
        _, n8 = _scipy_label(frame_mask0, structure=np.ones((3, 3)))
        assert n8 == 1, f"frame must be 8-connected, got {n8} components"

        # Rotate layout according to clockhand.
        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)  # up=0, right=1, down=2, left=3
        frame_mask = np.rot90(frame_mask0, (-dir_k) % 4)

        # Build composite by tiling the unit sprite into active frame cells.
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

# --------------- generator (refine_8conn_fill_overlap) ---------------

# ── CFG: tunable generation parameters ──────────────────────────────
GRID_H_RANGE = (20, 30)         # output grid height
GRID_W = 30                     # output grid width (fixed)
LEGEND_COUNT_WEIGHTS = [1, 4]   # weights for [1-legend, 2-legend]
UNIT_SIZE_RANGE = (2, 5)        # NxN unit sprite side length
MIN_CANVAS_W = 8                # reject if canvas narrower than this
FRAME_H_RANGE = (4, 9)          # frame mask height (before rotation)
FRAME_W_RANGE = (3, 6)          # frame mask width  (before rotation)
FRAME_DENSITY = 0.5             # density for random_sprite frame generation
BAND_WIDTH_MIN = 4              # minimum band width for canvas background
MARKER_PANEL_HEIGHT = 8         # marker square (5) + rotation patch (3)
USABLE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]  # all colors except 0 (bg) and 6 (separator)
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up/right/down/left clock-hand offsets
GEN_RETRIES = 300               # max attempts in generate_input


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
    """Return a copy of sprite with the two colors swapped."""
    sa, sb = sorted([c1, c2])
    out = sprite.copy()
    out[sprite == sa] = sb
    out[sprite == sb] = sa
    return out


def _design_legend(leg, unit_colors, frame_color, marker_r0, grid_h, canvas_w):
    """Fill in sprite, frame, and rotation for one legend dict. Returns False on failure."""
    N = leg["N"]
    uc1, uc2 = unit_colors

    # Unit sprite: NxN, fully filled, exactly 2 colors
    sprite = np.full((N, N), uc1, dtype=int)
    num_c2 = random.randint(1, N * N - 1)
    for p in random.sample(range(N * N), num_c2):
        sprite[p // N, p % N] = uc2

    legend_sprite = _color_swap(sprite, uc1, uc2)

    avail = marker_r0 - (N + 1)
    # Enumerate valid (frame_H, frame_W, rot_k) combos
    combos = []
    for fH in range(*FRAME_H_RANGE):
        if avail < fH + 1:
            continue
        for fW in range(*FRAME_W_RANGE):
            for rk in range(4):
                rh, rw = (fH, fW) if rk % 2 == 0 else (fW, fH)
                if rh * N <= grid_h and rw * N <= canvas_w:
                    combos.append((fH, fW, rk))
    if not combos:
        return False
    frame_H, frame_W, rot_k = random.choice(combos)

    frame_mask = random_sprite(
        frame_H, frame_W, density=FRAME_DENSITY, connectivity=8,
        color_palette=[1], symmetry="not_symmetric"
    ) != 0
    rot_frame = np.rot90(frame_mask, (-rot_k) % 4)
    comp = _build_composite(sprite, rot_frame)

    leg.update(sprite=sprite, legend_sprite=legend_sprite, frame_mask=frame_mask,
               frame_H=frame_H, frame_W=frame_W, frame_color=frame_color,
               rot_k=rot_k, rotated_frame=rot_frame, composite=comp, composite_mask=(comp != 0))
    return True


def _build_composite(sprite, rotated_frame):
    """Tile the unit sprite into every active cell of the rotated frame."""
    N = sprite.shape[0]
    fh, fw = rotated_frame.shape
    comp = np.zeros((fh * N, fw * N), dtype=int)
    sprite_mask = (sprite != 0) & (sprite != 6)
    for i in range(fh):
        for j in range(fw):
            if not rotated_frame[i, j]:
                continue
            r0, c0 = i * N, j * N
            comp[r0:r0 + N, c0:c0 + N][sprite_mask] = sprite[sprite_mask]
    return comp


def _comp_size(leg):
    """Return (height, width) of the composite after rotation."""
    return leg["composite"].shape


def _valid_anchor_placements(leg, grid_h, canvas_w):
    """Enumerate all in-bounds placements for one legend under the new anchor semantics."""
    ch, cw = _comp_size(leg)
    placements = []
    for pr in range(3):
        for pc in range(3):
            top = _anchor_to_top_left(pr, grid_h, ch)
            left = _anchor_to_top_left(pc, canvas_w, cw)
            if top < 0 or left < 0 or top + ch > grid_h or left + cw > canvas_w:
                continue
            legacy_top = _legacy_anchor_to_top_left(pr, grid_h, ch)
            legacy_left = _legacy_anchor_to_top_left(pc, canvas_w, cw)
            placements.append(
                {
                    "pr": pr,
                    "pc": pc,
                    "top": top,
                    "left": left,
                    "differs_from_legacy": (top != legacy_top) or (left != legacy_left),
                }
            )
    return placements


def _pixel_overlap(leg_a, place_a, leg_b, place_b):
    """True iff two placed composites share any nonzero pixel."""
    a_mask = leg_a["composite_mask"]
    b_mask = leg_b["composite_mask"]
    a_h, a_w = a_mask.shape
    b_h, b_w = b_mask.shape
    a_top, a_left = place_a["top"], place_a["left"]
    b_top, b_left = place_b["top"], place_b["left"]
    inter_top = max(a_top, b_top)
    inter_left = max(a_left, b_left)
    inter_bottom = min(a_top + a_h, b_top + b_h)
    inter_right = min(a_left + a_w, b_left + b_w)
    if inter_top >= inter_bottom or inter_left >= inter_right:
        return False
    a_view = a_mask[inter_top - a_top:inter_bottom - a_top, inter_left - a_left:inter_right - a_left]
    b_view = b_mask[inter_top - b_top:inter_bottom - b_top, inter_left - b_left:inter_right - b_left]
    return bool(np.any(a_view & b_view))


def _assign_anchors(legends, grid_h, canvas_w):
    """Assign D16 anchors, requiring at least one legend to differ from the legacy semantics."""
    if len(legends) == 1:
        placements = [p for p in _valid_anchor_placements(legends[0], grid_h, canvas_w) if p["differs_from_legacy"]]
        if not placements:
            return False
        pick = random.choice(placements)
        legends[0].update(pr=pick["pr"], pc=pick["pc"], comp_top=pick["top"], comp_left=pick["left"])
        return True

    a, b = legends
    a_placements = _valid_anchor_placements(a, grid_h, canvas_w)
    b_placements = _valid_anchor_placements(b, grid_h, canvas_w)
    if not a_placements or not b_placements:
        return False
    valid_pairs = []
    for ap in a_placements:
        for bp in b_placements:
            if not (ap["differs_from_legacy"] or bp["differs_from_legacy"]):
                continue
            if _pixel_overlap(a, ap, b, bp):
                continue
            valid_pairs.append((ap, bp))
    if not valid_pairs:
        return False
    ap, bp = random.choice(valid_pairs)
    a.update(pr=ap["pr"], pc=ap["pc"], comp_top=ap["top"], comp_left=ap["left"])
    b.update(pr=bp["pr"], pc=bp["pc"], comp_top=bp["top"], comp_left=bp["left"])
    return True


def _build_banded_canvas(grid_h, canvas_w, bg1, bg2):
    """Build a canvas with alternating horizontal or vertical color bands."""
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
    """Stamp each legend's tiled composite onto the canvas (non-{0,6} pixels overwrite)."""
    output = canvas.copy()
    for leg in legends:
        t, l = leg["comp_top"], leg["comp_left"]
        comp = leg["composite"]
        ch, cw = comp.shape

        # Overwrite canvas where composite has content and canvas isn't color 6
        region = output[t:t + ch, l:l + cw]
        write_mask = (comp != 0) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]

    # Per legend: pick N, compute legend_w
    legends = []
    for _ in range(num_legends):
        N = random.randint(*UNIT_SIZE_RANGE)
        legends.append({"N": N, "legend_w": N + 2})

    # Layout widths
    left_lw = legends[0]["legend_w"]
    left_total = left_lw + 1  # legend + 1 separator col
    right_total = (1 + legends[1]["legend_w"]) if num_legends == 2 else 0
    canvas_w = GRID_W - left_total - right_total
    if canvas_w < MIN_CANVAS_W:
        return None

    # Color allocation
    colors = _allocate_colors(num_legends)
    if colors is None:
        return None
    unit_color_sets, (bg1, bg2), frame_colors = colors

    marker_r0 = grid_h - MARKER_PANEL_HEIGHT

    # Design each legend panel
    for li in range(num_legends):
        if not _design_legend(legends[li], unit_color_sets[li], frame_colors[li],
                              marker_r0, grid_h, canvas_w):
            return None

    # Assign anchor positions
    if not _assign_anchors(legends, grid_h, canvas_w):
        return None

    # Build canvas and overlay composites
    canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2)
    output = _overlay_composites(canvas, legends)

    # Assemble input grid
    grid = np.zeros((grid_h, GRID_W), dtype=int)
    canvas_c0 = left_total
    grid[:, canvas_c0:canvas_c0 + canvas_w] = canvas

    _place_legend(grid, legends[0], col_start=0, sep_col=left_lw,
                  marker_r0=marker_r0, is_right=False)
    if num_legends == 2:
        r_sep = canvas_c0 + canvas_w
        _place_legend(grid, legends[1], col_start=r_sep + 1, sep_col=r_sep,
                      marker_r0=marker_r0, is_right=True)

    # Validate via solver
    try:
        if np.array_equal(main(grid), output):
            return grid
    except Exception:
        pass
    return None


def _place_legend(grid, leg, col_start, sep_col, marker_r0, is_right):
    """Draw one legend panel (separator, marker square, rotation patch, sprite, frame)."""
    N = leg["N"]
    grid[:, sep_col] = 6

    # Marker square column: sprite at col_start+1 must fit within the 5-wide panel
    mc0 = (col_start + max(0, N - 4)) if not is_right else (sep_col + max(0, N - 3))

    # 5x5 border-6 marker square
    grid[marker_r0:marker_r0 + 5, mc0:mc0 + 5] = 0
    grid[marker_r0, mc0:mc0 + 5] = 6
    grid[marker_r0 + 4, mc0:mc0 + 5] = 6
    grid[marker_r0:marker_r0 + 5, mc0] = 6
    grid[marker_r0:marker_r0 + 5, mc0 + 4] = 6
    grid[marker_r0 + 1 + leg["pr"], mc0 + 1 + leg["pc"]] = 4

    # Rotation patch: 3x5 block with 6-borders on cols 0,4 and 3x3 clock-hand inside
    rr0 = marker_r0 + 5
    grid[rr0:rr0 + 3, mc0:mc0 + 5] = 0
    grid[rr0:rr0 + 3, mc0] = 6
    grid[rr0:rr0 + 3, mc0 + 4] = 6
    cx_r, cx_c = rr0 + 1, mc0 + 2
    dr, dc = ROTATION_DIRS[leg["rot_k"]]
    grid[cx_r, cx_c] = 7
    grid[cx_r + dr, cx_c + dc] = 7

    # Unit sprite at row 1
    sp_col = col_start + 1 - mc0
    grid[1:1 + N, mc0 + sp_col:mc0 + sp_col + N] = leg["legend_sprite"]

    # Frame mask between unit sprite bottom and marker top
    fH, fW = leg["frame_H"], leg["frame_W"]
    avail = marker_r0 - (N + 1)
    gap = random.randint(1, max(1, avail - fH))
    fr_start = N + 1 + gap
    f_col = random.randint(0, max(0, 4 - fW))
    frame_region = grid[fr_start:fr_start + fH, mc0 + f_col:mc0 + f_col + fW]
    frame_region[leg["frame_mask"]] = leg["frame_color"]


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
