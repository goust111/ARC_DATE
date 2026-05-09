# Combo variant of a32d8b75 (index 22)
# combo_spec: D5/2+D7/2
# Name: 双旋转+数量编码
#
# Design analysis for the fused variant
# a) Concrete behavior and trace:
#    The original task had one 3x3 clock-hand, and the whole composite was rotated once after
#    tiling. This fusion changes two steps together:
#      original: read one direction k -> tile(sprite, frame) -> rotate whole composite by k
#      fused   : count color-7 pixels -> frame_k, count color-8 pixels -> sprite_k
#                -> rotate(frame, frame_k), rotate(sprite, sprite_k)
#                -> tile(rotated sprite into rotated frame)
#    Small example:
#      - If color 7 occupies 3 cells, frame_k = 2 (180 degrees).
#      - If color 8 occupies 1 cell, sprite_k = 0 (0 degrees).
#      - The large frame turns upside-down, but the repeated unit motif keeps its original
#        orientation. That cannot be explained by any single post-hoc whole-object rotation.
#    Because both count codes must live inside one shared 3x3 patch, the center cell is left as
#    a neutral hub and each color grows along its own non-overlapping outward path. The rule still
#    reads as “two hands start from the center”, but now count matters and direction does not.
# b) Complexity calibration:
#    We keep exactly four values per rotation (1..4 pixels -> 0/90/180/270). This preserves the
#    original difficulty level while making the change visible. The two colors are fixed and the
#    paths are disjoint, so decoding is not ambiguous. Direction is randomized by rotating/flipping
#    the whole patch, preventing accidental reuse of the original directional heuristic.
# c) Hidden assumptions review:
#    The old solver assumed one rotation was enough and that the furthest marker pixel carried the
#    meaning. Both assumptions are invalid now. The new solver instead assumes:
#      - the patch contains only colors 7 and 8 besides 0/6,
#      - each of those colors appears 1..4 times,
#      - each color forms one 4-connected arm touching the shared center hub,
#      - frame and sprite may require different rotations.
#    The generator explicitly enforces all four assumptions.
# d) Placement legality:
#    Sprite rotation does not change the unit size because the unit remains NxN, so composite size
#    still depends only on the rotated frame dimensions. The generator keeps the original
#    output-first fit checks and rejects 2-legend overlap cases.
# e) Visual observability:
#    The generator forces both the unit sprite and frame mask to be rotation-asymmetric, and it
#    rejects any sample whose fused output matches a legacy whole-rotation interpretation. Human
#    solvers can therefore see that changing the small motif and changing the large frame are two
#    separate controls. Patch orientation is also decorrelated from the encoded values, making the
#    count-based rule inferable from a few examples.
# f) Visual cleanliness:
#    The marker panel footprint stays 8x5. The top 5x5 square still handles anchor placement, and
#    the bottom 3x3 patch is reserved for the dual count code only. The two colored arms occupy
#    disjoint halves of the ring around the center, so the legend remains compact rather than busy.
# g) Change inferability:
#    No extra color remapping is introduced beyond the original sprite color swap. Colors 7 and 8
#    are reserved for the two rotation counts, so examples consistently teach “orange count controls
#    frame” and “teal count controls sprite” without relying on hidden numeric properties.

import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4, random_sprite


FRAME_ROT_MARKER_COLOR = 7
SPRITE_ROT_MARKER_COLOR = 8
POSITION_MARKER_COLOR = 4
FOUR_CONNECTED = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
CENTER_NEIGHBORS = [(0, 1), (1, 0), (1, 2), (2, 1)]

# Two 4-connected paths sharing the same conceptual hub at the center.
# They partition the 8 non-center cells, so both count codes fit in one 3x3 patch.
FRAME_COUNT_PATH = [(1, 2), (0, 2), (0, 1), (0, 0)]
SPRITE_COUNT_PATH = [(1, 0), (2, 0), (2, 1), (2, 2)]


def _separator_runs(g: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous runs of full-height columns that are all color 6."""
    _h, w = g.shape
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
    _h, w = g.shape
    runs = _separator_runs(g)
    assert runs, "must have at least one color-6 separator run"
    left = runs[0]
    right: tuple[int, int] | None = None
    for s, e in reversed(runs[1:]):
        if s >= w - 10 and s > left[1] + 1:
            right = (s, e)
            break
    return left, right


def _count_rotation_from_color(sub: np.ndarray, color: int) -> int:
    mask = sub == color
    count = int(mask.sum())
    assert 1 <= count <= 4, f"color {color} count must be 1..4, got {count}"
    _, n4 = _scipy_label(mask, structure=FOUR_CONNECTED)
    assert n4 == 1, f"color {color} marker must be a single 4-connected arm"
    assert any(mask[r, c] for r, c in CENTER_NEIGHBORS), f"color {color} arm must touch center hub"
    return count - 1


def _rotation_patch_looks_valid(g: np.ndarray, r0: int, c0: int) -> bool:
    sub = g[r0 + 5 : r0 + 8, c0 + 1 : c0 + 4]
    if sub.shape != (3, 3):
        return False
    if int(sub[1, 1]) != 0:
        return False
    allowed = {0, FRAME_ROT_MARKER_COLOR, SPRITE_ROT_MARKER_COLOR}
    if any(int(x) not in allowed for x in np.unique(sub)):
        return False
    try:
        _count_rotation_from_color(sub, FRAME_ROT_MARKER_COLOR)
        _count_rotation_from_color(sub, SPRITE_ROT_MARKER_COLOR)
    except AssertionError:
        return False
    return True


def _all_marker_panel_squares(g: np.ndarray) -> list[tuple[int, int]]:
    """List all 5x5 border-6 squares that look like the top of a marker panel."""
    h, w = g.shape
    out: list[tuple[int, int]] = []
    for r0 in range(h - 7):
        for c0 in range(w - 4):
            sub = g[r0 : r0 + 5, c0 : c0 + 5]
            if not (
                np.all(sub[0, :] == 6)
                and np.all(sub[4, :] == 6)
                and np.all(sub[:, 0] == 6)
                and np.all(sub[:, 4] == 6)
            ):
                continue
            inner = sub[1:4, 1:4]
            if np.count_nonzero((inner != 0) & (inner != 6)) != 1:
                continue
            if not _rotation_patch_looks_valid(g, r0, c0):
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


def _extract_frame_bbox(
    legend: np.ndarray, unit_bb: tuple[int, int, int, int], border6_r0: int
) -> tuple[int, int, int, int] | None:
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


def _position_indices_from_border_square(
    legend: np.ndarray, border6_r0: int, border6_c0: int
) -> tuple[int, int] | None:
    sq = legend[border6_r0 : border6_r0 + 5, border6_c0 : border6_c0 + 5]
    inner = sq[1:4, 1:4]
    pts = np.argwhere((inner != 0) & (inner != 6))
    if pts.size == 0:
        return None
    r, c = int(pts[0, 0]), int(pts[0, 1])
    return r, c


def _decode_dual_count_rotations(g: np.ndarray, border6_r0: int, border6_c0: int) -> tuple[int, int]:
    """
    Decode two independent rotations from the 3x3 patch below the marker square.

    The patch uses a shared empty center hub. Color 7 and color 8 each occupy 1..4
    pixels along their own arm, and those counts map to 0/90/180/270 degrees.
    """
    h, w = g.shape
    r0 = border6_r0 + 5
    c0 = border6_c0 + 1
    assert r0 + 2 < h and c0 + 2 < w, "rotation marker patch out of bounds"
    sub = g[r0 : r0 + 3, c0 : c0 + 3]
    assert int(sub[1, 1]) == 0, "rotation patch must keep the center hub empty"
    frame_k = _count_rotation_from_color(sub, FRAME_ROT_MARKER_COLOR)
    sprite_k = _count_rotation_from_color(sub, SPRITE_ROT_MARKER_COLOR)
    return frame_k, sprite_k


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


def _build_composite(unit_sprite: np.ndarray, frame_mask0: np.ndarray, frame_k: int, sprite_k: int) -> np.ndarray:
    frame_mask = np.rot90(frame_mask0, (-frame_k) % 4)
    sprite = np.rot90(unit_sprite, (-sprite_k) % 4)
    m_unit = (sprite != 0) & (sprite != 6)
    uh, uw = sprite.shape
    fh, fw = frame_mask.shape
    comp = np.zeros((fh * uh, fw * uw), dtype=int)
    for i in range(fh):
        for j in range(fw):
            if not frame_mask[i, j]:
                continue
            rr0 = i * uh
            cc0 = j * uw
            comp[rr0 : rr0 + uh, cc0 : cc0 + uw][m_unit] = sprite[m_unit]
    return comp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode dual-count legend panels and stamp the assembled composites onto the canvas.

    Input structure:
    - Full-height color-6 separators split the grid into one or two legend panels and a main canvas.
    - Each legend panel still contains a unit sprite, a frame template, a 5x5 anchor square, and a 3x3 rotation patch.
    - In that lower patch, color 7 and color 8 each occupy 1..4 pixels along separate hands around a shared center hub; the counts, not the directions, encode two rotations.

    Rule:
    - Read the unit sprite from the legend, including the original color-swap behavior, and read the frame mask above the marker square.
    - Count the color-7 pixels to rotate the frame, count the color-8 pixels to rotate the unit sprite, then tile the rotated sprite into every active cell of the rotated frame.
    - Place the resulting composite on the canvas using the anchor encoded by the marker pixel inside the 5x5 square.

    Key insight:
    - The bottom patch no longer describes one global turn. It carries two independent quantity codes, so the solver must ignore hand direction entirely and decode the large-scale frame orientation separately from the small repeated tile orientation.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    main_canvas = g[:, canvas_c0:canvas_c1].copy()

    marker_squares = _all_marker_panel_squares(g)
    assert marker_squares, "no marker panel found"

    eff_h, eff_w = main_canvas.shape
    full6_cols = [c for c in range(eff_w) if np.all(main_canvas[:, c] == 6)]
    if full6_cols:
        eff_w = min(full6_cols)
    full6_rows = [r for r in range(eff_h) if np.all(main_canvas[r, :] == 6)]
    if full6_rows:
        eff_h = min(full6_rows)

    out = main_canvas.copy()
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
        frame_mask0 = (frame != 0) & (frame != 6)
        _, n8 = _scipy_label(frame_mask0, structure=np.ones((3, 3), dtype=int))
        assert n8 == 1, f"frame must be 8-connected, got {n8} components"

        frame_k, sprite_k = _decode_dual_count_rotations(g, marker_r0, marker_c0)
        comp = _build_composite(unit_sprite, frame_mask0, frame_k, sprite_k)

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


# --------------- generator (dual split-rotation count code) ---------------

GRID_H_RANGE = (20, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [1, 4]
UNIT_SIZE_RANGE = (2, 5)
TWO_LEGEND_GRID_H_MIN = 24
TWO_LEGEND_UNIT_SIZE_RANGE = (2, 4)
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)
FRAME_W_RANGE = (3, 6)
FRAME_DENSITY = 0.5
BAND_WIDTH_MIN = 4
MARKER_PANEL_HEIGHT = 8
CORE_COLORS = [1, 2, 3, 4, 5, 9]
GEN_RETRIES = 400
PRIMARY_BRANCH_RETRIES = 300
_NEXT_LEGEND_COUNT = 1


def _allocate_colors(num_legends: int):
    """Pick unit colors (2 per legend), 2 background colors, and 1 frame color per legend."""
    pool = list(CORE_COLORS)
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
    all_set = set(CORE_COLORS)
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


def _all_rotations_distinct(arr: np.ndarray) -> bool:
    rots = [np.rot90(arr, k) for k in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = rots[i], rots[j]
            if a.shape == b.shape and np.array_equal(a, b):
                return False
    return True


def _make_asymmetric_unit_sprite(n: int, c1: int, c2: int) -> np.ndarray | None:
    for _ in range(200):
        sprite = np.full((n, n), c1, dtype=int)
        num_c2 = random.randint(1, n * n - 1)
        for p in random.sample(range(n * n), num_c2):
            sprite[p // n, p % n] = c2
        if _all_rotations_distinct(sprite):
            return sprite
    return None


def _make_asymmetric_frame_mask(h: int, w: int) -> np.ndarray | None:
    for _ in range(200):
        frame_mask = random_sprite(
            h,
            w,
            density=FRAME_DENSITY,
            connectivity=8,
            color_palette=[1],
            symmetry="not_symmetric",
        ) != 0
        _, n8 = _scipy_label(frame_mask, structure=np.ones((3, 3), dtype=int))
        if n8 != 1:
            continue
        if _all_rotations_distinct(frame_mask):
            return frame_mask
    return None


def _make_dual_rotation_patch(frame_k: int, sprite_k: int) -> np.ndarray:
    """
    Build a 3x3 patch with two count-coded arms around an empty center hub.

    The whole patch is randomly rotated/flipped so direction carries no information.
    """
    patch = np.zeros((3, 3), dtype=int)
    for rr, cc in FRAME_COUNT_PATH[: frame_k + 1]:
        patch[rr, cc] = FRAME_ROT_MARKER_COLOR
    for rr, cc in SPRITE_COUNT_PATH[: sprite_k + 1]:
        patch[rr, cc] = SPRITE_ROT_MARKER_COLOR
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    patch = np.rot90(patch, random.randint(0, 3))
    return patch


def _design_legend(
    leg: dict,
    unit_colors: tuple[int, int],
    frame_color: int,
    marker_r0: int,
    grid_h: int,
    canvas_w: int,
    compact_mode: bool = False,
) -> bool:
    """Fill in sprite, frame, and fused dual count rotations for one legend dict."""
    n = leg["N"]
    uc1, uc2 = unit_colors
    sprite = _make_asymmetric_unit_sprite(n, uc1, uc2)
    if sprite is None:
        return False
    legend_sprite = _color_swap(sprite, uc1, uc2)

    avail = marker_r0 - (n + 1)
    combos = []
    for f_h in range(*FRAME_H_RANGE):
        if avail < f_h + 1:
            continue
        for f_w in range(*FRAME_W_RANGE):
            for frame_k in range(4):
                rh, rw = (f_h, f_w) if frame_k % 2 == 0 else (f_w, f_h)
                if rh * n > grid_h or rw * n > canvas_w:
                    continue
                for sprite_k in range(4):
                    if sprite_k == frame_k:
                        continue
                    combos.append((f_h, f_w, frame_k, sprite_k))
    if not combos:
        return False

    if compact_mode:
        combos.sort(
            key=lambda item: (
                ((item[0] if item[2] % 2 == 0 else item[1]) * (item[1] if item[2] % 2 == 0 else item[0])),
                abs(item[2] - item[3]),
            )
        )
        combos = combos[: min(len(combos), 24)]
    random.shuffle(combos)
    for frame_h, frame_w, frame_k, sprite_k in combos:
        frame_mask = _make_asymmetric_frame_mask(frame_h, frame_w)
        if frame_mask is None:
            continue
        comp = _build_composite(sprite, frame_mask, frame_k, sprite_k)
        legacy_same = False
        for k in range(4):
            if np.array_equal(comp, _build_composite(sprite, frame_mask, k, k)):
                legacy_same = True
                break
        if legacy_same:
            continue
        leg.update(
            sprite=sprite,
            legend_sprite=legend_sprite,
            frame_mask=frame_mask,
            frame_H=frame_h,
            frame_W=frame_w,
            frame_color=frame_color,
            frame_rot_k=frame_k,
            sprite_rot_k=sprite_k,
            rotated_frame=np.rot90(frame_mask, (-frame_k) % 4),
        )
        return True
    return False


def _comp_size(leg: dict) -> tuple[int, int]:
    """Return (height, width) of the composite after the frame rotation."""
    rf = leg["rotated_frame"]
    n = leg["N"]
    return rf.shape[0] * n, rf.shape[1] * n


def _assign_anchors(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
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


def _build_banded_canvas(grid_h: int, canvas_w: int, bg1: int, bg2: int) -> np.ndarray:
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


def _overlay_composites(canvas: np.ndarray, legends: list[dict]) -> np.ndarray:
    """Stamp each legend's tiled composite onto the canvas (non-{0,6} pixels overwrite)."""
    output = canvas.copy()
    for leg in legends:
        comp = _build_composite(leg["sprite"], leg["frame_mask"], leg["frame_rot_k"], leg["sprite_rot_k"])
        t, l = leg["comp_top"], leg["comp_left"]
        ch, cw = comp.shape
        region = output[t : t + ch, l : l + cw]
        write_mask = (comp != 0) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


def _place_legend(
    grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int, is_right: bool
) -> None:
    """Draw one legend panel (separator, marker square, dual-count patch, sprite, frame)."""
    n = leg["N"]
    grid[:, sep_col] = 6

    panel_w = 5
    legend_w = leg["legend_w"]
    assert legend_w >= panel_w, "legend must reserve at least a 5-column panel"
    mc0 = (sep_col - panel_w) if not is_right else (col_start + legend_w - panel_w)

    grid[marker_r0 : marker_r0 + 5, mc0 : mc0 + 5] = 0
    grid[marker_r0, mc0 : mc0 + 5] = 6
    grid[marker_r0 + 4, mc0 : mc0 + 5] = 6
    grid[marker_r0 : marker_r0 + 5, mc0] = 6
    grid[marker_r0 : marker_r0 + 5, mc0 + 4] = 6
    grid[marker_r0 + 1 + leg["pr"], mc0 + 1 + leg["pc"]] = POSITION_MARKER_COLOR

    rr0 = marker_r0 + 5
    grid[rr0 : rr0 + 3, mc0 : mc0 + 5] = 0
    grid[rr0 : rr0 + 3, mc0] = 6
    grid[rr0 : rr0 + 3, mc0 + 4] = 6
    grid[rr0 : rr0 + 3, mc0 + 1 : mc0 + 4] = _make_dual_rotation_patch(leg["frame_rot_k"], leg["sprite_rot_k"])

    sp_col = random.randint(0, panel_w - n)
    grid[1 : 1 + n, mc0 + sp_col : mc0 + sp_col + n] = leg["legend_sprite"]

    f_h, f_w = leg["frame_H"], leg["frame_W"]
    avail = marker_r0 - (n + 1)
    gap = random.randint(1, max(1, avail - f_h))
    fr_start = n + 1 + gap
    f_col = random.randint(0, panel_w - f_w)
    frame_region = grid[fr_start : fr_start + f_h, mc0 + f_col : mc0 + f_col + f_w]
    frame_region[leg["frame_mask"]] = leg["frame_color"]


def _gen_attempt(num_legends: int | None = None) -> np.ndarray | None:
    if num_legends is None:
        num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]

    if num_legends == 2:
        grid_h = random.randint(max(GRID_H_RANGE[0], TWO_LEGEND_GRID_H_MIN), GRID_H_RANGE[1])
        unit_size_range = TWO_LEGEND_UNIT_SIZE_RANGE
    else:
        grid_h = random.randint(*GRID_H_RANGE)
        unit_size_range = UNIT_SIZE_RANGE

    legends = []
    for _ in range(num_legends):
        n = random.randint(*unit_size_range)
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

    marker_r0 = grid_h - MARKER_PANEL_HEIGHT
    for li in range(num_legends):
        if not _design_legend(
            legends[li],
            unit_color_sets[li],
            frame_colors[li],
            marker_r0,
            grid_h,
            canvas_w,
            compact_mode=(num_legends == 2),
        ):
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
    global _NEXT_LEGEND_COUNT
    preferred = _NEXT_LEGEND_COUNT
    _NEXT_LEGEND_COUNT = 1 if preferred == 2 else 2
    fallback = 1 if preferred == 2 else 2
    for target, budget in (
        (preferred, PRIMARY_BRANCH_RETRIES),
        (fallback, GEN_RETRIES - PRIMARY_BRANCH_RETRIES),
    ):
        for _ in range(budget):
            result = _gen_attempt(target)
            if result is not None:
                return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
