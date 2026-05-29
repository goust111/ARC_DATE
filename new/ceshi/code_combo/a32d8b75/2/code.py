# Combo variant of a32d8b75 (index 2)
# combo_spec: D1/2+D3/0
# Name: 计数匹配+嵌套层次框架

import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4


# Design analysis for the fused D1/2 + D3/0 variant
# a) Concrete behavior:
# - The legend is now explicitly recursive: a unit sprite sits above two separate frame components,
#   where the top frame is the outer template and the lower frame is the inner template.
# - Both frame components are holey bordered masks. Their colored cells still encode placement sites,
#   but their enclosed zero holes also matter. Example:
#   - Original task: one flat frame cell directly receives one unit sprite block.
#   - D3 alone: one outer-frame cell receives one whole inner-frame composite.
#   - Fused rule: first count holes in the inner frame. If the inner frame has k holes, enlarge the
#     unit sprite into a k x k repeated seed, then stamp that seed into every active inner-frame cell.
#     Next count holes in the outer frame. If the outer frame has m holes, enlarge the entire inner
#     composite into an m x m repeated macro-seed, then stamp that macro-seed into every active
#     outer-frame cell. Rotation and anchoring apply after the recursive construction.
# - A concrete trace is easy to visualize with the generator's discrete templates:
#   - Inner 1-hole frame = 3x3 ring, so the unit sprite is used once per inner cell.
#   - Outer 2-hole frame = 3x5 double-window ring, so each active outer cell receives a 2x2 block
#     of the inner composite. The output therefore becomes much larger than either the original task
#     or the plain nested-frame variant.
# b) Complexity calibration:
# - Hole counts are limited to {1, 2}. A 2-hole template is still visually obvious, but keeping the
#   count set this small avoids combinatorial explosion in composite size.
# - The generator forbids the (outer=2, inner=2) case. That combination is logically valid but would
#   dominate the canvas and reduce sample diversity. It also forbids the visually trivial (1,1) case
#   so every sample visibly uses the counting dimension rather than collapsing to plain nesting.
# - Two simultaneous counted recursive composites are intentionally avoided in generation: on this
#   canvas budget they either overlap or force a trivial pair. Keeping one legend per sample preserves
#   the fused rule in every training example and keeps branch coverage honest.
# - The frame library is discrete and repeated across samples: 1-hole means a 3x3 ring; 2-hole means
#   a 3x5 or 5x3 double-window ring. This gives human solvers repeated evidence for the hole-count rule.
# c) Hidden assumptions from the original solver that must change:
# - `_extract_frame_bbox(...)` assumed a single flat frame. That is false now; the solver must extract
#   exactly two vertically ordered 8-connected frame components.
# - The original composite builder assumed one expansion scale `frame cell -> unit sprite`. That is now
#   a two-stage recursion: `inner frame cell -> repeated unit block`, then `outer frame cell -> repeated
#   inner composite`.
# - Zeros inside the frame bbox can no longer be treated as irrelevant background. They are the counted
#   holes, so the solver must rely on solid frame borders and 4-connected hole counting.
# - Rotation can no longer act on a single frame mask before assembly. It is applied to the fully built
#   recursive composite so that both count-controlled expansions participate in the visible orientation.
# d) Placement legality:
# - The generator computes the fully assembled composite for each legend before anchor assignment.
# - Bounding boxes must fit entirely inside the canvas after rotation. The solver still supports
#   multiple legends, but the generator emits one counted legend per sample so the fusion does not
#   collapse into a forced trivial branch.
# - The legend construction guarantees exactly one sprite component above exactly two frame components,
#   with at least one zero row separating all vertical blocks.
# e) Visual observability:
# - Humans can see both signals: the number of holes in each frame is small and countable, and the
#   resulting output size changes sharply when one level uses the 2-hole template.
# - Because the output is a nested composite, a solver cannot explain it with a single flat frame; and
#   because holes alter the scale, a solver cannot explain it with plain nesting alone.
# f) Visual neatness:
# - The frame section uses only two clean ring-like motifs, making the holes easy to isolate by eye.
# - Backgrounds remain simple 2-color bands, and overlays never intersect, so the fractal structure is
#   not obscured by clutter.
# g) Color inference:
# - No rule depends on color ordering. Colors merely separate roles: separator=6, anchor marker=4,
#   rotation marker=7, frame color distinct from unit colors, and background colors distinct from unit colors.
# h) Degeneration checks used after implementation:
# - Samples are later measured against a "nested but no counting" ablation. If the fused generator is
#   healthy, every sample should differ from that ablation because at least one recursive level is enlarged
#   by a 2-hole frame.
# - The hole-pair distribution is also measured; if (1,1) were produced, the fusion would collapse to a
#   plain nested-frame task.
# i) New solver assumptions guaranteed by `generate_input()`:
# - Exactly two 8-connected frame components appear between sprite and marker, ordered outer then inner.
# - Every frame component has a solid border and either 1 or 2 enclosed 4-connected holes.
# - Hole-count pairs belong to {(2,1), (1,2)} so the count effect is always visible.
# - The fully assembled composite fits inside the canvas at the selected anchor.


STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
STRUCT8 = np.ones((3, 3), dtype=int)

FRAME_LIBRARY: dict[int, list[np.ndarray]] = {
    1: [
        np.array(
            [
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
            ],
            dtype=bool,
        )
    ],
    2: [
        np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
            ],
            dtype=bool,
        ),
    ],
}


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
            if not (
                np.all(sub[0, :] == 6)
                and np.all(sub[4, :] == 6)
                and np.all(sub[:, 0] == 6)
                and np.all(sub[:, 4] == 6)
            ):
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
    """Read the top-most 4-connected non-{0,6} component and color-unswap it."""
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


def _extract_nested_frames(
    legend: np.ndarray, unit_bb: tuple[int, int, int, int], marker_r0: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Read two vertically ordered 8-connected frame components: outer first, inner second."""
    r_start = unit_bb[2] + 1
    r_end = marker_r0 - 1
    if r_start > r_end:
        return None
    sub = legend[r_start : r_end + 1, :]
    mask = (sub != 0) & (sub != 6)
    labeled, n8 = _scipy_label(mask, structure=STRUCT8)
    if n8 != 2:
        return None

    comps: list[tuple[int, int, int, int]] = []
    for idx in range(1, n8 + 1):
        pts = np.argwhere(labeled == idx)
        pts[:, 0] += r_start
        r0 = int(pts[:, 0].min())
        r1 = int(pts[:, 0].max())
        c0 = int(pts[:, 1].min())
        c1 = int(pts[:, 1].max())
        comps.append((r0, c0, r1, c1))
    comps.sort(key=lambda bb: (bb[0], bb[1], bb[2], bb[3]))

    def _crop_mask(bb: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = bb
        piece = legend[r0 : r1 + 1, c0 : c1 + 1]
        out = (piece != 0) & (piece != 6)
        _, piece_n8 = _scipy_label(out, structure=STRUCT8)
        assert piece_n8 == 1, "each frame component must be 8-connected"
        return out

    outer_bb, inner_bb = comps
    return _crop_mask(outer_bb), _crop_mask(inner_bb)


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


def _rotation_from_clockhand(legend: np.ndarray, border6_r0: int, border6_c0: int) -> int:
    """
    Read the 3x3 clock-hand marker below the 5x5 anchor square.

    Use the furthest non-{0,6} pixel from the center to determine direction:
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


def _tile_object(mask: np.ndarray, obj: np.ndarray) -> np.ndarray:
    """Tile an object into every active cell of a boolean mask."""
    block_mask = obj != 0
    oh, ow = obj.shape
    mh, mw = mask.shape
    out = np.zeros((mh * oh, mw * ow), dtype=int)
    for i in range(mh):
        for j in range(mw):
            if not mask[i, j]:
                continue
            rr0 = i * oh
            cc0 = j * ow
            out[rr0 : rr0 + oh, cc0 : cc0 + ow][block_mask] = obj[block_mask]
    return out


def _build_fused_composite(
    unit_sprite: np.ndarray,
    outer_mask: np.ndarray,
    inner_mask: np.ndarray,
    outer_holes: int,
    inner_holes: int,
    dir_k: int,
) -> np.ndarray:
    """
    Build a counted recursive composite:
      unit -> repeated by inner-hole count -> stamped into inner frame
           -> repeated by outer-hole count -> stamped into outer frame -> rotated.
    """
    unit_block = np.tile(unit_sprite, (inner_holes, inner_holes))
    inner_comp = _tile_object(inner_mask, unit_block)
    outer_block = np.tile(inner_comp, (outer_holes, outer_holes))
    comp0 = _tile_object(outer_mask, outer_block)
    return np.rot90(comp0, (-dir_k) % 4)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode recursive counted frame legends and place the resulting composites onto the canvas.

    Input structure:
    - Full-height color-6 separator columns isolate the main canvas from one or two 5-column legend panels.
    - Each legend panel contains, from top to bottom, a two-color unit sprite, an outer holey frame, an inner holey frame,
      a 5x5 anchor square, and a 3x3 rotation marker.
    - The zero holes inside both frame components are intentional signal, not empty filler.

    Rule:
    - Read and color-unswap the unit sprite. Count enclosed 4-connected holes in the inner frame and repeat the unit sprite
      in an inner_holes x inner_holes square before stamping it into every active inner-frame cell.
    - Count enclosed holes in the outer frame and repeat that entire inner composite in an outer_holes x outer_holes square
      before stamping it into every active outer-frame cell. Rotate the finished recursive composite by the clock-hand marker.
    - Place the rotated result on the canvas using the anchor encoded by the marker square.

    Key insight:
    - The frame section now carries two independent kinds of information at once: colored pixels say where the next-level
      template goes, while zero holes inside that template say how large the next-level payload becomes before recursion continues.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
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

        frames = _extract_nested_frames(panel, unit_bb, marker_r0)
        if frames is None:
            continue
        outer_mask, inner_mask = frames
        outer_frame = outer_mask.astype(int)
        inner_frame = inner_mask.astype(int)
        outer_holes = _count_frame_holes(outer_frame)
        inner_holes = _count_frame_holes(inner_frame)

        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        comp = _build_fused_composite(unit_sprite, outer_mask, inner_mask, outer_holes, inner_holes, dir_k)

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


# --------------- generator (counted_recursive_nested_frames) ---------------

GRID_H_RANGE = (38, 44)
GRID_W = 44
LEGEND_PANEL_W = 5
MARKER_PANEL_HEIGHT = 8
BAND_WIDTH_MIN = 5
USABLE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 500
PAIR_CHOICES = [(2, 1), (1, 2)]


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

    all_set = set(USABLE_COLORS)
    frame_colors = []
    for uc in unit_color_sets:
        frame_colors.append(random.choice(list(all_set - set(uc))))
    return unit_color_sets, bg_colors, frame_colors


def _color_swap(sprite: np.ndarray, c1: int, c2: int) -> np.ndarray:
    """Return a copy of sprite with the two colors swapped."""
    sa, sb = sorted([c1, c2])
    out = sprite.copy()
    out[sprite == sa] = sb
    out[sprite == sb] = sa
    return out


def _sprite_library(c1: int, c2: int) -> list[np.ndarray]:
    """Small asymmetric 2-color unit sprites that keep recursive composites compact."""
    return [
        np.array([[c1], [c2]], dtype=int),
        np.array([[c2], [c1]], dtype=int),
        np.array([[c1, c2]], dtype=int),
        np.array([[c2, c1]], dtype=int),
    ]


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
    """Stamp each legend's recursive composite onto the canvas."""
    output = canvas.copy()
    for leg in legends:
        comp = leg["composite"]
        t, l = leg["comp_top"], leg["comp_left"]
        ch, cw = comp.shape
        region = output[t : t + ch, l : l + cw]
        write_mask = (comp != 0) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


def _comp_size(leg: dict) -> tuple[int, int]:
    comp = leg["composite"]
    return int(comp.shape[0]), int(comp.shape[1])


def _assign_anchors(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
    """Assign anchor positions, avoiding overlap for the 2-legend case."""
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


def _random_frame_instance(hole_count: int) -> np.ndarray:
    return random.choice(FRAME_LIBRARY[hole_count]).copy()


def _design_legend(
    leg: dict,
    unit_colors: tuple[int, int],
    frame_color: int,
    pair_choices: list[tuple[int, int]],
    marker_r0: int,
    grid_h: int,
    canvas_w: int,
) -> bool:
    """Choose a fully feasible legend specification and prebuild its output composite."""
    c1, c2 = unit_colors
    candidates: list[dict] = []
    for pair in pair_choices:
        outer_holes, inner_holes = pair
        for sprite in _sprite_library(c1, c2):
            legend_sprite = _color_swap(sprite, c1, c2)
            for outer_mask in FRAME_LIBRARY[outer_holes]:
                for inner_mask in FRAME_LIBRARY[inner_holes]:
                    min_rows = 1 + sprite.shape[0] + 1 + outer_mask.shape[0] + 1 + inner_mask.shape[0]
                    if min_rows > marker_r0:
                        continue
                    for rot_k in range(4):
                        comp = _build_fused_composite(
                            sprite, outer_mask, inner_mask, outer_holes, inner_holes, rot_k
                        )
                        ch, cw = comp.shape
                        if ch > grid_h or cw > canvas_w:
                            continue
                        candidates.append(
                            {
                                "sprite": sprite,
                                "legend_sprite": legend_sprite,
                                "outer_holes": outer_holes,
                                "inner_holes": inner_holes,
                                "outer_mask": outer_mask,
                                "inner_mask": inner_mask,
                                "frame_color": frame_color,
                                "rot_k": rot_k,
                                "composite": comp,
                            }
                        )
    if not candidates:
        return False

    choice = random.choice(candidates)
    sprite = choice["sprite"]
    outer_mask = choice["outer_mask"]
    inner_mask = choice["inner_mask"]

    spare = marker_r0 - (1 + sprite.shape[0] + 1 + outer_mask.shape[0] + 1 + inner_mask.shape[0])
    extra1 = random.randint(0, min(3, spare))
    spare -= extra1
    extra2 = random.randint(0, min(3, spare))
    gap1 = 1 + extra1
    gap2 = 1 + extra2
    outer_r0 = 1 + sprite.shape[0] + gap1
    inner_r0 = outer_r0 + outer_mask.shape[0] + gap2
    sprite_c0 = random.randint(0, LEGEND_PANEL_W - sprite.shape[1])
    outer_c0 = random.randint(0, LEGEND_PANEL_W - outer_mask.shape[1])
    inner_c0 = random.randint(0, LEGEND_PANEL_W - inner_mask.shape[1])

    leg.update(choice)
    leg.update(
        sprite_c0=sprite_c0,
        outer_r0=outer_r0,
        outer_c0=outer_c0,
        inner_r0=inner_r0,
        inner_c0=inner_c0,
    )
    return True


def _place_legend(grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int) -> None:
    """Draw one 5-column legend panel plus its separator column."""
    grid[:, sep_col] = 6
    panel = grid[:, col_start : col_start + LEGEND_PANEL_W]
    panel[:] = 0

    panel[marker_r0 : marker_r0 + 5, :] = 0
    panel[marker_r0, :] = 6
    panel[marker_r0 + 4, :] = 6
    panel[marker_r0 : marker_r0 + 5, 0] = 6
    panel[marker_r0 : marker_r0 + 5, 4] = 6
    panel[marker_r0 + 1 + leg["pr"], 1 + leg["pc"]] = 4

    rr0 = marker_r0 + 5
    panel[rr0 : rr0 + 3, :] = 0
    panel[rr0 : rr0 + 3, 0] = 6
    panel[rr0 : rr0 + 3, 4] = 6
    cx_r, cx_c = rr0 + 1, 2
    dr, dc = ROTATION_DIRS[leg["rot_k"]]
    panel[cx_r, cx_c] = 7
    panel[cx_r + dr, cx_c + dc] = 7

    sprite = leg["legend_sprite"]
    panel[1 : 1 + sprite.shape[0], leg["sprite_c0"] : leg["sprite_c0"] + sprite.shape[1]] = sprite

    outer_region = panel[
        leg["outer_r0"] : leg["outer_r0"] + leg["outer_mask"].shape[0],
        leg["outer_c0"] : leg["outer_c0"] + leg["outer_mask"].shape[1],
    ]
    outer_region[leg["outer_mask"]] = leg["frame_color"]

    inner_region = panel[
        leg["inner_r0"] : leg["inner_r0"] + leg["inner_mask"].shape[0],
        leg["inner_c0"] : leg["inner_c0"] + leg["inner_mask"].shape[1],
    ]
    inner_region[leg["inner_mask"]] = leg["frame_color"]


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    num_legends = 1

    left_total = LEGEND_PANEL_W + 1
    right_total = (1 + LEGEND_PANEL_W) if num_legends == 2 else 0
    canvas_w = GRID_W - left_total - right_total
    if canvas_w <= 0:
        return None

    colors = _allocate_colors(num_legends)
    if colors is None:
        return None
    unit_color_sets, (bg1, bg2), frame_colors = colors

    marker_r0 = grid_h - MARKER_PANEL_HEIGHT
    legends = [{} for _ in range(num_legends)]

    for li in range(num_legends):
        if not _design_legend(
            legends[li],
            unit_color_sets[li],
            frame_colors[li],
            PAIR_CHOICES,
            marker_r0,
            grid_h,
            canvas_w,
        ):
            return None

    if not _assign_anchors(legends, grid_h, canvas_w):
        return None

    canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2)
    output = _overlay_composites(canvas, legends)

    grid = np.zeros((grid_h, GRID_W), dtype=int)
    canvas_c0 = left_total
    grid[:, canvas_c0 : canvas_c0 + canvas_w] = canvas

    _place_legend(grid, legends[0], col_start=0, sep_col=LEGEND_PANEL_W, marker_r0=marker_r0)
    if num_legends == 2:
        r_sep = canvas_c0 + canvas_w
        _place_legend(grid, legends[1], col_start=r_sep + 1, sep_col=r_sep, marker_r0=marker_r0)

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
