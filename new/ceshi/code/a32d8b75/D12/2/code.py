# Variant of a32d8b75
# Dimension: D12 — 复合精灵的放置约束 (meta)
# Change: 复合精灵必须完全在画布内（无裁剪），叠加时非{0,6}像素覆写画布，画布中颜色6的像素保留... → 精灵与画布颜色交互：仅覆写与精灵不同色的画布像素

from __future__ import annotations

import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4, random_sprite


# Design analysis for the D12 variant
# a) Concrete behavior and trace:
#    The legend-decoding pipeline stays the same: each legend panel still contains one unit
#    sprite, one frame mask, one 5x5 anchor square, and one 3x3 rotation mark. The only
#    changed step is the final per-pixel interaction between the constructed composite and the
#    canvas. A sprite pixel now writes only when the canvas color at that location is different.
#    This rule is visually equivalent to the old solver on ordinary non-6 backgrounds because
#    "write same color" and "leave same color unchanged" look identical. To make D12 observable,
#    the generator intentionally places internal color-6 blocks inside the canvas under selected
#    active composite tiles. Old logic would preserve those 6-pixels; the new variant paints over
#    them because 6 differs from every sprite color. Example: if a 3x3 sprite tile lands on a 3x3
#    canvas block filled with 6, the old task leaves a magenta hole while the new task fills that
#    block with the tiled sprite colors.
# b) Complexity calibration:
#    The background is restricted to the same two colors used by all legends' sprites, plus a
#    small number of injected 6-blocks. This keeps the color vocabulary low and repeated across
#    examples, so humans can compare "same-color contact", "different non-6 contact", and
#    "different color-6 contact" without extra irrelevant variation.
# c) Implicit assumptions review:
#    In the original main(), the overlay loop relied on two now-stale assumptions:
#    - `if out[rr, cc] == 6: continue` assumed canvas color 6 was always protected.
#    - unconditional `out[rr, cc] = val` assumed equality with the canvas did not matter.
#    The variant removes the first assumption and replaces the second with a color comparison.
#    The non-overlap assertion between different legends is kept, because D12 only changes
#    canvas interaction, not inter-composite placement.
# d) Placement legality:
#    The generator still computes placements output-first, keeps every composite fully in bounds,
#    and rejects overlapping composite bounding boxes. Injected 6-blocks are aligned to active
#    rotated-frame tiles, so every inserted 6-pixel is guaranteed to be overwritten by the new
#    solver and never forms a full-height/full-width 6 separator inside the canvas.
# e) Visual observability:
#    Each example contains both:
#    - same-color contacts between sprite pixels and ordinary background colors, and
#    - overwritten internal 6-blocks that differ from the original task.
#    The second point is what makes the variant externally distinguishable from the source task.
# f) Visual cleanliness:
#    The injected 6-regions are full N x N tile blocks under active composite cells rather than
#    scattered noise, so the input remains readable and the output shows coherent filled-in areas.
# g) Color-change inferability:
#    No hidden ordering by color id is used. The only color relation is visible equality versus
#    inequality at the overlap location. The generator measures same-color hits, different-color
#    hits, and overwritten-6 hits and rejects degenerate samples with too little interaction.
# h) New solver assumptions and generator guarantees:
#    - main() assumes legends remain outside the output canvas; generate_input() preserves that.
#    - main() assumes active composite pixels from different legends do not overlap; the generator
#      enforces non-overlapping composite bounding boxes.
#    - main() assumes any internal color-6 canvas patches are ordinary canvas content, not
#      separators; the generator only creates finite 6-blocks that never span a full row/column.


def _separator_runs(g: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous runs of full-height columns that are entirely color 6."""
    h, w = g.shape
    cols6 = [c for c in range(w) if np.all(g[:, c] == 6)]
    if not cols6:
        return []
    runs: list[tuple[int, int]] = []
    start = end = cols6[0]
    for c in cols6[1:]:
        if c == end + 1:
            end = c
        else:
            runs.append((start, end))
            start = end = c
    runs.append((start, end))
    return runs


def _split_canvas(g: np.ndarray) -> tuple[tuple[int, int], tuple[int, int] | None]:
    """
    Identify the left separator run and, if present, a right separator near the far edge.
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
    """List all 5x5 border-6 squares that behave like the top of an 8x5 marker panel."""
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
    """Read the top-most 4-connected non-{0,6} component and swap its two colors."""
    mask = (legend != 0) & (legend != 6)
    pts_all = np.argwhere(mask)
    if pts_all.size == 0:
        return None
    rmin = int(pts_all[:, 0].min())
    cseed = int(np.argwhere(mask[rmin, :])[0, 0])

    h, w = legend.shape
    vis = np.zeros((h, w), dtype=bool)
    stack = [(rmin, cseed)]
    vis[rmin, cseed] = True
    pts: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        pts.append((r, c))
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not vis[rr, cc]:
                vis[rr, cc] = True
                stack.append((rr, cc))

    r0, c0, r1, c1 = bbox_coords(np.asarray(pts))
    spr = legend[r0 : r1 + 1, c0 : c1 + 1].copy()
    cols = sorted({int(x) for x in np.unique(spr) if int(x) not in (0, 6)})
    if len(cols) == 2:
        a, b = cols
        out = spr.copy()
        out[spr == a] = -1
        out[spr == b] = a
        out[out == -1] = b
        spr = out
    return spr, (r0, c0, r1, c1)


def _extract_frame_bbox(
    legend: np.ndarray, unit_bb: tuple[int, int, int, int], border6_r0: int
) -> tuple[int, int, int, int] | None:
    """Locate the frame pixels between the unit sprite and the marker square."""
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
    return r0, c0, r1, c1


def _position_indices_from_border_square(
    g: np.ndarray, border6_r0: int, border6_c0: int
) -> tuple[int, int] | None:
    sq = g[border6_r0 : border6_r0 + 5, border6_c0 : border6_c0 + 5]
    inner = sq[1:4, 1:4]
    pts = np.argwhere((inner != 0) & (inner != 6))
    if pts.size == 0:
        return None
    r, c = map(int, pts[0])
    return r, c


def _rotation_from_clockhand(g: np.ndarray, border6_r0: int, border6_c0: int) -> int:
    """
    Read the 3x3 clock-hand patch below the marker square.

    Furthest non-{0,6} pixel from the center determines direction:
    up -> 0, right -> 1, down -> 2, left -> 3.
    """
    h, w = g.shape
    r0 = border6_r0 + 5
    c0 = border6_c0 + 1
    assert r0 + 2 < h and c0 + 2 < w, "rotation marker patch out of bounds"
    sub = g[r0 : r0 + 3, c0 : c0 + 3]
    pts = np.argwhere((sub != 0) & (sub != 6))
    assert pts.size > 0, "no rotation marker pixel found"
    center = np.array([1, 1], dtype=int)
    dists = np.abs(pts - center).sum(axis=1)
    pr, pc = pts[int(dists.argmax())]
    dr, dc = int(pr - 1), int(pc - 1)
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
    """Convert a start/middle/end anchor choice into the placed object's top-left offset."""
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
    """Tile the unit sprite into every active frame cell."""
    unit_mask = (unit_sprite != 0) & (unit_sprite != 6)
    uh, uw = unit_sprite.shape
    fh, fw = frame_mask.shape
    comp = np.zeros((fh * uh, fw * uw), dtype=int)
    for i in range(fh):
        for j in range(fw):
            if not frame_mask[i, j]:
                continue
            r0 = i * uh
            c0 = j * uw
            comp[r0 : r0 + uh, c0 : c0 + uw][unit_mask] = unit_sprite[unit_mask]
    return comp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode legend panels, assemble enlarged sprites, and place them onto the canvas.

    Input structure:
    - Full-height color-6 separator columns carve the grid into one left legend, a main canvas,
      and optionally a right legend.
    - Every legend panel still provides a two-color unit sprite, a frame template, a 5x5 anchor
      square, and a 3x3 rotation clock-hand.

    Rule:
    - Expand the unit sprite across every active frame cell, then rotate the resulting composite
      according to the clock-hand.
    - Place the rotated composite at the canvas anchor encoded by the marker square, but only
      write a sprite pixel when the canvas color at that location is different; equal colors stay
      as they are.

    Key insight:
    - The marker square still determines where the composite is anchored, but the final image is
      also controlled by a pixelwise color comparison: same-color contacts disappear into the
      background, while any different canvas color, including internal 6-patches, gets replaced.
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

        frame_bb = _extract_frame_bbox(panel, unit_bb, marker_r0)
        if frame_bb is None:
            continue
        fr0, fc0, fr1, fc1 = frame_bb
        frame = panel[fr0 : fr1 + 1, fc0 : fc1 + 1]
        frame_mask0 = (frame != 0) & (frame != 6)
        _, n8 = _scipy_label(frame_mask0, structure=np.ones((3, 3)))
        assert n8 == 1, f"frame must be 8-connected, got {n8} components"

        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        frame_mask = np.rot90(frame_mask0, (-dir_k) % 4)
        comp = _build_composite(unit_sprite, frame_mask)

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
                val = int(comp[r, c])
                if val == 0 or val == 6:
                    continue
                assert not composite_written[rr, cc], "composites must not overlap"
                composite_written[rr, cc] = True
                if out[rr, cc] != val:
                    out[rr, cc] = val
    return out


# --------------- generator ---------------

GRID_H_RANGE = (20, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [1, 4]
UNIT_SIZE_RANGE = (2, 5)
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)
FRAME_W_RANGE = (3, 6)
FRAME_DENSITY = 0.5
BAND_WIDTH_MIN = 3
MARKER_PANEL_HEIGHT = 8
USABLE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 400


def _allocate_colors(num_legends: int) -> tuple[list[tuple[int, int]], tuple[int, int], list[int]]:
    """All legends share one two-color sprite palette; frame colors stay outside it."""
    shared = tuple(random.sample(USABLE_COLORS, 2))
    frame_pool = [c for c in USABLE_COLORS if c not in shared]
    if len(frame_pool) >= num_legends:
        frame_colors = random.sample(frame_pool, num_legends)
    else:
        frame_colors = [random.choice(frame_pool) for _ in range(num_legends)]
    unit_sets = [shared for _ in range(num_legends)]
    return unit_sets, shared, frame_colors


def _color_swap(sprite: np.ndarray, c1: int, c2: int) -> np.ndarray:
    out = sprite.copy()
    out[sprite == c1] = -1
    out[sprite == c2] = c1
    out[out == -1] = c2
    return out


def _random_unit_sprite(n: int, colors: tuple[int, int]) -> np.ndarray:
    """Generate a filled two-color NxN sprite with visible internal variation."""
    c1, c2 = colors
    min_c2 = max(1, n)
    max_c2 = min(n * n - 1, n * n - max(1, n))
    for _ in range(50):
        sprite = np.full((n, n), c1, dtype=int)
        num_c2 = random.randint(min_c2, max_c2)
        for p in random.sample(range(n * n), num_c2):
            sprite[p // n, p % n] = c2
        row_mix = any(np.any(row == c1) and np.any(row == c2) for row in sprite)
        col_mix = any(np.any(col == c1) and np.any(col == c2) for col in sprite.T)
        if row_mix and col_mix:
            return sprite
    return sprite


def _design_legend(
    leg: dict, unit_colors: tuple[int, int], frame_color: int, marker_r0: int, grid_h: int, canvas_w: int
) -> bool:
    """Fill in sprite, frame, rotation, and prebuilt composite for one legend."""
    n = leg["N"]
    sprite = _random_unit_sprite(n, unit_colors)
    legend_sprite = _color_swap(sprite, *unit_colors)

    avail = marker_r0 - (n + 1)
    combos: list[tuple[int, int, int]] = []
    for frame_h in range(*FRAME_H_RANGE):
        if avail < frame_h + 1:
            continue
        for frame_w in range(*FRAME_W_RANGE):
            for rot_k in range(4):
                rh, rw = (frame_h, frame_w) if rot_k % 2 == 0 else (frame_w, frame_h)
                if rh * n <= grid_h and rw * n <= canvas_w:
                    combos.append((frame_h, frame_w, rot_k))
    if not combos:
        return False

    frame_h, frame_w, rot_k = random.choice(combos)
    frame_mask = random_sprite(
        frame_h,
        frame_w,
        density=FRAME_DENSITY,
        connectivity=8,
        color_palette=[1],
        symmetry="not_symmetric",
    ) != 0
    rot_frame = np.rot90(frame_mask, (-rot_k) % 4)
    comp = _build_composite(sprite, rot_frame)

    leg.update(
        sprite=sprite,
        legend_sprite=legend_sprite,
        frame_mask=frame_mask,
        frame_H=frame_h,
        frame_W=frame_w,
        frame_color=frame_color,
        rot_k=rot_k,
        rotated_frame=rot_frame,
        comp=comp,
    )
    return True


def _comp_size(leg: dict) -> tuple[int, int]:
    return tuple(map(int, leg["comp"].shape))


def _assign_anchors(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
    """Assign non-overlapping placements based on start/middle/end anchors."""
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
    a_h, a_w = _comp_size(a)
    b_h, b_w = _comp_size(b)
    pairs: list[tuple[int, int, int, int, int, int, int, int]] = []
    for apr in range(3):
        for apc in range(3):
            at = _anchor_to_top_left(apr, grid_h, a_h)
            al = _anchor_to_top_left(apc, canvas_w, a_w)
            if at < 0 or al < 0 or at + a_h > grid_h or al + a_w > canvas_w:
                continue
            for bpr in range(3):
                for bpc in range(3):
                    bt = _anchor_to_top_left(bpr, grid_h, b_h)
                    bl = _anchor_to_top_left(bpc, canvas_w, b_w)
                    if bt < 0 or bl < 0 or bt + b_h > grid_h or bl + b_w > canvas_w:
                        continue
                    if at < bt + b_h and bt < at + a_h and al < bl + b_w and bl < al + a_w:
                        continue
                    pairs.append((apr, apc, at, al, bpr, bpc, bt, bl))
    if not pairs:
        return False
    apr, apc, at, al, bpr, bpc, bt, bl = random.choice(pairs)
    a.update(pr=apr, pc=apc, comp_top=at, comp_left=al)
    b.update(pr=bpr, pc=bpc, comp_top=bt, comp_left=bl)
    return True


def _build_banded_canvas(grid_h: int, canvas_w: int, colors: tuple[int, int]) -> np.ndarray:
    """Create a clean two-color striped canvas using the shared sprite palette."""
    bg1, bg2 = colors
    canvas = np.zeros((grid_h, canvas_w), dtype=int)
    band_dir = random.choice(["h", "v"])
    dim = grid_h if band_dir == "h" else canvas_w
    max_bw = max(BAND_WIDTH_MIN, min(6, max(3, dim // 2)))
    bw = random.randint(BAND_WIDTH_MIN, max_bw)
    band_indices = (np.arange(dim) // bw) % 2
    palette = np.array([bg1, bg2], dtype=int)
    if band_dir == "h":
        canvas[:] = palette[band_indices][:, None]
    else:
        canvas[:] = palette[band_indices][None, :]
    return canvas


def _inject_canvas_sixes(canvas: np.ndarray, legends: list[dict]) -> None:
    """
    Paint compact 6-blocks under selected active composite tiles.

    Each injected block is exactly one active N x N tile, so the new solver must overwrite it.
    """
    for leg in legends:
        rf = leg["rotated_frame"]
        n = leg["N"]
        top = leg["comp_top"]
        left = leg["comp_left"]
        active = [tuple(map(int, rc)) for rc in np.argwhere(rf)]
        assert active, "rotated frame must have at least one active cell"
        num_blocks = random.randint(1, min(3, len(active)))
        for i, j in random.sample(active, num_blocks):
            r0 = top + i * n
            c0 = left + j * n
            canvas[r0 : r0 + n, c0 : c0 + n] = 6


def _overlay_composites(canvas: np.ndarray, legends: list[dict]) -> np.ndarray:
    """Apply the new D12 interaction: only different-colored canvas pixels are replaced."""
    output = canvas.copy()
    for leg in legends:
        comp = leg["comp"]
        t = leg["comp_top"]
        l = leg["comp_left"]
        h, w = comp.shape
        region = output[t : t + h, l : l + w]
        write_mask = (comp != 0) & (region != comp)
        region[write_mask] = comp[write_mask]
    return output


def _overlay_composites_old_rule(canvas: np.ndarray, legends: list[dict]) -> np.ndarray:
    """Reference overlay for the source task: preserve canvas 6, overwrite other colors."""
    output = canvas.copy()
    for leg in legends:
        comp = leg["comp"]
        t = leg["comp_top"]
        l = leg["comp_left"]
        h, w = comp.shape
        region = output[t : t + h, l : l + w]
        write_mask = (comp != 0) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


def _interaction_metrics(canvas: np.ndarray, legends: list[dict]) -> dict[str, int]:
    """Measure how strongly a sample exercises same-color and different-color interactions."""
    same_non6 = 0
    diff_non6 = 0
    diff_6 = 0
    total = 0
    for leg in legends:
        comp = leg["comp"]
        t = leg["comp_top"]
        l = leg["comp_left"]
        h, w = comp.shape
        region = canvas[t : t + h, l : l + w]
        mask = comp != 0
        same_non6 += int(np.sum(mask & (region == comp) & (region != 6)))
        diff_non6 += int(np.sum(mask & (region != comp) & (region != 6)))
        diff_6 += int(np.sum(mask & (region == 6)))
        total += int(np.sum(mask))
    return {
        "same_non6": same_non6,
        "diff_non6": diff_non6,
        "diff_6": diff_6,
        "total": total,
    }


def _place_legend(
    grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int, is_right: bool
) -> None:
    """Draw one legend panel into the full input grid."""
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

    frame_h, frame_w = leg["frame_H"], leg["frame_W"]
    avail = marker_r0 - (n + 1)
    gap = random.randint(1, max(1, avail - frame_h))
    fr_start = n + 1 + gap
    f_col = random.randint(0, max(0, 4 - frame_w))
    frame_region = grid[fr_start : fr_start + frame_h, mc0 + f_col : mc0 + f_col + frame_w]
    frame_region[leg["frame_mask"]] = leg["frame_color"]


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]

    legends = []
    for _ in range(num_legends):
        n = random.randint(*UNIT_SIZE_RANGE)
        legends.append({"N": n, "legend_w": n + 2})

    left_lw = legends[0]["legend_w"]
    left_total = left_lw + 1
    right_total = 1 + legends[1]["legend_w"] if num_legends == 2 else 0
    canvas_w = GRID_W - left_total - right_total
    if canvas_w < MIN_CANVAS_W:
        return None

    unit_sets, bg_colors, frame_colors = _allocate_colors(num_legends)
    marker_r0 = grid_h - MARKER_PANEL_HEIGHT

    for idx in range(num_legends):
        if not _design_legend(legends[idx], unit_sets[idx], frame_colors[idx], marker_r0, grid_h, canvas_w):
            return None

    if not _assign_anchors(legends, grid_h, canvas_w):
        return None

    canvas = _build_banded_canvas(grid_h, canvas_w, bg_colors)
    _inject_canvas_sixes(canvas, legends)

    metrics = _interaction_metrics(canvas, legends)
    if metrics["diff_6"] < 4 * num_legends:
        return None
    if metrics["same_non6"] < max(4, metrics["total"] // 12):
        return None
    if metrics["diff_non6"] < max(4, metrics["total"] // 12):
        return None

    output = _overlay_composites(canvas, legends)
    old_output = _overlay_composites_old_rule(canvas, legends)
    if int(np.sum(output != old_output)) < metrics["diff_6"]:
        return None

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
