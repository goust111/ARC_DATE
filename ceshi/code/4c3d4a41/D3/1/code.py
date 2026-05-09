# Variant of 4c3d4a41
# Dimension: D3 — 目标对象：框架结构 (dst)
# Change: 右侧固定位置的灰色矩形框架（8x11），左墙有2格开口（行4-5），内部有4条彩色列（金字塔形长度3,4,4,3）... → 多个小框架替代单一大框架

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg as _bg_color
from arc_utils import colors_present, get_bg, paste

# Design analysis for the D3 variant
# variant_analysis:
#   concretization:
#     - Keep the original palette semantics: bg=0, grey=5, four distinct column colors.
#     - Keep the original vertical mechanics: each colored column is a top-aligned stack whose
#       visible colored cells end immediately above the inserted bar projection.
#     - Replace the single 8x11 destination frame with 2 or 3 independent full-height frames.
#       A 1-column frame has width 3; a 2-column frame has width 5; all frames keep the same
#       left-wall opening at rows 4-5, so the split is visually obvious.
#     - Split the source bar into the same number of disconnected bar fragments. Each fragment
#       carries 1 or 2 upward projections and is placed on the left in shuffled order.
#   behavior_trace:
#     - Original: one global frame, one global bar, one global horizontal shift, then recolor
#       all four destination columns in one pass.
#     - Variant: the four original column lengths [3,4,4,3] are partitioned into contiguous
#       groups [2,2], [1,1,2], or [2,1,1]. Each group becomes its own small frame and its own
#       bar fragment. Example [1,1,2] yields frame signatures (3), (4), (4,3); the left-side
#       bars are shuffled, then each fragment must be matched back to the frame with the same
#       ordered height signature before insertion.
#     - Difference: the solver can no longer treat destination geometry as one bbox or assume
#       “largest grey component = frame, next = bar”. It must solve several independent matches.
#   complexity_calibration:
#     - Only 2 or 3 frames are allowed. This is enough to make the split plainly visible while
#       keeping the puzzle readable from a few examples.
#     - Group sizes are restricted to 1 or 2 columns, which preserves the original projection
#       spacing and keeps the visual language close to the source task.
#     - The allowed partitions are chosen so every frame signature is unique; this avoids
#       accidental ambiguity while still forcing genuine multi-object matching.
#   implicit_assumption_audit:
#     - `grey_comps[0]` as the only slot no longer holds: there are several tall grey frame
#       components plus several short bar components.
#     - One global `dst_cols` set is wrong in the variant: each frame has its own destination
#       columns and its own local horizontal shift.
#     - One global `placed_by_col` map is also wrong: columns from different frames must never
#       interact during recoloring.
#     - The generator therefore guarantees: full-height frames are disconnected from each other,
#       bar fragments are disconnected from frames, and every frame signature has exactly one
#       matching bar signature.
#   placement_legality:
#     - The canvas is widened from 20 to 21 only to preserve one blank column between the left
#       bar area and the widest 3-frame layout on the right; this prevents unintended grey
#       connectivity across source and destination objects.
#     - All frames are right-aligned, separated by one bg column, and stay within rows 0-7.
#     - All source bars share the original base row 5, occupy cols 0-6 with one-column gaps,
#       and never overlap each other.
#   observability_and_cleanliness:
#     - A human can see the core change immediately: instead of one large frame swallowing one
#       bar, several smaller frames each receive a different fragment.
#     - The matching cue is visible geometry, not hidden ids: a frame with column silhouette
#       (3,4) must receive the bar fragment with projection silhouette (3,4).
#     - Colors are not remapped or sorted by id; they remain tied to their original columns.


BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

H, W = 8, 21
FRAME_R0 = 0
FRAME_H = 8
LEFT_WALL_OPEN_ROWS = (4, 5)
SLOT_TOP_ROW = 1
BAR_R0 = 2
BAR_PATCH_H = 4
BAR_BASE_ROW = BAR_R0 + BAR_PATCH_H - 1
FRAME_GAP = 1
BAR_GAP = 1
ORIG_SLOT_LENS = [3, 4, 4, 3]
PARTITIONS = [(2, 2), (1, 1, 2), (2, 1, 1)]
MAX_ATTEMPTS = 300


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _frame_patch(ncols: int) -> np.ndarray:
    width = 2 * ncols + 1
    patch = np.zeros((FRAME_H, width), dtype=int)
    patch[[0, -1], :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    patch[list(LEFT_WALL_OPEN_ROWS), 0] = BG
    return patch


def _bar_patch(signature: tuple[int, ...]) -> np.ndarray:
    patch = np.zeros((BAR_PATCH_H, 2 * len(signature) - 1), dtype=int)
    patch[-1, :] = GREY
    for i, height in enumerate(signature):
        patch[BAR_PATCH_H - height :, 2 * i] = GREY
    return patch


def _paint_group_columns(
    grid: np.ndarray,
    frame_c0: int,
    slot_lengths: tuple[int, ...],
    colors: tuple[int, ...],
    counts: tuple[int, ...],
    *,
    output_mode: bool,
) -> None:
    for i, (length, color, count) in enumerate(zip(slot_lengths, colors, counts)):
        col = frame_c0 + 1 + 2 * i
        cap = length - count
        grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
        if not output_mode:
            grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + length, col] = color
            continue

        visible = min(count, BAR_BASE_ROW - length)
        if visible <= 0:
            continue
        row_end = BAR_BASE_ROW - length
        grid[row_end - visible + 1 : row_end + 1, col] = color


def _frame_signature_from_grid(g: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> tuple[int, ...]:
    r0, c0, r1, c1 = bbox
    sig = []
    for c in range(c0 + 1, c1):
        length = int(np.count_nonzero(g[r0 + 1 : r1, c] != bg))
        if length > 0:
            sig.append(length)
    return tuple(sig)


def _frame_color_cols(g: np.ndarray, bbox: tuple[int, int, int, int], bg: int, grey: int) -> list[int]:
    r0, c0, r1, c1 = bbox
    cols = []
    for c in range(c0 + 1, c1):
        if np.any((g[r0 : r1 + 1, c] != bg) & (g[r0 : r1 + 1, c] != grey)):
            cols.append(int(c))
    return cols


def _bar_signature_and_cols(pts: np.ndarray) -> tuple[tuple[int, ...], list[int], int]:
    base_row = max(int(r) for r, _ in pts)
    col_min_row = {}
    for r, c in pts:
        rr = int(r)
        cc = int(c)
        col_min_row[cc] = min(col_min_row.get(cc, rr), rr)
    proj_cols = sorted([c for c, rmin in col_min_row.items() if rmin < base_row])
    signature = tuple(base_row - col_min_row[c] + 1 for c in proj_cols)
    return signature, proj_cols, base_row


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit several grey bar fragments into several split grey frames by matching their local silhouettes, then restack colors above each fragment.

    Input structure:
    - bg plus grey structural pieces forming multiple tall destination frames on the right and multiple short source bar fragments on the left.
    - Each frame contains 1 or 2 colored columns; together the frame columns still realize the original [3,4,4,3] pyramid, but split across separate frame components.

    Rule:
    - Match each source bar fragment to the frame whose ordered column-height silhouette matches the fragment's ordered projection-height silhouette.
    - Move each matched fragment into its frame so the fragment base sits one cell above the frame interior bottom and the projections align with that frame's colored columns.
    - For every destination column in that frame, stack the original column's colored pixels immediately above the inserted projection, clipped to the available vertical space.

    Key insight: The puzzle is no longer one bar entering one slot; the correct assembly comes from pairing each disconnected frame fragment with the bar fragment that has the same local height signature.
    """
    g = np.asarray(grid, dtype=int)

    bg = _bg_color(g)
    other = g[g != bg]
    if other.size == 0:
        return g.copy()
    grey = _most_common(other)

    grey_comps = [(len(pts), _bbox(pts), pts) for pts in _cc4(g == grey)]
    if len(grey_comps) < 4:
        return g.copy()

    frames = []
    bars = []
    for item in grey_comps:
        bbox = item[1]
        height = bbox[2] - bbox[0] + 1
        if height >= FRAME_H - 1:
            frames.append(item)
        else:
            bars.append(item)

    if not frames or not bars:
        return g.copy()

    frames.sort(key=lambda t: t[1][1])
    out = np.full_like(g, bg)

    for _, frame_bbox, _ in frames:
        r0, c0, r1, c1 = frame_bbox
        out[r0 : r1 + 1, c0 : c1 + 1] = g[r0 : r1 + 1, c0 : c1 + 1]
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[(region != bg) & (region != grey)] = bg

    bars_by_sig: dict[tuple[int, ...], list[np.ndarray]] = {}
    for _, _, pts in bars:
        sig, _, _ = _bar_signature_and_cols(pts)
        if sig:
            bars_by_sig.setdefault(sig, []).append(pts)

    for _, frame_bbox, _ in frames:
        sig = _frame_signature_from_grid(g, frame_bbox, bg)
        candidates = bars_by_sig.get(sig, [])
        if not candidates:
            continue
        bar_pts = candidates.pop(0)
        _, src_proj_cols, bar_base_row = _bar_signature_and_cols(bar_pts)

        sr0, sc0, sr1, sc1 = frame_bbox
        dst_cols = _frame_color_cols(g, frame_bbox, bg, grey)
        if not src_proj_cols or not dst_cols:
            continue

        target_base_row = sr1 - 2
        dy = target_base_row - bar_base_row

        dx = None
        if len(src_proj_cols) == len(dst_cols):
            deltas = [d - s for s, d in zip(src_proj_cols, dst_cols)]
            if len(set(deltas)) == 1:
                dx = deltas[0]
        if dx is None:
            best = (-1, 0)
            dst_set = set(dst_cols)
            for cand_dx in range(-g.shape[1], g.shape[1] + 1):
                score = sum((c + cand_dx) in dst_set for c in src_proj_cols)
                if score > best[0]:
                    best = (score, cand_dx)
            dx = best[1]

        placed = []
        for r, c in bar_pts:
            rr = int(r) + dy
            cc = int(c) + dx
            if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
                out[rr, cc] = grey
                placed.append((rr, cc))

        placed_by_col = {}
        for rr, cc in placed:
            placed_by_col.setdefault(cc, []).append(rr)

        interior_top = sr0 + 1
        interior_bottom = sr1 - 1
        for col in dst_cols:
            if col not in placed_by_col:
                continue

            bar_top = min(placed_by_col[col])
            desired_bottom = bar_top - 1
            if desired_bottom < interior_top:
                continue

            rows = [r for r in range(sr0, sr1 + 1) if g[r, col] != bg and g[r, col] != grey]
            if not rows:
                continue
            colors = [int(g[r, col]) for r in rows]

            max_len = desired_bottom - interior_top + 1
            if max_len <= 0:
                continue
            if len(colors) > max_len:
                colors = colors[-max_len:]

            start_row = max(interior_top, desired_bottom - len(colors) + 1)
            for i, color in enumerate(colors):
                rr = start_row + i
                if interior_top <= rr <= interior_bottom:
                    out[rr, col] = color

    return out


def _partition_groups(
    sizes: tuple[int, ...],
    colors: list[int],
    counts: list[int],
) -> list[dict]:
    groups = []
    idx = 0
    for size in sizes:
        slot_lengths = tuple(ORIG_SLOT_LENS[idx : idx + size])
        groups.append(
            {
                "signature": slot_lengths,
                "colors": tuple(colors[idx : idx + size]),
                "counts": tuple(counts[idx : idx + size]),
            }
        )
        idx += size
    return groups


def _frame_positions(groups: list[dict]) -> list[int]:
    widths = [2 * len(group["signature"]) + 1 for group in groups]
    total_width = sum(widths) + FRAME_GAP * (len(widths) - 1)
    start = W - total_width
    positions = []
    cur = start
    for width in widths:
        positions.append(cur)
        cur += width + FRAME_GAP
    return positions


def _bar_positions(bar_order: list[int], groups: list[dict]) -> list[int]:
    widths = [2 * len(groups[idx]["signature"]) - 1 for idx in bar_order]
    positions = []
    cur = 0
    for width in widths:
        positions.append(cur)
        cur += width + BAR_GAP
    return positions


def _build_grid(groups: list[dict], bar_order: list[int], *, output_mode: bool) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)

    frame_positions = _frame_positions(groups)
    for group, frame_c0 in zip(groups, frame_positions):
        paste(grid, _frame_patch(len(group["signature"])), FRAME_R0, frame_c0)
        _paint_group_columns(
            grid,
            frame_c0,
            group["signature"],
            group["colors"],
            group["counts"],
            output_mode=output_mode,
        )
        if output_mode:
            paste(grid, _bar_patch(group["signature"]), BAR_R0, frame_c0 + 1, transparent=BG)

    if not output_mode:
        for idx, bar_c0 in zip(bar_order, _bar_positions(bar_order, groups)):
            paste(grid, _bar_patch(groups[idx]["signature"]), BAR_R0, bar_c0, transparent=BG)

    return grid


def _sample_params(rng: np.random.Generator) -> tuple[list[dict], list[int]]:
    partition = PARTITIONS[int(rng.integers(0, len(PARTITIONS)))]
    colors = [int(v) for v in rng.choice(PALETTE, size=4, replace=False)]
    counts = [int(rng.integers(1, length + 1)) for length in ORIG_SLOT_LENS]
    groups = _partition_groups(partition, colors, counts)

    n = len(groups)
    order = list(rng.permutation(n))
    if order == list(range(n)):
        order = order[1:] + order[:1]
    return groups, order


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    groups, bar_order = _sample_params(rng)
    grid = _build_grid(groups, bar_order, output_mode=False)
    expected = _build_grid(groups, bar_order, output_mode=True)

    signatures = [group["signature"] for group in groups]
    if len(set(signatures)) != len(signatures):
        return None
    if get_bg(grid) != BG:
        return None
    if sorted(colors_present(grid, exclude=[BG, GREY])) != sorted({c for group in groups for c in group["colors"]}):
        return None
    if np.array_equal(bar_order, np.arange(len(groups)).tolist()):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
