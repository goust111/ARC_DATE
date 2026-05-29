# Combo variant of 4c3d4a41 (index 6)
# combo_spec: D3/1+D15/0
# Name: 多小框架+多杆多框架配对

import itertools

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg as _bg_color
from arc_utils import colors_present, get_bg, paste


# Design analysis for the combo variant
# combo_analysis:
#   concretization_and_behavior_trace:
#     - Keep the original local action: a grey toothed bar is inserted into a half-open grey
#       frame, and each frame column's colored stack is trimmed so the visible color ends
#       immediately above the inserted tooth.
#     - Replace the single 4-column frame with 2 or 3 full-height mini-frames on the right.
#       Each mini-frame contains 1 or 2 colored columns, obtained by splitting the original
#       length profile [3,4,4,3] into contiguous groups:
#         (2,2) or (1,1,2) or (2,1,1).
#     - Replace the single source bar with the same number of disconnected source bars on the
#       left. A 1-column frame gets a 1-tooth bar; a 2-column frame gets a 2-tooth bar.
#     - New visible cue: each tooth carries a colored tag directly above it. The tag colors,
#       read left-to-right, exactly match the destination frame's colored columns.
#     - Example trace:
#         original: one frame with 4 columns + one bar -> one dx/dy -> one recoloring pass.
#         combo: partition (1,1,2) gives three frames with color signatures (2), (7), (3,9).
#                The three source bars appear on the left in shuffled order, e.g. (3,9), (7), (2).
#                The 2-tooth bar pairs by count immediately; the two 1-tooth bars are count-tied,
#                so their colored tags decide which singleton frame receives which bar.
#         difference: the solver must first decompose the scene into several frame/bar objects,
#         then solve a per-count matching problem, and only after that perform the original
#         insertion-and-restacking operation independently for each pair.
#   complexity_calibration:
#     - Mini-frames are limited to 1 or 2 columns. This keeps the scene readable while making
#       the split visually obvious.
#     - Only three partitions are allowed. They cover N=2 and N=3 without creating many edge
#       cases or hidden matching rules.
#     - All bar tooth heights are restricted to 2..4. This is deliberate: if height 1 were
#       allowed, a "tooth" could collapse into the base row and the visible tooth count would
#       stop matching the intended frame-column count.
#     - All four frame-column colors are distinct globally. This guarantees that a frame's
#       color signature is a human-visible identifier when count alone is ambiguous.
#   implicit_assumption_audit:
#     - Old assumption: "largest grey component is the only frame." Broken: there are 2 or 3
#       full-height frames, all disconnected.
#     - Old assumption: "second-largest grey component is the only bar." Broken: there are 2
#       or 3 source bars, and 1-tooth bars can even be narrower than before.
#     - Old assumption: one global destination-column set and one global dx. Broken: each frame
#       owns its own destination columns and its own matched bar.
#     - Old assumption: bar geometry alone identifies the target. Broken on purpose: in (2,2)
#       both bars have the same tooth count; in (1,1,2) or (2,1,1), the singleton bars tie.
#       The solver must consult colored tags once count-based grouping is insufficient.
#     - New combined risk: a colored tag could be mistaken for a destination-column pixel.
#       The solver avoids this by reading frame colors only from inside frame bboxes and bar
#       tags only from cells immediately above detected tooth tops.
#   placement_legality:
#     - Canvas is widened to 23 columns so up to three right-aligned frames and three left-side
#       bars fit with blank separation.
#     - Frames remain full-height (8 rows) and are separated by one background column.
#     - Bars share the original base row 5, are placed left-to-right with one background column
#       between them, and never overlap frames.
#     - Colored tags sit directly above tooth tops, so they remain visually attached to their
#       bars while staying outside frame regions.
#   visual_observability:
#     - A human can immediately see that the right side contains several small receiving frames
#       instead of one large one.
#     - The N:N nature is also visible: several independent source bars disappear from the left
#       and appear inside distinct frames on the right.
#     - The matching rule is legible from 2-3 examples:
#         count narrows candidates to "1-tooth vs 2-tooth";
#         tag colors break ties inside same-count groups.
#       No hidden dependence on numeric color ids is used.
#   visual_cleanliness:
#     - Only two structural colors remain: bg and grey. Colors only appear in frame columns and
#       in the small bar tags, so the scene does not become cluttered.
#     - Tags are placed on the same columns as teeth, which makes the bar-color association
#       obvious without extra markers or text-like patterns.
#   color_change_inference:
#     - Colors are not remapped. A bar tag color does not transform into a new color; it only
#       identifies which frame columns belong to that bar, and then the original frame-column
#       colors are restacked above the inserted bar.
#   new_solver_assumptions_checked_against_generator:
#     - Every frame uses the exact half-open skeleton and width 3 or 5.
#     - Every bar uses width 1 or 3, a full base row, and visible teeth in every designated
#       projection column because heights are sampled from 2..4.
#     - Within any count class that appears more than once, the generator shuffles the bars so
#       the left-to-right order disagrees with the frames' left-to-right order. This forces the
#       color-tag cue to matter instead of allowing a trivial same-order pairing.
#     - Within any repeated count class, the generator also forbids duplicate tooth-height
#       signatures. Matching is still by count+color, but this extra constraint ensures that a
#       wrong count-only pairing produces a measurably different output instead of accidentally
#       collapsing to the same result.
#     - Each frame's ordered color signature is unique, because the four source colors are all
#       distinct and groups are contiguous slices of those four columns.


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

H, W = 8, 23
FRAME_R0 = 0
FRAME_H = 8
LEFT_WALL_OPEN_ROWS = (4, 5)
FRAME_GAP = 1

BAR_R0 = 2
BAR_PATCH_H = 4
BAR_BASE_ROW = BAR_R0 + BAR_PATCH_H - 1
BAR_GAP = 1

ORIG_SLOT_LENS = [3, 4, 4, 3]
PARTITIONS = ((2, 2), (1, 1, 2), (2, 1, 1))
HEIGHT_RANGE = (2, 5)  # visible teeth only: {2,3,4}
MAX_ATTEMPTS = 300


def _frame_patch(ncols: int) -> np.ndarray:
    width = 2 * ncols + 1
    patch = np.zeros((FRAME_H, width), dtype=int)
    patch[[0, -1], :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    patch[list(LEFT_WALL_OPEN_ROWS), 0] = BG
    return patch


def _bar_patch(heights: tuple[int, ...]) -> np.ndarray:
    width = 2 * len(heights) - 1
    patch = np.zeros((BAR_PATCH_H, width), dtype=int)
    patch[-1, :] = GREY
    for i, height in enumerate(heights):
        patch[BAR_PATCH_H - int(height) :, 2 * i] = GREY
    return patch


def _paint_frame_columns(
    grid: np.ndarray,
    frame_c0: int,
    slot_lengths: tuple[int, ...],
    colors: tuple[int, ...],
    counts: tuple[int, ...],
    *,
    heights: tuple[int, ...] | None,
) -> None:
    for i, (length, color, count) in enumerate(zip(slot_lengths, colors, counts)):
        col = frame_c0 + 1 + 2 * i
        cap = int(length) - int(count)
        grid[1 : 1 + cap, col] = GREY
        if heights is None:
            grid[1 + cap : 1 + int(length), col] = int(color)
            continue

        height = int(heights[i])
        visible = min(int(count), BAR_BASE_ROW - height)
        if visible <= 0:
            continue
        row_end = BAR_BASE_ROW - height
        grid[row_end - visible + 1 : row_end + 1, col] = int(color)


def _paint_bar_tags(grid: np.ndarray, bar_c0: int, heights: tuple[int, ...], colors: tuple[int, ...]) -> None:
    for i, (height, color) in enumerate(zip(heights, colors)):
        tag_r = BAR_BASE_ROW - int(height)
        tag_c = bar_c0 + 2 * i
        grid[tag_r, tag_c] = int(color)


def _frame_kind(pts: np.ndarray) -> int | None:
    r0, c0, r1, c1 = _bbox(pts)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if r0 != FRAME_R0 or r1 != FRAME_R0 + FRAME_H - 1:
        return None
    if height != FRAME_H or width not in {3, 5}:
        return None

    ptset = {tuple(map(int, p)) for p in pts}
    if not all((r0, c) in ptset for c in range(c0, c1 + 1)):
        return None
    if not all((r1, c) in ptset for c in range(c0, c1 + 1)):
        return None
    if not all((r, c1) in ptset for r in range(r0, r1 + 1)):
        return None

    expected_left = {r0 + off for off in (0, 1, 2, 3, 6, 7)}
    actual_left = {r for r in range(r0, r1 + 1) if (r, c0) in ptset}
    if actual_left != expected_left:
        return None
    return (width - 1) // 2


def _bar_kind(pts: np.ndarray) -> int | None:
    r0, c0, r1, c1 = _bbox(pts)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if r1 != BAR_BASE_ROW or height > BAR_PATCH_H or width not in {1, 3}:
        return None

    ptset = {tuple(map(int, p)) for p in pts}
    if not all((r1, c) in ptset for c in range(c0, c1 + 1)):
        return None

    for r, c in ptset:
        if r == r1:
            continue
        if (c - c0) % 2 != 0:
            return None

    ncols = (width + 1) // 2
    top_by_col = {}
    for r, c in ptset:
        top_by_col[c] = min(top_by_col.get(c, r), r)
    proj_cols = [c for c, top in top_by_col.items() if top < r1]
    if len(proj_cols) != ncols:
        return None
    return ncols


def _frame_info(g: np.ndarray, bbox: tuple[int, int, int, int], bg: int, grey: int) -> dict:
    r0, c0, r1, c1 = bbox
    dst_cols = []
    color_sig = []
    colors_by_col = {}
    for c in range(c0 + 1, c1):
        rows = [r for r in range(r0, r1 + 1) if g[r, c] != bg and g[r, c] != grey]
        if not rows:
            continue
        dst_cols.append(int(c))
        colors = [int(g[r, c]) for r in rows]
        color_sig.append(colors[0])
        colors_by_col[int(c)] = colors
    return {
        "bbox": bbox,
        "ncols": len(dst_cols),
        "dst_cols": dst_cols,
        "color_sig": tuple(color_sig),
        "colors_by_col": colors_by_col,
    }


def _bar_info(g: np.ndarray, bbox: tuple[int, int, int, int], pts: np.ndarray, bg: int, grey: int) -> dict | None:
    _r0, c0, r1, _c1 = bbox
    col_min_row = {}
    for r, c in pts:
        rr = int(r)
        cc = int(c)
        col_min_row[cc] = min(col_min_row.get(cc, rr), rr)

    proj_cols = sorted([int(c) for c, rmin in col_min_row.items() if int(rmin) < r1])
    if not proj_cols:
        return None

    tag_sig = []
    for c in proj_cols:
        tag_r = int(col_min_row[c]) - 1
        if tag_r < 0:
            return None
        color = int(g[tag_r, c])
        if color in (bg, grey):
            return None
        tag_sig.append(color)

    return {
        "bbox": bbox,
        "pts": pts,
        "ncols": len(proj_cols),
        "proj_cols": proj_cols,
        "base_row": int(r1),
        "tag_sig": tuple(tag_sig),
        "src_c0": int(c0),
    }


def _detect_scene_objects(g: np.ndarray, bg: int, grey: int) -> tuple[list[dict], list[dict]]:
    frames = []
    bars = []
    for pts in _cc4(g == grey):
        bbox = _bbox(pts)
        n_frame = _frame_kind(pts)
        if n_frame is not None:
            info = _frame_info(g, bbox, bg, grey)
            if info["ncols"] == n_frame:
                frames.append(info)
            continue

        n_bar = _bar_kind(pts)
        if n_bar is not None:
            info = _bar_info(g, bbox, pts, bg, grey)
            if info is not None and info["ncols"] == n_bar:
                bars.append(info)

    frames.sort(key=lambda item: item["bbox"][1])
    bars.sort(key=lambda item: item["bbox"][1])
    return frames, bars


def _compute_dx(src_proj_cols: list[int], dst_cols: list[int], width: int) -> int:
    if len(src_proj_cols) == len(dst_cols):
        deltas = [d - s for s, d in zip(src_proj_cols, dst_cols)]
        if len(set(deltas)) == 1:
            return int(deltas[0])

    best = (-1, 0)
    dst_set = set(dst_cols)
    for cand_dx in range(-width, width + 1):
        score = sum((c + cand_dx) in dst_set for c in src_proj_cols)
        if score > best[0]:
            best = (score, cand_dx)
    return int(best[1])


def _pair_frames_and_bars(frames: list[dict], bars: list[dict]) -> list[tuple[dict, dict]] | None:
    frames_by_ncols: dict[int, list[dict]] = {}
    bars_by_ncols: dict[int, list[dict]] = {}
    for frame in frames:
        frames_by_ncols.setdefault(frame["ncols"], []).append(frame)
    for bar in bars:
        bars_by_ncols.setdefault(bar["ncols"], []).append(bar)

    if set(frames_by_ncols) != set(bars_by_ncols):
        return None

    pairs = []
    for ncols in sorted(frames_by_ncols):
        fgroup = frames_by_ncols[ncols]
        bgroup = bars_by_ncols.get(ncols, [])
        if len(fgroup) != len(bgroup):
            return None

        if len(fgroup) == 1:
            pairs.append((fgroup[0], bgroup[0]))
            continue

        by_sig: dict[tuple[int, ...], list[dict]] = {}
        for bar in bgroup:
            by_sig.setdefault(bar["tag_sig"], []).append(bar)
        for frame in fgroup:
            candidates = by_sig.get(frame["color_sig"], [])
            if not candidates:
                return None
            pairs.append((frame, candidates.pop(0)))

    return pairs


def _pairing_observability_metric(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    other = g[g != bg]
    if other.size == 0:
        return {"ambiguous_classes": 0, "ambiguous_frames": 0, "ambiguous_order_mismatch": 0}
    grey = _most_common(other)
    frames, bars = _detect_scene_objects(g, bg, grey)

    frames_by_ncols: dict[int, list[dict]] = {}
    bars_by_ncols: dict[int, list[dict]] = {}
    for frame in frames:
        frames_by_ncols.setdefault(frame["ncols"], []).append(frame)
    for bar in bars:
        bars_by_ncols.setdefault(bar["ncols"], []).append(bar)

    ambiguous_classes = 0
    ambiguous_frames = 0
    ambiguous_order_mismatch = 0
    for ncols, fgroup in frames_by_ncols.items():
        if len(fgroup) <= 1:
            continue
        ambiguous_classes += 1
        ambiguous_frames += len(fgroup)
        frame_seq = [tuple(f["color_sig"]) for f in sorted(fgroup, key=lambda item: item["bbox"][1])]
        bgroup = bars_by_ncols.get(ncols, [])
        bar_seq = [tuple(b["tag_sig"]) for b in sorted(bgroup, key=lambda item: item["bbox"][1])]
        if frame_seq != bar_seq:
            ambiguous_order_mismatch += 1

    return {
        "ambiguous_classes": ambiguous_classes,
        "ambiguous_frames": ambiguous_frames,
        "ambiguous_order_mismatch": ambiguous_order_mismatch,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move several tagged grey bars into several split grey frames, matching by tooth count first and tag colors when counts tie.

    Input structure:
    - Background plus multiple grey structural components: 2 or 3 full-height mini-frames on the right and the same number of source bars on the left.
    - Each mini-frame contains 1 or 2 colored columns; each bar has 1 or 2 visible teeth and a colored tag sitting directly above every tooth.

    Rule:
    - Detect every frame and every bar. Group them by visible count: number of frame columns versus number of bar teeth.
    - If a count group has only one frame and one bar, pair them directly. If several frames and bars share the same count, match them by the left-to-right sequence of colors: the frame's column colors must equal the bar's tooth-tag colors.
    - For each matched pair, place the bar into the frame so its base sits one cell above the frame floor, align the teeth with that frame's colored columns, and restack the original colored cells immediately above the placed teeth.

    Key insight:
    - This is no longer a single global alignment problem. The solver must first solve a visible N:N correspondence problem across several disconnected objects, using count as the coarse key and color tags as the tie-breaker, before applying the original insert-and-stack mechanic to each matched pair.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    other = g[g != bg]
    if other.size == 0:
        return g.copy()
    grey = _most_common(other)

    frames, bars = _detect_scene_objects(g, bg, grey)
    if not frames or len(frames) != len(bars):
        return g.copy()

    pairs = _pair_frames_and_bars(frames, bars)
    if pairs is None or len(pairs) != len(frames):
        return g.copy()

    out = np.full_like(g, bg)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = g[r0 : r1 + 1, c0 : c1 + 1]
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[(region != bg) & (region != grey)] = bg

    for frame, bar in pairs:
        sr0, sc0, sr1, sc1 = frame["bbox"]
        target_base_row = sr1 - 2
        dy = target_base_row - int(bar["base_row"])
        dx = _compute_dx(bar["proj_cols"], frame["dst_cols"], g.shape[1])

        placed = []
        for r, c in bar["pts"]:
            rr = int(r) + dy
            cc = int(c) + dx
            if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
                out[rr, cc] = grey
                placed.append((rr, cc))

        placed_by_col: dict[int, list[int]] = {}
        for rr, cc in placed:
            placed_by_col.setdefault(int(cc), []).append(int(rr))

        interior_top = sr0 + 1
        interior_bottom = sr1 - 1
        for col in frame["dst_cols"]:
            if col not in placed_by_col:
                continue
            bar_top = min(placed_by_col[col])
            desired_bottom = bar_top - 1
            if desired_bottom < interior_top:
                continue

            colors = list(frame["colors_by_col"][col])
            max_len = desired_bottom - interior_top + 1
            if max_len <= 0:
                continue
            if len(colors) > max_len:
                colors = colors[-max_len:]

            start_row = max(interior_top, desired_bottom - len(colors) + 1)
            for i, color in enumerate(colors):
                rr = start_row + i
                if interior_top <= rr <= interior_bottom:
                    out[rr, col] = int(color)

    return out


def _partition_groups(
    partition: tuple[int, ...],
    colors: list[int],
    counts: list[int],
    heights_by_group: list[tuple[int, ...]],
) -> list[dict]:
    groups = []
    idx = 0
    for size, heights in zip(partition, heights_by_group):
        groups.append(
            {
                "ncols": int(size),
                "slot_lengths": tuple(ORIG_SLOT_LENS[idx : idx + size]),
                "colors": tuple(int(v) for v in colors[idx : idx + size]),
                "counts": tuple(int(v) for v in counts[idx : idx + size]),
                "heights": tuple(int(v) for v in heights),
            }
        )
        idx += size
    return groups


def _frame_positions(groups: list[dict]) -> list[int]:
    widths = [2 * group["ncols"] + 1 for group in groups]
    total_width = sum(widths) + FRAME_GAP * (len(widths) - 1)
    start = W - total_width
    positions = []
    cur = start
    for width in widths:
        positions.append(cur)
        cur += width + FRAME_GAP
    return positions


def _bar_positions(bar_order: list[int], groups: list[dict]) -> list[int]:
    widths = [2 * groups[idx]["ncols"] - 1 for idx in bar_order]
    positions = []
    cur = 0
    for width in widths:
        positions.append(cur)
        cur += width + BAR_GAP
    return positions


def _order_requires_color(groups: list[dict], bar_order: list[int]) -> bool:
    by_ncols: dict[int, list[int]] = {}
    for idx, group in enumerate(groups):
        by_ncols.setdefault(group["ncols"], []).append(idx)
    for ncols, indices in by_ncols.items():
        if len(indices) <= 1:
            continue
        seen = [idx for idx in bar_order if groups[idx]["ncols"] == ncols]
        if seen == indices:
            return False
    return bar_order != list(range(len(groups)))


def _sample_bar_order(rng: np.random.Generator, groups: list[dict]) -> list[int]:
    candidates = []
    for perm in itertools.permutations(range(len(groups))):
        order = list(perm)
        if _order_requires_color(groups, order):
            candidates.append(order)
    return candidates[int(rng.integers(0, len(candidates)))]


def _ambiguous_groups_have_distinct_heights(groups: list[dict]) -> bool:
    by_ncols: dict[int, list[tuple[int, ...]]] = {}
    for group in groups:
        by_ncols.setdefault(group["ncols"], []).append(group["heights"])
    for heights_list in by_ncols.values():
        if len(heights_list) > 1 and len(set(heights_list)) != len(heights_list):
            return False
    return True


def _build_input(groups: list[dict], bar_order: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)

    frame_positions = _frame_positions(groups)
    for frame_c0, group in zip(frame_positions, groups):
        paste(grid, _frame_patch(group["ncols"]), FRAME_R0, frame_c0)
        _paint_frame_columns(
            grid,
            frame_c0,
            group["slot_lengths"],
            group["colors"],
            group["counts"],
            heights=None,
        )

    for idx, bar_c0 in zip(bar_order, _bar_positions(bar_order, groups)):
        group = groups[idx]
        paste(grid, _bar_patch(group["heights"]), BAR_R0, bar_c0, transparent=BG)
        _paint_bar_tags(grid, bar_c0, group["heights"], group["colors"])

    return grid


def _build_output(groups: list[dict]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)

    for frame_c0, group in zip(_frame_positions(groups), groups):
        paste(grid, _frame_patch(group["ncols"]), FRAME_R0, frame_c0)
        paste(grid, _bar_patch(group["heights"]), BAR_R0, frame_c0 + 1, transparent=BG)
        _paint_frame_columns(
            grid,
            frame_c0,
            group["slot_lengths"],
            group["colors"],
            group["counts"],
            heights=group["heights"],
        )

    return grid


def _sample_params(rng: np.random.Generator) -> tuple[list[dict], list[int], list[int]]:
    partition = PARTITIONS[int(rng.integers(0, len(PARTITIONS)))]
    colors = [int(v) for v in rng.choice(PALETTE, size=4, replace=False)]
    counts = [int(rng.integers(1, length + 1)) for length in ORIG_SLOT_LENS]
    heights_by_group = [
        tuple(int(rng.integers(*HEIGHT_RANGE)) for _ in range(size))
        for size in partition
    ]
    groups = _partition_groups(partition, colors, counts, heights_by_group)
    bar_order = _sample_bar_order(rng, groups)
    return groups, bar_order, colors


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    groups, bar_order, colors = _sample_params(rng)
    if not _ambiguous_groups_have_distinct_heights(groups):
        return None
    grid = _build_input(groups, bar_order)
    expected = _build_output(groups)

    metric = _pairing_observability_metric(grid)
    if metric["ambiguous_classes"] == 0 or metric["ambiguous_order_mismatch"] == 0:
        return None
    if get_bg(grid) != BG:
        return None
    if sorted(colors_present(grid, exclude=[BG, GREY])) != sorted(colors):
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
