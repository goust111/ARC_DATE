# Combo variant of 4c3d4a41 (index 29)
# combo_spec: D1/0+D8/2+D13/2
# Name: 颜色匹配+多余凸起+双杆

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, get_bg, paste
from arc_utils import get_bg as _bg_color


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


# Design analysis
# a) Concrete behavior and trace:
#    - The source task has one grey frame on the right, one grey bar on the left, four target slot
#      columns in the frame, and a direct "align four protrusions to four columns" move.
#    - The fused task keeps the same frame-and-stacking end effect but changes the evidence chain:
#      there are now two candidate bars, each bar has 5 or 6 colored protrusions, and the frame still
#      has only 4 colored slot columns.
#    - The intended bar contains exactly one consecutive 4-protrusion color window whose left->right
#      colors equal the frame's 4 slot-column colors. Any remaining protrusions are edge extras with
#      non-frame colors. The other bar is a distractor with 5 or 6 protrusions too, but its best
#      4-protrusion window only matches 2-3 frame colors in the correct positions.
#    - Trace:
#        old task: choose the only bar -> align all protrusions by absolute columns -> stack colors.
#        fused task: compare both bars by best 4-window color score -> keep the winning bar ->
#                    within it isolate the winning 4-window -> translate only that 7-cell core into
#                    the frame -> stack the frame colors above those 4 matched protrusions.
#      The visible difference is that both bars look plausible at a glance, both have extra teeth,
#      and the useful 4-tooth block can start at different offsets inside the chosen bar.
#
# b) Complexity calibration:
#    - The frame still has exactly 4 slot columns, so the destination side stays simple.
#    - The bar side adds only two bounded freedoms: 5 vs 6 protrusions, and the active 4-window can
#      start at one of 2-3 positions. That is enough to force real selection without creating a huge
#      combinatorial search space.
#    - The distractor is constrained to a best score of 2 or 3, not 0, so "compare color-match
#      quality" is visible across examples rather than collapsing into "one obviously unrelated bar".
#
# c) Old-assumption audit:
#    - "Second-largest grey component is the bar" breaks twice: there are now two bars, and the bars
#      are no longer grey-only because their protrusions are colored. The solver must inspect
#      non-background connected components, detect the frame by geometry, and score multiple bars.
#    - "All protrusions are used" breaks because each chosen bar has 5 or 6 protrusions but only 4
#      can map to the frame. The solver must crop to the matched 4-protrusion core.
#    - "Absolute column order alone determines dx" breaks because both bars share the same spacing and
#      extra protrusions create multiple geometric windows. The decisive cue is the color sequence.
#    - New fused assumption: the useful 4 protrusions form one consecutive block. The generator keeps
#      extras on the ends so that cropping to the matched window is well defined and visually clean.
#
# d) Placement legality:
#    - The canvas widens to 8x36 so two bars of width 9 or 11 fit at cols 1 and 13 while remaining
#      disconnected from each other and from the frame at cols 25..35.
#    - The frame keeps the original 8x11 shell, so the output core still fits entirely inside the
#      frame interior when aligned to slot cols 27,29,31,33.
#    - All protrusion heights are sampled in [2,4], ensuring each protrusion has at least one visible
#      colored cell above the grey base and never grows out of the 4-row bar patch.
#
# e) Visual observability:
#    - A human sees one frame with 4 distinct slot colors and two separate bars whose protrusions are
#      also color coded. The correct bar is the one that visibly contains the frame's 4-color sequence
#      in order; the wrong bar has only a partial ordered match.
#    - Because the active 4-window is not always the leftmost one, examples show that the solver must
#      identify which protrusions matter, not just count protrusions or always drop the right edge.
#
# f) Visual cleanliness:
#    - Extra protrusions are confined to bar edges, so the input stays readable and the output remains
#      a compact 7-cell inserted core instead of an 11-cell cluttered bar inside the frame.
#    - The frame area is unchanged except for the same style of color stacking as in the source task,
#      so the added complexity stays localized to the bar-selection region.
#
# g) Color-inference constraints:
#    - The 4 frame slot colors are distinct. The correct bar's active 4-window repeats those exact
#      colors in order, while its extra protrusions use colors outside that 4-color set.
#    - The distractor bar is constructed so its best ordered 4-window has score 2 or 3 against the
#      frame colors. This gives repeated evidence for "score by ordered color agreement" without
#      requiring hidden color-id arithmetic.
#
# h) Generator guarantees for solver assumptions:
#    - Exactly one candidate bar has a consecutive 4-protrusion window with perfect ordered color
#      match to the 4 slot columns.
#    - The other candidate bar has no perfect window and its best score is 2 or 3.
#    - The chosen bar has 5 or 6 protrusions, and the winning window start varies across samples.
#
# i) Post-change assumption audit for main():
#    - Frame detection assumes one component has a uniquely taller bbox than every bar; the generator
#      preserves frame height 8 versus bar height 4.
#    - Slot extraction assumes each target slot column is monochrome in the input; generation keeps the
#      original contiguous single-color column stacks.
#    - Bar scoring assumes protrusions are evenly spaced enough for a constant-dx 4-window test; both
#      bars are generated on the same 2-column rhythm.
#    - Cropping assumes the winning 4-window is consecutive; the generator places extras only on the
#      left edge, right edge, or both edges of that window.


def _slot_columns(g: np.ndarray, bg: int, grey: int, bbox: tuple[int, int, int, int]):
    sr0, sc0, sr1, sc1 = bbox
    info = []
    for c in range(sc0, sc1 + 1):
        slot_rows = [r for r in range(sr0 + 1, sr1) if int(g[r, c]) != bg]
        if not slot_rows:
            continue
        rows = [r for r in range(sr0, sr1 + 1) if int(g[r, c]) != bg and int(g[r, c]) != grey]
        if not rows:
            continue
        colors = {int(g[r, c]) for r in rows}
        if len(colors) != 1:
            continue
        info.append((int(c), int(next(iter(colors))), rows, slot_rows))
    info.sort(key=lambda t: t[0])
    return info


def _bar_info(g: np.ndarray, bg: int, grey: int, pts):
    base_row = max(int(r) for r, _ in pts)
    cols: dict[int, list[int]] = {}
    for r, c in pts:
        cols.setdefault(int(c), []).append(int(r))

    proj_cols: list[int] = []
    proj_colors: list[int] = []
    for c in sorted(cols):
        rows = sorted(cols[c])
        if rows[0] >= base_row:
            continue
        above_colors = [
            int(g[r, c])
            for r in rows
            if r < base_row and int(g[r, c]) != bg and int(g[r, c]) != grey
        ]
        if not above_colors:
            continue
        proj_cols.append(int(c))
        proj_colors.append(_most_common(above_colors))

    return {
        "pts": pts,
        "bbox": _bbox(pts),
        "base_row": int(base_row),
        "proj_cols": proj_cols,
        "proj_colors": proj_colors,
    }


def _best_window_match(
    proj_cols: list[int],
    proj_colors: list[int],
    dst_cols: list[int],
    dst_colors: list[int],
):
    best = None
    need = len(dst_cols)
    if len(proj_cols) < need:
        return None

    for start in range(len(proj_cols) - need + 1):
        block_cols = proj_cols[start : start + need]
        block_colors = proj_colors[start : start + need]
        deltas = [dst - src for src, dst in zip(block_cols, dst_cols)]
        consistent = len(set(deltas)) == 1
        score = sum(int(a == b) for a, b in zip(block_colors, dst_colors)) if consistent else -1
        exact = consistent and score == need
        key = (1 if exact else 0, score, 1 if consistent else 0, -start)
        if best is None or key > best["key"]:
            best = {
                "start": int(start),
                "end": int(start + need - 1),
                "dx": int(deltas[0]) if consistent else 0,
                "score": int(score),
                "consistent": bool(consistent),
                "exact": bool(exact),
                "key": key,
            }
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pick the better-matching colored bar, keep only its useful 4-protrusion core, and insert it into the frame.

    Input structure:
    - Background plus one right-side grey frame and two left-side candidate bars, all as disconnected non-background components.
    - The frame contains exactly four colored slot columns, while each bar has a grey base and five or six upward colored protrusions.

    Rule:
    - Read the four frame-column colors from left to right, then score each bar by the best consecutive four-protrusion window whose columns can translate onto the frame columns.
    - Choose the bar with the higher ordered color-match score, keep only that winning four-protrusion window, and move the cropped 7-cell core upward into the frame so its base sits one cell above the interior floor.
    - Preserve the frame, erase both input bars, and restack each frame column's colored cells immediately above the aligned protrusion in the inserted core.

    Key insight:
    - Geometry alone is no longer enough: both candidate bars have extra protrusions, so the solver must use ordered color agreement to choose the correct bar and then crop away the unmatched edge protrusions.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    other = g[g != bg]
    if other.size == 0:
        return g.copy()
    grey = _most_common(other)

    comps = [(len(pts), _bbox(pts), pts) for pts in _cc4(g != bg)]
    if len(comps) < 3:
        return g.copy()

    def frame_key(item):
        size, bbox, _pts = item
        r0, c0, r1, c1 = bbox
        return (r1 - r0 + 1, c1 - c0 + 1, size, c0)

    frame_entry = max(comps, key=frame_key)
    _frame_size, frame_bbox, _frame_pts = frame_entry
    sr0, sc0, sr1, sc1 = frame_bbox

    out = np.full_like(g, bg)
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region[(slot_region != bg) & (slot_region != grey)] = bg

    slot_info = _slot_columns(g, bg, grey, frame_bbox)
    if len(slot_info) != 4:
        return out
    dst_cols = [col for col, _color, _rows, _slot_rows in slot_info]
    dst_colors = [color for _col, color, _rows, _slot_rows in slot_info]

    candidate_entries = [item for item in comps if item != frame_entry and item[1][3] < sc0]
    best_choice = None
    best_key = None
    for _size, bbox, pts in candidate_entries:
        r0, c0, r1, c1 = bbox
        if r1 - r0 + 1 < 2 or c1 - c0 + 1 < 7:
            continue
        info = _bar_info(g, bg, grey, pts)
        match = _best_window_match(info["proj_cols"], info["proj_colors"], dst_cols, dst_colors)
        if match is None:
            continue
        key = (
            1 if match["exact"] else 0,
            match["score"],
            1 if match["consistent"] else 0,
            -abs(len(info["proj_cols"]) - len(dst_cols) - 1),
            -abs(len(info["proj_cols"]) - len(dst_cols) - 2),
            -match["start"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best_choice = (info, match)

    if best_choice is None:
        return out

    info, match = best_choice
    bar_pts = info["pts"]
    bar_base_row = info["base_row"]
    keep_c0 = info["proj_cols"][match["start"]]
    keep_c1 = info["proj_cols"][match["end"]]
    dx = match["dx"]
    dy = (sr1 - 2) - bar_base_row

    placed: list[tuple[int, int]] = []
    for r, c in bar_pts:
        rr = int(r) + dy
        cc = int(c) + dx
        if not (keep_c0 <= int(c) <= keep_c1):
            continue
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = int(g[r, c])
            placed.append((rr, cc))

    placed_by_col: dict[int, list[int]] = {}
    for rr, cc in placed:
        placed_by_col.setdefault(int(cc), []).append(int(rr))

    interior_top = sr0 + 1
    interior_bottom = sr1 - 1
    for col, _slot_color, rows, slot_rows in slot_info:
        if col not in placed_by_col:
            continue
        bar_top = min(placed_by_col[col])
        desired_bottom = bar_top - 1
        if desired_bottom < interior_top:
            continue

        colors = [int(g[r, col]) for r in rows]
        cap_len = len(slot_rows) - len(rows)
        for rr in slot_rows[:cap_len]:
            out[rr, col] = grey

        max_len = desired_bottom - interior_top + 1
        if max_len <= 0:
            continue
        if len(colors) > max_len:
            colors = colors[-max_len:]

        start_row = max(desired_bottom - len(colors) + 1, interior_top)
        for i, color in enumerate(colors):
            rr = start_row + i
            if interior_top <= rr <= interior_bottom:
                out[rr, col] = int(color)

    return out


# --------------- generator ---------------

BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

H, W = 8, 36

FRAME_R0, FRAME_C0 = 0, 25
FRAME_H, FRAME_W = 8, 11
LEFT_WALL_OPEN_ROWS = (4, 5)

SLOT_COLS = [27, 29, 31, 33]
SLOT_LENS = [3, 4, 4, 3]
SLOT_TOP_ROW = 1

BAR_R0 = 2
BAR_H = 4
BAR_INPUT_C0S = [1, 13]
BAR_OUTPUT_C0 = SLOT_COLS[0]
BAR_BASE_ABS_ROW = BAR_R0 + BAR_H - 1

PROJ_COUNT_CHOICES = (5, 6)
HEIGHT_RANGE = (2, 5)
MAX_ATTEMPTS = 400


def _frame_patch() -> np.ndarray:
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[[0, -1], :] = GREY
    patch[:, -1] = GREY
    patch[:, 0] = GREY
    patch[list(LEFT_WALL_OPEN_ROWS), 0] = BG
    return patch


def _proj_local_cols(count: int) -> list[int]:
    return [2 * i for i in range(count)]


def _bar_patch(heights: list[int], proj_colors: list[int]) -> np.ndarray:
    local_cols = _proj_local_cols(len(heights))
    width = local_cols[-1] + 1
    patch = np.zeros((BAR_H, width), dtype=int)
    patch[-1, :] = GREY
    for local_c, height, color in zip(local_cols, heights, proj_colors):
        patch[BAR_H - height : BAR_H - 1, local_c] = int(color)
    return patch


def _paint_slot_column(
    grid: np.ndarray,
    col: int,
    length: int,
    color: int,
    count: int,
    height: int | None,
) -> None:
    cap = length - count
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    if height is None:
        grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + length, col] = int(color)
        return

    visible = min(count, BAR_BASE_ABS_ROW - height)
    if visible > 0:
        row_end = BAR_BASE_ABS_ROW - height
        grid[row_end - visible + 1 : row_end + 1, col] = int(color)


def _core_patch(active_heights: list[int], active_colors: list[int]) -> np.ndarray:
    patch = np.zeros((BAR_H, 7), dtype=int)
    patch[-1, :] = GREY
    for local_c, height, color in zip([0, 2, 4, 6], active_heights, active_colors):
        patch[BAR_H - height : BAR_H - 1, local_c] = int(color)
    return patch


def _best_window_score(seq: list[int], target: list[int]) -> int:
    best = -1
    need = len(target)
    for start in range(len(seq) - need + 1):
        block = seq[start : start + need]
        best = max(best, sum(int(a == b) for a, b in zip(block, target)))
    return best


def _sample_correct_bar(
    rng: np.random.Generator,
    slot_colors: list[int],
) -> tuple[list[int], list[int], int]:
    count = int(rng.choice(PROJ_COUNT_CHOICES))
    start = int(rng.integers(0, count - 3))
    heights = [int(rng.integers(*HEIGHT_RANGE)) for _ in range(count)]
    colors = [None] * count
    for i, color in enumerate(slot_colors):
        colors[start + i] = int(color)

    extra_colors = list(rng.choice([c for c in PALETTE if int(c) not in slot_colors], size=count - 4, replace=False))
    extra_iter = iter(int(c) for c in extra_colors)
    for i in range(count):
        if colors[i] is None:
            colors[i] = next(extra_iter)

    return heights, [int(c) for c in colors], start


def _sample_distractor_bar(
    rng: np.random.Generator,
    slot_colors: list[int],
) -> tuple[list[int], list[int]]:
    extras = [int(c) for c in PALETTE if int(c) not in slot_colors]
    for _ in range(200):
        count = int(rng.choice(PROJ_COUNT_CHOICES))
        start = int(rng.integers(0, count - 3))
        target_score = int(rng.choice([2, 3]))

        block = list(slot_colors)
        mismatch_count = 4 - target_score
        mismatch_pos = rng.choice(4, size=mismatch_count, replace=False)
        replacement_colors = list(rng.choice(extras, size=mismatch_count, replace=False))
        for idx, repl in zip(mismatch_pos, replacement_colors):
            block[int(idx)] = int(repl)

        colors: list[int | None] = [None] * count
        for i, color in enumerate(block):
            colors[start + i] = int(color)

        used = {int(c) for c in block}
        fill_pool = [int(c) for c in PALETTE if int(c) not in used]
        if len(fill_pool) < count - 4:
            continue
        fillers = list(rng.choice(fill_pool, size=count - 4, replace=False))
        filler_iter = iter(int(c) for c in fillers)
        for i in range(count):
            if colors[i] is None:
                colors[i] = next(filler_iter)

        seq = [int(c) for c in colors]
        best_score = _best_window_score(seq, slot_colors)
        if 2 <= best_score <= 3:
            heights = [int(rng.integers(*HEIGHT_RANGE)) for _ in range(count)]
            return heights, seq

    raise RuntimeError("failed to sample distractor bar")


def _build_input(
    slot_colors: list[int],
    colored_counts: list[int],
    correct_bar_heights: list[int],
    correct_bar_colors: list[int],
    distractor_heights: list[int],
    distractor_colors: list[int],
    correct_pos_index: int,
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    for col, length, color, count in zip(SLOT_COLS, SLOT_LENS, slot_colors, colored_counts):
        _paint_slot_column(grid, col, length, color, count, None)

    paste(grid, _bar_patch(correct_bar_heights, correct_bar_colors), BAR_R0, BAR_INPUT_C0S[correct_pos_index])
    paste(
        grid,
        _bar_patch(distractor_heights, distractor_colors),
        BAR_R0,
        BAR_INPUT_C0S[1 - correct_pos_index],
    )
    return grid


def _build_expected(
    slot_colors: list[int],
    colored_counts: list[int],
    active_heights: list[int],
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _core_patch(active_heights, slot_colors), BAR_R0, BAR_OUTPUT_C0)
    for col, length, color, count, height in zip(SLOT_COLS, SLOT_LENS, slot_colors, colored_counts, active_heights):
        _paint_slot_column(grid, col, length, color, count, height)
    return grid


def _sample_params(rng: np.random.Generator):
    slot_colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]

    correct_bar_heights, correct_bar_colors, start = _sample_correct_bar(rng, slot_colors)
    active_heights = correct_bar_heights[start : start + 4]

    distractor_heights, distractor_colors = _sample_distractor_bar(rng, slot_colors)
    correct_pos_index = int(rng.integers(0, 2))

    return (
        slot_colors,
        colored_counts,
        correct_bar_heights,
        correct_bar_colors,
        active_heights,
        distractor_heights,
        distractor_colors,
        correct_pos_index,
        start,
    )


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    (
        slot_colors,
        colored_counts,
        correct_bar_heights,
        correct_bar_colors,
        active_heights,
        distractor_heights,
        distractor_colors,
        correct_pos_index,
        _start,
    ) = _sample_params(rng)

    grid = _build_input(
        slot_colors,
        colored_counts,
        correct_bar_heights,
        correct_bar_colors,
        distractor_heights,
        distractor_colors,
        correct_pos_index,
    )
    expected = _build_expected(slot_colors, colored_counts, active_heights)

    if get_bg(grid) != BG:
        return None
    if len(colors_present(grid, exclude=[BG])) < 6:
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
