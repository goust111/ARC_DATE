# Combo variant of 4c3d4a41 (index 28)
# combo_spec: D13/1+D17/1
# Name: 散落像素+双结构色

import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import bbox_coords, components_mask, get_bg as _bg_color
    from arc_utils import colors_present, get_bg, paste
except ModuleNotFoundError:  # Allow direct file import from the variant folder.
    sys.path.append(str(Path(__file__).resolve().parents[5] / "solvers"))
    from arc_utils import bbox_coords, components_mask, get_bg as _bg_color
    from arc_utils import colors_present, get_bg, paste


# Design analysis for the combo variant D13/1 + D17/1
# a) Concrete behavior trace:
#    - Base task: grey frame + grey bar, and the only non-structural colors are the four functional slot columns.
#      The original solver can treat "non-bg and non-grey inside the puzzle" as a single semantic class.
#    - Combo input: the frame stays grey, the movable bar becomes blue, and 2-3 isolated decorative pixels are scattered
#      outside the frame using colors that do not appear in the slot columns. Example:
#        original branch: most-common non-bg == grey -> two grey components -> clear everything outside frame in output.
#        combo branch: grey no longer identifies the bar, and clearing everything outside the frame would wrongly delete
#        the decorative pixels that must survive unchanged. The solver must therefore detect frame/bar by shape, keep
#        outside-frame colored singletons untouched, and restrict slot-column reasoning to the frame bbox only.
#    - In a concrete sample, a blue comb at cols 1-7 moves to cols 11-17, while three isolated colored pixels at
#      positions like (0,0), (6,4), (7,7) remain exactly where they were in both input and output.
# b) Complexity calibration:
#    - Only two visible changes are added and they cooperate cleanly: structural roles split by color, and non-functional
#      colored noise appears outside the working area.
#    - The noise is limited to 2-3 isolated single pixels chosen from colors disjoint from the four slot colors, so the
#      grid is visibly different from the base task without becoming cluttered or introducing extra object classes.
# c) Implicit-assumption audit of the original main():
#    - `grey = _most_common(other)` is invalid because the bar is no longer grey.
#    - Grouping only grey connected components loses the bar entirely.
#    - Building the output from a blank canvas and copying only the slot bbox would erase the decorative outside pixels.
#    - Treating every non-bg/non-grey pixel globally as a functional column would accidentally include the scattered noise.
#    - Painting the inserted bar using the frame color would destroy the new color-role split.
# d) Placement legality:
#    - Frame bbox stays fixed at rows 0-7, cols 9-19; input bar stays at rows 2-5, cols 1-7; noise candidates are chosen
#      from fixed cells outside both regions, so no overlap or out-of-bounds placement occurs.
#    - Output bar placement remains the original legal translation into cols 11-17 with base row 5, fully inside the frame.
# e) Visual observability:
#    - Across 2-4 examples, a human can see that the grey hollow structure is stationary, the blue comb is the moving part,
#      and some extra colored singletons outside the frame never participate in the transformation.
#    - Because the noise colors are disjoint from the slot colors, the solver's discrimination problem is visually inferable
#      rather than hidden in raw color ids.
# f) Visual cleanliness:
#    - Noise is sparse and isolated, never stacked on top of the bar or frame, so the main interaction zone remains readable.
#    - Inside the frame there are still only three layers of meaning: frame walls/caps, moved blue bar, and stacked column colors.
# g) Change inferability:
#    - No arbitrary recoloring is introduced. Frame color, bar color, slot colors, and decorative colors all persist.
#      The only rule is role-based: only colored cells inside the frame drive alignment and restacking; colored cells outside
#      the frame are inert and preserved.
# 9) New solver assumptions and generator guarantees:
#    - Assumption: exactly one component matches the open-frame geometry. Guaranteed by the fixed frame patch.
#    - Assumption: exactly one component matches the comb-like full-base bar geometry. Guaranteed by the fixed blue bar patch.
#    - Assumption: no decorative pixel is placed inside the frame bbox or inside the input bar bbox. Guaranteed by the
#      sampled noise positions.
#    - Assumption: decorative colors are disjoint from both structural colors and the four functional slot colors, so any
#      non-frame color inside the frame is functional. Guaranteed by sampling slot colors first and noise colors from the remainder.


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _local_mask(pts: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _is_frame_component(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 5 or w < 7:
        return False
    if not mask[0, :].all() or not mask[-1, :].all():
        return False
    if not mask[:, -1].all():
        return False
    if not mask[0, 0] or not mask[-1, 0]:
        return False
    if (~mask[1:-1, 0]).sum() == 0:
        return False
    return True


def _is_bar_component(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 2 or w < 5:
        return False
    if not mask[-1, :].all():
        return False
    if not mask[:-1, :].any():
        return False
    tall_cols = 0
    for c in range(w):
        rows = np.flatnonzero(mask[:, c])
        if rows.size == 0:
            return False
        top = int(rows[0])
        if not mask[top:, c].all():
            return False
        if top < h - 1:
            tall_cols += 1
    return tall_cols >= 2


def _find_frame_and_bar(g: np.ndarray, bg: int):
    comps = []
    for color in colors_present(g, exclude=[bg]):
        for pts in _cc4(g == color):
            bbox = _bbox(pts)
            comps.append(
                {
                    "color": int(color),
                    "pts": pts,
                    "bbox": bbox,
                    "size": int(len(pts)),
                    "mask": _local_mask(pts, bbox),
                }
            )

    frame_candidates = [rec for rec in comps if _is_frame_component(rec["mask"])]
    if not frame_candidates:
        return None, None
    frame_candidates.sort(key=lambda rec: (-rec["size"], rec["bbox"]))
    frame = frame_candidates[0]

    bar_candidates = [rec for rec in comps if rec is not frame and _is_bar_component(rec["mask"])]
    if not bar_candidates:
        return frame, None
    bar_candidates.sort(key=lambda rec: (-rec["size"], rec["bbox"]))
    return frame, bar_candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert a blue comb-like bar into a grey open frame, align its posts to the functional slot columns, and leave scattered outside pixels untouched.

    Input structure:
    - Background is 0; the right-hand open frame is grey, while the detached movable bar is blue.
    - Inside the frame, four colored vertical stacks mark the columns that the bar posts must match.
    - Outside the frame, a few isolated decorative colored pixels are present; they are not part of the mechanism and can share neither the frame color nor the bar color.

    Rule:
    - Detect the frame and the bar by geometry, not by assuming they share one structure color.
    - Move the blue bar into the frame so its base sits one row above the frame floor and its post columns align with the colored stacks inside the frame.
    - Restack each functional color column directly above the moved bar, but preserve every decorative outside-frame pixel exactly as it was.

    Key insight:
    - The critical separation is by role and location: colored cells inside the frame control alignment and stacking, while colored cells outside the frame are inert noise that must simply survive.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)
    frame, bar = _find_frame_and_bar(g, bg)
    if frame is None or bar is None:
        return g.copy()

    frame_color = frame["color"]
    bar_color = bar["color"]
    sr0, sc0, sr1, sc1 = frame["bbox"]
    bar_pts = bar["pts"]

    out = g.copy()
    for r, c in bar_pts:
        out[int(r), int(c)] = bg

    slot_region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region[(slot_region != bg) & (slot_region != frame_color)] = bg

    bar_base_row = max(int(r) for r, _ in bar_pts)
    target_base_row = (sr1 - 1) - 1
    dy = target_base_row - bar_base_row

    col_min_row = {}
    for r, c in bar_pts:
        r = int(r)
        c = int(c)
        col_min_row[c] = min(col_min_row.get(c, r), r)
    src_proj_cols = sorted(c for c, rmin in col_min_row.items() if rmin < bar_base_row)

    dst_cols_set = set()
    for r, c in np.argwhere((g != bg) & (g != frame_color)):
        r = int(r)
        c = int(c)
        if sr0 <= r <= sr1 and sc0 <= c <= sc1:
            dst_cols_set.add(c)
    dst_cols = sorted(dst_cols_set)

    dx = None
    if src_proj_cols and len(src_proj_cols) == len(dst_cols):
        deltas = [d - s for s, d in zip(src_proj_cols, dst_cols)]
        if len(set(deltas)) == 1:
            dx = deltas[0]
    if dx is None:
        best = (-1, 0)
        dst_set = set(dst_cols)
        for cand_dx in range(-w, w + 1):
            score = sum((c + cand_dx) in dst_set for c in src_proj_cols)
            if score > best[0]:
                best = (score, cand_dx)
        dx = best[1]

    placed = []
    for r, c in bar_pts:
        rr = int(r) + dy
        cc = int(c) + dx
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = bar_color
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

        rows = [r for r in range(sr0, sr1 + 1) if g[r, col] != bg and g[r, col] != frame_color]
        if not rows:
            continue
        colors = [int(g[r, col]) for r in rows]

        max_len = desired_bottom - interior_top + 1
        if max_len <= 0:
            continue
        if len(colors) > max_len:
            colors = colors[-max_len:]

        start_row = max(desired_bottom - len(colors) + 1, interior_top)
        for i, colv in enumerate(colors):
            rr = start_row + i
            if interior_top <= rr <= interior_bottom:
                out[rr, col] = colv

    return out


BG = 0
BAR_COLOR = 1
FRAME_COLOR = 5
PALETTE = np.array([2, 3, 4, 6, 7, 8, 9], dtype=int)

H, W = 8, 20

FRAME_R0, FRAME_C0 = 0, 9
FRAME_H, FRAME_W = 8, 11
LEFT_WALL_OPEN_ROWS = (4, 5)

SLOT_COLS = [11, 13, 15, 17]
SLOT_LENS = [3, 4, 4, 3]
SLOT_TOP_ROW = 1

BAR_INPUT_C0 = 1
BAR_OUTPUT_C0 = 11
BAR_R0 = 2
BAR_H, BAR_W = 4, 7
BAR_LOCAL_PROJ_COLS = [0, 2, 4, 6]
BAR_BASE_ABS_ROW = BAR_R0 + BAR_H - 1

NOISE_CANDIDATES = [(0, 0), (0, 4), (1, 7), (3, 8), (6, 0), (6, 4), (7, 7)]

HEIGHT0_RANGE = (2, 5)
HEIGHT_REST_RANGE = (1, 5)
MAX_ATTEMPTS = 300


def _frame_patch() -> np.ndarray:
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[[0, -1], :] = FRAME_COLOR
    patch[:, -1] = FRAME_COLOR
    patch[:, 0] = FRAME_COLOR
    patch[list(LEFT_WALL_OPEN_ROWS), 0] = BG
    return patch


def _bar_patch(heights: list[int]) -> np.ndarray:
    patch = np.zeros((BAR_H, BAR_W), dtype=int)
    patch[-1, :] = BAR_COLOR
    for local_c, height in zip(BAR_LOCAL_PROJ_COLS, heights):
        patch[BAR_H - height : BAR_H, local_c] = BAR_COLOR
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
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = FRAME_COLOR
    if height is None:
        grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + length, col] = color
        return
    visible = min(count, BAR_BASE_ABS_ROW - height)
    if visible > 0:
        row_end = BAR_BASE_ABS_ROW - height
        grid[row_end - visible + 1 : row_end + 1, col] = color


def _paint_noise(grid: np.ndarray, noise_pixels: list[tuple[int, int, int]]) -> None:
    for r, c, color in noise_pixels:
        grid[int(r), int(c)] = int(color)


def _build_grid(
    heights: list[int],
    slot_colors: list[int],
    colored_counts: list[int],
    noise_pixels: list[tuple[int, int, int]],
    bar_c0: int,
    output_mode: bool,
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    if not output_mode:
        paste(grid, _bar_patch(heights), BAR_R0, bar_c0, transparent=BG)
    for col, length, color, count, height in zip(SLOT_COLS, SLOT_LENS, slot_colors, colored_counts, heights):
        _paint_slot_column(grid, col, length, color, count, height if output_mode else None)
    if output_mode:
        paste(grid, _bar_patch(heights), BAR_R0, bar_c0, transparent=BG)
    _paint_noise(grid, noise_pixels)
    return grid


def _sample_params(
    rng: np.random.Generator,
) -> tuple[list[int], list[int], list[int], list[tuple[int, int, int]]]:
    slot_colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]
    # Balance the visible bar-shape branches directly instead of relying on reject sampling.
    tall_count = int(rng.choice([2, 3, 4]))
    tall_indices = {0}
    if tall_count > 1:
        extra = rng.choice(np.arange(1, len(SLOT_COLS)), size=tall_count - 1, replace=False)
        tall_indices.update(int(idx) for idx in np.atleast_1d(extra))
    heights = []
    for idx in range(len(SLOT_COLS)):
        if idx == 0 or idx in tall_indices:
            heights.append(int(rng.integers(*HEIGHT0_RANGE if idx == 0 else HEIGHT_REST_RANGE)))
            if heights[-1] == 1:
                heights[-1] = 2
        else:
            heights.append(1)

    remaining = [int(v) for v in PALETTE if int(v) not in slot_colors]
    rng.shuffle(remaining)
    noise_count = int(rng.integers(2, min(4, len(remaining) + 1)))
    positions_idx = rng.choice(len(NOISE_CANDIDATES), size=noise_count, replace=False)
    noise_pixels = [
        (NOISE_CANDIDATES[int(pos_idx)][0], NOISE_CANDIDATES[int(pos_idx)][1], remaining[i])
        for i, pos_idx in enumerate(positions_idx)
    ]
    return heights, slot_colors, colored_counts, noise_pixels


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, slot_colors, colored_counts, noise_pixels = _sample_params(rng)
    grid = _build_grid(heights, slot_colors, colored_counts, noise_pixels, BAR_INPUT_C0, output_mode=False)
    expected = _build_grid(heights, slot_colors, colored_counts, noise_pixels, BAR_OUTPUT_C0, output_mode=True)

    if get_bg(grid) != BG:
        return None
    if BAR_COLOR in slot_colors or FRAME_COLOR in slot_colors:
        return None
    noise_colors = [color for _, _, color in noise_pixels]
    if set(noise_colors) & set(slot_colors):
        return None
    if sorted(colors_present(grid, exclude=[BG, FRAME_COLOR, BAR_COLOR])) != sorted(slot_colors + noise_colors):
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
