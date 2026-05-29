# Combo variant of 4c3d4a41 (index 11)
# combo_spec: D1/2+D2/0
# Name: 轮廓互补匹配+向下凸起

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg as _bg_color
from arc_utils import colors_present, get_bg, paste

# Design analysis for the combo variant
# a) Concrete behavior and trace
#    - Keep the original macro-task "insert a grey bar into a grey frame and re-place the colors",
#      but combine two changes at once:
#      1) the bar is vertically inverted, so its full base is the TOP row and the four odd-column
#         teeth point downward into a top opening;
#      2) colors no longer stay in the same absolute column, but instead follow a complementary
#         contour match between input colored heights and the free spaces below the teeth.
#    - Supportive geometry change: all four frame columns are length 4. This is necessary because
#      the complementary levels must realize the four distinct values {1,2,3,4}; with outer columns
#      of length 3, one level would be missing and the shape match would become ambiguous.
#    - Small trace: if downward tooth heights left->right are [3,1,4,2], then the free spaces below
#      them are [2,4,1,3]. If the input colored heights on frame cols [11,13,15,17] are [4,2,3,1],
#      then colors move to output cols [13,11,17,15] respectively. After insertion, each matched
#      color stack hangs immediately below its tooth instead of above it.
# b) Complexity calibration
#    - The generator uses a permutation of tooth heights [1,2,3,4], so the complementary free-space
#      profile is also a permutation of [4,3,2,1]. The input colored heights are a derangement of
#      that target profile, guaranteeing a visible non-identity remapping.
#    - This keeps the rule human-readable: there are only four contour levels, each used exactly once,
#      so the lock-and-key correspondence is unambiguous without introducing extra symbols.
# c) Hidden assumptions audited
#    - The original solver assumed the bar base was on the lowest row (`max(r)`). Here the base is the
#      highest row (`min(r)`), so translation and tooth-bottom detection must both flip.
#    - The original solver also assumed output attachment happens above each tooth. The combo variant
#      instead needs the available space below each tooth: `interior_bottom - tooth_bottom`.
#    - The original same-column coloring loop is invalid. We must read each input column's colored
#      height, match that height to exactly one free-space value, and repaint at the matched output
#      column. The generator guarantees those four heights are unique so the bijection exists.
#    - Grey caps inside the input frame are no longer meaningful after remapping; they are treated as
#      input-only cues. The solver therefore reconstructs only the frame skeleton from the largest grey
#      component, then repaints bar and colors from scratch inside it.
# d) Placement legality
#    - Canvas stays 8x20. The frame bbox remains rows 0..7, cols 9..19 with a top opening across
#      absolute cols 11..17. The input bar bbox is rows 0..3, cols 1..7; the inserted bar bbox is
#      rows 2..5, cols 11..17.
#    - With tooth heights [1,2,3,4], the free spaces below the teeth are [4,3,2,1], so every matched
#      color stack fits exactly inside interior rows 3..6, 4..6, 5..6, or 6..6. No overlaps occur
#      because each destination column is unique.
# e) Visual observability
#    - Across examples, a human sees three consistent cues together: the frame opens on top, the bar
#      teeth point downward, and the colored-height profile in the input is a permuted complement of
#      the tooth-depth profile. The output makes the remapping visible because every color shifts to
#      the tooth with matching complementary clearance.
#    - No rule depends on numeric color ids. Only visible bar shape, visible colored stack heights, and
#      their spatial relation drive the transformation.
# f) Visual cleanliness
#    - The frame interior still uses sparse odd columns [11,13,15,17], leaving blank even columns that
#      separate the four teeth and four color stacks. Output contains only the frame border, the inserted
#      bar, and one solid color stack per destination column, so the scene remains legible.
# g) New assumptions and generator guarantees
#    - The solver assumes four tooth columns spaced every other column, a top-open frame, four colored
#      input columns with unique heights 1..4, and a unique complementary match from heights to spaces.
#    - The generator enforces those assumptions by sampling a non-monotone permutation of tooth heights,
#      deriving target free spaces as `5 - height`, and choosing the input colored heights as a
#      derangement of those four target values.


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert a downward-toothed grey bar into a top-open grey frame by complementary contour match, then hang each color stack below its matched tooth.

    Input structure:
    - A grey structural color forms two main components: a large frame on the right whose opening is on the top edge, and a smaller horizontal bar on the left whose full base is its top row and whose four teeth extend downward.
    - Inside the frame, four colored columns sit on fixed odd columns with possible grey caps above them; the visible colored heights define a lock-like profile.

    Rule:
    - Move the bar down into the frame so its top base row sits one cell below the interior ceiling, keeping the same four tooth columns aligned with the slot columns.
    - Ignore absolute color-column identity. Instead, match each input colored height to the unique tooth whose free space below equals that height.
    - Repaint that column's full color stack immediately below the matched tooth.

    Key insight: The decisive relation is a complementary contour in the downward direction: deeper teeth leave shorter hanging space, shallower teeth leave taller hanging space, and colors follow that lock-and-key fit rather than their original x-position.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape

    bg = _bg_color(g)
    other = g[g != bg]
    if other.size == 0:
        return g.copy()
    grey = _most_common(other)

    def comps_of_color(color: int):
        comps = [(len(pts), _bbox(pts), pts) for pts in _cc4(g == color)]
        comps.sort(key=lambda t: (-t[0], t[1]))
        return comps

    grey_comps = comps_of_color(grey)
    if len(grey_comps) < 2:
        return g.copy()

    (_, slot_bbox, slot_pts), (_, _bar_bbox, bar_pts) = grey_comps[0], grey_comps[1]
    sr0, sc0, sr1, sc1 = slot_bbox
    interior_top = sr0 + 1
    interior_bottom = sr1 - 1

    out = np.full_like(g, bg)
    for r, c in slot_pts:
        out[int(r), int(c)] = grey

    bar_base_row = min(int(r) for r, _ in bar_pts)
    target_base_row = interior_top + 1
    dy = target_base_row - bar_base_row

    bar_cols = sorted({int(c) for _, c in bar_pts})
    src_tooth_cols = bar_cols[::2]

    slot_info = []
    for c in range(sc0, sc1 + 1):
        rows = [r for r in range(sr0, sr1 + 1) if g[r, c] != bg and g[r, c] != grey]
        if not rows:
            continue
        colors = [int(g[r, c]) for r in rows]
        slot_info.append({"col": int(c), "count": len(rows), "colors": colors})

    dst_cols = sorted(info["col"] for info in slot_info)
    if not dst_cols or not src_tooth_cols:
        return out

    dx = None
    if len(src_tooth_cols) == len(dst_cols):
        deltas = [d - s for s, d in zip(src_tooth_cols, dst_cols)]
        if len(set(deltas)) == 1:
            dx = deltas[0]
    if dx is None:
        best = (-1, 0)
        dst_set = set(dst_cols)
        for cand_dx in range(-w, w + 1):
            score = sum((c + cand_dx) in dst_set for c in src_tooth_cols)
            if score > best[0]:
                best = (score, cand_dx)
        dx = best[1]

    placed = []
    col_max_row = {}
    for r, c in bar_pts:
        rr = int(r) + dy
        cc = int(c) + dx
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = grey
            placed.append((rr, cc))
        c = int(c)
        col_max_row[c] = max(col_max_row.get(c, int(r)), int(r))

    placed_by_col = {}
    for rr, cc in placed:
        placed_by_col.setdefault(int(cc), []).append(int(rr))

    space_to_dst = {}
    dst_to_bottom = {}
    for src_col in src_tooth_cols:
        dst_col = int(src_col + dx)
        if dst_col not in placed_by_col:
            continue
        tooth_bottom = int(col_max_row[src_col] + dy)
        free_space = interior_bottom - tooth_bottom
        space_to_dst.setdefault(int(free_space), []).append(dst_col)
        dst_to_bottom[dst_col] = tooth_bottom

    for info in slot_info:
        count = int(info["count"])
        colors = info["colors"]
        dst_for_count = space_to_dst.get(count, [])
        if len(dst_for_count) != 1:
            continue
        col = int(dst_for_count[0])
        start_row = int(dst_to_bottom[col] + 1)
        max_len = interior_bottom - start_row + 1
        if max_len <= 0:
            continue
        if len(colors) > max_len:
            colors = colors[:max_len]
        for i, colv in enumerate(colors):
            rr = start_row + i
            if interior_top <= rr <= interior_bottom:
                out[rr, col] = int(colv)

    return out


BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

H, W = 8, 20

FRAME_R0, FRAME_C0 = 0, 9
FRAME_H, FRAME_W = 8, 11
TOP_OPEN_LOCAL_COLS = tuple(range(2, 9))  # absolute cols 11..17

SLOT_COLS = [11, 13, 15, 17]
SLOT_LEN = 4
SLOT_TOP_ROW = 1

BAR_INPUT_R0, BAR_INPUT_C0 = 0, 1
BAR_OUTPUT_R0, BAR_OUTPUT_C0 = 2, 11
BAR_H, BAR_W = 4, 7
BAR_LOCAL_TOOTH_COLS = [0, 2, 4, 6]
BAR_BASE_OUTPUT_ROW = BAR_OUTPUT_R0

MAX_ATTEMPTS = 400


def _frame_patch() -> np.ndarray:
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    patch[0, list(TOP_OPEN_LOCAL_COLS)] = BG
    return patch


def _bar_patch(heights: list[int]) -> np.ndarray:
    patch = np.zeros((BAR_H, BAR_W), dtype=int)
    patch[0, :] = GREY
    for local_c, height in zip(BAR_LOCAL_TOOTH_COLS, heights):
        patch[:height, local_c] = GREY
    return patch


def _paint_input_slot_column(grid: np.ndarray, col: int, color: int, count: int) -> None:
    cap = SLOT_LEN - count
    if cap > 0:
        grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + SLOT_LEN, col] = color


def _paint_output_slot_column(grid: np.ndarray, col: int, color: int, count: int, height: int) -> None:
    start_row = BAR_BASE_OUTPUT_ROW + height
    if count > 0:
        grid[start_row : start_row + count, col] = color


def _build_input_grid(heights: list[int], colors: list[int], input_counts: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_INPUT_R0, BAR_INPUT_C0)
    for col, color, count in zip(SLOT_COLS, colors, input_counts):
        _paint_input_slot_column(grid, col, color, count)
    return grid


def _build_expected_grid(heights: list[int], colors: list[int], input_counts: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_OUTPUT_R0, BAR_OUTPUT_C0)

    count_to_color = {int(count): int(color) for color, count in zip(colors, input_counts)}
    target_counts = [5 - int(height) for height in heights]
    for col, height, target_count in zip(SLOT_COLS, heights, target_counts):
        _paint_output_slot_column(grid, col, count_to_color[int(target_count)], int(target_count), int(height))
    return grid


def _sample_heights(rng: np.random.Generator) -> list[int]:
    while True:
        heights = [int(v) for v in rng.permutation(np.array([1, 2, 3, 4], dtype=int))]
        if heights != sorted(heights) and heights != sorted(heights, reverse=True):
            return heights


def _sample_input_counts(target_counts: list[int], rng: np.random.Generator) -> list[int]:
    target = np.array(target_counts, dtype=int)
    for _ in range(100):
        cand = [int(v) for v in rng.permutation(target)]
        if all(a != b for a, b in zip(cand, target_counts)):
            return cand
    return [int(v) for v in target[::-1]]


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    heights = _sample_heights(rng)
    target_counts = [5 - int(height) for height in heights]
    input_counts = _sample_input_counts(target_counts, rng)
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    return heights, colors, input_counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, colors, input_counts = _sample_params(rng)
    grid = _build_input_grid(heights, colors, input_counts)
    expected = _build_expected_grid(heights, colors, input_counts)

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
