# Variant of a47bf94d
# Dimension: D14 — 管道颜色体系 (symbol_instructions)
# Change: 三色管道系统——主管道用8号色（青色），交叉标记用5号色（灰色），矩形连接块用9号色（栗色），三种结构色固定且不与tok... → 管道宽度编码传输内容

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

# Design analysis for the D14 width-coded variant
# ------------------------------------------------
# concrete_behavior:
#   - The original task had two complementary 3x3 halves:
#       * `_DIAMOND_OFF`: four cardinal cells, the "cross-like" half.
#       * `_X_OFF`: diagonals plus center, the "corner-star" half.
#   - A 3x3 solid block is still the union of those two halves.
#   - The variant changes only one semantic lever:
#       * thin pipe (1-pixel width) transmits the cross-like half,
#         so the source side ends as the X-half and the far side receives the diamond-half.
#       * thick pipe (2-pixel width) transmits the X-half,
#         so the source side ends as the diamond-half and the far side receives the X-half.
#   - To make this visible, every grid contains both thin and thick pipes, and at least one
#     thin-pipe solid source plus one thick-pipe solid source. The same solid input therefore
#     decomposes differently depending on pipe width, which is directly observable in 2-3 examples.
#   - Example trace:
#       * original logic: a solid block always becomes diamond locally and sends X remotely.
#       * new logic on a thin pipe: the same solid block becomes X locally and sends diamond remotely.
#       * difference: the transmitted half is no longer decided by token type; it is decided by width.
#
# complexity_calibration:
#   - The width signal uses only two bins, matching the reference width-constraint idea from 332f06d7.
#   - Keeping more than two widths would create unnecessary ambiguity; keeping only one width would
#     collapse back to the original behavior. Two repeated categories are enough for human inference.
#   - Width affects shape choice only; token colors stay unchanged and still serve only as identity tags.
#
# implicit_assumption_review_against_original_main:
#   - Original `track_mask = g != bg` was safe because all non-token structure pixels were semantically
#     equivalent. In the variant, width itself matters, so the solver must read component geometry.
#   - Original endpoint logic relied on 1-pixel-wide degree-1 graph ends. That fails on 2-pixel pipes:
#     a thick endpoint is a short row of two cells, not a single degree-1 cell.
#   - Original `_trace_wire` assumed a centerline graph and would wander inside a 2-wide corridor.
#   - Original routing rule `solid -> emit X`, `X -> emit diamond` hardcoded token-shape semantics.
#     That assumption is invalid here; width now selects the emitted half.
#   - To keep the width information unambiguous, the generator no longer uses a shared crossover graph.
#     Shared junctions would mix thin and thick semantics in the same connected region and make the
#     width cue visually muddy. Instead, each pipe is its own orthogonal connected component.
#
# placement_legality:
#   - Pipes are separated by at least one blank column/row in 4-neighborhood terms, so components do
#     not accidentally merge even when some are 2 pixels wide.
#   - Thick pipes always extend to the east for vertical segments and to the south for horizontal
#     segments. This gives a deterministic anchor column for the token center two cells away.
#   - Token colors avoid {5,8,9}, so token detection cannot confuse structure colors with payload colors.
#   - Optional color-9 rectangles are painted only inside already thick pipe cells, preserving connectivity.
#
# visual_observability_and_cleanliness:
#   - All channels run top-to-bottom like the original, but each one is an isolated pipe so the width cue
#     stays readable. Thin vs thick is visible near both endpoints, where the causal comparison matters most.
#   - The grid never relies on invisible ordering by color id; all inference comes from shape and width.
#   - 9-rectangles remain as localized structural decorations on thick pipes, but they do not alter the rule.
#
# new_solver_assumptions_and_generator_guarantees:
#   - Each structural connected component has exactly one top endpoint row and one bottom endpoint row.
#   - Tokens, if present, are centered exactly two cells outside those endpoint anchors.
#   - For a thin component, any source token is either solid or X; for a thick component, any source token
#     is either solid or diamond. The generator enforces this so the solver can interpret the source side.

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_STRUCT_COLORS = (5, 8, 9)

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID_OFF = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]


def _find_solid_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for maximal 3x3 solid blocks."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            col = int(sub[0, 0])
            if col == bg or np.any(sub != col):
                continue
            maximal = True
            for rr in range(r0, r0 + 3):
                for cc in range(c0, c0 + 3):
                    for nr, nc in neighbors4(rr, cc):
                        if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                            if int(g[nr, nc]) == col:
                                maximal = False
                                break
                    if not maximal:
                        break
                if not maximal:
                    break
            if maximal:
                found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _find_pattern_3x3(
    g: np.ndarray,
    bg: int,
    offsets: list[tuple[int, int]],
) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for isolated 3x3 patterns."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}

    mask = np.zeros((3, 3), dtype=bool)
    for dr, dc in offsets:
        mask[1 + dr, 1 + dc] = True

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            vals = sub[mask]
            if vals.size == 0:
                continue
            col = int(vals[0])
            if col == bg:
                continue
            if np.any(vals != col) or np.any(sub[~mask] != bg):
                continue
            isolated = True
            for dr, dc in offsets:
                rr, cc = r0 + 1 + dr, c0 + 1 + dc
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                        if int(g[nr, nc]) == col:
                            isolated = False
                            break
                if not isolated:
                    break
            if isolated:
                found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _find_x_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    return _find_pattern_3x3(g, bg, _X_OFF)


def _find_diamond_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    return _find_pattern_3x3(g, bg, _DIAMOND_OFF)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=4)]


def _stamp_token(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets: list[tuple[int, int]]) -> None:
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _clear_3x3(grid: np.ndarray, center: tuple[int, int], bg: int) -> None:
    r0, c0 = center[0] - 1, center[1] - 1
    grid[r0 : r0 + 3, c0 : c0 + 3] = bg


def _run_length_east(cells: set[tuple[int, int]], r: int, c: int) -> int:
    run = 0
    while (r, c + run) in cells:
        run += 1
    return run


def _describe_component(cells: set[tuple[int, int]]) -> dict[str, tuple[int, int] | int]:
    rs = [r for r, _ in cells]
    top_r = min(rs)
    bot_r = max(rs)
    top_c = min(c for r, c in cells if r == top_r)
    bot_c = min(c for r, c in cells if r == bot_r)
    top_width = _run_length_east(cells, top_r, top_c)
    bot_width = _run_length_east(cells, bot_r, bot_c)
    width = 2 if max(top_width, bot_width) >= 2 else 1
    return {
        "top_anchor": (top_r, top_c),
        "bot_anchor": (bot_r, bot_c),
        "width": width,
    }


def _token_kind_at(
    center: tuple[int, int],
    solids: dict[tuple[int, int], tuple[int, tuple[int, int]]],
    xs: dict[tuple[int, int], tuple[int, tuple[int, int]]],
    diamonds: dict[tuple[int, int], tuple[int, tuple[int, int]]],
) -> tuple[str | None, int | None]:
    if center in solids:
        return "solid", int(solids[center][0])
    if center in xs:
        return "x", int(xs[center][0])
    if center in diamonds:
        return "diamond", int(diamonds[center][0])
    return None, None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each colored 3x3 token into two complementary halves, with pipe width deciding which half travels.

    Input structure:
    - Background cells surround several independent top-to-bottom pipe components made from structural colors 8 and 9.
    - Each pipe has width 1 or 2 pixels and has token slots exactly two cells beyond its top and bottom endpoint anchors.
    - Tokens use non-structural colors and appear as solid 3x3 blocks or as one of the two half-shapes.

    Rule:
    - Thin pipes transmit the four-cardinal half, so their source side ends as an X-shape and the far side receives a diamond-shape.
    - Thick pipes transmit the X-shape, so their source side ends as a diamond-shape and the far side receives an X-shape.
    - A solid source block is cleared then replaced by the stationary half; a source half-shape stays in place; the transmitted half is stamped at the opposite endpoint.

    Key insight:
    - The token color still identifies which source pairs with which destination, but the transmitted content is no longer inferred from the token type alone.
    - Width is the deciding cue: 1-pixel pipes carry the cardinal half, 2-pixel pipes carry the diagonal-plus-center half.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)
    diamonds = _find_diamond_3x3(g, bg)
    track_mask = np.isin(g, _STRUCT_COLORS)
    comps = _components(track_mask)

    out = g.copy()

    for pts in comps:
        cells = set(pts)
        desc = _describe_component(cells)
        width = int(desc["width"])
        top_anchor = tuple(desc["top_anchor"])
        bot_anchor = tuple(desc["bot_anchor"])
        top_center = (top_anchor[0] - 2, top_anchor[1])
        bot_center = (bot_anchor[0] + 2, bot_anchor[1])

        transmitted_kind = "diamond" if width == 1 else "x"
        stationary_kind = "x" if width == 1 else "diamond"
        transmitted_offsets = _DIAMOND_OFF if width == 1 else _X_OFF
        stationary_offsets = _X_OFF if width == 1 else _DIAMOND_OFF

        top_kind, top_col = _token_kind_at(top_center, solids, xs, diamonds)
        bot_kind, bot_col = _token_kind_at(bot_center, solids, xs, diamonds)

        candidates: list[tuple[str, str, int]] = []
        if top_kind == "solid" or top_kind == stationary_kind:
            candidates.append(("top", top_kind, int(top_col)))
        if bot_kind == "solid" or bot_kind == stationary_kind:
            candidates.append(("bottom", bot_kind, int(bot_col)))

        if not candidates:
            continue

        solid_candidates = [c for c in candidates if c[1] == "solid"]
        source_side, source_kind, color = solid_candidates[0] if solid_candidates else candidates[0]

        if source_side == "top":
            source_center = top_center
            far_center = bot_center
        else:
            source_center = bot_center
            far_center = top_center

        if source_kind == "solid":
            _clear_3x3(out, source_center, bg)
            _stamp_token(out, source_center[0], source_center[1], color, stationary_offsets)
        else:
            _stamp_token(out, source_center[0], source_center[1], color, stationary_offsets)

        _stamp_token(out, far_center[0], far_center[1], color, transmitted_offsets)

    return out


# --------------- generator ---------------

CHANNEL_COUNTS = [4, 5]
CHANNEL_SPACING = [4, 5]
LEFT_MARGIN_RANGE = (2, 3)
TOKEN_ROW_OPTIONS = [2, 3]
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GRID_H_RANGE = (20, 26)
GRID_W_MIN = 22
GRID_W_MAX = 26
EP_TOKEN_GAP = 2

SOLID_SOURCE_PROB = 0.65
FAR_PREPLACE_PROB = 0.30
RECT_PROB = 0.55
GEN_MAX_RETRIES = 400


def _paint_segment(mask: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], width: int) -> None:
    r0, c0 = p0
    r1, c1 = p1
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        mask[r0, lo : hi + 1] = True
        if width == 2:
            mask[r0 + 1, lo : hi + 1] = True
    else:
        lo, hi = sorted((r0, r1))
        mask[lo : hi + 1, c0] = True
        if width == 2:
            mask[lo : hi + 1, c0 + 1] = True


def _component_mask_from_path(
    h: int,
    w: int,
    points: list[tuple[int, int]],
    width: int,
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for i in range(len(points) - 1):
        _paint_segment(mask, points[i], points[i + 1], width)
    return mask


def _touches_existing(mask: np.ndarray, occupied: np.ndarray) -> bool:
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        if occupied[r, c]:
            return True
        for nr, nc in neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                return True
    return False


def _outward_shift(index: int, n: int) -> int:
    mid = (n - 1) / 2.0
    if index < mid:
        return -1
    if index > mid:
        return 1
    return 0


def _try_build_channel(
    h: int,
    w: int,
    top_ep_row: int,
    bot_ep_row: int,
    col: int,
    width: int,
    lane_shift: int,
    occupied: np.ndarray,
) -> dict | None:
    lane_col = col + lane_shift
    if lane_col < 1:
        return None
    if width == 2 and lane_col + 1 >= w:
        return None

    upper_lo = top_ep_row + 2
    upper_hi = max(upper_lo, (top_ep_row + bot_ep_row) // 2 - 1)
    lower_lo = min(bot_ep_row - 2, (top_ep_row + bot_ep_row) // 2 + 1)
    lower_hi = bot_ep_row - 2
    if upper_lo > upper_hi or lower_lo > lower_hi:
        return None

    for _ in range(30):
        r1 = random.randint(upper_lo, upper_hi)
        r2 = random.randint(lower_lo, lower_hi)
        if r1 >= r2:
            continue
        points = [
            (top_ep_row, col),
            (r1, col),
            (r1, lane_col),
            (r2, lane_col),
            (r2, col),
            (bot_ep_row, col),
        ]
        mask = _component_mask_from_path(h, w, points, width)
        if _touches_existing(mask, occupied):
            continue
        return {
            "mask": mask,
            "points": points,
            "width": width,
            "col": col,
            "lane_col": lane_col,
            "r1": r1,
            "r2": r2,
        }
    return None


def _paint_rectangles(grid: np.ndarray, channels: list[dict]) -> None:
    thick = [ch for ch in channels if ch["width"] == 2 and random.random() < RECT_PROB]
    for ch in thick:
        lane_col = int(ch["lane_col"])
        r1 = int(ch["r1"])
        r2 = int(ch["r2"])
        if r2 - r1 + 1 < 3:
            continue
        rect_h = random.randint(2, min(3, r2 - r1 + 1))
        rect_r0 = random.randint(r1, r2 - rect_h + 1)
        grid[rect_r0 : rect_r0 + rect_h, lane_col : lane_col + 2] = 9


def _source_stationary_offsets(width: int) -> list[tuple[int, int]]:
    return _X_OFF if width == 1 else _DIAMOND_OFF


def _transmitted_offsets(width: int) -> list[tuple[int, int]]:
    return _DIAMOND_OFF if width == 1 else _X_OFF


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(CHANNEL_COUNTS)
    spacing = random.choice(CHANNEL_SPACING)
    first_col = random.randint(*LEFT_MARGIN_RANGE)
    ch_cols = [first_col + i * spacing for i in range(n)]
    w = max(ch_cols[-1] + 4, GRID_W_MIN)
    if w > GRID_W_MAX:
        return None

    top_token_row = random.choice(TOKEN_ROW_OPTIONS)
    top_ep_row = top_token_row + EP_TOKEN_GAP

    h_min = max(GRID_H_RANGE[0], top_ep_row + 14)
    h = random.randint(h_min, GRID_H_RANGE[1])
    bot_token_row = h - top_token_row - 1
    bot_ep_row = bot_token_row - EP_TOKEN_GAP
    if bot_ep_row - top_ep_row < 8:
        return None

    token_colors = random.sample(TOKEN_COLORS_POOL, n)

    widths = [random.choice([1, 2]) for _ in range(n)]
    if all(v == widths[0] for v in widths):
        widths[0] = 1
        widths[-1] = 2
    thin_ids = [i for i, v in enumerate(widths) if v == 1]
    thick_ids = [i for i, v in enumerate(widths) if v == 2]
    if not thin_ids or not thick_ids:
        return None

    occupied = np.zeros((h, w), dtype=bool)
    channels: list[dict] = []
    for i, col in enumerate(ch_cols):
        base_shift = _outward_shift(i, n)
        shift_choices = [base_shift]
        if base_shift != 0:
            shift_choices.append(0)
        random.shuffle(shift_choices)

        built = None
        for shift in shift_choices:
            built = _try_build_channel(h, w, top_ep_row, bot_ep_row, col, widths[i], shift, occupied)
            if built is not None:
                break
        if built is None:
            return None
        occupied |= built["mask"]
        channels.append(built)

    base = np.zeros((h, w), dtype=int)
    base[occupied] = 8
    _paint_rectangles(base, channels)

    source_sides = [random.choice(["top", "bottom"]) for _ in range(n)]
    source_modes = ["solid" if random.random() < SOLID_SOURCE_PROB else "stationary" for _ in range(n)]
    source_modes[random.choice(thin_ids)] = "solid"
    source_modes[random.choice(thick_ids)] = "solid"

    output = base.copy()
    inp = base.copy()

    variant_witness = 0
    for i, ch in enumerate(channels):
        color = int(token_colors[i])
        width = int(ch["width"])
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[i])
        source_center = top_center if source_sides[i] == "top" else bot_center
        far_center = bot_center if source_sides[i] == "top" else top_center
        stationary_offsets = _source_stationary_offsets(width)
        transmitted_offsets = _transmitted_offsets(width)

        _stamp_token(output, source_center[0], source_center[1], color, stationary_offsets)
        _stamp_token(output, far_center[0], far_center[1], color, transmitted_offsets)

        if source_modes[i] == "solid":
            _stamp_token(inp, source_center[0], source_center[1], color, _SOLID_OFF)
            if width == 1:
                variant_witness += 1
            if random.random() < FAR_PREPLACE_PROB:
                _stamp_token(inp, far_center[0], far_center[1], color, transmitted_offsets)
        else:
            _stamp_token(inp, source_center[0], source_center[1], color, stationary_offsets)

    if variant_witness == 0:
        return None

    if np.array_equal(main(inp), output):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
