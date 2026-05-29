# Variant of 9aaea919
# Dimension: D1 — 匹配条件机制 (match_cond)
# Change: marker_pixels — 红色(2)标记条标识源车道，绿色(3)标记条标识目标车道，通过标记颜色直接指定源和目标... → 车道位置关系决定角色

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the D1 positional-role variant
# -----------------------------------------------
# concrete_behavior:
#   - The 30x30 canvas, four fixed-width lanes, plus-shaped sprites, bottom-up stacking,
#     and grey recoloring all stay the same as the source task.
#   - The visible change is that row 29 no longer contains any marker bars.
#   - Instead, role comes from horizontal order alone: the leftmost lane is always the
#     source and the rightmost lane is always the destination. The two interior lanes are
#     unchanged spectators.
# trace_example:
#   - Original: markers on row 29 explicitly say "lane 0 is red/source" and
#     "lane 3 is green/target", then the transfer logic greys lane 0 and grows lane 3.
#   - Variant: the same transfer happens, but the input omits those bars, so the solver
#     must first sort the occupied lanes by x-position and then apply the same count
#     transfer from the outer-left lane to the outer-right lane.
# complexity_calibration:
#   - Keeping four lanes and fixed outer-lane roles makes the rule learnable from a few
#     examples, but removing markers still creates a real D1 change because role recovery
#     now depends on spatial order rather than a dedicated color cue.
#   - To avoid accidental color-based overfitting, the generator frequently uses colors 2
#     and 3 as ordinary sprite colors on arbitrary lanes.
# implicit_assumptions_replaced:
#   - `_find_marker_bars(...)` and marker-removal loops from the original task relied on
#     colors 2 and 3 being dedicated bars; that is no longer valid.
#   - "nearest green lane" also relied on explicit target markers; here there is exactly
#     one positional target, the rightmost lane.
#   - Horizontal alignment is now derived from lane left edges instead of marker centers.
# placement_legality:
#   - Each lane still occupies a disjoint 5-column slot, so copied sprites never overlap
#     across lanes.
#   - The generator enforces `target_count + source_count <= 6`, which guarantees the
#     destination stack fits inside rows 1..27 on the existing period-4 lattice.
# visual_observability_and_cleanliness:
#   - The absence of row-29 bars is easy to notice.
#   - Across examples, the left extreme lane consistently turns grey while the right
#     extreme lane consistently grows upward, which makes the positional rule visible.
# new_solver_assumptions_audited:
#   - every non-background component is a monochrome 3x5 plus sprite;
#   - all four lanes are occupied and distinguishable by their left column;
#   - the leftmost lane is the only source and the rightmost lane is the only target;
#   - the target lane has at least one input sprite so its color and stacking period are
#     observable.
#   The generator explicitly constructs inputs that satisfy all four assumptions.


# --------------- shared constants ---------------

_H, _W = 30, 30
_NUM_LANES = 4
_LANE_STARTS = [3, 9, 15, 21]
_SPRITE_H, _SPRITE_W = 3, 5
_BOTTOM_TOP = 25
_PERIOD = 4
_GREY = 5


def _lane_groups(g: np.ndarray, bg: int) -> list[dict]:
    """
    Extract occupied lanes by grouping monochrome 3x5 sprite components by left column.
    """
    lanes_by_c0: dict[int, list[list[tuple[int, int]]]] = {}

    for pts_arr in components_mask(g != bg, connectivity=4):
        pts = [(int(r), int(c)) for r, c in pts_arr]
        colors = {int(g[r, c]) for r, c in pts}
        r0, c0, r1, c1 = bbox_coords(pts_arr)
        if len(colors) != 1:
            continue
        if (r1 - r0 + 1, c1 - c0 + 1, len(pts)) != (_SPRITE_H, _SPRITE_W, 11):
            continue
        lanes_by_c0.setdefault(int(c0), []).append(pts)

    lanes = []
    for c0 in sorted(lanes_by_c0):
        sprites = lanes_by_c0[c0]
        sprites_sorted = []
        for pts in sprites:
            rmin = min(r for r, _ in pts)
            sprites_sorted.append((rmin, pts))
        sprites_sorted.sort()

        tops = [rmin for rmin, _pts in sprites_sorted]
        sample_pts = sprites_sorted[0][1]
        color = int(g[sample_pts[0][0], sample_pts[0][1]])
        if len(tops) >= 2:
            diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
            period = int(np.gcd.reduce(diffs)) if diffs else _PERIOD
        else:
            period = _SPRITE_H + 1

        lanes.append(
            {
                "c0": int(c0),
                "color": int(color),
                "sprites": [pts for _rmin, pts in sprites_sorted],
                "tops": tops,
                "period": int(period),
            }
        )
    return lanes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the outer-left lane's sprites into the outer-right lane, recoloring and stacking.

    Input structure:
    - bg = most common color. There are no marker bars; instead, four aligned vertical lanes
      are visible only through repeated monochrome sprites.
    - Each occupied lane contains identical 3x5 plus sprites placed on the same vertical
      period-4 lattice.

    Rule:
    - Treat the leftmost lane as the source and the rightmost lane as the destination.
    - Source sprites turn grey (5); the same number of sprite copies are inserted into empty
      slots above the destination stack, recolored to the destination lane's color.
    - Middle lanes are untouched.

    Key insight: Role assignment is positional rather than chromatic, so the solver must
    sort lanes by x-position before applying the original count-transfer logic.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    lanes = _lane_groups(g, bg)
    assert len(lanes) == _NUM_LANES, "expected four occupied sprite lanes"

    src = lanes[0]
    dst = lanes[-1]
    assert src["sprites"] and dst["sprites"], "expected both outer lanes to contain sprites"

    out = g.copy()

    for pts in src["sprites"]:
        for r, c in pts:
            out[r, c] = _GREY

    src_ref_pts = src["sprites"][0]
    src_ref_r0 = min(r for r, _ in src_ref_pts)
    src_offsets = [(r - src_ref_r0, c - src["c0"]) for r, c in src_ref_pts]
    sprite_h = max(dr for dr, _dc in src_offsets) + 1

    occupied = set(dst["tops"])
    bottom0 = max(dst["tops"])
    placed = 0
    k = 1
    while placed < len(src["sprites"]):
        new_top = bottom0 - k * dst["period"]
        k += 1
        if new_top < 0 or new_top + sprite_h > h:
            if new_top < 0:
                break
            continue
        if new_top in occupied:
            continue

        dst_cells = [(new_top + dr, dst["c0"] + dc) for dr, dc in src_offsets]
        if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in dst_cells):
            continue
        if any(int(out[rr, cc]) != bg for rr, cc in dst_cells):
            continue

        for rr, cc in dst_cells:
            out[rr, cc] = dst["color"]
        occupied.add(new_top)
        placed += 1

    return out


# --------------- generator ---------------

_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_TARGET_TOTAL = 6
_GEN_RETRIES = 300


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return a 3x5 plus sprite with transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    """Return top rows for `count` sprites, ordered top-to-bottom."""
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    """Stamp `count` sprites of `color` into one fixed lane."""
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, c0, transparent=bg)


def _choose_lane_colors(rng: np.random.Generator, bg: int) -> list[int]:
    """
    Frequently include colors 2 and 3 as ordinary lane colors so role inference cannot
    fall back to old marker-color semantics.
    """
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    special = [c for c in (2, 3) if c in sprite_palette]

    if len(special) == 2 and rng.random() < 0.8:
        rest = [c for c in sprite_palette if c not in special]
        chosen = special + rng.choice(rest, size=_NUM_LANES - 2, replace=False).tolist()
    else:
        chosen = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()

    rng.shuffle(chosen)
    return [int(c) for c in chosen]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    lane_colors = _choose_lane_colors(rng, bg)
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    src_lane = 0
    dst_lane = _NUM_LANES - 1
    gain = int(input_counts[src_lane])
    if input_counts[dst_lane] + gain > _MAX_TARGET_TOTAL:
        return None

    output_counts = input_counts[:]
    output_counts[dst_lane] += gain

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx == src_lane else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, int(color), int(output_counts[lane_idx]), bg)

    inp = out.copy()
    _paint_lane(inp, src_lane, int(lane_colors[src_lane]), int(input_counts[src_lane]), bg)

    dst_c0 = _LANE_STARTS[dst_lane]
    for r0 in _lane_tops(output_counts[dst_lane])[:gain]:
        inp[r0:r0 + _SPRITE_H, dst_c0:dst_c0 + _SPRITE_W] = bg

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
