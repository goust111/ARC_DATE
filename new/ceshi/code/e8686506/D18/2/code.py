# Variant of e8686506
# Dimension: D18 — 拼合矩形的内部结构模式 (meta)
# Change: 拼合后大碎片（框架）在外围、小碎片填入凹陷和边角处，形成分层嵌套的矩形... → 碎片横向条带排列

"""
Design analysis for the D18 variant
===================================

spec:
  variant_core:
    old: output was anchored by 1-2 outline/frame pieces; the remaining pieces filled
      irregular interior holes, so the solver had to do 2D placement under LR symmetry.
    new: output is still a dense LR-symmetric portrait rectangle, but every non-bg
      component becomes one full-width horizontal stripe. Assembly is therefore 1D:
      detect the common stripe width, keep each stripe's height/color, and stack the
      stripes top-to-bottom.

  a_concretization_and_trace:
    input:
      - bg is the dominant color.
      - each non-bg component is a solid horizontal rectangle of width W and height 1 or 2.
      - all bars are scattered with gaps, but their vertical order is preserved.
    output:
      - dense HxW rectangle.
      - each stripe occupies full rows across all columns.
      - adjacent stripes use different colors so components stay separate in output.
    example_trace:
      old:
        - total area 30 -> choose 6x5.
        - detect frame piece(s) whose union bbox already equals 6x5.
        - search 2D placements for the remaining irregular fragments inside the frame.
      new:
        - components are [red 1x5] [blue 2x5] [yellow 1x5] [green 2x5], scattered at
          increasing y positions.
        - common width W=5 is read directly from the bars.
        - total area 30 gives H=6.
        - sorting by input top-row yields red, blue, yellow, green; stacking them
          produces the 6x5 stripe rectangle.
      difference:
        - the output is now driven by 1D y-order instead of 2D contour matching.

  b_complexity_calibration:
    - If every stripe had height 1, the task would reduce to "compress colored rows".
      To keep the change visible and non-trivial, the generator uses a mix of height-1
      and height-2 stripes in most samples.
    - Width stays constant within one sample, but the solver infers it from the pieces
      instead of hardcoding it.
    - Color count stays in the original 4-6 range so repeated colors still appear, but
      only non-adjacently, preventing accidental component merges.

  c_hidden_assumptions_review:
    - Old _build_requirements() assumed singleton corners / top-bottom bars / paired
      side lines. Those anchors are specific to the frame regime and are removed.
    - Old frame search assumed 1-2 components already span the full output bbox under
      8-connectivity. This is false for stripes, because no stripe spans the full height.
    - Old _assemble() searched arbitrary 2D placements and enforced LR constraints per
      cell. In the stripe variant, lateral placement freedom no longer exists; keeping
      that search would add fake ambiguity and fail on valid bar inputs.
    - New main() assumes:
      1. every non-bg component is a solid rectangle,
      2. all rectangles share one width,
      3. their top-to-bottom order is meaningful.
      The generator explicitly guarantees all three.

  d_placement_legality:
    - Generator places stripes on a 13-15 canvas with at least one empty row between
      consecutive pieces, so there is no overlap and 4/8-connectivity remains separate.
    - Because vertical order is the clue, top rows are strictly increasing.
    - Random x offsets are allowed; they do not affect the rule.

  e_visual_observability:
    - Across 2-4 examples, a human sees that outputs are stacked horizontal bands rather
      than an outlined frame with interior fragments.
    - The operative visible cue is geometric: same-width horizontal bars appear in input,
      and the output is those bars compressed vertically in the same y-order.
    - No invisible color-id sorting is used.

  f_visual_tidiness:
    - Pieces never overlap and are separated by background, so each stripe is easy to
      identify.
    - Output has one simple layered structure instead of mixed contour/hole interactions.

  g_color_inference:
    - Colors are preserved piece-by-piece; there is no remapping rule to infer.
    - Distinct colors are limited to 4-6 and every used color appears in at least one
      stripe, which keeps examples readable.
"""

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg


def _valid_dims() -> list[tuple[int, int]]:
    """Keep the original portrait factor family, but reinterpret it as stripe outputs."""
    results = []
    for area in range(25, 41):
        dims = [(area // w, w) for w in range(1, area + 1) if area % w == 0 and area // w >= w]
        best_gap = min(abs(h - w) for h, w in dims)
        best = [(h, w) for h, w in dims if abs(h - w) == best_gap]
        if len(best) == 1:
            h, w = best[0]
            if w == 5 and h <= 8:
                results.append((h, w))
    return results


_VALID_DIMS = _valid_dims()
_DIM_WEIGHTS = [h * w for h, w in _VALID_DIMS]

_N_COLORS_MIN = 4
_N_COLORS_MAX = 6
_STRIPE_COUNT_MIN = 4
_STRIPE_COUNT_MAX = 7
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_GEN_MAX_RETRIES = 300


def _read_bands(grid: np.ndarray, bg: int) -> list[dict]:
    bands = []
    for col in sorted(set(map(int, np.unique(grid))) - {bg}):
        for pts in components_mask(grid == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            area = int(len(pts))
            bands.append(
                {
                    "col": int(col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "h": h,
                    "w": w,
                    "area": area,
                }
            )
    return bands


def _stack_bands(bands: list[dict], width: int) -> np.ndarray:
    total_area = sum(b["area"] for b in bands)
    assert total_area % width == 0, "total stripe area must be divisible by the common width"
    height = total_area // width
    out = np.empty((height, width), dtype=int)
    cursor = 0
    for band in sorted(bands, key=lambda b: (b["r0"], b["c0"])):
        out[cursor : cursor + band["h"], :] = band["col"]
        cursor += band["h"]
    assert cursor == height, "stacked stripe heights must fill the output exactly"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repack scattered horizontal bar fragments into a compact left-right symmetric stripe rectangle.

    Input structure:
    - The dominant color is the background; every remaining component is a solid horizontal bar.
    - All bars share one hidden common width, while their heights and colors vary.
    - The bars are scattered apart, and their top-to-bottom order in the input is meaningful.

    Rule:
    - Extract every non-background component, verify that each one is a filled rectangle, and infer
      the shared stripe width from those bars.
    - Sort the bars by their vertical position in the input.
    - Stack them tightly in that order, preserving each bar's height and color, to form the output rectangle.

    Key insight:
    - The old task needed 2D contour matching around a frame; this variant collapses the assembly to 1D.
    - Once the common bar width is known, total area gives the output height and the input y-order gives
      the unique stripe order.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    bands = _read_bands(g, bg)
    assert bands, "grid must contain at least one non-background bar"

    widths = {b["w"] for b in bands}
    assert len(widths) == 1, "all bars must share one common width"
    width = next(iter(widths))

    for band in bands:
        assert band["area"] == band["h"] * band["w"], "each piece must be a solid rectangle"
        assert band["w"] >= band["h"], "pieces must be horizontal bars, not tall columns"

    out = _stack_bands(bands, width)
    assert np.array_equal(out, out[:, ::-1]), "horizontal stripes must remain LR-symmetric"
    return out.astype(int)


def _choose_heights(total_h: int) -> list[int]:
    valid_counts = [
        k
        for k in range(max(_STRIPE_COUNT_MIN, (total_h + 1) // 2), min(_STRIPE_COUNT_MAX, total_h) + 1)
    ]
    assert valid_counts, "must have at least one feasible stripe-count choice"
    if len(valid_counts) > 1 and total_h in valid_counts and random.random() < 0.75:
        valid_counts = [k for k in valid_counts if k != total_h]
    k = random.choice(valid_counts)
    n_twos = total_h - k
    heights = [2] * n_twos + [1] * (k - n_twos)
    random.shuffle(heights)
    return heights


def _choose_colors(n_stripes: int) -> list[int]:
    n_colors = random.randint(_N_COLORS_MIN, min(_N_COLORS_MAX, n_stripes))
    base = list(range(n_colors))
    counts = {c: 1 for c in base}
    for _ in range(n_stripes - n_colors):
        counts[random.choice(base)] += 1

    for _ in range(200):
        seq = []
        prev = None
        remaining = counts.copy()
        for _pos in range(n_stripes):
            opts = [c for c in base if remaining[c] > 0 and c != prev]
            if not opts:
                break
            max_count = max(remaining[c] for c in opts)
            heavy = [c for c in opts if remaining[c] == max_count]
            choice_pool = heavy if random.random() < 0.75 else opts
            col = random.choice(choice_pool)
            seq.append(col)
            remaining[col] -= 1
            prev = col
        else:
            return seq
    raise RuntimeError("failed to build a non-adjacent stripe color sequence")


def _build_output() -> tuple[np.ndarray, int, list[dict]]:
    h, w = random.choices(_VALID_DIMS, weights=_DIM_WEIGHTS, k=1)[0]
    heights = _choose_heights(h)
    abstract_cols = _choose_colors(len(heights))

    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    used_abstract = sorted(set(abstract_cols))
    cmap = {a: palette[i] for i, a in enumerate(used_abstract)}

    out = np.empty((h, w), dtype=int)
    stripes = []
    cursor = 0
    for sh, acol in zip(heights, abstract_cols):
        real_col = cmap[acol]
        out[cursor : cursor + sh, :] = real_col
        stripes.append({"col": real_col, "h": sh, "w": w})
        cursor += sh
    return out, bg, stripes


def _scatter(stripes: list[dict], bg: int) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    total_h = sum(s["h"] for s in stripes)
    min_between = len(stripes) - 1
    slack = rows - total_h - min_between
    if slack < 0:
        raise ValueError("canvas is too short to preserve stripe order with gaps")

    gaps = [0] * (len(stripes) + 1)
    for i in range(1, len(stripes)):
        gaps[i] = 1
    for _ in range(slack):
        gaps[random.randint(0, len(stripes))] += 1

    canvas = np.full((rows, cols), bg, dtype=int)
    r = gaps[0]
    for idx, stripe in enumerate(stripes):
        c = random.randint(0, cols - stripe["w"])
        canvas[r : r + stripe["h"], c : c + stripe["w"]] = stripe["col"]
        r += stripe["h"] + gaps[idx + 1]
    return canvas


def _count_output_components(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _gen_attempt() -> np.ndarray | None:
    out, bg, stripes = _build_output()
    if not np.array_equal(out, out[:, ::-1]):
        return None
    n_colors = len(np.unique(out))
    if not (_N_COLORS_MIN <= n_colors <= _N_COLORS_MAX):
        return None
    if not (_STRIPE_COUNT_MIN <= _count_output_components(out) <= _STRIPE_COUNT_MAX):
        return None
    if not any(s["h"] == 2 for s in stripes):
        return None

    try:
        inp = _scatter(stripes, bg)
        got = main(inp)
    except Exception:
        return None
    if np.array_equal(got, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 D18 input")
