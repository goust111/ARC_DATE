# Combo variant of 7c66cb00 (index 4)
# combo_spec: D1/3+D8/2+D9/2
# Name: 含干扰精灵的选择性擦除与面积排序匹配

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
# - The scene keeps the original visual family: a 30x24 canvas, 2-3 full-width bars
#   stacked in the lower area, and several compact two-tone sprites in the top area.
# - The fused change is not "color decides the bar" anymore. Instead:
#   1. Detect whole top sprites as connected non-background objects.
#   2. Sort sprites by occupied area from large to small.
#   3. Sort bars by vertical position from top to bottom.
#   4. Pair by rank: largest sprite -> topmost bar, second largest -> second bar, etc.
#   5. Only those paired sprites are erased from the top and docked into bars.
#   6. Any extra smaller sprites are distractors and stay untouched in the top area.
# - Small trace:
#   * Suppose there are 3 bars and 5 top sprites with areas [20, 15, 12, 6, 4].
#   * Original task would use outer-color matching and clear every top sprite.
#   * Here the top three areas [20, 15, 12] are the matched set because there are
#     three bars; areas [6, 4] are distractors. The matched set is recolored by its
#     destination bar and moved down; the two small distractors remain exactly where
#     they started.
# - This realizes the fusion literally: D1 changes matching to size ranking, D9 adds
#   extra distractor sprites, and D8 changes erase behavior so only matched sprites
#   are removed.
#
# b) Complexity calibration
# - Every sample has exactly one matched sprite per bar plus 1-2 distractors, so the
#   solver must genuinely perform selection rather than relying on a one-to-one object
#   count.
# - All matched sprites are generated with distinct areas, and every distractor is
#   strictly smaller than the smallest matched sprite. That keeps the ordinal rule
#   visible and prevents hidden tie-break ambiguity.
# - Matched sprites deliberately use a deranged outer-color assignment relative to
#   their destination bars, so the task cannot collapse back into the original
#   color-matching rule.
#
# c) Implicit-assumption audit of the original solver
# - Original code scanned top objects per monochrome color component. That breaks the
#   fused rule because an inner patch would become its own ranked item, which is wrong:
#   ranking and selective erasure must operate on whole visual sprites.
# - Original code cleared the full top area before any matching. That also breaks the
#   fused rule because distractors must remain visible in the output.
# - Original matching relied on `sprite.outer_color == bar.fill`. That assumption is
#   intentionally false here, so the solver must replace it with an ordinal zip between
#   sorted bars and sorted whole-sprite objects.
# - The D9 single variant described distractors as color-unmatched. In this combo,
#   that color cue remains in the generator for visual clarity, but the actual reason a
#   sprite is unmatched is that it falls outside the first `len(bars)` positions in the
#   area ranking. This resolves the cross-dimension conflict cleanly.
#
# d) Placement legality
# - Bars preserve the original geometry: full width, border on both outer columns, and
#   one background row between neighboring bars.
# - Each bar receives exactly one matched sprite, so docked outputs cannot collide.
# - Top sprites are placed with a one-cell gap and stay inside cols 1..W-2.
# - Matched sprite heights are chosen relative to their ranked destination bars so the
#   bottom-aligned docking is always legal.
#
# e) Visual observability
# - A human can see bar order directly.
# - A human can compare sprite sizes directly because sprites are compact rectangles
#   with an optional enclosed minority inner block and all matched areas are distinct.
# - The distractors are visibly smaller than the matched sprites and also use colors
#   not found in the bars, so repeated examples support the idea that they are extra
#   top objects rather than alternative bar targets.
#
# f) Visual cleanliness
# - The top area stays readable because sprites are sparse and separated.
# - The output is also clean because only one transformed sprite appears in each bar.
# - There is no mixed overlay in the same region: matched sprites disappear from their
#   original location, while distractors remain where they were.
#
# g) Change inferability
# - The recolor rule remains structured and local to the destination bar:
#   outer cells become the bar border color, inner cells become the bar fill color.
# - Because the destination bar is determined by size order rather than by numeric
#   color ids, the task uses only visually accessible features.
# - The generator enforces a persistent gap between matched and distractor areas, so a
#   human can infer that "the larger sprites move, the extra small ones stay".


GRID_H = 30
GRID_W = 24
BAR_COUNT_RANGE = (2, 3)
DISTRACTOR_COUNT_RANGE = (1, 2)
BAR_HEIGHT_RANGE = (4, 8)
MAX_THREE_BAR_TOTAL = 20
SPRITE_H_RANGE = (2, 5)
SPRITE_W_RANGE = (2, 9)
INNER_COLOR_PROB = 0.55
GEN_RETRIES = 600


def _find_bars(g: np.ndarray, bg: int) -> list[dict]:
    bars: list[dict] = []
    current_bar: dict | None = None
    h, w = g.shape
    for r in range(h):
        row = g[int(r), :]
        non_bg = [int(v) for v in row if int(v) != int(bg)]
        if non_bg:
            fill_color, count = Counter(non_bg).most_common(1)[0]
            if int(count) > int(w * 0.7):
                border = int(row[0]) if int(row[0]) != int(fill_color) else int(row[-1])
                if current_bar and int(current_bar["fill"]) == int(fill_color):
                    current_bar["r_max"] = int(r)
                else:
                    if current_bar:
                        bars.append(current_bar)
                    current_bar = {
                        "fill": int(fill_color),
                        "border": int(border),
                        "r_min": int(r),
                        "r_max": int(r),
                    }
            elif current_bar:
                bars.append(current_bar)
                current_bar = None
        elif current_bar:
            bars.append(current_bar)
            current_bar = None
    if current_bar:
        bars.append(current_bar)
    return bars


def _extract_top_sprites(g: np.ndarray, pattern_area_end: int, bg: int) -> list[dict]:
    top = g[:pattern_area_end, :]
    mask = top != int(bg)
    if not bool(mask.any()):
        return []
    lab, n = label_mask(mask, connectivity=4)
    sprites: list[dict] = []
    for region_id in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(region_id))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        content = np.array(crop_bbox(g, (r0, c0, r1, c1)), dtype=int)
        colors = Counter(int(v) for v in content.flatten() if int(v) != int(bg))
        if not colors:
            continue
        outer_color = int(colors.most_common(1)[0][0])
        sprites.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "h": int(content.shape[0]),
                "w": int(content.shape[1]),
                "area": int(len(coords)),
                "outer_color": int(outer_color),
                "content": content,
            }
        )
    return sprites


def _sorted_bars_and_sprites(g: np.ndarray, bg: int) -> tuple[list[dict], list[dict], int]:
    bars = _find_bars(g, bg)
    assert bars, "No bars found"
    pattern_area_end = int(min(int(bar["r_min"]) for bar in bars))
    sprites = _extract_top_sprites(g, pattern_area_end, bg)
    assert len(sprites) >= len(bars), "Need at least as many sprites as bars"
    bars_sorted = sorted(bars, key=lambda bar: int(bar["r_min"]))
    sprites_sorted = sorted(sprites, key=lambda sp: (-int(sp["area"]), int(sp["r0"]), int(sp["c0"])))
    return bars_sorted, sprites_sorted, pattern_area_end


def _transform_for_bar(sprite: dict, bar: dict, bg: int) -> np.ndarray:
    transformed = np.array(sprite["content"], dtype=int).copy()
    outer_mask = transformed == int(sprite["outer_color"])
    non_bg_mask = transformed != int(bg)
    transformed[outer_mask] = int(bar["border"])
    transformed[non_bg_mask & (~outer_mask)] = int(bar["fill"])
    return transformed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock the largest top sprites into the bars and keep smaller distractors in place.

    Input structure:
    - The background is the most common color.
    - The lower region contains 2-3 full-width horizontal bars, each with one fill color
      and border cells on the left and right edges.
    - The upper region contains more connected two-tone sprites than there are bars; the
      extra smaller sprites are distractors.

    Rule:
    - Detect whole top sprites as connected non-background objects.
    - Sort sprites by occupied area from large to small, and sort bars from top to bottom.
    - Pair the first `len(bars)` sprites with the bars by rank.
    - For each matched sprite, keep its original x-position, bottom-align it in the
      paired bar, recolor outer cells to the bar border color and inner cells to the bar
      fill color, then erase only that matched source sprite from the top region.
    - Leave every smaller unmatched distractor sprite unchanged in the top area.

    Key insight:
    - Matching is decided by relative size order, not by color. The bars provide the
      destination colors, while the extra small sprites are excluded because they fall
      below the first `n` positions in the area ranking and therefore are never erased.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    bars_sorted, sprites_sorted, _ = _sorted_bars_and_sprites(g, bg)
    matched = sprites_sorted[: len(bars_sorted)]

    for bar, sprite in zip(bars_sorted, matched):
        r0 = int(sprite["r0"])
        c0 = int(sprite["c0"])
        h = int(sprite["h"])
        w = int(sprite["w"])
        mask = np.array(sprite["content"], dtype=int) != int(bg)
        out[r0 : r0 + h, c0 : c0 + w][mask] = int(bg)

        transformed = _transform_for_bar(sprite, bar, bg)
        place_r = int(bar["r_max"] - h + 1)
        paste(out, transformed, place_r, c0, transparent=int(bg))

    return out


def _required_top_area(total_sprites: int) -> int:
    return 7 + 2 * max(0, int(total_sprites) - 3)


def _sample_bar_heights(n_bars: int, total_sprites: int) -> list[int]:
    min_top_area = _required_top_area(total_sprites)
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(n_bars)]
        if n_bars == 3 and sum(heights) > MAX_THREE_BAR_TOTAL:
            continue
        if GRID_H - sum(heights) - (n_bars - 1) < min_top_area:
            continue
        return heights


def _build_bars(bg: int, fills: list[int], borders: list[int], heights: list[int]) -> tuple[np.ndarray, list[dict]]:
    grid = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    bars: list[dict] = [None] * len(heights)  # type: ignore[assignment]
    r = GRID_H
    for i in range(len(heights) - 1, -1, -1):
        h = int(heights[i])
        r0, r1 = r - h, r - 1
        grid[r0 : r1 + 1, 1 : GRID_W - 1] = int(fills[i])
        grid[r0 : r1 + 1, 0] = int(borders[i])
        grid[r0 : r1 + 1, GRID_W - 1] = int(borders[i])
        bars[i] = {
            "fill": int(fills[i]),
            "border": int(borders[i]),
            "h": int(h),
            "r_min": int(r0),
            "r_max": int(r1),
        }
        r = r0 - 1
    return grid, bars


def _derangement(n: int) -> list[int]:
    if n == 2:
        return [1, 0]
    shift = random.choice([1, 2])
    return [int((i + shift) % n) for i in range(n)]


def _candidate_dims(max_h: int, *, min_area: int = 8) -> list[tuple[int, int, int]]:
    dims: list[tuple[int, int, int]] = []
    for h in range(SPRITE_H_RANGE[0], min(SPRITE_H_RANGE[1], int(max_h)) + 1):
        for w in range(SPRITE_W_RANGE[0], SPRITE_W_RANGE[1] + 1):
            area = int(h * w)
            if area >= int(min_area):
                dims.append((area, int(h), int(w)))
    dims.sort(key=lambda item: (-int(item[0]), -int(item[1]), -int(item[2])))
    return dims


def _choose_matched_dims(bars: list[dict]) -> list[tuple[int, int, int]] | None:
    candidate_lists = [_candidate_dims(int(bar["h"]), min_area=8) for bar in bars]
    choice: list[tuple[int, int, int]] = []

    def backtrack(i: int, prev_area: int | None) -> bool:
        if i == len(bars):
            return True
        candidates = candidate_lists[i][:]
        random.shuffle(candidates)
        candidates.sort(key=lambda item: -int(item[0]))
        for area, h, w in candidates:
            if prev_area is not None and int(area) >= int(prev_area):
                continue
            if any(int(area) == int(prev[0]) for prev in choice):
                continue
            choice.append((int(area), int(h), int(w)))
            if backtrack(i + 1, int(area)):
                return True
            choice.pop()
        return False

    if backtrack(0, None):
        return choice
    return None


def _sample_inner_bbox(h: int, w: int) -> tuple[int, int, int, int] | None:
    valid = [
        (ih, iw)
        for ih in range(1, h - 1)
        for iw in range(1, w - 1)
        if (w - iw) % 2 == 0 and int(ih * iw * 2) < int(h * w)
    ]
    if not valid:
        return None
    ih, iw = random.choice(valid)
    ir0 = random.randint(1, h - ih - 1)
    ic0 = (w - iw) // 2
    return int(ir0), int(ic0), int(ih), int(iw)


def _make_patch(h: int, w: int, outer_color: int, inner_choices: list[int]) -> np.ndarray:
    patch = np.full((int(h), int(w)), int(outer_color), dtype=int)
    if inner_choices and h >= 3 and w >= 3 and random.random() < INNER_COLOR_PROB:
        inner_bbox = _sample_inner_bbox(int(h), int(w))
        if inner_bbox is not None:
            ir0, ic0, ih, iw = inner_bbox
            patch[ir0 : ir0 + ih, ic0 : ic0 + iw] = int(random.choice(inner_choices))
    return patch


def _small_distractor_dims(n_distractors: int) -> list[tuple[int, int, int]]:
    pool = [(4, 2, 2), (6, 2, 3)]
    random.shuffle(pool)
    return pool[: int(n_distractors)]


def _rect_separated(rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int) -> bool:
    r1 = int(r0 + h - 1)
    c1 = int(c0 + w - 1)
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _place_sprites(sprites: list[dict], top_h: int) -> list[dict] | None:
    top_rects: list[tuple[int, int, int, int]] = []
    placed: list[dict] = []
    for sp in sorted(sprites, key=lambda item: (-int(item["area"]), -int(item["w"]), -int(item["h"]))):
        h = int(sp["h"])
        w = int(sp["w"])
        candidates: list[tuple[int, int]] = []
        for c0 in range(1, GRID_W - w):
            for r0 in range(0, top_h - h + 1):
                if _rect_separated(top_rects, int(r0), int(c0), int(h), int(w)):
                    candidates.append((int(r0), int(c0)))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placed.append({**sp, "r0": int(r0), "c0": int(c0)})
        top_rects.append((int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))
    return placed


def _render(base: np.ndarray, placed: list[dict], bars: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    inp = base.copy()
    out = base.copy()
    bars_sorted = sorted(bars, key=lambda bar: int(bar["r_min"]))

    for sp in placed:
        paste(inp, np.array(sp["patch"], dtype=int), int(sp["r0"]), int(sp["c0"]), transparent=int(bg))

    matched_sorted = sorted(
        [sp for sp in placed if bool(sp["matched"])],
        key=lambda sp: (-int(sp["area"]), int(sp["r0"]), int(sp["c0"])),
    )
    distractors = [sp for sp in placed if not bool(sp["matched"])]

    for sp in distractors:
        paste(out, np.array(sp["patch"], dtype=int), int(sp["r0"]), int(sp["c0"]), transparent=int(bg))

    for bar, sp in zip(bars_sorted, matched_sorted):
        sprite = {
            "content": np.array(sp["patch"], dtype=int),
            "outer_color": int(sp["outer_color"]),
            "h": int(sp["h"]),
            "c0": int(sp["c0"]),
        }
        transformed = _transform_for_bar(sprite, bar, bg)
        paste(out, transformed, int(bar["r_max"] - sp["h"] + 1), int(sp["c0"]), transparent=int(bg))

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_bars = random.randint(*BAR_COUNT_RANGE)
    n_distractors = random.randint(*DISTRACTOR_COUNT_RANGE)
    total_sprites = n_bars + n_distractors

    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    fills = [int(c) for c in palette[1 : 1 + n_bars]]
    borders = [int(c) for c in palette[1 + n_bars : 1 + 2 * n_bars]]
    distractor_colors = [int(c) for c in palette[1 + 2 * n_bars :]]
    if len(distractor_colors) < 2:
        return None

    heights = _sample_bar_heights(n_bars, total_sprites)
    base, bars = _build_bars(bg, fills, borders, heights)
    top_h = int(min(int(bar["r_min"]) for bar in bars))

    matched_dims = _choose_matched_dims(sorted(bars, key=lambda bar: int(bar["r_min"])))
    if matched_dims is None:
        return None
    if int(matched_dims[-1][0]) <= 6:
        return None

    outer_perm = _derangement(n_bars)
    matched_sprites: list[dict] = []
    for target_idx in range(n_bars):
        area, h, w = matched_dims[target_idx]
        outer_idx = int(outer_perm[target_idx])
        inner_choices = [int(fill) for i, fill in enumerate(fills) if i != int(outer_idx)]
        if n_bars >= 3:
            strict_inner = [int(fill) for i, fill in enumerate(fills) if i != int(outer_idx) and i != int(target_idx)]
            if strict_inner:
                inner_choices = strict_inner
        patch = _make_patch(int(h), int(w), int(fills[outer_idx]), inner_choices)
        matched_sprites.append(
            {
                "matched": True,
                "target_idx": int(target_idx),
                "outer_color": int(fills[outer_idx]),
                "patch": patch,
                "h": int(h),
                "w": int(w),
                "area": int(area),
            }
        )

    distractor_dims = _small_distractor_dims(n_distractors)
    if len(distractor_dims) != n_distractors:
        return None
    distractor_sprites: list[dict] = []
    for area, h, w in distractor_dims:
        outer_color = int(random.choice(distractor_colors))
        inner_choices = [int(c) for c in distractor_colors if int(c) != int(outer_color)]
        patch = _make_patch(int(h), int(w), outer_color, inner_choices)
        distractor_sprites.append(
            {
                "matched": False,
                "target_idx": None,
                "outer_color": int(outer_color),
                "patch": patch,
                "h": int(h),
                "w": int(w),
                "area": int(area),
            }
        )

    sprites = matched_sprites + distractor_sprites
    placed = _place_sprites(sprites, top_h)
    if placed is None:
        return None

    inp, out = _render(base, placed, bars, bg)

    if int(get_bg(inp)) != int(bg):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
