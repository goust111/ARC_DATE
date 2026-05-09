# Combo variant of 7c66cb00 (index 1)
# combo_spec: D2/0+D3/0+D6/2
# Name: 三色精灵少数色匹配与外层消失

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste

# Design analysis for the combo variant
# -------------------------------------
# a) Concretization + behavior trace
#    - Input keeps the original global layout: full-width horizontal bars below and
#      separated sprites above them. The local sprite grammar changes to exactly 3
#      nested fill colors: outer shell, middle ring, and smallest core.
#    - Small concrete example:
#        sprite rows:
#          AAAAAAA
#          ABBBBBA
#          ABBCCBA
#          ABBBBBA
#          AAAAAAA
#      Suppose bars have fills {A,B,C} and the C-bar has border D.
#      Original task would have matched by A and preserved both visible layers.
#      This combo instead matches by the minority core C, docks the whole 5x7 shape
#      into the C-bar, then applies: A -> bg, B -> D, C -> fill(C).
#      The docked result becomes:
#          0000000
#          0DDDDD0
#          0DDCDD0
#          0DDDDD0
#          0000000
#      pasted opaquely, so the outer shell becomes a visible bg moat cut out of the
#      target bar instead of transparently revealing the bar fill.
#    - The key differences from the source task are therefore:
#      1. Matching key moves from the dominant outer color to the smallest core color.
#      2. The sprite is still docked as one object, not split across multiple bars.
#      3. The outer layer vanishes into bg while the middle layer inherits the target
#         bar border color.
# b) Complexity calibration
#    - We fix the bar count to 3 and every sprite uses a permutation of all 3 bar fill
#      colors. This makes "three layers" visually undeniable while keeping the color
#      vocabulary small and structured.
#    - Sprite geometry is restricted to symmetric nested rectangles of height 5 and
#      widths 7 or 9. That keeps layer parsing easy for humans and for the solver.
#    - The generator forces at least one pair of sprites with the same outer color but
#      different core colors, so the old "look at the dominant shell" heuristic fails
#      on every sample rather than only occasionally.
# c) Hidden assumption audit against the original main()
#    - Original top-object detection iterated per color component. That would detect
#      one 3-color sprite multiple times, so the combo must detect whole multicolor
#      components instead.
#    - Original matching assumed the most common sprite color was the target key. The
#      combo invalidates that assumption; matching must use the innermost layer.
#    - Original recoloring mapped at most two sprite colors and pasted transparently.
#      Here the outer layer becomes true bg, so the transformed sprite must be pasted
#      opaquely or the disappearance would be invisible.
#    - Original generator only needed sprite height <= outer-target bar height. The
#      combo matches by the core color, so generation must ensure height <= core bar.
# d) Placement legality
#    - Top sprites remain 4-connected and separated by at least one bg cell, which
#      makes multicolor component extraction unambiguous.
#    - Docked sprites reserve their full width only in the core-matched target bar,
#      because the whole transformed bbox is pasted there and can carve bg holes.
#    - All placements stay within cols 1..W-2 so no sprite can overwrite a bar's side
#      border columns.
# e) Visual observability
#    - The core is always a centered 1x3 or 1x5 segment, never a single isolated pixel,
#      so the "minority color" is still visibly readable by a human.
#    - Because at least two sprites share an outer color but land in different bars,
#      2-3 training examples are sufficient to reject outer-color matching.
#    - The output always shows a border-colored middle ring surrounded by a bg moat,
#      making the outer-layer disappearance explicit.
# f) Visual cleanliness
#    - Only one transformed copy is pasted per sprite, so bars do not accumulate the
#      recursive multi-bar descendants used in the standalone D3 variant.
#    - Distinct fill and border palettes avoid accidental color-role collisions.
# g) Color-change inferability
#    - All sprite colors come from the 3 bar fills, and the only new colors in the
#      output are the matched bar's border and bg. The structural cue is therefore:
#      smallest color chooses destination, middle ring adopts border, outer ring dies.
# 9) New solver assumptions and generator guarantees
#    - main() assumes every top sprite has exactly 3 non-bg colors with strict nesting.
#      generate_input() enforces this with centered nested rectangles.
#    - main() assumes exactly one color touches the sprite bbox border and exactly one
#      smaller core sits inside the middle layer. Generation guarantees those bboxes.
#    - main() assumes there are exactly 3 bars with unique fill colors. Generation
#      fixes the bar count and samples all 7 visible colors distinctly.


GRID_H = 30
GRID_W = 24
BAR_COUNT = 3
SPRITE_COUNT_RANGE = (2, 3)
BAR_HEIGHT_RANGE = (5, 8)
MAX_TOTAL_BAR_HEIGHT = 20
MIN_TOP_AREA = 7
SPRITE_H = 5
SPRITE_W_CHOICES = (7, 9)
GEN_RETRIES = 400
MAX_TOP_ROW_COLOR_COUNT = int(GRID_W * 0.7)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        int(outer[0]) <= int(inner[0])
        and int(outer[1]) <= int(inner[1])
        and int(outer[2]) >= int(inner[2])
        and int(outer[3]) >= int(inner[3])
    )


def _find_bars(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    bars: list[dict[str, int]] = []
    current_bar: dict[str, int] | None = None
    h, w = grid.shape

    for r in range(h):
        row = grid[r, :]
        non_bg = [int(v) for v in row if int(v) != int(bg)]
        if non_bg:
            fill_color, count = Counter(non_bg).most_common(1)[0]
            if int(count) > int(w * 0.7):
                left = int(row[0])
                right = int(row[-1])
                border = left if left != int(fill_color) else right
                if current_bar is not None and int(current_bar["fill"]) == int(fill_color):
                    current_bar["r_max"] = int(r)
                else:
                    if current_bar is not None:
                        current_bar["h"] = int(current_bar["r_max"] - current_bar["r_min"] + 1)
                        bars.append(current_bar)
                    current_bar = {
                        "fill": int(fill_color),
                        "border": int(border),
                        "r_min": int(r),
                        "r_max": int(r),
                        "h": 1,
                    }
            elif current_bar is not None:
                current_bar["h"] = int(current_bar["r_max"] - current_bar["r_min"] + 1)
                bars.append(current_bar)
                current_bar = None
        elif current_bar is not None:
            current_bar["h"] = int(current_bar["r_max"] - current_bar["r_min"] + 1)
            bars.append(current_bar)
            current_bar = None

    if current_bar is not None:
        current_bar["h"] = int(current_bar["r_max"] - current_bar["r_min"] + 1)
        bars.append(current_bar)
    return bars


def _extract_layers(patch: np.ndarray, bg: int) -> dict[str, dict[str, int | tuple[int, int, int, int] | bool]]:
    infos = []
    h, w = patch.shape
    for color in sorted(set(map(int, patch.flatten())) - {int(bg)}):
        mask = patch == int(color)
        coords = np.argwhere(mask)
        bbox = bbox_coords(coords)
        touches_border = bool(
            mask[0, :].any() or mask[h - 1, :].any() or mask[:, 0].any() or mask[:, w - 1].any()
        )
        infos.append(
            {
                "color": int(color),
                "bbox": tuple(map(int, bbox)),
                "bbox_area": int(_bbox_area(bbox)),
                "touches_border": touches_border,
            }
        )

    assert len(infos) == 3, "Each sprite must have exactly three colors"

    outer_candidates = [info for info in infos if bool(info["touches_border"])]
    assert len(outer_candidates) == 1, "Expected a single outer layer touching the sprite border"
    outer = outer_candidates[0]

    inner_infos = [info for info in infos if int(info["color"]) != int(outer["color"])]
    inner_infos.sort(key=lambda info: int(info["bbox_area"]), reverse=True)
    middle, core = inner_infos

    assert _contains(outer["bbox"], middle["bbox"])
    assert _contains(middle["bbox"], core["bbox"])
    return {"outer": outer, "middle": middle, "core": core}


def _transform_sprite(
    patch: np.ndarray,
    *,
    middle_color: int,
    core_color: int,
    target_border: int,
    target_fill: int,
    bg: int,
) -> np.ndarray:
    transformed = np.full_like(patch, int(bg))
    transformed[patch == int(middle_color)] = int(target_border)
    transformed[patch == int(core_color)] = int(target_fill)
    return transformed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock three-color nested sprites into bars selected by their smallest color.

    Input structure:
    - bg is the most common color.
    - The bottom region contains exactly three full-width horizontal bars, each with a
      fill color in the interior and a border color on the two side columns.
    - The top region contains separated three-color sprites: one outer shell touching
      the bbox border, one enclosed middle band, and one smallest core.

    Rule:
    - For each sprite, match the bar whose fill color equals the sprite's innermost
      core color rather than its dominant outer color.
    - Paste the full sprite bbox at the same x-position, bottom-aligned in that bar.
    - Recolor layerwise on the docked copy: outer shell -> bg, middle band -> target
      bar border, core -> target bar fill.
    - Erase the original top-area sprites.

    Key insight:
    - The decisive cue is the least-area inner core, and the outer shell does not
      survive docking at all: it becomes a visible bg cutout around a border-colored
      middle ring.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    bars = _find_bars(g, bg)
    assert len(bars) == BAR_COUNT, "Variant expects exactly three horizontal bars"
    bars_by_fill = {int(bar["fill"]): bar for bar in bars}

    pattern_area_end = int(min(int(bar["r_min"]) for bar in bars))
    top = g[:pattern_area_end, :]

    lab, n = label_mask(top != int(bg), connectivity=4)
    sprites = []
    for region_id in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(region_id))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        content = crop_bbox(top, (r0, c0, r1, c1))
        layers = _extract_layers(content, bg)
        sprites.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "content": content,
                "layers": layers,
            }
        )

    out[:pattern_area_end, :] = np.where(out[:pattern_area_end, :] != int(bg), int(bg), out[:pattern_area_end, :])

    for sprite in sprites:
        content = np.array(sprite["content"], dtype=int)
        layers = sprite["layers"]
        core_color = int(layers["core"]["color"])
        middle_color = int(layers["middle"]["color"])
        target_bar = bars_by_fill[core_color]

        transformed = _transform_sprite(
            content,
            middle_color=middle_color,
            core_color=core_color,
            target_border=int(target_bar["border"]),
            target_fill=int(target_bar["fill"]),
            bg=int(bg),
        )
        paste(
            out,
            transformed,
            int(target_bar["r_max"] - content.shape[0] + 1),
            int(sprite["c0"]),
        )

    return out


def _sample_bar_heights() -> list[int]:
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(BAR_COUNT)]
        if sum(heights) > MAX_TOTAL_BAR_HEIGHT:
            continue
        if GRID_H - sum(heights) - (BAR_COUNT - 1) < MIN_TOP_AREA:
            continue
        return heights


def _build_bars(bg: int, fills: list[int], borders: list[int], heights: list[int]) -> tuple[np.ndarray, list[dict[str, int]]]:
    grid = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    bars: list[dict[str, int]] = [None] * len(heights)  # type: ignore[list-item]
    r = GRID_H
    for i in range(len(heights) - 1, -1, -1):
        h = int(heights[i])
        r0, r1 = int(r - h), int(r - 1)
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
        r = int(r0 - 1)
    return grid, bars


def _make_sprite(order: tuple[int, int, int], bars: list[dict[str, int]]) -> dict[str, int | tuple[int, int, int, int] | np.ndarray]:
    outer_idx, middle_idx, core_idx = map(int, order)
    w = int(random.choice(SPRITE_W_CHOICES))
    mw = int(w - 2)
    cw = int(3 if mw == 5 else random.choice((3, 5)))
    if cw >= mw:
        cw = int(mw - 2)

    patch = np.full((SPRITE_H, w), int(bars[outer_idx]["fill"]), dtype=int)
    middle_bbox = (1, 1, 3, mw)
    core_bbox = (2, (w - cw) // 2, 1, cw)

    mr0, mc0, mh, _ = middle_bbox
    cr0, cc0, ch, _ = core_bbox
    patch[mr0 : mr0 + mh, mc0 : mc0 + mw] = int(bars[middle_idx]["fill"])
    patch[cr0 : cr0 + ch, cc0 : cc0 + cw] = int(bars[core_idx]["fill"])

    return {
        "outer_idx": int(outer_idx),
        "middle_idx": int(middle_idx),
        "core_idx": int(core_idx),
        "patch": patch,
        "h": int(SPRITE_H),
        "w": int(w),
        "middle_bbox": tuple(map(int, middle_bbox)),
        "core_bbox": tuple(map(int, core_bbox)),
    }


def _interval_free(occ: np.ndarray, start: int, length: int) -> bool:
    return not occ[int(start) : int(start + length)].any()


def _rect_separated(rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int) -> bool:
    r1, c1 = int(r0 + h - 1), int(c0 + w - 1)
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _fits_top_row_color_budget(
    sprite: dict[str, int | tuple[int, int, int, int] | np.ndarray],
    r0: int,
    row_color_counts: list[Counter[int]],
) -> bool:
    patch = np.array(sprite["patch"], dtype=int)
    for rr in range(patch.shape[0]):
        row_counts = Counter(int(v) for v in patch[rr, :])
        for color, count in row_counts.items():
            if int(row_color_counts[int(r0 + rr)][int(color)]) + int(count) > MAX_TOP_ROW_COLOR_COUNT:
                return False
    return True


def _choose_orders(n_sprites: int) -> list[tuple[int, int, int]]:
    outer_idx = random.randrange(BAR_COUNT)
    others = [idx for idx in range(BAR_COUNT) if idx != outer_idx]
    first = (outer_idx, others[0], others[1])
    second = (outer_idx, others[1], others[0])
    orders = [first, second]

    if n_sprites == 3:
        remaining = [order for order in itertools.permutations(range(BAR_COUNT)) if order not in orders]
        orders.append(random.choice(remaining))

    random.shuffle(orders)
    return [tuple(map(int, order)) for order in orders]


def _combo_signal_is_visible(placed: list[dict]) -> bool:
    """
    Require an explicit D2 cue in every generated sample.

    At least one sprite pair must share the same outer layer while swapping the
    middle/core destinations. This forces the target bar to depend on the
    smallest core color instead of the dominant shell or the middle ring.
    """
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            a = placed[i]
            b = placed[j]
            if int(a["outer_idx"]) != int(b["outer_idx"]):
                continue
            if int(a["core_idx"]) == int(b["core_idx"]):
                continue
            if int(a["middle_idx"]) != int(b["core_idx"]):
                continue
            if int(a["core_idx"]) != int(b["middle_idx"]):
                continue
            return True
    return False


def _place_sprites(sprites: list[dict[str, int | tuple[int, int, int, int] | np.ndarray]], top_h: int) -> list[dict] | None:
    bar_occ = [np.zeros(GRID_W, dtype=bool) for _ in range(BAR_COUNT)]
    top_rects: list[tuple[int, int, int, int]] = []
    row_color_counts = [Counter() for _ in range(top_h)]
    placed = []

    for sprite in sprites:
        h, w = int(sprite["h"]), int(sprite["w"])
        core_idx = int(sprite["core_idx"])
        candidates = []
        for c0 in range(1, GRID_W - w):
            if not _interval_free(bar_occ[core_idx], c0, w):
                continue
            for r0 in range(0, top_h - h + 1):
                if _rect_separated(top_rects, r0, c0, h, w) and _fits_top_row_color_budget(
                    sprite, r0, row_color_counts
                ):
                    candidates.append((r0, c0))
        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        bar_occ[core_idx][c0 : c0 + w] = True
        patch = np.array(sprite["patch"], dtype=int)
        for rr in range(patch.shape[0]):
            row_color_counts[int(r0 + rr)].update(int(v) for v in patch[rr, :])
        placed.append({**sprite, "r0": int(r0), "c0": int(c0)})
        top_rects.append((int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))

    return placed


def _render(base: np.ndarray, placed: list[dict], bars: list[dict[str, int]], bg: int) -> tuple[np.ndarray, np.ndarray]:
    inp, out = base.copy(), base.copy()

    for sprite in placed:
        patch = np.array(sprite["patch"], dtype=int)
        r0, c0 = int(sprite["r0"]), int(sprite["c0"])
        paste(inp, patch, r0, c0, transparent=int(bg))

        target_bar = bars[int(sprite["core_idx"])]
        middle_bar = bars[int(sprite["middle_idx"])]
        transformed = _transform_sprite(
            patch,
            middle_color=int(middle_bar["fill"]),
            core_color=int(target_bar["fill"]),
            target_border=int(target_bar["border"]),
            target_fill=int(target_bar["fill"]),
            bg=int(bg),
        )
        paste(out, transformed, int(target_bar["r_max"] - patch.shape[0] + 1), c0)

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_sprites = random.randint(*SPRITE_COUNT_RANGE)

    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    fills = [int(v) for v in palette[1 : 1 + BAR_COUNT]]
    borders = [int(v) for v in palette[1 + BAR_COUNT : 1 + 2 * BAR_COUNT]]

    heights = _sample_bar_heights()
    base, bars = _build_bars(bg, fills, borders, heights)
    top_h = int(bars[0]["r_min"])

    orders = _choose_orders(n_sprites)
    sprites = [_make_sprite(order, bars) for order in orders]
    sprites.sort(key=lambda sprite: (-int(sprite["w"]), -int(sprite["core_bbox"][3])))

    placed = _place_sprites(sprites, top_h)
    if placed is None:
        return None
    if not _combo_signal_is_visible(placed):
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
