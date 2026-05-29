# Combo variant of 7c66cb00 (index 18)
# combo_spec: D3/1+D9/0+D12/0
# Name: 小网格单色精灵一对一严格匹配

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import get_bg, label_mask, paste
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[4] / "solvers"))
    from arc_utils import get_bg, label_mask, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concretization + behavior trace
#    - The fused task keeps the same global layout family as 7c66cb00: full-width
#      horizontal bars live below, sprites live above, and output docks each sprite
#      into a bar at the same x-position.
#    - Concrete small example on the new 20x15 canvas:
#        top sprite A:        A A A
#                             A 0 A
#                             A A A
#        top sprite B:          B B
#                               B B
#        top sprite C:    C C C C
#                         0 C 0 C
#      Bottom bars are 3 stacked strips with fills A, B, C and border colors
#      P, Q, R. In the original task a sprite could be 2-tone and bars could
#      receive 0, 1, or 2 sprites. In the fused task every sprite is single-color,
#      so the whole connected mask matches one unique bar by that one color only.
#      Output clears the top area and docks:
#        A-mask -> same shape recolored to border P inside fill-A bar
#        B-mask -> same shape recolored to border Q inside fill-B bar
#        C-mask -> same shape recolored to border R inside fill-C bar
#    - The visible differences from the source task are therefore:
#      1. No inner-color region exists anywhere in a sprite.
#      2. Sprite count equals bar count exactly, so every bar receives one sprite.
#      3. The entire scene is compressed into 20x15 with shorter bars and smaller
#         2x2..3x5 sprites, which makes the top area visually tight.
# b) Complexity calibration
#    - If the small-grid variant allowed only 2 bars, the top half would become too
#      empty on a 20x15 board. To keep D12 visually obvious without adding messy
#      floating bars, generation fixes the compact case of 3 bars and 3 sprites.
#    - Fills remain unique across bars and every sprite uses exactly one fill color,
#      so color matching is strict and human-readable instead of ambiguous.
#    - Sprite geometry comes from a small bank of connected 2x2..3x5 masks with
#      occasional holes/notches. That keeps output recognizable while avoiding the
#      visual flatness of using only solid rectangles.
# c) Hidden assumption audit against the original main()
#    - Original bar detection used a loose "dominant row color" heuristic suited to
#      larger 30x24 bars. On a 20x15 board we tighten this to exact bar rows:
#      interior columns 1..W-2 are one fill color and the two edge cells share one
#      border color distinct from the fill.
#    - Original sprite extraction iterated per color and then inspected each bbox for
#      inner colors. The fused task invalidates that dependency; we detect whole
#      top-area non-bg connected components and require exactly one non-bg color per
#      component.
#    - Original placement loop allowed multiple matching sprites per bar. The fused
#      task forbids that, so main() now builds a one-to-one color map and asserts
#      that every bar fill appears exactly once among the top sprites.
#    - Original recoloring had to distinguish outer and inner layers. That branch is
#      removed; all non-bg sprite cells simply become the matched bar's border color.
# d) Placement legality
#    - The generator keeps all top sprites inside cols 1..W-2 so docking never writes
#      over the bar edge columns by accident.
#    - Top sprites are separated in 4-neighborhood terms, not merely by color, because
#      the solver treats any touching non-bg cells as one component.
#    - Every sampled sprite height is <= every sampled bar height range, and the final
#      dock position is bottom-aligned inside the unique matched bar.
# e) Visual observability
#    - Because each sprite color reappears as exactly one bar fill and every bar gets
#      exactly one docked copy, the one-to-one correspondence is directly visible.
#    - Compact top placement is enforced by small top-area height and by preferring
#      placements that sit low, so the 20x15 shrink is apparent from examples.
#    - Border colors are sampled distinct from all fills; the recolor rule is thus
#      readable as "sprite color -> that bar's side color" rather than color aliasing.
# f) Visual cleanliness
#    - There is only one pasted copy per sprite and no inner sub-sprite recursion, so
#      bars remain legible even in the smaller canvas.
#    - The shape bank is small and connected, which keeps the scene structured instead
#      of noisy.
# g) Color-change inferability
#    - The input uses exactly 1 bg + 3 fill + 3 border colors. The only output change
#      is that each mono-color sprite adopts the border color of the bar whose fill
#      matched the sprite. The bar itself supplies the structural cue for the mapping.
# 9) New solver assumptions and generator guarantees
#    - main() assumes exactly 3 bars with unique fills and exactly 3 top sprites with
#      one non-bg color each. generate_input() constructs exactly that.
#    - main() assumes bar rows follow the exact full-width strip schema described
#      above. generate_input() renders bars only in that schema.
#    - main() assumes top components do not touch orthogonally. generate_input()
#      enforces 4-neighborhood separation while still keeping sprites visually close.


GRID_H = 20
GRID_W = 15
BAR_COUNT = 3
BAR_HEIGHT_RANGE = (3, 5)
BAR_TOTAL_RANGE = (12, 14)
SPRITE_H_RANGE = (2, 3)
SPRITE_W_RANGE = (2, 5)
GEN_RETRIES = 500

SHAPE_LIBRARY = (
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1, 1], [0, 1, 0, 1]], dtype=bool),
    np.array([[1, 1, 1, 1], [1, 0, 1, 0], [1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]], dtype=bool),
)


def _is_bar_row(row: np.ndarray, bg: int) -> tuple[bool, int | None, int | None]:
    fill = int(row[1])
    border = int(row[0])
    if fill == int(bg) or border == int(bg) or border == fill:
        return False, None, None
    if int(row[-1]) != border:
        return False, None, None
    if not np.all(row[1:-1] == fill):
        return False, None, None
    return True, fill, border


def _find_bars(grid: np.ndarray, bg: int) -> list[dict]:
    bars: list[dict] = []
    r = 0
    while r < grid.shape[0]:
        ok, fill, border = _is_bar_row(grid[r], bg)
        if not ok:
            r += 1
            continue
        r0 = r
        while r < grid.shape[0]:
            same, same_fill, same_border = _is_bar_row(grid[r], bg)
            if not same or int(same_fill) != int(fill) or int(same_border) != int(border):
                break
            r += 1
        bars.append(
            {
                "fill": int(fill),
                "border": int(border),
                "r_min": int(r0),
                "r_max": int(r - 1),
                "h": int(r - r0),
            }
        )
    assert len(bars) == BAR_COUNT, "expected exactly three compact bars"
    fills = [int(bar["fill"]) for bar in bars]
    assert len(set(fills)) == len(fills), "bar fills must be unique"
    return bars


def _find_sprites(grid: np.ndarray, bg: int, top_h: int) -> list[dict]:
    top = grid[:top_h, :]
    lab, n = label_mask(top != int(bg), connectivity=4)
    sprites: list[dict] = []
    for region_id in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(region_id))
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        patch = np.full((int(r1) - int(r0) + 1, int(c1) - int(c0) + 1), int(bg), dtype=int)
        local = coords - np.array([[int(r0), int(c0)]])
        patch[local[:, 0], local[:, 1]] = top[coords[:, 0], coords[:, 1]]
        colors = sorted({int(v) for v in patch.flatten() if int(v) != int(bg)})
        assert len(colors) == 1, "every compact sprite must be monochrome"
        sprites.append(
            {
                "color": int(colors[0]),
                "r0": int(r0),
                "c0": int(c0),
                "h": int(patch.shape[0]),
                "w": int(patch.shape[1]),
                "content": patch,
            }
        )
    assert len(sprites) == BAR_COUNT, "expected one top sprite per bar"
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock compact monochrome sprites into uniquely color-matched bars on a smaller grid.

    Input structure:
    - The canvas is a tight 20x15 board with three full-width horizontal bars stacked low.
    - Each bar uses one fill color across cols 1..W-2 and one border color on the two side cells.
    - Above the bars sit exactly three small monochrome connected sprites, each using one unique bar fill color.

    Rule:
    - Match each sprite to the single bar whose fill color equals the sprite color.
    - Erase all top sprites from the input area.
    - Recolor the matched sprite so every non-bg cell becomes that bar's border color.
    - Paste the recolored sprite into the matched bar at the original x-position, bottom-aligned inside the bar.

    Key insight:
    - Once the inner color layer is removed and the counts are forced to 1:1, color alone determines the destination:
      each top sprite is a direct token for one bar fill, and the docked copy simply adopts that bar's side color.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    bars = _find_bars(g, bg)
    top_h = int(min(int(bar["r_min"]) for bar in bars))
    sprites = _find_sprites(g, bg, top_h)

    fill_to_bar = {int(bar["fill"]): bar for bar in bars}
    sprite_colors = [int(sprite["color"]) for sprite in sprites]
    assert len(set(sprite_colors)) == len(sprite_colors), "sprite colors must be unique"
    assert set(sprite_colors) == set(fill_to_bar.keys()), "sprite colors must match bar fills exactly"

    out[:top_h, :] = np.where(out[:top_h, :] == int(bg), out[:top_h, :], int(bg))

    for sprite in sprites:
        bar = fill_to_bar[int(sprite["color"])]
        transformed = np.where(sprite["content"] == int(bg), int(bg), int(bar["border"]))
        paste(
            out,
            transformed,
            int(bar["r_max"]) - int(sprite["h"]) + 1,
            int(sprite["c0"]),
            transparent=int(bg),
        )

    return out


def _sample_bar_heights() -> list[int]:
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(BAR_COUNT)]
        total = sum(heights)
        if BAR_TOTAL_RANGE[0] <= total <= BAR_TOTAL_RANGE[1]:
            return heights


def _build_bars(bg: int, fills: list[int], borders: list[int], heights: list[int]) -> tuple[np.ndarray, list[dict]]:
    grid = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    bars: list[dict] = [None] * BAR_COUNT  # type: ignore[assignment]
    r = GRID_H
    for i in range(BAR_COUNT - 1, -1, -1):
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


def _sample_sprite_mask() -> np.ndarray:
    candidates = [shape for shape in SHAPE_LIBRARY if SPRITE_H_RANGE[0] <= shape.shape[0] <= SPRITE_H_RANGE[1] and SPRITE_W_RANGE[0] <= shape.shape[1] <= SPRITE_W_RANGE[1]]
    mask = candidates[random.randrange(len(candidates))].copy()
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return mask


def _expand4(mask: np.ndarray) -> np.ndarray:
    expanded = mask.copy()
    expanded[1:, :] |= mask[:-1, :]
    expanded[:-1, :] |= mask[1:, :]
    expanded[:, 1:] |= mask[:, :-1]
    expanded[:, :-1] |= mask[:, 1:]
    return expanded


def _candidate_cells(mask: np.ndarray, top_h: int, c0: int, r0: int) -> np.ndarray:
    full = np.zeros((top_h, GRID_W), dtype=bool)
    h, w = mask.shape
    full[r0 : r0 + h, c0 : c0 + w] = mask
    return full


def _place_sprites(sprites: list[dict], top_h: int) -> list[dict] | None:
    order = sorted(range(len(sprites)), key=lambda i: (-int(sprites[i]["mask"].sum()), -int(sprites[i]["h"]), -int(sprites[i]["w"])))
    placed: list[dict | None] = [None] * len(sprites)

    def dfs(depth: int, occ: np.ndarray) -> bool:
        if depth == len(order):
            return True
        idx = order[depth]
        sprite = sprites[idx]
        mask = sprite["mask"]
        h, w = int(sprite["h"]), int(sprite["w"])
        expanded_occ = _expand4(occ)
        candidates: list[tuple[int, float, int, int]] = []
        for r0 in range(0, top_h - h + 1):
            for c0 in range(1, GRID_W - w):
                full = _candidate_cells(mask, top_h, c0, r0)
                if np.any(full & expanded_occ):
                    continue
                gap_below = top_h - (r0 + h)
                center_offset = abs((c0 + (w - 1) / 2.0) - (GRID_W - 1) / 2.0)
                candidates.append((int(gap_below), float(center_offset), int(r0), int(c0)))
        random.shuffle(candidates)
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

        for _gap_below, _center_offset, r0, c0 in candidates:
            full = _candidate_cells(mask, top_h, c0, r0)
            placed[idx] = {**sprite, "r0": int(r0), "c0": int(c0)}
            if dfs(depth + 1, occ | full):
                return True
            placed[idx] = None
        return False

    if not dfs(0, np.zeros((top_h, GRID_W), dtype=bool)):
        return None

    result = [sprite for sprite in placed if sprite is not None]
    max_bottom = max(int(sprite["r0"]) + int(sprite["h"]) for sprite in result)
    if top_h - max_bottom > 1:
        return None
    return result


def _make_sprites(bg: int, fills: list[int]) -> list[dict]:
    sprites: list[dict] = []
    for fill in fills:
        mask = _sample_sprite_mask()
        patch = np.where(mask, int(fill), int(bg))
        sprites.append(
            {
                "color": int(fill),
                "mask": mask,
                "patch": patch,
                "h": int(mask.shape[0]),
                "w": int(mask.shape[1]),
            }
        )
    return sprites


def _render(base: np.ndarray, bars: list[dict], placed: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    inp = base.copy()
    out = base.copy()
    fill_to_bar = {int(bar["fill"]): bar for bar in bars}

    for sprite in placed:
        bar = fill_to_bar[int(sprite["color"])]
        paste(inp, sprite["patch"], int(sprite["r0"]), int(sprite["c0"]), transparent=int(bg))
        transformed = np.where(sprite["patch"] == int(bg), int(bg), int(bar["border"]))
        paste(
            out,
            transformed,
            int(bar["r_max"]) - int(sprite["h"]) + 1,
            int(sprite["c0"]),
            transparent=int(bg),
        )

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    fills = [int(v) for v in palette[1 : 1 + BAR_COUNT]]
    borders = [int(v) for v in palette[1 + BAR_COUNT : 1 + 2 * BAR_COUNT]]

    heights = _sample_bar_heights()
    base, bars = _build_bars(bg, fills, borders, heights)
    top_h = int(bars[0]["r_min"])
    sprites = _make_sprites(bg, fills)
    placed = _place_sprites(sprites, top_h)
    if placed is None:
        return None

    inp, out = _render(base, bars, placed, bg)

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
