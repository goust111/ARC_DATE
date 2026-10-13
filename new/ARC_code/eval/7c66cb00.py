from __future__ import annotations
from collections import Counter
import numpy as np
from arc_utils import bbox_coords, crop_bbox, find_components, get_bg, label_mask, paste
import random
from arc_utils import get_bg, paste

def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock two-tone sprites into color-matched horizontal bars with a color swap.

    Input structure:
    - bg = most common color.
    - Top area: small two-tone sprites (outer color + inner color).
    - Below: horizontal bars spanning most of the row, each with a fill color and border color on edges.

    Rule:
    - Match each sprite to the bar whose fill color equals the sprite's outer color.
    - Place the sprite at its original x-position, aligned to the bar's bottom.
    - Recolor: sprite outer → bar border, sprite inner → bar fill.
    - Erase original sprites from the top area.

    Key insight: Color matching is inverted — the sprite's dominant color identifies its target bar, then gets swapped to the bar's accent color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()
    bg = int(Counter(g.flatten()).most_common(1)[0][0])

    # Find bars.
    bars = []
    current_bar = None
    for r in range(H):
        row = g[int(r), :]
        non_bg = [int(v) for v in row if int(v) != int(bg)]
        if non_bg:
            c = Counter(non_bg)
            fill_color, count = c.most_common(1)[0]
            if int(count) > int(W * 0.7):
                border = int(row[0]) if int(row[0]) != int(fill_color) else int(row[-1])
                if current_bar and int(current_bar["fill"]) == int(fill_color):
                    current_bar["r_max"] = int(r)
                else:
                    if current_bar:
                        bars.append(current_bar)
                    current_bar = {"fill": int(fill_color), "border": int(border), "r_min": int(r), "r_max": int(r)}
            else:
                if current_bar:
                    bars.append(current_bar)
                    current_bar = None
        else:
            if current_bar:
                bars.append(current_bar)
                current_bar = None
    if current_bar:
        bars.append(current_bar)

    assert bars, "No bars found — grid must contain at least one horizontal bar"

    pattern_area_end = int(min(int(b["r_min"]) for b in bars))

    # Find patterns in top area: monochrome components by color, but we need full bbox content.
    patterns = []
    top = g[:pattern_area_end, :]
    for col in sorted(set(map(int, top.flatten())) - {int(bg)}):
        mask = top == int(col)
        if not bool(mask.any()):
            continue
        lab, n = label_mask(mask, connectivity=4)
        for region_id in range(1, int(n) + 1):
            coords = np.argwhere(lab == int(region_id))
            if coords.size == 0:
                continue
            b = bbox_coords(coords)
            r0, c0, r1, c1 = map(int, b)
            bbox = crop_bbox(g, (r0, c0, r1, c1))

            colors_in_bbox = Counter(map(int, bbox.flatten()))
            colors_in_bbox.pop(int(bg), None)
            if not colors_in_bbox:
                continue
            outer_color = int(colors_in_bbox.most_common(1)[0][0])
            inner_colors = [int(c) for c in colors_in_bbox.keys() if int(c) != int(outer_color)]
            inner_color = int(inner_colors[0]) if inner_colors else None

            patterns.append(
                {
                    "outer_color": int(outer_color),
                    "inner_color": inner_color,
                    "r0": int(r0),
                    "c0": int(c0),
                    "h": int(bbox.shape[0]),
                    "w": int(bbox.shape[1]),
                    "content": bbox,
                }
            )

    # Clear patterns from output.
    out[:pattern_area_end, :] = np.where(out[:pattern_area_end, :] != int(bg), int(bg), out[:pattern_area_end, :])

    # Match and place patterns.
    for bar in bars:
        matching = [p for p in patterns if int(p["outer_color"]) == int(bar["fill"])]
        for pat in matching:
            transformed = np.array(pat["content"], dtype=int).copy()
            for rr in range(int(pat["h"])):
                for cc in range(int(pat["w"])):
                    v = int(transformed[int(rr), int(cc)])
                    if v == int(pat["outer_color"]):
                        transformed[int(rr), int(cc)] = int(bar["border"])
                    elif pat["inner_color"] is not None and v == int(pat["inner_color"]):
                        transformed[int(rr), int(cc)] = int(bar["fill"])

            place_r = int(bar["r_max"] - pat["h"] + 1)
            place_c = int(pat["c0"])
            paste(out, transformed, place_r, place_c, transparent=int(bg))

    return out

# --------------- generator (Codex attempt 2) ---------------

# --- Configuration constants ---
GRID_H = 30                   # canvas height
GRID_W = 24                   # canvas width
BAR_COUNT_RANGE = (2, 3)      # number of horizontal bars
SPRITE_COUNT_RANGE = (2, 3)   # number of sprites above bars
BAR_HEIGHT_RANGE = (4, 8)     # per-bar row height
MAX_THREE_BAR_TOTAL = 20      # max total bar height when n_bars == 3
MIN_TOP_AREA = 7              # minimum rows above first bar for sprite area
SPRITE_H_RANGE = (2, 5)      # sprite height (clamped to bar height)
SPRITE_W_RANGE = (2, 9)      # sprite width
INNER_COLOR_PROB = 0.45       # probability of adding an inner-color region
GEN_RETRIES = 300             # max attempts in generate_input()


def _sample_bar_heights(n_bars: int) -> list[int]:
    """Sample bar heights that leave enough room for the sprite top area."""
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(n_bars)]
        if n_bars == 3 and sum(heights) > MAX_THREE_BAR_TOTAL:
            continue
        if GRID_H - sum(heights) - (n_bars - 1) < MIN_TOP_AREA:
            continue
        return heights


def _build_bars(bg: int, fills: list[int], borders: list[int],
                heights: list[int]) -> tuple[np.ndarray, list[dict]]:
    """Render horizontal bars bottom-up and return (grid, bar_info_list)."""
    grid = np.full((GRID_H, GRID_W), bg, dtype=int)
    bars = [None] * len(heights)
    r = GRID_H
    for i in range(len(heights) - 1, -1, -1):
        h = heights[i]
        r0, r1 = r - h, r - 1
        grid[r0:r1 + 1, 1:GRID_W - 1] = fills[i]
        grid[r0:r1 + 1, 0] = borders[i]
        grid[r0:r1 + 1, GRID_W - 1] = borders[i]
        bars[i] = {"fill": fills[i], "border": borders[i],
                    "h": h, "r_min": r0, "r_max": r1}
        r = r0 - 1
    return grid, bars


def _make_sprite(outer_idx: int, bars: list[dict]) -> dict:
    """Create a sprite patch with outer color and optional inner color region."""
    outer_fill = bars[outer_idx]["fill"]
    h = random.randint(SPRITE_H_RANGE[0], min(SPRITE_H_RANGE[1], bars[outer_idx]["h"]))
    w = random.randint(*SPRITE_W_RANGE)
    patch = np.full((h, w), outer_fill, dtype=int)

    inner_idx = None
    inner_bbox = None
    if len(bars) >= 2 and h >= 3 and w >= 3 and random.random() < INNER_COLOR_PROB:
        inner_choices = [i for i in range(len(bars)) if i != outer_idx]
        inner_idx = random.choice(inner_choices)
        # Enumerate valid inner region sizes: centered horizontally, minority area
        valid_inner = [(ih, iw)
                       for ih in range(1, h - 1)
                       for iw in range(1, w - 1)
                       if (w - iw) % 2 == 0 and ih * iw * 2 < h * w]
        if valid_inner:
            ih, iw = random.choice(valid_inner)
            ir0 = random.randint(1, h - ih - 1)
            ic0 = (w - iw) // 2
            patch[ir0:ir0 + ih, ic0:ic0 + iw] = bars[inner_idx]["fill"]
            inner_bbox = (ir0, ic0, ih, iw)
        else:
            inner_idx = None

    return {"outer_idx": outer_idx, "inner_idx": inner_idx,
            "patch": patch, "h": h, "w": w, "inner_bbox": inner_bbox}


def _interval_free(occ: np.ndarray, start: int, length: int) -> bool:
    """Check that occ[start:start+length] is all False."""
    return not occ[start:start + length].any()


def _rect_separated(rects: list[tuple[int, int, int, int]],
                     r0: int, c0: int, h: int, w: int) -> bool:
    """Check that a new rect has at least 1-cell gap from all existing rects."""
    r1, c1 = r0 + h - 1, c0 + w - 1
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _place_sprites(sprites: list[dict], bars: list[dict],
                   top_h: int) -> list[dict] | None:
    """Find valid placements for all sprites. Returns placed list or None."""
    bar_occ = [np.zeros(GRID_W, dtype=bool) for _ in bars]
    top_rects: list[tuple[int, int, int, int]] = []
    placed = []

    for sp in sprites:
        h, w = sp["h"], sp["w"]
        outer_idx, inner_idx = sp["outer_idx"], sp["inner_idx"]

        candidates = []
        for c0 in range(1, GRID_W - w):
            if not _interval_free(bar_occ[outer_idx], c0, w):
                continue
            if inner_idx is not None:
                _, ic0, _, iw = sp["inner_bbox"]
                if not _interval_free(bar_occ[inner_idx], c0 + ic0, iw):
                    continue
            for r0 in range(0, top_h - h + 1):
                if _rect_separated(top_rects, r0, c0, h, w):
                    candidates.append((r0, c0))

        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        bar_occ[outer_idx][c0:c0 + w] = True
        if inner_idx is not None:
            _, ic0, _, iw = sp["inner_bbox"]
            bar_occ[inner_idx][c0 + ic0:c0 + ic0 + iw] = True

        placed.append({**sp, "r0": r0, "c0": c0})
        top_rects.append((r0, c0, r0 + h - 1, c0 + w - 1))

    return placed


def _render(base: np.ndarray, placed: list[dict], bars: list[dict],
            bg: int) -> tuple[np.ndarray, np.ndarray]:
    """Render input (sprites in top area) and output (sprites docked in bars)."""
    inp, out = base.copy(), base.copy()

    for sp in placed:
        patch = sp["patch"].copy()
        outer_bar = bars[sp["outer_idx"]]
        h, c0 = sp["h"], sp["c0"]

        # Input: paste original sprite in top area
        paste(inp, patch, sp["r0"], c0, transparent=bg)

        # Output: color-swap and dock into outer bar, bottom-aligned
        transformed = patch.copy()
        transformed[transformed == outer_bar["fill"]] = outer_bar["border"]
        if sp["inner_idx"] is not None:
            inner_bar = bars[sp["inner_idx"]]
            transformed[transformed == inner_bar["fill"]] = outer_bar["fill"]
        paste(out, transformed, outer_bar["r_max"] - h + 1, c0, transparent=bg)

        # Output: inner sub-sprite also docks independently into its own bar
        if sp["inner_idx"] is not None:
            _, ic0, ih, iw = sp["inner_bbox"]
            inner_bar = bars[sp["inner_idx"]]
            inner_patch = np.full((ih, iw), inner_bar["border"], dtype=int)
            paste(out, inner_patch, inner_bar["r_max"] - ih + 1, c0 + ic0)

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt. Returns input grid or None on failure."""
    n_bars = random.randint(*BAR_COUNT_RANGE)
    n_sprites = random.randint(*SPRITE_COUNT_RANGE)

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    fills = palette[1:1 + n_bars]
    borders = palette[1 + n_bars:1 + 2 * n_bars]

    heights = _sample_bar_heights(n_bars)
    base, bars = _build_bars(bg, fills, borders, heights)
    top_h = bars[0]["r_min"]

    # Assign each sprite to a bar (up to 2 sprites per bar)
    outer_slots = [i for i in range(n_bars) for _ in range(2)]
    random.shuffle(outer_slots)
    sprites = [_make_sprite(outer_slots[i], bars) for i in range(n_sprites)]
    sprites.sort(key=lambda s: (s["inner_idx"] is None, -s["w"], -s["h"]))

    placed = _place_sprites(sprites, bars, top_h)
    if placed is None:
        return None

    inp, out = _render(base, placed, bars, bg)

    if get_bg(inp) != bg:
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
