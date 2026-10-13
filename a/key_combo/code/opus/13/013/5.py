from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = (1, 3, 4, 5, 6, 7, 8)
GEN_RETRIES = 400

SPRITE_TEMPLATES = (
    np.array(
        [
            [0, 1, 0],
            [1, 2, 1],
            [0, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 0],
            [1, 2, 1],
            [1, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 0],
            [1, 2, 1],
            [0, 1, 0],
        ],
        dtype=int,
    ),
)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _paint_patch(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int = 0) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != transparent
    view[mask] = patch[mask]


def _detect_bars(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    bars: list[dict] = []
    for r in range(h):
        nz = np.flatnonzero(grid[r] != bg)
        if len(nz) < max(6, w // 2):
            continue
        if np.any(np.diff(nz) != 1):
            continue
        c0 = int(nz[0])
        c1 = int(nz[-1])
        if c1 - c0 + 1 < max(6, w - 4):
            continue
        border = int(grid[r, c0])
        if border != int(grid[r, c1]):
            continue
        if c1 - c0 < 2:
            continue
        interior = grid[r, c0 + 1 : c1]
        fills = set(map(int, interior.tolist()))
        if len(fills) != 1:
            continue
        fill = int(next(iter(fills)))
        if fill == bg or fill == border:
            continue
        bars.append({"row": r, "c0": c0, "c1": c1, "fill": fill, "border": border})
    return bars


def _detect_sprites(grid: np.ndarray, bg: int, top_limit: int) -> list[dict]:
    region = grid[:top_limit].copy()
    components = _find_components(region != bg)
    sprites: list[dict] = []
    for coords in components:
        r0, c0, r1, c1 = _bbox(coords)
        patch = region[r0 : r1 + 1, c0 : c1 + 1].copy()
        colors = [int(v) for v in np.unique(patch) if int(v) != bg]
        if len(colors) != 2:
            continue
        counts = Counter(int(v) for v in patch.flatten() if int(v) != bg)
        outer, inner = counts.most_common(2)[0][0], counts.most_common(2)[1][0]
        inner_coords = np.argwhere(patch == inner)
        anchor_offset = int(np.floor(inner_coords[:, 1].mean()))
        sprites.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "outer": int(outer),
                "inner": int(inner),
                "anchor_offset": anchor_offset,
                "coords": coords,
            }
        )
    sprites.sort(key=lambda sp: (sp["bbox"][0], sp["bbox"][1]))
    return sprites


def _assign_sprites_to_bars(sprites: list[dict], bars: list[dict]) -> dict[int, int]:
    by_fill_bars: dict[int, list[int]] = {}
    by_fill_sprites: dict[int, list[int]] = {}
    for idx, bar in enumerate(bars):
        by_fill_bars.setdefault(int(bar["fill"]), []).append(idx)
    for idx, sprite in enumerate(sprites):
        by_fill_sprites.setdefault(int(sprite["outer"]), []).append(idx)

    assignments: dict[int, int] = {}
    used_bars: set[int] = set()
    for fill, sprite_indices in by_fill_sprites.items():
        bar_indices = by_fill_bars.get(int(fill), [])
        assert len(sprite_indices) <= len(bar_indices), "every sprite outer color must have a target bar"

        unmatched = list(bar_indices)
        for sprite_idx in sprite_indices:
            sprite = sprites[sprite_idx]
            border_matches = [bar_idx for bar_idx in unmatched if int(bars[bar_idx]["border"]) == int(sprite["inner"])]
            if len(border_matches) == 1:
                picked = border_matches[0]
                assignments[sprite_idx] = picked
                used_bars.add(picked)
                unmatched.remove(picked)

        remaining_sprites = [idx for idx in sprite_indices if idx not in assignments]
        remaining_bars = [idx for idx in bar_indices if idx not in used_bars]
        remaining_sprites.sort(key=lambda idx: sprites[idx]["bbox"][:2])
        remaining_bars.sort(key=lambda idx: bars[idx]["row"])
        assert len(remaining_sprites) <= len(remaining_bars), "not enough bars for unmatched sprites"
        for sprite_idx, bar_idx in zip(remaining_sprites, remaining_bars):
            assignments[sprite_idx] = bar_idx
            used_bars.add(bar_idx)
    return assignments


def _swap_patch_colors(patch: np.ndarray, outer: int, inner: int, border: int, fill: int) -> np.ndarray:
    out = patch.copy()
    out[patch == int(outer)] = int(border)
    out[patch == int(inner)] = int(fill)
    return out


def _draw_connector(out: np.ndarray, upper_row: int, lower_row: int, upper_col: int, lower_col: int, color: int) -> None:
    start_r = upper_row + 1
    end_r = lower_row - 1
    if start_r > end_r:
        return
    if upper_col == lower_col:
        out[start_r : end_r + 1, upper_col] = int(color)
        return
    mid_r = (start_r + end_r) // 2
    out[start_r : mid_r + 1, upper_col] = int(color)
    out[mid_r, min(upper_col, lower_col) : max(upper_col, lower_col) + 1] = int(color)
    out[mid_r : end_r + 1, lower_col] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock two-tone sprites into matching bars, swap their colors, then connect adjacent equal post-swap colors.

    Input structure:
    - The background is black.
    - The top area contains several small two-tone sprites made of an outer color surrounding an inner color.
    - Lower rows contain several long one-row horizontal bars; each bar has one fill color in the middle and one border color at its two ends.

    Rule:
    - Match each sprite to a bar whose fill color equals the sprite outer color, preserving the sprite's horizontal position.
    - Dock the sprite so its bottom row sits on the matched bar row, then recolor sprite outer pixels to the bar border color and sprite inner pixels to the bar fill color.
    - Erase the sprites from their original top positions.
    - For each vertically adjacent pair of bars, if one docked sprite on the upper bar and one docked sprite on the lower bar have the same post-swap inner color, connect them through the gap with that color.
    - When the two sprite anchor columns match, draw a straight vertical segment; otherwise draw an L-shaped path that stays inside the inter-bar gap.

    Key insight:
    - The connection test uses the sprite color after docking and swapping, so two originally different sprites can connect only after tracing how each one inherits its target bar's colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    bars = _detect_bars(g, bg)
    assert len(bars) >= 2, "expected multiple bars"
    top_limit = min(int(bar["row"]) for bar in bars)
    sprites = _detect_sprites(g, bg, top_limit)
    assert sprites, "expected sprites above the bars"
    assignments = _assign_sprites_to_bars(sprites, bars)

    out = g.copy()
    for sprite in sprites:
        for r, c in sprite["coords"]:
            out[r, c] = bg

    docked_by_bar: dict[int, list[dict]] = {i: [] for i in range(len(bars))}
    for sprite_idx, bar_idx in assignments.items():
        sprite = sprites[sprite_idx]
        bar = bars[bar_idx]
        patch = _swap_patch_colors(sprite["patch"], sprite["outer"], sprite["inner"], bar["border"], bar["fill"])
        h, w = patch.shape
        place_r = int(bar["row"]) - h + 1
        place_c = int(sprite["bbox"][1])
        assert place_r >= 0 and 0 <= place_c and place_c + w <= g.shape[1]
        _paint_patch(out, patch, place_r, place_c, transparent=bg)
        docked_by_bar[bar_idx].append(
            {
                "anchor_col": int(place_c + sprite["anchor_offset"]),
                "post_inner": int(bar["fill"]),
            }
        )

    for upper_idx in range(len(bars) - 1):
        lower_idx = upper_idx + 1
        matches = [
            (upper_sprite, lower_sprite)
            for upper_sprite in docked_by_bar.get(upper_idx, [])
            for lower_sprite in docked_by_bar.get(lower_idx, [])
            if int(upper_sprite["post_inner"]) == int(lower_sprite["post_inner"])
        ]
        if not matches:
            continue
        assert len(matches) == 1, "generator assumes at most one connection per adjacent bar pair"
        upper_sprite, lower_sprite = matches[0]
        _draw_connector(
            out,
            int(bars[upper_idx]["row"]),
            int(bars[lower_idx]["row"]),
            int(upper_sprite["anchor_col"]),
            int(lower_sprite["anchor_col"]),
            int(upper_sprite["post_inner"]),
        )

    return out


def _make_sprite_patch(outer: int, inner: int) -> np.ndarray:
    base = random.choice(SPRITE_TEMPLATES).copy()
    k = random.randrange(4)
    patch = np.rot90(base, k)
    out = np.zeros_like(patch, dtype=int)
    out[patch == 1] = int(outer)
    out[patch == 2] = int(inner)
    return out


def _choose_pair_layout(n_bars: int) -> list[tuple[int, int]]:
    options: list[list[tuple[int, int]]] = []
    for start in range(n_bars - 1):
        options.append([(start, start + 1)])
    for start1 in range(n_bars - 1):
        for start2 in range(start1 + 2, n_bars - 1):
            options.append([(start1, start1 + 1), (start2, start2 + 1)])
    valid = [layout for layout in options if len({idx for pair in layout for idx in pair}) == 2 * len(layout)]
    one_pair = [layout for layout in valid if len(layout) == 1]
    two_pair = [layout for layout in valid if len(layout) == 2]
    if n_bars >= 4 and two_pair and random.random() < 0.35:
        return random.choice(two_pair)
    return random.choice(one_pair)


def _sample_bar_colors(n_bars: int, pair_layout: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    fills = [None] * n_bars
    used_fill_colors: list[int] = []
    pair_indices = set()
    for a, b in pair_layout:
        fill = random.choice([color for color in PALETTE if color not in used_fill_colors] or list(PALETTE))
        fills[a] = fill
        fills[b] = fill
        used_fill_colors.append(fill)
        pair_indices.add(a)
        pair_indices.add(b)
    for idx in range(n_bars):
        if idx in pair_indices:
            continue
        choices = [color for color in PALETTE if color not in used_fill_colors]
        if not choices:
            choices = list(PALETTE)
        fills[idx] = random.choice(choices)
        used_fill_colors.append(fills[idx])

    borders: list[int] = []
    available = list(PALETTE)
    random.shuffle(available)
    for idx in range(n_bars):
        choices = [color for color in PALETTE if color != fills[idx] and color not in borders]
        if not choices:
            choices = [color for color in PALETTE if color != fills[idx]]
        borders.append(random.choice(choices))
    return [int(v) for v in fills], [int(v) for v in borders]


def _sample_geometry(n_bars: int) -> tuple[int, int, list[int], int, int]:
    w = random.randint(18, 24)
    top_room = random.randint(6, 8)
    gaps = [random.randint(2, 3) for _ in range(n_bars - 1)]
    first_row = top_room + 1
    rows = [first_row]
    for gap in gaps:
        rows.append(rows[-1] + gap + 1)
    h = rows[-1] + 1 + random.randint(1, 3)
    return h, w, rows, 1, w - 2


def _plan_sprite_columns(
    rows: list[int], bar_left: int, bar_right: int, pair_layout: list[tuple[int, int]]
) -> list[int] | None:
    n_bars = len(rows)
    min_c = bar_left + 1
    max_c = bar_right - 3
    if min_c > max_c:
        return None
    columns = [None] * n_bars
    paired_indices = set()
    for a, b in pair_layout:
        paired_indices.add(a)
        paired_indices.add(b)
        if random.random() < 0.5:
            col = random.randint(min_c, max_c)
            columns[a] = col
            columns[b] = col
        else:
            left = random.randint(min_c, max_c)
            choices = [c for c in range(min_c, max_c + 1) if abs(c - left) >= 2]
            if not choices:
                return None
            right = random.choice(choices)
            columns[a], columns[b] = left, right

    for idx in range(n_bars):
        if idx in paired_indices:
            continue
        columns[idx] = random.randint(min_c, max_c)
    return [int(c) for c in columns]


def _place_top_sprites(grid: np.ndarray, sprites: list[np.ndarray], columns: list[int], top_room: int) -> bool:
    occupied = np.zeros_like(grid, dtype=bool)
    placements: list[tuple[int, int]] = []
    order = list(range(len(sprites)))
    order.sort(key=lambda idx: (columns[idx], idx))
    for idx in order:
        patch = sprites[idx]
        h, w = patch.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(0, top_room - h + 1):
            c0 = columns[idx]
            rr0 = max(0, r0 - 1)
            rr1 = min(grid.shape[0], r0 + h + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(grid.shape[1], c0 + w + 1)
            region = occupied[rr0:rr1, cc0:cc1]
            if region.any():
                continue
            candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        placements.append((r0, c0))
        occupied[r0 : r0 + h, c0 : c0 + w] |= patch != BG

    for idx, (r0, c0) in zip(order, placements):
        _paint_patch(grid, sprites[idx], r0, c0, transparent=BG)
    return True


def _count_connectors(output: np.ndarray, input_grid: np.ndarray) -> int:
    return int(np.count_nonzero((output != BG) & (input_grid == BG)))


def _generate_once() -> np.ndarray | None:
    n_bars = random.randint(3, 5)
    pair_layout = _choose_pair_layout(n_bars)
    h, w, rows, bar_left, bar_right = _sample_geometry(n_bars)
    fills, borders = _sample_bar_colors(n_bars, pair_layout)
    columns = _plan_sprite_columns(rows, bar_left, bar_right, pair_layout)
    if columns is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for idx, row in enumerate(rows):
        grid[row, bar_left] = borders[idx]
        grid[row, bar_right] = borders[idx]
        grid[row, bar_left + 1 : bar_right] = fills[idx]

    sprites = [_make_sprite_patch(fills[idx], borders[idx]) for idx in range(n_bars)]
    if not _place_top_sprites(grid, sprites, columns, rows[0]):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if _count_connectors(out, grid) <= 0:
        return None

    paired_fills = {fills[a] for a, _ in pair_layout}
    for i in range(n_bars - 1):
        same_fill = fills[i] == fills[i + 1]
        if same_fill != (fills[i] in paired_fills and fills[i + 1] in paired_fills):
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_once()
        if grid is not None and grid.ndim == 2 and max(grid.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
