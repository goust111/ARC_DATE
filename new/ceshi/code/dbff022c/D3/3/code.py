# Variant of dbff022c
# Dimension: D3 — 调色板阅读方向的推断规则 (symbol_instructions)
# Change: 阅读方向由调色板所在边缘决定：上边缘→从上到下，下边缘→从下到上，左边缘→从左到右；第一行/列始终是边框颜色... → 同一网格中多个调色板使用不同阅读方向

import random

import numpy as np

from arc_utils import components_mask, neighbors4


# ---------------------------------------------------------------------------
# Design analysis for the D3 variant
#
# a) Concrete behavior:
#    Each grid contains exactly two edge palettes on opposite sides: either
#    left+right or top+bottom. The two palettes may share border-key colors,
#    and shared keys map to different fill colors. Example:
#      left palette contains 4->2, right palette contains 4->7.
#      A hollow border-4 sprite near the left side fills with 2, while a
#      hollow border-4 sprite near the right side fills with 7.
#    The original solver used one global legend. That fails here because the
#    same border color can no longer determine one global fill color.
#
# b) Complexity calibration:
#    Only two palettes exist and only one grouping axis exists per grid, so the
#    rule is still learnable from a few examples. To make the difference visible,
#    every generated sample uses at least one shared border key on both sides.
#
# c) Implicit-assumption audit:
#    - The old single-palette search is invalid because two valid palettes must
#      be kept instead of competing for "best palette".
#    - The old global border->fill mapping is invalid because shared keys can
#      map differently in the two palettes.
#    - Hole filling therefore must choose a palette per enclosed region.
#
# d) Placement legality:
#    Palettes avoid corners so a left/right palette cannot masquerade as a
#    top/bottom palette, or vice versa. Sprites stay off the outer border and
#    are separated from one another and from palettes by at least one bg cell.
#
# e) Visual observability:
#    The side-based grouping is visible: same-colored hollow sprites on opposite
#    sides fill differently according to the nearer palette.
#
# f) Visual cleanliness:
#    Only opposite-edge pairs are used, never corner-sharing mixed edges, which
#    keeps the border legends isolated and easy to parse by eye.
#
# g) Color-change inferability:
#    Each palette has only 2-4 entries, and one shared key is guaranteed to be
#    used on both sides with different fills. The cue is spatial, not numeric.
# ---------------------------------------------------------------------------


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _palette_mask(shape: tuple[int, int], side: str, start: int, end: int) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if side == "top":
        mask[0:2, start:end] = True
    elif side == "bottom":
        mask[h - 2 : h, start:end] = True
    elif side == "left":
        mask[start:end, 0:2] = True
    else:
        mask[start:end, w - 2 : w] = True
    return mask


def _palette_lines(g: np.ndarray, side: str, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = g.shape
    if side == "top":
        return g[0, start:end], g[1, start:end]
    if side == "bottom":
        return g[h - 1, start:end], g[h - 2, start:end]
    if side == "left":
        return g[start:end, 0], g[start:end, 1]
    return g[start:end, w - 1], g[start:end, w - 2]


def _scan_nonbg_runs(arr: np.ndarray, bg: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    vals = [int(x) for x in arr.tolist()]
    for i, v in enumerate(vals + [bg]):
        if v != bg and start is None:
            start = i
        elif v == bg and start is not None:
            if i - start >= 2:
                runs.append((start, i))
            start = None
    return runs


def _palette_is_isolated(g: np.ndarray, bg: int, side: str, start: int, end: int) -> bool:
    h, w = g.shape
    pal = _palette_mask((h, w), side, start, end)
    for r, c in zip(*np.where(pal), strict=False):
        r = int(r)
        c = int(c)
        v = int(g[r, c])
        if v == bg:
            return False
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and not pal[rr, cc] and int(g[rr, cc]) == v:
                return False
    return True


def _extract_palette(g: np.ndarray, bg: int, side: str) -> dict | None:
    h, w = g.shape
    if side == "top":
        if h < 2:
            return None
        outer = g[0, :]
        inner = g[1, :]
    elif side == "bottom":
        if h < 2:
            return None
        outer = g[h - 1, :]
        inner = g[h - 2, :]
    elif side == "left":
        if w < 2:
            return None
        outer = g[:, 0]
        inner = g[:, 1]
    else:
        if w < 2:
            return None
        outer = g[:, w - 1]
        inner = g[:, w - 2]

    best = None
    for start, end in _scan_nonbg_runs(outer, bg):
        if any(int(inner[i]) == bg for i in range(start, end)):
            continue
        border_line, fill_line = _palette_lines(g, side, start, end)
        border_vals = [int(x) for x in border_line.tolist()]
        fill_vals = [int(x) for x in fill_line.tolist()]
        if len(set(border_vals)) != len(border_vals):
            continue
        if len(set(fill_vals)) != len(fill_vals):
            continue
        if not _palette_is_isolated(g, bg, side, start, end):
            continue
        score = (end - start, len(set(border_vals + fill_vals)))
        cand = (score, start, end, border_vals, fill_vals)
        if best is None or cand[0] > best[0]:
            best = cand

    if best is None:
        return None

    _, start, end, border_vals, fill_vals = best
    mask = _palette_mask(g.shape, side, int(start), int(end))
    return {
        "side": side,
        "start": int(start),
        "end": int(end),
        "mapping": {int(b): int(f) for b, f in zip(border_vals, fill_vals, strict=False)},
        "mask": mask,
        "cells": [(int(r), int(c)) for r, c in zip(*np.where(mask), strict=False)],
    }


def _find_palettes(g: np.ndarray, bg: int) -> list[dict]:
    palettes = []
    for side in ("top", "bottom", "left", "right"):
        info = _extract_palette(g, bg, side)
        if info is not None:
            palettes.append(info)

    sides = {p["side"] for p in palettes}
    if {"top", "bottom"} <= sides:
        return [p for p in palettes if p["side"] in {"top", "bottom"}]
    if {"left", "right"} <= sides:
        return [p for p in palettes if p["side"] in {"left", "right"}]
    raise AssertionError("_find_palettes: expected an opposite-edge palette pair")


def _palette_distance(pts: np.ndarray, palette: dict) -> int:
    best = 10**9
    for r, c in pts:
        rr = int(r)
        cc = int(c)
        for pr, pc in palette["cells"]:
            d = abs(rr - pr) + abs(cc - pc)
            if d < best:
                best = d
    return int(best)


def _choose_palette_for_border(border: int, pts: np.ndarray, palettes: list[dict]) -> dict | None:
    matches = [p for p in palettes if border in p["mapping"]]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    return min(matches, key=lambda p: (_palette_distance(pts, p), p["side"]))


def _fill_enclosed_bg_regions(out: np.ndarray, bg: int, palettes: list[dict]) -> None:
    h, w = out.shape
    pal_mask = np.zeros((h, w), dtype=bool)
    for pal in palettes:
        pal_mask |= pal["mask"]

    bg_mask = (out == bg) & (~pal_mask)
    comps = components_mask(bg_mask, connectivity=4)
    for pts in comps:
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        adj_counts: dict[int, int] = {}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and not pal_mask[rr, cc]:
                    v = int(out[rr, cc])
                    if v != bg:
                        adj_counts[v] = adj_counts.get(v, 0) + 1
        if not adj_counts:
            continue

        chosen_palette = None
        chosen_border = None
        for border, _cnt in sorted(adj_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            chosen_palette = _choose_palette_for_border(int(border), pts, palettes)
            if chosen_palette is not None:
                chosen_border = int(border)
                break
        if chosen_palette is None or chosen_border is None:
            continue

        fill = int(chosen_palette["mapping"][chosen_border])
        for r, c in pts:
            out[int(r), int(c)] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hollow sprites by reading two edge legends instead of one global legend.

    Input structure:
    - Background cells dominate the border, while hollow one-color sprites sit away from the edges.
    - Two small 2×N or N×2 palette blocks appear on opposite edges, and each block lists border colors on its outer line with matching fill colors on the inner line.

    Rule:
    - Detect both palette blocks and decode each one using the read direction implied by its own edge.
    - Find every enclosed background hole inside a sprite, determine its surrounding border color, and resolve that color through the compatible palette nearest to that sprite group.
    - Fill the hole with the fill color from that selected palette; holes whose border color appears in no palette stay unchanged.

    Key insight:
    - The same border color can appear in both legends with different fill targets, so color alone is insufficient; the sprite must be matched to the correct edge legend before applying the mapping.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    palettes = _find_palettes(g, bg)
    assert len(palettes) == 2, "main: expected exactly two palettes"

    out = g.copy()
    _fill_enclosed_bg_regions(out, bg, palettes)
    return out


# --------------- generator (c) ---------------

# -- Configuration constants --------------------------------------------------
_GRID_H_MIN, _GRID_H_MAX = 5, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15

_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 4
_P_TOP_BOTTOM_PAIR = 0.35

_SPRITE_H_MAX = 7
_SPRITE_W_MAX = 6
_SPRITE_DIM_SUM_CAP = 10
_SPRITE_CELL_MAX = 16

_P_HORIZ_WALL = 0.3
_P_VERT_WALL = 0.3
_P_BORDER_REMOVAL = 0.4
_P_DIAMOND = 0.2
_P_EXTRA_FILLED = 0.4

_MAX_RETRIES = 500
# -- End configuration --------------------------------------------------------


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _make_diamond() -> np.ndarray:
    s = np.zeros((3, 3), dtype=bool)
    s[0, 1] = s[1, 0] = s[1, 2] = s[2, 1] = True
    return s


def _make_hollow_sprite(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    sprite = np.ones((h, w), dtype=bool)
    if h >= 3 and w >= 3:
        sprite[1:-1, 1:-1] = False
    if h >= 5 and w >= 4 and rng.random() < _P_HORIZ_WALL:
        br = rng.randint(2, h - 2)
        n = rng.randint(1, w - 2)
        s = rng.randint(1, w - n - 1)
        sprite[br, s : s + n] = True
    if h >= 4 and w >= 5 and rng.random() < _P_VERT_WALL:
        bc = rng.randint(2, w - 2)
        n = rng.randint(1, h - 2)
        s = rng.randint(1, h - n - 1)
        sprite[s : s + n, bc] = True
    if rng.random() < _P_BORDER_REMOVAL:
        bcells = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if sprite[r, c] and (r in (0, h - 1) or c in (0, w - 1))
        ]
        rng.shuffle(bcells)
        for cr, cc in bcells[: rng.randint(1, min(4, len(bcells)))]:
            test = sprite.copy()
            test[cr, cc] = False
            if _get_enclosed_holes(test):
                sprite = test
    if not _get_enclosed_holes(sprite) or int(sprite.sum()) > _SPRITE_CELL_MAX:
        return None
    return sprite


def _build_palette(side: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    n = len(border_keys)
    if side in ("top", "bottom"):
        pal = np.zeros((2, n), dtype=int)
        outer, inner = (0, 1) if side == "top" else (1, 0)
        pal[outer] = border_keys
        pal[inner] = fill_values
    else:
        pal = np.zeros((n, 2), dtype=int)
        outer, inner = (0, 1) if side == "left" else (1, 0)
        pal[:, outer] = border_keys
        pal[:, inner] = fill_values
    return pal


def _place_palette(grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, side: str, rng: random.Random) -> None:
    h, w = grid.shape
    pal_h, pal_w = pal.shape
    if side == "top":
        pr0, pc0 = 0, rng.randint(1, w - pal_w - 1)
    elif side == "bottom":
        pr0, pc0 = h - pal_h, rng.randint(1, w - pal_w - 1)
    elif side == "left":
        pr0, pc0 = rng.randint(1, h - pal_h - 1), 0
    else:
        pr0, pc0 = rng.randint(1, h - pal_h - 1), w - pal_w
    grid[pr0 : pr0 + pal_h, pc0 : pc0 + pal_w] = pal
    occ[max(0, pr0 - 1) : min(h, pr0 + pal_h + 1), max(0, pc0 - 1) : min(w, pc0 + pal_w + 1)] = True


def _find_valid_positions(
    h: int,
    w: int,
    occ: np.ndarray,
    mask: np.ndarray,
    holes: list[np.ndarray],
    predicate,
) -> list[tuple[int, int]]:
    sh, sw = mask.shape
    positions = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            if r0 == 0 or c0 == 0 or r0 + sh == h or c0 + sw == w:
                continue
            if occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)].any():
                continue
            if any(
                r0 + hr == 0 or c0 + hc == 0 or r0 + hr == h - 1 or c0 + hc == w - 1
                for hpts in holes
                for hr, hc in hpts
            ):
                continue
            if not predicate(r0, c0, sh, sw):
                continue
            positions.append((r0, c0))
    return positions


def _paint_sprite(
    grid: np.ndarray,
    r0: int,
    c0: int,
    mask: np.ndarray,
    color: int,
    fill: int | None,
    holes: list[np.ndarray],
) -> None:
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = color
    if fill is not None:
        for hpts in holes:
            for hr, hc in hpts:
                grid[r0 + hr, c0 + hc] = fill


def _choose_axis(h: int, rng: random.Random) -> tuple[str, tuple[str, str]]:
    if h >= 9 and rng.random() < _P_TOP_BOTTOM_PAIR:
        return "tb", ("top", "bottom")
    return "lr", ("left", "right")


def _make_pair_mappings(rng: random.Random, n: int) -> tuple[dict[str, dict[int, int]], int, int]:
    colors = list(range(1, 10))
    rng.shuffle(colors)
    shared_key = colors.pop()
    distractor_color = colors.pop()

    needed_extra = 2 * (n - 1)
    if len(colors) < needed_extra:
        return {}, -1, -1
    extras = [colors.pop() for _ in range(needed_extra)]
    left_keys = [shared_key] + extras[: n - 1]
    right_keys = [shared_key] + extras[n - 1 :]

    left_fill_shared = rng.randint(1, 9)
    right_fill_shared = rng.randint(1, 9)
    while right_fill_shared == left_fill_shared:
        right_fill_shared = rng.randint(1, 9)

    def make_fills(first_fill: int) -> list[int]:
        fills = [first_fill]
        pool = list(range(1, 10))
        rng.shuffle(pool)
        for c in pool:
            if c not in fills:
                fills.append(c)
            if len(fills) == n:
                break
        return fills

    left_fills = make_fills(left_fill_shared)
    right_fills = make_fills(right_fill_shared)

    left_pairs = list(zip(left_keys, left_fills, strict=False))
    right_pairs = list(zip(right_keys, right_fills, strict=False))
    rng.shuffle(left_pairs)
    rng.shuffle(right_pairs)

    left_map = {int(k): int(v) for k, v in left_pairs}
    right_map = {int(k): int(v) for k, v in right_pairs}
    return {"a": left_map, "b": right_map}, int(shared_key), int(distractor_color)


def _zone_predicate(axis: str, side: str | None, h: int, w: int):
    row_mid = (h - 1) / 2.0
    col_mid = (w - 1) / 2.0
    if axis == "lr":
        if side == "left":
            return lambda r0, c0, sh, sw: c0 + (sw - 1) / 2.0 < col_mid
        if side == "right":
            return lambda r0, c0, sh, sw: c0 + (sw - 1) / 2.0 > col_mid
        return lambda r0, c0, sh, sw: True
    if side == "top":
        return lambda r0, c0, sh, sw: r0 + (sh - 1) / 2.0 < row_mid
    if side == "bottom":
        return lambda r0, c0, sh, sw: r0 + (sh - 1) / 2.0 > row_mid
    return lambda r0, c0, sh, sw: True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    axis, sides = _choose_axis(h, rng)

    max_n = min(_PALETTE_N_MAX, (h - 2) if axis == "lr" else (w - 2))
    if max_n < _PALETTE_N_MIN:
        return None
    n = rng.randint(_PALETTE_N_MIN, max_n)

    max_sh = min(_SPRITE_H_MAX, h - (6 if axis == "tb" else 2))
    max_sw = min(_SPRITE_W_MAX, w - (6 if axis == "lr" else 2))
    if h <= 5:
        max_sh = min(max_sh, 3)
    if max_sh < 3 or max_sw < 3:
        return None

    mappings, shared_key, distractor_color = _make_pair_mappings(rng, n)
    if not mappings:
        return None

    palette_maps = {sides[0]: mappings["a"], sides[1]: mappings["b"]}

    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    for side in sides:
        mapping = palette_maps[side]
        keys = list(mapping.keys())
        fills = [mapping[k] for k in keys]
        pal = _build_palette(side, keys, fills)
        _place_palette(grid, occ, pal, side, rng)

    sprite_specs = [
        {"group": sides[0], "color": shared_key, "fill": palette_maps[sides[0]][shared_key]},
        {"group": sides[1], "color": shared_key, "fill": palette_maps[sides[1]][shared_key]},
        {"group": None, "color": distractor_color, "fill": None},
    ]

    if rng.random() < _P_EXTRA_FILLED:
        extra_side = rng.choice(list(sides))
        extra_keys = [k for k in palette_maps[extra_side] if k != shared_key]
        if extra_keys:
            extra_key = rng.choice(extra_keys)
            sprite_specs.append({"group": extra_side, "color": extra_key, "fill": palette_maps[extra_side][extra_key]})

    rng.shuffle(sprite_specs)

    placed = []
    for spec in sprite_specs:
        made = False
        for _ in range(40):
            sh = rng.randint(3, max_sh)
            sw = rng.randint(3, max_sw)
            if sh + sw > _SPRITE_DIM_SUM_CAP:
                sw = max(3, _SPRITE_DIM_SUM_CAP - sh)
            if sw > max_sw:
                sw = max_sw
            if sh == 3 and sw == 3 and rng.random() < _P_DIAMOND:
                mask = _make_diamond()
            else:
                mask = _make_hollow_sprite(sh, sw, rng)
            if mask is None:
                continue
            holes = _get_enclosed_holes(mask)
            predicate = _zone_predicate(axis, spec["group"], h, w)
            positions = _find_valid_positions(h, w, occ, mask, holes, predicate)
            if not positions:
                continue
            r0, c0 = rng.choice(positions)
            _paint_sprite(grid, r0, c0, mask, spec["color"], spec["fill"], holes)
            placed.append((r0, c0, mask, spec["fill"]))
            occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True
            made = True
            break
        if not made:
            return None

    output = grid.copy()
    inp = grid.copy()
    for r0, c0, mask, fill in placed:
        if fill is None:
            continue
        for hpts in _get_enclosed_holes(mask):
            for hr, hc in hpts:
                inp[r0 + hr, c0 + hc] = bg

    border = np.concatenate([inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
