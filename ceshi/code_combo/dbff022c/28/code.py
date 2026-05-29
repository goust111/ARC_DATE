# Combo variant of dbff022c (index 28)
# combo_spec: D3/3+D10/0
# Name: 多调色板不同方向 + 严格一一对应

import random

import numpy as np

from arc_utils import components_mask, neighbors4


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior:
#    Start from the original task: one edge palette globally maps border colors
#    to fill colors, some palette entries may be unused, and one distractor
#    sprite may have no palette entry at all.
#
#    In this combo variant there are exactly two palettes on opposite edges.
#    Each palette must be decoded using the direction implied by its own edge,
#    and every palette entry is used by exactly one hollow sprite.
#
#    Concrete example:
#      top palette outer/inner lines encode 4->2 and 7->6
#      bottom palette outer/inner lines encode 4->9 and 8->3
#      There are exactly four hollow sprites:
#        - a top-group border-4 sprite fills with 2
#        - a top-group border-7 sprite fills with 6
#        - a bottom-group border-4 sprite fills with 9
#        - a bottom-group border-8 sprite fills with 3
#      The original solver would fail because border color 4 no longer has one
#      global fill target, and because ignoring one of the palettes would miss
#      half of the sprites.
#
# b) Complexity calibration:
#    The visual change must be obvious without becoming noisy. Each sample uses
#    two palettes, exactly four mapped sprites, and one border color shared by
#    both palettes with different fills. That keeps the search space small but
#    makes the multi-palette grouping indispensable.
#
# c) Implicit-assumption audit:
#    - Old assumption: a single "best" palette exists. Broken here; we must
#      detect and keep both palettes.
#    - Old assumption: border color alone chooses a fill color. Broken here
#      because the shared key appears in both palettes with different outputs.
#    - Old assumption: some enclosed regions may remain bg because of a
#      distractor sprite. Removed here; every generated sprite is mapped.
#    - New solver assumption: if a border color appears in both palettes, the
#      intended sprite lies spatially nearer to its own palette. The generator
#      guarantees this by placing palette groups on opposite halves.
#
# d) Placement legality:
#    Palettes avoid corners, sprites stay off the outer border, and all objects
#    keep at least one background cell of separation. This prevents accidental
#    overlap and prevents palette detection from absorbing sprite fragments.
#
# e) Visual observability:
#    Human solvers can see two legends on opposite edges and can compare the
#    repeated shared-key color across both halves. The same border color filling
#    differently in opposite groups is the core visible cue.
#
# f) Visual cleanliness:
#    Only one palette pair is present, never mixed-edge triples. Each palette
#    has length two, so the legends remain compact and the grid is not crowded
#    even though all entries are used.
#
# g) Color-change inferability:
#    The mapping is still local and structured: single-color hollow borders,
#    compact legends, and one repeated key across palettes. There are no unused
#    entries and no unmapped sprites, so every non-bg clue participates in the
#    rule instead of adding noise.
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
        rr = int(r)
        cc = int(c)
        v = int(g[rr, cc])
        if v == bg:
            return False
        for nr, nc in neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and not pal[nr, nc] and int(g[nr, nc]) == v:
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
    raise AssertionError("_find_palettes: expected one opposite-edge palette pair")


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
    for pts in components_mask(bg_mask, connectivity=4):
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
    Fill hollow sprites by combining two edge legends with exact entry usage.

    Input structure:
    - Background cells dominate the border, and hollow single-color sprites sit away from the edge.
    - Exactly two compact 2xN or Nx2 palette blocks appear on opposite edges, and each palette keeps border colors on its outer line with fill colors on its inner line.
    - Every palette entry corresponds to one sprite, so there are no unused legend pairs and no unmapped hollow distractors.

    Rule:
    - Detect both palettes and decode each one according to the reading direction implied by that palette's edge.
    - For every enclosed background hole inside a sprite, identify the surrounding border color, then choose the compatible palette for that region.
    - Fill the hole with the mapped color from the selected palette, leaving the palettes themselves unchanged.

    Key insight:
    - A border color may appear in both legends with different fill targets, so the solver must use both the color match and the sprite's spatial grouping relative to the two edge palettes.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    palettes = _find_palettes(g, bg)
    assert len(palettes) == 2, "main: expected exactly two palettes"

    out = g.copy()
    _fill_enclosed_bg_regions(out, bg, palettes)
    return out


# --------------- generator (c) ---------------

_GRID_H_MIN, _GRID_H_MAX = 8, 13
_GRID_W_MIN, _GRID_W_MAX = 11, 15

_PALETTE_LEN = 2
_P_TOP_BOTTOM_PAIR = 0.25

_SPRITE_H_MAX = 5
_SPRITE_W_MAX = 5
_SPRITE_DIM_SUM_CAP = 8
_SPRITE_CELL_MAX = 12

_P_HORIZ_WALL = 0.3
_P_VERT_WALL = 0.3
_P_BORDER_REMOVAL = 0.25
_P_DIAMOND = 0.1

_MAX_RETRIES = 700


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _make_diamond() -> np.ndarray:
    s = np.zeros((3, 3), dtype=bool)
    s[0, 1] = True
    s[1, 0] = True
    s[1, 2] = True
    s[2, 1] = True
    return s


def _make_hollow_sprite(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    sprite = np.ones((h, w), dtype=bool)
    sprite[1:-1, 1:-1] = False

    if h >= 5 and w >= 4 and rng.random() < _P_HORIZ_WALL:
        r = rng.randint(2, h - 2)
        n = rng.randint(1, w - 2)
        c0 = rng.randint(1, w - 1 - n)
        sprite[r, c0 : c0 + n] = True

    if h >= 4 and w >= 5 and rng.random() < _P_VERT_WALL:
        c = rng.randint(2, w - 2)
        n = rng.randint(1, h - 2)
        r0 = rng.randint(1, h - 1 - n)
        sprite[r0 : r0 + n, c] = True

    if rng.random() < _P_BORDER_REMOVAL:
        border_cells = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if sprite[r, c] and (r in (0, h - 1) or c in (0, w - 1))
        ]
        rng.shuffle(border_cells)
        for rr, cc in border_cells[: rng.randint(1, min(3, len(border_cells)))]:
            test = sprite.copy()
            test[rr, cc] = False
            if _get_enclosed_holes(test):
                sprite = test

    if not _get_enclosed_holes(sprite):
        return None
    if int(sprite.sum()) > _SPRITE_CELL_MAX:
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
    ph, pw = pal.shape
    if side == "top":
        r0, c0 = 0, rng.randint(1, w - pw - 1)
    elif side == "bottom":
        r0, c0 = h - ph, rng.randint(1, w - pw - 1)
    elif side == "left":
        r0, c0 = rng.randint(1, h - ph - 1), 0
    else:
        r0, c0 = rng.randint(1, h - ph - 1), w - pw
    grid[r0 : r0 + ph, c0 : c0 + pw] = pal
    occ[max(0, r0 - 1) : min(h, r0 + ph + 1), max(0, c0 - 1) : min(w, c0 + pw + 1)] = True


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
    fill: int,
    holes: list[np.ndarray],
) -> None:
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = color
    for hpts in holes:
        for hr, hc in hpts:
            grid[r0 + hr, c0 + hc] = fill


def _choose_axis(h: int, rng: random.Random) -> tuple[str, tuple[str, str]]:
    if h >= 10 and rng.random() < _P_TOP_BOTTOM_PAIR:
        return "tb", ("top", "bottom")
    return "lr", ("left", "right")


def _make_fill_list(rng: random.Random, shared_fill: int, n: int) -> list[int]:
    fills = [shared_fill]
    pool = list(range(1, 10))
    rng.shuffle(pool)
    for color in pool:
        if color not in fills:
            fills.append(color)
        if len(fills) == n:
            break
    return fills


def _make_pair_mappings(rng: random.Random) -> tuple[dict[str, dict[int, int]], int]:
    colors = list(range(1, 10))
    rng.shuffle(colors)
    shared_key = colors.pop()
    unique_a = colors.pop()
    unique_b = colors.pop()

    shared_fill_a = rng.choice([c for c in range(1, 10) if c != shared_key])
    shared_fill_b = rng.choice([c for c in range(1, 10) if c not in {shared_key, shared_fill_a}])

    keys_a = [shared_key, unique_a]
    keys_b = [shared_key, unique_b]
    fills_a = _make_fill_list(rng, shared_fill_a, _PALETTE_LEN)
    fills_b = _make_fill_list(rng, shared_fill_b, _PALETTE_LEN)

    pairs_a = list(zip(keys_a, fills_a, strict=False))
    pairs_b = list(zip(keys_b, fills_b, strict=False))
    rng.shuffle(pairs_a)
    rng.shuffle(pairs_b)

    map_a = {int(k): int(v) for k, v in pairs_a}
    map_b = {int(k): int(v) for k, v in pairs_b}
    return {"a": map_a, "b": map_b}, int(shared_key)


def _zone_predicate(axis: str, side: str, h: int, w: int):
    row_mid = (h - 1) / 2.0
    col_mid = (w - 1) / 2.0
    if axis == "lr":
        if side == "left":
            return lambda r0, c0, sh, sw: c0 + (sw - 1) / 2.0 < col_mid
        return lambda r0, c0, sh, sw: c0 + (sw - 1) / 2.0 > col_mid
    if side == "top":
        return lambda r0, c0, sh, sw: r0 + (sh - 1) / 2.0 < row_mid
    return lambda r0, c0, sh, sw: r0 + (sh - 1) / 2.0 > row_mid


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    axis, sides = _choose_axis(h, rng)

    max_len = (h - 2) if axis == "lr" else (w - 2)
    if max_len < _PALETTE_LEN:
        return None

    max_sh = min(_SPRITE_H_MAX, h - (6 if axis == "tb" else 2))
    max_sw = min(_SPRITE_W_MAX, w - (6 if axis == "lr" else 2))
    if h <= 9:
        max_sh = min(max_sh, 4)
    if w <= 12:
        max_sw = min(max_sw, 4)
    if max_sh < 3 or max_sw < 3:
        return None

    mappings, shared_key = _make_pair_mappings(rng)
    palette_maps = {sides[0]: mappings["a"], sides[1]: mappings["b"]}

    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    for side in sides:
        mapping = palette_maps[side]
        keys = list(mapping.keys())
        fills = [mapping[k] for k in keys]
        pal = _build_palette(side, keys, fills)
        _place_palette(grid, occ, pal, side, rng)

    sprite_specs = []
    for side in sides:
        for color, fill in palette_maps[side].items():
            sprite_specs.append({"side": side, "color": int(color), "fill": int(fill)})
    rng.shuffle(sprite_specs)

    placed = []
    for spec in sprite_specs:
        made = False
        predicate = _zone_predicate(axis, spec["side"], h, w)
        for _ in range(60):
            sh = rng.randint(3, max_sh)
            sw = rng.randint(3, max_sw)
            if sh + sw > _SPRITE_DIM_SUM_CAP:
                sw = max(3, _SPRITE_DIM_SUM_CAP - sh)
            if sw > max_sw:
                continue

            if sh == 3 and sw == 3 and rng.random() < _P_DIAMOND:
                mask = _make_diamond()
            else:
                mask = _make_hollow_sprite(sh, sw, rng)
            if mask is None:
                continue

            holes = _get_enclosed_holes(mask)
            positions = _find_valid_positions(h, w, occ, mask, holes, predicate)
            if not positions:
                continue

            r0, c0 = rng.choice(positions)
            _paint_sprite(grid, r0, c0, mask, spec["color"], spec["fill"], holes)
            placed.append((r0, c0, mask, holes, spec["side"], spec["color"]))
            occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True
            made = True
            break
        if not made:
            return None

    output = grid.copy()
    inp = grid.copy()
    for r0, c0, _mask, holes, _side, _color in placed:
        for hpts in holes:
            for hr, hc in hpts:
                inp[r0 + hr, c0 + hc] = bg

    border = np.concatenate([inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    try:
        palettes = _find_palettes(inp, bg)
        if len(palettes) != 2:
            return None
        if sum(len(p["mapping"]) for p in palettes) != len(placed):
            return None
        shared = set(palettes[0]["mapping"]) & set(palettes[1]["mapping"])
        if shared_key not in shared:
            return None
        if palettes[0]["mapping"][shared_key] == palettes[1]["mapping"][shared_key]:
            return None
        if not np.array_equal(main(inp), output):
            return None
    except Exception:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
