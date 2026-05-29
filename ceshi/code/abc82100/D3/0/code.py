# Variant of abc82100
# Dimension: D3 — 源对象类型 (src)
# Change: 散落的单个彩色像素（不属于任何规则示例的非背景像素）... → 多像素连通色块作为源

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the D3 variant
# ----------------------------------
# a) Concrete behavior:
#    - The legend mechanism stays the same as the original task: a colored
#      source pixel, a 4-neighbor target pixel, and a nearby teal stencil define
#      a rule.
#    - The changed part is the set of free-standing source objects. They are no
#      longer isolated pixels. Each one is a 4-connected monochrome polyomino.
#    - Each source component triggers exactly one stencil copy. The stencil is
#      aligned to the component's top-left occupied pixel (minimum row, then
#      minimum column among ties). This is the visible "component anchor".
#    - Example trace:
#      original: three red single pixels => stamp the teal stencil three times,
#      once on each red pixel.
#      variant: three red blobs, e.g. a domino, an L-triomino, a square =>
#      identify three red components, find each blob's top-left pixel, and stamp
#      once per blob on that anchor. The rest of the blob body does not create
#      extra copies.
# b) Complexity calibration:
#    - Every scattered source object has size >= 2, so samples cannot collapse
#      back to the single-pixel source behavior.
#    - A small shape vocabulary with random rotations keeps the grids readable
#      while still forcing component recognition.
#    - We keep unique source colors per rule, like the original task, so color
#      matching is unambiguous and repeated across multiple source blobs.
# c) Hidden assumptions in the original solver that no longer hold:
#    - `for sr, sc in np.argwhere(g == rule["source_color"])`: this assumed
#      every source-colored pixel is an independent source object. In the
#      variant, that would over-stamp once per cell and is therefore wrong.
#    - `if (sr, sc) in legend_pixels: continue`: this only removes the legend's
#      source example pixel, not whole source components. The new solver must
#      extract connected components after excluding legend pixels.
#    - Preserving non-rule colors still works, but only because the generator
#      continues to avoid bystanders that use any source color.
# d) Placement legality:
#    - Legend regions, source blobs, and bystanders are separated by reserved
#      4-neighborhoods so no accidental colored adjacencies can invent false
#      rules.
#    - Source blobs of the same color are also separated, ensuring distinct
#      connected components for the solver.
# e) Visual observability:
#    - The change is visible because inputs now contain colored blobs instead of
#      lone pixels, while outputs still show exactly one recolored stencil per
#      blob.
#    - The chosen anchor is geometric and visible: the stencil aligns to the
#      blob's top-left occupied cell.
# f) Visual cleanliness:
#    - Blob sizes stay small and padding is enforced around placed objects.
#    - Output stamps may clip at the border, as in the original task, but the
#      generator avoids dense clutter by using fewer, larger source objects.
# g) Color inference:
#    - Colors still serve only as rule identities learned from the legend.
#      Geometry decides where to apply the stencil; no invisible color ordering
#      or numeric color arithmetic is used.
# Review of new solver assumptions:
#    - Each rule color forms one or more 4-connected non-legend components.
#    - The component anchor is recoverable as the top-left occupied cell.
#    - The generator guarantees no source blob touches any other non-background
#      colored object by 4-neighborhood, so rule discovery remains reliable.


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _discover_rules(grid: np.ndarray, bg: int, pattern_color: int = 8) -> list[dict]:
    h, w = grid.shape
    patterns = components_mask(grid == pattern_color, connectivity=8)
    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color in (int(bg), int(pattern_color)):
                continue

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                tr = r + dr
                tc = c + dc
                if not (0 <= tr < h and 0 <= tc < w):
                    continue

                target_color = int(grid[tr, tc])
                if target_color in (int(bg), int(pattern_color)) or target_color == source_color:
                    continue

                anchor = (tr + dr, tc + dc)
                candidates: list[tuple[int, int, np.ndarray]] = []
                for i, pattern in enumerate(patterns):
                    dist = _anchor_distance(pattern, anchor)
                    if dist <= 1:
                        candidates.append((int(dist), int(i), pattern))

                if not candidates:
                    continue

                candidates.sort(key=lambda item: (item[0], len(item[2])))
                _dist, pattern_index, pattern = candidates[0]
                key = ((int(r), int(c)), (int(tr), int(tc)), int(pattern_index))
                if key in seen:
                    continue
                seen.add(key)
                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "pattern": np.asarray(pattern, dtype=int),
                    }
                )

    return rules


def _component_anchor(component: np.ndarray) -> tuple[int, int]:
    cells = sorted((int(r), int(c)) for r, c in component)
    return cells[0]


def _source_components(
    grid: np.ndarray, source_color: int, legend_pixels: set[tuple[int, int]]
) -> list[np.ndarray]:
    mask = grid == int(source_color)
    if legend_pixels:
        mask = mask.copy()
        for r, c in legend_pixels:
            if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]:
                mask[r, c] = False
    return components_mask(mask, connectivity=4)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local color-to-stencil rules, then apply them to colored source blobs.

    Input structure:
    - Black background with teal (8) stencil components, legend source->target
      color pairs, and additional same-color 4-connected source blobs elsewhere.
    - A rule example is still a 4-neighbor colored pair whose forward direction
      reaches an anchor that touches the corresponding teal stencil.

    Rule:
    - For each learned rule, find every non-legend 4-connected component of the
      rule's source color and copy the teal stencil once for that whole blob.
    - The stencil is translated so the legend anchor lands on the blob's
      top-left occupied pixel, then recolored to the rule's target color.
    - Remove the legend pixels, all teal stencil pixels, and the source blobs.

    Key insight: The local legend still teaches the color substitution and the
    stencil shape, but the free-standing source objects must now be grouped into
    connected blobs before choosing a single geometric anchor for each stamp.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=pattern_color)
    legend_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])
        source_colors.add(rule["source_color"])

    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or color in source_colors:
            continue
        out[r, c] = color

    for rule in rules:
        ar, ac = rule["anchor"]
        pattern = rule["pattern"]
        for component in _source_components(g, rule["source_color"], legend_pixels):
            sr, sc = _component_anchor(component)
            dr = sr - ar
            dc = sc - ac
            rows = pattern[:, 0] + dr
            cols = pattern[:, 1] + dc
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = rule["target_color"]

    return out


# --------------- generator (c) ---------------

# Grid dimensions
GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 6, 2: 10, 3: 14, 4: 18}

# Rule sampling
RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

# Stencil size
STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 25

# Source blob counts
SOURCE_BLOCK_MIN = 3
SOURCE_BLOCK_MAX_CAP = 6
SOURCE_BLOCK_AREA_DIV = 45

# Bystander pixels
BYSTANDER_MAX = 5

# Placement retry budgets
LEGEND_RETRIES = 500
SOURCE_BLOCK_RETRIES_PER_BLOCK = 80
BYSTANDER_RETRIES = 50
OUTER_RETRIES = 400

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

_BASE_SOURCE_SHAPES = [
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (0, 1), (1, 0)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 1)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 1)],
]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _make_stencil(min_pixels: int, max_pixels: int):
    n = random.randint(min_pixels, max_pixels)
    coords = {(0, 0)}
    for _ in range(n - 1):
        frontier = set()
        for r, c in coords:
            for dr, dc in _DIRS8:
                p = (r + dr, c + dc)
                if p not in coords:
                    frontier.add(p)
        coords.add(random.choice(list(frontier)))
    coords_list = sorted(coords)
    return coords_list, random.choice(coords_list)


def _normalize_shape(cells) -> list[tuple[int, int]]:
    cells = [(int(r), int(c)) for r, c in cells]
    anchor = min(cells)
    return sorted((r - anchor[0], c - anchor[1]) for r, c in cells)


def _rotate_reflect_shape(cells) -> list[tuple[int, int]]:
    transformed = []
    flip = random.choice([False, True])
    turns = random.randint(0, 3)
    for r, c in cells:
        rr, cc = int(r), int(c)
        if flip:
            cc = -cc
        for _ in range(turns):
            rr, cc = cc, -rr
        transformed.append((rr, cc))
    return _normalize_shape(transformed)


def _make_source_shape() -> list[tuple[int, int]]:
    base = random.choice(_BASE_SOURCE_SHAPES)
    return _rotate_reflect_shape(base)


def _build_rules(num_rules: int, area: int):
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(1, area // STENCIL_MIN_AREA_DIV), max_s)
    block_max = min(SOURCE_BLOCK_MAX_CAP, max(SOURCE_BLOCK_MIN, area // SOURCE_BLOCK_AREA_DIV))

    rules, used = [], set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used.update((src_c, tgt_c))
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(r - anchor[0], c - anchor[1]) for r, c in stencil]
        rules.append(
            {
                "src": src_c,
                "tgt": tgt_c,
                "rel": rel,
                "n_blocks": random.randint(SOURCE_BLOCK_MIN, block_max),
            }
        )
    return rules, used


def _place_legends(rules, inp, occupied, reserved):
    H, W = inp.shape
    for rule in rules:
        rel = rule["rel"]
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr, sc = random.randint(0, H - 1), random.randint(0, W - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            if not (0 <= tr < H and 0 <= tc < W and 0 <= ar < H and 0 <= ac < W):
                continue
            teal_cells = set()
            ok = True
            for rr, cc in rel:
                nr, nc = ar + rr, ac + cc
                if not (0 <= nr < H and 0 <= nc < W):
                    ok = False
                    break
                teal_cells.add((nr, nc))
            if not ok:
                continue
            all_cells = {(sr, sc), (tr, tc), (ar, ac)} | teal_cells
            if all_cells & (occupied | reserved):
                continue
            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for nr, nc in teal_cells:
                inp[nr, nc] = 8
            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            rule["legend"] = {"source": (sr, sc), "target": (tr, tc), "anchor": (ar, ac)}
            placed = True
            break
        if not placed:
            return False
    return True


def _place_source_blocks(rules, inp, occupied, reserved):
    H, W = inp.shape
    for rule in rules:
        blocks = []
        target_count = rule["n_blocks"]
        for _ in range(target_count * SOURCE_BLOCK_RETRIES_PER_BLOCK):
            if len(blocks) >= target_count:
                break
            rel = _make_source_shape()
            ar, ac = random.randint(0, H - 1), random.randint(0, W - 1)
            cells = {(ar + rr, ac + cc) for rr, cc in rel}
            if any(not (0 <= r < H and 0 <= c < W) for r, c in cells):
                continue
            if cells & (occupied | reserved):
                continue
            for r, c in cells:
                inp[r, c] = rule["src"]
            occupied.update(cells)
            _reserve_4neighbors(reserved, cells)
            blocks.append({"anchor": (ar, ac), "cells": sorted(cells), "rel": rel})
        if len(blocks) < SOURCE_BLOCK_MIN:
            return False
        rule["blocks"] = blocks
    return True


def _place_bystanders(used_colors, inp, occupied, reserved):
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []
    H, W = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) not in occupied and (r, c) not in reserved:
                bc = random.choice(bystander_colors)
                inp[r, c] = bc
                placed.append((r, c, bc))
                occupied.add((r, c))
                _reserve_4neighbors(reserved, [(r, c)])
                break
    return placed


def _build_output(rules, bystander_pos, H, W):
    output = np.zeros((H, W), dtype=int)
    for r, c, bc in bystander_pos:
        output[r, c] = bc
    for rule in rules:
        rel = rule["rel"]
        for block in rule["blocks"]:
            sr, sc = block["anchor"]
            for rr, cc in rel:
                nr, nc = sr + rr, sc + cc
                if 0 <= nr < H and 0 <= nc < W:
                    output[nr, nc] = rule["tgt"]
    return output


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    H = random.randint(min_dim, GRID_DIM_MAX)
    W = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, H * W)
    occupied, reserved = set(), set()
    inp = np.zeros((H, W), dtype=int)

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    if not _place_source_blocks(rules, inp, occupied, reserved):
        return None
    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved)

    output = _build_output(rules, bystander_pos, H, W)
    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
