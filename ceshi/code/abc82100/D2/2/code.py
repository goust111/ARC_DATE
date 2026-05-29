# Variant of abc82100
# Dimension: D2 — 规则示例的编码格式 (symbol_instructions)
# Change: 规则示例由相邻的源色-目标色像素对加上附近的teal模板图案构成，颜色对直接编码映射关系... → 模板颜色非teal而是与目标色相同

from __future__ import annotations

import random
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import components_mask, get_bg


# Design analysis for the D2 variant
# analysis:
#   concrete_behavior:
#     original_rule_encoding:
#       - each legend used a source pixel, an adjacent target pixel, and a nearby stencil drawn in fixed teal
#       - solver could detect stencils globally by extracting every teal component once
#     variant_rule_encoding:
#       - the stencil is no longer teal; every stencil cell is drawn directly in the target color
#       - visually the legend becomes "source pixel touching the front edge of a same-colored blob"
#       - the target pixel is still the single cell adjacent to the source in the legend, but it now shares its color with the blob behind it
#     traced_difference:
#       - original flow: scan source->target pair, then look up a nearby teal component
#       - new flow: scan source->target pair, temporarily remove that target cell from the target-color mask, then find the nearby remaining target-color component
#       - this local removal is the key change that separates "the one target cell in the legend" from "the rest of the same-colored stencil"
#   complexity_calibration:
#     - colors remain globally unique across all source and target roles, so same-colored stencil pixels only belong to the legend that introduced that target color
#     - stencils keep their 1-12 pixel irregular 8-connected shapes, so the visual effect of recoloring remains obvious
#     - target color 8 is still excluded from the palette, preventing accidental regression to the original teal-coded look
#   implicit_assumptions_audit:
#     - original _discover_rules assumed every stencil pixel had color 8; this breaks completely and must be replaced
#     - original main preserved every non-bg non-teal pixel unless it was a legend pixel or a scattered source; this is no longer safe because stencil pixels are ordinary target-colored pixels and must now be explicitly removed
#     - original generator could place the stencil anywhere near the anchor because color separation made false matches rare; now we must also prevent stencil cells from becoming additional 4-neighbors of the source pixel
#   placement_legality:
#     - legend source and legend target remain 4-neighbors in a cardinal direction
#     - the stencil anchor is still the continued cell beyond the target, and the generated stencil always includes that anchor
#     - generator rejects legend placements where the same-colored stencil would overlap the source or target cell, or where any stencil cell besides the target becomes 4-neighbor to the source
#     - the reserved-neighbor policy is kept so scattered sources and bystanders do not create accidental adjacent source-target pairs near a legend stencil
#   visual_observability:
#     - humans can still read each legend as a local example because the source pixel is the only differently colored cell attached to a target-colored blob
#     - removing all teal from the inputs makes the variant visible immediately across examples
#   cleanliness:
#     - scattered source pixels remain isolated, legends stay separated by 4-neighbor reservations, and bystanders avoid all used source/target colors
#   color_inference:
#     - the color mapping is still given directly by the adjacent source-target pair; the variant only changes how the stencil is encoded, not the mapping semantics


_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _pattern_signature(pattern: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple((int(r), int(c)) for r, c in np.asarray(pattern, dtype=int))


def _discover_rules(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], tuple[tuple[int, int], ...]]] = set()

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color == int(bg):
                continue

            for dr, dc in _DIRS4:
                tr = r + dr
                tc = c + dc
                ar = tr + dr
                ac = tc + dc
                if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
                    continue

                target_color = int(grid[tr, tc])
                if target_color in (int(bg), source_color):
                    continue

                target_mask = grid == target_color
                target_mask[tr, tc] = False
                candidates: list[tuple[int, int, np.ndarray]] = []
                for pattern in components_mask(target_mask, connectivity=8):
                    dist = _anchor_distance(pattern, (ar, ac))
                    if dist <= 1:
                        candidates.append((int(dist), len(pattern), pattern))

                if not candidates:
                    continue

                candidates.sort(key=lambda item: (item[0], item[1]))
                _dist, _size, pattern = candidates[0]
                signature = _pattern_signature(pattern)
                key = ((int(r), int(c)), (int(tr), int(tc)), signature)
                if key in seen:
                    continue
                seen.add(key)
                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": (int(ar), int(ac)),
                        "pattern": np.asarray(pattern, dtype=int),
                    }
                )

    return rules


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use local source-target-pattern examples to replace matching source pixels.

    Input structure:
    - Black background with isolated colored pixels and legend stencils whose cells
      already use the same color as the legend's target pixel.
    - A rule example is a 4-neighbor colored pair source->target whose continued
      direction reaches an anchor inside or touching another component of target.color.

    Rule:
    - For each rule example, copy that nearby target-colored component onto every
      other pixel of source.color, keeping the copied component in target.color.
    - Translation maps the rule anchor onto the source pixel being replaced.
    - Remove the legend pair and remove the legend stencil component itself.

    Key insight: the source-target adjacency still encodes the recoloring rule, but
    the stencil is hidden among pixels of the same color as the target cell, so the
    target cell must be separated conceptually from the rest of its same-colored blob.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    rules = _discover_rules(g, bg=bg)
    legend_pixels: set[tuple[int, int]] = set()
    pattern_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])
        pattern_pixels.update((int(r), int(c)) for r, c in rule["pattern"])
        source_colors.add(rule["source_color"])

    consumed_pixels = legend_pixels | pattern_pixels
    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere(g != int(bg))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in consumed_pixels or color in source_colors:
            continue
        out[r, c] = color

    for rule in rules:
        ar, ac = rule["anchor"]
        pattern = rule["pattern"]
        for sr, sc in map(tuple, np.argwhere(g == rule["source_color"])):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            dr = sr - ar
            dc = sc - ac
            rows = pattern[:, 0] + dr
            cols = pattern[:, 1] + dc
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = rule["target_color"]

    return out


# --------------- generator (c) ---------------

GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 5, 2: 10, 3: 14, 4: 18}

RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 25

SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 8
SCATTERED_AREA_DIV = 30

BYSTANDER_MAX = 5

LEGEND_RETRIES = 500
SCATTERED_RETRIES_PER_PIXEL = 50
BYSTANDER_RETRIES = 50
OUTER_RETRIES = 300


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
    anchor = random.choice(coords_list)
    return coords_list, anchor


def _build_rules(num_rules: int, area: int):
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(1, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules = []
    used = set()
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
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
    return rules, used


def _legend_is_valid(sr: int, sc: int, tr: int, tc: int, pattern_cells: set[tuple[int, int]]) -> bool:
    if (sr, sc) in pattern_cells or (tr, tc) in pattern_cells:
        return False
    for nr, nc in pattern_cells:
        if abs(nr - sr) + abs(nc - sc) == 1:
            return False
    return True


def _place_legends(rules, inp, occupied, reserved):
    H, W = inp.shape
    for rule in rules:
        rel = rule["rel"]
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr = random.randint(0, H - 1)
            sc = random.randint(0, W - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            if not (0 <= tr < H and 0 <= tc < W and 0 <= ar < H and 0 <= ac < W):
                continue

            pattern_cells: set[tuple[int, int]] = set()
            ok = True
            for rr, cc in rel:
                nr, nc = ar + rr, ac + cc
                if not (0 <= nr < H and 0 <= nc < W):
                    ok = False
                    break
                pattern_cells.add((nr, nc))
            if not ok or not _legend_is_valid(sr, sc, tr, tc, pattern_cells):
                continue

            occupied_now = {(sr, sc), (tr, tc)} | pattern_cells
            reserved_now = occupied_now | {(ar, ac)}
            if occupied_now & occupied:
                continue
            if reserved_now & reserved:
                continue

            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for nr, nc in pattern_cells:
                inp[nr, nc] = rule["tgt"]
            occupied.update(occupied_now)
            _reserve_4neighbors(reserved, reserved_now)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_scattered(rules, inp, occupied, reserved):
    H, W = inp.shape
    for rule in rules:
        positions = []
        for _ in range(rule["n_sc"] * SCATTERED_RETRIES_PER_PIXEL):
            if len(positions) >= rule["n_sc"]:
                break
            r = random.randint(0, H - 1)
            c = random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            positions.append((r, c))
            occupied.add((r, c))
            _reserve_4neighbors(reserved, [(r, c)])
        if len(positions) < SCATTERED_MIN:
            return False
        rule["scattered"] = positions
        for r, c in positions:
            inp[r, c] = rule["src"]
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
            r = random.randint(0, H - 1)
            c = random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            color = random.choice(bystander_colors)
            inp[r, c] = color
            placed.append((r, c, color))
            occupied.add((r, c))
            _reserve_4neighbors(reserved, [(r, c)])
            break
    return placed


def _build_output(rules, bystander_pos, H: int, W: int):
    output = np.zeros((H, W), dtype=int)
    for r, c, color in bystander_pos:
        output[r, c] = color
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for rr, cc in rule["rel"]:
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
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    inp = np.zeros((H, W), dtype=int)

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    if not _place_scattered(rules, inp, occupied, reserved):
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
