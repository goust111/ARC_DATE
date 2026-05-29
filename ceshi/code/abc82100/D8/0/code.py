# Variant of abc82100
# Dimension: D8 — 输出网格形状 (grid_shape)
# Change: 输出网格尺寸与输入相同（same）... → 紧凑裁剪到非背景像素范围

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import bbox_coords, components_mask, get_bg
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "solvers"))
    from arc_utils import bbox_coords, components_mask, get_bg


# Design analysis for the D8 variant:
# - Concrete behavior:
#   The local rule-learning part is unchanged: a source pixel next to a target
#   pixel, with a nearby teal stencil, teaches "source color -> stencil painted
#   in target color". The only rule change is that the final canvas is no longer
#   the input-sized board; it is the smallest rectangle containing every
#   non-background output pixel.
# - Small execution trace:
#   Suppose the uncropped result on a 16x16 board only paints rows 4..10 and
#   cols 3..9. The original task would return a 16x16 grid with a thick black
#   border. This variant must instead return the 7x7 subgrid [4:11, 3:10].
# - Complexity calibration:
#   The rule structure, colors, stencil reuse, and scattered source counts stay
#   the same, so the puzzle is still "learn local examples, then stamp". The
#   generator only adds interior padding so the crop is consistently observable.
# - Hidden assumptions in the original main():
#   `out = np.full_like(g, bg)` is still valid as an intermediate canvas, but
#   `return out` is no longer valid because same-size output was the old D8
#   assumption. The preservation of bystanders remains safe because tight crop
#   is based on the final non-background bbox, not on input positions.
# - Placement legality:
#   The generator computes safe source ranges from each stencil's relative
#   offsets, then places scattered source pixels and bystanders strictly inside
#   an interior window. This guarantees that every stamped pixel stays in-bounds
#   and away from the outer border, so the crop is smaller than the input.
# - Visual observability:
#   Every accepted sample has output pixels bounded away from all four sides of
#   the input grid. A human sees the same legend-learning rule as before, plus
#   the consistent disappearance of the empty border in the output.
# - Neatness / color inference:
#   Source-target color examples remain the only cue for recoloring. Bystanders
#   still use colors outside all rule colors, so they remain visually separate
#   and preserved after cropping.


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _tight_crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return np.array(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int)


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


def _solve_uncropped(grid: np.ndarray) -> np.ndarray:
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

    out = np.full((h, w), int(bg), dtype=int)

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local color-stencil examples, paint all matching source pixels, then crop.

    Input structure:
    - A mostly black grid contains isolated colored pixels plus several teal (8)
      stencil components.
    - Each rule example is a 4-neighbor colored pair source->target whose
      forward continuation points to an anchor touching that rule's teal stencil.

    Rule:
    - Reuse every learned teal stencil on all other pixels of the same source
      color, translating the rule anchor onto the scattered source pixel and
      recoloring the stencil to the target color.
    - Erase the teal stencil components and the colored example pairs.
    - After all painting, remove the empty border by taking the minimal bounding
      box of non-background output pixels.

    Key insight: The example pair still teaches a color substitution plus a
    stencil anchor; the new twist is that only the painted result matters, so
    the final answer is the tight crop around that painted foreground.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = _solve_uncropped(g)
    return _tight_crop_non_bg(out, bg=int(bg))


# --------------- generator (c) ---------------

# Grid dimensions
GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 5, 2: 10, 3: 14, 4: 18}

# Rule sampling
RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

# Stencil size
STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 25

# Scattered source pixels per rule
SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 8
SCATTERED_AREA_DIV = 30

# Bystander pixels
BYSTANDER_MAX = 5

# Crop visibility
CROP_PAD_MIN = 1
CROP_PAD_MAX = 4
PAD_RETRIES = 80

# Placement retry budgets
LEGEND_RETRIES = 500
OUTER_RETRIES = 300

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r + dr), int(c + dc)))


def _make_stencil(min_pixels: int, max_pixels: int):
    n = random.randint(min_pixels, max_pixels)
    coords = {(0, 0)}
    for _ in range(n - 1):
        frontier = set()
        for r, c in coords:
            for dr, dc in _DIRS8:
                p = (int(r + dr), int(c + dc))
                if p not in coords:
                    frontier.add(p)
        coords.add(random.choice(list(frontier)))
    coords_list = sorted(coords)
    return coords_list, random.choice(coords_list)


def _rel_bounds(rel: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [int(r) for r, _ in rel]
    cs = [int(c) for _, c in rel]
    return min(rs), max(rs), min(cs), max(cs)


def _build_rules(num_rules: int, area: int):
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(1, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules, used = [], set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used.update((src_c, tgt_c))
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(int(r - anchor[0]), int(c - anchor[1])) for r, c in stencil]
        rules.append(
            {
                "src": int(src_c),
                "tgt": int(tgt_c),
                "rel": rel,
                "bounds": _rel_bounds(rel),
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
    return rules, used


def _choose_content_padding(rules, H: int, W: int):
    row_cap = max(CROP_PAD_MIN, min(CROP_PAD_MAX, max(1, (H - 2) // 3)))
    col_cap = max(CROP_PAD_MIN, min(CROP_PAD_MAX, max(1, (W - 2) // 3)))
    for _ in range(PAD_RETRIES):
        top = random.randint(CROP_PAD_MIN, row_cap)
        bottom = random.randint(CROP_PAD_MIN, row_cap)
        left = random.randint(CROP_PAD_MIN, col_cap)
        right = random.randint(CROP_PAD_MIN, col_cap)
        if top + bottom >= H or left + right >= W:
            continue
        ok = True
        for rule in rules:
            min_r, max_r, min_c, max_c = rule["bounds"]
            if top - min_r > (H - 1 - bottom - max_r):
                ok = False
                break
            if left - min_c > (W - 1 - right - max_c):
                ok = False
                break
        if ok:
            return {"top": top, "bottom": bottom, "left": left, "right": right}
    return None


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
                nr, nc = int(ar + rr), int(ac + cc)
                if not (0 <= nr < H and 0 <= nc < W):
                    ok = False
                    break
                teal_cells.add((nr, nc))
            if not ok:
                continue

            all_cells = {(int(sr), int(sc)), (int(tr), int(tc)), (int(ar), int(ac))} | teal_cells
            if all_cells & (occupied | reserved):
                continue

            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for nr, nc in teal_cells:
                inp[nr, nc] = 8
            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_scattered(rules, inp, occupied, reserved, pads):
    H, W = inp.shape
    ordered = sorted(
        rules,
        key=lambda rule: (
            (H - pads["bottom"] - rule["bounds"][1]) - (pads["top"] - rule["bounds"][0]) + 1
        )
        * (
            (W - pads["right"] - rule["bounds"][3]) - (pads["left"] - rule["bounds"][2]) + 1
        ),
    )
    for rule in ordered:
        min_r, max_r, min_c, max_c = rule["bounds"]
        r0 = int(pads["top"] - min_r)
        r1 = int(H - 1 - pads["bottom"] - max_r)
        c0 = int(pads["left"] - min_c)
        c1 = int(W - 1 - pads["right"] - max_c)
        if r0 > r1 or c0 > c1:
            return False

        candidates = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        random.shuffle(candidates)
        positions: list[tuple[int, int]] = []
        for r, c in candidates:
            if len(positions) >= rule["n_sc"]:
                break
            if (r, c) in occupied or (r, c) in reserved:
                continue
            positions.append((int(r), int(c)))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(int(r), int(c))])
        if len(positions) < SCATTERED_MIN:
            return False
        rule["scattered"] = positions
        for r, c in positions:
            inp[r, c] = rule["src"]
    return True


def _place_bystanders(used_colors, inp, occupied, reserved, pads):
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []

    H, W = inp.shape
    cells = [
        (r, c)
        for r in range(int(pads["top"]), int(H - pads["bottom"]))
        for c in range(int(pads["left"]), int(W - pads["right"]))
    ]
    random.shuffle(cells)
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed = []
    for r, c in cells:
        if len(placed) >= n:
            break
        if (r, c) in occupied or (r, c) in reserved:
            continue
        bc = int(random.choice(bystander_colors))
        inp[r, c] = bc
        placed.append((int(r), int(c), bc))
        occupied.add((int(r), int(c)))
        _reserve_4neighbors(reserved, [(int(r), int(c))])
    return placed


def _build_output(rules, bystander_pos, H, W):
    output = np.zeros((H, W), dtype=int)
    for r, c, bc in bystander_pos:
        output[int(r), int(c)] = int(bc)
    for rule in rules:
        rel = rule["rel"]
        for sr, sc in rule["scattered"]:
            for rr, cc in rel:
                nr, nc = int(sr + rr), int(sc + cc)
                if 0 <= nr < H and 0 <= nc < W:
                    output[nr, nc] = int(rule["tgt"])
    return output


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    H = random.randint(min_dim, GRID_DIM_MAX)
    W = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, H * W)
    pads = _choose_content_padding(rules, H, W)
    if pads is None:
        return None

    occupied, reserved = set(), set()
    inp = np.zeros((H, W), dtype=int)

    if not _place_scattered(rules, inp, occupied, reserved, pads):
        return None
    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved, pads)
    if not _place_legends(rules, inp, occupied, reserved):
        return None

    full_output = _build_output(rules, bystander_pos, H, W)
    cropped_output = _tight_crop_non_bg(full_output, 0)
    if cropped_output.shape[0] >= H or cropped_output.shape[1] >= W:
        return None

    try:
        if np.array_equal(main(inp), cropped_output):
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
