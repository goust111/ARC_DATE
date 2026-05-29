# Variant of abc82100
# Dimension: D4 — teal模板的形状类型 (dst)
# Change: teal模板是由teal像素构成的任意连通图案（菱形、线段等），定义了要复制的形状... → 模板需按比例缩放

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the D4 scaling variant
# a) Concrete behavior:
# - The original task has a legend "source color -> target color -> teal stencil"
#   and then isolated scattered source pixels. Each scattered source pixel copies
#   the teal stencil at one fixed size.
# - The variant keeps the same legend, but each scattered source instance becomes
#   a NW-to-SE same-color diagonal run. The head cell of the run is the placement
#   point, and the run length is the scale factor k in {1,2,3}.
# - The teal legend component is now a prototype. In the output, each prototype
#   cell expands into a k x k block, so one rule can produce the same shape at
#   multiple sizes. Example:
#   original: prototype rel coords {(0,0),(0,1)} stamped at source => two pixels.
#   variant: if the scattered run has length 3, the same prototype becomes two
#   adjacent 3x3 blocks, aligned from the run head.
# - This changes the pipeline exactly at the "apply rule to scattered sources"
#   stage. Rule discovery stays local and color-based; only the interpretation of
#   each non-legend source occurrence changes from "single source pixel" to
#   "head of a same-color diagonal run whose length encodes scale".
# b) Complexity calibration:
# - Scale factors are limited to 1/2/3 so the behavior is visible but not noisy.
# - Every generated rule gets at least one scale-1 instance and at least one
#   larger instance, so the size change is repeated inside the same sample.
# - Prototypes are kept small (2-5 teal cells in at most a 3x3 span), which
#   makes 2x and 3x scaling legible without filling the grid.
# c) Hidden assumptions from the original main():
# - `_discover_rules(...)` assumes a 4-neighbor source/target pair and an anchor
#   whose continuation reaches a teal component. This still holds; the generator
#   places the anchor directly on a teal cell, so distance 0 always works.
# - The original stamping loop treated every remaining source-colored pixel as an
#   independent source instance. That breaks here because diagonal marker cells
#   share the source color. The new solver must only use the NW-most head of a
#   diagonal run; tail cells must be ignored.
# - The original output-preservation loop removed every pixel whose color is a
#   learned source color. That remains valid, and now conveniently erases both
#   the run heads and their same-color scale markers.
# d) Placement legality:
# - Scattered diagonal runs are placed with an 8-neighbor exclusion zone so runs
#   from the same rule cannot merge or accidentally extend each other.
# - Full scaled stamps are kept in bounds and non-overlapping in the generator.
#   This is stricter than the source task but keeps the scale cue readable.
# - Legends are placed after scattered runs and are forbidden from colliding with
#   input objects or with cells reserved for the final stamped output.
# e) Visual observability:
# - A human can compare a short run and a long run of the same source color and
#   see the same target-colored prototype appear small vs. large.
# - Because the encoding is geometric (run length), no hidden use of color ids,
#   ordering, or numeric color arithmetic is involved.
# f) Visual cleanliness:
# - Output stamps do not overlap each other, bystanders stay outside stamp areas,
#   and legends stay away from those areas. This avoids clutter that would hide
#   the scaling effect.
# g) Color inference:
# - Color mapping remains the same explicit local legend as in the source task,
#   so the only new thing to infer is that diagonal-run length controls scale.
# main()/generator contract introduced by this variant:
# - Every transformable source instance must be the NW head of a SE diagonal run
#   of its own color.
# - The generator guarantees that no legend source pixel is diagonally continued
#   by another cell of the same color, so legend examples are not mistaken for
#   scattered instances.


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


def _diagonal_scale_head(grid: np.ndarray, r: int, c: int, color: int) -> int:
    """Return the SE-diagonal run length if (r,c) is the NW-most head, else 0."""
    h, w = grid.shape
    if int(grid[r, c]) != int(color):
        return 0
    if r > 0 and c > 0 and int(grid[r - 1, c - 1]) == int(color):
        return 0

    k = 0
    rr, cc = int(r), int(c)
    while 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(color):
        k += 1
        rr += 1
        cc += 1
    return k


def _scaled_pattern_cells(rel_pattern: np.ndarray, head: tuple[int, int], scale: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    hr, hc = int(head[0]), int(head[1])
    k = int(scale)
    for rr, cc in rel_pattern:
        base_r = hr + int(rr) * k
        base_c = hc + int(cc) * k
        for dr in range(k):
            for dc in range(k):
                cells.append((base_r + dr, base_c + dc))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local color-substitution legends and stamp scaled teal prototypes.

    Input structure:
    - Black background with local source->target legend pairs and nearby teal (8)
      prototype components.
    - Away from the legends, each source instance is a NW-to-SE diagonal run of a
      rule's source color; the run head is the placement point and the run length
      encodes the scale factor.

    Rule:
    - Discover each legend as an adjacent colored pair whose forward anchor
      touches a teal prototype component.
    - For every non-legend diagonal run of the matching source color, place a
      scaled copy of that prototype at the run head, recolored to the rule's
      target color; scale k expands each prototype cell into a k x k block.
    - Remove all teal prototype pixels, legend pixels, and source-color diagonal
      runs from the output.

    Key insight: The legend still teaches the color substitution and prototype
    shape, while the visible diagonal run length supplies the extra scale needed
    to decide how large each stamped copy should be.
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
        pattern = np.asarray(rule["pattern"], dtype=int)
        rel_pattern = pattern - np.array([[int(ar), int(ac)]], dtype=int)
        for sr, sc in map(tuple, np.argwhere(g == rule["source_color"])):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            scale = _diagonal_scale_head(g, sr, sc, rule["source_color"])
            if scale <= 0:
                continue
            for rr, cc in _scaled_pattern_cells(rel_pattern, (sr, sc), scale):
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = rule["target_color"]

    return out


# --------------- generator (D4 scaling variant) ---------------

GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 8, 2: 12, 3: 15, 4: 18}

RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

PROTOTYPE_MIN_CELLS = 2
PROTOTYPE_MAX_CELLS = 5
PROTOTYPE_MAX_SPAN = 3

SCALE_CHOICES = [1, 2, 3]
SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 6
SCATTERED_AREA_DIV = 45

BYSTANDER_MAX = 4

LEGEND_RETRIES = 700
SCATTERED_RETRIES_PER_INSTANCE = 120
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 400

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_8neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                reserved.add((int(r) + dr, int(c) + dc))


def _make_prototype() -> tuple[list[tuple[int, int]], tuple[int, int]]:
    for _ in range(200):
        n = random.randint(PROTOTYPE_MIN_CELLS, PROTOTYPE_MAX_CELLS)
        coords = {(0, 0)}
        for _ in range(n - 1):
            frontier = set()
            for r, c in coords:
                for dr, dc in _DIRS8:
                    p = (r + dr, c + dc)
                    if p not in coords:
                        frontier.add(p)
            coords.add(random.choice(list(frontier)))

        arr = np.asarray(sorted(coords), dtype=int)
        r0 = int(arr[:, 0].min())
        c0 = int(arr[:, 1].min())
        arr[:, 0] -= r0
        arr[:, 1] -= c0
        span_r = int(arr[:, 0].max()) + 1
        span_c = int(arr[:, 1].max()) + 1
        if span_r > PROTOTYPE_MAX_SPAN or span_c > PROTOTYPE_MAX_SPAN:
            continue

        coords_list = [tuple(map(int, rc)) for rc in arr.tolist()]
        anchor = random.choice(coords_list)
        return coords_list, anchor
    raise RuntimeError("failed to build a compact prototype")


def _choose_scales(n_scattered: int) -> list[int]:
    scales = [1, random.choice([2, 3])]
    while len(scales) < n_scattered:
        scales.append(random.choice(SCALE_CHOICES))
    random.shuffle(scales)
    return scales


def _build_rules(num_rules: int, area: int):
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules: list[dict] = []
    used_colors: set[int] = set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used_colors.update((src_c, tgt_c))
        prototype, anchor = _make_prototype()
        rel = sorted((int(r - anchor[0]), int(c - anchor[1])) for r, c in prototype)
        n_sc = random.randint(SCATTERED_MIN, sc_max)
        rules.append(
            {
                "src": int(src_c),
                "tgt": int(tgt_c),
                "rel": rel,
                "scales": _choose_scales(n_sc),
            }
        )
    return rules, used_colors


def _scaled_stamp_cells(rel, head: tuple[int, int], scale: int) -> set[tuple[int, int]]:
    return set(_scaled_pattern_cells(np.asarray(rel, dtype=int), head, scale))


def _place_scattered(rules, inp, occupied, reserved, output_occupied):
    H, W = inp.shape
    for rule in rules:
        placements: list[tuple[int, int, int]] = []
        for scale in sorted(rule["scales"], reverse=True):
            placed = False
            for _ in range(SCATTERED_RETRIES_PER_INSTANCE):
                sr = random.randint(0, H - 1)
                sc = random.randint(0, W - 1)
                chain = [(sr + i, sc + i) for i in range(scale)]
                if any(not (0 <= r < H and 0 <= c < W) for r, c in chain):
                    continue
                if any((r, c) in occupied or (r, c) in reserved for r, c in chain):
                    continue

                stamp_cells = _scaled_stamp_cells(rule["rel"], (sr, sc), scale)
                if any(not (0 <= r < H and 0 <= c < W) for r, c in stamp_cells):
                    continue
                if stamp_cells & output_occupied:
                    continue

                for r, c in chain:
                    inp[r, c] = rule["src"]
                occupied.update(chain)
                _reserve_8neighbors(reserved, chain)
                output_occupied.update(stamp_cells)
                placements.append((int(sr), int(sc), int(scale)))
                placed = True
                break
            if not placed:
                return False

        rule["scattered"] = placements
    return True


def _place_bystanders(used_colors, inp, occupied, reserved, output_occupied):
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
            if (r, c) in occupied or (r, c) in reserved or (r, c) in output_occupied:
                continue
            color = random.choice(bystander_colors)
            inp[r, c] = color
            placed.append((int(r), int(c), int(color)))
            occupied.add((r, c))
            output_occupied.add((r, c))
            _reserve_8neighbors(reserved, [(r, c)])
            break
    return placed


def _place_legends(rules, inp, occupied, reserved, output_occupied):
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

            teal_cells = {(ar + rr, ac + cc) for rr, cc in rel}
            if any(not (0 <= r < H and 0 <= c < W) for r, c in teal_cells):
                continue

            all_cells = {(sr, sc), (tr, tc)} | teal_cells
            if all_cells & occupied:
                continue
            if all_cells & reserved:
                continue
            if all_cells & output_occupied:
                continue

            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for r, c in teal_cells:
                inp[r, c] = 8
            occupied.update(all_cells)
            _reserve_8neighbors(reserved, all_cells)
            placed = True
            break
        if not placed:
            return False
    return True


def _build_output(rules, bystander_pos, H, W):
    output = np.zeros((H, W), dtype=int)
    for r, c, color in bystander_pos:
        output[r, c] = color
    for rule in rules:
        for sr, sc, scale in rule["scattered"]:
            for r, c in _scaled_stamp_cells(rule["rel"], (sr, sc), scale):
                if 0 <= r < H and 0 <= c < W:
                    output[r, c] = rule["tgt"]
    return output


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    H = random.randint(min_dim, GRID_DIM_MAX)
    W = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, H * W)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    output_occupied: set[tuple[int, int]] = set()
    inp = np.zeros((H, W), dtype=int)

    if not _place_scattered(rules, inp, occupied, reserved, output_occupied):
        return None
    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved, output_occupied)
    if not _place_legends(rules, inp, occupied, reserved, output_occupied):
        return None

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
