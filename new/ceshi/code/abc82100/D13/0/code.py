# Variant of abc82100
# Dimension: D13 — 源色-目标色映射的对称性 (meta)
# Change: 映射可以是任意的（如A→B，C→D），不要求对称或逆映射... → 强制双向映射（A→B则B→A）

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the D13 variant
# a) Concrete behavior:
#    Each logical rule is now a swap pair {A, B}, not a one-way mapping. The input
#    still contains teal stencil components plus local color-pair legends, but every
#    stencil-backed legend A->B must be accompanied by the reciprocal legend B->A.
#    To make that visible, the generator places both legends around the same teal
#    component on opposite sides of one anchor cell, yielding a symmetric layout like
#    A B [teal] A B along one axis. Scattered A pixels stamp the stencil in color B,
#    and scattered B pixels stamp the same stencil in color A.
#    Small trace: original logic accepts a single legend A->B and changes every loose
#    A into a B-colored stencil. The variant instead requires two legends around the
#    same stencil, so loose A and loose B are both active and swap into each other.
# b) Complexity calibration:
#    The change is not reduced to a trivial palette permutation. Each swap pair still
#    needs a local example, a nontrivial stencil, and repeated scattered evidence for
#    both colors. Colors are limited to disjoint pairs from the original palette so
#    humans can infer the swap without a large, noisy mapping table.
# c) Old assumptions that no longer suffice:
#    The original solver can apply arbitrary directed rules independently. Under D13,
#    that is too weak because a single discovered A->B legend should not count unless
#    the reciprocal B->A exists for the same stencil/anchor. The new solver therefore
#    filters discovered rules down to reciprocal pairs sharing the same stencil and
#    anchor before it removes source-colored scattered pixels and stamps outputs.
# d) Placement legality:
#    The generator keeps all legends, stencil cells, and scattered source pixels
#    inside bounds and non-overlapping in the input. Reciprocal legends share one
#    stencil anchor but use opposite sides of the same axis, so they do not collide.
#    Four-neighbor reservation prevents accidental extra legends from forming.
# e) Visual observability:
#    The reciprocal structure is visible in 2-4 examples because each active color
#    appears both as a source and as a target in the legend, and both colors have
#    multiple scattered occurrences. The swap is inferable from geometry, not from
#    numeric color ordering.
# f) Visual cleanliness:
#    Output stamps are prevented from overlapping each other, and bystanders are
#    placed outside all future stamp footprints. This keeps the doubled swap activity
#    readable after introducing reciprocal mappings.
# g) Color-change inferability:
#    Each swap pair uses exactly two colors with one shared stencil, repeated across
#    multiple scattered pixels of both colors. That repeated, mirrored evidence makes
#    the A<->B relationship directly observable.


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _pattern_key(pattern: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple((int(r), int(c)) for r, c in np.asarray(pattern, dtype=int))


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


def _discover_swap_groups(rules: list[dict]) -> list[dict]:
    grouped: dict[tuple[frozenset[int], tuple[int, int], tuple[tuple[int, int], ...]], dict] = {}
    key_order: list[tuple[frozenset[int], tuple[int, int], tuple[tuple[int, int], ...]]] = []

    for rule in rules:
        color_pair = frozenset((int(rule["source_color"]), int(rule["target_color"])))
        if len(color_pair) != 2:
            continue
        key = (color_pair, tuple(rule["anchor"]), _pattern_key(rule["pattern"]))
        if key not in grouped:
            grouped[key] = {"dirs": {}, "rules": []}
            key_order.append(key)
        grouped[key]["dirs"][(int(rule["source_color"]), int(rule["target_color"]))] = rule
        grouped[key]["rules"].append(rule)

    swap_groups: list[dict] = []
    for key in key_order:
        dirs = grouped[key]["dirs"]
        first_rule = grouped[key]["rules"][0]
        a = int(first_rule["source_color"])
        b = int(first_rule["target_color"])
        if (b, a) not in dirs:
            continue
        swap_groups.append(
            {
                "colors": (a, b),
                "rules_by_source": {
                    a: dirs[(a, b)],
                    b: dirs[(b, a)],
                },
            }
        )

    return swap_groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use reciprocal local legends to swap two source colors through one stencil.

    Input structure:
    - Black background with isolated colored pixels and teal (8) stencil components.
    - A valid legend is still a 4-neighbor source->target pair whose continued
      direction reaches a teal-backed anchor, but every usable legend must appear
      together with its reverse target->source for the same stencil.

    Rule:
    - Learn only those color mappings that form a bidirectional pair A->B and B->A
      around one teal component.
    - For each loose A pixel, stamp that stencil at the pixel and recolor it to B;
      for each loose B pixel, stamp the same stencil recolored to A.
    - Erase all legend pixels and all teal stencil components from the result.

    Key insight: A stencil is now activated by a symmetric color swap, so the same
    nearby template proves both directions of one A<->B exchange.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=pattern_color)
    swap_groups = _discover_swap_groups(rules)

    legend_pixels: set[tuple[int, int]] = set()
    active_colors: set[int] = set()
    for group in swap_groups:
        for color, rule in group["rules_by_source"].items():
            active_colors.add(int(color))
            legend_pixels.add(tuple(rule["source_pos"]))
            legend_pixels.add(tuple(rule["target_pos"]))

    out = np.full_like(g, int(bg))

    # Preserve ordinary colored pixels that are not part of a reciprocal swap.
    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or color in active_colors:
            continue
        out[r, c] = color

    for group in swap_groups:
        for source_color, rule in group["rules_by_source"].items():
            ar, ac = rule["anchor"]
            pattern = rule["pattern"]
            target_color = int(rule["target_color"])
            for sr, sc in map(tuple, np.argwhere(g == int(source_color))):
                sr = int(sr)
                sc = int(sc)
                if (sr, sc) in legend_pixels:
                    continue
                dr = sr - int(ar)
                dc = sc - int(ac)
                rows = pattern[:, 0] + dr
                cols = pattern[:, 1] + dc
                ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
                out[rows[ok], cols[ok]] = target_color

    return out


# --------------- generator (c) ---------------

# Grid dimensions
GRID_DIM_MAX = 20
MIN_DIM_BY_SWAP_PAIRS = {1: 6, 2: 10, 3: 14, 4: 18}

# Swap-pair sampling
SWAP_PAIR_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

# Stencil size
STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 25

# Scattered source pixels per color within a swap pair
SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 8
SCATTERED_AREA_DIV = 35

# Bystander pixels
BYSTANDER_MAX = 5

# Placement retry budgets
LEGEND_RETRIES = 600
SCATTERED_RETRIES_PER_PIXEL = 80
BYSTANDER_RETRIES = 60
OUTER_RETRIES = 400

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


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
                candidate = (r + dr, c + dc)
                if candidate not in coords:
                    frontier.add(candidate)
        coords.add(random.choice(list(frontier)))
    coords_list = sorted(coords)
    anchor = random.choice(coords_list)
    return coords_list, anchor


def _stamp_cells(rel, anchor_r: int, anchor_c: int, H: int, W: int) -> set[tuple[int, int]]:
    cells = set()
    for rr, cc in rel:
        nr = int(anchor_r + rr)
        nc = int(anchor_c + cc)
        if 0 <= nr < H and 0 <= nc < W:
            cells.add((nr, nc))
    return cells


def _build_swaps(num_pairs: int, area: int):
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(1, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    swaps = []
    used_colors = set()
    for i in range(num_pairs):
        color_a = colors[2 * i]
        color_b = colors[2 * i + 1]
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(r - anchor[0], c - anchor[1]) for r, c in stencil]
        swaps.append(
            {
                "a": color_a,
                "b": color_b,
                "rel": rel,
                "n_a": random.randint(SCATTERED_MIN, sc_max),
                "n_b": random.randint(SCATTERED_MIN, sc_max),
            }
        )
        used_colors.update((color_a, color_b))

    return swaps, used_colors


def _place_legends(swaps, inp, occupied, reserved):
    H, W = inp.shape
    for swap in swaps:
        rel = swap["rel"]
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            ar = random.randint(0, H - 1)
            ac = random.randint(0, W - 1)

            legend_cells = {
                (ar - 2 * dr, ac - 2 * dc),
                (ar - dr, ac - dc),
                (ar + dr, ac + dc),
                (ar + 2 * dr, ac + 2 * dc),
            }
            if not all(0 <= r < H and 0 <= c < W for r, c in legend_cells):
                continue

            teal_cells = _stamp_cells(rel, ar, ac, H, W)
            if len(teal_cells) != len(rel):
                continue

            all_cells = legend_cells | teal_cells
            if len(all_cells) != len(legend_cells) + len(teal_cells):
                continue
            if all_cells & (occupied | reserved):
                continue

            a = int(swap["a"])
            b = int(swap["b"])
            inp[ar - 2 * dr, ac - 2 * dc] = a
            inp[ar - dr, ac - dc] = b
            inp[ar + dr, ac + dc] = a
            inp[ar + 2 * dr, ac + 2 * dc] = b
            for nr, nc in teal_cells:
                inp[nr, nc] = 8

            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            placed = True
            break

        if not placed:
            return False

    return True


def _place_scattered(swaps, inp, occupied, reserved, output_reserved):
    H, W = inp.shape
    for swap in swaps:
        swap["scattered"] = {}
        for color_key, count_key in (("a", "n_a"), ("b", "n_b")):
            color = int(swap[color_key])
            positions = []
            for _ in range(int(swap[count_key]) * SCATTERED_RETRIES_PER_PIXEL):
                if len(positions) >= int(swap[count_key]):
                    break
                r = random.randint(0, H - 1)
                c = random.randint(0, W - 1)
                if (r, c) in occupied or (r, c) in reserved:
                    continue
                stamp = _stamp_cells(swap["rel"], r, c, H, W)
                if not stamp or stamp & output_reserved:
                    continue
                positions.append((r, c))
                occupied.add((r, c))
                _reserve_4neighbors(reserved, [(r, c)])
                output_reserved.update(stamp)
            if len(positions) < SCATTERED_MIN:
                return False
            swap["scattered"][color] = positions
            for r, c in positions:
                inp[r, c] = color
    return True


def _place_bystanders(used_colors, inp, occupied, reserved, output_reserved):
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
            if (r, c) in occupied or (r, c) in reserved or (r, c) in output_reserved:
                continue
            color = random.choice(bystander_colors)
            inp[r, c] = color
            placed.append((r, c, color))
            occupied.add((r, c))
            _reserve_4neighbors(reserved, [(r, c)])
            break
    return placed


def _build_output(swaps, bystander_pos, H: int, W: int):
    output = np.zeros((H, W), dtype=int)
    for r, c, color in bystander_pos:
        output[int(r), int(c)] = int(color)
    for swap in swaps:
        for sr, sc in swap["scattered"][int(swap["a"])]:
            for rr, cc in swap["rel"]:
                nr = int(sr + rr)
                nc = int(sc + cc)
                if 0 <= nr < H and 0 <= nc < W:
                    output[nr, nc] = int(swap["b"])
        for sr, sc in swap["scattered"][int(swap["b"])]:
            for rr, cc in swap["rel"]:
                nr = int(sr + rr)
                nc = int(sc + cc)
                if 0 <= nr < H and 0 <= nc < W:
                    output[nr, nc] = int(swap["a"])
    return output


def _gen_attempt():
    num_pairs = random.choice(SWAP_PAIR_WEIGHTS)
    min_dim = MIN_DIM_BY_SWAP_PAIRS[num_pairs]
    H = random.randint(min_dim, GRID_DIM_MAX)
    W = random.randint(min_dim, GRID_DIM_MAX)

    swaps, used_colors = _build_swaps(num_pairs, H * W)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    output_reserved: set[tuple[int, int]] = set()
    inp = np.zeros((H, W), dtype=int)

    if not _place_legends(swaps, inp, occupied, reserved):
        return None
    if not _place_scattered(swaps, inp, occupied, reserved, output_reserved):
        return None
    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved, output_reserved)

    output = _build_output(swaps, bystander_pos, H, W)
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
