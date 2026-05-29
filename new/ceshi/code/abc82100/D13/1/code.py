# Variant of abc82100
# Dimension: D13 — 源色-目标色映射的对称性 (meta)
# Change: 映射可以是任意的（如A→B，C→D），不要求对称或逆映射... → 循环映射（A→B→C→A）

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the D13 variant
# a) Concreteization and behavior trace
#    - The input still uses local legends of the form source -> target -> anchor,
#      with a nearby teal stencil that will later be copied.
#    - The crucial change is that only the three legends whose colors form the
#      directed cycle A->B, B->C, C->A are operative. A fourth legend may appear
#      as a non-cyclic distractor and must be ignored.
#    - Non-legend pixels may use any of the three cycle colors. Each such pixel is
#      a source and gets replaced by its own rule's stencil recolored to the next
#      color in the cycle. Non-legend pixels of the distractor color stay unchanged.
#    - Small trace:
#        original task: if legends say 2->4 and 6->7, then every scattered 2 gives
#        a 4-stencil and every scattered 6 gives a 7-stencil. Rules are unrelated.
#        variant: legends say 2->4, 4->7, 7->2. If we processed in-place, the new
#        4 pixels created from color 2 could be mistaken for fresh sources and then
#        turn into 7. Therefore the solver must read all source positions from the
#        original grid first, then stamp once.
# b) Complexity calibration
#    - Cycle length is fixed to 3 so the repeated A/B/C evidence is visible in a
#      few ARC examples instead of requiring a large color graph.
#    - One extra non-cycle legend is added on purpose. This is the smallest extra
#      adaptation that makes the cycle itself semantically necessary instead of a
#      cosmetic relabeling that the original independent-rule solver would still solve.
#    - Each rule keeps its own stencil; the only meta-change is the cyclic color
#      relation. This preserves the original task's visual language.
#    - Stencils are forced to have at least 2 cells so an incorrect sequential
#      cascade can actually diverge from the intended simultaneous output.
# c) Hidden assumptions in the original main()
#    - Discovering rules from local legends still works unchanged.
#    - The old loop "for each rule, scan g == source_color and stamp immediately"
#      is only safe because it reads from the immutable input grid. In this variant
#      that snapshot behavior is no longer incidental; it is the core correctness
#      condition because every target color is also another source color.
#    - Preserving non-consumed colored pixels still works if we exclude all three
#      cycle colors, since every cycle color is meant to be transformed.
# d) Placement legality
#    - Legends are kept separated exactly as before so each source-target pair has
#      one nearby teal component and no accidental competitors.
#    - Scattered source pixels are placed so their full stencil fits in bounds and
#      different output stamps do not overlap. This keeps the cycle effect readable
#      and avoids ambiguous overwrite conventions unrelated to D13.
#    - Distractor-color scattered pixels are placed outside all active stamp areas,
#      so preserving them is unambiguous.
# e) Visual observability
#    - Every sample contains the three directed legends A->B, B->C, C->A, plus one
#      broken pair whose colors do not close the loop. A human can see that only the
#      cycle colors appear once as a source and once as a target.
#    - Generator rejection compares the intended output with a naive sequential
#      cascade simulation and keeps only samples where they differ, ensuring the
#      "apply simultaneously" aspect is not just theoretical.
# f) Visual cleanliness
#    - Full in-bounds, non-overlapping stamps are preferred for scattered sources.
#      This removes clutter from clipped or colliding shapes and makes the cycle
#      the main thing to notice.
# g) Color-change inferability
#    - Only three active colors participate in the cycle. The distractor source color
#      and any other bystander colors are excluded from the cycle and remain unchanged.


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


def _find_cycle_rules(rules: list[dict]) -> list[dict]:
    for first in rules:
        for second in rules:
            if second is first or second["source_color"] != first["target_color"]:
                continue
            for third in rules:
                if third is first or third is second:
                    continue
                if third["source_color"] != second["target_color"]:
                    continue
                if third["target_color"] != first["source_color"]:
                    continue
                colors = {
                    int(first["source_color"]),
                    int(second["source_color"]),
                    int(third["source_color"]),
                }
                if len(colors) == 3:
                    return [first, second, third]
    return []


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn the unique three-color cycle of local source-target-pattern examples and apply it once.

    Input structure:
    - Black background with isolated colored pixels plus teal (8) stencil components.
    - Several local legends show 4-neighbor source->target pairs whose continued
      direction reaches anchors touching teal stencils.
    - Exactly three of those legends form one directed cycle A->B, B->C, C->A;
      any extra legend outside that cycle is a distractor.

    Rule:
    - Every non-legend pixel whose color is A, B, or C is a source pixel for the
      corresponding cycle legend. Copy that legend's teal stencil onto the source
      pixel, recoloring it to the next color in the cycle.
    - All three source-color groups must be read from the original grid before any
      stamping, so newly painted target pixels do not transform again in the same step.
    - Remove all legend pixels and all teal stencil pixels; preserve bystanders and
      distractor-color pixels whose colors are outside the three-color cycle.

    Key insight: the legends do not define independent substitutions anymore; only the
    closed three-color loop is valid. Once that loop is identified, the output must be
    built from a snapshot of the original cycle-colored source pixels rather than by
    in-place recoloring.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=pattern_color)
    cycle_rules = _find_cycle_rules(rules)
    active_rules = cycle_rules if cycle_rules else rules

    legend_pixels: set[tuple[int, int]] = set()
    cycle_colors: set[int] = set()
    for rule in rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])
    for rule in active_rules:
        cycle_colors.add(int(rule["source_color"]))

    source_positions: dict[int, list[tuple[int, int]]] = {color: [] for color in cycle_colors}
    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        if (r, c) in legend_pixels:
            continue
        color = int(g[r, c])
        if color in source_positions:
            source_positions[color].append((r, c))

    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or color in cycle_colors:
            continue
        out[r, c] = color

    for rule in active_rules:
        ar, ac = rule["anchor"]
        pattern = rule["pattern"]
        for sr, sc in source_positions.get(int(rule["source_color"]), []):
            dr = int(sr) - int(ar)
            dc = int(sc) - int(ac)
            rows = pattern[:, 0] + dr
            cols = pattern[:, 1] + dc
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = int(rule["target_color"])

    return out


# --------------- generator ---------------

GRID_DIM_MIN = 18
GRID_DIM_MAX = 20
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

STENCIL_MAX_CAP = 5
STENCIL_MIN_AREA_DIV = 120
STENCIL_MAX_AREA_DIV = 45

SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 4
SCATTERED_AREA_DIV = 60
DECOY_SCATTERED_MIN = 2
DECOY_SCATTERED_MAX = 3

BYSTANDER_MAX = 2

LEGEND_RETRIES = 700
SCATTERED_RETRIES_PER_PIXEL = 90
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 400

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _reserve_8neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                reserved.add((int(r) + dr, int(c) + dc))


def _make_stencil(min_pixels: int, max_pixels: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    n = random.randint(max(2, min_pixels), max(2, max_pixels))
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


def _build_rules(area: int) -> tuple[list[dict], set[int]]:
    chosen = random.sample(NON_BG_NON_TEAL_COLORS, 5)
    cycle_colors = chosen[:3]
    decoy_src, decoy_tgt = chosen[3], chosen[4]

    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(2, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules: list[dict] = []
    for i, src_c in enumerate(cycle_colors):
        tgt_c = cycle_colors[(i + 1) % 3]
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(int(r - anchor[0]), int(c - anchor[1])) for r, c in stencil]
        rules.append(
            {
                "src": int(src_c),
                "tgt": int(tgt_c),
                "rel": rel,
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
                "active": True,
            }
        )
    stencil, anchor = _make_stencil(min_s, max_s)
    rel = [(int(r - anchor[0]), int(c - anchor[1])) for r, c in stencil]
    rules.append(
        {
            "src": int(decoy_src),
            "tgt": int(decoy_tgt),
            "rel": rel,
            "n_sc": random.randint(DECOY_SCATTERED_MIN, DECOY_SCATTERED_MAX),
            "active": False,
        }
    )
    return rules, {int(c) for c in chosen}


def _translated_cells(rel: list[tuple[int, int]], sr: int, sc: int) -> set[tuple[int, int]]:
    return {(int(sr) + int(rr), int(sc) + int(cc)) for rr, cc in rel}


def _fits_translation(rel: list[tuple[int, int]], sr: int, sc: int, h: int, w: int) -> bool:
    for rr, cc in rel:
        nr = int(sr) + int(rr)
        nc = int(sc) + int(cc)
        if not (0 <= nr < h and 0 <= nc < w):
            return False
    return True


def _place_legends(
    rules: list[dict],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    h, w = inp.shape
    for rule in rules:
        rel = rule["rel"]
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr, sc = random.randint(0, h - 1), random.randint(0, w - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
                continue

            teal_cells = _translated_cells(rel, ar, ac)
            if any(not (0 <= r < h and 0 <= c < w) for r, c in teal_cells):
                continue
            if (sr, sc) in teal_cells or (tr, tc) in teal_cells:
                continue

            all_cells = {(sr, sc), (tr, tc), (ar, ac)} | teal_cells
            if all_cells & occupied or all_cells & reserved:
                continue

            inp[sr, sc] = int(rule["src"])
            inp[tr, tc] = int(rule["tgt"])
            for r, c in teal_cells:
                inp[r, c] = 8
            occupied.update(all_cells)
            _reserve_8neighbors(reserved, all_cells)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_scattered(
    rules: list[dict],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    h, w = inp.shape
    footprint_used = set(occupied)
    for rule in rules:
        positions: list[tuple[int, int]] = []
        budget = rule["n_sc"] * SCATTERED_RETRIES_PER_PIXEL
        for _ in range(budget):
            if len(positions) >= rule["n_sc"]:
                break
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            stamp_cells = {(r, c)}
            if rule["active"]:
                if not _fits_translation(rule["rel"], r, c, h, w):
                    continue
                stamp_cells = _translated_cells(rule["rel"], r, c)
                if stamp_cells & footprint_used:
                    continue
                if stamp_cells & reserved:
                    continue
            elif (r, c) in footprint_used:
                continue

            positions.append((r, c))
            occupied.add((r, c))
            if rule["active"]:
                footprint_used.update(stamp_cells)
            _reserve_4neighbors(reserved, [(r, c)])

        min_needed = SCATTERED_MIN if rule["active"] else DECOY_SCATTERED_MIN
        if len(positions) < min_needed:
            return False
        rule["scattered"] = positions
        for r, c in positions:
            inp[r, c] = int(rule["src"])
    reserved.update(footprint_used)
    return True


def _place_bystanders(
    used_colors: set[int],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []

    h, w = inp.shape
    n = random.randint(0, BYSTANDER_MAX)
    placed: list[tuple[int, int, int]] = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            color = random.choice(bystander_colors)
            inp[r, c] = int(color)
            placed.append((r, c, int(color)))
            occupied.add((r, c))
            _reserve_4neighbors(reserved, [(r, c)])
            break
    return placed


def _build_output(rules: list[dict], bystanders: list[tuple[int, int, int]], h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for r, c, color in bystanders:
        out[int(r), int(c)] = int(color)
    for rule in rules:
        if not rule["active"]:
            for sr, sc in rule["scattered"]:
                out[int(sr), int(sc)] = int(rule["src"])
            continue
        for sr, sc in rule["scattered"]:
            for rr, cc in rule["rel"]:
                nr = int(sr) + int(rr)
                nc = int(sc) + int(cc)
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = int(rule["tgt"])
    return out


def _build_naive_cascade_output(
    rules: list[dict],
    bystanders: list[tuple[int, int, int]],
    h: int,
    w: int,
) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for r, c, color in bystanders:
        out[int(r), int(c)] = int(color)

    positions_by_color: dict[int, set[tuple[int, int]]] = {}
    for rule in rules:
        if not rule["active"]:
            continue
        positions_by_color.setdefault(int(rule["src"]), set()).update(
            {(int(r), int(c)) for r, c in rule["scattered"]}
        )

    for rule in rules:
        if not rule["active"]:
            continue
        src = int(rule["src"])
        tgt = int(rule["tgt"])
        current_sources = list(positions_by_color.get(src, set()))
        for sr, sc in current_sources:
            for rr, cc in rule["rel"]:
                nr = int(sr) + int(rr)
                nc = int(sc) + int(cc)
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = tgt
                    positions_by_color.setdefault(tgt, set()).add((nr, nc))
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_DIM_MIN, GRID_DIM_MAX)
    w = random.randint(GRID_DIM_MIN, GRID_DIM_MAX)

    rules, used_colors = _build_rules(h * w)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    inp = np.zeros((h, w), dtype=int)

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    if not _place_scattered(rules, inp, occupied, reserved):
        return None
    bystanders = _place_bystanders(used_colors, inp, occupied, reserved)

    output = _build_output(rules, bystanders, h, w)
    naive = _build_naive_cascade_output(rules, bystanders, h, w)
    if np.array_equal(output, naive):
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
