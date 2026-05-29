# Variant of abc82100
# Dimension: D6 — 颜色重着色规则 (transform)
# Change: teal模板像素统一重着色为规则的目标色（单一颜色替换）... → 目标色由源像素的局部环境决定

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the D6 variant
# ----------------------------------
# spec_variant:
#   concrete_behavior:
#   - The legend still teaches only "which source color owns which teal stencil".
#   - The adjacent legend guide pixel no longer supplies a global target color.
#     Instead it is merely one local witness: its color is chosen so that the
#     legend source pixel itself would also have that color as its 4-neighbor
#     majority.
#   - Every scattered source pixel now carries a small cross-shaped local
#     environment: three colored 4-neighbors are placed around it, with a 2-vs-1
#     color split. The repeated color is the vote winner and therefore the stamp
#     color for that specific source instance.
#   behavior_trace:
#   - Original: if a legend shows `2 -> 4` next to stencil S, then every non-legend
#     `2` stamps S in color `4`, regardless of surrounding pixels.
#   - Variant: the same legend still links source color `2` to stencil S, but a
#     scattered `2` with neighbors `{6, 6, 9}` stamps S in `6`, while another `2`
#     with neighbors `{3, 3, 6}` stamps S in `3`.
#   - Difference: the shared property is now only the stencil identity; the output
#     color becomes per-instance and must be read from visible local evidence.
#   complexity_calibration:
#   - The local vote uses only a small shared palette of 2-3 non-source colors, so
#     examples repeat enough for humans to notice the majority rule.
#   - At least one rule in each generated grid is forced to produce two different
#     vote winners across its scattered source pixels, preventing regression to the
#     old fixed-target behavior.
#   implicit_assumptions_review:
#   - `_discover_rules(...)` originally depended on a differently colored adjacent
#     pixel to locate the anchor and stencil. That still holds, so the legend pair
#     is preserved.
#   - The old `main()` wrote `rule["target_color"]` for every source occurrence.
#     That assumed one target per source color; it is the central line that must
#     change for D6.
#   - The old preserve step skipped every pixel whose color belonged to any source
#     color. That remains valid only because the generator now guarantees that
#     contextual support pixels never use source colors.
#   - The old legend placer could allow teal cells to overlap or touch the example
#     source in ways that were mostly harmless before. Here that would corrupt the
#     local vote, so legend placement explicitly forbids teal from occupying the
#     source/guide cells or the source's other 4-neighbors.
#   placement_legality:
#   - Each scattered source is placed strictly away from borders so all 4-neighbors
#     exist.
#   - Its full cross footprint is reserved: one center, three colored support
#     neighbors, and one intentionally blank neighbor. Later objects cannot enter
#     that footprint or its immediate 4-neighborhood, so the vote stays stable.
#   - Legends, scattered contexts, and optional bystanders are kept disjoint in the
#     input; output stamps may still clip at borders only if the stencil itself
#     extends outward after translation, matching the original task family.
#   visual_observability:
#   - The visible cue is count-based and spatial: repeated colors around a source
#     outnumber the minority color by 2-to-1, and the stamped stencil adopts that
#     repeated color.
#   - Because the same source color can yield multiple output colors within one
#     grid, the old fixed mapping is visibly contradicted.
#   cleanliness:
#   - Support pixels live in compact 4-neighbor crosses around sources; other
#     colored objects are kept separated, so the voting evidence remains legible.
#   new_main_assumptions_checked_against_generator:
#   - Every non-legend source has a unique non-background 4-neighbor majority.
#   - Support colors are never in `source_colors`, so the preserve logic does not
#     accidentally erase them before stamps are applied.
#   - No unrelated colored pixel is allowed to enter a source's 4-neighborhood, so
#     the per-source vote read by `main()` matches the generator's plan exactly.


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

            for dr, dc in _DIRS4:
                tr = r + dr
                tc = c + dc
                if not (0 <= tr < h and 0 <= tc < w):
                    continue

                guide_color = int(grid[tr, tc])
                if guide_color in (int(bg), int(pattern_color)) or guide_color == source_color:
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
                        "guide_color": int(guide_color),
                        "source_pos": (int(r), int(c)),
                        "guide_pos": (int(tr), int(tc)),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "pattern": np.asarray(pattern, dtype=int),
                    }
                )

    return rules


def _majority_non_bg_neighbor_color(
    grid: np.ndarray, r: int, c: int, bg: int
) -> int | None:
    counts: dict[int, int] = {}
    h, w = grid.shape
    for dr, dc in _DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            color = int(grid[nr, nc])
            if color != int(bg):
                counts[color] = counts.get(color, 0) + 1
    if not counts:
        return None
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    if len(winners) != 1:
        return None
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn stencil ownership from local source-guide-pattern examples, then stamp
    each scattered source using a locally voted color.

    Input structure:
    - Black background with teal (8) stencil components, legend pairs of adjacent
      non-background colors, and scattered copies of the legend source colors.
    - A legend is a 4-neighbor source->guide pair whose continued direction reaches
      an anchor touching a teal component; this identifies which stencil belongs to
      that source color.
    - Each scattered source is surrounded by a small colored 4-neighborhood that
      determines its output color.

    Rule:
    - For each discovered legend, copy its teal stencil onto every other pixel of
      the same source color, translating the legend anchor onto that source pixel.
    - The copied stencil is recolored per instance: inspect the source pixel's
      4-neighbors and pick the unique most frequent non-background color.
    - Remove all legend pixels and all original teal pixels; preserve unrelated
      ordinary colored pixels unless a later stamp covers them.

    Key insight: the legend no longer defines a global target color. It only tells
    you which source color uses which stencil; the actual recolor comes from the
    local majority around each scattered source occurrence.
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
        legend_pixels.add(rule["guide_pos"])
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
        for sr, sc in map(tuple, np.argwhere(g == rule["source_color"])):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            target_color = _majority_non_bg_neighbor_color(g, sr, sc, int(bg))
            if target_color is None:
                continue
            dr = sr - ar
            dc = sc - ac
            rows = pattern[:, 0] + dr
            cols = pattern[:, 1] + dc
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = int(target_color)

    return out


# --------------- generator (c) ---------------

# ── CFG: tunable generation parameters ──────────────────────────────
GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 7, 2: 10, 3: 14, 4: 18}

RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 25

SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 8
SCATTERED_AREA_DIV = 30
SCATTERED_MAX_BY_RULES = {1: 8, 2: 6, 3: 5, 4: 4}

BYSTANDER_MAX = 5

LEGEND_RETRIES = 500
SCATTERED_RETRIES_PER_PIXEL = 80
BYSTANDER_RETRIES = 60
OUTER_RETRIES = 300

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((r + dr, c + dc))


def _make_stencil(min_pixels: int, max_pixels: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
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


def _build_rules(num_rules: int, area: int) -> tuple[list[dict], set[int]]:
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(1, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(
        SCATTERED_MAX_CAP,
        SCATTERED_MAX_BY_RULES[num_rules],
        max(SCATTERED_MIN, area // SCATTERED_AREA_DIV),
    )

    rules: list[dict] = []
    source_colors: set[int] = set()
    for i in range(num_rules):
        src_c = colors[i]
        source_colors.add(src_c)
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(r - anchor[0], c - anchor[1]) for r, c in stencil]
        rules.append(
            {
                "src": src_c,
                "rel": rel,
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
    return rules, source_colors


def _pick_vote_palette(source_colors: set[int]) -> list[int]:
    available = [c for c in NON_BG_NON_TEAL_COLORS if c not in source_colors]
    k = min(len(available), random.choice([2, 3]))
    if k < 2:
        raise RuntimeError("Need at least two non-source colors for local voting")
    return random.sample(available, k)


def _assign_majority_sequences(rules: list[dict], vote_palette: list[int]) -> None:
    varying_rules = []
    for rule in rules:
        majors = [random.choice(vote_palette) for _ in range(rule["n_sc"])]
        rule["majority_colors"] = majors
        if len(set(majors)) >= 2:
            varying_rules.append(rule)
    if varying_rules:
        return
    rule = random.choice(rules)
    rule["majority_colors"][0] = vote_palette[0]
    rule["majority_colors"][1] = vote_palette[1]


def _place_legends(
    rules: list[dict],
    vote_palette: list[int],
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

            teal_cells = set()
            ok = True
            for rr, cc in rel:
                nr, nc = ar + rr, ac + cc
                if not (0 <= nr < h and 0 <= nc < w):
                    ok = False
                    break
                teal_cells.add((nr, nc))
            if not ok:
                continue
            if (sr, sc) in teal_cells or (tr, tc) in teal_cells:
                continue

            other_source_neighbors = {
                (sr + rr, sc + cc)
                for rr, cc in _DIRS4
                if (sr + rr, sc + cc) != (tr, tc)
            }
            if teal_cells & other_source_neighbors:
                continue

            actual_cells = {(sr, sc), (tr, tc)} | teal_cells
            if actual_cells & (occupied | reserved):
                continue

            guide_color = random.choice(vote_palette)
            inp[sr, sc] = rule["src"]
            inp[tr, tc] = guide_color
            for nr, nc in teal_cells:
                inp[nr, nc] = 8

            occupied.update(actual_cells)
            reserved.update(other_source_neighbors)
            _reserve_4neighbors(reserved, actual_cells)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_scattered(
    rules: list[dict],
    vote_palette: list[int],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    h, w = inp.shape
    for rule in rules:
        positions: list[tuple[int, int]] = []
        support_pixels: list[tuple[int, int, int]] = []
        for majority_color in rule["majority_colors"]:
            placed = False
            for _ in range(SCATTERED_RETRIES_PER_PIXEL):
                sr = random.randint(1, h - 2)
                sc = random.randint(1, w - 2)
                cross = [(sr + dr, sc + dc) for dr, dc in _DIRS4]
                footprint = {(sr, sc), *cross}
                if footprint & (occupied | reserved):
                    continue

                blank_idx = random.randrange(4)
                support_idxs = [i for i in range(4) if i != blank_idx]
                random.shuffle(support_idxs)
                minority_color = random.choice([c for c in vote_palette if c != majority_color])
                color_plan = [majority_color, majority_color, minority_color]
                random.shuffle(color_plan)

                actual_support = []
                for slot_idx, color in zip(support_idxs, color_plan):
                    rr, cc = cross[slot_idx]
                    actual_support.append((rr, cc, color))

                inp[sr, sc] = rule["src"]
                occupied.add((sr, sc))
                positions.append((sr, sc))
                for rr, cc, color in actual_support:
                    inp[rr, cc] = color
                    occupied.add((rr, cc))
                    support_pixels.append((rr, cc, color))

                reserved.update(footprint)
                _reserve_4neighbors(reserved, {(sr, sc)} | {(rr, cc) for rr, cc, _ in actual_support})
                placed = True
                break
            if not placed:
                return False

        rule["scattered"] = positions
        rule["support_pixels"] = support_pixels
    return True


def _place_bystanders(
    source_colors: set[int],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    available = [c for c in NON_BG_NON_TEAL_COLORS if c not in source_colors]
    if not available:
        return []
    h, w = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(available) * 2))
    placed: list[tuple[int, int, int]] = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            color = random.choice(available)
            inp[r, c] = color
            placed.append((r, c, color))
            occupied.add((r, c))
            _reserve_4neighbors(reserved, [(r, c)])
            break
    return placed


def _build_output(
    rules: list[dict], preserved_pixels: list[tuple[int, int, int]], h: int, w: int
) -> np.ndarray:
    output = np.zeros((h, w), dtype=int)
    for r, c, color in preserved_pixels:
        output[r, c] = color
    for rule in rules:
        rel = rule["rel"]
        for (sr, sc), target_color in zip(rule["scattered"], rule["majority_colors"]):
            for rr, cc in rel:
                nr, nc = sr + rr, sc + cc
                if 0 <= nr < h and 0 <= nc < w:
                    output[nr, nc] = int(target_color)
    return output


def _gen_attempt() -> np.ndarray | None:
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(min_dim, GRID_DIM_MAX)

    rules, source_colors = _build_rules(num_rules, h * w)
    vote_palette = _pick_vote_palette(source_colors)
    _assign_majority_sequences(rules, vote_palette)

    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    inp = np.zeros((h, w), dtype=int)

    if not _place_legends(rules, vote_palette, inp, occupied, reserved):
        return None
    if not _place_scattered(rules, vote_palette, inp, occupied, reserved):
        return None
    bystanders = _place_bystanders(source_colors, inp, occupied, reserved)

    preserved = bystanders[:]
    for rule in rules:
        preserved.extend(rule["support_pixels"])

    output = _build_output(rules, preserved, h, w)
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
