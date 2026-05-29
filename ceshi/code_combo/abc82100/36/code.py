# Combo variant of abc82100 (index 36)
# combo_spec: D1/4+D11/1
# Name: 频率消歧+分散teal片段

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, get_bg


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - Original task: each legend is `source pixel -> target pixel -> teal stencil`,
#      and every scattered pixel of the same source color uses that one stencil.
#    - This combo breaks both simplifying assumptions at once.
#      1. D11/1: a legend's stencil is no longer one teal component; it is the union
#         of several 8-disconnected teal fragments inside a compact 5x5 window.
#      2. D1/4: the legend source color is no longer enough to identify which rule a
#         scattered color should use. Instead, source colors are matched to legends by
#         global frequency rank: the most frequent scattered color gets the largest
#         fragmented stencil, the next-most-frequent color gets the next-largest one,
#         and so on.
#    - Small trace:
#      input may contain three legends with the same source color S but different
#      target colors and teal fragment unions of sizes 8, 6, and 4. Elsewhere there
#      are scattered colors A, B, C with counts 6, 5, 3. The old solver would either
#      fail or try to use S literally. The combo solver instead:
#        1. globally counts A/B/C,
#        2. locally reconstructs the three fragmented teal unions,
#        3. sorts colors by frequency and legends by template size,
#        4. maps A->size8 rule, B->size6 rule, C->size4 rule.
# b) Complexity calibration
#    - Rule count is forced to 2-3. With one rule, frequency ranking would be
#      invisible.
#    - Source-color frequencies are all distinct and differ by at least one, while
#      template sizes are also all distinct. This keeps the ranking readable and
#      removes hidden tie-breaking.
#    - Every stencil has at least 2 fragments and at least 4 teal pixels, so the
#      local regrouping step is always genuinely needed.
# c) Implicit assumption audit of the original main()
#    - Original `_discover_rules()` assumed one teal component per legend. That fails
#      here; we must merge every teal fragment inside the anchor's local radius-2
#      window into one pattern.
#    - Original application loop assumed `source_color -> one rule`. That is exactly
#      what D1/4 removes. The new application step must first discover all legends,
#      then rank legends by pattern size, then rank scattered colors by global count.
#    - Original preservation logic skipped any pixel whose color was a legend source
#      color. That is no longer meaningful because legend source colors are shared and
#      are only decorative. The new solver instead preserves only colors that are not
#      among the top-frequency scattered source classes.
# d) Placement legality
#    - Generator keeps each legend's fragmented stencil fully inside a radius-2 window
#      around its anchor and reserves that whole window from other objects.
#    - Scattered pixels and bystanders are 4-neighbor separated, preventing accidental
#      colored pairs that would look like new legends.
#    - Legends are also kept apart from each other so local teal windows never merge.
# e) Visual observability
#    - Humans can see that the legends do not distinguish themselves by source color:
#      at least two legends reuse the same source color.
#    - They can also see a clear global cue: some scattered colors appear many times,
#      others fewer times, and the larger fragmented templates correspond to the more
#      frequent colors.
#    - The global signal and local signal reinforce each other: the rule choice comes
#      from frequency, but the stamped geometry requires correctly assembling the teal
#      fragments.
# f) Visual cleanliness
#    - The local 5x5 legend window limits teal clutter.
#    - Scattered sources stay isolated, and bystanders use only low-frequency leftover
#      colors, so the frequency ranking remains visually legible.
# g) Color-rule inferability
#    - No color-id ordering is used. Only visible counts and visible template sizes
#      matter.
#    - Target colors remain legend-local and are carried by the matched legend once
#      ranking picks that legend.
# h) Degeneration checks the generator is designed to satisfy
#    - Fragmentation check: every legend has at least 2 teal components in its local
#      window.
#    - Frequency check: the scattered source-color counts are strictly ordered.
#    - Non-originality check: the base solver, which still keys off legend source
#      color, should usually disagree because the legends intentionally reuse source
#      colors while the scattered colors are matched by frequency instead.
# i) New solver assumptions and generator guarantees
#    - Assumption: the source colors are exactly the top `len(rules)` non-legend
#      non-teal colors by frequency.
#      Guarantee: generator gives every true source color a distinct count >= 3 and
#      any bystander color count <= 2.
#    - Assumption: legend template sizes are distinct.
#      Guarantee: generator samples fragmented stencils with unique total pixel counts.
#    - Assumption: the local radius-2 teal window around a legend contains only that
#      legend's fragments.
#      Guarantee: generator reserves the full window during placement.


LOCAL_RADIUS = 2

GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {2: 10, 3: 14}

RULE_COUNT_WEIGHTS = [2, 2, 3, 3]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

STENCIL_TOTAL_MIN = 4
STENCIL_TOTAL_MAX_CAP = 9
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 24
FRAGMENT_COUNT_MIN = 2
FRAGMENT_COUNT_MAX = 4
FRAGMENT_SIZE_CAP = 3
FRAGMENT_RETRIES = 240

SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 8
SCATTERED_AREA_DIV = 26
BYSTANDER_MAX = 3
BYSTANDER_FREQ_MAX = 2

LEGEND_RETRIES = 600
SCATTERED_RETRIES_PER_PIXEL = 80
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 400

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_LOCAL_CELLS = [
    (r, c)
    for r in range(-LOCAL_RADIUS, LOCAL_RADIUS + 1)
    for c in range(-LOCAL_RADIUS, LOCAL_RADIUS + 1)
]


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _max_anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return max(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _collect_local_pattern(
    patterns: list[np.ndarray], anchor: tuple[int, int], radius: int
) -> tuple[np.ndarray, tuple[int, ...]] | None:
    component_ids: list[int] = []
    coords: set[tuple[int, int]] = set()
    anchor_is_teal = False

    for i, pattern in enumerate(patterns):
        if _max_anchor_distance(pattern, anchor) > radius:
            continue
        component_ids.append(int(i))
        for r, c in pattern:
            cell = (int(r), int(c))
            coords.add(cell)
            if cell == anchor:
                anchor_is_teal = True

    if not anchor_is_teal or len(component_ids) < 2:
        return None

    merged = np.asarray(sorted(coords), dtype=int)
    return merged, tuple(component_ids)


def _discover_rules(
    grid: np.ndarray, bg: int, pattern_color: int = 8, local_radius: int = LOCAL_RADIUS
) -> list[dict]:
    h, w = grid.shape
    patterns = components_mask(grid == pattern_color, connectivity=8)
    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], tuple[int, ...]]] = set()

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

                target_color = int(grid[tr, tc])
                if target_color in (int(bg), int(pattern_color)) or target_color == source_color:
                    continue

                anchor = (tr + dr, tc + dc)
                if not (0 <= anchor[0] < h and 0 <= anchor[1] < w):
                    continue

                pattern_info = _collect_local_pattern(patterns, anchor, local_radius)
                if pattern_info is None:
                    continue
                pattern, component_ids = pattern_info

                key = ((int(r), int(c)), (int(tr), int(tc)), component_ids)
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
                        "pattern": pattern,
                        "pattern_size": int(len(pattern)),
                        "component_ids": component_ids,
                    }
                )

    return rules


def _ordered_scattered_colors(
    grid: np.ndarray, legend_pixels: set[tuple[int, int]], bg: int, pattern_color: int, count: int
) -> list[int]:
    counts: Counter[int] = Counter()
    first_seen: dict[int, tuple[int, int]] = {}
    for r, c in map(tuple, np.argwhere((grid != int(bg)) & (grid != pattern_color))):
        r = int(r)
        c = int(c)
        if (r, c) in legend_pixels:
            continue
        color = int(grid[r, c])
        counts[color] += 1
        first_seen.setdefault(color, (r, c))

    ordered = sorted(
        counts.items(),
        key=lambda kv: (-int(kv[1]), int(first_seen[kv[0]][0]), int(first_seen[kv[0]][1]), int(kv[0])),
    )
    return [int(color) for color, _freq in ordered[:count]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild fragmented teal legends, then match scattered colors by global frequency rank.

    Input structure:
    - The grid has black background, compact local legends, scattered colored source pixels,
      and teal(8) pixels.
    - Each legend is still a 4-neighbor source->target pair followed by an anchor cell, but
      the nearby teal stencil is split into several disconnected fragments inside a 5x5 window.
    - Legend source colors may repeat, so they no longer uniquely identify which scattered
      source color should use which legend.

    Rule:
    - Recover each legend and merge all teal fragments in its local anchor window into one
      stencil. Rank the legends by total stencil size from largest to smallest.
    - Among non-legend colored pixels, count color frequencies globally and take the top
      `number_of_legends` colors. Rank those colors from most frequent to least frequent.
    - Pair the two rankings in order: the most frequent scattered color uses the largest
      reconstructed stencil, the next-most-frequent color uses the next-largest stencil, and
      so on. Stamp the chosen stencil onto every pixel of that color, recolored to the paired
      legend's target color. Remove all legends and teal fragments.

    Key insight:
    - Solving now requires two scales of reasoning at once: global counting to decide which
      color gets which rule, and local grouping to reconstruct each legend's fragmented teal
      stencil before stamping it.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=pattern_color)
    legend_pixels: set[tuple[int, int]] = set()
    for rule in rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])

    ordered_rules = sorted(
        rules,
        key=lambda rule: (-int(rule["pattern_size"]), int(rule["anchor"][0]), int(rule["anchor"][1])),
    )
    ordered_colors = _ordered_scattered_colors(
        g, legend_pixels=legend_pixels, bg=int(bg), pattern_color=pattern_color, count=len(ordered_rules)
    )
    color_to_rule = {
        int(color): ordered_rules[i]
        for i, color in enumerate(ordered_colors)
        if i < len(ordered_rules)
    }

    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or color in color_to_rule:
            continue
        out[r, c] = color

    for color, rule in color_to_rule.items():
        ar, ac = rule["anchor"]
        pattern = rule["pattern"]
        for sr, sc in map(tuple, np.argwhere(g == int(color))):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            dr = sr - int(ar)
            dc = sc - int(ac)
            rows = pattern[:, 0] + dr
            cols = pattern[:, 1] + dc
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = int(rule["target_color"])

    return out


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r + dr), int(c + dc)))


def _reserve_window(reserved: set[tuple[int, int]], anchor: tuple[int, int], radius: int) -> None:
    ar, ac = anchor
    for rr in range(ar - radius, ar + radius + 1):
        for cc in range(ac - radius, ac + radius + 1):
            reserved.add((int(rr), int(cc)))


def _far_from_cells(cell: tuple[int, int], cells: set[tuple[int, int]]) -> bool:
    return all(_chebyshev(cell, other) >= 2 for other in cells)


def _random_partition(total: int, parts: int, piece_cap: int) -> list[int]:
    pieces = [1] * parts
    remaining = total - parts
    while remaining > 0:
        candidates = [i for i, size in enumerate(pieces) if size < piece_cap]
        if not candidates:
            return []
        idx = random.choice(candidates)
        pieces[idx] += 1
        remaining -= 1
    random.shuffle(pieces)
    return pieces


def _grow_fragment(
    start: tuple[int, int], size: int, blocked: set[tuple[int, int]]
) -> set[tuple[int, int]] | None:
    fragment = {start}
    while len(fragment) < size:
        frontier: set[tuple[int, int]] = set()
        for r, c in fragment:
            for dr, dc in _DIRS8:
                nr, nc = r + dr, c + dc
                cell = (int(nr), int(nc))
                if cell in fragment:
                    continue
                if max(abs(cell[0]), abs(cell[1])) > LOCAL_RADIUS:
                    continue
                if not _far_from_cells(cell, blocked):
                    continue
                frontier.add(cell)
        if not frontier:
            return None
        fragment.add(random.choice(sorted(frontier)))
    return fragment


def _make_fragmented_stencil(total: int) -> tuple[list[tuple[int, int]], int]:
    for _ in range(FRAGMENT_RETRIES):
        n_fragments = random.randint(FRAGMENT_COUNT_MIN, min(FRAGMENT_COUNT_MAX, total))
        sizes = _random_partition(total, n_fragments, FRAGMENT_SIZE_CAP)
        if not sizes:
            continue

        first = _grow_fragment((0, 0), sizes[0], blocked=set())
        if first is None:
            continue
        fragments = [first]
        used = set(first)

        ok = True
        for size in sizes[1:]:
            starts = [cell for cell in _LOCAL_CELLS if cell not in used and _far_from_cells(cell, used)]
            if not starts:
                ok = False
                break
            start = random.choice(starts)
            fragment = _grow_fragment(start, size, blocked=used)
            if fragment is None:
                ok = False
                break
            fragments.append(fragment)
            used.update(fragment)
        if not ok:
            continue

        coords = sorted(used)
        return coords, len(fragments)

    raise RuntimeError("failed to generate fragmented stencil")


def _build_rules(num_rules: int, area: int):
    colors = random.sample(NON_BG_NON_TEAL_COLORS, 2 * num_rules + 1)
    random.shuffle(colors)
    source_colors = colors[:num_rules]
    target_colors = colors[num_rules : 2 * num_rules]
    legend_source_color = int(colors[2 * num_rules])

    max_s = min(STENCIL_TOTAL_MAX_CAP, max(STENCIL_TOTAL_MIN + num_rules - 1, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(STENCIL_TOTAL_MIN, area // STENCIL_MIN_AREA_DIV), max_s - (num_rules - 1))
    size_candidates = list(range(min_s, max_s + 1))
    if len(size_candidates) < num_rules:
        size_candidates = list(range(STENCIL_TOTAL_MIN, STENCIL_TOTAL_MAX_CAP + 1))
    stencil_sizes = sorted(random.sample(size_candidates, num_rules), reverse=True)

    rules = []
    for i in range(num_rules):
        coords, fragment_count = _make_fragmented_stencil(stencil_sizes[i])
        rules.append(
            {
                "legend_src": int(legend_source_color),
                "tgt": int(target_colors[i]),
                "rel": [(int(r), int(c)) for r, c in coords],
                "pattern_size": int(len(coords)),
                "fragment_count": int(fragment_count),
            }
        )

    max_sc = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN + num_rules - 1, area // SCATTERED_AREA_DIV))
    count_choices = list(range(SCATTERED_MIN, max_sc + 1))
    if len(count_choices) < num_rules:
        count_choices = list(range(SCATTERED_MIN, SCATTERED_MAX_CAP + 1))
    chosen_counts = sorted(random.sample(count_choices, num_rules), reverse=True)
    source_specs = [
        {"color": int(source_colors[i]), "count": int(chosen_counts[i])}
        for i in range(num_rules)
    ]

    used_colors = set(source_colors) | set(target_colors)
    return rules, source_specs, used_colors


def _place_legends(rules, inp, occupied, reserved) -> bool:
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
                teal_cells.add((int(nr), int(nc)))
            if not ok or (ar, ac) not in teal_cells:
                continue
            if (sr, sc) in teal_cells or (tr, tc) in teal_cells:
                continue

            own_cells = {(sr, sc), (tr, tc)} | teal_cells
            window_cells = {
                (rr, cc)
                for rr in range(ar - LOCAL_RADIUS, ar + LOCAL_RADIUS + 1)
                for cc in range(ac - LOCAL_RADIUS, ac + LOCAL_RADIUS + 1)
            }
            if own_cells & (occupied | reserved):
                continue
            if (window_cells - own_cells) & occupied:
                continue

            inp[sr, sc] = int(rule["legend_src"])
            inp[tr, tc] = int(rule["tgt"])
            for nr, nc in teal_cells:
                inp[nr, nc] = 8
            occupied.update(own_cells)
            _reserve_4neighbors(reserved, own_cells)
            _reserve_window(reserved, (ar, ac), LOCAL_RADIUS)
            rule["source_pos"] = (int(sr), int(sc))
            rule["target_pos"] = (int(tr), int(tc))
            rule["anchor"] = (int(ar), int(ac))
            placed = True
            break
        if not placed:
            return False
    return True


def _place_scattered(source_specs, inp, occupied, reserved) -> bool:
    h, w = inp.shape
    for spec in source_specs:
        positions = []
        for _ in range(spec["count"] * SCATTERED_RETRIES_PER_PIXEL):
            if len(positions) >= spec["count"]:
                break
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            positions.append((int(r), int(c)))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(r, c)])
        if len(positions) < spec["count"]:
            return False
        spec["positions"] = positions
        for r, c in positions:
            inp[r, c] = int(spec["color"])
    return True


def _place_bystanders(used_colors, source_specs, inp, occupied, reserved):
    available = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not available:
        return []
    h, w = inp.shape
    source_min = min(int(spec["count"]) for spec in source_specs)
    n = random.randint(0, min(BYSTANDER_MAX, len(available)))
    placed = []
    for color in random.sample(available, n):
        freq = random.randint(1, min(BYSTANDER_FREQ_MAX, source_min - 1))
        positions = []
        for _ in range(freq * BYSTANDER_RETRIES):
            if len(positions) >= freq:
                break
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            positions.append((int(r), int(c)))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(r, c)])
            inp[r, c] = int(color)
        if len(positions) != freq:
            return None
        placed.append({"color": int(color), "positions": positions})
    return placed


def _build_output(discovered_rules, source_specs, bystanders, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for item in bystanders:
        for r, c in item["positions"]:
            out[int(r), int(c)] = int(item["color"])

    ordered_rules = sorted(discovered_rules, key=lambda rule: -int(rule["pattern_size"]))
    ordered_sources = sorted(source_specs, key=lambda spec: -int(spec["count"]))
    for rule, spec in zip(ordered_rules, ordered_sources):
        for sr, sc in spec["positions"]:
            ar, ac = rule["anchor"]
            for pr, pc in rule["pattern"]:
                nr = int(sr + int(pr) - int(ar))
                nc = int(sc + int(pc) - int(ac))
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = int(rule["target_color"])
    return out


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(min_dim, GRID_DIM_MAX)

    rules, source_specs, used_colors = _build_rules(num_rules, h * w)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    inp = np.zeros((h, w), dtype=int)

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    if not _place_scattered(source_specs, inp, occupied, reserved):
        return None
    bystanders = _place_bystanders(used_colors, source_specs, inp, occupied, reserved)
    if bystanders is None:
        return None

    discovered_rules = _discover_rules(inp, bg=get_bg(inp))
    if len(discovered_rules) != num_rules:
        return None
    if len({int(rule["pattern_size"]) for rule in discovered_rules}) != num_rules:
        return None
    if any(len(rule["component_ids"]) < 2 for rule in discovered_rules):
        return None

    output = _build_output(discovered_rules, source_specs, bystanders, h, w)

    try:
        if not np.array_equal(main(inp), output):
            return None
    except Exception:
        return None

    counts = Counter(int(spec["count"]) for spec in source_specs)
    if len(counts) != num_rules:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
