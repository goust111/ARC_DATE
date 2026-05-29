# Combo variant of abc82100 (index 20)
# combo_spec: D9/2+D12/2
# Name: 规则链式应用+距离排序处理

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the fused variant:
# a) Behavior trace:
#    Original abc82100 applies every discovered rule independently to the input's
#    scattered source pixels. Here the legends are linked into one color chain
#    (c0->c1, c1->c2, ...), so step i consumes the current ci pixels and creates
#    ci+1 pixels for the next step. Inside one step, sources are not treated as
#    a fixed batch: we sort them by distance to the grid center, stamp nearest
#    first, and skip any farther source that has already been overwritten.
# b) Complexity calibration:
#    To keep the chain visible but not chaotic, the generator uses 2-3 rules,
#    compact 2-4 cell stencils inside a 3x3 neighborhood, and only a few seed
#    pixels for the first color. This keeps the chained growth readable.
# c) Broken original assumptions:
#    The original solver assumes "all source colors in the input are terminal
#    inputs" and "rule order is irrelevant". Both fail now: later rule sources
#    appear only after earlier steps, and step-local order can suppress farther
#    sources. The new solver therefore removes legends first, derives a unique
#    color-chain order, and mutates one working grid in place.
# d) Placement legality:
#    Legends are spaced in a dedicated top band, scattered seeds are isolated
#    from each other and from legends by 4-neighbor reservation, and all placed
#    cells stay in bounds. The generator validates every sample with `main`.
# e) Visual observability:
#    The chain is human-visible because legend target colors feed directly into
#    the next legend's source color, and the legends themselves are laid out in
#    chain order. The distance rule is visible because the generator guarantees
#    at least one nearer source whose stamp covers a farther same-colored source.
# f) Visual cleanliness:
#    Small stencils, moderate grid sizes, a legend band, and few bystanders keep
#    the dynamic area separated from the explanatory examples.
# g) Inferability:
#    Only a short color chain is used, with repeated evidence across multiple
#    legend pairs and a deterministic geometric ordering based on distance to the
#    center. No hidden ordering based on raw color ids is used.


GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {2: 12, 3: 15}
RULE_COUNT_WEIGHTS = [2, 2, 2, 3, 3]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
PATTERN_COLOR = 8
PATTERN_SIZE_CHOICES = [2, 3, 4]
FIRST_RULE_SEEDS = (3, 4)
BYSTANDER_MAX = 2
LEGEND_GAP = 2
OUTER_RETRIES = 400
PLACEMENT_RETRIES = 300

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


def _center_dist2(pos: tuple[int, int], shape: tuple[int, int]) -> int:
    r, c = map(int, pos)
    h, w = map(int, shape)
    dr = 2 * int(r) - (int(h) - 1)
    dc = 2 * int(c) - (int(w) - 1)
    return int(dr * dr + dc * dc)


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells: set[tuple[int, int]] | list[tuple[int, int]]) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _discover_rules(grid: np.ndarray, bg: int, pattern_color: int = PATTERN_COLOR) -> list[dict]:
    h, w = grid.shape
    patterns = components_mask(grid == int(pattern_color), connectivity=8)
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


def _order_rule_chain(rules: list[dict]) -> list[dict]:
    if not rules:
        return []

    by_source: dict[int, dict] = {}
    incoming: dict[int, int] = {}
    for rule in rules:
        src = int(rule["source_color"])
        tgt = int(rule["target_color"])
        by_source[src] = rule
        incoming[tgt] = incoming.get(tgt, 0) + 1

    starts = [rule for rule in rules if int(rule["source_color"]) not in incoming]
    starts.sort(key=lambda rule: (int(rule["source_pos"][0]), int(rule["source_pos"][1])))
    if not starts:
        starts = sorted(rules, key=lambda rule: (int(rule["source_pos"][0]), int(rule["source_pos"][1])))

    ordered: list[dict] = []
    used: set[int] = set()
    for start in starts:
        cur = start
        while id(cur) not in used:
            ordered.append(cur)
            used.add(id(cur))
            nxt = by_source.get(int(cur["target_color"]))
            if nxt is None:
                break
            cur = nxt

    for rule in sorted(rules, key=lambda item: (int(item["source_pos"][0]), int(item["source_pos"][1]))):
        if id(rule) not in used:
            ordered.append(rule)
            used.add(id(rule))
    return ordered


def _prepare_work_grid(grid: np.ndarray, rules: list[dict], bg: int, pattern_color: int = PATTERN_COLOR) -> np.ndarray:
    work = np.asarray(grid, dtype=int).copy()
    work[work == int(pattern_color)] = int(bg)
    for rule in rules:
        sr, sc = map(int, rule["source_pos"])
        tr, tc = map(int, rule["target_pos"])
        work[sr, sc] = int(bg)
        work[tr, tc] = int(bg)
    return work


def _stamp(work: np.ndarray, pattern: np.ndarray, anchor: tuple[int, int], target_color: int, pos: tuple[int, int]) -> int:
    h, w = work.shape
    ar, ac = map(int, anchor)
    sr, sc = map(int, pos)
    dr = int(sr) - int(ar)
    dc = int(sc) - int(ac)
    rows = pattern[:, 0] + dr
    cols = pattern[:, 1] + dc
    ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
    work[rows[ok], cols[ok]] = int(target_color)
    return int(ok.sum())


def _simulate_chain(grid: np.ndarray, rules: list[dict], *, bg: int) -> tuple[np.ndarray, dict]:
    work = _prepare_work_grid(grid, rules, bg=bg)
    h, w = work.shape
    step_stats: list[dict] = []

    for step_index, rule in enumerate(rules):
        source_color = int(rule["source_color"])
        positions = [tuple(map(int, rc)) for rc in np.argwhere(work == int(source_color))]
        positions.sort(key=lambda rc: (_center_dist2(rc, (h, w)), int(rc[0]), int(rc[1])))

        processed = 0
        skipped = 0
        stamped_cells = 0
        for sr, sc in positions:
            if int(work[int(sr), int(sc)]) != int(source_color):
                skipped += 1
                continue
            processed += 1
            work[int(sr), int(sc)] = int(bg)
            stamped_cells += _stamp(work, rule["pattern"], rule["anchor"], int(rule["target_color"]), (int(sr), int(sc)))

        step_stats.append(
            {
                "step": int(step_index),
                "source_color": int(source_color),
                "target_color": int(rule["target_color"]),
                "seen": int(len(positions)),
                "processed": int(processed),
                "skipped": int(skipped),
                "stamped_cells": int(stamped_cells),
            }
        )

    metrics = {
        "step_stats": step_stats,
        "active_steps": int(sum(1 for stat in step_stats if int(stat["processed"]) > 0)),
        "skipped_total": int(sum(int(stat["skipped"]) for stat in step_stats)),
    }
    return work, metrics


def _simulate_parallel_baseline(grid: np.ndarray, rules: list[dict], *, bg: int) -> np.ndarray:
    work = _prepare_work_grid(grid, rules, bg=bg)
    base = work.copy()
    out = work.copy()
    for rule in rules:
        src = int(rule["source_color"])
        positions = [tuple(map(int, rc)) for rc in np.argwhere(base == int(src))]
        for sr, sc in positions:
            out[int(sr), int(sc)] = int(bg)
            _stamp(out, rule["pattern"], rule["anchor"], int(rule["target_color"]), (int(sr), int(sc)))
    return out


def behavior_metrics(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    rules = _order_rule_chain(_discover_rules(g, bg=bg, pattern_color=PATTERN_COLOR))
    fused, fused_stats = _simulate_chain(g, rules, bg=bg)
    baseline = _simulate_parallel_baseline(g, rules, bg=bg)
    fused_stats["parallel_diff_cells"] = int(np.count_nonzero(fused != baseline))
    fused_stats["rule_count"] = int(len(rules))
    return fused_stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply linked legend rules as a sequential stencil pipeline with center-first source handling.

    Input structure:
    - Black background with several local source->target examples, each followed by a nearby teal stencil.
    - The examples form one visible color chain because each legend's target color is the next legend's source color.
    - Away from the legends are isolated seed pixels of the first chain color, plus optional unrelated bystanders.

    Rule:
    - Erase every legend pixel and every teal stencil from the working grid, then execute the discovered rules in chain order.
    - In each step, look at the current pixels of that step's source color, sort them by distance to the grid center (nearest first), and process them one by one.
    - A source still present at its turn stamps the step's teal stencil onto itself, recolored to the target color; if an earlier stamp already covered a later source, the later one is skipped.

    Key insight:
    - One layer of order comes from the color chain between legends, while the second layer comes from center-distance ordering within each step.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    rules = _order_rule_chain(_discover_rules(g, bg=bg, pattern_color=PATTERN_COLOR))
    out, _stats = _simulate_chain(g, rules, bg=bg)
    return out


def _make_small_stencil(require_diagonal: bool) -> list[tuple[int, int]]:
    allowed = {(int(r), int(c)) for r in (-1, 0, 1) for c in (0, 1)}
    allowed.remove((0, 0))
    diagonal_allowed = [(-1, 1), (1, 1)]

    while True:
        size = random.choice(PATTERN_SIZE_CHOICES)
        coords = {(0, 0)}
        while len(coords) < size:
            frontier: set[tuple[int, int]] = set()
            for r, c in coords:
                for dr, dc in _DIRS8:
                    cand = (int(r) + dr, int(c) + dc)
                    if cand in allowed and cand not in coords:
                        frontier.add(cand)
            if not frontier:
                break
            if require_diagonal and len(coords) == size - 1 and not any(cell in coords for cell in diagonal_allowed):
                diag_frontier = [cell for cell in sorted(frontier) if cell in diagonal_allowed]
                if diag_frontier:
                    frontier = set(diag_frontier)
            coords.add(random.choice(sorted(frontier)))

        if len(coords) != size:
            continue
        if require_diagonal and not any(cell in coords for cell in diagonal_allowed):
            continue
        return sorted((int(r), int(c)) for r, c in coords)


def _build_rules(num_rules: int) -> tuple[list[dict], set[int]]:
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)
    chain = colors[: num_rules + 1]

    rules: list[dict] = []
    used_colors = set(chain)
    for i in range(num_rules):
        rel = _make_small_stencil(require_diagonal=(i == 0))
        rules.append(
            {
                "src": int(chain[i]),
                "tgt": int(chain[i + 1]),
                "rel": rel,
            }
        )
    return rules, used_colors


def _legend_bbox(rule: dict) -> tuple[int, int, int, int]:
    cols = [0, 1] + [2 + int(cc) for _rr, cc in rule["rel"]]
    rows = [0] + [int(rr) for rr, _cc in rule["rel"]]
    return int(min(rows)), int(min(cols)), int(max(rows)), int(max(cols))


def _place_legends(rules: list[dict], inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]) -> int | None:
    h, w = inp.shape
    row_shift = 1 - min(int(_legend_bbox(rule)[0]) for rule in rules)
    legend_bottom = row_shift + max(int(_legend_bbox(rule)[2]) for rule in rules)
    cursor = 1

    for rule in rules:
        min_r, min_c, max_r, max_c = _legend_bbox(rule)
        height = int(max_r - min_r + 1)
        width = int(max_c - min_c + 1)
        source_r = int(row_shift)
        source_c = int(cursor - min_c)
        if source_r + min_r < 0 or source_r + max_r >= h:
            return None
        if source_c + min_c < 0 or source_c + max_c >= w:
            return None

        tr, tc = int(source_r), int(source_c + 1)
        ar, ac = int(source_r), int(source_c + 2)
        teal_cells = {(int(ar + rr), int(ac + cc)) for rr, cc in rule["rel"]}
        cells = {(int(source_r), int(source_c)), (int(tr), int(tc))} | teal_cells
        if cells & occupied or cells & reserved:
            return None

        inp[int(source_r), int(source_c)] = int(rule["src"])
        inp[int(tr), int(tc)] = int(rule["tgt"])
        for rr, cc in teal_cells:
            inp[int(rr), int(cc)] = int(PATTERN_COLOR)

        occupied.update(cells)
        _reserve_4neighbors(reserved, cells)
        rule["legend_source"] = (int(source_r), int(source_c))
        rule["legend_target"] = (int(tr), int(tc))
        cursor += int(width) + LEGEND_GAP

    return int(legend_bottom)


def _all_translated_in_bounds(rel: list[tuple[int, int]], pos: tuple[int, int], shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    pr, pc = map(int, pos)
    for rr, cc in rel:
        nr = int(pr) + int(rr)
        nc = int(pc) + int(cc)
        if not (0 <= nr < h and 0 <= nc < w):
            return False
    return True


def _find_cover_pair(
    rel: list[tuple[int, int]],
    shape: tuple[int, int],
    min_row: int,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = map(int, shape)
    deltas = [p for p in rel if p != (0, 0) and p not in _DIRS4]
    deltas.sort(key=lambda p: (-abs(int(p[0])) - abs(int(p[1])), int(p[0]), int(p[1])))
    cells = [(r, c) for r in range(int(min_row), h) for c in range(w)]
    cells.sort(key=lambda rc: (_center_dist2(rc, (h, w)), int(rc[0]), int(rc[1])))

    for dr, dc in deltas:
        for sr, sc in cells:
            fr, fc = int(sr + dr), int(sc + dc)
            near = (int(sr), int(sc))
            far = (int(fr), int(fc))
            if not (0 <= fr < h and 0 <= fc < w):
                continue
            if near in occupied or near in reserved or far in occupied or far in reserved:
                continue
            if not _all_translated_in_bounds(rel, near, (h, w)):
                continue
            if _center_dist2(near, (h, w)) >= _center_dist2(far, (h, w)):
                continue
            return near, far
    return None


def _place_first_rule_seeds(
    rules: list[dict],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    min_row: int,
) -> bool:
    h, w = inp.shape
    first = rules[0]
    pair = _find_cover_pair(first["rel"], (h, w), min_row, occupied, reserved)
    if pair is None:
        return False

    desired = random.randint(FIRST_RULE_SEEDS[0], FIRST_RULE_SEEDS[1])
    positions = [pair[0], pair[1]]
    used_dist = {_center_dist2(pair[0], (h, w)), _center_dist2(pair[1], (h, w))}
    occupied.update(positions)
    _reserve_4neighbors(reserved, positions)

    tries = 0
    while len(positions) < desired and tries < PLACEMENT_RETRIES:
        tries += 1
        r = random.randint(int(min_row), h - 1)
        c = random.randint(0, w - 1)
        pos = (int(r), int(c))
        if pos in occupied or pos in reserved:
            continue
        dist = _center_dist2(pos, (h, w))
        if dist in used_dist:
            continue
        if not _all_translated_in_bounds(first["rel"], pos, (h, w)):
            continue
        positions.append(pos)
        used_dist.add(dist)
        occupied.add(pos)
        _reserve_4neighbors(reserved, [pos])

    if len(positions) < FIRST_RULE_SEEDS[0]:
        return False

    first["scattered"] = positions
    for r, c in positions:
        inp[int(r), int(c)] = int(first["src"])
    return True


def _place_bystanders(used_colors: set[int], inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]], min_row: int) -> list[tuple[int, int, int]]:
    h, w = inp.shape
    colors = [c for c in NON_BG_NON_TEAL_COLORS if int(c) not in used_colors]
    if not colors:
        return []

    count = random.randint(0, BYSTANDER_MAX)
    placed: list[tuple[int, int, int]] = []
    for _ in range(count):
        for _try in range(PLACEMENT_RETRIES):
            r = random.randint(int(min_row), h - 1)
            c = random.randint(0, w - 1)
            pos = (int(r), int(c))
            if pos in occupied or pos in reserved:
                continue
            color = random.choice(colors)
            inp[int(r), int(c)] = int(color)
            placed.append((int(r), int(c), int(color)))
            occupied.add(pos)
            _reserve_4neighbors(reserved, [pos])
            break
    return placed


def _gen_attempt() -> np.ndarray | None:
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    h = random.randint(MIN_DIM_BY_RULES[num_rules], GRID_DIM_MAX)
    w = random.randint(MIN_DIM_BY_RULES[num_rules], GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules)
    inp = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()

    legend_bottom = _place_legends(rules, inp, occupied, reserved)
    if legend_bottom is None:
        return None

    min_row = int(max(legend_bottom + 3, h // 2))
    if min_row >= h:
        return None

    if not _place_first_rule_seeds(rules, inp, occupied, reserved, min_row):
        return None
    _place_bystanders(used_colors, inp, occupied, reserved, min_row)

    try:
        out = main(inp)
    except Exception:
        return None

    metrics = behavior_metrics(inp)
    if int(metrics["rule_count"]) != int(num_rules):
        return None
    if int(metrics["active_steps"]) != int(num_rules):
        return None
    if int(metrics["skipped_total"]) < 1:
        return None
    if int(metrics["parallel_diff_cells"]) < 1:
        return None
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
