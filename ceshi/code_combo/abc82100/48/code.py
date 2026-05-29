# Combo variant of abc82100 (index 48)
# combo_spec: D10/2+D16/0
# Name: 远距离颜色对+局部规则-源配对

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, get_bg

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    Original abc82100 used a global color dictionary. A compact legend
#    `src,tgt,anchor+teal` taught "every other pixel of color src becomes this
#    teal stencil recolored to tgt".
#    This combo changes two linked stages at once:
#    - D16/0: the legend color pair is no longer adjacent. It is a cardinal
#      ray `S . . T A` or `S . . . T A`, so source and target are separated by
#      a fixed span of 3 or 4 cells for the whole grid.
#    - D10/2: each legend only applies to one nearby execution source of the
#      same color, placed at Chebyshev distance 2 or 3 from the legend source.
#    Small example:
#      input cluster: legend source S at (5,5), target T at (5,8), anchor A at
#      (5,9), teal stencil touching A, and a second S at (7,4).
#      output cluster: erase the legend and teal cells, then translate the teal
#      stencil so that A lands on (7,4) and recolor it to T's color.
#    Difference from the original:
#    - the solver must first detect the non-adjacent source-target relation;
#    - then it must pair each legend with a local execution source instead of
#      applying the rule to every pixel of that source color globally.
# b) Complexity calibration:
#    - One span is shared by the whole grid and chosen from {3, 4}. This keeps
#      the long-distance cue obvious without exploding the search space.
#    - Each grid has 2-5 local clusters, and at least one source color is reused
#      across multiple clusters with different targets. This blocks a degenerate
#      global-color interpretation.
#    - Bystanders use colors outside all legend source/target colors so they do
#      not become accidental execution candidates.
# c) Implicit-assumption review:
#    - Original `_discover_rules` assumed `target` was a 4-neighbor; here it
#      must scan a small set of longer spans and select the dominant span.
#    - Original `main()` assumed source color uniquely identified a rule; that
#      is false here because the same source color intentionally appears in
#      multiple nearby clusters with different targets.
#    - Original `main()` deleted every non-legend pixel whose color matched a
#      learned source color; that would wrongly erase execution pixels belonging
#      to other local legends. The combo solver therefore erases only execution
#      pixels that were explicitly assigned to a nearby legend.
#    - The new assignment rule assumes each execution source is the unique
#      nearest same-colored candidate within distance {2,3} of its own legend.
#      The generator enforces this by spacing footprints apart and checking
#      same-color clusters against each other.
# d) Placement legality:
#    - Every legend uses `target = source + span*dir` and `anchor = target + dir`
#      with all cells in bounds.
#    - The execution source is never on the legend ray, stays at Chebyshev
#      distance 2 or 3 from the legend source, and stays at least 2 cells away
#      from the legend pair and teal stencil.
#    - Future colored placements avoid reserved cardinal rays around all colored
#      cells, preventing accidental extra long-distance legends.
# e) Visual observability:
#    - Reused source colors with different nearby targets make the local-pairing
#      rule visible.
#    - Repeated long gaps between source and target make the non-adjacent legend
#      relation visible.
#    - Both cues can be inferred from 2-4 examples without hidden color ordering.
# f) Visual cleanliness:
#    - Cluster footprints include legend, teal stencil, execution source, and
#      final stamp area, then receive a one-cell moat. This keeps clusters from
#      visually merging.
#    - Output stamps are kept in bounds and non-overlapping for readability.
# g) Anti-degeneration target:
#    - Generated samples should always have at least one repeated source color
#      and all discovered legends should use the same span in {3,4}. Under those
#      conditions, the original adjacent/global solver should usually disagree.

PAIR_SPAN_OPTIONS = (3, 4)
LOCAL_SOURCE_DISTS = {2, 3}

# Grid dimensions
GRID_DIM_MAX = 20
MIN_DIM_BY_GROUPS = {2: 9, 3: 12, 4: 15, 5: 18}

# Local cluster counts
GROUP_COUNT_WEIGHTS = [2, 2, 3, 3, 4, 4, 5]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

# Stencil sizing
STENCIL_MAX_CAP = 8
STENCIL_MIN_AREA_DIV = 120
STENCIL_MAX_AREA_DIV = 40

# Bystanders
BYSTANDER_MAX = 4

# Retry budgets
CLUSTER_RETRIES = 900
BYSTANDER_RETRIES = 100
OUTER_RETRIES = 500

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
_LOCAL_OFFSETS = [
    (-3, -1),
    (-3, 0),
    (-3, 1),
    (-2, -2),
    (-2, -1),
    (-2, 1),
    (-2, 2),
    (-1, -3),
    (-1, -2),
    (-1, 2),
    (-1, 3),
    (1, -3),
    (1, -2),
    (1, 2),
    (1, 3),
    (2, -2),
    (2, -1),
    (2, 1),
    (2, 2),
    (3, -1),
    (3, 0),
    (3, 1),
]


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _expand_with_moat(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    expanded = set(cells)
    for r, c in list(cells):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                expanded.add((int(r) + dr, int(c) + dc))
    return expanded


def _reserve_color_rays(
    reserved: set[tuple[int, int]], cells, max_span: int = max(PAIR_SPAN_OPTIONS)
) -> None:
    for r, c in cells:
        r = int(r)
        c = int(c)
        for dr, dc in _DIRS4:
            for step in range(1, max_span + 1):
                reserved.add((r + step * dr, c + step * dc))


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


def _allocate_counts(num_groups: int, num_families: int) -> list[int]:
    counts = [1] * num_families
    for _ in range(num_groups - num_families):
        counts[random.randrange(num_families)] += 1
    random.shuffle(counts)
    return counts


def _offsets_for_direction(dr: int, dc: int) -> list[tuple[int, int]]:
    if dr == 0:
        return [offset for offset in _LOCAL_OFFSETS if offset[0] != 0]
    return [offset for offset in _LOCAL_OFFSETS if offset[1] != 0]


def _min_distance_to_cells(cell: tuple[int, int], others: set[tuple[int, int]]) -> int:
    return min(_chebyshev(cell, other) for other in others)


def _discover_legends(grid: np.ndarray, bg: int, pattern_color: int = 8) -> list[dict]:
    h, w = grid.shape
    patterns = components_mask(grid == pattern_color, connectivity=8)
    candidates_by_span: dict[int, list[dict]] = {span: [] for span in PAIR_SPAN_OPTIONS}
    seen_by_span: dict[int, set[tuple[tuple[int, int], tuple[int, int], int]]] = {
        span: set() for span in PAIR_SPAN_OPTIONS
    }

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color in (int(bg), int(pattern_color)):
                continue

            for dr, dc in _DIRS4:
                for span in PAIR_SPAN_OPTIONS:
                    tr = r + span * dr
                    tc = c + span * dc
                    ar = tr + dr
                    ac = tc + dc
                    if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
                        continue

                    target_color = int(grid[tr, tc])
                    if target_color in (int(bg), int(pattern_color)) or target_color == source_color:
                        continue

                    anchor = (int(ar), int(ac))
                    pattern_hits: list[tuple[int, int, np.ndarray]] = []
                    for i, pattern in enumerate(patterns):
                        dist = _anchor_distance(pattern, anchor)
                        if dist <= 1:
                            pattern_hits.append((int(dist), int(i), pattern))

                    if not pattern_hits:
                        continue

                    pattern_hits.sort(key=lambda item: (item[0], len(item[2])))
                    _dist, pattern_index, pattern = pattern_hits[0]
                    key = ((int(r), int(c)), (int(tr), int(tc)), int(pattern_index))
                    if key in seen_by_span[span]:
                        continue
                    seen_by_span[span].add(key)
                    candidates_by_span[span].append(
                        {
                            "source_color": int(source_color),
                            "target_color": int(target_color),
                            "source_pos": (int(r), int(c)),
                            "target_pos": (int(tr), int(tc)),
                            "anchor": anchor,
                            "pattern": np.asarray(pattern, dtype=int),
                            "span": int(span),
                        }
                    )

    best_span = max(PAIR_SPAN_OPTIONS, key=lambda span: len(candidates_by_span[span]))
    legends = candidates_by_span[best_span]
    legends.sort(key=lambda item: (item["source_pos"], item["target_pos"], item["target_color"]))
    return legends


def _assign_local_sources(
    grid: np.ndarray, legends: list[dict], bg: int, pattern_color: int = 8
) -> tuple[dict[int, tuple[int, int]], set[tuple[int, int]]]:
    legend_pixels: set[tuple[int, int]] = set()
    legends_by_color: dict[int, list[int]] = {}
    for i, legend in enumerate(legends):
        legend_pixels.add(legend["source_pos"])
        legend_pixels.add(legend["target_pos"])
        legends_by_color.setdefault(int(legend["source_color"]), []).append(i)

    positions_by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c in map(tuple, np.argwhere((grid != int(bg)) & (grid != int(pattern_color)))):
        pos = (int(r), int(c))
        if pos in legend_pixels:
            continue
        color = int(grid[pos])
        positions_by_color.setdefault(color, []).append(pos)

    assignments: dict[int, tuple[int, int]] = {}
    used_positions: set[tuple[int, int]] = set()
    for i, legend in enumerate(legends):
        source_color = int(legend["source_color"])
        instruction_cells = {
            legend["source_pos"],
            legend["target_pos"],
            legend["anchor"],
        } | {tuple(map(int, cell)) for cell in legend["pattern"]}
        candidates: list[tuple[int, int, int, int]] = []
        for pos in positions_by_color.get(source_color, []):
            if pos in used_positions:
                continue
            dist = _chebyshev(pos, legend["source_pos"])
            if dist not in LOCAL_SOURCE_DISTS:
                continue
            if _min_distance_to_cells(pos, instruction_cells) < 2:
                continue
            peer_dists = [_chebyshev(pos, legends[j]["source_pos"]) for j in legends_by_color[source_color]]
            if dist != min(peer_dists) or peer_dists.count(dist) != 1:
                continue
            candidates.append(
                (
                    dist,
                    abs(pos[0] - legend["anchor"][0]) + abs(pos[1] - legend["anchor"][1]),
                    pos[0],
                    pos[1],
                )
            )

        if candidates:
            candidates.sort()
            chosen = (int(candidates[0][2]), int(candidates[0][3]))
            assignments[i] = chosen
            used_positions.add(chosen)

    return assignments, legend_pixels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use nearby execution pixels and long-gap legends to place recolored stencils.

    Input structure:
    - Black background with several compact clusters plus optional bystanders.
    - In each cluster, a legend source pixel and legend target pixel lie on the
      same row or column but are separated by 3 or 4 cells; one more step in
      that direction reaches an anchor touching a teal (8) stencil component.
    - A second pixel of the legend source color sits 2-3 cells away from the
      legend source and acts as the local execution site for that legend only.

    Rule:
    - Discover the dominant long-distance source-to-target span, then treat
      each such source-target-anchor+teal configuration as one local legend.
    - For each legend, find the nearby same-colored execution pixel that is
      uniquely closest to that legend among legends of that same source color.
    - Translate the legend's teal stencil so its anchor lands on the execution
      pixel, recolor the stencil to the legend target color, and erase legend
      pixels, teal pixels, and the assigned execution pixels.

    Key insight: Color alone is insufficient now. The solver must combine a
    repeated non-adjacent source-target spacing with local "instruction nearby
    execution source" pairing to decide which stencil goes where.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8
    h, w = g.shape

    legends = _discover_legends(g, bg=bg, pattern_color=pattern_color)
    assignments, legend_pixels = _assign_local_sources(g, legends, bg=bg, pattern_color=pattern_color)
    execution_pixels = set(assignments.values())

    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        pos = (int(r), int(c))
        if pos in legend_pixels or pos in execution_pixels:
            continue
        out[pos] = int(g[pos])

    for i, legend in enumerate(legends):
        if i not in assignments:
            continue
        sr, sc = assignments[i]
        ar, ac = legend["anchor"]
        pattern = legend["pattern"]
        dr = sr - ar
        dc = sc - ac
        rows = pattern[:, 0] + dr
        cols = pattern[:, 1] + dc
        ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
        out[rows[ok], cols[ok]] = int(legend["target_color"])

    return out


def _build_groups(num_groups: int, area: int) -> tuple[list[dict], set[int], int]:
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(2, area // STENCIL_MIN_AREA_DIV), max_s)
    pair_span = random.choice(PAIR_SPAN_OPTIONS)

    max_families = min(3, num_groups - 1 if num_groups > 1 else 1)
    num_families = random.randint(1, max_families)
    src_colors = colors[:num_families]
    target_pool = [c for c in NON_BG_NON_TEAL_COLORS if c not in src_colors]
    counts = _allocate_counts(num_groups, num_families)

    groups: list[dict] = []
    used_colors: set[int] = set()
    for src, count in zip(src_colors, counts):
        targets = random.sample(target_pool, count)
        for tgt in targets:
            stencil, anchor = _make_stencil(min_s, max_s)
            rel = [(r - anchor[0], c - anchor[1]) for r, c in stencil]
            groups.append({"src": int(src), "tgt": int(tgt), "rel": rel, "span": int(pair_span)})
            used_colors.add(int(src))
            used_colors.add(int(tgt))

    random.shuffle(groups)
    return groups, used_colors, pair_span


def _same_color_pairing_ok(candidate_group: dict, placed_groups: list[dict]) -> bool:
    src = int(candidate_group["src"])
    own_source = tuple(candidate_group["legend_source"])
    own_exec = tuple(candidate_group["exec"])
    own_dist = _chebyshev(own_exec, own_source)

    for other in placed_groups:
        if int(other["src"]) != src:
            continue
        other_source = tuple(other["legend_source"])
        other_exec = tuple(other["exec"])
        if _chebyshev(own_exec, other_source) <= own_dist:
            return False
        if _chebyshev(other_exec, own_source) <= _chebyshev(other_exec, other_source):
            return False
    return True


def _place_clusters(groups: list[dict], inp: np.ndarray) -> tuple[bool, set[tuple[int, int]]]:
    h, w = inp.shape
    blocked: set[tuple[int, int]] = set()
    occupied_output: set[tuple[int, int]] = set()
    placed_groups: list[dict] = []

    for group in groups:
        rel = group["rel"]
        span = int(group["span"])
        placed = False
        for _ in range(CLUSTER_RETRIES):
            dr, dc = random.choice(_DIRS4)
            off_r, off_c = random.choice(_offsets_for_direction(dr, dc))
            sr, sc = random.randint(0, h - 1), random.randint(0, w - 1)
            tr, tc = sr + span * dr, sc + span * dc
            ar, ac = tr + dr, tc + dc
            rr, rc = sr - dr, sc - dc
            er, ec = sr + off_r, sc + off_c
            if not (
                0 <= tr < h
                and 0 <= tc < w
                and 0 <= ar < h
                and 0 <= ac < w
                and 0 <= er < h
                and 0 <= ec < w
            ):
                continue

            legend_cells = {(sr, sc), (tr, tc)}
            teal_cells: set[tuple[int, int]] = set()
            stamp_cells: set[tuple[int, int]] = set()
            ok = True
            for rr, cc in rel:
                teal = (ar + rr, ac + cc)
                stamp = (er + rr, ec + cc)
                if not (
                    0 <= teal[0] < h
                    and 0 <= teal[1] < w
                    and 0 <= stamp[0] < h
                    and 0 <= stamp[1] < w
                ):
                    ok = False
                    break
                teal_cells.add(teal)
                stamp_cells.add(stamp)
            if not ok:
                continue

            instruction_cells = legend_cells | teal_cells | {(ar, ac)}
            if _chebyshev((er, ec), (sr, sc)) not in LOCAL_SOURCE_DISTS:
                continue
            if _min_distance_to_cells((er, ec), instruction_cells) < 2:
                continue
            if _min_distance_to_cells((rr, rc), teal_cells) <= 1:
                continue
            if stamp_cells & instruction_cells:
                continue

            # Reserve the cell just behind the legend source too. Otherwise the
            # same long-distance pair can sometimes be read backward if another
            # teal stencil lands within Chebyshev 1 of that reverse anchor.
            cluster_cells = instruction_cells | {(er, ec), (rr, rc)} | stamp_cells
            colored_cells = {(sr, sc), (tr, tc), (er, ec)}
            if cluster_cells & blocked:
                continue
            if stamp_cells & occupied_output:
                continue

            group["legend_source"] = (sr, sc)
            group["legend_target"] = (tr, tc)
            group["anchor"] = (ar, ac)
            group["exec"] = (er, ec)
            group["stamp"] = stamp_cells
            group["footprint"] = cluster_cells
            if not _same_color_pairing_ok(group, placed_groups):
                continue

            inp[sr, sc] = group["src"]
            inp[tr, tc] = group["tgt"]
            inp[er, ec] = group["src"]
            for nr, nc in teal_cells:
                inp[nr, nc] = 8

            blocked.update(_expand_with_moat(cluster_cells))
            _reserve_color_rays(blocked, colored_cells)
            occupied_output.update(stamp_cells)
            placed_groups.append(group)
            placed = True
            break
        if not placed:
            return False, blocked
    return True, blocked


def _place_bystanders(
    used_colors: set[int], inp: np.ndarray, blocked: set[tuple[int, int]]
) -> list[tuple[int, int, int]]:
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []

    h, w = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors)))
    chosen_colors = random.sample(bystander_colors, n) if n else []
    placed: list[tuple[int, int, int]] = []
    for color in chosen_colors:
        for _ in range(BYSTANDER_RETRIES):
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in blocked:
                continue
            inp[r, c] = int(color)
            placed.append((int(r), int(c), int(color)))
            blocked.update(_expand_with_moat({(int(r), int(c))}))
            _reserve_color_rays(blocked, [(int(r), int(c))])
            break
    return placed


def _build_output(
    groups: list[dict], bystanders: list[tuple[int, int, int]], h: int, w: int
) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for r, c, color in bystanders:
        out[r, c] = color
    for group in groups:
        for r, c in group["stamp"]:
            out[r, c] = int(group["tgt"])
    return out


def _gen_attempt() -> np.ndarray | None:
    num_groups = random.choice(GROUP_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_GROUPS[num_groups]
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(min_dim, GRID_DIM_MAX)

    groups, used_colors, _pair_span = _build_groups(num_groups, h * w)
    inp = np.zeros((h, w), dtype=int)
    placed, blocked = _place_clusters(groups, inp)
    if not placed:
        return None

    bystanders = _place_bystanders(used_colors, inp, blocked)
    output = _build_output(groups, bystanders, h, w)
    bg = get_bg(inp)
    legends = _discover_legends(inp, bg=bg)
    assignments, _ = _assign_local_sources(inp, legends, bg=bg)
    src_hist = Counter(int(item["source_color"]) for item in legends)
    if len(legends) != len(groups):
        return None
    if len(assignments) != len(groups):
        return None
    if len({int(item["span"]) for item in legends}) != 1:
        return None
    if not any(count > 1 for count in src_hist.values()):
        return None
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
