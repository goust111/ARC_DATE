# Combo variant of abc82100 (index 40)
# combo_spec: D8/2+D10/2
# Name: 仅输出变换区域+局部规则-源配对

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the fused D8/2 + D10/2 variant
# a) Concrete behavior and trace:
#    Original abc82100 learns a global mapping "source color -> target color +
#    teal stencil" from each legend, then stamps that stencil onto every other
#    pixel of the same source color anywhere in the grid, and finally keeps the
#    full canvas. The fused task changes two stages at once:
#    1. D10/2 localizes rule use. Each legend only applies to one nearby source
#       pixel of the same color, sitting at Chebyshev distance 2 or 3 from that
#       legend. Reused source colors are allowed, so color alone is no longer a
#       sufficient key.
#    2. D8/2 crops the answer. After performing those local stamps and erasing the
#       instruction material, return only the tight bounding box of the remaining
#       transformed area.
#    Example trace: suppose two local clusters both use source color red. One
#    red->yellow legend sits near a red execution pixel and its teal shape; another
#    red->green legend sits near a different red execution pixel. The old solver
#    would stamp both legends onto both red pixels and keep the full board. The
#    fused solver instead pairs each red pixel with its nearby legend only, stamps
#    one yellow shape and one green shape, erases the legends, then crops away the
#    surrounding empty frame.
# b) Complexity calibration:
#    The generator uses 2-5 local clusters and deliberately reuses at least one
#    source color across multiple legends with different targets. That is enough to
#    force local pairing and prevent accidental regression to the original
#    global-by-color behavior. Stencils remain small and non-overlapping so the
#    task stays readable rather than noisy.
# c) Hidden assumptions review for main():
#    - Legend discovery still assumes that every meaningful teal component belongs
#      to a source->target example. The generator places teal only inside legends.
#    - Source-color lookup by itself is no longer valid. main() therefore assigns
#      execution pixels by local geometry, not by "all pixels of that color".
#    - The local assignment assumes a same-color execution pixel is uniquely
#      nearest to its own legend among legends of that same source color. The
#      generator enforces cluster spacing so this uniqueness holds.
#    - Cropping assumes the intended answer is non-empty and lies inside a compact
#      interior region. The generator keeps all stamped output away from the outer
#      border so the crop is meaningful.
# d) Placement legality:
#    Each cluster contains five roles: legend source, legend target, teal stencil,
#    nearby execution source, and the final stamped output footprint. Cluster
#    footprints are kept in bounds, output stamps do not overlap, and a one-cell
#    moat keeps separate clusters from creating accidental extra legends.
# e) Visual observability:
#    Across examples the human sees compact local "instruction + nearby source"
#    clusters in the input, but only the recolored teal shapes remain in the
#    output. Because at least one source color repeats with different nearby target
#    colors, locality is visibly necessary; because the output is cropped, the
#    instruction cells clearly disappear from the returned frame.
# f) Visual cleanliness:
#    All transformed shapes live inside one interior activity region, which makes
#    the crop easy to notice. Legends remain close to their execution sources, but
#    cluster moats prevent the input from degenerating into an unreadable tangle.
# g) Change inferability:
#    No rule depends on hidden color ordering. Every color change is justified by
#    an adjacent visible legend target next to the matching teal stencil, and the
#    repeated-source-color cases make the local matching rule observable from a
#    few training examples.
# Main()/generator assumption audit:
#    main() now relies on three generator guarantees beyond the original task:
#    1. Every intended legend has exactly one claimable nearby execution source of
#       the same color.
#    2. Any repeated source color appears in multiple local clusters whose
#       execution pixels are separated enough for unique nearest-legend matching.
#    3. The union of stamped output cells is strictly smaller than the input frame,
#       so the crop dimension really changes.


_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
LOCAL_SOURCE_DISTS = {2, 3}
_LOCAL_OFFSETS = [
    (-3, -1), (-3, 0), (-3, 1),
    (-2, -2), (-2, -1), (-2, 1), (-2, 2),
    (-1, -3), (-1, -2), (-1, 2), (-1, 3),
    (1, -3), (1, -2), (1, 2), (1, 3),
    (2, -2), (2, -1), (2, 1), (2, 2),
    (3, -1), (3, 0), (3, 1),
]


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _discover_legends(grid: np.ndarray, bg: int, pattern_color: int = 8) -> list[dict]:
    h, w = grid.shape
    patterns = components_mask(grid == pattern_color, connectivity=8)
    legends: list[dict] = []
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
                legends.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "pattern": np.asarray(pattern, dtype=int),
                    }
                )

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
        positions_by_color.setdefault(int(grid[pos]), []).append(pos)

    assignments: dict[int, tuple[int, int]] = {}
    used_positions: set[tuple[int, int]] = set()
    for i, legend in enumerate(legends):
        source_color = int(legend["source_color"])
        candidates: list[tuple[int, int, int, int]] = []
        same_color_legends = legends_by_color[source_color]
        for pos in positions_by_color.get(source_color, []):
            if pos in used_positions:
                continue
            dist = _chebyshev(pos, legend["source_pos"])
            if dist not in LOCAL_SOURCE_DISTS:
                continue
            peer_dists = [_chebyshev(pos, legends[j]["source_pos"]) for j in same_color_legends]
            if dist != min(peer_dists) or peer_dists.count(dist) != 1:
                continue
            candidates.append(
                (
                    dist,
                    abs(int(pos[0]) - int(legend["anchor"][0])) + abs(int(pos[1]) - int(legend["anchor"][1])),
                    int(pos[0]),
                    int(pos[1]),
                )
            )

        if candidates:
            candidates.sort()
            chosen = (int(candidates[0][2]), int(candidates[0][3]))
            assignments[i] = chosen
            used_positions.add(chosen)

    return assignments, legend_pixels


def _crop_to_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != int(bg))
    if coords.size == 0:
        return np.full((1, 1), int(bg), dtype=int)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.asarray(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each local legend to its nearby execution source, stamp the teal pattern there,
    then return only the cropped transformation region.

    Input structure:
    - Black background with several compact local clusters rather than one global legend area.
    - In each cluster, a colored source-target pair points toward a nearby teal (8) stencil,
      and a second pixel of the same source color sits 2-3 cells away as the execution site.

    Rule:
    - Discover each legend from the source->target pair plus its adjacent teal stencil.
    - For that legend, find the uniquely nearest same-colored execution pixel in the local
      neighborhood, stamp the teal stencil there recolored to the legend target color, and
      do not apply the rule to other far-away pixels of the same color.
    - Erase the legend pixels, teal instruction pixels, and paired execution sources, then
      crop the final answer to the tight bounding box of the remaining transformed output.

    Key insight: The same source color can appear in multiple clusters with different nearby
    instructions, so the rule is determined by spatial pairing, not by color alone; after
    the local transforms, only the occupied result region is returned.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8
    h, w = g.shape

    legends = _discover_legends(g, bg=bg, pattern_color=pattern_color)
    assignments, legend_pixels = _assign_local_sources(g, legends, bg=bg, pattern_color=pattern_color)
    local_source_pixels = set(assignments.values())

    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != int(pattern_color)))):
        pos = (int(r), int(c))
        if pos in legend_pixels or pos in local_source_pixels:
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

    return _crop_to_non_bg(out, bg=bg)


# --------------- generator (fused variant) ---------------

GRID_DIM_MAX = 20
MIN_DIM_BY_GROUPS = {2: 12, 3: 14, 4: 16, 5: 18}
GROUP_COUNT_WEIGHTS = [2, 2, 3, 3, 4, 4, 5]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

STENCIL_MIN_PIXELS = 2
STENCIL_MAX_PIXELS = 7
STENCIL_SPAN_MAX = 4

CLUSTER_RETRIES = 1200
BOX_RETRIES = 120
OUTER_RETRIES = 500


def _expand_with_moat(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    expanded = set(cells)
    for r, c in list(cells):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                expanded.add((int(r + dr), int(c + dc)))
    return expanded


def _rel_bounds(rel: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _c in rel]
    cols = [int(c) for _r, c in rel]
    return min(rows), max(rows), min(cols), max(cols)


def _make_stencil() -> tuple[list[tuple[int, int]], tuple[int, int]]:
    for _ in range(200):
        n = random.randint(STENCIL_MIN_PIXELS, STENCIL_MAX_PIXELS)
        coords = {(0, 0)}
        for _ in range(n - 1):
            frontier = set()
            for r, c in coords:
                for dr, dc in _DIRS8:
                    pos = (int(r + dr), int(c + dc))
                    if pos not in coords:
                        frontier.add(pos)
            coords.add(random.choice(list(frontier)))
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        if max(rows) - min(rows) + 1 > STENCIL_SPAN_MAX:
            continue
        if max(cols) - min(cols) + 1 > STENCIL_SPAN_MAX:
            continue
        coords_list = sorted(coords)
        return coords_list, random.choice(coords_list)
    raise RuntimeError("Failed to sample bounded stencil")


def _allocate_counts(num_groups: int, num_families: int) -> list[int]:
    counts = [1] * num_families
    for _ in range(num_groups - num_families):
        counts[random.randrange(num_families)] += 1
    random.shuffle(counts)
    return counts


def _build_groups(num_groups: int) -> tuple[list[dict], set[int]]:
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_families = min(3, num_groups - 1)
    num_families = random.randint(1, max_families)
    src_colors = colors[:num_families]
    target_pool = [c for c in NON_BG_NON_TEAL_COLORS if c not in src_colors]
    counts = _allocate_counts(num_groups, num_families)

    groups: list[dict] = []
    used_colors: set[int] = set()
    for src, count in zip(src_colors, counts):
        targets = random.sample(target_pool, count)
        for tgt in targets:
            stencil, anchor = _make_stencil()
            rel = [(int(r - anchor[0]), int(c - anchor[1])) for r, c in stencil]
            bounds = _rel_bounds(rel)
            groups.append({"src": int(src), "tgt": int(tgt), "rel": rel, "bounds": bounds})
            used_colors.add(int(src))
            used_colors.add(int(tgt))

    random.shuffle(groups)
    return groups, used_colors


def _choose_grid(num_groups: int, groups: list[dict]) -> tuple[int, int]:
    max_stamp_h = max(int(g["bounds"][1] - g["bounds"][0] + 1) for g in groups)
    max_stamp_w = max(int(g["bounds"][3] - g["bounds"][2] + 1) for g in groups)
    min_dim = MIN_DIM_BY_GROUPS[num_groups]
    min_h = max(min_dim, max_stamp_h + 6)
    min_w = max(min_dim, max_stamp_w + 6)
    h = random.randint(min_h, GRID_DIM_MAX)
    w = random.randint(min_w, GRID_DIM_MAX)
    return h, w


def _choose_result_box(h: int, w: int, groups: list[dict]) -> tuple[int, int, int, int] | None:
    max_stamp_h = max(int(g["bounds"][1] - g["bounds"][0] + 1) for g in groups)
    max_stamp_w = max(int(g["bounds"][3] - g["bounds"][2] + 1) for g in groups)
    min_area = sum(len(g["rel"]) for g in groups) + 4 * len(groups)
    min_box_h = max(max_stamp_h + 1, 5)
    min_box_w = max(max_stamp_w + 1, 5)

    for _ in range(BOX_RETRIES):
        top_margin = random.randint(1, min(3, h - 2))
        bottom_margin = random.randint(1, min(3, h - top_margin - 1))
        left_margin = random.randint(1, min(3, w - 2))
        right_margin = random.randint(1, min(3, w - left_margin - 1))
        r0 = int(top_margin)
        c0 = int(left_margin)
        r1 = int(h - 1 - bottom_margin)
        c1 = int(w - 1 - right_margin)
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        if box_h < min_box_h or box_w < min_box_w:
            continue
        if box_h * box_w < min_area:
            continue
        return r0, c0, r1, c1
    return None


def _offsets_for_direction(dr: int, dc: int) -> list[tuple[int, int]]:
    return [offset for offset in _LOCAL_OFFSETS if dr * offset[0] + dc * offset[1] < 0]


def _min_distance_to_cells(cell: tuple[int, int], others: set[tuple[int, int]]) -> int:
    return min(_chebyshev(cell, other) for other in others)


def _inside_box(cells: set[tuple[int, int]], box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = map(int, box)
    return all(r0 <= int(r) <= r1 and c0 <= int(c) <= c1 for r, c in cells)


def _place_clusters(groups: list[dict], inp: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    h, w = inp.shape
    blocked: set[tuple[int, int]] = set()
    occupied_output: set[tuple[int, int]] = set()
    r0, c0, r1, c1 = map(int, box)

    for group in groups:
        rel = group["rel"]
        placed = False
        for _ in range(CLUSTER_RETRIES):
            dr, dc = random.choice(_DIRS4)
            off_r, off_c = random.choice(_offsets_for_direction(dr, dc))
            er = random.randint(r0, r1)
            ec = random.randint(c0, c1)
            sr = er - off_r
            sc = ec - off_c
            tr = sr + dr
            tc = sc + dc
            ar = sr + 2 * dr
            ac = sc + 2 * dc

            if not (0 <= sr < h and 0 <= sc < w and 0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
                continue

            exec_pos = (int(er), int(ec))
            legend_cells = {(int(sr), int(sc)), (int(tr), int(tc))}
            if exec_pos in legend_cells:
                continue
            if _chebyshev(exec_pos, (int(sr), int(sc))) not in LOCAL_SOURCE_DISTS:
                continue

            teal_cells: set[tuple[int, int]] = set()
            stamp_cells: set[tuple[int, int]] = set()
            ok = True
            for rr, cc in rel:
                teal = (int(ar + rr), int(ac + cc))
                stamp = (int(er + rr), int(ec + cc))
                if not (0 <= teal[0] < h and 0 <= teal[1] < w and 0 <= stamp[0] < h and 0 <= stamp[1] < w):
                    ok = False
                    break
                teal_cells.add(teal)
                stamp_cells.add(stamp)
            if not ok:
                continue
            if not _inside_box(stamp_cells, box):
                continue

            instruction_cells = legend_cells | teal_cells
            if exec_pos in teal_cells:
                continue
            if legend_cells & teal_cells:
                continue
            if stamp_cells & instruction_cells:
                continue
            if _min_distance_to_cells(exec_pos, instruction_cells) < 2:
                continue

            my_legend_source = (int(sr), int(sc))
            my_exec_dist = _chebyshev(exec_pos, my_legend_source)
            pairing_ok = True
            for peer in groups:
                if peer is group or "exec" not in peer or int(peer["src"]) != int(group["src"]):
                    continue
                peer_legend_source = peer["legend_source"]
                peer_exec = peer["exec"]
                peer_exec_dist = _chebyshev(peer_exec, peer_legend_source)
                if my_exec_dist >= _chebyshev(exec_pos, peer_legend_source):
                    pairing_ok = False
                    break
                if peer_exec_dist >= _chebyshev(peer_exec, my_legend_source):
                    pairing_ok = False
                    break
            if not pairing_ok:
                continue

            cluster_cells = instruction_cells | {exec_pos} | stamp_cells
            if cluster_cells & blocked:
                continue
            if stamp_cells & occupied_output:
                continue

            inp[int(sr), int(sc)] = int(group["src"])
            inp[int(tr), int(tc)] = int(group["tgt"])
            inp[int(er), int(ec)] = int(group["src"])
            for nr, nc in teal_cells:
                inp[int(nr), int(nc)] = 8

            group["exec"] = exec_pos
            group["legend_source"] = my_legend_source
            group["stamp"] = stamp_cells
            group["footprint"] = cluster_cells
            blocked.update(_expand_with_moat(cluster_cells))
            occupied_output.update(stamp_cells)
            placed = True
            break
        if not placed:
            return False
    return True


def _build_output(groups: list[dict], h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for group in groups:
        for r, c in group["stamp"]:
            out[int(r), int(c)] = int(group["tgt"])
    return out


def _has_reused_source(groups: list[dict]) -> bool:
    counts: dict[int, int] = {}
    for group in groups:
        counts[int(group["src"])] = counts.get(int(group["src"]), 0) + 1
    return max(counts.values(), default=0) >= 2


def _gen_attempt() -> np.ndarray | None:
    num_groups = random.choice(GROUP_COUNT_WEIGHTS)
    groups, _used_colors = _build_groups(num_groups)
    if not _has_reused_source(groups):
        return None

    h, w = _choose_grid(num_groups, groups)
    box = _choose_result_box(h, w, groups)
    if box is None:
        return None

    inp = np.zeros((h, w), dtype=int)
    if not _place_clusters(groups, inp, box):
        return None

    full_output = _build_output(groups, h, w)
    expected = _crop_to_non_bg(full_output, bg=0)
    if expected.shape[0] >= h or expected.shape[1] >= w:
        return None

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
