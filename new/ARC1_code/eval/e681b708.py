from collections import Counter, deque

import numpy as np

from arc_utils import Color, components_mask, in_bounds, neighbors4, neighbors8


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each isolated blue dot by the majority-colored marker bordering its open region.

    Input structure
    - Black is the background.
    - Blue cells form two roles: multi-cell horizontal/vertical wall segments and isolated one-cell dots.
    - Non-blue foreground colors are marker cells sitting on the walls, endpoints, or region corners.

    Rule
    - Treat multi-cell blue components and all colored marker cells as barriers.
    - Flood-fill the remaining black cells plus isolated blue dots into 4-connected open regions.
    - For each region containing isolated blue dots, count the distinct colored marker cells that touch
      the region in the 8-neighborhood; recolor every isolated blue dot in that region to the unique
      majority marker color.

    Key insight
    - The colored wall/corner markers label compartments by vote; the scattered blue dots are the
      only cells to rewrite, while the blue wall network and marker cells stay fixed.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    dot_color = Color.BLUE
    assert int(g.max()) <= 9 and int(g.min()) >= 0
    assert bg == int(Counter(map(int, g.ravel())).most_common(1)[0][0])

    out = g.copy()
    h, w = g.shape

    blue_components = components_mask(g == dot_color, connectivity=4)
    blue_dots = {
        tuple(map(int, comp[0]))
        for comp in blue_components
        if int(comp.shape[0]) == 1
    }
    wall_cells = {
        tuple(map(int, rc))
        for comp in blue_components
        if int(comp.shape[0]) > 1
        for rc in comp
    }
    assert blue_dots
    assert wall_cells

    passable = g == bg
    for r, c in blue_dots:
        passable[r, c] = True

    seen = np.zeros((h, w), dtype=bool)
    for start_r in range(h):
        for start_c in range(w):
            if not passable[start_r, start_c] or seen[start_r, start_c]:
                continue

            queue = deque([(start_r, start_c)])
            seen[start_r, start_c] = True
            region_dots: list[tuple[int, int]] = []
            adjacent_markers: set[tuple[int, int]] = set()

            while queue:
                r, c = queue.popleft()
                if (r, c) in blue_dots:
                    region_dots.append((r, c))

                for nr, nc in neighbors4(r, c):
                    if in_bounds(g.shape, nr, nc) and passable[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

                for nr, nc in neighbors8(r, c):
                    if not in_bounds(g.shape, nr, nc):
                        continue
                    color = int(g[nr, nc])
                    if color not in (bg, dot_color):
                        adjacent_markers.add((nr, nc))

            if not region_dots:
                continue

            marker_counts = Counter(int(g[r, c]) for r, c in adjacent_markers)
            assert marker_counts
            max_count = max(marker_counts.values())
            winners = [color for color, count in marker_counts.items() if count == max_count]
            assert len(winners) == 1
            fill_color = winners[0]

            for r, c in region_dots:
                out[r, c] = fill_color

    return out

# --------------- generator (o5x) ---------------

# Canvas and base lattice bounds.
GRID_H_MIN = 20
GRID_H_MAX = 27
GRID_W_MIN = 22
GRID_W_MAX = 26
CUT_GAP_MIN = 4
CUT_GAP_MAX = 12
LAYOUT_LINE_COMBOS = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (1, 4), (4, 1)]

# Wall graph growth limits.
MARKER_COUNT_MIN = 9
MARKER_COUNT_MAX = 17
SPUR_MARKER_EXTRA_MAX = 6
SEGMENT_COUNT_MIN = 8
SEGMENT_COUNT_MAX = 18
SPUR_DISTANCE_MIN = 3
SPUR_DISTANCE_MAX = 12
CARDINAL_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Marker color assignment search limits.
MARKER_PALETTE = [Color.RED, Color.GREEN, Color.TEAL, Color.YELLOW, Color.PINK]
MARKER_COLOR_COUNT_CHOICES = [3, 4]
MARKERS_PER_COLOR_MIN = 1
MARKERS_PER_COLOR_MAX = 6
COLOR_SOLVE_ATTEMPTS = 40
COLOR_BACKTRACK_CALL_LIMIT = 60000

# Region and dot population limits.
REGION_COUNT_MIN = 6
REGION_COUNT_MAX = 11
DOT_COUNT_MIN = 20
DOT_COUNT_MAX = 35
DOTS_PER_REGION_MIN = 2
DOTS_PER_REGION_MAX = 5
GEN_ATTEMPTS = 300


def _gen_random_choice(seq):
    return seq[int(np.random.randint(len(seq)))]


def _gen_bounded_counts(total, parts, lo, hi):
    if total < parts * lo or total > parts * hi:
        return None
    counts = [lo] * parts
    extra = total - parts * lo
    while extra:
        avail = [i for i, v in enumerate(counts) if v < hi]
        i = avail[int(np.random.randint(len(avail)))]
        add = int(np.random.randint(1, min(hi - counts[i], extra) + 1))
        counts[i] += add
        extra -= add
    np.random.shuffle(counts)
    return counts


def _gen_cuts(size, n_lines):
    gaps = _gen_bounded_counts(size - 1, n_lines + 1, CUT_GAP_MIN, CUT_GAP_MAX)
    if gaps is None:
        return None
    cuts = []
    cur = 0
    for gap in gaps[:-1]:
        cur += gap
        cuts.append(cur)
    return cuts


def _gen_base_grid(h, w, n_v, n_h):
    rows = _gen_cuts(h, n_h)
    cols = _gen_cuts(w, n_v)
    if rows is None or cols is None:
        return None

    g = np.zeros((h, w), dtype=int)
    for c in cols:
        g[:, c] = Color.BLUE
    for r in rows:
        g[r, :] = Color.BLUE

    markers = set()
    for c in cols:
        markers.add((0, c))
        markers.add((h - 1, c))
    for r in rows:
        markers.add((r, 0))
        markers.add((r, w - 1))
    for r in rows:
        for c in cols:
            markers.add((r, c))
    for r, c in markers:
        g[r, c] = Color.RED

    base_segments = n_v * (n_h + 1) + n_h * (n_v + 1)
    return g, markers, base_segments


def _gen_blue_run(g, r, c, dr, dc):
    n = 0
    rr, cc = r + dr, c + dc
    while in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == Color.BLUE:
        n += 1
        rr += dr
        cc += dc
    return n


def _gen_path_from(start, dr, dc, dist):
    sr, sc = start
    path = [(sr + dr * i, sc + dc * i) for i in range(1, dist)]
    end = (sr + dr * dist, sc + dc * dist)
    return path, end


def _gen_path_clear(g, start, path, end):
    if not in_bounds(g.shape, end[0], end[1]) or int(g[end]) != Color.BLACK:
        return False
    path_set = set(path)
    for r, c in path:
        if not in_bounds(g.shape, r, c) or int(g[r, c]) != Color.BLACK:
            return False
    for r, c in path:
        for nr, nc in neighbors4(r, c):
            if not in_bounds(g.shape, nr, nc):
                continue
            if (nr, nc) == start or (nr, nc) in path_set:
                continue
            if int(g[nr, nc]) != Color.BLACK:
                return False
    for nr, nc in neighbors4(end[0], end[1]):
        if not in_bounds(g.shape, nr, nc):
            continue
        if (nr, nc) in path_set or (nr, nc) == start:
            continue
        if int(g[nr, nc]) != Color.BLACK:
            return False
    return True


def _gen_spur_paths(g, start, directions):
    """Return clear straight spur options from one wall or marker anchor."""
    dirs = list(directions)
    np.random.shuffle(dirs)
    out = []
    for dr, dc in dirs:
        for dist in range(SPUR_DISTANCE_MIN, SPUR_DISTANCE_MAX + 1):
            path, end = _gen_path_from(start, dr, dc, dist)
            if _gen_path_clear(g, start, path, end):
                out.append((start, path, end))
    return out


def _gen_marker_spur_candidates(g, markers):
    out = []
    for start in markers:
        sr, sc = start
        blue_degree = sum(
            1
            for nr, nc in neighbors4(sr, sc)
            if in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == Color.BLUE
        )
        if blue_degree > 2:
            continue
        out.extend(_gen_spur_paths(g, start, CARDINAL_DIRECTIONS))
    return out


def _gen_wall_branch_candidates(g):
    out = []
    for r, c in zip(*np.where(g == Color.BLUE)):
        r = int(r)
        c = int(c)
        up = _gen_blue_run(g, r, c, -1, 0)
        down = _gen_blue_run(g, r, c, 1, 0)
        left = _gen_blue_run(g, r, c, 0, -1)
        right = _gen_blue_run(g, r, c, 0, 1)
        branch_dirs = []
        if up >= 2 and down >= 2 and left == 0 and right == 0:
            branch_dirs = [(0, -1), (0, 1)]
        elif left >= 2 and right >= 2 and up == 0 and down == 0:
            branch_dirs = [(-1, 0), (1, 0)]
        out.extend(_gen_spur_paths(g, (r, c), branch_dirs))
    return out


def _gen_add_spurs(g, markers, base_segments):
    target_markers = int(
        np.random.randint(
            max(MARKER_COUNT_MIN, len(markers)),
            min(MARKER_COUNT_MAX, len(markers) + SPUR_MARKER_EXTRA_MAX) + 1,
        )
    )
    segment_count = base_segments
    while len(markers) < target_markers:
        need = target_markers - len(markers)
        choices = []
        if need >= 1:
            marker_choices = _gen_marker_spur_candidates(g, markers)
            if marker_choices:
                choices.append(("marker", marker_choices))
        if need >= 2:
            branch_choices = _gen_wall_branch_candidates(g)
            if branch_choices:
                choices.append(("branch", branch_choices))
        if not choices:
            return False
        kind, candidates = choices[int(np.random.randint(len(choices)))]
        start, path, end = candidates[int(np.random.randint(len(candidates)))]
        if kind == "branch":
            g[start] = Color.RED
            markers.add(start)
            segment_count += 2
        else:
            segment_count += 1
        for r, c in path:
            g[r, c] = Color.BLUE
        g[end] = Color.RED
        markers.add(end)
        if segment_count > SEGMENT_COUNT_MAX:
            return False
    return SEGMENT_COUNT_MIN <= segment_count <= SEGMENT_COUNT_MAX


def _gen_region_marker_lists(g, marker_list):
    marker_to_idx = {pos: i for i, pos in enumerate(marker_list)}
    marker_set = set(marker_list)
    comps = components_mask(g == Color.BLACK, connectivity=4)
    regions = []
    for comp in comps:
        adj = set()
        for r, c in comp:
            r = int(r)
            c = int(c)
            for nr, nc in neighbors8(r, c):
                if in_bounds(g.shape, nr, nc) and (nr, nc) in marker_set:
                    adj.add(marker_to_idx[(nr, nc)])
        regions.append(sorted(adj))
    return comps, regions


def _gen_region_valid(counts):
    used = [v for v in counts if v]
    if len(used) < 2:
        return False
    top = max(used)
    return sum(1 for v in used if v == top) == 1


def _gen_region_possible(region, assigned, remaining, n_colors):
    counts = [0] * n_colors
    unknown = 0
    for idx in region:
        color = assigned[idx]
        if color < 0:
            unknown += 1
        else:
            counts[color] += 1

    def rec(pos):
        if pos == unknown:
            return _gen_region_valid(counts)
        order = list(range(n_colors))
        np.random.shuffle(order)
        for color in order:
            if remaining[color] <= 0:
                continue
            remaining[color] -= 1
            counts[color] += 1
            if rec(pos + 1):
                counts[color] -= 1
                remaining[color] += 1
                return True
            counts[color] -= 1
            remaining[color] += 1
        return False

    return rec(0)


def _gen_solve_marker_colors(g, marker_list, regions):
    n_markers = len(marker_list)
    degrees = [0] * n_markers
    for region in regions:
        for idx in region:
            degrees[idx] += 1

    for _ in range(COLOR_SOLVE_ATTEMPTS):
        n_colors = int(np.random.choice(MARKER_COLOR_COUNT_CHOICES))
        if n_markers > MARKERS_PER_COLOR_MAX * n_colors:
            continue
        palette = list(np.random.choice(MARKER_PALETTE, size=n_colors, replace=False))
        targets = _gen_bounded_counts(
            n_markers, n_colors, MARKERS_PER_COLOR_MIN, MARKERS_PER_COLOR_MAX
        )
        if targets is None:
            continue
        assigned = [-1] * n_markers
        remaining = targets[:]
        order = list(range(n_markers))
        order.sort(key=lambda i: (-degrees[i], np.random.random()))
        calls = [0]

        def backtrack(pos):
            calls[0] += 1
            if calls[0] > COLOR_BACKTRACK_CALL_LIMIT:
                return False
            if pos == n_markers:
                return all(_gen_region_valid([sum(assigned[i] == c for i in region) for c in range(n_colors)]) for region in regions)
            idx = order[pos]
            colors = [c for c in range(n_colors) if remaining[c] > 0]
            np.random.shuffle(colors)
            for color in colors:
                assigned[idx] = color
                remaining[color] -= 1
                if all(_gen_region_possible(region, assigned, remaining[:], n_colors) for region in regions):
                    if backtrack(pos + 1):
                        return True
                remaining[color] += 1
                assigned[idx] = -1
            return False

        if backtrack(0):
            for idx, (r, c) in enumerate(marker_list):
                g[r, c] = int(palette[assigned[idx]])
            return True
    return False


def _gen_region_majorities(g, marker_list, regions):
    majorities = []
    for region in regions:
        counts = {}
        for idx in region:
            color = int(g[marker_list[idx]])
            counts[color] = counts.get(color, 0) + 1
        if len(counts) < 2:
            return None
        top = max(counts.values())
        winners = [color for color, count in counts.items() if count == top]
        if len(winners) != 1:
            return None
        majorities.append(winners[0])
    return majorities


def _gen_place_dots(g, comps, majorities):
    n_regions = len(comps)
    total = int(
        np.random.randint(
            max(DOT_COUNT_MIN, DOTS_PER_REGION_MIN * n_regions),
            min(DOT_COUNT_MAX, DOTS_PER_REGION_MAX * n_regions) + 1,
        )
    )
    dot_counts = _gen_bounded_counts(total, n_regions, DOTS_PER_REGION_MIN, DOTS_PER_REGION_MAX)
    if dot_counts is None:
        return None

    candidates = []
    for comp in comps:
        cells = []
        for r, c in comp:
            r = int(r)
            c = int(c)
            if any(
                in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == Color.BLUE
                for nr, nc in neighbors4(r, c)
            ):
                continue
            cells.append((r, c))
        candidates.append(cells)

    expected = g.copy()
    blocked = set()
    order = sorted(range(n_regions), key=lambda i: len(candidates[i]) / dot_counts[i])
    for idx in order:
        np.random.shuffle(candidates[idx])
        placed = 0
        while placed < dot_counts[idx]:
            avail = [p for p in candidates[idx] if p not in blocked]
            if not avail:
                return None
            pos = avail[int(np.random.randint(len(avail)))]
            r, c = pos
            g[r, c] = Color.BLUE
            expected[r, c] = majorities[idx]
            blocked.add(pos)
            for nr, nc in neighbors4(r, c):
                if in_bounds(g.shape, nr, nc):
                    blocked.add((nr, nc))
            placed += 1
    return expected


def _gen_layout():
    """Choose lattice line counts and dimensions compatible with gap bounds."""
    n_v, n_h = _gen_random_choice(LAYOUT_LINE_COMBOS)
    valid_h = [
        h
        for h in range(GRID_H_MIN, GRID_H_MAX + 1)
        if CUT_GAP_MIN * (n_h + 1) <= h - 1 <= CUT_GAP_MAX * (n_h + 1)
    ]
    valid_w = [
        w
        for w in range(GRID_W_MIN, GRID_W_MAX + 1)
        if CUT_GAP_MIN * (n_v + 1) <= w - 1 <= CUT_GAP_MAX * (n_v + 1)
    ]
    if not valid_h or not valid_w:
        return None
    h = _gen_random_choice(valid_h)
    w = _gen_random_choice(valid_w)
    return h, w, n_v, n_h


def _gen_attempt():
    layout = _gen_layout()
    if layout is None:
        return None
    h, w, n_v, n_h = layout
    built = _gen_base_grid(h, w, n_v, n_h)
    if built is None:
        return None
    g, markers, base_segments = built
    if not _gen_add_spurs(g, markers, base_segments):
        return None

    marker_list = list(markers)
    comps, regions = _gen_region_marker_lists(g, marker_list)
    if not REGION_COUNT_MIN <= len(comps) <= REGION_COUNT_MAX:
        return None
    if any(len(region) < 2 for region in regions):
        return None
    if not _gen_solve_marker_colors(g, marker_list, regions):
        return None
    majorities = _gen_region_majorities(g, marker_list, regions)
    if majorities is None:
        return None

    expected = _gen_place_dots(g, comps, majorities)
    if expected is None:
        return None
    try:
        actual = main(g)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return g


def generate_input():
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
