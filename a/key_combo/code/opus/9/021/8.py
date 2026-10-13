from __future__ import annotations

from collections import Counter, defaultdict, deque

import numpy as np


BLACK = 0
BLUE = 1
MARKER_COLORS = (2, 3, 4)
GREY = 5
BAND = 2
GEN_RETRIES = 400
RNG = np.random.default_rng()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    components: dict[int, np.ndarray] = {}
    next_id = 1
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            labels[r, c] = next_id
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        _in_bounds(mask.shape, nr, nc)
                        and mask[nr, nc]
                        and labels[nr, nc] == 0
                    ):
                        labels[nr, nc] = next_id
                        q.append((nr, nc))
            components[next_id] = np.array(coords, dtype=int)
            next_id += 1
    return labels, components


def _find_anchor(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == BLUE)
    assert coords.shape[0] == 4, "expected a unique 2x2 blue anchor"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "anchor must be a solid 2x2 block"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BLUE), "anchor must be solid"
    return int(r0), int(c0)


def _distance_to_anchor(r: int, c: int, anchor_r: int, anchor_c: int) -> int:
    return min(
        abs(r - rr) + abs(c - cc)
        for rr in (anchor_r, anchor_r + 1)
        for cc in (anchor_c, anchor_c + 1)
    )


def _is_adjacent_to_anchor(r: int, c: int, anchor_r: int, anchor_c: int) -> bool:
    return any(
        abs(r - rr) + abs(c - cc) == 1
        for rr in (anchor_r, anchor_r + 1)
        for cc in (anchor_c, anchor_c + 1)
    )


def _marker_destination(
    r: int, c: int, anchor_r: int, anchor_c: int
) -> tuple[int, int]:
    if r in (anchor_r, anchor_r + 1):
        if c < anchor_c:
            return r, anchor_c - 1
        if c > anchor_c + 1:
            return r, anchor_c + 2
        raise AssertionError("marker on anchor row cannot start inside the anchor corridor")

    if c in (anchor_c, anchor_c + 1):
        if r < anchor_r:
            return anchor_r - 1, c
        if r > anchor_r + 1:
            return anchor_r + 2, c
        raise AssertionError("marker on anchor column cannot start inside the anchor corridor")

    raise AssertionError("each marker must be aligned with the anchor by row or column")


def _path_cells(
    src: tuple[int, int], dst: tuple[int, int]
) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = src, dst
    if r0 == r1:
        step = 1 if c1 > c0 else -1
        return [(r0, c) for c in range(c0 + step, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 > r0 else -1
        return [(r, c0) for r in range(r0 + step, r1 + step, step)]
    raise AssertionError("slide path must be axial")


def _analyze(grid: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    anchor_r, anchor_c = _find_anchor(g)
    marker_coords = np.argwhere(np.isin(g, MARKER_COLORS))
    assert marker_coords.size > 0, "expected at least one marker"

    labels, components = _label_components(g == BLACK)
    assert components, "expected black cell interiors"

    records: list[dict[str, object]] = []
    for rr, cc in marker_coords:
        r = int(rr)
        c = int(cc)
        color = int(g[r, c])
        dst_r, dst_c = _marker_destination(r, c, anchor_r, anchor_c)
        assert _is_adjacent_to_anchor(dst_r, dst_c, anchor_r, anchor_c), (
            "marker must stop on the grey ring around the anchor"
        )

        for pr, pc in _path_cells((r, c), (dst_r, dst_c)):
            assert int(g[pr, pc]) == GREY, "marker path to the anchor must be uninterrupted grey track"

        neighbor_ids = {
            int(labels[nr, nc])
            for nr, nc in _neighbors4(dst_r, dst_c)
            if _in_bounds(g.shape, nr, nc) and int(labels[nr, nc]) > 0
        }
        assert len(neighbor_ids) == 1, "each stopped marker must border exactly one black cell"
        component_id = next(iter(neighbor_ids))

        records.append(
            {
                "pos": (r, c),
                "color": color,
                "dest": (dst_r, dst_c),
                "component_id": component_id,
                "distance": _distance_to_anchor(r, c, anchor_r, anchor_c),
            }
        )

    return labels, components, records


def _winning_color(records: list[dict[str, object]]) -> int:
    counts = Counter(int(rec["color"]) for rec in records)
    best_count = max(counts.values())
    tied_colors = [color for color, count in counts.items() if count == best_count]
    if len(tied_colors) == 1:
        return int(tied_colors[0])

    farthest_by_color = {
        color: max(
            int(rec["distance"])
            for rec in records
            if int(rec["color"]) == color
        )
        for color in tied_colors
    }
    best_distance = max(farthest_by_color.values())
    winners = [color for color, dist in farthest_by_color.items() if dist == best_distance]
    assert len(winners) == 1, "tied colors must be separated by a unique farthest marker"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide colored track markers into the anchor ring, vote within the adjacent cells, and flood-fill each voted cell.

    Input structure:
    - A grey lattice separates black rectangular cell interiors into a regular grid.
    - One unique blue 2x2 anchor sits on an internal lattice junction.
    - Red, green, and yellow single-pixel markers lie on the anchor-aligned grey tracks.

    Rule:
    - Move each marker straight along its row or column toward the blue anchor until it reaches the grey ring cell that is 4-neighbor-adjacent to the anchor.
    - Associate that stopping ring cell with the unique black cell it borders, using the stop location rather than the marker's original position.
    - For each black cell that receives markers, choose the majority marker color; if colors tie, use the color of the uniquely farthest original marker among the tied colors; fill that whole cell with the winner. Restore all track pixels to grey and keep the blue anchor unchanged.

    Key insight:
    - The lattice is both a routing network and a partition: markers first collapse onto anchor-adjacent ring cells, and only those final docking positions determine which neighboring cell gets to vote.
    """
    g = np.asarray(grid, dtype=int)
    labels, components, records = _analyze(g)

    out = g.copy()
    out[np.isin(out, MARKER_COLORS)] = GREY

    votes: dict[int, list[dict[str, object]]] = defaultdict(list)
    for rec in records:
        votes[int(rec["component_id"])].append(rec)

    for component_id, component_records in votes.items():
        winner = _winning_color(component_records)
        coords = components[component_id]
        out[coords[:, 0], coords[:, 1]] = winner

    return out


def _cell_top(index: int, cell_size: int) -> int:
    return BAND + index * (cell_size + BAND)


def _build_lattice(
    n_rows: int, n_cols: int, cell_size: int
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int]]]]:
    height = n_rows * cell_size + (n_rows + 1) * BAND
    width = n_cols * cell_size + (n_cols + 1) * BAND
    grid = np.full((height, width), GREY, dtype=int)

    cells: list[list[tuple[int, int, int, int]]] = []
    for i in range(n_rows):
        row: list[tuple[int, int, int, int]] = []
        r0 = _cell_top(i, cell_size)
        for j in range(n_cols):
            c0 = _cell_top(j, cell_size)
            r1 = r0 + cell_size - 1
            c1 = c0 + cell_size - 1
            grid[r0 : r1 + 1, c0 : c1 + 1] = BLACK
            row.append((r0, c0, r1, c1))
        cells.append(row)
    return grid, cells


def _approach_specs(
    shape: tuple[int, int], anchor_r: int, anchor_c: int, anchor_cell_r: int, anchor_cell_c: int
) -> dict[str, dict[str, object]]:
    h, w = shape
    return {
        "north_left": {
            "cell": (anchor_cell_r, anchor_cell_c),
            "positions": [(r, anchor_c) for r in range(0, anchor_r - 1)],
        },
        "north_right": {
            "cell": (anchor_cell_r, anchor_cell_c + 1),
            "positions": [(r, anchor_c + 1) for r in range(0, anchor_r - 1)],
        },
        "south_left": {
            "cell": (anchor_cell_r + 1, anchor_cell_c),
            "positions": [(r, anchor_c) for r in range(anchor_r + 3, h)],
        },
        "south_right": {
            "cell": (anchor_cell_r + 1, anchor_cell_c + 1),
            "positions": [(r, anchor_c + 1) for r in range(anchor_r + 3, h)],
        },
        "west_top": {
            "cell": (anchor_cell_r, anchor_cell_c),
            "positions": [(anchor_r, c) for c in range(0, anchor_c - 1)],
        },
        "west_bottom": {
            "cell": (anchor_cell_r + 1, anchor_cell_c),
            "positions": [(anchor_r + 1, c) for c in range(0, anchor_c - 1)],
        },
        "east_top": {
            "cell": (anchor_cell_r, anchor_cell_c + 1),
            "positions": [(anchor_r, c) for c in range(anchor_c + 3, w)],
        },
        "east_bottom": {
            "cell": (anchor_cell_r + 1, anchor_cell_c + 1),
            "positions": [(anchor_r + 1, c) for c in range(anchor_c + 3, w)],
        },
    }


CELL_APPROACHES: dict[tuple[int, int], tuple[str, str]] = {
    (0, 0): ("north_left", "west_top"),
    (0, 1): ("north_right", "east_top"),
    (1, 0): ("south_left", "west_bottom"),
    (1, 1): ("south_right", "east_bottom"),
}


def _distance_choices(
    positions: list[tuple[int, int]], anchor_r: int, anchor_c: int
) -> list[tuple[int, tuple[int, int]]]:
    return sorted(
        [(_distance_to_anchor(r, c, anchor_r, anchor_c), (r, c)) for r, c in positions],
        key=lambda item: (item[0], item[1][0], item[1][1]),
    )


def _pick_with_bias(
    choices: list[tuple[int, tuple[int, int]]], prefer_far: bool
) -> tuple[int, tuple[int, int]]:
    assert choices
    window = min(3, len(choices))
    band = choices[-window:] if prefer_far else choices[:window]
    return band[int(RNG.integers(len(band)))]


def _pick_distinct_pair(
    first_choices: list[tuple[int, tuple[int, int]]],
    second_choices: list[tuple[int, tuple[int, int]]],
) -> tuple[tuple[int, tuple[int, int]], tuple[int, tuple[int, int]]]:
    for _ in range(40):
        first = _pick_with_bias(first_choices, prefer_far=True)
        second = _pick_with_bias(second_choices, prefer_far=False)
        if first[0] != second[0]:
            return first, second
    first = first_choices[-1]
    second = next((item for item in second_choices if item[0] != first[0]), second_choices[0])
    assert first[0] != second[0], "tie cell requires two different source distances"
    return first, second


def _marker_plan(
    shape: tuple[int, int], anchor_r: int, anchor_c: int, anchor_cell_r: int, anchor_cell_c: int
) -> list[tuple[tuple[int, int], int]]:
    approaches = _approach_specs(shape, anchor_r, anchor_c, anchor_cell_r, anchor_cell_c)
    local_cells = list(CELL_APPROACHES.keys())
    contest_cell = local_cells[int(RNG.integers(len(local_cells)))]
    other_cells = [cell for cell in local_cells if cell != contest_cell]
    secondary_cell = other_cells[int(RNG.integers(len(other_cells)))]
    tertiary_cell = None
    if RNG.random() < 0.55:
        remainder = [cell for cell in other_cells if cell != secondary_cell]
        tertiary_cell = remainder[int(RNG.integers(len(remainder)))]

    plan: list[tuple[tuple[int, int], int]] = []
    used_approaches: set[str] = set()
    colors = list(MARKER_COLORS)
    RNG.shuffle(colors)
    contest_colors = colors[:2]

    first_name, second_name = CELL_APPROACHES[contest_cell]
    first_choices = _distance_choices(approaches[first_name]["positions"], anchor_r, anchor_c)
    second_choices = _distance_choices(approaches[second_name]["positions"], anchor_r, anchor_c)
    assert first_choices and second_choices, "contest cell must have available sources"
    far_pick, near_pick = _pick_distinct_pair(first_choices, second_choices)
    if RNG.random() < 0.5:
        plan.append((far_pick[1], contest_colors[0]))
        plan.append((near_pick[1], contest_colors[1]))
    else:
        plan.append((far_pick[1], contest_colors[1]))
        plan.append((near_pick[1], contest_colors[0]))
    used_approaches.update({first_name, second_name})

    majority_branch = RNG.random() < 0.65
    secondary_color = int(contest_colors[int(RNG.integers(len(contest_colors)))]) if not majority_branch else int(colors[2])
    sec_a, sec_b = CELL_APPROACHES[secondary_cell]
    if majority_branch:
        for approach_name, prefer_far in ((sec_a, True), (sec_b, False)):
            choices = _distance_choices(approaches[approach_name]["positions"], anchor_r, anchor_c)
            assert choices, "majority cell must have available sources"
            pick = _pick_with_bias(choices, prefer_far=prefer_far)
            plan.append((pick[1], secondary_color))
            used_approaches.add(approach_name)
    else:
        approach_name = sec_a if RNG.random() < 0.5 else sec_b
        choices = _distance_choices(approaches[approach_name]["positions"], anchor_r, anchor_c)
        assert choices, "secondary cell must have an available source"
        pick = _pick_with_bias(choices, prefer_far=RNG.random() < 0.5)
        plan.append((pick[1], secondary_color))
        used_approaches.add(approach_name)

    if tertiary_cell is not None:
        tertiary_color = int(MARKER_COLORS[int(RNG.integers(len(MARKER_COLORS)))])
        options = [name for name in CELL_APPROACHES[tertiary_cell] if name not in used_approaches]
        if options:
            approach_name = options[int(RNG.integers(len(options)))]
            choices = _distance_choices(approaches[approach_name]["positions"], anchor_r, anchor_c)
            assert choices, "tertiary cell must have an available source"
            pick = _pick_with_bias(choices, prefer_far=RNG.random() < 0.5)
            plan.append((pick[1], tertiary_color))

    return plan


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        labels, components, records = _analyze(grid)
    except AssertionError:
        return False

    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    votes: dict[int, list[dict[str, object]]] = defaultdict(list)
    for rec in records:
        votes[int(rec["component_id"])].append(rec)

    if len(votes) < 2:
        return False

    if not any(
        len(component_records) >= 2 and len({int(rec["color"]) for rec in component_records}) >= 2
        for component_records in votes.values()
    ):
        return False

    totals = Counter(int(rec["color"]) for rec in records)
    if len(set(totals.values())) == 1:
        return False

    used_directions = {
        "horizontal" if int(rec["pos"][0]) in (_find_anchor(grid)[0], _find_anchor(grid)[0] + 1) else "vertical"
        for rec in records
    }
    if len(used_directions) < 2:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if max(out.shape) > 30:
        return False
    if np.count_nonzero(out == BLUE) != 4:
        return False

    winner_colors = set()
    for component_id, component_records in votes.items():
        winner = _winning_color(component_records)
        winner_colors.add(winner)
        coords = components[component_id]
        if not np.all(out[coords[:, 0], coords[:, 1]] == winner):
            return False

    return bool(winner_colors)


def _gen_attempt() -> np.ndarray | None:
    cell_size = int(RNG.integers(2, 4))
    n_rows = int(RNG.integers(3, 6))
    n_cols = int(RNG.integers(3, 6))
    height = n_rows * cell_size + (n_rows + 1) * BAND
    width = n_cols * cell_size + (n_cols + 1) * BAND
    if max(height, width) > 30:
        return None

    grid, _cells = _build_lattice(n_rows, n_cols, cell_size)
    anchor_cell_r = int(RNG.integers(0, n_rows - 1))
    anchor_cell_c = int(RNG.integers(0, n_cols - 1))
    anchor_r = _cell_top(anchor_cell_r, cell_size) + cell_size
    anchor_c = _cell_top(anchor_cell_c, cell_size) + cell_size
    grid[anchor_r : anchor_r + 2, anchor_c : anchor_c + 2] = BLUE

    try:
        plan = _marker_plan(grid.shape, anchor_r, anchor_c, anchor_cell_r, anchor_cell_c)
    except AssertionError:
        return None

    for (r, c), color in plan:
        if int(grid[r, c]) != GREY:
            return None
        grid[r, c] = int(color)

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid routed-voting ARC input")
