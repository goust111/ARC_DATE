from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5

MAX_SIDE = 30
MAX_ATTEMPTS = 500


Coord = tuple[int, int]
Candidate = tuple[Coord, Coord]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_from_mask(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    regions: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue
            stack = [(r, c)]
            labels[r, c] = len(regions)
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = len(regions)
                        stack.append((nr, nc))
            regions.append(np.array(coords, dtype=int))
    return labels, regions


def _draw_segment(grid: np.ndarray, a: Coord, b: Coord, color: int) -> None:
    r0, c0 = a
    r1, c1 = b
    assert r0 == r1 or c0 == c1
    if r0 == r1:
        c_lo, c_hi = sorted((c0, c1))
        grid[r0, c_lo : c_hi + 1] = color
    else:
        r_lo, r_hi = sorted((r0, r1))
        grid[r_lo : r_hi + 1, c0] = color


def _draw_polyline(grid: np.ndarray, points: list[Coord], color: int) -> None:
    for a, b in zip(points, points[1:]):
        _draw_segment(grid, a, b, color)


def _region_labels(grid: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    mask = np.zeros_like(grid, dtype=bool)
    mask[1:-1, :] = grid[1:-1, :] != GREEN
    return _components_from_mask(mask)


def _fill_color(seed_colors: set[int]) -> int | None:
    if not seed_colors:
        return None
    if seed_colors == {BLUE}:
        return BLUE
    if seed_colors == {RED}:
        return RED
    return YELLOW


def _plan_seeds(grid: np.ndarray, labels: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, _w = grid.shape
    seeds: list[tuple[int, int, int, int]] = []

    for r, c in np.argwhere(grid == GRAY):
        r = int(r)
        c = int(c)
        top_dist = r
        bottom_dist = h - 1 - r
        if top_dist <= bottom_dist:
            nr, nc, color = r - 1, c, BLUE
        else:
            nr, nc, color = r + 1, c, RED
        assert 0 < nr < h - 1
        assert int(grid[nr, nc]) == BLACK
        region_id = int(labels[nr, nc])
        assert region_id >= 0
        seeds.append((nr, nc, color, region_id))
    return seeds


def _control_candidates(grid: np.ndarray, labels: np.ndarray, regions: list[np.ndarray]) -> dict[int, dict[str, list[Candidate]]]:
    h, _w = grid.shape
    out: dict[int, dict[str, list[Candidate]]] = {}

    for region_id, coords in enumerate(regions):
        blue: list[Candidate] = []
        red: list[Candidate] = []
        for r, c in coords:
            r = int(r)
            c = int(c)
            if int(grid[r, c]) != BLACK:
                continue
            if r >= 2 and int(labels[r - 1, c]) == region_id and r <= h - 1 - r:
                blue.append(((r, c), (r - 1, c)))
            if r <= h - 3 and int(labels[r + 1, c]) == region_id and r > h - 1 - r:
                red.append(((r, c), (r + 1, c)))
        out[region_id] = {"blue": blue, "red": red}
    return out


def _sample_sorted_positions(low: int, high: int, count: int, gap: int) -> list[int] | None:
    choices = list(range(low, high + 1))
    for _ in range(80):
        xs = sorted(random.sample(choices, count))
        if all(b - a >= gap for a, b in zip(xs, xs[1:])):
            return xs
    return None


def _build_partition() -> tuple[np.ndarray, np.ndarray, list[np.ndarray]] | None:
    h = random.randint(12, 18)
    w = random.randint(14, 22)
    grid = np.zeros((h, w), dtype=int)
    grid[0, :] = BLUE
    grid[-1, :] = RED

    anchors_count = random.randint(4, 7)
    anchor_cols = _sample_sorted_positions(2, w - 3, anchors_count, gap=2)
    if anchor_cols is None:
        return None

    top_row = 1
    bottom_row = h - 2
    start_row = random.randint(3, h - 4)
    end_row = random.randint(3, h - 4)

    anchor_rows: list[int] = []
    row = random.choice((top_row, bottom_row))
    for _ in range(anchors_count):
        anchor_rows.append(row)
        row = bottom_row if row == top_row else top_row

    points: list[Coord] = [(start_row, 0)]
    current = points[0]
    for col, target_row in zip(anchor_cols, anchor_rows):
        turn = (current[0], col)
        if turn != current:
            points.append(turn)
            current = turn
        if current[0] != target_row:
            current = (target_row, col)
            points.append(current)
    tail = (current[0], w - 1)
    if tail != current:
        points.append(tail)
        current = tail
    if current[0] != end_row:
        points.append((end_row, w - 1))

    _draw_polyline(grid, points, GREEN)
    labels, regions = _region_labels(grid)
    if not (4 <= len(regions) <= 6):
        return None
    if min(len(coords) for coords in regions) < 8:
        return None
    return grid, labels, regions


def _pick_candidate(
    candidates: list[Candidate],
    used_controls: set[Coord],
    used_seeds: set[Coord],
) -> Candidate | None:
    order = candidates[:]
    random.shuffle(order)
    for control, seed in order:
        if control in used_controls or control in used_seeds:
            continue
        if seed in used_controls or seed in used_seeds:
            continue
        return control, seed
    return None


def _pick_conflict_pair(
    blue_candidates: list[Candidate],
    red_candidates: list[Candidate],
    used_controls: set[Coord],
    used_seeds: set[Coord],
) -> list[tuple[Coord, Coord, int]] | None:
    blue_order = blue_candidates[:]
    red_order = red_candidates[:]
    random.shuffle(blue_order)
    random.shuffle(red_order)
    for blue_control, blue_seed in blue_order:
        if blue_control in used_controls or blue_control in used_seeds:
            continue
        if blue_seed in used_controls or blue_seed in used_seeds:
            continue
        for red_control, red_seed in red_order:
            if red_control in used_controls or red_control in used_seeds:
                continue
            if red_seed in used_controls or red_seed in used_seeds:
                continue
            if blue_control == red_control or blue_seed == red_seed:
                continue
            return [
                (blue_control, blue_seed, BLUE),
                (red_control, red_seed, RED),
            ]
    return None


def _choose_roles(candidates: dict[int, dict[str, list[Candidate]]], region_count: int) -> tuple[int, int, int, int] | None:
    conflict_regions = [rid for rid in range(region_count) if candidates[rid]["blue"] and candidates[rid]["red"]]
    blue_regions = [rid for rid in range(region_count) if candidates[rid]["blue"]]
    red_regions = [rid for rid in range(region_count) if candidates[rid]["red"]]

    random.shuffle(conflict_regions)
    for conflict_region in conflict_regions:
        blue_only_options = [rid for rid in blue_regions if rid != conflict_region]
        red_only_options = [rid for rid in red_regions if rid != conflict_region]
        random.shuffle(blue_only_options)
        random.shuffle(red_only_options)
        for blue_only_region in blue_only_options:
            for red_only_region in red_only_options:
                if red_only_region == blue_only_region:
                    continue
                remaining = [
                    rid
                    for rid in range(region_count)
                    if rid not in {conflict_region, blue_only_region, red_only_region}
                ]
                if not remaining:
                    continue
                empty_region = random.choice(remaining)
                return conflict_region, blue_only_region, red_only_region, empty_region
    return None


def _validate_instance(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= MAX_SIDE
    assert np.all(grid[0, :] == BLUE)
    assert np.all(grid[-1, :] == RED)

    labels, regions = _region_labels(grid)
    assert 4 <= len(regions) <= 6

    seeds = _plan_seeds(grid, labels)
    region_seed_sets = {rid: set() for rid in range(len(regions))}
    seed_lookup: dict[Coord, int] = {}
    for r, c, color, region_id in seeds:
        prev = seed_lookup.get((r, c))
        assert prev in (None, color)
        seed_lookup[(r, c)] = color
        region_seed_sets[region_id].add(color)

    out = main(grid)
    pure_blue_regions = 0
    pure_red_regions = 0
    mixed_regions = 0
    empty_regions = 0

    for region_id, coords in enumerate(regions):
        fill = _fill_color(region_seed_sets[region_id])
        saw_fill = False
        saw_black = False
        for r, c in coords:
            r = int(r)
            c = int(c)
            if (r, c) in seed_lookup:
                assert int(out[r, c]) == seed_lookup[(r, c)]
            elif int(grid[r, c]) == GRAY:
                assert int(out[r, c]) == GRAY
            elif int(grid[r, c]) == BLACK:
                expected = BLACK if fill is None else fill
                assert int(out[r, c]) == expected
                saw_fill |= expected != BLACK
                saw_black |= expected == BLACK
        if not region_seed_sets[region_id]:
            assert saw_black
            empty_regions += 1
        elif region_seed_sets[region_id] == {BLUE}:
            assert saw_fill
            pure_blue_regions += 1
        elif region_seed_sets[region_id] == {RED}:
            assert saw_fill
            pure_red_regions += 1
        else:
            assert saw_fill
            assert any(int(out[int(r), int(c)]) == YELLOW for r, c in coords if int(grid[int(r), int(c)]) == BLACK)
            mixed_regions += 1

    assert empty_regions >= 1
    assert pure_blue_regions >= 1
    assert pure_red_regions >= 1
    assert mixed_regions >= 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each gray control pixel one step toward its nearer edge bar, then flood-fill the control's partition according to the seed colors that land inside it.

    Input structure:
    - The top row is a full blue edge bar and the bottom row is a full red edge bar on a black background.
    - A single one-cell-thick green orthogonal divider line snakes between the bars and partitions the interior into several enclosed regions.
    - Gray singleton control pixels sit inside those regions and never lie on the divider.

    Rule:
    - For each gray control pixel, compare its row-distance to the top blue bar and the bottom red bar.
    - Paint the immediately adjacent cell one step upward for the nearer top bar, or one step downward for the nearer bottom bar; ties extend upward and use blue.
    - Treat those newly painted cells as region seeds: a region with only blue seeds flood-fills its black cells blue, a region with only red seeds flood-fills its black cells red, a region with both seed colors flood-fills its black cells yellow, and a region with no seeds stays black.
    - Preserve the original gray controls, the green divider, and the two edge bars.

    Key insight:
    - The one-step extension is not the final output; it activates the partition containing that seed, and mixed blue/red activation inside one partition produces a third color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= MAX_SIDE
    assert np.all(g[0, :] == BLUE)
    assert np.all(g[-1, :] == RED)

    labels, regions = _region_labels(g)
    out = g.copy()
    region_seed_sets = {rid: set() for rid in range(len(regions))}
    claimed_seeds: dict[Coord, int] = {}

    for r, c, color, region_id in _plan_seeds(g, labels):
        prev = claimed_seeds.get((r, c))
        assert prev in (None, color)
        claimed_seeds[(r, c)] = color
        out[r, c] = color
        region_seed_sets[region_id].add(color)

    for region_id, coords in enumerate(regions):
        fill = _fill_color(region_seed_sets[region_id])
        if fill is None:
            continue
        for r, c in coords:
            r = int(r)
            c = int(c)
            if int(out[r, c]) == BLACK:
                out[r, c] = fill
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _build_partition()
        if built is None:
            continue
        grid, labels, regions = built
        candidates = _control_candidates(grid, labels, regions)
        roles = _choose_roles(candidates, len(regions))
        if roles is None:
            continue

        conflict_region, blue_only_region, red_only_region, empty_region = roles
        region_modes: dict[int, tuple[str, ...]] = {}
        for region_id in range(len(regions)):
            if region_id == empty_region:
                region_modes[region_id] = ()
            elif region_id == conflict_region:
                region_modes[region_id] = ("blue", "red")
            elif region_id == blue_only_region:
                region_modes[region_id] = ("blue",)
            elif region_id == red_only_region:
                region_modes[region_id] = ("red",)
            else:
                options = []
                if candidates[region_id]["blue"]:
                    options.append("blue")
                if candidates[region_id]["red"]:
                    options.append("red")
                random.shuffle(options)
                region_modes[region_id] = tuple(options[:1])

        used_controls: set[Coord] = set()
        used_seeds: set[Coord] = set()
        placements: list[tuple[Coord, Coord, int]] = []

        conflict_pair = _pick_conflict_pair(
            candidates[conflict_region]["blue"],
            candidates[conflict_region]["red"],
            used_controls,
            used_seeds,
        )
        if conflict_pair is None:
            continue
        for control, seed, color in conflict_pair:
            used_controls.add(control)
            used_seeds.add(seed)
            placements.append((control, seed, color))

        picked = _pick_candidate(candidates[blue_only_region]["blue"], used_controls, used_seeds)
        if picked is None:
            continue
        used_controls.add(picked[0])
        used_seeds.add(picked[1])
        placements.append((picked[0], picked[1], BLUE))

        picked = _pick_candidate(candidates[red_only_region]["red"], used_controls, used_seeds)
        if picked is None:
            continue
        used_controls.add(picked[0])
        used_seeds.add(picked[1])
        placements.append((picked[0], picked[1], RED))

        target_controls = random.randint(4, 6)
        extra_regions = list(range(len(regions)))
        random.shuffle(extra_regions)
        while len(placements) < target_controls:
            added = False
            for region_id in extra_regions:
                modes = region_modes[region_id]
                if not modes:
                    continue
                for mode in random.sample(list(modes), len(modes)):
                    picked = _pick_candidate(candidates[region_id][mode], used_controls, used_seeds)
                    if picked is None:
                        continue
                    color = BLUE if mode == "blue" else RED
                    used_controls.add(picked[0])
                    used_seeds.add(picked[1])
                    placements.append((picked[0], picked[1], color))
                    added = True
                    break
                if added:
                    break
            if not added:
                break
        if not (4 <= len(placements) <= 6):
            continue

        inp = grid.copy()
        for (r, c), _seed, _color in placements:
            inp[r, c] = GRAY

        try:
            _validate_instance(inp)
        except AssertionError:
            continue
        return inp

    raise RuntimeError("failed to generate valid input")
