import random
from collections import Counter

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MARKER_COLORS = (RED, GREEN, YELLOW)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _d4_orbit(n: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    pts = {
        (r, c),
        (c, n - 1 - r),
        (n - 1 - r, n - 1 - c),
        (n - 1 - c, r),
        (r, n - 1 - c),
        (n - 1 - r, c),
        (c, r),
        (n - 1 - c, n - 1 - r),
    }
    return tuple(sorted(pts))


def _label_black_regions(grid: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = grid.shape
    labels = np.zeros((h, w), dtype=int)
    regions: list[np.ndarray] = []
    next_label = 1
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != 0 or labels[r, c] != 0:
                continue
            stack = [(r, c)]
            labels[r, c] = next_label
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if not _in_bounds(grid.shape, nr, nc):
                        continue
                    if int(grid[nr, nc]) != 0 or labels[nr, nc] != 0:
                        continue
                    labels[nr, nc] = next_label
                    stack.append((nr, nc))
            regions.append(np.array(coords, dtype=int))
            next_label += 1
    return labels, regions


def _region_votes(grid: np.ndarray, regions: list[np.ndarray]) -> list[Counter]:
    votes: list[Counter] = []
    for coords in regions:
        seen: set[tuple[int, int]] = set()
        counts: Counter = Counter()
        for r, c in coords:
            for dr, dc in DIR8:
                nr, nc = int(r) + dr, int(c) + dc
                if not _in_bounds(grid.shape, nr, nc):
                    continue
                color = int(grid[nr, nc])
                if color in MARKER_COLORS and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    counts[color] += 1
        votes.append(counts)
    return votes


def _strict_majority_color(counts: Counter) -> int:
    assert counts, "every open region must see at least one marker"
    ordered = counts.most_common()
    assert len(ordered) == 1 or ordered[0][1] > ordered[1][1], "region vote must have a strict majority"
    return int(ordered[0][0])


def _complete_symmetry(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square grid"
    n = g.shape[0]
    out = g.copy()
    for r in range(n):
        for c in range(n):
            if int(g[r, c]) != GRAY:
                continue
            orbit = _d4_orbit(n, r, c)
            visible = {int(g[rr, cc]) for rr, cc in orbit if int(g[rr, cc]) != GRAY}
            assert len(visible) == 1, "each masked orbit must expose exactly one visible color"
            out[r, c] = visible.pop()
    return out


def _solve_completed(completed: np.ndarray) -> np.ndarray:
    labels, regions = _label_black_regions(completed)
    assert regions, "expected at least one open region"
    votes = _region_votes(completed, regions)
    out = completed.copy()
    for coords, counts in zip(regions, votes):
        color = _strict_majority_color(counts)
        out[coords[:, 0], coords[:, 1]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete grey-masked eightfold-symmetric cells, then recolor each open region by its adjacent marker majority.

    Input structure:
    - A square black-background grid contains blue wall segments with eightfold symmetry.
    - Red, green, and yellow marker pixels sit on region-boundary cells inside the open areas.
    - Some marker cells are replaced by grey masks; each masked cell's full D4 orbit reveals one unique visible color.

    Rule:
    - Restore every grey cell from the unique non-grey color in its eight-way symmetry orbit.
    - Treat blue walls and all marker pixels as obstacles, and split the remaining black cells into 4-connected open regions.
    - For each region, count the colors of all marker pixels that touch that region in the 8-neighborhood.
    - Recolor every black cell in the region to that region's strict majority marker color, while keeping walls and markers unchanged.

    Key insight:
    - The masks do not just hide decoration: they hide marker votes, so symmetry completion must happen before the regional majority-counting step.
    """
    completed = _complete_symmetry(np.asarray(grid, dtype=int))
    return _solve_completed(completed)


def _build_wall_layout(n: int, ring_half: int) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], int, list[int]]:
    g = np.zeros((n, n), dtype=int)
    mid = n // 2
    lo = mid - ring_half
    hi = mid + ring_half
    g[0, :] = BLUE
    g[-1, :] = BLUE
    g[:, 0] = BLUE
    g[:, -1] = BLUE
    g[lo : hi + 1, lo] = BLUE
    g[lo : hi + 1, hi] = BLUE
    g[lo, lo : hi + 1] = BLUE
    g[hi, lo : hi + 1] = BLUE
    g[mid, : lo + 1] = BLUE
    g[mid, hi:] = BLUE
    g[: lo + 1, mid] = BLUE
    g[hi:, mid] = BLUE
    labels, regions = _label_black_regions(g)
    assert len(regions) == 5, "wall layout must create four outer regions and one center region"
    center_label = int(labels[mid, mid])
    quadrant_labels = [
        int(labels[1, 1]),
        int(labels[1, n - 2]),
        int(labels[n - 2, 1]),
        int(labels[n - 2, n - 2]),
    ]
    assert len(set(quadrant_labels)) == 4
    return g, labels, regions, center_label, quadrant_labels


def _is_boundary_open_cell(grid: np.ndarray, r: int, c: int) -> bool:
    for dr, dc in DIR8:
        nr, nc = r + dr, c + dc
        if not _in_bounds(grid.shape, nr, nc) or int(grid[nr, nc]) != 0:
            return True
    return False


def _enumerate_orbits(
    wall_grid: np.ndarray,
    base_labels: np.ndarray,
    center_label: int,
    quadrant_labels: list[int],
) -> tuple[list[dict], list[dict]]:
    n = wall_grid.shape[0]
    seen: set[tuple[tuple[int, int], ...]] = set()
    outer: list[dict] = []
    central: list[dict] = []
    for r in range(n):
        for c in range(n):
            if int(wall_grid[r, c]) != 0 or not _is_boundary_open_cell(wall_grid, r, c):
                continue
            orbit = _d4_orbit(n, r, c)
            if orbit in seen:
                continue
            seen.add(orbit)
            if not all(int(wall_grid[rr, cc]) == 0 and _is_boundary_open_cell(wall_grid, rr, cc) for rr, cc in orbit):
                continue
            counts = Counter(int(base_labels[rr, cc]) for rr, cc in orbit)
            if 0 in counts:
                continue
            if len(counts) == 1 and center_label in counts:
                central.append({"orbit": orbit, "counts": counts, "per_region": counts[center_label]})
                continue
            if center_label in counts:
                continue
            if set(counts) != set(quadrant_labels):
                continue
            per_quad = {counts[label] for label in quadrant_labels}
            if len(per_quad) != 1:
                continue
            outer.append({"orbit": orbit, "counts": counts, "per_region": next(iter(per_quad))})
    outer.sort(key=lambda item: (item["per_region"], len(item["orbit"]), item["orbit"]))
    central.sort(key=lambda item: (item["per_region"], len(item["orbit"]), item["orbit"]))
    return outer, central


def _place_orbit(grid: np.ndarray, orbit: tuple[tuple[int, int], ...], color: int) -> None:
    for r, c in orbit:
        grid[r, c] = color


def _candidate_combo(
    wall_grid: np.ndarray,
    outer: list[dict],
    central: list[dict],
    rng: random.Random,
) -> tuple[dict, dict, dict, dict]:
    outer_one = [item for item in outer if item["per_region"] == 1]
    outer_two = [item for item in outer if item["per_region"] == 2]
    assert outer_one and outer_two and central
    rng.shuffle(outer_one)
    rng.shuffle(outer_two)
    central_pool = central[:]
    rng.shuffle(central_pool)
    for loser in outer_two:
        loser_cells = set(loser["orbit"])
        for winner_one in outer_one:
            if loser_cells & set(winner_one["orbit"]):
                continue
            used = loser_cells | set(winner_one["orbit"])
            for winner_two in outer_two:
                if used & set(winner_two["orbit"]):
                    continue
                used2 = used | set(winner_two["orbit"])
                for center in central_pool:
                    if used2 & set(center["orbit"]):
                        continue
                    trial = wall_grid.copy()
                    _place_orbit(trial, loser["orbit"], RED)
                    _place_orbit(trial, winner_one["orbit"], GREEN)
                    _place_orbit(trial, winner_two["orbit"], GREEN)
                    _place_orbit(trial, center["orbit"], YELLOW)
                    labels, regions = _label_black_regions(trial)
                    if len(regions) != 5:
                        continue
                    if any(len(region) == 0 for region in regions):
                        continue
                    return loser, winner_one, winner_two, center
    raise RuntimeError("failed to find a non-degenerate orbit combination")


def _build_full_grid(
    wall_grid: np.ndarray,
    loser_orbit: dict,
    winner_one: dict,
    winner_two: dict,
    center_orbit: dict,
    rng: random.Random,
) -> tuple[np.ndarray, int, int, int]:
    winner_color, loser_color, center_color = rng.sample(list(MARKER_COLORS), 3)
    full = wall_grid.copy()
    _place_orbit(full, loser_orbit["orbit"], loser_color)
    _place_orbit(full, winner_one["orbit"], winner_color)
    _place_orbit(full, winner_two["orbit"], winner_color)
    _place_orbit(full, center_orbit["orbit"], center_color)
    return full, winner_color, loser_color, center_color


def _mask_region_cells(
    grid: np.ndarray,
    orbit: tuple[tuple[int, int], ...],
    base_labels: np.ndarray,
    target_labels: set[int],
) -> None:
    for r, c in orbit:
        if int(base_labels[r, c]) in target_labels:
            grid[r, c] = GRAY


def _region_masks(full_grid: np.ndarray, base_labels: np.ndarray, region_labels: list[int]) -> dict[int, np.ndarray]:
    masks: dict[int, np.ndarray] = {}
    for label in region_labels:
        coords = np.argwhere((base_labels == int(label)) & (full_grid == 0))
        assert len(coords) > 0
        masks[int(label)] = coords
    return masks


def _votes_on_region_masks(grid: np.ndarray, region_masks: dict[int, np.ndarray]) -> dict[int, Counter]:
    out: dict[int, Counter] = {}
    for label, coords in region_masks.items():
        seen: set[tuple[int, int]] = set()
        counts: Counter = Counter()
        for r, c in coords:
            for dr, dc in DIR8:
                nr, nc = int(r) + dr, int(c) + dc
                if not _in_bounds(grid.shape, nr, nc):
                    continue
                color = int(grid[nr, nc])
                if color in MARKER_COLORS and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    counts[color] += 1
        out[int(label)] = counts
    return out


def _orbit_visible_colors(grid: np.ndarray) -> None:
    n = grid.shape[0]
    seen: set[tuple[tuple[int, int], ...]] = set()
    for r in range(n):
        for c in range(n):
            orbit = _d4_orbit(n, r, c)
            if orbit in seen:
                continue
            seen.add(orbit)
            colors = {int(grid[rr, cc]) for rr, cc in orbit if int(grid[rr, cc]) != GRAY}
            assert len(colors) == 1, "every orbit must expose a unique visible non-grey color"


def _semantic_check(
    inp: np.ndarray,
    full_grid: np.ndarray,
    base_labels: np.ndarray,
    quadrant_labels: list[int],
    no_visible_label: int,
    changed_visible_labels: set[int],
    winner_color: int,
) -> None:
    completed = _complete_symmetry(inp)
    assert np.array_equal(completed, full_grid), "symmetry completion must recover the intended full grid"
    _orbit_visible_colors(inp)
    solved = main(inp)
    solved_full = _solve_completed(full_grid)
    assert np.array_equal(solved, solved_full), "masked and completed paths must agree"

    region_masks = _region_masks(full_grid, base_labels, quadrant_labels)
    visible_votes = _votes_on_region_masks(inp, region_masks)
    full_votes = _votes_on_region_masks(full_grid, region_masks)

    changed = []
    no_visible = []
    for label in quadrant_labels:
        full_counts = full_votes[label]
        assert _strict_majority_color(full_counts) == winner_color
        vis_counts = visible_votes[label]
        if not vis_counts:
            no_visible.append(label)
            continue
        if _strict_majority_color(vis_counts) != winner_color:
            changed.append(label)

    assert no_visible_label in no_visible and len(no_visible) >= 1
    assert changed_visible_labels.issubset(set(changed))
    assert len(changed) >= 2


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.choice([13, 15])
    ring_half = rng.choice([2, 3] if n == 15 else [2])
    wall_grid, base_labels, _regions, center_label, quadrant_labels = _build_wall_layout(n, ring_half)
    outer, central = _enumerate_orbits(wall_grid, base_labels, center_label, quadrant_labels)
    loser_orbit, winner_one, winner_two, center_orbit = _candidate_combo(wall_grid, outer, central, rng)
    full_grid, winner_color, loser_color, _center_color = _build_full_grid(
        wall_grid, loser_orbit, winner_one, winner_two, center_orbit, rng
    )

    inp = full_grid.copy()
    untouched_label = rng.choice(quadrant_labels)
    masked_labels = [label for label in quadrant_labels if label != untouched_label]
    no_visible_label = rng.choice(masked_labels)
    changed_visible_labels = set(label for label in masked_labels if label != no_visible_label)

    _mask_region_cells(inp, winner_one["orbit"], base_labels, set(masked_labels))
    _mask_region_cells(inp, winner_two["orbit"], base_labels, set(masked_labels))
    _mask_region_cells(inp, loser_orbit["orbit"], base_labels, {no_visible_label})

    center_cells = list(center_orbit["orbit"])
    rng.shuffle(center_cells)
    extra_center_masks = rng.randint(0, max(0, len(center_cells) - 1))
    for r, c in center_cells[:extra_center_masks]:
        inp[r, c] = GRAY

    try:
        _semantic_check(
            inp,
            full_grid,
            base_labels,
            quadrant_labels,
            no_visible_label,
            changed_visible_labels,
            winner_color,
        )
    except AssertionError:
        return None
    assert winner_color != loser_color
    return inp


def generate_input() -> np.ndarray:
    for seed in range(GEN_ATTEMPTS):
        rng = random.Random(random.randrange(1 << 30) + seed)
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
