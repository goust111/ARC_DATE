from __future__ import annotations

from collections import Counter

import numpy as np


BLACK = 0
WALL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _detect_wall_color(grid: np.ndarray) -> int:
    colors = [int(color) for color in np.unique(grid)]
    if WALL in colors:
        return WALL
    non_black = [color for color in colors if color != BLACK]
    assert non_black, "expected a visible wall color"
    return max(non_black, key=lambda color: int(np.count_nonzero(grid == color)))


def _extract_structure(grid: np.ndarray, wall_color: int | None = None):
    g = np.asarray(grid, dtype=int)
    wall = _detect_wall_color(g) if wall_color is None else int(wall_color)
    h, w = g.shape

    region_ids = np.full((h, w), -1, dtype=int)
    regions: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if int(g[r, c]) == wall or region_ids[r, c] != -1:
                continue
            rid = len(regions)
            stack = [(int(r), int(c))]
            region_ids[r, c] = rid
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and region_ids[nr, nc] == -1 and int(g[nr, nc]) != wall:
                        region_ids[nr, nc] = rid
                        stack.append((nr, nc))
            regions.append(coords)

    shared_lengths: Counter[tuple[int, int]] = Counter()
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != wall:
                continue

            if 0 < r < h - 1:
                a = int(region_ids[r - 1, c])
                b = int(region_ids[r + 1, c])
                if a >= 0 and b >= 0 and a != b:
                    shared_lengths[tuple(sorted((a, b)))] += 1

            if 0 < c < w - 1:
                a = int(region_ids[r, c - 1])
                b = int(region_ids[r, c + 1])
                if a >= 0 and b >= 0 and a != b:
                    shared_lengths[tuple(sorted((a, b)))] += 1

    return wall, regions, region_ids, dict(shared_lengths)


def _analyze_grid(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    wall, regions, region_ids, shared_lengths = _extract_structure(g)

    votes_by_region: list[Counter[int]] = []
    resolved: dict[int, int] = {}
    tie_regions: list[tuple[int, tuple[int, int]]] = []
    color_totals: Counter[int] = Counter()

    for rid, coords in enumerate(regions):
        votes: Counter[int] = Counter()
        for r, c in coords:
            color = int(g[r, c])
            if color not in (BLACK, wall):
                votes[color] += 1
                color_totals[color] += 1
        assert votes, "every region must contain at least one colored vote"

        max_count = max(votes.values())
        leaders = sorted(color for color, count in votes.items() if count == max_count)
        if len(leaders) == 1:
            resolved[rid] = int(leaders[0])
        else:
            assert len(leaders) == 2, "the only allowed tie is a two-color deadlock"
            tie_regions.append((rid, (int(leaders[0]), int(leaders[1]))))
        votes_by_region.append(votes)

    assert len(tie_regions) == 1, "expected exactly one tied region"
    tie_region, tie_colors = tie_regions[0]

    neighbor_lengths: dict[int, int] = {}
    for (a, b), length in shared_lengths.items():
        if a == tie_region:
            neighbor_lengths[int(b)] = int(length)
        elif b == tie_region:
            neighbor_lengths[int(a)] = int(length)

    assert len(neighbor_lengths) >= 2, "the tied region must have multiple neighbors"
    max_length = max(neighbor_lengths.values())
    longest_neighbors = sorted(nid for nid, length in neighbor_lengths.items() if length == max_length)
    assert len(longest_neighbors) == 1, "shared wall length must have a unique maximum"
    borrowed_from = int(longest_neighbors[0])
    assert borrowed_from in resolved, "the longest-wall neighbor must have a unique majority color"
    borrowed_color = int(resolved[borrowed_from])

    final_colors = {rid: color for rid, color in resolved.items()}
    final_colors[tie_region] = borrowed_color

    return {
        "wall": wall,
        "regions": regions,
        "region_ids": region_ids,
        "shared_lengths": shared_lengths,
        "votes_by_region": votes_by_region,
        "resolved": resolved,
        "tie_region": tie_region,
        "tie_colors": tie_colors,
        "neighbor_lengths": neighbor_lengths,
        "borrowed_from": borrowed_from,
        "borrowed_color": borrowed_color,
        "final_colors": final_colors,
        "region_areas": [len(coords) for coords in regions],
        "color_totals": color_totals,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each wall-bounded region by local majority, except the sole tied region which borrows from its longest-wall neighbor.

    Input structure:
    - Black floor cells are partitioned by one-cell-thick grey walls into 4-connected regions.
    - Each region contains scattered colored noise pixels and otherwise black cells.
    - Exactly one region has a two-color tie for the highest local count, while every other region has a unique local majority.

    Rule:
    - For every non-tied region, count only its non-black, non-wall pixels and fill the whole region with that majority color.
    - For the unique tied region, do not break the tie internally and do not use global color frequency.
    - Measure which neighboring resolved region shares the longest grey wall segment with the tied region, and fill the tied region with that neighbor's resolved color.
    - Keep every grey wall cell unchanged.

    Key insight:
    - The deadlocked region is decided by geometry rather than votes, so the solver must first resolve the other regions and then measure shared wall length in the wall-defined coordinate frame.
    """
    g = np.asarray(grid, dtype=int)
    analysis = _analyze_grid(g)

    out = np.full_like(g, int(analysis["wall"]))
    for rid, coords in enumerate(analysis["regions"]):
        fill_color = int(analysis["final_colors"][rid])
        for r, c in coords:
            out[r, c] = fill_color
    return out


MIN_GRID_SIZE = 16
MAX_GRID_SIZE = 28
MIN_REGIONS = 4
MAX_REGIONS = 6
MIN_REGION_SIDE = 3
GEN_ATTEMPTS = 500


def _positive_partition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    total = int(total)
    parts = int(parts)
    if parts == 1:
        return [total]
    cuts = np.sort(rng.choice(np.arange(1, total), size=parts - 1, replace=False))
    values = [int(cuts[0])]
    for i in range(1, len(cuts)):
        values.append(int(cuts[i] - cuts[i - 1]))
    values.append(int(total - cuts[-1]))
    return values


def _build_partition(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(MIN_GRID_SIZE, MAX_GRID_SIZE + 1))
    width = int(rng.integers(MIN_GRID_SIZE, MAX_GRID_SIZE + 1))
    target_regions = int(rng.integers(MIN_REGIONS, MAX_REGIONS + 1))

    grid = np.full((height, width), BLACK, dtype=int)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL

    rectangles = [(1, 1, height - 2, width - 2)]
    for _ in range(target_regions - 1):
        options: list[tuple[int, str]] = []
        weights: list[int] = []
        for idx, (r0, c0, r1, c1) in enumerate(rectangles):
            rect_height = r1 - r0 + 1
            rect_width = c1 - c0 + 1
            area = rect_height * rect_width
            if rect_height >= 2 * MIN_REGION_SIDE + 1:
                options.append((idx, "h"))
                weights.append(area)
            if rect_width >= 2 * MIN_REGION_SIDE + 1:
                options.append((idx, "v"))
                weights.append(area)
        if not options:
            return None

        weights_arr = np.asarray(weights, dtype=float)
        weights_arr /= weights_arr.sum()
        chosen = int(rng.choice(len(options), p=weights_arr))
        rect_idx, orient = options[chosen]
        r0, c0, r1, c1 = rectangles.pop(rect_idx)

        if orient == "h":
            wall_row = int(rng.integers(r0 + MIN_REGION_SIDE, r1 - MIN_REGION_SIDE + 1))
            grid[wall_row, c0 : c1 + 1] = WALL
            rectangles.append((r0, c0, wall_row - 1, c1))
            rectangles.append((wall_row + 1, c0, r1, c1))
        else:
            wall_col = int(rng.integers(c0 + MIN_REGION_SIDE, c1 - MIN_REGION_SIDE + 1))
            grid[r0 : r1 + 1, wall_col] = WALL
            rectangles.append((r0, c0, r1, wall_col - 1))
            rectangles.append((r0, wall_col + 1, r1, c1))

    return grid


def _choose_tie_candidate(regions, shared_lengths, rng: np.random.Generator):
    region_areas = [len(coords) for coords in regions]
    candidates = []

    for rid in range(len(regions)):
        neighbor_lengths: dict[int, int] = {}
        for (a, b), length in shared_lengths.items():
            if a == rid:
                neighbor_lengths[int(b)] = int(length)
            elif b == rid:
                neighbor_lengths[int(a)] = int(length)

        if len(neighbor_lengths) < 2 or region_areas[rid] < 10:
            continue

        ordered = sorted(neighbor_lengths.items(), key=lambda item: (-item[1], item[0]))
        if ordered[0][1] == ordered[1][1]:
            continue

        longest_neighbor = int(ordered[0][0])
        anti_area = any(region_areas[nid] > region_areas[longest_neighbor] for nid in neighbor_lengths if nid != longest_neighbor)
        candidates.append(
            {
                "rid": rid,
                "neighbor_lengths": neighbor_lengths,
                "longest_neighbor": longest_neighbor,
                "max_length": int(ordered[0][1]),
                "neighbor_count": len(neighbor_lengths),
                "area": region_areas[rid],
                "anti_area": anti_area,
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (item["anti_area"], item["neighbor_count"], item["max_length"], item["area"]),
        reverse=True,
    )
    top_k = min(3, len(candidates))
    return candidates[int(rng.integers(top_k))]


def _sample_unique_votes(
    area: int,
    majority_color: int,
    allowed_minor_colors: list[int],
    rng: np.random.Generator,
) -> Counter[int] | None:
    max_noise = min(int(area) - 1, 7)
    if max_noise < 3 or not allowed_minor_colors:
        return None

    majority_min = 2 if max_noise == 3 else 3
    majority_max = min(5, max_noise - 1)
    if majority_max < majority_min:
        return None
    majority_count = int(rng.integers(majority_min, majority_max + 1))

    other_cap = min(majority_count - 1, max_noise - majority_count)
    if other_cap < 1:
        return None
    other_total = int(rng.integers(1, other_cap + 1))
    n_minors = min(len(allowed_minor_colors), other_total, 1 + int(other_total >= 2 and rng.random() < 0.55))
    if n_minors <= 0:
        return None

    minor_counts = _positive_partition(other_total, n_minors, rng)
    minor_colors = [int(color) for color in rng.choice(np.asarray(allowed_minor_colors, dtype=int), size=n_minors, replace=False)]

    votes: Counter[int] = Counter({int(majority_color): majority_count})
    for color, count in zip(minor_colors, minor_counts):
        votes[int(color)] += int(count)
    return votes


def _sample_tie_votes(
    area: int,
    tie_colors: tuple[int, int],
    allowed_minor_colors: list[int],
    rng: np.random.Generator,
) -> Counter[int] | None:
    max_noise = min(int(area) - 1, 8)
    max_tie_count = min(4, max_noise // 2)
    if max_tie_count < 2:
        return None

    tie_count = int(rng.integers(2, max_tie_count + 1))
    remaining = max_noise - 2 * tie_count
    votes: Counter[int] = Counter({int(tie_colors[0]): tie_count, int(tie_colors[1]): tie_count})

    if remaining > 0 and allowed_minor_colors and tie_count > 2 and rng.random() < 0.6:
        extra = int(rng.integers(1, min(2, tie_count - 1, remaining) + 1))
        votes[int(rng.choice(np.asarray(allowed_minor_colors, dtype=int)))] += extra

    return votes


def _boost_global_tie_color(
    vote_maps: dict[int, Counter[int]],
    non_tie_regions: list[int],
    regions,
    majority_colors: dict[int, int],
    dominant_color: int,
    other_tie_color: int,
    rng: np.random.Generator,
) -> bool:
    for _ in range(4):
        dominant_total = sum(votes.get(dominant_color, 0) for votes in vote_maps.values())
        other_total = sum(votes.get(other_tie_color, 0) for votes in vote_maps.values())
        if dominant_total > other_total:
            return True

        candidates = []
        for rid in non_tie_regions:
            votes = vote_maps[rid]
            majority_count = int(votes[int(majority_colors[rid])])
            used = sum(votes.values())
            dominant_here = int(votes.get(dominant_color, 0))
            if used + 1 > len(regions[rid]) - 1:
                continue
            if dominant_here + 1 >= majority_count:
                continue
            candidates.append(rid)

        if not candidates:
            return False
        chosen = int(rng.choice(np.asarray(candidates, dtype=int)))
        vote_maps[chosen][int(dominant_color)] += 1

    dominant_total = sum(votes.get(dominant_color, 0) for votes in vote_maps.values())
    other_total = sum(votes.get(other_tie_color, 0) for votes in vote_maps.values())
    return dominant_total > other_total


def _paint_votes(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    votes: Counter[int],
    rng: np.random.Generator,
) -> None:
    total = sum(votes.values())
    assert total <= len(coords) - 1, "each region should keep visible black cells"

    indices = rng.choice(len(coords), size=total, replace=False)
    payload: list[int] = []
    for color, count in votes.items():
        payload.extend([int(color)] * int(count))
    rng.shuffle(payload)

    for idx, color in zip(indices.tolist(), payload):
        r, c = coords[int(idx)]
        grid[r, c] = int(color)


def _render_output(shape: tuple[int, int], regions, final_colors: dict[int, int]) -> np.ndarray:
    out = np.full(shape, WALL, dtype=int)
    for rid, coords in enumerate(regions):
        color = int(final_colors[rid])
        for r, c in coords:
            out[r, c] = color
    return out


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    tie_region: int,
    tie_colors: tuple[int, int],
    majority_colors: dict[int, int],
    candidate,
    challenge_global: bool,
    dominant_tie_color: int,
) -> bool:
    try:
        analysis = _analyze_grid(inp)
    except AssertionError:
        return False

    if analysis["tie_region"] != tie_region:
        return False
    if tuple(sorted(analysis["tie_colors"])) != tuple(sorted(tie_colors)):
        return False
    if analysis["borrowed_from"] != int(candidate["longest_neighbor"]):
        return False
    if analysis["neighbor_lengths"] != candidate["neighbor_lengths"]:
        return False

    for rid, color in majority_colors.items():
        if analysis["resolved"].get(rid) != int(color):
            return False

    borrowed_color = int(majority_colors[int(candidate["longest_neighbor"])])
    if analysis["borrowed_color"] != borrowed_color:
        return False
    if borrowed_color in tie_colors:
        return False

    if challenge_global:
        other_tie_color = tie_colors[0] if dominant_tie_color == tie_colors[1] else tie_colors[1]
        if analysis["color_totals"][int(dominant_tie_color)] <= analysis["color_totals"][int(other_tie_color)]:
            return False

    return np.array_equal(main(inp), expected)


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    base = _build_partition(rng)
    if base is None:
        return None

    wall, regions, _region_ids, shared_lengths = _extract_structure(base, WALL)
    if wall != WALL:
        return None

    candidate = _choose_tie_candidate(regions, shared_lengths, rng)
    if candidate is None:
        return None

    tie_region = int(candidate["rid"])
    non_tie_regions = [rid for rid in range(len(regions)) if rid != tie_region]
    if len(non_tie_regions) + 2 > len(PALETTE):
        return None

    color_order = [int(color) for color in rng.permutation(np.asarray(PALETTE, dtype=int)).tolist()]
    tie_colors = (int(color_order[0]), int(color_order[1]))
    majority_colors = {
        rid: int(color_order[idx + 2])
        for idx, rid in enumerate(non_tie_regions)
    }

    challenge_global = bool(rng.random() < 0.7)
    dominant_tie_color = tie_colors[int(rng.integers(2))]

    vote_maps: dict[int, Counter[int]] = {}
    for rid in non_tie_regions:
        if challenge_global:
            allowed_minors = [color for color in PALETTE if color not in (majority_colors[rid], tie_colors[0], tie_colors[1])]
        else:
            allowed_minors = [color for color in PALETTE if color != majority_colors[rid]]
        votes = _sample_unique_votes(len(regions[rid]), int(majority_colors[rid]), allowed_minors, rng)
        if votes is None:
            return None
        vote_maps[rid] = votes

    tie_minor_colors = [color for color in PALETTE if color not in tie_colors]
    tie_votes = _sample_tie_votes(len(regions[tie_region]), tie_colors, tie_minor_colors, rng)
    if tie_votes is None:
        return None
    vote_maps[tie_region] = tie_votes

    if challenge_global:
        other_tie_color = tie_colors[0] if dominant_tie_color == tie_colors[1] else tie_colors[1]
        if not _boost_global_tie_color(
            vote_maps,
            non_tie_regions,
            regions,
            majority_colors,
            int(dominant_tie_color),
            int(other_tie_color),
            rng,
        ):
            return None

    inp = base.copy()
    for rid, coords in enumerate(regions):
        _paint_votes(inp, coords, vote_maps[rid], rng)

    final_colors = {rid: color for rid, color in majority_colors.items()}
    final_colors[tie_region] = int(majority_colors[int(candidate["longest_neighbor"])])
    expected = _render_output(inp.shape, regions, final_colors)

    if not _validate_instance(
        inp,
        expected,
        tie_region,
        tie_colors,
        majority_colors,
        candidate,
        challenge_global,
        int(dominant_tie_color),
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tied-majority region instance")
