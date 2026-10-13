import numpy as np


BG = 0
LEGEND_COLORS = (1, 2, 3, 4)
OBSTACLE = 5
SEED = 8
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
MAX_GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _legend_info(grid: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    row = np.asarray(grid[0], dtype=int)
    nz = np.flatnonzero(row != BG)
    assert nz.size > 0, "expected a top-row legend"
    splits = np.where(np.diff(nz) > 1)[0]
    runs = np.split(nz, splits + 1)
    valid: list[tuple[int, int, tuple[int, ...]]] = []
    for run in runs:
        vals = tuple(int(v) for v in row[run[0] : run[-1] + 1])
        if len(run) in (3, 4) and vals == LEGEND_COLORS[: len(run)]:
            valid.append((int(run[0]), int(run[-1]), vals))
    assert len(valid) == 1, "expected exactly one contiguous legend bar"
    return valid[0]


def _obstacle_ids(grid: np.ndarray) -> np.ndarray:
    ids = np.zeros_like(grid, dtype=int)
    for i, coords in enumerate(_components(grid == OBSTACLE), start=1):
        ids[coords[:, 0], coords[:, 1]] = i
    return ids


def _ray_hits(obstacle_ids: np.ndarray, r: int, c: int) -> tuple[int, int, int, int]:
    h, w = obstacle_ids.shape
    hits = []
    for dr, dc in DIAGONALS:
        rr, cc = int(r) + dr, int(c) + dc
        hit = 0
        while 0 <= rr < h and 0 <= cc < w:
            oid = int(obstacle_ids[rr, cc])
            if oid:
                hit = oid
                break
            rr += dr
            cc += dc
        hits.append(hit)
    return tuple(hits)


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    c0, c1, legend = _legend_info(g)
    ids = _obstacle_ids(g)
    out = g.copy()
    out[0, c0 : c1 + 1] = BG
    for r, c in np.argwhere(g == SEED):
        hits = _ray_hits(ids, int(r), int(c))
        count = int(sum(hit > 0 for hit in hits))
        assert 1 <= count <= len(legend), "seed collision count must index the legend"
        out[int(r), int(c)] = int(legend[count - 1])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each teal seed by indexing the top legend with its total diagonal obstacle-hit count.

    Input structure:
    - The background is black.
    - The top row contains one contiguous legend bar of length 3 or 4 whose left-to-right colors are blue, red, green, and optionally yellow.
    - Several grey solid rectangles are scattered below the legend as obstacles.
    - Several teal single-pixel seeds lie on empty cells away from the legend and rectangles.

    Rule:
    - For each teal seed, cast four diagonal rays from the seed: up-left, up-right, down-left, and down-right.
    - Each ray advances cell by cell until it either leaves the grid or first reaches a grey obstacle pixel.
    - Count one collision for a ray iff its stopping reason is hitting grey; sum the four rays to get C.
    - Recolor the seed with the C-th legend color using 1-based indexing, erase the legend to black, and leave the obstacles unchanged.

    Key insight:
    - The obstacle layout determines a small geometric count, and the legend converts that collision count into the final seed color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _oracle_output(g)


def _solid_rect_ok(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return 2 <= h <= 3 and 2 <= w <= 4 and coords.shape[0] == h * w


def _seed_candidates(
    grid: np.ndarray, obstacle_ids: np.ndarray, legend_len: int, *, strict_gap: bool
) -> dict[int, list[dict]]:
    h, w = grid.shape
    groups: dict[int, list[dict]] = {k: [] for k in range(1, legend_len + 1)}
    for r in range(2, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != BG:
                continue
            if strict_gap:
                rr0, rr1 = max(0, r - 1), min(h, r + 2)
                cc0, cc1 = max(0, c - 1), min(w, c + 2)
                neighborhood = grid[rr0:rr1, cc0:cc1]
                if np.any(neighborhood == OBSTACLE):
                    continue
            hits = _ray_hits(obstacle_ids, r, c)
            count = int(sum(hit > 0 for hit in hits))
            if 1 <= count <= legend_len:
                groups[count].append(
                    {
                        "r": int(r),
                        "c": int(c),
                        "count": count,
                        "hits": hits,
                        "sig": tuple(int(hit > 0) for hit in hits),
                    }
                )
    return groups


def _far_enough(candidate: dict, chosen: list[dict]) -> bool:
    return all(
        max(abs(int(candidate["r"]) - int(other["r"])), abs(int(candidate["c"]) - int(other["c"]))) >= 2
        for other in chosen
    )


def _shuffle_list(items: list, rng: np.random.Generator) -> list:
    if not items:
        return []
    order = rng.permutation(len(items))
    return [items[int(i)] for i in order]


def _choose_mandatory(
    groups: dict[int, list[dict]], required_counts: list[int], rng: np.random.Generator
) -> list[dict] | None:
    ordered = sorted(required_counts, key=lambda k: len(groups[k]))

    def rec(i: int, chosen: list[dict]) -> list[dict] | None:
        if i == len(ordered):
            return chosen
        count = ordered[i]
        for cand in _shuffle_list(groups[count], rng):
            if _far_enough(cand, chosen):
                result = rec(i + 1, chosen + [cand])
                if result is not None:
                    return result
        return None

    return rec(0, [])


def _selection_valid(chosen: list[dict], obstacle_count: int, legend_len: int) -> bool:
    if not (3 <= len(chosen) <= 5):
        return False
    counts = [int(item["count"]) for item in chosen]
    if len(set(counts)) < 2:
        return False
    if any(not (1 <= c <= legend_len) for c in counts):
        return False

    used_obstacles = {hit for item in chosen for hit in item["hits"] if hit > 0}
    if used_obstacles != set(range(1, obstacle_count + 1)):
        return False

    if not any(len({hit for hit in item["hits"] if hit > 0}) >= 2 for item in chosen):
        return False

    repeated = {}
    for item in chosen:
        repeated.setdefault(int(item["count"]), set()).add(tuple(item["sig"]))
    if len(chosen) > len(repeated):
        if not any(len(sigs) >= 2 for sigs in repeated.values()):
            return False

    return True


def _choose_seeds(
    groups: dict[int, list[dict]], obstacle_count: int, legend_len: int, rng: np.random.Generator
) -> list[dict] | None:
    available = [count for count in range(1, legend_len + 1) if groups[count]]
    if len(available) < 2:
        return None

    for _ in range(120):
        if legend_len == 4 and len(available) == 4 and float(rng.random()) < 0.55:
            distinct_target = 4
        elif len(available) >= 3 and float(rng.random()) < 0.7:
            distinct_target = 3
        else:
            distinct_target = 2
        distinct_target = min(distinct_target, len(available))

        preferred = []
        for value in (1, legend_len):
            if value in available and value not in preferred:
                preferred.append(value)
        mids = [value for value in available if value not in preferred]
        mids = _shuffle_list(mids, rng)
        needed = preferred + mids
        required_counts = needed[:distinct_target]
        if len(required_counts) < distinct_target:
            continue

        base = _choose_mandatory(groups, required_counts, rng)
        if base is None:
            continue

        target_seed_count = int(rng.integers(max(3, distinct_target), 6))
        chosen = list(base)
        used_positions = {(int(item["r"]), int(item["c"])) for item in chosen}
        extras_pool = []
        for count in available:
            for cand in groups[count]:
                pos = (int(cand["r"]), int(cand["c"]))
                if pos not in used_positions:
                    extras_pool.append(cand)

        while len(chosen) < target_seed_count:
            feasible = [cand for cand in extras_pool if _far_enough(cand, chosen)]
            if not feasible:
                break

            signatures_by_count: dict[int, set[tuple[int, ...]]] = {}
            covered = {hit for item in chosen for hit in item["hits"] if hit > 0}
            scored = []
            for item in chosen:
                signatures_by_count.setdefault(int(item["count"]), set()).add(tuple(item["sig"]))
            for cand in feasible:
                score = 0
                new_hits = {hit for hit in cand["hits"] if hit > 0} - covered
                score += 4 * len(new_hits)
                known = signatures_by_count.get(int(cand["count"]), set())
                if known and tuple(cand["sig"]) not in known:
                    score += 2
                elif int(cand["count"]) not in signatures_by_count:
                    score += 1
                score += int(cand["count"])
                score += float(rng.random()) * 0.1
                scored.append((score, cand))
            scored.sort(key=lambda item: item[0], reverse=True)
            chosen.append(scored[0][1])
            extras_pool = [cand for cand in extras_pool if cand is not scored[0][1]]

        if _selection_valid(chosen, obstacle_count, legend_len):
            return chosen
    return None


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        c0, c1, legend = _legend_info(grid)
    except AssertionError:
        return False
    legend_len = len(legend)
    if c1 - c0 + 1 != legend_len:
        return False

    allowed = {BG, OBSTACLE, SEED, *legend}
    if not set(map(int, np.unique(grid))).issubset(allowed):
        return False

    grey_comps = _components(grid == OBSTACLE)
    if not (3 <= len(grey_comps) <= 5):
        return False
    if not all(_solid_rect_ok(coords) for coords in grey_comps):
        return False

    seeds = np.argwhere(grid == SEED)
    if not (3 <= len(seeds) <= 5):
        return False

    ids = _obstacle_ids(grid)
    counts = []
    signatures = {}
    used_obstacles = set()
    multi_obstacle_seed = False
    for r, c in seeds:
        hits = _ray_hits(ids, int(r), int(c))
        count = int(sum(hit > 0 for hit in hits))
        if not (1 <= count <= legend_len):
            return False
        counts.append(count)
        signatures.setdefault(count, set()).add(tuple(int(hit > 0) for hit in hits))
        hit_set = {hit for hit in hits if hit > 0}
        used_obstacles |= hit_set
        if len(hit_set) >= 2:
            multi_obstacle_seed = True

    if len(set(counts)) < 2:
        return False
    if used_obstacles != set(range(1, len(grey_comps) + 1)):
        return False
    if not multi_obstacle_seed:
        return False
    if len(seeds) > len(signatures) and not any(len(sigs) >= 2 for sigs in signatures.values()):
        return False

    try:
        expected = _oracle_output(grid)
        actual = main(grid)
    except AssertionError:
        return False
    return bool(np.array_equal(actual, expected))


def _place_obstacles(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    h, w = grid.shape
    count = int(rng.integers(3, 6))
    ids = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    blocked[0, :] = True

    for oid in range(1, count + 1):
        placed = False
        for _ in range(160):
            oh = int(rng.integers(2, 4))
            ow = int(rng.integers(2, 5))
            r0 = int(rng.integers(2, h - oh + 1))
            c0 = int(rng.integers(0, w - ow + 1))
            rr0, rr1 = max(1, r0 - 1), min(h, r0 + oh + 1)
            cc0, cc1 = max(0, c0 - 1), min(w, c0 + ow + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            grid[r0 : r0 + oh, c0 : c0 + ow] = OBSTACLE
            ids[r0 : r0 + oh, c0 : c0 + ow] = oid
            blocked[r0 : r0 + oh, c0 : c0 + ow] = True
            placed = True
            break
        if not placed:
            return None

    coords = np.argwhere(ids > 0)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    if (int(r1 - r0 + 1) < max(5, h // 3)) or (int(c1 - c0 + 1) < max(6, w // 3)):
        return None
    return ids


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(15, 23))
    w = int(rng.integers(16, 25))
    legend_len = 4 if float(rng.random()) < 0.55 else 3

    grid = np.zeros((h, w), dtype=int)
    legend_c0 = int(rng.integers(0, w - legend_len + 1))
    grid[0, legend_c0 : legend_c0 + legend_len] = np.array(LEGEND_COLORS[:legend_len], dtype=int)

    obstacle_ids = _place_obstacles(grid, rng)
    if obstacle_ids is None:
        return None

    groups = _seed_candidates(grid, obstacle_ids, legend_len, strict_gap=True)
    if sum(bool(groups[k]) for k in groups) < 2:
        groups = _seed_candidates(grid, obstacle_ids, legend_len, strict_gap=False)
    seeds = _choose_seeds(groups, int(obstacle_ids.max()), legend_len, rng)
    if seeds is None:
        return None

    for seed in seeds:
        grid[int(seed["r"]), int(seed["c"])] = SEED

    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
