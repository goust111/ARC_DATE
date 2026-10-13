from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

NONBLACK_TILE_COLORS = (BLUE, RED, GREEN, YELLOW)
MAX_GENERATION_ATTEMPTS = 500


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _tile_grid(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    pr, pc = map(int, tile.shape)
    out = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r, c] = int(tile[r % pr, c % pc])
    return out


def _candidate_periods(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    candidates: list[dict] = []

    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            phase_counts = [[Counter() for _ in range(pc)] for _ in range(pr)]
            total_seen = 0
            for r in range(h):
                for c in range(w):
                    color = int(g[r, c])
                    if color == PINK:
                        continue
                    phase_counts[r % pr][c % pc][color] += 1
                    total_seen += 1

            tile = np.full((pr, pc), -1, dtype=int)
            agree = 0
            valid = True
            for rr in range(pr):
                for cc in range(pc):
                    counts = phase_counts[rr][cc]
                    if not counts:
                        valid = False
                        break
                    phase_total = sum(counts.values())
                    best_count = max(counts.values())
                    winners = [color for color, count in counts.items() if count == best_count]
                    if len(winners) != 1 or best_count * 2 <= phase_total:
                        valid = False
                        break
                    tile[rr, cc] = winners[0]
                    agree += best_count
                if not valid:
                    break

            if valid:
                candidates.append(
                    {
                        "period": (pr, pc),
                        "tile": tile,
                        "agree": agree,
                        "area": pr * pc,
                    }
                )

    return candidates


def _infer_period_and_tile(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    candidates = _candidate_periods(grid)
    assert candidates, "expected a periodic wallpaper with a unique majority tile"
    candidates.sort(key=lambda item: (-item["agree"], item["area"], item["period"][0], item["period"][1]))
    best = candidates[0]
    ambiguous = [
        cand
        for cand in candidates
        if cand["agree"] == best["agree"] and cand["area"] == best["area"]
    ]
    assert len(ambiguous) == 1, "expected a unique smallest best-scoring period"
    pr, pc = best["period"]
    return int(pr), int(pc), best["tile"].copy()


def _seed_connected_anomalies(anomaly: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    out = np.zeros_like(anomaly, dtype=bool)
    seen = {seed}
    q = deque([seed])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not in_bounds(anomaly.shape, nr, nc) or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            if anomaly[nr, nc]:
                out[nr, nc] = True
                q.append((nr, nc))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the repeating tile, isolate mismatched cells, then recolor only the seed-reachable anomaly region to magenta while repairing the rest.

    Input structure:
    - The grid is a periodic wallpaper built from a small 3x3 or 4x4 tile using black plus 2-3 colors from {blue, red, green, yellow}.
    - A single magenta seed pixel is not part of the true tile.
    - Some non-seed cells are corrupted: their colors differ from the correct tile color at the same phase, and these corrupted cells form 1-3 4-connected clusters.

    Rule:
    - Infer the smallest repeated tile by testing candidate row/column periods and taking the unique best-scoring majority color for each phase, ignoring only the magenta seed.
    - Mark every non-seed cell whose color disagrees with that phase's inferred tile color as an anomaly.
    - Starting from the magenta seed, flood only through 4-neighbor anomaly cells.
    - Recolor the reached anomaly cells to magenta and repair every unreached anomaly cell back to its inferred tile color; keep all other cells unchanged.

    Key insight:
    - The task is not just wallpaper repair: the inferred periodic tile defines which cells are anomalous, and only that anomaly mask reveals which corrupted region should be diverted into the seed color instead of being restored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    seeds = np.argwhere(g == PINK)
    assert len(seeds) == 1, "expected exactly one magenta seed"
    seed = tuple(map(int, seeds[0]))

    _pr, _pc, tile = _infer_period_and_tile(g)
    correct = _tile_grid(tile, g.shape)
    anomaly = (g != PINK) & (g != correct)
    seed_connected = _seed_connected_anomalies(anomaly, seed)

    out = g.copy()
    out[seed_connected] = PINK
    repair = anomaly & ~seed_connected
    out[repair] = correct[repair]
    out[seed] = PINK
    return out


def _choose_shape(period: int) -> int:
    repeats = random.choice((4, 5))
    extra = random.randint(0, period - 1)
    return repeats * period + extra


def _random_tile(pr: int, pc: int, shape: tuple[int, int]) -> np.ndarray | None:
    for _ in range(200):
        nonblack_count = random.randint(2, 3)
        palette = list(random.sample(NONBLACK_TILE_COLORS, nonblack_count))
        if random.random() < 0.6:
            palette.append(BLACK)
        random.shuffle(palette)

        tile = np.array(random.choices(palette, k=pr * pc), dtype=int).reshape(pr, pc)
        if len(set(map(int, tile.flat))) < max(2, len(palette) - 1):
            continue

        clean = _tile_grid(tile, shape)
        got_pr, got_pc, got_tile = _infer_period_and_tile(clean)
        if (got_pr, got_pc) != (pr, pc):
            continue
        if not np.array_equal(got_tile, tile):
            continue
        return tile
    return None


def _grow_cluster(shape: tuple[int, int], size: int, blocked: set[tuple[int, int]]) -> set[tuple[int, int]] | None:
    h, w = map(int, shape)
    starts = [(r, c) for r in range(h) for c in range(w) if (r, c) not in blocked]
    random.shuffle(starts)
    for start in starts:
        cluster = {start}
        frontier = {
            (nr, nc)
            for nr, nc in neighbors4(*start)
            if in_bounds(shape, nr, nc) and (nr, nc) not in blocked
        }
        while len(cluster) < size:
            if not frontier:
                break
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cluster.add(cell)
            for nr, nc in neighbors4(*cell):
                if (
                    in_bounds(shape, nr, nc)
                    and (nr, nc) not in cluster
                    and (nr, nc) not in blocked
                ):
                    frontier.add((nr, nc))
        if len(cluster) == size:
            return cluster
    return None


def _blocked_with_margin(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    blocked = set(cells)
    for r, c in cells:
        for nr, nc in neighbors4(r, c):
            blocked.add((nr, nc))
    return blocked


def _pick_seed_inside(cluster: set[tuple[int, int]]) -> tuple[int, int] | None:
    candidates = []
    for cell in cluster:
        rest = set(cluster)
        rest.remove(cell)
        if rest and _is_connected(rest):
            candidates.append(cell)
    return random.choice(candidates) if candidates else None


def _pick_seed_adjacent(
    shape: tuple[int, int],
    target_cluster: set[tuple[int, int]],
    all_clusters: list[set[tuple[int, int]]],
) -> tuple[int, int] | None:
    target_cells = set(target_cluster)
    others = set().union(*(cluster for cluster in all_clusters if cluster is not target_cluster))
    options = []
    h, w = map(int, shape)
    for r in range(h):
        for c in range(w):
            if (r, c) in target_cells or (r, c) in others:
                continue
            hits_target = sum((nr, nc) in target_cells for nr, nc in neighbors4(r, c))
            hits_other = sum((nr, nc) in others for nr, nc in neighbors4(r, c))
            if hits_target >= 1 and hits_other == 0:
                options.append((r, c))
    return random.choice(options) if options else None


def _corrupt_clusters(
    clean: np.ndarray,
    clusters: list[set[tuple[int, int]]],
    target_idx: int,
    seed_mode: str,
) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    inp = clean.copy()
    palette = sorted(set(map(int, clean.flat)))

    for cluster in clusters:
        for r, c in cluster:
            correct = int(clean[r, c])
            choices = [color for color in palette if color != correct]
            inp[r, c] = random.choice(choices)

    target_cluster = clusters[target_idx]
    if seed_mode == "inside":
        seed = _pick_seed_inside(target_cluster)
        if seed is None:
            raise ValueError("unable to place internal seed")
        connected = np.zeros(clean.shape, dtype=bool)
        for r, c in target_cluster:
            if (r, c) != seed:
                connected[r, c] = True
    else:
        seed = _pick_seed_adjacent(clean.shape, target_cluster, clusters)
        if seed is None:
            raise ValueError("unable to place adjacent seed")
        connected = np.zeros(clean.shape, dtype=bool)
        for r, c in target_cluster:
            connected[r, c] = True

    inp[seed] = PINK
    return inp, seed, connected


def _expected_output(clean: np.ndarray, seed: tuple[int, int], seed_connected: np.ndarray, anomaly: np.ndarray) -> np.ndarray:
    out = clean.copy()
    out[seed_connected] = PINK
    out[seed] = PINK
    out[anomaly & ~seed_connected] = clean[anomaly & ~seed_connected]
    return out


def _validate_instance(
    inp: np.ndarray,
    clean: np.ndarray,
    pr: int,
    pc: int,
    expected_connected: np.ndarray,
) -> bool:
    candidates = _candidate_periods(inp)
    if not candidates:
        return False
    candidates.sort(key=lambda item: (-item["agree"], item["area"], item["period"][0], item["period"][1]))
    best = candidates[0]
    if best["period"] != (pr, pc):
        return False
    if sum(cand["agree"] == best["agree"] and cand["area"] == best["area"] for cand in candidates) != 1:
        return False

    seeds = np.argwhere(inp == PINK)
    if len(seeds) != 1:
        return False
    seed = tuple(map(int, seeds[0]))

    _got_pr, _got_pc, tile = _infer_period_and_tile(inp)
    correct = _tile_grid(tile, inp.shape)
    anomaly = (inp != PINK) & (inp != correct)
    if not np.array_equal(correct, clean):
        return False
    if not np.array_equal(_seed_connected_anomalies(anomaly, seed), expected_connected):
        return False

    disconnected = anomaly & ~expected_connected
    if not expected_connected.any() or not disconnected.any():
        return False
    if len(components_mask(anomaly)) < 2:
        return False

    out = main(inp)
    expected = _expected_output(clean, seed, expected_connected, anomaly)
    if not np.array_equal(out, expected):
        return False

    shortcut_fix = clean.copy()
    shortcut_fix[seed] = PINK
    if np.array_equal(out, shortcut_fix):
        return False

    shortcut_pink = inp.copy()
    shortcut_pink[anomaly] = PINK
    shortcut_pink[seed] = PINK
    if np.array_equal(out, shortcut_pink):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    pr = random.choice((3, 4))
    pc = random.choice((3, 4))
    h = _choose_shape(pr)
    w = _choose_shape(pc)
    shape = (h, w)

    tile = _random_tile(pr, pc, shape)
    if tile is None:
        return None
    clean = _tile_grid(tile, shape)

    n_clusters = random.choice((2, 3))
    target_idx = random.randrange(n_clusters)
    seed_mode = random.choice(("inside", "adjacent"))

    sizes = []
    for idx in range(n_clusters):
        if idx == target_idx:
            sizes.append(random.randint(2, 6) if seed_mode == "inside" else random.randint(1, 6))
        else:
            sizes.append(random.randint(1, 5))

    clusters: list[set[tuple[int, int]]] = []
    blocked: set[tuple[int, int]] = set()
    for size in sizes:
        cluster = _grow_cluster(shape, size, blocked)
        if cluster is None:
            return None
        clusters.append(cluster)
        blocked |= _blocked_with_margin(cluster)

    try:
        inp, _seed, expected_connected = _corrupt_clusters(clean, clusters, target_idx, seed_mode)
    except ValueError:
        return None

    if not _validate_instance(inp, clean, pr, pc, expected_connected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-anomaly flood-repair input")
