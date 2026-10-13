import numpy as np
from collections import Counter


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GREY = 5
GEN_RETRIES = 400
PATH_TRIES = 240
LAYOUT_TRIES = 120


def _most_common_color(values: np.ndarray) -> int:
    return Counter(map(int, values.reshape(-1))).most_common(1)[0][0]


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _ordered_path(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    coord_set = set(coords)
    neighbors = {}
    endpoints = []
    for cell in coords:
        cell_neighbors = [n for n in _neighbors4(cell[0], cell[1], 10**9, 10**9) if n in coord_set]
        if len(cell_neighbors) > 2:
            raise AssertionError("path component branches")
        neighbors[cell] = cell_neighbors
        if len(cell_neighbors) == 1:
            endpoints.append(cell)
    if len(endpoints) != 2:
        raise AssertionError("path component must have exactly two endpoints")

    start = min(endpoints)
    ordered = [start]
    prev = None
    cur = start
    while True:
        nxt = [cell for cell in neighbors[cur] if cell != prev]
        if not nxt:
            break
        if len(nxt) != 1:
            raise AssertionError("path traversal is ambiguous")
        prev, cur = cur, nxt[0]
        ordered.append(cur)
    if len(ordered) != len(coords):
        raise AssertionError("path component must be a single simple path")
    return ordered


def _layout_features(length: int, seed_positions: list[int]) -> dict[str, bool]:
    seeds = {pos: idx for idx, pos in enumerate(seed_positions, start=1)}
    claims = [[] for _ in range(length)]
    overlap = False
    end_trunc = False
    seed_block = False
    for idx, pos in enumerate(seed_positions, start=1):
        for direction in (-1, 1):
            for step in range(1, idx + 1):
                q = pos + direction * step
                if q < 0 or q >= length:
                    end_trunc = True
                    break
                if q in seeds:
                    seed_block = True
                    break
                claims[q].append(idx)
    for pos in range(length):
        if pos not in seeds and len(claims[pos]) >= 2:
            overlap = True
            break
    uncovered = any(pos not in seeds and not claims[pos] for pos in range(length))
    return {
        "overlap": overlap,
        "end_trunc": end_trunc,
        "seed_block": seed_block,
        "uncovered": uncovered,
    }


def _choose_seed_positions(length: int, scenario: str, rng: np.random.Generator) -> list[int] | None:
    if scenario == "overlap":
        if length < 6:
            return None
        for _ in range(LAYOUT_TRIES):
            gap = int(rng.choice([2, 3]))
            low = 1 if length >= gap + 4 else 0
            high = length - gap - 2
            if high < low:
                continue
            first = int(rng.integers(low, high + 1))
            positions = [first, first + gap]
            if length >= 12 and rng.random() < 0.5:
                extra_low = positions[-1] + 4
                extra_high = length - 2
                if extra_low <= extra_high:
                    positions.append(int(rng.integers(extra_low, extra_high + 1)))
            features = _layout_features(length, positions)
            if features["overlap"] and features["seed_block"]:
                return positions
        return None

    if scenario == "truncate":
        if length < 6:
            return None
        for _ in range(LAYOUT_TRIES):
            left = int(rng.choice([0, 1]))
            right = int(rng.choice([length - 2, length - 1]))
            if right - left <= 3:
                continue
            positions = sorted([left, right])
            features = _layout_features(length, positions)
            if features["end_trunc"] and not features["overlap"]:
                return positions
        return None

    if scenario == "separate":
        if length < 8:
            return None
        for _ in range(LAYOUT_TRIES):
            use_three = length >= 15 and rng.random() < 0.5
            if use_three:
                p1_max = length - 14
                if p1_max < 1:
                    continue
                p1 = int(rng.integers(1, p1_max + 1))
                p2_low = p1 + 4
                p2_high = length - 9
                if p2_low > p2_high:
                    continue
                p2 = int(rng.integers(p2_low, p2_high + 1))
                p3_low = p2 + 6
                p3_high = length - 4
                if p3_low > p3_high:
                    continue
                positions = [p1, p2, int(rng.integers(p3_low, p3_high + 1))]
            else:
                p1_max = length - 7
                if p1_max < 1:
                    continue
                p1 = int(rng.integers(1, p1_max + 1))
                p2_low = p1 + 4
                p2_high = length - 3
                if p2_low > p2_high:
                    continue
                positions = [p1, int(rng.integers(p2_low, p2_high + 1))]
            features = _layout_features(length, positions)
            if not features["overlap"] and not features["seed_block"] and not features["end_trunc"]:
                return positions
        return None

    raise ValueError(f"unknown scenario: {scenario}")


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        dr1 = path[i][0] - path[i - 1][0]
        dc1 = path[i][1] - path[i - 1][1]
        dr2 = path[i + 1][0] - path[i][0]
        dc2 = path[i + 1][1] - path[i][1]
        if (dr1, dc1) != (dr2, dc2):
            turns += 1
    return turns


def _is_valid_extension(
    cell: tuple[int, int],
    prev: tuple[int, int],
    used: set[tuple[int, int]],
    reserved: np.ndarray,
) -> bool:
    h, w = reserved.shape
    r, c = cell
    if not (0 <= r < h and 0 <= c < w):
        return False
    if reserved[r, c] or cell in used:
        return False
    attached = 0
    for nr, nc in _neighbors4(r, c, h, w):
        if (nr, nc) in used:
            if (nr, nc) != prev:
                return False
            attached += 1
    return attached == 1


def _future_option_count(
    cell: tuple[int, int],
    used: set[tuple[int, int]],
    reserved: np.ndarray,
) -> int:
    h, w = reserved.shape
    total = 0
    for nr, nc in _neighbors4(cell[0], cell[1], h, w):
        if _is_valid_extension((nr, nc), cell, used, reserved):
            total += 1
    return total


def _build_path(
    h: int,
    w: int,
    reserved: np.ndarray,
    min_len: int,
    max_len: int,
    want_turn: bool,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    free_cells = [tuple(map(int, cell)) for cell in np.argwhere(~reserved)]
    if not free_cells:
        return None

    for _ in range(PATH_TRIES):
        start = free_cells[int(rng.integers(len(free_cells)))]
        target = int(rng.integers(min_len, max_len + 1))
        path = [start]
        used = {start}
        last_dir = None

        while len(path) < target:
            cur = path[-1]
            candidates = []
            for dr, dc in DIR4:
                nxt = (cur[0] + dr, cur[1] + dc)
                if not _is_valid_extension(nxt, cur, used, reserved):
                    continue
                used_with_next = used | {nxt}
                future = _future_option_count(nxt, used_with_next, reserved)
                turns = 1 if last_dir is not None and (dr, dc) != last_dir else 0
                score = future * 3 + turns * (3 if want_turn else 1) + rng.random()
                candidates.append((score, nxt, (dr, dc)))

            if not candidates:
                break

            candidates.sort(key=lambda item: item[0], reverse=True)
            keep = candidates[: max(1, min(3, len(candidates)))]
            _, nxt, move_dir = keep[int(rng.integers(len(keep)))]
            path.append(nxt)
            used.add(nxt)
            last_dir = move_dir

        if len(path) >= min_len and (not want_turn or _turn_count(path) >= 1):
            return path
    return None


def _reserve(mask: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in coords:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                mask[rr, cc] = True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Spread each colored seed along its grey path by a distance equal to its order on that path.

    Input structure:
    - The background is the most common color.
    - One non-background color forms one-pixel-wide 4-connected path components, and colored seed pixels replace some path cells.
    - Each connected path component is a simple path with exactly two endpoints when grey and seed pixels are considered together.

    Rule:
    - For each path component, find its two endpoints and choose the start endpoint by smaller row, then smaller column.
    - Traverse the path from that start endpoint; the first seed encountered has order 1, the second has order 2, and so on.
    - A seed of order k colors up to k grey path cells in both directions, stopping at the path end or immediately before another seed.
    - If two seeds would color the same grey pixel, keep the color of the smaller-order seed; seed pixels themselves stay unchanged.

    Key insight:
    - The important hidden variable is not seed color but seed order along an oriented path, because that order defines both the expansion radius and the priority in overlaps.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()

    non_bg = g != bg
    if not np.any(non_bg):
        return out

    path_color = Counter(map(int, g[non_bg])).most_common(1)[0][0]
    claim_rank = np.full(g.shape, 10**9, dtype=int)

    for comp in _components(non_bg):
        ordered = _ordered_path(comp)
        seeds = [(idx, int(g[cell])) for idx, cell in enumerate(ordered) if int(g[cell]) != path_color]
        for rank, (path_idx, color) in enumerate(seeds, start=1):
            for direction in (-1, 1):
                for step in range(1, rank + 1):
                    target_idx = path_idx + direction * step
                    if target_idx < 0 or target_idx >= len(ordered):
                        break
                    rr, cc = ordered[target_idx]
                    if int(g[rr, cc]) != path_color:
                        break
                    if claim_rank[rr, cc] > rank:
                        claim_rank[rr, cc] = rank
                        out[rr, cc] = color
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(12, 25))
    w = int(rng.integers(12, 25))
    bg_choices = [color for color in range(10) if color != GREY]
    bg = int(rng.choice(bg_choices))
    grid = np.full((h, w), bg, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    path_count = int(rng.choice([1, 2, 3], p=[0.2, 0.4, 0.4]))
    base_scenarios = ["overlap", "truncate", "separate"]
    rng.shuffle(base_scenarios)
    if path_count == 1:
        scenarios = [base_scenarios[0]]
    elif path_count == 2:
        scenarios = base_scenarios[:2]
    else:
        scenarios = base_scenarios

    palette = [color for color in range(10) if color not in (bg, GREY)]
    active_color_count = int(rng.integers(2, min(5, len(palette)) + 1))
    active_colors = list(map(int, rng.choice(palette, size=active_color_count, replace=False)))
    all_seed_colors: list[int] = []

    for scenario in scenarios:
        min_len = {"overlap": 6, "truncate": 6, "separate": 8}[scenario]
        max_len = {"overlap": 15, "truncate": 16, "separate": 18}[scenario]
        want_turn = bool(rng.random() < (0.8 if scenario != "truncate" else 0.6))
        path = _build_path(h, w, reserved, min_len=min_len, max_len=max_len, want_turn=want_turn, rng=rng)
        if path is None:
            return None

        actual_order = path if path[0] <= path[-1] else list(reversed(path))
        seed_positions = _choose_seed_positions(len(actual_order), scenario, rng)
        if seed_positions is None:
            return None

        for cell in path:
            grid[cell] = GREY

        for seed_index, pos in enumerate(seed_positions):
            if all_seed_colors and rng.random() < 0.35:
                color = int(rng.choice(all_seed_colors))
            else:
                color = int(rng.choice(active_colors))
            rr, cc = actual_order[pos]
            grid[rr, cc] = color
            all_seed_colors.append(color)

        _reserve(reserved, path)

    non_bg = grid[grid != bg]
    counts = Counter(map(int, non_bg))
    if counts[GREY] <= max(count for color, count in counts.items() if color != GREY):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid grid")
