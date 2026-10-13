from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    GREY = 5


DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GRID_SIDE_MIN = 10
GRID_SIDE_MAX = 16
BAR_LENGTHS = (2, 3, 4)
GENERATION_RETRIES = 400

_TOKEN_COUNT_ORDER = random.sample([1, 2, 3], 3)
_GENERATE_CALL_COUNT = 0


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if not in_bounds(mask.shape, nr, nc):
                        continue
                    if not bool(mask[nr, nc]) or bool(seen[nr, nc]):
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _rotate_clockwise(dr: int, dc: int, quarter_turns: int) -> tuple[int, int]:
    turns = int(quarter_turns) % 4
    for _ in range(turns):
        dr, dc = dc, -dr
    return int(dr), int(dc)


def _ray_from_pivot(pivot: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    pr, pc = pivot
    dr, dc = direction
    return [(pr + step * dr, pc + step * dc) for step in range(1, int(length) + 1)]


def _count_blue_tokens(grid: np.ndarray) -> int:
    blue_components = components_mask(grid == Color.BLUE)
    assert 1 <= len(blue_components) <= 3, "expected 1-3 blue token components"
    for comp in blue_components:
        assert comp.shape[0] == 1, "blue tokens must be isolated single pixels"
        r, c = map(int, comp[0])
        for nr, nc in neighbors4(r, c):
            if in_bounds(grid.shape, nr, nc):
                assert int(grid[nr, nc]) not in {Color.RED, Color.GREY}, "blue tokens must be isolated from bars and pivots"
    return len(blue_components)


def _extract_segments(grid: np.ndarray):
    pivots = [tuple(map(int, rc)) for rc in np.argwhere(grid == Color.GREY)]
    assert 2 <= len(pivots) <= 3, "expected 2-3 pivots"

    all_red = {tuple(map(int, rc)) for rc in np.argwhere(grid == Color.RED)}
    used_red: set[tuple[int, int]] = set()
    segments = []

    for pivot in pivots:
        pr, pc = pivot
        adjacent = []
        for nr, nc in neighbors4(pr, pc):
            if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == Color.RED:
                adjacent.append((nr, nc))
        assert len(adjacent) == 1, "each pivot must touch exactly one red arm"

        ar, ac = adjacent[0]
        dr, dc = ar - pr, ac - pc
        coords = []
        step = 1
        while True:
            rr = pr + step * dr
            cc = pc + step * dc
            if not in_bounds(grid.shape, rr, cc) or int(grid[rr, cc]) != Color.RED:
                break
            coords.append((rr, cc))
            step += 1

        assert 2 <= len(coords) <= 4, "red bars must have length 2-4"
        expected = set(_ray_from_pivot(pivot, (dr, dc), len(coords)))
        assert expected == set(coords), "red arm must be a straight contiguous ray from its pivot"
        used_red.update(coords)
        segments.append(
            {
                "pivot": pivot,
                "direction": (dr, dc),
                "length": len(coords),
                "source": coords,
            }
        )

    assert used_red == all_red, "all red cells must belong to exactly one pivot arm"
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each pivot-attached red bar clockwise by a number of quarter-turns set by the blue token count, leaving green traces behind.

    Input structure:
    - The grid is a black background with 2-3 grey pivot pixels, each touching exactly one straight red bar of length 2-4.
    - The red bar for a pivot extends only from one side of that pivot and different pivot-bar structures are separated.
    - There are 1-3 isolated blue singleton pixels that are not 4-neighbor adjacent to any pivot or red bar.

    Rule:
    - Count the isolated blue pixels and call that number N.
    - For every grey pivot, recover its red bar and rotate that bar clockwise around the pivot by N quarter-turns.
    - Recolor every original red-bar cell green, then paint the rotated destination cells red, with red taking priority on overlaps.
    - Keep the grey pivots and blue token pixels unchanged.

    Key insight:
    - The blue pixels do not interact locally with the bars; they act only as a global counter that selects the common rotation angle applied independently around every pivot.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert int(bg) == Color.BLACK, "expected black background"

    turns = _count_blue_tokens(g)
    segments = _extract_segments(g)

    out = g.copy()
    source_cells: list[tuple[int, int]] = []
    dest_cells: list[tuple[int, int]] = []

    for segment in segments:
        pivot = segment["pivot"]
        length = int(segment["length"])
        rotated_dir = _rotate_clockwise(*segment["direction"], quarter_turns=turns)
        source_cells.extend(segment["source"])
        dest_cells.extend(_ray_from_pivot(pivot, rotated_dir, length))

    for r, c in source_cells:
        out[r, c] = Color.GREEN
    for r, c in dest_cells:
        assert in_bounds(g.shape, r, c), "rotated arm left the grid"
        assert int(g[r, c]) not in {Color.BLUE, Color.GREY}, "rotated arm cannot overwrite tokens or pivots"
        out[r, c] = Color.RED

    return out


def _halo(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        for nr, nc in neighbors4(r, c):
            if in_bounds(shape, nr, nc):
                out.add((nr, nc))
    return out


def _candidate_configs(shape: tuple[int, int], turns: int):
    h, w = shape
    configs = []
    for pr in range(h):
        for pc in range(w):
            pivot = (pr, pc)
            for direction in DIRS4:
                rotated = _rotate_clockwise(*direction, quarter_turns=turns)
                for length in BAR_LENGTHS:
                    source = _ray_from_pivot(pivot, direction, length)
                    target = _ray_from_pivot(pivot, rotated, length)
                    footprint = set(source) | set(target) | {pivot}
                    if all(in_bounds(shape, r, c) for r, c in footprint):
                        configs.append(
                            {
                                "pivot": pivot,
                                "direction": direction,
                                "length": length,
                                "source": source,
                                "target": target,
                                "footprint": footprint,
                            }
                        )
    random.shuffle(configs)
    return configs


def _select_configs(shape: tuple[int, int], turns: int, pivot_count: int):
    candidates = _candidate_configs(shape, turns)
    chosen = []
    reserved: set[tuple[int, int]] = set()

    def backtrack(start: int) -> bool:
        if len(chosen) == int(pivot_count):
            return True
        remaining_needed = int(pivot_count) - len(chosen)
        for idx in range(start, len(candidates)):
            if len(candidates) - idx < remaining_needed:
                break
            candidate = candidates[idx]
            if candidate["footprint"] & reserved:
                continue
            chosen.append(candidate)
            added = _halo(candidate["footprint"], shape)
            old_reserved = reserved.copy()
            reserved.update(added)
            if backtrack(idx + 1):
                return True
            chosen.pop()
            reserved.clear()
            reserved.update(old_reserved)
        return False

    if backtrack(0):
        return chosen
    return None


def _place_tokens(grid: np.ndarray, token_count: int, forbidden: set[tuple[int, int]]):
    h, w = grid.shape
    candidates = [(r, c) for r in range(h) for c in range(w) if (r, c) not in forbidden]
    random.shuffle(candidates)
    tokens: list[tuple[int, int]] = []
    token_reserved = set(forbidden)

    for r, c in candidates:
        if (r, c) in token_reserved:
            continue
        tokens.append((r, c))
        token_reserved.update(_halo({(r, c)}, grid.shape))
        if len(tokens) == int(token_count):
            return tokens
    return None


def _latent_output(inp: np.ndarray, configs) -> np.ndarray:
    out = inp.copy()
    for config in configs:
        for r, c in config["source"]:
            out[r, c] = Color.GREEN
    for config in configs:
        for r, c in config["target"]:
            out[r, c] = Color.RED
    return out


def _next_branch_settings() -> tuple[int, int]:
    global _GENERATE_CALL_COUNT
    idx = _GENERATE_CALL_COUNT
    _GENERATE_CALL_COUNT += 1
    token_count = _TOKEN_COUNT_ORDER[idx % len(_TOKEN_COUNT_ORDER)]
    pivot_count = 2 + (idx % 2)
    return int(token_count), int(pivot_count)


def _gen_attempt(token_count: int, pivot_count: int) -> np.ndarray | None:
    min_side = max(GRID_SIDE_MIN, 9 + pivot_count)
    h = random.randint(min_side, GRID_SIDE_MAX)
    w = random.randint(min_side, GRID_SIDE_MAX)

    configs = _select_configs((h, w), token_count, pivot_count)
    if configs is None:
        return None

    grid = np.full((h, w), Color.BLACK, dtype=int)
    forbidden = set()
    for config in configs:
        pr, pc = config["pivot"]
        grid[pr, pc] = Color.GREY
        for r, c in config["source"]:
            grid[r, c] = Color.RED
        forbidden.update(_halo(config["footprint"], grid.shape))

    tokens = _place_tokens(grid, token_count, forbidden)
    if tokens is None:
        return None
    for r, c in tokens:
        grid[r, c] = Color.BLUE

    expected = _latent_output(grid, configs)
    if not np.array_equal(main(grid), expected):
        return None

    unique_colors = set(map(int, np.unique(grid)))
    assert unique_colors == {Color.BLACK, Color.BLUE, Color.RED, Color.GREY}
    return grid


def generate_input() -> np.ndarray:
    token_count, pivot_count = _next_branch_settings()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(token_count, pivot_count)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid scattered-token rotation input")
