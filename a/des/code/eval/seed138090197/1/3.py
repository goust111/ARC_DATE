from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
MARKER = 3
SEPARATOR = 9
PATTERN_COLORS = (1, 2, 4, 5, 6, 7, 8)

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR4 = tuple(DIRS.values())

CELL_COUNT_MIN, CELL_COUNT_MAX = 2, 5
CELL_SIZE_MIN, CELL_SIZE_MAX = 4, 7
SEED_COUNT_EXTRA_MAX = 3
GEN_ATTEMPTS = 400

SCENARIOS = (
    (False, False, False),
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (True, True, False),
    (True, False, True),
    (False, True, True),
    (True, True, True),
)


def _regular_intervals(separators: list[int], length: int) -> list[tuple[int, int]]:
    assert separators, "expected separator lines"
    bounds = [-1, *sorted(separators), int(length)]
    intervals = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
    sizes = [end - start for start, end in intervals]
    assert min(sizes) > 0, "separator layout leaves empty cells"
    assert len(set(sizes)) == 1, "cells must have a uniform size"
    return intervals


def _cell_layout(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_separators = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEPARATOR)]
    col_separators = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEPARATOR)]
    row_intervals = _regular_intervals(row_separators, grid.shape[0])
    col_intervals = _regular_intervals(col_separators, grid.shape[1])
    assert len(row_intervals) == len(col_intervals), "expected an N x N cell grid"
    return row_intervals, col_intervals


def _clean_block(block: np.ndarray) -> np.ndarray:
    out = np.array(block, copy=True)
    out[out == MARKER] = BG
    return out


def _seed_dirs(block: np.ndarray) -> set[str]:
    dirs: set[str] = set()
    if np.any(block[0, :] == MARKER):
        dirs.add("up")
    if np.any(block[-1, :] == MARKER):
        dirs.add("down")
    if np.any(block[:, 0] == MARKER):
        dirs.add("left")
    if np.any(block[:, -1] == MARKER):
        dirs.add("right")
    return dirs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each seed-cell pattern one step into marked neighboring cells.

    The grid is split by maroon separator lines into equal cells. A seed cell
    contains a non-black pattern plus one or more green edge markers. Markers
    indicate directions; the pattern itself is copied, once, into each directly
    adjacent empty cell in those directions. Green markers are removed from the
    output, and conflicts prefer the lexicographically earliest seed cell.
    """
    g = np.asarray(grid, dtype=int)
    row_intervals, col_intervals = _cell_layout(g)
    n = len(row_intervals)

    cleaned = np.array(g, copy=True)
    cleaned[cleaned == MARKER] = BG
    out = np.array(cleaned, copy=True)

    seeds: list[tuple[int, int, set[str], np.ndarray]] = []
    occupied = np.zeros((n, n), dtype=bool)
    for cr, (r0, r1) in enumerate(row_intervals):
        for cc, (c0, c1) in enumerate(col_intervals):
            block = g[r0:r1, c0:c1]
            clean = cleaned[r0:r1, c0:c1]
            occupied[cr, cc] = bool(np.any(clean != BG))
            dirs = _seed_dirs(block)
            if dirs and occupied[cr, cc]:
                seeds.append((cr, cc, dirs, clean.copy()))

    proposals: dict[tuple[int, int], list[tuple[int, int, np.ndarray]]] = {}
    for sr, sc, dirs, pattern in sorted(seeds, key=lambda item: (item[0], item[1])):
        for name in dirs:
            dr, dc = DIRS[name]
            tr = sr + dr
            tc = sc + dc
            if not (0 <= tr < n and 0 <= tc < n):
                continue
            if occupied[tr, tc]:
                continue
            proposals.setdefault((tr, tc), []).append((sr, sc, pattern))

    for (tr, tc), claims in proposals.items():
        sr, sc, pattern = min(claims, key=lambda item: (item[0], item[1]))
        del sr, sc
        r0, r1 = row_intervals[tr]
        c0, c1 = col_intervals[tc]
        mask = pattern != BG
        out[r0:r1, c0:c1][mask] = pattern[mask]

    return out


def _neighbors(n: int, pos: tuple[int, int]) -> list[tuple[str, tuple[int, int]]]:
    r, c = pos
    out: list[tuple[str, tuple[int, int]]] = []
    for name, (dr, dc) in DIRS.items():
        nr = r + dr
        nc = c + dc
        if 0 <= nr < n and 0 <= nc < n:
            out.append((name, (nr, nc)))
    return out


def _direction_between(src: tuple[int, int], dst: tuple[int, int]) -> str:
    dr = dst[0] - src[0]
    dc = dst[1] - src[1]
    for name, step in DIRS.items():
        if step == (dr, dc):
            return name
    raise ValueError(f"non-adjacent cells: {src} -> {dst}")


def _connected_points(side: int, target_area: int) -> list[tuple[int, int]]:
    start = (random.randrange(side), random.randrange(side))
    pts = {start}
    while len(pts) < target_area:
        frontier = []
        for r, c in pts:
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < side and 0 <= nc < side and (nr, nc) not in pts:
                    frontier.append((nr, nc))
        if not frontier:
            break
        pts.add(random.choice(frontier))
    return sorted(pts)


def _make_seed_block(cell_size: int, dirs: Iterable[str], seen: set[bytes]) -> np.ndarray | None:
    max_area = max(1, (cell_size - 2) * (cell_size - 2) - 1)
    for _ in range(80):
        block = np.full((cell_size, cell_size), BG, dtype=int)
        area = random.randint(1, min(max_area, cell_size + 1))
        pts = _connected_points(cell_size - 2, area)
        colors = random.sample(PATTERN_COLORS, 2)
        use_two_colors = len(pts) >= 2 and random.random() < 0.45
        split = random.randint(1, len(pts) - 1) if use_two_colors else 0
        shuffled = pts[:]
        random.shuffle(shuffled)
        for idx, (r, c) in enumerate(shuffled):
            color = colors[1] if use_two_colors and idx >= split else colors[0]
            block[r + 1, c + 1] = color

        for name in dirs:
            span = random.randint(1, cell_size - 2)
            start = random.randint(1, cell_size - span - 1)
            if name == "up":
                block[0, start : start + span] = MARKER
            elif name == "down":
                block[-1, start : start + span] = MARKER
            elif name == "left":
                block[start : start + span, 0] = MARKER
            else:
                block[start : start + span, -1] = MARKER

        signature = _clean_block(block).tobytes()
        if signature not in seen:
            seen.add(signature)
            return block
    return None


def _render_grid(n: int, cell_size: int, blocks: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    side = n * cell_size + (n - 1)
    grid = np.full((side, side), BG, dtype=int)
    for i in range(1, n):
        sep = i * (cell_size + 1) - 1
        grid[sep, :] = SEPARATOR
        grid[:, sep] = SEPARATOR

    for r in range(n):
        for c in range(n):
            r0 = r * (cell_size + 1)
            c0 = c * (cell_size + 1)
            block = blocks.get((r, c))
            if block is not None:
                grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = block
    return grid


def _simulate_output(n: int, cell_size: int, blocks: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    clean_blocks: dict[tuple[int, int], np.ndarray] = {}
    occupied: dict[tuple[int, int], bool] = {}
    proposals: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for r in range(n):
        for c in range(n):
            block = blocks.get((r, c), np.full((cell_size, cell_size), BG, dtype=int))
            clean = _clean_block(block)
            clean_blocks[(r, c)] = clean
            occupied[(r, c)] = bool(np.any(clean != BG))

    seeds = sorted(blocks.items())
    for (sr, sc), block in seeds:
        dirs = _seed_dirs(block)
        pattern = clean_blocks[(sr, sc)]
        if not dirs or not occupied[(sr, sc)]:
            continue
        for name in dirs:
            dr, dc = DIRS[name]
            tr = sr + dr
            tc = sc + dc
            if not (0 <= tr < n and 0 <= tc < n):
                continue
            if occupied[(tr, tc)]:
                continue
            proposals.setdefault((tr, tc), []).append((sr, sc))

    out_blocks = {pos: clean.copy() for pos, clean in clean_blocks.items()}
    for target, sources in proposals.items():
        winner = min(sources)
        out_blocks[target] = clean_blocks[winner].copy()
    return _render_grid(n, cell_size, out_blocks)


def _feature_stats(n: int, seeds: dict[tuple[int, int], dict[str, object]]) -> tuple[int, int, bool]:
    occupied = set(seeds)
    successful_targets: dict[tuple[int, int], list[tuple[int, int]]] = {}
    blocked = 0
    multi = any(len(spec["dirs"]) >= 2 for spec in seeds.values())
    for src, spec in seeds.items():
        for name in spec["dirs"]:
            dr, dc = DIRS[name]
            target = (src[0] + dr, src[1] + dc)
            if target in occupied:
                blocked += 1
            else:
                successful_targets.setdefault(target, []).append(src)
    conflict = sum(1 for sources in successful_targets.values() if len(sources) >= 2)
    return len(successful_targets), blocked, multi and conflict >= 0


def _ensure_seed(seeds: dict[tuple[int, int], dict[str, object]], pos: tuple[int, int]) -> dict[str, object]:
    if pos not in seeds:
        seeds[pos] = {"dirs": set()}
    return seeds[pos]


def _choose_conflict(
    n: int,
    seeds: dict[tuple[int, int], dict[str, object]],
    reserved_empty: set[tuple[int, int]],
) -> bool:
    candidates = [(r, c) for r in range(n) for c in range(n) if (r, c) not in reserved_empty]
    random.shuffle(candidates)
    for target in candidates:
        neighbors = _neighbors(n, target)
        if len(neighbors) < 2:
            continue
        picked = random.sample(neighbors, 2)
        reserved_empty.add(target)
        for _dir_from_target, src in picked:
            _ensure_seed(seeds, src)["dirs"].add(_direction_between(src, target))
        return True
    return False


def _choose_blocked(
    n: int,
    seeds: dict[tuple[int, int], dict[str, object]],
    reserved_empty: set[tuple[int, int]],
) -> bool:
    pairs = []
    for r in range(n):
        for c in range(n):
            src = (r, c)
            for name, dst in _neighbors(n, src):
                if dst not in reserved_empty:
                    pairs.append((src, name, dst))
    if not pairs:
        return False
    src, name, dst = random.choice(pairs)
    _ensure_seed(seeds, src)["dirs"].add(name)
    _ensure_seed(seeds, dst)
    return True


def _choose_multi(
    n: int,
    seeds: dict[tuple[int, int], dict[str, object]],
    reserved_empty: set[tuple[int, int]],
) -> bool:
    candidates = [(r, c) for r in range(n) for c in range(n) if (r, c) not in reserved_empty]
    random.shuffle(candidates)
    for pos in candidates:
        options = [name for name, _dst in _neighbors(n, pos)]
        if len(options) < 2:
            continue
        count = random.randint(2, min(3, len(options)))
        _ensure_seed(seeds, pos)["dirs"].update(random.sample(options, count))
        return True
    return False


def _fill_missing_dirs(
    n: int,
    seeds: dict[tuple[int, int], dict[str, object]],
    reserved_empty: set[tuple[int, int]],
) -> None:
    occupied = set(seeds)
    for pos, spec in seeds.items():
        options = _neighbors(n, pos)
        if not spec["dirs"]:
            preferred = [name for name, dst in options if dst not in occupied and dst not in reserved_empty]
            if not preferred:
                preferred = [name for name, _dst in options]
            spec["dirs"].add(random.choice(preferred))
        elif random.random() < 0.25:
            extras = [name for name, dst in options if name not in spec["dirs"] and dst not in reserved_empty]
            if extras:
                spec["dirs"].add(random.choice(extras))


def _ensure_successful_copy(
    n: int,
    seeds: dict[tuple[int, int], dict[str, object]],
    reserved_empty: set[tuple[int, int]],
) -> bool:
    occupied = set(seeds)
    for pos, spec in seeds.items():
        for name in spec["dirs"]:
            dr, dc = DIRS[name]
            if (pos[0] + dr, pos[1] + dc) not in occupied:
                return True

    candidates = list(seeds)
    random.shuffle(candidates)
    for pos in candidates:
        options = [name for name, dst in _neighbors(n, pos) if dst not in occupied and dst not in reserved_empty]
        if options:
            seeds[pos]["dirs"].add(random.choice(options))
            return True

    free_positions = [(r, c) for r in range(n) for c in range(n) if (r, c) not in occupied and (r, c) not in reserved_empty]
    random.shuffle(free_positions)
    for pos in free_positions:
        options = [name for name, dst in _neighbors(n, pos) if dst not in occupied and dst not in reserved_empty]
        if not options:
            continue
        seeds[pos] = {"dirs": {random.choice(options)}}
        return True
    return False


def _build_seed_specs(n: int) -> dict[tuple[int, int], dict[str, object]] | None:
    want_conflict, want_blocked, want_multi = random.choice(SCENARIOS)
    seeds: dict[tuple[int, int], dict[str, object]] = {}
    reserved_empty: set[tuple[int, int]] = set()

    if want_conflict and not _choose_conflict(n, seeds, reserved_empty):
        return None
    if want_blocked and not _choose_blocked(n, seeds, reserved_empty):
        return None
    if want_multi and not _choose_multi(n, seeds, reserved_empty):
        return None

    max_seed_slots = n * n - len(reserved_empty)
    min_seeds = max(2, len(seeds))
    if min_seeds > max_seed_slots:
        return None
    target_count = random.randint(min_seeds, min(max_seed_slots, min_seeds + SEED_COUNT_EXTRA_MAX))

    free = [(r, c) for r in range(n) for c in range(n) if (r, c) not in seeds and (r, c) not in reserved_empty]
    random.shuffle(free)
    for pos in free[: max(0, target_count - len(seeds))]:
        _ensure_seed(seeds, pos)

    _fill_missing_dirs(n, seeds, reserved_empty)
    if not _ensure_successful_copy(n, seeds, reserved_empty):
        return None

    occupied = set(seeds)
    successful_targets: dict[tuple[int, int], list[tuple[int, int]]] = {}
    blocked_moves = 0
    for src, spec in seeds.items():
        for name in spec["dirs"]:
            dr, dc = DIRS[name]
            target = (src[0] + dr, src[1] + dc)
            if target in occupied:
                blocked_moves += 1
            else:
                successful_targets.setdefault(target, []).append(src)

    if want_conflict and not any(len(sources) >= 2 for sources in successful_targets.values()):
        return None
    if want_blocked and blocked_moves == 0:
        return None
    if want_multi and not any(len(spec["dirs"]) >= 2 for spec in seeds.values()):
        return None
    if not successful_targets:
        return None
    return seeds


def _background_dominates(grid: np.ndarray) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)]) == BG


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(CELL_COUNT_MIN, CELL_COUNT_MAX)
    cell_size = random.randint(CELL_SIZE_MIN, CELL_SIZE_MAX)
    seeds = _build_seed_specs(n)
    if seeds is None:
        return None

    seen_patterns: set[bytes] = set()
    blocks: dict[tuple[int, int], np.ndarray] = {}
    for pos, spec in sorted(seeds.items()):
        block = _make_seed_block(cell_size, spec["dirs"], seen_patterns)
        if block is None:
            return None
        blocks[pos] = block

    inp = _render_grid(n, cell_size, blocks)
    expected = _simulate_output(n, cell_size, blocks)
    if not _background_dominates(inp):
        return None
    if np.array_equal(inp, expected):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
