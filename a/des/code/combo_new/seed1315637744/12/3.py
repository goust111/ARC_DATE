from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
ARROW_OFFSETS = {
    (-1, 0): [(-1, 0), (-1, -1), (-1, 1)],
    (1, 0): [(1, 0), (1, -1), (1, 1)],
    (0, -1): [(0, -1), (-1, -1), (1, -1)],
    (0, 1): [(0, 1), (-1, 1), (1, 1)],
}

GEN_RETRIES = 500
PLACE_RETRIES = 500


def _in_bounds(r: int, c: int, height: int, width: int) -> bool:
    return 0 <= r < height and 0 <= c < width


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if not _in_bounds(nr, nc, height, width):
                        continue
                    if not bool(mask[nr, nc]) or bool(seen[nr, nc]):
                        continue
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bar_center(coords: list[tuple[int, int]]) -> tuple[int, int] | None:
    if len(coords) != 3:
        return None
    rows = sorted(r for r, _ in coords)
    cols = sorted(c for _, c in coords)
    unique_rows = sorted(set(rows))
    unique_cols = sorted(set(cols))
    if len(unique_rows) == 1 and cols == list(range(cols[0], cols[0] + 3)):
        return unique_rows[0], cols[1]
    if len(unique_cols) == 1 and rows == list(range(rows[0], rows[0] + 3)):
        return rows[1], unique_cols[0]
    return None


def _find_seed_groups(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    height, width = grid.shape
    groups: list[tuple[int, int, int, int, int]] = []

    for comp in _components4(grid == GREY):
        center = _bar_center(comp)
        if center is None:
            continue
        cr, cc = center
        seed_candidates = []
        for nr, nc in _neighbors4(cr, cc):
            if not _in_bounds(nr, nc, height, width):
                continue
            if (nr, nc) in comp:
                continue
            color = int(grid[nr, nc])
            if color not in (BG, GREY):
                seed_candidates.append((int(nr), int(nc)))
        assert len(seed_candidates) == 1, "each arrow must point to exactly one seed"
        sr, sc = seed_candidates[0]
        dr, dc = cr - sr, cc - sc
        assert (dr, dc) in ARROW_OFFSETS, "seed must touch the arrow at its center"
        groups.append((int(sr), int(sc), int(grid[sr, sc]), int(dr), int(dc)))

    groups.sort(key=lambda item: (item[2], item[0], item[1], item[3], item[4]))
    return groups


def _extent_to_edge(r: int, c: int, dr: int, dc: int, height: int, width: int) -> int:
    if dr == -1:
        return int(r)
    if dr == 1:
        return int(height - 1 - r)
    if dc == -1:
        return int(c)
    return int(width - 1 - c)


def _core_cells(r: int, c: int, dr: int, dc: int) -> set[tuple[int, int]]:
    cells = {(int(r), int(c))}
    for orow, ocol in ARROW_OFFSETS[(int(dr), int(dc))]:
        cells.add((int(r + orow), int(c + ocol)))
    return cells


def _path_cells(r: int, c: int, dr: int, dc: int, steps: int, height: int, width: int) -> list[tuple[int, int]]:
    cells = []
    for step in range(1, int(steps) + 1):
        rr = int(r + dr * step)
        cc = int(c + dc * step)
        if not _in_bounds(rr, cc, height, width):
            break
        cells.append((int(rr), int(cc)))
    return cells


def _halo(cells: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if _in_bounds(int(rr), int(cc), height, width):
                    out.add((int(rr), int(cc)))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow same-color rays from seeded arrow markers, with each ray length set by the global seed count of that color.

    Input structure:
    - The background is black.
    - Each instruction group contains one colored seed pixel and one adjacent gray arrow made of three gray pixels.
    - Every gray arrow is a straight 3-cell bar whose center touches its seed; the seed side determines whether the arrow points up, down, left, or right.
    - The same color may appear in several seed groups, and all groups of one color share one direction.

    Rule:
    - Count how many seed pixels exist for each non-gray foreground color in the whole grid.
    - For every seed, follow its arrow direction and trace a straight segment of that many steps, excluding the seed itself.
    - Recolor every in-bounds non-gray step of that segment with the seed color; if the segment reaches the grid edge, stop there.
    - Keep all original gray arrow pixels unchanged.

    Key insight:
    - The arrow gives only the local direction, while the segment length comes from the global frequency of that seed color across the entire grid.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    groups = _find_seed_groups(g)
    if not groups:
        return out

    counts = Counter(color for _r, _c, color, _dr, _dc in groups)
    color_dirs: dict[int, tuple[int, int]] = {}
    for _r, _c, color, dr, dc in groups:
        prev = color_dirs.setdefault(int(color), (int(dr), int(dc)))
        assert prev == (int(dr), int(dc)), "all seeds of one color must share one direction"

    height, width = g.shape
    for r, c, color, dr, dc in groups:
        for rr, cc in _path_cells(int(r), int(c), int(dr), int(dc), int(counts[int(color)]), height, width):
            if int(g[rr, cc]) == GREY:
                continue
            out[rr, cc] = int(color)
    return out


def _sample_seed_position(
    rng: np.random.Generator,
    dr: int,
    dc: int,
    height: int,
    width: int,
) -> tuple[int, int]:
    if (dr, dc) == (-1, 0):
        return int(rng.integers(1, height)), int(rng.integers(1, width - 1))
    if (dr, dc) == (1, 0):
        return int(rng.integers(0, height - 1)), int(rng.integers(1, width - 1))
    if (dr, dc) == (0, -1):
        return int(rng.integers(1, height - 1)), int(rng.integers(1, width))
    return int(rng.integers(1, height - 1)), int(rng.integers(0, width - 1))


def _place_seed(
    rng: np.random.Generator,
    height: int,
    width: int,
    steps: int,
    dr: int,
    dc: int,
    used: set[tuple[int, int]],
    guard: set[tuple[int, int]],
    mode: str,
) -> tuple[tuple[int, int], bool] | None:
    for _ in range(PLACE_RETRIES):
        r, c = _sample_seed_position(rng, int(dr), int(dc), height, width)
        extent = _extent_to_edge(int(r), int(c), int(dr), int(dc), height, width)
        if extent < 2:
            continue
        is_truncated = extent < int(steps)
        if mode == "trunc" and not (2 <= extent < int(steps)):
            continue
        if mode == "full" and extent < int(steps):
            continue

        core = _core_cells(int(r), int(c), int(dr), int(dc))
        if not all(_in_bounds(rr, cc, height, width) for rr, cc in core):
            continue

        path = set(_path_cells(int(r), int(c), int(dr), int(dc), int(steps), height, width))
        if len(path) < 2:
            continue

        if core & used:
            continue
        if path & used:
            continue
        if (core | path) & guard:
            continue
        if _halo(core, height, width) & used:
            continue

        used.update(core | path)
        guard.update(_halo(core, height, width))
        return (int(r), int(c)), bool(is_truncated)
    return None


def _build_grid(
    height: int,
    width: int,
    placements: list[tuple[int, int, int, int, int]],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for r, c, color, dr, dc in placements:
        grid[int(r), int(c)] = int(color)
        for rr, cc in _core_cells(int(r), int(c), int(dr), int(dc)):
            if (rr, cc) != (int(r), int(c)):
                grid[int(rr), int(cc)] = GREY
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_colors = int(rng.integers(2, 5))
    colors = [int(x) for x in rng.choice(SEED_COLORS, size=n_colors, replace=False)]
    counts = [int(rng.integers(2, 5)) for _ in range(n_colors)]
    if len(set(counts)) == 1:
        idx = int(rng.integers(0, n_colors))
        choices = [value for value in (2, 3, 4) if value != counts[idx]]
        counts[idx] = int(rng.choice(choices))

    perm = [int(x) for x in rng.permutation(len(DIRECTIONS))]
    directions = [DIRECTIONS[idx] for idx in perm[:n_colors]]

    total_seeds = sum(counts)
    min_side = max(12, 8 + total_seeds)
    if min_side > 30:
        return None
    max_side = min(30, min_side + 6)
    height = int(rng.integers(min_side, max_side + 1))
    width = int(rng.integers(min_side, max_side + 1))

    trunc_candidates = [idx for idx, count in enumerate(counts) if count >= 3]
    need_truncation = bool(trunc_candidates and rng.random() < 0.7)
    forced_trunc_idx = int(rng.choice(trunc_candidates)) if need_truncation else None

    used: set[tuple[int, int]] = set()
    guard: set[tuple[int, int]] = set()
    placements: list[tuple[int, int, int, int, int]] = []
    truncated_seen = False
    full_seen = False

    for color_idx, color in enumerate(colors):
        steps = int(counts[color_idx])
        dr, dc = directions[color_idx]

        modes = ["full"] * steps
        if color_idx == forced_trunc_idx:
            modes[0] = "trunc"
        else:
            for i in range(steps):
                if steps >= 3 and rng.random() < 0.35:
                    modes[i] = "trunc"
        rng.shuffle(modes)
        if color_idx == forced_trunc_idx and "trunc" in modes:
            trunc_pos = modes.index("trunc")
            modes[0], modes[trunc_pos] = modes[trunc_pos], modes[0]

        for mode in modes:
            placed = _place_seed(
                rng,
                height,
                width,
                steps,
                int(dr),
                int(dc),
                used,
                guard,
                mode,
            )
            if placed is None:
                return None
            (r, c), is_truncated = placed
            placements.append((int(r), int(c), int(color), int(dr), int(dc)))
            truncated_seen = truncated_seen or bool(is_truncated)
            full_seen = full_seen or (not bool(is_truncated))

    if need_truncation and not truncated_seen:
        return None
    if not full_seen:
        return None

    grid = _build_grid(height, width, placements)

    groups = _find_seed_groups(grid)
    if len(groups) != len(placements):
        return None

    parsed_counts = Counter(color for _r, _c, color, _dr, _dc in groups)
    if parsed_counts != Counter({int(colors[i]): int(counts[i]) for i in range(n_colors)}):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if not np.all(out[grid == GREY] == GREY):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
