from __future__ import annotations

from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
MAGENTA = 6

SEED_COLORS = (BLUE, RED, GREEN)
SEED_ROWS = (2, 3, 4)
_RNG = np.random.default_rng()


def _seed_mask(grid: np.ndarray) -> np.ndarray:
    return (grid != BLACK) & (grid != GREY)


def _valid_columns(grid: np.ndarray) -> set[int]:
    return {int(c) for c in np.flatnonzero(grid[0] == GREY)}


def _find_seeds(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere(_seed_mask(grid))
    seeds = [(int(r), int(c), int(grid[r, c])) for r, c in coords]
    seeds.sort()
    return seeds


def _collect_claims(
    grid: np.ndarray, *, respect_mask: bool
) -> tuple[dict[tuple[int, int], set[int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    valid_columns = _valid_columns(g)
    seeds = _find_seeds(g)
    seed_positions = {(r, c) for r, c, _color in seeds}
    claims: dict[tuple[int, int], set[int]] = defaultdict(set)
    branch_counts: list[tuple[int, int]] = []

    for r, c, color in seeds:
        left_count = 0
        right_count = 0
        max_k = (height - 1 - r) // 2
        for k in range(1, max_k + 1):
            rr = r + 2 * k
            for dc, side in ((-k, 0), (k, 1)):
                cc = c + dc
                if not (0 <= cc < width):
                    continue
                if respect_mask and cc not in valid_columns:
                    continue
                if (rr, cc) in seed_positions:
                    continue
                claims[(rr, cc)].add(color)
                if side == 0:
                    left_count += 1
                else:
                    right_count += 1
        branch_counts.append((left_count, right_count))

    return claims, branch_counts


def _solve(grid: np.ndarray, *, respect_mask: bool) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    claims, _branch_counts = _collect_claims(out, respect_mask=respect_mask)
    for (r, c), colors in claims.items():
        out[r, c] = MAGENTA if len(colors) >= 2 else next(iter(colors))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed diagonally downward every two rows, but only through mask-enabled columns, and recolor cross-color collisions magenta.

    Input structure:
    - The grid has a black background.
    - Row 0 is a grey/black mask row whose grey cells mark valid growth columns.
    - Rows 2 to 4 contain 2 or 3 isolated colored seed pixels on valid columns.

    Rule:
    - Read the valid column set from the grey cells in row 0.
    - For each seed at (r, c), place same-color pixels at (r+2k, c-k) and
      (r+2k, c+k) for k >= 1 while those cells stay in bounds; masked-out
      columns are skipped.
    - After all seeds emit their trajectories independently, any non-seed cell
      claimed by two or more different colors becomes magenta; single-color
      claims keep that color, and the original seeds and mask row stay unchanged.

    Key insight:
    - The top-row mask does not just decorate the scene: it gates which diagonal
      steps survive, which in turn changes where cross-color collisions can occur.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"
    return _solve(g, respect_mask=True)


def _alternating_mask(width: int, phase: int) -> np.ndarray:
    row = np.zeros(width, dtype=int)
    row[np.arange(width) % 2 == phase] = GREY
    return row


def _render_input(
    height: int, width: int, phase: int, seeds: list[tuple[int, int, int]]
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[0] = _alternating_mask(width, phase)
    for r, c, color in seeds:
        grid[r, c] = color
    return grid


def _is_alternating_mask(row: np.ndarray) -> bool:
    vals = set(map(int, row))
    return vals == {BLACK, GREY} and bool(np.all(row[:-1] != row[1:]))


def _structure_is_valid(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    height, width = map(int, g.shape)
    if height < 7 or width < 7:
        return False
    if not _is_alternating_mask(g[0]):
        return False

    seeds = _find_seeds(g)
    if len(seeds) not in {2, 3}:
        return False
    colors = [color for _r, _c, color in seeds]
    if len(set(colors)) != len(colors):
        return False
    if any(color not in SEED_COLORS for color in colors):
        return False

    valid_columns = _valid_columns(g)
    for r, c, _color in seeds:
        if r not in SEED_ROWS:
            return False
        if c not in valid_columns:
            return False

    cols = [c for _r, c, _color in seeds]
    if any(abs(cols[i] - cols[j]) < 3 for i in range(len(cols)) for j in range(i + 1, len(cols))):
        return False

    return int(np.count_nonzero(_seed_mask(g))) == len(seeds)


def _collision_positions(claims: dict[tuple[int, int], set[int]]) -> set[tuple[int, int]]:
    return {pos for pos, colors in claims.items() if len(colors) >= 2}


def _example_is_valid(grid: np.ndarray) -> bool:
    if not _structure_is_valid(grid):
        return False

    masked_claims, masked_branches = _collect_claims(grid, respect_mask=True)
    unmasked_claims, _unmasked_branches = _collect_claims(grid, respect_mask=False)
    if not masked_claims or not unmasked_claims:
        return False

    masked_collisions = _collision_positions(masked_claims)
    unmasked_collisions = _collision_positions(unmasked_claims)
    if not masked_collisions:
        return False
    if len(masked_collisions) > 6:
        return False

    if np.array_equal(_solve(grid, respect_mask=True), _solve(grid, respect_mask=False)):
        return False

    removed_cells = set(unmasked_claims) - set(masked_claims)
    if not removed_cells:
        return False

    if not any(len(colors) == 1 for colors in masked_claims.values()):
        return False

    if not any(left != right for left, right in masked_branches):
        return False

    if len(_find_seeds(grid)) == 3 and not (unmasked_collisions - masked_collisions):
        return False

    total_foreground = int(np.count_nonzero(main(grid)))
    if not (12 <= total_foreground <= 55):
        return False

    return True


def _candidate_pair_columns(
    width: int, phase: int, row: int, height: int
) -> list[tuple[int, int]]:
    valid_columns = set(range(phase, width, 2))
    pairs: list[tuple[int, int]] = []
    for gap in (4, 8, 12):
        if row + gap >= height:
            continue
        for c1 in range(phase, width, 2):
            c2 = c1 + gap
            if c2 in valid_columns and c2 < width:
                pairs.append((c1, c2))
    pairs.sort(key=lambda pair: min(pair[0], width - 1 - pair[1]))
    return pairs


def _third_seed_candidates(
    width: int,
    phase: int,
    row: int,
    height: int,
    occupied_columns: set[int],
) -> list[int]:
    valid_columns = set(range(phase, width, 2))
    candidates: list[int] = []
    for anchor in sorted(occupied_columns):
        for gap in (6, 10):
            if row + gap >= height:
                continue
            for sign in (-1, 1):
                c3 = anchor + sign * gap
                if c3 not in valid_columns or not (0 <= c3 < width):
                    continue
                if all(abs(c3 - c) >= 3 for c in occupied_columns):
                    candidates.append(c3)
    return sorted(set(candidates), key=lambda c: min(c, width - 1 - c))


def _generate_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(15, 22))
    width = int(_RNG.integers(18, 25))
    phase = int(_RNG.integers(0, 2))
    row = int(_RNG.choice(SEED_ROWS))
    num_seeds = 3 if float(_RNG.random()) < 0.68 else 2

    pair_candidates = _candidate_pair_columns(width, phase, row, height)
    if not pair_candidates:
        return None

    if float(_RNG.random()) < 0.6:
        pair = pair_candidates[0]
    else:
        pair = pair_candidates[int(_RNG.integers(0, len(pair_candidates)))]
    c1, c2 = pair

    colors = list(_RNG.permutation(SEED_COLORS)[:num_seeds])
    seeds = [(row, c1, colors[0]), (row, c2, colors[1])]

    if num_seeds == 3:
        candidates = _third_seed_candidates(width, phase, row, height, {c1, c2})
        if not candidates:
            return None
        if float(_RNG.random()) < 0.6:
            c3 = candidates[-1]
        else:
            c3 = candidates[int(_RNG.integers(0, len(candidates)))]
        seeds.append((row, c3, colors[2]))

    if num_seeds == 2 and float(_RNG.random()) < 0.5:
        seeds[0] = (int(_RNG.choice(SEED_ROWS)), seeds[0][1], seeds[0][2])
        seeds[1] = (seeds[0][0], seeds[1][1], seeds[1][2])

    grid = _render_input(height, width, phase, seeds)
    return grid if _example_is_valid(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid masked-growth collision example")
