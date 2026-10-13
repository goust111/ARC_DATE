from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
SEED_COLORS = (1, 5, 6, 7, 8, 9)

PREFS = ("left", "up", "right")
OFFSET_TO_PREF = {-1: "left", 0: "up", 1: "right"}
PREF_TO_OFFSET = {name: offset for offset, name in OFFSET_TO_PREF.items()}
PHASE_DIRS = {
    "left": (((-1, -1),), ((-1, 1), (1, -1), (1, 1))),
    "up": (((-1, -1), (-1, 1)), ((1, -1), (1, 1))),
    "right": (((-1, 1),), ((-1, -1), (1, -1), (1, 1))),
}

BOARD_H_RANGE = (7, 13)
BOARD_W_RANGE = (8, 15)
SEEDS_PER_DOMAIN = (2, 3)
MAX_RETRIES = 400


def _base_board(board_h: int, width: int, green_parity: int) -> np.ndarray:
    rr, cc = np.indices((board_h, width))
    return np.where((rr + cc) % 2 == green_parity, GREEN, YELLOW).astype(int)


def _checker_color(r: int, c: int, green_parity: int) -> int:
    return GREEN if (r + c) % 2 == green_parity else YELLOW


def _infer_green_parity(board: np.ndarray) -> int:
    rr, cc = np.indices(board.shape)
    green_cells = np.argwhere(board == GREEN)
    yellow_cells = np.argwhere(board == YELLOW)
    if len(green_cells):
        parity = int((green_cells[0, 0] + green_cells[0, 1]) % 2)
    elif len(yellow_cells):
        parity = 1 - int((yellow_cells[0, 0] + yellow_cells[0, 1]) % 2)
    else:
        raise AssertionError("checkerboard colors not visible")
    assert not np.any((board == GREEN) & (((rr + cc) % 2) != parity))
    assert not np.any((board == YELLOW) & (((rr + cc) % 2) == parity))
    return parity


def _read_preference(grid: np.ndarray, domain_color: int) -> str:
    block_cols = np.flatnonzero(grid[-1] == domain_color)
    assert block_cols.size > 0, "rule block missing"
    red_cols = np.flatnonzero(grid[-2] == RED)
    matches: list[tuple[float, int, int]] = []
    center = float(block_cols.mean())
    for red_c in red_cols:
        diffs = [int(red_c - col) for col in block_cols if abs(int(red_c - col)) <= 1]
        if diffs:
            best = min(diffs, key=abs)
            matches.append((abs(red_c - center), best, int(red_c)))
    assert matches, "red marker missing for rule block"
    matches.sort()
    offset = matches[0][1]
    assert offset in OFFSET_TO_PREF
    return OFFSET_TO_PREF[offset]


def _find_seeds(board: np.ndarray) -> list[tuple[int, int, int]]:
    mask = ~np.isin(board, (BLACK, RED, GREEN, YELLOW))
    seeds = [(int(r), int(c), int(board[r, c])) for r, c in np.argwhere(mask)]
    colors = [color for _, _, color in seeds]
    assert len(colors) == len(set(colors)), "seed colors must be unique"
    return seeds


def _domain_mask(board_h: int, width: int, green_parity: int, domain_color: int) -> np.ndarray:
    rr, cc = np.indices((board_h, width))
    green_mask = (rr + cc) % 2 == green_parity
    return green_mask if domain_color == GREEN else ~green_mask


def _simulate_domain(
    board_h: int,
    width: int,
    green_parity: int,
    domain_color: int,
    preference: str,
    seeds: list[tuple[int, int, int]],
) -> np.ndarray:
    mask = _domain_mask(board_h, width, green_parity, domain_color)
    status = np.full((board_h, width), -3, dtype=int)
    status[mask] = -1
    for idx, (r, c, _color) in enumerate(seeds):
        assert mask[r, c]
        assert status[r, c] == -1
        status[r, c] = idx

    if not seeds:
        return status

    while True:
        proposals: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
        for r, c in np.argwhere(status >= 0):
            owner = int(status[r, c])
            for phase, dirs in enumerate(PHASE_DIRS[preference]):
                for dr, dc in dirs:
                    nr = int(r) + dr
                    nc = int(c) + dc
                    if 0 <= nr < board_h and 0 <= nc < width and status[nr, nc] == -1:
                        prev = proposals[(nr, nc)].get(owner)
                        if prev is None or phase < prev:
                            proposals[(nr, nc)][owner] = phase

        if not proposals:
            break

        claimed: list[tuple[tuple[int, int], int]] = []
        blocked: list[tuple[int, int]] = []
        for cell, by_owner in proposals.items():
            best_phase = min(by_owner.values())
            winners = [owner for owner, phase in by_owner.items() if phase == best_phase]
            if len(winners) == 1:
                claimed.append((cell, winners[0]))
            else:
                blocked.append(cell)

        if not claimed and not blocked:
            break

        for r, c in blocked:
            status[r, c] = -2
        for (r, c), owner in claimed:
            status[r, c] = owner

    return status


def _solve_from_parts(
    board_h: int,
    width: int,
    green_parity: int,
    preferences: dict[int, str],
    seeds: list[tuple[int, int, int]],
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, list[tuple[int, int, int]]]]:
    out = np.full((board_h + 1, width), BLACK, dtype=int)
    board = _base_board(board_h, width, green_parity)
    out[:board_h, :] = board

    seeds_by_domain = {GREEN: [], YELLOW: []}
    for seed in seeds:
        r, c, _color = seed
        seeds_by_domain[_checker_color(r, c, green_parity)].append(seed)

    statuses: dict[int, np.ndarray] = {}
    board_out = out[:board_h, :]
    for domain_color in (GREEN, YELLOW):
        status = _simulate_domain(
            board_h,
            width,
            green_parity,
            domain_color,
            preferences[domain_color],
            seeds_by_domain[domain_color],
        )
        statuses[domain_color] = status
        for idx, (_r, _c, seed_color) in enumerate(seeds_by_domain[domain_color]):
            board_out[status == idx] = seed_color

    return out, statuses, seeds_by_domain


def _parse_input(grid: np.ndarray) -> tuple[int, int, int, dict[int, str], list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3 and g.shape[1] >= 5
    board = g[:-1, :]
    green_parity = _infer_green_parity(board)
    preferences = {
        GREEN: _read_preference(g, GREEN),
        YELLOW: _read_preference(g, YELLOW),
    }
    seeds = _find_seeds(board)
    return board.shape[0], board.shape[1], green_parity, preferences, seeds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed across its checkerboard domain, with the rule bar deciding
    which diagonal direction is attempted first for green and yellow cells.
    """
    board_h, width, green_parity, preferences, seeds = _parse_input(grid)
    out, _statuses, _groups = _solve_from_parts(board_h, width, green_parity, preferences, seeds)
    return out


def _sample_rule_columns(width: int, prefs: dict[int, str], rng: random.Random) -> tuple[int, int]:
    offsets = {color: PREF_TO_OFFSET[pref] for color, pref in prefs.items()}
    candidates = list(range(1, width - 1))
    for _ in range(200):
        green_col, yellow_col = rng.sample(candidates, 2)
        red_cols = [green_col + offsets[GREEN], yellow_col + offsets[YELLOW]]
        positions = {green_col, yellow_col, red_cols[0], red_cols[1]}
        if len(positions) == 4 and abs(green_col - yellow_col) >= 4:
            return green_col, yellow_col
    raise RuntimeError("failed to place rule markers")


def _pick_cells(
    cells: list[tuple[int, int]],
    count: int,
    taken: list[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    pool = cells[:]
    rng.shuffle(pool)
    chosen: list[tuple[int, int]] = []
    for cell in pool:
        if all(max(abs(cell[0] - r), abs(cell[1] - c)) >= 2 for r, c in taken + chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    return None


def _preference_effective(
    board_h: int,
    width: int,
    green_parity: int,
    preferences: dict[int, str],
    seeds: list[tuple[int, int, int]],
) -> bool:
    baseline, _statuses, groups = _solve_from_parts(board_h, width, green_parity, preferences, seeds)
    for domain_color in (GREEN, YELLOW):
        if len(groups[domain_color]) < 2:
            continue
        for alt in PREFS:
            if alt == preferences[domain_color]:
                continue
            alt_prefs = dict(preferences)
            alt_prefs[domain_color] = alt
            alt_out, _alt_statuses, _alt_groups = _solve_from_parts(
                board_h, width, green_parity, alt_prefs, seeds
            )
            if not np.array_equal(baseline, alt_out):
                return True
    return False


def _areas_ok(statuses: dict[int, np.ndarray], groups: dict[int, list[tuple[int, int, int]]]) -> bool:
    for domain_color in (GREEN, YELLOW):
        count = len(groups[domain_color])
        if count == 0:
            return False
        for idx in range(count):
            if int(np.sum(statuses[domain_color] == idx)) < 2:
                return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    board_h = rng.randint(*BOARD_H_RANGE)
    width = rng.randint(*BOARD_W_RANGE)
    green_parity = rng.randint(0, 1)
    preferences = {
        GREEN: rng.choice(PREFS),
        YELLOW: rng.choice(PREFS),
    }

    try:
        green_rule_col, yellow_rule_col = _sample_rule_columns(width, preferences, rng)
    except RuntimeError:
        return None

    grid = np.full((board_h + 1, width), BLACK, dtype=int)
    grid[:board_h, :] = _base_board(board_h, width, green_parity)

    green_red_col = green_rule_col + PREF_TO_OFFSET[preferences[GREEN]]
    yellow_red_col = yellow_rule_col + PREF_TO_OFFSET[preferences[YELLOW]]
    grid[board_h - 1, green_red_col] = RED
    grid[board_h - 1, yellow_red_col] = RED
    grid[board_h, green_rule_col] = GREEN
    grid[board_h, yellow_rule_col] = YELLOW

    domain_cells = {GREEN: [], YELLOW: []}
    for r in range(1, board_h - 1):
        for c in range(1, width - 1):
            domain_cells[_checker_color(r, c, green_parity)].append((r, c))

    n_green = rng.choice(SEEDS_PER_DOMAIN)
    n_yellow = rng.choice(SEEDS_PER_DOMAIN)
    total = n_green + n_yellow
    if total > len(SEED_COLORS):
        return None

    chosen_green = _pick_cells(domain_cells[GREEN], n_green, [], rng)
    if chosen_green is None:
        return None
    chosen_yellow = _pick_cells(domain_cells[YELLOW], n_yellow, chosen_green, rng)
    if chosen_yellow is None:
        return None

    colors = list(rng.sample(SEED_COLORS, total))
    seeds: list[tuple[int, int, int]] = []
    for pos, color in zip(chosen_green + chosen_yellow, colors):
        r, c = pos
        seeds.append((r, c, int(color)))
        grid[r, c] = int(color)

    out, statuses, groups = _solve_from_parts(board_h, width, green_parity, preferences, seeds)
    if not _areas_ok(statuses, groups):
        return None
    if not np.any(grid[:-1, :] == RED):
        return None
    has_boundary = any(np.any(status == -2) for status in statuses.values())
    if not has_boundary and not _preference_effective(board_h, width, green_parity, preferences, seeds):
        return None
    if np.array_equal(grid, out):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
