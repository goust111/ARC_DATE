from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GEN_ATTEMPTS = 400


@dataclass(frozen=True)
class Arrow:
    center: tuple[int, int]
    direction: str
    tail: tuple[int, int]
    tail_color: int


DIR_SPECS = {
    "U": {"vec": (-1, 0), "head": [(-1, -1), (-1, 0), (-1, 1)], "tail": (1, 0)},
    "D": {"vec": (1, 0), "head": [(1, -1), (1, 0), (1, 1)], "tail": (-1, 0)},
    "L": {"vec": (0, -1), "head": [(-1, -1), (0, -1), (1, -1)], "tail": (0, 1)},
    "R": {"vec": (0, 1), "head": [(-1, 1), (0, 1), (1, 1)], "tail": (0, -1)},
}


def _pattern_cells(center: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    r, c = center
    spec = DIR_SPECS[direction]
    cells = [(r, c)]
    cells.extend((r + dr, c + dc) for dr, dc in spec["head"])
    tr, tc = spec["tail"]
    cells.append((r + tr, c + tc))
    return cells


def _tail_pos(center: tuple[int, int], direction: str) -> tuple[int, int]:
    r, c = center
    tr, tc = DIR_SPECS[direction]["tail"]
    return r + tr, c + tc


def _step_direction(a: tuple[int, int], b: tuple[int, int]) -> str:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1 and c0 < c1:
        return "R"
    if r0 == r1 and c0 > c1:
        return "L"
    if c0 == c1 and r0 < r1:
        return "D"
    if c0 == c1 and r0 > r1:
        return "U"
    raise AssertionError("consecutive chain nodes must share a row or a column")


def _parse_arrows(grid: np.ndarray) -> list[Arrow]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    arrows: list[Arrow] = []

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != GREY:
                continue

            matches: list[Arrow] = []
            block = g[r - 1 : r + 2, c - 1 : c + 2]
            for direction, spec in DIR_SPECS.items():
                head_ok = all(int(g[r + dr, c + dc]) == GREY for dr, dc in spec["head"])
                if not head_ok:
                    continue

                tr, tc = spec["tail"]
                tail_color = int(g[r + tr, c + tc])
                if tail_color in (BG,):
                    continue

                pattern = {(1, 1), *((dr + 1, dc + 1) for dr, dc in spec["head"]), (tr + 1, tc + 1)}
                ok = True
                for rr in range(3):
                    for cc in range(3):
                        val = int(block[rr, cc])
                        if (rr, cc) in pattern:
                            if (rr, cc) == (tr + 1, tc + 1):
                                if val == BG:
                                    ok = False
                            elif val != GREY:
                                ok = False
                        elif val != BG:
                            ok = False
                if ok:
                    matches.append(
                        Arrow(center=(r, c), direction=direction, tail=(r + tr, c + tc), tail_color=tail_color)
                    )

            assert len(matches) <= 1, "arrow center matches multiple directions"
            if matches:
                arrows.append(matches[0])

    covered = np.zeros_like(g, dtype=bool)
    for arrow in arrows:
        for r, c in _pattern_cells(arrow.center, arrow.direction):
            assert not covered[r, c], "arrows must not overlap"
            covered[r, c] = True

    assert np.array_equal(covered, g != BG), "all non-background pixels must belong to arrows"
    assert any(arrow.tail_color != GREY for arrow in arrows), "expected at least one colored start arrow"
    return arrows


def _next_arrow_lookup(arrows: list[Arrow]) -> dict[tuple[int, int], Arrow | None]:
    by_row: dict[int, list[Arrow]] = {}
    by_col: dict[int, list[Arrow]] = {}
    for arrow in arrows:
        r, c = arrow.center
        by_row.setdefault(r, []).append(arrow)
        by_col.setdefault(c, []).append(arrow)

    lookup: dict[tuple[int, int], Arrow | None] = {}
    for arrow in arrows:
        r, c = arrow.center
        dr, dc = DIR_SPECS[arrow.direction]["vec"]
        if dr == 0:
            candidates = [
                other
                for other in by_row[r]
                if (other.center[1] - c) * dc > 0
            ]
            key = lambda other: abs(other.center[1] - c)
        else:
            candidates = [
                other
                for other in by_col[c]
                if (other.center[0] - r) * dr > 0
            ]
            key = lambda other: abs(other.center[0] - r)
        lookup[arrow.center] = min(candidates, key=key) if candidates else None
    return lookup


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow each colored arrow chain and move the start color to the terminal arrow tail.

    Input structure:
    - The background is black.
    - Grey arrows are scattered across the grid; each arrow fits inside a 3x3 box and uses a grey center, a grey three-pixel head, and a one-pixel tail on the opposite side.
    - Some tails are recolored to a non-black, non-grey color; these arrows are the starts, while fully grey arrows are relays or terminals.
    - Each non-terminal arrow points along its row or column toward the next arrow in the chain, with only background between them.

    Rule:
    - Detect every arrow and its direction from the local 3x3 pattern.
    - For each colored start arrow, repeatedly jump to the nearest arrow found straight ahead in the current arrow's pointing direction.
    - Stop when no farther arrow exists in that direction; this last arrow is the terminal for that start color.
    - Replace every original colored start tail with grey, then paint the terminal arrow tail with the start color, leaving all other arrow pixels unchanged.

    Key insight:
    - The colored pixel is only a payload marker; the real task is to parse arrow orientation, traverse the directed chain, and transfer that payload to the final reachable tail.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    arrows = _parse_arrows(g)
    arrow_by_center = {arrow.center: arrow for arrow in arrows}
    next_lookup = _next_arrow_lookup(arrows)

    out = g.copy()
    starts = [arrow for arrow in arrows if arrow.tail_color != GREY]
    for arrow in starts:
        out[arrow.tail] = GREY

    terminal_colors: dict[tuple[int, int], int] = {}
    for start in starts:
        current = start
        seen: set[tuple[int, int]] = set()
        while True:
            assert current.center not in seen, "arrow chains must not loop"
            seen.add(current.center)
            nxt = next_lookup[current.center]
            if nxt is None:
                break
            current = arrow_by_center[nxt.center]

        prev = terminal_colors.get(current.tail)
        if prev is not None:
            assert prev == start.tail_color, "multiple start colors cannot target the same terminal tail"
        terminal_colors[current.tail] = start.tail_color

    for (r, c), color in terminal_colors.items():
        out[r, c] = color
    return out


def _paint_arrow(grid: np.ndarray, center: tuple[int, int], direction: str, tail_color: int) -> None:
    r, c = center
    grid[r, c] = GREY
    for dr, dc in DIR_SPECS[direction]["head"]:
        grid[r + dr, c + dc] = GREY
    tr, tc = DIR_SPECS[direction]["tail"]
    grid[r + tr, c + tc] = tail_color


def _resource_counts(length: int, start_horizontal: bool) -> tuple[int, int]:
    segments = length - 1
    if start_horizontal:
        return 1 + segments // 2, 1 + (segments + 1) // 2
    return 1 + (segments + 1) // 2, 1 + segments // 2


def _chain_positions(
    length: int,
    start_horizontal: bool,
    rows: list[int],
    cols: list[int],
) -> list[tuple[int, int]]:
    row_idx = 0
    col_idx = 0
    positions: list[tuple[int, int]] = []
    for i in range(length):
        positions.append((rows[row_idx], cols[col_idx]))
        if i == length - 1:
            break
        if start_horizontal:
            if i % 2 == 0:
                col_idx += 1
            else:
                row_idx += 1
        else:
            if i % 2 == 0:
                row_idx += 1
            else:
                col_idx += 1
    return positions


def _safe_terminal_dirs(
    last_center: tuple[int, int],
    previous_center: tuple[int, int],
    grid_shape: tuple[int, int],
) -> list[str]:
    h, w = grid_shape
    blocked = _step_direction(last_center, previous_center)
    options: list[str] = []
    for direction in DIR_SPECS:
        if direction == blocked:
            continue
        cells = _pattern_cells(last_center, direction)
        if all(0 <= r < h and 0 <= c < w for r, c in cells):
            options.append(direction)
    return options


def _sample_chain_specs(
    rng: random.Random,
    row_slots: list[int],
    col_slots: list[int],
) -> list[tuple[int, bool]]:
    max_chains = min(3, len(row_slots) - 1, len(col_slots) - 1)
    if max_chains <= 0:
        raise RuntimeError("grid too small for arrow chains")

    for _ in range(80):
        chain_count = rng.randint(1, max_chains)
        specs = [(rng.randint(2, 5), rng.choice([False, True])) for _ in range(chain_count)]
        rows_needed = sum(_resource_counts(length, start_horizontal)[0] for length, start_horizontal in specs)
        cols_needed = sum(_resource_counts(length, start_horizontal)[1] for length, start_horizontal in specs)
        if rows_needed <= len(row_slots) and cols_needed <= len(col_slots):
            return specs
    raise RuntimeError("failed to fit chain specs into the sampled grid")


def _gen_attempt(rng: random.Random) -> np.ndarray:
    h = rng.randint(15, 30)
    w = rng.randint(15, 30)
    row_slots = list(range(1, h - 1, 3))
    col_slots = list(range(1, w - 1, 3))
    specs = _sample_chain_specs(rng, row_slots, col_slots)

    available_rows = row_slots[:]
    available_cols = col_slots[:]
    rng.shuffle(available_rows)
    rng.shuffle(available_cols)

    chain_positions: list[list[tuple[int, int]]] = []
    chain_directions: list[list[str]] = []
    for length, start_horizontal in specs:
        row_need, col_need = _resource_counts(length, start_horizontal)
        rows = [available_rows.pop() for _ in range(row_need)]
        cols = [available_cols.pop() for _ in range(col_need)]
        positions = _chain_positions(length, start_horizontal, rows, cols)

        directions = [_step_direction(a, b) for a, b in zip(positions, positions[1:])]
        terminal_options = _safe_terminal_dirs(positions[-1], positions[-2], (h, w))
        if not terminal_options:
            raise RuntimeError("failed to pick a terminal direction")
        directions.append(rng.choice(terminal_options))

        chain_positions.append(positions)
        chain_directions.append(directions)

    grid = np.full((h, w), BG, dtype=int)
    start_colors = rng.sample(COLOR_POOL, len(specs))
    for chain_idx, positions in enumerate(chain_positions):
        directions = chain_directions[chain_idx]
        for arrow_idx, center in enumerate(positions):
            tail_color = start_colors[chain_idx] if arrow_idx == 0 else GREY
            _paint_arrow(grid, center, directions[arrow_idx], tail_color)

    arrows = _parse_arrows(grid)
    starts = [arrow for arrow in arrows if arrow.tail_color != GREY]
    assert len(starts) == len(specs)
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            return _gen_attempt(rng)
        except (AssertionError, RuntimeError):
            continue
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
