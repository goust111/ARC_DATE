from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
TEAL = 8

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    height, width = map(int, shape)
    return 0 <= int(r) < height and 0 <= int(c) < width


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    for dr, dc in NEIGHBORS4:
        rr, cc = int(r) + dr, int(c) + dc
        if _in_bounds(shape, rr, cc):
            yield rr, cc


def _non_separator_segments(all_separator: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_separator in enumerate(all_separator):
        if bool(is_separator):
            if start is not None:
                segments.append((start, idx))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        segments.append((start, len(all_separator)))
    return segments


def _extract_panels(grid: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    row_segments = _non_separator_segments(np.all(grid == PINK, axis=1))
    col_segments = _non_separator_segments(np.all(grid == PINK, axis=0))
    assert len(row_segments) == 2, "expected two panel rows"
    assert len(col_segments) == 2, "expected two panel columns"

    row_heights = {r1 - r0 for r0, r1 in row_segments}
    col_widths = {c1 - c0 for c0, c1 in col_segments}
    assert len(row_heights) == 1 and len(col_widths) == 1, "panel shapes must match"

    panels: list[np.ndarray] = []
    anchors: list[tuple[int, int]] = []
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            patch = np.asarray(grid[r0:r1, c0:c1], dtype=int)
            assert patch.size > 0, "empty panel"
            assert not np.any(patch == PINK), "separator leaked into panel interior"
            panels.append(patch)
            anchors.append((int(r0), int(c0)))
    return panels, anchors


def _adjacent_with_color(panel: np.ndarray, cell: tuple[int, int], color: int) -> list[tuple[int, int]]:
    return [
        (rr, cc)
        for rr, cc in _neighbors4(cell[0], cell[1], panel.shape)
        if int(panel[rr, cc]) == int(color)
    ]


def _process_panel(panel: np.ndarray) -> tuple[np.ndarray, int]:
    out = np.asarray(panel, dtype=int).copy()
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    used_reds: set[tuple[int, int]] = set()
    used_greens: set[tuple[int, int]] = set()

    for rr, cc in zip(*np.where(panel == GREEN)):
        green = (int(rr), int(cc))
        red_neighbors = _adjacent_with_color(panel, green, RED)
        if not red_neighbors:
            continue
        assert len(red_neighbors) == 1, "each paired green must touch exactly one red"
        red = red_neighbors[0]
        green_neighbors = _adjacent_with_color(panel, red, GREEN)
        assert green_neighbors == [green], "each paired red must touch exactly one green"
        assert red not in used_reds and green not in used_greens, "dominoes must be disjoint"
        used_reds.add(red)
        used_greens.add(green)
        pairs.append((green, red))

    for green, red in pairs:
        out[green] = TEAL
        out[red] = BLACK
    return out, int(np.count_nonzero(out))


def _overlay_processed_panels(panels: list[np.ndarray], counts: list[int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = panels[0].shape
    out = np.zeros((height, width), dtype=int)
    winners = np.full((height, width), -1, dtype=int)
    for r in range(height):
        for c in range(width):
            contributors = [
                (counts[idx], -idx, idx, int(panel[r, c]))
                for idx, panel in enumerate(panels)
                if int(panel[r, c]) != BLACK
            ]
            if contributors:
                _count, _neg_idx, idx, color = max(contributors)
                out[r, c] = color
                winners[r, c] = idx
    return out, winners


def main(grid: np.ndarray) -> np.ndarray:
    """Overlay four separator-split panels after pre-processing red-green dominoes inside each panel.

    Input structure:
    - Magenta separator lines form a bordered 2x2 layout of four equal rectangular panels.
    - Each panel has black background plus sparse blue, yellow, red, and green single-cell marks.
    - Some red and green cells form isolated 4-neighbor dominoes inside a panel; other red/green cells are distractors.

    Rule:
    - Split the grid into the four equal panels using the magenta separator rows and columns.
    - In each panel independently, convert every valid red-green domino by recoloring the green cell to teal and erasing the red cell to black.
    - Count each panel's surviving non-black cells after that pre-processing, then transparently overlay the processed panels by relative position.
    - If several processed panels contribute to the same output cell, keep the color from the panel with the largest survivor count; if those counts tie, prefer the panel whose top-left corner is earlier in row-major order in the input.

    Key insight:
    - The domino step changes both local colors and each panel's global survivor count, so overlap conflicts can only be resolved correctly after every panel has been pre-processed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    panels, _anchors = _extract_panels(g)
    assert len(panels) == 4, "expected four panels"

    processed: list[np.ndarray] = []
    counts: list[int] = []
    for panel in panels:
        proc, count = _process_panel(panel)
        processed.append(proc)
        counts.append(count)

    out, _winners = _overlay_processed_panels(processed, counts)
    return out


def _count_options(required_non_teal: int) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    for survivors in range(4, 7):
        for dominoes in range(0, 4):
            if survivors + dominoes < 5 or survivors + dominoes > 8:
                continue
            if survivors < required_non_teal + dominoes:
                continue
            options.append((survivors, dominoes))
    return options


def _pick_swing_counts() -> tuple[tuple[int, int], tuple[int, int]]:
    loser_options = [opt for opt in _count_options(2) if opt[1] >= 2]
    random.shuffle(loser_options)
    for loser_survivors, loser_dominoes in loser_options:
        loser_input = loser_survivors + loser_dominoes
        winner_options = [
            (winner_survivors, winner_dominoes)
            for winner_survivors, winner_dominoes in _count_options(2)
            if winner_survivors > loser_survivors
            and loser_input > winner_survivors + winner_dominoes
        ]
        if winner_options:
            return (
                (loser_survivors, loser_dominoes),
                random.choice(winner_options),
            )
    raise RuntimeError("failed to sample swing counts")


def _checkerboard_cells(height: int, width: int, parity: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(height)
        for c in range(width)
        if (r + c) % 2 == parity
    ]


def _processed_to_input(processed: np.ndarray) -> np.ndarray | None:
    teal_cells = [tuple(map(int, cell)) for cell in np.argwhere(processed == TEAL)]
    unpaired_greens = {
        tuple(map(int, cell))
        for cell in np.argwhere(processed == GREEN)
    }
    green_cells = set(teal_cells) | unpaired_greens

    input_panel = processed.copy()
    input_panel[input_panel == TEAL] = BLACK

    def candidates_for(green: tuple[int, int], chosen_reds: set[tuple[int, int]]) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for rr, cc in _neighbors4(green[0], green[1], processed.shape):
            if int(processed[rr, cc]) != BLACK:
                continue
            if (rr, cc) in chosen_reds:
                continue
            green_neighbors = {
                (gr, gc)
                for gr, gc in _neighbors4(rr, cc, processed.shape)
                if (gr, gc) in green_cells
            }
            if green_neighbors == {green}:
                out.append((rr, cc))
        return out

    ordering = teal_cells[:]
    random.shuffle(ordering)
    ordering.sort(key=lambda cell: len(candidates_for(cell, set())))

    def backtrack(idx: int, chosen_reds: set[tuple[int, int]]) -> dict[tuple[int, int], tuple[int, int]] | None:
        if idx == len(ordering):
            return {}
        green = ordering[idx]
        options = candidates_for(green, chosen_reds)
        random.shuffle(options)
        for red in options:
            chosen_reds.add(red)
            tail = backtrack(idx + 1, chosen_reds)
            chosen_reds.remove(red)
            if tail is not None:
                tail[green] = red
                return tail
        return None

    mapping = backtrack(0, set())
    if mapping is None:
        return None

    for green, red in mapping.items():
        input_panel[green] = GREEN
        input_panel[red] = RED

    rebuilt, _count = _process_panel(input_panel)
    if not np.array_equal(rebuilt, processed):
        return None
    return input_panel


def _random_non_teal_color() -> int:
    return random.choice([BLUE, YELLOW, RED, GREEN])


def _build_processed_panels(
    height: int,
    width: int,
) -> tuple[list[np.ndarray], dict[str, object]] | None:
    parity = random.randint(0, 1)
    candidate_cells = _checkerboard_cells(height, width, parity)
    use_tie = random.random() < 0.55

    swing_loser, swing_winner = random.sample(range(4), 2)
    remaining = [idx for idx in range(4) if idx not in {swing_loser, swing_winner}]
    tie_pair = tuple(remaining) if use_tie else ()

    loser_counts, winner_counts = _pick_swing_counts()
    count_map: dict[int, tuple[int, int]] = {
        swing_loser: loser_counts,
        swing_winner: winner_counts,
    }

    if use_tie:
        tie_options = [opt for opt in _count_options(1) if opt[0] in {4, 5}]
        tie_choice = random.choice(tie_options)
        count_map[remaining[0]] = tie_choice
        count_map[remaining[1]] = random.choice(
            [opt for opt in tie_options if opt[0] == tie_choice[0]]
        )
    else:
        general_options = _count_options(0)
        count_map[remaining[0]] = random.choice(general_options)
        count_map[remaining[1]] = random.choice(general_options)

    required_positions: dict[int, list[tuple[int, int]]] = {idx: [] for idx in range(4)}
    shared_map: dict[tuple[int, int], tuple[int, ...]] = {}

    conflict_positions = random.sample(candidate_cells, 3 if use_tie else 2)
    primary_conflicts = conflict_positions[:2]
    for pos in primary_conflicts:
        required_positions[swing_loser].append(pos)
        required_positions[swing_winner].append(pos)
        shared_map[pos] = (swing_loser, swing_winner)

    if use_tie:
        tie_pos = conflict_positions[2]
        for idx in tie_pair:
            required_positions[idx].append(tie_pos)
        shared_map[tie_pos] = tie_pair

    total_unique_needed = sum(count_map[idx][0] for idx in range(4)) - sum(
        len(panels) - 1 for panels in shared_map.values()
    )
    if total_unique_needed > len(candidate_cells):
        return None

    forbidden_unique = set(shared_map)
    free_unique = [cell for cell in candidate_cells if cell not in forbidden_unique]
    random.shuffle(free_unique)

    panel_positions: dict[int, set[tuple[int, int]]] = {
        idx: set(required_positions[idx]) for idx in range(4)
    }
    cursor = 0
    order = sorted(range(4), key=lambda idx: count_map[idx][0] - len(required_positions[idx]), reverse=True)
    for idx in order:
        survivors, dominoes = count_map[idx]
        need = survivors - len(required_positions[idx])
        if need < dominoes:
            return None
        if cursor + need > len(free_unique):
            return None
        picks = free_unique[cursor : cursor + need]
        cursor += need
        panel_positions[idx].update(picks)

    processed_panels: list[np.ndarray] = []
    conflict_colors: dict[tuple[int, int], int] = {}

    for pos in primary_conflicts:
        colors = random.sample([BLUE, YELLOW, RED, GREEN], 2)
        conflict_colors[(swing_loser, pos)] = colors[0]
        conflict_colors[(swing_winner, pos)] = colors[1]

    if use_tie:
        tie_pos = conflict_positions[2]
        colors = random.sample([BLUE, YELLOW, RED, GREEN], 2)
        conflict_colors[(tie_pair[0], tie_pos)] = colors[0]
        conflict_colors[(tie_pair[1], tie_pos)] = colors[1]

    for idx in range(4):
        survivors, dominoes = count_map[idx]
        panel = np.zeros((height, width), dtype=int)
        required = set(required_positions[idx])
        extras = list(panel_positions[idx] - required)
        random.shuffle(extras)
        teal_positions = set(extras[:dominoes])
        for pos in panel_positions[idx]:
            if pos in teal_positions:
                panel[pos] = TEAL
            elif (idx, pos) in conflict_colors:
                panel[pos] = conflict_colors[(idx, pos)]
            else:
                panel[pos] = _random_non_teal_color()

        if int(np.count_nonzero(panel)) != survivors:
            return None
        processed_panels.append(panel)

    counts = [count_map[idx][0] for idx in range(4)]
    input_counts = [count_map[idx][0] + count_map[idx][1] for idx in range(4)]
    expected, winners = _overlay_processed_panels(processed_panels, counts)

    if any(input_counts[idx] < 5 or input_counts[idx] > 8 for idx in range(4)):
        return None
    if counts[swing_loser] >= counts[swing_winner]:
        return None
    if input_counts[swing_loser] <= input_counts[swing_winner]:
        return None
    for pos in primary_conflicts:
        contributors = [idx for idx in range(4) if int(processed_panels[idx][pos]) != BLACK]
        if contributors != [swing_loser, swing_winner] and contributors != [swing_winner, swing_loser]:
            return None
        if int(processed_panels[swing_loser][pos]) == int(processed_panels[swing_winner][pos]):
            return None
        if int(winners[pos]) != int(swing_winner):
            return None

    if use_tie:
        tie_pos = conflict_positions[2]
        tie_counts = [counts[idx] for idx in tie_pair]
        if tie_counts[0] != tie_counts[1]:
            return None
        tie_winner = min(tie_pair)
        if int(winners[tie_pos]) != tie_winner:
            return None

    meta = {
        "expected": expected,
        "counts": counts,
        "input_counts": input_counts,
        "swing_loser": swing_loser,
        "swing_winner": swing_winner,
        "primary_conflicts": tuple(primary_conflicts),
    }
    return processed_panels, meta


def _assemble_grid(panels: list[np.ndarray]) -> np.ndarray:
    panel_h, panel_w = panels[0].shape
    grid = np.full((2 * panel_h + 3, 2 * panel_w + 3), PINK, dtype=int)
    offsets = [
        (1, 1),
        (1, panel_w + 2),
        (panel_h + 2, 1),
        (panel_h + 2, panel_w + 2),
    ]
    for panel, (r0, c0) in zip(panels, offsets):
        grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = panel
    return grid


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(7, 8)
    panel_w = random.randint(7, 8)

    built = _build_processed_panels(panel_h, panel_w)
    if built is None:
        return None
    processed_panels, meta = built

    input_panels: list[np.ndarray] = []
    for processed in processed_panels:
        input_panel = _processed_to_input(processed)
        if input_panel is None:
            return None
        input_panels.append(input_panel)

    grid = _assemble_grid(input_panels)
    expected = np.asarray(meta["expected"], dtype=int)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 5")
