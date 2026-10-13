import random
from collections import Counter

import numpy as np


STRIPE_COLOR_POOL = [1, 2, 3, 4, 6, 8, 9]
NOISE_COLOR_POOL = [0, 5, 7]
HEIGHT_RANGE = (9, 13)
WIDTH_RANGE = (10, 14)
MAX_GENERATION_ATTEMPTS = 400


def _row_majority_color(row: np.ndarray) -> int:
    counts = Counter(int(v) for v in row)
    best_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == best_count]
    assert len(winners) == 1, "row majority must be unique"
    return int(winners[0])


def _nearest_matching_row(row_colors: list[int], src_r: int, color: int) -> int:
    candidates = [r for r, stripe_color in enumerate(row_colors) if stripe_color == int(color)]
    assert candidates, "misplaced pixel color must match at least one stripe row"
    return min(candidates, key=lambda r: (abs(r - int(src_r)), -int(r)))


def _analyze_grid(grid: np.ndarray) -> tuple[list[int], list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_colors = [_row_majority_color(g[r]) for r in range(g.shape[0])]
    stripe_colors = set(row_colors)
    misplaced: list[dict] = []
    noise: list[dict] = []

    for r in range(g.shape[0]):
        row_color = row_colors[r]
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color == row_color:
                continue
            if color in stripe_colors:
                misplaced.append(
                    {
                        "src": (int(r), int(c)),
                        "color": color,
                        "target": (_nearest_matching_row(row_colors, r, color), int(c)),
                    }
                )
            else:
                noise.append({"src": (int(r), int(c)), "color": color})

    return row_colors, misplaced, noise


def _resolved_output(grid: np.ndarray) -> tuple[np.ndarray, list[int], list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    row_colors, misplaced, noise = _analyze_grid(g)
    out = np.repeat(np.asarray(row_colors, dtype=int)[:, None], g.shape[1], axis=1)

    # Resolve same-target collisions exactly as the rule specifies, even though
    # the generator normally avoids them.
    winners: dict[tuple[int, int], dict] = {}
    for item in misplaced:
        src_r, _src_c = item["src"]
        dst = item["target"]
        trip = abs(int(dst[0]) - int(src_r))
        current = winners.get(dst)
        if current is None or trip < current["trip"]:
            winners[dst] = {"trip": trip, "item": item}

    for dst, winner in winners.items():
        color = int(winner["item"]["color"])
        out[int(dst[0]), int(dst[1])] = color

    return out, row_colors, misplaced, noise


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore stripe rows by routing each off-stripe pixel to the nearest row whose stripe color matches it.

    Input structure:
    - The grid has no single background and is built from horizontal stripe rows.
    - Each row has one dominant stripe color occupying a clear majority of the row.
    - A few cells per row may instead hold another row's stripe color or a noise color that matches no stripe.

    Rule:
    - Determine each row's stripe color by taking the unique most frequent color in that row.
    - For every non-stripe cell, if its color matches some row's stripe color, treat it as a misplaced pixel and assign it to the nearest row with that stripe color, breaking equal-distance ties toward the lower row while keeping the same column.
    - If a non-stripe cell's color matches no row stripe color, erase it back to the current row's stripe color.
    - Apply the moves simultaneously; when two misplaced pixels target the same destination cell, keep the one with the shorter trip and restore the other source cell to its source-row stripe color.

    Key insight:
    - Row-majority stripe detection is not the final restoration step by itself; it creates the global row-to-color address book that decides which off-stripe pixels are valid relocations and which are mere noise.
    """
    out, _row_colors, _misplaced, _noise = _resolved_output(np.asarray(grid, dtype=int))
    return out


def _sample_dimensions() -> tuple[int, int]:
    return random.randint(*HEIGHT_RANGE), random.randint(*WIDTH_RANGE)


def _sample_row_colors(height: int, want_tie: bool) -> tuple[list[int], dict]:
    color_count = random.randint(3, min(5, height - 1))
    palette = random.sample(STRIPE_COLOR_POOL, color_count)
    row_colors = [random.choice(palette) for _ in range(height)]

    for color in palette:
        if color not in row_colors:
            row_colors[random.randrange(height)] = color

    tie_info: dict = {}
    if want_tie and height >= 3:
        middle = random.randint(1, height - 2)
        repeated = random.choice(palette)
        others = [color for color in palette if color != repeated]
        row_colors[middle - 1] = repeated
        row_colors[middle + 1] = repeated
        row_colors[middle] = random.choice(others)
        tie_info = {"src_row": middle, "color": repeated}

    if len(set(row_colors)) < 3:
        available_rows = list(range(height))
        random.shuffle(available_rows)
        for row, color in zip(available_rows, palette):
            row_colors[row] = color

    return row_colors, tie_info


def _choose_noise_color(stripe_palette: set[int]) -> int:
    choices = [color for color in NOISE_COLOR_POOL if color not in stripe_palette]
    assert choices
    return random.choice(choices)


def _empty_grid(row_colors: list[int], width: int) -> np.ndarray:
    return np.repeat(np.asarray(row_colors, dtype=int)[:, None], width, axis=1)


def _free_columns(width: int, forbidden: set[int]) -> list[int]:
    cols = [c for c in range(width) if c not in forbidden]
    random.shuffle(cols)
    return cols


def _plan_misplaced(
    grid: np.ndarray,
    row_colors: list[int],
    row_used_cols: list[set[int]],
    reserved_targets: set[tuple[int, int]],
    src_r: int,
    color: int,
) -> bool:
    target_r = _nearest_matching_row(row_colors, src_r, color)
    cols = _free_columns(grid.shape[1], row_used_cols[src_r])
    for col in cols:
        if (target_r, col) in reserved_targets:
            continue
        if target_r != src_r and col in row_used_cols[target_r]:
            continue
        grid[src_r, col] = int(color)
        row_used_cols[src_r].add(col)
        reserved_targets.add((int(target_r), int(col)))
        return True
    return False


def _plan_noise(
    grid: np.ndarray,
    row_used_cols: list[set[int]],
    src_r: int,
    stripe_palette: set[int],
) -> bool:
    cols = _free_columns(grid.shape[1], row_used_cols[src_r])
    if not cols:
        return False
    col = cols[0]
    grid[src_r, col] = _choose_noise_color(stripe_palette)
    row_used_cols[src_r].add(col)
    return True


def _validate_instance(
    grid: np.ndarray,
    expected_row_colors: list[int],
    want_noise: bool,
    tie_info: dict,
) -> bool:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if h > 30 or w > 30:
        return False

    row_colors, misplaced, noise = _analyze_grid(g)
    if row_colors != list(expected_row_colors):
        return False
    if len(set(row_colors)) < 3 or len(set(row_colors)) > 5:
        return False

    stripe_colors = set(row_colors)
    if not stripe_colors.issubset(set(STRIPE_COLOR_POOL)):
        return False

    for r in range(h):
        counts = Counter(int(v) for v in g[r])
        if counts[row_colors[r]] <= int(np.floor(0.7 * w)):
            return False
        off_count = sum(1 for c in range(w) if int(g[r, c]) != row_colors[r])
        if off_count > 3:
            return False

    if want_noise and not noise:
        return False
    if not misplaced:
        return False

    seen_targets = set()
    for item in misplaced:
        src_r, src_c = item["src"]
        target_r, target_c = item["target"]
        if item["color"] == row_colors[src_r]:
            return False
        if target_c != src_c:
            return False
        if (target_r, target_c) in seen_targets:
            return False
        seen_targets.add((target_r, target_c))

    if tie_info:
        src_row = int(tie_info["src_row"])
        repeated = int(tie_info["color"])
        candidates = [r for r, color in enumerate(row_colors) if color == repeated]
        if src_row - 1 not in candidates or src_row + 1 not in candidates:
            return False
        tie_hits = [
            item
            for item in misplaced
            if item["src"][0] == src_row and item["color"] == repeated and item["target"][0] == src_row + 1
        ]
        if not tie_hits:
            return False

    out = main(g)
    expected = np.repeat(np.asarray(row_colors, dtype=int)[:, None], w, axis=1)
    if not np.array_equal(out, expected):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height, width = _sample_dimensions()
    want_tie = random.random() < 0.55
    want_noise = random.random() < 0.8
    want_double = random.random() < 0.55

    row_colors, tie_info = _sample_row_colors(height, want_tie=want_tie)
    stripe_palette = set(row_colors)
    grid = _empty_grid(row_colors, width)
    row_used_cols = [set() for _ in range(height)]
    reserved_targets: set[tuple[int, int]] = set()

    if tie_info:
        if not _plan_misplaced(
            grid,
            row_colors,
            row_used_cols,
            reserved_targets,
            int(tie_info["src_row"]),
            int(tie_info["color"]),
        ):
            return None

    if want_noise:
        noise_row = random.randrange(height)
        if not _plan_noise(grid, row_used_cols, noise_row, stripe_palette):
            return None

    if want_double:
        candidate_rows = [r for r in range(height) if len({color for color in stripe_palette if color != row_colors[r]}) >= 2]
        random.shuffle(candidate_rows)
        placed = False
        for src_r in candidate_rows:
            colors = [color for color in stripe_palette if color != row_colors[src_r]]
            random.shuffle(colors)
            first_ok = _plan_misplaced(grid, row_colors, row_used_cols, reserved_targets, src_r, colors[0])
            second_ok = first_ok and _plan_misplaced(grid, row_colors, row_used_cols, reserved_targets, src_r, colors[1])
            if second_ok:
                placed = True
                break
        if not placed:
            return None

    total_rows = list(range(height))
    random.shuffle(total_rows)
    for src_r in total_rows:
        current_off = len(row_used_cols[src_r])
        max_more = 3 - current_off
        if max_more <= 0:
            continue

        possible_colors = [color for color in stripe_palette if color != row_colors[src_r]]
        random.shuffle(possible_colors)
        extra_misplaced = random.choices([0, 1, 2], weights=[4, 4, 2], k=1)[0]
        extra_misplaced = min(extra_misplaced, max_more)
        placed_count = 0
        for color in possible_colors:
            if placed_count >= extra_misplaced:
                break
            if _plan_misplaced(grid, row_colors, row_used_cols, reserved_targets, src_r, color):
                placed_count += 1

        if len(row_used_cols[src_r]) < 3 and random.random() < 0.25:
            _plan_noise(grid, row_used_cols, src_r, stripe_palette)

    if not _validate_instance(grid, row_colors, want_noise=want_noise, tie_info=tie_info):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stripe-relocation input")
