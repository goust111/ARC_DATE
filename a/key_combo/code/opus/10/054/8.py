from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
DIVIDER = 5
PALETTE = (1, 2, 3, 4, 7, 8)
GRID_MIN = 13
GRID_MAX = 19
MIN_MARGIN = 4
OUTPUT_SIZE = 2
MAX_GENERATION_ATTEMPTS = 600
COUNT_ATTEMPTS = 300


def _full_divider_rows(grid: np.ndarray) -> list[int]:
    return [int(r) for r in np.where(np.all(grid == DIVIDER, axis=1))[0]]


def _full_divider_cols(grid: np.ndarray) -> list[int]:
    return [int(c) for c in np.where(np.all(grid == DIVIDER, axis=0))[0]]


def _quadrant_boxes(shape: tuple[int, int], divider_r: int, divider_c: int) -> list[tuple[int, int, int, int]]:
    h, w = map(int, shape)
    boxes = [
        (0, divider_r, 0, divider_c),
        (0, divider_r, divider_c + 1, w),
        (divider_r + 1, h, 0, divider_c),
        (divider_r + 1, h, divider_c + 1, w),
    ]
    for r0, r1, c0, c1 in boxes:
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "each quadrant must have at least two rows and two columns"
    return boxes


def _split_box_rows(box: tuple[int, int, int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, r1, _c0, _c1 = box
    mid = r0 + (r1 - r0) // 2
    assert r0 < mid < r1, "quadrant needs non-empty upper and lower bands"
    return (r0, mid), (mid, r1)


def _box_colors(grid: np.ndarray, box: tuple[int, int, int, int]) -> list[int]:
    r0, r1, c0, c1 = box
    values = set(map(int, grid[r0:r1, c0:c1].ravel()))
    return sorted(values - {BG, DIVIDER})


def _quadrant_scores(grid: np.ndarray, box: tuple[int, int, int, int]) -> dict[int, int]:
    (top0, top1), (bot0, bot1) = _split_box_rows(box)
    _r0, _r1, c0, c1 = box
    scores: dict[int, int] = {}
    for color in _box_colors(grid, box):
        above = int(np.count_nonzero(grid[top0:top1, c0:c1] == color))
        below = int(np.count_nonzero(grid[bot0:bot1, c0:c1] == color))
        scores[int(color)] = below - above
    assert scores, "each quadrant must contain at least one colored marker"
    return scores


def _winner_from_scores(scores: dict[int, int]) -> int:
    best = max(scores.values())
    winners = [color for color, score in scores.items() if score == best]
    assert len(winners) == 1, "each quadrant must have a unique winning color"
    return int(winners[0])


def _global_growth_winner(grid: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> int | None:
    totals: Counter[int] = Counter()
    for box in boxes:
        for color, score in _quadrant_scores(grid, box).items():
            totals[int(color)] += int(score)
    assert totals
    best = max(totals.values())
    winners = [color for color, score in totals.items() if score == best]
    if len(winners) != 1:
        return None
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pick one winning color per quadrant by comparing its lower-band and upper-band counts.

    Input structure:
    - The background is black.
    - One full-width gray row and one full-height gray column cross inside the grid.
    - The crossing gray lines partition the colored markers into four rectangular quadrants.
    - Each quadrant contains 2-3 scattered single-pixel colors chosen from a small palette.

    Rule:
    - Detect the unique gray horizontal divider and gray vertical divider.
    - Treat each of the four non-gray rectangles around that cross as one independent quadrant.
    - Within each quadrant, split the quadrant's rows into an upper band and a lower band.
    - For every non-background, non-gray color in that quadrant, compute
      `lower_count - upper_count`.
    - Output a 2x2 grid whose cells are the uniquely maximal colors for
      top-left, top-right, bottom-left, and bottom-right.

    Key insight:
    - The gray cross does not select one global winner; it creates four separate
      local counting problems, each reduced to a single output cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider_rows = _full_divider_rows(g)
    divider_cols = _full_divider_cols(g)
    assert len(divider_rows) == 1, "expected one full-width divider row"
    assert len(divider_cols) == 1, "expected one full-height divider column"

    divider_r = divider_rows[0]
    divider_c = divider_cols[0]
    boxes = _quadrant_boxes(g.shape, divider_r, divider_c)

    out = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=int)
    for index, box in enumerate(boxes):
        qr, qc = divmod(index, OUTPUT_SIZE)
        out[qr, qc] = _winner_from_scores(_quadrant_scores(g, box))
    return out


def _candidate_pairs(diff: int, top_cap: int, bottom_cap: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    top_limit = min(4, int(top_cap))
    bottom_limit = min(5, int(bottom_cap))
    for above in range(top_limit + 1):
        below = above + int(diff)
        if 0 <= below <= bottom_limit and 1 <= above + below <= 6:
            pairs.append((int(above), int(below)))
    return pairs


def _sample_quadrant_counts(
    colors: list[int],
    top_cap: int,
    bottom_cap: int,
    *,
    force_majority_counterexample: bool,
    force_negative_loser: bool,
) -> dict[int, dict[str, int]] | None:
    for _ in range(COUNT_ATTEMPTS):
        winner = random.choice(colors)
        losers = [color for color in colors if color != winner]
        negative_color = random.choice(losers) if force_negative_loser else None
        majority_color = random.choice(losers) if force_majority_counterexample else None

        winner_diff = random.randint(2, 4)
        winner_options = sorted(
            _candidate_pairs(winner_diff, top_cap, bottom_cap),
            key=lambda pair: (pair[0] + pair[1], pair[0]),
        )
        if not winner_options:
            continue
        winner_pair = random.choice(winner_options[: min(3, len(winner_options))])
        spec: dict[int, dict[str, int]] = {
            int(winner): {
                "above": int(winner_pair[0]),
                "below": int(winner_pair[1]),
                "diff": int(winner_diff),
            }
        }

        ok = True
        for color in losers:
            if color == negative_color:
                diff = random.randint(-3, -1)
            elif color == majority_color:
                diff = random.randint(0, winner_diff - 1)
            else:
                diff = random.randint(-2, winner_diff - 1)

            options = _candidate_pairs(diff, top_cap, bottom_cap)
            if not options:
                ok = False
                break

            if color == majority_color:
                options = [pair for pair in options if pair[0] + pair[1] > winner_pair[0] + winner_pair[1]]
                options = sorted(options, key=lambda pair: (pair[0] + pair[1], pair[1]), reverse=True)
            else:
                random.shuffle(options)
            if not options:
                ok = False
                break

            pair = options[0]
            spec[int(color)] = {"above": int(pair[0]), "below": int(pair[1]), "diff": int(diff)}

        if not ok:
            continue

        used_top = sum(item["above"] for item in spec.values())
        used_bottom = sum(item["below"] for item in spec.values())
        if used_top > top_cap or used_bottom > bottom_cap:
            continue

        scores = {color: item["diff"] for color, item in spec.items()}
        if _winner_from_scores(scores) != winner:
            continue

        if force_majority_counterexample:
            totals = {color: item["above"] + item["below"] for color, item in spec.items()}
            if max(totals, key=totals.get) == winner:
                continue
        if force_negative_loser and all(item["diff"] >= 0 for color, item in spec.items() if color != winner):
            continue

        return spec
    return None


def _place_color_points(
    grid: np.ndarray,
    color: int,
    count: int,
    cells: list[tuple[int, int]],
) -> None:
    chosen = random.sample(cells, count)
    for cell in chosen:
        cells.remove(cell)
    for r, c in chosen:
        grid[r, c] = int(color)


def _boxes_to_expected(quadrant_specs: list[dict[int, dict[str, int]]]) -> np.ndarray:
    expected = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=int)
    for index, spec in enumerate(quadrant_specs):
        qr, qc = divmod(index, OUTPUT_SIZE)
        scores = {color: item["diff"] for color, item in spec.items()}
        expected[qr, qc] = _winner_from_scores(scores)
    return expected


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    divider_r = random.randint(MIN_MARGIN, height - MIN_MARGIN - 1)
    divider_c = random.randint(MIN_MARGIN, width - MIN_MARGIN - 1)
    boxes = _quadrant_boxes((height, width), divider_r, divider_c)

    forced_three = random.randrange(4)
    forced_two = random.choice([idx for idx in range(4) if idx != forced_three])
    forced_majority = random.randrange(4)
    forced_negative = random.randrange(4)

    quadrant_specs: list[dict[int, dict[str, int]]] = []
    quadrant_colors: list[list[int]] = []
    for index, box in enumerate(boxes):
        r0, r1, c0, c1 = box
        qh = r1 - r0
        qw = c1 - c0
        top_cap = (qh // 2) * qw
        bottom_cap = (qh - qh // 2) * qw
        if index == forced_three:
            n_colors = 3
        elif index == forced_two:
            n_colors = 2
        else:
            n_colors = random.choice((2, 3))
        colors = random.sample(PALETTE, n_colors)
        spec = _sample_quadrant_counts(
            colors,
            top_cap,
            bottom_cap,
            force_majority_counterexample=index == forced_majority,
            force_negative_loser=index == forced_negative,
        )
        if spec is None:
            return None
        quadrant_specs.append(spec)
        quadrant_colors.append(colors)

    expected = _boxes_to_expected(quadrant_specs)
    if len(set(map(int, expected.ravel()))) < 2:
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[divider_r, :] = DIVIDER
    grid[:, divider_c] = DIVIDER

    for box, spec in zip(boxes, quadrant_specs):
        (top0, top1), (bot0, bot1) = _split_box_rows(box)
        _r0, _r1, c0, c1 = box
        top_cells = [(r, c) for r in range(top0, top1) for c in range(c0, c1)]
        bottom_cells = [(r, c) for r in range(bot0, bot1) for c in range(c0, c1)]
        random.shuffle(top_cells)
        random.shuffle(bottom_cells)

        for color, item in random.sample(list(spec.items()), len(spec)):
            _place_color_points(grid, color, item["above"], top_cells)
            _place_color_points(grid, color, item["below"], bottom_cells)

    if not np.array_equal(_boxes_to_expected([{color: {"diff": score} for color, score in _quadrant_scores(grid, box).items()} for box in boxes]), expected):
        return None

    if not np.array_equal(main(grid), expected):
        return None

    global_winner = _global_growth_winner(grid, boxes)
    if global_winner is None or np.all(expected == global_winner):
        return None

    majority_counterexample = False
    for box in boxes:
        winner = _winner_from_scores(_quadrant_scores(grid, box))
        totals = {
            color: int(np.count_nonzero(grid[box[0] : box[1], box[2] : box[3]] == color))
            for color in _box_colors(grid, box)
        }
        if max(totals, key=totals.get) != winner:
            majority_counterexample = True
            break
    if not majority_counterexample:
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_attempt()
        if built is not None:
            grid, _expected = built
            return grid
    raise RuntimeError("failed to generate a valid quadrant-growth input")
