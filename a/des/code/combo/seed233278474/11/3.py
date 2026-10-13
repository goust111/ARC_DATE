from __future__ import annotations

from collections import Counter

import numpy as np


BLACK = 0
GRAY = 5
VOTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _detect_line_color(body: np.ndarray) -> int:
    row_counts: Counter[int] = Counter()
    col_counts: Counter[int] = Counter()

    for r in range(body.shape[0]):
        vals = np.unique(body[r])
        if len(vals) == 1:
            row_counts[int(vals[0])] += 1

    for c in range(body.shape[1]):
        vals = np.unique(body[:, c])
        if len(vals) == 1:
            col_counts[int(vals[0])] += 1

    candidates = [color for color in row_counts if col_counts[color] > 0]
    assert candidates
    return max(
        candidates,
        key=lambda color: (row_counts[color] + col_counts[color], row_counts[color], col_counts[color]),
    )


def _spans_from_lines(line_mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for idx, is_line in enumerate(line_mask):
        if is_line:
            if start < idx:
                spans.append((int(start), int(idx)))
            start = idx + 1
    if start < len(line_mask):
        spans.append((int(start), int(len(line_mask))))
    assert spans
    return spans


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grid cell region by majority vote, using the bottom legend row as the tie-break order.

    Input structure:
    - Black background.
    - Complete gray horizontal/vertical lines partition the main area into rectangles.
    - The last row, below the final gray horizontal line, is a legend of single-pixel colors.

    Rule:
    - For each rectangular region, count non-black, non-gray pixels inside it.
    - Use the most frequent color; if tied, choose the color that appears farther left in the legend.
    - Clear the region and fill it entirely with the winning color.
    - Empty regions stay black.
    - Remove the legend row in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] >= 2

    body = g[:-1].copy()
    legend = g[-1]

    line_color = _detect_line_color(body)
    row_lines = np.all(body == int(line_color), axis=1)
    col_lines = np.all(body == int(line_color), axis=0)
    assert bool(row_lines.any())
    assert bool(col_lines.any())

    bg_counts = Counter(map(int, body[body != int(line_color)].ravel()))
    assert bg_counts
    bg = bg_counts.most_common(1)[0][0]

    priority = [int(v) for v in legend.tolist() if int(v) not in {int(bg), int(line_color)}]
    assert priority
    assert len(priority) == len(set(priority))
    rank = {color: idx for idx, color in enumerate(priority)}

    row_spans = _spans_from_lines(row_lines)
    col_spans = _spans_from_lines(col_lines)

    out = body.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            patch = body[r0:r1, c0:c1]
            votes = patch[(patch != int(bg)) & (patch != int(line_color))]
            if votes.size == 0:
                out[r0:r1, c0:c1] = int(bg)
                continue

            counts = Counter(map(int, votes.tolist()))
            for color in counts:
                assert color in rank
            best = max(counts.values())
            winners = [color for color, count in counts.items() if count == best]
            fill = min(winners, key=lambda color: rank[color])
            out[r0:r1, c0:c1] = int(fill)

    return out


def _build_layout(
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    n_rows = int(rng.integers(1, 5))
    n_cols = int(rng.integers(2, 5))
    row_heights = rng.integers(2, 6, size=n_rows)
    col_widths = rng.integers(2, 7, size=n_cols)

    top_border = bool(rng.integers(0, 2))
    left_border = bool(rng.integers(0, 2))
    right_border = bool(rng.integers(0, 2))

    body_h = int(row_heights.sum()) + (n_rows - 1) + 1 + int(top_border)
    body_w = int(col_widths.sum()) + (n_cols - 1) + int(left_border) + int(right_border)
    body = np.full((body_h, body_w), BLACK, dtype=int)

    row_spans: list[tuple[int, int]] = []
    col_spans: list[tuple[int, int]] = []
    row_lines: list[int] = []
    col_lines: list[int] = []

    r = 0
    if top_border:
        row_lines.append(r)
        r += 1
    for idx, height in enumerate(row_heights.tolist()):
        r0 = r
        r += int(height)
        row_spans.append((r0, r))
        row_lines.append(r)
        if idx < n_rows - 1:
            r += 1

    c = 0
    if left_border:
        col_lines.append(c)
        c += 1
    for idx, width in enumerate(col_widths.tolist()):
        c0 = c
        c += int(width)
        col_spans.append((c0, c))
        if idx < n_cols - 1 or right_border:
            col_lines.append(c)
            c += 1

    body[row_lines, :] = GRAY
    body[:, col_lines] = GRAY
    return body, row_spans, col_spans


def _legend_setup(
    rng: np.random.Generator,
    width: int,
) -> tuple[np.ndarray, list[int], dict[int, int]]:
    max_colors = min(len(VOTE_COLORS), width)
    min_colors = min(3, max_colors)
    n_colors = int(rng.integers(min_colors, max_colors + 1))
    priority = rng.choice(np.array(VOTE_COLORS, dtype=int), size=n_colors, replace=False).tolist()
    slots = sorted(rng.choice(width, size=n_colors, replace=False).tolist())

    legend = np.full(width, BLACK, dtype=int)
    for col, color in zip(slots, priority):
        legend[int(col)] = int(color)
    return legend, [int(color) for color in priority], {int(color): idx for idx, color in enumerate(priority)}


def _pick_modes(rng: np.random.Generator, region_count: int) -> list[str]:
    modes = ["unique"] * region_count

    if region_count >= 2 and bool(rng.integers(0, 2)):
        empty_idx = int(rng.integers(region_count))
        modes[empty_idx] = "empty"

    if bool(rng.integers(0, 2)):
        tie_candidates = [idx for idx, mode in enumerate(modes) if mode != "empty"]
        tie_idx = int(rng.choice(tie_candidates))
        modes[tie_idx] = "tie"

    for idx, mode in enumerate(modes):
        if mode != "unique":
            continue
        roll = float(rng.random())
        if roll < 0.18:
            modes[idx] = "empty"
        elif roll < 0.40:
            modes[idx] = "tie"

    if all(mode == "empty" for mode in modes):
        modes[int(rng.integers(region_count))] = "unique"
    return modes


def _region_votes(
    rng: np.random.Generator,
    shape: tuple[int, int],
    priority: list[int],
    rank: dict[int, int],
    mode: str,
) -> tuple[np.ndarray, int]:
    h, w = map(int, shape)
    area = int(h * w)
    patch = np.full((h, w), BLACK, dtype=int)

    if mode == "empty":
        return patch, BLACK

    coords = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(coords)
    pos = 0

    if mode == "tie":
        max_tied = min(3, len(priority), area)
        tied_n = int(rng.integers(2, max_tied + 1))
        tied_colors = rng.choice(np.array(priority, dtype=int), size=tied_n, replace=False).tolist()
        max_count = max(1, min(2, area // tied_n))
        tied_count = int(rng.integers(1, max_count + 1))
        for color in tied_colors:
            for _ in range(tied_count):
                r, c = coords[pos]
                pos += 1
                patch[int(r), int(c)] = int(color)
        winner = min((int(color) for color in tied_colors), key=lambda color: rank[color])
        return patch, winner

    winner = int(rng.choice(np.array(priority, dtype=int)))
    max_points = min(area, 6)
    winner_count = int(rng.integers(1, min(4, max_points) + 1))
    for _ in range(winner_count):
        r, c = coords[pos]
        pos += 1
        patch[int(r), int(c)] = int(winner)

    remaining_capacity = max_points - winner_count
    if remaining_capacity > 0 and len(priority) > 1 and winner_count > 1:
        other_colors = [color for color in priority if int(color) != int(winner)]
        other_n = int(rng.integers(0, min(2, len(other_colors), remaining_capacity) + 1))
        if other_n > 0:
            chosen = rng.choice(np.array(other_colors, dtype=int), size=other_n, replace=False).tolist()
            other_counts = {int(color): 0 for color in chosen}
            remaining = int(rng.integers(1, min(remaining_capacity, other_n * (winner_count - 1)) + 1))
            while remaining > 0:
                available = [
                    color
                    for color in chosen
                    if other_counts[int(color)] < winner_count - 1
                ]
                color = int(rng.choice(np.array(available, dtype=int)))
                other_counts[color] += 1
                remaining -= 1
            for color, count in other_counts.items():
                for _ in range(count):
                    r, c = coords[pos]
                    pos += 1
                    patch[int(r), int(c)] = int(color)

    return patch, winner


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    body, row_spans, col_spans = _build_layout(rng)
    legend, priority, rank = _legend_setup(rng, body.shape[1])

    regions = [(rspan, cspan) for rspan in row_spans for cspan in col_spans]
    modes = _pick_modes(rng, len(regions))
    rng.shuffle(modes)

    expected = body.copy()
    for mode, ((r0, r1), (c0, c1)) in zip(modes, regions):
        patch, fill = _region_votes(rng, (r1 - r0, c1 - c0), priority, rank, mode)
        body[r0:r1, c0:c1] = patch
        expected[r0:r1, c0:c1] = int(fill)

    inp = np.vstack([body, legend])
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
