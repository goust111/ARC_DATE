from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

RANK_COLORS = {
    1: [BLUE],
    2: [BLUE, RED],
    3: [BLUE, GREEN, RED],
    4: [BLUE, GREEN, YELLOW, RED],
}

MAX_GENERATION_ATTEMPTS = 300
BLOCK_ROWS_RANGE = (2, 6)
BLOCK_COLS_RANGE = (2, 6)
TEMPLATE_COUNT_CHOICES = (2, 3, 4)
INPUT_COLORS = (0, 5, 6, 7, 8, 9)


def _analyze_blocks(grid: np.ndarray) -> tuple[list[tuple[int, int, tuple[int, ...]]], Counter, list[tuple[int, ...]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height >= 2 and width >= 2, "grid must contain at least one 2x2 block"
    assert height % 2 == 0 and width % 2 == 0, "grid shape must be divisible into 2x2 blocks"

    placements: list[tuple[int, int, tuple[int, ...]]] = []
    counts: Counter[tuple[int, ...]] = Counter()
    first_seen: dict[tuple[int, ...], int] = {}

    order = 0
    for r in range(0, height, 2):
        for c in range(0, width, 2):
            signature = tuple(map(int, g[r : r + 2, c : c + 2].reshape(-1)))
            placements.append((r, c, signature))
            counts[signature] += 1
            if signature not in first_seen:
                first_seen[signature] = order
            order += 1

    assert 1 <= len(counts) <= 4, "the rule defines at most four distinct 2x2 templates"
    ordered = sorted(counts, key=lambda signature: (-counts[signature], first_seen[signature]))
    return placements, counts, ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each 2x2 template by a solid 2x2 rarity-color block.

    The grid is partitioned directly into 2x2 blocks. Equal blocks are the same
    template. Templates are ranked by occurrence count (descending), breaking
    ties by the first row-major position where each template appears.

    Color mapping:
    - 2 templates: most common -> blue, rarest -> red
    - 3 templates: blue, green, red
    - 4 templates: blue, green, yellow, red
    """
    g = np.asarray(grid, dtype=int)
    placements, _counts, ordered = _analyze_blocks(g)
    rank_colors = RANK_COLORS[len(ordered)]
    fill_by_template = {
        signature: rank_colors[index] for index, signature in enumerate(ordered)
    }

    out = np.empty_like(g)
    for r, c, signature in placements:
        out[r : r + 2, c : c + 2] = fill_by_template[signature]
    return out


def _random_sorted_parts(total: int, parts: int, *, positive: bool) -> list[int]:
    values = [1 if positive else 0 for _ in range(parts)]
    remaining = total - sum(values)
    if remaining < 0:
        raise ValueError("invalid partition request")
    for _ in range(remaining):
        values[random.randrange(parts)] += 1
    values.sort(reverse=True)
    return values


def _distinct_counts(total_blocks: int, template_count: int) -> list[int] | None:
    minimum_total = template_count * (template_count + 1) // 2
    if total_blocks < minimum_total:
        return None

    base = list(range(template_count, 0, -1))
    extras = _random_sorted_parts(total_blocks - minimum_total, template_count, positive=False)
    counts = [base_count + extra for base_count, extra in zip(base, extras)]
    assert sum(counts) == total_blocks
    assert all(counts[i] > counts[i + 1] for i in range(len(counts) - 1))
    return counts


def _tied_counts(total_blocks: int, template_count: int) -> list[int] | None:
    for _ in range(200):
        counts = _random_sorted_parts(total_blocks, template_count, positive=True)
        if any(counts[i] == counts[i + 1] for i in range(len(counts) - 1)):
            return counts
    return None


def _choose_counts(total_blocks: int, template_count: int, tie_mode: bool) -> list[int] | None:
    if tie_mode:
        return _tied_counts(total_blocks, template_count)
    return _distinct_counts(total_blocks, template_count)


def _random_template(bg: int, accent_pool: list[int]) -> tuple[int, ...]:
    mode = random.choice(("solid", "single", "pair_same", "pair_distinct"))
    cells = [bg, bg, bg, bg]

    if mode == "solid":
        return tuple(cells)

    if mode == "single":
        cells[random.randrange(4)] = random.choice(accent_pool)
        return tuple(cells)

    p0, p1 = random.sample(range(4), 2)
    if mode == "pair_same":
        color = random.choice(accent_pool)
        cells[p0] = color
        cells[p1] = color
        return tuple(cells)

    c0, c1 = random.sample(accent_pool, 2)
    cells[p0] = c0
    cells[p1] = c1
    return tuple(cells)


def _make_templates(bg: int, template_count: int) -> list[tuple[int, ...]] | None:
    accent_pool = [color for color in INPUT_COLORS if color != bg]
    templates: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    for _ in range(200):
        while len(templates) < template_count:
            template = _random_template(bg, accent_pool)
            if template in seen:
                continue
            templates.append(template)
            seen.add(template)
        if len(templates) == template_count:
            return templates

    return None


def _first_template_order(block_ids: np.ndarray) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    for value in block_ids.reshape(-1):
        block_id = int(value)
        if block_id not in seen:
            seen.add(block_id)
            order.append(block_id)
    return order


def _build_grid(
    block_rows: int,
    block_cols: int,
    templates: list[tuple[int, ...]],
    counts: list[int],
) -> np.ndarray:
    ids = [template_id for template_id, count in enumerate(counts) for _ in range(count)]
    random.shuffle(ids)
    block_ids = np.array(ids, dtype=int).reshape(block_rows, block_cols)

    grid = np.empty((block_rows * 2, block_cols * 2), dtype=int)
    for br in range(block_rows):
        for bc in range(block_cols):
            block = np.array(templates[int(block_ids[br, bc])], dtype=int).reshape(2, 2)
            grid[2 * br : 2 * br + 2, 2 * bc : 2 * bc + 2] = block
    return grid


def _valid_background_dominance(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.reshape(-1)))
    bg_count = counts.get(int(bg), 0)
    return all(color == int(bg) or count < bg_count for color, count in counts.items())


def _branch_metadata(grid: np.ndarray) -> tuple[int, bool]:
    placements, counts, ordered = _analyze_blocks(grid)
    del placements
    values = [counts[signature] for signature in ordered]
    tie_present = any(values[i] == values[i + 1] for i in range(len(values) - 1))
    return len(ordered), tie_present


def _gen_attempt() -> np.ndarray | None:
    template_count = random.choice(TEMPLATE_COUNT_CHOICES)
    tie_mode = random.choice((False, True))

    block_rows = random.randint(*BLOCK_ROWS_RANGE)
    block_cols = random.randint(*BLOCK_COLS_RANGE)
    total_blocks = block_rows * block_cols
    counts = _choose_counts(total_blocks, template_count, tie_mode)
    if counts is None:
        return None

    bg = random.choice(INPUT_COLORS)
    templates = _make_templates(bg, template_count)
    if templates is None:
        return None

    grid = _build_grid(block_rows, block_cols, templates, counts)
    if not _valid_background_dominance(grid, bg):
        return None

    actual_template_count, actual_tie_mode = _branch_metadata(grid)
    if actual_template_count != template_count or actual_tie_mode != tie_mode:
        return None

    out = main(grid)
    expected_colors = set(RANK_COLORS[template_count])
    if set(map(int, np.unique(out))) != expected_colors:
        return None

    for r in range(0, out.shape[0], 2):
        for c in range(0, out.shape[1], 2):
            if np.unique(out[r : r + 2, c : c + 2]).size != 1:
                return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
