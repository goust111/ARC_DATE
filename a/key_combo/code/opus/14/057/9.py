from __future__ import annotations

from collections import Counter
import random

import numpy as np


PANEL_SIZE = 10
INPUT_SHAPE = (10, 22)
BACKGROUND = 0
SEPARATOR = 5
MASK_COLOR = 9
LEFT_COLOR_POOL = (1, 2, 3, 4, 7)
GENERATION_RETRIES = 600


def _extract_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    height, width = g.shape
    panel_size = int(height)
    assert width >= 2 * panel_size + 1
    assert np.all(g[:, panel_size] == SEPARATOR)
    if width > 2 * panel_size + 1:
        assert np.all(g[:, 2 * panel_size + 1 :] == BACKGROUND)

    left = g[:, :panel_size]
    right = g[:, panel_size + 1 : 2 * panel_size + 1]
    return left.copy(), right.copy()


def _selected_counts(left: np.ndarray, right: np.ndarray) -> Counter[int]:
    assert left.shape == right.shape
    assert set(map(int, np.unique(right))) <= {BACKGROUND, MASK_COLOR}

    selected = left[right == MASK_COLOR]
    counts = Counter(int(color) for color in selected if int(color) != BACKGROUND)
    assert counts
    return counts


def _ranked_colors(counts: Counter[int]) -> list[int]:
    values = list(counts.values())
    assert len(set(values)) == len(values), "expected unique frequency ranks"
    return [color for color, _count in sorted(counts.items(), key=lambda item: -item[1])]


def _pack_counts(counts: Counter[int], panel_size: int) -> np.ndarray:
    ordered = _ranked_colors(counts)
    out = np.zeros((panel_size, panel_size), dtype=int)
    scan = [(r, c) for c in range(panel_size) for r in range(panel_size - 1, -1, -1)]

    idx = 0
    for color in ordered:
        for _ in range(int(counts[color])):
            assert idx < len(scan)
            r, c = scan[idx]
            out[r, c] = int(color)
            idx += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Filter the left panel by the right-panel mask, then frequency-pack the selected colors into bottom-up columns.

    Input structure:
    - The input is a square left panel, one full grey separator column, and a same-size right panel mask; any extra columns are black padding.
    - The left panel is densely filled with non-black foreground colors.
    - The right panel contains only black and maroon pixels, where maroon marks the sampled coordinates.

    Rule:
    - For every maroon mask cell in the right panel, read the color at the same `(r, c)` position in the left panel.
    - Count the selected non-black colors and sort them by strictly descending frequency.
    - Output a single panel of the same square size, filling colors in that order while scanning columns left-to-right and, within each column, bottom-to-top.
    - Leave every unused output cell black.

    Key insight:
    The mask is not a crop target: it defines which source pixels enter the frequency count, and that filtered count order fully determines the packed output.
    """
    left, right = _extract_panels(grid)
    counts = _selected_counts(left, right)
    return _pack_counts(counts, left.shape[0])


def _random_composition(total: int, n_parts: int) -> list[int]:
    assert total >= 0 and n_parts >= 1
    if n_parts == 1:
        return [int(total)]
    if total == 0:
        return [0] * int(n_parts)

    cuts = sorted(random.sample(range(total + n_parts - 1), n_parts - 1))
    parts: list[int] = []
    prev = -1
    for cut in cuts + [total + n_parts - 1]:
        parts.append(int(cut - prev - 1))
        prev = int(cut)
    assert sum(parts) == total
    return parts


def _sample_descending_distinct_counts(total: int, n_colors: int, min_each: int = 3) -> list[int] | None:
    base = [int(min_each + n_colors - 1 - i) for i in range(n_colors)]
    remaining = int(total) - sum(base)
    if remaining < 0:
        return None

    increments = sorted(_random_composition(remaining, n_colors), reverse=True)
    counts = [int(b + inc) for b, inc in zip(base, increments)]
    if any(counts[i] <= counts[i + 1] for i in range(len(counts) - 1)):
        return None
    return counts


def _has_mid_column_boundary(ordered_counts: list[int], panel_size: int) -> bool:
    prefix = 0
    for count in ordered_counts[:-1]:
        prefix += int(count)
        if prefix % panel_size != 0:
            return True
    return False


def _make_noise_mask(area: int, panel_size: int) -> np.ndarray:
    mask = np.zeros((panel_size, panel_size), dtype=bool)
    cells = [(r, c) for r in range(panel_size) for c in range(panel_size)]
    for r, c in random.sample(cells, area):
        mask[r, c] = True
    return mask


def _make_blob_mask(area: int, panel_size: int) -> np.ndarray:
    mask = np.zeros((panel_size, panel_size), dtype=bool)
    seed = (random.randrange(panel_size), random.randrange(panel_size))
    mask[seed] = True
    frontier = {seed}

    while int(mask.sum()) < area:
        r, c = random.choice(tuple(frontier))
        neighbors = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < panel_size and 0 <= nc < panel_size and not mask[nr, nc]:
                neighbors.append((nr, nc))
        if neighbors:
            nr, nc = random.choice(neighbors)
            mask[nr, nc] = True
            frontier.add((nr, nc))
        else:
            frontier.remove((r, c))
            if not frontier:
                remaining = [(rr, cc) for rr in range(panel_size) for cc in range(panel_size) if not mask[rr, cc]]
                rr, cc = random.choice(remaining)
                mask[rr, cc] = True
                frontier.add((rr, cc))
    return mask


def _make_banded_mask(area: int, panel_size: int) -> np.ndarray:
    row_weight = np.array([random.uniform(-1.5, 1.5) for _ in range(panel_size)])
    col_weight = np.array([random.uniform(-1.5, 1.5) for _ in range(panel_size)])
    noise = np.random.uniform(-0.35, 0.35, size=(panel_size, panel_size))
    score = row_weight[:, None] + col_weight[None, :] + noise

    flat_idx = np.argsort(score, axis=None)[-int(area) :]
    mask = np.zeros((panel_size, panel_size), dtype=bool)
    mask[np.unravel_index(flat_idx, mask.shape)] = True
    return mask


def _sample_mask(area: int, panel_size: int) -> np.ndarray:
    mode = random.choice(("noise", "blob", "banded"))
    if mode == "noise":
        return _make_noise_mask(area, panel_size)
    if mode == "blob":
        return _make_blob_mask(area, panel_size)
    return _make_banded_mask(area, panel_size)


def _sample_selected_plan(panel_size: int) -> tuple[list[int], dict[int, int], list[int]] | None:
    n_colors = random.choice((3, 4, 5))
    total_selected = random.randint(30, 60)
    ordered_counts = _sample_descending_distinct_counts(total_selected, n_colors, min_each=3)
    if ordered_counts is None:
        return None
    if max(ordered_counts) <= panel_size:
        return None
    if not _has_mid_column_boundary(ordered_counts, panel_size):
        return None

    colors = random.sample(list(LEFT_COLOR_POOL), n_colors)
    color_order = colors[:]
    random.shuffle(color_order)
    selected_map = {int(color): int(count) for color, count in zip(color_order, ordered_counts)}
    return colors, selected_map, ordered_counts


def _sample_total_counts(selected_map: dict[int, int], panel_area: int) -> dict[int, int] | None:
    colors = list(selected_map)
    selected_rank = [color for color, _count in sorted(selected_map.items(), key=lambda item: -item[1])]
    if len(selected_rank) < 2:
        return None

    remaining = int(panel_area) - sum(selected_map.values())
    if remaining <= 0:
        return None

    bonus_color = random.choice(selected_rank[1:])
    top_color = selected_rank[0]
    min_bonus = selected_map[top_color] - selected_map[bonus_color] + 1
    if min_bonus > remaining:
        return None

    bonus_extra = random.randint(min_bonus, remaining)
    other_colors = [color for color in colors if color != bonus_color]
    extras = {int(bonus_color): int(bonus_extra)}
    pieces = _random_composition(remaining - bonus_extra, len(other_colors))
    random.shuffle(pieces)
    for color, piece in zip(other_colors, pieces):
        extras[int(color)] = int(piece)

    totals = {int(color): int(selected_map[color] + extras[color]) for color in colors}
    if len(set(totals.values())) != len(totals):
        return None

    total_rank = [color for color, _count in sorted(totals.items(), key=lambda item: -item[1])]
    if total_rank == selected_rank:
        return None
    return totals


def _fill_positions(positions: list[tuple[int, int]], counts: dict[int, int]) -> dict[tuple[int, int], int]:
    shuffled_positions = positions[:]
    random.shuffle(shuffled_positions)

    color_stream: list[int] = []
    colors = list(counts)
    random.shuffle(colors)
    for color in colors:
        color_stream.extend([int(color)] * int(counts[color]))

    random.shuffle(color_stream)
    assert len(color_stream) == len(shuffled_positions)
    return {pos: int(color) for pos, color in zip(shuffled_positions, color_stream)}


def _build_left_panel(mask: np.ndarray, selected_map: dict[int, int], totals: dict[int, int]) -> np.ndarray:
    panel_size = int(mask.shape[0])
    left = np.zeros((panel_size, panel_size), dtype=int)

    masked_positions = [tuple(map(int, pos)) for pos in np.argwhere(mask)]
    unmasked_positions = [tuple(map(int, pos)) for pos in np.argwhere(~mask)]
    outside_counts = {int(color): int(totals[color] - selected_map[color]) for color in totals}

    masked_fill = _fill_positions(masked_positions, selected_map)
    unmasked_fill = _fill_positions(unmasked_positions, outside_counts)

    for (r, c), color in masked_fill.items():
        left[r, c] = int(color)
    for (r, c), color in unmasked_fill.items():
        left[r, c] = int(color)

    assert np.all(left != BACKGROUND)
    return left


def _render_input(left: np.ndarray, mask: np.ndarray) -> np.ndarray:
    grid = np.zeros(INPUT_SHAPE, dtype=int)
    grid[:, :PANEL_SIZE] = left
    grid[:, PANEL_SIZE] = SEPARATOR
    grid[:, PANEL_SIZE + 1 : 2 * PANEL_SIZE + 1] = np.where(mask, MASK_COLOR, BACKGROUND)
    return grid


def _validate_semantics(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.shape == INPUT_SHAPE
    assert np.all(grid[:, PANEL_SIZE] == SEPARATOR)
    assert np.all(grid[:, 2 * PANEL_SIZE + 1 :] == BACKGROUND)

    left, right = _extract_panels(grid)
    assert left.shape == (PANEL_SIZE, PANEL_SIZE)
    assert right.shape == (PANEL_SIZE, PANEL_SIZE)
    assert np.all(left != BACKGROUND)
    assert set(map(int, np.unique(left))) <= set(LEFT_COLOR_POOL)
    assert set(map(int, np.unique(right))) <= {BACKGROUND, MASK_COLOR}

    selected_area = int(np.count_nonzero(right == MASK_COLOR))
    assert 30 <= selected_area <= 60

    selected_counts = _selected_counts(left, right)
    assert 3 <= len(selected_counts) <= 5
    assert len(set(selected_counts.values())) == len(selected_counts)

    total_counts = Counter(map(int, left.ravel()))
    assert len(total_counts) == len(selected_counts)
    assert len(set(total_counts.values())) == len(total_counts)

    selected_rank = _ranked_colors(selected_counts)
    total_rank = [color for color, _count in sorted(total_counts.items(), key=lambda item: -item[1])]
    assert selected_rank != total_rank, "mask must change the frequency order"

    ordered_counts = [int(selected_counts[color]) for color in selected_rank]
    assert max(ordered_counts) > PANEL_SIZE
    assert _has_mid_column_boundary(ordered_counts, PANEL_SIZE)

    out = main(grid)
    assert out.shape == (PANEL_SIZE, PANEL_SIZE)
    assert np.count_nonzero(out) == sum(selected_counts.values())
    assert np.array_equal(out, _pack_counts(selected_counts, PANEL_SIZE))


def _gen_attempt() -> np.ndarray | None:
    selected_plan = _sample_selected_plan(PANEL_SIZE)
    if selected_plan is None:
        return None
    _colors, selected_map, _ordered_counts = selected_plan

    mask_area = sum(selected_map.values())
    mask = _sample_mask(mask_area, PANEL_SIZE)
    totals = _sample_total_counts(selected_map, PANEL_SIZE * PANEL_SIZE)
    if totals is None:
        return None

    left = _build_left_panel(mask, selected_map, totals)
    grid = _render_input(left, mask)
    _validate_semantics(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
