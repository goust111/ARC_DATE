from __future__ import annotations

import numpy as np


PANEL_SIZE = 5
GRID_SHAPE = (5, 23)
SEPARATOR_COLOR = 5
SEPARATOR_COLS = (5, 11, 17)
PANEL_STARTS = (0, 6, 12, 18)
PANEL_COLORS = (1, 2, 3, 4)
ALL_CELLS = tuple((r, c) for r in range(PANEL_SIZE) for c in range(PANEL_SIZE))
COMMON_FIXED_ORDERS = (
    (0, 1, 2, 3),
    (3, 2, 1, 0),
    (2, 3, 0, 1),
)
GEN_RETRIES = 400


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < PANEL_SIZE and 0 <= nc < PANEL_SIZE:
            out.append((nr, nc))
    return out


def _extract_panels(grid: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.shape == GRID_SHAPE, f"expected {GRID_SHAPE}, got {g.shape}"
    assert np.all(g[:, SEPARATOR_COLS] == SEPARATOR_COLOR), "separator columns must be solid grey"

    panels: list[np.ndarray] = []
    colors: list[int] = []
    for start in PANEL_STARTS:
        panel = g[:, start : start + PANEL_SIZE]
        panel_colors = sorted(set(map(int, np.unique(panel))) - {0})
        assert len(panel_colors) == 1, "each panel must contain exactly one foreground color"
        color = panel_colors[0]
        assert color in PANEL_COLORS, "panel color must be one of blue/red/green/yellow"
        panels.append(panel)
        colors.append(color)

    assert len(set(colors)) == 4, "panel colors must be distinct"
    return panels, colors


def _panel_counts(panels: list[np.ndarray]) -> list[int]:
    return [int(np.count_nonzero(panel)) for panel in panels]


def _front_order_from_counts(counts: list[int]) -> list[int]:
    assert len(set(counts)) == 4, "panel foreground counts must be pairwise distinct"
    return sorted(range(4), key=lambda idx: (-counts[idx], idx))


def _overlay_panels(panels: list[np.ndarray], front_order: list[int]) -> np.ndarray:
    out = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
    for panel_index in reversed(front_order):
        mask = panels[panel_index] != 0
        out[mask] = panels[panel_index][mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Overlay the four aligned panels by sorting them from densest to sparsest foreground.

    Input structure:
    - The input is a 5x23 grid split into four 5x5 panels by three full-height grey separator columns.
    - Each panel uses black as background and contains exactly one distinct foreground color from blue/red/green/yellow.
    - The four panels are already spatially aligned, so equal coordinates across panels refer to the same output location.

    Rule:
    - Count the non-black pixels in each panel.
    - Rank the panels from most non-black pixels to fewest; this ranking is the front-to-back layer order.
    - Start from a black 5x5 canvas and overlay panels from back to front, treating black as transparent.

    Key insight:
    - The layer priority is not encoded by panel position; it is encoded implicitly by each panel's foreground density, so counting is required before overlaying.
    """
    panels, _ = _extract_panels(grid)
    counts = _panel_counts(panels)
    front_order = _front_order_from_counts(counts)
    return _overlay_panels(panels, front_order)


def _sample_counts(rng: np.random.Generator) -> list[int] | None:
    mode = int(rng.integers(0, 3))
    if mode == 0:
        values = [3]
        pool = [v for v in range(4, 11)]
        values.extend(rng.choice(pool, size=3, replace=False).tolist())
    elif mode == 1:
        values = [10]
        pool = [v for v in range(3, 10)]
        values.extend(rng.choice(pool, size=3, replace=False).tolist())
    else:
        values = rng.choice(np.arange(3, 11), size=4, replace=False).tolist()

    values = sorted(map(int, values), reverse=True)
    if values[0] - values[-1] < 3:
        return None
    return values


def _sample_cluster(
    rng: np.random.Generator,
    available: set[tuple[int, int]],
    count: int,
) -> set[tuple[int, int]]:
    available = set(available)
    assert 0 < count <= len(available)

    cells: set[tuple[int, int]] = set()
    seed = tuple(available)[int(rng.integers(0, len(available)))]
    cells.add(seed)
    available.remove(seed)

    while len(cells) < count:
        frontier = {
            nbr
            for cell in cells
            for nbr in _neighbors4(cell)
            if nbr in available
        }
        start_new_component = len(cells) >= 2 and rng.random() < 0.18
        if frontier and not start_new_component:
            choice = tuple(frontier)[int(rng.integers(0, len(frontier)))]
        else:
            choice = tuple(available)[int(rng.integers(0, len(available)))]
        cells.add(choice)
        available.remove(choice)
    return cells


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _choose_visible_counts(
    rng: np.random.Generator,
    counts_by_rank: list[int],
) -> list[int] | None:
    visible = [counts_by_rank[0]]
    for count in counts_by_rank[1:]:
        hi = min(4, count - 1)
        if hi < 1:
            return None
        visible.append(int(rng.integers(1, hi + 1)))
    if sum(visible) > 19:
        return None
    return visible


def _build_masks(
    rng: np.random.Generator,
    front_order: list[int],
    counts_by_panel: list[int],
) -> list[np.ndarray] | None:
    counts_by_rank = [counts_by_panel[idx] for idx in front_order]
    visible_by_rank = _choose_visible_counts(rng, counts_by_rank)
    if visible_by_rank is None:
        return None

    winners_by_rank: list[set[tuple[int, int]]] = []
    unused = set(ALL_CELLS)
    for visible_count in visible_by_rank:
        if visible_count > len(unused):
            return None
        chosen = _sample_cluster(rng, unused, visible_count)
        winners_by_rank.append(chosen)
        unused -= chosen

    masks_by_panel = [np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool) for _ in range(4)]
    for rank, panel_index in enumerate(front_order):
        masks_by_panel[panel_index] = _cells_to_mask(winners_by_rank[rank])

    for rank in range(1, 4):
        panel_index = front_order[rank]
        hidden_budget = counts_by_panel[panel_index] - visible_by_rank[rank]
        assert hidden_budget >= 1

        chosen_hidden: set[tuple[int, int]] = set()
        prev_winner = tuple(winners_by_rank[rank - 1])[int(rng.integers(0, len(winners_by_rank[rank - 1])))]
        chosen_hidden.add(prev_winner)

        if rank >= 2 and len(chosen_hidden) < hidden_budget and rng.random() < 0.75:
            front_winner = tuple(winners_by_rank[0])[int(rng.integers(0, len(winners_by_rank[0])))]
            chosen_hidden.add(front_winner)
        if rank == 3 and len(chosen_hidden) < hidden_budget and rng.random() < 0.55:
            second_winner = tuple(winners_by_rank[1])[int(rng.integers(0, len(winners_by_rank[1])))]
            chosen_hidden.add(second_winner)

        candidate_pool = set().union(*winners_by_rank[:rank]) - chosen_hidden
        while len(chosen_hidden) < hidden_budget:
            if not candidate_pool:
                return None
            choice = tuple(candidate_pool)[int(rng.integers(0, len(candidate_pool)))]
            chosen_hidden.add(choice)
            candidate_pool.remove(choice)

        mask = masks_by_panel[panel_index].copy()
        for r, c in chosen_hidden:
            mask[r, c] = True
        masks_by_panel[panel_index] = mask

    return masks_by_panel


def _render_input(panel_colors: list[int], masks_by_panel: list[np.ndarray]) -> np.ndarray:
    grid = np.zeros(GRID_SHAPE, dtype=int)
    grid[:, SEPARATOR_COLS] = SEPARATOR_COLOR
    for panel_index, start in enumerate(PANEL_STARTS):
        color = panel_colors[panel_index]
        mask = masks_by_panel[panel_index]
        grid[:, start : start + PANEL_SIZE][mask] = color
    return grid


def _check_chain_dependency(masks_by_panel: list[np.ndarray], front_order: list[int]) -> bool:
    for rank in range(1, 4):
        front_panel = front_order[rank - 1]
        back_panel = front_order[rank]
        if rank == 1:
            front_only_winners = masks_by_panel[front_panel]
        else:
            earlier_front = np.any(
                np.stack([masks_by_panel[idx] for idx in front_order[: rank - 1]], axis=0),
                axis=0,
            )
            front_only_winners = masks_by_panel[front_panel] & ~earlier_front
        if not np.any(front_only_winners & masks_by_panel[back_panel]):
            return False
    return True


def _generated_instance_ok(grid: np.ndarray) -> bool:
    try:
        panels, colors = _extract_panels(grid)
        counts = _panel_counts(panels)
        front_order = _front_order_from_counts(counts)
        out = main(grid)
    except Exception:
        return False

    if tuple(front_order) in COMMON_FIXED_ORDERS:
        return False
    if set(colors) != set(PANEL_COLORS):
        return False
    if any(not (3 <= count <= 10) for count in counts):
        return False
    if len(set(counts)) != 4:
        return False

    masks_by_panel = [panel != 0 for panel in panels]
    coverage = np.sum(np.stack(masks_by_panel, axis=0), axis=0)
    if int(coverage.max()) < 2:
        return False
    if np.count_nonzero(out) < 7 or np.count_nonzero(out) > 19:
        return False
    if set(map(int, np.unique(out))) != {0, 1, 2, 3, 4}:
        return False

    for panel_index, color in enumerate(colors):
        visible = (out == color) & masks_by_panel[panel_index]
        if not np.any(visible):
            return False

    if not _check_chain_dependency(masks_by_panel, front_order):
        return False

    correct = out
    differing_fixed_orders = 0
    for fixed_order in COMMON_FIXED_ORDERS:
        alt = _overlay_panels(panels, list(fixed_order))
        if not np.array_equal(alt, correct):
            differing_fixed_orders += 1
    if differing_fixed_orders != len(COMMON_FIXED_ORDERS):
        return False

    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    ranked_counts = _sample_counts(rng)
    if ranked_counts is None:
        return None

    front_order = rng.permutation(4).tolist()
    if tuple(front_order) in COMMON_FIXED_ORDERS:
        return None

    counts_by_panel = [0] * 4
    for rank, panel_index in enumerate(front_order):
        counts_by_panel[panel_index] = ranked_counts[rank]

    masks_by_panel = _build_masks(rng, front_order, counts_by_panel)
    if masks_by_panel is None:
        return None

    panel_colors = rng.permutation(np.array(PANEL_COLORS, dtype=int)).tolist()
    grid = _render_input(panel_colors, masks_by_panel)
    if _generated_instance_ok(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
