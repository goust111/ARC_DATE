from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


def _most_common_color(grid: np.ndarray, exclude: set[int] | None = None) -> int:
    values = [int(v) for v in np.asarray(grid).ravel()]
    if exclude:
        values = [v for v in values if v not in exclude]
    counts = Counter(values)
    assert counts, "no colors available"
    return int(counts.most_common(1)[0][0])


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    h, w = shape
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            comp.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc, mask.shape):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        parts.append(comp)
    return parts


def _compose_panels(panels: list[np.ndarray], order: list[int], bg: int) -> np.ndarray:
    out = np.full_like(panels[0], bg)
    for panel_index in order:
        panel = panels[panel_index]
        mask = panel != bg
        out[mask] = panel[mask]
    return out


def _unique_topological_order(colors: list[int], below_to_above: dict[int, set[int]]) -> list[int]:
    colors = [int(c) for c in colors]
    indegree = {color: 0 for color in colors}
    for src in colors:
        for dst in below_to_above[src]:
            indegree[dst] += 1

    order: list[int] = []
    remaining = set(colors)
    while remaining:
        frontier = sorted(color for color in remaining if indegree[color] == 0)
        assert len(frontier) == 1, "legend relations must determine a unique layer order"
        color = frontier[0]
        order.append(color)
        remaining.remove(color)
        for nxt in below_to_above[color]:
            indegree[nxt] -= 1
    return order


def _parse_input(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)

    solid_rows = [(r, int(g[r, 0])) for r in range(h) if np.all(g[r] == g[r, 0]) and int(g[r, 0]) != 0]
    assert len(solid_rows) >= 2, "expected separator rows"

    parsed = None
    for i in range(len(solid_rows)):
        for j in range(i + 1, len(solid_rows)):
            panel_sep_row, sep_color = solid_rows[i]
            legend_sep_row, sep_color_2 = solid_rows[j]
            if sep_color != sep_color_2:
                continue
            if panel_sep_row <= 0 or legend_sep_row <= panel_sep_row + 1 or legend_sep_row >= h - 1:
                continue
            panel_h = panel_sep_row
            lower_h = legend_sep_row - panel_sep_row - 1
            if panel_h != lower_h:
                continue
            top_height = legend_sep_row
            sep_cols = [c for c in range(w) if np.all(g[:top_height, c] == sep_color)]
            if len(sep_cols) != 1:
                continue
            panel_sep_col = sep_cols[0]
            if panel_sep_col <= 0 or w - panel_sep_col - 1 <= 0:
                continue
            panel_w = panel_sep_col
            right_w = w - panel_sep_col - 1
            if panel_w != right_w:
                continue
            parsed = (panel_h, panel_w, panel_sep_row, panel_sep_col, legend_sep_row, sep_color)
            break
        if parsed is not None:
            break
    assert parsed is not None, "could not parse panel layout"

    panel_h, panel_w, panel_sep_row, panel_sep_col, legend_sep_row, sep_color = parsed
    upper_left = g[:panel_h, :panel_w]
    upper_right = g[:panel_h, panel_sep_col + 1 :]
    lower_left = g[panel_sep_row + 1 : legend_sep_row, :panel_w]
    lower_right = g[panel_sep_row + 1 : legend_sep_row, panel_sep_col + 1 :]
    panels = [upper_left, upper_right, lower_left, lower_right]

    bg = _most_common_color(g[g != sep_color])
    panel_colors: list[int] = []
    for panel in panels:
        colors = sorted(set(map(int, np.unique(panel))) - {bg})
        assert len(colors) == 1, "each panel must contain exactly one foreground color"
        assert colors[0] != sep_color, "separator color cannot appear inside panels"
        panel_colors.append(colors[0])
    assert len(set(panel_colors)) == 4, "panel colors must be distinct"

    legend = g[legend_sep_row + 1 :, :]
    return panels, panel_colors, legend, bg, sep_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compose the four panel layers by decoding a unique cover order from the legend dominoes.

    Input structure:
    - The upper part of the grid is split by separator lines into a 2x2 set of equal-size panels.
    - Each panel has one unique foreground color on a shared background, and corresponding cells align across panels.
    - Below another separator line sits a legend band containing isolated two-cell dominoes whose colors come from the panel colors.

    Rule:
    - Read each legend domino as a cover relation: in a horizontal domino the left color covers the right color, and in a vertical domino the upper color covers the lower color.
    - Combine all such relations transitively to recover the unique full order from bottom layer to top layer.
    - Starting from the bottom panel, draw every panel's non-background cells onto one single-panel canvas; later layers overwrite earlier ones.

    Key insight:
    - The quadrant positions are a decoy; only the legend domino orientations determine the stacking depth.
    """
    panels, panel_colors, legend, bg, _sep_color = _parse_input(grid)

    legend_colors = set(panel_colors)
    legend_mask = np.isin(legend, list(legend_colors))
    dominoes = _components(legend_mask)
    assert dominoes, "expected at least one legend domino"

    below_to_above = {color: set() for color in panel_colors}
    for comp in dominoes:
        assert len(comp) == 2, "every legend object must be a two-cell domino"
        (r0, c0), (r1, c1) = sorted(comp)
        color0 = int(legend[r0, c0])
        color1 = int(legend[r1, c1])
        assert color0 in legend_colors and color1 in legend_colors, "legend uses only panel colors"
        assert color0 != color1, "domino halves must use two different panel colors"
        manhattan = abs(r0 - r1) + abs(c0 - c1)
        assert manhattan == 1, "domino cells must be orthogonally adjacent"
        if r0 == r1:
            left = (r0, c0) if c0 < c1 else (r1, c1)
            right = (r1, c1) if c0 < c1 else (r0, c0)
            coverer = int(legend[left])
            covered = int(legend[right])
        else:
            upper = (r0, c0) if r0 < r1 else (r1, c1)
            lower = (r1, c1) if r0 < r1 else (r0, c0)
            coverer = int(legend[upper])
            covered = int(legend[lower])
        below_to_above[covered].add(coverer)

    bottom_to_top_colors = _unique_topological_order(panel_colors, below_to_above)
    color_to_panel = {color: i for i, color in enumerate(panel_colors)}
    bottom_to_top_panels = [color_to_panel[color] for color in bottom_to_top_colors]
    return _compose_panels(panels, bottom_to_top_panels, bg)


PANEL_H_RANGE = (4, 6)
PANEL_W_RANGE = (4, 6)
LEGEND_H_RANGE = (4, 6)
VISIBLE_MIN_PER_PANEL = 3
BG_MIN = 1
GEN_RETRIES = 400
EMIT_DEDUP_RETRIES = 200

_SEEN_INPUT_SIGS: set[tuple[tuple[int, int], bytes]] = set()


def _random_partition(total: int, parts: int, min_each: int) -> list[int] | None:
    if total < parts * min_each:
        return None
    cuts = [0, total - parts * min_each]
    cuts.extend(random.randint(0, total - parts * min_each) for _ in range(parts - 1))
    cuts.sort()
    values = []
    for i in range(parts):
        values.append(min_each + cuts[i + 1] - cuts[i])
    random.shuffle(values)
    return values


def _build_winner_map(panel_h: int, panel_w: int, order: list[int]) -> np.ndarray | None:
    area = panel_h * panel_w
    max_bg = min(max(BG_MIN + 1, area // 4), area - 4 * VISIBLE_MIN_PER_PANEL)
    if max_bg < BG_MIN:
        return None
    bg_count = random.randint(BG_MIN, max_bg)
    visible_total = area - bg_count
    counts = _random_partition(visible_total, 4, VISIBLE_MIN_PER_PANEL)
    if counts is None:
        return None

    winners = np.full((panel_h, panel_w), -1, dtype=int)
    cells = [(r, c) for r in range(panel_h) for c in range(panel_w)]
    random.shuffle(cells)
    cursor = 0
    for panel_index, count in zip(order, counts):
        for r, c in cells[cursor : cursor + count]:
            winners[r, c] = panel_index
        cursor += count
    return winners


def _visible_counts(winners: np.ndarray) -> dict[int, int]:
    out: dict[int, int] = {}
    for panel_index in range(4):
        out[panel_index] = int(np.count_nonzero(winners == panel_index))
    return out


def _build_panel_masks(winners: np.ndarray, order: list[int]) -> list[np.ndarray] | None:
    order_pos = {panel: i for i, panel in enumerate(order)}
    masks = [(winners == panel) for panel in range(4)]
    visible_sets = {panel: [tuple(map(int, rc)) for rc in np.argwhere(winners == panel)] for panel in range(4)}

    for lower_pos in range(3):
        lower_panel = order[lower_pos]
        upper_panel = order[lower_pos + 1]
        choices = visible_sets[upper_panel]
        if not choices:
            return None
        r, c = random.choice(choices)
        masks[lower_panel][r, c] = True

    area = winners.size
    for panel in range(4):
        pos = order_pos[panel]
        hidden_allowed = np.zeros_like(winners, dtype=bool)
        for upper_pos in range(pos + 1, 4):
            hidden_allowed |= winners == order[upper_pos]
        hidden_cells = [tuple(map(int, rc)) for rc in np.argwhere(hidden_allowed & (~masks[panel]))]
        random.shuffle(hidden_cells)
        extra_cap = min(len(hidden_cells), max(1, area // 4))
        extra = random.randint(0, extra_cap)
        for r, c in hidden_cells[:extra]:
            masks[panel][r, c] = True

    if any(int(mask.sum()) < VISIBLE_MIN_PER_PANEL for mask in masks):
        return None
    return masks


def _sample_relations(order_colors: list[int]) -> list[tuple[int, int]]:
    base = [
        (order_colors[1], order_colors[0]),
        (order_colors[2], order_colors[1]),
        (order_colors[3], order_colors[2]),
    ]
    extra_candidates = [
        (order_colors[2], order_colors[0]),
        (order_colors[3], order_colors[1]),
        (order_colors[3], order_colors[0]),
    ]
    relations = base.copy()
    if random.random() < 0.6:
        relations.append(random.choice(extra_candidates))
    random.shuffle(relations)
    return relations


def _place_dominoes(legend_h: int, width: int, relations: list[tuple[int, int]]) -> np.ndarray | None:
    legend = np.zeros((legend_h, width), dtype=int)
    blocked = np.zeros((legend_h, width), dtype=bool)

    orientations = ["H", "V"] + [random.choice(["H", "V"]) for _ in range(len(relations) - 2)]
    random.shuffle(orientations)

    for (coverer, covered), orientation in zip(relations, orientations):
        candidates = []
        for r in range(legend_h):
            for c in range(width):
                if orientation == "H":
                    cells = [(r, c), (r, c + 1)]
                    if c + 1 >= width:
                        continue
                else:
                    cells = [(r, c), (r + 1, c)]
                    if r + 1 >= legend_h:
                        continue

                r0 = max(0, min(rr for rr, _ in cells) - 1)
                r1 = min(legend_h - 1, max(rr for rr, _ in cells) + 1)
                c0 = max(0, min(cc for _, cc in cells) - 1)
                c1 = min(width - 1, max(cc for _, cc in cells) + 1)
                if np.any(blocked[r0 : r1 + 1, c0 : c1 + 1]):
                    continue
                candidates.append(cells)
        if not candidates:
            return None

        cells = random.choice(candidates)
        if orientation == "H":
            (r0, c0), (r1, c1) = cells
            legend[r0, c0] = coverer
            legend[r1, c1] = covered
        else:
            (r0, c0), (r1, c1) = cells
            legend[r0, c0] = coverer
            legend[r1, c1] = covered

        rr0 = max(0, min(rr for rr, _ in cells) - 1)
        rr1 = min(legend_h - 1, max(rr for rr, _ in cells) + 1)
        cc0 = max(0, min(cc for _, cc in cells) - 1)
        cc1 = min(width - 1, max(cc for _, cc in cells) + 1)
        blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return legend


def _render_input(
    panel_h: int,
    panel_w: int,
    legend_h: int,
    sep_color: int,
    bg: int,
    panel_colors: list[int],
    masks: list[np.ndarray],
    legend: np.ndarray,
) -> np.ndarray:
    height = 2 * panel_h + 2 + legend_h
    width = 2 * panel_w + 1
    grid = np.full((height, width), bg, dtype=int)

    grid[panel_h, :] = sep_color
    grid[2 * panel_h + 1, :] = sep_color
    grid[: 2 * panel_h + 1, panel_w] = sep_color

    origins = [
        (0, 0),
        (0, panel_w + 1),
        (panel_h + 1, 0),
        (panel_h + 1, panel_w + 1),
    ]
    for panel_index, (r0, c0) in enumerate(origins):
        patch = np.where(masks[panel_index], panel_colors[panel_index], bg)
        grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = patch

    grid[2 * panel_h + 2 :, :] = legend
    return grid


def _all_orders_change_output(panels: list[np.ndarray], true_order: list[int], bg: int) -> bool:
    expected = _compose_panels(panels, true_order, bg)
    for order in itertools.permutations(range(4)):
        order = list(order)
        if order == true_order:
            continue
        if np.array_equal(_compose_panels(panels, order, bg), expected):
            return False
    return True


def _adjacent_relations_are_necessary(masks: list[np.ndarray], order: list[int]) -> bool:
    for pos in range(3):
        lower = order[pos]
        upper = order[pos + 1]
        higher_union = np.zeros_like(masks[0], dtype=bool)
        for higher_pos in range(pos + 2, 4):
            higher_union |= masks[order[higher_pos]]
        decisive = masks[lower] & masks[upper] & (~higher_union)
        if not np.any(decisive):
            return False
    return True


def _legend_order_matches(relations: list[tuple[int, int]], order_colors: list[int]) -> bool:
    graph = {color: set() for color in order_colors}
    for coverer, covered in relations:
        graph[covered].add(coverer)
    try:
        recovered = _unique_topological_order(order_colors, graph)
    except AssertionError:
        return False
    return recovered == order_colors


def _attempt_generate() -> np.ndarray | None:
    panel_h = random.randint(*PANEL_H_RANGE)
    panel_w = random.randint(*PANEL_W_RANGE)
    legend_h = random.randint(*LEGEND_H_RANGE)
    bg = 0

    all_colors = list(range(1, 10))
    panel_colors = random.sample(all_colors, 4)
    sep_color = random.choice([c for c in all_colors if c not in panel_colors])
    order = random.sample(range(4), 4)

    winners = _build_winner_map(panel_h, panel_w, order)
    if winners is None:
        return None
    if min(_visible_counts(winners).values()) < VISIBLE_MIN_PER_PANEL:
        return None

    masks = _build_panel_masks(winners, order)
    if masks is None:
        return None
    if not _adjacent_relations_are_necessary(masks, order):
        return None

    order_colors = [panel_colors[p] for p in order]
    relations = _sample_relations(order_colors)
    if not _legend_order_matches(relations, order_colors):
        return None

    legend = _place_dominoes(legend_h, 2 * panel_w + 1, relations)
    if legend is None:
        return None

    grid = _render_input(panel_h, panel_w, legend_h, sep_color, bg, panel_colors, masks, legend)
    panels, parsed_colors, _legend, parsed_bg, _sep = _parse_input(grid)
    assert parsed_colors == panel_colors
    if parsed_bg != bg:
        return None

    expected = _compose_panels(panels, order, bg)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if len(set(map(int, actual.ravel())) - {bg}) != 4:
        return None
    if not _all_orders_change_output(panels, order, bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(EMIT_DEDUP_RETRIES):
        for _ in range(GEN_RETRIES):
            grid = _attempt_generate()
            if grid is None:
                continue
            sig = (tuple(map(int, grid.shape)), grid.tobytes())
            if sig in _SEEN_INPUT_SIGS:
                continue
            _SEEN_INPUT_SIGS.add(sig)
            return grid
    raise RuntimeError("failed to generate a valid input")
