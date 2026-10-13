from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


def _axis_segments(separator_mask: np.ndarray) -> list[slice]:
    n = int(separator_mask.size)
    groups: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if separator_mask[i]:
            j = i + 1
            while j < n and separator_mask[j]:
                j += 1
            groups.append((i, j))
            i = j
        else:
            i += 1

    segments: list[slice] = []
    start = 0
    for group_start, group_stop in groups:
        if start < group_start:
            segments.append(slice(start, group_start))
        start = group_stop
    if start < n:
        segments.append(slice(start, n))
    return segments


def _most_common_color(grid: np.ndarray) -> int:
    return Counter(map(int, grid.reshape(-1))).most_common(1)[0][0]


def _parse_layout(grid: np.ndarray) -> tuple[int, list[int], dict[int, np.ndarray]]:
    h, w = grid.shape
    candidates: list[tuple[int, list[int], dict[int, np.ndarray]]] = []

    for separator in map(int, np.unique(grid)):
        full_rows = np.all(grid == separator, axis=1)
        full_cols = np.all(grid == separator, axis=0)
        if not full_rows.any() and not full_cols.any():
            continue

        row_slices = _axis_segments(full_rows)
        col_slices = _axis_segments(full_cols)
        if not row_slices or not col_slices:
            continue
        if len(row_slices) * len(col_slices) < 2:
            continue

        row_heights = {sl.stop - sl.start for sl in row_slices}
        col_widths = {sl.stop - sl.start for sl in col_slices}
        if 0 in row_heights or 0 in col_widths:
            continue
        if len(row_heights) != 1 or len(col_widths) != 1:
            continue

        ref = grid[row_slices[-1], col_slices[-1]]
        background = _most_common_color(ref)
        if background == separator:
            continue

        sequence = [int(v) for v in ref.reshape(-1) if int(v) != background]
        if len(sequence) < 2 or len(sequence) != len(set(sequence)):
            continue

        panels_by_color: dict[int, np.ndarray] = {}
        ok = True
        for r_idx, row_slice in enumerate(row_slices):
            for c_idx, col_slice in enumerate(col_slices):
                if r_idx == len(row_slices) - 1 and c_idx == len(col_slices) - 1:
                    continue
                panel = grid[row_slice, col_slice]
                colors = {int(v) for v in np.unique(panel) if int(v) != background}
                if len(colors) != 1:
                    ok = False
                    break
                color = next(iter(colors))
                if color == separator or color in panels_by_color:
                    ok = False
                    break
                panels_by_color[color] = panel.copy()
            if not ok:
                break
        if not ok:
            continue

        if set(sequence) != set(panels_by_color) or len(sequence) != len(panels_by_color):
            continue

        candidates.append((background, sequence, panels_by_color))

    assert len(candidates) == 1, f"expected one valid layout, found {len(candidates)}"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge partitioned layer panels using the color order encoded in the bottom-right reference panel.

    Input structure:
    - Full-row and full-column separator lines split the grid into equal-sized rectangular panels.
    - Every panel except the bottom-right one is a single-color layer drawn on a black background.
    - The bottom-right panel is a reference panel whose non-black pixels list the layer colors in
      row-major reading order from bottom layer to top layer.

    Rule:
    - Read the reference panel's non-background colors from left to right, top to bottom.
    - Match each color to the layer panel that uses that color.
    - Overlay the layer panels in that order; each non-background pixel hides anything below it.

    Output:
    - A single panel with the same shape as each layer panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    background, sequence, panels_by_color = _parse_layout(g)
    panel_shape = next(iter(panels_by_color.values())).shape
    out = np.full(panel_shape, background, dtype=int)

    for color in sequence:
        panel = panels_by_color[color]
        out[panel != background] = color
    return out


BG = 0
SEP = 5
LAYER_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
LAYOUT_CHOICES = ((2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (3, 3))
PANEL_SIDE_MIN = 4
PANEL_SIDE_MAX = 6
GENERATION_RETRIES = 300
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in NEIGHBORS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _random_shortest_path(
    rng: random.Random,
    h: int,
    w: int,
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    if start == goal:
        return [start]

    blocked = set(blocked)
    blocked.discard(start)
    blocked.discard(goal)
    queue = deque([start])
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while queue:
        cur = queue.popleft()
        neighbors = _neighbors4(cur[0], cur[1], h, w)
        rng.shuffle(neighbors)
        for nxt in neighbors:
            if nxt in blocked or nxt in parents:
                continue
            parents[nxt] = cur
            if nxt == goal:
                path = [goal]
                node = cur
                while node is not None:
                    path.append(node)
                    node = parents[node]
                path.reverse()
                return path
            queue.append(nxt)
    return None


def _line_feature(
    rng: random.Random,
    cells: set[tuple[int, int]],
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    base = rng.choice(tuple(cells))
    directions = list(NEIGHBORS4)
    rng.shuffle(directions)
    for dr, dc in directions:
        length = rng.randint(1, 3)
        rr, cc = base
        feature: list[tuple[int, int]] = []
        for _ in range(length):
            rr += dr
            cc += dc
            if not (0 <= rr < h and 0 <= cc < w):
                break
            if (rr, cc) in forbidden:
                break
            if (rr, cc) not in cells:
                feature.append((rr, cc))
        if feature:
            return feature
    return []


def _rect_feature(
    rng: random.Random,
    cells: set[tuple[int, int]],
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    base_r, base_c = rng.choice(tuple(cells))
    rect_h = rng.randint(1, 2)
    rect_w = rng.randint(1, 3)
    top = base_r - rng.randint(0, rect_h - 1)
    left = base_c - rng.randint(0, rect_w - 1)
    feature: list[tuple[int, int]] = []
    for r in range(top, top + rect_h):
        for c in range(left, left + rect_w):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if (r, c) in forbidden or (r, c) in cells:
                continue
            feature.append((r, c))
    return feature


def _bend_feature(
    rng: random.Random,
    cells: set[tuple[int, int]],
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    base_r, base_c = rng.choice(tuple(cells))
    dr1, dc1 = rng.choice(NEIGHBORS4)
    if dr1 == 0:
        turn_options = ((1, 0), (-1, 0))
    else:
        turn_options = ((0, 1), (0, -1))
    dr2, dc2 = rng.choice(turn_options)

    feature: list[tuple[int, int]] = []
    rr, cc = base_r, base_c
    for dr, dc in ((dr1, dc1), (dr2, dc2)):
        steps = rng.randint(1, 2)
        for _ in range(steps):
            rr += dr
            cc += dc
            if not (0 <= rr < h and 0 <= cc < w):
                return feature
            if (rr, cc) in forbidden:
                return feature
            if (rr, cc) not in cells and (rr, cc) not in feature:
                feature.append((rr, cc))
    return feature


def _blob_feature(
    rng: random.Random,
    cells: set[tuple[int, int]],
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    frontier: list[tuple[int, int]] = []
    for r, c in cells:
        for nbr in _neighbors4(r, c, h, w):
            if nbr in cells or nbr in forbidden or nbr in frontier:
                continue
            frontier.append(nbr)
    if not frontier:
        return []
    rng.shuffle(frontier)
    return [frontier[0]]


def _build_layer_mask(
    rng: random.Random,
    h: int,
    w: int,
    required: list[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> np.ndarray | None:
    required = list(dict.fromkeys(required))
    if any(pos in forbidden for pos in required):
        return None

    cells = set(required)
    if len(required) == 2:
        path = _random_shortest_path(rng, h, w, required[0], required[1], forbidden)
        if path is None:
            return None
        cells.update(path)

    free_cells = h * w - len(forbidden)
    min_target = max(len(cells), 3)
    max_target = min(free_cells, max(min_target, int(round(h * w * rng.uniform(0.28, 0.58)))))
    target = rng.randint(min_target, max_target)

    feature_builders = (_blob_feature, _line_feature, _rect_feature, _bend_feature)
    style = rng.randrange(len(feature_builders))
    stalls = 0
    while len(cells) < target and stalls < 60:
        builder = feature_builders[(style + stalls) % len(feature_builders)]
        feature = builder(rng, cells, h, w, forbidden)
        new_cells = [pos for pos in feature if pos not in cells and pos not in forbidden]
        if not new_cells:
            stalls += 1
            continue
        for pos in new_cells:
            if len(cells) >= target:
                break
            cells.add(pos)
        stalls = 0

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _compose_masks(
    order: list[int],
    masks_by_color: dict[int, np.ndarray],
    background: int,
) -> tuple[np.ndarray, np.ndarray]:
    shape = next(iter(masks_by_color.values())).shape
    out = np.full(shape, background, dtype=int)
    cover_count = np.zeros(shape, dtype=int)
    for color in order:
        mask = masks_by_color[color]
        cover_count += mask.astype(int)
        out[mask] = color
    return out, cover_count


def _make_reference_panel(
    rng: random.Random,
    h: int,
    w: int,
    order: list[int],
    background: int,
) -> np.ndarray:
    panel = np.full((h, w), background, dtype=int)
    slots = rng.sample(range(h * w), len(order))
    slots.sort()
    for slot, color in zip(slots, order):
        panel[slot // w, slot % w] = color
    return panel


def _assemble_grid(
    rows: int,
    cols: int,
    panel_h: int,
    panel_w: int,
    panels: list[np.ndarray],
    reference: np.ndarray,
) -> np.ndarray:
    total_h = rows * panel_h + (rows - 1)
    total_w = cols * panel_w + (cols - 1)
    grid = np.full((total_h, total_w), SEP, dtype=int)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            r0 = r * (panel_h + 1)
            c0 = c * (panel_w + 1)
            if r == rows - 1 and c == cols - 1:
                panel = reference
            else:
                panel = panels[idx]
                idx += 1
            grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = panel
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows, cols = rng.choice(LAYOUT_CHOICES)
    layer_count = rows * cols - 1
    panel_h = rng.randint(PANEL_SIDE_MIN, PANEL_SIDE_MAX)
    panel_w = rng.randint(PANEL_SIDE_MIN, PANEL_SIDE_MAX)

    colors = rng.sample(LAYER_COLOR_POOL, layer_count)
    order = colors[:]
    rng.shuffle(order)

    positions = [(r, c) for r in range(panel_h) for c in range(panel_w)]
    reserved_and_anchor = rng.sample(positions, layer_count + 1)
    reserved = {color: reserved_and_anchor[i] for i, color in enumerate(order)}
    anchor = reserved_and_anchor[-1]
    overlap_layers = set(rng.sample(order, rng.randint(2, min(4, layer_count))))

    masks_by_color: dict[int, np.ndarray] = {}
    protected: list[tuple[int, int]] = []
    for color in order:
        required = [reserved[color]]
        if color in overlap_layers:
            required.append(anchor)
        mask = _build_layer_mask(rng, panel_h, panel_w, required, set(protected))
        if mask is None or not mask[reserved[color]]:
            return None
        masks_by_color[color] = mask
        protected.append(reserved[color])

    composed, cover_count = _compose_masks(order, masks_by_color, BG)
    if int(cover_count.max()) < 2:
        return None
    if set(map(int, composed[composed != BG])) != set(colors):
        return None
    if np.count_nonzero(composed == BG) < 2:
        return None

    reference = _make_reference_panel(rng, panel_h, panel_w, order, BG)

    placement_order = colors[:]
    rng.shuffle(placement_order)
    layer_panels: list[np.ndarray] = []
    for color in placement_order:
        panel = np.full((panel_h, panel_w), BG, dtype=int)
        panel[masks_by_color[color]] = color
        layer_panels.append(panel)

    grid = _assemble_grid(rows, cols, panel_h, panel_w, layer_panels, reference)
    if not np.array_equal(main(grid), composed):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
