from __future__ import annotations

import random

import numpy as np


BG = 0
SEPARATOR = 6
BITS = (1, 2, 4)
PAIR_LABELS = {3: (0, 1), 5: (0, 2), 6: (1, 2)}
COLOR_POOL = [1, 2, 3, 4, 5, 7, 8, 9]
PANEL_H_RANGE = (7, 9)
PANEL_W_RANGE = (6, 9)
TARGET_SIZE_RANGE = range(8, 16)
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    h, w = shape
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(mask.shape, r, c):
            if mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _has_solid_non_bg_row(panel: np.ndarray, bg: int) -> bool:
    for r in range(panel.shape[0]):
        if np.all(panel[r, :] == int(panel[r, 0])) and int(panel[r, 0]) != bg:
            return True
    return False


def _split_panels(grid: np.ndarray) -> tuple[list[np.ndarray], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, _w = g.shape
    bg = _most_common_color(g)

    solid_rows = [
        (r, int(g[r, 0]))
        for r in range(h)
        if np.all(g[r, :] == int(g[r, 0])) and int(g[r, 0]) != bg
    ]
    matches = []
    for i in range(len(solid_rows)):
        for j in range(i + 1, len(solid_rows)):
            r0, color0 = solid_rows[i]
            r1, color1 = solid_rows[j]
            if color0 != color1:
                continue
            panel_h = r0
            if panel_h <= 0:
                continue
            if r1 - r0 - 1 != panel_h:
                continue
            if h - r1 - 1 != panel_h:
                continue
            matches.append((r0, r1, color0, panel_h))

    assert len(matches) == 1, "expected exactly one evenly spaced separator pair"
    sep0, sep1, sep_color, panel_h = matches[0]
    panels = [
        g[:sep0, :],
        g[sep0 + 1 : sep1, :],
        g[sep1 + 1 :, :],
    ]
    assert len({panel.shape for panel in panels}) == 1
    assert panels[0].shape[0] == panel_h
    return panels, bg, sep_color


def _overlay_panels(
    panels: list[np.ndarray], panel_colors: list[int], color_counts: dict[int, int], bg: int
) -> np.ndarray:
    h, w = panels[0].shape
    out = np.full((h, w), bg, dtype=int)
    masks = [panel != bg for panel in panels]
    for r in range(h):
        for c in range(w):
            active = [idx for idx in range(3) if masks[idx][r, c]]
            if len(active) == 1:
                out[r, c] = panel_colors[active[0]]
            elif len(active) == 2:
                i, j = active
                out[r, c] = panel_colors[i] if color_counts[panel_colors[i]] < color_counts[panel_colors[j]] else panel_colors[j]
            elif len(active) == 3:
                out[r, c] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay three separator-split masks and resolve two-layer conflicts by global rarity.

    Input structure:
    - The grid is a vertical stack of three equal-size black-background panels.
    - Two single-row full-width magenta separators split the stack into the three panels.
    - Each panel contains exactly one non-black foreground color forming one irregular mask.

    Rule:
    - Remove the separator rows and align the three panels cell by cell.
    - Count the total number of pixels of each panel color in the whole input grid.
    - Keep single-layer pixels unchanged, choose the rarer color on every two-layer overlap,
      and turn three-layer overlaps back to black.

    Key insight:
    - The spatial alignment tells you where conflicts happen, but the winning color on each
      two-layer conflict is encoded globally by which panel color is less frequent overall.
    """
    panels, bg, sep_color = _split_panels(grid)
    panel_colors = []
    for panel in panels:
        colors = sorted(set(map(int, np.unique(panel))) - {bg})
        assert len(colors) == 1, "each panel must contain exactly one foreground color"
        panel_colors.append(int(colors[0]))
    assert len(set(panel_colors)) == 3, "panel colors must be distinct"
    assert sep_color not in panel_colors, "separator color must be distinct from panel colors"

    color_counts = {color: int(np.count_nonzero(np.asarray(grid) == color)) for color in panel_colors}
    assert len(set(color_counts.values())) == 3, "foreground frequencies must be distinct"
    return _overlay_panels(panels, panel_colors, color_counts, bg)


def _pairwise_id_directions(colors: list[int], counts: list[int]) -> list[bool]:
    dirs = []
    for label, (i, j) in PAIR_LABELS.items():
        _ = label
        rarer = i if counts[i] < counts[j] else j
        other = j if rarer == i else i
        dirs.append(colors[rarer] > colors[other])
    return dirs


def _sample_colors_and_counts() -> tuple[list[int], list[int]]:
    for _ in range(200):
        colors = random.sample(COLOR_POOL, 3)
        counts = random.sample(list(TARGET_SIZE_RANGE), 3)
        directions = _pairwise_id_directions(colors, counts)
        if any(directions) and not all(directions):
            return colors, counts
    raise RuntimeError("failed to sample anti-shortcut color/count assignment")


def _build_core(shape: tuple[int, int]) -> np.ndarray | None:
    h, w = shape
    labels = np.zeros((h, w), dtype=np.uint8)
    center_r = random.randint(2, h - 3)
    center_c = random.randint(2, w - 3)
    labels[center_r, center_c] = 7

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    omitted = random.randrange(4)
    branch_dirs = [directions[i] for i in range(4) if i != omitted]
    random.shuffle(branch_dirs)

    pair_items = list(PAIR_LABELS)
    random.shuffle(pair_items)
    lengths = [random.randint(1, 2) for _ in range(3)]
    if sum(lengths) < 4:
        lengths[random.randrange(3)] = 2

    for label, (dr, dc), length in zip(pair_items, branch_dirs, lengths):
        for step in range(1, length + 1):
            rr = center_r + dr * step
            cc = center_c + dc * step
            if not (0 <= rr < h and 0 <= cc < w):
                return None
            labels[rr, cc] = label

    if random.random() < 0.35:
        dr, dc = directions[omitted]
        rr = center_r + dr
        cc = center_c + dc
        if 0 <= rr < h and 0 <= cc < w and labels[rr, cc] == 0:
            labels[rr, cc] = 7

    return labels


def _grow_labels(labels: np.ndarray, target_counts: list[int]) -> np.ndarray | None:
    corners = random.sample(
        [(0, 0), (0, labels.shape[1] - 1), (labels.shape[0] - 1, 0), (labels.shape[0] - 1, labels.shape[1] - 1)],
        3,
    )
    counts = [int(np.count_nonzero(labels & bit)) for bit in BITS]

    while any(counts[i] < target_counts[i] for i in range(3)):
        progress = False
        order = [i for i in range(3) if counts[i] < target_counts[i]]
        random.shuffle(order)
        order.sort(key=lambda idx: target_counts[idx] - counts[idx], reverse=True)
        for idx in order:
            bit = BITS[idx]
            frontier = set()
            for r, c in np.argwhere((labels & bit) != 0):
                for rr, cc in _neighbors4(labels.shape, int(r), int(c)):
                    if labels[rr, cc] == 0:
                        frontier.add((rr, cc))
            if not frontier:
                continue
            goal_r, goal_c = corners[idx]
            ranked = sorted(
                frontier,
                key=lambda pos: (abs(pos[0] - goal_r) + abs(pos[1] - goal_c), random.random()),
            )
            choice_pool = ranked[: min(3, len(ranked))]
            rr, cc = random.choice(choice_pool)
            labels[rr, cc] = bit
            counts[idx] += 1
            progress = True
        if not progress:
            return None
    return labels


def _labels_to_panels(labels: np.ndarray, colors: list[int]) -> list[np.ndarray]:
    panels = []
    for bit, color in zip(BITS, colors):
        panel = np.full(labels.shape, BG, dtype=int)
        panel[(labels & bit) != 0] = color
        panels.append(panel)
    return panels


def _transform_labels(labels: np.ndarray) -> np.ndarray:
    out = labels
    out = np.rot90(out, k=random.randint(0, 3))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return np.array(out, dtype=np.uint8, copy=True)


def _overlay_from_labels(labels: np.ndarray, colors: list[int], counts: list[int]) -> np.ndarray:
    out = np.full(labels.shape, BG, dtype=int)
    count_map = {color: count for color, count in zip(colors, counts)}
    panels = _labels_to_panels(labels, colors)
    return _overlay_panels(panels, colors, count_map, BG)


def _validate_generated_instance(grid: np.ndarray, labels: np.ndarray, colors: list[int], counts: list[int]) -> None:
    assert grid.ndim == 2
    assert max(grid.shape) <= 30
    assert len(set(colors)) == 3
    assert SEPARATOR not in colors
    assert len(set(counts)) == 3

    for idx, bit in enumerate(BITS):
        mask = (labels & bit) != 0
        assert int(mask.sum()) == counts[idx]
        assert _is_connected(mask)

    assert np.count_nonzero(labels == 7) >= 1
    for label in PAIR_LABELS:
        assert np.count_nonzero(labels == label) >= 1
    assert np.count_nonzero(labels == 0) >= 1

    directions = _pairwise_id_directions(colors, counts)
    assert any(directions) and not all(directions), "must refute color-id ordering shortcuts"

    expected = _overlay_from_labels(labels, colors, counts)
    assert np.count_nonzero(expected == BG) > np.count_nonzero(labels == 7), "need true background as well as triple-overlap black"
    for color in colors:
        assert np.count_nonzero(expected == color) >= 1

    rarity_order = sorted(range(3), key=lambda idx: counts[idx])
    middle_idx = rarity_order[1]
    common_idx = rarity_order[2]
    middle_pair = next(label for label, pair in PAIR_LABELS.items() if set(pair) == {middle_idx, common_idx})
    assert np.any(labels == middle_pair)
    assert np.all(expected[labels == middle_pair] == colors[middle_idx])

    parsed_panels, bg, sep_color = _split_panels(grid)
    assert bg == BG
    assert sep_color == SEPARATOR
    for panel, color in zip(parsed_panels, colors):
        assert not _has_solid_non_bg_row(panel, bg)
        unique_colors = sorted(set(map(int, np.unique(panel))) - {bg})
        assert unique_colors == [color]

    solved = main(grid)
    assert np.array_equal(solved, expected)


def _generate_case() -> np.ndarray | None:
    panel_h = random.randint(*PANEL_H_RANGE)
    panel_w = random.randint(*PANEL_W_RANGE)
    colors, counts = _sample_colors_and_counts()

    labels = _build_core((panel_h, panel_w))
    if labels is None:
        return None
    labels = _grow_labels(labels, counts)
    if labels is None:
        return None
    labels = _transform_labels(labels)

    panels = _labels_to_panels(labels, colors)
    if any(_has_solid_non_bg_row(panel, BG) for panel in panels):
        return None

    grid = np.full((labels.shape[0] * 3 + 2, labels.shape[1]), BG, dtype=int)
    grid[: labels.shape[0], :] = panels[0]
    grid[labels.shape[0], :] = SEPARATOR
    grid[labels.shape[0] + 1 : 2 * labels.shape[0] + 1, :] = panels[1]
    grid[2 * labels.shape[0] + 1, :] = SEPARATOR
    grid[2 * labels.shape[0] + 2 :, :] = panels[2]

    _validate_generated_instance(grid, labels, colors, counts)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid three-layer overlay puzzle")
