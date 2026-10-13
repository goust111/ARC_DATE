from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

PANEL_COLORS = [RED, BLUE, GREEN]
GRID_SIZES = [8, 9]
MARKER_COUNT_CHOICES = [1, 2, 3, 4]
MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_left(coords: np.ndarray) -> int:
    return int(coords[:, 1].min())


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    assert w == 3 * h + 2, "expected three HxH panels separated by two columns"
    sep_a = int(grid[0, h])
    sep_b = int(grid[0, 2 * h + 1])
    assert sep_a == sep_b == GREY, "separator bars must be solid grey columns"
    assert bool(np.all(grid[:, h] == GREY))
    assert bool(np.all(grid[:, 2 * h + 1] == GREY))
    return [
        grid[:, :h],
        grid[:, h + 1 : 2 * h + 1],
        grid[:, 2 * h + 2 :],
    ]


def _ordered_objects(panel: np.ndarray) -> list[dict[str, np.ndarray | int]]:
    mask = (panel != BG) & (panel != YELLOW)
    comps = _components4(mask)
    assert len(comps) == 3, "each panel must contain exactly three foreground objects"
    ordered = []
    for coords in comps:
        colors = {int(panel[r, c]) for r, c in coords}
        assert len(colors) == 1, "each object must be monochrome"
        ordered.append({"coords": coords, "color": next(iter(colors))})
    ordered.sort(key=lambda obj: (_bbox_left(obj["coords"]), int(obj["coords"][:, 0].min())))
    lefts = [_bbox_left(obj["coords"]) for obj in ordered]
    assert len(set(lefts)) == 3, "object left edges must be distinct"
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge one selected object from each panel, where yellow-marker counts rank the panels and assign left/middle/right selection roles.

    Input structure:
    - The grid is three square HxH panels on a black background, separated by two solid grey columns.
    - Each panel contains exactly three monochrome 4-connected foreground objects ordered left-to-right, plus several yellow marker pixels on the background.
    - Yellow counts differ across the three panels, so the panels have a unique highest, middle, and lowest marker count.

    Rule:
    - Count yellow pixels in each panel and rank the panels from most markers to fewest.
    - The highest-count panel contributes its leftmost object, the middle-count panel contributes its middle object, and the lowest-count panel contributes its rightmost object.
    - Copy those three objects onto a fresh HxH black canvas at their original panel-relative coordinates, drawing highest-count first and lowest-count last.

    Key insight:
    - The yellow pixels are not output; they encode a permutation of the three panels, and that permutation decides which left/middle/right object each panel contributes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    panels = _split_panels(g)
    marker_counts = [int(np.count_nonzero(panel == YELLOW)) for panel in panels]
    assert len(set(marker_counts)) == 3, "marker counts must be pairwise distinct"

    ranked_panels = sorted(range(3), key=lambda i: marker_counts[i], reverse=True)
    out = np.full((g.shape[0], g.shape[0]), BG, dtype=int)
    for rank, panel_index in enumerate(ranked_panels):
        obj = _ordered_objects(panels[panel_index])[rank]
        color = int(obj["color"])
        for r, c in obj["coords"]:
            out[int(r), int(c)] = color
    return out


def _random_connected_patch(height: int, width: int, color: int) -> np.ndarray:
    area = height * width
    min_cells = 2 if area >= 2 else 1
    max_cells = min(area, 5)
    target = random.randint(min_cells, max_cells)

    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier = set()
    for nbr in _neighbors4(*start):
        rr, cc = nbr
        if 0 <= rr < height and 0 <= cc < width:
            frontier.add((rr, cc))

    while len(cells) < target and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        rr, cc = cell
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))

    patch = np.full((height, width), BG, dtype=int)
    for r, c in cells:
        patch[r, c] = color
    return patch


def _random_composition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.choices(range(total + 1), k=parts - 1))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _sample_object_columns(h: int, widths: list[int]) -> list[int] | None:
    gap1 = random.randint(1, 2)
    gap2 = random.randint(1, 2)
    total = sum(widths) + gap1 + gap2
    if total > h:
        return None
    slack = h - total
    margin_left, extra_12, extra_23, _margin_right = _random_composition(slack, 4)
    c0 = margin_left
    c1 = c0 + widths[0] + gap1 + extra_12
    c2 = c1 + widths[1] + gap2 + extra_23
    return [c0, c1, c2]


def _place_markers(panel: np.ndarray, count: int) -> bool:
    h, w = panel.shape
    blocked = set()
    for r, c in np.argwhere(panel != BG):
        blocked.add((int(r), int(c)))
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w:
                blocked.add((nr, nc))

    candidates = [(r, c) for r in range(h) for c in range(w) if (r, c) not in blocked]
    random.shuffle(candidates)

    chosen: list[tuple[int, int]] = []
    for cell in candidates:
        if any(abs(cell[0] - rr) + abs(cell[1] - cc) == 1 for rr, cc in chosen):
            continue
        chosen.append(cell)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        return False

    for r, c in chosen:
        panel[r, c] = YELLOW
    return True


def _make_panel(h: int, color: int, selected_slot: int, marker_count: int) -> np.ndarray | None:
    panel = np.full((h, h), BG, dtype=int)

    widths = [random.randint(1, 3) for _ in range(3)]
    columns = _sample_object_columns(h, widths)
    if columns is None:
        return None

    row_bias = random.randrange(max(1, h - 3))
    for i in range(3):
        obj_h = random.randint(1, min(4, h))
        if i == selected_slot and random.random() < 0.6:
            r0 = min(h - obj_h, row_bias)
        else:
            r0 = random.randint(0, h - obj_h)
        c0 = columns[i]
        patch = _random_connected_patch(obj_h, widths[i], color)
        assert np.any(patch == color)
        panel[r0 : r0 + obj_h, c0 : c0 + widths[i]][patch == color] = color

    if len(_ordered_objects(panel)) != 3:
        return None
    if not _place_markers(panel, marker_count):
        return None
    return panel


def _marker_counts_for_panels() -> list[int]:
    sampled = sorted(random.sample(MARKER_COUNT_CHOICES, 3), reverse=True)
    ranked_panels = random.sample(range(3), 3)
    counts = [0, 0, 0]
    for rank, panel_index in enumerate(ranked_panels):
        counts[panel_index] = sampled[rank]
    return counts


def _selected_union_size(grid: np.ndarray) -> tuple[int, int]:
    panels = _split_panels(grid)
    marker_counts = [int(np.count_nonzero(panel == YELLOW)) for panel in panels]
    ranked_panels = sorted(range(3), key=lambda i: marker_counts[i], reverse=True)
    chosen = []
    for rank, panel_index in enumerate(ranked_panels):
        chosen.append(_ordered_objects(panels[panel_index])[rank]["coords"])
    total = sum(int(coords.shape[0]) for coords in chosen)
    union = {tuple(map(int, cell)) for coords in chosen for cell in coords}
    return total, len(union)


def _gen_attempt() -> np.ndarray | None:
    h = random.choice(GRID_SIZES)
    panel_colors = random.sample(PANEL_COLORS, 3)
    marker_counts = _marker_counts_for_panels()
    ranked_panels = sorted(range(3), key=lambda i: marker_counts[i], reverse=True)
    selected_slots = [0, 0, 0]
    for rank, panel_index in enumerate(ranked_panels):
        selected_slots[panel_index] = rank

    panels = []
    for i in range(3):
        panel = _make_panel(h, panel_colors[i], selected_slots[i], marker_counts[i])
        if panel is None:
            return None
        panels.append(panel)

    grid = np.full((h, 3 * h + 2), BG, dtype=int)
    grid[:, :h] = panels[0]
    grid[:, h] = GREY
    grid[:, h + 1 : 2 * h + 1] = panels[1]
    grid[:, 2 * h + 1] = GREY
    grid[:, 2 * h + 2 :] = panels[2]

    out = main(grid)
    assert out.shape == (h, h)
    non_bg = int(np.count_nonzero(out != BG))
    if not (4 <= non_bg <= h * h - 4):
        return None

    # Make overlaps show up sometimes without requiring them every time.
    if random.random() < 0.25:
        total, union = _selected_union_size(grid)
        if union == total:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 6")
