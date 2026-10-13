from __future__ import annotations

from collections import deque
import random

import numpy as np


Cell = tuple[int, int]


GRID_MIN = 8
GRID_MAX = 16
GEN_ATTEMPTS = 400
DECOR_RETRIES = 40

DECOR_SHAPES: tuple[tuple[Cell, ...], ...] = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
)

SIZE_CHOICES: tuple[tuple[int, int], ...] = tuple(
    (h, w)
    for h in range(GRID_MIN, GRID_MAX + 1)
    for w in range(GRID_MIN, GRID_MAX + 1)
    if abs(h - w) <= 1
)


def _neighbors4(r: int, c: int) -> tuple[Cell, Cell, Cell, Cell]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _ordered_colors(g: np.ndarray) -> list[int]:
    seen: set[int] = set()
    colors: list[int] = []
    for value in g.flat:
        color = int(value)
        if color not in seen:
            seen.add(color)
            colors.append(color)
    return colors


def _components(mask: np.ndarray) -> list[list[Cell]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Cell]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[Cell] = deque([(r, c)])
            seen[r, c] = True
            pts: list[Cell] = []

            while q:
                x, y = q.popleft()
                pts.append((x, y))
                for nx, ny in _neighbors4(x, y):
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))

            comps.append(pts)

    return comps


def _majority_winners(values: list[int]) -> list[int]:
    counts: dict[int, int] = {}
    order: list[int] = []
    for value in values:
        if value not in counts:
            counts[value] = 0
            order.append(value)
        counts[value] += 1

    if not order:
        return []

    best = max(counts[value] for value in order)
    return [value for value in order if counts[value] == best]


def _dominant_value(values: list[int]) -> int:
    winners = _majority_winners(values)
    assert winners, "expected at least one value"
    return int(winners[0])


def _touch_sides(pts: list[Cell], h: int, w: int) -> set[str]:
    sides: set[str] = set()
    for r, c in pts:
        if r == 0:
            sides.add("top")
        if r == h - 1:
            sides.add("bottom")
        if c == 0:
            sides.add("left")
        if c == w - 1:
            sides.add("right")
    return sides


def _is_simple_path(pts: list[Cell]) -> bool:
    pset = set(pts)
    endpoints = 0
    for r, c in pts:
        degree = sum((nr, nc) in pset for nr, nc in _neighbors4(r, c))
        if degree > 2:
            return False
        if degree == 1:
            endpoints += 1
        elif degree == 0:
            return False
    return endpoints == 2


def _find_crack(
    g: np.ndarray,
) -> tuple[int, np.ndarray, list[list[Cell]]]:
    h, w = g.shape
    best: tuple[tuple[int, int, int], int, np.ndarray, list[list[Cell]]] | None = None

    for color in _ordered_colors(g):
        mask = g == color
        comps = _components(mask)
        if len(comps) != 1:
            continue

        pts = comps[0]
        if len(pts) < min(h, w):
            continue
        if not _is_simple_path(pts):
            continue

        side_hits = _touch_sides(pts, h, w)
        if len(side_hits) < 2:
            continue

        remaining = _components(~mask)
        if len(remaining) != 2:
            continue

        sizes = [len(comp) for comp in remaining]
        score = (min(sizes), len(pts), len(side_hits))
        if best is None or score > best[0]:
            best = (score, int(color), mask, remaining)

    assert best is not None, "could not identify a unique crack path"
    _, crack_color, crack_mask, side_comps = best
    return crack_color, crack_mask, side_comps


def _component_labels(
    g: np.ndarray, comps: list[list[Cell]]
) -> tuple[np.ndarray, list[int], list[int]]:
    labels = np.full(g.shape, -1, dtype=int)
    sizes: list[int] = []
    dominant_colors: list[int] = []

    for idx, comp in enumerate(comps):
        sizes.append(len(comp))
        dominant_colors.append(_dominant_value([int(g[r, c]) for r, c in comp]))
        for r, c in comp:
            labels[r, c] = idx

    return labels, sizes, dominant_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a 1-pixel-wide crack by repainting each crack cell with the majority
    color among its 4-neighbor non-crack cells.
    """
    g = np.asarray(grid, dtype=int)
    crack_color, crack_mask, side_comps = _find_crack(g)
    labels, side_sizes, side_dominant = _component_labels(g, side_comps)

    out = g.copy()
    for r0, c0 in zip(*np.where(crack_mask), strict=False):
        r = int(r0)
        c = int(c0)
        neighbor_colors: list[int] = []
        neighbor_labels: list[int] = []

        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and not crack_mask[nr, nc]:
                neighbor_colors.append(int(g[nr, nc]))
                neighbor_labels.append(int(labels[nr, nc]))

        assert neighbor_colors, "every crack cell must touch at least one non-crack cell"

        color_winners = _majority_winners(neighbor_colors)
        if len(color_winners) == 1:
            out[r, c] = color_winners[0]
            continue

        label_winners = _majority_winners([lbl for lbl in neighbor_labels if lbl >= 0])
        fill = None

        if len(label_winners) == 1:
            label = label_winners[0]
            same_side_local = [
                color
                for color, lbl in zip(neighbor_colors, neighbor_labels, strict=False)
                if lbl == label
            ]
            side_local_winners = _majority_winners(same_side_local)
            if len(side_local_winners) == 1:
                fill = side_local_winners[0]
            else:
                fill = side_dominant[label]

        if fill is None:
            adjacent_labels = [lbl for lbl in dict.fromkeys(neighbor_labels) if lbl >= 0]
            if adjacent_labels:
                biggest = max(side_sizes[lbl] for lbl in adjacent_labels)
                biggest_labels = [lbl for lbl in adjacent_labels if side_sizes[lbl] == biggest]
                if len(biggest_labels) == 1:
                    fill = side_dominant[biggest_labels[0]]

        if fill is None:
            fill = color_winners[0]

        out[r, c] = int(fill)

    assert not np.any(out == crack_color), "crack color should be fully repaired"
    return out


# --------------- generator ---------------


def _alternating_steps(vertical_steps: int, horizontal_steps: int, start_vertical: bool) -> list[str]:
    steps: list[str] = []
    rem_v = vertical_steps
    rem_h = horizontal_steps
    vertical = start_vertical

    while rem_v or rem_h:
        if vertical:
            if rem_v == 0:
                return []
            steps.append("V")
            rem_v -= 1
        else:
            if rem_h == 0:
                return []
            steps.append("H")
            rem_h -= 1
        vertical = not vertical

    return steps


def _staircase_path(h: int, w: int, mirror: bool, start_vertical: bool) -> list[Cell]:
    vertical_steps = h - 1
    horizontal_steps = w - 1
    steps = _alternating_steps(vertical_steps, horizontal_steps, start_vertical)
    assert steps, "staircase path requires alternating steps"

    r = 0
    c = w - 1 if mirror else 0
    dc = -1 if mirror else 1
    pts = [(r, c)]

    for step in steps:
        if step == "V":
            r += 1
        else:
            c += dc
        pts.append((r, c))

    return pts


def _transform_shape(shape: tuple[Cell, ...]) -> list[Cell]:
    pts = list(shape)
    for _ in range(random.randint(0, 3)):
        pts = [(c, -r) for r, c in pts]
    if random.random() < 0.5:
        pts = [(r, -c) for r, c in pts]

    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return [(r - min_r, c - min_c) for r, c in pts]


def _place_decorations(
    grid: np.ndarray,
    region: list[Cell],
    main_color: int,
    banned_colors: set[int],
) -> None:
    h, w = grid.shape
    region_set = set(region)
    interior = [
        (r, c)
        for r, c in region
        if all(0 <= nr < h and 0 <= nc < w and (nr, nc) in region_set for nr, nc in _neighbors4(r, c))
    ]
    if not interior:
        return

    occupied: set[Cell] = set()
    for _ in range(random.randint(0, 2)):
        available = [color for color in range(10) if color not in banned_colors and color != main_color]
        if not available:
            break

        color = random.choice(available)
        placed = False
        for _ in range(DECOR_RETRIES):
            anchor = random.choice(interior)
            shape = _transform_shape(random.choice(DECOR_SHAPES))
            cells = [(anchor[0] + dr, anchor[1] + dc) for dr, dc in shape]

            if any(cell not in region_set or cell in occupied for cell in cells):
                continue
            if any(
                any((nr, nc) not in region_set for nr, nc in _neighbors4(r, c))
                for r, c in cells
            ):
                continue

            for r, c in cells:
                grid[r, c] = color
                occupied.add((r, c))
            banned_colors.add(color)
            placed = True
            break

        if not placed:
            continue


def _validate_generated(grid: np.ndarray) -> bool:
    crack_color, crack_mask, side_comps = _find_crack(grid)
    if min(len(comp) for comp in side_comps) < min(grid.shape):
        return False

    for r0, c0 in zip(*np.where(crack_mask), strict=False):
        r = int(r0)
        c = int(c0)
        neighbor_colors = [
            int(grid[nr, nc])
            for nr, nc in _neighbors4(r, c)
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and not crack_mask[nr, nc]
        ]
        if len(_majority_winners(neighbor_colors)) != 1:
            return False

    repaired = main(grid)
    if np.any(repaired == crack_color):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h, w = random.choice(SIZE_CHOICES)

        if h > w:
            start_vertical = True
        elif w > h:
            start_vertical = False
        else:
            start_vertical = random.choice([False, True])

        crack_path = _staircase_path(
            h,
            w,
            mirror=random.choice([False, True]),
            start_vertical=start_vertical,
        )
        crack_mask = np.zeros((h, w), dtype=bool)
        for r, c in crack_path:
            crack_mask[r, c] = True

        side_comps = _components(~crack_mask)
        if len(side_comps) != 2:
            continue

        base_colors = random.sample(range(10), 3)
        crack_color = base_colors[0]
        main_colors = random.sample([color for color in range(10) if color != crack_color], 2)

        grid = np.empty((h, w), dtype=int)
        grid[crack_mask] = crack_color
        for idx, comp in enumerate(side_comps):
            for r, c in comp:
                grid[r, c] = main_colors[idx]

        banned_colors = {crack_color, *main_colors}
        _place_decorations(grid, side_comps[0], main_colors[0], banned_colors)
        _place_decorations(grid, side_comps[1], main_colors[1], banned_colors)

        k = random.randint(0, 3)
        if k:
            grid = np.rot90(grid, k).copy()

        if _validate_generated(grid):
            return grid

    raise RuntimeError("failed to generate a valid crack-repair input")
