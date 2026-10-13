from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
ORTHOGONAL_STEPS: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _priority_order(grid: np.ndarray, bg: int) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    for raw in grid[-1]:
        color = int(raw)
        if color == bg or color in seen:
            continue
        seen.add(color)
        order.append(color)
    assert order, "bottom row must contain a non-background priority strip"
    return order


def _infer_layers(body: np.ndarray, colors: list[int], bg: int) -> dict[int, np.ndarray]:
    """Recover hidden crossing cells by extending each color through straight-through gaps."""
    layers = {color: (body == color).copy() for color in colors}
    non_bg = body != bg
    changed = True

    while changed:
        changed = False
        for color in colors:
            mask = layers[color]
            add = np.zeros_like(mask, dtype=bool)
            h, w = mask.shape

            for r in range(h):
                for c in range(w):
                    if not non_bg[r, c] or mask[r, c]:
                        continue

                    horizontal = c > 0 and c + 1 < w and mask[r, c - 1] and mask[r, c + 1]
                    vertical = r > 0 and r + 1 < h and mask[r - 1, c] and mask[r + 1, c]
                    if horizontal or vertical:
                        add[r, c] = True

            if np.any(add):
                layers[color] = mask | add
                changed = True

    return layers


def _is_simple_path(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False

    points = {tuple(map(int, coord)) for coord in coords}
    start = next(iter(points))
    stack = [start]
    seen = {start}

    while stack:
        r, c = stack.pop()
        for dr, dc in ORTHOGONAL_STEPS:
            nxt = (r + dr, c + dc)
            if nxt in points and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)

    if len(seen) != len(points):
        return False

    degrees: list[int] = []
    for r, c in points:
        degree = sum((r + dr, c + dc) in points for dr, dc in ORTHOGONAL_STEPS)
        if degree > 2:
            return False
        degrees.append(degree)

    if len(points) == 1:
        return True
    return degrees.count(1) == 2 and degrees.count(2) == len(points) - 2


def _final_body(body: np.ndarray, priorities: list[int], bg: int) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    observed_colors = {int(color) for color in np.unique(body) if int(color) != bg}
    assert observed_colors.issubset(set(priorities)), "every visible path color must appear in the priority strip"

    layers = _infer_layers(body, priorities, bg)
    final = np.full(body.shape, bg, dtype=int)
    for color in reversed(priorities):
        final[layers[color]] = color
    return final, layers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split overlapped orthogonal paths into color layers using the bottom priority strip,
    then keep only the highest-priority path that remains a complete 4-connected path.

    Input structure:
    - The dominant color is the background.
    - The last row is a left-to-right priority strip whose non-background colors are
      ordered from highest to lowest priority.
    - Above that row are single-cell-wide orthogonal paths with unique colors.
      Crossings are shown using only one of the two path colors.

    Rule:
    - Reconstruct hidden crossing ownership by allowing each color to continue straight
      through non-background crossing cells where it has opposite neighbors.
    - Recolor every occupied crossing cell to the higher-priority occupying color.
    - Among the recolored visible paths, keep only the highest-priority color whose
      pixels still form a single simple 4-connected path; erase everything else,
      including the priority strip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2

    bg = _majority_color(g)
    priorities = _priority_order(g, bg)
    body = g[:-1, :]

    final, _layers = _final_body(body, priorities, bg)

    winner = None
    for color in priorities:
        if _is_simple_path(final == color):
            winner = color
            break

    out = np.full_like(g, bg)
    if winner is not None:
        out[:-1, :][final == winner] = winner
    return out


def _trace_segment(a: Coord, b: Coord) -> tuple[list[Coord], str]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)], "h"
    if a[1] == b[1]:
        step = 1 if b[0] >= a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)], "v"
    raise AssertionError("path segments must be orthogonal")


def _build_path(points: list[Coord] | None, color: int) -> dict | None:
    if points is None:
        return None

    orientations: dict[Coord, set[str]] = {}
    cells: set[Coord] = set()

    for i in range(len(points) - 1):
        segment, axis = _trace_segment(points[i], points[i + 1])
        for j, cell in enumerate(segment):
            if cell in orientations and not (j == 0 and cell == points[i]):
                return None
            orientations.setdefault(cell, set()).add(axis)
            cells.add(cell)

    if len(cells) < 5:
        return None

    degrees: dict[Coord, int] = {}
    for r, c in cells:
        degree = sum((r + dr, c + dc) in cells for dr, dc in ORTHOGONAL_STEPS)
        if degree > 2:
            return None
        degrees[(r, c)] = degree

    if len(cells) > 1 and sum(degree == 1 for degree in degrees.values()) != 2:
        return None

    return {
        "color": int(color),
        "cells": cells,
        "degrees": degrees,
        "orient": {
            cell: ("hv" if len(axes) == 2 else next(iter(axes)))
            for cell, axes in orientations.items()
        },
    }


def _straight_interior(path: dict, cell: Coord) -> bool:
    return path["degrees"][cell] == 2 and path["orient"][cell] in {"h", "v"}


def _candidate_tracks(limit: int) -> list[int]:
    start = random.choice([1, 2])
    tracks = list(range(start, limit - 1, 2))
    if len(tracks) < 5:
        tracks = list(range(1, limit - 1, 2))
    return tracks


def _sample_points(kind: str, rows: list[int], cols: list[int]) -> list[Coord] | None:
    if kind == "h":
        r = random.choice(rows)
        c0, c1 = sorted(random.sample(cols, 2))
        return [(r, c0), (r, c1)] if c1 - c0 >= 4 else None

    if kind == "v":
        c = random.choice(cols)
        r0, r1 = sorted(random.sample(rows, 2))
        return [(r0, c), (r1, c)] if r1 - r0 >= 4 else None

    if kind == "l":
        r0, r1 = random.sample(rows, 2)
        c0, c1 = random.sample(cols, 2)
        if random.random() < 0.5:
            return [(r0, c0), (r0, c1), (r1, c1)]
        return [(r0, c0), (r1, c0), (r1, c1)]

    if kind == "hvh":
        if len(cols) < 3:
            return None
        middle = random.choice(cols[1:-1])
        left_cols = [c for c in cols if c < middle - 1]
        right_cols = [c for c in cols if c > middle + 1]
        if not left_cols or not right_cols:
            return None
        r0, r1 = random.sample(rows, 2)
        points = [
            (r0, random.choice(left_cols)),
            (r0, middle),
            (r1, middle),
            (r1, random.choice(right_cols)),
        ]
        return points if random.random() < 0.5 else list(reversed(points))

    if kind == "vhv":
        if len(rows) < 3:
            return None
        middle = random.choice(rows[1:-1])
        top_rows = [r for r in rows if r < middle - 1]
        bottom_rows = [r for r in rows if r > middle + 1]
        if not top_rows or not bottom_rows:
            return None
        c0, c1 = random.sample(cols, 2)
        points = [
            (random.choice(top_rows), c0),
            (middle, c0),
            (middle, c1),
            (random.choice(bottom_rows), c1),
        ]
        return points if random.random() < 0.5 else list(reversed(points))

    return None


def _can_add_path(path: dict, paths: list[dict], occupants: dict[Coord, list[int]], crossings: set[Coord]) -> bool:
    local_crossings: list[Coord] = []

    for cell in path["cells"]:
        holders = occupants.get(cell, [])
        if len(holders) > 1:
            return False
        if not holders:
            continue

        if cell in crossings:
            return False
        if any(abs(cell[0] - r) + abs(cell[1] - c) == 1 for r, c in crossings):
            return False
        if any(abs(cell[0] - r) + abs(cell[1] - c) == 1 for r, c in local_crossings):
            return False

        other = paths[holders[0]]
        if not (_straight_interior(path, cell) and _straight_interior(other, cell)):
            return False
        if path["orient"][cell] == other["orient"][cell]:
            return False
        local_crossings.append(cell)

    return True


def _add_path(path: dict, paths: list[dict], occupants: dict[Coord, list[int]], crossings: set[Coord]) -> None:
    index = len(paths)
    paths.append(path)
    for cell in path["cells"]:
        occupants.setdefault(cell, []).append(index)
        if len(occupants[cell]) == 2:
            crossings.add(cell)


def _render_ground_truth(
    body_shape: tuple[int, int],
    bg: int,
    paths: list[dict],
    priorities: list[int],
) -> tuple[np.ndarray, list[int]]:
    layers = {path["color"]: np.zeros(body_shape, dtype=bool) for path in paths}
    for path in paths:
        for r, c in path["cells"]:
            layers[path["color"]][r, c] = True

    final = np.full(body_shape, bg, dtype=int)
    for color in reversed(priorities):
        final[layers[color]] = color

    survivors = [color for color in priorities if _is_simple_path(final == color)]
    assert survivors, "at least one path must remain complete"
    return final, survivors


def _render_input_grid(
    body_shape: tuple[int, int],
    bg: int,
    paths: list[dict],
    occupants: dict[Coord, list[int]],
    priorities: list[int],
) -> np.ndarray:
    body_h, body_w = body_shape
    grid = np.full((body_h + 1, body_w), bg, dtype=int)
    rank = {color: i for i, color in enumerate(priorities)}

    crossing_choices: list[tuple[Coord, int]] = []
    for cell, ids in occupants.items():
        colors_here = [paths[i]["color"] for i in ids]
        if len(colors_here) == 2:
            lower = max(colors_here, key=lambda color: rank[color])
            crossing_choices.append((cell, lower))

    forced_lower = random.choice(crossing_choices) if crossing_choices else None

    for cell, ids in occupants.items():
        colors_here = [paths[i]["color"] for i in ids]
        if len(colors_here) == 1:
            grid[cell] = colors_here[0]
        elif forced_lower is not None and cell == forced_lower[0]:
            grid[cell] = forced_lower[1]
        else:
            grid[cell] = random.choice(colors_here)

    start = random.randint(0, body_w - len(priorities))
    grid[-1, start : start + len(priorities)] = priorities
    return grid


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(800):
        bg = random.choice([0, 1, 8])
        body_h = random.randint(11, 16)
        body_w = random.randint(12, 18)
        n_paths = random.randint(3, 5)
        colors = random.sample([color for color in range(10) if color != bg], n_paths)
        rows = _candidate_tracks(body_h)
        cols = _candidate_tracks(body_w)
        if len(rows) < 4 or len(cols) < 4:
            continue

        paths: list[dict] = []
        occupants: dict[Coord, list[int]] = {}
        crossings: set[Coord] = set()

        for color in colors:
            placed = False
            for _ in range(500):
                kind = random.choice(["h", "v", "l", "l", "hvh", "vhv"])
                path = _build_path(_sample_points(kind, rows, cols), color)
                if path is None:
                    continue
                if _can_add_path(path, paths, occupants, crossings):
                    _add_path(path, paths, occupants, crossings)
                    placed = True
                    break
            if not placed:
                break

        if len(paths) != n_paths or len(crossings) < 2:
            continue

        priorities = []
        final_body = None
        survivors: list[int] = []
        require_mixed = random.random() < 0.5

        for _ in range(100):
            trial = random.sample([path["color"] for path in paths], len(paths))
            final_trial, survivor_trial = _render_ground_truth((body_h, body_w), bg, paths, trial)
            broken_non_top = [color for color in trial[1:] if color not in survivor_trial]
            surviving_non_top = [color for color in trial[1:] if color in survivor_trial]
            if broken_non_top and (surviving_non_top or not require_mixed):
                priorities = trial
                final_body = final_trial
                survivors = survivor_trial
                break

        if final_body is None:
            continue

        grid = _render_input_grid((body_h, body_w), bg, paths, occupants, priorities)
        expected = np.full_like(grid, bg)
        expected[:-1, :][final_body == survivors[0]] = survivors[0]

        if np.array_equal(main(grid), expected):
            return grid, expected

    raise RuntimeError("failed to generate a valid ARC instance")


def generate_input() -> np.ndarray:
    grid, _expected = _generate_case()
    return grid
