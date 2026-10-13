import random
from typing import Iterable

import numpy as np


BG = 0
OBSTACLE = 5
TOWER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400

COUNT_CHOICES = {
    2: [
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 3),
        (2, 4),
        (2, 5),
        (3, 4),
        (3, 5),
    ],
    3: [
        (1, 2, 3),
        (1, 2, 4),
        (1, 2, 5),
        (1, 3, 4),
        (2, 3, 4),
    ],
}

FAMILY_SIDES = {
    "H": ("left", "right"),
    "V": ("top", "bottom"),
    "NW": ("left", "top"),
    "NE": ("right", "top"),
    "SW": ("left", "bottom"),
    "SE": ("right", "bottom"),
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tower_direction(r: int, c: int, h: int, w: int) -> tuple[int, int]:
    on_top = r == 0
    on_bottom = r == h - 1
    on_left = c == 0
    on_right = c == w - 1
    assert on_top or on_bottom or on_left or on_right, "tower must lie on the border"
    assert sum((on_top, on_bottom, on_left, on_right)) == 1, "corner towers are unsupported"
    if on_top:
        return 1, 0
    if on_bottom:
        return -1, 0
    if on_left:
        return 0, 1
    return 0, -1


def _detect_towers(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    towers: list[dict] = []
    colors_seen: set[int] = set()
    for r in range(h):
        for c in range(w):
            value = int(grid[r, c])
            if value in (BG, OBSTACLE):
                continue
            assert r in (0, h - 1) or c in (0, w - 1), "only border pixels may be towers"
            dr, dc = _tower_direction(r, c, h, w)
            assert value not in colors_seen, "tower colors must be unique"
            colors_seen.add(value)
            towers.append(
                {
                    "index": 0,
                    "color": value,
                    "pos": (int(r), int(c)),
                    "dir": (int(dr), int(dc)),
                }
            )
    towers.sort(key=lambda tower: (tower["pos"][0], tower["pos"][1]))
    for index, tower in enumerate(towers):
        tower["index"] = int(index)
    return towers


def _detect_obstacles(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    h, w = grid.shape
    labels = np.full((h, w), -1, dtype=int)
    obstacles: list[dict] = []
    for coords in _components(grid == OBSTACLE):
        r0, c0, r1, c1 = _bbox(coords)
        assert 0 < r0 <= r1 < h - 1 and 0 < c0 <= c1 < w - 1, "obstacles must be interior"
        rect_h = r1 - r0 + 1
        rect_w = c1 - c0 + 1
        assert coords.shape[0] == rect_h * rect_w, "grey components must be solid rectangles"
        obs_id = len(obstacles)
        labels[r0 : r1 + 1, c0 : c1 + 1] = obs_id
        obstacles.append(
            {
                "id": obs_id,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "contacts": [],
            }
        )
    assert obstacles, "expected at least one obstacle"
    return obstacles, labels


def _analyze_scene(grid: np.ndarray) -> tuple[list[dict], list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 8 and w >= 8, "grid is too small"

    towers = _detect_towers(g)
    obstacles, obstacle_ids = _detect_obstacles(g)
    tower_positions = {tower["pos"] for tower in towers}
    rays: list[dict] = []

    for tower in towers:
        r, c = tower["pos"]
        dr, dc = tower["dir"]
        cells = [(int(r), int(c))]
        hit_id = None
        hit_cell = None

        while True:
            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if (nr, nc) in tower_positions:
                break
            if g[nr, nc] == OBSTACLE:
                hit_id = int(obstacle_ids[nr, nc])
                hit_cell = (int(nr), int(nc))
                assert hit_id >= 0
                obstacles[hit_id]["contacts"].append(
                    {
                        "tower_index": int(tower["index"]),
                        "color": int(tower["color"]),
                        "tower_pos": tower["pos"],
                        "surface": hit_cell,
                    }
                )
                break
            cells.append((int(nr), int(nc)))
            r, c = nr, nc

        rays.append(
            {
                "color": int(tower["color"]),
                "tower_pos": tower["pos"],
                "cells": cells,
                "hit_id": hit_id,
                "hit_cell": hit_cell,
            }
        )

    for obstacle in obstacles:
        obstacle["hit_count"] = int(len(obstacle["contacts"]))

    return towers, obstacles, rays


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast edge rays toward the interior, rank grey rectangles by how many rays hit
    them, then recolor each rectangle by the rank-th contacting ray.
    """
    g = np.asarray(grid, dtype=int)
    towers, obstacles, rays = _analyze_scene(g)
    assert towers, "expected at least one tower"

    out = g.copy()
    for ray in rays:
        color = int(ray["color"])
        for r, c in ray["cells"]:
            out[r, c] = color

    ranked = sorted(
        obstacles,
        key=lambda obstacle: (
            int(obstacle["hit_count"]),
            int(obstacle["bbox"][0]),
            int(obstacle["bbox"][1]),
        ),
    )

    for rank, obstacle in enumerate(ranked, start=1):
        contacts = obstacle["contacts"]
        assert contacts, "every obstacle must be hit by at least one ray"
        assert len(contacts) >= rank, "ranked obstacle lacks enough contacting rays"
        color = int(contacts[rank - 1]["color"])
        r0, c0, r1, c1 = obstacle["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = color

    return out


def _scatter(total: int, slots: int) -> list[int]:
    parts = [0] * slots
    for _ in range(total):
        parts[random.randrange(slots)] += 1
    return parts


def _sample_intervals(start: int, stop: int, sizes: Iterable[int], gap: int = 1) -> list[tuple[int, int]] | None:
    sizes = list(sizes)
    usable = int(stop - start + 1)
    needed = int(sum(sizes) + gap * (len(sizes) - 1))
    if needed > usable:
        return None
    slack = usable - needed
    extras = _scatter(slack, len(sizes) + 1)
    pos = start + extras[0]
    intervals: list[tuple[int, int]] = []
    for i, size in enumerate(sizes):
        end = pos + int(size) - 1
        intervals.append((int(pos), int(end)))
        if i < len(sizes) - 1:
            pos = end + 1 + gap + extras[i + 1]
    return intervals


def _contact_positions(
    bbox: tuple[int, int, int, int],
    h: int,
    w: int,
    allowed_sides: tuple[str, ...],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    positions: list[tuple[int, int]] = []
    if "left" in allowed_sides:
        positions.extend((int(r), 0) for r in range(r0, r1 + 1))
    if "right" in allowed_sides:
        positions.extend((int(r), w - 1) for r in range(r0, r1 + 1))
    if "top" in allowed_sides:
        positions.extend((0, int(c)) for c in range(c0, c1 + 1))
    if "bottom" in allowed_sides:
        positions.extend((h - 1, int(c)) for c in range(c0, c1 + 1))
    return positions


def _free_rows(obstacles: list[dict], h: int) -> list[int]:
    blocked = set()
    for obstacle in obstacles:
        r0, _, r1, _ = obstacle["bbox"]
        blocked.update(range(r0, r1 + 1))
    return [r for r in range(1, h - 1) if r not in blocked]


def _free_cols(obstacles: list[dict], w: int) -> list[int]:
    blocked = set()
    for obstacle in obstacles:
        _, c0, _, c1 = obstacle["bbox"]
        blocked.update(range(c0, c1 + 1))
    return [c for c in range(1, w - 1) if c not in blocked]


def _pick_distractors(
    family: str,
    obstacles: list[dict],
    h: int,
    w: int,
    budget: int,
) -> list[tuple[int, int]]:
    if budget <= 0 or family not in {"H", "V"}:
        return []

    distractors: list[tuple[int, int]] = []
    if family == "H":
        rows = _free_rows(obstacles, h)
        random.shuffle(rows)
        for row in rows:
            if len(distractors) >= budget:
                break
            distractors.append((int(row), random.choice([0, w - 1])))
    else:
        cols = _free_cols(obstacles, w)
        random.shuffle(cols)
        for col in cols:
            if len(distractors) >= budget:
                break
            distractors.append((random.choice([0, h - 1]), int(col)))
    return distractors


def _gen_attempt() -> np.ndarray | None:
    n_obstacles = random.choice([2, 3])
    family = random.choice(list(FAMILY_SIDES))
    allowed_sides = FAMILY_SIDES[family]

    counts_sorted = list(random.choice(COUNT_CHOICES[n_obstacles]))
    sizes_r = [random.randint(3, 4) for _ in range(n_obstacles)]
    sizes_c = [random.randint(3, 4) for _ in range(n_obstacles)]
    random.shuffle(sizes_r)
    random.shuffle(sizes_c)

    min_h = 4 + sum(sizes_r) + (n_obstacles - 1)
    min_w = 4 + sum(sizes_c) + (n_obstacles - 1)
    h = random.randint(min_h, min_h + 4)
    w = random.randint(min_w, min_w + 4)

    row_intervals = _sample_intervals(2, h - 3, sizes_r, gap=1)
    col_intervals = _sample_intervals(2, w - 3, sizes_c, gap=1)
    if row_intervals is None or col_intervals is None:
        return None

    if family in {"NW", "SE"}:
        col_order = list(range(n_obstacles))
    elif family in {"NE", "SW"}:
        col_order = list(range(n_obstacles - 1, -1, -1))
    else:
        col_order = random.sample(range(n_obstacles), n_obstacles)

    obstacles: list[dict] = []
    counts = counts_sorted[:]
    random.shuffle(counts)
    for i, row_idx in enumerate(range(n_obstacles)):
        r0, r1 = row_intervals[row_idx]
        c0, c1 = col_intervals[col_order[i]]
        obstacles.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "target_count": int(counts[i]),
            }
        )

    tower_positions: list[tuple[int, int]] = []
    expected_counts: dict[tuple[int, int, int, int], int] = {}
    for obstacle in obstacles:
        bbox = obstacle["bbox"]
        choices = _contact_positions(bbox, h, w, allowed_sides)
        if obstacle["target_count"] > len(choices):
            return None
        selected = random.sample(choices, obstacle["target_count"])
        tower_positions.extend(selected)
        expected_counts[bbox] = int(obstacle["target_count"])

    if len(tower_positions) > len(TOWER_COLORS):
        return None

    extra_budget = min(2, len(TOWER_COLORS) - len(tower_positions))
    extra_count = random.randint(0, extra_budget)
    tower_positions.extend(_pick_distractors(family, obstacles, h, w, extra_count))
    if len(set(tower_positions)) != len(tower_positions):
        return None

    colors = random.sample(TOWER_COLORS, len(tower_positions))
    grid = np.full((h, w), BG, dtype=int)
    for obstacle in obstacles:
        r0, c0, r1, c1 = obstacle["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = OBSTACLE
    for color, (r, c) in zip(colors, tower_positions):
        grid[r, c] = int(color)

    try:
        _towers, analyzed_obstacles, _rays = _analyze_scene(grid)
    except AssertionError:
        return None

    actual_counts = {tuple(obstacle["bbox"]): int(obstacle["hit_count"]) for obstacle in analyzed_obstacles}
    if actual_counts != expected_counts:
        return None

    ranked_counts = sorted(actual_counts.values())
    for rank, count in enumerate(ranked_counts, start=1):
        if count < rank:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
