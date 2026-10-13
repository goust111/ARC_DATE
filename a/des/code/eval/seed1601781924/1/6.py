from __future__ import annotations

from collections import deque
import random
import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7
MAROON = 9

SOURCE_COLORS = (RED, BLUE, YELLOW)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

MIX_MAP = {
    frozenset([RED]): RED,
    frozenset([BLUE]): BLUE,
    frozenset([YELLOW]): YELLOW,
    frozenset([RED, BLUE]): MAGENTA,
    frozenset([RED, YELLOW]): ORANGE,
    frozenset([BLUE, YELLOW]): GREEN,
    frozenset([RED, BLUE, YELLOW]): MAROON,
}

SUBSETS_BY_SIZE = {
    1: [(RED,), (BLUE,), (YELLOW,)],
    2: [(RED, BLUE), (RED, YELLOW), (BLUE, YELLOW)],
    3: [(RED, BLUE, YELLOW)],
}

GEN_RETRIES = 300


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels = np.zeros(mask.shape, dtype=int)
    next_label = 0
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            next_label += 1
            q = deque([(r, c)])
            labels[r, c] = next_label
            while q:
                cr, cc = q.popleft()
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_label
                        q.append((nr, nc))
    return labels, next_label


def _find_sources(grid: np.ndarray) -> list[dict[str, object]]:
    sources: list[dict[str, object]] = []
    for color in SOURCE_COLORS:
        labels, n_labels = _label_components(grid == color)
        for label_id in range(1, n_labels + 1):
            cells = np.argwhere(labels == label_id)
            if len(cells) != 4:
                continue
            rows = cells[:, 0]
            cols = cells[:, 1]
            if int(rows.max()) - int(rows.min()) != 1:
                continue
            if int(cols.max()) - int(cols.min()) != 1:
                continue
            points = [(int(r), int(c)) for r, c in cells]
            sources.append({"color": int(color), "cells": points})
    return sources


def _find_containers(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    centers: list[tuple[int, int]] = []
    ring = np.ones((3, 3), dtype=bool)
    ring[1, 1] = False
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != BG:
                continue
            patch = grid[r - 1 : r + 2, c - 1 : c + 2]
            if patch.shape != (3, 3):
                continue
            if bool(np.all(patch[ring] == GRAY)):
                centers.append((r, c))
    return centers


def _adjacent_gray_labels(labels: np.ndarray, cells: list[tuple[int, int]]) -> set[int]:
    h, w = labels.shape
    connected: set[int] = set()
    for r, c in cells:
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and int(labels[nr, nc]) > 0:
                connected.add(int(labels[nr, nc]))
    return connected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate source colors through connected gray pipe networks into hollow gray containers.
    """
    g = np.asarray(grid, dtype=int)
    gray_labels, n_gray = _label_components(g == GRAY)
    component_colors = {label_id: set() for label_id in range(1, n_gray + 1)}

    for source in _find_sources(g):
        color = int(source["color"])
        cells = list(source["cells"])
        for label_id in _adjacent_gray_labels(gray_labels, cells):
            component_colors[label_id].add(color)

    out = g.copy()
    for r, c in _find_containers(g):
        frame_cells = [
            (r - 1, c - 1),
            (r - 1, c),
            (r - 1, c + 1),
            (r, c - 1),
            (r, c + 1),
            (r + 1, c - 1),
            (r + 1, c),
            (r + 1, c + 1),
        ]
        colors: set[int] = set()
        for fr, fc in frame_cells:
            label_id = int(gray_labels[fr, fc])
            if label_id > 0:
                colors.update(component_colors.get(label_id, set()))
        if colors:
            out[r, c] = MIX_MAP[frozenset(colors)]
    return out


def _random_composition(total: int, parts: int, rng: random.Random) -> list[int]:
    buckets = [0] * parts
    for _ in range(total):
        buckets[rng.randrange(parts)] += 1
    return buckets


def _pack_starts(total_len: int, item_len: int, count: int, rng: random.Random) -> list[int] | None:
    required = count * item_len + max(0, count - 1)
    if required > total_len:
        return None
    gaps = _random_composition(total_len - required, count + 1, rng)
    starts: list[int] = []
    pos = gaps[0]
    for idx in range(count):
        starts.append(pos)
        pos += item_len
        if idx + 1 < count:
            pos += 1 + gaps[idx + 1]
    return starts


def _draw_source(grid: np.ndarray, top: int, left: int, color: int) -> None:
    grid[top : top + 2, left : left + 2] = color


def _draw_container(grid: np.ndarray, top: int, left: int) -> tuple[int, int]:
    grid[top : top + 3, left : left + 3] = GRAY
    grid[top + 1, left + 1] = BG
    return top + 1, left + 1


def _draw_line(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    if r0 == r1:
        c_start = min(c0, c1)
        c_end = max(c0, c1)
        grid[r0, c_start : c_end + 1] = color
        return
    if c0 == c1:
        r_start = min(r0, r1)
        r_end = max(r0, r1)
        grid[r_start : r_end + 1, c0] = color
        return
    raise ValueError("only axis-aligned lines are supported")


def _transform_points(
    points: list[tuple[int, int, int]],
    shape: tuple[int, int],
    *,
    flip_lr: bool,
    flip_ud: bool,
    transpose: bool,
) -> list[tuple[int, int, int]]:
    h, w = shape
    out = points
    if flip_lr:
        out = [(r, w - 1 - c, color) for r, c, color in out]
    if flip_ud:
        out = [(h - 1 - r, c, color) for r, c, color in out]
    if transpose:
        out = [(c, r, color) for r, c, color in out]
    return out


def _choose_network_specs(rng: random.Random) -> list[tuple[int, ...]]:
    n_networks = rng.choice([2, 3])
    if n_networks == 3:
        sizes = [1, 2, 3]
        rng.shuffle(sizes)
    else:
        sizes = rng.sample([1, 2, 3], 2)
    return [tuple(rng.choice(SUBSETS_BY_SIZE[size])) for size in sizes]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    subset_specs = _choose_network_specs(rng)
    network_specs: list[dict[str, object]] = []
    for subset in subset_specs:
        sources = list(subset)
        if rng.random() < 0.35:
            sources.append(rng.choice(sources))
        rng.shuffle(sources)
        n_containers = 1 if rng.random() < 0.45 else 2
        source_height = len(sources) * 2 + max(0, len(sources) - 1)
        container_height = n_containers * 3 + max(0, n_containers - 1)
        band_height = max(source_height, container_height) + rng.randint(0, 1)
        network_specs.append(
            {
                "subset": subset,
                "sources": sources,
                "n_containers": n_containers,
                "band_height": band_height,
            }
        )

    band_gap = 1
    height = sum(int(spec["band_height"]) for spec in network_specs) + band_gap * (len(network_specs) - 1)
    width = rng.randint(18, 28)
    if height > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    expected_centers: list[tuple[int, int, int]] = []

    row0 = 0
    for spec in network_specs:
        band_h = int(spec["band_height"])
        sources = list(spec["sources"])
        n_containers = int(spec["n_containers"])

        source_rows = _pack_starts(band_h, 2, len(sources), rng)
        container_rows = _pack_starts(band_h, 3, n_containers, rng)
        if source_rows is None or container_rows is None:
            return None

        src_left = rng.randint(1, 3)
        cont_left_min = max(width - 6, src_left + 9)
        cont_left_max = width - 3
        if cont_left_min > cont_left_max:
            return None
        cont_left = rng.randint(cont_left_min, cont_left_max)
        hub_min = src_left + 4
        hub_max = cont_left - 2
        if hub_min > hub_max:
            return None
        hub_col = rng.randint(hub_min, hub_max)

        anchor_rows: list[int] = []
        source_records: list[tuple[int, int, int, int]] = []
        for offset, color in zip(source_rows, sources):
            top = row0 + offset
            anchor_row = top + rng.choice([0, 1])
            source_records.append((top, src_left, anchor_row, int(color)))
            anchor_rows.append(anchor_row)

        container_records: list[tuple[int, int, int, int]] = []
        expected_color = MIX_MAP[frozenset(spec["subset"])]
        for offset in container_rows:
            top = row0 + offset
            center_row = top + 1
            container_records.append((top, cont_left, center_row, expected_color))
            anchor_rows.append(center_row)

        _draw_line(grid, min(anchor_rows), hub_col, max(anchor_rows), hub_col, GRAY)
        for top, left, anchor_row, color in source_records:
            _draw_source(grid, top, left, color)
            _draw_line(grid, anchor_row, left + 2, anchor_row, hub_col, GRAY)
        for top, left, center_row, center_color in container_records:
            center = _draw_container(grid, top, left)
            _draw_line(grid, center_row, hub_col, center_row, left, GRAY)
            expected_centers.append((center[0], center[1], center_color))

        row0 += band_h + band_gap

    flip_lr = rng.random() < 0.5
    flip_ud = rng.random() < 0.5
    transpose = rng.random() < 0.5
    if flip_lr:
        grid = np.fliplr(grid)
    if flip_ud:
        grid = np.flipud(grid)
    if transpose:
        grid = grid.T.copy()
    expected_centers = _transform_points(
        expected_centers,
        (height, width),
        flip_lr=flip_lr,
        flip_ud=flip_ud,
        transpose=transpose,
    )

    if len(_find_containers(grid)) != len(expected_centers):
        return None
    out = main(grid)
    for r, c, color in expected_centers:
        if int(out[r, c]) != int(color):
            return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
