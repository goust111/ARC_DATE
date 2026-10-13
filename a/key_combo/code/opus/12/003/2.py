from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
WALL = 5
MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def components_of_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def dilate_cells(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int) -> set[tuple[int, int]]:
    h, w = shape
    out = set(cells)
    frontier = set(cells)
    for _ in range(radius):
        nxt: set[tuple[int, int]] = set()
        for r, c in frontier:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in out:
                    nxt.add((nr, nc))
        out |= nxt
        frontier = nxt
    return out


def extract_palette(grid: np.ndarray) -> tuple[list[int], tuple[int, int], np.ndarray]:
    row = np.asarray(grid[0], dtype=int)
    colored = [c for c, val in enumerate(row) if val not in (BG, WALL)]
    assert colored, "expected a palette segment on the top row"

    segments: list[tuple[int, int]] = []
    start = colored[0]
    prev = colored[0]
    for col in colored[1:]:
        if col != prev + 1:
            segments.append((start, prev))
            start = col
        prev = col
    segments.append((start, prev))
    assert len(segments) == 1, "expected one contiguous top-row palette"

    c0, c1 = segments[0]
    palette = [int(v) for v in row[c0 : c1 + 1]]
    assert 2 <= len(palette) <= 4
    assert len(set(palette)) == len(palette), "palette colors must be unique"

    palette_mask = np.zeros(grid.shape, dtype=bool)
    palette_mask[0, c0 : c1 + 1] = True
    return palette, (c0, c1), palette_mask


def analyze_grid(grid: np.ndarray) -> tuple[list[int], list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    palette, _span, palette_mask = extract_palette(g)

    traversable = (g != WALL) & ~palette_mask
    chambers: list[dict] = []
    for comp in components_of_mask(traversable):
        touches_border = any(r == 0 or c == 0 or r == g.shape[0] - 1 or c == g.shape[1] - 1 for r, c in comp)
        if touches_border:
            continue
        markers = [(r, c) for r, c in comp if int(g[r, c]) not in (BG, WALL)]
        assert len(markers) == 1, "each enclosed chamber must contain exactly one marker"
        marker = markers[0]
        chambers.append(
            {
                "cells": comp,
                "marker": marker,
                "color": int(g[marker]),
            }
        )
    return palette, chambers, palette_mask


def render_output(grid: np.ndarray, palette: list[int], chambers: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    index_by_color = {color: i for i, color in enumerate(palette)}
    plen = len(palette)
    for chamber in chambers:
        mr, mc = chamber["marker"]
        offset = index_by_color.get(chamber["color"], 0)
        for r, c in chamber["cells"]:
            if (r, c) == (mr, mc):
                continue
            dist = abs(r - mr) + abs(c - mc)
            out[r, c] = palette[(dist + offset) % plen]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each enclosed wall chamber with Manhattan-distance rings whose phase is set by the chamber marker.

    Input structure:
    - The grid is mostly black background with grey wall pixels that enclose several irregular chambers.
    - The top row contains one contiguous palette segment of 2-4 unique non-black, non-grey colors.
    - Each enclosed non-wall chamber contains exactly one colored marker pixel; the marker may or may not reuse a palette color.

    Rule:
    - Treat grey cells as walls and ignore the top-row palette cells when partitioning the remaining non-wall area into 4-connected components.
    - Keep any component touching the grid border unchanged; for every enclosed component, locate its unique marker.
    - For each non-marker cell in that chamber, compute Manhattan distance to the marker and color it with `palette[(distance + offset) mod len(palette)]`, where `offset` is the marker color's index in the palette or `0` if the marker color is absent from the palette.
    - Preserve the marker itself, the palette row, the walls, and the open border-connected background.

    Key insight:
    - The marker pixel has three linked roles at once: it identifies which enclosed chamber to fill, it provides the Manhattan-distance origin, and its visible color selects the cyclic palette phase.
    """
    palette, chambers, _palette_mask = analyze_grid(np.asarray(grid, dtype=int))
    return render_output(grid, palette, chambers)


def grow_blob(shape: tuple[int, int], target: int, blocked: np.ndarray, rng: random.Random) -> set[tuple[int, int]] | None:
    h, w = shape
    seeds = [
        (r, c)
        for r in range(2, h - 2)
        for c in range(2, w - 2)
        if not blocked[r, c]
    ]
    rng.shuffle(seeds)
    for seed in seeds[:80]:
        cells = {seed}
        frontier = {pos for pos in neighbors4(*seed) if 2 <= pos[0] < h - 2 and 2 <= pos[1] < w - 2 and not blocked[pos]}
        while frontier and len(cells) < target:
            weighted: list[tuple[int, int]] = []
            for pos in frontier:
                shared = sum((nbr in cells) for nbr in neighbors4(*pos))
                weighted.extend([pos] * (1 + 2 * shared))
            pick = rng.choice(weighted)
            frontier.remove(pick)
            cells.add(pick)
            for nbr in neighbors4(*pick):
                nr, nc = nbr
                if 2 <= nr < h - 2 and 2 <= nc < w - 2 and not blocked[nr, nc] and nbr not in cells:
                    frontier.add(nbr)
        if len(cells) != target:
            continue
        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
        bbox_area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
        if target >= 10 and bbox_area == target:
            continue
        return cells
    return None


def wall_ring(cells: set[tuple[int, int]], interior_mask: np.ndarray) -> set[tuple[int, int]]:
    h, w = interior_mask.shape
    ring: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not interior_mask[nr, nc]:
                ring.add((nr, nc))
    return ring


def shortest_wall_path(
    wall_mask: np.ndarray,
    interior_mask: np.ndarray,
    starts: set[tuple[int, int]],
    goals: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = wall_mask.shape
    allowed = ~interior_mask
    allowed[0, :] = False
    q = deque()
    prev: dict[tuple[int, int], tuple[int, int] | None] = {}
    for pos in starts:
        q.append(pos)
        prev[pos] = None

    while q:
        r, c = q.popleft()
        if (r, c) in goals:
            path = []
            cur: tuple[int, int] | None = (r, c)
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return path
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not allowed[nr, nc]:
                continue
            nxt = (nr, nc)
            if nxt in prev:
                continue
            prev[nxt] = (r, c)
            q.append(nxt)
    return None


def connect_wall_network(wall_mask: np.ndarray, interior_mask: np.ndarray, rng: random.Random) -> np.ndarray:
    mask = wall_mask.copy()
    while True:
        comps = components_of_mask(mask)
        if len(comps) <= 1:
            break
        comps = sorted(comps, key=len, reverse=True)
        merged = set(comps[0])
        target_comp = set(rng.choice(comps[1:]))
        path = shortest_wall_path(mask, interior_mask, merged, target_comp)
        if path is None:
            return wall_mask
        for r, c in path:
            mask[r, c] = True
    if not np.any(mask[:, 0]) and not np.any(mask[:, -1]) and not np.any(mask[-1, :]) and rng.random() < 0.6:
        border_targets = {
            (r, 0) for r in range(1, mask.shape[0])
        } | {
            (r, mask.shape[1] - 1) for r in range(1, mask.shape[0])
        } | {
            (mask.shape[0] - 1, c) for c in range(mask.shape[1])
        }
        path = shortest_wall_path(mask, interior_mask, set(map(tuple, np.argwhere(mask))), border_targets)
        if path is not None:
            for r, c in path:
                mask[r, c] = True
    return mask


def choose_marker_colors(palette: list[int], chamber_count: int, rng: random.Random) -> list[int]:
    chosen: list[int] = []
    palette_copy = palette[:]
    rng.shuffle(palette_copy)
    for color in palette_copy[: min(2, chamber_count)]:
        chosen.append(color)

    non_palette = [c for c in MARKER_COLORS if c not in palette and c not in chosen]
    if chamber_count > len(chosen) and non_palette and (chamber_count > len(palette) or rng.random() < 0.55):
        chosen.append(rng.choice(non_palette))

    pool = [c for c in MARKER_COLORS if c not in chosen]
    rng.shuffle(pool)
    while len(chosen) < chamber_count:
        chosen.append(pool.pop())
    rng.shuffle(chosen)
    return chosen


def choose_marker_cell(cells: set[tuple[int, int]], rng: random.Random) -> tuple[int, int]:
    core = []
    all_cells = list(cells)
    for r, c in all_cells:
        degree = sum((nbr in cells) for nbr in neighbors4(r, c))
        if degree >= 3:
            core.append((r, c))
    return rng.choice(core or all_cells)


def sample_layout(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 18)
    w = rng.randint(14, 18)
    chamber_count = rng.randint(3, 5)
    palette_len = rng.randint(2, 4)
    palette = rng.sample(MARKER_COLORS, palette_len)

    interior_mask = np.zeros((h, w), dtype=bool)
    blocked = np.zeros((h, w), dtype=bool)
    blobs: list[set[tuple[int, int]]] = []
    sizes = [rng.randint(8, 25) for _ in range(chamber_count)]
    sizes.sort(reverse=True)

    for size in sizes:
        blob = grow_blob((h, w), size, blocked, rng)
        if blob is None:
            return None
        blobs.append(blob)
        for r, c in blob:
            interior_mask[r, c] = True
        reserved = dilate_cells(blob, (h, w), 2)
        for r, c in reserved:
            blocked[r, c] = True

    wall_mask = np.zeros((h, w), dtype=bool)
    for blob in blobs:
        for r, c in wall_ring(blob, interior_mask):
            wall_mask[r, c] = True
    wall_mask = connect_wall_network(wall_mask, interior_mask, rng)

    grid = np.full((h, w), BG, dtype=int)
    grid[wall_mask] = WALL

    start = rng.randint(1, w - palette_len - 1)
    grid[0, start : start + palette_len] = np.array(palette, dtype=int)

    marker_colors = choose_marker_colors(palette, chamber_count, rng)
    for blob, color in zip(blobs, marker_colors):
        marker = choose_marker_cell(blob, rng)
        grid[marker] = color

    if np.count_nonzero(grid[0] != BG) != palette_len:
        return None
    return grid


def validate_instance(grid: np.ndarray) -> bool:
    try:
        assert isinstance(grid, np.ndarray)
        assert grid.ndim == 2
        assert max(grid.shape) <= 30
        assert get_bg(grid) == BG

        palette, chambers, palette_mask = analyze_grid(grid)
        assert 3 <= len(chambers) <= 5
        assert 2 <= len(palette) <= 4
        assert np.all(grid[palette_mask] != BG)
        assert np.all(grid[palette_mask] != WALL)
        assert len(set(palette)) == len(palette)

        areas = [len(ch["cells"]) for ch in chambers]
        assert all(8 <= area <= 25 for area in areas)

        marker_colors = [ch["color"] for ch in chambers]
        assert len(set(marker_colors)) == len(marker_colors)

        palette_index = {color: i for i, color in enumerate(palette)}
        offsets = [palette_index.get(color, 0) for color in marker_colors]
        assert len(set(offsets)) >= 2
        assert any(color in palette_index for color in marker_colors)

        expected = render_output(grid, palette, chambers)
        actual = main(grid)
        assert np.array_equal(actual, expected)
        assert not np.array_equal(actual, grid)

        border_connected = components_of_mask(((grid != WALL) & ~palette_mask))
        assert any(any(r == 0 or c == 0 or r == grid.shape[0] - 1 or c == grid.shape[1] - 1 for r, c in comp) for comp in border_connected)

        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random(random.randrange(1 << 30))
        grid = sample_layout(rng)
        if grid is not None and validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid chamber-ring instance")
