from __future__ import annotations

import random

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE = list(range(10))
SOURCE_SIZE = 3
MIN_MARGIN = 4
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[counts.argmax()])


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    mask = grid == int(color)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _is_solid_rect(cells: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    return len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _centroid(cells: list[tuple[int, int]]) -> tuple[float, float]:
    n = float(len(cells))
    return sum(r for r, _ in cells) / n, sum(c for _, c in cells) / n


def _dist_point_to_rect(r: int, c: int, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    dr = max(r0 - r, 0, r - r1)
    dc = max(c0 - c, 0, c - c1)
    return int(dr + dc)


def _dist_comp_to_source(cells: list[tuple[int, int]], source: dict) -> int:
    return min(_dist_point_to_rect(r, c, source["bbox"]) for r, c in cells)


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = sorted(int(v) for v in np.unique(g) if int(v) != bg)

    sources: list[dict] = []
    objects: list[dict] = []
    for color in colors:
        for comp in _components_of_color(g, color):
            r0, c0, r1, c1 = _bbox(comp)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if len(comp) >= 9 and min(height, width) >= 3 and _is_solid_rect(comp):
                sources.append(
                    {
                        "color": int(color),
                        "cells": comp,
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
                    }
                )
            else:
                assert 1 <= len(comp) <= 3, "free objects must be 1-3 pixels"
                objects.append({"color": int(color), "cells": comp})

    assert sources, "grid must contain at least one gravity source"
    return bg, sources, objects


def _assign_source(obj: dict, sources: list[dict]) -> tuple[dict, int]:
    same_color = [src for src in sources if src["color"] == obj["color"]]
    candidates = same_color if same_color else sources
    ranked = sorted(
        candidates,
        key=lambda src: (
            _dist_comp_to_source(obj["cells"], src),
            src["bbox"][0],
            src["bbox"][1],
            src["color"],
        ),
    )
    source = ranked[0]
    return source, _dist_comp_to_source(obj["cells"], source)


def _move_step(obj: dict, source: dict) -> tuple[int, int]:
    orow, ocol = _centroid(obj["cells"])
    srow, scol = source["center"]
    dr = orow - srow
    dc = ocol - scol
    use_vertical = abs(dr) > abs(dc) or (abs(dr) == abs(dc) and dr != 0)
    if use_vertical:
        return (1, 0) if dr < 0 else (-1, 0)
    return (0, 1) if dc < 0 else (0, -1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull small floating objects straight toward their nearest gravity source and stack them against its edge.

    Input structure:
    - bg is the dominant background color.
    - One or more large monochrome solid rectangles act as gravity sources; each source is much larger than the floating objects.
    - Several small monochrome 4-connected objects of size 1-3 share a source color and sit around the sources with empty space between components.

    Rule:
    - Identify each small object and match it to the nearest source of the same color, using Manhattan distance from the object to the source body.
    - For that matched pair, compare the object's row and column offset from the source center; move the whole object along the axis with the larger offset toward the source.
    - Process objects from near to far for each source, translating each one step-by-step until the next step would hit the source or an already stacked object.
    - Erase the object's original location to background and leave the translated object at its final stacked position.

    Key insight:
    - The motion is not along the shortest diagonal route: each object behaves like a rigid rook move that first resolves the larger coordinate difference, so stack order matters.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    bg, sources, objects = _parse_scene(g)

    out = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    for source in sources:
        for r, c in source["cells"]:
            out[r, c] = source["color"]
            occupied[r, c] = True

    moves = []
    for obj in objects:
        source, dist = _assign_source(obj, sources)
        moves.append(
            (
                int(dist),
                int(source["bbox"][0]),
                int(source["bbox"][1]),
                min(obj["cells"]),
                obj,
                _move_step(obj, source),
            )
        )

    moves.sort(key=lambda item: item[:4])

    for _dist, _sr, _sc, _anchor, obj, (dr, dc) in moves:
        current = list(obj["cells"])
        while True:
            shifted = [(r + dr, c + dc) for r, c in current]
            blocked = False
            for nr, nc in shifted:
                if not (0 <= nr < h and 0 <= nc < w) or occupied[nr, nc]:
                    blocked = True
                    break
            if blocked:
                break
            current = shifted

        for r, c in current:
            out[r, c] = obj["color"]
            occupied[r, c] = True

    return out


def _sample_source_positions(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int]] = []
    for _ in range(200):
        if len(positions) == count:
            break
        cand = (
            random.randint(MIN_MARGIN, h - SOURCE_SIZE - MIN_MARGIN),
            random.randint(MIN_MARGIN, w - SOURCE_SIZE - MIN_MARGIN),
        )
        if all(abs(cand[0] - r0) >= 10 or abs(cand[1] - c0) >= 10 for r0, c0 in positions):
            positions.append(cand)
    return positions if len(positions) == count else None


def _cells_for_group(
    source: dict,
    side: str,
    length: int,
    offset: int,
    distance: int,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = source["bbox"]
    if side == "up":
        row = r0 - distance
        return [(row, c0 + offset + dc) for dc in range(length)]
    if side == "down":
        row = r1 + distance
        return [(row, c0 + offset + dc) for dc in range(length)]
    if side == "left":
        col = c0 - distance
        return [(r0 + offset + dr, col) for dr in range(length)]
    col = c1 + distance
    return [(r0 + offset + dr, col) for dr in range(length)]


def _can_place_cells(grid: np.ndarray, bg: int, cells: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != bg:
            return False
    for r, c in cells:
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cell_set and int(grid[nr, nc]) != bg:
                return False
    return True


def _sample_distances(count: int, available: int) -> list[int] | None:
    if available < 2 * count:
        return None
    if count == 1:
        return [random.randint(2, available)]
    first = random.randint(2, available - 2)
    second = random.randint(first + 2, available)
    return [first, second]


def _place_group(
    grid: np.ndarray,
    bg: int,
    source: dict,
    side: str,
    count: int,
) -> bool:
    r0, c0, r1, c1 = source["bbox"]
    if side in {"up", "down"}:
        available = r0 if side == "up" else grid.shape[0] - 1 - r1
    else:
        available = c0 if side == "left" else grid.shape[1] - 1 - c1

    distances = _sample_distances(count, available)
    if distances is None:
        return False

    length = random.randint(1, SOURCE_SIZE)
    offset = random.randint(0, SOURCE_SIZE - length)
    for distance in distances:
        cells = _cells_for_group(source, side, length, offset, distance)
        if not _can_place_cells(grid, bg, cells):
            return False
        for r, c in cells:
            grid[r, c] = source["color"]
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 28)
    w = random.randint(18, 28)
    bg = random.choice(PALETTE)

    n_sources = random.choice([1, 2])
    positions = _sample_source_positions(h, w, n_sources)
    if positions is None:
        return None

    source_colors = random.sample([c for c in PALETTE if c != bg], n_sources)
    grid = np.full((h, w), bg, dtype=int)

    sources: list[dict] = []
    for (r0, c0), color in zip(positions, source_colors):
        cells = [(r, c) for r in range(r0, r0 + SOURCE_SIZE) for c in range(c0, c0 + SOURCE_SIZE)]
        for r, c in cells:
            grid[r, c] = int(color)
        sources.append(
            {
                "color": int(color),
                "cells": cells,
                "bbox": (int(r0), int(c0), int(r0 + SOURCE_SIZE - 1), int(c0 + SOURCE_SIZE - 1)),
                "center": (float(r0 + 1), float(c0 + 1)),
            }
        )

    counts = [random.choice([1, 2]) for _ in range(n_sources)]
    if all(count == 1 for count in counts):
        counts[random.randrange(n_sources)] = 2

    for source, count in zip(sources, counts):
        vertical_side = random.choice(["up", "down"])
        horizontal_side = random.choice(["left", "right"])
        if not _place_group(grid, bg, source, vertical_side, count):
            return None
        if not _place_group(grid, bg, source, horizontal_side, count):
            return None

    try:
        _bg, parsed_sources, parsed_objects = _parse_scene(grid)
        if _bg != bg:
            return None
        if len(parsed_sources) != n_sources:
            return None
        if len(parsed_objects) != 2 * sum(counts):
            return None
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
