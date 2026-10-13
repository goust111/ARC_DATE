from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
MAROON = 9
MARKER_COLORS = np.array([1, 2, 3, 4, 6, 7, 8], dtype=int)

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# Offsets are expressed relative to the arrow-tip pixel.
ARROW_OFFSETS = {
    "up": ((0, 0), (1, 0), (2, 0), (3, 0), (1, -1), (1, 1)),
    "down": ((0, 0), (-1, 0), (-2, 0), (-3, 0), (-1, -1), (-1, 1)),
    "left": ((0, 0), (0, 1), (0, 2), (0, 3), (-1, 1), (1, 1)),
    "right": ((0, 0), (0, -1), (0, -2), (0, -3), (-1, -1), (1, -1)),
}

OFFSETS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)

GEN_H_RANGE = (12, 21)
GEN_W_RANGE = (12, 21)
GEN_MIN_PATH_CELLS = 4
GEN_MAX_ARROWS = 3
GEN_MAX_ATTEMPTS = 400


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in OFFSETS8:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _translate(offsets: tuple[tuple[int, int], ...], origin: tuple[int, int]) -> set[tuple[int, int]]:
    orow, ocol = origin
    return {(orow + dr, ocol + dc) for dr, dc in offsets}


def _detect_arrow(comp: list[tuple[int, int]]) -> tuple[str, tuple[int, int]] | None:
    pts = set(comp)
    matches: list[tuple[str, tuple[int, int]]] = []
    for tip in pts:
        for direction, offsets in ARROW_OFFSETS.items():
            if _translate(offsets, tip) == pts:
                matches.append((direction, tip))
    if len(matches) != 1:
        return None
    return matches[0]


def _find_arrows(grid: np.ndarray) -> list[dict[str, object]]:
    arrows: list[dict[str, object]] = []
    for comp in _components8(grid == GREY):
        if len(comp) != 6:
            continue
        detected = _detect_arrow(comp)
        if detected is None:
            continue
        direction, tip = detected
        arrows.append(
            {
                "direction": direction,
                "tip": tip,
                "cells": tuple(sorted(comp)),
            }
        )
    arrows.sort(key=lambda item: (item["tip"][0], item["tip"][1], item["direction"]))
    return arrows


def _forward_line(h: int, w: int, tip: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    dr, dc = DIRS[direction]
    r, c = tip[0] + dr, tip[1] + dc
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    for arrow in _find_arrows(g):
        direction = str(arrow["direction"])
        tip = tuple(arrow["tip"])
        current_color = GREY
        markers: list[tuple[int, int]] = []
        endpoint: tuple[int, int] | None = None

        for r, c in _forward_line(h, w, tip, direction):
            value = int(g[r, c])
            if value == MAROON:
                endpoint = (r, c)
                break
            if value not in (BG, GREY):
                current_color = value
                markers.append((r, c))

        if endpoint is None:
            continue

        for r, c in markers:
            out[r, c] = BG
        out[endpoint] = current_color

    return out


def _expanded_footprint(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    expanded = set(cells)
    for r, c in list(cells):
        for dr, dc in OFFSETS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                expanded.add((nr, nc))
    return expanded


def _enumerate_candidates(h: int, w: int, blocked: set[tuple[int, int]]) -> dict[str, list[dict[str, object]]]:
    by_direction: dict[str, list[dict[str, object]]] = {name: [] for name in DIRS}
    for direction, offsets in ARROW_OFFSETS.items():
        for r in range(h):
            for c in range(w):
                tip = (r, c)
                arrow_cells = _translate(offsets, tip)
                if not all(0 <= rr < h and 0 <= cc < w for rr, cc in arrow_cells):
                    continue
                path = _forward_line(h, w, tip, direction)
                if len(path) < GEN_MIN_PATH_CELLS:
                    continue
                footprint = set(arrow_cells) | set(path)
                if footprint & blocked:
                    continue
                by_direction[direction].append(
                    {
                        "direction": direction,
                        "tip": tip,
                        "arrow_cells": tuple(sorted(arrow_cells)),
                        "path": tuple(path),
                        "footprint": footprint,
                    }
                )
    return by_direction


def _sample_spaced_indices(limit: int, count: int, rng: np.random.Generator) -> list[int]:
    if count <= 0:
        return []
    for _ in range(80):
        picked = sorted(int(v) for v in rng.choice(limit, size=count, replace=False))
        if all(b - a > 1 for a, b in zip(picked, picked[1:])):
            return picked
    fallback = list(range(0, limit, 2))
    if len(fallback) < count:
        fallback = list(range(1, limit, 2))
    return fallback[:count]


def _sample_marker_indices(path_len: int, rng: np.random.Generator) -> list[int]:
    pre_endpoint = path_len - 1
    if pre_endpoint <= 0:
        return []

    max_markers = min(4, (pre_endpoint + 1) // 2)
    branch = int(rng.integers(0, 3))
    if branch == 0 or max_markers == 0:
        count = 0
    elif branch == 1:
        count = 1
    else:
        count = 1 if max_markers == 1 else int(rng.integers(2, max_markers + 1))

    count = min(count, max_markers)
    return _sample_spaced_indices(pre_endpoint, count, rng)


def _target_arrow_count(h: int, w: int, rng: np.random.Generator) -> int:
    area = h * w
    if area < 180:
        lo, hi = 1, 2
    elif area < 280:
        lo, hi = 1, 3
    else:
        lo, hi = 2, 3
    hi = min(hi, GEN_MAX_ARROWS)
    return int(rng.integers(lo, hi + 1))


def _validate_generated(grid: np.ndarray, specs: list[dict[str, object]]) -> bool:
    arrows = _find_arrows(grid)
    if len(arrows) != len(specs):
        return False

    found = sorted((item["direction"], item["tip"]) for item in arrows)
    expected = sorted((item["direction"], item["tip"]) for item in specs)
    if found != expected:
        return False

    out = main(grid)
    for spec in specs:
        path = list(spec["path"])
        markers = list(spec["markers"])
        endpoint = path[-1]

        if sum(int(grid[r, c] == MAROON) for r, c in path) != 1:
            return False
        if int(grid[endpoint]) != MAROON:
            return False
        if any(int(grid[r, c]) in (GREY, MAROON) for r, c in path[:-1]):
            return False

        final_color = int(spec["final_color"])
        if int(out[endpoint]) != final_color:
            return False
        if any(int(out[r, c]) != BG for r, c in markers):
            return False
        if any(int(out[r, c]) != GREY for r, c in spec["arrow_cells"]):
            return False

    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GEN_H_RANGE))
    w = int(rng.integers(*GEN_W_RANGE))
    grid = np.full((h, w), BG, dtype=int)

    blocked: set[tuple[int, int]] = set()
    specs: list[dict[str, object]] = []
    arrow_count = _target_arrow_count(h, w, rng)

    for _ in range(arrow_count):
        by_direction = _enumerate_candidates(h, w, blocked)
        available_dirs = [name for name, items in by_direction.items() if items]
        if not available_dirs:
            return None

        direction = str(rng.choice(np.array(available_dirs, dtype=object)))
        spec = by_direction[direction][int(rng.integers(0, len(by_direction[direction])))]
        path = list(spec["path"])
        marker_indices = _sample_marker_indices(len(path), rng)
        marker_cells = [path[idx] for idx in marker_indices]
        marker_count = len(marker_cells)

        if marker_count:
            marker_colors = [int(v) for v in rng.choice(MARKER_COLORS, size=marker_count, replace=False)]
        else:
            marker_colors = []

        for r, c in spec["arrow_cells"]:
            grid[r, c] = GREY
        for (r, c), color in zip(marker_cells, marker_colors):
            grid[r, c] = color
        endpoint = path[-1]
        grid[endpoint] = MAROON

        specs.append(
            {
                "direction": direction,
                "tip": spec["tip"],
                "arrow_cells": spec["arrow_cells"],
                "path": spec["path"],
                "markers": tuple(marker_cells),
                "final_color": marker_colors[-1] if marker_colors else GREY,
            }
        )
        blocked |= _expanded_footprint(set(spec["footprint"]), h, w)

    if not _validate_generated(grid, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
