import random
from collections import Counter, deque

import numpy as np


GREY = 5
GRID_MIN = 18
GRID_MAX = 28
SOURCE_COUNT_MIN = 2
SOURCE_COUNT_MAX = 5
LINE_COUNT_MIN = 2
LINE_COUNT_MAX = 5
LINE_LENGTH_MIN = 2
LINE_LENGTH_MAX = 6
GAP_MIN = 1
GAP_MAX = 5
EDGE_MARGIN = 2
MAX_GENERATE_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if (
                        _in_bounds(grid.shape, nr, nc)
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_sources(grid: np.ndarray, bg: int) -> list[dict]:
    sources: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid)))):
        if color in (bg, GREY):
            continue
        for comp in _components_of_color(grid, color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if len(comp) != 9 or r1 - r0 != 2 or c1 - c0 != 2:
                continue
            if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
                cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
                sources.append(
                    {
                        "color": color,
                        "r0": r0,
                        "c0": c0,
                        "r1": r1,
                        "c1": c1,
                        "cells": cells,
                    }
                )
    return sources


def _find_line_components(grid: np.ndarray) -> list[dict]:
    comps = []
    for comp in _components_of_color(grid, GREY):
        rows = sorted({r for r, _ in comp})
        cols = sorted({c for _, c in comp})
        if len(rows) == 1:
            r = rows[0]
            c0, c1 = min(cols), max(cols)
            if len(comp) != c1 - c0 + 1:
                continue
            cells = [(r, c) for c in range(c0, c1 + 1)]
            comps.append(
                {
                    "orientation": "h",
                    "cells": cells,
                    "endpoints": [((r, c0), (0, -1)), ((r, c1), (0, 1))],
                }
            )
        elif len(cols) == 1:
            c = cols[0]
            r0, r1 = min(rows), max(rows)
            if len(comp) != r1 - r0 + 1:
                continue
            cells = [(r, c) for r in range(r0, r1 + 1)]
            comps.append(
                {
                    "orientation": "v",
                    "cells": cells,
                    "endpoints": [((r0, c), (-1, 0)), ((r1, c), (1, 0))],
                }
            )
    return comps


def _scan_from_endpoint(
    grid: np.ndarray,
    bg: int,
    source_lookup: dict[tuple[int, int], int],
    start: tuple[int, int],
    delta: tuple[int, int],
) -> tuple[int, list[tuple[int, int]], int] | None:
    r, c = start[0] + delta[0], start[1] + delta[1]
    path: list[tuple[int, int]] = []
    while _in_bounds(grid.shape, r, c):
        if (r, c) in source_lookup:
            return source_lookup[(r, c)], path, len(path) + 1
        if int(grid[r, c]) != bg:
            return None
        path.append((r, c))
        r += delta[0]
        c += delta[1]
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Extend each grey line toward the first source square on its pointing side, then recolor it."""
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    sources = _find_sources(g, bg)
    source_lookup = {cell: source["color"] for source in sources for cell in source["cells"]}

    out = g.copy()
    for line in _find_line_components(g):
        candidates = []
        for endpoint, delta in line["endpoints"]:
            hit = _scan_from_endpoint(g, bg, source_lookup, endpoint, delta)
            if hit is not None:
                candidates.append(hit)
        assert candidates, "every grey line should point toward at least one source"

        color, extension, _distance = min(candidates, key=lambda item: item[2])
        for r, c in line["cells"]:
            out[r, c] = color
        for r, c in extension:
            out[r, c] = color

    return out


def _choose_colors(n_sources: int) -> tuple[int, list[int]]:
    palette = [c for c in range(10) if c != GREY]
    bg = random.choice(palette)
    source_colors = [c for c in palette if c != bg]
    random.shuffle(source_colors)
    return bg, source_colors[:n_sources]


def _source_can_fit(grid: np.ndarray, r0: int, c0: int, bg: int) -> bool:
    r1, c1 = r0 + 2, c0 + 2
    rr0, rr1 = max(0, r0 - 1), min(grid.shape[0] - 1, r1 + 1)
    cc0, cc1 = max(0, c0 - 1), min(grid.shape[1] - 1, c1 + 1)
    return bool(np.all(grid[rr0 : rr1 + 1, cc0 : cc1 + 1] == bg))


def _place_sources(grid: np.ndarray, colors: list[int]) -> list[dict] | None:
    h, w = grid.shape
    sources: list[dict] = []
    for color in colors:
        placed = False
        for _ in range(80):
            r0 = random.randint(EDGE_MARGIN, h - EDGE_MARGIN - 3)
            c0 = random.randint(EDGE_MARGIN, w - EDGE_MARGIN - 3)
            if not _source_can_fit(grid, r0, c0, int(grid[0, 0])):
                continue
            grid[r0 : r0 + 3, c0 : c0 + 3] = color
            cells = {(r, c) for r in range(r0, r0 + 3) for c in range(c0, c0 + 3)}
            sources.append(
                {
                    "color": color,
                    "r0": r0,
                    "c0": c0,
                    "r1": r0 + 2,
                    "c1": c0 + 2,
                    "cells": cells,
                }
            )
            placed = True
            break
        if not placed:
            return None
    return sources


def _build_candidate(
    source: dict,
    orientation: str,
    side: str,
    offset: int,
    gap: int,
    length: int,
    shape: tuple[int, int],
) -> dict | None:
    h, w = shape
    r0, c0, r1, c1 = source["r0"], source["c0"], source["r1"], source["c1"]

    if orientation == "h":
        row = r0 + offset
        if side == "left":
            near_c = c0 - gap
            far_c = near_c - length + 1
            if far_c < 0:
                return None
            corridor = [(row, c) for c in range(far_c, c0)]
            line_cells = [(row, c) for c in range(far_c, near_c + 1)]
            far_endpoint = (row, far_c)
            far_delta = (0, -1)
        else:
            near_c = c1 + gap
            far_c = near_c + length - 1
            if far_c >= w:
                return None
            corridor = [(row, c) for c in range(c1 + 1, far_c + 1)]
            line_cells = [(row, c) for c in range(near_c, far_c + 1)]
            far_endpoint = (row, far_c)
            far_delta = (0, 1)
    else:
        col = c0 + offset
        if side == "top":
            near_r = r0 - gap
            far_r = near_r - length + 1
            if far_r < 0:
                return None
            corridor = [(r, col) for r in range(far_r, r0)]
            line_cells = [(r, col) for r in range(far_r, near_r + 1)]
            far_endpoint = (far_r, col)
            far_delta = (-1, 0)
        else:
            near_r = r1 + gap
            far_r = near_r + length - 1
            if far_r >= h:
                return None
            corridor = [(r, col) for r in range(r1 + 1, far_r + 1)]
            line_cells = [(r, col) for r in range(near_r, far_r + 1)]
            far_endpoint = (far_r, col)
            far_delta = (1, 0)

    extension = [cell for cell in corridor if cell not in set(line_cells)]
    return {
        "source_color": source["color"],
        "source_cells": set(source["cells"]),
        "corridor": corridor,
        "line_cells": line_cells,
        "extension": extension,
        "far_endpoint": far_endpoint,
        "far_delta": far_delta,
    }


def _candidate_is_clear(grid: np.ndarray, bg: int, candidate: dict) -> bool:
    corridor = set(candidate["corridor"])
    source_cells = candidate["source_cells"]

    for r, c in corridor:
        if int(grid[r, c]) != bg:
            return False

    for r, c in corridor:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if not _in_bounds(grid.shape, nr, nc):
                    continue
                if (nr, nc) in corridor or (nr, nc) in source_cells:
                    continue
                if int(grid[nr, nc]) != bg:
                    return False

    r, c = candidate["far_endpoint"]
    dr, dc = candidate["far_delta"]
    r += dr
    c += dc
    while _in_bounds(grid.shape, r, c):
        if int(grid[r, c]) != bg:
            return False
        r += dr
        c += dc

    return True


def _orientation_plan(n_lines: int) -> list[str]:
    plan = ["h", "v"] * (n_lines // 2)
    if n_lines % 2 == 1:
        plan.append(random.choice(["h", "v"]))
    random.shuffle(plan)
    return plan


def _place_lines(grid: np.ndarray, bg: int, sources: list[dict], n_lines: int) -> list[dict] | None:
    line_records: list[dict] = []
    used_ports: set[tuple[int, str, str, int]] = set()

    for orientation in _orientation_plan(n_lines):
        ports = []
        for idx, _source in enumerate(sources):
            if orientation == "h":
                for side in ("left", "right"):
                    for offset in range(3):
                        ports.append((idx, side, offset))
            else:
                for side in ("top", "bottom"):
                    for offset in range(3):
                        ports.append((idx, side, offset))
        random.shuffle(ports)

        placed = False
        for idx, side, offset in ports:
            port_key = (idx, orientation, side, offset)
            if port_key in used_ports:
                continue

            gaps = list(range(GAP_MIN, GAP_MAX + 1))
            lengths = list(range(LINE_LENGTH_MIN, LINE_LENGTH_MAX + 1))
            random.shuffle(gaps)
            random.shuffle(lengths)
            for gap in gaps:
                for length in lengths:
                    candidate = _build_candidate(
                        sources[idx],
                        orientation,
                        side,
                        offset,
                        gap,
                        length,
                        grid.shape,
                    )
                    if candidate is None or not _candidate_is_clear(grid, bg, candidate):
                        continue
                    for r, c in candidate["line_cells"]:
                        grid[r, c] = GREY
                    line_records.append(candidate)
                    used_ports.add(port_key)
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    return line_records


def _validate_generated_scene(
    grid: np.ndarray,
    bg: int,
    sources: list[dict],
    line_records: list[dict],
) -> bool:
    if _most_common_color(grid) != bg:
        return False

    found_sources = _find_sources(grid, bg)
    if len(found_sources) != len(sources):
        return False

    found_lines = _find_line_components(grid)
    if len(found_lines) != len(line_records):
        return False

    source_lookup = {cell: source["color"] for source in found_sources for cell in source["cells"]}
    for line in found_lines:
        hits = []
        for endpoint, delta in line["endpoints"]:
            hit = _scan_from_endpoint(grid, bg, source_lookup, endpoint, delta)
            if hit is not None:
                hits.append(hit)
        if not hits:
            return False
        best_distance = min(hit[2] for hit in hits)
        if sum(hit[2] == best_distance for hit in hits) != 1:
            return False

    out = main(grid)
    for record in line_records:
        for r, c in record["line_cells"]:
            if int(out[r, c]) != record["source_color"]:
                return False
        for r, c in record["extension"]:
            if int(out[r, c]) != record["source_color"]:
                return False
        for r, c in record["source_cells"]:
            if int(out[r, c]) != record["source_color"]:
                return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    n_sources = random.randint(SOURCE_COUNT_MIN, SOURCE_COUNT_MAX)
    max_lines = min(LINE_COUNT_MAX, n_sources + 1)
    n_lines = random.randint(LINE_COUNT_MIN, max_lines)

    bg, source_colors = _choose_colors(n_sources)
    grid = np.full((h, w), bg, dtype=int)

    sources = _place_sources(grid, source_colors)
    if sources is None:
        return None

    line_records = _place_lines(grid, bg, sources, n_lines)
    if line_records is None:
        return None

    if not _validate_generated_scene(grid, bg, sources, line_records):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATE_ATTEMPTS} attempts")
