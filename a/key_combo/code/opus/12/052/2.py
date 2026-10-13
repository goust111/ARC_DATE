from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BACKGROUND = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_MIN = 13
GRID_MAX = 18
MIN_LINE_LENGTH = 4
MAX_GENERATION_ATTEMPTS = 1500
LAYOUT_ATTEMPTS = 250
MARKER_ATTEMPTS = 80


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _diag_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    assert dr != 0 and abs(dr) == abs(dc), "endpoints must define a strict 45-degree segment"
    step_r = _sign(dr)
    step_c = _sign(dc)
    return [(r0 + k * step_r, c0 + k * step_c) for k in range(abs(dr) + 1)]


def _line_and_marker_roles(grid: np.ndarray) -> tuple[list[int], list[int]]:
    counts = Counter(map(int, grid.ravel()))
    line_colors = sorted(color for color, count in counts.items() if color != BACKGROUND and count == 2)
    marker_colors = sorted(color for color, count in counts.items() if color != BACKGROUND and count == 1)
    assert 3 <= len(line_colors) <= 4, "expected 3-4 line colors"
    assert 3 <= len(marker_colors) <= 5, "expected 3-5 marker colors"
    return line_colors, marker_colors


def _draw_all_lines(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    line_colors, _marker_colors = _line_and_marker_roles(g)
    items: list[dict[str, object]] = []
    for color in line_colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        assert len(coords) == 2, "each line color must appear exactly twice"
        endpoints = tuple(sorted(coords))
        cells = _diag_cells(endpoints[0], endpoints[1])
        items.append({"color": int(color), "endpoints": endpoints, "first": endpoints[0], "cells": cells})
    items.sort(key=lambda item: item["first"])

    out = g.copy()
    for item in items:
        color = int(item["color"])
        for r, c in item["cells"]:
            out[r, c] = color
    return out, items


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify singleton colors as chamber votes and duplicated colors as diagonal endpoints, then fill each chamber by its adjacent vote.

    Input structure:
    - The grid has black background plus several isolated non-black pixels.
    - Some non-black colors appear exactly twice; each such pair is the endpoint pair of one strict 45-degree diagonal segment.
    - Other non-black colors appear exactly once; these singleton pixels are vote markers, distinct from the line colors.

    Rule:
    - Split non-background colors by frequency: count-2 colors are line colors and count-1 colors are marker colors.
    - Draw every diagonal segment between the two pixels of each line color, processing colors by the raster order of their first endpoint; later lines overwrite earlier ones at crossings.
    - Find every maximal 4-connected black chamber after the lines are drawn.
    - For each chamber, collect all orthogonally adjacent marker pixels and fill the whole chamber with the most frequent adjacent marker color; if tied, choose the marker whose pixel is highest, then leftmost.

    Key insight:
    - The same raw colored pixels mix structural and semantic roles, and the hidden separator is not shape or position but whether a color appears once or twice.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    rendered, _line_items = _draw_all_lines(g)
    _line_colors, marker_colors = _line_and_marker_roles(g)
    marker_set = set(marker_colors)
    marker_pixels = {
        tuple(map(int, rc)): int(g[tuple(rc)])
        for rc in np.argwhere(np.isin(g, np.array(marker_colors, dtype=int)))
    }

    out = rendered.copy()
    chambers = _components_mask(rendered == BACKGROUND)
    for chamber in chambers:
        adjacent: dict[tuple[int, int], int] = {}
        for r0, c0 in chamber:
            for r, c in _neighbors4(int(r0), int(c0)):
                if 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
                    color = marker_pixels.get((r, c))
                    if color in marker_set:
                        adjacent[(r, c)] = int(color)

        if not adjacent:
            continue

        counts = Counter(adjacent.values())
        top = max(counts.values())
        winners = {color for color, count in counts.items() if count == top}
        fill = min(
            ((r, c, color) for (r, c), color in adjacent.items() if color in winners),
            key=lambda item: (item[0], item[1]),
        )[2]
        out[chamber[:, 0], chamber[:, 1]] = int(fill)

    return out


def _sample_segment(shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = shape
    max_len = min(h, w) - 3
    assert max_len >= MIN_LINE_LENGTH
    for _ in range(80):
        length = random.randint(MIN_LINE_LENGTH, max_len)
        dr, dc = random.choice([(-1, -1), (-1, 1), (1, -1), (1, 1)])
        r0 = random.randrange(h)
        c0 = random.randrange(w)
        r1 = r0 + dr * length
        c1 = c0 + dc * length
        if not (0 <= r1 < h and 0 <= c1 < w):
            continue
        return tuple(sorted(((r0, c0), (r1, c1))))
    raise RuntimeError("failed to sample a diagonal segment")


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _build_line_layout(
    shape: tuple[int, int], line_count: int, require_crossing: bool
) -> tuple[list[dict[str, object]], np.ndarray] | None:
    h, w = shape
    for _ in range(LAYOUT_ATTEMPTS):
        segments: list[dict[str, object]] = []
        occupancy: dict[tuple[int, int], int] = {}
        endpoint_cells: list[tuple[int, int]] = []
        ok = True
        for _idx in range(line_count):
            placed = False
            for _ in range(120):
                a, b = _sample_segment(shape)
                if _manhattan(a, b) <= 2:
                    continue
                if any(_manhattan(a, p) == 1 or _manhattan(b, p) == 1 for p in endpoint_cells):
                    continue
                cells = _diag_cells(a, b)
                intersections = [cell for cell in cells if cell in occupancy]
                if len(intersections) > 1:
                    continue
                if intersections:
                    cross = intersections[0]
                    if cross in (a, b):
                        continue
                overlap_ok = True
                for seg in segments:
                    other_cells = set(seg["cells"])
                    common = other_cells.intersection(cells)
                    if not common:
                        continue
                    if len(common) > 1:
                        overlap_ok = False
                        break
                    only = next(iter(common))
                    other_endpoints = set(seg["endpoints"])
                    if only in other_endpoints or only in {a, b}:
                        overlap_ok = False
                        break
                if not overlap_ok:
                    continue
                segments.append({"endpoints": (a, b), "cells": cells})
                for cell in cells:
                    occupancy[cell] = occupancy.get(cell, 0) + 1
                endpoint_cells.extend([a, b])
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        crossings = sum(1 for count in occupancy.values() if count > 1)
        if require_crossing and crossings == 0:
            continue

        line_mask = np.zeros((h, w), dtype=bool)
        for seg in segments:
            for r, c in seg["cells"]:
                line_mask[r, c] = True
        chambers = _components_mask(~line_mask)
        if not (3 <= len(chambers) <= 5):
            continue
        if min(len(chamber) for chamber in chambers) < 3:
            continue
        return segments, line_mask
    return None


def _still_connected_after_removals(
    coords: np.ndarray, removed: set[tuple[int, int]]
) -> bool:
    remaining = [tuple(map(int, rc)) for rc in coords if tuple(map(int, rc)) not in removed]
    if not remaining:
        return False
    remaining_set = set(remaining)
    q = deque([remaining[0]])
    seen = {remaining[0]}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in remaining_set and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(remaining_set)


def _marker_candidates(
    chamber: np.ndarray,
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    chosen: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    coords = [tuple(map(int, rc)) for rc in chamber]
    random.shuffle(coords)
    out: list[tuple[int, int]] = []
    for cell in coords:
        r, c = cell
        if r in (0, h - 1) or c in (0, w - 1):
            continue
        if cell in blocked or cell in chosen:
            continue
        if any(_manhattan(cell, other) == 1 for other in blocked | chosen):
            continue
        if _still_connected_after_removals(chamber, {cell}):
            out.append(cell)
    return out


def _pick_markers(
    rendered_without_markers: np.ndarray,
    chambers: list[np.ndarray],
    endpoints: set[tuple[int, int]],
    want_extra: bool,
) -> list[tuple[int, int]] | None:
    shape = rendered_without_markers.shape
    for _ in range(MARKER_ATTEMPTS):
        chosen: list[tuple[int, int]] = []
        chosen_set: set[tuple[int, int]] = set()
        chamber_choices: list[list[tuple[int, int]]] = []
        failed = False
        for chamber in chambers:
            candidates = _marker_candidates(chamber, shape, endpoints, chosen_set)
            if not candidates:
                failed = True
                break
            pick = random.choice(candidates)
            chosen.append(pick)
            chosen_set.add(pick)
            chamber_choices.append(candidates)
        if failed:
            continue

        max_extra = min(5 - len(chambers), len(chambers))
        extra_count = 0
        if want_extra and max_extra > 0:
            extra_count = random.randint(1, max_extra)
        elif max_extra > 0 and random.random() < 0.35:
            extra_count = random.randint(0, max_extra)

        if extra_count:
            chamber_order = list(range(len(chambers)))
            random.shuffle(chamber_order)
            for idx in chamber_order:
                if extra_count == 0:
                    break
                chamber = chambers[idx]
                removed_now = {cell for cell in chosen if cell in set(map(tuple, chamber.tolist()))}
                candidates = [
                    cell
                    for cell in chamber_choices[idx]
                    if cell not in chosen_set
                    and all(_manhattan(cell, other) > 1 for other in chosen_set)
                    and _still_connected_after_removals(chamber, removed_now | {cell})
                ]
                if not candidates:
                    continue
                pick = random.choice(candidates)
                chosen.append(pick)
                chosen_set.add(pick)
                extra_count -= 1
        if not (3 <= len(chosen) <= 5):
            continue

        grid = rendered_without_markers.copy()
        for r, c in chosen:
            grid[r, c] = -1
        if len(_components_mask(grid == BACKGROUND)) != len(chambers):
            continue
        return chosen
    return None


def _adjacent_marker_positions(grid: np.ndarray, chamber: np.ndarray) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for r0, c0 in chamber:
        for r, c in _neighbors4(int(r0), int(c0)):
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and int(grid[r, c]) != BACKGROUND:
                pos = (int(r), int(c))
                if pos not in seen:
                    seen.add(pos)
                    out.append(pos)
    return out


def _crossing_count(line_items: list[dict[str, object]]) -> int:
    counts: Counter[tuple[int, int]] = Counter()
    for item in line_items:
        for cell in item["cells"]:
            counts[cell] += 1
    return sum(1 for count in counts.values() if count > 1)


def _validate_instance(grid: np.ndarray, require_crossing: bool, want_extra: bool) -> bool:
    line_colors, marker_colors = _line_and_marker_roles(grid)
    points = [tuple(map(int, rc)) for rc in np.argwhere(grid != BACKGROUND)]
    for i, a in enumerate(points):
        for b in points[i + 1 :]:
            if _manhattan(a, b) == 1:
                return False

    for color in line_colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        if len(coords) != 2:
            return False
        if abs(coords[0][0] - coords[1][0]) != abs(coords[0][1] - coords[1][1]):
            return False

    rendered, line_items = _draw_all_lines(grid)
    crossings = _crossing_count(line_items)
    if require_crossing and crossings == 0:
        return False
    chambers = _components_mask(rendered == BACKGROUND)
    if not (3 <= len(chambers) <= 5):
        return False

    multi_marker_chamber = False
    for chamber in chambers:
        adjacent = [
            pos
            for pos in _adjacent_marker_positions(grid, chamber)
            if int(grid[pos]) in marker_colors
        ]
        if not adjacent:
            return False
        if len(adjacent) > 1:
            multi_marker_chamber = True
    if want_extra and not multi_marker_chamber:
        return False

    out = main(grid)
    for pos in np.argwhere(rendered != BACKGROUND):
        r, c = map(int, pos)
        if int(out[r, c]) != int(rendered[r, c]):
            return False

    chamber_colors = []
    for chamber in chambers:
        colors = {int(out[r, c]) for r, c in chamber}
        if len(colors) != 1:
            return False
        chamber_colors.extend(colors)
    return len(set(chamber_colors)) >= 2


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    line_count = random.choice([3, 4])
    require_crossing = random.random() < 0.5
    layout = _build_line_layout((h, w), line_count, require_crossing=require_crossing)
    if layout is None:
        return None
    segments, _line_mask = layout

    line_colors = random.sample(PALETTE, line_count)
    marker_palette = [color for color in PALETTE if color not in line_colors]

    output = np.zeros((h, w), dtype=int)
    input_grid = np.zeros((h, w), dtype=int)
    endpoints: set[tuple[int, int]] = set()
    line_items: list[dict[str, object]] = []
    for color, seg in zip(line_colors, segments):
        a, b = seg["endpoints"]
        endpoints.update([a, b])
        cells = seg["cells"]
        line_items.append({"color": color, "endpoints": seg["endpoints"], "first": min(a, b), "cells": cells})
        input_grid[a] = color
        input_grid[b] = color
    line_items.sort(key=lambda item: item["first"])
    for item in line_items:
        for r, c in item["cells"]:
            output[r, c] = int(item["color"])

    chambers = _components_mask(output == BACKGROUND)
    want_extra = len(chambers) < 5 and random.random() < 0.7
    marker_positions = _pick_markers(output, chambers, endpoints, want_extra=want_extra)
    if marker_positions is None:
        return None

    marker_colors = random.sample(marker_palette, len(marker_positions))
    for color, pos in zip(marker_colors, marker_positions):
        input_grid[pos] = color

    if not _validate_instance(input_grid, require_crossing=require_crossing, want_extra=want_extra):
        return None
    if np.array_equal(main(input_grid), input_grid):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
