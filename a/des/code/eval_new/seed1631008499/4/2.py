from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_full_2x2(cells: list[tuple[int, int]]) -> bool:
    if len(cells) != 4:
        return False
    r0, c0, r1, c1 = _bbox(cells)
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        return False
    return set(cells) == {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}


def _pipe_neighbors(pipe_mask: np.ndarray, cell: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = pipe_mask.shape
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and bool(pipe_mask[nr, nc]):
            out.append((nr, nc))
    return out


def _continue_straight(
    current: tuple[int, int],
    previous: tuple[int, int],
    neighbors: list[tuple[int, int]],
) -> tuple[int, int] | None:
    dr = current[0] - previous[0]
    dc = current[1] - previous[1]
    straight = (current[0] + dr, current[1] + dc)
    if straight in neighbors:
        return straight

    options = [cell for cell in neighbors if cell != previous]
    if len(options) == 1:
        return options[0]
    return None


def _parse_scene(grid: np.ndarray) -> tuple[int, int, dict[int, int], list[dict[str, object]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    counts = Counter(map(int, g.ravel()))
    bg = int(counts.most_common(1)[0][0])

    non_bg_colors = [int(color) for color in counts if int(color) != bg]
    assert len(non_bg_colors) >= 3, "expected pipe color plus multiple source colors"

    source_blocks: list[dict[str, object]] = []
    source_colors: set[int] = set()
    for color in non_bg_colors:
        comps = _components(g == color)
        blocks = [cells for cells in comps if _is_full_2x2(cells)]
        if not blocks:
            continue
        assert len(blocks) == 1, "each source color must have exactly one 2x2 source block"
        source_colors.add(int(color))
        source_blocks.append({"color": int(color), "cells": blocks[0]})

    assert len(source_blocks) >= 2, "expected at least two sources"
    pipe_candidates = [color for color in non_bg_colors if color not in source_colors]
    assert len(pipe_candidates) == 1, "expected exactly one pipe color"
    pipe_color = int(pipe_candidates[0])
    pipe_mask = g == pipe_color

    source_cells = {cell for block in source_blocks for cell in block["cells"]}  # type: ignore[index]
    legend_pixels: list[tuple[int, int, int]] = []
    for color in source_colors:
        extra = [(int(r), int(c)) for r, c in np.argwhere(g == color) if (int(r), int(c)) not in source_cells]
        assert len(extra) == 1, "each source color must appear once in the corner legend"
        legend_pixels.append((extra[0][0], extra[0][1], int(color)))
    legend_pixels.sort(key=lambda item: (item[0], item[1]))
    priority = {color: rank for rank, (_r, _c, color) in enumerate(legend_pixels)}

    sources: list[dict[str, object]] = []
    for block in source_blocks:
        color = int(block["color"])  # type: ignore[index]
        adj: dict[tuple[int, int], tuple[int, int]] = {}
        for sr, sc in block["cells"]:  # type: ignore[index]
            for dr, dc in DIR4:
                nr, nc = sr + dr, sc + dc
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and int(g[nr, nc]) == pipe_color:
                    adj[(nr, nc)] = (sr, sc)
        assert len(adj) == 1, "each source block must touch exactly one pipe endpoint"
        start = next(iter(adj))
        prev = adj[start]
        sources.append({"color": color, "start": start, "prev": prev})

    return bg, pipe_color, priority, sources, pipe_mask


def _solve(grid: np.ndarray, *, collect_stats: bool = False) -> tuple[np.ndarray, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    _bg, pipe_color, priority, sources, pipe_mask = _parse_scene(g)
    out = g.copy()
    owned_intersections: dict[tuple[int, int], int] = {}
    stats: list[dict[str, int]] = []

    ordered_sources = sorted(sources, key=lambda src: priority[int(src["color"])])  # type: ignore[index]
    for source in ordered_sources:
        color = int(source["color"])  # type: ignore[index]
        current = tuple(source["start"])  # type: ignore[arg-type]
        previous = tuple(source["prev"])  # type: ignore[arg-type]
        seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        colored = 0
        intersections = 0
        blocked = 0

        while True:
            state = (current, previous)
            if state in seen:
                break
            seen.add(state)

            r, c = current
            here = int(out[r, c])
            if here not in (pipe_color, color):
                blocked = 1
                break

            neighbors = _pipe_neighbors(pipe_mask, current)
            if len(neighbors) >= 3:
                owner = owned_intersections.get(current)
                if owner is not None and owner != color:
                    blocked = 1
                    break
                if owner is None:
                    owned_intersections[current] = color
                intersections += 1

            out[r, c] = color
            colored += 1

            next_cells = [cell for cell in neighbors if cell != previous]
            if not next_cells:
                break
            if len(next_cells) == 1:
                previous, current = current, next_cells[0]
                continue

            nxt = _continue_straight(current, previous, neighbors)
            if nxt is None:
                break
            previous, current = current, nxt

        if collect_stats:
            stats.append(
                {
                    "color": color,
                    "colored": colored,
                    "intersections": intersections,
                    "blocked": blocked,
                }
            )

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate source colors through a pipe network, resolving crossings by legend priority.

    Input structure:
    - bg is the most common color.
    - One color forms a width-1 orthogonal pipe network with straight segments and crossings.
    - Several distinct 2x2 solid source blocks touch pipe endpoints, and a corner column of singleton pixels lists those source colors from high to low priority.

    Rule:
    - Each source injects its own color into the unique adjacent pipe endpoint and advances along that pipe path.
    - On ordinary pipe cells, the color keeps moving forward; at a crossing, only the highest-priority arriving color occupies the crossing and continues straight through it.
    - Lower-priority colors stop immediately before a crossing already claimed by a higher-priority color.
    - Source blocks, legend pixels, and untouched pipe cells remain unchanged.

    Key insight:
    - The singleton corner legend gives a global priority order for the source colors, so the final pipe coloring is obtained by tracing each source path while letting higher-priority flows claim crossings first.
    """
    return _solve(grid, collect_stats=False)[0]


BG = 0
PIPE = 5
SOURCE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GRID_MIN = 20
GRID_MAX = 28
TOTAL_SOURCES_RANGE = (3, 5)
LINE_COUNT_RANGE = (2, 4)
SPACING = 3
LEFT_MARGIN = 4
BOTTOM_MARGIN = 4
GEN_RETRIES = 400


def _sample_spaced(count: int, lo: int, hi: int, gap: int) -> list[int] | None:
    if lo > hi:
        return None
    values = list(range(lo, hi + 1))
    chosen: list[int] = []
    for _ in range(count):
        options = [value for value in values if all(abs(value - other) >= gap for other in chosen)]
        if not options:
            return None
        chosen.append(random.choice(options))
    return sorted(chosen)


def _render_source(
    grid: np.ndarray,
    *,
    kind: str,
    anchor: int,
    side: str,
    color: int,
    top: int,
    bottom: int,
    left: int,
    right: int,
) -> None:
    if kind == "h":
        r = anchor
        if side == "left":
            grid[r : r + 2, left - 2 : left] = color
        else:
            grid[r : r + 2, right + 1 : right + 3] = color
        return

    c = anchor
    if side == "top":
        grid[top - 2 : top, c : c + 2] = color
    else:
        grid[bottom + 1 : bottom + 3, c : c + 2] = color


def _legend_positions(height: int, width: int, count: int, side: str) -> list[tuple[int, int]]:
    col = 0 if side == "left" else width - 1
    return [(row, col) for row in range(count)]


def _gen_attempt() -> np.ndarray | None:
    total_sources = random.randint(*TOTAL_SOURCES_RANGE)
    horizontal_sources = random.randint(1, total_sources - 1)
    vertical_sources = total_sources - horizontal_sources

    num_h = random.randint(max(LINE_COUNT_RANGE[0], horizontal_sources), LINE_COUNT_RANGE[1])
    num_v = random.randint(max(LINE_COUNT_RANGE[0], vertical_sources), LINE_COUNT_RANGE[1])

    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    if total_sources > len(SOURCE_COLORS):
        return None

    legend_side = random.choice(["left", "right"])
    legend_height = total_sources
    top = legend_height + 3
    bottom = h - BOTTOM_MARGIN
    left = LEFT_MARGIN
    right = w - LEFT_MARGIN

    rows = _sample_spaced(num_h, top + 2, bottom - 2, SPACING)
    cols = _sample_spaced(num_v, left + 2, right - 2, SPACING)
    if rows is None or cols is None:
        return None

    source_specs: list[tuple[str, int, str]] = []
    for row in random.sample(rows, horizontal_sources):
        source_specs.append(("h", int(row), random.choice(["left", "right"])))
    for col in random.sample(cols, vertical_sources):
        source_specs.append(("v", int(col), random.choice(["top", "bottom"])))

    random.shuffle(source_specs)
    orientations = [kind for kind, _anchor, _side in source_specs]
    if all(kind == orientations[0] for kind in orientations):
        return None

    for _ in range(20):
        priority_specs = list(source_specs)
        random.shuffle(priority_specs)
        priority_kinds = [kind for kind, _anchor, _side in priority_specs]
        first_h = priority_kinds.index("h")
        last_h = len(priority_kinds) - 1 - priority_kinds[::-1].index("h")
        first_v = priority_kinds.index("v")
        last_v = len(priority_kinds) - 1 - priority_kinds[::-1].index("v")
        if not (last_h < first_v or last_v < first_h):
            break
    else:
        return None

    legend_colors = random.sample(SOURCE_COLORS, total_sources)
    spec_to_color = {spec: legend_colors[i] for i, spec in enumerate(priority_specs)}

    grid = np.full((h, w), BG, dtype=int)
    for row in rows:
        grid[int(row), left : right + 1] = PIPE
    for col in cols:
        grid[top : bottom + 1, int(col)] = PIPE

    for spec in source_specs:
        kind, anchor, side = spec
        _render_source(
            grid,
            kind=kind,
            anchor=anchor,
            side=side,
            color=spec_to_color[spec],
            top=top,
            bottom=bottom,
            left=left,
            right=right,
        )

    for (r, c), color in zip(_legend_positions(h, w, total_sources, legend_side), legend_colors, strict=True):
        grid[r, c] = color

    try:
        out, stats = _solve(grid, collect_stats=True)
    except AssertionError:
        return None

    changed_pipe = (grid == PIPE) & (out != PIPE)
    unchanged_pipe = (grid == PIPE) & (out == PIPE)
    if not bool(np.any(changed_pipe)):
        return None
    if not bool(np.any(unchanged_pipe)):
        return None
    if not any(stat["blocked"] for stat in stats):
        return None
    if not any(stat["intersections"] > 0 for stat in stats):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-priority puzzle")
