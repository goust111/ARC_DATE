from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg  # noqa: E402


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
WALL_COLOR = int(Color.GREY)
SOURCE_COLORS = [int(c) for c in Color.NOT_BLACK if int(c) != WALL_COLOR]

GRID_HEIGHT_RANGE = (12, 20)
SECTOR_WIDTH_RANGE = (6, 8)
DIVIDER_WIDTH = 1
SECTOR_COUNT = 3
MAX_GENERATION_ATTEMPTS = 300


def _parse_scene(grid: np.ndarray) -> tuple[int, np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    wall_mask = np.zeros(g.shape, dtype=bool)
    sources: list[dict] = []

    for comp in find_components(g, bg=bg, connectivity=8, monochromatic=True):
        coords = comp["coords"]
        if int(comp["area"]) == 1:
            r, c = map(int, coords[0])
            sources.append({"color": int(comp["color"]), "pos": (r, c)})
        else:
            wall_mask[coords[:, 0], coords[:, 1]] = True

    return bg, wall_mask, sources


def _collect_hits(shape: tuple[int, int], wall_mask: np.ndarray, sources: list[dict]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    hits: dict[tuple[int, int], list[tuple[int, int]]] = {}
    H, W = map(int, shape)

    for src_id, src in enumerate(sources):
        r0, c0 = src["pos"]
        for dr, dc in DIRS4:
            r = r0 + dr
            c = c0 + dc
            dist = 1
            while 0 <= r < H and 0 <= c < W and not wall_mask[r, c]:
                hits.setdefault((r, c), []).append((src_id, dist))
                r += dr
                c += dc
                dist += 1

    return hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal only the cells where exactly two unobstructed cardinal rays from different singleton sources cross.

    Input structure:
    - Black is the background.
    - Every non-black 8-connected monochrome component of size greater than one is a wall cluster.
    - Every non-black 8-connected monochrome component of size one is a light source pixel.

    Rule:
    - Each source emits invisible rays up, down, left, and right; a ray stops before the first wall cell or at the grid edge.
    - Keep the rays themselves invisible and preserve all original wall and source pixels.
    - Recolor only a background cell that is traversed by exactly two rays from different sources: use the nearer source's color, and leave the cell black on a distance tie.

    Key insight:
    - The visible output is not a rectangle-corner completion; it is the residue of a hidden ray-cast simulation with wall occlusion and a per-intersection nearest-source decision.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, wall_mask, sources = _parse_scene(g)
    assert len(sources) >= 2, "expected at least two singleton light sources"

    out = g.copy()
    source_cells = {tuple(src["pos"]) for src in sources}

    for (r, c), hits in _collect_hits(g.shape, wall_mask, sources).items():
        if len(hits) != 2:
            continue
        if (r, c) in source_cells or int(g[r, c]) != bg:
            continue

        (src_a, dist_a), (src_b, dist_b) = hits
        if src_a == src_b or dist_a == dist_b:
            continue

        winner = src_a if dist_a < dist_b else src_b
        out[r, c] = int(sources[winner]["color"])

    return out


def _sector_bounds(widths: list[int]) -> list[tuple[int, int]]:
    sectors = []
    cursor = 0
    for i, width in enumerate(widths):
        sectors.append((cursor, cursor + width - 1))
        cursor += width
        if i < len(widths) - 1:
            cursor += DIVIDER_WIDTH
    return sectors


def _candidate_pairs(kind: str, H: int, left: int, right: int) -> list[tuple[int, int]]:
    usable_w = right - left + 1
    max_dr = min(8, H - 3)
    max_dc = usable_w - 3
    if max_dr < 2 or max_dc < 2:
        return []

    if kind == "tie":
        return [(d, d) for d in range(2, min(max_dr, max_dc) + 1)]

    return [(dr, dc) for dr in range(2, max_dr + 1) for dc in range(2, max_dc + 1) if dr != dc]


def _trace_reaches(wall_mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> bool:
    sr, sc = start
    er, ec = end
    if sr != er and sc != ec:
        return False

    dr = 0 if sr == er else (1 if er > sr else -1)
    dc = 0 if sc == ec else (1 if ec > sc else -1)
    r, c = sr + dr, sc + dc
    while (r, c) != (er, ec):
        if wall_mask[r, c]:
            return False
        r += dr
        c += dc
    return not wall_mask[er, ec]


def _winner_color(src_a: dict, src_b: dict, cell: tuple[int, int], bg: int) -> int:
    da = abs(src_a["pos"][0] - cell[0]) + abs(src_a["pos"][1] - cell[1])
    db = abs(src_b["pos"][0] - cell[0]) + abs(src_b["pos"][1] - cell[1])
    if da == db:
        return bg
    return int(src_a["color"] if da < db else src_b["color"])


def _place_pair(
    grid: np.ndarray,
    sector: tuple[int, int],
    kind: str,
    colors: tuple[int, int],
) -> dict | None:
    H, _W = grid.shape
    left, right = sector
    options = _candidate_pairs(kind, H, left, right)
    if not options:
        return None

    dr, dc = random.choice(options)
    base_r_max = H - dr - 2
    base_c_min = left + 1
    base_c_max = right - dc - 1
    if base_r_max < 1 or base_c_min > base_c_max:
        return None

    r0 = random.randint(1, base_r_max)
    c0 = random.randint(base_c_min, base_c_max)
    flip = random.choice([False, True])

    if not flip:
        src_top = {"color": int(colors[0]), "pos": (r0, c0)}
        src_bottom = {"color": int(colors[1]), "pos": (r0 + dr, c0 + dc)}
        top_corner = (r0, c0 + dc)
        bottom_corner = (r0 + dr, c0)
    else:
        src_top = {"color": int(colors[0]), "pos": (r0, c0 + dc)}
        src_bottom = {"color": int(colors[1]), "pos": (r0 + dr, c0)}
        top_corner = (r0, c0)
        bottom_corner = (r0 + dr, c0 + dc)

    blocker_cells: list[tuple[int, int]] = []
    blocked_corner = None
    open_corner = None
    if kind == "blocked":
        block_axis = random.choice(["h", "v"])
        if block_axis == "h":
            block_c = random.randint(min(src_top["pos"][1], top_corner[1]) + 1, max(src_top["pos"][1], top_corner[1]) - 1)
            blocker_cells = [(r0, block_c), (r0 + 1, block_c)]
            blocked_corner = top_corner
            open_corner = bottom_corner
        else:
            block_r = random.randint(src_top["pos"][0] + 1, bottom_corner[0] - 1)
            wall_c = src_top["pos"][1]
            extra_c = wall_c + 1 if not flip else wall_c - 1
            blocker_cells = [(block_r, wall_c), (block_r, extra_c)]
            blocked_corner = bottom_corner
            open_corner = top_corner

    for src in (src_top, src_bottom):
        rr, cc = src["pos"]
        grid[rr, cc] = int(src["color"])
    for rr, cc in blocker_cells:
        grid[rr, cc] = WALL_COLOR

    return {
        "kind": kind,
        "sources": (src_top, src_bottom),
        "top_corner": top_corner,
        "bottom_corner": bottom_corner,
        "blocked_corner": blocked_corner,
        "open_corner": open_corner,
        "blocker_cells": blocker_cells,
    }


def _validate_attempt(grid: np.ndarray, out: np.ndarray, events: list[dict]) -> bool:
    bg = int(Color.BLACK)
    parsed_bg, wall_mask, parsed_sources = _parse_scene(grid)
    if parsed_bg != bg or len(parsed_sources) != 6:
        return False

    expected_changed: dict[tuple[int, int], int] = {}
    for event in events:
        src_top, src_bottom = event["sources"]
        top_corner = event["top_corner"]
        bottom_corner = event["bottom_corner"]
        kind = event["kind"]

        if kind == "near":
            if not (_trace_reaches(wall_mask, src_top["pos"], top_corner) and _trace_reaches(wall_mask, src_bottom["pos"], top_corner)):
                return False
            if not (_trace_reaches(wall_mask, src_top["pos"], bottom_corner) and _trace_reaches(wall_mask, src_bottom["pos"], bottom_corner)):
                return False
            expected_changed[top_corner] = _winner_color(src_top, src_bottom, top_corner, bg)
            expected_changed[bottom_corner] = _winner_color(src_top, src_bottom, bottom_corner, bg)
        elif kind == "tie":
            if _winner_color(src_top, src_bottom, top_corner, bg) != bg:
                return False
            if _winner_color(src_top, src_bottom, bottom_corner, bg) != bg:
                return False
            if not (_trace_reaches(wall_mask, src_top["pos"], top_corner) and _trace_reaches(wall_mask, src_bottom["pos"], top_corner)):
                return False
            if not (_trace_reaches(wall_mask, src_top["pos"], bottom_corner) and _trace_reaches(wall_mask, src_bottom["pos"], bottom_corner)):
                return False
        else:
            blocked_corner = event["blocked_corner"]
            open_corner = event["open_corner"]
            assert blocked_corner is not None and open_corner is not None

            top_to_blocked = _trace_reaches(wall_mask, src_top["pos"], blocked_corner)
            bottom_to_blocked = _trace_reaches(wall_mask, src_bottom["pos"], blocked_corner)
            if top_to_blocked or not bottom_to_blocked:
                return False
            if not (_trace_reaches(wall_mask, src_top["pos"], open_corner) and _trace_reaches(wall_mask, src_bottom["pos"], open_corner)):
                return False
            expected_changed[open_corner] = _winner_color(src_top, src_bottom, open_corner, bg)

    if any(color == bg for color in expected_changed.values()):
        return False

    changed_cells = {tuple(map(int, cell)) for cell in np.argwhere(out != grid)}
    if changed_cells != set(expected_changed):
        return False
    for cell, color in expected_changed.items():
        if int(out[cell]) != int(color):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_HEIGHT_RANGE)
    sector_widths = [random.randint(*SECTOR_WIDTH_RANGE) for _ in range(SECTOR_COUNT)]
    W = sum(sector_widths) + (SECTOR_COUNT - 1) * DIVIDER_WIDTH
    if W > 30:
        return None

    grid = np.full((H, W), int(Color.BLACK), dtype=int)
    sectors = _sector_bounds(sector_widths)
    for left, right in sectors[:-1]:
        divider_left = right + 1
        grid[:, divider_left : divider_left + DIVIDER_WIDTH] = WALL_COLOR

    event_order = ["near", "tie", "blocked"]
    random.shuffle(event_order)
    colors = random.sample(SOURCE_COLORS, 2 * SECTOR_COUNT)

    events = []
    for i, (sector, kind) in enumerate(zip(sectors, event_order)):
        event = _place_pair(grid, sector, kind, (colors[2 * i], colors[2 * i + 1]))
        if event is None:
            return None
        events.append(event)

    out = main(grid)
    if not _validate_attempt(grid, out, events):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid invisible-ray-intersection input")
