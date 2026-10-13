from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
RING = (RED, GREEN, BLUE)
PORTAL_COLORS = (5, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = int(cr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _line_info(coords: np.ndarray) -> tuple[str, list[tuple[int, int]]] | None:
    if int(coords.shape[0]) < 2:
        return None

    rows = sorted(int(r) for r in coords[:, 0])
    cols = sorted(int(c) for c in coords[:, 1])

    if len(set(rows)) == 1:
        c0 = cols[0]
        if cols == list(range(c0, c0 + len(cols))):
            r = rows[0]
            return "h", [(int(r), int(c)) for c in cols]

    if len(set(cols)) == 1:
        r0 = rows[0]
        if rows == list(range(r0, r0 + len(rows))):
            c = cols[0]
            return "v", [(int(r), int(c)) for r in rows]

    return None


def _find_portals(grid: np.ndarray) -> tuple[dict[int, list[dict]], dict[tuple[int, int], tuple[int, int, int, str]]]:
    special = {BLACK, RED, GREEN, BLUE, YELLOW}
    colors = sorted(int(c) for c in np.unique(grid) if int(c) not in special)

    portals: dict[int, list[dict]] = {}
    lookup: dict[tuple[int, int], tuple[int, int, int, str]] = {}

    for color in colors:
        mask = grid == int(color)
        comps = _components(mask)
        assert len(comps) == 2, "each portal color must appear as exactly two components"
        infos: list[dict] = []
        total = 0
        for comp in comps:
            total += int(comp.shape[0])
            line = _line_info(comp)
            assert line is not None, "portal component must be a straight line"
            orientation, cells = line
            assert len(cells) == 3, "portal component must have length 3"
            infos.append({"orientation": orientation, "cells": cells})
        assert total == 6, "each portal color must occupy exactly 6 cells"
        assert infos[0]["orientation"] == infos[1]["orientation"], "portal pair must share orientation"
        portals[int(color)] = infos
        for portal_idx, info in enumerate(infos):
            for offset, cell in enumerate(info["cells"]):
                lookup[cell] = (int(color), int(portal_idx), int(offset), str(info["orientation"]))

    assert portals, "expected at least one portal pair"
    return portals, lookup


def _find_start(grid: np.ndarray, portal_lookup: dict[tuple[int, int], tuple[int, int, int, str]]) -> dict:
    comps = _components(grid == RED)
    candidates: list[dict] = []
    for comp in comps:
        line = _line_info(comp)
        if line is None:
            continue
        orientation, cells = line
        if not 2 <= len(cells) <= 4:
            continue

        hits: list[tuple[tuple[int, int], tuple[int, int]]] = []
        if orientation == "h":
            left, right = cells[0], cells[-1]
            if (int(left[0]), int(left[1] - 1)) in portal_lookup:
                hits.append((left, (0, -1)))
            if (int(right[0]), int(right[1] + 1)) in portal_lookup:
                hits.append((right, (0, 1)))
        else:
            top, bottom = cells[0], cells[-1]
            if (int(top[0] - 1), int(top[1])) in portal_lookup:
                hits.append((top, (-1, 0)))
            if (int(bottom[0] + 1), int(bottom[1])) in portal_lookup:
                hits.append((bottom, (1, 0)))

        if len(hits) == 1:
            near, direction = hits[0]
            candidates.append(
                {
                    "orientation": orientation,
                    "cells": cells,
                    "near": near,
                    "dir": direction,
                }
            )

    assert len(candidates) == 1, "expected exactly one valid red start segment"
    return candidates[0]


def _trace_path(grid: np.ndarray) -> tuple[np.ndarray, int, str]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    yellow_pts = np.argwhere(g == YELLOW)
    assert int(yellow_pts.shape[0]) == 1, "expected one yellow endpoint"
    endpoint = (int(yellow_pts[0, 0]), int(yellow_pts[0, 1]))

    portals, portal_lookup = _find_portals(g)
    start = _find_start(g, portal_lookup)

    out = g.copy()
    pos = (int(start["near"][0]), int(start["near"][1]))
    dr, dc = map(int, start["dir"])
    color_idx = 0
    teleports = 0
    seen_states: set[tuple[int, int, int, int, int]] = set()

    while True:
        nr, nc = int(pos[0] + dr), int(pos[1] + dc)
        if not (0 <= nr < h and 0 <= nc < w):
            return out, int(teleports), "boundary"
        if (nr, nc) == endpoint:
            return out, int(teleports), "endpoint"

        hit = portal_lookup.get((nr, nc))
        if hit is not None:
            color, portal_idx, offset, orientation = hit
            if dr != 0:
                assert orientation == "h", "vertical travel must enter a horizontal portal"
            else:
                assert orientation == "v", "horizontal travel must enter a vertical portal"
            pair = portals[int(color)][1 - int(portal_idx)]
            pos = pair["cells"][int(offset)]
            color_idx = (int(color_idx) + 1) % len(RING)
            teleports += 1
            state = (int(pos[0]), int(pos[1]), int(dr), int(dc), int(color_idx))
            assert state not in seen_states, "path entered a teleport loop"
            seen_states.add(state)
            continue

        assert int(g[nr, nc]) == BLACK, "path may only extend through background cells"
        out[nr, nc] = int(RING[color_idx])
        pos = (int(nr), int(nc))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify paired portals, teleport the path through them, and rotate path colors after each teleport.
    """
    return _trace_path(np.asarray(grid, dtype=int))[0]


def _valid_offsets(row: int, height: int) -> list[int]:
    return [off for off in range(3) if 0 <= int(row - off) <= int(height - 3)]


def _portal_cells(orientation: str, start_r: int, start_c: int) -> list[tuple[int, int]]:
    if orientation == "h":
        return [(int(start_r), int(start_c + i)) for i in range(3)]
    return [(int(start_r + i), int(start_c)) for i in range(3)]


def _expand(cells: Iterable[tuple[int, int]], height: int, width: int, radius: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < height and 0 <= nc < width:
                    out.add((int(nr), int(nc)))
    return out


def _place_extra_pair(
    grid: np.ndarray,
    color: int,
    blocked_exact: set[tuple[int, int]],
    blocked_margin: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    for _ in range(200):
        orientation = random.choice(("h", "v"))
        if orientation == "h":
            r1 = random.randint(0, h - 1)
            c1 = random.randint(0, w - 3)
            r2 = random.randint(0, h - 1)
            c2 = random.randint(0, w - 3)
        else:
            r1 = random.randint(0, h - 3)
            c1 = random.randint(0, w - 1)
            r2 = random.randint(0, h - 3)
            c2 = random.randint(0, w - 1)

        cells1 = _portal_cells(orientation, r1, c1)
        cells2 = _portal_cells(orientation, r2, c2)
        all_cells = cells1 + cells2

        if len(set(all_cells)) != 6:
            continue
        if any(cell in blocked_exact or cell in blocked_margin for cell in all_cells):
            continue
        if any(int(grid[r, c]) != BLACK for r, c in all_cells):
            continue
        if any(max(abs(r1 - r2), abs(c1 - c2)) <= 1 for r1, c1 in cells1 for r2, c2 in cells2):
            continue

        for r, c in all_cells:
            grid[r, c] = int(color)
        return True
    return False


def _apply_random_transform(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    if random.randrange(2):
        out = np.fliplr(out)
    if random.randrange(2):
        out = out.T
    return np.array(out, dtype=int, copy=True)


def _generate_canonical() -> tuple[np.ndarray, int, str] | None:
    chain_len = random.randint(1, 3)
    hit_endpoint = bool(random.randrange(2))

    if chain_len == 3:
        start_pad = random.randint(0, 2)
        jump_lens = [random.randint(2, 4) for _ in range(chain_len)]
        travel_lens = [random.randint(1, 3) for _ in range(chain_len - 1)]
        final_len = random.randint(1, 4)
    elif chain_len == 2:
        start_pad = random.randint(0, 3)
        jump_lens = [random.randint(2, 5) for _ in range(chain_len)]
        travel_lens = [random.randint(1, 4) for _ in range(chain_len - 1)]
        final_len = random.randint(1, 5)
    else:
        start_pad = random.randint(0, 3)
        jump_lens = [random.randint(2, 6) for _ in range(chain_len)]
        travel_lens = []
        final_len = random.randint(1, 6)

    start_len = random.randint(2, 4)
    tail_margin = random.randint(1, 4) if hit_endpoint else 0

    entrances: list[int] = [int(start_pad + start_len)]
    exits: list[int] = []
    for i in range(chain_len):
        exits.append(int(entrances[i] + jump_lens[i]))
        if i + 1 < chain_len:
            entrances.append(int(exits[i] + 1 + travel_lens[i]))

    if hit_endpoint:
        endpoint_col = int(exits[-1] + 1 + final_len)
        width = int(endpoint_col + 1 + tail_margin)
        stop_reason = "endpoint"
    else:
        endpoint_col = -1
        width = int(exits[-1] + 1 + final_len)
        stop_reason = "boundary"

    if not 10 <= width <= 28:
        return None

    min_height = max(8, chain_len + 5)
    height = random.randint(min_height, 18)
    rows = random.sample(list(range(1, height - 1)), chain_len + 1)

    pair_offsets: list[int] = []
    for i in range(chain_len):
        valid = sorted(set(_valid_offsets(rows[i], height)) & set(_valid_offsets(rows[i + 1], height)))
        if not valid:
            return None
        pair_offsets.append(int(random.choice(valid)))

    colors = list(PORTAL_COLORS)
    random.shuffle(colors)
    used_colors = colors[:chain_len]
    extra_colors = colors[chain_len:]

    grid = np.full((height, width), BLACK, dtype=int)
    grid[int(rows[0]), int(start_pad) : int(start_pad + start_len)] = RED

    used_portal_cells: set[tuple[int, int]] = set()
    future_path_cells: set[tuple[int, int]] = set()

    for i in range(chain_len):
        offset = pair_offsets[i]
        top_a = int(rows[i] - offset)
        top_b = int(rows[i + 1] - offset)
        col_a = int(entrances[i])
        col_b = int(exits[i])

        a_cells = _portal_cells("v", top_a, col_a)
        b_cells = _portal_cells("v", top_b, col_b)
        for r, c in a_cells + b_cells:
            assert int(grid[r, c]) == BLACK
            grid[r, c] = int(used_colors[i])
            used_portal_cells.add((int(r), int(c)))

        if i + 1 < chain_len:
            for c in range(int(col_b + 1), int(entrances[i + 1])):
                future_path_cells.add((int(rows[i + 1]), int(c)))

    if hit_endpoint:
        grid[int(rows[-1]), int(endpoint_col)] = YELLOW
        for c in range(int(exits[-1] + 1), int(endpoint_col)):
            future_path_cells.add((int(rows[-1]), int(c)))
    else:
        for c in range(int(exits[-1] + 1), width):
            future_path_cells.add((int(rows[-1]), int(c)))

        free_cells = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if int(grid[r, c]) == BLACK and (int(r), int(c)) not in future_path_cells
        ]
        if not free_cells:
            return None
        er, ec = random.choice(free_cells)
        grid[int(er), int(ec)] = YELLOW

    occupied = {(int(r), int(c)) for r, c in np.argwhere(grid != BLACK)}
    blocked_exact = set(future_path_cells) | set(occupied)
    blocked_margin = _expand(occupied, height, width, radius=1)

    max_extras = min(2, len(extra_colors))
    extra_count = random.randint(0, max_extras)
    for color in extra_colors[:extra_count]:
        if not _place_extra_pair(grid, int(color), blocked_exact, blocked_margin):
            return None
        new_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == int(color))}
        blocked_exact |= new_cells
        blocked_margin |= _expand(new_cells, height, width, radius=1)

    return grid, int(chain_len), stop_reason


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _generate_canonical()
        if built is None:
            continue
        grid, expected_teleports, expected_stop = built
        candidate = _apply_random_transform(grid)

        try:
            _out, teleports, stop_reason = _trace_path(candidate)
        except AssertionError:
            continue

        if teleports != int(expected_teleports):
            continue
        if stop_reason != expected_stop:
            continue
        return np.array(candidate, dtype=int, copy=True)

    raise RuntimeError("failed to generate a valid portal puzzle")
