from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
RED = 2
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
GEAR_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
RELATIONS = ["right", "left", "down", "up", "diag_dr", "diag_dl", "diag_ur", "diag_ul"]
CW_CORNERS = {"tl", "br"}
CCW_CORNERS = {"tr", "bl"}
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _component_cells(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_color_components(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    comp_idx = np.full((h, w), -1, dtype=int)
    comps: list[dict] = []
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in (BG, RED) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                comp_idx[rr, cc] = len(comps)
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(g.shape, nr, nc) and not bool(seen[nr, nc]) and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            comps.append(
                {
                    "color": color,
                    "cells": set(cells),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps, comp_idx


def _extract_gears(grid: np.ndarray) -> tuple[list[dict], int]:
    g = np.asarray(grid, dtype=int)
    red_pts = np.argwhere(g == RED)
    assert len(red_pts) == 1, "expected exactly one red driver marker"
    red_r, red_c = map(int, red_pts[0])

    comps, comp_idx = _extract_color_components(g)
    assert comps, "expected at least one gear body"

    driver_candidates = {
        int(comp_idx[nr, nc])
        for dr, dc in DIR4
        if _in_bounds(g.shape, red_r + dr, red_c + dc)
        for nr, nc in [(red_r + dr, red_c + dc)]
        if int(comp_idx[nr, nc]) >= 0
    }
    assert len(driver_candidates) == 1, "red marker must belong to exactly one gear"
    driver_idx = next(iter(driver_candidates))

    gears: list[dict] = []
    for idx, comp in enumerate(comps):
        cells = set(comp["cells"])
        if idx == driver_idx:
            cells.add((red_r, red_c))
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        gear = {
            "color": int(comp["color"]),
            "cells": cells,
            "bbox": (min(rows), min(cols), max(rows), max(cols)),
        }
        if idx == driver_idx:
            gear["marker"] = (red_r, red_c)
        gears.append(gear)
    return gears, driver_idx


def _mesh_graph(gears: list[dict]) -> list[set[int]]:
    halos: list[set[tuple[int, int]]] = []
    for gear in gears:
        halo: set[tuple[int, int]] = set()
        for r, c in gear["cells"]:
            for dr, dc in DIR8:
                halo.add((r + dr, c + dc))
        halo -= gear["cells"]
        halos.append(halo)

    graph = [set() for _ in gears]
    for i in range(len(gears)):
        for j in range(i + 1, len(gears)):
            if halos[i] & gears[j]["cells"]:
                graph[i].add(j)
                graph[j].add(i)
    return graph


def _driver_spin(gear: dict) -> int:
    marker = gear["marker"]
    r0, c0, r1, c1 = gear["bbox"]
    if marker in {(r0, c0), (r1, c1)}:
        return 1
    if marker in {(r0, c1), (r1, c0)}:
        return -1
    raise AssertionError("driver marker must sit on a bbox corner")


def _reachable(graph: list[set[int]], start: int) -> set[int]:
    seen = {int(start)}
    q = deque([int(start)])
    while q:
        node = q.popleft()
        for nxt in graph[node]:
            if nxt not in seen:
                seen.add(int(nxt))
                q.append(int(nxt))
    return seen


def _propagate_spins(gears: list[dict], driver_idx: int, graph: list[set[int]]) -> dict[int, int]:
    spins = {int(driver_idx): _driver_spin(gears[driver_idx])}
    q = deque([int(driver_idx)])
    while q:
        node = q.popleft()
        for nxt in graph[node]:
            expected = -int(spins[node])
            if nxt in spins:
                assert spins[nxt] == expected, "meshing graph must admit alternating spins"
                continue
            spins[nxt] = expected
            q.append(int(nxt))
    return spins


def _edge_targets(gear: dict, spin: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    cells = gear["cells"]
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    targets: set[tuple[int, int]] = set()

    if spin == 1:
        for r, c in cells:
            if r == r0:
                nr, nc = r - 1, c
                if _in_bounds(shape, nr, nc):
                    targets.add((nr, nc))
            if c == c1:
                nr, nc = r, c + 1
                if _in_bounds(shape, nr, nc):
                    targets.add((nr, nc))
    else:
        for r, c in cells:
            if r == r1:
                nr, nc = r + 1, c
                if _in_bounds(shape, nr, nc):
                    targets.add((nr, nc))
            if c == c0:
                nr, nc = r, c - 1
                if _in_bounds(shape, nr, nc):
                    targets.add((nr, nc))
    return targets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat connected colored blobs as gears. The red corner marker inside one gear
    encodes the initial spin, spin alternates across the 8-neighbor meshing graph,
    and each driven gear expands outward along two spin-dependent sides.
    """
    g = np.asarray(grid, dtype=int)
    gears, driver_idx = _extract_gears(g)
    graph = _mesh_graph(gears)
    spins = _propagate_spins(gears, driver_idx, graph)

    out = g.copy()
    marker_r, marker_c = gears[driver_idx]["marker"]
    out[marker_r, marker_c] = gears[driver_idx]["color"]

    fills: dict[tuple[int, int], int] = {}
    for idx, spin in spins.items():
        color = int(gears[idx]["color"])
        for r, c in _edge_targets(gears[idx], int(spin), g.shape):
            if int(g[r, c]) != BG:
                continue
            if (r, c) in fills:
                assert fills[(r, c)] == color, "expansions should not conflict"
            else:
                fills[(r, c)] = color

    for (r, c), color in fills.items():
        out[r, c] = color
    return out


def _has_expansion_conflict(grid: np.ndarray, gears: list[dict], driver_idx: int, graph: list[set[int]]) -> bool:
    g = np.asarray(grid, dtype=int)
    fills: dict[tuple[int, int], int] = {}
    spins = _propagate_spins(gears, driver_idx, graph)
    for idx, spin in spins.items():
        color = int(gears[idx]["color"])
        for r, c in _edge_targets(gears[idx], int(spin), g.shape):
            if int(g[r, c]) != BG:
                continue
            if (r, c) in fills and fills[(r, c)] != color:
                return True
            fills[(r, c)] = color
    return False


def _local_connected(mask: np.ndarray) -> bool:
    return len(_component_cells(mask)) == 1


def _corner_supported(mask: np.ndarray, corner: str) -> bool:
    h, w = mask.shape
    if corner == "tl":
        neighbors = [(0, 1), (1, 0)]
    elif corner == "tr":
        neighbors = [(0, w - 2), (1, w - 1)]
    elif corner == "bl":
        neighbors = [(h - 2, 0), (h - 1, 1)]
    else:
        neighbors = [(h - 2, w - 1), (h - 1, w - 2)]
    return any(mask[r, c] for r, c in neighbors)


def _sample_shape() -> np.ndarray:
    h = random.randint(3, 6)
    w = random.randint(3, 6)
    mask = np.ones((h, w), dtype=bool)
    edge_cells = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r in (0, h - 1) or c in (0, w - 1)) and (r, c) not in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    ]
    random.shuffle(edge_cells)
    removal_budget = random.choice([0, 0, 1, 1, 2, 2, 3])
    removed = 0
    for r, c in edge_cells:
        if removed >= removal_budget:
            break
        mask[r, c] = False
        if _local_connected(mask) and all(_corner_supported(mask, corner) for corner in ("tl", "tr", "bl", "br")):
            removed += 1
        else:
            mask[r, c] = True
    return mask


def _mask_to_cells(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        cells.add((top + int(r), left + int(c)))
    return cells


def _make_gear(mask: np.ndarray, color: int, top: int, left: int) -> dict:
    h, w = mask.shape
    return {
        "mask": mask.copy(),
        "color": int(color),
        "top": int(top),
        "left": int(left),
        "bbox": (int(top), int(left), int(top + h - 1), int(left + w - 1)),
        "cells": _mask_to_cells(mask, int(top), int(left)),
    }


def _candidate_positions(parent: dict, child_mask: np.ndarray, relation: str) -> list[tuple[int, int]]:
    ph, pw = parent["mask"].shape
    ch, cw = child_mask.shape
    pr0, pc0, pr1, pc1 = parent["bbox"]

    if relation == "right":
        col = pc1 + 1
        return [(row, col) for row in range(pr0 - ch + 1, pr1 + 1)]
    if relation == "left":
        col = pc0 - cw
        return [(row, col) for row in range(pr0 - ch + 1, pr1 + 1)]
    if relation == "down":
        row = pr1 + 1
        return [(row, col) for col in range(pc0 - cw + 1, pc1 + 1)]
    if relation == "up":
        row = pr0 - ch
        return [(row, col) for col in range(pc0 - cw + 1, pc1 + 1)]
    if relation == "diag_dr":
        return [(pr1 + 1, pc1 + 1)]
    if relation == "diag_dl":
        return [(pr1 + 1, pc0 - cw)]
    if relation == "diag_ur":
        return [(pr0 - ch, pc1 + 1)]
    return [(pr0 - ch, pc0 - cw)]


def _fits_with_margin(gear: dict, shape: tuple[int, int], margin: int = 1) -> bool:
    r0, c0, r1, c1 = gear["bbox"]
    return margin <= r0 and margin <= c0 and r1 < shape[0] - margin and c1 < shape[1] - margin


def _candidate_ok(candidate: dict, placed: list[dict], parent_idx: int | None, grid_shape: tuple[int, int]) -> bool:
    if not _fits_with_margin(candidate, grid_shape):
        return False
    occupied = set().union(*(gear["cells"] for gear in placed)) if placed else set()
    if occupied & candidate["cells"]:
        return False
    temp = placed + [candidate]
    neighbors = _mesh_graph(temp)[-1]
    if parent_idx is None:
        return len(neighbors) == 0
    return neighbors == {int(parent_idx)}


def _random_tree_parents(active_count: int) -> list[int]:
    if active_count == 1:
        return []
    if active_count < 3 or random.random() < 0.5:
        return [i - 1 for i in range(1, active_count)]
    return [random.randint(0, i - 1) for i in range(1, active_count)]


def _place_active_gears(masks: list[np.ndarray], colors: list[int], grid_shape: tuple[int, int]) -> list[dict] | None:
    h, w = grid_shape
    first_mask = masks[0]
    max_top = h - first_mask.shape[0] - 2
    max_left = w - first_mask.shape[1] - 2
    if max_top < 2 or max_left < 2:
        return None

    start_top = random.randint(max(2, h // 4 - 2), min(max_top, (3 * h) // 4 - first_mask.shape[0]))
    start_left = random.randint(max(2, w // 4 - 2), min(max_left, (3 * w) // 4 - first_mask.shape[1]))
    placed = [_make_gear(first_mask, colors[0], start_top, start_left)]

    parents = _random_tree_parents(len(masks))
    for child_idx in range(1, len(masks)):
        parent_idx = parents[child_idx - 1]
        relations = RELATIONS[:]
        random.shuffle(relations)
        child_mask = masks[child_idx]
        found = None
        for relation in relations:
            candidates = _candidate_positions(placed[parent_idx], child_mask, relation)
            random.shuffle(candidates)
            for top, left in candidates:
                candidate = _make_gear(child_mask, colors[child_idx], int(top), int(left))
                if _candidate_ok(candidate, placed, parent_idx, grid_shape):
                    found = candidate
                    break
            if found is not None:
                break
        if found is None:
            return None
        placed.append(found)
    return placed


def _place_isolates(
    placed: list[dict], isolate_masks: list[np.ndarray], isolate_colors: list[int], grid_shape: tuple[int, int]
) -> list[dict] | None:
    h, w = grid_shape
    out = list(placed)
    for mask, color in zip(isolate_masks, isolate_colors, strict=False):
        spots = [
            (top, left)
            for top in range(1, h - mask.shape[0])
            for left in range(1, w - mask.shape[1])
        ]
        random.shuffle(spots)
        found = None
        for top, left in spots:
            candidate = _make_gear(mask, int(color), int(top), int(left))
            if _candidate_ok(candidate, out, None, grid_shape):
                found = candidate
                break
        if found is None:
            return None
        out.append(found)
    return out


def _corner_absolute(gear: dict, corner: str) -> tuple[int, int]:
    r0, c0, r1, c1 = gear["bbox"]
    if corner == "tl":
        return r0, c0
    if corner == "tr":
        return r0, c1
    if corner == "bl":
        return r1, c0
    return r1, c1


def _safe_driver_corners(gears: list[dict], driver_idx: int) -> dict[str, list[str]]:
    driver = gears[driver_idx]
    other_cells = set().union(*(gear["cells"] for i, gear in enumerate(gears) if i != driver_idx))
    groups = {"cw": [], "ccw": []}
    for corner in ("tl", "tr", "bl", "br"):
        rr, cc = _corner_absolute(driver, corner)
        if any((rr + dr, cc + dc) in other_cells for dr, dc in DIR8):
            continue
        group = "cw" if corner in CW_CORNERS else "ccw"
        groups[group].append(corner)
    return groups


def _render_input(gears: list[dict], driver_idx: int, driver_corner: str) -> np.ndarray:
    rows = [gear["bbox"][2] for gear in gears]
    cols = [gear["bbox"][3] for gear in gears]
    grid = np.full((max(rows) + 2, max(cols) + 2), BG, dtype=int)
    for gear in gears:
        for r, c in gear["cells"]:
            grid[r, c] = int(gear["color"])
    rr, cc = _corner_absolute(gears[driver_idx], driver_corner)
    grid[rr, cc] = RED
    return grid


def _generate_attempt() -> np.ndarray | None:
    wants_isolates = random.random() < 0.5
    active_count = random.randint(2, 5)
    max_isolates = min(2, len(GEAR_COLORS) - active_count)
    isolate_count = random.randint(1, max_isolates) if wants_isolates and max_isolates > 0 else 0
    total = active_count + isolate_count

    min_side = 16 + 2 * total
    grid_shape = (random.randint(min_side, min_side + 8), random.randint(min_side, min_side + 8))
    colors = random.sample(GEAR_COLORS, total)
    active_masks = [_sample_shape() for _ in range(active_count)]
    isolate_masks = [_sample_shape() for _ in range(isolate_count)]

    active_gears = _place_active_gears(active_masks, colors[:active_count], grid_shape)
    if active_gears is None:
        return None
    all_gears = _place_isolates(active_gears, isolate_masks, colors[active_count:], grid_shape)
    if all_gears is None:
        return None

    safe = _safe_driver_corners(all_gears, 0)
    desired_group = "cw" if random.random() < 0.5 else "ccw"
    if not safe[desired_group]:
        return None
    driver_corner = random.choice(safe[desired_group])

    inp = _render_input(all_gears, 0, driver_corner)
    gears, driver_idx = _extract_gears(inp)
    graph = _mesh_graph(gears)
    reached = _reachable(graph, driver_idx)
    if len(gears) != total:
        return None
    if len(reached) != active_count:
        return None
    if any(graph[idx] for idx in range(len(gears)) if idx not in reached):
        return None
    if _has_expansion_conflict(inp, gears, driver_idx, graph):
        return None
    spin = _driver_spin(gears[driver_idx])
    if (desired_group == "cw" and spin != 1) or (desired_group == "ccw" and spin != -1):
        return None

    out = main(inp)
    if np.array_equal(inp, out):
        return None
    if int(np.count_nonzero(out != inp)) < active_count + 1:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gear-rotation ARC input")
