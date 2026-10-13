import math
from collections import deque

import numpy as np


BACKGROUND = 8  # azure
TARGET = 5      # grey
RULER = 9       # maroon
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_RNG = np.random.default_rng()

# --------------- solver ---------------


def _connected_components(mask: np.ndarray) -> list[dict]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            arr = np.array(cells, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local[arr[:, 0] - r0, arr[:, 1] - c0] = True
            comps.append(
                {
                    "cells": [tuple(map(int, cell)) for cell in cells],
                    "cell_set": set(cells),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": local,
                }
            )
    return comps


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _scaled_pixels(mask: np.ndarray, bbox: tuple[int, int, int, int], scale: int) -> list[tuple[int, int]]:
    r0, c0, _r1, _c1 = bbox
    h, w = mask.shape
    top = _round_half_up(r0 + (h - h * scale) / 2.0)
    left = _round_half_up(c0 + (w - w * scale) / 2.0)

    pixels: list[tuple[int, int]] = []
    for pr, pc in np.argwhere(mask):
        base_r = top + int(pr) * scale
        base_c = left + int(pc) * scale
        for dr in range(scale):
            for dc in range(scale):
                pixels.append((base_r + dr, base_c + dc))
    return pixels


def _trace_ruler(
    grid: np.ndarray,
    target_cells: set[tuple[int, int]],
    start: tuple[int, int],
    dr: int,
    dc: int,
) -> tuple[tuple[int, int], ...] | None:
    sr, sc = start
    if (sr - dr, sc - dc) not in target_cells:
        return None

    step1 = (sr + dr, sc + dc)
    if 0 <= step1[0] < grid.shape[0] and 0 <= step1[1] < grid.shape[1]:
        if int(grid[step1]) != BACKGROUND:
            return None

    cells = [start]
    count = 1
    while count < 4:
        gap = (sr + (2 * count - 1) * dr, sc + (2 * count - 1) * dc)
        nxt = (sr + 2 * count * dr, sc + 2 * count * dc)

        if not (0 <= gap[0] < grid.shape[0] and 0 <= gap[1] < grid.shape[1]):
            break
        if int(grid[gap]) != BACKGROUND:
            return None
        if not (0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1]):
            break
        if int(grid[nxt]) == RULER:
            cells.append(nxt)
            count += 1
            continue
        if int(grid[nxt]) == BACKGROUND:
            break
        return None

    extra_gap = (sr + (2 * count - 1) * dr, sc + (2 * count - 1) * dc)
    extra_maroon = (sr + 2 * count * dr, sc + 2 * count * dc)
    if 0 <= extra_gap[0] < grid.shape[0] and 0 <= extra_gap[1] < grid.shape[1]:
        if int(grid[extra_gap]) != BACKGROUND:
            return None
    if 0 <= extra_maroon[0] < grid.shape[0] and 0 <= extra_maroon[1] < grid.shape[1]:
        if int(grid[extra_maroon]) == RULER:
            return None

    return tuple(cells)


def _extract_ruler(grid: np.ndarray, obj: dict) -> tuple[int, tuple[tuple[int, int], ...]]:
    adjacent_maroons: set[tuple[int, int]] = set()
    for r, c in obj["cells"]:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == RULER:
                adjacent_maroons.add((nr, nc))

    assert adjacent_maroons, "each target must touch a ruler"

    traced: dict[tuple[tuple[int, int], ...], int] = {}
    for start in adjacent_maroons:
        for dr, dc in DIRS4:
            ruler_cells = _trace_ruler(grid, obj["cell_set"], start, dr, dc)
            if ruler_cells is not None:
                traced[ruler_cells] = len(ruler_cells)

    assert len(traced) == 1, "ruler association must be unique"
    ruler_cells = next(iter(traced))
    scale = traced[ruler_cells]
    assert 1 <= scale <= 4
    return scale, ruler_cells


def main(grid: np.ndarray) -> np.ndarray:
    """Scale each grey target by the tick count of its adjacent maroon ruler."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = _connected_components(g == TARGET)
    assert objects, "expected at least one target object"

    parsed: list[tuple[dict, int]] = []
    for obj in objects:
        scale, _ruler = _extract_ruler(g, obj)
        parsed.append((obj, scale))

    out = np.where((g != TARGET) & (g != RULER), g, BACKGROUND)
    parsed.sort(key=lambda item: (item[0]["bbox"][0], item[0]["bbox"][1]))

    h, w = g.shape
    for obj, scale in parsed:
        for r, c in _scaled_pixels(obj["mask"], obj["bbox"], scale):
            if 0 <= r < h and 0 <= c < w and int(out[r, c]) == BACKGROUND:
                out[r, c] = TARGET
    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 27)
GRID_W_RANGE = (18, 27)
OBJECT_COUNT_RANGE = (2, 5)
MAX_SHAPE_SIDE = 3
MAX_SHAPE_AREA = 5
PLACE_TRIES = 120
POSITION_TRIES = 160
GEN_ATTEMPTS = 400


def _valid_contacts(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    cells = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    contacts: list[tuple[int, int, int, int]] = []
    for r, c in cells:
        for dr, dc in DIRS4:
            outside = (r + dr, c + dc)
            if outside in cells:
                continue
            adj = sum((outside[0] + rr, outside[1] + cc) in cells for rr, cc in DIRS4)
            if adj == 1:
                contacts.append((r, c, dr, dc))
    return contacts


def _random_shape() -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    for _ in range(80):
        h = int(_RNG.integers(1, MAX_SHAPE_SIDE + 1))
        w = int(_RNG.integers(1, MAX_SHAPE_SIDE + 1))
        area = int(_RNG.integers(1, min(MAX_SHAPE_AREA, h * w) + 1))
        cells = {(
            int(_RNG.integers(0, h)),
            int(_RNG.integers(0, w)),
        )}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(_RNG.integers(0, len(frontier)))])
        if len(cells) != area:
            continue

        arr = np.array(sorted(cells), dtype=int)
        r0, c0 = arr.min(axis=0)
        r1, c1 = arr.max(axis=0)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[arr[:, 0] - r0, arr[:, 1] - c0] = True
        contacts = _valid_contacts(mask)
        if contacts:
            return mask, contacts
    return None


def _make_spec(
    mask: np.ndarray,
    contact: tuple[int, int, int, int],
    scale: int,
    top: int,
    left: int,
    grid_h: int,
    grid_w: int,
) -> dict | None:
    h, w = mask.shape
    target_local = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    target_local_set = set(target_local)
    cr, cc, dr, dc = contact
    start = (cr + dr, cc + dc)

    maroons_local = [(start[0] + 2 * k * dr, start[1] + 2 * k * dc) for k in range(scale)]
    gaps_local = [(start[0] + (2 * k + 1) * dr, start[1] + (2 * k + 1) * dc) for k in range(scale - 1)]

    for idx, pos in enumerate(maroons_local + gaps_local):
        if pos in target_local_set:
            return None
        adj = sum((pos[0] + rr, pos[1] + cc2) in target_local_set for rr, cc2 in DIRS4)
        if idx == 0:
            if adj != 1:
                return None
        elif adj != 0:
            return None

    target_abs = [(top + r, left + c) for r, c in target_local]
    maroons_abs = [(top + r, left + c) for r, c in maroons_local]
    gaps_abs = [(top + r, left + c) for r, c in gaps_local]
    structure_abs = target_abs + maroons_abs + gaps_abs

    if not structure_abs:
        return None
    if any(not (0 <= r < grid_h and 0 <= c < grid_w) for r, c in structure_abs):
        return None
    if len(set(structure_abs)) != len(structure_abs):
        return None

    bbox = (top, left, top + h - 1, left + w - 1)
    scaled_abs = _scaled_pixels(mask, bbox, scale)
    if any(not (0 <= r < grid_h and 0 <= c < grid_w) for r, c in scaled_abs):
        return None

    return {
        "mask": mask,
        "bbox": bbox,
        "scale": scale,
        "target": target_abs,
        "maroons": maroons_abs,
        "gaps": gaps_abs,
        "structure": structure_abs,
        "scaled": scaled_abs,
    }


def _touches_blocked(spec: dict, blocked: set[tuple[int, int]]) -> bool:
    return any(cell in blocked for cell in spec["structure"])


def _add_blocked(spec: dict, blocked: set[tuple[int, int]], grid_h: int, grid_w: int) -> None:
    for r, c in spec["structure"]:
        blocked.add((r, c))
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_h and 0 <= nc < grid_w:
                blocked.add((nr, nc))


def _has_scaled_overlap(specs: list[dict]) -> bool:
    seen: set[tuple[int, int]] = set()
    for spec in specs:
        cur = set(spec["scaled"])
        if seen & cur:
            return True
        seen |= cur
    return False


def _gen_attempt() -> np.ndarray | None:
    grid_h = int(_RNG.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    grid_w = int(_RNG.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    n_objects = int(_RNG.integers(OBJECT_COUNT_RANGE[0], OBJECT_COUNT_RANGE[1] + 1))
    want_overlap = bool(_RNG.integers(0, 2))

    scales = [int(_RNG.integers(1, 5)) for _ in range(n_objects)]
    if want_overlap and max(scales) < 3:
        scales[int(_RNG.integers(0, n_objects))] = int(_RNG.integers(3, 5))

    blocked: set[tuple[int, int]] = set()
    specs: list[dict] = []
    scaled_seen: set[tuple[int, int]] = set()
    has_overlap = False
    overlap_idx = int(_RNG.integers(1, n_objects)) if want_overlap and n_objects >= 2 else -1

    for idx in range(n_objects):
        scale = scales[idx]
        placed = False
        for _ in range(PLACE_TRIES):
            shape_info = _random_shape()
            if shape_info is None:
                continue
            mask, contacts = shape_info

            horizontal = bool(_RNG.integers(0, 2))
            axis_contacts = [c for c in contacts if (c[2] == 0) == horizontal]
            if not axis_contacts:
                axis_contacts = contacts
            contact = axis_contacts[int(_RNG.integers(0, len(axis_contacts)))]

            h, w = mask.shape
            for _ in range(POSITION_TRIES):
                top = int(_RNG.integers(0, grid_h - h + 1))
                left = int(_RNG.integers(0, grid_w - w + 1))
                spec = _make_spec(mask, contact, scale, top, left, grid_h, grid_w)
                if spec is None or _touches_blocked(spec, blocked):
                    continue
                overlaps_now = any(cell in scaled_seen for cell in spec["scaled"])
                if not want_overlap and overlaps_now:
                    continue
                if want_overlap and idx == overlap_idx and not has_overlap and not overlaps_now:
                    continue
                specs.append(spec)
                _add_blocked(spec, blocked, grid_h, grid_w)
                scaled_seen.update(spec["scaled"])
                has_overlap = has_overlap or overlaps_now
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    if _has_scaled_overlap(specs) != want_overlap:
        return None

    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    for spec in specs:
        for r, c in spec["target"]:
            grid[r, c] = TARGET
        for r, c in spec["maroons"]:
            grid[r, c] = RULER

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
