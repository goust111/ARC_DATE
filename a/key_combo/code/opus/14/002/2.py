from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
DEFECT_COLORS = (2, 3, 4)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _orbit(h: int, w: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            {
                (r, c),
                (h - 1 - r, c),
                (r, w - 1 - c),
                (h - 1 - r, w - 1 - c),
            }
        )
    )


def _black_components(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    h, w = grid.shape
    comp_ids = np.full((h, w), -1, dtype=int)
    comps: list[dict[str, object]] = []

    for sr in range(h):
        for sc in range(w):
            if grid[sr, sc] != BG or comp_ids[sr, sc] >= 0:
                continue

            cid = len(comps)
            q = deque([(sr, sc)])
            comp_ids[sr, sc] = cid
            cells: list[tuple[int, int]] = []
            touches_border = False

            while q:
                r, c = q.popleft()
                cells.append((r, c))
                if r in (0, h - 1) or c in (0, w - 1):
                    touches_border = True
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if _in_bounds((h, w), nr, nc) and grid[nr, nc] == BG and comp_ids[nr, nc] < 0:
                        comp_ids[nr, nc] = cid
                        q.append((nr, nc))

            comps.append({"cells": cells, "touches_border": touches_border})

    return comp_ids, comps


def _adjacent_enclosed_components(
    repaired: np.ndarray,
    comp_ids: np.ndarray,
    comps: list[dict[str, object]],
    r: int,
    c: int,
) -> set[int]:
    ids = set()
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if _in_bounds(repaired.shape, nr, nc) and repaired[nr, nc] == BG:
            cid = int(comp_ids[nr, nc])
            if not comps[cid]["touches_border"]:
                ids.add(cid)
    return ids


def _find_defects(grid: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    seen_orbits: set[tuple[tuple[int, int], ...]] = set()
    defects: list[tuple[int, int, int]] = []

    for r in range(h):
        for c in range(w):
            orb = _orbit(h, w, r, c)
            if orb in seen_orbits:
                continue
            seen_orbits.add(orb)

            values = [int(grid[rr, cc]) for rr, cc in orb]
            special_indices = [idx for idx, value in enumerate(values) if value not in (BG, BLUE)]
            if not special_indices:
                continue

            if len(orb) <= 1 or len(special_indices) != 1:
                continue

            special_idx = special_indices[0]
            other_values = [value for idx, value in enumerate(values) if idx != special_idx]
            if other_values and all(value == BLUE for value in other_values):
                rr, cc = orb[special_idx]
                defects.append((rr, cc, values[special_idx]))

    return defects


def main(grid: np.ndarray) -> np.ndarray:
    """Use symmetric defect pixels as fill seeds, flood their enclosed black regions, then restore perfect blue symmetry.

    Input structure:
    - The grid has black background with a blue foreground pattern that is bilaterally symmetric about the grid center.
    - A small number of blue foreground pixels have been replaced by distinct non-black, non-blue defect colors.
    - Each defect lies on the boundary of a black enclosed region separated by the blue pattern.

    Rule:
    - Find the defect pixels by comparing each cell with its horizontal, vertical, and 180-degree mirror orbit.
    - Temporarily repair every defect back to blue, then for each defect fill the unique adjacent enclosed black component with that defect's original color.
    - Leave black regions without a defect unchanged, and finally keep every defect position blue.

    Key insight:
    - Each wrong-colored pixel simultaneously signals where symmetry is broken and which enclosed region should be painted, while the symmetric blue pattern itself acts as the flood-fill boundary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    defects = _find_defects(g)
    assert defects, "expected at least one symmetry-breaking defect"

    repaired = g.copy()
    for r, c, _color in defects:
        repaired[r, c] = BLUE

    comp_ids, comps = _black_components(repaired)
    out = repaired.copy()
    used_components: set[int] = set()

    for r, c, color in defects:
        adjacent = _adjacent_enclosed_components(repaired, comp_ids, comps, r, c)
        assert len(adjacent) == 1, "each defect must border exactly one enclosed region"
        cid = next(iter(adjacent))
        assert cid not in used_components, "different defects must seed different enclosed regions"
        used_components.add(cid)
        for rr, cc in comps[cid]["cells"]:
            out[rr, cc] = color

    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    grid[r0, c0 : c1 + 1] = BLUE
    grid[r1, c0 : c1 + 1] = BLUE
    grid[r0 : r1 + 1, c0] = BLUE
    grid[r0 : r1 + 1, c1] = BLUE


def _build_base_grid(h: int, w: int, mode: str, r0: int, c0: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    r1 = h - 1 - r0
    c1 = w - 1 - c0
    _draw_frame(grid, r0, c0, r1, c1)

    if "v" in mode:
        grid[r0 : r1 + 1, w // 2] = BLUE
    if "h" in mode:
        grid[h // 2, c0 : c1 + 1] = BLUE

    return grid


def _boundary_candidates(
    grid: np.ndarray,
    comp_ids: np.ndarray,
    comps: list[dict[str, object]],
    cid: int,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    candidates = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BLUE or len(_orbit(h, w, r, c)) <= 1:
                continue
            adjacent = _adjacent_enclosed_components(grid, comp_ids, comps, r, c)
            if adjacent == {cid}:
                candidates.append((r, c))
    return candidates


def _choose_defect_positions(
    grid: np.ndarray,
    comp_ids: np.ndarray,
    comps: list[dict[str, object]],
    chamber_ids: list[int],
) -> dict[int, tuple[int, int]] | None:
    candidate_map = {cid: _boundary_candidates(grid, comp_ids, comps, cid) for cid in chamber_ids}
    if any(not candidate_map[cid] for cid in chamber_ids):
        return None

    order = sorted(chamber_ids, key=lambda cid: len(candidate_map[cid]))
    chosen: dict[int, tuple[int, int]] = {}
    used_orbits: set[tuple[tuple[int, int], ...]] = set()
    h, w = grid.shape

    def backtrack(idx: int) -> bool:
        if idx == len(order):
            return True
        cid = order[idx]
        options = candidate_map[cid][:]
        random.shuffle(options)
        for pos in options:
            orb = _orbit(h, w, pos[0], pos[1])
            if orb in used_orbits:
                continue
            used_orbits.add(orb)
            chosen[cid] = pos
            if backtrack(idx + 1):
                return True
            used_orbits.remove(orb)
            del chosen[cid]
        return False

    return chosen if backtrack(0) else None


def _expected_output(
    base: np.ndarray,
    comps: list[dict[str, object]],
    positions: dict[int, tuple[int, int]],
    colors: dict[int, int],
) -> np.ndarray:
    out = base.copy()
    for cid, color in colors.items():
        for r, c in comps[cid]["cells"]:
            out[r, c] = color
    for pos in positions.values():
        out[pos] = BLUE
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randrange(11, 20, 2)
    w = random.randrange(11, 20, 2)

    defect_count = random.choice((2, 3))
    if defect_count == 3:
        mode = "vh"
    else:
        mode = random.choice(("v", "h", "vh"))

    max_r0 = h // 2 - 3
    max_c0 = w // 2 - 3
    if max_r0 < 1 or max_c0 < 1:
        return None
    r0 = random.randint(1, max_r0)
    c0 = random.randint(1, max_c0)

    base = _build_base_grid(h, w, mode, r0, c0)
    comp_ids, comps = _black_components(base)
    chamber_ids = [cid for cid, comp in enumerate(comps) if not comp["touches_border"]]

    if len(chamber_ids) < defect_count:
        return None
    selected_chambers = random.sample(chamber_ids, defect_count)
    positions = _choose_defect_positions(base, comp_ids, comps, selected_chambers)
    if positions is None:
        return None

    defect_palette = random.sample(list(DEFECT_COLORS), defect_count)
    colors = {cid: color for cid, color in zip(selected_chambers, defect_palette)}

    inp = base.copy()
    for cid, pos in positions.items():
        inp[pos] = colors[cid]

    expected = _expected_output(base, comps, positions, colors)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetric defect fill puzzle")
