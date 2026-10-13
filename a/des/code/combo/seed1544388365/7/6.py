from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


BLACK = 0
MAGENTA = 6
INPUT_COLORS = (1, 2, 3, 4, 5, 7, 8, 9)
GENERATE_RETRIES = 400


_ARROW_SPECS: dict[tuple[int, int], dict[str, Any]] = {
    (0, 1): {
        "cells": ((0, 0), (1, 0), (1, 1)),
        "start": (1, 1),
        "missing": (0, 1),
    },
    (0, -1): {
        "cells": ((0, 1), (1, 0), (1, 1)),
        "start": (1, 0),
        "missing": (0, 0),
    },
    (1, 0): {
        "cells": ((0, 0), (0, 1), (1, 0)),
        "start": (1, 0),
        "missing": (1, 1),
    },
    (-1, 0): {
        "cells": ((0, 0), (0, 1), (1, 1)),
        "start": (0, 1),
        "missing": (1, 0),
    },
}
_DIRECTIONS = tuple(_ARROW_SPECS.keys())


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(coords)
    return parts


def _analyze_arrow(coords: list[tuple[int, int]]) -> dict[str, Any] | None:
    if len(coords) != 3:
        return None
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 != 1 or c1 - c0 != 1:
        return None

    mask = np.zeros((2, 2), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    if int(mask.sum()) != 3:
        return None

    missing = np.argwhere(~mask)
    if missing.shape != (1, 2):
        return None
    miss = (int(missing[0, 0]), int(missing[0, 1]))

    direction = None
    spec = None
    for cand_dir, cand_spec in _ARROW_SPECS.items():
        if cand_spec["missing"] == miss:
            direction = cand_dir
            spec = cand_spec
            break
    if direction is None or spec is None:
        return None

    sr = r0 + int(spec["start"][0])
    sc = c0 + int(spec["start"][1])
    return {
        "direction": direction,
        "start": (sr, sc),
        "bbox": (r0, c0, r1, c1),
        "coords": tuple(sorted((int(r), int(c)) for r, c in coords)),
    }


def _find_arrows(grid: np.ndarray) -> list[dict[str, Any]]:
    arrows: list[dict[str, Any]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == BLACK:
            continue
        for coords in _components(grid == color):
            arrow = _analyze_arrow(coords)
            if arrow is None:
                continue
            arrow["color"] = color
            arrows.append(arrow)
    arrows.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    return arrows


def _ray_cells(start: tuple[int, int], direction: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch each 3-pixel L arrow along the direction encoded by its missing corner.

    Missing-corner mapping:
    - top-right missing -> move right
    - top-left missing -> move left
    - bottom-right missing -> move down
    - bottom-left missing -> move up

    The ray is simulated against the original input:
    - black -> paint a same-color trail and continue
    - same color -> absorb and continue
    - different non-black -> stop and replace that pixel by magenta
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    h, w = g.shape

    for arrow in _find_arrows(g):
        color = int(arrow["color"])
        dr, dc = arrow["direction"]
        r, c = arrow["start"]
        nr, nc = r + dr, c + dc
        while 0 <= nr < h and 0 <= nc < w:
            cell = int(g[nr, nc])
            if cell == BLACK:
                out[nr, nc] = color
                nr += dr
                nc += dc
                continue
            if cell == color:
                nr += dr
                nc += dc
                continue
            out[nr, nc] = MAGENTA
            break

    return out


def _reserve_with_moat(reserved: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = reserved.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                reserved[rr, cc] = True


def _sample_scenarios(rng: np.random.Generator, n_arrows: int) -> list[str]:
    scenarios: list[str] = []
    if n_arrows >= 1:
        scenarios.append(str(rng.choice(("trail", "absorb_boundary"))))
    if n_arrows >= 2:
        scenarios.append(str(rng.choice(("collide", "absorb_collide"))))
    if n_arrows >= 3:
        scenarios.append(str(rng.choice(("absorb_boundary", "absorb_collide"))))
    while len(scenarios) < n_arrows:
        scenarios.append(str(rng.choice(("trail", "absorb_boundary", "collide", "absorb_collide"))))
    rng.shuffle(scenarios)
    return scenarios


def _plan_ray_contents(
    rng: np.random.Generator,
    ray_len: int,
    scenario: str,
) -> tuple[list[int], int | None] | None:
    if ray_len < 2:
        return None

    want_absorb = scenario in {"absorb_boundary", "absorb_collide"}
    want_collision = scenario in {"collide", "absorb_collide"}

    if not want_absorb and not want_collision:
        return ([], None)

    if not want_absorb and want_collision:
        blocker = int(rng.integers(1, ray_len))
        return ([], blocker)

    max_absorb = min(3, ray_len // 2)
    if want_collision:
        max_absorb = min(max_absorb, (ray_len - 1) // 2)
    if max_absorb <= 0:
        return None

    absorb_count = int(rng.integers(1, max_absorb + 1))
    if want_collision:
        min_needed = 2 * absorb_count + 1
    else:
        min_needed = 2 * absorb_count
    if ray_len < min_needed:
        return None

    slack = ray_len - min_needed
    extras = [0] * (absorb_count + 1)
    for _ in range(slack):
        extras[int(rng.integers(0, absorb_count + 1))] += 1

    absorb_positions: list[int] = []
    pos = 1 + extras[0]
    absorb_positions.append(pos)
    for idx in range(1, absorb_count):
        pos += 2 + extras[idx]
        absorb_positions.append(pos)

    blocker = None
    if want_collision:
        blocker = absorb_positions[-1] + 1 + extras[-1]
    return (absorb_positions, blocker)


def _can_use_cells(reserved: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    return all(not bool(reserved[r, c]) for r, c in cells)


def _place_arrow(
    rng: np.random.Generator,
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
    direction: tuple[int, int],
    scenario: str,
) -> bool:
    h, w = grid.shape
    spec = _ARROW_SPECS[direction]

    candidates: list[tuple[int, int, list[tuple[int, int]]]] = []
    for r0 in range(h - 1):
        for c0 in range(w - 1):
            arrow_cells = [(r0 + dr, c0 + dc) for dr, dc in spec["cells"]]
            start = (r0 + int(spec["start"][0]), c0 + int(spec["start"][1]))
            ray = _ray_cells(start, direction, h, w)
            if len(ray) < 2:
                continue
            if not _can_use_cells(reserved, arrow_cells):
                continue
            if not _can_use_cells(reserved, ray):
                continue
            candidates.append((r0, c0, ray))

    if not candidates:
        return False

    rng.shuffle(candidates)
    for r0, c0, ray in candidates:
        plan = _plan_ray_contents(rng, len(ray), scenario)
        if plan is None:
            continue
        absorb_positions, blocker_pos = plan
        blocker_color = None
        if blocker_pos is not None:
            blocker_choices = [cand for cand in INPUT_COLORS if int(cand) != color]
            blocker_color = int(rng.choice(blocker_choices))

        arrow_cells = [(r0 + dr, c0 + dc) for dr, dc in spec["cells"]]
        occupied = list(arrow_cells)
        for idx in absorb_positions:
            occupied.append(ray[idx])
        if blocker_pos is not None:
            occupied.append(ray[blocker_pos])

        if any(int(grid[r, c]) != BLACK for r, c in occupied):
            continue

        for r, c in arrow_cells:
            grid[r, c] = color
        for idx in absorb_positions:
            rr, cc = ray[idx]
            grid[rr, cc] = color
        if blocker_pos is not None and blocker_color is not None:
            rr, cc = ray[blocker_pos]
            grid[rr, cc] = blocker_color

        _reserve_with_moat(reserved, arrow_cells + ray)
        return True

    return False


def _place_extra_single(
    rng: np.random.Generator,
    grid: np.ndarray,
    reserved: np.ndarray,
) -> bool:
    h, w = grid.shape
    cells = [(r, c) for r in range(h) for c in range(w) if not bool(reserved[r, c]) and int(grid[r, c]) == BLACK]
    if not cells:
        return False
    r, c = cells[int(rng.integers(0, len(cells)))]
    grid[r, c] = int(rng.choice(INPUT_COLORS))
    _reserve_with_moat(reserved, [(r, c)])
    return True


def _valid_structure(grid: np.ndarray, expected_arrows: int) -> bool:
    if int(np.any(grid == MAGENTA)):
        return False

    arrows = _find_arrows(grid)
    if len(arrows) != expected_arrows:
        return False

    arrow_coords = {tuple(sorted(arrow["coords"])) for arrow in arrows}
    for color in np.unique(grid):
        color = int(color)
        if color == BLACK:
            continue
        for comp in _components(grid == color):
            if len(comp) == 1:
                continue
            comp_key = tuple(sorted((int(r), int(c)) for r, c in comp))
            if len(comp) == 3 and comp_key in arrow_coords:
                continue
            return False

    out = main(grid)
    if not np.any(out != grid):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(12, 21))
    w = int(rng.integers(12, 21))
    n_arrows = int(rng.integers(2, 6))

    colors = [int(v) for v in rng.choice(np.array(INPUT_COLORS, dtype=int), size=n_arrows, replace=False)]
    directions = list(rng.permutation(np.array(_DIRECTIONS, dtype=object))[: min(n_arrows, len(_DIRECTIONS))])
    while len(directions) < n_arrows:
        directions.append(_DIRECTIONS[int(rng.integers(0, len(_DIRECTIONS)))])
    rng.shuffle(directions)

    scenarios = _sample_scenarios(rng, n_arrows)
    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    order = list(range(n_arrows))
    rng.shuffle(order)
    for idx in order:
        if not _place_arrow(
            rng=rng,
            grid=grid,
            reserved=reserved,
            color=colors[idx],
            direction=tuple(int(v) for v in directions[idx]),
            scenario=scenarios[idx],
        ):
            return None

    n_extra = int(rng.integers(1, 5))
    placed_extra = 0
    for _ in range(n_extra):
        placed_extra += int(_place_extra_single(rng, grid, reserved))
    if placed_extra == 0:
        return None

    if not _valid_structure(grid, n_arrows):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle instance")
