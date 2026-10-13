from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

_ARC_ROOT = Path("/home/lixinyu/snap/new")
if str(_ARC_ROOT) not in sys.path:
    sys.path.append(str(_ARC_ROOT))

from solvers.arc_utils import Color, bbox_coords, components_mask, get_bg, in_bounds, neighbors4


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

SIDES = ("left", "right", "top", "bottom")
OPENING_FOR_SIDE = {
    "left": "right",
    "right": "left",
    "top": "down",
    "bottom": "up",
}
SEED_COLORS = (Color.BLUE, Color.RED, Color.GREEN)


def _edge_cells(bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "up":
        return [(r0, c) for c in range(c0, c1 + 1)]
    if side == "down":
        return [(r1, c) for c in range(c0, c1 + 1)]
    if side == "left":
        return [(r, c0) for r in range(r0, r1 + 1)]
    if side == "right":
        return [(r, c1) for r in range(r0, r1 + 1)]
    raise ValueError(side)


def _wall_cells_from_bbox(
    bbox: tuple[int, int, int, int], opening: str
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()

    if opening == "up":
        for r in range(r0, r1 + 1):
            cells.add((r, c0))
            cells.add((r, c1))
        for c in range(c0, c1 + 1):
            cells.add((r1, c))
    elif opening == "down":
        for r in range(r0, r1 + 1):
            cells.add((r, c0))
            cells.add((r, c1))
        for c in range(c0, c1 + 1):
            cells.add((r0, c))
    elif opening == "left":
        for c in range(c0, c1 + 1):
            cells.add((r0, c))
            cells.add((r1, c))
        for r in range(r0, r1 + 1):
            cells.add((r, c1))
    elif opening == "right":
        for c in range(c0, c1 + 1):
            cells.add((r0, c))
            cells.add((r1, c))
        for r in range(r0, r1 + 1):
            cells.add((r, c0))
    else:
        raise ValueError(opening)
    return cells


def _bbox_from_start(
    start: tuple[int, int], opening: str, height: int, width: int
) -> tuple[int, int, int, int]:
    r, c = start
    if opening == "up":
        return (r, c - width // 2, r + height - 1, c + width // 2)
    if opening == "down":
        return (r - height + 1, c - width // 2, r, c + width // 2)
    if opening == "left":
        return (r - height // 2, c, r + height // 2, c + width - 1)
    if opening == "right":
        return (r - height // 2, c - width + 1, r + height // 2, c)
    raise ValueError(opening)


def _feasible_dims(
    shape: tuple[int, int], start: tuple[int, int], opening: str
) -> list[tuple[int, int, int]]:
    h_lim, w_lim = shape
    r, c = start
    dims: list[tuple[int, int, int]] = []

    if opening in ("up", "down"):
        max_h = h_lim - r if opening == "up" else r + 1
        max_half_w = min(c, w_lim - 1 - c)
        max_w = 2 * max_half_w + 1
        for height in range(4, min(8, max_h) + 1):
            for width in range(5, min(11, max_w) + 1, 2):
                area = (height - 1) * (width - 2)
                dims.append((height, width, area))
    else:
        max_half_h = min(r, h_lim - 1 - r)
        max_h = 2 * max_half_h + 1
        max_w = w_lim - c if opening == "left" else c + 1
        for height in range(5, min(11, max_h) + 1, 2):
            for width in range(4, min(8, max_w) + 1):
                area = (height - 2) * (width - 1)
                dims.append((height, width, area))
    return [item for item in dims if item[2] >= 8]


def _flood_region_in_bbox(
    grid: np.ndarray, bbox: tuple[int, int, int, int], seed: tuple[int, int]
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    q = deque([seed])
    seen = {seed}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if nr < r0 or nr > r1 or nc < c0 or nc > c1:
                continue
            if (nr, nc) in seen or grid[nr, nc] == Color.GREY:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return seen


def _opening_midpoint(
    cavity: set[tuple[int, int]], bbox: tuple[int, int, int, int], opening: str
) -> tuple[int, int]:
    gap = [cell for cell in _edge_cells(bbox, opening) if cell in cavity]
    if opening in ("up", "down"):
        gap.sort(key=lambda rc: rc[1])
    else:
        gap.sort(key=lambda rc: rc[0])
    return gap[len(gap) // 2]


def _analyze_wall(grid: np.ndarray, coords: np.ndarray) -> dict:
    bbox = tuple(map(int, bbox_coords(coords)))
    r0, c0, r1, c1 = bbox
    edge_counts = {}
    for side in DIRS:
        edge = _edge_cells(bbox, side)
        edge_counts[side] = sum(int(grid[r, c] == Color.GREY) for r, c in edge)
    opening = min(edge_counts, key=edge_counts.get)

    seeds = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if grid[r, c] not in (Color.BLACK, Color.GREY)
    ]
    assert len(seeds) == 1
    seed = seeds[0]
    color = int(grid[seed])
    cavity = _flood_region_in_bbox(grid, bbox, seed)
    start = _opening_midpoint(cavity, bbox, opening)
    return {
        "bbox": bbox,
        "opening": opening,
        "direction": DIRS[opening],
        "seed": seed,
        "color": color,
        "cavity": cavity,
        "area": len(cavity),
        "start": start,
    }


def _solve(grid: np.ndarray, *, return_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    wall_components = components_mask(g == Color.GREY, connectivity=4)
    assert len(wall_components) == 3

    structures = [_analyze_wall(g, comp) for comp in wall_components]
    for info in structures:
        for r, c in info["cavity"]:
            out[r, c] = info["color"]

    rays = []
    for idx, info in enumerate(structures):
        rays.append(
            {
                "id": idx,
                "color": info["color"],
                "area": info["area"],
                "direction": info["direction"],
                "head": info["start"],
                "start": info["start"],
                "active": True,
            }
        )

    beam_owner: dict[tuple[int, int], int] = {}
    collisions = []
    max_steps = g.shape[0] * g.shape[1]

    for _ in range(max_steps):
        claims: dict[tuple[int, int], list[int]] = {}
        active_before = sum(int(ray["active"]) for ray in rays)
        if not active_before:
            break

        for ray in rays:
            if not ray["active"]:
                continue
            dr, dc = ray["direction"]
            nr = ray["head"][0] + dr
            nc = ray["head"][1] + dc
            if not in_bounds(g.shape, nr, nc):
                ray["active"] = False
                continue
            cell = (nr, nc)
            val = int(out[cell])
            if val == Color.GREY:
                ray["active"] = False
                continue
            if val != Color.BLACK and cell not in beam_owner:
                ray["active"] = False
                continue
            claims.setdefault(cell, []).append(ray["id"])

        if not claims:
            break

        losers: set[int] = set()
        winners: list[tuple[int, tuple[int, int]]] = []

        for cell, claimants in claims.items():
            incumbent = beam_owner.get(cell)
            contest = list(claimants)
            if incumbent is not None and incumbent not in contest:
                contest.append(incumbent)
            winner = max(contest, key=lambda rid: rays[rid]["area"])

            if len(claimants) > 1 or incumbent is not None:
                collisions.append(
                    {
                        "cell": cell,
                        "claimants": tuple(claimants),
                        "incumbent": incumbent,
                        "winner": winner,
                    }
                )

            if incumbent is not None and winner == incumbent:
                losers.update(claimants)
                continue

            for rid in claimants:
                if rid != winner:
                    losers.add(rid)
            winners.append((winner, cell))

        for rid in losers:
            rays[rid]["active"] = False

        moved = False
        for rid, cell in winners:
            if not rays[rid]["active"]:
                continue
            out[cell] = rays[rid]["color"]
            beam_owner[cell] = rid
            rays[rid]["head"] = cell
            moved = True

        if not moved:
            break

    if return_trace:
        return out, {"structures": structures, "collisions": collisions, "rays": rays}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood-fill each semi-enclosed cavity, then emit opening-aligned rays whose collisions are resolved by filled area.

    Input structure:
    - The background is black.
    - Three grey one-pixel-thick semi-enclosures appear as U/C-like wall components, each with one colored seed inside.
    - The three seed colors are blue, red, and green, and each wall's cavity area is different.

    Rule:
    - For each grey wall, flood-fill every black cell in its cavity from the interior seed, restricted to that wall's bbox.
    - Infer the wall opening from the bbox edge with the missing grey side and use the midpoint of that opening as the ray source.
    - Extend one 1-pixel ray outward in the opening direction using the cavity's fill color.
    - If multiple rays try to enter the same cell, or a later ray reaches an already-painted beam cell, the ray tied to the larger filled cavity area wins that cell and continues while smaller rays stop before it.
    - Grey wall pixels stay unchanged.

    Key insight:
    - The flood-fill step creates both the ray color and the hidden priority value: cavity area decides which ray survives a collision.
    """
    return _solve(np.asarray(grid, dtype=int))


def _pairwise_min_gap(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    best = 10**9
    for r1, c1 in a:
        for r2, c2 in b:
            best = min(best, max(abs(r1 - r2), abs(c1 - c2)))
    return best


def _line_start_for_side(side: str, collision: tuple[int, int], distance: int) -> tuple[int, int]:
    r, c = collision
    if side == "left":
        return (r, c - distance)
    if side == "right":
        return (r, c + distance)
    if side == "top":
        return (r - distance, c)
    if side == "bottom":
        return (r + distance, c)
    raise ValueError(side)


def _sample_layout() -> tuple[int, int, tuple[int, int], list[str], dict[str, int], str]:
    height = random.randint(20, 28)
    width = random.randint(20, 28)
    collision = (
        random.randint(7, height - 8),
        random.randint(7, width - 8),
    )
    sides = random.sample(list(SIDES), 3)
    mode = random.choice(("sync", "staggered"))
    winner_side = random.choice(sides)
    base = random.randint(4, 6)
    distances: dict[str, int] = {}
    for side in sides:
        if mode == "sync":
            distances[side] = base
        elif side == winner_side:
            distances[side] = base
        else:
            distances[side] = base + random.randint(0, 2)
    return height, width, collision, sides, distances, winner_side


def _choose_structure_specs(
    shape: tuple[int, int], collision: tuple[int, int], sides: list[str], distances: dict[str, int], winner_side: str
) -> dict[str, dict] | None:
    starts = {side: _line_start_for_side(side, collision, distances[side]) for side in sides}
    dim_options = {}
    for side in sides:
        opening = OPENING_FOR_SIDE[side]
        options = _feasible_dims(shape, starts[side], opening)
        if not options:
            return None
        random.shuffle(options)
        dim_options[side] = options

    for _ in range(250):
        specs = {}
        areas = []
        occupied_sets = []
        for side in sides:
            opening = OPENING_FOR_SIDE[side]
            height, width, area = random.choice(dim_options[side])
            bbox = _bbox_from_start(starts[side], opening, height, width)
            wall = _wall_cells_from_bbox(bbox, opening)
            cavity = {
                (r, c)
                for r in range(bbox[0], bbox[2] + 1)
                for c in range(bbox[1], bbox[3] + 1)
                if (r, c) not in wall
            }
            specs[side] = {
                "opening": opening,
                "bbox": bbox,
                "wall": wall,
                "cavity": cavity,
                "start": starts[side],
                "area": area,
            }
            areas.append(area)
            occupied_sets.append(wall | cavity)

        if len(set(areas)) != 3:
            continue
        if specs[winner_side]["area"] != max(areas):
            continue
        ok = True
        for i in range(len(sides)):
            for j in range(i + 1, len(sides)):
                if _pairwise_min_gap(occupied_sets[i], occupied_sets[j]) < 4:
                    ok = False
        if ok:
            return specs
    return None


def _paint_structure(grid: np.ndarray, spec: dict, color: int) -> None:
    for r, c in spec["wall"]:
        grid[r, c] = Color.GREY
    candidates = [cell for cell in spec["cavity"] if cell != spec["start"]]
    seed = random.choice(candidates)
    grid[seed] = color


def _semantic_checks(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    seed_cells = [(r, c) for r, c in np.argwhere(np.isin(grid, SEED_COLORS))]
    if len(seed_cells) != 3:
        return False
    if set(int(grid[r, c]) for r, c in seed_cells) != set(SEED_COLORS):
        return False

    try:
        out, trace = _solve(grid, return_trace=True)
    except Exception:
        return False

    structures = trace["structures"]
    if len(structures) != 3:
        return False
    areas = [info["area"] for info in structures]
    if len(set(areas)) != 3:
        return False
    if len({info["opening"] for info in structures}) == 1:
        return False
    if not trace["collisions"]:
        return False

    ray_pixels = np.argwhere((out != grid) & (grid == Color.BLACK))
    if len(ray_pixels) < 6:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        height, width, collision, sides, distances, winner_side = _sample_layout()
        specs = _choose_structure_specs((height, width), collision, sides, distances, winner_side)
        if specs is None:
            continue

        grid = np.zeros((height, width), dtype=int)
        color_order = list(SEED_COLORS)
        random.shuffle(color_order)
        for side, color in zip(sides, color_order):
            _paint_structure(grid, specs[side], color)

        if _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
