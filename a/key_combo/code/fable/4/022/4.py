from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = tuple(range(10))

VERTICAL_RAY_DIMS = ((1, 1), (1, 2), (2, 1), (2, 2), (2, 3))
HORIZONTAL_RAY_DIMS = ((1, 1), (2, 1), (1, 2), (2, 2), (3, 2))
OFF_RAY_DIMS = ((1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2))

GRID_MIN = 14
GRID_MAX = 24
GEN_RETRIES = 600
PLACEMENT_RETRIES = 200


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.asarray(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": int(coords.shape[0]),
                    "bbox": _bbox(coords),
                }
            )
    return comps


def _line3_info(coords: np.ndarray) -> tuple[tuple[int, int], str] | None:
    if int(coords.shape[0]) != 3:
        return None
    rows = sorted(int(r) for r in coords[:, 0])
    cols = sorted(int(c) for c in coords[:, 1])
    if len(set(rows)) == 1 and cols == list(range(cols[0], cols[0] + 3)):
        return (rows[0], cols[1]), "h"
    if len(set(cols)) == 1 and rows == list(range(rows[0], rows[0] + 3)):
        return (rows[1], cols[0]), "v"
    return None


def _find_bar_and_control(components: list[dict]) -> tuple[int, int, tuple[int, int], tuple[int, int]]:
    singleton_at = {
        (int(comp["coords"][0, 0]), int(comp["coords"][0, 1])): idx
        for idx, comp in enumerate(components)
        if int(comp["area"]) == 1
    }
    candidates: list[tuple[int, int, tuple[int, int], tuple[int, int]]] = []
    for idx, comp in enumerate(components):
        info = _line3_info(np.asarray(comp["coords"], dtype=int))
        if info is None:
            continue
        midpoint, _axis = info
        bar_cells = {tuple(map(int, rc)) for rc in comp["coords"]}
        control_ids = []
        for dr, dc in DIRS4:
            pos = (midpoint[0] + dr, midpoint[1] + dc)
            if pos in bar_cells:
                continue
            if pos in singleton_at:
                control_ids.append(singleton_at[pos])
        control_ids = sorted(set(control_ids))
        if len(control_ids) != 1:
            continue
        control_idx = control_ids[0]
        ctrl_r, ctrl_c = map(int, components[control_idx]["coords"][0])
        direction = (midpoint[0] - ctrl_r, midpoint[1] - ctrl_c)
        if direction in DIRS4:
            candidates.append((idx, control_idx, midpoint, direction))
    assert len(candidates) == 1, f"expected exactly one marker-bar/control pair, found {len(candidates)}"
    return candidates[0]


def _hits_ray(coords: np.ndarray, midpoint: tuple[int, int], direction: tuple[int, int]) -> bool:
    mr, mc = midpoint
    dr, dc = direction
    if dr != 0:
        return any(int(c) == mc and (int(r) - mr) * dr > 0 for r, c in coords)
    return any(int(r) == mr and (int(c) - mc) * dc > 0 for r, c in coords)


def _ray_distance(coords: np.ndarray, midpoint: tuple[int, int], direction: tuple[int, int]) -> int:
    mr, mc = midpoint
    dr, dc = direction
    dists = []
    if dr != 0:
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if cc == mc and (rr - mr) * dr > 0:
                dists.append(abs(rr - mr))
    else:
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if rr == mr and (cc - mc) * dc > 0:
                dists.append(abs(cc - mc))
    assert dists
    return min(dists)


def _can_occupy(grid: np.ndarray, coords: np.ndarray, bg: int) -> bool:
    h, w = grid.shape
    if np.any(coords[:, 0] < 0) or np.any(coords[:, 0] >= h) or np.any(coords[:, 1] < 0) or np.any(coords[:, 1] >= w):
        return False
    return bool(np.all(grid[coords[:, 0], coords[:, 1]] == bg))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push only the control-colored objects that lie on the ray cast away from the control pixel.

    Input structure:
    - The background is the dominant color.
    - One straight 3-cell marker bar has a singleton control pixel attached to its midpoint.
    - The remaining foreground consists of separated 4-connected objects in several colors.

    Rule:
    - Read the push direction from the bar midpoint away from the control pixel.
    - Objects in the control pixel's color are movable; all other foreground stays fixed.
    - Among those movable objects, only the ones intersecting the outgoing ray are translated.
    - Process the ray-hit movable objects from front to back, sliding each one until the next step
      would hit the grid edge, a fixed object, or an already-settled movable object.

    Key insight:
    - The control pixel encodes two things at once: its position selects the push direction, and
      its color selects which objects the ray-driven compaction is allowed to move.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = _find_components(g, bg)
    bar_idx, control_idx, midpoint, direction = _find_bar_and_control(components)
    movable_color = int(components[control_idx]["color"])

    ray_objects = []
    for idx, comp in enumerate(components):
        if idx in (bar_idx, control_idx):
            continue
        if int(comp["color"]) != movable_color:
            continue
        if _hits_ray(np.asarray(comp["coords"], dtype=int), midpoint, direction):
            ray_objects.append(comp)

    ordered = sorted(ray_objects, key=lambda comp: _ray_distance(comp["coords"], midpoint, direction), reverse=True)
    out = g.copy()
    step = np.asarray(direction, dtype=int)

    for comp in ordered:
        coords = np.asarray(comp["coords"], dtype=int)
        color = int(comp["color"])
        out[coords[:, 0], coords[:, 1]] = bg
        shifted = coords.copy()
        while True:
            trial = shifted + step
            if not _can_occupy(out, trial, bg):
                break
            shifted = trial
        out[shifted[:, 0], shifted[:, 1]] = color

    return out


def _interval_overlap(start: int, length: int, interval: tuple[int, int]) -> bool:
    end = start + length - 1
    lo, hi = interval
    return not (end < lo or start > hi)


def _sample_ray_spec(vertical: bool) -> dict:
    h, w = random.choice(VERTICAL_RAY_DIMS if vertical else HORIZONTAL_RAY_DIMS)
    anchor = random.randrange(w if vertical else h)
    return {
        "h": h,
        "w": w,
        "anchor": anchor,
        "length": h if vertical else w,
    }


def _ray_top_left(
    midpoint: tuple[int, int],
    direction: tuple[int, int],
    dist: int,
    spec: dict,
) -> tuple[int, int]:
    mr, mc = midpoint
    dr, dc = direction
    h = int(spec["h"])
    w = int(spec["w"])
    anchor = int(spec["anchor"])
    if dr == 1:
        return mr + dist, mc - anchor
    if dr == -1:
        bottom = mr - dist
        return bottom - h + 1, mc - anchor
    if dc == 1:
        return mr - anchor, mc + dist
    right = mc - dist
    return mr - anchor, right - w + 1


def _fits_empty(grid: np.ndarray, bg: int, top: int, left: int, h: int, w: int) -> bool:
    gh, gw = grid.shape
    if top < 0 or left < 0 or top + h > gh or left + w > gw:
        return False
    return bool(np.all(grid[top : top + h, left : left + w] == bg))


def _expanded_has_fg(grid: np.ndarray, bg: int, top: int, left: int, h: int, w: int, pad: int) -> bool:
    gh, gw = grid.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(gh, top + h + pad)
    c1 = min(gw, left + w + pad)
    return bool(np.any(grid[r0:r1, c0:c1] != bg))


def _place_random_rect(
    grid: np.ndarray,
    bg: int,
    color: int,
    dims: tuple[tuple[int, int], ...],
    *,
    corridor_axis: str,
    corridor_interval: tuple[int, int],
    pad: int = 1,
) -> bool:
    gh, gw = grid.shape
    for _ in range(PLACEMENT_RETRIES):
        h, w = random.choice(dims)
        top = random.randint(0, gh - h)
        left = random.randint(0, gw - w)
        if corridor_axis == "col" and _interval_overlap(left, w, corridor_interval):
            continue
        if corridor_axis == "row" and _interval_overlap(top, h, corridor_interval):
            continue
        if _expanded_has_fg(grid, bg, top, left, h, w, pad):
            continue
        grid[top : top + h, left : left + w] = color
        return True
    return False


def _count_ray_moves(inp: np.ndarray, out: np.ndarray) -> int:
    return int(np.count_nonzero(inp != out))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    bg = random.choice(PALETTE)

    direction = random.choice(DIRS4)
    vertical = direction[0] != 0
    ray_count = random.randint(2, 4)
    offray_movable_count = random.randint(1, 3)
    extra_obstacle_count = random.randint(1, 3)
    stopper_on_ray = random.random() < 0.5

    free_colors = [c for c in PALETTE if c != bg]
    bar_color, movable_color = random.sample(free_colors, 2)
    obstacle_palette = [c for c in free_colors if c not in {bar_color, movable_color}]

    ray_specs = [_sample_ray_spec(vertical) for _ in range(ray_count)]
    stopper_spec = _sample_ray_spec(vertical) if stopper_on_ray else None
    on_ray_specs = ray_specs + ([stopper_spec] if stopper_spec is not None else [])

    if vertical:
        left_extent = max(int(spec["anchor"]) for spec in on_ray_specs)
        right_extent = max(int(spec["w"]) - 1 - int(spec["anchor"]) for spec in on_ray_specs)
        corridor_axis = "col"
        corridor_half = (left_extent, right_extent)
    else:
        up_extent = max(int(spec["anchor"]) for spec in on_ray_specs)
        down_extent = max(int(spec["h"]) - 1 - int(spec["anchor"]) for spec in on_ray_specs)
        corridor_axis = "row"
        corridor_half = (up_extent, down_extent)

    back_gap = random.randint(1, 3)
    between_gaps = [random.randint(1, 3) for _ in range(ray_count - 1)]
    front_gap = random.randint(1, 4)

    dists: list[int] = []
    dist = back_gap
    for i, spec in enumerate(ray_specs):
        dists.append(dist)
        dist += int(spec["length"])
        if i < ray_count - 1:
            dist += between_gaps[i]

    if stopper_spec is not None:
        stopper_dist = dist + front_gap
        space_needed = stopper_dist + int(stopper_spec["length"]) - 1
    else:
        stopper_dist = None
        space_needed = dist + front_gap - 1

    if vertical:
        mc_low = max(1, corridor_half[0])
        mc_high = min(w - 2, w - 1 - corridor_half[1])
        if direction[0] == 1:
            mr_low = 1
            mr_high = min(h - 2, h - 1 - space_needed)
        else:
            mr_low = max(1, space_needed)
            mr_high = h - 2
        if mc_low > mc_high or mr_low > mr_high:
            return None
        midpoint = (random.randint(mr_low, mr_high), random.randint(mc_low, mc_high))
    else:
        mr_low = max(1, corridor_half[0])
        mr_high = min(h - 2, h - 1 - corridor_half[1])
        if direction[1] == 1:
            mc_low = 1
            mc_high = min(w - 2, w - 1 - space_needed)
        else:
            mc_low = max(1, space_needed)
            mc_high = w - 2
        if mr_low > mr_high or mc_low > mc_high:
            return None
        midpoint = (random.randint(mr_low, mr_high), random.randint(mc_low, mc_high))

    grid = np.full((h, w), bg, dtype=int)
    mr, mc = midpoint
    dr, dc = direction
    control = (mr - dr, mc - dc)

    if vertical:
        grid[mr, mc - 1 : mc + 2] = bar_color
    else:
        grid[mr - 1 : mr + 2, mc] = bar_color
    grid[control] = movable_color

    corridor_interval = (
        mc - corridor_half[0],
        mc + corridor_half[1],
    ) if vertical else (
        mr - corridor_half[0],
        mr + corridor_half[1],
    )

    for spec, dist_i in zip(ray_specs, dists):
        top, left = _ray_top_left(midpoint, direction, dist_i, spec)
        hh = int(spec["h"])
        ww = int(spec["w"])
        if not _fits_empty(grid, bg, top, left, hh, ww):
            return None
        grid[top : top + hh, left : left + ww] = movable_color

    if stopper_spec is not None and stopper_dist is not None:
        top, left = _ray_top_left(midpoint, direction, stopper_dist, stopper_spec)
        hh = int(stopper_spec["h"])
        ww = int(stopper_spec["w"])
        if not _fits_empty(grid, bg, top, left, hh, ww):
            return None
        grid[top : top + hh, left : left + ww] = random.choice(obstacle_palette)

    for _ in range(offray_movable_count):
        if not _place_random_rect(
            grid,
            bg,
            movable_color,
            OFF_RAY_DIMS,
            corridor_axis=corridor_axis,
            corridor_interval=corridor_interval,
        ):
            return None

    for _ in range(extra_obstacle_count):
        if not _place_random_rect(
            grid,
            bg,
            random.choice(obstacle_palette),
            OFF_RAY_DIMS,
            corridor_axis=corridor_axis,
            corridor_interval=corridor_interval,
        ):
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if _count_ray_moves(grid, out) < ray_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
