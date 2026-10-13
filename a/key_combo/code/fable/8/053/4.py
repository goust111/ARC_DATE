import random
from collections import deque

import numpy as np


BACKGROUND = 0
TERRAIN = 5
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_HEIGHT_RANGE = (16, 24)
GRID_WIDTH_RANGE = (12, 18)
OBJECT_COUNT_CHOICES = (4, 5)
AREA_RANGE = (2, 5)
MAX_POLYOMINO_ATTEMPTS = 200
MAX_SCENE_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _find_objects(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BACKGROUND, TERRAIN) or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            coords = np.array(cells, dtype=int)
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": int(len(cells)),
                    "bottom": int(coords[:, 0].max()),
                    "top": int(coords[:, 0].min()),
                    "left": int(coords[:, 1].min()),
                }
            )
    return objects


def _surface_from_terrain(terrain_mask: np.ndarray) -> np.ndarray:
    h, w = terrain_mask.shape
    surface = np.full(w, h, dtype=int)
    for c in range(w):
        rows = np.flatnonzero(terrain_mask[:, c])
        if rows.size:
            surface[c] = int(rows[0])
    return surface


def _bottoms_by_col(coords: np.ndarray) -> dict[int, int]:
    bottoms: dict[int, int] = {}
    for r, c in coords:
        c = int(c)
        bottoms[c] = max(bottoms.get(c, -10**9), int(r))
    return bottoms


def _tops_by_col(coords: np.ndarray) -> dict[int, int]:
    tops: dict[int, int] = {}
    for r, c in coords:
        c = int(c)
        tops[c] = min(tops.get(c, 10**9), int(r))
    return tops


def _clear_rows(coords: np.ndarray, surface: np.ndarray) -> int:
    bottoms = _bottoms_by_col(coords)
    return min(int(surface[c]) - int(r) - 1 for c, r in bottoms.items())


def _sort_objects(objects: list[dict], order: str) -> list[dict]:
    if order == "low_first":
        key = lambda obj: (-obj["bottom"], obj["left"], obj["top"])
    elif order == "high_first":
        key = lambda obj: (obj["bottom"], obj["left"], obj["top"])
    else:
        raise ValueError(f"unknown order: {order}")
    return sorted(objects, key=key)


def _simulate(
    grid: np.ndarray,
    *,
    order: str = "low_first",
    dynamic_terrain: bool = True,
    return_trace: bool = False,
):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    terrain_mask = g == TERRAIN
    objects = _find_objects(g)
    assert objects, "expected at least one colored object"

    sorted_objects = _sort_objects(objects, order)
    bottoms = [obj["bottom"] for obj in sorted_objects]
    assert len(set(bottoms)) == len(bottoms), "object bottom rows must be unique"

    initial_surface = _surface_from_terrain(terrain_mask)
    out = np.full_like(g, BACKGROUND)
    out[terrain_mask] = TERRAIN
    output_occ = terrain_mask.copy()
    current_terrain = terrain_mask.copy()
    trace = []

    for obj in sorted_objects:
        coords = obj["coords"]
        area = int(obj["area"])
        color = int(obj["color"])
        surface = _surface_from_terrain(current_terrain) if dynamic_terrain else initial_surface
        clear = _clear_rows(coords, surface)
        assert clear >= 0, "colored objects must start above terrain"
        shift = min(area, clear)
        moved = coords.copy()
        moved[:, 0] += shift
        rr = moved[:, 0]
        cc = moved[:, 1]
        assert not output_occ[rr, cc].any(), "final objects must not overlap terrain or each other"

        merged = clear <= area
        final_color = TERRAIN if merged else color
        out[rr, cc] = final_color
        output_occ[rr, cc] = True
        if merged:
            current_terrain[rr, cc] = True

        trace.append(
            {
                "color": color,
                "area": area,
                "bottom": int(obj["bottom"]),
                "clear": int(clear),
                "shift": int(shift),
                "merged": bool(merged),
                "final_color": int(final_color),
                "initial_coords": coords.copy(),
                "final_coords": moved.copy(),
            }
        )

    if return_trace:
        return out, trace
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop colored objects in low-first order, turning every touching object into new grey terrain.

    Input structure:
    - The background is black, a connected grey terrain occupies the lower part of the grid,
      and several non-grey monochrome 4-connected objects float above it.
    - Different objects may overlap in column span, and lower objects can become supports for
      higher ones after they land.
    - Each object's pixel count is its personal falling budget.

    Rule:
    - Process the colored objects from lower bottom row to higher bottom row.
    - For the current object, compute the empty rows to the current grey terrain directly below
      its bottom profile, then move it down by min(its area, that clearance) rows.
    - If that move brings the object into contact with terrain, recolor the whole object grey and
      merge it into the terrain immediately; otherwise keep its original color and leave it suspended.

    Key insight:
    - The terrain is dynamic: only landed grey objects become new support, so both the low-to-high
      processing order and each object's own area are necessary to determine later landings.
    """
    return _simulate(grid, order="low_first", dynamic_terrain=True)


def _normalize(coords: set[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    arr = np.array([(r - min_r, c - min_c) for r, c in coords], dtype=int)
    if random.random() < 0.5:
        arr[:, 0] *= -1
    if random.random() < 0.5:
        arr[:, 1] *= -1
    if random.random() < 0.5:
        arr = arr[:, ::-1]
    min_r = int(arr[:, 0].min())
    min_c = int(arr[:, 1].min())
    arr[:, 0] -= min_r
    arr[:, 1] -= min_c
    return arr


def _random_polyomino(area: int, *, min_height: int = 1) -> np.ndarray:
    for _ in range(MAX_POLYOMINO_ATTEMPTS):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            cells.add(random.choice(tuple(frontier)))
        arr = _normalize(cells)
        height = int(arr[:, 0].max()) + 1
        width = int(arr[:, 1].max()) + 1
        if height <= 4 and width <= 4 and height >= min_height:
            return arr
    raise RuntimeError("failed to generate polyomino")


def _terrain_surface(height: int, width: int) -> np.ndarray:
    base = height - random.randint(4, 6)
    surface = np.full(width, base, dtype=int)
    for _ in range(random.randint(2, 4)):
        seg_len = random.randint(2, min(5, width))
        c0 = random.randint(0, width - seg_len)
        raise_by = random.randint(1, min(5, base - 3))
        surface[c0 : c0 + seg_len] = np.minimum(surface[c0 : c0 + seg_len], base - raise_by)
    if len(set(map(int, surface))) == 1:
        c0 = random.randint(0, width - 2)
        surface[c0 : c0 + 2] -= 2
    return surface


def _terrain_grid(surface: np.ndarray) -> np.ndarray:
    height = int(surface.max()) + 6
    grid = np.full((height, len(surface)), BACKGROUND, dtype=int)
    for c, top in enumerate(surface):
        grid[int(top) :, c] = TERRAIN
    return grid


def _mark_blocked(blocked: np.ndarray, coords: np.ndarray) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = int(r) + dr
                nc = int(c) + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _candidate_positions(
    shape: np.ndarray,
    *,
    surface: np.ndarray,
    initial_surface: np.ndarray,
    input_blocked: np.ndarray,
    output_occ: np.ndarray,
    bottom_cap: int,
    branch: str,
    require_chain: bool,
) -> list[dict]:
    h, w = input_blocked.shape
    area = int(shape.shape[0])
    max_r = int(shape[:, 0].max())
    max_c = int(shape[:, 1].max())
    local_bottoms = _bottoms_by_col(shape)

    starts = list(range(w - max_c))
    random.shuffle(starts)
    candidates = []
    for c0 in starts:
        limit = min(int(surface[c0 + cc]) - int(rr) - 1 for cc, rr in local_bottoms.items())
        if limit < 1:
            continue

        if branch == "merge":
            gap_values = list(range(1, min(area, limit) + 1))
        elif branch == "suspend":
            gap_values = list(range(area + 1, min(limit, area + 4) + 1))
        else:
            raise ValueError(f"unknown branch: {branch}")
        random.shuffle(gap_values)

        for gap in gap_values:
            r0 = limit - gap
            if r0 < 0:
                continue
            bottom = r0 + max_r
            if bottom >= bottom_cap:
                continue

            coords = shape.copy()
            coords[:, 0] += r0
            coords[:, 1] += c0
            if input_blocked[coords[:, 0], coords[:, 1]].any():
                continue

            shift = min(area, gap)
            final_coords = coords.copy()
            final_coords[:, 0] += shift
            if final_coords[:, 0].max() >= h or output_occ[final_coords[:, 0], final_coords[:, 1]].any():
                continue

            if require_chain:
                clear_initial = _clear_rows(coords, initial_surface)
                shift_initial = min(area, clear_initial)
                merged_initial = clear_initial <= area
                if shift_initial == shift and merged_initial == (gap <= area):
                    continue
                delta = shift_initial - shift
            else:
                delta = 0

            candidates.append(
                {
                    "coords": coords,
                    "final_coords": final_coords,
                    "bottom": int(bottom),
                    "gap": int(gap),
                    "merged": bool(gap <= area),
                    "shift_delta": int(delta),
                }
            )
    if require_chain:
        candidates.sort(key=lambda cand: cand["shift_delta"], reverse=True)
    return candidates


def _update_surface(surface: np.ndarray, final_coords: np.ndarray) -> None:
    tops = _tops_by_col(final_coords)
    for c, top in tops.items():
        surface[c] = min(int(surface[c]), int(top))


def _validate_scene(grid: np.ndarray) -> bool:
    try:
        dynamic_out, dynamic_trace = _simulate(grid, order="low_first", dynamic_terrain=True, return_trace=True)
    except Exception:
        return False

    if len(dynamic_trace) < 4:
        return False
    if len({item["bottom"] for item in dynamic_trace}) != len(dynamic_trace):
        return False

    merged = sum(int(item["merged"]) for item in dynamic_trace)
    suspended = len(dynamic_trace) - merged
    if merged == 0 or suspended == 0:
        return False

    try:
        static_out, static_trace = _simulate(grid, order="low_first", dynamic_terrain=False, return_trace=True)
        static_same = np.array_equal(dynamic_out, static_out)
    except Exception:
        static_trace = None
        static_same = False
    if static_same:
        return False

    if static_trace is not None:
        changed = any(
            dyn["merged"] != sta["merged"]
            or dyn["shift"] != sta["shift"]
            or not np.array_equal(dyn["final_coords"], sta["final_coords"])
            for dyn, sta in zip(dynamic_trace, static_trace)
        )
        if not changed:
            return False

    try:
        reverse_out = _simulate(grid, order="high_first", dynamic_terrain=True)
        reverse_same = np.array_equal(dynamic_out, reverse_out)
    except Exception:
        reverse_same = False
    if reverse_same:
        return False

    if len(set(map(int, np.unique(dynamic_out)))) < 3:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    surface_initial = _terrain_surface(height, width)
    grid = np.full((height, width), BACKGROUND, dtype=int)
    for c, top in enumerate(surface_initial):
        grid[int(top) :, c] = TERRAIN

    surface = surface_initial.copy()
    output_occ = grid == TERRAIN
    input_blocked = np.zeros((height, width), dtype=bool)
    colors = random.sample(OBJECT_COLORS, random.choice(OBJECT_COUNT_CHOICES))
    roles = [("merge", False, 2), ("suspend", False, 1), ("merge", True, 1)]
    for _ in range(len(colors) - len(roles)):
        roles.append((random.choice(("merge", "suspend")), False, 1))

    bottom_cap = height
    for color, (branch, require_chain, min_height) in zip(colors, roles):
        placed = False
        for _ in range(80):
            area = random.randint(max(3, AREA_RANGE[0]), AREA_RANGE[1]) if min_height >= 2 else random.randint(*AREA_RANGE)
            shape = _random_polyomino(area, min_height=min_height)
            candidates = _candidate_positions(
                shape,
                surface=surface,
                initial_surface=surface_initial,
                input_blocked=input_blocked,
                output_occ=output_occ,
                bottom_cap=bottom_cap,
                branch=branch,
                require_chain=require_chain,
            )
            if not candidates:
                continue
            keep = candidates[: min(len(candidates), 12)]
            cand = random.choice(keep)
            coords = cand["coords"]
            grid[coords[:, 0], coords[:, 1]] = color
            _mark_blocked(input_blocked, coords)
            final_coords = cand["final_coords"]
            output_occ[final_coords[:, 0], final_coords[:, 1]] = True
            if cand["merged"]:
                _update_surface(surface, final_coords)
            bottom_cap = cand["bottom"]
            placed = True
            break
        if not placed:
            return None

    if not _validate_scene(grid):
        return None
    if not np.array_equal(main(grid), _simulate(grid, order="low_first", dynamic_terrain=True)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_SCENE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chain-settling input")
