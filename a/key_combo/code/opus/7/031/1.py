from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}

SHAPES = (
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2)),
)


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=int)
    mins = coords.min(axis=0)
    return coords - mins


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, offset: tuple[int, int] = (0, 0)) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    ro, co = offset
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr + ro, cc + co))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _projection_interval(coords: np.ndarray, direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    if direction in {"up", "down"}:
        return c0, c1
    return r0, r1


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _distance_to_target(coords: np.ndarray, direction: str, shape: tuple[int, int]) -> int:
    h, w = shape
    r0, c0, r1, c1 = _bbox(coords)
    if direction == "up":
        return r0 - 1
    if direction == "down":
        return (h - 2) - r1
    if direction == "left":
        return c0 - 1
    return (w - 2) - c1


def _build_frame(h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[0, 1 : w - 1] = RED
    grid[h - 1, 1 : w - 1] = BLUE
    grid[1 : h - 1, 0] = YELLOW
    grid[1 : h - 1, w - 1] = GREEN
    return grid


def _frame_direction_map(grid: np.ndarray) -> dict[int, str]:
    h, w = grid.shape
    top = int(grid[0, 1])
    bottom = int(grid[h - 1, 1])
    left = int(grid[1, 0])
    right = int(grid[1, w - 1])
    assert grid[0, 0] == grid[0, w - 1] == grid[h - 1, 0] == grid[h - 1, w - 1] == BG
    assert np.all(grid[0, 1 : w - 1] == top)
    assert np.all(grid[h - 1, 1 : w - 1] == bottom)
    assert np.all(grid[1 : h - 1, 0] == left)
    assert np.all(grid[1 : h - 1, w - 1] == right)
    mapping = {top: "up", bottom: "down", left: "left", right: "right"}
    assert len(mapping) == 4 and BG not in mapping
    return mapping


def _extract_sprites(grid: np.ndarray) -> list[dict]:
    direction_by_color = _frame_direction_map(grid)
    components = _components(grid[1:-1, 1:-1] != BG, offset=(1, 1))
    sprites: list[dict] = []
    for coords in components:
        values = grid[coords[:, 0], coords[:, 1]].astype(int)
        marker_colors = [int(v) for v in values if int(v) in direction_by_color]
        assert len(marker_colors) == 1
        marker_color = marker_colors[0]
        sprites.append(
            {
                "coords": coords.copy(),
                "values": values.copy(),
                "direction": direction_by_color[marker_color],
            }
        )
    assert 3 <= len(sprites) <= 5
    return sprites


def _slide_sprite(
    coords: np.ndarray,
    direction: str,
    settled: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, int, str]:
    h, w = shape
    dr, dc = DIR_VECS[direction]
    cur = coords.copy()
    steps = 0
    while True:
        nxt = cur + np.array([dr, dc], dtype=int)
        if np.any(nxt[:, 0] < 1) or np.any(nxt[:, 0] > h - 2) or np.any(nxt[:, 1] < 1) or np.any(nxt[:, 1] > w - 2):
            return cur, steps, "border"
        if np.any(settled[nxt[:, 0], nxt[:, 1]]):
            return cur, steps, "obstacle"
        cur = nxt
        steps += 1


def _simulate(grid_shape: tuple[int, int], sprites: list[dict]) -> list[dict]:
    ordered = sorted(
        sprites,
        key=lambda s: (_distance_to_target(s["coords"], s["direction"], grid_shape), *_bbox(s["coords"])[:2]),
    )
    settled = np.zeros(grid_shape, dtype=bool)
    result: list[dict] = []
    for sprite in ordered:
        final_coords, steps, stop_reason = _slide_sprite(sprite["coords"], sprite["direction"], settled, grid_shape)
        assert not np.any(settled[final_coords[:, 0], final_coords[:, 1]])
        settled[final_coords[:, 0], final_coords[:, 1]] = True
        item = dict(sprite)
        item["final_coords"] = final_coords
        item["steps"] = steps
        item["stop_reason"] = stop_reason
        result.append(item)
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each marked gray sprite toward the matching frame side until it settles against the frame or an earlier-settled sprite.

    Input structure:
    - A black background surrounds a one-cell rectangular frame whose four sides have distinct colors and whose corners stay black.
    - Inside the frame are 3 to 5 separate 4-connected sprites made of gray cells, with exactly one side-colored marker embedded in each sprite.
    - The marker color matches one frame side and serves as that sprite's direction label.

    Rule:
    - Read the four frame-side colors as the four directions: top, bottom, left, and right.
    - Treat each interior component as a rigid sprite and slide it one cell at a time toward the side whose color appears as its unique marker.
    - Process sprites from smaller distance-to-target-side to larger distance, breaking ties by row-major scan order; once a sprite settles, it blocks later sprites.

    Key insight:
    - The colored pixel is not an independent moving marker: it is an instruction embedded inside a whole rigid sprite, so the task is ordered multi-direction sliding with collision blocking.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 5 and g.shape[1] >= 5
    sprites = _extract_sprites(g)
    settled = _simulate(g.shape, sprites)

    out = g.copy()
    out[1:-1, 1:-1] = BG
    for sprite in settled:
        coords = sprite["final_coords"]
        values = sprite["values"]
        out[coords[:, 0], coords[:, 1]] = values
    return out


def _random_shape() -> np.ndarray:
    coords = np.array(random.choice(SHAPES), dtype=int)
    if random.random() < 0.5:
        coords[:, 1] *= -1
    rot = random.randrange(4)
    for _ in range(rot):
        coords = np.stack([-coords[:, 1], coords[:, 0]], axis=1)
    return _normalize_coords(coords)


def _render_values(shape_coords: np.ndarray, marker_color: int) -> np.ndarray:
    values = np.full(len(shape_coords), GREY, dtype=int)
    values[random.randrange(len(shape_coords))] = marker_color
    return values


def _coords_for_anchor(shape_coords: np.ndarray, anchor: tuple[int, int]) -> np.ndarray:
    return shape_coords + np.array(anchor, dtype=int)


def _path_cells(start: np.ndarray, direction: str, steps: int) -> np.ndarray:
    dr, dc = DIR_VECS[direction]
    traces = []
    cur = start.copy()
    for _ in range(steps):
        cur = cur + np.array([dr, dc], dtype=int)
        traces.append(cur.copy())
    if not traces:
        return np.zeros((0, 2), dtype=int)
    return np.concatenate(traces, axis=0)


def _touches_existing(coords: np.ndarray, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        r = int(r)
        c = int(c)
        if occupied[r, c]:
            return True
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                return True
    return False


def _candidate_sprites(
    grid_shape: tuple[int, int],
    shape_coords: np.ndarray,
    direction: str,
    min_distance: int,
    settled: np.ndarray,
    input_occ: np.ndarray,
    reserved_paths: np.ndarray,
    same_dir_intervals: list[tuple[int, int]],
    require_obstacle: bool,
) -> list[dict]:
    h, w = grid_shape
    r0, c0, r1, c1 = _bbox(shape_coords)
    shp_h = r1 - r0 + 1
    shp_w = c1 - c0 + 1
    candidates: list[dict] = []
    for ar in range(1, h - 1 - shp_h + 1):
        for ac in range(1, w - 1 - shp_w + 1):
            coords = _coords_for_anchor(shape_coords, (ar, ac))
            if _touches_existing(coords, input_occ):
                continue
            if np.any(settled[coords[:, 0], coords[:, 1]]) or np.any(reserved_paths[coords[:, 0], coords[:, 1]]):
                continue
            interval = _projection_interval(coords, direction)
            if any(_intervals_overlap(interval, old) for old in same_dir_intervals):
                continue
            dist = _distance_to_target(coords, direction, grid_shape)
            if dist < min_distance:
                continue
            final_coords, steps, stop_reason = _slide_sprite(coords, direction, settled, grid_shape)
            if steps == 0:
                continue
            if require_obstacle and stop_reason != "obstacle":
                continue
            path = _path_cells(coords, direction, steps)
            if path.size:
                if np.any(input_occ[path[:, 0], path[:, 1]]) or np.any(reserved_paths[path[:, 0], path[:, 1]]):
                    continue
            candidates.append(
                {
                    "coords": coords,
                    "direction": direction,
                    "distance": dist,
                    "final_coords": final_coords,
                    "steps": steps,
                    "stop_reason": stop_reason,
                    "path": path,
                    "interval": interval,
                }
            )
    return candidates


def _direction_sequence(count: int) -> list[str]:
    dirs = ["up", "down", "left", "right"]
    if count == 3:
        seq = random.sample(dirs, 3)
    else:
        seq = dirs[:]
        while len(seq) < count:
            seq.append(random.choice(dirs))
    random.shuffle(seq)
    return seq


def _build_input() -> np.ndarray | None:
    h = random.randint(13, 17)
    w = random.randint(13, 17)
    grid = _build_frame(h, w)
    count = random.randint(3, 5)
    directions = _direction_sequence(count)
    marker_by_direction = {"up": RED, "down": BLUE, "left": YELLOW, "right": GREEN}

    obstacle_index = random.randrange(1, count)
    settled = np.zeros((h, w), dtype=bool)
    input_occ = np.zeros((h, w), dtype=bool)
    reserved_paths = np.zeros((h, w), dtype=bool)
    same_dir_intervals: dict[str, list[tuple[int, int]]] = {d: [] for d in DIR_VECS}
    sprites: list[dict] = []
    min_distance = 1

    for idx, direction in enumerate(directions):
        require_obstacle = idx == obstacle_index
        shape_coords = _random_shape()
        candidates = _candidate_sprites(
            grid_shape=(h, w),
            shape_coords=shape_coords,
            direction=direction,
            min_distance=min_distance,
            settled=settled,
            input_occ=input_occ,
            reserved_paths=reserved_paths,
            same_dir_intervals=same_dir_intervals[direction],
            require_obstacle=require_obstacle,
        )
        if require_obstacle and not candidates:
            return None
        if not candidates:
            return None

        if require_obstacle:
            choices = candidates
        else:
            choices = [c for c in candidates if c["stop_reason"] == "border"] or candidates
        chosen = random.choice(choices)
        chosen["values"] = _render_values(chosen["coords"], marker_by_direction[direction])
        sprites.append(chosen)

        input_occ[chosen["coords"][:, 0], chosen["coords"][:, 1]] = True
        if chosen["path"].size:
            reserved_paths[chosen["path"][:, 0], chosen["path"][:, 1]] = True
        settled[chosen["final_coords"][:, 0], chosen["final_coords"][:, 1]] = True
        same_dir_intervals[direction].append(chosen["interval"])
        min_distance = chosen["distance"] + 1

    moved = sum(sprite["steps"] > 0 for sprite in sprites)
    collisions = sum(sprite["stop_reason"] == "obstacle" for sprite in sprites)
    if moved < 3 or collisions < 1:
        return None

    for sprite in sprites:
        grid[sprite["coords"][:, 0], sprite["coords"][:, 1]] = sprite["values"]

    components = _extract_sprites(grid)
    if len(components) != count:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
