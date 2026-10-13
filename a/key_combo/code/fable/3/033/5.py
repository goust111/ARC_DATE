from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

OPPOSITE_DIR = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}

BG = 0
PALETTE = list(range(1, 10))
MAX_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return comps


def _infer_direction(marker: tuple[int, int], objects: list[dict]) -> str:
    mr, mc = marker
    all_coords = np.vstack([obj["coords"] for obj in objects])
    r0, c0 = all_coords.min(axis=0)
    r1, c1 = all_coords.max(axis=0)

    candidates = []
    if int(c1) < mc and int(r0) <= mr <= int(r1):
        candidates.append("right")
    if int(c0) > mc and int(r0) <= mr <= int(r1):
        candidates.append("left")
    if int(r1) < mr and int(c0) <= mc <= int(c1):
        candidates.append("down")
    if int(r0) > mr and int(c0) <= mc <= int(c1):
        candidates.append("up")
    if len(candidates) == 1:
        return candidates[0]

    relaxed = []
    if int(c1) < mc:
        relaxed.append("right")
    if int(c0) > mc:
        relaxed.append("left")
    if int(r1) < mr:
        relaxed.append("down")
    if int(r0) > mr:
        relaxed.append("up")
    assert len(relaxed) == 1, "could not infer a unique movement direction"
    return relaxed[0]


def _distance_key(obj: dict, marker: tuple[int, int], direction: str) -> int:
    mr, mc = marker
    r0, c0, r1, c1 = obj["bbox"]
    if direction == "right":
        return mc - c1
    if direction == "left":
        return c0 - mc
    if direction == "down":
        return mr - r1
    return r0 - mr


def _front_coord(coords: np.ndarray, direction: str) -> int:
    if direction == "right":
        return int(coords[:, 1].max())
    if direction == "left":
        return int(coords[:, 1].min())
    if direction == "down":
        return int(coords[:, 0].max())
    return int(coords[:, 0].min())


def _shift_coords(coords: np.ndarray, direction: str, steps: int) -> np.ndarray:
    dr, dc = DIR_VECS[direction]
    return np.asarray(coords, dtype=int) + np.array([dr * steps, dc * steps], dtype=int)


def _profile(coords: np.ndarray, direction: str) -> dict[int, int]:
    prof: dict[int, int] = {}
    if direction == "right":
        for r, c in coords:
            rr = int(r)
            prof[rr] = max(prof.get(rr, -10**9), int(c))
    elif direction == "left":
        for r, c in coords:
            rr = int(r)
            prof[rr] = min(prof.get(rr, 10**9), int(c))
    elif direction == "down":
        for r, c in coords:
            cc = int(c)
            prof[cc] = max(prof.get(cc, -10**9), int(r))
    else:
        for r, c in coords:
            cc = int(c)
            prof[cc] = min(prof.get(cc, 10**9), int(r))
    return prof


def _obstacle_index(obstacles: set[tuple[int, int]], direction: str) -> dict[int, list[int]]:
    idx: dict[int, list[int]] = {}
    if direction in {"right", "left"}:
        for r, c in obstacles:
            idx.setdefault(int(r), []).append(int(c))
    else:
        for r, c in obstacles:
            idx.setdefault(int(c), []).append(int(r))
    for values in idx.values():
        values.sort()
    return idx


def _contact_distance(coords: np.ndarray, obstacles: set[tuple[int, int]], direction: str) -> float:
    prof = _profile(coords, direction)
    idx = _obstacle_index(obstacles, direction)
    best: int | None = None

    for orth, lead in prof.items():
        if orth not in idx:
            continue
        values = idx[orth]
        dist: int | None = None
        if direction == "right":
            for val in values:
                if val > lead:
                    dist = int(val - lead - 1)
                    break
        elif direction == "left":
            for val in reversed(values):
                if val < lead:
                    dist = int(lead - val - 1)
                    break
        elif direction == "down":
            for val in values:
                if val > lead:
                    dist = int(val - lead - 1)
                    break
        else:
            for val in reversed(values):
                if val < lead:
                    dist = int(lead - val - 1)
                    break
        if dist is not None:
            best = dist if best is None else min(best, dist)

    return float("inf") if best is None else float(best)


def _in_bounds(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.all((0 <= coords[:, 0]) & (coords[:, 0] < h))
        and np.all((0 <= coords[:, 1]) & (coords[:, 1] < w))
    )


def _expanded8(coords: np.ndarray, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each object toward the lone marker, capped by the object's own area and stopped early by contact.

    Input structure:
    - A dominant background contains one isolated single-pixel marker and several monochromatic 4-connected objects.
    - Every object lies on the same side of the marker and is aligned along one cardinal movement direction.

    Rule:
    - Identify the unique direction from the objects toward the marker, then process objects from nearest to farthest.
    - For each object, slide it rigidly toward the marker by the smaller of: its area, or the number of steps needed to become edge-adjacent to the marker or to any already-settled object.
    - Keep the marker fixed and redraw all moved objects on a same-size background canvas.

    Key insight:
    - The motion is neither pure area-based translation nor pure stacking: each object is limited by its own area, but earlier stopped objects become new blockers that can shorten later moves.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _find_components(g, bg)
    singletons = [comp for comp in comps if int(comp["area"]) == 1]
    assert len(singletons) == 1, "expected exactly one single-pixel marker"

    marker_comp = singletons[0]
    marker = tuple(map(int, marker_comp["coords"][0]))
    marker_color = int(marker_comp["color"])
    objects = [comp for comp in comps if int(comp["area"]) > 1]
    assert objects, "expected at least one movable object"

    direction = _infer_direction(marker, objects)
    objects.sort(key=lambda obj: _distance_key(obj, marker, direction))

    out = np.full_like(g, bg)
    out[marker] = marker_color
    obstacles: set[tuple[int, int]] = {marker}

    for obj in objects:
        coords = np.asarray(obj["coords"], dtype=int)
        limit = _contact_distance(coords, obstacles, direction)
        steps = int(obj["area"]) if np.isinf(limit) else min(int(obj["area"]), int(limit))
        shifted = _shift_coords(coords, direction, steps)
        assert _in_bounds(shifted, g.shape), "shifted object must stay in bounds"
        for r, c in shifted:
            cell = (int(r), int(c))
            assert cell not in obstacles, "objects may stop adjacent but not overlap obstacles"
            out[cell] = int(obj["color"])
        obstacles.update((int(r), int(c)) for r, c in shifted)

    return out


def _random_shape() -> tuple[np.ndarray, tuple[int, int]]:
    height = random.randint(2, 4)
    width = random.randint(2, 4)
    area = random.randint(2, min(6, height * width))
    anchor = (random.randint(0, height - 1), random.randint(0, width - 1))
    cells = {anchor}
    frontier = set()

    def add_frontier(cell: tuple[int, int]) -> None:
        r, c = cell
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(anchor)
    while len(cells) < area:
        if not frontier:
            return _random_shape()
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        add_frontier(cell)

    arr = np.asarray(sorted(cells), dtype=int)
    mins = arr.min(axis=0)
    arr -= mins
    anchor_r = int(anchor[0] - mins[0])
    anchor_c = int(anchor[1] - mins[1])
    return arr, (anchor_r, anchor_c)


def _place_shape(coords: np.ndarray, fixed_offset: int, axis_offset: int, direction: str) -> np.ndarray:
    if direction in {"left", "right"}:
        return coords + np.array([fixed_offset, axis_offset], dtype=int)
    return coords + np.array([axis_offset, fixed_offset], dtype=int)


def _candidate_axis_values(coords: np.ndarray, marker: tuple[int, int], shape: tuple[int, int], direction: str) -> list[int]:
    h, w = shape
    max_r = int(coords[:, 0].max())
    max_c = int(coords[:, 1].max())
    mr, mc = marker
    if direction == "right":
        vals = list(range(0, mc - max_c))
        vals.sort(reverse=True)
        return vals
    if direction == "left":
        vals = list(range(mc + 1, w - max_c))
        vals.sort()
        return vals
    if direction == "down":
        vals = list(range(0, mr - max_r))
        vals.sort(reverse=True)
        return vals
    vals = list(range(mr + 1, h - max_r))
    vals.sort()
    return vals


def _paint_cells(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _make_modes(count: int) -> list[str]:
    assert count >= 3
    modes = [random.choice(["contact", "area"]), "contact"]
    while len(modes) < count:
        modes.append(random.choice(["contact", "area"]))
    if "area" not in modes:
        modes[-1] = "area"
    return modes


def _ordered_condition(value: int, previous: int | None, direction: str) -> bool:
    if previous is None:
        return True
    if direction in {"right", "down"}:
        return value < previous
    return value > previous


def _build_input() -> tuple[np.ndarray, np.ndarray] | None:
    direction = random.choice(["up", "down", "left", "right"])
    count = random.randint(3, 5)
    modes = _make_modes(count)

    if direction in {"left", "right"}:
        h = random.randint(13, 20)
        w = random.randint(18, 28)
        lane = random.randint(3, h - 4)
        if direction == "right":
            marker = (lane, random.randint(w // 2 + 2, w - 3))
        else:
            marker = (lane, random.randint(2, w // 2 - 2))
    else:
        h = random.randint(18, 28)
        w = random.randint(13, 20)
        lane = random.randint(3, w - 4)
        if direction == "down":
            marker = (random.randint(h // 2 + 2, h - 3), lane)
        else:
            marker = (random.randint(2, h // 2 - 2), lane)

    colors = random.sample(PALETTE, count + 1)
    marker_color = colors[0]
    object_colors = colors[1:]

    final_obstacles: set[tuple[int, int]] = {marker}
    initial_used: set[tuple[int, int]] = set()
    initial_forbidden = _expanded8(np.asarray([marker], dtype=int), (h, w))
    previous_initial_front: int | None = None
    placements: list[dict] = []

    for idx in range(count):
        for _ in range(50):
            local, anchor = _random_shape()
            if direction in {"left", "right"}:
                fixed_offset = lane - anchor[0]
                if fixed_offset < 0 or fixed_offset + int(local[:, 0].max()) >= h:
                    continue
            else:
                fixed_offset = lane - anchor[1]
                if fixed_offset < 0 or fixed_offset + int(local[:, 1].max()) >= w:
                    continue

            area = int(local.shape[0])
            axis_values = _candidate_axis_values(local, marker, (h, w), direction)
            if not axis_values:
                continue

            candidates = []
            for axis_offset in axis_values:
                final_coords = _place_shape(local, fixed_offset, axis_offset, direction)
                final_cells = {(int(r), int(c)) for r, c in final_coords}
                if final_cells & final_obstacles:
                    continue

                gap = _contact_distance(final_coords, final_obstacles, direction)
                if modes[idx] == "contact":
                    if int(gap) != 0:
                        continue
                    move_options = list(range(1, area))
                else:
                    if not np.isfinite(gap) or int(gap) < 1:
                        continue
                    move_options = [area]

                random.shuffle(move_options)
                for move_steps in move_options:
                    initial_coords = _shift_coords(final_coords, OPPOSITE_DIR[direction], move_steps)
                    if not _in_bounds(initial_coords, (h, w)):
                        continue
                    if any((int(r), int(c)) in initial_forbidden for r, c in initial_coords):
                        continue
                    initial_cells = {(int(r), int(c)) for r, c in initial_coords}
                    if initial_cells & initial_used:
                        continue
                    front = _front_coord(initial_coords, direction)
                    if not _ordered_condition(front, previous_initial_front, direction):
                        continue
                    candidates.append((axis_offset, move_steps, gap, final_coords, initial_coords))

            if not candidates:
                continue

            if modes[idx] == "area":
                small_gap = [cand for cand in candidates if 1 <= int(cand[2]) <= 3]
                if small_gap:
                    candidates = small_gap
            choice = random.choice(candidates)
            _axis_offset, move_steps, _gap, final_coords, initial_coords = choice
            placements.append(
                {
                    "color": object_colors[idx],
                    "initial": initial_coords,
                    "final": final_coords,
                    "area": area,
                    "mode": modes[idx],
                    "steps": move_steps,
                }
            )
            final_obstacles.update((int(r), int(c)) for r, c in final_coords)
            initial_used.update((int(r), int(c)) for r, c in initial_coords)
            initial_forbidden.update(_expanded8(initial_coords, (h, w)))
            previous_initial_front = _front_coord(initial_coords, direction)
            break
        else:
            return None

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    inp[marker] = marker_color
    out[marker] = marker_color
    for placement in placements:
        _paint_cells(inp, placement["initial"], placement["color"])
        _paint_cells(out, placement["final"], placement["color"])

    actual = main(inp)
    if not np.array_equal(actual, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        built = _build_input()
        if built is not None:
            return built[0]
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
