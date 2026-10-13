from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


def _bg(grid: np.ndarray) -> int:
    return Counter(map(int, grid.ravel())).most_common(1)[0][0]


def _components(grid: np.ndarray, bg: int) -> list[dict]:
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
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "color": color,
                    "coords": set(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _arrow_cells(tip: tuple[int, int], direction: str, side: int) -> set[tuple[int, int]]:
    r, c = tip
    if direction == "R":
        return {(r, c), (r, c - 1), (r + side, c)}
    if direction == "L":
        return {(r, c), (r, c + 1), (r + side, c)}
    if direction == "D":
        return {(r, c), (r - 1, c), (r, c + side)}
    if direction == "U":
        return {(r, c), (r + 1, c), (r, c + side)}
    raise ValueError(direction)


def _match_arrow(coords: set[tuple[int, int]]) -> tuple[str, tuple[int, int]] | None:
    if len(coords) != 3:
        return None
    for tip in coords:
        for direction in DIRS:
            for side in (-1, 1):
                if coords == _arrow_cells(tip, direction, side):
                    return direction, tip
    return None


def _find_arrows_and_objects(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    arrows: list[dict] = []
    objects: list[dict] = []
    for comp in _components(grid, bg):
        matched = _match_arrow(comp["coords"])
        if matched is None:
            objects.append(comp)
        else:
            direction, tip = matched
            arrows.append(
                {
                    "color": int(comp["color"]),
                    "coords": set(comp["coords"]),
                    "direction": direction,
                    "tip": tip,
                }
            )
    arrows.sort(key=lambda a: a["tip"])
    return arrows, objects


def _ray_target(arrow: dict, object_at: dict[tuple[int, int], int], shape: tuple[int, int]) -> int | None:
    h, w = shape
    dr, dc = DIRS[arrow["direction"]]
    r, c = arrow["tip"]
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        idx = object_at.get((r, c))
        if idx is not None:
            return idx
        r += dr
        c += dc
    return None


def _translated_coords(obj: dict, direction: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = obj["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if direction == "R":
        nr0, nc0 = r0, w - width
    elif direction == "L":
        nr0, nc0 = r0, 0
    elif direction == "D":
        nr0, nc0 = h - height, c0
    else:
        nr0, nc0 = 0, c0
    return [(nr0 + r - r0, nc0 + c - c0) for r, c in obj["coords"]]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg(g)
    out = g.copy()
    arrows, objects = _find_arrows_and_objects(g, bg)

    object_at: dict[tuple[int, int], int] = {}
    for idx, obj in enumerate(objects):
        for coord in obj["coords"]:
            object_at[coord] = idx

    assignments: list[tuple[int, dict]] = []
    used: set[int] = set()
    for arrow in arrows:
        for coord in arrow["coords"]:
            out[coord] = bg
        target = _ray_target(arrow, object_at, g.shape)
        if target is not None and target not in used:
            assignments.append((target, arrow))
            used.add(target)

    for idx, _ in assignments:
        for r, c in objects[idx]["coords"]:
            out[r, c] = bg

    for idx, arrow in assignments:
        for r, c in _translated_coords(objects[idx], arrow["direction"], g.shape):
            out[r, c] = int(arrow["color"])

    return out


SHAPES = [
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
]


def _shape_cells(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c in np.argwhere(mask)}


def _in_bounds(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _padded_clear(cells: set[tuple[int, int]], blocked: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> bool:
    if not _in_bounds(cells, h, w):
        return False
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) in blocked:
                    return False
    return True


def _ray_cells(tip: tuple[int, int], direction: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = DIRS[direction]
    r, c = tip[0] + dr, tip[1] + dc
    cells = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _candidate_pair(mask: np.ndarray, direction: str, h: int, w: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]], list[tuple[int, int]]] | None:
    mh, mw = mask.shape
    for _ in range(80):
        top = random.randint(2, h - mh - 3)
        left = random.randint(2, w - mw - 3)
        obj = _shape_cells(mask, top, left)
        occ = list(zip(*np.where(mask)))
        gap = random.randint(2, 5)
        side = random.choice([-1, 1])
        if direction == "R":
            local_r, local_c = random.choice(occ)
            hit = (top + int(local_r), left + min(c for r, c in occ if r == local_r))
            tip = (hit[0], hit[1] - gap - 1)
        elif direction == "L":
            local_r, local_c = random.choice(occ)
            hit = (top + int(local_r), left + max(c for r, c in occ if r == local_r))
            tip = (hit[0], hit[1] + gap + 1)
        elif direction == "D":
            local_r, local_c = random.choice(occ)
            hit = (top + min(r for r, c in occ if c == local_c), left + int(local_c))
            tip = (hit[0] - gap - 1, hit[1])
        else:
            local_r, local_c = random.choice(occ)
            hit = (top + max(r for r, c in occ if c == local_c), left + int(local_c))
            tip = (hit[0] + gap + 1, hit[1])
        arr = _arrow_cells(tip, direction, side)
        if not _in_bounds(arr, h, w):
            continue
        ray = _ray_cells(tip, direction, (h, w))
        if hit not in ray:
            continue
        return obj, arr, ray
    return None


def _landing_cells(obj: set[tuple[int, int]], direction: str, shape: tuple[int, int]) -> set[tuple[int, int]]:
    rows = [r for r, _ in obj]
    cols = [c for _, c in obj]
    r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
    fake = {"coords": obj, "bbox": (r0, c0, r1, c1)}
    return set(_translated_coords(fake, direction, shape))


def _place_random_object(grid: np.ndarray, color: int, blocked: set[tuple[int, int]]) -> bool:
    h, w = grid.shape
    mask = random.choice(SHAPES)
    mh, mw = mask.shape
    for _ in range(200):
        top = random.randint(2, h - mh - 3)
        left = random.randint(2, w - mw - 3)
        cells = _shape_cells(mask, top, left)
        if _padded_clear(cells, blocked, h, w):
            for r, c in cells:
                grid[r, c] = color
            blocked.update(cells)
            return True
    return False


def _attempt() -> np.ndarray | None:
    h = random.randint(18, 29)
    w = random.randint(18, 29)
    grid = np.zeros((h, w), dtype=int)
    pair_count = random.randint(1, 3)
    extra_count = random.randint(1, 3)
    colors = random.sample(list(range(1, 10)), pair_count * 2 + extra_count)
    arrow_colors = colors[:pair_count]
    object_colors = colors[pair_count:]
    blocked: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    landings: set[tuple[int, int]] = set()

    for idx in range(pair_count):
        placed = False
        for _ in range(300):
            direction = random.choice(list(DIRS))
            mask = random.choice(SHAPES)
            cand = _candidate_pair(mask, direction, h, w)
            if cand is None:
                continue
            obj, arr, ray = cand
            hit_count = len(obj & set(ray))
            if hit_count == 0:
                continue
            landing = _landing_cells(obj, direction, (h, w))
            if not _padded_clear(obj, blocked | reserved, h, w):
                continue
            if not _padded_clear(arr, blocked | reserved | obj, h, w):
                continue
            if (set(ray) - obj) & (blocked | reserved):
                continue
            if landing & (blocked - obj) or landing & landings:
                continue
            for r, c in obj:
                grid[r, c] = object_colors[idx]
            for r, c in arr:
                grid[r, c] = arrow_colors[idx]
            blocked.update(obj)
            blocked.update(arr)
            reserved.update(set(ray) - obj)
            landings.update(landing)
            placed = True
            break
        if not placed:
            return None

    for color in object_colors[pair_count : pair_count + extra_count]:
        if not _place_random_object(grid, color, blocked | reserved | landings):
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate arrow translation puzzle")


if __name__ == "__main__":
    for _ in range(20):
        sample = generate_input()
        solved = main(sample)
        assert isinstance(sample, np.ndarray)
        assert isinstance(solved, np.ndarray)
        assert sample.shape == solved.shape
