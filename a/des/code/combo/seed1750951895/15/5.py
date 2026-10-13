from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_objects(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []

    for r0 in range(h):
        for c0 in range(w):
            if visited[r0, c0] or int(grid[r0, c0]) == bg:
                continue
            color = int(grid[r0, c0])
            stack = [(r0, c0)]
            visited[r0, c0] = True
            cells: list[tuple[int, int]] = []

            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            r1, r2 = min(rows), max(rows)
            c1, c2 = min(cols), max(cols)
            if r2 - r1 != 3 or c2 - c1 != 3:
                continue

            sub = grid[r1 : r2 + 1, c1 : c2 + 1]
            mask = sub != bg
            if not np.all((sub == bg) | (sub == color)):
                continue

            solid = np.all(sub == color)
            frame = np.all(sub[[0, 0, 0, 0, 3, 3, 3, 3], [0, 1, 2, 3, 0, 1, 2, 3]] == color)
            frame = frame and np.all(sub[1:3, 0] == color) and np.all(sub[1:3, 3] == color) and np.all(sub[1:3, 1:3] == bg)
            if solid:
                kind = "solid"
            elif frame and int(mask.sum()) == 12:
                kind = "frame"
            else:
                continue

            objects.append(
                {
                    "top": int(r1),
                    "left": int(c1),
                    "color": color,
                    "kind": kind,
                    "sub": sub.copy(),
                }
            )

    return objects


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    objects = _find_objects(g, bg)

    frames = sorted((obj for obj in objects if obj["kind"] == "frame"), key=lambda obj: (int(obj["top"]), int(obj["left"])))
    solids = sorted((obj for obj in objects if obj["kind"] == "solid"), key=lambda obj: (int(obj["top"]), int(obj["left"])))

    out_h = 4 * max(len(frames), len(solids))
    if out_h == 0:
        return np.zeros((0, 8), dtype=int)

    out = np.full((out_h, 8), bg, dtype=int)
    for i, obj in enumerate(frames):
        out[i * 4 : (i + 1) * 4, 0:4] = np.asarray(obj["sub"], dtype=int)
    for i, obj in enumerate(solids):
        out[i * 4 : (i + 1) * 4, 4:8] = np.asarray(obj["sub"], dtype=int)
    return out


def _make_frame(color: int, bg: int) -> np.ndarray:
    tile = np.full((4, 4), color, dtype=int)
    tile[1:3, 1:3] = bg
    return tile


def _make_solid(color: int) -> np.ndarray:
    return np.full((4, 4), color, dtype=int)


def _can_place(occupied: np.ndarray, r0: int, c0: int) -> bool:
    h, w = occupied.shape
    r1 = max(0, r0 - 1)
    c1 = max(0, c0 - 1)
    r2 = min(h, r0 + 5)
    c2 = min(w, c0 + 5)
    return not bool(occupied[r1:r2, c1:c2].any())


def _place_object(grid: np.ndarray, occupied: np.ndarray, obj: np.ndarray, rng: random.Random) -> bool:
    h, w = grid.shape
    spots = [(r, c) for r in range(h - 3) for c in range(w - 3) if _can_place(occupied, r, c)]
    if not spots:
        return False
    r0, c0 = rng.choice(spots)
    grid[r0 : r0 + 4, c0 : c0 + 4] = obj
    occupied[r0 : r0 + 4, c0 : c0 + 4] = True
    return True


def _expected_output(specs: list[tuple[str, int, np.ndarray]], bg: int) -> np.ndarray:
    frames = sorted((item for item in specs if item[0] == "frame"), key=lambda item: (item[1],))
    solids = sorted((item for item in specs if item[0] == "solid"), key=lambda item: (item[1],))
    out = np.full((4 * max(len(frames), len(solids)), 8), bg, dtype=int)
    for i, (_kind, _top, tile) in enumerate(frames):
        out[i * 4 : (i + 1) * 4, 0:4] = tile
    for i, (_kind, _top, tile) in enumerate(solids):
        out[i * 4 : (i + 1) * 4, 4:8] = tile
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = 0
    h = rng.randint(16, 30)
    w = rng.randint(16, 30)
    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    n_frames = rng.randint(1, 5)
    n_solids = rng.randint(1, 5)
    total = n_frames + n_solids

    if (h - 3) * (w - 3) < total:
        return None

    colors = list(range(1, 10))
    rng.shuffle(colors)
    if total > len(colors):
        return None

    kinds = ["frame"] * n_frames + ["solid"] * n_solids
    rng.shuffle(kinds)

    specs: list[tuple[str, int, np.ndarray]] = []
    for index, kind in enumerate(kinds):
        color = colors[index]
        tile = _make_frame(color, bg) if kind == "frame" else _make_solid(color)
        if not _place_object(grid, occupied, tile, rng):
            return None
        top = int(np.argwhere(grid == color)[:, 0].min())
        specs.append((kind, top, tile.copy()))

    expected = _expected_output(specs, bg)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    objects = _find_objects(grid, bg)
    counts = Counter(obj["kind"] for obj in objects)
    if counts.get("frame", 0) != n_frames or counts.get("solid", 0) != n_solids:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 500 attempts")
