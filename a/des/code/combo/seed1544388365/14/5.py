from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

GEN_COLORS = (1, 2, 3, 4, 6, 7, 8)
INTERIOR_SHAPES = ((2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (3, 3))
DIRECTION_COUNTS = (1, 2, 2, 3, 3, 4, 4, 5)
GEN_RETRIES = 400
PLACE_RETRIES = 250
LEGEND_RESERVED_ROWS = 5


def _components(mask: np.ndarray, conn8: bool = False) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neigh = DIRS8 if conn8 else DIRS4
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in neigh:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _decode_legend(grid: np.ndarray) -> dict[int, tuple[tuple[int, int], ...]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 3 and w >= 3, "legend needs at least three rows and columns"

    mapping: dict[int, tuple[tuple[int, int], ...]] = {}
    for c in range(1, w - 1):
        color = int(g[1, c])
        if color in (BLACK, GREY):
            continue
        dirs = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if int(g[1 + dr, c + dc]) == GREY:
                    dirs.append((dr, dc))
        assert dirs, "each legend color must expose at least one grey growth direction"
        assert color not in mapping, "legend center colors must be unique"
        mapping[color] = tuple(dirs)

    assert mapping, "legend must contain at least one color template"
    return mapping


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict[str, object]] = []

    for pts in _components(g == MAROON, conn8=False):
        r0, c0, r1, c1 = _bbox(pts)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        if not (
            np.all(g[r0, c0 : c1 + 1] == MAROON)
            and np.all(g[r1, c0 : c1 + 1] == MAROON)
            and np.all(g[r0 : r1 + 1, c0] == MAROON)
            and np.all(g[r0 : r1 + 1, c1] == MAROON)
        ):
            continue

        perimeter = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
        if len(pts) != perimeter:
            continue

        seed_pts = [
            (r, c)
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
            if int(g[r, c]) not in (BLACK, MAROON)
        ]
        assert len(seed_pts) == 1, "each frame must contain exactly one seed pixel"
        sr, sc = seed_pts[0]
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "seed": (sr, sc),
                "color": int(g[sr, sc]),
                "ih": r1 - r0 - 1,
                "iw": c1 - c0 - 1,
            }
        )

    frames.sort(key=lambda item: tuple(item["bbox"]))  # deterministic order
    assert frames, "expected at least one maroon frame"
    return frames


def _grow_points(
    base: np.ndarray,
    seed: tuple[int, int],
    directions: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    steps: int,
    protected_rows: int,
) -> set[tuple[int, int]]:
    g = np.asarray(base, dtype=int)
    h, w = g.shape

    occupied = {seed}
    frontier = {seed}
    for _ in range(int(steps)):
        nxt: set[tuple[int, int]] = set()
        for r, c in frontier:
            for dr, dc in directions:
                rr, cc = r + dr, c + dc
                while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == MAROON:
                    rr += dr
                    cc += dc
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if rr < protected_rows:
                    continue
                if int(g[rr, cc]) == MAROON:
                    continue
                nxt.add((rr, cc))
        nxt -= occupied
        if not nxt:
            break
        occupied |= nxt
        frontier = nxt
    return occupied


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    legend = _decode_legend(g)
    frames = _find_frames(g)

    out = g.copy()
    for frame in frames:
        color = int(frame["color"])
        assert color in legend, "seed color must appear in the legend"
        seed = tuple(frame["seed"])
        steps = int(frame["ih"]) * int(frame["iw"])
        grown = _grow_points(g, seed, legend[color], steps, protected_rows=3)
        for r, c in grown:
            if r < 3 or int(g[r, c]) == MAROON:
                continue
            if (r, c) != seed and int(g[r, c]) != BLACK:
                continue
            out[r, c] = color
    return out


def _choose_directions(rng: random.Random) -> tuple[tuple[int, int], ...]:
    count = rng.choice(DIRECTION_COUNTS)
    return tuple(rng.sample(DIRS8, count))


def _simulate_object(
    ih: int,
    iw: int,
    seed_inner: tuple[int, int],
    directions: tuple[tuple[int, int], ...],
) -> dict[str, object]:
    steps = ih * iw
    pad = 2 * steps + 4
    h = ih + 2 + 2 * pad
    w = iw + 2 + 2 * pad
    top = pad
    left = pad

    base = np.full((h, w), BLACK, dtype=int)
    base[top, left : left + iw + 2] = MAROON
    base[top + ih + 1, left : left + iw + 2] = MAROON
    base[top : top + ih + 2, left] = MAROON
    base[top : top + ih + 2, left + iw + 1] = MAROON

    seed_rel = (1 + seed_inner[0], 1 + seed_inner[1])
    seed_abs = (top + seed_rel[0], left + seed_rel[1])
    base[seed_abs] = 1
    occupied_abs = _grow_points(base, seed_abs, directions, steps, protected_rows=0)

    frame_cells = {(r, c) for r in range(ih + 2) for c in range(iw + 2)}
    occupied_rel = {(r - top, c - left) for r, c in occupied_abs}
    reserved = frame_cells | occupied_rel
    r0, c0, r1, c1 = _bbox(reserved)
    return {
        "ih": ih,
        "iw": iw,
        "seed_rel": seed_rel,
        "occupied_rel": occupied_rel,
        "reserved": reserved,
        "bbox": (r0, c0, r1, c1),
    }


def _place_objects(
    objects: list[dict[str, object]],
    grid_h: int,
    grid_w: int,
    rng: random.Random,
) -> list[tuple[dict[str, object], int, int]] | None:
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    occupied[:LEGEND_RESERVED_ROWS, :] = True
    placements: list[tuple[dict[str, object], int, int]] = []

    order = sorted(
        objects,
        key=lambda obj: (
            -((int(obj["bbox"][2]) - int(obj["bbox"][0]) + 1) * (int(obj["bbox"][3]) - int(obj["bbox"][1]) + 1)),
            -len(obj["reserved"]),
        ),
    )

    for obj in order:
        r0, c0, r1, c1 = obj["bbox"]
        min_roff = LEGEND_RESERVED_ROWS - r0
        max_roff = grid_h - 2 - r1
        min_coff = 1 - c0
        max_coff = grid_w - 2 - c1
        if min_roff > max_roff or min_coff > max_coff:
            return None

        for _ in range(PLACE_RETRIES):
            roff = rng.randint(min_roff, max_roff)
            coff = rng.randint(min_coff, max_coff)
            coords = [(r + roff, c + coff) for r, c in obj["reserved"]]
            if any(bool(occupied[r, c]) for r, c in coords):
                continue
            for r, c in coords:
                occupied[r, c] = True
            placements.append((obj, roff, coff))
            break
        else:
            return None

    return placements


def _build_legend(
    grid: np.ndarray,
    legend_colors: list[int],
    legend_dirs: dict[int, tuple[tuple[int, int], ...]],
    rng: random.Random,
) -> None:
    content_w = 3 * len(legend_colors) + max(0, len(legend_colors) - 1)
    start = rng.randint(0, grid.shape[1] - content_w)
    for i, color in enumerate(legend_colors):
        left = start + 4 * i
        center_c = left + 1
        grid[1, center_c] = color
        for dr, dc in legend_dirs[color]:
            grid[1 + dr, center_c + dc] = GREY


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_frames = rng.choice((1, 2, 2, 3))
    num_legend = rng.randint(num_frames, min(len(GEN_COLORS), num_frames + 2))
    legend_colors = rng.sample(list(GEN_COLORS), num_legend)
    legend_dirs = {color: _choose_directions(rng) for color in legend_colors}
    if num_legend > 1 and len({legend_dirs[color] for color in legend_colors}) == 1:
        legend_dirs[legend_colors[-1]] = _choose_directions(rng)

    frame_colors = rng.sample(legend_colors, num_frames)
    objects: list[dict[str, object]] = []
    for color in frame_colors:
        ih, iw = rng.choice(INTERIOR_SHAPES)
        seed_inner = (rng.randrange(ih), rng.randrange(iw))
        obj = _simulate_object(ih, iw, seed_inner, legend_dirs[color])
        obj["color"] = color
        objects.append(obj)

    legend_w = 3 * num_legend + max(0, num_legend - 1)
    max_obj_h = max(int(obj["bbox"][2]) - int(obj["bbox"][0]) + 1 for obj in objects)
    max_obj_w = max(int(obj["bbox"][3]) - int(obj["bbox"][1]) + 1 for obj in objects)
    total_obj_h = sum(int(obj["bbox"][2]) - int(obj["bbox"][0]) + 1 for obj in objects)
    total_obj_w = sum(int(obj["bbox"][3]) - int(obj["bbox"][1]) + 1 for obj in objects)

    grid_h = max(16, LEGEND_RESERVED_ROWS + 2 + max_obj_h, LEGEND_RESERVED_ROWS + 2 + (2 * total_obj_h + 2) // 3)
    grid_w = max(legend_w + 2, max_obj_w + 3, 4 + (2 * total_obj_w + 2) // 3)
    grid_h += rng.randint(0, 3)
    grid_w += rng.randint(0, 4)

    placements = None
    for _ in range(10):
        placements = _place_objects(objects, grid_h, grid_w, rng)
        if placements is not None:
            break
        grid_h += 2
        grid_w += 2
    if placements is None:
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    _build_legend(grid, legend_colors, legend_dirs, rng)

    for obj, roff, coff in placements:
        ih = int(obj["ih"])
        iw = int(obj["iw"])
        color = int(obj["color"])
        grid[roff, coff : coff + iw + 2] = MAROON
        grid[roff + ih + 1, coff : coff + iw + 2] = MAROON
        grid[roff : roff + ih + 2, coff] = MAROON
        grid[roff : roff + ih + 2, coff + iw + 1] = MAROON
        sr, sc = obj["seed_rel"]
        grid[roff + sr, coff + sc] = color

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    if not np.array_equal(out[:3, :], grid[:3, :]):
        return None
    if not np.array_equal(out[grid == MAROON], grid[grid == MAROON]):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle")
