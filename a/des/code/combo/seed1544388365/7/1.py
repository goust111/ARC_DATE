from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRECTIONS = ("up", "down", "left", "right")
VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((int(r0), int(c)))
        cells.add((int(r1), int(c)))
    for r in range(r0, r1 + 1):
        cells.add((int(r), int(c0)))
        cells.add((int(r), int(c1)))
    return cells


def _gap_direction(gap: tuple[int, int], r0: int, c0: int, r1: int, c1: int) -> str | None:
    r, c = gap
    sides = []
    if r == r0 and c0 < c < c1:
        sides.append("up")
    if r == r1 and c0 < c < c1:
        sides.append("down")
    if c == c0 and r0 < r < r1:
        sides.append("left")
    if c == c1 and r0 < r < r1:
        sides.append("right")
    if len(sides) != 1:
        return None
    return sides[0]


def _mirror_patch(patch: np.ndarray, direction: str) -> np.ndarray:
    if direction in ("left", "right"):
        return np.fliplr(patch)
    return np.flipud(patch)


def _copy_origin(
    interior_top: int,
    interior_left: int,
    interior_h: int,
    interior_w: int,
    gap: tuple[int, int],
    direction: str,
    anchor: tuple[int, int],
) -> tuple[int, int]:
    gap_r, gap_c = gap
    anchor_r, anchor_c = anchor
    if direction == "right":
        gap_offset = gap_r - interior_top
        return int(anchor_r - gap_offset), int(anchor_c)
    if direction == "left":
        gap_offset = gap_r - interior_top
        return int(anchor_r - gap_offset), int(anchor_c - (interior_w - 1))
    if direction == "down":
        gap_offset = gap_c - interior_left
        return int(anchor_r), int(anchor_c - gap_offset)
    gap_offset = gap_c - interior_left
    return int(anchor_r - (interior_h - 1)), int(anchor_c - gap_offset)


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _components4(grid == GREY):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        perimeter = _perimeter_cells(r0, c0, r1, c1)
        comp_set = set(comp)
        if not comp_set.issubset(perimeter):
            continue
        missing = sorted(perimeter - comp_set)
        if len(missing) != 1:
            continue
        gap = missing[0]
        direction = _gap_direction(gap, r0, c0, r1, c1)
        if direction is None or int(grid[gap]) != BG:
            continue

        interior = np.asarray(grid[r0 + 1 : r1, c0 + 1 : c1], dtype=int)
        if interior.size == 0 or np.any(interior == GREY):
            continue
        seeds = np.argwhere((interior != BG) & (interior != GREY))
        if int(len(seeds)) != 1:
            continue
        sr, sc = map(int, seeds[0])
        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "gap": (int(gap[0]), int(gap[1])),
                "direction": direction,
                "interior_top": int(r0 + 1),
                "interior_left": int(c0 + 1),
                "interior": interior,
                "seed_rel": (int(sr), int(sc)),
                "seed_color": int(interior[sr, sc]),
            }
        )
    frames.sort(key=lambda fr: (fr["bbox"][0], fr["bbox"][1], fr["bbox"][2], fr["bbox"][3]))
    return frames


def _find_anchor(
    grid: np.ndarray,
    gap: tuple[int, int],
    direction: str,
    color: int,
) -> tuple[int, int] | None:
    h, w = grid.shape
    dr, dc = VECTORS[direction]
    r, c = gap[0] + dr, gap[1] + dc
    while 0 <= r < h and 0 <= c < w:
        if int(grid[r, c]) == color:
            return (int(r), int(c))
        r += dr
        c += dc
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each framed interior through its border gap and place the copy using the
    same-colored outside anchor found on the gap's outward ray.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    for frame in _find_frames(g):
        anchor = _find_anchor(g, frame["gap"], frame["direction"], frame["seed_color"])
        if anchor is None:
            continue

        patch = _mirror_patch(frame["interior"], frame["direction"])
        ph, pw = patch.shape
        top, left = _copy_origin(
            frame["interior_top"],
            frame["interior_left"],
            ph,
            pw,
            frame["gap"],
            frame["direction"],
            anchor,
        )
        if top < 0 or left < 0 or top + ph > h or left + pw > w:
            continue

        colored = np.argwhere(patch != BG)
        for rr, cc in colored:
            out[int(top + rr), int(left + cc)] = int(patch[int(rr), int(cc)])

    return out


def _rect_cells(top: int, left: int, height: int, width: int) -> set[tuple[int, int]]:
    return {
        (int(r), int(c))
        for r in range(top, top + height)
        for c in range(left, left + width)
    }


def _ray_cells(gap: tuple[int, int], direction: str, distance: int) -> list[tuple[int, int]]:
    dr, dc = VECTORS[direction]
    return [
        (int(gap[0] + dr * step), int(gap[1] + dc * step))
        for step in range(1, distance + 1)
    ]


def _copied_seed_position(spec: dict) -> tuple[int, int]:
    seed_r, seed_c = spec["seed_rel"]
    height = spec["interior_h"]
    width = spec["interior_w"]
    copy_top, copy_left = spec["copy_origin"]
    if spec["direction"] in ("left", "right"):
        mr, mc = seed_r, width - 1 - seed_c
    else:
        mr, mc = height - 1 - seed_r, seed_c
    return (int(copy_top + mr), int(copy_left + mc))


def _expected_output(inp: np.ndarray, specs: Iterable[dict]) -> np.ndarray:
    out = np.asarray(inp, dtype=int).copy()
    for spec in specs:
        rr, cc = _copied_seed_position(spec)
        out[rr, cc] = int(spec["color"])
    return out


def _sample_seed(rng: random.Random, direction: str, height: int, width: int) -> tuple[int, int] | None:
    if direction == "right":
        if width < 2:
            return None
        return int(rng.randrange(height)), int(rng.randrange(width - 1))
    if direction == "left":
        if width < 2:
            return None
        return int(rng.randrange(height)), int(rng.randrange(1, width))
    if direction == "down":
        if height < 2:
            return None
        return int(rng.randrange(height - 1)), int(rng.randrange(width))
    if height < 2:
        return None
    return int(rng.randrange(1, height)), int(rng.randrange(width))


def _feasible_position_range(
    grid_h: int,
    grid_w: int,
    direction: str,
    height: int,
    width: int,
    distance: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if direction == "right":
        r0_min, r0_max = 0, grid_h - (height + 2)
        c0_min, c0_max = 0, grid_w - (2 * width + distance + 1)
    elif direction == "left":
        r0_min, r0_max = 0, grid_h - (height + 2)
        c0_min, c0_max = width + distance - 1, grid_w - (width + 2)
    elif direction == "down":
        r0_min, r0_max = 0, grid_h - (2 * height + distance + 1)
        c0_min, c0_max = 0, grid_w - (width + 2)
    else:
        r0_min, r0_max = height + distance - 1, grid_h - (height + 2)
        c0_min, c0_max = 0, grid_w - (width + 2)
    if r0_min > r0_max or c0_min > c0_max:
        return None
    return (int(r0_min), int(r0_max)), (int(c0_min), int(c0_max))


def _build_spec(
    grid_h: int,
    grid_w: int,
    color: int,
    direction: str,
    r0: int,
    c0: int,
    height: int,
    width: int,
    gap_offset: int,
    seed_rel: tuple[int, int],
    distance: int,
) -> dict:
    inner_top = r0 + 1
    inner_left = c0 + 1
    if direction == "right":
        gap = (inner_top + gap_offset, c0 + width + 1)
        anchor = (gap[0], gap[1] + distance)
    elif direction == "left":
        gap = (inner_top + gap_offset, c0)
        anchor = (gap[0], gap[1] - distance)
    elif direction == "down":
        gap = (r0 + height + 1, inner_left + gap_offset)
        anchor = (gap[0] + distance, gap[1])
    else:
        gap = (r0, inner_left + gap_offset)
        anchor = (gap[0] - distance, gap[1])
    copy_origin = _copy_origin(inner_top, inner_left, height, width, gap, direction, anchor)
    return {
        "color": int(color),
        "direction": direction,
        "outer_top": int(r0),
        "outer_left": int(c0),
        "interior_h": int(height),
        "interior_w": int(width),
        "gap": (int(gap[0]), int(gap[1])),
        "anchor": (int(anchor[0]), int(anchor[1])),
        "gap_offset": int(gap_offset),
        "seed_rel": (int(seed_rel[0]), int(seed_rel[1])),
        "copy_origin": (int(copy_origin[0]), int(copy_origin[1])),
    }


def _protected_cells(spec: dict) -> set[tuple[int, int]]:
    frame = _rect_cells(
        spec["outer_top"],
        spec["outer_left"],
        spec["interior_h"] + 2,
        spec["interior_w"] + 2,
    )
    copy_top, copy_left = spec["copy_origin"]
    copy_box = _rect_cells(copy_top, copy_left, spec["interior_h"], spec["interior_w"])
    ray = set(_ray_cells(spec["gap"], spec["direction"], abs(spec["anchor"][0] - spec["gap"][0]) + abs(spec["anchor"][1] - spec["gap"][1])))
    return frame | copy_box | ray


def _try_place_one(
    rng: random.Random,
    grid_h: int,
    grid_w: int,
    color: int,
    direction: str,
    reserved: set[tuple[int, int]],
) -> dict | None:
    for _ in range(500):
        height = rng.randint(2, min(6, grid_h - 2))
        width = rng.randint(2, min(6, grid_w - 2))
        if direction in ("left", "right"):
            max_dist = min(8, grid_w - (2 * width + 1))
            gap_offset = int(rng.randrange(height))
        else:
            max_dist = min(8, grid_h - (2 * height + 1))
            gap_offset = int(rng.randrange(width))
        if max_dist < 1:
            continue
        distance = rng.randint(1, max_dist)
        seed_rel = _sample_seed(rng, direction, height, width)
        if seed_rel is None:
            continue

        ranges = _feasible_position_range(grid_h, grid_w, direction, height, width, distance)
        if ranges is None:
            continue
        (r0_min, r0_max), (c0_min, c0_max) = ranges
        r0 = rng.randint(r0_min, r0_max)
        c0 = rng.randint(c0_min, c0_max)
        spec = _build_spec(
            grid_h,
            grid_w,
            color,
            direction,
            r0,
            c0,
            height,
            width,
            gap_offset,
            seed_rel,
            distance,
        )

        protected = _protected_cells(spec)
        copied_seed = _copied_seed_position(spec)
        if copied_seed == spec["anchor"]:
            continue
        if protected & reserved:
            continue
        reserved.update(protected)
        return spec
    return None


def _render_input(grid_h: int, grid_w: int, specs: Iterable[dict]) -> np.ndarray:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for spec in specs:
        r0 = spec["outer_top"]
        c0 = spec["outer_left"]
        h = spec["interior_h"]
        w = spec["interior_w"]
        r1 = r0 + h + 1
        c1 = c0 + w + 1
        for pr, pc in _perimeter_cells(r0, c0, r1, c1):
            if (pr, pc) != spec["gap"]:
                grid[pr, pc] = GREY
        seed_r = r0 + 1 + spec["seed_rel"][0]
        seed_c = c0 + 1 + spec["seed_rel"][1]
        grid[seed_r, seed_c] = spec["color"]
        grid[spec["anchor"]] = spec["color"]
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(16, 30)
    grid_w = rng.randint(16, 30)
    max_objects = 2 if min(grid_h, grid_w) < 20 else 3 if min(grid_h, grid_w) < 26 else 4
    num_objects = rng.choice([1, 2, 2, max_objects])
    num_objects = min(int(num_objects), len(PALETTE), len(DIRECTIONS))

    colors = rng.sample(PALETTE, k=num_objects)
    directions = rng.sample(list(DIRECTIONS), k=num_objects)
    reserved: set[tuple[int, int]] = set()
    specs: list[dict] = []

    for color, direction in zip(colors, directions):
        spec = _try_place_one(rng, grid_h, grid_w, color, direction, reserved)
        if spec is None:
            return None
        specs.append(spec)

    inp = _render_input(grid_h, grid_w, specs)
    detected = _find_frames(inp)
    if len(detected) != len(specs):
        return None
    expected = _expected_output(inp, specs)
    produced = main(inp)
    if np.array_equal(inp, produced):
        return None
    if not np.array_equal(produced, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
