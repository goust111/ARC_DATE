import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
OBSTACLE = 9
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8]
SIDES = ("up", "down", "left", "right")
SIDE_TO_STEP = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
SLOT_DIRS = {
    "tl": ("up", "left"),
    "tr": ("up", "right"),
    "bl": ("down", "left"),
    "br": ("down", "right"),
}
GRID_MIN = 20
GRID_MAX = 30
SLOT_MIN = 9
MAX_ATTEMPTS = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_boundary(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c0 + w):
        cells.add((r0, c))
        cells.add((r0 + h - 1, c))
    for r in range(r0, r0 + h):
        cells.add((r, c0))
        cells.add((r, c0 + w - 1))
    return cells


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict[str, object]] = []
    for comp in _components4(g == FRAME):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        boundary = _frame_boundary(r0, c0, h, w)
        comp_set = set(comp)
        missing = list(boundary - comp_set)
        assert len(missing) == 1, "each frame must have exactly one missing border pixel"
        assert comp_set == boundary - set(missing), "grey component must be a rectangular border with one gap"
        gap = missing[0]

        if gap[0] == r0 and c0 < gap[1] < c1:
            side = "up"
        elif gap[0] == r1 and c0 < gap[1] < c1:
            side = "down"
        elif gap[1] == c0 and r0 < gap[0] < r1:
            side = "left"
        elif gap[1] == c1 and r0 < gap[0] < r1:
            side = "right"
        else:
            raise AssertionError("gap must sit on the middle of one side")

        seed_cells = [
            (r, c)
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
            if int(g[r, c]) not in (BG, FRAME)
        ]
        assert len(seed_cells) == 1, "each frame must contain exactly one colored seed"
        seed = seed_cells[0]
        if side in ("up", "down"):
            assert seed[1] == gap[1], "vertical projection requires seed and gap to share a column"
        else:
            assert seed[0] == gap[0], "horizontal projection requires seed and gap to share a row"

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "side": side,
                "step": SIDE_TO_STEP[side],
                "seed": seed,
            }
        )
    frames.sort(key=lambda item: (item["bbox"], item["seed"]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shoot each seed out through its frame gap and stop at maroon blockers.

    Input structure:
    - The background is black.
    - Grey 4-connected components are near-closed rectangular frames whose border is missing exactly one pixel on one side.
    - Each frame encloses exactly one non-black colored seed pixel, and isolated maroon pixels elsewhere act as blockers.

    Rule:
    - For each grey frame, locate the unique border gap; its side determines the shooting direction.
    - Starting from the enclosed seed, paint a straight ray in the seed color along that direction, filling the interior segment and the gap cell.
    - Continue painting outside the frame until the next step would leave the grid or hit a maroon obstacle; the obstacle itself stays maroon and the remaining grey border stays unchanged.

    Key insight:
    - The missing border pixel is both a directional cue and the exit door: the seed does not move, but its color projects straight through that single opening.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for frame in _find_frames(g):
        seed_r, seed_c = frame["seed"]
        dr, dc = frame["step"]
        color = int(g[seed_r, seed_c])
        r, c = seed_r, seed_c

        while True:
            r += dr
            c += dc
            if not _in_bounds(g.shape, r, c):
                break
            if int(g[r, c]) == OBSTACLE:
                break
            out[r, c] = color

    return out


def _slot_boxes(h: int, w: int, split_r: int, split_c: int) -> dict[str, tuple[int, int, int, int]]:
    return {
        "tl": (0, split_r - 1, 0, split_c - 1),
        "tr": (0, split_r - 1, split_c + 1, w - 1),
        "bl": (split_r + 1, h - 1, 0, split_c - 1),
        "br": (split_r + 1, h - 1, split_c + 1, w - 1),
    }


def _outside_cells(shape: tuple[int, int], r0: int, c0: int, h: int, w: int, gap: tuple[int, int], side: str) -> list[tuple[int, int]]:
    gh, gw = shape
    gr, gc = gap
    if side == "up":
        return [(r, gc) for r in range(gr - 1, -1, -1)]
    if side == "down":
        return [(r, gc) for r in range(r0 + h, gh)]
    if side == "left":
        return [(gr, c) for c in range(gc - 1, -1, -1)]
    return [(gr, c) for c in range(c0 + w, gw)]


def _make_object(
    rng: random.Random,
    shape: tuple[int, int],
    slot_box: tuple[int, int, int, int],
    slot_name: str,
    color: int,
) -> dict[str, object] | None:
    slot_r0, slot_r1, slot_c0, slot_c1 = slot_box
    slot_h = slot_r1 - slot_r0 + 1
    slot_w = slot_c1 - slot_c0 + 1

    for _ in range(80):
        side = rng.choice(SLOT_DIRS[slot_name])
        pad_top = 2 if side == "up" else 1
        pad_bottom = 2 if side == "down" else 1
        pad_left = 2 if side == "left" else 1
        pad_right = 2 if side == "right" else 1

        max_h = min(6, slot_h - pad_top - pad_bottom)
        max_w = min(6, slot_w - pad_left - pad_right)
        if max_h < 3 or max_w < 3:
            continue

        h = rng.randint(3, max_h)
        w = rng.randint(3, max_w)
        r_min = slot_r0 + pad_top
        r_max = slot_r1 - pad_bottom - h + 1
        c_min = slot_c0 + pad_left
        c_max = slot_c1 - pad_right - w + 1
        if r_min > r_max or c_min > c_max:
            continue

        r0 = rng.randint(r_min, r_max)
        c0 = rng.randint(c_min, c_max)
        if side in ("up", "down"):
            gap = (r0 if side == "up" else r0 + h - 1, c0 + rng.randint(1, w - 2))
            seed = (rng.randint(r0 + 1, r0 + h - 2), gap[1])
        else:
            gap = (r0 + rng.randint(1, h - 2), c0 if side == "left" else c0 + w - 1)
            seed = (gap[0], rng.randint(c0 + 1, c0 + w - 2))

        outside = _outside_cells(shape, r0, c0, h, w, gap, side)
        if len(outside) < 2:
            continue

        blocked = bool(rng.randint(0, 1))
        obstacle = None
        if blocked:
            obstacle_idx = rng.randint(1, len(outside) - 1)
            obstacle = outside[obstacle_idx]

        frame_cells = _frame_boundary(r0, c0, h, w)
        frame_cells.remove(gap)
        step = SIDE_TO_STEP[side]
        expected_ray = []
        rr, cc = seed
        while True:
            rr += step[0]
            cc += step[1]
            if not _in_bounds(shape, rr, cc):
                break
            if obstacle is not None and (rr, cc) == obstacle:
                break
            expected_ray.append((rr, cc))

        return {
            "bbox": (r0, c0, h, w),
            "side": side,
            "step": step,
            "gap": gap,
            "seed": seed,
            "color": color,
            "frame_cells": frame_cells,
            "ray_cells": expected_ray,
            "obstacle": obstacle,
        }
    return None


def _draw_input(shape: tuple[int, int], specs: list[dict[str, object]], extras: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        for r, c in spec["frame_cells"]:
            grid[r, c] = FRAME
        sr, sc = spec["seed"]
        grid[sr, sc] = spec["color"]
        if spec["obstacle"] is not None:
            orow, ocol = spec["obstacle"]
            grid[orow, ocol] = OBSTACLE
    for r, c in extras:
        grid[r, c] = OBSTACLE
    return grid


def _draw_output(inp: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        color = int(spec["color"])
        for r, c in spec["ray_cells"]:
            out[r, c] = color
    return out


def _extra_obstacle_candidates(grid: np.ndarray, forbidden: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = grid.shape
    candidates = []
    for r in range(h):
        for c in range(w):
            if (r, c) in forbidden or int(grid[r, c]) != BG:
                continue
            ok = True
            for nr, nc in _neighbors8(r, c):
                if _in_bounds(grid.shape, nr, nc) and (int(grid[nr, nc]) != BG or (nr, nc) in forbidden):
                    ok = False
                    break
            if ok:
                candidates.append((r, c))
    return candidates


def _validate_obstacles(grid: np.ndarray) -> bool:
    obstacle_cells = [tuple(map(int, cell)) for cell in np.argwhere(grid == OBSTACLE)]
    for r, c in obstacle_cells:
        for nr, nc in _neighbors8(r, c):
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) != BG:
                return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    split_r = rng.randint(SLOT_MIN, h - SLOT_MIN - 1)
    split_c = rng.randint(SLOT_MIN, w - SLOT_MIN - 1)
    boxes = _slot_boxes(h, w, split_r, split_c)

    frame_count = rng.randint(1, 4)
    slots = rng.sample(list(boxes), frame_count)
    colors = rng.sample(SEED_COLORS, frame_count)

    specs: list[dict[str, object]] = []
    for slot_name, color in zip(slots, colors):
        spec = _make_object(rng, (h, w), boxes[slot_name], slot_name, color)
        if spec is None:
            return None
        specs.append(spec)

    forbidden: set[tuple[int, int]] = set()
    for spec in specs:
        r0, c0, box_h, box_w = spec["bbox"]
        for r in range(r0, r0 + box_h):
            for c in range(c0, c0 + box_w):
                forbidden.add((r, c))
        forbidden |= set(spec["frame_cells"])
        forbidden.add(spec["seed"])
        forbidden.add(spec["gap"])
        forbidden |= set(spec["ray_cells"])
        if spec["obstacle"] is not None:
            forbidden.add(spec["obstacle"])

    inp = _draw_input((h, w), specs, extras=[])
    candidates = _extra_obstacle_candidates(inp, forbidden)
    rng.shuffle(candidates)
    extra_count = rng.randint(0, min(4, len(candidates)))
    extras = candidates[:extra_count]
    inp = _draw_input((h, w), specs, extras)
    expected = _draw_output(inp, specs)

    if not _validate_obstacles(inp):
        return None
    frames = _find_frames(inp)
    if len(frames) != frame_count:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
