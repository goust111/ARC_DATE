from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 5
COUNT_TO_BORDER = {1: 2, 2: 1, 3: 3}
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rectangular_ring(coords: list[tuple[int, int]]) -> bool:
    if not coords:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False

    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    return set(coords) == expected


def _wall_bboxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    walls = []
    for comp in _components(grid == WALL):
        if _is_rectangular_ring(comp):
            walls.append(_bbox(comp))
    walls.sort()
    return walls


def _seed_wall(grid: np.ndarray, r: int, c: int) -> str | None:
    touches = []
    if grid[r - 1, c] == WALL:
        touches.append("top")
    if grid[r + 1, c] == WALL:
        touches.append("bottom")
    if grid[r, c - 1] == WALL:
        touches.append("left")
    if grid[r, c + 1] == WALL:
        touches.append("right")
    if len(touches) != 1:
        return None
    return touches[0]


def _draw_outer_border(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    out[r0 - 1, c0 - 1 : c1 + 2] = color
    out[r1 + 1, c0 - 1 : c1 + 2] = color
    out[r0 - 1 : r1 + 2, c0 - 1] = color
    out[r0 - 1 : r1 + 2, c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow wall-adjacent seed pixels into full-width/full-height stripes, then
    add an outside border whose color depends on the number of stripes.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for r0, c0, r1, c1 in _wall_bboxes(g):
        seeds: list[tuple[int, int, int, str]] = []
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                color = int(g[r, c])
                if color in (BG, WALL):
                    continue
                wall = _seed_wall(g, r, c)
                if wall is not None:
                    seeds.append((r, c, color, wall))

        horizontal = [(r, color) for r, _c, color, wall in seeds if wall in ("left", "right")]
        vertical = [(c, color) for _r, c, color, wall in seeds if wall in ("top", "bottom")]

        for row, color in horizontal:
            out[row, c0 + 1 : c1] = color
        for col, color in vertical:
            out[r0 + 1 : r1, col] = color

        border_color = COUNT_TO_BORDER.get(len(seeds))
        if border_color is not None:
            _draw_outer_border(out, r0, c0, r1, c1, border_color)

    return out


PATTERNS = {
    1: [(1, 0), (0, 1)],
    2: [(2, 0), (1, 1), (0, 2)],
    3: [(3, 0), (2, 1), (1, 2), (0, 3)],
}
MAX_ATTEMPTS = 300


def _sample_frame_spec() -> dict:
    seed_count = random.randint(1, 3)
    hcount, vcount = random.choice(PATTERNS[seed_count])

    h_lo = max(5, hcount + 4)
    w_lo = max(5, vcount + 4)
    h_hi = max(h_lo, min(11, hcount + 7))
    w_hi = max(w_lo, min(12, vcount + 8))
    outer_h = random.randint(h_lo, h_hi)
    outer_w = random.randint(w_lo, w_hi)

    rows = random.sample(list(range(2, outer_h - 2)), hcount)
    cols = random.sample(list(range(2, outer_w - 2)), vcount)

    seed_specs: list[dict] = []
    for row in rows:
        wall = random.choice(("left", "right"))
        local_c = 1 if wall == "left" else outer_w - 2
        seed_specs.append({"local_r": row, "local_c": local_c, "wall": wall})
    for col in cols:
        wall = random.choice(("top", "bottom"))
        local_r = 1 if wall == "top" else outer_h - 2
        seed_specs.append({"local_r": local_r, "local_c": col, "wall": wall})

    colors = random.sample(SEED_COLORS, seed_count)
    random.shuffle(seed_specs)
    for spec, color in zip(seed_specs, colors, strict=True):
        spec["color"] = color

    return {
        "outer_h": outer_h,
        "outer_w": outer_w,
        "reserve_h": outer_h + 4,
        "reserve_w": outer_w + 4,
        "seeds": seed_specs,
    }


def _place_frames(height: int, width: int, specs: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    placed: list[dict] = []

    for spec in sorted(specs, key=lambda item: item["reserve_h"] * item["reserve_w"], reverse=True):
        reserve_h = int(spec["reserve_h"])
        reserve_w = int(spec["reserve_w"])
        if reserve_h > height or reserve_w > width:
            return None

        ok = False
        for _ in range(120):
            rr = random.randint(0, height - reserve_h)
            cc = random.randint(0, width - reserve_w)
            if blocked[rr : rr + reserve_h, cc : cc + reserve_w].any():
                continue

            r0 = rr + 2
            c0 = cc + 2
            r1 = r0 + int(spec["outer_h"]) - 1
            c1 = c0 + int(spec["outer_w"]) - 1

            grid[r0, c0 : c1 + 1] = WALL
            grid[r1, c0 : c1 + 1] = WALL
            grid[r0 : r1 + 1, c0] = WALL
            grid[r0 : r1 + 1, c1] = WALL

            realized_seeds = []
            for seed in spec["seeds"]:
                sr = r0 + int(seed["local_r"])
                sc = c0 + int(seed["local_c"])
                color = int(seed["color"])
                grid[sr, sc] = color
                realized_seeds.append({"r": sr, "c": sc, "color": color, "wall": seed["wall"]})

            blocked[rr : rr + reserve_h, cc : cc + reserve_w] = True
            placed.append({"bbox": (r0, c0, r1, c1), "seeds": realized_seeds})
            ok = True
            break

        if not ok:
            return None

    return grid, placed


def _expected_output(inp: np.ndarray, frames: list[dict]) -> np.ndarray:
    out = inp.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        horizontal = [(seed["r"], seed["color"]) for seed in frame["seeds"] if seed["wall"] in ("left", "right")]
        vertical = [(seed["c"], seed["color"]) for seed in frame["seeds"] if seed["wall"] in ("top", "bottom")]

        for row, color in horizontal:
            out[row, c0 + 1 : c1] = color
        for col, color in vertical:
            out[r0 + 1 : r1, col] = color

        _draw_outer_border(out, r0, c0, r1, c1, COUNT_TO_BORDER[len(frame["seeds"])])
    return out


def _valid_layout(inp: np.ndarray, frames: list[dict]) -> bool:
    if not np.array_equal(_wall_bboxes(inp), sorted(frame["bbox"] for frame in frames)):
        return False

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        if r0 < 2 or c0 < 2 or r1 >= inp.shape[0] - 2 or c1 >= inp.shape[1] - 2:
            return False
        for seed in frame["seeds"]:
            if _seed_wall(inp, int(seed["r"]), int(seed["c"])) != seed["wall"]:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    num_frames = random.randint(1, 3)
    specs = [_sample_frame_spec() for _ in range(num_frames)]

    min_h = max(max(spec["reserve_h"] for spec in specs) + 1, 16 + 2 * (num_frames - 1))
    min_w = max(max(spec["reserve_w"] for spec in specs) + 1, 16 + 2 * (num_frames - 1))
    height = random.randint(min_h, max(min_h, 30))
    width = random.randint(min_w, max(min_w, 30))

    placed = _place_frames(height, width, specs)
    if placed is None:
        return None
    inp, frames = placed

    if not _valid_layout(inp, frames):
        return None

    expected = _expected_output(inp, frames)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
