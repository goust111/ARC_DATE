from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDES = ("top", "bottom", "left", "right")
SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _side_cells(box: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    if side == "top":
        return [(r0, c) for c in range(c0, c1 + 1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0, c1 + 1)]
    if side == "left":
        return [(r, c0) for r in range(r0, r1 + 1)]
    return [(r, c1) for r in range(r0, r1 + 1)]


def _gap_cell(box: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if side == "top":
        return r0, (c0 + c1) // 2
    if side == "bottom":
        return r1, (c0 + c1) // 2
    if side == "left":
        return (r0 + r1) // 2, c0
    return (r0 + r1) // 2, c1


def _wall_color(g: np.ndarray) -> int:
    counts = Counter(int(v) for v in g.ravel() if int(v) != BG)
    assert counts, "expected at least one wall and seed"
    return int(counts.most_common(1)[0][0])


def _wall_infos(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    wall = _wall_color(g)
    infos: list[dict] = []
    for comp in _components(g == wall):
        box = _bbox(comp)
        r0, c0, r1, c1 = box
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue

        comp_set = set(comp)
        assert all(r in (r0, r1) or c in (c0, c1) for r, c in comp), (
            "wall pixels must lie on a rectangle boundary"
        )

        counts = {
            side: sum((r, c) in comp_set for r, c in _side_cells(box, side))
            for side in SIDES
        }
        weak_count = min(counts.values())
        weak_sides = [side for side, count in counts.items() if count == weak_count]
        assert len(weak_sides) == 1, "expected one uniquely weakest side"
        weak = weak_sides[0]
        strong_counts = [count for side, count in counts.items() if side != weak]
        assert len(set(strong_counts)) == 1 and strong_counts[0] == weak_count + 1, (
            "weak side must have exactly one fewer wall pixel"
        )

        inside = g[r0 + 1 : r1, c0 + 1 : c1]
        seeds = [
            (r0 + 1 + int(rr), c0 + 1 + int(cc), int(inside[rr, cc]))
            for rr, cc in np.argwhere((inside != BG) & (inside != wall))
        ]
        assert len(seeds) == 1, "each wall must contain exactly one seed pixel"
        sr, sc, seed_color = seeds[0]
        gap = _gap_cell(box, weak)

        dr, dc = SIDE_TO_DIR[weak]
        if dr:
            assert sc == gap[1], "seed must align vertically with the weak-side gap"
        else:
            assert sr == gap[0], "seed must align horizontally with the weak-side gap"

        assert 0 <= gap[0] < h and 0 <= gap[1] < w
        infos.append(
            {
                "box": box,
                "weak": weak,
                "gap": gap,
                "seed": (sr, sc),
                "seed_color": seed_color,
                "dir": (dr, dc),
            }
        )

    assert infos, "expected at least one rectangular wall"
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open each rectangular wall on its weakest side and fire the enclosed seed color outward as a ray.

    Input structure:
    - The background is black.
    - One wall color forms several separated one-pixel-thick rectangular boundary
      components; for each rectangle, exactly one side contains one fewer wall
      pixel than the other three sides.
    - Each rectangle encloses exactly one single-pixel seed in a unique non-wall
      color, aligned with the center of the weakest side.

    Rule:
    - For every wall, compare the four side pixel counts and identify the unique
      minimum side as the weak side.
    - Open the center cell of that weak side, then cast a one-pixel-wide ray from
      the seed in the corresponding cardinal direction.
    - Paint black cells on the ray with the seed color, keep the seed pixel in
      place, and stop before painting any wall pixel from another rectangle.

    Key insight:
    - The ray direction is not given by the seed color or position alone; it is
      determined by first counting the wall pixels on each side and finding the
      uniquely shortest side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert max(h, w) <= 30, "ARC grids must fit within 30x30"

    infos = _wall_infos(g)
    wall = _wall_color(g)
    out = g.copy()

    for info in infos:
        gr, gc = info["gap"]
        out[gr, gc] = BG

    for info in infos:
        sr, sc = info["seed"]
        seed_color = int(info["seed_color"])
        dr, dc = info["dir"]
        r, c = sr + dr, sc + dc
        while 0 <= r < h and 0 <= c < w:
            if int(out[r, c]) == wall:
                break
            if int(out[r, c]) == BG:
                out[r, c] = seed_color
            r += dr
            c += dc

    return out


def _paint_wall(grid: np.ndarray, r0: int, c0: int, size: int, wall: int) -> None:
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    grid[r0, c0 : c1 + 1] = wall
    grid[r1, c0 : c1 + 1] = wall
    grid[r0 : r1 + 1, c0] = wall
    grid[r0 : r1 + 1, c1] = wall


def _weak_side_options(box: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    return [_gap_cell(box, side)]


def _ray_cells(shape: tuple[int, int], seed: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    r, c = seed[0] + dr, seed[1] + dc
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _dilated_box_slice(
    shape: tuple[int, int], r0: int, c0: int, r1: int, c1: int, pad: int
) -> tuple[slice, slice]:
    h, w = shape
    return slice(max(0, r0 - pad), min(h, r1 + pad + 1)), slice(max(0, c0 - pad), min(w, c1 + pad + 1))


def _try_generate() -> np.ndarray | None:
    h = random.randint(15, 30)
    w = random.randint(15, 30)
    wall_count = random.randint(2, 4)
    wall = random.choice([1, 5, 8])
    seed_palette = [c for c in range(1, 10) if c != wall]
    random.shuffle(seed_palette)
    if len(seed_palette) < wall_count:
        return None

    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    specs: list[dict] = []
    dirs = list(SIDES)
    random.shuffle(dirs)

    for idx in range(wall_count):
        side = dirs[idx % len(dirs)]
        size = random.choice([5, 7, 9])
        if size + 2 > h or size + 2 > w:
            return None
        placed = False
        for _ in range(150):
            r0 = random.randint(1, h - size - 1)
            c0 = random.randint(1, w - size - 1)
            r1 = r0 + size - 1
            c1 = c0 + size - 1
            rs, cs = _dilated_box_slice((h, w), r0, c0, r1, c1, pad=1)
            if occupied[rs, cs].any():
                continue
            box = (r0, c0, r1, c1)
            gap = _gap_cell(box, side)
            dr, dc = SIDE_TO_DIR[side]
            if dr:
                seed = (random.randint(r0 + 1, r1 - 1), gap[1])
            else:
                seed = (gap[0], random.randint(c0 + 1, c1 - 1))
            specs.append(
                {
                    "box": box,
                    "size": size,
                    "side": side,
                    "seed": seed,
                    "color": seed_palette[idx],
                }
            )
            occupied[rs, cs] = True
            placed = True
            break
        if not placed:
            return None

    for spec in specs:
        r0, c0, r1, c1 = spec["box"]
        _paint_wall(grid, r0, c0, spec["size"], wall)
        missing_choices = _weak_side_options(spec["box"], spec["side"])
        if not missing_choices:
            return None
        mr, mc = random.choice(missing_choices)
        grid[mr, mc] = BG
        sr, sc = spec["seed"]
        grid[sr, sc] = spec["color"]

    # Keep the examples legible: reject cases where a ray would run through a
    # different wall's already-missing weak pixel and enter its interior.
    wall_mask = grid == wall
    for spec in specs:
        direction = SIDE_TO_DIR[spec["side"]]
        for r, c in _ray_cells((h, w), spec["seed"], direction):
            if wall_mask[r, c]:
                break
            for other in specs:
                if other is spec:
                    continue
                or0, oc0, or1, oc1 = other["box"]
                if or0 < r < or1 and oc0 < c < oc1:
                    return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _try_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid weak-wall ray puzzle")
