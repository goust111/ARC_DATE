from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


_BG = 0
_FRAME = 5
_SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
_MAX_GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _find_frame_color(grid: np.ndarray, bg: int) -> int | None:
    counts = Counter(int(v) for v in grid.ravel() if int(v) != bg)
    if not counts:
        return None
    return max(counts, key=counts.get)


def _frame_regions(grid: np.ndarray, bg: int, frame_color: int) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    regions: list[dict[str, int]] = []
    for comp in _components(g == frame_color):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        if not np.all(g[r0, c0 : c1 + 1] == frame_color):
            continue
        if not np.all(g[r1, c0 : c1 + 1] == frame_color):
            continue
        if not np.all(g[r0 : r1 + 1, c0] == frame_color):
            continue
        if not np.all(g[r0 : r1 + 1, c1] == frame_color):
            continue
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0 or bool(np.any(interior == frame_color)):
            continue
        seeds = np.argwhere((interior != bg) & (interior != frame_color))
        if seeds.shape[0] != 1:
            continue
        sr = int(r0 + 1 + seeds[0, 0])
        sc = int(c0 + 1 + seeds[0, 1])
        regions.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "sr": sr,
                "sc": sc,
                "color": int(g[sr, sc]),
            }
        )
    regions.sort(key=lambda reg: (reg["r0"], reg["c0"]))
    return regions


def _paint_region(out: np.ndarray, region: dict[str, int]) -> None:
    r0 = int(region["r0"])
    c0 = int(region["c0"])
    r1 = int(region["r1"])
    c1 = int(region["c1"])
    sr = int(region["sr"])
    sc = int(region["sc"])
    color = int(region["color"])

    inner_r0 = r0 + 1
    inner_c0 = c0 + 1
    inner_r1 = r1 - 1
    inner_c1 = c1 - 1
    inner_h = inner_r1 - inner_r0 + 1
    inner_w = inner_c1 - inner_c0 + 1

    if inner_h == inner_w:
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            rr = sr
            cc = sc
            while inner_r0 <= rr <= inner_r1 and inner_c0 <= cc <= inner_c1:
                out[rr, cc] = color
                rr += dr
                cc += dc
    elif inner_h > inner_w:
        out[inner_r0 : inner_r1 + 1, sc] = color
    else:
        out[sr, inner_c0 : inner_c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each colored seed inside a grey frame according to the frame's aspect ratio.

    Input structure:
    - A black background surrounds several disconnected one-pixel-thick grey closed frames.
    - Each frame encloses a rectangular interior and contains exactly one non-black, non-grey seed pixel.
    - Frames can be square, vertically elongated, or horizontally elongated.

    Rule:
    - Identify each grey closed frame and classify its interior as square, tall rectangle, or wide rectangle.
    - In a square region, grow the seed along the four diagonal directions until the frame stops it, producing an X.
    - In a tall rectangle, fill the entire interior column containing the seed; in a wide rectangle, fill the entire interior row containing the seed.
    - Use the seed's own color for the growth, and keep the grey frames and black background unchanged.

    Key insight:
    - The frame geometry is the hidden instruction: the same kind of seed turns into a diagonal X, a vertical line, or a horizontal line depending only on the enclosing border shape.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")

    bg = int(Counter(int(v) for v in g.ravel()).most_common(1)[0][0])
    frame_color = _find_frame_color(g, bg)
    if frame_color is None:
        return g.copy()

    regions = _frame_regions(g, bg, frame_color)
    out = g.copy()
    for region in regions:
        _paint_region(out, region)
    return out


def _shape_sequence(count: int) -> list[str]:
    shapes = ["square", "tall", "wide"]
    offset = random.randrange(len(shapes))
    seq = [shapes[(offset + i) % len(shapes)] for i in range(count)]
    random.shuffle(seq)
    return seq


def _sample_region(shape: str, color: int) -> dict[str, int | str]:
    if shape == "square":
        side = random.choice([3, 5, 7, 9])
        inner_h = side
        inner_w = side
        seed_r = side // 2
        seed_c = side // 2
    elif shape == "tall":
        inner_w = random.randint(3, 5)
        inner_h = random.randint(inner_w + 2, min(9, inner_w + 5))
        seed_r = random.randint(0, inner_h - 1)
        seed_c = random.randint(0, inner_w - 1)
    else:
        inner_h = random.randint(3, 5)
        inner_w = random.randint(inner_h + 2, min(9, inner_h + 5))
        seed_r = random.randint(0, inner_h - 1)
        seed_c = random.randint(0, inner_w - 1)

    return {
        "shape": shape,
        "inner_h": int(inner_h),
        "inner_w": int(inner_w),
        "outer_h": int(inner_h + 2),
        "outer_w": int(inner_w + 2),
        "seed_r": int(seed_r),
        "seed_c": int(seed_c),
        "color": int(color),
    }


def _valid_positions(blocked: np.ndarray, outer_h: int, outer_w: int) -> list[tuple[int, int]]:
    h, w = blocked.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - outer_h):
        for c0 in range(1, w - outer_w):
            rr0 = r0 - 1
            cc0 = c0 - 1
            rr1 = r0 + outer_h + 1
            cc1 = c0 + outer_w + 1
            if not bool(blocked[rr0:rr1, cc0:cc1].any()):
                positions.append((r0, c0))
    return positions


def _place_region(grid: np.ndarray, blocked: np.ndarray, spec: dict[str, int | str], top: int, left: int) -> dict[str, int]:
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    color = int(spec["color"])
    seed_r = int(spec["seed_r"])
    seed_c = int(spec["seed_c"])

    bottom = top + outer_h - 1
    right = left + outer_w - 1
    grid[top, left : right + 1] = _FRAME
    grid[bottom, left : right + 1] = _FRAME
    grid[top : bottom + 1, left] = _FRAME
    grid[top : bottom + 1, right] = _FRAME
    abs_sr = top + 1 + seed_r
    abs_sc = left + 1 + seed_c
    grid[abs_sr, abs_sc] = color

    blocked[top - 1 : bottom + 2, left - 1 : right + 2] = True
    return {
        "r0": int(top),
        "c0": int(left),
        "r1": int(bottom),
        "c1": int(right),
        "sr": int(abs_sr),
        "sc": int(abs_sc),
        "color": int(color),
    }


def _build_expected(inp: np.ndarray, regions: list[dict[str, int]]) -> np.ndarray:
    out = inp.copy()
    for region in regions:
        _paint_region(out, region)
    return out


def _generate_one() -> np.ndarray | None:
    n_regions = random.randint(3, 6)
    shapes = _shape_sequence(n_regions)
    colors = random.sample(_SEED_COLORS, k=n_regions)
    specs = [_sample_region(shape, color) for shape, color in zip(shapes, colors, strict=False)]
    specs.sort(key=lambda spec: (-(int(spec["outer_h"]) * int(spec["outer_w"])), str(spec["shape"])))

    min_side = min(30, 12 + 2 * n_regions)
    height = random.randint(min_side, 30)
    width = random.randint(min_side, 30)
    grid = np.full((height, width), _BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    placed: list[dict[str, int]] = []
    for spec in specs:
        positions = _valid_positions(blocked, int(spec["outer_h"]), int(spec["outer_w"]))
        if not positions:
            return None
        top, left = random.choice(positions)
        placed.append(_place_region(grid, blocked, spec, top, left))

    expected = _build_expected(grid, placed)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {_MAX_GEN_ATTEMPTS} attempts")
