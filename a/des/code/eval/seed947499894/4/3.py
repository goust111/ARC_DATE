from __future__ import annotations

from collections import Counter
import random

import numpy as np


RED = 2
MAROON = 9
FREE_COLORS = [0, 1, 3, 4, 5, 6, 7, 8]
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _corner_background(g: np.ndarray) -> int:
    corners = [int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]
    return int(Counter(corners).most_common(1)[0][0])


def _find_axis_column(g: np.ndarray, bg: int) -> int:
    h, w = g.shape
    candidates = []
    for c in range(w):
        if c != w - 1 - c:
            continue
        if int(g[0, c]) == bg:
            continue
        if bool(np.all(g[:, c] == g[0, c])):
            candidates.append(c)
    assert len(candidates) == 1
    return int(candidates[0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _paint_scaled_mirror(
    out: np.ndarray,
    coords: list[tuple[int, int]],
    color: int,
    scale: int,
    axis_col: int,
) -> None:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    box_w = c1 - c0 + 1
    anchor_c = 2 * axis_col - c1

    for r, c in coords:
        rr0 = r0 + (r - r0) * scale
        cc0 = anchor_c + (box_w - 1 - (c - c0)) * scale
        assert 0 <= rr0 and rr0 + scale <= out.shape[0]
        assert 0 <= cc0 and cc0 + scale <= out.shape[1]
        out[rr0 : rr0 + scale, cc0 : cc0 + scale] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each left-half shape across the central maroon axis and scale the copy
    by the number of red seed pixels embedded in that shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _corner_background(g)
    axis_col = _find_axis_column(g, bg)
    left = g[:, :axis_col]
    mask = left != bg
    comps = _components(mask)
    assert comps

    out = g.copy()
    for comp in comps:
        colors = {int(g[r, c]) for r, c in comp}
        body_colors = colors - {RED}
        assert len(body_colors) == 1
        color = int(next(iter(body_colors)))
        seed_count = sum(int(g[r, c]) == RED for r, c in comp)
        assert seed_count in {1, 2, 3}
        _paint_scaled_mirror(out, comp, color, seed_count, axis_col)

    return out


def _normalize(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0, c0 = min(rows), min(cols)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _transform(cells: tuple[tuple[int, int], ...], rotation: int, flip: bool) -> tuple[tuple[int, int], ...]:
    pts = list(cells)
    if flip:
        pts = [(r, -c) for r, c in pts]
    for _ in range(rotation % 4):
        pts = [(c, -r) for r, c in pts]
    return _normalize(pts)


def _canonical(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return min(_transform(cells, rot, flip) for rot in range(4) for flip in (False, True))


def _shape_dims(cells: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _c in cells) + 1, max(c for _r, c in cells) + 1


def _random_polyomino(min_area: int, max_area: int = 6) -> tuple[tuple[int, int], ...]:
    for _ in range(300):
        area = random.randint(min_area, max_area)
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            cells.add(random.choice(tuple(frontier)))

        shape = _normalize(cells)
        shape = _transform(shape, random.randrange(4), bool(random.getrandbits(1)))
        h, w = _shape_dims(shape)
        if 2 <= h <= 3 and 2 <= w <= 3:
            return shape
    raise RuntimeError("failed to sample a polyomino")


def _distribute_extra(extra: int, slots: int) -> list[int]:
    gaps = [1] * slots
    for _ in range(extra):
        gaps[random.randrange(slots)] += 1
    return gaps


def _build_specs():
    n_shapes = 3 if random.random() < 0.7 else 4
    scales = [1, 2, 3]
    if n_shapes == 4:
        scales.append(random.choice([1, 2, 3]))
    random.shuffle(scales)

    bg = random.choice(FREE_COLORS)
    shape_palette = [c for c in FREE_COLORS if c != bg]
    random.shuffle(shape_palette)

    specs = []
    seen_shapes = set()
    for idx, scale in enumerate(scales):
        min_area = max(3, scale + 1)
        for _ in range(300):
            shape = _random_polyomino(min_area=min_area)
            canon = _canonical(shape)
            if canon in seen_shapes:
                continue
            seen_shapes.add(canon)
            break
        else:
            raise RuntimeError("failed to sample distinct shapes")

        seeds = tuple(sorted(random.sample(list(shape), scale)))
        h, w = _shape_dims(shape)
        specs.append(
            {
                "shape": shape,
                "seeds": seeds,
                "scale": int(scale),
                "color": int(shape_palette[idx]),
                "h": int(h),
                "w": int(w),
            }
        )

    max_scaled_w = max(spec["w"] * spec["scale"] for spec in specs)
    left_w_lo = max(10, max_scaled_w + 4)
    left_w_hi = max(left_w_lo, max_scaled_w + 8)
    left_w = random.randint(left_w_lo, left_w_hi)

    order = list(range(n_shapes))
    random.shuffle(order)
    gaps = _distribute_extra(random.randint(0, n_shapes + 4), n_shapes + 1)
    row = gaps[0]
    for slot, spec_idx in enumerate(order):
        spec = specs[spec_idx]
        spec["r0"] = int(row)
        row += spec["h"] * spec["scale"] + gaps[slot + 1]
    height = int(row)

    for spec in specs:
        lo = spec["w"] * (spec["scale"] - 1)
        hi = left_w - spec["w"]
        if hi >= 1:
            lo = max(lo, 1)
        spec["c0"] = int(random.randint(lo, hi))

    return bg, height, left_w, specs


def _render_input(bg: int, height: int, left_w: int, specs) -> np.ndarray:
    grid = np.full((height, 2 * left_w + 1), bg, dtype=int)
    grid[:, left_w] = MAROON
    for spec in specs:
        r0 = spec["r0"]
        c0 = spec["c0"]
        color = spec["color"]
        for dr, dc in spec["shape"]:
            grid[r0 + dr, c0 + dc] = int(color)
        for dr, dc in spec["seeds"]:
            grid[r0 + dr, c0 + dc] = RED
    return grid


def _render_expected(inp: np.ndarray, left_w: int, specs) -> np.ndarray:
    out = inp.copy()
    axis_col = left_w
    for spec in specs:
        coords = [(spec["r0"] + dr, spec["c0"] + dc) for dr, dc in spec["shape"]]
        _paint_scaled_mirror(out, coords, spec["color"], spec["scale"], axis_col)
    return out


def _valid_instance(inp: np.ndarray, expected: np.ndarray) -> bool:
    left_w = inp.shape[1] // 2
    if not bool(np.all(inp[:, left_w] == MAROON)):
        return False
    if not bool(np.all(inp[:, left_w + 1 :] == _corner_background(inp))):
        return False
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        bg, height, left_w, specs = _build_specs()
        inp = _render_input(bg, height, left_w, specs)
        expected = _render_expected(inp, left_w, specs)
        if _valid_instance(inp, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
