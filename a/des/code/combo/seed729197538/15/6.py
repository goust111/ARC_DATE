from __future__ import annotations

import random

import numpy as np


BLACK = 0
FRAME = 5
NON_FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SYMMETRIES = ("horizontal", "vertical", "rotate180")
GEN_ATTEMPTS = 400


def _detect_symmetry(grid: np.ndarray) -> str:
    tl = int(grid[0, 0])
    tr = int(grid[0, -1])
    bl = int(grid[-1, 0])
    br = int(grid[-1, -1])

    if tl == tr and bl == br and tl != bl:
        return "horizontal"
    if tl == bl and tr == br and tl != tr:
        return "vertical"
    if tl == br and tr == bl and tl != tr:
        return "rotate180"
    raise AssertionError("unrecognized corner code")


def _apply_symmetry(arr: np.ndarray, symmetry: str) -> np.ndarray:
    if symmetry == "horizontal":
        return np.flipud(arr)
    if symmetry == "vertical":
        return np.fliplr(arr)
    if symmetry == "rotate180":
        return np.rot90(arr, 2)
    raise AssertionError(f"unknown symmetry: {symmetry}")


def _repair_inner(inner: np.ndarray, symmetry: str) -> np.ndarray:
    mirrored = _apply_symmetry(inner, symmetry)
    conflict = (inner != BLACK) & (mirrored != BLACK) & (inner != mirrored)
    assert not bool(conflict.any()), "paired pixels may not disagree on non-black colors"

    out = inner.copy()
    fill = (out == BLACK) & (mirrored != BLACK)
    out[fill] = mirrored[fill]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the interior pattern into the symmetry indicated by the corner markers.

    The outer border is preserved. Inside the border, every position is merged with
    its indicated symmetric counterpart, preferring the non-black value.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 5 and w >= 5

    edge = np.concatenate(
        [
            g[0, 1:-1],
            g[-1, 1:-1],
            g[1:-1, 0],
            g[1:-1, -1],
        ]
    )
    frame_colors = {int(v) for v in np.unique(edge)}
    assert len(frame_colors) == 1
    assert next(iter(frame_colors)) != BLACK

    symmetry = _detect_symmetry(g)
    out = g.copy()
    out[1:-1, 1:-1] = _repair_inner(g[1:-1, 1:-1], symmetry)
    return out


def _partner(r: int, c: int, h: int, w: int, symmetry: str) -> tuple[int, int]:
    if symmetry == "horizontal":
        return h - 1 - r, c
    if symmetry == "vertical":
        return r, w - 1 - c
    if symmetry == "rotate180":
        return h - 1 - r, w - 1 - c
    raise AssertionError(symmetry)


def _canonical_mask(h: int, w: int, symmetry: str) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if symmetry == "horizontal":
        mask[: (h + 1) // 2, :] = True
        return mask
    if symmetry == "vertical":
        mask[:, : (w + 1) // 2] = True
        return mask

    for r in range(h):
        for c in range(w):
            pr, pc = _partner(r, c, h, w, symmetry)
            if (r, c) <= (pr, pc):
                mask[r, c] = True
    return mask


def _paint(seed: np.ndarray, allowed: np.ndarray, coords: list[tuple[int, int]], color: int) -> int:
    added = 0
    h, w = seed.shape
    for r, c in coords:
        if 0 <= r < h and 0 <= c < w and allowed[r, c]:
            if seed[r, c] == BLACK:
                added += 1
            seed[r, c] = color
    return added


def _random_shape(h: int, w: int, points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r, c = random.choice(points)
    kind = random.choice(("rect", "hline", "vline", "plus", "ell"))

    if kind == "rect":
        hh = random.randint(1, max(1, min(4, h)))
        ww = random.randint(1, max(1, min(4, w)))
        top = r - random.randint(0, hh - 1)
        left = c - random.randint(0, ww - 1)
        return [(rr, cc) for rr in range(top, top + hh) for cc in range(left, left + ww)]

    if kind == "hline":
        length = random.randint(2, max(2, min(6, w)))
        left = c - random.randint(0, length - 1)
        return [(r, cc) for cc in range(left, left + length)]

    if kind == "vline":
        length = random.randint(2, max(2, min(6, h)))
        top = r - random.randint(0, length - 1)
        return [(rr, c) for rr in range(top, top + length)]

    if kind == "plus":
        arm = random.randint(1, 2)
        coords = [(r, c)]
        coords.extend((r + d, c) for d in range(-arm, arm + 1))
        coords.extend((r, c + d) for d in range(-arm, arm + 1))
        return coords

    vlen = random.randint(2, max(2, min(5, h)))
    hlen = random.randint(2, max(2, min(5, w)))
    down = random.choice((True, False))
    right = random.choice((True, False))
    vrange = range(r, r + vlen) if down else range(r - vlen + 1, r + 1)
    hrange = range(c, c + hlen) if right else range(c - hlen + 1, c + 1)
    coords = [(rr, c) for rr in vrange]
    coords.extend((r, cc) for cc in hrange)
    return coords


def _symmetrize(seed: np.ndarray, symmetry: str) -> np.ndarray | None:
    mirrored = _apply_symmetry(seed, symmetry)
    conflict = (seed != BLACK) & (mirrored != BLACK) & (seed != mirrored)
    if bool(conflict.any()):
        return None

    out = seed.copy()
    fill = (out == BLACK) & (mirrored != BLACK)
    out[fill] = mirrored[fill]
    return out


def _colored_pair_orbits(arr: np.ndarray, symmetry: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = arr.shape
    out: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            pr, pc = _partner(r, c, h, w, symmetry)
            if (r, c) >= (pr, pc):
                continue
            if int(arr[r, c]) != BLACK and int(arr[pr, pc]) == int(arr[r, c]):
                out.append(((r, c), (pr, pc)))
    return out


def _other_symmetries(symmetry: str) -> list[str]:
    return [name for name in SYMMETRIES if name != symmetry]


def _corner_pattern(symmetry: str, a: int, b: int) -> tuple[int, int, int, int]:
    if symmetry == "horizontal":
        return a, a, b, b
    if symmetry == "vertical":
        return a, b, a, b
    if symmetry == "rotate180":
        return a, b, b, a
    raise AssertionError(symmetry)


def _build_pattern(h: int, w: int, symmetry: str, palette: list[int]) -> np.ndarray | None:
    allowed = _canonical_mask(h, w, symmetry)
    points = [tuple(map(int, pt)) for pt in np.argwhere(allowed)]
    seed = np.zeros((h, w), dtype=int)

    for color in palette:
        shape_tries = 0
        while shape_tries < 20:
            if _paint(seed, allowed, _random_shape(h, w, points), color) > 0:
                break
            shape_tries += 1
        if shape_tries == 20:
            return None

    extra_shapes = random.randint(2, 5)
    for _ in range(extra_shapes):
        _paint(seed, allowed, _random_shape(h, w, points), random.choice(palette))

    out = _symmetrize(seed, symmetry)
    if out is None:
        return None

    colored = int(np.count_nonzero(out))
    area = h * w
    if colored < 12 or colored > area // 2:
        return None

    for other in _other_symmetries(symmetry):
        if np.array_equal(out, _apply_symmetry(out, other)):
            return None

    if len({int(v) for v in np.unique(out) if int(v) != BLACK}) < 2:
        return None

    return out


def _embed(inner: np.ndarray, symmetry: str, corner_a: int, corner_b: int) -> np.ndarray:
    h, w = inner.shape
    grid = np.full((h + 2, w + 2), BLACK, dtype=int)
    grid[0, :] = FRAME
    grid[-1, :] = FRAME
    grid[:, 0] = FRAME
    grid[:, -1] = FRAME
    tl, tr, bl, br = _corner_pattern(symmetry, corner_a, corner_b)
    grid[0, 0] = tl
    grid[0, -1] = tr
    grid[-1, 0] = bl
    grid[-1, -1] = br
    grid[1:-1, 1:-1] = inner
    return grid


def _gen_attempt() -> np.ndarray | None:
    symmetry = random.choice(SYMMETRIES)
    inner_h = random.randint(7, 14)
    inner_w = random.randint(7, 14)

    palette_size = random.randint(2, 4)
    palette = random.sample(NON_FRAME_COLORS, palette_size)
    output_inner = _build_pattern(inner_h, inner_w, symmetry, palette)
    if output_inner is None:
        return None

    pair_orbits = _colored_pair_orbits(output_inner, symmetry)
    if len(pair_orbits) < 6:
        return None

    input_inner = output_inner.copy()
    for a, b in random.sample(pair_orbits, random.randint(3, 6)):
        r, c = random.choice((a, b))
        input_inner[r, c] = BLACK

    remaining = [c for c in NON_FRAME_COLORS if c not in palette]
    corner_a, corner_b = random.sample(remaining, 2)
    inp = _embed(input_inner, symmetry, corner_a, corner_b)
    expected = _embed(output_inner, symmetry, corner_a, corner_b)

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
