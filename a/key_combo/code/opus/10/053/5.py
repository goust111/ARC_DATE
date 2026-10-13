from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
RED = 2
GREEN = 3
FG_COLORS = (1, 4, 5, 6, 7, 8, 9)
GEN_RETRIES = 400
_GEN_CALL_COUNT = 0


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).flat))
    return int(counts.most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.asarray(grid).flat))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    k = int(factor)
    if k <= 0:
        raise ValueError("scale factor must be positive")
    return np.kron(np.asarray(sprite, dtype=int), np.ones((k, k), dtype=int))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(component: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in component)


def _count_enclosed_black_regions(grid: np.ndarray) -> int:
    black_components = _components(np.asarray(grid, dtype=int) == BG)
    return sum(not _touches_border(component, grid.shape) for component in black_components)


def _paint_exterior_black(mask: np.ndarray) -> np.ndarray:
    black = np.asarray(mask, dtype=bool)
    h, w = black.shape
    exterior = np.zeros_like(black, dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if black[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if black[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and black[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                stack.append((nr, nc))
    return exterior


def main(grid: np.ndarray) -> np.ndarray:
    """Scale the grid by its number of enclosed background regions, then recolor exterior and interior background.

    Input structure:
    - The grid is a small black-background canvas with exactly one non-black foreground color.
    - The foreground is a single 4-connected orthogonal line/wall object.
    - Black cells split into one exterior region and 2 to 4 enclosed regions.

    Rule:
    - Count the number K of black 4-connected components that do not touch the canvas border.
    - Replace every input cell by a K x K block of the same color.
    - In the scaled grid, keep the foreground color unchanged, recolor border-connected black cells green, and recolor the remaining black cells red.

    Key insight:
    - The scale factor is a topological count: it is the number of enclosed black regions created by the foreground wall, not a fixed constant and not a color-count cue.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert 5 <= g.shape[0] <= 7 and 5 <= g.shape[1] <= 7

    bg = get_bg(g)
    assert bg == BG
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1
    fg = int(fg_colors[0])

    fg_components = _components(g == fg)
    assert len(fg_components) == 1

    k = _count_enclosed_black_regions(g)
    assert 2 <= k <= 4

    scaled = scale_sprite(g, k)
    black_mask = scaled == BG
    exterior = _paint_exterior_black(black_mask)
    assert exterior.any()

    out = scaled.copy()
    out[exterior] = GREEN
    out[black_mask & ~exterior] = RED
    return out


def _lattice(cell_rows: int, cell_cols: int) -> np.ndarray:
    grid = np.ones((2 * cell_rows + 1, 2 * cell_cols + 1), dtype=int)
    grid[1::2, 1::2] = BG
    return grid


def _full_2x2_opening_options() -> list[list[tuple[int, int]]]:
    return [
        [(0, 1), (1, 0)],
        [(0, 3), (1, 4)],
        [(4, 1), (3, 0)],
        [(4, 3), (3, 4)],
    ]


def _full_2x2_merge_options() -> list[tuple[int, int]]:
    return [(1, 2), (2, 1), (2, 3), (3, 2)]


def _embed_with_margin(base: np.ndarray, rng: random.Random) -> np.ndarray:
    bh, bw = base.shape
    total_pad_h = rng.randint(2, 7 - bh)
    total_pad_w = rng.randint(2, 7 - bw)
    top = rng.randint(1, total_pad_h - 1)
    bottom = total_pad_h - top
    left = rng.randint(1, total_pad_w - 1)
    right = total_pad_w - left
    out = np.zeros((bh + top + bottom, bw + left + right), dtype=int)
    out[top : top + bh, left : left + bw] = base
    return out


def _rotate_and_flip(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.array(grid, copy=True)
    out = np.rot90(out, rng.randrange(4))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = np.flipud(out)
    return np.asarray(out, dtype=int)


def _build_binary_pattern(k: int, rng: random.Random) -> np.ndarray:
    if k == 4:
        base = _lattice(2, 2)
    elif k == 3:
        base = _lattice(2, 2)
        if rng.random() < 0.5:
            chamber = rng.randrange(4)
            opening = rng.choice(_full_2x2_opening_options()[chamber])
            base[opening] = BG
        else:
            merge = rng.choice(_full_2x2_merge_options())
            base[merge] = BG
    elif k == 2:
        branch = rng.random()
        if branch < 0.34:
            base = _lattice(1, 2)
        elif branch < 0.68:
            base = _lattice(2, 1)
        else:
            base = _lattice(2, 2)
            if rng.random() < 0.5:
                opened = rng.sample(range(4), 2)
                for chamber in opened:
                    opening = rng.choice(_full_2x2_opening_options()[chamber])
                    base[opening] = BG
            else:
                merge = rng.choice(_full_2x2_merge_options())
                base[merge] = BG
                chamber = rng.randrange(4)
                opening = rng.choice(_full_2x2_opening_options()[chamber])
                base[opening] = BG
    else:
        raise ValueError(f"unsupported hole count {k}")

    embedded = _embed_with_margin(base, rng)
    return _rotate_and_flip(embedded, rng)


def _validate_input(grid: np.ndarray, expected_k: int) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert 5 <= grid.shape[0] <= 7 and 5 <= grid.shape[1] <= 7
    assert get_bg(grid) == BG

    fg_colors = colors_present(grid, exclude={BG})
    assert len(fg_colors) == 1
    fg = fg_colors[0]
    assert fg not in (RED, GREEN)

    fg_components = _components(grid == fg)
    assert len(fg_components) == 1

    k = _count_enclosed_black_regions(grid)
    assert k == expected_k

    black_components = _components(grid == BG)
    assert any(_touches_border(component, grid.shape) for component in black_components)


def _validate_semantics(inp: np.ndarray, out: np.ndarray, expected_k: int) -> None:
    h, w = inp.shape
    assert out.shape == (h * expected_k, w * expected_k)
    fg = colors_present(inp, exclude={BG})[0]

    for r in range(h):
        for c in range(w):
            block = out[r * expected_k : (r + 1) * expected_k, c * expected_k : (c + 1) * expected_k]
            if inp[r, c] == fg:
                assert np.all(block == fg)

    black_components = _components(inp == BG)
    red_components = 0
    for component in black_components:
        fill = GREEN if _touches_border(component, inp.shape) else RED
        if fill == RED:
            red_components += 1
        for r, c in component:
            block = out[r * expected_k : (r + 1) * expected_k, c * expected_k : (c + 1) * expected_k]
            assert np.all(block == fill)

    assert red_components == expected_k
    assert np.all(out[0, :] != BG)
    assert np.all(out[-1, :] != BG)
    assert np.all(out[:, 0] != BG)
    assert np.all(out[:, -1] != BG)


def _planned_hole_count() -> int:
    return (2, 3, 4)[_GEN_CALL_COUNT % 3]


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    planned_k = _planned_hole_count()
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        binary = _build_binary_pattern(planned_k, rng)
        fg = rng.choice(FG_COLORS)
        grid = np.where(binary == 1, fg, BG).astype(int)
        try:
            _validate_input(grid, planned_k)
            out = main(grid)
            _validate_semantics(grid, out, planned_k)
        except AssertionError:
            continue
        _GEN_CALL_COUNT += 1
        return grid
    raise RuntimeError("failed to generate a valid input")
