from __future__ import annotations

import random

import numpy as np


Square = dict[str, int]

ODD_CENTER_COLOR = 4
INPUT_COLORS = (1, 2, 3)
INPUT_SIZES = (4, 5)
MAX_GENERATION_ATTEMPTS = 300
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in NEIGHBORS4]


def _find_color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _detect_squares(grid: np.ndarray) -> list[Square]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    squares: list[Square] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != 0):
        assert color != ODD_CENTER_COLOR, "yellow is reserved for odd-square centers"
        for comp in _find_color_components(g, color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            side_h = r1 - r0 + 1
            side_w = c1 - c0 + 1
            assert side_h == side_w, "each object must be a square"
            assert len(comp) == side_h * side_w, "each square must be solid"
            squares.append(
                {
                    "color": color,
                    "r0": r0,
                    "c0": c0,
                    "side": side_h,
                }
            )

    assert squares, "input must contain at least one foreground square"
    return squares


def _mark_odd_centers(shape: tuple[int, int], squares: list[Square]) -> tuple[np.ndarray, np.ndarray]:
    marked = np.zeros(shape, dtype=int)
    kinds = np.zeros(shape, dtype=np.int8)
    # kinds: 0 bg, 1 even square, 2 odd non-center, 3 odd center
    odd_count = 0
    even_count = 0

    for sq in squares:
        color = int(sq["color"])
        r0 = int(sq["r0"])
        c0 = int(sq["c0"])
        side = int(sq["side"])
        marked[r0 : r0 + side, c0 : c0 + side] = color

        if side % 2 == 0:
            kinds[r0 : r0 + side, c0 : c0 + side] = 1
            even_count += 1
            continue

        kinds[r0 : r0 + side, c0 : c0 + side] = 2
        center_r = r0 + side // 2
        center_c = c0 + side // 2
        marked[center_r, center_c] = ODD_CENTER_COLOR
        kinds[center_r, center_c] = 3
        odd_count += 1

    assert odd_count >= 1, "input must contain at least one odd square"
    assert even_count >= 1, "input must contain at least one even square"
    return marked, kinds


def _expand_from_marked(grid: np.ndarray, marked: np.ndarray, kinds: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    out = np.zeros((h * h, w * w), dtype=int)
    yellow_mask = marked == ODD_CENTER_COLOR

    for r in range(h):
        for c in range(w):
            kind = int(kinds[r, c])
            if kind == 0:
                continue

            block = np.zeros((h, w), dtype=int)
            if kind == 1:
                block.fill(int(grid[r, c]))
            elif kind == 2:
                color = int(grid[r, c])
                block[marked == color] = color
            else:
                block[yellow_mask] = ODD_CENTER_COLOR

            out[r * h : (r + 1) * h, c * w : (c + 1) * w] = block
    return out


def _render_expected(grid: np.ndarray, squares: list[Square]) -> np.ndarray:
    marked, kinds = _mark_odd_centers(grid.shape, squares)
    return _expand_from_marked(grid, marked, kinds)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each square pixel with one of three masks determined by square parity and center marking.

    Input structure:
    - The input is a small black-background grid containing 2-3 separated solid colored squares.
    - Foreground squares are axis-aligned, use non-yellow colors, and may have side length 1, 2, or 3.
    - At least one square has odd side length and at least one square has even side length.

    Rule:
    - Identify every solid square and determine whether its side length is odd or even.
    - Recolor the center pixel of each odd-sized square to yellow; even-sized squares are not marked.
    - Expand every input cell into a full input-sized block: even-square pixels become solid same-color blocks, odd non-center pixels use the marked grid's same-color mask, yellow center pixels use the yellow-center mask, and black background stays black.

    Key insight:
    - The same marked grid acts both as the object map and as the mask template, but odd, even, and center pixels read that template in different ways.
    """
    g = np.asarray(grid, dtype=int)
    squares = _detect_squares(g)
    marked, kinds = _mark_odd_centers(g.shape, squares)
    return _expand_from_marked(g, marked, kinds)


def _candidate_configs(n: int) -> list[tuple[tuple[int, ...], int]]:
    if n == 4:
        return [
            ((2, 1), 3),
            ((2, 1, 1), 4),
            ((2, 2, 1), 2),
        ]
    return [
        ((2, 1), 2),
        ((2, 3), 2),
        ((2, 1, 1), 2),
        ((2, 2, 1), 1),
        ((2, 1, 3), 3),
    ]


def _pick_side_list(n: int) -> list[int]:
    configs = _candidate_configs(n)
    total = sum(weight for _, weight in configs)
    pick = random.randint(1, total)
    for sides, weight in configs:
        pick -= weight
        if pick <= 0:
            chosen = list(sides)
            random.shuffle(chosen)
            return chosen
    raise AssertionError("failed to sample side list")


def _placement_candidates(n: int, side: int, occupied: np.ndarray) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r0 in range(n - side + 1):
        for c0 in range(n - side + 1):
            ok = True
            for r in range(r0, r0 + side):
                for c in range(c0, c0 + side):
                    if occupied[r, c]:
                        ok = False
                        break
                    for nr, nc in _neighbors4(r, c):
                        if 0 <= nr < n and 0 <= nc < n and occupied[nr, nc]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                candidates.append((r0, c0))
    random.shuffle(candidates)
    return candidates


def _place_squares(n: int, sides: list[int], colors: list[int]) -> list[Square] | None:
    order = sorted(range(len(sides)), key=lambda idx: (-sides[idx], random.random()))
    occupied = np.zeros((n, n), dtype=bool)
    placed: list[Square | None] = [None] * len(sides)

    def dfs(i: int) -> bool:
        if i == len(order):
            return True
        idx = order[i]
        side = sides[idx]
        for r0, c0 in _placement_candidates(n, side, occupied):
            occupied[r0 : r0 + side, c0 : c0 + side] = True
            placed[idx] = {"color": colors[idx], "r0": r0, "c0": c0, "side": side}
            if dfs(i + 1):
                return True
            occupied[r0 : r0 + side, c0 : c0 + side] = False
            placed[idx] = None
        return False

    if not dfs(0):
        return None
    return [sq for sq in placed if sq is not None]


def _build_input(n: int, squares: list[Square]) -> np.ndarray:
    grid = np.zeros((n, n), dtype=int)
    for sq in squares:
        r0 = int(sq["r0"])
        c0 = int(sq["c0"])
        side = int(sq["side"])
        grid[r0 : r0 + side, c0 : c0 + side] = int(sq["color"])
    return grid


def _gen_attempt() -> np.ndarray | None:
    n = random.choices(INPUT_SIZES, weights=(2, 3), k=1)[0]
    sides = _pick_side_list(n)
    colors = random.sample(list(INPUT_COLORS), len(sides))
    squares = _place_squares(n, sides, colors)
    if squares is None:
        return None

    grid = _build_input(n, squares)
    expected = _render_expected(grid, squares)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
