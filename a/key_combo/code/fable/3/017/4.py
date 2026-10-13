from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
SPRITE_SIZE = 3
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if int(connectivity) == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        pts: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _find_table_bbox(g: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for coords in _components(g == GREY, connectivity=4):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != w or h < 5:
            continue
        frame = g[r0 : r1 + 1, c0 : c1 + 1]
        perimeter = np.zeros_like(frame, dtype=bool)
        perimeter[0, :] = True
        perimeter[-1, :] = True
        perimeter[:, 0] = True
        perimeter[:, -1] = True
        if np.all(frame[perimeter] == GREY) and not np.any(frame[~perimeter] == GREY):
            if int(coords.shape[0]) == int(perimeter.sum()):
                candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, "expected one grey single-pixel square table frame"
    return candidates[0]


def _palette_colors(g: np.ndarray) -> list[int]:
    return sorted(int(c) for c in np.unique(g) if int(c) not in (BLACK, GREY))


def _validate_visible_clues(table: np.ndarray, palette: tuple[int, ...]) -> None:
    symbols = set(map(int, palette))
    for r in range(table.shape[0]):
        vals = [int(v) for v in table[r, :] if int(v) != BLACK]
        assert all(v in symbols for v in vals), "filled table colors must belong to the palette"
        assert len(vals) == len(set(vals)), "a visible row repeats a color"
    for c in range(table.shape[1]):
        vals = [int(v) for v in table[:, c] if int(v) != BLACK]
        assert all(v in symbols for v in vals), "filled table colors must belong to the palette"
        assert len(vals) == len(set(vals)), "a visible column repeats a color"


def _latin_solutions(table: np.ndarray, palette: tuple[int, ...], limit: int = 2) -> list[np.ndarray]:
    n = table.shape[0]
    _validate_visible_clues(table, palette)
    work = table.copy()
    row_used = [set(int(v) for v in work[r, :] if int(v) != BLACK) for r in range(n)]
    col_used = [set(int(v) for v in work[:, c] if int(v) != BLACK) for c in range(n)]
    blanks = {(int(r), int(c)) for r, c in np.argwhere(work == BLACK)}
    solutions: list[np.ndarray] = []

    def search() -> None:
        if len(solutions) >= int(limit):
            return
        if not blanks:
            solutions.append(work.copy())
            return

        best_cell: tuple[int, int] | None = None
        best_options: list[int] | None = None
        for r, c in blanks:
            opts = [color for color in palette if color not in row_used[r] and color not in col_used[c]]
            if best_options is None or len(opts) < len(best_options):
                best_cell = (r, c)
                best_options = opts
                if len(opts) == 0:
                    break
        if best_cell is None or not best_options:
            return

        r, c = best_cell
        blanks.remove((r, c))
        for color in best_options:
            work[r, c] = int(color)
            row_used[r].add(int(color))
            col_used[c].add(int(color))
            search()
            row_used[r].remove(int(color))
            col_used[c].remove(int(color))
            work[r, c] = BLACK
            if len(solutions) >= int(limit):
                break
        blanks.add((r, c))

    search()
    return solutions


def _solve_latin_unique(table: np.ndarray, palette: tuple[int, ...]) -> np.ndarray:
    assert table.ndim == 2 and table.shape[0] == table.shape[1], "expected a square table"
    assert len(palette) == table.shape[0], "palette size must match table size"
    assert bool(np.any(table == BLACK)), "expected at least one blank table cell"
    sols = _latin_solutions(table, palette, limit=2)
    assert len(sols) == 1, "table completion must be unique"
    return sols[0]


def _extract_external_sprites(
    g: np.ndarray,
    table_bbox: tuple[int, int, int, int],
    palette: tuple[int, ...],
) -> dict[int, np.ndarray]:
    r0, c0, r1, c1 = table_bbox
    outside = np.ones(g.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    sprites: dict[int, np.ndarray] = {}

    for color in palette:
        comps = _components((g == int(color)) & outside, connectivity=8)
        assert len(comps) == 1, "each palette color must have one external sprite"
        sr0, sc0, sr1, sc1 = _bbox(comps[0])
        assert (sr1 - sr0 + 1, sc1 - sc0 + 1) == (SPRITE_SIZE, SPRITE_SIZE)
        sprite = np.full((SPRITE_SIZE, SPRITE_SIZE), BLACK, dtype=int)
        for rr, cc in comps[0]:
            sprite[int(rr) - sr0, int(cc) - sc0] = int(color)
        assert sprite[0, :].any() and sprite[-1, :].any()
        assert sprite[:, 0].any() and sprite[:, -1].any()
        sprites[int(color)] = sprite
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a Latin-square color table, then draw only the inferred colors as matching 3x3 legend sprites.

    Input structure:
    - A black background contains one grey single-pixel square frame around an n x n color table.
    - Black table cells are blanks; visible non-black, non-grey cells are palette clues with no row or column repeats.
    - Outside the frame, each palette color appears as exactly one same-color sprite with a 3x3 bounding box.

    Rule:
    - Use the Latin-square row and column uniqueness constraints to find the unique color for every blank table cell.
    - Build a black 3n x 3n output whose 3x3 blocks correspond to the table cells in the same row and column frame.
    - For originally blank cells only, copy the external sprite of the inferred color into the matching output block.

    Key insight:
    - The visible table colors identify the missing Latin-square symbols, while the external legend supplies the sprite
      shape to paint for those missing symbols; filled table cells deliberately leave their output blocks blank.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    table_bbox = _find_table_bbox(g)
    r0, c0, r1, c1 = table_bbox
    table = g[r0 + 1 : r1, c0 + 1 : c1].copy()
    n = int(table.shape[0])
    assert table.shape == (n, n)

    palette = tuple(_palette_colors(g))
    solution = _solve_latin_unique(table, palette)
    sprites = _extract_external_sprites(g, table_bbox, palette)

    out = np.full((SPRITE_SIZE * n, SPRITE_SIZE * n), BLACK, dtype=int)
    for r, c in np.argwhere(table == BLACK):
        rr, cc = int(r), int(c)
        color = int(solution[rr, cc])
        out[SPRITE_SIZE * rr : SPRITE_SIZE * (rr + 1), SPRITE_SIZE * cc : SPRITE_SIZE * (cc + 1)] = sprites[color]
    return out


# ---------------------------- generator ----------------------------

_RNG = np.random.default_rng()
_SPRITE_BANK: list[np.ndarray] | None = None


def _rand_int(low: int, high: int) -> int:
    return int(_RNG.integers(int(low), int(high) + 1))


def _sprite_bank() -> list[np.ndarray]:
    global _SPRITE_BANK
    if _SPRITE_BANK is not None:
        return _SPRITE_BANK

    masks: list[np.ndarray] = []
    for bits in range(1, 1 << (SPRITE_SIZE * SPRITE_SIZE)):
        mask = np.array(
            [(bits >> i) & 1 for i in range(SPRITE_SIZE * SPRITE_SIZE)],
            dtype=bool,
        ).reshape((SPRITE_SIZE, SPRITE_SIZE))
        area = int(mask.sum())
        if area < 4 or area > 8:
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if len(_components(mask, connectivity=8)) != 1:
            continue
        masks.append(mask)
    assert masks
    _SPRITE_BANK = masks
    return masks


def _random_sprite(color: int) -> np.ndarray:
    mask = random.choice(_sprite_bank()).copy()
    sprite = np.full((SPRITE_SIZE, SPRITE_SIZE), BLACK, dtype=int)
    sprite[mask] = int(color)
    return sprite


def _full_latin_square(n: int, colors: tuple[int, ...]) -> np.ndarray:
    rows = list(range(n))
    cols = list(range(n))
    random.shuffle(rows)
    random.shuffle(cols)
    symbols = list(colors)
    random.shuffle(symbols)
    out = np.empty((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            out[r, c] = int(symbols[(rows[r] + cols[c]) % n])
    return out


def _blank_pattern_ok(puzzle: np.ndarray, full: np.ndarray) -> bool:
    blank = puzzle == BLACK
    n = int(puzzle.shape[0])
    count = int(blank.sum())
    if count < max(3, n):
        return False
    hidden_colors = {int(v) for v in full[blank]}
    if len(hidden_colors) < min(3, n):
        return False
    row_counts = blank.sum(axis=1)
    col_counts = blank.sum(axis=0)
    if int(np.count_nonzero(row_counts)) < 2 or int(np.count_nonzero(col_counts)) < 2:
        return False
    if int(row_counts.max()) < 2 and int(col_counts.max()) < 2:
        return False
    if int(row_counts.max()) == n or int(col_counts.max()) == n:
        return False
    return True


def _make_puzzle(full: np.ndarray, palette: tuple[int, ...]) -> np.ndarray | None:
    n = int(full.shape[0])
    for _ in range(180):
        puzzle = full.copy()
        visible_counts = {int(c): int(np.count_nonzero(puzzle == int(c))) for c in palette}
        min_blanks = max(3, n)
        max_blanks = min(n * n - n, 2 * n + 2)
        target = _rand_int(min_blanks, max_blanks)
        cells = [(r, c) for r in range(n) for c in range(n)]
        random.shuffle(cells)

        for r, c in cells:
            if int(np.count_nonzero(puzzle == BLACK)) >= target:
                break
            color = int(puzzle[r, c])
            if visible_counts[color] <= 1:
                continue
            puzzle[r, c] = BLACK
            visible_counts[color] -= 1
            sols = _latin_solutions(puzzle, palette, limit=2)
            if len(sols) != 1 or not np.array_equal(sols[0], full):
                puzzle[r, c] = color
                visible_counts[color] += 1

        if _blank_pattern_ok(puzzle, full):
            return puzzle
    return None


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    g = int(gap)
    return ar1 + g < br0 or br1 + g < ar0 or ac1 + g < bc0 or bc1 + g < ac0


def _place_sprite_rects(
    shape: tuple[int, int],
    occupied: list[tuple[int, int, int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    h, w = map(int, shape)
    candidates = [(r, c) for r in range(1, h - SPRITE_SIZE) for c in range(1, w - SPRITE_SIZE)]
    for _ in range(80):
        random.shuffle(candidates)
        rects = list(occupied)
        chosen: list[tuple[int, int]] = []
        for r, c in candidates:
            box = (int(r), int(c), int(r) + SPRITE_SIZE - 1, int(c) + SPRITE_SIZE - 1)
            if all(_separated(box, prev, gap=1) for prev in rects):
                chosen.append((int(r), int(c)))
                rects.append(box)
                if len(chosen) == int(count):
                    return chosen
    return None


def _render_expected(full: np.ndarray, puzzle: np.ndarray, sprites: dict[int, np.ndarray]) -> np.ndarray:
    n = int(full.shape[0])
    out = np.full((SPRITE_SIZE * n, SPRITE_SIZE * n), BLACK, dtype=int)
    for r, c in np.argwhere(puzzle == BLACK):
        rr, cc = int(r), int(c)
        color = int(full[rr, cc])
        out[SPRITE_SIZE * rr : SPRITE_SIZE * (rr + 1), SPRITE_SIZE * cc : SPRITE_SIZE * (cc + 1)] = sprites[color]
    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4, 4, 5, 5, 6])
    palette = tuple(random.sample(COLOR_POOL, n))
    full = _full_latin_square(n, palette)
    puzzle = _make_puzzle(full, palette)
    if puzzle is None:
        return None

    table_side = n + 2
    min_side = max(17, table_side + 8)
    h = _rand_int(min_side, 30)
    w = _rand_int(min_side, 30)
    tr = _rand_int(1, h - table_side - 1)
    tc = _rand_int(1, w - table_side - 1)
    table_box = (tr, tc, tr + table_side - 1, tc + table_side - 1)
    positions = _place_sprite_rects((h, w), [table_box], n)
    if positions is None:
        return None

    sprites = {int(color): _random_sprite(int(color)) for color in palette}
    grid = np.full((h, w), BLACK, dtype=int)
    grid[tr : tr + table_side, tc] = GREY
    grid[tr : tr + table_side, tc + table_side - 1] = GREY
    grid[tr, tc : tc + table_side] = GREY
    grid[tr + table_side - 1, tc : tc + table_side] = GREY
    grid[tr + 1 : tr + 1 + n, tc + 1 : tc + 1 + n] = puzzle

    color_order = list(palette)
    random.shuffle(color_order)
    for color, (r, c) in zip(color_order, positions):
        sprite = sprites[int(color)]
        mask = sprite != BLACK
        grid[r : r + SPRITE_SIZE, c : c + SPRITE_SIZE][mask] = sprite[mask]

    expected = _render_expected(full, puzzle, sprites)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid Latin legend input")
