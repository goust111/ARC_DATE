from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


class Color:
    BLACK = 0
    NOT_BLACK = tuple(range(1, 10))


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    H, W = mask.shape
    components: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _ordered_palette(border: np.ndarray) -> list[int]:
    ordered = []
    seen = set()
    for value in border.flat:
        color = int(value)
        if color not in seen:
            seen.add(color)
            ordered.append(color)
    return ordered


def _enumerate_solutions(
    border: np.ndarray,
    givens: np.ndarray,
    *,
    use_ban: bool,
    limit: int,
) -> list[np.ndarray]:
    n = int(border.shape[0])
    assert border.shape == (n, n)
    assert givens.shape == (n, n)

    palette = _ordered_palette(border)
    assert len(palette) == n
    color_to_idx = {color: idx for idx, color in enumerate(palette)}
    palette_arr = np.array(palette, dtype=int)

    border_idx = np.vectorize(lambda v: color_to_idx[int(v)])(border)
    given_idx = np.full((n, n), -1, dtype=int)
    for r in range(n):
        for c in range(n):
            value = int(givens[r, c])
            if value >= 0:
                assert value in color_to_idx
                given_idx[r, c] = color_to_idx[value]

    perms = list(itertools.permutations(range(n)))
    row_options: list[list[tuple[int, ...]]] = []
    for r in range(n):
        options = []
        for perm in perms:
            ok = True
            for c, sym in enumerate(perm):
                if given_idx[r, c] >= 0 and sym != int(given_idx[r, c]):
                    ok = False
                    break
                if use_ban and sym == int(border_idx[r, c]):
                    ok = False
                    break
            if ok:
                options.append(perm)
        if not options:
            return []
        row_options.append(options)

    assigned: list[tuple[int, ...] | None] = [None] * n
    used_cols = [set() for _ in range(n)]
    solutions_idx: list[np.ndarray] = []

    def _compatible(row: int) -> list[tuple[int, ...]]:
        compatible = []
        for perm in row_options[row]:
            if all(perm[c] not in used_cols[c] for c in range(n)):
                compatible.append(perm)
        return compatible

    def _future_possible() -> bool:
        for row in range(n):
            if assigned[row] is None and not _compatible(row):
                return False
        return True

    def search() -> None:
        if len(solutions_idx) >= limit:
            return
        best_row = -1
        best_options: list[tuple[int, ...]] | None = None
        for row in range(n):
            if assigned[row] is not None:
                continue
            compatible = _compatible(row)
            if not compatible:
                return
            if best_options is None or len(compatible) < len(best_options):
                best_row = row
                best_options = compatible
                if len(best_options) == 1:
                    break

        if best_options is None:
            sol_idx = np.empty((n, n), dtype=int)
            for row, perm in enumerate(assigned):
                assert perm is not None
                sol_idx[row] = np.array(perm, dtype=int)
            solutions_idx.append(sol_idx)
            return

        for perm in best_options:
            assigned[best_row] = perm
            for c, sym in enumerate(perm):
                used_cols[c].add(sym)
            if _future_possible():
                search()
            for c, sym in enumerate(perm):
                used_cols[c].remove(sym)
            assigned[best_row] = None
            if len(solutions_idx) >= limit:
                return

    search()
    return [palette_arr[sol_idx] for sol_idx in solutions_idx]


def _parse_frames(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[list[tuple[int, int, int, int]]]]:
    mask = grid != Color.BLACK
    components = _connected_components(mask)
    assert components, "expected a grid of non-black square frames"

    frames = []
    side = None
    for coords in components:
        r0, c0, r1, c1 = _bbox_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h == w and h >= 4, "each component must be a hollow square frame"
        if side is None:
            side = h
        assert side == h, "all frames must have the same side length"

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border_vals = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
        border_color = int(border_vals[0])
        assert border_color != Color.BLACK
        assert np.all(border_vals == border_color), "frame border must be monochrome"

        interior = patch[1:-1, 1:-1]
        inside_colors = {int(v) for v in np.unique(interior) if int(v) != Color.BLACK}
        assert len(inside_colors) <= 1, "frame interior must be blank or solid-filled"
        if inside_colors:
            fill_color = inside_colors.pop()
            assert np.all(interior == fill_color), "filled interior must be solid"
            assert fill_color != border_color, "clue color must differ from its border color"
            expected_area = h * h
        else:
            assert np.all(interior == Color.BLACK), "blank interior must be black"
            fill_color = -1
            expected_area = 4 * h - 4
        assert int(coords.shape[0]) == expected_area

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "border": border_color,
                "given": fill_color,
            }
        )

    row_starts = sorted({frame["bbox"][0] for frame in frames})
    col_starts = sorted({frame["bbox"][1] for frame in frames})
    n = len(row_starts)
    assert n == len(col_starts) and len(frames) == n * n, "frames must form an n x n array"

    row_index = {r0: idx for idx, r0 in enumerate(row_starts)}
    col_index = {c0: idx for idx, c0 in enumerate(col_starts)}
    border = np.full((n, n), -1, dtype=int)
    givens = np.full((n, n), -1, dtype=int)
    bboxes: list[list[tuple[int, int, int, int]]] = [[(-1, -1, -1, -1) for _ in range(n)] for _ in range(n)]

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        rr = row_index[r0]
        cc = col_index[c0]
        assert border[rr, cc] < 0, "frame layout must not overlap in object coordinates"
        border[rr, cc] = int(frame["border"])
        givens[rr, cc] = int(frame["given"])
        bboxes[rr][cc] = (r0, c0, r1, c1)

    assert np.all(border >= 0)
    palette = set(map(int, border.flat))
    assert len(palette) == n, "border colors must expose exactly the n-color palette"
    given_colors = {int(v) for v in givens.flat if int(v) >= 0}
    assert given_colors.issubset(palette)
    return border, givens, bboxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the interiors of a hollow-square array so the interior colors form a Latin square and avoid each frame's border color.

    Input structure:
    - The background is black.
    - The foreground is an n x n array of equal hollow square frames separated by black gaps.
    - Every frame has a monochrome non-black border; some interiors are solid-filled with a palette color and the rest are black.

    Rule:
    - Keep all frame borders and every given interior unchanged.
    - Treat the set of border colors as the hidden n-color palette.
    - Fill each blank interior so that every object-row and every object-column uses all palette colors exactly once, and each interior color differs from that frame's own border color.

    Key insight:
    - The row/column uniqueness lives on the frame array, not on pixel rows, and each border color is both a palette member and a local "not this color" constraint for that frame.
    """
    g = np.asarray(grid, dtype=int)
    border, givens, bboxes = _parse_frames(g)
    solutions = _enumerate_solutions(border, givens, use_ban=True, limit=2)
    assert len(solutions) == 1, "puzzle must have a unique completion"
    solved = solutions[0]

    out = g.copy()
    n = int(border.shape[0])
    for r in range(n):
        for c in range(n):
            r0, c0, r1, c1 = bboxes[r][c]
            out[r0 + 1 : r1, c0 + 1 : c1] = int(solved[r, c])
    return out


def _latin_indices(n: int) -> np.ndarray:
    base = np.fromfunction(lambda r, c: (r + c) % n, (n, n), dtype=int)
    rows = random.sample(range(n), n)
    cols = random.sample(range(n), n)
    syms = random.sample(range(n), n)
    square = base[rows][:, cols]
    square = np.vectorize(lambda v: syms[int(v)])(square)
    if random.random() < 0.5:
        square = square.T
    return square.astype(int)


def _derangements(n: int) -> list[tuple[int, ...]]:
    return [perm for perm in itertools.permutations(range(n)) if all(perm[i] != i for i in range(n))]


def _make_border(solution: np.ndarray) -> np.ndarray:
    n = int(solution.shape[0])
    options = _derangements(n)
    border = np.empty_like(solution)
    for r in range(n):
        perm = random.choice(options)
        border[r] = solution[r, list(perm)]
    return border


def _givens_from_clues(solution: np.ndarray, clues: set[int]) -> np.ndarray:
    n = int(solution.shape[0])
    givens = np.full((n, n), -1, dtype=int)
    for idx in clues:
        r, c = divmod(idx, n)
        givens[r, c] = int(solution[r, c])
    return givens


def _pick_clues(solution: np.ndarray, border: np.ndarray) -> set[int] | None:
    n = int(solution.shape[0])
    blank = np.full((n, n), -1, dtype=int)
    if len(_enumerate_solutions(border, blank, use_ban=True, limit=2)) <= 1:
        return None

    clues: set[int] = set()
    for _ in range(n * n):
        givens = _givens_from_clues(solution, clues)
        sampled = _enumerate_solutions(border, givens, use_ban=True, limit=12)
        if len(sampled) == 1:
            break

        best_score = -1
        best_cells: list[int] = []
        for idx in range(n * n):
            if idx in clues:
                continue
            r, c = divmod(idx, n)
            target = int(solution[r, c])
            mismatch = sum(int(sol[r, c]) != target for sol in sampled)
            if mismatch == 0:
                continue
            distinct = len({int(sol[r, c]) for sol in sampled})
            score = mismatch * 10 + distinct
            if score > best_score:
                best_score = score
                best_cells = [idx]
            elif score == best_score:
                best_cells.append(idx)

        if not best_cells:
            return None
        clues.add(random.choice(best_cells))
    else:
        return None

    changed = True
    while changed:
        changed = False
        for idx in random.sample(list(clues), len(clues)):
            trial = set(clues)
            trial.remove(idx)
            trial_givens = _givens_from_clues(solution, trial)
            if len(_enumerate_solutions(border, trial_givens, use_ban=True, limit=2)) == 1:
                clues = trial
                changed = True

    return clues


def _clues_valid(solution: np.ndarray, border: np.ndarray, clues: set[int]) -> bool:
    n = int(solution.shape[0])
    if not (1 <= len(clues) <= n * n - 1):
        return False
    if n * n - len(clues) < n:
        return False

    givens = _givens_from_clues(solution, clues)
    with_ban = _enumerate_solutions(border, givens, use_ban=True, limit=2)
    if len(with_ban) != 1:
        return False
    if not np.array_equal(with_ban[0], solution):
        return False

    if len(_enumerate_solutions(border, givens, use_ban=False, limit=2)) <= 1:
        return False
    if len(_enumerate_solutions(border, np.full((n, n), -1, dtype=int), use_ban=True, limit=2)) <= 1:
        return False

    for idx in clues:
        trial = set(clues)
        trial.remove(idx)
        trial_givens = _givens_from_clues(solution, trial)
        if len(_enumerate_solutions(border, trial_givens, use_ban=True, limit=2)) == 1:
            return False

    blank_mask = np.ones((n, n), dtype=bool)
    for idx in clues:
        r, c = divmod(idx, n)
        blank_mask[r, c] = False
    if not all(blank_mask[r].any() for r in range(n)):
        return False
    if not all(blank_mask[:, c].any() for c in range(n)):
        return False
    return True


def _geometry_for_n(n: int) -> tuple[int, int, int, int, int, int] | None:
    options = []
    for side in range(4, 8):
        for gap in (1, 2):
            core = n * side + (n - 1) * gap
            if core > 28:
                continue
            slack = 30 - core
            if slack >= 2:
                options.append((side, gap, core, slack))
    if not options:
        return None

    side, gap, core, slack = random.choice(options)
    pad_h = random.randint(2, slack)
    pad_w = random.randint(2, slack)
    top = random.randint(1, pad_h - 1)
    left = random.randint(1, pad_w - 1)
    height = core + pad_h
    width = core + pad_w
    return side, gap, top, left, height, width


def _render_input(
    border: np.ndarray,
    solution: np.ndarray,
    clues: set[int],
    *,
    side: int,
    gap: int,
    top: int,
    left: int,
    height: int,
    width: int,
) -> np.ndarray:
    grid = np.full((height, width), Color.BLACK, dtype=int)
    n = int(border.shape[0])
    for r in range(n):
        for c in range(n):
            color = int(border[r, c])
            r0 = top + r * (side + gap)
            c0 = left + c * (side + gap)
            grid[r0, c0 : c0 + side] = color
            grid[r0 + side - 1, c0 : c0 + side] = color
            grid[r0 : r0 + side, c0] = color
            grid[r0 : r0 + side, c0 + side - 1] = color

            idx = r * n + c
            if idx in clues:
                grid[r0 + 1 : r0 + side - 1, c0 + 1 : c0 + side - 1] = int(solution[r, c])
    return grid


def _render_output(
    border: np.ndarray,
    solution: np.ndarray,
    *,
    side: int,
    gap: int,
    top: int,
    left: int,
    height: int,
    width: int,
) -> np.ndarray:
    return _render_input(
        border,
        solution,
        set(range(int(border.shape[0]) ** 2)),
        side=side,
        gap=gap,
        top=top,
        left=left,
        height=height,
        width=width,
    )


def _gen_attempt() -> np.ndarray | None:
    n = random.choice((3, 4, 5))
    geometry = _geometry_for_n(n)
    if geometry is None:
        return None
    side, gap, top, left, height, width = geometry

    colors = random.sample(Color.NOT_BLACK, n)
    solution = np.array(colors, dtype=int)[_latin_indices(n)]
    border = _make_border(solution)
    if not np.all(solution != border):
        return None

    clues = _pick_clues(solution, border)
    if clues is None or not _clues_valid(solution, border, clues):
        return None

    inp = _render_input(
        border,
        solution,
        clues,
        side=side,
        gap=gap,
        top=top,
        left=left,
        height=height,
        width=width,
    )
    expected = _render_output(
        border,
        solution,
        side=side,
        gap=gap,
        top=top,
        left=left,
        height=height,
        width=width,
    )

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    bg_count = Counter(map(int, inp.flat))[Color.BLACK]
    fg_counts = Counter(map(int, inp.flat))
    fg_counts.pop(Color.BLACK, None)
    if fg_counts and bg_count <= max(fg_counts.values()):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid border-palette Latin-square puzzle")
