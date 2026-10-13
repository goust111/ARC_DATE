from __future__ import annotations

import itertools

import numpy as np


BG = 0
CONFLICT = 4
INPUT_COLORS = (2, 3)
GEN_ATTEMPTS = 600
RNG = np.random.default_rng()


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _coords_on_diag(shape: tuple[int, int], diag_key: int) -> list[tuple[int, int]]:
    height, width = map(int, shape)
    cells: list[tuple[int, int]] = []
    for r in range(height):
        c = r - int(diag_key)
        if 0 <= c < width:
            cells.append((r, c))
    return cells


def _source_sequences(grid: np.ndarray) -> dict[int, list[int]]:
    by_diag: dict[int, list[tuple[int, int]]] = {}
    for r, c in np.argwhere(grid != BG):
        rr = int(r)
        cc = int(c)
        by_diag.setdefault(rr - cc, []).append((rr, int(grid[rr, cc])))
    return {
        diag_key: [color for _row, color in sorted(entries)]
        for diag_key, entries in by_diag.items()
    }


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    out = np.zeros((2 * height, 2 * width), dtype=int)

    by_diag: dict[int, list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g != BG):
        rr = int(r)
        cc = int(c)
        by_diag.setdefault(rr - cc, []).append((rr, int(g[rr, cc])))

    for diag_key, entries in by_diag.items():
        starts = sorted(entries)
        seen: set[int] = set()
        idx = 0
        for r, c in _coords_on_diag(out.shape, diag_key):
            while idx < len(starts) and starts[idx][0] <= r:
                seen.add(int(starts[idx][1]))
                idx += 1
            if not seen:
                continue
            out[r, c] = CONFLICT if len(seen) > 1 else next(iter(seen))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the two-color sparse pattern along the down-right diagonal and mark opposite-color overlaps yellow.

    Input structure:
    - The input is a small black-background grid whose foreground contains exactly two non-black colors.
    - Foreground pixels are sparse and are interpreted as transparent stamps: only non-black cells matter.
    - Pixels sharing the same (r-c) diagonal class can meet again after different diagonal offsets.

    Rule:
    - Let the input size be HxW and create a black 2H x 2W canvas.
    - For every non-negative offset k, stamp each input foreground pixel at (r+k, c+k) while that target lies inside the canvas.
    - Stamping onto black writes the color, stamping onto the same color keeps it unchanged, and stamping onto a different non-black color turns that cell yellow (4), which then remains yellow.

    Key insight:
    - Diagonal tiling makes pixels from the same input (r-c) class accumulate on one output diagonal, and yellow appears exactly when that accumulated history contains both foreground colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    height, width = map(int, g.shape)
    out = np.zeros((2 * height, 2 * width), dtype=int)

    fg_coords = np.argwhere(g != BG)
    for k in range(min(2 * height, 2 * width)):
        for r, c in fg_coords:
            rr = int(r) + k
            cc = int(c) + k
            if rr >= out.shape[0] or cc >= out.shape[1]:
                continue
            color = int(g[int(r), int(c)])
            prev = int(out[rr, cc])
            if prev == BG:
                out[rr, cc] = color
            elif prev == color or prev == CONFLICT:
                continue
            else:
                out[rr, cc] = CONFLICT

    return out


def _all_far(cells: list[tuple[int, int]], used: list[tuple[int, int]]) -> bool:
    for r0, c0 in cells:
        for r1, c1 in used:
            if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                return False
    return True


def _pairwise_separated(cells: list[tuple[int, int]]) -> bool:
    for i, (r0, c0) in enumerate(cells):
        for r1, c1 in cells[i + 1 :]:
            if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                return False
    return True


def _diag_combos(
    diag_cells: list[tuple[int, int]],
    count: int,
    used: list[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    combos: list[list[tuple[int, int]]] = []
    for idxs in itertools.combinations(range(len(diag_cells)), count):
        if any(b - a < 2 for a, b in zip(idxs, idxs[1:])):
            continue
        cells = [diag_cells[i] for i in idxs]
        if _all_far(cells, used):
            combos.append(cells)
    return combos


def _pick_diag_cells(
    shape: tuple[int, int],
    diag_key: int,
    used: list[tuple[int, int]],
    prefer_three: bool,
) -> list[tuple[int, int]] | None:
    diag_cells = _coords_on_diag(shape, diag_key)
    counts = [3, 2] if prefer_three else [2, 3]
    for count in counts:
        combos = _diag_combos(diag_cells, count, used)
        if combos:
            return combos[int(RNG.integers(len(combos)))]
    return None


def _pick_singleton(
    shape: tuple[int, int],
    used: list[tuple[int, int]],
    forbidden_diags: set[int],
) -> tuple[int, int] | None:
    height, width = map(int, shape)
    candidates = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if (r, c) not in used
        and (r - c) not in forbidden_diags
        and _all_far([(r, c)], used)
    ]
    if not candidates:
        candidates = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if (r, c) not in used and _all_far([(r, c)], used)
        ]
    if not candidates:
        return None
    return candidates[int(RNG.integers(len(candidates)))]


def _is_valid_input(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        return False
    height, width = map(int, g.shape)
    if not (5 <= height <= 7 and 5 <= width <= 7):
        return False

    colors = _colors_present(g, exclude={BG})
    if colors != list(INPUT_COLORS):
        return False

    total_fg = int(np.count_nonzero(g))
    if not (4 <= total_fg <= 8):
        return False

    counts = {color: int(np.count_nonzero(g == color)) for color in INPUT_COLORS}
    if min(counts.values()) < 2:
        return False

    used = [tuple(map(int, rc)) for rc in np.argwhere(g != BG)]
    if not _pairwise_separated(used):
        return False

    by_diag = _source_sequences(g)
    if len(by_diag) < 2:
        return False

    mixed_diags = [d for d, seq in by_diag.items() if len(seq) >= 2 and len(set(seq)) > 1]
    mono_diags = [d for d, seq in by_diag.items() if len(seq) >= 2 and len(set(seq)) == 1]
    if not mixed_diags or not mono_diags:
        return False

    brute = main(g)
    oracle = _oracle_output(g)
    if not np.array_equal(brute, oracle):
        return False

    out_colors = set(_colors_present(brute, exclude={BG}))
    if out_colors != {INPUT_COLORS[0], INPUT_COLORS[1], CONFLICT}:
        return False

    yellow_count = int(np.count_nonzero(brute == CONFLICT))
    occupied_count = int(np.count_nonzero(brute != BG))
    if yellow_count == 0 or yellow_count >= occupied_count:
        return False

    mixed_visible = False
    for diag_key in mixed_diags:
        seq = [int(brute[r, c]) for r, c in _coords_on_diag(brute.shape, diag_key)]
        painted = [v for v in seq if v != BG]
        if CONFLICT in painted:
            first_conflict = painted.index(CONFLICT)
            if first_conflict > 0 and all(v in INPUT_COLORS for v in painted[:first_conflict]):
                if all(v == CONFLICT for v in painted[first_conflict:]):
                    mixed_visible = True
                    break
    if not mixed_visible:
        return False

    mono_visible = False
    for diag_key in mono_diags:
        seq = [int(brute[r, c]) for r, c in _coords_on_diag(brute.shape, diag_key)]
        painted = [v for v in seq if v != BG]
        if len(painted) >= 3 and CONFLICT not in painted and len(set(painted)) == 1:
            mono_visible = True
            break
    if not mono_visible:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = int(RNG.integers(5, 8))
    width = int(RNG.integers(5, 8))
    grid = np.zeros((height, width), dtype=int)

    candidate_diags = [d for d in range(-(width - 1), height) if len(_coords_on_diag(grid.shape, d)) >= 3]
    if len(candidate_diags) < 2:
        return None

    perm = RNG.permutation(candidate_diags)
    mixed_diag = int(perm[0])
    mono_diag = int(perm[1])

    used: list[tuple[int, int]] = []

    mixed_cells = _pick_diag_cells(grid.shape, mixed_diag, used, prefer_three=bool(RNG.integers(2)))
    if mixed_cells is None:
        return None
    used.extend(mixed_cells)

    mixed_colors = [INPUT_COLORS[0], INPUT_COLORS[1]]
    if len(mixed_cells) == 3:
        mixed_colors.append(INPUT_COLORS[int(RNG.integers(2))])
    mixed_colors = list(np.array(mixed_colors, dtype=int)[RNG.permutation(len(mixed_colors))])
    for (r, c), color in zip(mixed_cells, mixed_colors):
        grid[r, c] = int(color)

    mono_cells = _pick_diag_cells(grid.shape, mono_diag, used, prefer_three=bool(RNG.integers(2)))
    if mono_cells is None:
        return None
    used.extend(mono_cells)

    mono_color = int(INPUT_COLORS[int(RNG.integers(2))])
    for r, c in mono_cells:
        grid[r, c] = mono_color

    counts = {color: int(np.count_nonzero(grid == color)) for color in INPUT_COLORS}
    occupied_diags = {mixed_diag, mono_diag}

    for color in INPUT_COLORS:
        while counts[color] < 2:
            cell = _pick_singleton(grid.shape, used, occupied_diags)
            if cell is None:
                return None
            r, c = cell
            grid[r, c] = color
            used.append(cell)
            occupied_diags.add(r - c)
            counts[color] += 1

    min_total = int(np.count_nonzero(grid))
    max_total = min(8, min_total + 2)
    target_total = int(RNG.integers(min_total, max_total + 1))
    while int(np.count_nonzero(grid)) < target_total:
        cell = _pick_singleton(grid.shape, used, occupied_diags)
        if cell is None:
            break
        r, c = cell
        color = int(INPUT_COLORS[int(RNG.integers(2))])
        grid[r, c] = color
        used.append(cell)
        occupied_diags.add(r - c)

    return grid if _is_valid_input(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-conflict tiling input")
