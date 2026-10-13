from __future__ import annotations

import random

import numpy as np


BLACK = 0
NON_BLACK_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _extract_palette(grid: np.ndarray) -> list[int]:
    row = np.asarray(grid[0], dtype=int)
    assert row.size >= 2, "palette row must have at least two columns"
    assert row[0] != BLACK, "palette must start at the leftmost cell"

    k = 0
    while k < row.size and int(row[k]) != BLACK:
        k += 1

    palette = [int(x) for x in row[:k]]
    assert palette, "palette row must contain at least one color"
    assert len(set(palette)) == len(palette), "palette colors must be distinct"
    assert all(color != BLACK for color in palette), "palette colors must be non-black"
    assert np.all(row[k:] == BLACK), "cells after the palette must be black"
    return palette


def _collect_samples(grid: np.ndarray, palette: list[int]) -> list[tuple[int, int, int]]:
    palette_set = set(palette)
    samples: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(grid[1:] != BLACK):
        rr = int(r) + 1
        cc = int(c)
        color = int(grid[rr, cc])
        assert color in palette_set, "sample colors must come from the palette row"
        samples.append((rr, cc, color))
    assert samples, "expected at least one colored sample below the palette row"
    return samples


def _candidate_centers(
    shape: tuple[int, int], palette: list[int], samples: list[tuple[int, int, int]]
) -> list[tuple[int, int]]:
    height, width = map(int, shape)
    k = len(palette)
    color_to_index = {color: idx for idx, color in enumerate(palette)}

    centers: list[tuple[int, int]] = []
    for cr in range(1, height):
        for cc in range(width):
            ok = True
            for r, c, color in samples:
                d = abs(r - cr) + abs(c - cc)
                if d == 0 or (d - 1) % k != color_to_index[color]:
                    ok = False
                    break
            if ok:
                centers.append((cr, cc))
    return centers


def _render_output(shape: tuple[int, int], palette: list[int], center: tuple[int, int]) -> np.ndarray:
    height, width = map(int, shape)
    cr, cc = map(int, center)
    k = len(palette)

    out = np.zeros((height, width), dtype=int)
    out[0, :k] = np.asarray(palette, dtype=int)
    if height == 1:
        return out

    rows, cols = np.indices((height - 1, width))
    distances = np.abs((rows + 1) - cr) + np.abs(cols - cc)
    board = np.asarray(palette, dtype=int)[(distances - 1) % k]
    board[distances == 0] = BLACK
    out[1:, :] = board
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full Manhattan-ring field by inferring the hidden center from sparse palette-coded samples.

    Input structure:
    - The top row begins with a contiguous left-aligned palette of distinct non-black colors, and the rest of that row is black.
    - Below the palette row is a black board containing only a small number of single-pixel colored samples.
    - Every sample color appears in the palette row, while the true ring center is never drawn in the input.

    Rule:
    - Find the unique board cell that can serve as a hidden center such that each sample's color matches
      the palette entry indexed by its Manhattan distance: distance d>0 uses palette[(d-1) mod k].
    - Keep the palette row unchanged.
    - Recolor every board cell by the same distance-to-center rule, except the center cell itself, which stays black.

    Key insight:
    - The colored dots are sparse observations of a missing concentric Manhattan distance field, so the task is to invert
      the palette-distance encoding to recover the unseen center before filling the whole board.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 3 and g.shape[1] >= 3, "expected a palette row plus a board"

    palette = _extract_palette(g)
    samples = _collect_samples(g, palette)
    centers = _candidate_centers(g.shape, palette, samples)
    assert len(centers) == 1, "expected a unique hidden center"
    return _render_output(g.shape, palette, centers[0])


# --------------- generator ---------------

TOTAL_HEIGHT_RANGE = (8, 18)
TOTAL_WIDTH_RANGE = (8, 20)
PALETTE_LEN_RANGE = (3, 6)
SAMPLE_COUNT_RANGE = (4, 8)
MAX_GENERATION_ATTEMPTS = 500


def _board_max_distance(height: int, width: int, center: tuple[int, int]) -> int:
    cr, cc = center
    corners = [(1, 0), (1, width - 1), (height - 1, 0), (height - 1, width - 1)]
    return max(abs(r - cr) + abs(c - cc) for r, c in corners)


def _choose_layout() -> tuple[int, int, list[int], tuple[int, int]]:
    for _ in range(200):
        height = random.randint(*TOTAL_HEIGHT_RANGE)
        width = random.randint(*TOTAL_WIDTH_RANGE)
        max_palette = min(PALETTE_LEN_RANGE[1], width - 1)
        if max_palette < PALETTE_LEN_RANGE[0]:
            continue
        palette_len = random.randint(PALETTE_LEN_RANGE[0], max_palette)
        palette = random.sample(NON_BLACK_COLORS, palette_len)

        center_r = random.randint(2, height - 2)
        center_c = random.randint(1, width - 2)
        center = (center_r, center_c)
        if _board_max_distance(height, width, center) <= palette_len:
            continue
        return height, width, palette, center
    raise RuntimeError("failed to choose a non-degenerate layout")


def _make_cell(r: int, c: int, color: int, distance: int, center: tuple[int, int]) -> dict[str, int | tuple[int, int] | bool]:
    cr, cc = center
    dr = -1 if r < cr else 1 if r > cr else 0
    dc = -1 if c < cc else 1 if c > cc else 0
    return {
        "r": r,
        "c": c,
        "color": color,
        "d": distance,
        "axis": dr == 0 or dc == 0,
        "quad": (dr, dc),
    }


def _field_cells(shape: tuple[int, int], palette: list[int], center: tuple[int, int]) -> list[dict[str, int | tuple[int, int] | bool]]:
    height, width = shape
    cr, cc = center
    k = len(palette)
    cells: list[dict[str, int | tuple[int, int] | bool]] = []
    for r in range(1, height):
        for c in range(width):
            if (r, c) == (cr, cc):
                continue
            distance = abs(r - cr) + abs(c - cc)
            color = palette[(distance - 1) % k]
            cells.append(_make_cell(r, c, color, distance, center))
    return cells


def _as_samples(cells: list[dict[str, int | tuple[int, int] | bool]], idxs: list[int]) -> list[tuple[int, int, int]]:
    return [
        (int(cells[idx]["r"]), int(cells[idx]["c"]), int(cells[idx]["color"]))
        for idx in idxs
    ]


def _basic_diversity_ok(
    cells: list[dict[str, int | tuple[int, int] | bool]],
    idxs: list[int],
    center: tuple[int, int],
    palette_len: int,
) -> bool:
    if len(idxs) < SAMPLE_COUNT_RANGE[0] or len(idxs) > SAMPLE_COUNT_RANGE[1]:
        return False

    chosen = [cells[idx] for idx in idxs]
    distances = {int(cell["d"]) for cell in chosen}
    colors = {int(cell["color"]) for cell in chosen}
    rows = {int(cell["r"]) for cell in chosen}
    cols = {int(cell["c"]) for cell in chosen}
    quads = {cell["quad"] for cell in chosen}

    if len(distances) < 3 or len(colors) < min(3, palette_len):
        return False
    if len(rows) == 1 or len(cols) == 1:
        return False
    if len(quads) < 3:
        return False
    if not any(int(cell["d"]) > palette_len for cell in chosen):
        return False
    if not any(not bool(cell["axis"]) for cell in chosen):
        return False

    cr, cc = center
    if not any(int(cell["r"]) < cr for cell in chosen):
        return False
    if not any(int(cell["r"]) > cr for cell in chosen):
        return False
    if not any(int(cell["c"]) < cc for cell in chosen):
        return False
    if not any(int(cell["c"]) > cc for cell in chosen):
        return False

    repeated_color = False
    for color in colors:
        seen_distances = {int(cell["d"]) for cell in chosen if int(cell["color"]) == color}
        if len(seen_distances) >= 2:
            repeated_color = True
            break
    return repeated_color


def _semantic_checks(
    shape: tuple[int, int],
    palette: list[int],
    center: tuple[int, int],
    cells: list[dict[str, int | tuple[int, int] | bool]],
    idxs: list[int],
) -> bool:
    if not _basic_diversity_ok(cells, idxs, center, len(palette)):
        return False

    centers = _candidate_centers(shape, palette, _as_samples(cells, idxs))
    if centers != [center]:
        return False

    essential = 0
    for idx in idxs:
        reduced = [other for other in idxs if other != idx]
        reduced_centers = _candidate_centers(shape, palette, _as_samples(cells, reduced))
        if len(reduced_centers) != 1:
            essential += 1
    return essential >= 2


def _choose_repeated_pair(
    cells: list[dict[str, int | tuple[int, int] | bool]], palette_len: int
) -> tuple[int, int]:
    by_color: dict[int, list[int]] = {}
    for idx, cell in enumerate(cells):
        by_color.setdefault(int(cell["color"]), []).append(idx)

    best: tuple[tuple[int, int, int, int], tuple[int, int]] | None = None
    for idxs in by_color.values():
        near = [idx for idx in idxs if int(cells[idx]["d"]) <= palette_len]
        far = [idx for idx in idxs if int(cells[idx]["d"]) > palette_len]
        for a in near:
            for b in far:
                if a == b:
                    continue
                ca = cells[a]
                cb = cells[b]
                quad_score = len({ca["quad"], cb["quad"]})
                off_axis_score = int(not bool(ca["axis"])) + int(not bool(cb["axis"]))
                span = abs(int(ca["r"]) - int(cb["r"])) + abs(int(ca["c"]) - int(cb["c"]))
                dist_gap = int(cb["d"]) - int(ca["d"])
                score = (quad_score, off_axis_score, span, dist_gap)
                if best is None or score > best[0]:
                    best = (score, (a, b))
    assert best is not None, "expected a repeated palette color across two distance cycles"
    return best[1]


def _pick_distinct_color_point(
    cells: list[dict[str, int | tuple[int, int] | bool]],
    chosen: list[int],
    forbidden_colors: set[int],
) -> int | None:
    used_quads = {cells[idx]["quad"] for idx in chosen}
    best: tuple[tuple[int, int, int, int], int] | None = None
    for idx, cell in enumerate(cells):
        if idx in chosen or int(cell["color"]) in forbidden_colors:
            continue
        score = (
            int(not bool(cell["axis"])),
            int(int(cell["d"]) > 0),
            int(cell["quad"] not in used_quads),
            int(cell["d"]),
        )
        if best is None or score > best[0]:
            best = (score, idx)
    return None if best is None else best[1]


def _build_seed_set(
    cells: list[dict[str, int | tuple[int, int] | bool]], palette_len: int
) -> list[int]:
    first, second = _choose_repeated_pair(cells, palette_len)
    chosen = [first, second]
    used_colors = {int(cells[first]["color"])}

    third = _pick_distinct_color_point(cells, chosen, used_colors)
    assert third is not None, "expected a second palette color in the sampled hints"
    chosen.append(third)
    used_colors.add(int(cells[third]["color"]))

    fourth = _pick_distinct_color_point(cells, chosen, used_colors)
    assert fourth is not None, "expected a third palette color in the sampled hints"
    chosen.append(fourth)
    return chosen


def _greedy_unique_subset(
    shape: tuple[int, int],
    palette: list[int],
    center: tuple[int, int],
    cells: list[dict[str, int | tuple[int, int] | bool]],
    seed: list[int],
) -> list[int]:
    selected = list(dict.fromkeys(seed))
    remaining = [idx for idx in range(len(cells)) if idx not in selected]

    while _candidate_centers(shape, palette, _as_samples(cells, selected)) != [center]:
        used_colors = {int(cells[idx]["color"]) for idx in selected}
        used_quads = {cells[idx]["quad"] for idx in selected}
        best: tuple[tuple[int, int, int, int, int], int] | None = None
        for idx in remaining:
            trial = selected + [idx]
            remaining_centers = _candidate_centers(shape, palette, _as_samples(cells, trial))
            cell = cells[idx]
            score = (
                -len(remaining_centers),
                int(int(cell["color"]) not in used_colors),
                int(cell["quad"] not in used_quads),
                int(not bool(cell["axis"])),
                int(cell["d"]),
            )
            if best is None or score > best[0]:
                best = (score, idx)
        assert best is not None, "expected to find a center-disambiguating sample"
        picked = best[1]
        selected.append(picked)
        remaining.remove(picked)

    return selected


def _prune_subset(
    shape: tuple[int, int],
    palette: list[int],
    center: tuple[int, int],
    cells: list[dict[str, int | tuple[int, int] | bool]],
    idxs: list[int],
) -> list[int]:
    selected = list(idxs)
    changed = True
    while changed:
        changed = False
        order = selected[:]
        random.shuffle(order)
        for idx in order:
            trial = [other for other in selected if other != idx]
            if _candidate_centers(shape, palette, _as_samples(cells, trial)) != [center]:
                continue
            if not _basic_diversity_ok(cells, trial, center, len(palette)):
                continue
            selected = trial
            changed = True
            break
    return selected


def _generate_sparse_samples(
    shape: tuple[int, int], palette: list[int], center: tuple[int, int]
) -> list[tuple[int, int, int]]:
    cells = _field_cells(shape, palette, center)
    seed = _build_seed_set(cells, len(palette))
    selected = _greedy_unique_subset(shape, palette, center, cells, seed)
    selected = _prune_subset(shape, palette, center, cells, selected)

    if not _semantic_checks(shape, palette, center, cells, selected):
        raise RuntimeError("semantic checks rejected the sampled hints")

    return _as_samples(cells, selected)


def _build_input(
    shape: tuple[int, int], palette: list[int], samples: list[tuple[int, int, int]]
) -> np.ndarray:
    height, width = shape
    grid = np.zeros((height, width), dtype=int)
    grid[0, : len(palette)] = np.asarray(palette, dtype=int)
    for r, c, color in samples:
        grid[r, c] = color
    return grid


def _generate_candidate() -> np.ndarray:
    height, width, palette, center = _choose_layout()
    shape = (height, width)
    samples = _generate_sparse_samples(shape, palette, center)
    grid = _build_input(shape, palette, samples)
    expected = _render_output(shape, palette, center)
    actual = main(grid)
    assert np.array_equal(actual, expected), "generator/solver mismatch"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            return _generate_candidate()
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid sparse Manhattan-ring reconstruction input")
