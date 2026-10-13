from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]

BACKGROUND = 0
TEMPLATE_COLOR = 5
COPY_COLORS = (1, 2, 3, 6, 7, 8, 9)
DIAGONAL_RUN_LENGTH = 5
BASE_SLOTS = ((0, 0), (0, 7), (7, 0), (7, 7))
MAX_GENERATION_ATTEMPTS = 400

BASE_TEMPLATE = np.array(
    [
        [1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1],
    ],
    dtype=bool,
)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _coords_from_mask(mask: np.ndarray) -> tuple[Coord, ...]:
    return tuple((int(r), int(c)) for r, c in np.argwhere(mask))


def _components(mask: np.ndarray, connectivity: int) -> list[list[Coord]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        deltas = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        deltas = tuple(
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0)
        )

    comps: list[list[Coord]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if visited[rr, cc]:
            continue
        stack = [(rr, cc)]
        visited[rr, cc] = True
        comp: list[Coord] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for dr, dc in deltas:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _translate(offsets: tuple[Coord, ...], top_left: Coord) -> frozenset[Coord]:
    r0, c0 = top_left
    return frozenset((r0 + dr, c0 + dc) for dr, dc in offsets)


def _paint_cells(grid: np.ndarray, cells: frozenset[Coord] | tuple[Coord, ...] | list[Coord], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _template_offsets(grid: np.ndarray) -> tuple[Coord, ...]:
    mask = grid == TEMPLATE_COLOR
    comps = _components(mask, connectivity=4)
    assert len(comps) == 1, "expected exactly one 4-connected grey template"
    coords = comps[0]
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _diagonal_runs(grid: np.ndarray, color: int) -> list[list[Coord]]:
    mask = grid == int(color)
    h, w = mask.shape
    starts: list[Coord] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if rr == 0 or cc == 0 or not mask[rr - 1, cc - 1]:
            starts.append((rr, cc))

    seen: set[Coord] = set()
    runs: list[list[Coord]] = []
    for start_r, start_c in sorted(starts):
        run: list[Coord] = []
        r = start_r
        c = start_c
        while r < h and c < w and mask[r, c]:
            run.append((r, c))
            seen.add((r, c))
            r += 1
            c += 1
        runs.append(run)

    all_cells = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    assert seen == all_cells, "every colored segment must be a maximal down-right diagonal run"
    return runs


def _candidate_top_lefts(
    fragment: list[Coord] | tuple[Coord, ...],
    template_offsets: tuple[Coord, ...],
    shape: tuple[int, int],
) -> set[Coord]:
    h, w = shape
    frag_set = set(fragment)
    first_r, first_c = fragment[0]
    candidates: set[Coord] = set()
    for dr, dc in template_offsets:
        top_left = (first_r - dr, first_c - dc)
        translated = _translate(template_offsets, top_left)
        if any(r < 0 or c < 0 or r >= h or c >= w for r, c in translated):
            continue
        if frag_set <= translated:
            candidates.add(top_left)
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter each colored diagonal segment to its even-index pixels, then complete the uniquely aligned template copy.

    Input structure:
    - The background is black, and one grey 4-connected sprite is the only complete template.
    - Each other foreground color forms a one-cell-wide down-right diagonal run.
    - For every such run, only the even-indexed pixels (counted from the upper-left end) belong to a translated copy of the grey template.

    Rule:
    - Extract the grey template mask from its tight bounding box.
    - For each non-grey diagonal run, keep indices 0, 2, 4, ... as the fragment and find the unique translation whose template cells contain that fragment.
    - Paint the whole translated template in that run's color and erase the odd-indexed diagonal pixels back to background.

    Key insight:
    - The visible diagonal is deliberately misleading: only after odd/even filtering do the surviving pixels become a matchable template fragment.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(_get_bg(g))
    template_offsets = _template_offsets(g)

    out = g.copy()
    colors = [int(color) for color in np.unique(g) if int(color) not in (bg, TEMPLATE_COLOR)]
    for color in colors:
        for run in _diagonal_runs(g, color):
            assert run, "each foreground color must contribute at least one diagonal run"
            fragment = run[::2]
            candidates = _candidate_top_lefts(fragment, template_offsets, g.shape)
            assert len(candidates) == 1, "filtered fragment must determine a unique template placement"
            top_left = next(iter(candidates))

            for idx, (r, c) in enumerate(run):
                if idx % 2 == 1:
                    out[r, c] = bg

            for r, c in _translate(template_offsets, top_left):
                out[r, c] = color
    return out


def _sparse_diagonal_triples(mask: np.ndarray) -> tuple[tuple[Coord, Coord, Coord], ...]:
    h, w = mask.shape
    cells = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    triples: set[tuple[Coord, Coord, Coord]] = set()
    for r, c in cells:
        if r + 4 >= h or c + 4 >= w:
            continue
        triple = ((r, c), (r + 2, c + 2), (r + 4, c + 4))
        if all(cell in cells for cell in triple) and (r + 1, c + 1) not in cells and (r + 3, c + 3) not in cells:
            triples.add(triple)
    return tuple(sorted(triples))


def _template_variants() -> tuple[dict[str, object], ...]:
    raw: list[np.ndarray] = []
    for seed in (BASE_TEMPLATE, np.fliplr(BASE_TEMPLATE)):
        for k in range(4):
            raw.append(np.rot90(seed, k))

    seen: set[tuple[tuple[int, int], bytes]] = set()
    catalog: list[dict[str, object]] = []
    for mask in raw:
        key = (tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes())
        if key in seen:
            continue
        seen.add(key)
        anchors = _sparse_diagonal_triples(mask)
        if anchors:
            catalog.append({"mask": mask.astype(bool).copy(), "anchors": anchors})
    return tuple(catalog)


TEMPLATE_VARIANTS = _template_variants()
assert TEMPLATE_VARIANTS, "expected at least one template variant with a sparse diagonal anchor triple"


def _build_expected(
    shape: tuple[int, int],
    template_offsets: tuple[Coord, ...],
    template_top_left: Coord,
    copies: list[dict[str, object]],
) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    _paint_cells(out, _translate(template_offsets, template_top_left), TEMPLATE_COLOR)
    for copy in copies:
        _paint_cells(out, copy["full_cells"], int(copy["color"]))
    return out


def _build_instance() -> tuple[np.ndarray, np.ndarray, tuple[Coord, ...], Coord, list[dict[str, object]]]:
    variant = random.choice(TEMPLATE_VARIANTS)
    template_mask = np.array(variant["mask"], dtype=bool)
    anchor_line = random.choice(variant["anchors"])
    template_offsets = tuple(sorted(_coords_from_mask(template_mask)))

    extra_h = random.randint(0, 2)
    extra_w = random.randint(0, 2)
    row_shift = random.randint(0, extra_h)
    col_shift = random.randint(0, extra_w)
    h = 13 + extra_h
    w = 13 + extra_w
    slots = [(row_shift + r, col_shift + c) for r, c in BASE_SLOTS]

    num_copies = random.choice((2, 3))
    selected_slots = random.sample(slots, num_copies + 1)
    random.shuffle(selected_slots)
    template_top_left = selected_slots[0]
    copy_slots = selected_slots[1:]
    colors = random.sample(COPY_COLORS, num_copies)

    grid = np.full((h, w), BACKGROUND, dtype=int)
    _paint_cells(grid, _translate(template_offsets, template_top_left), TEMPLATE_COLOR)

    copies: list[dict[str, object]] = []
    anchor_start = anchor_line[0]
    for top_left, color in zip(copy_slots, colors):
        run = [(top_left[0] + anchor_start[0] + i, top_left[1] + anchor_start[1] + i) for i in range(DIAGONAL_RUN_LENGTH)]
        for r, c in run:
            grid[r, c] = int(color)
        copies.append(
            {
                "color": int(color),
                "top_left": top_left,
                "run": run,
                "fragment": run[::2],
                "full_cells": _translate(template_offsets, top_left),
            }
        )

    expected = _build_expected(grid.shape, template_offsets, template_top_left, copies)
    return grid, expected, template_offsets, template_top_left, copies


def _validate_instance(
    grid: np.ndarray,
    expected: np.ndarray,
    template_offsets: tuple[Coord, ...],
    template_top_left: Coord,
    copies: list[dict[str, object]],
) -> bool:
    if not (grid.ndim == 2 and expected.shape == grid.shape):
        return False
    if not (6 <= len(template_offsets) <= 10):
        return False

    grey_cells = _translate(template_offsets, template_top_left)
    grey_mask = grid == TEMPLATE_COLOR
    grey_components = _components(grey_mask, connectivity=4)
    if len(grey_components) != 1 or frozenset(grey_components[0]) != grey_cells:
        return False
    if len(grey_cells) <= DIAGONAL_RUN_LENGTH:
        return False

    occupied_full: list[frozenset[Coord]] = []
    seen_colors: set[int] = set()
    allowed_input_cells = set(grey_cells)

    for copy in copies:
        color = int(copy["color"])
        run = list(copy["run"])
        fragment = list(copy["fragment"])
        full_cells = frozenset(copy["full_cells"])
        top_left = tuple(copy["top_left"])

        if color in seen_colors:
            return False
        seen_colors.add(color)
        if len(run) != DIAGONAL_RUN_LENGTH or len(fragment) != 3:
            return False
        if any(grid[r, c] != color for r, c in run):
            return False
        if any((run[i + 1][0] - run[i][0], run[i + 1][1] - run[i][1]) != (1, 1) for i in range(len(run) - 1)):
            return False
        if set(fragment) - full_cells:
            return False
        if any(cell in full_cells for cell in run[1::2]):
            return False
        if len(full_cells) <= len(fragment):
            return False
        if _candidate_top_lefts(fragment, template_offsets, grid.shape) != {top_left}:
            return False
        if _candidate_top_lefts(run, template_offsets, grid.shape):
            return False
        if full_cells & grey_cells:
            return False
        if any(full_cells & other for other in occupied_full):
            return False

        occupied_full.append(full_cells)
        allowed_input_cells.update(run)

    actual_input_cells = {(int(r), int(c)) for r, c in np.argwhere(grid != BACKGROUND)}
    if actual_input_cells != allowed_input_cells:
        return False

    rebuilt_expected = _build_expected(grid.shape, template_offsets, template_top_left, copies)
    return np.array_equal(expected, rebuilt_expected)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid, expected, template_offsets, template_top_left, copies = _build_instance()
        if not _validate_instance(grid, expected, template_offsets, template_top_left, copies):
            continue
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid non-degenerate instance")
