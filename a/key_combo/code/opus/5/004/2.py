from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
AXIS = 5
SOURCE = 7
MIRROR_NW_SE = 2
MIRROR_NE_SW = 8

NW_SE = "nw_se"
NE_SW = "ne_sw"
ORIENTATIONS = (NW_SE, NE_SW)
SIDES = (0, 1)

GRID_MIN = 13
GRID_MAX = 19
MAX_ATTEMPTS = 400

_SIDE_SPEC_CACHE: dict[str, dict[str, list[dict[str, tuple[int, ...]]]]] = {}


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _find_axis(grid: np.ndarray) -> tuple[str, int]:
    row_hits = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == AXIS)]
    col_hits = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == AXIS)]
    candidates = [("row", r) for r in row_hits] + [("col", c) for c in col_hits]
    assert len(candidates) == 1, "expected exactly one full row or column axis"
    return candidates[0]


def _reflect(axis_kind: str, axis_index: int, r: int, c: int) -> tuple[int, int]:
    if axis_kind == "row":
        return 2 * axis_index - r, c
    return r, 2 * axis_index - c


def _component_orientation(coords: np.ndarray) -> str:
    diffs = {int(r - c) for r, c in coords}
    sums = {int(r + c) for r, c in coords}
    if len(diffs) == 1 and len(sums) > 1:
        return NW_SE
    if len(sums) == 1 and len(diffs) > 1:
        return NE_SW
    raise AssertionError("orange component must be a pure diagonal bar")


def _component_side(axis_kind: str, axis_index: int, coords: np.ndarray) -> int:
    if axis_kind == "row":
        rows = coords[:, 0]
        if int(rows.max()) < axis_index:
            return 0
        if int(rows.min()) > axis_index:
            return 1
    else:
        cols = coords[:, 1]
        if int(cols.max()) < axis_index:
            return 0
        if int(cols.min()) > axis_index:
            return 1
    raise AssertionError("orange component must lie fully on one side of the axis")


def _minority_orientation(pixel_totals: dict[str, int], segment_counts: dict[str, int]) -> str:
    nw_pixels = pixel_totals[NW_SE]
    ne_pixels = pixel_totals[NE_SW]
    if nw_pixels != ne_pixels:
        return NW_SE if nw_pixels < ne_pixels else NE_SW

    nw_segments = segment_counts[NW_SE]
    ne_segments = segment_counts[NE_SW]
    if nw_segments != ne_segments:
        return NW_SE if nw_segments < ne_segments else NE_SW
    return NE_SW


def _parse_scene(grid: np.ndarray) -> tuple[tuple[str, int], list[dict[str, object]], dict[int, dict[str, dict[str, int]]]]:
    axis = _find_axis(grid)
    orange_comps = _components(grid == SOURCE)
    assert orange_comps, "expected orange diagonal bars"

    records: list[dict[str, object]] = []
    stats = {
        side: {
            "pixels": {NW_SE: 0, NE_SW: 0},
            "segments": {NW_SE: 0, NE_SW: 0},
        }
        for side in SIDES
    }
    for comp in orange_comps:
        area = int(comp.shape[0])
        assert 2 <= area <= 5, "diagonal bar length must be between 2 and 5"
        orient = _component_orientation(comp)
        side = _component_side(axis[0], axis[1], comp)
        if axis[0] == "row":
            assert all(abs(int(r) - axis[1]) > 1 for r, _c in comp), "bar touches row axis"
        else:
            assert all(abs(int(c) - axis[1]) > 1 for _r, c in comp), "bar touches column axis"
        stats[side]["pixels"][orient] += area
        stats[side]["segments"][orient] += 1
        records.append({"coords": comp, "orientation": orient, "side": side, "area": area})

    for side in SIDES:
        for orient in ORIENTATIONS:
            assert stats[side]["segments"][orient] > 0, "each side must contain both orientations"
    return axis, records, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the minority diagonal-bar orientation on each axis side independently.

    Input structure:
    - The background is black, with one full grey row or one full grey column acting as a reflection axis.
    - Orange foreground objects are isolated single-cell-wide diagonal bars, each 8-connected and aligned purely along the NW-SE or NE-SW diagonal direction.
    - Every orange bar lies strictly on one side of the axis, and each side contains at least one bar of each diagonal orientation.

    Rule:
    - Classify every orange bar by diagonal orientation.
    - Split the bars by which side of the axis they occupy, then on each side compare the total orange pixel count of the two orientations.
    - On each side, choose the orientation with fewer pixels; if pixel totals tie, choose the one with fewer bars; if still tied, choose NE-SW.
    - Reflect only the chosen-side bars across the axis, coloring mirrored NW-SE copies red and mirrored NE-SW copies teal.
    - Keep the original orange bars and the grey axis unchanged, and skip any mirrored pixel whose destination is already non-background.

    Key insight:
    - The minority choice is not global: each side of the axis decides independently which diagonal orientation gets mirrored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    axis, records, stats = _parse_scene(g)
    side_choice = {
        side: _minority_orientation(stats[side]["pixels"], stats[side]["segments"])
        for side in SIDES
    }

    out = g.copy()
    h, w = g.shape
    for rec in records:
        orient = rec["orientation"]
        side = rec["side"]
        if orient != side_choice[side]:
            continue
        paint = MIRROR_NW_SE if orient == NW_SE else MIRROR_NE_SW
        for r, c in rec["coords"]:
            rr, cc = _reflect(axis[0], axis[1], int(r), int(c))
            assert 0 <= rr < h and 0 <= cc < w
            if out[rr, cc] == BG:
                out[rr, cc] = paint
    return out


def _mark_buffer(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        if _in_bounds(blocked.shape, r, c):
            blocked[r, c] = True
        for nr, nc in _neighbors8(r, c):
            if _in_bounds(blocked.shape, nr, nc):
                blocked[nr, nc] = True


def _segment_cells(start_r: int, start_c: int, orient: str, length: int) -> list[tuple[int, int]]:
    if orient == NW_SE:
        return [(start_r + k, start_c + k) for k in range(length)]
    return [(start_r + k, start_c - k) for k in range(length)]


def _fits_side(axis_kind: str, axis_index: int, side: int, cells: list[tuple[int, int]]) -> bool:
    if axis_kind == "row":
        rows = [r for r, _c in cells]
        if any(abs(r - axis_index) <= 1 for r in rows):
            return False
        return max(rows) < axis_index if side == 0 else min(rows) > axis_index
    cols = [c for _r, c in cells]
    if any(abs(c - axis_index) <= 1 for c in cols):
        return False
    return max(cols) < axis_index if side == 0 else min(cols) > axis_index


def _enumerate_candidates(
    shape: tuple[int, int], axis_kind: str, axis_index: int, side: int, orient: str, length: int
) -> list[list[tuple[int, int]]]:
    h, w = shape
    out: list[list[tuple[int, int]]] = []
    if orient == NW_SE:
        c_range = range(w - length + 1)
    else:
        c_range = range(length - 1, w)
    for r0 in range(h - length + 1):
        for c0 in c_range:
            cells = _segment_cells(r0, c0, orient, length)
            if _fits_side(axis_kind, axis_index, side, cells):
                out.append(cells)
    random.shuffle(out)
    return out


def _orientation_color(orient: str) -> int:
    return MIRROR_NW_SE if orient == NW_SE else MIRROR_NE_SW


def _select_side_spec(minority: str) -> dict[str, tuple[int, ...]]:
    if not _SIDE_SPEC_CACHE:
        cache = {
            NW_SE: defaultdict(list),
            NE_SW: defaultdict(list),
        }
        choices = [2, 3, 4, 5]
        for nw_count in (1, 2):
            for ne_count in (1, 2):
                for nw_lengths in np.array(np.meshgrid(*([choices] * nw_count))).T.reshape(-1, nw_count):
                    nw = tuple(int(x) for x in nw_lengths)
                    for ne_lengths in np.array(np.meshgrid(*([choices] * ne_count))).T.reshape(-1, ne_count):
                        ne = tuple(int(x) for x in ne_lengths)
                        pixel_totals = {NW_SE: sum(nw), NE_SW: sum(ne)}
                        seg_counts = {NW_SE: len(nw), NE_SW: len(ne)}
                        picked = _minority_orientation(pixel_totals, seg_counts)
                        if pixel_totals[NW_SE] != pixel_totals[NE_SW]:
                            case = "pixel_less"
                        elif seg_counts[NW_SE] != seg_counts[NE_SW]:
                            case = "pixel_tie_count_less"
                        else:
                            case = "full_tie"
                        cache[picked][case].append({NW_SE: nw, NE_SW: ne})
        _SIDE_SPEC_CACHE.update({k: dict(v) for k, v in cache.items()})

    cases = _SIDE_SPEC_CACHE[minority]
    weights = []
    labels = []
    for label in ("pixel_less", "pixel_tie_count_less", "full_tie"):
        if label not in cases or not cases[label]:
            continue
        weight = {"pixel_less": 4, "pixel_tie_count_less": 3, "full_tie": 2}[label]
        weights.append(weight)
        labels.append(label)
    case = random.choices(labels, weights=weights, k=1)[0]
    return random.choice(cases[case])


def _build_expected_output(
    inp: np.ndarray, axis: tuple[str, int], placed: list[dict[str, object]]
) -> np.ndarray:
    out = inp.copy()
    for rec in placed:
        if not rec["reflect"]:
            continue
        paint = _orientation_color(rec["orientation"])
        for r, c in rec["cells"]:
            rr, cc = _reflect(axis[0], axis[1], r, c)
            if out[rr, cc] == BG:
                out[rr, cc] = paint
    return out


def _validate_generated(
    inp: np.ndarray,
    expected: np.ndarray,
    intended_minority: dict[int, str],
) -> bool:
    try:
        axis, _records, stats = _parse_scene(inp)
    except AssertionError:
        return False
    derived = {
        side: _minority_orientation(stats[side]["pixels"], stats[side]["segments"])
        for side in SIDES
    }
    if derived != intended_minority:
        return False
    if derived[0] == derived[1]:
        return False
    out = main(inp)
    if not np.array_equal(out, expected):
        return False
    if not np.any(out == MIRROR_NW_SE) or not np.any(out == MIRROR_NE_SW):
        return False
    if axis[0] == "row" and not np.all(out[axis[1], :] == AXIS):
        return False
    if axis[0] == "col" and not np.all(out[:, axis[1]] == AXIS):
        return False
    return True


def _attempt() -> np.ndarray | None:
    axis_kind = random.choice(("row", "col"))
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    if axis_kind == "row":
        if h % 2 == 0:
            h += 1
        axis_index = h // 2
    else:
        if w % 2 == 0:
            w += 1
        axis_index = w // 2
    if max(h, w) > 30:
        return None

    axis = (axis_kind, axis_index)
    grid = np.full((h, w), BG, dtype=int)
    if axis_kind == "row":
        grid[axis_index, :] = AXIS
        axis_cells = [(axis_index, c) for c in range(w)]
    else:
        grid[:, axis_index] = AXIS
        axis_cells = [(r, axis_index) for r in range(h)]

    blocked = np.zeros_like(grid, dtype=bool)
    _mark_buffer(blocked, axis_cells)

    minority = {0: random.choice(ORIENTATIONS)}
    minority[1] = NE_SW if minority[0] == NW_SE else NW_SE
    specs = {side: _select_side_spec(minority[side]) for side in SIDES}

    requests: list[dict[str, object]] = []
    for side in SIDES:
        for orient in ORIENTATIONS:
            for length in specs[side][orient]:
                requests.append(
                    {
                        "side": side,
                        "orientation": orient,
                        "length": int(length),
                        "reflect": orient == minority[side],
                    }
                )
    random.shuffle(requests)
    requests.sort(key=lambda rec: (not rec["reflect"], -int(rec["length"])))

    placed: list[dict[str, object]] = []
    for req in requests:
        candidates = _enumerate_candidates(
            grid.shape,
            axis_kind,
            axis_index,
            int(req["side"]),
            str(req["orientation"]),
            int(req["length"]),
        )
        chosen = None
        chosen_mirror = None
        for cells in candidates:
            if any(blocked[r, c] for r, c in cells):
                continue
            if req["reflect"]:
                mirrored = [_reflect(axis_kind, axis_index, r, c) for r, c in cells]
                if any(blocked[r, c] for r, c in mirrored):
                    continue
                chosen_mirror = mirrored
            chosen = cells
            break
        if chosen is None:
            return None

        for r, c in chosen:
            grid[r, c] = SOURCE
        _mark_buffer(blocked, chosen)
        if chosen_mirror is not None:
            _mark_buffer(blocked, chosen_mirror)
        placed.append(
            {
                "side": int(req["side"]),
                "orientation": str(req["orientation"]),
                "cells": chosen,
                "reflect": bool(req["reflect"]),
            }
        )

    expected = _build_expected_output(grid, axis, placed)
    if not _validate_generated(grid, expected, minority):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
