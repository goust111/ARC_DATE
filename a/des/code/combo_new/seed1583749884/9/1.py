from __future__ import annotations

import random

import numpy as np


_DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_axis(grid: np.ndarray, bg: int) -> tuple[str, int, int]:
    h, w = grid.shape
    candidates: list[tuple[str, int, int]] = []

    for r in range(h):
        color = int(grid[r, 0])
        if color != bg and np.all(grid[r] == color):
            candidates.append(("row", r, color))

    for c in range(w):
        color = int(grid[0, c])
        if color != bg and np.all(grid[:, c] == color):
            candidates.append(("col", c, color))

    assert len(candidates) == 1, "expected exactly one full horizontal or vertical axis"
    return candidates[0]


def _is_axis_cell(axis_kind: str, axis_index: int, r: int, c: int) -> bool:
    return r == axis_index if axis_kind == "row" else c == axis_index


def _component_dict(coords: list[tuple[int, int]], color: int) -> dict:
    arr = np.array(coords, dtype=int)
    rows = arr[:, 0]
    cols = arr[:, 1]
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())
    return {
        "color": int(color),
        "area": int(len(coords)),
        "coords": arr,
        "r0": r0,
        "r1": r1,
        "c0": c0,
        "c1": c1,
        "center2_row": int(r0 + r1),
        "center2_col": int(c0 + c1),
    }


def _extract_components(grid: np.ndarray, bg: int, axis_kind: str, axis_index: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or _is_axis_cell(axis_kind, axis_index, r, c):
                continue
            color = int(grid[r, c])
            if color == bg:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in _DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or _is_axis_cell(axis_kind, axis_index, nr, nc):
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            comps.append(_component_dict(coords, color))

    assert comps, "expected mirrored foreground objects on both sides of the axis"
    return comps


def _pair_components(comps: list[dict], axis_kind: str, axis_index: int) -> list[tuple[dict, dict]]:
    first_side: list[dict] = []
    second_side: list[dict] = []

    if axis_kind == "row":
        for comp in comps:
            if comp["r1"] < axis_index:
                first_side.append(comp)
            elif comp["r0"] > axis_index:
                second_side.append(comp)
            else:
                raise AssertionError("objects may not cross or touch the symmetry axis")

        lookup: dict[tuple[int, int], dict] = {}
        for comp in second_side:
            key = (comp["center2_row"], comp["center2_col"])
            assert key not in lookup, "mirrored partner centers must be unique"
            lookup[key] = comp

        pairs: list[tuple[dict, dict]] = []
        for comp in first_side:
            target = (4 * axis_index - comp["center2_row"], comp["center2_col"])
            assert target in lookup, "each upper object must have one mirrored lower partner"
            pairs.append((comp, lookup.pop(target)))
        assert not lookup and len(first_side) == len(second_side), "pairing must be bijective"
        return pairs

    for comp in comps:
        if comp["c1"] < axis_index:
            first_side.append(comp)
        elif comp["c0"] > axis_index:
            second_side.append(comp)
        else:
            raise AssertionError("objects may not cross or touch the symmetry axis")

    lookup = {}
    for comp in second_side:
        key = (comp["center2_row"], comp["center2_col"])
        assert key not in lookup, "mirrored partner centers must be unique"
        lookup[key] = comp

    pairs = []
    for comp in first_side:
        target = (comp["center2_row"], 4 * axis_index - comp["center2_col"])
        assert target in lookup, "each left object must have one mirrored right partner"
        pairs.append((comp, lookup.pop(target)))
    assert not lookup and len(first_side) == len(second_side), "pairing must be bijective"
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the larger mirrored object but repaint it with the smaller partner's color.

    Input structure:
    - The background is black.
    - A full grey horizontal or vertical line divides the grid into two mirrored sides.
    - Each side contains several 4-connected monochrome objects, and every object has one
      partner whose bbox center is mirrored across the axis.

    Rule:
    - Pair objects by mirrored bbox-center position across the axis.
    - If one object has larger area, keep its shape and position but recolor it with the
      smaller partner's color, then erase the smaller partner to background.
    - If both areas are equal, keep the upper or left object unchanged and erase the
      lower or right object.
    - Leave the axis line unchanged.

    Key insight:
    - Pairing is determined by geometric mirror position, but the surviving object's shape
      and final color come from different members of the pair.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    axis_kind, axis_index, _axis_color = _find_axis(g, bg)
    comps = _extract_components(g, bg, axis_kind, axis_index)
    pairs = _pair_components(comps, axis_kind, axis_index)

    out = g.copy()
    for first, second in pairs:
        if first["area"] == second["area"]:
            out[second["coords"][:, 0], second["coords"][:, 1]] = bg
            continue

        winner, loser = (first, second) if first["area"] > second["area"] else (second, first)
        out[loser["coords"][:, 0], loser["coords"][:, 1]] = bg
        out[winner["coords"][:, 0], winner["coords"][:, 1]] = int(loser["color"])

    return out


# --------------- generator ---------------

_GEN_BG = 0
_GEN_AXIS_COLOR = 5
_GEN_OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
_GEN_BRANCHES = ("first_bigger", "second_bigger", "tie")
_GEN_PAIR_COUNT_RANGE = (1, 4)
_GEN_MIN_SIZE = 15
_GEN_MAX_SIZE = 29
_GEN_MIN_BBOX = 2
_GEN_MAX_BBOX = 6
_GEN_AXIS_GAP = 1
_GEN_BBOX_PAD = 1
_GEN_ATTEMPTS = 400


def _rand_odd(lo: int, hi: int) -> int:
    values = [value for value in range(lo, hi + 1) if value % 2 == 1]
    return int(random.choice(values))


def _seed_path_cells(h: int, w: int) -> set[tuple[int, int]]:
    start_col = 0 if random.choice([False, True]) else w - 1
    horiz_step = 1 if start_col == 0 else -1

    r = 0
    c = start_col
    cells = {(r, c)}
    steps = [(1, 0)] * (h - 1) + [(0, horiz_step)] * (w - 1)
    random.shuffle(steps)
    for dr, dc in steps:
        r += dr
        c += dc
        cells.add((r, c))

    if random.choice([False, True]):
        cells = {(h - 1 - rr, cc) for rr, cc in cells}
    if random.choice([False, True]):
        cells = {(rr, w - 1 - cc) for rr, cc in cells}
    return cells


def _make_mask(h: int, w: int, area: int) -> np.ndarray:
    cells = _seed_path_cells(h, w)
    while len(cells) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in _DIRS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(random.choice(tuple(frontier)))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _sample_pair_masks(axis_kind: str, grid_shape: tuple[int, int], branch: str) -> tuple[np.ndarray, np.ndarray]:
    h_total, w_total = grid_shape
    axis_index = (h_total // 2) if axis_kind == "row" else (w_total // 2)
    max_h = min(_GEN_MAX_BBOX, axis_index - _GEN_AXIS_GAP) if axis_kind == "row" else min(_GEN_MAX_BBOX, h_total)
    max_w = min(_GEN_MAX_BBOX, axis_index - _GEN_AXIS_GAP) if axis_kind == "col" else min(_GEN_MAX_BBOX, w_total)
    assert max_h >= _GEN_MIN_BBOX and max_w >= _GEN_MIN_BBOX

    h = random.randint(_GEN_MIN_BBOX, max_h)
    w = random.randint(_GEN_MIN_BBOX, max_w)
    min_area = h + w - 1
    max_area = h * w

    if branch == "tie":
        area = random.randint(min_area, max_area)
        return _make_mask(h, w, area), _make_mask(h, w, area)

    small = random.randint(min_area, max_area - 1)
    big = random.randint(small + 1, max_area)
    if branch == "first_bigger":
        return _make_mask(h, w, big), _make_mask(h, w, small)
    return _make_mask(h, w, small), _make_mask(h, w, big)


def _bbox_clear(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - _GEN_BBOX_PAD)
    rr1 = min(blocked.shape[0], r0 + h + _GEN_BBOX_PAD)
    cc0 = max(0, c0 - _GEN_BBOX_PAD)
    cc1 = min(blocked.shape[1], c0 + w + _GEN_BBOX_PAD)
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _mark_bbox(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - _GEN_BBOX_PAD)
    rr1 = min(blocked.shape[0], r0 + h + _GEN_BBOX_PAD)
    cc0 = max(0, c0 - _GEN_BBOX_PAD)
    cc1 = min(blocked.shape[1], c0 + w + _GEN_BBOX_PAD)
    blocked[rr0:rr1, cc0:cc1] = True


def _candidate_origins(
    blocked: np.ndarray, axis_kind: str, axis_index: int, h: int, w: int
) -> list[tuple[int, int]]:
    h_total, w_total = blocked.shape
    origins: list[tuple[int, int]] = []

    if axis_kind == "row":
        max_r0 = axis_index - _GEN_AXIS_GAP - h
        if max_r0 < 0:
            return origins
        for r0 in range(max_r0 + 1):
            for c0 in range(w_total - w + 1):
                if _bbox_clear(blocked, r0, c0, h, w):
                    origins.append((int(r0), int(c0)))
        return origins

    max_c0 = axis_index - _GEN_AXIS_GAP - w
    if max_c0 < 0:
        return origins
    for r0 in range(h_total - h + 1):
        for c0 in range(max_c0 + 1):
            if _bbox_clear(blocked, r0, c0, h, w):
                origins.append((int(r0), int(c0)))
    return origins


def _place_specs(axis_kind: str, grid_shape: tuple[int, int], specs: list[dict]) -> list[tuple[int, int]] | None:
    h_total, w_total = grid_shape
    blocked = np.zeros((h_total, w_total), dtype=bool)
    axis_index = (h_total // 2) if axis_kind == "row" else (w_total // 2)
    placements: list[tuple[int, int] | None] = [None] * len(specs)

    order = sorted(range(len(specs)), key=lambda idx: int(specs[idx]["mask_first"].size), reverse=True)
    for idx in order:
        mask = specs[idx]["mask_first"]
        h, w = mask.shape
        origins = _candidate_origins(blocked, axis_kind, axis_index, int(h), int(w))
        if not origins:
            return None
        r0, c0 = random.choice(origins)
        placements[idx] = (int(r0), int(c0))
        _mark_bbox(blocked, int(r0), int(c0), int(h), int(w))

    return [(int(r0), int(c0)) for r0, c0 in placements if r0 is not None and c0 is not None]


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    coords = np.argwhere(mask)
    rows = coords[:, 0] + int(r0)
    cols = coords[:, 1] + int(c0)
    grid[rows, cols] = int(color)


def _generate_attempt() -> np.ndarray | None:
    pair_count = random.randint(*_GEN_PAIR_COUNT_RANGE)
    base_size = _GEN_MIN_SIZE + 2 * (pair_count - 1)
    axis_kind = random.choice(["row", "col"])

    if axis_kind == "row":
        h = _rand_odd(base_size, _GEN_MAX_SIZE)
        w = random.randint(base_size, _GEN_MAX_SIZE)
    else:
        h = random.randint(base_size, _GEN_MAX_SIZE)
        w = _rand_odd(base_size, _GEN_MAX_SIZE)

    colors = random.sample(_GEN_OBJECT_COLORS, 2 * pair_count)
    specs: list[dict] = []
    for pair_index in range(pair_count):
        branch = random.choice(_GEN_BRANCHES)
        mask_first, mask_second = _sample_pair_masks(axis_kind, (h, w), branch)
        specs.append(
            {
                "branch": branch,
                "mask_first": mask_first,
                "mask_second": mask_second,
                "color_first": int(colors[2 * pair_index]),
                "color_second": int(colors[2 * pair_index + 1]),
            }
        )

    placements = _place_specs(axis_kind, (h, w), specs)
    if placements is None or len(placements) != len(specs):
        return None

    grid = np.full((h, w), _GEN_BG, dtype=int)
    if axis_kind == "row":
        axis_index = h // 2
        grid[axis_index, :] = _GEN_AXIS_COLOR
    else:
        axis_index = w // 2
        grid[:, axis_index] = _GEN_AXIS_COLOR

    for spec, (r0, c0) in zip(specs, placements):
        mask_first = spec["mask_first"]
        mask_second = spec["mask_second"]
        h_box, w_box = mask_first.shape
        if axis_kind == "row":
            second_r0 = h - h_box - r0
            second_c0 = c0
        else:
            second_r0 = r0
            second_c0 = w - w_box - c0

        _paint_mask(grid, mask_first, int(r0), int(c0), int(spec["color_first"]))
        _paint_mask(grid, mask_second, int(second_r0), int(second_c0), int(spec["color_second"]))

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-pair puzzle")
