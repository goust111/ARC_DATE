from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG_COLOR = 7
PAIR_COLORS = [c for c in range(10) if c != BG_COLOR]
ORIENTATIONS = ("H", "V")
LINE_LENGTHS = (1, 2, 3, 4, 5, 6)
GRID_HEIGHT_RANGE = (12, 22)
GRID_WIDTH_RANGE = (12, 22)
PAIR_COUNT_RANGE = (1, 4)
GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 200

# Base templates use anchor (0, 0) on the attachment side.
RIGHT_ATTACH_TEMPLATES = [
    {(0, 0), (-1, 0), (1, 0), (0, -1)},
    {(0, 0), (-1, 0), (1, 0), (0, -1), (-1, -1)},
    {(0, 0), (-1, 0), (1, 0), (0, -1), (1, -1)},
    {(0, 0), (-1, 0), (1, 0), (-1, -1), (0, -1), (1, -1)},
    {(0, 0), (-1, 0), (1, 0), (0, -1), (0, -2)},
    {(0, 0), (-1, 0), (1, 0), (-1, -1), (0, -1), (-1, -2)},
    {(0, 0), (-1, 0), (1, 0), (-1, -1), (0, -1), (0, -2)},
    {(0, 0), (-1, 0), (1, 0), (0, -1), (1, -1), (0, -2)},
]


@dataclass(frozen=True)
class PairSpec:
    color: int
    input_coords: np.ndarray
    output_coords: np.ndarray
    height: int
    width: int


@dataclass(frozen=True)
class BridgeCandidate:
    orientation: str
    segment: np.ndarray
    comp_a: np.ndarray
    comp_b: np.ndarray


def _dominant_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.min(coords[:, 0])),
        int(np.min(coords[:, 1])),
        int(np.max(coords[:, 0])),
        int(np.max(coords[:, 1])),
    )


def _has_vertical_neighbor(mask: np.ndarray, r: int, c: int) -> bool:
    return (r > 0 and mask[r - 1, c]) or (r + 1 < mask.shape[0] and mask[r + 1, c])


def _has_horizontal_neighbor(mask: np.ndarray, r: int, c: int) -> bool:
    return (c > 0 and mask[r, c - 1]) or (c + 1 < mask.shape[1] and mask[r, c + 1])


def _component_lookup(comps: list[np.ndarray]) -> dict[tuple[int, int], int]:
    lookup: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(comps):
        for r, c in comp:
            lookup[(int(r), int(c))] = idx
    return lookup


def _validate_horizontal(mask: np.ndarray, r: int, start: int, end: int) -> BridgeCandidate | None:
    w = mask.shape[1]
    if start == 0 or end == w - 1 or not mask[r, start - 1] or not mask[r, end + 1]:
        return None
    rem = mask.copy()
    rem[r, start : end + 1] = False
    comps = _components(rem)
    if len(comps) != 2 or min(len(comp) for comp in comps) < 2:
        return None
    lookup = _component_lookup(comps)
    left_idx = lookup.get((r, start - 1))
    right_idx = lookup.get((r, end + 1))
    if left_idx is None or right_idx is None or left_idx == right_idx:
        return None
    segment = np.array([(r, c) for c in range(start, end + 1)], dtype=int)
    return BridgeCandidate("H", segment, comps[left_idx], comps[right_idx])


def _validate_vertical(mask: np.ndarray, c: int, start: int, end: int) -> BridgeCandidate | None:
    h = mask.shape[0]
    if start == 0 or end == h - 1 or not mask[start - 1, c] or not mask[end + 1, c]:
        return None
    rem = mask.copy()
    rem[start : end + 1, c] = False
    comps = _components(rem)
    if len(comps) != 2 or min(len(comp) for comp in comps) < 2:
        return None
    lookup = _component_lookup(comps)
    top_idx = lookup.get((start - 1, c))
    bottom_idx = lookup.get((end + 1, c))
    if top_idx is None or bottom_idx is None or top_idx == bottom_idx:
        return None
    segment = np.array([(r, c) for r in range(start, end + 1)], dtype=int)
    return BridgeCandidate("V", segment, comps[top_idx], comps[bottom_idx])


def _find_bridge(mask: np.ndarray) -> BridgeCandidate:
    h, w = mask.shape
    candidates: list[BridgeCandidate] = []

    for r in range(h):
        c = 0
        while c < w:
            if mask[r, c] and not _has_vertical_neighbor(mask, r, c):
                start = c
                while c + 1 < w and mask[r, c + 1] and not _has_vertical_neighbor(mask, r, c + 1):
                    c += 1
                end = c
                candidate = _validate_horizontal(mask, r, start, end)
                if candidate is not None:
                    candidates.append(candidate)
            c += 1

    for c in range(w):
        r = 0
        while r < h:
            if mask[r, c] and not _has_horizontal_neighbor(mask, r, c):
                start = r
                while r + 1 < h and mask[r + 1, c] and not _has_horizontal_neighbor(mask, r + 1, c):
                    r += 1
                end = r
                candidate = _validate_vertical(mask, c, start, end)
                if candidate is not None:
                    candidates.append(candidate)
            r += 1

    assert candidates, "expected one bridge line per color"
    candidates.sort(key=lambda cand: (len(cand.segment), min(len(cand.comp_a), len(cand.comp_b))), reverse=True)
    if len(candidates) > 1:
        top = (len(candidates[0].segment), min(len(candidates[0].comp_a), len(candidates[0].comp_b)))
        nxt = (len(candidates[1].segment), min(len(candidates[1].comp_a), len(candidates[1].comp_b)))
        assert top != nxt, "ambiguous bridge candidate"
    return candidates[0]


def _translate(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return coords + np.array([dr, dc], dtype=int)


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    out = np.full_like(g, bg)
    occupied = np.zeros_like(g, dtype=bool)

    for color in sorted(int(c) for c in np.unique(g) if c != bg):
        mask = g == color
        comps = _components(mask)
        assert len(comps) == 1, "each pair color must form one connected structure"
        bridge = _find_bridge(mask)

        if bridge.orientation == "H":
            left, right = bridge.comp_a, bridge.comp_b
            if _bbox(left)[1] > _bbox(right)[1]:
                left, right = right, left
            gap = len(bridge.segment)
            left_move = (gap + 1) // 2
            right_move = gap // 2
            moved_groups = [
                _translate(left, 0, left_move),
                _translate(right, 0, -right_move),
            ]
        else:
            top, bottom = bridge.comp_a, bridge.comp_b
            if _bbox(top)[0] > _bbox(bottom)[0]:
                top, bottom = bottom, top
            gap = len(bridge.segment)
            top_move = (gap + 1) // 2
            bottom_move = gap // 2
            moved_groups = [
                _translate(top, top_move, 0),
                _translate(bottom, -bottom_move, 0),
            ]

        for moved in moved_groups:
            assert np.all((0 <= moved[:, 0]) & (moved[:, 0] < g.shape[0]))
            assert np.all((0 <= moved[:, 1]) & (moved[:, 1] < g.shape[1]))
            rows = moved[:, 0]
            cols = moved[:, 1]
            assert not np.any(occupied[rows, cols]), "moved objects must not overlap"
            occupied[rows, cols] = True
            out[rows, cols] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid)


def _transform_cells(cells: set[tuple[int, int]], fn) -> set[tuple[int, int]]:
    return {fn(r, c) for r, c in cells}


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = np.array(sorted((r - min_r, c - min_c) for r, c in cells), dtype=int)
    anchor = (-min_r, -min_c)
    return norm, anchor


def _is_4_connected(coords: np.ndarray) -> bool:
    cells = {tuple(map(int, rc)) for rc in coords}
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _shape_for_side(side: str) -> tuple[np.ndarray, tuple[int, int]]:
    cells = set(random.choice(RIGHT_ATTACH_TEMPLATES))

    if side == "left":
        cells = _transform_cells(cells, lambda r, c: (r, -c))
    elif side == "down":
        cells = _transform_cells(cells, lambda r, c: (c, -r))
    elif side == "up":
        cells = _transform_cells(cells, lambda r, c: (-c, r))
    elif side != "right":
        raise ValueError(f"unknown side: {side}")

    if side in {"right", "left"} and random.random() < 0.5:
        cells = _transform_cells(cells, lambda r, c: (-r, c))
    if side in {"up", "down"} and random.random() < 0.5:
        cells = _transform_cells(cells, lambda r, c: (r, -c))

    coords, anchor = _normalize_shape(cells)
    assert _is_4_connected(coords), "object template must stay 4-connected after transforms"
    if side == "right":
        assert anchor[1] == int(np.max(coords[:, 1]))
    elif side == "left":
        assert anchor[1] == int(np.min(coords[:, 1]))
    elif side == "down":
        assert anchor[0] == int(np.max(coords[:, 0]))
    else:
        assert side == "up"
        assert anchor[0] == int(np.min(coords[:, 0]))
    return coords, anchor


def _build_horizontal_pair(color: int) -> PairSpec:
    left_cells, left_anchor = _shape_for_side("right")
    right_cells, right_anchor = _shape_for_side("left")
    gap = random.choice(LINE_LENGTHS)

    anchor_r = max(left_anchor[0], right_anchor[0])
    left_offset = np.array([anchor_r - left_anchor[0], 0], dtype=int)
    left_cells = left_cells + left_offset
    left_anchor_col = left_anchor[1]

    right_col = left_anchor_col + gap + 1
    right_offset = np.array([anchor_r - right_anchor[0], right_col - right_anchor[1]], dtype=int)
    right_cells = right_cells + right_offset

    line = np.array([(anchor_r, c) for c in range(left_anchor_col + 1, right_col)], dtype=int)
    input_coords = np.vstack([left_cells, line, right_cells])

    out_left = _translate(left_cells, 0, (gap + 1) // 2)
    out_right = _translate(right_cells, 0, -(gap // 2))
    output_coords = np.vstack([out_left, out_right])

    height = int(np.max(input_coords[:, 0])) + 1
    width = int(np.max(input_coords[:, 1])) + 1
    return PairSpec(color, input_coords, output_coords, height, width)


def _build_vertical_pair(color: int) -> PairSpec:
    top_cells, top_anchor = _shape_for_side("down")
    bottom_cells, bottom_anchor = _shape_for_side("up")
    gap = random.choice(LINE_LENGTHS)

    anchor_c = max(top_anchor[1], bottom_anchor[1])
    top_offset = np.array([0, anchor_c - top_anchor[1]], dtype=int)
    top_cells = top_cells + top_offset
    top_anchor_row = top_anchor[0]

    bottom_row = top_anchor_row + gap + 1
    bottom_offset = np.array([bottom_row - bottom_anchor[0], anchor_c - bottom_anchor[1]], dtype=int)
    bottom_cells = bottom_cells + bottom_offset

    line = np.array([(r, anchor_c) for r in range(top_anchor_row + 1, bottom_row)], dtype=int)
    input_coords = np.vstack([top_cells, line, bottom_cells])

    out_top = _translate(top_cells, (gap + 1) // 2, 0)
    out_bottom = _translate(bottom_cells, -(gap // 2), 0)
    output_coords = np.vstack([out_top, out_bottom])

    height = int(np.max(input_coords[:, 0])) + 1
    width = int(np.max(input_coords[:, 1])) + 1
    return PairSpec(color, input_coords, output_coords, height, width)


def _build_pair(color: int, orientation: str) -> PairSpec:
    if orientation == "H":
        return _build_horizontal_pair(color)
    if orientation == "V":
        return _build_vertical_pair(color)
    raise ValueError(f"unknown orientation: {orientation}")


def _rect_clear(used: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(used.shape[0] - 1, r1 + pad)
    cc1 = min(used.shape[1] - 1, c1 + pad)
    return not np.any(used[rr0 : rr1 + 1, cc0 : cc1 + 1])


def _paint(grid: np.ndarray, coords: np.ndarray, color: int, offset_r: int, offset_c: int) -> None:
    shifted = coords + np.array([offset_r, offset_c], dtype=int)
    rows = shifted[:, 0]
    cols = shifted[:, 1]
    assert np.all((0 <= rows) & (rows < grid.shape[0]))
    assert np.all((0 <= cols) & (cols < grid.shape[1]))
    assert np.all(grid[rows, cols] == BG_COLOR)
    grid[rows, cols] = color


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    if pair_count > len(PAIR_COLORS):
        return None

    colors = random.sample(PAIR_COLORS, pair_count)
    orientations = [random.choice(ORIENTATIONS) for _ in range(pair_count)]
    specs = [_build_pair(color, orientation) for color, orientation in zip(colors, orientations)]
    specs.sort(key=lambda spec: spec.height * spec.width, reverse=True)

    used = np.zeros((height, width), dtype=bool)
    placements: list[tuple[PairSpec, int, int]] = []
    for spec in specs:
        if spec.height > height or spec.width > width:
            return None
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(0, height - spec.height)
            c0 = random.randint(0, width - spec.width)
            r1 = r0 + spec.height - 1
            c1 = c0 + spec.width - 1
            if not _rect_clear(used, r0, c0, r1, c1, pad=1):
                continue
            used[r0 : r1 + 1, c0 : c1 + 1] = True
            placements.append((spec, r0, c0))
            placed = True
            break
        if not placed:
            return None

    grid = np.full((height, width), BG_COLOR, dtype=int)
    expected = np.full((height, width), BG_COLOR, dtype=int)
    for spec, r0, c0 in placements:
        _paint(grid, spec.input_coords, spec.color, r0, c0)
        _paint(expected, spec.output_coords, spec.color, r0, c0)

    if np.count_nonzero(grid != BG_COLOR) * 2 >= grid.size:
        return None
    if not np.array_equal(_solve(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
