from __future__ import annotations

import random
from functools import lru_cache

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
MAROON = 9

PANEL_SIZE = 11
INPUT_SHAPE = (PANEL_SIZE, PANEL_SIZE * 2 + 1)
MIN_REGION_SIDE = 2
GENERATION_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                r0 = min(r0, rr)
                c0 = min(c0, cc)
                r1 = max(r1, rr)
                c1 = max(c1, cc)
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(
                {
                    "coords": np.array(coords, dtype=int),
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )
    return comps


def _separator_column(grid: np.ndarray) -> int:
    h, w = grid.shape
    cols = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    assert cols, "expected a full-height gray separator column"
    valid = [c for c in cols if c > 0 and c < w - 1 and c == w - c - 1]
    assert len(valid) == 1, "expected exactly one centered separator column"
    return valid[0]


def _render_ranked_components(mask: np.ndarray) -> np.ndarray:
    comps = _components(mask)
    ranked = sorted(comps, key=lambda comp: (-int(comp["area"]), *comp["bbox"]))
    out = np.zeros(mask.shape, dtype=int)
    for rank, comp in enumerate(ranked):
        if rank == 0:
            color = GREEN
        elif rank == 1:
            color = YELLOW
        elif rank == 2:
            color = ORANGE
        else:
            color = MAROON
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the 4-connected components of the two-panel double-empty intersection by descending area.

    Input structure:
    - The grid contains two equal-size square panels separated by one full-height gray column.
    - The left panel uses black background with red 4-connected foreground shapes.
    - The right panel uses black background with blue 4-connected foreground shapes.

    Rule:
    - Remove the gray separator column and align the two panels cell by cell.
    - Mark a cell as double-empty exactly when the aligned left and right cells are both black.
    - Split the double-empty mask into 4-connected components and sort them by area from largest to smallest.
    - Paint ranks 0, 1, 2 with green, yellow, orange, and paint every later rank maroon; every non-double-empty cell becomes black.

    Key insight:
    - The result is not the raw empty-space intersection alone: the intersection must be decomposed into separate 4-connected regions, then those regions are recolored by area rank.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _separator_column(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape == right.shape

    double_empty = (left == BLACK) & (right == BLACK)
    return _render_ranked_components(double_empty)


def _apply_shared_symmetry(*masks: np.ndarray) -> tuple[np.ndarray, ...]:
    out = [mask.copy() for mask in masks]
    if random.random() < 0.5:
        out = [mask.T.copy() for mask in out]
    if random.random() < 0.5:
        out = [np.flipud(mask).copy() for mask in out]
    if random.random() < 0.5:
        out = [np.fliplr(mask).copy() for mask in out]
    return tuple(out)


def _partition_empty_regions() -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    for _ in range(200):
        target_regions = random.choices([3, 4, 5], weights=[1, 3, 2])[0]
        regions = [(0, PANEL_SIZE, 0, PANEL_SIZE)]
        segments: list[list[tuple[int, int]]] = []

        while len(regions) < target_regions:
            candidates: list[tuple[int, str]] = []
            for idx, (r0, r1, c0, c1) in enumerate(regions):
                if r1 - r0 >= 2 * MIN_REGION_SIDE + 1:
                    candidates.append((idx, "h"))
                if c1 - c0 >= 2 * MIN_REGION_SIDE + 1:
                    candidates.append((idx, "v"))
            if not candidates:
                break

            idx, orient = random.choice(candidates)
            r0, r1, c0, c1 = regions.pop(idx)
            if orient == "h":
                split_r = random.randint(r0 + MIN_REGION_SIDE, r1 - MIN_REGION_SIDE - 1)
                segments.append([(split_r, c) for c in range(c0, c1)])
                regions.append((r0, split_r, c0, c1))
                regions.append((split_r + 1, r1, c0, c1))
            else:
                split_c = random.randint(c0 + MIN_REGION_SIDE, c1 - MIN_REGION_SIDE - 1)
                segments.append([(r, split_c) for r in range(r0, r1)])
                regions.append((r0, r1, c0, split_c))
                regions.append((r0, r1, split_c + 1, c1))

        if len(regions) != target_regions:
            continue

        empty_mask = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)
        areas = []
        for r0, r1, c0, c1 in regions:
            area = (r1 - r0) * (c1 - c0)
            areas.append(area)
            empty_mask[r0:r1, c0:c1] = True

        if len(set(areas)) != len(areas):
            continue
        if len(_components(empty_mask)) != target_regions:
            continue

        return empty_mask, segments

    raise RuntimeError("failed to partition empty regions")


@lru_cache(maxsize=None)
def _chunk_partitions(length: int) -> tuple[tuple[int, ...], ...]:
    if length == 0:
        return ((),)
    out = []
    for size in (2, 3, 4):
        if size > length:
            continue
        for tail in _chunk_partitions(length - size):
            out.append((size,) + tail)
    return tuple(out)


def _segment_options(cells: list[tuple[int, int]]) -> list[np.ndarray]:
    options: list[np.ndarray] = []
    for parts in _chunk_partitions(len(cells)):
        if not parts:
            continue
        for start_left in (False, True):
            choice = np.zeros(len(cells), dtype=bool)
            pos = 0
            is_left = start_left
            for part in parts:
                choice[pos : pos + part] = is_left
                pos += part
                is_left = not is_left
            options.append(choice)
    random.shuffle(options)
    return options


def _side_component_areas(mask: np.ndarray) -> list[int]:
    return [int(comp["area"]) for comp in _components(mask)]


def _valid_side_masks(left_mask: np.ndarray, right_mask: np.ndarray, empty_mask: np.ndarray) -> bool:
    if np.any(left_mask & right_mask):
        return False
    if not np.array_equal(~(left_mask | right_mask), empty_mask):
        return False
    if np.array_equal(~left_mask, empty_mask) or np.array_equal(~right_mask, empty_mask):
        return False

    left_areas = _side_component_areas(left_mask)
    right_areas = _side_component_areas(right_mask)
    if not left_areas or not right_areas:
        return False
    if not all(2 <= area <= 8 for area in left_areas + right_areas):
        return False
    return True


def _search_side_masks(empty_mask: np.ndarray, segments: list[list[tuple[int, int]]]) -> tuple[np.ndarray, np.ndarray] | None:
    barrier_mask = ~empty_mask
    options_per_segment = [_segment_options(cells) for cells in segments]
    order = list(range(len(segments)))
    order.sort(key=lambda idx: len(options_per_segment[idx]))
    chosen: dict[int, np.ndarray] = {}

    def build_masks() -> tuple[np.ndarray, np.ndarray]:
        left_mask = np.zeros_like(barrier_mask, dtype=bool)
        for seg_idx, cells in enumerate(segments):
            choice = chosen[seg_idx]
            for keep_left, (r, c) in zip(choice, cells):
                left_mask[r, c] = bool(keep_left)
        right_mask = barrier_mask & ~left_mask
        return left_mask, right_mask

    def backtrack(pos: int) -> tuple[np.ndarray, np.ndarray] | None:
        if pos == len(order):
            left_mask, right_mask = build_masks()
            if _valid_side_masks(left_mask, right_mask, empty_mask):
                return left_mask, right_mask
            return None

        seg_idx = order[pos]
        for option in options_per_segment[seg_idx]:
            chosen[seg_idx] = option
            found = backtrack(pos + 1)
            if found is not None:
                return found
        chosen.pop(seg_idx, None)
        return None

    return backtrack(0)


def _compose_input(left_mask: np.ndarray, right_mask: np.ndarray) -> np.ndarray:
    h, w = left_mask.shape
    inp = np.zeros((h, w * 2 + 1), dtype=int)
    inp[:, w] = GREY
    left_panel = inp[:, :w]
    right_panel = inp[:, w + 1 :]
    left_panel[left_mask] = RED
    right_panel[right_mask] = BLUE
    return inp


def _semantic_check(inp: np.ndarray, empty_mask: np.ndarray) -> None:
    assert inp.shape == INPUT_SHAPE
    out = main(inp)
    expected = _render_ranked_components(empty_mask)
    assert np.array_equal(out, expected)

    sep = _separator_column(inp)
    left = inp[:, :sep] == RED
    right = inp[:, sep + 1 :] == BLUE

    assert np.array_equal((~left) & (~right), empty_mask)
    comps = _components(empty_mask)
    areas = [int(comp["area"]) for comp in comps]
    assert len(comps) >= 3
    assert len(set(areas)) == len(areas)
    assert all(2 <= area <= 8 for area in _side_component_areas(left))
    assert all(2 <= area <= 8 for area in _side_component_areas(right))
    assert np.any(left) and np.any(right)
    assert not np.array_equal(~left, empty_mask)
    assert not np.array_equal(~right, empty_mask)

    if len(comps) >= 4:
        assert np.any(out == MAROON)
    else:
        assert not np.any(out == MAROON)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        empty_mask, segments = _partition_empty_regions()
        side_masks = _search_side_masks(empty_mask, segments)
        if side_masks is None:
            continue
        left_mask, right_mask = side_masks

        left_mask, right_mask = _apply_shared_symmetry(left_mask, right_mask)
        empty_mask = (~left_mask) & (~right_mask)

        inp = _compose_input(left_mask, right_mask)
        try:
            _semantic_check(inp, empty_mask)
        except AssertionError:
            continue
        return inp

    raise RuntimeError("failed to generate a valid input for puzzle 8")
