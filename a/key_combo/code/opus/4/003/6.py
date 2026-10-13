from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5

BODY_COLORS = {1: BLUE, 2: RED, 3: GREEN}

GRID_HEIGHT_RANGE = (13, 19)
GRID_WIDTH_RANGE = (13, 19)
MAX_GENERATION_ATTEMPTS = 400
OVERLAP_PROBABILITY = 0.72
CLIP_PROBABILITY = 0.28
MAX_OVERLAP_CELLS = 18


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows, cols = zip(*coords)
    return min(rows), min(cols), max(rows), max(cols)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bar_coords(spec: dict) -> list[tuple[int, int]]:
    if spec["orientation"] == "h":
        return [(spec["row"], c) for c in range(spec["col0"], spec["col0"] + spec["length"])]
    return [(r, spec["col"]) for r in range(spec["row0"], spec["row0"] + spec["length"])]


def _bar_bbox(spec: dict) -> tuple[int, int, int, int]:
    return _bbox_coords(_bar_coords(spec))


def _make_spec(orientation: str, anchor0: int, anchor1: int, length: int, rank: int) -> dict:
    if orientation == "h":
        spec = {"orientation": "h", "row": int(anchor0), "col0": int(anchor1)}
    else:
        spec = {"orientation": "v", "row0": int(anchor0), "col": int(anchor1)}
    spec["length"] = int(length)
    spec["rank"] = int(rank)
    spec["cells"] = _bar_coords(spec)
    spec["bbox"] = _bar_bbox(spec)
    return spec


def _bars_touch(spec_a: dict, spec_b: dict) -> bool:
    for ra, ca in spec_a["cells"]:
        for rb, cb in spec_b["cells"]:
            if max(abs(ra - rb), abs(ca - cb)) <= 1:
                return True
    return False


def _ring_cells(
    bbox: tuple[int, int, int, int], layer: int, shape: tuple[int, int]
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    rr0 = r0 - layer
    cc0 = c0 - layer
    rr1 = r1 + layer
    cc1 = c1 + layer
    height, width = shape
    cells: set[tuple[int, int]] = set()

    for c in range(cc0, cc1 + 1):
        if 0 <= rr0 < height and 0 <= c < width:
            cells.add((rr0, c))
        if 0 <= rr1 < height and 0 <= c < width:
            cells.add((rr1, c))
    for r in range(rr0 + 1, rr1):
        if 0 <= r < height and 0 <= cc0 < width:
            cells.add((r, cc0))
        if 0 <= r < height and 0 <= cc1 < width:
            cells.add((r, cc1))
    return cells


def _outline_cells(spec: dict, shape: tuple[int, int]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for layer in range(1, spec["rank"] + 1):
        cells |= _ring_cells(spec["bbox"], layer, shape)
    return cells


def _render_from_specs(specs: list[dict], shape: tuple[int, int], bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    body_mask = np.zeros(shape, dtype=bool)
    ranked = sorted(specs, key=lambda spec: spec["rank"])

    for spec in ranked:
        color = BODY_COLORS[spec["rank"]]
        for r, c in spec["cells"]:
            out[r, c] = color
            body_mask[r, c] = True

    for spec in ranked:
        for layer in range(1, spec["rank"] + 1):
            for r, c in _ring_cells(spec["bbox"], layer, shape):
                if not body_mask[r, c]:
                    out[r, c] = YELLOW
    return out


def _find_gray_bars(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    visited = np.zeros_like(g, dtype=bool)
    bars: list[dict] = []

    for r in range(height):
        for c in range(width):
            if g[r, c] != GRAY or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and not visited[nr, nc] and g[nr, nc] == GRAY:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            bbox = _bbox_coords(coords)
            r0, c0, r1, c1 = bbox
            area = len(coords)

            if r0 == r1:
                assert area == c1 - c0 + 1, "horizontal bar must be contiguous"
                assert sorted(c for _r, c in coords) == list(range(c0, c1 + 1))
                spec = _make_spec("h", r0, c0, area, rank=0)
            else:
                assert c0 == c1, "bar must be axis-aligned"
                assert area == r1 - r0 + 1, "vertical bar must be contiguous"
                assert sorted(r for r, _c in coords) == list(range(r0, r1 + 1))
                spec = _make_spec("v", r0, c0, area, rank=0)

            assert spec["bbox"][0] > 0 and spec["bbox"][1] > 0
            assert spec["bbox"][2] < height - 1 and spec["bbox"][3] < width - 1
            bars.append(spec)

    assert len(bars) == 3, "expected exactly three gray bars"
    lengths = [spec["length"] for spec in bars]
    assert len(set(lengths)) == 3, "bar lengths must be distinct"
    for i in range(3):
        for j in range(i + 1, 3):
            assert not _bars_touch(bars[i], bars[j]), "bars must not touch"
    return bars


def _rank_spec(spec: dict, rank: int) -> dict:
    return _make_spec(spec["orientation"], spec["bbox"][0], spec["bbox"][1], spec["length"], rank)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rank the three gray bars by length, recolor them, and draw rank-thick yellow outline rings around each bar.

    Input structure:
    - The background is black, and exactly three gray 1-pixel-thick bars appear on it.
    - Each gray object is a single horizontal or vertical straight segment, and the three lengths are all different.
    - The bars do not touch each other or the grid border.

    Rule:
    - Rank the bars from shortest to longest.
    - Recolor the bar bodies by rank: shortest to blue, middle to red, longest to green.
    - Around each bar, draw concentric rectangular yellow outline rings whose count equals that bar's rank; each ring is the next 1-cell border outside the previous one, including corners.
    - Never overwrite recolored bar-body cells with outline pixels; when outline regions from different bars overlap, the higher-rank bar is applied later and therefore has priority.
    - Discard any outline cells that would lie outside the grid.

    Key insight:
    - Length ranking drives two linked effects at once: body recoloring and outline thickness, so the longest bar both becomes green and claims the widest yellow territory.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bars = sorted(_find_gray_bars(g), key=lambda spec: spec["length"])
    ranked_specs = [_rank_spec(spec, rank) for rank, spec in enumerate(bars, start=1)]
    return _render_from_specs(ranked_specs, g.shape, _get_bg(g))


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _all_full_fit_specs(shape: tuple[int, int], length: int, rank: int, orientation: str) -> list[dict]:
    height, width = shape
    specs: list[dict] = []
    if orientation == "h":
        for row in range(rank, height - rank):
            for col0 in range(rank, width - length - rank + 1):
                specs.append(_make_spec("h", row, col0, length, rank))
    else:
        for row0 in range(rank, height - length - rank + 1):
            for col in range(rank, width - rank):
                specs.append(_make_spec("v", row0, col, length, rank))
    return specs


def _all_non_touching_specs(
    shape: tuple[int, int],
    length: int,
    rank: int,
    orientation_choices: tuple[str, ...],
    existing: list[dict],
) -> list[dict]:
    candidates: list[dict] = []
    for orientation in orientation_choices:
        for spec in _all_full_fit_specs(shape, length, rank, orientation):
            if all(not _bars_touch(spec, other) for other in existing):
                candidates.append(spec)
    random.shuffle(candidates)
    return candidates


def _outline_overlap_count(specs: list[dict], shape: tuple[int, int]) -> int:
    body_cells = {cell for spec in specs for cell in spec["cells"]}
    per_bar = [_outline_cells(spec, shape) - body_cells for spec in specs]
    overlap: set[tuple[int, int]] = set()
    for i in range(len(per_bar)):
        for j in range(i + 1, len(per_bar)):
            overlap |= per_bar[i] & per_bar[j]
    return len(overlap)


def _pairwise_overlap_counts(specs: list[dict], shape: tuple[int, int]) -> dict[tuple[int, int], int]:
    body_cells = {cell for spec in specs for cell in spec["cells"]}
    per_bar = {spec["rank"]: _outline_cells(spec, shape) - body_cells for spec in specs}
    return {
        (ra, rb): len(per_bar[ra] & per_bar[rb])
        for ra in per_bar
        for rb in per_bar
        if ra < rb
    }


def _has_clipped_outline(specs: list[dict], shape: tuple[int, int]) -> bool:
    height, width = shape
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        pad = spec["rank"]
        if r0 - pad < 0 or c0 - pad < 0 or r1 + pad >= height or c1 + pad >= width:
            return True
    return False


def _construct_input(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for spec in specs:
        for r, c in spec["cells"]:
            grid[r, c] = GRAY
    return grid


def _spec_key(spec: dict) -> tuple:
    return (
        spec["orientation"],
        spec["length"],
        spec["rank"],
        spec["bbox"],
    )


def _validate_candidate(
    shape: tuple[int, int],
    specs: list[dict],
    want_overlap: bool,
    want_clip: bool,
    require_longest_overlap: bool,
) -> np.ndarray | None:
    input_grid = _construct_input(shape, specs)

    recovered = sorted(_find_gray_bars(input_grid), key=lambda spec: spec["length"])
    recovered_ranked = [_rank_spec(spec, rank) for rank, spec in enumerate(recovered, start=1)]
    if sorted(_spec_key(spec) for spec in recovered_ranked) != sorted(_spec_key(spec) for spec in specs):
        return None

    expected = _render_from_specs(specs, shape, BLACK)
    actual = main(input_grid)
    if not np.array_equal(actual, expected):
        return None

    overlap_count = _outline_overlap_count(specs, shape)
    if want_overlap and overlap_count == 0:
        return None
    if not want_overlap and overlap_count != 0:
        return None
    if overlap_count > MAX_OVERLAP_CELLS:
        return None

    pairwise = _pairwise_overlap_counts(specs, shape)
    if require_longest_overlap and not any(3 in pair and count > 0 for pair, count in pairwise.items()):
        return None

    if _has_clipped_outline(specs, shape) != want_clip:
        return None

    if np.count_nonzero(actual == YELLOW) == 0:
        return None
    if np.count_nonzero(actual == BLACK) <= input_grid.size // 5:
        return None
    return input_grid


def _sample_lengths(shape: tuple[int, int]) -> list[int]:
    max_len = min(shape) - 6
    return sorted(random.sample(range(3, max_len + 1), 3))


def _parallel_pair(
    shape: tuple[int, int],
    long_len: int,
    partner_len: int,
    partner_rank: int,
    orientation: str,
    clip_partner: bool,
) -> tuple[dict, dict] | None:
    height, width = shape
    gaps = list(range(3, min(5, 3 + partner_rank) + 1))
    random.shuffle(gaps)

    for gap in gaps:
        if orientation == "h":
            row_pairs: list[tuple[int, int]] = []
            if clip_partner:
                if partner_rank != 2:
                    continue
                for partner_row in (1, height - 2):
                    for long_row in (partner_row + gap + 1, partner_row - gap - 1):
                        if 3 <= long_row <= height - 4:
                            row_pairs.append((long_row, partner_row))
                partner_col_min = partner_rank
                partner_col_max = width - partner_len - partner_rank
            else:
                for long_row in range(3, height - 3):
                    for partner_row in (long_row + gap + 1, long_row - gap - 1):
                        if partner_rank <= partner_row <= height - 1 - partner_rank:
                            row_pairs.append((long_row, partner_row))
                partner_col_min = partner_rank
                partner_col_max = width - partner_len - partner_rank

            long_col_min = 3
            long_col_max = width - long_len - 3
            if long_col_min > long_col_max or partner_col_min > partner_col_max:
                continue

            col_pairs: list[tuple[int, int]] = []
            for long_col in range(long_col_min, long_col_max + 1):
                for partner_col in range(partner_col_min, partner_col_max + 1):
                    shared = _interval_overlap(
                        long_col,
                        long_col + long_len - 1,
                        partner_col,
                        partner_col + partner_len - 1,
                    )
                    if shared >= 2:
                        col_pairs.append((long_col, partner_col))

            if not row_pairs or not col_pairs:
                continue

            random.shuffle(row_pairs)
            random.shuffle(col_pairs)
            for long_row, partner_row in row_pairs:
                for long_col, partner_col in col_pairs:
                    long_spec = _make_spec("h", long_row, long_col, long_len, 3)
                    partner_spec = _make_spec("h", partner_row, partner_col, partner_len, partner_rank)
                    if not _bars_touch(long_spec, partner_spec):
                        return long_spec, partner_spec
        else:
            col_pairs = []
            if clip_partner:
                if partner_rank != 2:
                    continue
                for partner_col in (1, width - 2):
                    for long_col in (partner_col + gap + 1, partner_col - gap - 1):
                        if 3 <= long_col <= width - 4:
                            col_pairs.append((long_col, partner_col))
                partner_row_min = partner_rank
                partner_row_max = height - partner_len - partner_rank
            else:
                for long_col in range(3, width - 3):
                    for partner_col in (long_col + gap + 1, long_col - gap - 1):
                        if partner_rank <= partner_col <= width - 1 - partner_rank:
                            col_pairs.append((long_col, partner_col))
                partner_row_min = partner_rank
                partner_row_max = height - partner_len - partner_rank

            long_row_min = 3
            long_row_max = height - long_len - 3
            if long_row_min > long_row_max or partner_row_min > partner_row_max:
                continue

            row_pairs: list[tuple[int, int]] = []
            for long_row in range(long_row_min, long_row_max + 1):
                for partner_row in range(partner_row_min, partner_row_max + 1):
                    shared = _interval_overlap(
                        long_row,
                        long_row + long_len - 1,
                        partner_row,
                        partner_row + partner_len - 1,
                    )
                    if shared >= 2:
                        row_pairs.append((long_row, partner_row))

            if not col_pairs or not row_pairs:
                continue

            random.shuffle(col_pairs)
            random.shuffle(row_pairs)
            for long_col, partner_col in col_pairs:
                for long_row, partner_row in row_pairs:
                    long_spec = _make_spec("v", long_row, long_col, long_len, 3)
                    partner_spec = _make_spec("v", partner_row, partner_col, partner_len, partner_rank)
                    if not _bars_touch(long_spec, partner_spec):
                        return long_spec, partner_spec
    return None


def _generate_overlap_case(shape: tuple[int, int], lengths: list[int], want_clip: bool) -> np.ndarray | None:
    rank_to_length = {1: lengths[0], 2: lengths[1], 3: lengths[2]}
    partner_rank = 2 if want_clip or random.random() < 0.6 else 1
    remaining_rank = 1 if partner_rank == 2 else 2
    axis = random.choice(("h", "v"))

    pair = _parallel_pair(shape, rank_to_length[3], rank_to_length[partner_rank], partner_rank, axis, want_clip)
    if pair is None:
        return None
    long_spec, partner_spec = pair

    third_choices = (axis, "v" if axis == "h" else "h")
    third_specs = _all_non_touching_specs(shape, rank_to_length[remaining_rank], remaining_rank, third_choices, [long_spec, partner_spec])
    for third_spec in third_specs:
        specs = sorted([long_spec, partner_spec, third_spec], key=lambda spec: spec["rank"])
        grid = _validate_candidate(shape, specs, True, want_clip, True)
        if grid is not None:
            return grid
    return None


def _generate_separated_case(shape: tuple[int, int], lengths: list[int]) -> np.ndarray | None:
    orientation_pattern = [random.choice(("h", "v")) for _ in range(3)]
    if random.random() < 0.6:
        idx_h, idx_v = random.sample(range(3), 2)
        orientation_pattern[idx_h] = "h"
        orientation_pattern[idx_v] = "v"

    specs: list[dict] = []
    for rank, length in enumerate(lengths, start=1):
        orientation_choices = (orientation_pattern[rank - 1],)
        candidates = _all_non_touching_specs(shape, length, rank, orientation_choices, specs)
        if not candidates:
            return None
        spec = random.choice(candidates)
        specs.append(spec)

    return _validate_candidate(shape, specs, False, False, False)


def generate_input() -> np.ndarray:
    want_overlap = random.random() < OVERLAP_PROBABILITY
    want_clip = want_overlap and random.random() < CLIP_PROBABILITY
    for _ in range(MAX_GENERATION_ATTEMPTS):
        shape = (
            random.randrange(*GRID_HEIGHT_RANGE),
            random.randrange(*GRID_WIDTH_RANGE),
        )
        lengths = _sample_lengths(shape)

        if want_overlap:
            grid = _generate_overlap_case(shape, lengths, want_clip)
        else:
            grid = _generate_separated_case(shape, lengths)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")


def _semantic_check(grid: np.ndarray) -> dict[str, int | bool]:
    bars = sorted(_find_gray_bars(grid), key=lambda spec: spec["length"])
    specs = [_rank_spec(spec, rank) for rank, spec in enumerate(bars, start=1)]
    out = main(grid)
    expected = _render_from_specs(specs, grid.shape, BLACK)
    assert np.array_equal(out, expected)
    return {
        "overlap_cells": _outline_overlap_count(specs, grid.shape),
        "clipped": _has_clipped_outline(specs, grid.shape),
        "mixed_orientation": len({spec["orientation"] for spec in specs}) > 1,
        "yellow_cells": int(np.count_nonzero(out == YELLOW)),
    }
