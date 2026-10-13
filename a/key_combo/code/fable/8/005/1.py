from __future__ import annotations

import random

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _line_cells(a: tuple[int, int], b: tuple[int, int], *, include_ends: bool) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        start = lo if include_ends else lo + 1
        stop = hi + 1 if include_ends else hi
        return [(int(r0), int(c)) for c in range(start, stop)]
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        start = lo if include_ends else lo + 1
        stop = hi + 1 if include_ends else hi
        return [(int(r), int(c0)) for r in range(start, stop)]
    raise AssertionError("paired pixels must share a row or a column")


def _meeting_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        gap = hi - lo - 1
        if gap % 2 == 1:
            return [(int(r0), int((lo + hi) // 2))]
        mid = (lo + hi - 1) // 2
        return [(int(r0), int(mid)), (int(r0), int(mid + 1))]
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        gap = hi - lo - 1
        if gap % 2 == 1:
            return [(int((lo + hi) // 2), int(c0))]
        mid = (lo + hi - 1) // 2
        return [(int(mid), int(c0)), (int(mid + 1), int(c0))]
    raise AssertionError("paired pixels must share a row or a column")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each same-color endpoint pair inward, keep only the central residue, and crop out the revealed sprite.

    Input structure:
    - A black background contains isolated non-black pixels, and each color appears exactly twice.
    - The two pixels of each color lie on one shared row or one shared column.
    - Every cell strictly between paired pixels is black, and the eventual meeting residues form a compact hidden figure.

    Rule:
    - For each color, treat its two pixels as a pair moving simultaneously toward each other without leaving a trail.
    - If the number of black cells between them is odd, keep the single central meeting cell of that color.
    - If the number of black cells between them is even, keep the two central adjacent cells of that color.
    - Overlay all residues on black and return their minimal bounding-box crop.

    Key insight:
    - The visible endpoints are only launch markers; the real picture is the set of midpoints implied by each pair's alignment and spacing parity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = np.zeros_like(g)
    colors = [int(color) for color in np.unique(g) if int(color) != BLACK]
    assert colors, "expected at least one non-black pair"

    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each color must appear exactly twice"
        a = (int(coords[0, 0]), int(coords[0, 1]))
        b = (int(coords[1, 0]), int(coords[1, 1]))
        assert a[0] == b[0] or a[1] == b[1], "same-color pixels must share a row or a column"
        inner = _line_cells(a, b, include_ends=False)
        assert inner, "paired pixels must have at least one black cell between them"
        assert all(int(g[r, c]) == BLACK for r, c in inner), "only black cells may lie between a pair"
        for r, c in _meeting_cells(a, b):
            assert out[r, c] == BLACK, "meeting residues must not overlap"
            out[r, c] = color

    box = _bbox(out != BLACK)
    assert box is not None, "expected at least one meeting residue"
    return _crop_bbox(out, box)


def _shape_cells(size: int, axis: str, anchor: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = anchor
    if size == 1:
        return [(r, c)]
    if axis == "h":
        return [(r, c), (r, c + 1)]
    return [(r, c), (r + 1, c)]


def _adjacent_to_union(cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for dr, dc in DIR4:
            if (r + dr, c + dc) in occupied:
                return True
    return False


def _connected(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _normalize_segments(
    segments: list[dict[str, object]]
) -> tuple[list[dict[str, object]], tuple[int, int], set[tuple[int, int]]]:
    occupied = {cell for seg in segments for cell in seg["cells"]}
    rows = [r for r, _ in occupied]
    cols = [c for _, c in occupied]
    r0, c0 = min(rows), min(cols)
    normalized = []
    shifted = set()
    for seg in segments:
        cells = [(r - r0, c - c0) for r, c in seg["cells"]]
        normalized.append({"size": seg["size"], "axis": seg["axis"], "cells": cells})
        shifted.update(cells)
    height = max(r for r, _ in shifted) + 1
    width = max(c for _, c in shifted) + 1
    return normalized, (height, width), shifted


def _line_signatures_unique(segments: list[dict[str, object]]) -> bool:
    h_rows: set[int] = set()
    v_cols: set[int] = set()
    for seg in segments:
        axis = str(seg["axis"])
        cells = seg["cells"]
        if axis == "h":
            row = int(cells[0][0])
            if row in h_rows:
                return False
            h_rows.add(row)
        else:
            col = int(cells[0][1])
            if col in v_cols:
                return False
            v_cols.add(col)
    return True


def _choose_specs() -> list[tuple[int, str]]:
    segment_count = random.randint(4, 7)
    double_count = random.randint(1, segment_count - 1)
    single_count = segment_count - double_count

    specs = [(2, random.choice(("h", "v"))) for _ in range(double_count)]
    specs.extend((1, random.choice(("h", "v"))) for _ in range(single_count))

    axes = {axis for _, axis in specs}
    if len(axes) == 1:
        size, axis = specs[0]
        specs[0] = (size, "v" if axis == "h" else "h")
    random.shuffle(specs)
    return specs


def _build_target_sprite() -> tuple[list[dict[str, object]], np.ndarray]:
    temp_size = 11
    center = temp_size // 2

    for _ in range(400):
        specs = _choose_specs()
        max_h = random.randint(4, 7)
        max_w = random.randint(4, 7)

        placed: list[dict[str, object]] = []
        occupied: set[tuple[int, int]] = set()
        failed = False

        for index, (size, axis) in enumerate(specs):
            if index == 0:
                cells = _shape_cells(size, axis, (center, center))
                if any(r >= temp_size or c >= temp_size for r, c in cells):
                    failed = True
                    break
                placed.append({"size": size, "axis": axis, "cells": cells})
                occupied.update(cells)
                continue

            candidates = []
            for r in range(temp_size):
                for c in range(temp_size):
                    cells = _shape_cells(size, axis, (r, c))
                    if any(rr < 0 or cc < 0 or rr >= temp_size or cc >= temp_size for rr, cc in cells):
                        continue
                    if any(cell in occupied for cell in cells):
                        continue
                    if not _adjacent_to_union(cells, occupied):
                        continue

                    new_occupied = occupied | set(cells)
                    rows = [rr for rr, _ in new_occupied]
                    cols = [cc for _, cc in new_occupied]
                    if max(rows) - min(rows) + 1 > max_h or max(cols) - min(cols) + 1 > max_w:
                        continue
                    candidates.append(cells)

            if not candidates:
                failed = True
                break

            cells = random.choice(candidates)
            placed.append({"size": size, "axis": axis, "cells": cells})
            occupied.update(cells)

        if failed:
            continue

        segments, shape, sprite_cells = _normalize_segments(placed)
        height, width = shape
        if height < 2 or width < 2:
            continue
        if not _connected(sprite_cells):
            continue
        if len({r for r, _ in sprite_cells}) < 2 or len({c for _, c in sprite_cells}) < 2:
            continue
        if not any(seg["size"] == 1 for seg in segments):
            continue
        if not any(seg["size"] == 2 for seg in segments):
            continue
        if not _line_signatures_unique(segments):
            continue

        bbox_area = height * width
        if len(sprite_cells) * 2 < bbox_area:
            continue

        colors = random.sample(PALETTE, len(segments))
        out = np.zeros(shape, dtype=int)
        for color, seg in zip(colors, segments):
            seg["color"] = color
            for r, c in seg["cells"]:
                out[r, c] = color
        return segments, out

    raise RuntimeError("failed to build a compact target sprite")


def _endpoints_for_segment(seg: dict[str, object], offset: tuple[int, int], arm: int) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = offset
    cells = [(int(r) + dr, int(c) + dc) for r, c in seg["cells"]]
    axis = str(seg["axis"])
    if int(seg["size"]) == 1:
        r, c = cells[0]
        if axis == "h":
            return (r, c - arm), (r, c + arm)
        return (r - arm, c), (r + arm, c)
    if axis == "h":
        left = min(c for _, c in cells)
        right = max(c for _, c in cells)
        row = cells[0][0]
        return (row, left - arm), (row, right + arm)
    top = min(r for r, _ in cells)
    bottom = max(r for r, _ in cells)
    col = cells[0][1]
    return (top - arm, col), (bottom + arm, col)


def _isolated(points: list[tuple[int, int]]) -> bool:
    for i, (r0, c0) in enumerate(points):
        for r1, c1 in points[i + 1 :]:
            if abs(r0 - r1) + abs(c0 - c1) <= 1:
                return False
    return True


def _valid_pair_layout(pairs: list[tuple[tuple[int, int], tuple[int, int]]], shape: tuple[int, int]) -> bool:
    height, width = shape
    all_points = [point for pair in pairs for point in pair]
    if len(set(all_points)) != len(all_points):
        return False
    if not _isolated(all_points):
        return False
    if not all(0 <= r < height and 0 <= c < width for r, c in all_points):
        return False

    point_set = set(all_points)
    for a, b in pairs:
        for cell in _line_cells(a, b, include_ends=False):
            if cell in point_set:
                return False
    return True


def _search_pairs(
    segments: list[dict[str, object]], offset: tuple[int, int], shape: tuple[int, int], max_arm: int
) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    pairs: list[tuple[tuple[int, int], tuple[int, int]] | None] = [None] * len(segments)
    order = list(range(len(segments)))
    random.shuffle(order)
    order.sort(key=lambda idx: (0 if int(segments[idx]["size"]) == 2 else 1, random.random()))

    def compatible(
        new_pair: tuple[tuple[int, int], tuple[int, int]],
        existing: list[tuple[tuple[int, int], tuple[int, int]]],
    ) -> bool:
        if not _valid_pair_layout(existing + [new_pair], shape):
            return False
        return True

    def backtrack(pos: int, chosen: list[tuple[tuple[int, int], tuple[int, int]]]) -> bool:
        if pos == len(order):
            return True

        seg_idx = order[pos]
        seg = segments[seg_idx]
        arms = list(range(1, max_arm + 1))
        random.shuffle(arms)
        for arm in arms:
            pair = _endpoints_for_segment(seg, offset, arm)
            if not compatible(pair, chosen):
                continue
            pairs[seg_idx] = pair
            chosen.append(pair)
            if backtrack(pos + 1, chosen):
                return True
            chosen.pop()
            pairs[seg_idx] = None
        return False

    if backtrack(0, []):
        return [pair for pair in pairs if pair is not None]
    return None


def _validate_generated_case(inp: np.ndarray, expected: np.ndarray) -> None:
    out = main(inp)
    assert np.array_equal(out, expected), "generated sample must follow the rule exactly"

    colors = [int(color) for color in np.unique(inp) if int(color) != BLACK]
    assert len(colors) >= 4, "need multiple independent pairs"
    residue_lengths = []
    axes = set()
    for color in colors:
        coords = np.argwhere(inp == color)
        assert coords.shape == (2, 2)
        a = (int(coords[0, 0]), int(coords[0, 1]))
        b = (int(coords[1, 0]), int(coords[1, 1]))
        axes.add("h" if a[0] == b[0] else "v")
        residue_lengths.append(len(_meeting_cells(a, b)))

    assert 1 in residue_lengths and 2 in residue_lengths, "both parity branches must be present"
    assert axes == {"h", "v"}, "samples should use both row and column pairings"

    box = _bbox(inp != BLACK)
    assert box is not None
    cropped_input = _crop_bbox(inp, box)
    assert not np.array_equal(cropped_input, expected), "hidden figure must differ from the visible endpoints"


def generate_input() -> np.ndarray:
    for _ in range(500):
        segments, expected = _build_target_sprite()
        sprite_h, sprite_w = expected.shape
        for _ in range(20):
            max_arm = random.randint(2, 8)
            top = max_arm + random.randint(1, 3)
            left = max_arm + random.randint(1, 3)
            height = top + sprite_h + max_arm + random.randint(1, 3)
            width = left + sprite_w + max_arm + random.randint(1, 3)
            if max(height, width) > 30:
                continue

            offset = (top, left)
            pairs = _search_pairs(segments, offset, (height, width), max_arm)
            if pairs is None:
                continue

            inp = np.zeros((height, width), dtype=int)
            for seg, (a, b) in zip(segments, pairs):
                color = int(seg["color"])
                inp[a] = color
                inp[b] = color

            try:
                _validate_generated_case(inp, expected)
            except AssertionError:
                continue
            return inp

    raise RuntimeError("failed to generate a valid input grid")
