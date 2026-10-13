import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

PIECE_COUNT_RANGE = (3, 5)
AREA_POOL = (2, 3, 4, 5, 6)
MAX_GENERATION_ATTEMPTS = 400
TARGET_LAYOUT_ATTEMPTS = 160
SOURCE_LAYOUT_ATTEMPTS = 160


def _normalize_cells(cells):
    arr = np.array(sorted(set(cells)), dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    arr = arr[order]
    return [tuple(map(int, rc)) for rc in arr]


def _shape_dims(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return max(rows) + 1, max(cols) + 1


def _is_4_connected(cells):
    cell_set = {tuple(map(int, rc)) for rc in cells}
    if not cell_set:
        return False
    stack = [next(iter(cell_set))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nb = (r + dr, c + dc)
            if nb in cell_set and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cell_set)


def _translated_cells(cells, top, left):
    return {(top + r, left + c) for r, c in cells}


def _stack_metrics(pieces, anchors):
    translated = [
        _translated_cells(piece["cells"], int(anchor[0]), int(anchor[1]))
        for piece, anchor in zip(pieces, anchors, strict=False)
    ]
    occupancy = {}
    pair_overlaps = 0
    for i, cells in enumerate(translated):
        for cell in cells:
            occupancy[cell] = occupancy.get(cell, 0) + 1
        for j in range(i):
            if cells & translated[j]:
                pair_overlaps += 1
    overlap_cells = sum(1 for count in occupancy.values() if count > 1)
    max_depth = max(occupancy.values(), default=0)
    return {
        "max_depth": int(max_depth),
        "overlap_cells": int(overlap_cells),
        "pair_overlaps": int(pair_overlaps),
    }


def _parse_pieces(grid):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    top_marks = {}
    for c in range(1, w):
        color = int(g[0, c])
        if color != BG:
            assert color not in top_marks
            top_marks[color] = c

    left_marks = {}
    for r in range(1, h):
        color = int(g[r, 0])
        if color != BG:
            assert color not in left_marks
            left_marks[color] = r

    assert top_marks
    assert top_marks.keys() == left_marks.keys()
    ruler_colors = set(top_marks.keys())
    interior_colors = {int(v) for v in np.unique(g[1:, 1:]) if int(v) != BG}
    assert interior_colors == ruler_colors

    pieces = []
    for color, target_c in top_marks.items():
        coords = np.argwhere(g[1:, 1:] == int(color))
        assert coords.size > 0
        coords = coords + 1
        abs_cells = [tuple(map(int, rc)) for rc in coords]
        assert _is_4_connected(abs_cells)
        area = len(abs_cells)
        assert 2 <= area <= 6

        src_r = min(r for r, _ in abs_cells)
        src_c = min(c for _, c in abs_cells)
        rel_cells = [(r - src_r, c - src_c) for r, c in abs_cells]
        ph, pw = _shape_dims(rel_cells)

        target_r = int(left_marks[color])
        assert 1 <= target_r <= h - ph
        assert 1 <= target_c <= w - pw

        pieces.append(
            {
                "color": int(color),
                "area": int(area),
                "cells": rel_cells,
                "h": int(ph),
                "w": int(pw),
                "src_top": int(src_r),
                "src_left": int(src_c),
                "target_top": int(target_r),
                "target_left": int(target_c),
                "abs_cells": abs_cells,
            }
        )
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    pieces = _parse_pieces(g)
    out = g.copy()

    for piece in pieces:
        for r, c in piece["abs_cells"]:
            out[r, c] = BG

    pieces = sorted(
        pieces,
        key=lambda piece: (
            int(piece["area"]),
            int(piece["src_top"]),
            int(piece["src_left"]),
            int(piece["target_top"]),
            int(piece["target_left"]),
        ),
    )
    for piece in pieces:
        top = int(piece["target_top"])
        left = int(piece["target_left"])
        color = int(piece["color"])
        for dr, dc in piece["cells"]:
            out[top + dr, left + dc] = color

    out[0, :] = g[0, :]
    out[:, 0] = g[:, 0]
    return out


def _random_polyomino(area, rng):
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in DIR4:
                nb = (r + dr, c + dc)
                if nb not in cells:
                    frontier.add(nb)
        frontier = list(sorted(frontier))
        cells.add(frontier[int(rng.integers(0, len(frontier)))])

    arr = np.array(list(cells), dtype=int)
    for _ in range(int(rng.integers(0, 4))):
        arr = np.stack([arr[:, 1], -arr[:, 0]], axis=1)
    if bool(rng.integers(0, 2)):
        arr[:, 0] *= -1
    if bool(rng.integers(0, 2)):
        arr[:, 1] *= -1
    return _normalize_cells([tuple(map(int, rc)) for rc in arr])


def _assign_unique(maxima, rng, preset=None):
    values = [None] * len(maxima)
    used = set() if preset is None else set(int(v) for v in preset if v is not None)
    preset = [None] * len(maxima) if preset is None else list(preset)

    for idx, value in enumerate(preset):
        if value is not None:
            if value < 1 or value > int(maxima[idx]):
                return None
            values[idx] = int(value)

    order = [int(i) for i in rng.permutation(len(maxima)) if values[int(i)] is None]
    for idx in order:
        candidates = [v for v in range(1, int(maxima[idx]) + 1) if v not in used]
        if not candidates:
            return None
        choice = int(candidates[int(rng.integers(0, len(candidates)))])
        values[idx] = choice
        used.add(choice)
    return values


def _choose_non_overlapping_targets(pieces, height, width, rng):
    row_max = [height - int(piece["h"]) for piece in pieces]
    col_max = [width - int(piece["w"]) for piece in pieces]
    for _ in range(TARGET_LAYOUT_ATTEMPTS):
        rows = _assign_unique(row_max, rng)
        cols = _assign_unique(col_max, rng)
        if rows is None or cols is None:
            continue
        anchors = list(zip(rows, cols, strict=False))
        metrics = _stack_metrics(pieces, anchors)
        if metrics["max_depth"] == 1:
            return anchors
    return None


def _seed_overlap_pair(piece_a, piece_b, height, width, rng):
    for _ in range(TARGET_LAYOUT_ATTEMPTS):
        top_a = int(rng.integers(1, height - int(piece_a["h"]) + 1))
        left_a = int(rng.integers(1, width - int(piece_a["w"]) + 1))
        cells_a = _translated_cells(piece_a["cells"], top_a, left_a)

        row_lo = max(1, top_a - int(piece_b["h"]) + 1)
        row_hi = min(height - int(piece_b["h"]), top_a + int(piece_a["h"]) - 1)
        col_lo = max(1, left_a - int(piece_b["w"]) + 1)
        col_hi = min(width - int(piece_b["w"]), left_a + int(piece_a["w"]) - 1)

        candidates = []
        for top_b in range(row_lo, row_hi + 1):
            if top_b == top_a:
                continue
            for left_b in range(col_lo, col_hi + 1):
                if left_b == left_a:
                    continue
                overlap = len(cells_a & _translated_cells(piece_b["cells"], top_b, left_b))
                if 1 <= overlap <= 3:
                    candidates.append((top_b, left_b, overlap))
        if candidates:
            candidates.sort(key=lambda item: (abs(item[2] - 2), item[0], item[1]))
            top_b, left_b, _ = candidates[int(rng.integers(0, min(len(candidates), 6)))]
            return (top_a, left_a), (int(top_b), int(left_b))
    return None


def _choose_overlapping_targets(pieces, height, width, rng):
    n = len(pieces)
    pair_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for pair_slot in rng.permutation(len(pair_indices)):
        i, j = pair_indices[int(pair_slot)]
        seed = _seed_overlap_pair(pieces[i], pieces[j], height, width, rng)
        if seed is None:
            continue

        rows = [None] * n
        cols = [None] * n
        rows[i], cols[i] = seed[0]
        rows[j], cols[j] = seed[1]

        row_max = [height - int(piece["h"]) for piece in pieces]
        col_max = [width - int(piece["w"]) for piece in pieces]

        for _ in range(TARGET_LAYOUT_ATTEMPTS):
            filled_rows = _assign_unique(row_max, rng, preset=rows)
            filled_cols = _assign_unique(col_max, rng, preset=cols)
            if filled_rows is None or filled_cols is None:
                continue
            anchors = list(zip(filled_rows, filled_cols, strict=False))
            metrics = _stack_metrics(pieces, anchors)
            if (
                metrics["max_depth"] == 2
                and metrics["pair_overlaps"] >= 1
                and metrics["overlap_cells"] <= 4
            ):
                return anchors
    return None


def _rectangles_too_close(rect_a, rect_b, pad=1):
    ar0, ac0, ar1, ac1 = rect_a
    br0, bc0, br1, bc1 = rect_b
    return not (
        ar1 + pad < br0 - pad
        or br1 + pad < ar0 - pad
        or ac1 + pad < bc0 - pad
        or bc1 + pad < ac0 - pad
    )


def _place_sources(pieces, anchors, height, width, rng):
    order = sorted(
        range(len(pieces)),
        key=lambda idx: (int(pieces[idx]["h"]) * int(pieces[idx]["w"]), int(pieces[idx]["area"])),
        reverse=True,
    )
    source_rects = []
    positions = [None] * len(pieces)

    for idx in order:
        piece = pieces[idx]
        for _ in range(SOURCE_LAYOUT_ATTEMPTS):
            top = int(rng.integers(1, height - int(piece["h"]) + 1))
            left = int(rng.integers(1, width - int(piece["w"]) + 1))
            if (top, left) == anchors[idx]:
                continue
            rect = (top, left, top + int(piece["h"]) - 1, left + int(piece["w"]) - 1)
            if any(_rectangles_too_close(rect, other, pad=1) for other in source_rects):
                continue
            positions[idx] = (top, left)
            source_rects.append(rect)
            break
        else:
            return None
    return positions


def _render_input(height, width, pieces, anchors, sources):
    grid = np.zeros((height, width), dtype=int)
    for piece, anchor in zip(pieces, anchors, strict=False):
        grid[0, int(anchor[1])] = int(piece["color"])
        grid[int(anchor[0]), 0] = int(piece["color"])
    for piece, source in zip(pieces, sources, strict=False):
        top, left = map(int, source)
        for dr, dc in piece["cells"]:
            grid[top + dr, left + dc] = int(piece["color"])
    return grid


def _validate_generated(grid, expect_overlap):
    pieces = _parse_pieces(grid)
    anchors = [(int(piece["target_top"]), int(piece["target_left"])) for piece in pieces]
    metrics = _stack_metrics(pieces, anchors)
    if expect_overlap:
        assert metrics["max_depth"] == 2
        assert metrics["pair_overlaps"] >= 1
    else:
        assert metrics["max_depth"] == 1
    for piece in pieces:
        assert (int(piece["src_top"]), int(piece["src_left"])) != (
            int(piece["target_top"]),
            int(piece["target_left"]),
        )
    out = main(grid)
    assert np.array_equal(out[0, :], grid[0, :])
    assert np.array_equal(out[:, 0], grid[:, 0])


def _gen_attempt(rng):
    piece_count = int(rng.integers(PIECE_COUNT_RANGE[0], PIECE_COUNT_RANGE[1] + 1))
    min_size = max(12, 2 * piece_count + 6)
    height = int(rng.integers(min_size, min_size + 5))
    width = int(rng.integers(min_size, min_size + 5))

    colors = [int(v) for v in rng.choice(np.arange(1, 10), size=piece_count, replace=False)]
    areas = [int(v) for v in rng.choice(np.array(AREA_POOL), size=piece_count, replace=False)]

    pieces = []
    for color, area in zip(colors, areas, strict=False):
        cells = _random_polyomino(int(area), rng)
        ph, pw = _shape_dims(cells)
        if ph >= height or pw >= width:
            return None
        pieces.append(
            {
                "color": int(color),
                "area": int(area),
                "cells": cells,
                "h": int(ph),
                "w": int(pw),
            }
        )

    want_overlap = bool(rng.integers(0, 2))
    if want_overlap:
        anchors = _choose_overlapping_targets(pieces, height, width, rng)
    else:
        anchors = _choose_non_overlapping_targets(pieces, height, width, rng)
    if anchors is None:
        return None

    sources = _place_sources(pieces, anchors, height, width, rng)
    if sources is None:
        return None

    grid = _render_input(height, width, pieces, anchors, sources)
    _validate_generated(grid, want_overlap)
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
