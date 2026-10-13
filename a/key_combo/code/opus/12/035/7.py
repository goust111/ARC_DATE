from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
MARKER = 1
BASE = 4
FILL = 8

GRID_HEIGHT_RANGE = (12, 18)
GRID_WIDTH_RANGE = (16, 22)
RECT_HEIGHT_RANGE = (4, 6)
RECT_WIDTH_RANGE = (5, 8)
RECT_COUNT_OPTIONS = (2, 3)
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_RETRIES = 60


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(counts.most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _extract_rectangles(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    assert bg == BG, "expected black background"

    rects: list[np.ndarray] = []
    for coords in _components(g != bg):
        r0, c0, r1, c1 = _bbox(coords)
        patch = _crop(g, (r0, c0, r1, c1))
        assert np.all(patch != bg), "rectangle must be dense inside its bbox"
        assert set(map(int, np.unique(patch))) <= {BASE, MARKER}, "unexpected rectangle colors"

        markers = np.argwhere(patch == MARKER)
        h, w = patch.shape
        assert 1 <= len(markers) <= 3, "each source rectangle must carry 1-3 markers"
        for mr, mc in markers:
            assert 0 < int(mr) < h - 1 and 0 < int(mc) < w - 1, "markers must stay in the interior"
        row_counts = np.count_nonzero(patch == MARKER, axis=1)
        assert int(row_counts.max()) <= 1, "each source row may contain at most one marker"
        rects.append(patch)

    assert 2 <= len(rects) <= 3, "expected 2-3 rectangles"
    shapes = {rect.shape for rect in rects}
    assert len(shapes) == 1, "all rectangles must share one size"
    return rects


def _merge_rectangles(rects: list[np.ndarray]) -> np.ndarray:
    h, w = rects[0].shape
    merged = np.full((h, w), BASE, dtype=int)
    for rect in rects:
        marker_positions = rect == MARKER
        merged[marker_positions] = MARKER
    return merged


def _paint_trigger_rows(merged: np.ndarray) -> np.ndarray:
    out = merged.copy()
    row_counts = np.count_nonzero(merged == MARKER, axis=1)
    for r, count in enumerate(row_counts):
        if int(count) >= 2:
            yellow = out[r] == BASE
            assert np.any(yellow), "triggered rows must contain fillable base cells"
            out[r, yellow] = FILL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay same-sized marked rectangles, then flood-fill rows whose merged marker density crosses the threshold.

    Input structure:
    - The grid has a black background with 2-3 separated dense rectangles.
    - Every rectangle has the same size and uses yellow as its base color.
    - Each rectangle contains 1-3 blue single-pixel markers, all strictly inside the rectangle,
      and no source row contains more than one marker.

    Rule:
    - Detect the same-sized rectangles and align them by local coordinates.
    - Overlay all blue marker pixels onto one yellow rectangle of that shared size.
    - Count blue markers row by row in the merged rectangle.
    - For every row whose merged blue count is at least 2, recolor the yellow cells in that row
      to teal while leaving the blue markers unchanged.
    - Return only the merged rectangle, cropped to its own size.

    Key insight:
    - No individual rectangle ever reaches the per-row threshold, so the teal flood-fill can only
      be inferred after transparently overlaying multiple rectangles in shared local coordinates.
    """
    rects = _extract_rectangles(np.asarray(grid, dtype=int))
    merged = _merge_rectangles(rects)
    row_counts = np.count_nonzero(merged == MARKER, axis=1)
    assert np.any(row_counts >= 2), "the merged rectangle must contain at least one triggered row"
    return _paint_trigger_rows(merged)


def _build_patch(shape: tuple[int, int], markers: list[tuple[int, int]]) -> np.ndarray:
    patch = np.full(shape, BASE, dtype=int)
    for r, c in markers:
        patch[int(r), int(c)] = MARKER
    return patch


def _build_expected(shape: tuple[int, int], layer_markers: list[list[tuple[int, int]]]) -> np.ndarray:
    merged = np.full(shape, BASE, dtype=int)
    for markers in layer_markers:
        for r, c in markers:
            merged[int(r), int(c)] = MARKER
    return _paint_trigger_rows(merged)


def _nontrigger_rows(row_counts: dict[int, int]) -> list[int]:
    return [r for r, count in row_counts.items() if count < 2]


def _add_marker(
    layer_idx: int,
    row: int,
    layer_markers: list[list[tuple[int, int]]],
    layer_rows: list[set[int]],
    occupied_cols_by_row: dict[int, set[int]],
    interior_cols: list[int],
    row_counts: dict[int, int],
) -> bool:
    if row in layer_rows[layer_idx]:
        return False
    available_cols = [c for c in interior_cols if c not in occupied_cols_by_row[row]]
    if not available_cols:
        return False
    col = random.choice(available_cols)
    layer_markers[layer_idx].append((row, col))
    layer_rows[layer_idx].add(row)
    occupied_cols_by_row[row].add(col)
    row_counts[row] += 1
    return True


def _oracle_valid_layers(shape: tuple[int, int], layer_markers: list[list[tuple[int, int]]]) -> bool:
    h, w = shape
    merged_counts = np.zeros(h, dtype=int)
    for markers in layer_markers:
        if not (1 <= len(markers) <= 3):
            return False
        rows = [r for r, _c in markers]
        if len(rows) != len(set(rows)):
            return False
        for r, c in markers:
            if not (0 < r < h - 1 and 0 < c < w - 1):
                return False
            merged_counts[int(r)] += 1
    return np.any(merged_counts >= 2) and np.any(merged_counts == 1)


def _make_layer_markers(h: int, w: int, n_rects: int) -> list[list[tuple[int, int]]] | None:
    interior_rows = list(range(1, h - 1))
    interior_cols = list(range(1, w - 1))
    if len(interior_rows) < 2 or len(interior_cols) < 2:
        return None

    layer_markers: list[list[tuple[int, int]]] = [[] for _ in range(n_rects)]
    layer_rows: list[set[int]] = [set() for _ in range(n_rects)]
    occupied_cols_by_row = {row: set() for row in interior_rows}
    row_counts = {row: 0 for row in interior_rows}

    max_trigger_rows = min(2, len(interior_rows) - 1)
    trigger_count = 1 if max_trigger_rows == 1 else random.choice([1, 2])
    trigger_rows = random.sample(interior_rows, trigger_count)

    for row in trigger_rows:
        participants = random.randint(2, n_rects)
        chosen_layers = random.sample(range(n_rects), participants)
        for layer_idx in chosen_layers:
            if not _add_marker(
                layer_idx,
                row,
                layer_markers,
                layer_rows,
                occupied_cols_by_row,
                interior_cols,
                row_counts,
            ):
                return None

    nontrigger_rows = [row for row in interior_rows if row not in trigger_rows]
    if not nontrigger_rows:
        return None
    single_row = random.choice(nontrigger_rows)
    empty_layers = [idx for idx, markers in enumerate(layer_markers) if not markers]
    single_layer = random.choice(empty_layers if empty_layers else list(range(n_rects)))
    if not _add_marker(
        single_layer,
        single_row,
        layer_markers,
        layer_rows,
        occupied_cols_by_row,
        interior_cols,
        row_counts,
    ):
        return None

    for layer_idx in range(n_rects):
        if layer_markers[layer_idx]:
            continue
        candidate_rows = [row for row in trigger_rows if row not in layer_rows[layer_idx]]
        if not candidate_rows:
            candidate_rows = [
                row
                for row in nontrigger_rows
                if row_counts[row] == 0 and row not in layer_rows[layer_idx]
            ]
        if not candidate_rows:
            return None
        if not _add_marker(
            layer_idx,
            random.choice(candidate_rows),
            layer_markers,
            layer_rows,
            occupied_cols_by_row,
            interior_cols,
            row_counts,
        ):
            return None

    extra_steps = random.randint(0, n_rects)
    for _ in range(extra_steps):
        growable = [idx for idx, markers in enumerate(layer_markers) if len(markers) < 3]
        if not growable:
            break
        layer_idx = random.choice(growable)
        trigger_candidates = [row for row in trigger_rows if row not in layer_rows[layer_idx]]
        quiet_candidates = [
            row for row in nontrigger_rows if row_counts[row] == 0 and row not in layer_rows[layer_idx]
        ]
        if trigger_candidates and quiet_candidates:
            candidate_rows = trigger_candidates if random.random() < 0.6 else quiet_candidates
        else:
            candidate_rows = trigger_candidates or quiet_candidates
        if not candidate_rows:
            continue
        _add_marker(
            layer_idx,
            random.choice(candidate_rows),
            layer_markers,
            layer_rows,
            occupied_cols_by_row,
            interior_cols,
            row_counts,
        )

    if not _oracle_valid_layers((h, w), layer_markers):
        return None
    return layer_markers


def _can_place(
    top: int,
    left: int,
    shape: tuple[int, int],
    placed: list[tuple[int, int, int, int]],
    grid_shape: tuple[int, int],
) -> bool:
    h, w = shape
    gh, gw = grid_shape
    if top < 1 or left < 1 or top + h > gh - 1 or left + w > gw - 1:
        return False
    r0, c0, r1, c1 = top, left, top + h - 1, left + w - 1
    for pr0, pc0, pr1, pc1 in placed:
        separated_rows = r1 + 1 < pr0 or pr1 + 1 < r0
        separated_cols = c1 + 1 < pc0 or pc1 + 1 < c0
        if not (separated_rows or separated_cols):
            return False
    return True


def _scatter_patches(
    grid_shape: tuple[int, int],
    patches: list[np.ndarray],
) -> np.ndarray | None:
    gh, gw = grid_shape
    grid = np.full((gh, gw), BG, dtype=int)
    placed: list[tuple[int, int, int, int]] = []
    order = list(range(len(patches)))
    random.shuffle(order)

    for patch_idx in order:
        patch = patches[patch_idx]
        h, w = patch.shape
        candidates = [
            (r, c)
            for r in range(1, gh - h)
            for c in range(1, gw - w)
            if _can_place(r, c, patch.shape, placed, grid_shape)
        ]
        if not candidates:
            return None
        top, left = random.choice(candidates)
        grid[top : top + h, left : left + w] = patch
        placed.append((top, left, top + h - 1, left + w - 1))
    return grid


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*RECT_HEIGHT_RANGE)
    w = random.randint(*RECT_WIDTH_RANGE)
    n_rects = random.choice(RECT_COUNT_OPTIONS)

    layer_markers = _make_layer_markers(h, w, n_rects)
    if layer_markers is None:
        return None

    patches = [_build_patch((h, w), markers) for markers in layer_markers]
    expected = _build_expected((h, w), layer_markers)
    assert np.any(expected == FILL)
    assert np.any(np.count_nonzero(expected == MARKER, axis=1) == 1)

    grid_h = random.randint(max(GRID_HEIGHT_RANGE[0], h + 4), GRID_HEIGHT_RANGE[1])
    grid_w = random.randint(max(GRID_WIDTH_RANGE[0], w + 4), GRID_WIDTH_RANGE[1])

    for _ in range(PLACEMENT_RETRIES):
        grid = _scatter_patches((grid_h, grid_w), patches)
        if grid is None:
            continue
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
