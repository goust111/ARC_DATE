from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


for _parent in Path(__file__).resolve().parents:
    _utils_file = _parent / "new" / "solvers" / "arc_utils.py"
    if _utils_file.exists():
        _utils_dir = str(_utils_file.parent)
        if _utils_dir not in sys.path:
            sys.path.insert(0, _utils_dir)
        break
else:
    raise ImportError("Could not locate new/solvers/arc_utils.py")

from arc_utils import Color, find_components, paste


BG = Color.BLACK
RED = Color.RED
GREEN = Color.GREEN
YELLOW = Color.YELLOW

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_HEIGHT_CHOICES = (13, 14)
GEN_WIDTH_RANGE = (13, 16)
GEN_SEGMENT_WIDTHS = (3, 4)
GEN_OBJECT_COUNT_RANGE = (3, 5)
GEN_UPPER_BAND_RANGE = (6, 7)
GEN_ATTEMPTS = 500


def _segment_span(grid: np.ndarray) -> tuple[int, int]:
    cols = np.flatnonzero(grid[0] == YELLOW)
    assert cols.size > 0, "expected a yellow top boundary segment"
    assert int(cols[-1] - cols[0] + 1) == int(cols.size), "segment must be contiguous"
    return int(cols[0]), int(cols[-1])


def _green_rows(height: int) -> set[int]:
    return {r for r in range(0, int(height), 2) if (r // 2) % 3 == 2}


def _settle_red(
    grid: np.ndarray, labels: np.ndarray | None = None
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    height, width = grid.shape
    out = np.array(grid, copy=True)
    out[out == RED] = BG

    out_labels = None
    if labels is not None:
        out_labels = np.zeros_like(labels)

    for c in range(width):
        static_rows = np.flatnonzero((grid[:, c] != BG) & (grid[:, c] != RED)).tolist()
        boundaries = [-1, *static_rows, height]
        for prev_static, next_static in zip(boundaries, boundaries[1:]):
            start = prev_static + 1
            stop = next_static
            if start >= stop:
                continue

            if labels is None:
                red_rows = np.flatnonzero(grid[start:stop, c] == RED) + start
                count = int(red_rows.size)
                if count:
                    out[stop - count : stop, c] = RED
                continue

            source_rows = [r for r in range(start, stop) if int(labels[r, c]) > 0]
            for dst_r, src_r in zip(
                range(stop - 1, stop - 1 - len(source_rows), -1),
                reversed(source_rows),
            ):
                out[dst_r, c] = RED
                out_labels[dst_r, c] = labels[src_r, c]

    if labels is None:
        return out
    return out, out_labels


def _apply_stripes(grid: np.ndarray, span: tuple[int, int]) -> np.ndarray:
    out = np.array(grid, copy=True)
    c0, c1 = span
    height, _width = out.shape

    for r in range(0, height, 2):
        segment = out[r, c0 : c1 + 1]
        if (r // 2) % 3 == 2:
            row_mask = out[r] != BG
            out[r, row_mask] = GREEN
            segment[segment == BG] = GREEN
        else:
            segment[segment == BG] = YELLOW
        out[r, c0 : c1 + 1] = segment

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Settle the red foreground by column gravity, then recolor it where the propagated stripe cycle turns green.

    Input structure:
    - The grid has black background, one contiguous yellow segment on the top border, and several small red 4-connected objects below it.
    - The yellow segment occupies consecutive columns in row 0 and defines the stripe span.
    - Red pixels may share columns, but only red pixels move; other non-background pixels stay fixed.

    Rule:
    - Keep the top yellow segment fixed.
    - Apply gravity independently in each column to the red pixels: within each vertical compartment bounded by fixed non-red pixels or the grid edge, pack the red pixels to the bottom of that compartment.
    - Starting from row 0, draw one-row stripes on every other row (0, 2, 4, 6, ...), cycling colors yellow, yellow, green.
    - On yellow stripe rows, only paint background cells inside the segment's column span yellow.
    - On green stripe rows, recolor every non-background cell in the whole row green, and also fill background cells inside the segment span green.

    Key insight:
    - Gravity changes only the final row coordinates of the red pixels, while the stripe schedule is fixed by the top segment; their intersection determines exactly which settled pixels get recolored green.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    span = _segment_span(g)
    settled = _settle_red(g)
    return _apply_stripes(settled, span)


def _coords_to_patch(coords: set[tuple[int, int]] | list[tuple[int, int]], value: int = RED) -> np.ndarray:
    pts = np.array(list(coords), dtype=int)
    pts -= pts.min(axis=0)
    height = int(pts[:, 0].max()) + 1
    width = int(pts[:, 1].max()) + 1
    patch = np.zeros((height, width), dtype=int)
    patch[pts[:, 0], pts[:, 1]] = int(value)
    return patch


def _random_polyomino(area: int, *, max_col_count: int | None = None) -> np.ndarray:
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    frontier.add((r + dr, c + dc))
            frontier.difference_update(cells)
            cells.add(random.choice(tuple(frontier)))

        patch = _coords_to_patch(cells)
        col_counts = np.count_nonzero(patch == RED, axis=0)
        if max_col_count is not None and int(col_counts.max()) > int(max_col_count):
            continue
        if patch.shape[0] < 2 or patch.shape[1] < 2:
            continue
        if int(np.count_nonzero(patch == RED)) == int(patch.size):
            continue
        return patch

    raise RuntimeError("failed to generate a valid polyomino")


def _make_special_patch(stem_len: int) -> np.ndarray:
    foot_row = random.randrange(stem_len)
    if random.random() < 0.5:
        coords = {(r, 0) for r in range(stem_len)}
        coords.add((foot_row, 1))
    else:
        coords = {(r, 1) for r in range(stem_len)}
        coords.add((foot_row, 0))
    return _coords_to_patch(coords)


def _can_place(
    grid: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    row_max: int,
    used_cols: set[int],
    *,
    forbidden_span: tuple[int, int] | None = None,
) -> bool:
    height, width = grid.shape
    ph, pw = patch.shape
    if r0 < 1 or c0 < 1 or r0 + ph - 1 > row_max or c0 + pw >= width:
        return False

    fg = np.argwhere(patch == RED)
    obj_cols = {int(c0 + pc) for _pr, pc in fg}
    if obj_cols & used_cols:
        return False
    if forbidden_span is not None:
        s0, s1 = forbidden_span
        if any(s0 <= col <= s1 for col in obj_cols):
            return False

    for pr, pc in fg:
        rr = int(r0 + pr)
        cc = int(c0 + pc)
        if int(grid[rr, cc]) != BG:
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < height and 0 <= nc < width and int(grid[nr, nc]) != BG:
                    return False
    return True


def _place_patch(
    grid: np.ndarray,
    labels: np.ndarray,
    patch: np.ndarray,
    obj_id: int,
    row_max: int,
    used_cols: set[int],
    *,
    forbidden_span: tuple[int, int] | None = None,
) -> bool:
    ph, pw = patch.shape
    height, width = grid.shape
    candidates = []
    for r0 in range(2, row_max - ph + 2):
        for c0 in range(1, width - pw):
            if _can_place(
                grid,
                patch,
                r0,
                c0,
                row_max,
                used_cols,
                forbidden_span=forbidden_span,
            ):
                candidates.append((r0, c0))

    if not candidates:
        return False

    r0, c0 = random.choice(candidates)
    label_patch = np.where(patch == RED, obj_id, 0)
    paste(grid, patch, r0, c0, transparent=BG)
    paste(labels, label_patch, r0, c0, transparent=0)
    used_cols.update(int(c0 + pc) for _pr, pc in np.argwhere(patch == RED))
    return True


def _limited_green_solver(grid: np.ndarray, span: tuple[int, int]) -> np.ndarray:
    out = np.array(grid, copy=True)
    c0, c1 = span
    for r in range(0, out.shape[0], 2):
        segment = out[r, c0 : c1 + 1]
        if (r // 2) % 3 == 2:
            segment[segment != BG] = GREEN
            segment[segment == BG] = GREEN
        else:
            segment[segment == BG] = YELLOW
        out[r, c0 : c1 + 1] = segment
    return out


def _validate_example(grid: np.ndarray, labels: np.ndarray, object_count: int) -> bool:
    red_components = find_components(
        grid,
        bg=BG,
        colors=[RED],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    if len(red_components) != object_count:
        return False
    if not all(3 <= int(comp["area"]) <= 8 for comp in red_components):
        return False

    span = _segment_span(grid)
    if not 3 <= span[1] - span[0] + 1 <= 4:
        return False

    settled, settled_labels = _settle_red(grid, labels)
    expected = _apply_stripes(settled, span)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return False

    if np.array_equal(expected, _apply_stripes(grid, span)):
        return False
    if np.array_equal(expected, _apply_stripes(settled, (0, grid.shape[1] - 1))):
        return False
    if np.array_equal(expected, _limited_green_solver(settled, span)):
        return False

    green_rows = _green_rows(grid.shape[0])
    partial_ids = []
    untouched_ids = []
    recolored_outside_span = False

    for obj_id in range(1, object_count + 1):
        coords = np.argwhere(settled_labels == obj_id)
        if coords.size == 0:
            return False
        rows = {int(r) for r, _c in coords}
        has_green_row = any(r in green_rows for r in rows)
        has_non_green_row = any(r not in green_rows for r in rows)
        if has_green_row and has_non_green_row:
            partial_ids.append(obj_id)
        if not has_green_row:
            untouched_ids.append(obj_id)
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if rr in green_rows and not (span[0] <= cc <= span[1]):
                recolored_outside_span = True

    if not partial_ids or not untouched_ids or not recolored_outside_span:
        return False
    if not np.any(expected == RED):
        return False
    if not np.any(expected == GREEN):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    height = random.choice(GEN_HEIGHT_CHOICES)
    width = random.randint(*GEN_WIDTH_RANGE)
    grid = np.zeros((height, width), dtype=int)
    labels = np.zeros_like(grid)

    segment_width = random.choice(GEN_SEGMENT_WIDTHS)
    segment_start = random.randint(1, width - segment_width - 2)
    segment_span = (segment_start, segment_start + segment_width - 1)
    grid[0, segment_span[0] : segment_span[1] + 1] = YELLOW

    object_count = random.randint(*GEN_OBJECT_COUNT_RANGE)
    row_max = random.randint(*GEN_UPPER_BAND_RANGE)
    stem_len = height - 10
    used_cols: set[int] = set()

    special_patch = _make_special_patch(stem_len)
    if not _place_patch(
        grid,
        labels,
        special_patch,
        1,
        row_max,
        used_cols,
        forbidden_span=segment_span,
    ):
        return None

    try:
        unaffected_patch = _random_polyomino(random.randint(4, 5), max_col_count=stem_len - 1)
    except RuntimeError:
        return None
    if not _place_patch(grid, labels, unaffected_patch, 2, row_max, used_cols):
        return None

    for obj_id in range(3, object_count + 1):
        area = random.randint(3, 6)
        max_col_count = random.randint(2, stem_len)
        try:
            patch = _random_polyomino(area, max_col_count=max_col_count)
        except RuntimeError:
            return None
        if not _place_patch(grid, labels, patch, obj_id, row_max, used_cols):
            return None

    if not _validate_example(grid, labels, object_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
