from __future__ import annotations

import random

import numpy as np


HORIZONTAL_SEPARATOR_COLOR = 2  # red
VERTICAL_SEPARATOR_COLOR = 1  # blue

GRID_MIN = 10
GRID_MAX = 20
MIN_BAND = 3
MAX_BLOB_AREA = 5
GEN_RETRIES = 400

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG_CHOICES = [0, 3, 4, 5, 6, 7, 8, 9]


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _separator_lines(grid: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    row_seps = [r for r in range(h) if np.all(grid[r, :] != bg)]
    col_seps = [c for c in range(w) if np.all(grid[:, c] != bg)]
    assert row_seps or col_seps, "expected at least one separator line"
    return row_seps, col_seps


def _bands(length: int, separators: list[int]) -> list[tuple[int, int]]:
    separators = sorted(separators)
    bands: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        if start <= sep - 1:
            bands.append((start, sep - 1))
        start = sep + 1
    if start <= length - 1:
        bands.append((start, length - 1))
    assert bands, "separator lines cannot cover the whole grid"
    return bands


def _region_plans(
    grid: np.ndarray, bg: int
) -> tuple[list[dict[str, object]], list[int], list[int]]:
    h, w = grid.shape
    row_seps, col_seps = _separator_lines(grid, bg)
    row_bands = _bands(h, row_seps)
    col_bands = _bands(w, col_seps)

    plans: list[dict[str, object]] = []
    for row_idx, (r0, r1) in enumerate(row_bands):
        for col_idx, (c0, c1) in enumerate(col_bands):
            horizontal_candidates: list[tuple[str, int, int]] = []
            vertical_candidates: list[tuple[str, int, int]] = []

            if row_idx > 0:
                horizontal_candidates.append(("top", row_seps[row_idx - 1], row_idx - 1))
            if row_idx < len(row_seps):
                horizontal_candidates.append(("bottom", row_seps[row_idx], row_idx + 1))

            if col_idx > 0:
                vertical_candidates.append(("left", col_seps[col_idx - 1], col_idx - 1))
            if col_idx < len(col_seps):
                vertical_candidates.append(("right", col_seps[col_idx], col_idx + 1))

            axis = None
            side = None
            separator = None
            dst_rows = (r0, r1)
            dst_cols = (c0, c1)

            if horizontal_candidates:
                assert len(horizontal_candidates) == 1, "horizontal adjacency is ambiguous"
                side, separator, target_row_idx = horizontal_candidates[0]
                dst_rows = row_bands[target_row_idx]
                axis = "h"
            elif vertical_candidates:
                assert len(vertical_candidates) == 1, "vertical adjacency is ambiguous"
                side, separator, target_col_idx = vertical_candidates[0]
                dst_cols = col_bands[target_col_idx]
                axis = "v"

            plans.append(
                {
                    "src_rows": (r0, r1),
                    "src_cols": (c0, c1),
                    "dst_rows": dst_rows,
                    "dst_cols": dst_cols,
                    "axis": axis,
                    "side": side,
                    "separator": separator,
                }
            )
    return plans, row_seps, col_seps


def _reflect_interval(src0: int, src1: int, sep: int, dst0: int, dst1: int) -> tuple[int, int]:
    lo = max(src0, 2 * sep - dst1)
    hi = min(src1, 2 * sep - dst0)
    assert lo <= hi, "expected at least one source cell that reflects into the target band"
    return lo, hi


def _safe_window(plan: dict[str, object]) -> tuple[int, int, int, int]:
    r0, r1 = plan["src_rows"]  # type: ignore[assignment]
    c0, c1 = plan["src_cols"]  # type: ignore[assignment]
    axis = plan["axis"]
    if axis == "h":
        sep = int(plan["separator"])
        tr0, tr1 = plan["dst_rows"]  # type: ignore[assignment]
        safe_r0, safe_r1 = _reflect_interval(r0, r1, sep, tr0, tr1)
        return safe_r0, safe_r1, c0, c1
    if axis == "v":
        sep = int(plan["separator"])
        tc0, tc1 = plan["dst_cols"]  # type: ignore[assignment]
        safe_c0, safe_c1 = _reflect_interval(c0, c1, sep, tc0, tc1)
        return r0, r1, safe_c0, safe_c1
    return r0, r1, c0, c1


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror each region's contents across its adjacent separator line.

    Input structure
    - A dominant background color fills the canvas.
    - Full non-background rows/columns act as separator lines.
    - Sparse foreground objects sit inside the rectangular regions between separators.

    Rule
    - Keep separator lines unchanged on the output canvas.
    - For each region, mirror its contents across an adjacent horizontal separator if one exists.
    - Otherwise mirror across an adjacent vertical separator.
    - When both are adjacent, the horizontal separator takes priority.
    - Erase the source objects; later regions in row-major order overwrite earlier reflected pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    plans, row_seps, col_seps = _region_plans(g, bg)

    out = np.full_like(g, bg)
    if row_seps:
        out[row_seps, :] = g[row_seps, :]
    if col_seps:
        out[:, col_seps] = g[:, col_seps]

    row_sep_set = set(row_seps)
    col_sep_set = set(col_seps)

    for plan in plans:
        r0, r1 = plan["src_rows"]  # type: ignore[assignment]
        c0, c1 = plan["src_cols"]  # type: ignore[assignment]
        axis = plan["axis"]
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        local_points = np.argwhere(region != bg)

        if axis is None:
            assert local_points.size == 0, "regions without separator guidance must be empty"
            continue
        if local_points.size == 0:
            continue

        sep = int(plan["separator"])
        dst_r0, dst_r1 = plan["dst_rows"]  # type: ignore[assignment]
        dst_c0, dst_c1 = plan["dst_cols"]  # type: ignore[assignment]

        for lr, lc in local_points:
            sr = r0 + int(lr)
            sc = c0 + int(lc)
            if axis == "h":
                rr = 2 * sep - sr
                cc = sc
            else:
                rr = sr
                cc = 2 * sep - sc

            assert dst_r0 <= rr <= dst_r1 and dst_c0 <= cc <= dst_c1, "reflected pixel left its target region"
            assert rr not in row_sep_set and cc not in col_sep_set, "reflected pixel hit a separator"
            out[rr, cc] = int(g[sr, sc])

    return out


def _draw_separators(grid: np.ndarray, row_sep: int | None, col_sep: int | None) -> None:
    if row_sep is not None:
        grid[row_sep, :] = HORIZONTAL_SEPARATOR_COLOR
    if col_sep is not None:
        grid[:, col_sep] = VERTICAL_SEPARATOR_COLOR
    if row_sep is not None and col_sep is not None:
        grid[row_sep, col_sep] = random.choice(
            [HORIZONTAL_SEPARATOR_COLOR, VERTICAL_SEPARATOR_COLOR]
        )


def _is_far_from_occupied(cell: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    r, c = cell
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if (rr, cc) in occupied:
                return False
    return True


def _grow_blob(
    window: tuple[int, int, int, int],
    occupied: set[tuple[int, int]],
    target_area: int,
    require_gap: bool,
) -> set[tuple[int, int]] | None:
    r0, r1, c0, c1 = window
    seeds = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) in occupied:
                continue
            if require_gap and not _is_far_from_occupied((r, c), occupied):
                continue
            seeds.append((r, c))
    if not seeds:
        return None

    blob = {random.choice(seeds)}
    while len(blob) < target_area:
        frontier: list[tuple[int, int]] = []
        for r, c in blob:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                    continue
                if (nr, nc) in blob or (nr, nc) in occupied:
                    continue
                if require_gap and not _is_far_from_occupied((nr, nc), occupied):
                    continue
                frontier.append((nr, nc))
        if not frontier:
            break
        blob.add(random.choice(frontier))

    return blob if blob else None


def _populate_region(
    grid: np.ndarray,
    plan: dict[str, object],
    palette: list[int],
) -> bool:
    window = _safe_window(plan)
    wr0, wr1, wc0, wc1 = window
    capacity = (wr1 - wr0 + 1) * (wc1 - wc0 + 1)
    assert capacity > 0

    object_count = 1
    if capacity >= 12 and random.random() < 0.6:
        object_count = 2

    occupied: set[tuple[int, int]] = set()
    colors = random.sample(palette, k=min(object_count, len(palette)))
    while len(colors) < object_count:
        colors.append(random.choice(palette))

    for idx in range(object_count):
        remaining = object_count - idx
        free_budget = capacity - len(occupied) - (remaining - 1)
        max_area = min(MAX_BLOB_AREA, max(1, free_budget))
        blob = None

        for require_gap in (True, False):
            for _ in range(50):
                target_area = random.randint(1, max_area)
                candidate = _grow_blob(window, occupied, target_area, require_gap)
                if candidate is None:
                    continue
                blob = candidate
                break
            if blob is not None:
                break

        if blob is None:
            return False

        color = colors[idx]
        for r, c in blob:
            grid[r, c] = color
        occupied |= blob

    return True


def _attempt_input() -> np.ndarray | None:
    mode = random.choice(["horizontal", "vertical", "cross"])
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    bg = random.choice(BG_CHOICES)

    if mode in {"horizontal", "cross"} and h < 2 * MIN_BAND + 1:
        return None
    if mode in {"vertical", "cross"} and w < 2 * MIN_BAND + 1:
        return None

    row_sep = None
    col_sep = None
    if mode in {"horizontal", "cross"}:
        row_sep = random.randint(MIN_BAND, h - MIN_BAND - 1)
    if mode in {"vertical", "cross"}:
        col_sep = random.randint(MIN_BAND, w - MIN_BAND - 1)

    grid = np.full((h, w), bg, dtype=int)
    _draw_separators(grid, row_sep, col_sep)

    if _get_bg(grid) != bg:
        return None

    try:
        plans, _row_seps, _col_seps = _region_plans(grid, bg)
    except AssertionError:
        return None

    palette = [c for c in range(10) if c not in {bg, HORIZONTAL_SEPARATOR_COLOR, VERTICAL_SEPARATOR_COLOR}]
    random.shuffle(palette)

    for plan in plans:
        if plan["axis"] is None:
            return None
        if not _populate_region(grid, plan, palette):
            return None

    if _get_bg(grid) != bg:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
