from __future__ import annotations

import random

import numpy as np


MAX_GENERATION_ATTEMPTS = 400
PALETTE = tuple(range(10))

CHAIN_LENGTH_RANGE = (3, 7)
MAX_HEIGHT_IN_CHAIN = 18
CANVAS_EXTRA_HEIGHT_RANGE = (2, 8)
SIDE_MARGIN_RANGE = (0, 3)
COLUMN_STEP_RANGE = (2, 4)
LAYOUT_MODES = ("bottom", "top", "random")


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _half_slice(top: int, height: int, half: str) -> slice:
    mid = top + height // 2
    if half == "top":
        return slice(top, mid)
    assert half == "bottom"
    return slice(mid, top + height)


def _parse_scene(grid: np.ndarray) -> tuple[int, int, int, list[dict[str, int | str | None]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h > 0 and w > 0

    bg = _get_bg(g)
    fg_colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert len(fg_colors) == 2

    fg_cols = [c for c in range(w) if np.any(g[:, c] != bg)]
    assert len(fg_cols) >= 2

    col_sets = {
        color: {c for c in fg_cols if np.any(g[:, c] == color)}
        for color in fg_colors
    }
    rod_candidates = [color for color, cols in col_sets.items() if cols == set(fg_cols)]
    assert len(rod_candidates) == 1
    rod_color = int(rod_candidates[0])
    mark_color = int(next(color for color in fg_colors if color != rod_color))

    rods: list[dict[str, int | str | None]] = []
    seen_heights: set[int] = set()
    for c in fg_cols:
        rows = np.flatnonzero(g[:, c] != bg)
        assert rows.size > 0
        r0 = int(rows[0])
        r1 = int(rows[-1])
        expected_rows = np.arange(r0, r1 + 1)
        assert np.array_equal(rows, expected_rows)

        segment = g[r0 : r1 + 1, c]
        assert set(int(v) for v in np.unique(segment)) <= {rod_color, mark_color}
        height = int(segment.size)
        assert height % 2 == 0
        assert height not in seen_heights
        seen_heights.add(height)

        marker_rows = np.flatnonzero(segment == mark_color)
        if marker_rows.size == 0:
            marked_half = None
        else:
            assert marker_rows.size == height // 2
            top_half = np.arange(0, height // 2)
            bottom_half = np.arange(height // 2, height)
            if np.array_equal(marker_rows, top_half):
                marked_half = "top"
            else:
                assert np.array_equal(marker_rows, bottom_half)
                marked_half = "bottom"

        rods.append(
            {
                "col": int(c),
                "top": r0,
                "bottom": r1,
                "height": height,
                "half": marked_half,
            }
        )

    marked_rods = [rod for rod in rods if rod["half"] is not None]
    assert marked_rods

    rods_by_height = {int(rod["height"]): rod for rod in rods}
    for rod in marked_rods:
        target_height = int(rod["height"]) + 2
        assert target_height in rods_by_height
        assert rods_by_height[target_height]["half"] is None

    return bg, rod_color, mark_color, rods


def main(grid: np.ndarray) -> np.ndarray:
    """Move each marked half-segment to the unique pure rod whose height is two pixels larger."""
    g = np.asarray(grid, dtype=int)
    bg, rod_color, mark_color, rods = _parse_scene(g)

    out = g.copy()
    rods_by_height = {int(rod["height"]): rod for rod in rods}
    for rod in rods:
        half = rod["half"]
        if half is None:
            continue

        src_top = int(rod["top"])
        src_height = int(rod["height"])
        src_col = int(rod["col"])
        target = rods_by_height[src_height + 2]
        dst_top = int(target["top"])
        dst_height = int(target["height"])
        dst_col = int(target["col"])

        out[_half_slice(src_top, src_height, str(half)), src_col] = rod_color
        out[_half_slice(dst_top, dst_height, str(half)), dst_col] = mark_color

    assert _get_bg(out) == bg
    return out


def _valid_mark_index_sets(num_rods: int) -> list[tuple[int, ...]]:
    usable = num_rods - 1
    valid: list[tuple[int, ...]] = []
    for mask in range(1, 1 << usable):
        idxs = tuple(i for i in range(usable) if (mask >> i) & 1)
        if all(b - a > 1 for a, b in zip(idxs, idxs[1:])):
            valid.append(idxs)
    assert valid
    return valid


def _pick_heights(num_rods: int) -> list[int]:
    max_start = MAX_HEIGHT_IN_CHAIN - 2 * (num_rods - 1)
    starts = list(range(2, max_start + 1, 2))
    assert starts
    start = random.choice(starts)
    return [start + 2 * i for i in range(num_rods)]


def _pick_columns(num_rods: int) -> tuple[list[int], int]:
    cols = [random.randint(*SIDE_MARGIN_RANGE)]
    for _ in range(num_rods - 1):
        cols.append(cols[-1] + random.randint(*COLUMN_STEP_RANGE))
    width = cols[-1] + 1 + random.randint(*SIDE_MARGIN_RANGE)
    return cols, width


def _pick_tops(mode: str, heights: list[int], canvas_height: int) -> dict[int, int]:
    max_height = max(heights)
    if mode == "bottom":
        bottom = random.randint(max_height - 1, canvas_height - 1)
        return {height: bottom - height + 1 for height in heights}
    if mode == "top":
        top = random.randint(0, canvas_height - max_height)
        return {height: top for height in heights}
    assert mode == "random"
    return {height: random.randint(0, canvas_height - height) for height in heights}


def _gen_attempt() -> np.ndarray | None:
    num_rods = random.randint(*CHAIN_LENGTH_RANGE)
    heights = _pick_heights(num_rods)
    max_height = max(heights)

    canvas_height = max_height + random.randint(*CANVAS_EXTRA_HEIGHT_RANGE)
    cols, canvas_width = _pick_columns(num_rods)

    bg, rod_color, mark_color = random.sample(PALETTE, 3)
    marked_idxs = random.choice(_valid_mark_index_sets(num_rods))
    marked_heights = {heights[idx] for idx in marked_idxs}
    marked_halves = {
        height: random.choice(("top", "bottom"))
        for height in marked_heights
    }

    shuffled_heights = heights[:]
    random.shuffle(shuffled_heights)
    tops = _pick_tops(random.choice(LAYOUT_MODES), shuffled_heights, canvas_height)

    grid = np.full((canvas_height, canvas_width), int(bg), dtype=int)
    for col, height in zip(cols, shuffled_heights, strict=False):
        top = tops[height]
        grid[top : top + height, col] = rod_color
        half = marked_halves.get(height)
        if half is not None:
            grid[_half_slice(top, height, half), col] = mark_color

    try:
        parsed_bg, parsed_rod, parsed_mark, rods = _parse_scene(grid)
        assert parsed_bg == bg
        assert parsed_rod == rod_color
        assert parsed_mark == mark_color
        parsed_marks = {
            int(rod["height"]): str(rod["half"])
            for rod in rods
            if rod["half"] is not None
        }
        assert parsed_marks == marked_halves
        out = main(grid)
        assert isinstance(out, np.ndarray)
        assert out.shape == grid.shape
        assert not np.array_equal(out, grid)
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_GENERATION_ATTEMPTS} attempts")
