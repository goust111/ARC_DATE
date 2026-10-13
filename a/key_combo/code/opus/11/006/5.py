from __future__ import annotations

import numpy as np


BG = 0
FRAME_COLOR = 9
GEN_HEIGHT = 20
GEN_INTERNAL_WIDTHS = (8, 10, 12)
GEN_ATTEMPTS = 400
GEN_SHAPE_ATTEMPTS = 200
GEN_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)

LEFT = "left"
RIGHT = "right"
SPLIT = "split"
LEFT_EDGE = "left_edge"
RIGHT_EDGE = "right_edge"


def _marker_positions(config: str, row: int, inner_width: int) -> set[tuple[int, int]]:
    if config == LEFT_EDGE:
        return {(row, 0), (row, 1)}
    if config == RIGHT_EDGE:
        return {(row, inner_width - 2), (row, inner_width - 1)}
    if config == SPLIT:
        return {(row, 0), (row, inner_width - 1)}
    raise AssertionError(f"unknown marker config: {config}")


def _config_from_cols(cols: list[int], inner_width: int) -> str | None:
    if cols == [0, 1]:
        return LEFT_EDGE
    if cols == [inner_width - 2, inner_width - 1]:
        return RIGHT_EDGE
    if cols == [0, inner_width - 1]:
        return SPLIT
    return None


def _target_side(config: str, ref_side: str) -> str:
    if config == LEFT_EDGE:
        return LEFT
    if config == RIGHT_EDGE:
        return RIGHT
    if config == SPLIT:
        return ref_side
    raise AssertionError(f"unknown marker config: {config}")


def _render_half(
    local_cells: tuple[tuple[int, int], ...],
    center_row: int,
    side: str,
    half_width: int,
) -> set[tuple[int, int]]:
    base_col = 0 if side == LEFT else half_width
    return {(center_row + dr, base_col + dc) for dr, dc in local_cells}


def _render_full(
    local_cells: tuple[tuple[int, int], ...],
    center_row: int,
    side: str,
    half_width: int,
) -> set[tuple[int, int]]:
    inner_width = 2 * half_width
    half_cells = _render_half(local_cells, center_row, side, half_width)
    mirrored = {(r, inner_width - 1 - c) for r, c in half_cells}
    return half_cells | mirrored


def _mask_to_local_cells(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    mid = mask.shape[0] // 2
    cells = [(int(r - mid), int(c)) for r, c in np.argwhere(mask)]
    return tuple(sorted(cells))


def _local_mask(local_cells: tuple[tuple[int, int], ...], half_width: int) -> np.ndarray:
    drs = [dr for dr, _ in local_cells]
    min_dr = min(drs)
    max_dr = max(drs)
    height = max_dr - min_dr + 1
    mask = np.zeros((height, half_width), dtype=bool)
    for dr, dc in local_cells:
        mask[dr - min_dr, dc] = True
    return mask


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape
    assert width >= 6
    inner_width = width - 2
    assert inner_width % 2 == 0
    half_width = inner_width // 2

    frame_color = int(g[0, 0])
    assert frame_color != BG
    assert np.all(g[:, 0] == frame_color)
    assert np.all(g[:, -1] == frame_color)

    interior = g[:, 1:-1]
    colors = [int(c) for c in np.unique(interior) if int(c) not in (BG, frame_color)]
    assert colors

    ref_color: int | None = None
    ref_side: str | None = None
    ref_center: int | None = None
    local_cells: tuple[tuple[int, int], ...] | None = None
    jobs: list[tuple[int, int, str]] = []

    for color in colors:
        coords = np.argwhere(interior == color)
        assert coords.size > 0
        rows = sorted({int(r) for r, _ in coords})
        cols = sorted(int(c) for _, c in coords)

        if len(coords) == 2 and len(rows) == 1:
            config = _config_from_cols(cols, inner_width)
            if config is not None:
                jobs.append((color, rows[0], config))
                continue

        assert ref_color is None, "expected exactly one reference half-glyph color"
        assert len(coords) > 2, "reference half-glyph must contain more than two cells"
        if np.all(coords[:, 1] < half_width):
            ref_side = LEFT
            col_offset = 0
        elif np.all(coords[:, 1] >= half_width):
            ref_side = RIGHT
            col_offset = half_width
        else:
            raise AssertionError("reference half-glyph must lie entirely in one half")

        r0 = int(coords[:, 0].min())
        r1 = int(coords[:, 0].max())
        assert (r1 - r0) % 2 == 0, "reference bbox height must be odd"
        ref_center = (r0 + r1) // 2
        ref_color = color
        local_cells = tuple(
            sorted((int(r - ref_center), int(c - col_offset)) for r, c in coords)
        )

    assert ref_color is not None
    assert ref_side is not None
    assert ref_center is not None
    assert local_cells is not None
    assert 1 <= len(jobs) <= 2

    mask = _local_mask(local_cells, half_width)
    assert not np.array_equal(mask, np.fliplr(mask)), "half template must not be left-right symmetric"

    return {
        "height": height,
        "width": width,
        "inner_width": inner_width,
        "half_width": half_width,
        "frame_color": frame_color,
        "ref_color": ref_color,
        "ref_side": ref_side,
        "ref_center": ref_center,
        "local_cells": local_cells,
        "jobs": sorted(jobs),
    }


def _build_output(parsed: dict) -> np.ndarray:
    height = parsed["height"]
    width = parsed["width"]
    half_width = parsed["half_width"]
    frame_color = parsed["frame_color"]
    ref_side = parsed["ref_side"]
    local_cells = parsed["local_cells"]
    jobs = parsed["jobs"]

    out = np.full((height, width), BG, dtype=int)
    out[:, 0] = frame_color
    out[:, -1] = frame_color

    claimed: dict[tuple[int, int], int] = {}
    for color, row, config in jobs:
        target = _target_side(config, ref_side)
        cells = _render_full(local_cells, row, target, half_width)
        for r, c in cells:
            assert 0 <= r < height
            assert 0 <= c < 2 * half_width
            prev = claimed.get((r, c))
            assert prev is None or prev == color, "reconstructed glyphs must not overlap"
            claimed[(r, c)] = color

    for (r, c), color in claimed.items():
        out[r, c + 1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Erase the reference half-glyph and rebuild mirrored full glyphs at rows encoded by edge marker pairs.

    Input structure:
    - The grid has a black interior bounded by two full-height maroon frame bars.
    - Inside the frame there is one colored reference half-glyph that lies entirely in the left or right half.
    - One or two other colors appear only as two-pixel marker pairs on a single row: both at the left edge, both at the right edge, or split across the two edges.

    Rule:
    - Read the reference half-glyph as a local shape inside one half of the interior, then erase it.
    - For each marker pair, align the half-glyph's vertical center to the marker row and place the observed half on the side encoded by the marker columns.
    - Mirror that placed half across the interior vertical axis to complete a full symmetric glyph in the marker color; the frame stays unchanged and the marker pixels are covered.

    Key insight:
    - The marker pair does not name a shape by color; its edge placement tells you which side should receive the observed half before mirroring completes the figure.
    """
    parsed = _analyze_input(np.asarray(grid, dtype=int))
    return _build_output(parsed)


def _random_half_mask(rng: np.random.Generator, half_width: int) -> np.ndarray:
    for _ in range(GEN_SHAPE_ATTEMPTS):
        height = int(rng.choice((5, 7, 9)))
        mid = height // 2
        mask = np.zeros((height, half_width), dtype=bool)
        mask[mid, :] = True

        top_col = int(rng.integers(0, half_width))
        top_len = int(rng.integers(1, mid + 1))
        mask[mid - top_len : mid, top_col] = True

        bot_col = int(rng.integers(0, half_width))
        bot_len = int(rng.integers(1, mid + 1))
        mask[mid + 1 : mid + 1 + bot_len, bot_col] = True

        def add_cap(row: int, col: int) -> None:
            dirs = []
            if col > 0:
                dirs.append(-1)
            if col < half_width - 1:
                dirs.append(1)
            if not dirs or rng.random() < 0.2:
                return
            direction = dirs[int(rng.integers(0, len(dirs)))]
            max_span = col if direction < 0 else (half_width - 1 - col)
            span = int(rng.integers(1, min(2, max_span) + 1))
            for step in range(1, span + 1):
                mask[row, col + direction * step] = True

        add_cap(mid - top_len, top_col)
        add_cap(mid + bot_len, bot_col)

        branch_rows = [r for r in range(height) if r != mid and mask[r].any()]
        if branch_rows and rng.random() < 0.8:
            row = int(branch_rows[int(rng.integers(0, len(branch_rows)))])
            occupied = np.flatnonzero(mask[row])
            edge_col = int(occupied[0] if rng.random() < 0.5 else occupied[-1])
            dirs = []
            if edge_col > 0:
                dirs.append(-1)
            if edge_col < half_width - 1:
                dirs.append(1)
            if dirs:
                direction = dirs[int(rng.integers(0, len(dirs)))]
                mask[row, edge_col + direction] = True

        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.count_nonzero(mask) < half_width + 3:
            continue
        return mask
    raise RuntimeError("failed to generate a non-degenerate half-glyph")


def _empty_grid(height: int, inner_width: int) -> np.ndarray:
    grid = np.full((height, inner_width + 2), BG, dtype=int)
    grid[:, 0] = FRAME_COLOR
    grid[:, -1] = FRAME_COLOR
    return grid


def _job_rows(
    height: int,
    local_cells: tuple[tuple[int, int], ...],
    half_width: int,
    ref_side: str,
    configs: list[str],
    rng: np.random.Generator,
) -> list[int] | None:
    drs = [dr for dr, _ in local_cells]
    valid_rows = list(range(-min(drs), height - max(drs)))

    if len(configs) == 1:
        return [int(rng.choice(valid_rows))]

    choices: list[tuple[int, int]] = []
    for row0 in valid_rows:
        cells0 = _render_full(local_cells, row0, _target_side(configs[0], ref_side), half_width)
        for row1 in valid_rows:
            if row0 == row1:
                continue
            cells1 = _render_full(local_cells, row1, _target_side(configs[1], ref_side), half_width)
            if cells0.isdisjoint(cells1):
                choices.append((row0, row1))

    if not choices:
        return None
    row0, row1 = choices[int(rng.integers(0, len(choices)))]
    return [int(row0), int(row1)]


def _reference_center(
    height: int,
    local_cells: tuple[tuple[int, int], ...],
    half_width: int,
    ref_side: str,
    marker_cells: set[tuple[int, int]],
    rng: np.random.Generator,
) -> int | None:
    drs = [dr for dr, _ in local_cells]
    valid_rows = list(range(-min(drs), height - max(drs)))
    good = []
    for row in valid_rows:
        ref_cells = _render_half(local_cells, row, ref_side, half_width)
        if ref_cells.isdisjoint(marker_cells):
            good.append(row)
    if not good:
        return None
    return int(rng.choice(good))


def _metadata_output(meta: dict) -> np.ndarray:
    parsed = {
        "height": meta["height"],
        "width": meta["inner_width"] + 2,
        "half_width": meta["half_width"],
        "frame_color": FRAME_COLOR,
        "ref_side": meta["ref_side"],
        "local_cells": meta["local_cells"],
        "jobs": meta["jobs"],
    }
    return _build_output(parsed)


def _validate_generated(inp: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    try:
        parsed = _analyze_input(inp)
        solved = main(inp)
    except Exception:
        return False

    if not np.array_equal(solved, expected):
        return False
    if not np.array_equal(solved, _metadata_output(meta)):
        return False

    if parsed["ref_side"] != meta["ref_side"]:
        return False
    if parsed["local_cells"] != meta["local_cells"]:
        return False
    if parsed["jobs"] != sorted(meta["jobs"]):
        return False

    left_cells = _render_full(meta["local_cells"], 10, LEFT, meta["half_width"])
    right_cells = _render_full(meta["local_cells"], 10, RIGHT, meta["half_width"])
    if left_cells == right_cells:
        return False

    claimed: set[tuple[int, int]] = set()
    for _color, row, config in meta["jobs"]:
        cells = _render_full(
            meta["local_cells"],
            row,
            _target_side(config, meta["ref_side"]),
            meta["half_width"],
        )
        if claimed & cells:
            return False
        claimed |= cells

    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    height = GEN_HEIGHT
    inner_width = int(rng.choice(GEN_INTERNAL_WIDTHS))
    half_width = inner_width // 2

    half_mask = _random_half_mask(rng, half_width)
    local_cells = _mask_to_local_cells(half_mask)
    ref_side = LEFT if rng.random() < 0.5 else RIGHT
    opposite_config = RIGHT_EDGE if ref_side == LEFT else LEFT_EDGE

    if rng.random() < 0.7:
        configs = [SPLIT, opposite_config]
    else:
        configs = [SPLIT if rng.random() < 0.5 else opposite_config]

    rows = _job_rows(height, local_cells, half_width, ref_side, configs, rng)
    if rows is None:
        return None

    color_count = 1 + len(configs)
    picked = [int(x) for x in rng.choice(np.array(GEN_COLORS), size=color_count, replace=False)]
    ref_color = picked[0]
    marker_colors = picked[1:]
    jobs = [(marker_colors[i], rows[i], configs[i]) for i in range(len(configs))]

    marker_cells = set()
    for _color, row, config in jobs:
        marker_cells |= _marker_positions(config, row, inner_width)

    ref_center = _reference_center(height, local_cells, half_width, ref_side, marker_cells, rng)
    if ref_center is None:
        return None

    inp = _empty_grid(height, inner_width)
    for r, c in _render_half(local_cells, ref_center, ref_side, half_width):
        inp[r, c + 1] = ref_color
    for color, row, config in jobs:
        for r, c in _marker_positions(config, row, inner_width):
            inp[r, c + 1] = color

    meta = {
        "height": height,
        "inner_width": inner_width,
        "half_width": half_width,
        "ref_side": ref_side,
        "ref_color": ref_color,
        "ref_center": ref_center,
        "local_cells": local_cells,
        "jobs": jobs,
    }
    expected = _metadata_output(meta)

    if not _validate_generated(inp, expected, meta):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid half-glyph mirror inputs")
