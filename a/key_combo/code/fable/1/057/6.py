from __future__ import annotations

import math
import random

import numpy as np


BG_COLOR = 0
FILL_COLOR = 8
INPUT_PALETTE = (1, 2, 3, 4, 6, 7, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return components


def _normalized_mask(grid: np.ndarray, component: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    return grid[r0 : r1 + 1, c0 : c1 + 1] != bg


def _gcd_diffs(values: list[int]) -> int:
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    step = diffs[0]
    for diff in diffs[1:]:
        step = math.gcd(step, diff)
    return int(step)


def _infer_step(values: list[int]) -> int:
    uniq = sorted(set(int(v) for v in values))
    assert len(uniq) >= 3, "need at least three visible line coordinates to infer the arithmetic step"
    step = _gcd_diffs(uniq)
    assert step > 0, "arithmetic step must be positive"
    assert all((b - a) % step == 0 for a, b in zip(uniq, uniq[1:])), "all coordinates must share one step"
    return step


def _extend_sequence(values: list[int], step: int, low: int, high: int) -> list[int]:
    assert values
    uniq = sorted(set(int(v) for v in values))
    assert low <= uniq[0] <= uniq[-1] <= high
    start = uniq[0]
    while start - step >= low:
        start -= step

    seq: list[int] = []
    cur = start
    while cur <= high:
        seq.append(cur)
        cur += step

    assert set(uniq).issubset(seq), "visible coordinates must lie on the restored sequence"
    assert seq[0] - step < low and seq[-1] + step > high, "sequence must be maximally extended to canvas bounds"
    return seq


def _paint_sprite(dst: np.ndarray, sprite_mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = sprite_mask.shape
    window = dst[top : top + h, left : left + w]
    window[sprite_mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the hidden arithmetic row-and-column lattice of identical sprites and stamp cyan copies on every missing lattice point.

    Input structure:
    - A black background contains several separated monochrome sprites.
    - Every sprite has the same shape and color, and each sprite bbox top-left is the sprite's lattice anchor.
    - The visible row anchors are a subset of one arithmetic progression and the visible column anchors are a subset of another, with whole lattice rows or columns absent from the drawing.

    Rule:
    - Detect the identical sprites and collect their bbox top-left row and column coordinates.
    - Infer the row step and column step from those arithmetic samples, fill any missing interior terms, and extend both sequences to the furthest top-left positions where the same sprite still fits fully inside the canvas.
    - Keep existing sprites unchanged, and place a cyan copy of the same sprite at every empty point in the restored row-by-column cartesian lattice.

    Key insight:
    - The target grid is not determined by the visible row and column sets alone; the solver must first reconstruct the hidden arithmetic lattice lines, including unseen whole rows or columns and the bounded extensions at both ends.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(np.bincount(g.ravel()).argmax())
    assert bg == BG_COLOR, "expected a black background"

    components = _find_components(g, bg)
    assert len(components) >= 5, "expected multiple visible sprites"

    colors = {comp["color"] for comp in components}
    assert len(colors) == 1, "all visible sprites must share one color"
    fg_color = next(iter(colors))
    assert fg_color != FILL_COLOR, "input sprite color must differ from the cyan fill color"

    sprite_mask = _normalized_mask(g, components[0], bg)
    sprite_h, sprite_w = sprite_mask.shape
    assert sprite_mask.any(), "sprite mask cannot be empty"
    assert sprite_mask[0].any() and sprite_mask[-1].any(), "sprite must touch the top and bottom of its bbox"
    assert sprite_mask[:, 0].any() and sprite_mask[:, -1].any(), "sprite must touch the left and right of its bbox"

    positions: set[tuple[int, int]] = set()
    for component in components:
        assert np.array_equal(_normalized_mask(g, component, bg), sprite_mask), "all components must be identical sprites"
        r0, c0, r1, c1 = component["bbox"]
        assert (r1 - r0 + 1, c1 - c0 + 1) == (sprite_h, sprite_w), "all sprite bboxes must match"
        positions.add((int(r0), int(c0)))

    rows_obs = sorted({r for r, _c in positions})
    cols_obs = sorted({c for _r, c in positions})
    row_step = _infer_step(rows_obs)
    col_step = _infer_step(cols_obs)

    full_rows = _extend_sequence(rows_obs, row_step, 0, g.shape[0] - sprite_h)
    full_cols = _extend_sequence(cols_obs, col_step, 0, g.shape[1] - sprite_w)

    full_positions = {(r, c) for r in full_rows for c in full_cols}
    assert positions < full_positions, "at least one lattice point must be missing from the input"
    assert rows_obs[0] != full_rows[0] and rows_obs[-1] != full_rows[-1], "row sequence must extend beyond visible extremes"
    assert cols_obs[0] != full_cols[0] and cols_obs[-1] != full_cols[-1], "column sequence must extend beyond visible extremes"

    out = g.copy()
    for r, c in full_positions:
        if (r, c) not in positions:
            _paint_sprite(out, sprite_mask, r, c, FILL_COLOR)
    return out


SPRITE_LIBRARY = (
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
)
GEN_RETRIES = 400


def _random_sprite() -> np.ndarray:
    sprite = np.array(random.choice(SPRITE_LIBRARY), dtype=bool)
    sprite = np.rot90(sprite, random.randrange(4))
    if random.random() < 0.5:
        sprite = np.fliplr(sprite)
    if random.random() < 0.5:
        sprite = np.flipud(sprite)
    return sprite.copy()


def _choose_visible_indices(n_total: int, hide_interior: bool) -> list[int]:
    interior = list(range(1, n_total - 1))
    if not hide_interior:
        return interior
    for _ in range(200):
        count = random.randint(3, len(interior) - 1)
        vis = sorted(random.sample(interior, count))
        if _gcd_diffs(vis) == 1:
            return vis
    raise RuntimeError("failed to choose visible arithmetic indices")


def _choose_step(sprite_extent: int, n_total: int) -> int:
    step_min = sprite_extent + 1
    step_max = min(6, (30 - sprite_extent) // (n_total - 1))
    assert step_min <= step_max, "no feasible arithmetic step for this sprite size"
    return random.randint(step_min, step_max)


def _build_full_line(sprite_extent: int, n_total: int, step: int) -> tuple[list[int], int]:
    base_size = (n_total - 1) * step + sprite_extent
    extra_budget = 30 - base_size
    start = random.randint(0, min(step - 1, extra_budget))
    end_slack = random.randint(0, min(step - 1, extra_budget - start))
    total_size = base_size + start + end_slack
    line = [start + i * step for i in range(n_total)]
    return line, total_size


def _choose_observed_cells(row_ids: list[int], col_ids: list[int]) -> set[tuple[int, int]]:
    chosen: set[tuple[int, int]] = set()
    for rid in row_ids:
        chosen.add((rid, random.choice(col_ids)))
    for cid in col_ids:
        if all(cc != cid for _rr, cc in chosen):
            chosen.add((random.choice(row_ids), cid))

    all_cells = [(rid, cid) for rid in row_ids for cid in col_ids if (rid, cid) not in chosen]
    random.shuffle(all_cells)

    total = len(row_ids) * len(col_ids)
    min_count = max(len(row_ids), len(col_ids)) + 1
    max_count = total - max(1, total // 4)
    max_count = max(max_count, min_count)
    target = random.randint(min_count, max_count)
    while len(chosen) < target and all_cells:
        chosen.add(all_cells.pop())
    return chosen


def _render_input(
    shape: tuple[int, int],
    sprite_mask: np.ndarray,
    input_color: int,
    observed_positions: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.full(shape, BG_COLOR, dtype=int)
    for r, c in observed_positions:
        _paint_sprite(grid, sprite_mask, r, c, input_color)
    return grid


def _render_output(
    shape: tuple[int, int],
    sprite_mask: np.ndarray,
    input_color: int,
    rows: list[int],
    cols: list[int],
    observed_positions: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.full(shape, BG_COLOR, dtype=int)
    for r in rows:
        for c in cols:
            color = input_color if (r, c) in observed_positions else FILL_COLOR
            _paint_sprite(grid, sprite_mask, r, c, color)
    return grid


def _validate_instance(
    inp: np.ndarray,
    sprite_mask: np.ndarray,
    input_color: int,
    rows_all: list[int],
    cols_all: list[int],
    observed_positions: set[tuple[int, int]],
    mode: str,
) -> None:
    components = _find_components(inp, BG_COLOR)
    assert len(components) == len(observed_positions), "sprites must stay separated in the input"
    assert all(comp["color"] == input_color for comp in components), "input must use one sprite color"

    rows_obs = sorted({comp["bbox"][0] for comp in components})
    cols_obs = sorted({comp["bbox"][1] for comp in components})
    assert len(rows_obs) >= 3 and len(cols_obs) >= 3, "both axes need enough visible coordinates"

    row_step = _infer_step(rows_obs)
    col_step = _infer_step(cols_obs)
    assert row_step == rows_all[1] - rows_all[0], "row arithmetic step must be uniquely inferable"
    assert col_step == cols_all[1] - cols_all[0], "column arithmetic step must be uniquely inferable"

    full_rows = _extend_sequence(rows_obs, row_step, 0, inp.shape[0] - sprite_mask.shape[0])
    full_cols = _extend_sequence(cols_obs, col_step, 0, inp.shape[1] - sprite_mask.shape[1])
    assert full_rows == rows_all, "row extension must recover the intended hidden lattice"
    assert full_cols == cols_all, "column extension must recover the intended hidden lattice"

    assert rows_obs[0] != rows_all[0] and rows_obs[-1] != rows_all[-1], "row extremes must be hidden"
    assert cols_obs[0] != cols_all[0] and cols_obs[-1] != cols_all[-1], "column extremes must be hidden"
    if mode in ("row", "both"):
        assert any(r not in rows_obs for r in rows_all[1:-1]), "generator must hide an interior lattice row"
    if mode in ("col", "both"):
        assert any(c not in cols_obs for c in cols_all[1:-1]), "generator must hide an interior lattice column"

    expected = _render_output(inp.shape, sprite_mask, input_color, rows_all, cols_all, observed_positions)
    actual = main(inp)
    assert np.array_equal(actual, expected), "solver output must match the intended completed lattice"

    visible_only = _render_output(inp.shape, sprite_mask, input_color, rows_obs, cols_obs, observed_positions)
    assert not np.array_equal(visible_only, expected), "visible rows/cols alone must be insufficient"

    trimmed_rows = list(range(rows_obs[0], rows_obs[-1] + 1, row_step))
    trimmed_cols = list(range(cols_obs[0], cols_obs[-1] + 1, col_step))
    no_extension = _render_output(inp.shape, sprite_mask, input_color, trimmed_rows, trimmed_cols, observed_positions)
    assert not np.array_equal(no_extension, expected), "canvas-bound extension must affect the result"

    full_positions = {(r, c) for r in rows_all for c in cols_all}
    assert observed_positions < full_positions, "cartesian completion must add at least one sprite"
    added_count = len(full_positions - observed_positions)
    assert added_count >= 4, "added cyan sprites should be visually clear"


def _gen_attempt() -> np.ndarray | None:
    mode = random.choices(("row", "col", "both"), weights=(3, 3, 4), k=1)[0]
    sprite_mask = _random_sprite()
    sprite_h, sprite_w = sprite_mask.shape

    n_rows_total = 6 if mode in ("row", "both") else 5
    n_cols_total = 6 if mode in ("col", "both") else 5

    step_r = _choose_step(sprite_h, n_rows_total)
    step_c = _choose_step(sprite_w, n_cols_total)
    rows_all, grid_h = _build_full_line(sprite_h, n_rows_total, step_r)
    cols_all, grid_w = _build_full_line(sprite_w, n_cols_total, step_c)

    visible_row_ids = _choose_visible_indices(n_rows_total, mode in ("row", "both"))
    visible_col_ids = _choose_visible_indices(n_cols_total, mode in ("col", "both"))
    observed_cells = _choose_observed_cells(visible_row_ids, visible_col_ids)
    observed_positions = {(rows_all[rid], cols_all[cid]) for rid, cid in observed_cells}
    input_color = random.choice(INPUT_PALETTE)

    inp = _render_input((grid_h, grid_w), sprite_mask, input_color, observed_positions)
    try:
        _validate_instance(inp, sprite_mask, input_color, rows_all, cols_all, observed_positions, mode)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-lattice ARC input")
