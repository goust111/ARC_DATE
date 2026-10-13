from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


MAX_GENERATION_ATTEMPTS = 300
BACKGROUND = 0
BASE_SHAPES_5 = (
    np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=bool,
    ),
)


def _mask_to_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    assert rows.size > 0, "sprite mask must be non-empty"
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _apply_transform(mask: np.ndarray, transform_id: int) -> np.ndarray:
    out = mask
    if transform_id >= 4:
        out = np.fliplr(out)
        transform_id -= 4
    return np.rot90(out, k=transform_id)


def _pad_to_tile(mask: np.ndarray, tile_size: int) -> np.ndarray:
    pad_total = tile_size - mask.shape[0]
    assert mask.shape[0] == mask.shape[1] == 5
    assert pad_total in (0, 2)
    pad = pad_total // 2
    return np.pad(mask, ((pad, pad), (pad, pad)), mode="constant")


def _detect_separator(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    h, w = grid.shape
    candidates = []
    for color in np.unique(grid):
        rows = np.where(np.all(grid == color, axis=1))[0]
        cols = np.where(np.all(grid == color, axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            continue
        row_spans = _spans(h, rows)
        col_spans = _spans(w, cols)
        if len(row_spans) != len(col_spans):
            continue
        if not row_spans or not col_spans:
            continue
        row_sizes = {r1 - r0 + 1 for r0, r1 in row_spans}
        col_sizes = {c1 - c0 + 1 for c0, c1 in col_spans}
        if len(row_sizes) == 1 and len(col_sizes) == 1 and row_sizes == col_sizes:
            candidates.append((int(color), rows, cols))
    assert len(candidates) == 1, "expected exactly one separator color"
    return candidates[0]


def _spans(length: int, separators: Iterable[int]) -> list[tuple[int, int]]:
    seps = sorted(int(x) for x in separators)
    if not seps:
        return [(0, length - 1)]
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in seps:
        assert start <= sep - 1, "separator lines cannot touch each other"
        spans.append((start, sep - 1))
        start = sep + 1
    assert start <= length - 1
    spans.append((start, length - 1))
    return spans


def _mode_color(values: np.ndarray) -> int:
    counts = Counter(int(v) for v in values.ravel())
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def _extract_features(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    separator, sep_rows, sep_cols = _detect_separator(g)
    row_spans = _spans(g.shape[0], sep_rows)
    col_spans = _spans(g.shape[1], sep_cols)
    n = len(row_spans)
    assert n >= 3 and n == len(col_spans)
    tile_h = row_spans[0][1] - row_spans[0][0] + 1
    tile_w = col_spans[0][1] - col_spans[0][0] + 1
    assert tile_h == tile_w

    bg = _mode_color(g[g != separator])
    tiles = []
    norm_shape_keys: list[tuple[tuple[int, ...], ...]] = []
    full_shape_masks: list[np.ndarray] = []
    shape_ids = np.empty((n, n), dtype=int)
    color_ids = np.empty((n, n), dtype=int)
    colors_seen: list[int] = []

    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            mask = tile != bg
            assert np.any(mask), "every tile must contain one sprite"
            sprite_colors = np.unique(tile[mask])
            assert len(sprite_colors) == 1, "each tile sprite must be monochrome"
            color = int(sprite_colors[0])
            assert color != separator
            norm_key = _mask_to_key(_crop_mask(mask))
            full_mask = mask.copy()

            if norm_key not in norm_shape_keys:
                norm_shape_keys.append(norm_key)
                full_shape_masks.append(full_mask)
            shape_id = norm_shape_keys.index(norm_key)

            if color not in colors_seen:
                colors_seen.append(color)
            color_id = colors_seen.index(color)

            shape_ids[r, c] = shape_id
            color_ids[r, c] = color_id
            tiles.append(
                {
                    "rc": (r, c),
                    "bbox": (r0, c0, r1, c1),
                    "tile": tile.copy(),
                    "mask": mask,
                    "full_mask": full_mask,
                    "shape_id": shape_id,
                    "color": color,
                    "color_id": color_id,
                }
            )

    assert len(norm_shape_keys) == n, "shape count must match tile-grid side"
    assert len(colors_seen) == n, "color count must match tile-grid side"
    return {
        "grid": g,
        "separator": separator,
        "bg": bg,
        "n": n,
        "tile_size": tile_h,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "tiles": tiles,
        "shape_ids": shape_ids,
        "color_ids": color_ids,
        "shape_masks": full_shape_masks,
        "colors": colors_seen,
    }


def _line_violations(table: np.ndarray) -> tuple[list[int], dict[int, int], list[int], dict[int, int]]:
    n = table.shape[0]
    target = set(range(n))
    bad_rows: list[int] = []
    row_missing: dict[int, int] = {}
    bad_cols: list[int] = []
    col_missing: dict[int, int] = {}

    for r in range(n):
        seen = set(int(v) for v in table[r, :])
        if seen != target:
            missing = list(target - seen)
            assert len(missing) == 1
            bad_rows.append(r)
            row_missing[r] = missing[0]
            assert any(np.count_nonzero(table[r, :] == k) == 2 for k in seen)

    for c in range(n):
        seen = set(int(v) for v in table[:, c])
        if seen != target:
            missing = list(target - seen)
            assert len(missing) == 1
            bad_cols.append(c)
            col_missing[c] = missing[0]
            assert any(np.count_nonzero(table[:, c] == k) == 2 for k in seen)

    return bad_rows, row_missing, bad_cols, col_missing


def _is_valid_solution(shape_ids: np.ndarray, color_ids: np.ndarray) -> bool:
    n = shape_ids.shape[0]
    target = set(range(n))
    for r in range(n):
        if set(int(v) for v in shape_ids[r, :]) != target:
            return False
        if set(int(v) for v in color_ids[r, :]) != target:
            return False
    for c in range(n):
        if set(int(v) for v in shape_ids[:, c]) != target:
            return False
        if set(int(v) for v in color_ids[:, c]) != target:
            return False
    combos = {(int(shape_ids[r, c]), int(color_ids[r, c])) for r in range(n) for c in range(n)}
    return len(combos) == n * n


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the only sprite tile whose shape and color simultaneously violate dual Latin-square constraints.

    Input structure:
    - Separator rows and columns partition the grid into an n x n array of equal sprite tiles.
    - Each tile contains one monochrome sprite on a shared background, so every tile has an independent shape attribute and color attribute.
    - Exactly one tile is wrong: in its tile-row and tile-column, one shape repeats and one color repeats, while every other tile is already consistent.

    Rule:
    - Audit tile rows and tile columns twice: once by sprite shape and once by sprite color.
    - The unique bad tile is where the shape-violation row/column and the color-violation row/column intersect.
    - Replace that tile by the shape missing from its row and column, drawn with the color missing from its row and column, using the same pose as the other copies of that shape.

    Key insight:
    - Shape and color form two independent Latin squares; only by repairing each attribute separately and then recombining them do you recover the unique missing sprite.
    """
    info = _extract_features(grid)
    shape_bad_rows, shape_row_missing, shape_bad_cols, shape_col_missing = _line_violations(info["shape_ids"])
    color_bad_rows, color_row_missing, color_bad_cols, color_col_missing = _line_violations(info["color_ids"])

    assert len(shape_bad_rows) == len(shape_bad_cols) == 1, "expected one bad shape row and one bad shape column"
    assert len(color_bad_rows) == len(color_bad_cols) == 1, "expected one bad color row and one bad color column"

    wrong_r, wrong_c = shape_bad_rows[0], shape_bad_cols[0]
    assert (wrong_r, wrong_c) == (color_bad_rows[0], color_bad_cols[0]), "shape and color audits must identify the same tile"

    fixed_shape = shape_row_missing[wrong_r]
    assert fixed_shape == shape_col_missing[wrong_c]
    fixed_color_id = color_row_missing[wrong_r]
    assert fixed_color_id == color_col_missing[wrong_c]
    fixed_color = info["colors"][fixed_color_id]

    out = info["grid"].copy()
    tile_bbox = (
        info["row_spans"][wrong_r][0],
        info["col_spans"][wrong_c][0],
        info["row_spans"][wrong_r][1],
        info["col_spans"][wrong_c][1],
    )
    r0, c0, r1, c1 = tile_bbox
    tile = np.full((r1 - r0 + 1, c1 - c0 + 1), info["bg"], dtype=int)
    mask = info["shape_masks"][fixed_shape]
    tile[mask] = fixed_color
    out[r0 : r1 + 1, c0 : c1 + 1] = tile

    repaired = _extract_features(out)
    assert _is_valid_solution(repaired["shape_ids"], repaired["color_ids"])
    return out


def _latin_pair(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    assert n in (3, 5)
    multipliers = [a for a in range(2, n) if np.gcd(a, n) == 1]
    a = int(rng.choice(multipliers))

    s = np.fromfunction(lambda r, c: (r + c) % n, (n, n), dtype=int)
    col = np.fromfunction(lambda r, c: (r + a * c) % n, (n, n), dtype=int)
    assert len({(int(s[r, c]), int(col[r, c])) for r in range(n) for c in range(n)}) == n * n

    row_perm = rng.permutation(n)
    col_perm = rng.permutation(n)
    s = s[row_perm][:, col_perm]
    col = col[row_perm][:, col_perm]
    s = rng.permutation(n)[s]
    col = rng.permutation(n)[col]
    return s.astype(int), col.astype(int)


def _build_shape_library(n: int, tile_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    idxs = rng.choice(len(BASE_SHAPES_5), size=n, replace=False)
    shapes = []
    seen = set()
    for idx in idxs:
        for _ in range(8):
            transformed = _apply_transform(BASE_SHAPES_5[int(idx)], int(rng.integers(0, 8)))
            key = _mask_to_key(transformed)
            if key not in seen:
                seen.add(key)
                shapes.append(_pad_to_tile(transformed, tile_size))
                break
        else:
            transformed = BASE_SHAPES_5[int(idx)]
            shapes.append(_pad_to_tile(transformed, tile_size))
    assert len(shapes) == n
    return shapes


def _render_board(shape_ids: np.ndarray, color_ids: np.ndarray, shape_masks: list[np.ndarray], colors: list[int], separator: int) -> np.ndarray:
    n = shape_ids.shape[0]
    tile_size = shape_masks[0].shape[0]
    size = n * tile_size + (n - 1)
    grid = np.full((size, size), BACKGROUND, dtype=int)

    for k in range(1, n):
        p = k * tile_size + (k - 1)
        grid[p, :] = separator
        grid[:, p] = separator

    for r in range(n):
        for c in range(n):
            r0 = r * (tile_size + 1)
            c0 = c * (tile_size + 1)
            tile = np.full((tile_size, tile_size), BACKGROUND, dtype=int)
            tile[shape_masks[int(shape_ids[r, c])]] = colors[int(color_ids[r, c])]
            grid[r0 : r0 + tile_size, c0 : c0 + tile_size] = tile
    return grid


def _enumerate_repairs(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    info = _extract_features(grid)
    candidates: list[tuple[int, int, int, int]] = []
    n = info["n"]
    base_shapes = info["shape_ids"].copy()
    base_colors = info["color_ids"].copy()
    for r in range(n):
        for c in range(n):
            cur_shape = int(base_shapes[r, c])
            cur_color = int(base_colors[r, c])
            for new_shape in range(n):
                for new_color in range(n):
                    if new_shape == cur_shape and new_color == cur_color:
                        continue
                    shape_trial = base_shapes.copy()
                    color_trial = base_colors.copy()
                    shape_trial[r, c] = new_shape
                    color_trial[r, c] = new_color
                    if _is_valid_solution(shape_trial, color_trial):
                        candidates.append((r, c, new_shape, new_color))
    return candidates


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.choice([3, 5]))
    tile_size = 5 if n == 5 else int(rng.choice([5, 7]))
    shape_masks = _build_shape_library(n, tile_size, rng)

    palette = [c for c in range(1, 10)]
    separator = int(rng.choice(palette))
    sprite_colors = list(rng.choice([c for c in palette if c != separator], size=n, replace=False))

    shape_ids, color_ids = _latin_pair(n, rng)
    solved = _render_board(shape_ids, color_ids, shape_masks, sprite_colors, separator)

    wrong_r = int(rng.integers(0, n))
    wrong_c = int(rng.integers(0, n))
    correct_shape = int(shape_ids[wrong_r, wrong_c])
    correct_color = int(color_ids[wrong_r, wrong_c])
    wrong_shape = int(rng.choice([k for k in range(n) if k != correct_shape]))
    wrong_color = int(rng.choice([k for k in range(n) if k != correct_color]))

    corrupt_shapes = shape_ids.copy()
    corrupt_colors = color_ids.copy()
    corrupt_shapes[wrong_r, wrong_c] = wrong_shape
    corrupt_colors[wrong_r, wrong_c] = wrong_color
    inp = _render_board(corrupt_shapes, corrupt_colors, shape_masks, sprite_colors, separator)

    repairs = _enumerate_repairs(inp)
    if repairs != [(wrong_r, wrong_c, correct_shape, correct_color)]:
        return None

    out = main(inp)
    if not np.array_equal(out, solved):
        return None

    parsed_in = _extract_features(inp)
    parsed_out = _extract_features(out)
    shape_bad_rows, _, shape_bad_cols, _ = _line_violations(parsed_in["shape_ids"])
    color_bad_rows, _, color_bad_cols, _ = _line_violations(parsed_in["color_ids"])
    if (shape_bad_rows, shape_bad_cols) != ([wrong_r], [wrong_c]):
        return None
    if (color_bad_rows, color_bad_cols) != ([wrong_r], [wrong_c]):
        return None
    if not _is_valid_solution(parsed_out["shape_ids"], parsed_out["color_ids"]):
        return None
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-Latin repair puzzle")
