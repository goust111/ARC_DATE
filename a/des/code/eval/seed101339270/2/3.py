from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
LEGEND_SIDES = ("top", "bottom", "left", "right")
REGION_LAYOUTS = ((1, 2), (2, 1), (1, 3), (3, 1), (2, 2))
GEN_RETRIES = 400
TILE_RETRIES = 200


def _full_line_indices(grid: np.ndarray, color: int) -> tuple[list[int], list[int]]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == color)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]
    return rows, cols


def _find_separator_color(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    best: tuple[int, int, list[int], list[int]] | None = None
    for raw_color in np.unique(grid):
        color = int(raw_color)
        rows, cols = _full_line_indices(grid, color)
        score = len(rows) + len(cols)
        if score < 2:
            continue
        if best is None or score > best[0]:
            best = (score, color, rows, cols)
    assert best is not None, "expected a separator color with multiple full lines"
    return best[1], best[2], best[3]


def _legend_pairs_from_columns(strip: np.ndarray, sep: int) -> dict[int, int] | None:
    assert strip.shape[0] == 2
    mapping: dict[int, int] = {}
    active = 0
    for c in range(strip.shape[1]):
        a = int(strip[0, c])
        b = int(strip[1, c])
        if a == sep or b == sep:
            assert a == sep and b == sep
            continue
        if a == BACKGROUND and b == BACKGROUND:
            continue
        if a in (BACKGROUND, sep) or b in (BACKGROUND, sep) or a == b:
            return None
        mapping[a] = b
        active += 1
    if active < 2 or len(mapping) != active:
        return None
    return mapping


def _legend_pairs_from_rows(strip: np.ndarray, sep: int) -> dict[int, int] | None:
    assert strip.shape[1] == 2
    mapping: dict[int, int] = {}
    active = 0
    for r in range(strip.shape[0]):
        a = int(strip[r, 0])
        b = int(strip[r, 1])
        if a == sep or b == sep:
            assert a == sep and b == sep
            continue
        if a == BACKGROUND and b == BACKGROUND:
            continue
        if a in (BACKGROUND, sep) or b in (BACKGROUND, sep) or a == b:
            return None
        mapping[a] = b
        active += 1
    if active < 2 or len(mapping) != active:
        return None
    return mapping


def _parse_legend(
    grid: np.ndarray, sep: int, sep_rows: list[int], sep_cols: list[int]
) -> tuple[dict[int, int], tuple[int, int, int, int], str]:
    h, w = grid.shape
    candidates: list[tuple[str, dict[int, int], tuple[int, int, int, int]]] = []

    if 2 in sep_rows:
        mapping = _legend_pairs_from_columns(grid[0:2, :], sep)
        if mapping is not None:
            candidates.append(("top", mapping, (3, h, 0, w)))
    if h - 3 in sep_rows:
        mapping = _legend_pairs_from_columns(grid[h - 2 : h, :], sep)
        if mapping is not None:
            candidates.append(("bottom", mapping, (0, h - 3, 0, w)))
    if 2 in sep_cols:
        mapping = _legend_pairs_from_rows(grid[:, 0:2], sep)
        if mapping is not None:
            candidates.append(("left", mapping, (0, h, 3, w)))
    if w - 3 in sep_cols:
        mapping = _legend_pairs_from_rows(grid[:, w - 2 : w], sep)
        if mapping is not None:
            candidates.append(("right", mapping, (0, h, 0, w - 3)))

    assert len(candidates) == 1, "legend side must be uniquely identifiable"
    side, mapping, payload = candidates[0]
    return mapping, payload, side


def _spans_between(start: int, end: int, separators: Iterable[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cur = start
    for s in sorted(int(x) for x in separators if start <= int(x) < end):
        if cur < s:
            spans.append((cur, s))
        cur = s + 1
    if cur < end:
        spans.append((cur, end))
    return spans


def _sample_info(region: np.ndarray, color: int) -> tuple[int, int, np.ndarray] | None:
    coords = np.argwhere(region == color)
    if coords.size == 0:
        return None

    h, w = region.shape
    rmin, cmin = coords.min(axis=0)
    rmax, cmax = coords.max(axis=0)
    candidates: list[tuple[int, int, np.ndarray]] = []

    def add_candidate(anchor_r: int, anchor_c: int, tile_h: int, tile_w: int) -> None:
        if tile_h >= h or tile_w >= w:
            return
        mask = region[anchor_r : anchor_r + tile_h, anchor_c : anchor_c + tile_w] == color
        if not mask.any():
            return
        if not np.all(mask.any(axis=1)) or not np.all(mask.any(axis=0)):
            return
        candidates.append((anchor_r, anchor_c, mask))

    if rmin == 0 and cmin == 0:
        add_candidate(0, 0, int(rmax) + 1, int(cmax) + 1)
    if rmin == 0 and cmax == w - 1:
        add_candidate(0, int(cmin), int(rmax) + 1, w - int(cmin))
    if rmax == h - 1 and cmin == 0:
        add_candidate(int(rmin), 0, h - int(rmin), int(cmax) + 1)
    if rmax == h - 1 and cmax == w - 1:
        add_candidate(int(rmin), int(cmin), h - int(rmin), w - int(cmin))

    if len(candidates) != 1:
        return None
    return candidates[0]


def _repair_region(region: np.ndarray, mapping: dict[int, int], sep: int) -> np.ndarray:
    present = {int(v) for v in np.unique(region) if int(v) not in {BACKGROUND, sep}}
    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for color in present:
        if color not in mapping:
            continue
        info = _sample_info(region, color)
        if info is not None:
            anchor_r, anchor_c, mask = info
            candidates.append((color, anchor_r, anchor_c, mask))

    assert len(candidates) == 1, "each region must contain exactly one corner sample tile"
    sample_color, anchor_r, anchor_c, tile_mask = candidates[0]
    pattern_color = int(mapping[sample_color])

    h, w = region.shape
    tile_h, tile_w = tile_mask.shape
    rr = np.arange(h)[:, None]
    cc = np.arange(w)[None, :]
    repeated = tile_mask[(rr - anchor_r) % tile_h, (cc - anchor_c) % tile_w]

    sample_box = np.zeros((h, w), dtype=bool)
    sample_box[anchor_r : anchor_r + tile_h, anchor_c : anchor_c + tile_w] = True
    visible = region == pattern_color
    assert np.all(repeated[visible]), "visible pattern pixels must agree with the sample tile"

    out = region.copy()
    out[region == sample_color] = BACKGROUND
    repair = (region == BACKGROUND) & ~sample_box & repeated
    out[repair] = pattern_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair repeated monochrome patterns inside separator-defined regions by using
    a corner sample tile and an edge legend that maps sample colors to pattern colors.
    """
    g = np.asarray(grid, dtype=int)
    sep, sep_rows, sep_cols = _find_separator_color(g)
    mapping, (r0, r1, c0, c1), _side = _parse_legend(g, sep, sep_rows, sep_cols)

    row_spans = _spans_between(r0, r1, sep_rows)
    col_spans = _spans_between(c0, c1, sep_cols)
    assert len(row_spans) * len(col_spans) >= 2, "expected multiple payload regions"

    out = g.copy()
    for rs, re in row_spans:
        for cs, ce in col_spans:
            out[rs:re, cs:ce] = _repair_region(g[rs:re, cs:ce], mapping, sep)
    return out


def _has_smaller_period(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for ph in range(1, h + 1):
        if h % ph != 0:
            continue
        for pw in range(1, w + 1):
            if w % pw != 0 or (ph == h and pw == w):
                continue
            base = mask[:ph, :pw]
            tiled = np.tile(base, (h // ph, w // pw))
            if np.array_equal(mask, tiled):
                return True
    return False


def _make_tile(tile_h: int, tile_w: int) -> np.ndarray | None:
    area = tile_h * tile_w
    for _ in range(TILE_RETRIES):
        p = random.uniform(0.35, 0.7)
        mask = np.random.random((tile_h, tile_w)) < p
        if mask.sum() < 2 or mask.sum() >= area:
            continue
        if not np.all(mask.any(axis=1)) or not np.all(mask.any(axis=0)):
            continue
        if _has_smaller_period(mask):
            continue
        return mask
    return None


def _region_pattern_mask(
    region_h: int,
    region_w: int,
    tile_mask: np.ndarray,
    anchor_r: int,
    anchor_c: int,
) -> np.ndarray:
    tile_h, tile_w = tile_mask.shape
    rr = np.arange(region_h)[:, None]
    cc = np.arange(region_w)[None, :]
    return tile_mask[(rr - anchor_r) % tile_h, (cc - anchor_c) % tile_w]


def _corner_anchor(
    corner: str, region_h: int, region_w: int, tile_h: int, tile_w: int
) -> tuple[int, int]:
    if corner == "top_left":
        return 0, 0
    if corner == "top_right":
        return 0, region_w - tile_w
    if corner == "bottom_left":
        return region_h - tile_h, 0
    assert corner == "bottom_right"
    return region_h - tile_h, region_w - tile_w


def _separator_layout(
    counts: tuple[int, int], sizes: list[int], offset: int
) -> tuple[list[tuple[int, int]], list[int]]:
    spans: list[tuple[int, int]] = []
    lines: list[int] = []
    cur = offset
    for idx, size in enumerate(sizes):
        spans.append((cur, cur + size))
        cur += size
        if idx != len(sizes) - 1:
            lines.append(cur)
            cur += 1
    assert len(spans) == counts[0] or len(spans) == counts[1]
    return spans, lines


def _choose_legend_slots(length: int, blocked: set[int], count: int) -> list[int] | None:
    slots = [idx for idx in range(length) if idx not in blocked]
    if len(slots) < count:
        return None
    return random.sample(slots, count)


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    region_rows, region_cols = random.choice(REGION_LAYOUTS)
    region_count = region_rows * region_cols
    side = random.choice(LEGEND_SIDES)

    colors = list(COLOR_POOL)
    random.shuffle(colors)
    sample_colors = colors[:region_count]
    pattern_colors = colors[region_count : 2 * region_count]
    mapping = dict(zip(sample_colors, pattern_colors))

    row_sizes = [random.randint(6, 9) for _ in range(region_rows)]
    col_sizes = [random.randint(6, 9) for _ in range(region_cols)]
    payload_h = sum(row_sizes) + (region_rows - 1)
    payload_w = sum(col_sizes) + (region_cols - 1)

    if side in ("top", "bottom"):
        grid_h = payload_h + 3
        grid_w = payload_w
        payload_r0 = 3 if side == "top" else 0
        payload_c0 = 0
        legend_sep_row = 2 if side == "top" else grid_h - 3
        legend_rows = (0, 1) if side == "top" else (grid_h - 2, grid_h - 1)
    else:
        grid_h = payload_h
        grid_w = payload_w + 3
        payload_r0 = 0
        payload_c0 = 3 if side == "left" else 0
        legend_sep_col = 2 if side == "left" else grid_w - 3
        legend_cols = (0, 1) if side == "left" else (grid_w - 2, grid_w - 1)

    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)

    row_spans, row_lines = _separator_layout((region_rows, region_cols), row_sizes, payload_r0)
    col_spans, col_lines = _separator_layout((region_rows, region_cols), col_sizes, payload_c0)

    if side in ("top", "bottom"):
        row_lines = [legend_sep_row] + row_lines
    else:
        col_lines = [legend_sep_col] + col_lines

    for r in row_lines:
        grid[r, :] = SEPARATOR
    for c in col_lines:
        grid[:, c] = SEPARATOR

    if side in ("top", "bottom"):
        legend_cols_used = _choose_legend_slots(grid_w, set(col_lines), region_count)
        if legend_cols_used is None:
            raise RuntimeError("not enough legend columns")
        for c, sample_color in zip(legend_cols_used, sample_colors):
            pattern_color = mapping[sample_color]
            grid[legend_rows[0], c] = sample_color
            grid[legend_rows[1], c] = pattern_color
    else:
        legend_rows_used = _choose_legend_slots(grid_h, set(row_lines), region_count)
        if legend_rows_used is None:
            raise RuntimeError("not enough legend rows")
        for r, sample_color in zip(legend_rows_used, sample_colors):
            pattern_color = mapping[sample_color]
            grid[r, legend_cols[0]] = sample_color
            grid[r, legend_cols[1]] = pattern_color

    expected = grid.copy()
    corners = ("top_left", "top_right", "bottom_left", "bottom_right")
    tile_kinds: list[tuple[int, int, np.ndarray, str]] = []

    for idx, ((rs, re), (cs, ce)) in enumerate(
        [(r_span, c_span) for r_span in row_spans for c_span in col_spans]
    ):
        region_h = re - rs
        region_w = ce - cs
        sample_color = sample_colors[idx]
        pattern_color = pattern_colors[idx]

        tile_h = random.randint(2, min(4, region_h - 2))
        tile_w = random.randint(2, min(4, region_w - 2))
        tile_mask = _make_tile(tile_h, tile_w)
        if tile_mask is None:
            raise RuntimeError("failed to build tile mask")

        corner = random.choice(corners)
        anchor_r, anchor_c = _corner_anchor(corner, region_h, region_w, tile_h, tile_w)
        repeated = _region_pattern_mask(region_h, region_w, tile_mask, anchor_r, anchor_c)

        sample_box = np.zeros((region_h, region_w), dtype=bool)
        sample_box[anchor_r : anchor_r + tile_h, anchor_c : anchor_c + tile_w] = True
        pattern_cells = repeated & ~sample_box
        visible_cells = np.argwhere(pattern_cells)
        if len(visible_cells) < 4:
            raise RuntimeError("region leaves too little visible pattern")

        region_in = np.full((region_h, region_w), BACKGROUND, dtype=int)
        region_out = np.full((region_h, region_w), BACKGROUND, dtype=int)
        region_in[pattern_cells] = pattern_color
        region_out[repeated & ~sample_box] = pattern_color
        region_in[
            anchor_r : anchor_r + tile_h, anchor_c : anchor_c + tile_w
        ][tile_mask] = sample_color

        max_holes = max(1, min(6, int(pattern_cells.sum()) // 4))
        hole_count = random.randint(1, max_holes)
        hole_choices = random.sample(list(map(tuple, visible_cells)), hole_count)
        for rr, cc in hole_choices:
            region_in[int(rr), int(cc)] = BACKGROUND

        assert np.any(region_in == pattern_color), "pattern color must remain visible"
        assert np.any(region_in == sample_color), "sample tile must be visible"
        grid[rs:re, cs:ce] = region_in
        expected[rs:re, cs:ce] = region_out
        tile_kinds.append((tile_h, tile_w, tile_mask, corner))

    produced = main(grid)
    assert np.array_equal(produced, expected), "generator produced a case inconsistent with main()"
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid, _expected = _generate_case()
        except (AssertionError, RuntimeError):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
