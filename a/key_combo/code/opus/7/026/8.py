from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
TEAL = 8

CONTROL_COLORS = (BLUE, GREEN, YELLOW, ORANGE, TEAL)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bands_from_separators(start: int, end: int, separators: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    cur = start
    for sep in sorted(separators):
        if cur <= sep - 1:
            bands.append((cur, sep - 1))
        cur = sep + 1
    if cur <= end:
        bands.append((cur, end))
    return bands


def _blank_rows(grid: np.ndarray) -> list[int]:
    return [r for r in range(1, grid.shape[0]) if np.all(grid[r, :] == BLACK)]


def _blank_cols(grid: np.ndarray) -> list[int]:
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == BLACK)]


def _markers_by_band(grid: np.ndarray, col_bands: list[tuple[int, int]]) -> dict[int, int]:
    marker_by_band: dict[int, int] = {}
    for band_idx, (c0, c1) in enumerate(col_bands):
        hits = [(c, int(grid[0, c])) for c in range(c0, c1 + 1) if int(grid[0, c]) in CONTROL_COLORS]
        assert len(hits) == 1, "each column band must contain exactly one top-row control pixel"
        marker_by_band[band_idx] = hits[0][1]
    return marker_by_band


def _band_index(pos: int, bands: list[tuple[int, int]]) -> int:
    for idx, (lo, hi) in enumerate(bands):
        if lo <= pos <= hi:
            return idx
    raise AssertionError("position is not inside any band")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint full black separator rows/columns red, then recolor each grey partition fragment by that partition's top-row control color.

    Input structure:
    - The background is black, red is absent, and the top row contains distinct colored singleton control pixels.
    - Rows below the top row may include completely black separator rows, and the whole grid may include completely black separator columns.
    - Grey rectangular blocks occupy the non-separator area; some are local to one partition and some are pieces of a larger implied rectangle split by separator lines.

    Rule:
    - Detect every fully black row except row 0 and every fully black column, and paint those entire lines red.
    - Treat the red lines as partition boundaries over rows 1..H-1 and record each partition's row range and column range.
    - In each column partition range, find the unique control pixel on row 0 whose column lies inside that range.
    - Recolor every grey 4-connected fragment inside that partition to the control pixel's color, while preserving the top-row controls and the red separator lines.

    Key insight:
    - The separator lines create local control scopes: after partitioning, a grey fragment is colored by its partition's control column even when the fragment's own bbox does not contain that column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert RED not in set(map(int, np.unique(g))), "red is reserved for output separator lines"

    h, w = g.shape
    blank_rows = _blank_rows(g)
    blank_cols = _blank_cols(g)
    assert blank_rows, "expected at least one blank separator row"
    assert blank_cols, "expected at least one blank separator column"

    row_bands = _bands_from_separators(1, h - 1, blank_rows)
    col_bands = _bands_from_separators(0, w - 1, blank_cols)
    assert row_bands and col_bands

    marker_by_band = _markers_by_band(g, col_bands)

    non_top = g[1:, :]
    allowed = {BLACK, GREY}
    assert set(map(int, np.unique(non_top))).issubset(allowed), "rows below the top row may only contain black or grey"

    out = g.copy()
    out[blank_rows, :] = RED
    out[:, blank_cols] = RED

    for band_idx, (c0, c1) in enumerate(col_bands):
        color = marker_by_band[band_idx]
        for r0, r1 in row_bands:
            part = g[r0 : r1 + 1, c0 : c1 + 1]
            mask = part == GREY
            if np.any(mask):
                out[r0 : r1 + 1, c0 : c1 + 1][mask] = color

    return out


def _random_band_sizes(rng: np.random.Generator, count: int, low: int, high: int) -> list[int]:
    sizes = rng.integers(low, high + 1, size=count).astype(int).tolist()
    return sizes


def _build_ranges(start: int, sizes: list[int]) -> tuple[list[tuple[int, int]], list[int]]:
    bands: list[tuple[int, int]] = []
    blanks: list[int] = []
    cur = start
    for idx, size in enumerate(sizes):
        bands.append((cur, cur + size - 1))
        cur += size
        if idx != len(sizes) - 1:
            blanks.append(cur)
            cur += 1
    return bands, blanks


def _sample_interval_within_band(
    rng: np.random.Generator,
    band: tuple[int, int],
    *,
    min_len: int = 1,
    avoid: int | None = None,
) -> tuple[int, int] | None:
    lo, hi = band
    candidates: list[tuple[int, int]] = []
    for a in range(lo, hi + 1):
        for b in range(a + min_len - 1, hi + 1):
            if avoid is not None and a <= avoid <= b:
                continue
            candidates.append((a, b))
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _sample_spanning_interval(
    rng: np.random.Generator,
    first_band: tuple[int, int],
    last_band: tuple[int, int],
) -> tuple[int, int]:
    a = int(rng.integers(first_band[0], first_band[1] + 1))
    b = int(rng.integers(last_band[0], last_band[1] + 1))
    return a, b


def _rectangle_mask(
    shape: tuple[int, int],
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    blank_rows: set[int],
    blank_cols: set[int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r in range(r0, r1 + 1):
        if r in blank_rows:
            continue
        for c in range(c0, c1 + 1):
            if c in blank_cols:
                continue
            mask[r, c] = True
    return mask


def _mask_conflicts(mask: np.ndarray, occupied: np.ndarray) -> bool:
    h, w = mask.shape
    coords = np.argwhere(mask)
    for r, c in coords:
        r = int(r)
        c = int(c)
        if occupied[r, c]:
            return True
        for nr, nc in _neighbors4(r, c, h, w):
            if occupied[nr, nc]:
                return True
    return False


def _fragment_infos(
    mask: np.ndarray,
    col_bands: list[tuple[int, int]],
    marker_cols: dict[int, int],
) -> list[dict]:
    infos: list[dict] = []
    for comp in _components_mask(mask):
        r0, c0, r1, c1 = _bbox_coords(comp)
        band_idx = _band_index(c0, col_bands)
        marker_col = marker_cols[band_idx]
        infos.append(
            {
                "bbox": (r0, c0, r1, c1),
                "band_idx": band_idx,
                "contains_marker": c0 <= marker_col <= c1,
            }
        )
    return infos


def _wrong_bbox_scope(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    blank_rows = _blank_rows(g)
    blank_cols = _blank_cols(g)
    out[blank_rows, :] = RED
    out[:, blank_cols] = RED
    markers = {c: int(g[0, c]) for c in range(g.shape[1]) if int(g[0, c]) in CONTROL_COLORS}
    for comp in _components_mask(g == GREY):
        _r0, c0, _r1, c1 = _bbox_coords(comp)
        matches = [color for c, color in markers.items() if c0 <= c <= c1]
        if len(matches) == 1:
            out[comp[:, 0], comp[:, 1]] = matches[0]
    return out


def _expected_from_layout(
    grid: np.ndarray,
    blank_rows: list[int],
    blank_cols: list[int],
    col_bands: list[tuple[int, int]],
    marker_colors: dict[int, int],
) -> np.ndarray:
    out = grid.copy()
    out[blank_rows, :] = RED
    out[:, blank_cols] = RED
    for band_idx, (c0, c1) in enumerate(col_bands):
        color = marker_colors[band_idx]
        band_mask = grid[:, c0 : c1 + 1] == GREY
        out[:, c0 : c1 + 1][band_mask] = color
    return out


def _place_object(
    rng: np.random.Generator,
    occupied: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    blank_rows: set[int],
    blank_cols: set[int],
    marker_cols: dict[int, int],
    *,
    row_span: tuple[int, int],
    col_span: tuple[int, int],
    force_exclude_marker: bool = False,
) -> dict | None:
    h, w = occupied.shape
    rb0, rb1 = row_span
    cb0, cb1 = col_span
    for _ in range(120):
        if rb0 == rb1:
            r_interval = _sample_interval_within_band(rng, row_bands[rb0], min_len=2)
        else:
            r_interval = _sample_spanning_interval(rng, row_bands[rb0], row_bands[rb1])
        if r_interval is None:
            continue
        if cb0 == cb1:
            avoid = marker_cols[cb0] if force_exclude_marker else None
            c_interval = _sample_interval_within_band(
                rng,
                col_bands[cb0],
                min_len=1,
                avoid=avoid,
            )
        else:
            c_interval = _sample_spanning_interval(rng, col_bands[cb0], col_bands[cb1])
        if c_interval is None:
            continue

        r0, r1 = r_interval
        c0, c1 = c_interval
        mask = _rectangle_mask((h, w), r0, r1, c0, c1, blank_rows, blank_cols)
        if not np.any(mask):
            continue
        if _mask_conflicts(mask, occupied):
            continue

        fragments = _fragment_infos(mask, col_bands, marker_cols)
        if force_exclude_marker and not any(not info["contains_marker"] for info in fragments):
            continue
        if len(fragments) > 4:
            continue
        occupied[mask] = True
        return {
            "mask": mask,
            "row_span": row_span,
            "col_span": col_span,
            "fragments": fragments,
        }
    return None


def _generate_candidate() -> np.ndarray | None:
    rng = np.random.default_rng()

    marker_count = int(rng.integers(4, 6))
    row_band_count = int(rng.integers(2, 4))

    col_sizes = _random_band_sizes(rng, marker_count, 2, 4)
    if max(col_sizes) < 3:
        return None
    row_sizes = _random_band_sizes(rng, row_band_count, 2, 5)

    col_bands, blank_cols = _build_ranges(0, col_sizes)
    row_bands, blank_rows = _build_ranges(1, row_sizes)
    h = row_bands[-1][1] + 1
    w = col_bands[-1][1] + 1
    if h > 30 or w > 30:
        return None

    marker_palette = rng.choice(np.array(CONTROL_COLORS, dtype=int), size=marker_count, replace=False)
    marker_cols = {idx: int(rng.integers(c0, c1 + 1)) for idx, (c0, c1) in enumerate(col_bands)}
    marker_colors = {idx: int(marker_palette[idx]) for idx in range(marker_count)}

    grid = np.zeros((h, w), dtype=int)
    for idx, col in marker_cols.items():
        grid[0, col] = marker_colors[idx]

    occupied = np.zeros((h, w), dtype=bool)
    blank_row_set = set(blank_rows)
    blank_col_set = set(blank_cols)
    objects: list[dict] = []

    split_col_start = int(rng.integers(0, marker_count - 1))
    split_row_start = int(rng.integers(0, row_band_count - 1))
    split_rows_end = split_row_start + (1 if rng.random() < 0.65 else 0)
    split = _place_object(
        rng,
        occupied,
        row_bands,
        col_bands,
        blank_row_set,
        blank_col_set,
        marker_cols,
        row_span=(split_row_start, min(split_rows_end, row_band_count - 1)),
        col_span=(split_col_start, split_col_start + 1),
    )
    if split is None:
        return None
    objects.append(split)

    wide_bands = [idx for idx, (c0, c1) in enumerate(col_bands) if c1 - c0 + 1 >= 3]
    if not wide_bands:
        return None
    off_band = int(rng.choice(wide_bands))
    off = _place_object(
        rng,
        occupied,
        row_bands,
        col_bands,
        blank_row_set,
        blank_col_set,
        marker_cols,
        row_span=(int(rng.integers(0, row_band_count)),) * 2,
        col_span=(off_band, off_band),
        force_exclude_marker=True,
    )
    if off is None:
        return None
    objects.append(off)

    if row_band_count >= 2:
        row_object_col = int(rng.integers(0, marker_count))
        row_object = _place_object(
            rng,
            occupied,
            row_bands,
            col_bands,
            blank_row_set,
            blank_col_set,
            marker_cols,
            row_span=(split_row_start, min(split_row_start + 1, row_band_count - 1)),
            col_span=(row_object_col, row_object_col),
        )
        if row_object is None:
            return None
        objects.append(row_object)

    target_objects = int(rng.integers(4, 7))
    extra_attempts = 0
    while len(objects) < target_objects and extra_attempts < 200:
        extra_attempts += 1
        kind = str(rng.choice(np.array(["local", "split_col", "split_row", "split_both"], dtype=object)))
        if kind == "local":
            rb = int(rng.integers(0, row_band_count))
            cb = int(rng.integers(0, marker_count))
            obj = _place_object(
                rng,
                occupied,
                row_bands,
                col_bands,
                blank_row_set,
                blank_col_set,
                marker_cols,
                row_span=(rb, rb),
                col_span=(cb, cb),
            )
        elif kind == "split_col" and marker_count >= 2:
            cb0 = int(rng.integers(0, marker_count - 1))
            rb = int(rng.integers(0, row_band_count))
            obj = _place_object(
                rng,
                occupied,
                row_bands,
                col_bands,
                blank_row_set,
                blank_col_set,
                marker_cols,
                row_span=(rb, rb),
                col_span=(cb0, cb0 + 1),
            )
        elif kind == "split_row" and row_band_count >= 2:
            rb0 = int(rng.integers(0, row_band_count - 1))
            cb = int(rng.integers(0, marker_count))
            obj = _place_object(
                rng,
                occupied,
                row_bands,
                col_bands,
                blank_row_set,
                blank_col_set,
                marker_cols,
                row_span=(rb0, rb0 + 1),
                col_span=(cb, cb),
            )
        else:
            if row_band_count < 2 or marker_count < 2:
                continue
            rb0 = int(rng.integers(0, row_band_count - 1))
            cb0 = int(rng.integers(0, marker_count - 1))
            obj = _place_object(
                rng,
                occupied,
                row_bands,
                col_bands,
                blank_row_set,
                blank_col_set,
                marker_cols,
                row_span=(rb0, rb0 + 1),
                col_span=(cb0, cb0 + 1),
            )
        if obj is not None:
            objects.append(obj)
    if len(objects) < target_objects:
        return None

    grid[occupied] = GREY

    actual_blank_rows = _blank_rows(grid)
    actual_blank_cols = _blank_cols(grid)
    if actual_blank_rows != blank_rows or actual_blank_cols != blank_cols:
        return None

    expected = _expected_from_layout(grid, blank_rows, blank_cols, col_bands, marker_colors)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    grey_components = _components_mask(grid == GREY)
    if len(grey_components) < 4:
        return None
    if not any(obj["col_span"][1] > obj["col_span"][0] for obj in objects):
        return None
    if not any(obj["row_span"][1] > obj["row_span"][0] for obj in objects):
        return None

    fragment_off_marker = False
    for comp in grey_components:
        _r0, c0, _r1, c1 = _bbox_coords(comp)
        band_idx = _band_index(c0, col_bands)
        marker_col = marker_cols[band_idx]
        if not (c0 <= marker_col <= c1):
            fragment_off_marker = True
            break
    if not fragment_off_marker:
        return None

    if np.array_equal(_wrong_bbox_scope(grid), expected):
        return None

    used_fill_colors = set(int(v) for v in np.unique(expected)) - {BLACK, RED}
    if len(used_fill_colors) < 3:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid partitioned recoloring input")
