import numpy as np


_RNG = np.random.default_rng()
_ALL_COLORS = tuple(range(10))
_GEN_RETRY_LIMIT = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, exclude=()) -> list[int]:
    excluded = {int(x) for x in exclude}
    return sorted(int(x) for x in np.unique(np.asarray(grid, dtype=int)) if int(x) not in excluded)


def _band_info(full_flags: np.ndarray) -> tuple[list[int], list[int]]:
    starts: list[int] = []
    lengths: list[int] = []
    i = 0
    n = len(full_flags)
    while i < n:
        if bool(full_flags[i]):
            i += 1
            continue
        start = i
        while i < n and not bool(full_flags[i]):
            i += 1
        starts.append(int(start))
        lengths.append(int(i - start))
    return starts, lengths


def _parse_layout(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    candidates = []
    for color in _colors_present(g, exclude={bg}):
        full_rows = np.all(g == color, axis=1)
        full_cols = np.all(g == color, axis=0)
        if not np.any(full_rows) or not np.any(full_cols):
            continue

        row_starts, row_lengths = _band_info(full_rows)
        col_starts, col_lengths = _band_info(full_cols)
        if not row_starts or not col_starts:
            continue
        if len(set(row_lengths)) != 1 or len(set(col_lengths)) != 1:
            continue

        region_rows = len(row_starts)
        region_cols = len(col_starts)
        region_h = row_lengths[0]
        region_w = col_lengths[0]
        if region_h != region_rows or region_w != region_cols:
            continue

        expected_sep_rows = {row_starts[k] + region_h for k in range(region_rows - 1)}
        expected_sep_cols = {col_starts[k] + region_w for k in range(region_cols - 1)}
        if set(np.flatnonzero(full_rows)) != expected_sep_rows:
            continue
        if set(np.flatnonzero(full_cols)) != expected_sep_cols:
            continue
        if g.shape != (region_rows * region_h + region_rows - 1, region_cols * region_w + region_cols - 1):
            continue

        candidates.append(
            {
                "bg": int(bg),
                "line": int(color),
                "row_starts": row_starts,
                "col_starts": col_starts,
                "R": int(region_rows),
                "C": int(region_cols),
                "sep_rows": sorted(expected_sep_rows),
                "sep_cols": sorted(expected_sep_cols),
            }
        )

    assert len(candidates) == 1, "expected exactly one valid separator color"
    return candidates[0]


def _band_index_and_offset(pos: int, starts: list[int], size: int) -> tuple[int, int]:
    for idx, start in enumerate(starts):
        if start <= pos < start + size:
            return int(idx), int(pos - start)
    raise AssertionError("position not inside any region")


def _swap_address(r: int, c: int, layout: dict) -> tuple[int, int]:
    i, p = _band_index_and_offset(int(r), layout["row_starts"], layout["R"])
    j, q = _band_index_and_offset(int(c), layout["col_starts"], layout["C"])
    rr = layout["row_starts"][p] + i
    cc = layout["col_starts"][q] + j
    return int(rr), int(cc)


def _marker_coords(grid: np.ndarray, layout: dict) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    marker_mask = (g != layout["bg"]) & (g != layout["line"])
    return np.argwhere(marker_mask)


def _decode_marker(r: int, c: int, layout: dict) -> tuple[int, int, int, int]:
    i, p = _band_index_and_offset(int(r), layout["row_starts"], layout["R"])
    j, q = _band_index_and_offset(int(c), layout["col_starts"], layout["C"])
    return int(i), int(j), int(p), int(q)


def main(grid: np.ndarray) -> np.ndarray:
    """Swap each marker's region address with its in-region offset while preserving the separator system.

    Input structure:
    - One background color is cut by one line color into a regular grid of full
      horizontal and vertical separators.
    - The complete separators split the board into R x C equal regions whose
      sizes are exactly R rows by C columns.
    - Region interiors contain colored single-pixel markers, and the separator
      color also appears as shorter non-spanning decoy segments.

    Rule:
    - Only rows or columns fully covered by the line color count as separators.
    - For each marker, read its region index (i, j) and its local offset (p, q)
      inside that region; erase the marker and redraw it at region (p, q) with
      local offset (i, j), keeping its color.
    - Preserve every full separator and every short decoy segment unchanged.

    Key insight:
    - The board is a two-level coordinate system: the outer region address and
      the inner offset have the same shape, so the move is a direct swap of
      those two addresses rather than an ordinary translation or mirror.
    """
    g = np.asarray(grid, dtype=int)
    layout = _parse_layout(g)
    out = g.copy()

    markers = _marker_coords(g, layout)
    if len(markers) == 0:
        return out

    out[markers[:, 0], markers[:, 1]] = layout["bg"]
    destinations = set()
    for r, c in markers:
        rr, cc = _swap_address(int(r), int(c), layout)
        assert g[rr, cc] != layout["line"], "marker destination overlaps separator color"
        destinations.add((int(rr), int(cc)))
        out[rr, cc] = int(g[int(r), int(c)])

    assert len(destinations) == len(markers)
    return out


def _row_starts(R: int) -> list[int]:
    return [k * (R + 1) for k in range(R)]


def _col_starts(C: int) -> list[int]:
    return [k * (C + 1) for k in range(C)]


def _source_cell(address: tuple[int, int, int, int], row_starts: list[int], col_starts: list[int]) -> tuple[int, int]:
    i, j, p, q = address
    return int(row_starts[i] + p), int(col_starts[j] + q)


def _dest_cell(address: tuple[int, int, int, int], row_starts: list[int], col_starts: list[int]) -> tuple[int, int]:
    i, j, p, q = address
    return int(row_starts[p] + i), int(col_starts[q] + j)


def _nonseparator_full(mask: np.ndarray, sep_rows: list[int], sep_cols: list[int]) -> bool:
    for r in range(mask.shape[0]):
        if r not in sep_rows and bool(np.all(mask[r])):
            return True
    for c in range(mask.shape[1]):
        if c not in sep_cols and bool(np.all(mask[:, c])):
            return True
    return False


def _place_partial_segments(
    grid: np.ndarray,
    line_mask: np.ndarray,
    R: int,
    C: int,
    row_starts: list[int],
    col_starts: list[int],
    sep_rows: list[int],
    sep_cols: list[int],
    line_color: int,
) -> int:
    region_cells = R * C * R * C
    target = int(_RNG.integers(1, min(5, max(2, region_cells // 6)) + 1))
    placed = 0
    attempts = 0

    while placed < target and attempts < 200:
        attempts += 1
        orient = "h" if _RNG.random() < 0.5 else "v"
        band_i = int(_RNG.integers(R))
        band_j = int(_RNG.integers(C))
        rs = row_starts[band_i]
        cs = col_starts[band_j]

        if orient == "h":
            length = int(_RNG.integers(2, C + 1))
            local_r = int(_RNG.integers(R))
            start_c = int(_RNG.integers(0, C - length + 1))
            coords = [(rs + local_r, cs + start_c + dc) for dc in range(length)]
        else:
            length = int(_RNG.integers(2, R + 1))
            local_c = int(_RNG.integers(C))
            start_r = int(_RNG.integers(0, R - length + 1))
            coords = [(rs + start_r + dr, cs + local_c) for dr in range(length)]

        if any(bool(line_mask[r, c]) for r, c in coords):
            continue

        probe = line_mask.copy()
        for r, c in coords:
            probe[r, c] = True
        if _nonseparator_full(probe, sep_rows, sep_cols):
            continue

        for r, c in coords:
            line_mask[r, c] = True
            grid[r, c] = line_color
        placed += 1

    return placed


def _candidate_addresses(free_mask: np.ndarray, row_starts: list[int], col_starts: list[int], R: int, C: int) -> list[tuple[int, int, int, int]]:
    candidates = []
    for i in range(R):
        for j in range(C):
            for p in range(R):
                for q in range(C):
                    src = _source_cell((i, j, p, q), row_starts, col_starts)
                    dst = _dest_cell((i, j, p, q), row_starts, col_starts)
                    if bool(free_mask[src]) and bool(free_mask[dst]):
                        candidates.append((i, j, p, q))
    return candidates


def _sample_markers(candidates: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]] | None:
    both = [a for a in candidates if a[0] != a[2] and a[1] != a[3]]
    one_axis = [a for a in candidates if (a[0] != a[2]) ^ (a[1] != a[3])]
    fixed = [a for a in candidates if a[0] == a[2] and a[1] == a[3]]
    if not both or not one_axis:
        return None

    marker_count = int(_RNG.integers(3, min(8, len(candidates)) + 1))
    chosen = [_RNG.choice(both), _RNG.choice(one_axis)]
    chosen = [tuple(map(int, a)) for a in chosen]

    if fixed and _RNG.random() < 0.45 and len(chosen) < marker_count:
        picked = tuple(map(int, _RNG.choice(fixed)))
        if picked not in chosen:
            chosen.append(picked)

    remaining = [a for a in candidates if a not in chosen]
    _RNG.shuffle(remaining)
    for address in remaining:
        if len(chosen) >= marker_count:
            break
        chosen.append(tuple(map(int, address)))

    if len(chosen) < 3:
        return None
    if not any(a[0] != a[2] for a in chosen):
        return None
    if not any(a[1] != a[3] for a in chosen):
        return None
    if not any(a[0] != a[2] and a[1] != a[3] for a in chosen):
        return None
    if not any((a[0] != a[2]) ^ (a[1] != a[3]) for a in chosen):
        return None
    return chosen


def _assign_marker_colors(count: int, bg: int, line: int) -> list[int]:
    palette = [c for c in _ALL_COLORS if c not in (bg, line)]
    _RNG.shuffle(palette)
    color_count = int(_RNG.integers(2, min(5, count, len(palette)) + 1))
    chosen = palette[:color_count]
    colors = []
    for idx in range(count):
        if idx < color_count:
            colors.append(int(chosen[idx]))
        else:
            colors.append(int(chosen[int(_RNG.integers(color_count))]))
    _RNG.shuffle(colors)
    return colors


def _semantic_check(inp: np.ndarray) -> bool:
    try:
        layout = _parse_layout(inp)
        out = main(inp)
        round_trip = main(out)
    except Exception:
        return False

    if np.array_equal(inp, out):
        return False
    if not np.array_equal(round_trip, inp):
        return False
    if _get_bg(inp) != layout["bg"]:
        return False

    g = np.asarray(inp, dtype=int)
    marker_mask = (g != layout["bg"]) & (g != layout["line"])
    if np.count_nonzero(marker_mask) < 3:
        return False

    line_in = g == layout["line"]
    line_out = out == layout["line"]
    if not np.array_equal(line_in, line_out):
        return False
    if _nonseparator_full(line_in, layout["sep_rows"], layout["sep_cols"]):
        return False

    moved_row = False
    moved_col = False
    both_axes = False
    one_axis = False
    for r, c in np.argwhere(marker_mask):
        rr, cc = _swap_address(int(r), int(c), layout)
        if int(out[rr, cc]) != int(g[r, c]):
            return False
        i, j, p, q = _decode_marker(int(r), int(c), layout)
        moved_row |= i != p
        moved_col |= j != q
        both_axes |= i != p and j != q
        one_axis |= (i != p) ^ (j != q)

    if not (moved_row and moved_col and both_axes and one_axis):
        return False

    out_marker_colors = sorted(int(x) for x in out[(out != layout["bg"]) & (out != layout["line"])])
    in_marker_colors = sorted(int(x) for x in g[marker_mask])
    return out_marker_colors == in_marker_colors


def _gen_attempt() -> np.ndarray | None:
    R = int(_RNG.choice(np.array([2, 3, 4, 5]), p=np.array([0.15, 0.35, 0.30, 0.20])))
    C = int(_RNG.choice(np.array([2, 3, 4, 5]), p=np.array([0.15, 0.35, 0.30, 0.20])))

    row_starts = _row_starts(R)
    col_starts = _col_starts(C)
    sep_rows = [s + R for s in row_starts[:-1]]
    sep_cols = [s + C for s in col_starts[:-1]]
    H = R * R + (R - 1)
    W = C * C + (C - 1)

    bg, line = _RNG.choice(np.array(_ALL_COLORS), size=2, replace=False)
    bg = int(bg)
    line = int(line)
    grid = np.full((H, W), bg, dtype=int)
    line_mask = np.zeros((H, W), dtype=bool)

    for r in sep_rows:
        grid[r, :] = line
        line_mask[r, :] = True
    for c in sep_cols:
        grid[:, c] = line
        line_mask[:, c] = True

    if _place_partial_segments(grid, line_mask, R, C, row_starts, col_starts, sep_rows, sep_cols, line) == 0:
        return None

    free_mask = ~line_mask
    candidates = _candidate_addresses(free_mask, row_starts, col_starts, R, C)
    addresses = _sample_markers(candidates)
    if addresses is None:
        return None

    colors = _assign_marker_colors(len(addresses), bg, line)
    for address, color in zip(addresses, colors):
        r, c = _source_cell(address, row_starts, col_starts)
        grid[r, c] = int(color)

    if not _semantic_check(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 门牌互换 input")
