from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

import numpy as np


BG = 0
CYAN = 8
SEPARATOR = 5
PALETTE = [1, 2, 3, 4, 5, 6, 7, 9]
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(counts.most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _solid_separator_row(grid: np.ndarray, bg: int) -> int:
    rows = [
        r
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != bg and np.all(grid[r] == int(grid[r, 0]))
    ]
    assert len(rows) == 1, "expected exactly one solid non-background separator row"
    return int(rows[0])


def _legend_mapping(grid: np.ndarray, sep: int, bg: int) -> tuple[dict[int, tuple[int, int]], int]:
    top = grid[:sep]
    comps = []
    for coords in _components(top != bg):
        vals = {int(top[r, c]) for r, c in coords}
        assert len(vals) == 1, "legend objects must be monochrome"
        color = next(iter(vals))
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        comps.append(
            {
                "coords": coords,
                "color": color,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "area": int(coords.shape[0]),
            }
        )

    sprites = []
    markers = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if comp["area"] == 1:
            markers.append(comp)
            continue
        assert h == w, "legend sprites must be square"
        assert comp["area"] == h * w, "legend sprites must be solid filled squares"
        sprites.append(comp)

    assert 3 <= len(sprites) <= 4, "legend must contain 3 or 4 sprites"
    assert len(sprites) == len(markers), "legend sprites and markers must pair one-to-one"

    sizes = {comp["bbox"][2] - comp["bbox"][0] + 1 for comp in sprites}
    assert len(sizes) == 1, "all legend sprites must share one side length"
    size = int(next(iter(sizes)))

    marker_colors = [comp["color"] for comp in markers]
    assert len(set(marker_colors)) == len(marker_colors), "legend marker colors must be unique"

    sprites.sort(key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
    markers.sort(key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))

    mapping: dict[int, tuple[int, int]] = {}
    for sprite, marker in zip(sprites, markers):
        r0, c0, r1, c1 = sprite["bbox"]
        mr, mc = map(int, marker["coords"][0])
        assert mr > r1, "legend markers must lie below their sprites"
        center_err = abs(2 * mc - (c0 + c1))
        assert center_err <= 1, "legend marker must align to the sprite center column"
        mapping[int(marker["color"])] = (int(sprite["color"]), size)

    return mapping, size


def _stamp_cells(
    shape: tuple[int, int],
    center_r: int,
    center_c: int,
    size: int,
    shift: int,
) -> list[tuple[int, int]]:
    top = int(center_r) - size // 2 + shift
    left = int(center_c) - size // 2 + shift
    cells = []
    for rr in range(top, top + size):
        for cc in range(left, left + size):
            if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                cells.append((rr, cc))
    return cells


def _render_from_lower(lower: np.ndarray, mapping: dict[int, tuple[int, int]]) -> np.ndarray:
    out = np.full(lower.shape, BG, dtype=int)
    active_cols = {int(c) for c in np.where(np.any(lower == CYAN, axis=0))[0]}
    markers = [
        (int(r), int(c), int(lower[r, c]))
        for r, c in np.argwhere(np.isin(lower, list(mapping)))
    ]
    markers.sort(key=lambda item: (item[1], item[0]))

    for r, c, color in markers:
        sprite_color, size = mapping[color]
        shift = 1 if c in active_cols else 0
        for rr, cc in _stamp_cells(lower.shape, r, c, size, shift):
            out[rr, cc] = int(sprite_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace lower markers by their legend squares, shifting only markers in cyan-activated columns.

    Input structure:
    - Black background surrounds a top legend, one solid separator row, and a lower marker field.
    - The legend contains 3 or 4 equal-size filled square sprites, each with a unique singleton marker color below it.
    - The lower field contains only background, cyan activator pixels, and singleton markers using the legend marker colors.

    Rule:
    - Pair each legend marker color with the filled square sprite above it by left-to-right order.
    - Discard the legend and separator; keep the grid width and only the rows below the separator.
    - In the lower field, every column containing at least one cyan pixel is active.
    - Replace each lower marker with its mapped filled square: inactive-column markers are centered on the marker, while active-column markers use the same centered placement shifted one cell down-right.
    - Clip any out-of-bounds part of a square and paint markers from left to right, then top to bottom, so later squares overwrite earlier ones.

    Key insight:
    - The legend decides which square color to stamp, while cyan pixels never appear in the output and only switch the stamping mode for their entire column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _get_bg(g)
    sep = _solid_separator_row(g, bg)
    mapping, _size = _legend_mapping(g, sep, bg)
    lower = g[sep + 1 :]

    valid = {bg, CYAN, *mapping.keys()}
    assert set(map(int, np.unique(lower))) <= valid, "lower field contains unknown colors"
    return _render_from_lower(lower, mapping)


def _plan_legend(size: int, n_entries: int, width: int) -> list[int]:
    min_gap = 1
    max_gap = 3
    total_span = n_entries * size
    free = width - total_span
    assert free >= (n_entries - 1) * min_gap
    max_left = free - (n_entries - 1) * min_gap
    left = random.randint(0, max_left)
    remaining = free - left
    gaps = []
    for idx in range(n_entries - 1):
        slots_left = n_entries - 2 - idx
        lo = min_gap
        hi = min(max_gap, remaining - slots_left * min_gap)
        gap = random.randint(lo, hi)
        gaps.append(gap)
        remaining -= gap
    starts = []
    col = left
    for idx in range(n_entries):
        starts.append(col)
        if idx < n_entries - 1:
            col += size + gaps[idx]
    return starts


def _lattice_positions(length: int, step: int) -> list[int]:
    vals = list(range(0, length, step))
    if len(vals) < 4:
        vals = list(range(0, length, max(1, step - 1)))
    return vals


def _sample_dimensions(size: int, n_entries: int) -> tuple[int, int, int]:
    top_total = size + 4
    min_width = n_entries * size + (n_entries - 1) + 2
    for _ in range(80):
        width = random.randint(max(min_width, 16), 24)
        lower_h = random.randint(max(10, 3 * size), min(16, 30 - top_total))
        rows = _lattice_positions(lower_h, size)
        cols = _lattice_positions(width, size)
        if len(rows) >= 4 and len(cols) >= max(4, n_entries):
            return width, lower_h, top_total
    raise RuntimeError("failed to sample usable dimensions")


def _sample_palettes(n_entries: int) -> tuple[list[int], list[int]]:
    picked = random.sample(PALETTE, 2 * n_entries)
    return picked[:n_entries], picked[n_entries:]


def _assign_marker_columns(
    colors: list[int],
    lattice_cols: list[int],
    lattice_rows: list[int],
) -> tuple[list[dict], set[int], set[int]] | None:
    branch_color = random.choice(colors)
    n_markers = random.randint(len(colors) + 1, len(colors) + 4)
    tags = [("active", branch_color), ("inactive", branch_color)]
    base_colors = colors[:]
    random.shuffle(base_colors)
    for color in base_colors:
        tags.append((random.choice(["active", "inactive"]), color))
    while len(tags) < n_markers:
        tags.append((random.choice(["active", "inactive"]), random.choice(colors)))

    active_cols: list[int] = []
    inactive_cols: list[int] = []
    used = set()
    counts = defaultdict(int)
    records = []
    shared_column_bias = random.random() < 0.5

    for idx, (tag, color) in enumerate(tags):
        pool = active_cols if tag == "active" else inactive_cols
        cap = len(lattice_rows) - (1 if tag == "active" else 0)
        candidates = [c for c in pool if counts[c] < cap]
        if shared_column_bias and candidates and random.random() < 0.45:
            col = random.choice(candidates)
        else:
            free = [
                c
                for c in lattice_cols
                if c not in used and counts[c] < cap
            ]
            if tag == "active" and idx == 0:
                boundary = [c for c in free if c == 0]
                if boundary:
                    free = boundary
            if not free and candidates:
                col = random.choice(candidates)
            elif free:
                col = random.choice(free)
                used.add(col)
                if tag == "active":
                    active_cols.append(col)
                else:
                    inactive_cols.append(col)
            else:
                return None
        counts[col] += 1
        records.append({"tag": tag, "color": int(color), "col": int(col)})

    if not active_cols or not inactive_cols:
        return None
    if len({rec["color"] for rec in records}) != len(colors):
        return None
    if not any(rec["color"] == branch_color and rec["tag"] == "active" for rec in records):
        return None
    if not any(rec["color"] == branch_color and rec["tag"] == "inactive" for rec in records):
        return None
    return records, set(active_cols), set(inactive_cols)


def _assign_rows(
    records: list[dict],
    active_cols: set[int],
    lattice_rows: list[int],
) -> tuple[list[dict], list[tuple[int, int]]] | None:
    by_col: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        by_col[int(record["col"])].append(record)

    cyans = []
    for col, items in by_col.items():
        rows = lattice_rows[:]
        random.shuffle(rows)
        need = len(items) + (1 if col in active_cols else 0)
        if need > len(rows):
            return None

        if col in active_cols and 0 in rows:
            rows.remove(0)
            rows.insert(0, 0)

        chosen = rows[:need]
        marker_rows = chosen[: len(items)]
        cyan_rows = chosen[len(items) :]
        random.shuffle(marker_rows)
        for record, row in zip(items, marker_rows):
            record["row"] = int(row)
        for row in cyan_rows:
            cyans.append((int(row), int(col)))

    return records, cyans


def _build_input_from_plan(
    width: int,
    lower_h: int,
    top_total: int,
    size: int,
    sprite_colors: list[int],
    marker_colors: list[int],
    records: list[dict],
    cyans: list[tuple[int, int]],
) -> np.ndarray:
    legend_h = size + 3
    sep = legend_h
    grid = np.full((top_total + lower_h, width), BG, dtype=int)
    starts = _plan_legend(size, len(marker_colors), width)
    for idx, c0 in enumerate(starts):
        grid[:size, c0 : c0 + size] = int(sprite_colors[idx])
        anchor_col = c0 + size // 2
        grid[size + 1, anchor_col] = int(marker_colors[idx])
    grid[sep, :] = SEPARATOR

    lower = grid[sep + 1 :]
    for record in records:
        lower[int(record["row"]), int(record["col"])] = int(record["color"])
    for r, c in cyans:
        assert int(lower[r, c]) == BG
        lower[int(r), int(c)] = CYAN
    return grid


def _mapping_from_palettes(sprite_colors: list[int], marker_colors: list[int], size: int) -> dict[int, tuple[int, int]]:
    return {
        int(marker): (int(sprite), int(size))
        for sprite, marker in zip(sprite_colors, marker_colors)
    }


def _lower_from_input(grid: np.ndarray) -> np.ndarray:
    sep = _solid_separator_row(grid, BG)
    return grid[sep + 1 :]


def _check_spacing(lower: np.ndarray, size: int, marker_colors: list[int]) -> bool:
    pts = [(int(r), int(c)) for r, c in np.argwhere(np.isin(lower, [CYAN, *marker_colors]))]
    for i, (r1, c1) in enumerate(pts):
        for r2, c2 in pts[i + 1 :]:
            if max(abs(r1 - r2), abs(c1 - c2)) < size:
                return False
    return True


def _semantic_checks(
    grid: np.ndarray,
    sprite_colors: list[int],
    marker_colors: list[int],
    size: int,
) -> bool:
    lower = _lower_from_input(grid)
    mapping = _mapping_from_palettes(sprite_colors, marker_colors, size)
    out = _render_from_lower(lower, mapping)
    if not np.array_equal(main(grid), out):
        return False
    if not _check_spacing(lower, size, marker_colors):
        return False

    active_cols = {int(c) for c in np.where(np.any(lower == CYAN, axis=0))[0]}
    marker_pts = [(int(r), int(c), int(lower[r, c])) for r, c in np.argwhere(np.isin(lower, marker_colors))]
    if not marker_pts:
        return False
    if not any(c in active_cols for _r, c, _k in marker_pts):
        return False
    if not any(c not in active_cols for _r, c, _k in marker_pts):
        return False
    if {k for _r, _c, k in marker_pts} != set(marker_colors):
        return False

    branch_seen = False
    by_color = defaultdict(list)
    for r, c, k in marker_pts:
        by_color[k].append((r, c))
        active_cells = set(_stamp_cells(lower.shape, r, c, size, 1))
        inactive_cells = set(_stamp_cells(lower.shape, r, c, size, 0))
        if c in active_cols and active_cells != inactive_cells:
            branch_seen = True
        if c not in active_cols and active_cells != inactive_cells:
            branch_seen = True
    if not branch_seen:
        return False
    if not any(
        any(c in active_cols for _r, c in pts) and any(c not in active_cols for _r, c in pts)
        for pts in by_color.values()
    ):
        return False

    centered = np.full(lower.shape, BG, dtype=int)
    for r, c, k in sorted(marker_pts, key=lambda item: (item[1], item[0])):
        color, _ = mapping[k]
        for rr, cc in _stamp_cells(lower.shape, r, c, size, 0):
            centered[rr, cc] = color
    if np.array_equal(centered, out):
        return False

    shifted = np.full(lower.shape, BG, dtype=int)
    for r, c, k in sorted(marker_pts, key=lambda item: (item[1], item[0])):
        color, _ = mapping[k]
        for rr, cc in _stamp_cells(lower.shape, r, c, size, 1):
            shifted[rr, cc] = color
    if np.array_equal(shifted, out):
        return False

    reversed_mapping = {
        int(marker_colors[idx]): (int(sprite_colors[::-1][idx]), size)
        for idx in range(len(marker_colors))
    }
    if len(marker_colors) > 1 and np.array_equal(_render_from_lower(lower, reversed_mapping), out):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    size = random.choice([3, 4])
    n_entries = random.choice([3, 4])
    width, lower_h, top_total = _sample_dimensions(size, n_entries)
    sprite_colors, marker_colors = _sample_palettes(n_entries)

    lattice_rows = _lattice_positions(lower_h, size)
    lattice_cols = _lattice_positions(width, size)
    assigned = _assign_marker_columns(marker_colors, lattice_cols, lattice_rows)
    if assigned is None:
        return None
    records, active_cols, _inactive_cols = assigned
    row_assignment = _assign_rows(records, active_cols, lattice_rows)
    if row_assignment is None:
        return None
    records, cyans = row_assignment

    grid = _build_input_from_plan(
        width,
        lower_h,
        top_total,
        size,
        sprite_colors,
        marker_colors,
        records,
        cyans,
    )
    if not _semantic_checks(grid, sprite_colors, marker_colors, size):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
