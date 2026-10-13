from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
GEN_RETRIES = 400
PATCH_RETRIES = 120
SOURCE_SET_RETRIES = 80
PALETTE = tuple(range(1, 10))
SIDE_ORDER = ("top", "bottom", "left", "right")


def _dominant_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _detect_dividers(
    grid: np.ndarray,
    bg: int,
) -> tuple[int, list[int], list[int], dict[int, int], dict[int, int]]:
    h, w = grid.shape
    best = None

    for color in map(int, np.unique(grid)):
        if color == bg:
            continue

        row_gaps: dict[int, int] = {}
        for r in range(h):
            row = grid[r]
            if np.all((row == color) | (row == bg)) and np.count_nonzero(row == bg) == 1:
                row_gaps[int(r)] = int(np.flatnonzero(row == bg)[0])

        col_gaps: dict[int, int] = {}
        for c in range(w):
            col = grid[:, c]
            if np.all((col == color) | (col == bg)) and np.count_nonzero(col == bg) == 1:
                col_gaps[int(c)] = int(np.flatnonzero(col == bg)[0])

        if not row_gaps or not col_gaps:
            continue

        rows = sorted(row_gaps)
        cols = sorted(col_gaps)
        if any(b - a < 2 for a, b in zip(rows, rows[1:])):
            continue
        if any(b - a < 2 for a, b in zip(cols, cols[1:])):
            continue

        intersections = sum(int(grid[r, c] == color) for r in rows for c in cols)
        score = (len(rows) + len(cols), intersections)
        if best is None or score > best[0]:
            best = (score, color, rows, cols, row_gaps, col_gaps)

    assert best is not None, "failed to detect divider color and separator lines"
    return best[1], best[2], best[3], best[4], best[5]


def _bands_from_separators(separators: list[int], size: int) -> list[tuple[int, int]]:
    bands = []
    start = 0
    for sep in separators:
        end = int(sep) - 1
        assert start <= end, "separator lines must leave non-empty regions"
        bands.append((start, end))
        start = int(sep) + 1
    assert start <= size - 1, "final region band must be non-empty"
    bands.append((start, size - 1))
    return bands


def _extract_patch(region: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(region != bg)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return region[r0 : r1 + 1, c0 : c1 + 1].copy()


def _center_range(lo: int, hi: int, size: int) -> tuple[int, int] | None:
    left = int(lo) + size // 2
    right = int(hi) - (size - 1 - size // 2)
    if left > right:
        return None
    return left, right


def _stamp_patch(
    out: np.ndarray,
    patch: np.ndarray,
    target_box: tuple[int, int, int, int],
    side: str,
    gap_pos: int,
    bg: int,
) -> None:
    if side in {"left", "right"}:
        patch = np.fliplr(patch)
    else:
        patch = np.flipud(patch)

    ph, pw = patch.shape
    tr0, tr1, tc0, tc1 = target_box
    if side == "top":
        dr0 = tr1 - ph + 1
        dc0 = int(gap_pos) - pw // 2
    elif side == "bottom":
        dr0 = tr0
        dc0 = int(gap_pos) - pw // 2
    elif side == "left":
        dr0 = int(gap_pos) - ph // 2
        dc0 = tc1 - pw + 1
    else:
        assert side == "right"
        dr0 = int(gap_pos) - ph // 2
        dc0 = tc0

    assert tr0 <= dr0 and dr0 + ph - 1 <= tr1, "stamped patch must fit vertically"
    assert tc0 <= dc0 and dc0 + pw - 1 <= tc1, "stamped patch must fit horizontally"

    for rr in range(ph):
        for cc in range(pw):
            val = int(patch[rr, cc])
            if val == bg:
                continue
            r = dr0 + rr
            c = dc0 + cc
            prev = int(out[r, c])
            assert prev in (bg, val), "multiple stamps must not conflict"
            out[r, c] = val


def _neighbor_side(src: tuple[int, int], dst: tuple[int, int]) -> str:
    sr, sc = src
    dr, dc = dst
    if dr == sr - 1 and dc == sc:
        return "top"
    if dr == sr + 1 and dc == sc:
        return "bottom"
    if dr == sr and dc == sc - 1:
        return "left"
    if dr == sr and dc == sc + 1:
        return "right"
    raise AssertionError("expected an orthogonally adjacent region")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each template through visible separator gaps into neighboring blank regions.

    The grid is partitioned by one divider color whose horizontal and vertical lines each have
    a single background-colored gap. A non-empty region acts as a template source. Whenever the
    gap on one of its bordering divider lines lies within that region's span, the gap connects
    this source to the orthogonally adjacent blank region on the other side. The template's tight
    bounding-box patch is stamped into that blank region, flipped across the relevant axis,
    aligned to the gap, and placed flush against the shared divider.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    divider, sep_rows, sep_cols, row_gaps, col_gaps = _detect_dividers(g, bg)
    row_bands = _bands_from_separators(sep_rows, g.shape[0])
    col_bands = _bands_from_separators(sep_cols, g.shape[1])

    regions: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    patches: dict[tuple[int, int], np.ndarray] = {}
    occupied: dict[tuple[int, int], bool] = {}

    for i, (r0, r1) in enumerate(row_bands):
        for j, (c0, c1) in enumerate(col_bands):
            box = (r0, r1, c0, c1)
            regions[(i, j)] = box
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(region == divider), "divider color must stay on separator lines"
            filled = bool(np.any(region != bg))
            occupied[(i, j)] = filled
            if filled:
                patches[(i, j)] = _extract_patch(region, bg)

    out = g.copy()
    max_i = len(row_bands) - 1
    max_j = len(col_bands) - 1

    for (i, j), patch in patches.items():
        r0, r1, c0, c1 = regions[(i, j)]

        if i > 0:
            gap_c = row_gaps[sep_rows[i - 1]]
            if c0 <= gap_c <= c1 and not occupied[(i - 1, j)]:
                _stamp_patch(out, patch, regions[(i - 1, j)], "top", gap_c, bg)

        if i < max_i:
            gap_c = row_gaps[sep_rows[i]]
            if c0 <= gap_c <= c1 and not occupied[(i + 1, j)]:
                _stamp_patch(out, patch, regions[(i + 1, j)], "bottom", gap_c, bg)

        if j > 0:
            gap_r = col_gaps[sep_cols[j - 1]]
            if r0 <= gap_r <= r1 and not occupied[(i, j - 1)]:
                _stamp_patch(out, patch, regions[(i, j - 1)], "left", gap_r, bg)

        if j < max_j:
            gap_r = col_gaps[sep_cols[j]]
            if r0 <= gap_r <= r1 and not occupied[(i, j + 1)]:
                _stamp_patch(out, patch, regions[(i, j + 1)], "right", gap_r, bg)

    return out


def _layout_from_sizes(
    row_sizes: list[int],
    col_sizes: list[int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int], list[int]]:
    row_bands = []
    sep_rows = []
    cursor = 0
    for idx, size in enumerate(row_sizes):
        row_bands.append((cursor, cursor + size - 1))
        cursor += size
        if idx + 1 < len(row_sizes):
            sep_rows.append(cursor)
            cursor += 1

    col_bands = []
    sep_cols = []
    cursor = 0
    for idx, size in enumerate(col_sizes):
        col_bands.append((cursor, cursor + size - 1))
        cursor += size
        if idx + 1 < len(col_sizes):
            sep_cols.append(cursor)
            cursor += 1

    return row_bands, col_bands, sep_rows, sep_cols


def _build_gap_graph(
    n_rows: int,
    n_cols: int,
) -> tuple[list[int], list[int], dict[tuple[int, int], set[tuple[int, int]]]]:
    horiz_choices = [random.randrange(n_cols) for _ in range(n_rows - 1)]
    vert_choices = [random.randrange(n_rows) for _ in range(n_cols - 1)]
    adj: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)

    for i in range(n_rows):
        for j in range(n_cols):
            adj[(i, j)]

    for i, j in enumerate(horiz_choices):
        a = (i, j)
        b = (i + 1, j)
        adj[a].add(b)
        adj[b].add(a)

    for j, i in enumerate(vert_choices):
        a = (i, j)
        b = (i, j + 1)
        adj[a].add(b)
        adj[b].add(a)

    return horiz_choices, vert_choices, adj


def _orientation_flags(
    sources: list[tuple[int, int]],
    adj: dict[tuple[int, int], set[tuple[int, int]]],
) -> tuple[bool, bool]:
    has_vertical = False
    has_horizontal = False
    for src in sources:
        for dst in adj[src]:
            side = _neighbor_side(src, dst)
            if side in {"top", "bottom"}:
                has_vertical = True
            else:
                has_horizontal = True
    return has_vertical, has_horizontal


def _choose_sources(
    adj: dict[tuple[int, int], set[tuple[int, int]]],
) -> list[tuple[int, int]]:
    candidates = [node for node, neighbors in adj.items() if neighbors]
    best: list[tuple[int, int]] = []
    best_score = (-1, -1, -1)

    for _ in range(SOURCE_SET_RETRIES):
        order = candidates[:]
        random.shuffle(order)
        chosen: list[tuple[int, int]] = []
        used: set[tuple[int, int]] = set()
        for node in order:
            neighbors = adj[node]
            if node in used:
                continue
            if any(neighbor in used for neighbor in neighbors):
                continue
            chosen.append(node)
            used.add(node)
            used.update(neighbors)

        has_vertical, has_horizontal = _orientation_flags(chosen, adj)
        score = (len(chosen), int(has_vertical) + int(has_horizontal), sum(len(adj[node]) for node in chosen))
        if score > best_score:
            best = chosen
            best_score = score

    assert best, "gap graph must expose at least one usable template source"
    limit = min(4, len(best))
    keep = random.randint(1, limit)
    random.shuffle(best)
    return best[:keep]


def _grow_connected_cells(height: int, width: int, target: int) -> set[tuple[int, int]] | None:
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            if r > 0 and (r - 1, c) not in cells:
                frontier.add((r - 1, c))
            if r + 1 < height and (r + 1, c) not in cells:
                frontier.add((r + 1, c))
            if c > 0 and (r, c - 1) not in cells:
                frontier.add((r, c - 1))
            if c + 1 < width and (r, c + 1) not in cells:
                frontier.add((r, c + 1))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    return cells


def _touches_all_sides(cells: set[tuple[int, int]], height: int, width: int) -> bool:
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    return 0 in rows and height - 1 in rows and 0 in cols and width - 1 in cols


def _make_patch(
    region_h: int,
    region_w: int,
    active_sides: list[str],
    divider: int,
) -> np.ndarray | None:
    need_tall = any(side in {"top", "bottom"} for side in active_sides)
    need_wide = any(side in {"left", "right"} for side in active_sides)
    min_h = 3 if need_tall else 1
    min_w = 3 if need_wide else 1
    max_h = max(min_h, min(5, region_h - 2))
    max_w = max(min_w, min(5, region_w - 2))
    h_opts = [value for value in range(min_h, max_h + 1, 2)]
    w_opts = [value for value in range(min_w, max_w + 1, 2)]
    if not h_opts or not w_opts:
        return None

    usable_colors = [color for color in PALETTE if color != divider]
    region_cap = max(1, (region_h * region_w) // 2)

    dim_pairs = [(ph, pw) for ph in h_opts for pw in w_opts]
    random.shuffle(dim_pairs)

    for _ in range(PATCH_RETRIES):
        ph, pw = random.choice(dim_pairs)
        area_min = ph + pw - 1
        area_max = min(ph * pw, region_cap)
        if area_max < area_min:
            continue

        cells = _grow_connected_cells(ph, pw, random.randint(area_min, area_max))
        if cells is None or not _touches_all_sides(cells, ph, pw):
            continue

        patch = np.zeros((ph, pw), dtype=int)
        color_count = random.randint(1, min(3, len(usable_colors)))
        palette = random.sample(usable_colors, color_count)
        ordered_cells = list(cells)
        random.shuffle(ordered_cells)

        for idx, (r, c) in enumerate(ordered_cells):
            color = palette[idx] if idx < len(palette) else random.choice(palette)
            patch[r, c] = color

        if need_wide and np.array_equal(patch, np.fliplr(patch)):
            continue
        if need_tall and np.array_equal(patch, np.flipud(patch)):
            continue
        return patch

    return None


def _paint_patch(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    patch: np.ndarray,
) -> tuple[int, int]:
    r0, r1, c0, c1 = box
    ph, pw = patch.shape
    start_r = random.randint(r0, r1 - ph + 1)
    start_c = random.randint(c0, c1 - pw + 1)
    mask = patch != BG
    region = grid[start_r : start_r + ph, start_c : start_c + pw]
    assert not np.any(region[mask] != BG)
    region[mask] = patch[mask]
    return start_r, start_c


def _render_expected(
    inp: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    sep_rows: list[int],
    sep_cols: list[int],
    row_gap_pos: dict[int, int],
    col_gap_pos: dict[int, int],
    sources: list[tuple[int, int]],
    source_patches: dict[tuple[int, int], np.ndarray],
    adj: dict[tuple[int, int], set[tuple[int, int]]],
) -> np.ndarray:
    out = inp.copy()
    regions = {
        (i, j): (row_bands[i][0], row_bands[i][1], col_bands[j][0], col_bands[j][1])
        for i in range(len(row_bands))
        for j in range(len(col_bands))
    }

    for src in sources:
        patch = source_patches[src]
        for dst in adj[src]:
            side = _neighbor_side(src, dst)
            if side == "top":
                gap = row_gap_pos[sep_rows[src[0] - 1]]
            elif side == "bottom":
                gap = row_gap_pos[sep_rows[src[0]]]
            elif side == "left":
                gap = col_gap_pos[sep_cols[src[1] - 1]]
            else:
                gap = col_gap_pos[sep_cols[src[1]]]
            _stamp_patch(out, patch, regions[dst], side, gap, BG)

    return out


def _generate_attempt() -> np.ndarray | None:
    n_rows = random.randint(3, 5)
    n_cols = random.randint(3, 5)
    row_sizes = [random.randint(5, 8) for _ in range(n_rows)]
    col_sizes = [random.randint(5, 8) for _ in range(n_cols)]
    row_bands, col_bands, sep_rows, sep_cols = _layout_from_sizes(row_sizes, col_sizes)
    divider = random.choice(PALETTE)

    horiz_choices, vert_choices, adj = _build_gap_graph(n_rows, n_cols)
    sources = _choose_sources(adj)

    source_patches: dict[tuple[int, int], np.ndarray] = {}
    for src in sources:
        i, j = src
        box = (row_bands[i][0], row_bands[i][1], col_bands[j][0], col_bands[j][1])
        active_sides = sorted((_neighbor_side(src, dst) for dst in adj[src]), key=SIDE_ORDER.index)
        patch = _make_patch(box[1] - box[0] + 1, box[3] - box[2] + 1, active_sides, divider)
        if patch is None:
            return None
        source_patches[src] = patch

    row_gap_pos: dict[int, int] = {}
    for idx, region_col in enumerate(horiz_choices):
        sep_r = sep_rows[idx]
        c0, c1 = col_bands[region_col]
        src = (idx, region_col) if (idx, region_col) in source_patches else (idx + 1, region_col)
        if src in source_patches:
            rng = _center_range(c0, c1, source_patches[src].shape[1])
            if rng is None:
                return None
            row_gap_pos[sep_r] = random.randint(rng[0], rng[1])
        else:
            row_gap_pos[sep_r] = random.randint(c0, c1)

    col_gap_pos: dict[int, int] = {}
    for idx, region_row in enumerate(vert_choices):
        sep_c = sep_cols[idx]
        r0, r1 = row_bands[region_row]
        src = (region_row, idx) if (region_row, idx) in source_patches else (region_row, idx + 1)
        if src in source_patches:
            rng = _center_range(r0, r1, source_patches[src].shape[0])
            if rng is None:
                return None
            col_gap_pos[sep_c] = random.randint(rng[0], rng[1])
        else:
            col_gap_pos[sep_c] = random.randint(r0, r1)

    height = row_bands[-1][1] + 1
    width = col_bands[-1][1] + 1
    inp = np.full((height, width), BG, dtype=int)
    inp[sep_rows, :] = divider
    inp[:, sep_cols] = divider

    for sep_r, gap_c in row_gap_pos.items():
        inp[sep_r, gap_c] = BG
    for sep_c, gap_r in col_gap_pos.items():
        inp[gap_r, sep_c] = BG

    for src, patch in source_patches.items():
        i, j = src
        box = (row_bands[i][0], row_bands[i][1], col_bands[j][0], col_bands[j][1])
        _paint_patch(inp, box, patch)

    expected = _render_expected(
        inp,
        row_bands,
        col_bands,
        sep_rows,
        sep_cols,
        row_gap_pos,
        col_gap_pos,
        sources,
        source_patches,
        adj,
    )

    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
