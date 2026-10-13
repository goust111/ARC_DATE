import random

import numpy as np


BG = 0
LATTICE = 8
DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
PALETTE = [1, 2, 3, 4, 5, 6, 7, 9]
MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _parse_legend(row: np.ndarray) -> dict[int, int]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for c, value in enumerate(row):
        if int(value) != BG and start is None:
            start = int(c)
        elif int(value) == BG and start is not None:
            runs.append((start, int(c) - 1))
            start = None
    if start is not None:
        runs.append((start, row.shape[0] - 1))

    mapping: dict[int, int] = {}
    for c0, c1 in runs:
        assert c1 - c0 + 1 == 2, "each legend entry must be an adjacent pixel pair"
        src = int(row[c0])
        dst = int(row[c0 + 1])
        assert src not in (BG, LATTICE)
        assert dst not in (BG, LATTICE)
        assert src != dst
        assert src not in mapping, "legend source colors must be unique"
        mapping[src] = dst

    assert len(mapping) >= 2, "expected at least two legend mappings"
    return mapping


def _full_line_positions(sub: np.ndarray, lattice_color: int, axis: int) -> list[int]:
    positions: list[int] = []
    if axis == 0:
        for r in range(sub.shape[0]):
            if bool(np.all(sub[r, :] == lattice_color)):
                positions.append(int(r))
    else:
        for c in range(sub.shape[1]):
            if bool(np.all(sub[:, c] == lattice_color)):
                positions.append(int(c))
    return positions


def _cell_size(lines: list[int]) -> int:
    gaps = [b - a - 1 for a, b in zip(lines, lines[1:])]
    assert gaps and min(gaps) >= 2, "each cavity interior must be at least 2 cells wide"
    assert len(set(gaps)) == 1, "cavity grid must be regular"
    return int(gaps[0])


def _cavity_bounds(h_lines: list[int], v_lines: list[int], i: int, j: int) -> tuple[int, int, int, int]:
    return h_lines[i] + 1, v_lines[j] + 1, h_lines[i + 1] - 1, v_lines[j + 1] - 1


def _locate_marker(
    r: int,
    c: int,
    h_lines: list[int],
    v_lines: list[int],
) -> tuple[int, int, tuple[int, int]]:
    i = int(np.searchsorted(h_lines, r) - 1)
    j = int(np.searchsorted(v_lines, c) - 1)
    assert 0 <= i < len(h_lines) - 1 and 0 <= j < len(v_lines) - 1
    r0, c0, r1, c1 = _cavity_bounds(h_lines, v_lines, i, j)
    assert r0 <= r <= r1 and c0 <= c <= c1

    if (r, c) == (r0, c0):
        return i, j, (-1, -1)
    if (r, c) == (r0, c1):
        return i, j, (-1, 1)
    if (r, c) == (r1, c0):
        return i, j, (1, -1)
    if (r, c) == (r1, c1):
        return i, j, (1, 1)
    raise AssertionError("marker must sit on a cavity corner cell")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill diagonal neighboring cavities according to corner-coded markers and a bottom color legend.

    Input structure:
    - The main body is a single-color lattice frame enclosing a regular array of black cavities.
    - Some cavity corner cells contain singleton colored markers; the marker's corner position is one of NW, NE, SW, or SE.
    - The last row is a legend made of adjacent pixel pairs where the left color is a marker color and the right color is its fill color.

    Rule:
    - Parse the legend to map every marker color to an output fill color.
    - For each marker, determine which cavity contains it and read its corner position to obtain a diagonal direction.
    - Clear all markers back to black, then fill every black cell of the diagonally adjacent target cavity in that direction with the legend-mapped fill color.

    Key insight:
    - Each marker carries two instructions at once: its color says what color to paint, while its corner position says which diagonal neighboring cavity to paint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    legend = _parse_legend(g[-1])

    body = g[:-1]
    body_mask = body != BG
    r0, c0, r1, c1 = _bbox(body_mask)
    sub = body[r0 : r1 + 1, c0 : c1 + 1]

    non_bg = sub[sub != BG]
    assert non_bg.size > 0
    colors, counts = np.unique(non_bg, return_counts=True)
    lattice_color = int(colors[np.argmax(counts)])

    h_local = _full_line_positions(sub, lattice_color, axis=0)
    v_local = _full_line_positions(sub, lattice_color, axis=1)
    assert len(h_local) >= 3 and len(v_local) >= 3

    _cell_size(h_local)
    _cell_size(v_local)

    h_lines = [r0 + r for r in h_local]
    v_lines = [c0 + c for c in v_local]
    n_rows = len(h_lines) - 1
    n_cols = len(v_lines) - 1

    marker_coords = np.argwhere((body != BG) & (body != lattice_color))
    assert marker_coords.size > 0, "expected at least one marker"

    instructions: list[tuple[tuple[int, int], int, tuple[int, int]]] = []
    for rr, cc in marker_coords:
        color = int(g[rr, cc])
        assert color in legend, "every marker color must appear in the legend"
        i, j, (dr, dc) = _locate_marker(int(rr), int(cc), h_lines, v_lines)
        ti = i + dr
        tj = j + dc
        assert 0 <= ti < n_rows and 0 <= tj < n_cols, "marker must point to a full diagonal neighbor cavity"
        instructions.append(((ti, tj), int(legend[color]), (int(rr), int(cc))))

    target_to_color: dict[tuple[int, int], int] = {}
    for target, fill_color, _pos in instructions:
        if target in target_to_color:
            assert target_to_color[target] == fill_color, "conflicting fills for the same target cavity"
        target_to_color[target] = fill_color

    out = g.copy()
    for _target, _fill, (rr, cc) in instructions:
        out[rr, cc] = BG

    for (ti, tj), fill_color in target_to_color.items():
        cr0, cc0, cr1, cc1 = _cavity_bounds(h_lines, v_lines, ti, tj)
        region = out[cr0 : cr1 + 1, cc0 : cc1 + 1].copy()
        region[region == BG] = fill_color
        out[cr0 : cr1 + 1, cc0 : cc1 + 1] = region

    return out


def _build_lattice(n_rows: int, n_cols: int, cell_h: int, cell_w: int) -> np.ndarray:
    h = n_rows * cell_h + n_rows + 1
    w = n_cols * cell_w + n_cols + 1
    lattice = np.full((h, w), BG, dtype=int)
    for r in range(0, h, cell_h + 1):
        lattice[r, :] = LATTICE
    for c in range(0, w, cell_w + 1):
        lattice[:, c] = LATTICE
    return lattice


def _marker_cell(i: int, j: int, dr: int, dc: int, cell_h: int, cell_w: int) -> tuple[int, int]:
    r0 = 1 + i * (cell_h + 1)
    c0 = 1 + j * (cell_w + 1)
    r1 = r0 + cell_h - 1
    c1 = c0 + cell_w - 1
    rr = r0 if dr < 0 else r1
    cc = c0 if dc < 0 else c1
    return int(rr), int(cc)


def _target_bounds(i: int, j: int, cell_h: int, cell_w: int) -> tuple[int, int, int, int]:
    r0 = 1 + i * (cell_h + 1)
    c0 = 1 + j * (cell_w + 1)
    r1 = r0 + cell_h - 1
    c1 = c0 + cell_w - 1
    return int(r0), int(c0), int(r1), int(c1)


def _all_candidates(n_rows: int, n_cols: int) -> dict[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]]:
    by_dir: dict[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]] = {d: [] for d in DIRS}
    for i in range(n_rows):
        for j in range(n_cols):
            src = (i, j)
            for dr, dc in DIRS:
                ti = i + dr
                tj = j + dc
                if 0 <= ti < n_rows and 0 <= tj < n_cols:
                    by_dir[(dr, dc)].append((src, (ti, tj), (dr, dc)))
    return by_dir


def _direction_list(count: int) -> list[tuple[int, int]]:
    if count <= 4:
        return random.sample(DIRS, count)
    out = list(DIRS)
    out.extend(random.choice(DIRS) for _ in range(count - 4))
    random.shuffle(out)
    return out


def _pick_instructions(
    n_rows: int,
    n_cols: int,
    min_colors: int,
) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] | None:
    candidates = _all_candidates(n_rows, n_cols)
    max_markers = min(6, (n_rows * n_cols) // 2)
    if max_markers < min_colors:
        return None

    counts = list(range(max(min_colors, 2), max_markers + 1))
    random.shuffle(counts)
    counts.sort(key=lambda value: (value < 4, -value))

    for count in counts:
        directions = _direction_list(count)
        for _ in range(300):
            used_src: set[tuple[int, int]] = set()
            used_tgt: set[tuple[int, int]] = set()
            chosen: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
            ok = True
            for direction in directions:
                pool = [
                    item
                    for item in candidates[direction]
                    if item[0] not in used_src
                    and item[1] not in used_tgt
                    and item[0] not in used_tgt
                    and item[1] not in used_src
                ]
                if not pool:
                    ok = False
                    break
                src, tgt, diag = random.choice(pool)
                chosen.append((src, tgt, diag))
                used_src.add(src)
                used_tgt.add(tgt)
            if ok and len(chosen) >= min_colors:
                return chosen
    return None


def _legend_mapping(width: int) -> tuple[dict[int, int], list[int]] | None:
    max_pairs = min(4, len(PALETTE) // 2, (width + 1) // 3)
    if max_pairs < 2:
        return None
    n_pairs = random.randint(2, max_pairs)
    marker_colors = random.sample(PALETTE, n_pairs)
    fill_pool = [c for c in PALETTE if c not in marker_colors]
    fill_colors = random.sample(fill_pool, n_pairs)
    return dict(zip(marker_colors, fill_colors)), marker_colors


def _render_input(
    n_rows: int,
    n_cols: int,
    cell_h: int,
    cell_w: int,
    top: int,
    left: int,
    gap: int,
    right_pad: int,
    mapping: dict[int, int],
    instructions: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]],
) -> tuple[np.ndarray, dict]:
    lattice = _build_lattice(n_rows, n_cols, cell_h, cell_w)
    lat_h, lat_w = lattice.shape
    height = top + lat_h + gap + 1
    width = left + lat_w + right_pad
    grid = np.full((height, width), BG, dtype=int)
    grid[top : top + lat_h, left : left + lat_w] = lattice

    marker_colors = list(mapping)
    color_assignments = marker_colors + random.choices(marker_colors, k=len(instructions) - len(marker_colors))
    random.shuffle(color_assignments)

    spec: list[dict] = []
    for color, (src, tgt, diag) in zip(color_assignments, instructions):
        rr, cc = _marker_cell(src[0], src[1], diag[0], diag[1], cell_h, cell_w)
        rr += top
        cc += left
        grid[rr, cc] = int(color)
        spec.append(
            {
                "marker": (rr, cc),
                "target": tgt,
                "fill": int(mapping[color]),
            }
        )

    legend_len = 3 * len(mapping) - 1
    start = random.randint(0, width - legend_len)
    order = list(mapping.items())
    random.shuffle(order)
    col = start
    for src_color, fill_color in order:
        grid[-1, col] = int(src_color)
        grid[-1, col + 1] = int(fill_color)
        col += 3

    meta = {
        "top": top,
        "left": left,
        "cell_h": cell_h,
        "cell_w": cell_w,
        "spec": spec,
    }
    return grid, meta


def _expected_output(inp: np.ndarray, meta: dict) -> np.ndarray:
    out = inp.copy()
    top = int(meta["top"])
    left = int(meta["left"])
    cell_h = int(meta["cell_h"])
    cell_w = int(meta["cell_w"])
    target_to_color: dict[tuple[int, int], int] = {}

    for item in meta["spec"]:
        rr, cc = item["marker"]
        out[int(rr), int(cc)] = BG
        target = tuple(item["target"])
        fill = int(item["fill"])
        if target in target_to_color:
            assert target_to_color[target] == fill
        target_to_color[target] = fill

    for (i, j), fill in target_to_color.items():
        r0, c0, r1, c1 = _target_bounds(int(i), int(j), cell_h, cell_w)
        r0 += top
        r1 += top
        c0 += left
        c1 += left
        out[r0 : r1 + 1, c0 : c1 + 1] = fill

    return out


def _choose_geometry() -> tuple[int, int, int, int, int, int, int, int] | None:
    options: list[tuple[int, int, int, int, int, int, int, int]] = []
    for n_rows in range(3, 7):
        for n_cols in range(3, 7):
            for cell_h in range(2, 5):
                for cell_w in range(2, 5):
                    lat_h = n_rows * cell_h + n_rows + 1
                    lat_w = n_cols * cell_w + n_cols + 1
                    for top in range(0, 3):
                        for left in range(0, 3):
                            for gap in range(1, 4):
                                for right_pad in range(0, 3):
                                    height = top + lat_h + gap + 1
                                    width = left + lat_w + right_pad
                                    if height <= 30 and width <= 30 and (width + 1) // 3 >= 2:
                                        options.append((n_rows, n_cols, cell_h, cell_w, top, left, gap, right_pad))
    if not options:
        return None
    return random.choice(options)


def _gen_attempt() -> np.ndarray | None:
    geometry = _choose_geometry()
    if geometry is None:
        return None
    n_rows, n_cols, cell_h, cell_w, top, left, gap, right_pad = geometry

    width = left + (n_cols * cell_w + n_cols + 1) + right_pad
    legend_info = _legend_mapping(width)
    if legend_info is None:
        return None
    mapping, marker_colors = legend_info

    instructions = _pick_instructions(n_rows, n_cols, len(marker_colors))
    if instructions is None:
        return None

    inp, meta = _render_input(
        n_rows=n_rows,
        n_cols=n_cols,
        cell_h=cell_h,
        cell_w=cell_w,
        top=top,
        left=left,
        gap=gap,
        right_pad=right_pad,
        mapping=mapping,
        instructions=instructions,
    )
    expected = _expected_output(inp, meta)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    unique_directions = {tuple(item[2]) for item in instructions}
    if len(instructions) >= 4 and len(unique_directions) < 4:
        return None
    if len(set(item["fill"] for item in meta["spec"])) < 2:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid angle-cavity input")
