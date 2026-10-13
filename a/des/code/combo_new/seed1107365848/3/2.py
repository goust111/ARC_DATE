import numpy as np


BG = 0
MARKER = 2
SEPARATOR = 5
SEED_COLORS = (1, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c),
        (r + 1, c),
        (r, c - 1),
        (r, c + 1),
    ]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _separator_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    row_colors = {int(grid[r, 0]) for r in range(h) if np.all(grid[r, :] == grid[r, 0])}
    col_colors = {int(grid[0, c]) for c in range(w) if np.all(grid[:, c] == grid[0, c])}
    candidates = row_colors & col_colors
    assert candidates, "expected a separator color that forms full rows and full columns"
    best = None
    best_score = -1
    for color in candidates:
        score = sum(int(np.all(grid[r, :] == color)) for r in range(h))
        score += sum(int(np.all(grid[:, c] == color)) for c in range(w))
        if score > best_score:
            best = int(color)
            best_score = int(score)
    assert best is not None
    return int(best)


def _region_intervals(n: int, cuts: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + [int(x) for x in cuts] + [int(n)]
    parts: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:]):
        if right - left > 1:
            parts.append((int(left + 1), int(right - 1)))
    return parts


def _tile_origins(start: int, tile_size: int, total: int, direction: int, enabled: bool) -> list[int]:
    if not enabled:
        return [int(start)]
    pos = int(start)
    origins: list[int] = []
    step = int(direction * tile_size)
    while pos < total and pos + tile_size - 1 >= 0:
        origins.append(int(pos))
        pos += int(step)
    return origins


def _render_region(
    h: int,
    w: int,
    seed_patch: np.ndarray,
    seed_box: tuple[int, int, int, int],
    horizontal: bool,
    vertical: bool,
    bg: int,
) -> np.ndarray:
    sr0, sc0, sr1, sc1 = map(int, seed_box)
    sh, sw = int(sr1 - sr0 + 1), int(sc1 - sc0 + 1)
    row_dir = 1 if sr0 == 0 else -1
    col_dir = 1 if sc0 == 0 else -1
    row_origins = _tile_origins(sr0, sh, h, row_dir, vertical)
    col_origins = _tile_origins(sc0, sw, w, col_dir, horizontal)
    out = np.full((h, w), int(bg), dtype=int)
    filled = np.argwhere(seed_patch != int(bg))
    for orow in row_origins:
        for ocol in col_origins:
            for pr, pc in filled:
                rr = int(orow + pr)
                cc = int(ocol + pc)
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = int(seed_patch[int(pr), int(pc)])
    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    sep = _separator_color(g)
    residual = g[g != int(sep)]
    assert residual.size > 0, "grid must contain non-separator cells"
    bg = int(np.bincount(residual).argmax())

    sep_rows = [int(r) for r in range(h) if np.all(g[r, :] == int(sep))]
    sep_cols = [int(c) for c in range(w) if np.all(g[:, c] == int(sep))]
    row_parts = _region_intervals(h, sep_rows)
    col_parts = _region_intervals(w, sep_cols)
    assert row_parts and col_parts, "expected at least one region in each dimension"

    out = g.copy()

    for r0, r1 in row_parts:
        for c0, c1 in col_parts:
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            rh, rw = region.shape
            mask = (region != int(bg)) & (region != int(sep))
            comps = sorted(_components(mask), key=len, reverse=True)
            assert comps, "each region must contain a seed pattern"

            seed_comp = comps[0]
            markers = [comp[0] for comp in comps[1:]]
            assert all(len(comp) == 1 for comp in comps[1:]), "all non-seed objects must be singleton markers"
            assert 1 <= len(markers) <= 2, "expected one or two markers per region"

            sr0, sc0, sr1, sc1 = _bbox(seed_comp)
            touches_top = sr0 == 0
            touches_bottom = sr1 == rh - 1
            touches_left = sc0 == 0
            touches_right = sc1 == rw - 1
            assert touches_top ^ touches_bottom, "seed must be anchored to top or bottom"
            assert touches_left ^ touches_right, "seed must be anchored to left or right"

            horizontal = any(int(mc) in (0, rw - 1) and 0 < int(mr) < rh - 1 for mr, mc in markers)
            vertical = any(int(mr) in (0, rh - 1) and 0 < int(mc) < rw - 1 for mr, mc in markers)
            assert horizontal or vertical, "markers must encode at least one tiling axis"

            seed_patch = region[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
            rendered = _render_region(
                rh,
                rw,
                seed_patch,
                (sr0, sc0, sr1, sc1),
                horizontal=horizontal,
                vertical=vertical,
                bg=bg,
            )
            for mr, mc in markers:
                rendered[int(mr), int(mc)] = int(bg)
            out[r0 : r1 + 1, c0 : c1 + 1] = rendered

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile each region's corner seed pattern along the axis or axes indicated by opposite-edge markers.

    Input structure:
    - The background is black, and one uniform separator color forms full horizontal and vertical lines that partition the grid into rectangular regions.
    - Each region contains one small multi-color seed pattern whose bounding box sits in a corner of the region.
    - The same region also contains one or two singleton marker pixels on the edge or edges opposite that seed corner.

    Rule:
    - Detect the separator rows and columns to recover every rectangular region.
    - In each region, treat the largest non-background component as the seed pattern and the remaining singleton edge pixels as direction markers.
    - A marker on the left or right edge means tile the seed horizontally; a marker on the top or bottom edge means tile it vertically; if both appear, tile in both directions as a repeated 2D wallpaper.
    - Copy the seed by whole bounding-box steps, clip partial copies at the region boundary, remove the marker pixels, and keep the separator lines unchanged.

    Key insight:
    - The singleton markers are not target locations; their edge orientation tells you which repetition axis to use for the corner-anchored seed tile.
    """
    return _solve(grid)


def _random_connected_mask(rng: np.random.Generator, sh: int, sw: int) -> np.ndarray | None:
    area = int(sh * sw)
    low = max(3, area // 2)
    for _ in range(200):
        target = int(rng.integers(low, area))
        start = (int(rng.integers(0, sh)), int(rng.integers(0, sw)))
        cells = {start}
        frontier = [start]
        while frontier and len(cells) < target:
            base = frontier[int(rng.integers(0, len(frontier)))]
            br, bc = base
            neighbors = []
            for nr, nc in ((br - 1, bc), (br + 1, bc), (br, bc - 1), (br, bc + 1)):
                if 0 <= nr < sh and 0 <= nc < sw and (nr, nc) not in cells:
                    neighbors.append((int(nr), int(nc)))
            if not neighbors:
                frontier.remove(base)
                continue
            pick = neighbors[int(rng.integers(0, len(neighbors)))]
            cells.add(pick)
            frontier.append(pick)
        mask = np.zeros((sh, sw), dtype=bool)
        for r, c in cells:
            mask[int(r), int(c)] = True
        if not mask[0, :].any():
            continue
        if not mask[-1, :].any():
            continue
        if not mask[:, 0].any():
            continue
        if not mask[:, -1].any():
            continue
        if bool(mask.all()):
            continue
        return mask
    return None


def _make_seed_patch(rng: np.random.Generator, sh: int, sw: int) -> np.ndarray | None:
    mask = _random_connected_mask(rng, sh, sw)
    if mask is None:
        return None
    patch = np.full((sh, sw), BG, dtype=int)
    occupied = np.argwhere(mask)
    n_colors = min(int(len(SEED_COLORS)), max(2, int(rng.integers(2, min(5, len(occupied) + 1)))))
    colors = rng.choice(np.array(SEED_COLORS, dtype=int), size=n_colors, replace=False)
    order = occupied.copy()
    rng.shuffle(order)
    for i, (r, c) in enumerate(order):
        if i < n_colors:
            color = int(colors[i])
        else:
            color = int(colors[int(rng.integers(0, len(colors)))])
        patch[int(r), int(c)] = int(color)
    return patch


def _seed_box_for_corner(h: int, w: int, sh: int, sw: int, corner: str) -> tuple[int, int, int, int]:
    sr0 = 0 if "top" in corner else h - sh
    sc0 = 0 if "left" in corner else w - sw
    return int(sr0), int(sc0), int(sr0 + sh - 1), int(sc0 + sw - 1)


def _sorted_inner_positions(length: int) -> list[int]:
    center = (length - 1) / 2.0
    vals = list(range(1, max(1, length - 1)))
    vals = [v for v in vals if 0 < v < length - 1]
    vals.sort(key=lambda v: (abs(v - center), v))
    return vals


def _pick_marker_index(candidates: list[int], rng: np.random.Generator) -> int | None:
    if not candidates:
        return None
    head = candidates[: min(3, len(candidates))]
    return int(head[int(rng.integers(0, len(head)))])


def _choose_markers(
    rng: np.random.Generator,
    preview: np.ndarray,
    corner: str,
    mode: str,
) -> list[tuple[int, int]] | None:
    h, w = preview.shape
    markers: list[tuple[int, int]] = []

    if mode in ("v", "hv"):
        edge_r = h - 1 if "top" in corner else 0
        cols = [c for c in _sorted_inner_positions(w) if int(preview[edge_r, c]) == BG]
        pick = _pick_marker_index(cols, rng)
        if pick is None:
            return None
        markers.append((int(edge_r), int(pick)))

    if mode in ("h", "hv"):
        edge_c = w - 1 if "left" in corner else 0
        rows = [r for r in _sorted_inner_positions(h) if int(preview[r, edge_c]) == BG]
        pick = _pick_marker_index(rows, rng)
        if pick is None:
            return None
        markers.append((int(pick), int(edge_c)))

    return markers


def _balanced_modes(rng: np.random.Generator, total: int) -> list[str]:
    base = ["h", "v", "hv"]
    modes = [base[i % 3] for i in range(total)]
    rng.shuffle(modes)
    return modes


def _sample_layout(rng: np.random.Generator) -> tuple[list[int], list[int]] | None:
    nr, nc = [(2, 2), (2, 3), (3, 2)][int(rng.integers(0, 3))]
    heights = [int(rng.integers(5, 10)) for _ in range(nr)]
    widths = [int(rng.integers(5, 10)) for _ in range(nc)]
    total_h = sum(heights) + (nr - 1)
    total_w = sum(widths) + (nc - 1)
    if total_h > 30 or total_w > 30:
        return None
    return heights, widths


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    layout = _sample_layout(rng)
    if layout is None:
        return None
    heights, widths = layout
    nr, nc = len(heights), len(widths)
    h = sum(heights) + (nr - 1)
    w = sum(widths) + (nc - 1)
    grid = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)

    row_edges = []
    running = 0
    for i, size in enumerate(heights):
        running += int(size)
        if i < nr - 1:
            row_edges.append(int(running))
            running += 1
    col_edges = []
    running = 0
    for i, size in enumerate(widths):
        running += int(size)
        if i < nc - 1:
            col_edges.append(int(running))
            running += 1

    for r in row_edges:
        grid[int(r), :] = SEPARATOR
        expected[int(r), :] = SEPARATOR
    for c in col_edges:
        grid[:, int(c)] = SEPARATOR
        expected[:, int(c)] = SEPARATOR

    row_parts = _region_intervals(h, row_edges)
    col_parts = _region_intervals(w, col_edges)
    modes = _balanced_modes(rng, len(row_parts) * len(col_parts))
    corners = ["top-left", "top-right", "bottom-left", "bottom-right"]

    mode_idx = 0
    for r0, r1 in row_parts:
        for c0, c1 in col_parts:
            rh = int(r1 - r0 + 1)
            rw = int(c1 - c0 + 1)
            mode = modes[mode_idx]
            mode_idx += 1
            placed = False
            for _ in range(200):
                corner = corners[int(rng.integers(0, len(corners)))]
                max_sh = min(4, rh - 2)
                max_sw = min(4, rw - 2)
                if max_sh < 2 or max_sw < 2:
                    return None
                sh = int(rng.integers(2, max_sh + 1))
                sw = int(rng.integers(2, max_sw + 1))
                patch = _make_seed_patch(rng, sh, sw)
                if patch is None:
                    continue
                seed_box = _seed_box_for_corner(rh, rw, sh, sw, corner)
                preview = _render_region(
                    rh,
                    rw,
                    patch,
                    seed_box,
                    horizontal=mode in ("h", "hv"),
                    vertical=mode in ("v", "hv"),
                    bg=BG,
                )
                markers = _choose_markers(rng, preview, corner, mode)
                if markers is None:
                    continue

                region_in = np.full((rh, rw), BG, dtype=int)
                sr0, sc0, sr1, sc1 = seed_box
                region_in[sr0 : sr1 + 1, sc0 : sc1 + 1] = patch
                for mr, mc in markers:
                    if region_in[int(mr), int(mc)] != BG:
                        break
                    region_in[int(mr), int(mc)] = MARKER
                else:
                    region_out = preview.copy()
                    for mr, mc in markers:
                        region_out[int(mr), int(mc)] = BG
                    grid[r0 : r1 + 1, c0 : c1 + 1] = region_in
                    expected[r0 : r1 + 1, c0 : c1 + 1] = region_out
                    placed = True
                    break
            if not placed:
                return None

    solved = _solve(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tiling-marker puzzle input")
