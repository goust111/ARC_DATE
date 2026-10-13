import numpy as np


BG = 0
WORK = 5
MARK = 7
PALETTE = (1, 2, 3, 4, 6)
MAX_ATTEMPTS = 600


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _marker_positions(region_sizes: list[int]) -> list[int]:
    pos = region_sizes[0]
    markers: list[int] = []
    for size in region_sizes[1:]:
        markers.append(pos)
        pos += 1 + size
    return markers


def _bands_from_markers(size: int, markers: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + list(markers) + [size]
    bands: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:]):
        bands.append((left + 1, right))
    return bands


def _sorted_palette_points(patch: np.ndarray) -> list[tuple[int, int, int]]:
    points = []
    for r, c in np.argwhere((patch != BG) & (patch != WORK) & (patch != MARK)):
        points.append((int(r), int(c), int(patch[r, c])))
    points.sort(key=lambda item: (item[0], item[1]))
    return points


def _filled_patch(height: int, width: int, colors: list[int]) -> np.ndarray:
    assert colors
    layers = min((height + 1) // 2, (width + 1) // 2)
    out = np.full((height, width), colors[-1], dtype=int)
    for depth in range(layers):
        color = int(colors[min(depth, len(colors) - 1)])
        out[depth : height - depth, depth : width - depth] = color
    return out


def _structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    work_bbox = _bbox(g != BG)
    assert work_bbox is not None, "expected one non-black work area"
    r0, c0, r1, c1 = work_bbox
    work = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(work != BG), "work area must be a solid non-black rectangle with interior markers"

    marker_local = np.argwhere(work == MARK)
    assert 2 <= marker_local.shape[0] <= 3, "expected 2-3 orange markers"

    row_marks = sorted(set(int(r) for r, _c in marker_local))
    col_marks = sorted(set(int(c) for _r, c in marker_local))
    assert len(row_marks) == len(marker_local)
    assert len(col_marks) == len(marker_local)

    region_rows = _bands_from_markers(work.shape[0], row_marks)
    region_cols = _bands_from_markers(work.shape[1], col_marks)
    assert all((r1_ - r0_) >= 4 for r0_, r1_ in region_rows)
    assert all((c1_ - c0_) >= 4 for c0_, c1_ in region_cols)

    return {
        "grid": g,
        "bbox": work_bbox,
        "work": work,
        "row_marks": row_marks,
        "col_marks": col_marks,
        "region_rows": region_rows,
        "region_cols": region_cols,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand orange row/column markers into a grid and fill each resulting sub-rectangle with concentric bands from its local diagonal palette.

    Input structure:
    - Black background surrounds one solid gray rectangular work area.
    - Inside the work area, 2-3 orange singleton markers occupy distinct rows and distinct columns.
    - The remaining non-gray singleton pixels are palette markers placed inside the sub-rectangles on short down-right diagonals.

    Rule:
    - Within the gray work area, expand every orange marker to its full row and full column.
    - These orange rows and columns partition the work area into disjoint sub-rectangles, excluding the orange lines themselves.
    - For each sub-rectangle, collect its palette pixels, sort them by row then column, and use their colors as an outer-to-inner concentric fill sequence.
    - If there are fewer palette pixels than concentric layers, extend the final color through the remaining center; if there are more, ignore the extra inner palette pixels.
    - Keep the exterior black background unchanged and keep all orange partition lines orange.

    Key insight:
    - The orange markers are both visible drawing instructions and spatial separators, so each sub-rectangle must be solved independently using only the diagonal palette points that lie inside it.
    """
    info = _structure(grid)
    g = info["grid"]
    r0, c0, r1, c1 = info["bbox"]
    out = g.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = WORK

    for lr in info["row_marks"]:
        out[r0 + lr, c0 : c1 + 1] = MARK
    for lc in info["col_marks"]:
        out[r0 : r1 + 1, c0 + lc] = MARK

    work = info["work"]
    for rr0, rr1 in info["region_rows"]:
        for cc0, cc1 in info["region_cols"]:
            patch = work[rr0:rr1, cc0:cc1]
            palette_points = _sorted_palette_points(patch)
            assert palette_points, "each sub-rectangle must have at least one palette marker"
            colors = [color for _r, _c, color in palette_points]
            filled = _filled_patch(rr1 - rr0, cc1 - cc0, colors)
            out[r0 + rr0 : r0 + rr1, c0 + cc0 : c0 + cc1] = filled

    return out


def _sample_region_sizes(rng: np.random.Generator, n_regions: int) -> list[int]:
    small = int(rng.choice([4, 5]))
    big = 6
    if n_regions == 3:
        other = int(rng.choice([4, 5, 6]))
        sizes = [small, big, other]
    else:
        rest = [int(rng.choice([4, 5])) for _ in range(n_regions - 2)]
        sizes = [small, big, *rest]
    rng.shuffle(sizes)
    return sizes


def _choose_distinct_colors(rng: np.random.Generator, count: int) -> list[int]:
    return [int(v) for v in rng.choice(np.array(PALETTE, dtype=int), size=count, replace=False)]


def _pick_region_points(
    rng: np.random.Generator,
    height: int,
    width: int,
    specs: list[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    points: list[tuple[int, int, int]] = []
    occupied: set[tuple[int, int]] = set()
    used_diagonals: set[int] = set()
    for length, color in specs:
        placed = False
        for _ in range(80):
            sr = int(rng.integers(0, height - length + 1))
            sc = int(rng.integers(0, width - length + 1))
            diag_id = sc - sr
            if diag_id in used_diagonals:
                continue
            cells = [(sr + step, sc + step) for step in range(length)]
            if any(cell in occupied for cell in cells):
                continue
            used_diagonals.add(diag_id)
            for rr, cc in cells:
                occupied.add((rr, cc))
                points.append((rr, cc, color))
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place diagonal palette points")
    return points


def _region_plan(
    rng: np.random.Generator,
    region_h: int,
    region_w: int,
    role: str,
) -> list[tuple[int, int]]:
    if role == "overflow":
        return [(3, _choose_distinct_colors(rng, 1)[0])]
    if role == "shortage":
        return [(1, _choose_distinct_colors(rng, 1)[0])]
    if role == "multi":
        colors = _choose_distinct_colors(rng, 2)
        if min(region_h, region_w) >= 5 and rng.random() < 0.5:
            return [(2, colors[0]), (1, colors[1])]
        return [(1, colors[0]), (1, colors[1])]

    mode = int(rng.integers(0, 4))
    if mode == 0:
        return [(int(rng.choice([1, 2, 3])), _choose_distinct_colors(rng, 1)[0])]
    if mode == 1:
        colors = _choose_distinct_colors(rng, 2)
        return [(1, colors[0]), (1, colors[1])]
    if mode == 2 and min(region_h, region_w) >= 5:
        colors = _choose_distinct_colors(rng, 2)
        return [(2, colors[0]), (1, colors[1])]
    return [(int(rng.choice([1, 2])), _choose_distinct_colors(rng, 1)[0])]


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        info = _structure(inp)
    except Exception:
        return False

    r0, c0, r1, c1 = info["bbox"]
    if not np.all(inp[:r0, :] == BG):
        return False
    if not np.all(inp[r1 + 1 :, :] == BG):
        return False
    if not np.all(inp[r0 : r1 + 1, :c0] == BG):
        return False
    if not np.all(inp[r0 : r1 + 1, c1 + 1 :] == BG):
        return False
    if not np.all(out[:r0, :] == BG):
        return False
    if not np.all(out[r1 + 1 :, :] == BG):
        return False
    if not np.all(out[r0 : r1 + 1, :c0] == BG):
        return False
    if not np.all(out[r0 : r1 + 1, c1 + 1 :] == BG):
        return False

    for lr in info["row_marks"]:
        if not np.all(out[r0 + lr, c0 : c1 + 1] == MARK):
            return False
    for lc in info["col_marks"]:
        if not np.all(out[r0 : r1 + 1, c0 + lc] == MARK):
            return False

    has_shortage = False
    has_overflow = False
    has_multicolor = False
    region_sigs: set[bytes] = set()
    work = info["work"]
    for rr0, rr1 in info["region_rows"]:
        for cc0, cc1 in info["region_cols"]:
            patch = work[rr0:rr1, cc0:cc1]
            points = _sorted_palette_points(patch)
            if not points:
                return False
            colors = [color for _r, _c, color in points]
            layers = min((rr1 - rr0 + 1) // 2, (cc1 - cc0 + 1) // 2)
            if len(points) < layers:
                has_shortage = True
            if len(points) > layers:
                has_overflow = True
            if len(set(colors)) > 1:
                has_multicolor = True
            expected = _filled_patch(rr1 - rr0, cc1 - cc0, colors)
            actual = out[r0 + rr0 : r0 + rr1, c0 + cc0 : c0 + cc1]
            if not np.array_equal(actual, expected):
                return False
            region_sigs.add(actual.tobytes() + str(actual.shape).encode())

    return has_shortage and has_overflow and has_multicolor and len(region_sigs) >= 3


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_markers = int(rng.choice([2, 3]))
    n_regions = n_markers + 1

    row_sizes = _sample_region_sizes(rng, n_regions)
    col_sizes = _sample_region_sizes(rng, n_regions)

    row_marks = _marker_positions(row_sizes)
    col_marks = _marker_positions(col_sizes)
    work_h = sum(row_sizes) + n_markers
    work_w = sum(col_sizes) + n_markers

    top = int(rng.integers(2, 4))
    left = int(rng.integers(2, 4))
    bottom = int(rng.integers(2, 4))
    right = int(rng.integers(2, 4))
    grid_h = top + work_h + bottom
    grid_w = left + work_w + right
    if max(grid_h, grid_w) > 30:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    wr0, wc0 = top, left
    grid[wr0 : wr0 + work_h, wc0 : wc0 + work_w] = WORK

    marker_cols_perm = [col_marks[i] for i in rng.permutation(n_markers)]
    for rr, cc in zip(row_marks, marker_cols_perm):
        grid[wr0 + rr, wc0 + cc] = MARK

    row_bands = _bands_from_markers(work_h, row_marks)
    col_bands = _bands_from_markers(work_w, col_marks)

    regions = []
    for ri, (rr0, rr1) in enumerate(row_bands):
        for ci, (cc0, cc1) in enumerate(col_bands):
            height = rr1 - rr0
            width = cc1 - cc0
            layers = min((height + 1) // 2, (width + 1) // 2)
            regions.append(
                {
                    "index": (ri, ci),
                    "r0": rr0,
                    "r1": rr1,
                    "c0": cc0,
                    "c1": cc1,
                    "height": height,
                    "width": width,
                    "layers": layers,
                }
            )

    short_candidates = [region for region in regions if region["layers"] >= 3]
    overflow_candidates = [region for region in regions if region["layers"] == 2]
    multi_candidates = [region for region in regions if region["layers"] >= 2]
    if not short_candidates or not overflow_candidates or not multi_candidates:
        return None

    shortage_region = short_candidates[int(rng.integers(0, len(short_candidates)))]["index"]
    overflow_pool = [region for region in overflow_candidates if region["index"] != shortage_region]
    if not overflow_pool:
        return None
    overflow_region = overflow_pool[int(rng.integers(0, len(overflow_pool)))]["index"]
    multi_pool = [region for region in multi_candidates if region["index"] not in {shortage_region, overflow_region}]
    if not multi_pool:
        multi_region = shortage_region
    else:
        multi_region = multi_pool[int(rng.integers(0, len(multi_pool)))]["index"]

    for region in regions:
        idx = region["index"]
        if idx == overflow_region:
            role = "overflow"
        elif idx == shortage_region:
            role = "shortage"
        elif idx == multi_region:
            role = "multi"
        else:
            role = "default"

        specs = _region_plan(rng, region["height"], region["width"], role)
        try:
            points = _pick_region_points(rng, region["height"], region["width"], specs)
        except RuntimeError:
            return None
        for rr, cc, color in points:
            grid[wr0 + region["r0"] + rr, wc0 + region["c0"] + cc] = color

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, out):
        return None
    if not _semantic_checks(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
