import random
from collections import Counter, deque

import numpy as np


PALETTE = list(range(10))
MAX_GENERATION_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _count_bends(coords: np.ndarray) -> int:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    bends = 0
    endpoints = 0
    for rr, cc in np.argwhere(mask):
        neighbors = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                neighbors.append((dr, dc))
        degree = len(neighbors)
        assert degree in (1, 2), "bent shapes must be simple 1-cell-wide paths"
        if degree == 1:
            endpoints += 1
            continue
        if neighbors[0][0] != neighbors[1][0] and neighbors[0][1] != neighbors[1][1]:
            bends += 1
    assert endpoints == 2, "bent shapes must have exactly two endpoints"
    return int(bends)


def _plate_color_and_rows(grid: np.ndarray, bg: int) -> tuple[int, list[int]]:
    candidates = []
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        comps = _components(grid == color)
        if not comps:
            continue
        rows = sorted({int(r) for r, _c in np.argwhere(grid == color)})
        if len(rows) < 2:
            continue
        if any(b - a <= 1 for a, b in zip(rows, rows[1:])):
            continue
        if all(_bbox_coords(comp)[0] == _bbox_coords(comp)[2] for comp in comps):
            candidates.append((color, rows))

    assert candidates, "expected one horizontal-plate color"
    max_row_count = max(len(rows) for _color, rows in candidates)
    winners = [(color, rows) for color, rows in candidates if len(rows) == max_row_count]
    assert len(winners) == 1, "plate color must be unique"
    return winners[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Drop each bent path straight down, spending one unit of board-piercing power per solid plate it crosses.

    Input structure:
    - The most common color is the background.
    - Several monochrome bent path objects hang in the upper region, with pairwise non-overlapping bbox column spans.
    - Several non-adjacent one-pixel horizontal plates of one shared color lie below; some plate rows already contain background gaps.

    Rule:
    - For each bent shape, its piercing budget equals its number of right-angle bends.
    - Scan the plate rows from top to bottom along that shape's bbox columns: a row that is already all background there is a free pass, otherwise the shape erases that whole column interval on the plate and spends one budget point.
    - After the budget reaches zero, the shape keeps passing only through fully open rows and finally stops immediately above the next row that is not fully open; all untouched plate cells stay unchanged.

    Key insight:
    - The output is determined by replaying a per-shape fall through the existing plate gaps: bend count controls how many solid encounters get punched, while already-open gaps do not consume that budget.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    plate_color, plate_rows = _plate_color_and_rows(g, bg)

    shape_comps = _components((g != bg) & (g != plate_color))
    assert shape_comps, "expected bent shapes above the plates"

    shapes = []
    for comp in shape_comps:
        colors = np.unique(g[comp[:, 0], comp[:, 1]])
        assert colors.size == 1, "each falling object must be monochrome"
        color = int(colors[0])
        r0, c0, r1, c1 = _bbox_coords(comp)
        bends = _count_bends(comp)
        assert bends in (1, 2, 3), "shape bend counts must be between 1 and 3"
        assert r1 < plate_rows[0], "shapes must start above all plates"
        shapes.append(
            {
                "color": color,
                "coords": comp,
                "bbox": (r0, c0, r1, c1),
                "bends": bends,
            }
        )

    shapes.sort(key=lambda item: item["bbox"][1])
    for left, right in zip(shapes, shapes[1:]):
        assert left["bbox"][3] < right["bbox"][1], "shape column spans must not overlap"

    out = g.copy()
    for shape in shapes:
        coords = shape["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for shape in shapes:
        r0, c0, r1, c1 = shape["bbox"]
        budget = int(shape["bends"])
        stop_row = None
        for plate_row in plate_rows:
            if np.all(out[plate_row, c0 : c1 + 1] == bg):
                continue
            if budget > 0:
                out[plate_row, c0 : c1 + 1] = bg
                budget -= 1
            else:
                stop_row = int(plate_row)
                break

        assert stop_row is not None, "every shape must have a stopping plate"
        dr = stop_row - 1 - r1
        shifted = shape["coords"] + np.array([dr, 0], dtype=int)
        assert shifted[:, 0].min() >= 0 and shifted[:, 0].max() < g.shape[0]
        assert shifted[:, 1].min() >= 0 and shifted[:, 1].max() < g.shape[1]
        assert np.all(out[shifted[:, 0], shifted[:, 1]] == bg), "fallen shape overlaps existing cells"
        out[shifted[:, 0], shifted[:, 1]] = shape["color"]

    return out


def _transform_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(coords, dtype=int)
    if random.random() < 0.5:
        arr[:, 1] *= -1
    for _ in range(random.randint(0, 3)):
        arr = np.column_stack((arr[:, 1], -arr[:, 0]))
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    return [tuple(map(int, cell)) for cell in arr]


def _make_shape_patch(bends: int, color: int, bg: int) -> np.ndarray:
    for _ in range(100):
        lengths = [random.randint(2, 4) for _ in range(bends + 1)]
        if bends == 1:
            dirs = [(1, 0), (0, 1)]
        elif bends == 2:
            dirs = [(1, 0), (0, 1), (1, 0)]
        else:
            dirs = [(0, 1), (1, 0), (0, -1), (1, 0)]

        coords = [(0, 0)]
        r = c = 0
        for (dr, dc), length in zip(dirs, lengths):
            for _ in range(length - 1):
                r += dr
                c += dc
                coords.append((r, c))

        coords = _transform_coords(coords)
        cells = np.array(coords, dtype=int)
        h = int(cells[:, 0].max()) + 1
        w = int(cells[:, 1].max()) + 1
        patch = np.full((h, w), bg, dtype=int)
        patch[cells[:, 0], cells[:, 1]] = color
        try:
            if _count_bends(np.argwhere(patch != bg)) == bends:
                return patch
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid bent shape")


def _sample_bend_list(n_shapes: int) -> list[int]:
    bends = random.sample([1, 2, 3], k=min(3, n_shapes))
    while len(bends) < n_shapes:
        bends.append(random.choice([1, 2, 3]))
    random.shuffle(bends)
    return bends


def _sample_plate_rows(height: int, n_plates: int, min_first_row: int) -> list[int] | None:
    first_max = height - 2 - 2 * (n_plates - 1)
    if min_first_row > first_max:
        return None
    rows = [random.randint(min_first_row, first_max)]
    for i in range(1, n_plates):
        remaining = n_plates - i - 1
        lo = rows[-1] + 2
        hi = height - 2 - 2 * remaining
        if lo > hi:
            return None
        rows.append(random.randint(lo, hi))
    return rows


def _lane_pattern(bends: int, n_plates: int, force_gap: bool | None) -> tuple[list[bool], int]:
    pattern = [False] * n_plates
    if force_gap is False:
        stop_idx = bends
        for i in range(bends):
            pattern[i] = True
        pattern[stop_idx] = True
    else:
        stop_lo = bends + 1 if force_gap else bends
        stop_idx = random.randint(stop_lo, n_plates - 1)
        solid_before = random.sample(range(stop_idx), bends)
        for i in solid_before:
            pattern[i] = True
        pattern[stop_idx] = True

    for i in range(stop_idx + 1, n_plates):
        pattern[i] = random.random() < 0.6

    if force_gap is True:
        assert any(not pattern[i] for i in range(stop_idx))
    if force_gap is False:
        assert all(pattern[i] for i in range(stop_idx))
    return pattern, stop_idx


def _sample_lane_starts(widths: list[int], width: int) -> list[int] | None:
    base_spaces = [random.randint(1, 2)]
    for _ in range(len(widths) - 1):
        base_spaces.append(random.randint(2, 4))
    base_spaces.append(random.randint(1, 2))
    min_width = sum(widths) + sum(base_spaces)
    if min_width > width:
        return None

    extra = width - min_width
    for _ in range(extra):
        base_spaces[random.randrange(len(base_spaces))] += 1

    starts = []
    cursor = base_spaces[0]
    for i, shape_width in enumerate(widths):
        starts.append(cursor)
        cursor += shape_width + base_spaces[i + 1]
    return starts


def _render_expected(
    inp: np.ndarray,
    records: list[dict],
    plate_rows: list[int],
    bg: int,
) -> np.ndarray | None:
    out = inp.copy()
    for rec in records:
        patch = rec["patch"]
        top = rec["top"]
        left = rec["left"]
        mask = patch != bg
        cells = np.argwhere(mask)
        out[top + cells[:, 0], left + cells[:, 1]] = bg

    for rec in sorted(records, key=lambda item: item["left"]):
        patch = rec["patch"]
        mask = patch != bg
        h, w = patch.shape
        left = rec["left"]
        stop_row = plate_rows[rec["stop_idx"]]
        for solid, plate_row in zip(rec["pattern"][: rec["stop_idx"]], plate_rows[: rec["stop_idx"]]):
            if solid:
                out[plate_row, left : left + w] = bg
        dest_r = stop_row - h
        if dest_r < 0 or dest_r + h > out.shape[0]:
            return None
        region = out[dest_r : dest_r + h, left : left + w]
        if np.any(region[mask] != bg):
            return None
        region[mask] = rec["color"]
    return out


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    n_shapes = random.randint(2, 4)
    bends_list = _sample_bend_list(n_shapes)
    bg, plate_color = random.sample(PALETTE, 2)
    shape_colors = random.sample([c for c in PALETTE if c not in (bg, plate_color)], n_shapes)

    patches = [_make_shape_patch(bends, color, bg) for bends, color in zip(bends_list, shape_colors)]
    widths = [patch.shape[1] for patch in patches]
    max_shape_h = max(int(patch.shape[0]) for patch in patches)

    n_plates = random.randint(max(max(bends_list) + 2, 5), 7)
    min_height = max_shape_h + 2 * n_plates + 2
    if min_height > 30:
        return None
    height = random.randint(min_height, min(30, min_height + 5))

    rows = _sample_plate_rows(height, n_plates, max_shape_h + 2)
    if rows is None:
        return None

    min_width = sum(widths) + 2 * n_shapes + 2
    if min_width > 30:
        return None
    width_lo = max(min_width, 18)
    width_hi = min(30, max(width_lo, min_width + 6))
    width = random.randint(width_lo, width_hi)
    starts = _sample_lane_starts(widths, width)
    if starts is None:
        return None

    if n_shapes == 1:
        force_modes = [True]
    else:
        force_modes = [None] * n_shapes
        gap_idx, solid_idx = random.sample(range(n_shapes), 2)
        force_modes[gap_idx] = True
        force_modes[solid_idx] = False

    patterns = []
    used_gap_branch = False
    for bends, mode in zip(bends_list, force_modes):
        pattern, stop_idx = _lane_pattern(bends, n_plates, mode)
        patterns.append((pattern, stop_idx))
        used_gap_branch |= any(not pattern[i] for i in range(stop_idx))
    if not used_gap_branch:
        return None

    grid = np.full((height, width), bg, dtype=int)
    grid[rows, :] = plate_color

    records = []
    for patch, bends, color, left, (pattern, stop_idx) in zip(
        patches, bends_list, shape_colors, starts, patterns
    ):
        h, w = patch.shape
        for plate_row, solid in zip(rows, pattern):
            if not solid:
                grid[plate_row, left : left + w] = bg
        max_top = rows[0] - h - 1
        if max_top < 0:
            return None
        top = random.randint(0, max_top)
        mask = patch != bg
        region = grid[top : top + h, left : left + w]
        if np.any(region[mask] != bg):
            return None
        region[mask] = patch[mask]
        records.append(
            {
                "patch": patch,
                "top": top,
                "left": left,
                "color": color,
                "bends": bends,
                "pattern": pattern,
                "stop_idx": stop_idx,
            }
        )

    expected = _render_expected(grid, records, rows, bg)
    if expected is None:
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_scene()
        if built is None:
            continue
        inp, expected = built
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected) and not np.array_equal(inp, expected):
            return inp
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
