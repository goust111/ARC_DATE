import numpy as np


BG = 0
GREY = 5
PORTAL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
HEIGHT_RANGE = (16, 24)
WIDTH_RANGE = (16, 24)
SEGMENT_COUNT_RANGE = (3, 5)
BAND_WIDTH_RANGE = (12, 19)
SEGMENT_LEN_RANGE = (3, 6)
THICKNESS_RANGE = (4, 9)
EXTRA_PAIR_PROB = 0.6
MIN_COLUMN_GAP = 3
GENERATION_RETRIES = 600
_RNG = np.random.default_rng()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the vertical grey strips captured by each same-color portal pair onto the partner column.

    Input structure:
    - The background is black.
    - One non-black color forms a single large connected grey pattern.
    - Every other non-black color appears as exactly two isolated portal pixels of the same color.
    - The two pixels of each portal color lie on the same row, and each portal touches grey vertically above, below, or on both sides.

    Rule:
    - For each portal pair on row r, inspect each portal's own column separately.
    - Collect the continuous grey cells directly above and directly below that portal in the same column.
    - Replay those row offsets in the partner portal's column and paint the copied cells with the portal color.
    - Keep the original grey pattern and all portal pixels unchanged.

    Key insight:
    - The portal color gives the pairing, while the copied shape is just the grey run's vertical offset pattern transferred across to the partner column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = BG if np.any(g == BG) else int(np.bincount(g.ravel()).argmax())
    grey = _infer_grey_color(g, bg)
    out = g.copy()

    portal_colors = sorted(int(color) for color in np.unique(g) if int(color) not in {bg, grey})
    portal_cells = {tuple(map(int, coord)) for coord in np.argwhere((g != bg) & (g != grey))}

    for color in portal_colors:
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2)
        coords = coords[np.argsort(coords[:, 1])]
        r0, c0 = map(int, coords[0])
        r1, c1 = map(int, coords[1])
        assert r0 == r1

        offsets0 = _vertical_offsets(g, grey, r0, c0)
        offsets1 = _vertical_offsets(g, grey, r1, c1)
        assert offsets0 and offsets1

        for dr in offsets0:
            rr = r1 + dr
            if (rr, c1) not in portal_cells:
                out[rr, c1] = color
        for dr in offsets1:
            rr = r0 + dr
            if (rr, c0) not in portal_cells:
                out[rr, c0] = color

    return out


def _infer_grey_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    assert colors
    counts = {color: int(np.sum(grid == color)) for color in colors}
    grey = max(colors, key=lambda color: counts[color])
    assert counts[grey] > 2
    return grey


def _vertical_offsets(grid: np.ndarray, grey: int, row: int, col: int) -> list[int]:
    h = int(grid.shape[0])
    offsets: list[int] = []

    rr = row - 1
    while rr >= 0 and int(grid[rr, col]) == grey:
        offsets.append(rr - row)
        rr -= 1

    rr = row + 1
    while rr < h and int(grid[rr, col]) == grey:
        offsets.append(rr - row)
        rr += 1

    return offsets


def _bounded_composition(total: int, parts: int, lo: int, hi: int) -> list[int] | None:
    for _ in range(200):
        remaining = total
        result: list[int] = []
        for idx in range(parts):
            left = parts - idx - 1
            low = max(lo, remaining - hi * left)
            high = min(hi, remaining - lo * left)
            if low > high:
                break
            value = int(_RNG.integers(low, high + 1))
            result.append(value)
            remaining -= value
        else:
            if remaining == 0:
                return result
    return None


def _build_band(height: int, width: int) -> tuple[np.ndarray, np.ndarray] | None:
    band_w_hi = min(BAND_WIDTH_RANGE[1], width - 2)
    if band_w_hi < BAND_WIDTH_RANGE[0]:
        return None

    band_w = int(_RNG.integers(BAND_WIDTH_RANGE[0], band_w_hi + 1))
    left = int(_RNG.integers(1, width - band_w))
    seg_count = int(_RNG.integers(SEGMENT_COUNT_RANGE[0], SEGMENT_COUNT_RANGE[1]))
    lengths = _bounded_composition(band_w, seg_count, SEGMENT_LEN_RANGE[0], SEGMENT_LEN_RANGE[1])
    if lengths is None:
        return None

    top = np.full(width, -1, dtype=int)
    bottom = np.full(width, -1, dtype=int)

    start_top_hi = height - THICKNESS_RANGE[0] - 2
    if start_top_hi < 2:
        return None
    current_top = int(_RNG.integers(2, start_top_hi + 1))
    max_thickness = min(THICKNESS_RANGE[1], height - current_top - 2)
    if max_thickness < THICKNESS_RANGE[0]:
        return None
    thickness = int(_RNG.integers(THICKNESS_RANGE[0], max_thickness + 1))
    current_bottom = current_top + thickness - 1

    col = left
    changed = False
    for seg_len in lengths:
        top[col : col + seg_len] = current_top
        bottom[col : col + seg_len] = current_bottom
        col += seg_len

        if col >= left + band_w:
            continue

        next_top = current_top
        next_bottom = current_bottom
        for _ in range(80):
            delta_top = int(_RNG.integers(-2, 3))
            delta_bottom = int(_RNG.integers(-2, 3))
            trial_top = current_top + delta_top
            trial_bottom = current_bottom + delta_bottom
            trial_thickness = trial_bottom - trial_top + 1
            if not (1 <= trial_top <= height - 5):
                continue
            if not (trial_top + THICKNESS_RANGE[0] - 1 <= trial_bottom <= height - 2):
                continue
            if not (THICKNESS_RANGE[0] <= trial_thickness <= THICKNESS_RANGE[1]):
                continue
            if max(current_top, trial_top) > min(current_bottom, trial_bottom):
                continue
            next_top = trial_top
            next_bottom = trial_bottom
            if next_top != current_top or next_bottom != current_bottom:
                changed = True
            break

        current_top = next_top
        current_bottom = next_bottom

    if not changed:
        return None
    return top, bottom


def _groups_by_row(top: np.ndarray, bottom: np.ndarray, height: int) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    top_groups: dict[int, list[int]] = {}
    bottom_groups: dict[int, list[int]] = {}

    for col in range(int(top.shape[0])):
        if int(top[col]) >= 1:
            row = int(top[col]) - 1
            top_groups.setdefault(row, []).append(col)
        if int(bottom[col]) >= 0 and int(bottom[col]) < height - 1:
            row = int(bottom[col]) + 1
            bottom_groups.setdefault(row, []).append(col)

    return top_groups, bottom_groups


def _pick_two_columns(columns: list[int], used_columns: set[int]) -> tuple[int, int] | None:
    candidates = [int(col) for col in columns if int(col) not in used_columns]
    if len(candidates) < 2:
        return None

    order = candidates.copy()
    _RNG.shuffle(order)
    for idx, c0 in enumerate(order):
        far = [c1 for c1 in order[idx + 1 :] if abs(int(c1) - int(c0)) >= MIN_COLUMN_GAP]
        if far:
            c1 = int(_RNG.choice(far))
            return tuple(sorted((int(c0), c1)))
    return None


def _pick_row_group(groups: dict[int, list[int]], used_rows: set[int], used_columns: set[int]) -> tuple[int, tuple[int, int]] | None:
    rows = [int(row) for row in groups if int(row) not in used_rows]
    _RNG.shuffle(rows)
    for row in rows:
        pair = _pick_two_columns(groups[row], used_columns)
        if pair is not None:
            return row, pair
    return None


def _hole_groups(mask: np.ndarray) -> dict[int, list[int]]:
    height, width = map(int, mask.shape)
    groups: dict[int, list[int]] = {}
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            if not bool(mask[row, col]):
                continue
            if not bool(mask[row - 1, col] and mask[row + 1, col]):
                continue
            if not bool(mask[row, col - 1] and mask[row, col + 1]):
                continue
            groups.setdefault(row, []).append(col)
    return groups


def _component_count(mask: np.ndarray) -> int:
    seen = np.zeros_like(mask, dtype=bool)
    height, width = map(int, mask.shape)
    count = 0

    for row in range(height):
        for col in range(width):
            if not bool(mask[row, col]) or bool(seen[row, col]):
                continue
            count += 1
            stack = [(row, col)]
            seen[row, col] = True
            while stack:
                rr, cc = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _teleport_cells(grid: np.ndarray, color: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    grey = _infer_grey_color(grid, BG)
    coords = np.argwhere(grid == color)
    coords = coords[np.argsort(coords[:, 1])]
    row = int(coords[0, 0])
    c0 = int(coords[0, 1])
    c1 = int(coords[1, 1])
    offsets0 = _vertical_offsets(grid, grey, row, c0)
    offsets1 = _vertical_offsets(grid, grey, row, c1)
    left_to_right = {(row + dr, c1) for dr in offsets0}
    right_to_left = {(row + dr, c0) for dr in offsets1}
    return left_to_right, right_to_left


def _is_valid_input(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if np.count_nonzero(g == GREY) < 30:
        return False
    if _component_count(g == GREY) != 1:
        return False

    colors = [int(color) for color in np.unique(g) if int(color) not in {BG, GREY}]
    if len(colors) < 3 or len(colors) > 4:
        return False

    used_columns: set[int] = set()
    portal_cells: set[tuple[int, int]] = set()
    pair_info: list[tuple[int, np.ndarray]] = []
    seen_both = False
    seen_top_only = False
    seen_bottom_only = False

    for color in colors:
        coords = np.argwhere(g == color)
        if coords.shape != (2, 2):
            return False
        coords = coords[np.argsort(coords[:, 1])]
        if int(coords[0, 0]) != int(coords[1, 0]):
            return False
        pair_info.append((color, coords.copy()))
        for row, col in coords.tolist():
            if int(col) in used_columns:
                return False
            used_columns.add(int(col))
            portal_cells.add((int(row), int(col)))

            above = row - 1 >= 0 and int(g[row - 1, col]) == GREY
            below = row + 1 < g.shape[0] and int(g[row + 1, col]) == GREY
            if not (above or below):
                return False
            seen_both |= above and below
            seen_top_only |= above and not below
            seen_bottom_only |= below and not above

    painted_cells: set[tuple[int, int]] = set()
    for color, _coords in pair_info:
        lr, rl = _teleport_cells(g, color)
        if lr & portal_cells or rl & portal_cells:
            return False
        if painted_cells & lr or painted_cells & rl:
            return False
        painted_cells |= lr | rl

    if not (seen_both and seen_top_only and seen_bottom_only):
        return False

    try:
        out = main(g)
    except Exception:
        return False
    if np.array_equal(out, g):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(HEIGHT_RANGE[0], HEIGHT_RANGE[1] + 1))
    width = int(_RNG.integers(WIDTH_RANGE[0], WIDTH_RANGE[1] + 1))
    band = _build_band(height, width)
    if band is None:
        return None
    top, bottom = band

    grid = np.full((height, width), BG, dtype=int)
    for col in range(width):
        if int(top[col]) >= 0:
            grid[int(top[col]) : int(bottom[col]) + 1, col] = GREY

    top_groups, bottom_groups = _groups_by_row(top, bottom, height)
    if not any(len(cols) >= 2 for cols in top_groups.values()):
        return None
    if not any(len(cols) >= 2 for cols in bottom_groups.values()):
        return None

    used_rows: set[int] = set()
    used_columns: set[int] = set()
    pairs: list[tuple[str, int, int, int]] = []

    top_pick = _pick_row_group(top_groups, used_rows, used_columns)
    if top_pick is None:
        return None
    top_row, (top_c0, top_c1) = top_pick
    pairs.append(("top", top_row, top_c0, top_c1))
    used_rows.add(top_row)
    used_columns.update((top_c0, top_c1))

    bottom_pick = _pick_row_group(bottom_groups, used_rows, used_columns)
    if bottom_pick is None:
        return None
    bottom_row, (bottom_c0, bottom_c1) = bottom_pick
    pairs.append(("bottom", bottom_row, bottom_c0, bottom_c1))
    used_rows.add(bottom_row)
    used_columns.update((bottom_c0, bottom_c1))

    hole_pick = _pick_row_group(_hole_groups(grid == GREY), used_rows, used_columns)
    if hole_pick is None:
        return None
    hole_row, (hole_c0, hole_c1) = hole_pick
    grid[hole_row, hole_c0] = BG
    grid[hole_row, hole_c1] = BG
    if _component_count(grid == GREY) != 1:
        return None
    pairs.append(("hole", hole_row, hole_c0, hole_c1))
    used_rows.add(hole_row)
    used_columns.update((hole_c0, hole_c1))

    if _RNG.random() < EXTRA_PAIR_PROB:
        extras: list[tuple[str, int, tuple[int, int]]] = []
        for name, groups in (("top", top_groups), ("bottom", bottom_groups)):
            pick = _pick_row_group(groups, used_rows, used_columns)
            if pick is not None:
                row, cols = pick
                extras.append((name, row, cols))
        if extras:
            name, row, cols = extras[int(_RNG.integers(0, len(extras)))]
            pairs.append((name, row, int(cols[0]), int(cols[1])))
            used_rows.add(row)
            used_columns.update(cols)

    colors = list(_RNG.choice(PORTAL_COLORS, size=len(pairs), replace=False))
    for color, (_, row, c0, c1) in zip(colors, pairs, strict=False):
        if grid[row, c0] != BG or grid[row, c1] != BG:
            return None
        grid[row, c0] = int(color)
        grid[row, c1] = int(color)

    if not _is_valid_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid portal-transfer ARC input")
