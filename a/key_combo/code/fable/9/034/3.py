import random

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.asarray(grid, dtype=int).ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    candidates: list[tuple[int, int, int, int, int]] = []
    for color in colors_present(g, exclude={bg}):
        mask = g == color
        box = _bbox_from_mask(mask)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        if r0 == 0 or c0 == 0 or r1 == g.shape[0] - 1 or c1 == g.shape[1] - 1:
            continue
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 4 or w < 4:
            continue
        perimeter_count = 2 * h + 2 * w - 4
        if int(mask.sum()) != perimeter_count:
            continue
        if np.any(g[r0 + 1 : r1, c0 + 1 : c1] == color):
            continue
        if not np.all(g[r0, c0 : c1 + 1] == color):
            continue
        if not np.all(g[r1, c0 : c1 + 1] == color):
            continue
        if not np.all(g[r0 : r1 + 1, c0] == color):
            continue
        if not np.all(g[r0 : r1 + 1, c1] == color):
            continue
        candidates.append((color, r0, c0, r1, c1))
    assert len(candidates) == 1, "expected a unique one-cell rectangular frame"
    return candidates[0]


def _find_stripe_rows(grid: np.ndarray, bg: int, frame_color: int, box: tuple[int, int, int, int]) -> tuple[int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = box
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    inner_width = inner.shape[1]
    candidates: list[tuple[int, np.ndarray]] = []
    for color in colors_present(g, exclude={bg, frame_color}):
        row_mask = np.all(inner == color, axis=1)
        if not row_mask.any():
            continue
        total = int((g == color).sum())
        expected = int(row_mask.sum()) * inner_width
        if total != expected:
            continue
        if int((inner == color).sum()) != expected:
            continue
        candidates.append((color, row_mask))
    assert len(candidates) == 1, "expected a unique full-row stripe color inside the frame"
    stripe_color, row_mask = candidates[0]
    assert row_mask.any() and (~row_mask).any(), "sampled stripe rows must show both on and off rows"
    return stripe_color, row_mask


def _infer_periodic_rows(total_rows: int, observed_start: int, observed_mask: np.ndarray) -> np.ndarray:
    observed = np.asarray(observed_mask, dtype=bool)
    assert observed.ndim == 1 and observed.size > 0
    for period in range(2, total_rows + 1):
        residues: dict[int, bool] = {}
        ok = True
        for row, value in enumerate(observed, start=observed_start):
            residue = row % period
            prev = residues.get(residue)
            if prev is None:
                residues[residue] = bool(value)
            elif prev != bool(value):
                ok = False
                break
        if not ok or len(residues) != period:
            continue
        full = np.array([residues[r % period] for r in range(total_rows)], dtype=bool)
        if np.array_equal(full[observed_start : observed_start + len(observed)], observed):
            return full
    raise AssertionError("stripe rows must define a unique global periodic mask")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project the framed periodic stripe sample into full-width rows, then let obstacles erase the downwind side.

    Input structure:
    - The grid has a dominant background, one one-cell-thick rectangular frame away from the canvas border,
      and several full interior rows of a second color inside that frame.
    - Those interior stripe rows follow a fixed row-period/phase pattern, while all other interior cells stay background.
    - Outside the frame lie singleton obstacle pixels of a third color; some share stripe-row extensions and some do not.

    Rule:
    - Read the stripe-row periodic mask from the frame interior and paint every row with the same global phase
      across the entire canvas using the frame color.
    - Erase the frame itself and the original stripe color; only the projected full-row stripes remain.
    - Obstacles stay unchanged, and if an obstacle lies on a projected stripe row, every cell farther from the
      frame center along that row is reset to background, creating a one-sided shadow.

    Key insight:
    - The frame is just a local phase template for a whole-canvas row pattern, and each obstacle only blocks
      the stripe downstream on the side opposite the frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = get_bg(g)

    frame_color, r0, c0, r1, c1 = _find_frame(g, bg)
    stripe_color, observed_mask = _find_stripe_rows(g, bg, frame_color, (r0, c0, r1, c1))
    selected_rows = _infer_periodic_rows(h, r0 + 1, observed_mask)

    out = np.full_like(g, bg)
    out[selected_rows, :] = frame_color

    obstacle_mask = (g != bg) & (g != frame_color) & (g != stripe_color)
    obstacle_coords = np.argwhere(obstacle_mask)
    frame_center_c = (c0 + c1) / 2.0

    for r_raw, c_raw in obstacle_coords:
        r = int(r_raw)
        c = int(c_raw)
        if selected_rows[r]:
            assert c != frame_center_c, "shadow direction would be ambiguous at the frame center line"
            if c < frame_center_c:
                out[r, :c] = bg
            else:
                out[r, c + 1 :] = bg

    out[obstacle_mask] = g[obstacle_mask]
    return out


GRID_H_RANGE = (12, 23)
GRID_W_RANGE = (15, 27)
PERIOD_RANGE = (3, 6)
ON_COUNT_RANGE = (1, 2)
MAX_ATTEMPTS = 400
BG = 0
PALETTE = tuple(range(1, 10))


def _mask_from_params(length: int, period: int, phase: int, on_count: int) -> np.ndarray:
    rows = np.arange(length)
    return ((rows - phase) % period) < on_count


def _expected_output(
    shape: tuple[int, int],
    bg: int,
    frame_color: int,
    obstacle_color: int,
    selected_rows: np.ndarray,
    obstacles: list[tuple[int, int]],
    frame_box: tuple[int, int, int, int],
) -> np.ndarray:
    h, w = shape
    _, c0, _, c1 = frame_box
    out = np.full((h, w), bg, dtype=int)
    out[selected_rows, :] = frame_color
    center_c = (c0 + c1) / 2.0
    for r, c in obstacles:
        if selected_rows[r]:
            if c < center_c:
                out[r, :c] = bg
            else:
                out[r, c + 1 :] = bg
    for r, c in obstacles:
        out[r, c] = obstacle_color
    return out


def _unique_min_period(total_rows: int, start_row: int, observed_mask: np.ndarray, expected_period: int) -> bool:
    observed = np.asarray(observed_mask, dtype=bool)
    hit_periods: list[int] = []
    for period in range(2, total_rows + 1):
        residues: dict[int, bool] = {}
        ok = True
        for row, value in enumerate(observed, start=start_row):
            residue = row % period
            prev = residues.get(residue)
            if prev is None:
                residues[residue] = bool(value)
            elif prev != bool(value):
                ok = False
                break
        if ok and len(residues) == period:
            hit_periods.append(period)
    return bool(hit_periods) and hit_periods[0] == expected_period


def _pick_frame_box(h: int, w: int, inner_h: int, inner_w: int) -> tuple[int, int, int, int] | None:
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    if h < frame_h + 3 or w < frame_w + 7:
        return None
    side = random.choice(("left", "right"))
    if side == "left":
        max_left = min(4, w - frame_w - 5)
        if max_left < 2:
            return None
        c0 = random.randint(2, max_left)
    else:
        max_right = min(4, w - frame_w - 5)
        if max_right < 2:
            return None
        right_margin = random.randint(2, max_right)
        c0 = w - right_margin - frame_w
    if c0 < 1 or c0 + frame_w >= w:
        return None
    min_top = 1
    max_top = h - frame_h - 2
    if max_top < min_top:
        return None
    r0 = random.randint(min_top, max_top)
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1
    if r1 >= h - 1 or c1 >= w - 1:
        return None
    if c0 < 2 or c1 > w - 3:
        return None
    return r0, c0, r1, c1


def _place_obstacle(obstacles: set[tuple[int, int]], row: int, cols: list[int]) -> tuple[int, int] | None:
    random.shuffle(cols)
    for col in cols:
        cell = (row, col)
        if cell not in obstacles:
            obstacles.add(cell)
            return cell
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    frame_color, stripe_color, obstacle_color = random.sample(PALETTE, 3)

    period = random.randint(*PERIOD_RANGE)
    on_count = random.randint(*ON_COUNT_RANGE)
    off_count = period - on_count
    if off_count < 1:
        return None

    inner_h_min = max(period + 2, 6)
    inner_h_max = min(h - 5, period + 6)
    inner_w_max = min(w - 7, 9)
    if inner_h_min > inner_h_max or inner_w_max < 5:
        return None
    inner_h = random.randint(inner_h_min, inner_h_max)
    inner_w = random.randint(5, inner_w_max)
    box = _pick_frame_box(h, w, inner_h, inner_w)
    if box is None:
        return None
    r0, c0, r1, c1 = box

    valid_phases = []
    for phase in range(period):
        full_mask = _mask_from_params(h, period, phase, on_count)
        observed = full_mask[r0 + 1 : r1]
        if full_mask[r0] or full_mask[r1]:
            continue
        if observed.sum() < 2 or (~observed).sum() < 2:
            continue
        if not full_mask[:r0].any() and not full_mask[r1 + 1 :].any():
            continue
        if not _unique_min_period(h, r0 + 1, observed, period):
            continue
        valid_phases.append(phase)
    if not valid_phases:
        return None

    phase = random.choice(valid_phases)
    selected_rows = _mask_from_params(h, period, phase, on_count)
    observed = selected_rows[r0 + 1 : r1]
    stripe_rows_inside = [r for r in range(r0 + 1, r1) if selected_rows[r]]
    gap_rows_inside = [r for r in range(r0 + 1, r1) if not selected_rows[r]]
    if len(stripe_rows_inside) < 2 or not gap_rows_inside:
        return None

    left_cols = list(range(0, c0 - 1))
    right_cols = list(range(c1 + 2, w))
    if not left_cols or not right_cols:
        return None

    obstacles: set[tuple[int, int]] = set()
    shadow_rows = random.sample(stripe_rows_inside, 2)
    if _place_obstacle(obstacles, shadow_rows[0], left_cols.copy()) is None:
        return None
    if _place_obstacle(obstacles, shadow_rows[1], right_cols.copy()) is None:
        return None

    off_row = random.choice(gap_rows_inside)
    off_side_cols = left_cols.copy() if random.random() < 0.5 else right_cols.copy()
    if _place_obstacle(obstacles, off_row, off_side_cols) is None:
        return None

    outside_selected = [r for r in range(h) if selected_rows[r] and not (r0 <= r <= r1)]
    if outside_selected and random.random() < 0.7:
        row = random.choice(outside_selected)
        cols = left_cols.copy() if random.random() < 0.5 else right_cols.copy()
        _place_obstacle(obstacles, row, cols)

    extra_pool = [r for r in range(h) if r not in {row for row, _ in obstacles}]
    random.shuffle(extra_pool)
    for row in extra_pool[: random.randint(0, 2)]:
        cols = left_cols.copy() if random.random() < 0.5 else right_cols.copy()
        _place_obstacle(obstacles, row, cols)

    inp = np.full((h, w), BG, dtype=int)
    inp[r0, c0 : c1 + 1] = frame_color
    inp[r1, c0 : c1 + 1] = frame_color
    inp[r0 : r1 + 1, c0] = frame_color
    inp[r0 : r1 + 1, c1] = frame_color
    for r in stripe_rows_inside:
        inp[r, c0 + 1 : c1] = stripe_color
    for r, c in obstacles:
        inp[r, c] = obstacle_color

    expected = _expected_output(inp.shape, BG, frame_color, obstacle_color, selected_rows, sorted(obstacles), box)
    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, inp):
        return None
    if np.array_equal(_expected_output(inp.shape, BG, frame_color, obstacle_color, selected_rows, [], box), expected):
        return None
    if not any(selected_rows[r] and c < (c0 + c1) / 2.0 for r, c in obstacles):
        return None
    if not any(selected_rows[r] and c > (c0 + c1) / 2.0 for r, c in obstacles):
        return None
    if not any((not selected_rows[r]) for r, _ in obstacles):
        return None
    if expected.shape != inp.shape or max(expected.shape) > 30:
        return None
    if len(colors_present(inp)) != 4:
        return None
    if not np.array_equal(observed, np.all(inp[r0 + 1 : r1, c0 + 1 : c1] == stripe_color, axis=1)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 034/3")
