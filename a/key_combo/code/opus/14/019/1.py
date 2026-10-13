from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

GEN_RETRIES = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_rectangle(grid: np.ndarray) -> tuple[int, int, int, int]:
    body = (grid == BLUE) | (grid == GREEN)
    r0, c0, r1, c1 = _bbox(body)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all((patch == BLUE) | (patch == GREEN)), "blue rectangle must remain a full blue/green block"
    return r0, c0, r1, c1


def _gap_bounds(rect_right: int, local_row: int) -> tuple[int, int] | None:
    if local_row <= 0:
        return None
    return rect_right - local_row + 1, rect_right


def _periodic_hits(
    local_row: int, green_col: int, step: int, rect_right: int, total_cols: int
) -> tuple[list[int], list[int]]:
    gap = _gap_bounds(rect_right, local_row)
    fills: list[int] = []
    spill: list[int] = []
    col = int(green_col)
    first = True
    while col < total_cols:
        if gap is not None and gap[0] <= col <= gap[1] and not first:
            fills.append(int(col))
        elif col > rect_right:
            spill.append(int(col))
        first = False
        col += step
    return fills, spill


def _apply_transform_from_rect(
    grid: np.ndarray, rect: tuple[int, int, int, int]
) -> np.ndarray:
    r0, c0, r1, c1 = rect
    out = np.asarray(grid, dtype=int).copy()

    # Remove the visible blue pixels; green markers are overlays and stay put.
    out[out == BLUE] = BLACK

    # Shear the entire underlying solid rectangle, including cells hidden by green markers.
    for rr in range(r0, r1 + 1):
        local_row = rr - r0
        for cc in range(c0, c1 + 1):
            dst_c = cc - local_row
            if 0 <= dst_c < out.shape[1] and out[rr, dst_c] == BLACK:
                out[rr, dst_c] = BLUE

    for rr in range(r0, r1 + 1):
        reds = [int(c) for c in np.flatnonzero(grid[rr] == RED)]
        greens = [int(c) for c in np.flatnonzero(grid[rr] == GREEN)]
        assert len(greens) <= 1, "expected at most one green marker per rectangle row"
        assert len(reds) in (0, 2), "expected either zero or two red markers on a row"
        if len(reds) != 2 or len(greens) != 1:
            continue

        step = abs(reds[1] - reds[0])
        assert step > 0
        gap = _gap_bounds(c1, rr - r0)
        if gap is None:
            continue

        start = greens[0]
        col = start
        while col < out.shape[1]:
            if col > c1:
                break
            if gap[0] <= col <= gap[1] and out[rr, col] == BLACK:
                out[rr, col] = GREEN
            col += step

    out[out == RED] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Shear a blue rectangle down-left and periodically fill only the emptied gap cells.

    Input structure:
    - Black background with one central blue rectangle whose rows may contain a single
      green marker overlaid inside the rectangle.
    - On some rectangle rows, exactly two red markers lie outside the rectangle and encode
      a row-specific period by their column gap.
    - Only rows carrying both the red pair and the green marker participate in the periodic fill.

    Rule:
    - Identify the blue/green rectangle and shear its row r left by r cells, preserving the
      green markers in place and discarding blue pixels that move past the left border.
    - For every marked row, let step be the absolute column distance between its two red markers.
      Starting at the green marker column, advance rightward by step and paint green only on
      cells that are black after the shear and also belong to that row's shear-created gap
      inside the rectangle's original column span.
    - Finally erase all red markers to black.

    Key insight:
    - The rectangle shear determines a row-dependent right-side gap of width r, while the red
      gap supplies the period and the green marker supplies the modular phase; the fill is
      uniquely determined only by combining all three constraints.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    rect = _find_rectangle(g)
    return _apply_transform_from_rect(g, rect)


def _pick_geometry() -> tuple[int, int, int, int, int, int]:
    rect_h = random.randint(4, 7)
    rect_w = random.randint(max(6, rect_h + 1), 10)

    clip_left = random.random() < 0.4
    if clip_left:
        left_margin = random.randint(1, rect_h - 2)
    else:
        left_margin = random.randint(rect_h - 1, rect_h + 4)
    right_margin = random.randint(4, 8)

    top_margin = random.randint(1, 4)
    bottom_margin = random.randint(1, 4)

    rows = top_margin + rect_h + bottom_margin
    cols = left_margin + rect_w + right_margin
    assert rows <= 30 and cols <= 30
    return rows, cols, top_margin, left_margin, rect_h, rect_w


def _enumerate_progressions(
    local_row: int, rect_left: int, rect_right: int, total_cols: int, steps: list[int]
) -> list[dict]:
    options = []
    for step in steps:
        for green_col in range(rect_left, rect_right + 1):
            fills, spill = _periodic_hits(local_row, green_col, step, rect_right, total_cols)
            if spill:
                options.append(
                    {
                        "step": int(step),
                        "green": int(green_col),
                        "fills": fills,
                        "spill": spill,
                    }
                )
    return options


def _find_compare_choice(
    rows: list[int], rect_left: int, rect_right: int, total_cols: int, steps: list[int]
) -> tuple[tuple[int, int], dict] | None:
    pair_options: list[tuple[tuple[int, int], dict]] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            r_small, r_large = rows[i], rows[j]
            for step in steps:
                for green_col in range(rect_left, rect_right + 1):
                    fills_small, _ = _periodic_hits(r_small, green_col, step, rect_right, total_cols)
                    fills_large, spill_large = _periodic_hits(r_large, green_col, step, rect_right, total_cols)
                    if len(fills_small) >= 1 and len(fills_large) > len(fills_small) and spill_large:
                        pair_options.append(
                            (
                                (r_small, r_large),
                                {
                                    "step": int(step),
                                    "green": int(green_col),
                                    "fills_small": fills_small,
                                    "fills_large": fills_large,
                                    "spill_large": spill_large,
                                },
                            )
                        )
    if not pair_options:
        return None
    return random.choice(pair_options)


def _choose_phase_choice(
    local_row: int,
    rect_left: int,
    rect_right: int,
    total_cols: int,
    steps: list[int],
    compare_step: int,
    compare_fills: list[int],
) -> dict | None:
    same_step_zero: list[dict] = []
    same_step_diff: list[dict] = []
    any_other: list[dict] = []
    for option in _enumerate_progressions(local_row, rect_left, rect_right, total_cols, steps):
        if option["step"] == compare_step:
            if len(option["fills"]) == 0:
                same_step_zero.append(option)
            elif option["fills"] != compare_fills:
                same_step_diff.append(option)
        elif option["fills"] != compare_fills:
            any_other.append(option)

    if same_step_zero:
        return random.choice(same_step_zero)
    if same_step_diff and random.random() < 0.75:
        return random.choice(same_step_diff)
    if any_other:
        return random.choice(any_other)
    if same_step_diff:
        return random.choice(same_step_diff)
    return None


def _choose_extra_choice(
    local_row: int,
    rect_left: int,
    rect_right: int,
    total_cols: int,
    steps: list[int],
    used: set[tuple[int, int]],
) -> dict | None:
    candidates = _enumerate_progressions(local_row, rect_left, rect_right, total_cols, steps)
    random.shuffle(candidates)
    for option in candidates:
        key = (option["step"], option["green"])
        if key not in used:
            return option
    return candidates[0] if candidates else None


def _row_side_options(local_row: int, step: int, rect_left: int, rect_right: int, total_cols: int) -> list[str]:
    options: list[str] = []
    safe_left_width = rect_left - local_row
    right_width = total_cols - rect_right - 1
    if safe_left_width >= step + 1:
        options.append("left")
    if right_width >= step + 1:
        options.append("right")
    return options


def _place_red_pair(
    local_row: int,
    step: int,
    side: str,
    rect_left: int,
    rect_right: int,
    total_cols: int,
) -> tuple[int, int]:
    if side == "left":
        end = rect_left - local_row - 1
        start = random.randint(0, end - step)
        return int(start), int(start + step)

    start = random.randint(rect_right + 1, total_cols - 1 - step)
    return int(start), int(start + step)


def _expected_from_config(
    inp: np.ndarray,
    rect: tuple[int, int, int, int],
) -> np.ndarray:
    return _apply_transform_from_rect(inp, rect)


def _semantic_checks(
    inp: np.ndarray,
    out: np.ndarray,
    rect: tuple[int, int, int, int],
    row_configs: dict[int, dict],
    compare_rows: tuple[int, int],
    phase_row: int,
) -> bool:
    r0, c0, r1, c1 = rect
    rect_h = r1 - r0 + 1

    if len(row_configs) < 3 or len(row_configs) >= rect_h:
        return False
    if compare_rows[0] not in row_configs or compare_rows[1] not in row_configs or phase_row not in row_configs:
        return False

    cfg_small = row_configs[compare_rows[0]]
    cfg_large = row_configs[compare_rows[1]]
    cfg_phase = row_configs[phase_row]
    if cfg_small["step"] != cfg_large["step"]:
        return False
    if len(cfg_small["fills"]) < 1 or len(cfg_large["fills"]) <= len(cfg_small["fills"]):
        return False
    if cfg_phase["step"] == cfg_large["step"] and cfg_phase["fills"] == cfg_large["fills"]:
        return False

    if not np.array_equal(out, _expected_from_config(inp, rect)):
        return False
    if np.array_equal(inp, out):
        return False

    added_green = int(np.count_nonzero((out == GREEN) & (inp != GREEN)))
    if added_green == 0:
        return False

    # There must be at least one progression beat beyond the gap that stays black.
    has_black_spill = False
    for local_row, cfg in row_configs.items():
        rr = r0 + local_row
        red_cols = set(cfg["reds"])
        for col in cfg["spill"]:
            if col in red_cols:
                continue
            if out[rr, col] == BLACK:
                has_black_spill = True
                break
        if has_black_spill:
            break
    if not has_black_spill:
        return False

    # Every marked row should still show its original green marker after the transform.
    for local_row, cfg in row_configs.items():
        rr = r0 + local_row
        if out[rr, cfg["green"]] != GREEN:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    rows, cols, top, left, rect_h, rect_w = _pick_geometry()
    rect = (top, left, top + rect_h - 1, left + rect_w - 1)
    rect_left, rect_right = left, left + rect_w - 1

    active_pool = list(range(1, rect_h))
    if len(active_pool) < 3:
        return None
    active_count = random.randint(3, min(len(active_pool), 5))
    active_rows = sorted(random.sample(active_pool, active_count))

    max_step = min(5, cols - rect_right - 2)
    if max_step < 2:
        return None
    steps = list(range(2, max_step + 1))

    compare_choice = _find_compare_choice(active_rows, rect_left, rect_right, cols, steps)
    if compare_choice is None:
        return None
    compare_rows, compare_cfg = compare_choice

    remaining = [r for r in active_rows if r not in compare_rows]
    if not remaining:
        return None
    phase_row = random.choice(remaining)
    phase_cfg = _choose_phase_choice(
        phase_row,
        rect_left,
        rect_right,
        cols,
        steps,
        compare_cfg["step"],
        compare_cfg["fills_large"],
    )
    if phase_cfg is None:
        return None

    row_configs: dict[int, dict] = {
        compare_rows[0]: {
            "step": compare_cfg["step"],
            "green": compare_cfg["green"],
            "fills": compare_cfg["fills_small"],
            "spill": compare_cfg["spill_large"],
        },
        compare_rows[1]: {
            "step": compare_cfg["step"],
            "green": compare_cfg["green"],
            "fills": compare_cfg["fills_large"],
            "spill": compare_cfg["spill_large"],
        },
        phase_row: phase_cfg.copy(),
    }

    used_pairs = {(cfg["step"], cfg["green"]) for cfg in row_configs.values()}
    for local_row in active_rows:
        if local_row in row_configs:
            continue
        extra = _choose_extra_choice(local_row, rect_left, rect_right, cols, steps, used_pairs)
        if extra is None:
            return None
        row_configs[local_row] = extra.copy()
        used_pairs.add((extra["step"], extra["green"]))

    feasible_sides = {
        local_row: _row_side_options(local_row, cfg["step"], rect_left, rect_right, cols)
        for local_row, cfg in row_configs.items()
    }
    if any(not options for options in feasible_sides.values()):
        return None

    side_choice: dict[int, str] = {}
    left_candidates = [r for r, options in feasible_sides.items() if "left" in options]
    right_candidates = [r for r, options in feasible_sides.items() if "right" in options]
    if left_candidates and right_candidates:
        left_row = random.choice(left_candidates)
        right_pool = [r for r in right_candidates if r != left_row] or right_candidates
        side_choice[left_row] = "left"
        side_choice[random.choice(right_pool)] = "right"
    for local_row, options in feasible_sides.items():
        if local_row in side_choice:
            continue
        side_choice[local_row] = random.choice(options)

    grid = np.zeros((rows, cols), dtype=int)
    grid[top : top + rect_h, left : left + rect_w] = BLUE

    for local_row, cfg in row_configs.items():
        rr = top + local_row
        green_col = cfg["green"]
        grid[rr, green_col] = GREEN
        reds = _place_red_pair(local_row, cfg["step"], side_choice[local_row], rect_left, rect_right, cols)
        cfg["reds"] = reds
        grid[rr, reds[0]] = RED
        grid[rr, reds[1]] = RED

    try:
        out = main(grid)
    except Exception:
        return None

    if not _semantic_checks(grid, out, rect, row_configs, compare_rows, phase_row):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shear-periodic-fill input")
