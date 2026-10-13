from __future__ import annotations

import numpy as np

from arc_utils import bbox, components_mask, paste, random_free_location
import random


def _split_panels(g: np.ndarray) -> tuple[list[np.ndarray], set[int]]:
    h, w = g.shape
    candidates: list[tuple[list[np.ndarray], set[int]]] = []

    for horizontal in (False, True):
        dim = w if horizontal else h
        for n in range(3, dim + 1):
            if dim % n:
                continue
            ph, pw = (h, w // n) if horizontal else (h // n, w)
            if ph < 2 or pw < 2:
                continue
            if horizontal:
                panels = [g[:, i * pw : (i + 1) * pw] for i in range(n)]
            else:
                panels = [g[i * ph : (i + 1) * ph, :] for i in range(n)]

            template = panels[-1]
            template_colors = set(map(int, template.ravel()))
            diff_colors: set[int] = set()
            diff_counts = []
            for panel in panels[:-1]:
                diff = panel != template
                diff_counts.append(int(diff.sum()))
                diff_colors.update(map(int, panel[diff].ravel()))

            if not diff_colors or not all(diff_counts):
                continue
            if diff_colors & template_colors:
                continue
            occurrences = {
                color: sum(bool(np.any(panel == color)) for panel in panels[:-1])
                for color in diff_colors
            }
            if not all(count >= 2 for count in occurrences.values()):
                continue
            candidates.append((panels, diff_colors))

    assert len(candidates) == 1
    return candidates[0]


def _intervals(line: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    i = 0
    while i < len(line):
        if not line[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(line) and line[j + 1]:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def _shift_intervals(
    intervals: list[tuple[int, int]], step: int, length: int
) -> list[tuple[int, int]]:
    shifted = []
    for start, end in intervals:
        new_start = start + step
        new_end = end + step
        assert 0 <= new_start <= new_end < length
        shifted.append((new_start, new_end))
    return shifted


def _is_prefix(shorter: list[tuple[int, int]], longer: list[tuple[int, int]]) -> bool:
    return len(shorter) <= len(longer) and all(a == b for a, b in zip(shorter, longer))


def _required_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    b = bbox(mask)
    assert b is not None
    return b


def _motion_axis(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    ar0, ac0, ar1, ac1 = _required_bbox(a)
    br0, bc0, br1, bc1 = _required_bbox(b)
    row_motion = abs(br0 - ar0) + abs(br1 - ar1)
    col_motion = abs(bc0 - ac0) + abs(bc1 - ac1)
    assert row_motion != col_motion

    if row_motion > col_motion:
        deltas = [br0 - ar0, br1 - ar1]
        axis = 0
    else:
        deltas = [bc0 - ac0, bc1 - ac1]
        axis = 1
    nonzero = [delta for delta in deltas if delta]
    assert nonzero and all((delta > 0) == (nonzero[0] > 0) for delta in nonzero)
    direction = 1 if nonzero[0] > 0 else -1
    return axis, direction


def _draw_intervals(mask: np.ndarray, axis: int, line: int, intervals: list[tuple[int, int]]) -> None:
    for start, end in intervals:
        if axis == 0:
            mask[start : end + 1, line] = True
        else:
            mask[line, start : end + 1] = True


def _extrapolate_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert a.any() and b.any()
    assert not np.array_equal(a, b)

    axis, direction = _motion_axis(a, b)
    added = b & ~a
    assert added.any()
    added_coords = np.argwhere(added)
    extent = int(added_coords[:, axis].max() - added_coords[:, axis].min() + 1)
    global_step = direction * extent

    out = np.zeros_like(b, dtype=bool)
    line_count = b.shape[1 - axis]
    length = b.shape[axis]
    for line in range(line_count):
        if axis == 0:
            a_intervals = _intervals(a[:, line])
            b_intervals = _intervals(b[:, line])
        else:
            a_intervals = _intervals(a[line, :])
            b_intervals = _intervals(b[line, :])

        if not a_intervals and not b_intervals:
            next_intervals: list[tuple[int, int]] = []
        elif a_intervals and len(a_intervals) == len(b_intervals):
            next_intervals = []
            for (a0, a1), (b0, b1) in zip(a_intervals, b_intervals):
                new_start = b0 + (b0 - a0)
                new_end = b1 + (b1 - a1)
                assert 0 <= new_start <= new_end < length
                next_intervals.append((new_start, new_end))
        elif a_intervals and _is_prefix(a_intervals, b_intervals):
            extra_count = len(b_intervals) - len(a_intervals)
            assert 0 < extra_count <= len(a_intervals)
            previous = a_intervals[-extra_count:]
            extra = b_intervals[-extra_count:]
            steps = {
                end_new - end_old
                for old, new in zip(previous, extra)
                for end_old, end_new in zip(old, new)
            }
            assert len(steps) == 1
            step = steps.pop()
            assert step
            next_intervals = list(b_intervals) + _shift_intervals(extra, step, length)
        elif not a_intervals and b_intervals:
            next_intervals = list(b_intervals) + _shift_intervals(
                b_intervals, global_step, length
            )
        else:
            raise AssertionError

        _draw_intervals(out, axis, line, next_intervals)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Predict the next panel in a framed sequence by continuing each scheduled colored interval.

    Input structure
    - The grid is made from three or more equal panels in one row or one column.
    - The final input panel is the static target template; every earlier panel differs from it only by dynamic colors absent from the template.
    - Each dynamic color appears in a regular temporal schedule across the earlier panels.

    Rule
    - Keep the final template panel as the output canvas.
    - For every dynamic color whose schedule reaches the next panel, compare its last two appearances.
    - Extrapolate each one-dimensional interval along the color's changing row or column axis and paint that predicted mask.

    Key insight
    - The task is not asking for the last visible panel, but for the next frame after the blank/template slot.
    """
    g = np.asarray(grid, dtype=int)
    panels, diff_colors = _split_panels(g)
    template = panels[-1]
    prior = panels[:-1]
    next_time = len(prior)

    predictions: list[tuple[int, np.ndarray]] = []
    for color in diff_colors:
        times = [i for i, panel in enumerate(prior) if np.any(panel == color)]
        assert len(times) >= 2
        gaps = [b - a for a, b in zip(times, times[1:])]
        assert len(set(gaps)) == 1
        period = gaps[0]
        if next_time - times[-1] != period:
            continue

        previous = prior[times[-2]] == color
        current = prior[times[-1]] == color
        predictions.append((color, _extrapolate_mask(previous, current)))

    assert predictions
    occupied = np.zeros_like(template, dtype=bool)
    for _color, mask in predictions:
        assert not np.any(occupied & mask)
        occupied |= mask

    out = template.copy()
    for color, mask in predictions:
        out[mask] = color
    return out

# --------------- generator (o5x) ---------------

# Canvas, orientation, and color sampling.
PANEL_SIZE_MIN, PANEL_SIZE_MAX = 5, 11
PANEL_COUNT_CHOICES, ORIENTATION_CHOICES = (3, 5), (False, True)
COLOR_POOL, TOTAL_COLOR_CHOICES, MOTION_AXIS_CHOICES = range(10), (4, 5), (0, 1)

# Template decoration density and symmetry.
TEMPLATE_MIN_DECO_CELLS = 4
TEMPLATE_DECO_MIN_RATIO, TEMPLATE_DECO_MAX_RATIO = 0.10, 0.30

# Blob motif and growth bounds.
BLOB_GROWTH_MIN, BLOB_GROWTH_MAX = 1, 4
BLOB_MAX_INITIAL_CELLS, BLOB_MAX_COMPONENTS = 7, 2
BLOB_MAX_LINE_COUNT, BLOB_MAX_WIDTH = 3, 3
BLOB_DIRECTIONS = (-1, 1)

# Dynamic-color schedules.
SHORT_PRIOR_COUNT, MAX_DYNAMIC_COLORS = 2, 3
SCHEDULE_ALWAYS_2 = {"period": 1, "times": [0, 1], "scheduled": True}
SCHEDULE_ALWAYS_4 = {"period": 1, "times": [0, 1, 2, 3], "scheduled": True}
SCHEDULE_EVEN_4 = {"period": 2, "times": [0, 2], "scheduled": True}
SCHEDULE_FIRST3_4 = {"period": 1, "times": [0, 1, 2], "scheduled": False}
SCHEDULE_ODD_4 = {"period": 2, "times": [1, 3], "scheduled": False}
SCHEDULE_SECOND_CHOICES = (SCHEDULE_ALWAYS_4, SCHEDULE_EVEN_4)
SCHEDULE_THIRD_CHOICES = (SCHEDULE_ALWAYS_4, SCHEDULE_EVEN_4, SCHEDULE_FIRST3_4, SCHEDULE_ODD_4)

# Placement and retry limits.
MASK_TRANSPARENT, PLACEMENT_BORDER = 0, 1
GENERATION_RETRIES = 300

def _sym_orbits(h: int, w: int) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if (r, c) in seen:
                continue
            orbit = sorted({(r, c), (r, w - 1 - c), (h - 1 - r, c), (h - 1 - r, w - 1 - c)})
            seen.update(orbit)
            orbits.append(orbit)
    return orbits


def _make_template(h: int, w: int, bg: int, deco: int) -> np.ndarray:
    out = np.full((h, w), bg, dtype=int)
    area = h * w
    min_deco = max(TEMPLATE_MIN_DECO_CELLS, int(np.ceil(area * TEMPLATE_DECO_MIN_RATIO)))
    max_deco = max(min_deco, int(np.floor(area * TEMPLATE_DECO_MAX_RATIO)))
    target = random.randint(min_deco, max_deco)
    orbits = _sym_orbits(h, w)
    random.shuffle(orbits)
    count = 0
    for orbit in orbits:
        if count >= target:
            break
        if count + len(orbit) > max_deco and count >= min_deco:
            continue
        for r, c in orbit:
            out[r, c] = deco
        count += len(orbit)
    return out


def _base_patterns(axis_len: int, line_len: int, max_back: int) -> list[tuple[int, list[list[tuple[int, int]]]]]:
    patterns: list[tuple[int, list[list[tuple[int, int]]]]] = []
    for growth in range(BLOB_GROWTH_MIN, BLOB_GROWTH_MAX + 1):
        extra = max_back * growth
        max_initial_width = axis_len - extra
        if max_initial_width < 1:
            continue

        for lines in range(1, min(BLOB_MAX_LINE_COUNT, line_len) + 1):
            for width in range(1, min(BLOB_MAX_WIDTH, max_initial_width, BLOB_MAX_INITIAL_CELLS // lines) + 1):
                patterns.append((growth, [[(0, width - 1)] for _ in range(lines)]))

        if line_len >= 2 and max_initial_width >= 2:
            for lines in range(2, min(BLOB_MAX_LINE_COUNT + 1, line_len) + 1):
                width = min(2, max_initial_width)
                if lines * width <= BLOB_MAX_INITIAL_CELLS and width + 1 + extra <= axis_len:
                    patterns.append((growth, [[(i % 2, i % 2 + width - 1)] for i in range(lines)]))

        if line_len >= 3 and max_initial_width >= 3:
            tee = [[(1, 1)], [(0, 2)], [(1, 1)]]
            if 3 + extra <= axis_len:
                patterns.append((growth, tee))

        if line_len >= 2:
            gap = extra + 1
            if gap + 2 + extra <= axis_len:
                for lines in range(2, min(BLOB_MAX_LINE_COUNT, line_len) + 1):
                    split = [[(0, 0), (gap + 1, gap + 1)] for _ in range(lines)]
                    if 2 * lines <= BLOB_MAX_INITIAL_CELLS:
                        patterns.append((growth, split))
    return patterns


def _draw_local(
    intervals0: list[list[tuple[int, int]]],
    growth: int,
    max_back: int,
    back: int,
    direction: int,
    axis: int,
) -> np.ndarray:
    grown = max_back - back
    width = max(end for row in intervals0 for _start, end in row) + max_back * growth + 1
    local = np.zeros((len(intervals0), width), dtype=bool)
    for r, row in enumerate(intervals0):
        for start, end in row:
            local[r, start : end + grown * growth + 1] = True
    if direction < 0:
        local = np.fliplr(local)
    if axis == 0:
        local = local.T
    return local


def _make_blob(axis_len: int, line_len: int, axis: int, max_back: int) -> tuple[int, dict[int, np.ndarray]] | None:
    patterns = _base_patterns(axis_len, line_len, max_back)
    if not patterns:
        return None
    by_growth: dict[int, list[list[list[tuple[int, int]]]]] = {}
    for growth, intervals0 in patterns:
        by_growth.setdefault(growth, []).append(intervals0)
    growth = random.choice(sorted(by_growth))
    intervals0 = random.choice(by_growth[growth])
    direction = random.choice(BLOB_DIRECTIONS)
    masks = {
        back: _draw_local(intervals0, growth, max_back, back, direction, axis)
        for back in range(max_back + 1)
    }
    largest = masks[0]
    if largest.shape[0] > (line_len if axis == 1 else axis_len):
        return None
    if largest.shape[1] > (axis_len if axis == 1 else line_len):
        return None
    if len(components_mask(masks[max_back], connectivity=4)) > BLOB_MAX_COMPONENTS:
        return None
    return growth, masks


def _schedules(n_prior: int, n_colors: int) -> list[dict]:
    if n_prior == SHORT_PRIOR_COUNT:
        return [SCHEDULE_ALWAYS_2 for _ in range(n_colors)]

    schedules = [SCHEDULE_ALWAYS_4, random.choice(SCHEDULE_SECOND_CHOICES)]
    if n_colors == MAX_DYNAMIC_COLORS:
        schedules.append(random.choice(SCHEDULE_THIRD_CHOICES))
    return schedules


def _place_masks(
    occupied: np.ndarray,
    local_masks: dict[int, np.ndarray],
) -> dict[int, np.ndarray] | None:
    largest = local_masks[0]
    patch = np.where(largest, 1, MASK_TRANSPARENT)
    try:
        r0, c0 = random_free_location(
            occupied, patch, transparent=MASK_TRANSPARENT, border=PLACEMENT_BORDER
        )
    except ValueError:
        return None
    paste(occupied, patch, r0, c0, transparent=MASK_TRANSPARENT)

    out: dict[int, np.ndarray] = {}
    for back, local in local_masks.items():
        full = np.zeros_like(occupied, dtype=bool)
        full[r0 : r0 + local.shape[0], c0 : c0 + local.shape[1]] = local
        out[back] = full
    return out


def _place_dynamic_masks(
    occupied: np.ndarray,
    n_dynamic: int,
    schedules: list[dict],
    motion_axis: int,
    n_prior: int,
    axis_len: int,
    line_len: int,
) -> dict[int, dict[int, np.ndarray]] | None:
    """Return full-panel masks by color index, or None if any blob fails."""
    order = list(range(n_dynamic))
    random.shuffle(order)
    placed: dict[int, dict[int, np.ndarray]] = {}
    for idx in order:
        sched = schedules[idx]
        if sched["scheduled"]:
            max_back = max((n_prior - t) // sched["period"] for t in sched["times"])
        else:
            last = sched["times"][-1]
            max_back = max((last - t) // sched["period"] for t in sched["times"])

        blob = _make_blob(axis_len, line_len, motion_axis, max_back)
        if blob is None:
            return None
        _growth, local_masks = blob
        full_masks = _place_masks(occupied, local_masks)
        if full_masks is None:
            return None
        placed[idx] = full_masks
    return placed


def _paint_dynamic_masks(
    panels: list[np.ndarray],
    output: np.ndarray,
    dynamic: list[int],
    schedules: list[dict],
    placed: dict[int, dict[int, np.ndarray]],
    n_prior: int,
) -> None:
    """Mutate prior panels and output with the preplaced dynamic-color masks."""
    for idx, color in enumerate(dynamic):
        sched = schedules[idx]
        full_masks = placed[idx]
        if sched["scheduled"]:
            for t in sched["times"]:
                back = (n_prior - t) // sched["period"]
                panels[t][full_masks[back]] = color
            output[full_masks[0]] = color
        else:
            last = sched["times"][-1]
            for t in sched["times"]:
                back = (last - t) // sched["period"]
                panels[t][full_masks[back]] = color


def _assemble_valid_input(
    panels: list[np.ndarray],
    template: np.ndarray,
    output: np.ndarray,
    horizontal: bool,
) -> np.ndarray | None:
    """Return tiled input only when the solver oracle matches planned output."""
    if horizontal:
        inp = np.concatenate([*panels, template], axis=1)
    else:
        inp = np.concatenate([*panels, template], axis=0)

    try:
        predicted = main(inp)
    except Exception:
        return None
    if not np.array_equal(predicted, output):
        return None
    return inp


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(PANEL_SIZE_MIN, PANEL_SIZE_MAX)
    w = random.randint(PANEL_SIZE_MIN, PANEL_SIZE_MAX)
    horizontal = random.choice(ORIENTATION_CHOICES)
    n_panels = random.choice(PANEL_COUNT_CHOICES)
    n_prior = n_panels - 1

    colors = random.sample(COLOR_POOL, random.choice(TOTAL_COLOR_CHOICES))
    bg, deco = colors[0], colors[1]
    dynamic = colors[2:]

    template = _make_template(h, w, bg, deco)
    occupied = np.where(template == bg, MASK_TRANSPARENT, 1)
    motion_axis = random.choice(MOTION_AXIS_CHOICES)
    schedules = _schedules(n_prior, len(dynamic))

    panels = [template.copy() for _ in range(n_prior)]
    output = template.copy()
    axis_len = h - 2 if motion_axis == 0 else w - 2
    line_len = w - 2 if motion_axis == 0 else h - 2

    placed = _place_dynamic_masks(
        occupied, len(dynamic), schedules, motion_axis, n_prior, axis_len, line_len
    )
    if placed is None:
        return None
    _paint_dynamic_masks(panels, output, dynamic, schedules, placed, n_prior)
    return _assemble_valid_input(panels, template, output, horizontal)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
