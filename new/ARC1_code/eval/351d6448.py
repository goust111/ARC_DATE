from __future__ import annotations

import numpy as np

from arc_utils import colors_present, paste
import random


def _split_panels(g: np.ndarray) -> tuple[list[np.ndarray], int]:
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(g[0, 0])
    separator_rows = [
        r
        for r in range(g.shape[0])
        if np.all(g[r, :] == int(g[r, 0])) and int(g[r, 0]) != bg
    ]
    assert separator_rows, "expected solid separator rows"

    panel_h = separator_rows[0]
    assert panel_h > 0, "first panel must be nonempty"
    expected_rows = [panel_h + i * (panel_h + 1) for i in range(len(separator_rows))]
    assert separator_rows == expected_rows, "separator rows must be evenly spaced"
    assert g.shape[0] == (len(separator_rows) + 1) * panel_h + len(separator_rows)

    panels = []
    start = 0
    for sep in separator_rows:
        panels.append(g[start:sep, :])
        start = sep + 1
    panels.append(g[start:, :])
    assert len({panel.shape for panel in panels}) == 1, "all panels must have the same shape"
    return panels, bg


def _all_panel_colors(panels: list[np.ndarray], bg: int) -> list[int]:
    seen: list[int] = []
    for panel in panels:
        for color in colors_present(panel, exclude={bg}):
            if color not in seen:
                seen.append(color)
    return seen


def _translate_array(a: np.ndarray, dr: int, dc: int, bg: int) -> np.ndarray:
    out = np.full_like(a, bg)
    height, width = map(int, a.shape)
    for r, c in np.argwhere(a != bg):
        nr = int(r) + dr
        nc = int(c) + dc
        if 0 <= nr < height and 0 <= nc < width:
            out[nr, nc] = int(a[r, c])
    return out


def _translate_mask(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    height, width = map(int, mask.shape)
    for r, c in np.argwhere(mask):
        nr = int(r) + dr
        nc = int(c) + dc
        if 0 <= nr < height and 0 <= nc < width:
            out[nr, nc] = True
    return out


def _next_by_translation(panels: list[np.ndarray], bg: int) -> np.ndarray | None:
    height, width = map(int, panels[0].shape)
    candidates = []
    for dr in range(-height, height + 1):
        for dc in range(-width, width + 1):
            if dr == 0 and dc == 0:
                continue
            if all(
                np.array_equal(_translate_array(panels[i], dr, dc, bg), panels[i + 1])
                for i in range(len(panels) - 1)
            ):
                candidates.append((dr, dc))

    if not candidates:
        return None
    assert len(candidates) == 1, "expected a unique panel translation"
    dr, dc = candidates[0]
    return _translate_array(panels[-1], dr, dc, bg)


def _increasing_color(panels: list[np.ndarray], bg: int) -> int | None:
    candidates = []
    for color in _all_panel_colors(panels, bg):
        masks = [panel == color for panel in panels]
        is_increasing = all(
            np.all((~masks[i]) | masks[i + 1]) for i in range(len(masks) - 1)
        )
        changed = any(
            not np.array_equal(masks[i], masks[i + 1]) for i in range(len(masks) - 1)
        )
        if is_increasing and changed:
            candidates.append(color)

    if not candidates:
        return None
    assert len(candidates) == 1, "expected a unique increasing color"
    return int(candidates[0])


def _mask_shift(src: np.ndarray, dst: np.ndarray) -> tuple[int, int] | None:
    if int(src.sum()) == 0 or int(src.sum()) != int(dst.sum()):
        return None

    height, width = map(int, src.shape)
    candidates = []
    for dr in range(-height, height + 1):
        for dc in range(-width, width + 1):
            if np.array_equal(_translate_mask(src, dr, dc), dst):
                candidates.append((dr, dc))

    if not candidates:
        return None
    assert len(candidates) == 1, "expected a unique increment shift"
    return candidates[0]


def _next_by_repeated_increment(panels: list[np.ndarray], bg: int) -> np.ndarray | None:
    color = _increasing_color(panels, bg)
    if color is None:
        return None

    masks = [panel == color for panel in panels]
    increments = [masks[i + 1] & ~masks[i] for i in range(len(masks) - 1)]
    if any(not increment.any() for increment in increments):
        return None

    first_shift = _mask_shift(increments[0], increments[1])
    second_shift = _mask_shift(increments[1], increments[2])
    if first_shift is None or second_shift is None or first_shift != second_shift:
        return None

    dr, dc = second_shift
    next_increment = _translate_mask(increments[-1], dr, dc)
    assert next_increment.any(), "next increment must stay on the panel"
    assert not np.any(masks[-1] & next_increment), "next increment must add new cells"

    out = panels[-1].copy()
    out[next_increment] = color
    return out


def _next_by_fixed_mask_prefix(panels: list[np.ndarray], bg: int) -> np.ndarray | None:
    foreground_masks = [panel != bg for panel in panels]
    if not all(np.array_equal(foreground_masks[0], mask) for mask in foreground_masks[1:]):
        return None

    color = _increasing_color(panels, bg)
    if color is None:
        return None

    foreground = foreground_masks[0]
    height, width = map(int, foreground.shape)
    columns = np.broadcast_to(np.arange(width), (height, width))
    active_masks = [panel == color for panel in panels]
    options = []

    right_fronts = []
    right_ok = True
    for mask in active_masks:
        if not mask.any():
            right_ok = False
            break
        front = int(np.argwhere(mask)[:, 1].max())
        if not np.array_equal(mask, foreground & (columns <= front)):
            right_ok = False
            break
        right_fronts.append(front)
    if right_ok:
        steps = [right_fronts[i + 1] - right_fronts[i] for i in range(len(right_fronts) - 1)]
        if len(set(steps)) == 1 and steps[0] > 0:
            options.append(("right", right_fronts[-1], steps[0]))

    left_fronts = []
    left_ok = True
    for mask in active_masks:
        if not mask.any():
            left_ok = False
            break
        front = int(np.argwhere(mask)[:, 1].min())
        if not np.array_equal(mask, foreground & (columns >= front)):
            left_ok = False
            break
        left_fronts.append(front)
    if left_ok:
        steps = [left_fronts[i + 1] - left_fronts[i] for i in range(len(left_fronts) - 1)]
        if len(set(steps)) == 1 and steps[0] < 0:
            options.append(("left", left_fronts[-1], steps[0]))

    if not options:
        return None
    assert len(options) == 1, "expected a unique prefix growth direction"
    direction, front, step = options[0]

    if direction == "right":
        next_cells = foreground & (columns > front) & (columns <= front + step)
    else:
        next_cells = foreground & (columns < front) & (columns >= front + step)

    assert next_cells.any(), "prefix growth must add foreground cells"
    assert not np.any(active_masks[-1] & next_cells), "prefix growth must add new cells"

    out = panels[-1].copy()
    out[next_cells] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Continue the four-frame panel sequence and return the next panel.

    Input structure
    The input is a vertical stack of four equal-size panels separated by solid
    full-width rows.  The panel background is the shared top-left color, and the
    output has the shape of one panel.

    Rule
    Continue the panel history by the unique visible extrapolation: a whole
    non-background sprite may translate by a constant offset, an increasing color
    may add the same shifted increment each step, or an increasing color may
    recolor the next columns of a fixed foreground mask.

    Key insight
    The separator rows turn the grid into an animation strip; the answer is the
    next frame, not a crop or a direct copy of any existing frame.
    """
    g = np.asarray(grid, dtype=int)
    panels, bg = _split_panels(g)
    assert len(panels) == 4, "expected exactly four observed panels"

    matches = []
    for model in (
        _next_by_translation,
        _next_by_repeated_increment,
        _next_by_fixed_mask_prefix,
    ):
        out = model(panels, bg)
        if out is not None:
            matches.append(out)

    assert len(matches) == 1, "expected exactly one panel extrapolation model"
    return matches[0]

# --------------- generator (o5x) ---------------

# Canvas and palette.
GEN_BG_COLOR = 0
GEN_PANEL_HEIGHT = 3
GEN_PANEL_WIDTH_MIN, GEN_PANEL_WIDTH_MAX = 8, 15
GEN_PANEL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_PANEL_COLOR_COUNTS = [1, 2]
GEN_PREFERRED_SEPARATOR_COLOR = 5
GEN_SEPARATOR_COLORS = list(range(1, 10))

# Animation counts and allowed motion.
GEN_FRAME_COUNT = 5
GEN_OBSERVED_FRAME_COUNT = 4
GEN_MOTION_STEPS = (-1, 0, 1)
GEN_MOTION_DELTAS = [
    (dr, dc) for dr in GEN_MOTION_STEPS for dc in GEN_MOTION_STEPS
    if (dr, dc) != (0, 0)
]
GEN_TRANSLATION_TIMES = range(GEN_FRAME_COUNT)
GEN_INCREMENT_TIMES = range(-1, GEN_OBSERVED_FRAME_COUNT)

# Sprite, increment, and prefix shape controls.
GEN_LOCAL_SHAPE_MAX_HEIGHT, GEN_LOCAL_SHAPE_MAX_WIDTH = 3, 3
GEN_SPRITE_CELL_MIN, GEN_SPRITE_CELL_MAX = 2, 4
GEN_INCREMENT_CELL_MIN, GEN_INCREMENT_CELL_MAX = 1, 3
GEN_PREFIX_STEP_CHOICES = [1, 2]
GEN_PREFIX_TWO_STEP_MIN_WIDTH = 9
GEN_PREFIX_ADVANCE_COUNT = GEN_FRAME_COUNT - 1
GEN_PREFIX_BAND_CHECK_COUNT = GEN_OBSERVED_FRAME_COUNT - 1
GEN_PREFIX_ROWS = [0, 1, 2]
GEN_PREFIX_EDGE_ROWS = [1, 2]
GEN_PREFIX_MIN_ROWS_PER_COLUMN = 1
GEN_PREFIX_VARIATION_ROW, GEN_PREFIX_REPAIR_ROW = 0, 1
GEN_PREFIX_FLIP_CHOICES = [False, True]

# Retry budget.
GEN_MAX_ATTEMPTS = 300


def _gen_stack(frames: list[np.ndarray], sep_color: int) -> np.ndarray:
    rows = []
    width = int(frames[0].shape[1])
    sep = np.full((1, width), int(sep_color), dtype=int)
    for frame in frames[:-1]:
        rows.extend([frame, sep.copy()])
    rows.append(frames[-1])
    return np.vstack(rows)


def _gen_sep_color(panel_colors: set[int]) -> int:
    if GEN_PREFERRED_SEPARATOR_COLOR not in panel_colors:
        return GEN_PREFERRED_SEPARATOR_COLOR
    return random.choice([c for c in GEN_SEPARATOR_COLORS if c not in panel_colors])


def _gen_connected_coords(max_h: int, max_w: int, count: int) -> list[tuple[int, int]]:
    coords = {(random.randrange(max_h), random.randrange(max_w))}
    while len(coords) < count:
        r, c = random.choice(tuple(coords))
        nbrs = [
            (r - 1, c),
            (r + 1, c),
            (r, c - 1),
            (r, c + 1),
        ]
        random.shuffle(nbrs)
        for nr, nc in nbrs:
            if 0 <= nr < max_h and 0 <= nc < max_w:
                coords.add((nr, nc))
                break
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return sorted((r - r0, c - c0) for r, c in coords)


def _gen_small_sprite(colors: list[int]) -> np.ndarray:
    count = random.randint(GEN_SPRITE_CELL_MIN, GEN_SPRITE_CELL_MAX)
    coords = _gen_connected_coords(GEN_LOCAL_SHAPE_MAX_HEIGHT, GEN_LOCAL_SHAPE_MAX_WIDTH, count)
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    sprite = np.zeros((h, w), dtype=int)
    if len(colors) == 1:
        for r, c in coords:
            sprite[r, c] = colors[0]
    else:
        shuffled = coords[:]
        random.shuffle(shuffled)
        split = random.randint(1, len(shuffled) - 1)
        for r, c in shuffled[:split]:
            sprite[r, c] = colors[0]
        for r, c in shuffled[split:]:
            sprite[r, c] = colors[1]
    return sprite


def _gen_paint_sprite(shape: tuple[int, int], sprite: np.ndarray, r0: int, c0: int) -> np.ndarray:
    frame = np.zeros(shape, dtype=int)
    paste(frame, sprite, r0, c0, transparent=GEN_BG_COLOR)
    return frame


def _gen_start_bounds(limit: int, size: int, delta: int, times: range) -> tuple[int, int]:
    """Return legal starting positions so a shifted local span stays in-bounds."""
    offsets = [int(t) * int(delta) for t in times]
    low = max(0, *[-offset for offset in offsets])
    high = min(limit - size, *[limit - size - offset for offset in offsets])
    return low, high


def _gen_motion_starts(height: int, width: int, local_h: int, local_w: int,
                       times: range) -> list[tuple[int, int, int, int]]:
    """Enumerate (dr, dc, r0, c0) in the original scan order."""
    starts = []
    for dr, dc in GEN_MOTION_DELTAS:
        r_min, r_max = _gen_start_bounds(height, local_h, dr, times)
        c_min, c_max = _gen_start_bounds(width, local_w, dc, times)
        for r0 in range(r_min, r_max + 1):
            for c0 in range(c_min, c_max + 1):
                starts.append((dr, dc, r0, c0))
    return starts


def _gen_coords_cover_origin(coords: np.ndarray, r0: int, c0: int) -> bool:
    return any(r0 + int(r) == 0 and c0 + int(c) == 0 for r, c in coords)


def _gen_model_a() -> tuple[list[np.ndarray], set[int]] | None:
    height = GEN_PANEL_HEIGHT
    width = random.randint(GEN_PANEL_WIDTH_MIN, GEN_PANEL_WIDTH_MAX)
    colors = random.sample(GEN_PANEL_COLORS, random.choice(GEN_PANEL_COLOR_COUNTS))
    sprite = _gen_small_sprite(colors)
    sh, sw = map(int, sprite.shape)
    sprite_coords = np.argwhere(sprite != GEN_BG_COLOR)

    placements = []
    for dr, dc, r0, c0 in _gen_motion_starts(height, width, sh, sw, GEN_TRANSLATION_TIMES):
        if not _gen_coords_cover_origin(sprite_coords, r0, c0):
            placements.append((dr, dc, r0, c0))
    if not placements:
        return None

    dr, dc, r0, c0 = random.choice(placements)
    frames = [
        _gen_paint_sprite((height, width), sprite, r0 + t * dr, c0 + t * dc)
        for t in GEN_TRANSLATION_TIMES
    ]
    return frames, set(colors)


def _gen_shifted_coords(
    local_coords: list[tuple[int, int]],
    r0: int,
    c0: int,
    dr: int,
    dc: int,
    t: int,
) -> list[tuple[int, int]]:
    return [(r0 + r + t * dr, c0 + c + t * dc) for r, c in local_coords]


def _gen_increment_is_disjoint(
    inc_coords: list[tuple[int, int]],
    r0: int,
    c0: int,
    dr: int,
    dc: int,
) -> bool:
    seen: set[tuple[int, int]] = set()
    for t in GEN_INCREMENT_TIMES:
        piece = _gen_shifted_coords(inc_coords, r0, c0, dr, dc, t)
        if any((r, c) in seen for r, c in piece):
            return False
        seen.update(piece)
    return True


def _gen_model_b() -> tuple[list[np.ndarray], set[int]] | None:
    height = GEN_PANEL_HEIGHT
    width = random.randint(GEN_PANEL_WIDTH_MIN, GEN_PANEL_WIDTH_MAX)
    color = random.choice(GEN_PANEL_COLORS)
    inc_count = random.randint(GEN_INCREMENT_CELL_MIN, GEN_INCREMENT_CELL_MAX)
    inc_coords = _gen_connected_coords(
        GEN_LOCAL_SHAPE_MAX_HEIGHT,
        GEN_LOCAL_SHAPE_MAX_WIDTH,
        inc_count,
    )
    inc_h = max(r for r, _ in inc_coords) + 1
    inc_w = max(c for _, c in inc_coords) + 1

    options = []
    for dr, dc, r0, c0 in _gen_motion_starts(height, width, inc_h, inc_w, GEN_INCREMENT_TIMES):
        if _gen_increment_is_disjoint(inc_coords, r0, c0, dr, dc):
            first_piece = _gen_shifted_coords(inc_coords, r0, c0, dr, dc, -1)
            if (0, 0) not in first_piece:
                options.append((dr, dc, r0, c0))
    if not options:
        return None

    dr, dc, r0, c0 = random.choice(options)
    frames = []
    occupied: set[tuple[int, int]] = set()
    for t in GEN_INCREMENT_TIMES:
        occupied.update(_gen_shifted_coords(inc_coords, r0, c0, dr, dc, t))
        frame = np.zeros((height, width), dtype=int)
        for r, c in occupied:
            frame[r, c] = color
        frames.append(frame)
    return frames, {color}


def _gen_rows_for_column(global_c: int, width: int) -> list[int]:
    rows = GEN_PREFIX_ROWS
    if global_c in (0, width - 1):
        rows = GEN_PREFIX_EDGE_ROWS
    size = random.randint(GEN_PREFIX_MIN_ROWS_PER_COLUMN, len(rows))
    return random.sample(rows, size)


def _gen_model_c() -> tuple[list[np.ndarray], set[int]] | None:
    height = GEN_PANEL_HEIGHT
    width = random.randint(GEN_PANEL_WIDTH_MIN, GEN_PANEL_WIDTH_MAX)
    step_choices = [GEN_PREFIX_STEP_CHOICES[0]]
    if width >= GEN_PREFIX_TWO_STEP_MIN_WIDTH:
        step_choices.append(GEN_PREFIX_STEP_CHOICES[1])
    step = random.choice(step_choices)
    min_span = GEN_PREFIX_ADVANCE_COUNT * step + 1
    if min_span > width:
        return None

    span = random.randint(min_span, width)
    c0 = random.randint(0, width - span)
    start_j = random.randint(0, span - 1 - GEN_PREFIX_ADVANCE_COUNT * step)
    fronts = [c0 + start_j + t * step for t in GEN_TRANSLATION_TIMES]
    mask = np.zeros((height, width), dtype=bool)
    for j in range(span):
        gc = c0 + j
        for r in _gen_rows_for_column(gc, width):
            mask[r, gc] = True

    band_counts = []
    for k in range(GEN_PREFIX_BAND_CHECK_COUNT):
        lo = c0 + start_j + k * step
        hi = c0 + start_j + (k + 1) * step
        band_counts.append(int(mask[:, lo + 1 : hi + 1].sum()))
    if len(set(band_counts)) == 1:
        target_c = c0 + start_j + step + 1
        if (
            target_c < width
            and not mask[GEN_PREFIX_VARIATION_ROW, target_c]
            and target_c not in (0, width - 1)
        ):
            mask[GEN_PREFIX_VARIATION_ROW, target_c] = True
        elif target_c < width and int(mask[:, target_c].sum()) > 1:
            rows = np.flatnonzero(mask[:, target_c])
            mask[int(rows[0]), target_c] = False

    used_rows = set(np.flatnonzero(mask.any(axis=1)).tolist())
    if len(used_rows) < 2:
        mask[GEN_PREFIX_REPAIR_ROW, c0 + span // 2] = True

    inc_color, static_color = random.sample(GEN_PANEL_COLORS, 2)
    cols = np.broadcast_to(np.arange(width), (height, width))
    frames = []
    for front in fronts:
        frame = np.zeros((height, width), dtype=int)
        active = mask & (cols <= front)
        frame[mask] = static_color
        frame[active] = inc_color
        frames.append(frame)

    if random.choice(GEN_PREFIX_FLIP_CHOICES):
        frames = [np.fliplr(frame).copy() for frame in frames]

    return frames, {inc_color, static_color}


def _gen_validate(frames: list[np.ndarray], panel_colors: set[int]) -> np.ndarray | None:
    if int(frames[0][0, 0]) != 0:
        return None
    sep_color = _gen_sep_color(panel_colors)
    inp = _gen_stack(frames[:4], sep_color)
    try:
        out = main(inp)
    except Exception:
        return None
    if np.array_equal(out, frames[4]):
        return inp
    return None


def _gen_attempt() -> np.ndarray | None:
    builder = random.choice([_gen_model_a, _gen_model_b, _gen_model_c])
    made = builder()
    if made is None:
        return None
    frames, panel_colors = made
    return _gen_validate(frames, panel_colors)


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid 351d6448 input")
