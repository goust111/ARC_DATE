from __future__ import annotations

import random

import numpy as np


BG = 0
RAY = 1
TRIGGER = 6

GRID_BOUNDS = (18, 24)
EMITTER_COUNT_OPTIONS = ((2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2))
H_DUPLICATE = "horizontal_duplicate"
V_DUPLICATE = "vertical_duplicate"
BOTH_DUPLICATE = "both_duplicate"
DUPLICATE_MODES = (H_DUPLICATE, V_DUPLICATE, BOTH_DUPLICATE)
MIN_GAP = 3
MAX_ATTEMPTS = 400


def _choose_spaced_positions(size: int, count: int, *, forbidden: set[int] | None = None) -> list[int]:
    forbidden = set() if forbidden is None else set(forbidden)
    candidates = [idx for idx in range(2, size - 1) if idx not in forbidden]
    for _ in range(200):
        picks = sorted(random.sample(candidates, count))
        if all(b - a >= MIN_GAP for a, b in zip(picks, picks[1:])):
            return picks
    raise RuntimeError("failed to choose spaced positions")


def _choose_trigger_axis_values(
    size: int,
    count: int,
    blocked: set[int],
    duplicate: bool,
) -> list[int]:
    values = [idx for idx in range(2, size - 1) if idx not in blocked]
    for _ in range(200):
        if duplicate:
            dup_size = 2 if count == 2 else random.choice((2, 2, min(3, count)))
            dup_value = random.choice(values)
            others = count - dup_size
            if others > 0:
                rest_candidates = [v for v in values if v != dup_value]
                rest = sorted(random.sample(rest_candidates, others))
                if any(b - a < 2 for a, b in zip(rest, rest[1:])):
                    continue
            else:
                rest = []
            picked = [dup_value] * dup_size + rest
        else:
            picked = sorted(random.sample(values, count))
            if any(b - a < 2 for a, b in zip(picked, picked[1:])):
                continue
        random.shuffle(picked)
        return picked
    raise RuntimeError("failed to assign trigger coordinates")


def _parse_input(grid: np.ndarray) -> tuple[list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    left_rows = [r for r in range(h) if int(g[r, 0]) == RAY]
    top_cols = [c for c in range(w) if int(g[0, c]) == RAY]
    triggers = [(int(r), int(c)) for r, c in np.argwhere(g == TRIGGER)]

    horizontal_triggers: list[tuple[int, int]] = []
    vertical_triggers: list[tuple[int, int]] = []
    left_row_set = set(left_rows)
    top_col_set = set(top_cols)
    for r, c in triggers:
        on_horizontal = r in left_row_set
        on_vertical = c in top_col_set
        if on_horizontal and not on_vertical:
            horizontal_triggers.append((r, c))
        elif on_vertical and not on_horizontal:
            vertical_triggers.append((r, c))
        else:
            raise ValueError("each trigger must align with exactly one initial emitter ray")
    return left_rows, top_cols, horizontal_triggers, vertical_triggers


def _trace_rays(
    shape: tuple[int, int],
    left_rows: list[int],
    top_cols: list[int],
    horizontal_triggers: list[tuple[int, int]],
    vertical_triggers: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape
    counts = np.zeros((h, w), dtype=int)
    initial_mask = np.zeros((h, w), dtype=bool)
    split_mask = np.zeros((h, w), dtype=bool)

    for r in left_rows:
        initial_mask[r, :] = True
        counts[r, :] += 1
    for c in top_cols:
        initial_mask[:, c] = True
        counts[:, c] += 1

    for r, c in horizontal_triggers:
        split_mask[:, c] = True
        counts[:, c] += 1
        counts[r, c] += 1
    for r, c in vertical_triggers:
        split_mask[r, :] = True
        counts[r, :] += 1
        counts[r, c] += 1

    return counts, initial_mask, split_mask


def _render_output(
    shape: tuple[int, int],
    left_rows: list[int],
    top_cols: list[int],
    horizontal_triggers: list[tuple[int, int]],
    vertical_triggers: list[tuple[int, int]],
    *,
    collapse_duplicate_splits: bool = False,
    keep_triggers_magenta: bool = True,
) -> np.ndarray:
    trigger_cells = set(horizontal_triggers) | set(vertical_triggers)
    if collapse_duplicate_splits:
        h, w = shape
        counts = np.zeros((h, w), dtype=int)
        initial_mask = np.zeros((h, w), dtype=bool)
        split_mask = np.zeros((h, w), dtype=bool)
        for r in left_rows:
            initial_mask[r, :] = True
            counts[r, :] += 1
        for c in top_cols:
            initial_mask[:, c] = True
            counts[:, c] += 1
        for c in {c for _r, c in horizontal_triggers}:
            split_mask[:, c] = True
            counts[:, c] += 1
        for r in {r for r, _c in vertical_triggers}:
            split_mask[r, :] = True
            counts[r, :] += 1
    else:
        counts, initial_mask, split_mask = _trace_rays(shape, left_rows, top_cols, horizontal_triggers, vertical_triggers)

    out = np.zeros(shape, dtype=int)
    out[initial_mask & ~split_mask] = RAY
    out[split_mask & ~initial_mask] = TRIGGER

    crossings = counts >= 2
    out[crossings] = np.where(counts[crossings] % 2 == 1, RAY, TRIGGER)

    if keep_triggers_magenta:
        for r, c in trigger_cells:
            out[r, c] = TRIGGER
    for r in left_rows:
        out[r, 0] = RAY
    for c in top_cols:
        out[0, c] = RAY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend border rays, split them at magenta triggers, and color the resulting weave by ray-count parity.

    Input structure:
    - Black background with blue emitters on the left edge and top edge.
    - Left-edge emitters define full horizontal initial rays; top-edge emitters define full vertical initial rays.
    - Internal magenta trigger pixels each lie on exactly one initial ray, never on an initial horizontal-vertical crossing.

    Rule:
    - Draw every left-edge emitter row across the grid in blue and every top-edge emitter column down the grid in blue.
    - At a trigger on a horizontal initial ray, emit two vertical magenta branch rays; at a trigger on a vertical initial ray, emit two horizontal magenta branch rays.
    - For every pixel touched by at least two rays, count all traversing rays with directional multiplicity: odd counts become blue, even counts become magenta.
    - Trigger pixels stay magenta, emitter pixels stay blue, and untouched cells remain black.

    Key insight:
    - The final pattern depends on both where branches are created and how many distinct rays overlap at each crossing, so duplicated branch lines produce visible odd-vs-even weave changes.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")
    left_rows, top_cols, horizontal_triggers, vertical_triggers = _parse_input(g)
    return _render_output(g.shape, left_rows, top_cols, horizontal_triggers, vertical_triggers)


def _build_candidate() -> np.ndarray:
    h = random.randint(*GRID_BOUNDS)
    w = random.randint(*GRID_BOUNDS)
    left_count, top_count = random.choice(EMITTER_COUNT_OPTIONS)
    duplicate_mode = random.choice(DUPLICATE_MODES)
    if duplicate_mode == BOTH_DUPLICATE and left_count == 2 and top_count == 2:
        duplicate_mode = random.choice((H_DUPLICATE, V_DUPLICATE))

    left_rows = _choose_spaced_positions(h, left_count)
    top_cols = _choose_spaced_positions(w, top_count)
    horizontal_cols = _choose_trigger_axis_values(
        w,
        left_count,
        set(top_cols),
        duplicate_mode in {H_DUPLICATE, BOTH_DUPLICATE},
    )
    vertical_rows = _choose_trigger_axis_values(
        h,
        top_count,
        set(left_rows),
        duplicate_mode in {V_DUPLICATE, BOTH_DUPLICATE},
    )

    grid = np.zeros((h, w), dtype=int)
    for r in left_rows:
        grid[r, 0] = RAY
    for c in top_cols:
        grid[0, c] = RAY
    for r, c in zip(left_rows, horizontal_cols):
        grid[r, c] = TRIGGER
    for r, c in zip(vertical_rows, top_cols):
        grid[r, c] = TRIGGER
    return grid


def _validate_scene(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if not set(map(int, np.unique(g))).issubset({BG, RAY, TRIGGER}):
        return False

    left_rows, top_cols, horizontal_triggers, vertical_triggers = _parse_input(g)
    trigger_cells = set(horizontal_triggers) | set(vertical_triggers)
    if not (2 <= len(left_rows) <= 4 and 2 <= len(top_cols) <= 4):
        return False
    if len(trigger_cells) != len(horizontal_triggers) + len(vertical_triggers):
        return False
    if not (4 <= len(trigger_cells) <= 6):
        return False
    if len(horizontal_triggers) != len(left_rows) or len(vertical_triggers) != len(top_cols):
        return False

    counts, initial_mask, split_mask = _trace_rays(g.shape, left_rows, top_cols, horizontal_triggers, vertical_triggers)
    out = _render_output(g.shape, left_rows, top_cols, horizontal_triggers, vertical_triggers)
    if out.shape != g.shape or not set(map(int, np.unique(out))).issubset({BG, RAY, TRIGGER}):
        return False

    left_row_set = set(left_rows)
    top_col_set = set(top_cols)
    for r, c in trigger_cells:
        if r == 0 or c == 0 or r == g.shape[0] - 1 or c == g.shape[1] - 1:
            return False
        if (r in left_row_set) == (c in top_col_set):
            return False

    if np.count_nonzero(initial_mask) == 0 or np.count_nonzero(split_mask) == 0:
        return False
    if np.count_nonzero(counts >= 2) < 6:
        return False

    odd_visible = [
        (r, c)
        for r in range(g.shape[0])
        for c in range(g.shape[1])
        if counts[r, c] >= 2 and counts[r, c] % 2 == 1 and (r, c) not in trigger_cells
    ]
    if len(odd_visible) < 2:
        return False
    if not any(counts[r, c] % 2 == 1 for r, c in trigger_cells):
        return False

    if np.count_nonzero(out) > int(g.size * 0.55):
        return False

    initial_only = np.zeros_like(g)
    for r in left_rows:
        initial_only[r, :] = RAY
    for c in top_cols:
        initial_only[:, c] = RAY
    for r, c in trigger_cells:
        initial_only[r, c] = TRIGGER
    for r in left_rows:
        initial_only[r, 0] = RAY
    for c in top_cols:
        initial_only[0, c] = RAY
    if np.array_equal(out, initial_only):
        return False

    no_trigger_override = _render_output(
        g.shape,
        left_rows,
        top_cols,
        horizontal_triggers,
        vertical_triggers,
        keep_triggers_magenta=False,
    )
    if np.array_equal(out, no_trigger_override):
        return False

    unique_split_shortcut = _render_output(
        g.shape,
        left_rows,
        top_cols,
        horizontal_triggers,
        vertical_triggers,
        collapse_duplicate_splits=True,
    )
    if np.array_equal(out, unique_split_shortcut):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        candidate = _build_candidate()
        if _validate_scene(candidate):
            return candidate
    raise RuntimeError("failed to generate a valid ray-weave input")
