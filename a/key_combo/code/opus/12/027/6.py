from __future__ import annotations

from collections import deque
from itertools import product
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
TEAL = 8
MAROON = 9

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

MATCH_PATTERNS = [
    (2, 2),
    (2, 3),
    (3, 2),
]

DISTRACTOR_PATTERNS = [
    (3, 3),
    (2, 4),
    (4, 2),
    (4,),
    (5,),
    (3, 5),
    (5, 3),
    (2, 2, 2),
    (2, 3, 2),
    (3, 2, 2),
]

MAX_GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    h, w = shape
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = np.asarray(grid == int(color), dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, mask.shape):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _normalized_mask(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _pattern_shape_key(pattern: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    width = max(pattern)
    coords = []
    for r, run in enumerate(pattern):
        start = width - int(run)
        for c in range(start, width):
            coords.append((r, c))
    return tuple(sorted(coords))


def _parse_strip_row(row: np.ndarray) -> dict | None:
    vals = [int(v) for v in row]
    if any(v == TEAL for v in vals):
        return None

    i = 0
    while i < len(vals) and vals[i] == MAROON:
        i += 1
    blue_start = i
    while i < len(vals) and vals[i] == BLUE:
        i += 1
    blue_end = i - 1
    while i < len(vals) and vals[i] == RED:
        i += 1
    red_end = i - 1

    if blue_start > blue_end or blue_end >= red_end:
        return None
    if any(v != BLACK for v in vals[i:]):
        return None

    return {
        "blue_start": int(blue_start),
        "blue_end": int(blue_end),
        "blue_width": int(blue_end - blue_start + 1),
        "red_count": int(red_end - blue_end),
    }


def _align_rows(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, dict]], int]:
    g = np.asarray(grid, dtype=int)
    parsed_rows: list[tuple[int, dict]] = []
    for r in range(g.shape[0]):
        parsed = _parse_strip_row(g[r])
        if parsed is not None:
            parsed_rows.append((r, parsed))

    assert parsed_rows, "expected at least one strip row"
    c_min = min(parsed["blue_end"] for _, parsed in parsed_rows)

    out = g.copy()
    for r, parsed in parsed_rows:
        blue_width = int(parsed["blue_width"])
        red_count = int(parsed["red_count"])
        new_start = max(0, c_min - blue_width + 1)

        out[r, :] = BLACK
        out[r, :new_start] = MAROON
        out[r, new_start : new_start + blue_width] = BLUE
        out[r, new_start + blue_width : new_start + blue_width + red_count] = RED

    return out, parsed_rows, int(c_min)


def _recolor_matching_components(aligned: np.ndarray) -> np.ndarray:
    teal_components = _components_of_color(aligned, TEAL)
    assert len(teal_components) == 1, "expected exactly one teal template object"
    template_shape = _normalized_mask(teal_components[0])

    out = aligned.copy()
    for comp in _components_of_color(aligned, BLUE):
        if _normalized_mask(comp) == template_shape:
            out[comp[:, 0], comp[:, 1]] = TEAL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align each row's blue run by its right edge, then recolor the emergent blue components that match the teal template.

    Input structure:
    - A black background contains several horizontal strip rows made of maroon, blue, and red runs.
    - Different rows end their blue runs at different columns, so the blue pixels do not initially form the target shapes.
    - One separate teal 4-connected object acts as a shape template and does not overlap the strip rows.

    Rule:
    - Detect every strip row as [maroon*][blue+][red+][black*] and find the minimum original right edge of the blue run.
    - Slide each blue run so its right edge aligns to that minimum column; if a run is too wide, clamp its left edge to column 0.
    - Keep each row's blue width and red suffix length, refill the cells before the aligned blue run with maroon, and keep all non-strip rows unchanged.
    - On the aligned grid, find the 4-connected blue components and recolor every component whose normalized mask matches the teal template mask.

    Key insight:
    - The teal template is matched only after the row alignment, because the alignment changes vertical adjacency and creates blue components that do not exist in the original layout.
    """
    aligned, _parsed_rows, _c_min = _align_rows(np.asarray(grid, dtype=int))
    return _recolor_matching_components(aligned)


def _template_coords(pattern: tuple[int, ...], top: int, left: int) -> list[tuple[int, int]]:
    width = max(pattern)
    coords = []
    for r, run in enumerate(pattern):
        start = width - int(run)
        for c in range(start, width):
            coords.append((top + r, left + c))
    return coords


def _component_shape_counts(grid: np.ndarray, color: int) -> dict[tuple[tuple[int, int], ...], int]:
    counts: dict[tuple[tuple[int, int], ...], int] = {}
    for comp in _components_of_color(grid, color):
        key = _normalized_mask(comp)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _aligned_match_count(grid: np.ndarray) -> tuple[int, int, int]:
    aligned, parsed_rows, c_min = _align_rows(grid)
    teal_components = _components_of_color(aligned, TEAL)
    assert len(teal_components) == 1
    template_shape = _normalized_mask(teal_components[0])

    match_count = 0
    total_blue_components = 0
    for comp in _components_of_color(aligned, BLUE):
        total_blue_components += 1
        if _normalized_mask(comp) == template_shape:
            match_count += 1

    pre_counts = _component_shape_counts(grid, BLUE)
    assert pre_counts.get(template_shape, 0) == 0, "template shape must not already exist before alignment"
    original_ends = {parsed["blue_end"] for _, parsed in parsed_rows}
    assert len(original_ends) >= 2, "blue right edges must vary across rows"
    return match_count, total_blue_components, c_min


def _choose_distractors(
    template_pattern: tuple[int, ...],
    c_min: int,
    count: int,
    include_clamp: bool,
) -> list[tuple[int, ...]]:
    template_key = _pattern_shape_key(template_pattern)
    candidates = []
    clamp_candidates = []
    for pattern in DISTRACTOR_PATTERNS:
        key = _pattern_shape_key(pattern)
        if key == template_key:
            continue
        candidates.append(pattern)
        if max(pattern) > c_min + 1:
            clamp_candidates.append(pattern)

    chosen: list[tuple[int, ...]] = []
    if include_clamp:
        random.shuffle(clamp_candidates)
        if not clamp_candidates:
            raise RuntimeError("no clamp-capable distractor available")
        chosen.append(clamp_candidates[0])

    remaining = [pattern for pattern in candidates if pattern not in chosen]
    random.shuffle(remaining)
    chosen.extend(remaining[: max(0, count - len(chosen))])

    if len(chosen) != count:
        raise RuntimeError("failed to choose enough distractors")
    return chosen


def _row_shift_choices(width: int, c_min: int) -> list[int]:
    new_start = max(0, c_min - int(width) + 1)
    max_shift = 3 - new_start
    return list(range(max_shift + 1))


def _sample_shifts(pattern: tuple[int, ...], c_min: int, require_variation: bool, require_zero: bool) -> tuple[int, ...]:
    choices = [_row_shift_choices(width, c_min) for width in pattern]
    all_combos = list(product(*choices))
    valid = []
    for combo in all_combos:
        if require_variation and len(set(combo)) == 1:
            continue
        if require_zero and 0 not in combo:
            continue
        valid.append(combo)
    if not valid:
        raise RuntimeError("no valid shift assignment")
    return tuple(random.choice(valid))


def _build_candidate() -> np.ndarray:
    c_min = random.choice([2, 3])
    template_pattern = random.choice(MATCH_PATTERNS)
    match_count = random.choice([2, 3])
    distractor_count = random.choice([1, 2])
    include_clamp = random.random() < 0.65
    distractors = _choose_distractors(template_pattern, c_min, distractor_count, include_clamp)

    groups = [template_pattern] * match_count + distractors
    random.shuffle(groups)

    group_specs = []
    anchor_used = False
    template_groups_to_place = match_count
    for pattern in groups:
        is_match = pattern == template_pattern and template_groups_to_place > 0
        if is_match:
            shifts = _sample_shifts(
                pattern,
                c_min,
                require_variation=True,
                require_zero=not anchor_used,
            )
            anchor_used = anchor_used or (0 in shifts)
            template_groups_to_place -= 1
        else:
            shifts = _sample_shifts(pattern, c_min, require_variation=False, require_zero=False)
        reds = tuple(random.randint(1, 4) for _ in pattern)
        group_specs.append({"pattern": pattern, "shifts": shifts, "reds": reds, "is_match": is_match})

    assert anchor_used

    row_records = []
    row_cursor = random.choice([0, 1])
    group_tops = []
    for group in group_specs:
        group_tops.append(row_cursor)
        for width, shift, red_count in zip(group["pattern"], group["shifts"], group["reds"]):
            new_start = max(0, c_min - int(width) + 1)
            old_start = new_start + int(shift)
            row_records.append(
                {
                    "row": row_cursor,
                    "width": int(width),
                    "old_start": int(old_start),
                    "red_count": int(red_count),
                }
            )
            row_cursor += 1
        row_cursor += 1

    template_h = len(template_pattern)
    template_w = max(template_pattern)
    top_for_template = row_cursor + 1
    h = top_for_template + template_h + random.choice([0, 1])

    max_used_col = 0
    for rec in row_records:
        max_used_col = max(max_used_col, rec["old_start"] + rec["width"] + rec["red_count"])
    w_min = max(13, max_used_col + 2, template_w + 8)
    w = random.randint(w_min, min(18, w_min + 3))

    grid = np.zeros((h, w), dtype=int)
    for rec in row_records:
        r = rec["row"]
        old_start = rec["old_start"]
        width = rec["width"]
        red_count = rec["red_count"]
        grid[r, :old_start] = MAROON
        grid[r, old_start : old_start + width] = BLUE
        grid[r, old_start + width : old_start + width + red_count] = RED

    min_template_left = max(max_used_col + 1, w - template_w - 2)
    max_template_left = w - template_w
    if min_template_left > max_template_left:
        return None
    template_left = random.randint(min_template_left, max_template_left)
    for r, c in _template_coords(template_pattern, top_for_template, template_left):
        grid[r, c] = TEAL

    try:
        match_count_seen, blue_component_count, computed_c_min = _aligned_match_count(grid)
    except AssertionError:
        return None
    if computed_c_min != c_min:
        return None
    if match_count_seen != match_count:
        return None
    if blue_component_count <= match_count_seen:
        return None
    if include_clamp and not any(rec["width"] > c_min + 1 for rec in row_records):
        return None

    aligned, parsed_rows, _ = _align_rows(grid)
    if len(parsed_rows) != len(row_records):
        return None
    if len(_components_of_color(grid, TEAL)) != 1:
        return None
    if len(_components_of_color(aligned, TEAL)) != 1:
        return None

    out = main(grid)
    recolored_components = len(_components_of_color(out, TEAL))
    if recolored_components < 1 + match_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
