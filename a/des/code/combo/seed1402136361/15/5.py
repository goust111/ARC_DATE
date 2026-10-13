import random
from typing import Iterable

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Learn legend transformations from the left panel and apply them to the right panel."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _mode_color(g)
    _, sep_cols = _find_separator_band(g, bg)
    left = g[:, : sep_cols[0]]
    right = g[:, sep_cols[-1] + 1 :]
    assert left.size > 0 and right.size > 0, "separator must split legend and main areas"

    rules = _extract_rules(left, bg)
    out = right.copy()
    occupied = np.zeros_like(right, dtype=bool)

    ordered_rules = sorted(
        rules,
        key=lambda pair: pair[0].shape[0] * pair[0].shape[1],
        reverse=True,
    )
    for before, after in ordered_rules:
        h, w = before.shape
        for r in range(right.shape[0] - h + 1):
            for c in range(right.shape[1] - w + 1):
                if occupied[r : r + h, c : c + w].any():
                    continue
                if np.array_equal(right[r : r + h, c : c + w], before):
                    occupied[r : r + h, c : c + w] = True
                    out[r : r + h, c : c + w] = after

    return out


# --------------- helpers ---------------

COLOR_POOL = list(range(10))
PATTERN_SIZE_CHOICES = (3, 4, 5)
RULE_COUNT_CHOICES = (2, 3)
LEGEND_SIDE_PAD_CHOICES = (1, 2)
LEGEND_TOP_BOTTOM_PAD_CHOICES = (1, 2)
LEGEND_BETWEEN_PAIR_GAP_CHOICES = (2, 3)
SEPARATOR_WIDTH_CHOICES = (1, 2)
SLOT_PAD_CHOICES = (1, 2)
SLOT_ROWS_CHOICES = (2, 3, 4)
EXTRA_MATCH_CHOICES = (0, 1, 2, 3)
DISTRACTOR_COUNT_CHOICES = (0, 1, 2)
RULE_COLOR_COUNT_CHOICES = (2, 3, 4)
GENERATION_RETRIES = 400


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _row_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = None
    for idx, present in enumerate(mask.tolist()):
        if present and start is None:
            start = idx
        elif not present and start is not None:
            spans.append((start, idx - 1))
            start = None
    if start is not None:
        spans.append((start, len(mask) - 1))
    return spans


def _find_separator_band(grid: np.ndarray, bg: int) -> tuple[int, list[int]]:
    h, w = grid.shape
    candidates: list[tuple[int, list[int]]] = []
    c = 0
    while c < w:
        col = grid[:, c]
        if np.all(col == col[0]) and int(col[0]) != bg:
            color = int(col[0])
            start = c
            c += 1
            while c < w and np.all(grid[:, c] == color):
                c += 1
            cols = list(range(start, c))
            left = grid[:, :start]
            right = grid[:, c:]
            if left.size > 0 and right.size > 0 and np.any(left != bg) and np.any(right != bg):
                candidates.append((color, cols))
        else:
            c += 1
    assert len(candidates) == 1, "expected exactly one full-height separator band"
    return candidates[0]


def _extract_rules(left: np.ndarray, bg: int) -> list[tuple[np.ndarray, np.ndarray]]:
    occupied_rows = np.any(left != bg, axis=1)
    spans = _row_spans(occupied_rows)
    assert spans and len(spans) % 2 == 0, "legend must contain an even number of occupied row blocks"

    rules: list[tuple[np.ndarray, np.ndarray]] = []
    seen = set()
    for i in range(0, len(spans), 2):
        top = spans[i]
        bottom = spans[i + 1]
        top_h = top[1] - top[0] + 1
        bottom_h = bottom[1] - bottom[0] + 1
        assert top_h == bottom_h, "paired legend patterns must share the same height"
        assert bottom[0] - top[1] == 2, "paired legend patterns must be separated by one background row"

        top_cols = np.where(np.any(left[top[0] : top[1] + 1] != bg, axis=0))[0]
        bottom_cols = np.where(np.any(left[bottom[0] : bottom[1] + 1] != bg, axis=0))[0]
        assert top_cols.size > 0 and bottom_cols.size > 0
        c0 = int(min(top_cols[0], bottom_cols[0]))
        c1 = int(max(top_cols[-1], bottom_cols[-1]))

        before = left[top[0] : top[1] + 1, c0 : c1 + 1].copy()
        after = left[bottom[0] : bottom[1] + 1, c0 : c1 + 1].copy()
        assert before.shape == after.shape
        key = before.tobytes()
        assert key not in seen, "legend input patterns must be unique"
        seen.add(key)
        rules.append((before, after))

    return rules


def _available_colors(excluded: Iterable[int]) -> list[int]:
    blocked = set(int(x) for x in excluded)
    return [color for color in COLOR_POOL if color not in blocked]


def _random_pattern(
    rng: random.Random,
    h: int,
    w: int,
    bg: int,
    palette: list[int],
) -> np.ndarray:
    pattern = np.full((h, w), bg, dtype=int)
    border_choices = palette * 2
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                pattern[r, c] = rng.choice(border_choices)
            else:
                choices = [bg, bg] + palette
                pattern[r, c] = rng.choice(choices)

    if h > 2 and w > 2 and np.all(pattern[1:-1, 1:-1] == bg):
        rr = rng.randrange(1, h - 1)
        cc = rng.randrange(1, w - 1)
        pattern[rr, cc] = rng.choice(palette)

    return pattern


def _transform_pattern(
    rng: random.Random,
    before: np.ndarray,
    bg: int,
    palette: list[int],
) -> np.ndarray:
    h, w = before.shape
    non_bg_colors = [int(c) for c in np.unique(before) if int(c) != bg]
    branches = ["recolor"]
    if len(non_bg_colors) >= 2:
        branches.append("swap")
    if h > 2 and w > 2 and np.any(before[1:-1, 1:-1] == bg):
        branches.append("fill")

    for _ in range(30):
        branch = rng.choice(branches)
        after = before.copy()
        if branch == "recolor":
            src = rng.choice(non_bg_colors)
            targets = [color for color in palette if color != src]
            if not targets:
                continue
            tgt = rng.choice(targets)
            after[before == src] = tgt
        elif branch == "swap":
            a, b = rng.sample(non_bg_colors, 2)
            mask_a = before == a
            mask_b = before == b
            after[mask_a] = b
            after[mask_b] = a
        else:
            holes = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1) if before[r, c] == bg]
            if not holes:
                continue
            count = rng.randint(1, max(1, len(holes) // 2))
            fill_color = rng.choice(palette)
            for r, c in rng.sample(holes, count):
                after[r, c] = fill_color

        if not np.array_equal(after, before):
            return after

    raise RuntimeError("failed to build a non-trivial transformed pattern")


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch


def _gen_rule_pair(
    rng: random.Random,
    h: int,
    w: int,
    bg: int,
    sep: int,
    used_before: set[bytes],
) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(80):
        color_count = rng.choice(RULE_COLOR_COUNT_CHOICES)
        pool = _available_colors((bg, sep))
        palette = rng.sample(pool, color_count)
        before = _random_pattern(rng, h, w, bg, palette)
        key = before.tobytes()
        if key in used_before:
            continue
        after = _transform_pattern(rng, before, bg, palette)
        used_before.add(key)
        return before, after
    return None


def _gen_distractor(
    rng: random.Random,
    h: int,
    w: int,
    bg: int,
    sep: int,
    forbidden: set[bytes],
) -> np.ndarray | None:
    for _ in range(80):
        color_count = rng.choice(RULE_COLOR_COUNT_CHOICES)
        pool = _available_colors((bg, sep))
        palette = rng.sample(pool, color_count)
        pattern = _random_pattern(rng, h, w, bg, palette)
        key = pattern.tobytes()
        if key in forbidden:
            continue
        forbidden.add(key)
        return pattern
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    bg, sep = rng.sample(COLOR_POOL, 2)
    rule_count = rng.choice(RULE_COUNT_CHOICES)
    h = rng.choice(PATTERN_SIZE_CHOICES)
    w = rng.choice(PATTERN_SIZE_CHOICES)
    legend_side_pad = rng.choice(LEGEND_SIDE_PAD_CHOICES)
    legend_top_pad = rng.choice(LEGEND_TOP_BOTTOM_PAD_CHOICES)
    legend_bottom_pad = rng.choice(LEGEND_TOP_BOTTOM_PAD_CHOICES)
    pair_gap = rng.choice(LEGEND_BETWEEN_PAIR_GAP_CHOICES)
    sep_width = rng.choice(SEPARATOR_WIDTH_CHOICES)
    slot_pad = rng.choice(SLOT_PAD_CHOICES)

    rules: list[tuple[np.ndarray, np.ndarray]] = []
    used_before: set[bytes] = set()
    for _ in range(rule_count):
        pair = _gen_rule_pair(rng, h, w, bg, sep, used_before)
        if pair is None:
            return None
        rules.append(pair)

    extra_matches = rng.choice(EXTRA_MATCH_CHOICES)
    matched_rule_ids = list(range(rule_count))
    matched_rule_ids.extend(rng.randrange(rule_count) for _ in range(extra_matches))
    rng.shuffle(matched_rule_ids)

    distractor_count = rng.choice(DISTRACTOR_COUNT_CHOICES)
    total_objects = len(matched_rule_ids) + distractor_count

    slot_rows = rng.choice(SLOT_ROWS_CHOICES)
    min_slot_cols = max(3, (total_objects + slot_rows - 1) // slot_rows)
    max_slot_cols = max(min_slot_cols, min_slot_cols + 2)
    slot_cols = rng.randint(min_slot_cols, max_slot_cols)

    legend_block_h = 2 * h + 1
    legend_h = legend_top_pad + rule_count * legend_block_h + (rule_count - 1) * pair_gap + legend_bottom_pad
    legend_w = w + 2 * legend_side_pad
    main_h = slot_rows * h + (slot_rows + 1) * slot_pad
    main_w = slot_cols * w + (slot_cols + 1) * slot_pad

    canvas_h = max(legend_h, main_h)
    canvas_w = legend_w + sep_width + main_w
    grid = np.full((canvas_h, canvas_w), bg, dtype=int)
    right = np.full((canvas_h, main_w), bg, dtype=int)
    expected = right.copy()

    legend_top = rng.randint(0, canvas_h - legend_h)
    legend_left = legend_side_pad
    y = legend_top + legend_top_pad
    for idx, (before, after) in enumerate(rules):
        _place_patch(grid, before, y, legend_left)
        y += h + 1
        _place_patch(grid, after, y, legend_left)
        y += h
        if idx != rule_count - 1:
            y += pair_gap

    grid[:, legend_w : legend_w + sep_width] = sep

    all_slots = [(r, c) for r in range(slot_rows) for c in range(slot_cols)]
    if total_objects > len(all_slots):
        return None
    chosen_slots = rng.sample(all_slots, total_objects)

    main_top = rng.randint(0, canvas_h - main_h)
    slot_base_r = main_top + slot_pad
    slot_base_c = slot_pad
    shift_cap = slot_pad - 1

    forbidden = set(used_before)
    objects: list[tuple[np.ndarray, np.ndarray]] = []
    for rule_id in matched_rule_ids:
        before, after = rules[rule_id]
        objects.append((before, after))
    for _ in range(distractor_count):
        pattern = _gen_distractor(rng, h, w, bg, sep, forbidden)
        if pattern is None:
            return None
        objects.append((pattern, pattern))
    rng.shuffle(objects)

    for (slot_r, slot_c), (before, after) in zip(chosen_slots, objects):
        shift_r = rng.randint(0, shift_cap) if shift_cap > 0 else 0
        shift_c = rng.randint(0, shift_cap) if shift_cap > 0 else 0
        top = slot_base_r + slot_r * (h + slot_pad) + shift_r
        left = slot_base_c + slot_c * (w + slot_pad) + shift_c
        _place_patch(right, before, top, left)
        _place_patch(expected, after, top, left)

    grid[:, legend_w + sep_width :] = right

    try:
        result = main(grid)
    except AssertionError:
        return None

    if result.shape != expected.shape:
        return None
    if not np.array_equal(result, expected):
        return None
    if np.array_equal(result, right):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-rewrite ARC input")
