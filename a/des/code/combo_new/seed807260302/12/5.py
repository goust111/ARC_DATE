from __future__ import annotations

import random

import numpy as np


BG = 0
SEP_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400

SHAPE_KINDS = (
    "rect_fill",
    "rect_frame",
    "line_h",
    "line_v",
    "diag_main",
    "diag_anti",
    "plus",
    "el",
)


def _parse_rule_row(row: np.ndarray) -> tuple[int, int] | None:
    nz = np.flatnonzero(row != BG)
    if nz.size == 0:
        return None
    assert nz.size == 2, "each rule row must contain exactly two non-background cells"
    assert int(nz[1]) == int(nz[0]) + 1, "rule pixels must form one 2-cell horizontal bar"
    src = int(row[nz[0]])
    dst = int(row[nz[1]])
    assert src != dst, "rule source and destination colors must differ"
    return src, dst


def _extract_rules(panel: np.ndarray) -> list[tuple[int, int]]:
    rules: list[tuple[int, int]] = []
    for row in panel:
        rule = _parse_rule_row(row)
        if rule is None:
            assert np.all(row == BG), "non-rule rows must be blank"
            continue
        rules.append(rule)
    assert rules, "right panel must contain at least one replacement rule"
    return rules


def _separator_col(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates: list[int] = []
    for c in range(1, w - 1):
        col = grid[:, c]
        if int(col[0]) == BG or not np.all(col == col[0]):
            continue
        left = grid[:, :c]
        right = grid[:, c + 1 :]
        if left.size == 0 or right.size == 0 or not np.any(left != BG):
            continue
        try:
            _extract_rules(right)
        except AssertionError:
            continue
        candidates.append(c)
    assert len(candidates) == 1, "expected a unique full-height separator column"
    return candidates[0]


def _apply_rules(left: np.ndarray, rules: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(left, copy=True)
    for src, dst in rules:
        out[out == src] = dst
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decrypt the left pattern by applying the ordered color-replacement bars shown on the right.

    Input structure:
    - A single full-height vertical separator column splits the grid into a left picture and a right rule panel.
    - The left side is a black-background multicolor pattern using several non-black colors.
    - The right side is a black-background list of 2-cell horizontal color bars, one rule per occupied row.

    Rule:
    - Identify each occupied row in the right panel as a replacement rule "left pixel color -> right pixel color".
    - Read those rules from top to bottom and apply them to the left picture one after another, in place.
    - Return only the transformed left picture, removing the separator column and the rule panel.

    Key insight:
    - The rule bars form a sequential substitution chain: later rules must see colors created by earlier rules.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _separator_col(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    rules = _extract_rules(right)
    return _apply_rules(left, rules)


def _required_initial_colors(rules: list[tuple[int, int]]) -> list[int]:
    produced_before: set[int] = set()
    required: list[int] = []
    for src, dst in rules:
        if src not in produced_before:
            required.append(src)
        produced_before.add(dst)
    return required


def _apply_parallel_once(left: np.ndarray, rules: list[tuple[int, int]]) -> np.ndarray:
    original = np.array(left, copy=True)
    out = np.array(left, copy=True)
    for src, dst in rules:
        out[original == src] = dst
    return out


def _shape_coords(kind: str, h: int, w: int, rng: random.Random) -> list[tuple[int, int]] | None:
    if kind == "rect_fill":
        sh = rng.randint(2, min(5, h - 2))
        sw = rng.randint(2, min(5, w - 2))
        r0 = rng.randint(0, h - sh)
        c0 = rng.randint(0, w - sw)
        return [(r, c) for r in range(r0, r0 + sh) for c in range(c0, c0 + sw)]

    if kind == "rect_frame":
        sh = rng.randint(3, min(6, h - 2))
        sw = rng.randint(3, min(6, w - 2))
        r0 = rng.randint(0, h - sh)
        c0 = rng.randint(0, w - sw)
        coords = []
        for r in range(r0, r0 + sh):
            for c in range(c0, c0 + sw):
                if r in (r0, r0 + sh - 1) or c in (c0, c0 + sw - 1):
                    coords.append((r, c))
        return coords

    if kind == "line_h":
        length = rng.randint(3, min(7, w))
        r = rng.randrange(h)
        c0 = rng.randint(0, w - length)
        return [(r, c) for c in range(c0, c0 + length)]

    if kind == "line_v":
        length = rng.randint(3, min(7, h - 1))
        c = rng.randrange(w)
        r0 = rng.randint(0, h - length)
        return [(r, c) for r in range(r0, r0 + length)]

    if kind == "diag_main":
        length = rng.randint(3, min(6, h, w))
        r0 = rng.randint(0, h - length)
        c0 = rng.randint(0, w - length)
        return [(r0 + i, c0 + i) for i in range(length)]

    if kind == "diag_anti":
        length = rng.randint(3, min(6, h, w))
        r0 = rng.randint(0, h - length)
        c0 = rng.randint(length - 1, w - 1)
        return [(r0 + i, c0 - i) for i in range(length)]

    if kind == "plus":
        arm = rng.randint(1, min(2, (h - 3) // 2, (w - 3) // 2))
        r = rng.randint(arm, h - arm - 1)
        c = rng.randint(arm, w - arm - 1)
        coords = {(r, c)}
        for d in range(1, arm + 1):
            coords.add((r - d, c))
            coords.add((r + d, c))
            coords.add((r, c - d))
            coords.add((r, c + d))
        return list(coords)

    if kind == "el":
        vr = rng.randint(3, min(6, h - 1))
        hc = rng.randint(3, min(6, w - 1))
        orient = rng.randrange(4)
        if orient == 0:
            r0 = rng.randint(0, h - vr)
            c0 = rng.randint(0, w - hc)
            return [(r0 + i, c0) for i in range(vr)] + [(r0 + vr - 1, c0 + j) for j in range(hc)]
        if orient == 1:
            r0 = rng.randint(0, h - vr)
            c0 = rng.randint(hc - 1, w - 1)
            return [(r0 + i, c0) for i in range(vr)] + [(r0 + vr - 1, c0 - j) for j in range(hc)]
        if orient == 2:
            r0 = rng.randint(vr - 1, h - 1)
            c0 = rng.randint(0, w - hc)
            return [(r0 - i, c0) for i in range(vr)] + [(r0 - vr + 1, c0 + j) for j in range(hc)]
        r0 = rng.randint(vr - 1, h - 1)
        c0 = rng.randint(hc - 1, w - 1)
        return [(r0 - i, c0) for i in range(vr)] + [(r0 - vr + 1, c0 - j) for j in range(hc)]

    return None


def _place_shape(grid: np.ndarray, color: int, rng: random.Random, preferred: str | None = None) -> bool:
    h, w = grid.shape
    kinds = [preferred] if preferred is not None else []
    kinds += [kind for kind in SHAPE_KINDS if kind != preferred]
    for _ in range(50):
        kind = rng.choice(kinds)
        coords = _shape_coords(kind, h, w, rng)
        if not coords:
            continue
        coords = list(dict.fromkeys(coords))
        if len(coords) < 3:
            continue
        if any(grid[r, c] != BG for r, c in coords):
            continue
        for r, c in coords:
            grid[r, c] = color
        return True
    return False


def _build_left_pattern(height: int, width: int, active_colors: list[int], emphasis_color: int, rng: random.Random) -> np.ndarray | None:
    grid = np.full((height, width), BG, dtype=int)
    color_order = active_colors[:]
    rng.shuffle(color_order)

    for color in color_order:
        if not _place_shape(grid, color, rng):
            return None

    if int(np.sum(grid == emphasis_color)) < 5 and not _place_shape(grid, emphasis_color, rng):
        return None

    extra_shapes = rng.randint(len(active_colors), len(active_colors) + 4)
    for _ in range(extra_shapes):
        _place_shape(grid, rng.choice(active_colors), rng)

    for color in active_colors:
        if int(np.sum(grid == color)) < 3 and not _place_shape(grid, color, rng):
            return None

    density = float(np.mean(grid != BG))
    present = sorted({int(v) for v in grid[grid != BG]})
    if density < 0.18 or density > 0.62:
        return None
    if present != sorted(active_colors):
        return None
    return grid


def _sample_rules(rng: random.Random) -> tuple[list[tuple[int, int]], list[int], list[int]] | None:
    num_rules = rng.randint(3, 5)
    chain_len = rng.randint(2, min(4, num_rules))
    extra_count = num_rules - chain_len

    min_support = max(5, chain_len + extra_count)
    if min_support > len(PALETTE):
        return None
    support_size = rng.randint(min_support, min(len(PALETTE), min_support + 2))
    support = rng.sample(PALETTE, support_size)

    chain_colors = rng.sample(support, chain_len + 1)
    chain_rules = list(zip(chain_colors, chain_colors[1:]))

    available_extra_sources = [c for c in support if c not in chain_colors[:-1]]
    if len(available_extra_sources) < extra_count:
        return None
    extra_sources = rng.sample(available_extra_sources, extra_count)

    extra_rules: list[tuple[int, int]] = []
    for src in extra_sources:
        target_choices = [c for c in support if c != src]
        extra_rules.append((src, rng.choice(target_choices)))

    chain_positions = set(rng.sample(range(num_rules), chain_len))
    ordered_rules: list[tuple[int, int]] = []
    chain_idx = 0
    extra_idx = 0
    for i in range(num_rules):
        if i in chain_positions:
            ordered_rules.append(chain_rules[chain_idx])
            chain_idx += 1
        else:
            ordered_rules.append(extra_rules[extra_idx])
            extra_idx += 1

    required_initial = _required_initial_colors(ordered_rules)
    if len(required_initial) > 5:
        return None
    return ordered_rules, support, chain_colors


def _choose_active_colors(
    support: list[int],
    rules: list[tuple[int, int]],
    chain_colors: list[int],
    rng: random.Random,
) -> list[int] | None:
    required = list(dict.fromkeys(_required_initial_colors(rules)))
    if len(required) > 5:
        return None

    active = set(required)
    chain_middle = [c for c in chain_colors[1:-1] if c not in active]
    mixed_mode = bool(chain_middle and rng.random() < 0.5)
    if mixed_mode:
        active.add(rng.choice(chain_middle))

    max_active = min(5, len(support))
    min_active = max(3, len(active))
    if min_active > max_active:
        return None
    active_count = rng.randint(min_active, max_active)

    banned = set(chain_middle) if not mixed_mode else set()
    choices = [c for c in support if c not in active and c not in banned]
    if len(choices) < active_count - len(active):
        choices = [c for c in support if c not in active]
    if len(choices) < active_count - len(active):
        return None
    active.update(rng.sample(choices, active_count - len(active)))
    return sorted(active)


def _sample_rule_rows(height: int, count: int, rng: random.Random) -> list[int] | None:
    if rng.random() < 0.5:
        offset = rng.randint(0, 1)
        rows = list(range(offset, height, 2))
        if len(rows) < count:
            return None
        return sorted(rng.sample(rows, count))
    return sorted(rng.sample(range(height), count))


def _stepwise_changes(left: np.ndarray, rules: list[tuple[int, int]]) -> list[int]:
    cur = np.array(left, copy=True)
    changed: list[int] = []
    for src, dst in rules:
        count = int(np.sum(cur == src))
        changed.append(count)
        cur[cur == src] = dst
    return changed


def _non_bg_uniform_columns(grid: np.ndarray) -> list[int]:
    cols: list[int] = []
    for c in range(grid.shape[1]):
        if int(grid[0, c]) != BG and np.all(grid[:, c] == grid[0, c]):
            cols.append(c)
    return cols


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    sample = _sample_rules(rng)
    if sample is None:
        return None
    rules, support, chain_colors = sample

    active_colors = _choose_active_colors(support, rules, chain_colors, rng)
    if active_colors is None:
        return None

    height = rng.randint(max(10, len(rules) + 3), 20)
    right_width = rng.randint(3, 6)
    left_width = rng.randint(10, min(18, 29 - right_width))

    left = _build_left_pattern(height, left_width, active_colors, chain_colors[0], rng)
    if left is None:
        return None

    if len({int(v) for v in left[left != BG]}) < 3:
        return None

    step_changes = _stepwise_changes(left, rules)
    if not all(count > 0 for count in step_changes):
        return None

    sequential = _apply_rules(left, rules)
    parallel = _apply_parallel_once(left, rules)
    if np.array_equal(sequential, parallel):
        return None
    if np.array_equal(sequential, left):
        return None

    rule_rows = _sample_rule_rows(height, len(rules), rng)
    if rule_rows is None:
        return None

    grid = np.full((height, left_width + 1 + right_width), BG, dtype=int)
    grid[:, :left_width] = left
    grid[:, left_width] = SEP_COLOR

    for row, (src, dst) in zip(rule_rows, rules):
        x = rng.randint(0, right_width - 2)
        grid[row, left_width + 1 + x] = src
        grid[row, left_width + 1 + x + 1] = dst

    uniform_cols = _non_bg_uniform_columns(grid)
    if uniform_cols != [left_width]:
        return None

    try:
        parsed_sep = _separator_col(grid)
        parsed_rules = _extract_rules(grid[:, parsed_sep + 1 :])
    except AssertionError:
        return None

    if parsed_sep != left_width or parsed_rules != rules:
        return None

    out = main(grid)
    if not np.array_equal(out, sequential):
        return None
    if out.shape != left.shape:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chained-replacement puzzle")
