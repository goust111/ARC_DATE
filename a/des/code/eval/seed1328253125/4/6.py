from __future__ import annotations

import random

import numpy as np


GREY = 5
MAROON = 9
ALL_COLORS = tuple(range(10))
SAFE_COLORS = tuple(color for color in ALL_COLORS if color not in (GREY, MAROON))
GEN_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _rectangles_intersect(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _find_rule_card(grid: np.ndarray) -> tuple[int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    candidates: list[tuple[int, int, int, int]] = []

    for r0 in range(height - 3):
        if r0 > height // 2:
            break
        for c0 in range(width - 4):
            if g[r0, c0] != GREY:
                continue
            for r1 in range(r0 + 3, height):
                if g[r1, c0] != GREY:
                    continue
                for c1 in range(c0 + 4, width):
                    if c1 < width // 2:
                        continue
                    if g[r0, c1] != GREY or g[r1, c1] != GREY:
                        continue
                    if not np.all(g[r0, c0 : c1 + 1] == GREY):
                        continue
                    if not np.all(g[r1, c0 : c1 + 1] == GREY):
                        continue
                    if not np.all(g[r0 : r1 + 1, c0] == GREY):
                        continue
                    if not np.all(g[r0 : r1 + 1, c1] == GREY):
                        continue

                    inner_h = r1 - r0 - 1
                    inner_w = c1 - c0 - 1
                    if inner_h < 2 or inner_w < 3 or inner_w % 2 == 0:
                        continue
                    mid = c0 + 1 + inner_w // 2
                    if not np.all(g[r0 + 1 : r1, mid] == MAROON):
                        continue
                    left_w = mid - (c0 + 1)
                    right_w = c1 - mid - 1
                    if left_w < 1 or left_w != right_w:
                        continue
                    candidates.append((r0, c0, r1, c1))

    assert candidates, "failed to find the rule card"
    candidates.sort(key=lambda box: (box[0], -box[3], -((box[2] - box[0] + 1) * (box[3] - box[1] + 1))))
    return candidates[0]


def _extract_rule(
    grid: np.ndarray, card: tuple[int, int, int, int]
) -> tuple[int, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    r0, c0, r1, c1 = card
    mid = (c0 + c1) // 2
    pattern = g[r0 + 1 : r1, c0 + 1 : mid].copy()
    replacement = g[r0 + 1 : r1, mid + 1 : c1].copy()
    assert pattern.shape == replacement.shape
    assert pattern.size > 0
    assert np.any(pattern != bg)
    assert np.any(replacement != bg)
    return bg, pattern, replacement


def _window_matches(window: np.ndarray, pattern: np.ndarray, bg: int) -> bool:
    return bool(np.all((pattern == bg) | (window == pattern)))


def _apply_rule(
    grid: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int], np.ndarray, np.ndarray, list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    card = _find_rule_card(g)
    bg, pattern, replacement = _extract_rule(g, card)
    ph, pw = map(int, pattern.shape)
    r0, c0, r1, c1 = card

    out = g.copy()
    match_positions: list[tuple[int, int]] = []
    for r in range(height - ph + 1):
        for c in range(width - pw + 1):
            window_box = (r, c, r + ph - 1, c + pw - 1)
            if _rectangles_intersect(window_box, card):
                continue
            window = out[r : r + ph, c : c + pw]
            if not _window_matches(window, pattern, bg):
                continue
            match_positions.append((r, c))
            overwrite = replacement != bg
            window[overwrite] = replacement[overwrite]

    out[r0 : r1 + 1, c0 : c1 + 1] = bg
    return out, card, pattern, replacement, match_positions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply the card's wildcard find/replace rule across the main canvas, then remove the card.

    Input structure:
    - The most common color is the background.
    - A small grey-framed card sits in the upper-right area.
    - Inside the card, a maroon vertical line splits a left match pattern and a right replacement pattern.
    - Background cells on the card are wildcards on the left and preserve-original on the right.

    Rule:
    - Scan the non-card part of the grid in row-major order.
    - Each matching window is updated immediately, so later overlapping windows see earlier replacements.
    - After all replacements, erase the entire rule card to background.
    """
    out, _card, _pattern, _replacement, _match_positions = _apply_rule(grid)
    return out


def _valid_window_positions(
    height: int, width: int, ph: int, pw: int, card: tuple[int, int, int, int]
) -> list[tuple[int, int]]:
    positions = []
    for r in range(height - ph + 1):
        for c in range(width - pw + 1):
            box = (r, c, r + ph - 1, c + pw - 1)
            if not _rectangles_intersect(box, card):
                positions.append((r, c))
    return positions


def _choose_pattern(
    bg: int, ph: int, pw: int, *, overlap_mode: bool, use_wildcards: bool
) -> np.ndarray:
    pattern = np.full((ph, pw), bg, dtype=int)
    cells = [(r, c) for r in range(ph) for c in range(pw)]
    area = len(cells)
    fixed_hi = area - 1 if use_wildcards else area
    fixed_count = random.randint(2, fixed_hi)
    fixed_cells = random.sample(cells, fixed_count)
    palette = [color for color in SAFE_COLORS if color != bg]

    if overlap_mode:
        color = random.choice(palette)
        for r, c in fixed_cells:
            pattern[r, c] = color
        return pattern

    n_colors = random.randint(1, min(3, len(palette)))
    colors = random.sample(palette, n_colors)
    for r, c in fixed_cells:
        pattern[r, c] = random.choice(colors)
    return pattern


def _choose_replacement(
    bg: int, ph: int, pw: int, *, use_preserve: bool
) -> np.ndarray:
    replacement = np.full((ph, pw), bg, dtype=int)
    cells = [(r, c) for r in range(ph) for c in range(pw)]
    area = len(cells)
    non_bg_hi = area - 1 if use_preserve else area
    non_bg_lo = 1 if use_preserve else max(2, area // 2)
    non_bg_count = random.randint(non_bg_lo, non_bg_hi)
    chosen_cells = random.sample(cells, non_bg_count)
    palette = [color for color in SAFE_COLORS if color != bg]
    n_colors = random.randint(1, min(3, len(palette)))
    colors = random.sample(palette, n_colors)
    for r, c in chosen_cells:
        replacement[r, c] = random.choice(colors)
    return replacement


def _sample_anchors(
    positions: list[tuple[int, int]],
    ph: int,
    pw: int,
    *,
    overlap_mode: bool,
) -> list[tuple[int, int]] | None:
    if not positions:
        return None

    pos_set = set(positions)
    count = random.randint(2, 4) if overlap_mode else random.randint(1, 4)
    first = random.choice(positions)
    anchors = [first]

    if overlap_mode:
        offsets = [
            (dr, dc)
            for dr in range(-(ph - 1), ph)
            for dc in range(-(pw - 1), pw)
            if (dr != 0 or dc != 0) and abs(dr) < ph and abs(dc) < pw
        ]
        random.shuffle(offsets)
        second = None
        for dr, dc in offsets:
            candidate = (first[0] + dr, first[1] + dc)
            if candidate not in pos_set:
                continue
            if _rectangles_intersect(
                (first[0], first[1], first[0] + ph - 1, first[1] + pw - 1),
                (candidate[0], candidate[1], candidate[0] + ph - 1, candidate[1] + pw - 1),
            ):
                second = candidate
                break
        if second is None:
            return None
        anchors.append(second)

    random.shuffle(positions)
    for pos in positions:
        if len(anchors) >= count:
            break
        box = (pos[0], pos[1], pos[0] + ph - 1, pos[1] + pw - 1)
        if all(
            not _rectangles_intersect(
                box, (ar, ac, ar + ph - 1, ac + pw - 1)
            )
            for ar, ac in anchors
        ):
            anchors.append(pos)

    return anchors


def _place_random_rectangles(
    grid: np.ndarray,
    reserved: list[tuple[int, int, int, int]],
    *,
    bg: int,
    colors: list[int],
) -> None:
    height, width = map(int, grid.shape)
    attempts = 0
    target = random.randint(1, 4)
    placed = 0
    while placed < target and attempts < 40:
        attempts += 1
        h = random.randint(1, 3)
        w = random.randint(1, 3)
        if h >= height or w >= width:
            continue
        r0 = random.randint(0, height - h)
        c0 = random.randint(0, width - w)
        box = (r0, c0, r0 + h - 1, c0 + w - 1)
        if any(_rectangles_intersect(box, other) for other in reserved):
            continue
        color = random.choice(colors)
        grid[r0 : r0 + h, c0 : c0 + w] = color
        placed += 1


def _draw_card(
    grid: np.ndarray,
    card: tuple[int, int, int, int],
    pattern: np.ndarray,
    replacement: np.ndarray,
    *,
    bg: int,
) -> None:
    r0, c0, r1, c1 = card
    ph, pw = map(int, pattern.shape)
    assert r1 - r0 + 1 == ph + 2
    assert c1 - c0 + 1 == 2 * pw + 3

    grid[r0 : r1 + 1, c0 : c1 + 1] = bg
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    mid = c0 + pw + 1
    grid[r0 + 1 : r1, c0 + 1 : mid] = pattern
    grid[r0 + 1 : r1, mid] = MAROON
    grid[r0 + 1 : r1, mid + 1 : c1] = replacement


def _generate_attempt() -> np.ndarray | None:
    bg = random.choice(SAFE_COLORS)
    ph = random.randint(2, 4)
    pw = random.randint(2, 4)
    overlap_mode = random.random() < 0.4
    use_wildcards = random.random() < 0.5
    use_preserve = random.random() < 0.5

    pattern = _choose_pattern(bg, ph, pw, overlap_mode=overlap_mode, use_wildcards=use_wildcards)
    replacement = _choose_replacement(bg, ph, pw, use_preserve=use_preserve)

    card_h = ph + 2
    card_w = 2 * pw + 3
    min_height = max(12, card_h + ph + 3)
    min_width = max(16, card_w + pw + 5)
    height = random.randint(min_height, 24)
    width = random.randint(min_width, 30)

    top_margin = random.randint(0, 1)
    right_margin = random.randint(0, 1)
    card_r0 = top_margin
    card_c1 = width - 1 - right_margin
    card_r1 = card_r0 + card_h - 1
    card_c0 = card_c1 - card_w + 1
    card = (card_r0, card_c0, card_r1, card_c1)

    positions = _valid_window_positions(height, width, ph, pw, card)
    anchors = _sample_anchors(positions, ph, pw, overlap_mode=overlap_mode)
    if anchors is None or not anchors:
        return None

    grid = np.full((height, width), bg, dtype=int)

    anchor_boxes = [(r, c, r + ph - 1, c + pw - 1) for r, c in anchors]
    reserved = anchor_boxes + [card]
    distractor_colors = [color for color in SAFE_COLORS if color not in (bg, *np.unique(pattern[pattern != bg]))]
    if not distractor_colors:
        distractor_colors = [color for color in SAFE_COLORS if color != bg]
    _place_random_rectangles(grid, reserved, bg=bg, colors=distractor_colors)

    required: dict[tuple[int, int], int] = {}
    for ar, ac in anchors:
        for lr in range(ph):
            for lc in range(pw):
                color = int(pattern[lr, lc])
                if color == bg:
                    continue
                key = (ar + lr, ac + lc)
                prev = required.get(key)
                if prev is not None and prev != color:
                    return None
                required[key] = color

    wildcard_fill_candidates = []
    for ar, ac in anchors:
        for lr in range(ph):
            for lc in range(pw):
                if int(pattern[lr, lc]) != bg:
                    continue
                key = (ar + lr, ac + lc)
                if key not in required:
                    wildcard_fill_candidates.append(key)

    if use_wildcards:
        wildcard_fill_candidates = sorted(set(wildcard_fill_candidates))
        if not wildcard_fill_candidates:
            return None
        fill_palette = [color for color in SAFE_COLORS if color not in (bg, *set(required.values()))]
        if not fill_palette:
            fill_palette = [color for color in SAFE_COLORS if color != bg]
        fill_count = random.randint(1, min(4, len(wildcard_fill_candidates)))
        for gr, gc in random.sample(wildcard_fill_candidates, fill_count):
            grid[gr, gc] = random.choice(fill_palette)

    for (gr, gc), color in required.items():
        grid[gr, gc] = color

    _draw_card(grid, card, pattern, replacement, bg=bg)

    if _most_common_color(grid) != bg:
        return None
    try:
        out, found_card, found_pattern, found_replacement, matches = _apply_rule(grid)
    except AssertionError:
        return None

    if found_card != card:
        return None
    if not np.array_equal(found_pattern, pattern):
        return None
    if not np.array_equal(found_replacement, replacement):
        return None
    if not matches or len(matches) > 7:
        return None

    if any(
        not _window_matches(grid[r : r + ph, c : c + pw], pattern, bg)
        for r, c in anchors
    ):
        return None

    if use_wildcards:
        wildcard_used = False
        for r, c in matches:
            for lr in range(ph):
                for lc in range(pw):
                    if int(pattern[lr, lc]) == bg and int(grid[r + lr, c + lc]) != bg:
                        wildcard_used = True
                        break
                if wildcard_used:
                    break
            if wildcard_used:
                break
        if not wildcard_used:
            return None

    input_without_card = grid.copy()
    input_without_card[card_r0 : card_r1 + 1, card_c0 : card_c1 + 1] = bg
    if np.array_equal(input_without_card, out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
