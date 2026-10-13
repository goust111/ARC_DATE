from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

PANEL_COLORS = (RED, BLUE, GREEN)
PANEL_WIDTH = 8
TOP_HEIGHT = 4
STRIPE_HEIGHT_CHOICES = (5, 7)
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while queue:
                rr, cc = queue.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
    return count


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _count_holes(mask: np.ndarray) -> int:
    r0, c0, r1, c1 = _bbox(mask)
    cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
    h, w = cropped.shape
    bg = ~cropped
    seen = np.zeros_like(bg, dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            touches_border = r in (0, h - 1) or c in (0, w - 1)
            while queue:
                rr, cc = queue.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        if nr in (0, h - 1) or nc in (0, w - 1):
                            touches_border = True
                        queue.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _parse_panels(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    sep_cols = [c for c in range(w) if np.all(g[:, c] == GREY)]
    assert len(sep_cols) == 2
    left_w = sep_cols[0]
    mid_w = sep_cols[1] - sep_cols[0] - 1
    right_w = w - sep_cols[1] - 1
    assert left_w == mid_w == right_w
    pw = left_w
    panels = [
        g[:, :pw],
        g[:, sep_cols[0] + 1 : sep_cols[1]],
        g[:, sep_cols[1] + 1 :],
    ]
    assert all(panel.shape == (h, pw) for panel in panels)
    return panels


def _find_stripe_start(panel: np.ndarray) -> int:
    h = panel.shape[0]
    row_codes: list[int | None] = []
    for r in range(h):
        row = panel[r]
        if np.all(row == BLACK):
            row_codes.append(BLACK)
        elif np.all(row == YELLOW):
            row_codes.append(YELLOW)
        else:
            row_codes.append(None)

    candidates = []
    for start in range(1, h - 2):
        suffix = row_codes[start:]
        if any(code is None for code in suffix):
            continue
        if any(suffix[i] == suffix[i + 1] for i in range(len(suffix) - 1)):
            continue
        upper = panel[:start]
        if not np.any((upper != BLACK) & (upper != YELLOW)):
            continue
        candidates.append(start)
    assert candidates
    return int(min(candidates))


def _panel_sprite_mask(panel: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    stripe_start = _find_stripe_start(panel)
    upper = panel[:stripe_start]
    sprite_colors = [int(color) for color in np.unique(upper) if int(color) not in (BLACK, YELLOW)]
    assert len(sprite_colors) == 1
    sprite_color = sprite_colors[0]
    mask = upper == sprite_color
    assert _component_count(mask) == 1
    return mask, stripe_start, panel[stripe_start:]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Roll each panel's stripe area downward by the hole count of the sprite above it.

    Input structure:
    - Two solid grey separator columns split the grid into three equal-width vertical panels.
    - In each panel, the upper region contains one connected colored sprite on black background.
    - The lower region is a horizontal stripe block made of full black and full yellow rows.

    Rule:
    - Read the three panels independently.
    - In each panel, count the number of enclosed black holes inside the upper sprite.
    - Cyclically shift that same panel's lower stripe block downward by exactly that hole count, keeping the sprite and separators unchanged.

    Key insight:
    - The sprite is not transformed directly; its internal topology supplies the row-roll amount for a separate stripe region in the same panel.
    """
    g = np.asarray(grid, dtype=int)
    panels = _parse_panels(g)

    out = g.copy()
    panel_width = panels[0].shape[1]
    starts = (0, panel_width + 1, 2 * (panel_width + 1))
    for panel, c0 in zip(panels, starts):
        sprite_mask, stripe_start, stripe = _panel_sprite_mask(panel)
        shift = _count_holes(sprite_mask)
        assert shift in (1, 2, 3)
        out[stripe_start:, c0 : c0 + panel_width] = np.roll(stripe, shift=shift, axis=0)
    return out


def _mask_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


BASE_SPRITES = {
    1: _mask_from_rows(
        [
            "1111111",
            "1000001",
            "1000001",
            "1111111",
        ]
    ),
    2: _mask_from_rows(
        [
            "1111111",
            "1000101",
            "1000101",
            "1111111",
        ]
    ),
    3: _mask_from_rows(
        [
            "1111111",
            "1010101",
            "1010101",
            "1111111",
        ]
    ),
}


def _valid_sprite(mask: np.ndarray, target_holes: int) -> bool:
    if not np.any(mask):
        return False
    if any(not mask[r].any() for r in range(mask.shape[0])):
        return False
    area = int(mask.sum())
    if not (14 <= area <= 26):
        return False
    if _component_count(mask) != 1:
        return False
    if _count_holes(mask) != target_holes:
        return False
    return True


def _mutated_sprite(target_holes: int) -> np.ndarray:
    base = BASE_SPRITES[target_holes].copy()
    if random.random() < 0.5:
        base = np.fliplr(base)

    canvas = np.zeros((TOP_HEIGHT, PANEL_WIDTH), dtype=bool)
    shift = random.randint(0, PANEL_WIDTH - base.shape[1])
    canvas[:, shift : shift + base.shape[1]] = base

    mutation_steps = random.randint(8, 18)
    for _ in range(mutation_steps):
        add_mode = random.random() < 0.55
        if add_mode:
            candidates = [
                (r, c)
                for r in range(TOP_HEIGHT)
                for c in range(PANEL_WIDTH)
                if not canvas[r, c]
                and any(
                    0 <= nr < TOP_HEIGHT and 0 <= nc < PANEL_WIDTH and canvas[nr, nc]
                    for nr, nc in _neighbors4(r, c)
                )
            ]
        else:
            candidates = [(r, c) for r in range(TOP_HEIGHT) for c in range(PANEL_WIDTH) if canvas[r, c]]
        random.shuffle(candidates)
        for r, c in candidates:
            trial = canvas.copy()
            trial[r, c] = add_mode
            if _valid_sprite(trial, target_holes):
                canvas = trial
                break

    assert _valid_sprite(canvas, target_holes)
    return canvas


def _make_stripes(height: int, start_color: int) -> np.ndarray:
    stripe = np.zeros((height, PANEL_WIDTH), dtype=int)
    color = int(start_color)
    for r in range(height):
        stripe[r, :] = color
        color = YELLOW if color == BLACK else BLACK
    return stripe


def _render_panel(sprite_mask: np.ndarray, color: int, stripe: np.ndarray) -> np.ndarray:
    panel = np.zeros((TOP_HEIGHT + stripe.shape[0], PANEL_WIDTH), dtype=int)
    panel[:TOP_HEIGHT][sprite_mask] = int(color)
    panel[TOP_HEIGHT:] = stripe
    return panel


def _expected_output_from_parts(panels: list[np.ndarray], hole_counts: list[int]) -> np.ndarray:
    pieces = []
    for panel, shift in zip(panels, hole_counts):
        stripe = panel[TOP_HEIGHT:]
        out_panel = panel.copy()
        out_panel[TOP_HEIGHT:] = np.roll(stripe, shift=int(shift), axis=0)
        pieces.append(out_panel)
    sep = np.full((panels[0].shape[0], 1), GREY, dtype=int)
    return np.hstack([pieces[0], sep, pieces[1], sep, pieces[2]])


def _semantic_checks(grid: np.ndarray, expected: np.ndarray) -> None:
    panels = _parse_panels(grid)
    areas = []
    hole_counts = []
    for panel in panels:
        sprite_mask, stripe_start, stripe = _panel_sprite_mask(panel)
        assert stripe_start == TOP_HEIGHT
        assert stripe.shape[0] in STRIPE_HEIGHT_CHOICES and stripe.shape[0] % 2 == 1
        assert all(np.all(stripe[r] == stripe[r, 0]) for r in range(stripe.shape[0]))
        assert all(stripe[r, 0] != stripe[r + 1, 0] for r in range(stripe.shape[0] - 1))
        variants = {np.roll(stripe, shift=s, axis=0).tobytes() for s in (1, 2, 3)}
        assert len(variants) == 3
        areas.append(int(sprite_mask.sum()))
        hole_counts.append(_count_holes(sprite_mask))

    assert sorted(hole_counts) == [1, 2, 3]
    area_order = tuple(idx for idx, _area in sorted(enumerate(areas), key=lambda item: item[1]))
    hole_order = tuple(idx for idx, _holes in sorted(enumerate(hole_counts), key=lambda item: item[1]))
    assert area_order != hole_order
    assert area_order != tuple(reversed(hole_order))

    actual = main(grid)
    assert np.array_equal(actual, expected)


def _generate_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    stripe_height = random.choice(STRIPE_HEIGHT_CHOICES)
    hole_counts = random.sample([1, 2, 3], 3)

    sprites = [_mutated_sprite(hole_count) for hole_count in hole_counts]
    stripes = [_make_stripes(stripe_height, random.choice((BLACK, YELLOW))) for _ in range(3)]
    panels = [
        _render_panel(sprite_mask=sprite, color=color, stripe=stripe)
        for sprite, color, stripe in zip(sprites, PANEL_COLORS, stripes)
    ]

    expected = _expected_output_from_parts(panels, hole_counts)
    sep = np.full((TOP_HEIGHT + stripe_height, 1), GREY, dtype=int)
    grid = np.hstack([panels[0], sep, panels[1], sep, panels[2]])
    try:
        _semantic_checks(grid, expected)
    except AssertionError:
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _generate_candidate()
        if candidate is not None:
            return candidate[0]
    raise RuntimeError("failed to generate a valid input for puzzle 11")
