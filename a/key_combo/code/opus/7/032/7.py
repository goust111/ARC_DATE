from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
RED = 2
GREY = 5
TEAL = 8
RESERVED = {BG, RED, GREY, TEAL}
RECT_COLORS = [1, 3, 4, 6, 7, 9]


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _find_monochrome_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            seen[r, c] = True
            color = int(g[r, c])
            if color == BG:
                continue

            stack = [(r, c)]
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )

    return comps


def _black_runs(row: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    c = 0
    w = len(row)
    while c < w:
        if int(row[c]) != BG:
            c += 1
            continue
        c0 = c
        while c < w and int(row[c]) == BG:
            c += 1
        runs.append((c0, c))
    return runs


def _platform_and_slots(grid: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    platform_rows = [r for r in range(g.shape[0]) if bool(np.all(g[r] == GREY))]
    assert platform_rows, "expected a full grey platform row"
    platform_r = max(platform_rows)
    assert platform_r > 0, "platform must have a slot row above it"

    slot_row = g[platform_r - 1]
    assert bool(np.all(np.isin(slot_row, [BG, GREY]))), "slot row must contain only background and grey walls"
    runs = _black_runs(slot_row)
    assert len(runs) == 3, "expected exactly three slots"

    slots = [(c0, c1 - c0) for c0, c1 in runs]
    widths = [width for _c0, width in slots]
    assert len(set(widths)) == len(widths), "slot widths must be unique"
    return platform_r, slots


def _rectangles(grid: np.ndarray) -> dict[int, dict]:
    rectangles: dict[int, dict] = {}
    for comp in _find_monochrome_components(grid):
        color = int(comp["color"])
        if color in RESERVED:
            continue
        if int(comp["area"]) <= 1:
            continue
        assert color not in rectangles, "each rectangle color must have exactly one multi-cell component"
        rectangles[color] = comp

    assert rectangles, "expected movable rectangles"
    return rectangles


def _bar_offsets(grid: np.ndarray) -> dict[int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    offsets: defaultdict[int, int] = defaultdict(int)

    for comp in _find_monochrome_components(g):
        color = int(comp["color"])
        if color not in (RED, TEAL):
            continue
        r0, c0, r1, c1 = comp["bbox"]
        assert c0 == c1, "bars must be one column wide"
        tag_r = int(r1) + 1
        assert tag_r < h, "every bar must have a tag below it"
        owner = int(g[tag_r, c0])
        assert owner not in RESERVED, "bar tag must use a rectangle color"
        delta = int(comp["area"]) if color == TEAL else -int(comp["area"])
        offsets[owner] += delta

    assert offsets, "expected arithmetic bars"
    return dict(offsets)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each rectangle so its width matches a platform slot, then insert it at the bar-coded height.

    Input structure:
    - The background is black, with a full grey platform row at the bottom and a grey slot row above it containing exactly three black gaps of distinct widths.
    - Above the platform are three solid monochrome rectangles whose dimensions are 2-by-k or k-by-2, with one unique rectangle color per target slot.
    - The right side contains thin vertical bars in teal and red; each bar has a single tag pixel directly below it whose color matches one rectangle.

    Rule:
    - For each rectangle, treat its long side as the post-rotation width and match that width to the unique slot with the same width.
    - For the same rectangle color, sum the heights of its teal bars and subtract the heights of its red bars to get a nonnegative vertical offset h.
    - Remove the rectangle from its original position, then place a solid 2-by-slot_width copy left-aligned inside the matched slot so its bottom edge sits h cells above the platform row.
    - Keep the bars, tags, and platform unchanged.

    Key insight:
    - Horizontal placement is encoded only by width-matching against the slots, while vertical placement is encoded independently by signed bar-height arithmetic; both constraints must be satisfied for the same object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    platform_r, slots = _platform_and_slots(g)
    slot_by_width = {width: c0 for c0, width in slots}
    rectangles = _rectangles(g)
    offsets = _bar_offsets(g)

    out = g.copy()
    for rect in rectangles.values():
        rr, cc = rect["coords"][:, 0], rect["coords"][:, 1]
        out[rr, cc] = BG

    for color, rect in rectangles.items():
        r0, c0, r1, c1 = rect["bbox"]
        rect_h = int(r1 - r0 + 1)
        rect_w = int(c1 - c0 + 1)
        assert min(rect_h, rect_w) == 2, "each movable object must have a side of length 2"
        target_w = max(rect_h, rect_w)
        assert target_w in slot_by_width, "rectangle width must match a slot after rotation"
        assert color in offsets, "every rectangle must have associated bars"

        lift = int(offsets[color])
        assert 0 <= lift <= 4, "bar arithmetic must produce a nonnegative offset in [0,4]"

        dest_r0 = int(platform_r) - 2 - lift
        dest_c0 = int(slot_by_width[target_w])
        patch = out[dest_r0 : dest_r0 + 2, dest_c0 : dest_c0 + target_w]
        assert patch.shape == (2, target_w), "destination rectangle must fit in the grid"
        assert bool(np.all(patch == BG)), "slot interior above the platform must be empty"
        patch[:] = int(color)

    return out


def _recipe_options(offset: int) -> list[list[tuple[int, int]]]:
    return {
        0: [[(TEAL, 1), (RED, 1)], [(TEAL, 2), (RED, 2)], [(TEAL, 3), (RED, 3)]],
        1: [[(TEAL, 1)], [(TEAL, 2), (RED, 1)], [(TEAL, 3), (RED, 2)]],
        2: [[(TEAL, 2)], [(TEAL, 1), (TEAL, 1)], [(TEAL, 3), (RED, 1)]],
        3: [[(TEAL, 3)], [(TEAL, 1), (TEAL, 2)], [(TEAL, 4), (RED, 1)]],
        4: [[(TEAL, 4)], [(TEAL, 2), (TEAL, 2)]],
    }[int(offset)]


def _choose_recipes(colors: list[int]) -> tuple[dict[int, int], dict[int, list[tuple[int, int]]]]:
    offsets = {color: random.randint(0, 4) for color in colors}
    recipes = {color: random.choice(_recipe_options(offsets[color])) for color in colors}

    if not any(any(bar_color == RED for bar_color, _height in recipe) for recipe in recipes.values()):
        eligible = [color for color in colors if any(bar_color == RED for recipe in _recipe_options(offsets[color]) for bar_color, _ in recipe)]
        if eligible:
            color = random.choice(eligible)
            red_options = [recipe for recipe in _recipe_options(offsets[color]) if any(bar_color == RED for bar_color, _ in recipe)]
            recipes[color] = random.choice(red_options)

    return offsets, recipes


def _can_place_rect(grid: np.ndarray, r0: int, c0: int, h: int, w: int, forbidden_cols: set[int]) -> bool:
    gh, gw = grid.shape
    if r0 < 1 or c0 < 1 or r0 + h >= gh - 1 or c0 + w >= gw - 1:
        return False
    if any(c in forbidden_cols for c in range(c0, c0 + w)):
        return False

    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(gh, r0 + h + 1)
    cc1 = min(gw, c0 + w + 1)
    return bool(np.all(grid[rr0:rr1, cc0:cc1] == BG))


def _build_attempt() -> np.ndarray | None:
    colors = random.sample(RECT_COLORS, 3)
    offsets, recipes = _choose_recipes(colors)

    slot_widths = [2, 3, 4]
    random.shuffle(slot_widths)
    width_by_color = {color: width for color, width in zip(colors, slot_widths)}

    total_bars = sum(len(recipe) for recipe in recipes.values())
    if total_bars > 6:
        return None

    left_margin = random.randint(1, 2)
    gap1 = random.randint(1, 2)
    gap2 = random.randint(1, 2)
    bar_gap = random.randint(2, 3)
    right_margin = random.randint(1, 2)

    slot_span = left_margin + slot_widths[0] + gap1 + slot_widths[1] + gap2 + slot_widths[2]
    bar_region_w = 2 * total_bars - 1
    min_w = slot_span + bar_gap + bar_region_w + right_margin
    if min_w > 30:
        return None

    if min_w > 30:
        return None
    w = random.randint(max(18, min_w), 30)
    h = random.randint(16, 20)
    if max(h, w) > 30:
        return None

    tag_row_max = min(10, h - 7)
    if tag_row_max < 6:
        return None
    tag_row = random.randint(6, tag_row_max)

    grid = np.zeros((h, w), dtype=int)
    grid[h - 1, :] = GREY
    grid[h - 2, :] = GREY

    slot_starts: list[int] = []
    c = left_margin
    for idx, width in enumerate(slot_widths):
        slot_starts.append(c)
        grid[h - 2, c : c + width] = BG
        c += width
        if idx == 0:
            c += gap1
        elif idx == 1:
            c += gap2

    bar_start = max(c + bar_gap, w - (bar_region_w + right_margin))
    if bar_start + bar_region_w + right_margin > w:
        return None

    bar_columns = [bar_start + 2 * i for i in range(total_bars)]
    bar_specs: list[tuple[int, int, int]] = []
    for color in random.sample(colors, len(colors)):
        recipe = recipes[color][:]
        random.shuffle(recipe)
        for bar_color, bar_h in recipe:
            bar_specs.append((color, bar_color, bar_h))
    random.shuffle(bar_specs)

    for (owner, bar_color, bar_h), col in zip(bar_specs, bar_columns):
        top = tag_row - bar_h
        if top < 1:
            return None
        grid[top:tag_row, col] = int(bar_color)
        grid[tag_row, col] = int(owner)

    forbidden_cols = set(bar_columns)
    for start, width in zip(slot_starts, slot_widths):
        forbidden_cols.update(range(start, start + width))

    rect_specs = []
    for color in colors:
        width = int(width_by_color[color])
        if width == 2 or random.random() < 0.5:
            shape = (2, width)
        else:
            shape = (width, 2)
        rect_specs.append((max(shape), color, shape))
    rect_specs.sort(reverse=True)

    for _priority, color, (rh, rw) in rect_specs:
        candidates = []
        for r0 in range(1, h - rh - 2):
            for c0 in range(1, w - rw - 1):
                if _can_place_rect(grid, r0, c0, rh, rw, forbidden_cols):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        grid[r0 : r0 + rh, c0 : c0 + rw] = int(color)

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if not np.array_equal(main(grid.copy()), out):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
