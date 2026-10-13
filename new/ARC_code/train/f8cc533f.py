from __future__ import annotations

import numpy as np

from arc_utils import bbox, colors_present, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete every colored sprite by applying the union of all aligned partial sprite masks.

    Input structure:
    - The dominant color is the background.
    - Each non-background color appears as a separated partial copy of one common sprite mask.
    - The copies share a top-left-aligned sprite window; some copies may be missing lower or right edge pixels.

    Rule:
    - Crop each same-color mask to its bounding box.
    - Pad the crops to the maximum observed height and width, aligned at the top-left, and OR them into one complete template.
    - Fill that template at every sprite location using the sprite's own color.

    Key insight:
    - No single input sprite is complete, but their missing cells are complementary, so the aligned union reveals the full mask.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert len(colors) >= 2, "expected multiple partial sprite copies"

    sprites: list[tuple[int, int, int, np.ndarray]] = []
    for color in colors:
        mask = g == color
        box = bbox(mask)
        assert box is not None, "each foreground color must be present"
        r0, c0, r1, c1 = box
        local = mask[r0 : r1 + 1, c0 : c1 + 1]
        assert np.any(local), "expected a nonempty sprite mask"
        sprites.append((int(color), int(r0), int(c0), local))

    template_h = max(local.shape[0] for _, _, _, local in sprites)
    template_w = max(local.shape[1] for _, _, _, local in sprites)
    assert template_h > 0 and template_w > 0

    template = np.zeros((template_h, template_w), dtype=bool)
    for _, _, _, local in sprites:
        local_h, local_w = map(int, local.shape)
        template[:local_h, :local_w] |= local
    assert np.any(template), "expected a nonempty recovered sprite template"

    out = g.copy()
    for color, r0, c0, _ in sprites:
        r1 = r0 + template_h
        c1 = c0 + template_w
        assert r1 <= height and c1 <= width, "sprite window must fit in the grid"
        region = out[r0:r1, c0:c1]
        assert np.all((region == bg) | (region == color)), "sprite windows must not overlap other colors"
        region[template] = color

    return out

# --------------- generator (o5x) ---------------

# Canvas and palette bounds.
GRID_HEIGHT_RANGE = (11, 22)
GRID_WIDTH_RANGE = (18, 22)
PALETTE_SIZE = 10

# Template geometry and growth.
TEMPLATE_SIDE_RANGE = (5, 7)
TEMPLATE_FILL_FRACTION_RANGE = (0.48, 0.84)
TEMPLATE_STOP_PROBABILITY = 0.35
PATH_VERTICAL_FIRST_PROBABILITY = 0.5
ORTHOGONAL_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Sprite layout constraints.
SPRITE_COUNT_RANGE = (2, 4)
WINDOW_GAP = 1

# Per-sprite damage limits.
DAMAGE_FRACTION_RANGE = (0.07, 0.33)
DAMAGE_MISSING_PIXEL_RANGE = (1, 6)

# Rejection sampling limit.
GENERATION_RETRY_LIMIT = 300


def _gen_path(mask: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    r, c = a
    tr, tc = b
    if random.random() < PATH_VERTICAL_FIRST_PROBABILITY:
        while r != tr:
            mask[r, c] = True
            r += 1 if tr > r else -1
        while c != tc:
            mask[r, c] = True
            c += 1 if tc > c else -1
    else:
        while c != tc:
            mask[r, c] = True
            c += 1 if tc > c else -1
        while r != tr:
            mask[r, c] = True
            r += 1 if tr > r else -1
    mask[tr, tc] = True


def _gen_orbit(side: int, r: int, c: int) -> set[tuple[int, int]]:
    last = side - 1
    return {
        (r, c),
        (c, r),
        (last - r, c),
        (r, last - c),
        (last - r, last - c),
        (last - c, last - r),
        (c, last - r),
        (last - c, r),
    }


def _gen_add_orbit(mask: np.ndarray, r: int, c: int) -> None:
    for rr, cc in _gen_orbit(mask.shape[0], r, c):
        mask[rr, cc] = True


def _gen_template(side: int) -> np.ndarray | None:
    min_fill, max_fill = TEMPLATE_FILL_FRACTION_RANGE
    min_area = int(np.ceil(min_fill * side * side))
    max_area = int(np.floor(max_fill * side * side))
    target = random.randint(min_area, max_area)

    mask = np.zeros((side, side), dtype=bool)
    center = side // 2
    top_c = random.randrange(side)
    _gen_path(mask, (center, center), (0, top_c))
    for r, c in list(map(tuple, np.argwhere(mask))):
        _gen_add_orbit(mask, int(r), int(c))

    if int(mask.sum()) > max_area:
        return None
    target = max(target, int(mask.sum()))

    while int(mask.sum()) < target:
        orbit_choices = []
        for r, c in np.argwhere(mask):
            for dr, dc in ORTHOGONAL_STEPS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < side and 0 <= nc < side and not mask[nr, nc]:
                    orbit = _gen_orbit(side, int(nr), int(nc))
                    added = sum(1 for p in orbit if not mask[p])
                    if added and int(mask.sum()) + added <= max_area:
                        orbit_choices.append((int(nr), int(nc), added))
        if not orbit_choices:
            return None
        if int(mask.sum()) >= min_area and random.random() < TEMPLATE_STOP_PROBABILITY:
            break
        r, c, _ = random.choice(orbit_choices)
        _gen_add_orbit(mask, r, c)

    if not (min_area <= int(mask.sum()) <= max_area):
        return None
    return mask


def _gen_windows(height: int, width: int, side: int, count: int) -> list[tuple[int, int]] | None:
    windows: list[tuple[int, int]] = []
    candidates = [(r, c) for r in range(height - side + 1) for c in range(width - side + 1)]
    for _ in range(count):
        valid = []
        for r, c in candidates:
            ok = True
            for pr, pc in windows:
                separated_rows = (
                    r + side <= pr - WINDOW_GAP or pr + side <= r - WINDOW_GAP
                )
                separated_cols = (
                    c + side <= pc - WINDOW_GAP or pc + side <= c - WINDOW_GAP
                )
                if not (separated_rows or separated_cols):
                    ok = False
                    break
            if ok:
                valid.append((r, c))
        if not valid:
            return None
        windows.append(random.choice(valid))
    return windows


def _gen_damaged_masks(template: np.ndarray, count: int) -> list[np.ndarray] | None:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(template)]
    area = len(coords)
    min_damage, max_damage = DAMAGE_FRACTION_RANGE
    min_missing, max_missing = DAMAGE_MISSING_PIXEL_RANGE
    low = max(min_missing, int(np.ceil(min_damage * area)))
    high = min(max_missing, int(np.floor(max_damage * area)))
    if low > high:
        return None

    top = random.choice([p for p in coords if p[0] == 0])
    left = random.choice([p for p in coords if p[1] == 0])
    bottom = random.choice([p for p in coords if p[0] == template.shape[0] - 1])
    right = random.choice([p for p in coords if p[1] == template.shape[1] - 1])

    missing_counts = {p: 0 for p in coords}
    masks = []
    for i in range(count):
        protect = {top, left}
        if i == 0:
            protect |= {bottom, right}
        k = random.randint(low, high)
        missing = set()
        for _ in range(k):
            choices = [
                p
                for p in coords
                if p not in protect
                and p not in missing
                and missing_counts[p] < count - 1
            ]
            if not choices:
                return None
            p = random.choice(choices)
            missing.add(p)
            missing_counts[p] += 1
        m = template.copy()
        if missing:
            rr, cc = zip(*missing)
            m[rr, cc] = False
        masks.append(m)

    if not np.array_equal(np.logical_or.reduce(masks), template):
        return None
    return masks


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    side = random.randint(*TEMPLATE_SIDE_RANGE)
    count = random.randint(*SPRITE_COUNT_RANGE)

    template = _gen_template(side)
    if template is None:
        return None

    windows = _gen_windows(height, width, side, count)
    if windows is None:
        return None

    masks = _gen_damaged_masks(template, count)
    if masks is None:
        return None

    colors = list(range(PALETTE_SIZE))
    bg = random.choice(colors)
    fg_colors = random.sample([c for c in colors if c != bg], count)

    inp = np.full((height, width), bg, dtype=int)
    expected = inp.copy()
    for color, (r0, c0), damaged in zip(fg_colors, windows, masks):
        inp[r0 : r0 + side, c0 : c0 + side][damaged] = color
        expected[r0 : r0 + side, c0 : c0 + side][template] = color

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid f8cc533f input")
