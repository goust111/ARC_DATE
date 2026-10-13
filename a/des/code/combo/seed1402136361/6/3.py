from __future__ import annotations

import random

import numpy as np


GEOMETRIES = (
    "id",
    "flip_lr",
    "flip_ud",
    "rot90",
    "rot180",
    "rot270",
    "transpose",
    "anti_transpose",
)
RULES = [
    (geom, swap)
    for geom in GEOMETRIES
    for swap in (False, True)
    if not (geom == "id" and not swap)
]
SHAPE_SWAPPING_GEOMS = {"rot90", "rot270", "transpose", "anti_transpose"}

MAX_GENERATION_ATTEMPTS = 500
PAIR_COUNT_CHOICES = (2, 3, 4)
SPRITE_DIM_VALUES = (3, 4, 5)
PAIR_GAP_VALUES = (2, 3, 4)
REGION_EXTRA_WIDTH_RANGE = (2, 6)
TOP_MARGIN_VALUES = (1, 2, 3, 4)
BOTTOM_MARGIN_VALUES = (1, 2, 3, 4, 5)
PALETTE = tuple(range(10))


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    max_count = int(counts.max())
    winners = values[counts == max_count]
    assert len(winners) == 1
    return int(winners[0])


def _tight_crop(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != bg)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _group_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
        else:
            runs.append((start, prev))
            start = prev = idx
    runs.append((start, prev))
    return runs


def _find_separator_span(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates: list[int] = []
    h, w = grid.shape
    for c in range(w):
        col = grid[:, c]
        color = int(col[0])
        if color == bg or not np.all(col == color):
            continue
        if c == 0 or c == w - 1:
            continue
        if not np.any(grid[:, :c] != bg):
            continue
        if not np.any(grid[:, c + 1 :] != bg):
            continue
        candidates.append(c)

    runs = _group_runs(candidates)
    assert runs
    mid = (w - 1) / 2.0
    return min(runs, key=lambda span: abs((span[0] + span[1]) / 2.0 - mid))


def _row_bands(region: np.ndarray, bg: int) -> list[tuple[int, int]]:
    row_has_content = np.any(region != bg, axis=1)
    bands: list[tuple[int, int]] = []
    r = 0
    while r < region.shape[0]:
        if not bool(row_has_content[r]):
            r += 1
            continue
        start = r
        while r + 1 < region.shape[0] and bool(row_has_content[r + 1]):
            r += 1
        bands.append((start, r))
        r += 1
    return bands


def _crop_band(region: np.ndarray, band: tuple[int, int], bg: int) -> np.ndarray:
    r0, r1 = band
    return _tight_crop(region[int(r0) : int(r1) + 1], bg)


def _extract_training_pairs(left_region: np.ndarray, bg: int) -> list[tuple[np.ndarray, np.ndarray]]:
    bands = _row_bands(left_region, bg)
    assert len(bands) >= 4 and len(bands) % 2 == 0

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for idx in range(0, len(bands), 2):
        upper = bands[idx]
        lower = bands[idx + 1]
        assert lower[0] - upper[1] == 2
        inp = _crop_band(left_region, upper, bg)
        out = _crop_band(left_region, lower, bg)
        assert inp.shape == out.shape
        pairs.append((inp, out))
        if idx + 2 < len(bands):
            assert bands[idx + 2][0] - lower[1] >= 3
    assert len(pairs) >= 2
    return pairs


def _extract_test_input(right_region: np.ndarray, bg: int) -> np.ndarray:
    bands = _row_bands(right_region, bg)
    assert len(bands) == 1
    return _crop_band(right_region, bands[0], bg)


def _apply_geometry(sprite: np.ndarray, geometry: str) -> np.ndarray:
    if geometry == "id":
        return sprite.copy()
    if geometry == "flip_lr":
        return np.fliplr(sprite).copy()
    if geometry == "flip_ud":
        return np.flipud(sprite).copy()
    if geometry == "rot90":
        return np.rot90(sprite, -1).copy()
    if geometry == "rot180":
        return np.rot90(sprite, 2).copy()
    if geometry == "rot270":
        return np.rot90(sprite, 1).copy()
    if geometry == "transpose":
        return sprite.T.copy()
    if geometry == "anti_transpose":
        return np.fliplr(np.flipud(sprite)).T.copy()
    raise AssertionError(f"unknown geometry {geometry}")


def _swap_two_foreground_colors(sprite: np.ndarray, bg: int) -> np.ndarray:
    colors = [int(c) for c in np.unique(sprite) if int(c) != bg]
    assert len(colors) == 2
    a, b = colors
    out = sprite.copy()
    out[sprite == a] = b
    out[sprite == b] = a
    return out


def _apply_rule(sprite: np.ndarray, bg: int, geometry: str, swap: bool) -> np.ndarray:
    out = _apply_geometry(sprite, geometry)
    if swap:
        out = _swap_two_foreground_colors(out, bg)
    return out


def _matching_rules(
    train_pairs: list[tuple[np.ndarray, np.ndarray]], bg: int
) -> list[tuple[str, bool]]:
    matches: list[tuple[str, bool]] = []
    for geometry, swap in RULES:
        ok = True
        for inp, out in train_pairs:
            try:
                pred = _apply_rule(inp, bg, geometry, swap)
            except AssertionError:
                ok = False
                break
            if pred.shape != out.shape or not np.array_equal(pred, out):
                ok = False
                break
        if ok:
            matches.append((geometry, swap))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the transformation shown by the training pairs in the left panel and
    apply it to the test sprite in the right panel.

    The rule family is the closure of horizontal flip, vertical flip, 90-degree
    clockwise rotation, and swapping the two foreground colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    sep_c0, sep_c1 = _find_separator_span(g, bg)
    left_region = g[:, :sep_c0]
    right_region = g[:, sep_c1 + 1 :]
    assert left_region.size > 0 and right_region.size > 0

    train_pairs = _extract_training_pairs(left_region, bg)
    test_input = _extract_test_input(right_region, bg)

    matches = _matching_rules(train_pairs, bg)
    assert len(matches) == 1
    geometry, swap = matches[0]
    result = _apply_rule(test_input, bg, geometry, swap)
    return _tight_crop(result, bg)


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _connect_mask(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    while True:
        comps = _mask_components(out)
        if len(comps) <= 1:
            return out
        base = random.choice(comps[0])
        target_comp = min(
            comps[1:],
            key=lambda comp: min(abs(base[0] - r) + abs(base[1] - c) for r, c in comp),
        )
        target = min(target_comp, key=lambda rc: abs(base[0] - rc[0]) + abs(base[1] - rc[1]))
        r, c = base
        tr, tc = target
        if random.random() < 0.5:
            while r != tr:
                out[r, c] = True
                r += 1 if tr > r else -1
            while c != tc:
                out[r, c] = True
                c += 1 if tc > c else -1
        else:
            while c != tc:
                out[r, c] = True
                c += 1 if tc > c else -1
            while r != tr:
                out[r, c] = True
                r += 1 if tr > r else -1
        out[tr, tc] = True


def _random_sprite(h: int, w: int, bg: int, color_a: int, color_b: int) -> np.ndarray:
    for _ in range(300):
        density = random.uniform(0.35, 0.7)
        mask = np.random.random((h, w)) < density
        for r in range(h):
            if not bool(mask[r].any()):
                mask[r, random.randrange(w)] = True
        for c in range(w):
            if not bool(mask[:, c].any()):
                mask[random.randrange(h), c] = True
        mask = _connect_mask(mask)
        if int(mask.sum()) < max(5, h + w - 2):
            continue
        if int(mask.sum()) >= h * w:
            continue

        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        random.shuffle(coords)
        split = random.randint(1, len(coords) - 1)
        color_b_cells = set(coords[:split])

        sprite = np.full((h, w), bg, dtype=int)
        for r, c in coords:
            sprite[r, c] = color_b if (r, c) in color_b_cells else color_a
        return sprite
    raise RuntimeError("failed to sample sprite")


def _rule_changes_sprite(sprite: np.ndarray, bg: int, rule: tuple[str, bool]) -> bool:
    geometry, swap = rule
    return not np.array_equal(sprite, _apply_rule(sprite, bg, geometry, swap))


def _sample_rule() -> tuple[str, bool]:
    return random.choice(RULES)


def _sample_dims(rule: tuple[str, bool]) -> tuple[int, int]:
    geometry, _swap = rule
    if geometry in SHAPE_SWAPPING_GEOMS:
        side = random.choice(SPRITE_DIM_VALUES)
        return side, side
    return random.choice(SPRITE_DIM_VALUES), random.choice(SPRITE_DIM_VALUES)


def _sample_train_and_test_sprites(
    h: int, w: int, bg: int, color_a: int, color_b: int, rule: tuple[str, bool], pair_count: int
) -> tuple[list[np.ndarray], np.ndarray] | None:
    sprites: list[np.ndarray] = []
    needed = pair_count + 1
    for _ in range(needed * 20):
        sprite = _random_sprite(h, w, bg, color_a, color_b)
        if not _rule_changes_sprite(sprite, bg, rule):
            continue
        sprites.append(sprite)
        if len(sprites) == needed:
            break
    if len(sprites) < needed:
        return None
    return sprites[:-1], sprites[-1]


def _build_input_grid(
    train_inputs: list[np.ndarray],
    train_outputs: list[np.ndarray],
    test_input: np.ndarray,
    bg: int,
    sep: int,
) -> np.ndarray:
    h, w = train_inputs[0].shape
    pair_count = len(train_inputs)

    left_width = w + random.randint(*REGION_EXTRA_WIDTH_RANGE)
    right_width = w + random.randint(*REGION_EXTRA_WIDTH_RANGE)
    pair_gaps = [random.choice(PAIR_GAP_VALUES) for _ in range(pair_count - 1)]
    top_margin = random.choice(TOP_MARGIN_VALUES)
    bottom_margin = random.choice(BOTTOM_MARGIN_VALUES)

    pair_tops: list[int] = []
    y = top_margin
    for idx in range(pair_count):
        pair_tops.append(y)
        y += 2 * h + 1
        if idx + 1 < pair_count:
            y += pair_gaps[idx]
    height = y + bottom_margin

    grid = np.full((height, left_width + 1 + right_width), bg, dtype=int)
    grid[:, left_width] = sep

    for top, inp, out in zip(pair_tops, train_inputs, train_outputs):
        x_in = random.randint(0, left_width - w)
        x_out = random.randint(0, left_width - w)
        grid[top : top + h, x_in : x_in + w] = inp
        grid[top + h + 1 : top + 2 * h + 1, x_out : x_out + w] = out

    test_block_top = random.randint(0, height - (2 * h + 1))
    test_x = random.randint(0, right_width - w)
    rc0 = left_width + 1 + test_x
    grid[test_block_top : test_block_top + h, rc0 : rc0 + w] = test_input
    return grid


def _make_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    rule = _sample_rule()
    h, w = _sample_dims(rule)
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    bg, sep, color_a, color_b = random.sample(PALETTE, 4)

    sample = _sample_train_and_test_sprites(h, w, bg, color_a, color_b, rule, pair_count)
    if sample is None:
        return None
    train_inputs, test_input = sample
    train_outputs = [_apply_rule(sprite, bg, *rule) for sprite in train_inputs]

    train_pairs = list(zip(train_inputs, train_outputs))
    matches = _matching_rules(train_pairs, bg)
    if len(matches) != 1 or matches[0] != rule:
        return None

    expected = _tight_crop(_apply_rule(test_input, bg, *rule), bg)
    if np.array_equal(expected, test_input):
        return None

    grid = _build_input_grid(train_inputs, train_outputs, test_input, bg, sep)
    if _dominant_color(grid) != bg:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _make_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid ARC panel")
