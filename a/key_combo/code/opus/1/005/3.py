import itertools
import random
from collections import defaultdict

import numpy as np


BG = 0
DIVIDER = 4
BAR = 5
SPRITE_COLORS = (1, 2, 3, 6, 7)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _find_components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _bbox(coords):
    coords = list(coords)
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _normalize(coords):
    coords = list(coords)
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _shape_dims(shape):
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _mirror_shape(shape):
    _, w = _shape_dims(shape)
    return _normalize((r, w - 1 - c) for r, c in shape)


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_divider_column(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == DIVIDER)]
    assert len(cols) == 1, "expected exactly one full-height divider"
    return cols[0]


def _extract_bar_and_notches(grid: np.ndarray, divider_col: int):
    left = grid[:, :divider_col]
    bar_mask = left == BAR
    bar_components = _find_components(bar_mask)
    assert bar_components, "expected a gray bar on the left side"
    bar_coords = max(bar_components, key=len)
    r0, c0, r1, c1 = _bbox(bar_coords)

    box = grid[r0 : r1 + 1, c0 : c1 + 1]
    hole_mask = box == BG
    hole_components = _find_components(hole_mask)
    notches = []
    for comp in hole_components:
        touches_top = any(rr == 0 for rr, _ in comp)
        touches_bottom = any(rr == box.shape[0] - 1 for rr, _ in comp)
        if touches_top or touches_bottom:
            notches.append([(r0 + rr, c0 + cc) for rr, cc in comp])
    assert notches, "expected at least one notch"
    return {"bbox": (r0, c0, r1, c1), "coords": bar_coords}, notches


def _extract_sprites(grid: np.ndarray, divider_col: int, bg: int):
    mask = np.zeros_like(grid, dtype=bool)
    right = grid[:, divider_col + 1 :]
    valid = (right != bg) & (right != DIVIDER)
    mask[:, divider_col + 1 :] = valid
    sprites = []
    for comp in _find_components(mask):
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1, "each sprite must be monochrome"
        sprites.append(
            {
                "coords": comp,
                "shape": _normalize(comp),
                "color": int(next(iter(colors))),
            }
        )
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect right-side sprites across the divider, fit matching mirror-shapes into left-side bar notches, and erase the moved originals.

    Input structure:
    - Black background with one full-height yellow vertical divider splitting the grid into a left notch-bar side and a right sprite side.
    - The left side contains one large gray horizontal bar whose top and/or bottom boundary has 2-3 background notches.
    - The right side contains several small monochrome 4-connected sprites in distinct colors; some become notch matches only after left-right reflection.

    Rule:
    - Use the yellow divider as the reflection axis and horizontally mirror each right-side sprite shape while ignoring absolute position.
    - Match each mirrored shape to a left-side notch by exact 4-connected shape equality up to translation.
    - Fill the matched notch with the sprite's original color and erase that sprite's original right-side cells to background.
    - Leave the gray bar, divider, and any unmatched distractor sprites unchanged.

    Key insight:
    - The notch matches the sprite's mirror, not the sprite's original orientation, so direct shape matching on the right-side objects fails.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    divider_col = _find_divider_column(g)
    _bar, notches = _extract_bar_and_notches(g, divider_col)
    sprites = _extract_sprites(g, divider_col, bg)

    notch_by_shape = {}
    for notch in notches:
        shape = _normalize(notch)
        assert shape not in notch_by_shape, "notch shapes must be unique"
        notch_by_shape[shape] = notch

    mirrored_map = defaultdict(list)
    for sprite in sprites:
        mirrored_map[_mirror_shape(sprite["shape"])].append(sprite)

    out = g.copy()
    for shape, notch in notch_by_shape.items():
        if shape not in mirrored_map:
            continue
        assert len(mirrored_map[shape]) == 1, "each notch must have at most one mirrored sprite match"
        sprite = mirrored_map[shape][0]
        for r, c in sprite["coords"]:
            out[r, c] = bg
        for r, c in notch:
            out[r, c] = sprite["color"]
    return out


def _is_connected_shape(shape):
    cells = set(shape)
    stack = [next(iter(cells))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _enumerate_shapes():
    seen = set()
    shapes = []
    for h in range(2, 4):
        for w in range(2, 5):
            cells = [(r, c) for r in range(h) for c in range(w)]
            for size in range(3, 6):
                for combo in itertools.combinations(cells, size):
                    if not _is_connected_shape(combo):
                        continue
                    norm = _normalize(combo)
                    hh, ww = _shape_dims(norm)
                    if len(norm) == hh * ww:
                        continue
                    if norm == _mirror_shape(norm):
                        continue
                    if norm not in seen:
                        seen.add(norm)
                        shapes.append(norm)
    return tuple(shapes)


ALL_SHAPES = _enumerate_shapes()
SHAPES_BY_HEIGHT = {
    2: tuple(shape for shape in ALL_SHAPES if _shape_dims(shape)[0] <= 2),
    3: tuple(shape for shape in ALL_SHAPES if _shape_dims(shape)[0] <= 3),
}


def _composition(total: int, parts: int):
    if total <= 0:
        return [0] * parts
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _pack_intervals(length: int, sizes, min_gap: int):
    extra = length - sum(sizes) - min_gap * (len(sizes) - 1)
    if extra < 0:
        return None
    gaps = _composition(extra, len(sizes) + 1)
    intervals = []
    pos = gaps[0]
    for idx, size in enumerate(sizes):
        intervals.append((pos, pos + size - 1))
        pos += size
        if idx + 1 < len(sizes):
            pos += min_gap + gaps[idx + 1]
    return intervals


def _pick_matched_shapes(bar_h: int, count: int):
    pool = list(SHAPES_BY_HEIGHT[bar_h - 1])
    for _ in range(200):
        random.shuffle(pool)
        chosen = []
        blocked = set()
        for shape in pool:
            if shape in blocked or _mirror_shape(shape) in blocked:
                continue
            chosen.append(shape)
            blocked.add(shape)
            blocked.add(_mirror_shape(shape))
            if len(chosen) == count:
                break
        if len(chosen) != count:
            continue
        if count >= 2 and random.random() < 0.7:
            areas = [len(shape) for shape in chosen]
            if len(set(areas)) == len(areas):
                continue
        return chosen
    return None


def _pick_distractor_shapes(notch_shapes, count: int):
    notch_set = set(notch_shapes)
    notch_areas = {len(shape) for shape in notch_shapes}
    safe_pool = [
        shape
        for shape in ALL_SHAPES
        if shape not in notch_set and _mirror_shape(shape) not in notch_set
    ]
    if len(safe_pool) < count:
        return None

    chosen = []
    same_area = [shape for shape in safe_pool if len(shape) in notch_areas]
    if not same_area:
        return None
    random.shuffle(same_area)
    while same_area and len(chosen) < count:
        shape = same_area.pop()
        if shape not in chosen:
            chosen.append(shape)
    if len(chosen) == count:
        return chosen

    random.shuffle(safe_pool)
    for shape in safe_pool:
        if shape in chosen:
            continue
        chosen.append(shape)
        if len(chosen) == count:
            return chosen
    return None


def _shape_to_cells(shape, top: int, left: int):
    return [(top + r, left + c) for r, c in shape]


def _place_notches(grid, divider_col: int, bar_h: int, notch_shapes):
    h, _ = grid.shape
    bar_top = max(2, h // 2 - bar_h // 2 + random.randint(-1, 1))
    bar_top = min(bar_top, h - bar_h - 2)
    bar_left = 1
    bar_right = divider_col - 2
    bar_width = bar_right - bar_left + 1
    available = bar_width - 2
    widths = [max(c for _, c in shape) + 1 for shape in notch_shapes]
    intervals = _pack_intervals(available, widths, min_gap=1)
    if intervals is None:
        return None

    grid[bar_top : bar_top + bar_h, bar_left : bar_right + 1] = BAR
    absolute_notches = []
    for shape, (start, _end) in zip(notch_shapes, intervals):
        shape_h, _shape_w = _shape_dims(shape)
        side = random.choice(["top", "bottom"])
        row_off = 0 if side == "top" else bar_h - shape_h
        col_off = 1 + start
        cells = _shape_to_cells(shape, bar_top + row_off, bar_left + col_off)
        for r, c in cells:
            grid[r, c] = BG
        absolute_notches.append(cells)
    return {"top": bar_top, "left": bar_left, "right": bar_right, "height": bar_h, "notches": absolute_notches}


def _occupied_with_margin(mask: np.ndarray):
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _place_sprites(grid, divider_col: int, specs):
    h, w = grid.shape
    placements = []
    occupied = grid != BG
    for spec in sorted(specs, key=lambda item: len(item["shape"]), reverse=True):
        shape = spec["shape"]
        color = spec["color"]
        shape_h, shape_w = _shape_dims(shape)
        blocked = _occupied_with_margin(occupied)
        candidates = []
        for top in range(1, h - shape_h):
            for left in range(divider_col + 2, w - shape_w):
                cells = _shape_to_cells(shape, top, left)
                if any(blocked[r, c] for r, c in cells):
                    continue
                candidates.append((top, left))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        cells = _shape_to_cells(shape, top, left)
        for r, c in cells:
            grid[r, c] = color
            occupied[r, c] = True
        placements.append(
            {
                "shape": shape,
                "color": color,
                "cells": cells,
                "target": spec["target"],
            }
        )
    return placements


def _render_expected(input_grid: np.ndarray, placements, notches):
    out = input_grid.copy()
    for sprite in placements:
        if sprite["target"] is None:
            continue
        notch = notches[sprite["target"]]
        for r, c in sprite["cells"]:
            out[r, c] = BG
        for r, c in notch:
            out[r, c] = sprite["color"]
    return out


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, meta):
    divider_col = _find_divider_column(inp)
    _bar, notches = _extract_bar_and_notches(inp, divider_col)
    sprites = _extract_sprites(inp, divider_col, BG)

    notch_shapes = {_normalize(notch) for notch in notches}
    assert len(notch_shapes) == len(meta["notches"]) == len(notches)
    assert len(sprites) == len(meta["sprites"])
    matched = [sprite for sprite in meta["sprites"] if sprite["target"] is not None]
    assert len(matched) >= 2
    assert len(meta["sprites"]) > len(matched)

    mirrored_hits = defaultdict(int)
    original_hits = defaultdict(int)
    for sprite in sprites:
        shape = sprite["shape"]
        mirrored = _mirror_shape(shape)
        if mirrored in notch_shapes:
            mirrored_hits[mirrored] += 1
        if shape in notch_shapes:
            original_hits[shape] += 1

    assert not original_hits, "direct original-shape matching must fail for every sprite"
    assert len(mirrored_hits) == len(matched)
    assert all(count == 1 for count in mirrored_hits.values())
    assert set(mirrored_hits) == {_normalize(notch) for notch in meta["notches"]}

    distractors = [sprite for sprite in meta["sprites"] if sprite["target"] is None]
    assert distractors, "must include at least one distractor sprite"
    assert any(len(sprite["shape"]) in {len(n) for n in meta["notch_shapes"]} for sprite in distractors), (
        "need at least one same-area distractor to block area-only shortcuts"
    )

    actual = main(inp)
    assert np.array_equal(actual, expected)


def _gen_attempt():
    bar_h = 3
    match_count = random.choice([2, 3])
    total_sprites = random.randint(match_count + 1, 5)
    distractor_count = total_sprites - match_count

    notch_shapes = _pick_matched_shapes(bar_h, match_count)
    if notch_shapes is None:
        return None
    distractor_shapes = _pick_distractor_shapes(notch_shapes, distractor_count)
    if distractor_shapes is None:
        return None

    widths = [max(c for _, c in shape) + 1 for shape in notch_shapes]
    min_divider = sum(widths) + len(widths) + 5
    if min_divider > 17:
        return None

    h = random.randint(15, 19)
    w = random.randint(max(22, min_divider + 9), 28)
    divider_col = random.randint(max(13, min_divider), w - 8)

    inp = np.full((h, w), BG, dtype=int)
    inp[:, divider_col] = DIVIDER
    notch_layout = _place_notches(inp, divider_col, bar_h, notch_shapes)
    if notch_layout is None:
        return None

    colors = random.sample(SPRITE_COLORS, total_sprites)
    specs = [
        {"shape": _mirror_shape(shape), "color": colors[idx], "target": idx}
        for idx, shape in enumerate(notch_shapes)
    ]
    specs.extend(
        {"shape": shape, "color": colors[match_count + idx], "target": None}
        for idx, shape in enumerate(distractor_shapes)
    )
    placements = _place_sprites(inp, divider_col, specs)
    if placements is None:
        return None

    expected = _render_expected(inp, placements, notch_layout["notches"])
    meta = {
        "notches": notch_layout["notches"],
        "notch_shapes": notch_shapes,
        "sprites": placements,
    }

    try:
        _semantic_checks(inp, expected, meta)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid reflection-notch assembly input")
