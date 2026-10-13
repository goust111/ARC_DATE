import random

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRY_LIMIT = 300

Rect = tuple[int, int, int, int]


def _bbox(mask: np.ndarray) -> Rect | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _area(rect: Rect) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _contains(outer: Rect, inner: Rect) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = shape
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


def _fill_rect(grid: np.ndarray, rect: Rect, color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _perimeter_coords(rect: Rect) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        if r1 != r0:
            coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        if c1 != c0:
            coords.append((r, c1))
    return coords


def _render_completed(shape: tuple[int, int], layers: list[tuple[int, int, Rect]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for _area_value, color, rect in sorted(layers, reverse=True):
        _fill_rect(out, rect, color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Complete sparse nested rectangles, then absorb or erase gray noise from their completed boundaries.

    Input structure:
    - The background is black.
    - Each non-black non-gray color appears as sparse boundary samples of one axis-aligned rectangle.
    - Rectangle bboxes have distinct areas and are strictly nested from outermost to innermost.
    - Gray singleton pixels are scattered near the rectangles or out in the background.

    Rule:
    - Take the bbox of each non-black non-gray color as that layer's rectangle.
    - Fill all rectangles from largest area to smallest area so inner layers overwrite outer ones.
    - For each gray pixel, count its 4-neighbors in this completed rectangle image.
    - If at least two orthogonal neighbors belong to rectangles, recolor the gray pixel with the color of the smallest-area adjacent rectangle; otherwise erase it to black.

    Key insight:
    - The sparse colored samples define a nested stack of solid rectangles, and that completed stack tells both which gray pixels are real attached noise and which visible layer color should win near an overlap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    layers: list[tuple[int, int, Rect]] = []
    for color in sorted(int(c) for c in np.unique(g) if c not in (BG, GREY)):
        rect = _bbox(g == color)
        assert rect is not None
        layers.append((_area(rect), color, rect))

    assert layers
    areas = [area for area, _color, _rect in layers]
    assert len(set(areas)) == len(areas)

    layers.sort(reverse=True)
    for i in range(len(layers) - 1):
        assert _contains(layers[i][2], layers[i + 1][2])
        assert layers[i][2] != layers[i + 1][2]

    completed = _render_completed(g.shape, layers)
    result = completed.copy()
    area_by_color = {color: area for area, color, _rect in layers}

    for r, c in np.argwhere(g == GREY):
        adjacent_colors = [
            int(completed[rr, cc])
            for rr, cc in _neighbors4(g.shape, int(r), int(c))
            if int(completed[rr, cc]) != BG
        ]
        if len(adjacent_colors) >= 2:
            best_color = min(set(adjacent_colors), key=lambda color: area_by_color[color])
            result[int(r), int(c)] = best_color
        else:
            result[int(r), int(c)] = BG
    return result


def _choose_rectangles() -> tuple[list[Rect], tuple[int, int]] | None:
    layer_count = random.choice((3, 4))
    gaps = [random.choice((1, 1, 2)) for _ in range(layer_count - 1)]
    inner_h = random.randint(3, 6)
    inner_w = random.randint(3, 6)
    total_inset = sum(gaps)
    outer_h = inner_h + 2 * total_inset
    outer_w = inner_w + 2 * total_inset

    top_margin = random.randint(1, 3)
    left_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    h = top_margin + outer_h + bottom_margin
    w = left_margin + outer_w + right_margin
    if max(h, w) > 30:
        return None

    outer = (top_margin, left_margin, top_margin + outer_h - 1, left_margin + outer_w - 1)
    rects: list[Rect] = []
    offset = 0
    rects.append(outer)
    for gap in gaps:
        offset += gap
        rects.append((outer[0] + offset, outer[1] + offset, outer[2] - offset, outer[3] - offset))

    if any(rect[2] - rect[0] + 1 < 3 or rect[3] - rect[1] + 1 < 3 for rect in rects):
        return None
    if len({_area(rect) for rect in rects}) != len(rects):
        return None
    return rects, (h, w)


def _choose_samples(rect: Rect) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    selected = set(corners)

    side_options = [
        [(r0, c) for c in range(c0 + 1, c1)],
        [(r1, c) for c in range(c0 + 1, c1)],
        [(r, c0) for r in range(r0 + 1, r1)],
        [(r, c1) for r in range(r0 + 1, r1)],
    ]
    nonempty_sides = [side for side in side_options if side]
    random.shuffle(nonempty_sides)
    for side in nonempty_sides[: random.randint(1, len(nonempty_sides))]:
        selected.add(random.choice(side))

    remaining = [pos for pos in _perimeter_coords(rect) if pos not in selected]
    extra_count = random.randint(0, min(2, len(remaining)))
    if extra_count:
        selected.update(random.sample(remaining, extra_count))
    return selected


def _classify_noise_sites(
    expected: np.ndarray,
    sample_mask: np.ndarray,
    area_by_color: dict[int, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    same_color: list[tuple[int, int]] = []
    inner_wins: list[tuple[int, int]] = []
    deletable: list[tuple[int, int]] = []

    for r in range(expected.shape[0]):
        for c in range(expected.shape[1]):
            if sample_mask[r, c]:
                continue
            adj = [int(expected[rr, cc]) for rr, cc in _neighbors4(expected.shape, r, c) if int(expected[rr, cc]) != BG]
            if expected[r, c] != BG:
                if len(adj) < 2:
                    continue
                adj_colors = set(adj)
                if len(adj_colors) == 1:
                    if any(int(expected[rr, cc]) == BG for rr, cc in _neighbors4(expected.shape, r, c)):
                        same_color.append((r, c))
                else:
                    best_color = min(adj_colors, key=lambda color: area_by_color[color])
                    if int(expected[r, c]) == best_color:
                        inner_wins.append((r, c))
            elif len(adj) < 2:
                deletable.append((r, c))

    return same_color, inner_wins, deletable


def _is_isolated(pos: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in chosen)


def _pick_isolated(
    candidates: list[tuple[int, int]],
    chosen: list[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    pool = candidates[:]
    random.shuffle(pool)
    picked: list[tuple[int, int]] = []
    blocked = chosen[:]
    for pos in pool:
        if _is_isolated(pos, blocked):
            picked.append(pos)
            blocked.append(pos)
            if len(picked) == count:
                return picked
    return None


def _gen_attempt() -> np.ndarray | None:
    rect_info = _choose_rectangles()
    if rect_info is None:
        return None
    rects, shape = rect_info
    colors = random.sample(PALETTE, len(rects))
    layers = [(_area(rect), color, rect) for color, rect in zip(colors, rects)]
    expected = _render_completed(shape, layers)
    area_by_color = {color: area for area, color, _rect in layers}

    inp = np.zeros(shape, dtype=int)
    sample_mask = np.zeros(shape, dtype=bool)
    for color, rect in zip(colors, rects):
        for r, c in _choose_samples(rect):
            inp[r, c] = color
            sample_mask[r, c] = True

    same_color, inner_wins, deletable = _classify_noise_sites(expected, sample_mask, area_by_color)
    if not same_color or not inner_wins or len(deletable) < 2:
        return None

    chosen: list[tuple[int, int]] = []

    same_pick = _pick_isolated(same_color, chosen, 1)
    if same_pick is None:
        return None
    chosen.extend(same_pick)

    inner_pick = _pick_isolated(inner_wins, chosen, random.randint(2, 3))
    if inner_pick is None:
        return None
    chosen.extend(inner_pick)

    delete_pick = _pick_isolated(deletable, chosen, 2)
    if delete_pick is None:
        return None
    chosen.extend(delete_pick)

    target_noise_count = random.randint(5, 10)
    extra_pool = same_color + inner_wins + deletable
    random.shuffle(extra_pool)
    for pos in extra_pool:
        if len(chosen) >= target_noise_count:
            break
        if pos in chosen:
            continue
        if _is_isolated(pos, chosen):
            chosen.append(pos)

    if not (5 <= len(chosen) <= 10):
        return None

    for r, c in chosen:
        inp[r, c] = GREY

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid puzzle input")
