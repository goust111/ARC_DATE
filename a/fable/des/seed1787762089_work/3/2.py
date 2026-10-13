import random

import numpy as np


BG = 0
RECOG_COLORS = [1, 3, 7]
DRAW_COLORS = [4, 6, 9]
SEP_COLOR = 5
MAX_GENERATION_ATTEMPTS = 300


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


GLYPH_CATALOG = [
    _mask(["11110", "10000", "11100", "10000", "11111"]),
    _mask(["00111", "00100", "11110", "00100", "11100"]),
    _mask(["11111", "00100", "01100", "00100", "00111"]),
    _mask(["11100", "00100", "01110", "00100", "00111"]),
    _mask(["11111", "10000", "11110", "00010", "11110"]),
    _mask(["01110", "01000", "01110", "01010", "11110"]),
    _mask(["11110", "00100", "01100", "10100", "11111"]),
    _mask(["11111", "00010", "00110", "00100", "11100"]),
]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    start = tuple(map(int, pts[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


for _catalog_mask in GLYPH_CATALOG:
    assert _catalog_mask.shape == (5, 5)
    assert _is_connected(_catalog_mask)
    assert not np.array_equal(_catalog_mask, np.rot90(_catalog_mask, -1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _is_anchor_pair(p0: tuple[int, int], p1: tuple[int, int]) -> bool:
    r0, c0 = p0
    r1, c1 = p1
    return (r0 == r1 and abs(c0 - c1) == 4) or (c0 == c1 and abs(r0 - r1) == 4)


def _unique_anchor_pairs(points: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pts = sorted({(int(r), int(c)) for r, c in points})
    assert len(pts) % 2 == 0
    used = [False] * len(pts)
    solutions: list[list[tuple[tuple[int, int], tuple[int, int]]]] = []

    def backtrack(current: list[tuple[tuple[int, int], tuple[int, int]]]) -> None:
        if len(solutions) > 1:
            return
        first = -1
        for i, flag in enumerate(used):
            if not flag:
                first = i
                break
        if first == -1:
            solutions.append(current.copy())
            return
        used[first] = True
        for j in range(first + 1, len(pts)):
            if used[j] or not _is_anchor_pair(pts[first], pts[j]):
                continue
            used[j] = True
            current.append((pts[first], pts[j]))
            backtrack(current)
            current.pop()
            used[j] = False
        used[first] = False

    backtrack([])
    assert len(solutions) == 1
    return solutions[0]


def _parse_dictionary(grid: np.ndarray, bg: int) -> dict[int, tuple[np.ndarray, int]]:
    h, w = grid.shape
    sep_rows = [r for r in range(h) if np.all(grid[r, :] == SEP_COLOR)]
    assert sep_rows
    sep_r = int(sep_rows[0])
    sample_r = sep_r - 1
    glyph_r0 = sample_r - 5
    assert glyph_r0 >= 0

    sample_cols = [int(c) for c in np.flatnonzero(grid[sample_r, :] != bg)]
    assert len(sample_cols) == 3

    templates: dict[int, tuple[np.ndarray, int]] = {}
    for center_c in sample_cols:
        c0 = center_c - 2
        c1 = center_c + 3
        assert 0 <= c0 and c1 <= w
        box = grid[glyph_r0:sample_r, c0:c1]
        colors = [int(v) for v in np.unique(box) if int(v) != bg]
        assert len(colors) == 1
        recog = colors[0]
        templates[recog] = (box == recog, int(grid[sample_r, center_c]))
    assert len(templates) == 3
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    h, w = g.shape
    sep_rows = [r for r in range(h) if np.all(g[r, :] == SEP_COLOR)]
    assert sep_rows
    sep_r = int(sep_rows[0])
    work_r0 = sep_r + 1
    templates = _parse_dictionary(g, bg)

    out = g.copy()
    work = g[work_r0:, :]
    for recog, (mask, draw_color) in templates.items():
        points = [tuple(map(int, pt)) for pt in np.argwhere(work == recog)]
        if not points:
            continue
        pairs = _unique_anchor_pairs(points)
        for (r0, c0), (r1, c1) in pairs:
            if r0 == r1:
                top = int(r0 - 2)
                left = int(min(c0, c1))
                draw_mask = mask
            else:
                top = int(min(r0, r1))
                left = int(c0 - 2)
                draw_mask = np.rot90(mask, -1)
            assert 0 <= top <= work.shape[0] - 5
            assert 0 <= left <= w - 5
            patch = out[work_r0 + top : work_r0 + top + 5, left : left + 5]
            patch[:] = bg
            patch[draw_mask] = int(draw_color)
    return out


def _transform_mask(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = mask.copy()
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = np.flipud(out)
    k = rng.randrange(4)
    return np.rot90(out, k)


def _choose_glyphs(rng: random.Random) -> list[np.ndarray] | None:
    for _ in range(40):
        indices = rng.sample(range(len(GLYPH_CATALOG)), 3)
        masks = [_transform_mask(GLYPH_CATALOG[idx], rng) for idx in indices]
        if len({mask.tobytes() for mask in masks}) != 3:
            continue
        return masks
    return None


def _sample_orientations(num_pairs: int, rng: random.Random) -> list[str]:
    if num_pairs == 2:
        result = ["H", "V"]
    elif num_pairs == 3:
        result = ["H", "V", rng.choice(["H", "V"])]
    else:
        result = ["H", "H", "V", "V"]
    rng.shuffle(result)
    return result


def _sample_pair_colors(num_pairs: int, recog_colors: list[int], rng: random.Random) -> list[int]:
    if num_pairs == 2:
        colors = rng.sample(recog_colors, 2)
    elif num_pairs == 3:
        if rng.random() < 0.5:
            colors = rng.sample(recog_colors, 3)
        else:
            pair = rng.sample(recog_colors, 2)
            colors = pair + [rng.choice(pair)]
    else:
        colors = recog_colors[:] + [rng.choice(recog_colors)]
    rng.shuffle(colors)
    return colors


def _anchor_cells(top: int, left: int, orientation: str) -> list[tuple[int, int]]:
    if orientation == "H":
        return [(top + 2, left), (top + 2, left + 4)]
    return [(top, left + 2), (top + 4, left + 2)]


def _place_pair_boxes(
    work_h: int,
    width: int,
    colors: list[int],
    orientations: list[str],
    rng: random.Random,
) -> list[dict[str, int | str]] | None:
    occ = np.zeros((work_h, width), dtype=bool)
    placed: list[dict[str, int | str]] = []
    order = list(range(len(colors)))
    rng.shuffle(order)
    for idx in order:
        candidates: list[tuple[int, int]] = []
        for top in range(work_h - 4):
            for left in range(width - 4):
                r0 = max(0, top - 1)
                r1 = min(work_h, top + 6)
                c0 = max(0, left - 1)
                c1 = min(width, left + 6)
                if occ[r0:r1, c0:c1].any():
                    continue
                candidates.append((top, left))
        if not candidates:
            return None
        top, left = rng.choice(candidates)
        placed.append(
            {
                "top": int(top),
                "left": int(left),
                "color": int(colors[idx]),
                "orientation": orientations[idx],
            }
        )
        occ[max(0, top - 1) : min(work_h, top + 6), max(0, left - 1) : min(width, left + 6)] = True
    return placed


def _build_dictionary(
    grid: np.ndarray,
    masks: list[np.ndarray],
    recog_colors: list[int],
    draw_colors: list[int],
    box_lefts: list[int],
) -> dict[int, tuple[np.ndarray, int]]:
    mapping: dict[int, tuple[np.ndarray, int]] = {}
    for left, mask, recog, draw_color in zip(box_lefts, masks, recog_colors, draw_colors):
        patch = grid[0:5, left : left + 5]
        patch[mask] = int(recog)
        grid[5, left + 2] = int(draw_color)
        mapping[int(recog)] = (mask, int(draw_color))
    grid[6, :] = SEP_COLOR
    return mapping


def _validate_pairings(grid: np.ndarray, templates: dict[int, tuple[np.ndarray, int]]) -> bool:
    work = grid[7:, :]
    try:
        for recog in templates:
            points = [tuple(map(int, pt)) for pt in np.argwhere(work == recog)]
            if points:
                _unique_anchor_pairs(points)
    except AssertionError:
        return False
    return True


def _expected_output(
    base_grid: np.ndarray,
    placements: list[dict[str, int | str]],
    templates: dict[int, tuple[np.ndarray, int]],
) -> np.ndarray:
    out = base_grid.copy()
    for item in placements:
        top = int(item["top"])
        left = int(item["left"])
        recog = int(item["color"])
        orientation = str(item["orientation"])
        mask, draw_color = templates[recog]
        draw_mask = mask if orientation == "H" else np.rot90(mask, -1)
        patch = out[7 + top : 7 + top + 5, left : left + 5]
        patch[:] = BG
        patch[draw_mask] = int(draw_color)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_pairs = rng.randint(2, 4)
    gap1 = rng.randint(1, 3)
    gap2 = rng.randint(1, 3)
    dict_span = 15 + gap1 + gap2
    min_width = max(dict_span, 22 if num_pairs == 4 else 19)
    width = rng.randint(min_width, 29)
    work_h = rng.randint(12 if num_pairs >= 3 else 10, 18)
    height = 7 + work_h

    masks = _choose_glyphs(rng)
    if masks is None:
        return None

    left_margin = rng.randint(0, width - dict_span)
    box_lefts = [left_margin, left_margin + 5 + gap1, left_margin + 10 + gap1 + gap2]

    recog_colors = rng.sample(RECOG_COLORS, 3)
    draw_colors = rng.sample(DRAW_COLORS, 3)
    orientations = _sample_orientations(num_pairs, rng)
    pair_colors = _sample_pair_colors(num_pairs, recog_colors, rng)

    placements = _place_pair_boxes(work_h, width, pair_colors, orientations, rng)
    if placements is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    templates = _build_dictionary(grid, masks, recog_colors, draw_colors, box_lefts)

    for item in placements:
        recog = int(item["color"])
        for r, c in _anchor_cells(int(item["top"]), int(item["left"]), str(item["orientation"])):
            grid[7 + r, c] = recog

    if not _validate_pairings(grid, templates):
        return None

    expected = _expected_output(grid, placements, templates)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
