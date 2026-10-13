from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
NON_SPECIAL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

BODY_HEIGHT_RANGE = (6, 12)
BODY_WIDTH_RANGE = (6, 12)
ACTIVE_COLOR_COUNT_RANGE = (2, 4)
MAX_BRANCH_SAMPLES = 4
GEN_RETRIES = 300


def _legend_height(grid: np.ndarray) -> int:
    h, w = grid.shape
    assert h >= 4 and w >= 5, "grid must fit bars, body, and a 3-column legend"

    count = 0
    for r in range(h - 1, 0, -1):
        if np.all(grid[r, w - 3 : w] != BLACK) and np.all(grid[r, : w - 3] == BLACK):
            count += 1
        else:
            break

    assert count > 0, "expected a bottom-right legend block"
    return count


def _read_layout(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    legend_h = _legend_height(g)
    h, w = g.shape
    body_h = h - 1 - legend_h
    body_w = w - 4
    assert body_h > 0 and body_w > 0, "body must be non-empty"

    assert np.all(g[: h - legend_h, w - 3 : w] == BLACK), "legend columns must be isolated above the legend"
    assert np.all(g[h - legend_h : h, 1 : 1 + body_w] == BLACK), "legend rows must be isolated left of the legend"

    row_codes = g[1 : 1 + body_h, 0].copy()
    col_codes = g[0, 1 : 1 + body_w].copy()
    template = g[1 : 1 + body_h, 1 : 1 + body_w].copy()

    legend_rows = g[h - legend_h : h, w - 3 : w]
    mix_map: dict[tuple[int, int], int] = {}
    for a, b, mixed in legend_rows.tolist():
        a = int(a)
        b = int(b)
        mixed = int(mixed)
        assert a != BLACK and b != BLACK and mixed != BLACK, "legend colors must be active"
        assert a != b, "legend rows should encode two distinct colors"
        assert (a, b) not in mix_map, "legend pair should be unique"
        mix_map[(a, b)] = mixed
        mix_map[(b, a)] = mixed

    return template, row_codes, col_codes, mix_map


def _resolved_color(row_color: int, col_color: int, mix_map: dict[tuple[int, int], int]) -> int:
    if row_color == BLACK and col_color == BLACK:
        return BLACK
    if row_color == BLACK:
        return col_color
    if col_color == BLACK:
        return row_color
    if row_color == col_color:
        return row_color
    assert (row_color, col_color) in mix_map, "missing legend entry for mixed colors"
    return mix_map[(row_color, col_color)]


def _paint_template(
    template: np.ndarray,
    row_codes: np.ndarray,
    col_codes: np.ndarray,
    mix_map: dict[tuple[int, int], int],
) -> np.ndarray:
    h, w = template.shape
    out = np.full((h, w), BLACK, dtype=int)
    valid_template = (template == BLACK) | (template == GREY)
    assert np.all(valid_template), "body may only contain black and grey"

    for r in range(h):
        row_color = int(row_codes[r])
        for c in range(w):
            if int(template[r, c]) != GREY:
                continue
            out[r, c] = _resolved_color(row_color, int(col_codes[c]), mix_map)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read row/column color bars, apply them only at grey template pixels, and use the
    bottom-right 3-column legend to resolve mixed-color intersections.
    """
    template, row_codes, col_codes, mix_map = _read_layout(grid)
    return _paint_template(template, row_codes, col_codes, mix_map)


def _all_pairs(colors: list[int]) -> list[tuple[int, int]]:
    pairs = []
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            pairs.append((int(colors[i]), int(colors[j])))
    return pairs


def _sample_codes(length: int, active_colors: list[int]) -> np.ndarray:
    base = [BLACK] + [int(color) for color in active_colors]
    assert length >= len(base), "dimension must fit black plus every active color"
    codes = base + [random.choice(base) for _ in range(length - len(base))]
    random.shuffle(codes)
    return np.array(codes, dtype=int)


def _branch_name(row_color: int, col_color: int) -> str:
    if row_color == BLACK and col_color == BLACK:
        return "none"
    if row_color != BLACK and col_color == BLACK:
        return "row_only"
    if row_color == BLACK and col_color != BLACK:
        return "col_only"
    if row_color == col_color:
        return "same"
    return "mixed"


def _choose_grey_mask(row_codes: np.ndarray, col_codes: np.ndarray) -> np.ndarray:
    body_h = int(row_codes.size)
    body_w = int(col_codes.size)
    candidates: dict[str, list[tuple[int, int]]] = {
        "none": [],
        "row_only": [],
        "col_only": [],
        "same": [],
        "mixed": [],
    }

    for r in range(body_h):
        for c in range(body_w):
            name = _branch_name(int(row_codes[r]), int(col_codes[c]))
            candidates[name].append((r, c))

    for name, cells in candidates.items():
        assert cells, f"missing candidate cells for branch {name}"

    mask = np.zeros((body_h, body_w), dtype=bool)
    for name, cells in candidates.items():
        take = random.randint(1, min(MAX_BRANCH_SAMPLES, len(cells)))
        for r, c in random.sample(cells, take):
            mask[r, c] = True

    remaining = [(r, c) for r in range(body_h) for c in range(body_w) if not mask[r, c]]
    max_extra = min(len(remaining), max(1, (body_h * body_w) // 6))
    extra = random.randint(0, max_extra)
    if extra:
        for r, c in random.sample(remaining, extra):
            mask[r, c] = True

    for r in range(body_h):
        if not np.any(mask[r]):
            c = random.randrange(body_w)
            mask[r, c] = True
    for c in range(body_w):
        if not np.any(mask[:, c]):
            r = random.randrange(body_h)
            mask[r, c] = True

    if np.all(mask):
        r = random.randrange(body_h)
        c = random.randrange(body_w)
        mask[r, c] = False

    assert np.any(mask), "template needs at least one grey pixel"
    assert np.any(~mask), "template needs at least one black pixel"
    return mask


def _sample_mix_map(active_colors: list[int]) -> dict[tuple[int, int], int]:
    mix_map: dict[tuple[int, int], int] = {}
    for a, b in _all_pairs(active_colors):
        choices = [color for color in NON_SPECIAL_COLORS if color not in (a, b)]
        mix_map[(a, b)] = int(random.choice(choices))
    return mix_map


def _symmetrize_mix_map(mix_map: dict[tuple[int, int], int]) -> dict[tuple[int, int], int]:
    out = dict(mix_map)
    for (a, b), mixed in mix_map.items():
        out[(b, a)] = int(mixed)
    return out


def _build_grid(
    template: np.ndarray,
    row_codes: np.ndarray,
    col_codes: np.ndarray,
    mix_map: dict[tuple[int, int], int],
) -> np.ndarray:
    body_h, body_w = template.shape
    legend_rows = _all_pairs(sorted({pair[0] for pair in mix_map} | {pair[1] for pair in mix_map}))
    legend_pairs = [pair for pair in legend_rows if pair in mix_map]
    random.shuffle(legend_pairs)

    grid = np.full((1 + body_h + len(legend_pairs), 1 + body_w + 3), BLACK, dtype=int)
    grid[1 : 1 + body_h, 0] = row_codes
    grid[0, 1 : 1 + body_w] = col_codes
    grid[1 : 1 + body_h, 1 : 1 + body_w] = template

    for idx, (a, b) in enumerate(legend_pairs):
        r = 1 + body_h + idx
        if random.random() < 0.5:
            left, middle = a, b
        else:
            left, middle = b, a
        grid[r, 1 + body_w :] = (left, middle, mix_map[(a, b)])

    return grid


def _attempt_generate() -> np.ndarray | None:
    active_count = random.randint(*ACTIVE_COLOR_COUNT_RANGE)
    active_colors = random.sample(list(NON_SPECIAL_COLORS), active_count)

    body_h = random.randint(max(BODY_HEIGHT_RANGE[0], active_count + 1), BODY_HEIGHT_RANGE[1])
    body_w = random.randint(max(BODY_WIDTH_RANGE[0], active_count + 1), BODY_WIDTH_RANGE[1])

    row_codes = _sample_codes(body_h, active_colors)
    col_codes = _sample_codes(body_w, active_colors)
    grey_mask = _choose_grey_mask(row_codes, col_codes)
    template = np.where(grey_mask, GREY, BLACK).astype(int)
    mix_map = _sample_mix_map(active_colors)
    grid = _build_grid(template, row_codes, col_codes, mix_map)
    expected = _paint_template(template, row_codes, col_codes, _symmetrize_mix_map(mix_map))
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
