from __future__ import annotations

from collections import Counter
import math

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(exclude)
    return sorted(colors)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full sprite by OR-ing top-left-aligned partial copies, then repaint it at
    every copy location using that copy's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    bg = _get_bg(g)
    colors = _colors_present(g, exclude={bg})
    assert len(colors) >= 2, "expected at least two foreground colors"

    sprites: list[tuple[int, int, int, np.ndarray]] = []
    template_h = 0
    template_w = 0
    for color in colors:
        mask = g == color
        box = _bbox(mask)
        assert box is not None, "every foreground color must appear"
        r0, c0, r1, c1 = box
        local = mask[r0 : r1 + 1, c0 : c1 + 1]
        assert np.any(local), "each sprite crop must be non-empty"
        sprites.append((int(color), int(r0), int(c0), local))
        template_h = max(template_h, int(local.shape[0]))
        template_w = max(template_w, int(local.shape[1]))

    template = np.zeros((template_h, template_w), dtype=bool)
    for _color, _r0, _c0, local in sprites:
        lh, lw = map(int, local.shape)
        template[:lh, :lw] |= local
    assert np.any(template), "recovered template must be non-empty"

    out = g.copy()
    for color, r0, c0, _local in sprites:
        assert r0 + template_h <= height and c0 + template_w <= width
        region = out[r0 : r0 + template_h, c0 : c0 + template_w]
        assert np.all((region == bg) | (region == color)), "sprite windows must be disjoint"
        region[template] = color
    return out


# --------------- generator ---------------

_COLORS = tuple(range(10))
_GRID_SLACK_RANGE = (1, 7)
_GRID_GAP = 2
_GRID_RETRY_LIMIT = 400
_COUNT_RANGE = (2, 5)
_TEMPLATE_HEIGHT_RANGE = (4, 8)
_TEMPLATE_WIDTH_RANGE = (4, 8)
_TEMPLATE_FILL_RANGE = (0.38, 0.68)
_MIN_VISIBLE_FRACTION = 0.5


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _add_random_path(
    mask: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    rng: np.random.Generator,
) -> None:
    r, c = start
    tr, tc = end
    mask[r, c] = True
    while (r, c) != (tr, tc):
        moves: list[tuple[int, int]] = []
        if r != tr:
            moves.append((r + (1 if tr > r else -1), c))
        if c != tc:
            moves.append((r, c + (1 if tc > c else -1)))
        r, c = moves[int(rng.integers(len(moves)))]
        mask[r, c] = True


def _gen_template(rng: np.random.Generator, height: int, width: int, count: int) -> np.ndarray | None:
    area = height * width
    min_fill = max(count + 5, math.ceil(_TEMPLATE_FILL_RANGE[0] * area))
    max_fill = max(min_fill, math.floor(_TEMPLATE_FILL_RANGE[1] * area))

    mask = np.zeros((height, width), dtype=bool)
    anchor = (0, 0)
    bottom = (height - 1, int(rng.integers(width)))
    right = (int(rng.integers(height)), width - 1)

    _add_random_path(mask, anchor, bottom, rng)
    _add_random_path(mask, anchor, right, rng)

    initial = int(mask.sum())
    if initial > max_fill:
        return None

    target = int(rng.integers(max(min_fill, initial), max_fill + 1))
    while int(mask.sum()) < target:
        frontier = {
            nbr
            for r, c in map(tuple, np.argwhere(mask))
            for nbr in _neighbors4(int(r), int(c), height, width)
            if not mask[nbr]
        }
        if not frontier:
            return None
        rr, cc = tuple(frontier)[int(rng.integers(len(frontier)))]
        mask[rr, cc] = True

    box = _bbox(mask)
    assert box is not None
    if box != (0, 0, height - 1, width - 1):
        return None
    return mask


def _choose_distinct_cells(
    coords: list[tuple[int, int]],
    forced_missing: list[set[tuple[int, int]]],
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    count = len(forced_missing)
    order = list(range(count))
    rng.shuffle(order)
    chosen: list[tuple[int, int] | None] = [None] * count
    used: set[tuple[int, int]] = set()
    for idx in order:
        options = [cell for cell in coords if cell not in used and cell not in forced_missing[idx]]
        if not options:
            return None
        cell = options[int(rng.integers(len(options)))]
        chosen[idx] = cell
        used.add(cell)
    return [cell for cell in chosen if cell is not None]


def _build_partials(
    template: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> list[np.ndarray] | None:
    height, width = map(int, template.shape)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(template)]
    anchor = (0, 0)
    other_coords = [cell for cell in coords if cell != anchor]
    area = len(coords)
    min_visible = max(4, math.ceil(_MIN_VISIBLE_FRACTION * area))

    bottom_cells = {(height - 1, c) for c in range(width) if template[height - 1, c]}
    right_cells = {(r, width - 1) for r in range(height) if template[r, width - 1]}

    forced_missing = [set() for _ in range(count)]
    trim_h = bool(rng.integers(2))
    trim_w = bool(rng.integers(2))
    trim_h_idx = None
    trim_w_idx = None
    if trim_h:
        trim_h_idx = int(rng.integers(count))
        forced_missing[trim_h_idx] |= bottom_cells
    if trim_w:
        trim_w_idx = int(rng.integers(count))
        if count == 2 and trim_h_idx is not None:
            trim_w_idx = trim_h_idx
        forced_missing[trim_w_idx] |= right_cells

    essential = _choose_distinct_cells(other_coords, forced_missing, rng)
    if essential is None:
        return None

    presence: dict[tuple[int, int], set[int]] = {anchor: set(range(count))}
    for idx, cell in enumerate(essential):
        presence[cell] = {idx}

    for cell in other_coords:
        if cell in presence:
            continue
        allowed = {idx for idx in range(count) if cell not in forced_missing[idx]}
        if not allowed:
            return None

        if len(allowed) == 1:
            subset = set(allowed)
        else:
            roll = float(rng.random())
            if count == 2:
                if roll < 0.55:
                    subset = set(allowed)
                else:
                    allowed_list = sorted(allowed)
                    subset = {allowed_list[int(rng.integers(len(allowed_list)))]}
            else:
                allowed_list = sorted(allowed)
                if roll < 0.25:
                    subset = set(allowed_list)
                elif roll < 0.85:
                    subset = set(allowed_list)
                    drop = allowed_list[int(rng.integers(len(allowed_list)))]
                    subset.remove(drop)
                else:
                    keep_size = int(rng.integers(1, len(allowed_list)))
                    pick = rng.choice(len(allowed_list), size=keep_size, replace=False)
                    subset = {allowed_list[int(i)] for i in np.atleast_1d(pick)}
        presence[cell] = subset

    partials = [np.zeros_like(template, dtype=bool) for _ in range(count)]
    for cell, owners in presence.items():
        for idx in owners:
            partials[idx][cell] = True

    if not np.array_equal(np.logical_or.reduce(partials), template):
        return None

    if trim_h and trim_h_idx is not None:
        box = _bbox(partials[trim_h_idx])
        assert box is not None
        if box[2] == height - 1:
            return None
    if trim_w and trim_w_idx is not None:
        box = _bbox(partials[trim_w_idx])
        assert box is not None
        if box[3] == width - 1:
            return None

    for partial in partials:
        box = _bbox(partial)
        if box is None:
            return None
        if box[0] != 0 or box[1] != 0:
            return None
        visible = int(partial.sum())
        if visible < min_visible or visible >= area:
            return None

    for idx in range(count):
        others = [partials[j] for j in range(count) if j != idx]
        if np.array_equal(np.logical_or.reduce(others), template):
            return None

    return partials


def _separated(a: tuple[int, int], b: tuple[int, int], height: int, width: int, gap: int) -> bool:
    ar, ac = a
    br, bc = b
    return (
        ar + height + gap <= br
        or br + height + gap <= ar
        or ac + width + gap <= bc
        or bc + width + gap <= ac
    )


def _place_windows(
    grid_h: int,
    grid_w: int,
    height: int,
    width: int,
    count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(grid_h - height + 1) for c in range(grid_w - width + 1)]
    rng.shuffle(candidates)
    windows: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(_separated((r, c), prev, height, width, _GRID_GAP) for prev in windows):
            windows.append((int(r), int(c)))
            if len(windows) == count:
                return windows
    return None


def _expected_grid(
    shape: tuple[int, int],
    bg: int,
    colors: list[int],
    windows: list[tuple[int, int]],
    template: np.ndarray,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    th, tw = map(int, template.shape)
    for color, (r0, c0) in zip(colors, windows, strict=False):
        out[r0 : r0 + th, c0 : c0 + tw][template] = color
    return out


def _attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    count = int(rng.integers(_COUNT_RANGE[0], _COUNT_RANGE[1] + 1))
    template_h = int(rng.integers(_TEMPLATE_HEIGHT_RANGE[0], _TEMPLATE_HEIGHT_RANGE[1] + 1))
    template_w = int(rng.integers(_TEMPLATE_WIDTH_RANGE[0], _TEMPLATE_WIDTH_RANGE[1] + 1))

    template = _gen_template(rng, template_h, template_w, count)
    if template is None:
        return None

    partials = _build_partials(template, count, rng)
    if partials is None:
        return None

    layout_rows = int(rng.integers(1, min(count, 3) + 1))
    layout_cols = int(math.ceil(count / layout_rows))
    min_h = layout_rows * template_h + (layout_rows + 1) * _GRID_GAP
    min_w = layout_cols * template_w + (layout_cols + 1) * _GRID_GAP
    grid_h = min_h + int(rng.integers(_GRID_SLACK_RANGE[0], _GRID_SLACK_RANGE[1]))
    grid_w = min_w + int(rng.integers(_GRID_SLACK_RANGE[0], _GRID_SLACK_RANGE[1]))

    windows = _place_windows(grid_h, grid_w, template_h, template_w, count, rng)
    if windows is None:
        return None

    bg = int(rng.integers(len(_COLORS)))
    fg_choices = [color for color in _COLORS if color != bg]
    rng.shuffle(fg_choices)
    colors = [int(color) for color in fg_choices[:count]]

    inp = np.full((grid_h, grid_w), bg, dtype=int)
    for color, (r0, c0), partial in zip(colors, windows, partials, strict=False):
        inp[r0 : r0 + template_h, c0 : c0 + template_w][partial] = color

    if int(np.count_nonzero(inp == bg)) <= max(int(np.count_nonzero(inp == color)) for color in colors):
        return None

    expected = _expected_grid(inp.shape, bg, colors, windows, template)
    try:
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GRID_RETRY_LIMIT):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for task 6")
