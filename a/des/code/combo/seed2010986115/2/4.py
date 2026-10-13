from __future__ import annotations

import random

import numpy as np


BG = 0
SEPARATOR = 8
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 9)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask)


def _shape_signature(mask: np.ndarray) -> tuple[str, ...]:
    base = _tight_mask(mask)
    variants = []
    for k in range(4):
        rotated = np.rot90(base, k)
        variants.append(_mask_key(rotated))
        variants.append(_mask_key(np.flipud(rotated)))
        variants.append(_mask_key(np.fliplr(rotated)))
    return min(variants)


def _find_separator_row(grid: np.ndarray) -> tuple[int, int]:
    azure_rows = [r for r in range(1, grid.shape[0] - 1) if np.all(grid[r] == SEPARATOR)]
    if len(azure_rows) == 1:
        return int(azure_rows[0]), SEPARATOR

    candidates = []
    for r in range(1, grid.shape[0] - 1):
        values = np.unique(grid[r])
        if values.size == 1 and int(values[0]) != BG:
            candidates.append((r, int(values[0])))
    assert len(candidates) == 1, "expected exactly one full separator row"
    return int(candidates[0][0]), int(candidates[0][1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each mirrored grey silhouette according to the legend segment hit by
    the matching top shape's bounding-box center column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_row, sep_color = _find_separator_row(g)
    legend = g[0]
    assert np.all(legend != BG), "legend row must be fully occupied"
    assert np.all(legend != sep_color), "legend row cannot use the separator color"

    top = g[1:sep_row, :]
    colors = [int(color) for color in np.unique(top) if color not in (BG, sep_color)]
    assert colors, "expected at least one top object"

    out = g.copy()
    for color in colors:
        for comp in _components(top == color):
            abs_coords = comp.copy()
            abs_coords[:, 0] += 1
            _r0, c0, _r1, c1 = _bbox(abs_coords)
            center_col = (c0 + c1) // 2
            target_color = int(legend[center_col])
            assert target_color not in (BG, sep_color)

            mirrored = abs_coords.copy()
            mirrored[:, 0] = 2 * sep_row - mirrored[:, 0]
            assert np.all((0 <= mirrored[:, 0]) & (mirrored[:, 0] < g.shape[0]))
            out[mirrored[:, 0], mirrored[:, 1]] = target_color

    return out


# --------------- generator ---------------

GRID_WIDTH_RANGE = (18, 30)
SEPARATOR_ROW_RANGE = (7, 11)
SEGMENT_COUNT_RANGE = (2, 4)
SEGMENT_MIN_WIDTH = 4
OBJECT_COUNT_RANGE = (2, 5)
SHAPE_HEIGHT_RANGE = (2, 5)
SHAPE_WIDTH_RANGE = (2, 6)
SHAPE_AREA_MIN = 4
GENERATION_ATTEMPTS = 400
SHAPE_ATTEMPTS = 120


def _split_total(total: int, parts: int, min_part: int) -> list[int] | None:
    extra = total - parts * min_part
    if extra < 0:
        return None
    out = [min_part] * parts
    for _ in range(extra):
        out[random.randrange(parts)] += 1
    return out


def _random_shape() -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        h = random.randint(*SHAPE_HEIGHT_RANGE)
        w = random.randint(*SHAPE_WIDTH_RANGE)
        max_area = h * w
        if max_area < SHAPE_AREA_MIN:
            continue

        area_min = max(SHAPE_AREA_MIN, (max_area + 2) // 3)
        area_max = max(area_min, max_area - 1)
        area = random.randint(area_min, area_max)

        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        mask[seed] = True

        while int(mask.sum()) < area:
            frontier = []
            for r in range(h):
                for c in range(w):
                    if mask[r, c]:
                        continue
                    if any(0 <= nr < h and 0 <= nc < w and mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                        frontier.append((r, c))
            if not frontier:
                break
            rr, cc = random.choice(frontier)
            mask[rr, cc] = True

        if int(mask.sum()) != area:
            continue

        tight = _tight_mask(mask)
        if tight.shape[0] < 2 or tight.shape[1] < 2:
            continue
        if int(tight.sum()) < SHAPE_AREA_MIN:
            continue
        return tight
    return None


def _sample_unique_shapes(count: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    signatures: set[tuple[str, ...]] = set()
    for _ in range(count):
        for _ in range(SHAPE_ATTEMPTS):
            shape = _random_shape()
            if shape is None:
                continue
            sig = _shape_signature(shape)
            if sig in signatures:
                continue
            signatures.add(sig)
            shapes.append(shape)
            break
        else:
            return None
    return shapes


def _place_shapes(shapes: list[np.ndarray], top_h: int, width: int) -> list[tuple[np.ndarray, int, int]] | None:
    blocked = np.zeros((top_h, width), dtype=bool)
    placed: list[tuple[np.ndarray, int, int]] = []

    for shape in sorted(shapes, key=lambda x: (-int(x.sum()), -x.shape[0], -x.shape[1])):
        h, w = shape.shape
        if h > top_h or w > width:
            return None
        candidates = []
        for r0 in range(top_h - h + 1):
            for c0 in range(width - w + 1):
                rs = max(0, r0 - 1)
                re = min(top_h, r0 + h + 1)
                cs = max(0, c0 - 1)
                ce = min(width, c0 + w + 1)
                if not blocked[rs:re, cs:ce].any():
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placed.append((shape, r0, c0))
        rs = max(0, r0 - 1)
        re = min(top_h, r0 + h + 1)
        cs = max(0, c0 - 1)
        ce = min(width, c0 + w + 1)
        blocked[rs:re, cs:ce] = True

    return placed


def _legend_row(width: int) -> np.ndarray | None:
    seg_count = random.randint(*SEGMENT_COUNT_RANGE)
    widths = _split_total(width, seg_count, SEGMENT_MIN_WIDTH)
    if widths is None:
        return None
    colors = random.sample(PALETTE, seg_count)
    legend = np.zeros(width, dtype=int)
    cursor = 0
    for seg_w, color in zip(widths, colors):
        legend[cursor : cursor + seg_w] = int(color)
        cursor += seg_w
    return legend


def _build_example() -> np.ndarray | None:
    width = random.randint(*GRID_WIDTH_RANGE)
    legend = _legend_row(width)
    if legend is None:
        return None

    sep_row = random.randint(*SEPARATOR_ROW_RANGE)
    height = 2 * sep_row
    top_h = sep_row - 1
    if top_h < SHAPE_HEIGHT_RANGE[0]:
        return None

    object_count = random.randint(*OBJECT_COUNT_RANGE)
    shapes = _sample_unique_shapes(object_count)
    if shapes is None:
        return None

    placements = _place_shapes(shapes, top_h, width)
    if placements is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[0, :] = legend
    grid[sep_row, :] = SEPARATOR

    top_colors = random.sample(PALETTE, object_count)
    target_colors: list[int] = []
    changed = False

    for color, (shape, r0, c0) in zip(top_colors, placements):
        rows, cols = np.where(shape)
        abs_rows = rows + r0 + 1
        abs_cols = cols + c0
        grid[abs_rows, abs_cols] = int(color)

        center_col = (c0 + c0 + shape.shape[1] - 1) // 2
        target = int(legend[center_col])
        target_colors.append(target)
        changed = changed or (target != int(color))

        mirrored_rows = 2 * sep_row - abs_rows
        grid[mirrored_rows, abs_cols] = GREY

    if len(set(target_colors)) < 2:
        return None
    if not changed:
        return None

    solved = main(grid)
    recolored = solved[sep_row + 1 :, :]
    if GREY in set(map(int, recolored.flat)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _build_example()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
