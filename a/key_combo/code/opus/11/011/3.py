from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
TEMPLATE_COLORS = (1, 2, 3, 6, 7, 8, 9)
FOUR_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))

_GEN_CALL_COUNT = 0
_GEN_MODE_CYCLE = ("horizontal", "vertical")
_SHAPE_LIBRARY = (
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
)


def _neighbors4(r: int, c: int):
    for dr, dc in FOUR_DIRS:
        yield r + dr, c + dc


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _shape_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        for candidate in (rotated, np.fliplr(rotated)):
            norm = _normalize_mask(candidate)
            if not any(np.array_equal(norm, prior) for prior in variants):
                variants.append(norm)
    return variants


def _same_shape_under_d4(a: np.ndarray, b: np.ndarray) -> bool:
    norm_b = _normalize_mask(b)
    return any(np.array_equal(v, norm_b) for v in _shape_variants(a))


def _extract_divider(grid: np.ndarray) -> tuple[int, int]:
    grey = np.argwhere(grid == GREY)
    assert len(grey) > 0, "expected grey divider"
    divider_r = int(grey[:, 0].max())
    divider_c = int(grey[:, 1].max())
    expected = {(r, divider_c) for r in range(divider_r + 1)}
    expected |= {(divider_r, c) for c in range(divider_c + 1)}
    actual = {tuple(map(int, rc)) for rc in grey}
    assert actual == expected, "expected one L-shaped divider in the upper-left corner"
    return divider_r, divider_c


def _legend_templates(grid: np.ndarray, divider_r: int, divider_c: int):
    legend = grid[:divider_r, :divider_c]
    forbidden = {BLACK, YELLOW, GREY}
    comps = _component_coords(~np.isin(legend, list(forbidden)))
    assert len(comps) == 2, "expected exactly two legend templates"
    templates = {}
    for coords in comps:
        mask = _crop_mask(coords)
        r0, c0, _r1, _c1 = _bbox(coords)
        if r0 == 0 and c0 > 0:
            direction = "horizontal"
        elif c0 == 0 and r0 > 0:
            direction = "vertical"
        else:
            raise AssertionError("template position must encode a unique direction")
        assert direction not in templates, "each direction must appear once in the legend"
        templates[direction] = mask
    assert set(templates) == {"horizontal", "vertical"}
    return templates


def _query_components(grid: np.ndarray) -> list[np.ndarray]:
    comps = _component_coords(grid == YELLOW)
    assert 3 <= len(comps) <= 4, "expected 3 or 4 yellow query sprites"
    return comps


def _match_direction(query_mask: np.ndarray, templates: dict[str, np.ndarray]) -> str:
    matches = [
        direction
        for direction, template_mask in templates.items()
        if _same_shape_under_d4(template_mask, query_mask)
    ]
    assert len(matches) == 1, "each query sprite must match exactly one legend template"
    return matches[0]


def _paint_tiling(out: np.ndarray, coords: np.ndarray, direction: str) -> None:
    h, w = out.shape
    r0, c0, r1, c1 = _bbox(coords)
    period = (r1 - r0 + 1) if direction == "vertical" else (c1 - c0 + 1)
    axis = 0 if direction == "vertical" else 1
    for sign in (-1, 1):
        step = 1
        while True:
            shift = sign * step * period
            moved_any = False
            for r, c in coords:
                nr = int(r) + (shift if axis == 0 else 0)
                nc = int(c) + (shift if axis == 1 else 0)
                if 0 <= nr < h and 0 <= nc < w:
                    moved_any = True
                    if out[nr, nc] == BLACK:
                        out[nr, nc] = YELLOW
            if not moved_any:
                break
            step += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each yellow query sprite to a legend template and extend it periodically in the template-coded direction.

    Input structure:
    - The background is black, and a grey L-shaped divider in the upper-left encloses a legend region.
    - The legend contains two non-yellow monochrome template sprites: one touching the legend's top row and one touching its left column.
    - Outside the legend are 3 or 4 yellow 4-connected query sprites whose shapes match one legend template up to rotation or reflection.

    Rule:
    - Extract the two legend templates and use their placement to decode direction: top-row template means horizontal tiling, left-column template means vertical tiling.
    - For each yellow query sprite, compare only shape, allowing rotations and reflections, and find the unique matching template.
    - Repeat that exact query sprite in both directions along the decoded axis with period equal to the query bbox width or height, clipping at the grid edge and never overwriting non-background cells.

    Key insight:
    - Shape matching identifies which legend sprite a query belongs to, and the template's position inside the legend silently encodes whether the query should tile horizontally or vertically.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider_r, divider_c = _extract_divider(g)
    templates = _legend_templates(g, divider_r, divider_c)
    queries = _query_components(g)

    out = g.copy()
    for coords in queries:
        direction = _match_direction(_crop_mask(coords), templates)
        _paint_tiling(out, coords, direction)
    return out


def _random_variant(mask: np.ndarray) -> np.ndarray:
    return random.choice(_shape_variants(mask))


def _render_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> np.ndarray:
    coords = []
    for r, c in np.argwhere(mask):
        rr = top + int(r)
        cc = left + int(c)
        grid[rr, cc] = color
        coords.append((rr, cc))
    return np.array(coords, dtype=int)


def _interval_ok(start: int, length: int, used: list[tuple[int, int]]) -> bool:
    end = start + length - 1
    for other_start, other_end in used:
        if not (end + 2 <= other_start or other_end + 2 <= start):
            return False
    return True


def _tiled_cells(shape: tuple[int, int], coords: np.ndarray, direction: str) -> set[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = _bbox(coords)
    period = (r1 - r0 + 1) if direction == "vertical" else (c1 - c0 + 1)
    cells = {tuple(map(int, rc)) for rc in coords}
    axis = 0 if direction == "vertical" else 1
    for sign in (-1, 1):
        step = 1
        while True:
            shift = sign * step * period
            moved_any = False
            for r, c in coords:
                nr = int(r) + (shift if axis == 0 else 0)
                nc = int(c) + (shift if axis == 1 else 0)
                if 0 <= nr < h and 0 <= nc < w:
                    moved_any = True
                    cells.add((nr, nc))
            if not moved_any:
                break
            step += 1
    return cells


def _pick_template_pair() -> tuple[np.ndarray, np.ndarray]:
    top_shape, left_shape = random.sample(_SHAPE_LIBRARY, 2)
    assert not _same_shape_under_d4(top_shape, left_shape)
    return top_shape, left_shape


def _place_legend(grid: np.ndarray, divider_r: int, divider_c: int):
    top_shape, left_shape = _pick_template_pair()
    top_mask = _random_variant(top_shape)
    left_mask = _random_variant(left_shape)

    top_h, top_w = top_mask.shape
    left_h, left_w = left_mask.shape
    top_col = divider_c - top_w
    left_row = divider_r - left_h
    if top_col < 1 or left_row < 1:
        return None

    top_color, left_color = random.sample(TEMPLATE_COLORS, 2)
    _render_mask(grid, top_mask, 0, top_col, top_color)
    _render_mask(grid, left_mask, left_row, 0, left_color)

    legend = grid[:divider_r, :divider_c]
    comps = _component_coords(~np.isin(legend, [BLACK]))
    if len(comps) != 2:
        return None
    return {"horizontal": top_shape, "vertical": left_shape}


def _place_horizontal_queries(
    grid: np.ndarray, template_shape: np.ndarray, divider_r: int, divider_c: int, count: int
) -> list[np.ndarray] | None:
    h, w = grid.shape
    rows_used: list[tuple[int, int]] = []
    coords_list: list[np.ndarray] = []
    for _ in range(count):
        query_mask = _random_variant(template_shape)
        qh, qw = query_mask.shape
        row_candidates = list(range(divider_r + 2, h - qh + 1))
        col_lo = max(divider_c + 3, 2 * qw)
        col_hi = w - 3 * qw
        if col_lo > col_hi or not row_candidates:
            return None
        random.shuffle(row_candidates)
        placed = False
        for top in row_candidates:
            if not _interval_ok(top, qh, rows_used):
                continue
            left = random.randint(col_lo, col_hi)
            coords = _render_mask(grid, query_mask, top, left, YELLOW)
            rows_used.append((top, top + qh - 1))
            coords_list.append(coords)
            placed = True
            break
        if not placed:
            return None
    return coords_list


def _place_vertical_queries(
    grid: np.ndarray, template_shape: np.ndarray, divider_r: int, divider_c: int, count: int
) -> list[np.ndarray] | None:
    h, w = grid.shape
    cols_used: list[tuple[int, int]] = []
    coords_list: list[np.ndarray] = []
    for _ in range(count):
        query_mask = _random_variant(template_shape)
        qh, qw = query_mask.shape
        col_candidates = list(range(divider_c + 2, w - qw + 1))
        row_lo = max(divider_r + 3, 2 * qh)
        row_hi = h - 3 * qh
        if row_lo > row_hi or not col_candidates:
            return None
        random.shuffle(col_candidates)
        placed = False
        for left in col_candidates:
            if not _interval_ok(left, qw, cols_used):
                continue
            top = random.randint(row_lo, row_hi)
            coords = _render_mask(grid, query_mask, top, left, YELLOW)
            cols_used.append((left, left + qw - 1))
            coords_list.append(coords)
            placed = True
            break
        if not placed:
            return None
    return coords_list


def _validate_generation(grid: np.ndarray, mode: str, expected_count: int) -> bool:
    query_comps = _query_components(grid)
    if len(query_comps) != expected_count:
        return False

    occupied_sets = []
    for coords in query_comps:
        if not (3 <= len(coords) <= 5):
            return False
        occupied_sets.append(_tiled_cells(grid.shape, coords, mode))

    for i in range(len(occupied_sets)):
        for j in range(i + 1, len(occupied_sets)):
            if occupied_sets[i] & occupied_sets[j]:
                return False

    out = main(grid)
    if np.count_nonzero(out == YELLOW) <= np.count_nonzero(grid == YELLOW):
        return False
    return True


def _generate_attempt(mode: str) -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    divider_r = random.randint(5, 7)
    divider_c = random.randint(5, 7)
    grid = np.zeros((h, w), dtype=int)
    grid[: divider_r + 1, divider_c] = GREY
    grid[divider_r, : divider_c + 1] = GREY

    templates = _place_legend(grid, divider_r, divider_c)
    if templates is None:
        return None

    query_count = random.randint(3, 4)
    if mode == "horizontal":
        coords_list = _place_horizontal_queries(grid, templates["horizontal"], divider_r, divider_c, query_count)
    else:
        coords_list = _place_vertical_queries(grid, templates["vertical"], divider_r, divider_c, query_count)
    if coords_list is None:
        return None

    if not _validate_generation(grid, mode, query_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    mode = _GEN_MODE_CYCLE[_GEN_CALL_COUNT % len(_GEN_MODE_CYCLE)]
    _GEN_CALL_COUNT += 1
    for _ in range(400):
        grid = _generate_attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
