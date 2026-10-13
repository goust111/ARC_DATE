from __future__ import annotations

from itertools import combinations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
ORANGE = 7

ANCHOR_COLORS = (BLUE, RED, GREEN, YELLOW, ORANGE)
CORNERS = ("tl", "tr", "bl", "br")
PAIR_OPTIONS = tuple(combinations(ANCHOR_COLORS, 2))

GRID_HEIGHT_RANGE = (16, 25)
GRID_WIDTH_RANGE = (16, 25)
LINE_COUNT_OPTIONS = (2, 3)
LINE_SIZE_MIN = 2
LINE_SIZE_MAX = 6
PLACEMENT_PADDING = 1
MAX_GENERATION_ATTEMPTS = 400

_RNG = np.random.default_rng()


def _corner_origin(corner: str, height: int, width: int) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, width - 2
    if corner == "bl":
        return height - 2, 0
    if corner == "br":
        return height - 2, width - 2
    raise ValueError(f"unknown corner: {corner}")


def _corner_cells(corner: str, height: int, width: int) -> set[tuple[int, int]]:
    r0, c0 = _corner_origin(corner, height, width)
    return {(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)}


def _dilate(
    coords: set[tuple[int, int]],
    height: int,
    width: int,
    pad: int = PLACEMENT_PADDING,
) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    expanded.add((nr, nc))
    return expanded


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append(comp)

    return components


def _find_color_table(grid: np.ndarray) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    height, width = grid.shape
    candidates = []

    for corner in CORNERS:
        r0, c0 = _corner_origin(corner, height, width)
        patch = grid[r0 : r0 + 2, c0 : c0 + 2]
        if patch.shape != (2, 2):
            continue
        if bool(np.all((patch != BLACK) & (patch != GRAY))):
            rows = [
                (int(patch[0, 0]), int(patch[0, 1])),
                (int(patch[1, 0]), int(patch[1, 1])),
            ]
            candidates.append((rows, _corner_cells(corner, height, width)))

    assert len(candidates) == 1, f"expected one corner color table, got {len(candidates)}"
    return candidates[0]


def _is_colored_cell(grid: np.ndarray, row: int, col: int) -> bool:
    height, width = grid.shape
    if not (0 <= row < height and 0 <= col < width):
        return False
    value = int(grid[row, col])
    return value not in (BLACK, GRAY)


def _parse_line(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict[str, object]:
    coords = sorted(comp)
    size = len(coords)

    if size == 1:
        row, col = coords[0]
        horizontal = _is_colored_cell(grid, row, col - 1) and _is_colored_cell(grid, row, col + 1)
        vertical = _is_colored_cell(grid, row - 1, col) and _is_colored_cell(grid, row + 1, col)
        assert horizontal != vertical, "single grey pixel line must be horizontal or vertical"
        if horizontal:
            anchors = ((row, col - 1), (row, col + 1))
        else:
            anchors = ((row - 1, col), (row + 1, col))
        mid = (row, col)
    else:
        rows = {row for row, _col in coords}
        cols = {col for _row, col in coords}

        if len(rows) == 1:
            row = coords[0][0]
            c0 = min(col for _row, col in coords)
            c1 = max(col for _row, col in coords)
            assert size == c1 - c0 + 1, "horizontal line must be contiguous"
            anchors = ((row, c0 - 1), (row, c1 + 1))
            mid = (row, c0 + (size - 1) // 2)
        else:
            assert len(cols) == 1, "grey component must be a straight line"
            col = coords[0][1]
            r0 = min(row for row, _col in coords)
            r1 = max(row for row, _col in coords)
            assert size == r1 - r0 + 1, "vertical line must be contiguous"
            anchors = ((r0 - 1, col), (r1 + 1, col))
            mid = (r0 + (size - 1) // 2, col)

    a_pos, b_pos = anchors
    assert _is_colored_cell(grid, *a_pos) and _is_colored_cell(grid, *b_pos), "line endpoints must be colored anchors"

    return {
        "size": size,
        "anchors": anchors,
        "anchor_colors": (int(grid[a_pos]), int(grid[b_pos])),
        "mid": mid,
    }


def _lookup_square_color(
    table_rows: list[tuple[int, int]],
    anchor_a: tuple[int, int],
    color_a: int,
    anchor_b: tuple[int, int],
    color_b: int,
) -> int:
    target = frozenset((int(color_a), int(color_b)))
    for left, right in table_rows:
        if frozenset((int(left), int(right))) == target:
            return int(right)
    return int(color_a if anchor_a <= anchor_b else color_b)


def _paint_square(
    out: np.ndarray,
    top: int,
    left: int,
    size: int,
    color: int,
    protected: set[tuple[int, int]],
) -> None:
    height, width = out.shape
    assert 0 <= top and top + size <= height
    assert 0 <= left and left + size <= width

    for row in range(top, top + size):
        for col in range(left, left + size):
            if (row, col) not in protected:
                out[row, col] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    table_rows, _table_cells = _find_color_table(g)
    grey_components = _find_components(g == GRAY)
    assert grey_components, "expected at least one grey connector"

    protected = {
        (int(row), int(col))
        for row, col in np.argwhere((g != BLACK) & (g != GRAY))
    }
    out = g.copy()
    out[g == GRAY] = BLACK

    for comp in sorted(grey_components, key=lambda pts: min(pts)):
        line = _parse_line(g, comp)
        size = int(line["size"])
        mid_row, mid_col = line["mid"]
        anchor_a, anchor_b = line["anchors"]
        color_a, color_b = line["anchor_colors"]
        fill_color = _lookup_square_color(table_rows, anchor_a, color_a, anchor_b, color_b)

        top = int(mid_row) - (size - 1) // 2
        left = int(mid_col) - (size - 1) // 2
        _paint_square(out, top, left, size, fill_color, protected)

    return out


def _line_geometry(top: int, left: int, size: int, orientation: str) -> dict[str, object]:
    mid = (top + (size - 1) // 2, left + (size - 1) // 2)
    square = {
        (row, col)
        for row in range(top, top + size)
        for col in range(left, left + size)
    }

    if orientation == "H":
        row = mid[0]
        gray = {(row, left + offset) for offset in range(size)}
        anchors = ((row, left - 1), (row, left + size))
    else:
        assert orientation == "V"
        col = mid[1]
        gray = {(top + offset, col) for offset in range(size)}
        anchors = ((top - 1, col), (top + size, col))

    return {
        "mid": mid,
        "square": square,
        "gray": gray,
        "anchors": anchors,
    }


def _candidate_positions(height: int, width: int, size: int, orientation: str) -> list[tuple[int, int]]:
    if orientation == "H":
        top_lo, top_hi = 0, height - size
        left_lo, left_hi = 1, width - size - 2
    else:
        top_lo, top_hi = 1, height - size - 2
        left_lo, left_hi = 0, width - size

    if top_lo > top_hi or left_lo > left_hi:
        return []
    return [
        (top, left)
        for top in range(top_lo, top_hi + 1)
        for left in range(left_lo, left_hi + 1)
    ]


def _choose_pairs(pool: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    if count == 0:
        return []
    picks = np.atleast_1d(_RNG.choice(len(pool), size=count, replace=False))
    return [pool[int(idx)] for idx in picks]


def _build_line_templates(height: int, width: int, line_count: int) -> tuple[list[dict[str, object]], list[tuple[int, int]]]:
    matched_count = int(_RNG.integers(1, min(2, line_count) + 1))

    matched_pairs = _choose_pairs(list(PAIR_OPTIONS), matched_count)
    remaining_pairs = [pair for pair in PAIR_OPTIONS if pair not in matched_pairs]
    unmatched_pairs = _choose_pairs(remaining_pairs, line_count - matched_count)
    used_pairs = matched_pairs + unmatched_pairs

    spare_pairs = [pair for pair in PAIR_OPTIONS if pair not in used_pairs]
    if matched_count == 2:
        table_pairs = matched_pairs[:]
    else:
        table_pairs = matched_pairs + [_choose_pairs(spare_pairs, 1)[0]]

    table_rows = []
    for pair in table_pairs:
        if bool(_RNG.integers(0, 2)):
            row = (pair[0], pair[1])
        else:
            row = (pair[1], pair[0])
        table_rows.append((int(row[0]), int(row[1])))
    if bool(_RNG.integers(0, 2)):
        table_rows.reverse()

    max_size = min(LINE_SIZE_MAX, height - 3, width - 3)
    if max_size < LINE_SIZE_MIN:
        raise RuntimeError("grid is too small for the task layout")

    raw_templates = [{"pair": pair, "matched": True} for pair in matched_pairs]
    raw_templates.extend({"pair": pair, "matched": False} for pair in unmatched_pairs)

    line_templates: list[dict[str, object]] = []
    for idx in np.atleast_1d(_RNG.permutation(len(raw_templates))):
        item = dict(raw_templates[int(idx)])
        if item["matched"]:
            orientation = "H" if bool(_RNG.integers(0, 2)) else "V"
        else:
            orientation = "V"

        size = int(_RNG.integers(LINE_SIZE_MIN, max_size + 1))
        endpoint_colors = list(item["pair"])
        if bool(_RNG.integers(0, 2)):
            endpoint_colors.reverse()

        item["orientation"] = orientation
        item["size"] = size
        item["endpoint_colors"] = (int(endpoint_colors[0]), int(endpoint_colors[1]))
        line_templates.append(item)

    if len(line_templates) > 1 and len({int(item["size"]) for item in line_templates}) == 1 and max_size > LINE_SIZE_MIN:
        idx = int(_RNG.integers(0, len(line_templates)))
        current = int(line_templates[idx]["size"])
        choices = [size for size in range(LINE_SIZE_MIN, max_size + 1) if size != current]
        line_templates[idx]["size"] = int(choices[int(_RNG.integers(0, len(choices)))])

    return line_templates, table_rows


def _place_lines(
    height: int,
    width: int,
    line_templates: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    blocked: set[tuple[int, int]] = set()
    for corner in CORNERS:
        blocked |= _dilate(_corner_cells(corner, height, width), height, width)

    placed: list[dict[str, object] | None] = [None] * len(line_templates)
    order = sorted(range(len(line_templates)), key=lambda idx: int(line_templates[idx]["size"]), reverse=True)

    for idx in order:
        item = line_templates[idx]
        size = int(item["size"])
        orientation = str(item["orientation"])
        endpoint_colors = item["endpoint_colors"]

        candidates = _candidate_positions(height, width, size, orientation)
        if not candidates:
            return None

        chosen = None
        for cand_idx in np.atleast_1d(_RNG.permutation(len(candidates))):
            top, left = candidates[int(cand_idx)]
            geom = _line_geometry(top, left, size, orientation)
            anchors_only = set(geom["anchors"])
            raw = geom["square"] | geom["gray"] | anchors_only
            if raw & blocked:
                continue
            chosen = (top, left, geom, raw)
            break

        if chosen is None:
            return None

        top, left, geom, raw = chosen
        anchors = (
            (geom["anchors"][0], int(endpoint_colors[0])),
            (geom["anchors"][1], int(endpoint_colors[1])),
        )
        blocked |= _dilate(raw, height, width)

        placed[idx] = {
            "pair": item["pair"],
            "matched": item["matched"],
            "orientation": orientation,
            "size": size,
            "top": top,
            "left": left,
            "mid": geom["mid"],
            "square": geom["square"],
            "gray": geom["gray"],
            "anchors": anchors,
        }

    assert all(item is not None for item in placed)
    return [item for item in placed if item is not None]


def _render_input(
    height: int,
    width: int,
    corner: str,
    table_rows: list[tuple[int, int]],
    placed_lines: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    r0, c0 = _corner_origin(corner, height, width)
    grid[r0 : r0 + 2, c0 : c0 + 2] = np.array(table_rows, dtype=int)

    for line in placed_lines:
        for anchor_pos, color in line["anchors"]:
            grid[anchor_pos] = int(color)
        for row, col in line["gray"]:
            grid[row, col] = GRAY

    return grid


def _expected_output(
    grid: np.ndarray,
    table_rows: list[tuple[int, int]],
    placed_lines: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    out[out == GRAY] = BLACK
    protected = {
        (int(row), int(col))
        for row, col in np.argwhere((grid != BLACK) & (grid != GRAY))
    }

    for line in placed_lines:
        (anchor_a, color_a), (anchor_b, color_b) = line["anchors"]
        fill_color = _lookup_square_color(table_rows, anchor_a, color_a, anchor_b, color_b)
        _paint_square(out, int(line["top"]), int(line["left"]), int(line["size"]), fill_color, protected)

    return out


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(*GRID_HEIGHT_RANGE))
    width = int(_RNG.integers(*GRID_WIDTH_RANGE))
    line_count = int(LINE_COUNT_OPTIONS[int(_RNG.integers(0, len(LINE_COUNT_OPTIONS)))])
    corner = CORNERS[int(_RNG.integers(0, len(CORNERS)))]

    line_templates, table_rows = _build_line_templates(height, width, line_count)
    placed_lines = _place_lines(height, width, line_templates)
    if placed_lines is None:
        return None

    grid = _render_input(height, width, corner, table_rows, placed_lines)
    expected = _expected_output(grid, table_rows, placed_lines)

    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
