from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

OBJECT_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
RNG = np.random.default_rng()

MAX_GENERATION_ATTEMPTS = 300


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]] | set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((int(r) - r0, int(c) - c0) for r, c in coords))


def _norm_dims(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(int(r) for r, _ in coords) + 1, max(int(c) for _, c in coords) + 1


def _mirror_normalized(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    _h, w = _norm_dims(coords)
    return _normalize([(int(r), int(w - 1 - c)) for r, c in coords])


def _mirror_in_cell(coords: tuple[tuple[int, int], ...], width: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(width - 1 - c)) for r, c in coords))


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    values = sorted(int(x) for x in indices)
    out: list[tuple[int, int]] = []
    start = end = values[0]
    for value in values[1:]:
        if value == end + 1:
            end = value
        else:
            out.append((start, end))
            start = end = value
    out.append((start, end))
    return out


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _monochrome_components(grid: np.ndarray, region: np.ndarray, excluded: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if not bool(region[r, c]) or bool(seen[r, c]) or color in excluded:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and bool(region[nr, nc])
                        and not bool(seen[nr, nc])
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "coords": coords, "shape_key": _normalize(coords)})
    return comps


def _extract_table(grid: np.ndarray) -> dict:
    grey_comps = _mask_components(grid == GREY)
    assert grey_comps, "expected a grey table"
    table_coords = max(grey_comps, key=len)
    r0, c0, r1, c1 = _bbox(table_coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    grey = sub == GREY

    row_runs = _runs([rr for rr in range(sub.shape[0]) if bool(grey[rr, :].all())])
    col_runs = _runs([cc for cc in range(sub.shape[1]) if bool(grey[:, cc].all())])
    assert len(col_runs) == 3, "expected left border, divider, right border"
    assert 4 <= len(row_runs) <= 6, "expected 3 to 5 table rows"

    row_spans = [(row_runs[i][1] + 1, row_runs[i + 1][0] - 1) for i in range(len(row_runs) - 1)]
    col_spans = [(col_runs[i][1] + 1, col_runs[i + 1][0] - 1) for i in range(len(col_runs) - 1)]
    assert len(col_spans) == 2, "expected two table columns"
    assert all(a <= b for a, b in row_spans)
    assert all(a <= b for a, b in col_spans)

    left_span = (int(c0 + col_spans[0][0]), int(c0 + col_spans[0][1]))
    right_span = (int(c0 + col_spans[1][0]), int(c0 + col_spans[1][1]))
    assert left_span[1] - left_span[0] == right_span[1] - right_span[0], "table cells must share a width"

    rows = [(int(r0 + a), int(r0 + b)) for a, b in row_spans]
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "rows": rows,
        "left_span": left_span,
        "right_span": right_span,
    }


def _cell_color_coords(grid: np.ndarray, row_span: tuple[int, int], col_span: tuple[int, int], color: int) -> tuple[tuple[int, int], ...]:
    r0, r1 = row_span
    c0, c1 = col_span
    cell = grid[r0 : r1 + 1, c0 : c1 + 1]
    allowed = {BLACK, color}
    assert set(map(int, np.unique(cell))).issubset(allowed)
    coords = [(int(r), int(c)) for r, c in np.argwhere(cell == color)]
    assert coords, "expected a template inside the left cell"
    return tuple(sorted(coords))


def _paint_cell(grid: np.ndarray, row_span: tuple[int, int], col_span: tuple[int, int], coords: tuple[tuple[int, int], ...], color: int) -> None:
    r0, r1 = row_span
    c0, c1 = col_span
    grid[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    for dr, dc in coords:
        grid[int(r0 + dr), int(c0 + dc)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match left-side object shapes to azure templates in the right-side table.

    Matched rows copy the template shape into the row's right cell using the object's color.
    Unmatched rows receive a horizontally mirrored maroon copy of the left-cell template.
    Left-side source objects are erased; the grey table and azure templates remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    table = _extract_table(g)
    left_span = table["left_span"]
    right_span = table["right_span"]
    cell_w = int(left_span[1] - left_span[0] + 1)

    templates: list[dict] = []
    shape_to_row: dict[tuple[tuple[int, int], ...], int] = {}
    for idx, row_span in enumerate(table["rows"]):
        cell_coords = _cell_color_coords(g, row_span, left_span, AZURE)
        shape_key = _normalize(cell_coords)
        assert shape_key not in shape_to_row, "template shapes must be unique"
        shape_to_row[shape_key] = int(idx)
        templates.append({"cell_coords": cell_coords, "shape_key": shape_key})

    region = np.zeros_like(g, dtype=bool)
    region[:, : table["bbox"][1]] = True
    objects = _monochrome_components(g, region=region, excluded={BLACK, GREY, AZURE, MAROON})

    row_to_color: dict[int, int] = {}
    out = g.copy()
    for obj in objects:
        key = obj["shape_key"]
        assert key in shape_to_row, "every object must match a template by shape"
        row_idx = int(shape_to_row[key])
        assert row_idx not in row_to_color, "each template row may match at most one object"
        row_to_color[row_idx] = int(obj["color"])
        for r, c in obj["coords"]:
            out[int(r), int(c)] = BLACK

    for idx, row_span in enumerate(table["rows"]):
        if idx in row_to_color:
            coords = templates[idx]["cell_coords"]
            color = int(row_to_color[idx])
        else:
            coords = _mirror_in_cell(templates[idx]["cell_coords"], cell_w)
            color = MAROON
        _paint_cell(out, row_span, right_span, coords, color)

    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for dr, dc in DIR8:
        rs = slice(max(0, dr), mask.shape[0] + min(0, dr))
        cs = slice(max(0, dc), mask.shape[1] + min(0, dc))
        tr = slice(max(0, -dr), mask.shape[0] + min(0, -dr))
        tc = slice(max(0, -dc), mask.shape[1] + min(0, -dc))
        out[tr, tc] |= mask[rs, cs]
    return out


def _random_connected_shape(area: int, max_h: int, max_w: int) -> tuple[tuple[int, int], ...]:
    for _ in range(400):
        coords: set[tuple[int, int]] = {(0, 0)}
        while len(coords) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in coords:
                for dr, dc in DIR4:
                    cell = (int(r + dr), int(c + dc))
                    if cell in coords or cell in frontier:
                        continue
                    trial = _normalize(coords | {cell})
                    h, w = _norm_dims(trial)
                    if h <= max_h and w <= max_w:
                        frontier.append(cell)
            if not frontier:
                break
            pick = frontier[int(RNG.integers(0, len(frontier)))]
            coords.add(pick)
        if len(coords) != area:
            continue
        norm = _normalize(coords)
        if _mirror_normalized(norm) == norm:
            continue
        return norm
    raise RuntimeError("failed to sample a non-symmetric connected shape")


def _shape_mask(shape_key: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = _norm_dims(shape_key)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape_key:
        mask[int(r), int(c)] = True
    return mask


def _place_object(grid: np.ndarray, occupied: np.ndarray, shape_key: tuple[tuple[int, int], ...], color: int, table_left: int) -> list[tuple[int, int]] | None:
    mask = _shape_mask(shape_key)
    h, w = mask.shape
    col_max = int(table_left - w - 2)
    if col_max < 1:
        return None

    blocked = _dilate8(occupied)
    for _ in range(400):
        r0 = int(RNG.integers(0, grid.shape[0] - h + 1))
        c0 = int(RNG.integers(0, col_max + 1))
        if bool((blocked[r0 : r0 + h, c0 : c0 + w] & mask).any()):
            continue
        coords: list[tuple[int, int]] = []
        for dr, dc in shape_key:
            rr = int(r0 + dr)
            cc = int(c0 + dc)
            grid[rr, cc] = int(color)
            occupied[rr, cc] = True
            coords.append((rr, cc))
        return coords
    return None


def _build_expected(
    inp: np.ndarray,
    row_spans: list[tuple[int, int]],
    left_span: tuple[int, int],
    right_span: tuple[int, int],
    templates: list[tuple[tuple[int, int], ...]],
    object_info: list[dict],
    row_to_color: dict[int, int],
) -> np.ndarray:
    out = inp.copy()
    cell_w = int(left_span[1] - left_span[0] + 1)
    for obj in object_info:
        for r, c in obj["coords"]:
            out[int(r), int(c)] = BLACK
    for idx, row_span in enumerate(row_spans):
        if idx in row_to_color:
            coords = templates[idx]
            color = int(row_to_color[idx])
        else:
            coords = _mirror_in_cell(templates[idx], cell_w)
            color = MAROON
        _paint_cell(out, row_span, right_span, coords, color)
    return out


def _generate_case() -> np.ndarray | None:
    row_count = int(RNG.integers(3, 6))
    cell_h = int(RNG.integers(6, 9))
    cell_w = int(RNG.integers(6, 9))
    table_h = row_count * cell_h + row_count + 1
    table_w = 2 * cell_w + 3

    top_margin = int(RNG.integers(1, 4))
    bottom_margin = int(RNG.integers(1, 4))
    left_zone_w = int(RNG.integers(12, 19))
    gap = int(RNG.integers(2, 5))
    right_margin = int(RNG.integers(1, 4))

    height = int(table_h + top_margin + bottom_margin)
    width = int(left_zone_w + gap + table_w + right_margin)
    table_top = int(top_margin)
    table_left = int(left_zone_w + gap)

    inp = np.full((height, width), BLACK, dtype=int)

    for k in range(row_count + 1):
        rr = int(table_top + k * (cell_h + 1))
        inp[rr, table_left : table_left + table_w] = GREY
    for cc in (table_left, table_left + cell_w + 1, table_left + 2 * cell_w + 2):
        inp[table_top : table_top + table_h, cc] = GREY

    row_spans = [
        (int(table_top + 1 + k * (cell_h + 1)), int(table_top + cell_h + k * (cell_h + 1)))
        for k in range(row_count)
    ]
    left_span = (int(table_left + 1), int(table_left + cell_w))
    right_span = (int(table_left + cell_w + 2), int(table_left + 2 * cell_w + 1))

    templates: list[tuple[tuple[int, int], ...]] = []
    used_keys: set[tuple[tuple[int, int], ...]] = set()
    used_mirrors: set[tuple[tuple[int, int], ...]] = set()
    for row_span in row_spans:
        _r0, _r1 = row_span
        for _ in range(400):
            area = int(RNG.integers(3, 9))
            shape_key = _random_connected_shape(area, cell_h - 2, cell_w - 2)
            mirror_key = _mirror_normalized(shape_key)
            if shape_key in used_keys or shape_key in used_mirrors or mirror_key in used_keys:
                continue
            h, w = _norm_dims(shape_key)
            off_r = int(RNG.integers(1, cell_h - h))
            off_c = int(RNG.integers(1, cell_w - w))
            cell_coords = tuple(sorted((int(r + off_r), int(c + off_c)) for r, c in shape_key))
            templates.append(cell_coords)
            used_keys.add(shape_key)
            used_mirrors.add(mirror_key)
            for dr, dc in cell_coords:
                inp[int(row_span[0] + dr), int(left_span[0] + dc)] = AZURE
            break
        else:
            return None

    has_missing = bool(RNG.integers(0, 2))
    if has_missing:
        missing_count = int(RNG.integers(1, min(2, row_count - 1) + 1))
    else:
        missing_count = 0
    object_rows = sorted(RNG.choice(row_count, size=row_count - missing_count, replace=False).tolist())
    colors = RNG.choice(np.array(OBJECT_COLORS, dtype=int), size=len(object_rows), replace=False).tolist()

    occupied = np.zeros_like(inp, dtype=bool)
    object_info: list[dict] = []
    row_to_color: dict[int, int] = {}
    for row_idx, color in zip(object_rows, colors):
        shape_key = _normalize(templates[row_idx])
        coords = _place_object(inp, occupied, shape_key, int(color), table_left)
        if coords is None:
            return None
        object_info.append({"coords": coords, "color": int(color)})
        row_to_color[int(row_idx)] = int(color)

    expected = _build_expected(inp, row_spans, left_span, right_span, templates, object_info, row_to_color)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input for puzzle 4")
