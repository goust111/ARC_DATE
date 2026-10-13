from __future__ import annotations

import itertools
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components_for_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = np.asarray(grid == int(color), dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _comb_info(coords: np.ndarray) -> dict | None:
    if coords.shape[0] < 2:
        return None

    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    if c1 - c0 != 2:
        return None

    height = r1 - r0 + 1
    patch = np.zeros((height, 3), dtype=bool)
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = True

    row_types: list[str] = []
    for row in patch:
        solid = np.array([True, True, True])
        open_row = np.array([True, False, True])
        if np.array_equal(row, solid):
            row_types.append("solid")
        elif np.array_equal(row, open_row):
            row_types.append("open")
        else:
            return None

    for i in range(1, len(row_types)):
        if row_types[i] == row_types[i - 1]:
            return None

    solid_rows = sum(kind == "solid" for kind in row_types)
    if not (1 <= solid_rows <= 5):
        return None

    return {
        "n": int(solid_rows),
        "bbox": (r0, c0, r1, c1),
        "coords": {tuple(map(int, cell)) for cell in coords.tolist()},
        "start_open": row_types[0] == "open",
        "end_open": row_types[-1] == "open",
        "height": int(height),
    }


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = sorted(int(color) for color in np.unique(g) if int(color) != bg)
    assert colors, "expected at least one foreground color"

    records: list[dict] = []
    for color in colors:
        comps = _components_for_color(g, color)
        assert len(comps) == 2, "each color must form exactly one marker and one comb component"

        markers = [comp for comp in comps if comp.shape[0] == 1]
        combs = [info for comp in comps if (info := _comb_info(comp)) is not None]
        assert len(markers) == 1, "each color must have exactly one singleton marker"
        assert len(combs) == 1, "each color must have exactly one valid comb"

        marker_r, marker_c = map(int, markers[0][0])
        records.append(
            {
                "color": int(color),
                "marker": (marker_r, marker_c),
                "n": int(combs[0]["n"]),
                "comb_coords": set(combs[0]["coords"]),
                "comb_bbox": combs[0]["bbox"],
            }
        )

    marker_rows = [record["marker"][0] for record in records]
    assert len(marker_rows) == len(set(marker_rows)), "marker rows must be unique"
    records.sort(key=lambda record: record["marker"][0])
    return bg, records


def _path_cells(shape: tuple[int, int], marker: tuple[int, int], n: int) -> tuple[set[tuple[int, int]], int]:
    height, width = map(int, shape)
    r, c = map(int, marker)
    end_c = min(c + int(n), width - 1)
    end_r = min(r + int(n), height - 1)

    cells = {(r, cc) for cc in range(c, end_c + 1)}
    cells.update((rr, end_c) for rr in range(r + 1, end_r + 1))
    return cells, end_c


def _draw_paths(base: np.ndarray, records: list[dict]) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    height, width = out.shape
    for record in records:
        r, c = record["marker"]
        n = int(record["n"])
        color = int(record["color"])
        end_c = min(c + n, width - 1)
        out[r, c : end_c + 1] = color
        end_r = min(r + n, height - 1)
        if end_r > r:
            out[r + 1 : end_r + 1, end_c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each colored comb as a length parameter, then replace the combs with matching L-shaped rays from the markers.

    Input structure:
    - The grid has a black background, several single-pixel colored markers on the left, and one disconnected three-column comb per marker color on the right.
    - Every comb alternates solid three-pixel rows with open side-pair rows in a fixed three-column span.
    - Marker rows are distinct, and the combs do not overlap each other.

    Rule:
    - For each marker, find the comb of the same color and count how many solid three-pixel rows it contains; call that count N.
    - Erase every comb from the grid.
    - In top-to-bottom marker order, extend each marker horizontally to the right by N cells (clipped by the right border), then from that horizontal endpoint extend downward by N cells (clipped by the bottom border).
    - Keep the original marker as the start of its ray; if later rays overlap earlier ones, later colors overwrite earlier ones.

    Key insight:
    - The combs are not output objects: they are color-matched numeric encoders whose solid-row counts control the lengths of remote L-shaped rays.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, records = _parse_scene(g)
    out = g.copy()
    for record in records:
        for r, c in record["comb_coords"]:
            out[r, c] = bg
    out = _draw_paths(out, records)
    return out


COLOR_POOL = tuple(range(1, 10))
COUNT_OPTIONS = (3, 4)
WIDTH_RANGE = (15, 18)
MIN_HEIGHT = 15
MAX_HEIGHT = 30
MAX_GENERATION_ATTEMPTS = 800


def _make_record(color: int, n: int, start_open: bool, end_open: bool) -> dict:
    assert n >= 1
    return {
        "color": int(color),
        "n": int(n),
        "start_open": bool(start_open),
        "end_open": bool(end_open),
    }


def _comb_height(record: dict) -> int:
    return 2 * int(record["n"]) - 1 + int(record["start_open"]) + int(record["end_open"])


def _render_comb(grid: np.ndarray, top: int, left: int, record: dict) -> set[tuple[int, int]]:
    color = int(record["color"])
    n = int(record["n"])
    row = int(top)
    cells: set[tuple[int, int]] = set()

    def paint_open(rr: int) -> None:
        grid[rr, left] = color
        grid[rr, left + 2] = color
        cells.add((rr, left))
        cells.add((rr, left + 2))

    def paint_solid(rr: int) -> None:
        grid[rr, left : left + 3] = color
        for cc in range(left, left + 3):
            cells.add((rr, cc))

    if record["start_open"]:
        paint_open(row)
        row += 1

    for idx in range(n):
        paint_solid(row)
        row += 1
        if idx != n - 1 or record["end_open"]:
            paint_open(row)
            row += 1

    return cells


def _choose_records() -> list[dict] | None:
    count = random.choice(COUNT_OPTIONS)
    colors = random.sample(COLOR_POOL, count)
    ns = random.sample(range(1, 6), count)

    records: list[dict] = []
    for color, n in zip(colors, ns):
        options = [(False, False), (False, True), (True, False), (True, True)]
        if n == 1:
            options = [option for option in options if option != (False, False)]
        start_open, end_open = random.choice(options)
        records.append(_make_record(color, n, start_open, end_open))

    extra_counts = {int(record["start_open"]) + int(record["end_open"]) for record in records}
    if len(extra_counts) < 2:
        return None
    if not any(record["start_open"] for record in records):
        return None
    if all(record["start_open"] for record in records):
        return None
    return records


def _choose_marker_rows(records: list[dict]) -> tuple[list[dict], list[int], int]:
    ordered = random.sample(records, len(records))
    rows = [random.randint(1, 3)]
    for record in ordered[:-1]:
        rows.append(rows[-1] + int(record["n"]) + random.randint(1, 3))

    last_n = int(ordered[-1]["n"])
    clip_last = random.random() < 0.5
    if clip_last:
        clearance = random.randint(0, last_n - 1)
    else:
        clearance = random.randint(last_n, last_n + 2)
    required_height = rows[-1] + 1 + clearance
    return ordered, rows, required_height


def _choose_comb_rows(records: list[dict], height: int) -> tuple[list[dict], list[int]] | None:
    ordered = random.sample(records, len(records))
    heights = [_comb_height(record) for record in ordered]
    gaps = [random.randint(1, 2) for _ in range(len(ordered) - 1)]
    span = sum(heights) + sum(gaps)
    if span > height:
        return None

    top = random.randint(0, height - span)
    rows = [top]
    for comb_h, gap in zip(heights[:-1], gaps):
        rows.append(rows[-1] + comb_h + gap)
    return ordered, rows


def _choose_marker_cols(shape: tuple[int, int], ordered_records: list[dict], rows: list[int]) -> dict[int, tuple[int, int]] | None:
    candidates = list(itertools.product((0, 1, 2), repeat=len(ordered_records)))
    random.shuffle(candidates)
    for cols in candidates:
        used: set[tuple[int, int]] = set()
        placements: dict[int, tuple[int, int]] = {}
        ok = True
        for record, row, col in zip(ordered_records, rows, cols):
            placements[int(record["color"])] = (int(row), int(col))
            cells, _ = _path_cells(shape, (row, col), int(record["n"]))
            if used & cells:
                ok = False
                break
            used.update(cells)
        if ok:
            return placements
    return None


def _expected_from_records(shape: tuple[int, int], records: list[dict], marker_positions: dict[int, tuple[int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    marker_records: list[dict] = []
    for record in records:
        color = int(record["color"])
        r, c = marker_positions[color]
        grid[r, c] = color
        marker_records.append({"color": color, "marker": (r, c), "n": int(record["n"])})
    marker_records.sort(key=lambda record: record["marker"][0])
    return _draw_paths(grid, marker_records)


def _validate_intended_semantics(
    inp: np.ndarray,
    records: list[dict],
    marker_positions: dict[int, tuple[int, int]],
    expected: np.ndarray,
) -> bool:
    parsed_bg, parsed_records = _parse_scene(inp)
    if parsed_bg != 0:
        return False

    intended_n = {int(record["color"]): int(record["n"]) for record in records}
    intended_markers = {int(color): tuple(map(int, pos)) for color, pos in marker_positions.items()}
    parsed_n = {int(record["color"]): int(record["n"]) for record in parsed_records}
    parsed_markers = {int(record["color"]): tuple(map(int, record["marker"])) for record in parsed_records}
    if parsed_n != intended_n or parsed_markers != intended_markers:
        return False

    all_path_cells: set[tuple[int, int]] = set()
    all_comb_cells: set[tuple[int, int]] = set()
    for record in parsed_records:
        path_cells, _ = _path_cells(inp.shape, record["marker"], int(record["n"]))
        if all_path_cells & path_cells:
            return False
        all_path_cells.update(path_cells)
        if all_comb_cells & record["comb_coords"]:
            return False
        all_comb_cells.update(record["comb_coords"])

    if all_path_cells & all_comb_cells:
        return False

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    if np.array_equal(inp, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    records = _choose_records()
    if records is None:
        return None

    marker_order, marker_rows, marker_min_height = _choose_marker_rows(records)
    max_comb_span = sum(_comb_height(record) for record in records) + 2 * (len(records) - 1)
    min_height = max(MIN_HEIGHT, marker_min_height, max_comb_span)
    if min_height > MAX_HEIGHT:
        return None

    height = random.randint(min_height, MAX_HEIGHT)
    width = random.randint(*WIDTH_RANGE)

    comb_layout = _choose_comb_rows(records, height)
    if comb_layout is None:
        return None
    comb_order, comb_rows = comb_layout

    marker_positions = _choose_marker_cols((height, width), marker_order, marker_rows)
    if marker_positions is None:
        return None

    comb_left = random.randint(max(10, width - 6), width - 3)
    inp = np.zeros((height, width), dtype=int)
    expected = _expected_from_records(inp.shape, records, marker_positions)

    for color, (r, c) in marker_positions.items():
        inp[r, c] = int(color)

    for record, top in zip(comb_order, comb_rows):
        record["comb_coords"] = _render_comb(inp, top, comb_left, record)

    if not _validate_intended_semantics(inp, records, marker_positions, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
