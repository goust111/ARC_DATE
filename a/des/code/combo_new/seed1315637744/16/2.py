from __future__ import annotations

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _extract_lines(grid: np.ndarray, bg: int) -> dict[int, dict[str, int | str]]:
    lines: dict[int, dict[str, int | str]] = {}
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        coords = np.argwhere(grid == color)
        assert coords.size > 0, f"missing visible cells for color {color}"

        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])

        if len(rows) == 1 and len(cols) >= 2:
            row = int(rows[0])
            c0 = int(cols.min())
            c1 = int(cols.max())
            assert np.all(grid[row, c0 : c1 + 1] != bg), "horizontal line cannot have background gaps"
            lines[color] = {
                "color": color,
                "orientation": "H",
                "fixed": row,
                "start": c0,
                "end": c1,
            }
        elif len(cols) == 1 and len(rows) >= 2:
            col = int(cols[0])
            r0 = int(rows.min())
            r1 = int(rows.max())
            assert np.all(grid[r0 : r1 + 1, col] != bg), "vertical line cannot have background gaps"
            lines[color] = {
                "color": color,
                "orientation": "V",
                "fixed": col,
                "start": r0,
                "end": r1,
            }
        else:
            raise AssertionError(f"color {color} does not form a single visible row/column trace")

    return lines


def _crossing_edges(grid: np.ndarray, lines: dict[int, dict[str, int | str]]) -> set[tuple[int, int]]:
    horizontals = [line for line in lines.values() if line["orientation"] == "H"]
    verticals = [line for line in lines.values() if line["orientation"] == "V"]

    edges: set[tuple[int, int]] = set()
    crossing_counts = {color: 0 for color in lines}

    for h in horizontals:
        row = int(h["fixed"])
        c0 = int(h["start"])
        c1 = int(h["end"])
        h_color = int(h["color"])

        for v in verticals:
            col = int(v["fixed"])
            r0 = int(v["start"])
            r1 = int(v["end"])
            v_color = int(v["color"])

            if not (c0 <= col <= c1 and r0 <= row <= r1):
                continue

            visible = int(grid[row, col])
            crossing_counts[h_color] += 1
            crossing_counts[v_color] += 1

            if visible == h_color:
                edges.add((v_color, h_color))
            elif visible == v_color:
                edges.add((h_color, v_color))
            else:
                raise AssertionError("a crossing must show one of the two participating line colors")

    assert edges, "expected at least one crossing"
    assert all(count > 0 for count in crossing_counts.values()), "every line must participate in a crossing"
    return edges


def _unique_topological_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    outgoing = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}

    for lower, upper in edges:
        if upper not in outgoing[lower]:
            outgoing[lower].add(upper)
            indegree[upper] += 1

    order: list[int] = []
    ready = sorted(color for color in colors if indegree[color] == 0)

    while ready:
        assert len(ready) == 1, "crossing constraints must determine a unique total order"
        color = ready.pop()
        order.append(color)
        new_ready: list[int] = []
        for nxt in sorted(outgoing[color]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                new_ready.append(nxt)
        ready = sorted(ready + new_ready)

    assert len(order) == len(colors), "crossing constraints must be acyclic and cover all lines"
    return order


def _paint_line(out: np.ndarray, line: dict[str, int | str]) -> None:
    color = int(line["color"])
    if line["orientation"] == "H":
        row = int(line["fixed"])
        out[row, int(line["start"]) : int(line["end"]) + 1] = color
    else:
        col = int(line["fixed"])
        out[int(line["start"]) : int(line["end"]) + 1, col] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Keep only the globally topmost woven line by reconstructing hidden line extents and crossing order.

    Input structure:
    - A dominant background surrounds several one-cell-wide monochrome horizontal or vertical line segments.
    - Each foreground color belongs to exactly one line segment, and crossings may hide one line beneath another.
    - The visible cells of any one color still lie on a single row or a single column, with visible endpoints bounding the full segment.

    Rule:
    - Reconstruct each line's full span from its visible row/column and its extreme visible cells, including crossing cells hidden by other colors.
    - At every horizontal/vertical crossing, treat the visible color as the upper line and the other participating line as lower, creating a partial depth order.
    - Topologically sort these relations; the puzzle guarantees one unique full front-to-back order.
    - Output only the single line at the front of that order, erasing every other line back to the background color.

    Key insight:
    - The topmost line is often not identified by a single local crossing; it must be inferred globally by propagating above/below relations through the whole weaving graph.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _get_bg(g)
    lines = _extract_lines(g, bg)
    edges = _crossing_edges(g, lines)
    order = _unique_topological_order(sorted(lines), edges)

    out = np.full_like(g, bg)
    _paint_line(out, lines[order[-1]])
    return out


# --------------- generator (att1) ---------------

_RNG = np.random.default_rng()

MIN_LINE_COUNT = 3
MAX_LINE_COUNT_EXCLUSIVE = 9
COLOR_DOMAIN = np.arange(10, dtype=int)
POSITION_BASE_RANGE = (2, 5)
POSITION_GAP_RANGE = (3, 7)
POSITION_TAIL_RANGE = (3, 6)
END_PAD_RANGE = (1, 3)
SYMMETRY_COUNT = 8
MAX_GENERATION_ATTEMPTS = 400


def _axis_positions(count: int) -> tuple[list[int], int]:
    start = int(_RNG.integers(*POSITION_BASE_RANGE))
    positions = [start]
    for _ in range(1, count):
        positions.append(positions[-1] + int(_RNG.integers(*POSITION_GAP_RANGE)))
    size = positions[-1] + int(_RNG.integers(*POSITION_TAIL_RANGE)) + 1
    return positions, size


def _apply_symmetry(grid: np.ndarray, code: int) -> np.ndarray:
    out = grid
    if code >= 4:
        out = out.T
    if code & 1:
        out = np.flipud(out)
    if code & 2:
        out = np.fliplr(out)
    return np.array(out, copy=True)


def _build_chain_scene() -> tuple[np.ndarray, np.ndarray] | None:
    line_count = int(_RNG.integers(MIN_LINE_COUNT, MAX_LINE_COUNT_EXCLUSIVE))
    start_horizontal = bool(int(_RNG.integers(0, 2)))
    orientations = ["H" if (start_horizontal == (i % 2 == 0)) else "V" for i in range(line_count)]

    h_count = sum(ori == "H" for ori in orientations)
    v_count = line_count - h_count
    rows, height = _axis_positions(h_count)
    cols, width = _axis_positions(v_count)
    if max(height, width) > 30:
        return None

    bg = int(_RNG.choice(COLOR_DOMAIN))
    line_colors = _RNG.choice(COLOR_DOMAIN[COLOR_DOMAIN != bg], size=line_count, replace=False)

    lines: list[dict[str, int | str]] = []
    h_idx = 0
    v_idx = 0
    for color, orientation in zip(line_colors.tolist(), orientations):
        if orientation == "H":
            fixed = rows[h_idx]
            h_idx += 1
        else:
            fixed = cols[v_idx]
            v_idx += 1
        lines.append({"color": int(color), "orientation": orientation, "fixed": int(fixed)})

    for i, line in enumerate(lines):
        if line["orientation"] == "H":
            neighbor_coords = []
            if i > 0:
                neighbor_coords.append(int(lines[i - 1]["fixed"]))
            if i + 1 < line_count:
                neighbor_coords.append(int(lines[i + 1]["fixed"]))
            left_pad = int(_RNG.integers(*END_PAD_RANGE))
            right_pad = int(_RNG.integers(*END_PAD_RANGE))
            line["start"] = min(neighbor_coords) - left_pad
            line["end"] = max(neighbor_coords) + right_pad
            if not (0 <= int(line["start"]) < int(line["end"]) < width):
                return None
        else:
            neighbor_coords = []
            if i > 0:
                neighbor_coords.append(int(lines[i - 1]["fixed"]))
            if i + 1 < line_count:
                neighbor_coords.append(int(lines[i + 1]["fixed"]))
            top_pad = int(_RNG.integers(*END_PAD_RANGE))
            bottom_pad = int(_RNG.integers(*END_PAD_RANGE))
            line["start"] = min(neighbor_coords) - top_pad
            line["end"] = max(neighbor_coords) + bottom_pad
            if not (0 <= int(line["start"]) < int(line["end"]) < height):
                return None

    inp = np.full((height, width), bg, dtype=int)
    for line in lines:
        _paint_line(inp, line)

    out = np.full_like(inp, bg)
    _paint_line(out, lines[-1])

    symmetry = int(_RNG.integers(0, SYMMETRY_COUNT))
    inp = _apply_symmetry(inp, symmetry)
    out = _apply_symmetry(out, symmetry)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_chain_scene()
        if built is None:
            continue
        inp, expected = built
        try:
            if np.array_equal(main(inp), expected):
                return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate valid woven-line input")
