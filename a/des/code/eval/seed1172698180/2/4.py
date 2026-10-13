import random

import numpy as np


BODY_START_ROW = 2


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _marker_relations(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    assert grid.shape[0] > BODY_START_ROW
    relations: list[tuple[int, int]] = []
    for c in range(grid.shape[1]):
        top = int(grid[0, c])
        bottom = int(grid[1, c])
        if top == bg and bottom == bg:
            continue
        assert top != bg and bottom != bg and top != bottom
        relations.append((top, bottom))
    assert relations, "expected at least one priority marker"
    return relations


def _reachability(colors: set[int], relations: list[tuple[int, int]]) -> dict[int, set[int]]:
    adj = {color: set() for color in colors}
    indeg = {color: 0 for color in colors}
    for higher, lower in relations:
        assert higher in colors and lower in colors
        if lower not in adj[higher]:
            adj[higher].add(lower)
            indeg[lower] += 1

    frontier = sorted(color for color, deg in indeg.items() if deg == 0)
    topo: list[int] = []
    while frontier:
        color = frontier.pop(0)
        topo.append(color)
        for nxt in sorted(adj[color]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                frontier.append(nxt)
                frontier.sort()
    assert len(topo) == len(colors), "priority markers must form an acyclic order"

    reach = {color: set() for color in colors}
    for color in reversed(topo):
        for lower in adj[color]:
            reach[color].add(lower)
            reach[color].update(reach[lower])
    return reach


def _higher_color(a: int, b: int, reach: dict[int, set[int]]) -> int:
    a_over_b = b in reach[a]
    b_over_a = a in reach[b]
    assert a_over_b != b_over_a, "crossing colors must be comparable"
    return a if a_over_b else b


def _extract_paths(grid: np.ndarray, bg: int) -> dict[int, dict]:
    body = grid[BODY_START_ROW:, :]
    path_colors = [int(color) for color in np.unique(body) if int(color) != bg]
    assert path_colors, "expected at least one path"

    paths: dict[int, dict] = {}
    for color in path_colors:
        coords = np.argwhere(body == color)
        assert coords.size > 0
        coords[:, 0] += BODY_START_ROW

        rows = sorted({int(r) for r, _c in coords})
        cols = sorted({int(c) for _r, c in coords})
        visible = {(int(r), int(c)) for r, c in coords}

        if len(rows) == 1 and len(cols) > 1:
            row = rows[0]
            paths[color] = {
                "color": color,
                "orientation": "h",
                "row": row,
                "c0": cols[0],
                "c1": cols[-1],
                "visible": visible,
            }
        elif len(cols) == 1 and len(rows) > 1:
            col = cols[0]
            paths[color] = {
                "color": color,
                "orientation": "v",
                "col": col,
                "r0": rows[0],
                "r1": rows[-1],
                "visible": visible,
            }
        else:
            raise AssertionError("each path color must form a straight 1-pixel-wide segment")
    return paths


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    relations = _marker_relations(g, bg)
    paths = _extract_paths(g, bg)

    path_colors = set(paths)
    marker_colors = {color for edge in relations for color in edge}
    assert path_colors == marker_colors, "markers must describe exactly the path colors"

    reach = _reachability(path_colors, relations)
    ordered_colors = sorted(path_colors)
    for i, a in enumerate(ordered_colors):
        for b in ordered_colors[i + 1 :]:
            _ = _higher_color(a, b, reach)

    horizontals = [path for path in paths.values() if path["orientation"] == "h"]
    verticals = [path for path in paths.values() if path["orientation"] == "v"]
    assert horizontals and verticals, "expected both horizontal and vertical paths"

    crossings = {color: 0 for color in path_colors}
    visible_crossings = {color: 0 for color in path_colors}
    for h_path in horizontals:
        row = h_path["row"]
        for v_path in verticals:
            col = v_path["col"]
            if not (h_path["c0"] <= col <= h_path["c1"]):
                continue
            if not (v_path["r0"] <= row <= v_path["r1"]):
                continue

            shown = int(g[row, col])
            expected = _higher_color(h_path["color"], v_path["color"], reach)
            assert shown in {h_path["color"], v_path["color"]}
            assert shown == expected, "crossing visibility must match the priority markers"

            crossings[h_path["color"]] += 1
            crossings[v_path["color"]] += 1
            visible_crossings[shown] += 1

    to_erase = {
        color
        for color in path_colors
        if crossings[color] > 0 and visible_crossings[color] == 0
    }

    out = g.copy()
    for color in to_erase:
        for r, c in paths[color]["visible"]:
            out[r, c] = bg
    return out


GEN_BG = 0
GEN_PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GRID_H_RANGE = (11, 18)
GRID_W_RANGE = (15, 22)
PATH_COUNT_OPTIONS = [(2, 2), (2, 3), (3, 2), (3, 3)]
POSITION_GAP = 2
H_ROW_RANGE_PAD = 3
V_COL_RANGE_PAD = 2
PATH_END_PAD_RANGE = (1, 3)
VERT_PATH_END_PAD_RANGE = (1, 2)
MARKER_GAP_OPTIONS = [1, 2]
ORDER_MODES = ["grouped", "interleaved"]
EXTRA_RELATION_MAX = 2
GEN_RETRIES = 400


def _sample_spaced_positions(
    count: int,
    low: int,
    high: int,
    min_gap: int,
    rng: random.Random,
) -> list[int] | None:
    choices = list(range(low, high + 1))
    if len(choices) < count:
        return None
    for _ in range(200):
        picked = sorted(rng.sample(choices, count))
        if all(b - a >= min_gap for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _priority_order(
    h_colors: list[int],
    v_colors: list[int],
    mode: str,
    rng: random.Random,
) -> list[int]:
    hs = h_colors[:]
    vs = v_colors[:]
    rng.shuffle(hs)
    rng.shuffle(vs)

    if mode == "grouped":
        if rng.choice([False, True]):
            return hs + vs
        return vs + hs

    order: list[int] = []
    turn = rng.choice(["h", "v"])
    while hs or vs:
        if turn == "h" and hs:
            order.append(hs.pop())
        elif turn == "v" and vs:
            order.append(vs.pop())
        elif hs:
            order.append(hs.pop())
        else:
            order.append(vs.pop())
        turn = "v" if turn == "h" else "h"
    return order


def _marker_edge_list(priority: list[int], rng: random.Random) -> list[tuple[int, int]]:
    edges = [(priority[i], priority[i + 1]) for i in range(len(priority) - 1)]
    extras = [
        (priority[i], priority[j])
        for i in range(len(priority))
        for j in range(i + 2, len(priority))
    ]
    if extras:
        extra_count = rng.randint(0, min(EXTRA_RELATION_MAX, len(extras)))
        edges.extend(rng.sample(extras, extra_count))
    return edges


def _draw_segment(grid: np.ndarray, color: int, spec: dict) -> None:
    if spec["orientation"] == "h":
        grid[spec["row"], spec["c0"] : spec["c1"] + 1] = color
    else:
        grid[spec["r0"] : spec["r1"] + 1, spec["col"]] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h_count, v_count = rng.choice(PATH_COUNT_OPTIONS)
    total = h_count + v_count

    height = rng.randint(*GRID_H_RANGE)
    width = rng.randint(*GRID_W_RANGE)

    colors = rng.sample(GEN_PALETTE, total)
    rng.shuffle(colors)
    h_colors = colors[:h_count]
    v_colors = colors[h_count:]

    priority = _priority_order(h_colors, v_colors, rng.choice(ORDER_MODES), rng)
    marker_edges = _marker_edge_list(priority, rng)

    marker_gap = rng.choice(MARKER_GAP_OPTIONS)
    marker_span = 1 + (len(marker_edges) - 1) * (marker_gap + 1)
    if marker_span > width:
        return None

    row_low = BODY_START_ROW + H_ROW_RANGE_PAD
    row_high = height - H_ROW_RANGE_PAD
    col_low = V_COL_RANGE_PAD
    col_high = width - 1 - V_COL_RANGE_PAD
    rows = _sample_spaced_positions(h_count, row_low, row_high, POSITION_GAP, rng)
    cols = _sample_spaced_positions(v_count, col_low, col_high, POSITION_GAP, rng)
    if rows is None or cols is None:
        return None

    row_min, row_max = min(rows), max(rows)
    col_min, col_max = min(cols), max(cols)

    grid = np.full((height, width), GEN_BG, dtype=int)

    marker_start = rng.randint(0, width - marker_span)
    marker_cols = [marker_start + i * (marker_gap + 1) for i in range(len(marker_edges))]
    for c, (higher, lower) in zip(marker_cols, marker_edges):
        grid[0, c] = higher
        grid[1, c] = lower

    specs: dict[int, dict] = {}
    for color, row in zip(h_colors, rows):
        left_room = min(PATH_END_PAD_RANGE[1], col_min)
        right_room = min(PATH_END_PAD_RANGE[1], width - 1 - col_max)
        if left_room < PATH_END_PAD_RANGE[0] or right_room < PATH_END_PAD_RANGE[0]:
            return None
        left_pad = rng.randint(PATH_END_PAD_RANGE[0], left_room)
        right_pad = rng.randint(PATH_END_PAD_RANGE[0], right_room)
        specs[color] = {
            "orientation": "h",
            "row": row,
            "c0": col_min - left_pad,
            "c1": col_max + right_pad,
        }

    for color, col in zip(v_colors, cols):
        top_room = min(VERT_PATH_END_PAD_RANGE[1], row_min - BODY_START_ROW)
        bottom_room = min(VERT_PATH_END_PAD_RANGE[1], height - 1 - row_max)
        if top_room < VERT_PATH_END_PAD_RANGE[0] or bottom_room < VERT_PATH_END_PAD_RANGE[0]:
            return None
        top_pad = rng.randint(VERT_PATH_END_PAD_RANGE[0], top_room)
        bottom_pad = rng.randint(VERT_PATH_END_PAD_RANGE[0], bottom_room)
        specs[color] = {
            "orientation": "v",
            "col": col,
            "r0": row_min - top_pad,
            "r1": row_max + bottom_pad,
        }

    rank = {color: i for i, color in enumerate(priority)}
    for color in sorted(priority, key=lambda value: rank[value], reverse=True):
        _draw_segment(grid, color, specs[color])

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if not np.array_equal(out[:BODY_START_ROW, :], grid[:BODY_START_ROW, :]):
        return None

    changed = (out != grid) & (grid != GEN_BG)
    removed_colors = {int(grid[r, c]) for r, c in np.argwhere(changed)}
    if not removed_colors or len(removed_colors) >= total:
        return None

    body_colors = {int(color) for color in np.unique(out[BODY_START_ROW:, :]) if int(color) != GEN_BG}
    if not (body_colors & (set(colors) - removed_colors)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
