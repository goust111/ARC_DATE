import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PATH_COLORS = [1, 2, 3, 4, 6, 7, 8]
NON_PATH = {BLACK, GREY, MAROON}
MAX_GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as coordinate arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if not _in_bounds(mask.shape, nr, nc):
                        continue
                    if seen[nr, nc] or not mask[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _find_nodes(grid: np.ndarray) -> dict[tuple[int, int], int]:
    nodes: dict[tuple[int, int], int] = {}
    h, w = grid.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            block = grid[r - 1 : r + 2, c - 1 : c + 2]
            if block.shape != (3, 3):
                continue
            center = int(block[1, 1])
            if center in NON_PATH:
                continue
            flat = block.reshape(-1).copy()
            ring = np.delete(flat, 4)
            if np.all(ring == GREY):
                nodes[(r, c)] = center
    return nodes


def _source_side_candidates(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    sides = {
        "up": [(r0 - 1, c0), (r0 - 1, c1)],
        "down": [(r1 + 1, c0), (r1 + 1, c1)],
        "left": [(r0, c0 - 1), (r1, c0 - 1)],
        "right": [(r0, c1 + 1), (r1, c1 + 1)],
    }
    found: dict[str, list[tuple[int, int]]] = {}
    for side, cells in sides.items():
        keep = []
        for r, c in cells:
            if _in_bounds(grid.shape, r, c) and int(grid[r, c]) not in NON_PATH:
                keep.append((r, c))
        if keep:
            found[side] = keep
    return found


def _parse_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2-D"

    nodes = _find_nodes(g)
    node_centers = set(nodes)

    sources = []
    path_dirs: dict[int, tuple[int, int]] = {}
    source_boxes: dict[int, tuple[int, int, int, int]] = {}
    source_starts: dict[int, tuple[int, int]] = {}

    for comp in _components(g == MAROON):
        assert comp.shape[0] == 4, "every maroon source must be a 2x2 block"
        r0, c0 = np.min(comp, axis=0)
        r1, c1 = np.max(comp, axis=0)
        bbox = (int(r0), int(c0), int(r1), int(c1))
        assert bbox[2] - bbox[0] == 1 and bbox[3] - bbox[1] == 1, "every source must be 2x2"

        side_cells = _source_side_candidates(g, bbox)
        assert len(side_cells) == 1, "each source must attach to exactly one path side"
        side, cells = next(iter(side_cells.items()))
        colors = {int(g[r, c]) for r, c in cells}
        assert len(colors) == 1, "a source may touch only one path color"
        color = next(iter(colors))
        assert color not in path_dirs, "path colors must be unique across sources"

        if side == "up":
            direction = (-1, 0)
        elif side == "down":
            direction = (1, 0)
        elif side == "left":
            direction = (0, -1)
        else:
            direction = (0, 1)

        start = cells[0]
        sources.append({"color": color, "bbox": bbox, "dir": direction, "start": start})
        path_dirs[color] = direction
        source_boxes[color] = bbox
        source_starts[color] = start

    assert sources, "expected at least one source"
    sources.sort(key=lambda source: (source["bbox"][0], source["bbox"][1]))
    for idx, source in enumerate(sources):
        source["order"] = idx

    for color, direction in path_dirs.items():
        coords = np.argwhere(g == color)
        visible = [(int(r), int(c)) for r, c in coords if (int(r), int(c)) not in node_centers]
        assert visible, "every path color must have visible pixels outside node centers"
        dr, dc = direction
        if dr == 0:
            rows = {r for r, _c in visible}
            assert len(rows) == 1, "horizontal paths must stay on one row"
            assert source_starts[color][0] in rows
        else:
            cols = {c for _r, c in visible}
            assert len(cols) == 1, "vertical paths must stay on one column"
            assert source_starts[color][1] in cols

    return {
        "grid": g,
        "nodes": nodes,
        "node_centers": node_centers,
        "sources": sources,
        "path_dirs": path_dirs,
        "source_boxes": source_boxes,
    }


def _trace_signal(structure: dict, source: dict) -> dict:
    g = structure["grid"]
    nodes = structure["nodes"]
    path_dirs = structure["path_dirs"]

    origin_color = int(source["color"])
    cur_color = origin_color
    direction = tuple(source["dir"])
    pos = tuple(source["start"])
    switches = 0
    stays = 0
    steps = 0
    seen: set[tuple[int, int, int, int, int]] = set()

    for _ in range(g.size * 4):
        state = (pos[0], pos[1], cur_color, direction[0], direction[1])
        assert state not in seen, "signal path must not cycle"
        seen.add(state)

        nr, nc = pos[0] + direction[0], pos[1] + direction[1]
        if not _in_bounds(g.shape, nr, nc):
            return {
                "origin_color": origin_color,
                "final_path_color": cur_color,
                "terminal": pos,
                "switches": switches,
                "stays": stays,
                "steps": steps,
                "order": int(source["order"]),
            }

        nxt = int(g[nr, nc])
        if nxt == cur_color:
            pos = (nr, nc)
            steps += 1
            continue

        if nxt == GREY:
            center = (pos[0] + 2 * direction[0], pos[1] + 2 * direction[1])
            assert center in nodes, "grey barrier must belong to a routing node"
            next_color = int(nodes[center])
            next_dir = tuple(path_dirs[next_color])
            exit_pos = (center[0] + 2 * next_dir[0], center[1] + 2 * next_dir[1])
            assert _in_bounds(g.shape, exit_pos[0], exit_pos[1]), "node exit must stay in bounds"
            assert int(g[exit_pos]) == next_color, "node route color must point to an incident path"
            if next_color == cur_color:
                stays += 1
            else:
                switches += 1
            pos = exit_pos
            cur_color = next_color
            direction = next_dir
            steps += 1
            continue

        if nxt in (BLACK, MAROON):
            return {
                "origin_color": origin_color,
                "final_path_color": cur_color,
                "terminal": pos,
                "switches": switches,
                "stays": stays,
                "steps": steps,
                "order": int(source["order"]),
            }

        raise AssertionError("signals may only see their own path, a node, or black")

    raise AssertionError("signal exceeded step budget")


def _simulate(structure: dict) -> list[dict]:
    return [_trace_signal(structure, source) for source in structure["sources"]]


def _transform(grid: np.ndarray, *, return_meta: bool = False):
    structure = _parse_structure(grid)
    traces = _simulate(structure)
    out = structure["grid"].copy()

    for source in structure["sources"]:
        r0, c0, r1, c1 = source["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(source["color"])

    winners: dict[tuple[int, int], dict] = {}
    for trace in traces:
        terminal = tuple(trace["terminal"])
        prev = winners.get(terminal)
        key = (int(trace["steps"]), int(trace["order"]))
        if prev is None or key < (int(prev["steps"]), int(prev["order"])):
            winners[terminal] = trace

    for trace in winners.values():
        tr, tc = trace["terminal"]
        out[tr, tc] = int(trace["origin_color"])

    if return_meta:
        return out, traces, structure, winners
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route signals through a colored path network and recolor each claimed terminal with the originating source color.

    Input structure:
    - Background is black.
    - Several one-pixel-wide colored paths run horizontally or vertically; each path color is unique and each path has a 2x2 maroon source block on one end.
    - When a horizontal and vertical path cross, the crossing is replaced by a 3x3 node whose outer ring is grey and whose center pixel is one of the incident path colors.
    - The opposite end of every path is a single colored terminal pixel followed by black background.

    Rule:
    - Replace every maroon source block with the color of its adjacent path.
    - Each path has a forward direction from its maroon source toward its terminal.
    - Launch one signal from every source and follow the current path in that forward direction.
    - When a signal reaches a node, read the node-center color and continue along the path of that color, again following that chosen path from its own source side toward its terminal.
    - When a signal reaches a terminal, recolor that terminal pixel with the signal's original source-path color; if several signals target the same terminal, the shorter routed path claims it.
    - All other path pixels and node pixels remain unchanged.

    Key insight:
    - The maroon source fixes each path's travel direction, while the node center color chooses which directed path the signal should follow next.
    """
    return _transform(grid)


def _sample_positions(count: int, limit: int) -> list[int] | None:
    candidates = list(range(4, limit - 4))
    if len(candidates) < count:
        return None
    for _ in range(200):
        picks = sorted(random.sample(candidates, count))
        if all(b - a >= 4 for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _build_grid(
    h_rows: list[int],
    v_cols: list[int],
    h_colors: list[int],
    v_colors: list[int],
    h_dir: int,
    v_dir: int,
    routes: dict[tuple[int, int], int],
    shape: tuple[int, int],
) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BLACK, dtype=int)

    for row, color in zip(h_rows, h_colors):
        if h_dir > 0:
            grid[row : row + 2, 0:2] = MAROON
            grid[row, 2 : w - 1] = color
        else:
            grid[row : row + 2, w - 2 : w] = MAROON
            grid[row, 1 : w - 2] = color

    for col, color in zip(v_cols, v_colors):
        if v_dir > 0:
            grid[0:2, col : col + 2] = MAROON
            grid[2 : h - 1, col] = color
        else:
            grid[h - 2 : h, col : col + 2] = MAROON
            grid[1 : h - 2, col] = color

    for row, h_color in zip(h_rows, h_colors):
        for col, v_color in zip(v_cols, v_colors):
            grid[row - 1 : row + 2, col - 1 : col + 2] = GREY
            route = routes[(row, col)]
            assert route in (h_color, v_color)
            grid[row, col] = route

    return grid


def _gen_attempt() -> np.ndarray | None:
    n_h = random.choice([2, 3])
    n_v = random.choice([2, 3])
    if n_h + n_v > len(PATH_COLORS):
        return None

    height = random.randint(18, 30)
    width = random.randint(18, 30)
    h_rows = _sample_positions(n_h, height)
    v_cols = _sample_positions(n_v, width)
    if h_rows is None or v_cols is None:
        return None

    colors = random.sample(PATH_COLORS, n_h + n_v)
    h_colors = colors[:n_h]
    v_colors = colors[n_h:]
    h_dir = random.choice([-1, 1])
    v_dir = random.choice([-1, 1])

    want_more_changes = random.choice([False, True])

    for _ in range(200):
        routes: dict[tuple[int, int], int] = {}
        for row, h_color in zip(h_rows, h_colors):
            for col, v_color in zip(v_cols, v_colors):
                routes[(row, col)] = random.choice([h_color, v_color])

        grid = _build_grid(h_rows, v_cols, h_colors, v_colors, h_dir, v_dir, routes, (height, width))

        try:
            out, traces, _structure, winners = _transform(grid, return_meta=True)
        except AssertionError:
            continue

        per_terminal: dict[tuple[int, int], list[dict]] = {}
        for trace in traces:
            per_terminal.setdefault(tuple(trace["terminal"]), []).append(trace)
        ambiguous = False
        for group in per_terminal.values():
            group.sort(key=lambda trace: (int(trace["steps"]), int(trace["order"])))
            if len(group) > 1 and int(group[0]["steps"]) == int(group[1]["steps"]):
                ambiguous = True
                break
        if ambiguous:
            continue

        total_switches = sum(int(trace["switches"]) for trace in traces)
        total_stays = sum(int(trace["stays"]) for trace in traces)
        changed = sum(int(trace["origin_color"] != trace["final_path_color"]) for trace in traces)
        unchanged = len(traces) - changed
        lit_count = len(winners)
        dark_count = len(traces) - lit_count

        if total_switches == 0 or total_stays == 0:
            continue
        if changed == 0 or unchanged == 0:
            continue
        if want_more_changes and changed < 2:
            continue
        if dark_count == 0:
            continue
        if np.array_equal(grid, out):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid routing-network input")
