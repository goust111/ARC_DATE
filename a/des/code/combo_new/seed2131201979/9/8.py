from __future__ import annotations

import random

import numpy as np


START_COLOR = 2  # red
END_COLOR = 1  # blue
BG_COLOR = 0
RELAY_COLOR_POOL = (3, 4, 5, 6, 7, 8, 9)

MIN_RELAYS = 1
MAX_RELAYS = 5
STEP_MIN = 4
STEP_MAX = 8
END_DIST_MIN = 2
END_DIST_MAX = 6
PADDING_TOTAL_MIN = 2
PADDING_TOTAL_MAX = 8
GEN_RETRIES = 600


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _make_marker(kind: str, color: int, coord: tuple[int, int]) -> dict[str, int | str]:
    r, c = map(int, coord)
    return {
        "kind": kind,
        "color": int(color),
        "r0": r,
        "r1": r,
        "c0": c,
        "c1": c,
    }


def _make_relay(color: int, coords: np.ndarray) -> dict[str, int | str]:
    assert coords.shape == (4, 2), "each relay must occupy four cells"
    rows = np.sort(coords[:, 0].astype(int))
    cols = np.sort(coords[:, 1].astype(int))
    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])
    assert r1 - r0 == 1 and c1 - c0 == 1, "relay must be a 2x2 block"
    expected = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    actual = {tuple(map(int, cell)) for cell in coords}
    assert actual == expected, "relay cells must form a solid 2x2 square"
    return {
        "kind": "relay",
        "color": int(color),
        "r0": r0,
        "r1": r1,
        "c0": c0,
        "c1": c1,
    }


def _relay_bbox(top_left: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = map(int, top_left)
    return r, c, r + 1, c + 1


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _extract_nodes(grid: np.ndarray) -> tuple[list[dict[str, int | str]], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)

    start_coords = np.argwhere(g == START_COLOR)
    end_coords = np.argwhere(g == END_COLOR)
    assert start_coords.shape[0] == 1, "expected exactly one red start marker"
    assert end_coords.shape[0] == 1, "expected exactly one blue end marker"

    start = _make_marker("start", START_COLOR, tuple(map(int, start_coords[0])))
    end = _make_marker("end", END_COLOR, tuple(map(int, end_coords[0])))

    nodes: list[dict[str, int | str]] = [start]
    seen_colors = {bg, START_COLOR, END_COLOR}
    relay_colors = sorted(set(map(int, np.unique(g))) - seen_colors)
    assert relay_colors, "expected at least one relay color"
    for color in relay_colors:
        coords = np.argwhere(g == color)
        nodes.append(_make_relay(color, coords))
    nodes.append(end)
    return nodes, bg


def _link_between(
    grid: np.ndarray,
    bg: int,
    a: dict[str, int | str],
    b: dict[str, int | str],
) -> tuple[str, int] | None:
    row_overlap = min(int(a["r1"]), int(b["r1"])) - max(int(a["r0"]), int(b["r0"])) + 1
    if row_overlap == 1:
        lane = max(int(a["r0"]), int(b["r0"]))
        if int(a["c1"]) < int(b["c0"]):
            if int(b["c0"]) - int(a["c1"]) >= 2 and np.all(grid[lane, int(a["c1"]) + 1 : int(b["c0"])] == bg):
                return "h", lane
        elif int(b["c1"]) < int(a["c0"]):
            if int(a["c0"]) - int(b["c1"]) >= 2 and np.all(grid[lane, int(b["c1"]) + 1 : int(a["c0"])] == bg):
                return "h", lane

    col_overlap = min(int(a["c1"]), int(b["c1"])) - max(int(a["c0"]), int(b["c0"])) + 1
    if col_overlap == 1:
        lane = max(int(a["c0"]), int(b["c0"]))
        if int(a["r1"]) < int(b["r0"]):
            if int(b["r0"]) - int(a["r1"]) >= 2 and np.all(grid[int(a["r1"]) + 1 : int(b["r0"]), lane] == bg):
                return "v", lane
        elif int(b["r1"]) < int(a["r0"]):
            if int(a["r0"]) - int(b["r1"]) >= 2 and np.all(grid[int(b["r1"]) + 1 : int(a["r0"]), lane] == bg):
                return "v", lane

    return None


def _infer_chain(
    grid: np.ndarray,
) -> tuple[list[dict[str, int | str]], dict[tuple[int, int], tuple[str, int]], list[int], int]:
    nodes, bg = _extract_nodes(grid)
    start_idx = 0
    end_idx = len(nodes) - 1

    adjacency: list[list[int]] = [[] for _ in nodes]
    edge_info: dict[tuple[int, int], tuple[str, int]] = {}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            link = _link_between(grid, bg, nodes[i], nodes[j])
            if link is None:
                continue
            adjacency[i].append(j)
            adjacency[j].append(i)
            edge_info[(i, j)] = link

    assert len(adjacency[start_idx]) == 1, "start marker must connect to exactly one relay"
    assert len(adjacency[end_idx]) == 1, "end marker must connect to exactly one relay"
    for idx, node in enumerate(nodes):
        if node["kind"] == "relay":
            assert len(adjacency[idx]) == 2, "every relay must sit on the unique chain"

    order = [start_idx]
    prev = -1
    cur = start_idx
    while cur != end_idx:
        options = [nbr for nbr in adjacency[cur] if nbr != prev]
        assert len(options) == 1, "chain order must be unambiguous"
        nxt = options[0]
        order.append(nxt)
        prev, cur = cur, nxt

    assert len(order) == len(nodes), "chain must visit every relay exactly once"
    assert len(set(order)) == len(order), "chain may not loop back on itself"
    return nodes, edge_info, order, bg


def _segment_cells(
    src: dict[str, int | str],
    dst: dict[str, int | str],
    axis: str,
    lane: int,
) -> list[tuple[int, int]]:
    if axis == "h":
        sign = 1 if int(src["c1"]) < int(dst["c0"]) else -1
        if src["kind"] == "relay":
            start_c = int(src["c1"]) + 1 if sign > 0 else int(src["c0"]) - 1
        else:
            start_c = int(src["c0"])
        if dst["kind"] == "relay":
            end_c = int(dst["c0"]) - 1 if sign > 0 else int(dst["c1"]) + 1
        else:
            end_c = int(dst["c0"]) - 1 if sign > 0 else int(dst["c0"]) + 1
        assert start_c != end_c or src["kind"] != "relay", "relay segment must cover at least one cell"
        c0, c1 = sorted((start_c, end_c))
        return [(lane, c) for c in range(c0, c1 + 1)]

    assert axis == "v"
    sign = 1 if int(src["r1"]) < int(dst["r0"]) else -1
    if src["kind"] == "relay":
        start_r = int(src["r1"]) + 1 if sign > 0 else int(src["r0"]) - 1
    else:
        start_r = int(src["r0"])
    if dst["kind"] == "relay":
        end_r = int(dst["r0"]) - 1 if sign > 0 else int(dst["r1"]) + 1
    else:
        end_r = int(dst["r0"]) - 1 if sign > 0 else int(dst["r0"]) + 1
    assert start_r != end_r or src["kind"] != "relay", "relay segment must cover at least one cell"
    r0, r1 = sorted((start_r, end_r))
    return [(r, lane) for r in range(r0, r1 + 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the start-to-end route through the aligned 2x2 relays, changing color after each relay and erasing the relays afterward.

    Input structure:
    - The background is the dominant color.
    - There is one red single-pixel start marker, one blue single-pixel end marker, and several relay colors.
    - Each relay color forms exactly one solid 2x2 block, and relay colors are distinct from red, blue, and the background.
    - Alignment is encoded structurally: consecutive route objects share exactly one row or one column of visibility through background cells.

    Rule:
    - Recover the unique chain that starts at the red marker, passes through every relay once, and ends at the blue marker.
    - Draw a one-cell-wide straight segment for each consecutive pair, stopping at relay edges rather than painting through relay cells.
    - The segment leaving the start is red; after each relay the next segment uses that relay's color.
    - Output a clean background canvas with only the start marker, end marker, and all path segments; erase every 2x2 relay block.

    Key insight:
    - The 2x2 relays are hidden waypoints: their unique row/column visibility recovers the route order, and each relay's own color becomes the paint for the next leg.
    """
    g = np.asarray(grid, dtype=int)
    nodes, edge_info, order, bg = _infer_chain(g)

    out = np.full_like(g, bg)
    for node in nodes:
        if node["kind"] != "relay":
            out[int(node["r0"]), int(node["c0"])] = int(node["color"])

    for src_idx, dst_idx in zip(order, order[1:]):
        key = (src_idx, dst_idx) if src_idx < dst_idx else (dst_idx, src_idx)
        axis, lane = edge_info[key]
        color = int(nodes[src_idx]["color"])
        for r, c in _segment_cells(nodes[src_idx], nodes[dst_idx], axis, lane):
            out[r, c] = color

    return out


def _sample_relay_positions(relay_count: int) -> list[tuple[int, int]] | None:
    relays = [(0, 0)]
    for _ in range(relay_count - 1):
        options: list[tuple[int, int]] = []
        base_r, base_c = relays[-1]
        for axis in ("h", "v"):
            for major_sign in (-1, 1):
                for minor_sign in (-1, 1):
                    for step in range(STEP_MIN, STEP_MAX + 1):
                        if axis == "h":
                            nr = base_r + minor_sign
                            nc = base_c + major_sign * step
                        else:
                            nr = base_r + major_sign * step
                            nc = base_c + minor_sign
                        options.append((nr, nc))
        random.shuffle(options)

        placed = False
        for candidate in options:
            cbbox = _relay_bbox(candidate)
            if any(_bbox_overlap(cbbox, _relay_bbox(existing)) for existing in relays):
                continue
            trial = relays + [candidate]
            rows = [r for r, _ in trial]
            cols = [c for _, c in trial]
            if max(rows) - min(rows) > 22 or max(cols) - min(cols) > 22:
                continue
            relays.append(candidate)
            placed = True
            break
        if not placed:
            return None
    return relays


def _sample_marker(relay: tuple[int, int], axis: str) -> tuple[int, int]:
    r, c = relay
    lane = random.randint(0, 1)
    dist = random.randint(END_DIST_MIN, END_DIST_MAX)
    side = random.choice((-1, 1))
    if axis == "h":
        return r + lane, c - dist if side < 0 else c + 1 + dist
    return (r - dist if side < 0 else r + 1 + dist), c + lane


def _shift_layout(
    relays: list[tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    min_r = min([start[0], end[0]] + [r for r, _ in relays])
    max_r = max([start[0], end[0]] + [r + 1 for r, _ in relays])
    min_c = min([start[1], end[1]] + [c for _, c in relays])
    max_c = max([start[1], end[1]] + [c + 1 for _, c in relays])
    core_h = max_r - min_r + 1
    core_w = max_c - min_c + 1
    if core_h > 28 or core_w > 28:
        return None

    max_extra_h = min(PADDING_TOTAL_MAX, 30 - core_h)
    max_extra_w = min(PADDING_TOTAL_MAX, 30 - core_w)
    if max_extra_h < PADDING_TOTAL_MIN or max_extra_w < PADDING_TOTAL_MIN:
        return None

    extra_h = random.randint(PADDING_TOTAL_MIN, max_extra_h)
    extra_w = random.randint(PADDING_TOTAL_MIN, max_extra_w)
    top_pad = random.randint(1, extra_h - 1)
    left_pad = random.randint(1, extra_w - 1)
    shift_r = top_pad - min_r
    shift_c = left_pad - min_c

    shifted_relays = [(r + shift_r, c + shift_c) for r, c in relays]
    shifted_start = (start[0] + shift_r, start[1] + shift_c)
    shifted_end = (end[0] + shift_r, end[1] + shift_c)
    shape = (core_h + extra_h, core_w + extra_w)
    return shifted_relays, shifted_start, shifted_end, shape


def _render_input(
    relays: list[tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
    shape: tuple[int, int],
) -> np.ndarray:
    grid = np.full(shape, BG_COLOR, dtype=int)
    grid[start] = START_COLOR
    grid[end] = END_COLOR
    colors = random.sample(RELAY_COLOR_POOL, k=len(relays))
    for (r, c), color in zip(relays, colors):
        grid[r : r + 2, c : c + 2] = int(color)
    return grid


def _segments_are_disjoint(
    nodes: list[dict[str, int | str]],
    edge_info: dict[tuple[int, int], tuple[str, int]],
    order: list[int],
) -> bool:
    seen: set[tuple[int, int]] = set()
    for src_idx, dst_idx in zip(order, order[1:]):
        key = (src_idx, dst_idx) if src_idx < dst_idx else (dst_idx, src_idx)
        axis, lane = edge_info[key]
        for cell in _segment_cells(nodes[src_idx], nodes[dst_idx], axis, lane):
            if cell in seen:
                return False
            seen.add(cell)
    return True


def _valid_generated_grid(grid: np.ndarray) -> bool:
    try:
        nodes, edge_info, order, bg = _infer_chain(grid)
        out = main(grid)
    except AssertionError:
        return False

    if bg != BG_COLOR:
        return False
    if not _segments_are_disjoint(nodes, edge_info, order):
        return False
    if out.shape != grid.shape:
        return False
    if np.sum(grid == START_COLOR) != 1 or np.sum(grid == END_COLOR) != 1:
        return False
    if np.array_equal(out, grid):
        return False
    if np.sum(out != BG_COLOR) <= 3:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    relay_count = random.randint(MIN_RELAYS, MAX_RELAYS)
    relays = _sample_relay_positions(relay_count)
    if relays is None:
        return None

    start_axis = random.choice(("h", "v"))
    end_axis = random.choice(("h", "v"))
    start = _sample_marker(relays[0], start_axis)
    end = _sample_marker(relays[-1], end_axis)
    if start == end:
        return None

    shifted = _shift_layout(relays, start, end)
    if shifted is None:
        return None
    placed_relays, placed_start, placed_end, shape = shifted

    grid = _render_input(placed_relays, placed_start, placed_end, shape)
    if not _valid_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid relay-chain ARC input")
