from __future__ import annotations

import random
from collections import Counter
from typing import Iterable

import numpy as np


DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
DIR_FROM_DELTA = {v: k for k, v in DIRS.items()}
SIDES = ("U", "D", "L", "R")
PALETTE = list(range(10))
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int, region: tuple[int, int, int, int] | None = None) -> list[list[tuple[int, int]]]:
    if region is None:
        r0, c0, r1, c1 = 0, 0, grid.shape[0] - 1, grid.shape[1] - 1
    else:
        r0, c0, r1, c1 = region
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if int(grid[r, c]) != int(color) or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                        continue
                    if (nr, nc) in seen or int(grid[nr, nc]) != int(color):
                        continue
                    seen.add((nr, nc))
                    stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_separator_col(grid: np.ndarray, bg: int) -> int:
    cols = [
        c
        for c in range(grid.shape[1])
        if int(grid[0, c]) != int(bg) and np.all(grid[:, c] == int(grid[0, c]))
    ]
    assert len(cols) == 1, "expected one solid separator column"
    return int(cols[0])


def _parse_legend(grid: np.ndarray, sep: int, bg: int) -> dict[int, str]:
    left = grid[:, :sep]
    colors = sorted(set(map(int, left.flat)) - {int(bg)})
    mapping: dict[int, str] = {}
    for color in colors:
        comps = _components_of_color(left, color)
        assert len(comps) == 2, "each legend color must have a sample and one clock hand"
        sample = next((comp for comp in comps if len(comp) == 4), None)
        hand = next((comp for comp in comps if len(comp) == 2), None)
        assert sample is not None and hand is not None, "legend panel must contain a 2x2 sample and a 2-cell hand"
        sr0, sc0, sr1, sc1 = _bbox(sample)
        assert (sr1 - sr0 + 1, sc1 - sc0 + 1) == (2, 2), "legend sample must be a 2x2 square"
        center = (sr0 + 1, sc0 + 4)
        hand_set = set(hand)
        assert center in hand_set, "clock hand must include the center pixel"
        arm = next(pt for pt in hand if pt != center)
        delta = (arm[0] - center[0], arm[1] - center[1])
        assert delta in DIR_FROM_DELTA, "clock hand must point in one cardinal direction"
        mapping[int(color)] = DIR_FROM_DELTA[delta]
    assert mapping, "legend must not be empty"
    return mapping


def _wall_info(grid: np.ndarray, sep: int, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    right = grid[:, sep + 1 :]
    colors = sorted(set(map(int, right.flat)) - {int(bg)})
    best: tuple[int, tuple[int, int, int, int]] | None = None
    for color in colors:
        coords = [(r, c + sep + 1) for r, c in np.argwhere(right == color)]
        if len(coords) < 8:
            continue
        b = _bbox(coords)
        r0, c0, r1, c1 = b
        border = {
            *( (r0, c) for c in range(c0, c1 + 1) ),
            *( (r1, c) for c in range(c0, c1 + 1) ),
            *( (r, c0) for r in range(r0, r1 + 1) ),
            *( (r, c1) for r in range(r0, r1 + 1) ),
        }
        if set(coords) != border:
            continue
        if best is None or len(coords) > (2 * ((best[1][2] - best[1][0] + 1) + (best[1][3] - best[1][1] + 1)) - 4):
            best = (int(color), b)
    assert best is not None, "expected one hollow wall rectangle on the right canvas"
    return best


def _marker_positions(
    grid: np.ndarray,
    legend_dirs: dict[int, str],
    wall_bbox: tuple[int, int, int, int],
    wall_color: int,
    bg: int,
) -> dict[tuple[int, int], int]:
    r0, c0, r1, c1 = wall_bbox
    markers: dict[tuple[int, int], int] = {}
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            value = int(grid[r, c])
            if value == int(bg):
                continue
            assert value != int(wall_color), "wall color must appear only on the wall frame"
            assert value in legend_dirs, "canvas markers must use legend colors"
            markers[(r, c)] = value
    assert markers, "expected at least one marker inside the wall"
    return markers


def _first_obstacle(grid: np.ndarray, start: tuple[int, int], direction: str, bg: int) -> tuple[tuple[int, int], int]:
    dr, dc = DIRS[direction]
    r, c = start
    while True:
        r += dr
        c += dc
        assert 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1], "ray must hit a wall before leaving the grid"
        value = int(grid[r, c])
        if value != int(bg):
            return (r, c), value


def _outgoing_edges(
    grid: np.ndarray,
    legend_dirs: dict[int, str],
    markers: dict[tuple[int, int], int],
    wall_color: int,
    bg: int,
) -> dict[tuple[int, int], tuple[str, tuple[int, int]]]:
    outgoing: dict[tuple[int, int], tuple[str, tuple[int, int]]] = {}
    marker_set = set(markers)
    for pos, color in markers.items():
        hit, hit_value = _first_obstacle(grid, pos, legend_dirs[int(color)], bg)
        if hit_value == int(wall_color):
            outgoing[pos] = ("wall", hit)
            continue
        assert hit in marker_set, "the first non-background obstacle must be a wall or another marker"
        outgoing[pos] = ("marker", hit)
    return outgoing


def _assert_acyclic(outgoing: dict[tuple[int, int], tuple[str, tuple[int, int]]]) -> None:
    state: dict[tuple[int, int], int] = {}

    def dfs(node: tuple[int, int]) -> None:
        cur = state.get(node, 0)
        assert cur != 1, "relay graph must be acyclic"
        if cur == 2:
            return
        state[node] = 1
        kind, nxt = outgoing[node]
        if kind == "marker":
            dfs(nxt)
        state[node] = 2

    for node in outgoing:
        dfs(node)


def _trace_wall_hit(
    start: tuple[int, int],
    outgoing: dict[tuple[int, int], tuple[str, tuple[int, int]]],
) -> tuple[int, int]:
    cur = start
    seen: set[tuple[int, int]] = set()
    while True:
        assert cur not in seen, "relay graph must not cycle"
        seen.add(cur)
        kind, nxt = outgoing[cur]
        if kind == "wall":
            return nxt
        cur = nxt


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the left legend to decode each marker color into a launch direction, relay rays through marker turns, and recolor the wall cells they finally hit.

    Input structure:
    - A single separator column splits the grid into a left legend area and a right canvas.
    - Each legend panel contains one color sample plus a 3×3 clock-hand cue for that color's direction.
    - The right canvas contains one hollow non-background wall rectangle and several singleton marker pixels inside it.

    Rule:
    - Decode every marker color via the legend to get its initial flight direction.
    - A ray moves through background until the first non-background obstacle: a wall cell ends the ray, while another marker cell redirects the ray from that marker using the hit marker's legend direction and keeps the original carried color.
    - Clear all marker pixels in the output and recolor only the wall cells reached by the launching markers.

    Key insight:
    - The legend maps colors to directions, and the final hit is determined by the full relay polyline rather than by connecting each marker straight to its nearest wall.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    sep = _find_separator_col(g, bg)
    legend_dirs = _parse_legend(g, sep, bg)
    wall_color, wall_bbox = _wall_info(g, sep, bg)
    markers = _marker_positions(g, legend_dirs, wall_bbox, wall_color, bg)
    outgoing = _outgoing_edges(g, legend_dirs, markers, wall_color, bg)
    _assert_acyclic(outgoing)

    indegree = {pos: 0 for pos in markers}
    for kind, nxt in outgoing.values():
        if kind == "marker":
            indegree[nxt] += 1

    # The specification leaves shared wall-hit overwrites ambiguous.
    # Resolve that by launching only the outermost chain heads recovered from indegree zero.
    launchers = [pos for pos, deg in indegree.items() if deg == 0]
    assert launchers, "expected at least one chain head"

    out = g.copy()
    for r, c in markers:
        out[r, c] = bg

    for pos in launchers:
        wall_hit = _trace_wall_hit(pos, outgoing)
        out[wall_hit] = int(markers[pos])

    return out


def _legend_panel(grid: np.ndarray, top: int, color: int, direction: str) -> None:
    grid[top : top + 2, 1:3] = int(color)
    center = (top + 1, 5)
    arm = (center[0] + DIRS[direction][0], center[1] + DIRS[direction][1])
    grid[center] = int(color)
    grid[arm] = int(color)


def _build_chain(
    side: str,
    length: int,
    wall_bbox: tuple[int, int, int, int],
    occupied: set[tuple[int, int]],
) -> list[dict[str, object]] | None:
    r0, c0, r1, c1 = wall_bbox
    for _ in range(120):
        nodes: list[tuple[tuple[int, int], str]] = []
        if side == "U":
            hit_c = random.randint(c0 + 2, c1 - 2)
            tail = (random.randint(r0 + 2, min(r0 + 4, r1 - 2)), hit_c)
            nodes = [(tail, "U")]
            if length >= 2:
                mid_dir = random.choice(("L", "R"))
                if mid_dir == "R":
                    max_dx = tail[1] - (c0 + 1)
                    if max_dx < 2:
                        continue
                    mid = (tail[0], tail[1] - random.randint(2, min(4, max_dx)))
                else:
                    max_dx = (c1 - 1) - tail[1]
                    if max_dx < 2:
                        continue
                    mid = (tail[0], tail[1] + random.randint(2, min(4, max_dx)))
                nodes.insert(0, (mid, mid_dir))
            if length == 3:
                mid = nodes[0][0]
                head_dir = random.choice(("U", "D"))
                if head_dir == "D":
                    max_dy = mid[0] - (r0 + 1)
                    if max_dy < 2:
                        continue
                    head = (mid[0] - random.randint(2, min(4, max_dy)), mid[1])
                else:
                    max_dy = (r1 - 1) - mid[0]
                    if max_dy < 2:
                        continue
                    head = (mid[0] + random.randint(2, min(4, max_dy)), mid[1])
                nodes.insert(0, (head, head_dir))
        elif side == "D":
            hit_c = random.randint(c0 + 2, c1 - 2)
            tail = (random.randint(max(r1 - 4, r0 + 2), r1 - 2), hit_c)
            nodes = [(tail, "D")]
            if length >= 2:
                mid_dir = random.choice(("L", "R"))
                if mid_dir == "R":
                    max_dx = tail[1] - (c0 + 1)
                    if max_dx < 2:
                        continue
                    mid = (tail[0], tail[1] - random.randint(2, min(4, max_dx)))
                else:
                    max_dx = (c1 - 1) - tail[1]
                    if max_dx < 2:
                        continue
                    mid = (tail[0], tail[1] + random.randint(2, min(4, max_dx)))
                nodes.insert(0, (mid, mid_dir))
            if length == 3:
                mid = nodes[0][0]
                head_dir = random.choice(("U", "D"))
                if head_dir == "D":
                    max_dy = mid[0] - (r0 + 1)
                    if max_dy < 2:
                        continue
                    head = (mid[0] - random.randint(2, min(4, max_dy)), mid[1])
                else:
                    max_dy = (r1 - 1) - mid[0]
                    if max_dy < 2:
                        continue
                    head = (mid[0] + random.randint(2, min(4, max_dy)), mid[1])
                nodes.insert(0, (head, head_dir))
        elif side == "L":
            hit_r = random.randint(r0 + 2, r1 - 2)
            tail = (hit_r, random.randint(c0 + 2, min(c0 + 4, c1 - 2)))
            nodes = [(tail, "L")]
            if length >= 2:
                mid_dir = random.choice(("U", "D"))
                if mid_dir == "D":
                    max_dy = tail[0] - (r0 + 1)
                    if max_dy < 2:
                        continue
                    mid = (tail[0] - random.randint(2, min(4, max_dy)), tail[1])
                else:
                    max_dy = (r1 - 1) - tail[0]
                    if max_dy < 2:
                        continue
                    mid = (tail[0] + random.randint(2, min(4, max_dy)), tail[1])
                nodes.insert(0, (mid, mid_dir))
            if length == 3:
                mid = nodes[0][0]
                head_dir = random.choice(("L", "R"))
                if head_dir == "R":
                    max_dx = mid[1] - (c0 + 1)
                    if max_dx < 2:
                        continue
                    head = (mid[0], mid[1] - random.randint(2, min(4, max_dx)))
                else:
                    max_dx = (c1 - 1) - mid[1]
                    if max_dx < 2:
                        continue
                    head = (mid[0], mid[1] + random.randint(2, min(4, max_dx)))
                nodes.insert(0, (head, head_dir))
        else:
            hit_r = random.randint(r0 + 2, r1 - 2)
            tail = (hit_r, random.randint(max(c1 - 4, c0 + 2), c1 - 2))
            nodes = [(tail, "R")]
            if length >= 2:
                mid_dir = random.choice(("U", "D"))
                if mid_dir == "D":
                    max_dy = tail[0] - (r0 + 1)
                    if max_dy < 2:
                        continue
                    mid = (tail[0] - random.randint(2, min(4, max_dy)), tail[1])
                else:
                    max_dy = (r1 - 1) - tail[0]
                    if max_dy < 2:
                        continue
                    mid = (tail[0] + random.randint(2, min(4, max_dy)), tail[1])
                nodes.insert(0, (mid, mid_dir))
            if length == 3:
                mid = nodes[0][0]
                head_dir = random.choice(("L", "R"))
                if head_dir == "R":
                    max_dx = mid[1] - (c0 + 1)
                    if max_dx < 2:
                        continue
                    head = (mid[0], mid[1] - random.randint(2, min(4, max_dx)))
                else:
                    max_dx = (c1 - 1) - mid[1]
                    if max_dx < 2:
                        continue
                    head = (mid[0], mid[1] + random.randint(2, min(4, max_dx)))
                nodes.insert(0, (head, head_dir))

        coords = [pos for pos, _dir in nodes]
        if len(set(coords)) != len(coords):
            continue
        if any(pos in occupied for pos in coords):
            continue
        for rr, cc in coords:
            if not (r0 + 1 <= rr <= r1 - 1 and c0 + 1 <= cc <= c1 - 1):
                break
        else:
            return [{"pos": pos, "dir": direction, "side": side} for pos, direction in nodes]
    return None


def _chain_layout(n_markers: int) -> list[int]:
    choices = {
        3: [[2, 1], [3]],
        4: [[3, 1], [2, 2]],
        5: [[3, 2]],
    }
    return random.choice(choices[n_markers])


def _manual_output(
    grid: np.ndarray,
    markers: list[dict[str, object]],
    chain_specs: list[list[dict[str, object]]],
    bg: int,
) -> np.ndarray:
    out = grid.copy()
    for marker in markers:
        r, c = marker["pos"]
        out[r, c] = int(bg)
    for chain in chain_specs:
        launcher = chain[0]
        tail = chain[-1]
        side = str(tail["dir"])
        tr, tc = tail["pos"]
        if side == "U":
            hit = (next(r for r in range(tr - 1, -1, -1) if int(grid[r, tc]) != int(bg)), tc)
        elif side == "D":
            hit = (next(r for r in range(tr + 1, grid.shape[0]) if int(grid[r, tc]) != int(bg)), tc)
        elif side == "L":
            hit = (tr, next(c for c in range(tc - 1, -1, -1) if int(grid[tr, c]) != int(bg)))
        else:
            hit = (tr, next(c for c in range(tc + 1, grid.shape[1]) if int(grid[tr, c]) != int(bg)))
        out[hit] = int(launcher["color"])
    return out


def _expected_outgoing(
    grid: np.ndarray,
    chain_specs: list[list[dict[str, object]]],
    bg: int,
) -> dict[tuple[int, int], tuple[str, tuple[int, int]]]:
    expected: dict[tuple[int, int], tuple[str, tuple[int, int]]] = {}
    for chain in chain_specs:
        for i, node in enumerate(chain):
            pos = tuple(node["pos"])
            if i + 1 < len(chain):
                expected[pos] = ("marker", tuple(chain[i + 1]["pos"]))
                continue
            hit, hit_value = _first_obstacle(grid, pos, str(node["dir"]), bg)
            assert hit_value != int(bg), "tail ray must terminate at a non-background obstacle"
            expected[pos] = ("wall", hit)
    return expected


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    chain_specs: list[list[dict[str, object]]],
) -> bool:
    try:
        bg = _get_bg(grid)
        sep = _find_separator_col(grid, bg)
        legend_dirs = _parse_legend(grid, sep, bg)
        wall_color, wall_bbox = _wall_info(grid, sep, bg)
        markers = _marker_positions(grid, legend_dirs, wall_bbox, wall_color, bg)
        outgoing = _outgoing_edges(grid, legend_dirs, markers, wall_color, bg)
        _assert_acyclic(outgoing)

        if outgoing != _expected_outgoing(grid, chain_specs, bg):
            return False

        indegree = {pos: 0 for pos in markers}
        relay_edges = 0
        for kind, nxt in outgoing.values():
            if kind == "marker":
                relay_edges += 1
                indegree[nxt] += 1
        if relay_edges == 0:
            return False

        launchers = [pos for pos, deg in indegree.items() if deg == 0]
        expected_launchers = [tuple(chain[0]["pos"]) for chain in chain_specs]
        if set(launchers) != set(expected_launchers):
            return False

        actual_hits = {pos: _trace_wall_hit(pos, outgoing) for pos in launchers}
        if len(set(actual_hits.values())) != len(actual_hits):
            return False

        changed_by_relay = False
        cleared_markers = np.where(np.isin(grid, list(legend_dirs)), bg, grid)
        for chain in chain_specs:
            if len(chain) < 2:
                continue
            launcher = chain[0]
            lr, lc = launcher["pos"]
            direct_hit, direct_value = _first_obstacle(
                cleared_markers,
                (lr, lc),
                str(launcher["dir"]),
                bg,
            )
            if direct_value == wall_color and direct_hit != actual_hits[(lr, lc)]:
                changed_by_relay = True
                break
        if not changed_by_relay:
            return False

        for chain in chain_specs:
            dirs = [str(node["dir"]) for node in chain]
            if len(chain) >= 2 and len(set(dirs)) == 1:
                return False

        return np.array_equal(main(grid), expected)
    except (AssertionError, KeyError, StopIteration):
        return False


def _gen_attempt() -> np.ndarray | None:
    n_markers = random.randint(3, 5)
    legend_h = 4 * n_markers - 1
    h = random.randint(max(legend_h + 2, 18), 28)
    w = random.randint(22, 30)
    sep = random.randint(8, 9)
    if w - sep - 3 < 10:
        return None

    picked = random.sample(PALETTE, 3 + n_markers)
    bg, sep_color, wall_color = picked[:3]
    marker_colors = picked[3:]

    grid = np.full((h, w), int(bg), dtype=int)
    grid[:, sep] = int(sep_color)

    legend_top = random.randint(1, h - legend_h - 1)

    ring_h = random.randint(max(10, legend_h // 2), min(h - 2, 18))
    ring_w = random.randint(10, min(w - sep - 3, 18))
    ring_r0 = random.randint(1, h - ring_h - 1)
    ring_c0 = random.randint(sep + 2, w - ring_w - 1)
    ring_r1 = ring_r0 + ring_h - 1
    ring_c1 = ring_c0 + ring_w - 1
    wall_bbox = (ring_r0, ring_c0, ring_r1, ring_c1)

    grid[ring_r0, ring_c0 : ring_c1 + 1] = int(wall_color)
    grid[ring_r1, ring_c0 : ring_c1 + 1] = int(wall_color)
    grid[ring_r0 : ring_r1 + 1, ring_c0] = int(wall_color)
    grid[ring_r0 : ring_r1 + 1, ring_c1] = int(wall_color)

    lengths = _chain_layout(n_markers)
    sides = random.sample(SIDES, len(lengths))
    occupied: set[tuple[int, int]] = set()
    chains: list[list[dict[str, object]]] = []
    for side, length in zip(sides, lengths):
        chain = _build_chain(side, length, wall_bbox, occupied)
        if chain is None:
            return None
        for node in chain:
            occupied.add(node["pos"])
        chains.append(chain)

    nodes = [node for chain in chains for node in chain]
    random.shuffle(marker_colors)
    for node, color in zip(nodes, marker_colors):
        node["color"] = int(color)
        r, c = node["pos"]
        grid[r, c] = int(color)

    legend_order = marker_colors[:]
    random.shuffle(legend_order)
    dir_by_color = {int(node["color"]): str(node["dir"]) for node in nodes}
    for i, color in enumerate(legend_order):
        _legend_panel(grid, legend_top + 4 * i, int(color), dir_by_color[int(color)])

    if _get_bg(grid) != int(bg):
        return None

    expected = _manual_output(grid, nodes, chains, bg)
    if not _semantic_checks(grid, expected, chains):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid relay-ray ARC input")
