from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
DIRS4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield int(r + dr), int(c + dc)


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    h, _w = grid.shape
    candidates: list[tuple[int, int, list[int]]] = []

    for color in sorted(set(map(int, grid.ravel()))):
        rows = [r for r in range(h) if np.all(grid[r, :] == int(color))]
        if not rows:
            continue
        if rows[0] == 0 or rows[-1] == h - 1:
            continue

        prev = 0
        lengths = []
        ok = True
        for r in rows:
            if r < prev:
                ok = False
                break
            lengths.append(int(r - prev))
            prev = r + 1
        lengths.append(int(h - prev))
        if not ok or len(lengths) < 3 or min(lengths) <= 0:
            continue
        if len(set(lengths)) != 1:
            continue
        candidates.append((len(lengths), lengths[0], rows))

    assert candidates, "expected full-width separator rows between equal panels"
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    panel_count, panel_h, sep_rows = candidates[0]

    panels = []
    start = 0
    for r in sep_rows:
        panels.append(grid[start:r, :].copy())
        start = r + 1
    panels.append(grid[start:, :].copy())
    assert len(panels) == panel_count
    assert all(panel.shape == (panel_h, grid.shape[1]) for panel in panels)
    assert any(not np.array_equal(panels[i], panels[i + 1]) for i in range(len(panels) - 1))
    return panels


def _find_liquid_and_pipe(panels: list[np.ndarray]) -> tuple[int, int]:
    colors = sorted(set(int(v) for panel in panels for v in panel.ravel()))
    series = {
        color: [int(np.count_nonzero(panel == color)) for panel in panels]
        for color in colors
    }

    pairs: list[tuple[int, int, int]] = []
    for liquid in colors:
        liquid_diffs = [
            series[liquid][i + 1] - series[liquid][i]
            for i in range(len(panels) - 1)
        ]
        if sum(liquid_diffs) <= 0 or any(d <= 0 for d in liquid_diffs):
            continue

        for pipe in colors:
            if pipe == liquid:
                continue
            pipe_diffs = [
                series[pipe][i + 1] - series[pipe][i]
                for i in range(len(panels) - 1)
            ]
            if any(d >= 0 for d in pipe_diffs):
                continue
            if liquid_diffs != [-d for d in pipe_diffs]:
                continue

            ok = True
            for before, after in zip(panels, panels[1:]):
                changed = before != after
                if not np.any(changed):
                    ok = False
                    break
                if not np.all((before[changed] == pipe) & (after[changed] == liquid)):
                    ok = False
                    break
            if ok:
                pairs.append((sum(liquid_diffs), int(liquid), int(pipe)))

    assert pairs, "expected exactly one pipe color that is progressively recolored"
    pairs.sort(reverse=True)
    if len(pairs) > 1:
        assert pairs[0][0] > pairs[1][0] or pairs[0][1:] == pairs[1][1:]
    return int(pairs[0][1]), int(pairs[0][2])


def _channel_graph(mask: np.ndarray) -> tuple[set[Coord], dict[Coord, int]]:
    coords = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    assert coords, "expected a visible channel"

    degrees: dict[Coord, int] = {}
    edge_sum = 0
    for r, c in coords:
        deg = sum((nr, nc) in coords for nr, nc in _neighbors4(r, c))
        assert 1 <= deg <= 4, "channel cells must be 4-connected linework"
        degrees[(r, c)] = int(deg)
        edge_sum += int(deg)

    start = next(iter(coords))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nb in _neighbors4(r, c):
            if nb in coords and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    assert seen == coords, "expected one connected tree channel"
    assert edge_sum // 2 == len(coords) - 1, "channel must be a tree, not a loop"
    return coords, degrees


def _find_root(coords: set[Coord], degrees: dict[Coord, int], shape: tuple[int, int]) -> Coord:
    h, w = shape
    endpoints = [p for p, deg in degrees.items() if deg == 1]
    assert len(endpoints) >= 2, "tree channel needs an entrance and leaves"

    def edge_dist(p: Coord) -> int:
        r, c = p
        return min(int(r), int(c), int(h - 1 - r), int(w - 1 - c))

    min_dist = min(edge_dist(p) for p in endpoints)
    closest = [p for p in endpoints if edge_dist(p) == min_dist]
    if len(closest) == 1:
        return closest[0]

    top_r = min(r for r, _c in closest)
    top = [p for p in closest if p[0] == top_r]
    assert len(top) == 1, "expected a unique entrance endpoint"
    return top[0]


def _segment_depths(mask: np.ndarray) -> dict[Coord, int]:
    coords, degrees = _channel_graph(mask)
    root = _find_root(coords, degrees, mask.shape)

    depths: dict[Coord, int] = {root: 1}
    stack: list[tuple[Coord, Coord | None, Coord | None, int]] = [(root, None, None, 1)]
    while stack:
        cur, parent, in_dir, depth = stack.pop()
        r, c = cur
        for nb in sorted(_neighbors4(r, c)):
            if nb not in coords or nb == parent:
                continue
            assert nb not in depths, "channel must have a unique path from the root"
            move = (int(nb[0] - r), int(nb[1] - c))
            next_depth = int(depth)
            if parent is not None:
                if degrees[cur] >= 3:
                    next_depth += 1
                elif degrees[cur] == 2 and move != in_dir:
                    next_depth += 1
            depths[nb] = int(next_depth)
            stack.append((nb, cur, move, int(next_depth)))

    assert len(depths) == len(coords)
    return depths


def _observed_depth(panel: np.ndarray, depths: dict[Coord, int], liquid: int, pipe: int) -> int:
    filled = [depth for p, depth in depths.items() if int(panel[p]) == int(liquid)]
    assert filled, "expected at least the first pipe segment to be filled"
    current = max(filled)
    for p, depth in depths.items():
        expected_color = liquid if depth <= current else pipe
        assert int(panel[p]) == int(expected_color), "liquid must fill a depth prefix"
    return int(current)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance liquid one straight segment farther through a repeated tree-shaped pipe.

    Input structure:
    - Full-width separator rows divide the grid into equal panel frames stacked in time order.
    - Every panel contains the same single-cell-wide tree channel, with one entrance endpoint near the frame.
    - One channel color is empty pipe and another is liquid; from panel to panel, only pipe cells are recolored to liquid.

    Rule:
    - Split the channel at bends and branch points, then number straight segments by segment distance from the entrance.
    - The panels show consecutive prefixes of those segment depths; each branch advances at the same segment step and finished leaves stop.
    - Return one panel, matching the last panel except that all channel cells at the next segment depth are recolored as liquid.

    Key insight:
    - Time advances by whole straight pipe segments, not by cell count; the first two equal-length segments are a decoy for fixed-distance growth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    panels = _split_panels(g)
    liquid, pipe = _find_liquid_and_pipe(panels)

    last = panels[-1].copy()
    channel = (last == int(liquid)) | (last == int(pipe))
    depths = _segment_depths(channel)

    observed = [_observed_depth(panel, depths, liquid, pipe) for panel in panels]
    assert all(b == a + 1 for a, b in zip(observed, observed[1:])), "panels must be consecutive frames"

    next_depth = observed[-1] + 1
    out = last.copy()
    for p, depth in depths.items():
        if depth <= next_depth:
            out[p] = int(liquid)
    return out


# ----------------------------- generator -----------------------------


_MAX_ATTEMPTS = 500
_PANEL_COUNT = 3


def _add_ray(cells: set[Coord], start: Coord, direction: Coord, steps: int) -> Coord:
    r, c = start
    dr, dc = direction
    end = start
    for i in range(1, int(steps) + 1):
        end = (int(r + dr * i), int(c + dc * i))
        cells.add(end)
    return end


def _make_tree(panel_h: int, panel_w: int) -> tuple[np.ndarray, dict[Coord, int]] | None:
    for _ in range(200):
        seg_len = random.choice([2, 3])
        side = random.choice([-1, 1])
        second_branch = random.choice([-1, 1])
        run_len = random.randint(1, 4)
        down_len = random.randint(2, 3)
        turn_len = random.randint(2, 4)
        tail_len = random.randint(1, 2)

        root_c = random.randint(2, panel_w - 3)
        root = (1, root_c)
        cells: set[Coord] = {root}

        corner = _add_ray(cells, root, (1, 0), seg_len - 1)
        branch = _add_ray(cells, corner, (0, side), seg_len)
        _add_ray(cells, branch, (0, side), run_len)
        lower_corner = _add_ray(cells, branch, (1, 0), down_len)
        turn_end = _add_ray(cells, lower_corner, (0, second_branch), turn_len)
        _add_ray(cells, turn_end, (1, 0), tail_len)

        if any(r <= 0 or r >= panel_h - 1 or c <= 0 or c >= panel_w - 1 for r, c in cells):
            continue
        if any((r, c) != root and (r == 1 or c == 1 or r == panel_h - 2 or c == panel_w - 2) for r, c in cells):
            continue

        mask = np.zeros((panel_h, panel_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        try:
            depths = _segment_depths(mask)
        except AssertionError:
            continue

        if _find_root(*_channel_graph(mask), mask.shape) != root:
            continue
        counts = Counter(depths.values())
        if max(counts) < _PANEL_COUNT + 2:
            continue
        if counts[1] != seg_len or counts[2] != seg_len:
            continue
        if counts[3] == seg_len:
            continue
        if not any(deg >= 3 for deg in _channel_graph(mask)[1].values()):
            continue
        return mask, depths

    return None


def _render_panel(
    panel_h: int,
    panel_w: int,
    depths: dict[Coord, int],
    fill_depth: int,
    *,
    bg: int,
    frame: int,
    pipe: int,
    liquid: int,
) -> np.ndarray:
    panel = np.full((panel_h, panel_w), int(bg), dtype=int)
    panel[0, :] = int(frame)
    panel[-1, :] = int(frame)
    panel[:, 0] = int(frame)
    panel[:, -1] = int(frame)
    for p, depth in depths.items():
        panel[p] = int(liquid if depth <= fill_depth else pipe)
    return panel


def _stack_panels(panels: list[np.ndarray], sep: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    width = panels[0].shape[1]
    for i, panel in enumerate(panels):
        if i:
            rows.append(np.full((1, width), int(sep), dtype=int))
        rows.append(panel)
    return np.vstack(rows)


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(7, 9)
    panel_w = random.randint(12, 23)
    if _PANEL_COUNT * panel_h + (_PANEL_COUNT - 1) > 30:
        return None

    made = _make_tree(panel_h, panel_w)
    if made is None:
        return None
    _mask, depths = made
    counts = Counter(depths.values())

    colors = random.sample(range(10), 5)
    bg, frame, sep, pipe, liquid = map(int, colors)
    panels = [
        _render_panel(panel_h, panel_w, depths, depth, bg=bg, frame=frame, pipe=pipe, liquid=liquid)
        for depth in range(1, _PANEL_COUNT + 1)
    ]
    expected = _render_panel(
        panel_h,
        panel_w,
        depths,
        _PANEL_COUNT + 1,
        bg=bg,
        frame=frame,
        pipe=pipe,
        liquid=liquid,
    )
    inp = _stack_panels(panels, sep)

    liquid_counts = [int(np.count_nonzero(panel == liquid)) for panel in panels]
    if liquid_counts[0] != liquid_counts[1] - liquid_counts[0]:
        return None
    if liquid_counts[2] - liquid_counts[1] == liquid_counts[0]:
        return None
    if counts[_PANEL_COUNT + 1] <= 0:
        return None
    if any(np.array_equal(expected, panel) for panel in panels):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid tree-pipe segment input")
