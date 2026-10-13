import random

import numpy as np


BG = 0
WALL = 5
LIQUID_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 300


def _longest_wall_run(g: np.ndarray) -> tuple[int, int, int]:
    best = None
    h, w = g.shape
    for r in range(h):
        c = 0
        while c < w:
            if int(g[r, c]) != WALL:
                c += 1
                continue
            c0 = c
            while c < w and int(g[r, c]) == WALL:
                c += 1
            c1 = c - 1
            length = c1 - c0 + 1
            if best is None or length > best[0]:
                best = (length, r, c0, c1)
    assert best is not None, "expected at least one grey pipe/bottom run"
    _, r, c0, c1 = best
    return int(r), int(c0), int(c1)


def _parse_containers(g: np.ndarray) -> tuple[list[dict], int]:
    bottom, c0, c1 = _longest_wall_run(g)
    wall_cols = []
    for c in range(c0, c1 + 1):
        if int(g[bottom, c]) != WALL:
            continue
        r = bottom
        while r >= 0 and int(g[r, c]) == WALL:
            r -= 1
        top = r + 1
        if bottom - top + 1 >= 2:
            wall_cols.append((int(c), int(top)))

    assert len(wall_cols) >= 4 and len(wall_cols) % 2 == 0, "expected paired container walls"

    containers = []
    for i in range(0, len(wall_cols), 2):
        left, top_l = wall_cols[i]
        right, top_r = wall_cols[i + 1]
        assert top_l == top_r, "paired container walls must have the same top"
        assert right - left >= 2, "container interior must be non-empty"

        content = []
        seen_air = False
        for r in range(bottom - 1, top_l - 1, -1):
            row = g[r, left + 1 : right]
            vals = set(map(int, row.tolist()))
            if vals == {BG}:
                seen_air = True
                continue
            assert len(vals) == 1, "each liquid row must be a solid color"
            color = next(iter(vals))
            assert color not in (BG, WALL), "liquid rows cannot be background or wall"
            assert not seen_air, "liquid must be contiguous from the bottom"
            content.append(int(color))

        containers.append(
            {
                "left": int(left),
                "right": int(right),
                "top": int(top_l),
                "bottom": int(bottom),
                "capacity": int(bottom - top_l),
                "content": content,
                "initial_colors": set(content),
            }
        )

    capacities = {c["capacity"] for c in containers}
    assert len(capacities) == 1, "all containers must have the same interior height"
    return containers, int(bottom)


def _parse_valves(g: np.ndarray, containers: list[dict]) -> list[dict]:
    valves = []
    for i in range(len(containers) - 1):
        left_box = containers[i]
        right_box = containers[i + 1]
        row = int(left_box["bottom"] - 1)
        c0 = int(left_box["right"] + 1)
        c1 = int(right_box["left"] - 1)
        assert c0 <= c1, "adjacent containers must have a pipe gap"

        found = []
        for c in range(c0, c1 + 1):
            color = int(g[row, c])
            if color not in (BG, WALL):
                found.append((row, c, color))
        assert len(found) == 1, "each pipe gap must contain exactly one valve pixel"
        vr, vc, color = found[0]

        matches = []
        if color in left_box["initial_colors"]:
            matches.append(i)
        if color in right_box["initial_colors"]:
            matches.append(i + 1)
        assert len(matches) == 1, "valve color must identify exactly one connected source"

        src = matches[0]
        dst = i + 1 if src == i else i
        valves.append({"row": int(vr), "col": int(vc), "color": int(color), "src": src, "dst": dst})
    return valves


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer liquid between pipe-connected containers according to the valve color.

    Input structure:
    - The background is black and grey pixels form same-height open-top U-shaped containers.
    - Adjacent containers are joined by a one-pixel-high grey pipe on the container bottom row.
    - A single colored valve pixel sits just above each pipe, and its color matches the liquid in one of the two connected containers.

    Rule:
    - For each pipe from left to right, the valve color identifies the source container; the other connected container is the target.
    - If the source liquid level is higher than the target level, remove liquid rows from the source top and add rows of the source color above the target liquid until the source is no longer higher or the source empties.
    - If the source level is not higher, no liquid moves.
    - Every valve pixel becomes black; grey walls and pipes stay fixed.

    Key insight: The valve color is a pointer to the source, while the balancing compares only liquid height in rows, not container width or liquid volume.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    containers, _bottom = _parse_containers(g)
    valves = _parse_valves(g, containers)

    for valve in valves:
        src = containers[valve["src"]]
        dst = containers[valve["dst"]]
        color = int(valve["color"])
        while len(src["content"]) > len(dst["content"]) and src["content"]:
            src["content"].pop()
            assert len(dst["content"]) < dst["capacity"], "target container overflow"
            dst["content"].append(color)

    out = g.copy()
    for valve in valves:
        out[valve["row"], valve["col"]] = BG

    for box in containers:
        top, bottom = int(box["top"]), int(box["bottom"])
        left, right = int(box["left"]), int(box["right"])
        out[top:bottom, left + 1 : right] = BG
        for depth, color in enumerate(box["content"]):
            r = bottom - 1 - depth
            out[r, left + 1 : right] = int(color)
    return out


def _container_bounds(widths: list[int], gaps: list[int], left_margin: int) -> list[tuple[int, int]]:
    bounds = []
    c = left_margin
    for i, width in enumerate(widths):
        left = c
        right = left + width + 1
        bounds.append((left, right))
        c = right + 1
        if i < len(gaps):
            c += gaps[i]
    return bounds


def _pick_layout() -> tuple[int, int, int, list[int], list[int], list[tuple[int, int]]] | None:
    n = random.randint(2, 5)
    inner_h = random.randint(3, 7)
    widths = [random.randint(2, 6) for _ in range(n)]
    gaps = [random.randint(1, 4) for _ in range(n - 1)]
    span = sum(w + 2 for w in widths) + sum(gaps)
    if span > 28:
        return None

    left_margin = random.randint(0, 30 - span)
    bounds = _container_bounds(widths, gaps, left_margin)
    used_w = bounds[-1][1] + 1
    right_extra = random.randint(0, 30 - used_w)
    cols = used_w + right_extra

    top = random.randint(1, 4)
    bottom_margin = random.randint(0, 4)
    rows = top + inner_h + 1 + bottom_margin
    if rows > 30 or cols > 30:
        return None
    return rows, top, inner_h, widths, gaps, bounds


def _heights_and_sources(n: int, inner_h: int, active: bool) -> tuple[list[int], list[int]] | None:
    if active:
        chosen_gap = random.randrange(n - 1)
        source = random.choice([chosen_gap, chosen_gap + 1])
        target = chosen_gap + 1 if source == chosen_gap else chosen_gap
        base_min = 0 if n == 2 else 1
        if base_min > inner_h - 2:
            return None
        base = random.randint(base_min, inner_h - 2)
        heights = [base] * n
        heights[source] = base + 2

        sources = []
        for i in range(n - 1):
            if i == chosen_gap:
                sources.append(source)
            elif i in (source, target):
                sources.append(i + 1)
            elif i + 1 in (source, target):
                sources.append(i)
            else:
                sources.append(random.choice([i, i + 1]))
        return heights, sources

    base = random.randint(1, inner_h)
    heights = [base] * n
    sources = [random.choice([i, i + 1]) for i in range(n - 1)]
    return heights, sources


def _render(rows: int, top: int, inner_h: int, bounds: list[tuple[int, int]], heights: list[int],
            colors: list[int], sources: list[int]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    bottom = top + inner_h
    grid = np.full((rows, max(r for _, r in bounds) + 1), BG, dtype=int)

    first_left = bounds[0][0]
    last_right = bounds[-1][1]
    grid[bottom, first_left : last_right + 1] = WALL

    for i, (left, right) in enumerate(bounds):
        grid[top : bottom + 1, left] = WALL
        grid[top : bottom + 1, right] = WALL
        h = heights[i]
        if h:
            grid[bottom - h : bottom, left + 1 : right] = colors[i]

    valves = []
    for i, source in enumerate(sources):
        c0 = bounds[i][1] + 1
        c1 = bounds[i + 1][0] - 1
        vc = random.randint(c0, c1)
        vr = bottom - 1
        grid[vr, vc] = colors[source]
        valves.append((vr, vc))

    return grid, valves


def _gen_attempt() -> np.ndarray | None:
    layout = _pick_layout()
    if layout is None:
        return None
    rows, top, inner_h, _widths, _gaps, bounds = layout
    n = len(bounds)

    active = random.random() < 0.5
    sampled = _heights_and_sources(n, inner_h, active)
    if sampled is None:
        return None
    heights, sources = sampled

    colors = random.sample(LIQUID_COLORS, n)
    grid, valve_positions = _render(rows, top, inner_h, bounds, heights, colors, sources)
    if max(grid.shape) > 30:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    valve_mask = np.zeros_like(grid, dtype=bool)
    for r, c in valve_positions:
        valve_mask[r, c] = True
    liquid_changed = bool(np.any((out != grid) & ~valve_mask))
    if active != liquid_changed:
        return None
    if not np.all(out[valve_mask] == BG):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
