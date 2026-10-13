from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
DIR4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GENERATION_RETRIES = 300
PLACEMENT_RETRIES = 500


def _neighbors(cell: tuple[int, int], mask: np.ndarray) -> list[tuple[int, int]]:
    r, c = cell
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
            out.append((rr, cc))
    return out


def _rotate(pt: tuple[int, int], turns: int) -> tuple[int, int]:
    r, c = pt
    turns %= 4
    for _ in range(turns):
        r, c = c, -r
    return (r, c)


def _square_cells(center: tuple[int, int], side: int) -> set[tuple[int, int]]:
    r, c = center
    top = r - (side - 1) // 2
    left = c - (side - 1) // 2
    return {
        (rr, cc)
        for rr in range(top, top + side)
        for cc in range(left, left + side)
    }


def _buffer_cells(cells: set[tuple[int, int]], radius: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                out.add((r + dr, c + dc))
    return out


def _normalize_component(
    path: list[tuple[int, int]],
    extra: set[tuple[int, int]],
    branch: str,
) -> dict:
    pipe = set(path) | set(extra)
    sink = path[-1]
    side = min(len(path) - 1, 5)
    reserve = _buffer_cells(pipe, radius=1) | _buffer_cells(_square_cells(sink, side), radius=1)
    all_cells = pipe | reserve
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)

    def shift(cell: tuple[int, int]) -> tuple[int, int]:
        return (cell[0] - min_r, cell[1] - min_c)

    path2 = [shift(cell) for cell in path]
    extra2 = {shift(cell) for cell in extra}
    pipe2 = {shift(cell) for cell in pipe}
    reserve2 = {shift(cell) for cell in reserve}
    height = max(r for r, _ in reserve2) + 1
    width = max(c for _, c in reserve2) + 1
    return {
        "path": path2,
        "extra": extra2,
        "pipe": pipe2,
        "reserve": reserve2,
        "source": path2[0],
        "sink": path2[-1],
        "n": len(path2) - 1,
        "side": side,
        "height": height,
        "width": width,
        "branch": branch,
    }


def _make_t_straight(rng: np.random.Generator) -> dict:
    n = int(rng.choice([3, 5, 7]))
    anchor = int(rng.integers(1, n - 1))
    path = [(0, x) for x in range(n + 1)]
    extra = {(1, anchor), (1, anchor + 1)}
    turns = int(rng.integers(0, 4))
    return _normalize_component([_rotate(p, turns) for p in path], {_rotate(p, turns) for p in extra}, "t_straight")


def _make_t_right(rng: np.random.Generator) -> dict:
    n = int(rng.choice([5, 7, 9]))
    stem = int(rng.integers(2, n - 1))
    tail = n - stem - 1
    path = [(stem + 1, 0)]
    path.extend((y, 0) for y in range(stem, -1, -1))
    path.extend((0, x) for x in range(1, tail + 1))
    extra = {(0, -1), (1, -1)}
    turns = int(rng.integers(0, 4))
    return _normalize_component([_rotate(p, turns) for p in path], {_rotate(p, turns) for p in extra}, "t_right")


def _make_cross(rng: np.random.Generator) -> dict:
    n = int(rng.choice([5, 7, 9]))
    center = int(rng.integers(2, n - 1))
    path = [(0, x) for x in range(n + 1)]
    extra = {(-1, center), (-1, center - 1), (1, center), (1, center + 1)}
    turns = int(rng.integers(0, 4))
    return _normalize_component([_rotate(p, turns) for p in path], {_rotate(p, turns) for p in extra}, "cross")


def _make_simple(rng: np.random.Generator) -> dict:
    n = int(rng.choice([2, 3, 4]))
    style = "straight" if n == 2 else str(rng.choice(["straight", "bend"]))
    if style == "straight":
        path = [(0, x) for x in range(n + 1)]
    else:
        vertical = int(rng.integers(1, n))
        horizontal = n - vertical
        path = [(vertical, 0)]
        path.extend((y, 0) for y in range(vertical - 1, -1, -1))
        path.extend((0, x) for x in range(1, horizontal + 1))
    turns = int(rng.integers(0, 4))
    return _normalize_component([_rotate(p, turns) for p in path], set(), "simple")


def _build_component(kind: str, rng: np.random.Generator) -> dict:
    if kind == "t_straight":
        return _make_t_straight(rng)
    if kind == "t_right":
        return _make_t_right(rng)
    if kind == "cross":
        return _make_cross(rng)
    if kind == "simple":
        return _make_simple(rng)
    raise ValueError(f"unknown component kind: {kind}")


def _place_components(
    comps: list[dict],
    colors: np.ndarray,
    h: int,
    w: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.full((h, w), BG, dtype=int)
    reserved: set[tuple[int, int]] = set()
    placed: list[dict] = []

    order = list(range(len(comps)))
    rng.shuffle(order)
    order.sort(key=lambda idx: len(comps[idx]["reserve"]), reverse=True)

    for idx in order:
        comp = comps[idx]
        color = int(colors[idx])
        ok = False
        for _ in range(PLACEMENT_RETRIES):
            if comp["height"] > h or comp["width"] > w:
                return None
            dr = int(rng.integers(0, h - comp["height"] + 1))
            dc = int(rng.integers(0, w - comp["width"] + 1))
            reserve = {(r + dr, c + dc) for r, c in comp["reserve"]}
            if reserve & reserved:
                continue
            pipe = {(r + dr, c + dc) for r, c in comp["pipe"]}
            path = [(r + dr, c + dc) for r, c in comp["path"]]
            source = path[0]
            for r, c in pipe:
                grid[r, c] = GREY
            grid[source] = color
            reserved |= reserve
            placed.append(
                {
                    "source": source,
                    "sink": path[-1],
                    "path": path,
                    "pipe": pipe,
                    "n": comp["n"],
                    "side": comp["side"],
                    "color": color,
                    "branch": comp["branch"],
                }
            )
            ok = True
            break
        if not ok:
            return None
    return grid, placed


def _infer_colors(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    counts = Counter(map(int, g.ravel()))
    bg = BG if BG in counts else counts.most_common(1)[0][0]
    non_bg = [int(c) for c in counts if int(c) != int(bg)]
    assert non_bg, "expected non-background colors"
    grey = max(non_bg, key=lambda c: (counts[c], -c))
    sources = [tuple(map(int, rc)) for rc in np.argwhere((g != bg) & (g != grey))]
    assert sources, "expected at least one source"
    return int(bg), int(grey), sources


def _trace_source(
    grid: np.ndarray,
    passable: np.ndarray,
    source: tuple[int, int],
    grey: int,
    all_sources: set[tuple[int, int]],
) -> tuple[tuple[int, int], int]:
    prev: tuple[int, int] | None = None
    cur = source
    seen: set[tuple[tuple[int, int] | None, tuple[int, int]]] = set()

    n_grey = 0
    while True:
        state = (prev, cur)
        assert state not in seen, "cycle encountered while tracing"
        seen.add(state)

        nbrs = _neighbors(cur, passable)
        deg = len(nbrs)
        if cur != source:
            assert int(grid[cur]) == int(grey), "only source endpoints may be colored"
            if deg == 1:
                return cur, n_grey

        if prev is None:
            assert deg == 1, "source must be a dead end"
            nxt = nbrs[0]
        else:
            dr = cur[0] - prev[0]
            dc = cur[1] - prev[1]
            candidates = [cell for cell in nbrs if cell != prev]
            if deg == 2:
                assert len(candidates) == 1
                nxt = candidates[0]
            elif deg == 3:
                straight = (cur[0] + dr, cur[1] + dc)
                if straight in candidates:
                    nxt = straight
                else:
                    right = (cur[0] + dc, cur[1] - dr)
                    assert right in candidates, "expected right turn at T junction"
                    nxt = right
            elif deg == 4:
                straight = (cur[0] + dr, cur[1] + dc)
                assert straight in candidates, "expected straight continuation at cross"
                nxt = straight
            else:
                raise AssertionError(f"unexpected degree {deg} at {cur}")

        assert nxt not in all_sources, "paths may not terminate at another source"
        prev, cur = cur, nxt
        if int(grid[cur]) == int(grey):
            n_grey += 1


def _trace_all(grid: np.ndarray) -> list[tuple[tuple[int, int], int, int]]:
    g = np.asarray(grid, dtype=int)
    bg, grey, sources = _infer_colors(g)
    passable = g != bg
    source_set = set(sources)
    routes: list[tuple[tuple[int, int], int, int]] = []
    for source in sources:
        sink, n_grey = _trace_source(g, passable, source, grey, source_set)
        routes.append((sink, n_grey, int(g[source])))
    return routes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow each source along the pipe network and paint a distance-sized square
    around the reached dead end, leaving pipe cells unchanged.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg, grey, sources = _infer_colors(g)
    passable = g != bg
    source_set = set(sources)
    original_bg = g == bg

    seen_sinks: set[tuple[int, int]] = set()
    for source in sources:
        sink, n_grey = _trace_source(g, passable, source, grey, source_set)
        assert sink not in seen_sinks, "each sink must belong to exactly one source"
        seen_sinks.add(sink)
        side = min(n_grey, 5)
        for rr, cc in _square_cells(sink, side):
            if 0 <= rr < h and 0 <= cc < w and bool(original_bg[rr, cc]):
                out[rr, cc] = int(g[source])
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    kinds = ["t_straight", "t_right", "cross", "simple"]
    if bool(rng.integers(0, 2)):
        kinds.append(str(rng.choice(["t_straight", "t_right", "cross", "simple"])))

    comps = [_build_component(kind, rng) for kind in kinds]
    sizes = {comp["side"] for comp in comps}
    if len(sizes) < 2:
        return None

    total_reserved = sum(len(comp["reserve"]) for comp in comps)
    max_dim = max(max(comp["height"], comp["width"]) for comp in comps)
    base = max(max_dim + 1, int(np.ceil(np.sqrt(total_reserved * 1.8))))
    if base > 28:
        return None

    h = int(rng.integers(base, min(31, base + 4)))
    w = int(rng.integers(base, min(31, base + 4)))
    colors = rng.choice(PALETTE, size=len(comps), replace=False)
    placed_result = _place_components(comps, colors, h, w, rng)
    if placed_result is None:
        return None
    grid, placed = placed_result

    try:
        routes = _trace_all(grid)
    except AssertionError:
        return None

    if len(routes) != len(placed):
        return None
    sink_to_expected = {tuple(comp["sink"]): (int(comp["n"]), int(comp["color"])) for comp in placed}
    for sink, n_grey, color in routes:
        if sink not in sink_to_expected:
            return None
        exp_n, exp_color = sink_to_expected[sink]
        if n_grey != exp_n or color != exp_color:
            return None

    bg, grey, _sources = _infer_colors(grid)
    passable = grid != bg
    grey_dead_ends = {
        tuple(map(int, rc))
        for rc in np.argwhere(grid == grey)
        if len(_neighbors(tuple(map(int, rc)), passable)) == 1
    }
    if grey_dead_ends != set(sink_to_expected):
        return None

    try:
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
