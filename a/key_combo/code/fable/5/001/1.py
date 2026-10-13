from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
DIRS: tuple[Coord, ...] = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                yield r + dr, c + dc


def _bbox(cells: set[Coord]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _components_of_color(g: np.ndarray, color: int) -> list[set[Coord]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[set[Coord]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != int(color):
                continue
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            comp: set[Coord] = set()
            while q:
                rr, cc = q.popleft()
                comp.add((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == int(color)
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _cup_geometry(tail: Coord, direction: Coord, side: int, depth: int) -> tuple[set[Coord], set[Coord]]:
    """Return frame cells and interior cells for one single-tail U-shaped cup."""
    tr, tc = map(int, tail)
    dr, dc = map(int, direction)
    half = int(side) // 2
    frame: set[Coord] = {(tr, tc)}
    interior: set[Coord] = set()

    if (dr, dc) == (0, 1):  # tail points east; body and opening lie west of it
        closed_c = tc - 1
        open_c = tc - int(depth)
        top = tr - half
        bottom = tr + half
        for r in range(top, bottom + 1):
            frame.add((r, closed_c))
        for c in range(open_c, closed_c + 1):
            frame.add((top, c))
            frame.add((bottom, c))
        for r in range(top + 1, bottom):
            for c in range(open_c, closed_c):
                interior.add((r, c))

    elif (dr, dc) == (0, -1):  # tail points west; body and opening lie east of it
        closed_c = tc + 1
        open_c = tc + int(depth)
        top = tr - half
        bottom = tr + half
        for r in range(top, bottom + 1):
            frame.add((r, closed_c))
        for c in range(closed_c, open_c + 1):
            frame.add((top, c))
            frame.add((bottom, c))
        for r in range(top + 1, bottom):
            for c in range(closed_c + 1, open_c + 1):
                interior.add((r, c))

    elif (dr, dc) == (1, 0):  # tail points down; body and opening lie above it
        closed_r = tr - 1
        open_r = tr - int(depth)
        left = tc - half
        right = tc + half
        for c in range(left, right + 1):
            frame.add((closed_r, c))
        for r in range(open_r, closed_r + 1):
            frame.add((r, left))
            frame.add((r, right))
        for r in range(open_r, closed_r):
            for c in range(left + 1, right):
                interior.add((r, c))

    else:
        assert (dr, dc) == (-1, 0)  # tail points up; body and opening lie below it
        closed_r = tr + 1
        open_r = tr + int(depth)
        left = tc - half
        right = tc + half
        for c in range(left, right + 1):
            frame.add((closed_r, c))
        for r in range(closed_r, open_r + 1):
            frame.add((r, left))
            frame.add((r, right))
        for r in range(closed_r + 1, open_r + 1):
            for c in range(left + 1, right):
                interior.add((r, c))

    return frame, interior


def _match_cup_component(cells: set[Coord], shape: tuple[int, int]) -> dict | None:
    r0, c0, r1, c1 = _bbox(cells)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    trials: list[tuple[Coord, int, int]] = []

    if h >= 5 and h % 2 == 1 and w >= 5:
        trials.append(((0, 1), h, w - 1))
        trials.append(((0, -1), h, w - 1))
    if w >= 5 and w % 2 == 1 and h >= 5:
        trials.append(((1, 0), w, h - 1))
        trials.append(((-1, 0), w, h - 1))

    matches = []
    for direction, side, depth in trials:
        dr, dc = direction
        if (dr, dc) == (0, 1):
            tail = (r0 + side // 2, c1)
        elif (dr, dc) == (0, -1):
            tail = (r0 + side // 2, c0)
        elif (dr, dc) == (1, 0):
            tail = (r1, c0 + side // 2)
        else:
            tail = (r0, c0 + side // 2)
        frame, interior = _cup_geometry(tail, direction, side, depth)
        if cells == frame and all(_in_bounds(shape, r, c) for r, c in frame | interior):
            matches.append(
                {
                    "tail": tail,
                    "direction": direction,
                    "side": int(side),
                    "depth": int(depth),
                    "frame": frame,
                    "interior": interior,
                }
            )

    if len(matches) != 1:
        return None
    return matches[0]


def _find_cups(g: np.ndarray, bg: int) -> list[dict]:
    cups: list[dict] = []
    for color in sorted(set(map(int, g.ravel())) - {int(bg)}):
        color_cups = []
        for comp in _components_of_color(g, color):
            if len(comp) < 12:
                continue
            matched = _match_cup_component(comp, g.shape)
            if matched is not None:
                color_cups.append({**matched, "border": int(color)})
        assert len(color_cups) <= 1, "one border color may describe at most one cup"
        cups.extend(color_cups)

    assert len(cups) >= 2, "expected at least two cup frames"
    tails = [cup["tail"] for cup in cups]
    assert len(tails) == len(set(tails)), "cup tails must be unique"

    border_colors = {int(cup["border"]) for cup in cups}
    for cup in cups:
        colors = {
            int(g[r, c])
            for r, c in cup["interior"]
            if int(g[r, c]) != int(bg)
        }
        assert len(colors) == 1, "each cup interior must contain one content color"
        content = next(iter(colors))
        assert content not in border_colors, "content color cannot be a border color"
        content_cells = {
            (r, c)
            for r, c in cup["interior"]
            if int(g[r, c]) == int(content)
        }
        assert content_cells, "each cup must contain ammunition pixels"
        cup["content"] = int(content)
        cup["content_cells"] = content_cells
        cup["ammo"] = len(content_cells)

    return cups


def _find_initial_spark(g: np.ndarray, bg: int, cups: list[dict]) -> tuple[Coord, int]:
    border_colors = {int(cup["border"]) for cup in cups}
    content_colors = {int(cup["content"]) for cup in cups}
    tails = {cup["tail"]: i for i, cup in enumerate(cups)}
    counts = Counter(map(int, g.ravel()))
    candidates: list[tuple[Coord, int]] = []

    for color, count in counts.items():
        color = int(color)
        if color == int(bg) or color in border_colors or color in content_colors or int(count) != 1:
            continue
        locs = np.argwhere(g == color)
        assert locs.shape[0] == 1
        r, c = map(int, locs[0])
        adjacent = [
            idx
            for nr, nc in _neighbors8(r, c)
            for tail, idx in tails.items()
            if (nr, nc) == tail
        ]
        if len(adjacent) == 1:
            candidates.append(((r, c), adjacent[0]))

    assert len(candidates) == 1, "expected one unique spark adjacent to one tail"
    return candidates[0]


def _solve_with_meta(grid: np.ndarray) -> tuple[np.ndarray, list[dict], set[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    cups = _find_cups(g, bg)
    _spark, initial = _find_initial_spark(g, bg, cups)

    tail_to_cup = {cup["tail"]: i for i, cup in enumerate(cups)}
    out = g.copy()
    fired: set[int] = set()
    q: deque[int] = deque([initial])

    while q:
        idx = q.popleft()
        if idx in fired:
            continue
        cup = cups[idx]
        fired.add(idx)

        for r, c in cup["content_cells"]:
            out[r, c] = bg

        dr, dc = cup["direction"]
        r = int(cup["tail"][0]) + int(dr)
        c = int(cup["tail"][1]) + int(dc)
        for _ in range(int(cup["ammo"])):
            if not _in_bounds(out.shape, r, c):
                break
            if int(out[r, c]) != int(bg):
                hit = tail_to_cup.get((r, c))
                if hit is not None and hit not in fired:
                    q.append(hit)
                break
            out[r, c] = int(cup["content"])
            r += int(dr)
            c += int(dc)

    return out, cups, fired


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire only the cup frames ignited by the spark or by a previous shot hitting their tail spike.

    Input structure:
    - The dominant color is the background.
    - Several separated U-shaped cup frames appear, each with a unique border color.
    - Every cup has one single-cell tail spike on its closed side and content pixels of one non-border color inside its open mouth.
    - Exactly one globally unique spark pixel is 8-adjacent to one cup tail.

    Rule:
    - The spark-adjacent cup fires first; when a cup fires, its interior content pixels are cleared.
    - From that cup's tail, advance in the tail direction for at most as many cells as it had content pixels.
    - Background cells on that ray are painted with the cup content color; the ray stops at the first non-background cell or the grid edge.
    - If the blocking non-background cell is another cup's tail, that cup fires next; cups never reached by such a tail hit stay unchanged.

    Key insight:
    - The number of content pixels is the ammunition range, so the visible chain depends on whether each shot can actually reach the next tail, not on clearing every cup.
    """
    return _solve_with_meta(grid)[0]


# ---------------- generator ----------------

PALETTE = tuple(range(10))
MAX_ATTEMPTS = 400


def _paint_cup(
    grid: np.ndarray,
    tail: Coord,
    direction: Coord,
    side: int,
    depth: int,
    border: int,
    content: int,
    ammo: int,
) -> None:
    frame, interior = _cup_geometry(tail, direction, side, depth)
    assert len(interior) >= int(ammo)
    for r, c in frame:
        assert _in_bounds(grid.shape, r, c)
        assert int(grid[r, c]) == int(_GEN_BG)
        grid[r, c] = int(border)
    cells = random.sample(sorted(interior), int(ammo))
    for r, c in cells:
        assert _in_bounds(grid.shape, r, c)
        assert int(grid[r, c]) == int(_GEN_BG)
        grid[r, c] = int(content)


def _path_cells(a: Coord, b: Coord) -> set[Coord]:
    ar, ac = a
    br, bc = b
    cells: set[Coord] = set()
    if ar == br:
        step = 1 if bc > ac else -1
        for c in range(ac + step, bc, step):
            cells.add((ar, c))
    else:
        assert ac == bc
        step = 1 if br > ar else -1
        for r in range(ar + step, br, step):
            cells.add((r, ac))
    return cells


def _clear_ray(grid: np.ndarray, bg: int, tail: Coord, direction: Coord, distance: int) -> bool:
    r, c = tail
    dr, dc = direction
    for _ in range(int(distance) - 1):
        r += dr
        c += dc
        if not _in_bounds(grid.shape, r, c) or int(grid[r, c]) != int(bg):
            return False
    return True


def _content_capacity(side: int, depth: int) -> int:
    return (int(side) - 2) * (int(depth) - 1)


def _distance(a: Coord, b: Coord) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _choose_dimensions() -> tuple[int, int, list[int], list[int], int, int, int, int] | None:
    h = random.randint(23, 30)
    w = random.randint(23, 30)
    sides = [random.choice((5, 7)) for _ in range(4)]
    depths = [random.choice((4, 5, 6)) for _ in range(4)]
    caps = [_content_capacity(sides[i], depths[i]) for i in range(4)]

    max_horizontal = min(10, caps[0], caps[2])
    max_vertical = min(10, caps[1], caps[3])
    if max_horizontal < 6 or max_vertical < 6:
        return None

    left_min = max(depths[0], sides[3] // 2) + 2
    right_max = w - 1 - max(depths[2], sides[1] // 2) - 2
    top_min = max(sides[0] // 2, depths[1]) + 2
    bottom_max = h - 1 - max(sides[2] // 2, depths[3]) - 2

    horizontal_options = [
        d for d in range(6, max_horizontal + 1)
        if left_min + d <= right_max
    ]
    vertical_options = [
        d for d in range(6, max_vertical + 1)
        if top_min + d <= bottom_max
    ]
    if not horizontal_options or not vertical_options:
        return None

    dx = random.choice(horizontal_options)
    dy = random.choice(vertical_options)
    left = random.randint(left_min, right_max - dx)
    top = random.randint(top_min, bottom_max - dy)
    return h, w, sides, depths, top, left, dy, dx


def _transform_grid(g: np.ndarray) -> np.ndarray:
    k = random.randrange(4)
    out = np.rot90(g, k)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _semantic_accepts(inp: np.ndarray) -> bool:
    try:
        out, cups, fired = _solve_with_meta(inp)
    except Exception:
        return False
    if not (len(cups) == 4 and 2 <= len(fired) <= 3):
        return False

    bg = _get_bg(inp)
    cleared = 0
    preserved = 0
    for i, cup in enumerate(cups):
        content_cells = cup["content_cells"]
        if i in fired:
            if not all(int(out[r, c]) == bg for r, c in content_cells):
                return False
            cleared += len(content_cells)
        else:
            if not all(int(out[r, c]) == int(inp[r, c]) for r, c in content_cells):
                return False
            preserved += len(content_cells)

    painted = int(np.count_nonzero((out != inp) & (out != bg)))
    return cleared > 0 and preserved > 0 and painted >= 6


def _gen_attempt() -> np.ndarray | None:
    global _GEN_BG

    dims = _choose_dimensions()
    if dims is None:
        return None
    h, w, sides, depths, top, left, dy, dx = dims
    bottom = top + dy
    right = left + dx
    tails = [(top, left), (top, right), (bottom, right), (bottom, left)]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    distances = [
        _distance(tails[i], tails[(i + 1) % 4])
        for i in range(4)
    ]

    colors = list(PALETTE)
    _GEN_BG = random.choice(colors)
    rest = [c for c in colors if c != _GEN_BG]
    random.shuffle(rest)
    borders = rest[:4]
    contents = rest[4:8]
    spark = rest[8]

    reach_count = random.choice((2, 3))
    ammo: list[int] = []
    for i in range(4):
        cap = _content_capacity(sides[i], depths[i])
        if i < reach_count - 1:
            ammo.append(distances[i])
        elif i == reach_count - 1:
            ammo.append(random.randint(2, min(cap, distances[i] - 1)))
        else:
            ammo.append(random.randint(2, min(cap, max(2, distances[i] - 1))))
    if any(ammo[i] > _content_capacity(sides[i], depths[i]) for i in range(4)):
        return None

    grid = np.full((h, w), _GEN_BG, dtype=int)
    try:
        for i in range(4):
            _paint_cup(
                grid,
                tails[i],
                directions[i],
                sides[i],
                depths[i],
                borders[i],
                contents[i],
                ammo[i],
            )
    except AssertionError:
        return None

    path = set()
    for i in range(4):
        path |= _path_cells(tails[i], tails[(i + 1) % 4])
    spark_candidates = [
        rc for rc in _neighbors8(*tails[0])
        if _in_bounds(grid.shape, *rc)
        and int(grid[rc]) == int(_GEN_BG)
        and rc not in path
        and rc != (tails[0][0], tails[0][1] + 1)
    ]
    if not spark_candidates:
        return None
    sr, sc = random.choice(spark_candidates)
    grid[sr, sc] = spark

    for i in range(reach_count - 1):
        if not _clear_ray(grid, _GEN_BG, tails[i], directions[i], distances[i]):
            return None

    transformed = _transform_grid(grid)
    if not _semantic_accepts(transformed):
        return None
    return transformed


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fire-chain cup input")


_GEN_BG = 0
