import random
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = tuple(range(10))
GRID_ROWS = (12, 24)
GRID_COLS = (12, 24)
COLOR_COUNT = (1, 3)
RECT_SIDE = (2, 6)
GEN_ATTEMPTS = 400
OBJECT_ATTEMPTS = 80


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return components


def _claim(grid: np.ndarray, claims: dict[tuple[int, int], int], coord: tuple[int, int], color: int, bg: int) -> None:
    r, c = coord
    assert 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]
    base = int(grid[r, c])
    assert base == bg or base == color
    prior = claims.get(coord)
    if prior is not None:
        assert prior == color
    claims[coord] = color


def _claim_range(
    grid: np.ndarray,
    claims: dict[tuple[int, int], int],
    start: tuple[int, int],
    end: tuple[int, int],
    step: tuple[int, int],
    skip: tuple[int, int],
    color: int,
    bg: int,
) -> None:
    r, c = start
    while True:
        if (r, c) != skip:
            _claim(grid, claims, (r, c), color, bg)
        if (r, c) == end:
            break
        r += step[0]
        c += step[1]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)
    components = _find_components(g, bg)
    assert components

    by_color: dict[int, list[dict]] = {}
    for comp in components:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    claims: dict[tuple[int, int], int] = {}
    clears: set[tuple[int, int]] = set()

    for color, comps in by_color.items():
        rectangles = []
        markers = []
        for comp in comps:
            r0, c0, r1, c1 = comp["bbox"]
            if comp["area"] > 1 and comp["area"] == (r1 - r0 + 1) * (c1 - c0 + 1):
                rectangles.append(comp)
            elif comp["area"] == 1:
                markers.append(comp)
            else:
                raise AssertionError(f"unexpected component for color {color}: {comp}")

        assert len(rectangles) == 1 and markers
        rect = rectangles[0]
        r0, c0, r1, c1 = rect["bbox"]

        for marker in markers:
            mr, mc = marker["coords"][0]
            horizontal = r0 <= mr <= r1 and (mc < c0 or mc > c1)
            vertical = c0 <= mc <= c1 and (mr < r0 or mr > r1)
            assert horizontal ^ vertical

            if horizontal:
                dc = -1 if mc < c0 else 1
                side_c = c0 - 1 if dc < 0 else c1 + 1
                for rr in (mr - 1, mr, mr + 1):
                    _claim(g, claims, (rr, side_c), color, bg)
                for rr in (mr - 1, mr + 1):
                    _claim(g, claims, (rr, mc), color, bg)
                _claim_range(g, claims, (mr, side_c), (mr, mc + dc), (0, dc), (mr, mc), color, bg)
            else:
                dr = -1 if mr < r0 else 1
                side_r = r0 - 1 if dr < 0 else r1 + 1
                for cc in (mc - 1, mc, mc + 1):
                    _claim(g, claims, (side_r, cc), color, bg)
                for cc in (mc - 1, mc + 1):
                    _claim(g, claims, (mr, cc), color, bg)
                _claim_range(g, claims, (side_r, mc), (mr + dr, mc), (dr, 0), (mr, mc), color, bg)

            clears.add((mr, mc))

    assert not (set(claims) & clears)

    out = g.copy()
    for (r, c), color in claims.items():
        out[r, c] = color
    for r, c in clears:
        out[r, c] = bg
    return out


def _rect_region_clear(grid: np.ndarray, bg: int, r0: int, c0: int, h: int, w: int) -> bool:
    r1, c1 = r0 + h - 1, c0 + w - 1
    rr0 = max(0, r0 - 1)
    rr1 = min(grid.shape[0], r1 + 2)
    cc0 = max(0, c0 - 1)
    cc1 = min(grid.shape[1], c1 + 2)
    return np.all(grid[rr0:rr1, cc0:cc1] == bg)


def _marker_clear(grid: np.ndarray, bg: int, r: int, c: int) -> bool:
    if int(grid[r, c]) != bg:
        return False
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) != bg:
            return False
    return True


def _sample_spaced_positions(start: int, end: int, count: int) -> list[int] | None:
    positions = list(range(start, end + 1))
    random.shuffle(positions)
    chosen: list[int] = []
    for pos in positions:
        if all(abs(pos - prev) >= 3 for prev in chosen):
            chosen.append(pos)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _choose_marker_plan(r0: int, c0: int, h: int, w: int, shape: tuple[int, int]) -> list[tuple[str, int, int]]:
    H, W = shape
    options = []
    if r0 >= 3:
        options.append(("top", w, r0 - 1))
    if H - (r0 + h) >= 2:
        options.append(("bottom", w, H - (r0 + h)))
    if c0 >= 3:
        options.append(("left", h, c0 - 1))
    if W - (c0 + w) >= 2:
        options.append(("right", h, W - (c0 + w)))
    if not options:
        return []

    random.shuffle(options)
    target_total = random.randint(1, min(5, sum(2 if span >= 4 else 1 for _side, span, _depth in options)))
    plan: list[tuple[str, int, int]] = []
    remaining = target_total

    for i, (side, span, depth) in enumerate(options):
        if remaining <= 0:
            break
        sides_left = len(options) - i - 1
        max_here = 2 if span >= 4 else 1
        max_here = min(max_here, remaining)
        min_here = 1 if remaining > sides_left else 0
        count = random.randint(min_here, max_here)
        if count == 0:
            continue
        plan.append((side, count, depth))
        remaining -= count

    if remaining != 0:
        return []
    return plan


def _place_object(grid: np.ndarray, bg: int, color: int) -> bool:
    H, W = grid.shape
    for _ in range(OBJECT_ATTEMPTS):
        h = random.randint(*RECT_SIDE)
        w = random.randint(*RECT_SIDE)
        if h + 4 > H or w + 4 > W:
            continue
        r0 = random.randint(1, H - h - 2)
        c0 = random.randint(1, W - w - 2)
        if not _rect_region_clear(grid, bg, r0, c0, h, w):
            continue

        plan = _choose_marker_plan(r0, c0, h, w, grid.shape)
        if not plan:
            continue

        markers: list[tuple[int, int]] = []
        ok = True
        for side, count, depth in plan:
            if side in ("top", "bottom"):
                cols = _sample_spaced_positions(c0, c0 + w - 1, count)
                if cols is None:
                    ok = False
                    break
                max_dist = depth
                for col in cols:
                    dist = random.randint(2, max_dist)
                    row = r0 - dist if side == "top" else r0 + h - 1 + dist
                    markers.append((row, col))
            else:
                rows = _sample_spaced_positions(r0, r0 + h - 1, count)
                if rows is None:
                    ok = False
                    break
                max_dist = depth
                for row in rows:
                    dist = random.randint(2, max_dist)
                    col = c0 - dist if side == "left" else c0 + w - 1 + dist
                    markers.append((row, col))
        if not ok:
            continue
        if not all(_marker_clear(grid, bg, r, c) for r, c in markers):
            continue

        grid[r0 : r0 + h, c0 : c0 + w] = color
        for r, c in markers:
            grid[r, c] = color
        return True

    return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        H = random.randint(*GRID_ROWS)
        W = random.randint(*GRID_COLS)
        bg = random.choice(PALETTE)
        colors = [c for c in PALETTE if c != bg]
        random.shuffle(colors)
        color_count = random.randint(*COLOR_COUNT)

        grid = np.full((H, W), bg, dtype=int)
        ok = True
        for color in colors[:color_count]:
            if not _place_object(grid, bg, color):
                ok = False
                break
        if not ok:
            continue

        comp_colors = Counter(int(v) for v in grid.flat if int(v) != bg)
        if len(comp_colors) != color_count:
            continue

        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(grid, out):
            continue
        return grid

    raise RuntimeError("failed to generate valid input")
