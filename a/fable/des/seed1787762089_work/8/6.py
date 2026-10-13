import random

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

H_BOUNDS = (14, 20)
W_BOUNDS = (18, 24)
GEN_ATTEMPTS = 400
ROUTE_ATTEMPTS = 250


def _path_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc > ac else -1
        return [(ar, c) for c in range(ac + step, bc, step)]
    if ac == bc:
        step = 1 if br > ar else -1
        return [(r, ac) for r in range(ar + step, br, step)]

    cells = []
    hstep = 1 if bc > ac else -1
    for c in range(ac + hstep, bc + hstep, hstep):
        cells.append((ar, c))
    vstep = 1 if br > ar else -1
    for r in range(ar + vstep, br, vstep):
        cells.append((r, bc))
    return cells


def _legend_candidates(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], list[int]]]:
    h, w = grid.shape
    candidates: list[tuple[tuple[int, int, int, int], list[int]]] = []
    for inner_len in (3, 4):
        width = inner_len + 2
        if h < 3 or w < width:
            continue
        for r0, c0 in ((0, 0), (0, w - width), (h - 3, 0), (h - 3, w - width)):
            sub = grid[r0 : r0 + 3, c0 : c0 + width]
            if sub.shape != (3, width):
                continue
            if not np.all(sub[0, :] == GREY):
                continue
            if not np.all(sub[2, :] == GREY):
                continue
            if not np.all(sub[:, 0] == GREY):
                continue
            if not np.all(sub[:, -1] == GREY):
                continue
            seq = [int(x) for x in sub[1, 1:-1]]
            if any(color in (BG, GREY) for color in seq):
                continue
            if len(set(seq)) != len(seq):
                continue
            candidates.append(((r0, c0, r0 + 2, c0 + width - 1), seq))
    return candidates


def _find_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[int], np.ndarray]:
    candidates = _legend_candidates(grid)
    assert len(candidates) == 1, "expected exactly one corner legend"
    bbox, seq = candidates[0]
    r0, c0, r1, c1 = bbox
    mask = np.zeros(grid.shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return bbox, seq, mask


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _bbox, sequence, legend_mask = _find_legend(g)

    stations: dict[int, tuple[int, int]] = {}
    for color in sequence:
        coords = np.argwhere((g == color) & (~legend_mask))
        assert len(coords) == 1, f"expected one station for color {color}"
        stations[color] = (int(coords[0, 0]), int(coords[0, 1]))

    out = g.copy()
    allowed_colors = set(sequence)
    for r, c in np.argwhere((~legend_mask) & (g != BG)):
        if int(g[r, c]) not in allowed_colors:
            out[r, c] = BG

    for src_color, dst_color in zip(sequence, sequence[1:]):
        start = stations[src_color]
        end = stations[dst_color]
        for r, c in _path_cells(start, end):
            if out[r, c] == BG:
                out[r, c] = src_color

    return out


def _legend_bbox(h: int, w: int, k: int, corner: str) -> tuple[int, int, int, int]:
    width = k + 2
    if corner == "tl":
        return 0, 0, 2, width - 1
    if corner == "tr":
        return 0, w - width, 2, w - 1
    if corner == "bl":
        return h - 3, 0, h - 1, width - 1
    assert corner == "br"
    return h - 3, w - width, h - 1, w - 1


def _paint_legend(grid: np.ndarray, bbox: tuple[int, int, int, int], seq: list[int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
    grid[r0 + 1, c0 + 1 : c1] = np.array(seq, dtype=int)


def _build_blocked(h: int, w: int, bbox: tuple[int, int, int, int]) -> np.ndarray:
    blocked = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    blocked[0, :] = True
    blocked[h - 1, :] = True
    blocked[:, 0] = True
    blocked[:, w - 1] = True
    return blocked


def _is_isolated(pos: tuple[int, int], points: set[tuple[int, int]]) -> bool:
    r, c = pos
    for pr, pc in points:
        if abs(pr - r) + abs(pc - c) == 1:
            return False
    return True


def _matches_kind(a: tuple[int, int], b: tuple[int, int], kind: str) -> bool:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    if kind == "H":
        return dr == 0 and abs(dc) >= 2
    if kind == "V":
        return dc == 0 and abs(dr) >= 2
    assert kind == "L"
    return abs(dr) >= 2 and abs(dc) >= 2


def _choose_segment_kinds(k: int) -> list[str]:
    kinds = ["L"] * (k - 1)
    mode = random.choice(("L", "H", "V"))
    if mode != "L":
        kinds[random.randrange(k - 1)] = mode
    return kinds


def _sample_route(
    h: int, w: int, blocked: np.ndarray, kinds: list[str]
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    allowed = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if not blocked[r, c]
    ]
    if len(allowed) < len(kinds) + 1:
        return None

    for _ in range(ROUTE_ATTEMPTS):
        start = random.choice(allowed)
        stations = [start]
        station_set = {start}
        used_path: set[tuple[int, int]] = set()
        ok = True

        for kind in kinds:
            a = stations[-1]
            candidates: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
            for b in allowed:
                if b in station_set or b in used_path:
                    continue
                if not _matches_kind(a, b, kind):
                    continue
                if not _is_isolated(b, station_set):
                    continue
                cells = _path_cells(a, b)
                if not cells:
                    continue
                if any(blocked[r, c] for r, c in cells):
                    continue
                if any((r, c) in used_path for r, c in cells):
                    continue
                if any((r, c) in station_set for r, c in cells):
                    continue
                candidates.append((b, cells))
            if not candidates:
                ok = False
                break
            b, cells = random.choice(candidates)
            stations.append(b)
            station_set.add(b)
            used_path.update(cells)

        if ok:
            return stations, used_path
    return None


def _place_distractors(
    grid: np.ndarray,
    station_points: list[tuple[int, int]],
    used_path: set[tuple[int, int]],
    blocked: np.ndarray,
    legend_colors: list[int],
) -> bool:
    available_colors = [color for color in PALETTE if color not in legend_colors]
    count = random.randint(1, min(3, len(available_colors)))
    distractor_colors = random.sample(available_colors, count)

    h, w = grid.shape
    candidates = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if not blocked[r, c]
    ]
    random.shuffle(candidates)

    placed = set(station_points)
    for color in distractor_colors:
        pos = None
        for cand in candidates:
            if cand in placed or cand in used_path:
                continue
            if not _is_isolated(cand, placed):
                continue
            pos = cand
            break
        if pos is None:
            return False
        grid[pos] = color
        placed.add(pos)
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*H_BOUNDS)
    w = random.randint(*W_BOUNDS)
    k = random.choice((3, 4))
    legend_colors = random.sample(list(PALETTE), k)
    corner = random.choice(("tl", "tr", "bl", "br"))

    grid = np.full((h, w), BG, dtype=int)
    bbox = _legend_bbox(h, w, k, corner)
    _paint_legend(grid, bbox, legend_colors)
    blocked = _build_blocked(h, w, bbox)

    kinds = _choose_segment_kinds(k)
    route = _sample_route(h, w, blocked, kinds)
    if route is None:
        return None
    stations, used_path = route

    for color, pos in zip(legend_colors, stations):
        grid[pos] = color

    if not _place_distractors(grid, stations, used_path, blocked, legend_colors):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
