import random
from collections import Counter, deque

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

GEN_RETRIES = 400
GRID_H_RANGE = (12, 22)
GRID_W_RANGE = (14, 24)
LOOP_COUNT_RANGE = (1, 4)
MARKER_MODE_CHOICES = ("all_zero", "mixed", "all_positive")


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray, conn8: bool = False) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neigh = NEIGHBORS8 if conn8 else NEIGHBORS4
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in neigh:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _analyze_outline_component(
    body: np.ndarray, comp: list[tuple[int, int]], bg: int
) -> dict[str, set[tuple[int, int]]] | None:
    h, w = body.shape
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    pr0, pr1 = max(0, r0 - 1), min(h - 1, r1 + 1)
    pc0, pc1 = max(0, c0 - 1), min(w - 1, c1 + 1)

    blocked = np.zeros((pr1 - pr0 + 1, pc1 - pc0 + 1), dtype=bool)
    for r, c in comp:
        blocked[r - pr0, c - pc0] = True

    reachable = np.zeros_like(blocked)
    q: deque[tuple[int, int]] = deque()
    local_h, local_w = blocked.shape

    for rr in range(local_h):
        for cc in (0, local_w - 1):
            if not blocked[rr, cc] and not reachable[rr, cc]:
                reachable[rr, cc] = True
                q.append((rr, cc))
    for cc in range(local_w):
        for rr in (0, local_h - 1):
            if not blocked[rr, cc] and not reachable[rr, cc]:
                reachable[rr, cc] = True
                q.append((rr, cc))

    while q:
        rr, cc = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < local_h and 0 <= nc < local_w and not blocked[nr, nc] and not reachable[nr, nc]:
                reachable[nr, nc] = True
                q.append((nr, nc))

    inside_local = (~blocked) & (~reachable)
    if not bool(inside_local.any()):
        return None

    inside: set[tuple[int, int]] = set()
    for rr, cc in np.argwhere(inside_local):
        inside.add((int(pr0 + rr), int(pc0 + cc)))

    outer_bg: set[tuple[int, int]] = set()
    for r, c in comp:
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if int(body[nr, nc]) != bg:
                continue
            if (nr, nc) not in inside:
                outer_bg.add((nr, nc))

    if not outer_bg:
        return None
    return {"inside": inside, "outer_bg": outer_bg}


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[1] < 2:
        raise ValueError("expected a 2D grid with a left palette column")

    h, _w = g.shape
    palette = g[:, 0].copy()
    body = g[:, 1:]
    bg = _mode_color(body)

    body_colors = [int(c) for c in np.unique(body) if int(c) != bg]
    if not body_colors:
        out = np.full_like(g, bg)
        out[:, 0] = palette
        return out

    analyzed: dict[int, list[dict[str, set[tuple[int, int]]]]] = {}
    best_color = None
    best_score = None
    for color in body_colors:
        infos = []
        for comp in _components(body == color, conn8=True):
            info = _analyze_outline_component(body, comp, bg)
            if info is not None:
                infos.append(info)
        score = (len(infos), int((body == color).sum()))
        if best_score is None or score > best_score:
            best_score = score
            best_color = color
        analyzed[color] = infos

    assert best_color is not None
    outline_color = int(best_color)
    outline_infos = analyzed[outline_color]
    assert outline_infos, "failed to identify any closed outline components"

    out = np.full_like(g, bg)
    out[:, 0] = palette

    for info in outline_infos:
        marker_count = sum(
            1
            for r, c in info["inside"]
            if int(body[r, c]) != bg and int(body[r, c]) != outline_color
        )
        assert 0 <= marker_count < h, "marker count must fit in palette rows"
        color = int(palette[marker_count])
        for r, c in info["outer_bg"]:
            out[r, c + 1] = color

    return out


def _normalize_coords(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _dilate4_coords(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        for dr, dc in NEIGHBORS4:
            out.add((r + dr, c + dc))
    return out


def _boundary_coords(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        if any((r + dr, c + dc) not in cells for dr, dc in NEIGHBORS4):
            out.add((r, c))
    return out


def _coords_component_count(cells: set[tuple[int, int]], conn8: bool = False) -> int:
    if not cells:
        return 0
    pending = set(cells)
    neigh = NEIGHBORS8 if conn8 else NEIGHBORS4
    count = 0
    while pending:
        count += 1
        start = pending.pop()
        q = deque([start])
        while q:
            r, c = q.popleft()
            for dr, dc in neigh:
                nbr = (r + dr, c + dc)
                if nbr in pending:
                    pending.remove(nbr)
                    q.append(nbr)
    return count


def _random_polyomino(rng: random.Random, area: int, box_h: int, box_w: int) -> set[tuple[int, int]]:
    start = (rng.randrange(box_h), rng.randrange(box_w))
    cells = {start}
    frontier = set()
    for dr, dc in NEIGHBORS4:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < box_h and 0 <= nc < box_w:
            frontier.add((nr, nc))
    while len(cells) < area and frontier:
        cell = rng.choice(list(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for dr, dc in NEIGHBORS4:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                frontier.add((nr, nc))
    return _normalize_coords(cells)


def _random_inner_shape(rng: random.Random) -> set[tuple[int, int]]:
    family = rng.choice(["rectangle", "diamond", "polyomino"])
    if family == "rectangle":
        ih = rng.randint(2, 4)
        iw = rng.randint(2, 5)
        return {(r, c) for r in range(ih) for c in range(iw)}
    if family == "diamond":
        radius = rng.randint(2, 3)
        center = radius
        cells = {
            (r, c)
            for r in range(2 * radius + 1)
            for c in range(2 * radius + 1)
            if abs(r - center) + abs(c - center) <= radius
        }
        return cells
    box_h = rng.randint(3, 5)
    box_w = rng.randint(3, 5)
    area = rng.randint(4, min(8, box_h * box_w))
    return _random_polyomino(rng, area, box_h, box_w)


def _build_loop_shape(rng: random.Random) -> dict[str, object] | None:
    inner = _random_inner_shape(rng)
    region = _dilate4_coords(inner)
    outline = _boundary_coords(region)
    interior = region - outline
    outer = set()
    for r, c in region:
        for dr, dc in NEIGHBORS4:
            nbr = (r + dr, c + dc)
            if nbr not in region:
                outer.add(nbr)

    if not interior or not outer:
        return None
    if _coords_component_count(outline, conn8=True) != 1:
        return None

    footprint = outline | interior | outer
    min_r = min(r for r, _ in footprint)
    min_c = min(c for _, c in footprint)
    shift = lambda cells: {(r - min_r, c - min_c) for r, c in cells}

    outline = shift(outline)
    interior = shift(interior)
    outer = shift(outer)
    footprint = shift(footprint)

    height = max(r for r, _ in footprint) + 1
    width = max(c for _, c in footprint) + 1
    return {
        "outline": outline,
        "interior": interior,
        "outer": outer,
        "height": int(height),
        "width": int(width),
    }


def _sample_palette(h: int, usable_colors: list[int], rng: random.Random) -> np.ndarray:
    palette = []
    prev = None
    for _ in range(h):
        choices = [c for c in usable_colors if c != prev] or usable_colors
        color = rng.choice(choices)
        palette.append(int(color))
        prev = color
    return np.array(palette, dtype=int)


def _reserve_ok(reserved: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = reserved.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + height + 1)
    c1 = min(w, left + width + 1)
    return not bool(reserved[r0:r1, c0:c1].any())


def _mark_reserved(reserved: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    h, w = reserved.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + height + 1)
    c1 = min(w, left + width + 1)
    reserved[r0:r1, c0:c1] = True


def _choose_marker_count(
    max_count: int, mode: str, idx: int, total: int, rng: random.Random
) -> int:
    if mode == "all_zero":
        return 0
    if mode == "all_positive":
        return rng.randint(1, max_count)
    if idx == 0:
        return 0
    if idx == total - 1:
        return rng.randint(1, max_count)
    return rng.randint(0, max_count)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    if w < 8:
        return None

    colors = list(range(10))
    bg = rng.choice(colors)
    remaining = [c for c in colors if c != bg]
    rng.shuffle(remaining)
    outline_color = remaining.pop()
    marker_color = remaining.pop()
    palette_pool = remaining[: rng.randint(3, min(6, len(remaining)))]
    if not palette_pool:
        return None
    palette = _sample_palette(h, palette_pool, rng)

    inp = np.full((h, w), bg, dtype=int)
    inp[:, 0] = palette
    expected = np.full((h, w), bg, dtype=int)
    expected[:, 0] = palette

    reserved = np.zeros((h, w), dtype=bool)
    reserved[:, 0] = True
    reserved[:, 1] = True

    mode = rng.choice(MARKER_MODE_CHOICES)
    target_loops = rng.randint(*LOOP_COUNT_RANGE)
    loops = []

    for idx in range(target_loops):
        placed = False
        for _ in range(60):
            shape = _build_loop_shape(rng)
            if shape is None:
                continue
            sh = int(shape["height"])
            sw = int(shape["width"])
            if sh + 2 > h or sw + 3 > w:
                continue
            row_choices = list(range(1, h - sh))
            col_choices = list(range(2, w - sw))
            rng.shuffle(row_choices)
            rng.shuffle(col_choices)

            max_markers = min(h - 1, len(shape["interior"]), 6)
            if max_markers <= 0:
                continue
            k = _choose_marker_count(max_markers, mode, idx, target_loops, rng)
            if k > len(shape["interior"]):
                continue

            for top in row_choices:
                found_spot = False
                for left in col_choices:
                    if not _reserve_ok(reserved, top, left, sh, sw):
                        continue

                    interior_cells = sorted(shape["interior"])
                    marker_cells = set(rng.sample(interior_cells, k)) if k else set()

                    for dr, dc in shape["outline"]:
                        inp[top + dr, left + dc] = outline_color
                    for dr, dc in marker_cells:
                        inp[top + dr, left + dc] = marker_color
                    color = int(palette[k])
                    for dr, dc in shape["outer"]:
                        expected[top + dr, left + dc] = color

                    _mark_reserved(reserved, top, left, sh, sw)
                    loops.append((shape, top, left, k))
                    placed = True
                    found_spot = True
                    break
                if found_spot:
                    break
            if placed:
                break
        if not placed:
            return None

    if not loops:
        return None

    if _mode_color(inp) != bg:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to sample a valid puzzle")
