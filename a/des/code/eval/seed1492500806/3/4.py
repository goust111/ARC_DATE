import random
from collections import Counter, defaultdict, deque
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400
UNFILLED = -1
BLOCKED = -2


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def detect_grey(grid: np.ndarray, bg: int) -> int:
    best_color = None
    best_score = (-1, -1)
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        comps = components4(grid == color)
        if not comps:
            continue
        straight = True
        max_len = 0
        total = 0
        for comp in comps:
            rows = {r for r, _ in comp}
            cols = {c for _, c in comp}
            if len(rows) != 1 and len(cols) != 1:
                straight = False
                break
            max_len = max(max_len, len(comp))
            total += len(comp)
        if straight and max_len > 1:
            score = (total, max_len)
            if score > best_score:
                best_score = score
                best_color = color
    if best_color is not None:
        return int(best_color)

    counts = Counter(int(v) for v in grid.flat if int(v) != bg)
    if not counts:
        return bg
    return int(counts.most_common(1)[0][0])


def locate_interval(bounds: list[int], value: int) -> int:
    for i in range(len(bounds) - 1):
        if bounds[i] <= value < bounds[i + 1]:
            return i
    raise ValueError(f"value {value} outside bounds {bounds}")


def infer_orientation_map(grid: np.ndarray, grey: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    comps = components4(g == grey)
    if not comps:
        return np.zeros((h, w), dtype=np.int8)

    row_bounds = {0, h}
    col_bounds = {0, w}
    comp_info = []
    for comp in comps:
        rows = sorted({r for r, _ in comp})
        cols = sorted({c for _, c in comp})
        if len(rows) == 1:
            orient = 0
            c0, c1 = cols[0], cols[-1]
            col_bounds.add(c0)
            col_bounds.add(c1 + 1)
            bbox = (rows[0], rows[0], c0, c1)
        elif len(cols) == 1:
            orient = 1
            r0, r1 = rows[0], rows[-1]
            row_bounds.add(r0)
            row_bounds.add(r1 + 1)
            bbox = (r0, r1, cols[0], cols[0])
        else:
            raise AssertionError("grey components must be straight line segments")
        comp_info.append((orient, bbox, comp))

    row_bounds = sorted(row_bounds)
    col_bounds = sorted(col_bounds)

    orient_map = np.zeros((h, w), dtype=np.int8)
    for ri in range(len(row_bounds) - 1):
        r0, r1 = row_bounds[ri], row_bounds[ri + 1]
        for ci in range(len(col_bounds) - 1):
            c0, c1 = col_bounds[ci], col_bounds[ci + 1]
            sub = g[r0:r1, c0:c1] == grey
            pts = np.argwhere(sub)
            if pts.size == 0:
                center_r = (r0 + r1 - 1) / 2.0
                center_c = (c0 + c1 - 1) / 2.0
                best = None
                best_orient = 0
                for orient, bbox, _comp in comp_info:
                    rr0, rr1, cc0, cc1 = bbox
                    if orient == 0:
                        dist = abs(center_r - rr0) + min(abs(center_c - cc0), abs(center_c - cc1), 0 if cc0 <= center_c <= cc1 else 10**9)
                    else:
                        dist = abs(center_c - cc0) + min(abs(center_r - rr0), abs(center_r - rr1), 0 if rr0 <= center_r <= rr1 else 10**9)
                    key = (dist, orient)
                    if best is None or key < best:
                        best = key
                        best_orient = orient
                orient_map[r0:r1, c0:c1] = best_orient
                continue

            rows = {int(p[0]) for p in pts}
            cols = {int(p[1]) for p in pts}
            if len(rows) == 1:
                orient_map[r0:r1, c0:c1] = 0
            elif len(cols) == 1:
                orient_map[r0:r1, c0:c1] = 1
            else:
                raise AssertionError("each fiber rectangle must contain one straight grey segment")

    return orient_map


def spread_ink(grid: np.ndarray, orient_map: np.ndarray, bg: int, grey: int) -> tuple[np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    owners = np.full((h, w), UNFILLED, dtype=int)
    frontier: list[tuple[int, int]] = []

    for r, c in np.argwhere((g != bg) & (g != grey)):
        rr, cc = int(r), int(c)
        owners[rr, cc] = int(g[rr, cc])
        frontier.append((rr, cc))

    conflict_steps = 0
    contested_cells = 0
    while frontier:
        attempts: dict[tuple[int, int], set[int]] = defaultdict(set)
        for r, c in frontier:
            color = int(owners[r, c])
            if color < 0:
                continue
            if orient_map[r, c] == 0:
                neigh = ((r, c - 1), (r, c + 1))
            else:
                neigh = ((r - 1, c), (r + 1, c))
            for nr, nc in neigh:
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                current = int(owners[nr, nc])
                if current == UNFILLED:
                    attempts[(nr, nc)].add(color)
                elif current >= 0 and current != color:
                    conflict_steps += 1

        if not attempts:
            break

        new_frontier: list[tuple[int, int]] = []
        for pos, colors in attempts.items():
            r, c = pos
            if owners[r, c] != UNFILLED:
                continue
            if len(colors) == 1:
                color = next(iter(colors))
                owners[r, c] = color
                new_frontier.append(pos)
            else:
                owners[r, c] = BLOCKED
                contested_cells += 1
        frontier = new_frontier

    out = np.full_like(g, bg)
    filled = owners >= 0
    out[filled] = owners[filled]

    turn_pixels = 0
    if frontier is not None:
        seed_positions = np.argwhere((g != bg) & (g != grey))
        for r, c in seed_positions:
            rr, cc = int(r), int(c)
            color = int(g[rr, cc])
            source_orient = int(orient_map[rr, cc])
            mask = owners == color
            if np.any(mask & (orient_map != source_orient)):
                turn_pixels += 1

    stats = {
        "filled": int(np.count_nonzero(filled)),
        "contested": int(contested_cells),
        "blocked_hits": int(conflict_steps),
        "turned_sources": int(turn_pixels),
        "grey_covered": int(np.count_nonzero((g == grey) & filled)),
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate colored ink spreading along hidden fiber bands reconstructed from grey guide segments.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = detect_grey(g, bg)
    orient_map = infer_orientation_map(g, grey)
    out, _stats = spread_ink(g, orient_map, bg, grey)
    return out


def random_partition(total: int, parts: int, lo: int, hi: int, rng: random.Random) -> list[int] | None:
    sizes = [lo] * parts
    remaining = total - lo * parts
    if remaining < 0:
        return None
    room = [hi - lo] * parts
    while remaining > 0:
        choices = [i for i in range(parts) if room[i] > 0]
        if not choices:
            return None
        i = rng.choice(choices)
        sizes[i] += 1
        room[i] -= 1
        remaining -= 1
    rng.shuffle(sizes)
    return sizes


def make_orientation_grid(rows: int, cols: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(200):
        orient = np.array([[rng.randint(0, 1) for _ in range(cols)] for _ in range(rows)], dtype=np.int8)
        if np.all(np.any(orient == 0, axis=1)) and np.all(np.any(orient == 1, axis=1)):
            if np.all(np.any(orient == 0, axis=0)) and np.all(np.any(orient == 1, axis=0)):
                return orient
    return None


def build_bounds(sizes: list[int]) -> list[int]:
    bounds = [0]
    for size in sizes:
        bounds.append(bounds[-1] + int(size))
    return bounds


def choose_segment_rows(heights: list[int], orient: np.ndarray, rng: random.Random) -> list[list[int | None]]:
    rows = [[None for _ in range(orient.shape[1])] for _ in range(orient.shape[0])]
    rbounds = build_bounds(heights)
    for i in range(orient.shape[0]):
        choices = [rbounds[i] + 1, rbounds[i + 1] - 2]
        if rng.random() < 0.5:
            choices.reverse()
        toggle = 0
        for j in range(orient.shape[1]):
            if orient[i, j] == 0:
                rows[i][j] = choices[toggle % 2]
                toggle += 1
    return rows


def choose_segment_cols(widths: list[int], orient: np.ndarray, rng: random.Random) -> list[list[int | None]]:
    cols = [[None for _ in range(orient.shape[1])] for _ in range(orient.shape[0])]
    cbounds = build_bounds(widths)
    for j in range(orient.shape[1]):
        choices = [cbounds[j] + 1, cbounds[j + 1] - 2]
        if rng.random() < 0.5:
            choices.reverse()
        toggle = 0
        for i in range(orient.shape[0]):
            if orient[i, j] == 1:
                cols[i][j] = choices[toggle % 2]
                toggle += 1
    return cols


def cell_id_from_bounds(r: int, c: int, rbounds: list[int], cbounds: list[int]) -> tuple[int, int]:
    return locate_interval(rbounds, r), locate_interval(cbounds, c)


def generate_attempt(rng: random.Random) -> np.ndarray | None:
    row_bands = rng.randint(2, 4)
    col_bands = rng.randint(2, 4)

    heights = random_partition(rng.randint(4 * row_bands, 6 * row_bands), row_bands, 4, 6, rng)
    widths = random_partition(rng.randint(4 * col_bands, 6 * col_bands), col_bands, 4, 6, rng)
    if heights is None or widths is None:
        return None

    h = sum(heights)
    w = sum(widths)
    if not (8 <= h <= 24 and 8 <= w <= 24):
        return None

    orient = make_orientation_grid(row_bands, col_bands, rng)
    if orient is None:
        return None

    rbounds = build_bounds(heights)
    cbounds = build_bounds(widths)
    h_rows = choose_segment_rows(heights, orient, rng)
    v_cols = choose_segment_cols(widths, orient, rng)

    grid = np.full((h, w), BG, dtype=int)
    for i in range(row_bands):
        for j in range(col_bands):
            r0, r1 = rbounds[i], rbounds[i + 1]
            c0, c1 = cbounds[j], cbounds[j + 1]
            if orient[i, j] == 0:
                rr = int(h_rows[i][j])
                grid[rr, c0:c1] = GREY
            else:
                cc = int(v_cols[i][j])
                grid[r0:r1, cc] = GREY

    cells = [(i, j) for i in range(row_bands) for j in range(col_bands)]
    rng.shuffle(cells)
    max_seeds = min(len(cells), 4)
    min_seeds = 2 if len(cells) >= 2 else 1
    seed_count = rng.randint(min_seeds, max_seeds)
    seed_colors = rng.sample(PALETTE, seed_count)

    placed = []
    for (i, j), color in zip(cells, seed_colors):
        r0, r1 = rbounds[i], rbounds[i + 1]
        c0, c1 = cbounds[j], cbounds[j + 1]
        options = [(r, c) for r in range(r0, r1) for c in range(c0, c1) if grid[r, c] == BG]
        if not options:
            return None
        if rng.random() < 0.5:
            opposite = 1 - int(orient[i, j])
            turning = [(r, c) for (r, c) in options if (
                (opposite == 0 and any(cell_id_from_bounds(r, nc, rbounds, cbounds) != (i, j) for nc in (0, w - 1))) or
                (opposite == 1 and any(cell_id_from_bounds(nr, c, rbounds, cbounds) != (i, j) for nr in (0, h - 1)))
            )]
            if turning:
                options = turning
        r, c = rng.choice(options)
        grid[r, c] = color
        placed.append((r, c, color))

    if len(placed) < min_seeds:
        return None

    solved = main(grid)
    if np.array_equal(solved, grid):
        return None

    grey_mask = grid == GREY
    orient_map = infer_orientation_map(grid, GREY)
    out, stats = spread_ink(grid, orient_map, BG, GREY)
    if not np.array_equal(solved, out):
        return None
    if stats["filled"] <= len(placed) + 6:
        return None
    if stats["turned_sources"] == 0:
        return None

    want_conflict = rng.random() < 0.5
    has_conflict = stats["contested"] > 0 or stats["blocked_hits"] > 0
    if want_conflict and not has_conflict:
        return None
    if not want_conflict and has_conflict and rng.random() < 0.7:
        return None

    if np.count_nonzero(grey_mask) < 8:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
