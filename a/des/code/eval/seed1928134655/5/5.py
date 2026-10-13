import numpy as np


BG = 0
WALL = 5
TOWER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
EDGES = ("top", "bottom", "left", "right")
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (12, 18)
GRID_W_RANGE = (12, 18)
TOWER_COUNT_CHOICES = (2, 3, 4)
EXTRA_WALL_COUNT_RANGE = (0, 2)
WALL_LENGTH_RANGE = (2, 5)
GENERATE_RETRIES = 400


def _edge_coords(shape: tuple[int, int], edge: str) -> list[tuple[int, int]]:
    h, w = shape
    if edge == "top":
        return [(0, c) for c in range(w)]
    if edge == "bottom":
        return [(h - 1, c) for c in range(w)]
    if edge == "left":
        return [(r, 0) for r in range(h)]
    if edge == "right":
        return [(r, w - 1) for r in range(h)]
    raise ValueError(f"unknown edge {edge}")


def _is_on_edge(cell: tuple[int, int], edge: str, shape: tuple[int, int]) -> bool:
    r, c = cell
    h, w = shape
    if edge == "top":
        return r == 0
    if edge == "bottom":
        return r == h - 1
    if edge == "left":
        return c == 0
    if edge == "right":
        return c == w - 1
    raise ValueError(f"unknown edge {edge}")


def _legend_candidates(grid: np.ndarray) -> list[tuple[str, dict[int, int], dict[int, tuple[int, int]], np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []

    for edge in EDGES:
        coords = _edge_coords((h, w), edge)
        vals = [int(g[r, c]) for r, c in coords]
        if WALL in vals:
            continue

        color_idx = [i for i, v in enumerate(vals) if v not in (BG, WALL)]
        if not color_idx or color_idx[0] != 0:
            continue

        legend_colors = [vals[i] for i in color_idx]
        if len(set(legend_colors)) != len(legend_colors):
            continue

        if any(v != BG for i, v in enumerate(vals) if i not in color_idx):
            continue

        radii: dict[int, int] = {}
        towers: dict[int, tuple[int, int]] = {}
        valid = True
        for pos_i, idx in enumerate(color_idx):
            next_idx = color_idx[pos_i + 1] if pos_i + 1 < len(color_idx) else len(vals)
            radius = next_idx - idx - 1
            color = vals[idx]
            if radius < 1:
                valid = False
                break
            radii[int(color)] = int(radius)

            pts = np.argwhere(g == color)
            if pts.shape[0] != 2:
                valid = False
                break

            off_edge = []
            edge_hits = 0
            for rr, cc in pts.astype(int, copy=False):
                cell = (int(rr), int(cc))
                if _is_on_edge(cell, edge, (h, w)):
                    edge_hits += 1
                else:
                    off_edge.append(cell)
            if edge_hits != 1 or len(off_edge) != 1:
                valid = False
                break
            towers[int(color)] = off_edge[0]

        if not valid:
            continue

        protected = np.zeros_like(g, dtype=bool)
        for r, c in coords:
            protected[r, c] = True
        candidates.append((edge, radii, towers, protected))

    return candidates


def _parse_legend(grid: np.ndarray) -> tuple[str, dict[int, int], dict[int, tuple[int, int]], np.ndarray]:
    candidates = _legend_candidates(grid)
    assert len(candidates) == 1, "expected exactly one valid legend edge"
    return candidates[0]


def _simulate(
    grid: np.ndarray,
    radii: dict[int, int],
    towers: dict[int, tuple[int, int]],
    protected: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    coverers: dict[tuple[int, int], list[tuple[int, int]]] = {}
    blocked_rays = 0

    for color, (tr, tc) in towers.items():
        radius = int(radii[color])
        for dr, dc in DIRS:
            hit_wall = False
            for step in range(1, radius + 1):
                nr = tr + dr * step
                nc = tc + dc * step
                if not (0 <= nr < h and 0 <= nc < w):
                    break
                if int(g[nr, nc]) == WALL:
                    hit_wall = True
                    break
                if int(g[nr, nc]) == BG and not bool(protected[nr, nc]):
                    coverers.setdefault((nr, nc), []).append((step, int(color)))
            if hit_wall:
                blocked_rays += 1

    out = g.copy()
    tie_cells = 0
    overlap_win_cells = 0
    tower_paint_counts = {int(color): 0 for color in towers}

    for (r, c), hits in coverers.items():
        best_dist = min(dist for dist, _color in hits)
        best_colors = {color for dist, color in hits if dist == best_dist}
        if len(best_colors) == 1:
            winner = next(iter(best_colors))
            out[r, c] = winner
            tower_paint_counts[winner] += 1
            if len({color for _dist, color in hits}) >= 2:
                overlap_win_cells += 1
        else:
            tie_cells += 1

    stats = {
        "tie_cells": tie_cells,
        "overlap_win_cells": overlap_win_cells,
        "blocked_rays": blocked_rays,
        "painted_cells": int(np.count_nonzero(out != g)),
        "tower_paint_counts": tower_paint_counts,
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read edge legends, then color each black cell by the nearest visible tower signal.

    Legends occupy one whole grid edge. Each entry is encoded as one colored pixel followed
    along that edge by R black cells, where R is the tower's broadcast radius.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    edge, radii, towers, protected = _parse_legend(g)
    _ = edge

    non_wall_colors = set(int(v) for v in np.unique(g) if int(v) not in (BG, WALL))
    assert non_wall_colors == set(radii), "legend colors must match tower colors"

    out, _stats = _simulate(g, radii, towers, protected)
    return out


def _sample_composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    assert total >= parts >= 1
    if parts == 1:
        return [total]
    cuts = sorted(int(x) for x in rng.choice(np.arange(1, total), size=parts - 1, replace=False))
    prev = 0
    pieces = []
    for cut in cuts + [total]:
        pieces.append(cut - prev)
        prev = cut
    return pieces


def _interior_bounds(shape: tuple[int, int], edge: str) -> tuple[int, int, int, int]:
    h, w = shape
    row_lo, row_hi = 1, h - 2
    col_lo, col_hi = 1, w - 2
    if edge == "top":
        row_lo = 2
    elif edge == "bottom":
        row_hi = h - 3
    elif edge == "left":
        col_lo = 2
    elif edge == "right":
        col_hi = w - 3
    return row_lo, row_hi, col_lo, col_hi


def _build_legend_sequence(colors: list[int], radii: dict[int, int]) -> list[int]:
    seq: list[int] = []
    for color in colors:
        seq.append(int(color))
        seq.extend([BG] * int(radii[color]))
    return seq


def _wall_cells(orientation: str, anchor: tuple[int, int], length: int) -> list[tuple[int, int]]:
    r, c = anchor
    if orientation == "h":
        return [(r, c + i) for i in range(length)]
    if orientation == "v":
        return [(r + i, c) for i in range(length)]
    raise ValueError(f"unknown orientation {orientation}")


def _place_random_wall(
    shape: tuple[int, int],
    edge: str,
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    row_lo, row_hi, col_lo, col_hi = _interior_bounds(shape, edge)
    h, w = shape
    _ = (h, w)
    for _attempt in range(120):
        orientation = str(rng.choice(["h", "v"]))
        length = int(rng.integers(WALL_LENGTH_RANGE[0], WALL_LENGTH_RANGE[1] + 1))
        if orientation == "h":
            if col_hi - col_lo + 1 < length:
                continue
            r = int(rng.integers(row_lo, row_hi + 1))
            c = int(rng.integers(col_lo, col_hi - length + 2))
            cells = _wall_cells("h", (r, c), length)
        else:
            if row_hi - row_lo + 1 < length:
                continue
            r = int(rng.integers(row_lo, row_hi - length + 2))
            c = int(rng.integers(col_lo, col_hi + 1))
            cells = _wall_cells("v", (r, c), length)
        if any(cell in occupied or cell in forbidden for cell in cells):
            continue
        return cells
    return None


def _core_pair_setup(
    shape: tuple[int, int],
    edge: str,
    radii: dict[int, int],
    colors: list[int],
    rng: np.random.Generator,
) -> tuple[dict[int, tuple[int, int]], set[tuple[int, int]], str] | None:
    row_lo, row_hi, col_lo, col_hi = _interior_bounds(shape, edge)
    ranked = sorted(colors, key=lambda color: (radii[color], color), reverse=True)
    if len(ranked) < 2 or min(radii[ranked[0]], radii[ranked[1]]) < 3:
        return None

    placements: dict[int, tuple[int, int]] = {}
    clear_path: set[tuple[int, int]] = set()

    options = []
    max_h_gap = col_hi - col_lo
    max_v_gap = row_hi - row_lo
    tmax = min(radii[ranked[0]], radii[ranked[1]]) - 1
    if max_h_gap >= 4 and tmax >= 2:
        options.append("h")
    if max_v_gap >= 4 and tmax >= 2:
        options.append("v")
    if not options:
        return None

    orientation = str(rng.choice(options))
    if orientation == "h":
        t_hi = min(tmax, max_h_gap // 2, 4)
        if t_hi < 2:
            return None
        t = int(rng.integers(2, t_hi + 1))
        gap = 2 * t
        row_candidates = list(range(row_lo + 1, row_hi)) or list(range(row_lo, row_hi + 1))
        if not row_candidates:
            return None
        r = int(rng.choice(row_candidates))
        c1 = int(rng.integers(col_lo, col_hi - gap + 2))
        c2 = c1 + gap
        placements[ranked[0]] = (r, c1)
        placements[ranked[1]] = (r, c2)
        clear_path = {(r, c) for c in range(c1, c2 + 1)}
    else:
        t_hi = min(tmax, max_v_gap // 2, 4)
        if t_hi < 2:
            return None
        t = int(rng.integers(2, t_hi + 1))
        gap = 2 * t
        col_candidates = list(range(col_lo + 1, col_hi)) or list(range(col_lo, col_hi + 1))
        if not col_candidates:
            return None
        c = int(rng.choice(col_candidates))
        r1 = int(rng.integers(row_lo, row_hi - gap + 2))
        r2 = r1 + gap
        placements[ranked[0]] = (r1, c)
        placements[ranked[1]] = (r2, c)
        clear_path = {(r, c) for r in range(r1, r2 + 1)}

    return placements, clear_path, orientation


def _blocking_wall(
    shape: tuple[int, int],
    edge: str,
    towers: dict[int, tuple[int, int]],
    radii: dict[int, int],
    core_orientation: str,
    occupied: set[tuple[int, int]],
    clear_path: set[tuple[int, int]],
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    row_lo, row_hi, col_lo, col_hi = _interior_bounds(shape, edge)
    colors = list(towers)
    rng.shuffle(colors)

    for color in colors:
        tr, tc = towers[color]
        radius = radii[color]
        if core_orientation == "h":
            dirs = [(-1, 0), (1, 0)]
            segment_orientation = "h"
        else:
            dirs = [(0, -1), (0, 1)]
            segment_orientation = "v"
        rng.shuffle(dirs)

        for dr, dc in dirs:
            max_dist = radius - 1
            if max_dist < 1:
                continue
            distances = list(range(1, min(max_dist, 3) + 1))
            rng.shuffle(distances)
            for dist in distances:
                wr = tr + dr * dist
                wc = tc + dc * dist
                if not (row_lo <= wr <= row_hi and col_lo <= wc <= col_hi):
                    continue

                length_hi = min(WALL_LENGTH_RANGE[1], (col_hi - col_lo + 1) if segment_orientation == "h" else (row_hi - row_lo + 1))
                lengths = list(range(WALL_LENGTH_RANGE[0], length_hi + 1))
                rng.shuffle(lengths)
                for length in lengths:
                    if segment_orientation == "h":
                        start_lo = max(col_lo, wc - length + 1)
                        start_hi = min(wc, col_hi - length + 1)
                        if start_lo > start_hi:
                            continue
                        c0 = int(rng.integers(start_lo, start_hi + 1))
                        cells = _wall_cells("h", (wr, c0), length)
                    else:
                        start_lo = max(row_lo, wr - length + 1)
                        start_hi = min(wr, row_hi - length + 1)
                        if start_lo > start_hi:
                            continue
                        r0 = int(rng.integers(start_lo, start_hi + 1))
                        cells = _wall_cells("v", (r0, wc), length)

                    if any(cell in occupied or cell in clear_path for cell in cells):
                        continue
                    return cells
    return None


def _place_extra_towers(
    shape: tuple[int, int],
    edge: str,
    colors: list[int],
    towers: dict[int, tuple[int, int]],
    occupied: set[tuple[int, int]],
    clear_path: set[tuple[int, int]],
    rng: np.random.Generator,
) -> bool:
    row_lo, row_hi, col_lo, col_hi = _interior_bounds(shape, edge)
    existing = list(towers.values())
    for color in colors:
        if color in towers:
            continue
        placed = False
        for _attempt in range(200):
            if existing and rng.random() < 0.5:
                base_r, base_c = existing[int(rng.integers(0, len(existing)))]
                if rng.random() < 0.5:
                    rr = base_r
                    cc = int(rng.integers(col_lo, col_hi + 1))
                else:
                    rr = int(rng.integers(row_lo, row_hi + 1))
                    cc = base_c
            else:
                rr = int(rng.integers(row_lo, row_hi + 1))
                cc = int(rng.integers(col_lo, col_hi + 1))

            cell = (rr, cc)
            if cell in occupied or cell in clear_path:
                continue
            if min(abs(rr - r0) + abs(cc - c0) for r0, c0 in existing) < 3:
                continue
            towers[color] = cell
            occupied.add(cell)
            existing.append(cell)
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    edge = str(rng.choice(EDGES))
    legend_len = w if edge in ("top", "bottom") else h

    feasible_counts = [k for k in TOWER_COUNT_CHOICES if 2 * k <= legend_len]
    if not feasible_counts:
        return None
    tower_count = int(rng.choice(feasible_counts))

    radii_parts = _sample_composition(legend_len - tower_count, tower_count, rng)
    if len(sorted(radii_parts, reverse=True)) < 2 or sorted(radii_parts, reverse=True)[1] < 3:
        return None

    colors = [int(c) for c in rng.choice(np.array(TOWER_COLORS), size=tower_count, replace=False)]
    rng.shuffle(colors)
    radii = {int(color): int(radius) for color, radius in zip(colors, radii_parts)}

    core = _core_pair_setup((h, w), edge, radii, colors, rng)
    if core is None:
        return None
    towers, clear_path, core_orientation = core

    occupied: set[tuple[int, int]] = set(_edge_coords((h, w), edge))
    occupied.update(towers.values())

    block_wall = _blocking_wall((h, w), edge, towers, radii, core_orientation, occupied, clear_path, rng)
    if block_wall is None:
        return None
    walls = set(block_wall)
    occupied.update(walls)

    if not _place_extra_towers((h, w), edge, colors, towers, occupied, clear_path, rng):
        return None

    extra_wall_target = int(rng.integers(EXTRA_WALL_COUNT_RANGE[0], EXTRA_WALL_COUNT_RANGE[1] + 1))
    for _ in range(extra_wall_target):
        extra = _place_random_wall((h, w), edge, occupied, clear_path, rng)
        if extra is None:
            continue
        walls.update(extra)
        occupied.update(extra)

    legend_order = colors[:]
    rng.shuffle(legend_order)
    legend_seq = _build_legend_sequence(legend_order, radii)
    if len(legend_seq) != legend_len:
        return None

    grid = np.zeros((h, w), dtype=int)
    for (r, c), value in zip(_edge_coords((h, w), edge), legend_seq):
        grid[r, c] = int(value)

    for color, (r, c) in towers.items():
        grid[r, c] = int(color)
    for r, c in walls:
        grid[r, c] = WALL

    try:
        parsed_edge, parsed_radii, parsed_towers, protected = _parse_legend(grid)
    except AssertionError:
        return None
    if parsed_edge != edge or parsed_radii != radii or parsed_towers != towers:
        return None

    out_expected, stats = _simulate(grid, radii, towers, protected)
    try:
        out_actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(out_expected, out_actual):
        return None

    if stats["tie_cells"] < 1:
        return None
    if stats["overlap_win_cells"] < 1:
        return None
    if stats["blocked_rays"] < 1:
        return None
    if stats["painted_cells"] < 6 or stats["painted_cells"] > (h * w) // 2:
        return None

    paint_counts = stats["tower_paint_counts"]
    if sum(1 for count in paint_counts.values() if count > 0) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATE_RETRIES} attempts")
