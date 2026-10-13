from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
MARKER = 3
SIDES = ("top", "bottom", "left", "right")
PALETTE = (1, 2, 4, 5, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
INF = 10**9
GEN_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _infer_side_colors(grid: np.ndarray) -> dict[str, int]:
    h, w = grid.shape
    assert h >= 5 and w >= 5, "grid must be at least 5x5"

    strips = {
        "top": set(map(int, grid[0, 1:-1])),
        "bottom": set(map(int, grid[-1, 1:-1])),
        "left": set(map(int, grid[1:-1, 0])),
        "right": set(map(int, grid[1:-1, -1])),
    }
    assert all(len(colors) == 1 for colors in strips.values()), "each side must show one full-span color"
    side_colors = {side: next(iter(colors)) for side, colors in strips.items()}
    assert len(set(side_colors.values())) == 4, "the four border sides must have distinct colors"
    assert all(color not in (BG, MARKER) for color in side_colors.values()), "border colors cannot be bg or green"
    return side_colors


def _source_cells(shape: tuple[int, int]) -> dict[str, list[tuple[int, int]]]:
    h, w = shape
    return {
        "top": [(0, c) for c in range(w)],
        "bottom": [(h - 1, c) for c in range(w)],
        "left": [(r, 0) for r in range(h)],
        "right": [(r, w - 1) for r in range(h)],
    }


def _side_distance_map(grid: np.ndarray, side: str) -> np.ndarray:
    h, w = grid.shape
    dist = np.full((h, w), INF, dtype=int)
    q: deque[tuple[int, int]] = deque()

    for r, c in _source_cells((h, w))[side]:
        if dist[r, c] != 0:
            dist[r, c] = 0
            q.append((r, c))

    while q:
        r, c = q.popleft()
        nd = int(dist[r, c]) + 1
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if not _in_bounds((h, w), nr, nc):
                continue
            if grid[nr, nc] != BG:
                continue
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                q.append((nr, nc))

    return dist


def _all_side_distances(grid: np.ndarray) -> dict[str, np.ndarray]:
    return {side: _side_distance_map(grid, side) for side in SIDES}


def _target_distance(grid: np.ndarray, dist: np.ndarray, r: int, c: int) -> int:
    if grid[r, c] == BG:
        return int(dist[r, c])

    best = INF
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if _in_bounds(grid.shape, nr, nc):
            best = min(best, int(dist[nr, nc]) + 1)
    return int(best)


def _visible_sides(grid: np.ndarray, r: int, c: int) -> set[str]:
    h, w = grid.shape
    visible: set[str] = set()

    if all(int(grid[rr, c]) == BG for rr in range(1, r)):
        visible.add("top")
    if all(int(grid[rr, c]) == BG for rr in range(r + 1, h - 1)):
        visible.add("bottom")
    if all(int(grid[r, cc]) == BG for cc in range(1, c)):
        visible.add("left")
    if all(int(grid[r, cc]) == BG for cc in range(c + 1, w - 1)):
        visible.add("right")
    return visible


def _manhattan_side(shape: tuple[int, int], r: int, c: int) -> str | None:
    h, w = shape
    raw = {
        "top": r,
        "bottom": h - 1 - r,
        "left": c,
        "right": w - 1 - c,
    }
    best = min(raw.values())
    nearest = [side for side, value in raw.items() if value == best]
    return nearest[0] if len(nearest) == 1 else None


def _adjacent_internal_wall_colors(grid: np.ndarray, r: int, c: int) -> set[int]:
    h, w = grid.shape
    colors: set[int] = set()
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if 1 <= nr < h - 1 and 1 <= nc < w - 1:
            color = int(grid[nr, nc])
            if color not in (BG, MARKER):
                colors.add(color)
    return colors


def _bg_degree(grid: np.ndarray, r: int, c: int) -> int:
    degree = 0
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == BG:
            degree += 1
    return degree


def _marker_analysis(grid: np.ndarray) -> list[dict]:
    side_colors = _infer_side_colors(grid)
    dist_maps = _all_side_distances(grid)
    analyses: list[dict] = []
    for r_raw, c_raw in np.argwhere(grid == MARKER):
        r = int(r_raw)
        c = int(c_raw)
        side_dists = {side: _target_distance(grid, dist_maps[side], r, c) for side in SIDES}
        finite = {side: value for side, value in side_dists.items() if value < INF}
        winner_side = None
        winner_color = None
        if finite:
            best = min(finite.values())
            nearest = [side for side, value in finite.items() if value == best]
            assert len(nearest) == 1, "every reachable marker must have a unique nearest side"
            winner_side = nearest[0]
            winner_color = int(side_colors[winner_side])

        visible = _visible_sides(grid, r, c)
        adjacent_colors = _adjacent_internal_wall_colors(grid, r, c)
        analyses.append(
            {
                "coord": (r, c),
                "reachable": winner_side is not None,
                "winner_side": winner_side,
                "winner_color": winner_color,
                "side_dists": side_dists,
                "visible": visible,
                "visible_decoy": winner_side is not None and bool(visible) and winner_side not in visible,
                "manhattan_side": _manhattan_side(grid.shape, r, c),
                "manhattan_wrong": winner_side is not None and _manhattan_side(grid.shape, r, c) not in (None, winner_side),
                "adjacent_wall_colors": adjacent_colors,
                "adj_wrong": winner_color is not None and any(color != winner_color for color in adjacent_colors),
            }
        )
    return analyses


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each green marker by the truly nearest border source after flooding around wall fragments.

    Input structure:
    - Four distinct full-span border lines occupy the top, bottom, left, and right sides; the side color is identified by the run spanning that side, while corner cells simply belong to adjacent border colors.
    - Interior cells contain black background, green singleton marker pixels, and wall fragments whose colors may match some border lines.
    - Interior non-background non-marker cells are obstacles only; they are not extra sources even if they share a border color.

    Rule:
    - Treat every cell on each border side as a source for that side's color.
    - Compute 4-connected shortest-path distance through background cells only; wall fragments and markers block movement, but a marker cell may be used as the final endpoint.
    - Recolor each reachable green marker to the color of its unique nearest border side; if no border side can reach it, erase it to background. Leave all wall fragments and border lines unchanged.

    Key insight:
    - The source is defined by full-span border placement, not by color alone, so internal same-colored fragments stay walls; straight sightlines and raw Manhattan distance are both decoys, and only geodesic flood distance through the background determines the winner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    side_colors = _infer_side_colors(g)
    dist_maps = _all_side_distances(g)
    out = g.copy()

    for r_raw, c_raw in np.argwhere(g == MARKER):
        r = int(r_raw)
        c = int(c_raw)
        side_dists = {side: _target_distance(g, dist_maps[side], r, c) for side in SIDES}
        finite = {side: value for side, value in side_dists.items() if value < INF}
        if not finite:
            out[r, c] = BG
            continue

        best = min(finite.values())
        nearest = [side for side, value in finite.items() if value == best]
        assert len(nearest) == 1, "marker must have a unique nearest side"
        out[r, c] = int(side_colors[nearest[0]])

    return out


def _paint_borders(grid: np.ndarray, side_colors: dict[str, int]) -> None:
    grid[0, :] = int(side_colors["top"])
    grid[-1, :] = int(side_colors["bottom"])
    grid[:, 0] = int(side_colors["left"])
    grid[:, -1] = int(side_colors["right"])


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> bool:
    if not cells:
        return False
    for r, c in cells:
        if not (1 <= r < grid.shape[0] - 1 and 1 <= c < grid.shape[1] - 1):
            return False
        if int(grid[r, c]) != BG:
            return False
    for r, c in cells:
        grid[r, c] = int(color)
    return True


def _segment_with_gate(
    orientation: str, fixed: int, start: int, end: int, gate_start: int, gate_len: int
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for pos in range(start, end + 1):
        if gate_start <= pos < gate_start + gate_len:
            continue
        if orientation == "v":
            cells.append((pos, fixed))
        else:
            cells.append((fixed, pos))
    return cells


def _free_background_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    cells = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) == BG:
                cells.append((r, c))
    return cells


def _place_stub(grid: np.ndarray, side_colors: dict[str, int]) -> bool:
    h, w = grid.shape
    options: list[list[tuple[int, int]]] = []
    for _ in range(20):
        horizontal = random.choice([True, False])
        length = random.randint(3, 5)
        if horizontal:
            r = random.randint(2, h - 3)
            c0 = random.randint(2, w - length - 2)
            cells = [(r, c0 + i) for i in range(length)]
        else:
            c = random.randint(2, w - 3)
            r0 = random.randint(2, h - length - 2)
            cells = [(r0 + i, c) for i in range(length)]
        if all(int(grid[r, c]) == BG for r, c in cells):
            options.append(cells)
    if not options:
        return False
    return _paint_cells(grid, random.choice(options), random.choice(list(side_colors.values())))


def _place_unreachable_ring(grid: np.ndarray, side_colors: dict[str, int]) -> tuple[int, int] | None:
    h, w = grid.shape
    candidates = []
    for r in range(3, h - 3):
        for c in range(3, w - 3):
            ring = [
                (r - 1, c - 1),
                (r - 1, c),
                (r - 1, c + 1),
                (r, c - 1),
                (r, c + 1),
                (r + 1, c - 1),
                (r + 1, c),
                (r + 1, c + 1),
            ]
            halo = [(rr, cc) for rr in range(r - 2, r + 3) for cc in range(c - 2, c + 3)]
            if all(int(grid[rr, cc]) == BG for rr, cc in halo):
                candidates.append((r, c, ring))
    if not candidates:
        return None
    r, c, ring = random.choice(candidates)
    if not _paint_cells(grid, ring, random.choice(list(side_colors.values()))):
        return None
    return (r, c)


def _build_base_grid() -> tuple[np.ndarray, tuple[int, int]] | None:
    h = random.randint(15, 23)
    w = random.randint(15, 23)
    side_colors = dict(zip(SIDES, random.sample(PALETTE, 4)))

    grid = np.full((h, w), BG, dtype=int)
    _paint_borders(grid, side_colors)

    vc = random.randint(max(4, w // 3), min(w - 5, (2 * w) // 3))
    gate_len = random.choice([1, 1, 2])
    gate_row = random.randint(3, h - 4 - gate_len + 1)
    if not _paint_cells(
        grid,
        _segment_with_gate("v", vc, 1, h - 2, gate_row, gate_len),
        random.choice(list(side_colors.values())),
    ):
        return None

    upper_side = random.choice(["left", "right"])
    lower_side = "right" if upper_side == "left" else "left"

    upper_row = random.randint(2, max(2, gate_row - 1))
    lower_row = random.randint(min(h - 3, gate_row + gate_len + 1), h - 3)

    if upper_side == "left":
        upper_start, upper_end = 2, vc - 1
        lower_start, lower_end = vc + 1, w - 3
    else:
        upper_start, upper_end = vc + 1, w - 3
        lower_start, lower_end = 2, vc - 1
    if upper_end - upper_start < 3 or lower_end - lower_start < 3:
        return None

    upper_gate = random.randint(upper_start + 1, upper_end - 1)
    lower_gate = random.randint(lower_start + 1, lower_end - 1)
    if not _paint_cells(
        grid,
        _segment_with_gate("h", upper_row, upper_start, upper_end, upper_gate, 1),
        random.choice(list(side_colors.values())),
    ):
        return None
    if not _paint_cells(
        grid,
        _segment_with_gate("h", lower_row, lower_start, lower_end, lower_gate, 1),
        random.choice(list(side_colors.values())),
    ):
        return None

    if upper_side == "left":
        vc2 = random.randint(vc + 2, w - 3)
        part_start = upper_row + 1
        part_end = h - 3
    else:
        vc2 = random.randint(2, vc - 2)
        part_start = 2
        part_end = lower_row - 1
    if part_end - part_start >= 3:
        gate2 = random.randint(part_start + 1, part_end - 1)
        if not _paint_cells(
            grid,
            _segment_with_gate("v", vc2, part_start, part_end, gate2, 1),
            random.choice(list(side_colors.values())),
        ):
            return None

    if random.random() < 0.85 and not _place_stub(grid, side_colors):
        return None
    ring_center = _place_unreachable_ring(grid, side_colors)
    if ring_center is None:
        return None

    return grid, ring_center


def _candidate_infos(grid: np.ndarray, forbidden: set[tuple[int, int]]) -> list[dict]:
    side_colors = _infer_side_colors(grid)
    dist_maps = _all_side_distances(grid)
    infos: list[dict] = []
    h, w = grid.shape
    for r in range(2, h - 2):
        for c in range(2, w - 2):
            if (r, c) in forbidden or int(grid[r, c]) != BG:
                continue
            if _bg_degree(grid, r, c) < 2:
                continue

            side_dists = {side: int(dist_maps[side][r, c]) for side in SIDES}
            finite = {side: value for side, value in side_dists.items() if value < INF}
            if not finite:
                continue
            best = min(finite.values())
            nearest = [side for side, value in finite.items() if value == best]
            if len(nearest) != 1:
                continue

            winner_side = nearest[0]
            winner_color = int(side_colors[winner_side])
            visible = _visible_sides(grid, r, c)
            adjacent_colors = _adjacent_internal_wall_colors(grid, r, c)
            infos.append(
                {
                    "coord": (r, c),
                    "winner_side": winner_side,
                    "winner_color": winner_color,
                    "visible": visible,
                    "visible_decoy": bool(visible) and winner_side not in visible,
                    "manhattan_side": _manhattan_side(grid.shape, r, c),
                    "manhattan_wrong": _manhattan_side(grid.shape, r, c) not in (None, winner_side),
                    "adj_wrong": any(color != winner_color for color in adjacent_colors),
                }
            )
    return infos


def _far_enough(cell: tuple[int, int], chosen: list[tuple[int, int]], min_gap: int = 4) -> bool:
    return all(abs(cell[0] - r) + abs(cell[1] - c) >= min_gap for r, c in chosen)


def _pick_info(
    infos: list[dict],
    chosen: list[dict],
    *,
    require_visible_decoy: bool = False,
    require_manhattan_wrong: bool = False,
    require_adj_wrong: bool = False,
    different_winner_from: dict | None = None,
) -> dict | None:
    chosen_cells = [item["coord"] for item in chosen]
    pool = []
    for info in infos:
        if info in chosen:
            continue
        if not _far_enough(info["coord"], chosen_cells):
            continue
        if require_visible_decoy and not info["visible_decoy"]:
            continue
        if require_manhattan_wrong and not info["manhattan_wrong"]:
            continue
        if require_adj_wrong and not info["adj_wrong"]:
            continue
        if different_winner_from is not None and info["winner_color"] == different_winner_from["winner_color"]:
            continue
        pool.append(info)
    return random.choice(pool) if pool else None


def _assemble_grid() -> np.ndarray | None:
    built = _build_base_grid()
    if built is None:
        return None
    grid, unreachable_center = built

    infos = _candidate_infos(grid, {unreachable_center})
    if len(infos) < 2:
        return None

    chosen: list[dict] = []

    first = _pick_info(infos, chosen, require_visible_decoy=True, require_adj_wrong=True)
    if first is None:
        first = _pick_info(infos, chosen, require_visible_decoy=True)
    if first is None:
        return None
    chosen.append(first)

    second = None
    if not first["manhattan_wrong"]:
        second = _pick_info(
            infos,
            chosen,
            require_manhattan_wrong=True,
            different_winner_from=first,
        )
    if second is None:
        second = _pick_info(infos, chosen, different_winner_from=first)
    if second is None:
        return None
    chosen.append(second)

    if not any(item["adj_wrong"] for item in chosen):
        third = _pick_info(infos, chosen, require_adj_wrong=True)
        if third is None:
            return None
        chosen.append(third)
    elif random.random() < 0.6:
        extra = _pick_info(infos, chosen)
        if extra is not None:
            chosen.append(extra)

    final = grid.copy()
    for info in chosen:
        r, c = info["coord"]
        final[r, c] = MARKER
    final[unreachable_center] = MARKER
    return final


def _semantic_check(inp: np.ndarray) -> None:
    out = main(inp)
    assert out.shape == inp.shape
    assert np.array_equal(out[inp != MARKER], inp[inp != MARKER]), "non-marker cells must stay unchanged"

    side_colors = _infer_side_colors(inp)
    internal_side_wall_count = 0
    for r in range(1, inp.shape[0] - 1):
        for c in range(1, inp.shape[1] - 1):
            if int(inp[r, c]) in set(side_colors.values()):
                internal_side_wall_count += 1
    assert internal_side_wall_count > 0, "must contain internal wall fragments in border colors"

    markers = _marker_analysis(inp)
    assert len(markers) >= 3, "need several markers"

    reachable = [info for info in markers if info["reachable"]]
    unreachable = [info for info in markers if not info["reachable"]]
    assert reachable, "must include reachable markers"
    assert unreachable, "must include an unreachable marker"
    assert len({info["winner_color"] for info in reachable}) >= 2, "reachable markers should not all inherit the same side"
    assert any(info["visible_decoy"] for info in reachable), "must include a sightline decoy"
    assert any(info["manhattan_wrong"] for info in reachable), "must include a Manhattan-distance decoy"
    assert any(info["adj_wrong"] for info in reachable), "must include a same-color wall fragment decoy"

    for info in reachable:
        r, c = info["coord"]
        assert int(out[r, c]) == int(info["winner_color"])
    for info in unreachable:
        r, c = info["coord"]
        assert int(out[r, c]) == BG


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _assemble_grid()
        if grid is None:
            continue
        try:
            _semantic_check(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
