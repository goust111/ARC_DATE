import random

import numpy as np


BLACK = 0
GREY = 5
PORTAL_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_RETRIES = 400
PLACEMENT_ATTEMPTS = 250
PAIR_COUNT_RANGE = (2, 4)
PATH_LEN_RANGE = (1, 6)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _chebyshev_halo(cells: list[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    halo: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if _in_bounds(shape, rr, cc):
                    halo.add((rr, cc))
    return halo


def _trace_path(grid: np.ndarray, marker: tuple[int, int]) -> dict[str, object]:
    H, W = grid.shape
    r0, c0 = marker
    grey_neighbors = [
        (nr, nc)
        for dr, dc in DIRS4
        if _in_bounds((H, W), nr := r0 + dr, nc := c0 + dc) and int(grid[nr, nc]) == GREY
    ]
    assert len(grey_neighbors) == 1, "each portal marker must touch exactly one grey path"

    nr, nc = grey_neighbors[0]
    dr, dc = nr - r0, nc - c0
    prev = (r0, c0)
    cur = (nr, nc)
    cells: list[tuple[int, int]] = []

    while _in_bounds((H, W), cur[0], cur[1]) and int(grid[cur[0], cur[1]]) == GREY:
        cr, cc = cur
        forward = (cr + dr, cc + dc)
        for sdr, sdc in DIRS4:
            rr, cc2 = cr + sdr, cc + sdc
            if (rr, cc2) == prev or (rr, cc2) == forward:
                continue
            if _in_bounds((H, W), rr, cc2) and int(grid[rr, cc2]) == GREY:
                raise AssertionError("grey path must be a straight, unbranched segment")

        cells.append(cur)
        prev = cur
        cur = forward

    end_r, end_c = prev
    if _in_bounds((H, W), cur[0], cur[1]):
        assert int(grid[cur[0], cur[1]]) == BLACK, "path free end must face background"

    return {
        "marker": marker,
        "dir": (dr, dc),
        "length": len(cells),
        "cells": cells,
        "end": (end_r, end_c),
    }


def _extract_paths(grid: np.ndarray) -> dict[int, list[dict[str, object]]]:
    portal_colors = sorted(int(color) for color in np.unique(grid) if int(color) not in (BLACK, GREY))
    assert portal_colors, "expected at least one portal pair"

    markers_by_color: dict[int, list[tuple[int, int]]] = {color: [] for color in portal_colors}
    for r, c in np.argwhere((grid != BLACK) & (grid != GREY)):
        markers_by_color[int(grid[r, c])].append((int(r), int(c)))

    paths_by_color: dict[int, list[dict[str, object]]] = {}
    seen_grey: set[tuple[int, int]] = set()
    for color, markers in markers_by_color.items():
        assert len(markers) == 2, "each portal color must appear exactly twice"
        infos = []
        for marker in markers:
            info = _trace_path(grid, marker)
            path_cells = set(info["cells"])
            if seen_grey & path_cells:
                raise AssertionError("grey paths must be distinct segments")
            seen_grey |= path_cells
            infos.append(info)
        paths_by_color[color] = infos

    all_grey = {tuple(map(int, rc)) for rc in np.argwhere(grid == GREY)}
    assert seen_grey == all_grey, "every grey pixel must belong to exactly one portal path"
    return paths_by_color


def main(grid: np.ndarray) -> np.ndarray:
    """Teleport each grey path's length through its paired portal and keep growing from the opposite free end.

    Input structure:
    - The background is black.
    - Grey pixels form several straight, unbranched horizontal or vertical path segments.
    - Each path touches exactly one colored singleton marker at one end, and its other end faces background.
    - Every portal color appears exactly twice, linking two different path starts.

    Rule:
    - For each colored marker, follow its adjacent grey segment away from the marker and measure that path length.
    - Jump to the other marker of the same color, inspect that marker's own grey segment, and take its marker-to-free-end direction as the exit direction.
    - Starting just beyond that exit segment's free end, add grey pixels in the same direction for the measured length.
    - New grey pixels only replace black background; growth stops early at the first non-black cell or the grid boundary.
    - All original colored markers and original grey paths remain in place.

    Key insight:
    - The paired colors are not two ends of one line: each path exports its length, and that length reappears as extra growth beyond the other same-colored path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    paths_by_color = _extract_paths(g)
    out = g.copy()

    growths: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for color in sorted(paths_by_color):
        info_a, info_b = paths_by_color[color]
        growths.append((int(info_a["length"]), tuple(info_b["end"]), tuple(info_b["dir"])))
        growths.append((int(info_b["length"]), tuple(info_a["end"]), tuple(info_a["dir"])))

    for length, end, direction in growths:
        er, ec = end
        dr, dc = direction
        for step in range(1, length + 1):
            rr = er + dr * step
            cc = ec + dc * step
            if not _in_bounds(g.shape, rr, cc):
                break
            if int(g[rr, cc]) != BLACK:
                break
            out[rr, cc] = GREY

    return out


def _choose_grid_shape(pair_count: int) -> tuple[int, int]:
    base = 12 + 3 * pair_count
    H = random.randint(base, min(30, base + 8))
    W = random.randint(base, min(30, base + 8))
    return H, W


def _build_path_cells(marker: tuple[int, int], direction: tuple[int, int], length: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    mr, mc = marker
    dr, dc = direction
    path_cells = [(mr + dr * step, mc + dc * step) for step in range(1, length + 1)]
    return path_cells, path_cells[-1]


def _after_space(shape: tuple[int, int], end: tuple[int, int], direction: tuple[int, int]) -> int:
    H, W = shape
    er, ec = end
    dr, dc = direction
    rr, cc = er + dr, ec + dc
    count = 0
    while 0 <= rr < H and 0 <= cc < W:
        count += 1
        rr += dr
        cc += dc
    return count


def _reserve_corridor(end: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    er, ec = end
    dr, dc = direction
    return [(er + dr * step, ec + dc * step) for step in range(1, length + 1)]


def _pick_modes(specs: list[dict[str, int | str]]) -> None:
    boundary_candidates = []
    for idx, spec in enumerate(specs):
        incoming = int(spec["incoming"])
        if incoming > 1 and random.random() < 0.5:
            spec["mode"] = "boundary"
            boundary_candidates.append(idx)
        else:
            spec["mode"] = "open"

    if not boundary_candidates:
        candidates = [idx for idx, spec in enumerate(specs) if int(spec["incoming"]) > 1]
        if candidates:
            specs[random.choice(candidates)]["mode"] = "boundary"

    open_indices = [idx for idx, spec in enumerate(specs) if spec["mode"] == "open"]
    if not open_indices:
        specs[random.randrange(len(specs))]["mode"] = "open"


def _place_spec(
    grid: np.ndarray,
    hard_blocked: set[tuple[int, int]],
    color: int,
    own_length: int,
    incoming_length: int,
    mode: str,
) -> dict[str, object] | None:
    H, W = grid.shape
    dirs = DIRS4[:]
    random.shuffle(dirs)

    for _ in range(PLACEMENT_ATTEMPTS):
        dr, dc = random.choice(dirs)
        marker = (random.randrange(H), random.randrange(W))
        path_cells, end = _build_path_cells(marker, (dr, dc), own_length)

        if not all(_in_bounds((H, W), r, c) for r, c in [marker] + path_cells):
            continue

        free_space = _after_space((H, W), end, (dr, dc))
        if free_space < 1:
            continue
        if mode == "open":
            if free_space < incoming_length:
                continue
            corridor_len = incoming_length
        else:
            if not (1 <= free_space < incoming_length):
                continue
            corridor_len = free_space

        object_cells = [marker] + path_cells
        halo = _chebyshev_halo(object_cells, (H, W))
        corridor = _reserve_corridor(end, (dr, dc), corridor_len)

        if any(cell in hard_blocked for cell in halo):
            continue
        if any(cell in hard_blocked for cell in corridor):
            continue

        grid[marker] = color
        for r, c in path_cells:
            grid[r, c] = GREY
        hard_blocked |= halo
        hard_blocked |= set(corridor)

        return {
            "color": color,
            "marker": marker,
            "dir": (dr, dc),
            "length": own_length,
            "incoming": incoming_length,
            "mode": mode,
        }

    return None


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    H, W = _choose_grid_shape(pair_count)
    grid = np.full((H, W), BLACK, dtype=int)
    hard_blocked: set[tuple[int, int]] = set()

    colors = random.sample(PORTAL_PALETTE, pair_count)
    specs: list[dict[str, int | str]] = []
    for color in colors:
        len_a = random.randint(*PATH_LEN_RANGE)
        len_b = random.randint(*PATH_LEN_RANGE)
        specs.append({"color": color, "length": len_a, "incoming": len_b, "mode": "open"})
        specs.append({"color": color, "length": len_b, "incoming": len_a, "mode": "open"})

    if not any(int(spec["incoming"]) > 1 for spec in specs):
        return None
    _pick_modes(specs)

    random.shuffle(specs)
    specs.sort(key=lambda spec: -(int(spec["length"]) + int(spec["incoming"])))

    placed: list[dict[str, object]] = []
    for spec in specs:
        info = _place_spec(
            grid,
            hard_blocked,
            int(spec["color"]),
            int(spec["length"]),
            int(spec["incoming"]),
            str(spec["mode"]),
        )
        if info is None:
            return None
        placed.append(info)

    if len({tuple(info["dir"]) for info in placed}) < 2:
        return None
    if not any(info["mode"] == "boundary" for info in placed):
        return None

    black_count = int(np.count_nonzero(grid == BLACK))
    if black_count <= int(np.count_nonzero(grid != BLACK)):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid portal-path grid after {GEN_RETRIES} attempts")
