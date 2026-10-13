import random
from collections import Counter, deque

import numpy as np


GEAR_MASK = np.ones((5, 5), dtype=bool)
GEAR_MASK[0, 0] = False
GEAR_MASK[0, 4] = False
GEAR_MASK[4, 0] = False
GEAR_MASK[4, 4] = False

SIDE_ORDER = ("up", "right", "down", "left")
END_OFFSETS = {
    "up": (-2, 0),
    "right": (0, 2),
    "down": (2, 0),
    "left": (0, -2),
}
STEP_TO_SIDE = {
    (-6, 0): "up",
    (0, 6): "right",
    (6, 0): "down",
    (0, -6): "left",
}
DIR4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]

BG_COLOR = 0
GEAR_COUNT_RANGE = (2, 6)
MARGIN_RANGE = (1, 6)
GEN_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rotate_side(side: str, turns: int) -> str:
    return SIDE_ORDER[(SIDE_ORDER.index(side) + turns) % 4]


def _opposite(side: str) -> str:
    return _rotate_side(side, 2)


def _side_from_delta(dr: int, dc: int) -> str:
    side = STEP_TO_SIDE.get((dr, dc))
    assert side is not None, "gears must mesh on the 6-spaced lattice"
    return side


def _marker_side(center: tuple[int, int], marker_cell: tuple[int, int]) -> str:
    dr = marker_cell[0] - center[0]
    dc = marker_cell[1] - center[1]
    for side, (sdr, sdc) in END_OFFSETS.items():
        if (dr, dc) == (sdr, sdc):
            return side
    raise AssertionError("marker must sit on one of the four terminal teeth")


def _analyze_gears(grid: np.ndarray) -> tuple[list[dict], int, int]:
    comps = _components(grid != BG_COLOR)
    assert comps, "expected at least one gear"

    gears: list[dict] = []
    driver_idx: int | None = None
    marker_color: int | None = None

    for coords in comps:
        assert len(coords) == int(GEAR_MASK.sum()), "each gear must occupy 21 cells"
        r0, c0, r1, c1 = _bbox(coords)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (5, 5), "each gear bbox must be 5x5"

        local = np.zeros((5, 5), dtype=bool)
        for r, c in coords:
            local[r - r0, c - c0] = True
        assert np.array_equal(local, GEAR_MASK), "component mask must be the chamfered 5x5 gear"

        colors = Counter(int(grid[r, c]) for r, c in coords)
        center = (r0 + 2, c0 + 2)
        gear = {
            "coords": coords,
            "top_left": (r0, c0),
            "center": center,
            "gear_color": None,
            "mark_side": None,
        }

        if len(colors) == 1:
            gear["gear_color"] = next(iter(colors))
        else:
            assert len(colors) == 2, "only the driver may have a second color"
            counts = sorted(colors.values())
            assert counts == [1, 20], "driver must have exactly one recolored tooth"
            assert driver_idx is None, "expected a unique driver gear"

            gear_color = next(color for color, cnt in colors.items() if cnt == 20)
            mark_color = next(color for color, cnt in colors.items() if cnt == 1)
            mark_cell = next((r, c) for r, c in coords if int(grid[r, c]) == mark_color)

            gear["gear_color"] = gear_color
            gear["mark_side"] = _marker_side(center, mark_cell)
            driver_idx = len(gears)
            marker_color = mark_color

        gears.append(gear)

    assert driver_idx is not None, "expected exactly one marked driver gear"
    assert marker_color is not None
    return gears, driver_idx, marker_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate a connected gear train by propagating alternating quarter-turns from the marked driver.

    Input structure:
    - Background is black.
    - Each gear is the fixed 5x5 shape obtained by removing the four corners.
    - Exactly one gear has a differently colored marker on one terminal tooth.

    Rule:
    - The marked driver rotates 90 degrees clockwise.
    - Every meshed neighbor rotates 90 degrees the opposite way, and this alternates through the gear graph.
    - Unmarked driven gears receive the driver's marker color on the tooth opposite the force-input side,
      then that tooth follows the gear's own quarter-turn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    gears, driver_idx, marker_color = _analyze_gears(g)

    adj: dict[int, list[int]] = {idx: [] for idx in range(len(gears))}
    for i in range(len(gears)):
        r1, c1 = gears[i]["center"]
        for j in range(i + 1, len(gears)):
            r2, c2 = gears[j]["center"]
            if (abs(r1 - r2) == 6 and c1 == c2) or (abs(c1 - c2) == 6 and r1 == r2):
                adj[i].append(j)
                adj[j].append(i)

    rot_dir: dict[int, int] = {driver_idx: 1}
    parent: dict[int, int | None] = {driver_idx: None}
    depth: dict[int, int] = {driver_idx: 0}
    q = deque([driver_idx])
    while q:
        cur = q.popleft()
        for nxt in adj[cur]:
            if nxt not in rot_dir:
                rot_dir[nxt] = -rot_dir[cur]
                parent[nxt] = cur
                depth[nxt] = depth[cur] + 1
                q.append(nxt)
            else:
                assert rot_dir[nxt] == -rot_dir[cur], "meshing directions must alternate consistently"

    for idx in rot_dir:
        if idx == driver_idx:
            continue
        pred_count = sum(1 for nbr in adj[idx] if depth.get(nbr) == depth[idx] - 1)
        assert pred_count == 1, "each driven gear must have a unique upstream driver"

    final_sides: dict[int, str] = {
        driver_idx: _rotate_side(gears[driver_idx]["mark_side"], 1)
    }
    for idx in rot_dir:
        if idx == driver_idx:
            continue
        p = parent[idx]
        assert p is not None
        cr, cc = gears[idx]["center"]
        pr, pc = gears[p]["center"]
        input_side = _side_from_delta(pr - cr, pc - cc)
        source_side = _opposite(input_side)
        final_sides[idx] = _rotate_side(source_side, rot_dir[idx])

    out = g.copy()
    for idx, side in final_sides.items():
        gear = gears[idx]
        for r, c in gear["coords"]:
            out[r, c] = int(gear["gear_color"])
        dr, dc = END_OFFSETS[side]
        rr, cc = gear["center"][0] + dr, gear["center"][1] + dc
        out[rr, cc] = int(marker_color)

    return out


def _neighbor_count(cell: tuple[int, int], cells: set[tuple[int, int]]) -> int:
    r, c = cell
    return sum((r + dr, c + dc) in cells for dr, dc in DIR4)


def _grow_chain(count: int) -> list[tuple[int, int]] | None:
    cells = [(0, 0)]
    cell_set = {(0, 0)}
    while len(cells) < count:
        endpoints = [
            cell
            for cell in cells
            if _neighbor_count(cell, cell_set) <= 1
        ]
        random.shuffle(endpoints)
        placed = False
        for endpoint in endpoints:
            er, ec = endpoint
            options = []
            for dr, dc in DIR4:
                nxt = (er + dr, ec + dc)
                if nxt in cell_set:
                    continue
                if _neighbor_count(nxt, cell_set) == 1:
                    options.append(nxt)
            if options:
                choice = random.choice(options)
                cells.append(choice)
                cell_set.add(choice)
                placed = True
                break
        if not placed:
            return None
    return cells


def _grow_branched(count: int) -> list[tuple[int, int]] | None:
    assert count >= 3
    cells = [(0, 0)]
    cell_set = {(0, 0)}
    first_two = random.sample(DIR4, 2)
    for dr, dc in first_two:
        nxt = (dr, dc)
        cells.append(nxt)
        cell_set.add(nxt)

    while len(cells) < count:
        options = []
        for cell in cells:
            r, c = cell
            for dr, dc in DIR4:
                nxt = (r + dr, c + dc)
                if nxt in cell_set:
                    continue
                if _neighbor_count(nxt, cell_set) == 1:
                    options.append(nxt)
        if not options:
            return None
        choice = random.choice(options)
        cells.append(choice)
        cell_set.add(choice)
    return cells


def _draw_gear(grid: np.ndarray, top: int, left: int, color: int) -> None:
    patch = grid[top : top + 5, left : left + 5]
    patch[GEAR_MASK] = int(color)


def _gen_attempt() -> np.ndarray | None:
    gear_count = random.randint(*GEAR_COUNT_RANGE)
    branched = gear_count >= 3 and random.random() < 0.5
    if branched:
        cells = _grow_branched(gear_count)
    else:
        cells = _grow_chain(gear_count)
    if cells is None:
        return None

    min_r = min(r for r, _ in cells)
    max_r = max(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    max_c = max(c for _, c in cells)

    top_margin = random.randint(*MARGIN_RANGE)
    bottom_margin = random.randint(*MARGIN_RANGE)
    left_margin = random.randint(*MARGIN_RANGE)
    right_margin = random.randint(*MARGIN_RANGE)

    height = top_margin + bottom_margin + 5 + 6 * (max_r - min_r)
    width = left_margin + right_margin + 5 + 6 * (max_c - min_c)
    grid = np.full((height, width), BG_COLOR, dtype=int)

    palette = random.sample(range(1, 10), gear_count + 1)
    marker_color = palette[0]
    gear_colors = palette[1:]
    driver_idx = random.randrange(gear_count)
    driver_side = random.choice(SIDE_ORDER)

    tops_lefts: list[tuple[int, int]] = []
    for idx, (cell_r, cell_c) in enumerate(cells):
        top = top_margin + 6 * (cell_r - min_r)
        left = left_margin + 6 * (cell_c - min_c)
        _draw_gear(grid, top, left, gear_colors[idx])
        tops_lefts.append((top, left))

    driver_top, driver_left = tops_lefts[driver_idx]
    ddr, ddc = END_OFFSETS[driver_side]
    grid[driver_top + 2 + ddr, driver_left + 2 + ddc] = marker_color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(grid, out):
        return None
    if int((grid == marker_color).sum()) != 1:
        return None
    if int((out == marker_color).sum()) != gear_count:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_ATTEMPTS} attempts")
