from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


MARKER = 9
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR4 = tuple(DIRS.values())

GRID_H_RANGE = (18, 29)
GRID_W_RANGE = (18, 29)
TOWER_COUNT_RANGE = (2, 4)
EXTRA_RECEIVER_MAX = 2
MIN_RAY_LEN = 5
GEN_RETRIES = 400
PLACEMENT_TRIES = 300
PALETTE = list(range(10))


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


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
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _tower_cells(r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + dr, c0 + dc) for dr in range(3) for dc in range(3)]


def _marker_cell(r0: int, c0: int, direction: str) -> tuple[int, int]:
    if direction == "up":
        return (r0 - 1, c0 + 1)
    if direction == "down":
        return (r0 + 3, c0 + 1)
    if direction == "left":
        return (r0 + 1, c0 - 1)
    assert direction == "right"
    return (r0 + 1, c0 + 3)


def _band_footprint(r0: int, c0: int, direction: str, h: int, w: int) -> list[tuple[int, int]]:
    if direction == "up":
        return [(r, c) for r in range(0, r0) for c in range(c0, c0 + 3)]
    if direction == "down":
        return [(r, c) for r in range(r0 + 3, h) for c in range(c0, c0 + 3)]
    if direction == "left":
        return [(r, c) for r in range(r0, r0 + 3) for c in range(0, c0)]
    assert direction == "right"
    return [(r, c) for r in range(r0, r0 + 3) for c in range(c0 + 3, w)]


def _receiver_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _band_centers(tower: dict, h: int, w: int) -> list[tuple[int, int]]:
    r0 = int(tower["r0"])
    c0 = int(tower["c0"])
    direction = str(tower["direction"])
    if direction == "up":
        return [(r, c0 + 1) for r in range(2, r0 - 2)]
    if direction == "down":
        return [(r, c0 + 1) for r in range(r0 + 5, h - 2)]
    if direction == "left":
        return [(r0 + 1, c) for c in range(2, c0 - 2)]
    assert direction == "right"
    return [(r0 + 1, c) for c in range(c0 + 5, w - 2)]


def _paint_cross(grid: np.ndarray, center: tuple[int, int], center_color: int, arm_color: int) -> None:
    r, c = center
    grid[r, c] = int(center_color)
    grid[r - 1, c] = int(arm_color)
    grid[r + 1, c] = int(arm_color)
    grid[r, c - 1] = int(arm_color)
    grid[r, c + 1] = int(arm_color)


def _mark_guard(guard: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = guard.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    guard[nr, nc] = True


def _detect_towers(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    towers: list[dict] = []
    for comp in _components((grid != bg) & (grid != MARKER)):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if len(comp) != 9 or r1 - r0 != 2 or c1 - c0 != 2:
            continue
        color = int(grid[r0, c0])
        if any(int(grid[r, c]) != color for r, c in comp):
            continue
        dirs = []
        for direction in DIRS:
            mr, mc = _marker_cell(r0, c0, direction)
            if 0 <= mr < h and 0 <= mc < w and int(grid[mr, mc]) == MARKER:
                dirs.append(direction)
        assert len(dirs) == 1, "each tower must have exactly one centered maroon marker"
        towers.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "color": int(color),
                "direction": dirs[0],
            }
        )
    assert towers, "expected at least one signal tower"
    return towers


def _detect_receivers(grid: np.ndarray, bg: int, tower_cells: set[tuple[int, int]]) -> list[dict]:
    h, w = grid.shape
    receivers: list[dict] = []
    used_centers: set[tuple[int, int]] = set()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if (r, c) in tower_cells or int(grid[r, c]) in (bg, MARKER):
                continue
            center_color = int(grid[r, c])
            arm_coords = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            arm_colors = [int(grid[rr, cc]) for rr, cc in arm_coords]
            if any((rr, cc) in tower_cells for rr, cc in arm_coords):
                continue
            if len(set(arm_colors)) != 1:
                continue
            arm_color = arm_colors[0]
            if arm_color in (bg, MARKER, center_color):
                continue
            used_centers.add((r, c))
            receivers.append(
                {
                    "center": (int(r), int(c)),
                    "center_color": int(center_color),
                    "arm_color": int(arm_color),
                }
            )
    assert used_centers == {receiver["center"] for receiver in receivers}
    return receivers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Towers emit 3-wide bands toward the grid edge, then receivers either keep or erase
    a local segment depending on whether the band color matches the receiver center.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    assert bg != MARKER, "background must differ from maroon markers"

    towers = _detect_towers(g, bg)
    tower_cells = {cell for tower in towers for cell in _tower_cells(int(tower["r0"]), int(tower["c0"]))}
    receivers = _detect_receivers(g, bg, tower_cells)

    out = g.copy()
    bands: list[dict] = []
    for tower in towers:
        footprint = _band_footprint(int(tower["r0"]), int(tower["c0"]), str(tower["direction"]), h, w)
        painted = {(r, c) for r, c in footprint if int(g[r, c]) == bg}
        for r, c in painted:
            out[r, c] = int(tower["color"])
        bands.append(
            {
                "color": int(tower["color"]),
                "direction": str(tower["direction"]),
                "footprint": set(footprint),
                "painted": painted,
            }
        )

    for receiver in receivers:
        center = tuple(receiver["center"])
        for band in bands:
            if center not in band["footprint"]:
                continue
            if int(band["color"]) == int(receiver["center_color"]):
                continue
            dr, dc = DIRS[str(band["direction"])]
            pr, pc = dc, dr
            for step in range(-2, 3):
                ar = center[0] + dr * step
                ac = center[1] + dc * step
                for offset in (-1, 0, 1):
                    rr = ar + pr * offset
                    cc = ac + pc * offset
                    if (rr, cc) in band["painted"]:
                        out[rr, cc] = bg

    out[g == MARKER] = bg
    return out


def _tower_candidate_valid(
    tower_cells: list[tuple[int, int]],
    marker: tuple[int, int],
    footprint: list[tuple[int, int]],
    occupied: np.ndarray,
    guard: np.ndarray,
    footprints: np.ndarray,
) -> bool:
    actual = tower_cells + [marker]
    return (
        all(not occupied[r, c] for r, c in actual)
        and all(not guard[r, c] for r, c in actual)
        and all(not occupied[r, c] for r, c in footprint)
        and all(not footprints[r, c] for r, c in footprint)
    )


def _place_towers(
    rng: random.Random, grid: np.ndarray, occupied: np.ndarray, guard: np.ndarray, tower_count: int
) -> tuple[list[dict], np.ndarray] | None:
    h, w = grid.shape
    footprints = np.zeros((h, w), dtype=bool)
    colors = rng.sample([c for c in PALETTE if c not in (int(grid[0, 0]), MARKER)], tower_count)
    towers: list[dict] = []

    for color in colors:
        placed = False
        for _ in range(PLACEMENT_TRIES):
            direction = rng.choice(list(DIRS))
            if direction == "up":
                if h < 8:
                    return None
                r0 = rng.randint(MIN_RAY_LEN, h - 4)
                c0 = rng.randint(1, w - 5)
            elif direction == "down":
                if h < 8:
                    return None
                r0 = rng.randint(1, h - MIN_RAY_LEN - 4)
                c0 = rng.randint(1, w - 5)
            elif direction == "left":
                if w < 8:
                    return None
                r0 = rng.randint(1, h - 5)
                c0 = rng.randint(MIN_RAY_LEN, w - 4)
            else:
                if w < 8:
                    return None
                r0 = rng.randint(1, h - 5)
                c0 = rng.randint(1, w - MIN_RAY_LEN - 4)

            tower_cells = _tower_cells(r0, c0)
            marker = _marker_cell(r0, c0, direction)
            footprint = _band_footprint(r0, c0, direction, h, w)
            if not footprint:
                continue
            if not _tower_candidate_valid(tower_cells, marker, footprint, occupied, guard, footprints):
                continue

            for r, c in tower_cells:
                grid[r, c] = int(color)
                occupied[r, c] = True
            grid[marker] = MARKER
            occupied[marker] = True
            _mark_guard(guard, tower_cells + [marker])
            for r, c in footprint:
                footprints[r, c] = True
            towers.append(
                {
                    "r0": int(r0),
                    "c0": int(c0),
                    "color": int(color),
                    "direction": direction,
                    "marker": marker,
                    "footprint": set(footprint),
                }
            )
            placed = True
            break
        if not placed:
            return None

    return towers, footprints


def _choose_receiver_colors(
    rng: random.Random, bg: int, center_color: int | None = None, forbidden: set[int] | None = None
) -> tuple[int, int]:
    forbidden = set() if forbidden is None else set(forbidden)
    base = [c for c in PALETTE if c not in forbidden and c not in (bg, MARKER)]
    if center_color is None:
        center_color = rng.choice(base)
    arms = [c for c in base if c != center_color]
    return int(center_color), int(rng.choice(arms))


def _place_band_receivers(
    rng: random.Random,
    grid: np.ndarray,
    occupied: np.ndarray,
    guard: np.ndarray,
    towers: list[dict],
) -> list[dict] | None:
    h, w = grid.shape
    bg = int(grid[0, 0])
    states = ["match", "mismatch"]
    if len(towers) == 2:
        receiver_modes = states.copy()
    elif len(towers) == 3:
        receiver_modes = ["match", "mismatch", rng.choice(states)]
    else:
        receiver_modes = ["match", "match", "mismatch", "mismatch"]
    rng.shuffle(receiver_modes)

    receivers: list[dict] = []
    for tower, mode in zip(towers, receiver_modes, strict=True):
        candidates = _band_centers(tower, h, w)
        rng.shuffle(candidates)
        placed = False
        for center in candidates:
            cells = _receiver_cells(center)
            if any(occupied[r, c] or guard[r, c] for r, c in cells):
                continue
            if mode == "match":
                center_color, arm_color = _choose_receiver_colors(
                    rng, bg, center_color=int(tower["color"]), forbidden={int(tower["color"])}
                )
            else:
                mismatch_colors = [c for c in PALETTE if c not in (bg, MARKER, int(tower["color"]))]
                center_color = rng.choice(mismatch_colors)
                _, arm_color = _choose_receiver_colors(
                    rng, bg, center_color=int(center_color), forbidden={int(tower["color"])}
                )

            _paint_cross(grid, center, center_color, arm_color)
            for r, c in cells:
                occupied[r, c] = True
            _mark_guard(guard, cells)
            receivers.append(
                {
                    "center": center,
                    "mode": mode,
                    "tower_color": int(tower["color"]),
                    "center_color": int(center_color),
                }
            )
            placed = True
            break
        if not placed:
            return None

    return receivers


def _place_extra_receivers(
    rng: random.Random,
    grid: np.ndarray,
    occupied: np.ndarray,
    guard: np.ndarray,
    footprints: np.ndarray,
    count: int,
) -> bool:
    h, w = grid.shape
    bg = int(grid[0, 0])
    for _ in range(count):
        placed = False
        for _ in range(PLACEMENT_TRIES):
            center = (rng.randint(1, h - 2), rng.randint(1, w - 2))
            cells = _receiver_cells(center)
            if any(occupied[r, c] or guard[r, c] for r, c in cells):
                continue
            if any(footprints[r, c] for r, c in cells):
                continue
            center_color, arm_color = _choose_receiver_colors(rng, bg)
            _paint_cross(grid, center, center_color, arm_color)
            for r, c in cells:
                occupied[r, c] = True
            _mark_guard(guard, cells)
            placed = True
            break
        if not placed:
            return False
    return True


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice([c for c in PALETTE if c != MARKER])
    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    guard = np.zeros((h, w), dtype=bool)

    tower_count = rng.randint(TOWER_COUNT_RANGE[0], TOWER_COUNT_RANGE[1])
    placed = _place_towers(rng, grid, occupied, guard, tower_count)
    if placed is None:
        return None
    towers, footprints = placed

    band_receivers = _place_band_receivers(rng, grid, occupied, guard, towers)
    if band_receivers is None:
        return None

    extra_count = rng.randint(0, EXTRA_RECEIVER_MAX)
    if extra_count > 0 and not _place_extra_receivers(rng, grid, occupied, guard, footprints, extra_count):
        return None

    if not any(receiver["mode"] == "match" for receiver in band_receivers):
        return None
    if not any(receiver["mode"] == "mismatch" for receiver in band_receivers):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _generate_attempt(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(out, grid):
            continue
        if np.any(grid == MARKER) and np.any(out == MARKER):
            continue
        return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
