from __future__ import annotations

import random
from typing import Dict, Iterable, List, Tuple

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

DIRS: Dict[str, Tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
VERTICAL = ("up", "down")
SIGNAL_COLORS = (BLUE, RED, GREEN, MAGENTA, CYAN)
GEN_RETRIES = 500


def _marker_length(grid: np.ndarray, r: int, c: int, side: str) -> int:
    dr, dc = DIRS[side]
    h, w = grid.shape
    length = 0
    rr = r + dr
    cc = c + dc
    while 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == MAROON and length < 3:
        length += 1
        rr += dr
        cc += dc
    if length == 0:
        return 0
    if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == MAROON:
        return 0
    return length


def _detect_towers(grid: np.ndarray) -> List[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    towers: List[dict] = []
    reserved = {BLACK, YELLOW, GREY, ORANGE, MAROON}

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            color = int(g[r, c])
            if color in reserved:
                continue

            up_len = _marker_length(g, r, c, "up")
            down_len = _marker_length(g, r, c, "down")
            if (up_len > 0) == (down_len > 0):
                continue

            marker_side = "up" if up_len else "down"
            priority = up_len or down_len
            arms: List[str] = []
            valid = True

            for name, (dr, dc) in DIRS.items():
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    valid = False
                    break
                val = int(g[rr, cc])
                if name == marker_side:
                    if val != MAROON:
                        valid = False
                        break
                else:
                    if val == GREY:
                        arms.append(name)
                    elif val != BLACK:
                        valid = False
                        break

            if not valid or not arms:
                continue

            towers.append(
                {
                    "center": (r, c),
                    "color": color,
                    "priority": priority,
                    "marker_side": marker_side,
                    "arms": tuple(sorted(arms)),
                }
            )

    towers.sort(key=lambda t: (int(t["priority"]), int(t["center"][0]), int(t["center"][1])))
    return towers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Signal towers broadcast along their remaining grey arms in priority order.
    Yellow receivers take the first signal color that reaches them, then turn
    orange if a later tower reaches the same receiver. Colored receivers block
    later rays, but untouched yellow receivers remain transparent to the ray
    that activates them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    towers = _detect_towers(g)
    out = g.copy()

    receiver_mask = g == YELLOW
    static_blockers = (g != BLACK) & (~receiver_mask)

    for tower in towers:
        tr, tc = map(int, tower["center"])
        signal = int(tower["color"])
        for arm in tower["arms"]:
            dr, dc = DIRS[arm]
            r = tr + 2 * dr
            c = tc + 2 * dc
            while 0 <= r < h and 0 <= c < w:
                if bool(static_blockers[r, c]):
                    break
                if bool(receiver_mask[r, c]):
                    if int(out[r, c]) == YELLOW:
                        out[r, c] = signal
                    else:
                        out[r, c] = ORANGE
                        break
                elif int(out[r, c]) != BLACK:
                    break
                r += dr
                c += dc

    return out


def _tower_cells(tower: dict) -> Dict[Tuple[int, int], int]:
    r, c = map(int, tower["center"])
    cells: Dict[Tuple[int, int], int] = {(r, c): int(tower["color"])}
    for arm in tower["arms"]:
        dr, dc = DIRS[arm]
        cells[(r + dr, c + dc)] = GREY
    mr, mc = DIRS[str(tower["marker_side"])]
    for step in range(1, int(tower["priority"]) + 1):
        cells[(r + mr * step, c + mc * step)] = MAROON
    return cells


def _cells_in_bounds(cells: Iterable[Tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _dilate(coords: Iterable[Tuple[int, int]], h: int, w: int) -> set[Tuple[int, int]]:
    out: set[Tuple[int, int]] = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _can_place_tower(tower: dict, h: int, w: int, forbidden: set[Tuple[int, int]]) -> bool:
    cells = tuple(_tower_cells(tower).keys())
    if not _cells_in_bounds(cells, h, w):
        return False
    return not any(cell in forbidden for cell in cells)


def _paint_tower(grid: np.ndarray, tower: dict) -> None:
    for (r, c), color in _tower_cells(tower).items():
        grid[r, c] = color


def _ray_cells(tower: dict, arm: str, static_mask: np.ndarray) -> List[Tuple[int, int]]:
    h, w = static_mask.shape
    tr, tc = map(int, tower["center"])
    dr, dc = DIRS[arm]
    r = tr + 2 * dr
    c = tc + 2 * dc
    cells: List[Tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w and not bool(static_mask[r, c]):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _choose_subset(required: str, pool: List[str]) -> Tuple[str, ...]:
    extra_count = random.randint(0, len(pool))
    extras = random.sample(pool, extra_count)
    return tuple(sorted({required, *extras}))


def _random_free_receiver(
    candidates: List[Tuple[int, int]],
    taken: set[Tuple[int, int]],
    banned: set[Tuple[int, int]],
) -> Tuple[int, int] | None:
    random.shuffle(candidates)
    for cell in candidates:
        if cell in taken or cell in banned:
            continue
        return cell
    return None


def _receiver_ban(cells: Iterable[Tuple[int, int]], h: int, w: int) -> set[Tuple[int, int]]:
    return _dilate(cells, h, w)


def _line_after(
    ray: List[Tuple[int, int]],
    pivot: Tuple[int, int],
    min_gap: int,
    coverage: Dict[Tuple[int, int], set[int]],
    owner: int,
) -> List[Tuple[int, int]]:
    if pivot not in ray:
        return []
    idx = ray.index(pivot)
    start = idx + 1 + max(0, min_gap)
    out: List[Tuple[int, int]] = []
    for cell in ray[start:]:
        if coverage.get(cell, set()) == {owner}:
            out.append(cell)
    return out


def _random_tower(
    priority: int,
    h: int,
    w: int,
    color: int,
    forbidden: set[Tuple[int, int]],
    blocked_rows: set[int],
    blocked_cols: set[int],
) -> dict | None:
    for _ in range(250):
        marker_side = random.choice(VERTICAL)
        if marker_side == "up":
            r = random.randint(priority, h - 2)
        else:
            r = random.randint(1, h - priority - 1)
        c = random.randint(1, w - 2)
        arm_choices = [name for name in DIRS if name != marker_side]
        required = random.choice(arm_choices)
        arms = _choose_subset(required, [name for name in arm_choices if name != required])
        tower = {
            "center": (r, c),
            "color": color,
            "priority": priority,
            "marker_side": marker_side,
            "arms": arms,
        }
        cells = tuple(_tower_cells(tower).keys())
        if any(rr in blocked_rows or cc in blocked_cols for rr, cc in cells):
            continue
        if _can_place_tower(tower, h, w, forbidden):
            return tower
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 20)
    w = random.randint(14, 22)
    tower_count = random.choice((2, 3))
    colors = random.sample(SIGNAL_COLORS, tower_count)

    if h < 10 or w < 10:
        return None

    cross_r = random.randint(4, h - 5)
    cross_c = random.randint(4, w - 5)
    hdir = random.choice(("left", "right"))
    vdir = random.choice(("up", "down"))

    if hdir == "right":
        a_cols = list(range(1, cross_c - 1))
    else:
        a_cols = list(range(cross_c + 2, w - 1))
    if vdir == "down":
        b_rows = list(range(2, cross_r - 1))
    else:
        b_rows = list(range(cross_r + 2, h - 2))
    if not a_cols or not b_rows:
        return None

    tower_a = {
        "center": (cross_r, random.choice(a_cols)),
        "color": colors[0],
        "priority": 1,
        "marker_side": random.choice(VERTICAL),
    }
    extra_a = [name for name in DIRS if name not in {tower_a["marker_side"], hdir}]
    tower_a["arms"] = _choose_subset(hdir, extra_a)

    tower_b = {
        "center": (random.choice(b_rows), cross_c),
        "color": colors[1],
        "priority": 2,
        "marker_side": "down" if vdir == "up" else "up",
    }
    extra_b = [name for name in DIRS if name not in {tower_b["marker_side"], vdir}]
    tower_b["arms"] = _choose_subset(vdir, extra_b)

    towers = [tower_a, tower_b]
    forbidden: set[Tuple[int, int]] = set()
    for tower in towers:
        if not _can_place_tower(tower, h, w, forbidden):
            return None
        forbidden |= _dilate(_tower_cells(tower).keys(), h, w)

    if tower_count == 3:
        tower_c = _random_tower(
            priority=3,
            h=h,
            w=w,
            color=colors[2],
            forbidden=forbidden,
            blocked_rows={cross_r},
            blocked_cols={cross_c},
        )
        if tower_c is None:
            return None
        towers.append(tower_c)
        forbidden |= _dilate(_tower_cells(tower_c).keys(), h, w)

    inp = np.full((h, w), BLACK, dtype=int)
    for tower in towers:
        _paint_tower(inp, tower)

    static_mask = inp != BLACK
    rays: Dict[Tuple[int, str], List[Tuple[int, int]]] = {}
    coverage: Dict[Tuple[int, int], set[int]] = {}
    for idx, tower in enumerate(towers):
        for arm in tower["arms"]:
            cells = _ray_cells(tower, arm, static_mask)
            rays[(idx, arm)] = cells
            for cell in cells:
                coverage.setdefault(cell, set()).add(idx)

    cross = (cross_r, cross_c)
    main_a = rays.get((0, hdir), [])
    main_b = rays.get((1, vdir), [])
    if cross not in main_a or cross not in main_b:
        return None
    if coverage.get(cross, set()) != {0, 1}:
        return None

    single_candidates = _line_after(main_a, cross, min_gap=1, coverage=coverage, owner=0)
    blocked_candidates = _line_after(main_b, cross, min_gap=1, coverage=coverage, owner=1)
    uncovered = [cell for cell in np.argwhere(inp == BLACK)]
    uncovered_cells = [
        (int(r), int(c))
        for r, c in uncovered
        if cell_not_covered((int(r), int(c)), coverage)
    ]
    if not single_candidates or not blocked_candidates or not uncovered_cells:
        return None

    receivers: set[Tuple[int, int]] = set()
    receiver_banned = _receiver_ban(receivers, h, w)

    receivers.add(cross)
    receiver_banned = _receiver_ban(receivers, h, w)

    single_receiver = _random_free_receiver(single_candidates[:], receivers, receiver_banned - {cross})
    if single_receiver is None:
        return None
    receivers.add(single_receiver)
    receiver_banned = _receiver_ban(receivers, h, w)

    blocked_receiver = _random_free_receiver(blocked_candidates[:], receivers, receiver_banned - {cross})
    if blocked_receiver is None:
        return None
    receivers.add(blocked_receiver)
    receiver_banned = _receiver_ban(receivers, h, w)

    idle_receiver = _random_free_receiver(uncovered_cells[:], receivers, receiver_banned - {cross})
    if idle_receiver is None:
        return None
    receivers.add(idle_receiver)
    receiver_banned = _receiver_ban(receivers, h, w)

    unique_ray_cells = [
        cell
        for cell, owners in coverage.items()
        if len(owners) == 1 and cell not in receivers
    ]
    multi_ray_cells = [
        cell
        for cell, owners in coverage.items()
        if len(owners) >= 2 and cell not in receivers and owners != {0, 1}
    ]
    extra_uncovered = [cell for cell in uncovered_cells if cell not in receivers]

    for pool, count_range in (
        (unique_ray_cells, (0, 3)),
        (multi_ray_cells, (0, 1)),
        (extra_uncovered, (0, 3)),
    ):
        count = random.randint(*count_range)
        random.shuffle(pool)
        for cell in pool:
            if count == 0:
                break
            if cell in receiver_banned:
                continue
            receivers.add(cell)
            receiver_banned = _receiver_ban(receivers, h, w)
            count -= 1

    for r, c in receivers:
        if inp[r, c] != BLACK:
            return None
        inp[r, c] = YELLOW

    try:
        out = main(inp)
    except Exception:
        return None

    if int(out[cross]) != ORANGE:
        return None
    if int(out[single_receiver]) != int(tower_a["color"]):
        return None
    if int(out[blocked_receiver]) != YELLOW:
        return None
    if int(out[idle_receiver]) != YELLOW:
        return None

    receiver_values = [int(out[r, c]) for r, c in receivers]
    if not any(val == ORANGE for val in receiver_values):
        return None
    if not any(val == YELLOW for val in receiver_values):
        return None
    if not any(val not in {YELLOW, ORANGE} for val in receiver_values):
        return None

    if len(_detect_towers(inp)) != tower_count:
        return None
    return inp


def cell_not_covered(cell: Tuple[int, int], coverage: Dict[Tuple[int, int], set[int]]) -> bool:
    return cell not in coverage


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower input")
