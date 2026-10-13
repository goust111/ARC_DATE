import random
from collections import Counter, deque

import numpy as np


ARROW_SPECS = {
    "up": {
        "forward": (-1, 0),
        "tail": (1, 0),
        "base": (2, 0),
        "head": ((0, 0), (1, -1), (1, 1), (2, -1), (2, 0), (2, 1)),
    },
    "down": {
        "forward": (1, 0),
        "tail": (-1, 0),
        "base": (-2, 0),
        "head": ((0, 0), (-1, -1), (-1, 1), (-2, -1), (-2, 0), (-2, 1)),
    },
    "left": {
        "forward": (0, -1),
        "tail": (0, 1),
        "base": (0, 2),
        "head": ((0, 0), (-1, 1), (1, 1), (-1, 2), (0, 2), (1, 2)),
    },
    "right": {
        "forward": (0, 1),
        "tail": (0, -1),
        "base": (0, -2),
        "head": ((0, 0), (-1, -1), (1, -1), (-1, -2), (0, -2), (1, -2)),
    },
}

DIR_ORDER = ("up", "right", "down", "left")
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

GRID_H_MIN, GRID_H_MAX = 14, 24
GRID_W_MIN, GRID_W_MAX = 14, 24
ARROW_COUNT_MIN, ARROW_COUNT_MAX = 2, 4
TAIL_LEN_MIN, TAIL_LEN_MAX = 1, 4
GAP_MIN, GAP_MAX = 1, 5
PAIR_PLACEMENT_TRIES = 250
GEN_RETRIES = 400

_GEN_CALL_COUNT = 0


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _component_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS8:
                    nr, nc = rr + dr, cc + dc
                    if not _in_bounds(h, w, nr, nc) or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "coord_set": set(coords),
                    "bbox": _component_bbox(coords),
                }
            )
    return comps


def _is_target_component(comp: dict) -> bool:
    coords = comp["coords"]
    if len(coords) != 4:
        return False
    r0, c0, r1, c1 = comp["bbox"]
    if (r1 - r0, c1 - c0) != (1, 1):
        return False
    return comp["coord_set"] == {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}


def _match_arrow_component(comp: dict) -> dict:
    coords = comp["coord_set"]
    matches = []
    for tip in coords:
        tr, tc = tip
        for direction, spec in ARROW_SPECS.items():
            head = {(tr + dr, tc + dc) for dr, dc in spec["head"]}
            if not head.issubset(coords):
                continue
            base = (tr + spec["base"][0], tc + spec["base"][1])
            tail_dir = spec["tail"]
            tail = []
            cur = (base[0] + tail_dir[0], base[1] + tail_dir[1])
            while cur in coords:
                tail.append(cur)
                cur = (cur[0] + tail_dir[0], cur[1] + tail_dir[1])
            if not tail:
                continue
            expected = head | set(tail)
            if expected != coords:
                continue
            matches.append(
                {
                    "color": int(comp["color"]),
                    "direction": direction,
                    "tip": tip,
                    "base_center": base,
                    "tail_dir": tail_dir,
                    "tail_end": tail[-1],
                    "coords": coords,
                }
            )
    assert len(matches) == 1, "each non-target component must match exactly one arrow geometry"
    return matches[0]


def _find_target_for_arrow(
    grid: np.ndarray,
    arrow: dict,
    pixel_to_target: dict[tuple[int, int], int],
    targets: list[dict],
    bg: int,
) -> tuple[dict, int]:
    h, w = grid.shape
    dr, dc = ARROW_SPECS[arrow["direction"]]["forward"]
    r, c = arrow["tip"][0] + dr, arrow["tip"][1] + dc
    gap = 0
    while _in_bounds(h, w, r, c):
        value = int(grid[r, c])
        if value == bg:
            gap += 1
            r += dr
            c += dc
            continue
        target_idx = pixel_to_target.get((r, c))
        assert target_idx is not None, "the first object on an arrow ray must be a target square"
        return targets[target_idx], gap
    raise AssertionError("arrow ray must encounter a target")


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    comps = _components(g, bg)
    assert comps, "expected arrows and targets"

    targets = [comp for comp in comps if _is_target_component(comp)]
    arrows = [_match_arrow_component(comp) for comp in comps if not _is_target_component(comp)]
    assert arrows and targets, "expected at least one arrow and one target"

    pixel_to_target: dict[tuple[int, int], int] = {}
    for idx, target in enumerate(targets):
        for cell in target["coords"]:
            pixel_to_target[cell] = idx

    out = g.copy()
    for arrow in arrows:
        target, gap = _find_target_for_arrow(g, arrow, pixel_to_target, targets, bg)
        for r, c in target["coords"]:
            out[r, c] = arrow["color"]
        tail_r, tail_c = arrow["tail_end"]
        dr, dc = arrow["tail_dir"]
        for step in range(1, gap + 1):
            nr = tail_r + dr * step
            nc = tail_c + dc * step
            assert _in_bounds(g.shape[0], g.shape[1], nr, nc), "tail extension must stay in bounds"
            assert int(g[nr, nc]) == bg, "tail extension must only paint background"
            out[nr, nc] = arrow["color"]

    return out


def _arrow_cells(tip: tuple[int, int], direction: str, tail_len: int) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    spec = ARROW_SPECS[direction]
    tr, tc = tip
    head = {(tr + dr, tc + dc) for dr, dc in spec["head"]}
    base = (tr + spec["base"][0], tc + spec["base"][1])
    tail = {
        (base[0] + spec["tail"][0] * step, base[1] + spec["tail"][1] * step)
        for step in range(1, tail_len + 1)
    }
    return head | tail, (base[0] + spec["tail"][0] * tail_len, base[1] + spec["tail"][1] * tail_len)


def _target_cells(tip: tuple[int, int], direction: str, gap: int, side_choice: int) -> set[tuple[int, int]]:
    tr, tc = tip
    if direction == "right":
        r0, c0 = tr - side_choice, tc + gap + 1
    elif direction == "left":
        r0, c0 = tr - side_choice, tc - gap - 2
    elif direction == "down":
        r0, c0 = tr + gap + 1, tc - side_choice
    else:
        r0, c0 = tr - gap - 2, tc - side_choice
    return {(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)}


def _ray_gap_cells(tip: tuple[int, int], direction: str, gap: int) -> set[tuple[int, int]]:
    dr, dc = ARROW_SPECS[direction]["forward"]
    return {(tip[0] + dr * step, tip[1] + dc * step) for step in range(1, gap + 1)}


def _tail_extension_cells(tail_end: tuple[int, int], direction: str, gap: int) -> set[tuple[int, int]]:
    dr, dc = ARROW_SPECS[direction]["tail"]
    return {(tail_end[0] + dr * step, tail_end[1] + dc * step) for step in range(1, gap + 1)}


def _dilate(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if _in_bounds(h, w, nr, nc):
                    out.add((nr, nc))
    return out


def _all_in_bounds(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    return all(_in_bounds(h, w, r, c) for r, c in cells)


def _balanced_directions(count: int, rng: random.Random) -> list[str]:
    global _GEN_CALL_COUNT
    offset = _GEN_CALL_COUNT % len(DIR_ORDER)
    _GEN_CALL_COUNT += 1
    rotated = list(DIR_ORDER[offset:] + DIR_ORDER[:offset])
    rng.shuffle(rotated)
    return rotated[:count]


def _place_pair(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    arrow_color: int,
    target_color: int,
    direction: str,
    rng: random.Random,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    h, w = grid.shape
    for _ in range(PAIR_PLACEMENT_TRIES):
        tail_len = rng.randint(TAIL_LEN_MIN, TAIL_LEN_MAX)
        gap = rng.randint(GAP_MIN, GAP_MAX)
        side_choice = rng.randint(0, 1)
        tip = (rng.randrange(h), rng.randrange(w))

        arrow_cells, tail_end = _arrow_cells(tip, direction, tail_len)
        target_cells = _target_cells(tip, direction, gap, side_choice)
        gap_cells = _ray_gap_cells(tip, direction, gap)
        extension_cells = _tail_extension_cells(tail_end, direction, gap)

        actual_cells = arrow_cells | target_cells
        needed_bg = gap_cells | extension_cells
        if not _all_in_bounds(actual_cells | needed_bg, h, w):
            continue
        if actual_cells & needed_bg:
            continue
        if any(cell in blocked for cell in actual_cells | needed_bg):
            continue

        reserve = _dilate(actual_cells, h, w) | needed_bg
        if any(cell in blocked for cell in reserve):
            continue

        for r, c in arrow_cells:
            grid[r, c] = arrow_color
        for r, c in target_cells:
            grid[r, c] = target_color
        blocked |= reserve
        return arrow_cells, target_cells, reserve
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_H_MIN, GRID_H_MAX)
    w = rng.randint(GRID_W_MIN, GRID_W_MAX)
    bg = rng.randrange(10)
    grid = np.full((h, w), bg, dtype=int)

    arrow_count = rng.randint(ARROW_COUNT_MIN, ARROW_COUNT_MAX)
    if 1 + 2 * arrow_count > 10:
        return None

    palette = [color for color in range(10) if color != bg]
    rng.shuffle(palette)
    arrow_colors = palette[:arrow_count]
    target_colors = palette[arrow_count : 2 * arrow_count]
    directions = _balanced_directions(arrow_count, rng)

    blocked: set[tuple[int, int]] = set()
    for arrow_color, target_color, direction in zip(arrow_colors, target_colors, directions):
        placed = _place_pair(grid, blocked, arrow_color, target_color, direction, rng)
        if placed is None:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
