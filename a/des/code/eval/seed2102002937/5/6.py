import random
from collections import defaultdict

import numpy as np


BLACK = 0
GREY = 5
PALETTE = tuple(color for color in range(10) if color not in (BLACK, GREY))
DIRECTIONS = ("up", "down", "left", "right")
ARM_START = {
    "up": (-2, 0),
    "down": (2, 0),
    "left": (0, -2),
    "right": (0, 2),
}
ARM_STEP = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# --------------- solver ---------------


def _connected_components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in steps:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _find_towers(grid: np.ndarray) -> list[dict]:
    towers: list[dict] = []
    for coords in _connected_components(grid == GREY, connectivity=4):
        rows = [r for r, _c in coords]
        cols = [c for _r, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if len(coords) != 9 or r1 - r0 != 2 or c1 - c0 != 2:
            continue
        if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY):
            continue
        towers.append(
            {
                "center": (r0 + 1, c0 + 1),
                "body_bbox": (r0, c0, r1, c1),
            }
        )
    towers.sort(key=lambda tower: (tower["center"][0], tower["center"][1]))
    assert towers, "expected at least one grey 3x3 tower"
    return towers


def _scan_arm(grid: np.ndarray, center: tuple[int, int], direction: str) -> tuple[int, list[tuple[int, int]]]:
    h, w = grid.shape
    sr, sc = ARM_START[direction]
    dr, dc = ARM_STEP[direction]
    r = center[0] + sr
    c = center[1] + sc
    if not (0 <= r < h and 0 <= c < w):
        return 0, []
    color = int(grid[r, c])
    if color in (BLACK, GREY):
        return 0, []

    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == color:
        cells.append((r, c))
        r += dr
        c += dc
    return len(cells), cells


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], list[tuple[int, int, int]]]:
    towers = _find_towers(grid)
    arm_set: set[tuple[int, int]] = set()

    for tower in towers:
        center = tower["center"]
        arms: dict[str, int] = {}
        tower_arm_cells: list[tuple[int, int]] = []
        for direction in DIRECTIONS:
            arm_len, cells = _scan_arm(grid, center, direction)
            arms[direction] = arm_len
            for cell in cells:
                assert cell not in arm_set, "arms must not overlap"
                arm_set.add(cell)
                tower_arm_cells.append(cell)

        cr, cc = center
        tower["arms"] = arms
        tower["arm_cells"] = tower_arm_cells
        tower["cover_bbox"] = (
            cr - 1 - arms["up"],
            cc - 1 - arms["left"],
            cr + 1 + arms["down"],
            cc + 1 + arms["right"],
        )

    receiver_mask = (grid != BLACK) & (grid != GREY)
    for r, c in arm_set:
        receiver_mask[r, c] = False

    receivers: list[tuple[int, int, int]] = []
    for coords in _connected_components(receiver_mask, connectivity=8):
        assert len(coords) == 1, "receivers must be isolated single pixels"
        r, c = coords[0]
        receivers.append((r, c, int(grid[r, c])))

    return towers, receivers


def main(grid: np.ndarray) -> np.ndarray:
    """Use arm lengths as asymmetric rectangle extents, then recolor each tower from the nearest receiver inside its coverage."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    towers, receivers = _parse_scene(g)
    out = g.copy()

    for tower in towers:
        for r, c in tower["arm_cells"]:
            out[r, c] = BLACK

        cr, cc = tower["center"]
        r0, c0, r1, c1 = tower["body_bbox"]
        top, left, bottom, right = tower["cover_bbox"]

        candidates = []
        for rr, cc2, color in receivers:
            if top <= rr <= bottom and left <= cc2 <= right:
                dist2 = (rr - cr) * (rr - cr) + (cc2 - cc) * (cc2 - cc)
                candidates.append((dist2, rr, cc2, color))

        if candidates:
            _dist2, _rr, _cc2, fill_color = min(candidates)
            out[r0 : r1 + 1, c0 : c1 + 1] = fill_color

    return out


# --------------- generator ---------------

GRID_MIN = 22
GRID_MAX = 30
TOWER_COUNT_CHOICES = (2, 3)
ARM_LENGTH_MIN = 1
ARM_LENGTH_MAX = 5
MAX_GENERATE_ATTEMPTS = 400
MAX_TOWER_TRIES = 200
EXTRA_RECEIVER_RANGE = (0, 2)


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    at, al, ab, ar = a
    bt, bl, bb, br = b
    return not (ab < bt or bb < at or ar < bl or br < al)


def _neighbor_cells8(cell: tuple[int, int], h: int, w: int) -> set[tuple[int, int]]:
    r, c = cell
    result = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                result.add((nr, nc))
    return result


def _neighbor_cells4(cell: tuple[int, int], h: int, w: int) -> set[tuple[int, int]]:
    r, c = cell
    result = {(r, c)}
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            result.add((nr, nc))
    return result


def _body_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    cr, cc = center
    return {(r, c) for r in range(cr - 1, cr + 2) for c in range(cc - 1, cc + 2)}


def _sample_tower_spec(rng: random.Random, desired: str) -> tuple[dict[str, int], dict[str, int]]:
    while True:
        min_arms = 2 if desired == "multi" else 1
        arm_count = rng.randint(min_arms, 4)
        dirs = rng.sample(DIRECTIONS, arm_count)
        lengths = {direction: 0 for direction in DIRECTIONS}
        for direction in dirs:
            lengths[direction] = rng.randint(ARM_LENGTH_MIN, ARM_LENGTH_MAX)
        if desired != "multi" or sum(lengths.values()) >= 3:
            break

    colors = {direction: rng.choice(PALETTE) for direction in dirs}
    return lengths, colors


def _make_tower(
    h: int,
    w: int,
    center: tuple[int, int],
    lengths: dict[str, int],
    colors: dict[str, int],
    desired: str,
) -> dict:
    cr, cc = center
    body = _body_cells(center)
    arm_cells_by_dir: dict[str, list[tuple[int, int]]] = {}
    all_arm_cells: list[tuple[int, int]] = []
    forbidden_receivers: set[tuple[int, int]] = set()

    for direction in DIRECTIONS:
        start_dr, start_dc = ARM_START[direction]
        step_dr, step_dc = ARM_STEP[direction]
        length = lengths[direction]
        cells: list[tuple[int, int]] = []
        if length == 0:
            nr = cr + start_dr
            nc = cc + start_dc
            if 0 <= nr < h and 0 <= nc < w:
                forbidden_receivers.add((nr, nc))
        else:
            for k in range(length):
                nr = cr + start_dr + step_dr * k
                nc = cc + start_dc + step_dc * k
                cells.append((nr, nc))
                all_arm_cells.append((nr, nc))
            nr = cr + start_dr + step_dr * length
            nc = cc + start_dc + step_dc * length
            if 0 <= nr < h and 0 <= nc < w:
                forbidden_receivers.add((nr, nc))
        arm_cells_by_dir[direction] = cells

    return {
        "center": center,
        "desired": desired,
        "body_cells": body,
        "body_bbox": (cr - 1, cc - 1, cr + 1, cc + 1),
        "arm_lengths": lengths,
        "arm_colors": colors,
        "arm_cells_by_dir": arm_cells_by_dir,
        "arm_cells": all_arm_cells,
        "cover_bbox": (
            cr - 1 - lengths["up"],
            cc - 1 - lengths["left"],
            cr + 1 + lengths["down"],
            cc + 1 + lengths["right"],
        ),
        "forbidden_receivers": forbidden_receivers,
    }


def _available_cells_for_tower(
    tower: dict,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    top, left, bottom, right = tower["cover_bbox"]
    available: list[tuple[int, int, int]] = []
    cr, cc = tower["center"]
    occupied = tower["body_cells"] | set(tower["arm_cells"])
    forbidden = tower["forbidden_receivers"]
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            if (r, c) in occupied or (r, c) in blocked or (r, c) in forbidden:
                continue
            dist2 = (r - cr) * (r - cr) + (c - cc) * (c - cc)
            available.append((dist2, r, c))
    return available


def _try_pick_receivers(
    rng: random.Random,
    tower: dict,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    available = _available_cells_for_tower(tower, blocked)
    rng.shuffle(available)
    available.sort(key=lambda item: (item[0], item[1], item[2]))

    if tower["desired"] == "none":
        return []
    if tower["desired"] == "single":
        if not available:
            return None
        _dist2, r, c = rng.choice(available)
        return [(r, c)]

    assert tower["desired"] == "multi"
    near_candidates = available[:]
    rng.shuffle(near_candidates)
    preferred_counts = [2, 3]
    rng.shuffle(preferred_counts)

    for near_dist, nr, nc in near_candidates:
        farther = [(d, r, c) for d, r, c in available if d > near_dist]
        if not farther:
            continue
        rng.shuffle(farther)
        farther.sort(key=lambda item: (item[0], item[1], item[2]))
        for target_count in preferred_counts:
            chosen = [(nr, nc)]
            for _dist2, r, c in farther:
                cell = (r, c)
                if all(abs(r - rr) > 1 or abs(c - cc) > 1 for rr, cc in chosen):
                    chosen.append(cell)
                    if len(chosen) == target_count:
                        return chosen
            if len(chosen) >= 2:
                return chosen
    return None


def _tower_count_for_grid(rng: random.Random, h: int, w: int) -> int:
    if min(h, w) < 24:
        return 2
    return rng.choice(TOWER_COUNT_CHOICES)


def _scenario_list(rng: random.Random, tower_count: int) -> list[str]:
    base = ["none", "single", "multi"]
    if tower_count == 2:
        scenarios = rng.sample(base, 2)
    else:
        scenarios = base[:]
        rng.shuffle(scenarios)
    scenarios.sort(key=lambda value: {"multi": 0, "single": 1, "none": 2}[value])
    return scenarios


def _place_towers(rng: random.Random, h: int, w: int) -> list[dict] | None:
    tower_count = _tower_count_for_grid(rng, h, w)
    scenarios = _scenario_list(rng, tower_count)
    towers: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    must_be_black: set[tuple[int, int]] = set()
    grey_body_buffer: set[tuple[int, int]] = set()

    for desired in scenarios:
        placed = None
        for _ in range(MAX_TOWER_TRIES):
            lengths, colors = _sample_tower_spec(rng, desired)
            min_r = 1 + lengths["up"]
            max_r = h - 2 - lengths["down"]
            min_c = 1 + lengths["left"]
            max_c = w - 2 - lengths["right"]
            if min_r > max_r or min_c > max_c:
                continue

            center = (rng.randint(min_r, max_r), rng.randint(min_c, max_c))
            tower = _make_tower(h, w, center, lengths, colors, desired)
            if any(_rects_overlap(tower["cover_bbox"], other["cover_bbox"]) for other in towers):
                continue
            tower_occupied = tower["body_cells"] | set(tower["arm_cells"])
            if tower["body_cells"] & grey_body_buffer:
                continue
            if tower_occupied & must_be_black:
                continue
            if tower["forbidden_receivers"] & occupied:
                continue
            if _try_pick_receivers(rng, tower, blocked=set()) is None:
                continue
            placed = tower
            break

        if placed is None:
            return None
        towers.append(placed)
        occupied |= placed["body_cells"] | set(placed["arm_cells"])
        must_be_black |= placed["forbidden_receivers"]
        for cell in placed["body_cells"]:
            grey_body_buffer |= _neighbor_cells4(cell, h, w)

    towers.sort(key=lambda tower: (tower["center"][0], tower["center"][1]))
    return towers


def _build_receiver_layout(
    rng: random.Random,
    h: int,
    w: int,
    towers: list[dict],
) -> tuple[list[tuple[int, int, int]], dict[int, list[tuple[int, int, int]]]] | None:
    receivers: list[tuple[int, int, int]] = []
    receiver_by_tower: dict[int, list[tuple[int, int, int]]] = {}
    blocked_receivers: set[tuple[int, int]] = set()

    for idx, tower in enumerate(towers):
        chosen = _try_pick_receivers(rng, tower, blocked_receivers)
        if chosen is None:
            return None

        colors = rng.sample(PALETTE, len(chosen)) if chosen else []
        tower_receivers: list[tuple[int, int, int]] = []
        for (r, c), color in zip(chosen, colors):
            receiver = (r, c, color)
            receivers.append(receiver)
            tower_receivers.append(receiver)
            blocked_receivers |= _neighbor_cells8((r, c), h, w)
        receiver_by_tower[idx] = tower_receivers

    forbidden_extra: set[tuple[int, int]] = set()
    covered: set[tuple[int, int]] = set()
    occupied: set[tuple[int, int]] = set()
    for tower in towers:
        occupied |= tower["body_cells"] | set(tower["arm_cells"])
        forbidden_extra |= tower["forbidden_receivers"]
        top, left, bottom, right = tower["cover_bbox"]
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                covered.add((r, c))

    extra_slots = []
    for r in range(h):
        for c in range(w):
            cell = (r, c)
            if cell in occupied or cell in covered or cell in blocked_receivers or cell in forbidden_extra:
                continue
            extra_slots.append(cell)

    rng.shuffle(extra_slots)
    extra_count = rng.randint(EXTRA_RECEIVER_RANGE[0], EXTRA_RECEIVER_RANGE[1])
    for cell in extra_slots:
        if extra_count == 0:
            break
        if cell in blocked_receivers or cell in occupied or cell in covered or cell in forbidden_extra:
            continue
        r, c = cell
        color = rng.choice(PALETTE)
        receiver = (r, c, color)
        receivers.append(receiver)
        blocked_receivers |= _neighbor_cells8(cell, h, w)
        extra_count -= 1

    return receivers, receiver_by_tower


def _render_input(h: int, w: int, towers: list[dict], receivers: list[tuple[int, int, int]]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for tower in towers:
        r0, c0, r1, c1 = tower["body_bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
        for direction, cells in tower["arm_cells_by_dir"].items():
            if not cells:
                continue
            color = tower["arm_colors"][direction]
            for r, c in cells:
                grid[r, c] = color

    for r, c, color in receivers:
        grid[r, c] = color
    return grid


def _expected_output(grid: np.ndarray, towers: list[dict], receivers: list[tuple[int, int, int]]) -> np.ndarray:
    out = grid.copy()
    for tower in towers:
        for r, c in tower["arm_cells"]:
            out[r, c] = BLACK

        cr, cc = tower["center"]
        top, left, bottom, right = tower["cover_bbox"]
        r0, c0, r1, c1 = tower["body_bbox"]
        candidates = []
        for rr, cc2, color in receivers:
            if top <= rr <= bottom and left <= cc2 <= right:
                dist2 = (rr - cr) * (rr - cr) + (cc2 - cc) * (cc2 - cc)
                candidates.append((dist2, rr, cc2, color))
        if candidates:
            _dist2, _rr, _cc2, fill_color = min(candidates)
            out[r0 : r1 + 1, c0 : c1 + 1] = fill_color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    towers = _place_towers(rng, h, w)
    if towers is None:
        return None

    receiver_layout = _build_receiver_layout(rng, h, w, towers)
    if receiver_layout is None:
        return None
    receivers, _receiver_by_tower = receiver_layout

    grid = _render_input(h, w, towers, receivers)
    expected = _expected_output(grid, towers, receivers)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Randomly generate non-overlapping towers, asymmetric arm rectangles, and receiver layouts covering the main rule branches."""
    rng = random.Random()
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATE_ATTEMPTS} attempts")
