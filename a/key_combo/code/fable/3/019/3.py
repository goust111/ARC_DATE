from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, in_bounds


BLACK = Color.BLACK
GREY = Color.GREY
RAY_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]

# Visual clockwise order on the grid: up -> right -> down -> left.
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
TOP_DIR = 2
RIGHT_DIR = 3
BOTTOM_DIR = 0
LEFT_DIR = 1

GEN_ATTEMPTS = 400
RAY_ATTEMPTS = 300
FUEL_ATTEMPTS = 120
MIN_SIDE = 12
MAX_SIDE = 22
MIN_RAYS = 2
MAX_RAYS = 4
MIN_PATH_LEN = 2
MAX_PATH_LEN = 10
MAX_SEGMENT_LEN = 6


def _is_border(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _marker_side(shape: tuple[int, int], r: int, c: int) -> str:
    h, w = shape
    if r == 0 and 0 < c < w - 1:
        return "top"
    if c == w - 1 and 0 < r < h - 1:
        return "right"
    if r == h - 1 and 0 < c < w - 1:
        return "bottom"
    if c == 0 and 0 < r < h - 1:
        return "left"
    raise AssertionError("marker must lie on exactly one non-corner border side")


def _inward_dir(shape: tuple[int, int], r: int, c: int) -> int:
    side = _marker_side(shape, r, c)
    if side == "top":
        return TOP_DIR
    if side == "right":
        return RIGHT_DIR
    if side == "bottom":
        return BOTTOM_DIR
    return LEFT_DIR


def _border_positions(shape: tuple[int, int], side: str) -> list[tuple[int, int]]:
    h, w = shape
    if side == "top":
        return [(0, c) for c in range(1, w - 1)]
    if side == "right":
        return [(r, w - 1) for r in range(1, h - 1)]
    if side == "bottom":
        return [(h - 1, c) for c in range(1, w - 1)]
    return [(r, 0) for r in range(1, h - 1)]


def _rotate_clockwise(dir_idx: int) -> int:
    return (dir_idx + 1) % 4


def _step(r: int, c: int, dir_idx: int) -> tuple[int, int]:
    dr, dc = DIRS[dir_idx]
    return r + dr, c + dc


def _forward_free_cells(
    shape: tuple[int, int],
    occupied: np.ndarray,
    start: tuple[int, int],
    dir_idx: int,
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    r, c = start
    while True:
        nr, nc = _step(r, c, dir_idx)
        if not in_bounds(shape, nr, nc) or occupied[nr, nc]:
            break
        cells.append((nr, nc))
        r, c = nr, nc
    return cells


def _distance_to_border(shape: tuple[int, int], start: tuple[int, int], dir_idx: int) -> int:
    r, c = start
    h, w = shape
    if dir_idx == 0:
        return r
    if dir_idx == 1:
        return w - 1 - c
    if dir_idx == 2:
        return h - 1 - r
    return c


def _trace_ray(
    out: np.ndarray,
    start: tuple[int, int],
    dir_idx: int,
    fuel: int,
    color: int,
) -> None:
    r, c = start
    for _ in range(int(fuel)):
        for _ in range(4):
            nr, nc = _step(r, c, dir_idx)
            if in_bounds(out.shape, nr, nc) and int(out[nr, nc]) != GREY:
                break
            dir_idx = _rotate_clockwise(dir_idx)
        else:
            raise AssertionError("ray is trapped on all four sides")
        r, c = nr, nc
        out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each border marker into a same-color ray whose painted length is set by scattered fuel pixels.

    Input structure:
    - The background is black.
    - Several distinct non-grey marker pixels sit on non-corner border cells.
    - Interior pixels of each marker color are scattered fuel units, and grey pixels are blockers.

    Rule:
    - Keep the border markers and grey blockers.
    - For each marker, count the interior pixels of its color; that count is the ray length.
    - Starting from the marker and moving perpendicular to its border side, paint one cell per fuel unit.
      Whenever the next cell would be grey or outside the grid, rotate the direction clockwise and continue.
    - All fuel pixels are erased unless a ray later repaints their location.

    Key insight:
    - Same-color interior pixels are not part of the final shape; they are a length supply that gets converted into
      a routed path, while grey pixels only steer the path and do not consume any length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    border_mask = np.zeros_like(g, dtype=bool)
    border_mask[0, :] = True
    border_mask[h - 1, :] = True
    border_mask[:, 0] = True
    border_mask[:, w - 1] = True

    marker_positions: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(border_mask & (g != BLACK) & (g != GREY)):
        r = int(r)
        c = int(c)
        if (r in (0, h - 1)) and (c in (0, w - 1)):
            continue
        marker_positions.append((r, c, int(g[r, c])))
    assert marker_positions, "expected at least one border marker"
    colors = [color for _, _, color in marker_positions]
    assert len(colors) == len(set(colors)), "marker colors must be distinct"

    out = np.full_like(g, BLACK)
    out[g == GREY] = GREY
    for r, c, color in marker_positions:
        out[r, c] = color

    for r, c, color in sorted(marker_positions):
        dir_idx = _inward_dir((h, w), r, c)
        fuel = int(np.count_nonzero(g == color)) - 1
        assert fuel >= 0, "fuel count must be nonnegative"
        _trace_ray(out, (r, c), dir_idx, fuel, color)

    return out


def _build_ray(
    shape: tuple[int, int],
    occupied: np.ndarray,
    color: int,
    *,
    require_blocker_turn: bool = False,
    require_boundary_turn: bool = False,
    force_straight: bool = False,
) -> dict[str, object] | None:
    sides = ["top", "right", "bottom", "left"]
    for _ in range(RAY_ATTEMPTS):
        side = random.choice(sides)
        marker_candidates = [pos for pos in _border_positions(shape, side) if not occupied[pos]]
        if not marker_candidates:
            continue
        marker = random.choice(marker_candidates)
        dir_idx = _inward_dir(shape, marker[0], marker[1])
        if force_straight:
            segments = 1
        elif require_blocker_turn or require_boundary_turn:
            segments = random.choices([2, 3, 4], weights=[6, 3, 1])[0]
        else:
            segments = random.choices([1, 2, 3, 4], weights=[4, 4, 2, 1])[0]

        local_occ = occupied.copy()
        local_occ[marker] = True
        current = marker
        current_dir = dir_idx
        path: list[tuple[int, int]] = []
        blockers: list[tuple[int, int]] = []
        used_blocker_turn = False
        used_boundary_turn = False
        ok = True

        for seg_idx in range(segments):
            last = seg_idx == segments - 1
            free_cells = _forward_free_cells(shape, local_occ, current, current_dir)
            if not free_cells:
                ok = False
                break

            choices: list[tuple[str, int, tuple[int, int] | None]] = []

            if last:
                max_len = min(len(free_cells), MAX_SEGMENT_LEN)
                for run_len in range(MIN_PATH_LEN if not path else 1, max_len + 1):
                    choices.append(("end", run_len, None))
            else:
                dist_border = _distance_to_border(shape, current, current_dir)
                if MIN_PATH_LEN <= dist_border <= min(len(free_cells), MAX_SEGMENT_LEN):
                    endpoint = free_cells[dist_border - 1]
                    turned = _step(endpoint[0], endpoint[1], _rotate_clockwise(current_dir))
                    if in_bounds(shape, turned[0], turned[1]) and not local_occ[turned]:
                        choices.append(("boundary", dist_border, None))

                max_run = min(len(free_cells) - 1, MAX_SEGMENT_LEN)
                for run_len in range(1, max_run + 1):
                    blocker = free_cells[run_len]
                    if _is_border(shape, blocker[0], blocker[1]):
                        continue
                    endpoint = free_cells[run_len - 1]
                    turned = _step(endpoint[0], endpoint[1], _rotate_clockwise(current_dir))
                    if in_bounds(shape, turned[0], turned[1]) and not local_occ[turned]:
                        choices.append(("blocker", run_len, blocker))

            if require_blocker_turn and not used_blocker_turn:
                blocker_choices = [choice for choice in choices if choice[0] == "blocker"]
                if blocker_choices:
                    choices = blocker_choices
            elif require_boundary_turn and not used_boundary_turn:
                boundary_choices = [choice for choice in choices if choice[0] == "boundary"]
                if boundary_choices:
                    choices = boundary_choices

            if not choices:
                ok = False
                break

            mode, run_len, blocker = random.choice(choices)
            segment_cells = free_cells[:run_len]
            path.extend(segment_cells)
            for cell in segment_cells:
                local_occ[cell] = True
            current = segment_cells[-1]

            if mode == "blocker":
                assert blocker is not None
                blockers.append(blocker)
                local_occ[blocker] = True
                used_blocker_turn = True
                current_dir = _rotate_clockwise(current_dir)
            elif mode == "boundary":
                used_boundary_turn = True
                current_dir = _rotate_clockwise(current_dir)

        if not ok:
            continue
        if not (MIN_PATH_LEN <= len(path) <= MAX_PATH_LEN):
            continue
        if require_blocker_turn and not used_blocker_turn:
            continue
        if require_boundary_turn and not used_boundary_turn:
            continue

        return {
            "color": int(color),
            "marker": marker,
            "path": path,
            "blockers": blockers,
            "has_blocker_turn": used_blocker_turn,
            "has_boundary_turn": used_boundary_turn,
        }
    return None


def _paint_blueprint(shape: tuple[int, int], rays: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for ray in rays:
        color = int(ray["color"])
        mr, mc = ray["marker"]
        out[mr, mc] = color
        for r, c in ray["path"]:
            out[r, c] = color
        for r, c in ray["blockers"]:
            out[r, c] = GREY
    return out


def _scatter_fuels(
    shape: tuple[int, int],
    rays: list[dict[str, object]],
    blueprint: np.ndarray,
) -> np.ndarray | None:
    inp = np.full(shape, BLACK, dtype=int)
    inp[blueprint == GREY] = GREY
    reserved = blueprint != BLACK

    for ray in rays:
        mr, mc = ray["marker"]
        inp[mr, mc] = int(ray["color"])

    total_fuel = sum(len(ray["path"]) for ray in rays)
    interior = [(r, c) for r in range(1, shape[0] - 1) for c in range(1, shape[1] - 1) if not reserved[r, c]]
    if len(interior) < total_fuel:
        return None

    for _ in range(FUEL_ATTEMPTS):
        shuffled = interior[:]
        random.shuffle(shuffled)
        banned = reserved.copy()
        chosen: list[tuple[int, int]] = []
        for r, c in shuffled:
            if banned[r, c]:
                continue
            chosen.append((r, c))
            for rr in range(max(1, r - 1), min(shape[0] - 1, r + 2)):
                for cc in range(max(1, c - 1), min(shape[1] - 1, c + 2)):
                    banned[rr, cc] = True
            if len(chosen) == total_fuel:
                break
        if len(chosen) < total_fuel:
            continue

        random.shuffle(chosen)
        out = inp.copy()
        idx = 0
        for ray in rays:
            fuel_count = len(ray["path"])
            for r, c in chosen[idx : idx + fuel_count]:
                out[r, c] = int(ray["color"])
            idx += fuel_count
        return out
    return None


def _build_blueprint() -> tuple[np.ndarray, list[dict[str, object]]] | None:
    h = random.randint(MIN_SIDE, MAX_SIDE)
    w = random.randint(MIN_SIDE, MAX_SIDE)
    shape = (h, w)
    ray_count = random.randint(MIN_RAYS, MAX_RAYS)
    colors = random.sample(RAY_COLORS, ray_count)

    occupied = np.zeros(shape, dtype=bool)
    rays: list[dict[str, object]] = []

    boundary_required_index = 1 if ray_count > 1 and random.random() < 0.7 else None
    straight_preferred_index = 2 if ray_count > 2 and random.random() < 0.65 else None

    for idx, color in enumerate(colors):
        ray = _build_ray(
            shape,
            occupied,
            color,
            require_blocker_turn=(idx == 0),
            require_boundary_turn=(boundary_required_index == idx),
            force_straight=(straight_preferred_index == idx),
        )
        if ray is None:
            return None
        rays.append(ray)
        occupied[ray["marker"]] = True
        for cell in ray["path"]:
            occupied[cell] = True
        for cell in ray["blockers"]:
            occupied[cell] = True

    blueprint = _paint_blueprint(shape, rays)
    if not np.any(blueprint == GREY):
        return None
    if not any(bool(ray["has_blocker_turn"]) for ray in rays):
        return None
    return blueprint, rays


def _validate_instance(inp: np.ndarray, blueprint: np.ndarray, rays: list[dict[str, object]]) -> bool:
    if not np.array_equal(inp == GREY, blueprint == GREY):
        return False
    if not np.array_equal(inp[blueprint == GREY], blueprint[blueprint == GREY]):
        return False

    for ray in rays:
        color = int(ray["color"])
        mr, mc = ray["marker"]
        if int(inp[mr, mc]) != color:
            return False
        if sum(1 for rr, cc in ray["path"] if int(inp[rr, cc]) == color) != 0:
            return False
        fuel = int(np.count_nonzero(inp == color)) - 1
        if fuel != len(ray["path"]):
            return False

    solved = main(inp)
    return bool(np.array_equal(solved, blueprint))


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_blueprint()
        if built is None:
            continue
        blueprint, rays = built
        inp = _scatter_fuels(blueprint.shape, rays, blueprint)
        if inp is None:
            continue
        if _validate_instance(inp, blueprint, rays):
            return inp
    raise RuntimeError("failed to generate a valid fuel-ray instance")
