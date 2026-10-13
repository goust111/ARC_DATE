from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_SOLVERS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVERS_DIR) not in sys.path:
    sys.path.append(str(_SOLVERS_DIR))

from arc_utils import Color, bbox_coords, find_components, get_bg


GREY = Color.GREY
BLACK = Color.BLACK
ARC_COLORS = tuple(range(10))
BORDER_COLORS = tuple(c for c in ARC_COLORS if c not in (BLACK, GREY))
NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)


def _direction(color: int, top_color: int, bottom_color: int) -> int:
    assert color in {top_color, bottom_color}
    return -1 if int(color) == int(top_color) else 1


def _other_color(color: int, top_color: int, bottom_color: int) -> int:
    assert color in {top_color, bottom_color}
    return int(bottom_color) if int(color) == int(top_color) else int(top_color)


def _parse_grid(grid: np.ndarray) -> tuple[np.ndarray, int, int, int, list[dict], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert 4 <= h <= 30 and 1 <= w <= 30, "grid dimensions must fit ARC bounds"

    bg = int(get_bg(g))
    top_color = int(g[0, 0])
    bottom_color = int(g[h - 1, 0])
    assert top_color != bottom_color, "border bars must have two different colors"
    assert top_color not in {bg, GREY} and bottom_color not in {bg, GREY}
    assert bool(np.all(g[0, :] == top_color)), "top row must be a full color bar"
    assert bool(np.all(g[h - 1, :] == bottom_color)), "bottom row must be a full color bar"
    assert set(map(int, np.unique(g))).issubset({bg, top_color, bottom_color, GREY})

    switches = {tuple(map(int, rc)) for rc in np.argwhere(g == GREY)}
    for r, c in switches:
        assert 0 < r < h - 1, "switches must lie in the middle area"
        for dr, dc in NEIGHBORS8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                assert int(g[rr, cc]) != GREY, "gray switches must be isolated pixels"

    middle = g[1 : h - 1, :]
    objects: list[dict] = []
    for color in (top_color, bottom_color):
        for comp in find_components(
            middle,
            bg=bg,
            connectivity=8,
            monochromatic=True,
            colors=[color],
            return_format="dict",
        ):
            coords = np.asarray(comp["coords"], dtype=int) + np.array([1, 0], dtype=int)
            b = bbox_coords(coords)
            r0, _c0, r1, _c1 = b
            assert 0 < r0 <= r1 < h - 1, "objects must stay between the border bars"
            objects.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": b,
                    "area": int(coords.shape[0]),
                }
            )

    assert objects, "expected at least one sliding object"
    objects.sort(key=lambda obj: (obj["bbox"][1], obj["bbox"][0], obj["color"]))

    used_cols: set[int] = set()
    for obj in objects:
        cols = set(map(int, obj["coords"][:, 1]))
        assert used_cols.isdisjoint(cols), "sliding objects must not share columns"
        used_cols.update(cols)

    return g, bg, top_color, bottom_color, objects, switches


def _at_target(coords: np.ndarray, color: int, h: int, top_color: int, bottom_color: int) -> bool:
    r0, _c0, r1, _c1 = bbox_coords(coords)
    if int(color) == int(top_color):
        return int(r0) == 1
    assert int(color) == int(bottom_color)
    return int(r1) == h - 2


def _solve(grid: np.ndarray, *, trace: bool = False):
    g, bg, top_color, bottom_color, objects, switches = _parse_grid(grid)
    h, _w = g.shape
    live_switches = set(switches)
    finals: list[tuple[np.ndarray, int]] = []
    trigger_log: list[dict] = []

    max_steps = h * (len(switches) + len(objects) + 5)
    for obj_index, obj in enumerate(objects):
        coords = np.asarray(obj["coords"], dtype=int).copy()
        color = int(obj["color"])
        obj_triggers: list[tuple[int, int]] = []
        steps = 0

        while not _at_target(coords, color, h, top_color, bottom_color):
            direction = _direction(color, top_color, bottom_color)
            next_coords = coords + np.array([direction, 0], dtype=int)
            assert bool(np.all((1 <= next_coords[:, 0]) & (next_coords[:, 0] <= h - 2)))

            hit = sorted(
                {
                    (int(r), int(c))
                    for r, c in next_coords
                    if (int(r), int(c)) in live_switches
                }
            )
            coords = next_coords
            if hit:
                for cell in hit:
                    live_switches.remove(cell)
                    obj_triggers.append(cell)
                color = _other_color(color, top_color, bottom_color)

            steps += 1
            assert steps <= max_steps, "sliding simulation did not terminate"

        finals.append((coords, color))
        trigger_log.append({"object_index": obj_index, "switches": obj_triggers})

    out = np.full_like(g, bg)
    out[0, :] = top_color
    out[h - 1, :] = bottom_color

    for r, c in sorted(live_switches):
        out[r, c] = GREY

    for coords, color in finals:
        assert bool(np.all(out[coords[:, 0], coords[:, 1]] == bg))
        out[coords[:, 0], coords[:, 1]] = int(color)

    if trace:
        return out, {
            "triggered_switches": sorted(set(switches) - live_switches),
            "remaining_switches": sorted(live_switches),
            "object_triggers": trigger_log,
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide border-colored objects vertically, reversing and recoloring at gray switches.

    Input structure:
    - The top and bottom rows are full-width bars in two different non-gray colors.
    - The middle area contains separated monochrome objects whose colors match one
      of the two bars.
    - Isolated gray pixels are switches, and no two sliding objects use the same columns.

    Rule:
    - Each object slides vertically toward the border bar with the same color.
    - Whenever any cell of the moving object steps onto a gray switch, that switch
      disappears, the whole object changes to the other border color, and the object
      reverses direction.
    - Further switches are handled the same way until the object stops against the
      inside row of the border matching its current color; untouched switches remain.

    Key insight:
    - The final side and color are determined by the ordered switches encountered
      along the vertical trajectory, so the path must be simulated segment by segment.
    """
    return _solve(grid)


# --------------- generator ---------------

GENERATION_RETRIES = 500
HEIGHT_RANGE = (12, 25)
WIDTH_RANGE = (12, 28)
OBJECT_COUNT_RANGE = (2, 5)
EXTRA_SWITCH_RANGE = (0, 2)
CHAIN_EXTRA_SWITCH_RANGE = (1, 3)

_GEN_CALLS = 0

SHAPE_MASKS = (
    np.array([[1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _random_shape() -> np.ndarray:
    mask = random.choice(SHAPE_MASKS).copy()
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    if random.random() < 0.5:
        mask = np.flipud(mask)
    return _trim_mask(mask)


def _spread_columns(widths: list[int], grid_w: int) -> list[int] | None:
    min_total = sum(widths) + len(widths) + 1
    if min_total > grid_w:
        return None

    gaps = [1] * (len(widths) + 1)
    for _ in range(grid_w - min_total):
        gaps[random.randrange(len(gaps))] += 1

    starts: list[int] = []
    c = gaps[0]
    for i, width in enumerate(widths):
        starts.append(c)
        c += width + gaps[i + 1]
    return starts


def _coords_from_mask(mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    return np.argwhere(mask) + np.array([int(r0), int(c0)], dtype=int)


def _paint_coords(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _cell_is_away_from_gray(cell: tuple[int, int], switches: set[tuple[int, int]]) -> bool:
    r, c = cell
    return all(abs(r - rr) > 1 or abs(c - cc) > 1 for rr, cc in switches)


def _available_switch_cells(
    moved_coords: np.ndarray,
    occupied: np.ndarray,
    switches: set[tuple[int, int]],
    h: int,
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r, c in moved_coords:
        cell = (int(r), int(c))
        if not (0 < cell[0] < h - 1):
            continue
        if occupied[cell]:
            continue
        if not _cell_is_away_from_gray(cell, switches):
            continue
        cells.append(cell)
    random.shuffle(cells)
    return cells


def _segment_distance(coords: np.ndarray, color: int, h: int, top_color: int, bottom_color: int) -> int:
    r0, _c0, r1, _c1 = bbox_coords(coords)
    if int(color) == int(top_color):
        return int(r0 - 1)
    assert int(color) == int(bottom_color)
    return int((h - 2) - r1)


def _plan_object_switches(
    coords0: np.ndarray,
    color0: int,
    event_count: int,
    occupied: np.ndarray,
    switches: set[tuple[int, int]],
    h: int,
    top_color: int,
    bottom_color: int,
) -> list[tuple[int, int]] | None:
    if event_count == 0:
        return []

    for _attempt in range(120):
        coords = coords0.copy()
        color = int(color0)
        local_switches = set(switches)
        planned: list[tuple[int, int]] = []

        ok = True
        for event_index in range(event_count):
            distance = _segment_distance(coords, color, h, top_color, bottom_color)
            if distance <= 0:
                ok = False
                break

            direction = _direction(color, top_color, bottom_color)
            choices: list[tuple[int, tuple[int, int], np.ndarray]] = []
            step_order = list(range(1, distance + 1))
            random.shuffle(step_order)
            for step in step_order:
                moved = coords + np.array([direction * step, 0], dtype=int)
                candidates = _available_switch_cells(moved, occupied, local_switches, h)
                for cell in candidates:
                    choices.append((step, cell, moved))

            random.shuffle(choices)
            if not choices:
                ok = False
                break

            remaining = event_count - event_index - 1
            picked = None
            for _step, cell, moved in choices:
                next_color = _other_color(color, top_color, bottom_color)
                if remaining == 0 or _segment_distance(moved, next_color, h, top_color, bottom_color) > 0:
                    picked = (cell, moved, next_color)
                    break
            if picked is None:
                ok = False
                break

            cell, coords, color = picked
            local_switches.add(cell)
            planned.append(cell)

        if ok and len(planned) == event_count:
            return planned
    return None


def _direct_slide_output(grid: np.ndarray) -> np.ndarray:
    g, bg, top_color, bottom_color, objects, switches = _parse_grid(grid)
    h, _w = g.shape
    out = np.full_like(g, bg)
    out[0, :] = top_color
    out[h - 1, :] = bottom_color

    for r, c in sorted(switches):
        out[r, c] = GREY

    for obj in objects:
        coords = np.asarray(obj["coords"], dtype=int)
        color = int(obj["color"])
        r0, _c0, r1, _c1 = bbox_coords(coords)
        if color == top_color:
            moved = coords + np.array([1 - r0, 0], dtype=int)
        else:
            moved = coords + np.array([(h - 2) - r1, 0], dtype=int)
        out[moved[:, 0], moved[:, 1]] = color
    return out


def _add_extra_switches(
    grid: np.ndarray,
    count: int,
    forbidden_cols: set[int],
    switches: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = grid.shape
    added: list[tuple[int, int]] = []
    for _ in range(count):
        candidates = [
            (r, c)
            for r in range(1, h - 1)
            for c in range(w)
            if c not in forbidden_cols
            and int(grid[r, c]) == BLACK
            and _cell_is_away_from_gray((r, c), switches)
        ]
        if not candidates:
            break
        cell = random.choice(candidates)
        grid[cell] = GREY
        switches.add(cell)
        added.append(cell)
    return added


def _scenario_event_counts(object_count: int, scenario: int) -> list[int]:
    counts = [0] * object_count
    indices = list(range(object_count))
    random.shuffle(indices)

    if scenario == 1:
        counts[indices[0]] = 1
    elif scenario == 2:
        counts[indices[0]] = random.choice([2, 3])
        if object_count > 2 and random.random() < 0.45:
            counts[indices[1]] = 1
    return counts


def _gen_attempt(scenario: int) -> np.ndarray | None:
    h = random.randint(*(HEIGHT_RANGE if scenario != 2 else (16, HEIGHT_RANGE[1])))
    w = random.randint(*WIDTH_RANGE)
    top_color, bottom_color = random.sample(BORDER_COLORS, 2)

    object_count = random.randint(*OBJECT_COUNT_RANGE)
    masks = [_random_shape() for _ in range(object_count)]
    starts = _spread_columns([int(mask.shape[1]) for mask in masks], w)
    if starts is None:
        return None

    colors = [top_color, bottom_color]
    colors.extend(random.choice([top_color, bottom_color]) for _ in range(object_count - 2))
    random.shuffle(colors)

    grid = np.full((h, w), BLACK, dtype=int)
    grid[0, :] = top_color
    grid[h - 1, :] = bottom_color

    objects: list[dict] = []
    for mask, c0, color in zip(masks, starts, colors):
        sh, _sw = map(int, mask.shape)
        lo = 2
        hi = h - sh - 2
        if hi < lo:
            return None
        r0 = random.randint(lo, hi)
        coords = _coords_from_mask(mask, r0, c0)
        _paint_coords(grid, coords, color)
        objects.append({"coords": coords, "color": int(color)})

    occupied = grid != BLACK
    desired_counts = _scenario_event_counts(object_count, scenario)
    switches: set[tuple[int, int]] = set()
    planned_switches: list[tuple[int, int]] = []

    for obj, event_count in zip(objects, desired_counts):
        planned = _plan_object_switches(
            obj["coords"],
            int(obj["color"]),
            int(event_count),
            occupied,
            switches,
            h,
            top_color,
            bottom_color,
        )
        if planned is None:
            return None
        for cell in planned:
            grid[cell] = GREY
            switches.add(cell)
            planned_switches.append(cell)

    object_cols = {int(c) for obj in objects for c in obj["coords"][:, 1]}
    if scenario == 2:
        extra_count = random.randint(*CHAIN_EXTRA_SWITCH_RANGE)
    else:
        extra_count = random.randint(*EXTRA_SWITCH_RANGE)
    extra_switches = _add_extra_switches(grid, extra_count, object_cols, switches)

    if not _gen_validates(grid, scenario, set(planned_switches), set(extra_switches)):
        return None
    return grid


def _gen_validates(
    grid: np.ndarray,
    scenario: int,
    planned_switches: set[tuple[int, int]],
    extra_switches: set[tuple[int, int]],
) -> bool:
    try:
        out, trace = _solve(grid, trace=True)
    except AssertionError:
        return False

    if out.ndim != 2 or max(out.shape) > 30 or max(grid.shape) > 30:
        return False
    if np.array_equal(out, grid):
        return False

    triggered = set(map(tuple, trace["triggered_switches"]))
    if triggered != planned_switches:
        return False
    if any(int(out[cell]) != GREY for cell in extra_switches):
        return False

    per_object_counts = [len(item["switches"]) for item in trace["object_triggers"]]
    if scenario == 0 and any(per_object_counts):
        return False
    if scenario == 1 and max(per_object_counts, default=0) != 1:
        return False
    if scenario == 2 and max(per_object_counts, default=0) < 2:
        return False

    if planned_switches and np.array_equal(out, _direct_slide_output(grid)):
        return False
    return True


def generate_input() -> np.ndarray:
    global _GEN_CALLS
    scenario = _GEN_CALLS % 3
    _GEN_CALLS += 1

    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(scenario)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-switch input")
