from __future__ import annotations

from importlib.machinery import SourceFileLoader
import random

import numpy as np

_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared_fable_6_014_2",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
in_bounds = _ARC_UTILS.in_bounds
neighbors4 = _ARC_UTILS.neighbors4

BG = Color.BLACK
LAUNCHER_COLOR = Color.GREY
OBJECT_COLORS = [c for c in Color.NOT_BLACK if c != LAUNCHER_COLOR]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (15, 24)
GRID_W_RANGE = (15, 24)
CHAIN_LEN_RANGE = (1, 5)
DECOY_COUNT_RANGE = (0, 4)
STEP_RANGE = (4, 8)
GEN_ATTEMPTS = 500
LOCAL_ATTEMPTS = 80


def _add(a: tuple[int, int], b: tuple[int, int], k: int = 1) -> tuple[int, int]:
    return int(a[0] + k * b[0]), int(a[1] + k * b[1])


def _left(d: tuple[int, int]) -> tuple[int, int]:
    dr, dc = d
    return -dc, dr


def _right(d: tuple[int, int]) -> tuple[int, int]:
    dr, dc = d
    return dc, -dr


def _component_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in coords}


def _neighbor_degree(cells: set[tuple[int, int]], rc: tuple[int, int]) -> int:
    return sum((nr, nc) in cells for nr, nc in neighbors4(*rc))


def _launcher_cells(center: tuple[int, int], emit_dir: tuple[int, int]) -> set[tuple[int, int]]:
    perp = _left(emit_dir)
    return {
        center,
        _add(center, perp),
        _add(center, perp, -1),
        _add(center, emit_dir, -1),
    }


def _object_cells(center: tuple[int, int], emit_dir: tuple[int, int]) -> set[tuple[int, int]]:
    cells = {
        (center[0] + dr, center[1] + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
    }
    cells.add(_add(center, emit_dir, 2))
    return cells


def _all_in_bounds(shape: tuple[int, int], cells: set[tuple[int, int]]) -> bool:
    return all(in_bounds(shape, r, c) for r, c in cells)


def _clear_with_buffer(
    shape: tuple[int, int],
    cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> bool:
    if not _all_in_bounds(shape, cells):
        return False
    h, w = map(int, shape)
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) in occupied:
                    return False
    return True


def _parse_launcher_component(comp: dict) -> dict:
    cells = _component_set(comp["coords"])
    assert len(cells) == 4

    centers = [rc for rc in cells if _neighbor_degree(cells, rc) == 3]
    assert len(centers) == 1
    center = centers[0]

    offsets = [(r - center[0], c - center[1]) for r, c in cells if (r, c) != center]
    bar_pair = None
    control_offset = None
    for i in range(3):
        for j in range(i + 1, 3):
            if offsets[i][0] == -offsets[j][0] and offsets[i][1] == -offsets[j][1]:
                bar_pair = (offsets[i], offsets[j])
                control_offset = next(offsets[k] for k in range(3) if k not in {i, j})
                break
        if bar_pair is not None:
            break

    assert bar_pair is not None and control_offset is not None
    assert control_offset in DIRS
    emit_dir = (-control_offset[0], -control_offset[1])

    return {
        "center": center,
        "emit_dir": emit_dir,
        "cells": cells,
        "color": int(comp["color"]),
    }


def _parse_nose_object(comp: dict) -> dict:
    cells = _component_set(comp["coords"])
    assert len(cells) == 10

    nose_cells = [rc for rc in cells if _neighbor_degree(cells, rc) == 1]
    assert len(nose_cells) == 1
    nose = nose_cells[0]

    square = cells - {nose}
    rows = [r for r, _c in square]
    cols = [c for _r, c in square]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert r1 - r0 == 2 and c1 - c0 == 2
    expected_square = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    assert square == expected_square

    center = (r0 + 1, c0 + 1)
    dr = nose[0] - center[0]
    dc = nose[1] - center[1]
    assert (dr, dc) in {(-2, 0), (2, 0), (0, -2), (0, 2)}
    emit_dir = (dr // 2, dc // 2)

    return {
        "center": center,
        "emit_dir": emit_dir,
        "cells": cells,
        "color": int(comp["color"]),
    }


def _parse_scene(grid: np.ndarray) -> tuple[int, dict, list[dict]]:
    bg = get_bg(grid)
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    launcher = None
    objects: list[dict] = []
    for comp in comps:
        area = int(comp["area"])
        if area == 4:
            parsed = _parse_launcher_component(comp)
            assert launcher is None, "expected exactly one launcher"
            launcher = parsed
        elif area == 10:
            objects.append(_parse_nose_object(comp))
        else:
            raise AssertionError("unexpected foreground component")

    assert launcher is not None, "missing launcher"
    return int(bg), launcher, objects


def _simulate_hits(
    shape: tuple[int, int],
    source: tuple[int, int],
    emit_dir: tuple[int, int],
    objects: list[dict],
) -> list[int]:
    cell_to_object: dict[tuple[int, int], int] = {}
    for obj_id, obj in enumerate(objects):
        for rc in obj["cells"]:
            cell_to_object[rc] = obj_id

    alive = set(range(len(objects)))
    hit_order: list[int] = []
    current_source = source
    current_dir = emit_dir

    while True:
        r, c = current_source
        dr, dc = current_dir
        hit = None
        while True:
            r += dr
            c += dc
            if not in_bounds(shape, r, c):
                break
            obj_id = cell_to_object.get((r, c))
            if obj_id is not None and obj_id in alive:
                hit = obj_id
                break
        if hit is None:
            break

        alive.remove(hit)
        hit_order.append(hit)
        current_source = objects[hit]["center"]
        current_dir = objects[hit]["emit_dir"]

    return hit_order


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Initial ray hits only the first object; that object is erased and relays the next ray using its own nose direction.

    Input structure:
    - Black background.
    - One gray launcher: a 3-pixel straight bar plus a same-color control pixel touching the bar midpoint.
    - Several same-color objects: each is a solid 3x3 square with one extra nose pixel centered just outside one side.

    Rule:
    - Fire from the launcher center in the direction away from its control pixel.
    - Each ray erases only the first still-present object it meets.
    - The erased object's center becomes the next source, and its nose direction becomes the next ray direction.
    - Stop when a ray exits the grid without hitting another object.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"

    bg, launcher, objects = _parse_scene(grid)
    erased = set(_simulate_hits(grid.shape, launcher["center"], launcher["emit_dir"], objects))

    out = grid.copy()
    for obj_id in erased:
        for r, c in objects[obj_id]["cells"]:
            out[r, c] = bg
    return out


def _render_scene(shape: tuple[int, int], launcher: dict, objects: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for r, c in launcher["cells"]:
        grid[r, c] = launcher["color"]
    for obj in objects:
        for r, c in obj["cells"]:
            grid[r, c] = obj["color"]
    return grid


def _turn_choices(d: tuple[int, int]) -> list[tuple[int, int]]:
    return [d, _left(d), _right(d)]


def _build_chain(shape: tuple[int, int]) -> tuple[dict, list[dict]] | None:
    h, w = map(int, shape)
    for _ in range(LOCAL_ATTEMPTS):
        launch_dir = random.choice(DIRS)
        launcher_center = (random.randint(2, h - 3), random.randint(2, w - 3))
        launcher = {
            "center": launcher_center,
            "emit_dir": launch_dir,
            "cells": _launcher_cells(launcher_center, launch_dir),
            "color": LAUNCHER_COLOR,
        }
        if not _all_in_bounds(shape, launcher["cells"]):
            continue

        chain_len = random.randint(*CHAIN_LEN_RANGE)
        occupied = set(launcher["cells"])
        objects: list[dict] = []
        source = launcher_center
        current_dir = launch_dir

        ok = True
        for i in range(chain_len):
            candidates: list[dict] = []
            nose_choices = DIRS if i == chain_len - 1 else _turn_choices(current_dir)
            for dist in range(STEP_RANGE[0], STEP_RANGE[1] + 1):
                center = _add(source, current_dir, dist)
                for emit_dir in nose_choices:
                    cells = _object_cells(center, emit_dir)
                    if not _clear_with_buffer(shape, cells, occupied):
                        continue
                    candidates.append(
                        {
                            "center": center,
                            "emit_dir": emit_dir,
                            "cells": cells,
                            "color": random.choice(OBJECT_COLORS),
                        }
                    )
            if not candidates:
                ok = False
                break

            chosen = random.choice(candidates)
            objects.append(chosen)
            occupied |= chosen["cells"]
            source = chosen["center"]
            current_dir = chosen["emit_dir"]

        if not ok:
            continue

        expected = list(range(chain_len))
        if _simulate_hits(shape, launcher_center, launch_dir, objects) != expected:
            continue
        return launcher, objects

    return None


def _add_decoys(shape: tuple[int, int], launcher: dict, chain_objects: list[dict]) -> list[dict]:
    objects = list(chain_objects)
    occupied = set(launcher["cells"])
    for obj in objects:
        occupied |= obj["cells"]

    target_prefix = list(range(len(chain_objects)))
    for _ in range(random.randint(*DECOY_COUNT_RANGE)):
        for _ in range(LOCAL_ATTEMPTS):
            center = (
                random.randint(2, shape[0] - 3),
                random.randint(2, shape[1] - 3),
            )
            emit_dir = random.choice(DIRS)
            cells = _object_cells(center, emit_dir)
            if not _clear_with_buffer(shape, cells, occupied):
                continue

            candidate = {
                "center": center,
                "emit_dir": emit_dir,
                "cells": cells,
                "color": random.choice(OBJECT_COLORS),
            }
            trial = objects + [candidate]
            if _simulate_hits(shape, launcher["center"], launcher["emit_dir"], trial) != target_prefix:
                continue

            objects.append(candidate)
            occupied |= candidate["cells"]
            break

    return objects


def _expected_output(shape: tuple[int, int], launcher: dict, objects: list[dict], chain_len: int) -> np.ndarray:
    out = _render_scene(shape, launcher, objects)
    for obj in objects[:chain_len]:
        for r, c in obj["cells"]:
            out[r, c] = BG
    return out


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_H_RANGE),
        random.randint(*GRID_W_RANGE),
    )
    chain = _build_chain(shape)
    if chain is None:
        return None

    launcher, chain_objects = chain
    objects = _add_decoys(shape, launcher, chain_objects)
    grid = _render_scene(shape, launcher, objects)
    out = main(grid)
    expected = _expected_output(shape, launcher, objects, len(chain_objects))

    if grid.shape[0] > 30 or grid.shape[1] > 30:
        return None
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chained-ray input")
