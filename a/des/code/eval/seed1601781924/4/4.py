from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 8
CONFLICT = 2
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
OBJECT_COLORS = [0, 1, 3, 4, 5, 6, 7, 9]

# Interpret each 3-pixel arrow as a 2x2 L triomino whose ray exits through
# the missing corner, along that corner's diagonal direction.
ARROW_SPECS = (
    {
        "cells_rel": ((0, 1), (1, 0), (1, 1)),
        "start_rel": (1, 1),
        "dir": (-1, -1),
    },
    {
        "cells_rel": ((0, 0), (1, 0), (1, 1)),
        "start_rel": (1, 0),
        "dir": (-1, 1),
    },
    {
        "cells_rel": ((0, 0), (0, 1), (1, 1)),
        "start_rel": (0, 1),
        "dir": (1, -1),
    },
    {
        "cells_rel": ((0, 0), (0, 1), (1, 0)),
        "start_rel": (0, 0),
        "dir": (1, 1),
    },
)

GENERATE_RETRIES = 600
GRID_H_RANGE = (12, 18)
GRID_W_RANGE = (12, 18)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components_non_bg(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(int(r), int(c))])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        _in_bounds(g.shape, nr, nc)
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _arrow_from_component(comp: dict) -> dict | None:
    coords = comp["coords"]
    if int(comp["area"]) != 3:
        return None
    r0, c0, r1, c1 = map(int, comp["bbox"])
    if r1 - r0 != 1 or c1 - c0 != 1:
        return None

    rel = {(int(r - r0), int(c - c0)) for r, c in coords}
    for spec in ARROW_SPECS:
        if rel != set(spec["cells_rel"]):
            continue
        cells = tuple(
            (int(r0 + rr), int(c0 + cc)) for rr, cc in spec["cells_rel"]
        )
        start = (int(r0 + spec["start_rel"][0]), int(c0 + spec["start_rel"][1]))
        return {
            "color": int(comp["color"]),
            "cells": cells,
            "start": start,
            "dir": tuple(spec["dir"]),
            "bbox": (r0, c0, r1, c1),
        }
    return None


def _parse_objects(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    seeds: list[dict] = []
    arrows: list[dict] = []

    for comp in _components_non_bg(grid):
        if int(comp["area"]) == 1:
            r, c = map(int, comp["coords"][0])
            seeds.append({"cell": (r, c), "color": int(comp["color"])})
            continue

        arrow = _arrow_from_component(comp)
        if arrow is None:
            raise AssertionError("unexpected non-background object")
        arrows.append(arrow)

    seeds.sort(key=lambda seed: seed["cell"])
    arrows.sort(key=lambda arrow: arrow["bbox"])
    return seeds, arrows


def _cardinal_param(
    start: tuple[int, int], delta: tuple[int, int], cell: tuple[int, int]
) -> int | None:
    dr, dc = map(int, delta)
    sr, sc = map(int, start)
    cr, cc = map(int, cell)
    if dr != 0:
        if cc != sc:
            return None
        diff = cr - sr
        if diff * dr <= 0:
            return None
        return int(diff * dr)
    if cr != sr:
        return None
    diff = cc - sc
    if diff * dc <= 0:
        return None
    return int(diff * dc)


def _diag_param(
    start: tuple[int, int], delta: tuple[int, int], cell: tuple[int, int]
) -> int | None:
    dr, dc = map(int, delta)
    sr, sc = map(int, start)
    cr, cc = map(int, cell)
    row_steps = (cr - sr) * dr
    col_steps = (cc - sc) * dc
    if row_steps <= 0 or col_steps <= 0 or row_steps != col_steps:
        return None
    return int(row_steps)


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    seeds, arrows = _parse_objects(g)

    solid = g != BG
    arrow_at: dict[tuple[int, int], int] = {}
    for arrow_id, arrow in enumerate(arrows):
        for cell in arrow["cells"]:
            arrow_at[cell] = int(arrow_id)

    out = g.copy()
    activated_by = [set() for _ in arrows]
    queue: deque[tuple[int, int, tuple[int, int], tuple[int, int]]] = deque()

    for seed_id, seed in enumerate(seeds):
        for delta in DIRS4:
            queue.append((int(seed_id), int(seed["color"]), seed["cell"], delta))

    while queue:
        seed_id, color, start, delta = queue.popleft()
        dr, dc = map(int, delta)
        r, c = map(int, start)
        while True:
            r += dr
            c += dc
            if not _in_bounds(g.shape, r, c):
                break

            arrow_id = arrow_at.get((int(r), int(c)))
            if arrow_id is not None:
                if seed_id not in activated_by[arrow_id]:
                    activated_by[arrow_id].add(int(seed_id))
                    arrow = arrows[arrow_id]
                    queue.append(
                        (
                            int(seed_id),
                            int(color),
                            arrow["start"],
                            tuple(arrow["dir"]),
                        )
                    )
                break

            if bool(solid[r, c]):
                break

            out[r, c] = int(color)

    for arrow_id, sources in enumerate(activated_by):
        if len(sources) <= 1:
            continue
        for r, c in arrows[arrow_id]["cells"]:
            out[r, c] = CONFLICT

    return out, {
        "seeds": seeds,
        "arrows": arrows,
        "activated_by": activated_by,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Broadcast each seed in the four cardinal directions. Whenever a ray hits a
    3-pixel arrow marker, activate that arrow and emit a new diagonal ray in the
    arrow's encoded direction. Arrows hit by multiple distinct seeds turn red.
    """
    out, _meta = _simulate(np.asarray(grid, dtype=int))
    return out


def _empty_state(h: int, w: int) -> dict:
    return {
        "shape": (int(h), int(w)),
        "occupied": set(),
        "reserved": set(),
        "seeds": [],
        "arrows": [],
    }


def _can_place_cells(state: dict, cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> bool:
    shape = state["shape"]
    cell_set = {(int(r), int(c)) for r, c in cells}
    if not cell_set:
        return False
    if len(cell_set) != len(tuple(cells)):
        return False
    for r, c in cell_set:
        if not _in_bounds(shape, r, c):
            return False
        if (r, c) in state["occupied"] or (r, c) in state["reserved"]:
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in state["occupied"] and (nr, nc) not in cell_set:
                return False
    return True


def _can_reserve_path(state: dict, path: list[tuple[int, int]]) -> bool:
    shape = state["shape"]
    for r, c in path:
        if not _in_bounds(shape, r, c):
            return False
        if (r, c) in state["occupied"] or (r, c) in state["reserved"]:
            return False
    return True


def _reserve_path(state: dict, path: list[tuple[int, int]]) -> None:
    for cell in path:
        state["reserved"].add((int(cell[0]), int(cell[1])))


def _place_arrow(state: dict, top_left: tuple[int, int], spec: dict) -> dict | None:
    r0, c0 = map(int, top_left)
    cells = [(int(r0 + rr), int(c0 + cc)) for rr, cc in spec["cells_rel"]]
    if not _can_place_cells(state, cells):
        return None

    arrow = {
        "top_left": (r0, c0),
        "cells": tuple(cells),
        "start": (int(r0 + spec["start_rel"][0]), int(c0 + spec["start_rel"][1])),
        "dir": tuple(spec["dir"]),
    }
    state["arrows"].append(arrow)
    state["occupied"].update(arrow["cells"])
    return arrow


def _place_seed(
    state: dict, cell: tuple[int, int], hit_dir: tuple[int, int] | None = None
) -> dict | None:
    rc = (int(cell[0]), int(cell[1]))
    if not _can_place_cells(state, [rc]):
        return None
    seed = {"cell": rc, "hit_dir": hit_dir}
    state["seeds"].append(seed)
    state["occupied"].add(rc)
    return seed


def _shuffled(items):
    items = list(items)
    random.shuffle(items)
    return items


def _try_place_random_arrow(state: dict, margin: int = 1) -> dict | None:
    h, w = state["shape"]
    for _ in range(160):
        spec = random.choice(ARROW_SPECS)
        if h - 2 - 2 * margin < 0 or w - 2 - 2 * margin < 0:
            return None
        r0 = random.randint(margin, h - 2 - margin)
        c0 = random.randint(margin, w - 2 - margin)
        arrow = _place_arrow(state, (r0, c0), spec)
        if arrow is not None:
            return arrow
    return None


def _try_place_seed_for_target_arrow(
    state: dict,
    target_arrow: dict,
    allowed_dirs: tuple[tuple[int, int], ...] = DIRS4,
    min_steps: int = 2,
    max_steps: int = 7,
) -> dict | None:
    target_cells = _shuffled(target_arrow["cells"])
    for delta in _shuffled(allowed_dirs):
        dr, dc = map(int, delta)
        for cell in target_cells:
            for steps in _shuffled(range(min_steps, max_steps + 1)):
                seed = (int(cell[0] - steps * dr), int(cell[1] - steps * dc))
                path = [
                    (int(seed[0] + dr * k), int(seed[1] + dc * k))
                    for k in range(1, steps)
                ]
                if not _can_place_cells(state, [seed]):
                    continue
                if not _can_reserve_path(state, path):
                    continue
                first_hit = min(
                    (
                        t
                        for t in (
                            _cardinal_param(seed, delta, other)
                            for other in target_arrow["cells"]
                        )
                        if t is not None
                    ),
                    default=None,
                )
                if first_hit != steps:
                    continue
                placed = _place_seed(state, seed, hit_dir=delta)
                if placed is None:
                    continue
                _reserve_path(state, path)
                return placed
    return None


def _try_place_source_arrow_for_target(
    state: dict,
    target_arrow: dict,
    min_steps: int = 2,
    max_steps: int = 7,
) -> dict | None:
    target_cells = _shuffled(target_arrow["cells"])
    for spec in _shuffled(ARROW_SPECS):
        dr, dc = map(int, spec["dir"])
        for target_cell in target_cells:
            for steps in _shuffled(range(min_steps, max_steps + 1)):
                start = (
                    int(target_cell[0] - steps * dr),
                    int(target_cell[1] - steps * dc),
                )
                top_left = (
                    int(start[0] - spec["start_rel"][0]),
                    int(start[1] - spec["start_rel"][1]),
                )
                cells = [
                    (int(top_left[0] + rr), int(top_left[1] + cc))
                    for rr, cc in spec["cells_rel"]
                ]
                candidate_set = set(cells)
                path = [
                    (int(start[0] + dr * k), int(start[1] + dc * k))
                    for k in range(1, steps)
                ]
                if not _can_place_cells(state, cells):
                    continue
                if any(cell in candidate_set for cell in path):
                    continue
                if not _can_reserve_path(state, path):
                    continue
                first_hit = min(
                    (
                        t
                        for t in (
                            _diag_param(start, spec["dir"], other)
                            for other in target_arrow["cells"]
                        )
                        if t is not None
                    ),
                    default=None,
                )
                if first_hit != steps:
                    continue
                arrow = _place_arrow(state, top_left, spec)
                if arrow is None:
                    continue
                _reserve_path(state, path)
                return arrow
    return None


def _render_state(state: dict) -> np.ndarray:
    h, w = state["shape"]
    grid = np.full((h, w), BG, dtype=int)

    pool = OBJECT_COLORS.copy()
    random.shuffle(pool)
    seed_colors = pool[: len(state["seeds"])]
    arrow_colors = pool[len(state["seeds"]) : len(state["seeds"]) + len(state["arrows"])]

    if len(arrow_colors) < len(state["arrows"]):
        rest = [c for c in OBJECT_COLORS if c not in seed_colors]
        arrow_colors = [rest[i % len(rest)] for i in range(len(state["arrows"]))]

    for seed, color in zip(state["seeds"], seed_colors):
        r, c = seed["cell"]
        grid[r, c] = int(color)

    for arrow, color in zip(state["arrows"], arrow_colors):
        for r, c in arrow["cells"]:
            grid[r, c] = int(color)

    return grid


def _features(grid: np.ndarray) -> dict:
    out, meta = _simulate(grid)
    activated_sizes = [len(srcs) for srcs in meta["activated_by"]]
    return {
        "out": out,
        "n_seeds": len(meta["seeds"]),
        "n_arrows": len(meta["arrows"]),
        "n_activated": sum(size > 0 for size in activated_sizes),
        "n_conflicts": sum(size > 1 for size in activated_sizes),
        "painted": int(np.sum((grid == BG) & (out != BG))),
    }


def _build_single_chain(state: dict) -> dict | None:
    chain_arrows = random.choice([2, 3, 4])
    current = _try_place_random_arrow(state, margin=2)
    if current is None:
        return None

    for _ in range(chain_arrows - 1):
        current = _try_place_source_arrow_for_target(state, current, min_steps=2, max_steps=6)
        if current is None:
            return None

    if _try_place_seed_for_target_arrow(state, current, min_steps=2, max_steps=7) is None:
        return None

    return {
        "expected_activated": chain_arrows,
        "expected_conflicts": 0,
    }


def _build_direct_conflict(state: dict) -> dict | None:
    tail_len = random.choice([0, 1, 2])
    current = _try_place_random_arrow(state, margin=2)
    if current is None:
        return None

    for _ in range(tail_len):
        current = _try_place_source_arrow_for_target(state, current, min_steps=2, max_steps=6)
        if current is None:
            return None

    first_seed = _try_place_seed_for_target_arrow(state, current, min_steps=2, max_steps=7)
    if first_seed is None:
        return None

    used_dir = first_seed["hit_dir"]
    remaining_dirs = tuple(delta for delta in DIRS4 if delta != used_dir)
    second_seed = _try_place_seed_for_target_arrow(
        state,
        current,
        allowed_dirs=remaining_dirs if remaining_dirs else DIRS4,
        min_steps=2,
        max_steps=7,
    )
    if second_seed is None:
        return None

    chain_size = tail_len + 1
    return {
        "expected_activated": chain_size,
        "expected_conflicts": chain_size,
    }


def _build_chain_conflict(state: dict) -> dict | None:
    tail_len = random.choice([0, 1])
    left_len = random.choice([1, 2])
    right_len = random.choice([0, 1])

    conflict_arrow = _try_place_random_arrow(state, margin=2)
    if conflict_arrow is None:
        return None

    current = conflict_arrow
    for _ in range(tail_len):
        current = _try_place_source_arrow_for_target(state, current, min_steps=2, max_steps=6)
        if current is None:
            return None
    conflict_arrow = current

    left_root = conflict_arrow
    for _ in range(left_len):
        left_root = _try_place_source_arrow_for_target(state, left_root, min_steps=2, max_steps=6)
        if left_root is None:
            return None
    if _try_place_seed_for_target_arrow(state, left_root, min_steps=2, max_steps=7) is None:
        return None

    if right_len == 0:
        if _try_place_seed_for_target_arrow(state, conflict_arrow, min_steps=2, max_steps=7) is None:
            return None
    else:
        right_root = conflict_arrow
        for _ in range(right_len):
            right_root = _try_place_source_arrow_for_target(state, right_root, min_steps=2, max_steps=6)
            if right_root is None:
                return None
        if _try_place_seed_for_target_arrow(state, right_root, min_steps=2, max_steps=7) is None:
            return None

    return {
        "expected_activated": (tail_len + 1) + left_len + right_len,
        "expected_conflicts": tail_len + 1,
    }


def _add_distractors(state: dict, count: int) -> bool:
    for _ in range(count):
        if _try_place_random_arrow(state, margin=1) is None:
            return False
    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    state = _empty_state(h, w)

    mode = random.choice(["single_chain", "direct_conflict", "chain_conflict"])
    if mode == "single_chain":
        expected = _build_single_chain(state)
    elif mode == "direct_conflict":
        expected = _build_direct_conflict(state)
    else:
        expected = _build_chain_conflict(state)
    if expected is None:
        return None

    distractor_count = random.choice([0, 0, 1, 1, 2])
    if distractor_count and not _add_distractors(state, distractor_count):
        return None

    grid = _render_state(state)
    feats = _features(grid)
    if feats["painted"] < 6:
        return None
    if feats["n_seeds"] != len(state["seeds"]):
        return None
    if feats["n_arrows"] != len(state["arrows"]):
        return None
    if feats["n_activated"] != expected["expected_activated"]:
        return None
    if feats["n_conflicts"] != expected["expected_conflicts"]:
        return None
    if feats["n_arrows"] != expected["expected_activated"] + distractor_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
