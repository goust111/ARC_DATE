from __future__ import annotations

import random

import numpy as np


BG = 0
PRISM = 4
RECEIVER = 5
TOWER_COLORS = (1, 2, 3, 6, 7, 8, 9)

HORIZONTAL = "horizontal"
VERTICAL = "vertical"
PLAN_TYPES = ("straight", "split", "restore")

GRID_H_RANGE = (13, 22)
GRID_W_RANGE = (13, 22)
EXTRA_PRISM_RANGE = (0, 3)
EXTRA_RECEIVER_RANGE = (0, 3)
TOWER_PLACEMENT_RETRIES = 180
GENERATE_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _ray_cells(
    start: tuple[int, int], step: tuple[int, int], shape: tuple[int, int]
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    r, c = start
    dr, dc = step
    while _in_bounds(shape, r, c):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _split_dirs(primary: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    if primary == (0, 1):
        return (-1, 1), (1, 1)
    if primary == (0, -1):
        return (-1, -1), (1, -1)
    if primary == (-1, 0):
        return (-1, -1), (-1, 1)
    if primary == (1, 0):
        return (1, -1), (1, 1)
    raise AssertionError("unsupported primary direction")


def _tower_dirs(orientation: str) -> tuple[tuple[int, int], tuple[int, int]]:
    if orientation == HORIZONTAL:
        return (0, -1), (0, 1)
    if orientation == VERTICAL:
        return (-1, 0), (1, 0)
    raise AssertionError("unsupported tower orientation")


def _tower_cells(center: tuple[int, int], orientation: str) -> list[tuple[int, int]]:
    r, c = center
    if orientation == HORIZONTAL:
        return [(r, c - 1), (r, c), (r, c + 1)]
    if orientation == VERTICAL:
        return [(r - 1, c), (r, c), (r + 1, c)]
    raise AssertionError("unsupported tower orientation")


def _sample_unique(cells: list[tuple[int, int]], low: int, high: int) -> list[tuple[int, int]]:
    if not cells:
        return []
    count = random.randint(low, min(high, len(cells)))
    return random.sample(cells, count)


def _find_towers(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    towers: list[dict[str, object]] = []
    special = {BG, PRISM, RECEIVER}

    for r0 in range(h):
        for c0 in range(w):
            color = int(g[r0, c0])
            if color in special or seen[r0, c0]:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (
                        _in_bounds((h, w), nr, nc)
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            assert len(coords) == 3, "every non-special component must be a 3-cell tower"
            rows = sorted(r for r, _ in coords)
            cols = sorted(c for _, c in coords)

            if rows[0] == rows[2] and cols == [cols[0], cols[0] + 1, cols[0] + 2]:
                orientation = HORIZONTAL
                center = (rows[0], cols[0] + 1)
            elif cols[0] == cols[2] and rows == [rows[0], rows[0] + 1, rows[0] + 2]:
                orientation = VERTICAL
                center = (rows[0] + 1, cols[0])
            else:
                raise AssertionError("tower components must be straight 3-cell bars")

            towers.append(
                {
                    "color": color,
                    "center": center,
                    "orientation": orientation,
                }
            )

    assert towers, "expected at least one signal tower"
    towers.sort(key=lambda tower: (*tower["center"], int(tower["color"])))
    return towers


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    towers = _find_towers(g)
    stats = {
        "painted_bg": 0,
        "recolored_receivers": 0,
        "prism_splits": 0,
        "prism_restores": 0,
    }

    stack: list[tuple[str, int, int, tuple[int, int], tuple[int, int], int]] = []
    for tower in towers:
        center = tuple(tower["center"])
        color = int(tower["color"])
        for primary in _tower_dirs(str(tower["orientation"])):
            sr = center[0] + 2 * primary[0]
            sc = center[1] + 2 * primary[1]
            stack.append(("card", sr, sc, primary, primary, color))

    while stack:
        mode, r, c, primary, step, color = stack.pop()
        dr, dc = step
        while _in_bounds(g.shape, r, c):
            cell = int(g[r, c])
            if cell == BG:
                if int(out[r, c]) == BG:
                    stats["painted_bg"] += 1
                out[r, c] = color
                r += dr
                c += dc
                continue

            if cell == RECEIVER:
                if int(out[r, c]) == RECEIVER:
                    stats["recolored_receivers"] += 1
                out[r, c] = color
                r += dr
                c += dc
                continue

            if cell == PRISM:
                if mode == "card":
                    stats["prism_splits"] += 1
                    for diag in _split_dirs(primary):
                        stack.append(
                            (
                                "diag",
                                r + diag[0],
                                c + diag[1],
                                primary,
                                diag,
                                color,
                            )
                        )
                else:
                    stats["prism_restores"] += 1
                    stack.append(
                        (
                            "card",
                            r + primary[0],
                            c + primary[1],
                            primary,
                            primary,
                            color,
                        )
                    )
                break

            break

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor receivers reached by tower rays that alternate between cardinal travel and prism-induced diagonal branches.

    Input structure:
    - Background is black.
    - Signal towers are isolated 3-cell bars in some non-special color.
      Horizontal towers emit left and right; vertical towers emit up and down.
    - Yellow pixels are prisms and grey pixels are receivers.

    Rule:
    - Each tower emits two rays in its orientation, starting just beyond the bar.
    - Cardinal rays paint background cells in the tower color.
    - A cardinal ray that reaches a prism stops there and splits into the two
      matching diagonal directions.
    - A diagonal ray that reaches a prism stops there and resumes the original
      cardinal direction that spawned that diagonal branch.
    - Rays recolor receivers, pass through them, and stop on boundaries or on
      any original non-special colored pixel.

    Key insight:
    - Prisms are alternators: cardinal travel becomes two diagonal branches, and
      each diagonal branch becomes cardinal travel again at the next prism.
    """
    out, _stats = _simulate(np.asarray(grid, dtype=int))
    return out


def _choose_plan_types() -> tuple[str, str]:
    while True:
        plan_types = (random.choice(PLAN_TYPES), random.choice(PLAN_TYPES))
        if plan_types != ("straight", "straight"):
            return plan_types


def _plan_direction(
    shape: tuple[int, int],
    center: tuple[int, int],
    primary: tuple[int, int],
    plan_type: str,
) -> dict[str, list[tuple[int, int]]] | None:
    start = (center[0] + 2 * primary[0], center[1] + 2 * primary[1])
    primary_cells = _ray_cells(start, primary, shape)
    if len(primary_cells) < 2:
        return None

    if plan_type == "straight":
        return {
            "prisms": [],
            "receivers": _sample_unique(primary_cells, 1, 2),
        }

    branches = _split_dirs(primary)
    split_candidates: list[tuple[int, tuple[int, int], list[list[tuple[int, int]]]]] = []
    for idx, prism1 in enumerate(primary_cells):
        diag_lists: list[list[tuple[int, int]]] = []
        ok = True
        for diag in branches:
            diag_start = (prism1[0] + diag[0], prism1[1] + diag[1])
            diag_cells = _ray_cells(diag_start, diag, shape)
            if not diag_cells:
                ok = False
                break
            diag_lists.append(diag_cells)
        if ok:
            split_candidates.append((idx, prism1, diag_lists))

    if not split_candidates:
        return None

    if plan_type == "split":
        idx, prism1, diag_lists = random.choice(split_candidates)
        receivers: set[tuple[int, int]] = set()
        prefix = primary_cells[:idx]
        if prefix and random.random() < 0.7:
            receivers.add(random.choice(prefix))
        for diag_cells in diag_lists:
            receivers.add(random.choice(diag_cells))
            if len(diag_cells) >= 3 and random.random() < 0.35:
                receivers.add(random.choice(diag_cells))
        return {
            "prisms": [prism1],
            "receivers": sorted(receivers),
        }

    restore_candidates: list[
        tuple[
            int,
            tuple[int, int],
            list[list[tuple[int, int]]],
            list[tuple[int, list[tuple[int, int]], list[list[tuple[int, int]]]]],
        ]
    ] = []
    for idx, prism1, diag_lists in split_candidates:
        options: list[tuple[int, list[tuple[int, int]], list[list[tuple[int, int]]]]] = []
        max_k = min(len(diag_cells) for diag_cells in diag_lists) - 1
        for k in range(max_k + 1):
            prism2_cells: list[tuple[int, int]] = []
            post_lists: list[list[tuple[int, int]]] = []
            ok = True
            for diag_cells in diag_lists:
                prism2 = diag_cells[k]
                post_start = (prism2[0] + primary[0], prism2[1] + primary[1])
                post_cells = _ray_cells(post_start, primary, shape)
                if not post_cells:
                    ok = False
                    break
                prism2_cells.append(prism2)
                post_lists.append(post_cells)
            if ok:
                options.append((k, prism2_cells, post_lists))
        if options:
            restore_candidates.append((idx, prism1, diag_lists, options))

    if not restore_candidates:
        return None

    idx, prism1, diag_lists, options = random.choice(restore_candidates)
    k, prism2_cells, post_lists = random.choice(options)
    receivers: set[tuple[int, int]] = set()
    prefix = primary_cells[:idx]
    if prefix and random.random() < 0.6:
        receivers.add(random.choice(prefix))
    for diag_cells, post_cells in zip(diag_lists, post_lists):
        diag_prefix = diag_cells[:k]
        if diag_prefix and random.random() < 0.7:
            receivers.add(random.choice(diag_prefix))
        receivers.add(random.choice(post_cells))
        if len(post_cells) >= 3 and random.random() < 0.35:
            receivers.add(random.choice(post_cells))
    return {
        "prisms": [prism1, *prism2_cells],
        "receivers": sorted(receivers),
    }


def _build_active_tower(
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, str]] | None:
    orientation = random.choice((HORIZONTAL, VERTICAL))
    color = random.choice(TOWER_COLORS)
    plan_types = _choose_plan_types()

    for _ in range(TOWER_PLACEMENT_RETRIES):
        center = (
            random.randint(2, shape[0] - 3),
            random.randint(2, shape[1] - 3),
        )
        body = _tower_cells(center, orientation)
        if not all(_in_bounds(shape, r, c) for r, c in body):
            continue

        trial = np.full(shape, BG, dtype=int)
        used = set(body)
        for r, c in body:
            trial[r, c] = color

        ok = True
        for primary, plan_type in zip(_tower_dirs(orientation), plan_types):
            plan = _plan_direction(shape, center, primary, plan_type)
            if plan is None:
                ok = False
                break

            for cell in plan["prisms"]:
                if cell in used:
                    ok = False
                    break
            if not ok:
                break
            for cell in plan["receivers"]:
                if cell in used:
                    ok = False
                    break
            if not ok:
                break

            for r, c in plan["prisms"]:
                trial[r, c] = PRISM
                used.add((r, c))
            for r, c in plan["receivers"]:
                trial[r, c] = RECEIVER
                used.add((r, c))

        if not ok:
            continue

        out, stats = _simulate(trial)
        if np.array_equal(out, trial):
            continue
        if stats["painted_bg"] < 2 or stats["recolored_receivers"] < 1:
            continue
        if stats["prism_splits"] < 1:
            continue
        if "restore" in plan_types and stats["prism_restores"] < 1:
            continue

        support = (trial != BG) | (out != trial)
        return trial, out, support, plan_types

    return None


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    h, w = mask.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = h - max(0, dr)
            src_c0 = max(0, -dc)
            src_c1 = w - max(0, dc)
            dst_r0 = max(0, dr)
            dst_r1 = h - max(0, -dr)
            dst_c0 = max(0, dc)
            dst_c1 = w - max(0, -dc)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _place_extras(grid: np.ndarray, support: np.ndarray) -> None:
    blocked = _dilate(support)
    free = [(int(r), int(c)) for r, c in np.argwhere((grid == BG) & ~blocked)]
    random.shuffle(free)

    n_prisms = random.randint(*EXTRA_PRISM_RANGE)
    n_receivers = random.randint(*EXTRA_RECEIVER_RANGE)

    for _ in range(min(n_prisms, len(free))):
        r, c = free.pop()
        grid[r, c] = PRISM
    for _ in range(min(n_receivers, len(free))):
        r, c = free.pop()
        grid[r, c] = RECEIVER


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        shape = (
            random.randint(*GRID_H_RANGE),
            random.randint(*GRID_W_RANGE),
        )
        built = _build_active_tower(shape)
        if built is None:
            continue

        grid, _out, support, plan_types = built
        final_grid = grid.copy()
        _place_extras(final_grid, support)

        final_out, stats = _simulate(final_grid)
        if np.array_equal(final_grid, final_out):
            continue
        if stats["painted_bg"] < 2 or stats["recolored_receivers"] < 1:
            continue
        if stats["prism_splits"] < 1:
            continue
        if "restore" in plan_types and stats["prism_restores"] < 1:
            continue
        return final_grid

    raise AssertionError("failed to generate a valid ARC instance")
