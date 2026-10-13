from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ARROW_COLORS = (RED, BLUE, GREEN)

DIR_TO_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ARROW_PATTERNS = {
    frozenset(((0, 0), (2, 0), (1, 1))): ("right", (1, 1)),
    frozenset(((0, 1), (2, 1), (1, 0))): ("left", (1, 0)),
    frozenset(((0, 0), (0, 2), (1, 1))): ("down", (1, 1)),
    frozenset(((1, 0), (1, 2), (0, 1))): ("up", (0, 1)),
}

SHAPE_RETRY_LIMIT = 200
TARGET_RETRY_LIMIT = 200
GENERATOR_RETRY_LIMIT = 400


def _neighbors(r: int, c: int, connectivity: int) -> tuple[tuple[int, int], ...]:
    if connectivity == 4:
        return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
    return (
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    )


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors(cr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _parse_arrow(comp: list[tuple[int, int]]) -> tuple[str, tuple[int, int], tuple[int, int], list[tuple[int, int]]]:
    assert len(comp) == 3, "each arrow must contain exactly three pixels"
    r0, c0, r1, c1 = _bbox(comp)
    pattern = frozenset((r - r0, c - c0) for r, c in comp)
    assert pattern in ARROW_PATTERNS, f"unrecognized arrow shape: {sorted(pattern)}"
    direction, head_rel = ARROW_PATTERNS[pattern]
    head = (r0 + head_rel[0], c0 + head_rel[1])
    return direction, DIR_TO_VEC[direction], head, [(int(r), int(c)) for r, c in comp]


def _find_targets(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    targets: list[dict] = []
    target_at: dict[tuple[int, int], int] = {}

    for comp in _components(grid == GREY, connectivity=4):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h == w and h in (3, 4), "targets must be 3x3 or 4x4 grey blocks"

        patch = np.full((h, w), BG, dtype=int)
        rows: dict[int, list[int]] = {}
        cols: dict[int, list[int]] = {}
        pixels = set()
        for r, c in comp:
            patch[r - r0, c - c0] = GREY
            rows.setdefault(int(r), []).append(int(c))
            cols.setdefault(int(c), []).append(int(r))
            pixels.add((int(r), int(c)))
            target_at[(int(r), int(c))] = len(targets)

        for values in rows.values():
            values.sort()
        for values in cols.values():
            values.sort()

        targets.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "patch": patch,
                "rows": rows,
                "cols": cols,
                "pixels": pixels,
            }
        )

    assert targets, "expected at least one grey target"
    return targets, target_at


def _ray_target(
    shape: tuple[int, int],
    target_at: dict[tuple[int, int], int],
    head: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[int, int]:
    h, w = shape
    r, c = head
    dr, dc = direction
    dist = 0
    while True:
        r += dr
        c += dc
        dist += 1
        if not (0 <= r < h and 0 <= c < w):
            break
        if (r, c) in target_at:
            return int(target_at[(r, c)]), int(dist)
    raise AssertionError(f"arrow at {head} does not point to any grey target")


def _find_arrows(grid: np.ndarray, target_at: dict[tuple[int, int], int]) -> list[dict]:
    arrows: list[dict] = []
    for color in ARROW_COLORS:
        for comp in _components(grid == color, connectivity=8):
            direction_name, direction, head, cells = _parse_arrow(comp)
            target_id, dist = _ray_target(grid.shape, target_at, head, direction)
            arrows.append(
                {
                    "color": int(color),
                    "direction_name": direction_name,
                    "direction": direction,
                    "head": head,
                    "cells": cells,
                    "target_id": int(target_id),
                    "distance": int(dist),
                }
            )
    assert arrows, "expected at least one arrow"
    return arrows


def _transform_patch(patch: np.ndarray, arrow_color: int) -> np.ndarray:
    if arrow_color == RED:
        return np.fliplr(patch)
    if arrow_color == BLUE:
        return np.rot90(patch, k=3)
    assert arrow_color == GREEN, "unexpected arrow color"
    out = patch.copy()
    border = np.zeros_like(out, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    out[border & (out != BG)] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Colored arrows point to grey target blocks. Arrow color selects the transform:
    red = horizontal flip, blue = rotate clockwise 90 degrees, green = recolor the
    outer ring to green. If multiple arrows hit the same block, apply them from the
    farthest arrow to the nearest, then erase all arrows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    targets, target_at = _find_targets(g)
    arrows = _find_arrows(g, target_at)

    out = g.copy()
    for arrow in arrows:
        for r, c in arrow["cells"]:
            out[r, c] = BG

    by_target: dict[int, list[dict]] = {idx: [] for idx in range(len(targets))}
    for arrow in arrows:
        by_target[arrow["target_id"]].append(arrow)

    for target_id, ops in by_target.items():
        if not ops:
            continue
        ops.sort(key=lambda item: (-item["distance"], item["head"][0], item["head"][1]))
        target = targets[target_id]
        r0, c0, r1, c1 = target["r0"], target["c0"], target["r1"], target["c1"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1].copy()
        for op in ops:
            patch = _transform_patch(patch, op["color"])
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[:] = BG
        region[patch != BG] = patch[patch != BG]

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    coords = int(mask.sum())
    return coords > 0 and len(_components(mask, connectivity=4)) == 1


def _interesting_shape(mask: np.ndarray) -> bool:
    if not bool(mask[1:-1, 1:-1].any()):
        return False
    if not bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
        return False
    if np.array_equal(mask, np.fliplr(mask)):
        return False
    for k in (1, 2, 3):
        if np.array_equal(mask, np.rot90(mask, k=k)):
            return False
    return _mask_connected(mask)


def _generate_shape(size: int) -> np.ndarray:
    cells = [(r, c) for r in range(size) for c in range(size)]
    low = 1 if size == 3 else 2
    high = 3 if size == 3 else 5
    for _ in range(SHAPE_RETRY_LIMIT):
        mask = np.ones((size, size), dtype=bool)
        remove_count = random.randint(low, high)
        for r, c in random.sample(cells, k=remove_count):
            mask[r, c] = False
        if _interesting_shape(mask):
            return mask
    raise RuntimeError("failed to generate a sufficiently asymmetric target shape")


def _boxes_far_apart(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_targets(height: int, width: int, n_targets: int) -> tuple[list[dict], np.ndarray]:
    targets: list[dict] = []
    gap = 4
    row_margin = 4
    col_margin = 4

    for tid in range(n_targets):
        placed = False
        for _ in range(TARGET_RETRY_LIMIT):
            size = random.choice((3, 4))
            if height - size - row_margin <= row_margin or width - size - col_margin <= col_margin:
                break
            r0 = random.randint(row_margin, height - size - row_margin)
            c0 = random.randint(col_margin, width - size - col_margin)
            bbox = (r0, c0, r0 + size - 1, c0 + size - 1)
            if all(_boxes_far_apart(bbox, other["bbox"], gap) for other in targets):
                mask = _generate_shape(size)
                pixels = {(r0 + r, c0 + c) for r, c in np.argwhere(mask)}
                rows: dict[int, list[int]] = {}
                cols: dict[int, list[int]] = {}
                for r, c in pixels:
                    rows.setdefault(int(r), []).append(int(c))
                    cols.setdefault(int(c), []).append(int(r))
                for values in rows.values():
                    values.sort()
                for values in cols.values():
                    values.sort()
                targets.append(
                    {
                        "id": tid,
                        "bbox": bbox,
                        "mask": mask,
                        "pixels": pixels,
                        "rows": rows,
                        "cols": cols,
                    }
                )
                placed = True
                break
        if not placed:
            raise RuntimeError("failed to place all targets")

    target_id_grid = np.full((height, width), -1, dtype=int)
    for target in targets:
        for r, c in target["pixels"]:
            target_id_grid[r, c] = int(target["id"])
    return targets, target_id_grid


def _arrow_cells(head: tuple[int, int], direction_name: str) -> list[tuple[int, int]]:
    r, c = head
    if direction_name == "right":
        return [(r - 1, c - 1), (r + 1, c - 1), (r, c)]
    if direction_name == "left":
        return [(r - 1, c + 1), (r + 1, c + 1), (r, c)]
    if direction_name == "down":
        return [(r - 1, c - 1), (r - 1, c + 1), (r, c)]
    assert direction_name == "up"
    return [(r + 1, c - 1), (r + 1, c + 1), (r, c)]


def _first_target_id(
    target_id_grid: np.ndarray, head: tuple[int, int], direction: tuple[int, int]
) -> tuple[int, int] | None:
    h, w = target_id_grid.shape
    r, c = head
    dr, dc = direction
    dist = 0
    while True:
        r += dr
        c += dc
        dist += 1
        if not (0 <= r < h and 0 <= c < w):
            return None
        target_id = int(target_id_grid[r, c])
        if target_id != -1:
            return target_id, int(dist)


def _arrow_halo_conflict(cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) in occupied:
                    return True
    return False


def _candidate_lines(target: dict, direction_name: str) -> list[tuple[int, int]]:
    if direction_name == "right":
        return [(row, min(cols)) for row, cols in target["rows"].items()]
    if direction_name == "left":
        return [(row, max(cols)) for row, cols in target["rows"].items()]
    if direction_name == "down":
        return [(col, min(rows)) for col, rows in target["cols"].items()]
    assert direction_name == "up"
    return [(col, max(rows)) for col, rows in target["cols"].items()]


def _build_arrow_candidate(
    target: dict,
    target_id_grid: np.ndarray,
    occupied_arrows: set[tuple[int, int]],
    direction_name: str,
    distance: int,
) -> dict | None:
    h, w = target_id_grid.shape
    tid = int(target["id"])
    direction = DIR_TO_VEC[direction_name]
    lines = _candidate_lines(target, direction_name)
    random.shuffle(lines)

    for axis_a, axis_b in lines:
        if direction_name == "right":
            head = (axis_a, axis_b - distance)
        elif direction_name == "left":
            head = (axis_a, axis_b + distance)
        elif direction_name == "down":
            head = (axis_b - distance, axis_a)
        else:
            head = (axis_b + distance, axis_a)

        cells = _arrow_cells(head, direction_name)
        if not all(0 <= r < h and 0 <= c < w for r, c in cells):
            continue
        if any(int(target_id_grid[r, c]) != -1 for r, c in cells):
            continue
        if any((r, c) in occupied_arrows for r, c in cells):
            continue
        if _arrow_halo_conflict(cells, occupied_arrows):
            continue

        first_hit = _first_target_id(target_id_grid, head, direction)
        if first_hit != (tid, distance):
            continue

        blocked = False
        for step in range(1, distance):
            r = head[0] + direction[0] * step
            c = head[1] + direction[1] * step
            if (r, c) in occupied_arrows:
                blocked = True
                break
        if blocked:
            continue

        return {
            "target_id": tid,
            "direction_name": direction_name,
            "direction": direction,
            "distance": int(distance),
            "head": head,
            "cells": cells,
        }

    return None


def _place_arrows(targets: list[dict], target_id_grid: np.ndarray) -> list[dict]:
    n_targets = len(targets)
    has_multi = n_targets == 2 or random.random() < 0.65

    specs: list[tuple[int, int]] = []
    if has_multi:
        special = random.randrange(n_targets)
        other_choices = [idx for idx in range(n_targets) if idx != special]
        other = random.choice(other_choices)
        specs.extend([(special, RED), (special, BLUE), (other, GREEN)])
        for idx in range(n_targets):
            if idx in (special, other):
                continue
            if random.random() < 0.45:
                specs.append((idx, random.choice(ARROW_COLORS)))
        if random.random() < 0.45:
            specs.append((special, random.choice(ARROW_COLORS)))
    else:
        chosen = random.sample(range(n_targets), k=3)
        specs.extend([(chosen[0], RED), (chosen[1], BLUE), (chosen[2], GREEN)])
        for idx in range(n_targets):
            if idx in chosen:
                continue
            if random.random() < 0.45:
                specs.append((idx, random.choice(ARROW_COLORS)))
        if random.random() < 0.35:
            specs.append((random.choice(chosen), random.choice(ARROW_COLORS)))

    counts = {idx: 0 for idx in range(n_targets)}
    for tid, _color in specs:
        counts[tid] += 1
    specs.sort(key=lambda item: (-counts[item[0]], item[1]))

    placed: list[dict] = []
    occupied_arrows: set[tuple[int, int]] = set()
    used_dirs: dict[int, set[str]] = {idx: set() for idx in range(n_targets)}
    used_dists: dict[int, set[int]] = {idx: set() for idx in range(n_targets)}

    for tid, color in specs:
        directions = list(DIR_TO_VEC)
        random.shuffle(directions)
        if counts[tid] > 1:
            directions = [name for name in directions if name not in used_dirs[tid]]
        distances = [d for d in range(2, 8) if d not in used_dists[tid]]
        random.shuffle(distances)

        candidate = None
        for direction_name in directions:
            for distance in distances:
                candidate = _build_arrow_candidate(
                    targets[tid], target_id_grid, occupied_arrows, direction_name, distance
                )
                if candidate is not None:
                    break
            if candidate is not None:
                break

        if candidate is None:
            raise RuntimeError("failed to place arrows without ambiguity")

        candidate["color"] = int(color)
        placed.append(candidate)
        used_dirs[tid].add(candidate["direction_name"])
        used_dists[tid].add(candidate["distance"])
        for cell in candidate["cells"]:
            occupied_arrows.add(cell)

    return placed


def _render_grid(height: int, width: int, targets: list[dict], arrows: list[dict]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for target in targets:
        for r, c in target["pixels"]:
            grid[r, c] = GREY
    for arrow in arrows:
        for r, c in arrow["cells"]:
            grid[r, c] = int(arrow["color"])
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATOR_RETRY_LIMIT):
        n_targets = random.randint(2, 4)
        height = random.randint(20, 28)
        width = random.randint(22, 30)

        if n_targets == 2:
            width += random.randint(0, 2)

        try:
            targets, target_id_grid = _place_targets(height, width, n_targets)
            arrows = _place_arrows(targets, target_id_grid)
        except RuntimeError:
            continue

        grid = _render_grid(height, width, targets, arrows)

        if not all(np.any(grid == color) for color in ARROW_COLORS):
            continue
        if not np.any(grid == GREY):
            continue

        try:
            out = main(grid)
        except Exception:
            continue

        if np.array_equal(out, grid):
            continue
        if int(np.sum(grid == RED)) == 0 or int(np.sum(grid == BLUE)) == 0 or int(np.sum(grid == GREEN)) == 0:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input grid")
