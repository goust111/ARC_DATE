from __future__ import annotations

import random

import numpy as np


BG = 0
SEED = 4
PETAL = 6
FRAME_COLORS = (1, 2, 3, 5, 7, 8, 9)

GEN_RETRIES = 400
GRID_MAX = 32
GRID_MIN_BY_GROUP_COUNT = {1: 18, 2: 22, 3: 26}
GROUP_COUNT_CHOICES = (1, 2, 3)
DEPTH_CHOICES = (1, 2, 3)
INNER_SIZE_MIN = 3
INNER_SIZE_MAX = 9
PLACEMENT_ATTEMPTS = 120
GROUP_GAP = 2


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _color_components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or color in (BG, SEED):
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            components.append({"color": color, "coords": np.asarray(coords, dtype=int)})
    return components


def _frame_from_component(component: dict[str, object]) -> dict[str, int] | None:
    coords = np.asarray(component["coords"], dtype=int)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return None
    expected_size = 2 * height + 2 * width - 4
    if int(coords.shape[0]) != expected_size:
        return None

    pts = {(int(r), int(c)) for r, c in coords}
    for r in range(r0, r1 + 1):
        if (r, c0) not in pts or (r, c1) not in pts:
            return None
    for c in range(c0, c1 + 1):
        if (r0, c) not in pts or (r1, c) not in pts:
            return None
    for r, c in pts:
        if r not in (r0, r1) and c not in (c0, c1):
            return None

    return {
        "color": int(component["color"]),
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "area": height * width,
    }


def _find_frames(grid: np.ndarray) -> list[dict[str, int]]:
    frames: list[dict[str, int]] = []
    for component in _color_components(grid):
        frame = _frame_from_component(component)
        if frame is not None:
            frames.append(frame)
    return frames


def _containing_frames(frames: list[dict[str, int]], row: int, col: int) -> list[dict[str, int]]:
    return [
        frame
        for frame in frames
        if int(frame["r0"]) < int(row) < int(frame["r1"]) and int(frame["c0"]) < int(col) < int(frame["c1"])
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each yellow seed into a magenta cross whose arm length equals the nesting depth
    of the rectangular hollow frames that contain it.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _find_frames(g)
    seeds = np.argwhere(g == SEED)

    for seed_row, seed_col in seeds:
        containing = _containing_frames(frames, int(seed_row), int(seed_col))
        if not containing:
            continue

        depth = len(containing)
        inner = min(containing, key=lambda frame: (int(frame["area"]), -int(frame["r0"]), -int(frame["c0"])))
        min_row = int(inner["r0"]) + 1
        max_row = int(inner["r1"]) - 1
        min_col = int(inner["c0"]) + 1
        max_col = int(inner["c1"]) - 1

        for step in range(1, depth + 1):
            candidates = (
                (int(seed_row) - step, int(seed_col)),
                (int(seed_row) + step, int(seed_col)),
                (int(seed_row), int(seed_col) - step),
                (int(seed_row), int(seed_col) + step),
            )
            for rr, cc in candidates:
                if min_row <= rr <= max_row and min_col <= cc <= max_col:
                    out[rr, cc] = PETAL

    out[g == SEED] = SEED
    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _bbox_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return not (
        ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0
    )


def _choose_inner_size(depth: int, limit: int) -> int:
    preferred_min = min(max(INNER_SIZE_MIN, 2 * depth + 1), limit)
    if random.random() < 0.5 and preferred_min <= limit:
        low = preferred_min
    else:
        low = INNER_SIZE_MIN
    return random.randint(low, limit)


def _distance_to_inner_wall(
    inner_bbox: tuple[int, int, int, int],
    pos: tuple[int, int],
) -> int:
    r0, c0, r1, c1 = map(int, inner_bbox)
    r, c = map(int, pos)
    min_row = r0 + 1
    max_row = r1 - 1
    min_col = c0 + 1
    max_col = c1 - 1
    return min(r - min_row, max_row - r, c - min_col, max_col - c)


def _pick_seed_positions(
    inner_bbox: tuple[int, int, int, int],
    depth: int,
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = map(int, inner_bbox)
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    if not cells:
        return None

    full = [pos for pos in cells if _distance_to_inner_wall(inner_bbox, pos) >= depth]
    truncated = [pos for pos in cells if _distance_to_inner_wall(inner_bbox, pos) < depth]
    max_seed_count = min(3, len(cells))
    target_count = random.randint(1, max_seed_count)

    wanted_types: list[str] = []
    if full and truncated:
        first = random.choice(("full", "truncated"))
        wanted_types.append(first)
        if target_count >= 2:
            wanted_types.append("truncated" if first == "full" else "full")
        while len(wanted_types) < target_count:
            wanted_types.append(random.choice(("full", "truncated")))
    elif full:
        wanted_types = ["full"] * target_count
    else:
        wanted_types = ["truncated"] * target_count

    def compatible(candidate: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
        return all(max(abs(candidate[0] - r), abs(candidate[1] - c)) > 1 for r, c in chosen)

    chosen: list[tuple[int, int]] = []
    for kind in wanted_types:
        primary = full if kind == "full" else truncated
        secondary = truncated if kind == "full" else full
        for pool in (primary, secondary, cells):
            options = [pos for pos in pool if compatible(pos, chosen)]
            if options:
                chosen.append(random.choice(options))
                break

    if not chosen:
        return None
    return chosen


def _make_group_blueprint(height: int, width: int) -> dict[str, object] | None:
    depth = random.choice(DEPTH_CHOICES)
    outer_extra = 4 * depth - 2
    max_inner_h = min(INNER_SIZE_MAX, height - outer_extra - 2)
    max_inner_w = min(INNER_SIZE_MAX, width - outer_extra - 2)
    if max_inner_h < INNER_SIZE_MIN or max_inner_w < INNER_SIZE_MIN:
        return None

    inner_h = _choose_inner_size(depth, max_inner_h)
    inner_w = _choose_inner_size(depth, max_inner_w)
    outer_h = inner_h + outer_extra
    outer_w = inner_w + outer_extra
    colors = random.sample(FRAME_COLORS, depth)

    return {
        "depth": depth,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "colors": colors,
    }


def _place_group(
    grid: np.ndarray,
    placed_bboxes: list[tuple[int, int, int, int]],
    blueprint: dict[str, object],
) -> dict[str, object] | None:
    h, w = grid.shape
    outer_h = int(blueprint["outer_h"])
    outer_w = int(blueprint["outer_w"])

    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = random.randint(1, h - outer_h - 1)
        c0 = random.randint(1, w - outer_w - 1)
        outer_bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
        if any(_bbox_overlaps(outer_bbox, other, GROUP_GAP) for other in placed_bboxes):
            continue

        depth = int(blueprint["depth"])
        colors = list(blueprint["colors"])
        for layer in range(depth):
            offset = 2 * layer
            bbox = (
                r0 + offset,
                c0 + offset,
                r0 + outer_h - 1 - offset,
                c0 + outer_w - 1 - offset,
            )
            _draw_frame(grid, bbox, int(colors[layer]))

        inner_bbox = (
            r0 + 2 * (depth - 1),
            c0 + 2 * (depth - 1),
            r0 + outer_h - 1 - 2 * (depth - 1),
            c0 + outer_w - 1 - 2 * (depth - 1),
        )
        seeds = _pick_seed_positions(inner_bbox, depth)
        if not seeds:
            grid[outer_bbox[0] : outer_bbox[2] + 1, outer_bbox[1] : outer_bbox[3] + 1] = BG
            continue
        for sr, sc in seeds:
            grid[int(sr), int(sc)] = SEED

        return {
            "depth": depth,
            "outer_bbox": outer_bbox,
            "inner_bbox": inner_bbox,
            "seeds": seeds,
        }

    return None


def _expected_output(grid: np.ndarray, groups: list[dict[str, object]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for group in groups:
        depth = int(group["depth"])
        r0, c0, r1, c1 = map(int, group["inner_bbox"])
        min_row = r0 + 1
        max_row = r1 - 1
        min_col = c0 + 1
        max_col = c1 - 1
        for seed_row, seed_col in group["seeds"]:
            for step in range(1, depth + 1):
                for rr, cc in (
                    (int(seed_row) - step, int(seed_col)),
                    (int(seed_row) + step, int(seed_col)),
                    (int(seed_row), int(seed_col) - step),
                    (int(seed_row), int(seed_col) + step),
                ):
                    if min_row <= rr <= max_row and min_col <= cc <= max_col:
                        out[rr, cc] = PETAL
    out[grid == SEED] = SEED
    return out


def _validate_generated_grid(grid: np.ndarray, groups: list[dict[str, object]]) -> bool:
    frames = _find_frames(grid)
    seeds = [tuple(map(int, pos)) for pos in np.argwhere(grid == SEED)]
    expected_seed_count = sum(len(group["seeds"]) for group in groups)
    expected_frame_count = sum(int(group["depth"]) for group in groups)

    if len(seeds) != expected_seed_count or len(frames) != expected_frame_count:
        return False

    frame_cells = set()
    for frame in frames:
        r0, c0, r1, c1 = map(int, (frame["r0"], frame["c0"], frame["r1"], frame["c1"]))
        for r in range(r0, r1 + 1):
            frame_cells.add((r, c0))
            frame_cells.add((r, c1))
        for c in range(c0, c1 + 1):
            frame_cells.add((r0, c))
            frame_cells.add((r1, c))

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            value = int(grid[r, c])
            if value not in (BG, SEED) and (r, c) not in frame_cells:
                return False

    frame_by_bbox = {
        (int(frame["r0"]), int(frame["c0"]), int(frame["r1"]), int(frame["c1"])): frame for frame in frames
    }
    for group in groups:
        depth = int(group["depth"])
        inner_bbox = tuple(map(int, group["inner_bbox"]))
        if inner_bbox not in frame_by_bbox:
            return False
        for seed in group["seeds"]:
            containing = _containing_frames(frames, int(seed[0]), int(seed[1]))
            if len(containing) != depth:
                return False
            innermost = min(containing, key=lambda frame: int(frame["area"]))
            bbox = (int(innermost["r0"]), int(innermost["c0"]), int(innermost["r1"]), int(innermost["c1"]))
            if bbox != inner_bbox:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    group_count = random.choice(GROUP_COUNT_CHOICES)
    min_size = GRID_MIN_BY_GROUP_COUNT[group_count]
    height = random.randint(min_size, GRID_MAX)
    width = random.randint(min_size, GRID_MAX)

    grid = np.full((height, width), BG, dtype=int)
    groups: list[dict[str, object]] = []
    placed_bboxes: list[tuple[int, int, int, int]] = []

    blueprints = []
    for _ in range(group_count):
        blueprint = _make_group_blueprint(height, width)
        if blueprint is None:
            return None
        blueprints.append(blueprint)
    blueprints.sort(key=lambda item: int(item["outer_h"]) * int(item["outer_w"]), reverse=True)

    for blueprint in blueprints:
        group = _place_group(grid, placed_bboxes, blueprint)
        if group is None:
            return None
        groups.append(group)
        placed_bboxes.append(tuple(map(int, group["outer_bbox"])))

    if not _validate_generated_grid(grid, groups):
        return None

    expected = _expected_output(grid, groups)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame blossom input")
