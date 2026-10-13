from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import itertools
import random

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE = tuple(range(10))

# Solver constants.
T_SHAPES = {
    ((0, 0), (0, 1), (0, 2), (1, 1)): ((1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)): ((-1, 0), (0, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 0)): ((0, 1), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 1)): ((0, -1), (1, 0)),
}

# Generator constants.
GRID_H_RANGE = (12, 20)
GRID_W_RANGE = (16, 26)
PAIR_COUNT_RANGE = (1, 3)
HIT_PATH_RANGE = (2, 8)
PAIR_RETRIES = 160
GENERATION_RETRIES = 500
ROW_GAP = 4

RIGHT_SHAPE = ((0, 0), (1, 0), (1, 1), (2, 0))
DOWN_SHAPE = ((0, 0), (0, 1), (0, 2), (1, 1))


@dataclass(frozen=True)
class TObject:
    color: int
    coords: tuple[tuple[int, int], ...]
    direction: tuple[int, int]
    stem_end: tuple[int, int]


def _get_bg(g: np.ndarray) -> int:
    colors, counts = np.unique(g, return_counts=True)
    order = np.argsort(counts)[::-1]
    bg = int(colors[order[0]])
    if len(order) > 1:
        assert int(counts[order[0]]) > int(counts[order[1]]), "background must be uniquely most common"
    return bg


def _find_components(g: np.ndarray, bg: int) -> list[dict]:
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[dict] = []

    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(g[r, c]) == bg:
                continue

            color = int(g[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            rows = [pt[0] for pt in coords]
            cols = [pt[1] for pt in coords]
            comps.append(
                {
                    "color": color,
                    "coords": tuple(sorted(coords)),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )

    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return comps


def _parse_objects(g: np.ndarray, bg: int) -> list[TObject]:
    comps = _find_components(g, bg)
    assert len(comps) >= 2, "expected multiple T objects"
    assert len({int(comp["color"]) for comp in comps}) == len(comps), "each T must have a unique color"

    objects: list[TObject] = []
    for comp in comps:
        assert int(comp["area"]) == 4, "each object must have four pixels"
        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        norm = tuple(sorted((r - r0, c - c0) for r, c in comp["coords"]))
        assert norm in T_SHAPES, f"unrecognized T shape: {norm}"
        direction, stem_local = T_SHAPES[norm]
        stem_end = (r0 + int(stem_local[0]), c0 + int(stem_local[1]))
        objects.append(
            TObject(
                color=int(comp["color"]),
                coords=tuple((int(r), int(c)) for r, c in comp["coords"]),
                direction=(int(direction[0]), int(direction[1])),
                stem_end=stem_end,
            )
        )

    return objects


def _analyze_instance(g: np.ndarray) -> tuple[int, list[TObject], dict[tuple[int, int], int], list[dict]]:
    bg = _get_bg(g)
    objects = _parse_objects(g, bg)
    H, W = g.shape

    pixel_to_obj: dict[tuple[int, int], int] = {}
    for idx, obj in enumerate(objects):
        for pt in obj.coords:
            assert pt not in pixel_to_obj, "objects must not overlap"
            pixel_to_obj[pt] = idx

    claims: dict[tuple[int, int], int] = {}
    rays: list[dict] = []
    for idx, obj in enumerate(objects):
        dr, dc = obj.direction
        rr, cc = obj.stem_end[0] + dr, obj.stem_end[1] + dc
        path: list[tuple[int, int]] = []

        while 0 <= rr < H and 0 <= cc < W:
            hit_idx = pixel_to_obj.get((rr, cc))
            if hit_idx is not None:
                target = objects[hit_idx]
                for step, pt in enumerate(path):
                    color = obj.color if step % 2 == 0 else target.color
                    previous = claims.get(pt)
                    assert previous in (None, color), "stripe paths must not conflict"
                    claims[pt] = color
                rays.append({"source": idx, "target": hit_idx, "path": tuple(path)})
                break

            assert int(g[rr, cc]) == bg, "ray must pass only through background before the hit"
            path.append((rr, cc))
            rr += dr
            cc += dc
        else:
            rays.append({"source": idx, "target": None, "path": tuple(path)})

    return bg, objects, claims, rays


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect oriented T-shaped signal towers and cast a ray from each tower's stem tip.

    Input structure:
    - The background is the most common color.
    - Every foreground object is a monochrome 4-pixel T tetromino with a unique color.
    - The extra stem cell indicates the tower's facing direction.

    Rule:
    - From each stem tip, cast a cardinal ray along the facing direction until it reaches another T pixel or exits the grid.
    - If the ray hits another object, fill the traversed background cells with alternating source and target colors, starting with the source color.
    - If the ray reaches the boundary first, draw nothing.
    - Keep all original T pixels unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg, _objects, claims, _rays = _analyze_instance(g)
    out = g.copy()
    for (r, c), color in claims.items():
        assert int(g[r, c]) == bg, "only background cells may be recolored"
        out[r, c] = int(color)
    return out


def _translate(shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> tuple[tuple[int, int], ...]:
    return tuple((r0 + r, c0 + c) for r, c in shape)


def _mark_buffer(mask: np.ndarray, coords: tuple[tuple[int, int], ...]) -> None:
    H, W = mask.shape
    for r, c in coords:
        r0 = max(0, int(r) - 1)
        r1 = min(H, int(r) + 2)
        c0 = max(0, int(c) - 1)
        c1 = min(W, int(c) + 2)
        mask[r0:r1, c0:c1] = True


def _paint_shape(grid: np.ndarray, coords: tuple[tuple[int, int], ...], color: int) -> None:
    for r, c in coords:
        grid[r, c] = int(color)


def _pair_row_choices(H: int, count: int) -> list[tuple[int, ...]]:
    tops = range(0, H - 2)
    return [
        combo
        for combo in itertools.combinations(tops, count)
        if all(combo[i + 1] - combo[i] >= ROW_GAP for i in range(len(combo) - 1))
    ]


def _place_pair(
    grid: np.ndarray,
    bg: int,
    source_color: int,
    target_color: int,
    top_row: int,
    buffer_mask: np.ndarray,
    clear_for_objects: np.ndarray,
    painted_path_mask: np.ndarray,
) -> bool:
    H, W = grid.shape
    max_path = min(HIT_PATH_RANGE[1], W - 5)
    if max_path < HIT_PATH_RANGE[0]:
        return False

    for _ in range(PAIR_RETRIES):
        path_len = random.randint(HIT_PATH_RANGE[0], max_path)
        max_source_col = W - (path_len + 5)
        if max_source_col < 0:
            return False
        source_col = random.randint(0, max_source_col)
        target_col = source_col + 2 + path_len

        source = _translate(RIGHT_SHAPE, top_row, source_col)
        target = _translate(DOWN_SHAPE, top_row + 1, target_col)
        object_cells = source + target

        if any(buffer_mask[r, c] or clear_for_objects[r, c] for r, c in object_cells):
            continue

        hit_path = tuple((top_row + 1, c) for c in range(source_col + 2, target_col))
        if any(int(grid[r, c]) != bg for r, c in hit_path):
            continue
        if any(painted_path_mask[r, c] for r, c in hit_path):
            continue

        miss_col = target_col + 1
        miss_clear = tuple((r, miss_col) for r in range(top_row + 3, H))
        if any(int(grid[r, c]) != bg for r, c in miss_clear):
            continue

        _paint_shape(grid, source, source_color)
        _paint_shape(grid, target, target_color)
        _mark_buffer(buffer_mask, source)
        _mark_buffer(buffer_mask, target)
        for r, c in hit_path:
            clear_for_objects[r, c] = True
            painted_path_mask[r, c] = True
        for r, c in miss_clear:
            clear_for_objects[r, c] = True
        return True

    return False


def _random_transform(g: np.ndarray) -> np.ndarray:
    out = np.rot90(g, random.randrange(4))
    if random.choice((False, True)):
        out = np.fliplr(out)
    return out.copy()


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    row_choices = _pair_row_choices(H, pair_count)
    if not row_choices:
        return None

    bg = random.choice(PALETTE)
    colors = random.sample([c for c in PALETTE if c != bg], 2 * pair_count)
    grid = np.full((H, W), bg, dtype=int)
    buffer_mask = np.zeros((H, W), dtype=bool)
    clear_for_objects = np.zeros((H, W), dtype=bool)
    painted_path_mask = np.zeros((H, W), dtype=bool)

    top_rows = list(random.choice(row_choices))
    random.shuffle(top_rows)
    for idx, top_row in enumerate(top_rows):
        if not _place_pair(
            grid,
            bg,
            colors[2 * idx],
            colors[2 * idx + 1],
            top_row,
            buffer_mask,
            clear_for_objects,
            painted_path_mask,
        ):
            return None

    grid = _random_transform(grid)

    try:
        _bg, objects, _claims, rays = _analyze_instance(grid)
        out = main(grid)
    except AssertionError:
        return None

    hits = sum(ray["target"] is not None for ray in rays)
    misses = len(rays) - hits
    if len(objects) != 2 * pair_count or hits != pair_count or misses != pair_count:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_RETRIES} attempts")
