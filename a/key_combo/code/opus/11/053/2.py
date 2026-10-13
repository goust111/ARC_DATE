from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import Color, find_components, paste
except ModuleNotFoundError:
    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import Color, find_components, paste


BG = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
GREEN = Color.GREEN
ORANGE = Color.ORANGE

MAX_ATTEMPTS = 500
HEIGHT_RANGE = (11, 15)
WIDTH_RANGE = (14, 18)
OBJECT_COUNT_RANGE = (3, 5)


def _normalize(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _rot90(cells: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    h = max(r for r, _ in cells) + 1
    return {(c, h - 1 - r) for r, c in cells}


def _flip_lr(cells: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    w = max(c for _, c in cells) + 1
    return {(r, w - 1 - c) for r, c in cells}


def _variants(base: set[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    cur = _normalize(base)
    for _ in range(4):
        seen.add(cur)
        seen.add(_normalize(_flip_lr(cur)))
        cur = _normalize(_rot90(cur))
    return sorted(seen)


BLUE_SHAPES: list[tuple[tuple[int, int], ...]] = []
for base_shape in [
    {(0, 0), (0, 1)},
    {(0, 0), (0, 1), (0, 2)},
    {(0, 0), (1, 0), (2, 0)},
    {(0, 0), (0, 1), (1, 0)},
    {(0, 0), (0, 1), (1, 1)},
    {(0, 0), (0, 1), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (0, 2), (1, 1)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 1), (1, 0), (1, 1), (2, 0)},
    {(0, 0), (0, 1), (1, 1), (1, 2)},
]:
    BLUE_SHAPES.extend(_variants(base_shape))
BLUE_SHAPES = sorted(set(BLUE_SHAPES))


SPRITE_MASKS: list[np.ndarray] = []
for base_shape in [
    {(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)},
    {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)},
    {(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)},
    {(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)},
    {(0, 1), (1, 0), (1, 1), (2, 1)},
]:
    for variant in _variants(base_shape):
        cells = set(variant)
        if max(r for r, _ in cells) < 3 and max(c for _, c in cells) < 3:
            mask = np.zeros((3, 3), dtype=int)
            for r, c in cells:
                mask[r, c] = GREEN
            if 3 <= int(np.count_nonzero(mask)) <= 6:
                SPRITE_MASKS.append(mask)

dedup_masks: dict[bytes, np.ndarray] = {}
for mask in SPRITE_MASKS:
    dedup_masks[mask.tobytes()] = mask
SPRITE_MASKS = list(dedup_masks.values())


def _neighbors8(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = map(int, shape)
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _marker_cells(ray_row: int) -> set[tuple[int, int]]:
    return {(ray_row, 0), (ray_row, 1), (ray_row, 2), (ray_row + 1, 1)}


def _sprite_bbox_cells(top: int, left: int) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r in range(3) for c in range(3)}


def _find_marker(grid: np.ndarray) -> tuple[int, int]:
    red = np.argwhere(grid == RED)
    assert red.shape == (3, 2), "expected exactly one three-pixel red marker bar"
    rows = sorted(set(map(int, red[:, 0])))
    cols = sorted(set(map(int, red[:, 1])))
    assert rows and len(rows) == 1 and cols == [0, 1, 2], "marker bar must be the left-edge horizontal triplet"
    ray_row = rows[0]
    orange = np.argwhere(grid == ORANGE)
    assert orange.shape == (1, 2), "expected a unique orange control pixel"
    control_r, control_c = map(int, orange[0])
    assert (control_r, control_c) == (ray_row + 1, 1), "control must sit just below the bar center"
    return ray_row, 1


def _find_sprite_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    green_components = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[GREEN],
    )
    candidates = []
    for comp in green_components:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        crop = grid[r0 : r1 + 1, c0 : c1 + 1]
        if int(np.count_nonzero(crop == GREEN)) == 9:
            continue
        candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, "expected a unique 3x3 green sprite"
    return candidates[0]


def _count_hit_objects(grid: np.ndarray, ray_row: int, center_col: int = 1) -> int:
    blue_components = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[BLUE],
    )
    hits = 0
    for comp in blue_components:
        coords = comp["coords"]
        if any(int(r) == ray_row and int(c) > center_col for r, c in coords):
            hits += 1
    return hits


def _render_output(sprite: np.ndarray, shape: tuple[int, int], top: int, left: int, copies: int) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for k in range(int(copies)):
        paste(out, sprite, top, left + 3 * k, transparent=BG, clip=True)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the blue objects crossed by the marker ray, then tile the green sprite that many times.

    Input structure:
    - The background is black.
    - A three-pixel horizontal red bar sits on the left edge, with one orange control pixel directly below its center.
    - Several separated blue 4-connected objects are scattered away from the marker, and one non-solid 3x3 green sprite sits near the lower-right.

    Rule:
    - Read the ray row from the red bar center and cast a one-pixel-wide ray horizontally to the right.
    - Count how many distinct blue 4-connected objects have at least one pixel on that ray row; multiple hit pixels from the same object still count once.
    - Erase the blue objects and the marker, then place the original 3x3 green sprite at its input position and repeat it horizontally for N total copies, clipping anything beyond the right edge.

    Key insight:
    - The ray is only a counting probe: blue object shape matters only through whether it intersects the exact ray row, and that count alone controls how many sprite copies appear.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    ray_row, center_col = _find_marker(g)
    sprite_r0, sprite_c0, sprite_r1, sprite_c1 = _find_sprite_bbox(g)
    sprite = g[sprite_r0 : sprite_r1 + 1, sprite_c0 : sprite_c1 + 1].copy()
    copies = _count_hit_objects(g, ray_row, center_col)
    assert copies > 0, "expected at least one blue object hit by the ray"
    return _render_output(sprite, g.shape, sprite_r0, sprite_c0, copies)


def _shape_cells(shape: tuple[tuple[int, int], ...], top: int, left: int) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c in shape}


def _ray_span(cells: set[tuple[int, int]], ray_row: int) -> int:
    return sum(1 for r, _ in cells if r == ray_row)


def _candidate_placements(
    shape: tuple[tuple[int, int], ...],
    grid_shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    ray_row: int,
    *,
    must_hit: bool,
    min_ray_cells: int = 1,
    near_only: bool = False,
    side: str | None = None,
) -> list[set[tuple[int, int]]]:
    h, w = map(int, grid_shape)
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    placements: list[set[tuple[int, int]]] = []

    for top in range(h - max_r):
        for left in range(3, w - max_c):
            cells = _shape_cells(shape, top, left)
            if cells & blocked:
                continue

            rows = [r for r, _ in cells]
            if must_hit:
                if _ray_span(cells, ray_row) < min_ray_cells:
                    continue
            else:
                if any(r == ray_row for r in rows):
                    continue
                if near_only and min(abs(r - ray_row) for r in rows) != 1:
                    continue
                if side == "above" and max(rows) >= ray_row:
                    continue
                if side == "below" and min(rows) <= ray_row:
                    continue

            placements.append(cells)
    return placements


def _choose_sprite_position(height: int, width: int, copies: int) -> tuple[int, int]:
    sprite_top = random.randint(max(0, height - 4), height - 3)
    clip_mode = random.random() < 0.55 or width - 3 * copies < max(width // 2, 0)

    if not clip_mode:
        hi = width - 3 * copies
        lo = max(width // 2, hi - 2)
        if lo <= hi:
            return sprite_top, random.randint(lo, hi)

    lo = max(width // 2, width - 5)
    hi = width - 3
    return sprite_top, random.randint(lo, hi)


def _paint(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _build_expected(grid: np.ndarray) -> np.ndarray:
    ray_row, _ = _find_marker(grid)
    sprite_r0, sprite_c0, sprite_r1, sprite_c1 = _find_sprite_bbox(grid)
    sprite = grid[sprite_r0 : sprite_r1 + 1, sprite_c0 : sprite_c1 + 1]
    copies = _count_hit_objects(grid, ray_row)
    return _render_output(sprite, grid.shape, sprite_r0, sprite_c0, copies)


def _validate_case(grid: np.ndarray) -> bool:
    try:
        ray_row, center_col = _find_marker(grid)
        sprite_r0, sprite_c0, sprite_r1, sprite_c1 = _find_sprite_bbox(grid)
        assert sprite_r0 >= grid.shape[0] // 2 and sprite_c0 >= grid.shape[1] // 2
        sprite = grid[sprite_r0 : sprite_r1 + 1, sprite_c0 : sprite_c1 + 1]
        assert sprite.shape == (3, 3)
        assert 3 <= int(np.count_nonzero(sprite == GREEN)) <= 6

        blue_components = find_components(
            grid,
            bg=BG,
            connectivity=4,
            monochromatic=True,
            colors=[BLUE],
        )
        assert OBJECT_COUNT_RANGE[0] <= len(blue_components) <= OBJECT_COUNT_RANGE[1]
        for comp in blue_components:
            assert 2 <= int(comp["area"]) <= 4

        blue_sets = [{(int(r), int(c)) for r, c in comp["coords"]} for comp in blue_components]
        for i in range(len(blue_sets)):
            for j in range(i + 1, len(blue_sets)):
                assert _neighbors8(blue_sets[i], grid.shape).isdisjoint(blue_sets[j])

        hits = [
            cells for cells in blue_sets
            if any(r == ray_row and c > center_col for r, c in cells)
        ]
        misses = [cells for cells in blue_sets if cells not in hits]
        assert 1 <= len(hits) < len(blue_sets)
        assert misses
        assert any(min(abs(r - ray_row) for r, _ in cells) == 1 for cells in misses)

        out = main(grid)
        expected = _build_expected(grid)
        assert np.array_equal(out, expected)
        assert out.shape == grid.shape
        assert np.count_nonzero(out == BLUE) == 0
        assert np.count_nonzero(out == RED) == 0
        assert np.count_nonzero(out == ORANGE) == 0
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    hit_count = random.randint(1, min(4, object_count - 1))
    miss_count = object_count - hit_count
    ray_row = random.randint(2, height - 5)

    grid = np.zeros((height, width), dtype=int)
    grid[ray_row, 0:3] = RED
    grid[ray_row + 1, 1] = ORANGE

    sprite = random.choice(SPRITE_MASKS).copy()
    sprite_top, sprite_left = _choose_sprite_position(height, width, hit_count)
    paste(grid, sprite, sprite_top, sprite_left)

    blocked = _neighbors8(_marker_cells(ray_row) | _sprite_bbox_cells(sprite_top, sprite_left), grid.shape)
    placed: list[set[tuple[int, int]]] = []

    force_multi_hit = hit_count >= 2 or random.random() < 0.7
    multi_hit_index = random.randrange(hit_count) if force_multi_hit else -1
    hit_indices = list(range(hit_count))
    random.shuffle(hit_indices)

    for i in hit_indices:
        need_span = 2 if i == multi_hit_index else 1
        shape_pool = BLUE_SHAPES[:]
        random.shuffle(shape_pool)
        choice: set[tuple[int, int]] | None = None
        for shape in shape_pool:
            candidates = _candidate_placements(
                shape,
                grid.shape,
                blocked,
                ray_row,
                must_hit=True,
                min_ray_cells=need_span,
            )
            if candidates:
                choice = random.choice(candidates)
                break
        if choice is None:
            return None
        placed.append(choice)
        blocked |= _neighbors8(choice, grid.shape)

    miss_specs: list[dict[str, object]] = []
    if miss_count >= 1:
        miss_specs.append({"near_only": True, "side": random.choice(["above", "below"])})
    if miss_count >= 2:
        first_side = str(miss_specs[0]["side"])
        miss_specs.append({"near_only": random.random() < 0.6, "side": "below" if first_side == "above" else "above"})
    while len(miss_specs) < miss_count:
        miss_specs.append(
            {
                "near_only": random.random() < 0.45,
                "side": random.choice([None, "above", "below"]),
            }
        )
    random.shuffle(miss_specs)

    for spec in miss_specs:
        shape_pool = BLUE_SHAPES[:]
        random.shuffle(shape_pool)
        choice = None
        for shape in shape_pool:
            candidates = _candidate_placements(
                shape,
                grid.shape,
                blocked,
                ray_row,
                must_hit=False,
                near_only=bool(spec["near_only"]),
                side=spec["side"],
            )
            if candidates:
                choice = random.choice(candidates)
                break
        if choice is None:
            return None
        placed.append(choice)
        blocked |= _neighbors8(choice, grid.shape)

    if len(placed) != object_count:
        return None

    random.shuffle(placed)
    for cells in placed:
        _paint(grid, cells, BLUE)

    if not _validate_case(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for opus/11/053/2")


if __name__ == "__main__":
    sample = generate_input()
    result = main(sample)
    print(Path(__file__).name, sample.shape, result.shape)
