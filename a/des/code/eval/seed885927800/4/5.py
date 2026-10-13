from __future__ import annotations

import random
import numpy as np


RED = 2
BLUE = 1
GREEN = 3
YELLOW = 4

SEED_TRANSFORMS: dict[int, str] = {
    RED: "rot90_cw",
    BLUE: "flip_lr",
    GREEN: "flip_ud",
    YELLOW: "rot180",
}

# --------------- solver ---------------


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _apply_transform(dr: int, dc: int, seed_color: int) -> tuple[int, int]:
    if seed_color == RED:
        return dc, -dr
    if seed_color == BLUE:
        return dr, -dc
    if seed_color == GREEN:
        return -dr, dc
    if seed_color == YELLOW:
        return -dr, -dc
    raise AssertionError(f"unsupported seed color: {seed_color}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each two-color connected object around its unique seed pixel, then remove the seed.

    Seed color decoding:
    - red    -> rotate 90 degrees clockwise about the seed
    - blue   -> mirror left-right about the seed
    - green  -> mirror up-down about the seed
    - yellow -> rotate 180 degrees about the seed
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    components = _find_components(g != bg)
    assert components, "expected at least one foreground object"

    out = np.full_like(g, bg)
    h, w = g.shape

    for coords in components:
        by_color: dict[int, list[tuple[int, int]]] = {}
        for r, c in coords:
            by_color.setdefault(int(g[r, c]), []).append((r, c))

        assert len(by_color) == 2, "each object must contain one body color and one seed color"

        singleton_seed_colors = [
            color for color, pts in by_color.items() if len(pts) == 1 and color in SEED_TRANSFORMS
        ]
        if len(singleton_seed_colors) == 1:
            seed_color = singleton_seed_colors[0]
        else:
            singleton_colors = [color for color, pts in by_color.items() if len(pts) == 1]
            assert len(singleton_colors) == 1, "expected exactly one singleton seed pixel"
            seed_color = singleton_colors[0]
            assert seed_color in SEED_TRANSFORMS, f"invalid seed color: {seed_color}"

        seed = by_color[seed_color][0]
        body_colors = [color for color in by_color if color != seed_color]
        assert len(body_colors) == 1
        body_color = body_colors[0]

        for r, c in by_color[body_color]:
            dr, dc = r - seed[0], c - seed[1]
            tdr, tdc = _apply_transform(dr, dc, seed_color)
            rr, cc = seed[0] + tdr, seed[1] + tdc
            assert 0 <= rr < h and 0 <= cc < w, "transformed object left the grid"
            assert out[rr, cc] == bg, "transformed objects must not overlap"
            out[rr, cc] = body_color

    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (22, 30)
GRID_WIDTH_RANGE = (22, 30)
OBJECT_COUNT_RANGE = (4, 6)
INPUT_GAP = 1
OUTPUT_GAP = 1

SHAPE_HEIGHT_RANGE = (4, 7)
SHAPE_WIDTH_RANGE = (4, 7)
SHAPE_CELL_RANGE = (8, 15)
SHAPE_ATTEMPTS = 120
GENERATION_ATTEMPTS = 400

BODY_COLOR_POOL = tuple(range(10))
MANDATORY_SEEDS = (RED, BLUE, GREEN, YELLOW)


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in coords and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _interior_cells(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    interior = []
    for r, c in coords:
        if all((nr, nc) in coords for nr, nc in _neighbors4(r, c)):
            interior.append((r, c))
    return interior


def _adjacent_count(cell: tuple[int, int], coords: set[tuple[int, int]]) -> int:
    return sum((nr, nc) in coords for nr, nc in _neighbors4(*cell))


def _apply_orientation(dr: int, dc: int, flip_lr: bool, rot_k: int) -> tuple[int, int]:
    if flip_lr:
        dc = -dc
    for _ in range(rot_k % 4):
        dr, dc = dc, -dr
    return dr, dc


def _orient_offsets(
    offsets: list[tuple[int, int]], flip_lr: bool, rot_k: int
) -> list[tuple[int, int]]:
    return [_apply_orientation(dr, dc, flip_lr, rot_k) for dr, dc in offsets]


def _grow_shape() -> set[tuple[int, int]]:
    h = random.randint(*SHAPE_HEIGHT_RANGE)
    w = random.randint(*SHAPE_WIDTH_RANGE)
    root = (random.randint(1, h - 2), random.randint(1, w - 2))

    coords = {
        root,
        (root[0] - 1, root[1]),
        (root[0] + 1, root[1]),
        (root[0], root[1] - 1),
        (root[0], root[1] + 1),
    }
    target = random.randint(SHAPE_CELL_RANGE[0], min(SHAPE_CELL_RANGE[1], h * w))

    while len(coords) < target:
        frontier: list[tuple[int, int]] = []
        weights: list[int] = []
        for r in range(h):
            for c in range(w):
                cell = (r, c)
                if cell in coords:
                    continue
                adj = _adjacent_count(cell, coords)
                if adj == 0:
                    continue
                frontier.append(cell)
                weights.append(1 + 2 * adj)
        if not frontier:
            break
        if random.random() < 0.55:
            best = max(weights)
            choices = [cell for cell, weight in zip(frontier, weights) if weight == best]
            coords.add(random.choice(choices))
        else:
            coords.add(random.choices(frontier, weights=weights, k=1)[0])
    return coords


def _generate_object(seed_color: int) -> dict[str, object] | None:
    for _ in range(SHAPE_ATTEMPTS):
        coords = _grow_shape()
        if len(coords) < SHAPE_CELL_RANGE[0]:
            continue

        interiors = _interior_cells(coords)
        if not interiors:
            continue

        center_r = sum(r for r, _c in coords) / len(coords)
        center_c = sum(c for _r, c in coords) / len(coords)
        random.shuffle(interiors)
        interiors.sort(
            key=lambda rc: abs(rc[0] - center_r) + abs(rc[1] - center_c),
            reverse=random.random() < 0.7,
        )

        for seed in interiors:
            body = set(coords)
            body.remove(seed)
            if not _is_connected(body):
                continue

            body_offsets = [(r - seed[0], c - seed[1]) for r, c in body]
            flip_lr = random.choice((False, True))
            rot_k = random.randrange(4)
            body_offsets = _orient_offsets(body_offsets, flip_lr=flip_lr, rot_k=rot_k)
            out_offsets = [_apply_transform(dr, dc, seed_color) for dr, dc in body_offsets]

            if set(out_offsets) == set(body_offsets):
                continue

            return {
                "seed_color": seed_color,
                "body_offsets": body_offsets,
                "out_offsets": out_offsets,
                "size": len(body_offsets),
            }
    return None


def _far_enough(
    coords: set[tuple[int, int]], used: set[tuple[int, int]], gap: int
) -> bool:
    for r, c in coords:
        for rr in range(r - gap, r + gap + 1):
            for cc in range(c - gap, c + gap + 1):
                if (rr, cc) in used:
                    return False
    return True


def _place_objects(
    specs: list[dict[str, object]], height: int, width: int
) -> list[dict[str, object]] | None:
    placements: list[dict[str, object]] = []
    used_input: set[tuple[int, int]] = set()
    used_output: set[tuple[int, int]] = set()

    order = sorted(specs, key=lambda spec: int(spec["size"]), reverse=True)
    for spec in order:
        body_offsets = list(spec["body_offsets"])
        out_offsets = list(spec["out_offsets"])
        input_offsets = body_offsets + [(0, 0)]
        all_offsets = input_offsets + out_offsets

        min_r = min(dr for dr, _dc in all_offsets)
        max_r = max(dr for dr, _dc in all_offsets)
        min_c = min(dc for _dr, dc in all_offsets)
        max_c = max(dc for _dr, dc in all_offsets)

        r_lo = -min_r
        r_hi = height - 1 - max_r
        c_lo = -min_c
        c_hi = width - 1 - max_c
        if r_lo > r_hi or c_lo > c_hi:
            return None

        candidates = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]
        random.shuffle(candidates)

        placed = None
        for seed_r, seed_c in candidates:
            input_coords = {(seed_r + dr, seed_c + dc) for dr, dc in input_offsets}
            output_coords = {(seed_r + dr, seed_c + dc) for dr, dc in out_offsets}

            if not _far_enough(input_coords, used_input, INPUT_GAP):
                continue
            if not _far_enough(output_coords, used_output, OUTPUT_GAP):
                continue

            placed = {
                "seed_color": int(spec["seed_color"]),
                "seed": (seed_r, seed_c),
                "body_offsets": body_offsets,
                "out_offsets": out_offsets,
            }
            used_input.update(input_coords)
            used_output.update(output_coords)
            break

        if placed is None:
            return None
        placements.append(placed)

    return placements


def _render_scene(
    placements: list[dict[str, object]], height: int, width: int, bg: int, body_colors: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.full((height, width), bg, dtype=int)
    out = np.full((height, width), bg, dtype=int)

    for placement, body_color in zip(placements, body_colors):
        seed_r, seed_c = placement["seed"]
        seed_color = int(placement["seed_color"])
        body_offsets = list(placement["body_offsets"])
        out_offsets = list(placement["out_offsets"])

        for dr, dc in body_offsets:
            grid[seed_r + dr, seed_c + dc] = body_color
        grid[seed_r, seed_c] = seed_color

        for dr, dc in out_offsets:
            out[seed_r + dr, seed_c + dc] = body_color

    return grid, out


def _body_colors(bg: int, count: int) -> list[int]:
    palette = [c for c in BODY_COLOR_POOL if c != bg and c not in SEED_TRANSFORMS]
    random.shuffle(palette)
    colors = []
    for i in range(count):
        if i < len(palette):
            colors.append(palette[i])
        else:
            colors.append(random.choice(palette))
    return colors


def _generate_attempt() -> np.ndarray | None:
    count = random.randint(*OBJECT_COUNT_RANGE)
    seed_colors = list(MANDATORY_SEEDS)
    while len(seed_colors) < count:
        seed_colors.append(random.choice(MANDATORY_SEEDS))
    random.shuffle(seed_colors)

    specs: list[dict[str, object]] = []
    for seed_color in seed_colors:
        spec = _generate_object(seed_color)
        if spec is None:
            return None
        specs.append(spec)

    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    placements = _place_objects(specs, height, width)
    if placements is None:
        return None

    bg_choices = [c for c in BODY_COLOR_POOL if c not in SEED_TRANSFORMS]
    bg = random.choice(bg_choices)
    body_colors = _body_colors(bg, len(placements))
    grid, expected = _render_scene(placements, height, width, bg, body_colors)

    if np.count_nonzero(grid == bg) <= grid.size // 2:
        return None
    if len(_find_components(grid != bg)) != len(placements):
        return None
    if {int(grid[r, c]) for placement in placements for r, c in [placement["seed"]]} != set(MANDATORY_SEEDS):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
