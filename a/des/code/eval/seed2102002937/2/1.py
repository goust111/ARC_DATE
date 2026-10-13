import random
from collections import Counter, deque

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# --------------- solver ---------------


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    coords = list(coords)
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def _rotate_once(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple((c, -r) for r, c in coords)


def _orientations(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    cur = _normalize(coords)
    out: list[tuple[tuple[int, int], ...]] = []
    seen = set()
    for _ in range(4):
        norm = _normalize(cur)
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
        cur = _rotate_once(cur)
    return out


def _canonical(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return min(_orientations(coords))


def _shape_size(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int]:
    norm = _normalize(coords)
    return 1 + max(r for r, _ in norm), 1 + max(c for _, c in norm)


def _find_template(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, int, tuple[int, int, int, int]]] = []
    for raw_color in np.unique(grid):
        color = int(raw_color)
        if color == bg:
            continue
        pts = np.argwhere(grid == color)
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 5 or w < 5:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not (np.all(sub[0, :] == color) and np.all(sub[-1, :] == color)):
            continue
        if not (np.all(sub[:, 0] == color) and np.all(sub[:, -1] == color)):
            continue
        if np.all(sub == color):
            continue
        candidates.append((h * w, int(np.count_nonzero(grid == color)), color, (int(r0), int(c0), int(r1), int(c1))))

    assert candidates, "expected a large bordered template rectangle"
    candidates.sort(reverse=True)
    return candidates[0][2], candidates[0][3]


def _extract_holes(grid: np.ndarray, bg: int, template_color: int, bbox: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    holes: list[dict] = []
    for comp in _components(sub != template_color):
        assert all(0 < r < sub.shape[0] - 1 and 0 < c < sub.shape[1] - 1 for r, c in comp), "hole must stay inside the template"
        seed_cells = [(r, c) for r, c in comp if int(sub[r, c]) != bg]
        assert len(seed_cells) == 1, "each hole must contain exactly one seed pixel"
        seed_r, seed_c = seed_cells[0]
        seed_color = int(sub[seed_r, seed_c])
        abs_cells = [(r0 + r, c0 + c) for r, c in comp]
        holes.append(
            {
                "cells": abs_cells,
                "seed_color": seed_color,
                # Treat the seed as part of the footprint so the final repaint covers it.
                "shape": _canonical(comp),
            }
        )
    assert holes, "expected at least one template hole"
    return holes


def _extract_sprites(grid: np.ndarray, bg: int, bbox: tuple[int, int, int, int]) -> tuple[int, list[dict]]:
    r0, c0, r1, c1 = bbox
    outside = np.ones_like(grid, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    sprite_colors = sorted(set(int(v) for v in np.unique(grid[outside])) - {bg})
    assert len(sprite_colors) == 1, "all scattered sprites should share one color"
    sprite_color = sprite_colors[0]
    sprites = []
    for comp in _components(outside & (grid == sprite_color)):
        sprites.append({"cells": comp, "shape": _canonical(comp)})
    assert sprites, "expected scattered sprites outside the template"
    return sprite_color, sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match outside monochrome sprites to template holes by rotation, recolor each match
    from the hole's seed pixel, erase the original sprites, and refill the holes.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    template_color, bbox = _find_template(g, bg)
    holes = _extract_holes(g, bg, template_color, bbox)
    _sprite_color, sprites = _extract_sprites(g, bg, bbox)

    hole_shapes = Counter(hole["shape"] for hole in holes)
    sprite_shapes = Counter(sprite["shape"] for sprite in sprites)
    assert hole_shapes == sprite_shapes, "hole and sprite shape multisets must match up to rotation"

    out = g.copy()
    for sprite in sprites:
        for r, c in sprite["cells"]:
            out[r, c] = bg
    for hole in holes:
        for r, c in hole["cells"]:
            out[r, c] = hole["seed_color"]
    return out


# --------------- generator ---------------


GEN_RETRIES = 400
GRID_RANGE = (22, 30)
TEMPLATE_RANGE = (10, 16)
HOLE_COUNT_CHOICES = (2, 3, 4)
COLOR_POOL = tuple(range(10))
SHAPE_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 120


def _is_connected(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> bool:
    coords = list(coords)
    if not coords:
        return False
    cells = set(coords)
    q = deque([coords[0]])
    seen = {coords[0]}
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _seed_candidates(shape: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    out = []
    for cell in shape:
        rest = [pt for pt in shape if pt != cell]
        if _is_connected(rest):
            out.append(cell)
    return out


def _random_shape(rng: random.Random) -> tuple[tuple[int, int], ...] | None:
    for _ in range(SHAPE_ATTEMPTS):
        box_h = rng.randint(2, 4)
        box_w = rng.randint(2, 4)
        max_area = min(box_h * box_w, 7)
        min_area = min(max_area, 4)
        target_area = rng.randint(min_area, max_area)

        start = (rng.randrange(box_h), rng.randrange(box_w))
        cells = {start}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))

        if len(cells) != target_area:
            continue

        shape = _normalize(tuple(cells))
        sh, sw = _shape_size(shape)
        if sh < 2 or sw < 2:
            continue
        if len(_orientations(shape)) < 2:
            continue
        if not _seed_candidates(shape):
            continue
        return shape
    return None


def _place_shapes(
    canvas_h: int,
    canvas_w: int,
    shapes: list[tuple[tuple[int, int], ...]],
    rng: random.Random,
    base_forbidden: np.ndarray | None = None,
    border_margin: int = 1,
) -> list[list[tuple[int, int]]] | None:
    forbidden = np.zeros((canvas_h, canvas_w), dtype=bool)
    if base_forbidden is not None:
        forbidden |= base_forbidden
    if border_margin > 0:
        forbidden[:border_margin, :] = True
        forbidden[-border_margin:, :] = True
        forbidden[:, :border_margin] = True
        forbidden[:, -border_margin:] = True

    placements: list[list[tuple[int, int]]] = []
    order = sorted(range(len(shapes)), key=lambda i: len(shapes[i]), reverse=True)
    placed: dict[int, list[tuple[int, int]]] = {}

    for idx in order:
        shape = shapes[idx]
        sh, sw = _shape_size(shape)
        candidates = []
        for top in range(canvas_h - sh + 1):
            for left in range(canvas_w - sw + 1):
                abs_cells = [(top + r, left + c) for r, c in shape]
                if any(forbidden[r, c] for r, c in abs_cells):
                    continue
                candidates.append(abs_cells)
        if not candidates:
            return None
        abs_cells = rng.choice(candidates)
        placed[idx] = abs_cells
        for r, c in abs_cells:
            for nr in range(max(0, r - 1), min(canvas_h, r + 2)):
                for nc in range(max(0, c - 1), min(canvas_w, c + 2)):
                    forbidden[nr, nc] = True

    for idx in range(len(shapes)):
        placements.append(placed[idx])
    return placements


def _choose_template_size(rng: random.Random, count: int) -> tuple[int, int]:
    base = rng.randint(*TEMPLATE_RANGE)
    extra = rng.randint(0, 3)
    return max(base, 8 + 2 * count), max(rng.randint(*TEMPLATE_RANGE) + extra, 8 + 2 * count)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    hole_count = rng.choice(HOLE_COUNT_CHOICES)
    palette = rng.sample(COLOR_POOL, 3 + hole_count)
    bg, template_color, sprite_color = palette[:3]
    seed_colors = palette[3:]

    base_shapes = []
    seen_shapes = set()
    for _ in range(hole_count * 20):
        shape = _random_shape(rng)
        if shape is None:
            return None
        canon = _canonical(shape)
        if canon in seen_shapes:
            continue
        seen_shapes.add(canon)
        base_shapes.append(shape)
        if len(base_shapes) == hole_count:
            break
    if len(base_shapes) != hole_count:
        return None

    hole_shapes = [rng.choice(_orientations(shape)) for shape in base_shapes]
    sprite_shapes = [rng.choice(_orientations(shape)) for shape in base_shapes]

    template_h, template_w = _choose_template_size(rng, hole_count)
    hole_cells = _place_shapes(template_h, template_w, hole_shapes, rng, border_margin=1)
    if hole_cells is None:
        return None

    sprite_bounds_h = max(_shape_size(shape)[0] for shape in sprite_shapes)
    sprite_bounds_w = max(_shape_size(shape)[1] for shape in sprite_shapes)
    min_grid_h = max(GRID_RANGE[0], template_h + sprite_bounds_h + 8)
    min_grid_w = max(GRID_RANGE[0], template_w + sprite_bounds_w + 8)
    max_grid_h = max(min_grid_h, GRID_RANGE[1])
    max_grid_w = max(min_grid_w, GRID_RANGE[1])
    if min_grid_h > max_grid_h or min_grid_w > max_grid_w:
        return None

    grid_h = rng.randint(min_grid_h, max_grid_h)
    grid_w = rng.randint(min_grid_w, max_grid_w)
    if grid_h - template_h < 4 or grid_w - template_w < 4:
        return None

    template_r0 = rng.randint(2, grid_h - template_h - 2)
    template_c0 = rng.randint(2, grid_w - template_w - 2)
    template_r1 = template_r0 + template_h - 1
    template_c1 = template_c0 + template_w - 1

    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    for r in range(max(0, template_r0 - 1), min(grid_h, template_r1 + 2)):
        for c in range(max(0, template_c0 - 1), min(grid_w, template_c1 + 2)):
            blocked[r, c] = True
    sprite_cells = _place_shapes(grid_h, grid_w, sprite_shapes, rng, base_forbidden=blocked, border_margin=1)
    if sprite_cells is None:
        return None

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    grid[template_r0 : template_r1 + 1, template_c0 : template_c1 + 1] = template_color

    for shape_cells, seed_color in zip(hole_cells, seed_colors):
        shape = _normalize(shape_cells)
        safe = _seed_candidates(shape)
        if not safe:
            return None
        seed_local = rng.choice(safe)
        seed_abs = None
        for abs_cell in shape_cells:
            if (abs_cell[0] - min(r for r, _ in shape_cells), abs_cell[1] - min(c for _, c in shape_cells)) == seed_local:
                seed_abs = abs_cell
                break
        if seed_abs is None:
            return None
        for r, c in shape_cells:
            grid[template_r0 + r, template_c0 + c] = bg
        sr, sc = seed_abs
        grid[template_r0 + sr, template_c0 + sc] = seed_color

    for shape_cells in sprite_cells:
        for r, c in shape_cells:
            grid[r, c] = sprite_color

    if np.count_nonzero(grid == bg) <= grid.size // 2:
        return None
    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
