from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


DIAGONAL_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
PROJECTILE_STEPS = 5
MAX_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    ]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_sprite(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict | None:
    if len(comp) != 3:
        return None
    colors = {int(grid[r, c]) for r, c in comp}
    if len(colors) != 1:
        return None
    cells = set(comp)
    corner: tuple[int, int] | None = None
    arms: list[tuple[int, int]] = []
    for r, c in comp:
        neighbors4 = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in cells:
                neighbors4.append((dr, dc))
        if len(neighbors4) == 2:
            if neighbors4[0][0] == neighbors4[1][0] or neighbors4[0][1] == neighbors4[1][1]:
                continue
            corner = (r, c)
            arms = neighbors4
            break
    if corner is None:
        return None

    dr = arms[0][0] + arms[1][0]
    dc = arms[0][1] + arms[1][1]
    if dr == 0 or dc == 0:
        return None
    direction = (1 if dr > 0 else -1, 1 if dc > 0 else -1)
    landing = (corner[0] + PROJECTILE_STEPS * direction[0], corner[1] + PROJECTILE_STEPS * direction[1])
    return {
        "color": int(next(iter(colors))),
        "apex": corner,
        "direction": direction,
        "landing": landing,
    }


def _draw_layer(out: np.ndarray, center: tuple[int, int], radius: int, color: int) -> None:
    r, c = center
    if radius == 0:
        out[r, c] = color
        return
    r0, r1 = r - radius, r + radius
    c0, c1 = c - radius, c + radius
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color


def _render_from_sprites(shape: tuple[int, int], sprites: list[dict], bg: int) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    landing_to_sprites: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for sprite in sprites:
        landing_to_sprites[sprite["landing"]].append(sprite)

    for landing, group in landing_to_sprites.items():
        ordered = sorted(
            group,
            key=lambda item: (
                abs(item["landing"][0] - item["apex"][0]) + abs(item["landing"][1] - item["apex"][1]),
                item["color"],
            ),
        )
        for radius, sprite in enumerate(ordered):
            _draw_layer(out, landing, radius, sprite["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire each 3-pixel elbow sprite toward its missing corner and build concentric square layers at shared landing cells.

    Input structure:
    - Black background with 3-4 isolated monochrome 3-pixel L sprites.
    - Each sprite occupies three corners of a 2x2 box, leaving one missing corner.
    - The elbow corner pixel is the firing apex, and the missing-corner diagonal gives the firing direction.

    Rule:
    - For each sprite, find the elbow corner and move 5 diagonal steps toward the missing corner to get its landing cell.
    - Erase all sprites to background.
    - Group sprites by landing cell; within each group, sort by apex-to-landing Manhattan distance, then by sprite color.
    - Paint the first color at the landing center, the second as a 3x3 square frame around it, the third as a 5x5 frame, and so on.

    Key insight:
    - Sprite orientation determines where each ray lands, and the number of rays converging on one landing cell determines how many concentric square layers are built there.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    sprites: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for comp in _components(g == color):
            sprite = _extract_sprite(g, comp)
            if sprite is None:
                continue
            lr, lc = sprite["landing"]
            if 0 <= lr < g.shape[0] and 0 <= lc < g.shape[1]:
                sprites.append(sprite)
    return _render_from_sprites(g.shape, sprites, bg)


def _sprite_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = apex
    dr, dc = direction
    return ((r, c), (r + dr, c), (r, c + dc))


def _path_cells(apex: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = apex
    dr, dc = direction
    return [(r + step * dr, c + step * dc) for step in range(1, PROJECTILE_STEPS + 1)]


def _sample_groups() -> list[int]:
    if random.random() < 0.5:
        return [2, 1]
    return [3, 1] if random.random() < 0.45 else [2, 1, 1]


def _landing_radius(group_size: int) -> int:
    return group_size - 1


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _choose_landings(shape: tuple[int, int], groups: list[int]) -> list[tuple[int, int]] | None:
    h, w = shape
    candidates = [(r, c) for r in range(5, h - 5) for c in range(5, w - 5)]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    boxes: list[tuple[int, int, int, int]] = []
    for group_size in groups:
        radius = _landing_radius(group_size)
        random.shuffle(candidates)
        picked = None
        for r, c in candidates:
            box = (r - radius - 1, c - radius - 1, r + radius + 1, c + radius + 1)
            if any(_boxes_overlap(box, other) for other in boxes):
                continue
            picked = (r, c)
            boxes.append(box)
            chosen.append(picked)
            break
        if picked is None:
            return None
    return chosen


def _valid_layout(shape: tuple[int, int], rays: list[dict]) -> bool:
    h, w = shape
    occupied: set[tuple[int, int]] = set()
    halos: set[tuple[int, int]] = set()
    all_paths: dict[int, set[tuple[int, int]]] = {}

    for idx, ray in enumerate(rays):
        for cell in ray["cells"]:
            r, c = cell
            if not (0 <= r < h and 0 <= c < w):
                return False
            if cell in occupied:
                return False
        all_paths[idx] = set(ray["path"])

    for idx, ray in enumerate(rays):
        for cell in ray["cells"]:
            for other_idx, path in all_paths.items():
                if other_idx == idx:
                    continue
                if cell in path:
                    return False
            if cell in halos:
                return False
        for r, c in ray["cells"]:
            occupied.add((r, c))
        for r, c in ray["cells"]:
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    if 0 <= rr < h and 0 <= cc < w:
                        halos.add((rr, cc))
        if any(ray["landing"] in other["cells"] for other in rays):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(21, 26)
    w = random.randint(21, 26)
    shape = (h, w)
    groups = _sample_groups()
    landings = _choose_landings(shape, groups)
    if landings is None:
        return None

    ray_specs: list[dict] = []
    colors = random.sample(PALETTE, sum(groups))
    color_idx = 0
    for landing, group_size in zip(landings, groups):
        directions = random.sample(DIAGONAL_DIRS, group_size)
        for direction in directions:
            apex = (
                landing[0] - PROJECTILE_STEPS * direction[0],
                landing[1] - PROJECTILE_STEPS * direction[1],
            )
            cells = _sprite_cells(apex, direction)
            ray_specs.append(
                {
                    "color": colors[color_idx],
                    "apex": apex,
                    "direction": direction,
                    "landing": landing,
                    "cells": cells,
                    "path": _path_cells(apex, direction),
                }
            )
            color_idx += 1

    if not _valid_layout(shape, ray_specs):
        return None

    grid = np.zeros(shape, dtype=int)
    for ray in ray_specs:
        for r, c in ray["cells"]:
            grid[r, c] = ray["color"]

    expected = _render_from_sprites(shape, ray_specs, 0)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    landing_counts = Counter(ray["landing"] for ray in ray_specs)
    if max(landing_counts.values()) < 2:
        return None
    if 1 not in landing_counts.values():
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
