from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0

GRID_H_RANGE = (10, 17)
GRID_W_RANGE = (10, 17)
SPRITE_COUNT_RANGE = (4, 7)
PLACEMENT_TRIES = 250
MAX_GENERATION_ATTEMPTS = 400


SPRITE_TEMPLATES = {
    "down": {
        "cells": ((0, 0), (0, 1), (1, 0)),
        "step": (1, 0),
        "front": (1, 0),
    },
    "up": {
        "cells": ((0, 1), (1, 0), (1, 1)),
        "step": (-1, 0),
        "front": (0, 1),
    },
    "right": {
        "cells": ((0, 0), (1, 0), (1, 1)),
        "step": (0, 1),
        "front": (1, 1),
    },
    "left": {
        "cells": ((0, 0), (0, 1), (1, 1)),
        "step": (0, -1),
        "front": (0, 0),
    },
}

PATTERN_TO_GEOMETRY = {
    frozenset(spec["cells"]): (direction, spec["step"], spec["front"])
    for direction, spec in SPRITE_TEMPLATES.items()
}


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        queue = deque([(r, c)])
        comp: list[tuple[int, int]] = []

        while queue:
            cr, cc = queue.popleft()
            comp.append((cr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))

        comps.append(comp)

    return comps


def _decode_sprites(grid: np.ndarray) -> tuple[list[dict[str, object]], dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    sprites: list[dict[str, object]] = []
    coord_to_sprite: dict[tuple[int, int], int] = {}

    for comp in _components4(g != BACKGROUND):
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each sprite must be monochrome"
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0 = min(rows)
        r1 = max(rows)
        c0 = min(cols)
        c1 = max(cols)
        assert len(comp) == 3 and r1 - r0 == 1 and c1 - c0 == 1, "expected 3-cell 2x2 L sprites"

        pattern = frozenset((r - r0, c - c0) for r, c in comp)
        assert pattern in PATTERN_TO_GEOMETRY, "unexpected sprite shape"
        direction, step, front_rel = PATTERN_TO_GEOMETRY[pattern]
        front = (r0 + front_rel[0], c0 + front_rel[1])

        sprite_id = len(sprites)
        sprites.append(
            {
                "id": sprite_id,
                "color": int(next(iter(colors))),
                "coords": tuple(sorted(comp)),
                "bbox": (r0, c0, r1, c1),
                "direction": direction,
                "step": step,
                "front": front,
            }
        )
        for cell in comp:
            coord_to_sprite[cell] = sprite_id

    assert sprites, "expected at least one sprite"
    sprites.sort(key=lambda sprite: tuple(sprite["bbox"]))  # type: ignore[arg-type]

    remapped_coord_to_sprite: dict[tuple[int, int], int] = {}
    for idx, sprite in enumerate(sprites):
        sprite["id"] = idx
        for cell in sprite["coords"]:  # type: ignore[index]
            remapped_coord_to_sprite[cell] = idx

    return sprites, remapped_coord_to_sprite


def _trace_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sprites, coord_to_sprite = _decode_sprites(g)

    painted: dict[tuple[int, int], int] = {}
    outgoing: dict[int, int] = {}
    recolor: dict[int, int] = {}
    ray_lengths: dict[int, int] = {}

    for sprite in sprites:
        source_id = int(sprite["id"])
        color = int(sprite["color"])
        dr, dc = sprite["step"]  # type: ignore[assignment]
        fr, fc = sprite["front"]  # type: ignore[assignment]
        r = fr + dr
        c = fc + dc
        ray_len = 0

        while 0 <= r < h and 0 <= c < w:
            if int(g[r, c]) == BACKGROUND:
                assert (r, c) not in painted, "rays must not overlap"
                painted[(r, c)] = color
                ray_len += 1
                r += dr
                c += dc
                continue

            target_id = coord_to_sprite.get((r, c))
            assert target_id is not None and target_id != source_id, "rays may only stop on other sprites"
            assert target_id not in recolor, "each target sprite must be hit by at most one ray"
            outgoing[source_id] = target_id
            recolor[target_id] = color
            break

        ray_lengths[source_id] = ray_len

    return {
        "sprites": sprites,
        "painted": painted,
        "outgoing": outgoing,
        "recolor": recolor,
        "ray_lengths": ray_lengths,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shoot straight rays from 3-cell L arrows and recolor the first sprite each ray hits.

    Input structure:
    - Background is black.
    - Foreground objects are isolated monochrome 3-cell L sprites, each filling
      three cells of a 2x2 box.

    Rule:
    - Decode each of the four L orientations as one cardinal-facing arrow.
    - Start from the sprite's front cell and cast a one-cell-wide ray in that
      direction through background cells.
    - Paint traversed background cells in the source color.
    - Stop when the ray reaches the first cell of another sprite; do not
      overwrite the impact cell itself, but recolor the whole hit sprite to the
      ray's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    traced = _trace_scene(g)
    out = g.copy()

    for (r, c), color in traced["painted"].items():  # type: ignore[index]
        out[r, c] = int(color)

    sprites = traced["sprites"]  # type: ignore[assignment]
    for target_id, color in traced["recolor"].items():  # type: ignore[index]
        coords = sprites[int(target_id)]["coords"]  # type: ignore[index]
        for r, c in coords:
            out[r, c] = int(color)

    return out


def _sample_directions(count: int) -> list[str]:
    directions = list(SPRITE_TEMPLATES)
    result = directions * (count // len(directions))
    result.extend(random.sample(directions, count % len(directions)))
    random.shuffle(result)
    return result


def _candidate_origins(direction: str, h: int, w: int) -> list[tuple[int, int]]:
    if direction == "up":
        rows = range(1, h - 1)
        cols = range(0, w - 1)
    elif direction == "down":
        rows = range(0, h - 2)
        cols = range(0, w - 1)
    elif direction == "left":
        rows = range(0, h - 1)
        cols = range(1, w - 1)
    else:
        assert direction == "right"
        rows = range(0, h - 1)
        cols = range(0, w - 2)
    result = [(int(r), int(c)) for r in rows for c in cols]
    random.shuffle(result)
    return result


def _place_sprites(h: int, w: int, directions: list[str], colors: list[int]) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    for direction, color in zip(directions, colors):
        cells = SPRITE_TEMPLATES[direction]["cells"]
        placed = False
        for r0, c0 in _candidate_origins(direction, h, w):
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + 3)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + 3)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue

            for dr, dc in cells:
                grid[r0 + dr, c0 + dc] = color
            blocked[rr0:rr1, cc0:cc1] = True
            placed = True
            break

        if not placed:
            return None

    return grid


def _has_chain(outgoing: dict[int, int]) -> bool:
    return any(target in outgoing and outgoing[target] != source for source, target in outgoing.items())


def _good_candidate(grid: np.ndarray, require_chain: bool) -> bool:
    try:
        traced = _trace_scene(grid)
        solved = main(grid)
    except AssertionError:
        return False

    sprites = traced["sprites"]  # type: ignore[assignment]
    outgoing = traced["outgoing"]  # type: ignore[assignment]
    ray_lengths = traced["ray_lengths"]  # type: ignore[assignment]

    hit_count = len(outgoing)
    no_hit_count = len(sprites) - hit_count
    if hit_count < 1 or no_hit_count < 1:
        return False
    if min(ray_lengths.values()) < 1:
        return False
    if np.array_equal(solved, grid):
        return False

    non_bg = int(np.count_nonzero(grid))
    if non_bg * 5 >= grid.size:
        return False

    if require_chain and not _has_chain(outgoing):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    sprite_count = random.randint(*SPRITE_COUNT_RANGE)

    directions = _sample_directions(sprite_count)
    colors = random.sample(range(1, 10), sprite_count)
    grid = _place_sprites(h, w, directions, colors)
    if grid is None:
        return None
    if not _good_candidate(grid, require_chain=random.random() < 0.25):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
