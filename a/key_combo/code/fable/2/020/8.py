from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, colors_present, find_components, get_bg, paste, random_free_location, random_sprite


BG = Color.BLACK
RESIDUE = Color.GREY
SPRITE_COLORS = [color for color in Color.NOT_BLACK if color != RESIDUE]

GRID_HEIGHT_RANGE = (18, 29)
GRID_WIDTH_RANGE = (18, 29)
SPRITE_COUNT_CHOICES = (2, 2, 3)
SPRITE_HEIGHT_RANGE = (2, 4)
SPRITE_WIDTH_RANGE = (3, 5)
GAP_CHOICES = (1, 2, 3)
PLACEMENT_PADDING = 2
GENERATION_RETRIES = 400


def _component_sort_key(component: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = component["bbox"]
    return int(r0), int(c0), int(r1), int(c1)


def _mirror_local_coord(
    r: int,
    c: int,
    shape: tuple[int, int],
    orientation: str,
) -> tuple[int, int]:
    h, w = map(int, shape)
    if orientation == "vertical":
        return int(r), int(w - 1 - c)
    if orientation == "horizontal":
        return int(h - 1 - r), int(c)
    raise ValueError(f"unknown orientation: {orientation}")


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict], list[tuple[int, int]]]:
    bg = int(get_bg(grid))
    components = find_components(grid, bg=bg, connectivity=8, monochromatic=False)
    sprites: list[dict] = []
    residues: list[tuple[int, int]] = []

    for component in sorted(components, key=_component_sort_key):
        coords = [(int(r), int(c)) for r, c in component["coords"]]
        if int(component["area"]) == 1:
            r, c = coords[0]
            if int(grid[r, c]) == RESIDUE:
                residues.append((r, c))
                continue

        cells = [(r, c, int(grid[r, c])) for r, c in coords]
        sprites.append(
            {
                "cells": cells,
                "coords": frozenset((r, c) for r, c, _color in cells),
                "bbox": tuple(map(int, component["bbox"])),
            }
        )

    assert sprites, "expected at least one source sprite"
    assert residues, "expected at least one grey residue pixel"
    return bg, sprites, residues


def _candidate_paint(
    sprite: dict,
    orientation: str,
    axis2: int,
    shape: tuple[int, int],
    blocked: frozenset[tuple[int, int]],
) -> dict[tuple[int, int], int] | None:
    height, width = map(int, shape)
    paint: dict[tuple[int, int], int] = {}

    for r, c, color in sprite["cells"]:
        if orientation == "vertical":
            rr, cc = int(r), int(axis2 - c)
        else:
            rr, cc = int(axis2 - r), int(c)
        if not (0 <= rr < height and 0 <= cc < width):
            return None
        if (rr, cc) in blocked:
            return None
        paint[(rr, cc)] = int(color)

    if len(paint) != len(sprite["cells"]):
        return None
    return paint


def _sprite_candidates(
    sprite: dict,
    residues: list[tuple[int, int]],
    residue_to_index: dict[tuple[int, int], int],
    shape: tuple[int, int],
    blocked: frozenset[tuple[int, int]],
) -> list[dict]:
    axis_options = {"vertical": set(), "horizontal": set()}
    for rr, cc in residues:
        for sr, sc, _color in sprite["cells"]:
            if sr == rr:
                axis_options["vertical"].add(int(sc + cc))
            if sc == cc:
                axis_options["horizontal"].add(int(sr + rr))

    candidates: list[dict] = []
    for orientation in ("vertical", "horizontal"):
        for axis2 in sorted(axis_options[orientation]):
            paint = _candidate_paint(sprite, orientation, axis2, shape, blocked)
            if paint is None:
                continue
            residue_mask = 0
            for coord in paint:
                residue_index = residue_to_index.get(coord)
                if residue_index is not None:
                    residue_mask |= 1 << residue_index
            if residue_mask == 0:
                continue

            candidates.append(
                {
                    "orientation": orientation,
                    "axis2": int(axis2),
                    "paint": paint,
                    "support": frozenset(paint),
                    "residue_mask": int(residue_mask),
                }
            )

    candidates.sort(
        key=lambda cand: (
            -cand["residue_mask"].bit_count(),
            cand["orientation"],
            cand["axis2"],
        )
    )
    return candidates


def _solve_scene(grid: np.ndarray) -> tuple[int, list[dict], list[tuple[int, int]], list[dict[int, dict]], list[list[dict]]]:
    bg, sprites, residues = _parse_scene(grid)
    blocked = frozenset(coord for sprite in sprites for coord in sprite["coords"])
    residue_to_index = {coord: idx for idx, coord in enumerate(residues)}

    candidate_lists: list[list[dict]] = []
    for sprite in sprites:
        candidates = _sprite_candidates(sprite, residues, residue_to_index, grid.shape, blocked)
        candidate_lists.append(candidates)

    order = sorted(range(len(sprites)), key=lambda idx: (len(candidate_lists[idx]), -len(sprites[idx]["cells"])))
    suffix_cover = [0] * (len(order) + 1)
    for pos in range(len(order) - 1, -1, -1):
        cover = 0
        for candidate in candidate_lists[order[pos]]:
            cover |= candidate["residue_mask"]
        suffix_cover[pos] = suffix_cover[pos + 1] | cover

    all_residues_mask = (1 << len(residues)) - 1
    solutions: list[dict[int, dict]] = []

    def search(
        pos: int,
        used_residues: int,
        occupied_support: frozenset[tuple[int, int]],
        chosen: dict[int, dict],
    ) -> None:
        if len(solutions) > 1:
            return
        if pos == len(order):
            if used_residues == all_residues_mask:
                solutions.append(chosen.copy())
            return
        if (used_residues | suffix_cover[pos]) != all_residues_mask:
            return

        sprite_idx = order[pos]
        for candidate in candidate_lists[sprite_idx]:
            if candidate["residue_mask"] & used_residues:
                continue
            if candidate["support"] & occupied_support:
                continue
            chosen[sprite_idx] = candidate
            search(
                pos + 1,
                used_residues | candidate["residue_mask"],
                occupied_support | candidate["support"],
                chosen,
            )
            del chosen[sprite_idx]

    search(0, 0, frozenset(), {})
    return bg, sprites, residues, solutions, candidate_lists


def _render_solution(grid: np.ndarray, solution: dict[int, dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for candidate in solution.values():
        for (r, c), color in candidate["paint"].items():
            out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full colored mirror copies whose only surviving input traces are unlabeled grey residue pixels.

    Input structure:
    - The background is black.
    - Several complete multicolor sprites appear as separated foreground objects.
    - Additional isolated grey singletons are scattered elsewhere; each one is a surviving cell from some missing horizontal or vertical mirror copy.
    - The mirror axes are never drawn.

    Rule:
    - Treat each grey singleton as an unlabeled residue and each larger foreground component as an original sprite.
    - For every sprite, enumerate the horizontal and vertical axes whose reflected support could contain one or more residues.
    - Choose the unique global assignment in which every grey residue belongs to exactly one sprite reflection and the completed reflections do not overlap any original sprite or each other.
    - Draw every reflected copy in the sprite's original colors, so each grey residue is restored to the source color it came from.

    Key insight:
    - The grey traces have lost their colors, so sprite ownership and mirror-axis selection must be solved jointly from reflection consistency across the whole scene.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _bg, _sprites, _residues, solutions, _candidate_lists = _solve_scene(g)
    assert len(solutions) == 1, "expected exactly one consistent joint assignment"
    return _render_solution(g, solutions[0])


def _well_separated(coords: tuple[tuple[int, int], ...]) -> bool:
    for (r1, c1), (r2, c2) in itertools.combinations(coords, 2):
        if max(abs(int(r1) - int(r2)), abs(int(c1) - int(c2))) <= 1:
            return False
    return True


def _pick_residue_sources(sprite: np.ndarray) -> list[tuple[int, int]] | None:
    cells = [(int(r), int(c)) for r, c in np.argwhere(sprite != BG)]
    random.shuffle(cells)

    for count in (3, 2):
        valid = [
            combo
            for combo in itertools.combinations(cells, count)
            if _well_separated(combo)
        ]
        if not valid:
            continue

        colorful = [
            combo
            for combo in valid
            if len({int(sprite[r, c]) for r, c in combo}) >= 2
        ]
        chosen = random.choice(colorful or valid)
        return [(int(r), int(c)) for r, c in chosen]
    return None


def _random_multicolor_sprite() -> tuple[np.ndarray, list[tuple[int, int]]]:
    for _ in range(200):
        height = random.randint(*SPRITE_HEIGHT_RANGE)
        width = random.randint(*SPRITE_WIDTH_RANGE)
        palette = random.sample(SPRITE_COLORS, random.choice((2, 3)))
        try:
            sprite = random_sprite(
                m=height,
                n=width,
                density=random.uniform(0.45, 0.75),
                color_palette=palette,
                connectivity=4,
                background=BG,
            )
        except AssertionError:
            continue
        if int(np.sum(sprite != BG)) < 4:
            continue
        if len(colors_present(sprite, exclude={BG})) < 2:
            continue
        residue_sources = _pick_residue_sources(sprite)
        if residue_sources is None:
            continue
        return sprite, residue_sources
    raise RuntimeError("failed to generate a suitable multicolor sprite")


def _choose_orientations(sprite_count: int) -> list[str]:
    if sprite_count == 2 and random.random() < 0.7:
        orientations = ["vertical", "horizontal"]
        random.shuffle(orientations)
        return orientations

    orientations = [random.choice(("vertical", "horizontal")) for _ in range(sprite_count)]
    if sprite_count >= 3 and len(set(orientations)) == 1 and random.random() < 0.7:
        orientations[random.randrange(sprite_count)] = "horizontal" if orientations[0] == "vertical" else "vertical"
    return orientations


def _pair_layout(
    sprite_shape: tuple[int, int],
    orientation: str,
    side: int,
    gap: int,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    height, width = map(int, sprite_shape)
    gap = int(gap)
    if orientation == "vertical":
        pair_shape = (height, 2 * width + gap)
        if side > 0:
            return pair_shape, (0, 0), (0, width + gap)
        return pair_shape, (0, width + gap), (0, 0)

    pair_shape = (2 * height + gap, width)
    if side > 0:
        return pair_shape, (0, 0), (height + gap, 0)
    return pair_shape, (height + gap, 0), (0, 0)


def _scene_is_interesting(candidate_lists: list[list[dict]], residue_count: int) -> bool:
    total_candidates = sum(len(candidates) for candidates in candidate_lists)
    if total_candidates <= len(candidate_lists):
        return False

    for residue_idx in range(residue_count):
        owners = 0
        residue_bit = 1 << residue_idx
        for candidates in candidate_lists:
            if any(candidate["residue_mask"] & residue_bit for candidate in candidates):
                owners += 1
        if owners >= 2:
            return True

    return any(len({candidate["orientation"] for candidate in candidates}) >= 2 for candidates in candidate_lists)


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    sprite_count = random.choice(SPRITE_COUNT_CHOICES)
    orientations = _choose_orientations(sprite_count)

    occupancy = np.zeros((height, width), dtype=int)
    grid = np.zeros((height, width), dtype=int)
    expected = np.zeros((height, width), dtype=int)
    restored_residue_colors: list[int] = []

    for sprite_idx in range(sprite_count):
        sprite, residue_sources = _random_multicolor_sprite()
        orientation = orientations[sprite_idx]
        side = random.choice((-1, 1))
        gap = random.choice(GAP_CHOICES)
        pair_shape, original_offset, mirror_offset = _pair_layout(sprite.shape, orientation, side, gap)
        pair_patch = np.ones(pair_shape, dtype=int)

        try:
            pair_r0, pair_c0 = random_free_location(
                occupancy,
                pair_patch,
                transparent=BG,
                border=1,
                padding=PLACEMENT_PADDING,
            )
        except ValueError:
            return None

        paste(occupancy, pair_patch, pair_r0, pair_c0)
        original_r0 = pair_r0 + original_offset[0]
        original_c0 = pair_c0 + original_offset[1]
        mirror_r0 = pair_r0 + mirror_offset[0]
        mirror_c0 = pair_c0 + mirror_offset[1]

        paste(grid, sprite, original_r0, original_c0, transparent=BG)
        paste(expected, sprite, original_r0, original_c0, transparent=BG)

        for r, c in np.argwhere(sprite != BG):
            rr, cc = _mirror_local_coord(int(r), int(c), sprite.shape, orientation)
            expected[mirror_r0 + rr, mirror_c0 + cc] = int(sprite[r, c])

        for r, c in residue_sources:
            rr, cc = _mirror_local_coord(int(r), int(c), sprite.shape, orientation)
            grid[mirror_r0 + rr, mirror_c0 + cc] = RESIDUE
            restored_residue_colors.append(int(sprite[r, c]))

    if len(set(restored_residue_colors)) < 2:
        return None

    _bg, _sprites, residues, solutions, candidate_lists = _solve_scene(grid)
    if len(solutions) != 1:
        return None
    if not _scene_is_interesting(candidate_lists, len(residues)):
        return None

    solved = _render_solution(grid, solutions[0])
    if not np.array_equal(solved, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid residue-mirror scene")
