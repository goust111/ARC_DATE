from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
POSITIVE_SEED_SHAPES = ((1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2), (3, 3))
ZERO_SEED_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2))
MAX_GENERATION_ATTEMPTS = 500


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    box = _mask_bbox(mask)
    assert box is not None, "expected a non-empty mask"
    return crop_bbox(mask.astype(int), box).astype(bool)


def _shift_mask(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    src_r0 = max(0, -int(dr))
    src_r1 = min(h, h - int(dr))
    src_c0 = max(0, -int(dc))
    src_c1 = min(w, w - int(dc))
    dst_r0 = src_r0 + int(dr)
    dst_r1 = src_r1 + int(dr)
    dst_c0 = src_c0 + int(dc)
    dst_c1 = src_c1 + int(dc)
    if src_r0 < src_r1 and src_c0 < src_c1:
        out[dst_r0:dst_r1, dst_c0:dst_c1] = mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out |= _shift_mask(mask, dr, dc)
    return out


def _diagonal_growth_round(figure: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    diag = (
        _shift_mask(figure, -1, -1)
        | _shift_mask(figure, -1, 1)
        | _shift_mask(figure, 1, -1)
        | _shift_mask(figure, 1, 1)
    )
    orth = (
        _shift_mask(figure, -1, 0)
        | _shift_mask(figure, 1, 0)
        | _shift_mask(figure, 0, -1)
        | _shift_mask(figure, 0, 1)
    )
    return diag & (~orth) & (~occupied)


def _grow_diagonally(seed_mask: np.ndarray, steps: int, occupied: np.ndarray | None = None) -> np.ndarray:
    figure = seed_mask.astype(bool).copy()
    blocked = figure.copy() if occupied is None else occupied.astype(bool).copy()
    for _ in range(int(steps)):
        new_cells = _diagonal_growth_round(figure, blocked)
        figure |= new_cells
        blocked |= new_cells
    return figure


def _grow_full8(seed_mask: np.ndarray, steps: int) -> np.ndarray:
    figure = seed_mask.astype(bool).copy()
    for _ in range(int(steps)):
        figure = _dilate8(figure)
    return figure


def _normalized_shape(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "shape must be non-empty"
    r0, c0 = coords.min(axis=0)
    shifted = coords - np.array([r0, c0], dtype=int)
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _seed_and_counters(grid: np.ndarray, bg: int, color: int) -> tuple[dict, list[dict]]:
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=True, colors=[int(color)])
    assert comps, f"expected at least one component for color {color}"

    non_singletons = [comp for comp in comps if int(comp["area"]) > 1]
    if non_singletons:
        assert len(non_singletons) == 1, f"color {color} must have one non-singleton seed"
        seed = non_singletons[0]
        counters = [comp for comp in comps if comp is not seed]
        assert all(int(comp["area"]) == 1 for comp in counters), f"color {color} counters must be singleton pixels"
        return seed, counters

    assert len(comps) == 1, f"color {color} is ambiguous: multiple isolated singletons but no larger seed"
    return comps[0], []


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed diagonally for as many rounds as its same-color singleton counters indicate, then erase the counters.

    Input structure:
    - The dominant background contains several well-separated colors.
    - For each color there is one seed component: either one pixel or a small solid rectangle.
    - Additional components of that same color are isolated singleton pixels that do not touch any same-color pixel under 8-neighborhood.
    - Different colors are placed far enough apart that their finished growths stay disjoint and non-adjacent.

    Rule:
    - For each color, count its isolated singleton pixels; call that count k.
    - Starting from only the seed component, grow that color for k rounds.
    - In one round, color every background cell that is diagonally adjacent to the current figure while not being orthogonally adjacent to the current figure.
    - After all colors finish growing, remove every singleton counter pixel and keep only the seeds plus their growth traces.

    Key insight:
    - The scattered single pixels are not part of the target shape; they are step counters for a deliberately unusual diagonal-only growth operator.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    colors = sorted(int(color) for color in np.unique(g) if int(color) != bg)
    base_occupied = g != bg
    out = np.full_like(g, bg)

    for color in colors:
        seed, counters = _seed_and_counters(g, bg, color)
        seed_mask = np.zeros_like(g, dtype=bool)
        coords = np.asarray(seed["coords"], dtype=int)
        seed_mask[coords[:, 0], coords[:, 1]] = True

        grown = _grow_diagonally(seed_mask, len(counters), base_occupied)
        assert not np.any((out != bg) & grown), "grown figures must not overlap"
        out[grown] = color

    return out


def _local_seed_and_final(shape: tuple[int, int], steps: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = map(int, shape)
    pad = int(steps) + 2
    seed_canvas = np.zeros((h + 2 * pad, w + 2 * pad), dtype=bool)
    seed_canvas[pad : pad + h, pad : pad + w] = True
    final_canvas = _grow_diagonally(seed_canvas, steps)
    box = _mask_bbox(final_canvas)
    assert box is not None
    return crop_bbox(seed_canvas.astype(int), box).astype(bool), crop_bbox(final_canvas.astype(int), box).astype(bool)


def _sample_profile() -> tuple[str, int, list[int]]:
    profile = random.choices(
        ("contrast", "shared_steps", "singleton_zero"),
        weights=(3, 4, 3),
        k=1,
    )[0]

    if profile == "contrast":
        n_colors = random.choice((2, 3, 4))
        counts = random.sample((1, 2, 3, 4), n_colors)
        return profile, n_colors, counts

    if profile == "shared_steps":
        n_colors = random.choice((3, 4))
        shared = random.randint(1, 4)
        other = random.choice([k for k in (1, 2, 3, 4) if k != shared])
        counts = [shared, shared, other]
        if n_colors == 4:
            extra = random.choice([0, 1, 2, 3, 4])
            if extra == shared and counts.count(shared) >= 2:
                extra = other
            counts.append(extra)
        random.shuffle(counts)
        return profile, n_colors, counts

    n_colors = random.choice((3, 4))
    shared = random.randint(1, 4)
    other = random.choice([k for k in (1, 2, 3, 4) if k != shared])
    counts = [0, shared, other]
    if n_colors == 4:
        counts.append(shared if random.random() < 0.5 else random.choice([k for k in (1, 2, 3, 4) if k != other]))
    random.shuffle(counts)
    return profile, n_colors, counts


def _sample_shapes(profile: str, counts: list[int]) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    used_for_count: dict[int, set[tuple[int, int]]] = {}
    singleton_used = False

    for count in counts:
        if count == 0:
            if profile == "singleton_zero" and not singleton_used and random.random() < 0.8:
                shape = (1, 1)
                singleton_used = True
            else:
                shape = random.choice(ZERO_SEED_SHAPES)
            shapes.append(shape)
            continue

        pool = list(POSITIVE_SEED_SHAPES)
        used = used_for_count.setdefault(int(count), set())
        choices = [shape for shape in pool if shape not in used]
        if not choices:
            choices = pool
        shape = random.choice(choices)
        used.add(shape)
        shapes.append(shape)

    return shapes


def _build_specs() -> list[dict]:
    profile, n_colors, counts = _sample_profile()
    colors = random.sample(PALETTE, n_colors)
    shapes = _sample_shapes(profile, counts)

    specs: list[dict] = []
    for color, steps, shape in zip(colors, counts, shapes, strict=True):
        seed_patch, final_patch = _local_seed_and_final(shape, steps)
        specs.append(
            {
                "color": int(color),
                "steps": int(steps),
                "shape": tuple(map(int, shape)),
                "seed_patch": seed_patch,
                "final_patch": final_patch,
            }
        )
    return specs


def _canvas_shape(specs: list[dict]) -> tuple[int, int]:
    max_h = max(int(spec["final_patch"].shape[0]) for spec in specs)
    max_w = max(int(spec["final_patch"].shape[1]) for spec in specs)
    total_area = sum(int(spec["final_patch"].sum()) for spec in specs)
    total_steps = sum(int(spec["steps"]) for spec in specs)
    base = max(max_h + 2, max_w + 2, int(np.ceil(np.sqrt(total_area + 8 * total_steps))) + 5)
    base = max(14, min(base, 26))
    h = random.randint(base, min(30, base + 4))
    w = random.randint(base, min(30, base + 4))
    return h, w


def _place_final_patches(specs: list[dict], shape: tuple[int, int]) -> bool:
    occupancy = np.zeros(shape, dtype=int)
    order = sorted(range(len(specs)), key=lambda idx: int(specs[idx]["final_patch"].sum()), reverse=True)

    for idx in order:
        spec = specs[idx]
        patch = spec["final_patch"].astype(int)
        try:
            top_left = random_free_location(
                occupancy,
                patch,
                transparent=BACKGROUND,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return False
        spec["top_left"] = top_left
        paste(occupancy, patch, top_left[0], top_left[1], transparent=BACKGROUND)
    return True


def _render_seed_only_grid(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for spec in specs:
        r0, c0 = spec["top_left"]
        seed_patch = spec["seed_patch"].astype(int) * int(spec["color"])
        paste(grid, seed_patch, r0, c0, transparent=BACKGROUND)
    return grid


def _render_expected_output(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    for spec in specs:
        r0, c0 = spec["top_left"]
        final_patch = spec["final_patch"].astype(int) * int(spec["color"])
        paste(out, final_patch, r0, c0, transparent=BACKGROUND)
    return out


def _place_counters(seed_grid: np.ndarray, expected: np.ndarray, specs: list[dict]) -> np.ndarray | None:
    grid = seed_grid.copy()
    reserved = (expected != BACKGROUND).astype(int)
    dot_patch = np.array([[1]], dtype=int)

    for spec in specs:
        for _ in range(int(spec["steps"])):
            try:
                r0, c0 = random_free_location(
                    reserved,
                    dot_patch,
                    transparent=BACKGROUND,
                    border=0,
                    padding=1,
                    padding_connectivity=8,
                )
            except ValueError:
                return None
            grid[r0, c0] = int(spec["color"])
            reserved[r0, c0] = 1
    return grid


def _seed_only_occupancy(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    seed_only = np.full_like(grid, BACKGROUND)
    for spec in specs:
        color = int(spec["color"])
        seed, _counters = _seed_and_counters(grid, BACKGROUND, color)
        coords = np.asarray(seed["coords"], dtype=int)
        seed_only[coords[:, 0], coords[:, 1]] = color
    return seed_only != BACKGROUND


def _validate_semantics(grid: np.ndarray, expected: np.ndarray, specs: list[dict]) -> bool:
    if not np.array_equal(main(grid), expected):
        return False

    base_occupied = grid != BACKGROUND
    seed_only_occupied = _seed_only_occupancy(grid, specs)

    positive_steps = 0
    diagonal_differs_from_full8 = 0

    for spec in specs:
        color = int(spec["color"])
        steps = int(spec["steps"])
        seed, counters = _seed_and_counters(grid, BACKGROUND, color)
        if len(counters) != steps:
            return False

        seed_mask = np.zeros_like(grid, dtype=bool)
        coords = np.asarray(seed["coords"], dtype=int)
        seed_mask[coords[:, 0], coords[:, 1]] = True

        grown_with_counters = _grow_diagonally(seed_mask, steps, base_occupied)
        grown_seed_only = _grow_diagonally(seed_mask, steps, seed_only_occupied)
        target_mask = expected == color
        if not np.array_equal(grown_with_counters, target_mask):
            return False
        if not np.array_equal(grown_seed_only, target_mask):
            return False

        if steps > 0:
            positive_steps += 1
            previous = _grow_diagonally(seed_mask, steps - 1, base_occupied)
            if np.array_equal(previous, target_mask):
                return False

            full8 = _normalized_shape(_crop_mask(_grow_full8(spec["seed_patch"], steps)))
            diagonal = _normalized_shape(spec["final_patch"])
            if full8 != diagonal:
                diagonal_differs_from_full8 += 1

    if positive_steps == 0:
        return False
    if diagonal_differs_from_full8 == 0:
        return False

    masks = [expected == int(spec["color"]) for spec in specs]
    for idx, mask_a in enumerate(masks):
        for mask_b in masks[idx + 1 :]:
            if np.any(_dilate8(mask_a) & mask_b):
                return False

    return True


def _attempt_generate() -> np.ndarray | None:
    specs = _build_specs()
    shape = _canvas_shape(specs)
    if not _place_final_patches(specs, shape):
        return None

    seed_grid = _render_seed_only_grid(shape, specs)
    expected = _render_expected_output(shape, specs)
    grid = _place_counters(seed_grid, expected, specs)
    if grid is None:
        return None

    if not _validate_semantics(grid, expected, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-growth counting instance")
