from __future__ import annotations

from collections import Counter
import random

import numpy as np
from scipy.ndimage import label


FOUR_CONNECTED = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=int,
)

GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 28)
TARGET_COUNT_RANGE = (2, 4)
LEGEND_SIZE_RANGE = (4, 5)
OBJECT_PADDING = 1
MAX_RETRIES = 400


def _boundary_bg(grid: np.ndarray) -> int:
    border = np.concatenate([grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]])
    return int(Counter(map(int, border)).most_common(1)[0][0])


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    labeled, count = label(mask.astype(int), structure=FOUR_CONNECTED)
    comps: list[np.ndarray] = []
    for cid in range(1, int(count) + 1):
        coords = np.argwhere(labeled == cid)
        if coords.size:
            comps.append(coords.astype(int))
    return comps


def _find_legend(grid: np.ndarray, bg: int) -> tuple[int, int, set[int], np.ndarray]:
    h, w = grid.shape
    for size in range(2, min(h, w) + 1):
        border_color = int(grid[size - 1, size - 1])
        if border_color == bg:
            continue
        if not np.all(grid[size - 1, :size] == border_color):
            continue
        if not np.all(grid[:size, size - 1] == border_color):
            continue
        interior = grid[: size - 1, : size - 1]
        colors = set(map(int, np.unique(interior)))
        colors.discard(bg)
        colors.discard(border_color)
        if not colors:
            continue
        legend_mask = np.zeros_like(grid, dtype=bool)
        legend_mask[:size, :size] = True
        return int(size), border_color, colors, legend_mask
    raise AssertionError("legend not found")


def _enclosed_bg_regions(subgrid: np.ndarray, bg: int) -> list[np.ndarray]:
    bg_mask = subgrid == bg
    regions = []
    for coords in _component_coords(bg_mask):
        rows = coords[:, 0]
        cols = coords[:, 1]
        if (
            np.any(rows == 0)
            or np.any(cols == 0)
            or np.any(rows == subgrid.shape[0] - 1)
            or np.any(cols == subgrid.shape[1] - 1)
        ):
            continue
        regions.append(coords)
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete complete-palette objects and fill one-missing-color holes from the corner legend.

    Input structure:
    - The top-left corner contains a small square legend whose bottom and right edges share one border color; the interior lists a target color set.
    - Elsewhere on the background sit several separated 4-connected multicolor objects.
    - Some objects contain enclosed background holes that do not touch the grid boundary.

    Rule:
    - Read the legend's interior colors as the required target palette.
    - Ignore the legend itself when scanning objects.
    - Erase any object whose color set contains every target color.
    - For any remaining object that is missing exactly one target color, fill all of its enclosed background holes with that missing color.

    Key insight:
    - The corner legend is an "expected palette" shared by every object: complete objects disappear, while one-missing objects reveal the answer by coloring their holes with the unique absent legend color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _boundary_bg(g)
    _size, _border_color, legend_colors, legend_mask = _find_legend(g, bg)

    out = g.copy()
    object_mask = (g != bg) & (~legend_mask)
    for coords in _component_coords(object_mask):
        rows = coords[:, 0]
        cols = coords[:, 1]
        obj_colors = {int(g[r, c]) for r, c in coords}
        if legend_colors.issubset(obj_colors):
            out[rows, cols] = bg
            continue

        missing = legend_colors - obj_colors
        if len(missing) != 1:
            continue

        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        subgrid = g[r0 : r1 + 1, c0 : c1 + 1]
        fill_color = next(iter(missing))
        for hole in _enclosed_bg_regions(subgrid, bg):
            out[r0 + hole[:, 0], c0 + hole[:, 1]] = fill_color

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _is_connected(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    _labeled, count = label(mask.astype(int), structure=FOUR_CONNECTED)
    return int(count) == 1


def _mask_holes(mask: np.ndarray) -> list[np.ndarray]:
    return _enclosed_bg_regions(np.where(mask, 1, 0), 0)


def _make_ring_mask(rng: random.Random) -> np.ndarray:
    h = rng.randint(5, 8)
    w = rng.randint(5, 8)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    options = ["none"]
    if w >= 6:
        options.append("vertical")
    if h >= 6:
        options.append("horizontal")
    split = rng.choice(options)
    if split == "vertical":
        wall_c = rng.randint(2, w - 3)
        mask[1:-1, wall_c] = True
    elif split == "horizontal":
        wall_r = rng.randint(2, h - 3)
        mask[wall_r, 1:-1] = True
    return mask


def _make_solid_mask(rng: random.Random) -> np.ndarray:
    h = rng.randint(3, 7)
    w = rng.randint(3, 7)
    mask = np.ones((h, w), dtype=bool)
    removals = rng.randint(1, max(1, (h * w) // 4))
    for _ in range(removals):
        candidates = []
        for r, c in np.argwhere(mask):
            if not (r in (0, h - 1) or c in (0, w - 1)):
                continue
            test = mask.copy()
            test[r, c] = False
            if int(test.sum()) < 5:
                continue
            if not _is_connected(test):
                continue
            if _mask_holes(test):
                continue
            candidates.append((int(r), int(c)))
        if not candidates:
            break
        rr, cc = rng.choice(candidates)
        mask[rr, cc] = False
    return _crop_mask(mask)


def _colorize_mask(mask: np.ndarray, colors: list[int], bg: int, rng: random.Random) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    seeds = rng.sample(coords, len(colors))
    patch = np.full(mask.shape, bg, dtype=int)
    for r, c in coords:
        best_idx = min(
            range(len(colors)),
            key=lambda idx: (
                abs(r - seeds[idx][0]) + abs(c - seeds[idx][1]),
                idx,
            ),
        )
        patch[r, c] = colors[best_idx]
    return patch


def _choose_object_palette(
    role: str,
    target_colors: list[int],
    extra_pool: list[int],
    cell_count: int,
    rng: random.Random,
) -> tuple[list[int], int | None]:
    if role == "delete":
        chosen = list(target_colors)
        missing = None
        extra_lo = 0
    elif role == "fill":
        missing = rng.choice(target_colors)
        chosen = [color for color in target_colors if color != missing]
        extra_lo = max(0, 2 - len(chosen))
    else:
        keep_target_max = max(0, len(target_colors) - 2)
        keep_target_count = rng.randint(0, keep_target_max) if keep_target_max else 0
        chosen = rng.sample(target_colors, keep_target_count)
        missing = None
        extra_lo = max(0, 2 - len(chosen))

    extra_hi = min(len(extra_pool), extra_lo + 2)
    extra_count = rng.randint(extra_lo, extra_hi) if extra_hi >= extra_lo else extra_lo
    chosen.extend(rng.sample(extra_pool, extra_count))

    while len(chosen) < min(2, cell_count):
        remaining = [color for color in extra_pool if color not in chosen]
        chosen.append(rng.choice(remaining))

    rng.shuffle(chosen)
    return chosen, missing


def _paste_patch(grid: np.ndarray, patch: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    view = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    view[mask] = patch[mask]


def _reserve_bbox(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[
        max(0, r0 - OBJECT_PADDING) : min(occupied.shape[0], r0 + h + OBJECT_PADDING),
        max(0, c0 - OBJECT_PADDING) : min(occupied.shape[1], c0 + w + OBJECT_PADDING),
    ] = True


def _candidate_positions(occupied: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    positions = []
    for r0 in range(1, occupied.shape[0] - h):
        for c0 in range(1, occupied.shape[1] - w):
            if occupied[r0 : r0 + h, c0 : c0 + w].any():
                continue
            positions.append((r0, c0))
    return positions


def _place_legend(
    grid: np.ndarray,
    occupied: np.ndarray,
    bg: int,
    border_color: int,
    target_colors: list[int],
    size: int,
    rng: random.Random,
) -> None:
    grid[:size, :size] = bg
    grid[size - 1, :size] = border_color
    grid[:size, size - 1] = border_color

    cells = [(r, c) for r in range(size - 1) for c in range(size - 1)]
    rng.shuffle(cells)
    for color, (r, c) in zip(target_colors, cells):
        grid[r, c] = color

    occupied[: min(grid.shape[0], size + OBJECT_PADDING), : min(grid.shape[1], size + OBJECT_PADDING)] = True


def _legend_matches_expectation(
    grid: np.ndarray, bg: int, expected_size: int, expected_targets: set[int]
) -> bool:
    try:
        size, _border_color, targets, _mask = _find_legend(grid, bg)
    except AssertionError:
        return False
    return size == expected_size and targets == expected_targets


def _role_specs(rng: random.Random) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = [("fill", "ring"), ("delete", rng.choice(["ring", "solid"])), ("keep", "solid")]
    for _ in range(rng.randint(0, 1)):
        specs.append(("fill", "ring"))
    for _ in range(rng.randint(0, 1)):
        specs.append(("delete", rng.choice(["ring", "solid"])))
    for _ in range(rng.randint(0, 1)):
        specs.append(("keep", rng.choice(["ring", "solid"])))
    rng.shuffle(specs)
    return specs


def _build_object(
    role: str,
    shape_kind: str,
    bg: int,
    target_colors: list[int],
    extra_pool: list[int],
    rng: random.Random,
) -> dict:
    mask = _make_ring_mask(rng) if shape_kind == "ring" else _make_solid_mask(rng)
    holes = _mask_holes(mask)
    if role == "fill" and not holes:
        raise AssertionError("fill object needs a hole")
    if shape_kind == "solid" and holes:
        raise AssertionError("solid object should not contain holes")

    colors, missing = _choose_object_palette(role, target_colors, extra_pool, int(mask.sum()), rng)
    patch = _colorize_mask(mask, colors, bg, rng)
    return {
        "role": role,
        "mask": mask,
        "patch": patch,
        "holes": holes,
        "missing": missing,
    }


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    all_colors = list(range(10))
    rng.shuffle(all_colors)
    bg = all_colors.pop()
    border_color = all_colors.pop()
    target_count = rng.randint(*TARGET_COUNT_RANGE)
    target_colors = [all_colors.pop() for _ in range(target_count)]
    extra_pool = list(all_colors)
    legend_size = rng.randint(*LEGEND_SIZE_RANGE)
    if legend_size - 1 < 2 or (legend_size - 1) * (legend_size - 1) < target_count:
        return None

    inp = np.full((h, w), bg, dtype=int)
    expected = inp.copy()
    occupied = np.zeros((h, w), dtype=bool)
    occupied[0, :] = True
    occupied[-1, :] = True
    occupied[:, 0] = True
    occupied[:, -1] = True

    _place_legend(inp, occupied, bg, border_color, target_colors, legend_size, rng)
    expected[:legend_size, :legend_size] = inp[:legend_size, :legend_size]
    if not _legend_matches_expectation(inp, bg, legend_size, set(target_colors)):
        return None

    objects = []
    for role, shape_kind in _role_specs(rng):
        try:
            obj = _build_object(role, shape_kind, bg, target_colors, extra_pool, rng)
        except (AssertionError, ValueError):
            return None
        mask = obj["mask"]
        positions = _candidate_positions(occupied, mask.shape[0], mask.shape[1])
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paste_patch(inp, obj["patch"], mask, r0, c0)
        _paste_patch(expected, obj["patch"], mask, r0, c0)
        _reserve_bbox(occupied, r0, c0, mask.shape[0], mask.shape[1])
        obj["origin"] = (r0, c0)
        objects.append(obj)

    deleted_count = 0
    filled_count = 0
    for obj in objects:
        r0, c0 = obj["origin"]
        mask = obj["mask"]
        if obj["role"] == "delete":
            expected[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = bg
            deleted_count += 1
        elif obj["role"] == "fill":
            fill_color = int(obj["missing"])
            for hole in obj["holes"]:
                expected[r0 + hole[:, 0], c0 + hole[:, 1]] = fill_color
            filled_count += 1

    if deleted_count == 0 or filled_count == 0:
        return None

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
