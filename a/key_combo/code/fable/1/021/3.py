from __future__ import annotations

import random
from collections import Counter

import numpy as np


GEOMETRY_NAMES = ("rot90", "rot180", "rot270", "flip_lr", "flip_ud")
INVERSE_GEOMETRY = {
    "rot90": "rot270",
    "rot180": "rot180",
    "rot270": "rot90",
    "flip_lr": "flip_lr",
    "flip_ud": "flip_ud",
}
GENERATION_RETRIES = 400
GENERATION_PROFILES = (
    ("single_mid", "rot90"),
    ("double_right", "flip_lr"),
    ("double_left", "flip_ud"),
    ("double_mid", "rot270"),
    ("single_mid", "rot180"),
)
GENERATION_CALLS = 0


def get_bg(grid: np.ndarray) -> int:
    """Return the most common color in the grid."""
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _apply_geometry(sprite: np.ndarray, geometry: str) -> np.ndarray:
    if geometry == "rot90":
        return np.rot90(sprite, k=-1)
    if geometry == "rot180":
        return np.rot90(sprite, k=2)
    if geometry == "rot270":
        return np.rot90(sprite, k=1)
    if geometry == "flip_lr":
        return np.fliplr(sprite)
    if geometry == "flip_ud":
        return np.flipud(sprite)
    raise ValueError(f"unknown geometry {geometry}")


def _invert_mapping(mapping: dict[int, int]) -> dict[int, int]:
    return {dst: src for src, dst in mapping.items()}


def _apply_color_map(sprite: np.ndarray, mapping: dict[int, int], bg: int) -> np.ndarray:
    out = np.asarray(sprite, dtype=int).copy()
    base = np.asarray(sprite, dtype=int)
    for src, dst in mapping.items():
        out[base == int(src)] = int(dst)
    out[base == int(bg)] = int(bg)
    return out


def _apply_step(sprite: np.ndarray, geometry: str, mapping: dict[int, int], bg: int) -> np.ndarray:
    return _apply_color_map(_apply_geometry(sprite, geometry), mapping, bg)


def _apply_inverse_step(sprite: np.ndarray, geometry: str, mapping: dict[int, int], bg: int) -> np.ndarray:
    inverse_mapping = _invert_mapping(mapping)
    inverse_geometry = INVERSE_GEOMETRY[geometry]
    return _apply_geometry(_apply_color_map(sprite, inverse_mapping, bg), inverse_geometry)


def _apply_step_k(sprite: np.ndarray, geometry: str, mapping: dict[int, int], bg: int, steps: int) -> np.ndarray:
    out = np.asarray(sprite, dtype=int).copy()
    if steps >= 0:
        for _ in range(steps):
            out = _apply_step(out, geometry, mapping, bg)
    else:
        for _ in range(-steps):
            out = _apply_inverse_step(out, geometry, mapping, bg)
    return out


def _apply_geometry_only_k(sprite: np.ndarray, geometry: str, steps: int) -> np.ndarray:
    out = np.asarray(sprite, dtype=int).copy()
    step_geometry = geometry if steps >= 0 else INVERSE_GEOMETRY[geometry]
    for _ in range(abs(steps)):
        out = _apply_geometry(out, step_geometry)
    return out


def _apply_color_only_k(sprite: np.ndarray, mapping: dict[int, int], bg: int, steps: int) -> np.ndarray:
    out = np.asarray(sprite, dtype=int).copy()
    step_mapping = mapping if steps >= 0 else _invert_mapping(mapping)
    for _ in range(abs(steps)):
        out = _apply_color_map(out, step_mapping, bg)
    return out


def _candidate_geometries(shape: tuple[int, int]) -> list[str]:
    h, w = map(int, shape)
    out = ["rot180", "flip_lr", "flip_ud"]
    if h == w:
        out = list(GEOMETRY_NAMES)
    return out


def _parse_cell_layout(grid: np.ndarray, bg: int) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[dict] = []

    for n_cells in range(3, w + 1):
        if (w + 1) % n_cells != 0:
            continue
        span = (w + 1) // n_cells
        cell_w = span - 1
        if cell_w < 2:
            continue

        divider_cols = [span * k - 1 for k in range(1, n_cells)]
        divider_values = []
        ok = True
        for c in divider_cols:
            column = g[:, c]
            if not np.all(column == column[0]):
                ok = False
                break
            divider_values.append(int(column[0]))
        if not ok:
            continue

        divider = divider_values[0]
        if divider == bg or any(value != divider for value in divider_values):
            continue

        cell_slices = []
        start = 0
        for _ in range(n_cells):
            cell_slices.append((slice(0, h), slice(start, start + cell_w)))
            start += cell_w + 1

        if any(np.any(g[rsl, csl] == divider) for rsl, csl in cell_slices):
            continue

        candidates.append(
            {
                "divider": divider,
                "n_cells": n_cells,
                "cell_w": cell_w,
                "cell_slices": cell_slices,
            }
        )

    assert len(candidates) == 1, "expected one valid divider partition"
    return candidates[0]


def _extract_cells(grid: np.ndarray, layout: dict) -> list[np.ndarray]:
    return [np.asarray(grid[rsl, csl], dtype=int).copy() for rsl, csl in layout["cell_slices"]]


def _occupied_mask(cells: list[np.ndarray], bg: int) -> list[bool]:
    return [bool(np.any(cell != bg)) for cell in cells]


def _observed_adjacent_pairs(cells: list[np.ndarray], occupied: list[bool]) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(cells[i], cells[i + 1]) for i in range(len(cells) - 1) if occupied[i] and occupied[i + 1]]


def _infer_pair_mapping(src: np.ndarray, dst: np.ndarray, geometry: str, bg: int) -> dict[int, int] | None:
    transformed = _apply_geometry(src, geometry)
    if not np.array_equal(transformed != bg, dst != bg):
        return None

    mapping: dict[int, int] = {}
    reverse: dict[int, int] = {}
    for color in sorted(set(map(int, transformed[transformed != bg].ravel()))):
        target_colors = np.unique(dst[transformed == color])
        if target_colors.size != 1:
            return None
        mapped = int(target_colors[0])
        if mapped == bg:
            return None
        if color in mapping and mapping[color] != mapped:
            return None
        if mapped in reverse and reverse[mapped] != color:
            return None
        mapping[color] = mapped
        reverse[mapped] = color

    if set(mapping.values()) != set(map(int, np.unique(dst[dst != bg]))):
        return None
    return mapping


def _merge_mapping(base: dict[int, int], extra: dict[int, int]) -> bool:
    reverse = _invert_mapping(base)
    for src, dst in extra.items():
        if src in base and base[src] != dst:
            return False
        if dst in reverse and reverse[dst] != src:
            return False
        base[src] = dst
        reverse[dst] = src
    return True


def _reconstruct_sequence(
    cells: list[np.ndarray],
    occupied: list[bool],
    geometry: str,
    mapping: dict[int, int],
    bg: int,
) -> list[np.ndarray] | None:
    visible = [i for i, present in enumerate(occupied) if present]
    assert visible, "expected at least one visible cell"

    palette = set()
    for i in visible:
        palette.update(map(int, np.unique(cells[i][cells[i] != bg])))

    if not palette <= set(mapping):
        return None
    if not palette <= set(mapping.values()):
        return None

    out: list[np.ndarray | None] = [None] * len(cells)
    anchor = visible[0]
    out[anchor] = cells[anchor].copy()

    for i in range(anchor + 1, len(cells)):
        out[i] = _apply_step(out[i - 1], geometry, mapping, bg)
    for i in range(anchor - 1, -1, -1):
        out[i] = _apply_inverse_step(out[i + 1], geometry, mapping, bg)

    resolved = [cell for cell in out if cell is not None]
    assert len(resolved) == len(cells)
    final = [np.asarray(cell, dtype=int) for cell in out]
    for i in visible:
        if not np.array_equal(final[i], cells[i]):
            return None
    return final


def _valid_models(cells: list[np.ndarray], occupied: list[bool], bg: int) -> list[dict]:
    pairs = _observed_adjacent_pairs(cells, occupied)
    assert pairs, "expected at least one visible adjacent pair"

    models = []
    for geometry in _candidate_geometries(cells[0].shape):
        mapping: dict[int, int] = {}
        ok = True
        for src, dst in pairs:
            pair_mapping = _infer_pair_mapping(src, dst, geometry, bg)
            if pair_mapping is None or not _merge_mapping(mapping, pair_mapping):
                ok = False
                break
        if not ok:
            continue

        sequence = _reconstruct_sequence(cells, occupied, geometry, mapping, bg)
        if sequence is None:
            continue
        models.append({"geometry": geometry, "mapping": mapping.copy(), "sequence": sequence})

    return models


def _compose_grid(cells: list[np.ndarray], divider: int, bg: int) -> np.ndarray:
    cell_h, cell_w = map(int, cells[0].shape)
    n_cells = len(cells)
    out = np.full((cell_h, n_cells * cell_w + (n_cells - 1)), int(bg), dtype=int)
    start = 0
    for i, cell in enumerate(cells):
        out[:, start : start + cell_w] = cell
        start += cell_w
        if i != n_cells - 1:
            out[:, start] = int(divider)
            start += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the empty divider-separated cells by inferring and composing one shared sprite step transform.

    Input structure
    - One color forms full-height vertical divider lines that split the canvas into one row of equal-sized cells.
    - Some cells are empty, while every non-empty cell contains the same multicolor sprite at a different step in one sequence.
    - A fixed step transform maps each sprite to the next one by combining one rotation or flip with one color permutation.

    Rule
    - Detect the unique divider partition and extract the equal cells.
    - From every adjacent visible cell pair, jointly infer the geometry part and the color-permutation part of the shared step transform.
    - Reconstruct the full sequence by composing the step transform forward and its inverse backward across the missing cells, then put those sprites back while keeping the dividers unchanged.

    Key insight
    - The sprite progression is defined by geometry and color together; the missing cells are recovered only by finding the single transform that makes all visible adjacent pairs agree and then composing it across the gap length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30

    bg = get_bg(g)
    layout = _parse_cell_layout(g, bg)
    cells = _extract_cells(g, layout)
    occupied = _occupied_mask(cells, bg)
    assert any(occupied) and not all(occupied), "expected at least one gap and at least one visible sprite"

    models = _valid_models(cells, occupied, bg)
    assert len(models) == 1, "expected a unique shared step transform"

    return _compose_grid(models[0]["sequence"], layout["divider"], bg)


def _neighbors4(r: int, c: int, size: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            out.append((nr, nc))
    return out


def _random_mask(size: int) -> np.ndarray | None:
    area = size * size
    low = max(size + 2, int(np.ceil(area * 0.42)))
    high = min(area - 2, int(np.floor(area * 0.64)))
    target = random.randint(low, high)

    cells = {(random.randrange(size), random.randrange(size))}
    while len(cells) < target:
        frontier = []
        for r, c in list(cells):
            for nr, nc in _neighbors4(r, c, size):
                if (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    mask = np.zeros((size, size), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    if not mask.any(axis=0).all() or not mask.any(axis=1).all():
        return None
    for geometry in _candidate_geometries(mask.shape):
        if np.array_equal(mask, _apply_geometry(mask, geometry)):
            return None
    return mask


def _random_sprite(size: int, bg: int, palette: list[int]) -> np.ndarray | None:
    for _ in range(120):
        mask = _random_mask(size)
        if mask is None:
            continue
        coords = [tuple(coord) for coord in np.argwhere(mask)]
        if len(coords) < len(palette):
            continue

        random.shuffle(coords)
        sprite = np.full((size, size), int(bg), dtype=int)
        for coord, color in zip(coords, palette):
            sprite[coord] = int(color)
        for coord in coords[len(palette) :]:
            sprite[coord] = int(random.choice(palette))
        return sprite
    return None


def _make_cycle_mapping(palette: list[int]) -> dict[int, int]:
    ordered = palette[:]
    random.shuffle(ordered)
    rotated = ordered[1:] + ordered[:1]
    return {src: dst for src, dst in zip(ordered, rotated)}


def _choose_profile() -> tuple[str, str]:
    global GENERATION_CALLS
    profile = GENERATION_PROFILES[GENERATION_CALLS % len(GENERATION_PROFILES)]
    GENERATION_CALLS += 1
    return profile


def _choose_cell_count_and_size(gap_mode: str) -> tuple[int, int]:
    if gap_mode == "double_mid":
        return 6, 4
    return random.choice([(5, 4), (5, 5), (6, 4)])


def _missing_indices(n_cells: int, gap_mode: str) -> set[int]:
    if gap_mode == "single_mid":
        center_choices = [n_cells // 2]
        if n_cells % 2 == 0:
            center_choices.append(n_cells // 2 - 1)
        return {random.choice(center_choices)}
    if gap_mode == "double_right":
        return {n_cells - 2, n_cells - 1}
    if gap_mode == "double_left":
        return {0, 1}
    if gap_mode == "double_mid":
        start = n_cells // 2 - 1
        return {start, start + 1}
    raise ValueError(f"unknown gap mode {gap_mode}")


def _nearest_visible(occupied: list[bool], idx: int) -> int:
    visible = [j for j, present in enumerate(occupied) if present]
    return min(visible, key=lambda j: (abs(j - idx), j))


def _composition_is_necessary(
    full_cells: list[np.ndarray],
    occupied: list[bool],
    geometry: str,
    mapping: dict[int, int],
    bg: int,
) -> bool:
    for idx, present in enumerate(occupied):
        if present:
            continue
        src_idx = _nearest_visible(occupied, idx)
        distance = idx - src_idx
        src = full_cells[src_idx]
        target = full_cells[idx]
        geometry_only = _apply_geometry_only_k(src, geometry, distance)
        color_only = _apply_color_only_k(src, mapping, bg, distance)
        if not np.array_equal(target, geometry_only) and not np.array_equal(target, color_only):
            return True
    return False


def _two_step_nontrivial(
    full_cells: list[np.ndarray],
    occupied: list[bool],
    geometry: str,
    mapping: dict[int, int],
    bg: int,
) -> bool:
    for idx, present in enumerate(occupied):
        if present:
            continue
        src_idx = _nearest_visible(occupied, idx)
        distance = abs(idx - src_idx)
        if distance != 2:
            continue
        if np.array_equal(full_cells[idx], full_cells[src_idx]):
            return False
        if np.array_equal(full_cells[idx], _apply_step_k(full_cells[src_idx], geometry, mapping, bg, 1)):
            return False
        return True
    return False


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    gap_mode, forced_geometry = _choose_profile()
    n_cells, cell_size = _choose_cell_count_and_size(gap_mode)

    bg, divider = random.sample(range(10), 2)
    available = [color for color in range(10) if color not in {bg, divider}]
    palette_size = random.choice([3, 4]) if cell_size == 5 else 3
    palette = random.sample(available, palette_size)
    mapping = _make_cycle_mapping(palette)

    sprite = _random_sprite(cell_size, bg, palette)
    if sprite is None:
        return None

    full_cells = [sprite]
    for _ in range(1, n_cells):
        full_cells.append(_apply_step(full_cells[-1], forced_geometry, mapping, bg))

    missing = _missing_indices(n_cells, gap_mode)
    occupied = [i not in missing for i in range(n_cells)]
    inp_cells = [cell.copy() if occupied[i] else np.full_like(cell, bg) for i, cell in enumerate(full_cells)]

    inp = _compose_grid(inp_cells, divider, bg)
    out = _compose_grid(full_cells, divider, bg)

    try:
        predicted = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(predicted, out):
        return None

    parsed_layout = _parse_cell_layout(inp, bg)
    parsed_cells = _extract_cells(inp, parsed_layout)
    parsed_occupied = _occupied_mask(parsed_cells, bg)
    if parsed_occupied != occupied:
        return None
    if len(_observed_adjacent_pairs(parsed_cells, parsed_occupied)) < 2:
        return None

    models = _valid_models(parsed_cells, parsed_occupied, bg)
    if len(models) != 1:
        return None
    if models[0]["geometry"] != forced_geometry:
        return None
    if models[0]["mapping"] != mapping:
        return None
    if not _composition_is_necessary(full_cells, occupied, forced_geometry, mapping, bg):
        return None
    if gap_mode in {"double_right", "double_left"} and not _two_step_nontrivial(full_cells, occupied, forced_geometry, mapping, bg):
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        built = _build_attempt()
        if built is not None:
            inp, _out = built
            return inp
    raise RuntimeError("failed to generate a valid input for fable/1/021/3")
