from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
MASK = 9
FOREGROUND_COLORS = (1, 2, 3)
GEN_GRID_SIZE = 5
MAX_GENERATION_ATTEMPTS = 400


def _rotation_orbits(size: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []

    for r in range(size):
        for c in range(size):
            if (r, c) in seen:
                continue
            orbit = []
            rr, cc = r, c
            for _ in range(4):
                orbit.append((rr, cc))
                rr, cc = cc, size - 1 - rr
            unique_orbit = tuple(sorted(set(orbit)))
            seen.update(unique_orbit)
            orbits.append(unique_orbit)

    orbits.sort(key=lambda orbit: (len(orbit), orbit))
    return orbits


def _complete_rotation_masks(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square grid"
    assert g.shape[0] % 2 == 1, "expected odd side length for a unique rotation center"

    out = g.copy()
    for orbit in _rotation_orbits(g.shape[0]):
        visible = {int(g[r, c]) for r, c in orbit if int(g[r, c]) != MASK}
        assert visible, "every rotation orbit must keep at least one visible pixel"
        assert len(visible) == 1, "masked orbit must reveal a unique completion color"
        fill = visible.pop()
        for r, c in orbit:
            if int(out[r, c]) == MASK:
                out[r, c] = fill
    return out


def _scale_self_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    size = int(g.shape[0])
    out = np.zeros((size * size, size * size), dtype=int)

    patterns = {
        int(color): np.where(g == color, int(color), BACKGROUND)
        for color in np.unique(g)
        if int(color) != BACKGROUND
    }

    for r, c in np.argwhere(g != BACKGROUND):
        color = int(g[r, c])
        out[r * size : (r + 1) * size, c * size : (c + 1) * size] = patterns[color]
    return out


def _scale_visible_only(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    size = int(g.shape[0])
    out = np.zeros((size * size, size * size), dtype=int)

    visible_colors = [
        int(color)
        for color in np.unique(g)
        if int(color) not in (BACKGROUND, MASK)
    ]
    patterns = {
        color: np.where(g == color, color, BACKGROUND)
        for color in visible_colors
    }

    for r in range(size):
        for c in range(size):
            color = int(g[r, c])
            if color in patterns:
                out[r * size : (r + 1) * size, c * size : (c + 1) * size] = patterns[color]
    return out


def _scale_orbit_local(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    size = int(g.shape[0])
    out = np.zeros((size * size, size * size), dtype=int)

    orbit_pattern: dict[tuple[int, int], np.ndarray] = {}
    for orbit in _rotation_orbits(size):
        color = int(g[orbit[0]])
        pattern = np.zeros_like(g)
        if color != BACKGROUND:
            for r, c in orbit:
                pattern[r, c] = color
        for coord in orbit:
            orbit_pattern[coord] = pattern

    for r, c in np.argwhere(g != BACKGROUND):
        out[r * size : (r + 1) * size, c * size : (c + 1) * size] = orbit_pattern[(int(r), int(c))]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the masked 4-fold rotational pattern, then expand each pixel into its color's full-grid self-mask.

    Input structure:
    - The input is a square grid whose unmasked cells are invariant under 90-degree rotation about the grid center.
    - Colors 1, 2, and 3 are foreground, 0 is background, and some cells are replaced by mask color 9.
    - Every masked cell belongs to a rotation orbit that still shows at least one unmasked copy of its true color.

    Rule:
    - For each mask cell, gather its 0/90/180/270-degree rotation orbit and replace the mask with the orbit's unique visible non-mask color.
    - After completion, expand the n x n grid into an (n*n) x (n*n) output by replacing each source cell with an n x n block.
    - A non-background block copies the completed grid's same-color mask: block position (i, j) is the source color exactly when the completed grid at (i, j) has that same color, otherwise it is 0.
    - Background cells produce all-0 blocks.

    Key insight:
    - The masks hide cells that are recoverable only from rotation symmetry, and those recovered cells change the global same-color mask used inside every expanded block of that color.
    """
    g = np.asarray(grid, dtype=int)
    completed = _complete_rotation_masks(g)
    return _scale_self_mask(completed)


GEN_ORBITS = _rotation_orbits(GEN_GRID_SIZE)
GEN_CENTER_ORBIT = next(orbit for orbit in GEN_ORBITS if len(orbit) == 1)
GEN_NONCENTER_ORBITS = [orbit for orbit in GEN_ORBITS if len(orbit) == 4]


def _build_clean_grid() -> tuple[np.ndarray, int]:
    family = random.choice(("medium", "sparse"))
    repeat_color = random.choice(FOREGROUND_COLORS)
    other_colors = [color for color in FOREGROUND_COLORS if color != repeat_color]
    shuffled_orbits = GEN_NONCENTER_ORBITS[:]
    random.shuffle(shuffled_orbits)

    orbit_colors: dict[tuple[tuple[int, int], ...], int] = {}
    if family == "medium":
        black_orbits = shuffled_orbits[:2]
        orbit_colors[shuffled_orbits[2]] = repeat_color
        orbit_colors[shuffled_orbits[3]] = repeat_color
        orbit_colors[shuffled_orbits[4]] = other_colors[0]
        orbit_colors[shuffled_orbits[5]] = other_colors[1]
        center_color = random.choice((BACKGROUND, repeat_color, other_colors[0], other_colors[1]))
    else:
        black_orbits = shuffled_orbits[:3]
        orbit_colors[shuffled_orbits[3]] = repeat_color
        orbit_colors[shuffled_orbits[4]] = other_colors[0]
        orbit_colors[shuffled_orbits[5]] = other_colors[1]
        center_color = repeat_color

    for orbit in black_orbits:
        orbit_colors[orbit] = BACKGROUND

    grid = np.zeros((GEN_GRID_SIZE, GEN_GRID_SIZE), dtype=int)
    for orbit, color in orbit_colors.items():
        for r, c in orbit:
            grid[r, c] = color
    center_r, center_c = GEN_CENTER_ORBIT[0]
    grid[center_r, center_c] = center_color
    return grid, repeat_color


def _pick_mask_counts(total_masks: int) -> list[int]:
    assert 3 <= total_masks <= 5
    anchor_max = min(3, total_masks)
    if total_masks >= 4 and random.random() < 0.7:
        anchor_count = random.randint(2, anchor_max)
    else:
        anchor_count = random.randint(1, anchor_max)

    counts = [anchor_count]
    remaining = total_masks - anchor_count
    while remaining > 0:
        count = random.randint(1, min(3, remaining))
        counts.append(count)
        remaining -= count
    return counts


def _apply_masks(clean: np.ndarray, repeat_color: int) -> np.ndarray:
    masked = clean.copy()
    repeat_orbits = [
        orbit
        for orbit in GEN_NONCENTER_ORBITS
        if int(clean[orbit[0]]) == repeat_color
    ]
    black_orbits = [
        orbit
        for orbit in GEN_NONCENTER_ORBITS
        if int(clean[orbit[0]]) == BACKGROUND
    ]
    assert repeat_orbits, "generator requires a repeated foreground color"

    total_masks = random.randint(3, 5)
    counts = _pick_mask_counts(total_masks)
    anchor_orbit = random.choice(repeat_orbits)

    remaining_orbits = [orbit for orbit in GEN_NONCENTER_ORBITS if orbit != anchor_orbit]
    chosen_orbits = [anchor_orbit]

    if len(counts) > 1 and len(repeat_orbits) > 1 and random.random() < 0.65:
        extra_repeat = random.choice([orbit for orbit in repeat_orbits if orbit != anchor_orbit])
        chosen_orbits.append(extra_repeat)
        remaining_orbits.remove(extra_repeat)

    if len(chosen_orbits) < len(counts) and black_orbits and random.random() < 0.65:
        black_choices = [orbit for orbit in black_orbits if orbit in remaining_orbits]
        if black_choices:
            black_orbit = random.choice(black_choices)
            chosen_orbits.append(black_orbit)
            remaining_orbits.remove(black_orbit)

    need = len(counts) - len(chosen_orbits)
    if need > 0:
        chosen_orbits.extend(random.sample(remaining_orbits, need))

    for orbit, count in zip(chosen_orbits, counts):
        for r, c in random.sample(list(orbit), count):
            masked[r, c] = MASK
    return masked


def _validate_instance(masked: np.ndarray, clean: np.ndarray, repeat_color: int) -> bool:
    if masked.shape != (GEN_GRID_SIZE, GEN_GRID_SIZE):
        return False
    if clean.shape != (GEN_GRID_SIZE, GEN_GRID_SIZE):
        return False
    if not np.array_equal(clean, np.rot90(clean, 1)):
        return False
    if set(map(int, np.unique(clean))) != {BACKGROUND, *FOREGROUND_COLORS}:
        return False
    if not np.all((masked == MASK) | (masked == clean)):
        return False

    mask_cells = np.argwhere(masked == MASK)
    if not 3 <= len(mask_cells) <= 5:
        return False
    if any(tuple(cell) == GEN_CENTER_ORBIT[0] for cell in mask_cells):
        return False

    masked_foreground = False
    repeated_foreground_mask = False
    for orbit in GEN_ORBITS:
        orbit_values = [int(masked[r, c]) for r, c in orbit]
        visible = {value for value in orbit_values if value != MASK}
        if not visible or len(visible) != 1:
            return False
        clean_color = int(clean[orbit[0]])
        if visible != {clean_color}:
            return False
        if orbit_values.count(MASK) >= len(orbit):
            return False
        if MASK in orbit_values and clean_color != BACKGROUND:
            masked_foreground = True
        if MASK in orbit_values and clean_color == repeat_color:
            repeated_foreground_mask = True

    if not masked_foreground or not repeated_foreground_mask:
        return False

    completed = _complete_rotation_masks(masked)
    if not np.array_equal(completed, clean):
        return False

    expected = _scale_self_mask(clean)
    if not np.array_equal(main(masked), expected):
        return False
    if np.array_equal(expected, _scale_visible_only(masked)):
        return False
    if np.array_equal(expected, _scale_orbit_local(clean)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        clean, repeat_color = _build_clean_grid()
        masked = _apply_masks(clean, repeat_color)
        if _validate_instance(masked, clean, repeat_color):
            return masked
    raise RuntimeError("failed to generate a valid rotational self-mask puzzle")
