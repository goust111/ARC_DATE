from __future__ import annotations

import random

import numpy as np


OCCLUDER_COLOR = 9
MAX_GENERATION_ATTEMPTS = 500


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _tight_crop(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != int(bg))
    assert len(coords) > 0
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _filled_rectangle_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    area = (int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1)
    if area != len(coords):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _rot90_pos(r: int, c: int, size: int) -> tuple[int, int]:
    return int(c), int(size - 1 - r)


def _orbit_of(r: int, c: int, size: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    rr, cc = int(r), int(c)
    for _ in range(4):
        if (rr, cc) not in seen:
            seen.add((rr, cc))
            out.append((rr, cc))
        rr, cc = _rot90_pos(rr, cc, size)
    return out


def _rotation_orbits(size: int) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[list[tuple[int, int]]] = []
    for r in range(size):
        for c in range(size):
            if (r, c) in seen:
                continue
            orbit = _orbit_of(r, c, size)
            seen.update(orbit)
            orbits.append(sorted(orbit))
    return orbits


def _rectangular_color_candidates(grid: np.ndarray, bg: int) -> list[int]:
    colors = sorted(set(map(int, grid.flat)) - {int(bg)})
    return [color for color in colors if _filled_rectangle_bbox(grid == color) is not None]


def _restore_candidate(grid: np.ndarray, bg: int, occ_color: int) -> np.ndarray | None:
    visible = (grid != int(bg)) & (grid != int(occ_color))
    if not np.any(visible):
        return None

    (r0, c0), (r1, c1) = np.argwhere(visible).min(axis=0), np.argwhere(visible).max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height != width:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    size = int(height)
    restored = patch.copy()
    filled_from_symmetry = 0

    for lr, lc in np.argwhere(patch == int(occ_color)):
        orbit = _orbit_of(int(lr), int(lc), size)
        orbit_colors = {
            int(patch[rr, cc])
            for rr, cc in orbit
            if int(patch[rr, cc]) not in (int(bg), int(occ_color))
        }
        if len(orbit_colors) > 1:
            return None
        if orbit_colors:
            restored[int(lr), int(lc)] = next(iter(orbit_colors))
            filled_from_symmetry += 1
        else:
            restored[int(lr), int(lc)] = int(bg)

    if filled_from_symmetry == 0:
        return None
    if np.any(restored == int(occ_color)):
        return None
    if not np.array_equal(restored, np.rot90(restored, 1)):
        return None
    if not np.any(restored != int(bg)):
        return None
    return _tight_crop(restored, int(bg))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a 4-fold rotation-symmetric hidden pattern from a solid occluding rectangle and return its tight crop.

    Input structure:
    - bg is the most common color.
    - One larger multicolor pattern uses colors other than bg and the occluder color, and its tight bbox is a square that is invariant under 90-degree rotation about its center.
    - A single unique-color solid rectangle lies on top of that pattern and hides part of it.

    Rule:
    - Identify the non-background color whose cells form one filled axis-aligned rectangle; treat it as the occluder.
    - Take the bbox of all remaining non-bg cells; this square bbox defines the hidden pattern's rotation frame and center.
    - For each occluded cell, rotate its position by 90, 180, and 270 degrees inside that square and copy the visible non-occluder color from the same orbit, or leave it bg if the whole orbit is bg there.
    - After filling the hidden cells, crop tightly around the restored non-bg pattern and output that crop.

    Key insight:
    - The solid rectangle is just an occluder; the visible pixels reveal the square rotation frame and provide the orbit colors needed to reconstruct the hidden part.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    solutions: list[np.ndarray] = []
    for color in _rectangular_color_candidates(g, bg):
        restored = _restore_candidate(g, bg, int(color))
        if restored is not None:
            solutions.append(restored)

    assert solutions
    sigs = {(tuple(arr.shape), arr.tobytes()) for arr in solutions}
    assert len(sigs) == 1
    return solutions[0].astype(int)


def _find_orbit_index(orbits: list[list[tuple[int, int]]], cell: tuple[int, int]) -> int:
    for idx, orbit in enumerate(orbits):
        if cell in orbit:
            return idx
    raise ValueError(f"cell {cell} not found in orbit list")


def _orbit_touches_mask(orbit: list[tuple[int, int]], mask: np.ndarray) -> bool:
    h, w = mask.shape
    for r, c in orbit:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
                return True
    return False


def _build_hidden_patch(size: int, bg: int, palette: list[int]) -> np.ndarray | None:
    orbits = _rotation_orbits(size)
    family = random.choice(["ring", "growth"])

    selected: set[int] = set()
    if family == "ring":
        for idx, orbit in enumerate(orbits):
            if any(r in (0, size - 1) or c in (0, size - 1) for r, c in orbit):
                selected.add(idx)
        target_area = random.randint(
            max(sum(len(orbits[idx]) for idx in selected) + 4, int(size * size * 0.34)),
            int(size * size * 0.58),
        )
        adj_weight, free_weight = 5.0, 1.0
    else:
        seed_cells = [
            (0, 0),
            (0, size // 2 if size % 2 else size // 2 - 1),
            (1, min(size - 2, max(1, size // 3))),
        ]
        for cell in seed_cells:
            selected.add(_find_orbit_index(orbits, cell))
        target_area = random.randint(
            max(sum(len(orbits[idx]) for idx in selected) + 8, int(size * size * 0.24)),
            int(size * size * 0.48),
        )
        adj_weight, free_weight = 8.0, 0.35

    mask = np.zeros((size, size), dtype=bool)
    for idx in selected:
        for r, c in orbits[idx]:
            mask[r, c] = True

    current_area = int(mask.sum())
    attempts = 0
    while current_area < target_area and attempts < 4 * len(orbits):
        attempts += 1
        candidates: list[int] = []
        weights: list[float] = []
        for idx, orbit in enumerate(orbits):
            if idx in selected or len(orbit) == 1:
                continue
            candidates.append(idx)
            weights.append(adj_weight if _orbit_touches_mask(orbit, mask) else free_weight)
        if not candidates:
            break
        pick = random.choices(candidates, weights=weights, k=1)[0]
        selected.add(pick)
        for r, c in orbits[pick]:
            mask[r, c] = True
        current_area = int(mask.sum())

    if not np.any(mask):
        return None
    if int(mask.sum()) < int(size * size * 0.22) or int(mask.sum()) > int(size * size * 0.65):
        return None

    orbit_ids = list(selected)
    random.shuffle(orbit_ids)
    n_colors = min(len(palette), min(len(orbit_ids), random.randint(3, 5)))
    if n_colors < 2:
        return None
    use_palette = random.sample(palette, n_colors)

    patch = np.full((size, size), int(bg), dtype=int)
    for idx, orbit_id in enumerate(orbit_ids):
        color = int(use_palette[idx]) if idx < n_colors else int(random.choice(use_palette))
        for r, c in orbits[orbit_id]:
            patch[r, c] = color

    if len(set(map(int, patch[patch != int(bg)]))) < 2:
        return None
    if _tight_crop(patch, bg).shape != (size, size):
        return None
    return patch


def _rect_mode_matches(mode: str, r0: int, c0: int, r1: int, c1: int, size: int) -> bool:
    mid_lo = (size - 1) // 2
    mid_hi = size // 2
    span_h = r0 <= mid_lo <= r1 and r0 <= mid_hi <= r1
    span_v = c0 <= mid_lo <= c1 and c0 <= mid_hi <= c1
    if mode == "corner":
        return not span_h and not span_v
    if mode == "bar":
        return span_h != span_v
    if mode == "central":
        return span_h and span_v
    return False


def _choose_occluder_rect(patch: np.ndarray, bg: int) -> tuple[int, int, int, int] | None:
    size = int(patch.shape[0])
    mask = patch != int(bg)
    orbits = _rotation_orbits(size)
    modes = ["corner", "bar", "central"]

    for _ in range(300):
        mode = random.choice(modes)
        h = random.randint(2, size - 2)
        w = random.randint(2, size - 2)
        if h >= size - 1 or w >= size - 1:
            continue
        r0 = random.randint(1, size - h - 1)
        c0 = random.randint(1, size - w - 1)
        r1 = r0 + h - 1
        c1 = c0 + w - 1
        if not _rect_mode_matches(mode, r0, c0, r1, c1, size):
            continue

        rect = np.zeros_like(mask, dtype=bool)
        rect[r0 : r1 + 1, c0 : c1 + 1] = True
        overlap = int(np.sum(mask & rect))
        if overlap < max(3, size // 3):
            continue
        if overlap > max(6, int(np.sum(mask) * 0.40)):
            continue

        valid = True
        for orbit in orbits:
            colored_cells = sum(1 for rr, cc in orbit if bool(mask[rr, cc]))
            hidden_cells = sum(1 for rr, cc in orbit if bool(rect[rr, cc]) and bool(mask[rr, cc]))
            if colored_cells > 0 and hidden_cells == colored_cells:
                valid = False
                break
        if valid:
            return int(r0), int(c0), int(r1), int(c1)
    return None


def _bg_is_unique_most_common(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)[::-1]
    if int(values[int(order[0])]) != int(bg):
        return False
    if len(order) > 1 and int(counts[int(order[0])]) == int(counts[int(order[1])]):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        size = random.randint(9, 19)
        bg = random.choice([color for color in range(10) if color != OCCLUDER_COLOR])
        palette = [color for color in range(10) if color not in (bg, OCCLUDER_COLOR)]
        patch = _build_hidden_patch(size, bg, palette)
        if patch is None:
            continue

        rect = _choose_occluder_rect(patch, bg)
        if rect is None:
            continue
        rr0, cc0, rr1, cc1 = rect

        max_h = 30 - size
        max_w = 30 - size
        if max_h < 4 or max_w < 4:
            continue
        grid_h = size + random.randint(4, max_h)
        grid_w = size + random.randint(4, max_w)
        top = random.randint(0, grid_h - size)
        left = random.randint(0, grid_w - size)

        grid = np.full((grid_h, grid_w), int(bg), dtype=int)
        grid[top : top + size, left : left + size] = patch
        grid[top + rr0 : top + rr1 + 1, left + cc0 : left + cc1 + 1] = int(OCCLUDER_COLOR)

        if not _bg_is_unique_most_common(grid, bg):
            continue
        if _rectangular_color_candidates(grid, bg) != [OCCLUDER_COLOR]:
            continue

        target = _tight_crop(patch, bg)
        try:
            solved = main(grid)
        except AssertionError:
            continue
        if not np.array_equal(solved, target):
            continue
        return grid.astype(int)

    raise RuntimeError("failed to generate a valid input within the retry budget")
