from __future__ import annotations

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDES = ("up", "down", "left", "right")
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

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
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _reflect_coords(
    coords: np.ndarray,
    bbox: tuple[int, int, int, int],
    side: str,
) -> np.ndarray:
    reflected = coords.copy()
    r0, c0, r1, c1 = bbox
    if side == "right":
        reflected[:, 1] = 2 * c1 + 1 - reflected[:, 1]
    elif side == "left":
        reflected[:, 1] = 2 * c0 - 1 - reflected[:, 1]
    elif side == "down":
        reflected[:, 0] = 2 * r1 + 1 - reflected[:, 0]
    elif side == "up":
        reflected[:, 0] = 2 * r0 - 1 - reflected[:, 0]
    else:
        raise ValueError(f"unknown side: {side}")
    return reflected


def _cup_bbox_inside_and_opening(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray, str]:
    cup_components = _components(grid == GREY)
    assert len(cup_components) == 1, "expected exactly one grey cup"
    bbox = _bbox_from_coords(cup_components[0])
    r0, c0, r1, c1 = bbox
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    inside = crop == BLACK
    assert inside.any(), "cup must contain a black cavity inside its bbox"

    side_hits = {
        "up": int(np.count_nonzero(inside[0, :])),
        "down": int(np.count_nonzero(inside[-1, :])),
        "left": int(np.count_nonzero(inside[:, 0])),
        "right": int(np.count_nonzero(inside[:, -1])),
    }
    openings = [side for side, count in side_hits.items() if count > 0]
    assert len(openings) == 1, "cup opening must lie on exactly one bbox side"
    return bbox, inside, openings[0]


def _sprite_coords_and_bbox(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    sprite_components = _components(grid == GREEN)
    assert len(sprite_components) == 1, "expected exactly one green sprite"
    coords = sprite_components[0]
    return coords, _bbox_from_coords(coords)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the cup interior, then mirror the distant sprite toward the side implied by the cup opening.

    Input structure:
    - A black background contains one grey cup-like container on the left and one green 4-connected sprite on the right.
    - The cup is a small open container: within its bounding box, black cavity cells touch exactly one side, which is the opening direction.
    - The sprite is separate from the cup and has enough surrounding black space for an adjacent mirror copy.

    Rule:
    - Detect which side of the cup bbox contains the opening.
    - Fill every black cell inside the cup bbox with yellow.
    - Use the same opening direction to place a mirror axis immediately outside the corresponding side of the sprite bbox.
    - Reflect the green sprite across that axis, paint the reflected copy red, and keep the original green sprite unchanged.

    Key insight:
    - The cup and sprite are spatially independent, but the cup opening direction remotely determines both the fill interpretation and the sprite's mirror axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    out = g.copy()

    cup_bbox, cup_inside, opening = _cup_bbox_inside_and_opening(g)
    r0, c0, r1, c1 = cup_bbox
    cup_view = out[r0 : r1 + 1, c0 : c1 + 1]
    cup_view[cup_inside] = YELLOW

    sprite_coords, sprite_bbox = _sprite_coords_and_bbox(g)
    mirrored = _reflect_coords(sprite_coords, sprite_bbox, opening)
    h, w = g.shape
    assert np.all((0 <= mirrored[:, 0]) & (mirrored[:, 0] < h))
    assert np.all((0 <= mirrored[:, 1]) & (mirrored[:, 1] < w))
    out[mirrored[:, 0], mirrored[:, 1]] = RED

    return out


def _rotate_open_up_cup(opening_width: int, side: str) -> np.ndarray:
    width = 2 + int(opening_width)
    cup = np.zeros((3, width), dtype=int)
    cup[:, 0] = GREY
    cup[:, -1] = GREY
    cup[-1, :] = GREY

    rotations = {"up": 0, "left": 1, "down": 2, "right": 3}
    return np.rot90(cup, k=rotations[side]).astype(int)


def _normalize_shape(coords: set[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    shifted = sorted((r - min_r, c - min_c) for r, c in coords)
    h = max(r for r, _ in shifted) + 1
    w = max(c for _, c in shifted) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c in shifted:
        patch[r, c] = GREEN
    return patch


def _is_non_symmetric(mask: np.ndarray) -> bool:
    binary = mask != BLACK
    return not np.array_equal(binary, np.flipud(binary)) and not np.array_equal(binary, np.fliplr(binary))


def _random_sprite(rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        target = int(rng.integers(3, 6))
        coords: set[tuple[int, int]] = {(0, 0)}
        frontier: set[tuple[int, int]] = {(1, 0), (-1, 0), (0, 1), (0, -1)}

        while len(coords) < target:
            candidates = [cell for cell in frontier if -2 <= cell[0] <= 2 and -2 <= cell[1] <= 2]
            if not candidates:
                break
            cell = candidates[int(rng.integers(len(candidates)))]
            frontier.remove(cell)
            coords.add(cell)
            r, c = cell
            for dr, dc in DIRS4:
                nb = (r + dr, c + dc)
                if nb not in coords:
                    frontier.add(nb)

        if len(coords) != target:
            continue

        patch = _normalize_shape(coords)
        if patch.shape[0] < 2 or patch.shape[1] < 2 or max(patch.shape) > 3:
            continue
        if len(_components(patch == GREEN)) != 1:
            continue
        if not _is_non_symmetric(patch):
            continue
        return patch

    raise RuntimeError("failed to generate a non-symmetric sprite")


def _can_place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    h, w = patch.shape
    if top < 0 or left < 0 or top + h > grid.shape[0] or left + w > grid.shape[1]:
        return False
    sub = grid[top : top + h, left : left + w]
    return not np.any((patch != BLACK) & (sub != BLACK))


def _absolute_coords(patch: np.ndarray, top: int, left: int) -> np.ndarray:
    coords = np.argwhere(patch != BLACK).copy()
    coords[:, 0] += int(top)
    coords[:, 1] += int(left)
    return coords


def _all_reflections_clear(
    grid: np.ndarray,
    sprite_coords: np.ndarray,
    sprite_bbox: tuple[int, int, int, int],
    cup_bbox: tuple[int, int, int, int],
) -> bool:
    h, w = grid.shape
    cup_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == GREY)}
    reflected_sets: list[frozenset[tuple[int, int]]] = []
    for side in SIDES:
        reflected = _reflect_coords(sprite_coords, sprite_bbox, side)
        if not np.all((0 <= reflected[:, 0]) & (reflected[:, 0] < h) & (0 <= reflected[:, 1]) & (reflected[:, 1] < w)):
            return False
        cells = frozenset((int(r), int(c)) for r, c in reflected)
        if cells & cup_cells:
            return False
        if any(grid[r, c] != BLACK for r, c in cells):
            return False
        reflected_sets.append(cells)

    if len(set(reflected_sets)) != 4:
        return False

    cup_c1 = int(cup_bbox[3])
    left_reflected_min_c = min(c for _, c in reflected_sets[2])
    return left_reflected_min_c - cup_c1 >= 2


def _sample_cup_position(
    grid_shape: tuple[int, int],
    cup_patch: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    h, w = grid_shape
    ch, cw = cup_patch.shape
    candidates = []
    max_c = max(1, w // 3 - cw)
    for r in range(1, h - ch):
        for c in range(1, max_c + 1):
            candidates.append((r, c))
    if not candidates:
        return None
    idx = int(rng.integers(len(candidates)))
    return candidates[idx]


def _sample_sprite_position(
    grid: np.ndarray,
    sprite_patch: np.ndarray,
    cup_bbox: tuple[int, int, int, int],
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    h, w = grid.shape
    sh, sw = sprite_patch.shape
    cup_c1 = int(cup_bbox[3])
    candidates = []

    top_lo = sh
    top_hi = h - 2 * sh
    left_lo = max(sw, w // 2, cup_c1 + 2 * sw + 2)
    left_hi = w - 2 * sw
    if top_lo > top_hi or left_lo > left_hi:
        return None

    for top in range(top_lo, top_hi + 1):
        for left in range(left_lo, left_hi + 1):
            if left - cup_c1 - 1 < 3:
                continue
            if not _can_place_patch(grid, sprite_patch, top, left):
                continue
            coords = _absolute_coords(sprite_patch, top, left)
            bbox = (top, left, top + sh - 1, left + sw - 1)
            if _all_reflections_clear(grid, coords, bbox, cup_bbox):
                candidates.append((top, left))

    if not candidates:
        return None
    idx = int(rng.integers(len(candidates)))
    return candidates[idx]


def _validate_generated_input(grid: np.ndarray) -> None:
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert set(map(int, np.unique(grid))).issubset({BLACK, GREEN, GREY})

    cup_bbox, cup_inside, opening = _cup_bbox_inside_and_opening(grid)
    assert cup_inside.any()
    assert 1 <= int(np.count_nonzero(cup_inside[0, :] if opening == "up" else cup_inside[-1, :] if opening == "down" else cup_inside[:, 0] if opening == "left" else cup_inside[:, -1])) <= 2

    sprite_coords, sprite_bbox = _sprite_coords_and_bbox(grid)
    sprite_crop = grid[sprite_bbox[0] : sprite_bbox[2] + 1, sprite_bbox[1] : sprite_bbox[3] + 1]
    assert _is_non_symmetric(sprite_crop)

    assert _all_reflections_clear(grid, sprite_coords, sprite_bbox, cup_bbox)

    out = main(grid)
    assert int(np.count_nonzero(out == YELLOW)) >= int(np.count_nonzero(cup_inside))
    red_components = _components(out == RED)
    assert len(red_components) == 1


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = int(rng.integers(12, 19))
        width = int(rng.integers(16, 23))
        grid = np.zeros((height, width), dtype=int)

        opening = SIDES[int(rng.integers(len(SIDES)))]
        opening_width = int(rng.integers(1, 3))
        cup_patch = _rotate_open_up_cup(opening_width, opening)
        cup_pos = _sample_cup_position(grid.shape, cup_patch, rng)
        if cup_pos is None:
            continue
        cup_r, cup_c = cup_pos
        grid[cup_r : cup_r + cup_patch.shape[0], cup_c : cup_c + cup_patch.shape[1]] = np.maximum(
            grid[cup_r : cup_r + cup_patch.shape[0], cup_c : cup_c + cup_patch.shape[1]],
            cup_patch,
        )
        cup_bbox = (cup_r, cup_c, cup_r + cup_patch.shape[0] - 1, cup_c + cup_patch.shape[1] - 1)

        sprite_patch = _random_sprite(rng)
        sprite_pos = _sample_sprite_position(grid, sprite_patch, cup_bbox, rng)
        if sprite_pos is None:
            continue
        sprite_r, sprite_c = sprite_pos
        sprite_mask = sprite_patch != BLACK
        grid[sprite_r : sprite_r + sprite_patch.shape[0], sprite_c : sprite_c + sprite_patch.shape[1]][sprite_mask] = GREEN

        try:
            _validate_generated_input(grid)
        except Exception:
            continue
        return grid

    raise RuntimeError("failed to generate a valid cup-controlled mirror puzzle")
