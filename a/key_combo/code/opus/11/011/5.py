from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
OCCLUDER = 6
AVAILABLE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
SYMMETRIES = ("horizontal", "vertical", "rot180", "rot90")
INFERENCE_ORDER = ("rot90", "rot180", "horizontal", "vertical")
FRAME_GAP = 1
PATTERN_CELLS_MIN = 4
PATTERN_CELLS_MAX = 8
GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_hollow_square(grid: np.ndarray, coords: np.ndarray, color: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    if side_h != side_w or side_h < 5:
        return False
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    ring = np.zeros((side_h, side_w), dtype=bool)
    ring[0, :] = True
    ring[-1, :] = True
    ring[:, 0] = True
    ring[:, -1] = True
    return np.array_equal(crop == color, ring)


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        for coords in _components(grid == color):
            if not _is_hollow_square(grid, coords, color):
                continue
            r0, c0, r1, c1 = _bbox(coords)
            frames.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "interior": grid[r0 + 1 : r1, c0 + 1 : c1].copy(),
                }
            )
    assert len(frames) == 3, "expected exactly three hollow square frames"
    return sorted(frames, key=lambda item: item["bbox"])


def _palette_signature(interior: np.ndarray) -> tuple[int, ...]:
    colors = {int(v) for v in interior.flat if int(v) not in (BG, OCCLUDER)}
    return tuple(sorted(colors))


def _apply_transform(arr: np.ndarray, symmetry: str) -> np.ndarray:
    if symmetry == "horizontal":
        return np.flipud(arr)
    if symmetry == "vertical":
        return np.fliplr(arr)
    if symmetry == "rot180":
        return np.rot90(arr, 2)
    if symmetry == "rot90":
        return np.rot90(arr, -1)
    raise ValueError(f"unknown symmetry: {symmetry}")


def _map_coord(r: int, c: int, size: int, symmetry: str) -> tuple[int, int]:
    if symmetry == "horizontal":
        return size - 1 - r, c
    if symmetry == "vertical":
        return r, size - 1 - c
    if symmetry == "rot180":
        return size - 1 - r, size - 1 - c
    if symmetry == "rot90":
        return c, size - 1 - r
    raise ValueError(f"unknown symmetry: {symmetry}")


def _holds_symmetry(arr: np.ndarray, symmetry: str) -> bool:
    return np.array_equal(arr, _apply_transform(arr, symmetry))


def _repair(interior: np.ndarray, symmetry: str) -> np.ndarray:
    out = interior.copy()
    size = out.shape[0]
    assert out.shape == (size, size)
    while True:
        changed = False
        magenta_cells = np.argwhere(out == OCCLUDER)
        for r, c in magenta_cells:
            rr, cc = _map_coord(int(r), int(c), size, symmetry)
            if out[rr, cc] == OCCLUDER:
                continue
            out[r, c] = out[rr, cc]
            changed = True
        if not changed:
            break
    return out


def _infer_symmetry(full: np.ndarray, damaged: np.ndarray) -> str:
    valid: list[str] = []
    for symmetry in INFERENCE_ORDER:
        if not _holds_symmetry(full, symmetry):
            continue
        if np.array_equal(_repair(damaged, symmetry), full):
            valid.append(symmetry)
    assert len(valid) == 1, "training pair must imply exactly one symmetry rule"
    return valid[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the training pair's symmetry type and use it to repair the magenta-occluded test pattern.

    Input structure:
    - The grid has a black background and exactly three colored hollow square frames of distinct colors.
    - Each frame encloses a square interior pattern; two frames form a training pair because their visible non-black, non-magenta palettes match.
    - In the training pair, one interior is complete and one is the same pattern with a magenta rectangular occluder; the remaining frame is the test case.

    Rule:
    - Detect the three hollow frames and crop their interiors.
    - Identify the training pair by matching interior color sets while ignoring black background and magenta occlusion.
    - From the complete training interior, infer which symmetry explains it: horizontal mirror, vertical mirror, 180-degree rotation, or 90-degree rotation.
    - Verify that the damaged training interior repairs to the complete one under that symmetry, then apply the same repair to the test interior.
    - Return only the repaired test interior, without the surrounding frame.

    Key insight:
    - The missing pixels are not filled by a fixed symmetry; the symmetry type itself must be learned from the paired training examples and then transferred to the test frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _find_frames(g)
    signatures: dict[tuple[int, ...], list[int]] = {}
    for idx, frame in enumerate(frames):
        signatures.setdefault(_palette_signature(frame["interior"]), []).append(idx)

    pair_groups = [indices for indices in signatures.values() if len(indices) == 2]
    assert len(pair_groups) == 1, "expected exactly one palette-matched training pair"
    train_pair = pair_groups[0]
    test_idx = next(idx for idx in range(3) if idx not in train_pair)

    first = frames[train_pair[0]]["interior"]
    second = frames[train_pair[1]]["interior"]
    if np.count_nonzero(first == OCCLUDER) == 0 and np.count_nonzero(second == OCCLUDER) > 0:
        full, damaged = first, second
    elif np.count_nonzero(second == OCCLUDER) == 0 and np.count_nonzero(first == OCCLUDER) > 0:
        full, damaged = second, first
    else:
        raise AssertionError("training pair must contain exactly one complete interior")

    symmetry = _infer_symmetry(full, damaged)
    return _repair(frames[test_idx]["interior"], symmetry)


def _orbit_cells(size: int, symmetry: str) -> list[list[tuple[int, int]]]:
    unseen = {(r, c) for r in range(size) for c in range(size)}
    orbits: list[list[tuple[int, int]]] = []
    while unseen:
        start = next(iter(unseen))
        orbit: list[tuple[int, int]] = []
        current = start
        while current not in orbit:
            orbit.append(current)
            current = _map_coord(current[0], current[1], size, symmetry)
        for cell in orbit:
            unseen.discard(cell)
        orbits.append(sorted(orbit))
    random.shuffle(orbits)
    return orbits


def _choose_orbit_subset(orbits: list[list[tuple[int, int]]], target: int) -> list[list[tuple[int, int]]] | None:
    order = list(range(len(orbits)))
    random.shuffle(order)

    def dfs(pos: int, total: int, chosen: list[int]) -> list[int] | None:
        if total == target:
            return chosen.copy()
        if total > target or pos >= len(order):
            return None
        for idx in range(pos, len(order)):
            orbit_idx = order[idx]
            orbit_size = len(orbits[orbit_idx])
            if total + orbit_size > target:
                continue
            chosen.append(orbit_idx)
            found = dfs(idx + 1, total + orbit_size, chosen)
            if found is not None:
                return found
            chosen.pop()
        return None

    found = dfs(0, 0, [])
    if found is None:
        return None
    return [orbits[idx] for idx in found]


def _symmetry_profile(pattern: np.ndarray) -> set[str]:
    return {symmetry for symmetry in SYMMETRIES if _holds_symmetry(pattern, symmetry)}


def _profile_matches(symmetry: str, profile: set[str]) -> bool:
    if symmetry == "rot90":
        return profile == {"rot90", "rot180"}
    return profile == {symmetry}


def _nonzero_bbox_size(pattern: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(pattern != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _generate_pattern(size: int, symmetry: str, palette: tuple[int, ...]) -> np.ndarray | None:
    orbits = _orbit_cells(size, symmetry)
    for _ in range(200):
        target_cells = random.randint(PATTERN_CELLS_MIN, PATTERN_CELLS_MAX)
        chosen = _choose_orbit_subset(orbits, target_cells)
        if chosen is None or len(chosen) < len(palette):
            continue

        pattern = np.zeros((size, size), dtype=int)
        colors = list(palette) + [random.choice(palette) for _ in range(len(chosen) - len(palette))]
        random.shuffle(colors)
        for orbit, color in zip(chosen, colors):
            for r, c in orbit:
                pattern[r, c] = int(color)

        if set(int(v) for v in pattern.flat if int(v) != BG) != set(palette):
            continue
        if not _profile_matches(symmetry, _symmetry_profile(pattern)):
            continue
        span_h, span_w = _nonzero_bbox_size(pattern)
        if min(span_h, span_w) < 3:
            continue
        return pattern
    return None


def _all_colors_visible_outside_mask(pattern: np.ndarray, mask: np.ndarray) -> bool:
    visible = {int(v) for v in pattern[~mask] if int(v) != BG}
    full = {int(v) for v in pattern.flat if int(v) != BG}
    return visible == full


def _find_occluded_version(
    pattern: np.ndarray,
    symmetry: str,
    *,
    require_unique: bool,
) -> np.ndarray | None:
    size = pattern.shape[0]
    rect_shapes = [(2, 3), (3, 2)]
    candidates: list[np.ndarray] = []
    for rect_h, rect_w in rect_shapes:
        for r0 in range(size - rect_h + 1):
            for c0 in range(size - rect_w + 1):
                mask = np.zeros_like(pattern, dtype=bool)
                mask[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
                if np.count_nonzero(pattern[mask] != BG) < 2:
                    continue
                if not _all_colors_visible_outside_mask(pattern, mask):
                    continue

                damaged = pattern.copy()
                damaged[mask] = OCCLUDER
                if not np.array_equal(_repair(damaged, symmetry), pattern):
                    continue

                if require_unique:
                    valid = []
                    for candidate in INFERENCE_ORDER:
                        if not _holds_symmetry(pattern, candidate):
                            continue
                        if np.array_equal(_repair(damaged, candidate), pattern):
                            valid.append(candidate)
                    if valid != [symmetry]:
                        continue

                candidates.append(damaged)

    if not candidates:
        return None
    return random.choice(candidates)


def _frame_patch(frame_color: int, interior: np.ndarray) -> np.ndarray:
    size = interior.shape[0] + 2
    patch = np.zeros((size, size), dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    patch[1:-1, 1:-1] = interior
    return patch


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    max_side = max(patch.shape[0] for patch in patches)
    min_dim = max(20, max_side + 2)
    height = random.randint(min_dim, 28)
    width = random.randint(min_dim, 28)
    canvas = np.zeros((height, width), dtype=int)
    boxes: list[tuple[int, int, int, int]] = []

    order = list(range(len(patches)))
    random.shuffle(order)
    placements: dict[int, tuple[int, int]] = {}

    for idx in order:
        patch = patches[idx]
        side = patch.shape[0]
        positions = [
            (r, c)
            for r in range(height - side + 1)
            for c in range(width - side + 1)
        ]
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            box = (r0, c0, r0 + side - 1, c0 + side - 1)
            if all(_boxes_separated(box, prev, FRAME_GAP) for prev in boxes):
                canvas[r0 : r0 + side, c0 : c0 + side] = patch
                boxes.append(box)
                placements[idx] = (r0, c0)
                placed = True
                break
        if not placed:
            return None

    return canvas


def _pick_disjoint_palettes(frame_colors: list[int], symmetry: str) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    pool = [color for color in AVAILABLE_COLORS if color not in frame_colors]
    random.shuffle(pool)
    train_size = 2 if symmetry == "rot90" else random.choice([2, 3])
    test_size = 2 if symmetry == "rot90" else random.choice([2, 3])
    if len(pool) < train_size + test_size:
        return None
    train_palette = tuple(sorted(pool[:train_size]))
    test_palette = tuple(sorted(pool[train_size : train_size + test_size]))
    if set(train_palette) == set(test_palette):
        return None
    return train_palette, test_palette


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    symmetry = random.choice(list(SYMMETRIES))
    train_size = random.randint(5, 7)
    test_size = random.randint(5, 7)
    frame_colors = random.sample(AVAILABLE_COLORS, 3)
    palettes = _pick_disjoint_palettes(frame_colors, symmetry)
    if palettes is None:
        return None
    train_palette, test_palette = palettes

    full_train = _generate_pattern(train_size, symmetry, train_palette)
    full_test = _generate_pattern(test_size, symmetry, test_palette)
    if full_train is None or full_test is None:
        return None

    damaged_train = _find_occluded_version(full_train, symmetry, require_unique=True)
    damaged_test = _find_occluded_version(full_test, symmetry, require_unique=False)
    if damaged_train is None or damaged_test is None:
        return None

    train_pair = [full_train, damaged_train]
    random.shuffle(train_pair)
    interiors = [train_pair[0], train_pair[1], damaged_test]
    patches = [
        _frame_patch(frame_color, interior)
        for frame_color, interior in zip(frame_colors, interiors)
    ]

    grid = _place_patches(patches)
    if grid is None:
        return None
    return grid, full_test


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        case = _generate_case()
        if case is None:
            continue
        grid, expected = case
        try:
            if np.array_equal(main(grid), expected):
                return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid ARC analogy-symmetry instance")
