import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
USABLE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SEED_BOX = (0, 0, 3, 3)
SEED_SLICE = (slice(1, 3), slice(1, 3))
MAX_GENERATION_ATTEMPTS = 400


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _make_border_masks() -> tuple[np.ndarray, ...]:
    masks: list[np.ndarray] = []

    top = np.zeros((4, 4), dtype=bool)
    top[0, :] = True
    bottom = np.zeros((4, 4), dtype=bool)
    bottom[3, :] = True
    left = np.zeros((4, 4), dtype=bool)
    left[:, 0] = True
    right = np.zeros((4, 4), dtype=bool)
    right[:, 3] = True

    masks.extend(
        [
            top | left | right,
            bottom | left | right,
            top | bottom | left,
            top | bottom | right,
        ]
    )
    masks.extend(
        [
            top | left,
            top | right,
            bottom | left,
            bottom | right,
        ]
    )
    return tuple(masks)


ALLOWED_BORDER_MASKS = _make_border_masks()


def _mirror_expand(seed: np.ndarray) -> np.ndarray:
    top = np.concatenate((seed, np.fliplr(seed)), axis=1)
    return np.concatenate((top, np.flipud(top)), axis=0)


def _find_seed_info(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    grey_components = _components(grid == GREY)
    candidates: list[tuple[tuple[int, int, int, int], np.ndarray]] = []

    for coords in grey_components:
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        if (r1 - r0 + 1, c1 - c0 + 1) != (4, 4):
            continue

        local = np.zeros((4, 4), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        if not any(np.array_equal(local, allowed) for allowed in ALLOWED_BORDER_MASKS):
            continue

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        seed = patch[SEED_SLICE]
        if seed.shape != (2, 2):
            continue
        if np.any(seed == BLACK) or np.any(seed == GREY):
            continue
        candidates.append((bbox, seed.copy()))

    assert len(candidates) == 1, "expected exactly one grey border marking one 2x2 seed"
    return candidates[0]


def _all_black_windows(grid: np.ndarray, size: int = 4) -> list[tuple[int, int]]:
    h, w = grid.shape
    hits: list[tuple[int, int]] = []
    for r in range(h - size + 1):
        for c in range(w - size + 1):
            if np.all(grid[r : r + size, c : c + size] == BLACK):
                hits.append((r, c))
    return hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a four-quadrant mirror tile from the grey-marked 2x2 seed and use it to fill every 4x4 black hole.

    Input structure:
    - A noisy color field contains one 2x2 seed whose four cells are non-black and non-grey.
    - That seed is identified only by one connected grey U-shaped or L-shaped border component around a 4x4 corner patch.
    - Elsewhere the grid contains one or more isolated 4x4 all-black square holes inside the colored field.

    Rule:
    - Locate the unique grey border and crop the 2x2 seed inside its 4x4 bounding box.
    - Form a 4x4 pattern by placing the seed in the upper-left, mirroring it left-right for the upper-right, then mirroring the whole top half downward.
    - Scan the grid from top-left to bottom-right and replace every fully black 4x4 window with that mirrored 4x4 pattern.
    - Finally erase the grey border pixels to black while leaving the seed itself and all other non-hole pixels unchanged.

    Key insight:
    - The fill tile is not present directly in the input: it must be constructed by recognizing the grey-framed 2x2 seed and expanding it by quadrant reflections so that its size exactly matches each 4x4 hole.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    seed_bbox, seed = _find_seed_info(g)
    pattern = _mirror_expand(seed)

    out = g.copy()
    for r, c in _all_black_windows(g, size=4):
        if np.all(out[r : r + 4, c : c + 4] == BLACK):
            out[r : r + 4, c : c + 4] = pattern

    r0, c0, r1, c1 = seed_bbox
    patch = out[r0 : r1 + 1, c0 : c1 + 1]
    patch[g[r0 : r1 + 1, c0 : c1 + 1] == GREY] = BLACK
    return out


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _random_seed() -> np.ndarray:
    while True:
        color_count = random.choice((2, 3))
        palette = random.sample(list(USABLE_COLORS), color_count)
        flat = [random.choice(palette) for _ in range(4)]
        if len(set(flat)) != color_count:
            continue
        seed = np.array(flat, dtype=int).reshape(2, 2)
        if np.array_equal(seed, np.fliplr(seed)) and np.array_equal(seed, np.flipud(seed)):
            continue
        return seed


def _noise_palette(seed: np.ndarray) -> list[int]:
    seed_colors = sorted(set(map(int, seed.ravel())))
    extra_pool = [c for c in USABLE_COLORS if c not in seed_colors]
    target_size = random.randint(max(4, len(seed_colors)), min(6, len(USABLE_COLORS)))
    extras = random.sample(extra_pool, target_size - len(seed_colors))
    palette = seed_colors + extras
    random.shuffle(palette)
    return palette


def _pick_boxes(
    h: int,
    w: int,
    count: int,
    forbidden: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    candidates = [(r, c) for r in range(1, h - 4) for c in range(1, w - 4)]
    random.shuffle(candidates)
    placed: list[tuple[int, int]] = []
    used_boxes = list(forbidden)

    for r, c in candidates:
        box = (r, c, r + 3, c + 3)
        if all(_boxes_separated(box, other, pad=1) for other in used_boxes):
            placed.append((r, c))
            used_boxes.append(box)
            if len(placed) == count:
                return placed
    raise ValueError("failed to place enough 4x4 boxes")


def _ensure_seed_colors_repeat(grid: np.ndarray, seed: np.ndarray, reserved: np.ndarray) -> None:
    free = np.argwhere(~reserved)
    if free.size == 0:
        return
    for color in sorted(set(map(int, seed.ravel()))):
        if np.count_nonzero(grid == color) > np.count_nonzero(seed == color):
            continue
        rr, cc = free[random.randrange(len(free))]
        grid[int(rr), int(cc)] = int(color)


def _validate_instance(
    grid: np.ndarray,
    seed: np.ndarray,
    border_mask: np.ndarray,
    holes: list[tuple[int, int]],
) -> None:
    detected_bbox, detected_seed = _find_seed_info(grid)
    assert np.array_equal(detected_seed, seed), "seed detection must recover the planted seed"
    assert detected_bbox == SEED_BOX, "generator keeps the seed marker in the corner patch"

    windows = _all_black_windows(grid, size=4)
    assert windows == sorted(holes), "input must contain exactly the intended full-black 4x4 holes"

    expected = grid.copy()
    pattern = _mirror_expand(seed)
    for r, c in holes:
        expected[r : r + 4, c : c + 4] = pattern
    expected[:4, :4][border_mask] = BLACK

    out = main(grid)
    assert np.array_equal(out, expected), "main() must match the puzzle semantics"
    assert not np.any(out == GREY), "grey border pixels must be erased in the output"


def _gen_attempt() -> np.ndarray:
    h = random.randint(14, 20)
    w = random.randint(14, 20)
    seed = _random_seed()
    border_mask = random.choice(ALLOWED_BORDER_MASKS)
    palette = _noise_palette(seed)

    grid = np.random.choice(np.array(palette, dtype=int), size=(h, w))
    grid = np.asarray(grid, dtype=int)

    grid[:4, :4] = BLACK
    grid[:4, :4][border_mask] = GREY
    grid[SEED_SLICE] = seed

    hole_count = random.randint(1, 3)
    decoy_count = random.choice((0, 1, 1, 2))
    forbidden = [SEED_BOX]
    holes = _pick_boxes(h, w, hole_count, forbidden)
    forbidden.extend((r, c, r + 3, c + 3) for r, c in holes)
    decoys = _pick_boxes(h, w, decoy_count, forbidden) if decoy_count else []

    for r, c in holes:
        grid[r : r + 4, c : c + 4] = BLACK

    for r, c in decoys:
        grid[r : r + 4, c : c + 4] = BLACK
        dr, dc = random.randrange(4), random.randrange(4)
        grid[r + dr, c + dc] = random.choice(palette)

    reserved = np.zeros((h, w), dtype=bool)
    reserved[:4, :4] = True
    for r, c in holes + decoys:
        reserved[r : r + 4, c : c + 4] = True
    _ensure_seed_colors_repeat(grid, seed, reserved)

    _validate_instance(grid, seed, border_mask, sorted(holes))
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            return _gen_attempt()
        except (AssertionError, ValueError):
            continue
    raise RuntimeError("failed to generate a valid input for puzzle 9")
