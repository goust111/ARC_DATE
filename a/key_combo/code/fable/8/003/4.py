import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
INPUT_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ARC_LIMIT = 30
GEN_RETRY_LIMIT = 400


def bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1].copy()


def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    k = int(factor)
    assert k >= 1
    return np.kron(np.asarray(sprite, dtype=int), np.ones((k, k), dtype=int))


def find_nonblack_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if g[r, c] == BLACK or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != BLACK and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            components.append(
                {
                    "coords": coords,
                    "bbox": bbox_from_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return components


def _is_solid_rectangle(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    return int(component["area"]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _extract_pattern_blocks(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    components = find_nonblack_components(grid)
    assert len(components) == 2, "expected exactly two non-black rectangles"
    assert all(_is_solid_rectangle(comp) for comp in components), "each component must fill its bbox"
    components = sorted(components, key=lambda comp: int(comp["area"]))
    assert int(components[0]["area"]) < int(components[1]["area"]), "expected one strictly larger rectangle"
    small = crop_bbox(grid, components[0]["bbox"])
    big = crop_bbox(grid, components[1]["bbox"])
    return small, big


def _evaluate_candidates(small: np.ndarray, big: np.ndarray) -> list[dict]:
    candidates: list[dict] = []
    big_h, big_w = big.shape

    for rot in range(4):
        rotated = np.rot90(small, rot)
        small_h, small_w = rotated.shape
        if big_h % small_h != 0 or big_w % small_w != 0:
            continue
        k_h = big_h // small_h
        k_w = big_w // small_w
        if k_h != k_w or k_h < 2:
            continue
        k = int(k_h)
        mismatch = np.zeros((small_h, small_w), dtype=bool)
        for r in range(small_h):
            for c in range(small_w):
                block = big[r * k : (r + 1) * k, c * k : (c + 1) * k]
                mismatch[r, c] = not bool(np.all(block == int(rotated[r, c])))
        candidates.append(
            {
                "rotation": int(rot),
                "scale": k,
                "rotated_small": rotated,
                "mismatch_mask": mismatch,
                "mismatch_count": int(mismatch.sum()),
            }
        )

    assert candidates, "no valid scale/rotation candidate"
    return candidates


def _solve_details(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    small, big = _extract_pattern_blocks(g)
    candidates = _evaluate_candidates(small, big)
    best_count = min(int(candidate["mismatch_count"]) for candidate in candidates)
    best = [candidate for candidate in candidates if int(candidate["mismatch_count"]) == best_count]
    assert len(best) == 1, "expected a unique minimum-mismatch rotation"
    return {"small": small, "big": big, "best": best[0], "candidates": candidates}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Upscale and rotate the small pattern to match the large one, then report the changed blocks in the reduced coordinate system.

    Input structure:
    - Black cells are the background.
    - The foreground contains exactly two filled multicolor rectangles: one small pattern and one larger pattern whose side lengths are an integer multiple of the small one, up to a 90-degree rotation.
    - The larger rectangle comes from integer upscaling and rotating the small pattern, then recoloring a few whole kxk blocks.

    Rule:
    - Detect the two rectangles and treat the smaller one as the source pattern.
    - For each 90-degree rotation whose dimensions are compatible with the larger rectangle, infer the integer scale k from the side-length ratio and count blockwise mismatches after upscaling.
    - Select the unique rotation with the fewest mismatching kxk blocks, then output a grid in that rotated small-pattern frame with black for matching blocks and red for mismatching blocks.

    Key insight:
    - The meaningful comparison is not pixel-by-pixel on the original canvas but block-by-block after normalizing the hidden upscale-and-rotate relationship between the two rectangles.
    """
    details = _solve_details(np.asarray(grid, dtype=int))
    mismatch = details["best"]["mismatch_mask"]
    out = np.full(mismatch.shape, BLACK, dtype=int)
    out[mismatch] = RED
    return out


def _rotation_signature(arr: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    a = np.asarray(arr, dtype=int)
    return tuple(map(int, a.shape)), tuple(map(int, a.ravel()))


def _all_rotations_distinct(pattern: np.ndarray) -> bool:
    signatures = {_rotation_signature(np.rot90(pattern, rot)) for rot in range(4)}
    return len(signatures) == 4


def _min_same_shape_rotation_diff(pattern: np.ndarray) -> int:
    rotations = [np.rot90(pattern, rot) for rot in range(4)]
    diffs = []
    for i in range(4):
        for j in range(i + 1, 4):
            if rotations[i].shape == rotations[j].shape:
                diffs.append(int(np.count_nonzero(rotations[i] != rotations[j])))
    return min(diffs) if diffs else 0


def _sample_small_pattern(rng: random.Random, scale: int) -> np.ndarray:
    max_side = 5 if scale <= 3 else 4
    for _ in range(300):
        make_square = rng.random() < 0.4
        if make_square:
            side = rng.randint(2, max_side)
            h, w = side, side
        else:
            h = rng.randint(2, max_side)
            w = rng.randint(2, max_side)
            if h == w:
                continue

        palette_size = rng.randint(3, min(5, h * w, len(INPUT_COLORS) - 2))
        palette = rng.sample(INPUT_COLORS, palette_size)
        values = [rng.choice(palette) for _ in range(h * w)]
        pattern = np.array(values, dtype=int).reshape(h, w)
        if len(set(map(int, pattern.ravel()))) < 3:
            continue
        if not _all_rotations_distinct(pattern):
            continue
        if _min_same_shape_rotation_diff(pattern) < 2:
            continue
        return pattern
    raise AssertionError("failed to sample a non-degenerate small pattern")


def _sample_change_mask(shape: tuple[int, int], rng: random.Random) -> np.ndarray:
    h, w = map(int, shape)
    cell_count = h * w
    max_changes = min(cell_count - 1, max(1, cell_count // 2))
    min_changes = 1 if cell_count <= 4 else 2
    count = rng.randint(min_changes, max_changes)
    chosen = rng.sample([(r, c) for r in range(h) for c in range(w)], count)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    return mask


def _mutate_large_pattern(
    rotated_small: np.ndarray,
    scale: int,
    change_mask: np.ndarray,
    noise_colors: list[int],
    rng: random.Random,
) -> np.ndarray:
    big = scale_sprite(rotated_small, scale)
    for r, c in np.argwhere(change_mask):
        color = int(rng.choice(noise_colors))
        big[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale] = color
    return big


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _place_rectangles(small: np.ndarray, big: np.ndarray, rng: random.Random) -> np.ndarray:
    small_h, small_w = small.shape
    big_h, big_w = big.shape

    layout_options = [
        (max(small_h, big_h) + 2, small_w + big_w + 3),
        (small_h + big_h + 3, max(small_w, big_w) + 2),
    ]
    rng.shuffle(layout_options)

    for min_h, min_w in layout_options:
        if min_h > ARC_LIMIT or min_w > ARC_LIMIT:
            continue
        for _ in range(120):
            h = rng.randint(min_h, min(ARC_LIMIT, min_h + 6))
            w = rng.randint(min_w, min(ARC_LIMIT, min_w + 6))
            grid = np.zeros((h, w), dtype=int)

            sr = rng.randint(1, h - small_h - 1)
            sc = rng.randint(1, w - small_w - 1)
            br = rng.randint(1, h - big_h - 1)
            bc = rng.randint(1, w - big_w - 1)

            small_box = (sr, sc, sr + small_h - 1, sc + small_w - 1)
            big_box = (br, bc, br + big_h - 1, bc + big_w - 1)
            if _boxes_overlap(small_box, big_box, pad=1):
                continue

            grid[sr : sr + small_h, sc : sc + small_w] = small
            grid[br : br + big_h, bc : bc + big_w] = big
            return grid

    raise AssertionError("failed to place the two rectangles")


def _semantic_check(grid: np.ndarray, expected_rotation: int, expected_scale: int, expected_mask: np.ndarray) -> bool:
    try:
        details = _solve_details(grid)
    except AssertionError:
        return False

    best = details["best"]
    candidates = details["candidates"]
    if len(candidates) < 2:
        return False
    if int(best["rotation"]) != int(expected_rotation):
        return False
    if int(best["scale"]) != int(expected_scale):
        return False
    if not np.array_equal(best["mismatch_mask"], expected_mask):
        return False
    if int(best["mismatch_count"]) <= 0 or int(best["mismatch_count"]) >= expected_mask.size:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    scale = rng.randint(2, 4)
    small = _sample_small_pattern(rng, scale)
    used_colors = sorted(set(map(int, small.ravel())))
    noise_colors = [color for color in INPUT_COLORS if color not in used_colors]
    if len(noise_colors) < 2:
        return None

    true_rotation = rng.randint(0, 3)
    rotated_small = np.rot90(small, true_rotation)
    change_mask = _sample_change_mask(rotated_small.shape, rng)
    big = _mutate_large_pattern(rotated_small, scale, change_mask, noise_colors, rng)

    try:
        grid = _place_rectangles(small, big, rng)
    except AssertionError:
        return None

    if not _semantic_check(grid, true_rotation, scale, change_mask):
        return None

    try:
        produced = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(produced, np.where(change_mask, RED, BLACK)):
        return None
    if max(grid.shape) > ARC_LIMIT or max(produced.shape) > ARC_LIMIT:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid input")
