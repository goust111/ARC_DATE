import random

import numpy as np


BG = 0
FRAME = 3
MASK = 6
PALETTE = (1, 2, 4, 5, 7, 8, 9)
INTERIOR_SIDES = (6, 8, 10)
MAX_GENERATION_ATTEMPTS = 400
SEED_ATTEMPTS = 200
VALIDATION_SAMPLES = 24


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _scale2(sprite: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(sprite, 2, axis=0), 2, axis=1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for coords in _components(grid == FRAME):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(patch, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(patch[border] == FRAME):
            continue
        if np.any(patch[~border] == FRAME):
            continue
        if int(border.sum()) != len(coords):
            continue
        candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, "expected one green rectangular frame"
    return candidates[0]


def _repair_horizontal(seed: np.ndarray) -> np.ndarray:
    repaired = seed.copy()
    h, _w = repaired.shape
    for r, c in np.argwhere(repaired == MASK):
        mr = h - 1 - int(r)
        color = int(repaired[mr, int(c)])
        assert color not in (BG, FRAME, MASK), "each mask must have one visible horizontal mirror partner"
        repaired[int(r), int(c)] = color
    assert not np.any(repaired == MASK), "all masked cells must be repaired"
    return repaired


def _repair_vertical_or_none(seed: np.ndarray) -> np.ndarray | None:
    repaired = seed.copy()
    _h, w = repaired.shape
    for r, c in np.argwhere(repaired == MASK):
        mc = w - 1 - int(c)
        color = int(repaired[int(r), mc])
        if color in (BG, FRAME, MASK):
            return None
        repaired[int(r), int(c)] = color
    return repaired


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop to the green frame, repair the masked seed by horizontal symmetry, then scale it by 2 to refill the frame.

    Input structure:
    - A black background grid contains exactly one green rectangular frame with a one-pixel border.
    - The frame interior has even height and width, and its upper-left quarter contains a smaller colored seed pattern.
    - The seed uses non-black, non-green colors, but 1-3 seed pixels are replaced by pink masks; stray pixels outside the frame are irrelevant.

    Rule:
    - Crop the output to the green frame's bounding box, keeping the frame border.
    - Inside the cropped frame, take the tight bounding box of the non-black, non-green interior seed, including pink masked cells.
    - Repair each pink seed cell by copying the color from its mirror position across the seed's horizontal midline, then scale the repaired seed by 2 in both directions.
    - Replace the cropped frame interior with that 2x enlarged repaired seed; it fills the interior exactly.

    Key insight:
    - Pink marks missing cells in the small seed itself, so the seed must be reconstructed at its original scale before the frame interior can be rebuilt by 2x pixel replication.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_bbox = _find_frame(g)
    out = _crop_bbox(g, frame_bbox)
    assert out.shape[0] >= 4 and out.shape[1] >= 4, "frame must have an interior"

    interior = out[1:-1, 1:-1]
    seed_bbox = _bbox((interior != BG) & (interior != FRAME))
    assert seed_bbox is not None, "expected one seed pattern inside the frame"
    seed = _crop_bbox(interior, seed_bbox)
    repaired = _repair_horizontal(seed)
    scaled = _scale2(repaired)
    assert scaled.shape == interior.shape, "repaired seed must double to the full frame interior"

    out[1:-1, 1:-1] = scaled
    return out


def _random_complete_seed(seed_h: int, seed_w: int) -> np.ndarray:
    top_rows = (seed_h + 1) // 2
    for _ in range(SEED_ATTEMPTS):
        palette = random.sample(PALETTE, random.randint(2, min(5, len(PALETTE))))
        top = np.random.choice(palette, size=(top_rows, seed_w))
        seed = np.empty((seed_h, seed_w), dtype=int)
        seed[:top_rows, :] = top
        for r in range(seed_h // 2):
            seed[seed_h - 1 - r, :] = seed[r, :]

        if len(np.unique(seed)) < 2:
            continue
        if np.array_equal(seed, np.fliplr(seed)):
            continue
        if len({tuple(row) for row in seed.tolist()}) < 2:
            continue
        if len({tuple(col) for col in seed.T.tolist()}) < 2:
            continue
        return seed
    raise RuntimeError("failed to sample a non-degenerate horizontally symmetric seed")


def _masked_seed(complete_seed: np.ndarray) -> np.ndarray:
    seed = complete_seed.copy()
    h, w = seed.shape
    candidates = [(r, c) for r in range(h // 2) for c in range(w)]
    k = random.randint(1, min(3, len(candidates)))
    chosen = random.sample(candidates, k)
    for r, c in chosen:
        seed[r, c] = MASK
    return seed


def _direct_scaled_output(frame_outer_h: int, frame_outer_w: int, masked_seed: np.ndarray) -> np.ndarray:
    out = np.full((frame_outer_h, frame_outer_w), FRAME, dtype=int)
    out[1:-1, 1:-1] = _scale2(masked_seed)
    return out


def _vertical_repair_output_or_none(frame_outer_h: int, frame_outer_w: int, masked_seed: np.ndarray) -> np.ndarray | None:
    repaired = _repair_vertical_or_none(masked_seed)
    if repaired is None:
        return None
    out = np.full((frame_outer_h, frame_outer_w), FRAME, dtype=int)
    out[1:-1, 1:-1] = _scale2(repaired)
    return out


def _isolated_noise_positions(
    grid_shape: tuple[int, int],
    forbidden: np.ndarray,
    count: int,
) -> list[tuple[int, int]]:
    h, w = grid_shape
    positions: list[tuple[int, int]] = []
    for _ in range(count * 20):
        r = random.randrange(h)
        c = random.randrange(w)
        if forbidden[r, c]:
            continue
        ok = True
        for rr, cc in positions:
            if max(abs(rr - r), abs(cc - c)) <= 1:
                ok = False
                break
        if ok:
            positions.append((r, c))
            forbidden[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = True
            if len(positions) == count:
                break
    return positions


def _expected_output(complete_seed: np.ndarray, interior_h: int, interior_w: int) -> np.ndarray:
    out = np.full((interior_h + 2, interior_w + 2), FRAME, dtype=int)
    out[1:-1, 1:-1] = _scale2(complete_seed)
    return out


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    complete_seed: np.ndarray,
    masked_seed: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    r0, c0, r1, c1 = frame_bbox
    cropped = inp[r0 : r1 + 1, c0 : c1 + 1]
    interior = cropped[1:-1, 1:-1]
    seed_bbox = _bbox((interior != BG) & (interior != FRAME))
    if seed_bbox is None or _crop_bbox(interior, seed_bbox).shape != complete_seed.shape:
        return False

    if not np.array_equal(_repair_horizontal(masked_seed), complete_seed):
        return False
    if np.array_equal(_direct_scaled_output(*expected.shape, masked_seed), expected):
        return False

    wrong_axis = _vertical_repair_output_or_none(*expected.shape, masked_seed)
    if wrong_axis is not None and np.array_equal(wrong_axis, expected):
        return False

    if np.array_equal(complete_seed, np.fliplr(complete_seed)):
        return False

    for r, c in np.argwhere(masked_seed == MASK):
        mr = masked_seed.shape[0] - 1 - int(r)
        if complete_seed[int(r), int(c)] != complete_seed[mr, int(c)]:
            return False
        if masked_seed[mr, int(c)] == MASK:
            return False

    return True


def _attempt_generate() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]] | None:
    interior_h = random.choice(INTERIOR_SIDES)
    interior_w = random.choice(INTERIOR_SIDES)
    frame_outer_h = interior_h + 2
    frame_outer_w = interior_w + 2
    seed_h = interior_h // 2
    seed_w = interior_w // 2

    canvas_h = random.randint(max(18, frame_outer_h + 4), 28)
    canvas_w = random.randint(max(18, frame_outer_w + 4), 28)
    top = random.randint(1, canvas_h - frame_outer_h - 1)
    left = random.randint(1, canvas_w - frame_outer_w - 1)
    frame_bbox = (top, left, top + frame_outer_h - 1, left + frame_outer_w - 1)

    complete_seed = _random_complete_seed(seed_h, seed_w)
    masked_seed = _masked_seed(complete_seed)
    expected = _expected_output(complete_seed, interior_h, interior_w)

    grid = np.zeros((canvas_h, canvas_w), dtype=int)
    grid[top : top + frame_outer_h, left : left + frame_outer_w] = BG
    grid[top, left : left + frame_outer_w] = FRAME
    grid[top + frame_outer_h - 1, left : left + frame_outer_w] = FRAME
    grid[top : top + frame_outer_h, left] = FRAME
    grid[top : top + frame_outer_h, left + frame_outer_w - 1] = FRAME
    grid[top + 1 : top + 1 + seed_h, left + 1 : left + 1 + seed_w] = masked_seed

    forbidden = np.zeros_like(grid, dtype=bool)
    forbidden[max(0, top - 1) : min(canvas_h, top + frame_outer_h + 1), max(0, left - 1) : min(canvas_w, left + frame_outer_w + 1)] = True
    noise_count = random.randint(0, 8)
    for r, c in _isolated_noise_positions(grid.shape, forbidden, noise_count):
        grid[r, c] = random.choice((0, 1, 2, 3, 4, 5, 6, 7, 8, 9))

    if not _semantic_checks(grid, expected, complete_seed, masked_seed, frame_bbox):
        return None
    return grid, expected, complete_seed, masked_seed, frame_bbox


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        attempt = _attempt_generate()
        if attempt is not None:
            return attempt[0]
    raise RuntimeError("failed to generate a valid input")


if __name__ == "__main__":
    sigs = set()
    for _ in range(VALIDATION_SAMPLES):
        sample = generate_input()
        sigs.add((sample.shape, sample.tobytes()))
        result = main(sample)
        assert sample.ndim == 2 and result.ndim == 2
        assert max(sample.shape) <= 30 and max(result.shape) <= 30
    print("samples", len(sigs))
