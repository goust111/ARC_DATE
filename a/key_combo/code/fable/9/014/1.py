from __future__ import annotations

import hashlib
import random
from collections import deque

import numpy as np


BLACK = 0
VISIBLE_COLORS = tuple(range(1, 10))
OBSERVED_ROTATIONS = (0, 1, 2, 3)
INVERSE_ROTATIONS = (0, 3, 2, 1)
SHAPE_MODES = ("line", "rect", "elbow", "blob")
MAX_GENERATION_ATTEMPTS = 500

_GENERATOR_CALLS = 0
_SEEN_OUTPUT_SIGNATURES: set[str] = set()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_nonblack(grid: np.ndarray) -> np.ndarray:
    box = _bbox(grid != BLACK)
    assert box is not None, "expected at least one non-black cell"
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue: deque[tuple[int, int]] = deque([start])
    h, w = mask.shape
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _split_quadrants(grid: np.ndarray) -> tuple[int, list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h == w and h % 2 == 0 and h >= 4, "expected a 2n x 2n square grid"
    n = h // 2
    quadrants = [
        g[:n, :n],
        g[:n, n:],
        g[n:, :n],
        g[n:, n:],
    ]
    return n, quadrants


def _normalize_quadrants(quadrants: list[np.ndarray]) -> list[np.ndarray]:
    return [np.rot90(quadrant, turns) for quadrant, turns in zip(quadrants, INVERSE_ROTATIONS)]


def _consensus_grid(normalized: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(normalized, axis=0)
    n = int(stack.shape[1])
    out = np.zeros((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            values, counts = np.unique(stack[:, r, c], return_counts=True)
            best = int(np.argmax(counts))
            assert int(counts[best]) >= 3, "each cell must have a 3-of-4 consensus"
            assert int(np.count_nonzero(counts == counts[best])) == 1, "consensus must be unique"
            out[r, c] = int(values[best])
    return out


def _extract_anomaly(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    _n, quadrants = _split_quadrants(grid)
    normalized = _normalize_quadrants(quadrants)
    consensus = _consensus_grid(normalized)
    mismatch_masks = [quadrant != consensus for quadrant in normalized]
    mismatch_counts = [int(mask.sum()) for mask in mismatch_masks]
    bad_quadrants = [idx for idx, count in enumerate(mismatch_counts) if count > 0]
    assert len(bad_quadrants) == 1, "expected exactly one tampered quadrant"
    tampered_idx = int(bad_quadrants[0])
    assert all(
        count == 0 for idx, count in enumerate(mismatch_counts) if idx != tampered_idx
    ), "three quadrants must agree perfectly after inverse rotation"
    tampered_mask = mismatch_masks[tampered_idx]
    assert _is_connected(tampered_mask), "tampered cells must form one connected shape"
    sprite = np.where(tampered_mask, normalized[tampered_idx], BLACK)
    return tampered_idx, consensus, tampered_mask, sprite


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal the tampered quadrant inside a rotational 2x2 mosaic and output the anomaly in base orientation.

    Input structure:
    - The input is a 2n x 2n square split into four n x n quadrants.
    - The upper-left quadrant is a base pattern; the other three are its 90, 180, and 270 degree counterclockwise rotations.
    - Exactly one quadrant has a connected set of pixels recolored using colors already present in the mosaic.

    Rule:
    - Inverse-rotate each quadrant back to the base orientation and compare them cell by cell.
    - The three matching quadrants define the consensus image; the unique disagreeing quadrant is the tampered one.
    - Keep only the tampered quadrant's disagreeing pixels, still using their rewritten colors, and output that difference map cropped to its bounding box with black elsewhere.

    Key insight:
    - The anomaly cannot be found by looking for a novel color, because the tamper uses the existing palette; only rotational consensus exposes which pixels were rewritten, and the output direction is the tampered quadrant rotated back to the base frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    _tampered_idx, _consensus, _mask, sprite = _extract_anomaly(g)
    return _crop_nonblack(sprite)


def _random_base(n: int, palette: list[int]) -> np.ndarray | None:
    grid = np.array([random.choice(palette) for _ in range(n * n)], dtype=int).reshape(n, n)
    forced_cells = random.sample(range(n * n), len(palette))
    for flat, color in zip(forced_cells, palette):
        grid.flat[flat] = color
    rotations = [np.rot90(grid, turns) for turns in OBSERVED_ROTATIONS]
    if len({rotation.tobytes() for rotation in rotations}) != 4:
        return None
    return grid


def _sample_line_mask(n: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    length = random.randint(2, max(2, n - 1))
    if random.random() < 0.5:
        r = random.randrange(n)
        c0 = random.randint(0, n - length)
        mask[r, c0 : c0 + length] = True
    else:
        c = random.randrange(n)
        r0 = random.randint(0, n - length)
        mask[r0 : r0 + length, c] = True
    return mask


def _sample_rect_mask(n: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    h = random.randint(2, max(2, min(n - 1, n // 2 + 1)))
    w = random.randint(2, max(2, min(n - 1, n // 2 + 1)))
    r0 = random.randint(0, n - h)
    c0 = random.randint(0, n - w)
    mask[r0 : r0 + h, c0 : c0 + w] = True
    return mask


def _sample_elbow_mask(n: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    arm_r = random.randint(2, max(2, n - 1))
    arm_c = random.randint(2, max(2, n - 1))
    r0 = random.randint(0, n - arm_r)
    c0 = random.randint(0, n - arm_c)
    corner = random.choice(
        [
            (r0, c0, 1, 1),
            (r0, c0 + arm_c - 1, 1, -1),
            (r0 + arm_r - 1, c0, -1, 1),
            (r0 + arm_r - 1, c0 + arm_c - 1, -1, -1),
        ]
    )
    cr, cc, sr, sc = corner
    for step in range(arm_r):
        mask[cr + step * sr, cc] = True
    for step in range(arm_c):
        mask[cr, cc + step * sc] = True
    return mask


def _sample_blob_mask(n: int) -> np.ndarray:
    target = random.randint(3, max(3, min(n * n // 3, n + 3)))
    cells = {(random.randrange(n), random.randrange(n))}
    while len(cells) < target:
        r, c = random.choice(tuple(cells))
        neighbors = []
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in cells:
                neighbors.append((nr, nc))
        if not neighbors:
            continue
        cells.add(random.choice(neighbors))
    mask = np.zeros((n, n), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _sample_connected_mask(n: int, mode: str) -> np.ndarray | None:
    samplers = {
        "line": _sample_line_mask,
        "rect": _sample_rect_mask,
        "elbow": _sample_elbow_mask,
        "blob": _sample_blob_mask,
    }
    for _ in range(80):
        mask = samplers[mode](n)
        area = int(mask.sum())
        if area < 2 or area > max(2, n * n // 3):
            continue
        if not _is_connected(mask):
            continue
        box = _bbox(mask)
        assert box is not None
        r0, c0, r1, c1 = box
        if r0 == r1 and c0 == c1:
            continue
        return mask
    return None


def _recolor_tamper(base: np.ndarray, mask: np.ndarray, palette: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    tampered = np.array(base, copy=True)
    coords = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    preferred_pool = random.sample(palette, random.randint(1, min(3, len(palette))))
    for r, c in coords:
        choices = [color for color in preferred_pool if color != int(base[r, c])]
        if not choices:
            choices = [color for color in palette if color != int(base[r, c])]
        tampered[r, c] = random.choice(choices)
    if np.array_equal(tampered, base):
        return None
    sprite = np.where(mask, tampered, BLACK)
    cropped = _crop_nonblack(sprite)
    if any(np.array_equal(cropped, np.rot90(cropped, turns)) for turns in (1, 2, 3)):
        return None
    return tampered, cropped


def _assemble_mosaic(base: np.ndarray, tampered_base: np.ndarray, tampered_idx: int) -> np.ndarray:
    quadrants = []
    for idx, turns in enumerate(OBSERVED_ROTATIONS):
        source = tampered_base if idx == tampered_idx else base
        quadrants.append(np.rot90(source, turns))
    top = np.concatenate(quadrants[:2], axis=1)
    bottom = np.concatenate(quadrants[2:], axis=1)
    return np.concatenate([top, bottom], axis=0)


def _output_signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _attempt_generation(call_index: int) -> np.ndarray | None:
    n = random.randint(4, 12)
    palette = random.sample(list(VISIBLE_COLORS), random.randint(4, 7))
    base = _random_base(n, palette)
    if base is None:
        return None

    tampered_idx = call_index % 4
    mode = SHAPE_MODES[call_index % len(SHAPE_MODES)]
    mask = _sample_connected_mask(n, mode)
    if mask is None:
        return None

    recolored = _recolor_tamper(base, mask, palette)
    if recolored is None:
        return None
    tampered_base, expected = recolored

    observed_sprite = np.rot90(np.where(mask, tampered_base, BLACK), OBSERVED_ROTATIONS[tampered_idx])
    if tampered_idx != 0 and np.array_equal(_crop_nonblack(observed_sprite), expected):
        return None

    grid = _assemble_mosaic(base, tampered_base, tampered_idx)
    try:
        found_idx, consensus, found_mask, sprite = _extract_anomaly(grid)
    except AssertionError:
        return None

    if found_idx != tampered_idx:
        return None
    if not np.array_equal(found_mask, mask):
        return None
    if not np.array_equal(consensus, base):
        return None
    if not np.array_equal(_crop_nonblack(sprite), expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    signature = _output_signature(expected)
    if signature in _SEEN_OUTPUT_SIGNATURES:
        return None
    _SEEN_OUTPUT_SIGNATURES.add(signature)
    return grid


def generate_input() -> np.ndarray:
    global _GENERATOR_CALLS
    for attempt in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generation(_GENERATOR_CALLS + attempt)
        if grid is not None:
            _GENERATOR_CALLS += 1
            return grid
    raise RuntimeError("failed to generate a valid rotational anomaly mosaic")
