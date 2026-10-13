from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

PATTERN_SIZE = 3
MIN_SCALE = 2
MAX_SCALE = 7
MAX_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _frame_from_grid(grid: np.ndarray) -> tuple[tuple[int, int, int, int], int]:
    candidates: list[tuple[tuple[int, int, int, int], int]] = []
    for coords in _components4(grid == RED):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != w or h < 8:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(patch[0, :] == RED):
            continue
        if not np.all(patch[-1, :] == RED):
            continue
        if not np.all(patch[:, 0] == RED):
            continue
        if not np.all(patch[:, -1] == RED):
            continue
        if not np.all(patch[1:-1, 1:-1] == BLACK):
            continue
        if int(np.count_nonzero(patch == RED)) != 4 * h - 4:
            continue
        inner = h - 2
        if inner % PATTERN_SIZE != 0:
            continue
        scale = inner // PATTERN_SIZE
        if scale < MIN_SCALE:
            continue
        candidates.append(((r0, c0, r1, c1), scale))

    assert len(candidates) == 1, "expected a unique red hollow square frame"
    return candidates[0]


def _pattern_from_grid(grid: np.ndarray, frame_box: tuple[int, int, int, int]) -> np.ndarray:
    blue_box = _bbox(grid == BLUE)
    assert blue_box is not None, "expected a blue 3x3 pattern"
    r0, c0, r1, c1 = blue_box
    assert (r1 - r0 + 1, c1 - c0 + 1) == (PATTERN_SIZE, PATTERN_SIZE), "blue pattern bbox must be 3x3"

    fr0, fc0, fr1, fc1 = frame_box
    assert r1 < fr0 or fr1 < r0 or c1 < fc0 or fc1 < c0, "pattern must not overlap the frame"

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert set(np.unique(patch)).issubset({BLACK, BLUE}), "pattern area must contain only black and blue"
    blue_count = int(np.count_nonzero(patch == BLUE))
    black_count = int(np.count_nonzero(patch == BLACK))
    assert 2 <= blue_count <= 5, "pattern needs 2..5 blue pixels"
    assert black_count >= 2, "pattern needs visible complement cells"
    return patch


def _scaled_binary(mask: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((scale, scale), dtype=int))


def _render_interior(pattern: np.ndarray, scale: int) -> np.ndarray:
    source = pattern == BLUE
    complement = pattern == BLACK
    interior_size = PATTERN_SIZE * scale
    stamp = _scaled_binary(complement, scale) * BLUE
    interior = np.zeros((interior_size, interior_size), dtype=int)

    for pr, pc in np.argwhere(source):
        rr = int(pr) * scale
        cc = int(pc) * scale
        h = interior_size - rr
        w = interior_size - cc
        if h <= 0 or w <= 0:
            continue
        patch = stamp[:h, :w]
        interior[rr:, cc:] = np.maximum(interior[rr:, cc:], patch)
    return interior


def _render_output(pattern: np.ndarray, scale: int) -> np.ndarray:
    interior = _render_interior(pattern, scale)
    out = np.full((interior.shape[0] + 2, interior.shape[1] + 2), RED, dtype=int)
    out[1:-1, 1:-1] = interior
    return out


def _count_clipped_stamps(pattern: np.ndarray) -> int:
    count = 0
    for pr, pc in np.argwhere(pattern == BLUE):
        if int(pr) != 0 or int(pc) != 0:
            count += 1
    return count


def _simple_alternatives(pattern: np.ndarray, scale: int) -> Iterable[np.ndarray]:
    source = _scaled_binary(pattern == BLUE, scale) * BLUE
    complement = _scaled_binary(pattern == BLACK, scale) * BLUE
    top_left_only = np.zeros_like(source)
    stamp = complement
    top_left_only[: stamp.shape[0], : stamp.shape[1]] = stamp
    dense_fill = np.full_like(source, BLUE)
    yield source
    yield complement
    yield top_left_only
    yield dense_fill


def _validate_case(grid: np.ndarray, pattern: np.ndarray, scale: int) -> None:
    out = main(grid)
    expected = _render_output(pattern, scale)
    assert np.array_equal(out, expected), "oracle mismatch"

    interior = expected[1:-1, 1:-1]
    assert np.any(interior == BLUE), "output interior must contain blue"
    assert np.any(interior == BLACK), "output interior must retain black"
    assert _count_clipped_stamps(pattern) >= 1, "at least one stamp must require cropping"
    assert int(np.count_nonzero(pattern == BLUE)) >= 2, "need multiple placement triggers"

    for alt in _simple_alternatives(pattern, scale):
        assert not np.array_equal(interior, alt), "case degenerates to a simpler one-step rule"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the red frame size to scale a self-complement stamp and crop it to the frame.

    Input structure:
    - The grid has black background, one red hollow square frame, and one separate 3x3 blue/black pattern.
    - The frame interior side length is exactly 3S for some integer S >= 2.
    - The blue cells of the 3x3 pattern define the coarse placement positions, while its black cells define the stamp mask.

    Rule:
    - Detect the unique red frame and compute S from its inner side length divided by 3.
    - Take the 3x3 pattern's complement mask: every black cell becomes part of a 3x3 binary stamp, every blue cell becomes empty.
    - Scale that complement stamp by S, and for every blue source cell place one cropped copy with its top-left aligned to the corresponding SxS block inside the frame interior.
    - Union all placed blue pixels, keep other interior cells black, and return only the cropped frame region including the red border.

    Key insight:
    - The same 3x3 pattern plays two roles at once: blue cells choose which coarse blocks launch a stamp, while black cells define the stamp's fine shape; the red frame silently encodes the scale.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_box, scale = _frame_from_grid(g)
    pattern = _pattern_from_grid(g, frame_box)
    return _render_output(pattern, scale)


def _pattern_candidates() -> list[np.ndarray]:
    patterns: list[np.ndarray] = []
    for bits in itertools.product((0, 1), repeat=PATTERN_SIZE * PATTERN_SIZE):
        arr = np.array(bits, dtype=int).reshape(PATTERN_SIZE, PATTERN_SIZE)
        blue_count = int(arr.sum())
        if not (2 <= blue_count <= 5):
            continue
        if int((arr == 0).sum()) < 2:
            continue
        box = _bbox(arr == 1)
        if box != (0, 0, PATTERN_SIZE - 1, PATTERN_SIZE - 1):
            continue
        if _count_clipped_stamps(arr) < 1:
            continue
        patterns.append(arr.astype(int))
    return patterns


_PATTERNS = _pattern_candidates()


def _sample_canvas_shape(rng: np.random.Generator, outer: int) -> tuple[int, int]:
    min_side = max(outer + 4, 14)
    max_side = 30
    h = int(rng.integers(min_side, max_side + 1))
    w = int(rng.integers(min_side, max_side + 1))
    return h, w


def _far_enough(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], gap: int = 2) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    scale = int(rng.integers(MIN_SCALE, MAX_SCALE + 1))
    pattern = _PATTERNS[int(rng.integers(0, len(_PATTERNS)))]
    expected = _render_output(pattern, scale)

    interior = expected[1:-1, 1:-1]
    if np.array_equal(interior, _scaled_binary(pattern == BLUE, scale) * BLUE):
        return None
    if np.array_equal(interior, _scaled_binary(pattern == BLACK, scale) * BLUE):
        return None

    outer = expected.shape[0]
    h, w = _sample_canvas_shape(rng, outer)
    grid = np.zeros((h, w), dtype=int)

    frame_r = int(rng.integers(0, h - outer + 1))
    frame_c = int(rng.integers(0, w - outer + 1))
    frame_box = (frame_r, frame_c, frame_r + outer - 1, frame_c + outer - 1)

    pattern_slots: list[tuple[int, int]] = []
    for pr in range(h - PATTERN_SIZE + 1):
        for pc in range(w - PATTERN_SIZE + 1):
            candidate = (pr, pc, pr + PATTERN_SIZE - 1, pc + PATTERN_SIZE - 1)
            if _far_enough(candidate, frame_box, gap=2):
                pattern_slots.append((pr, pc))
    if not pattern_slots:
        return None
    pat_r, pat_c = pattern_slots[int(rng.integers(0, len(pattern_slots)))]

    frame = np.full((outer, outer), RED, dtype=int)
    frame[1:-1, 1:-1] = BLACK
    grid[frame_r : frame_r + outer, frame_c : frame_c + outer] = frame
    grid[pat_r : pat_r + PATTERN_SIZE, pat_c : pat_c + PATTERN_SIZE] = np.where(pattern == 1, BLUE, BLACK)

    try:
        _validate_case(grid, np.where(pattern == 1, BLUE, BLACK), scale)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
