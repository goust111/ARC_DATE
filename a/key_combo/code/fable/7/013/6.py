"""ARC task: infer a hidden center from sparse parity-ring samples."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
RED = 2
YELLOW = 4
FRAME_COLORS = (1, 3, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 500


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _rectangular_frame(grid: np.ndarray, bg: int) -> tuple[int, int, int, int, int]:
    """Return the unique largest exact one-cell monochrome rectangular frame."""
    candidates: list[tuple[int, int, int, int, int, int]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        if len(coords) < 8:
            continue
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        perimeter = 2 * (r1 - r0 + c1 - c0)
        border_ok = (
            np.all(grid[r0, c0 : c1 + 1] == color)
            and np.all(grid[r1, c0 : c1 + 1] == color)
            and np.all(grid[r0 : r1 + 1, c0] == color)
            and np.all(grid[r0 : r1 + 1, c1] == color)
        )
        if border_ok and len(coords) == perimeter:
            area = (r1 - r0 - 1) * (c1 - c0 - 1)
            candidates.append((area, r0, c0, r1, c1, color))

    assert candidates, "expected a complete monochrome rectangular frame"
    largest_area = max(item[0] for item in candidates)
    largest = [item for item in candidates if item[0] == largest_area]
    assert len(largest) == 1, "expected a unique largest rectangular frame"
    _, r0, c0, r1, c1, color = largest[0]
    return r0, c0, r1, c1, color


def _consistent_centers(
    grid: np.ndarray, frame: tuple[int, int, int, int, int], bg: int
) -> list[tuple[int, int]]:
    r0, c0, r1, c1, _ = frame
    samples = [
        (int(r), int(c), int(grid[r, c]))
        for r, c in np.argwhere((grid == RED) | (grid == YELLOW))
        if r0 < r < r1 and c0 < c < c1
    ]
    assert samples, "expected red and yellow samples inside the frame"
    assert all(grid[r, c] in (bg, RED, YELLOW) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1))

    result: list[tuple[int, int]] = []
    for cr in range(r0 + 1, r1):
        for cc in range(c0 + 1, c1):
            # The hidden center is specifically an unpainted interior cell.
            if int(grid[cr, cc]) != bg:
                continue
            valid = True
            for sr, sc, color in samples:
                distance = max(abs(sr - cr), abs(sc - cc))
                expected = YELLOW if distance % 2 else RED
                if distance == 0 or color != expected:
                    valid = False
                    break
            if valid:
                result.append((cr, cc))
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the hidden center and complete its alternating Chebyshev-distance ring field.

    Input structure:
    - A dominant background surrounds one complete, one-cell-thick monochrome rectangular frame.
    - Sparse red and yellow sample pixels occur strictly inside the frame; all other interior
      cells are background, including the hidden center.

    Rule:
    - Find the unique unpainted interior cell for which every yellow sample has odd
      Chebyshev distance and every red sample has positive even Chebyshev distance.
    - Preserve the frame and samples, and color every interior background cell yellow at
      odd distance from that center and red at even distance, including distance zero.

    Key insight:
    - The colored dots are joint parity constraints on one invisible center, not noise or
      independent seeds; only after inverting all constraints can the concentric field be filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a two-dimensional grid"
    bg = _background(g)
    frame = _rectangular_frame(g, bg)
    r0, c0, r1, c1, _ = frame

    interior_colors = set(map(int, np.unique(g[r0 + 1 : r1, c0 + 1 : c1])))
    assert interior_colors <= {bg, RED, YELLOW}, "unexpected color inside the frame"
    assert RED in interior_colors and YELLOW in interior_colors, "both sample colors must occur"

    centers = _consistent_centers(g, frame, bg)
    assert len(centers) == 1, "sample constraints must determine exactly one hidden center"
    cr, cc = centers[0]

    out = g.copy()
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if int(g[r, c]) == bg:
                distance = max(abs(r - cr), abs(c - cc))
                out[r, c] = YELLOW if distance % 2 else RED
    return out


def _parity(center: tuple[int, int], cell: tuple[int, int]) -> int:
    return max(abs(center[0] - cell[0]), abs(center[1] - cell[1])) % 2


def _local_candidates(
    h: int, w: int, samples: dict[tuple[int, int], int]
) -> list[tuple[int, int]]:
    """Generator-side constraint oracle, in frame-interior coordinates."""
    answer = []
    for center in ((r, c) for r in range(h) for c in range(w)):
        if center in samples:
            continue
        if all(_parity(center, pixel) == parity for pixel, parity in samples.items()):
            answer.append(center)
    return answer


def _choose_samples(
    h: int, w: int, center: tuple[int, int]
) -> dict[tuple[int, int], int] | None:
    available = [(r, c) for r in range(h) for c in range(w) if (r, c) != center]
    by_parity = {p: [cell for cell in available if _parity(center, cell) == p] for p in (0, 1)}
    if not by_parity[0] or not by_parity[1]:
        return None

    # Seed both visible constraint types, then greedily pick a strong discriminator.
    samples: dict[tuple[int, int], int] = {}
    for parity in (0, 1):
        pixel = random.choice(by_parity[parity])
        samples[pixel] = parity

    while _local_candidates(h, w, samples) != [center] and len(samples) < 14:
        best_count = h * w + 1
        best_pixels: list[tuple[int, int]] = []
        for pixel in available:
            if pixel in samples:
                continue
            trial = dict(samples)
            trial[pixel] = _parity(center, pixel)
            count = len(_local_candidates(h, w, trial))
            if count < best_count:
                best_count, best_pixels = count, [pixel]
            elif count == best_count:
                best_pixels.append(pixel)
        if not best_pixels:
            return None
        pixel = random.choice(best_pixels)
        samples[pixel] = _parity(center, pixel)

    if _local_candidates(h, w, samples) != [center]:
        return None

    # Add scattered evidence only while neither color alone becomes sufficient.
    target_count = random.randint(5, min(9, max(5, (h * w) // 10)))
    extras = [pixel for pixel in available if pixel not in samples]
    random.shuffle(extras)
    for pixel in extras:
        if len(samples) >= target_count:
            break
        trial = dict(samples)
        trial[pixel] = _parity(center, pixel)
        red_only = {p: v for p, v in trial.items() if v == 0}
        yellow_only = {p: v for p, v in trial.items() if v == 1}
        if len(_local_candidates(h, w, red_only)) > 1 and len(_local_candidates(h, w, yellow_only)) > 1:
            samples = trial

    red_only = {p: v for p, v in samples.items() if v == 0}
    yellow_only = {p: v for p, v in samples.items() if v == 1}
    if len(samples) < 4 or len(_local_candidates(h, w, red_only)) <= 1 or len(_local_candidates(h, w, yellow_only)) <= 1:
        return None
    return samples


def _oracle_output(
    inp: np.ndarray, r0: int, c0: int, r1: int, c1: int, center: tuple[int, int]
) -> np.ndarray:
    """Direct construction oracle; it does not infer the center via ``main`` helpers."""
    out = inp.copy()
    cr, cc = center
    rows, cols = np.indices((r1 - r0 - 1, c1 - c0 - 1))
    distances = np.maximum(np.abs(rows - cr), np.abs(cols - cc))
    field = np.where(distances % 2, YELLOW, RED)
    patch = out[r0 + 1 : r1, c0 + 1 : c1]
    patch[patch == BACKGROUND] = field[patch == BACKGROUND]
    return out


def _gen_attempt() -> np.ndarray | None:
    ih = random.randint(6, 13)
    iw = random.randint(6, 13)
    fh, fw = ih + 2, iw + 2
    canvas_h = random.randint(fh + 2, min(25, fh + 7))
    canvas_w = random.randint(fw + 2, min(25, fw + 7))
    r0 = random.randint(1, canvas_h - fh - 1)
    c0 = random.randint(1, canvas_w - fw - 1)
    r1, c1 = r0 + fh - 1, c0 + fw - 1

    # Half the cases put the hidden center next to a frame edge; the rest range freely.
    if random.random() < 0.5:
        side = random.randrange(4)
        if side == 0:
            center = (0, random.randrange(iw))
        elif side == 1:
            center = (ih - 1, random.randrange(iw))
        elif side == 2:
            center = (random.randrange(ih), 0)
        else:
            center = (random.randrange(ih), iw - 1)
    else:
        center = (random.randrange(ih), random.randrange(iw))

    # Reject geometric-center shortcuts (including either central cell of an even dimension).
    central_rows = {(ih - 1) // 2, ih // 2}
    central_cols = {(iw - 1) // 2, iw // 2}
    if center[0] in central_rows and center[1] in central_cols:
        return None

    samples = _choose_samples(ih, iw, center)
    if samples is None:
        return None
    sample_rows = [p[0] for p in samples]
    sample_cols = [p[1] for p in samples]
    bbox_middle = (
        (min(sample_rows) + max(sample_rows)) // 2,
        (min(sample_cols) + max(sample_cols)) // 2,
    )
    if center == bbox_middle:
        return None

    grid = np.full((canvas_h, canvas_w), BACKGROUND, dtype=int)
    frame_color = random.choice(FRAME_COLORS)
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    for (lr, lc), parity in samples.items():
        grid[r0 + 1 + lr, c0 + 1 + lc] = YELLOW if parity else RED

    absolute_center = (r0 + 1 + center[0], c0 + 1 + center[1])
    if int(grid[absolute_center]) != BACKGROUND:
        return None
    expected = _oracle_output(grid, r0, c0, r1, c1, center)
    newly_red = np.count_nonzero((grid == BACKGROUND) & (expected == RED))
    newly_yellow = np.count_nonzero((grid == BACKGROUND) & (expected == YELLOW))
    if newly_red < 4 or newly_yellow < 4:
        return None

    # Independent semantic checks: exact frame, exact unique center, and oracle output.
    frame = _rectangular_frame(grid, BACKGROUND)
    if frame[:4] != (r0, c0, r1, c1):
        return None
    inferred = _consistent_centers(grid, frame, BACKGROUND)
    if inferred != [absolute_center]:
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a unique, non-degenerate hidden-center ring puzzle")
