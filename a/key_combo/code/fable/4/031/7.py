"""ARC task 7: repair a rectangular occlusion in a two-scale self-similar pattern."""

from __future__ import annotations

from collections import Counter
import hashlib
import math
import random

import numpy as np


PATCH_COLOR = 5
PALETTE = tuple(range(10))
SEED_SIDE_OPTIONS = (3, 4, 5)
MAX_GENERATION_ATTEMPTS = 80


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return an inclusive (r0, c0, r1, c1) bbox using ARC row/column coordinates."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _seed_shape(shape: tuple[int, int]) -> tuple[int, int]:
    """Infer the seed dimensions from its complete one-level expansion dimensions."""
    height, width = map(int, shape)
    seed_h = math.isqrt(height)
    seed_w = math.isqrt(width)
    assert seed_h * seed_h == height and seed_w * seed_w == width, (
        "each large-grid dimension must be the square of its seed dimension"
    )
    assert seed_h >= 2 and seed_w >= 2, "the seed must have two nontrivial dimensions"
    return seed_h, seed_w


def _strict_visible_background(grid: np.ndarray) -> int:
    """Return the uniquely most frequent non-patch color."""
    visible = grid[grid != PATCH_COLOR]
    assert visible.size, "the grey patch cannot cover the whole grid"
    counts = Counter(map(int, visible))
    ranked = counts.most_common()
    assert len(ranked) >= 3, "visible cells must show background and both block colors"
    assert ranked[0][1] > ranked[1][1], "background must be the unique visible mode"
    return int(ranked[0][0])


def _recover_seed(grid: np.ndarray, seed_h: int, seed_w: int, bg: int) -> np.ndarray:
    """Recover each fine-scale seed cell from all copies at the same block offset."""
    seed = np.zeros((seed_h, seed_w), dtype=bool)
    for inner_r in range(seed_h):
        for inner_c in range(seed_w):
            copies = grid[inner_r::seed_h, inner_c::seed_w]
            values = copies[copies != PATCH_COLOR]
            assert values.size, "every fine-scale seed position needs a visible copy"
            unique = set(map(int, np.unique(values)))
            if bg in unique:
                assert unique == {bg}, "one seed offset cannot be both background and foreground"
            else:
                seed[inner_r, inner_c] = True
    assert np.any(seed) and np.any(~seed), "the seed mask must contain both states"
    return seed


def _recover_block_colors(
    grid: np.ndarray, seed: np.ndarray, bg: int
) -> tuple[int, int]:
    """Read the one visible color assigned to each coarse seed state."""
    seed_h, seed_w = seed.shape
    colors_by_state = {False: set(), True: set()}
    for block_r in range(seed_h):
        for block_c in range(seed_w):
            block = grid[
                block_r * seed_h : (block_r + 1) * seed_h,
                block_c * seed_w : (block_c + 1) * seed_w,
            ]
            colors_by_state[bool(seed[block_r, block_c])].update(
                int(value)
                for value in np.unique(block)
                if int(value) not in {bg, PATCH_COLOR}
            )

    assert len(colors_by_state[True]) == 1, "foreground coarse blocks need one main color"
    assert len(colors_by_state[False]) == 1, "background coarse blocks need one secondary color"
    primary = next(iter(colors_by_state[True]))
    secondary = next(iter(colors_by_state[False]))
    assert primary != secondary, "the two coarse states must have distinct visible colors"
    return primary, secondary


def _render_vectorized(
    seed: np.ndarray, bg: int, primary: int, secondary: int
) -> np.ndarray:
    """Render a full one-level self-similar expansion from recovered semantics."""
    seed_h, seed_w = seed.shape
    rows = np.arange(seed_h * seed_h)[:, None]
    cols = np.arange(seed_w * seed_w)[None, :]
    fine = seed[rows % seed_h, cols % seed_w]
    coarse = seed[rows // seed_h, cols // seed_w]

    out = np.full((seed_h * seed_h, seed_w * seed_w), bg, dtype=int)
    out[fine & coarse] = primary
    out[fine & ~coarse] = secondary
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Repair a grey rectangle by reconstructing the seed's coarse/fine self-similar expansion.

    Input structure:
    - The large grid is a complete one-level expansion of a nontrivial rectangular binary seed mask;
      its height and width are respectively the squares of the seed height and width.
    - Every fine block repeats the seed foreground shape on one background color. Coarse foreground
      blocks use one main color, while coarse background blocks use one distinct secondary color.
    - One solid grey rectangular patch occludes part of the expansion, possibly across block boundaries.

    Rule:
    - Use visible copies at equal within-block coordinates to recover the seed mask, then use that same
      mask at the coarse block coordinates to classify blocks and read their main or secondary color.
    - Rebuild every cell from its exact coarse block coordinate and exact within-block coordinate,
      replacing the grey patch while leaving all visible cells unchanged.

    Key insight:
    - The pattern is its own legend: correct repair requires the same seed at both the coarse block
      scale and the fine within-block scale, rather than a mirror or a single repeated colored tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    seed_h, seed_w = _seed_shape(g.shape)

    patch_bbox = _bbox(g == PATCH_COLOR)
    assert patch_bbox is not None, "expected one grey occlusion patch"
    r0, c0, r1, c1 = patch_bbox
    assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == PATCH_COLOR), (
        "grey cells must form one solid rectangle"
    )

    bg = _strict_visible_background(g)
    assert bg != PATCH_COLOR, "background and patch colors must differ"
    seed = _recover_seed(g, seed_h, seed_w, bg)
    primary, secondary = _recover_block_colors(g, seed, bg)
    out = _render_vectorized(seed, bg, primary, secondary)

    visible = g != PATCH_COLOR
    assert np.array_equal(out[visible], g[visible]), (
        "visible evidence is inconsistent with one self-similar expansion"
    )
    return out


def _render_independent(
    seed: np.ndarray, bg: int, primary: int, secondary: int
) -> np.ndarray:
    """Direct block renderer kept independent from main's coordinate arrays."""
    seed_h, seed_w = seed.shape
    out = np.full((seed_h * seed_h, seed_w * seed_w), bg, dtype=int)
    for block_r in range(seed_h):
        for block_c in range(seed_w):
            color = primary if seed[block_r, block_c] else secondary
            for inner_r in range(seed_h):
                for inner_c in range(seed_w):
                    if seed[inner_r, inner_c]:
                        out[block_r * seed_h + inner_r, block_c * seed_w + inner_c] = color
    return out


def _nondegenerate_seed(seed_h: int, seed_w: int) -> np.ndarray | None:
    """Sample a mixed, aperiodic, asymmetric mask that visibly refutes mirror shortcuts."""
    area = seed_h * seed_w
    low = math.ceil(0.38 * area)
    high = math.floor(0.62 * area)
    count = random.randint(low, high)
    chosen = random.sample(range(area), count)
    seed = np.zeros((seed_h, seed_w), dtype=bool)
    seed.reshape(-1)[chosen] = True

    if any(np.all(row) or not np.any(row) for row in seed):
        return None
    if any(np.all(col) or not np.any(col) for col in seed.T):
        return None
    if np.array_equal(seed, np.flipud(seed)) or np.array_equal(seed, np.fliplr(seed)):
        return None
    if seed_h == seed_w and (
        np.array_equal(seed, seed.T) or np.array_equal(seed, np.rot90(seed, 2))
    ):
        return None
    if any(np.array_equal(seed, np.roll(seed, shift, axis=0)) for shift in range(1, seed_h)):
        return None
    if any(np.array_equal(seed, np.roll(seed, shift, axis=1)) for shift in range(1, seed_w)):
        return None
    return seed


def _candidate_patch_boxes(
    expected: np.ndarray, seed: np.ndarray, bg: int, primary: int, secondary: int
) -> list[tuple[int, int, int, int]]:
    """Return the first informative rectangle from a finite randomized candidate set.

    The old generator accumulated every valid rectangle before choosing one.  On a
    25x25 expansion that repeatedly performed tens of thousands of small NumPy
    operations even though only one rectangle is needed.  Shuffling the finite
    geometry candidates preserves varied placement while allowing an immediate
    return as soon as all semantic constraints are met.
    """
    height, width = expected.shape
    seed_h, seed_w = seed.shape
    max_area = max(12, int(0.30 * height * width))
    candidates: list[tuple[int, int, int, int]] = []
    for patch_h in range(3, min(2 * seed_h, height - 2) + 1):
        for patch_w in range(3, min(2 * seed_w, width - 2) + 1):
            area = patch_h * patch_w
            if area < 9 or area > max_area:
                continue
            for r0 in range(1, height - patch_h):
                for c0 in range(1, width - patch_w):
                    candidates.append((r0, c0, r0 + patch_h, c0 + patch_w))

    random.shuffle(candidates)
    horizontal_mirror = np.fliplr(expected)
    vertical_mirror = np.flipud(expected)
    for r0, c0, r1, c1 in candidates:
        # At least one rectangle edge must be off the coarse block grid.
        if (
            r0 % seed_h == 0
            and r1 % seed_h == 0
            and c0 % seed_w == 0
            and c1 % seed_w == 0
        ):
            continue

        rows = np.arange(r0, r1)[:, None]
        cols = np.arange(c0, c1)[None, :]
        coarse = seed[rows // seed_h, cols // seed_w]
        if not (np.any(coarse) and np.any(~coarse)):
            continue

        region = expected[r0:r1, c0:c1]
        # Within each coarse state, the patch must cover both fine background
        # and fine foreground. This makes both semantic steps independently necessary.
        if not ({bg, primary} <= set(map(int, np.unique(region[coarse])))):
            continue
        if not ({bg, secondary} <= set(map(int, np.unique(region[~coarse])))):
            continue
        if any(np.count_nonzero(region == color) < 2 for color in (bg, primary, secondary)):
            continue
        if np.array_equal(region, horizontal_mirror[r0:r1, c0:c1]):
            continue
        if np.array_equal(region, vertical_mirror[r0:r1, c0:c1]):
            continue
        return [(r0, c0, r1, c1)]
    return []


def _choose_patch_box(
    boxes: list[tuple[int, int, int, int]], seed: np.ndarray
) -> tuple[int, int, int, int]:
    """Choose reproducibly from the seed so one complete pattern has one generated occlusion."""
    payload = np.packbits(seed, axis=None).tobytes() + bytes(seed.shape)
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return boxes[int.from_bytes(digest, "big") % len(boxes)]


def _semantically_valid(
    inp: np.ndarray,
    expected: np.ndarray,
    seed: np.ndarray,
    bg: int,
    primary: int,
    secondary: int,
    box: tuple[int, int, int, int],
) -> bool:
    """Check uniqueness, two-scale dependence, anti-shortcuts, and the independent oracle."""
    if inp.ndim != 2 or max(inp.shape) > 30 or inp.shape != expected.shape:
        return False
    patch = inp == PATCH_COLOR
    if _bbox(patch) != (box[0], box[1], box[2] - 1, box[3] - 1):
        return False
    r0, c0, r1, c1 = box
    if not np.all(patch[r0:r1, c0:c1]) or np.any(inp[~patch] != expected[~patch]):
        return False

    # Every within-block coordinate is visibly classified, making the fine mask unique.
    seed_h, seed_w = seed.shape
    for inner_r in range(seed_h):
        for inner_c in range(seed_w):
            copies = inp[inner_r::seed_h, inner_c::seed_w]
            visible = copies[copies != PATCH_COLOR]
            if not visible.size:
                return False
            if seed[inner_r, inner_c]:
                if np.any(visible == bg):
                    return False
            elif np.any(visible != bg):
                return False

    # Both coarse states retain visible colored evidence, so their color mapping is unique.
    for state, color in ((True, primary), (False, secondary)):
        seen: set[int] = set()
        for block_r, block_c in np.argwhere(seed == state):
            block = inp[
                block_r * seed_h : (block_r + 1) * seed_h,
                block_c * seed_w : (block_c + 1) * seed_w,
            ]
            seen.update(int(v) for v in np.unique(block) if int(v) not in {bg, PATCH_COLOR})
        if seen != {color}:
            return False

    covered = expected[r0:r1, c0:c1]
    if set(map(int, np.unique(covered))) != {bg, primary, secondary}:
        return False
    if np.array_equal(expected, np.fliplr(expected)) or np.array_equal(expected, np.flipud(expected)):
        return False

    visible_counts = Counter(map(int, inp[inp != PATCH_COLOR]))
    if visible_counts[bg] <= max(visible_counts[primary], visible_counts[secondary]):
        return False
    try:
        actual = main(inp)
    except (AssertionError, IndexError, ValueError):
        return False
    return np.array_equal(actual, expected) and not np.array_equal(actual, inp)


def generate_input() -> np.ndarray:
    """Generate a varied, uniquely recoverable two-scale self-similar repair puzzle."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        seed_h = random.choice(SEED_SIDE_OPTIONS)
        seed_w = random.choice(SEED_SIDE_OPTIONS)
        seed = _nondegenerate_seed(seed_h, seed_w)
        if seed is None:
            continue

        bg, primary, secondary = random.sample(
            [color for color in PALETTE if color != PATCH_COLOR], 3
        )
        expected = _render_independent(seed, bg, primary, secondary)
        boxes = _candidate_patch_boxes(expected, seed, bg, primary, secondary)
        if not boxes:
            continue
        box = _choose_patch_box(boxes, seed)

        inp = expected.copy()
        r0, c0, r1, c1 = box
        inp[r0:r1, c0:c1] = PATCH_COLOR
        if _semantically_valid(inp, expected, seed, bg, primary, secondary, box):
            return inp
    raise RuntimeError("failed to generate a valid self-similar repair puzzle")
