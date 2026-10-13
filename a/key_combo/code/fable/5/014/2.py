from __future__ import annotations

import random
from collections import deque

import numpy as np


_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_ARC_COLORS = tuple(range(1, 10))


def _background(grid: np.ndarray) -> int:
    """Use the outer border as the frame scene's coordinate reference."""
    border = np.concatenate(
        (grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1])
    )
    colors, counts = np.unique(border, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    """Return 4-connected, same-color components using (row, column) coords."""
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    objects: list[dict] = []

    for r0 in range(height):
        for c0 in range(width):
            color = int(grid[r0, c0])
            if color == bg or seen[r0, c0]:
                continue
            seen[r0, c0] = True
            queue = deque([(r0, c0)])
            coords: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for dr, dc in _DIR4:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return objects


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    """Find components that are exactly one-cell-thick hollow square borders."""
    frames: list[dict] = []
    for obj in _monochrome_components(grid, bg):
        r0, c0, r1, c1 = obj["bbox"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height != width or height < 4:
            continue
        expected = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        if set(obj["coords"]) != expected:
            continue
        frames.append(
            {
                **obj,
                "side": height,
                "inner": grid[r0 + 1 : r1, c0 + 1 : c1].copy(),
            }
        )
    frames.sort(key=lambda frame: frame["bbox"][:2])
    return frames


def _palette(sprite: np.ndarray, bg: int) -> frozenset[int]:
    return frozenset(int(v) for v in np.unique(sprite) if int(v) != bg)


def _d4(sprite: np.ndarray) -> list[np.ndarray]:
    """The four rotations followed by their left-right reflections."""
    rotations = [np.rot90(sprite, k).copy() for k in range(4)]
    return rotations + [np.fliplr(x).copy() for x in rotations]


def _assemble(sprite: np.ndarray, scheme: tuple[int, int, int, int]) -> np.ndarray:
    variants = _d4(sprite)
    return np.block(
        [
            [variants[scheme[0]], variants[scheme[1]]],
            [variants[scheme[2]], variants[scheme[3]]],
        ]
    )


def _training_and_test(
    frames: list[dict], bg: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nonblank = [frame for frame in frames if _palette(frame["inner"], bg)]
    pair_candidates: list[tuple[int, int]] = []

    for i, first in enumerate(nonblank):
        for j in range(i + 1, len(nonblank)):
            second = nonblank[j]
            if _palette(first["inner"], bg) != _palette(second["inner"], bg):
                continue
            a, b = first["inner"], second["inner"]
            if b.shape == (2 * a.shape[0], 2 * a.shape[1]):
                pair_candidates.append((i, j))
            elif a.shape == (2 * b.shape[0], 2 * b.shape[1]):
                pair_candidates.append((j, i))

    assert len(pair_candidates) == 1, "expected one palette-matched n-to-2n training pair"
    small_idx, large_idx = pair_candidates[0]
    remaining = [
        frame["inner"]
        for i, frame in enumerate(nonblank)
        if i not in (small_idx, large_idx)
    ]
    assert len(remaining) == 1, "expected exactly one nonblank test frame"

    train = nonblank[small_idx]["inner"]
    tiled = nonblank[large_idx]["inner"]
    test = remaining[0]
    assert train.shape[0] == train.shape[1] and train.shape[0] > 0
    assert tiled.shape == (2 * train.shape[0], 2 * train.shape[1])
    assert test.shape == train.shape, "test sprite must use the training sprite's n x n frame"
    assert _palette(test, bg) != _palette(train, bg)
    return train, tiled, test


def main(grid: np.ndarray) -> np.ndarray:
    """Learn a per-quadrant dihedral tiling scheme from framed sprites and apply it to the test sprite.

    Input structure:
    - A background grid contains separated one-cell-thick monochrome hollow square frames.
    - Two nonblank interiors have sizes n x n and 2n x 2n and share exactly the same non-background palette; one other nonblank n x n interior has a different palette.
    - Any remaining frames are blank distractors, and the 2n x 2n interior consists of four transformed copies of the smaller matched sprite.

    Rule:
    - Match the unique training pair by equal interior palettes and the 1:2 size relation.
    - Split the larger training interior in its own frame coordinate system and identify, independently for each quadrant, its unique rotation or reflection of the n x n training sprite.
    - Apply those four inferred transformations in the same quadrant positions to the different-palette test sprite and return only the resulting 2n x 2n tiling.

    Key insight:
    - Palette equality identifies which framed objects teach the analogy, while exact per-quadrant geometric comparison reveals a four-part composition law rather than a pixelwise color mapping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 4, "expected a two-dimensional frame scene"
    bg = _background(g)
    frames = _find_frames(g, bg)
    assert len(frames) >= 3, "expected at least three hollow square frames"
    train, tiled, test = _training_and_test(frames, bg)

    n = train.shape[0]
    variants = _d4(train)
    assert len({variant.tobytes() for variant in variants}) == 8, (
        "training sprite must distinguish all eight rotations/reflections"
    )

    quadrants = (
        tiled[:n, :n],
        tiled[:n, n:],
        tiled[n:, :n],
        tiled[n:, n:],
    )
    scheme: list[int] = []
    for quadrant in quadrants:
        matches = [i for i, variant in enumerate(variants) if np.array_equal(quadrant, variant)]
        assert len(matches) == 1, "each training quadrant must identify one unique transform"
        scheme.append(matches[0])

    return _assemble(test, tuple(scheme))


# Generator -----------------------------------------------------------------

_GEN_ATTEMPTS = 400
_PENDING_COMPLEMENT: tuple[int, int, int, int] | None = None


def _random_asymmetric_sprite(n: int, colors: list[int]) -> np.ndarray | None:
    """Make a sparse sprite with its complete palette and a trivial D4 stabilizer."""
    for _ in range(100):
        sprite = np.zeros((n, n), dtype=int)
        count = random.randint(max(4, len(colors) + 1), min(n * n - 2, n + 5))
        cells = random.sample([(r, c) for r in range(n) for c in range(n)], count)
        assigned = colors.copy()
        assigned.extend(random.choice(colors) for _ in range(count - len(colors)))
        random.shuffle(assigned)
        for (r, c), color in zip(cells, assigned):
            sprite[r, c] = color
        variants = _d4(sprite)
        if len({variant.tobytes() for variant in variants}) == 8:
            return sprite
    return None


def _new_scheme_pair() -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Each consecutive pair covers all 8 D4 transforms; each call mixes both kinds."""
    first = random.sample(range(4), 2) + random.sample(range(4, 8), 2)
    random.shuffle(first)
    complement = [i for i in range(8) if i not in first]
    random.shuffle(complement)
    return tuple(first), tuple(complement)


def _candidate_scheme() -> tuple[tuple[int, int, int, int], tuple[int, int, int, int] | None]:
    if _PENDING_COMPLEMENT is not None:
        return _PENDING_COMPLEMENT, None
    return _new_scheme_pair()


def _boxes_clear(
    box: tuple[int, int, int], placed: list[tuple[int, int, int]], gap: int = 1
) -> bool:
    r0, c0, side = box
    r1, c1 = r0 + side - 1, c0 + side - 1
    for pr0, pc0, pside in placed:
        pr1, pc1 = pr0 + pside - 1, pc0 + pside - 1
        if not (
            r1 + gap < pr0
            or pr1 + gap < r0
            or c1 + gap < pc0
            or pc1 + gap < c0
        ):
            return False
    return True


def _place_boxes(
    height: int, width: int, sides: list[int]
) -> list[tuple[int, int, int]] | None:
    """Place frames with a visible background moat and no shared components."""
    indexed = sorted(enumerate(sides), key=lambda item: -item[1])
    placed_by_index: dict[int, tuple[int, int, int]] = {}
    placed: list[tuple[int, int, int]] = []
    for index, side in indexed:
        candidates = [
            (r, c, side)
            for r in range(1, height - side)
            for c in range(1, width - side)
            if _boxes_clear((r, c, side), placed)
        ]
        if not candidates:
            return None
        chosen = random.choice(candidates)
        placed.append(chosen)
        placed_by_index[index] = chosen
    return [placed_by_index[i] for i in range(len(sides))]


def _draw_frame(
    grid: np.ndarray,
    box: tuple[int, int, int],
    frame_color: int,
    sprite: np.ndarray,
) -> None:
    r0, c0, side = box
    r1, c1 = r0 + side - 1, c0 + side - 1
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    assert sprite.shape == (side - 2, side - 2)
    grid[r0 + 1 : r1, c0 + 1 : c1] = sprite


def _unique_pair_count(frames: list[dict], bg: int) -> int:
    nonblank = [frame for frame in frames if _palette(frame["inner"], bg)]
    count = 0
    for i, first in enumerate(nonblank):
        for second in nonblank[i + 1 :]:
            if _palette(first["inner"], bg) != _palette(second["inner"], bg):
                continue
            a, b = first["inner"], second["inner"]
            if b.shape == (2 * a.shape[0], 2 * a.shape[1]) or a.shape == (
                2 * b.shape[0],
                2 * b.shape[1],
            ):
                count += 1
    return count


def _validate_generated(
    grid: np.ndarray,
    train: np.ndarray,
    tiled: np.ndarray,
    test: np.ndarray,
    scheme: tuple[int, int, int, int],
) -> bool:
    """Check scene semantics, uniqueness, key dependencies, and the solver oracle."""
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    bg = _background(grid)
    frames = _find_frames(grid, bg)
    if len(frames) < 3 or _unique_pair_count(frames, bg) != 1:
        return False
    if _palette(train, bg) != _palette(tiled, bg):
        return False
    if not _palette(test, bg) or _palette(test, bg) == _palette(train, bg):
        return False
    if len({x.tobytes() for x in _d4(train)}) != 8:
        return False
    if len({x.tobytes() for x in _d4(test)}) != 8:
        return False
    if len(set(scheme)) != 4 or not any(i < 4 for i in scheme) or not any(i >= 4 for i in scheme):
        return False
    if not np.array_equal(tiled, _assemble(train, scheme)):
        return False

    expected = _assemble(test, scheme)
    swapped = _assemble(test, (scheme[1], scheme[0], scheme[2], scheme[3]))
    if np.array_equal(expected, swapped):
        return False
    if len({q.tobytes() for q in (expected[: test.shape[0], : test.shape[0]],
                                  expected[: test.shape[0], test.shape[0] :],
                                  expected[test.shape[0] :, : test.shape[0]],
                                  expected[test.shape[0] :, test.shape[0] :])}) != 4:
        return False
    try:
        actual = main(grid)
    except (AssertionError, ValueError):
        return False
    return actual.shape == expected.shape and np.array_equal(actual, expected)


def _generate_attempt(scheme: tuple[int, int, int, int]) -> np.ndarray | None:
    n = random.choice((3, 4, 5))
    train_palette_size = random.choice((1, 2))
    test_palette_size = random.choice((1, 2))
    sprite_colors = random.sample(_ARC_COLORS, train_palette_size + test_palette_size)
    train_colors = sprite_colors[:train_palette_size]
    test_colors = sprite_colors[train_palette_size:]

    train = _random_asymmetric_sprite(n, train_colors)
    test = _random_asymmetric_sprite(n, test_colors)
    if train is None or test is None:
        return None
    tiled = _assemble(train, scheme)

    blank_count = random.choice((0, 1, 2))
    small_side = n + 2
    large_side = 2 * n + 2
    blank_sides = [random.choice((small_side, large_side)) for _ in range(blank_count)]
    sides = [small_side, large_side, small_side] + blank_sides

    height = random.randint(22, 30)
    width = random.randint(22, 30)
    boxes = _place_boxes(height, width, sides)
    if boxes is None:
        return None

    remaining_frame_colors = [c for c in _ARC_COLORS if c not in sprite_colors]
    if not remaining_frame_colors:
        return None
    frame_colors = [random.choice(remaining_frame_colors) for _ in sides]
    sprites = [train, tiled, test] + [np.zeros((side - 2, side - 2), dtype=int) for side in blank_sides]

    order = list(range(len(sides)))
    random.shuffle(order)
    grid = np.zeros((height, width), dtype=int)
    for index in order:
        _draw_frame(grid, boxes[index], frame_colors[index], sprites[index])

    if not _validate_generated(grid, train, tiled, test, scheme):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a diverse, oracle-checked framed analogy scene."""
    global _PENDING_COMPLEMENT

    scheme, complement_after_success = _candidate_scheme()
    for _ in range(_GEN_ATTEMPTS):
        grid = _generate_attempt(scheme)
        if grid is not None:
            _PENDING_COMPLEMENT = complement_after_success
            return grid
    raise RuntimeError(f"failed to generate a valid quadrant analogy after {_GEN_ATTEMPTS} attempts")
