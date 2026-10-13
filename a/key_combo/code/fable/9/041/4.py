from __future__ import annotations

import itertools
from collections import Counter

import numpy as np


TRANSFORM_COUNT = 8
ARC_COLORS = tuple(range(10))
QUADRANT_NAMES = ("tl", "tr", "bl", "br")
MAX_GENERATION_ATTEMPTS = 400


def _apply_transform(grid: np.ndarray, transform: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if transform == 0:
        return g.copy()
    if transform == 1:
        return np.rot90(g, -1).copy()
    if transform == 2:
        return np.rot90(g, 2).copy()
    if transform == 3:
        return np.rot90(g, 1).copy()
    if transform == 4:
        return np.fliplr(g).copy()
    if transform == 5:
        return np.flipud(g).copy()
    if transform == 6:
        return g.T.copy()
    if transform == 7:
        return np.fliplr(np.rot90(g, 1))
    raise ValueError(f"unknown transform {transform}")


INVERSE_TRANSFORM = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7}


def _transform_coord(n: int, r: int, c: int, transform: int) -> tuple[int, int]:
    if transform == 0:
        return r, c
    if transform == 1:
        return c, n - 1 - r
    if transform == 2:
        return n - 1 - r, n - 1 - c
    if transform == 3:
        return n - 1 - c, r
    if transform == 4:
        return r, n - 1 - c
    if transform == 5:
        return n - 1 - r, c
    if transform == 6:
        return c, r
    if transform == 7:
        return n - 1 - c, n - 1 - r
    raise ValueError(f"unknown transform {transform}")


def _quadrants(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h == w and h % 2 == 0 and h >= 4, "expected an even square split into four equal quadrants"
    n = h // 2
    return [
        g[:n, :n].copy(),
        g[:n, n:].copy(),
        g[n:, :n].copy(),
        g[n:, n:].copy(),
    ]


def _score_stack(stack: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    n = stack.shape[1]
    counts = np.zeros((10, n, n), dtype=int)
    for color in ARC_COLORS:
        counts[color] = (stack == color).sum(axis=0)
    max_counts = counts.max(axis=0)
    majority = counts.argmax(axis=0)
    return int(max_counts.sum()), majority.astype(int), max_counts


def _best_alignment(quads: list[np.ndarray]) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]:
    inverse_versions = [
        [_apply_transform(quad, INVERSE_TRANSFORM[t]) for t in range(TRANSFORM_COUNT)]
        for quad in quads
    ]

    best_score = -1
    best_key: tuple[int, ...] | None = None
    best_transforms: tuple[int, int, int, int] | None = None
    best_majority: np.ndarray | None = None
    best_strength: np.ndarray | None = None

    for transforms in itertools.permutations(range(TRANSFORM_COUNT), 4):
        stack = np.stack([inverse_versions[i][t] for i, t in enumerate(transforms)], axis=0)
        score, majority, strength = _score_stack(stack)
        key = tuple(int(x) for x in majority.reshape(-1))
        if score > best_score or (score == best_score and (best_key is None or key < best_key)):
            best_score = score
            best_key = key
            best_transforms = tuple(int(t) for t in transforms)
            best_majority = majority.copy()
            best_strength = strength.copy()

    assert best_transforms is not None and best_majority is not None and best_strength is not None
    return best_transforms, best_majority, best_strength


def _extract_message(quads: list[np.ndarray], transforms: tuple[int, int, int, int]) -> tuple[np.ndarray, int]:
    aligned = np.stack(
        [_apply_transform(quad, INVERSE_TRANSFORM[t]) for quad, t in zip(quads, transforms)],
        axis=0,
    )
    n = aligned.shape[1]
    out = np.zeros((n, n), dtype=int)
    signal_colors: list[int] = []

    for r in range(n):
        for c in range(n):
            votes = Counter(map(int, aligned[:, r, c]))
            color, count = votes.most_common(1)[0]
            if count == 4:
                continue
            assert count == 3 and len(votes) == 2, "each dissent orbit must be a 3:1 vote"
            dissent_colors = [v for v, v_count in votes.items() if v_count == 1]
            assert len(dissent_colors) == 1
            signal = dissent_colors[0]
            signal_colors.append(signal)
            out[r, c] = signal
            assert signal != color, "the dissent pixel must differ from the orbit majority"

    assert signal_colors, "expected at least one dissent orbit"
    assert len(set(signal_colors)) == 1, "all dissent pixels must share one signal color"
    return out, signal_colors[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the four quadrant symmetries, then majority-vote away the carrier to reveal the minority message.

    Input structure:
    - The input is an even square grid split by its central horizontal and vertical boundaries into four equal square quadrants.
    - Each quadrant is the same asymmetric base pattern seen through one of the eight square symmetries
      (four rotations and four reflections), with the four quadrant transforms all distinct.
    - For some base coordinates, exactly one of the four corresponding quadrant pixels is overwritten by one shared
      signal color; every other orbit stays perfectly consistent across the quadrants.

    Rule:
    - Try all distinct assignments of square symmetries to the four quadrants and align each quadrant back to base
      coordinates with the inverse transform.
    - Choose the alignment that maximizes total orbit agreement across the whole grid.
    - For each aligned orbit, use the 4-pixel majority vote to recover the carrier color and detect 3:1 dissent orbits.
    - Output an N x N grid with background 0 and draw the shared signal color at every base coordinate whose orbit had one dissenting pixel.

    Key insight:
    - The message is not tied to a special color in the input; it is defined only by being the one-vote minority inside
      correctly aligned four-way symmetry orbits, so transform inference and majority voting have to be solved together.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    quads = _quadrants(g)
    transforms, _majority, _strength = _best_alignment(quads)
    message, _signal = _extract_message(quads, transforms)
    return message


def _random_palette(rng: np.random.Generator) -> list[int]:
    color_count = int(rng.integers(3, 6))
    return rng.choice(np.arange(1, 10), size=color_count, replace=False).astype(int).tolist()


def _has_no_d4_symmetry(base: np.ndarray) -> bool:
    return all(not np.array_equal(base, _apply_transform(base, t)) for t in range(1, TRANSFORM_COUNT))


def _sample_base(rng: np.random.Generator, n: int, palette: list[int], signal: int) -> np.ndarray | None:
    base = rng.choice(np.array(palette, dtype=int), size=(n, n), replace=True).astype(int)

    forced_cells = rng.choice(base.size, size=len(palette), replace=False)
    for idx, color in zip(forced_cells, palette):
        base[idx // n, idx % n] = int(color)

    signal_count = int(np.count_nonzero(base == signal))
    if signal_count < 2 or signal_count > (n * n) // 2:
        return None
    if np.count_nonzero(base != signal) < 2:
        return None
    if not _has_no_d4_symmetry(base):
        return None
    return base


def _canonical_transform(base: np.ndarray) -> int:
    variants = [(tuple(int(x) for x in _apply_transform(base, t).reshape(-1)), t) for t in range(TRANSFORM_COUNT)]
    variants.sort()
    return int(variants[0][1])


def _message_output(n: int, signal: int, coords: list[tuple[int, int]], canonical_t: int) -> np.ndarray:
    out = np.zeros((n, n), dtype=int)
    for r, c in coords:
        rr, cc = _transform_coord(n, r, c, canonical_t)
        out[rr, cc] = signal
    return out


def _build_input(
    base: np.ndarray,
    signal: int,
    message_coords: list[tuple[int, int]],
    transforms: list[int],
    dissent_quadrants: list[int],
) -> np.ndarray:
    n = base.shape[0]
    quads = [_apply_transform(base, t) for t in transforms]
    for (r, c), q_idx in zip(message_coords, dissent_quadrants):
        rr, cc = _transform_coord(n, r, c, transforms[q_idx])
        quads[q_idx][rr, cc] = signal
    return np.block([[quads[0], quads[1]], [quads[2], quads[3]]])


def _best_alignment_details(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    quads = _quadrants(grid)
    transforms, _majority, _strength = _best_alignment(quads)
    message, _signal = _extract_message(quads, transforms)
    return transforms, message


def _semantic_ok(
    grid: np.ndarray,
    expected: np.ndarray,
    base: np.ndarray,
    signal: int,
    message_coords: list[tuple[int, int]],
) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    inferred_transforms, inferred_message = _best_alignment_details(grid)
    if not np.array_equal(inferred_message, expected):
        return False
    if len(set(inferred_transforms)) != 4:
        return False

    n = base.shape[0]
    if not np.array_equal(expected.shape, (n, n)):
        return False
    if int(np.count_nonzero(expected == signal)) != len(message_coords):
        return False

    base_signal_cells = np.argwhere(base == signal)
    if len(base_signal_cells) < 2:
        return False
    if any(base[r, c] == signal for r, c in message_coords):
        return False

    non_message_signal = set(map(tuple, base_signal_cells)) - set(message_coords)
    if len(non_message_signal) < 2:
        return False

    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.integers(4, 9))
    palette = _random_palette(rng)
    signal = int(rng.choice(np.array(palette, dtype=int)))
    base = _sample_base(rng, n, palette, signal)
    if base is None:
        return None

    candidates = [(int(r), int(c)) for r, c in np.argwhere(base != signal)]
    max_messages = min(6, len(candidates))
    if max_messages < 2:
        return None
    message_count = int(rng.integers(2, max_messages + 1))
    rng.shuffle(candidates)
    message_coords = candidates[:message_count]

    transforms = rng.choice(np.arange(TRANSFORM_COUNT), size=4, replace=False).astype(int).tolist()
    dissent_quadrants = rng.integers(0, 4, size=message_count).astype(int).tolist()

    grid = _build_input(base, signal, message_coords, transforms, dissent_quadrants)
    canonical_t = _canonical_transform(base)
    expected = _message_output(n, signal, message_coords, canonical_t)

    if not _semantic_ok(grid, expected, base, signal, message_coords):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid minority-message symmetry carrier")
