from __future__ import annotations

import random

import numpy as np


_ARC_COLORS = (1, 2, 3, 4, 5, 6, 7, 8, 9)

# Each template is already left-right symmetric and chosen because sparse witnesses
# can make at least one smaller wrong period look locally valid while still failing
# the final symmetry check.
_FAMILY_TEMPLATES = (
    np.array([[0, 1, 1, 0], [1, 2, 2, 1]], dtype=int),
    np.array([[0, 1, 1, 0], [2, 3, 3, 2]], dtype=int),
    np.array([[0, 1, 1, 0], [0, 1, 1, 0], [1, 2, 2, 1]], dtype=int),
    np.array([[0, 1, 1, 0], [0, 1, 1, 0], [2, 3, 3, 2]], dtype=int),
    np.array([[0, 0, 1, 0, 0], [0, 1, 2, 1, 0]], dtype=int),
    np.array([[0, 0, 1, 0, 0], [2, 1, 3, 1, 2]], dtype=int),
    np.array([[0, 0, 1, 0, 0], [0, 2, 3, 2, 0]], dtype=int),
    np.array([[0, 0, 1, 0, 0], [0, 0, 2, 0, 0], [3, 1, 2, 1, 3]], dtype=int),
    np.array([[0, 0, 1, 0, 0], [0, 1, 2, 1, 0], [0, 0, 3, 0, 0]], dtype=int),
    np.array([[0, 0, 1, 1, 0, 0], [1, 0, 2, 2, 0, 1]], dtype=int),
    np.array([[0, 0, 1, 1, 0, 0], [2, 0, 3, 3, 0, 2]], dtype=int),
)

_GEN_ATTEMPTS = 300
_WITNESS_ATTEMPTS = 180


def _visible_points(grid: np.ndarray) -> list[tuple[int, int, int]]:
    return [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere(grid != 0)]


def _completion_from_mapping(
    shape: tuple[int, int],
    row_period: int,
    col_period: int,
    mapping: dict[tuple[int, int], int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for (rr, cc), color in mapping.items():
        out[rr::row_period, cc::col_period] = color
    return out


def _enumerate_candidates(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    points = _visible_points(g)
    candidates: list[dict[str, object]] = []

    for row_period in range(1, height + 1):
        for col_period in range(1, width + 1):
            mapping: dict[tuple[int, int], int] = {}
            ok = True
            for r, c, color in points:
                key = (r % row_period, c % col_period)
                prev = mapping.get(key)
                if prev is None:
                    mapping[key] = color
                elif prev != color:
                    ok = False
                    break
            if not ok:
                continue

            out = _completion_from_mapping(g.shape, row_period, col_period, mapping)
            candidates.append(
                {
                    "period": (row_period, col_period),
                    "mapping": mapping,
                    "out": out,
                    "symmetric": bool(np.array_equal(out, out[:, ::-1])),
                }
            )
    return candidates


def _minimal_period(full: np.ndarray) -> tuple[int, int]:
    height, width = full.shape
    for area in range(1, height * width + 1):
        for row_period in range(1, height + 1):
            if area % row_period != 0:
                continue
            col_period = area // row_period
            if col_period > width or height % row_period or width % col_period:
                continue
            if np.array_equal(full, np.tile(full[:row_period, :col_period], (height // row_period, width // col_period))):
                return row_period, col_period
    raise AssertionError("full pattern must have a finite period")


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the unique periodic completion whose filled canvas is left-right symmetric.

    Input structure:
    - A black background grid with even width and a sparse set of colored residual pixels.
    - The residual pixels come from a larger pattern that is periodic in rows and columns.
    - Some true foreground residue classes may have only one surviving witness, so the visible
      pixels can be compatible with multiple period hypotheses.

    Rule:
    - Enumerate every possible row period and column period.
    - Keep exactly those period pairs whose visible non-black pixels are residue-consistent:
      every occupied `(row mod P, col mod Q)` class has one color.
    - For each valid pair, fill every witnessed residue class across the whole canvas and leave
      unwitnessed classes black.
    - Return the unique completion whose full grid is exactly left-right symmetric.

    Key insight:
    - Periodicity alone is intentionally ambiguous: several candidate periods fit the residual
      pixels. The correct answer is the only periodic completion that also satisfies strict
      mirror symmetry about the vertical midline.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape
    assert 0 < height <= 30 and 0 < width <= 30
    assert width % 2 == 0

    candidates = _enumerate_candidates(g)
    symmetric = [cand for cand in candidates if cand["symmetric"]]
    assert len(symmetric) == 1, "expected exactly one symmetry-valid period candidate"
    return np.asarray(symmetric[0]["out"], dtype=int)


def _choose_repeat_counts(row_period: int, col_period: int) -> tuple[int, int]:
    row_options = [4, 5] if row_period == 2 else [3, 4]
    col_options = [
        repeat
        for repeat in (2, 3, 4)
        if col_period * repeat <= 18 and (col_period * repeat) % 2 == 0
    ]
    return random.choice(row_options), random.choice(col_options)


def _relabel_template(template: np.ndarray) -> np.ndarray:
    labels = sorted(int(x) for x in np.unique(template) if x != 0)
    colors = random.sample(_ARC_COLORS, k=len(labels))
    out = np.zeros_like(template)
    for src, dst in zip(labels, colors):
        out[template == src] = dst
    return out


def _residue_occurrences(
    shape: tuple[int, int], row_period: int, col_period: int
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    height, width = shape
    out: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in range(height):
        for c in range(width):
            out.setdefault((r % row_period, c % col_period), []).append((r, c))
    return out


def _witness_weight(r: int, c: int, height: int, width: int) -> int:
    score = 1
    if c < width // 2:
        score += 4
    if r < height // 2:
        score += 2
    if c != width - 1 - c:
        score += 1
    if r != height - 1 - r:
        score += 1
    return score


def _sample_witnesses(
    positions: list[tuple[int, int]], height: int, width: int
) -> list[tuple[int, int]]:
    count = 1 if random.random() < 0.72 else 2
    weights = [_witness_weight(r, c, height, width) for r, c in positions]
    picks: list[tuple[int, int]] = []
    for pos in random.choices(positions, weights=weights, k=count + 2):
        if pos not in picks:
            picks.append(pos)
        if len(picks) == count:
            return picks
    return picks or [random.choice(positions)]


def _valid_generated_instance(
    inp: np.ndarray, full: np.ndarray, row_period: int, col_period: int
) -> bool:
    candidates = _enumerate_candidates(inp)
    symmetric = [cand for cand in candidates if cand["symmetric"]]
    if len(candidates) < 2 or len(symmetric) != 1:
        return False

    chosen = symmetric[0]
    if chosen["period"] != (row_period, col_period):
        return False
    if not np.array_equal(np.asarray(chosen["out"], dtype=int), full):
        return False

    # Reject samples where "choose the smallest period" would also solve the puzzle.
    if not any(
        rp * cp < row_period * col_period
        for rp, cp in (cand["period"] for cand in candidates)
        if (rp, cp) != (row_period, col_period)
    ):
        return False

    if np.array_equal(inp, full) or np.array_equal(inp, inp[:, ::-1]):
        return False
    return True


def _sample_input_from_output(
    full: np.ndarray, row_period: int, col_period: int
) -> np.ndarray | None:
    height, width = full.shape
    occurrences = _residue_occurrences(full.shape, row_period, col_period)
    classes = [
        (rr, cc, int(full[rr, cc]))
        for rr in range(row_period)
        for cc in range(col_period)
        if full[rr, cc] != 0
    ]

    for _ in range(_WITNESS_ATTEMPTS):
        inp = np.zeros_like(full)
        shuffled = classes[:]
        random.shuffle(shuffled)
        for rr, cc, color in shuffled:
            picks = _sample_witnesses(occurrences[(rr, cc)], height, width)
            for r, c in picks:
                inp[r, c] = color

        if _valid_generated_instance(inp, full, row_period, col_period):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        base = np.array(random.choice(_FAMILY_TEMPLATES), copy=True)
        row_period, col_period = base.shape

        # Rolling the prototype varies the row-residue semantics without changing the rule.
        base = np.roll(base, shift=random.randrange(row_period), axis=0)
        tile = _relabel_template(base)

        row_repeats, col_repeats = _choose_repeat_counts(row_period, col_period)
        full = np.tile(tile, (row_repeats, col_repeats))
        if _minimal_period(full) != (row_period, col_period):
            continue
        assert np.array_equal(full, full[:, ::-1])

        inp = _sample_input_from_output(full, row_period, col_period)
        if inp is not None:
            return inp

    raise RuntimeError("failed to generate a valid ambiguous periodic-symmetry puzzle")
