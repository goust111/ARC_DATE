"""ARC task: reconstruct an unseen template from two complementary fragments."""

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]


def _components(mask: np.ndarray) -> list[set[Coord]]:
    """Return 8-connected true components in row/column coordinates."""
    seen: set[Coord] = set()
    result: list[set[Coord]] = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        start = (int(r), int(c))
        if start in seen:
            continue
        comp = {start}
        seen.add(start)
        queue = deque([start])
        while queue:
            rr, cc = queue.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == dc == 0:
                        continue
                    p = (rr + dr, cc + dc)
                    if (0 <= p[0] < h and 0 <= p[1] < w
                            and bool(mask[p]) and p not in seen):
                        seen.add(p)
                        comp.add(p)
                        queue.append(p)
        result.append(comp)
    return result


def _linear(p: Coord, variant: int) -> Coord:
    """One of the eight rotations/reflections, without a translation."""
    r, c = p
    transforms = (
        (r, c), (r, -c), (-r, c), (-r, -c),
        (c, r), (c, -r), (-c, r), (-c, -r),
    )
    return transforms[variant]


def _oriented(points: set[Coord], signature: set[Coord], variant: int) -> set[Coord]:
    """Transform points and set the transformed signature's upper-left to (0,0)."""
    ts = {_linear(p, variant) for p in signature}
    mr = min(r for r, _ in ts)
    mc = min(c for _, c in ts)
    return {(r - mr, c - mc) for r, c in (_linear(p, variant) for p in points)}


def _normal(points: set[Coord]) -> set[Coord]:
    mr = min(r for r, _ in points)
    mc = min(c for _, c in points)
    return {(r - mr, c - mc) for r, c in points}


def main(grid: np.ndarray) -> np.ndarray:
    """Rebuild an unseen two-color template and complete every signature-only copy.

    Input structure:
    - On a uniform background are separated rotated/reflected copies of one template.
    - One signature color occurs in every copy and its asymmetric signature is complete.
    - Exactly two copies also contain the pattern color, each missing a different part;
      one or more other copies contain the signature and no pattern pixels.

    Rule:
    - Use the complete signatures to align the two partial copies in one coordinate
      frame and take the union of their pattern pixels.
    - At every signature-only copy, transform that reconstructed pattern to the
      orientation shown by its signature and paint all its pattern pixels.
    - Preserve every pixel already present in the input.

    Key insight:
    - No displayed copy is the full template: only the orientation-aware union of
      the two complementary fragments supplies every pixel needed at the targets.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    comps = _components(g != bg)
    assert len(comps) >= 3, "expected two fragments and at least one target"

    component_colors = [{int(g[p]) for p in comp} for comp in comps]
    common = set.intersection(*component_colors)
    assert len(common) == 1, "one structural signature color must occur in every copy"
    signature_color = common.pop()
    other_colors = set().union(*component_colors) - {signature_color}
    assert len(other_colors) == 1, "expected one pattern color"
    pattern_color = other_colors.pop()

    sources = [comp for comp in comps if pattern_color in {int(g[p]) for p in comp}]
    targets = [comp for comp in comps if all(int(g[p]) == signature_color for p in comp)]
    assert len(sources) == 2 and targets

    def split(comp: set[Coord]) -> tuple[set[Coord], set[Coord]]:
        sig = {p for p in comp if int(g[p]) == signature_color}
        pat = {p for p in comp if int(g[p]) == pattern_color}
        return sig, pat

    sig0_abs, pat0_abs = split(sources[0])
    # Canonical coordinates use the first source signature's own upper-left origin.
    sr0 = min(r for r, _ in sig0_abs)
    sc0 = min(c for _, c in sig0_abs)
    sig0 = {(r - sr0, c - sc0) for r, c in sig0_abs}
    pat0 = {(r - sr0, c - sc0) for r, c in pat0_abs}

    sig1_abs, pat1_abs = split(sources[1])
    sig1_local = {(r, c) for r, c in sig1_abs}
    matches = [v for v in range(8) if _oriented(sig1_local, sig1_local, v) == sig0]
    assert len(matches) == 1, "signature orientation must be unique"
    pat1 = _oriented(pat1_abs, sig1_abs, matches[0])
    full_pattern = pat0 | pat1
    assert pat0 < full_pattern and pat1 < full_pattern, "both fragments must be incomplete"

    out = g.copy()
    for target in targets:
        observed = {p for p in target if int(g[p]) == signature_color}
        normalized_observed = _normal(observed)
        variants = [v for v in range(8) if _oriented(sig0, sig0, v) == normalized_observed]
        assert len(variants) == 1, "target signature orientation must be unique"
        variant = variants[0]
        transformed_pattern = _oriented(full_pattern, sig0, variant)
        top = min(r for r, _ in observed)
        left = min(c for _, c in observed)
        for r, c in transformed_pattern:
            rr, cc = top + r, left + c
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]
            assert int(out[rr, cc]) in (bg, pattern_color)
            out[rr, cc] = pattern_color
    return out


def _variants(points: set[Coord], signature: set[Coord]) -> list[set[Coord]]:
    return [_oriented(points, signature, v) for v in range(8)]


def _random_signature(rng: random.Random) -> set[Coord]:
    """Sample a connected signature whose eight dihedral images are distinct."""
    cells = [(r, c) for r in range(4) for c in range(4)]
    for _ in range(500):
        sig = set(rng.sample(cells, rng.choice((4, 5))))
        if len(_components(np.array([[((r, c) in sig) for c in range(4)] for r in range(4)]))) != 1:
            continue
        forms = {frozenset(_oriented(sig, sig, v)) for v in range(8)}
        if len(forms) == 8 and len({r for r, _ in sig}) >= 3 and len({c for _, c in sig}) >= 3:
            return sig
    raise RuntimeError("could not sample an asymmetric signature")


def _place(occupied: np.ndarray, shape: set[Coord], rng: random.Random) -> tuple[int, int] | None:
    """Choose an in-bounds position with a two-cell visual separation margin."""
    h = max(r for r, _ in shape) - min(r for r, _ in shape) + 1
    w = max(c for _, c in shape) - min(c for _, c in shape) + 1
    candidates = [(r, c) for r in range(1, occupied.shape[0] - h)
                  for c in range(1, occupied.shape[1] - w)]
    rng.shuffle(candidates)
    for top, left in candidates:
        region = occupied[max(0, top - 2):min(occupied.shape[0], top + h + 2),
                          max(0, left - 2):min(occupied.shape[1], left + w + 2)]
        if not bool(region.any()):
            return top, left
    return None


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    bg, signature_color, pattern_color = rng.sample(range(10), 3)
    signature = _random_signature(rng)

    # The pattern includes a variable irregular fringe around the signature.
    domain = {(r, c) for r in range(5) for c in range(5)}
    available = list(domain - signature)
    rng.shuffle(available)
    pattern = set(available[:rng.randint(8, 13)])
    # Require each pattern cell to belong to the same 8-connected complete object.
    full = signature | pattern
    mask = np.zeros((5, 5), dtype=bool)
    for p in full:
        mask[p] = True
    if len(_components(mask)) != 1:
        return None

    # Complementary, genuinely non-substitutable fragments: each owns >=2 pixels
    # absent from the other, while optional shared pixels make the union operation visible.
    pats = list(pattern)
    rng.shuffle(pats)
    a_only = set(pats[:2])
    b_only = set(pats[2:4])
    rest = pats[4:]
    shared = {p for p in rest if rng.random() < 0.55}
    remainder = [p for p in rest if p not in shared]
    frag_a = shared | a_only | {p for p in remainder if rng.random() < 0.5}
    frag_b = shared | b_only | (set(remainder) - frag_a)
    if frag_a | frag_b != pattern or not (frag_a < pattern and frag_b < pattern):
        return None

    target_count = rng.randint(1, 3)
    count = 2 + target_count
    H, W = rng.randint(18, 30), rng.randint(18, 30)
    grid = np.full((H, W), bg, dtype=int)
    expected = grid.copy()
    occupied = np.zeros((H, W), dtype=bool)
    orientations = rng.sample(range(8), count) if count <= 8 else [rng.randrange(8) for _ in range(count)]

    for i in range(count):
        v = orientations[i]
        oriented_sig = _oriented(signature, signature, v)
        oriented_full_pattern = _oriented(pattern, signature, v)
        visible_pattern = (_oriented(frag_a, signature, v) if i == 0 else
                           _oriented(frag_b, signature, v) if i == 1 else set())
        footprint = oriented_sig | oriented_full_pattern
        pos = _place(occupied, footprint, rng)
        if pos is None:
            return None
        top, left = pos
        for r, c in footprint:
            occupied[top + r, left + c] = True
        for r, c in oriented_sig:
            grid[top + r, left + c] = signature_color
            expected[top + r, left + c] = signature_color
        for r, c in visible_pattern:
            grid[top + r, left + c] = pattern_color
            expected[top + r, left + c] = pattern_color
        if i >= 2:
            for r, c in oriented_full_pattern:
                expected[top + r, left + c] = pattern_color

    # Semantic validation, not merely a shape/type check.
    try:
        solved = main(grid)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(solved, expected):
        return None
    additions = (expected == pattern_color) & (grid == bg)
    if int(additions.sum()) != target_count * len(pattern):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a diverse, uniquely alignable instance and validate its semantics."""
    rng = random.Random()
    for _ in range(600):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid half-shadow template instance")
