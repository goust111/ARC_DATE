import itertools
import random

import numpy as np


DAMAGE = 5
VISIBLE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
ROT_FWD = (0, 1, 2, 3)   # TL, TR=ccw90, BL=180, BR=cw90
ROT_INV = (0, 3, 2, 1)

SIDE_RANGE = (6, 10)          # output side n in [6, 10]
COLOR_COUNT_RANGE = (3, 4)    # number of mother colors in [3, 4]
GEN_RETRIES = 500

WITNESS_SUBSETS = ((0, 1, 2), (0, 1, 3), (0, 2, 3))
SPECIAL_SUBSETS = ((0,), (1,), (2,), (3,))
SURVIVAL_OPTIONS = (
    (0,),
    (1,),
    (2,),
    (3,),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
    (0, 1, 2),
    (0, 1, 3),
    (0, 2, 3),
    (1, 2, 3),
    (0, 1, 2, 3),
)
SURVIVAL_WEIGHTS = (1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 1)


def _neighbors4(r: int, c: int, n: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < n:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < n:
        yield r, c + 1


def _split_quadrants(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    assert h == w and h % 2 == 0, "expected a 2n x 2n square grid"
    n = h // 2
    return [
        grid[:n, :n].copy(),
        grid[:n, n:].copy(),
        grid[n:, :n].copy(),
        grid[n:, n:].copy(),
    ]


def _aligned_quadrants(grid: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(q, ROT_INV[i]) for i, q in enumerate(_split_quadrants(grid))]


class _UnionFind:
    def __init__(self):
        self.parent: dict[tuple[int, int], tuple[int, int]] = {}

    def add(self, x: tuple[int, int]) -> None:
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x: tuple[int, int]) -> tuple[int, int]:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: tuple[int, int], b: tuple[int, int]) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _infer_token_to_mother(aligned: list[np.ndarray]) -> dict[tuple[int, int], int]:
    uf = _UnionFind()
    tokens: set[tuple[int, int]] = set()
    for qi, quad in enumerate(aligned):
        for color in np.unique(quad):
            color = int(color)
            if color != DAMAGE:
                token = (qi, color)
                tokens.add(token)
                uf.add(token)

    for i in range(4):
        for j in range(i + 1, 4):
            ij: dict[int, int] = {}
            ji: dict[int, int] = {}
            overlap = (aligned[i] != DAMAGE) & (aligned[j] != DAMAGE)
            assert np.any(overlap), "every quadrant pair must overlap on some survivor cells"
            for r, c in np.argwhere(overlap):
                ci = int(aligned[i][r, c])
                cj = int(aligned[j][r, c])
                prev = ij.get(ci)
                assert prev in (None, cj), "pairwise color correspondence must be functional"
                ij[ci] = cj
                prev = ji.get(cj)
                assert prev in (None, ci), "pairwise color correspondence must be bijective on overlaps"
                ji[cj] = ci
                uf.union((i, ci), (j, cj))

    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for token in tokens:
        groups.setdefault(uf.find(token), []).append(token)

    token_to_mother: dict[tuple[int, int], int] = {}
    for members in groups.values():
        anchors = {color for qi, color in members if qi == 0}
        assert len(anchors) <= 1, "one equivalence class cannot match multiple mother colors"
        assert anchors, "every visible color class must be anchored by some surviving TL cell"
        mother_color = next(iter(anchors))
        for token in members:
            token_to_mother[token] = mother_color

    return token_to_mother


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the undamaged mother grid from four rotated, recolored, partially greyed pinwheel quadrants.

    Input structure:
    - The input is a 2n x 2n square split into four n x n quadrants.
    - These quadrants come from one hidden multicolor mother grid arranged as
      TL = original, TR = 90-degree counterclockwise rotation, BL = 180-degree
      rotation, BR = 90-degree clockwise rotation.
    - Each quadrant is recolored by its own color bijection, and some cells in
      every quadrant are replaced by grey damage pixels.

    Rule:
    - Inverse-rotate all four quadrants back into the mother-grid coordinate system.
    - Use cells that survive at matching coordinates across quadrants to infer which
      recolored quadrant colors correspond to the same mother color.
    - Anchor every inferred color class by the TL quadrant, whose surviving colors
      already use the mother palette, then use the combined survivors to repair all
      grey damage and emit the full n x n mother grid.

    Key insight:
    - Coordinate alignment comes from the fixed pinwheel rotations, while exact color
      decoding comes only from survivor overlaps across the damaged quadrants.
    """
    g = np.asarray(grid, dtype=int)
    aligned = _aligned_quadrants(g)
    token_to_mother = _infer_token_to_mother(aligned)

    n = aligned[0].shape[0]
    out = np.full((n, n), DAMAGE, dtype=int)
    for r in range(n):
        for c in range(n):
            recovered = {
                token_to_mother[(qi, int(aligned[qi][r, c]))]
                for qi in range(4)
                if int(aligned[qi][r, c]) != DAMAGE
            }
            assert recovered, "every mother cell must survive in at least one quadrant"
            assert len(recovered) == 1, "all surviving views of one cell must agree"
            out[r, c] = next(iter(recovered))

    assert not np.any(out == DAMAGE), "the reconstructed mother grid cannot contain grey damage"
    return out


def _choose_region_seeds(n: int, k: int) -> list[tuple[int, int]] | None:
    min_sep = max(2, n // 3)
    cells = [(r, c) for r in range(n) for c in range(n)]
    for _ in range(80):
        random.shuffle(cells)
        seeds: list[tuple[int, int]] = []
        for cell in cells:
            if all(max(abs(cell[0] - sr), abs(cell[1] - sc)) >= min_sep for sr, sc in seeds):
                seeds.append(cell)
                if len(seeds) == k:
                    return seeds
    return None


def _grow_connected_regions(n: int, k: int, seeds: list[tuple[int, int]]) -> np.ndarray | None:
    labels = np.full((n, n), -1, dtype=int)
    frontiers = [set() for _ in range(k)]
    areas = [1] * k

    def add_frontier(idx: int, r: int, c: int) -> None:
        for rr, cc in _neighbors4(r, c, n):
            if labels[rr, cc] < 0:
                frontiers[idx].add((rr, cc))

    for idx, (r, c) in enumerate(seeds):
        labels[r, c] = idx
    for idx, (r, c) in enumerate(seeds):
        add_frontier(idx, r, c)

    unassigned = n * n - k
    while unassigned:
        choices = [idx for idx in range(k) if frontiers[idx]]
        if not choices:
            return None
        weights = [
            (1.0 / (areas[idx] ** 1.25)) * (max(1, len(frontiers[idx])) ** 0.35)
            for idx in choices
        ]
        idx = random.choices(choices, weights=weights, k=1)[0]
        cell = random.choice(tuple(frontiers[idx]))
        for frontier in frontiers:
            frontier.discard(cell)
        labels[cell] = idx
        areas[idx] += 1
        unassigned -= 1
        add_frontier(idx, cell[0], cell[1])

    return labels


def _make_mother(n: int, colors: tuple[int, ...]) -> np.ndarray | None:
    k = len(colors)
    seeds = _choose_region_seeds(n, k)
    if seeds is None:
        return None
    labels = _grow_connected_regions(n, k, seeds)
    if labels is None:
        return None

    mother = np.zeros((n, n), dtype=int)
    for idx, color in enumerate(colors):
        mother[labels == idx] = int(color)

    min_count = max(5, n * n // (3 * k))
    if any(int(np.count_nonzero(mother == color)) < min_count for color in colors):
        return None
    if any(np.array_equal(mother, np.rot90(mother, krot)) for krot in (1, 2, 3)):
        return None
    return mother


def _choose_bijections(mother_colors: tuple[int, ...]) -> list[dict[int, int]] | None:
    identity = tuple(mother_colors)
    k = len(mother_colors)
    candidates = [
        tup
        for tup in itertools.permutations(VISIBLE_COLORS, k)
        if tup != identity and set(tup) != set(identity)
    ]
    if len(candidates) < 3:
        candidates = [tup for tup in itertools.permutations(VISIBLE_COLORS, k) if tup != identity]
    if len(candidates) < 3:
        return None

    chosen = random.sample(candidates, 3)
    return [
        {src: src for src in mother_colors},
        {src: dst for src, dst in zip(mother_colors, chosen[0])},
        {src: dst for src, dst in zip(mother_colors, chosen[1])},
        {src: dst for src, dst in zip(mother_colors, chosen[2])},
    ]


def _sample_survivors(mother: np.ndarray, colors: tuple[int, ...]) -> list[np.ndarray] | None:
    n = mother.shape[0]
    assignments: dict[tuple[int, int], tuple[int, ...]] = {}
    used: set[tuple[int, int]] = set()

    for color in colors:
        positions = [tuple(map(int, rc)) for rc in np.argwhere(mother == color)]
        if len(positions) < len(WITNESS_SUBSETS):
            return None
        chosen = random.sample(positions, len(WITNESS_SUBSETS))
        for pos, subset in zip(chosen, WITNESS_SUBSETS):
            assignments[pos] = subset
            used.add(pos)

    remaining = [(r, c) for r in range(n) for c in range(n) if (r, c) not in used]
    if len(remaining) < len(SPECIAL_SUBSETS):
        return None
    random.shuffle(remaining)
    for subset in SPECIAL_SUBSETS:
        pos = remaining.pop()
        assignments[pos] = subset

    for pos in remaining:
        assignments[pos] = random.choices(SURVIVAL_OPTIONS, weights=SURVIVAL_WEIGHTS, k=1)[0]

    survive = [np.zeros((n, n), dtype=bool) for _ in range(4)]
    for (r, c), subset in assignments.items():
        for qi in subset:
            survive[qi][r, c] = True

    ratios = [float(mask.mean()) for mask in survive]
    if any(ratio < 0.28 or ratio > 0.82 for ratio in ratios):
        return None
    if any(mask.all() or not mask.any() for mask in survive):
        return None
    return survive


def _apply_map(grid: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = np.empty_like(grid)
    for src, dst in mapping.items():
        out[grid == src] = int(dst)
    return out


def _render_input(
    mother: np.ndarray,
    mappings: list[dict[int, int]],
    survive: list[np.ndarray],
) -> np.ndarray:
    n = mother.shape[0]
    quadrants = []
    for qi in range(4):
        aligned = np.full((n, n), DAMAGE, dtype=int)
        recolored = _apply_map(mother, mappings[qi])
        aligned[survive[qi]] = recolored[survive[qi]]
        quadrants.append(np.rot90(aligned, ROT_FWD[qi]))

    top = np.hstack([quadrants[0], quadrants[1]])
    bottom = np.hstack([quadrants[2], quadrants[3]])
    return np.vstack([top, bottom])


def _validate_semantics(
    mother: np.ndarray,
    mappings: list[dict[int, int]],
    survive: list[np.ndarray],
    inp: np.ndarray,
) -> bool:
    colors = tuple(sorted(int(v) for v in np.unique(mother)))
    try:
        recovered = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(recovered, mother):
        return False
    if any(np.array_equal(inp_part, mother) for inp_part in _split_quadrants(inp)):
        return False

    for qi in range(4):
        if not np.any(~survive[qi]):
            return False
    if not np.any(~survive[0]):
        return False

    for color in colors:
        if not np.any((mother == color) & survive[0]):
            return False
        for i in range(4):
            for j in range(i + 1, 4):
                if not np.any((mother == color) & survive[i] & survive[j]):
                    return False

    for qi in (1, 2, 3):
        if not np.any(np.all(np.stack([survive[k] == (k == qi) for k in range(4)]), axis=0)):
            return False

    for qi in (1, 2, 3):
        recolored = _apply_map(mother, mappings[qi])
        if np.array_equal(recolored, mother):
            return False
        if np.array_equal(np.rot90(recolored, ROT_FWD[qi]), mother):
            return False

    aligned = _aligned_quadrants(inp)
    if any(not np.array_equal(aligned[qi] != DAMAGE, survive[qi]) for qi in range(4)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*SIDE_RANGE)
    k = random.randint(*COLOR_COUNT_RANGE)
    mother_colors = tuple(random.sample(VISIBLE_COLORS, k))

    mother = _make_mother(n, mother_colors)
    if mother is None:
        return None
    mappings = _choose_bijections(mother_colors)
    if mappings is None:
        return None
    survive = _sample_survivors(mother, mother_colors)
    if survive is None:
        return None

    inp = _render_input(mother, mappings, survive)
    if not _validate_semantics(mother, mappings, survive, inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pinwheel recolor-repair task")
