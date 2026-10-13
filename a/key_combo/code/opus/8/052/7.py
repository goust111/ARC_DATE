import random
from collections import deque
from itertools import product

import numpy as np


BG = 0
OBJ = 5
AREA_TO_COLOR = {2: 3, 3: 2, 4: 1}
GRID_MIN = 15
GRID_MAX = 18
SLOT_ROWS = (1, 6, 11)
SLOT_COLS = (1, 6, 11)
MAX_ATTEMPTS = 400

_RNG = random.Random()
_GEN_CALL_COUNT = 0
_USED_OUTPUT_KEYS: set[tuple[int, ...]] = set()
_REMAINING_SEQUENCES: dict[int, list[tuple[int, ...]]] = {
    n: [seq for seq in product((2, 3, 4), repeat=n) if len(set(seq)) >= 2]
    for n in (3, 4, 5)
}

_SHAPE_COORDS = {
    2: [
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
    ],
    3: [
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 1)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
    ],
    4: [
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (2, 0), (3, 0)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 1), (1, 1), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (1, 1), (2, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 0), (1, 1), (2, 1)),
    ],
}


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _normalize_coords(coords):
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(grid: np.ndarray):
    seen = np.zeros(grid.shape, dtype=bool)
    comps = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != OBJ:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == OBJ:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(
                {
                    "coords": tuple(sorted(cells)),
                    "bbox": _bbox(cells),
                    "area": len(cells),
                }
            )
    return comps


def _seed_from_grid(grid: np.ndarray) -> np.ndarray:
    comps = _components(grid)
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    n = len(comps)
    seed = np.zeros((n, n), dtype=int)
    for idx, comp in enumerate(comps):
        area = comp["area"]
        if area not in AREA_TO_COLOR:
            raise ValueError(f"unexpected object area: {area}")
        seed[idx, idx] = AREA_TO_COLOR[area]
    return seed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count gray objects, encode their sizes on a diagonal seed, and self-stamp that seed into larger blocks.

    Input structure:
    - The input is a black background containing 3 to 5 separated gray 4-connected objects.
    - Every foreground object uses color 5 and has area 2, 3, or 4.
    - Objects are ordered by the row-major order of their bounding-box top-left corners.

    Rule:
    - Count the gray objects to obtain N.
    - Recolor each object by area: 2 pixels -> green, 3 pixels -> red, 4 pixels -> blue.
    - Build an N x N black seed and place the recolored values on its main diagonal in the sorted object order.
    - Build an (N*N) x (N*N) black output split into N x N blocks of size N x N, and copy the full seed into every block whose corresponding seed cell is non-black.

    Key insight:
    - The object count determines both the seed size and the number of stamped blocks, while each object's area determines the diagonal color placed in sorted order.
    """
    g = np.asarray(grid, dtype=int)
    seed = _seed_from_grid(g)
    n = int(seed.shape[0])
    out = np.zeros((n * n, n * n), dtype=int)
    for r in range(n):
        for c in range(n):
            if int(seed[r, c]) != BG:
                out[r * n : (r + 1) * n, c * n : (c + 1) * n] = seed
    return out


def _coords_to_patch(coords) -> np.ndarray:
    norm = _normalize_coords(coords)
    r1 = max(r for r, _ in norm)
    c1 = max(c for _, c in norm)
    patch = np.zeros((r1 + 1, c1 + 1), dtype=int)
    for r, c in norm:
        patch[r, c] = OBJ
    return patch


def _sample_sequence(rng: random.Random) -> tuple[int, ...]:
    global _GEN_CALL_COUNT
    early_targets = {
        0: 3,
        1: 4,
        2: 5,
    }
    if _GEN_CALL_COUNT in early_targets:
        n = early_targets[_GEN_CALL_COUNT]
        candidates = [seq for seq in _REMAINING_SEQUENCES[n] if set(seq) == {2, 3, 4}]
        if candidates:
            seq = rng.choice(candidates)
            _REMAINING_SEQUENCES[n].remove(seq)
            _GEN_CALL_COUNT += 1
            return seq

    available_ns = [n for n in (3, 4, 5) if _REMAINING_SEQUENCES[n]]
    if not available_ns:
        raise RuntimeError("exhausted unique output sequences")
    n = rng.choice(available_ns)
    seq = rng.choice(_REMAINING_SEQUENCES[n])
    _REMAINING_SEQUENCES[n].remove(seq)
    _GEN_CALL_COUNT += 1
    return seq


def _release_sequence(seq: tuple[int, ...]) -> None:
    global _GEN_CALL_COUNT
    _REMAINING_SEQUENCES[len(seq)].append(seq)
    _GEN_CALL_COUNT -= 1


def _pick_slots(rng: random.Random, n: int):
    slots = [(r, c) for r in SLOT_ROWS for c in SLOT_COLS]
    for _ in range(200):
        chosen = rng.sample(slots, n)
        rows = [r for r, _ in chosen]
        if len(set(rows)) < 2:
            continue
        if not any(r1 == r2 and c1 != c2 for i, (r1, c1) in enumerate(chosen) for r2, c2 in chosen[i + 1 :]):
            continue
        if not any(r1 < r2 and c1 > c2 for r1, c1 in chosen for r2, c2 in chosen):
            continue
        return chosen
    raise RuntimeError("failed to choose informative slot layout")


def _expanded_bboxes_do_not_touch(comps) -> bool:
    for i, comp_a in enumerate(comps):
        a0, b0, a1, b1 = comp_a["bbox"]
        for comp_b in comps[i + 1 :]:
            c0, d0, c1, d1 = comp_b["bbox"]
            if not (a1 + 1 < c0 or c1 + 1 < a0 or b1 + 1 < d0 or d1 + 1 < b0):
                return False
    return True


def _build_candidate(rng: random.Random, seq: tuple[int, ...]) -> np.ndarray:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    grid = np.zeros((h, w), dtype=int)
    slots = _pick_slots(rng, len(seq))
    used_shapes = {2: set(), 3: set(), 4: set()}

    for area, (top, left) in zip(seq, slots):
        shape_pool = list(range(len(_SHAPE_COORDS[area])))
        rng.shuffle(shape_pool)
        shape_idx = None
        for idx in shape_pool:
            if idx not in used_shapes[area] or len(used_shapes[area]) >= len(_SHAPE_COORDS[area]):
                shape_idx = idx
                break
        if shape_idx is None:
            shape_idx = shape_pool[0]
        used_shapes[area].add(shape_idx)
        patch = _coords_to_patch(_SHAPE_COORDS[area][shape_idx])
        ph, pw = patch.shape
        if top + ph > h or left + pw > w:
            raise RuntimeError("slot placement out of bounds")
        grid[top : top + ph, left : left + pw] |= patch
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False
    colors = set(map(int, np.unique(grid)))
    if not colors.issubset({BG, OBJ}):
        return False

    comps = _components(grid)
    if not 3 <= len(comps) <= 5:
        return False
    areas = [comp["area"] for comp in comps]
    if any(area not in AREA_TO_COLOR for area in areas):
        return False
    if len(set(areas)) < 2:
        return False
    if not _expanded_bboxes_do_not_touch(comps):
        return False

    tops = sorted((comp["bbox"][0], comp["bbox"][1]) for comp in comps)
    if len({r for r, _ in tops}) < 2:
        return False
    if not any(r1 == r2 and c1 != c2 for i, (r1, c1) in enumerate(tops) for r2, c2 in tops[i + 1 :]):
        return False
    if not any(r1 < r2 and c1 > c2 for r1, c1 in tops for r2, c2 in tops):
        return False

    seed = _seed_from_grid(grid)
    n = len(comps)
    if seed.shape != (n, n):
        return False
    if np.count_nonzero(seed) != n:
        return False
    if not np.array_equal(seed, np.diag(np.diag(seed))):
        return False

    sorted_areas = [comp["area"] for comp in sorted(comps, key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))]
    expected_diag = [AREA_TO_COLOR[a] for a in sorted_areas]
    if list(np.diag(seed)) != expected_diag:
        return False

    out = main(grid)
    if out.shape != (n * n, n * n):
        return False
    for br in range(n):
        for bc in range(n):
            block = out[br * n : (br + 1) * n, bc * n : (bc + 1) * n]
            if int(seed[br, bc]) == BG:
                if np.any(block != BG):
                    return False
            else:
                if not np.array_equal(block, seed):
                    return False
    return True


def _output_key(grid: np.ndarray) -> tuple[int, ...]:
    out = main(grid)
    return tuple([out.shape[0], out.shape[1], *map(int, out.flatten())])


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        seq = _sample_sequence(_RNG)
        try:
            grid = _build_candidate(_RNG, seq)
        except RuntimeError:
            _release_sequence(seq)
            continue
        if not _validate_instance(grid):
            _release_sequence(seq)
            continue
        key = _output_key(grid)
        if key in _USED_OUTPUT_KEYS:
            _release_sequence(seq)
            continue
        _USED_OUTPUT_KEYS.add(key)
        return grid
    raise RuntimeError("failed to generate a valid input")
