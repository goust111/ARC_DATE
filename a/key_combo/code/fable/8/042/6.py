from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

ROUND_TO_COLOR = {
    1: BLUE,
    2: RED,
    3: GREEN,
    4: YELLOW,
}

_NBR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _patch_from_coords(coords: list[tuple[int, int]]) -> np.ndarray:
    arr = np.array(coords, dtype=int)
    arr -= arr.min(axis=0)
    h, w = arr.max(axis=0) + 1
    patch = np.zeros((int(h), int(w)), dtype=bool)
    patch[arr[:, 0], arr[:, 1]] = True
    return patch


def _trim(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _neighbor_counts(mask: np.ndarray) -> np.ndarray:
    counts = np.zeros(mask.shape, dtype=int)
    counts[1:, :] += mask[:-1, :]
    counts[:-1, :] += mask[1:, :]
    counts[:, 1:] += mask[:, :-1]
    counts[:, :-1] += mask[:, 1:]
    return counts


def _peel_rounds_from_mask(mask: np.ndarray) -> int:
    current = mask.astype(bool).copy()
    rounds = 0
    while current.any():
        peel = current & (_neighbor_counts(current) < 2)
        if not peel.any():
            raise AssertionError("component contains a non-eroding 2-core")
        current &= ~peel
        rounds += 1
    return rounds


def _peel_rounds(coords: np.ndarray) -> int:
    local = _patch_from_coords([tuple(map(int, rc)) for rc in coords])
    return _peel_rounds_from_mask(local)


def _gray_components(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    mask = g == GREY
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if visited[rr, cc]:
            continue
        stack = [(rr, cc)]
        visited[rr, cc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc2 = stack.pop()
            coords.append((cr, cc2))
            for dr, dc in _NBR4:
                nr, nc = cr + dr, cc2 + dc
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        components.append(np.array(coords, dtype=int))
    return components


def _component_records(grid: np.ndarray) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for coords in _gray_components(grid):
        records.append(
            {
                "coords": coords,
                "area": int(coords.shape[0]),
                "rounds": _peel_rounds(coords),
            }
        )
    return records


def _expected_output(grid: np.ndarray) -> np.ndarray:
    out = np.full(np.asarray(grid).shape, BLACK, dtype=int)
    for rec in _component_records(grid):
        rounds = int(rec["rounds"])
        color = ROUND_TO_COLOR[rounds]
        coords = rec["coords"]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray component by how many synchronous leaf-peeling rounds it survives.

    Input structure:
    - The grid is a black background with several separated gray 4-connected components.
    - Each gray component keeps its own shape and does not touch the others.
    - Different components can have similar area while differing in how long they survive erosion.

    Rule:
    - For each gray component independently, repeatedly and simultaneously erase every pixel that has fewer than two gray 4-neighbors inside that same component.
    - Count how many such rounds are needed until the component disappears completely.
    - Output the original component shape in place, recolored blue for 1 round, red for 2 rounds, green for 3 rounds, and yellow for 4 rounds; keep the background black.

    Key insight:
    - The color is determined by iterative support-based peeling depth, not by raw area, width, height, or any visible one-shot size statistic.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BLACK, GREY}, "input should contain only black background and gray components"

    out = np.full(g.shape, BLACK, dtype=int)
    for rec in _component_records(g):
        rounds = int(rec["rounds"])
        assert rounds in ROUND_TO_COLOR, "generator should keep component survival rounds within 1..4"
        coords = rec["coords"]
        out[coords[:, 0], coords[:, 1]] = ROUND_TO_COLOR[rounds]
    return out


_R1_TEMPLATES = [
    _patch_from_coords([(0, 0)]),
    _patch_from_coords([(0, 0), (0, 1)]),
]

_R2_AREA5_TEMPLATES = [
    _patch_from_coords([(1, 0), (0, 1), (1, 1), (1, 2), (2, 1)]),
    _patch_from_coords([(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)]),
]

_R2_TEMPLATES = _R2_AREA5_TEMPLATES + [
    _patch_from_coords([(0, 0), (0, 1), (0, 2), (0, 3)]),
    _patch_from_coords([(0, 0), (1, 0), (2, 0), (2, 1)]),
    _patch_from_coords([(1, 0), (1, 1), (1, 2), (0, 1), (2, 1), (1, 3)]),
]

_R3_AREA5_TEMPLATES = [
    _patch_from_coords([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]),
    _patch_from_coords([(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]),
    _patch_from_coords([(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]),
]

_R3_TEMPLATES = _R3_AREA5_TEMPLATES + [
    _patch_from_coords([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]),
    _patch_from_coords([(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)]),
]

_R4_TEMPLATES = [
    _patch_from_coords([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)]),
    _patch_from_coords([(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)]),
    _patch_from_coords([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)]),
    _patch_from_coords([(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4)]),
]

_ROUND_TEMPLATE_BANK = {
    1: _R1_TEMPLATES,
    2: _R2_TEMPLATES,
    3: _R3_TEMPLATES,
    4: _R4_TEMPLATES,
}


for _rounds, _patches in _ROUND_TEMPLATE_BANK.items():
    for _patch in _patches:
        assert _peel_rounds_from_mask(_patch) == _rounds


def _random_transform(patch: np.ndarray) -> np.ndarray:
    out = patch.copy()
    if random.random() < 0.5:
        out = np.fliplr(out)
    out = np.rot90(out, k=random.randrange(4))
    return _trim(out)


def _expand8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _place_components(patches: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray | None:
    h, w = shape
    grid = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    forbidden = np.zeros((h, w), dtype=bool)
    for patch in patches:
        ph, pw = patch.shape
        if ph > h or pw > w:
            return None
        placed = False
        for _ in range(200):
            r0 = random.randint(0, h - ph)
            c0 = random.randint(0, w - pw)
            block = np.zeros((h, w), dtype=bool)
            block[r0 : r0 + ph, c0 : c0 + pw] = patch
            if np.any(block & forbidden):
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw][patch] = GREY
            occupied |= block
            forbidden = _expand8(occupied)
            placed = True
            break
        if not placed:
            return None
    return grid


def _semantic_checks(inp: np.ndarray) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if set(map(int, np.unique(inp))) != {BLACK, GREY}:
        return False

    records = _component_records(inp)
    if len(records) < 4:
        return False

    rounds = [int(rec["rounds"]) for rec in records]
    if set(rounds) != {1, 2, 3, 4}:
        return False
    if any(r not in ROUND_TO_COLOR for r in rounds):
        return False

    area_round_pairs = [(int(rec["area"]), int(rec["rounds"])) for rec in records]
    if not any(a0 == a1 and r0 != r1 for i, (a0, r0) in enumerate(area_round_pairs) for a1, r1 in area_round_pairs[i + 1 :]):
        return False

    out = _expected_output(inp)
    if not np.array_equal(main(inp), out):
        return False

    for rec in records:
        coords = rec["coords"]
        color = ROUND_TO_COLOR[int(rec["rounds"])]
        if not np.all(out[coords[:, 0], coords[:, 1]] == color):
            return False

    return True


def _build_candidate() -> np.ndarray | None:
    patches = [
        _random_transform(random.choice(_R1_TEMPLATES)),
        _random_transform(random.choice(_R2_AREA5_TEMPLATES)),
        _random_transform(random.choice(_R3_AREA5_TEMPLATES)),
        _random_transform(random.choice(_R4_TEMPLATES)),
    ]

    extra_count = random.choices([0, 1, 2], weights=[2, 3, 2], k=1)[0]
    for _ in range(extra_count):
        target_round = random.choices([1, 2, 3, 4], weights=[1, 3, 3, 2], k=1)[0]
        patches.append(_random_transform(random.choice(_ROUND_TEMPLATE_BANK[target_round])))

    random.shuffle(patches)
    for _ in range(60):
        shape = (random.randint(14, 24), random.randint(14, 24))
        grid = _place_components(patches, shape)
        if grid is not None:
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        inp = _build_candidate()
        if inp is not None and _semantic_checks(inp):
            return inp
    raise RuntimeError("failed to generate a valid erosion-round puzzle")
