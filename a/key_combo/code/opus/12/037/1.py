from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
OBJECT = 5
HALO = 8

GRID_ROWS_RANGE = (24, 30)
GRID_COLS_RANGE = (24, 30)
GENERATION_RETRIES = 500
PLACEMENT_GAP = 1
EXTRA_HOLE_WEIGHTS = [0, 0, 1, 1, 2]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components from a boolean mask as (N,2) coord arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _count_holes_from_mask(mask: np.ndarray) -> int:
    """Count 4-connected background holes inside one object's bbox-local mask."""
    h, w = mask.shape
    bg_seen = np.zeros_like(mask, dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] or bg_seen[r, c]:
                continue
            q = deque([(r, c)])
            bg_seen[r, c] = True
            touches_edge = r in (0, h - 1) or c in (0, w - 1)
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not bg_seen[nr, nc]:
                        bg_seen[nr, nc] = True
                        if nr in (0, h - 1) or nc in (0, w - 1):
                            touches_edge = True
                        q.append((nr, nc))
            if not touches_edge:
                holes += 1
    return holes


def _parse_mask(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    turns = random.randrange(4)
    if turns:
        out = np.rot90(out, k=turns)
    return np.ascontiguousarray(out)


HOLE_FREE_TEMPLATES = (
    ("1111", "1111"),
    ("111", "110", "111"),
    ("1110", "1110", "0111"),
    ("1111", "0111", "0011"),
    ("111", "010", "111", "010"),
)

HOLE_TEMPLATES: dict[int, tuple[tuple[str, ...], ...]] = {
    1: (
        ("111", "101", "111"),
        ("1111", "1001", "1111"),
        ("1111", "1001", "1001", "1111"),
        ("111", "101", "101", "101", "111"),
    ),
    2: (
        ("11111", "10101", "11111"),
        ("1111", "1001", "1111", "1001", "1111"),
        ("11111", "10101", "10101", "11111"),
    ),
    3: (
        ("1111111", "1010101", "1111111"),
        ("11111", "10001", "11111", "10101", "11111"),
    ),
}


def _sample_object_mask(holes: int) -> np.ndarray:
    if holes == 0:
        template = random.choice(HOLE_FREE_TEMPLATES)
    else:
        template = random.choice(HOLE_TEMPLATES[holes])
    mask = _transform_mask(_parse_mask(template))
    assert _count_holes_from_mask(mask) == holes
    area = int(mask.sum())
    assert 8 <= area <= 20
    return mask


def _expanded_size(mask: np.ndarray, holes: int) -> tuple[int, int]:
    h, w = map(int, mask.shape)
    return h + 2 * holes, w + 2 * holes


def _place_expanded_boxes(
    grid_shape: tuple[int, int],
    specs: list[dict],
) -> bool:
    rows, cols = grid_shape
    occupied = np.zeros(grid_shape, dtype=bool)
    order = sorted(
        range(len(specs)),
        key=lambda idx: specs[idx]["expanded_shape"][0] * specs[idx]["expanded_shape"][1],
        reverse=True,
    )

    def rec(k: int) -> bool:
        if k == len(order):
            return True

        idx = order[k]
        spec = specs[idx]
        eh, ew = spec["expanded_shape"]
        starts = [
            (r, c)
            for r in range(1, rows - eh)
            for c in range(1, cols - ew)
            if not occupied[
                max(0, r - PLACEMENT_GAP) : min(rows, r + eh + PLACEMENT_GAP),
                max(0, c - PLACEMENT_GAP) : min(cols, c + ew + PLACEMENT_GAP),
            ].any()
        ]
        random.shuffle(starts)
        for r, c in starts:
            rr0 = max(0, r - PLACEMENT_GAP)
            cc0 = max(0, c - PLACEMENT_GAP)
            rr1 = min(rows, r + eh + PLACEMENT_GAP)
            cc1 = min(cols, c + ew + PLACEMENT_GAP)
            occupied[rr0:rr1, cc0:cc1] = True
            spec["expanded_top_left"] = (r, c)
            if rec(k + 1):
                return True
            occupied[rr0:rr1, cc0:cc1] = False
            spec.pop("expanded_top_left", None)
        return False

    return rec(0)


def _render_input_and_expected(
    grid_shape: tuple[int, int],
    specs: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = grid_shape
    inp = np.full((rows, cols), BG, dtype=int)
    expected = np.full((rows, cols), BG, dtype=int)

    for spec in specs:
        holes = spec["holes"]
        er0, ec0 = spec["expanded_top_left"]
        eh, ew = spec["expanded_shape"]
        mask = spec["mask"]
        mh, mw = mask.shape
        r0 = er0 + holes
        c0 = ec0 + holes
        r1 = r0 + mh
        c1 = c0 + mw

        if holes > 0:
            expected[er0 : er0 + eh, ec0 : ec0 + ew] = HALO
            expected[r0:r1, c0:c1] = BG

        inp[r0:r1, c0:c1][mask] = OBJECT
        expected[r0:r1, c0:c1][mask] = OBJECT

    return inp, expected


def _verify_metadata(inp: np.ndarray, specs: list[dict], expected: np.ndarray) -> None:
    comps = _components(inp == OBJECT)
    assert len(comps) == len(specs)
    found = sorted(
        (_bbox(comp), _count_holes_from_mask(_coords_to_local_mask(comp)))
        for comp in comps
    )
    planned = []
    for spec in specs:
        holes = spec["holes"]
        er0, ec0 = spec["expanded_top_left"]
        mh, mw = spec["mask"].shape
        r0 = er0 + holes
        c0 = ec0 + holes
        planned.append(((r0, c0, r0 + mh - 1, c0 + mw - 1), holes))
    assert found == sorted(planned)
    assert np.array_equal(main(inp), expected)


def _coords_to_local_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _sample_hole_counts() -> list[int]:
    counts = [0, 1, 2, 3]
    extra_count = random.randint(0, 2)
    for _ in range(extra_count):
        counts.append(random.choice(EXTRA_HOLE_WEIGHTS))
    random.shuffle(counts)
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """Expand each grey object by a teal rectangular halo whose thickness equals its enclosed hole count.

    Input structure
    - The grid is a black background with several separate grey 4-connected foreground objects.
    - Some grey objects enclose one or more black 4-connected holes fully inside their own bbox.
    - Objects are spaced apart so their halo regions do not need tie-breaking in normal task instances.

    Rule
    - Detect every grey 4-connected object and count how many enclosed black holes it has inside its bbox.
    - If an object has h=0 holes, leave it unchanged.
    - If an object has h>0 holes, expand its bbox by h cells in all four directions and paint only the outer rectangular ring teal.
    - Keep the original grey pixels and every black hole/background pixel inside the original bbox unchanged.

    Key insight
    - The same topological feature, the number of enclosed holes, determines both which objects transform and the exact halo thickness.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    objects: list[tuple[int, tuple[int, int, int, int]]] = []
    for coords in _components(g == OBJECT):
        local_mask = _coords_to_local_mask(coords)
        holes = _count_holes_from_mask(local_mask)
        objects.append((holes, _bbox(coords)))

    for holes, (r0, c0, r1, c1) in sorted(objects, key=lambda item: item[0]):
        if holes <= 0:
            continue
        er0 = max(0, r0 - holes)
        ec0 = max(0, c0 - holes)
        er1 = min(g.shape[0] - 1, r1 + holes)
        ec1 = min(g.shape[1] - 1, c1 + holes)
        ring = np.zeros_like(g, dtype=bool)
        ring[er0 : er1 + 1, ec0 : ec1 + 1] = True
        ring[r0 : r1 + 1, c0 : c1 + 1] = False
        out[ring] = HALO

    out[g == OBJECT] = OBJECT
    return out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid_shape = (
            random.randint(*GRID_ROWS_RANGE),
            random.randint(*GRID_COLS_RANGE),
        )
        specs = []
        for holes in _sample_hole_counts():
            mask = _sample_object_mask(holes)
            specs.append(
                {
                    "holes": holes,
                    "mask": mask,
                    "expanded_shape": _expanded_size(mask, holes),
                }
            )

        if not _place_expanded_boxes(grid_shape, specs):
            continue

        inp, expected = _render_input_and_expected(grid_shape, specs)
        try:
            _verify_metadata(inp, specs, expected)
        except AssertionError:
            continue

        hole_counts = sorted(spec["holes"] for spec in specs)
        if not {1, 2, 3}.issubset(hole_counts) or 0 not in hole_counts:
            continue
        return inp

    raise RuntimeError("failed to generate a valid input")
