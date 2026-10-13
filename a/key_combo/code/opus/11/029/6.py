from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
STAMP_SIZE = 3
STEP = 3
GEN_RETRY_LIMIT = 800
GEN_GRID_RANGE = (13, 18)  # high is exclusive
GEN_EXTRA_STAMP_CHOICES = (2, 3)
GEN_CLOSED_AREA_RANGE = (2, 12)
GEN_MAX_OVERLAP = 3
GEN_BLUE_AREA_RANGE = (18, 55)

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

CORNER_TO_STEP: dict[Coord, Coord] = {
    (0, 0): (-1, -1),
    (0, 2): (-1, 1),
    (2, 0): (1, -1),
    (2, 2): (1, 1),
}
STEP_TO_CORNER = {step: corner for corner, step in CORNER_TO_STEP.items()}
DIAGONAL_STEPS = tuple(STEP_TO_CORNER.keys())


def _shape(rows: tuple[str, str, str]) -> np.ndarray:
    return np.array([[1 if ch == "#" else 0 for ch in row] for row in rows], dtype=bool)


SEED_SHAPES: tuple[dict[str, object], ...] = (
    {"name": "L_tl", "family": "L", "mask": _shape(("###", "#..", "#.."))},
    {"name": "L_tr", "family": "L", "mask": _shape(("###", "..#", "..#"))},
    {"name": "L_bl", "family": "L", "mask": _shape(("#..", "#..", "###"))},
    {"name": "L_br", "family": "L", "mask": _shape(("..#", "..#", "###"))},
    {"name": "U_top", "family": "U", "mask": _shape(("#.#", "#.#", "###"))},
    {"name": "U_bottom", "family": "U", "mask": _shape(("###", "#.#", "#.#"))},
    {"name": "U_left", "family": "U", "mask": _shape(("###", "..#", "###"))},
    {"name": "U_right", "family": "U", "mask": _shape(("###", "#..", "###"))},
)


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[Coord] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[Coord]) -> BBox:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _stamp_once(out: np.ndarray, seed_mask: np.ndarray, top: int, left: int) -> bool:
    h, w = out.shape
    painted = False
    for r in range(STAMP_SIZE):
        for c in range(STAMP_SIZE):
            if not seed_mask[r, c]:
                continue
            rr = top + r
            cc = left + c
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = BLUE
                painted = True
    return painted


def _paint_chain(out: np.ndarray, seed_mask: np.ndarray, top: int, left: int, step: Coord) -> None:
    rr, cc = int(top), int(left)
    dr, dc = step
    while _stamp_once(out, seed_mask, rr, cc):
        rr += STEP * dr
        cc += STEP * dc


def _extract_seed(grid: np.ndarray, comp: list[Coord]) -> tuple[np.ndarray, Coord, Coord]:
    box = _bbox(comp)
    r0, c0, r1, c1 = box
    assert r1 - r0 == 2 and c1 - c0 == 2, "seed bbox must be exactly 3x3"
    sprite = _crop(grid, box)
    red_cells = np.argwhere(sprite == RED)
    assert len(red_cells) == 1, "each seed must contain exactly one red control cell"
    red_r, red_c = map(int, red_cells[0])
    corner = (red_r, red_c)
    assert corner in CORNER_TO_STEP, "red control must sit on a seed-bbox corner"
    seed_mask = sprite != BG
    assert seed_mask[corner], "red corner must belong to the seed shape"
    return seed_mask, (r0, c0), CORNER_TO_STEP[corner]


def _closed_regions(blue_mask: np.ndarray) -> list[list[Coord]]:
    h, w = blue_mask.shape
    reachable = np.zeros((h, w), dtype=bool)
    queue: deque[Coord] = deque()

    def try_enqueue(r: int, c: int) -> None:
        if 0 <= r < h and 0 <= c < w and not blue_mask[r, c] and not reachable[r, c]:
            reachable[r, c] = True
            queue.append((r, c))

    for r in range(h):
        try_enqueue(r, 0)
        try_enqueue(r, w - 1)
    for c in range(w):
        try_enqueue(0, c)
        try_enqueue(h - 1, c)

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            try_enqueue(nr, nc)

    regions: list[list[Coord]] = []
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if blue_mask[r, c] or reachable[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            region: list[Coord] = []
            while queue:
                rr, cc = queue.popleft()
                region.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not blue_mask[nr, nc]
                        and not reachable[nr, nc]
                        and not seen[nr, nc]
                    ):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            regions.append(region)
    return regions


def _filled_output_from_blue(blue_mask: np.ndarray) -> np.ndarray:
    out = np.zeros(blue_mask.shape, dtype=int)
    out[blue_mask] = BLUE
    for region in _closed_regions(blue_mask):
        for r, c in region:
            out[r, c] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each 3x3 seed block diagonally from its red-marked corner and fill the black region enclosed by the copied chains.

    Input structure:
    - The grid is a black background containing separate 3x3 blue seed shapes.
    - Each seed is an L or U pattern and contains exactly one red cell on a corner of its 3x3 bbox.
    - The red corner is only a direction marker; it belongs to the seed shape itself.

    Rule:
    - Split the non-black pixels into seed components and read the red corner of each seed bbox.
    - Replace the red marker with blue and stamp that 3x3 seed repeatedly along the matching diagonal, moving 3 rows and 3 columns per step, until the next stamp would be fully outside the grid.
    - After all diagonal stamp chains are drawn, find every black 4-connected region that cannot reach the grid boundary without crossing blue.
    - Fill every such enclosed region with green.

    Key insight:
    - The red corner does not survive in the output: it encodes the diagonal copy direction, and only the full pair of copied chains reveals which black cells become enclosed and must be filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _components(g != BG)
    assert comps, "expected at least one seed component"

    out = np.zeros_like(g)
    for comp in comps:
        seed_mask, origin, step = _extract_seed(g, comp)
        _paint_chain(out, seed_mask, origin[0], origin[1], step)

    return _filled_output_from_blue(out == BLUE)


def _visible_stamp_count(seed_mask: np.ndarray, origin: Coord, step: Coord, shape: tuple[int, int]) -> int:
    h, w = shape
    rr, cc = origin
    count = 0
    while True:
        visible = False
        for r in range(STAMP_SIZE):
            for c in range(STAMP_SIZE):
                if seed_mask[r, c] and 0 <= rr + r < h and 0 <= cc + c < w:
                    visible = True
                    break
            if visible:
                break
        if not visible:
            return count
        count += 1
        rr += STEP * step[0]
        cc += STEP * step[1]


def _render_seed_input(shape_mask: np.ndarray, origin: Coord, step: Coord, canvas_shape: tuple[int, int]) -> np.ndarray:
    h, w = canvas_shape
    out = np.zeros((h, w), dtype=int)
    r0, c0 = origin
    for r in range(STAMP_SIZE):
        for c in range(STAMP_SIZE):
            if shape_mask[r, c]:
                out[r0 + r, c0 + c] = BLUE
    red_r, red_c = STEP_TO_CORNER[step]
    out[r0 + red_r, c0 + red_c] = RED
    return out


def _render_chain_mask(shape_mask: np.ndarray, origin: Coord, step: Coord, canvas_shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(canvas_shape, dtype=int)
    _paint_chain(out, shape_mask, origin[0], origin[1], step)
    return out == BLUE


def _padded_boxes_disjoint(a: BBox, b: BBox, pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _candidate_origins(shape_mask: np.ndarray, step: Coord, canvas_shape: tuple[int, int]) -> list[Coord]:
    h, w = canvas_shape
    origins: list[Coord] = []
    for r0 in range(h - STAMP_SIZE + 1):
        for c0 in range(w - STAMP_SIZE + 1):
            extra = _visible_stamp_count(shape_mask, (r0, c0), step, canvas_shape) - 1
            if extra in GEN_EXTRA_STAMP_CHOICES:
                origins.append((r0, c0))
    random.shuffle(origins)
    return origins


def _move_red_marker(inp: np.ndarray, origin: Coord, old_step: Coord, new_corner: Coord) -> np.ndarray:
    out = inp.copy()
    old_r, old_c = STEP_TO_CORNER[old_step]
    out[origin[0] + old_r, origin[1] + old_c] = BLUE
    out[origin[0] + new_corner[0], origin[1] + new_corner[1]] = RED
    return out


def _direction_is_essential(inp: np.ndarray, expected: np.ndarray, shape_mask: np.ndarray, origin: Coord, step: Coord) -> bool:
    for corner, alt_step in CORNER_TO_STEP.items():
        if alt_step == step or not shape_mask[corner]:
            continue
        altered = _move_red_marker(inp, origin, step, corner)
        if not np.array_equal(main(altered), expected):
            return True
    return False


def _valid_generated_case(
    inp: np.ndarray,
    expected: np.ndarray,
    shape1: np.ndarray,
    origin1: Coord,
    step1: Coord,
    chain1: np.ndarray,
    shape2: np.ndarray,
    origin2: Coord,
    step2: Coord,
    chain2: np.ndarray,
) -> bool:
    comps = _components(inp != BG)
    if len(comps) != 2:
        return False

    for comp in comps:
        try:
            seed_mask, _origin, step = _extract_seed(inp, comp)
        except AssertionError:
            return False
        extra = _visible_stamp_count(seed_mask, _origin, step, inp.shape) - 1
        if extra not in GEN_EXTRA_STAMP_CHOICES:
            return False

    if np.array_equal(inp, expected):
        return False
    if not np.array_equal(main(inp), expected):
        return False
    if np.count_nonzero(expected == GREEN) == 0:
        return False

    closed = _closed_regions(chain1 | chain2)
    if len(closed) != 1:
        return False
    if not (GEN_CLOSED_AREA_RANGE[0] <= len(closed[0]) <= GEN_CLOSED_AREA_RANGE[1]):
        return False
    if _closed_regions(chain1) or _closed_regions(chain2):
        return False

    if not _direction_is_essential(inp, expected, shape1, origin1, step1):
        return False
    if not _direction_is_essential(inp, expected, shape2, origin2, step2):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randrange(*GEN_GRID_RANGE)
    w = random.randrange(*GEN_GRID_RANGE)
    canvas_shape = (h, w)

    step1, step2 = random.sample(DIAGONAL_STEPS, 2)
    family_mode = random.choice(("LL", "LU", "UL", "UU"))
    family1, family2 = family_mode[0], family_mode[1]

    shapes1 = [s for s in SEED_SHAPES if s["family"] == family1 and s["mask"][STEP_TO_CORNER[step1]]]
    shapes2 = [s for s in SEED_SHAPES if s["family"] == family2 and s["mask"][STEP_TO_CORNER[step2]]]
    if not shapes1 or not shapes2:
        return None

    spec1 = random.choice(shapes1)
    spec2 = random.choice(shapes2)
    shape1 = np.asarray(spec1["mask"], dtype=bool)
    shape2 = np.asarray(spec2["mask"], dtype=bool)

    origins1 = _candidate_origins(shape1, step1, canvas_shape)
    origins2 = _candidate_origins(shape2, step2, canvas_shape)
    if not origins1 or not origins2:
        return None

    for origin1 in origins1[:60]:
        box1 = (origin1[0], origin1[1], origin1[0] + 2, origin1[1] + 2)
        chain1 = _render_chain_mask(shape1, origin1, step1, canvas_shape)
        for origin2 in origins2[:60]:
            box2 = (origin2[0], origin2[1], origin2[0] + 2, origin2[1] + 2)
            if not _padded_boxes_disjoint(box1, box2, pad=1):
                continue

            inp = _render_seed_input(shape1, origin1, step1, canvas_shape)
            inp2 = _render_seed_input(shape2, origin2, step2, canvas_shape)
            if np.any((inp != BG) & (inp2 != BG)):
                continue
            inp = inp + inp2
            inp[inp > RED] = BLUE

            chain2 = _render_chain_mask(shape2, origin2, step2, canvas_shape)
            overlap = int(np.count_nonzero(chain1 & chain2))
            if overlap > GEN_MAX_OVERLAP:
                continue

            blue_mask = chain1 | chain2
            blue_area = int(np.count_nonzero(blue_mask))
            if not (GEN_BLUE_AREA_RANGE[0] <= blue_area <= GEN_BLUE_AREA_RANGE[1]):
                continue

            expected = _filled_output_from_blue(blue_mask)
            if _valid_generated_case(
                inp,
                expected,
                shape1,
                origin1,
                step1,
                chain1,
                shape2,
                origin2,
                step2,
                chain2,
            ):
                return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-enclosure seed puzzle")
