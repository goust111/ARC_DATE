from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

TEMPLATE_SIZE = 3
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _mask_connected(mask: np.ndarray) -> bool:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not cells:
        return False
    q = deque([cells[0]])
    seen = {cells[0]}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _enumerate_template_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1 << (TEMPLATE_SIZE * TEMPLATE_SIZE)):
        mask = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=bool)
        for idx in range(TEMPLATE_SIZE * TEMPLATE_SIZE):
            if bits & (1 << idx):
                mask[idx // TEMPLATE_SIZE, idx % TEMPLATE_SIZE] = True
        if not 4 <= int(mask.sum()) <= 6:
            continue
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if len(rows) != TEMPLATE_SIZE or len(cols) != TEMPLATE_SIZE:
            continue
        if not _mask_connected(mask):
            continue
        if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)) or np.array_equal(mask, np.rot90(mask, 2)):
            continue
        if not (np.any(~mask[0]) and np.any(~mask[-1]) and np.any(~mask[:, 0]) and np.any(~mask[:, -1])):
            continue
        masks.append(mask)
    assert masks, "expected non-empty template catalogue"
    return masks


TEMPLATE_CATALOG = _enumerate_template_masks()


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    window = grid[top : top + h, left : left + w]
    window[mask] = color


def _extract_template_mask(grid: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    blue_comps = _components_of_color(grid, BLUE)
    assert blue_comps, "expected blue template/copies"

    candidates = []
    for comp in blue_comps:
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 + 1 != TEMPLATE_SIZE or c1 - c0 + 1 != TEMPLATE_SIZE:
            continue
        mask = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=bool)
        for r, c in comp:
            mask[r - r0, c - c0] = True
        candidates.append((r0, c0, mask))

    assert candidates, "expected a 3x3 blue template"
    tr, tc, mask = min(candidates, key=lambda item: (item[0], item[1]))
    return (tr, tc), mask


def _find_existing_anchors(grid: np.ndarray, box: tuple[int, int, int, int], template_mask: np.ndarray) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    anchors: list[tuple[int, int]] = []
    for ar in range(r0, r1 - TEMPLATE_SIZE + 2):
        for ac in range(c0, c1 - TEMPLATE_SIZE + 2):
            blue_mask = grid[ar : ar + TEMPLATE_SIZE, ac : ac + TEMPLATE_SIZE] == BLUE
            if np.array_equal(blue_mask, template_mask):
                anchors.append((ar, ac))

    assert anchors, "expected at least one existing copy inside the marker box"
    inside_blue = grid[r0 : r1 + 1, c0 : c1 + 1] == BLUE
    covered = np.zeros_like(inside_blue, dtype=bool)
    for ar, ac in anchors:
        rr = ar - r0
        cc = ac - c0
        covered[rr : rr + TEMPLATE_SIZE, cc : cc + TEMPLATE_SIZE] |= template_mask
    assert np.array_equal(inside_blue, covered), "bbox blue pixels must be exactly full template copies"
    return sorted(anchors)


def _infer_periods(anchors: list[tuple[int, int]]) -> tuple[int, int]:
    row_to_cols: dict[int, list[int]] = defaultdict(list)
    col_to_rows: dict[int, list[int]] = defaultdict(list)
    for r, c in anchors:
        row_to_cols[r].append(c)
        col_to_rows[c].append(r)

    dx_candidates = []
    for cols in row_to_cols.values():
        cols = sorted(cols)
        for i in range(len(cols) - 1):
            dx_candidates.append(cols[i + 1] - cols[i])

    dy_candidates = []
    for rows in col_to_rows.values():
        rows = sorted(rows)
        for i in range(len(rows) - 1):
            dy_candidates.append(rows[i + 1] - rows[i])

    assert dx_candidates and dy_candidates, "periods must be inferable from both axes"
    dx = min(dx_candidates)
    dy = min(dy_candidates)
    assert dx > 0 and dy > 0
    return dy, dx


def _all_lattice_anchors(
    base: tuple[int, int], box: tuple[int, int, int, int], dy: int, dx: int
) -> list[tuple[int, int]]:
    r0, c0 = base
    _br0, _bc0, br1, bc1 = box
    anchors: list[tuple[int, int]] = []
    rr = r0
    while rr + TEMPLATE_SIZE - 1 <= br1:
        cc = c0
        while cc + TEMPLATE_SIZE - 1 <= bc1:
            anchors.append((rr, cc))
            cc += dx
        rr += dy
    return anchors


def _analyze(grid: np.ndarray) -> dict[str, object]:
    markers = np.argwhere(grid == RED)
    assert 4 <= len(markers) <= 6, "expected 4-6 red markers"
    mr0 = int(markers[:, 0].min())
    mc0 = int(markers[:, 1].min())
    mr1 = int(markers[:, 0].max())
    mc1 = int(markers[:, 1].max())
    box = (mr0, mc0, mr1, mc1)

    (tr, tc), template_mask = _extract_template_mask(grid)
    assert tr < mr0 and tc < mc0, "template must be isolated near the upper-left corner"

    anchors = _find_existing_anchors(grid, box, template_mask)
    base = min(anchors)
    dy, dx = _infer_periods(anchors)

    for r, c in anchors:
        assert r >= base[0] and c >= base[1], "base must be the top-left-most existing copy"
        assert (r - base[0]) % dy == 0 and (c - base[1]) % dx == 0, "copies must lie on one lattice"

    lattice = _all_lattice_anchors(base, box, dy, dx)
    anchor_set = set(anchors)
    assert set(anchors).issubset(set(lattice)), "existing copies must be a subset of the lattice"
    assert len(anchor_set) < len(lattice), "expected at least one missing copy to complete"
    assert len({r for r, _ in anchors}) >= 2 and len({c for _, c in anchors}) >= 2

    return {
        "box": box,
        "template_origin": (tr, tc),
        "template_mask": template_mask,
        "anchors": anchors,
        "base": base,
        "dy": dy,
        "dx": dx,
        "lattice": lattice,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Use the red-marker bounding box as a tiling region, infer the copy periods, complete the lattice, and recolor only the newly added copies green.

    Input structure:
    - The grid is mostly black, with one isolated blue 3x3 template sprite near the upper-left corner.
    - Four to six red marker pixels define a rectangular tiling region by their minimal axis-aligned bounding box.
    - Inside that box there are several full blue copies of the same 3x3 template on a translation lattice, while some lattice positions are empty.

    Rule:
    - Read the upper-left blue component as the template mask.
    - Take the bounding box of all red markers as the region in which copies may appear.
    - Detect the existing blue copies inside that region, infer the horizontal and vertical periods from their row/column separations, and use the top-left-most existing copy as the lattice origin.
    - Place the template at every lattice position that still fits in the red-marker box, keeping already present copy pixels blue and painting pixels from previously empty copies green; finally erase all red markers to black.

    Key insight:
    - The task combines three independent clues: the blue template says what to tile, the red markers say where the tiling is allowed, and the partially filled copy positions say which translation periods to use.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze(g)
    template_mask = info["template_mask"]
    anchors = set(info["anchors"])
    lattice = info["lattice"]

    out = g.copy()
    out[out == RED] = BLACK
    for ar, ac in lattice:
        color = BLUE if (ar, ac) in anchors else GREEN
        _paint_mask(out, ar, ac, template_mask, color)
    return out


def _occupancy_options(rows: int, cols: int) -> list[np.ndarray]:
    total = rows * cols
    min_on = int(np.ceil(0.4 * total))
    max_on = int(np.floor(0.6 * total))
    options: list[np.ndarray] = []
    for bits in range(1 << total):
        arr = np.zeros((rows, cols), dtype=bool)
        for idx in range(total):
            if bits & (1 << idx):
                arr[idx // cols, idx % cols] = True
        count = int(arr.sum())
        if not (min_on <= count <= max_on):
            continue
        if not arr[0, 0]:
            continue
        if count == total:
            continue
        has_h = any(arr[r, c] and arr[r, c + 1] for r in range(rows) for c in range(cols - 1))
        has_v = any(arr[r, c] and arr[r + 1, c] for r in range(rows - 1) for c in range(cols))
        if not (has_h and has_v):
            continue
        options.append(arr)
    return options


OCCUPANCY_OPTIONS = {
    (2, 3): _occupancy_options(2, 3),
    (3, 2): _occupancy_options(3, 2),
    (3, 3): _occupancy_options(3, 3),
}


def _layout_choice() -> tuple[int, int, int, int, int, int]:
    while True:
        rows, cols = random.choice(sorted(OCCUPANCY_OPTIONS))
        dy = 4 if rows == 3 else random.choice((4, 5))
        dx = 4 if cols == 3 else random.choice((4, 5))
        span_h = TEMPLATE_SIZE + (rows - 1) * dy
        span_w = TEMPLATE_SIZE + (cols - 1) * dx
        min_h_slack = max(0, 8 - span_h)
        max_h_slack = min(dy - 1, 12 - span_h)
        min_w_slack = max(0, 8 - span_w)
        max_w_slack = min(dx - 1, 12 - span_w)
        if min_h_slack > max_h_slack or min_w_slack > max_w_slack:
            continue
        h_slack = random.randint(min_h_slack, max_h_slack)
        w_slack = random.randint(min_w_slack, max_w_slack)
        return rows, cols, dy, dx, span_h + h_slack, span_w + w_slack


def _render_local_mask(
    box_h: int, box_w: int, template_mask: np.ndarray, anchors: list[tuple[int, int]]
) -> np.ndarray:
    mask = np.zeros((box_h, box_w), dtype=bool)
    for r, c in anchors:
        mask[r : r + TEMPLATE_SIZE, c : c + TEMPLATE_SIZE] |= template_mask
    return mask


def _choose_markers(full_mask: np.ndarray) -> list[tuple[int, int]] | None:
    box_h, box_w = full_mask.shape
    allowed = ~full_mask
    all_allowed = [tuple(map(int, rc)) for rc in np.argwhere(allowed)]
    if len(all_allowed) < 4:
        return None

    top = [(0, c) for c in range(box_w) if allowed[0, c]]
    bottom = [(box_h - 1, c) for c in range(box_w) if allowed[box_h - 1, c]]
    left = [(r, 0) for r in range(box_h) if allowed[r, 0]]
    right = [(r, box_w - 1) for r in range(box_h) if allowed[r, box_w - 1]]
    if not top or not bottom or not left or not right:
        return None

    for _ in range(200):
        chosen = {
            random.choice(top),
            random.choice(bottom),
            random.choice(left),
            random.choice(right),
        }
        remaining = [pos for pos in all_allowed if pos not in chosen]
        need = random.randint(max(0, 4 - len(chosen)), min(6 - len(chosen), len(remaining)))
        if need:
            chosen.update(random.sample(remaining, need))
        markers = sorted(chosen)
        rows = [r for r, _ in markers]
        cols = [c for _, c in markers]
        if min(rows) == 0 and max(rows) == box_h - 1 and min(cols) == 0 and max(cols) == box_w - 1 and 4 <= len(markers) <= 6:
            return markers
    return None


def _expected_output(
    inp: np.ndarray,
    template_mask: np.ndarray,
    box_origin: tuple[int, int],
    occupancy: np.ndarray,
    dy: int,
    dx: int,
    box_h: int,
    box_w: int,
) -> np.ndarray:
    out = inp.copy()
    out[out == RED] = BLACK
    box_r0, box_c0 = box_origin
    rows, cols = occupancy.shape
    for j in range(rows):
        for i in range(cols):
            ar = box_r0 + j * dy
            ac = box_c0 + i * dx
            if ar + TEMPLATE_SIZE > box_r0 + box_h or ac + TEMPLATE_SIZE > box_c0 + box_w:
                continue
            color = BLUE if occupancy[j, i] else GREEN
            _paint_mask(out, ar, ac, template_mask, color)
    return out


def _validate_generated(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        info = _analyze(inp)
    except Exception:
        return False

    box = info["box"]
    tr, tc = info["template_origin"]
    template_mask = info["template_mask"]
    anchors = info["anchors"]
    base = info["base"]
    dy = int(info["dy"])
    dx = int(info["dx"])
    lattice = info["lattice"]

    if not np.array_equal(main(inp), expected):
        return False
    if tr > 2 or tc > 2:
        return False

    r0, c0, r1, c1 = box
    out_box = expected[r0 : r1 + 1, c0 : c1 + 1]
    if not np.any(out_box == GREEN) or not np.any(out_box == BLUE):
        return False
    if np.any(expected == RED):
        return False

    row_to_cols: dict[int, list[int]] = defaultdict(list)
    col_to_rows: dict[int, list[int]] = defaultdict(list)
    for r, c in anchors:
        row_to_cols[r].append(c)
        col_to_rows[c].append(r)
    if min(
        min(cols[i + 1] - cols[i] for i in range(len(cols) - 1))
        for cols in [sorted(v) for v in row_to_cols.values() if len(v) >= 2]
    ) != dx:
        return False
    if min(
        min(rows[i + 1] - rows[i] for i in range(len(rows) - 1))
        for rows in [sorted(v) for v in col_to_rows.values() if len(v) >= 2]
    ) != dy:
        return False

    if any(all((c - base[1]) % cand == 0 for _r, c in anchors) for cand in range(4, dx)):
        return False
    if any(all((r - base[0]) % cand == 0 for r, _c in anchors) for cand in range(4, dy)):
        return False

    blue_outside_box = inp.copy()
    blue_outside_box[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    comps = _components_of_color(blue_outside_box, BLUE)
    if len(comps) != 1:
        return False
    only_comp = comps[0]
    cb = _bbox(only_comp)
    if cb[:2] != (tr, tc):
        return False
    comp_mask = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=bool)
    for r, c in only_comp:
        comp_mask[r - tr, c - tc] = True
    if not np.array_equal(comp_mask, template_mask):
        return False

    full_local = _render_local_mask(r1 - r0 + 1, c1 - c0 + 1, template_mask, [(r - r0, c - c0) for r, c in lattice])
    for rr, cc in np.argwhere(inp == RED):
        if full_local[int(rr) - r0, int(cc) - c0]:
            return False
    return True


def _generate_once() -> tuple[np.ndarray, np.ndarray] | None:
    template_mask = random.choice(TEMPLATE_CATALOG).copy()
    rows, cols, dy, dx, box_h, box_w = _layout_choice()
    occupancy = random.choice(OCCUPANCY_OPTIONS[(rows, cols)]).copy()

    local_lattice = [(j * dy, i * dx) for j in range(rows) for i in range(cols)]
    full_mask = _render_local_mask(box_h, box_w, template_mask, local_lattice)
    markers_local = _choose_markers(full_mask)
    if markers_local is None:
        return None

    grid_h = random.randint(max(16, box_h + 6), 22)
    grid_w = random.randint(max(16, box_w + 6), 22)
    template_r0 = random.randint(0, 2)
    template_c0 = random.randint(0, 2)
    box_r0 = random.randint(5, grid_h - box_h - 1)
    box_c0 = random.randint(5, grid_w - box_w - 1)

    inp = np.zeros((grid_h, grid_w), dtype=int)
    _paint_mask(inp, template_r0, template_c0, template_mask, BLUE)

    for j in range(rows):
        for i in range(cols):
            if not occupancy[j, i]:
                continue
            _paint_mask(inp, box_r0 + j * dy, box_c0 + i * dx, template_mask, BLUE)

    for lr, lc in markers_local:
        rr = box_r0 + lr
        cc = box_c0 + lc
        if int(inp[rr, cc]) != BLACK:
            return None
        inp[rr, cc] = RED

    expected = _expected_output(inp, template_mask, (box_r0, box_c0), occupancy, dy, dx, box_h, box_w)
    if not _validate_generated(inp, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        pair = _generate_once()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate a valid input for puzzle 6")
