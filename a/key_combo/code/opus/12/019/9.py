from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
GREY = 5

DIRS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)

GEN_ATTEMPTS = 500
GEN_GRID_MIN = 17
GEN_GRID_MAX = 23


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _is_connected(mask: np.ndarray, connectivity: int = 4) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(_components_mask(mask, connectivity=connectivity)) == 1


def _center_offsets(size: int) -> tuple[int, int]:
    before = (size - 1) // 2
    after = size - 1 - before
    return before, after


def _origin_from_center(center: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    br, _ar = _center_offsets(shape[0])
    bc, _ac = _center_offsets(shape[1])
    return int(center[0] - br), int(center[1] - bc)


def _center_from_origin(origin: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    br, _ar = _center_offsets(shape[0])
    bc, _ac = _center_offsets(shape[1])
    return int(origin[0] + br), int(origin[1] + bc)


def _bbox_from_origin(origin: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0 = origin
    h, w = shape
    return int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)


def _cells_in_bbox(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _line_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    dr = r1 - r0
    dc = c1 - c0
    assert (dr, dc) != (0, 0), "line endpoints must differ"
    assert dr == 0 or dc == 0 or abs(dr) == abs(dc), "expected horizontal, vertical, or 45-degree line"
    sr = 0 if dr == 0 else dr // abs(dr)
    sc = 0 if dc == 0 else dc // abs(dc)
    steps = max(abs(dr), abs(dc))
    return [(int(r0 + i * sr), int(c0 + i * sc)) for i in range(steps + 1)]


def _offsets(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def _find_sprite_template(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]]:
    comps = _components_mask(grid != BG, connectivity=4)
    sprite_hits = [coords for coords in comps if np.any(grid[coords[:, 0], coords[:, 1]] == RED)]
    assert len(sprite_hits) == 1, "expected one non-black component containing red"
    coords = sprite_hits[0]
    r0, c0, r1, c1 = _bbox_coords(coords)
    window = grid[r0 : r1 + 1, c0 : c1 + 1]
    red_offsets = _offsets(window == RED)
    support_offsets = _offsets(window == GREY)
    assert 3 <= len(red_offsets) <= 5, "expected a 3-5 pixel red stamp"
    assert support_offsets, "expected a non-empty support signature"
    return (r0, c0), window.shape, red_offsets, support_offsets


def _scan_matches(
    grid: np.ndarray,
    shape: tuple[int, int],
    red_offsets: list[tuple[int, int]],
    support_offsets: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = grid.shape
    bh, bw = shape
    out = grid.copy()
    matches: list[tuple[int, int]] = []
    for r0 in range(h - bh + 1):
        for c0 in range(w - bw + 1):
            if not all(out[r0 + dr, c0 + dc] == BG for dr, dc in red_offsets):
                continue
            if not all(grid[r0 + dr, c0 + dc] == GREY for dr, dc in support_offsets):
                continue
            matches.append((int(r0), int(c0)))
            for dr, dc in red_offsets:
                out[r0 + dr, c0 + dc] = RED
    return matches


def _render_with_metadata(
    grid: np.ndarray,
    shape: tuple[int, int],
    red_offsets: list[tuple[int, int]],
    original_origin: tuple[int, int],
    slot_origins: list[tuple[int, int]],
) -> np.ndarray:
    base = grid.copy()
    for r0, c0 in slot_origins:
        for dr, dc in red_offsets:
            base[r0 + dr, c0 + dc] = RED

    origin_center = _center_from_origin(original_origin, shape)
    order = sorted(slot_origins, key=lambda rc: (abs(_center_from_origin(rc, shape)[0] - origin_center[0]) + abs(_center_from_origin(rc, shape)[1] - origin_center[1]), rc[0], rc[1]))

    out = base.copy()
    for slot_origin in order:
        slot_center = _center_from_origin(slot_origin, shape)
        for r, c in _line_cells(origin_center, slot_center):
            out[r, c] = GREEN if base[r, c] == GREY else RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the red sprite into every support-matching gray slot, then connect the original sprite center to each new stamp center.

    Input structure:
    - The grid uses black background, gray support pixels, and one original multi-color sprite whose red cells define a stamp.
    - The original sprite is the unique non-black 4-connected component containing red, and its bbox may also contain gray support cells.
    - Elsewhere there are several gray empty slots with the same support signature, plus optional gray distractor clusters that do not match.

    Rule:
    - Extract the original sprite bbox, record its red-cell offsets as the stamp mask, and record its gray-cell offsets as the support signature.
    - Scan every bbox-sized window from top-left to bottom-right; any window whose stamp offsets are all black and whose support offsets are all gray is a valid empty slot.
    - Paint the stamp mask red in every valid slot, then connect the original bbox center to each new slot center with an inclusive horizontal, vertical, or 45-degree segment.
    - Line cells that are gray in the post-stamping base become green; every other line cell becomes red; lines are drawn from shorter Manhattan distance to longer, with later lines overwriting earlier ones.

    Key insight:
    - The gray support pattern serves two roles at once: it identifies where new copies of the stamp can appear, and the same gray cells later flip line pixels from red to green when the new copies are connected back to the original.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert set(map(int, np.unique(g))).issubset({BG, RED, GREY}), "expected only black, red, and gray"

    original_origin, shape, red_offsets, support_offsets = _find_sprite_template(g)
    matches = _scan_matches(g, shape, red_offsets, support_offsets)
    slot_origins = [rc for rc in matches if rc != original_origin]
    assert slot_origins, "expected at least one empty slot"

    return _render_with_metadata(g, shape, red_offsets, original_origin, slot_origins)


def _grow_red_mask(h: int, w: int, n: int) -> np.ndarray | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(80):
        seed = random.choice(cells)
        chosen = {seed}
        frontier = set()
        while len(chosen) < n:
            frontier.clear()
            for r, c in chosen:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in chosen:
                        frontier.add((nr, nc))
            if not frontier:
                break
            chosen.add(random.choice(tuple(frontier)))
        if len(chosen) != n:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        rows = {r for r, _c in chosen}
        cols = {c for _r, c in chosen}
        if len(rows) == 1 or len(cols) == 1:
            continue
        rr, cc = np.where(mask)
        if (rr.max() - rr.min() + 1) * (cc.max() - cc.min() + 1) == n:
            continue
        return mask
    return None


def _make_template() -> tuple[tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]] | None:
    dims = [(3, 3), (3, 4), (4, 3), (4, 4)]
    for _ in range(120):
        h, w = random.choice(dims)
        red_count = random.randint(3, 5)
        red_mask = _grow_red_mask(h, w, red_count)
        if red_mask is None:
            continue

        center = (_center_offsets(h)[0], _center_offsets(w)[0])
        free = [(r, c) for r in range(h) for c in range(w) if not red_mask[r, c]]
        if len(free) < 2:
            continue

        min_support = 2
        max_support = min(3, len(free))
        support_count = random.randint(min_support, max_support)
        for _ in range(100):
            chosen: set[tuple[int, int]] = set()
            if center in free and random.random() < 0.6:
                chosen.add(center)
            while len(chosen) < support_count:
                chosen.add(random.choice(free))

            union = red_mask.copy()
            for r, c in chosen:
                union[r, c] = True
            if not _is_connected(union, connectivity=4):
                continue
            if _bbox_coords(np.argwhere(union)) != (0, 0, h - 1, w - 1):
                continue
            return (h, w), _offsets(red_mask), sorted(chosen)
    return None


def _max_distance(center: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int], grid_shape: tuple[int, int]) -> int:
    cr, cc = center
    dr, dc = direction
    h, w = shape
    before_r, after_r = _center_offsets(h)
    before_c, after_c = _center_offsets(w)
    limits = []
    if dr < 0:
        limits.append(cr - before_r)
    elif dr > 0:
        limits.append(grid_shape[0] - 1 - after_r - cr)
    if dc < 0:
        limits.append(cc - before_c)
    elif dc > 0:
        limits.append(grid_shape[1] - 1 - after_c - cc)
    return int(min(limits)) if limits else 0


def _choose_layout(shape: tuple[int, int], slot_count: int) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = shape
    before_r, after_r = _center_offsets(h)
    before_c, after_c = _center_offsets(w)
    for _ in range(180):
        grid_h = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
        grid_w = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
        if grid_h < h + 8 or grid_w < w + 8:
            continue

        cr = random.randint(before_r + 4, grid_h - 1 - after_r - 4)
        cc = random.randint(before_c + 4, grid_w - 1 - after_c - 4)
        origin_center = (cr, cc)
        original_origin = _origin_from_center(origin_center, shape)
        original_box = _bbox_from_origin(original_origin, shape)

        directions = random.sample(DIRS, slot_count)
        slot_origins: list[tuple[int, int]] = []
        slot_boxes: list[tuple[int, int, int, int]] = []
        used_manhattan: set[int] = set()
        occupied_lines: set[tuple[int, int]] = set()
        existing_slot_cells: set[tuple[int, int]] = set()

        ok = True
        for direction in directions:
            max_d = _max_distance(origin_center, direction, shape, (grid_h, grid_w))
            choices = list(range(4, max_d + 1))
            random.shuffle(choices)
            picked = None
            for dist in choices:
                manhattan = dist * (abs(direction[0]) + abs(direction[1]))
                if manhattan in used_manhattan:
                    continue
                target_center = (origin_center[0] + direction[0] * dist, origin_center[1] + direction[1] * dist)
                slot_origin = _origin_from_center(target_center, shape)
                slot_box = _bbox_from_origin(slot_origin, shape)
                if not _boxes_separated(slot_box, original_box, pad=1):
                    continue
                if any(not _boxes_separated(slot_box, box, pad=1) for box in slot_boxes):
                    continue
                slot_cells = _cells_in_bbox(slot_box)
                if slot_cells & occupied_lines:
                    continue
                line = _line_cells(origin_center, target_center)
                inner = set(line[1:-1])
                if inner & occupied_lines:
                    continue
                if inner & existing_slot_cells:
                    continue
                picked = (slot_origin, slot_box, inner, manhattan)
                break
            if picked is None:
                ok = False
                break

            slot_origin, slot_box, inner, manhattan = picked
            slot_origins.append(slot_origin)
            slot_boxes.append(slot_box)
            used_manhattan.add(manhattan)
            occupied_lines |= inner
            existing_slot_cells |= _cells_in_bbox(slot_box)

        if ok:
            return (grid_h, grid_w), original_origin, slot_origins, directions
    return None


def _place_template_cells(grid: np.ndarray, origin: tuple[int, int], red_offsets: list[tuple[int, int]], support_offsets: list[tuple[int, int]], with_red: bool) -> None:
    r0, c0 = origin
    for dr, dc in support_offsets:
        grid[r0 + dr, c0 + dc] = GREY
    if with_red:
        for dr, dc in red_offsets:
            grid[r0 + dr, c0 + dc] = RED


def _match_set(
    grid: np.ndarray,
    shape: tuple[int, int],
    red_offsets: list[tuple[int, int]],
    support_offsets: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    return set(_scan_matches(grid, shape, red_offsets, support_offsets))


def _add_distractors(
    grid: np.ndarray,
    shape: tuple[int, int],
    red_offsets: list[tuple[int, int]],
    support_offsets: list[tuple[int, int]],
    protected: set[tuple[int, int]],
    target_matches: set[tuple[int, int]],
) -> int:
    added = 0
    desired = random.choices([0, 1, 2], weights=[2, 2, 1], k=1)[0]
    h, w = grid.shape
    for _ in range(desired):
        for _ in range(60):
            size = random.randint(1, 3)
            candidates = [(r, c) for r in range(h) for c in range(w) if grid[r, c] == BG and (r, c) not in protected]
            if not candidates:
                break
            seed = random.choice(candidates)
            cluster = {seed}
            frontier = {seed}
            while len(cluster) < size and frontier:
                rr, cc = random.choice(tuple(frontier))
                nbrs = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BG and (nr, nc) not in protected and (nr, nc) not in cluster:
                        nbrs.append((nr, nc))
                if not nbrs:
                    frontier.remove((rr, cc))
                    continue
                nxt = random.choice(nbrs)
                cluster.add(nxt)
                frontier.add(nxt)
            if len(cluster) != size:
                continue

            changed = []
            for r, c in cluster:
                grid[r, c] = GREY
                changed.append((r, c))
            if _match_set(grid, shape, red_offsets, support_offsets) == target_matches:
                protected.update(cluster)
                added += 1
                break
            for r, c in changed:
                grid[r, c] = BG
    return added


def _manual_expected(
    inp: np.ndarray,
    shape: tuple[int, int],
    red_offsets: list[tuple[int, int]],
    original_origin: tuple[int, int],
    slot_origins: list[tuple[int, int]],
) -> np.ndarray:
    return _render_with_metadata(inp, shape, red_offsets, original_origin, slot_origins)


def _attempt_generate() -> np.ndarray | None:
    template = _make_template()
    if template is None:
        return None
    shape, red_offsets, support_offsets = template

    slot_count = random.randint(2, 4)
    layout = _choose_layout(shape, slot_count)
    if layout is None:
        return None
    grid_shape, original_origin, slot_origins, _directions = layout

    inp = np.zeros(grid_shape, dtype=int)
    _place_template_cells(inp, original_origin, red_offsets, support_offsets, with_red=True)
    for slot_origin in slot_origins:
        _place_template_cells(inp, slot_origin, red_offsets, support_offsets, with_red=False)

    original_center = _center_from_origin(original_origin, shape)
    protected = set()
    protected |= _cells_in_bbox(_bbox_from_origin(original_origin, shape))
    for slot_origin in slot_origins:
        protected |= _cells_in_bbox(_bbox_from_origin(slot_origin, shape))
        protected |= set(_line_cells(original_center, _center_from_origin(slot_origin, shape)))

    target_matches = set(slot_origins)
    actual_matches = _match_set(inp, shape, red_offsets, support_offsets)
    if actual_matches != target_matches:
        return None

    _add_distractors(inp, shape, red_offsets, support_offsets, protected, target_matches)
    if _match_set(inp, shape, red_offsets, support_offsets) != target_matches:
        return None

    expected = _manual_expected(inp, shape, red_offsets, original_origin, slot_origins)
    if np.count_nonzero(expected == GREEN) == 0:
        return None
    line_mask = expected != _manual_expected(np.where(inp == RED, BG, inp), shape, red_offsets, original_origin, slot_origins)
    if np.count_nonzero(expected == RED) <= np.count_nonzero(inp == RED):
        return None
    if np.count_nonzero(line_mask & (expected == RED)) == 0:
        return None

    nonblack_comps = _components_mask(inp != BG, connectivity=4)
    if sum(np.any(inp[coords[:, 0], coords[:, 1]] == RED) for coords in nonblack_comps) != 1:
        return None

    try:
        solver_out = main(inp)
    except Exception:
        return None
    if not np.array_equal(solver_out, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
