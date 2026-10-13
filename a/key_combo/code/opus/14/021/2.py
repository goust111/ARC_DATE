from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
TRIANGLE = 5
PALETTE = (1, 2, 3, 4)
QUADRANT_COLORS = (2, 3, 4, 1)  # TL, TR, BL, BR observed palette, permuted per sample.

QUADRANTS = (0, 1, 2, 3)  # TL, TR, BL, BR
POSITION_ORDER = (0, 3, 2, 1)  # 75b8110e back-to-front shortcut to actively invalidate.
ROW_MAJOR_ORDER = (0, 1, 2, 3)

PRIORITY_TO_GAP = {
    1: (0, 0),  # missing top-left
    2: (0, 1),  # missing top-right
    3: (1, 0),  # missing bottom-left
    4: (1, 1),  # missing bottom-right
}
GAP_TO_PRIORITY = {gap: priority for priority, gap in PRIORITY_TO_GAP.items()}

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _corner_origin(h: int, w: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - 2
    if corner == "bl":
        return h - 2, 0
    if corner == "br":
        return h - 2, w - 2
    raise ValueError(f"unknown corner {corner}")


def _triangle_cells(h: int, w: int, corner: str, priority: int) -> set[tuple[int, int]]:
    r0, c0 = _corner_origin(h, w, corner)
    missing = PRIORITY_TO_GAP[priority]
    cells = {
        (r0, c0),
        (r0, c0 + 1),
        (r0 + 1, c0),
        (r0 + 1, c0 + 1),
    }
    cells.remove((r0 + missing[0], c0 + missing[1]))
    return cells


def _decode_triangle(region: np.ndarray) -> int:
    gray = np.argwhere(region == TRIANGLE)
    assert gray.shape[0] == 3, "expected exactly one 3-pixel gray triangle per quadrant"
    r0, c0 = gray.min(axis=0)
    r1, c1 = gray.max(axis=0)
    h, w = region.shape
    assert int(r1) == int(r0) + 1 and int(c1) == int(c0) + 1, "triangle must span one 2x2 box"
    assert (int(r0), int(c0)) in {
        (0, 0),
        (0, w - 2),
        (h - 2, 0),
        (h - 2, w - 2),
    }, "triangle box must touch a quadrant corner"
    full = {
        (int(r0), int(c0)),
        (int(r0), int(c0) + 1),
        (int(r0) + 1, int(c0)),
        (int(r0) + 1, int(c0) + 1),
    }
    missing = list(full - {tuple(map(int, rc)) for rc in gray})
    assert len(missing) == 1
    mr, mc = missing[0]
    return GAP_TO_PRIORITY[(mr - int(r0), mc - int(c0))]


def _quadrant_slices(h: int, w: int, q: int) -> tuple[slice, slice]:
    if q == 0:
        return slice(0, h), slice(0, w)
    if q == 1:
        return slice(0, h), slice(w, 2 * w)
    if q == 2:
        return slice(h, 2 * h), slice(0, w)
    if q == 3:
        return slice(h, 2 * h), slice(w, 2 * w)
    raise ValueError(f"unknown quadrant {q}")


def _overlay(masks: dict[int, np.ndarray], colors: dict[int, int], order: tuple[int, ...], bg: int) -> np.ndarray:
    h, w = next(iter(masks.values())).shape
    out = np.full((h, w), bg, dtype=int)
    for q in order:
        mask = masks[q]
        out[mask] = int(colors[q])
    return out


def _priority_order(priorities: dict[int, int]) -> tuple[int, ...]:
    return tuple(sorted(QUADRANTS, key=lambda q: priorities[q]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the four quadrant masks onto one canvas and overlay them in the layer order encoded by each corner triangle's missing direction.

    Input structure:
    - A black background grid is split into four equal quadrants of the same size.
    - Each quadrant contains one colored foreground mask plus one gray 3-pixel L triangle placed in a quadrant corner.
    - The gray triangle occupies three cells of a 2x2 corner box; its missing corner encodes one of four diagonal directions.

    Rule:
    - Read each quadrant independently and locate its gray 3-pixel corner triangle.
    - Convert the missing-corner direction into a layer priority: top-left gap -> 1, top-right gap -> 2, bottom-left gap -> 3, bottom-right gap -> 4.
    - Remove only the gray triangle pixels, keep the remaining colored mask as that quadrant's layer, and fold all four layers onto a single quadrant-sized canvas.
    - Draw the layers from lower priority to higher priority, so later layers overwrite earlier overlaps.

    Key insight:
    - The quadrant position no longer fixes the overlay stack; the stack must be decoded from the local orientation of the gray L triangle in each quadrant.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    hh, ww = map(int, g.shape)
    assert hh % 2 == 0 and ww % 2 == 0, "input must split into four equal quadrants"
    h, w = hh // 2, ww // 2
    bg = _get_bg(g)

    masks: dict[int, np.ndarray] = {}
    colors: dict[int, int] = {}
    priorities: dict[int, int] = {}
    for q in QUADRANTS:
        rs, cs = _quadrant_slices(h, w, q)
        region = g[rs, cs]
        priorities[q] = _decode_triangle(region)
        mask_region = (region != bg) & (region != TRIANGLE)
        present = sorted(set(map(int, np.unique(region[mask_region]))))
        assert len(present) == 1, "each quadrant must contain exactly one non-gray mask color"
        colors[q] = present[0]
        masks[q] = mask_region

    order = _priority_order(priorities)
    return _overlay(masks, colors, order, bg)


def _neighbors4(cell: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in NEIGHBORS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _zone_cells(h: int, w: int, q: int) -> list[tuple[int, int]]:
    r_mid = max(1, h // 2)
    c_mid = max(1, w // 2)
    if q == 0:
        rows = range(0, r_mid)
        cols = range(0, c_mid)
    elif q == 1:
        rows = range(0, r_mid)
        cols = range(c_mid, w)
    elif q == 2:
        rows = range(r_mid, h)
        cols = range(0, c_mid)
    else:
        rows = range(r_mid, h)
        cols = range(c_mid, w)
    return [(r, c) for r in rows for c in cols]


def _pick_anchor(
    rng: random.Random,
    h: int,
    w: int,
    q: int,
    available: set[tuple[int, int]],
    used: set[tuple[int, int]],
) -> tuple[int, int] | None:
    preferred = [cell for cell in _zone_cells(h, w, q) if cell in available and cell not in used]
    fallback = [cell for cell in available if cell not in used]
    pool = preferred or fallback
    if not pool:
        return None
    return rng.choice(pool)


def _grow_mask(
    rng: random.Random,
    h: int,
    w: int,
    seed: set[tuple[int, int]],
    allowed: set[tuple[int, int]],
    target_area: int,
) -> set[tuple[int, int]] | None:
    cells = set(seed)
    if not cells <= allowed:
        return None
    while len(cells) < target_area:
        frontier = sorted(
            {
                nbr
                for cell in cells
                for nbr in _neighbors4(cell, h, w)
                if nbr in allowed and nbr not in cells
            }
        )
        if frontier and (rng.random() < 0.88 or len(cells) == len(seed)):
            cells.add(rng.choice(frontier))
            continue
        remaining = sorted(allowed - cells)
        if not remaining:
            return None
        cells.add(rng.choice(remaining))
    return cells


def _mask_dict_to_arrays(
    h: int,
    w: int,
    cell_sets: dict[int, set[tuple[int, int]]],
) -> dict[int, np.ndarray]:
    masks: dict[int, np.ndarray] = {}
    for q, cells in cell_sets.items():
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        masks[q] = mask
    return masks


def _render_input(
    h: int,
    w: int,
    colors: dict[int, int],
    masks: dict[int, np.ndarray],
    triangle_cells: dict[int, set[tuple[int, int]]],
) -> np.ndarray:
    grid = np.full((2 * h, 2 * w), BG, dtype=int)
    for q in QUADRANTS:
        rs, cs = _quadrant_slices(h, w, q)
        region = grid[rs, cs]
        region[masks[q]] = int(colors[q])
        for r, c in triangle_cells[q]:
            region[r, c] = TRIANGLE
    return grid


def _choose_conflict_memberships(
    rng: random.Random,
    correct_order: tuple[int, ...],
) -> list[set[int]]:
    pos_index = {q: i for i, q in enumerate(POSITION_ORDER)}
    inversion_pairs = [
        {a, b}
        for i, a in enumerate(correct_order)
        for b in correct_order[i + 1 :]
        if pos_index[a] > pos_index[b]
    ]
    if not inversion_pairs:
        return []
    rng.shuffle(inversion_pairs)
    pair = inversion_pairs[0]
    memberships = [pair, set(correct_order[:3]), set(correct_order[1:]), set(correct_order)]
    return memberships


def _validate_sample(
    inp: np.ndarray,
    masks: dict[int, np.ndarray],
    colors: dict[int, int],
    priorities: dict[int, int],
    memberships: list[set[int]],
) -> bool:
    correct_order = _priority_order(priorities)
    out = _overlay(masks, colors, correct_order, BG)

    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, out):
        return False

    if np.array_equal(out, _overlay(masks, colors, POSITION_ORDER, BG)):
        return False
    if np.array_equal(out, _overlay(masks, colors, ROW_MAJOR_ORDER, BG)):
        return False

    union = np.logical_or.reduce([masks[q] for q in QUADRANTS])
    overlap_count = int(
        np.sum(sum(masks[q].astype(int) for q in QUADRANTS) >= 2)
    )
    area = int(union.size)
    union_count = int(union.sum())
    if not (max(8, area // 4) <= union_count <= int(area * 0.7)):
        return False
    if not (4 <= overlap_count <= max(6, area // 2)):
        return False

    visible_colors = {int(v) for v in np.unique(out) if int(v) != BG}
    if visible_colors != set(colors.values()):
        return False

    for q in QUADRANTS:
        if int(np.count_nonzero(out == colors[q])) < 1:
            return False
        if int(np.count_nonzero(masks[q])) < 4:
            return False

    if len({priorities[q] for q in QUADRANTS}) != 4:
        return False

    conflict_participation = {q: 0 for q in QUADRANTS}
    for members in memberships:
        if len(members) < 2:
            continue
        for q in members:
            conflict_participation[q] += 1
    if min(conflict_participation.values()) < 1:
        return False

    return True


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(5, 8)
    w = rng.randint(5, 8)

    colors = {q: color for q, color in zip(QUADRANTS, rng.sample(list(QUADRANT_COLORS), 4))}

    correct_order = list(POSITION_ORDER)
    while tuple(correct_order) == POSITION_ORDER or tuple(correct_order) == ROW_MAJOR_ORDER:
        correct_order = rng.sample(list(QUADRANTS), 4)
    priorities = {q: i + 1 for i, q in enumerate(correct_order)}

    corners = {q: rng.choice(("tl", "tr", "bl", "br")) for q in QUADRANTS}
    triangle_cells = {q: _triangle_cells(h, w, corners[q], priorities[q]) for q in QUADRANTS}

    available: dict[int, set[tuple[int, int]]] = {}
    for q in QUADRANTS:
        blocked = {
            (r, c)
            for r0, c0 in [_corner_origin(h, w, corners[q])]
            for r in range(r0, r0 + 2)
            for c in range(c0, c0 + 2)
        }
        available[q] = {
            (r, c)
            for r in range(h)
            for c in range(w)
            if (r, c) not in blocked
        }

    used_special: set[tuple[int, int]] = set()
    designated: dict[tuple[int, int], set[int]] = {}
    for q in QUADRANTS:
        anchor = _pick_anchor(rng, h, w, q, available[q], used_special)
        if anchor is None:
            return None
        designated[anchor] = {q}
        used_special.add(anchor)

    memberships = _choose_conflict_memberships(rng, tuple(correct_order))
    if len(memberships) < 4:
        return None

    shared = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if all((r, c) in available[q] for q in QUADRANTS) and (r, c) not in used_special
    ]
    if len(shared) < len(memberships):
        shared = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if all((r, c) in available[q] for q in QUADRANTS) and (r, c) not in used_special
        ]
    if len(shared) < len(memberships):
        return None
    rng.shuffle(shared)
    for cell, members in zip(shared, memberships):
        designated[cell] = set(members)
        used_special.add(cell)

    cell_sets: dict[int, set[tuple[int, int]]] = {q: set() for q in QUADRANTS}
    for cell, members in designated.items():
        for q in members:
            cell_sets[q].add(cell)

    disallowed_special = {
        q: {cell for cell, members in designated.items() if q not in members}
        for q in QUADRANTS
    }
    targets = {q: rng.randint(max(5, len(cell_sets[q]) + 1), min(10, len(available[q]) - 1)) for q in QUADRANTS}
    for q in QUADRANTS:
        allowed = available[q] - disallowed_special[q]
        grown = _grow_mask(rng, h, w, cell_sets[q], allowed, targets[q])
        if grown is None:
            return None
        cell_sets[q] = grown

    masks = _mask_dict_to_arrays(h, w, cell_sets)
    inp = _render_input(h, w, colors, masks, triangle_cells)
    if not _validate_sample(inp, masks, colors, priorities, memberships):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid triangle-overlay sample")
