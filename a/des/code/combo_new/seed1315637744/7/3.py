from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Generator configuration.
GEN_RETRIES = 400
GEN_SLOT_COUNTS = (2, 3, 4)
GEN_LAYOUTS = {
    2: ((1, 2), (2, 1), (2, 2)),
    3: ((1, 3), (3, 1), (2, 2)),
    4: ((2, 2),),
}
GEN_CELL_SIDE_RANGE = (5, 7)
GEN_CELL_GAP_CHOICES = (2, 3)
GEN_MIN_OUTER_EXTRA = 6
GEN_MAX_OUTER_EXTRA = 12
GEN_PIECE_PADDING = 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components from a boolean mask."""
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(H):
        for c in range(W):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(coords)

    return comps


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_key(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in np.argwhere(mask)))


def _find_template_bbox(g: np.ndarray) -> tuple[int, int, int, int]:
    template_mask = (g != BLACK) & (g != MAROON)
    comps = _components(template_mask)
    assert comps, "expected a central template component"
    template = max(comps, key=len)
    r0, c0, r1, c1 = _bbox(template)

    sub = template_mask[r0 : r1 + 1, c0 : c1 + 1]
    assert bool(sub[0, :].all()) and bool(sub[-1, :].all())
    assert bool(sub[:, 0].all()) and bool(sub[:, -1].all())
    return int(r0), int(c0), int(r1), int(c1)


def _extract_slots(g: np.ndarray, template_bbox: tuple[int, int, int, int]) -> list[dict[str, object]]:
    r0, c0, r1, c1 = template_bbox
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    H, W = sub.shape

    slots: list[dict[str, object]] = []
    for comp in _components(sub == BLACK):
        if any(rr in (0, H - 1) or cc in (0, W - 1) for rr, cc in comp):
            continue

        boundary: set[tuple[int, int]] = set()
        for rr, cc in comp:
            for dr, dc in DIR4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < H and 0 <= nc < W and int(sub[nr, nc]) != BLACK:
                    boundary.add((int(nr), int(nc)))

        marker_cells = [(rr, cc) for rr, cc in boundary if int(sub[rr, cc]) not in (BLACK, GREY)]
        assert len(marker_cells) == 1, "each slot must have exactly one colored border marker"
        mr, mc = marker_cells[0]

        global_coords = tuple((int(r0 + rr), int(c0 + cc)) for rr, cc in comp)
        slots.append(
            {
                "shape_key": _shape_key(comp),
                "coords": global_coords,
                "marker_color": int(sub[mr, mc]),
            }
        )

    assert slots, "expected at least one enclosed slot"
    return slots


def _extract_pieces(g: np.ndarray) -> dict[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    pieces: dict[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]] = {}
    for comp in _components(g == MAROON):
        key = _shape_key(comp)
        assert key not in pieces, "expected unique fragment shapes"
        pieces[key] = tuple(comp)
    assert pieces, "expected maroon fragments outside the template"
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit each maroon fragment into the matching template hole and recolor it from the hole's border marker.

    Input structure:
    - Black background with a central grey template plate whose outer contour is a rectangle.
    - Inside that grey plate are several enclosed black holes of different shapes; each hole has exactly one non-grey, non-black marker pixel on its grey boundary.
    - Outside the template plate are several disconnected maroon fragments, one per hole, with shapes matching the holes by translation only.

    Rule:
    - Find the central template and extract every enclosed black hole as a slot shape.
    - Match each maroon fragment to the unique slot with the same normalized shape.
    - Remove the maroon fragments from their original positions and fill the matched slot with the slot's marker color.
    - Keep the grey template and the colored marker pixels unchanged.

    Key insight:
    - The task combines three visible cues: hole shape identifies the destination, the single border marker identifies the repaint color, and the original maroon fragment color is only a temporary placeholder.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    template_bbox = _find_template_bbox(g)
    slots = _extract_slots(g, template_bbox)
    pieces = _extract_pieces(g)

    slot_keys = {slot["shape_key"] for slot in slots}
    assert slot_keys == set(pieces), "slot shapes and fragment shapes must match one-to-one"

    out = g.copy()
    out[out == MAROON] = BLACK
    for slot in slots:
        color = int(slot["marker_color"])
        for r, c in slot["coords"]:
            out[int(r), int(c)] = color
    return out


def _random_shape(max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(120):
        raw_h = random.randint(2, max_h)
        raw_w = random.randint(2, max_w)
        max_area = min(8, raw_h * raw_w)
        if max_area < 3:
            continue
        target_area = random.randint(3, max_area)

        cells = {(random.randrange(raw_h), random.randrange(raw_w))}
        for _grow in range(target_area * 20):
            if len(cells) >= target_area:
                break
            r, c = random.choice(tuple(cells))
            dr, dc = random.choice(DIR4)
            nr = r + dr
            nc = c + dc
            if 0 <= nr < raw_h and 0 <= nc < raw_w:
                cells.add((int(nr), int(nc)))
        if len(cells) != target_area:
            continue

        key = _shape_key(list(cells))
        h = max(r for r, _c in key) + 1
        w = max(c for _r, c in key) + 1
        if h < 2 or w < 2:
            continue
        if len(key) == h * w and random.random() < 0.7:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in key:
            mask[r, c] = True
        return mask
    return None


def _build_shapes(count: int, max_h: int, max_w: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    used: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(count):
        mask = None
        for _attempt in range(120):
            candidate = _random_shape(max_h, max_w)
            if candidate is None:
                continue
            key = _mask_key(candidate)
            if key in used:
                continue
            used.add(key)
            mask = candidate
            break
        if mask is None:
            return None
        shapes.append(mask)
    return shapes


def _marker_candidates(template: np.ndarray, hole_coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    H, W = template.shape
    hole_set = set(hole_coords)
    candidates: set[tuple[int, int]] = set()
    for rr, cc in hole_coords:
        for dr, dc in DIR4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in hole_set and int(template[nr, nc]) == GREY:
                candidates.add((int(nr), int(nc)))
    return sorted(candidates)


def _build_template() -> tuple[np.ndarray, list[dict[str, object]]] | None:
    slot_count = random.choice(GEN_SLOT_COUNTS)
    rows, cols = random.choice(GEN_LAYOUTS[slot_count])
    cell_h = random.randint(*GEN_CELL_SIDE_RANGE)
    cell_w = random.randint(*GEN_CELL_SIDE_RANGE)
    gap = random.choice(GEN_CELL_GAP_CHOICES)

    template_h = 2 + rows * cell_h + (rows - 1) * gap
    template_w = 2 + cols * cell_w + (cols - 1) * gap
    if template_h > 22 or template_w > 22:
        return None

    shapes = _build_shapes(slot_count, cell_h - 2, cell_w - 2)
    if shapes is None:
        return None

    template = np.full((template_h, template_w), GREY, dtype=int)
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    chosen_cells = random.sample(cells, slot_count)
    marker_colors = random.sample(MARKER_COLORS, slot_count)

    slots: list[dict[str, object]] = []
    for shape, (cell_r, cell_c), marker_color in zip(shapes, chosen_cells, marker_colors):
        sh, sw = shape.shape
        base_r = 1 + cell_r * (cell_h + gap)
        base_c = 1 + cell_c * (cell_w + gap)
        off_r = random.randint(1, cell_h - sh - 1)
        off_c = random.randint(1, cell_w - sw - 1)
        r0 = base_r + off_r
        c0 = base_c + off_c

        hole_coords: list[tuple[int, int]] = []
        for dr, dc in np.argwhere(shape):
            rr = int(r0 + dr)
            cc = int(c0 + dc)
            template[rr, cc] = BLACK
            hole_coords.append((rr, cc))

        candidates = _marker_candidates(template, hole_coords)
        if not candidates:
            return None
        mr, mc = random.choice(candidates)
        template[mr, mc] = int(marker_color)

        slots.append(
            {
                "shape_mask": shape.copy(),
                "shape_key": _mask_key(shape),
                "coords_local": tuple(hole_coords),
                "marker_color": int(marker_color),
            }
        )

    return template, slots


def _balanced_split(total_extra: int) -> tuple[int, int]:
    assert total_extra >= 4
    center = total_extra // 2
    lo = max(2, center - 2)
    hi = min(total_extra - 2, center + 2)
    if lo > hi:
        lo, hi = 2, total_extra - 2
    first = random.randint(lo, hi)
    return int(first), int(total_extra - first)


def _place_template_on_canvas(template: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    th, tw = template.shape
    max_extra_h = 30 - th
    max_extra_w = 30 - tw
    if max_extra_h < GEN_MIN_OUTER_EXTRA or max_extra_w < GEN_MIN_OUTER_EXTRA:
        return None

    extra_h = random.randint(GEN_MIN_OUTER_EXTRA, min(GEN_MAX_OUTER_EXTRA, max_extra_h))
    extra_w = random.randint(GEN_MIN_OUTER_EXTRA, min(GEN_MAX_OUTER_EXTRA, max_extra_w))
    top, bottom = _balanced_split(extra_h)
    left, right = _balanced_split(extra_w)

    H = th + top + bottom
    W = tw + left + right
    grid = np.full((H, W), BLACK, dtype=int)
    grid[top : top + th, left : left + tw] = template
    return grid, (int(top), int(left), int(top + th - 1), int(left + tw - 1))


def _can_place_piece(
    grid: np.ndarray,
    template_bbox: tuple[int, int, int, int],
    shape: np.ndarray,
    r0: int,
    c0: int,
) -> bool:
    H, W = grid.shape
    sh, sw = shape.shape
    tr0, tc0, tr1, tc1 = template_bbox
    if r0 < 0 or c0 < 0 or r0 + sh > H or c0 + sw > W:
        return False

    for dr, dc in np.argwhere(shape):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        if tr0 <= rr <= tr1 and tc0 <= cc <= tc1:
            return False
        if int(grid[rr, cc]) != BLACK:
            return False
        r1 = max(0, rr - GEN_PIECE_PADDING)
        r2 = min(H, rr + GEN_PIECE_PADDING + 1)
        c1 = max(0, cc - GEN_PIECE_PADDING)
        c2 = min(W, cc + GEN_PIECE_PADDING + 1)
        if bool(np.any(grid[r1:r2, c1:c2] != BLACK)):
            return False
    return True


def _paint_shape(grid: np.ndarray, shape: np.ndarray, r0: int, c0: int, color: int) -> None:
    for dr, dc in np.argwhere(shape):
        grid[int(r0 + dr), int(c0 + dc)] = int(color)


def _place_pieces(grid: np.ndarray, template_bbox: tuple[int, int, int, int], slots: list[dict[str, object]]) -> bool:
    order = slots[:]
    random.shuffle(order)
    order.sort(key=lambda slot: -int(np.sum(slot["shape_mask"])))

    for slot in order:
        shape = np.asarray(slot["shape_mask"], dtype=bool)
        sh, sw = shape.shape
        positions = [(r, c) for r in range(grid.shape[0] - sh + 1) for c in range(grid.shape[1] - sw + 1)]
        random.shuffle(positions)

        placed = False
        for r0, c0 in positions:
            if _can_place_piece(grid, template_bbox, shape, int(r0), int(c0)):
                _paint_shape(grid, shape, int(r0), int(c0), MAROON)
                placed = True
                break
        if not placed:
            return False
    return True


def _expected_output(grid: np.ndarray, template_bbox: tuple[int, int, int, int], slots: list[dict[str, object]]) -> np.ndarray:
    top, left, _bottom, _right = template_bbox
    out = grid.copy()
    out[out == MAROON] = BLACK
    for slot in slots:
        color = int(slot["marker_color"])
        for rr, cc in slot["coords_local"]:
            out[int(top + rr), int(left + cc)] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    built = _build_template()
    if built is None:
        return None
    template, slots = built

    placed = _place_template_on_canvas(template)
    if placed is None:
        return None
    grid, template_bbox = placed

    if not _place_pieces(grid, template_bbox, slots):
        return None

    expected = _expected_output(grid, template_bbox, slots)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    return grid if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC puzzle 3")
