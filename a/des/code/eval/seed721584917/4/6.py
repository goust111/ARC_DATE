import random
from collections import Counter

import numpy as np


BG = 0
TEMPLATE = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400
SIDES = ("top", "bottom", "left", "right")
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _normalize(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _rotate_norm(shape):
    h = 1 + max(r for r, _ in shape)
    rotated = [(c, h - 1 - r) for r, c in shape]
    return _normalize(rotated)


_BASE_L = ((0, 0), (1, 0), (1, 1))
L_SHAPES = [_BASE_L]
for _ in range(3):
    L_SHAPES.append(_rotate_norm(L_SHAPES[-1]))
L_SHAPES = tuple(L_SHAPES)


def _bg_color(grid):
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _neighbors4(r, c):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components_bool(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _components_by_color(grid, mask):
    h, w = grid.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or not mask[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append(comp)
    return comps


def _template_bbox(grid, template_color):
    coords = np.argwhere(grid == template_color)
    assert coords.size > 0, "template must exist"
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_structure(grid):
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    counts = np.bincount(g.ravel(), minlength=10)
    counts[bg] = -1
    template_color = int(np.argmax(counts))
    r0, c0, r1, c1 = _template_bbox(g, template_color)

    markers = {}
    marker_cells = set()

    def add_marker(r, c, dr, dc):
        color = int(g[r, c])
        if color in (bg, template_color):
            return
        assert color not in markers, "marker colors must be unique"
        markers[color] = ((r, c), (dr, dc))
        marker_cells.add((r, c))

    if r0 > 0:
        for c in range(c0, c1 + 1):
            add_marker(r0 - 1, c, 1, 0)
    if r1 + 1 < g.shape[0]:
        for c in range(c0, c1 + 1):
            add_marker(r1 + 1, c, -1, 0)
    if c0 > 0:
        for r in range(r0, r1 + 1):
            add_marker(r, c0 - 1, 0, 1)
    if c1 + 1 < g.shape[1]:
        for r in range(r0, r1 + 1):
            add_marker(r, c1 + 1, 0, -1)

    slot_mask = np.zeros_like(g, dtype=bool)
    slot_mask[r0 : r1 + 1, c0 : c1 + 1] = g[r0 : r1 + 1, c0 : c1 + 1] == bg
    slots = []
    cell_to_slot = {}
    for comp in _components_bool(slot_mask):
        if any(r in (r0, r1) or c in (c0, c1) for r, c in comp):
            continue
        norm = _normalize(comp)
        assert len(comp) == 3 and norm in L_SHAPES, "slots must be 3-cell L shapes"
        slot_id = len(slots)
        slots.append(comp)
        for cell in comp:
            cell_to_slot[cell] = slot_id

    piece_mask = (g != bg) & (g != template_color)
    for r, c in marker_cells:
        piece_mask[r, c] = False
    pieces = {}
    for comp in _components_by_color(g, piece_mask):
        color = int(g[comp[0]])
        norm = _normalize(comp)
        assert len(comp) == 3 and norm in L_SHAPES, "pieces must be 3-cell L shapes"
        assert color not in pieces, "piece colors must be unique"
        pieces[color] = comp

    return {
        "bg": bg,
        "template_color": template_color,
        "bbox": (r0, c0, r1, c1),
        "markers": markers,
        "marker_cells": marker_cells,
        "slots": slots,
        "cell_to_slot": cell_to_slot,
        "pieces": pieces,
    }


def _slot_for_marker(grid, bbox, template_color, marker_pos, delta, cell_to_slot):
    r0, c0, r1, c1 = bbox
    r, c = marker_pos
    dr, dc = delta
    r += dr
    c += dc
    crossed_template = False
    while r0 <= r <= r1 and c0 <= c <= c1:
        value = int(grid[r, c])
        if value == template_color:
            crossed_template = True
            r += dr
            c += dc
            continue
        if crossed_template and (r, c) in cell_to_slot:
            return cell_to_slot[(r, c)]
        break
    raise AssertionError("marker must point to a unique slot")


def _unique_rotation(piece_cells, slot_cells):
    current = _normalize(piece_cells)
    target = _normalize(slot_cells)
    matches = []
    for k in range(4):
        if current == target:
            matches.append(k)
        current = _rotate_norm(current)
    assert len(matches) == 1, "piece must match slot under exactly one rotation"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each external L-fragment to the edge marker of the same color, locate the
    adjacent internal slot indicated by that marker, and fill the slot after the unique
    rotation that makes the fragment's L shape agree with the slot. External fragments
    are removed, markers stay in place, and the filled slot keeps the fragment color.
    """
    g = np.asarray(grid, dtype=int)
    info = _extract_structure(g)

    markers = info["markers"]
    pieces = info["pieces"]
    slots = info["slots"]
    assert markers and len(markers) == len(pieces) == len(slots), "counts must agree"

    out = g.copy()
    for cells in pieces.values():
        for r, c in cells:
            out[r, c] = info["bg"]

    filled = set()
    for color, (marker_pos, delta) in markers.items():
        assert color in pieces, "every marker must have a same-color fragment"
        slot_id = _slot_for_marker(
            g,
            info["bbox"],
            info["template_color"],
            marker_pos,
            delta,
            info["cell_to_slot"],
        )
        assert slot_id not in filled, "each slot must be claimed once"
        _unique_rotation(pieces[color], slots[slot_id])
        for r, c in slots[slot_id]:
            out[r, c] = int(color)
        filled.add(slot_id)

    assert len(filled) == len(slots), "all slots must be filled"
    return out


def _slot_candidate(template_h, template_w, side, base, orientation, marker_pick):
    shape = L_SHAPES[orientation]
    if side == "top":
        br, bc = 1, base
        marker_row, marker_col = -1, None
        selector = lambda cell: cell[0] == 1
    elif side == "bottom":
        br, bc = template_h - 3, base
        marker_row, marker_col = template_h, None
        selector = lambda cell: cell[0] == template_h - 2
    elif side == "left":
        br, bc = base, 1
        marker_row, marker_col = None, -1
        selector = lambda cell: cell[1] == 1
    else:
        br, bc = base, template_w - 3
        marker_row, marker_col = None, template_w
        selector = lambda cell: cell[1] == template_w - 2

    cells = [(br + dr, bc + dc) for dr, dc in shape]
    if any(not (1 <= r <= template_h - 2 and 1 <= c <= template_w - 2) for r, c in cells):
        return None

    marker_options = []
    for r, c in cells:
        if selector((r, c)):
            if side in ("top", "bottom"):
                marker_options.append((marker_row, c))
            else:
                marker_options.append((r, marker_col))

    if not marker_options:
        return None

    marker_local = marker_options[marker_pick % len(marker_options)]
    return {"cells": cells, "marker_local": marker_local, "orientation": orientation, "side": side}


def _slot_conflicts(candidate_cells, blocked):
    return any(cell in blocked for cell in candidate_cells)


def _add_slot_block(candidate_cells, blocked):
    for r, c in candidate_cells:
        blocked.add((r, c))
        for nr, nc in _neighbors4(r, c):
            blocked.add((nr, nc))


def _max_slots_for_template(template_h, template_w):
    side_w = max(1, (template_w - 3) // 3)
    side_h = max(1, (template_h - 3) // 3)
    return 2 * (side_w + side_h)


def _place_slots(rng, template_h, template_w, count):
    blocked = set()
    used_markers = set()
    side_counts = Counter()
    slots = []

    for _ in range(count):
        placed = False
        for _ in range(300):
            min_count = min(side_counts.get(side, 0) for side in SIDES)
            if rng.random() < 0.7:
                side = rng.choice([s for s in SIDES if side_counts.get(s, 0) == min_count])
            else:
                side = rng.choice(SIDES)
            orientation = rng.randrange(4)
            if side in ("top", "bottom"):
                if template_w < 8:
                    return None
                base = rng.randint(2, template_w - 4)
            else:
                if template_h < 8:
                    return None
                base = rng.randint(2, template_h - 4)
            marker_pick = rng.randrange(2)
            slot = _slot_candidate(template_h, template_w, side, base, orientation, marker_pick)
            if slot is None:
                continue
            if _slot_conflicts(slot["cells"], blocked):
                continue
            if slot["marker_local"] in used_markers:
                continue
            _add_slot_block(slot["cells"], blocked)
            used_markers.add(slot["marker_local"])
            side_counts[side] += 1
            slots.append(slot)
            placed = True
            break
        if not placed:
            return None
    return slots


def _piece_cells_clear(grid, cells):
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BG:
            return False
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != BG:
                return False
    return True


def _place_piece(grid, rng, color, orientation, bbox):
    h, w = grid.shape
    shape = L_SHAPES[orientation]
    r0, c0, r1, c1 = bbox
    for _ in range(500):
        br = rng.randint(0, h - 2)
        bc = rng.randint(0, w - 2)
        cells = [(br + dr, bc + dc) for dr, dc in shape]
        if any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in cells):
            continue
        if not _piece_cells_clear(grid, cells):
            continue
        for r, c in cells:
            grid[r, c] = int(color)
        return cells
    return None


def _validate_generated(grid):
    try:
        info = _extract_structure(grid)
        markers = info["markers"]
        slots = info["slots"]
        pieces = info["pieces"]
        if len(markers) < 2 or len(markers) != len(slots) or len(markers) != len(pieces):
            return False
        claimed = set()
        for color, (marker_pos, delta) in markers.items():
            if color not in pieces:
                return False
            slot_id = _slot_for_marker(
                grid,
                info["bbox"],
                info["template_color"],
                marker_pos,
                delta,
                info["cell_to_slot"],
            )
            if slot_id in claimed:
                return False
            claimed.add(slot_id)
            _unique_rotation(pieces[color], slots[slot_id])
        out = main(grid)
        for cells in pieces.values():
            for r, c in cells:
                if int(out[r, c]) != info["bg"]:
                    return False
        return len(claimed) == len(slots)
    except Exception:
        return False


def _gen_attempt(rng):
    template_h = rng.randint(8, 12)
    template_w = rng.randint(8, 13)

    margin_v = rng.randint(4, 7)
    margin_h = rng.randint(4, 7)
    top = max(4, margin_v + rng.randint(-1, 1))
    bottom = max(4, margin_v + rng.randint(-1, 1))
    left = max(4, margin_h + rng.randint(-1, 1))
    right = max(4, margin_h + rng.randint(-1, 1))

    h = top + template_h + bottom
    w = left + template_w + right
    grid = np.full((h, w), BG, dtype=int)
    r0, c0 = top, left
    r1, c1 = r0 + template_h - 1, c0 + template_w - 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = TEMPLATE

    max_slots = min(6, _max_slots_for_template(template_h, template_w))
    if max_slots < 2:
        return None
    slot_count = rng.randint(2, max_slots)
    colors = rng.sample(COLOR_POOL, slot_count)

    slots = _place_slots(rng, template_h, template_w, slot_count)
    if slots is None:
        return None

    for i, slot in enumerate(slots):
        color = colors[i]
        piece_orientation = rng.randrange(4)
        slot["color"] = color
        slot["piece_orientation"] = piece_orientation

        for sr, sc in slot["cells"]:
            grid[r0 + sr, c0 + sc] = BG
        mr, mc = slot["marker_local"]
        grid[r0 + mr, c0 + mc] = int(color)

    for slot in slots:
        cells = _place_piece(
            grid,
            rng,
            slot["color"],
            slot["piece_orientation"],
            (r0, c0, r1, c1),
        )
        if cells is None:
            return None
        slot["piece_cells"] = cells

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
