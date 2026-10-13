import random

import numpy as np


BG = 0
RED = 2
MARKER = 3
YELLOW = 4
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _component_count(mask: np.ndarray) -> int:
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        count += 1
        stack = [(r, c)]
        seen[r, c] = True
        while stack:
            rr, cc = stack.pop()
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return count


def _find_marker_box(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    candidates = []
    for color in np.unique(g):
        color = int(color)
        if color == BG:
            continue
        mask = g == color
        if int(mask.sum()) != 4:
            continue
        box = _bbox(mask)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        if r1 - r0 + 1 == 2 and c1 - c0 + 1 == 2 and np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color):
            candidates.append((color, box))
    assert len(candidates) == 1, "expected exactly one solid 2x2 marker square"
    return candidates[0][1]


def _reflect_red(grid: np.ndarray, marker_box):
    out = np.asarray(grid, dtype=int).copy()
    r0, c0, r1, c1 = marker_box
    h, w = out.shape
    for r, c in np.argwhere(out == RED):
        r = int(r)
        c = int(c)
        for rr in (r, r0 + r1 - r):
            for cc in (c, c0 + c1 - c):
                if 0 <= rr < h and 0 <= cc < w and not (r0 <= rr <= r1 and c0 <= cc <= c1):
                    out[rr, cc] = RED
    return out


def _fragment_half(grid: np.ndarray, marker_box):
    rows, cols = np.argwhere(np.asarray(grid, dtype=int) == RED).T
    r0, c0, r1, c1 = marker_box
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    vertical = "top" if float(rows.mean()) < center_r else "bottom"
    horizontal = "left" if float(cols.mean()) < center_c else "right"
    return vertical, horizontal


def _candidate_openings(red_mask: np.ndarray, box):
    r0, c0, r1, c1 = box
    sides = {
        "top": [(r0, c) for c in range(c0, c1 + 1) if not red_mask[r0, c]],
        "bottom": [(r1, c) for c in range(c0, c1 + 1) if not red_mask[r1, c]],
        "left": [(r, c0) for r in range(r0, r1 + 1) if not red_mask[r, c0]],
        "right": [(r, c1) for r in range(r0, r1 + 1) if not red_mask[r, c1]],
    }
    result = {}
    for side, coords in sides.items():
        if not coords:
            continue
        if side in ("top", "bottom"):
            values = [c for _, c in coords]
        else:
            values = [r for r, _ in coords]
        if max(values) - min(values) + 1 == len(values):
            result[side] = coords
    assert result, "expected at least one boundary opening"
    return result


def _select_active_side(candidates, fragment_half):
    sides = set(candidates)
    vertical, horizontal = fragment_half
    if sides <= {"top", "bottom"}:
        return vertical if vertical in sides else sorted(sides)[0]
    if sides <= {"left", "right"}:
        return horizontal if horizontal in sides else sorted(sides)[0]
    if len(sides) == 1:
        return next(iter(sides))
    priority = [vertical, horizontal, "top", "bottom", "left", "right"]
    for side in priority:
        if side in sides:
            return side
    raise AssertionError("failed to select active opening")


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror the red corner around the green marker, fill the emergent cup, and emit yellow from the aligned opening.

    Input structure:
    - Black background with one green 2x2 marker square and one red connected corner fragment.
    - The red fragment stays in a single corner sector around the marker and may touch one marker-aligned strip to bridge one axis after reflection.
    - The completed outline becomes an almost closed orthogonal ring whose missing boundary cells lie on one axis pair.

    Rule:
    - Use the green 2x2 square as the biaxial mirror center and reflect every red cell horizontally, vertically, and by 180 degrees.
    - Find the completed red outline's bounding box and treat every black cell inside that box as cup interior, filling it yellow while leaving the green marker unchanged.
    - Detect the boundary gap side(s); the active opening is the gap on the same top/bottom or left/right side as the original red fragment.
    - From every yellow cell in that active opening, extend yellow straight outward until the ray hits a non-background cell or the grid edge.

    Key insight:
    - The fragment alone does not reveal whether the cup opens vertically or horizontally; you must first mirror it to expose the gap axis, then use the fragment's original half-plane to resolve which opposite opening emits the ray.
    """
    g = np.asarray(grid, dtype=int)
    marker_box = _find_marker_box(g)
    fragment_half = _fragment_half(g, marker_box)

    out = _reflect_red(g, marker_box)
    red_mask = out == RED
    box = _bbox(red_mask)
    assert box is not None, "expected red outline after reflection"
    r0, c0, r1, c1 = box

    fill_mask = np.zeros_like(red_mask, dtype=bool)
    fill_mask[r0 : r1 + 1, c0 : c1 + 1] = out[r0 : r1 + 1, c0 : c1 + 1] == BG
    out[fill_mask] = YELLOW

    openings = _candidate_openings(red_mask, box)
    active_side = _select_active_side(openings.keys(), fragment_half)
    dr, dc = SIDE_TO_DIR[active_side]

    for r, c in openings[active_side]:
        rr = r + dr
        cc = c + dc
        while 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and out[rr, cc] == BG:
            out[rr, cc] = YELLOW
            rr += dr
            cc += dc

    return out


_EXTENTS = [(2, 3), (3, 2), (2, 4), (4, 2), (3, 3), (3, 4), (4, 3), (4, 4)]


def _build_ring_mask(shape, marker_rc, extent_r, extent_c, gap_axis):
    h, w = shape
    mr, mc = marker_rc
    r0 = mr - extent_r
    r1 = mr + 1 + extent_r
    c0 = mc - extent_c
    c1 = mc + 1 + extent_c
    mask = np.zeros((h, w), dtype=bool)
    if gap_axis == "tb":
        mask[r0 : r1 + 1, c0] = True
        mask[r0 : r1 + 1, c1] = True
        mask[r0, c0 : c1 + 1] = True
        mask[r1, c0 : c1 + 1] = True
        mask[r0, mc : mc + 2] = False
        mask[r1, mc : mc + 2] = False
    else:
        mask[r0, c0 : c1 + 1] = True
        mask[r1, c0 : c1 + 1] = True
        mask[r0 : r1 + 1, c0] = True
        mask[r0 : r1 + 1, c1] = True
        mask[mr : mr + 2, c0] = False
        mask[mr : mr + 2, c1] = False
    return mask


def _build_source_fragment(shape, marker_rc, extent_r, extent_c, gap_axis, sector):
    h, w = shape
    mr, mc = marker_rc
    rt = mr - extent_r
    rb = mr + 1 + extent_r
    cl = mc - extent_c
    cr = mc + 1 + extent_c
    frag = np.zeros((h, w), dtype=bool)

    if gap_axis == "tb":
        if sector == ("top", "left"):
            frag[rt, cl:mc] = True
            frag[rt : mr + 1, cl] = True
        elif sector == ("top", "right"):
            frag[rt, mc + 2 : cr + 1] = True
            frag[rt : mr + 1, cr] = True
        elif sector == ("bottom", "left"):
            frag[rb, cl:mc] = True
            frag[mr + 1 : rb + 1, cl] = True
        else:
            frag[rb, mc + 2 : cr + 1] = True
            frag[mr + 1 : rb + 1, cr] = True
    else:
        if sector == ("top", "left"):
            frag[rt:mr, cl] = True
            frag[rt, cl : mc + 1] = True
        elif sector == ("top", "right"):
            frag[rt:mr, cr] = True
            frag[rt, mc + 1 : cr + 1] = True
        elif sector == ("bottom", "left"):
            frag[mr + 2 : rb + 1, cl] = True
            frag[rb, cl : mc + 1] = True
        else:
            frag[mr + 2 : rb + 1, cr] = True
            frag[rb, mc + 1 : cr + 1] = True
    return frag


def _semantic_check(inp: np.ndarray, intended_axis: str, intended_side: str) -> bool:
    marker_box = _find_marker_box(inp)
    full = _reflect_red(inp, marker_box)
    red_mask = full == RED
    box = _bbox(red_mask)
    if box is None or _component_count(inp == RED) != 1:
        return False
    openings = _candidate_openings(red_mask, box)
    if intended_axis == "tb" and set(openings) != {"top", "bottom"}:
        return False
    if intended_axis == "lr" and set(openings) != {"left", "right"}:
        return False
    active = _select_active_side(openings.keys(), _fragment_half(inp, marker_box))
    if active != intended_side:
        return False

    out = main(inp)
    r0, c0, r1, c1 = box
    if not np.all(out[marker_box[0] : marker_box[2] + 1, marker_box[1] : marker_box[3] + 1] == MARKER):
        return False
    if not np.any(out[r0 : r1 + 1, c0 : c1 + 1] == YELLOW):
        return False

    dr, dc = SIDE_TO_DIR[intended_side]
    has_external_ray = False
    for r, c in openings[intended_side]:
        rr = r + dr
        cc = c + dc
        while 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
            if out[rr, cc] == YELLOW:
                has_external_ray = True
                break
            if out[rr, cc] != BG:
                break
            rr += dr
            cc += dc
        if has_external_ray:
            break
    return has_external_ray


def _gen_attempt():
    extent_r, extent_c = random.choice(_EXTENTS)
    gap_axis = random.choice(["tb", "lr"])

    if gap_axis == "tb":
        active_side = random.choice(["top", "bottom"])
        quadrant = (active_side, random.choice(["left", "right"]))
    else:
        active_side = random.choice(["left", "right"])
        quadrant = (random.choice(["top", "bottom"]), active_side)

    outer_h = 2 * extent_r + 2
    outer_w = 2 * extent_c + 2
    top_margin = random.randint(1, 6 if active_side != "top" else 8)
    bottom_margin = random.randint(1, 6 if active_side != "bottom" else 8)
    left_margin = random.randint(1, 6 if active_side != "left" else 8)
    right_margin = random.randint(1, 6 if active_side != "right" else 8)
    h = outer_h + top_margin + bottom_margin
    w = outer_w + left_margin + right_margin
    if h > 30 or w > 30:
        return None

    mr = top_margin + extent_r
    mc = left_margin + extent_c
    fragment = _build_source_fragment((h, w), (mr, mc), extent_r, extent_c, gap_axis, quadrant)
    if not (5 <= int(fragment.sum()) <= 8):
        return None
    if _component_count(fragment) != 1:
        return None

    inp = np.full((h, w), BG, dtype=int)
    inp[fragment] = RED
    inp[mr : mr + 2, mc : mc + 2] = MARKER

    if not _semantic_check(inp, gap_axis, active_side):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
