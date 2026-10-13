import numpy as np


BG = 0
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250
DIRECTIONS = ("left", "right", "up", "down")
DIR_VECS = {
    "left": (0, -1),
    "right": (0, 1),
    "up": (-1, 0),
    "down": (1, 0),
}
SHAPE_KINDS = ("wide", "tall", "square")


def _components4(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray):
    rs = coords[:, 0]
    cs = coords[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _direction_from_dye(h: int, w: int, dye_r: int, dye_c: int) -> str:
    center_r = (h - 1) // 2
    center_c = (w - 1) // 2
    off_r = dye_r - center_r
    off_c = dye_c - center_c
    assert off_r != 0 or off_c != 0, "dye point may not sit exactly at the rectangle center"
    if off_c == 0:
        return "up" if off_r < 0 else "down"
    if off_r == 0:
        return "left" if off_c < 0 else "right"
    if abs(off_c) >= abs(off_r):
        return "left" if off_c < 0 else "right"
    return "up" if off_r < 0 else "down"


def _extract_objects(grid: np.ndarray):
    objects = []
    for coords in _components4(grid != BG):
        r0, c0, r1, c1 = _bbox(coords)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = sub.shape
        assert h >= 3 and w >= 3, "rectangles must be at least 3x3"
        assert np.all(sub != BG), "each mixed-color component must fill its bbox"
        colors, counts = np.unique(sub, return_counts=True)
        assert len(colors) == 2, "each rectangle must contain one base color and one dye color"
        dye_idx = np.flatnonzero(counts == 1)
        assert dye_idx.size == 1, "each rectangle must contain exactly one single-pixel dye"
        dye_idx = int(dye_idx[0])
        dye_color = int(colors[dye_idx])
        rect_color = int(colors[1 - dye_idx])
        assert int(counts[1 - dye_idx]) == h * w - 1, "base color must occupy the remaining rectangle"
        dye_pos = np.argwhere(sub == dye_color)
        dye_r, dye_c = map(int, dye_pos[0])
        direction = _direction_from_dye(h, w, dye_r, dye_c)
        shift = min(h, w)
        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": (h, w),
                "rect_color": rect_color,
                "dye_color": dye_color,
                "direction": direction,
                "shift": shift,
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each colored rectangle toward its internal dye point and recolor it with the dye color.

    Input structure:
    - The background is black.
    - Each non-black 4-connected component is a solid axis-aligned rectangle whose bbox is fully filled.
    - Inside every rectangle there is exactly one single-pixel dye of a different color, and dye colors are unique.

    Rule:
    - For each mixed-color rectangle component, locate the unique dye pixel and its color D.
    - Infer the movement direction from the dye position relative to the rectangle center: left/right if the horizontal offset dominates, otherwise up/down; if the dye lies on the center row or center column, use the other axis.
    - Let S be the smaller of the rectangle height and width, clear the original rectangle, and redraw the full rectangle translated by S cells in that direction.
    - Repaint the translated rectangle uniformly with color D; the original dye pixel does not remain visible separately.

    Key insight:
    - The dye pixel simultaneously encodes three things: direction by its offset from the center, distance by the rectangle short side, and final color by its own hue.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = np.zeros_like(g)
    occupied = np.zeros_like(g, dtype=bool)
    for obj in _extract_objects(g):
        r0, c0, r1, c1 = obj["bbox"]
        h, w = obj["shape"]
        dr, dc = DIR_VECS[obj["direction"]]
        shift = obj["shift"]
        nr0 = r0 + dr * shift
        nc0 = c0 + dc * shift
        nr1 = nr0 + h - 1
        nc1 = nc0 + w - 1
        assert 0 <= nr0 <= nr1 < g.shape[0]
        assert 0 <= nc0 <= nc1 < g.shape[1]
        assert not np.any(occupied[nr0 : nr1 + 1, nc0 : nc1 + 1]), "translated rectangles may not overlap"
        out[nr0 : nr1 + 1, nc0 : nc1 + 1] = obj["dye_color"]
        occupied[nr0 : nr1 + 1, nc0 : nc1 + 1] = True
    return out


def _boxes_conflict(a, b, pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _sample_dims(rng: np.random.Generator):
    kind = SHAPE_KINDS[int(rng.integers(0, len(SHAPE_KINDS)))]
    if kind == "square":
        side = int(rng.integers(3, 8))
        return side, side
    if kind == "wide":
        h = int(rng.integers(3, 7))
        w = int(rng.integers(h + 1, 9))
        return h, w
    w = int(rng.integers(3, 7))
    h = int(rng.integers(w + 1, 9))
    return h, w


def _sample_dye_local(h: int, w: int, direction: str, rng: np.random.Generator):
    center_r = (h - 1) // 2
    center_c = (w - 1) // 2
    if direction == "left":
        return center_r, int(rng.integers(0, center_c))
    if direction == "right":
        return center_r, int(rng.integers(center_c + 1, w))
    if direction == "up":
        return int(rng.integers(0, center_r)), center_c
    return int(rng.integers(center_r + 1, h)), center_c


def _place_bbox(h: int, w: int, direction: str, shift: int, grid_h: int, grid_w: int, src_boxes, dst_boxes, rng):
    dr, dc = DIR_VECS[direction]
    r_min = max(0, -dr * shift)
    r_max = grid_h - h - max(0, dr * shift)
    c_min = max(0, -dc * shift)
    c_max = grid_w - w - max(0, dc * shift)
    if r_min > r_max or c_min > c_max:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = int(rng.integers(r_min, r_max + 1))
        c0 = int(rng.integers(c_min, c_max + 1))
        src = (r0, c0, r0 + h - 1, c0 + w - 1)
        dst = (
            r0 + dr * shift,
            c0 + dc * shift,
            r0 + dr * shift + h - 1,
            c0 + dc * shift + w - 1,
        )
        if any(_boxes_conflict(src, other, pad=1) for other in src_boxes):
            continue
        if any(_boxes_conflict(dst, other, pad=1) for other in dst_boxes):
            continue
        return src, dst
    return None


def _expected_output(shape, placed_specs):
    out = np.zeros(shape, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    for spec in placed_specs:
        r0, c0, r1, c1 = spec["dst_bbox"]
        assert not np.any(occupied[r0 : r1 + 1, c0 : c1 + 1])
        out[r0 : r1 + 1, c0 : c1 + 1] = spec["dye_color"]
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
    return out


def _gen_attempt(rng: np.random.Generator):
    n = int(rng.integers(2, 5))
    min_side = 12 + 3 * n
    grid_h = int(rng.integers(min_side, 31))
    grid_w = int(rng.integers(min_side, 31))

    palette = rng.permutation(np.arange(1, 10))
    rect_colors = [int(c) for c in palette[:n]]
    dye_colors = [int(c) for c in palette[n : 2 * n]]
    direction_order = rng.permutation(len(DIRECTIONS))[:n]

    raw_specs = []
    for i in range(n):
        h, w = _sample_dims(rng)
        direction = DIRECTIONS[int(direction_order[i])]
        shift = min(h, w)
        raw_specs.append(
            {
                "h": h,
                "w": w,
                "direction": direction,
                "shift": shift,
                "rect_color": rect_colors[i],
                "dye_color": dye_colors[i],
            }
        )

    order = sorted(range(n), key=lambda idx: raw_specs[idx]["h"] * raw_specs[idx]["w"], reverse=True)
    src_boxes = []
    dst_boxes = []
    placed_specs = []
    for idx in order:
        spec = raw_specs[idx]
        placement = _place_bbox(
            spec["h"],
            spec["w"],
            spec["direction"],
            spec["shift"],
            grid_h,
            grid_w,
            src_boxes,
            dst_boxes,
            rng,
        )
        if placement is None:
            return None
        src_bbox, dst_bbox = placement
        src_boxes.append(src_bbox)
        dst_boxes.append(dst_bbox)
        placed_spec = dict(spec)
        placed_spec["src_bbox"] = src_bbox
        placed_spec["dst_bbox"] = dst_bbox
        placed_specs.append(placed_spec)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for spec in placed_specs:
        r0, c0, r1, c1 = spec["src_bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = spec["rect_color"]
        dye_r, dye_c = _sample_dye_local(spec["h"], spec["w"], spec["direction"], rng)
        grid[r0 + dye_r, c0 + dye_c] = spec["dye_color"]

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, _expected_output(grid.shape, placed_specs)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
