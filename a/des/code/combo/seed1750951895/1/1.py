import random
from collections import Counter

import numpy as np


BLACK = 0
COLORS = tuple(range(1, 10))
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

SHAPE_LIBRARY = [
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
]

MAX_GENERATION_ATTEMPTS = 300


def _normalize(coords):
    pts = np.array(coords, dtype=int)
    pts[:, 0] -= int(pts[:, 0].min())
    pts[:, 1] -= int(pts[:, 1].min())
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    return tuple((int(r), int(c)) for r, c in pts)


def _shape_variants(coords):
    pts = np.array(coords, dtype=int)
    out = []
    seen = set()
    for flip in (False, True):
        cur = pts.copy()
        if flip:
            cur[:, 1] *= -1
        for k in range(4):
            rot = cur.copy()
            for _ in range(k):
                rot = np.stack([-rot[:, 1], rot[:, 0]], axis=1)
            key = _normalize(rot)
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def _canonical_shape(coords):
    return min(_shape_variants(coords))


def _components(mask):
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for dr, dc in N4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _legend_order(grid):
    row = np.asarray(grid[-1], dtype=int)
    order = []
    prev = None
    for val in row:
        val = int(val)
        if val != BLACK and val != prev:
            order.append(val)
        prev = val
    assert len(order) >= 2
    return order


def _find_frame(grid, color):
    pts = [(int(r), int(c)) for r, c in np.argwhere(grid[:-1] == color)]
    assert pts
    r0, c0, r1, c1 = _bbox(pts)
    assert r1 - r0 >= 2 and c1 - c0 >= 2
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    assert set(pts) == expected
    return r0, c0, r1, c1


def _frame_holes(grid, frame_bbox):
    r0, c0, r1, c1 = frame_bbox
    inside = grid[r0 + 1 : r1, c0 + 1 : c1] == BLACK
    holes = []
    for comp in _components(inside):
        rel = _normalize(comp)
        abs_comp = [(r0 + 1 + r, c0 + 1 + c) for r, c in comp]
        holes.append({"coords": abs_comp, "shape": _canonical_shape(rel)})
    assert holes
    return holes


def _sprite_map(grid, frame_boxes):
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    blocked[-1, :] = True
    for r0, c0, r1, c1 in frame_boxes:
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    sprites = {}
    nonblack = (grid != BLACK) & (~blocked)
    for comp in _components(nonblack):
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1
        color = next(iter(colors))
        rel = _normalize([(r - min(r for r, _ in comp), c - min(c for _, c in comp)) for r, c in comp])
        key = _canonical_shape(rel)
        assert key not in sprites
        sprites[key] = color
    return sprites


def _render_nested_rings(order, counts):
    total = sum(counts[color] for color in order)
    size = 2 * total + 1
    out = np.zeros((size, size), dtype=int)
    inset = 0
    for color in order:
        thickness = counts[color]
        out[inset : size - inset, inset : size - inset] = color
        inset += thickness
    out[inset : size - inset, inset : size - inset] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend = _legend_order(g)
    frame_boxes = [_find_frame(g, color) for color in legend]
    sprites = _sprite_map(g, frame_boxes)

    hole_counts = {}
    for color, box in zip(legend, frame_boxes):
        holes = _frame_holes(g, box)
        hole_counts[color] = len(holes)
        for hole in holes:
            assert hole["shape"] in sprites

    return _render_nested_rings(legend, hole_counts)


def _can_place(mask, top, left, shape, pad=1):
    h, w = mask.shape
    sh = 1 + max(r for r, _ in shape)
    sw = 1 + max(c for _, c in shape)
    if top < 0 or left < 0 or top + sh > h or left + sw > w:
        return False
    for r, c in shape:
        rr = top + r
        cc = left + c
        r0 = max(0, rr - pad)
        r1 = min(h, rr + pad + 1)
        c0 = max(0, cc - pad)
        c1 = min(w, cc + pad + 1)
        if mask[r0:r1, c0:c1].any():
            return False
    return True


def _stamp(grid, top, left, shape, color):
    for r, c in shape:
        grid[top + r, left + c] = color


def _place_holes(rng, inner_h, inner_w, hole_shapes):
    occupied = np.zeros((inner_h, inner_w), dtype=bool)
    placed = []
    for shape in hole_shapes:
        variants = _shape_variants(shape)
        rng.shuffle(variants)
        ok = False
        for variant in variants:
            sh = 1 + max(r for r, _ in variant)
            sw = 1 + max(c for _, c in variant)
            if inner_h - sh - 2 < 0 or inner_w - sw - 2 < 0:
                continue
            tops = list(range(1, inner_h - sh))
            lefts = list(range(1, inner_w - sw))
            rng.shuffle(tops)
            rng.shuffle(lefts)
            for top in tops:
                for left in lefts:
                    if _can_place(occupied, top, left, variant, pad=1):
                        for r, c in variant:
                            occupied[top + r, left + c] = True
                        placed.append((top, left, variant))
                        ok = True
                        break
                if ok:
                    break
            if ok:
                break
        if not ok:
            return None
    return placed


def _place_rect(rng, occupied, rh, rw, row_limit):
    h, w = occupied.shape
    tops = list(range(1, row_limit - rh))
    lefts = list(range(1, w - rw - 1))
    rng.shuffle(tops)
    rng.shuffle(lefts)
    for top in tops:
        for left in lefts:
            r0 = max(0, top - 1)
            r1 = min(h, top + rh + 1)
            c0 = max(0, left - 1)
            c1 = min(w, left + rw + 1)
            if occupied[r0:r1, c0:c1].any():
                continue
            occupied[top : top + rh, left : left + rw] = True
            return top, left
    return None


def _place_sprite(rng, occupied, shape, row_limit):
    sh = 1 + max(r for r, _ in shape)
    sw = 1 + max(c for _, c in shape)
    pos = _place_rect(rng, occupied, sh, sw, row_limit)
    if pos is None:
        return None
    return pos


def _draw_legend(rng, grid, order):
    w = grid.shape[1]
    row = np.zeros(w, dtype=int)
    pos = 1
    for i, color in enumerate(order):
        run = rng.randint(1, 3)
        if pos + run > w - 1:
            return False
        row[pos : pos + run] = color
        pos += run
        if i != len(order) - 1:
            pos += rng.randint(1, 2)
    grid[-1] = row
    return True


def _build_instance(rng):
    h = rng.randint(22, 31)
    w = rng.randint(24, 35)
    n_frames = rng.randint(2, 4)
    hole_counts = [rng.randint(1, 4) for _ in range(n_frames)]
    total_holes = sum(hole_counts)
    if total_holes > len(SHAPE_LIBRARY):
        return None

    color_pool = list(COLORS)
    rng.shuffle(color_pool)
    frame_colors = color_pool[:n_frames]
    filler_colors = color_pool[n_frames : 2 * n_frames]
    sprite_palette = color_pool[2 * n_frames :] or list(COLORS)
    order = frame_colors[:]
    rng.shuffle(order)

    hole_shape_keys = rng.sample(SHAPE_LIBRARY, total_holes)
    hole_cursor = 0
    frame_specs = []
    for i in range(n_frames):
        hole_shapes = hole_shape_keys[hole_cursor : hole_cursor + hole_counts[i]]
        hole_cursor += hole_counts[i]
        inner_h = rng.randint(6, 10)
        inner_w = rng.randint(7, 12)
        placed = _place_holes(rng, inner_h, inner_w, hole_shapes)
        if placed is None:
            return None
        frame_specs.append(
            {
                "frame_color": frame_colors[i],
                "filler_color": filler_colors[i],
                "inner_h": inner_h,
                "inner_w": inner_w,
                "holes": placed,
            }
        )

    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    frame_boxes = {}
    usable_rows = h - 2

    for spec in frame_specs:
        fh = spec["inner_h"] + 2
        fw = spec["inner_w"] + 2
        pos = _place_rect(rng, occupied, fh, fw, usable_rows)
        if pos is None:
            return None
        top, left = pos
        bottom = top + fh - 1
        right = left + fw - 1
        grid[top : bottom + 1, left : right + 1] = spec["filler_color"]
        grid[top, left : right + 1] = spec["frame_color"]
        grid[bottom, left : right + 1] = spec["frame_color"]
        grid[top : bottom + 1, left] = spec["frame_color"]
        grid[top : bottom + 1, right] = spec["frame_color"]
        for hole_top, hole_left, variant in spec["holes"]:
            for r, c in variant:
                grid[top + 1 + hole_top + r, left + 1 + hole_left + c] = BLACK
        frame_boxes[spec["frame_color"]] = (top, left, bottom, right)

    sprite_colors = []
    for i in range(total_holes):
        sprite_colors.append(sprite_palette[i % len(sprite_palette)])
    rng.shuffle(sprite_colors)

    sprite_occupied = occupied.copy()
    sprite_occupied[-1, :] = True
    sprite_index = 0
    expected_counts = {}
    for spec in frame_specs:
        expected_counts[spec["frame_color"]] = len(spec["holes"])
        for hole_top, hole_left, variant in spec["holes"]:
            sprite_variant = rng.choice(_shape_variants(variant))
            pos = _place_sprite(rng, sprite_occupied, sprite_variant, usable_rows)
            if pos is None:
                return None
            top, left = pos
            color = sprite_colors[sprite_index]
            sprite_index += 1
            _stamp(grid, top, left, sprite_variant, color)
            for r, c in sprite_variant:
                sprite_occupied[top + r, left + c] = True

    if not _draw_legend(rng, grid, order):
        return None

    expected = _render_nested_rings(order, expected_counts)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_instance(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
