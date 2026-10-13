import random
from collections import Counter, deque

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ["top", "bottom", "left", "right"]
GEN_RETRIES = 400
EMPTY = -1


def _most_common(values) -> int:
    return int(Counter(int(v) for v in values).most_common(1)[0][0])


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(cells)
    return parts


def _shape_color(grid: np.ndarray, bg: int) -> int:
    vals = [int(v) for v in grid.flatten() if int(v) != bg]
    assert vals, "expected foreground pixels"
    return _most_common(vals)


def _rotate_point(point, shape, k: int):
    r, c = point
    h, w = shape
    if k == 0:
        return (r, c), (h, w)
    if k == 1:
        return (w - 1 - c, r), (w, h)
    if k == 2:
        return (h - 1 - r, w - 1 - c), (h, w)
    return (c, h - 1 - r), (w, h)


def _rotate_points(points, shape, k: int):
    rotated = []
    new_shape = shape
    for point in points:
        rp, new_shape = _rotate_point(point, shape, k)
        rotated.append(rp)
    return rotated, new_shape


def _find_notches(grid: np.ndarray, skeleton_cells, shape: int):
    r0, c0, r1, c1 = _bbox(skeleton_cells)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    occ = np.zeros((h, w), dtype=bool)
    skeleton_set = set(skeleton_cells)
    for r, c in skeleton_cells:
        occ[r - r0, c - c0] = True

    missing = ~occ
    reachable = np.zeros_like(missing)
    q = deque()
    for rr in range(h):
        for cc in (0, w - 1):
            if missing[rr, cc] and not reachable[rr, cc]:
                reachable[rr, cc] = True
                q.append((rr, cc))
    for cc in range(w):
        for rr in (0, h - 1):
            if missing[rr, cc] and not reachable[rr, cc]:
                reachable[rr, cc] = True
                q.append((rr, cc))
    while q:
        rr, cc = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and missing[nr, nc] and not reachable[nr, nc]:
                reachable[nr, nc] = True
                q.append((nr, nc))

    notches = {}
    for comp in _components(reachable):
        abs_cells = {(r0 + rr, c0 + cc) for rr, cc in comp}
        marker_positions = {}
        for rr, cc in abs_cells:
            for dr, dc in DIRS4:
                nr, nc = rr + dr, cc + dc
                if (nr, nc) in skeleton_set and int(grid[nr, nc]) != shape:
                    marker_positions[(nr, nc)] = int(grid[nr, nc])
        colors = set(marker_positions.values())
        assert len(colors) == 1, "each notch should have exactly one adjacent marker color"
        color = next(iter(colors))
        positions = [pos for pos, col in marker_positions.items() if col == color]
        assert len(positions) == 1, "each notch should have exactly one marker pixel"
        notches[color] = {"cells": abs_cells, "marker": positions[0]}
    assert notches, "expected at least one notch"
    return notches


def _fragment_info(grid: np.ndarray, cells, shape: int):
    colors = [(r, c, int(grid[r, c])) for r, c in cells if int(grid[r, c]) != shape]
    assert len(colors) == 1, "each loose fragment should have exactly one marker pixel"
    mr, mc, color = colors[0]
    r0, c0, r1, c1 = _bbox(cells)
    body = []
    for r, c in cells:
        if (r, c) == (mr, mc):
            continue
        assert int(grid[r, c]) == shape, "fragment body must use the shape color"
        body.append((r - r0, c - c0))
    return {
        "color": color,
        "shape": (r1 - r0 + 1, c1 - c0 + 1),
        "marker": (mr - r0, mc - c0),
        "body": body,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the scattered marked fragments into the skeleton's color-coded edge notches.

    Input structure:
    - The background is the dominant color.
    - One large foreground component is a skeleton drawn mostly in one shared shape color, with rectangular edge notches cut out of its bbox.
    - Each notch has one uniquely colored marker pixel on the skeleton side adjacent to that notch.
    - Smaller scattered components use the same shape color and each carries one marker pixel whose color matches exactly one notch marker.

    Rule:
    - Identify the skeleton, the notch regions inside its bbox, and the loose fragments.
    - Pair each fragment to the notch with the same marker color.
    - Try the four quarter-turn rotations; keep the unique rotation where the fragment's non-marker cells exactly fill the notch and the fragment marker lands on the notch marker cell.
    - Erase the fragments from their original locations and repaint all marker pixels with the shared shape color.

    Key insight:
    - The marker color chooses the destination notch, and the marker position is also the anchor that makes only one rotated placement fit exactly.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common(g.flatten())
    shape = _shape_color(g, bg)

    components = _components(g != bg)
    assert len(components) >= 2, "expected one skeleton and at least one loose fragment"
    skeleton = max(components, key=len)
    fragments = [comp for comp in components if comp is not skeleton]

    notch_by_color = _find_notches(g, skeleton, shape)
    fragment_by_color = {}
    for comp in fragments:
        info = _fragment_info(g, comp, shape)
        fragment_by_color[info["color"]] = info

    assert set(fragment_by_color) == set(notch_by_color), "fragment markers and notch markers must match exactly"

    out = np.full_like(g, bg)
    for r, c in skeleton:
        out[r, c] = shape

    for color, frag in fragment_by_color.items():
        target = notch_by_color[color]
        matches = []
        for k in range(4):
            body_rot, _ = _rotate_points(frag["body"], frag["shape"], k)
            marker_rot, _ = _rotate_point(frag["marker"], frag["shape"], k)
            off_r = target["marker"][0] - marker_rot[0]
            off_c = target["marker"][1] - marker_rot[1]
            body_abs = {(off_r + r, off_c + c) for r, c in body_rot}
            if body_abs == target["cells"]:
                matches.append(body_abs)
        assert len(matches) == 1, "each fragment should have exactly one valid rotation"
        for r, c in matches[0]:
            out[r, c] = shape
        mr, mc = target["marker"]
        out[mr, mc] = shape

    return out


def _build_fragment(side: str, depth: int, span: int, offset: int, shape: int, marker: int) -> np.ndarray:
    if side == "top":
        patch = np.full((depth + 1, span), EMPTY, dtype=int)
        patch[:depth, :] = shape
        patch[depth, offset] = marker
        return patch
    if side == "bottom":
        patch = np.full((depth + 1, span), EMPTY, dtype=int)
        patch[1:, :] = shape
        patch[0, offset] = marker
        return patch
    if side == "left":
        patch = np.full((span, depth + 1), EMPTY, dtype=int)
        patch[:, :depth] = shape
        patch[offset, depth] = marker
        return patch
    patch = np.full((span, depth + 1), EMPTY, dtype=int)
    patch[:, 1:] = shape
    patch[offset, 0] = marker
    return patch


def _can_place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > gh or c0 + w > gw:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(gh, r0 + h + 1)
    cc1 = min(gw, c0 + w + 1)
    if np.any(grid[rr0:rr1, cc0:cc1] != bg):
        return False
    return True


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int):
    h, w = patch.shape
    mask = patch != EMPTY
    grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]


def _expected_output(h: int, w: int, bg: int, shape: int, top: int, left: int, outer_h: int, outer_w: int) -> np.ndarray:
    out = np.full((h, w), bg, dtype=int)
    out[top : top + outer_h, left : left + outer_w] = shape
    return out


def _sample_notch_specs(rng: random.Random, sides, outer_h: int, outer_w: int, colors):
    specs = []
    for side, color in zip(sides, colors):
        if side in ("top", "bottom"):
            depth = rng.choice([2, 3])
            span = rng.choice([2, 3, 4])
            while span == depth:
                span = rng.choice([2, 3, 4])
            start = rng.randint(1, outer_w - span - 2)
            offset = rng.randint(0, span - 1)
        else:
            depth = rng.choice([2, 3])
            span = rng.choice([2, 3, 4])
            while span == depth:
                span = rng.choice([2, 3, 4])
            start = rng.randint(1, outer_h - span - 2)
            offset = rng.randint(0, span - 1)
        specs.append(
            {
                "side": side,
                "color": color,
                "depth": depth,
                "span": span,
                "start": start,
                "offset": offset,
            }
        )
    return specs


def _draw_skeleton(grid: np.ndarray, top: int, left: int, outer_h: int, outer_w: int, shape: int, bg: int, specs):
    grid[top : top + outer_h, left : left + outer_w] = shape
    for spec in specs:
        side = spec["side"]
        start = spec["start"]
        depth = spec["depth"]
        span = spec["span"]
        offset = spec["offset"]
        color = spec["color"]
        if side == "top":
            grid[top : top + depth, left + start : left + start + span] = bg
            grid[top + depth, left + start + offset] = color
        elif side == "bottom":
            grid[top + outer_h - depth : top + outer_h, left + start : left + start + span] = bg
            grid[top + outer_h - depth - 1, left + start + offset] = color
        elif side == "left":
            grid[top + start : top + start + span, left : left + depth] = bg
            grid[top + start + offset, left + depth] = color
        else:
            grid[top + start : top + start + span, left + outer_w - depth : left + outer_w] = bg
            grid[top + start + offset, left + outer_w - depth - 1] = color


def _place_fragment_in_band(rng: random.Random, grid: np.ndarray, patch: np.ndarray, band: str, top: int, left: int, outer_h: int, outer_w: int, bg: int) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    if band == "top":
        r_lo, r_hi = 0, top - h - 1
        c_lo, c_hi = 0, gw - w
    elif band == "bottom":
        r_lo, r_hi = top + outer_h + 1, gh - h
        c_lo, c_hi = 0, gw - w
    elif band == "left":
        r_lo, r_hi = 0, gh - h
        c_lo, c_hi = 0, left - w - 1
    else:
        r_lo, r_hi = 0, gh - h
        c_lo, c_hi = left + outer_w + 1, gw - w
    if r_lo > r_hi or c_lo > c_hi:
        return False
    for _ in range(80):
        r0 = rng.randint(r_lo, r_hi)
        c0 = rng.randint(c_lo, c_hi)
        if _can_place_patch(grid, patch, r0, c0, bg):
            _paste_patch(grid, patch, r0, c0)
            return True
    return False


def _gen_attempt(rng: random.Random):
    piece_count = rng.choice([2, 3, 4])
    top_margin = rng.randint(6, 7)
    bottom_margin = rng.randint(6, 7)
    left_margin = rng.randint(6, 7)
    right_margin = rng.randint(6, 7)

    outer_h = rng.randint(max(8, piece_count + 6), 12)
    outer_w = rng.randint(max(8, piece_count + 6), 14)
    h = top_margin + outer_h + bottom_margin
    w = left_margin + outer_w + right_margin
    if h > 30 or w > 30:
        return None

    palette = list(range(10))
    bg = rng.choice(palette)
    palette.remove(bg)
    shape = rng.choice(palette)
    palette.remove(shape)
    marker_colors = rng.sample(palette, piece_count)

    notch_sides = rng.sample(SIDES, piece_count)
    placement_bands = rng.sample(SIDES, piece_count)

    specs = _sample_notch_specs(rng, notch_sides, outer_h, outer_w, marker_colors)

    grid = np.full((h, w), bg, dtype=int)
    top = top_margin
    left = left_margin
    _draw_skeleton(grid, top, left, outer_h, outer_w, shape, bg, specs)
    skeleton_components = _components(grid != bg)
    if len(skeleton_components) != 1:
        return None
    try:
        notches = _find_notches(grid, skeleton_components[0], shape)
    except AssertionError:
        return None
    if len(notches) != piece_count or set(notches) != set(marker_colors):
        return None

    for spec, band in zip(specs, placement_bands):
        patch = _build_fragment(
            spec["side"],
            spec["depth"],
            spec["span"],
            spec["offset"],
            shape,
            spec["color"],
        )
        patch = np.rot90(patch, rng.randint(0, 3))
        if not _place_fragment_in_band(rng, grid, patch, band, top, left, outer_h, outer_w, bg):
            return None

    expected = _expected_output(h, w, bg, shape, top, left, outer_h, outer_w)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
