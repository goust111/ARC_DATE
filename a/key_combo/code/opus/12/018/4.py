from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GRAY = 5
MARKER_COLORS = (3, 4, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    k = int(factor)
    assert k >= 1
    return np.kron(sprite, np.ones((k, k), dtype=int))


def _inside_bbox(bbox: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _largest_red_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    red_comps = components_mask(grid == RED)
    assert red_comps, "expected a red foreground object"
    largest = max(red_comps, key=lambda pts: int(pts.shape[0]))
    return bbox_coords(largest)


def _find_holes(grid: np.ndarray, red_bbox: tuple[int, int, int, int]) -> dict[int, dict]:
    r0, c0, r1, c1 = red_bbox
    window = grid[r0 : r1 + 1, c0 : c1 + 1]
    comps = components_mask(window != RED)
    holes: dict[int, dict] = {}
    for pts in comps:
        hr0, hc0, hr1, hc1 = bbox_coords(pts)
        assert 0 < hr0 and hr1 < window.shape[0] - 1
        assert 0 < hc0 and hc1 < window.shape[1] - 1
        patch = window[hr0 : hr1 + 1, hc0 : hc1 + 1]
        assert np.all(patch != RED), "hole bbox must be fully non-red"
        marker_color = int(window[hr0, hc0])
        assert marker_color not in (BG, RED, GRAY, BLUE), "hole cue must be a unique marker color"
        abs_bbox = (r0 + hr0, c0 + hc0, r0 + hr1, c0 + hc1)
        holes[marker_color] = {
            "bbox": abs_bbox,
            "height": hr1 - hr0 + 1,
            "width": hc1 - hc0 + 1,
            "cue_color": marker_color,
        }
    assert 2 <= len(holes) <= 3
    return holes


def _find_pieces(grid: np.ndarray, red_bbox: tuple[int, int, int, int]) -> tuple[list[dict], list[dict]]:
    h, w = grid.shape
    mask = (grid != BG) & (grid != RED)
    comps = components_mask(mask)
    bodies: list[dict] = []
    anchors: list[dict] = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        if all(_inside_bbox(red_bbox, int(r), int(c)) for r, c in pts):
            continue
        colors = Counter(int(grid[r, c]) for r, c in pts)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        if set(colors) == {GRAY}:
            side = r1 - r0 + 1
            assert side == c1 - c0 + 1
            assert side in (1, 2)
            assert int(pts.shape[0]) == side * side
            anchors.append({"bbox": (r0, c0, r1, c1), "size": side})
            continue
        non_gray = [color for color in colors if color != GRAY]
        assert len(non_gray) == 1, "each body must contain exactly one marker color"
        marker_color = non_gray[0]
        assert colors[marker_color] == 1, "each body must have exactly one marker cell"
        body_h = r1 - r0 + 1
        body_w = c1 - c0 + 1
        marker_pos = tuple(map(int, np.argwhere(patch == marker_color)[0]))
        bodies.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "marker_color": int(marker_color),
                "marker_pos": marker_pos,
                "height": body_h,
                "width": body_w,
            }
        )
    assert 2 <= len(bodies) <= 3
    assert len(anchors) == len(bodies)
    return bodies, anchors


def _anchor_candidates(body_bbox: tuple[int, int, int, int], anchors: list[dict]) -> list[int]:
    br0, bc0, br1, bc1 = body_bbox
    matches = []
    for i, anchor in enumerate(anchors):
        ar0, ac0, ar1, ac1 = anchor["bbox"]
        if ac1 + 2 == bc0 and _intervals_overlap(ar0, ar1, br0, br1):
            matches.append(i)
        elif bc1 + 2 == ac0 and _intervals_overlap(ar0, ar1, br0, br1):
            matches.append(i)
        elif ar1 + 2 == br0 and _intervals_overlap(ac0, ac1, bc0, bc1):
            matches.append(i)
        elif br1 + 2 == ar0 and _intervals_overlap(ac0, ac1, bc0, bc1):
            matches.append(i)
    return matches


def _pair_bodies_to_anchors(bodies: list[dict], anchors: list[dict]) -> dict[int, int]:
    pairing: dict[int, int] = {}
    used: set[int] = set()
    for bi, body in enumerate(sorted(bodies, key=lambda item: item["bbox"])):
        candidates = [idx for idx in _anchor_candidates(body["bbox"], anchors) if idx not in used]
        assert len(candidates) == 1, "each body must have one adjacent square anchor"
        pairing[id(body)] = candidates[0]
        used.add(candidates[0])
    return pairing


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each marked sprite to the hole with the same cue color, scale it by the hole-to-anchor width ratio, and fill the hole.

    Input structure:
    - A black background contains one large red object near the center and 2-3 separate gray sprite assemblies around it.
    - The red object encloses 2-3 rectangular holes; each hole is black except for a unique marker color at its top-left cell.
    - Each outside assembly has a gray sprite body with one embedded marker pixel plus a detached gray square anchor block separated by a one-cell gap.

    Rule:
    - Pair a sprite body to the hole whose top-left cue color matches the sprite's embedded marker color.
    - Compute the integer scale factor as hole_width / anchor_width.
    - Scale the sprite body (not the anchor) by that factor, align the scaled body's bounding box to the hole's top-left corner, and place it into the hole.
    - Recolor scaled gray cells to blue while keeping the scaled marker-colored cells unchanged; leave the original outside sprites, anchors, and red object in place.

    Key insight:
    - The marker color chooses which sprite fills which hole, while the detached square anchor's size encodes the divisor needed to recover the sprite's scaling factor.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    red_bbox = _largest_red_bbox(g)
    holes = _find_holes(g, red_bbox)
    bodies, anchors = _find_pieces(g, red_bbox)
    pairing = _pair_bodies_to_anchors(bodies, anchors)

    hole_by_color = {color: data for color, data in holes.items()}
    assert set(hole_by_color) == {body["marker_color"] for body in bodies}

    for body in bodies:
        hole = hole_by_color[body["marker_color"]]
        anchor = anchors[pairing[id(body)]]
        scale = hole["width"] // anchor["size"]
        assert hole["width"] % anchor["size"] == 0
        assert scale >= 1
        scaled = scale_sprite(body["patch"], scale)
        assert scaled.shape == (hole["height"], hole["width"])
        scaled = scaled.copy()
        scaled[scaled == GRAY] = BLUE
        r0, c0, r1, c1 = hole["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = scaled

    return out


def _sample_specs(count: int) -> list[dict]:
    widths = [1, 2] if count == 2 else [1, 2, random.choice([1, 2])]
    random.shuffle(widths)
    scales = [2, 3] if count == 2 else [2, 3, random.choice([2, 3])]
    random.shuffle(scales)
    colors = random.sample(MARKER_COLORS, count)

    specs = []
    for i in range(count):
        width = widths[i]
        height = random.choice([3, 4, 5] if width == 1 else [2, 3])
        marker_cells = [(r, c) for r in range(height) for c in range(width)]
        marker_pos = random.choice(marker_cells)
        specs.append(
            {
                "body_h": height,
                "body_w": width,
                "anchor": width,
                "scale": scales[i],
                "marker_color": colors[i],
                "marker_pos": marker_pos,
            }
        )

    if all(spec["marker_pos"] == (0, 0) for spec in specs):
        target = random.choice(specs)
        choices = [(r, c) for r in range(target["body_h"]) for c in range(target["body_w"]) if (r, c) != (0, 0)]
        target["marker_pos"] = random.choice(choices)
    if count >= 2 and not any(spec["marker_pos"] == (0, 0) for spec in specs):
        target = random.choice(specs)
        target["marker_pos"] = (0, 0)
    return specs


def _layout_holes(specs: list[dict]) -> tuple[int, int, list[tuple[int, int]]]:
    rects = [(spec["body_h"] * spec["scale"], spec["body_w"] * spec["scale"]) for spec in specs]
    n = len(rects)
    pattern = random.choice(["row", "col"] if n == 2 else ["row", "col", "stack", "side"])
    gap_a = random.randint(1, 2)
    gap_b = random.randint(1, 2)
    margin_t = random.randint(0, 2)
    margin_b = random.randint(0, 2)
    margin_l = random.randint(0, 2)
    margin_r = random.randint(0, 2)

    if pattern == "row":
        inner_h = margin_t + max(h for h, _ in rects) + margin_b
        inner_w = margin_l + sum(w for _, w in rects) + gap_a * (n - 1) + margin_r
        pos = []
        c = margin_l
        for h, w in rects:
            r = random.randint(margin_t, inner_h - margin_b - h)
            pos.append((r, c))
            c += w + gap_a
    elif pattern == "col":
        inner_h = margin_t + sum(h for h, _ in rects) + gap_a * (n - 1) + margin_b
        inner_w = margin_l + max(w for _, w in rects) + margin_r
        pos = []
        r = margin_t
        for h, w in rects:
            c = random.randint(margin_l, inner_w - margin_r - w)
            pos.append((r, c))
            r += h + gap_a
    elif pattern == "stack":
        top = rects[:2]
        bottom = rects[2]
        top_w = top[0][1] + gap_a + top[1][1]
        top_h = max(top[0][0], top[1][0])
        inner_h = margin_t + top_h + gap_b + bottom[0] + margin_b
        inner_w = margin_l + max(top_w, bottom[1]) + margin_r
        top_start = random.randint(margin_l, inner_w - margin_r - top_w)
        bottom_start = random.randint(margin_l, inner_w - margin_r - bottom[1])
        pos = [
            (random.randint(margin_t, margin_t + top_h - top[0][0]), top_start),
            (random.randint(margin_t, margin_t + top_h - top[1][0]), top_start + top[0][1] + gap_a),
            (margin_t + top_h + gap_b, bottom_start),
        ]
    else:
        left = rects[:2]
        right = rects[2]
        left_h = left[0][0] + gap_a + left[1][0]
        left_w = max(left[0][1], left[1][1])
        inner_h = margin_t + max(left_h, right[0]) + margin_b
        inner_w = margin_l + left_w + gap_b + right[1] + margin_r
        pos = [
            (margin_t, random.randint(margin_l, margin_l + left_w - left[0][1])),
            (margin_t + left[0][0] + gap_a, random.randint(margin_l, margin_l + left_w - left[1][1])),
            (random.randint(margin_t, margin_t + max(left_h, right[0]) - right[0]), margin_l + left_w + gap_b),
        ]

    assert inner_h <= 18 and inner_w <= 18
    return inner_h, inner_w, pos


def _build_body_patch(spec: dict) -> np.ndarray:
    patch = np.full((spec["body_h"], spec["body_w"]), GRAY, dtype=int)
    mr, mc = spec["marker_pos"]
    patch[mr, mc] = spec["marker_color"]
    return patch


def _compound_patch(spec: dict) -> np.ndarray:
    body = _build_body_patch(spec)
    anchor = spec["anchor"]
    side = random.choice(["left", "right", "up", "down"])
    if side == "left":
        offset = random.randint(0, body.shape[0] - anchor)
        canvas = np.full((body.shape[0], anchor + 1 + body.shape[1]), BG, dtype=int)
        canvas[:, anchor + 1 :] = body
        canvas[offset : offset + anchor, :anchor] = GRAY
    elif side == "right":
        offset = random.randint(0, body.shape[0] - anchor)
        canvas = np.full((body.shape[0], body.shape[1] + 1 + anchor), BG, dtype=int)
        canvas[:, : body.shape[1]] = body
        canvas[offset : offset + anchor, body.shape[1] + 1 :] = GRAY
    elif side == "up":
        canvas = np.full((anchor + 1 + body.shape[0], body.shape[1]), BG, dtype=int)
        canvas[:anchor, :] = GRAY
        canvas[anchor + 1 :, :] = body
    else:
        canvas = np.full((body.shape[0] + 1 + anchor, body.shape[1]), BG, dtype=int)
        canvas[: body.shape[0], :] = body
        canvas[body.shape[0] + 1 :, :] = GRAY
    return canvas


def _mark_forbidden(forbidden: np.ndarray, coords: np.ndarray) -> None:
    h, w = forbidden.shape
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = int(r) + dr
                nc = int(c) + dc
                if 0 <= nr < h and 0 <= nc < w:
                    forbidden[nr, nc] = True


def _place_patch(grid: np.ndarray, forbidden: np.ndarray, patch: np.ndarray) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    nz = np.argwhere(patch != BG)
    for _ in range(300):
        r0 = random.randint(0, h - ph)
        c0 = random.randint(0, w - pw)
        placed = nz + np.array([r0, c0], dtype=int)
        if any(forbidden[int(r), int(c)] for r, c in placed):
            continue
        for pr, pc in nz:
            grid[r0 + int(pr), c0 + int(pc)] = int(patch[int(pr), int(pc)])
        _mark_forbidden(forbidden, placed)
        return True
    return False


def _naive_fill_output(inp: np.ndarray) -> np.ndarray:
    out = inp.copy()
    red_bbox = _largest_red_bbox(inp)
    holes = _find_holes(inp, red_bbox)
    for hole in holes.values():
        r0, c0, r1, c1 = hole["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BLUE
        out[r0, c0] = hole["cue_color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    count = random.choice([2, 3, 3])
    specs = _sample_specs(count)
    try:
        inner_h, inner_w, hole_positions = _layout_holes(specs)
    except Exception:
        return None

    red_h = inner_h + 2
    red_w = inner_w + 2
    if red_h > 22 or red_w > 22:
        return None

    extra_h = 30 - red_h
    extra_w = 30 - red_w
    if extra_h < 8 or extra_w < 8:
        return None

    top = random.randint(3, min(6, extra_h - 3))
    left = random.randint(3, min(6, extra_w - 3))
    bottom = random.randint(3, min(6, extra_h - top))
    right = random.randint(3, min(6, extra_w - left))
    H = red_h + top + bottom
    W = red_w + left + right
    if H > 30 or W > 30:
        return None

    grid = np.full((H, W), BG, dtype=int)
    red_r0 = top
    red_c0 = left
    grid[red_r0 : red_r0 + red_h, red_c0 : red_c0 + red_w] = RED

    for spec, (hr, hc) in zip(specs, hole_positions):
        hole_h = spec["body_h"] * spec["scale"]
        hole_w = spec["body_w"] * spec["scale"]
        r0 = red_r0 + 1 + hr
        c0 = red_c0 + 1 + hc
        grid[r0 : r0 + hole_h, c0 : c0 + hole_w] = BG
        grid[r0, c0] = spec["marker_color"]

    forbidden = np.zeros_like(grid, dtype=bool)
    red_coords = np.argwhere(grid == RED)
    _mark_forbidden(forbidden, red_coords)

    for spec in specs:
        if not _place_patch(grid, forbidden, _compound_patch(spec)):
            return None

    try:
        solved = main(grid)
        if np.array_equal(solved, _naive_fill_output(grid)):
            return None
        red_bbox = _largest_red_bbox(grid)
        holes = _find_holes(grid, red_bbox)
        bodies, anchors = _find_pieces(grid, red_bbox)
        if len(holes) != count or len(bodies) != count or len(anchors) != count:
            return None
        if not any(body["marker_pos"] != (0, 0) for body in bodies):
            return None
        for body in bodies:
            if np.any(body["patch"] == BG):
                return None
            hole = holes[body["marker_color"]]
            anchor_idx = _anchor_candidates(body["bbox"], anchors)
            if len(anchor_idx) != 1:
                return None
            anchor = anchors[anchor_idx[0]]
            scale = hole["width"] // anchor["size"]
            if hole["width"] % anchor["size"] != 0 or scale not in (2, 3):
                return None
            if body["height"] * scale != hole["height"] or body["width"] * scale != hole["width"]:
                return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
