import random
from collections import deque

import numpy as np


BG = 0
RING = 5
PALETTE_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400
PLACE_RETRIES = 200


def _components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Label 4-connected True components and return their coordinates."""
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue
            cid = len(comps)
            q = deque([(r, c)])
            labels[r, c] = cid
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = cid
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return labels, comps


def _read_palette(strip: np.ndarray, bg: int) -> list[int]:
    colors: list[int] = []
    prev = None
    for value in strip:
        color = int(value)
        if color != prev:
            if color != bg:
                colors.append(color)
            prev = color
    assert colors, "palette strip must contain at least one non-background color"
    return colors


def _body_bg(body: np.ndarray) -> int:
    border = np.concatenate((body[0], body[-1], body[:, 0], body[:, -1]))
    counts = np.bincount(border.astype(int), minlength=10)
    return int(np.argmax(counts))


def _solve(body: np.ndarray, palette: list[int]) -> np.ndarray:
    body = np.asarray(body, dtype=int)
    bg = _body_bg(body)
    fg_colors = [int(c) for c in np.unique(body) if int(c) != bg]
    assert len(fg_colors) == 1, "body must contain exactly one ring color"
    ring_color = fg_colors[0]

    bg_mask = body == bg
    ring_mask = body == ring_color
    bg_labels, bg_comps = _components(bg_mask)
    ring_labels, ring_comps = _components(ring_mask)
    assert ring_comps, "body must contain at least one ring"

    boundary_ids = {
        int(bg_labels[r, c])
        for r in range(body.shape[0])
        for c in range(body.shape[1])
        if bg_mask[r, c] and (r == 0 or c == 0 or r == body.shape[0] - 1 or c == body.shape[1] - 1)
    }
    assert boundary_ids, "outer background must touch the body border"

    ring_sides: list[tuple[int, int]] = []
    bg_to_rings = [[] for _ in bg_comps]
    for rid, pts in enumerate(ring_comps):
        touching = set()
        for r, c in pts:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < body.shape[0] and 0 <= nc < body.shape[1] and bg_mask[nr, nc]:
                    touching.add(int(bg_labels[nr, nc]))
        assert len(touching) == 2, "each ring must separate exactly two background regions"
        a, b = tuple(touching)
        ring_sides.append((a, b))
        bg_to_rings[a].append(rid)
        bg_to_rings[b].append(rid)

    depths = [-1] * len(bg_comps)
    q = deque()
    for cid in boundary_ids:
        if depths[cid] == -1:
            depths[cid] = 0
            q.append(cid)

    while q:
        cid = q.popleft()
        for rid in bg_to_rings[cid]:
            a, b = ring_sides[rid]
            other = b if a == cid else a
            if depths[other] == -1:
                depths[other] = depths[cid] + 1
                q.append(other)

    assert all(depth >= 0 for depth in depths), "all background regions must be reachable from outside"

    ring_depths = []
    for a, b in ring_sides:
        da, db = depths[a], depths[b]
        assert abs(da - db) == 1, "ring must separate adjacent depth levels"
        ring_depths.append(max(da, db))

    max_depth = max(ring_depths)
    assert max_depth <= len(palette), "palette strip is shorter than the nesting depth"

    out = np.full(body.shape, bg, dtype=int)
    for cid, pts in enumerate(bg_comps):
        depth = depths[cid]
        if depth == 0:
            continue
        out[pts[:, 0], pts[:, 1]] = int(palette[depth - 1])
    for rid, pts in enumerate(ring_comps):
        out[pts[:, 0], pts[:, 1]] = int(palette[ring_depths[rid] - 1])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested grey rings using the top strip as a depth legend.

    The first row is a left-to-right palette strip. The rows below contain
    closed rings of a single neutral color on a black background. Each ring's
    nesting depth determines its replacement color, and the background band
    owned by that depth is filled with the same color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2
    strip = g[0]
    body = g[1:]
    bg = _body_bg(body)
    palette = _read_palette(strip, bg)
    return _solve(body, palette)


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _make_stack(depth: int, rng: random.Random) -> dict:
    inner_h = rng.randint(3, 6)
    inner_w = rng.randint(3, 7)
    boxes = [(0, 0, inner_h - 1, inner_w - 1)]
    for _ in range(depth - 1):
        r0, c0, r1, c1 = boxes[0]
        mt = rng.randint(2, 3)
        mb = rng.randint(2, 3)
        ml = rng.randint(2, 3)
        mr = rng.randint(2, 3)
        boxes.insert(0, (r0 - mt, c0 - ml, r1 + mb, c1 + mr))

    min_r = min(r0 for r0, _, _, _ in boxes)
    min_c = min(c0 for _, c0, _, _ in boxes)
    shifted = [
        (r0 - min_r, c0 - min_c, r1 - min_r, c1 - min_c)
        for r0, c0, r1, c1 in boxes
    ]
    h = max(r1 for _, _, r1, _ in shifted) + 1
    w = max(c1 for _, _, _, c1 in shifted) + 1
    return {"boxes": shifted, "depth": depth, "h": h, "w": w}


def _bbox_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _place_stacks(stacks: list[dict], rng: random.Random) -> tuple[list[dict], int, int] | None:
    row_span = rng.randint(3, 6 + 5 * max(0, len(stacks) - 1))
    col_span = rng.randint(5, 8 + 7 * max(0, len(stacks) - 1))
    if rng.random() < 0.5:
        row_span, col_span = col_span, row_span

    placed = []
    order = sorted(stacks, key=lambda s: s["h"] * s["w"], reverse=True)
    for stack in order:
        done = False
        for _ in range(PLACE_RETRIES):
            top = rng.randint(0, row_span)
            left = rng.randint(0, col_span)
            outer = (top, left, top + stack["h"] - 1, left + stack["w"] - 1)
            if any(_bbox_overlaps(outer, item["outer"], pad=1) for item in placed):
                continue
            shifted_boxes = [
                (top + r0, left + c0, top + r1, left + c1)
                for r0, c0, r1, c1 in stack["boxes"]
            ]
            placed.append({"outer": outer, "boxes": shifted_boxes, "depth": stack["depth"]})
            done = True
            break
        if not done:
            return None

    min_r = min(item["outer"][0] for item in placed)
    min_c = min(item["outer"][1] for item in placed)
    max_r = max(item["outer"][2] for item in placed)
    max_c = max(item["outer"][3] for item in placed)
    top_margin = rng.randint(1, 4)
    bottom_margin = rng.randint(1, 4)
    left_margin = rng.randint(1, 4)
    right_margin = rng.randint(1, 4)
    shift_r = top_margin - min_r
    shift_c = left_margin - min_c

    final = []
    for item in placed:
        final.append(
            {
                "depth": item["depth"],
                "boxes": [
                    (r0 + shift_r, c0 + shift_c, r1 + shift_r, c1 + shift_c)
                    for r0, c0, r1, c1 in item["boxes"]
                ],
            }
        )

    body_h = max_r - min_r + 1 + top_margin + bottom_margin
    body_w = max_c - min_c + 1 + left_margin + right_margin
    return final, body_h, body_w


def _segment_widths(total: int, count: int, rng: random.Random) -> list[int]:
    min_width = 2
    widths = [min_width] * count
    extra = total - min_width * count
    assert extra >= 0
    for _ in range(extra):
        widths[rng.randrange(count)] += 1
    return widths


def _build_strip(width: int, palette: list[int], rng: random.Random) -> np.ndarray:
    widths = _segment_widths(width, len(palette), rng)
    strip = np.zeros(width, dtype=int)
    c0 = 0
    for color, seg_w in zip(palette, widths):
        strip[c0 : c0 + seg_w] = int(color)
        c0 += seg_w
    return strip


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_stacks = rng.randint(1, 3)
    depth_cap = {1: 5, 2: 4, 3: 3}[num_stacks]
    max_depth = rng.randint(2, depth_cap)
    depths = [max_depth]
    for _ in range(num_stacks - 1):
        depths.append(rng.randint(1, max_depth))
    rng.shuffle(depths)

    stacks = [_make_stack(depth, rng) for depth in depths]
    placed = _place_stacks(stacks, rng)
    if placed is None:
        return None
    placed_stacks, body_h, body_w = placed
    if body_w < 2 * max_depth or body_h < 5:
        return None

    body = np.full((body_h, body_w), BG, dtype=int)
    occupied = np.zeros_like(body, dtype=bool)
    for stack in placed_stacks:
        for bbox in stack["boxes"]:
            mask = _frame_mask(body.shape, bbox)
            if np.any(occupied & mask):
                return None
            occupied |= mask
            body[mask] = RING

    palette = rng.sample(PALETTE_POOL, k=max_depth)
    strip = _build_strip(body_w, palette, rng)
    grid = np.vstack([strip[np.newaxis, :], body])

    out = main(grid)
    if out.shape != body.shape:
        return None
    if np.any(out == RING):
        return None
    used = {int(v) for v in np.unique(out) if int(v) != BG}
    if used != set(palette[:max_depth]):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
