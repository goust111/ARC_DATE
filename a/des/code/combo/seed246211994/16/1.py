from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


SIDES = ("top", "bottom", "left", "right")
CONTACT_EDGE = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def label_mask(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
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
    cid = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            cid += 1
            labels[r, c] = cid
            q: deque[tuple[int, int]] = deque([(r, c)])
            while q:
                rr, cc = q.popleft()
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = cid
                        q.append((nr, nc))
    return labels, cid


def crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return arr[r0 : r1 + 1, c0 : c1 + 1], mask[r0 : r1 + 1, c0 : c1 + 1]


def paste(
    dst: np.ndarray,
    src: np.ndarray,
    top: int,
    left: int,
    *,
    mask: np.ndarray | None = None,
    transparent: int | None = None,
) -> None:
    h, w = src.shape
    sub = dst[top : top + h, left : left + w]
    if mask is not None:
        sub[mask] = src[mask]
    elif transparent is not None:
        m = src != int(transparent)
        sub[m] = src[m]
    else:
        sub[:] = src


def _is_contiguous(values: list[int]) -> bool:
    return bool(values) and all(values[i] + 1 == values[i + 1] for i in range(len(values) - 1))


def _edge_segment(mask: np.ndarray, edge: str) -> tuple[str, int, list[int]] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    if edge == "top":
        row = int(pts[:, 0].min())
        cols = sorted(int(v) for v in pts[pts[:, 0] == row][:, 1])
        return ("h", row, cols)
    if edge == "bottom":
        row = int(pts[:, 0].max())
        cols = sorted(int(v) for v in pts[pts[:, 0] == row][:, 1])
        return ("h", row, cols)
    if edge == "left":
        col = int(pts[:, 1].min())
        rows = sorted(int(v) for v in pts[pts[:, 1] == col][:, 0])
        return ("v", col, rows)
    if edge == "right":
        col = int(pts[:, 1].max())
        rows = sorted(int(v) for v in pts[pts[:, 1] == col][:, 0])
        return ("v", col, rows)
    return None


def _valid_attachment_orientation(mask: np.ndarray, side: str, edge_len: int) -> bool:
    if side in ("left", "right"):
        if not np.array_equal(mask, np.flipud(mask)):
            return False
        seg = _edge_segment(mask, CONTACT_EDGE[side])
        if seg is None:
            return False
        _, _edge_col, rows = seg
        return len(rows) == edge_len and _is_contiguous(rows)
    if not np.array_equal(mask, np.fliplr(mask)):
        return False
    seg = _edge_segment(mask, CONTACT_EDGE[side])
    if seg is None:
        return False
    _, _edge_row, cols = seg
    return len(cols) == edge_len and _is_contiguous(cols)


def _valid_rotations(mask: np.ndarray, side: str, edge_len: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(mask, k)
        rot = crop_to_mask(rot.astype(int), rot)[1]
        if _valid_attachment_orientation(rot, side, edge_len):
            out.append(rot)
    return out


def _attachment_geometry(side: str, hub_r: int, hub_c: int, n: int) -> tuple[str, int, list[int]]:
    if side == "top":
        return "bottom", hub_r - 1, list(range(hub_c + 1, hub_c + n - 1))
    if side == "bottom":
        return "top", hub_r + n, list(range(hub_c + 1, hub_c + n - 1))
    if side == "left":
        return "right", hub_c - 1, list(range(hub_r + 1, hub_r + n - 1))
    return "left", hub_c + n, list(range(hub_r + 1, hub_r + n - 1))


def _placement_for_side(mask: np.ndarray, side: str, hub_r: int, hub_c: int, n: int) -> tuple[int, int]:
    need_edge, fixed, span = _attachment_geometry(side, hub_r, hub_c, n)
    seg = _edge_segment(mask, need_edge)
    assert seg is not None
    if side in ("top", "bottom"):
        _, edge_row, cols = seg
        return int(fixed - edge_row), int(span[0] - cols[0])
    _, edge_col, rows = seg
    return int(span[0] - rows[0]), int(fixed - edge_col)


def _find_hub(grid: np.ndarray, bg: int) -> tuple[int, int, int, dict[str, int]] | None:
    h, w = grid.shape
    best = None
    for n in range(3, min(h, w) + 1):
        for r in range(h - n + 1):
            for c in range(w - n + 1):
                if (
                    int(grid[r, c]) != bg
                    or int(grid[r, c + n - 1]) != bg
                    or int(grid[r + n - 1, c]) != bg
                    or int(grid[r + n - 1, c + n - 1]) != bg
                ):
                    continue
                inner = grid[r + 1 : r + n - 1, c + 1 : c + n - 1]
                if inner.size and not np.all(inner == bg):
                    continue
                top = grid[r, c + 1 : c + n - 1]
                bottom = grid[r + n - 1, c + 1 : c + n - 1]
                left = grid[r + 1 : r + n - 1, c]
                right = grid[r + 1 : r + n - 1, c + n - 1]
                if min(top.size, bottom.size, left.size, right.size) == 0:
                    continue
                if any(np.any(side == bg) for side in (top, bottom, left, right)):
                    continue
                if any(len(set(map(int, side))) != 1 for side in (top, bottom, left, right)):
                    continue
                colors = {
                    "top": int(top[0]),
                    "bottom": int(bottom[0]),
                    "left": int(left[0]),
                    "right": int(right[0]),
                }
                if len(set(colors.values())) != 4:
                    continue
                hub_mask = np.zeros_like(grid, dtype=bool)
                hub_mask[r : r + n, c : c + n] = True
                matches = sum(bool(((grid == col) & (~hub_mask)).any()) for col in colors.values())
                center_dist = abs((r + (n - 1) / 2) - (h - 1) / 2) + abs((c + (n - 1) / 2) - (w - 1) / 2)
                score = (matches, n, -float(center_dist))
                if best is None or score > best[0]:
                    best = (score, (r, c, n, colors))
    return None if best is None else best[1]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    hub = _find_hub(g, bg)
    assert hub is not None
    hub_r, hub_c, n, colors = hub

    out = np.full((h, w), bg, dtype=int)
    out[hub_r : hub_r + n, hub_c : hub_c + n] = g[hub_r : hub_r + n, hub_c : hub_c + n]
    hub_mask = np.zeros((h, w), dtype=bool)
    hub_mask[hub_r : hub_r + n, hub_c : hub_c + n] = True
    edge_len = n - 2

    for side in SIDES:
        col = colors[side]
        color_mask = (g == col) & (~hub_mask)
        if not color_mask.any():
            continue
        labels, count = label_mask(color_mask, connectivity=8)
        comps: list[tuple[int, int, int, int, int]] = []
        for cid in range(1, count + 1):
            pts = np.argwhere(labels == cid)
            if pts.size == 0:
                continue
            r0, c0, r1, c1 = bbox_coords(pts)
            comps.append((int(pts.shape[0]), r0, c0, c1, cid))
        comps.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))

        placed = False
        for _area, _r0, _c0, _c1, cid in comps:
            sprite_mask = labels == cid
            sprite_arr = np.full((h, w), bg, dtype=int)
            sprite_arr[sprite_mask] = col
            sprite_arr, sprite_mask = crop_to_mask(sprite_arr, sprite_mask)
            for k in range(4):
                arr = np.rot90(sprite_arr, k)
                mask = np.rot90(sprite_mask, k)
                arr, mask = crop_to_mask(arr, mask)
                if not _valid_attachment_orientation(mask, side, edge_len):
                    continue
                top, left = _placement_for_side(mask, side, hub_r, hub_c, n)
                if top < 0 or left < 0 or top + arr.shape[0] > h or left + arr.shape[1] > w:
                    continue
                sub = out[top : top + arr.shape[0], left : left + arr.shape[1]]
                vals = arr[mask]
                existing = sub[mask]
                if np.any((existing != bg) & (existing != vals)):
                    continue
                sub[mask] = vals
                placed = True
                break
            if placed:
                break
    return out


def _transform_left_mask(mask: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        out = mask.copy()
    elif side == "right":
        out = np.fliplr(mask)
    elif side == "top":
        out = np.rot90(mask, -1)
    else:
        out = np.rot90(mask, 1)
    return crop_to_mask(out.astype(int), out)[1]


def _make_left_sprite_mask(edge_len: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(400):
        pad = rng.randint(0, 2)
        h = edge_len + 2 * pad
        w = rng.randint(2, min(6, edge_len + 3))
        grid = np.zeros((h, w), dtype=bool)
        run_start = pad
        run_stop = pad + edge_len
        has_extra = False

        for r in range((h + 1) // 2):
            rr = h - 1 - r
            in_run = run_start <= r < run_stop
            end = w - 1 if in_run else w - 2
            if end < 0:
                break
            start = rng.randint(0, end)
            grid[r, start : end + 1] = True
            grid[rr, start : end + 1] = True
            if start < end or not in_run:
                has_extra = True

        if not has_extra:
            continue
        rows_any = grid.any(axis=1)
        cols_any = grid.any(axis=0)
        if not rows_any.any() or not cols_any.any():
            continue
        grid = grid[np.ix_(rows_any, cols_any)]
        if grid.shape[0] < 2 or grid.shape[1] < 2:
            continue
        if int(grid.sum()) < edge_len + 1 or int(grid.sum()) > 18:
            continue
        if not np.array_equal(grid, np.flipud(grid)):
            continue
        if np.array_equal(grid, np.fliplr(grid)):
            continue
        _, count = label_mask(grid, connectivity=8)
        if count != 1:
            continue
        if len(_valid_rotations(grid, "left", edge_len)) != 1:
            continue
        return grid
    return None


def _make_sprite_mask(edge_len: int, side: str, rng: random.Random) -> np.ndarray | None:
    for _ in range(200):
        base = _make_left_sprite_mask(edge_len, rng)
        if base is None:
            continue
        mask = _transform_left_mask(base, side)
        if len(_valid_rotations(mask, side, edge_len)) == 1:
            return mask
    return None


def _build_hub(
    grid: np.ndarray,
    hub_r: int,
    hub_c: int,
    n: int,
    colors: dict[str, int],
) -> None:
    grid[hub_r, hub_c + 1 : hub_c + n - 1] = colors["top"]
    grid[hub_r + n - 1, hub_c + 1 : hub_c + n - 1] = colors["bottom"]
    grid[hub_r + 1 : hub_r + n - 1, hub_c] = colors["left"]
    grid[hub_r + 1 : hub_r + n - 1, hub_c + n - 1] = colors["right"]


def _boxes_are_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + padding < br0
        or br1 + padding < ar0
        or ac1 + padding < bc0
        or bc1 + padding < ac0
    )


def _find_free_location(
    canvas: np.ndarray,
    patch: np.ndarray,
    bg: int,
    boxes: list[tuple[int, int, int, int]],
    rng: random.Random,
    *,
    padding: int = 2,
) -> tuple[int, int] | None:
    ph, pw = patch.shape
    h, w = canvas.shape
    candidates: list[tuple[float, int, int]] = []
    for top in range(h - ph + 1):
        for left in range(w - pw + 1):
            bbox = (top, left, top + ph - 1, left + pw - 1)
            if not all(_boxes_are_separate(bbox, other, padding) for other in boxes):
                continue
            sub = canvas[top : top + ph, left : left + pw]
            m = patch != bg
            if np.any(sub[m] != bg):
                continue
            center_bias = abs((top + (ph - 1) / 2) - (h - 1) / 2) + abs((left + (pw - 1) / 2) - (w - 1) / 2)
            candidates.append((center_bias, top, left))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    best_band = candidates[: max(1, min(len(candidates), 24))]
    _score, top, left = rng.choice(best_band)
    return int(top), int(left)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = rng.randint(0, 9)
    remaining = [c for c in range(10) if c != bg]
    rng.shuffle(remaining)

    n = rng.randint(4, 6)
    edge_len = n - 2
    side_colors = dict(zip(SIDES, remaining[:4], strict=True))
    sprite_count = rng.randint(2, 4)
    sprite_sides = rng.sample(list(SIDES), sprite_count)

    sprite_masks: dict[str, np.ndarray] = {}
    for side in sprite_sides:
        mask = _make_sprite_mask(edge_len, side, rng)
        if mask is None:
            return None
        sprite_masks[side] = mask

    max_extent = max(max(mask.shape) for mask in sprite_masks.values())
    margin = max_extent + 2
    min_h = max(18, n + 2 * margin + 4)
    min_w = max(18, n + 2 * margin + 4)
    max_h = max(min_h + 4, 28)
    max_w = max(min_w + 4, 28)
    h = rng.randint(min_h, max_h)
    w = rng.randint(min_w, max_w)

    if h - n - 2 * margin < 0 or w - n - 2 * margin < 0:
        return None
    hub_r = rng.randint(margin, h - n - margin)
    hub_c = rng.randint(margin, w - n - margin)

    out = np.full((h, w), bg, dtype=int)
    _build_hub(out, hub_r, hub_c, n, side_colors)
    boxes = [(hub_r, hub_c, hub_r + n - 1, hub_c + n - 1)]

    attached: dict[str, np.ndarray] = {}
    for side in sprite_sides:
        color = side_colors[side]
        mask = sprite_masks[side]
        patch = np.where(mask, color, bg).astype(int)
        top, left = _placement_for_side(mask, side, hub_r, hub_c, n)
        if top < 0 or left < 0 or top + patch.shape[0] > h or left + patch.shape[1] > w:
            return None
        sub = out[top : top + patch.shape[0], left : left + patch.shape[1]]
        if np.any((patch != bg) & (sub != bg)):
            return None
        paste(out, patch, top, left, transparent=bg)
        boxes.append((top, left, top + patch.shape[0] - 1, left + patch.shape[1] - 1))
        attached[side] = patch

    inp = np.full((h, w), bg, dtype=int)
    _build_hub(inp, hub_r, hub_c, n, side_colors)
    canvas = inp.copy()
    placed_boxes = [(hub_r, hub_c, hub_r + n - 1, hub_c + n - 1)]

    order = list(sprite_sides)
    rng.shuffle(order)
    for side in order:
        patch = attached[side]
        k = rng.randint(0, 3)
        rot = np.rot90(patch, k)
        rot, rot_mask = crop_to_mask(rot, rot != bg)
        patch = np.where(rot_mask, rot, bg).astype(int)
        loc = _find_free_location(canvas, patch, bg, placed_boxes, rng, padding=2)
        if loc is None:
            return None
        top, left = loc
        paste(canvas, patch, top, left, transparent=bg)
        paste(inp, patch, top, left, transparent=bg)
        placed_boxes.append((top, left, top + patch.shape[0] - 1, left + patch.shape[1] - 1))

    free_colors = [c for c in remaining[4:] if c not in side_colors.values()]
    distractor_count = 0
    if free_colors:
        distractor_count = rng.choices([0, 1, 2], weights=[4, 3, 1])[0]
    for _ in range(min(distractor_count, len(free_colors))):
        dcolor = free_colors.pop()
        dside = rng.choice(SIDES)
        dmask = _make_sprite_mask(edge_len, dside, rng)
        if dmask is None:
            continue
        drot = np.rot90(np.where(dmask, dcolor, bg).astype(int), rng.randint(0, 3))
        drot, dmask2 = crop_to_mask(drot, drot != bg)
        dpatch = np.where(dmask2, drot, bg).astype(int)
        loc = _find_free_location(canvas, dpatch, bg, placed_boxes, rng, padding=2)
        if loc is None:
            continue
        top, left = loc
        paste(canvas, dpatch, top, left, transparent=bg)
        paste(inp, dpatch, top, left, transparent=bg)
        placed_boxes.append((top, left, top + dpatch.shape[0] - 1, left + dpatch.shape[1] - 1))

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 500 attempts")
