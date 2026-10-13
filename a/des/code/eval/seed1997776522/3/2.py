from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
FRAME = 5
MARKER = 9
SIDES = ("top", "bottom", "left", "right")
BODY_COLORS = (1, 2, 3, 4, 6, 7, 8)
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _c in coords]
    cols = [int(c) for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(grid == FRAME)
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert r1 > r0 and c1 > c0 and (r1 - r0) == (c1 - c0)
    return r0, c0, r1, c1


def _frame_border_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = box
    mask = np.zeros((h, w), dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _local_cells_from_component(grid: np.ndarray, comp: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    r0, c0, _r1, _c1 = _bbox(comp)
    return [(int(r - r0), int(c - c0), int(grid[r, c])) for r, c in comp]


def _paint_piece(out: np.ndarray, cells: list[tuple[int, int, int]], side: str) -> None:
    size = int(out.shape[0])
    rows = [int(r) for r, _c, _v in cells]
    cols = [int(c) for _r, c, _v in cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    height = int(max_r - min_r + 1)
    width = int(max_c - min_c + 1)

    if side == "top":
        dr = int(size - 1 - max_r)
        dc = int((size - width) // 2 - min_c)
    elif side == "bottom":
        dr = int(-min_r)
        dc = int((size - width) // 2 - min_c)
    elif side == "left":
        dr = int((size - height) // 2 - min_r)
        dc = int(size - 1 - max_c)
    else:
        assert side == "right"
        dr = int((size - height) // 2 - min_r)
        dc = int(-min_c)

    for r, c, value in cells:
        if int(value) == MARKER:
            continue
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < size and 0 <= cc < size:
            out[rr, cc] = int(value)


def _piece_render_coords(patch: np.ndarray, side: str, size: int) -> list[tuple[int, int, int]]:
    coords = np.argwhere(patch != BG)
    cells = [(int(r), int(c), int(patch[r, c])) for r, c in coords]
    rows = [int(r) for r, _c, _v in cells]
    cols = [int(c) for _r, c, _v in cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    height = int(max_r - min_r + 1)
    width = int(max_c - min_c + 1)

    if side == "top":
        dr = int(size - 1 - max_r)
        dc = int((size - width) // 2 - min_c)
    elif side == "bottom":
        dr = int(-min_r)
        dc = int((size - width) // 2 - min_c)
    elif side == "left":
        dr = int((size - height) // 2 - min_r)
        dc = int(size - 1 - max_c)
    else:
        dr = int((size - height) // 2 - min_r)
        dc = int(-min_c)

    placed: list[tuple[int, int, int]] = []
    for r, c, value in cells:
        if int(value) == MARKER:
            continue
        placed.append((int(r + dr), int(c + dc), int(value)))
    return placed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Layer scattered colored fragments onto the central canvas by marker-count order.

    - The grey square frame marks the canvas; only its black interior is returned.
    - Each fragment is a connected non-black component outside the frame border.
    - The fragment's body color matches exactly one colored midpoint on the frame border.
    - Fewer maroon marker pixels means an earlier layer; later layers overwrite earlier ones.
    """
    g = np.asarray(grid, dtype=int)
    frame_box = _frame_bbox(g)
    r0, c0, r1, c1 = frame_box
    canvas_size = int(r1 - r0 - 1)
    mid_r = int((r0 + r1) // 2)
    mid_c = int((c0 + c1) // 2)

    anchors = {
        "top": (int(r0), int(mid_c)),
        "bottom": (int(r1), int(mid_c)),
        "left": (int(mid_r), int(c0)),
        "right": (int(mid_r), int(c1)),
    }
    color_to_side = {
        int(g[r, c]): side
        for side, (r, c) in anchors.items()
        if int(g[r, c]) not in (BG, FRAME)
    }

    fg = g != BG
    fg[_frame_border_mask(g.shape, frame_box)] = False

    pieces: list[tuple[int, int, str, list[tuple[int, int, int]]]] = []
    for comp in _components(fg):
        values = [int(g[r, c]) for r, c in comp]
        marker_count = sum(int(v == MARKER) for v in values)
        body_colors = sorted(set(values) - {MARKER})
        if marker_count == 0 or len(body_colors) != 1:
            continue
        body = int(body_colors[0])
        if body not in color_to_side:
            continue
        pieces.append((int(marker_count), int(body), color_to_side[body], _local_cells_from_component(g, comp)))

    pieces.sort(key=lambda item: (int(item[0]), int(item[1])))
    out = np.full((canvas_size, canvas_size), BG, dtype=int)
    for _count, _body, side, cells in pieces:
        _paint_piece(out, cells, side)
    return out


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    start = tuple(map(int, pts[0]))
    q = deque([start])
    seen = {start}
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _boundary_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = int(rr + dr)
            nc = int(cc + dc)
            if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                cells.append((rr, cc))
                break
    return cells


def _interior_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if rr in (0, h - 1) or cc in (0, w - 1):
            continue
        if mask[rr - 1, cc] and mask[rr + 1, cc] and mask[rr, cc - 1] and mask[rr, cc + 1]:
            cells.append((rr, cc))
    return cells


def _build_piece_patch(height: int, width: int, marker_count: int, color: int) -> np.ndarray | None:
    min_area = int(marker_count + 6)
    for _ in range(120):
        mask = np.ones((height, width), dtype=bool)
        max_removals = min(int(height * width - min_area), int((2 * (height + width) - 4) // 2))
        if max_removals < 1:
            continue
        target_removals = random.randint(1, max_removals)
        removed = 0
        for _step in range(target_removals * 3):
            candidates = _boundary_cells(mask)
            random.shuffle(candidates)
            changed = False
            for r, c in candidates:
                trial = mask.copy()
                trial[int(r), int(c)] = False
                if not (trial[0, :].any() and trial[-1, :].any() and trial[:, 0].any() and trial[:, -1].any()):
                    continue
                if int(trial.sum()) < min_area:
                    continue
                if not _mask_connected(trial):
                    continue
                if len(_interior_cells(trial)) < marker_count:
                    continue
                mask = trial
                removed += 1
                changed = True
                break
            if not changed:
                break
        if removed == 0:
            continue
        interior = _interior_cells(mask)
        if len(interior) < marker_count:
            continue
        patch = np.full((height, width), BG, dtype=int)
        patch[mask] = int(color)
        for r, c in random.sample(interior, marker_count):
            patch[int(r), int(c)] = MARKER
        return patch
    return None


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        int(ar1 + pad) < int(br0)
        or int(br1 + pad) < int(ar0)
        or int(ac1 + pad) < int(bc0)
        or int(bc1 + pad) < int(ac0)
    )


def _place_patch_in_zone(
    grid: np.ndarray,
    patch: np.ndarray,
    zone: str,
    frame_box: tuple[int, int, int, int],
    used_boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    h, w = patch.shape
    H, W = grid.shape
    fr0, fc0, fr1, fc1 = frame_box

    if zone == "top":
        r_lo, r_hi = 1, int(fr0 - h - 2)
        c_lo, c_hi = 1, int(W - w - 2)
    elif zone == "bottom":
        r_lo, r_hi = int(fr1 + 2), int(H - h - 2)
        c_lo, c_hi = 1, int(W - w - 2)
    elif zone == "left":
        r_lo, r_hi = 1, int(H - h - 2)
        c_lo, c_hi = 1, int(fc0 - w - 2)
    else:
        assert zone == "right"
        r_lo, r_hi = 1, int(H - h - 2)
        c_lo, c_hi = int(fc1 + 2), int(W - w - 2)

    if r_lo > r_hi or c_lo > c_hi:
        return None

    for _ in range(150):
        r0 = random.randint(r_lo, r_hi)
        c0 = random.randint(c_lo, c_hi)
        box = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
        if any(_boxes_intersect(box, other, pad=1) for other in used_boxes):
            continue
        grid[r0 : r0 + h, c0 : c0 + w][patch != BG] = patch[patch != BG]
        used_boxes.append(box)
        return int(r0), int(c0)
    return None


def _render_expected(inner_size: int, pieces: list[dict]) -> tuple[np.ndarray, int]:
    out = np.full((inner_size, inner_size), BG, dtype=int)
    overlap = 0
    ordered = sorted(pieces, key=lambda piece: int(piece["marker_count"]))
    for piece in ordered:
        for r, c, value in _piece_render_coords(piece["patch"], piece["anchor_side"], inner_size):
            if int(out[r, c]) != BG:
                overlap += 1
            out[r, c] = int(value)
    return out, int(overlap)


def _gen_attempt() -> np.ndarray | None:
    inner_size = random.choice((7, 9, 11, 13))
    colors = random.sample(BODY_COLORS, 4)
    anchor_order = list(SIDES)
    placement_order = list(SIDES)
    random.shuffle(anchor_order)
    random.shuffle(placement_order)

    min_dim = max(4, inner_size // 2)
    dim_choices = [
        (h, w)
        for h in range(min_dim, inner_size + 1)
        for w in range(min_dim, inner_size + 1)
        if (h - 2) * (w - 2) >= 4
    ]
    random.shuffle(dim_choices)

    pieces: list[dict] = []
    for idx, marker_count in enumerate((1, 2, 3, 4)):
        patch = None
        choices = dim_choices[:]
        random.shuffle(choices)
        for h, w in choices:
            patch = _build_piece_patch(int(h), int(w), int(marker_count), int(colors[idx]))
            if patch is not None:
                break
        if patch is None:
            return None
        pieces.append(
            {
                "marker_count": int(marker_count),
                "color": int(colors[idx]),
                "anchor_side": anchor_order[idx],
                "placement_zone": placement_order[idx],
                "patch": patch,
            }
        )

    zone_to_piece = {str(piece["placement_zone"]): piece for piece in pieces}
    top_h = int(zone_to_piece["top"]["patch"].shape[0])
    bottom_h = int(zone_to_piece["bottom"]["patch"].shape[0])
    left_w = int(zone_to_piece["left"]["patch"].shape[1])
    right_w = int(zone_to_piece["right"]["patch"].shape[1])

    outer = int(inner_size + 2)
    pad_h = int(max(top_h, bottom_h) + random.randint(3, 6))
    pad_w = int(max(left_w, right_w) + random.randint(3, 6))
    H = int(outer + 2 * pad_h)
    W = int(outer + 2 * pad_w)
    fr0 = int(pad_h)
    fc0 = int(pad_w)
    fr1 = int(fr0 + outer - 1)
    fc1 = int(fc0 + outer - 1)
    frame_box = (fr0, fc0, fr1, fc1)

    grid = np.full((H, W), BG, dtype=int)
    grid[fr0, fc0 : fc1 + 1] = FRAME
    grid[fr1, fc0 : fc1 + 1] = FRAME
    grid[fr0 : fr1 + 1, fc0] = FRAME
    grid[fr0 : fr1 + 1, fc1] = FRAME

    mid_r = int((fr0 + fr1) // 2)
    mid_c = int((fc0 + fc1) // 2)
    side_pos = {
        "top": (int(fr0), int(mid_c)),
        "bottom": (int(fr1), int(mid_c)),
        "left": (int(mid_r), int(fc0)),
        "right": (int(mid_r), int(fc1)),
    }
    for piece in pieces:
        rr, cc = side_pos[str(piece["anchor_side"])]
        grid[rr, cc] = int(piece["color"])

    used_boxes = [frame_box]
    placement_items = pieces[:]
    random.shuffle(placement_items)
    for piece in placement_items:
        placed = _place_patch_in_zone(grid, piece["patch"], str(piece["placement_zone"]), frame_box, used_boxes)
        if placed is None:
            return None
        piece["input_pos"] = placed

    expected, overlap = _render_expected(inner_size, pieces)
    if overlap == 0:
        return None
    visible_colors = set(map(int, np.unique(expected))) - {BG}
    if len(visible_colors) < 3:
        return None
    visible_count = int((expected != BG).sum())
    if visible_count < inner_size * inner_size // 4 or visible_count > inner_size * inner_size - 2:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after repeated attempts")
