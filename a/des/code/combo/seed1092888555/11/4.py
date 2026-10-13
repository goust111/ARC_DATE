from __future__ import annotations

import random
from collections import deque

import numpy as np


def _components4(grid: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((color, cells))
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _side_counts(cells: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> dict[str, int]:
    r0, c0, r1, c1 = bbox
    mid_r = (r0 + r1) / 2.0
    mid_c = (c0 + c1) / 2.0
    counts = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    for r, c in cells:
        if r <= mid_r:
            counts["top"] += 1
        if r >= mid_r:
            counts["bottom"] += 1
        if c <= mid_c:
            counts["left"] += 1
        if c >= mid_c:
            counts["right"] += 1
    return counts


def _choose_side(cells: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> str:
    counts = _side_counts(cells, bbox)
    vertical_gap = counts["top"] - counts["bottom"]
    horizontal_gap = counts["left"] - counts["right"]
    if abs(vertical_gap) > abs(horizontal_gap):
        return "top" if vertical_gap > 0 else "bottom"
    if abs(horizontal_gap) > abs(vertical_gap):
        return "left" if horizontal_gap > 0 else "right"

    # Stable tie-breakers still based on visible geometry.
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h >= w:
        return "top" if vertical_gap >= 0 else "bottom"
    return "left" if horizontal_gap >= 0 else "right"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each monochrome 4-connected object into the emptier side of its own bounding box.

    The preferred side is inferred from object density in the bbox halves:
    compare top vs bottom and left vs right occupancy counts, use the stronger bias,
    and then reflect the object across the bbox midpoint perpendicular to that bias.
    The original object remains; only the mirrored copy is added.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = 0
    out = g.copy()
    for color, cells in _components4(g, bg):
        box = _bbox(cells)
        side = _choose_side(cells, box)
        r0, c0, r1, c1 = box

        for r, c in cells:
            if side in ("top", "bottom"):
                rr, cc = r0 + r1 - r, c
            else:
                rr, cc = r, c0 + c1 - c
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and int(out[rr, cc]) == bg:
                out[rr, cc] = color
    return out


def _connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    start = tuple(map(int, pts[0]))
    q = deque([start])
    seen = {start}
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _random_connected_mask(h: int, w: int, side: str, rng: random.Random) -> np.ndarray | None:
    rows, cols = np.indices((h, w))
    if side == "top":
        allowed = rows <= (h - 1) // 2
    elif side == "bottom":
        allowed = rows >= h // 2
    elif side == "left":
        allowed = cols <= (w - 1) // 2
    else:
        allowed = cols >= w // 2

    allowed_cells = [tuple(map(int, rc)) for rc in np.argwhere(allowed)]
    if len(allowed_cells) < 3:
        return None

    min_size = max(3, len(allowed_cells) // 3)
    max_size = max(min_size, len(allowed_cells) - 1)
    target = rng.randint(min_size, max_size)

    for _ in range(80):
        seed = rng.choice(allowed_cells)
        mask = np.zeros((h, w), dtype=bool)
        mask[seed] = True
        frontier = {seed}
        while int(mask.sum()) < target and frontier:
            base = rng.choice(tuple(frontier))
            nbrs = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = base[0] + dr, base[1] + dc
                if 0 <= nr < h and 0 <= nc < w and allowed[nr, nc] and not mask[nr, nc]:
                    nbrs.append((nr, nc))
            if nbrs:
                pick = rng.choice(nbrs)
                mask[pick] = True
                frontier.add(pick)
            else:
                frontier.remove(base)
                continue
            frontier.add(pick)

        if int(mask.sum()) < 3 or not _connected(mask):
            continue
        if not (mask[0].any() and mask[-1].any() or mask[:, 0].any() and mask[:, -1].any()):
            # Encourage bbox to use space so the mirror effect is visible.
            pass

        cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        r0, c0, r1, c1 = _bbox(cells)
        cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
        cells2 = [tuple(map(int, rc)) for rc in np.argwhere(cropped)]
        box2 = (0, 0, cropped.shape[0] - 1, cropped.shape[1] - 1)
        chosen = _choose_side(cells2, box2)
        if chosen != side:
            continue

        mirrored = np.zeros_like(cropped)
        for r, c in cells2:
            if side in ("top", "bottom"):
                rr, cc = box2[0] + box2[2] - r, c
            else:
                rr, cc = r, box2[1] + box2[3] - c
            mirrored[rr, cc] = True
        if not np.any(mirrored & ~cropped):
            continue

        counts = _side_counts(cells2, box2)
        if side == "top" and counts["top"] <= counts["bottom"]:
            continue
        if side == "bottom" and counts["bottom"] <= counts["top"]:
            continue
        if side == "left" and counts["left"] <= counts["right"]:
            continue
        if side == "right" and counts["right"] <= counts["left"]:
            continue
        return cropped
    return None


def _mirror_mask(mask: np.ndarray, side: str) -> np.ndarray:
    out = mask.copy()
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if side in ("top", "bottom"):
            rr, cc = h - 1 - r, c
        else:
            rr, cc = r, w - 1 - c
        out[rr, cc] = True
    return out


def _can_place(canvas: np.ndarray, top: int, left: int, mask: np.ndarray, side: str) -> bool:
    h, w = mask.shape
    H, W = canvas.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False

    mirrored = _mirror_mask(mask, side)
    for r in range(h):
        for c in range(w):
            if mirrored[r, c] and canvas[top + r, left + c] != 0:
                return False
    margin = 1
    r0 = max(0, top - margin)
    c0 = max(0, left - margin)
    r1 = min(H, top + h + margin)
    c1 = min(W, left + w + margin)
    return not np.any(canvas[r0:r1, c0:c1] != 0)


def _stamp(canvas: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    patch = canvas[top : top + h, left : left + w]
    patch[mask] = color


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        h = rng.randint(12, 22)
        w = rng.randint(12, 22)
        grid = np.zeros((h, w), dtype=int)

        n_objects = rng.randint(2, 5)
        sides = [rng.choice(("top", "bottom", "left", "right")) for _ in range(n_objects)]
        if n_objects >= 4:
            rng.shuffle(sides)
            sides[:4] = ["top", "bottom", "left", "right"]
            rng.shuffle(sides)

        colors = rng.sample(range(1, 10), n_objects)
        ok = True

        for side, color in zip(sides, colors, strict=False):
            placed = False
            for _obj_try in range(120):
                bh = rng.randint(3, 7)
                bw = rng.randint(3, 7)
                if side in ("top", "bottom") and bh < 4:
                    bh = 4
                if side in ("left", "right") and bw < 4:
                    bw = 4
                mask = _random_connected_mask(bh, bw, side, rng)
                if mask is None:
                    continue
                mh, mw = mask.shape
                if mh + 2 > h or mw + 2 > w:
                    continue
                for _place_try in range(80):
                    top = rng.randint(1, h - mh - 1)
                    left = rng.randint(1, w - mw - 1)
                    if _can_place(grid, top, left, mask, side):
                        _stamp(grid, top, left, mask, color)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        out = main(grid)
        if np.array_equal(out, grid):
            continue
        # Re-parse to make sure every object remains distinguishable and non-degenerate.
        comps = _components4(grid, 0)
        if len(comps) != n_objects:
            continue
        added = out != grid
        if not np.any(added):
            continue
        return grid
    raise RuntimeError("failed to generate valid input")
