import random
from collections import Counter, defaultdict

import numpy as np


BG = 0
FRAME = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return int(coords[:, 0].min()), int(coords[:, 1].min()), int(coords[:, 0].max()), int(coords[:, 1].max())


def _extract_patch(grid: np.ndarray, coords: np.ndarray, bg: int = BG) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])
    return patch, (r0, c0, r1, c1)


def _paste_centered(out: np.ndarray, patch: np.ndarray, slot: tuple[int, int, int, int], color: int, bg: int = BG) -> None:
    sr0, sc0, sr1, sc1 = slot
    ph, pw = patch.shape
    sh = sr1 - sr0 + 1
    sw = sc1 - sc0 + 1
    dr = sr0 + (sh - ph) // 2
    dc = sc0 + (sw - pw) // 2
    for r in range(ph):
        for c in range(pw):
            if int(patch[r, c]) != bg:
                out[dr + r, dc + c] = color


def _find_frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = grid != BG
    comps = _components(mask)
    assert comps
    frame_comp = max(comps, key=lambda pts: len(pts))
    return _bbox(frame_comp)


def _slot_boundaries(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[list[int], list[int]]:
    r0, c0, r1, c1 = bbox
    width = c1 - c0 + 1
    height = r1 - r0 + 1
    row_bounds = []
    for r in range(r0, r1 + 1):
        if int(np.sum(grid[r, c0 : c1 + 1] == FRAME)) >= max(3, (2 * width) // 3):
            row_bounds.append(r)
    col_bounds = []
    for c in range(c0, c1 + 1):
        if int(np.sum(grid[r0 : r1 + 1, c] == FRAME)) >= max(3, (2 * height) // 3):
            col_bounds.append(c)
    row_bounds = sorted(set(row_bounds))
    col_bounds = sorted(set(col_bounds))
    assert len(row_bounds) >= 3 and len(col_bounds) >= 3
    return row_bounds, col_bounds


def _slots_from_bounds(row_bounds: list[int], col_bounds: list[int]) -> list[list[tuple[int, int, int, int]]]:
    slots = []
    for i in range(len(row_bounds) - 1):
        row_slots = []
        for j in range(len(col_bounds) - 1):
            row_slots.append((row_bounds[i] + 1, col_bounds[j] + 1, row_bounds[i + 1] - 1, col_bounds[j + 1] - 1))
        slots.append(row_slots)
    return slots


def _edge_marker(grid: np.ndarray, row: int, c0: int, c1: int) -> int:
    vals = [int(grid[row, c]) for c in range(c0, c1 + 1) if int(grid[row, c]) not in (BG, FRAME)]
    assert len(vals) == 1
    return vals[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move same-color sprite pairs into a two-row slot frame.

    The grey frame is split into equal slots. A colored marker on the top edge names the
    destination color for the top-row slot in that column; a colored marker on the bottom
    edge names the destination color for the bottom-row slot in that column.

    For each color there are exactly two monochrome sprites outside the frame. The smaller
    sprite goes to that color's top-edge slot; the larger sprite goes to that color's
    bottom-edge slot. Sprites keep their shape and are centered within their slots.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    fr0, fc0, fr1, fc1 = _find_frame_bbox(g)
    row_bounds, col_bounds = _slot_boundaries(g, (fr0, fc0, fr1, fc1))
    slots = _slots_from_bounds(row_bounds, col_bounds)
    assert len(slots) == 2

    top_targets: dict[int, tuple[int, int, int, int]] = {}
    bottom_targets: dict[int, tuple[int, int, int, int]] = {}
    for j in range(len(slots[0])):
        color_top = _edge_marker(g, fr0, col_bounds[j] + 1, col_bounds[j + 1] - 1)
        color_bottom = _edge_marker(g, fr1, col_bounds[j] + 1, col_bounds[j + 1] - 1)
        top_targets[color_top] = slots[0][j]
        bottom_targets[color_bottom] = slots[1][j]

    left_mask = np.zeros_like(g, dtype=bool)
    left_mask[:, :fc0] = g[:, :fc0] != BG
    sprites_by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in _components(left_mask):
        color = int(g[tuple(comp[0])])
        if np.any(g[comp[:, 0], comp[:, 1]] != color):
            continue
        patch, bbox = _extract_patch(g, comp, bg=BG)
        sprites_by_color[color].append({"coords": comp, "patch": patch, "bbox": bbox, "area": int(len(comp))})

    for color in set(top_targets) | set(bottom_targets):
        pair = sorted(sprites_by_color[int(color)], key=lambda s: (s["area"], s["bbox"][0], s["bbox"][1]))
        assert len(pair) == 2
        for sprite in pair:
            out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = BG
        _paste_centered(out, pair[0]["patch"], top_targets[int(color)], int(color), bg=BG)
        _paste_centered(out, pair[1]["patch"], bottom_targets[int(color)], int(color), bg=BG)

    return out


def _random_shape(rng: random.Random, h_max: int, w_max: int, area_min: int, area_max: int) -> np.ndarray:
    dims = [(h, w) for h in range(1, h_max + 1) for w in range(1, w_max + 1) if h * w >= area_min]
    if not dims:
        raise RuntimeError("no feasible shape dimensions")
    for _ in range(100):
        h, w = rng.choice(dims)
        target = rng.randint(area_min, min(area_max, h * w))
        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        border = set()
        while len(cells) < target:
            border.clear()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        border.add((nr, nc))
            if not border:
                break
            cells.add(rng.choice(tuple(border)))
        if len(cells) != target:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        for r, c in cells:
            patch[r - r0, c - c0] = 1
        return patch
    raise RuntimeError("shape generation failed")


def _can_place(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, pad: int = 1) -> bool:
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    if np.any(grid[rr0:rr1, cc0:cc1] != BG):
        return False
    return True


def _stamp(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = patch.shape
    for r in range(h):
        for c in range(w):
            if patch[r, c]:
                grid[r0 + r, c0 + c] = color


def _build_frame(grid: np.ndarray, fr0: int, fc0: int, slot_h: int, slot_w: int, cols: int, top_colors: list[int], bottom_colors: list[int]) -> None:
    fr1 = fr0 + 2 * slot_h + 2
    fc1 = fc0 + cols * slot_w + cols
    grid[fr0 : fr1 + 1, fc0 : fc1 + 1] = FRAME
    for r in range(fr0 + 1, fr1):
        for c in range(fc0 + 1, fc1):
            if r == fr0 + slot_h + 1:
                grid[r, c] = FRAME
            elif any(c == fc0 + k * (slot_w + 1) for k in range(1, cols)):
                grid[r, c] = FRAME
            else:
                grid[r, c] = BG
    for j, color in enumerate(top_colors):
        c_mid = fc0 + 1 + j * (slot_w + 1) + slot_w // 2
        grid[fr0, c_mid] = int(color)
    for j, color in enumerate(bottom_colors):
        c_mid = fc0 + 1 + j * (slot_w + 1) + slot_w // 2
        grid[fr1, c_mid] = int(color)


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    cols = rng.randint(2, 4)
    slot_h = rng.randint(4, 7)
    slot_w = rng.randint(4, 7)
    colors = rng.sample([1, 2, 3, 4, 6, 7, 8, 9], cols)
    top_colors = colors[:]
    bottom_colors = colors[:]
    rng.shuffle(top_colors)
    rng.shuffle(bottom_colors)
    if top_colors == bottom_colors:
        rng.shuffle(bottom_colors)
        if top_colors == bottom_colors:
            bottom_colors = bottom_colors[::-1]

    frame_h = 2 * slot_h + 3
    frame_w = cols * slot_w + cols + 1
    left_w = rng.randint(max(10, cols * 5), max(12, cols * 7))
    H = frame_h + rng.randint(2, 6)
    W = left_w + frame_w + rng.randint(2, 5)
    grid = np.full((H, W), BG, dtype=int)

    fr0 = rng.randint(1, H - frame_h - 1)
    fc0 = W - frame_w - rng.randint(1, 3)
    _build_frame(grid, fr0, fc0, slot_h, slot_w, cols, top_colors, bottom_colors)

    top_slot_by_color = {}
    bottom_slot_by_color = {}
    for j in range(cols):
        top_slot_by_color[top_colors[j]] = (fr0 + 1, fc0 + 1 + j * (slot_w + 1), fr0 + slot_h, fc0 + j * (slot_w + 1) + slot_w)
        bottom_slot_by_color[bottom_colors[j]] = (fr0 + slot_h + 2, fc0 + 1 + j * (slot_w + 1), fr0 + 2 * slot_h + 1, fc0 + j * (slot_w + 1) + slot_w)

    placed = 0
    for color in colors:
        top_slot = top_slot_by_color[color]
        bottom_slot = bottom_slot_by_color[color]
        max_h_small = max(1, top_slot[2] - top_slot[0] + 1 - 1)
        max_w_small = max(1, top_slot[3] - top_slot[1] + 1 - 1)
        max_h_big = max(1, bottom_slot[2] - bottom_slot[0] + 1 - 1)
        max_w_big = max(1, bottom_slot[3] - bottom_slot[1] + 1 - 1)

        small = _random_shape(rng, max_h_small, max_w_small, 2, max(2, min(6, max_h_small * max_w_small - 1)))
        big_area_min = min(max_h_big * max_w_big, int(np.sum(small)) + 1)
        if big_area_min > max_h_big * max_w_big:
            return None
        big = _random_shape(rng, max_h_big, max_w_big, big_area_min, min(max_h_big * max_w_big, big_area_min + 4))
        if int(np.sum(big)) <= int(np.sum(small)):
            return None

        for patch in [small, big]:
            success = False
            ph, pw = patch.shape
            for _ in range(150):
                r0 = rng.randint(0, H - ph)
                c0 = rng.randint(0, max(0, fc0 - pw - 1))
                if _can_place(grid, patch, r0, c0, pad=1):
                    _stamp(grid, patch, r0, c0, int(color))
                    success = True
                    break
            if not success:
                return None
        placed += 1

    if placed != cols:
        return None
    try:
        out = main(grid)
    except Exception:
        return None
    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 300 attempts")
