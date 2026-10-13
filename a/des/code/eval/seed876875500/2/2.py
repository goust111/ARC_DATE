import random
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

BG_COLOR = 0
FRAME_COLOR = 5
GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _mask_from_coords(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    coords = tuple((int(r), int(c)) for r, c in np.argwhere(mask))
    return (int(mask.shape[0]), int(mask.shape[1])), coords


def _is_connected(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    return len(_components(mask)) == 1


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, int, int, int, int]:
    best = None
    best_score = None
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 5 or w < 5:
                continue
            patch = np.zeros((h, w), dtype=bool)
            for r, c in comp:
                patch[r - r0, c - c0] = True
            ring = np.zeros((h, w), dtype=bool)
            ring[0, :] = True
            ring[-1, :] = True
            ring[:, 0] = True
            ring[:, -1] = True
            if not np.array_equal(patch, ring):
                continue
            if (r1 - r0 + 1) * (c1 - c0 + 1) <= 25:
                continue
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            inner_colors = set(int(x) for x in np.unique(inner))
            if bg not in inner_colors or len(inner_colors - {bg}) == 0:
                continue
            score = (h * w, 2 * (h + w))
            if best_score is None or score > best_score:
                best_score = score
                best = (r0, c0, r1, c1, color)
    if best is None:
        raise AssertionError("failed to find central frame")
    return best


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fr0, fc0, fr1, fc1, frame_color = _find_frame(g, bg)

    inner = g[fr0 + 1 : fr1, fc0 + 1 : fc1]
    hole_infos = []
    for comp in _components(inner == bg):
        if any(r == 0 or c == 0 or r == inner.shape[0] - 1 or c == inner.shape[1] - 1 for r, c in comp):
            continue
        boundary_colors = set()
        for rr, cc in comp:
            ar = fr0 + 1 + rr
            ac = fc0 + 1 + cc
            for dr, dc in DIR4:
                nr, nc = ar + dr, ac + dc
                if fr0 + 1 <= nr < fr1 and fc0 + 1 <= nc < fc1:
                    val = int(g[nr, nc])
                    if val != bg and val != frame_color:
                        boundary_colors.add(val)
        if len(boundary_colors) != 1:
            raise AssertionError("each hole must belong to exactly one colored region")
        hole_infos.append(
            {
                "coords": [(fr0 + 1 + rr, fc0 + 1 + cc) for rr, cc in comp],
                "mask": _mask_from_coords(comp),
                "color": next(iter(boundary_colors)),
            }
        )

    outside_mask = g != bg
    outside_mask[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    piece_infos = []
    for comp in _components(outside_mask):
        piece_infos.append({"coords": comp, "mask": _mask_from_coords(comp)})

    hole_sigs = Counter(_shape_signature(info["mask"]) for info in hole_infos)
    piece_sigs = Counter(_shape_signature(info["mask"]) for info in piece_infos)
    if not hole_infos or hole_sigs != piece_sigs:
        raise AssertionError("outside fragments must match the frame holes by translation-only shape")

    out = g.copy()
    for info in piece_infos:
        for r, c in info["coords"]:
            out[r, c] = bg
    for info in hole_infos:
        for r, c in info["coords"]:
            out[r, c] = int(info["color"])
    return out


def _stair_line(length: int, low: int, high: int, rng: random.Random) -> np.ndarray | None:
    if low > high:
        return None
    cur = rng.randint(low, high)
    vals = []
    while len(vals) < length:
        seg = rng.randint(1, 3)
        vals.extend([cur] * seg)
        cur = max(low, min(high, cur + rng.choice([-1, 0, 1])))
    return np.array(vals[:length], dtype=int)


def _layout_labels(kind: str, h: int, w: int, rng: random.Random) -> np.ndarray | None:
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    min_w = max(3, min(5, w // 3))
    min_h = max(3, min(5, h // 3))

    if kind == "v2":
        cut = _stair_line(h, min_w, w - min_w, rng)
        if cut is None:
            return None
        return np.where(cols < cut[:, None], 0, 1).astype(int)

    if kind == "h2":
        cut = _stair_line(w, min_h, h - min_h, rng)
        if cut is None:
            return None
        return np.where(rows < cut[None, :], 0, 1).astype(int)

    if kind == "t3v":
        vcut = _stair_line(h, min_w, w - min_w, rng)
        hcut = _stair_line(w, min_h, h - min_h, rng)
        if vcut is None or hcut is None:
            return None
        right = cols >= vcut[:, None]
        top = rows < hcut[None, :]
        labels = np.zeros((h, w), dtype=int)
        labels[right & top] = 1
        labels[right & ~top] = 2
        return labels

    if kind == "t3h":
        hcut = _stair_line(w, min_h, h - min_h, rng)
        vcut = _stair_line(h, min_w, w - min_w, rng)
        if vcut is None or hcut is None:
            return None
        bottom = rows >= hcut[None, :]
        left = cols < vcut[:, None]
        labels = np.zeros((h, w), dtype=int)
        labels[bottom & left] = 1
        labels[bottom & ~left] = 2
        return labels

    if kind == "quad":
        vcut = _stair_line(h, min_w, w - min_w, rng)
        hcut = _stair_line(w, min_h, h - min_h, rng)
        if vcut is None or hcut is None:
            return None
        left = cols < vcut[:, None]
        top = rows < hcut[None, :]
        labels = np.zeros((h, w), dtype=int)
        labels[top & ~left] = 1
        labels[~top & left] = 2
        labels[~top & ~left] = 3
        return labels

    return None


def _valid_layout(labels: np.ndarray, n_regions: int) -> bool:
    for region_id in range(n_regions):
        mask = labels == region_id
        if int(mask.sum()) < 16 or not _is_connected(mask):
            return False
    return True


def _random_polyomino(rng: random.Random, area_lo: int, area_hi: int) -> np.ndarray | None:
    for _ in range(120):
        target = rng.randint(area_lo, area_hi)
        cells = {(0, 0)}
        frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        while len(cells) < target and frontier:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            r, c = cell
            for dr, dc in DIR4:
                nxt = (r + dr, c + dc)
                if nxt not in cells:
                    frontier.add(nxt)
        if len(cells) != target:
            continue
        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        norm = {(r - min_r, c - min_c) for r, c in cells}
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        if h < 2 or w < 2 or h > 4 or w > 5:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in norm:
            mask[r, c] = True
        if mask.all():
            continue
        if len(set(int(x) for x in mask.sum(axis=0))) == 1 and len(set(int(x) for x in mask.sum(axis=1))) == 1:
            continue
        return mask
    return None


def _place_hole(region_mask: np.ndarray, used: set, rng: random.Random) -> tuple[np.ndarray, int, int, tuple] | None:
    region_area = int(region_mask.sum())
    area_hi = min(7, max(4, region_area // 5))
    for _ in range(160):
        hole = _random_polyomino(rng, 3, area_hi)
        if hole is None:
            continue
        sig = _shape_signature(hole)
        if sig in used:
            continue
        h, w = hole.shape
        candidates = []
        hole_cells = [(int(rr), int(cc)) for rr, cc in np.argwhere(hole)]
        hole_set = set(hole_cells)
        for r0 in range(region_mask.shape[0] - h + 1):
            for c0 in range(region_mask.shape[1] - w + 1):
                if not all(region_mask[r0 + rr, c0 + cc] for rr, cc in hole_cells):
                    continue
                after = region_mask.copy()
                sub = after[r0 : r0 + h, c0 : c0 + w]
                if not np.all(sub[hole]):
                    continue
                sub[hole] = False
                ok = True
                for rr, cc in hole_cells:
                    for dr, dc in DIR4:
                        nr = rr + dr
                        nc = cc + dc
                        if (nr, nc) in hole_set:
                            continue
                        ar = r0 + nr
                        ac = c0 + nc
                        if not (0 <= ar < region_mask.shape[0] and 0 <= ac < region_mask.shape[1] and after[ar, ac]):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                if _is_connected(after):
                    candidates.append((r0, c0))
        if candidates:
            r0, c0 = rng.choice(candidates)
            return hole, int(r0), int(c0), sig
    return None


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b

    def gap1(x0: int, x1: int, y0: int, y1: int) -> int:
        if y1 < x0:
            return x0 - y1
        if x1 < y0:
            return y0 - x1
        return 0

    return gap1(ar0, ar1, br0, br1) + gap1(ac0, ac1, bc0, bc1)


def _place_pieces(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    pieces: list[tuple[np.ndarray, int]],
    bg: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    placed_boxes: list[tuple[int, int, int, int]] = []
    for mask, color in sorted(pieces, key=lambda item: -int(item[0].sum())):
        ph, pw = mask.shape
        candidates = []
        for r0 in range(1, h - ph):
            for c0 in range(1, w - pw):
                box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if _bbox_gap(box, frame_bbox) < 2:
                    continue
                if any(_bbox_gap(box, other) < 2 for other in placed_boxes):
                    continue
                if np.any(grid[r0 : r0 + ph, c0 : c0 + pw] != bg):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = rng.choice(candidates)
        patch = grid[r0 : r0 + ph, c0 : c0 + pw]
        patch[mask] = int(color)
        placed_boxes.append((r0, c0, r0 + ph - 1, c0 + pw - 1))
    return True


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    n_regions = rng.choice([2, 3, 4])
    if n_regions == 2:
        kind = rng.choice(["v2", "h2"])
    elif n_regions == 3:
        kind = rng.choice(["t3v", "t3h"])
    else:
        kind = "quad"

    inner_h = rng.randint(10, 15)
    inner_w = rng.randint(12, 18)
    labels = _layout_labels(kind, inner_h, inner_w, rng)
    if labels is None or not _valid_layout(labels, n_regions):
        return None

    frame_h = inner_h + 2
    frame_w = inner_w + 2
    grid_h = frame_h + rng.randint(7, 12)
    grid_w = frame_w + rng.randint(10, 16)
    fr0 = (grid_h - frame_h) // 2 + rng.randint(-1, 1)
    fc0 = (grid_w - frame_w) // 2 + rng.randint(-2, 2)
    fr0 = max(2, min(fr0, grid_h - frame_h - 2))
    fc0 = max(2, min(fc0, grid_w - frame_w - 2))
    fr1 = fr0 + frame_h - 1
    fc1 = fc0 + frame_w - 1

    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    grid[fr0, fc0 : fc1 + 1] = FRAME_COLOR
    grid[fr1, fc0 : fc1 + 1] = FRAME_COLOR
    grid[fr0 : fr1 + 1, fc0] = FRAME_COLOR
    grid[fr0 : fr1 + 1, fc1] = FRAME_COLOR

    available = [c for c in PALETTE if c not in {BG_COLOR, FRAME_COLOR}]
    picks = rng.sample(available, 2 * n_regions)
    region_colors = picks[:n_regions]
    piece_colors = picks[n_regions:]

    target = grid.copy()
    for region_id, color in enumerate(region_colors):
        target[fr0 + 1 : fr1, fc0 + 1 : fc1][labels == region_id] = int(color)

    hole_specs = []
    used_sigs = set()
    inp = target.copy()
    for region_id in rng.sample(list(range(n_regions)), n_regions):
        region_mask = labels == region_id
        placed = _place_hole(region_mask, used_sigs, rng)
        if placed is None:
            return None
        hole_mask, hr, hc, sig = placed
        used_sigs.add(sig)
        hole_specs.append((region_id, hole_mask, hr, hc))
        sub = inp[fr0 + 1 + hr : fr0 + 1 + hr + hole_mask.shape[0], fc0 + 1 + hc : fc0 + 1 + hc + hole_mask.shape[1]]
        sub[hole_mask] = BG_COLOR

    if get_bg(inp) != BG_COLOR:
        return None

    pieces = []
    order = list(range(len(hole_specs)))
    rng.shuffle(order)
    for piece_idx, hole_idx in enumerate(order):
        _region_id, hole_mask, _hr, _hc = hole_specs[hole_idx]
        pieces.append((hole_mask.copy(), piece_colors[piece_idx]))

    if not _place_pieces(inp, (fr0, fc0, fr1, fc1), pieces, BG_COLOR, rng):
        return None
    if get_bg(inp) != BG_COLOR:
        return None

    try:
        if not np.array_equal(main(inp), target):
            return None
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
