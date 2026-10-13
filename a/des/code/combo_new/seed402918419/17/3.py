import random
from collections import Counter, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GEN_RETRIES = 300
PATCH_RETRIES = 200
PLACEMENT_RETRIES = 500


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _background_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    border = []
    border.extend(int(v) for v in grid[0, :])
    border.extend(int(v) for v in grid[h - 1, :])
    if h > 2:
        border.extend(int(v) for v in grid[1 : h - 1, 0])
        border.extend(int(v) for v in grid[1 : h - 1, w - 1])
    return Counter(border).most_common(1)[0][0]


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _hole_count_from_mask(mask: np.ndarray) -> int:
    holes = 0
    bg_comps = _components_mask(~mask)
    h, w = mask.shape
    for comp in bg_comps:
        touches_edge = np.any(comp[:, 0] == 0) or np.any(comp[:, 0] == h - 1) or np.any(comp[:, 1] == 0) or np.any(comp[:, 1] == w - 1)
        if not touches_edge:
            holes += 1
    return holes


def _hole_count_component(grid: np.ndarray, coords: np.ndarray, bg: int) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = np.zeros(sub.shape, dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    bg_mask = sub == int(bg)
    holes = 0
    for comp in _components_mask(bg_mask):
        touches_edge = np.any(comp[:, 0] == 0) or np.any(comp[:, 0] == sub.shape[0] - 1) or np.any(comp[:, 1] == 0) or np.any(comp[:, 1] == sub.shape[1] - 1)
        if not touches_edge:
            holes += 1
    return holes


def _outer_shell(grid: np.ndarray, coords: np.ndarray, bg: int, layers: int) -> list[tuple[int, int]]:
    if layers <= 0:
        return []
    h, w = grid.shape
    region = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        region[int(r), int(c)] = True

    dist = np.full((h, w), -1, dtype=int)
    q = deque()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == int(bg) and dist[nr, nc] == -1:
                dist[nr, nc] = 1
                q.append((nr, nc))

    shell = []
    while q:
        r, c = q.popleft()
        d = int(dist[r, c])
        if d > layers:
            continue
        shell.append((r, c))
        if d == layers:
            continue
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not region[nr, nc] and int(grid[nr, nc]) == int(bg) and dist[nr, nc] == -1:
                dist[nr, nc] = d + 1
                q.append((nr, nc))
    return shell


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw same-color outer contour rings whose depth equals each region's enclosed-hole count.

    Input structure:
    - A dominant background color fills the border and most empty space.
    - Several 4-connected monochrome regions of distinct visible colors sit in the interior.
    - Each region may have 0, 1, 2, or 3 enclosed background holes, and regions do not touch.

    Rule:
    - Keep every original region unchanged, including any background holes inside it.
    - For each 4-connected colored region, count how many enclosed background components lie inside it.
    - If a region has N holes, paint all background cells at 4-neighbor distance 1 through N from that region with the region's own color.
    - Expansion only travels through original background cells, so grid edges and other regions block it naturally.

    Key insight:
    - The output ring thickness is not fixed by shape size; it is determined by the region's topology, specifically its number of enclosed holes.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = g.copy()

    for color in sorted(int(v) for v in np.unique(g) if int(v) != int(bg)):
        for coords in _components_mask(g == int(color)):
            holes = _hole_count_component(g, coords, bg)
            for r, c in _outer_shell(g, coords, bg, holes):
                out[r, c] = int(color)
    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _rect_margin_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 1) -> bool:
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def _add_tab(canvas: np.ndarray, top: int, left: int, core_h: int, core_w: int, rng: random.Random) -> None:
    side = rng.choice(["top", "bottom", "left", "right"])
    if side in ("top", "bottom"):
        seg_len = rng.randint(1, max(1, core_w - 1))
        start = rng.randint(0, core_w - seg_len)
        row = top - 1 if side == "top" else top + core_h
        c0 = left + start
        canvas[row, c0 : c0 + seg_len] = True
        if rng.random() < 0.25 and 0 < row < canvas.shape[0] - 1 and seg_len >= 2:
            row2 = row - 1 if side == "top" else row + 1
            if 0 <= row2 < canvas.shape[0]:
                c1 = c0 + rng.randint(0, seg_len - 1)
                c2 = c0 + rng.randint(c1 - c0, seg_len - 1) + 1
                canvas[row2, c1:c2] = True
    else:
        seg_len = rng.randint(1, max(1, core_h - 1))
        start = rng.randint(0, core_h - seg_len)
        col = left - 1 if side == "left" else left + core_w
        r0 = top + start
        canvas[r0 : r0 + seg_len, col] = True
        if rng.random() < 0.25 and 0 < col < canvas.shape[1] - 1 and seg_len >= 2:
            col2 = col - 1 if side == "left" else col + 1
            if 0 <= col2 < canvas.shape[1]:
                r1 = r0 + rng.randint(0, seg_len - 1)
                r2 = r0 + rng.randint(r1 - r0, seg_len - 1) + 1
                canvas[r1:r2, col2] = True


def _carve_bite(canvas: np.ndarray, top: int, left: int, core_h: int, core_w: int, rng: random.Random) -> None:
    side = rng.choice(["top", "bottom", "left", "right"])
    if side in ("top", "bottom"):
        if core_w <= 2:
            return
        seg_len = rng.randint(1, max(1, core_w - 2))
        start = rng.randint(0, core_w - seg_len)
        row = top if side == "top" else top + core_h - 1
        canvas[row, left + start : left + start + seg_len] = False
    else:
        if core_h <= 2:
            return
        seg_len = rng.randint(1, max(1, core_h - 2))
        start = rng.randint(0, core_h - seg_len)
        col = left if side == "left" else left + core_w - 1
        canvas[top + start : top + start + seg_len, col] = False


def _generate_patch(hole_count: int, rng: random.Random) -> np.ndarray:
    for _ in range(PATCH_RETRIES):
        min_h = 3 if hole_count == 0 else 5
        min_w = 3 if hole_count == 0 else 5
        core_h = rng.randint(min_h + hole_count // 2, min(10, 6 + hole_count * 2))
        core_w = rng.randint(min_w + hole_count // 2, min(11, 7 + hole_count * 2))
        canvas = np.zeros((core_h + 4, core_w + 4), dtype=bool)
        top = left = 2
        canvas[top : top + core_h, left : left + core_w] = True

        holes: list[tuple[int, int, int, int]] = []
        ok = True
        for _hole_idx in range(hole_count):
            placed = False
            for _ in range(120):
                max_hh = min(3, core_h - 2)
                max_ww = min(3, core_w - 2)
                if max_hh < 1 or max_ww < 1:
                    break
                hh = rng.randint(1, max_hh)
                ww = rng.randint(1, max_ww)
                r0 = rng.randint(1, core_h - hh - 1)
                c0 = rng.randint(1, core_w - ww - 1)
                rect = (r0, c0, r0 + hh - 1, c0 + ww - 1)
                if any(_rect_margin_overlap(rect, prev, margin=1) for prev in holes):
                    continue
                holes.append(rect)
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        for r0, c0, r1, c1 in holes:
            canvas[top + r0 : top + r1 + 1, left + c0 : left + c1 + 1] = False

        for _ in range(rng.randint(0, 3)):
            _add_tab(canvas, top, left, core_h, core_w, rng)
        if rng.random() < 0.45:
            _carve_bite(canvas, top, left, core_h, core_w, rng)
        if rng.random() < 0.20:
            _carve_bite(canvas, top, left, core_h, core_w, rng)

        patch = _crop_mask(canvas)
        if len(_components_mask(patch)) != 1:
            continue
        if _hole_count_from_mask(patch) != hole_count:
            continue
        if max(patch.shape) > 12:
            continue
        return patch
    raise RuntimeError(f"failed to build patch with {hole_count} holes")


def _global_cells(mask: np.ndarray, origin: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = origin
    pts = np.argwhere(mask)
    return [(int(r0 + r), int(c0 + c)) for r, c in pts]


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    row_gap = 0
    col_gap = 0
    if a[2] < b[0]:
        row_gap = int(b[0] - a[2])
    elif b[2] < a[0]:
        row_gap = int(a[0] - b[2])
    if a[3] < b[1]:
        col_gap = int(b[1] - a[3])
    elif b[3] < a[1]:
        col_gap = int(a[1] - b[3])
    return row_gap + col_gap


def _place_objects(
    h: int,
    w: int,
    patches: list[np.ndarray],
    hole_counts: list[int],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    order = sorted(range(len(patches)), key=lambda i: (-hole_counts[i], -(int(patches[i].sum()))))
    placements: list[tuple[int, int] | None] = [None] * len(patches)
    placed_boxes: dict[int, tuple[int, int, int, int]] = {}

    for idx in order:
        patch = patches[idx]
        ring = hole_counts[idx]
        ph, pw = patch.shape
        border_pad = ring + 1
        r_lo = border_pad
        c_lo = border_pad
        r_hi = h - ph - border_pad
        c_hi = w - pw - border_pad
        if r_lo > r_hi or c_lo > c_hi:
            return None

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = rng.randint(r_lo, r_hi)
            c0 = rng.randint(c_lo, c_hi)
            bbox = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
            ok = True
            for other_idx, other_bbox in placed_boxes.items():
                if _bbox_gap(bbox, other_bbox) < hole_counts[idx] + hole_counts[other_idx] + 1:
                    ok = False
                    break
            if not ok:
                continue
            placements[idx] = (r0, c0)
            placed_boxes[idx] = bbox
            placed = True
            break
        if not placed:
            return None

    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _analyze_regions(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    info = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        for comp in _components_mask(grid == int(color)):
            info.append((int(color), _hole_count_component(grid, comp, bg)))
    return info


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    bg = rng.choice(list(range(10)))
    grid = np.full((h, w), int(bg), dtype=int)

    num_objects = rng.choices([2, 3, 4], weights=[2, 3, 2])[0]
    hole_counts = rng.sample([0, 1, 2, 3], k=num_objects)
    if all(v == 0 for v in hole_counts):
        return None

    colors = rng.sample([c for c in range(10) if c != bg], k=num_objects)
    patches = []
    for holes in hole_counts:
        try:
            patches.append(_generate_patch(holes, rng))
        except RuntimeError:
            return None

    placements = _place_objects(h, w, patches, hole_counts, rng)
    if placements is None or len(placements) != num_objects:
        return None

    for color, patch, (r0, c0) in zip(colors, patches, placements):
        pts = np.argwhere(patch)
        for r, c in pts:
            grid[int(r0 + r), int(c0 + c)] = int(color)

    analyzed = sorted(hc for _color, hc in _analyze_regions(grid, bg))
    if analyzed != sorted(hole_counts):
        return None

    for comp in _components_mask(grid != int(bg)):
        r0, c0, r1, c1 = _bbox(comp)
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
