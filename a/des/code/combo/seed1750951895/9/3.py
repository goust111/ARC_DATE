import random
from collections import Counter, deque

import numpy as np


BG = 7
PALETTE = [c for c in range(10) if c != BG]
GEN_RETRIES = 400
PLACEMENT_RETRIES = 200


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _extract_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], tuple[int, int, int, int]]:
    candidates = []
    for comp in _components_mask(grid != bg):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if w != 2 or h < 2:
            continue
        if c0 not in (0, grid.shape[1] - 2):
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch == bg):
            continue
        if comp.shape[0] != h * w:
            continue
        mapping = {}
        ok = True
        for rr in range(h):
            src = int(patch[rr, 0])
            dst = int(patch[rr, 1])
            if src == dst or src in mapping:
                ok = False
                break
            mapping[src] = dst
        if ok:
            candidates.append((mapping, (r0, c0, r1, c1)))
    assert len(candidates) == 1, "expected a unique 2-column legend on the left or right edge"
    return candidates[0]


def _component_dicts_by_color(grid: np.ndarray, bg: int, banned_bbox: tuple[int, int, int, int]) -> dict[int, list[dict]]:
    r0, c0, r1, c1 = banned_bbox
    work = grid.copy()
    work[r0 : r1 + 1, c0 : c1 + 1] = bg
    result: dict[int, list[dict]] = {}
    for color in map(int, np.unique(work)):
        if color == bg:
            continue
        comps = []
        for coords in _components_mask(work == color):
            br0, bc0, br1, bc1 = _bbox(coords)
            mask = np.zeros((br1 - br0 + 1, bc1 - bc0 + 1), dtype=bool)
            mask[coords[:, 0] - br0, coords[:, 1] - bc0] = True
            comps.append(
                {
                    "coords": coords,
                    "bbox": (br0, bc0, br1, bc1),
                    "mask": mask,
                    "area": int(mask.sum()),
                }
            )
        if comps:
            result[color] = comps
    return result


def _scale_mask_nearest(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    src_h, src_w = mask.shape
    row_idx = np.floor(np.arange(out_h) * src_h / out_h).astype(int)
    col_idx = np.floor(np.arange(out_w) * src_w / out_w).astype(int)
    return mask[row_idx[:, None], col_idx[None, :]]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    legend, legend_bbox = _extract_legend(g, bg)
    objects = _component_dicts_by_color(g, bg, legend_bbox)

    out = g.copy()
    lr0, lc0, lr1, lc1 = legend_bbox
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = bg

    for color, comps in objects.items():
        assert color in legend, "every foreground color must appear in the legend"
        assert len(comps) == 2, "each source color must have exactly two objects"
        comps = sorted(comps, key=lambda d: (d["area"], d["mask"].shape[0] * d["mask"].shape[1]))
        small, large = comps
        if small["area"] == large["area"]:
            raise AssertionError("paired objects must have different areas")

        for comp in comps:
            coords = comp["coords"]
            out[coords[:, 0], coords[:, 1]] = bg

        large_coords = large["coords"]
        out[large_coords[:, 0], large_coords[:, 1]] = int(legend[color])

        br0, bc0, br1, bc1 = large["bbox"]
        target_h = br1 - br0 + 1
        target_w = bc1 - bc0 + 1
        scaled = _scale_mask_nearest(small["mask"], target_h, target_w)

        sr0, sc0, _, _ = small["bbox"]
        er = sr0 + target_h
        ec = sc0 + target_w
        assert er <= g.shape[0] and ec <= g.shape[1], "scaled object must remain in bounds"
        out[sr0:er, sc0:ec][scaled] = color

    return out


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray | None:
    for _ in range(80):
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = {start}
        while len(cells) < area and frontier:
            cr, cc = random.choice(tuple(frontier))
            nbrs = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    nbrs.append((nr, nc))
            if not nbrs:
                frontier.discard((cr, cc))
                continue
            nxt = random.choice(nbrs)
            cells.add(nxt)
            frontier.add(nxt)
        if len(cells) != area:
            continue
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if min(rows) != 0 or max(rows) != h - 1 or min(cols) != 0 or max(cols) != w - 1:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask
    return None


def _make_pair() -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(120):
        large_h = random.randint(3, 6)
        large_w = random.randint(3, 6)
        large_area = random.randint(max(large_h + large_w - 1, (large_h * large_w) // 2), large_h * large_w - 1)
        large = _random_connected_mask(large_h, large_w, large_area)
        if large is None:
            continue

        small_h = random.randint(2, large_h)
        small_w = random.randint(2, large_w)
        max_small_area = min(small_h * small_w - 1, large_area - 1)
        min_small_area = max(3, small_h + small_w - 1)
        if min_small_area > max_small_area:
            continue
        small_area = random.randint(min_small_area, max_small_area)
        small = _random_connected_mask(small_h, small_w, small_area)
        if small is None:
            continue
        return small, large
    return None


def _can_place(occ: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    h0 = max(0, top - pad)
    w0 = max(0, left - pad)
    h1 = min(occ.shape[0], top + h + pad)
    w1 = min(occ.shape[1], left + w + pad)
    return not occ[h0:h1, w0:w1].any()


def _stamp_occ(occ: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    occ[top : top + h, left : left + w] = True


def _draw_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    grid[top : top + h, left : left + w][mask] = color


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(22, 30)
    w = random.randint(22, 30)
    num_pairs = random.randint(2, 4)

    colors = random.sample(PALETTE, num_pairs * 2)
    sources = colors[:num_pairs]
    targets = colors[num_pairs:]
    legend_side = random.choice(["left", "right"])
    legend_top = random.randint(1, h - num_pairs - 1)
    legend_col = 0 if legend_side == "left" else w - 2

    grid = np.full((h, w), BG, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    for i, (src, dst) in enumerate(zip(sources, targets)):
        r = legend_top + i
        grid[r, legend_col] = src
        grid[r, legend_col + 1] = dst
    _stamp_occ(occ, legend_top, legend_col, num_pairs, 2)

    place_min_col = 3 if legend_side == "left" else 1
    place_max_col = w - 1 if legend_side == "left" else w - 3

    for color in sources:
        made = _make_pair()
        if made is None:
            return None
        small, large = made
        large_h, large_w = large.shape

        large_pos = None
        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(1, h - large_h - 1)
            left = random.randint(place_min_col, place_max_col - large_w)
            if _can_place(occ, top, left, large_h, large_w):
                large_pos = (top, left)
                break
        if large_pos is None:
            return None
        _stamp_occ(occ, large_pos[0], large_pos[1], large_h, large_w)
        _draw_mask(grid, large, large_pos[0], large_pos[1], color)

        small_pos = None
        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(1, h - large_h - 1)
            left = random.randint(place_min_col, place_max_col - large_w)
            if _can_place(occ, top, left, large_h, large_w):
                small_pos = (top, left)
                break
        if small_pos is None:
            return None
        _stamp_occ(occ, small_pos[0], small_pos[1], large_h, large_w)
        _draw_mask(grid, small, small_pos[0], small_pos[1], color)

    fg_counts = Counter(map(int, grid[grid != BG]))
    if any(fg_counts[c] == 0 for c in sources + targets):
        return None
    if sum(1 for c in fg_counts if c in sources) != num_pairs:
        return None
    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
