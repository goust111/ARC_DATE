import random
from collections import Counter

import numpy as np


BG = 8
GREY = 5
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(pts)
    return out


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _crop_mask(points: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(points)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches = False
            while stack:
                rr, cc = stack.pop()
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches = True
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches:
                holes += 1
    return holes


def _touches_edge(points: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in points)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    mapping: dict[int, int] = {}
    for color in sorted(set(map(int, np.unique(g))) - {bg, GREY}):
        for pts in _components(g == color):
            if not _touches_edge(pts, h, w):
                continue
            hc = _hole_count(_crop_mask(pts))
            assert hc not in mapping, "legend hole counts must be unique"
            mapping[hc] = color
    assert mapping, "expected at least one edge legend"

    objects = []
    for pts in _components(g == GREY):
        mask = _crop_mask(pts)
        hc = _hole_count(mask)
        if hc not in mapping:
            continue
        r0, c0, _r1, _c1 = _bbox(pts)
        objects.append(
            {
                "mask": mask,
                "color": int(mapping[hc]),
                "area": int(mask.sum()),
                "top": int(r0),
                "left": int(c0),
            }
        )

    if not objects:
        return np.full((1, 1), bg, dtype=int)

    objects.sort(key=lambda obj: (-obj["area"], obj["top"], obj["left"]))
    out_h = sum(obj["mask"].shape[0] for obj in objects)
    out_w = max(obj["mask"].shape[1] for obj in objects)
    out = np.full((out_h, out_w), bg, dtype=int)

    row = 0
    for obj in objects:
        mask = obj["mask"]
        hh, ww = mask.shape
        out[row : row + hh, :ww][mask] = obj["color"]
        row += hh
    return out


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    start = tuple(map(int, pts[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                nxt = (nr, nc)
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return len(seen) == int(mask.sum())


def _with_tabs(base: np.ndarray) -> np.ndarray:
    h, w = base.shape
    canvas = np.zeros((h + 4, w + 4), dtype=bool)
    canvas[2 : 2 + h, 2 : 2 + w] = base
    for _ in range(random.randint(0, 2)):
        side = random.choice(["top", "bottom", "left", "right"])
        if side in ("top", "bottom"):
            span = random.randint(1, max(1, w // 2))
            c0 = 2 + random.randint(0, w - span)
            depth = random.randint(1, 2)
            if side == "top":
                canvas[2 - depth : 2, c0 : c0 + span] = True
            else:
                canvas[2 + h : 2 + h + depth, c0 : c0 + span] = True
        else:
            span = random.randint(1, max(1, h // 2))
            r0 = 2 + random.randint(0, h - span)
            depth = random.randint(1, 2)
            if side == "left":
                canvas[r0 : r0 + span, 2 - depth : 2] = True
            else:
                canvas[r0 : r0 + span, 2 + w : 2 + w + depth] = True
    pts = [tuple(map(int, p)) for p in np.argwhere(canvas)]
    r0, c0, r1, c1 = _bbox(pts)
    return canvas[r0 : r1 + 1, c0 : c1 + 1]


def _make_shape(hole_count: int, legend: bool) -> np.ndarray:
    for _ in range(200):
        if hole_count == 0:
            h = random.randint(2, 4 if legend else 6)
            w = random.randint(2, 5 if legend else 7)
            mask = np.ones((h, w), dtype=bool)
        else:
            options = []
            for rows in range(1, hole_count + 1):
                cols = (hole_count + rows - 1) // rows
                for hole_h in ([1] if legend else [1, 2]):
                    for hole_w in ([1] if legend else [1, 2]):
                        for wall in ([1] if legend else [1, 2]):
                            h = rows * hole_h + (rows + 1) * wall
                            w = cols * hole_w + (cols + 1) * wall
                            if h <= (8 if legend else 11) and w <= (8 if legend else 11):
                                options.append((rows, cols, hole_h, hole_w, wall))
            rows, cols, hole_h, hole_w, wall = random.choice(options)
            mask = np.ones(
                (rows * hole_h + (rows + 1) * wall, cols * hole_w + (cols + 1) * wall),
                dtype=bool,
            )
            cells = [(r, c) for r in range(rows) for c in range(cols)]
            random.shuffle(cells)
            for r, c in cells[:hole_count]:
                rr = wall + r * (hole_h + wall)
                cc = wall + c * (hole_w + wall)
                mask[rr : rr + hole_h, cc : cc + hole_w] = False

        mask = _with_tabs(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, random.randint(0, 3))
        if (
            mask.sum() >= 4
            and _is_connected(mask)
            and _hole_count(mask) == hole_count
            and mask.shape[0] <= (8 if legend else 12)
            and mask.shape[1] <= (8 if legend else 12)
        ):
            return mask
    raise RuntimeError("failed to build shape")


def _can_place(occ: np.ndarray, top: int, left: int, mask: np.ndarray) -> bool:
    h, w = occ.shape
    mh, mw = mask.shape
    if top < 0 or left < 0 or top + mh > h or left + mw > w:
        return False
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + mh + 1)
    c1 = min(w, left + mw + 1)
    return not occ[r0:r1, c0:c1].any()


def _place(grid: np.ndarray, occ: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    mh, mw = mask.shape
    grid[top : top + mh, left : left + mw][mask] = color
    occ[top : top + mh, left : left + mw][mask] = True


def _place_legends(grid: np.ndarray, occ: np.ndarray, legends: list[tuple[int, int, np.ndarray]]) -> bool:
    h, w = grid.shape
    side = random.choice(["top", "bottom", "left", "right"])
    gap = random.randint(1, 2)
    if side in ("top", "bottom"):
        total = sum(mask.shape[1] for _, _, mask in legends) + gap * (len(legends) - 1)
        if total > w - 2:
            return False
        col = random.randint(1, w - 1 - total)
        order = legends[:]
        random.shuffle(order)
        for holes, color, mask in order:
            row = 0 if side == "top" else h - mask.shape[0]
            if not _can_place(occ, row, col, mask):
                return False
            _place(grid, occ, row, col, mask, color)
            col += mask.shape[1] + gap
    else:
        total = sum(mask.shape[0] for _, _, mask in legends) + gap * (len(legends) - 1)
        if total > h - 2:
            return False
        row = random.randint(1, h - 1 - total)
        order = legends[:]
        random.shuffle(order)
        for holes, color, mask in order:
            col = 0 if side == "left" else w - mask.shape[1]
            if not _can_place(occ, row, col, mask):
                return False
            _place(grid, occ, row, col, mask, color)
            row += mask.shape[0] + gap
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 28)
    w = random.randint(20, 28)
    grid = np.full((h, w), BG, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    legend_counts = sorted(random.sample([0, 1, 2, 3], random.randint(2, 4)))
    legend_colors = random.sample(LEGEND_COLORS, len(legend_counts))
    legends = [(hc, color, _make_shape(hc, legend=True)) for hc, color in zip(legend_counts, legend_colors, strict=False)]
    if not _place_legends(grid, occ, legends):
        return None

    grey_holes = legend_counts[:]
    grey_holes.extend(random.choices(legend_counts, k=random.randint(1, 3)))
    unmatched_pool = [hc for hc in range(5) if hc not in legend_counts]
    if not unmatched_pool:
        unmatched_pool = [4]
    grey_holes.extend(random.sample(unmatched_pool, k=random.randint(1, min(2, len(unmatched_pool)))))
    random.shuffle(grey_holes)

    for hc in grey_holes:
        placed = False
        for _ in range(120):
            mask = _make_shape(hc, legend=False)
            top = random.randint(2, h - mask.shape[0] - 2)
            left = random.randint(2, w - mask.shape[1] - 2)
            if _can_place(occ, top, left, mask):
                _place(grid, occ, top, left, mask, GREY)
                placed = True
                break
        if not placed:
            return None

    assert len(_components(grid == GREY)) >= len(legend_counts) + 1
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is None:
            continue
        out = main(grid)
        if isinstance(out, np.ndarray) and out.size > 0:
            return grid
    raise RuntimeError("failed to generate valid input")
