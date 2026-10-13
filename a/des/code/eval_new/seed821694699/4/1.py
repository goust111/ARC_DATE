from __future__ import annotations

import numpy as np
from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

Point = Tuple[int, int]


def _components(mask: np.ndarray) -> List[Tuple[Set[Point], int, int, int, int]]:
    """Find 4-neighbour connected components in a boolean mask."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    out: List[Tuple[Set[Point], int, int, int, int]] = []
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or visited[sr, sc]:
                continue
            q = deque([(sr, sc)])
            visited[sr, sc] = True
            cells: Set[Point] = set()
            minr = maxr = sr
            minc = maxc = sc

            while q:
                r, c = q.popleft()
                cells.add((r, c))
                minr = min(minr, r)
                maxr = max(maxr, r)
                minc = min(minc, c)
                maxc = max(maxc, c)

                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            out.append((cells, minr, maxr, minc, maxc))
    return out


def _normalize(points: Iterable[Point]) -> Set[Point]:
    pts = list(points)
    if not pts:
        return set()
    minr = min(r for r, _ in pts)
    minc = min(c for _, c in pts)
    return {(r - minr, c - minc) for r, c in pts}


def _detect_background_and_grey(grid: np.ndarray) -> Tuple[int, int]:
    """Return (bg color, separator column) using column-wise grey support heuristic."""
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[int(np.argmax(counts))])

    candidates = [int(v) for v in values if int(v) != bg]
    if not candidates:
        w = grid.shape[1]
        return bg, w // 2

    # Pick the color with the most complete vertical line support.
    best_grey = candidates[0]
    best_score = -1.0
    for color in candidates:
        col_counts = np.sum(grid == color, axis=0)
        score = float(np.max(col_counts)) / grid.shape[0]
        if score > best_score:
            best_score = score
            best_grey = color

    # Split at a near-full separator column.
    col_counts = np.sum(grid == best_grey, axis=0)
    sep_cols = np.where(col_counts >= 0.85 * grid.shape[0])[0]
    if sep_cols.size:
        sep = int(sep_cols[int(np.argmax(col_counts[sep_cols]))])
    else:
        sep = int(np.argmax(col_counts))
    return best_grey, sep


def _extract_templates(palette: np.ndarray, bg: int, grey: int) -> Dict[int, Set[Point]]:
    """Extract full template shapes from palette area."""
    templates: Dict[int, Set[Point]] = {}
    for color in np.unique(palette):
        color = int(color)
        if color in (bg, grey):
            continue
        color_mask = palette == color
        comps = _components(color_mask)
        if not comps:
            continue

        best = max(comps, key=lambda x: len(x[0]))
        cells = best[0]
        if len(cells) < 3:
            continue
        _, minr, maxr, minc, maxc = best
        templates[color] = {(r - minr, c - minc) for r, c in cells}
        _ = maxr
        _ = maxc
    return templates


def _is_hollow_frame(component: Set[Point], minr: int, maxr: int, minc: int, maxc: int) -> bool:
    if maxr - minr < 2 or maxc - minc < 2:
        return False
    h = maxr - minr + 1
    w = maxc - minc + 1

    # Ideal hollow rectangular frame size (without duplicates).
    expected = 2 * (h + w) - 4
    if len(component) < expected - 2 or len(component) > expected + 2:
        return False

    # Border must be fully present in perimeter and interior must be empty.
    for r in range(minr, maxr + 1):
        if (r, minc) not in component or (r, maxc) not in component:
            return False
    for c in range(minc, maxc + 1):
        if (minr, c) not in component or (maxr, c) not in component:
            return False

    for r in range(minr + 1, maxr):
        for c in range(minc + 1, maxc):
            if (r, c) in component:
                return False
    return True


def _extract_frames(grid: np.ndarray, grey: int, sep: int) -> List[Tuple[int, int, int, int]]:
    """Return frame boxes as (r0, r1, c0, c1) in global coordinates."""
    work = grid[:, sep + 1 :]
    comps = _components(work == grey)
    frames: List[Tuple[int, int, int, int]] = []
    for cells, minr, maxr, minc, maxc in comps:
        if not _is_hollow_frame(cells, minr, maxr, minc, maxc):
            continue
        r0, r1 = minr, maxr
        c0, c1 = minc + sep + 1, maxc + sep + 1
        frames.append((r0, r1, c0, c1))
    return frames


def _find_shift(fragment: Set[Point], template: Set[Point], frame_h: int, frame_w: int) -> Optional[Point]:
    """Find translation so that fragment + (dr,dc) lies inside template coordinates."""
    if not fragment or not template:
        return None

    template_set = template
    f0 = next(iter(fragment))

    candidates: List[Point] = []
    for tr, tc in template_set:
        dr, dc = tr - f0[0], tc - f0[1]
        ok = True
        for fr, fc in fragment:
            if (fr + dr, fc + dc) not in template_set:
                ok = False
                break
        if not ok:
            continue

        # Mapped points of template must stay in the frame interior.
        for rr, cc in template_set:
            fr = rr - dr
            fc = cc - dc
            if fr < 1 or fr >= frame_h - 1 or fc < 1 or fc >= frame_w - 1:
                ok = False
                break
        if not ok:
            continue

        candidates.append((dr, dc))

    if not candidates:
        return None
    # Deterministic tie-break; valid datasets usually leave this to one answer.
    candidates = sorted(set(candidates), key=lambda s: (abs(s[0]) + abs(s[1]), s[0], s[1]))
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    1) Input contract
    The input contains a left palette region, a one-column vertical separator in a grey
    color, and a right work region with rectangular grey frames and template fragments.

    2) Parsing phase
    Detect background and grey colors, recover all palette templates by color, and
    recover each frame via connected grey components in the work region.

    3) Placement phase
    For every frame, collect fragment coordinates S, match by color to a palette
    template T, then infer the unique shift (dr, dc) such that S + (dr, dc) is inside T.

    4) Rendering phase
    Copy the input and paint the full template translated by (-dr, -dc) only inside the
    frame interior; keep frame borders and existing pixels otherwise unchanged.
    """
    grid = np.asarray(grid)
    h, w = grid.shape
    out = grid.astype(int, copy=True)

    bg = int(np.bincount(grid.ravel()).argmax())
    grey, sep = _detect_background_and_grey(grid)

    palette = grid[:, :sep]
    templates = _extract_templates(palette, bg, grey)
    if not templates:
        return out

    frames = _extract_frames(grid, grey, sep)
    for r0, r1, c0, c1 in frames:
        frame_h = r1 - r0 + 1
        frame_w = c1 - c0 + 1

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0:
            continue

        frag_mask = (interior != bg) & (interior != grey)
        if not np.any(frag_mask):
            continue

        frag_colors, counts = np.unique(interior[frag_mask], return_counts=True)
        color = int(frag_colors[int(np.argmax(counts))])
        template = templates.get(color)
        if template is None:
            continue

        frag_cells = np.argwhere(frag_mask)
        fragment = {(int(r) + 1, int(c) + 1) for r, c in frag_cells}
        shift = _find_shift(fragment, template, frame_h, frame_w)
        if shift is None:
            continue
        dr, dc = shift

        # Paint template points mapped into the frame interior.
        for tr, tc in template:
            rr = r0 + (tr - dr)
            cc = c0 + (tc - dc)
            if r0 + 1 <= rr <= r1 - 1 and c0 + 1 <= cc <= c1 - 1:
                out[rr, cc] = color

    return out


def _shape_rotate(points: Set[Point], k: int) -> Set[Point]:
    pts = set(points)
    for _ in range(k % 4):
        pts = {(c, -r) for r, c in pts}
    return _normalize(pts)


def _shape_reflect(points: Set[Point], axis: int) -> Set[Point]:
    if axis == 0:
        return _normalize({(-r, c) for r, c in points})
    return _normalize({(r, -c) for r, c in points})


def _random_template_shape(rng: np.random.Generator) -> Set[Point]:
    bases = [
        {(0, 0), (0, 1), (0, 2), (1, 1)},
        {(0, 0), (1, 0), (2, 0), (2, 1)},
        {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)},
        {(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)},
        {(0, 0), (1, 0), (1, 1), (1, 2)},
    ]

    if rng.random() < 0.6:
        pts = set(rng.choice(bases))
    else:
        # Grow a random connected polyomino.
        target = int(rng.integers(4, 10))
        pts = {(0, 0)}
        frontier = [(0, 0)]
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while len(pts) < target:
            r, c = frontier[int(rng.integers(0, len(frontier)))]
            dr, dc = dirs[int(rng.integers(0, 4))]
            nxt = (r + dr, c + dc)
            if nxt in pts:
                continue
            pts.add(nxt)
            frontier.append(nxt)

    # Random orientation to diversify.
    pts = _shape_rotate(pts, int(rng.integers(0, 4)))
    if rng.random() < 0.5:
        pts = _shape_reflect(pts, 1)
    if rng.random() < 0.3:
        pts = _shape_reflect(pts, 0)

    # Optional extra random growth for further variety.
    if rng.random() < 0.5 and len(pts) < 10:
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for _ in range(2):
            if len(pts) >= 10:
                break
            anchor = list(pts)[int(rng.integers(0, len(pts)))]
            ar, ac = anchor
            dr, dc = dirs[int(rng.integers(0, 4))]
            cand = (ar + dr, ac + dc)
            if cand in pts:
                continue
            pts = set(pts)
            pts.add(cand)
            pts = _normalize(pts)

    return _normalize(pts)


def _can_place_with_margin(grid_used: np.ndarray, cells: Iterable[Point], H: int, W: int) -> bool:
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and grid_used[rr, cc]:
                    return False
    return True


def _mark_used(grid_used: np.ndarray, cells: Iterable[Point], H: int, W: int) -> None:
    for r, c in cells:
        if 0 <= r < H and 0 <= c < W:
            grid_used[r, c] = True


def generate_input() -> np.ndarray:
    """Generate a random ARC-style task instance matching the described puzzle rule."""
    rng = np.random.default_rng()
    bg = 0
    grey = 5

    H = int(rng.integers(14, 28))
    palette_w = int(rng.integers(6, 10))
    max_work = 30 - palette_w - 1
    work_w = int(rng.integers(max(10, max_work // 2), max_work + 1))
    W = palette_w + 1 + work_w

    grid = np.full((H, W), bg, dtype=int)
    sep = palette_w
    grid[:, sep] = grey

    color_pool = [1, 2, 3, 4, 6, 7, 8, 9]
    rng.shuffle(color_pool)

    n_templates = int(rng.integers(1, min(4, len(color_pool) + 1)))
    palette_used = np.zeros((H, palette_w), dtype=bool)

    placed_templates: List[Tuple[int, Set[Point], Tuple[int, int]]] = []
    for i in range(n_templates):
        color = color_pool[i]
        shape = _random_template_shape(rng)
        if len(shape) < 4:
            continue

        rows = [r for r, _ in shape]
        cols = [c for _, c in shape]
        th = max(rows) + 1
        tw = max(cols) + 1
        if th >= H - 2 or tw >= palette_w - 1:
            continue

        placed = False
        for _ in range(300):
            top = int(rng.integers(0, H - th - 1))
            left = int(rng.integers(0, palette_w - tw - 1))
            cells = [(top + r, left + c) for r, c in shape]
            if not _can_place_with_margin(palette_used, cells, H, palette_w):
                continue
            placed = True
            _mark_used(palette_used, cells, H, palette_w)
            for r, c in cells:
                grid[r, c] = color
            placed_templates.append((color, shape, (top, left)))
            break

        if not placed:
            continue

    work_used = np.zeros((H, W), dtype=bool)
    n_frames = min(len(placed_templates), int(rng.integers(1, max(2, len(placed_templates) + 1))))
    placed_any_frame = False
    for idx in range(n_frames):
        color, shape, _ = placed_templates[idx]
        pts = list(shape)
        th = max(r for r, _ in pts) + 1
        tw = max(c for _, c in pts) + 1

        min_h = th + 2
        min_w = tw + 2
        if min_h >= H or min_w >= W - sep - 1:
            continue

        frame_h = int(rng.integers(min_h, min(H - 1, min_h + 6) + 1))
        frame_w = int(rng.integers(min_w, min(W - sep - 1, min_w + 6) + 1))

        top_limit = H - frame_h - 1
        left_limit = W - frame_w
        if top_limit <= 1 or left_limit <= sep + 1:
            continue

        placed = False
        for _ in range(400):
            top = int(rng.integers(1, top_limit + 1))
            left = int(rng.integers(sep + 1, left_limit + 1))

            r0 = max(0, top - 1)
            r1 = min(H, top + frame_h + 1)
            c0 = max(0, left - 1)
            c1 = min(W, left + frame_w + 1)
            if np.any(work_used[r0:r1, c0:c1]):
                continue

            placed = True
            _mark_used(work_used, [(r, c) for r in range(top - 1, top + frame_h + 1) for c in range(left - 1, left + frame_w + 1)], H, W)
            for cc in range(left, left + frame_w):
                grid[top, cc] = grey
                grid[top + frame_h - 1, cc] = grey
            for rr in range(top, top + frame_h):
                grid[rr, left] = grey
                grid[rr, left + frame_w - 1] = grey

            inner_h = frame_h - 2
            inner_w = frame_w - 2
            if inner_h < th or inner_w < tw:
                break

            off_r = int(rng.integers(0, inner_h - th + 1))
            off_c = int(rng.integers(0, inner_w - tw + 1))

            full_points = [(top + 1 + off_r + r, left + 1 + off_c + c) for r, c in shape]
            missing = int(rng.integers(1, max(2, len(full_points) // 2) + 1))
            keep = len(full_points) - missing
            keep = max(1, min(len(full_points) - 1, keep))
            keep_idx = rng.choice(len(full_points), size=keep, replace=False)
            for k in keep_idx:
                rr, cc = full_points[int(k)]
                grid[rr, cc] = color
            placed_any_frame = True
            break

        if not placed:
            continue

    # Fallback: guarantee at least one valid frame with a matching template.
    if not placed_any_frame or not placed_templates:
        used_colors = {color for color, _, _ in placed_templates}
        fallback_color = 1
        for candidate in (1, 2, 3, 4, 6, 7, 8, 9):
            if candidate not in used_colors:
                fallback_color = candidate
                break

        shape = _random_template_shape(rng)
        if len(shape) < 4:
            shape = {(0, 0), (0, 1), (1, 0), (1, 1)}
        shape_points = list(shape)

        # Place a fallback template in the palette.
        pl_r = 0
        pl_c = 0
        th = max(r for r, _ in shape_points) + 1
        tw = max(c for _, c in shape_points) + 1
        pl_r = min(pl_r, max(0, H - th - 1))
        pl_c = min(pl_c, max(0, palette_w - tw - 1))
        for r, c in shape:
            if 0 <= pl_r + r < H and 0 <= pl_c + c < palette_w:
                grid[pl_r + r, pl_c + c] = fallback_color

        # Place a matching frame in the work area.
        frame_h = min(H - 2, th + 2)
        frame_w = min(W - sep - 1, tw + 2)
        top = max(1, min(6, H - frame_h - 1))
        left = max(sep + 1, min(sep + 2, W - frame_w))

        for cc in range(left, left + frame_w):
            grid[top, cc] = grey
            grid[top + frame_h - 1, cc] = grey
        for rr in range(top, top + frame_h):
            grid[rr, left] = grey
            grid[rr, left + frame_w - 1] = grey

        inner_h = frame_h - 2
        inner_w = frame_w - 2
        off_r = 0
        off_c = 0
        if inner_h > th:
            off_r = int(rng.integers(0, inner_h - th + 1))
        if inner_w > tw:
            off_c = int(rng.integers(0, inner_w - tw + 1))

        full_points = [(top + 1 + off_r + r, left + 1 + off_c + c) for r, c in shape]
        missing = int(rng.integers(1, max(2, len(full_points) // 2) + 1))
        keep = max(1, min(len(full_points) - 1, len(full_points) - missing))
        keep_idx = rng.choice(len(full_points), size=keep, replace=False)
        for k in keep_idx:
            rr, cc = full_points[int(k)]
            grid[rr, cc] = fallback_color

        for r, c in shape_points:
            grid[pl_r + r, pl_c + c] = fallback_color

    return grid


if __name__ == "__main__":
    g = generate_input()
    o = main(g)
    print(g.shape, g.shape, int(np.count_nonzero(g != o)))
