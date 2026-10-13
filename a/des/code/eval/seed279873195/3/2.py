import random
from collections import deque

import numpy as np


BG = 0
FRAME_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _normalize(points: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(points)
    norm = {(r - r0, c - c0) for r, c in points}
    return norm, (r1 - r0 + 1, c1 - c0 + 1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return mask[:0, :0]
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _is_connected(points: set[tuple[int, int]]) -> bool:
    if not points:
        return False
    start = next(iter(points))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nb = (r + dr, c + dc)
            if nb in points and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return len(seen) == len(points)


def _fills_bbox(points: set[tuple[int, int]]) -> bool:
    _, (h, w) = _normalize(points)
    return len(points) == h * w


def _candidate_rectangle_dims(mask_a: np.ndarray, mask_b: np.ndarray) -> set[tuple[int, int]]:
    pts_a = {(int(r), int(c)) for r, c in np.argwhere(mask_a)}
    pts_b = {(int(r), int(c)) for r, c in np.argwhere(mask_b)}
    h1, w1 = mask_a.shape
    h2, w2 = mask_b.shape
    candidates: set[tuple[int, int]] = set()
    for dr in range(-h2, h1 + 1):
        for dc in range(-w2, w1 + 1):
            shifted_b = {(r + dr, c + dc) for r, c in pts_b}
            if pts_a & shifted_b:
                continue
            union = pts_a | shifted_b
            rows = [r for r, _ in union]
            cols = [c for _, c in union]
            hh = max(rows) - min(rows) + 1
            ww = max(cols) - min(cols) + 1
            if hh * ww == len(union):
                candidates.add((hh, ww))
    return candidates


def _paired_rectangle_dims(comp_a: list[tuple[int, int]], comp_b: list[tuple[int, int]]) -> tuple[int, int]:
    pts_a, shape_a = _normalize(comp_a)
    pts_b, shape_b = _normalize(comp_b)
    mask_a = np.zeros(shape_a, dtype=bool)
    mask_b = np.zeros(shape_b, dtype=bool)
    for r, c in pts_a:
        mask_a[r, c] = True
    for r, c in pts_b:
        mask_b[r, c] = True
    candidates = _candidate_rectangle_dims(mask_a, mask_b)
    if not candidates:
        raise AssertionError("pair does not assemble into a rectangle")
    return min(candidates, key=lambda hw: (hw[0] * hw[1], hw[0], hw[1]))


def _find_target_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int], list[tuple[int, int]]]:
    best = None
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1] == color
            if not sub[0, :].all() or not sub[-1, :].all() or not sub[:, 0].all() or not sub[:, -1].all():
                continue
            divider_rows = [rr for rr in range(1, h - 1) if sub[rr, :].all()]
            allowed = np.zeros_like(sub, dtype=bool)
            allowed[0, :] = True
            allowed[-1, :] = True
            allowed[:, 0] = True
            allowed[:, -1] = True
            for rr in divider_rows:
                allowed[rr, :] = True
            if not np.array_equal(sub, allowed):
                continue
            bounds = [0] + divider_rows + [h - 1]
            cells = [(bounds[i] + 1, bounds[i + 1] - 1) for i in range(len(bounds) - 1)]
            if any(lo > hi for lo, hi in cells):
                continue
            heights = [hi - lo + 1 for lo, hi in cells]
            if len(set(heights)) != 1:
                continue
            score = (len(cells), h * w, r1, w)
            if best is None or score > best[0]:
                cell_rows = [(r0 + lo, r0 + hi) for lo, hi in cells]
                best = (score, color, (r0, c0, r1, c1), cell_rows)
    if best is None:
        raise AssertionError("target frame not found")
    return best[1], best[2], best[3]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frame_color, (r0, c0, r1, c1), cell_rows = _find_target_frame(g, bg)

    specs: list[tuple[int, int, int, int]] = []
    for color in np.unique(g):
        color = int(color)
        if color in (bg, frame_color):
            continue
        comps = _components(g == color)
        if len(comps) != 2:
            raise AssertionError("each fragment color must have exactly two components")
        h, w = _paired_rectangle_dims(comps[0], comps[1])
        specs.append((h * w, h, w, color))

    if len(specs) != len(cell_rows):
        raise AssertionError("number of colors must match number of target cells")

    out = g.copy()
    inner_left = c0 + 1
    inner_width = c1 - c0 - 1
    for idx, (_area, rect_h, rect_w, color) in enumerate(sorted(specs, key=lambda x: (x[0], x[1], x[2], x[3]))):
        cell_top, cell_bottom = cell_rows[idx]
        cell_height = cell_bottom - cell_top + 1
        draw_h = min(rect_h, cell_height)
        draw_w = min(rect_w, inner_width)
        if draw_h > 0 and draw_w > 0:
            out[cell_top : cell_top + draw_h, inner_left : inner_left + draw_w] = color
    return out


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = color
    return patch


def _random_connected_subset(h: int, w: int, size: int, rng: random.Random) -> set[tuple[int, int]] | None:
    all_cells = [(r, c) for r in range(h) for c in range(w)]
    subset = {rng.choice(all_cells)}
    while len(subset) < size:
        frontier = []
        for r, c in subset:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in subset:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        subset.add(rng.choice(frontier))
    return subset


def _make_pair_masks(rect_h: int, rect_w: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
    area = rect_h * rect_w
    cells = {(r, c) for r in range(rect_h) for c in range(rect_w)}
    for _ in range(300):
        size_a = rng.randint(2, area - 2)
        part_a = _random_connected_subset(rect_h, rect_w, size_a, rng)
        if part_a is None:
            continue
        part_b = cells - part_a
        if not _is_connected(part_b):
            continue
        if _fills_bbox(part_a) and _fills_bbox(part_b):
            continue

        full_a = np.zeros((rect_h, rect_w), dtype=bool)
        full_b = np.zeros((rect_h, rect_w), dtype=bool)
        for r, c in part_a:
            full_a[r, c] = True
        for r, c in part_b:
            full_b[r, c] = True

        k = rng.randrange(4)
        if k:
            full_a = np.rot90(full_a, k)
            full_b = np.rot90(full_b, k)
        if rng.random() < 0.5:
            full_a = np.fliplr(full_a)
            full_b = np.fliplr(full_b)

        rect_shape = full_a.shape
        mask_a = _crop_mask(full_a)
        mask_b = _crop_mask(full_b)
        if _candidate_rectangle_dims(mask_a, mask_b) == {rect_shape}:
            return mask_a, mask_b, rect_shape
    return None


def _choose_frame_size(
    rect_dims: list[tuple[int, int]], rng: random.Random
) -> tuple[int, int]:
    heights = [h for h, _ in rect_dims]
    widths = [w for _, w in rect_dims]
    max_h = max(heights)
    max_w = max(widths)

    mixed_heights = [v for v in range(1, max_h + 1) if any(h > v for h in heights) and any(h <= v for h in heights)]
    mixed_widths = [v for v in range(2, max_w + 1) if any(w > v for w in widths) and any(w <= v for w in widths)]

    modes = ["fit"]
    if max_h > 1:
        modes.append("clip_h")
    if max_w > 2:
        modes.append("clip_w")
    if max_h > 1 and max_w > 2:
        modes.append("clip_both")
    if mixed_heights or mixed_widths:
        modes.append("mixed")

    mode = rng.choice(modes)
    if mode == "fit":
        return rng.randint(max_h, max_h + 2), rng.randint(max_w, max_w + 3)
    if mode == "clip_h":
        return rng.randint(1, max_h - 1), rng.randint(max_w, max_w + 3)
    if mode == "clip_w":
        return rng.randint(max_h, max_h + 2), rng.randint(2, max_w - 1)
    if mode == "clip_both":
        return rng.randint(1, max_h - 1), rng.randint(2, max_w - 1)
    if mixed_heights and mixed_widths and rng.random() < 0.5:
        return rng.choice(mixed_heights), rng.choice(mixed_widths)
    if mixed_heights:
        return rng.choice(mixed_heights), rng.randint(max_w, max_w + 3)
    if mixed_widths:
        return rng.randint(max_h, max_h + 2), rng.choice(mixed_widths)
    return rng.randint(max_h, max_h + 2), rng.randint(max_w, max_w + 3)


def _draw_frame(grid: np.ndarray, top: int, left: int, cell_height: int, inner_width: int, rows: int) -> None:
    frame_h = 2 + rows * cell_height + (rows - 1)
    frame_w = inner_width + 2
    bottom = top + frame_h - 1
    right = left + frame_w - 1
    grid[top, left : right + 1] = FRAME_COLOR
    grid[bottom, left : right + 1] = FRAME_COLOR
    grid[top : bottom + 1, left] = FRAME_COLOR
    grid[top : bottom + 1, right] = FRAME_COLOR
    for idx in range(1, rows):
        divider = top + idx * cell_height + (idx - 1) + 1
        grid[divider, left : right + 1] = FRAME_COLOR


def _place_patch(
    grid: np.ndarray, patch: np.ndarray, blocked: np.ndarray, rng: random.Random
) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    if h > gh or w > gw:
        return False
    for _ in range(250):
        r = rng.randint(0, gh - h)
        c = rng.randint(0, gw - w)
        if blocked[r : r + h, c : c + w].any():
            continue
        rr0 = max(0, r - 1)
        rr1 = min(gh, r + h + 1)
        cc0 = max(0, c - 1)
        cc1 = min(gw, c + w + 1)
        blocked[rr0:rr1, cc0:cc1] = True
        sub = patch != BG
        grid[r : r + h, c : c + w][sub] = patch[sub]
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    color_count = rng.randint(2, 5)
    colors = rng.sample(PALETTE, color_count)

    pair_specs = []
    used_areas: set[int] = set()
    for color in colors:
        for _ in range(100):
            rect_h = rng.randint(2, 4)
            rect_w = rng.randint(2, 6)
            area = rect_h * rect_w
            if area in used_areas:
                continue
            pair = _make_pair_masks(rect_h, rect_w, rng)
            if pair is None:
                continue
            mask_a, mask_b, rect_shape = pair
            pair_specs.append((color, mask_a, mask_b, rect_shape))
            used_areas.add(area)
            break
        else:
            return None

    rect_dims = [shape for _color, _a, _b, shape in pair_specs]
    cell_height, inner_width = _choose_frame_size(rect_dims, rng)
    frame_h = 2 + color_count * cell_height + (color_count - 1)
    frame_w = inner_width + 2

    piece_bboxes = [mask.shape for _color, a, b, _shape in pair_specs for mask in (a, b)]
    max_piece_h = max(h for h, _ in piece_bboxes)
    max_piece_w = max(w for _, w in piece_bboxes)

    top_h = rng.randint(max(8, color_count + 5), max(12, color_count + 9))
    total_h = top_h + frame_h
    min_w = max(frame_w + 2, max_piece_w + 8, 16)
    total_w = rng.randint(min_w, min_w + 8)
    if total_h < max_piece_h + 3:
        return None

    grid = np.full((total_h, total_w), BG, dtype=int)
    frame_top = total_h - frame_h
    frame_left = rng.randint(1, total_w - frame_w - 1)
    _draw_frame(grid, frame_top, frame_left, cell_height, inner_width, color_count)

    blocked = np.zeros_like(grid, dtype=bool)
    blocked[max(0, frame_top - 1) :, :] = True

    patches = []
    for color, mask_a, mask_b, _shape in pair_specs:
        patches.append(_mask_to_patch(mask_a, color))
        patches.append(_mask_to_patch(mask_b, color))
    rng.shuffle(patches)
    patches.sort(key=lambda patch: patch.shape[0] * patch.shape[1], reverse=True)

    for patch in patches:
        if not _place_patch(grid, patch, blocked, rng):
            return None

    for color in colors:
        if len(_components(grid == color)) != 2:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
