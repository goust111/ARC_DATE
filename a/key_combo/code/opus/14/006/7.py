from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
GREY = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NON_SPECIAL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

TRIANGLE_TEMPLATES = {
    "iso": np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 2, 1, 1],
        ],
        dtype=int,
    ),
    "right_left": np.array(
        [
            [0, 1, 0],
            [0, 1, 1],
            [1, 2, 1],
        ],
        dtype=int,
    ),
    "right_right": np.array(
        [
            [0, 1, 0],
            [1, 1, 0],
            [1, 2, 1],
        ],
        dtype=int,
    ),
}

ROTATION_K = {"up": 0, "left": 1, "down": 2, "right": 3}
GEN_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_rectangles(grid: np.ndarray) -> list[dict]:
    rects = []
    for comp in _components(grid == GREY):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5:
            continue
        want = {(r, c0) for r in range(r0, r1 + 1)}
        want |= {(r, c1) for r in range(r0, r1 + 1)}
        want |= {(r0, c) for c in range(c0, c1 + 1)}
        want |= {(r1, c) for c in range(c0, c1 + 1)}
        if set(comp) != want:
            continue
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0 or np.any(interior == GREY):
            continue
        rects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "interior": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
            }
        )
    return rects


def _analyze_triangle(grid: np.ndarray) -> tuple[int, int, tuple[int, int], tuple[int, int]]:
    counts = Counter(
        int(v)
        for v in grid.ravel()
        if int(v) not in (BLACK, GREY)
    )
    assert len(counts) == 2
    marker_colors = [color for color, count in counts.items() if count == 1]
    assert len(marker_colors) == 1
    marker_color = marker_colors[0]
    body_colors = [color for color in counts if color != marker_color]
    assert len(body_colors) == 1
    body_color = body_colors[0]

    triangle_mask = (grid == marker_color) | (grid == body_color)
    marker = tuple(map(int, np.argwhere(grid == marker_color)[0]))

    exposed_candidates = []
    for dr, dc in DIRS4:
        nr, nc = marker[0] + dr, marker[1] + dc
        if _in_bounds(grid.shape, nr, nc) and triangle_mask[nr, nc]:
            continue
        score = 0
        rr, cc = marker[0] - dr, marker[1] - dc
        while _in_bounds(grid.shape, rr, cc) and triangle_mask[rr, cc]:
            score += 1
            rr -= dr
            cc -= dc
        if score:
            exposed_candidates.append((score, (dr, dc)))
    assert exposed_candidates

    exposed = max(exposed_candidates, key=lambda item: item[0])[1]
    direction = (-exposed[0], -exposed[1])

    coords = np.argwhere(triangle_mask)
    projection = coords[:, 0] * direction[0] + coords[:, 1] * direction[1]
    best = coords[projection == projection.max()]
    if len(best) > 1:
        if direction[0] == 0:
            idx = np.argmin(np.abs(best[:, 0] - marker[0]))
        else:
            idx = np.argmin(np.abs(best[:, 1] - marker[1]))
        tip = tuple(map(int, best[idx]))
    else:
        tip = tuple(map(int, best[0]))

    return body_color, marker_color, direction, tip


def _trace_ray(grid: np.ndarray, marker_color: int, direction: tuple[int, int], tip: tuple[int, int]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    out = grid.copy()
    path = []
    r, c = tip[0] + direction[0], tip[1] + direction[1]
    while _in_bounds(grid.shape, r, c):
        path.append((r, c))
        if grid[r, c] == BLACK:
            out[r, c] = marker_color
        r += direction[0]
        c += direction[1]
    return out, path


def _hit_order(path: list[tuple[int, int]], rects: list[dict]) -> list[dict]:
    ordered = []
    for rect in rects:
        r0, c0, r1, c1 = rect["interior"]
        first = None
        for idx, (r, c) in enumerate(path):
            if r0 <= r <= r1 and c0 <= c <= c1:
                first = idx
                break
        if first is not None:
            ordered.append((first, rect))
    ordered.sort(key=lambda item: item[0])
    return [rect for _, rect in ordered]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a triangle-directed ray and alternately fill each crossed rectangular frame interior.

    Input structure:
    - A black background contains one filled triangle in a single body color.
    - The triangle includes exactly one differently colored base marker pixel, and the grid also contains several single-pixel-wide grey rectangular frames with black interiors.
    - The triangle may face any cardinal direction, while the frames are axis-aligned and spatially separated.

    Rule:
    - Find the marker pixel and use its exposed outer side to infer the triangle base direction; the ray points in the opposite direction, from the marker through the triangle tip and onward.
    - Draw that ray from the cell after the tip to the grid edge using the marker color, but never overwrite grey frame borders.
    - For every frame whose interior is entered by the ray, fill its black interior in encounter order: 1st, 3rd, 5th, ... with the marker color; 2nd, 4th, 6th, ... with the triangle body color.

    Key insight:
    - The odd colored base marker is both the direction cue and one of the two alternating fill colors, so geometry selects which frames participate and their order selects which color they receive.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    body_color, marker_color, direction, tip = _analyze_triangle(g)
    rects = _find_rectangles(g)

    out, path = _trace_ray(g, marker_color, direction, tip)
    for rank, rect in enumerate(_hit_order(path, rects)):
        fill_color = marker_color if rank % 2 == 0 else body_color
        r0, c0, r1, c1 = rect["interior"]
        out[r0 : r1 + 1, c0 : c1 + 1] = fill_color
    return out


def _rotated_triangle(direction: str, variant: str, body_color: int, marker_color: int) -> np.ndarray:
    sprite = TRIANGLE_TEMPLATES[variant].copy()
    sprite[sprite == 1] = body_color
    sprite[sprite == 2] = marker_color
    return np.rot90(sprite, k=ROTATION_K[direction])


def _can_place_with_gap(grid: np.ndarray, r0: int, c0: int, h: int, w: int, gap: int = 1) -> bool:
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(grid.shape[0], r0 + h + gap)
    cc1 = min(grid.shape[1], c0 + w + gap)
    return np.all(grid[rr0:rr1, cc0:cc1] == BLACK)


def _stamp_triangle(grid: np.ndarray, sprite: np.ndarray, direction: str) -> bool:
    h, w = sprite.shape
    H, W = grid.shape
    if direction == "right":
        c_lo, c_hi = 1, max(1, W // 3 - w)
        r_lo, r_hi = 1, H - h - 2
    elif direction == "left":
        c_lo, c_hi = max(1, 2 * W // 3 - w), W - w - 2
        r_lo, r_hi = 1, H - h - 2
    elif direction == "down":
        r_lo, r_hi = 1, max(1, H // 3 - h)
        c_lo, c_hi = 1, W - w - 2
    else:
        r_lo, r_hi = max(1, 2 * H // 3 - h), H - h - 2
        c_lo, c_hi = 1, W - w - 2

    if r_lo > r_hi or c_lo > c_hi:
        return False
    r0 = random.randint(r_lo, r_hi)
    c0 = random.randint(c_lo, c_hi)
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[sprite != BLACK] = sprite[sprite != BLACK]
    return True


def _draw_frame(grid: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int) -> dict:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    return {"bbox": (r0, c0, r1, c1), "interior": (r0 + 1, c0 + 1, r1 - 1, c1 - 1)}


def _sample_on_ray_frames(grid: np.ndarray, tip: tuple[int, int], direction: tuple[int, int], count: int) -> list[dict] | None:
    frames = []
    cursor_r, cursor_c = tip
    for _ in range(count):
        inner_h = random.randint(3, 4)
        inner_w = random.randint(3, 4)
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        placed = None
        for _try in range(60):
            if direction == (0, 1):
                r_lo = max(0, tip[0] - inner_h)
                r_hi = min(grid.shape[0] - outer_h, tip[0] - 1)
                c_lo = cursor_c + 2
                c_hi = grid.shape[1] - outer_w
                if r_lo > r_hi or c_lo > c_hi:
                    return None
                r0 = random.randint(r_lo, r_hi)
                c0 = random.randint(c_lo, c_hi)
            elif direction == (0, -1):
                r_lo = max(0, tip[0] - inner_h)
                r_hi = min(grid.shape[0] - outer_h, tip[0] - 1)
                c_lo = 0
                c_hi = cursor_c - outer_w - 1
                if r_lo > r_hi or c_lo > c_hi:
                    return None
                r0 = random.randint(r_lo, r_hi)
                c0 = random.randint(c_lo, c_hi)
            elif direction == (1, 0):
                c_lo = max(0, tip[1] - inner_w)
                c_hi = min(grid.shape[1] - outer_w, tip[1] - 1)
                r_lo = cursor_r + 2
                r_hi = grid.shape[0] - outer_h
                if r_lo > r_hi or c_lo > c_hi:
                    return None
                r0 = random.randint(r_lo, r_hi)
                c0 = random.randint(c_lo, c_hi)
            else:
                c_lo = max(0, tip[1] - inner_w)
                c_hi = min(grid.shape[1] - outer_w, tip[1] - 1)
                r_lo = 0
                r_hi = cursor_r - outer_h - 1
                if r_lo > r_hi or c_lo > c_hi:
                    return None
                r0 = random.randint(r_lo, r_hi)
                c0 = random.randint(c_lo, c_hi)

            if not _can_place_with_gap(grid, r0, c0, outer_h, outer_w):
                continue
            placed = _draw_frame(grid, r0, c0, inner_h, inner_w)
            break
        if placed is None:
            return None

        frames.append(placed)
        if direction == (0, 1):
            cursor_c = placed["bbox"][3]
        elif direction == (0, -1):
            cursor_c = placed["bbox"][1]
        elif direction == (1, 0):
            cursor_r = placed["bbox"][2]
        else:
            cursor_r = placed["bbox"][0]
    return frames


def _ray_crosses_rect(tip: tuple[int, int], direction: tuple[int, int], rect: dict) -> bool:
    r0, c0, r1, c1 = rect["bbox"]
    if direction == (0, 1):
        return r0 < tip[0] < r1 and c0 > tip[1]
    if direction == (0, -1):
        return r0 < tip[0] < r1 and c1 < tip[1]
    if direction == (1, 0):
        return c0 < tip[1] < c1 and r0 > tip[0]
    return c0 < tip[1] < c1 and r1 < tip[0]


def _sample_off_ray_frames(
    grid: np.ndarray,
    tip: tuple[int, int],
    direction: tuple[int, int],
    count: int,
) -> list[dict] | None:
    frames = []
    for _ in range(count):
        placed = None
        for _try in range(120):
            inner_h = random.randint(3, 4)
            inner_w = random.randint(3, 4)
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            if grid.shape[0] - outer_h < 0 or grid.shape[1] - outer_w < 0:
                return None
            r0 = random.randint(0, grid.shape[0] - outer_h)
            c0 = random.randint(0, grid.shape[1] - outer_w)
            if not _can_place_with_gap(grid, r0, c0, outer_h, outer_w):
                continue
            candidate = {"bbox": (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)}
            if _ray_crosses_rect(tip, direction, candidate):
                continue
            placed = _draw_frame(grid, r0, c0, inner_h, inner_w)
            break
        if placed is None:
            return None
        frames.append(placed)
    return frames


def _build_attempt() -> np.ndarray | None:
    direction_name = random.choice(["up", "down", "left", "right"])
    if direction_name in ("left", "right"):
        height = random.randint(14, 22)
        width = random.randint(22, 30)
    else:
        height = random.randint(22, 30)
        width = random.randint(14, 22)

    grid = np.zeros((height, width), dtype=int)
    body_color, marker_color = random.sample(NON_SPECIAL_COLORS, 2)
    variant = random.choice(list(TRIANGLE_TEMPLATES))
    sprite = _rotated_triangle(direction_name, variant, body_color, marker_color)
    if not _stamp_triangle(grid, sprite, direction_name):
        return None

    try:
        _, _, direction, tip = _analyze_triangle(grid)
    except Exception:
        return None

    hit_count = random.randint(2, 3)
    total_frames = hit_count + random.randint(1, min(2, 4 - hit_count))
    hit_frames = _sample_on_ray_frames(grid, tip, direction, hit_count)
    if hit_frames is None:
        return None
    miss_frames = _sample_off_ray_frames(grid, tip, direction, total_frames - hit_count)
    if miss_frames is None:
        return None

    rects = _find_rectangles(grid)
    if len(rects) != total_frames:
        return None

    try:
        _, path = _trace_ray(grid, marker_color, direction, tip)
        ordered = _hit_order(path, rects)
        if len(ordered) != hit_count:
            return None
        if len(rects) - len(ordered) < 1:
            return None
        output = main(grid)
    except Exception:
        return None

    if output.shape != grid.shape or np.array_equal(output, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
