import random

import numpy as np


BG = 0
DIR_BY_SIDE = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_H_RANGE = (16, 28)
GRID_W_RANGE = (16, 28)
FRAME_COUNT_OPTIONS = (3, 4)
FRAME_H_RANGE = (4, 7)
FRAME_W_RANGE = (4, 7)
FRAME_PAD = 1
PLACE_RETRIES = 200
MAX_GENERATION_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _side_of_gap(bbox: tuple[int, int, int, int], gap: tuple[int, int]) -> str | None:
    r0, c0, r1, c1 = bbox
    gr, gc = gap
    if gr == r0 and c0 < gc < c1:
        return "top"
    if gr == r1 and c0 < gc < c1:
        return "bottom"
    if gc == c0 and r0 < gr < r1:
        return "left"
    if gc == c1 and r0 < gr < r1:
        return "right"
    return None


def _identify_frame(
    grid: np.ndarray, border_color: int, coords: list[tuple[int, int]]
) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    coord_set = set(coords)
    perimeter: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        perimeter.add((r0, c))
        perimeter.add((r1, c))
    for r in range(r0, r1 + 1):
        perimeter.add((r, c0))
        perimeter.add((r, c1))

    if not coord_set.issubset(perimeter):
        return None

    missing = [cell for cell in perimeter if cell not in coord_set]
    if len(missing) != 1 or len(coord_set) != len(perimeter) - 1:
        return None

    gap = missing[0]
    side = _side_of_gap((r0, c0, r1, c1), gap)
    if side is None or int(grid[gap]) != BG:
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0:
        return None
    fill_colors = np.unique(interior)
    if fill_colors.size != 1:
        return None
    fill_color = int(fill_colors[0])
    if fill_color in (BG, border_color):
        return None

    for cell in perimeter:
        if cell == gap:
            continue
        if int(grid[cell]) != border_color:
            return None

    return {
        "border": int(border_color),
        "fill": fill_color,
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap": (int(gap[0]), int(gap[1])),
        "side": side,
        "dir": DIR_BY_SIDE[side],
    }


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    colors = [int(c) for c in np.unique(g) if int(c) != BG]
    frames: list[dict[str, object]] = []
    for color in colors:
        for comp in _components(g == color):
            frame = _identify_frame(g, color, comp)
            if frame is not None:
                frames.append(frame)

    frames.sort(key=lambda fr: (fr["bbox"][0], fr["bbox"][1], fr["border"]))
    assert len(frames) >= 3, "expected at least three gapped frames"

    occupied = np.zeros_like(g, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        gap_r, gap_c = frame["gap"]
        frame_mask = np.ones((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        frame_mask[gap_r - r0, gap_c - c0] = False
        if np.any(occupied[r0 : r1 + 1, c0 : c1 + 1] & frame_mask):
            raise AssertionError("frame regions must not overlap")
        occupied[r0 : r1 + 1, c0 : c1 + 1] |= frame_mask

    assert np.array_equal(occupied, g != BG), "all non-black cells must belong to a frame"
    return frames


def _frame_index_at(
    frames: list[dict[str, object]], row: int, col: int, exclude: int
) -> int | None:
    for idx, frame in enumerate(frames):
        if idx == exclude:
            continue
        r0, c0, r1, c1 = frame["bbox"]
        if r0 <= row <= r1 and c0 <= col <= c1:
            return idx
    return None


def _trace_ray(
    frames: list[dict[str, object]], src_idx: int, shape: tuple[int, int]
) -> tuple[list[tuple[int, int]], int | None]:
    h, w = shape
    gap_r, gap_c = frames[src_idx]["gap"]
    dr, dc = frames[src_idx]["dir"]
    row = gap_r + dr
    col = gap_c + dc
    ray: list[tuple[int, int]] = []

    while 0 <= row < h and 0 <= col < w:
        target_idx = _frame_index_at(frames, row, col, exclude=src_idx)
        if target_idx is not None:
            return ray, target_idx
        ray.append((row, col))
        row += dr
        col += dc

    return ray, None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shoot fill-colored rays out of the gaps of rectangular frames and recolor any struck frame interior.

    Input structure:
    - The background is black.
    - The grid contains at least three one-pixel-thick rectangular frames with distinct border colors.
    - Each frame has a solid interior of a different color, and exactly one non-corner border cell is replaced by black to make a gap.

    Rule:
    - Detect the side that contains each frame's gap; that side determines whether the frame shoots upward, downward, leftward, or rightward.
    - Starting just outside the gap, draw a one-pixel-wide straight ray in the frame's interior color across black cells.
    - Stop the ray immediately before it would enter another frame; if it reaches the grid edge first, keep drawing up to the edge.
    - When a ray is stopped by another frame, recolor the whole interior of that target frame to the source frame's interior color while leaving all borders unchanged.

    Key insight:
    - The black gap is a direction cue, while the source color comes from the frame fill rather than the border; the ray only paints the background and transfers its color to the next frame it reaches.
    """
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    out = g.copy()

    for src_idx, frame in enumerate(frames):
        ray_cells, target_idx = _trace_ray(frames, src_idx, g.shape)
        fill_color = int(frame["fill"])
        for row, col in ray_cells:
            out[row, col] = fill_color
        if target_idx is not None:
            r0, c0, r1, c1 = frames[target_idx]["bbox"]
            out[r0 + 1 : r1, c0 + 1 : c1] = fill_color

    return out


def _boxes_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _trace_layout_ray(
    boxes: list[tuple[int, int, int, int]],
    src_idx: int,
    gap: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], int | None]:
    h, w = shape
    dr, dc = direction
    row = gap[0] + dr
    col = gap[1] + dc
    ray: list[tuple[int, int]] = []
    while 0 <= row < h and 0 <= col < w:
        for idx, (r0, c0, r1, c1) in enumerate(boxes):
            if idx != src_idx and r0 <= row <= r1 and c0 <= col <= c1:
                return ray, idx
        ray.append((row, col))
        row += dr
        col += dc
    return ray, None


def _gap_candidates_for_box(
    boxes: list[tuple[int, int, int, int]], idx: int, shape: tuple[int, int]
) -> list[dict[str, object]]:
    r0, c0, r1, c1 = boxes[idx]
    candidates: list[dict[str, object]] = []

    for col in range(c0 + 1, c1):
        gap = (r0, col)
        ray, target = _trace_layout_ray(boxes, idx, gap, DIR_BY_SIDE["top"], shape)
        if ray:
            candidates.append(
                {"gap": gap, "side": "top", "dir": DIR_BY_SIDE["top"], "ray": ray, "target": target}
            )

        gap = (r1, col)
        ray, target = _trace_layout_ray(boxes, idx, gap, DIR_BY_SIDE["bottom"], shape)
        if ray:
            candidates.append(
                {
                    "gap": gap,
                    "side": "bottom",
                    "dir": DIR_BY_SIDE["bottom"],
                    "ray": ray,
                    "target": target,
                }
            )

    for row in range(r0 + 1, r1):
        gap = (row, c0)
        ray, target = _trace_layout_ray(boxes, idx, gap, DIR_BY_SIDE["left"], shape)
        if ray:
            candidates.append(
                {
                    "gap": gap,
                    "side": "left",
                    "dir": DIR_BY_SIDE["left"],
                    "ray": ray,
                    "target": target,
                }
            )

        gap = (row, c1)
        ray, target = _trace_layout_ray(boxes, idx, gap, DIR_BY_SIDE["right"], shape)
        if ray:
            candidates.append(
                {
                    "gap": gap,
                    "side": "right",
                    "dir": DIR_BY_SIDE["right"],
                    "ray": ray,
                    "target": target,
                }
            )

    return candidates


def _choose_gap_assignment(
    rng: random.Random,
    boxes: list[tuple[int, int, int, int]],
    shape: tuple[int, int],
    desired_hits: int,
) -> list[dict[str, object]] | None:
    candidates_by_frame = [_gap_candidates_for_box(boxes, idx, shape) for idx in range(len(boxes))]
    if sum(any(c["target"] is not None for c in choices) for choices in candidates_by_frame) < desired_hits:
        return None

    order = sorted(range(len(boxes)), key=lambda idx: len(candidates_by_frame[idx]))
    chosen: list[dict[str, object] | None] = [None] * len(boxes)
    used_targets: set[int] = set()
    used_cells: set[tuple[int, int]] = set()

    def search(pos: int, hit_count: int) -> bool:
        if hit_count > desired_hits:
            return False
        remaining = len(order) - pos
        if hit_count + remaining < desired_hits:
            return False
        if pos == len(order):
            return hit_count == desired_hits

        frame_idx = order[pos]
        options = list(candidates_by_frame[frame_idx])
        rng.shuffle(options)

        options.sort(
            key=lambda cand: (
                0 if (cand["target"] is not None) == (hit_count < desired_hits) else 1,
                len(cand["ray"]),
            )
        )

        for cand in options:
            target = cand["target"]
            if target is not None and target in used_targets:
                continue
            ray_cells = set(cand["ray"])
            if ray_cells & used_cells:
                continue

            chosen[frame_idx] = cand
            if target is not None:
                used_targets.add(target)
            used_cells.update(ray_cells)

            if search(pos + 1, hit_count + int(target is not None)):
                return True

            used_cells.difference_update(ray_cells)
            if target is not None:
                used_targets.remove(target)
            chosen[frame_idx] = None

        return False

    if not search(0, 0):
        return None
    return [cand for cand in chosen if cand is not None]


def _place_boxes(
    rng: random.Random, h: int, w: int, n_frames: int
) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(n_frames):
        placed = False
        for _ in range(PLACE_RETRIES):
            fh = rng.randint(*FRAME_H_RANGE)
            fw = rng.randint(*FRAME_W_RANGE)
            if fh + 2 > h or fw + 2 > w:
                continue
            r0 = rng.randint(1, h - fh - 1)
            c0 = rng.randint(1, w - fw - 1)
            box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if all(_boxes_separated(box, other, FRAME_PAD) for other in boxes):
                boxes.append(box)
                placed = True
                break
        if not placed:
            return None
    return boxes


def _draw_frame(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    border_color: int,
    fill_color: int,
    gap: tuple[int, int],
) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = fill_color
    grid[r0, c0 : c1 + 1] = border_color
    grid[r1, c0 : c1 + 1] = border_color
    grid[r0 : r1 + 1, c0] = border_color
    grid[r0 : r1 + 1, c1] = border_color
    grid[gap] = BG


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    n_frames = rng.choice(FRAME_COUNT_OPTIONS)
    boxes = _place_boxes(rng, h, w, n_frames)
    if boxes is None:
        return None

    desired_hits = rng.randint(1, n_frames - 1)
    gaps = _choose_gap_assignment(rng, boxes, (h, w), desired_hits)
    if gaps is None:
        return None

    palette = list(range(1, 10))
    rng.shuffle(palette)
    if 2 * n_frames > len(palette):
        return None
    border_colors = palette[:n_frames]
    fill_colors = palette[n_frames : 2 * n_frames]

    grid = np.full((h, w), BG, dtype=int)
    for idx, box in enumerate(boxes):
        _draw_frame(grid, box, border_colors[idx], fill_colors[idx], gaps[idx]["gap"])

    frames = _find_frames(grid)
    if len(frames) != n_frames:
        return None

    hit_count = 0
    miss_count = 0
    used_rays: set[tuple[int, int]] = set()
    for src_idx in range(n_frames):
        ray, target_idx = _trace_ray(frames, src_idx, grid.shape)
        if set(ray) & used_rays:
            return None
        used_rays.update(ray)
        if target_idx is None:
            miss_count += 1
        else:
            hit_count += 1

    if hit_count != desired_hits or miss_count == 0:
        return None

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gapped-frame ray puzzle")
