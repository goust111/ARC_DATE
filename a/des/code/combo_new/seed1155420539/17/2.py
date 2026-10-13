from __future__ import annotations

from collections import deque
import random

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components_monochrome(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            min_r = max_r = r
            min_c = max_c = c
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                min_r = min(min_r, cr)
                max_r = max(max_r, cr)
                min_c = min(min_c, cc)
                max_c = max(max_c, cc)
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min_r, min_c, max_r, max_c),
                    "area": len(coords),
                }
            )
    return comps


def _classify_fence(comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in comp["coords"]:
        mask[r - r0, c - c0] = True

    if np.any(mask[1:-1, 1:-1]):
        return None
    if not (mask[0, 0] and mask[0, -1] and mask[-1, 0] and mask[-1, -1]):
        return None

    perimeter = 2 * h + 2 * w - 4
    missing = perimeter - int(mask.sum())
    if missing not in (2, 3):
        return None

    side_gaps: dict[str, tuple[int, int]] = {}
    for side, arr in (
        ("top", mask[0, 1:-1]),
        ("bottom", mask[-1, 1:-1]),
        ("left", mask[1:-1, 0]),
        ("right", mask[1:-1, -1]),
    ):
        zeros = np.flatnonzero(~arr)
        if len(zeros) == 0:
            continue
        if len(zeros) not in (2, 3):
            return None
        if int(zeros[-1]) - int(zeros[0]) + 1 != len(zeros):
            return None
        side_gaps[side] = (int(zeros[0]) + 1, int(zeros[-1]) + 1)

    if len(side_gaps) != 1:
        return None

    direction = next(iter(side_gaps))
    start, end = side_gaps[direction]
    if direction in ("left", "right"):
        gap_range = (r0 + start, r0 + end)
        axis = "row"
    else:
        gap_range = (c0 + start, c0 + end)
        axis = "col"

    return {
        "color": int(comp["color"]),
        "bbox": (r0, c0, r1, c1),
        "direction": direction,
        "gap_range": gap_range,
        "gap_axis": axis,
    }


def _find_fences(grid: np.ndarray, bg: int) -> list[dict]:
    fences = []
    for comp in _components_monochrome(grid, bg):
        info = _classify_fence(comp)
        if info is not None:
            fences.append(info)
    fences.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return fences


def _count_binary_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            count += 1
            queue = deque([(r, c)])
            seen[r, c] = True
            while queue:
                cr, cc = queue.popleft()
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if seen[nr, nc] or not bool(mask[nr, nc]):
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))
    return count


def _extract_object_pixels(grid: np.ndarray, fence: dict, bg: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = fence["bbox"]
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    obj_mask = inner != bg
    coords = np.argwhere(obj_mask)
    assert len(coords) > 0, "each fence must contain one interior object"
    assert _count_binary_components(obj_mask) == 1, "interior object must be connected"
    return [(r0 + 1 + int(r), c0 + 1 + int(c)) for r, c in coords]


def _find_target_fence(source: dict, fences: list[dict]) -> dict:
    r0, c0, r1, c1 = source["bbox"]
    g0, g1 = source["gap_range"]
    direction = source["direction"]
    candidates: list[tuple[tuple[int, int, int], dict]] = []

    for other in fences:
        if other is source:
            continue
        or0, oc0, or1, oc1 = other["bbox"]
        if direction == "right":
            if oc0 <= c1 or not _intervals_overlap(or0, or1, g0, g1):
                continue
            key = (oc0 - c1, or0, oc0)
        elif direction == "left":
            if oc1 >= c0 or not _intervals_overlap(or0, or1, g0, g1):
                continue
            key = (c0 - oc1, or0, oc0)
        elif direction == "bottom":
            if or0 <= r1 or not _intervals_overlap(oc0, oc1, g0, g1):
                continue
            key = (or0 - r1, or0, oc0)
        else:
            assert direction == "top"
            if or1 >= r0 or not _intervals_overlap(oc0, oc1, g0, g1):
                continue
            key = (r0 - or1, or0, oc0)
        candidates.append((key, other))

    assert candidates, "each gap direction must lead to another fence"
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _translation_delta(direction: str, target_bbox: tuple[int, int, int, int], obj_pixels: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _c in obj_pixels]
    cols = [c for _r, c in obj_pixels]
    tr0, tc0, tr1, tc1 = target_bbox
    if direction == "right":
        return 0, tc0 - 1 - max(cols)
    if direction == "left":
        return 0, tc1 + 1 - min(cols)
    if direction == "bottom":
        return tr0 - 1 - max(rows), 0
    assert direction == "top"
    return tr1 + 1 - min(rows), 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pour each fenced interior object out through the fence gap and pin it to the first fence in that direction.

    Input structure:
    - The background is black.
    - Several monochrome rectangular fences appear as one-pixel-thick borders with one contiguous gap of length 2-3 on exactly one side.
    - Each fence contains one small connected object of another color somewhere inside the border.
    - Fence groups are spaced apart, but every gap direction points toward another fence.

    Rule:
    - Detect every fence and identify which side contains its gap: top, bottom, left, or right.
    - Remove the interior object from its source fence.
    - Translate that object straight along the gap direction until its front edge becomes adjacent to the first other fence encountered in that direction.
    - Recolor the translated object with the source fence color; keep every fence, including its gap, unchanged.

    Key insight:
    - The gap side determines only the translation direction, while the moved object's final color comes from its source fence rather than from the destination fence.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    fences = _find_fences(g, bg)
    assert fences, "expected at least one fence"

    moves = []
    for fence in fences:
        obj_pixels = _extract_object_pixels(g, fence, bg)
        target = _find_target_fence(fence, fences)
        dr, dc = _translation_delta(fence["direction"], target["bbox"], obj_pixels)
        moves.append((fence, obj_pixels, dr, dc))

    out = g.copy()
    for _fence, obj_pixels, _dr, _dc in moves:
        for r, c in obj_pixels:
            out[r, c] = bg

    for fence, obj_pixels, dr, dc in moves:
        for r, c in obj_pixels:
            out[r + dr, c + dc] = int(fence["color"])
    return out


# --------------- generator (codex) ---------------

BG = 0
COLOR_POOL = tuple(range(1, 10))
GEN_ATTEMPTS = 400
TEMPLATE_WEIGHTS = (3, 3, 4)  # pair, triple, square
SIDE_CHOICES = (5, 6)
GAP_CHOICES = (2, 3)


def _grow_connected_shape(max_h: int, max_w: int) -> tuple[list[tuple[int, int]], int, int] | None:
    for _ in range(80):
        box_h = random.randint(1, max_h)
        box_w = random.randint(1, max_w)
        if box_h * box_w < 2:
            continue

        area_max = min(box_h * box_w, 5)
        area_min = 2 if area_max == 2 else 3
        area = random.randint(area_min, area_max)
        start = (random.randrange(box_h), random.randrange(box_w))
        cells = {start}

        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != area:
            continue

        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        coords = sorted((r - min_r, c - min_c) for r, c in cells)
        shape_h = max_r - min_r + 1
        shape_w = max_c - min_c + 1
        return coords, shape_h, shape_w
    return None


def _covering_segment(occupied: list[int], gap_len: int, side_len: int) -> int:
    low = max(1, max(occupied) - gap_len + 1)
    high = min(min(occupied), side_len - gap_len - 1)
    assert low <= high
    return random.randint(low, high)


def _build_fence(direction: str, border_color: int, obj_color: int, h: int, w: int) -> tuple[np.ndarray, dict] | None:
    gap_len = random.choice(GAP_CHOICES)
    inner_h = h - 2
    inner_w = w - 2

    if direction in ("left", "right"):
        max_shape_h = min(gap_len, inner_h)
        max_shape_w = min(3, inner_w)
    else:
        max_shape_h = min(3, inner_h)
        max_shape_w = min(gap_len, inner_w)

    shape = _grow_connected_shape(max_shape_h, max_shape_w)
    if shape is None:
        return None
    coords, shape_h, shape_w = shape

    if direction == "right":
        top = random.randint(0, inner_h - shape_h)
        left = random.randint(max(0, inner_w - shape_w - 1), inner_w - shape_w)
        occupied = [1 + top + r for r, _c in coords]
        gap_start = _covering_segment(occupied, gap_len, h)
        move_span = shape_w
    elif direction == "left":
        top = random.randint(0, inner_h - shape_h)
        left = random.randint(0, min(1, inner_w - shape_w))
        occupied = [1 + top + r for r, _c in coords]
        gap_start = _covering_segment(occupied, gap_len, h)
        move_span = shape_w
    elif direction == "bottom":
        top = random.randint(max(0, inner_h - shape_h - 1), inner_h - shape_h)
        left = random.randint(0, inner_w - shape_w)
        occupied = [1 + left + c for _r, c in coords]
        gap_start = _covering_segment(occupied, gap_len, w)
        move_span = shape_h
    else:
        assert direction == "top"
        top = random.randint(0, min(1, inner_h - shape_h))
        left = random.randint(0, inner_w - shape_w)
        occupied = [1 + left + c for _r, c in coords]
        gap_start = _covering_segment(occupied, gap_len, w)
        move_span = shape_h

    patch = np.full((h, w), BG, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color

    if direction == "right":
        patch[gap_start : gap_start + gap_len, -1] = BG
    elif direction == "left":
        patch[gap_start : gap_start + gap_len, 0] = BG
    elif direction == "bottom":
        patch[-1, gap_start : gap_start + gap_len] = BG
    else:
        patch[0, gap_start : gap_start + gap_len] = BG

    for r, c in coords:
        patch[1 + top + r, 1 + left + c] = obj_color

    return patch, {"span": move_span}


def _build_row_scene(directions: list[str]) -> np.ndarray | None:
    n = len(directions)
    height = random.choice(SIDE_CHOICES)
    widths = [random.choice(SIDE_CHOICES) for _ in directions]
    colors = random.sample(COLOR_POOL, 2 * n)
    border_colors = colors[:n]
    obj_colors = colors[n:]

    patches = []
    infos = []
    for direction, width, border_color, obj_color in zip(directions, widths, border_colors, obj_colors):
        built = _build_fence(direction, border_color, obj_color, height, width)
        if built is None:
            return None
        patch, info = built
        patches.append(patch)
        infos.append(info)

    gaps = []
    for i in range(n - 1):
        movers = []
        if directions[i] == "right":
            movers.append(infos[i]["span"])
        if directions[i + 1] == "left":
            movers.append(infos[i + 1]["span"])
        assert movers
        gaps.append(sum(movers) + 1)

    total_w = sum(widths) + sum(gaps)
    if total_w > 30:
        return None

    grid = np.full((height, total_w), BG, dtype=int)
    col = 0
    for i, patch in enumerate(patches):
        h, w = patch.shape
        grid[:h, col : col + w] = patch
        col += w
        if i < len(gaps):
            col += gaps[i]
    return grid


def _build_square_scene() -> np.ndarray | None:
    h_top = random.choice(SIDE_CHOICES)
    h_bottom = random.choice(SIDE_CHOICES)
    w_left = random.choice(SIDE_CHOICES)
    w_right = random.choice(SIDE_CHOICES)
    directions = ["right", "bottom", "left", "top"]
    sizes = [(h_top, w_left), (h_top, w_right), (h_bottom, w_right), (h_bottom, w_left)]
    colors = random.sample(COLOR_POOL, 8)
    border_colors = colors[:4]
    obj_colors = colors[4:]

    patches = []
    infos = []
    for direction, (h, w), border_color, obj_color in zip(directions, sizes, border_colors, obj_colors):
        built = _build_fence(direction, border_color, obj_color, h, w)
        if built is None:
            return None
        patch, info = built
        patches.append(patch)
        infos.append(info)

    gap_x = max(infos[0]["span"], infos[2]["span"]) + 1
    gap_y = max(infos[1]["span"], infos[3]["span"]) + 1
    total_h = h_top + gap_y + h_bottom
    total_w = w_left + gap_x + w_right
    if max(total_h, total_w) > 30:
        return None

    grid = np.full((total_h, total_w), BG, dtype=int)
    grid[0:h_top, 0:w_left] = patches[0]
    grid[0:h_top, w_left + gap_x : total_w] = patches[1]
    grid[h_top + gap_y : total_h, w_left + gap_x : total_w] = patches[2]
    grid[h_top + gap_y : total_h, 0:w_left] = patches[3]
    return grid


def _build_local_scene() -> tuple[np.ndarray, int] | None:
    template = random.choices(["pair", "triple", "square"], weights=TEMPLATE_WEIGHTS, k=1)[0]
    if template == "pair":
        local = _build_row_scene(["right", "left"])
        count = 2
    elif template == "triple":
        local = _build_row_scene(random.choice([["right", "left", "left"], ["right", "right", "left"]]))
        count = 3
    else:
        local = _build_square_scene()
        count = 4
    if local is None:
        return None
    if random.random() < 0.5:
        local = np.fliplr(local)
    local = np.rot90(local, random.randrange(4))
    return np.array(local, dtype=int), count


def _validate_generated(grid: np.ndarray, expected_fences: int) -> bool:
    try:
        bg = _background_color(grid)
        fences = _find_fences(grid, bg)
        if len(fences) != expected_fences:
            return False

        h, w = grid.shape
        moved: set[tuple[int, int]] = set()
        for fence in fences:
            obj_pixels = _extract_object_pixels(grid, fence, bg)
            target = _find_target_fence(fence, fences)
            dr, dc = _translation_delta(fence["direction"], target["bbox"], obj_pixels)
            for r, c in obj_pixels:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    return False
                if int(grid[nr, nc]) != bg:
                    return False
                if (nr, nc) in moved:
                    return False
                moved.add((nr, nc))

        out = main(grid)
        return out.shape == grid.shape and max(out.shape) <= 30
    except AssertionError:
        return False
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    built = _build_local_scene()
    if built is None:
        return None
    local, expected_fences = built
    lh, lw = local.shape
    extra_h = random.randint(0, min(6, 30 - lh))
    extra_w = random.randint(0, min(6, 30 - lw))
    height = lh + extra_h
    width = lw + extra_w
    top = random.randint(0, extra_h)
    left = random.randint(0, extra_w)

    grid = np.full((height, width), BG, dtype=int)
    grid[top : top + lh, left : left + lw] = local
    if not _validate_generated(grid, expected_fences):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
