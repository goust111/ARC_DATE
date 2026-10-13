from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIRS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(grid[r0, c0]) == bg:
                continue

            color = int(grid[r0, c0])
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []

            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for rr, cc in _neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and int(grid[rr, cc]) == color:
                        seen[rr, cc] = True
                        stack.append((rr, cc))

            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )

    return comps


def _is_rect_frame(grid: np.ndarray, comp: dict) -> bool:
    color = int(comp["color"])
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if int(comp["area"]) != 2 * h + 2 * w - 4:
        return False

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return bool(np.all(sub[mask] == color) and np.all(sub[~mask] != color))


def _decode_arrow(coords: list[tuple[int, int]]) -> dict | None:
    pts = {(int(r), int(c)) for r, c in coords}
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    matches: list[dict] = []

    right_tips = [(r, c) for r, c in pts if c == c1]
    if len(right_tips) == 1:
        tip_r, _ = right_tips[0]
        row_cols = sorted(c for r, c in pts if r == tip_r)
        if len(row_cols) >= 2 and row_cols == list(range(row_cols[0], c1 + 1)):
            base_c = c1 - 1
            expected = {(tip_r, c) for c in range(row_cols[0], c1 + 1)}
            expected |= {(tip_r - 1, base_c), (tip_r + 1, base_c)}
            if pts == expected and (tip_r - 1, base_c) in pts and (tip_r + 1, base_c) in pts:
                matches.append({"dir": "right", "steps": len(row_cols) - 1, "line_index": tip_r})

    left_tips = [(r, c) for r, c in pts if c == c0]
    if len(left_tips) == 1:
        tip_r, _ = left_tips[0]
        row_cols = sorted(c for r, c in pts if r == tip_r)
        if len(row_cols) >= 2 and row_cols == list(range(c0, row_cols[-1] + 1)):
            base_c = c0 + 1
            expected = {(tip_r, c) for c in range(c0, row_cols[-1] + 1)}
            expected |= {(tip_r - 1, base_c), (tip_r + 1, base_c)}
            if pts == expected and (tip_r - 1, base_c) in pts and (tip_r + 1, base_c) in pts:
                matches.append({"dir": "left", "steps": len(row_cols) - 1, "line_index": tip_r})

    up_tips = [(r, c) for r, c in pts if r == r0]
    if len(up_tips) == 1:
        _, tip_c = up_tips[0]
        col_rows = sorted(r for r, c in pts if c == tip_c)
        if len(col_rows) >= 2 and col_rows == list(range(r0, col_rows[-1] + 1)):
            base_r = r0 + 1
            expected = {(r, tip_c) for r in range(r0, col_rows[-1] + 1)}
            expected |= {(base_r, tip_c - 1), (base_r, tip_c + 1)}
            if pts == expected and (base_r, tip_c - 1) in pts and (base_r, tip_c + 1) in pts:
                matches.append({"dir": "up", "steps": len(col_rows) - 1, "line_index": tip_c})

    down_tips = [(r, c) for r, c in pts if r == r1]
    if len(down_tips) == 1:
        _, tip_c = down_tips[0]
        col_rows = sorted(r for r, c in pts if c == tip_c)
        if len(col_rows) >= 2 and col_rows == list(range(col_rows[0], r1 + 1)):
            base_r = r1 - 1
            expected = {(r, tip_c) for r in range(col_rows[0], r1 + 1)}
            expected |= {(base_r, tip_c - 1), (base_r, tip_c + 1)}
            if pts == expected and (base_r, tip_c - 1) in pts and (base_r, tip_c + 1) in pts:
                matches.append({"dir": "down", "steps": len(col_rows) - 1, "line_index": tip_c})

    if len(matches) != 1:
        return None
    return matches[0]


def _inside_box(r: int, c: int, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _arrow_gap_to_frame(
    frame_box: tuple[int, int, int, int],
    arrow_box: tuple[int, int, int, int],
    direction: str,
    line_index: int,
) -> int | None:
    fr0, fc0, fr1, fc1 = frame_box
    ar0, ac0, ar1, ac1 = arrow_box

    if direction == "right":
        if ac0 <= fc1 or not (fr0 <= line_index <= fr1):
            return None
        gap = ac0 - fc1 - 1
    elif direction == "left":
        if ac1 >= fc0 or not (fr0 <= line_index <= fr1):
            return None
        gap = fc0 - ac1 - 1
    elif direction == "down":
        if ar0 <= fr1 or not (fc0 <= line_index <= fc1):
            return None
        gap = ar0 - fr1 - 1
    else:
        if ar1 >= fr0 or not (fc0 <= line_index <= fc1):
            return None
        gap = fr0 - ar1 - 1

    return int(gap) if gap >= 1 else None


def _grow_mask(seed_coords: list[tuple[int, int]], direction: str, steps: int, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    dr, dc = DIRS[direction]
    mask = np.zeros((h, w), dtype=bool)
    for r, c in seed_coords:
        mask[int(r), int(c)] = True

    for _ in range(int(steps)):
        new_cells: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            rr = int(r) + dr
            cc = int(c) + dc
            if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                new_cells.append((rr, cc))
        if not new_cells:
            break
        for rr, cc in new_cells:
            mask[rr, cc] = True

    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove each grey frame and arrow, then grow the framed seed pattern in the arrow direction.

    The arrow length encodes the number of one-sided dilations. Results are overlaid in
    frame order (top-to-bottom, then left-to-right), so earlier grown objects win overlaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _most_common_color(g)

    comps = _find_monochrome_components(g, bg=bg)
    frames = [comp for comp in comps if _is_rect_frame(g, comp)]
    assert frames, "expected at least one rectangular frame"
    frames.sort(key=lambda comp: (int(comp["bbox"][0]), int(comp["bbox"][1])))

    frame_ids = {id(comp) for comp in frames}
    interiors = [
        (int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1))
        for r0, c0, r1, c1 in (comp["bbox"] for comp in frames)
    ]

    arrows: list[dict] = []
    for comp in comps:
        if id(comp) in frame_ids:
            continue
        if any(all(_inside_box(r, c, box) for r, c in comp["coords"]) for box in interiors):
            continue
        decoded = _decode_arrow(comp["coords"])
        if decoded is None:
            continue
        arrows.append(
            {
                "color": int(comp["color"]),
                "coords": comp["coords"],
                "bbox": tuple(map(int, comp["bbox"])),
                "dir": str(decoded["dir"]),
                "steps": int(decoded["steps"]),
                "line_index": int(decoded["line_index"]),
            }
        )

    used_arrow_ids: set[int] = set()
    objects: list[dict] = []
    for frame in frames:
        fr0, fc0, fr1, fc1 = map(int, frame["bbox"])
        frame_color = int(frame["color"])
        inner = g[fr0 + 1 : fr1, fc0 + 1 : fc1]
        seed_colors = [int(v) for v in np.unique(inner) if int(v) not in (bg, frame_color)]
        assert len(seed_colors) == 1, "each frame must contain exactly one seed color"
        color = seed_colors[0]

        seed_local = np.argwhere(inner == color)
        assert seed_local.size > 0, "seed must be nonempty"
        seed_coords = [(fr0 + 1 + int(r), fc0 + 1 + int(c)) for r, c in seed_local]

        candidates: list[tuple[int, int, dict]] = []
        for arrow in arrows:
            if id(arrow) in used_arrow_ids or int(arrow["color"]) != color:
                continue
            gap = _arrow_gap_to_frame(tuple(map(int, frame["bbox"])), tuple(map(int, arrow["bbox"])), arrow["dir"], arrow["line_index"])
            if gap is None:
                continue
            ar0, ac0, _ar1, _ac1 = arrow["bbox"]
            candidates.append((int(gap), int(ar0) * w + int(ac0), arrow))

        assert candidates, "could not match an arrow to a frame"
        candidates.sort(key=lambda item: (item[0], item[1]))
        arrow = candidates[0][2]
        used_arrow_ids.add(id(arrow))

        objects.append(
            {
                "frame_bbox": tuple(map(int, frame["bbox"])),
                "color": color,
                "seed_coords": seed_coords,
                "dir": str(arrow["dir"]),
                "steps": int(arrow["steps"]),
            }
        )

    out = np.full((h, w), int(bg), dtype=int)
    objects.sort(key=lambda obj: (int(obj["frame_bbox"][0]), int(obj["frame_bbox"][1])))
    for obj in objects:
        grown = _grow_mask(obj["seed_coords"], obj["dir"], obj["steps"], (h, w))
        paint = grown & (out == bg)
        out[paint] = int(obj["color"])
    return out


BG = 0
FRAME_COLOR = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GRID_H_RANGE = (20, 31)
GRID_W_RANGE = (20, 31)
OBJECT_COUNT_RANGE = (2, 5)
STEP_RANGE = (1, 6)
GEN_ATTEMPTS = 300


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return np.asarray(mask[np.ix_(rows, cols)], dtype=bool)


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    out = np.rot90(out, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return _crop_bool(out)


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        cells.add(random.choice(frontier))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return _crop_bool(mask)


def _shape_rect() -> np.ndarray:
    while True:
        h = random.randint(1, 3)
        w = random.randint(1, 3)
        if h * w >= 2:
            return np.ones((h, w), dtype=bool)


def _shape_line() -> np.ndarray:
    if random.random() < 0.5:
        return np.ones((1, random.randint(2, 4)), dtype=bool)
    return np.ones((random.randint(2, 4), 1), dtype=bool)


def _shape_l() -> np.ndarray:
    h = random.randint(2, 4)
    w = random.randint(2, 4)
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = True
    mask[-1, :] = True
    return mask


def _shape_plus() -> np.ndarray:
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, :] = True
    mask[:, 1] = True
    return mask


def _shape_t() -> np.ndarray:
    h = random.randint(2, 4)
    w = random.randint(3, 4)
    mid = w // 2
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[:, mid] = True
    return mask


def _shape_step() -> np.ndarray:
    h = random.randint(2, 3)
    w = random.randint(3, 4)
    mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        start = min(r, w - 2)
        mask[r, start : start + 2] = True
    return _crop_bool(mask)


def _shape_random() -> np.ndarray:
    h = random.randint(2, 4)
    w = random.randint(2, 4)
    area = random.randint(3, min(6, h * w))
    return _random_connected_mask(h, w, area)


def _random_seed_mask() -> np.ndarray:
    factory = random.choice([
        _shape_rect,
        _shape_line,
        _shape_l,
        _shape_plus,
        _shape_t,
        _shape_step,
        _shape_random,
    ])
    return _transform_mask(factory())


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.argwhere(mask).T
    grid[top + rows, left + cols] = int(color)


def _build_object_patch(color: int, direction: str, steps: int) -> dict:
    seed = _random_seed_mask()
    sh, sw = map(int, seed.shape)
    cavity_h = sh + 2 + random.randint(0, 2)
    cavity_w = sw + 2 + random.randint(0, 2)
    frame_h = cavity_h + 2
    frame_w = cavity_w + 2
    seed_off_r = random.randint(1, cavity_h - sh - 1)
    seed_off_c = random.randint(1, cavity_w - sw - 1)
    gap = random.randint(1, 2)

    if direction == "right":
        patch_h, patch_w = frame_h, frame_w + gap + steps + 1
        frame_top, frame_left = 0, 0
        arrow_line = random.randint(1, frame_h - 2)
        shaft_start = frame_w + gap
        shaft_cells = [(arrow_line, c) for c in range(shaft_start, shaft_start + steps)]
        extra = [(arrow_line, shaft_start + steps), (arrow_line - 1, shaft_start + steps - 1), (arrow_line + 1, shaft_start + steps - 1)]
    elif direction == "left":
        patch_h, patch_w = frame_h, steps + gap + 1 + frame_w
        frame_top, frame_left = 0, steps + gap + 1
        arrow_line = random.randint(1, frame_h - 2)
        shaft_end = steps
        shaft_cells = [(arrow_line, c) for c in range(1, shaft_end + 1)]
        extra = [(arrow_line, 0), (arrow_line - 1, 1), (arrow_line + 1, 1)]
    elif direction == "down":
        patch_h, patch_w = frame_h + gap + steps + 1, frame_w
        frame_top, frame_left = 0, 0
        arrow_line = random.randint(1, frame_w - 2)
        shaft_start = frame_h + gap
        shaft_cells = [(r, arrow_line) for r in range(shaft_start, shaft_start + steps)]
        extra = [(shaft_start + steps, arrow_line), (shaft_start + steps - 1, arrow_line - 1), (shaft_start + steps - 1, arrow_line + 1)]
    else:
        patch_h, patch_w = steps + gap + 1 + frame_h, frame_w
        frame_top, frame_left = steps + gap + 1, 0
        arrow_line = random.randint(1, frame_w - 2)
        shaft_end = steps
        shaft_cells = [(r, arrow_line) for r in range(1, shaft_end + 1)]
        extra = [(0, arrow_line), (1, arrow_line - 1), (1, arrow_line + 1)]

    patch = np.full((patch_h, patch_w), BG, dtype=int)

    patch[frame_top, frame_left : frame_left + frame_w] = FRAME_COLOR
    patch[frame_top + frame_h - 1, frame_left : frame_left + frame_w] = FRAME_COLOR
    patch[frame_top : frame_top + frame_h, frame_left] = FRAME_COLOR
    patch[frame_top : frame_top + frame_h, frame_left + frame_w - 1] = FRAME_COLOR

    seed_top = frame_top + 1 + seed_off_r
    seed_left = frame_left + 1 + seed_off_c
    _paint_mask(patch, seed, seed_top, seed_left, color)

    for r, c in shaft_cells + extra:
        patch[int(r), int(c)] = int(color)

    seed_coords = [(seed_top + int(r), seed_left + int(c)) for r, c in np.argwhere(seed)]
    frame_bbox = (frame_top, frame_left, frame_top + frame_h - 1, frame_left + frame_w - 1)

    return {
        "color": int(color),
        "dir": direction,
        "steps": int(steps),
        "patch": patch,
        "frame_bbox": frame_bbox,
        "seed_coords": seed_coords,
    }


def _sample_directions(count: int) -> list[str]:
    dirs = list(DIRS.keys())
    random.shuffle(dirs)
    chosen = dirs[: min(count, len(dirs))]
    while len(chosen) < count:
        chosen.append(random.choice(dirs))
    random.shuffle(chosen)
    return chosen


def _place_objects(objects: list[dict], height: int, width: int) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.full((height, width), BG, dtype=int)
    forbidden = np.zeros((height, width), dtype=bool)
    placed: list[dict | None] = [None] * len(objects)

    order = list(range(len(objects)))
    order.sort(key=lambda idx: objects[idx]["patch"].shape[0] * objects[idx]["patch"].shape[1], reverse=True)

    for idx in order:
        obj = objects[idx]
        patch = obj["patch"]
        ph, pw = map(int, patch.shape)
        if ph + 2 > height or pw + 2 > width:
            return None

        positions = [(r, c) for r in range(1, height - ph) for c in range(1, width - pw)]
        random.shuffle(positions)
        location = None
        for top, left in positions:
            if np.any(forbidden[top : top + ph, left : left + pw]):
                continue
            location = (top, left)
            break
        if location is None:
            return None

        top, left = location
        mask = patch != BG
        sub = grid[top : top + ph, left : left + pw]
        sub[mask] = patch[mask]

        r0 = max(0, top - 1)
        r1 = min(height, top + ph + 1)
        c0 = max(0, left - 1)
        c1 = min(width, left + pw + 1)
        forbidden[r0:r1, c0:c1] = True

        fr0, fc0, fr1, fc1 = obj["frame_bbox"]
        placed[idx] = {
            "color": int(obj["color"]),
            "dir": str(obj["dir"]),
            "steps": int(obj["steps"]),
            "frame_bbox": (top + fr0, left + fc0, top + fr1, left + fc1),
            "seed_coords": [(top + r, left + c) for r, c in obj["seed_coords"]],
        }

    return grid, [obj for obj in placed if obj is not None]


def _render_expected(shape: tuple[int, int], objects: list[dict]) -> tuple[np.ndarray, int]:
    out = np.full(shape, BG, dtype=int)
    masks: list[np.ndarray] = []
    for obj in objects:
        masks.append(_grow_mask(obj["seed_coords"], obj["dir"], obj["steps"], shape))

    overlap_counter = np.zeros(shape, dtype=int)
    for mask in masks:
        overlap_counter += mask.astype(int)
    overlap_pixels = int(np.sum(overlap_counter > 1))

    order = sorted(range(len(objects)), key=lambda i: (objects[i]["frame_bbox"][0], objects[i]["frame_bbox"][1]))
    for idx in order:
        color = int(objects[idx]["color"])
        paint = masks[idx] & (out == BG)
        out[paint] = color
    return out, overlap_pixels


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(SEED_COLORS, count)
    directions = _sample_directions(count)
    steps = [random.randint(*STEP_RANGE) for _ in range(count)]
    objects = [_build_object_patch(color, direction, step) for color, direction, step in zip(colors, directions, steps)]

    max_h = max(int(obj["patch"].shape[0]) for obj in objects)
    max_w = max(int(obj["patch"].shape[1]) for obj in objects)
    height = random.randint(max(GRID_H_RANGE[0], max_h + 2), GRID_H_RANGE[1] - 1)
    width = random.randint(max(GRID_W_RANGE[0], max_w + 2), GRID_W_RANGE[1] - 1)

    placed = _place_objects(objects, height, width)
    if placed is None:
        return None
    grid, placed_objects = placed

    expected, overlap_pixels = _render_expected(grid.shape, placed_objects)
    total_fg = int(np.sum(expected != BG))
    if overlap_pixels > max(8, total_fg // 3):
        return None
    if random.random() < 0.2 and overlap_pixels == 0:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
