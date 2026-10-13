from __future__ import annotations

import random

import numpy as np


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _frame_border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _extract_frames(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    bg = _bg_color(grid)
    frames: list[dict[str, object]] = []

    for color_value in np.unique(grid):
        color = int(color_value)
        if color == bg:
            continue

        rows, cols = np.where(grid == color)
        assert rows.size > 0, "each non-background color must be present"

        r0 = int(rows.min())
        r1 = int(rows.max())
        c0 = int(cols.min())
        c1 = int(cols.max())
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= 3 and width >= 3, "frames must have a non-empty interior"

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = _frame_border_mask(height, width)
        assert np.all(patch[border] == color), "frame border must be complete"

        interior = patch[1:-1, 1:-1]
        assert interior.size > 0
        assert np.all((interior == bg) | (interior == color)), "interior may only contain bg or frame color"

        pattern = interior == color
        assert np.any(pattern), "each frame must contain an interior pattern"

        frames.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "pattern": pattern.copy(),
                "center2": (r0 + r1, c0 + c1),
            }
        )

    frames.sort(key=lambda item: tuple(item["bbox"]))
    assert len(frames) >= 2, "need at least two frames"

    sizes = {
        (
            int(frame["bbox"][2]) - int(frame["bbox"][0]) + 1,
            int(frame["bbox"][3]) - int(frame["bbox"][1]) + 1,
        )
        for frame in frames
    }
    assert len(sizes) == 1, "all frames must have the same size"

    for i, frame_a in enumerate(frames):
        ar0, ac0, ar1, ac1 = frame_a["bbox"]
        for frame_b in frames[i + 1 :]:
            br0, bc0, br1, bc1 = frame_b["bbox"]
            separated = ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0
            assert separated, "frame boxes must not overlap"

    return bg, frames


def _nearest_indices_with_tiebreak(frames: list[dict[str, object]]) -> list[int]:
    nearest: list[int] = []
    for i, frame in enumerate(frames):
        center_r2, center_c2 = frame["center2"]
        choices: list[tuple[int, int, int, int]] = []
        for j, other in enumerate(frames):
            if i == j:
                continue
            other_r2, other_c2 = other["center2"]
            dist = abs(center_r2 - other_r2) + abs(center_c2 - other_c2)
            r0, c0, _, _ = other["bbox"]
            choices.append((dist, r0, c0, j))
        nearest.append(min(choices)[-1])
    return nearest


def _apply_rule(
    grid: np.ndarray,
    bg: int,
    frames: list[dict[str, object]],
    source_indices: list[int],
) -> np.ndarray:
    out = grid.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = bg

    for frame, source_idx in zip(frames, source_indices):
        r0, c0, r1, c1 = frame["bbox"]
        patch = out[r0 + 1 : r1, c0 + 1 : c1]
        patch[frames[source_idx]["pattern"]] = int(frame["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each frame's interior pattern with the nearest other frame's pattern mask,
    redrawn in the current frame's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, frames = _extract_frames(g)
    nearest = _nearest_indices_with_tiebreak(frames)
    return _apply_rule(g, bg, frames, nearest)


BG = 8
COLOR_POOL = tuple(color for color in range(10) if color != BG)

GRID_MIN = 18
GRID_MAX = 30
FRAME_COUNT_MIN = 3
FRAME_COUNT_MAX = 5
FRAME_H_MIN = 6
FRAME_H_MAX = 8
FRAME_W_MIN = 6
FRAME_W_MAX = 8
FRAME_PAD = 2
PLACE_ATTEMPTS = 150
PATTERN_ATTEMPTS = 120
MAX_GENERATE_ATTEMPTS = 400


def _boxes_separated(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
    pad: int,
) -> bool:
    ar0, ac0, ar1, ac1 = bbox_a
    br0, bc0, br1, bc1 = bbox_b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _candidate_boxes(
    height: int,
    width: int,
    frame_h: int,
    frame_w: int,
    placed: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, int, int]] = []
    for r0 in range(height - frame_h + 1):
        for c0 in range(width - frame_w + 1):
            bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
            if all(_boxes_separated(bbox, other, FRAME_PAD) for other in placed):
                candidates.append(bbox)
    return candidates


def _unique_nearest_indices(bboxes: list[tuple[int, int, int, int]]) -> list[int] | None:
    centers2 = [(r0 + r1, c0 + c1) for r0, c0, r1, c1 in bboxes]
    nearest: list[int] = []

    for i, (center_r2, center_c2) in enumerate(centers2):
        distances = sorted(
            (
                abs(center_r2 - other_r2) + abs(center_c2 - other_c2),
                j,
            )
            for j, (other_r2, other_c2) in enumerate(centers2)
            if j != i
        )
        if len(distances) >= 2 and distances[0][0] == distances[1][0]:
            return None
        nearest.append(distances[0][1])
    return nearest


def _random_polyomino(height: int, width: int, rng: random.Random) -> np.ndarray:
    total = height * width
    target = rng.randint(1, max(1, min(total - 1, (total + 1) // 2)))
    cells = {(rng.randrange(height), rng.randrange(width))}
    while len(cells) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells and (nr, nc) not in frontier:
                    frontier.append((nr, nc))
        cells.add(rng.choice(frontier))
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_block(height: int, width: int, rng: random.Random) -> np.ndarray | None:
    choices = [
        (block_h, block_w)
        for block_h in range(1, height + 1)
        for block_w in range(1, width + 1)
        if block_h * block_w < height * width
    ]
    if not choices:
        return None
    block_h, block_w = rng.choice(choices)
    r0 = rng.randrange(height - block_h + 1)
    c0 = rng.randrange(width - block_w + 1)
    mask = np.zeros((height, width), dtype=bool)
    mask[r0 : r0 + block_h, c0 : c0 + block_w] = True
    return mask


def _random_line(height: int, width: int, rng: random.Random) -> np.ndarray | None:
    options: list[str] = []
    if width >= 2:
        options.append("h")
    if height >= 2:
        options.append("v")
    if not options:
        return None
    orient = rng.choice(options)
    mask = np.zeros((height, width), dtype=bool)
    if orient == "h":
        row = rng.randrange(height)
        length = rng.randint(2, width)
        c0 = rng.randrange(width - length + 1)
        mask[row, c0 : c0 + length] = True
    else:
        col = rng.randrange(width)
        length = rng.randint(2, height)
        r0 = rng.randrange(height - length + 1)
        mask[r0 : r0 + length, col] = True
    return mask


def _random_elbow(height: int, width: int, rng: random.Random) -> np.ndarray | None:
    if height < 2 or width < 2:
        return None
    leg_h = rng.randint(2, height)
    leg_w = rng.randint(2, width)
    r0 = rng.randrange(height - leg_h + 1)
    c0 = rng.randrange(width - leg_w + 1)
    corner = rng.choice(("tl", "tr", "bl", "br"))
    mask = np.zeros((height, width), dtype=bool)
    if corner == "tl":
        mask[r0 : r0 + leg_h, c0] = True
        mask[r0, c0 : c0 + leg_w] = True
    elif corner == "tr":
        mask[r0 : r0 + leg_h, c0 + leg_w - 1] = True
        mask[r0, c0 : c0 + leg_w] = True
    elif corner == "bl":
        mask[r0 : r0 + leg_h, c0] = True
        mask[r0 + leg_h - 1, c0 : c0 + leg_w] = True
    else:
        mask[r0 : r0 + leg_h, c0 + leg_w - 1] = True
        mask[r0 + leg_h - 1, c0 : c0 + leg_w] = True
    return mask


def _random_cross(height: int, width: int, rng: random.Random) -> np.ndarray | None:
    if height < 3 or width < 3:
        return None
    center_r = rng.randint(1, height - 2)
    center_c = rng.randint(1, width - 2)
    up = rng.randint(1, center_r)
    down = rng.randint(1, height - 1 - center_r)
    left = rng.randint(1, center_c)
    right = rng.randint(1, width - 1 - center_c)
    mask = np.zeros((height, width), dtype=bool)
    mask[center_r - up : center_r + down + 1, center_c] = True
    mask[center_r, center_c - left : center_c + right + 1] = True
    return mask


def _random_stair(height: int, width: int, rng: random.Random) -> np.ndarray | None:
    if height < 2 or width < 2:
        return None
    steps = rng.randint(2, min(height, width))
    base_cells = [(0, 0)]
    cur_r, cur_c = 0, 0
    for _ in range(steps - 1):
        cur_c += 1
        base_cells.append((cur_r, cur_c))
        cur_r += 1
        base_cells.append((cur_r, cur_c))

    flip_r = rng.choice((False, True))
    flip_c = rng.choice((False, True))
    r0 = rng.randrange(height - steps + 1)
    c0 = rng.randrange(width - steps + 1)
    mask = np.zeros((height, width), dtype=bool)
    for r, c in base_cells:
        rr = steps - 1 - r if flip_r else r
        cc = steps - 1 - c if flip_c else c
        mask[r0 + rr, c0 + cc] = True
    return mask


def _valid_pattern(mask: np.ndarray | None) -> bool:
    if mask is None:
        return False
    count = int(np.count_nonzero(mask))
    return 0 < count <= (3 * mask.size) // 4 and count < mask.size


def _sample_pattern(height: int, width: int, rng: random.Random) -> np.ndarray | None:
    builders = [
        _random_polyomino,
        _random_block,
        _random_line,
        _random_elbow,
        _random_cross,
        _random_stair,
    ]
    rng.shuffle(builders)
    for builder in builders:
        mask = builder(height, width, rng)
        if _valid_pattern(mask):
            return mask
    return None


def _sample_distinct_patterns(
    count: int,
    interior_h: int,
    interior_w: int,
    rng: random.Random,
) -> list[np.ndarray] | None:
    patterns: list[np.ndarray] = []
    seen: set[bytes] = set()
    for _ in range(count):
        for _ in range(PATTERN_ATTEMPTS):
            mask = _sample_pattern(interior_h, interior_w, rng)
            if mask is None:
                continue
            sig = mask.tobytes()
            if sig not in seen:
                seen.add(sig)
                patterns.append(mask)
                break
        else:
            return None
    return patterns


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _render_scene(
    height: int,
    width: int,
    bboxes: list[tuple[int, int, int, int]],
    colors: list[int],
    patterns: list[np.ndarray],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for bbox, color, pattern in zip(bboxes, colors, patterns):
        _draw_frame(grid, bbox, color)
        r0, c0, r1, c1 = bbox
        patch = grid[r0 + 1 : r1, c0 + 1 : c1]
        patch[pattern] = color
    return grid


def _expected_output(
    inp: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    colors: list[int],
    patterns: list[np.ndarray],
    nearest: list[int],
) -> np.ndarray:
    frame_records = []
    for bbox, color, pattern in zip(bboxes, colors, patterns):
        r0, c0, r1, c1 = bbox
        frame_records.append(
            {
                "color": color,
                "bbox": bbox,
                "pattern": pattern.copy(),
                "center2": (r0 + r1, c0 + c1),
            }
        )
    return _apply_rule(inp, BG, frame_records, nearest)


def _sample_canvas(frame_count: int, frame_h: int, frame_w: int, rng: random.Random) -> tuple[int, int]:
    packed_side = int(np.ceil(np.sqrt(frame_count * (frame_h + FRAME_PAD) * (frame_w + FRAME_PAD)))) + 2
    height_lo = max(GRID_MIN, packed_side - 1)
    width_lo = max(GRID_MIN, packed_side - 1)
    height_hi = min(GRID_MAX, packed_side + 4)
    width_hi = min(GRID_MAX, packed_side + 4)
    height = rng.randint(height_lo, height_hi)
    width = rng.randint(width_lo, width_hi)
    if rng.random() < 0.5:
        height = min(GRID_MAX, height + rng.randint(0, 3))
    else:
        width = min(GRID_MAX, width + rng.randint(0, 3))
    return height, width


def _place_frames(
    height: int,
    width: int,
    frame_h: int,
    frame_w: int,
    frame_count: int,
    rng: random.Random,
) -> tuple[list[tuple[int, int, int, int]], list[int]] | None:
    for _ in range(PLACE_ATTEMPTS):
        bboxes: list[tuple[int, int, int, int]] = []
        for _ in range(frame_count):
            candidates = _candidate_boxes(height, width, frame_h, frame_w, bboxes)
            if not candidates:
                break
            bboxes.append(rng.choice(candidates))
        if len(bboxes) != frame_count:
            continue
        nearest = _unique_nearest_indices(bboxes)
        if nearest is not None:
            return bboxes, nearest
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    frame_count = rng.randint(FRAME_COUNT_MIN, FRAME_COUNT_MAX)
    frame_h = rng.randint(FRAME_H_MIN, FRAME_H_MAX)
    frame_w = rng.randint(FRAME_W_MIN, FRAME_W_MAX)
    height, width = _sample_canvas(frame_count, frame_h, frame_w, rng)

    placed = _place_frames(height, width, frame_h, frame_w, frame_count, rng)
    if placed is None:
        return None
    bboxes, nearest = placed

    patterns = _sample_distinct_patterns(frame_count, frame_h - 2, frame_w - 2, rng)
    if patterns is None:
        return None

    colors = rng.sample(COLOR_POOL, frame_count)
    inp = _render_scene(height, width, bboxes, colors, patterns)
    out = _expected_output(inp, bboxes, colors, patterns, nearest)

    if np.array_equal(inp, out):
        return None
    if _bg_color(inp) != BG:
        return None

    try:
        if np.array_equal(main(inp), out):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input")
