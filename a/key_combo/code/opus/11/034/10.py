import random

import numpy as np


BG = 0
FRAME = 5
PATTERN_COLORS = (1, 2, 3, 4, 6)
MAX_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _expanded_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _is_connected_mask(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _is_frame_component(grid: np.ndarray, comp: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(comp)
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
        return False

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros_like(sub, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if not np.all(sub[border] == FRAME):
        return False
    if np.any(sub[1:-1, 1:-1] == FRAME):
        return False
    return len(comp) == int(border.sum())


def _extract_frames(grid: np.ndarray) -> list[dict]:
    comps = _components_of_color(grid, FRAME)
    frames: list[dict] = []
    for comp in comps:
        assert _is_frame_component(grid, comp), "every grey component must be a one-pixel rectangular frame"
        r0, c0, r1, c1 = _bbox(comp)
        inner = grid[r0 + 1 : r1, c0 + 1 : c1].copy()
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner": inner,
                "count": int(np.count_nonzero(inner != BG)),
            }
        )
    assert 3 <= len(frames) <= 5, "expected 3 to 5 frames"
    counts = [frame["count"] for frame in frames]
    assert len(set(counts)) == len(counts), "frame pattern counts must be unique"
    return frames


def _compose_output(crops: list[np.ndarray]) -> np.ndarray:
    max_h = max(int(crop.shape[0]) for crop in crops)
    total_w = sum(int(crop.shape[1]) for crop in crops) + len(crops) - 1
    out = np.full((max_h, total_w), BG, dtype=int)

    cursor = 0
    for i, crop in enumerate(crops):
        h, w = crop.shape
        top = (max_h - h) // 2
        out[top : top + h, cursor : cursor + w] = crop
        cursor += w
        if i < len(crops) - 1:
            out[:, cursor] = FRAME
            cursor += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop each framed pattern, sort the crops by non-background pixel count, and concatenate them with grey separators.

    Input structure:
    - The grid has a black background and 3 to 5 disjoint grey rectangular frames.
    - Each frame is a one-pixel-thick rectangle whose interior is 3x3 to 5x5.
    - Inside each frame is one single-color 4-connected pattern with 2 to 8 non-black pixels, and different frames have different counts.

    Rule:
    - Detect every grey frame and crop only its interior, excluding the grey border.
    - Count the non-black pixels in each cropped interior.
    - Sort the cropped interiors from the smallest count to the largest count.
    - Build the output by placing the sorted crops left-to-right with one grey separator column between adjacent crops; the output height is the maximum crop height, and shorter crops are vertically centered with black padding, biased upward when exact centering is impossible.

    Key insight:
    - The frame only marks where to crop; the decisive ordering signal is the number of non-black pixels inside each cropped frame, not the frame position, size, or pattern color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _extract_frames(g)
    ordered = sorted(frames, key=lambda frame: frame["count"])
    return _compose_output([frame["inner"] for frame in ordered])


def _random_connected_shape(height: int, width: int, count: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    cells = {start}

    while len(cells) < count:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        nr, nc = random.choice(list(frontier))
        cells.add((nr, nc))
        mask[nr, nc] = True
    return mask


def _sample_sizes(n_frames: int) -> list[tuple[int, int]]:
    shared = random.choice([(4, 4), (4, 5), (5, 4), (5, 5)])
    sizes = [shared, shared]

    while len(sizes) < n_frames:
        sizes.append((random.randint(3, 5), random.randint(3, 5)))

    extra_ids = list(range(2, n_frames))
    mutable_ids = extra_ids[:]
    random.shuffle(mutable_ids)
    if not any(h == 5 for h, _ in sizes):
        i = mutable_ids.pop()
        sizes[i] = (5, random.randint(3, 5))
    if not any(h == 4 for h, _ in sizes):
        i = mutable_ids.pop() if mutable_ids else extra_ids[0]
        sizes[i] = (4, random.randint(3, 5))
    if len({w for _, w in sizes}) == 1:
        i = random.choice(extra_ids)
        sizes[i] = (sizes[i][0], 3 if sizes[i][1] != 3 else 5)

    random.shuffle(sizes)
    assert any(h == 5 for h, _ in sizes)
    assert any(h == 4 for h, _ in sizes)
    assert len({w for _, w in sizes}) >= 2
    assert len(set(sizes)) < len(sizes), "need at least one repeated size to defeat size-only shortcuts"
    return sizes


def _sample_counts(sizes: list[tuple[int, int]]) -> list[int]:
    ids = list(range(len(sizes)))
    areas = [h * w for h, w in sizes]
    for _ in range(200):
        counts = random.sample(range(2, 9), len(sizes))
        count_order = sorted(ids, key=lambda i: counts[i])
        area_order = sorted(ids, key=lambda i: (areas[i], sizes[i][0], sizes[i][1], i))
        if count_order != area_order:
            return counts
    raise RuntimeError("failed to decorrelate counts from frame area")


def _sample_colors(n_frames: int, counts: list[int]) -> list[int]:
    ids = list(range(n_frames))
    count_order = sorted(ids, key=lambda i: counts[i])
    for _ in range(200):
        colors = [random.choice(PATTERN_COLORS) for _ in range(n_frames)]
        color_order = sorted(ids, key=lambda i: (colors[i], i))
        if len(set(colors)) < n_frames or color_order != count_order:
            return colors
    raise RuntimeError("failed to decorrelate colors from count order")


def _layout_frames(specs: list[dict]) -> list[tuple[int, int]] | None:
    n_frames = len(specs)
    ids = list(range(n_frames))
    count_order = sorted(ids, key=lambda i: specs[i]["count"])

    split_options = [n_frames // 2]
    if n_frames % 2 == 1:
        split_options.append(n_frames // 2 + 1)

    for _ in range(300):
        visible_order = random.sample(ids, n_frames)
        if visible_order == count_order:
            continue

        top_count = random.choice(split_options)
        top_ids = visible_order[:top_count]
        bot_ids = visible_order[top_count:]

        top_gap = [random.randint(2, 4) for _ in range(max(0, len(top_ids) - 1))]
        bot_gap = [random.randint(2, 4) for _ in range(max(0, len(bot_ids) - 1))]
        row_gap = random.randint(2, 4)
        top_margin = random.randint(1, 2)
        bottom_margin = random.randint(1, 2)
        left_top = random.randint(1, 3)
        left_bot = random.randint(1, 5)
        right_margin = random.randint(1, 3)

        top_row_h = max(specs[i]["outer_h"] for i in top_ids)
        bot_row_h = max(specs[i]["outer_h"] for i in bot_ids)

        top_row_w = sum(specs[i]["outer_w"] for i in top_ids) + sum(top_gap)
        bot_row_w = sum(specs[i]["outer_w"] for i in bot_ids) + sum(bot_gap)

        height = top_margin + top_row_h + row_gap + bot_row_h + bottom_margin
        width = max(left_top + top_row_w + right_margin, left_bot + bot_row_w + right_margin)
        if width <= height:
            width = height + random.randint(1, 4)
        if height > 30 or width > 30:
            continue

        positions: list[tuple[int, int] | None] = [None] * n_frames
        cursor = left_top
        row_r = top_margin
        for j, idx in enumerate(top_ids):
            positions[idx] = (row_r, cursor)
            cursor += specs[idx]["outer_w"]
            if j < len(top_gap):
                cursor += top_gap[j]

        cursor = left_bot
        row_r = top_margin + top_row_h + row_gap
        for j, idx in enumerate(bot_ids):
            positions[idx] = (row_r, cursor)
            cursor += specs[idx]["outer_w"]
            if j < len(bot_gap):
                cursor += bot_gap[j]

        assert all(pos is not None for pos in positions)
        bboxes = []
        for idx, (r, c) in enumerate(positions):
            bbox = (r, c, r + specs[idx]["outer_h"] - 1, c + specs[idx]["outer_w"] - 1)
            bboxes.append(bbox)

        if any(_expanded_overlap(bboxes[i], bboxes[j], margin=1) for i in range(n_frames) for j in range(i + 1, n_frames)):
            continue

        row_major = sorted(ids, key=lambda i: (bboxes[i][0], bboxes[i][1]))
        col_major = sorted(ids, key=lambda i: (bboxes[i][1], bboxes[i][0]))
        if row_major == count_order or col_major == count_order:
            continue

        return [(int(r), int(c)) for r, c in positions]
    return None


def _render_grid(specs: list[dict], positions: list[tuple[int, int]]) -> np.ndarray:
    height = max(r + spec["outer_h"] for spec, (r, _c) in zip(specs, positions))
    width = max(c + spec["outer_w"] for spec, (_r, c) in zip(specs, positions))
    grid = np.full((height, width), BG, dtype=int)

    for spec, (r0, c0) in zip(specs, positions):
        r1 = r0 + spec["outer_h"] - 1
        c1 = c0 + spec["outer_w"] - 1
        grid[r0, c0 : c1 + 1] = FRAME
        grid[r1, c0 : c1 + 1] = FRAME
        grid[r0 : r1 + 1, c0] = FRAME
        grid[r0 : r1 + 1, c1] = FRAME

        inner = np.full((spec["inner_h"], spec["inner_w"]), BG, dtype=int)
        inner[spec["shape"]] = spec["color"]
        grid[r0 + 1 : r1, c0 + 1 : c1] = inner
    return grid


def _check_generated_structure(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if grid.shape[1] <= grid.shape[0]:
        return False

    frames = _extract_frames(grid)
    ids = list(range(len(frames)))
    heights = [frame["inner"].shape[0] for frame in frames]
    widths = [frame["inner"].shape[1] for frame in frames]
    counts = [frame["count"] for frame in frames]
    areas = [h * w for h, w in zip(heights, widths)]
    bboxes = [frame["bbox"] for frame in frames]

    if not any((max(heights) - h) % 2 == 1 for h in heights if h != max(heights)):
        return False
    if len(set(zip(heights, widths))) == len(frames):
        return False
    if sorted(ids, key=lambda i: counts[i]) == sorted(ids, key=lambda i: (areas[i], heights[i], widths[i], i)):
        return False
    if sorted(ids, key=lambda i: counts[i]) == sorted(ids, key=lambda i: (bboxes[i][0], bboxes[i][1])):
        return False
    if sorted(ids, key=lambda i: counts[i]) == sorted(ids, key=lambda i: (bboxes[i][1], bboxes[i][0])):
        return False

    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            if _expanded_overlap(bboxes[i], bboxes[j], margin=1):
                return False

    for frame in frames:
        inner = frame["inner"]
        h, w = inner.shape
        if not (3 <= h <= 5 and 3 <= w <= 5):
            return False
        mask = inner != BG
        if not (2 <= int(mask.sum()) <= 8):
            return False
        colors = set(map(int, inner[mask]))
        if len(colors) != 1 or next(iter(colors)) not in PATTERN_COLORS:
            return False
        if not _is_connected_mask(mask):
            return False

    out = main(grid)
    expected = _compose_output([frame["inner"] for frame in sorted(frames, key=lambda frame: frame["count"])])
    if not np.array_equal(out, expected):
        return False
    if out.ndim != 2 or max(out.shape) > 30:
        return False
    sep_cols = [i for i in range(out.shape[1]) if np.all(out[:, i] == FRAME)]
    if len(sep_cols) != len(frames) - 1:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        n_frames = random.randint(3, 5)
        sizes = _sample_sizes(n_frames)
        counts = _sample_counts(sizes)
        colors = _sample_colors(n_frames, counts)

        specs = []
        for (inner_h, inner_w), count, color in zip(sizes, counts, colors):
            shape = _random_connected_shape(inner_h, inner_w, count)
            specs.append(
                {
                    "inner_h": inner_h,
                    "inner_w": inner_w,
                    "outer_h": inner_h + 2,
                    "outer_w": inner_w + 2,
                    "count": count,
                    "color": color,
                    "shape": shape,
                }
            )

        positions = _layout_frames(specs)
        if positions is None:
            continue

        grid = _render_grid(specs, positions)
        if _check_generated_structure(grid):
            return grid
    raise RuntimeError("failed to generate a valid input after many attempts")
