import random

import numpy as np


BG_COLOR = 0
GRID_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

STAGING_SIZE = 30
CLUSTER_PADDING = 2
MAX_GENERATION_ATTEMPTS = 400
MAX_POSITION_ATTEMPTS = 200
MAX_OUTER_MARGIN = 3

SEED_COUNT_RANGE = (1, 3)
SEED_LENGTH_RANGE = (3, 8)
BASE_EXTENSION_RANGE = (0, 3)
CROSS_ARM_RANGE = (1, 4)
USED_CROSS_COUNT_RANGE = (1, 4)
EXTRA_CROSS_COUNT_RANGE = (0, 2)


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _straight_segment_info(coords: np.ndarray) -> tuple[str, list[tuple[int, int]]] | None:
    if len(coords) < 3:
        return None

    rows = coords[:, 0]
    cols = coords[:, 1]
    if np.all(rows == rows[0]):
        ordered_cols = np.sort(cols)
        want = np.arange(int(ordered_cols[0]), int(ordered_cols[-1]) + 1)
        if len(ordered_cols) == len(want) and np.array_equal(ordered_cols, want):
            row = int(rows[0])
            return "h", [(row, int(col)) for col in ordered_cols]

    if np.all(cols == cols[0]):
        ordered_rows = np.sort(rows)
        want = np.arange(int(ordered_rows[0]), int(ordered_rows[-1]) + 1)
        if len(ordered_rows) == len(want) and np.array_equal(ordered_rows, want):
            col = int(cols[0])
            return "v", [(int(row), col) for row in ordered_rows]

    return None


def _find_seed_segments(grid: np.ndarray, bg: int) -> list[tuple[int, str, list[tuple[int, int]]]]:
    segments: list[tuple[int, str, list[tuple[int, int]]]] = []
    non_seed_colors: list[int] = []

    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        info = _straight_segment_info(np.argwhere(grid == color))
        if info is None:
            non_seed_colors.append(color)
            continue
        orientation, cells = info
        segments.append((color, orientation, cells))

    assert len(non_seed_colors) == 1, "expected one non-seed grid color"
    assert 1 <= len(segments) <= 3, "expected one to three colored seed segments"
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow same-colored orthogonal branches at every grey-grid crossing covered by a colored seed segment.

    Input structure:
    - A black background contains grey horizontal and vertical line segments that cross into an irregular grid skeleton.
    - One to three non-grey colors each form a single straight horizontal or vertical seed segment laid on top of part of a grey line.
    - A seed segment may cover one or more grey crossings where a perpendicular grey line continues through the colored cell.

    Rule:
    - Keep the original grey skeleton and the original colored seed segments in place.
    - For every colored seed cell that sits on a grey crossing, extend that seed color along the seed's perpendicular direction.
    - In each perpendicular direction, keep coloring cells until the underlying non-background skeleton stops.

    Key insight:
    - The colored segment tells you which axis is the trunk, while the surrounding non-background scaffold reveals where perpendicular grey lines pass through it and how far each branch can grow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _dominant_color(g)
    support = g != bg
    segments = _find_seed_segments(g, bg)
    out = g.copy()
    height, width = g.shape

    for color, orientation, cells in segments:
        if orientation == "h":
            row = cells[0][0]
            for _row, col in cells:
                if not (0 < row < height - 1):
                    continue
                if not (support[row - 1, col] and support[row + 1, col]):
                    continue
                rr = row - 1
                while rr >= 0 and support[rr, col]:
                    out[rr, col] = color
                    rr -= 1
                rr = row + 1
                while rr < height and support[rr, col]:
                    out[rr, col] = color
                    rr += 1
        else:
            col = cells[0][1]
            for row, _col in cells:
                if not (0 < col < width - 1):
                    continue
                if not (support[row, col - 1] and support[row, col + 1]):
                    continue
                cc = col - 1
                while cc >= 0 and support[row, cc]:
                    out[row, cc] = color
                    cc -= 1
                cc = col + 1
                while cc < width and support[row, cc]:
                    out[row, cc] = color
                    cc += 1

    return out


def _build_horizontal_cluster(color: int) -> tuple[np.ndarray, np.ndarray]:
    seed_len = random.randint(*SEED_LENGTH_RANGE)
    left_ext = random.randint(*BASE_EXTENSION_RANGE)
    right_ext = random.randint(*BASE_EXTENSION_RANGE)
    if left_ext + right_ext == 0:
        if random.random() < 0.5:
            left_ext = 1
        else:
            right_ext = 1

    width = left_ext + seed_len + right_ext
    seed_start = left_ext
    seed_end = seed_start + seed_len - 1

    used_count = random.randint(USED_CROSS_COUNT_RANGE[0], min(USED_CROSS_COUNT_RANGE[1], seed_len))
    used_cols = sorted(random.sample(range(seed_start, seed_end + 1), used_count))

    extra_choices = list(range(seed_start)) + list(range(seed_end + 1, width))
    extra_limit = min(EXTRA_CROSS_COUNT_RANGE[1], len(extra_choices))
    extra_count = random.randint(EXTRA_CROSS_COUNT_RANGE[0], extra_limit) if extra_limit else 0
    extra_cols = sorted(random.sample(extra_choices, extra_count)) if extra_count else []

    arm_lengths = {
        col: (
            random.randint(*CROSS_ARM_RANGE),
            random.randint(*CROSS_ARM_RANGE),
        )
        for col in used_cols + extra_cols
    }
    max_up = max(up for up, _down in arm_lengths.values())
    max_down = max(down for _up, down in arm_lengths.values())
    row = max_up
    height = max_up + 1 + max_down

    inp = np.full((height, width), BG_COLOR, dtype=int)
    inp[row, :] = GRID_COLOR
    for col, (up, down) in arm_lengths.items():
        inp[row - up : row + down + 1, col] = GRID_COLOR
    inp[row, seed_start : seed_end + 1] = color

    out = inp.copy()
    for col in used_cols:
        up, down = arm_lengths[col]
        out[row - up : row + down + 1, col] = color

    return inp, out


def _orient_cluster(inp: np.ndarray, out: np.ndarray, orientation: str) -> tuple[np.ndarray, np.ndarray]:
    if orientation == "v":
        k = random.choice((1, 3))
        inp = np.rot90(inp, k)
        out = np.rot90(out, k)

    if random.random() < 0.5:
        inp = np.flipud(inp)
        out = np.flipud(out)
    if random.random() < 0.5:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    return np.ascontiguousarray(inp), np.ascontiguousarray(out)


def _make_cluster(color: int, orientation: str) -> tuple[np.ndarray, np.ndarray]:
    inp, out = _build_horizontal_cluster(color)
    return _orient_cluster(inp, out, orientation)


def _boxes_clear(top: int, left: int, height: int, width: int, boxes: list[tuple[int, int, int, int]]) -> bool:
    bottom = top + height - 1
    right = left + width - 1
    for other_top, other_left, other_bottom, other_right in boxes:
        if not (
            right + CLUSTER_PADDING < other_left
            or other_right + CLUSTER_PADDING < left
            or bottom + CLUSTER_PADDING < other_top
            or other_bottom + CLUSTER_PADDING < top
        ):
            return False
    return True


def _place_clusters(
    clusters: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray] | None:
    stage_inp = np.full((STAGING_SIZE, STAGING_SIZE), BG_COLOR, dtype=int)
    stage_out = np.full_like(stage_inp, BG_COLOR)
    order = sorted(range(len(clusters)), key=lambda idx: clusters[idx][0].shape[0] * clusters[idx][0].shape[1], reverse=True)
    boxes: list[tuple[int, int, int, int]] = []
    positions: dict[int, tuple[int, int]] = {}

    for idx in order:
        patch_inp, _patch_out = clusters[idx]
        height, width = patch_inp.shape
        if height > STAGING_SIZE or width > STAGING_SIZE:
            return None

        placed = False
        for _ in range(MAX_POSITION_ATTEMPTS):
            top = random.randint(0, STAGING_SIZE - height)
            left = random.randint(0, STAGING_SIZE - width)
            if not _boxes_clear(top, left, height, width, boxes):
                continue
            positions[idx] = (top, left)
            boxes.append((top, left, top + height - 1, left + width - 1))
            placed = True
            break
        if not placed:
            return None

    for idx, (patch_inp, patch_out) in enumerate(clusters):
        top, left = positions[idx]
        height, width = patch_inp.shape
        stage_inp[top : top + height, left : left + width] = patch_inp
        stage_out[top : top + height, left : left + width] = patch_out

    return stage_inp, stage_out


def _crop_stage(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != BG_COLOR)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)

    top_margin = random.randint(1, min(MAX_OUTER_MARGIN, int(r0))) if r0 > 0 else 0
    bottom_margin = random.randint(1, min(MAX_OUTER_MARGIN, int(inp.shape[0] - 1 - r1))) if r1 + 1 < inp.shape[0] else 0
    left_margin = random.randint(1, min(MAX_OUTER_MARGIN, int(c0))) if c0 > 0 else 0
    right_margin = random.randint(1, min(MAX_OUTER_MARGIN, int(inp.shape[1] - 1 - c1))) if c1 + 1 < inp.shape[1] else 0

    rs = slice(int(r0) - top_margin, int(r1) + 1 + bottom_margin)
    cs = slice(int(c0) - left_margin, int(c1) + 1 + right_margin)
    return inp[rs, cs].copy(), out[rs, cs].copy()


def _sample_orientations(count: int) -> list[str]:
    if count == 1:
        return [random.choice(("h", "v"))]
    if count == 2:
        return list(random.choice((("h", "h"), ("h", "v"), ("v", "h"), ("v", "v"))))
    return list(random.choice((("h", "h", "v"), ("h", "v", "h"), ("v", "h", "h"), ("v", "v", "h"), ("v", "h", "v"), ("h", "v", "v"))))


def _gen_attempt() -> np.ndarray | None:
    seed_count = random.randint(*SEED_COUNT_RANGE)
    colors = random.sample(SEED_COLORS, seed_count)
    orientations = _sample_orientations(seed_count)
    clusters = [_make_cluster(color, orientation) for color, orientation in zip(colors, orientations)]

    placed = _place_clusters(clusters)
    if placed is None:
        return None
    stage_inp, stage_out = placed
    inp, expected = _crop_stage(stage_inp, stage_out)

    if _dominant_color(inp) != BG_COLOR:
        return None
    if np.array_equal(inp, expected):
        return None
    if not (1 <= len(_find_seed_segments(inp, BG_COLOR)) <= 3):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
