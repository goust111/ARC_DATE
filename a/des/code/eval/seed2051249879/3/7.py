from __future__ import annotations

import random
from collections import Counter

import numpy as np


RED = 2
GREY = 5
GRID_MAX = 30
PATTERN_MIN = 15
PATTERN_MAX = 25
INNER_MIN = 3
INNER_MAX = 5
LAYOUTS = (
    (2, 2),
    (2, 3),
    (3, 2),
    (3, 3),
    (2, 4),
    (4, 2),
)
SAFE_COLORS = tuple(color for color in range(10) if color not in (RED, GREY))
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).reshape(-1), minlength=10)
    return int(np.argmax(counts))


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

    components: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            points: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                points.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(points, dtype=int))
    return components


def _find_window_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for coords in components_mask(grid == RED, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width < 3:
            continue
        expected_area = 2 * height + 2 * width - 4
        if int(coords.shape[0]) != expected_area:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((height, width), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(sub[border] == RED):
            continue
        if np.any(sub[~border] == RED):
            continue
        frames.append((r0, c0, r1, c1))
    assert frames, "expected at least one red window frame"
    return frames


def _group_windows_by_row(windows: list[dict[str, object]], tolerance: int) -> list[list[dict[str, object]]]:
    ordered = sorted(windows, key=lambda item: (item["cy"], item["cx"]))  # type: ignore[index]
    rows: list[list[dict[str, object]]] = []
    current = [ordered[0]]
    for window in ordered[1:]:
        mean_y = sum(float(item["cy"]) for item in current) / len(current)
        if abs(float(window["cy"]) - mean_y) <= float(tolerance):
            current.append(window)
        else:
            rows.append(sorted(current, key=lambda item: float(item["cx"])))
            current = [window]
    rows.append(sorted(current, key=lambda item: float(item["cx"])))
    return rows


def _compose_samples(sample_rows: list[list[np.ndarray]], sep_color: int = GREY) -> np.ndarray:
    inner_h, inner_w = sample_rows[0][0].shape
    row_count = len(sample_rows)
    col_count = len(sample_rows[0])
    out_h = row_count * inner_h + (row_count - 1)
    out_w = col_count * inner_w + (col_count - 1)
    out = np.full((out_h, out_w), int(sep_color), dtype=int)
    for row_idx, row in enumerate(sample_rows):
        for col_idx, sample in enumerate(row):
            top = row_idx * (inner_h + 1)
            left = col_idx * (inner_w + 1)
            out[top : top + inner_h, left : left + inner_w] = sample
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the interiors of red sampling windows and reassemble them by the windows' spatial layout.

    Input structure:
    - bg = most common color.
    - A large multi-color pattern fills most of the grid.
    - Several single-pixel red rectangular frames sit on top of that pattern.
    - All frames have the same interior size.

    Rule:
    - Detect every red frame and crop its interior.
    - Group frames into rows by nearby center y positions.
    - Sort each row left-to-right by center x.
    - Tile the cropped interiors in that row/column order, with one-pixel grey separators.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _find_window_frames(g)
    inner_shapes = {(r1 - r0 - 1, c1 - c0 - 1) for r0, c0, r1, c1 in frames}
    assert len(inner_shapes) == 1, "all windows must share one interior size"
    inner_h, inner_w = next(iter(inner_shapes))
    assert inner_h > 0 and inner_w > 0

    windows: list[dict[str, object]] = []
    for r0, c0, r1, c1 in frames:
        windows.append(
            {
                "cy": (r0 + r1) / 2.0,
                "cx": (c0 + c1) / 2.0,
                "sample": g[r0 + 1 : r1, c0 + 1 : c1].copy(),
            }
        )

    rows = _group_windows_by_row(windows, tolerance=inner_h)
    col_count = len(rows[0])
    assert all(len(row) == col_count for row in rows), "window rows must form a full matrix"
    sample_rows = [[item["sample"] for item in row] for row in rows]
    return _compose_samples(sample_rows, sep_color=GREY)


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    return list(np.random.multinomial(int(total), [1.0 / parts] * parts))


def _pack_starts(length: int, count: int, size: int, min_gap: int) -> list[int] | None:
    required = count * size + (count - 1) * min_gap
    extra = int(length) - int(required)
    if extra < 0:
        return None
    gaps = _random_composition(extra, count + 1)
    starts: list[int] = []
    pos = gaps[0]
    for idx in range(count):
        starts.append(int(pos))
        pos += size
        if idx + 1 < count:
            pos += min_gap + gaps[idx + 1]
    return starts


def _jittered_window_positions(
    row_starts: list[int],
    col_starts: list[int],
    frame_h: int,
    frame_w: int,
    pattern_h: int,
    pattern_w: int,
) -> list[list[tuple[int, int]]]:
    boxes: list[tuple[int, int, int, int]] = []
    placements: list[list[tuple[int, int]]] = []
    for row_start in row_starts:
        row_positions: list[tuple[int, int]] = []
        for col_start in col_starts:
            deltas = [(0, 0)]
            extras = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
            random.shuffle(extras)
            deltas.extend(extras)
            chosen = None
            for dr, dc in deltas:
                top = int(row_start + dr)
                left = int(col_start + dc)
                box = (top, left, top + frame_h - 1, left + frame_w - 1)
                if top < 0 or left < 0 or box[2] >= pattern_h or box[3] >= pattern_w:
                    continue
                overlap = False
                for other in boxes:
                    if not (box[2] < other[0] or other[2] < box[0] or box[3] < other[1] or other[3] < box[1]):
                        overlap = True
                        break
                if not overlap:
                    chosen = (top, left)
                    boxes.append(box)
                    break
            assert chosen is not None
            row_positions.append(chosen)
        placements.append(row_positions)
    return placements


def _safe_palette(bg: int, size: int) -> list[int]:
    choices = [color for color in SAFE_COLORS if color != bg]
    random.shuffle(choices)
    return choices[:size]


def _build_pattern(height: int, width: int, palette: list[int]) -> np.ndarray:
    rr, cc = np.indices((height, width))
    colors = np.array(palette, dtype=int)
    style = random.randrange(3)
    if style == 0:
        cell_h = random.randint(2, 4)
        cell_w = random.randint(2, 4)
        idx = ((rr // cell_h) + 2 * (cc // cell_w) + ((rr + cc) // max(2, min(cell_h, cell_w)))) % len(colors)
    elif style == 1:
        slope = random.randint(2, 4)
        idx = (((rr + 2 * cc) // slope) + (rr // random.randint(2, 4))) % len(colors)
    else:
        cell_h = random.randint(2, 4)
        cell_w = random.randint(2, 4)
        idx = (((rr // cell_h) ^ (cc // cell_w)) + ((rr + cc) // random.randint(3, 5))) % len(colors)
    pattern = colors[idx].copy()

    for _ in range(random.randint(3, 5)):
        accent = int(random.choice(palette))
        mode = random.choice(("hbar", "vbar", "diag", "frame", "block"))
        if mode == "hbar":
            band_h = random.randint(1, max(1, height // 5))
            top = random.randint(0, height - band_h)
            pattern[top : top + band_h, :] = accent
        elif mode == "vbar":
            band_w = random.randint(1, max(1, width // 5))
            left = random.randint(0, width - band_w)
            pattern[:, left : left + band_w] = accent
        elif mode == "diag":
            shift = random.randint(-width // 3, width // 3)
            thickness = random.randint(1, 2)
            pattern[np.abs(rr - cc - shift) <= thickness] = accent
        elif mode == "frame" and height >= 7 and width >= 7:
            top = random.randint(1, max(1, height // 3))
            left = random.randint(1, max(1, width // 3))
            bottom = random.randint(max(top + 2, 2 * height // 3), height - 2)
            right = random.randint(max(left + 2, 2 * width // 3), width - 2)
            pattern[top, left : right + 1] = accent
            pattern[bottom, left : right + 1] = accent
            pattern[top : bottom + 1, left] = accent
            pattern[top : bottom + 1, right] = accent
        else:
            block_h = random.randint(2, max(2, height // 4))
            block_w = random.randint(2, max(2, width // 4))
            top = random.randint(0, height - block_h)
            left = random.randint(0, width - block_w)
            pattern[top : top + block_h, left : left + block_w] = accent

    return pattern


def _choose_geometry(rows: int, cols: int) -> tuple[int, int, int, int, int, int] | None:
    feasible: list[tuple[int, int, int, int, int, int]] = []
    min_gap = 1
    for inner_h in range(INNER_MIN, INNER_MAX + 1):
        for inner_w in range(INNER_MIN, INNER_MAX + 1):
            frame_h = inner_h + 2
            frame_w = inner_w + 2
            min_pattern_h = max(PATTERN_MIN, rows * frame_h + (rows - 1) * min_gap)
            min_pattern_w = max(PATTERN_MIN, cols * frame_w + (cols - 1) * min_gap)
            if min_pattern_h > PATTERN_MAX or min_pattern_w > PATTERN_MAX:
                continue
            max_pattern_h = min(PATTERN_MAX, min_pattern_h + 5)
            max_pattern_w = min(PATTERN_MAX, min_pattern_w + 5)
            feasible.append(
                (
                    inner_h,
                    inner_w,
                    frame_h,
                    frame_w,
                    random.randint(min_pattern_h, max_pattern_h),
                    random.randint(min_pattern_w, max_pattern_w),
                )
            )
    if not feasible:
        return None
    inner_h, inner_w, frame_h, frame_w, pattern_h, pattern_w = random.choice(feasible)
    return inner_h, inner_w, frame_h, frame_w, pattern_h, pattern_w


def _choose_margin_pair(length: int) -> tuple[int, int] | None:
    pairs = [
        (before, after)
        for before in (2, 3, 4)
        for after in (2, 3, 4)
        if before + after + length <= GRID_MAX
    ]
    if not pairs:
        return None
    return random.choice(pairs)


def _expected_output_from_samples(sample_rows: list[list[np.ndarray]]) -> np.ndarray:
    return _compose_samples(sample_rows, sep_color=GREY)


def _gen_attempt() -> np.ndarray | None:
    rows, cols = random.choice(LAYOUTS)
    geometry = _choose_geometry(rows, cols)
    if geometry is None:
        return None
    inner_h, inner_w, frame_h, frame_w, pattern_h, pattern_w = geometry

    bg = random.choice(SAFE_COLORS)
    palette = _safe_palette(bg, random.randint(4, 6))
    pattern = _build_pattern(pattern_h, pattern_w, palette)
    if np.unique(pattern).size < 3:
        return None

    row_starts = _pack_starts(pattern_h, rows, frame_h, min_gap=1)
    col_starts = _pack_starts(pattern_w, cols, frame_w, min_gap=1)
    if row_starts is None or col_starts is None:
        return None
    window_positions = _jittered_window_positions(
        row_starts, col_starts, frame_h, frame_w, pattern_h, pattern_w
    )

    sample_rows: list[list[np.ndarray]] = []
    sample_signatures = set()
    for row in window_positions:
        sample_row: list[np.ndarray] = []
        for top, left in row:
            sample = pattern[top + 1 : top + 1 + inner_h, left + 1 : left + 1 + inner_w].copy()
            sample_row.append(sample)
            sample_signatures.add(sample.tobytes())
        sample_rows.append(sample_row)
    if len(sample_signatures) < max(2, min(4, rows * cols // 2)):
        return None

    vertical_margins = _choose_margin_pair(pattern_h)
    horizontal_margins = _choose_margin_pair(pattern_w)
    if vertical_margins is None or horizontal_margins is None:
        return None
    top_margin, bottom_margin = vertical_margins
    left_margin, right_margin = horizontal_margins

    grid = np.full(
        (top_margin + pattern_h + bottom_margin, left_margin + pattern_w + right_margin),
        int(bg),
        dtype=int,
    )
    pr0 = top_margin
    pc0 = left_margin
    grid[pr0 : pr0 + pattern_h, pc0 : pc0 + pattern_w] = pattern

    for row in window_positions:
        for top, left in row:
            r0 = pr0 + top
            c0 = pc0 + left
            r1 = r0 + frame_h - 1
            c1 = c0 + frame_w - 1
            grid[r0, c0 : c1 + 1] = RED
            grid[r1, c0 : c1 + 1] = RED
            grid[r0 : r1 + 1, c0] = RED
            grid[r0 : r1 + 1, c1] = RED

    counts = Counter(map(int, grid.reshape(-1)))
    if any(color != bg and count >= counts[bg] for color, count in counts.items()):
        return None

    expected = _expected_output_from_samples(sample_rows)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid window-sampling puzzle")
