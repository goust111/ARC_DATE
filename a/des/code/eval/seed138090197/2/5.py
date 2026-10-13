from __future__ import annotations

import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))

# Solver assumptions mirrored by the generator.
MIN_INNER = 2
MAX_INNER = 7
PAIR_COUNT_OPTIONS = (1, 2, 3)
PAIR_MODES = ("same", "row_only", "col_only", "both")

# Canvas and placement.
GRID_MIN = 18
GRID_MAX = 30
PLACEMENT_PADDING = 1

# Content generation.
MIN_CONTENT = 2
MAX_CONTENT = 6
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _frame_record(coords: np.ndarray, color: int) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None
    if len(coords) != 2 * (height + width) - 4:
        return None

    sub = np.zeros((height, width), dtype=bool)
    sub[coords[:, 0] - r0, coords[:, 1] - c0] = True
    expected = np.zeros_like(sub)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if not np.array_equal(sub, expected):
        return None

    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "inner_shape": (height - 2, width - 2),
    }


def _parse_portal_pairs(grid: np.ndarray) -> list[tuple[dict[str, object], dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict[str, object]] = []

    for color in np.unique(g):
        color = int(color)
        if color == BG:
            continue
        for coords in _components(g == color):
            frame = _frame_record(coords, color)
            if frame is not None:
                frames.append(frame)

    assert frames, "expected at least one rectangular frame"

    by_color: dict[int, list[dict[str, object]]] = {}
    for frame in frames:
        by_color.setdefault(int(frame["color"]), []).append(frame)

    pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    for color, group in by_color.items():
        assert len(group) == 2, f"frame color {color} must appear exactly twice"
        sources: list[dict[str, object]] = []
        targets: list[dict[str, object]] = []
        for frame in group:
            r0, c0, r1, c1 = frame["bbox"]
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            if np.any(inner != BG):
                sources.append(frame)
            else:
                targets.append(frame)
        assert len(sources) == 1 and len(targets) == 1, "each frame pair needs one source and one target"
        pairs.append((sources[0], targets[0]))
    return pairs


def _round_half_up(value: float) -> int:
    return int(np.floor(value + 0.5))


def _map_index(index: int, src_len: int, dst_len: int) -> int:
    assert src_len >= 1 and dst_len >= 1
    if src_len == 1:
        return _round_half_up((dst_len - 1) / 2.0)
    return _round_half_up(index * (dst_len - 1) / (src_len - 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color rectangular frames and project the source contents into the empty twin frame.

    Input structure:
    - Background is black.
    - Foreground contains several one-pixel-thick hollow rectangular frames.
    - Each frame color appears exactly twice, and exactly one frame in each pair contains non-black pixels inside.

    Rule:
    - Match frames by color.
    - For every non-black pixel inside the source frame, convert its interior row/column indices to normalized
      coordinates within the source interior.
    - Map those normalized coordinates into the target interior and round each mapped axis to the nearest cell.
    - Draw the same pixel color at the mapped target location; keep all original frames and source contents unchanged.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for source, target in _parse_portal_pairs(g):
        sr0, sc0, sr1, sc1 = source["bbox"]
        tr0, tc0, tr1, tc1 = target["bbox"]
        src_inner = g[sr0 + 1 : sr1, sc0 + 1 : sc1]
        dst_inner = g[tr0 + 1 : tr1, tc0 + 1 : tc1]
        assert np.all(dst_inner == BG), "target frame interior must be empty"

        src_h, src_w = src_inner.shape
        dst_h, dst_w = dst_inner.shape
        for sr, sc in np.argwhere(src_inner != BG):
            tr = _map_index(int(sr), src_h, dst_h)
            tc = _map_index(int(sc), src_w, dst_w)
            out[tr0 + 1 + tr, tc0 + 1 + tc] = int(src_inner[sr, sc])

    return out


def _different_value(value: int) -> int:
    choices = [candidate for candidate in range(MIN_INNER, MAX_INNER + 1) if candidate != value]
    return random.choice(choices)


def _sample_inner_shape() -> tuple[int, int]:
    return random.randint(MIN_INNER, MAX_INNER), random.randint(MIN_INNER, MAX_INNER)


def _sample_pair_shapes(mode: str) -> tuple[tuple[int, int], tuple[int, int]]:
    src_h, src_w = _sample_inner_shape()
    dst_h, dst_w = src_h, src_w
    if mode == "row_only":
        dst_h = _different_value(src_h)
    elif mode == "col_only":
        dst_w = _different_value(src_w)
    elif mode == "both":
        dst_h = _different_value(src_h)
        dst_w = _different_value(src_w)
    elif mode != "same":
        raise ValueError(mode)
    return (src_h, src_w), (dst_h, dst_w)


def _sample_pair_modes(pair_count: int) -> list[str]:
    modes = [random.choice(PAIR_MODES) for _ in range(pair_count)]
    if pair_count >= 2 and all(mode == "same" for mode in modes):
        modes[random.randrange(pair_count)] = random.choice(PAIR_MODES[1:])
    return modes


def _sample_grid_shape(rectangles: list[tuple[int, int]]) -> tuple[int, int] | None:
    total_area = sum(height * width for height, width in rectangles)
    max_height = max(height for height, _width in rectangles)
    max_width = max(width for _height, width in rectangles)

    row_min = max(GRID_MIN, max_height + 2)
    col_min = max(GRID_MIN, max_width + 2)
    if row_min > GRID_MAX or col_min > GRID_MAX:
        return None

    for _ in range(80):
        rows = random.randint(row_min, GRID_MAX)
        cols = random.randint(col_min, GRID_MAX)
        if rows * cols >= total_area * 2:
            return rows, cols
    return None


def _can_place(occupied: np.ndarray, r0: int, c0: int, height: int, width: int) -> bool:
    rows, cols = occupied.shape
    rr0 = max(0, r0 - PLACEMENT_PADDING)
    rr1 = min(rows, r0 + height + PLACEMENT_PADDING)
    cc0 = max(0, c0 - PLACEMENT_PADDING)
    cc1 = min(cols, c0 + width + PLACEMENT_PADDING)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_frames(
    grid_shape: tuple[int, int], pair_specs: list[dict[str, object]]
) -> dict[tuple[int, int], tuple[int, int]] | None:
    rows, cols = grid_shape
    occupied = np.zeros((rows, cols), dtype=bool)
    placements: dict[tuple[int, int], tuple[int, int]] = {}

    items: list[dict[str, object]] = []
    for pair_index, spec in enumerate(pair_specs):
        shape_a, shape_b = spec["shapes"]
        items.append({"key": (pair_index, 0), "shape": shape_a})
        items.append({"key": (pair_index, 1), "shape": shape_b})
    items.sort(key=lambda item: item["shape"][0] * item["shape"][1], reverse=True)

    for item in items:
        height, width = item["shape"]
        candidates = [
            (r0, c0)
            for r0 in range(rows - height + 1)
            for c0 in range(cols - width + 1)
            if _can_place(occupied, r0, c0, height, width)
        ]
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placements[item["key"]] = (r0, c0)
        occupied[r0 : r0 + height, c0 : c0 + width] = True

    return placements


def _draw_frame(grid: np.ndarray, top: int, left: int, outer_h: int, outer_w: int, color: int) -> None:
    bottom = top + outer_h - 1
    right = left + outer_w - 1
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color


def _sample_content(
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
    disallowed_color: int,
    allowed_colors: list[int],
) -> list[tuple[int, int, int]]:
    src_h, src_w = src_shape
    dst_h, dst_w = dst_shape
    buckets: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for sr in range(src_h):
        tr = _map_index(sr, src_h, dst_h)
        for sc in range(src_w):
            tc = _map_index(sc, src_w, dst_w)
            buckets.setdefault((tr, tc), []).append((sr, sc))

    target_positions = list(buckets.items())
    random.shuffle(target_positions)
    max_count = min(MAX_CONTENT, len(target_positions))
    assert max_count >= MIN_CONTENT
    count = random.randint(MIN_CONTENT, max_count)

    # Prefer a pattern that is not a single row or a single column when the source interior is large enough.
    for _ in range(40):
        selected_positions: list[tuple[int, int]] = []
        chosen_targets = random.sample(target_positions, count)
        for _mapped_target, source_options in chosen_targets:
            selected_positions.append(random.choice(source_options))
        rows = {row for row, _col in selected_positions}
        cols = {col for _row, col in selected_positions}
        if count < 3 or len(rows) > 1 or len(cols) > 1:
            break

    usable_colors = [color for color in allowed_colors if color != disallowed_color]
    palette_size = min(len(usable_colors), random.randint(2, min(4, len(usable_colors))))
    palette = random.sample(usable_colors, palette_size)
    return [
        (row, col, random.choice(palette))
        for row, col in selected_positions
    ]


def _build_candidate() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_OPTIONS)
    frame_colors = random.sample(COLOR_POOL, pair_count)
    pair_modes = _sample_pair_modes(pair_count)

    pair_specs: list[dict[str, object]] = []
    rectangles: list[tuple[int, int]] = []
    for pair_index in range(pair_count):
        inner_a, inner_b = _sample_pair_shapes(pair_modes[pair_index])
        outer_a = (inner_a[0] + 2, inner_a[1] + 2)
        outer_b = (inner_b[0] + 2, inner_b[1] + 2)
        source_slot = random.randrange(2)
        pair_specs.append(
            {
                "color": frame_colors[pair_index],
                "shapes": (outer_a, outer_b),
                "inners": (inner_a, inner_b),
                "source_slot": source_slot,
            }
        )
        rectangles.extend([outer_a, outer_b])

    grid_shape = _sample_grid_shape(rectangles)
    if grid_shape is None:
        return None

    placements = _place_frames(grid_shape, pair_specs)
    if placements is None:
        return None

    grid = np.full(grid_shape, BG, dtype=int)
    content_colors = [color for color in COLOR_POOL if color not in frame_colors]

    for pair_index, spec in enumerate(pair_specs):
        frame_color = int(spec["color"])
        source_slot = int(spec["source_slot"])
        inner_shapes = spec["inners"]
        outer_shapes = spec["shapes"]

        for slot in range(2):
            top, left = placements[(pair_index, slot)]
            outer_h, outer_w = outer_shapes[slot]
            _draw_frame(grid, top, left, outer_h, outer_w, frame_color)

        source_inner = inner_shapes[source_slot]
        target_inner = inner_shapes[1 - source_slot]
        source_top, source_left = placements[(pair_index, source_slot)]
        content = _sample_content(source_inner, target_inner, frame_color, content_colors)
        for row, col, color in content:
            grid[source_top + 1 + row, source_left + 1 + col] = color

    try:
        pairs = _parse_portal_pairs(grid)
        if len(pairs) != pair_count:
            return None
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _build_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid portal-frame input")
