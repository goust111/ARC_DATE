from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


LEGEND_COLOR = 7  # orange
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)

                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            components.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                    "coords": coords,
                }
            )

    return components


def _bbox_size(component: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = component["bbox"]
    return r1 - r0 + 1, c1 - c0 + 1


def _is_solid_square(grid: np.ndarray, component: dict) -> bool:
    h, w = _bbox_size(component)
    if h != w or component["area"] != h * w:
        return False
    r0, c0, r1, c1 = component["bbox"]
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == component["color"]))


def _is_hollow_square_frame(grid: np.ndarray, component: dict, bg: int) -> bool:
    h, w = _bbox_size(component)
    if h != w or h < 3 or component["area"] != 4 * h - 4:
        return False

    r0, c0, r1, c1 = component["bbox"]
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    color = component["color"]
    if not np.all(sub[0, :] == color):
        return False
    if not np.all(sub[-1, :] == color):
        return False
    if not np.all(sub[:, 0] == color):
        return False
    if not np.all(sub[:, -1] == color):
        return False
    if np.any(sub[1:-1, 1:-1] != bg):
        return False
    return True


def _pick_frames(grid: np.ndarray, components: list[dict], bg: int) -> list[dict]:
    frames = [component for component in components if _is_hollow_square_frame(grid, component, bg)]
    if not frames:
        raise ValueError("no container frames found")

    size_counts: dict[int, int] = defaultdict(int)
    for frame in frames:
        h, _ = _bbox_size(frame)
        size_counts[h] += 1
    target_size = max(size_counts.items(), key=lambda item: (item[1], item[0]))[0]
    return [frame for frame in frames if _bbox_size(frame)[0] == target_size]


def _pick_legend(grid: np.ndarray, components: list[dict], frames: list[dict]) -> list[dict]:
    frame_left = min(frame["bbox"][1] for frame in frames)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for component in components:
        by_color[component["color"]].append(component)

    best: tuple[tuple[int, int, int], list[dict]] | None = None
    for color, parts in by_color.items():
        if len(parts) < 2:
            continue
        if max(part["bbox"][3] for part in parts) >= frame_left:
            continue
        if not all(_is_solid_square(grid, part) for part in parts):
            continue

        ordered = sorted(parts, key=lambda part: (part["bbox"][0], part["bbox"][1]))
        sizes = [_bbox_size(part)[0] for part in ordered]
        if sizes != list(range(1, len(ordered) + 1)):
            continue
        if any(ordered[i]["bbox"][2] >= ordered[i + 1]["bbox"][0] for i in range(len(ordered) - 1)):
            continue
        if max(part["bbox"][1] for part in ordered) > min(part["bbox"][3] for part in ordered):
            continue

        score = (len(ordered), -max(part["bbox"][3] for part in ordered), -ordered[0]["bbox"][0])
        if best is None or score > best[0]:
            best = (score, ordered)

    if best is None:
        raise ValueError("legend not found")
    return best[1]


def _mask_from_components(shape: tuple[int, int], components: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for component in components:
        for r, c in component["coords"]:
            mask[r, c] = True
    return mask


def _draw_frame(canvas: np.ndarray, top: int, left: int, size: int, color: int) -> None:
    canvas[top, left : left + size] = color
    canvas[top + size - 1, left : left + size] = color
    canvas[top : top + size, left] = color
    canvas[top : top + size, left + size - 1] = color


def _fill_centered_square(canvas: np.ndarray, top: int, left: int, outer_size: int, fill_size: int, color: int) -> None:
    inner_size = outer_size - 2
    if fill_size <= 0 or fill_size > inner_size:
        return
    start_r = top + 1 + (inner_size - fill_size) // 2
    start_c = left + 1 + (inner_size - fill_size) // 2
    canvas[start_r : start_r + fill_size, start_c : start_c + fill_size] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _find_components(g, bg)
    frames = _pick_frames(g, components, bg)
    legend = _pick_legend(g, components, frames)

    legend_sizes = {_bbox_size(part)[0] for part in legend}
    frame_mask = _mask_from_components(g.shape, frames)
    legend_mask = _mask_from_components(g.shape, legend)

    frame_by_color = {int(frame["color"]): frame for frame in frames}
    counts: dict[int, int] = {}
    for color in frame_by_color:
        counts[color] = int(np.count_nonzero((g == color) & (~frame_mask) & (~legend_mask)))

    crop_r0 = min(frame["bbox"][0] for frame in frames)
    crop_c0 = min(frame["bbox"][1] for frame in frames)
    crop_r1 = max(frame["bbox"][2] for frame in frames)
    crop_c1 = max(frame["bbox"][3] for frame in frames)
    out = np.full((crop_r1 - crop_r0 + 1, crop_c1 - crop_c0 + 1), bg, dtype=int)

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        rr0 = r0 - crop_r0
        cc0 = c0 - crop_c0
        size = r1 - r0 + 1
        color = int(frame["color"])
        _draw_frame(out, rr0, cc0, size, color)
        fill_size = counts[color]
        if fill_size in legend_sizes:
            _fill_centered_square(out, rr0, cc0, size, fill_size, color)

    return out


def _sample_counts(rng: np.random.Generator, n_frames: int, max_k: int) -> list[int]:
    counts = rng.integers(1, max_k + 1, size=n_frames).tolist()
    counts[int(rng.integers(0, n_frames))] = max_k
    return [int(value) for value in counts]


def _row_positions(rng: np.random.Generator, n_frames: int, box_size: int) -> tuple[list[tuple[int, int]], int, int]:
    gaps = [int(rng.integers(4, 7)) for _ in range(n_frames - 1)]
    positions: list[tuple[int, int]] = []
    col = 0
    for i in range(n_frames):
        positions.append((0, col))
        if i < n_frames - 1:
            col += box_size + gaps[i]
    width = positions[-1][1] + box_size
    return positions, box_size, width


def _column_positions(rng: np.random.Generator, n_frames: int, box_size: int) -> tuple[list[tuple[int, int]], int, int]:
    gaps = [int(rng.integers(4, 7)) for _ in range(n_frames - 1)]
    positions: list[tuple[int, int]] = []
    row = 0
    for i in range(n_frames):
        positions.append((row, 0))
        if i < n_frames - 1:
            row += box_size + gaps[i]
    height = positions[-1][0] + box_size
    return positions, height, box_size


def _pack_row(rng: np.random.Generator, count: int, box_size: int) -> tuple[list[int], int]:
    gaps = [int(rng.integers(4, 7)) for _ in range(max(0, count - 1))]
    cols: list[int] = []
    col = 0
    for i in range(count):
        cols.append(col)
        if i < count - 1:
            col += box_size + gaps[i]
    width = 0 if not cols else cols[-1] + box_size
    return cols, width


def _grid_positions(rng: np.random.Generator, n_frames: int, box_size: int) -> tuple[list[tuple[int, int]], int, int]:
    top_count = int(rng.integers(1, n_frames))
    bottom_count = n_frames - top_count
    if bottom_count == 0:
        top_count -= 1
        bottom_count = 1

    top_cols, top_width = _pack_row(rng, top_count, box_size)
    bottom_cols, bottom_width = _pack_row(rng, bottom_count, box_size)
    width = max(top_width, bottom_width)
    top_offset = 0 if top_width == width else int(rng.integers(0, width - top_width + 1))
    bottom_offset = 0 if bottom_width == width else int(rng.integers(0, width - bottom_width + 1))
    row_gap = int(rng.integers(4, 7))

    positions: list[tuple[int, int]] = []
    for col in top_cols:
        positions.append((0, top_offset + col))
    for col in bottom_cols:
        positions.append((box_size + row_gap, bottom_offset + col))

    height = 2 * box_size + row_gap
    return positions, height, width


def _build_positions(rng: np.random.Generator, n_frames: int, box_size: int, layout: str) -> tuple[list[tuple[int, int]], int, int]:
    if layout == "row":
        return _row_positions(rng, n_frames, box_size)
    if layout == "column":
        return _column_positions(rng, n_frames, box_size)
    return _grid_positions(rng, n_frames, box_size)


def _can_place_scatter(canvas: np.ndarray, forbidden: np.ndarray, r: int, c: int, bg: int) -> bool:
    if int(canvas[r, c]) != bg:
        return False
    if forbidden[r, c]:
        return False
    h, w = canvas.shape
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < h and 0 <= nc < w and int(canvas[nr, nc]) != bg:
            return False
    return True


def _place_scatter_pixels(
    canvas: np.ndarray,
    forbidden: np.ndarray,
    color_queue: list[int],
    rng: np.random.Generator,
    bg: int,
) -> bool:
    h, w = canvas.shape
    for color in color_queue:
        candidates = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if _can_place_scatter(canvas, forbidden, r, c, bg)
        ]
        if not candidates:
            return False
        idx = int(rng.integers(0, len(candidates)))
        r, c = candidates[idx]
        canvas[r, c] = color
    return True


def _build_legend(max_k: int, bg: int, rng: np.random.Generator) -> np.ndarray:
    gaps = [int(rng.integers(1, 3)) for _ in range(max_k - 1)]
    height = sum(range(1, max_k + 1)) + sum(gaps)
    legend = np.full((height, max_k), bg, dtype=int)
    row = 0
    for size in range(1, max_k + 1):
        legend[row : row + size, :size] = LEGEND_COLOR
        if size < max_k:
            row += size + gaps[size - 1]
    return legend


def _build_expected_main(
    shape: tuple[int, int],
    positions: list[tuple[int, int]],
    box_size: int,
    frame_colors: list[int],
    counts: list[int],
    bg: int,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for (top, left), color, count in zip(positions, frame_colors, counts):
        _draw_frame(out, top, left, box_size, color)
        _fill_centered_square(out, top, left, box_size, count, color)
    return out


def _build_input_main(
    shape: tuple[int, int],
    positions: list[tuple[int, int]],
    box_size: int,
    frame_colors: list[int],
    counts: list[int],
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    inp = np.full(shape, bg, dtype=int)
    forbidden = np.zeros(shape, dtype=bool)
    for (top, left), color in zip(positions, frame_colors):
        _draw_frame(inp, top, left, box_size, color)
        forbidden[top : top + box_size, left : left + box_size] = True

    color_queue = [color for color, count in zip(frame_colors, counts) for _ in range(count)]
    rng.shuffle(color_queue)
    if not _place_scatter_pixels(inp, forbidden, color_queue, rng, bg):
        return None
    return inp


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = 0
    n_frames = int(rng.integers(3, 6))
    max_k = int(rng.integers(2, 6))
    inner_size = max_k + int(rng.integers(0, 3))
    box_size = inner_size + 2
    counts = _sample_counts(rng, n_frames, max_k)

    palette = [1, 2, 3, 4, 5, 6, 8, 9]
    rng.shuffle(palette)
    frame_colors = palette[:n_frames]

    layout = str(rng.choice(np.array(["row", "column", "grid"], dtype=object)))
    positions, main_h, main_w = _build_positions(rng, n_frames, box_size, layout)
    expected_main = _build_expected_main((main_h, main_w), positions, box_size, frame_colors, counts, bg)
    input_main = _build_input_main((main_h, main_w), positions, box_size, frame_colors, counts, bg, rng)
    if input_main is None:
        return None

    legend = _build_legend(max_k, bg, rng)
    legend_h, legend_w = legend.shape
    main_top = int(rng.integers(0, 4))
    legend_top = int(rng.integers(0, 4))
    left_margin = int(rng.integers(0, 3))
    between_gap = int(rng.integers(3, 7))
    right_margin = int(rng.integers(0, 4))
    bottom_margin = int(rng.integers(0, 4))

    height = max(main_top + main_h, legend_top + legend_h) + bottom_margin
    width = left_margin + legend_w + between_gap + main_w + right_margin
    grid = np.full((height, width), bg, dtype=int)

    grid[legend_top : legend_top + legend_h, left_margin : left_margin + legend_w] = legend
    main_left = left_margin + legend_w + between_gap
    grid[main_top : main_top + main_h, main_left : main_left + main_w] = input_main

    if _get_bg(grid) != bg:
        return None
    try:
        out = main(grid)
    except Exception:
        return None
    if not np.array_equal(out, expected_main):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
