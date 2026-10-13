from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
SEED_FRAME_COLOR = 5
MAX_GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(len(cells)),
                }
            )
    return comps


def _is_rectangular_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    cells = {tuple(map(int, xy)) for xy in comp["coords"]}
    if len(cells) != 2 * h + 2 * w - 4:
        return False

    for c in range(c0, c1 + 1):
        if (r0, c) not in cells or (r1, c) not in cells:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in cells or (r, c1) not in cells:
            return False
    return True


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen = {start}
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _extract_templates(grid: np.ndarray, seed_frame: dict, bg: int) -> dict[int, np.ndarray]:
    r0, c0, r1, c1 = seed_frame["bbox"]
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    templates: dict[int, np.ndarray] = {}
    for color0 in np.unique(interior):
        color = int(color0)
        if color in (bg, int(seed_frame["color"])):
            continue
        mask = interior == color
        assert _connected(mask), "each template color must form a single connected seed shape"
        coords = np.argwhere(mask)
        sr0, sc0, sr1, sc1 = _bbox(coords)
        templates[color] = mask[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
    assert templates, "expected at least one template inside the seed frame"
    return templates


def _scale_template(template: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    src_h, src_w = template.shape
    assert out_h % src_h == 0 and out_w % src_w == 0, "frame interior must be an integer multiple of template bbox"
    sy = out_h // src_h
    sx = out_w // src_w
    block = np.ones((sy, sx), dtype=int)
    return np.kron(template.astype(int), block).astype(bool)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read colored seed shapes from the unique non-empty frame, then scale each shape
    into every empty frame that shares its color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _background_color(g)

    comps = _monochrome_components(g, bg)
    frames = [comp for comp in comps if _is_rectangular_frame(comp)]
    assert frames, "expected at least one hollow rectangular frame"

    seed_index = None
    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        if np.any((interior != bg) & (interior != int(frame["color"]))):
            assert seed_index is None, "expected exactly one seed frame"
            seed_index = idx
    assert seed_index is not None, "could not locate the seed frame"

    seed_frame = frames[seed_index]
    templates = _extract_templates(g, seed_frame, bg)

    out = g.copy()
    for idx, frame in enumerate(frames):
        if idx == seed_index:
            continue
        color = int(frame["color"])
        assert color in templates, "every empty frame color must match a seed template color"
        r0, c0, r1, c1 = frame["bbox"]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        assert np.all(inner == bg), "target frame interiors must be empty in the input"
        scaled = _scale_template(templates[color], inner.shape[0], inner.shape[1])
        inner[scaled] = color
    return out


def _crop_shape(shape: np.ndarray) -> np.ndarray:
    coords = np.argwhere(shape)
    r0, c0, r1, c1 = _bbox(coords)
    return shape[r0 : r1 + 1, c0 : c1 + 1].copy()


def _shape_signature(shape: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in shape.astype(int))


def _all_transforms(shape: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    current = shape.copy()
    for _ in range(4):
        for variant in (current, np.fliplr(current)):
            cropped = _crop_shape(variant)
            sig = _shape_signature(cropped)
            if sig not in seen:
                seen.add(sig)
                variants.append(cropped)
        current = np.rot90(current)
    return variants


def _shape_library() -> list[np.ndarray]:
    bases = [
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1], [1, 1]], dtype=bool),
        np.array([[1, 1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
        np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
    ]

    library: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for base in bases:
        for variant in _all_transforms(base):
            area = int(variant.sum())
            if 3 <= area <= 6:
                sig = _shape_signature(variant)
                if sig not in seen:
                    seen.add(sig)
                    library.append(variant)
    return library


SHAPE_LIBRARY = _shape_library()


def _place_templates(
    interior_h: int, interior_w: int, shapes: list[np.ndarray], rng: np.random.Generator
) -> list[tuple[int, int]] | None:
    blocked = np.zeros((interior_h, interior_w), dtype=bool)
    positions: list[tuple[int, int]] = [(-1, -1)] * len(shapes)
    order = sorted(range(len(shapes)), key=lambda i: shapes[i].size, reverse=True)

    for idx in order:
        shape = shapes[idx]
        sh, sw = shape.shape
        candidates = []
        for r0 in range(1, interior_h - sh):
            for c0 in range(1, interior_w - sw):
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(interior_h, r0 + sh + 1)
                cc1 = min(interior_w, c0 + sw + 1)
                if not blocked[rr0:rr1, cc0:cc1].any():
                    candidates.append((r0, c0))
        if not candidates:
            return None
        choice = candidates[int(rng.integers(len(candidates)))]
        r0, c0 = choice
        positions[idx] = choice
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(interior_h, r0 + sh + 1)
        cc1 = min(interior_w, c0 + sw + 1)
        blocked[rr0:rr1, cc0:cc1] = True
    return positions


def _render_seed(
    grid: np.ndarray,
    top: int,
    left: int,
    interior_h: int,
    interior_w: int,
    color_shapes: list[tuple[int, np.ndarray]],
    positions: list[tuple[int, int]],
) -> None:
    bottom = top + interior_h + 1
    right = left + interior_w + 1
    grid[top, left : right + 1] = SEED_FRAME_COLOR
    grid[bottom, left : right + 1] = SEED_FRAME_COLOR
    grid[top : bottom + 1, left] = SEED_FRAME_COLOR
    grid[top : bottom + 1, right] = SEED_FRAME_COLOR

    for (color, shape), (r0, c0) in zip(color_shapes, positions, strict=False):
        sh, sw = shape.shape
        region = grid[top + 1 + r0 : top + 1 + r0 + sh, left + 1 + c0 : left + 1 + c0 + sw]
        region[shape] = color


def _row_partition(items: list[dict], n_rows: int, rng: np.random.Generator) -> list[list[dict]]:
    rows: list[list[dict]] = [[] for _ in range(n_rows)]
    order = list(items)
    rng.shuffle(order)
    for item in order:
        widths = []
        for row in rows:
            width = sum(entry["outer_w"] for entry in row) + max(0, len(row) - 1) * 2
            widths.append(width)
        lightest = min(widths)
        candidates = [i for i, width in enumerate(widths) if width == lightest]
        rows[int(rng.choice(candidates))].append(item)
    for row in rows:
        rng.shuffle(row)
    return rows


def _layout_frames(
    grid: np.ndarray,
    seed_bbox: tuple[int, int, int, int],
    frames: list[dict],
    rng: np.random.Generator,
) -> None:
    if not frames:
        return

    h, w = grid.shape
    _seed_r0, _seed_c0, seed_r1, seed_c1 = seed_bbox
    usable_left = seed_c1 + int(rng.integers(3, 6))
    usable_width = w - usable_left - 1
    assert usable_width >= max(frame["outer_w"] for frame in frames)

    row_options = list(range(min(3, len(frames)), 0, -1))
    placed = False
    for n_rows in row_options:
        for _ in range(40):
            rows = _row_partition(frames, n_rows, rng)
            row_heights = [max(frame["outer_h"] for frame in row) if row else 0 for row in rows]
            row_gaps = [[int(rng.integers(2, 5)) for _ in range(max(0, len(row) - 1))] for row in rows]
            row_widths = [
                sum(frame["outer_w"] for frame in row) + sum(gaps)
                for row, gaps in zip(rows, row_gaps, strict=False)
            ]
            if max(row_widths, default=0) > usable_width:
                continue

            v_gaps = [int(rng.integers(2, 5)) for _ in range(max(0, n_rows - 1))]
            total_h = sum(row_heights) + sum(v_gaps)
            if total_h > h - 2:
                continue

            top_start = int(rng.integers(1, h - total_h + 1))
            row_top = top_start
            for row_index, row in enumerate(rows):
                row_height = row_heights[row_index]
                gaps = row_gaps[row_index]
                row_width = row_widths[row_index]
                row_left = usable_left + int(rng.integers(0, usable_width - row_width + 1))
                cursor = row_left
                for idx, frame in enumerate(row):
                    y_jitter = int(rng.integers(0, row_height - frame["outer_h"] + 1))
                    frame["top"] = row_top + y_jitter
                    frame["left"] = cursor
                    cursor += frame["outer_w"]
                    if idx < len(gaps):
                        cursor += gaps[idx]
                row_top += row_height
                if row_index < len(v_gaps):
                    row_top += v_gaps[row_index]
            placed = True
            break
        if placed:
            break

    assert placed, "could not fit frames into the canvas"
    assert seed_r1 < h and seed_c1 < w


def _render_frame(grid: np.ndarray, frame: dict) -> None:
    r0 = frame["top"]
    c0 = frame["left"]
    outer_h = frame["outer_h"]
    outer_w = frame["outer_w"]
    color = frame["color"]
    grid[r0, c0 : c0 + outer_w] = color
    grid[r0 + outer_h - 1, c0 : c0 + outer_w] = color
    grid[r0 : r0 + outer_h, c0] = color
    grid[r0 : r0 + outer_h, c0 + outer_w - 1] = color


def _render_expected_fill(grid: np.ndarray, frame: dict) -> None:
    r0 = frame["top"]
    c0 = frame["left"]
    inner_h = frame["outer_h"] - 2
    inner_w = frame["outer_w"] - 2
    scaled = _scale_template(frame["shape"], inner_h, inner_w)
    region = grid[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w]
    region[scaled] = frame["color"]


def _sample_seed_shapes(
    rng: np.random.Generator,
) -> tuple[list[tuple[int, np.ndarray]], int, int, list[tuple[int, int]]] | None:
    n_templates = int(rng.integers(2, 4))
    colors = [int(c) for c in rng.choice(np.array([1, 2, 3, 4, 6, 7, 8, 9]), size=n_templates, replace=False)]
    shape_indices = rng.choice(len(SHAPE_LIBRARY), size=n_templates, replace=False)
    shapes = [SHAPE_LIBRARY[int(idx)] for idx in shape_indices]

    max_h = max(shape.shape[0] for shape in shapes)
    max_w = max(shape.shape[1] for shape in shapes)
    for _ in range(60):
        interior_h = int(rng.integers(max(6, max_h + 2), max(10, max_h + 6)))
        interior_w = int(rng.integers(max(7, max_w + 3), max(11, max_w + 8)))
        positions = _place_templates(interior_h, interior_w, shapes, rng)
        if positions is not None:
            pairs = list(zip(colors, shapes, strict=False))
            return pairs, interior_h, interior_w, positions
    return None


def _sample_frames(
    color_shapes: list[tuple[int, np.ndarray]], rng: np.random.Generator
) -> list[dict]:
    frames: list[dict] = []
    for color, shape in color_shapes:
        copies = int(rng.integers(1, 3))
        for _ in range(copies):
            sy = int(rng.integers(1, 4))
            sx = int(rng.integers(1, 4))
            frames.append(
                {
                    "color": color,
                    "shape": shape,
                    "outer_h": int(shape.shape[0] * sy + 2),
                    "outer_w": int(shape.shape[1] * sx + 2),
                }
            )

    if len(frames) == 6:
        removable = [idx for idx, frame in enumerate(frames) if sum(f["color"] == frame["color"] for f in frames) > 1]
        if removable and rng.random() < 0.8:
            del frames[int(rng.choice(removable))]

    if all(frame["outer_h"] == frame["shape"].shape[0] + 2 and frame["outer_w"] == frame["shape"].shape[1] + 2 for frame in frames):
        chosen = frames[int(rng.integers(len(frames)))]
        if rng.random() < 0.5:
            chosen["outer_h"] += chosen["shape"].shape[0]
        else:
            chosen["outer_w"] += chosen["shape"].shape[1]

    if not any(
        frame["outer_h"] - 2 != frame["shape"].shape[0] and frame["outer_w"] - 2 != frame["shape"].shape[1]
        for frame in frames
    ) and len(frames) > 1:
        chosen = frames[int(rng.integers(len(frames)))]
        chosen["outer_h"] += chosen["shape"].shape[0]
        chosen["outer_w"] += chosen["shape"].shape[1]
    return frames


def _grid_shape(
    seed_outer_h: int, seed_outer_w: int, frames: list[dict], rng: np.random.Generator
) -> tuple[int, int]:
    planned_rows = min(3, len(frames))
    max_frame_h = max(frame["outer_h"] for frame in frames)
    max_frame_w = max(frame["outer_w"] for frame in frames)
    total_frame_w = sum(frame["outer_w"] for frame in frames) + 4 * max(0, len(frames) - 1)
    total_frame_h = sum(frame["outer_h"] for frame in frames) + 4 * max(0, len(frames) - 1)
    h = max(seed_outer_h + 4, max_frame_h + 6, (total_frame_h + planned_rows - 1) // planned_rows + 8)
    w = max(seed_outer_w + max_frame_w + 8, seed_outer_w + (total_frame_w + planned_rows - 1) // planned_rows + 8)
    h += int(rng.integers(0, 4))
    w += int(rng.integers(0, 5))
    return h, w


def _attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    seed_sample = _sample_seed_shapes(rng)
    if seed_sample is None:
        return None
    color_shapes, interior_h, interior_w, positions = seed_sample
    frames = _sample_frames(color_shapes, rng)

    seed_outer_h = interior_h + 2
    seed_outer_w = interior_w + 2
    grid_h, grid_w = _grid_shape(seed_outer_h, seed_outer_w, frames, rng)

    non_bg_upper_bound = (2 * seed_outer_h + 2 * seed_outer_w - 4) + sum(int(shape.sum()) for _, shape in color_shapes)
    non_bg_upper_bound += sum(2 * frame["outer_h"] + 2 * frame["outer_w"] - 4 for frame in frames)
    while grid_h * grid_w <= non_bg_upper_bound * 2:
        grid_h += 2
        grid_w += 2

    grid_h = max(grid_h, seed_outer_h + 3)
    grid_w = max(grid_w, seed_outer_w + max(frame["outer_w"] for frame in frames) + 6)

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    out = inp.copy()

    seed_top = int(rng.integers(1, grid_h - seed_outer_h + 1))
    seed_left = int(rng.integers(1, 4))
    seed_bbox = (seed_top, seed_left, seed_top + seed_outer_h - 1, seed_left + seed_outer_w - 1)

    _render_seed(inp, seed_top, seed_left, interior_h, interior_w, color_shapes, positions)
    out[:] = inp

    try:
        _layout_frames(inp, seed_bbox, frames, rng)
    except AssertionError:
        return None

    for frame in frames:
        _render_frame(inp, frame)
        _render_frame(out, frame)
        _render_expected_fill(out, frame)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
