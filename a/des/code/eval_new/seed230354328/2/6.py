from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GENERATION_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _find_canvas_and_stamps(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    comps = _components(grid != BG)
    assert comps, "expected a canvas and at least one stamp"

    infos: list[dict[str, object]] = []
    for coords in comps:
        bbox = _bbox(coords)
        sub = _crop(grid, bbox)
        fg_colors = sorted({int(v) for v in sub[sub != BG].flat})
        infos.append(
            {
                "bbox": bbox,
                "sub": sub,
                "area": int(coords.shape[0]),
                "dense": int(coords.shape[0]) == int(sub.shape[0] * sub.shape[1]),
                "fg_colors": fg_colors,
            }
        )

    canvas_candidates = [info for info in infos if info["dense"] and len(info["fg_colors"]) > 1]
    assert canvas_candidates, "failed to identify the large canvas rectangle"
    canvas = max(canvas_candidates, key=lambda info: int(info["area"]))

    stamps = [info for info in infos if info is not canvas]
    assert 2 <= len(stamps) <= 4, "expected 2-4 stamp components"
    for stamp in stamps:
        assert len(stamp["fg_colors"]) == 1, "each stamp must be monochrome"
    stamps.sort(key=lambda info: (int(info["bbox"][0]), int(info["bbox"][1])))
    return np.asarray(canvas["sub"], dtype=int), stamps


def _match_locations(canvas: np.ndarray, stamp_mask: np.ndarray, color: int) -> list[tuple[int, int]]:
    sh, sw = stamp_mask.shape
    ch, cw = canvas.shape
    hits: list[tuple[int, int]] = []
    for r0 in range(ch - sh + 1):
        for c0 in range(cw - sw + 1):
            sub = canvas[r0 : r0 + sh, c0 : c0 + sw]
            if np.array_equal(sub == color, stamp_mask):
                hits.append((r0, c0))
    return hits


def _assemble_windows(windows: list[np.ndarray]) -> np.ndarray:
    assert windows, "need at least one extracted window"
    heights = {int(win.shape[0]) for win in windows}
    assert len(heights) == 1, "all windows must share a common height"
    if len(windows) == 1:
        return windows[0].copy()

    pieces: list[np.ndarray] = []
    for idx, win in enumerate(windows):
        if idx:
            pieces.append(np.full((win.shape[0], 1), GREY, dtype=int))
        pieces.append(win)
    return np.hstack(pieces)


def main(grid: np.ndarray) -> np.ndarray:
    """Locate each monochrome stamp inside the large canvas and extract the matched windows.

    Input structure:
    - The background is black.
    - One large dense multicolor rectangle is the canvas; it is separated from the other objects by black space.
    - The remaining objects are 2-4 scattered monochrome stamp shapes on the right or below the canvas.
    - Each stamp has one unique color, a cropped 3x3 to 5x5 bounding box, and the stamps in one instance share a common height.

    Rule:
    - Identify the large rectangle as the canvas and treat every other non-black connected component as a stamp.
    - For each stamp, search the canvas for the unique placement whose same-color pixels match the stamp shape exactly inside the stamp's bounding box.
    - Crop that matched bounding box from the canvas as a window.
    - Order the stamps by their positions in the input from top to bottom, then left to right, and concatenate the windows horizontally with a single grey separator column.

    Key insight:
    - Each stamp is both a template for exact shape-color matching and a crop window selector for the hidden region inside the larger picture.
    """
    g = np.asarray(grid, dtype=int)
    canvas, stamps = _find_canvas_and_stamps(g)

    windows: list[np.ndarray] = []
    for stamp in stamps:
        sub = np.asarray(stamp["sub"], dtype=int)
        color = int(stamp["fg_colors"][0])
        mask = sub == color
        hits = _match_locations(canvas, mask, color)
        assert len(hits) == 1, "each stamp must match the canvas exactly once"
        r0, c0 = hits[0]
        sh, sw = sub.shape
        windows.append(canvas[r0 : r0 + sh, c0 : c0 + sw].copy())

    return _assemble_windows(windows)


def _base_shapes(size: int) -> list[np.ndarray]:
    mid = size // 2
    shapes: list[np.ndarray] = []

    mask = np.zeros((size, size), dtype=bool)
    mask[:, 0] = True
    mask[-1, :] = True
    shapes.append(mask)

    mask = np.zeros((size, size), dtype=bool)
    mask[0, :] = True
    mask[:, mid] = True
    shapes.append(mask)

    mask = np.zeros((size, size), dtype=bool)
    mask[mid, :] = True
    mask[:, mid] = True
    shapes.append(mask)

    mask = np.zeros((size, size), dtype=bool)
    mask[:, 0] = True
    mask[:, -1] = True
    mask[-1, :] = True
    shapes.append(mask)

    mask = np.zeros((size, size), dtype=bool)
    mask[:, 0] = True
    mask[0, :] = True
    mask[mid, :] = True
    mask[-1, :] = True
    shapes.append(mask)

    mask = np.zeros((size, size), dtype=bool)
    mask[:, 0] = True
    mask[:, -1] = True
    mask[mid, :] = True
    shapes.append(mask)

    return shapes


def _shape_variants(size: int) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    for base in _base_shapes(size):
        for k in range(4):
            rot = np.rot90(base, k)
            for arr in (rot, np.fliplr(rot)):
                key = (arr.shape, tuple(int(v) for v in arr.ravel()))
                if key not in seen:
                    seen.add(key)
                    variants.append(arr.copy())
    return variants


def _rand_array(colors: list[int], shape: tuple[int, int]) -> np.ndarray:
    return np.asarray([[random.choice(colors) for _ in range(shape[1])] for _ in range(shape[0])], dtype=int)


def _band_sizes(total: int, parts: int, minimum: int) -> list[int]:
    sizes = [minimum] * parts
    extra = total - parts * minimum
    for _ in range(extra):
        sizes[random.randrange(parts)] += 1
    return sizes


def _window_layout(count: int) -> tuple[int, int]:
    options = {
        2: [(1, 2), (2, 1), (2, 2)],
        3: [(1, 3), (3, 1), (2, 2), (2, 3), (3, 2)],
        4: [(2, 2), (2, 3), (3, 2)],
    }
    return random.choice(options[count])


def _window_positions(count: int, size: int, canvas_h: int, canvas_w: int) -> list[tuple[int, int]] | None:
    slot_rows, slot_cols = _window_layout(count)
    if slot_rows * size > canvas_h or slot_cols * size > canvas_w:
        return None

    row_sizes = _band_sizes(canvas_h, slot_rows, size)
    col_sizes = _band_sizes(canvas_w, slot_cols, size)
    row_starts = [0]
    col_starts = [0]
    for value in row_sizes[:-1]:
        row_starts.append(row_starts[-1] + value)
    for value in col_sizes[:-1]:
        col_starts.append(col_starts[-1] + value)

    slots = [(rr, cc) for rr in range(slot_rows) for cc in range(slot_cols)]
    chosen = random.sample(slots, count)
    positions: list[tuple[int, int]] = []
    for rr, cc in chosen:
        r0 = row_starts[rr] + random.randint(0, row_sizes[rr] - size)
        c0 = col_starts[cc] + random.randint(0, col_sizes[cc] - size)
        positions.append((r0, c0))
    return positions


def _embed_windows(
    canvas_h: int,
    canvas_w: int,
    size: int,
    stamps: list[dict[str, object]],
    filler_colors: list[int],
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    positions = _window_positions(len(stamps), size, canvas_h, canvas_w)
    if positions is None:
        return None

    canvas = _rand_array(filler_colors, (canvas_h, canvas_w))
    windows: list[np.ndarray] = []
    hidden = list(zip(stamps, positions))
    for stamp, (r0, c0) in hidden:
        color = int(stamp["color"])
        mask = np.asarray(stamp["mask"], dtype=bool)
        window = _rand_array(filler_colors, (size, size))
        if len(filler_colors) >= 2 and np.all(window == window[0, 0]):
            window[0, -1] = filler_colors[1]
        window[mask] = color
        canvas[r0 : r0 + size, c0 : c0 + size] = window
        stamp["hidden_pos"] = (r0, c0)
        windows.append(window)
    return canvas, windows


def _paint_stamp(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = color
    grid[top : top + mask.shape[0], left : left + mask.shape[1]] = patch


def _build_left_layout(canvas: np.ndarray, stamps: list[dict[str, object]], size: int) -> np.ndarray:
    ch, cw = canvas.shape
    left_margin = random.randint(0, 2)
    right_margin = random.randint(0, 2)
    canvas_top = random.randint(0, 2)
    stamp_top = random.randint(0, 2)
    middle_gap = random.randint(2, 4)
    bottom_margin = random.randint(0, 2)
    gaps = [random.randint(1, 2) for _ in range(len(stamps) - 1)]
    stamp_height = len(stamps) * size + sum(gaps)

    h = max(canvas_top + ch, stamp_top + stamp_height) + bottom_margin
    w = left_margin + cw + middle_gap + size + right_margin
    grid = np.zeros((h, w), dtype=int)
    grid[canvas_top : canvas_top + ch, left_margin : left_margin + cw] = canvas

    cur_r = stamp_top
    stamp_c = left_margin + cw + middle_gap
    for idx, stamp in enumerate(stamps):
        _paint_stamp(grid, cur_r, stamp_c, np.asarray(stamp["mask"], dtype=bool), int(stamp["color"]))
        if idx < len(gaps):
            cur_r += size + gaps[idx]
    return grid


def _build_top_layout(canvas: np.ndarray, stamps: list[dict[str, object]], size: int) -> np.ndarray:
    ch, cw = canvas.shape
    top_margin = random.randint(0, 2)
    bottom_margin = random.randint(0, 2)
    canvas_left = random.randint(0, 2)
    stamp_left = random.randint(0, 2)
    middle_gap = random.randint(2, 4)
    right_margin = random.randint(0, 2)
    gaps = [random.randint(1, 2) for _ in range(len(stamps) - 1)]
    stamp_width = len(stamps) * size + sum(gaps)

    h = top_margin + ch + middle_gap + size + bottom_margin
    w = max(canvas_left + cw, stamp_left + stamp_width) + right_margin
    grid = np.zeros((h, w), dtype=int)
    grid[top_margin : top_margin + ch, canvas_left : canvas_left + cw] = canvas

    cur_c = stamp_left
    stamp_r = top_margin + ch + middle_gap
    for idx, stamp in enumerate(stamps):
        _paint_stamp(grid, stamp_r, cur_c, np.asarray(stamp["mask"], dtype=bool), int(stamp["color"]))
        if idx < len(gaps):
            cur_c += size + gaps[idx]
    return grid


def _sample_stamps(size: int, count: int) -> list[dict[str, object]] | None:
    variants = _shape_variants(size)
    if len(variants) < count:
        return None
    filler_count = random.randint(2, min(4, len(PALETTE) - count))
    chosen_colors = random.sample(PALETTE, count + filler_count)
    stamp_colors = chosen_colors[:count]
    chosen_masks = random.sample(variants, count)
    stamps = []
    for mask, color in zip(chosen_masks, stamp_colors):
        stamps.append({"mask": mask.copy(), "color": int(color)})
    random.shuffle(stamps)
    return stamps


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(3, 5)
    count = random.randint(2, 4)
    stamps = _sample_stamps(size, count)
    if stamps is None:
        return None

    filler_count = random.randint(2, min(4, len(PALETTE) - count))
    colors = random.sample(PALETTE, count + filler_count)
    stamp_colors = colors[:count]
    filler_colors = colors[count:]
    for stamp, color in zip(stamps, stamp_colors):
        stamp["color"] = int(color)

    layout = _window_layout(count)
    min_h = max(8, layout[0] * size)
    min_w = max(8, layout[1] * size)
    canvas_h = random.randint(min_h, 15)
    canvas_w = random.randint(min_w, 15)
    embedded = _embed_windows(canvas_h, canvas_w, size, stamps, filler_colors)
    if embedded is None:
        return None
    canvas, _ = embedded

    stamp_order = stamps[:]
    random.shuffle(stamp_order)
    expected_windows = []
    for stamp in stamp_order:
        r0, c0 = stamp["hidden_pos"]
        expected_windows.append(canvas[r0 : r0 + size, c0 : c0 + size].copy())
    expected = _assemble_windows(expected_windows)

    if random.choice([True, False]):
        grid = _build_left_layout(canvas, stamp_order, size)
    else:
        grid = _build_top_layout(canvas, stamp_order, size)

    try:
        if max(grid.shape) > 30 or max(expected.shape) > 30:
            return None
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
