import random
import numpy as np


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                if rr > 0 and mask[rr - 1, cc] and not seen[rr - 1, cc]:
                    seen[rr - 1, cc] = True
                    stack.append((rr - 1, cc))
                if rr + 1 < h and mask[rr + 1, cc] and not seen[rr + 1, cc]:
                    seen[rr + 1, cc] = True
                    stack.append((rr + 1, cc))
                if cc > 0 and mask[rr, cc - 1] and not seen[rr, cc - 1]:
                    seen[rr, cc - 1] = True
                    stack.append((rr, cc - 1))
                if cc + 1 < w and mask[rr, cc + 1] and not seen[rr, cc + 1]:
                    seen[rr, cc + 1] = True
                    stack.append((rr, cc + 1))
            comps.append(comp)
    return comps


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = bbox
    mask = np.zeros((h, w), dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _interior_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + 1, c0 + 1, r1 - 1, c1 - 1


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _extract_palette(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int]]]:
    h, w = grid.shape
    assert h >= 3
    palette_cols = [c for c in range(w) if int(grid[0, c]) != int(grid[1, c])]
    assert palette_cols, "expected a two-row palette strip"

    bg_samples = [int(grid[0, c]) for c in range(w) if int(grid[0, c]) == int(grid[1, c])]
    assert bg_samples, "expected visible background columns beside the palette"
    bg = max(set(bg_samples), key=bg_samples.count)

    entries: list[tuple[int, int, int]] = []
    seen: dict[int, int] = {}
    for c in palette_cols:
        key = int(grid[0, c])
        value = int(grid[1, c])
        if key in seen:
            assert seen[key] == value
            continue
        seen[key] = value
        entries.append((key, value, c))
    return bg, entries


def _find_rectangular_windows(canvas: np.ndarray, color: int) -> list[tuple[int, int, int, int]]:
    mask = canvas == int(color)
    windows: list[tuple[int, int, int, int]] = []
    for comp in _connected_components(mask):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        patch = mask[r0 : r1 + 1, c0 : c1 + 1]
        frame = np.zeros_like(patch, dtype=bool)
        frame[0, :] = True
        frame[-1, :] = True
        frame[:, 0] = True
        frame[:, -1] = True
        if np.array_equal(patch, frame):
            windows.append((r0, c0, r1, c1))
    windows.sort()
    return windows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only stripe pixels visible through rectangular windows, recolored by the palette value matched to each window border color.
    """
    g = np.asarray(grid, dtype=int)
    bg, palette = _extract_palette(g)
    canvas = g[2:, :]

    out = np.full(canvas.shape, bg, dtype=int)
    for key, value, _col in sorted(palette, key=lambda item: item[2]):
        for r0, c0, r1, c1 in _find_rectangular_windows(canvas, key):
            out[r0 + 1 : r1, c0 + 1 : c1] = int(value)
    return out


COLORS = tuple(range(10))
WIDTH_RANGE = (10, 18)
PALETTE_LEN_RANGE = (2, 4)
MAX_WINDOWS = 3
OVERLAP_PROB = 0.5
GEN_ATTEMPTS = 400


def _candidate_window(
    canvas_h: int,
    width: int,
    *,
    min_h: int = 3,
    min_w: int = 3,
    max_h: int | None = None,
    max_w: int | None = None,
) -> tuple[int, int, int, int]:
    max_h = canvas_h if max_h is None else max_h
    max_w = width if max_w is None else max_w
    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)
    r0 = random.randint(0, canvas_h - h)
    c0 = random.randint(0, width - w)
    return r0, c0, r0 + h - 1, c0 + w - 1


def _valid_window(
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    existing: list[tuple[int, int, int, int]],
    *,
    allow_overlap: bool,
) -> bool:
    candidate_border = _frame_mask(shape, bbox)
    for other in existing:
        if np.any(candidate_border & _frame_mask(shape, other)):
            return False
        if not allow_overlap and _boxes_overlap(_interior_bbox(bbox), _interior_bbox(other)):
            return False
    return True


def _has_overlap(boxes: list[tuple[int, int, int, int]]) -> bool:
    interiors = [_interior_bbox(box) for box in boxes]
    for i, first in enumerate(interiors):
        for second in interiors[i + 1 :]:
            if _boxes_overlap(first, second):
                return True
    return False


def _place_windows(canvas_h: int, width: int, window_count: int, want_overlap: bool) -> list[tuple[int, int, int, int]] | None:
    windows: list[tuple[int, int, int, int]] = []
    shape = (canvas_h, width)

    if want_overlap:
        outer = None
        for _ in range(80):
            candidate = _candidate_window(canvas_h, width, min_h=5, min_w=5)
            if _valid_window(candidate, shape, windows, allow_overlap=True):
                outer = candidate
                windows.append(candidate)
                break
        if outer is None:
            return None

        r0, c0, r1, c1 = outer
        inner_h_max = r1 - r0 - 1
        inner_w_max = c1 - c0 - 1
        if inner_h_max < 3 or inner_w_max < 3:
            return None
        for _ in range(80):
            h = random.randint(3, inner_h_max)
            w = random.randint(3, inner_w_max)
            inner_r0 = random.randint(r0 + 1, r1 - h)
            inner_c0 = random.randint(c0 + 1, c1 - w)
            inner = (inner_r0, inner_c0, inner_r0 + h - 1, inner_c0 + w - 1)
            if _valid_window(inner, shape, windows, allow_overlap=True):
                windows.append(inner)
                break
        if len(windows) < 2:
            return None

    while len(windows) < window_count:
        placed = False
        for _ in range(120):
            candidate = _candidate_window(
                canvas_h,
                width,
                max_h=canvas_h if want_overlap else max(3, canvas_h - (window_count > 1)),
                max_w=width if want_overlap else max(5, width - (window_count > 1) * 2),
            )
            if _valid_window(candidate, shape, windows, allow_overlap=want_overlap):
                windows.append(candidate)
                placed = True
                break
        if not placed:
            return None

    if want_overlap != _has_overlap(windows):
        return None
    return windows


def _render_input(
    bg: int,
    width: int,
    stripe_colors: list[int],
    palette_start: int,
    key_colors: list[int],
    value_colors: list[int],
    windows: list[dict[str, object]],
) -> np.ndarray:
    canvas_h = len(stripe_colors)
    grid = np.full((canvas_h + 2, width), bg, dtype=int)
    for idx, (key, value) in enumerate(zip(key_colors, value_colors)):
        col = palette_start + idx
        grid[0, col] = int(key)
        grid[1, col] = int(value)
    for r, color in enumerate(stripe_colors, start=2):
        grid[r, :] = int(color)

    canvas = grid[2:, :]
    for window in windows:
        r0, c0, r1, c1 = window["bbox"]
        key = int(window["key"])
        canvas[r0, c0 : c1 + 1] = key
        canvas[r1, c0 : c1 + 1] = key
        canvas[r0 : r1 + 1, c0] = key
        canvas[r0 : r1 + 1, c1] = key
    return grid


def _render_output(
    bg: int,
    canvas_h: int,
    width: int,
    value_colors: list[int],
    windows: list[dict[str, object]],
) -> np.ndarray:
    out = np.full((canvas_h, width), bg, dtype=int)
    for palette_idx in range(len(value_colors)):
        value = int(value_colors[palette_idx])
        for window in windows:
            if int(window["palette_idx"]) != palette_idx:
                continue
            r0, c0, r1, c1 = window["bbox"]
            out[r0 + 1 : r1, c0 + 1 : c1] = value
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(COLORS)
    width = random.randint(*WIDTH_RANGE)

    max_palette_len = min(PALETTE_LEN_RANGE[1], width - 2)
    if max_palette_len < PALETTE_LEN_RANGE[0]:
        return None

    window_cap = min(MAX_WINDOWS, max_palette_len)
    window_count = random.choices(
        list(range(1, window_cap + 1)),
        weights=[1, 2, 2][:window_cap],
        k=1,
    )[0]
    palette_len = random.randint(max(window_count, PALETTE_LEN_RANGE[0]), max_palette_len)
    want_overlap = window_count >= 2 and random.random() < OVERLAP_PROB

    usable = [c for c in COLORS if c != bg]
    key_colors = random.sample(usable, palette_len)
    stripe_pool = [c for c in usable if c not in key_colors]

    min_canvas_h = 5 if want_overlap else 4
    max_canvas_h = min(6, len(stripe_pool))
    if max_canvas_h < min_canvas_h:
        return None
    canvas_h = random.randint(min_canvas_h, max_canvas_h)

    stripe_colors = random.sample(stripe_pool, canvas_h)
    value_colors = [random.choice(stripe_pool) for _ in range(palette_len)]
    palette_start = random.randint(1, width - palette_len - 1)

    boxes = _place_windows(canvas_h, width, window_count, want_overlap)
    if boxes is None:
        return None

    used_palette_indices = random.sample(range(palette_len), window_count)
    random.shuffle(used_palette_indices)
    windows: list[dict[str, object]] = []
    for bbox, palette_idx in zip(boxes, used_palette_indices):
        windows.append(
            {
                "bbox": bbox,
                "palette_idx": palette_idx,
                "key": key_colors[palette_idx],
            }
        )

    grid = _render_input(
        bg=bg,
        width=width,
        stripe_colors=stripe_colors,
        palette_start=palette_start,
        key_colors=key_colors,
        value_colors=value_colors,
        windows=windows,
    )
    expected = _render_output(
        bg=bg,
        canvas_h=canvas_h,
        width=width,
        value_colors=value_colors,
        windows=windows,
    )
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
