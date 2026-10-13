import random
from collections import deque

import numpy as np


BG = 0
FRAME = 7
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _non_separator_spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for stop in separators + [length]:
        if stop > start:
            spans.append((start, stop))
        start = stop + 1
    return spans


def _spiral_positions(rows: int, cols: int) -> list[tuple[int, int]]:
    top, bottom = 0, rows - 1
    left, right = 0, cols - 1
    order: list[tuple[int, int]] = []
    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            order.append((top, c))
        top += 1
        for r in range(top, bottom + 1):
            order.append((r, right))
        right -= 1
        if top <= bottom:
            for c in range(right, left - 1, -1):
                order.append((bottom, c))
            bottom -= 1
        if left <= right:
            for r in range(bottom, top - 1, -1):
                order.append((r, left))
            left += 1
    return order


def _parse_window(sub: np.ndarray) -> dict[str, object] | None:
    h, w = map(int, sub.shape)
    if h < 3 or w < 3:
        return None

    border_coords: list[tuple[int, int]] = []
    for c in range(w):
        border_coords.append((0, c))
        border_coords.append((h - 1, c))
    for r in range(1, h - 1):
        border_coords.append((r, 0))
        border_coords.append((r, w - 1))

    label_cells = []
    for r, c in border_coords:
        val = int(sub[r, c])
        if val == FRAME:
            continue
        if val == BG:
            return None
        label_cells.append((r, c, val))
    if len(label_cells) != 1:
        return None

    inner = sub[1:-1, 1:-1]
    if inner.size == 0:
        return None
    if not set(map(int, np.unique(inner))).issubset({BG, FRAME}):
        return None

    sep_rows = [r for r in range(inner.shape[0]) if np.all(inner[r, :] == FRAME)]
    sep_cols = [c for c in range(inner.shape[1]) if np.all(inner[:, c] == FRAME)]
    sep_row_set = set(sep_rows)
    sep_col_set = set(sep_cols)

    for r, c in np.argwhere(inner == FRAME):
        rr = int(r)
        cc = int(c)
        if rr not in sep_row_set and cc not in sep_col_set:
            return None

    row_spans = _non_separator_spans(int(inner.shape[0]), sep_rows)
    col_spans = _non_separator_spans(int(inner.shape[1]), sep_cols)
    if not row_spans or not col_spans:
        return None

    cell_heights = {end - start for start, end in row_spans}
    cell_widths = {end - start for start, end in col_spans}
    if len(cell_heights) != 1 or len(cell_widths) != 1:
        return None

    cell_h = cell_heights.pop()
    cell_w = cell_widths.pop()
    if cell_h <= 0 or cell_w <= 0:
        return None

    return {
        "label": int(label_cells[0][2]),
        "R": len(row_spans),
        "C": len(col_spans),
        "cell_h": int(cell_h),
        "cell_w": int(cell_w),
        "row_spans": row_spans,
        "col_spans": col_spans,
    }


def _extract_payload(sub: np.ndarray, payload_h: int, payload_w: int) -> np.ndarray | None:
    if int(sub[0, 0]) in {BG, FRAME}:
        return None

    candidates: list[np.ndarray] = []

    if sub.shape == (payload_h, payload_w + 1):
        if np.all(sub[:, 1:] != BG) and np.all(sub[1:, 0] == BG):
            candidates.append(sub[:, 1:])

    if sub.shape == (payload_h + 1, payload_w):
        if np.all(sub[1:, :] != BG) and np.all(sub[0, 1:] == BG):
            candidates.append(sub[1:, :])

    if sub.shape == (payload_h + 1, payload_w + 1):
        if np.all(sub[1:, 1:] != BG) and np.all(sub[0, 1:] == BG) and np.all(sub[1:, 0] == BG):
            candidates.append(sub[1:, 1:])

    if len(candidates) != 1:
        return None
    return np.array(candidates[0], dtype=int)


def _window_and_cargo_info(grid: np.ndarray) -> tuple[dict[int, dict[str, object]], dict[int, np.ndarray]]:
    comps = _components(grid != BG)
    windows: dict[int, dict[str, object]] = {}
    others: list[tuple[tuple[int, int, int, int], np.ndarray]] = []

    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        sub = np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
        info = _parse_window(sub)
        if info is not None:
            info["bbox"] = (r0, c0, r1, c1)
            windows[int(info["label"])] = info
        else:
            others.append(((r0, c0, r1, c1), sub))

    cargos: dict[int, np.ndarray] = {}
    for _bbox_rect, sub in others:
        tag = int(sub[0, 0])
        if tag not in windows:
            continue
        win = windows[tag]
        payload_h = int(win["R"]) * int(win["cell_h"])
        payload_w = int(win["C"]) * int(win["cell_w"])
        payload = _extract_payload(sub, payload_h, payload_w)
        if payload is not None:
            cargos[tag] = payload

    assert windows, "expected at least one window"
    assert set(cargos) == set(windows), "each window must have exactly one matching cargo"
    return windows, cargos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each tagged cargo rectangle and fill its matching orange window partitions in clockwise spiral order.

    Input structure:
    - Black background with several orange rectangular windows whose interiors are divided by orange full-row and/or full-column separators into equal subregions.
    - Each window has exactly one unique non-black, non-orange tag pixel on its outer border.
    - Separate cargo sprites carry the matching tag color at their top-left corner; the remaining adjacent rectangle is the cargo payload.

    Rule:
    - Match each cargo to the window with the same tag color.
    - Read the window partitioning to recover an R x C grid of equal subregions with size h x w.
    - Ignore the cargo tag pixel, split the rectangular payload into R x C blocks of size h x w in ordinary row-major order.
    - Paste those blocks into the window subregions following a clockwise inward spiral, leaving borders, separators, tags, and unrelated pixels unchanged.

    Key insight:
    - The window geometry determines both how finely the cargo must be cut and how many blocks exist, but the destination order is the window's spiral traversal rather than the cargo's row-major layout.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    windows, cargos = _window_and_cargo_info(g)
    for label, win in windows.items():
        payload = cargos[label]
        r_spans = win["row_spans"]
        c_spans = win["col_spans"]
        bbox_r0, bbox_c0, _bbox_r1, _bbox_c1 = win["bbox"]
        R = int(win["R"])
        C = int(win["C"])
        cell_h = int(win["cell_h"])
        cell_w = int(win["cell_w"])

        blocks: list[np.ndarray] = []
        for rr in range(R):
            for cc in range(C):
                block = payload[rr * cell_h : (rr + 1) * cell_h, cc * cell_w : (cc + 1) * cell_w]
                blocks.append(block)

        for block, (rr, cc) in zip(blocks, _spiral_positions(R, C)):
            row_start, row_end = r_spans[rr]
            col_start, col_end = c_spans[cc]
            abs_r0 = int(bbox_r0 + 1 + row_start)
            abs_r1 = int(bbox_r0 + 1 + row_end)
            abs_c0 = int(bbox_c0 + 1 + col_start)
            abs_c1 = int(bbox_c0 + 1 + col_end)
            out[abs_r0:abs_r1, abs_c0:abs_c1] = block

    return out


def _boxes_overlap_with_margin(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int
) -> bool:
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def _random_block(rng: random.Random, h: int, w: int, colors: list[int]) -> np.ndarray:
    c1 = rng.choice(colors)
    c2 = rng.choice([c for c in colors if c != c1]) if len(colors) >= 2 else c1
    c3 = rng.choice(colors)
    style = rng.randrange(6)
    block = np.full((h, w), c1, dtype=int)

    if style == 0:
        return block
    if style == 1:
        for r in range(h):
            block[r, :] = c1 if r % 2 == 0 else c2
        return block
    if style == 2:
        for c in range(w):
            block[:, c] = c1 if c % 2 == 0 else c2
        return block
    if style == 3:
        for r in range(h):
            for c in range(w):
                block[r, c] = c1 if (r + c) % 2 == 0 else c2
        return block
    if style == 4:
        for r in range(h):
            for c in range(w):
                block[r, c] = [c1, c2, c3][(r + 2 * c) % 3]
        return block
    for r in range(h):
        for c in range(w):
            if r in {0, h - 1} or c in {0, w - 1}:
                block[r, c] = c1
            else:
                block[r, c] = c2 if (r + c) % 2 == 0 else c3
    return block


def _build_payload(
    rng: random.Random, rows: int, cols: int, cell_h: int, cell_w: int, colors: list[int]
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    seen: set[bytes] = set()
    for _ in range(rows * cols):
        for _attempt in range(24):
            block = _random_block(rng, cell_h, cell_w, colors)
            key = block.tobytes()
            if key not in seen or len(seen) >= max(4, len(colors) ** 2):
                seen.add(key)
                blocks.append(block)
                break
        else:
            blocks.append(_random_block(rng, cell_h, cell_w, colors))

    payload = np.zeros((rows * cell_h, cols * cell_w), dtype=int)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            payload[
                r * cell_h : (r + 1) * cell_h,
                c * cell_w : (c + 1) * cell_w,
            ] = blocks[idx]
            idx += 1
    return payload


def _sample_window_spec(rng: random.Random) -> dict[str, int | str]:
    mode = rng.randrange(4)
    if mode == 0:
        rows, cols = 1, 1
    elif mode == 1:
        rows, cols = rng.randint(2, 3), 1
    elif mode == 2:
        rows, cols = 1, rng.randint(2, 3)
    else:
        rows, cols = rng.randint(2, 3), rng.randint(2, 3)

    cell_h = rng.randint(1, 3)
    cell_w = rng.randint(1, 3)
    outer_h = rows * cell_h + rows + 1
    outer_w = cols * cell_w + cols + 1
    cargo_h = rows * cell_h
    cargo_w = cols * cell_w + 1

    return {
        "rows": rows,
        "cols": cols,
        "cell_h": cell_h,
        "cell_w": cell_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "cargo_h": cargo_h,
        "cargo_w": cargo_w,
        "mode": ["none", "rows", "cols", "both"][mode],
    }


def _place_in_region(
    rng: random.Random,
    sizes: list[tuple[int, int]],
    region: tuple[int, int, int, int],
    margin: int,
) -> list[tuple[int, int, int, int]] | None:
    rr0, cc0, rr1, cc1 = region
    placed: list[tuple[int, int, int, int] | None] = [None] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    current: list[tuple[int, int, int, int]] = []

    for idx in order:
        h, w = sizes[idx]
        if rr1 - rr0 + 1 < h or cc1 - cc0 + 1 < w:
            return None
        success = False
        for _ in range(200):
            r0 = rng.randint(rr0, rr1 - h + 1)
            c0 = rng.randint(cc0, cc1 - w + 1)
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_boxes_overlap_with_margin(box, other, margin) for other in current):
                continue
            current.append(box)
            placed[idx] = box
            success = True
            break
        if not success:
            return None

    return [box for box in placed if box is not None]


def _label_position(rng: random.Random, h: int, w: int, side: str) -> tuple[int, int]:
    if side == "top":
        return 0, rng.randint(1, w - 2)
    if side == "bottom":
        return h - 1, rng.randint(1, w - 2)
    if side == "left":
        return rng.randint(1, h - 2), 0
    return rng.randint(1, h - 2), w - 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_windows = rng.randint(1, 3)
    specs = [_sample_window_spec(rng) for _ in range(num_windows)]

    label_pool = [color for color in range(1, 10) if color != FRAME]
    rng.shuffle(label_pool)
    labels = label_pool[:num_windows]

    layout_mode = rng.choice(["top_bottom", "left_right"])

    max_window_h = max(int(spec["outer_h"]) for spec in specs)
    max_window_w = max(int(spec["outer_w"]) for spec in specs)
    max_cargo_h = max(int(spec["cargo_h"]) for spec in specs)
    max_cargo_w = max(int(spec["cargo_w"]) for spec in specs)

    if layout_mode == "top_bottom":
        min_h = max(18, max_window_h + max_cargo_h + 6)
        min_w = max(18, max(max_window_w, max_cargo_w) + 4)
        if min_h > 30 or min_w > 30:
            return None
        H = rng.randint(min_h, 30)
        W = rng.randint(min_w, 30)
        split = rng.randint(max_window_h + 1, H - max_cargo_h - 3)
        window_region = (1, 1, split - 1, W - 2)
        cargo_region = (split + 2, 1, H - 2, W - 2)
    else:
        min_h = max(18, max(max_window_h, max_cargo_h) + 4)
        min_w = max(18, max_window_w + max_cargo_w + 6)
        if min_h > 30 or min_w > 30:
            return None
        H = rng.randint(min_h, 30)
        W = rng.randint(min_w, 30)
        split = rng.randint(max_window_w + 1, W - max_cargo_w - 3)
        window_region = (1, 1, H - 2, split - 1)
        cargo_region = (1, split + 2, H - 2, W - 2)

    window_sizes = [(int(spec["outer_h"]), int(spec["outer_w"])) for spec in specs]
    cargo_sizes = [(int(spec["cargo_h"]), int(spec["cargo_w"])) for spec in specs]

    window_boxes = _place_in_region(rng, window_sizes, window_region, margin=1)
    cargo_boxes = _place_in_region(rng, cargo_sizes, cargo_region, margin=1)
    if window_boxes is None or cargo_boxes is None:
        return None

    grid = np.zeros((H, W), dtype=int)

    for spec, box, label in zip(specs, window_boxes, labels):
        rows = int(spec["rows"])
        cols = int(spec["cols"])
        cell_h = int(spec["cell_h"])
        cell_w = int(spec["cell_w"])
        r0, c0, r1, c1 = box
        grid[r0, c0 : c1 + 1] = FRAME
        grid[r1, c0 : c1 + 1] = FRAME
        grid[r0 : r1 + 1, c0] = FRAME
        grid[r0 : r1 + 1, c1] = FRAME
        for k in range(1, rows):
            rr = r0 + k * (cell_h + 1)
            grid[rr, c0 : c1 + 1] = FRAME
        for k in range(1, cols):
            cc = c0 + k * (cell_w + 1)
            grid[r0 : r1 + 1, cc] = FRAME
        side = rng.choice(["top", "bottom", "left", "right"])
        lr, lc = _label_position(rng, r1 - r0 + 1, c1 - c0 + 1, side)
        grid[r0 + lr, c0 + lc] = label

    for spec, box, label in zip(specs, cargo_boxes, labels):
        rows = int(spec["rows"])
        cols = int(spec["cols"])
        cell_h = int(spec["cell_h"])
        cell_w = int(spec["cell_w"])
        r0, c0, _r1, _c1 = box
        payload_colors = [color for color in range(1, 10) if color not in {BG, FRAME, label}]
        rng.shuffle(payload_colors)
        palette = payload_colors[: rng.randint(2, min(4, len(payload_colors)))]
        payload = _build_payload(rng, rows, cols, cell_h, cell_w, palette)
        grid[r0, c0] = label
        grid[r0 : r0 + payload.shape[0], c0 + 1 : c0 + 1 + payload.shape[1]] = payload

    try:
        windows, cargos = _window_and_cargo_info(grid)
    except AssertionError:
        return None

    if set(windows) != set(labels) or set(cargos) != set(labels):
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
