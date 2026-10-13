from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
GRAY = 5
SEPARATOR = 7
SPRITE_COLORS = [1, 2, 3, 4, 6, 8, 9]
MAX_GENERATION_ATTEMPTS = 500


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for sr in range(h):
        for sc in range(w):
            color = int(grid[sr, sc])
            if color == bg or seen[sr, sc]:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            cells = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords), "area": int(coords.shape[0])})
    return comps


def _find_separator_row(grid: np.ndarray, bg: int) -> int:
    rows = [
        r
        for r in range(grid.shape[0])
        if np.all(grid[r] == int(grid[r, 0])) and int(grid[r, 0]) != bg
    ]
    assert len(rows) == 1, "expected exactly one solid non-background separator row"
    return int(rows[0])


def _contiguous_interval(indices: list[int], name: str) -> tuple[int, int]:
    assert indices, f"expected at least one {name}"
    lo, hi = int(indices[0]), int(indices[-1])
    assert indices == list(range(lo, hi + 1)), f"{name} must occupy one contiguous interval"
    return lo, hi


def _prefix_down(grid: np.ndarray, col: int, color: int) -> int:
    n = 0
    while n < grid.shape[0] and int(grid[n, col]) == color:
        n += 1
    return int(n)


def _prefix_right(grid: np.ndarray, row: int, color: int) -> int:
    n = 0
    while n < grid.shape[1] and int(grid[row, n]) == color:
        n += 1
    return int(n)


def _parse_projection_bars(canvas: np.ndarray, bg: int) -> tuple[int, int, list[int], list[int]]:
    assert int(bg) != GRAY, "background must differ from the gray projection bars"
    assert np.all((canvas == bg) | (canvas == GRAY)), "main canvas may contain only background and gray bars"

    cols = [c for c in range(canvas.shape[1]) if int(canvas[0, c]) == GRAY]
    rows = [r for r in range(canvas.shape[0]) if int(canvas[r, 0]) == GRAY]
    r0, r1 = _contiguous_interval(rows, "horizontal projection rows")
    c0, c1 = _contiguous_interval(cols, "vertical projection columns")

    row_counts = [_prefix_right(canvas, r, GRAY) for r in range(r0, r1 + 1)]
    col_counts = [_prefix_down(canvas, c, GRAY) for c in range(c0, c1 + 1)]
    assert all(v > 0 for v in row_counts + col_counts), "projection counts must be positive"

    expected = np.full_like(canvas, bg)
    for dc, count in enumerate(col_counts):
        expected[:count, c0 + dc] = GRAY
    for dr, count in enumerate(row_counts):
        expected[r0 + dr, :count] = GRAY
    assert np.array_equal(canvas, expected), "gray cells must be exactly the two edge histograms"
    return r0, c0, row_counts, col_counts


def _histograms(mask: np.ndarray) -> tuple[list[int], list[int]]:
    return [int(x) for x in mask.sum(axis=1)], [int(x) for x in mask.sum(axis=0)]


def _legend_sprites(legend: np.ndarray, bg: int) -> list[dict]:
    comps = _components_by_color(legend, bg)
    assert len(comps) >= 2, "expected multiple legend sprites"
    colors = [int(comp["color"]) for comp in comps]
    assert len(colors) == len(set(colors)), "legend sprite colors must be unique"

    sprites = []
    for comp in comps:
        crop = _crop_bbox(legend, comp["bbox"])
        mask = crop == int(comp["color"])
        assert mask.any(), "sprite mask must be non-empty"
        assert np.all(mask.any(axis=0)) and np.all(mask.any(axis=1)), "sprite bbox must be tight"
        rows, cols = _histograms(mask)
        sprites.append({"color": int(comp["color"]), "mask": mask, "rows": rows, "cols": cols})
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the hidden legend sprite whose row and column projections are encoded by gray edge bars.

    Input structure:
    - A solid non-background separator row splits the grid into an upper main canvas and a lower legend.
    - The main canvas is otherwise background, with gray vertical bars hanging from the top edge and gray
      horizontal bars extending from the left edge.
    - The lower legend contains several separated, monochrome, differently colored sprite shapes.

    Rule:
    - Read the top-edge gray prefixes over their contiguous column interval as the hidden sprite's column
      counts, and read the left-edge gray prefixes over their contiguous row interval as its row counts.
    - Find the unique legend sprite whose tight-bbox row and column pixel-count histograms both match those
      two projections.
    - Return a grid the same size as the main canvas, with all gray bars removed and the matched sprite drawn
      in its own color at the row and column intervals indicated by the bars.

    Key insight:
    - The gray marks are not objects to keep; together they are the two one-dimensional projections of an
      absent legend sprite, and the bar intervals give that sprite's missing bbox position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _background_color(g)
    sep = _find_separator_row(g, bg)
    assert sep > 0 and sep + 1 < g.shape[0], "separator must leave a canvas and a legend"

    canvas = g[:sep]
    legend = g[sep + 1 :]
    r0, c0, row_counts, col_counts = _parse_projection_bars(canvas, bg)

    matches = [
        sprite
        for sprite in _legend_sprites(legend, bg)
        if sprite["rows"] == row_counts and sprite["cols"] == col_counts
    ]
    assert len(matches) == 1, "expected a unique legend sprite matching both projections"
    sprite = matches[0]
    mask = sprite["mask"]
    h, w = mask.shape
    assert h == len(row_counts) and w == len(col_counts), "matched sprite must fit the projection intervals"
    assert r0 + h <= canvas.shape[0] and c0 + w <= canvas.shape[1], "sprite placement must fit in canvas"

    out = np.full_like(canvas, bg)
    patch = out[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = int(sprite["color"])
    return out


# --------------- generator ---------------


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _random_tight_connected_mask(h: int, w: int) -> np.ndarray | None:
    for _ in range(600):
        p = random.uniform(0.35, 0.68)
        mask = np.random.random((h, w)) < p
        area = int(mask.sum())
        if area < max(h, w) or area > h * w - 1:
            continue
        if not (np.all(mask.any(axis=0)) and np.all(mask.any(axis=1))):
            continue
        if not _is_connected(mask):
            continue
        return mask
    return None


def _random_hidden_mask() -> np.ndarray | None:
    dims = [(h, w) for h in range(3, 7) for w in range(3, 7)]
    random.shuffle(dims)
    for h, w in dims:
        for _ in range(100):
            mask = _random_tight_connected_mask(h, w)
            if mask is None:
                continue
            rows, cols = _histograms(mask)
            if rows == rows[::-1] or cols == cols[::-1]:
                continue
            if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
                continue
            return mask
    return None


def _extra_sprite(existing: set[tuple[tuple[int, int], bytes]], hidden_rows: list[int], hidden_cols: list[int]) -> np.ndarray | None:
    for _ in range(300):
        if random.random() < 0.45:
            h, w = len(hidden_rows), len(hidden_cols)
        else:
            h, w = random.randint(2, 6), random.randint(2, 6)
        mask = _random_tight_connected_mask(h, w)
        if mask is None:
            continue
        sig = _mask_signature(mask)
        rows, cols = _histograms(mask)
        if sig in existing or (rows == hidden_rows and cols == hidden_cols):
            continue
        return mask
    return None


def _place_legend_sprites(width: int, height: int, sprites: list[dict]) -> np.ndarray | None:
    legend = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    order = list(range(len(sprites)))
    random.shuffle(order)
    order.sort(key=lambda i: int(sprites[i]["mask"].size), reverse=True)

    for idx in order:
        mask = sprites[idx]["mask"]
        h, w = mask.shape
        candidates = []
        for r in range(1, height - h):
            for c in range(1, width - w):
                rr0, rr1 = max(0, r - 1), min(height, r + h + 1)
                cc0, cc1 = max(0, c - 1), min(width, c + w + 1)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    candidates.append((r, c))
        if not candidates:
            return None
        r, c = random.choice(candidates)
        color = int(sprites[idx]["color"])
        legend[r : r + h, c : c + w][mask] = color
        occupied[max(0, r - 1) : min(height, r + h + 1), max(0, c - 1) : min(width, c + w + 1)] = True
    return legend


def _draw_bars(top_h: int, width: int, r0: int, c0: int, rows: list[int], cols: list[int]) -> np.ndarray:
    top = np.full((top_h, width), BG, dtype=int)
    for dc, count in enumerate(cols):
        top[:count, c0 + dc] = GRAY
    for dr, count in enumerate(rows):
        top[r0 + dr, :count] = GRAY
    return top


def _expected_output(shape: tuple[int, int], mask: np.ndarray, color: int, r0: int, c0: int) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    h, w = mask.shape
    out[r0 : r0 + h, c0 : c0 + w][mask] = int(color)
    return out


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, sprites: list[dict], hidden_idx: int, r0: int, c0: int) -> bool:
    hidden = sprites[hidden_idx]["mask"]
    hidden_rows, hidden_cols = _histograms(hidden)
    same_size = [s for s in sprites if s["mask"].shape == hidden.shape]
    row_only = [s for s in sprites if _histograms(s["mask"])[0] == hidden_rows]
    col_only = [s for s in sprites if _histograms(s["mask"])[1] == hidden_cols]
    full = [s for s in sprites if _histograms(s["mask"]) == (hidden_rows, hidden_cols)]
    if len(same_size) < 3 or len(row_only) < 2 or len(col_only) < 2 or len(full) != 1:
        return False
    if r0 <= max(hidden_cols) or c0 <= max(hidden_rows):
        return False
    try:
        actual = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    hidden = _random_hidden_mask()
    if hidden is None:
        return None

    hidden_rows, hidden_cols = _histograms(hidden)
    masks = [hidden, np.fliplr(hidden).copy(), np.flipud(hidden).copy()]
    existing = {_mask_signature(mask) for mask in masks}
    if len(existing) != len(masks):
        return None
    for _ in range(random.choice([1, 2])):
        extra = _extra_sprite(existing, hidden_rows, hidden_cols)
        if extra is None:
            return None
        masks.append(extra)
        existing.add(_mask_signature(extra))

    colors = random.sample(SPRITE_COLORS, len(masks))
    sprites = [{"mask": mask, "color": color} for mask, color in zip(masks, colors)]
    hidden_idx = 0
    order = list(range(len(sprites)))
    random.shuffle(order)
    sprites = [sprites[i] for i in order]
    hidden_idx = order.index(0)

    h, w = hidden.shape
    top_min_h = max(hidden_cols) + 1 + h
    top_min_w = max(hidden_rows) + 1 + w
    legend_h = random.randint(max(8, max(mask.shape[0] for mask in masks) + 3), 11)
    max_top_h = 30 - 1 - legend_h
    if top_min_h > max_top_h:
        return None
    top_h = random.randint(top_min_h, max_top_h)

    width_min = max(18, top_min_w, max(mask.shape[1] for mask in masks) + 3)
    if width_min > 30:
        return None
    width = random.randint(width_min, 30)
    legend = _place_legend_sprites(width, legend_h, sprites)
    if legend is None:
        return None

    r0 = random.randint(max(hidden_cols) + 1, top_h - h)
    c0 = random.randint(max(hidden_rows) + 1, width - w)
    top = _draw_bars(top_h, width, r0, c0, hidden_rows, hidden_cols)
    sep = np.full((1, width), SEPARATOR, dtype=int)
    inp = np.vstack([top, sep, legend])
    expected = _expected_output((top_h, width), hidden, int(sprites[hidden_idx]["color"]), r0, c0)

    if inp.shape[0] > 30 or inp.shape[1] > 30 or expected.shape[0] > 30 or expected.shape[1] > 30:
        return None
    if not _semantic_checks(inp, expected, sprites, hidden_idx, r0, c0):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fable/3/049/7 input")
