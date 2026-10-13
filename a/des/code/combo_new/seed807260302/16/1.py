from __future__ import annotations

import numpy as np


BG = 0
FRAME = 5
NON_SPECIAL_COLORS = tuple(c for c in range(10) if c not in {BG, FRAME})
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _border_majority(g: np.ndarray) -> int:
    border = np.concatenate((g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]))
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[int(cnts.argmax())])


def _find_palette(g: np.ndarray, bg: int) -> tuple[int, list[int]]:
    h, w = g.shape
    for r in range(h - 1, -1, -1):
        cols = [int(c) for c in np.flatnonzero(g[r, :] != bg)]
        if len(cols) < 2:
            continue
        if any(cols[i + 1] - cols[i] != 2 for i in range(len(cols) - 1)):
            continue

        ok = True
        for c in cols:
            if c > 0 and int(g[r, c - 1]) != bg:
                ok = False
                break
            if c + 1 < w and int(g[r, c + 1]) != bg:
                ok = False
                break
            if r > 0 and int(g[r - 1, c]) != bg:
                ok = False
                break
            if r + 1 < h and int(g[r + 1, c]) != bg:
                ok = False
                break
        if ok:
            return int(r), cols
    raise AssertionError("failed to find the bottom palette row")


def _component(
    g: np.ndarray,
    start: tuple[int, int],
    blocked: np.ndarray,
    seen: np.ndarray,
) -> tuple[int, list[tuple[int, int]]]:
    h, w = g.shape
    sr, sc = start
    color = int(g[sr, sc])
    stack = [(int(sr), int(sc))]
    seen[sr, sc] = True
    coords: list[tuple[int, int]] = []

    while stack:
        r, c = stack.pop()
        coords.append((int(r), int(c)))
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if blocked[nr, nc] or seen[nr, nc] or int(g[nr, nc]) != color:
                continue
            seen[nr, nc] = True
            stack.append((int(nr), int(nc)))
    return color, coords


def _frame_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    expected_area = 2 * (h + w) - 4
    if len(coords) != expected_area:
        return None

    pts = set(coords)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in {r0, r1} or c in {c0, c1}
            if on_border != ((r, c) in pts):
                return None
    return int(r0), int(c0), int(r1), int(c1)


def _find_frames(g: np.ndarray, bg: int, palette_row: int, palette_cols: list[int]) -> list[dict]:
    blocked = np.zeros_like(g, dtype=bool)
    blocked[palette_row, palette_cols] = True
    seen = blocked.copy()
    frames: list[dict] = []

    h, w = g.shape
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color, coords = _component(g, (r, c), blocked, seen)
            bbox = _frame_bbox(coords)
            if bbox is None:
                continue
            frames.append({"color": int(color), "coords": coords, "bbox": bbox})

    if not frames:
        raise AssertionError("failed to find any rectangular frames")
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each rectangular frame from a bottom palette using the number of interior color blocks.

    Input structure:
    - The background is black.
    - Several one-pixel-thick closed rectangular frames sit in the main area.
    - Each frame encloses a black interior with scattered one-pixel color blocks.
    - A bottom row contains a palette strip: distinct single pixels arranged left to right with one black gap between neighbors.

    Rule:
    - Count how many non-black color-block pixels lie strictly inside each rectangular frame.
    - Use that count N as a 1-based index into the bottom palette strip, read left to right.
    - Recolor the entire frame border to the selected palette color.
    - Clear every interior color-block pixel back to black, while leaving the palette strip unchanged.

    Key insight:
    - The colored pixels inside each frame are only counters; their quantity selects a color from the bottom lookup strip, which is then applied to the frame itself.
    """
    g = np.asarray(grid, dtype=int)
    bg = _border_majority(g)
    palette_row, palette_cols = _find_palette(g, bg)
    palette = [int(g[palette_row, c]) for c in palette_cols]
    frames = _find_frames(g, bg, palette_row, palette_cols)

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        count = int(np.count_nonzero(inner != bg))
        assert 1 <= count <= len(palette), "frame count must index the palette"
        target = int(palette[count - 1])

        for r, c in frame["coords"]:
            out[int(r), int(c)] = target
        out[r0 + 1 : r1, c0 + 1 : c1] = bg

    return out


def _sample_counts(rng: np.random.Generator, n_frames: int, palette_len: int) -> list[int]:
    counts = [int(rng.integers(1, palette_len + 1)) for _ in range(n_frames)]
    if n_frames > 1 and len(set(counts)) == 1:
        idx = int(rng.integers(0, n_frames))
        counts[idx] = 1 if counts[idx] != 1 else int(palette_len)
    return counts


def _frame_size_candidates(count: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for h in range(5, 10):
        for w in range(5, 11):
            interior = (h - 2) * (w - 2)
            if interior >= max(6, 2 * count):
                candidates.append((int(h), int(w)))
    return candidates


def _pick_inner_cells(
    rng: np.random.Generator,
    height: int,
    width: int,
    count: int,
) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(height) for c in range(width)]
    order = rng.permutation(len(cells))
    chosen: list[tuple[int, int]] = []
    for idx in order:
        r, c = cells[int(idx)]
        if all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in chosen):
            chosen.append((int(r), int(c)))
            if len(chosen) == count:
                return chosen

    fallback = rng.choice(len(cells), size=count, replace=False)
    return [cells[int(i)] for i in fallback]


def _place_frames(
    rng: np.random.Generator,
    grid_shape: tuple[int, int],
    sizes: list[tuple[int, int]],
) -> list[tuple[int, int, int, int]] | None:
    h, w = grid_shape
    used = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int, int, int] | None] = [None] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)

    for idx in order:
        fh, fw = sizes[idx]
        if h - fh - 2 <= 1 or w - fw - 2 <= 1:
            return None
        placed = False
        for _ in range(150):
            r0 = int(rng.integers(1, h - fh - 1))
            c0 = int(rng.integers(1, w - fw - 1))
            r1 = r0 + fh - 1
            c1 = c0 + fw - 1
            if r1 > h - 3:
                continue
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h - 1, r1 + 1)
            cc1 = min(w - 1, c1 + 1)
            if used[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            used[r0 : r1 + 1, c0 : c1 + 1] = True
            placements[idx] = (int(r0), int(c0), int(r1), int(c1))
            placed = True
            break
        if not placed:
            return None

    return [p for p in placements if p is not None]


def _build_input(rng: np.random.Generator) -> np.ndarray | None:
    palette_len = int(rng.integers(3, 7))
    n_frames = int(rng.integers(2, 6))
    palette = [int(x) for x in rng.choice(np.array(NON_SPECIAL_COLORS, dtype=int), size=palette_len, replace=False)]
    counts = _sample_counts(rng, n_frames, palette_len)
    sizes = [rng.choice(_frame_size_candidates(count)) for count in counts]
    sizes = [(int(h), int(w)) for h, w in sizes]

    min_h = max(14, max(h for h, _w in sizes) + 5)
    min_w = max(14, max(w for _h, w in sizes) + 5, 2 * palette_len + 3)
    grid_h = int(rng.integers(min_h, 31))
    grid_w = int(rng.integers(min_w, 31))

    placements = _place_frames(rng, (grid_h, grid_w), sizes)
    if placements is None:
        return None

    span = 2 * palette_len - 1
    if grid_w - span <= 1:
        return None
    palette_start = int(rng.integers(1, grid_w - span))

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    palette_row = grid_h - 1
    for i, color in enumerate(palette):
        grid[palette_row, palette_start + 2 * i] = int(color)

    for (r0, c0, r1, c1), count in zip(placements, counts, strict=False):
        grid[r0, c0 : c1 + 1] = FRAME
        grid[r1, c0 : c1 + 1] = FRAME
        grid[r0 : r1 + 1, c0] = FRAME
        grid[r0 : r1 + 1, c1] = FRAME

        ih = r1 - r0 - 1
        iw = c1 - c0 - 1
        cells = _pick_inner_cells(rng, ih, iw, count)
        colors = rng.choice(np.array(NON_SPECIAL_COLORS, dtype=int), size=count, replace=False)
        for (rr, cc), color in zip(cells, colors, strict=False):
            grid[r0 + 1 + rr, c0 + 1 + cc] = int(color)

    try:
        _ = main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _build_input(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
