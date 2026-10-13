from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _frame_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if len(coords) != 2 * (h + w) - 4:
        return None
    pts = set(coords)
    for r in range(r0, r1 + 1):
        if (r, c0) not in pts or (r, c1) not in pts:
            return None
    for c in range(c0, c1 + 1):
        if (r0, c) not in pts or (r1, c) not in pts:
            return None
    for r, c in pts:
        if r0 < r < r1 and c0 < c < c1:
            return None
    return int(r0), int(c0), int(r1), int(c1)


def _parse_legend_entries(strip: np.ndarray, bg: int, tile_size: int) -> list[tuple[int, np.ndarray]] | None:
    h, w = strip.shape
    if h != tile_size:
        return None
    entries: list[tuple[int, np.ndarray]] = []
    consumed = np.zeros_like(strip, dtype=bool)
    c = 0
    while c < w:
        if np.all(strip[:, c] == bg):
            c += 1
            continue
        col = strip[:, c]
        if np.any(col == bg):
            return None
        if len(set(map(int, col.tolist()))) != 1:
            return None
        if c + tile_size >= w:
            return None
        patch = strip[:, c + 1 : c + 1 + tile_size]
        if patch.shape != (tile_size, tile_size):
            return None
        if np.any(patch == bg):
            return None
        if len(set(map(int, patch.ravel().tolist()))) < 2:
            return None
        key = int(col[0])
        entries.append((key, patch.copy()))
        consumed[:, c : c + 1 + tile_size] = True
        c += 1 + tile_size
        while c < w and np.all(strip[:, c] == bg):
            c += 1
    if len(entries) < 2:
        return None
    keys = [key for key, _ in entries]
    if len(keys) != len(set(keys)):
        return None
    if int(np.count_nonzero(strip != bg)) != int(np.count_nonzero(consumed)):
        return None
    return entries


def _extract_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, np.ndarray], np.ndarray]:
    h, _ = grid.shape
    candidates: list[tuple[int, dict[int, np.ndarray], np.ndarray]] = []
    for tile_size in (2, 3):
        if h <= tile_size:
            continue
        if np.any(grid[:tile_size, :] != bg) and np.all(grid[tile_size, :] == bg):
            entries = _parse_legend_entries(grid[:tile_size, :], bg, tile_size)
            if entries is not None:
                mask = np.zeros_like(grid, dtype=bool)
                mask[:tile_size, :] = True
                mapping = {key: patch for key, patch in entries}
                candidates.append((len(entries) * 10 + tile_size, mapping, mask))
        if np.any(grid[h - tile_size :, :] != bg) and np.all(grid[h - tile_size - 1, :] == bg):
            entries = _parse_legend_entries(grid[h - tile_size :, :], bg, tile_size)
            if entries is not None:
                mask = np.zeros_like(grid, dtype=bool)
                mask[h - tile_size :, :] = True
                mapping = {key: patch for key, patch in entries}
                candidates.append((len(entries) * 10 + tile_size, mapping, mask))
    assert candidates, "expected a top or bottom legend strip"
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, mapping, mask = candidates[0]
    return mapping, mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow frame interior by looking up the key color in the edge legend.

    Input structure:
    - The background is dominant.
    - Several one-pixel-thick rectangular hollow frames appear away from the legend.
    - Each frame interior contains exactly one small non-background key blob.
    - A top or bottom legend strip contains entries made of a key-color index column
      followed by a 2x2 or 3x3 pattern patch.

    Rule:
    - Read the legend as key color -> pattern.
    - Detect every rectangular hollow frame.
    - For each frame, use the interior key color to choose the legend pattern.
    - Tile that pattern from the interior's top-left corner over every background
      cell inside the frame, preserving the border and key blob.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    mapping, legend_mask = _extract_legend(g, bg)

    work = g.copy()
    work[legend_mask] = bg
    out = g.copy()
    frames: list[tuple[int, int, int, int]] = []
    for color in sorted(set(map(int, np.unique(work))) - {bg}):
        for comp in _components(work == color):
            bbox = _frame_bbox(comp)
            if bbox is not None:
                frames.append(bbox)
    assert frames, "expected at least one hollow rectangular frame"

    for r0, c0, r1, c1 in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        key_values = set(map(int, inner[inner != bg].tolist()))
        assert len(key_values) == 1, "each frame must contain exactly one key color"
        key = next(iter(key_values))
        assert key in mapping, "frame key color must exist in the legend"
        pattern = mapping[key]
        tile_h, tile_w = pattern.shape
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if out[r, c] == bg:
                    out[r, c] = int(pattern[(r - (r0 + 1)) % tile_h, (c - (c0 + 1)) % tile_w])
    return out


_TEMPLATES_2 = [
    np.array([[0, 1], [1, 0]], dtype=int),
    np.array([[0, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [0, 1]], dtype=int),
    np.array([[0, 1], [2, 0]], dtype=int),
    np.array([[0, 2], [1, 2]], dtype=int),
]

_TEMPLATES_3 = [
    np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]], dtype=int),
    np.array([[0, 0, 1], [2, 1, 2], [1, 0, 0]], dtype=int),
    np.array([[0, 1, 0], [2, 2, 1], [0, 1, 2]], dtype=int),
]


def _random_pattern(tile_size: int, colors: list[int]) -> np.ndarray:
    base = random.choice(_TEMPLATES_2 if tile_size == 2 else _TEMPLATES_3).copy()
    base = np.rot90(base, random.randrange(4))
    if random.random() < 0.5:
        base = np.fliplr(base)
    symbol_count = int(base.max()) + 1
    chosen = random.sample(colors, symbol_count)
    patch = np.zeros_like(base, dtype=int)
    for idx, color in enumerate(chosen):
        patch[base == idx] = color
    return patch


def _legend_strip(entries: list[tuple[int, np.ndarray]], tile_size: int, width: int, bg: int) -> np.ndarray:
    gaps = [random.randint(0, 1) for _ in range(max(0, len(entries) - 1))]
    used = len(entries) * (tile_size + 1) + sum(gaps)
    strip = np.full((tile_size, width), bg, dtype=int)
    start = random.randint(0, width - used)
    c = start
    for idx, (key, patch) in enumerate(entries):
        strip[:, c] = key
        strip[:, c + 1 : c + 1 + tile_size] = patch
        c += tile_size + 1
        if idx < len(gaps):
            c += gaps[idx]
    return strip


def _key_blob(inner_h: int, inner_w: int) -> tuple[list[tuple[int, int]], int]:
    options = [([(0, 0)], 1)]
    if inner_w >= 2:
        options.append(([(0, 0), (0, 1)], 2))
    if inner_h >= 2:
        options.append(([(0, 0), (1, 0)], 2))
    cells, count = random.choice(options)
    if count == 1:
        r0 = random.randint(0, inner_h - 1)
        c0 = random.randint(0, inner_w - 1)
    elif cells[1][0] == 0:
        r0 = random.randint(0, inner_h - 1)
        c0 = random.randint(0, inner_w - 2)
    else:
        r0 = random.randint(0, inner_h - 2)
        c0 = random.randint(0, inner_w - 1)
    return [(r0 + dr, c0 + dc) for dr, dc in cells], count


def _place_frames(
    grid: np.ndarray,
    tile_size: int,
    side: str,
    frame_specs: list[tuple[int, int, int, int]],
    bg: int,
) -> list[tuple[int, int, int, int, int, int]]:
    h, w = grid.shape
    occupied = np.zeros_like(grid, dtype=bool)
    if side == "top":
        occupied[: tile_size + 1, :] = True
        row_start, row_stop = tile_size + 1, h - 1
    else:
        occupied[h - tile_size - 1 :, :] = True
        row_start, row_stop = 1, h - tile_size - 1
    placements: list[tuple[int, int, int, int, int, int]] = []
    for outer_h, outer_w, border_color, key_color in sorted(frame_specs, reverse=True):
        if row_stop - row_start < outer_h or w - 2 < outer_w:
            return []
        for _ in range(300):
            r = random.randint(row_start, row_stop - outer_h)
            c = random.randint(1, w - outer_w - 1)
            pr0 = max(0, r - 1)
            pc0 = max(0, c - 1)
            pr1 = min(h, r + outer_h + 1)
            pc1 = min(w, c + outer_w + 1)
            if np.any(occupied[pr0:pr1, pc0:pc1]):
                continue
            occupied[pr0:pr1, pc0:pc1] = True
            placements.append((r, c, outer_h, outer_w, border_color, key_color))
            break
        else:
            return []
    return placements


def _draw_frame(grid: np.ndarray, r: int, c: int, outer_h: int, outer_w: int, color: int) -> None:
    r1 = r + outer_h - 1
    c1 = c + outer_w - 1
    grid[r : r1 + 1, c] = color
    grid[r : r1 + 1, c1] = color
    grid[r, c : c1 + 1] = color
    grid[r1, c : c1 + 1] = color


def _build_case() -> np.ndarray | None:
    palette = list(range(10))
    bg = random.choice(palette)
    usable = [color for color in palette if color != bg]
    frame_count = random.randint(2, 4)
    extra_entries = random.randint(0, min(2, 6 - frame_count))
    entry_count = frame_count + extra_entries
    tile_size = random.choice([2, 3])
    side = random.choice(["top", "bottom"])

    key_colors = random.sample(usable, entry_count)
    pattern_pool = [color for color in usable if color not in key_colors]
    if len(pattern_pool) < 3:
        return None

    entries: list[tuple[int, np.ndarray]] = []
    seen_patterns: set[bytes] = set()
    for key_color in key_colors:
        patch = None
        for _ in range(40):
            candidate = _random_pattern(tile_size, pattern_pool)
            signature = candidate.tobytes()
            if signature not in seen_patterns:
                patch = candidate
                seen_patterns.add(signature)
                break
        if patch is None:
            patch = _random_pattern(tile_size, pattern_pool)
        entries.append((key_color, patch))
    random.shuffle(entries)

    gap_max = entry_count - 1
    legend_width_min = entry_count * (tile_size + 1)
    legend_width_max = legend_width_min + gap_max
    width_min = max(18, legend_width_max)
    if width_min > 30:
        return None

    frame_specs: list[tuple[int, int, int, int]] = []
    frame_entry_indices = random.sample(range(entry_count), frame_count)
    for idx in frame_entry_indices:
        inner_h = random.randint(3, 7)
        inner_w = random.randint(3, 8)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        border_color = random.choice(pattern_pool)
        key_color = entries[idx][0]
        frame_specs.append((outer_h, outer_w, border_color, key_color))

    max_outer_h = max(spec[0] for spec in frame_specs)
    height_min = max(16, tile_size + max_outer_h + 3)
    if height_min > 24:
        return None

    for _ in range(80):
        height = random.randint(height_min, 24)
        width = random.randint(width_min, 30)
        free_rows = height - tile_size - 2
        free_cols = width - 2
        reserved = sum((outer_h + 2) * (outer_w + 2) for outer_h, outer_w, _, _ in frame_specs)
        if reserved > free_rows * free_cols:
            continue

        grid = np.full((height, width), bg, dtype=int)
        strip = _legend_strip(entries, tile_size, width, bg)
        if side == "top":
            grid[:tile_size, :] = strip
        else:
            grid[height - tile_size :, :] = strip

        placements = _place_frames(grid, tile_size, side, frame_specs, bg)
        if not placements or len(placements) != frame_count:
            continue

        for r, c, outer_h, outer_w, border_color, key_color in placements:
            _draw_frame(grid, r, c, outer_h, outer_w, border_color)
            blob, _ = _key_blob(outer_h - 2, outer_w - 2)
            for dr, dc in blob:
                grid[r + 1 + dr, c + 1 + dc] = key_color

        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_case()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid input")
