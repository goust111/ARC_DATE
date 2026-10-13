import random
from collections import Counter

import numpy as np


ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            queue = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            for qr, qc in queue:
                comp.append((qr, qc))
                for dr, dc in ORTHO_DIRS:
                    nr, nc = qr + dr, qc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(comp)
    return parts


def _set_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    queue = [start]
    seen = {start}
    for r, c in queue:
        for dr, dc in ORTHO_DIRS:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen) == len(cells)


def _interior_mask(boundary_mask: np.ndarray) -> np.ndarray:
    h, w = boundary_mask.shape
    open_mask = ~boundary_mask
    outside = np.zeros((h, w), dtype=bool)
    queue: list[tuple[int, int]] = []

    def push(r: int, c: int) -> None:
        if open_mask[r, c] and not outside[r, c]:
            outside[r, c] = True
            queue.append((r, c))

    for r in range(h):
        push(r, 0)
        push(r, w - 1)
    for c in range(w):
        push(0, c)
        push(h - 1, c)

    for qr, qc in queue:
        for dr, dc in ORTHO_DIRS:
            nr, nc = qr + dr, qc + dc
            if 0 <= nr < h and 0 <= nc < w and open_mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                queue.append((nr, nc))

    return open_mask & ~outside


def _legend_side(bbox: tuple[int, int, int, int], h: int, w: int) -> str | None:
    r0, c0, r1, c1 = bbox
    sides = []
    if r0 == 0:
        sides.append("top")
    if r1 == h - 1:
        sides.append("bottom")
    if c0 == 0:
        sides.append("left")
    if c1 == w - 1:
        sides.append("right")
    if len(sides) != 1:
        return None
    return sides[0]


def _extract_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, np.ndarray], np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[str, tuple[int, int, int, int], np.ndarray]] = []
    for coords in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(coords)
        size_r = r1 - r0 + 1
        size_c = c1 - c0 + 1
        if size_r != size_c or size_r not in (3, 4):
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch == bg):
            continue
        if len(coords) != size_r * size_c:
            continue
        side = _legend_side((r0, c0, r1, c1), h, w)
        if side is None:
            continue
        pattern = patch[1:, 1:]
        if not np.array_equal(patch[0, 1:], pattern[0, :]):
            continue
        if not np.array_equal(patch[1:, 0], pattern[:, 0]):
            continue
        candidates.append((side, (r0, c0, r1, c1), patch))

    assert candidates, "main: expected edge legend entries"
    side_counts = Counter(side for side, _, _ in candidates)
    legend_side_name, legend_count = side_counts.most_common(1)[0]
    assert legend_count >= 1

    mapping: dict[int, np.ndarray] = {}
    legend_mask = np.zeros_like(grid, dtype=bool)
    for side, (r0, c0, r1, c1), patch in candidates:
        if side != legend_side_name:
            continue
        key = int(patch[0, 0])
        assert key not in mapping, "main: legend key colors must be unique"
        mapping[key] = patch[1:, 1:].copy()
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    assert mapping, "main: legend mapping must be non-empty"
    return mapping, legend_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode edge legend blocks keyed by seed colors, then tile the matching pattern
    across each enclosed region's interior and erase the legend.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    mapping, legend_mask = _extract_legend(g, bg)

    work = g.copy()
    work[legend_mask] = bg
    out = work.copy()
    transformed = 0

    for coords in _components(work != bg):
        if len(coords) < 8:
            continue
        colors = {int(work[r, c]) for r, c in coords}
        if len(colors) != 1:
            continue
        if any(r == 0 or c == 0 or r == g.shape[0] - 1 or c == g.shape[1] - 1 for r, c in coords):
            continue

        r0, c0, r1, c1 = _bbox(coords)
        boundary_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in coords:
            boundary_mask[r - r0, c - c0] = True
        interior = _interior_mask(boundary_mask)
        if not interior.any():
            continue

        patch = work[r0 : r1 + 1, c0 : c1 + 1]
        seed_colors = {int(v) for v in patch[interior] if int(v) != bg}
        if len(seed_colors) != 1:
            continue
        seed_color = next(iter(seed_colors))
        assert seed_color in mapping, "main: every seed color must appear in the legend"
        pattern = mapping[seed_color]

        rows, cols = np.where(interior)
        anchor_r = int(rows.min())
        anchor_c = int(cols.min())
        for lr, lc in zip(rows, cols):
            out[r0 + lr, c0 + lc] = int(
                pattern[(int(lr) - anchor_r) % pattern.shape[0], (int(lc) - anchor_c) % pattern.shape[1]]
            )
        transformed += 1

    assert transformed >= 1, "main: expected at least one enclosed region"
    out[legend_mask] = bg
    return out


# --------------- generator ---------------

BG = 0
GRID_MIN = 22
GRID_MAX = 30
OBJECT_COUNT_CHOICES = [1, 2, 3]
EXTRA_LEGEND_PROB = 0.4
MAX_RETRIES = 300
BUFFER = 1


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _boundary_cells(filled: set[tuple[int, int]]) -> set[tuple[int, int]]:
    boundary = set()
    for r, c in filled:
        for dr, dc in ORTHO_DIRS:
            if (r + dr, c + dc) not in filled:
                boundary.add((r, c))
                break
    return boundary


def _make_filled_shape(rng: random.Random) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    for _ in range(80):
        base_h = rng.randint(5, 8)
        base_w = rng.randint(5, 8)
        cells = {(r, c) for r in range(base_h) for c in range(base_w)}

        for _ in range(rng.randint(0, 3)):
            min_r, min_c, max_r, max_c = _bbox(list(cells))
            height = max_r - min_r + 1
            width = max_c - min_c + 1
            side = rng.choice(["top", "bottom", "left", "right"])
            if side in ("top", "bottom"):
                span = rng.randint(max(3, width // 2), width)
                start = rng.randint(min_c, max_c - span + 1)
                ext = rng.randint(1, 3)
                rows = range(min_r - ext, min_r) if side == "top" else range(max_r + 1, max_r + ext + 1)
                for r in rows:
                    for c in range(start, start + span):
                        cells.add((r, c))
            else:
                span = rng.randint(max(3, height // 2), height)
                start = rng.randint(min_r, max_r - span + 1)
                ext = rng.randint(1, 3)
                cols = range(min_c - ext, min_c) if side == "left" else range(max_c + 1, max_c + ext + 1)
                for r in range(start, start + span):
                    for c in cols:
                        cells.add((r, c))

        cells = _normalize(cells)
        boundary = _boundary_cells(cells)
        interior = cells - boundary
        if len(interior) < 4:
            continue
        if not _set_connected(boundary):
            continue
        if not _set_connected(interior):
            continue
        return cells, boundary, interior
    return None


def _pattern_checker(size: int, colors: list[int]) -> np.ndarray:
    arr = np.zeros((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            arr[r, c] = colors[(r + c) % len(colors)]
    return arr


def _pattern_vstripe(size: int, colors: list[int]) -> np.ndarray:
    arr = np.zeros((size, size), dtype=int)
    for c in range(size):
        arr[:, c] = colors[c % len(colors)]
    return arr


def _pattern_hstripe(size: int, colors: list[int]) -> np.ndarray:
    arr = np.zeros((size, size), dtype=int)
    for r in range(size):
        arr[r, :] = colors[r % len(colors)]
    return arr


def _pattern_lshape(size: int, colors: list[int]) -> np.ndarray:
    a, b = colors[0], colors[1]
    c = colors[2] if len(colors) > 2 else b
    arr = np.full((size, size), b, dtype=int)
    arr[:, 0] = a
    arr[-1, :] = a
    arr[0, -1] = c
    if size == 3:
        arr[1, 1] = c
    return arr


def _pattern_diag(size: int, colors: list[int]) -> np.ndarray:
    a, b = colors[0], colors[1]
    c = colors[2] if len(colors) > 2 else a
    arr = np.full((size, size), b, dtype=int)
    for r in range(size):
        for col in range(size):
            if r == col:
                arr[r, col] = a
            elif r > col:
                arr[r, col] = c
    return arr


def _pattern_ring(size: int, colors: list[int]) -> np.ndarray:
    a, b = colors[0], colors[1]
    c = colors[2] if len(colors) > 2 else a
    arr = np.full((size, size), a, dtype=int)
    arr[1:-1, 1:-1] = b
    arr[size // 2, size // 2] = c
    return arr


def _make_pattern(rng: random.Random, size: int, palette: list[int]) -> np.ndarray:
    if size == 2:
        colors = rng.sample(palette, rng.choice([2, 2, 3]))
        builders = [_pattern_checker, _pattern_vstripe, _pattern_hstripe, _pattern_lshape, _pattern_diag]
    else:
        colors = rng.sample(palette, rng.choice([2, 3]))
        builders = [_pattern_checker, _pattern_vstripe, _pattern_hstripe, _pattern_lshape, _pattern_diag, _pattern_ring]
    for _ in range(20):
        pattern = rng.choice(builders)(size, colors)
        if len(np.unique(pattern)) >= 2:
            return pattern
    raise RuntimeError("failed to build pattern")


def _legend_entry(key: int, pattern: np.ndarray) -> np.ndarray:
    size = pattern.shape[0] + 1
    entry = np.zeros((size, size), dtype=int)
    entry[0, 0] = key
    entry[0, 1:] = pattern[0, :]
    entry[1:, 0] = pattern[:, 0]
    entry[1:, 1:] = pattern
    return entry


def _place_legend(
    grid: np.ndarray,
    rng: random.Random,
    side: str,
    entries: list[np.ndarray],
) -> int:
    h, w = grid.shape
    sizes = [entry.shape[0] for entry in entries]
    band = max(sizes)
    gap = 1
    if side in ("top", "bottom"):
        total = sum(sizes) + gap * (len(entries) - 1)
        start = rng.randint(1, max(1, w - total - 1))
        cursor = start
        for entry in entries:
            size = entry.shape[0]
            r0 = 0 if side == "top" else h - size
            grid[r0 : r0 + size, cursor : cursor + size] = entry
            cursor += size + gap
    else:
        total = sum(sizes) + gap * (len(entries) - 1)
        start = rng.randint(1, max(1, h - total - 1))
        cursor = start
        for entry in entries:
            size = entry.shape[0]
            c0 = 0 if side == "left" else w - size
            grid[cursor : cursor + size, c0 : c0 + size] = entry
            cursor += size + gap
    return band


def _usable_area(h: int, w: int, side: str, band: int) -> tuple[int, int, int, int]:
    margin = band + 2
    if side == "top":
        return margin, h - 2, 1, w - 2
    if side == "bottom":
        return 1, h - margin - 1, 1, w - 2
    if side == "left":
        return 1, h - 2, margin, w - 2
    return 1, h - 2, 1, w - margin - 1


def _footprint_clear(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - BUFFER)
    cc0 = max(0, c0 - BUFFER)
    rr1 = min(occ.shape[0], r0 + h + BUFFER)
    cc1 = min(occ.shape[1], c0 + w + BUFFER)
    return not occ[rr0:rr1, cc0:cc1].any()


def _mark_footprint(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - BUFFER)
    cc0 = max(0, c0 - BUFFER)
    rr1 = min(occ.shape[0], r0 + h + BUFFER)
    cc1 = min(occ.shape[1], c0 + w + BUFFER)
    occ[rr0:rr1, cc0:cc1] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    side = rng.choice(["top", "bottom", "left", "right"])
    outline = rng.choice(list(range(1, 10)))

    object_count = rng.choice(OBJECT_COUNT_CHOICES)
    legend_count = object_count + (1 if rng.random() < EXTRA_LEGEND_PROB else 0)
    legend_count = min(legend_count, 4)

    available_keys = [color for color in range(1, 10) if color != outline]
    rng.shuffle(available_keys)
    keys = available_keys[:legend_count]
    used_keys = rng.sample(keys, object_count)

    pattern_palette = [color for color in range(1, 10) if color not in {outline, *keys}]
    if len(pattern_palette) < 3:
        return None

    entry_sizes = [rng.choice([2, 3]) for _ in range(legend_count)]
    entries = [_legend_entry(key, _make_pattern(rng, size, pattern_palette)) for key, size in zip(keys, entry_sizes)]

    total_len = sum(entry.shape[0] for entry in entries) + len(entries) - 1
    if side in ("top", "bottom") and total_len + 2 > w:
        return None
    if side in ("left", "right") and total_len + 2 > h:
        return None

    grid = np.full((h, w), BG, dtype=int)
    band = _place_legend(grid, rng, side, entries)
    usable_r0, usable_r1, usable_c0, usable_c1 = _usable_area(h, w, side, band)
    if usable_r1 - usable_r0 < 6 or usable_c1 - usable_c0 < 6:
        return None

    occ = np.zeros((h, w), dtype=bool)
    occ[grid != BG] = True
    objects = []
    for seed_color in used_keys:
        made = _make_filled_shape(rng)
        if made is None:
            return None
        _, boundary, interior = made
        shape_h = 1 + max(r for r, _ in boundary)
        shape_w = 1 + max(c for _, c in boundary)
        objects.append((seed_color, shape_h, shape_w, boundary, interior))

    objects.sort(key=lambda item: item[1] * item[2], reverse=True)

    for seed_color, shape_h, shape_w, boundary, interior in objects:
        placed = False
        row_choices = list(range(usable_r0, usable_r1 - shape_h + 2))
        col_choices = list(range(usable_c0, usable_c1 - shape_w + 2))
        rng.shuffle(row_choices)
        rng.shuffle(col_choices)
        for r0 in row_choices:
            for c0 in col_choices:
                if not _footprint_clear(occ, r0, c0, shape_h, shape_w):
                    continue
                for br, bc in boundary:
                    grid[r0 + br, c0 + bc] = outline
                seed_r, seed_c = rng.choice(list(interior))
                grid[r0 + seed_r, c0 + seed_c] = seed_color
                _mark_footprint(occ, r0, c0, shape_h, shape_w)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
