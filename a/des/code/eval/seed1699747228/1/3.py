from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
FRAME = 5
SEED = 2
PALETTE_POOL = (1, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
MAX_GRID_SIDE = 36

TEMPLATES = [
    {"slots": [(0, 0), (0, 1)], "edges": [(0, 1)]},
    {"slots": [(0, 0), (0, 1), (0, 2)], "edges": [(0, 1), (1, 2)]},
    {"slots": [(0, 0), (0, 1), (1, 1)], "edges": [(0, 1), (1, 2)]},
    {"slots": [(0, 0), (0, 1), (1, 0)], "edges": []},
    {"slots": [(0, 0), (0, 1), (1, 0), (1, 1)], "edges": []},
    {"slots": [(0, 0), (0, 1), (1, 0), (1, 1)], "edges": [(0, 1), (2, 3)]},
    {"slots": [(0, 0), (0, 1), (0, 2), (1, 0)], "edges": [(0, 1), (1, 2)]},
    {"slots": [(0, 0), (0, 1), (0, 2), (1, 1)], "edges": [(1, 0), (1, 2), (1, 3)]},
]

SEED_PATTERNS = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
]


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[dict[str, object]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c
            touches_edge = r == 0 or r == h - 1 or c == 0 or c == w - 1
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_edge = True
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(
                {
                    "coords": np.array(coords, dtype=int),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": len(coords),
                    "touches_edge": touches_edge,
                }
            )
    return comps


def _perimeter_is_color(grid: np.ndarray, color: int, r0: int, c0: int, r1: int, c1: int) -> bool:
    return bool(
        np.all(grid[r0, c0 : c1 + 1] == color)
        and np.all(grid[r1, c0 : c1 + 1] == color)
        and np.all(grid[r0 : r1 + 1, c0] == color)
        and np.all(grid[r0 : r1 + 1, c1] == color)
    )


def _enclosed_rectangles_for_color(grid: np.ndarray, color: int) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    interiors: list[tuple[int, int, int, int]] = []
    for comp in _components(grid != color):
        if bool(comp["touches_edge"]):
            continue
        r0, c0, r1, c1 = comp["bbox"]
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        if int(comp["area"]) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        if not _perimeter_is_color(grid, color, r0 - 1, c0 - 1, r1 + 1, c1 + 1):
            continue
        interiors.append((int(r0), int(c0), int(r1), int(c1)))
    return interiors


def _find_frame_color(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    candidates: list[tuple[int, int, list[tuple[int, int, int, int]]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        interiors = _enclosed_rectangles_for_color(grid, color)
        if interiors:
            candidates.append((len(interiors), color, interiors))
    assert candidates, "expected a frame color"
    best_count = max(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == best_count]
    assert len(best) == 1, "frame color must be unique"
    _count, color, interiors = best[0]
    return int(color), interiors


def _find_seed_color(
    grid: np.ndarray, islands: list[tuple[int, int, int, int]], bg: int
) -> int:
    colors: set[int] = set()
    for r0, c0, r1, c1 in islands:
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        colors.update(int(v) for v in np.unique(patch) if int(v) != bg)
    assert len(colors) == 1, "expected a single seed color"
    return colors.pop()


def _find_palette_rows(grid: np.ndarray, bg: int, frame_color: int, seed_color: int) -> list[int]:
    mask = ~np.isin(grid, [bg, frame_color, seed_color])
    comps = _components(mask)
    assert len(comps) == 1, "expected a single palette block"
    comp = comps[0]
    r0, c0, r1, c1 = comp["bbox"]
    assert int(comp["area"]) == (r1 - r0 + 1) * (c1 - c0 + 1), "palette must be rectangular"
    rows: list[int] = []
    for r in range(r0, r1 + 1):
        values = {int(v) for v in grid[r, c0 : c1 + 1]}
        assert len(values) == 1, "palette rows must be monochrome"
        rows.append(values.pop())
    assert len(set(rows)) == len(rows), "palette colors must be distinct"
    return rows


def _count_runs(line: np.ndarray) -> int:
    total = 0
    in_run = False
    for value in line.tolist():
        if value and not in_run:
            total += 1
            in_run = True
        elif not value:
            in_run = False
    return total


def _bridge_count(grid: np.ndarray, frame_color: int, inner_bbox: tuple[int, int, int, int]) -> int:
    ir0, ic0, ir1, ic1 = inner_bbox
    r0, c0, r1, c1 = ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1
    total = 0
    if r0 > 0:
        total += _count_runs(grid[r0 - 1, c0 + 1 : c1] == frame_color)
    if r1 + 1 < grid.shape[0]:
        total += _count_runs(grid[r1 + 1, c0 + 1 : c1] == frame_color)
    if c0 > 0:
        total += _count_runs(grid[r0 + 1 : r1, c0 - 1] == frame_color)
    if c1 + 1 < grid.shape[1]:
        total += _count_runs(grid[r0 + 1 : r1, c1 + 1] == frame_color)
    return total


def _seed_mask(patch: np.ndarray, seed_color: int) -> np.ndarray:
    coords = np.argwhere(patch == seed_color)
    assert coords.size > 0, "island must contain a seed"
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    return patch[r0 : r1 + 1, c0 : c1 + 1] == seed_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each island seed from the palette row indexed by its bridge count,
    then scale that seed by the island's size ratio relative to the smallest
    island interior while preserving frames, bridges, and the palette.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    frame_color, islands = _find_frame_color(g, bg)
    seed_color = _find_seed_color(g, islands, bg)
    palette_rows = _find_palette_rows(g, bg, frame_color, seed_color)

    inner_sizes = [(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in islands]
    min_h = min(size[0] for size in inner_sizes)
    min_w = min(size[1] for size in inner_sizes)

    out = g.copy()
    out[out == seed_color] = bg

    for r0, c0, r1, c1 in islands:
        inner_h = r1 - r0 + 1
        inner_w = c1 - c0 + 1
        factor = inner_w // min_w
        assert factor >= 1 and inner_w == min_w * factor and inner_h == min_h * factor

        degree = _bridge_count(g, frame_color, (r0, c0, r1, c1))
        assert 0 <= degree < len(palette_rows), "palette must cover every bridge count"
        color = palette_rows[degree]

        seed = _seed_mask(g[r0 : r1 + 1, c0 : c1 + 1], seed_color)
        scaled = np.kron(seed.astype(int), np.ones((factor, factor), dtype=int)).astype(bool)
        sh, sw = scaled.shape
        assert sh <= inner_h and sw <= inner_w
        dr = (inner_h - sh) // 2
        dc = (inner_w - sw) // 2
        target = out[r0 : r1 + 1, c0 : c1 + 1]
        target[dr : dr + sh, dc : dc + sw][scaled] = color

    return out


def _normalize_slots(slots: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in slots)
    min_c = min(c for _, c in slots)
    return [(r - min_r, c - min_c) for r, c in slots]


def _transform_template(
    template: dict[str, object]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    slots = list(template["slots"])
    if random.random() < 0.5:
        slots = [(c, r) for r, c in slots]
    slots = _normalize_slots(slots)
    max_r = max(r for r, _ in slots)
    max_c = max(c for _, c in slots)
    if random.random() < 0.5:
        slots = [(r, max_c - c) for r, c in slots]
    if random.random() < 0.5:
        slots = [(max_r - r, c) for r, c in slots]
    return _normalize_slots(slots), list(template["edges"])


def _size_profile(slot_rows: int, slot_cols: int) -> tuple[int, int, int]:
    big_case = slot_rows <= 2 and slot_cols <= 2 and random.random() < 0.35
    if big_case:
        return random.randint(2, 3), 3, 3
    if slot_rows == 3:
        base_h = random.randint(2, 3)
    else:
        base_h = random.randint(3, 4)
    if slot_cols == 3:
        base_w = 3
    else:
        base_w = random.randint(3, 4)
    return base_h, base_w, 2


def _choose_scales(count: int, max_factor: int) -> list[int]:
    if max_factor == 1:
        return [1] * count
    if random.random() < 0.25:
        return [1] * count
    scales = [random.randint(1, max_factor) for _ in range(count)]
    scales[random.randrange(count)] = 1
    if max(scales) == 1:
        scales[random.randrange(count)] = max_factor
    return scales


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _compatible_patterns(base_h: int, base_w: int) -> list[np.ndarray]:
    patterns = [pattern for pattern in SEED_PATTERNS if pattern.shape[0] <= base_h and pattern.shape[1] <= base_w]
    assert patterns, "expected at least one fitting seed pattern"
    return patterns


def _center_mask(mask: np.ndarray, h: int, w: int) -> tuple[int, int]:
    mh, mw = mask.shape
    return (h - mh) // 2, (w - mw) // 2


def _draw_seed(
    grid: np.ndarray, inner_bbox: tuple[int, int, int, int], pattern: np.ndarray, color: int
) -> None:
    r0, c0, r1, c1 = inner_bbox
    inner_h = r1 - r0 + 1
    inner_w = c1 - c0 + 1
    dr, dc = _center_mask(pattern, inner_h, inner_w)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    sub = patch[dr : dr + pattern.shape[0], dc : dc + pattern.shape[1]]
    sub[pattern] = color


def _draw_scaled_seed(
    grid: np.ndarray,
    inner_bbox: tuple[int, int, int, int],
    pattern: np.ndarray,
    color: int,
    factor: int,
) -> None:
    scaled = np.kron(pattern.astype(int), np.ones((factor, factor), dtype=int)).astype(bool)
    r0, c0, r1, c1 = inner_bbox
    inner_h = r1 - r0 + 1
    inner_w = c1 - c0 + 1
    dr, dc = _center_mask(scaled, inner_h, inner_w)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    sub = patch[dr : dr + scaled.shape[0], dc : dc + scaled.shape[1]]
    sub[scaled] = color


def _palette_origin(corner: str, h: int, w: int, ph: int, pw: int) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - pw
    if corner == "bl":
        return h - ph, 0
    return h - ph, w - pw


def _make_case() -> tuple[np.ndarray, np.ndarray] | None:
    template = random.choice(TEMPLATES)
    slots, edges = _transform_template(template)
    island_count = len(slots)
    slot_rows = max(r for r, _ in slots) + 1
    slot_cols = max(c for _, c in slots) + 1

    base_h, base_w, max_factor = _size_profile(slot_rows, slot_cols)
    scales = _choose_scales(island_count, max_factor)
    outer_sizes = [(base_h * scale + 2, base_w * scale + 2) for scale in scales]
    max_outer_h = max(h for h, _ in outer_sizes)
    max_outer_w = max(w for _, w in outer_sizes)
    row_gap = random.randint(2, 4)
    col_gap = random.randint(2, 5)

    degrees = [0] * island_count
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1
    palette_rows = max(degrees) + 1
    palette_colors = random.sample(PALETTE_POOL, palette_rows)
    palette_w = random.randint(2, 3)
    corner = random.choice(["tl", "tr", "bl", "br"])

    top_margin = random.randint(1, 2) + (palette_rows + 1 if corner in {"tl", "tr"} else 0)
    bottom_margin = random.randint(1, 2) + (palette_rows + 1 if corner in {"bl", "br"} else 0)
    left_margin = random.randint(1, 2) + (palette_w + 1 if corner in {"tl", "bl"} else 0)
    right_margin = random.randint(1, 2) + (palette_w + 1 if corner in {"tr", "br"} else 0)

    grid_h = top_margin + slot_rows * max_outer_h + (slot_rows - 1) * row_gap + bottom_margin
    grid_w = left_margin + slot_cols * max_outer_w + (slot_cols - 1) * col_gap + right_margin
    if max(grid_h, grid_w) > MAX_GRID_SIDE:
        return None

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    out = np.full((grid_h, grid_w), BG, dtype=int)

    palette_r0, palette_c0 = _palette_origin(corner, grid_h, grid_w, palette_rows, palette_w)
    for i, color in enumerate(palette_colors):
        inp[palette_r0 + i, palette_c0 : palette_c0 + palette_w] = color
        out[palette_r0 + i, palette_c0 : palette_c0 + palette_w] = color

    row_anchors = {
        r: top_margin + r * (max_outer_h + row_gap) + max_outer_h // 2 for r in range(slot_rows)
    }
    col_anchors = {
        c: left_margin + c * (max_outer_w + col_gap) + max_outer_w // 2 for c in range(slot_cols)
    }

    patterns = _compatible_patterns(base_h, base_w)
    island_boxes: list[tuple[int, int, int, int]] = []
    island_patterns: list[np.ndarray] = []

    for idx, (slot_r, slot_c) in enumerate(slots):
        outer_h, outer_w = outer_sizes[idx]
        band_top = top_margin + slot_r * (max_outer_h + row_gap)
        band_left = left_margin + slot_c * (max_outer_w + col_gap)
        r0 = band_top + (max_outer_h - outer_h) // 2
        c0 = band_left + (max_outer_w - outer_w) // 2
        _draw_frame(inp, r0, c0, outer_h, outer_w, FRAME)
        _draw_frame(out, r0, c0, outer_h, outer_w, FRAME)
        inner_bbox = (r0 + 1, c0 + 1, r0 + outer_h - 2, c0 + outer_w - 2)
        island_boxes.append(inner_bbox)
        pattern = random.choice(patterns)
        island_patterns.append(pattern)
        _draw_seed(inp, inner_bbox, pattern, SEED)
        _draw_scaled_seed(out, inner_bbox, pattern, palette_colors[degrees[idx]], scales[idx])

    for a, b in edges:
        ar0, ac0, ar1, ac1 = island_boxes[a][0] - 1, island_boxes[a][1] - 1, island_boxes[a][2] + 1, island_boxes[a][3] + 1
        br0, bc0, br1, bc1 = island_boxes[b][0] - 1, island_boxes[b][1] - 1, island_boxes[b][2] + 1, island_boxes[b][3] + 1
        sa = slots[a]
        sb = slots[b]
        if sa[0] == sb[0]:
            row = row_anchors[sa[0]]
            if ac0 < bc0:
                c_start, c_end = ac1, bc0
            else:
                c_start, c_end = bc1, ac0
            inp[row, c_start : c_end + 1] = FRAME
            out[row, c_start : c_end + 1] = FRAME
        else:
            col = col_anchors[sa[1]]
            if ar0 < br0:
                r_start, r_end = ar1, br0
            else:
                r_start, r_end = br1, ar0
            inp[r_start : r_end + 1, col] = FRAME
            out[r_start : r_end + 1, col] = FRAME

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _make_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
