from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
ARC_COLORS = tuple(range(10))
PALETTE_COLORS = tuple(color for color in ARC_COLORS if color != BLACK)
GENERATION_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

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
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_palette(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    candidates: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == BLACK:
            continue
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            if len(coords) != 4 or r1 - r0 + 1 != 2 or c1 - c0 + 1 != 2:
                continue
            candidates.append(
                {
                    "color": color,
                    "r0": r0,
                    "c0": c0,
                    "c1": c1,
                    "coords": coords,
                }
            )

    groups: dict[tuple[int, int], list[dict[str, object]]] = {}
    for entry in candidates:
        key = (int(entry["c0"]), int(entry["c1"]))
        groups.setdefault(key, []).append(entry)

    valid: list[tuple[int, int, list[dict[str, object]]]] = []
    for (c0, c1), entries in groups.items():
        ordered = sorted(entries, key=lambda item: int(item["r0"]))
        if len(ordered) < 2:
            continue
        if len({int(item["color"]) for item in ordered}) != len(ordered):
            continue

        ok = True
        for prev, cur in zip(ordered, ordered[1:], strict=False):
            prev_r0 = int(prev["r0"])
            cur_r0 = int(cur["r0"])
            if cur_r0 - prev_r0 != 3:
                ok = False
                break
            if not np.all(grid[prev_r0 + 2, c0 : c1 + 1] == BLACK):
                ok = False
                break
        if ok:
            valid.append((c0, -len(ordered), ordered))

    assert valid, "expected a palette column of stacked 2x2 swatches"
    _, _, palette_entries = min(valid, key=lambda item: (item[0], item[1]))

    palette_mask = np.zeros_like(grid, dtype=bool)
    palette_colors: list[int] = []
    for entry in palette_entries:
        coords = entry["coords"]
        palette_mask[coords[:, 0], coords[:, 1]] = True
        palette_colors.append(int(entry["color"]))

    return palette_colors, palette_mask


def _count_holes(grid: np.ndarray, coords: np.ndarray) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    black_comps = _components(crop == BLACK)

    holes = 0
    h, w = crop.shape
    for comp in black_comps:
        touches_border = np.any(
            (comp[:, 0] == 0) | (comp[:, 0] == h - 1) | (comp[:, 1] == 0) | (comp[:, 1] == w - 1)
        )
        if not touches_border:
            holes += 1
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey closed outline by indexing into the palette with its hole count.

    Input structure:
    - Background is black.
    - A vertical palette column on the left contains solid 2x2 swatches separated by one black row.
    - Grey 4-connected outline objects appear on the right; their enclosed black hole count varies.

    Rule:
    - Count the number of black 4-connected holes fully enclosed by each grey object.
    - Use that hole count as a 0-based index into the palette swatches.
    - Recolor all grey pixels of the object to the indexed palette color.
    - Leave the palette and black holes unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette_colors, palette_mask = _find_palette(g)
    object_mask = (g == GREY) & (~palette_mask)
    objects = _components(object_mask)
    assert objects, "expected at least one grey object"

    out = g.copy()
    for coords in objects:
        holes = _count_holes(g, coords)
        assert 0 <= holes < len(palette_colors), "hole count must index the palette"
        out[coords[:, 0], coords[:, 1]] = int(palette_colors[holes])
    return out


def _positive_composition(total: int, parts: int) -> list[int]:
    assert total >= parts >= 1
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    points = [0, *cuts, total]
    return [points[i + 1] - points[i] for i in range(parts)]


def _build_object(hole_count: int) -> np.ndarray:
    if hole_count == 0:
        if random.random() < 0.5:
            h = 2
            w = random.randint(2, 8)
        else:
            h = random.randint(2, 8)
            w = 2
        return np.full((h, w), GREY, dtype=int)

    vertical = random.random() < 0.5
    if vertical:
        inner_h = random.randint(1, 6)
        min_inner_w = 2 * hole_count - 1
        inner_w = min_inner_w + random.randint(0, 5)
        h = inner_h + 2
        w = inner_w + 2
        sprite = np.zeros((h, w), dtype=int)
        sprite[0, :] = GREY
        sprite[-1, :] = GREY
        sprite[:, 0] = GREY
        sprite[:, -1] = GREY

        chamber_widths = _positive_composition(inner_w - (hole_count - 1), hole_count)
        col = 1
        for width in chamber_widths[:-1]:
            col += width
            sprite[1:-1, col] = GREY
            col += 1
        return sprite

    inner_w = random.randint(1, 6)
    min_inner_h = 2 * hole_count - 1
    inner_h = min_inner_h + random.randint(0, 5)
    h = inner_h + 2
    w = inner_w + 2
    sprite = np.zeros((h, w), dtype=int)
    sprite[0, :] = GREY
    sprite[-1, :] = GREY
    sprite[:, 0] = GREY
    sprite[:, -1] = GREY

    chamber_heights = _positive_composition(inner_h - (hole_count - 1), hole_count)
    row = 1
    for height in chamber_heights[:-1]:
        row += height
        sprite[row, 1:-1] = GREY
        row += 1
    return sprite


def _reserve(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0], r0 + h + pad)
    cc1 = min(mask.shape[1], c0 + w + pad)
    mask[rr0:rr1, cc0:cc1] = True


def _can_place(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0], r0 + h + pad)
    cc1 = min(mask.shape[1], c0 + w + pad)
    return not mask[rr0:rr1, cc0:cc1].any()


def _paint_object(
    grid: np.ndarray,
    expected: np.ndarray,
    sprite: np.ndarray,
    top: int,
    left: int,
    out_color: int,
) -> None:
    h, w = sprite.shape
    grey_mask = sprite == GREY
    grid[top : top + h, left : left + w][grey_mask] = GREY
    expected[top : top + h, left : left + w][grey_mask] = out_color


def _sample_palette_colors(palette_len: int) -> list[int]:
    # Excluding grey keeps the visual mapping obvious; main still detects the palette structurally.
    return random.sample([c for c in PALETTE_COLORS if c != GREY], palette_len)


def _gen_attempt() -> np.ndarray | None:
    palette_len = random.randint(3, 6)
    object_count = random.randint(2, min(4, palette_len))
    hole_counts = random.sample(range(palette_len), object_count)
    palette_colors = _sample_palette_colors(palette_len)
    sprites = [(holes, _build_object(holes)) for holes in hole_counts]

    max_h = max(sprite.shape[0] for _holes, sprite in sprites)
    max_w = max(sprite.shape[1] for _holes, sprite in sprites)
    palette_h = 3 * palette_len - 1

    min_h = max(12, palette_h + 2, max_h + 2)
    min_w = max(18, 2 + 4 + max_w + 2)
    if min_h > 30 or min_w > 30:
        return None

    h = random.randint(min_h, 30)
    w = random.randint(min_w, 30)
    grid = np.zeros((h, w), dtype=int)
    expected = np.zeros((h, w), dtype=int)

    palette_top = random.randint(0, h - palette_h)
    palette_left = random.randint(0, 1)
    for idx, color in enumerate(palette_colors):
        r0 = palette_top + 3 * idx
        grid[r0 : r0 + 2, palette_left : palette_left + 2] = color
        expected[r0 : r0 + 2, palette_left : palette_left + 2] = color

    occupied = np.zeros((h, w), dtype=bool)
    _reserve(occupied, palette_top, palette_left, palette_h, 2, pad=2)
    object_left_min = palette_left + 2 + random.randint(2, 5)
    if object_left_min >= w - 1:
        return None

    for holes, sprite in sorted(sprites, key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True):
        sh, sw = sprite.shape
        if sh + 2 > h or object_left_min + sw + 1 > w:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(1, h - sh - 1)
            left = random.randint(object_left_min, w - sw - 1)
            if not _can_place(occupied, top, left, sh, sw, pad=1):
                continue
            _paint_object(grid, expected, sprite, top, left, palette_colors[holes])
            _reserve(occupied, top, left, sh, sw, pad=1)
            placed = True
            break
        if not placed:
            return None

    out = main(grid)
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid palette-by-hole ARC input")
