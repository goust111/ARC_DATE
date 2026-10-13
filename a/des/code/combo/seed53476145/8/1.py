from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
HOLE_COUNTS = (1, 2, 3)

# --------------- solver ---------------


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _binary_components(mask: np.ndarray) -> list[np.ndarray]:
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
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _count_holes_mask(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg_mask = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg_mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches_edge = False
            while stack:
                rr, cc = stack.pop()
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_edge = True
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bg_mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_edge:
                holes += 1
    return holes


def _count_holes(coords: np.ndarray) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return _count_holes_mask(mask)


def _legend_mapping(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    legend_mask = np.zeros((h, w), dtype=bool)
    mapping: dict[int, int] = {}

    for coords in _binary_components(grid == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        if c0 != c1:
            continue
        run = int(r1 - r0 + 1)
        if run not in HOLE_COUNTS or len(coords) != run:
            continue
        if c0 not in (0, w - 2):
            continue
        if not (r0 <= 7 or r1 >= h - 8):
            continue

        color_col = c0 + 1
        block = grid[r0 : r1 + 1, color_col]
        if np.any(block == BG) or np.any(block == GREY):
            continue
        colors = np.unique(block)
        if len(colors) != 1:
            continue

        color = int(colors[0])
        if run in mapping:
            assert mapping[run] == color, "legend mapping must be consistent"
        mapping[run] = color
        legend_mask[coords[:, 0], coords[:, 1]] = True
        legend_mask[r0 : r1 + 1, color_col] = True

    assert mapping, "expected a corner legend"
    legend_coords = np.argwhere(legend_mask)
    r0, c0, r1, c1 = _bbox(legend_coords)
    assert c1 - c0 == 1, "legend must occupy two columns"
    assert r1 - r0 <= 7, "legend must fit in a compact corner strip"
    assert (c0 == 0 or c1 == w - 1) and (r0 == 0 or r1 == h - 1), "legend must sit in a corner"
    return mapping, legend_mask


def _find_sprites(grid: np.ndarray, legend_mask: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    sprites: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or legend_mask[r, c] or grid[r, c] == BG:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or legend_mask[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            sprites.append(np.array(coords, dtype=int))

    assert sprites, "expected at least one sprite"
    return sprites


def _draw_outer_frame(out: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    h, w = out.shape
    r0, c0, r1, c1 = bbox
    rr0, cc0 = r0 - 1, c0 - 1
    rr1, cc1 = r1 + 1, c1 + 1
    assert 0 <= rr0 and 0 <= cc0 and rr1 < h and cc1 < w, "sprite must leave room for the outside frame"
    out[rr0, cc0 : cc1 + 1] = color
    out[rr1, cc0 : cc1 + 1] = color
    out[rr0 : rr1 + 1, cc0] = color
    out[rr0 : rr1 + 1, cc1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each sprite's outer bounding-box frame using the color keyed by its hole count.

    Input structure:
    - Background is black.
    - Several 4-connected monochrome sprites appear away from the corner legend.
    - The corner legend is a two-column strip of grey run-length markers and color blocks.

    Rule:
    - Count the number of enclosed black holes in each sprite.
    - Use the legend to map that count to a frame color.
    - Draw a 1-pixel-thick rectangle one cell outside the sprite's tight bbox.
    - Keep the sprites, their holes, and the legend unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    mapping, legend_mask = _legend_mapping(g)
    sprites = _find_sprites(g, legend_mask)

    out = g.copy()
    for coords in sprites:
        holes = _count_holes(coords)
        assert holes in mapping, "every sprite hole count must appear in the legend"
        _draw_outer_frame(out, _bbox(coords), mapping[holes])
    return out


# --------------- generator ---------------


COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GRID_MIN = 24
GRID_MAX = 32
OBJECT_COUNT_CHOICES = (3, 4)
GEN_ATTEMPTS = 400


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _tight_crop(arr: np.ndarray, color: int) -> np.ndarray:
    coords = np.argwhere(arr == color)
    r0, c0, r1, c1 = _bbox(coords)
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _base_sprite(hole_count: int, color: int, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        inner_h = _randint(rng, 2, 4)
        widths = [_randint(rng, 1, 3) for _ in range(hole_count)]
        h = inner_h + 2
        w = sum(widths) + hole_count + 1
        arr = np.full((h, w), color, dtype=int)
        c = 1
        for width in widths:
            arr[1:-1, c : c + width] = BG
            c += width + 1
    else:
        inner_w = _randint(rng, 2, 4)
        heights = [_randint(rng, 1, 3) for _ in range(hole_count)]
        h = sum(heights) + hole_count + 1
        w = inner_w + 2
        arr = np.full((h, w), color, dtype=int)
        r = 1
        for height in heights:
            arr[r : r + height, 1:-1] = BG
            r += height + 1
    return arr


def _add_tabs(arr: np.ndarray, color: int, rng: np.random.Generator) -> np.ndarray:
    patch = np.pad(arr, 2, constant_values=BG)
    for _ in range(_randint(rng, 0, 2)):
        coords = np.argwhere(patch == color)
        r0, c0, r1, c1 = _bbox(coords)
        side = _randint(rng, 0, 3)
        depth = _randint(rng, 1, 2)
        if side in (0, 1):
            span = c1 - c0 + 1
            seg_len = _randint(rng, 1, min(4, span))
            start = _randint(rng, c0, c1 - seg_len + 1)
            if side == 0 and r0 - depth >= 0:
                patch[r0 - depth : r0, start : start + seg_len] = color
            elif side == 1 and r1 + depth < patch.shape[0]:
                patch[r1 + 1 : r1 + depth + 1, start : start + seg_len] = color
        else:
            span = r1 - r0 + 1
            seg_len = _randint(rng, 1, min(4, span))
            start = _randint(rng, r0, r1 - seg_len + 1)
            if side == 2 and c0 - depth >= 0:
                patch[start : start + seg_len, c0 - depth : c0] = color
            elif side == 3 and c1 + depth < patch.shape[1]:
                patch[start : start + seg_len, c1 + 1 : c1 + depth + 1] = color
    return patch


def _make_sprite(hole_count: int, color: int, rng: np.random.Generator) -> np.ndarray:
    arr = _base_sprite(hole_count, color, rng)
    arr = _add_tabs(arr, color, rng)
    arr = np.rot90(arr, _randint(rng, 0, 3))
    if rng.random() < 0.5:
        arr = np.fliplr(arr)
    if rng.random() < 0.5:
        arr = np.flipud(arr)
    arr = _tight_crop(arr, color)
    assert _count_holes_mask(arr == color) == hole_count
    return arr


def _legend_patch(mapping: dict[int, int], rng: np.random.Generator) -> np.ndarray:
    order = list(rng.permutation(HOLE_COUNTS))
    total_h = sum(order) + len(order) - 1
    patch = np.zeros((total_h, 2), dtype=int)
    r = 0
    for idx, count in enumerate(order):
        patch[r : r + count, 0] = GREY
        patch[r : r + count, 1] = mapping[count]
        r += count
        if idx + 1 < len(order):
            r += 1
    return patch


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_rect(rect: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _place_patch(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    region = canvas[top : top + patch.shape[0], left : left + patch.shape[1]]
    mask = patch != BG
    region[mask] = patch[mask]


def _build_sample(rng: np.random.Generator) -> np.ndarray | None:
    h = _randint(rng, GRID_MIN, GRID_MAX)
    w = _randint(rng, GRID_MIN, GRID_MAX)

    object_count = int(rng.choice(OBJECT_COUNT_CHOICES))
    counts = [1, 2, 3]
    if object_count == 4:
        counts.append(_randint(rng, 1, 3))
    rng.shuffle(counts)

    palette = list(rng.permutation(COLOR_POOL))
    legend_colors = palette[:3]
    sprite_colors = palette[3 : 3 + object_count]

    mapping = {count: legend_colors[count - 1] for count in HOLE_COUNTS}
    legend = _legend_patch(mapping, rng)
    lh, lw = legend.shape

    top_side = rng.random() < 0.5
    left_side = rng.random() < 0.5
    legend_top = 0 if top_side else h - lh
    legend_left = 0 if left_side else w - lw
    legend_box = (legend_top, legend_left, legend_top + lh - 1, legend_left + lw - 1)

    sprites = []
    for hole_count, color in zip(counts, sprite_colors, strict=True):
        patch = _make_sprite(hole_count, color, rng)
        sprites.append({"holes": hole_count, "color": color, "patch": patch})

    order = sorted(range(len(sprites)), key=lambda i: -int(np.count_nonzero(sprites[i]["patch"])))
    placements: dict[int, tuple[int, int]] = {}
    blocked = [_expand_rect(legend_box, 1, h, w)]

    for idx in order:
        patch = sprites[idx]["patch"]
        ph, pw = patch.shape
        if ph + 2 >= h or pw + 2 >= w:
            return None
        placed = False
        for _ in range(300):
            r0 = int(rng.integers(1, h - ph))
            c0 = int(rng.integers(1, w - pw))
            bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            safety = _expand_rect(bbox, 2, h, w)
            if any(_overlap(safety, rect) for rect in blocked):
                continue
            placements[idx] = (r0, c0)
            blocked.append(safety)
            placed = True
            break
        if not placed:
            return None

    grid = np.zeros((h, w), dtype=int)
    _place_patch(grid, legend, legend_top, legend_left)
    for idx, (r0, c0) in placements.items():
        _place_patch(grid, sprites[idx]["patch"], r0, c0)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _build_sample(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid sample")
