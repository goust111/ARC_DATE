from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
ARC_COLORS = tuple(range(10))
KEY_COLORS = tuple(c for c in ARC_COLORS if c not in (BLACK, GREY))


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        pts: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(pts)
    return comps


def _exact_rect_ring_bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    ring = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    if set(pts) != ring:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _palette_templates(g: np.ndarray) -> dict[int, np.ndarray]:
    templates: dict[int, np.ndarray] = {}
    for color in np.unique(g):
        color = int(color)
        if color in (BLACK, GREY):
            continue
        for pts in _components(g == color):
            bbox = _exact_rect_ring_bbox(pts)
            if bbox is None:
                continue
            r0, c0, r1, c1 = bbox
            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            if interior.size == 0:
                continue
            pattern = interior != BLACK
            if not bool(pattern.any()):
                continue
            if color not in templates or pattern.size > templates[color].size:
                templates[color] = pattern.copy()
    return templates


def _grey_rectangles(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    rects = []
    for pts in _components(g == GREY):
        bbox = _exact_rect_ring_bbox(pts)
        if bbox is not None:
            rects.append(bbox)
    return rects


def _tile_into(out: np.ndarray, bbox: tuple[int, int, int, int], color: int, pattern: np.ndarray) -> None:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    ph, pw = pattern.shape
    out[r0 + 1 : r1, c0 + 1 : c1] = BLACK
    for rr in range(inner_h):
        for cc in range(inner_w):
            if pattern[rr % ph, cc % pw]:
                out[r0 + 1 + rr, c0 + 1 + cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile palette-selected patterns into the interiors of grey rectangular frames.

    Input structure:
    - The background is black.
    - A corner palette contains small rectangular swatches; each swatch has a distinct non-black, non-grey border color and a binary non-black pattern on black inside it.
    - Away from the palette are one-pixel-thick grey rectangular frames, each with a black interior containing exactly one non-black, non-grey marker pixel.

    Rule:
    - Read every palette swatch as a marker-color to binary-pattern mapping, using the swatch border color as the key.
    - For each grey frame, find the single colored marker in its interior and look up the palette pattern with the same color.
    - Clear that frame interior to black, then tile the looked-up pattern from the interior top-left corner, painting every non-black pattern cell with the marker color.
    - Leave grey borders and the palette unchanged.

    Key insight: The marker pixel is a lookup key for the palette border color; the selected binary template is then repeated across the enclosed region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    templates = _palette_templates(g)
    assert templates, "expected at least one palette swatch"

    out = g.copy()
    filled_any = False
    for bbox in _grey_rectangles(g):
        r0, c0, r1, c1 = bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        markers = [
            int(v)
            for v in interior.flatten().tolist()
            if int(v) not in (BLACK, GREY)
        ]
        if not markers:
            continue
        assert len(markers) == 1, "each grey rectangle interior must contain exactly one marker"
        marker = markers[0]
        assert marker in templates, "marker color must appear as a palette border"
        _tile_into(out, bbox, marker, templates[marker])
        filled_any = True

    assert filled_any, "expected at least one grey frame to fill"
    return out


def _pattern(kind: str, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    if kind == "diag":
        for r in range(h):
            m[r, min(w - 1, round(r * (w - 1) / max(1, h - 1)))] = True
    elif kind == "anti":
        for r in range(h):
            m[r, w - 1 - min(w - 1, round(r * (w - 1) / max(1, h - 1)))] = True
    elif kind == "cross":
        m[h // 2, :] = True
        m[:, w // 2] = True
    elif kind == "ell":
        m[:, 0] = True
        m[-1, :] = True
    elif kind == "tee":
        m[0, :] = True
        m[:, w // 2] = True
    elif kind == "checker":
        for r in range(h):
            for c in range(w):
                m[r, c] = (r + c) % 2 == 0
    elif kind == "corners":
        m[0, 0] = True
        m[0, -1] = True
        m[-1, 0] = True
        m[-1, -1] = True
        m[h // 2, w // 2] = True
    elif kind == "vstripe":
        m[:, w // 2] = True
        if w >= 4:
            m[:, 0] = True
    elif kind == "hstripe":
        m[h // 2, :] = True
        if h >= 4:
            m[0, :] = True
    else:
        for r in range(h):
            m[r, (2 * r) % w] = True
            if w > 2:
                m[r, (2 * r + 1) % w] = True

    if not bool(m.any()):
        m[0, 0] = True
    if bool(m.all()):
        m[0, 0] = False
    return m


def _palette_layouts(n: int, block_h: int, block_w: int, gap: int) -> list[tuple[str, int, int]]:
    layouts = [
        ("row", block_h, n * block_w + (n - 1) * gap),
        ("col", n * block_h + (n - 1) * gap, block_w),
    ]
    if n >= 3:
        rows = (n + 1) // 2
        layouts.append(("grid", rows * block_h + (rows - 1) * gap, 2 * block_w + gap))
    return layouts


def _block_offsets(layout: str, n: int, block_h: int, block_w: int, gap: int) -> list[tuple[int, int]]:
    if layout == "row":
        return [(0, i * (block_w + gap)) for i in range(n)]
    if layout == "col":
        return [(i * (block_h + gap), 0) for i in range(n)]
    return [((i // 2) * (block_h + gap), (i % 2) * (block_w + gap)) for i in range(n)]


def _corner_origin(corner: str, h: int, w: int, ph: int, pw: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - pw
    if corner == "BL":
        return h - ph, 0
    return h - ph, w - pw


def _reserve(occ: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = occ.shape
    occ[max(0, r0 - pad) : min(h, r1 + pad + 1), max(0, c0 - pad) : min(w, c1 + pad + 1)] = True


def _free_box(occ: np.ndarray, r0: int, c0: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = occ.shape
    if r0 < 1 or c0 < 1 or r0 + height > h - 1 or c0 + width > w - 1:
        return False
    return not bool(
        occ[
            max(0, r0 - pad) : min(h, r0 + height + pad),
            max(0, c0 - pad) : min(w, c0 + width + pad),
        ].any()
    )


def _draw_palette(
    grid: np.ndarray,
    occ: np.ndarray,
    rng: random.Random,
    key_colors: list[int],
    templates: dict[int, np.ndarray],
    block_h: int,
    block_w: int,
    layout: str,
    palette_h: int,
    palette_w: int,
) -> bool:
    h, w = grid.shape
    corner = rng.choice(["TL", "TR", "BL", "BR"])
    pr0, pc0 = _corner_origin(corner, h, w, palette_h, palette_w)
    offsets = _block_offsets(layout, len(key_colors), block_h, block_w, 1)
    ink_pool = [c for c in KEY_COLORS if c not in key_colors]
    if not ink_pool:
        return False

    for color, (dr, dc) in zip(key_colors, offsets):
        r0, c0 = pr0 + dr, pc0 + dc
        r1, c1 = r0 + block_h - 1, c0 + block_w - 1
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color

        ink = rng.choice([c for c in ink_pool if c != color])
        pattern = templates[color]
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        interior[pattern] = ink

    _reserve(occ, pr0, pc0, pr0 + palette_h - 1, pc0 + palette_w - 1, pad=1)
    return True


def _draw_frame(grid: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int, marker: int, rng: random.Random) -> None:
    fh, fw = inner_h + 2, inner_w + 2
    r1, c1 = r0 + fh - 1, c0 + fw - 1
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    mr = rng.randrange(inner_h)
    mc = rng.randrange(inner_w)
    grid[r0 + 1 + mr, c0 + 1 + mc] = marker


def _expected_from_specs(grid: np.ndarray, specs: list[tuple[tuple[int, int, int, int], int, np.ndarray]]) -> np.ndarray:
    out = grid.copy()
    for bbox, color, pattern in specs:
        _tile_into(out, bbox, color, pattern)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    n = rng.randint(2, 4)
    template_h = rng.randint(2, 4)
    template_w = rng.randint(2, 4)
    block_h = template_h + 2
    block_w = template_w + 2

    possible_layouts = [
        (name, ph, pw)
        for name, ph, pw in _palette_layouts(n, block_h, block_w, 1)
        if ph <= h and pw <= w
    ]
    if not possible_layouts:
        return None
    layout, palette_h, palette_w = rng.choice(possible_layouts)

    key_colors = rng.sample(list(KEY_COLORS), n)
    kinds = ["diag", "anti", "cross", "ell", "tee", "checker", "corners", "vstripe", "hstripe", "zigzag"]
    rng.shuffle(kinds)
    templates = {color: _pattern(kinds[i % len(kinds)], template_h, template_w) for i, color in enumerate(key_colors)}

    grid = np.full((h, w), BLACK, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    if not _draw_palette(grid, occ, rng, key_colors, templates, block_h, block_w, layout, palette_h, palette_w):
        return None

    specs: list[tuple[tuple[int, int, int, int], int, np.ndarray]] = []
    frame_order = key_colors[:]
    rng.shuffle(frame_order)
    for color in frame_order:
        placed = False
        for _ in range(120):
            inner_h = rng.randint(3, 8)
            inner_w = rng.randint(3, 10)
            fh, fw = inner_h + 2, inner_w + 2
            if fh > h - 2 or fw > w - 2:
                continue
            r0 = rng.randint(1, h - fh - 1)
            c0 = rng.randint(1, w - fw - 1)
            if not _free_box(occ, r0, c0, fh, fw, pad=1):
                continue
            _draw_frame(grid, r0, c0, inner_h, inner_w, color, rng)
            _reserve(occ, r0, c0, r0 + fh - 1, c0 + fw - 1, pad=1)
            specs.append(((r0, c0, r0 + fh - 1, c0 + fw - 1), color, templates[color]))
            placed = True
            break
        if not placed:
            return None

    expected = _expected_from_specs(grid, specs)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid palette-tiling grid")
