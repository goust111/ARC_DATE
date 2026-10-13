from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


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
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_palette_component(g: np.ndarray, coords: list[tuple[int, int]], bg: int) -> bool:
    h, w = g.shape
    r0, c0, r1, c1 = _bbox(coords)
    if c0 != c1:
        return False
    if c0 not in (0, w - 1):
        return False
    if not (r0 == 0 or r1 == h - 1):
        return False
    if len(coords) != r1 - r0 + 1:
        return False
    rows = sorted(r for r, _ in coords)
    if rows != list(range(r0, r1 + 1)):
        return False
    colors = [int(g[r, c]) for r, c in coords]
    if any(color == bg for color in colors):
        return False
    return len(set(colors)) == len(colors)


def _is_rectangular_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False
    pts = set(coords)
    if len(pts) != 2 * (height + width) - 4:
        return False
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_perimeter = r in (r0, r1) or c in (c0, c1)
            if on_perimeter != ((r, c) in pts):
                return False
    return True


def _find_palette(g: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    mask = g != bg
    candidates = [comp for comp in _components(mask) if _is_palette_component(g, comp, bg)]
    assert candidates, "expected a corner palette column"
    candidates.sort(key=lambda comp: (-len(comp), _bbox(comp)))
    coords = sorted(candidates[0], key=lambda rc: rc[0])
    palette = [int(g[r, c]) for r, c in coords]
    palette_mask = np.zeros_like(g, dtype=bool)
    for r, c in coords:
        palette_mask[r, c] = True
    return palette, palette_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray rectangular border by counting its interior markers and indexing a corner palette.

    Input structure:
    - The background is black.
    - Several closed 1-pixel-wide gray rectangular borders appear in the grid, and they do not touch each other.
    - Inside each rectangle are one or more colored marker pixels, while a corner of the grid contains a vertical palette column of distinct colors.

    Rule:
    - For each gray rectangular border, count how many non-background marker pixels lie strictly inside it.
    - Read the palette from top to bottom; the Nth palette color is the target color for the rectangle whose interior contains N markers.
    - Replace the full border with that target color, clear all interior markers back to the background, and leave the palette unchanged.

    Key insight:
    - The marker colors and positions are distractions; only the marker count matters, and that count is used as a lookup index into the shared palette.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(Counter(map(int, g.reshape(-1))).most_common(1)[0][0])

    palette, palette_mask = _find_palette(g, bg)
    assert palette, "palette must be non-empty"

    out = g.copy()
    frame_components: list[list[tuple[int, int]]] = []
    for color in sorted({int(v) for v in np.unique(g)}):
        if color == bg:
            continue
        color_mask = (g == color) & (~palette_mask)
        for comp in _components(color_mask):
            if _is_rectangular_frame(comp):
                frame_components.append(comp)

    assert frame_components, "expected at least one rectangular frame"
    frame_components.sort(key=_bbox)

    for frame in frame_components:
        r0, c0, r1, c1 = _bbox(frame)
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_count = int(np.count_nonzero(inner != bg))
        assert 1 <= marker_count <= len(palette), "marker count must index the palette"
        target = palette[marker_count - 1]

        for r, c in frame:
            out[r, c] = target
        out[r0 + 1 : r1, c0 + 1 : c1] = bg

    return out


# --------------- generator (codex) ---------------

CANVAS_SIZE = 30
BG = 0
FRAME_COLOR = 5
FRAME_COUNT_RANGE = (2, 5)
INTERIOR_HEIGHT_RANGE = range(2, 7)
INTERIOR_WIDTH_RANGE = range(2, 8)
PLACEMENT_TRIES = 120
MARKER_PLACEMENT_TRIES = 40
GENERATION_RETRIES = 400
PADDING_CHOICES = (1, 2, 3, 4)
CORNERS = ("tl", "tr", "bl", "br")


def _boxes_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _ring_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _palette_cells(corner: str, length: int) -> list[tuple[int, int]]:
    if corner == "tl":
        return [(r, 0) for r in range(length)]
    if corner == "tr":
        return [(r, CANVAS_SIZE - 1) for r in range(length)]
    if corner == "bl":
        start = CANVAS_SIZE - length
        return [(r, 0) for r in range(start, CANVAS_SIZE)]
    start = CANVAS_SIZE - length
    return [(r, CANVAS_SIZE - 1) for r in range(start, CANVAS_SIZE)]


def _spaced_positions(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(height) for c in range(width)]
    for _ in range(MARKER_PLACEMENT_TRIES):
        random.shuffle(cells)
        chosen: list[tuple[int, int]] = []
        for r, c in cells:
            if all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in chosen):
                chosen.append((r, c))
                if len(chosen) == count:
                    return chosen
    return None


def _frame_dims(marker_count: int) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for ih in INTERIOR_HEIGHT_RANGE:
        for iw in INTERIOR_WIDTH_RANGE:
            if ih * iw < marker_count:
                continue
            if ((ih + 1) // 2) * ((iw + 1) // 2) < marker_count:
                continue
            dims.append((ih, iw))
    return dims


def _crop_keep_corner(inp: np.ndarray, out: np.ndarray, corner: str) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != BG)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)

    if corner == "tl":
        top = 0
        left = 0
        bottom = min(CANVAS_SIZE, int(r1) + 1 + random.randint(1, max(1, min(4, CANVAS_SIZE - int(r1) - 1))))
        right = min(CANVAS_SIZE, int(c1) + 1 + random.randint(1, max(1, min(4, CANVAS_SIZE - int(c1) - 1))))
    elif corner == "tr":
        top = 0
        bottom = min(CANVAS_SIZE, int(r1) + 1 + random.randint(1, max(1, min(4, CANVAS_SIZE - int(r1) - 1))))
        right = CANVAS_SIZE
        left = max(0, int(c0) - random.randint(1, max(1, min(4, int(c0)))))
    elif corner == "bl":
        top = max(0, int(r0) - random.randint(1, max(1, min(4, int(r0)))))
        left = 0
        bottom = CANVAS_SIZE
        right = min(CANVAS_SIZE, int(c1) + 1 + random.randint(1, max(1, min(4, CANVAS_SIZE - int(c1) - 1))))
    else:
        top = max(0, int(r0) - random.randint(1, max(1, min(4, int(r0)))))
        left = max(0, int(c0) - random.randint(1, max(1, min(4, int(c0)))))
        bottom = CANVAS_SIZE
        right = CANVAS_SIZE

    cropped_inp = inp[top:bottom, left:right]
    cropped_out = out[top:bottom, left:right]
    assert max(cropped_inp.shape) <= 30 and max(cropped_out.shape) <= 30
    return cropped_inp, cropped_out


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    corner = random.choice(CORNERS)

    palette_colors = random.sample([c for c in range(10) if c not in (BG, FRAME_COLOR)], frame_count)
    counts = list(range(1, frame_count + 1))
    random.shuffle(counts)

    inp = np.full((CANVAS_SIZE, CANVAS_SIZE), BG, dtype=int)
    out = inp.copy()

    palette = _palette_cells(corner, frame_count)
    for (r, c), color in zip(palette, palette_colors, strict=True):
        inp[r, c] = color
        out[r, c] = color

    occupied_boxes = [_bbox(palette)]
    marker_pool = [c for c in range(10) if c not in (BG, FRAME_COLOR)]

    for marker_count in counts:
        dims = _frame_dims(marker_count)
        if not dims:
            return None

        placed = False
        random.shuffle(dims)
        for _ in range(PLACEMENT_TRIES):
            ih, iw = random.choice(dims)
            total_h = ih + 2
            total_w = iw + 2
            max_r0 = CANVAS_SIZE - total_h - 1
            max_c0 = CANVAS_SIZE - total_w - 1
            if max_r0 < 1 or max_c0 < 1:
                return None
            r0 = random.randint(1, max_r0)
            c0 = random.randint(1, max_c0)
            box = (r0, c0, r0 + total_h - 1, c0 + total_w - 1)
            if any(_boxes_conflict(box, other, gap=1) for other in occupied_boxes):
                continue

            positions = _spaced_positions(ih, iw, marker_count)
            if positions is None:
                continue

            target = palette_colors[marker_count - 1]
            for r, c in _ring_cells(*box):
                inp[r, c] = FRAME_COLOR
                out[r, c] = target

            for rr in range(r0 + 1, box[2]):
                for cc in range(c0 + 1, box[3]):
                    out[rr, cc] = BG

            for pr, pc in positions:
                rr = r0 + 1 + pr
                cc = c0 + 1 + pc
                inp[rr, cc] = random.choice(marker_pool)

            occupied_boxes.append(box)
            placed = True
            break

        if not placed:
            return None

    inp, out = _crop_keep_corner(inp, out, corner)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
