from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
RING_SIZE = 5
INNER_SIZE = 3
RECT_SIZE = 3

_RING_BORDER_MASK = np.ones((RING_SIZE, RING_SIZE), dtype=bool)
_RING_BORDER_MASK[1:-1, 1:-1] = False
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bbox_too_close(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0
) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _same_color_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue

            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for dr, dc in _NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )

    return comps


def _find_rings(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    rings: list[dict] = []
    for r in range(h - RING_SIZE + 1):
        for c in range(w - RING_SIZE + 1):
            patch = grid[r : r + RING_SIZE, c : c + RING_SIZE]
            border_vals = patch[_RING_BORDER_MASK]
            border_color = int(border_vals[0])
            if border_color == BG or not np.all(border_vals == border_color):
                continue

            marker = int(patch[2, 2])
            if marker in (BG, border_color):
                continue

            bbox = (r, c, r + RING_SIZE - 1, c + RING_SIZE - 1)
            assert all(
                not _bbox_too_close(bbox, other["bbox"], gap=0) for other in rings
            ), "expected non-overlapping 5x5 rings"
            rings.append(
                {
                    "bbox": bbox,
                    "border_color": border_color,
                    "marker_color": marker,
                    "pattern": patch[1:-1, 1:-1].copy(),
                }
            )

    assert rings, "expected at least one 5x5 ring object"
    return rings


def _find_rectangles(grid: np.ndarray, rings: list[dict]) -> dict[int, tuple[int, int]]:
    masked = np.array(grid, copy=True)
    for ring in rings:
        r0, c0, r1, c1 = ring["bbox"]
        masked[r0 : r1 + 1, c0 : c1 + 1] = BG

    targets: dict[int, tuple[int, int]] = {}
    for comp in _same_color_components(masked):
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != RECT_SIZE or w != RECT_SIZE or int(comp["area"]) != RECT_SIZE * RECT_SIZE:
            continue
        if not np.all(masked[r0 : r1 + 1, c0 : c1 + 1] == int(comp["color"])):
            continue
        color = int(comp["color"])
        assert color not in targets, "expected a unique 3x3 rectangle for each color"
        targets[color] = (r0, c0)

    assert targets, "expected at least one standalone 3x3 rectangle"
    return targets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each ring's inner 3x3 motif into the solid 3x3 rectangle keyed by the center marker color.

    Input structure
    - Background is black.
    - Each source object is a 5x5 monochrome hollow square whose 3x3 interior contains a pattern.
    - The center of that 3x3 interior is a non-black marker pixel whose color points to one solid 3x3 rectangle.

    Rule
    - Keep every 5x5 ring unchanged.
    - For each ring, overlay its 3x3 interior onto the rectangle whose fill color equals the ring center.
    - Black cells in the motif are transparent; non-black cells overwrite the rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= RING_SIZE and w >= RING_SIZE, "grid is too small for a 5x5 ring"

    rings = _find_rings(g)
    targets = _find_rectangles(g, rings)

    out = g.copy()
    for ring in rings:
        marker = int(ring["marker_color"])
        assert marker in targets, "each ring marker must match one 3x3 rectangle"
        tr, tc = targets[marker]
        pattern = np.asarray(ring["pattern"], dtype=int)
        for rr in range(INNER_SIZE):
            for cc in range(INNER_SIZE):
                color = int(pattern[rr, cc])
                if color != BG:
                    out[tr + rr, tc + cc] = color

    return out


# --------------- generator ---------------

_GEN_RETRY_LIMIT = 400
_GEN_GRID_MIN = 18
_GEN_GRID_MAX = 30
_GEN_PAIR_MIN = 2
_GEN_PAIR_MAX = 4
_GEN_OBJECT_GAP = 1
_GEN_COLORS = tuple(range(1, 10))


def _random_location(
    shape: tuple[int, int],
    box_h: int,
    box_w: int,
    existing: list[tuple[int, int, int, int]],
    gap: int,
) -> tuple[int, int] | None:
    h, w = shape
    candidates: list[tuple[int, int]] = []
    for r in range(h - box_h + 1):
        for c in range(w - box_w + 1):
            bbox = (r, c, r + box_h - 1, c + box_w - 1)
            if all(not _bbox_too_close(bbox, other, gap=gap) for other in existing):
                candidates.append((r, c))
    return random.choice(candidates) if candidates else None


def _sample_pattern(marker_color: int, border_color: int) -> np.ndarray:
    patch = np.zeros((INNER_SIZE, INNER_SIZE), dtype=int)
    patch[1, 1] = marker_color

    non_center = [(r, c) for r in range(INNER_SIZE) for c in range(INNER_SIZE) if (r, c) != (1, 1)]
    extra_count = random.randint(1, len(non_center))
    chosen = random.sample(non_center, extra_count)

    accent_pool = [c for c in _GEN_COLORS if c not in {marker_color, border_color}]
    accent_color = random.choice(accent_pool)
    accent_cell = random.choice(chosen)
    patch[accent_cell] = accent_color

    fill_pool = [c for c in _GEN_COLORS if c != border_color]
    for cell in chosen:
        if cell == accent_cell:
            continue
        patch[cell] = random.choice(fill_pool)

    return patch


def _draw_ring(grid: np.ndarray, top: int, left: int, border_color: int, pattern: np.ndarray) -> None:
    grid[top : top + RING_SIZE, left : left + RING_SIZE][_RING_BORDER_MASK] = border_color
    grid[top + 1 : top + 1 + INNER_SIZE, left + 1 : left + 1 + INNER_SIZE] = pattern


def _draw_rectangle(grid: np.ndarray, top: int, left: int, color: int) -> None:
    grid[top : top + RECT_SIZE, left : left + RECT_SIZE] = color


def _apply_transfer(grid: np.ndarray, top: int, left: int, pattern: np.ndarray) -> None:
    for r in range(INNER_SIZE):
        for c in range(INNER_SIZE):
            color = int(pattern[r, c])
            if color != BG:
                grid[top + r, left + c] = color


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(_GEN_PAIR_MIN, _GEN_PAIR_MAX)
    height = random.randint(_GEN_GRID_MIN, _GEN_GRID_MAX)
    width = random.randint(_GEN_GRID_MIN, _GEN_GRID_MAX)

    marker_colors = random.sample(_GEN_COLORS, pair_count)
    border_pool = [c for c in _GEN_COLORS if c not in marker_colors]
    border_colors = random.sample(border_pool, pair_count)

    existing: list[tuple[int, int, int, int]] = []
    ring_positions: list[tuple[int, int]] = []
    rect_positions: list[tuple[int, int]] = []

    for _ in range(pair_count):
        pos = _random_location((height, width), RING_SIZE, RING_SIZE, existing, _GEN_OBJECT_GAP)
        if pos is None:
            return None
        r, c = pos
        existing.append((r, c, r + RING_SIZE - 1, c + RING_SIZE - 1))
        ring_positions.append(pos)

    for _ in range(pair_count):
        pos = _random_location((height, width), RECT_SIZE, RECT_SIZE, existing, _GEN_OBJECT_GAP)
        if pos is None:
            return None
        r, c = pos
        existing.append((r, c, r + RECT_SIZE - 1, c + RECT_SIZE - 1))
        rect_positions.append(pos)

    grid = np.zeros((height, width), dtype=int)
    expected = np.zeros((height, width), dtype=int)

    patterns = [_sample_pattern(marker, border) for marker, border in zip(marker_colors, border_colors)]
    if all(np.all(pattern != BG) for pattern in patterns):
        pattern = patterns[0].copy()
        removable = [(r, c) for r in range(INNER_SIZE) for c in range(INNER_SIZE) if (r, c) != (1, 1)]
        for r, c in removable:
            if int(pattern[r, c]) == int(pattern[1, 1]):
                pattern[r, c] = BG
                break
        else:
            r, c = removable[0]
            pattern[r, c] = BG
        patterns[0] = pattern

    for (rr, rc), marker in zip(rect_positions, marker_colors):
        _draw_rectangle(grid, rr, rc, marker)
        _draw_rectangle(expected, rr, rc, marker)

    for (sr, sc), border, pattern in zip(ring_positions, border_colors, patterns):
        _draw_ring(grid, sr, sc, border, pattern)
        _draw_ring(expected, sr, sc, border, pattern)

    for (rr, rc), pattern in zip(rect_positions, patterns):
        _apply_transfer(expected, rr, rc, pattern)

    try:
        produced = main(grid)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid ring-to-rectangle transfer input")
