from __future__ import annotations

import random
from collections import deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = list(range(1, 10))
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(pts)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _hole_count(mask: np.ndarray) -> int:
    bg = ~mask
    h, w = bg.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_border = False
            while q:
                rr, cc = q.popleft()
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_border = True
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                holes += 1
    return int(holes)


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict[str, np.ndarray | int]]:
    h, _ = grid.shape
    objects: list[dict[str, np.ndarray | int]] = []
    for coords in _components(grid != bg):
        if np.any(coords[:, 0] == h - 1):
            continue
        r0, c0, r1, c1 = _bbox(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in coords:
            mask[r - r0, c - c0] = True
        objects.append(
            {
                "coords": coords,
                "mask": mask,
                "r0": r0,
                "c0": c0,
            }
        )
    return objects


def _paste_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.nonzero(mask)
    grid[top + rows, left + cols] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each foreground object to the dock slot indexed by its number of enclosed holes.

    Input structure:
    - The background is Black.
    - The bottom row contains equally spaced colored marker pixels; from left to right they define dock slots 1, 2, 3, ...
    - Above the markers, several 4-connected foreground objects of one source color are scattered in the upper part of the grid.
    - Different objects may contain 0, 1, 2, or 3 enclosed Black holes inside their own bounding boxes.

    Rule:
    - Count the number of enclosed Black holes N inside each foreground object.
    - Erase the object from its original position.
    - Translate the object to the area above the (N+1)-th bottom marker, keeping its shape unchanged and aligning its bbox bottom to the row just above the marker row.
    - Recolor the translated object with the color of that destination marker.

    Key insight:
    - The destination is determined by a topological property of each object (hole count), not by its original position or size.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)

    markers = [(c, int(g[h - 1, c])) for c in range(w) if int(g[h - 1, c]) != bg]
    markers.sort()
    assert markers, "expected at least one dock marker on the bottom row"

    objects = _extract_objects(g, bg)
    out = g.copy()

    for obj in objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bg)

    for obj in objects:
        mask = obj["mask"]
        hole_count = _hole_count(mask)
        assert hole_count < len(markers), "hole count exceeds available dock slots"
        anchor_col, dock_color = markers[hole_count]
        dest_r0 = h - 1 - mask.shape[0]
        dest_c0 = int(anchor_col) - mask.shape[1] // 2
        assert dest_r0 >= 0 and 0 <= dest_c0 and dest_c0 + mask.shape[1] <= w
        _paste_mask(out, mask, dest_r0, dest_c0, int(dock_color))

    return out


def _template(lines: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in lines], dtype=bool)


ZERO_HOLE_TEMPLATES = [
    _template(
        [
            "00100",
            "01110",
            "11111",
            "01110",
            "00100",
        ]
    ),
    _template(
        [
            "11111",
            "00100",
            "00100",
            "01110",
            "00100",
        ]
    ),
    _template(
        [
            "00100",
            "01100",
            "11100",
            "01110",
            "00110",
            "00111",
            "00010",
        ]
    ),
    _template(
        [
            "111",
            "111",
            "011",
            "011",
            "111",
        ]
    ),
]

HOLE_TEMPLATES = {
    1: [
        _template(
            [
                "11111",
                "10001",
                "10001",
                "10001",
                "11111",
            ]
        ),
        _template(
            [
                "11111",
                "11111",
                "11011",
                "11111",
                "11111",
            ]
        ),
        _template(
            [
                "111",
                "101",
                "111",
            ]
        ),
        _template(
            [
                "11111",
                "11111",
                "11111",
                "11011",
                "11111",
                "11111",
                "11111",
            ]
        ),
    ],
    2: [
        _template(
            [
                "11111",
                "11011",
                "11111",
                "11011",
                "11111",
            ]
        ),
        _template(
            [
                "11111",
                "11111",
                "10101",
                "11111",
                "11111",
            ]
        ),
        _template(
            [
                "11111",
                "11011",
                "11111",
                "11111",
                "11111",
                "11011",
                "11111",
            ]
        ),
        _template(
            [
                "11111",
                "11011",
                "11111",
                "11101",
                "11111",
            ]
        ),
    ],
    3: [
        _template(
            [
                "11111",
                "11011",
                "11111",
                "10101",
                "11111",
            ]
        ),
        _template(
            [
                "11111",
                "11011",
                "11111",
                "11011",
                "11111",
                "11011",
                "11111",
            ]
        ),
        _template(
            [
                "11111",
                "11011",
                "11111",
                "11101",
                "11111",
                "11011",
                "11111",
            ]
        ),
        _template(
            [
                "11111",
                "10111",
                "11111",
                "11101",
                "11111",
                "10111",
                "11111",
            ]
        ),
    ],
}


def _flip_mask(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _add_caps(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for side in ("top", "bottom"):
        if random.random() < 0.55 and out.shape[0] < 8:
            width = 3 if out.shape[1] == 5 and random.random() < 0.5 else 1
            c0 = out.shape[1] // 2 - width // 2
            if side == "top":
                grown = np.zeros((out.shape[0] + 1, out.shape[1]), dtype=bool)
                grown[1:, :] = out
                grown[0, c0 : c0 + width] = True
            else:
                grown = np.zeros((out.shape[0] + 1, out.shape[1]), dtype=bool)
                grown[:-1, :] = out
                grown[-1, c0 : c0 + width] = True
            out = grown
    return out


def _make_object(hole_count: int) -> np.ndarray:
    if hole_count == 0:
        mask = random.choice(ZERO_HOLE_TEMPLATES).copy()
    else:
        mask = random.choice(HOLE_TEMPLATES[hole_count]).copy()
    mask = _flip_mask(mask)
    mask = _add_caps(mask)
    mask = _trim_mask(mask)
    assert mask.shape[1] in (3, 5), "object width must fit inside dock spacing"
    assert _hole_count(mask) == hole_count
    return mask


def _pick_marker_layout() -> tuple[int, list[int]]:
    step = random.choice([6, 7])
    left = random.randint(2, 4)
    marker_cols = [left + i * step for i in range(4)]
    max_right = 29 - marker_cols[-1]
    if max_right < 2:
        raise ValueError("dock layout would exceed width 30")
    right = random.randint(2, max_right)
    width = marker_cols[-1] + right + 1
    return width, marker_cols


def _can_place(occupied: np.ndarray, top: int, left: int, mask: np.ndarray, pad: int = 1) -> bool:
    h, w = mask.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occupied.shape[0], top + h + pad)
    c1 = min(occupied.shape[1], left + w + pad)
    return not occupied[r0:r1, c0:c1].any()


def _mark_occupied(occupied: np.ndarray, top: int, left: int, mask: np.ndarray, pad: int = 1) -> None:
    h, w = mask.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occupied.shape[0], top + h + pad)
    c1 = min(occupied.shape[1], left + w + pad)
    occupied[r0:r1, c0:c1] = True


def _build_expected(
    grid: np.ndarray,
    placements: list[tuple[np.ndarray, int, int, int]],
    marker_cols: list[int],
    marker_colors: list[int],
    bg: int,
) -> np.ndarray:
    out = grid.copy()
    for mask, top, left, _hole_count_value in placements:
        rows, cols = np.nonzero(mask)
        out[top + rows, left + cols] = bg
    for mask, _top, _left, hole_count_value in placements:
        dest_r0 = out.shape[0] - 1 - mask.shape[0]
        dest_c0 = marker_cols[hole_count_value] - mask.shape[1] // 2
        assert dest_r0 >= 0 and 0 <= dest_c0 and dest_c0 + mask.shape[1] <= out.shape[1]
        _paste_mask(out, mask, dest_r0, dest_c0, marker_colors[hole_count_value])
    return out


def _gen_attempt() -> np.ndarray | None:
    object_color, *dock_colors = random.sample(PALETTE, 5)
    hole_counts = random.sample([0, 1, 2, 3], k=random.randint(2, 4))
    objects = [(hc, _make_object(hc)) for hc in hole_counts]
    max_h = max(mask.shape[0] for _, mask in objects)

    width, marker_cols = _pick_marker_layout()
    min_height = max(18, 2 * max_h + 3)
    if min_height > 30:
        return None
    height = random.randint(min_height, 30)
    top_band = height // 2

    grid = np.zeros((height, width), dtype=int)
    for c, color in zip(marker_cols, dock_colors):
        grid[height - 1, c] = int(color)

    occupied = np.zeros((height, width), dtype=bool)
    occupied[height - 1, :] = grid[height - 1, :] != 0

    placements: list[tuple[np.ndarray, int, int, int]] = []
    for hole_count_value, mask in sorted(objects, key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True):
        h, w = mask.shape
        max_top = top_band - h - 1
        if max_top < 0:
            return None
        placed = False
        for _ in range(200):
            top = random.randint(0, max_top)
            left = random.randint(0, width - w)
            if not _can_place(occupied, top, left, mask):
                continue
            _paste_mask(grid, mask, top, left, object_color)
            _mark_occupied(occupied, top, left, mask)
            placements.append((mask, top, left, hole_count_value))
            placed = True
            break
        if not placed:
            return None

    expected = _build_expected(grid, placements, marker_cols, dock_colors, 0)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dock-by-hole sample")
