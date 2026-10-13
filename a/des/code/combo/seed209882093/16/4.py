from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
FRAME_COLOR = 5
BLOCK_SIZE = 4
FRAME_INNER = 8
FRAME_OUTER = FRAME_INNER + 2
MAX_GENERATION_ATTEMPTS = 300

_CORNERS = ("tl", "tr", "bl", "br")
_MISSING_CELL = {
    "tl": (0, 0),
    "tr": (0, BLOCK_SIZE - 1),
    "bl": (BLOCK_SIZE - 1, 0),
    "br": (BLOCK_SIZE - 1, BLOCK_SIZE - 1),
}
_QUADRANT_OFFSET = {
    "tl": (0, 0),
    "tr": (0, BLOCK_SIZE),
    "bl": (BLOCK_SIZE, 0),
    "br": (BLOCK_SIZE, BLOCK_SIZE),
}


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                }
            )
    return comps


def _is_frame_component(grid: np.ndarray, bg: int, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    if r1 - r0 + 1 != FRAME_OUTER or c1 - c0 + 1 != FRAME_OUTER:
        return False
    if len(comp["coords"]) != FRAME_OUTER * FRAME_OUTER - FRAME_INNER * FRAME_INNER:
        return False

    color = int(comp["color"])
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros_like(sub, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool((sub[border] == color).all() and (sub[1:-1, 1:-1] == bg).all())


def _find_frame(grid: np.ndarray, bg: int, comps: list[dict]) -> dict:
    candidates = [comp for comp in comps if _is_frame_component(grid, bg, comp)]
    assert candidates, "expected one 10x10 hollow frame"
    candidates.sort(key=lambda comp: (int(comp["color"]) != FRAME_COLOR, comp["bbox"]))
    return candidates[0]


def _missing_corner(grid: np.ndarray, bg: int, comp: dict) -> str | None:
    r0, c0, r1, c1 = comp["bbox"]
    if r1 - r0 + 1 != BLOCK_SIZE or c1 - c0 + 1 != BLOCK_SIZE:
        return None
    if len(comp["coords"]) != BLOCK_SIZE * BLOCK_SIZE - 1:
        return None

    color = int(comp["color"])
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if int(np.count_nonzero(sub == color)) != BLOCK_SIZE * BLOCK_SIZE - 1:
        return None
    if int(np.count_nonzero(sub == bg)) != 1:
        return None

    for corner, (rr, cc) in _MISSING_CELL.items():
        if int(sub[rr, cc]) != bg:
            continue
        probe = sub.copy()
        probe[rr, cc] = color
        if bool((probe == color).all()):
            return corner
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move 4x4 blocks into the matching quadrant of the 8x8 target frame.

    The missing corner of each source block encodes the destination quadrant.
    Multiple blocks mapped to the same quadrant are painted in row-major input
    order, so later blocks overwrite earlier ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    comps = _components_by_color(g, bg)
    frame = _find_frame(g, bg, comps)
    fr0, fc0, _fr1, _fc1 = frame["bbox"]

    out = g.copy()
    buckets: dict[str, list[tuple[int, int, int]]] = {corner: [] for corner in _CORNERS}

    for comp in comps:
        if comp is frame:
            continue
        corner = _missing_corner(g, bg, comp)
        if corner is None:
            continue
        r0, c0, _r1, _c1 = comp["bbox"]
        buckets[corner].append((r0, c0, int(comp["color"])))
        out[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = bg

    for corner in _CORNERS:
        dr, dc = _QUADRANT_OFFSET[corner]
        rr = fr0 + 1 + dr
        cc = fc0 + 1 + dc
        for _r0, _c0, color in sorted(buckets[corner], key=lambda item: (item[0], item[1])):
            out[rr : rr + BLOCK_SIZE, cc : cc + BLOCK_SIZE] = color

    return out


def _make_block(color: int, corner: str) -> np.ndarray:
    block = np.full((BLOCK_SIZE, BLOCK_SIZE), int(color), dtype=int)
    rr, cc = _MISSING_CELL[corner]
    block[rr, cc] = BG
    return block


def _can_place_rect(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    height, width = occupied.shape
    if r0 < 1 or c0 < 1 or r0 + h >= height or c0 + w >= width:
        return False
    pr0 = max(0, r0 - pad)
    pc0 = max(0, c0 - pad)
    pr1 = min(height, r0 + h + pad)
    pc1 = min(width, c0 + w + pad)
    return not bool(occupied[pr0:pr1, pc0:pc1].any())


def _mark_rect(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    height, width = occupied.shape
    pr0 = max(0, r0 - pad)
    pc0 = max(0, c0 - pad)
    pr1 = min(height, r0 + h + pad)
    pc1 = min(width, c0 + w + pad)
    occupied[pr0:pr1, pc0:pc1] = True


def _place_rect(occupied: np.ndarray, h: int, w: int, pad: int = 1) -> tuple[int, int] | None:
    height, width = occupied.shape
    candidates = [(r, c) for r in range(1, height - h) for c in range(1, width - w)]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        if _can_place_rect(occupied, r0, c0, h, w, pad=pad):
            _mark_rect(occupied, r0, c0, h, w, pad=pad)
            return r0, c0
    return None


def _sample_corners(n_blocks: int, allow_duplicates: bool) -> list[str]:
    if not allow_duplicates:
        return random.sample(list(_CORNERS), n_blocks)

    while True:
        corners = [random.choice(_CORNERS) for _ in range(n_blocks)]
        counts = Counter(corners)
        if len(counts) >= 2 and max(counts.values()) >= 2:
            return corners


def _expected_from_layout(
    grid: np.ndarray,
    frame_pos: tuple[int, int],
    pieces: list[dict],
) -> np.ndarray:
    out = grid.copy()
    fr0, fc0 = frame_pos
    buckets: dict[str, list[dict]] = {corner: [] for corner in _CORNERS}

    for piece in pieces:
        r0 = int(piece["r0"])
        c0 = int(piece["c0"])
        out[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = BG
        buckets[str(piece["corner"])].append(piece)

    for corner in _CORNERS:
        dr, dc = _QUADRANT_OFFSET[corner]
        rr = fr0 + 1 + dr
        cc = fc0 + 1 + dc
        for piece in sorted(buckets[corner], key=lambda item: (int(item["r0"]), int(item["c0"]))):
            out[rr : rr + BLOCK_SIZE, cc : cc + BLOCK_SIZE] = int(piece["color"])

    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    duplicate_branch = random.random() < 0.55
    if duplicate_branch:
        n_blocks = random.randint(5, 8)
        corners = _sample_corners(n_blocks, allow_duplicates=True)
    else:
        n_blocks = random.randint(2, 4)
        corners = _sample_corners(n_blocks, allow_duplicates=False)

    min_side = 18 if n_blocks <= 4 else 24
    height = random.randint(min_side, min_side + 8)
    width = random.randint(min_side, min_side + 8)

    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    frame_pos = _place_rect(occupied, FRAME_OUTER, FRAME_OUTER, pad=1)
    assert frame_pos is not None
    fr0, fc0 = frame_pos
    grid[fr0 : fr0 + FRAME_OUTER, fc0 : fc0 + FRAME_OUTER] = FRAME_COLOR
    grid[fr0 + 1 : fr0 + 1 + FRAME_INNER, fc0 + 1 : fc0 + 1 + FRAME_INNER] = BG

    palette = [color for color in range(1, 10) if color != FRAME_COLOR]
    colors = random.sample(palette, n_blocks)
    pieces: list[dict] = []

    for color, corner in zip(colors, corners):
        pos = _place_rect(occupied, BLOCK_SIZE, BLOCK_SIZE, pad=1)
        if pos is None:
            raise RuntimeError("placement failed")
        r0, c0 = pos
        grid[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = _make_block(color, corner)
        pieces.append({"r0": r0, "c0": c0, "color": color, "corner": corner})

    expected = _expected_from_layout(grid, frame_pos, pieces)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid, expected = _generate_case()
        except RuntimeError:
            continue
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
