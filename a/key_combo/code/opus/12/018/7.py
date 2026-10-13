from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
SEP = 9
FOUR_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NOTCH_TO_TRANSFORM = {
    "top": lambda x: x.copy(),
    "right": np.fliplr,
    "bottom": np.flipud,
    "left": lambda x: np.flipud(np.fliplr(x)),
}


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _same_color_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in FOUR_DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": int(len(cells)),
                }
            )
    return components


def _is_color_connected(patch: np.ndarray, color: int) -> bool:
    coords = np.argwhere(patch == int(color))
    if coords.size == 0:
        return False
    h, w = patch.shape
    seen = np.zeros((h, w), dtype=bool)
    start = tuple(map(int, coords[0]))
    q: deque[tuple[int, int]] = deque([start])
    seen[start] = True
    count = 0
    while q:
        r, c = q.popleft()
        count += 1
        for dr, dc in FOUR_DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(patch[nr, nc]) == color:
                seen[nr, nc] = True
                q.append((nr, nc))
    return count == int(coords.shape[0])


def _tile_color(patch: np.ndarray) -> int:
    colors = sorted({int(v) for v in patch.flatten() if int(v) != BG})
    assert len(colors) == 1, "each tile should use exactly one foreground color"
    return int(colors[0])


def _tile_notch_side(patch: np.ndarray) -> str:
    h, w = patch.shape
    top_zeros = [c for c in range(w) if int(patch[0, c]) == BG]
    bottom_zeros = [c for c in range(w) if int(patch[h - 1, c]) == BG]
    left_zeros = [r for r in range(h) if int(patch[r, 0]) == BG]
    right_zeros = [r for r in range(h) if int(patch[r, w - 1]) == BG]
    side_zeros = {
        "top": top_zeros,
        "right": right_zeros,
        "bottom": bottom_zeros,
        "left": left_zeros,
    }

    candidates = []
    for side, idxs in side_zeros.items():
        if len(idxs) != 1:
            continue
        idx = idxs[0]
        limit = w if side in {"top", "bottom"} else h
        if idx in {0, limit - 1}:
            continue
        candidates.append(side)

    assert len(candidates) == 1, "expected exactly one non-corner notch side"
    side = candidates[0]
    for other, idxs in side_zeros.items():
        if other != side:
            assert len(idxs) == 0, "only the notch side may contain background on the border"
    return side


def _transform_tile(patch: np.ndarray, notch_side: str) -> np.ndarray:
    return np.asarray(NOTCH_TO_TRANSFORM[notch_side](patch), dtype=int)


def _center_key(component: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = component["bbox"]
    return int(r0 + r1), int(c0 + c1)


def _ordered_tiles(components: list[dict]) -> list[dict]:
    assert len(components) == 4, "expected exactly four tiles"
    by_row = sorted(components, key=_center_key)
    top = sorted(by_row[:2], key=lambda comp: _center_key(comp)[1])
    bottom = sorted(by_row[2:], key=lambda comp: _center_key(comp)[1])
    assert max(_center_key(comp)[0] for comp in top) < min(_center_key(comp)[0] for comp in bottom), (
        "tiles must form a clear top row and bottom row"
    )
    return top + bottom


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each notched tile according to its gap side, then stitch the four results into a maroon-separated 2x2 layout.

    Input structure:
    - The grid is mostly black background.
    - Exactly four separated monochrome rectangular tiles appear on the background.
    - Every tile has the same bbox size, contains only one non-black color, and has exactly one one-pixel border notch on a distinct side.

    Rule:
    - Detect the four tiles and order them by spatial position as top-left, top-right, bottom-left, bottom-right.
    - Read each tile's notch side: top keeps the tile unchanged, right means horizontal flip, bottom means vertical flip, left means a 180-degree rotation.
    - Create a (2h+1) x (2w+1) output filled with maroon, then place the transformed tiles in the four quadrants with a one-cell maroon separator row and column.
    - Tile pixels overwrite the maroon canvas, including black notch and interior gap cells.

    Key insight:
    - The missing border pixel is a self-contained geometric code: its side uniquely determines which reflection to apply before the final 2x2 stitch.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _same_color_components(g, bg=BG)
    ordered = _ordered_tiles(components)

    bboxes = [tuple(map(int, comp["bbox"])) for comp in ordered]
    heights = [bbox[2] - bbox[0] + 1 for bbox in bboxes]
    widths = [bbox[3] - bbox[1] + 1 for bbox in bboxes]
    assert len(set(heights)) == 1 and len(set(widths)) == 1, "all tiles must share one rectangle size"
    tile_h = heights[0]
    tile_w = widths[0]

    transformed: list[np.ndarray] = []
    for comp in ordered:
        patch = _crop_bbox(g, comp["bbox"])
        color = _tile_color(patch)
        assert _is_color_connected(patch, color), "tile color cells must stay 4-connected"
        side = _tile_notch_side(patch)
        transformed.append(_transform_tile(patch, side))

    out = np.full((2 * tile_h + 1, 2 * tile_w + 1), SEP, dtype=int)
    out[:tile_h, :tile_w] = transformed[0]
    out[:tile_h, tile_w + 1 :] = transformed[1]
    out[tile_h + 1 :, :tile_w] = transformed[2]
    out[tile_h + 1 :, tile_w + 1 :] = transformed[3]
    return out


TILE_H_CHOICES = [4, 5]
TILE_W_CHOICES = [4, 5]
GAP_CHOICES = [2, 3, 4, 5]
MARGIN_CHOICES = [0, 1, 2, 3, 4]
NON_SPECIAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 8]
MAX_GENERATION_ATTEMPTS = 400


def _make_tile_patch(h: int, w: int, color: int, notch_side: str) -> np.ndarray | None:
    patch = np.full((h, w), int(color), dtype=int)

    if notch_side == "top":
        patch[0, random.randint(1, w - 2)] = BG
    elif notch_side == "bottom":
        patch[h - 1, random.randint(1, w - 2)] = BG
    elif notch_side == "left":
        patch[random.randint(1, h - 2), 0] = BG
    else:
        patch[random.randint(1, h - 2), w - 1] = BG

    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    hole_count = random.randint(1, min(4, len(interior) - 1))
    for r, c in random.sample(interior, hole_count):
        patch[r, c] = BG

    if np.array_equal(patch, np.fliplr(patch)):
        return None
    if np.array_equal(patch, np.flipud(patch)):
        return None
    if np.array_equal(patch, np.rot90(patch, 2)):
        return None
    if not _is_color_connected(patch, color):
        return None
    try:
        if _tile_notch_side(patch) != notch_side:
            return None
    except AssertionError:
        return None
    return patch


def _generate_tile_set(h: int, w: int, directions: list[str]) -> list[np.ndarray] | None:
    colors = random.sample(NON_SPECIAL_COLORS, 4)
    patches: list[np.ndarray] = []
    for color, direction in zip(colors, directions):
        patch = None
        for _ in range(80):
            patch = _make_tile_patch(h, w, color, direction)
            if patch is not None:
                break
        if patch is None:
            return None
        patches.append(patch)
    return patches


def _semantic_check(inp: np.ndarray) -> None:
    comps = _same_color_components(inp, bg=BG)
    ordered = _ordered_tiles(comps)
    assert len({int(comp["color"]) for comp in ordered}) == 4, "tile colors must be distinct"

    heights = [comp["bbox"][2] - comp["bbox"][0] + 1 for comp in ordered]
    widths = [comp["bbox"][3] - comp["bbox"][1] + 1 for comp in ordered]
    assert len(set(heights)) == 1 and len(set(widths)) == 1, "all generated tiles must share one size"
    h = heights[0]
    w = widths[0]

    sides = []
    transformed = []
    for comp in ordered:
        patch = _crop_bbox(inp, comp["bbox"])
        color = _tile_color(patch)
        assert _is_color_connected(patch, color), "generated tile must stay connected"
        assert not np.array_equal(patch, np.fliplr(patch)), "tile must not be LR-symmetric"
        assert not np.array_equal(patch, np.flipud(patch)), "tile must not be UD-symmetric"
        assert not np.array_equal(patch, np.rot90(patch, 2)), "tile must not be 180-symmetric"
        side = _tile_notch_side(patch)
        sides.append(side)
        transformed.append(_transform_tile(patch, side))

    assert set(sides) == {"top", "right", "bottom", "left"}, "all four notch directions must appear exactly once"

    out = main(inp)
    assert out.shape == (2 * h + 1, 2 * w + 1), "output size must follow the 2x2 stitch rule"
    assert np.all(out[h, :] == SEP) and np.all(out[:, w] == SEP), "separator row and column must be maroon"
    assert np.array_equal(out[:h, :w], transformed[0])
    assert np.array_equal(out[:h, w + 1 :], transformed[1])
    assert np.array_equal(out[h + 1 :, :w], transformed[2])
    assert np.array_equal(out[h + 1 :, w + 1 :], transformed[3])


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        tile_h = random.choice(TILE_H_CHOICES)
        tile_w = random.choice(TILE_W_CHOICES)
        row_gap = random.choice(GAP_CHOICES)
        col_gap = random.choice(GAP_CHOICES)
        top = random.choice(MARGIN_CHOICES)
        left = random.choice(MARGIN_CHOICES)
        bottom = random.choice(MARGIN_CHOICES)
        right = random.choice(MARGIN_CHOICES)

        grid_h = top + tile_h + row_gap + tile_h + bottom
        grid_w = left + tile_w + col_gap + tile_w + right
        if max(grid_h, grid_w) > 30:
            continue

        directions = ["top", "right", "bottom", "left"]
        random.shuffle(directions)
        patches = _generate_tile_set(tile_h, tile_w, directions)
        if patches is None:
            continue

        grid = np.zeros((grid_h, grid_w), dtype=int)
        positions = [
            (top, left),
            (top, left + tile_w + col_gap),
            (top + tile_h + row_gap, left),
            (top + tile_h + row_gap, left + tile_w + col_gap),
        ]
        for patch, (r0, c0) in zip(patches, positions):
            grid[r0 : r0 + tile_h, c0 : c0 + tile_w] = patch

        try:
            _semantic_check(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
