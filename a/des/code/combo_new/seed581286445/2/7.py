from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
PALETTE = (1, 2, 3, 5, 6, 7, 8, 9)
GEN_RETRIES = 400
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _boundary_background(grid: np.ndarray) -> int:
    boundary = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    values, counts = np.unique(boundary, return_counts=True)
    return int(values[np.argmax(counts)])


def _tile_orientation(tile: np.ndarray) -> str:
    tl = int(tile[0, 0])
    tr = int(tile[0, -1])
    bl = int(tile[-1, 0])
    br = int(tile[-1, -1])
    if tl == br and tr != bl:
        return "main"
    if tr == bl and tl != br:
        return "anti"
    raise AssertionError("tile corners do not encode a unique diagonal split")


def _paint_ring(out: np.ndarray, bbox: tuple[int, int, int, int], bg: int, color: int) -> None:
    h, w = out.shape
    r0, c0, r1, c1 = bbox
    for r in range(max(0, r0 - 1), min(h, r1 + 2)):
        for c in range(max(0, c0 - 1), min(w, c1 + 2)):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            if int(out[r, c]) == int(bg):
                out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the uniquely odd diagonal-split tile and highlight its surrounding gap with yellow.

    Input structure:
    - The background is black.
    - Several equal-size rectangular tiles are separated by one-cell black gaps and arranged on the grid.
    - Every tile is completely filled by exactly two colors, split by one diagonal, so either the top-left/bottom-right corners match or the top-right/bottom-left corners match.
    - Exactly one tile uses the opposite diagonal split from the majority.

    Rule:
    - Segment the non-black tiles as connected rectangular components of equal size.
    - For each tile, read its split direction from which opposite corner pair shares a color.
    - Find the unique tile whose diagonal split direction differs from the majority and rotate that whole tile by 180 degrees in place.
    - Paint the one-cell-thick background ring immediately outside that tile's bounding box yellow, leaving every other tile and all remaining background cells unchanged.

    Key insight:
    - Tile colors vary independently, so the stable cue is geometric: the matched opposite-corner pair reveals the diagonal orientation, which identifies the lone anomaly before any pixel is modified.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(_boundary_background(g))

    comps = _components(g != int(bg))
    assert len(comps) >= 3, "expected several non-background tiles"

    records: list[dict[str, object]] = []
    shapes = set()
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert not bool((patch == int(bg)).any()), "tiles must be solid rectangles without background holes"
        colors = set(map(int, patch.ravel()))
        assert len(colors) == 2, "each tile must contain exactly two colors"
        shapes.add(tuple(map(int, patch.shape)))
        records.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "patch": patch.copy(),
                "orientation": _tile_orientation(patch),
            }
        )

    assert len(shapes) == 1, "all tiles must share one size"

    counts = {
        "main": sum(record["orientation"] == "main" for record in records),
        "anti": sum(record["orientation"] == "anti" for record in records),
    }
    assert counts["main"] != counts["anti"], "expected a unique majority orientation"
    majority = "main" if counts["main"] > counts["anti"] else "anti"
    outliers = [record for record in records if record["orientation"] != majority]
    assert len(outliers) == 1, "expected exactly one odd tile"

    target = outliers[0]
    r0, c0, r1, c1 = target["bbox"]
    out = g.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = np.rot90(target["patch"], 2)
    _paint_ring(out, target["bbox"], bg, YELLOW)
    return out


def _make_tile(height: int, width: int, orientation: str, colors: tuple[int, int]) -> np.ndarray:
    rr, cc = np.indices((height, width))
    if orientation == "main":
        mask = rr * (width - 1) <= cc * (height - 1)
    else:
        mask = rr * (width - 1) + cc * (height - 1) <= (height - 1) * (width - 1)
    return np.where(mask, int(colors[0]), int(colors[1])).astype(int)


def _expected_output(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    r0, c0, r1, c1 = bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = np.rot90(out[r0 : r1 + 1, c0 : c1 + 1], 2)
    _paint_ring(out, bbox, BG, YELLOW)
    return out


def _sample_layout() -> tuple[int, int, int, int, int, int] | None:
    for _ in range(200):
        tile_h = random.randint(3, 6)
        tile_w = random.randint(3, 6)
        rows = random.randint(2, 4)
        cols = random.randint(2, 4)
        margin_r = random.randint(1, 2)
        margin_c = random.randint(1, 2)
        height = 2 * margin_r + rows * tile_h + (rows - 1)
        width = 2 * margin_c + cols * tile_w + (cols - 1)
        if height <= 30 and width <= 30:
            return tile_h, tile_w, rows, cols, margin_r, margin_c
    return None


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    tile_h, tile_w, rows, cols, margin_r, margin_c = layout

    grid_h = 2 * margin_r + rows * tile_h + (rows - 1)
    grid_w = 2 * margin_c + cols * tile_w + (cols - 1)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    majority = random.choice(("main", "anti"))
    odd = "anti" if majority == "main" else "main"
    odd_index = random.randrange(rows * cols)
    odd_bbox: tuple[int, int, int, int] | None = None

    for idx in range(rows * cols):
        tile_r = idx // cols
        tile_c = idx % cols
        r0 = margin_r + tile_r * (tile_h + 1)
        c0 = margin_c + tile_c * (tile_w + 1)
        orientation = odd if idx == odd_index else majority
        colors = tuple(random.sample(PALETTE, 2))
        if random.random() < 0.5:
            colors = (colors[1], colors[0])
        tile = _make_tile(tile_h, tile_w, orientation, colors)
        grid[r0 : r0 + tile_h, c0 : c0 + tile_w] = tile
        if idx == odd_index:
            odd_bbox = (int(r0), int(c0), int(r0 + tile_h - 1), int(c0 + tile_w - 1))

    assert odd_bbox is not None
    expected = _expected_output(grid, odd_bbox)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-tile puzzle")
