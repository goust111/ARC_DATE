from __future__ import annotations

import random

import numpy as np


BG = 0
MAX_GENERATION_ATTEMPTS = 500
INPUT_SIZES = (4, 5)
COLORS = tuple(range(1, 10))


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mirror_mosaic(grid: np.ndarray) -> np.ndarray:
    rows = np.arange(grid.shape[0], dtype=int)
    cols = np.arange(grid.shape[1], dtype=int)
    row_order = np.concatenate((rows[::-1], rows, rows[::-1]))
    col_order = np.concatenate((cols[::-1], cols, cols[::-1]))
    return grid[row_order[:, None], col_order[None, :]]


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    ps = mask.astype(int).cumsum(axis=0).cumsum(axis=1)
    return np.pad(ps, ((1, 0), (1, 0)), constant_values=0)


def _rect_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    return int(ps[r1 + 1, c1 + 1] - ps[r0, c1 + 1] - ps[r1 + 1, c0] + ps[r0, c0])


def _detect_rectangles(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    colors = [int(color) for color in np.unique(grid) if int(color) != BG]
    raw: set[tuple[int, int, int, int, int]] = set()

    for color in colors:
        ps = _prefix_sum(grid == color)
        for r0 in range(h):
            for c0 in range(w):
                if int(grid[r0, c0]) != color:
                    continue
                for r1 in range(r0 + 2, h):
                    for c1 in range(c0 + 2, w):
                        area = (r1 - r0 + 1) * (c1 - c0 + 1)
                        if _rect_sum(ps, r0, c0, r1, c1) != area:
                            continue
                        if r0 > 0 and bool(np.all(grid[r0 - 1, c0 : c1 + 1] == color)):
                            continue
                        if r1 + 1 < h and bool(np.all(grid[r1 + 1, c0 : c1 + 1] == color)):
                            continue
                        if c0 > 0 and bool(np.all(grid[r0 : r1 + 1, c0 - 1] == color)):
                            continue
                        if c1 + 1 < w and bool(np.all(grid[r0 : r1 + 1, c1 + 1] == color)):
                            continue
                        raw.add((color, r0, c0, r1, c1))

    rects = [
        {
            "color": color,
            "bbox": (r0, c0, r1, c1),
            "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
        }
        for color, r0, c0, r1, c1 in sorted(raw)
    ]
    rects.sort(key=lambda rect: rect["area"], reverse=True)

    kept: list[dict] = []
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        contained = False
        for other in kept:
            if rect["color"] != other["color"]:
                continue
            or0, oc0, or1, oc1 = other["bbox"]
            if or0 <= r0 and oc0 <= c0 and r1 <= or1 and c1 <= oc1 and (other["bbox"] != rect["bbox"]):
                contained = True
                break
        if not contained:
            kept.append(rect)

    return kept


def _crosses_tile_boundary(bbox: tuple[int, int, int, int], tile_h: int, tile_w: int) -> bool:
    r0, c0, r1, c1 = bbox
    return any(r0 < cut <= r1 for cut in (tile_h, 2 * tile_h)) or any(
        c0 < cut <= c1 for cut in (tile_w, 2 * tile_w)
    )


def _border_mask(shape: tuple[int, int], rects: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        out[r0, c0 : c1 + 1] = True
        out[r1, c0 : c1 + 1] = True
        out[r0 : r1 + 1, c0] = True
        out[r0 : r1 + 1, c1] = True
    return out


def _touches_edge(rect: dict, h: int, w: int) -> bool:
    r0, c0, r1, c1 = rect["bbox"]
    return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1


def _rect_cells(rect: dict) -> tuple[slice, slice]:
    r0, c0, r1, c1 = rect["bbox"]
    return slice(r0, r1 + 1), slice(c0, c1 + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the input by whole-grid reflections, then hollow only the solid rectangles that emerge across tile boundaries.

    Input structure:
    - The input is a 4x4 or 5x5 grid with black background and 1-3 non-overlapping colored solid rectangles.
    - Rectangle colors are distinct, and at least one rectangle touches a grid edge while at least one other rectangle stays away from all edges.
    - The rectangles are axis-aligned and fully filled inside their bboxes.

    Rule:
    - Build a 3x3 mirror mosaic: the center tile is the input, vertical neighbors reverse row order, horizontal neighbors reverse column order, and corners reverse both.
    - In that mosaic, detect every maximal monochrome solid axis-aligned rectangle whose shorter side is at least 3 cells.
    - For each such rectangle, test whether its bbox crosses one of the tile boundaries at rows H and 2H or columns W and 2W.
    - Hollow only the cross-boundary rectangles by turning their strict interiors black while preserving every detected rectangle border.

    Key insight:
    - The rectangles that matter are not the original small ones but the larger solid blocks created only after mirrored tiling joins edge-touching pieces across tile seams.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    mosaic = _mirror_mosaic(g)
    rects = _detect_rectangles(mosaic)
    cross_rects = [rect for rect in rects if _crosses_tile_boundary(rect["bbox"], h, w)]
    cross_rects.sort(key=lambda rect: rect["area"], reverse=True)

    out = mosaic.copy()
    protected = _border_mask(mosaic.shape, rects)
    for rect in cross_rects:
        r0, c0, r1, c1 = rect["bbox"]
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            continue
        inner = np.ones((r1 - r0 - 1, c1 - c0 - 1), dtype=bool)
        keep = protected[r0 + 1 : r1, c0 + 1 : c1]
        inner[keep] = False
        patch = out[r0 + 1 : r1, c0 + 1 : c1]
        patch[inner] = BG
    return out


def _rect(r0: int, c0: int, h: int, w: int) -> dict:
    return {"bbox": (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))}


def _compatible(rect: dict, placed: list[dict], h: int, w: int) -> bool:
    r0, c0, r1, c1 = rect["bbox"]
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    for other in placed:
        or0, oc0, or1, oc1 = other["bbox"]
        if not (r1 < or0 or or1 < r0 or c1 < oc0 or oc1 < c0):
            return False
    return True


def _place_search(candidate_lists: list[list[dict]], h: int, w: int) -> list[dict] | None:
    placed: list[dict] = []

    def rec(idx: int) -> bool:
        if idx == len(candidate_lists):
            return True
        options = candidate_lists[idx][:]
        random.shuffle(options)
        for rect in options:
            if not _compatible(rect, placed, h, w):
                continue
            placed.append(rect)
            if rec(idx + 1):
                return True
            placed.pop()
        return False

    if rec(0):
        return [dict(item) for item in placed]
    return None


def _primary_candidates(h: int, w: int, mode: str) -> list[dict]:
    out: list[dict] = []
    if mode == "horizontal":
        for rh, rw in ((3, 2), (3, 3)):
            if rh > h or rw > w:
                continue
            for side_c in (0, w - rw):
                for r0 in range(0, h - rh + 1):
                    out.append(_rect(r0, side_c, rh, rw))
    elif mode == "vertical":
        for rh, rw in ((2, 3), (3, 3)):
            if rh > h or rw > w:
                continue
            for side_r in (0, h - rh):
                for c0 in range(0, w - rw + 1):
                    out.append(_rect(side_r, c0, rh, rw))
    else:
        for rh, rw in ((2, 2), (2, 3), (3, 2)):
            if rh > h or rw > w:
                continue
            for r0 in (0, h - rh):
                for c0 in (0, w - rw):
                    out.append(_rect(r0, c0, rh, rw))
    return out


def _interior_candidates(h: int, w: int) -> list[dict]:
    out: list[dict] = []
    for rh, rw in ((1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2), (3, 3)):
        if rh > h - 2 or rw > w - 2:
            continue
        for r0 in range(1, h - rh):
            for c0 in range(1, w - rw):
                out.append(_rect(r0, c0, rh, rw))
    return out


def _thin_edge_candidates(h: int, w: int) -> list[dict]:
    out: list[dict] = []
    for rh, rw in ((3, 1), (1, 3), (2, 1), (1, 2)):
        if rh > h or rw > w:
            continue
        for r0 in range(0, h - rh + 1):
            for c0 in range(0, w - rw + 1):
                rect = _rect(r0, c0, rh, rw)
                if _touches_edge(rect, h, w):
                    out.append(rect)
    return out


def _build_grid(h: int, w: int, rects: list[dict], colors: list[int]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for rect, color in zip(rects, colors):
        rs, cs = _rect_cells(rect)
        grid[rs, cs] = int(color)
    return grid


def _input_is_valid(grid: np.ndarray) -> bool:
    colors = [int(color) for color in np.unique(grid) if int(color) != BG]
    if not (2 <= len(colors) <= 3):
        return False

    h, w = grid.shape
    edge_count = 0
    interior_count = 0
    for color in colors:
        coords = np.argwhere(grid == color)
        r0, c0, r1, c1 = _bbox(coords)
        if len(coords) != (r1 - r0 + 1) * (c1 - c0 + 1):
            return False
        if not bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)):
            return False
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            edge_count += 1
        else:
            interior_count += 1
    return edge_count >= 1 and interior_count >= 1


def _semantic_checks(grid: np.ndarray) -> bool:
    if not _input_is_valid(grid):
        return False

    h, w = grid.shape
    mosaic = _mirror_mosaic(grid)
    rects = _detect_rectangles(mosaic)
    cross = [rect for rect in rects if _crosses_tile_boundary(rect["bbox"], h, w)]
    if not cross:
        return False

    if not any(min(rect["bbox"][2] - rect["bbox"][0] + 1, rect["bbox"][3] - rect["bbox"][1] + 1) >= 3 for rect in cross):
        return False

    out = main(grid)
    if out.shape != (3 * h, 3 * w):
        return False
    if not bool(np.any(out != mosaic)):
        return False

    thin_edge_seen = False
    for color in [int(color) for color in np.unique(grid) if int(color) != BG]:
        coords = np.argwhere(grid == color)
        r0, c0, r1, c1 = _bbox(coords)
        if (r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1) and min(r1 - r0 + 1, c1 - c0 + 1) == 1:
            thin_edge_seen = True
            break

    large_cross = any(rect["area"] >= 12 for rect in cross)
    return thin_edge_seen or large_cross or len(cross) >= 2


def _generate_attempt() -> np.ndarray | None:
    h = w = random.choice(INPUT_SIZES)
    primary_mode = random.choice(("horizontal", "vertical", "corner"))
    primary = _primary_candidates(h, w, primary_mode)
    interior = _interior_candidates(h, w)
    if not primary or not interior:
        return None

    candidate_lists = [primary, interior]
    if random.random() < 0.7:
        third_role = random.choice(("thin", "second_cross", "small_interior"))
        if third_role == "thin":
            third = _thin_edge_candidates(h, w)
        elif third_role == "second_cross":
            alt_modes = [mode for mode in ("horizontal", "vertical", "corner") if mode != primary_mode]
            third = _primary_candidates(h, w, random.choice(alt_modes))
        else:
            third = _interior_candidates(h, w)
        if third:
            candidate_lists.append(third)

    placed = _place_search(candidate_lists, h, w)
    if placed is None:
        return None

    colors = random.sample(COLORS, len(placed))
    grid = _build_grid(h, w, placed, colors)
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-tiling hollowing input")
