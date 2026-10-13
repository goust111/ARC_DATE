import itertools
import random

import numpy as np


BG = 0
GREY = 5
RED = 2
BLUE = 1
GREEN = 3
YELLOW = 4
CORNER_COLORS = (RED, BLUE, GREEN, YELLOW)
TRANSFORM_BY_COLOR = {
    RED: "identity",
    BLUE: "hflip",
    GREEN: "vflip",
    YELLOW: "rot180",
}
REGION_OWNER = {
    "top": "tl",
    "right": "tr",
    "bottom": "br",
    "left": "bl",
}
GEN_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[rows - r0, cols - c0] = True
    return mask, (r0, c0, r1, c1)


def _apply_transform(mask: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return mask.copy()
    if name == "hflip":
        return np.fliplr(mask)
    if name == "vflip":
        return np.flipud(mask)
    if name == "rot180":
        return np.rot90(mask, 2)
    raise AssertionError(f"unknown transform: {name}")


def _region_of_cell(r: int, c: int, h: int, w: int) -> str | None:
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    if (r, c) in corners:
        return "corner"
    y = r / (h - 1)
    x = c / (w - 1)
    if abs(y - x) < 1e-9 or abs(y - (1 - x)) < 1e-9:
        return None
    if y < x and y < 1 - x:
        return "top"
    if y > x and y > 1 - x:
        return "bottom"
    if y > x and y < 1 - x:
        return "left"
    return "right"


def _region_grid(h: int, w: int) -> np.ndarray:
    out = np.empty((h, w), dtype=object)
    for r in range(h):
        for c in range(w):
            out[r, c] = _region_of_cell(r, c, h, w)
    return out


def _build_shape_library() -> list[np.ndarray]:
    def connected(cells: tuple[tuple[int, int], ...]) -> bool:
        cell_set = set(cells)
        start = next(iter(cell_set))
        stack = [start]
        seen = {start}
        while stack:
            r, c = stack.pop()
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in cell_set and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        return seen == cell_set

    def key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(int(v) for v in row) for row in mask.astype(int))

    shapes = []
    seen = set()
    for h in range(2, 4):
        for w in range(2, 4):
            cells = [(r, c) for r in range(h) for c in range(w)]
            for area in range(3, 6):
                for combo in itertools.combinations(cells, area):
                    if not connected(combo):
                        continue
                    mask, _bbox = _tight_mask(np.array(combo, dtype=int))
                    if mask.shape != (h, w):
                        continue
                    if any(
                        np.array_equal(mask, _apply_transform(mask, name))
                        for name in ("hflip", "vflip", "rot180")
                    ):
                        continue
                    canon = min(
                        key(_apply_transform(mask, name))
                        for name in ("identity", "hflip", "vflip", "rot180")
                    )
                    if canon in seen:
                        continue
                    seen.add(canon)
                    shapes.append(mask)
    assert shapes, "shape library must be non-empty"
    return shapes


SHAPE_LIBRARY = _build_shape_library()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the canvas into four implicit diagonal regions and transform each grey
    object in place according to the corner marker that owns that region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 3 and w >= 3

    bg = _background_color(g)
    corners = {
        "tl": int(g[0, 0]),
        "tr": int(g[0, w - 1]),
        "bl": int(g[h - 1, 0]),
        "br": int(g[h - 1, w - 1]),
    }
    assert set(corners.values()) == set(CORNER_COLORS)

    non_bg = set(map(int, np.unique(g))) - {bg}
    object_colors = non_bg - set(CORNER_COLORS)
    assert len(object_colors) == 1
    object_color = next(iter(object_colors))

    regions = _region_grid(h, w)
    out = g.copy()
    objects = _components(g == object_color)
    assert objects, "expected at least one grey object"

    for coords in objects:
        mask, (r0, c0, r1, c1) = _tight_mask(coords)
        bbox_regions = {regions[r, c] for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        assert len(bbox_regions) == 1 and None not in bbox_regions and "corner" not in bbox_regions
        region = next(iter(bbox_regions))
        transform = TRANSFORM_BY_COLOR[corners[REGION_OWNER[region]]]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[mask] = bg
        patch[_apply_transform(mask, transform)] = object_color

    return out


def _bbox_candidates(region_grid: np.ndarray, region: str, shape: np.ndarray, occupied: np.ndarray) -> list[tuple[int, int]]:
    h, w = region_grid.shape
    sh, sw = shape.shape
    candidates = []
    for r0 in range(1, h - sh):
        for c0 in range(1, w - sw):
            if not np.all(region_grid[r0 : r0 + sh, c0 : c0 + sw] == region):
                continue
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + sh + 1)
            cc1 = min(w, c0 + sw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            candidates.append((r0, c0))
    return candidates


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(11, 17)
    w = random.randint(11, 17)
    grid = np.full((h, w), BG, dtype=int)

    corner_colors = list(CORNER_COLORS)
    random.shuffle(corner_colors)
    grid[0, 0], grid[0, w - 1], grid[h - 1, 0], grid[h - 1, w - 1] = corner_colors

    region_grid = _region_grid(h, w)
    occupied = np.zeros((h, w), dtype=bool)
    occupied[0, 0] = True
    occupied[0, w - 1] = True
    occupied[h - 1, 0] = True
    occupied[h - 1, w - 1] = True

    requests = ["top", "right", "bottom", "left"]
    requests += random.choices(requests, k=random.randint(0, 2))
    random.shuffle(requests)

    for region in requests:
        placements = []
        for shape in random.sample(SHAPE_LIBRARY, len(SHAPE_LIBRARY)):
            candidates = _bbox_candidates(region_grid, region, shape, occupied)
            if candidates:
                placements.append((shape, candidates))
        if not placements:
            return None
        shape, candidates = random.choice(placements)
        r0, c0 = random.choice(candidates)
        sh, sw = shape.shape
        grid[r0 : r0 + sh, c0 : c0 + sw][shape] = GREY
        occupied[r0 : r0 + sh, c0 : c0 + sw] = True

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
