import random
from collections import Counter

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


PAINT_COLORS = (
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
)
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.ravel())
    return int(counts.most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
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
            comps.append(np.array(coords, dtype=int))
    return comps


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _flood_from_border(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if mask[r, c] and not out[r, c]:
                out[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if mask[r, c] and not out[r, c]:
                out[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not out[nr, nc]:
                out[nr, nc] = True
                stack.append((nr, nc))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each empty closed outline with the color shown by same-area filled examples in the same grid.

    Input structure:
    - A black background contains multiple gray one-pixel outlines.
    - Some outlines are closed and already have their whole interior filled with a color.
    - Other closed outlines have black interiors, while some outlines have a one-cell opening.

    Rule:
    - Treat gray pixels as walls and split all non-gray cells into 4-connected regions.
    - Every non-gray region that does not touch the border is an enclosed interior; its area is the number of cells in that region.
    - Read an area-to-color codebook from enclosed regions that are already filled, then fill each still-black enclosed region whose area appears in that codebook.
    - Leave open regions, gray outlines, and the exterior background unchanged.

    Key insight:
    - The mapping is not global: each grid teaches its own fill rule through already-colored enclosed regions, and only true closures qualify for filling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    out = g.copy()
    codebook: dict[int, int] = {}
    pending: list[tuple[np.ndarray, int]] = []

    for coords in _component_coords(g != Color.GREY):
        if _touches_border(coords, g.shape):
            continue
        vals = g[coords[:, 0], coords[:, 1]]
        colors = sorted({int(v) for v in vals if int(v) != bg})
        area = int(coords.shape[0])
        if colors:
            assert len(colors) == 1
            color = colors[0]
            if area in codebook:
                assert codebook[area] == color
            else:
                codebook[area] = color
        else:
            pending.append((coords, area))

    for coords, area in pending:
        color = codebook.get(area)
        if color is not None:
            out[coords[:, 0], coords[:, 1]] = color

    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _sample_region(area: int) -> set[tuple[int, int]] | None:
    for _ in range(80):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            cells.add(random.choice(tuple(frontier)))

        cells = _normalize(cells)
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        h = max(rows) + 1
        w = max(cols) + 1
        if area >= 4 and (h == 1 or w == 1):
            continue
        if h > 6 or w > 6:
            continue
        return cells
    return None


def _build_patch(area: int, fill_color: int | None, closed: bool) -> dict | None:
    region = _sample_region(area)
    if region is None:
        return None

    max_r = max(r for r, _ in region)
    max_c = max(c for _, c in region)
    patch = np.full((max_r + 3, max_c + 3), Color.BLACK, dtype=int)
    region_mask = np.zeros_like(patch, dtype=bool)
    for r, c in region:
        region_mask[r + 1, c + 1] = True

    outline = np.zeros_like(patch, dtype=bool)
    for r, c in zip(*np.where(region_mask)):
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < patch.shape[0] and 0 <= nc < patch.shape[1] and not region_mask[nr, nc]:
                outline[nr, nc] = True
    patch[outline] = Color.GREY

    if fill_color is None:
        patch[region_mask] = Color.BLACK
    else:
        patch[region_mask] = int(fill_color)

    if not closed:
        exterior = _flood_from_border(patch == Color.BLACK)
        candidates = []
        for r, c in zip(*np.where(outline)):
            has_region = any(
                0 <= nr < patch.shape[0] and 0 <= nc < patch.shape[1] and region_mask[nr, nc]
                for nr, nc in _neighbors4(int(r), int(c))
            )
            has_exterior = any(
                0 <= nr < patch.shape[0]
                and 0 <= nc < patch.shape[1]
                and exterior[nr, nc]
                for nr, nc in _neighbors4(int(r), int(c))
            )
            if has_region and has_exterior:
                candidates.append((int(r), int(c)))
        if not candidates:
            return None
        gap_r, gap_c = random.choice(candidates)
        patch[gap_r, gap_c] = Color.BLACK

    region_coords = np.argwhere(region_mask).astype(int)
    return {"patch": patch, "region_coords": region_coords, "area": int(area)}


def _can_place(
    reserved: np.ndarray, top: int, left: int, h: int, w: int, gap: int
) -> bool:
    h0, w0 = reserved.shape
    if top + h > h0 or left + w > w0:
        return False
    r0 = max(0, top - gap)
    c0 = max(0, left - gap)
    r1 = min(h0, top + h + gap)
    c1 = min(w0, left + w + gap)
    return not reserved[r0:r1, c0:c1].any()


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    mask = patch != Color.BLACK
    grid[top : top + h, left : left + w][mask] = patch[mask]


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(18, 30)
    width = random.randint(18, 30)
    grid = np.full((height, width), Color.BLACK, dtype=int)
    expected = grid.copy()
    reserved = np.zeros((height, width), dtype=bool)

    n_areas = random.randint(2, 4)
    areas = random.sample(range(2, 9), n_areas)
    colors = random.sample(PAINT_COLORS, n_areas)
    area_to_color = dict(zip(areas, colors))

    objects = []
    for area in areas:
        exemplar_count = random.randint(1, 2)
        target_count = random.randint(1, 2)
        for _ in range(exemplar_count):
            obj = _build_patch(area, area_to_color[area], closed=True)
            if obj is None:
                return None
            obj["fill_result"] = None
            objects.append(obj)
        for _ in range(target_count):
            obj = _build_patch(area, None, closed=True)
            if obj is None:
                return None
            obj["fill_result"] = area_to_color[area]
            objects.append(obj)

    open_count = random.randint(1, 3)
    for _ in range(open_count):
        area = random.choice(areas if random.random() < 0.7 else list(range(2, 9)))
        obj = _build_patch(area, None, closed=False)
        if obj is None:
            return None
        obj["fill_result"] = None
        objects.append(obj)

    random.shuffle(objects)
    for obj in objects:
        patch = obj["patch"]
        ph, pw = patch.shape
        positions = [
            (r, c)
            for r in range(height - ph + 1)
            for c in range(width - pw + 1)
            if _can_place(reserved, r, c, ph, pw, gap=1)
        ]
        if not positions:
            return None
        top, left = random.choice(positions)
        _place_patch(grid, patch, top, left)
        _place_patch(expected, patch, top, left)
        reserved[top : top + ph, left : left + pw] = True
        fill_color = obj["fill_result"]
        if fill_color is not None:
            coords = obj["region_coords"]
            rr = coords[:, 0] + top
            cc = coords[:, 1] + left
            expected[rr, cc] = int(fill_color)

    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_scene()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate valid input for ARC task 3")
