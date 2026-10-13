import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
QUADRANTS = ("tl", "tr", "bl", "br")

GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 160
GRID_SIDE_RANGE = (22, 30)
FRAME_COUNT_CHOICES = (2, 3, 4)
INTERIOR_SIDE_RANGE = (4, 9)
TEMPLATE_CELL_RANGE = (3, 6)

SEED_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile each framed region with the same-color outside template, anchored from the corner implied by the internal seed.

    Input structure:
    - Black background contains several one-pixel-thick Grey rectangular frames.
    - Each frame interior has exactly one small 4-connected seed component of a single non-Black, non-Grey color.
    - Outside every Grey frame there is exactly one separate monochrome template component for each seed color.

    Rule:
    - For each Grey frame, read the seed color and locate the outside component of the same color, using its bounding box as the tile window.
    - Determine whether the seed lies in the top/bottom half and left/right half of the frame interior; that quadrant chooses the anchor corner.
    - Repeat the template periodically across the frame interior from that corner, clipping at the frame boundary and overwriting the original seed.

    Key insight:
    - Color links each seed to its template, while the seed's quadrant determines which corner of the periodic tiling grid is aligned to the framed region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _find_frames(g)
    assert frames, "expected at least one grey frame"

    frame_ids = np.full(g.shape, -1, dtype=int)
    for idx, (r0, c0, r1, c1) in enumerate(frames):
        frame_ids[r0 + 1 : r1, c0 + 1 : c1] = idx

    outside_templates = _outside_templates(g, frame_ids)
    out = g.copy()

    for r0, c0, r1, c1 in frames:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_mask = (interior != BLACK) & (interior != GREY)
        seed_components = _components(seed_mask)
        assert len(seed_components) == 1, "each frame must contain exactly one seed component"
        seed_coords = seed_components[0]

        seed_colors = np.unique(interior[seed_coords[:, 0], seed_coords[:, 1]])
        assert len(seed_colors) == 1, "seed component must be monochrome"
        seed_color = int(seed_colors[0])
        assert seed_color in outside_templates, "missing same-color outside template"

        template_mask = outside_templates[seed_color]
        quadrant = _seed_quadrant(seed_coords, interior.shape)
        tiled = _tile_from_corner(template_mask, interior.shape, quadrant)

        out[r0 + 1 : r1, c0 + 1 : c1] = BLACK
        out[r0 + 1 : r1, c0 + 1 : c1][tiled] = seed_color

    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        try:
            grid = _generate_attempt(rng)
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _component_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for coords in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        local = np.zeros((h, w), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.array_equal(local, border) and not np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == GREY):
            frames.append((r0, c0, r1, c1))
    frames.sort()
    return frames


def _outside_templates(grid: np.ndarray, frame_ids: np.ndarray) -> dict[int, np.ndarray]:
    templates: dict[int, np.ndarray] = {}
    for value in np.unique(grid):
        color = int(value)
        if color in (BLACK, GREY):
            continue
        outside: list[np.ndarray] = []
        for coords in _components(grid == color):
            ids = frame_ids[coords[:, 0], coords[:, 1]]
            if np.all(ids == -1):
                outside.append(coords)
            else:
                inside_ids = set(map(int, ids))
                assert len(inside_ids) == 1 and next(iter(inside_ids)) >= 0, "color component crosses a frame boundary"
        if outside:
            assert len(outside) == 1, "each color must have exactly one outside template"
            templates[color] = _component_to_mask(outside[0])
    return templates


def _seed_quadrant(seed_coords: np.ndarray, shape: tuple[int, int]) -> str:
    ih, iw = shape
    mean_r = float(seed_coords[:, 0].mean())
    mean_c = float(seed_coords[:, 1].mean())
    mid_r = (ih - 1) / 2.0
    mid_c = (iw - 1) / 2.0
    assert abs(mean_r - mid_r) > 1e-6 and abs(mean_c - mid_c) > 1e-6, "seed must not lie on the center axes"
    vertical = "t" if mean_r < mid_r else "b"
    horizontal = "l" if mean_c < mid_c else "r"
    return vertical + horizontal


def _tile_from_corner(template_mask: np.ndarray, out_shape: tuple[int, int], quadrant: str) -> np.ndarray:
    tile = template_mask
    top = quadrant[0] == "t"
    left = quadrant[1] == "l"
    if not top:
        tile = tile[::-1, :]
    if not left:
        tile = tile[:, ::-1]

    out_h, out_w = out_shape
    tile_h, tile_w = tile.shape
    row_offsets = np.arange(out_h) if top else (out_h - 1 - np.arange(out_h))
    col_offsets = np.arange(out_w) if left else (out_w - 1 - np.arange(out_w))
    row_mod = row_offsets % tile_h
    col_mod = col_offsets % tile_w
    return tile[row_mod[:, None], col_mod[None, :]]


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(*GRID_SIDE_RANGE)
    grid_w = rng.randint(*GRID_SIDE_RANGE)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    frame_count = rng.choice(FRAME_COUNT_CHOICES)
    colors = rng.sample(PALETTE, frame_count)
    quadrants = rng.sample(QUADRANTS, frame_count)

    specs = []
    for color, quadrant in zip(colors, quadrants):
        template = _random_template(rng)
        th, tw = template.shape
        min_h = max(INTERIOR_SIDE_RANGE[0], th + 1)
        min_w = max(INTERIOR_SIDE_RANGE[0], tw + 1)
        if min_h > INTERIOR_SIDE_RANGE[1] or min_w > INTERIOR_SIDE_RANGE[1]:
            return None
        inner_h = rng.randint(min_h, INTERIOR_SIDE_RANGE[1])
        inner_w = rng.randint(min_w, INTERIOR_SIDE_RANGE[1])
        specs.append(
            {
                "color": color,
                "quadrant": quadrant,
                "template": template,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "frame_h": inner_h + 2,
                "frame_w": inner_w + 2,
            }
        )

    for spec in sorted(specs, key=lambda item: item["frame_h"] * item["frame_w"], reverse=True):
        pos = _place_box(rng, reserved, spec["frame_h"], spec["frame_w"])
        if pos is None:
            return None
        spec["frame_r"], spec["frame_c"] = pos
        _occupy(reserved, spec["frame_r"], spec["frame_c"], spec["frame_h"], spec["frame_w"])

    for spec in specs:
        th, tw = spec["template"].shape
        pos = _place_box(rng, reserved, th, tw)
        if pos is None:
            return None
        spec["template_r"], spec["template_c"] = pos
        _occupy(reserved, spec["template_r"], spec["template_c"], th, tw)

    for spec in specs:
        r0 = spec["frame_r"]
        c0 = spec["frame_c"]
        frame_h = spec["frame_h"]
        frame_w = spec["frame_w"]
        inner_h = spec["inner_h"]
        inner_w = spec["inner_w"]
        color = spec["color"]

        grid[r0, c0 : c0 + frame_w] = GREY
        grid[r0 + frame_h - 1, c0 : c0 + frame_w] = GREY
        grid[r0 : r0 + frame_h, c0] = GREY
        grid[r0 : r0 + frame_h, c0 + frame_w - 1] = GREY

        for sr, sc in _seed_cells(inner_h, inner_w, spec["quadrant"], rng):
            grid[r0 + 1 + sr, c0 + 1 + sc] = color

        tr = spec["template_r"]
        tc = spec["template_c"]
        template = spec["template"]
        grid[tr : tr + template.shape[0], tc : tc + template.shape[1]][template] = color

    if np.array_equal(main(grid), grid):
        return None
    return grid


def _place_box(rng: random.Random, reserved: np.ndarray, box_h: int, box_w: int) -> tuple[int, int] | None:
    grid_h, grid_w = reserved.shape
    row_lo = 1
    col_lo = 1
    row_hi = grid_h - box_h - 1
    col_hi = grid_w - box_w - 1
    if row_hi < row_lo or col_hi < col_lo:
        return None
    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = rng.randint(row_lo, row_hi)
        c0 = rng.randint(col_lo, col_hi)
        if _can_place(reserved, r0, c0, box_h, box_w):
            return r0, c0
    return None


def _can_place(reserved: np.ndarray, r0: int, c0: int, box_h: int, box_w: int, pad: int = 1) -> bool:
    grid_h, grid_w = reserved.shape
    if r0 < 0 or c0 < 0 or r0 + box_h > grid_h or c0 + box_w > grid_w:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(grid_h, r0 + box_h + pad)
    cc1 = min(grid_w, c0 + box_w + pad)
    return not np.any(reserved[rr0:rr1, cc0:cc1])


def _occupy(reserved: np.ndarray, r0: int, c0: int, box_h: int, box_w: int, pad: int = 1) -> None:
    grid_h, grid_w = reserved.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(grid_h, r0 + box_h + pad)
    cc1 = min(grid_w, c0 + box_w + pad)
    reserved[rr0:rr1, cc0:cc1] = True


def _random_template(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        bbox_h = rng.randint(2, 3)
        bbox_w = rng.randint(2, 3)
        target = rng.randint(TEMPLATE_CELL_RANGE[0], min(TEMPLATE_CELL_RANGE[1], bbox_h * bbox_w))
        cells = {(rng.randrange(bbox_h), rng.randrange(bbox_w))}
        while len(cells) < target:
            frontier = list(cells)
            rng.shuffle(frontier)
            grown = False
            for r, c in frontier:
                candidates = []
                for nr, nc in _neighbors4(r, c, bbox_h, bbox_w):
                    if (nr, nc) not in cells:
                        candidates.append((nr, nc))
                if candidates:
                    cells.add(rng.choice(candidates))
                    grown = True
                    break
            if not grown:
                break

        mask = np.zeros((bbox_h, bbox_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        coords = np.argwhere(mask)
        cropped = mask[_bbox(coords)[0] : _bbox(coords)[2] + 1, _bbox(coords)[1] : _bbox(coords)[3] + 1]
        if cropped.sum() < TEMPLATE_CELL_RANGE[0]:
            continue
        if np.array_equal(cropped, cropped[::-1, :]) or np.array_equal(cropped, cropped[:, ::-1]):
            continue
        if cropped.all():
            continue
        return cropped
    raise RuntimeError("failed to sample a template")


def _seed_cells(inner_h: int, inner_w: int, quadrant: str, rng: random.Random) -> list[tuple[int, int]]:
    shapes = list(SEED_SHAPES)
    rng.shuffle(shapes)

    for shape0 in shapes:
        shape = list(shape0)
        max_r = max(r for r, _ in shape)
        max_c = max(c for _, c in shape)
        row_room = inner_h - 1 - max_r
        col_room = inner_w - 1 - max_c

        if quadrant[0] == "t":
            row_options = list(range(min(3, row_room + 1)))
        else:
            row_options = [row_room - d for d in range(min(3, row_room + 1))]
        if quadrant[1] == "l":
            col_options = list(range(min(3, col_room + 1)))
        else:
            col_options = [col_room - d for d in range(min(3, col_room + 1))]

        rng.shuffle(row_options)
        rng.shuffle(col_options)
        for row_shift in row_options:
            for col_shift in col_options:
                cells = [(row_shift + r, col_shift + c) for r, c in shape]
                local = np.array(cells, dtype=int)
                if _seed_quadrant(local, (inner_h, inner_w)) == quadrant:
                    return cells

    raise RuntimeError("failed to place an unambiguous seed")
