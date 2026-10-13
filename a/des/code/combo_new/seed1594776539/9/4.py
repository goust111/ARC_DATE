from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
LEGEND_INNER = 3
LEGEND_HEIGHT = LEGEND_INNER + 2
TEMP_CANVAS = 30
ROOM_COUNT_RANGE = (2, 4)
DISTRACTOR_COUNT_RANGE = (0, 2)
ROOM_OUTER_RANGE = (4, 8)
PLACEMENT_RETRIES = 300
GENERATION_RETRIES = 300

TEMPLATE_MASKS = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=bool),
    np.array([[0, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=bool),
    np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=bool),
    np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=bool),
    np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
]


def _neighbors4(r: int, c: int, h: int, w: int):
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
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_from_coords(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _room_boundary(room: np.ndarray) -> np.ndarray:
    h, w = room.shape
    boundary = np.zeros_like(room, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not room[r, c]:
                continue
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                boundary[r, c] = True
                continue
            if not (room[r - 1, c] and room[r + 1, c] and room[r, c - 1] and room[r, c + 1]):
                boundary[r, c] = True
    return boundary


def _enclosed_from_boundary(boundary: np.ndarray) -> np.ndarray:
    h, w = boundary.shape
    blocked = np.zeros((h + 2, w + 2), dtype=bool)
    blocked[1:-1, 1:-1] = boundary
    seen = np.zeros_like(blocked, dtype=bool)
    q = deque([(0, 0)])
    seen[0, 0] = True

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h + 2, w + 2):
            if blocked[nr, nc] or seen[nr, nc]:
                continue
            seen[nr, nc] = True
            q.append((nr, nc))

    interior = (~blocked) & (~seen)
    return interior[1:-1, 1:-1]


def _parse_legend(grid: np.ndarray) -> tuple[dict[int, np.ndarray], np.ndarray]:
    grey_components = [comp for comp in _components(grid == GREY) if np.any(comp[:, 0] == 0)]
    assert grey_components, "expected a top legend framed in grey"
    legend_coords = max(grey_components, key=len)
    legend_mask = _mask_from_coords(grid.shape, legend_coords)
    r0, c0, r1, c1 = _bbox(legend_coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]

    sep_rows = [r for r in range(sub.shape[0]) if np.all(sub[r, :] == GREY)]
    sep_cols = [c for c in range(sub.shape[1]) if np.all(sub[:, c] == GREY)]
    assert len(sep_rows) >= 2 and len(sep_cols) >= 2, "legend must have grey borders and dividers"

    inner_r0 = sep_rows[0] + 1
    inner_r1 = sep_rows[-1] - 1
    assert inner_r0 <= inner_r1, "legend cells must have positive height"

    templates: dict[int, np.ndarray] = {}
    widths = []
    for left, right in zip(sep_cols, sep_cols[1:]):
        if right - left <= 1:
            continue
        patch = sub[inner_r0 : inner_r1 + 1, left + 1 : right]
        colors = set(map(int, np.unique(patch))) - {BG, GREY}
        assert len(colors) == 1, "each legend cell must contain exactly one template color"
        color = int(next(iter(colors)))
        assert color not in templates, "legend colors must be unique"
        templates[color] = (patch == color)
        widths.append(int(patch.shape[1]))

    assert templates, "expected at least one legend template"
    heights = {int(mask.shape[0]) for mask in templates.values()}
    widths = set(widths)
    assert len(heights) == 1 and len(widths) == 1, "legend cells must share one size"
    cell_h = next(iter(heights))
    cell_w = next(iter(widths))
    assert cell_h % 2 == 1 and cell_w % 2 == 1, "stamp cells must have a center pixel"
    return templates, legend_mask


def _center_of_interior(interior: np.ndarray) -> tuple[int, int]:
    rows = np.flatnonzero(interior.any(axis=1))
    cols = np.flatnonzero(interior.any(axis=0))
    assert rows.size > 0 and cols.size > 0, "region interior must be non-empty"
    return int(rows[(rows.size - 1) // 2]), int(cols[(cols.size - 1) // 2])


def _paint_stamp(out: np.ndarray, top: int, left: int, interior: np.ndarray, color: int, stamp: np.ndarray) -> None:
    patch = out[top : top + interior.shape[0], left : left + interior.shape[1]]
    patch[interior] = BG

    center_r, center_c = _center_of_interior(interior)
    anchor_r = stamp.shape[0] // 2
    anchor_c = stamp.shape[1] // 2
    for sr, sc in np.argwhere(stamp):
        rr = center_r + int(sr) - anchor_r
        cc = center_c + int(sc) - anchor_c
        if 0 <= rr < interior.shape[0] and 0 <= cc < interior.shape[1] and interior[rr, cc]:
            patch[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the legend shape matching each seed color into the center of its closed grey region.

    Input structure:
    - The background is black.
    - Several single-pixel-wide grey closed boundaries enclose separate regions, each with exactly one colored seed pixel inside.
    - A grey-framed legend at the top is split into cells by grey vertical dividers, and each cell contains one small colored stamp template.

    Rule:
    - Match every region's seed color to the legend cell with the same color.
    - Recover the enclosed interior of the grey boundary, then take the middle interior row and middle interior column as the region center, using the upper-left middle position when the interior span is even.
    - Clear the seed and draw the matched stamp with its center pixel aligned to that region center, clipping any stamp pixels that would fall outside the region interior.

    Key insight:
    - The seed color is only a lookup key for the legend, while the placement point comes from the geometry of the enclosed space rather than from the seed location.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    templates, legend_mask = _parse_legend(g)
    region_grey = (g == GREY) & (~legend_mask)
    components = _components(region_grey)
    assert components, "expected at least one closed region"

    out = g.copy()
    for comp in components:
        r0, c0, r1, c1 = _bbox(comp)
        boundary = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        boundary[comp[:, 0] - r0, comp[:, 1] - c0] = True
        interior = _enclosed_from_boundary(boundary)
        assert interior.any(), "grey boundary must enclose an interior"
        assert len(_components(interior)) == 1, "enclosed region must be a single interior space"

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        seed_vals = patch[interior]
        seed_colors = set(map(int, np.unique(seed_vals))) - {BG}
        assert len(seed_colors) == 1, "each region interior must contain exactly one seed color"
        color = int(next(iter(seed_colors)))
        assert int(np.count_nonzero(seed_vals == color)) == 1, "the seed must be a single pixel"
        assert color in templates, "seed color must appear in the legend"

        _paint_stamp(out, r0, c0, interior, color, templates[color])

    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _cut_corner(room: np.ndarray, corner: str, cut_h: int, cut_w: int) -> None:
    if corner == "tl":
        room[:cut_h, :cut_w] = False
    elif corner == "tr":
        room[:cut_h, -cut_w:] = False
    elif corner == "bl":
        room[-cut_h:, :cut_w] = False
    else:
        room[-cut_h:, -cut_w:] = False


def _random_room() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    for _ in range(200):
        h = random.randint(*ROOM_OUTER_RANGE)
        w = random.randint(*ROOM_OUTER_RANGE)
        room = np.ones((h, w), dtype=bool)
        kind = random.choice(["rect", "single_cut", "double_cut"])
        if kind != "rect":
            max_cut_h = min(2, h - 3)
            max_cut_w = min(2, w - 3)
            if max_cut_h < 1 or max_cut_w < 1:
                continue
            count = 1 if kind == "single_cut" else 2
            for corner in random.sample(["tl", "tr", "bl", "br"], count):
                _cut_corner(room, corner, random.randint(1, max_cut_h), random.randint(1, max_cut_w))

        if not room.any() or len(_components(room)) != 1:
            continue

        boundary = _room_boundary(room)
        if len(_components(boundary)) != 1:
            continue

        interior = room & (~boundary)
        if not interior.any() or len(_components(interior)) != 1:
            continue

        return room, boundary, interior
    return None


def _render_legend(grid: np.ndarray, legend_left: int, colors: list[int], masks: list[np.ndarray]) -> tuple[int, int, int, int]:
    width = 1 + len(colors) * (LEGEND_INNER + 1)
    legend = np.full((LEGEND_HEIGHT, width), BG, dtype=int)
    legend[0, :] = GREY
    legend[-1, :] = GREY
    for c in range(0, width, LEGEND_INNER + 1):
        legend[:, c] = GREY

    for idx, (color, mask) in enumerate(zip(colors, masks)):
        c0 = 1 + idx * (LEGEND_INNER + 1)
        patch = legend[1:-1, c0 : c0 + LEGEND_INNER]
        patch[mask] = int(color)

    grid[:LEGEND_HEIGHT, legend_left : legend_left + width] = legend
    return 0, legend_left, LEGEND_HEIGHT - 1, legend_left + width - 1


def _place_regions(
    regions: list[dict],
    legend_box: tuple[int, int, int, int],
) -> list[tuple[dict, int, int]] | None:
    placements = []
    occupied = [(legend_box[0], legend_box[1] - 1, legend_box[2] + 2, legend_box[3] + 1)]
    for region in regions:
        h, w = region["room"].shape
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(LEGEND_HEIGHT + 2, TEMP_CANVAS - h)
            c0 = random.randint(0, TEMP_CANVAS - w)
            box = (r0 - 1, c0 - 1, r0 + h, c0 + w)
            if any(_boxes_overlap(box, other) for other in occupied):
                continue
            occupied.append(box)
            placements.append((region, r0, c0))
            break
        else:
            return None
    return placements


def _build_expected(inp: np.ndarray, placements: list[tuple[dict, int, int]]) -> np.ndarray:
    out = inp.copy()
    for region, r0, c0 in placements:
        _paint_stamp(out, r0, c0, region["interior"], region["color"], region["stamp"])
    return out


def _crop_bounds(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != BG)
    assert coords.size > 0, "generated grid must contain foreground"
    _r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    left_margin = random.randint(0, min(2, int(c0)))
    right_margin = random.randint(0, min(2, TEMP_CANVAS - 1 - int(c1)))
    bottom_margin = random.randint(0, min(2, TEMP_CANVAS - 1 - int(r1)))
    return 0, int(r1) + bottom_margin + 1, int(c0) - left_margin, int(c1) + right_margin + 1


def _crop_canvas(grid: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    r0, r1, c0, c1 = bounds
    return grid[r0:r1, c0:c1].copy()


def _gen_attempt() -> np.ndarray | None:
    region_count = random.randint(*ROOM_COUNT_RANGE)
    distractor_count = random.randint(*DISTRACTOR_COUNT_RANGE)
    total_templates = region_count + distractor_count
    if total_templates > len(COLOR_POOL) or total_templates > len(TEMPLATE_MASKS):
        return None

    colors = random.sample(COLOR_POOL, total_templates)
    masks = [mask.copy() for mask in random.sample(TEMPLATE_MASKS, total_templates)]
    legend_order = list(range(total_templates))
    random.shuffle(legend_order)
    legend_colors = [colors[i] for i in legend_order]
    legend_masks = [masks[i] for i in legend_order]

    legend_width = 1 + total_templates * (LEGEND_INNER + 1)
    legend_left = random.randint(0, TEMP_CANVAS - legend_width)

    region_indices = random.sample(range(total_templates), region_count)
    regions = []
    for idx in region_indices:
        room_data = _random_room()
        if room_data is None:
            return None
        room, boundary, interior = room_data
        seed_r, seed_c = random.choice(np.argwhere(interior).tolist())
        regions.append(
            {
                "color": int(colors[idx]),
                "stamp": masks[idx],
                "room": room,
                "boundary": boundary,
                "interior": interior,
                "seed": (int(seed_r), int(seed_c)),
            }
        )

    temp = np.full((TEMP_CANVAS, TEMP_CANVAS), BG, dtype=int)
    legend_box = _render_legend(temp, legend_left, legend_colors, legend_masks)
    placements = _place_regions(regions, legend_box)
    if placements is None:
        return None

    for region, r0, c0 in placements:
        boundary = region["boundary"]
        temp[r0 : r0 + boundary.shape[0], c0 : c0 + boundary.shape[1]][boundary] = GREY
        sr, sc = region["seed"]
        temp[r0 + sr, c0 + sc] = region["color"]

    expected = _build_expected(temp, placements)
    bounds = _crop_bounds(temp)
    inp = _crop_canvas(temp, bounds)
    expected = _crop_canvas(expected, bounds)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-centering ARC input")
