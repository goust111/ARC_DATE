from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste


BG = 0
SEP = 5
SPRITE_SIZE = 3
TOP_HEIGHT = 7
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_ATTEMPTS = 500


def _solid_separator_row(grid: np.ndarray, bg: int) -> int:
    rows = [
        r
        for r in range(grid.shape[0])
        if np.all(grid[r] == int(grid[r, 0])) and int(grid[r, 0]) != int(bg)
    ]
    assert len(rows) == 1, "expected one solid separator row"
    return int(rows[0])


def _extract_legend(top: np.ndarray, bg: int) -> list[tuple[int, np.ndarray]]:
    comps = find_components(top, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    sprites = []
    markers = []
    for comp in comps:
        area = int(comp["area"])
        if area == 1:
            markers.append(comp)
        else:
            patch = crop_bbox(top, comp["bbox"])
            sprites.append((comp, patch))

    assert len(sprites) == len(markers) and len(sprites) >= 2
    markers.sort(key=lambda comp: (int(comp["coords"][0][1]), int(comp["coords"][0][0])))
    sprites.sort(key=lambda item: (int(item[0]["bbox"][1]), int(item[0]["bbox"][0])))

    shapes = {patch.shape for _comp, patch in sprites}
    assert len(shapes) == 1, "legend sprites must share one bbox size"
    shape = next(iter(shapes))
    assert shape[0] % 2 == 1 and shape[1] % 2 == 1, "sprite bbox must have a center"

    legend = []
    for marker, (sprite_comp, patch) in zip(markers, sprites):
        r0, c0, r1, c1 = map(int, sprite_comp["bbox"])
        mr, mc = map(int, marker["coords"][0])
        assert mr == r1 + 2, "marker must be one row below its sprite"
        assert mc == (c0 + c1) // 2, "marker must align to sprite center column"
        legend.append((int(marker["color"]), patch.copy()))
    return legend


def _center_stamp(
    out: np.ndarray,
    patch: np.ndarray,
    center: tuple[int, int],
    endpoint_map: dict[tuple[int, int], int],
) -> None:
    h, w = patch.shape
    r0 = int(center[0]) - h // 2
    c0 = int(center[1]) - w // 2
    for pr in range(h):
        for pc in range(w):
            color = int(patch[pr, pc])
            if color == BG:
                continue
            rr = r0 + pr
            cc = c0 + pc
            assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1], "stamp must fit in output"
            if (rr, cc) in endpoint_map:
                continue
            out[rr, cc] = color


def _solve_from_parts(legend: list[tuple[int, np.ndarray]], lower: np.ndarray) -> np.ndarray:
    out = lower.copy()
    endpoint_map = {
        (int(r), int(c)): int(lower[r, c])
        for r, c in np.argwhere(lower != BG)
    }

    for marker_color, sprite in legend:
        coords = np.argwhere(lower == marker_color)
        assert coords.shape[0] == 2, "each marker color must have exactly two endpoints"
        coords = coords[np.argsort(coords[:, 0], kind="stable")]
        (ar, ac), (br, bc) = [tuple(map(int, coord)) for coord in coords]
        assert ar < br and ac != bc, "endpoints must differ in both row and column"

        c0 = min(ac, bc)
        c1 = max(ac, bc)
        out[ar, c0 : c1 + 1] = marker_color
        out[ar : br + 1, bc] = marker_color

        _center_stamp(out, sprite, (ar, bc), endpoint_map)

    for (r, c), color in endpoint_map.items():
        out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw each lower-field right-angle path and stamp its paired legend sprite at the path corner.

    Input structure:
    - The grid has a black background, a top legend of equal-size irregular monochrome sprites, singleton marker pixels below them, and one solid separator row.
    - Each legend marker is centered under one sprite; below the separator the operation field contains exactly two endpoints for each marker color.
    - The lower endpoints of one color are in different rows and different columns, and sprite stamps fit fully inside the lower field.

    Rule:
    - Pair legend sprites with marker colors by left-to-right order.
    - Discard the legend and separator, keeping only the lower field size for the output.
    - For each marker color in legend order, take the upper endpoint A and lower endpoint B, draw the color horizontally along A's row to B's column and then vertically down that column to B, and stamp the paired sprite centered on the corner at (A.row, B.col) without overwriting any endpoint pixels.

    Key insight:
    - The lower output needs both parts of the top legend: marker order says which sprite belongs to which endpoint color, while the path rule creates an emergent corner location that is not marked directly in the input.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    sep = _solid_separator_row(g, bg)
    legend = _extract_legend(g[:sep], bg)
    lower = g[sep + 1 :].copy()
    return _solve_from_parts(legend, lower)


def _mask_connected(mask: np.ndarray) -> bool:
    coords = {tuple(map(int, coord)) for coord in np.argwhere(mask)}
    if not coords:
        return False
    stack = [next(iter(coords))]
    seen = set()
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        r, c = cell
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in coords and (nr, nc) not in seen:
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _random_sprite_mask() -> np.ndarray:
    for _ in range(200):
        target = random.randint(4, 7)
        mask = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
        mask[1, 1] = True
        while int(mask.sum()) < target:
            frontier = []
            for r in range(SPRITE_SIZE):
                for c in range(SPRITE_SIZE):
                    if mask[r, c]:
                        continue
                    if (
                        (r > 0 and mask[r - 1, c])
                        or (r + 1 < SPRITE_SIZE and mask[r + 1, c])
                        or (c > 0 and mask[r, c - 1])
                        or (c + 1 < SPRITE_SIZE and mask[r, c + 1])
                    ):
                        frontier.append((r, c))
            if not frontier:
                break
            r, c = random.choice(frontier)
            mask[r, c] = True
        if not _mask_connected(mask):
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if np.all(mask):
            continue
        return mask
    raise RuntimeError("failed to generate a valid sprite mask")


def _legend_layout(width: int, count: int) -> list[int] | None:
    gaps = [random.randint(2, 4) for _ in range(count - 1)]
    total = count * SPRITE_SIZE + sum(gaps)
    if total > width:
        return None
    left = random.randint(0, width - total)
    starts = [left]
    for gap in gaps:
        starts.append(starts[-1] + SPRITE_SIZE + gap)
    return starts


def _path_cells(item: dict) -> set[tuple[int, int]]:
    ar, ac = item["a"]
    br, bc = item["b"]
    cells = {(ar, c) for c in range(min(ac, bc), max(ac, bc) + 1)}
    cells |= {(r, bc) for r in range(ar, br + 1)}
    return cells


def _stamp_cells(item: dict) -> set[tuple[int, int]]:
    corner_r, corner_c = item["corner"]
    cells = set()
    for r, c in np.argwhere(item["mask"]):
        cells.add((corner_r + int(r) - 1, corner_c + int(c) - 1))
    return cells


def _build_expected(lower_shape: tuple[int, int], items: list[dict]) -> np.ndarray:
    out = np.full(lower_shape, BG, dtype=int)
    legend = []
    for item in items:
        ar, ac = item["a"]
        br, bc = item["b"]
        out[ar, ac] = item["marker_color"]
        out[br, bc] = item["marker_color"]
        sprite = np.where(item["mask"], item["sprite_color"], BG).astype(int)
        legend.append((item["marker_color"], sprite))
    lower = out.copy()
    return _solve_from_parts(legend, lower)


def _validate_items(lower_shape: tuple[int, int], items: list[dict]) -> bool:
    endpoint_positions = set()
    directions = set()
    corners = set()
    for item in items:
        ar, ac = item["a"]
        br, bc = item["b"]
        if not (0 <= ar < br < lower_shape[0] and 0 <= ac < lower_shape[1] and 0 <= bc < lower_shape[1]):
            return False
        if abs(ac - bc) < 2 or abs(br - ar) < 2:
            return False
        directions.add(1 if ac < bc else -1)
        corners.add(item["corner"])
        endpoint_positions.add((ar, ac))
        endpoint_positions.add((br, bc))

    if len(corners) != len(items):
        return False
    if len(endpoint_positions) != 2 * len(items):
        return False
    if len(items) >= 2 and directions != {-1, 1}:
        return False

    path_sets = [_path_cells(item) for item in items]
    for i in range(len(path_sets)):
        for j in range(i + 1, len(path_sets)):
            if path_sets[i] & path_sets[j]:
                return False

    overlap_found = False
    painted = set()
    for idx, item in enumerate(items):
        stamp = _stamp_cells(item)
        if any(not (0 <= r < lower_shape[0] and 0 <= c < lower_shape[1]) for r, c in stamp):
            return False
        if stamp & endpoint_positions:
            return False
        if idx > 0 and stamp & painted:
            overlap_found = True
        painted |= path_sets[idx] - endpoint_positions
        painted |= stamp

    return overlap_found


def _make_items(lower_shape: tuple[int, int], count: int, sprite_masks: list[np.ndarray], colors: list[int]) -> list[dict] | None:
    height, width = lower_shape
    if count == 2:
        directions = [-1, 1]
    else:
        directions = [-1, 1, random.choice([-1, 1])]
        random.shuffle(directions)
    items = []
    for idx in range(count):
        direction = directions[idx]
        for _ in range(200):
            corner_r = random.randint(1, height - 3)
            corner_c = random.randint(1, width - 2)
            if direction > 0:
                if corner_c < 2:
                    continue
                ac = random.randint(0, corner_c - 2)
            else:
                if corner_c > width - 3:
                    continue
                ac = random.randint(corner_c + 2, width - 1)
            br = random.randint(corner_r + 2, height - 1)
            item = {
                "marker_color": colors[idx],
                "sprite_color": colors[count + idx],
                "mask": sprite_masks[idx],
                "a": (corner_r, ac),
                "b": (br, corner_c),
                "corner": (corner_r, corner_c),
            }
            path = _path_cells(item)
            if any(path & _path_cells(prev) for prev in items):
                continue
            stamp = _stamp_cells(item)
            endpoints = {(item["a"][0], item["a"][1]), (item["b"][0], item["b"][1])}
            if stamp & endpoints:
                continue
            other_endpoints = {pt for prev in items for pt in (prev["a"], prev["b"])}
            if stamp & other_endpoints:
                continue
            if any((pt in path) for pt in other_endpoints):
                continue
            items.append(item)
            break
        else:
            return None

    if not _validate_items(lower_shape, items):
        return None
    return items


def _build_input(lower_shape: tuple[int, int], items: list[dict], legend_starts: list[int], width: int) -> np.ndarray:
    legend = np.full((TOP_HEIGHT, width), BG, dtype=int)
    for item, c0 in zip(items, legend_starts):
        sprite = np.where(item["mask"], item["sprite_color"], BG).astype(int)
        paste(legend, sprite, 0, c0, transparent=BG)
        legend[4, c0 + 1] = item["marker_color"]

    lower = np.full(lower_shape, BG, dtype=int)
    for item in items:
        ar, ac = item["a"]
        br, bc = item["b"]
        lower[ar, ac] = item["marker_color"]
        lower[br, bc] = item["marker_color"]

    sep = np.full((1, width), SEP, dtype=int)
    return np.vstack([legend, sep, lower])


def _gen_attempt() -> np.ndarray | None:
    count = random.choice([2, 3])
    width = random.randint(13 if count == 2 else 15, 19)
    height = random.randint(9, 14)
    legend_starts = _legend_layout(width, count)
    if legend_starts is None:
        return None

    picked = random.sample(PALETTE, count * 2)
    marker_colors = picked[:count]
    sprite_colors = picked[count:]
    sprite_masks = []
    seen_masks = set()
    for _ in range(count):
        for _retry in range(100):
            mask = _random_sprite_mask()
            key = tuple(mask.ravel().tolist())
            if key not in seen_masks:
                seen_masks.add(key)
                sprite_masks.append(mask)
                break
        else:
            return None

    colors = marker_colors + sprite_colors
    items = _make_items((height, width), count, sprite_masks, colors)
    if items is None:
        return None
    for item, marker_color, sprite_color in zip(items, marker_colors, sprite_colors):
        item["marker_color"] = marker_color
        item["sprite_color"] = sprite_color

    inp = _build_input((height, width), items, legend_starts, width)
    expected = _build_expected((height, width), items)

    sep = _solid_separator_row(inp, BG)
    parsed_legend = _extract_legend(inp[:sep], BG)
    if [color for color, _patch in parsed_legend] != marker_colors:
        return None
    if not np.array_equal(main(inp), expected):
        return None

    out = expected
    if np.unique(out[out != BG]).size < count:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid corner-stamp ARC input")
