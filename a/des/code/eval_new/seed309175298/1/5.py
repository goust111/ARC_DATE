from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 8
BOUNDARY = 0
MARKER = 9
PALETTE_POOL = (1, 2, 3, 4, 5, 6, 7)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _get_bg(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return Counter(corners).most_common(1)[0][0]


def _classify_shape(marker_coords: np.ndarray) -> str:
    rows = {int(r) for r in marker_coords[:, 0]}
    cols = {int(c) for c in marker_coords[:, 1]}
    if len(rows) == 1:
        return "H"
    if len(cols) == 1:
        return "V"
    return "L"


def _extract_problem(grid: np.ndarray) -> tuple[int, list[int], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 4 and g.shape[1] >= 11

    bg = _get_bg(g)
    palette = [int(v) for v in g[-1] if int(v) not in (bg, BOUNDARY, MARKER)]
    assert len(palette) >= 5
    assert len(set(palette)) == len(palette)

    body = g[:-1]
    comps = _components(body != BOUNDARY)
    h, w = body.shape
    regions: list[dict[str, object]] = []
    for comp in comps:
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue
        coords = np.array(comp, dtype=int)
        values = body[coords[:, 0], coords[:, 1]]
        marker_coords = coords[values == MARKER]
        assert marker_coords.size > 0
        shape = _classify_shape(marker_coords)
        regions.append(
            {
                "coords": coords,
                "marker_coords": marker_coords,
                "count": int(marker_coords.shape[0]),
                "shape": shape,
                "top": int(coords[:, 0].min()),
                "left": int(coords[:, 1].min()),
            }
        )

    assert regions
    return bg, palette, regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region using a palette lookup from marker count and a texture lookup from marker shape.

    Input structure:
    - Azure background surrounds several Black 1-pixel closed boundaries whose interiors contain only background and Maroon marker pixels.
    - The bottom row is a left-to-right palette of isolated 1x1 colors separated by background gaps; the first palette color stands for count 1, the second for count 2, and so on.
    - Inside each enclosed region, the Maroon markers form either a horizontal line, a vertical line, or an L shape.

    Rule:
    - For each enclosed region, count its Maroon markers and select the palette color at that 1-based index.
    - If the markers are horizontal, paint only every other interior row with that color starting from the region's topmost interior row; if vertical, paint every other interior column starting from the leftmost interior column; if L-shaped, fill the whole interior solid.
    - Only interior background pixels are texture-painted; Maroon markers themselves are always replaced by the chosen fill color, while Black boundaries and the palette row stay unchanged.

    Key insight:
    - The same Maroon markers encode two independent signals: quantity chooses the palette color, while geometric arrangement chooses the interior texture.
    """
    g = np.asarray(grid, dtype=int)
    bg, palette, regions = _extract_problem(g)

    out = g.copy()
    for region in regions:
        coords = np.asarray(region["coords"], dtype=int)
        count = int(region["count"])
        shape = str(region["shape"])
        top = int(region["top"])
        left = int(region["left"])
        fill_color = int(palette[count - 1])

        for r, c in coords:
            value = int(g[int(r), int(c)])
            if value == MARKER:
                out[int(r), int(c)] = fill_color
                continue
            if value != bg:
                continue
            if shape == "L":
                out[int(r), int(c)] = fill_color
            elif shape == "H" and (int(r) - top) % 2 == 0:
                out[int(r), int(c)] = fill_color
            elif shape == "V" and (int(c) - left) % 2 == 0:
                out[int(r), int(c)] = fill_color

    return out


def _shape_cycle(count: int, rng: random.Random) -> list[str]:
    base = ["H", "V", "L"]
    start = rng.randrange(len(base))
    shapes = [base[(start + i) % len(base)] for i in range(count)]
    rng.shuffle(shapes)
    return shapes


def _make_region_spec(shape: str, rng: random.Random) -> dict[str, int | str]:
    if shape == "H":
        count = rng.randint(1, 5)
        inner_h = rng.randint(3, 7)
        inner_w = rng.randint(max(3, count), min(8, max(3, count) + 3))
        return {"shape": shape, "count": count, "inner_h": inner_h, "inner_w": inner_w}

    if shape == "V":
        count = rng.randint(2, 5)
        inner_h = rng.randint(count, min(7, count + 2))
        inner_w = rng.randint(3, 7)
        return {"shape": shape, "count": count, "inner_h": inner_h, "inner_w": inner_w}

    count = rng.randint(3, 5)
    arm_v = rng.randint(2, count - 1)
    arm_h = count - arm_v + 1
    inner_h = rng.randint(max(3, arm_v), min(7, max(3, arm_v) + 2))
    inner_w = rng.randint(max(3, arm_h), min(7, max(3, arm_h) + 2))
    return {
        "shape": shape,
        "count": count,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "arm_v": arm_v,
        "arm_h": arm_h,
    }


def _split_shelves(specs: list[dict[str, int | str]], rng: random.Random) -> list[list[dict[str, int | str]]]:
    specs = specs[:]
    rng.shuffle(specs)
    n = len(specs)
    if n == 2 and rng.random() < 0.45:
        return [[specs[0]], [specs[1]]]
    if n == 3:
        cut = rng.choice([1, 2])
        return [specs[:cut], specs[cut:]]
    if n == 4:
        return [specs[:2], specs[2:]]
    return [specs]


def _marker_offsets(spec: dict[str, int | str], rng: random.Random) -> list[tuple[int, int]]:
    shape = str(spec["shape"])
    count = int(spec["count"])
    inner_h = int(spec["inner_h"])
    inner_w = int(spec["inner_w"])

    if shape == "H":
        row = rng.randrange(inner_h)
        start = rng.randrange(inner_w - count + 1)
        return [(row, start + i) for i in range(count)]

    if shape == "V":
        col = rng.randrange(inner_w)
        start = rng.randrange(inner_h - count + 1)
        return [(start + i, col) for i in range(count)]

    arm_v = int(spec["arm_v"])
    arm_h = int(spec["arm_h"])
    top = rng.randrange(inner_h - arm_v + 1)
    left = rng.randrange(inner_w - arm_h + 1)
    orient = rng.choice(["ul", "ur", "dl", "dr"])

    if orient == "ul":
        elbow_r, elbow_c = top, left
        vertical = [(elbow_r + i, elbow_c) for i in range(arm_v)]
        horizontal = [(elbow_r, elbow_c + i) for i in range(arm_h)]
    elif orient == "ur":
        elbow_r, elbow_c = top, left + arm_h - 1
        vertical = [(elbow_r + i, elbow_c) for i in range(arm_v)]
        horizontal = [(elbow_r, elbow_c - i) for i in range(arm_h)]
    elif orient == "dl":
        elbow_r, elbow_c = top + arm_v - 1, left
        vertical = [(elbow_r - i, elbow_c) for i in range(arm_v)]
        horizontal = [(elbow_r, elbow_c + i) for i in range(arm_h)]
    else:
        elbow_r, elbow_c = top + arm_v - 1, left + arm_h - 1
        vertical = [(elbow_r - i, elbow_c) for i in range(arm_v)]
        horizontal = [(elbow_r, elbow_c - i) for i in range(arm_h)]

    return sorted(set(vertical + horizontal))


def _draw_region(
    grid: np.ndarray,
    top: int,
    left: int,
    spec: dict[str, int | str],
    rng: random.Random,
) -> None:
    inner_h = int(spec["inner_h"])
    inner_w = int(spec["inner_w"])
    bottom = top + inner_h + 1
    right = left + inner_w + 1

    grid[top : bottom + 1, left] = BOUNDARY
    grid[top : bottom + 1, right] = BOUNDARY
    grid[top, left : right + 1] = BOUNDARY
    grid[bottom, left : right + 1] = BOUNDARY

    for dr, dc in _marker_offsets(spec, rng):
        grid[top + 1 + dr, left + 1 + dc] = MARKER


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    region_count = rng.randint(2, 4)
    specs = [_make_region_spec(shape, rng) for shape in _shape_cycle(region_count, rng)]

    for spec in specs:
        spec["outer_h"] = int(spec["inner_h"]) + 2
        spec["outer_w"] = int(spec["inner_w"]) + 2

    shelves = _split_shelves(specs, rng)
    shelf_heights = [max(int(spec["outer_h"]) for spec in shelf) for shelf in shelves]
    shelf_gaps = [rng.randint(2, 4) for _ in range(len(shelves) - 1)]

    shelf_widths: list[int] = []
    shelf_inner_gaps: list[list[int]] = []
    for shelf in shelves:
        gaps = [rng.randint(2, 4) for _ in range(len(shelf) - 1)]
        shelf_inner_gaps.append(gaps)
        shelf_widths.append(sum(int(spec["outer_w"]) for spec in shelf) + sum(gaps))

    palette_len = 2 * 5 - 1
    width = max(11, max(shelf_widths) + 2, palette_len + 2)
    if width > 30:
        return None
    width += rng.randint(0, min(4, 30 - width))

    top_margin = rng.randint(1, 3)
    body_height = top_margin + sum(shelf_heights) + sum(shelf_gaps) + 1
    height = body_height + 1
    if height > 30:
        return None
    height += rng.randint(0, min(3, 30 - height))

    grid = np.full((height, width), BG, dtype=int)

    row_cursor = top_margin
    placed_specs: list[dict[str, int | str]] = []
    for shelf_idx, shelf in enumerate(shelves):
        shelf_h = shelf_heights[shelf_idx]
        content_w = shelf_widths[shelf_idx]
        left_margin = rng.randint(1, width - content_w - 1)
        col_cursor = left_margin

        for item_idx, spec in enumerate(shelf):
            outer_h = int(spec["outer_h"])
            top = row_cursor + rng.randint(0, shelf_h - outer_h)
            left = col_cursor
            _draw_region(grid, top, left, spec, rng)
            placed_specs.append(spec)
            col_cursor += int(spec["outer_w"])
            if item_idx < len(shelf) - 1:
                col_cursor += shelf_inner_gaps[shelf_idx][item_idx]

        row_cursor += shelf_h
        if shelf_idx < len(shelves) - 1:
            row_cursor += shelf_gaps[shelf_idx]

    palette = rng.sample(PALETTE_POOL, 5)
    start = rng.randint(1, width - palette_len - 1)
    for idx, color in enumerate(palette):
        grid[-1, start + 2 * idx] = int(color)

    try:
        bg, parsed_palette, regions = _extract_problem(grid)
    except AssertionError:
        return None

    if bg != BG or parsed_palette[:5] != palette:
        return None
    if len(regions) != len(placed_specs):
        return None

    expected = sorted((int(spec["count"]), str(spec["shape"])) for spec in placed_specs)
    observed = sorted((int(region["count"]), str(region["shape"])) for region in regions)
    if expected != observed:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
