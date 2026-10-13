from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GRAY = 5
MARKER_COLORS = (1, 2, 3)
ORTHO = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _color_components(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            touches_edge = False
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_edge = True
                for dr, dc in ORTHO:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.asarray(cells, dtype=int)
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(len(cells)),
                    "touches_edge": bool(touches_edge),
                }
            )
    return comps


def _template_from_component(component: dict) -> dict:
    r0, c0, r1, c1 = component["bbox"]
    local = component["coords"] - np.array([r0, c0], dtype=int)
    shape = (int(r1 - r0 + 1), int(c1 - c0 + 1))
    anchor = ((shape[0] - 1) // 2, (shape[1] - 1) // 2)
    return {
        "color": int(component["color"]),
        "bbox": component["bbox"],
        "coords": component["coords"].copy(),
        "local": local.astype(int),
        "shape": shape,
        "anchor": anchor,
        "area": int(component["area"]),
    }


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    top_by_color = {}
    left_by_color = {}
    for c in range(1, w):
        color = int(g[0, c])
        if color == BLACK:
            continue
        assert color in MARKER_COLORS, "unexpected top-edge color"
        assert color not in top_by_color, "top marker colors must be unique"
        top_by_color[color] = int(c)
    for r in range(1, h):
        color = int(g[r, 0])
        if color == BLACK:
            continue
        assert color in MARKER_COLORS, "unexpected left-edge color"
        assert color not in left_by_color, "left marker colors must be unique"
        left_by_color[color] = int(r)

    colors = sorted(top_by_color)
    assert 2 <= len(colors) <= 3, "expected 2-3 marker colors"
    assert set(colors) == set(left_by_color), "top and left marker colors must match"

    templates = {}
    sprite_components = {}
    for color in colors:
        comps = [comp for comp in _color_components(g, color) if not comp["touches_edge"]]
        assert len(comps) == 2, "each color must have exactly two internal sprites"
        areas = sorted(int(comp["area"]) for comp in comps)
        assert areas[0] < areas[1], "same-color sprites must have different sizes"
        big = max(comps, key=lambda comp: int(comp["area"]))
        templates[color] = _template_from_component(big)
        sprite_components[color] = comps

    return {
        "shape": (int(h), int(w)),
        "colors": colors,
        "top_by_color": top_by_color,
        "left_by_color": left_by_color,
        "templates": templates,
        "sprite_components": sprite_components,
    }


def _stamp_cells(template: dict, target: tuple[int, int], out_shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = out_shape
    tr, tc = target
    ar, ac = template["anchor"]
    cells = set()
    for rr, cc in template["local"]:
        r = int(tr - ar + int(rr))
        c = int(tc - ac + int(cc))
        if 0 <= r < h and 0 <= c < w:
            cells.add((r, c))
    return cells


def _render_scene(parsed: dict) -> np.ndarray:
    h, w = parsed["shape"]
    out = np.zeros((h, w), dtype=int)

    for color, col in parsed["top_by_color"].items():
        out[:, int(col)] = GRAY
    for color, row in parsed["left_by_color"].items():
        out[int(row), :] = GRAY

    colors_in_order = sorted(parsed["colors"], key=lambda color: parsed["left_by_color"][color])
    for color in colors_in_order:
        template = parsed["templates"][color]
        target = (parsed["left_by_color"][color], parsed["top_by_color"][color])
        for r, c in _stamp_cells(template, target, out.shape):
            out[r, c] = int(color)

    for color, col in parsed["top_by_color"].items():
        out[0, int(col)] = BLACK
    for color, row in parsed["left_by_color"].items():
        out[int(row), 0] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend same-color border rays, keep the larger sprite of each color, and stamp it onto the matching ray intersection.

    Input structure:
    - The grid has black background.
    - The top edge contains 2-3 singleton markers, and the left edge contains one singleton marker for each of the same colors.
    - In the lower-right area, each marker color has exactly two disconnected monochrome 4-connected sprites of different areas.

    Rule:
    - Draw a gray vertical ray down each top-marker column and a gray horizontal ray across each left-marker row.
    - Only intersections whose row-marker color matches the column-marker color become stamp targets.
    - For each color, discard the smaller sprite, use the larger sprite as a template, and align its bbox-center (rounded toward upper-left) to that target.
    - Stamp colors in increasing left-marker row order, clipping out-of-bounds cells, then erase all original markers and original sprite locations.

    Key insight:
    - The output targets are not all ray crossings: only same-color row/column pairs survive, and the kept sprite plus row order determines the final overlap pattern.
    """
    parsed = _parse_scene(np.asarray(grid, dtype=int))
    return _render_scene(parsed)


def _random_connected_shape(
    area: int,
    min_h: int,
    max_h: int,
    min_w: int,
    max_w: int,
    *,
    require_even_axis: bool = False,
    require_anchor_filled: bool = False,
    require_dim4: bool = False,
) -> np.ndarray:
    candidates = []
    for h in range(min_h, max_h + 1):
        for w in range(min_w, max_w + 1):
            if h * w < area:
                continue
            if require_even_axis and h % 2 == 1 and w % 2 == 1:
                continue
            if require_dim4 and h != 4 and w != 4:
                continue
            candidates.append((h, w))
    random.shuffle(candidates)

    for h, w in candidates:
        for _ in range(200):
            start = (random.randrange(h), random.randrange(w))
            cells = {start}
            while len(cells) < area:
                frontier = []
                for r, c in cells:
                    for dr, dc in ORTHO:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                            frontier.append((nr, nc))
                if not frontier:
                    break
                cells.add(random.choice(frontier))

            if len(cells) != area:
                continue

            coords = np.asarray(sorted(cells), dtype=int)
            r0, c0, r1, c1 = _bbox_coords(coords)
            coords[:, 0] -= r0
            coords[:, 1] -= c0
            th = int(r1 - r0 + 1)
            tw = int(c1 - c0 + 1)
            if area >= 5 and (th < 2 or tw < 2):
                continue
            if require_even_axis and th % 2 == 1 and tw % 2 == 1:
                continue
            if require_dim4 and th != 4 and tw != 4:
                continue
            if require_anchor_filled:
                anchor = ((th - 1) // 2, (tw - 1) // 2)
                if not np.any(np.all(coords == np.array(anchor, dtype=int), axis=1)):
                    continue
            return coords
    raise RuntimeError("failed to build connected shape")


def _shape_template(local_coords: np.ndarray, color: int) -> dict:
    coords = np.asarray(local_coords, dtype=int)
    bbox = _bbox_coords(coords)
    h = int(bbox[2] - bbox[0] + 1)
    w = int(bbox[3] - bbox[1] + 1)
    anchor = ((h - 1) // 2, (w - 1) // 2)
    return {
        "color": int(color),
        "local": coords.copy(),
        "shape": (h, w),
        "anchor": anchor,
        "area": int(len(coords)),
    }


def _can_place(local_coords: np.ndarray, top: int, left: int, blocked: np.ndarray, h: int, w: int, row_min: int, col_min: int) -> bool:
    cells = local_coords + np.array([top, left], dtype=int)
    if int(cells[:, 0].min()) < row_min or int(cells[:, 1].min()) < col_min:
        return False
    if int(cells[:, 0].max()) >= h - 1 or int(cells[:, 1].max()) >= w - 1:
        return False
    for r, c in cells:
        if blocked[int(r), int(c)]:
            return False
    return True


def _mark_blocked(blocked: np.ndarray, cells: np.ndarray) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, int(r) - 1), min(h, int(r) + 2)):
            for cc in range(max(0, int(c) - 1), min(w, int(c) + 2)):
                blocked[rr, cc] = True


def _sample_cluster_positions(count: int, upper: int) -> list[int]:
    for _ in range(200):
        start = random.randint(1, max(1, upper - 5))
        stop = min(upper, start + random.randint(max(2, count), 4))
        pool = list(range(start, stop + 1))
        if len(pool) >= count:
            return random.sample(pool, count)
    raise RuntimeError("failed to sample clustered positions")


def _build_spec() -> dict | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    color_count = random.choice([2, 2, 3, 3])
    colors = random.sample(list(MARKER_COLORS), color_count)

    clip_color = None
    if color_count == 3 and random.random() < 0.45:
        clip_color = random.choice(colors)

    cluster_colors = [color for color in colors if color != clip_color]
    cluster_row_limit = min(h - 5, 7)
    cluster_col_limit = min(w - 5, 7)
    if cluster_row_limit < len(cluster_colors) or cluster_col_limit < len(cluster_colors):
        return None

    left_by_color = {}
    top_by_color = {}
    for color, row in zip(random.sample(cluster_colors, len(cluster_colors)), _sample_cluster_positions(len(cluster_colors), cluster_row_limit)):
        left_by_color[color] = int(row)
    for color, col in zip(random.sample(cluster_colors, len(cluster_colors)), _sample_cluster_positions(len(cluster_colors), cluster_col_limit)):
        top_by_color[color] = int(col)

    if clip_color is not None:
        left_by_color[clip_color] = random.choice([h - 3, h - 2])
        top_by_color[clip_color] = random.choice([w - 3, w - 2])

    even_color = random.choice(colors) if random.random() < 0.6 else None
    templates = {}
    for color in colors:
        big_area = random.randint(5, 8)
        small_area = random.randint(1, min(3, big_area - 1))
        require_dim4 = color == clip_color
        big_local = _random_connected_shape(
            big_area,
            2,
            4,
            2,
            4,
            require_even_axis=(color == even_color),
            require_anchor_filled=True,
            require_dim4=require_dim4,
        )
        small_local = _random_connected_shape(small_area, 1, 3, 1, 3)
        templates[color] = {
            "big": _shape_template(big_local, color),
            "small": _shape_template(small_local, color),
        }

    blocked = np.zeros((h, w), dtype=bool)
    row_min = h // 2
    col_min = w // 2
    placements = {}
    items = [(color, kind) for color in colors for kind in ("big", "small")]
    random.shuffle(items)
    for color, kind in items:
        shape = templates[color][kind]
        sh, sw = shape["shape"]
        placed = False
        for _ in range(300):
            top = random.randint(row_min, h - sh - 1)
            left = random.randint(col_min, w - sw - 1)
            if not _can_place(shape["local"], top, left, blocked, h, w, row_min, col_min):
                continue
            cells = shape["local"] + np.array([top, left], dtype=int)
            _mark_blocked(blocked, cells)
            placements[(color, kind)] = (int(top), int(left))
            placed = True
            break
        if not placed:
            return None

    return {
        "shape": (h, w),
        "colors": colors,
        "top_by_color": top_by_color,
        "left_by_color": left_by_color,
        "templates": templates,
        "placements": placements,
        "clip_color": clip_color,
    }


def _render_input_from_spec(spec: dict) -> np.ndarray:
    h, w = spec["shape"]
    grid = np.zeros((h, w), dtype=int)

    for color in spec["colors"]:
        grid[0, spec["top_by_color"][color]] = int(color)
        grid[spec["left_by_color"][color], 0] = int(color)
        for kind in ("big", "small"):
            top, left = spec["placements"][(color, kind)]
            local = spec["templates"][color][kind]["local"]
            for rr, cc in local:
                grid[int(top + rr), int(left + cc)] = int(color)
    return grid


def _render_output_from_spec(spec: dict) -> tuple[np.ndarray, dict[int, set[tuple[int, int]]]]:
    h, w = spec["shape"]
    out = np.zeros((h, w), dtype=int)

    for color in spec["colors"]:
        out[:, spec["top_by_color"][color]] = GRAY
        out[spec["left_by_color"][color], :] = GRAY

    stamp_sets = {}
    order = sorted(spec["colors"], key=lambda color: spec["left_by_color"][color])
    for color in order:
        target = (spec["left_by_color"][color], spec["top_by_color"][color])
        cells = _stamp_cells(spec["templates"][color]["big"], target, (h, w))
        stamp_sets[color] = cells
        for r, c in cells:
            out[r, c] = int(color)

    for color in spec["colors"]:
        out[0, spec["top_by_color"][color]] = BLACK
        out[spec["left_by_color"][color], 0] = BLACK
    return out, stamp_sets


def _validate_spec(spec: dict, grid: np.ndarray, expected: np.ndarray, stamp_sets: dict[int, set[tuple[int, int]]]) -> bool:
    parsed = _parse_scene(grid)
    if parsed["top_by_color"] != spec["top_by_color"] or parsed["left_by_color"] != spec["left_by_color"]:
        return False

    for color in spec["colors"]:
        expected_areas = sorted(
            [
                int(spec["templates"][color]["big"]["area"]),
                int(spec["templates"][color]["small"]["area"]),
            ]
        )
        found_areas = sorted(int(comp["area"]) for comp in parsed["sprite_components"][color])
        if expected_areas != found_areas:
            return False

    order = sorted(spec["colors"], key=lambda color: spec["left_by_color"][color])
    overlap_ok = False
    for i, early in enumerate(order):
        for late in order[i + 1 :]:
            overlap = stamp_sets[early] & stamp_sets[late]
            if overlap:
                cell = next(iter(overlap))
                if int(expected[cell]) != int(late):
                    return False
                overlap_ok = True
    if not overlap_ok:
        return False

    all_stamp_cells = set().union(*stamp_sets.values()) if stamp_sets else set()
    visible_cross = []
    for row_color in spec["colors"]:
        for col_color in spec["colors"]:
            if row_color == col_color:
                continue
            cell = (spec["left_by_color"][row_color], spec["top_by_color"][col_color])
            if cell not in all_stamp_cells:
                visible_cross.append(cell)
    if not visible_cross:
        return False
    if any(int(expected[cell]) != GRAY for cell in visible_cross):
        return False

    if spec["clip_color"] is not None:
        clip_color = spec["clip_color"]
        full_area = int(spec["templates"][clip_color]["big"]["area"])
        if len(stamp_sets[clip_color]) >= full_area:
            return False

    actual = main(grid)
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            spec = _build_spec()
            if spec is None:
                continue
            grid = _render_input_from_spec(spec)
            expected, stamp_sets = _render_output_from_spec(spec)
            if _validate_spec(spec, grid, expected, stamp_sets):
                return grid
        except AssertionError:
            continue
        except RuntimeError:
            continue
    raise RuntimeError("failed to generate a valid input")
