from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

Coord = tuple[int, int]

DIRS4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8: tuple[Coord, ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

BG = 5
GRID_RANGE = (14, 17)
SPRITE_COUNT_CHOICES = (2, 3)
DISTANCE_RANGE = (2, 5)
MAX_GENERATION_ATTEMPTS = 400
COMPONENT_ATTEMPTS = 200

BASE_DOWN_TEMPLATES: tuple[tuple[Coord, ...], ...] = (
    ((-1, 0), (-2, 0), (-1, -1)),
    ((-1, 0), (-2, 0), (-1, -1), (-1, 1)),
    ((-1, 0), (-1, -1), (-2, -1), (-2, 0)),
    ((-1, 0), (-2, 0), (-3, 0), (-2, -1), (-2, 1)),
)


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _neighbors8(r: int, c: int):
    for dr, dc in DIRS8:
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _find_components(grid: np.ndarray, bg: int) -> list[list[Coord]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[Coord] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not _in_bounds((h, w), nr, nc):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) == bg:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append(cells)
    return comps


def _is_border(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _is_anchor(grid: np.ndarray, bg: int, cell: Coord) -> bool:
    h, w = grid.shape
    r, c = cell
    if not _is_border(r, c, h, w):
        return False
    if int(grid[r, c]) == bg:
        return False
    for nr, nc in _neighbors8(r, c):
        if _in_bounds((h, w), nr, nc) and int(grid[nr, nc]) != bg:
            return False
    return True


def _anchor_side(anchor: Coord, h: int, w: int) -> str:
    r, c = anchor
    if c == 0:
        return "left"
    if c == w - 1:
        return "right"
    if r == 0:
        return "top"
    if r == h - 1:
        return "bottom"
    raise AssertionError("anchor must lie on the border")


def _side_to_ray_dir(side: str) -> Coord:
    return {
        "top": (1, 0),
        "bottom": (-1, 0),
        "left": (0, 1),
        "right": (0, -1),
    }[side]


def _slide_delta(side: str, marker: Coord, anchor: Coord) -> tuple[int, int, int]:
    mr, mc = marker
    ar, ac = anchor
    if side in ("top", "bottom"):
        dc = ac - mc
        return 0, dc, abs(dc)
    dr = ar - mr
    return dr, 0, abs(dr)


def _rotate_cw(cells: tuple[Coord, ...]) -> tuple[Coord, ...]:
    return tuple((c, -r) for r, c in cells)


def _rotate_ccw(cells: tuple[Coord, ...]) -> tuple[Coord, ...]:
    return tuple((-c, r) for r, c in cells)


def _reflect_lr(cells: tuple[Coord, ...]) -> tuple[Coord, ...]:
    return tuple((r, -c) for r, c in cells)


def _dedupe_templates(templates: tuple[tuple[Coord, ...], ...]) -> tuple[tuple[Coord, ...], ...]:
    seen: set[tuple[Coord, ...]] = set()
    out: list[tuple[Coord, ...]] = []
    for cells in templates:
        normalized = tuple(sorted(cells))
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return tuple(out)


def _oriented_templates(ray_dir: Coord) -> tuple[tuple[Coord, ...], ...]:
    base_variants = _dedupe_templates(BASE_DOWN_TEMPLATES + tuple(_reflect_lr(t) for t in BASE_DOWN_TEMPLATES))
    if ray_dir == (1, 0):
        return base_variants
    if ray_dir == (0, 1):
        return _dedupe_templates(tuple(_rotate_ccw(t) for t in base_variants))
    if ray_dir == (-1, 0):
        return _dedupe_templates(tuple(_rotate_cw(_rotate_cw(t)) for t in base_variants))
    if ray_dir == (0, -1):
        return _dedupe_templates(tuple(_rotate_cw(t) for t in base_variants))
    raise AssertionError("unexpected ray direction")


def _halo(cells: set[Coord], shape: tuple[int, int]) -> set[Coord]:
    h, w = shape
    out = set(cells)
    for r, c in cells:
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                out.add((nr, nc))
    return out


def _translate_pixels(pixels: list[tuple[int, int, int]], dr: int, dc: int) -> list[tuple[int, int, int]]:
    return [(r + dr, c + dc, color) for r, c, color in pixels]


def _ray_to_blocker(start: Coord, ray_dir: Coord, grid: np.ndarray, bg: int) -> list[Coord]:
    h, w = grid.shape
    dr, dc = ray_dir
    r, c = start
    cells: list[Coord] = []
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == bg:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _choose_marker_color(grid: np.ndarray, coords: list[Coord], anchor_colors: set[int]) -> tuple[int, Coord]:
    color_to_cells: dict[int, list[Coord]] = {}
    for r, c in coords:
        color_to_cells.setdefault(int(grid[r, c]), []).append((r, c))
    ranked = sorted(
        (
            (len(cells), color, cells)
            for color, cells in color_to_cells.items()
            if color in anchor_colors
        ),
        key=lambda item: (item[0], item[1]),
    )
    assert ranked, "each sprite must contain a marker color matching one anchor"
    best_count = ranked[0][0]
    best = [item for item in ranked if item[0] == best_count]
    assert len(best) == 1, "marker color must be uniquely identifiable"
    marker_count, marker_color, marker_cells = best[0]
    assert marker_count == 1 and len(marker_cells) == 1, "marker must be a unique pixel"
    return int(marker_color), marker_cells[0]


def _infer_body_color(grid: np.ndarray, coords: list[Coord], marker_color: int) -> int:
    counts = Counter(int(grid[r, c]) for r, c in coords if int(grid[r, c]) != marker_color)
    assert counts, "sprite body must contain at least one non-marker pixel"
    ranked = counts.most_common()
    if len(ranked) > 1:
        assert ranked[0][1] > ranked[1][1], "body color must be the unique non-marker majority"
    return int(ranked[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each marker-bearing sprite to its matching border anchor, then cast a reverse ray whose tail length equals the slide distance.

    Input structure:
    - The background is the dominant gray field.
    - The interior contains 2-3 small 4-connected multi-color sprites; each sprite has exactly one minority-color marker pixel.
    - The border contains isolated anchor pixels whose colors uniquely match the sprite marker colors.

    Rule:
    - Pair every sprite with the anchor of the same marker color.
    - If the anchor lies on the top or bottom edge, slide the whole sprite horizontally until the marker column matches the anchor column; if the anchor lies on the left or right edge, slide it vertically until the marker row matches the anchor row.
    - Let D be the absolute slide distance along that axis, move the whole sprite, then cast a ray from the moved marker in the direction opposite the anchor edge using the sprite's majority body color until the next non-background cell or the grid edge.
    - Recolor the last D cells of that ray, counted back from the far end toward the marker, with the anchor/marker color; if the ray is shorter than D, recolor the entire ray.

    Key insight:
    - The alignment step is not just a translation: the slide distance extracted from anchor alignment becomes the tail-length parameter for the later ray recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _get_bg(g)

    components = _find_components(g, bg)
    anchor_by_color: dict[int, Coord] = {}
    sprite_components: list[list[Coord]] = []

    for coords in components:
        if len(coords) == 1 and _is_anchor(g, bg, coords[0]):
            r, c = coords[0]
            color = int(g[r, c])
            assert color not in anchor_by_color, "anchor colors must be unique"
            anchor_by_color[color] = (r, c)
        else:
            sprite_components.append(coords)

    assert anchor_by_color, "expected at least one anchor"
    assert sprite_components, "expected at least one sprite"
    assert len(anchor_by_color) == len(sprite_components), "anchors and sprites must pair one-to-one"

    records = []
    anchor_colors = set(anchor_by_color)
    used_marker_colors: set[int] = set()
    for coords in sprite_components:
        marker_color, marker_cell = _choose_marker_color(g, coords, anchor_colors)
        assert marker_color not in used_marker_colors, "each anchor color should match exactly one sprite"
        used_marker_colors.add(marker_color)
        body_color = _infer_body_color(g, coords, marker_color)
        anchor = anchor_by_color[marker_color]
        side = _anchor_side(anchor, h, w)
        dr, dc, distance = _slide_delta(side, marker_cell, anchor)
        assert distance >= 1, "slide distance must be positive"
        moved_pixels = [(r + dr, c + dc, int(g[r, c])) for r, c in coords]
        moved_marker = (marker_cell[0] + dr, marker_cell[1] + dc)
        assert all(_in_bounds((h, w), r, c) for r, c, _color in moved_pixels), "moved sprite must stay in-grid"
        records.append(
            {
                "source": coords,
                "moved_pixels": moved_pixels,
                "marker_color": marker_color,
                "body_color": body_color,
                "moved_marker": moved_marker,
                "distance": distance,
                "ray_dir": _side_to_ray_dir(side),
            }
        )

    out = g.copy()
    for record in records:
        for r, c in record["source"]:
            out[r, c] = bg

    for record in records:
        for r, c, color in record["moved_pixels"]:
            assert int(out[r, c]) == bg, "slid sprites must not overlap"
            out[r, c] = color

    for record in records:
        ray = _ray_to_blocker(record["moved_marker"], record["ray_dir"], out, bg)
        for r, c in ray:
            out[r, c] = int(record["body_color"])
        tail = ray[-min(len(ray), int(record["distance"])) :]
        for r, c in tail:
            out[r, c] = int(record["marker_color"])

    return out


def _paint_pixels(grid: np.ndarray, pixels: list[tuple[int, int, int]]) -> None:
    for r, c, color in pixels:
        grid[r, c] = int(color)


def _row_span(rel_cells: tuple[Coord, ...]) -> tuple[int, int]:
    rows = [r for r, _ in rel_cells]
    return min(rows), max(rows)


def _col_span(rel_cells: tuple[Coord, ...]) -> tuple[int, int]:
    cols = [c for _r, c in rel_cells]
    return min(cols), max(cols)


def _candidate_markers(
    side: str,
    anchor: Coord,
    rel_cells: tuple[Coord, ...],
    distance: int,
    mode: str,
    shape: tuple[int, int],
) -> list[Coord]:
    h, w = shape
    min_r, max_r = _row_span(rel_cells)
    min_c, max_c = _col_span(rel_cells)
    out: list[Coord] = []

    if side in ("top", "bottom"):
        anchor_col = anchor[1]
        if not (1 - min_c <= anchor_col <= (w - 2) - max_c):
            return out
        row_lo = max(1 - min_r, 1)
        row_hi = min((h - 2) - max_r, h - 2)
        for row in range(row_lo, row_hi + 1):
            ray_len = (h - 1 - row) if side == "top" else row
            if ray_len <= 0:
                continue
            if mode == "partial" and not (ray_len > distance):
                continue
            if mode == "full" and not (ray_len < distance):
                continue
            if mode == "equal" and ray_len != distance:
                continue
            out.append((row, anchor_col))
    else:
        anchor_row = anchor[0]
        if not (1 - min_r <= anchor_row <= (h - 2) - max_r):
            return out
        col_lo = max(1 - min_c, 1)
        col_hi = min((w - 2) - max_c, w - 2)
        for col in range(col_lo, col_hi + 1):
            ray_len = (w - 1 - col) if side == "left" else col
            if ray_len <= 0:
                continue
            if mode == "partial" and not (ray_len > distance):
                continue
            if mode == "full" and not (ray_len < distance):
                continue
            if mode == "equal" and ray_len != distance:
                continue
            out.append((anchor_row, col))
    return out


def _sprite_pixels_from_template(
    marker: Coord,
    rel_cells: tuple[Coord, ...],
    marker_color: int,
    body_color: int,
) -> list[tuple[int, int, int]]:
    mr, mc = marker
    pixels = [(mr, mc, marker_color)]
    for dr, dc in rel_cells:
        pixels.append((mr + dr, mc + dc, body_color))
    return pixels


def _choose_sides(count: int) -> list[str]:
    horizontal = random.choice([["top"], ["bottom"], ["top", "bottom"]])
    vertical = random.choice([["left"], ["right"], ["left", "right"]])
    if count == 2:
        return [random.choice(horizontal), random.choice(vertical)]
    pool = list({*horizontal, *vertical})
    if len(pool) < 3:
        remaining = [side for side in ("top", "bottom", "left", "right") if side not in pool]
        pool.append(random.choice(remaining))
    random.shuffle(pool)
    return pool[:3]


def _choose_anchor_positions(h: int, w: int, sides: list[str]) -> dict[str, Coord] | None:
    top_bottom_cols = list(range(2, w - 2))
    left_right_rows = list(range(2, h - 2))
    if len(top_bottom_cols) < len([s for s in sides if s in ("top", "bottom")]):
        return None
    if len(left_right_rows) < len([s for s in sides if s in ("left", "right")]):
        return None
    random.shuffle(top_bottom_cols)
    random.shuffle(left_right_rows)

    pos: dict[str, Coord] = {}
    tb_idx = 0
    lr_idx = 0
    for side in sides:
        if side == "top":
            pos[side] = (0, top_bottom_cols[tb_idx])
            tb_idx += 1
        elif side == "bottom":
            pos[side] = (h - 1, top_bottom_cols[tb_idx])
            tb_idx += 1
        elif side == "left":
            pos[side] = (left_right_rows[lr_idx], 0)
            lr_idx += 1
        else:
            pos[side] = (left_right_rows[lr_idx], w - 1)
            lr_idx += 1
    return pos


def _reserve_component(
    cells: set[Coord],
    blocked: set[Coord],
    occupied: set[Coord],
    shape: tuple[int, int],
) -> tuple[set[Coord], set[Coord]] | None:
    if cells & blocked:
        return None
    new_occupied = set(occupied) | cells
    new_blocked = set(blocked) | _halo(cells, shape)
    return new_blocked, new_occupied


def _sample_sprite_record(
    h: int,
    w: int,
    side: str,
    anchor: Coord,
    marker_color: int,
    body_color: int,
    mode: str,
    settled_blocked: set[Coord],
    settled_occupied: set[Coord],
    input_blocked: set[Coord],
    input_occupied: set[Coord],
) -> dict | None:
    shape = (h, w)
    ray_dir = _side_to_ray_dir(side)
    templates = list(_oriented_templates(ray_dir))
    for _ in range(COMPONENT_ATTEMPTS):
        rel_cells = random.choice(templates)
        distance = random.randint(*DISTANCE_RANGE)
        marker_options = _candidate_markers(side, anchor, rel_cells, distance, mode, shape)
        if not marker_options:
            continue
        marker = random.choice(marker_options)
        settled_pixels = _sprite_pixels_from_template(marker, rel_cells, marker_color, body_color)
        settled_cells = {(r, c) for r, c, _color in settled_pixels}
        if any(not (1 <= r < h - 1 and 1 <= c < w - 1) for r, c in settled_cells):
            continue

        reserved = _reserve_component(settled_cells, settled_blocked, settled_occupied, shape)
        if reserved is None:
            continue
        next_settled_blocked, next_settled_occupied = reserved

        settled_grid = np.full((h, w), BG, dtype=int)
        for r, c in settled_occupied:
            settled_grid[r, c] = 1
        _paint_pixels(settled_grid, settled_pixels)
        ray_cells = _ray_to_blocker(marker, ray_dir, settled_grid, BG)
        if not ray_cells:
            continue
        if mode == "partial" and not (len(ray_cells) > distance):
            continue
        if mode == "full" and not (len(ray_cells) < distance):
            continue
        if mode == "equal" and len(ray_cells) != distance:
            continue

        ray_set = set(ray_cells)
        other_occupied = next_settled_occupied - settled_cells
        if ray_set & other_occupied:
            continue
        ray_clearance = _halo(ray_set, shape) - settled_cells - {marker}
        if ray_clearance & other_occupied:
            continue

        slide_signs = [-1, 1]
        random.shuffle(slide_signs)
        chosen_input = None
        for sign in slide_signs:
            if side in ("top", "bottom"):
                dr, dc = 0, sign * distance
            else:
                dr, dc = sign * distance, 0
            input_pixels = _translate_pixels(settled_pixels, dr, dc)
            input_cells = {(r, c) for r, c, _color in input_pixels}
            if any(not (1 <= r < h - 1 and 1 <= c < w - 1) for r, c in input_cells):
                continue
            reserved_input = _reserve_component(input_cells, input_blocked, input_occupied, shape)
            if reserved_input is None:
                continue
            input_blocked2, input_occupied2 = reserved_input
            chosen_input = (input_pixels, input_cells, input_blocked2, input_occupied2, sign)
            break
        if chosen_input is None:
            continue

        input_pixels, input_cells, input_blocked2, input_occupied2, sign = chosen_input
        settled_blocked2 = next_settled_blocked | _halo(ray_set, shape)
        settled_occupied2 = next_settled_occupied | ray_set
        return {
            "side": side,
            "anchor": anchor,
            "marker_color": marker_color,
            "body_color": body_color,
            "distance": distance,
            "mode": mode,
            "ray_dir": ray_dir,
            "settled_pixels": settled_pixels,
            "settled_cells": settled_cells,
            "settled_marker": marker,
            "input_pixels": input_pixels,
            "input_cells": input_cells,
            "ray_cells": ray_cells,
            "input_shift_sign": sign,
            "settled_blocked": settled_blocked2,
            "settled_occupied": settled_occupied2,
            "input_blocked": input_blocked2,
            "input_occupied": input_occupied2,
        }
    return None


def _build_expected_output(h: int, w: int, anchors: list[tuple[int, int, int]], records: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    _paint_pixels(grid, anchors)
    for record in records:
        _paint_pixels(grid, record["settled_pixels"])
    for record in records:
        for r, c in record["ray_cells"]:
            grid[r, c] = int(record["body_color"])
        tail = record["ray_cells"][-min(len(record["ray_cells"]), int(record["distance"])) :]
        for r, c in tail:
            grid[r, c] = int(record["marker_color"])
    return grid


def _semantic_checks(inp: np.ndarray, expected_out: np.ndarray, records: list[dict]) -> bool:
    if np.array_equal(inp, expected_out):
        return False
    if len({record["distance"] for record in records}) < 2:
        return False
    if len(records) >= 2:
        sides = {record["side"] for record in records}
        if not (sides & {"top", "bottom"} and sides & {"left", "right"}):
            return False
        modes = {record["mode"] for record in records}
        if "partial" not in modes or "full" not in modes:
            return False
    for record in records:
        if len(record["ray_cells"]) == 0:
            return False
        if record["mode"] == "partial" and not (len(record["ray_cells"]) > record["distance"]):
            return False
        if record["mode"] == "full" and not (len(record["ray_cells"]) < record["distance"]):
            return False
        if record["mode"] == "equal" and len(record["ray_cells"]) != record["distance"]:
            return False
    return np.array_equal(main(inp), expected_out)


def _generate_once() -> np.ndarray | None:
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    count = random.choice(SPRITE_COUNT_CHOICES)
    sides = _choose_sides(count)
    anchors_by_side = _choose_anchor_positions(h, w, sides)
    if anchors_by_side is None:
        return None

    palette = [color for color in range(10) if color != BG]
    random.shuffle(palette)
    marker_colors = palette[:count]
    body_colors = palette[count : 2 * count]
    if len(body_colors) < count:
        return None

    modes = ["partial", "full"]
    if count == 3:
        modes.append(random.choice(["equal", "partial", "full"]))
    random.shuffle(modes)

    anchor_pixels = []
    for side, color in zip(sides, marker_colors):
        r, c = anchors_by_side[side]
        anchor_pixels.append((r, c, color))

    shape = (h, w)
    settled_blocked: set[Coord] = set()
    settled_occupied: set[Coord] = set()
    input_blocked: set[Coord] = set()
    input_occupied: set[Coord] = set()
    for r, c, _color in anchor_pixels:
        reserved = _reserve_component({(r, c)}, settled_blocked, settled_occupied, shape)
        if reserved is None:
            return None
        settled_blocked, settled_occupied = reserved
        reserved_input = _reserve_component({(r, c)}, input_blocked, input_occupied, shape)
        if reserved_input is None:
            return None
        input_blocked, input_occupied = reserved_input

    records: list[dict] = []
    items = list(zip(sides, marker_colors, body_colors, modes))
    random.shuffle(items)
    for side, marker_color, body_color, mode in items:
        record = _sample_sprite_record(
            h,
            w,
            side,
            anchors_by_side[side],
            marker_color,
            body_color,
            mode,
            settled_blocked,
            settled_occupied,
            input_blocked,
            input_occupied,
        )
        if record is None:
            return None
        settled_blocked = record["settled_blocked"]
        settled_occupied = record["settled_occupied"]
        input_blocked = record["input_blocked"]
        input_occupied = record["input_occupied"]
        records.append(record)

    inp = np.full((h, w), BG, dtype=int)
    _paint_pixels(inp, anchor_pixels)
    for record in records:
        _paint_pixels(inp, record["input_pixels"])

    expected_out = _build_expected_output(h, w, anchor_pixels, records)
    if not _semantic_checks(inp, expected_out, records):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
