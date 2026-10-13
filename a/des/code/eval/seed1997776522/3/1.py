import numpy as np


BLACK = 0
GREY = 5
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 300
TEMPLATE_ATTEMPTS = 200


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _rotate(offset: tuple[int, int], direction: str) -> tuple[int, int]:
    dr, dc = offset
    if direction == "right":
        return dr, dc
    if direction == "down":
        return dc, -dr
    if direction == "left":
        return -dr, -dc
    if direction == "up":
        return -dc, dr
    raise AssertionError(f"unknown direction: {direction}")


def _room_from_component(grid: np.ndarray, coords: np.ndarray) -> dict:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "room must enclose a non-empty interior"

    actual = {tuple(map(int, rc)) for rc in coords}
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))

    missing = expected - actual
    assert len(missing) == 1, "room border must have exactly one gap"
    gap = next(iter(missing))
    gr, gc = gap
    assert gap not in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}, "gap must not be a corner"

    if gr == r0:
        direction = "up"
    elif gr == r1:
        direction = "down"
    elif gc == c0:
        direction = "left"
    else:
        assert gc == c1
        direction = "right"

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    seed_coords = np.argwhere((interior != BLACK) & (interior != GREY))
    assert seed_coords.shape == (1, 2), "room interior must contain exactly one colored seed"
    sr, sc = map(int, seed_coords[0])
    seed_r = r0 + 1 + sr
    seed_c = c0 + 1 + sc
    seed_color = int(grid[seed_r, seed_c])

    return {
        "bbox": (r0, c0, r1, c1),
        "interior_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
        "gap": gap,
        "direction": direction,
        "seed_pos": (seed_r, seed_c),
        "seed_color": seed_color,
    }


def _extract_rooms(grid: np.ndarray) -> list[dict]:
    rooms = [_room_from_component(grid, comp) for comp in _components(grid == GREY)]
    assert rooms, "expected at least one grey room"
    return rooms


def _build_room_interior_mask(shape: tuple[int, int], rooms: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for room in rooms:
        r0, c0, r1, c1 = room["interior_bbox"]
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _parse_legend(grid: np.ndarray, rooms: list[dict]) -> dict[int, tuple[tuple[int, int], ...]]:
    room_mask = _build_room_interior_mask(grid.shape, rooms)
    legend_mask = (grid != BLACK) & (grid != GREY) & (~room_mask)
    legend_colors = [int(color) for color in np.unique(grid[legend_mask])]
    assert legend_colors, "expected a legend outside the rooms"

    mapping: dict[int, tuple[tuple[int, int], ...]] = {}
    for color in legend_colors:
        comps = _components((grid == color) & legend_mask)
        assert len(comps) == 2, "legend color must form one marker and one shape component"
        comps.sort(key=len)
        marker_comp, shape_comp = comps
        assert len(marker_comp) == 1, "legend marker must be a singleton"
        marker_r, marker_c = map(int, marker_comp[0])
        offsets = sorted((int(r - marker_r), int(c - marker_c)) for r, c in shape_comp)
        assert all(dc >= 1 for _dr, dc in offsets), "legend shape must lie to the right of the marker"
        assert min(dc for _dr, dc in offsets) == 1, "legend shape must start immediately to the right"
        assert max(dc for _dr, dc in offsets) <= 3, "legend shape width must be at most 3"
        row_offsets = [dr for dr, _dc in offsets]
        assert max(row_offsets) - min(row_offsets) + 1 <= 3, "legend shape height must be at most 3"
        mapping[color] = tuple(offsets)

    return mapping


def _reaches_gap(raw_positions: list[tuple[int, int]], gap: tuple[int, int], direction: str) -> bool:
    gr, gc = gap
    if direction == "right":
        return max(c for _r, c in raw_positions) >= gc
    if direction == "left":
        return min(c for _r, c in raw_positions) <= gc
    if direction == "down":
        return max(r for r, _c in raw_positions) >= gr
    assert direction == "up"
    return min(r for r, _c in raw_positions) <= gr


def _paint_room(out: np.ndarray, room: dict, template_offsets: tuple[tuple[int, int], ...]) -> None:
    seed_r, seed_c = room["seed_pos"]
    r0, c0, r1, c1 = room["interior_bbox"]
    gap = room["gap"]
    direction = room["direction"]
    color = room["seed_color"]

    transformed = [_rotate(offset, direction) for offset in template_offsets]
    step = max(dc for _dr, dc in template_offsets)
    if direction == "right":
        shift = (0, step)
    elif direction == "left":
        shift = (0, -step)
    elif direction == "down":
        shift = (step, 0)
    else:
        assert direction == "up"
        shift = (-step, 0)

    for copy_index in range(20):
        raw_positions = []
        for dr, dc in transformed:
            rr = seed_r + dr + copy_index * shift[0]
            cc = seed_c + dc + copy_index * shift[1]
            raw_positions.append((rr, cc))
            if r0 <= rr <= r1 and c0 <= cc <= c1 and out[rr, cc] == BLACK:
                out[rr, cc] = color
        if _reaches_gap(raw_positions, gap, direction):
            return
    raise AssertionError("growth did not reach the gap within the copy limit")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the color legend, match each seed to its template, then repeat that
    template from the seed toward the room gap until the growth reaches the gap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rooms = _extract_rooms(g)
    legend = _parse_legend(g, rooms)

    out = g.copy()
    for room in rooms:
        color = room["seed_color"]
        assert color in legend, "every seed color must appear in the legend"
        _paint_room(out, room, legend[color])
    return out


def _is_connected_cells(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    stack = [next(iter(cells))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _make_template(rng: np.random.Generator) -> dict:
    for _ in range(TEMPLATE_ATTEMPTS):
        h = int(rng.integers(2, 4))
        w = int(rng.integers(2, 4))
        anchor_row = int(rng.integers(0, h))
        forbidden = (anchor_row, 0)
        universe = [(r, c) for r in range(h) for c in range(w) if (r, c) != forbidden]
        area = int(rng.integers(max(2, w), min(6, len(universe)) + 1))
        pick = rng.choice(len(universe), size=area, replace=False)
        cells = {universe[int(i)] for i in pick}
        first_col_rows = {r for r, c in cells if c == 0}
        last_col_rows = [r for r, c in cells if c == w - 1]
        if not first_col_rows or len(last_col_rows) != 1:
            continue
        if last_col_rows[0] not in first_col_rows:
            continue
        if not _is_connected_cells(cells):
            continue
        offsets = tuple(sorted((r - anchor_row, c + 1) for r, c in cells))
        return {
            "h": h,
            "w": w,
            "anchor_row": anchor_row,
            "cells": tuple(sorted(cells)),
            "offsets": offsets,
            "contact_row": int(last_col_rows[0]),
        }
    raise RuntimeError("failed to build a valid legend template")


def _room_spec(template: dict, direction: str, repeats: int, rng: np.random.Generator) -> dict:
    transformed = [_rotate(offset, direction) for offset in template["offsets"]]
    step = int(template["w"])
    if direction == "right":
        shift = (0, step)
    elif direction == "left":
        shift = (0, -step)
    elif direction == "down":
        shift = (step, 0)
    else:
        assert direction == "up"
        shift = (-step, 0)

    union = {(0, 0)}
    for k in range(repeats):
        for dr, dc in transformed:
            union.add((dr + k * shift[0], dc + k * shift[1]))

    contact_offset = (template["contact_row"] - template["anchor_row"], template["w"])
    contact_rot = _rotate(contact_offset, direction)
    gap_rel = (contact_rot[0] + (repeats - 1) * shift[0], contact_rot[1] + (repeats - 1) * shift[1])

    rows = [r for r, _c in union]
    cols = [c for _r, c in union]
    top_margin = int(rng.integers(0, 2))
    bottom_margin = int(rng.integers(0, 2))
    left_margin = int(rng.integers(0, 2))
    right_margin = int(rng.integers(0, 2))

    border_top = min(rows) - 1 - top_margin
    border_bottom = max(rows) + 1 + bottom_margin
    border_left = min(cols) - 1 - left_margin
    border_right = max(cols) + 1 + right_margin

    if direction == "right":
        border_right = gap_rel[1]
    elif direction == "left":
        border_left = gap_rel[1]
    elif direction == "down":
        border_bottom = gap_rel[0]
    else:
        assert direction == "up"
        border_top = gap_rel[0]

    outer_h = border_bottom - border_top + 1
    outer_w = border_right - border_left + 1
    seed_rel = (-border_top, -border_left)
    gap_rel_in_bbox = (seed_rel[0] + gap_rel[0], seed_rel[1] + gap_rel[1])

    return {
        "direction": direction,
        "repeats": repeats,
        "outer_h": int(outer_h),
        "outer_w": int(outer_w),
        "seed_rel": (int(seed_rel[0]), int(seed_rel[1])),
        "gap_rel": (int(gap_rel_in_bbox[0]), int(gap_rel_in_bbox[1])),
    }


def _place_legend(grid: np.ndarray, templates: list[dict], colors: list[int], corner: str) -> tuple[int, int, int, int]:
    entry_gaps = [1] * (len(templates) - 1)
    block_h = sum(t["h"] for t in templates) + sum(entry_gaps)
    block_w = max(1 + t["w"] for t in templates)
    h, w = grid.shape

    if "top" in corner:
        r0 = 1
    else:
        r0 = h - block_h - 2
    if "left" in corner:
        c0 = 1
    else:
        c0 = w - block_w - 2

    row = r0
    for color, template in zip(colors, templates):
        marker_r = row + template["anchor_row"]
        marker_c = c0
        grid[marker_r, marker_c] = color
        for rr, cc in template["cells"]:
            grid[row + rr, c0 + 1 + cc] = color
        row += template["h"] + 1

    return r0, c0, r0 + block_h - 1, c0 + block_w - 1


def _bbox_free(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    H, W = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _mark_bbox(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    H, W = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _draw_room(
    grid: np.ndarray,
    top: int,
    left: int,
    spec: dict,
    color: int,
) -> None:
    bottom = top + spec["outer_h"] - 1
    right = left + spec["outer_w"] - 1
    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY
    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY
    gap_r = top + spec["gap_rel"][0]
    gap_c = left + spec["gap_rel"][1]
    grid[gap_r, gap_c] = BLACK
    seed_r = top + spec["seed_rel"][0]
    seed_c = left + spec["seed_rel"][1]
    grid[seed_r, seed_c] = color


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    room_count = int(rng.integers(2, 5))
    colors = list(rng.choice(PALETTE_COLORS, size=room_count, replace=False))
    templates = [_make_template(rng) for _ in range(room_count)]
    directions = list(rng.permutation(["up", "down", "left", "right"])[:room_count])
    repeats = [int(rng.integers(2, 5)) for _ in range(room_count)]
    specs = [_room_spec(template, direction, repeat, rng) for template, direction, repeat in zip(templates, directions, repeats)]

    max_room_h = max(spec["outer_h"] for spec in specs)
    max_room_w = max(spec["outer_w"] for spec in specs)
    legend_h = sum(template["h"] for template in templates) + (room_count - 1)
    legend_w = max(1 + template["w"] for template in templates)
    H = int(max(22, legend_h + max_room_h + 8, rng.integers(24, 33)))
    W = int(max(22, legend_w + max_room_w + 10, rng.integers(24, 33)))

    grid = np.full((H, W), BLACK, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    corner = str(rng.choice(["top-left", "top-right", "bottom-left", "bottom-right"]))
    lr0, lc0, lr1, lc1 = _place_legend(grid, templates, colors, corner)
    _mark_bbox(occupied, lr0, lc0, lr1 - lr0 + 1, lc1 - lc0 + 1, pad=1)

    order = list(rng.permutation(room_count))
    for idx in order:
        spec = specs[idx]
        placed = False
        for _ in range(200):
            top = int(rng.integers(0, H - spec["outer_h"] + 1))
            left = int(rng.integers(0, W - spec["outer_w"] + 1))
            if not _bbox_free(occupied, top, left, spec["outer_h"], spec["outer_w"], pad=1):
                continue
            _draw_room(grid, top, left, spec, colors[idx])
            _mark_bbox(occupied, top, left, spec["outer_h"], spec["outer_w"], pad=1)
            placed = True
            break
        if not placed:
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
