from __future__ import annotations

import numpy as np


BBox = tuple[int, int, int, int]


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> BBox:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return (r0, c0, r1, c1)


def _intersects(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _paste(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    grid[top : top + patch.shape[0], left : left + patch.shape[1]] = patch


def _find_sprites(grid: np.ndarray, bg: int) -> list[tuple[BBox, np.ndarray, int]]:
    sprites: list[tuple[BBox, np.ndarray, int]] = []
    for coords in _components(grid != bg):
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3) or len(coords) != 9:
            continue

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border_mask = np.ones((3, 3), dtype=bool)
        border_mask[1, 1] = False
        border_colors = set(map(int, patch[border_mask]))
        if len(border_colors) != 1:
            continue

        border_color = next(iter(border_colors))
        center_color = int(patch[1, 1])
        if border_color == bg or center_color == bg or border_color == center_color:
            continue
        sprites.append((box, patch.copy(), center_color))

    sprites.sort(key=lambda item: item[0])
    return sprites


def _find_target_rectangles(grid: np.ndarray, bg: int) -> dict[int, list[BBox]]:
    targets: dict[int, list[BBox]] = {}
    for color in sorted(map(int, np.unique(grid))):
        if color == bg:
            continue
        boxes: list[BBox] = []
        for coords in _components(grid == color):
            box = _bbox(coords)
            r0, c0, r1, c1 = box
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if len(coords) >= 2 and len(coords) == height * width:
                boxes.append(box)
        if boxes:
            targets[color] = boxes
    return targets


def _direction_to_target(sprite_bbox: BBox, target_bbox: BBox) -> tuple[int, int]:
    sr0, sc0, sr1, sc1 = sprite_bbox
    tr0, tc0, tr1, tc1 = target_bbox
    row_overlap = not (sr1 < tr0 or tr1 < sr0)
    col_overlap = not (sc1 < tc0 or tc1 < sc0)

    candidates: list[tuple[int, int]] = []
    if row_overlap and tc0 > sc1:
        candidates.append((0, 1))
    if row_overlap and tc1 < sc0:
        candidates.append((0, -1))
    if col_overlap and tr0 > sr1:
        candidates.append((1, 0))
    if col_overlap and tr1 < sr0:
        candidates.append((-1, 0))

    assert len(candidates) == 1, "expected exactly one cardinal direction to each target"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat each 3x3 sprite toward the unique solid rectangle whose color matches
    the sprite center, stepping by 3 cells until the first overlapping copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 3 and w >= 3

    bg = _dominant_color(g)
    sprites = _find_sprites(g, bg)
    assert sprites, "expected at least one control-center sprite"

    targets_by_color = _find_target_rectangles(g, bg)
    out = g.copy()

    for sprite_bbox, patch, center_color in sprites:
        assert center_color in targets_by_color, "missing target rectangle for sprite center color"
        boxes = targets_by_color[center_color]
        assert len(boxes) == 1, "expected exactly one target rectangle per control color"
        target_bbox = boxes[0]

        dr, dc = _direction_to_target(sprite_bbox, target_bbox)
        r, c = sprite_bbox[0], sprite_bbox[1]

        for _ in range(max(h, w) + 1):
            current_bbox = (r, c, r + 2, c + 2)
            assert 0 <= r and r + 2 < h and 0 <= c and c + 2 < w
            _paste(out, patch, r, c)
            if _intersects(current_bbox, target_bbox):
                break
            r += 3 * dr
            c += 3 * dc
        else:
            raise AssertionError("sprite did not reach its target")

    return out


PALETTE = tuple(range(10))
DIRECTIONS = ((0, 1), (0, -1), (1, 0), (-1, 0))

GRID_SIZE_MIN = 22
GRID_SIZE_MAX = 31
PAIR_COUNT_MIN = 2
PAIR_COUNT_MAX = 5
RECT_SIDE_MIN = 2
RECT_SIDE_MAX = 7
STEP_COUNT_MIN = 1
STEP_COUNT_MAX = 5
GENERATION_ATTEMPTS = 400
PAIR_PLACEMENT_ATTEMPTS = 300

_RNG = np.random.default_rng()


def _make_patch(body_color: int, center_color: int) -> np.ndarray:
    patch = np.full((3, 3), body_color, dtype=int)
    patch[1, 1] = center_color
    return patch


def _pair_spec(
    rng: np.random.Generator,
    control_color: int,
    body_color: int,
    direction: tuple[int, int],
) -> dict:
    rect_h = int(rng.integers(RECT_SIDE_MIN, RECT_SIDE_MAX))
    rect_w = int(rng.integers(RECT_SIDE_MIN, RECT_SIDE_MAX))
    steps = int(rng.integers(STEP_COUNT_MIN, STEP_COUNT_MAX))

    if direction[0] == 0:
        overlap = int(rng.integers(1, min(3, rect_w) + 1))
        lane_offset = int(rng.integers(-2, rect_h))
        sprite_row = lane_offset
        final_col = -(3 - overlap) if direction == (0, 1) else rect_w - overlap
        start_col = final_col - 3 * steps if direction == (0, 1) else final_col + 3 * steps
        path = [(sprite_row, start_col + 3 * direction[1] * t) for t in range(steps + 1)]
    else:
        overlap = int(rng.integers(1, min(3, rect_h) + 1))
        lane_offset = int(rng.integers(-2, rect_w))
        sprite_col = lane_offset
        final_row = -(3 - overlap) if direction == (1, 0) else rect_h - overlap
        start_row = final_row - 3 * steps if direction == (1, 0) else final_row + 3 * steps
        path = [(start_row + 3 * direction[0] * t, sprite_col) for t in range(steps + 1)]

    cells: set[tuple[int, int]] = set()
    for rr in range(rect_h):
        for cc in range(rect_w):
            cells.add((rr, cc))
    for pr, pc in path:
        for rr in range(pr, pr + 3):
            for cc in range(pc, pc + 3):
                cells.add((rr, cc))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]

    return {
        "control_color": control_color,
        "body_color": body_color,
        "rect_h": rect_h,
        "rect_w": rect_w,
        "direction": direction,
        "path_local": path,
        "cells_local": cells,
        "min_r": min(rows),
        "max_r": max(rows),
        "min_c": min(cols),
        "max_c": max(cols),
    }


def _expanded_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _try_place_pair(
    rng: np.random.Generator,
    grid: np.ndarray,
    reserved: np.ndarray,
    spec: dict,
) -> dict | None:
    h, w = grid.shape
    row_low = -spec["min_r"]
    row_high = h - 1 - spec["max_r"]
    col_low = -spec["min_c"]
    col_high = w - 1 - spec["max_c"]
    if row_low > row_high or col_low > col_high:
        return None

    patch = _make_patch(spec["body_color"], spec["control_color"])
    cells_local = spec["cells_local"]

    for _ in range(PAIR_PLACEMENT_ATTEMPTS):
        tr = int(rng.integers(row_low, row_high + 1))
        tc = int(rng.integers(col_low, col_high + 1))

        cells_global = {(tr + r, tc + c) for r, c in cells_local}
        expanded = _expanded_cells(cells_global, h, w)
        if any(reserved[r, c] for r, c in expanded):
            continue

        rect_h = spec["rect_h"]
        rect_w = spec["rect_w"]
        grid[tr : tr + rect_h, tc : tc + rect_w] = spec["control_color"]

        start_r, start_c = spec["path_local"][0]
        _paste(grid, patch, tr + start_r, tc + start_c)

        for r, c in expanded:
            reserved[r, c] = True

        return {
            "target_bbox": (tr, tc, tr + rect_h - 1, tc + rect_w - 1),
            "path_abs": [(tr + r, tc + c) for r, c in spec["path_local"]],
            "patch": patch,
        }

    return None


def _render_expected(grid: np.ndarray, placements: list[dict]) -> np.ndarray:
    out = grid.copy()
    for item in placements:
        for top, left in item["path_abs"]:
            _paste(out, item["patch"], top, left)
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_SIZE_MIN, GRID_SIZE_MAX))
    w = int(rng.integers(GRID_SIZE_MIN, GRID_SIZE_MAX))
    bg = int(rng.choice(PALETTE))
    grid = np.full((h, w), bg, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    pair_count = int(rng.integers(PAIR_COUNT_MIN, PAIR_COUNT_MAX))
    fg_colors = [color for color in PALETTE if color != bg]
    control_colors = list(map(int, rng.choice(fg_colors, size=pair_count, replace=False)))
    body_palette = [color for color in PALETTE if color not in {bg, *control_colors}]
    directions = [tuple(map(int, d)) for d in rng.permutation(np.array(DIRECTIONS))[:pair_count]]

    placements: list[dict] = []
    for control_color, direction in zip(control_colors, directions):
        body_color = int(rng.choice(body_palette))
        spec = _pair_spec(rng, control_color, body_color, direction)
        placed = _try_place_pair(rng, grid, reserved, spec)
        if placed is None:
            return None
        placements.append(placed)

    if _dominant_color(grid) != bg:
        return None

    expected = _render_expected(grid, placements)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
