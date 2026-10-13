from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ["top", "right", "bottom", "left"]
SIDE_TO_GRAVITY = {
    "top": "right",
    "right": "down",
    "bottom": "left",
    "left": "up",
}
ALL_COLORS = list(range(1, 10))
GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(len(cells)),
                }
            )
    return comps


def _detect_edge_line(grid: np.ndarray, bg: int) -> tuple[str, int]:
    sides: list[tuple[str, int]] = []
    for color in set(map(int, grid.ravel())) - {bg}:
        if np.all(grid[0, :] == color):
            sides.append(("top", int(color)))
        if np.all(grid[:, -1] == color):
            sides.append(("right", int(color)))
        if np.all(grid[-1, :] == color):
            sides.append(("bottom", int(color)))
        if np.all(grid[:, 0] == color):
            sides.append(("left", int(color)))
    assert len(sides) == 1, sides
    return sides[0]


def _is_edge_component(comp: dict, side: str, line_color: int, shape: tuple[int, int]) -> bool:
    h, w = shape
    if int(comp["color"]) != int(line_color):
        return False
    r0, c0, r1, c1 = comp["bbox"]
    area = int(comp["area"])
    if side == "top":
        return r0 == 0 and r1 == 0 and c0 == 0 and c1 == w - 1 and area == w
    if side == "right":
        return c0 == w - 1 and c1 == w - 1 and r0 == 0 and r1 == h - 1 and area == h
    if side == "bottom":
        return r0 == h - 1 and r1 == h - 1 and c0 == 0 and c1 == w - 1 and area == w
    return c0 == 0 and c1 == 0 and r0 == 0 and r1 == h - 1 and area == h


def _is_rectangle_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 7 or w < 7:
        return False
    if int(comp["area"]) != 2 * h + 2 * w - 4:
        return False
    mask = np.zeros((h, w), dtype=bool)
    local = comp["coords"] - np.array([r0, c0], dtype=int)
    mask[local[:, 0], local[:, 1]] = True
    return bool(mask[0, :].all() and mask[-1, :].all() and mask[:, 0].all() and mask[:, -1].all())


def _infer_template_axes(grid: np.ndarray, bg: int, components: list[dict], skip_ids: set[int]) -> tuple[bool, bool]:
    candidates: list[tuple[int, bool, bool]] = []
    for idx, comp in enumerate(components):
        if idx in skip_ids:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        hh, ww = patch.shape
        if hh % 2 == 0 or ww % 2 == 0:
            continue
        cr, cc = hh // 2, ww // 2
        if int(patch[cr, cc]) == bg:
            continue
        fg = patch != bg
        allowed = np.zeros_like(fg, dtype=bool)
        allowed[cr, :] = True
        allowed[:, cc] = True
        if bool(np.any(fg & ~allowed)):
            continue
        horizontal = bool(np.any(fg[cr, :cc]) or np.any(fg[cr, cc + 1 :]))
        vertical = bool(np.any(fg[:cr, cc]) or np.any(fg[cr + 1 :, cc]))
        if not (horizontal or vertical):
            continue
        candidates.append((int(comp["area"]), horizontal, vertical))
    assert candidates, "template sprite not found"
    _area, horizontal, vertical = min(candidates, key=lambda item: item[0])
    return horizontal, vertical


def _parse_rectangles(grid: np.ndarray) -> tuple[int, str, str, list[dict], tuple[bool, bool]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    side, line_color = _detect_edge_line(g, bg)
    gravity = SIDE_TO_GRAVITY[side]
    components = _monochrome_components(g, bg)
    rectangles: list[dict] = []
    skip_ids: set[int] = set()
    for idx, comp in enumerate(components):
        if _is_edge_component(comp, side, line_color, g.shape):
            skip_ids.add(idx)
            continue
        if _is_rectangle_frame(comp):
            rectangles.append(comp)
            skip_ids.add(idx)
    rectangles.sort(key=lambda comp: comp["bbox"])
    assert len(rectangles) >= 2, "expected multiple rectangle containers"
    axes = _infer_template_axes(g, bg, components, skip_ids)
    return bg, side, gravity, rectangles, axes


def _markers_in_rectangle(grid: np.ndarray, rect: dict, bg: int) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = rect["bbox"]
    border_color = int(rect["color"])
    markers: list[tuple[int, int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            color = int(grid[r, c])
            if color == bg or color == border_color:
                continue
            markers.append((r, c, color))
    assert markers, "each rectangle should contain at least one marker pixel"
    markers.sort()
    return markers


def _stamped_full_patch(shape: tuple[int, int], markers: list[tuple[int, int, int]], axes: tuple[bool, bool], bg: int) -> np.ndarray:
    h, w = shape
    horizontal, vertical = axes
    stamped = np.full((h, w), int(bg), dtype=int)
    for r, c, color in markers:
        if horizontal:
            stamped[r, 1 : w - 1] = int(color)
        if vertical:
            stamped[1 : h - 1, c] = int(color)
        stamped[r, c] = int(color)
    return stamped


def _compact_interior(interior: np.ndarray, gravity: str, bg: int) -> np.ndarray:
    h, w = interior.shape
    out = np.full((h, w), int(bg), dtype=int)
    if gravity in ("down", "up"):
        for c in range(w):
            values = [int(v) for v in interior[:, c] if int(v) != bg]
            if not values:
                continue
            if gravity == "down":
                out[h - len(values) :, c] = values
            else:
                out[: len(values), c] = values
    else:
        for r in range(h):
            values = [int(v) for v in interior[r, :] if int(v) != bg]
            if not values:
                continue
            if gravity == "right":
                out[r, w - len(values) :] = values
            else:
                out[r, : len(values)] = values
    return out


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, _side, gravity, rectangles, axes = _parse_rectangles(g)
    out = g.copy()
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        local_markers = [
            (r - r0, c - c0, color)
            for r, c, color in _markers_in_rectangle(g, rect, bg)
        ]
        stamped = _stamped_full_patch((r1 - r0 + 1, c1 - c0 + 1), local_markers, axes, bg)
        compacted = _compact_interior(stamped[1:-1, 1:-1], gravity, bg)
        out[r0 + 1 : r1, c0 + 1 : c1] = compacted
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp crossbars inside each rectangle, then gravity-pack the stamped pixels by the marked edge's clockwise direction.

    Input structure:
    - The grid has a black background, one full-edge marker line, one small external cross template, and 2-3 hollow rectangular containers with one-pixel-thick monochrome borders.
    - Each rectangle contains 1-2 isolated marker pixels whose colors differ from the rectangle border.
    - The cross template lies outside every rectangle and specifies whether horizontal arms, vertical arms, or both are active.

    Rule:
    - Read the edge marker by clockwise border travel: top -> right, right -> down, bottom -> left, left -> up.
    - For every rectangle and for every internal marker pixel, stamp the template centered on that marker by filling the whole reachable interior row and/or column up to the inner border wall; later markers in row-major order overwrite earlier overlaps.
    - Then, inside each rectangle independently, slide all non-background non-border stamped pixels in the gravity direction until they pack against the inner wall or previously stopped pixels, while leaving borders, the external template, and the edge marker line unchanged.

    Key insight:
    - The outside cross does not directly appear in the answer; it first generates a colored pixel set inside each box, and only that generated set is subsequently rearranged by the edge-directed gravity rule.
    """
    return _transform(np.asarray(grid, dtype=int))


def _compact_all_directions(interior: np.ndarray, bg: int) -> dict[str, np.ndarray]:
    return {
        gravity: _compact_interior(interior, gravity, bg)
        for gravity in ("up", "down", "left", "right")
    }


def _expanded_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _rectangle_patch(h: int, w: int, border_color: int, markers: list[tuple[int, int, int]]) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = int(border_color)
    patch[-1, :] = int(border_color)
    patch[:, 0] = int(border_color)
    patch[:, -1] = int(border_color)
    for r, c, color in markers:
        patch[int(r), int(c)] = int(color)
    return patch


def _sample_marker_positions(h: int, w: int, pattern: str) -> list[tuple[int, int]]:
    rows = list(range(2, h - 2))
    cols = list(range(2, w - 2))
    if pattern == "single":
        return [(random.choice(rows), random.choice(cols))]
    if pattern == "same_row":
        r = random.choice(rows)
        c1, c2 = sorted(random.sample(cols, 2))
        return [(r, c1), (r, c2)]
    if pattern == "same_col":
        c = random.choice(cols)
        r1, r2 = sorted(random.sample(rows, 2))
        return [(r1, c), (r2, c)]
    while True:
        a = (random.choice(rows), random.choice(cols))
        b = (random.choice(rows), random.choice(cols))
        if a != b and a[0] != b[0] and a[1] != b[1]:
            return [a, b]


def _candidate_rect_positions(h: int, w: int, rect_h: int, rect_w: int, side: str) -> list[tuple[int, int]]:
    top_margin = 2 if side == "top" else 1
    bottom_margin = 2 if side == "bottom" else 1
    left_margin = 2 if side == "left" else 1
    right_margin = 2 if side == "right" else 1
    positions = []
    for r0 in range(top_margin, h - bottom_margin - rect_h + 1):
        for c0 in range(left_margin, w - right_margin - rect_w + 1):
            positions.append((r0, c0))
    random.shuffle(positions)
    return positions


def _candidate_sprite_positions(grid_shape: tuple[int, int], sprite_shape: tuple[int, int], occupied_boxes: list[tuple[int, int, int, int]], side: str) -> list[tuple[int, int]]:
    h, w = grid_shape
    sh, sw = sprite_shape
    top_margin = 1 if side == "top" else 0
    bottom_margin = 1 if side == "bottom" else 0
    left_margin = 1 if side == "left" else 0
    right_margin = 1 if side == "right" else 0
    positions = []
    for r0 in range(top_margin, h - bottom_margin - sh + 1):
        for c0 in range(left_margin, w - right_margin - sw + 1):
            box = (r0, c0, r0 + sh - 1, c0 + sw - 1)
            if any(_expanded_overlap(box, other, pad=1) for other in occupied_boxes):
                continue
            positions.append((r0, c0))
    random.shuffle(positions)
    return positions


def _scene_semantics(side: str, rectangles: list[dict], bg: int) -> bool:
    gravity = SIDE_TO_GRAVITY[side]
    saw_movement = False
    saw_unique_direction = False
    saw_multi_marker = False
    saw_overlap_branch = False
    for rect in rectangles:
        h = int(rect["bbox"][2] - rect["bbox"][0] + 1)
        w = int(rect["bbox"][3] - rect["bbox"][1] + 1)
        local_markers = [(r, c, color) for r, c, color in rect["markers"]]
        if len(local_markers) >= 2:
            saw_multi_marker = True
            for i in range(len(local_markers)):
                for j in range(i + 1, len(local_markers)):
                    a = local_markers[i]
                    b = local_markers[j]
                    if (a[0] == b[0] or a[1] == b[1]) and a[2] != b[2]:
                        saw_overlap_branch = True
        stamped = _stamped_full_patch((h, w), local_markers, (True, True), bg)
        interior = stamped[1:-1, 1:-1]
        compacted = _compact_interior(interior, gravity, bg)
        if not np.array_equal(interior, compacted):
            saw_movement = True
        by_dir = _compact_all_directions(interior, bg)
        if all(not np.array_equal(by_dir[gravity], arr) for name, arr in by_dir.items() if name != gravity):
            saw_unique_direction = True
    return saw_movement and saw_unique_direction and saw_multi_marker and saw_overlap_branch


def _gen_attempt() -> np.ndarray | None:
    n_rectangles = random.choices([2, 3], weights=[3, 2])[0]
    side = random.choice(SIDES)
    h = random.randint(18, 24 if n_rectangles == 2 else 27)
    w = random.randint(18, 24 if n_rectangles == 2 else 27)
    if max(h, w) > 30:
        return None

    chosen = random.sample(ALL_COLORS, n_rectangles + 4)
    line_color = chosen[0]
    template_color = chosen[1]
    border_colors = chosen[2 : 2 + n_rectangles]
    marker_pool = chosen[2 + n_rectangles :]
    if len(marker_pool) < 2:
        return None

    rectangles = []
    occupied_boxes: list[tuple[int, int, int, int]] = []
    forced_multi = random.randrange(n_rectangles)
    forced_overlap = random.randrange(n_rectangles)
    for idx in range(n_rectangles):
        rect_h = random.randint(7, 9)
        rect_w = random.randint(7, 9)
        pattern_choices = ["single", "same_row", "same_col", "disjoint"]
        if idx == forced_multi:
            pattern_choices = ["same_row", "same_col", "disjoint"]
        if idx == forced_overlap:
            pattern_choices = ["same_row", "same_col"]
        pattern = random.choice(pattern_choices)
        positions = _candidate_rect_positions(h, w, rect_h, rect_w, side)
        placed = None
        for r0, c0 in positions:
            box = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            if any(_expanded_overlap(box, other, pad=1) for other in occupied_boxes):
                continue
            local_positions = _sample_marker_positions(rect_h, rect_w, pattern)
            if len(local_positions) == 1:
                colors = [random.choice(marker_pool)]
            else:
                colors = random.sample(marker_pool, 2)
            markers = [
                (int(local_r), int(local_c), int(color))
                for (local_r, local_c), color in zip(local_positions, colors)
            ]
            placed = {
                "bbox": box,
                "border_color": int(border_colors[idx]),
                "markers": markers,
            }
            break
        if placed is None:
            return None
        rectangles.append(placed)
        occupied_boxes.append(placed["bbox"])

    template = np.zeros((3, 3), dtype=int)
    template[1, :] = int(template_color)
    template[:, 1] = int(template_color)
    sprite_positions = _candidate_sprite_positions((h, w), template.shape, occupied_boxes, side)
    if not sprite_positions:
        return None
    sprite_r0, sprite_c0 = sprite_positions[0]

    grid = np.zeros((h, w), dtype=int)
    if side == "top":
        grid[0, :] = int(line_color)
    elif side == "right":
        grid[:, -1] = int(line_color)
    elif side == "bottom":
        grid[-1, :] = int(line_color)
    else:
        grid[:, 0] = int(line_color)

    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        patch = _rectangle_patch(r1 - r0 + 1, c1 - c0 + 1, rect["border_color"], rect["markers"])
        grid[r0 : r1 + 1, c0 : c1 + 1] = patch

    grid[sprite_r0 : sprite_r0 + 3, sprite_c0 : sprite_c0 + 3] = np.maximum(
        grid[sprite_r0 : sprite_r0 + 3, sprite_c0 : sprite_c0 + 3], template
    )

    if not _scene_semantics(side, rectangles, 0):
        return None

    parsed_bg, parsed_side, parsed_gravity, parsed_rects, axes = _parse_rectangles(grid)
    if parsed_bg != 0 or parsed_side != side or parsed_gravity != SIDE_TO_GRAVITY[side] or axes != (True, True):
        return None
    if len(parsed_rects) != n_rectangles:
        return None

    expected_boxes = sorted(rect["bbox"] for rect in rectangles)
    parsed_boxes = sorted(rect["bbox"] for rect in parsed_rects)
    if expected_boxes != parsed_boxes:
        return None

    out = _transform(grid)
    if np.array_equal(out, grid):
        return None
    if not any(np.any(out != 0, axis=1)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
