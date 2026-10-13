from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5

SIDE_ORDER = ("top", "left", "bottom", "right")
SIDE_COLORS = {"top": RED, "left": YELLOW, "bottom": GREEN, "right": BLUE}
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_color_component_rows_cols(grid: np.ndarray, color: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(grid == color)
    assert coords.size > 0, f"missing required color {color}"
    return coords[:, 0], coords[:, 1]


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    red_rows, red_cols = _find_color_component_rows_cols(grid, RED)
    green_rows, green_cols = _find_color_component_rows_cols(grid, GREEN)
    yellow_rows, yellow_cols = _find_color_component_rows_cols(grid, YELLOW)
    blue_rows, blue_cols = _find_color_component_rows_cols(grid, BLUE)

    assert len(set(map(int, red_rows))) == 1
    assert len(set(map(int, green_rows))) == 1
    assert len(set(map(int, yellow_cols))) == 1
    assert len(set(map(int, blue_cols))) == 1

    top = int(red_rows[0])
    bottom = int(green_rows[0])
    left = int(yellow_cols[0])
    right = int(blue_cols[0])
    assert top < bottom and left < right

    assert set(map(int, red_cols)) == set(range(left + 1, right))
    assert set(map(int, green_cols)) == set(range(left + 1, right))
    assert set(map(int, yellow_rows)) == set(range(top + 1, bottom))
    assert set(map(int, blue_rows)) == set(range(top + 1, bottom))

    assert int(grid[top, left]) == BLACK
    assert int(grid[top, right]) == BLACK
    assert int(grid[bottom, left]) == BLACK
    assert int(grid[bottom, right]) == BLACK
    return top, left, bottom, right


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == color
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.asarray(comp, dtype=int))
    return components


def _component_order_key(coords: np.ndarray) -> tuple[int, int, int, int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    return (-r1, -r0, c0, c1, -len(coords))


def _settle_components(
    components: list[np.ndarray],
    bottom_border_row: int,
    sort_descending: bool = True,
) -> tuple[list[np.ndarray], list[set[str]], list[int]]:
    ordered = sorted(
        components,
        key=_component_order_key,
        reverse=False,
    )
    if not sort_descending:
        ordered = list(reversed(ordered))

    occupied: set[tuple[int, int]] = set()
    settled: list[np.ndarray] = []
    blockers: list[set[str]] = []
    shifts: list[int] = []
    for coords in ordered:
        shift = 0
        while True:
            hit: set[str] = set()
            for r, c in coords:
                nr = int(r + shift + 1)
                nc = int(c)
                if nr == bottom_border_row:
                    hit.add("floor")
                elif (nr, nc) in occupied:
                    hit.add("object")
            if hit:
                break
            shift += 1

        moved = coords.copy()
        moved[:, 0] += shift
        settled.append(moved)
        blockers.append(hit)
        shifts.append(shift)
        for r, c in moved:
            occupied.add((int(r), int(c)))
    return settled, blockers, shifts


def _nearest_side_color(r: int, c: int, top: int, left: int, bottom: int, right: int) -> int:
    distances = {
        "top": int(r - top),
        "left": int(c - left),
        "bottom": int(bottom - r),
        "right": int(right - c),
    }
    nearest = min(distances.values())
    winners = [side for side in SIDE_ORDER if distances[side] == nearest]
    if len(winners) != 1:
        return GRAY
    return SIDE_COLORS[winners[0]]


def _render_frame_only(shape: tuple[int, int], frame: tuple[int, int, int, int]) -> np.ndarray:
    top, left, bottom, right = frame
    grid = np.zeros(shape, dtype=int)
    grid[top, left + 1 : right] = RED
    grid[bottom, left + 1 : right] = GREEN
    grid[top + 1 : bottom, left] = YELLOW
    grid[top + 1 : bottom, right] = BLUE
    return grid


def _render_output(
    shape: tuple[int, int],
    frame: tuple[int, int, int, int],
    settled_components: list[np.ndarray],
) -> np.ndarray:
    top, left, bottom, right = frame
    out = _render_frame_only(shape, frame)
    for coords in settled_components:
        for r, c in coords:
            out[r, c] = _nearest_side_color(int(r), int(c), top, left, bottom, right)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the gray sprites inside the colored frame, then recolor each landed pixel by its uniquely nearest frame side.

    Input structure:
    - The background is black.
    - Near the grid edge there is a one-cell-thick rectangular frame with black corners.
    - The frame sides use four fixed colors: top red, right blue, bottom green, and left yellow.
    - Inside the frame there are several disconnected gray 4-connected sprites that do not initially touch the frame.

    Rule:
    - Find every gray sprite and sort the sprites by the original bottom row of their bounding boxes, from lower to higher.
    - For each sprite in that order, move it straight downward as one rigid body until one more step would hit the green frame side or an already settled sprite pixel.
    - After all sprites settle, recolor each landed sprite pixel by the color of its uniquely nearest frame side using Manhattan distance to the four frame sides.
    - If a landed pixel is tied between two or more nearest sides, keep it gray; keep the frame and black background unchanged.

    Key insight:
    - The color depends on geometry after gravity, not before it: falling changes which side is nearest, and collision order determines the final landed layout.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    frame = _find_frame(g)
    top, left, bottom, right = frame
    gray_components = _components_of_color(g, GRAY)
    assert 3 <= len(gray_components) <= 5
    for coords in gray_components:
        r0, c0, r1, c1 = _bbox(coords)
        assert top < r0 <= r1 < bottom
        assert left < c0 <= c1 < right

    settled, _blockers, _shifts = _settle_components(gray_components, bottom)
    return _render_output(g.shape, frame, settled)


SHAPE_LIBRARY: list[tuple[tuple[int, int], ...]] = [
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
]


def _normalize_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _rotate_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    max_r = max(r for r, _ in cells)
    return _normalize_shape(tuple((c, max_r - r) for r, c in cells))


def _flip_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    max_c = max(c for _, c in cells)
    return _normalize_shape(tuple((r, max_c - c) for r, c in cells))


def _random_shape(role: str) -> tuple[tuple[int, int], ...]:
    candidates = []
    for base in SHAPE_LIBRARY:
        cells = base
        for _ in range(random.randint(0, 3)):
            cells = _rotate_shape(cells)
        if random.random() < 0.5:
            cells = _flip_shape(cells)
        cells = _normalize_shape(cells)
        area = len(cells)
        height = max(r for r, _ in cells) + 1
        width = max(c for _, c in cells) + 1
        if role in {"base", "mid", "top"}:
            if 3 <= area <= 8 and height <= 3 and width <= 4 and width >= 2:
                candidates.append(cells)
        else:
            if 3 <= area <= 6 and height <= 3 and width <= 3:
                candidates.append(cells)
    assert candidates
    return random.choice(candidates)


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _placed_coords(shape: tuple[tuple[int, int], ...], anchor_r: int, anchor_c: int) -> set[tuple[int, int]]:
    return {(anchor_r + r, anchor_c + c) for r, c in shape}


def _separated(coords: set[tuple[int, int]], others: list[set[tuple[int, int]]]) -> bool:
    occupied = set().union(*others) if others else set()
    for r, c in coords:
        if (r, c) in occupied:
            return False
        for dr, dc in NEIGHBORS8:
            if (r + dr, c + dc) in occupied:
                return False
    return True


def _settle_relative(
    objects: list[dict],
    interior_h: int,
) -> tuple[list[np.ndarray], list[set[str]], list[int], list[dict]]:
    components = []
    for obj in objects:
        coords = np.asarray(sorted(obj["coords"]), dtype=int)
        components.append(coords)
    settled, blockers, shifts = _settle_components(components, interior_h, sort_descending=True)
    ordered_objects = sorted(objects, key=lambda obj: _component_order_key(np.asarray(sorted(obj["coords"]), dtype=int)))
    return settled, blockers, shifts, ordered_objects


def _render_input_grid(
    grid_shape: tuple[int, int],
    frame: tuple[int, int, int, int],
    objects: list[dict],
) -> np.ndarray:
    top, left, _bottom, _right = frame
    grid = _render_frame_only(grid_shape, frame)
    r_off = top + 1
    c_off = left + 1
    for obj in objects:
        for r, c in obj["coords"]:
            grid[r_off + r, c_off + c] = GRAY
    return grid


def _render_output_grid_from_relative(
    grid_shape: tuple[int, int],
    frame: tuple[int, int, int, int],
    settled_relative: list[np.ndarray],
) -> np.ndarray:
    top, left, _bottom, _right = frame
    shifted = []
    for coords in settled_relative:
        moved = coords.copy()
        moved[:, 0] += top + 1
        moved[:, 1] += left + 1
        shifted.append(moved)
    return _render_output(grid_shape, frame, shifted)


def _class_map_for_relative(
    coords: np.ndarray,
    interior_frame: tuple[int, int, int, int],
) -> list[int]:
    top, left, bottom, right = interior_frame
    return [_nearest_side_color(int(r), int(c), top, left, bottom, right) for r, c in coords]


def _pixelwise_gravity(objects: list[dict], interior_h: int) -> set[tuple[int, int]]:
    pixels = sorted(
        [cell for obj in objects for cell in obj["coords"]],
        key=lambda rc: (-rc[0], rc[1]),
    )
    occupied: set[tuple[int, int]] = set()
    final: set[tuple[int, int]] = set()
    for r, c in pixels:
        nr = r
        while nr + 1 < interior_h and (nr + 1, c) not in occupied:
            nr += 1
        occupied.add((nr, c))
        final.add((nr, c))
    return final


def _recolor_without_gravity(
    grid_shape: tuple[int, int],
    frame: tuple[int, int, int, int],
    objects: list[dict],
) -> np.ndarray:
    top, left, bottom, right = frame
    out = _render_frame_only(grid_shape, frame)
    r_off = top + 1
    c_off = left + 1
    for obj in objects:
        for r, c in obj["coords"]:
            rr = r_off + r
            cc = c_off + c
            out[rr, cc] = _nearest_side_color(rr, cc, top, left, bottom, right)
    return out


def _validate_semantics(
    grid_shape: tuple[int, int],
    frame: tuple[int, int, int, int],
    interior_h: int,
    interior_w: int,
    objects: list[dict],
) -> bool:
    settled, blockers, shifts, ordered_objects = _settle_relative(objects, interior_h)
    correct_out = _render_output_grid_from_relative(grid_shape, frame, settled)
    input_grid = _render_input_grid(grid_shape, frame, objects)
    if not np.array_equal(main(input_grid), correct_out):
        return False

    input_bottoms = []
    changed_assignment = False
    for obj, final_coords, shift in zip(ordered_objects, settled, shifts):
        coords = np.asarray(sorted(obj["coords"]), dtype=int)
        input_bottoms.append(_bbox(coords)[2])
        if shift <= 0:
            return False
        before = _class_map_for_relative(coords, (-1, -1, interior_h, interior_w))
        after = _class_map_for_relative(final_coords, (-1, -1, interior_h, interior_w))
        if before != after:
            changed_assignment = True
    if len(set(input_bottoms)) != len(input_bottoms):
        return False
    if not changed_assignment:
        return False

    if not any("floor" in hit for hit in blockers):
        return False
    if not any("object" in hit for hit in blockers):
        return False

    sprite_colors = set()
    tie_present = False
    for coords in settled:
        for r, c in coords:
            color = _nearest_side_color(int(frame[0] + 1 + r), int(frame[1] + 1 + c), *frame)
            sprite_colors.add(color)
            if color == GRAY:
                tie_present = True
    if not tie_present:
        return False
    if len(sprite_colors) < 3:
        return False

    wrong_components = [np.asarray(sorted(obj["coords"]), dtype=int) for obj in objects]
    wrong_order_settled, _wrong_blockers, _wrong_shifts = _settle_components(
        wrong_components,
        interior_h,
        sort_descending=False,
    )
    wrong_order_out = _render_output_grid_from_relative(grid_shape, frame, wrong_order_settled)
    if np.array_equal(correct_out, wrong_order_out):
        return False

    pixelwise = _pixelwise_gravity(objects, interior_h)
    correct_pixels = {tuple(map(int, p)) for comp in settled for p in comp}
    if pixelwise == correct_pixels:
        return False

    no_gravity_out = _recolor_without_gravity(grid_shape, frame, objects)
    if np.array_equal(correct_out, no_gravity_out):
        return False

    return True


def _try_place_object(
    shape: tuple[tuple[int, int], ...],
    anchor_options: list[tuple[int, int]],
    others: list[set[tuple[int, int]]],
) -> tuple[int, int, set[tuple[int, int]]] | None:
    options = anchor_options[:]
    random.shuffle(options)
    for anchor_r, anchor_c in options:
        coords = _placed_coords(shape, anchor_r, anchor_c)
        if _separated(coords, others):
            return anchor_r, anchor_c, coords
    return None


def _candidate_roles(count: int) -> list[str]:
    if count == 3:
        return ["base", "top", random.choice(["left", "right"])]
    if count == 4:
        if random.random() < 0.5:
            return ["base", "mid", "top", random.choice(["left", "right"])]
        return ["base", "top", "left", "right"]
    return ["base", "mid", "top", "left", "right"]


def _build_objects(interior_h: int, interior_w: int, count: int) -> list[dict] | None:
    roles = _candidate_roles(count)
    objects: list[dict] = []
    placed_sets: list[set[tuple[int, int]]] = []
    center_band = max(1, interior_w // 2 - 1)
    central_col = random.randint(max(1, center_band - 1), min(interior_w - 4, center_band + 1))
    placement_order = [role for role in ["top", "mid", "base", "left", "right"] if role in roles]

    for role in placement_order:
        shape = _random_shape(role)
        height, width = _shape_dims(shape)
        if height > interior_h - 2 or width > interior_w - 2:
            return None

        options: list[tuple[int, int]] = []
        if role in {"base", "mid", "top"}:
            col_lo = max(1, central_col - 1)
            col_hi = min(interior_w - width - 1, central_col + 1)
            if col_lo > col_hi:
                return None
            if role == "top":
                row_lo = 1
                row_hi = max(row_lo, interior_h // 4)
            elif role == "mid":
                row_lo = max(2, interior_h // 5)
                row_hi = max(row_lo, interior_h // 3)
            else:
                row_lo = max(3, interior_h // 3)
                row_hi = max(row_lo, interior_h // 2)
            row_hi = min(row_hi, interior_h - height - 2)
            if row_lo > row_hi:
                return None
            for r in range(row_lo, row_hi + 1):
                for c in range(col_lo, col_hi + 1):
                    options.append((r, c))
        elif role == "left":
            row_lo = 1
            row_hi = min(interior_h - height - 2, interior_h // 2)
            col_lo = 1
            col_hi = min(interior_w - width - 1, 2)
            if row_lo > row_hi or col_lo > col_hi:
                return None
            for r in range(row_lo, row_hi + 1):
                for c in range(col_lo, col_hi + 1):
                    options.append((r, c))
        else:  # right
            row_lo = 1
            row_hi = min(interior_h - height - 2, interior_h // 2)
            col_lo = max(1, interior_w - width - 2)
            col_hi = interior_w - width - 1
            if row_lo > row_hi or col_lo > col_hi:
                return None
            for r in range(row_lo, row_hi + 1):
                for c in range(col_lo, col_hi + 1):
                    options.append((r, c))

        placed = _try_place_object(shape, options, placed_sets)
        if placed is None:
            return None
        anchor_r, anchor_c, coords = placed
        placed_sets.append(coords)
        objects.append({"role": role, "shape": shape, "coords": coords, "anchor": (anchor_r, anchor_c)})

    return objects


def generate_input() -> np.ndarray:
    for _ in range(800):
        interior_h = random.randint(8, 12)
        interior_w = random.randint(8, 12)
        if interior_h == interior_w:
            continue

        top_margin = random.randint(1, 3)
        left_margin = random.randint(1, 3)
        bottom_margin = random.randint(1, 3)
        right_margin = random.randint(1, 3)
        grid_h = top_margin + interior_h + 2 + bottom_margin
        grid_w = left_margin + interior_w + 2 + right_margin
        if max(grid_h, grid_w) > 30:
            continue

        frame = (
            top_margin,
            left_margin,
            top_margin + interior_h + 1,
            left_margin + interior_w + 1,
        )
        count = random.randint(3, 5)
        objects = _build_objects(interior_h, interior_w, count)
        if objects is None:
            continue

        input_grid = _render_input_grid((grid_h, grid_w), frame, objects)
        gray_components = _components_of_color(input_grid, GRAY)
        if len(gray_components) != count:
            continue
        if any(not (3 <= len(comp) <= 8) for comp in gray_components):
            continue

        if _validate_semantics((grid_h, grid_w), frame, interior_h, interior_w, objects):
            return input_grid

    raise RuntimeError("failed to generate a valid puzzle instance")
