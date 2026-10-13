from __future__ import annotations

import random
from collections import deque

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 500


SPRITE_TEMPLATES = (
    np.array([[1, 1]], dtype=int),
    np.array([[1], [1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [1, 1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
)

HORIZONTAL_DIRS = ("left", "right")
VERTICAL_DIRS = ("up", "down")
ALL_DIRS = HORIZONTAL_DIRS + VERTICAL_DIRS


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_h(box: BBox) -> int:
    return int(box[2] - box[0] + 1)


def _bbox_w(box: BBox) -> int:
    return int(box[3] - box[1] + 1)


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[Coord] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _frame_cells(box: BBox) -> set[Coord]:
    r0, c0, r1, c1 = box
    cells: set[Coord] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _is_hollow_rect(coords: np.ndarray) -> bool:
    box = _bbox_from_coords(coords)
    h = _bbox_h(box)
    w = _bbox_w(box)
    if h < 4 or w < 4:
        return False
    cell_set = {tuple(map(int, rc)) for rc in coords}
    return cell_set == _frame_cells(box)


def _inner_box(box: BBox) -> BBox:
    r0, c0, r1, c1 = box
    return r0 + 1, c0 + 1, r1 - 1, c1 - 1


def _bbox_inside(inner: BBox, outer: BBox) -> bool:
    return outer[0] <= inner[0] and inner[2] <= outer[2] and outer[1] <= inner[1] and inner[3] <= outer[3]


def _shape_coords(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    coords = np.argwhere(mask > 0).astype(int)
    coords[:, 0] += int(top)
    coords[:, 1] += int(left)
    return coords


def _shift_coords(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    moved = coords.copy()
    moved[:, 0] += int(dr)
    moved[:, 1] += int(dc)
    return moved


def _render_frame(grid: np.ndarray, box: BBox, color: int) -> None:
    for r, c in _frame_cells(box):
        grid[r, c] = int(color)


def _render_coords(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _classify(grid: np.ndarray) -> tuple[dict[int, dict], dict[int, dict]]:
    colors = sorted(int(x) for x in np.unique(grid) if int(x) != BG)
    frames: dict[int, dict] = {}
    sprites: dict[int, dict] = {}
    for color in colors:
        comps = _components(grid == color)
        frame_comps = [coords for coords in comps if _is_hollow_rect(coords)]
        sprite_comps = [coords for coords in comps if not _is_hollow_rect(coords)]
        assert len(frame_comps) == 1, f"color {color} must have exactly one hollow frame"
        assert len(sprite_comps) == 1, f"color {color} must have exactly one sprite"
        frames[color] = {"coords": frame_comps[0], "bbox": _bbox_from_coords(frame_comps[0])}
        sprites[color] = {"coords": sprite_comps[0], "bbox": _bbox_from_coords(sprite_comps[0])}
    assert len(frames) == len(sprites) >= 3
    return frames, sprites


def _movement_to_frame(sprite_box: BBox, frame_box: BBox) -> tuple[str, int]:
    candidates: list[tuple[str, int]] = []
    if sprite_box[2] < frame_box[0] and _intervals_overlap(sprite_box[1], sprite_box[3], frame_box[1], frame_box[3]):
        candidates.append(("down", frame_box[0] - sprite_box[2] - 1))
    if sprite_box[0] > frame_box[2] and _intervals_overlap(sprite_box[1], sprite_box[3], frame_box[1], frame_box[3]):
        candidates.append(("up", sprite_box[0] - frame_box[2] - 1))
    if sprite_box[3] < frame_box[1] and _intervals_overlap(sprite_box[0], sprite_box[2], frame_box[0], frame_box[2]):
        candidates.append(("right", frame_box[1] - sprite_box[3] - 1))
    if sprite_box[1] > frame_box[3] and _intervals_overlap(sprite_box[0], sprite_box[2], frame_box[0], frame_box[2]):
        candidates.append(("left", sprite_box[1] - frame_box[3] - 1))
    assert len(candidates) == 1, "sprite must lie in a unique cardinal direction from its paired frame"
    direction, distance = candidates[0]
    assert distance >= 1, "sprite must start away from its paired frame"
    return direction, int(distance)


def _direction_delta(direction: str) -> tuple[int, int]:
    if direction == "up":
        return -1, 0
    if direction == "down":
        return 1, 0
    if direction == "left":
        return 0, -1
    if direction == "right":
        return 0, 1
    raise AssertionError(f"unknown direction: {direction}")


def _simulate(grid: np.ndarray) -> dict:
    frames, sprites = _classify(grid)
    moved: dict[int, dict] = {}
    for color, sprite in sprites.items():
        direction, distance = _movement_to_frame(sprite["bbox"], frames[color]["bbox"])
        dr, dc = _direction_delta(direction)
        shifted = _shift_coords(sprite["coords"], dr * distance, dc * distance)
        moved[color] = {
            "coords": shifted,
            "bbox": _bbox_from_coords(shifted),
            "direction": direction,
            "distance": distance,
        }

    erased: set[int] = set()
    for color, sprite in moved.items():
        for other_color, frame in frames.items():
            if other_color == color:
                continue
            if _bbox_inside(sprite["bbox"], _inner_box(frame["bbox"])):
                erased.add(color)
                break

    return {"frames": frames, "sprites": sprites, "moved": moved, "erased": erased}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored sprite straight into contact with its same-color hollow frame, then erase sprites newly enclosed by another frame.

    Input structure:
    - The background is black.
    - Each non-black color appears in exactly two 4-connected components: one hollow rectangular frame and one small solid sprite.
    - Every sprite lies strictly above, below, left, or right of its same-color frame and shares rows or columns with it so it can slide along one straight axis.

    Rule:
    - Pair every sprite with the hollow frame of the same color.
    - Move each sprite straight toward its paired frame until the sprite stops with an edge flush against the frame's outer wall, leaving a one-cell contact gap of zero.
    - After all slides, compute each sprite's new bounding box.
    - Erase any sprite whose new bounding box lies completely inside the interior rectangle of a different-color hollow frame.
    - Leave all frames fixed and keep every non-erased sprite at its slid position.

    Key insight:
    - Color identifies the docking partner, but the deletion test is decided only after the docking slide rearranges the spatial layout.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    info = _simulate(g)
    out = np.zeros_like(g)
    for color, frame in info["frames"].items():
        _render_frame(out, frame["bbox"], color)
    for color, sprite in info["moved"].items():
        if color not in info["erased"]:
            _render_coords(out, sprite["coords"], color)
    return out


def _expanded_intersects(a: BBox, b: BBox, pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _coords_in_bounds(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    return bool(
        np.all(coords[:, 0] >= 0)
        and np.all(coords[:, 1] >= 0)
        and np.all(coords[:, 0] < shape[0])
        and np.all(coords[:, 1] < shape[1])
    )


def _pick_template(axis: str, *, target: bool) -> np.ndarray:
    choices = []
    for tmpl in SPRITE_TEMPLATES:
        h, w = tmpl.shape
        if axis == "horizontal":
            if w <= 2:
                choices.append(tmpl)
        else:
            if h <= 2:
                choices.append(tmpl)
        if not target:
            choices.append(tmpl)
    return random.choice(choices)


def _naive_pre_slide_erased(grid: np.ndarray) -> set[int]:
    frames, sprites = _classify(grid)
    erased: set[int] = set()
    for color, sprite in sprites.items():
        for other_color, frame in frames.items():
            if other_color == color:
                continue
            if _bbox_inside(sprite["bbox"], _inner_box(frame["bbox"])):
                erased.add(color)
                break
    return erased


def _target_layout(
    grid_shape: tuple[int, int],
    container_color: int,
    target_color: int,
    occupied_frame_boxes: list[BBox],
    input_occ: set[Coord],
    final_occ: set[Coord],
) -> tuple[dict, dict] | None:
    h, w = grid_shape
    direction = random.choice(ALL_DIRS)
    axis = "horizontal" if direction in HORIZONTAL_DIRS else "vertical"
    shape = _pick_template(axis, target=True)
    sh, sw = shape.shape
    frame_h = 4
    frame_w = 4
    cont_h = 6 if axis == "horizontal" else sh + frame_h + 2
    cont_w = sw + frame_w + 2 if axis == "horizontal" else 6
    if cont_h > 8 or cont_w > 8:
        return None

    gap = random.randint((sw if axis == "horizontal" else sh) + 1, (sw if axis == "horizontal" else sh) + 4)

    r_min, r_max = 1, h - cont_h - 2
    c_min, c_max = 1, w - cont_w - 2
    if direction == "left":
        c_min = max(c_min, gap)
    elif direction == "right":
        c_max = min(c_max, w - cont_w - gap - sw - 1)
    elif direction == "up":
        r_min = max(r_min, gap)
    else:
        r_max = min(r_max, h - cont_h - gap - sh - 1)
    if r_min > r_max or c_min > c_max:
        return None

    for _ in range(120):
        cont_r = random.randint(r_min, r_max)
        cont_c = random.randint(c_min, c_max)
        container_box = (cont_r, cont_c, cont_r + cont_h - 1, cont_c + cont_w - 1)
        if any(_expanded_intersects(container_box, other, pad=1) for other in occupied_frame_boxes):
            continue

        if direction == "left":
            frame_top = cont_r + 1
            frame_left = cont_c + 1 + sw
            row_off = random.randint(0, frame_h - sh)
            final_top = frame_top + row_off
            final_left = frame_left - sw
            input_top = final_top
            input_left = final_left - gap
        elif direction == "right":
            frame_top = cont_r + 1
            frame_left = cont_c + 1
            row_off = random.randint(0, frame_h - sh)
            final_top = frame_top + row_off
            final_left = frame_left + frame_w
            input_top = final_top
            input_left = final_left + gap
        elif direction == "up":
            frame_top = cont_r + 1 + sh
            frame_left = cont_c + 1
            col_off = random.randint(0, frame_w - sw)
            final_top = frame_top - sh
            final_left = frame_left + col_off
            input_top = final_top - gap
            input_left = final_left
        else:
            frame_top = cont_r + 1
            frame_left = cont_c + 1
            col_off = random.randint(0, frame_w - sw)
            final_top = frame_top + frame_h
            final_left = frame_left + col_off
            input_top = final_top + gap
            input_left = final_left

        target_frame_box = (frame_top, frame_left, frame_top + frame_h - 1, frame_left + frame_w - 1)
        final_coords = _shape_coords(shape, final_top, final_left)
        input_coords = _shape_coords(shape, input_top, input_left)
        if not _coords_in_bounds(input_coords, grid_shape):
            continue
        if not _bbox_inside(_bbox_from_coords(final_coords), _inner_box(container_box)):
            continue

        frame_cells = _frame_cells(container_box) | _frame_cells(target_frame_box)
        if any(cell in input_occ for cell in frame_cells):
            continue
        if any(tuple(map(int, rc)) in input_occ for rc in input_coords):
            continue
        if any(tuple(map(int, rc)) in frame_cells for rc in input_coords):
            continue
        if any(tuple(map(int, rc)) in final_occ for rc in final_coords):
            continue
        if any(tuple(map(int, rc)) in frame_cells for rc in final_coords):
            continue

        container = {"color": container_color, "bbox": container_box, "role": "container"}
        target = {
            "color": target_color,
            "shape": shape,
            "frame_bbox": target_frame_box,
            "input_coords": input_coords,
            "final_coords": final_coords,
            "direction": direction,
            "distance": gap,
        }
        return container, target
    return None


def _try_place_sprite_for_fixed_frame(
    grid_shape: tuple[int, int],
    color: int,
    frame_box: BBox,
    input_occ: set[Coord],
    final_occ: set[Coord],
    forbid_inner_boxes: list[BBox],
    forced_axis: str | None,
) -> dict | None:
    h, w = grid_shape
    r0, c0, r1, c1 = frame_box
    frame_h = _bbox_h(frame_box)
    frame_w = _bbox_w(frame_box)
    frame_cell_set = _frame_cells(frame_box)
    for _ in range(300):
        axis = forced_axis if forced_axis is not None else random.choice(("horizontal", "vertical"))
        direction = random.choice(HORIZONTAL_DIRS if axis == "horizontal" else VERTICAL_DIRS)
        shape = _pick_template(axis, target=False)
        sh, sw = shape.shape
        if direction in HORIZONTAL_DIRS and sh > frame_h:
            continue
        if direction in VERTICAL_DIRS and sw > frame_w:
            continue

        if direction == "left":
            row_off = random.randint(0, frame_h - sh)
            final_top = r0 + row_off
            final_left = c0 - sw
            gap = random.randint(1, 4)
            input_top = final_top
            input_left = final_left - gap
        elif direction == "right":
            row_off = random.randint(0, frame_h - sh)
            final_top = r0 + row_off
            final_left = c1 + 1
            gap = random.randint(1, 4)
            input_top = final_top
            input_left = final_left + gap
        elif direction == "up":
            col_off = random.randint(0, frame_w - sw)
            final_top = r0 - sh
            final_left = c0 + col_off
            gap = random.randint(1, 4)
            input_top = final_top - gap
            input_left = final_left
        else:
            col_off = random.randint(0, frame_w - sw)
            final_top = r1 + 1
            final_left = c0 + col_off
            gap = random.randint(1, 4)
            input_top = final_top + gap
            input_left = final_left

        input_coords = _shape_coords(shape, input_top, input_left)
        final_coords = _shape_coords(shape, final_top, final_left)
        if not _coords_in_bounds(input_coords, grid_shape):
            continue
        if not _coords_in_bounds(final_coords, grid_shape):
            continue
        if any(tuple(map(int, rc)) in input_occ for rc in input_coords):
            continue
        if any(tuple(map(int, rc)) in frame_cell_set for rc in input_coords):
            continue
        if any(tuple(map(int, rc)) in frame_cell_set for rc in final_coords):
            continue
        if any(tuple(map(int, rc)) in final_occ for rc in final_coords):
            continue

        input_box = _bbox_from_coords(input_coords)
        final_box = _bbox_from_coords(final_coords)
        if any(_bbox_inside(input_box, inner) for inner in forbid_inner_boxes):
            continue
        if any(_bbox_inside(final_box, inner) for inner in forbid_inner_boxes):
            continue

        return {
            "color": color,
            "shape": shape,
            "frame_bbox": frame_box,
            "input_coords": input_coords,
            "final_coords": final_coords,
            "direction": direction,
            "distance": gap,
        }
    return None


def _try_place_survivor(
    grid_shape: tuple[int, int],
    color: int,
    occupied_frame_boxes: list[BBox],
    input_occ: set[Coord],
    final_occ: set[Coord],
    forbid_inner_boxes: list[BBox],
    forced_axis: str | None,
) -> dict | None:
    h, w = grid_shape
    for _ in range(300):
        axis = forced_axis if forced_axis is not None else random.choice(("horizontal", "vertical"))
        direction = random.choice(HORIZONTAL_DIRS if axis == "horizontal" else VERTICAL_DIRS)
        shape = _pick_template(axis, target=False)
        sh, sw = shape.shape
        frame_h = random.randint(max(4, sh), 6) if axis == "horizontal" else random.randint(4, 6)
        frame_w = random.randint(4, 6) if axis == "horizontal" else random.randint(max(4, sw), 6)

        r0 = random.randint(1, h - frame_h - 2)
        c0 = random.randint(1, w - frame_w - 2)
        frame_box = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
        if any(_expanded_intersects(frame_box, other, pad=1) for other in occupied_frame_boxes):
            continue

        if direction == "left":
            row_off = random.randint(0, frame_h - sh)
            final_top = r0 + row_off
            final_left = c0 - sw
            gap = random.randint(1, 4)
            input_top = final_top
            input_left = final_left - gap
        elif direction == "right":
            row_off = random.randint(0, frame_h - sh)
            final_top = r0 + row_off
            final_left = c0 + frame_w
            gap = random.randint(1, 4)
            input_top = final_top
            input_left = final_left + gap
        elif direction == "up":
            col_off = random.randint(0, frame_w - sw)
            final_top = r0 - sh
            final_left = c0 + col_off
            gap = random.randint(1, 4)
            input_top = final_top - gap
            input_left = final_left
        else:
            col_off = random.randint(0, frame_w - sw)
            final_top = r0 + frame_h
            final_left = c0 + col_off
            gap = random.randint(1, 4)
            input_top = final_top + gap
            input_left = final_left

        input_coords = _shape_coords(shape, input_top, input_left)
        final_coords = _shape_coords(shape, final_top, final_left)
        if not _coords_in_bounds(input_coords, grid_shape):
            continue
        if not _coords_in_bounds(final_coords, grid_shape):
            continue

        frame_cell_set = _frame_cells(frame_box)
        if any(cell in input_occ for cell in frame_cell_set):
            continue
        if any(tuple(map(int, rc)) in input_occ for rc in input_coords):
            continue
        if any(tuple(map(int, rc)) in frame_cell_set for rc in input_coords):
            continue
        if any(tuple(map(int, rc)) in frame_cell_set for rc in final_coords):
            continue
        if any(tuple(map(int, rc)) in final_occ for rc in final_coords):
            continue

        final_box = _bbox_from_coords(final_coords)
        if any(_bbox_inside(final_box, inner) for inner in forbid_inner_boxes):
            continue
        if any(_bbox_inside(_bbox_from_coords(input_coords), inner) for inner in forbid_inner_boxes):
            continue

        return {
            "color": color,
            "frame_bbox": frame_box,
            "shape": shape,
            "input_coords": input_coords,
            "final_coords": final_coords,
            "direction": direction,
            "distance": gap,
        }
    return None


def _build_candidate() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    object_count = random.randint(3, 4)
    colors = random.sample(PALETTE, object_count)

    input_occ: set[Coord] = set()
    final_occ: set[Coord] = set()
    frame_boxes: list[BBox] = []
    frames: list[tuple[int, BBox]] = []
    sprites: list[dict] = []

    container_color = colors[0]
    target_color = colors[1]
    target_pack = _target_layout((h, w), container_color, target_color, frame_boxes, input_occ, final_occ)
    if target_pack is None:
        return None
    container, target = target_pack

    frame_boxes.append(container["bbox"])
    frames.append((container_color, container["bbox"]))
    for cell in _frame_cells(container["bbox"]):
        input_occ.add(cell)

    frame_boxes.append(target["frame_bbox"])
    frames.append((target_color, target["frame_bbox"]))
    for cell in _frame_cells(target["frame_bbox"]):
        input_occ.add(cell)

    for rc in target["input_coords"]:
        input_occ.add((int(rc[0]), int(rc[1])))
    for rc in target["final_coords"]:
        final_occ.add((int(rc[0]), int(rc[1])))
    sprites.append(target)

    target_axis = "horizontal" if target["direction"] in HORIZONTAL_DIRS else "vertical"
    need_vertical = target_axis == "horizontal"
    need_horizontal = target_axis == "vertical"

    fixed_axis = "vertical" if need_vertical else "horizontal"
    container_sprite = _try_place_sprite_for_fixed_frame(
        (h, w),
        container_color,
        container["bbox"],
        input_occ,
        final_occ,
        [_inner_box(target["frame_bbox"])],
        fixed_axis,
    )
    if container_sprite is None:
        return None
    for rc in container_sprite["input_coords"]:
        input_occ.add((int(rc[0]), int(rc[1])))
    for rc in container_sprite["final_coords"]:
        final_occ.add((int(rc[0]), int(rc[1])))
    sprites.append(container_sprite)
    if fixed_axis == "vertical":
        need_vertical = False
    else:
        need_horizontal = False

    remaining_colors = colors[2:object_count]
    for color in remaining_colors:
        forced_axis = None
        if need_vertical:
            forced_axis = "vertical"
            need_vertical = False
        elif need_horizontal:
            forced_axis = "horizontal"
            need_horizontal = False

        inner_boxes = [_inner_box(box) for _, box in frames]
        placed = _try_place_survivor((h, w), color, frame_boxes, input_occ, final_occ, inner_boxes, forced_axis)
        if placed is None:
            return None
        frame_boxes.append(placed["frame_bbox"])
        frames.append((color, placed["frame_bbox"]))
        for cell in _frame_cells(placed["frame_bbox"]):
            input_occ.add(cell)
        for rc in placed["input_coords"]:
            input_occ.add((int(rc[0]), int(rc[1])))
        for rc in placed["final_coords"]:
            final_occ.add((int(rc[0]), int(rc[1])))
        sprites.append(placed)

    grid = np.zeros((h, w), dtype=int)
    for color, box in frames:
        _render_frame(grid, box, color)
    for sprite in sprites:
        _render_coords(grid, sprite["input_coords"], sprite["color"])
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        info = _simulate(grid)
        out = main(grid)
        assert out.shape == grid.shape
        assert len(info["frames"]) in {3, 4}
        assert len(info["erased"]) >= 1
        assert len(info["erased"]) < len(info["frames"])
        assert np.any(out != grid)

        axes = {"horizontal" if v["direction"] in HORIZONTAL_DIRS else "vertical" for v in info["moved"].values()}
        assert axes == {"horizontal", "vertical"}
        assert any(v["distance"] >= 2 for v in info["moved"].values())

        pre = _naive_pre_slide_erased(grid)
        assert info["erased"] != pre
        assert len(info["erased"] - pre) >= 1

        frame_boxes = {color: frame["bbox"] for color, frame in info["frames"].items()}
        moved_boxes = {color: sprite["bbox"] for color, sprite in info["moved"].items()}
        for color, moved_box in moved_boxes.items():
            inside_any = any(
                _bbox_inside(moved_box, _inner_box(frame_box))
                for other_color, frame_box in frame_boxes.items()
                if other_color != color
            )
            assert inside_any == (color in info["erased"])

        colors = set(int(x) for x in np.unique(grid) if int(x) != 0)
        assert len(colors) == len(info["frames"]) == len(info["sprites"])
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is None:
            continue
        if _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid sliding-docking containment instance")
