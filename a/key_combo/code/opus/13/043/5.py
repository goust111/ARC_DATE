from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
SPRITE_COLORS = (1, 3, 4)
SIDE_ORDER = ("left", "right", "top", "bottom")
GEN_RETRY_LIMIT = 600


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
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _sorted_coords(coords: np.ndarray) -> np.ndarray:
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    return coords[order]


def _normalize(coords: np.ndarray) -> np.ndarray:
    mins = coords.min(axis=0)
    return _sorted_coords(coords - mins)


def _paint_coords(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _frame_local_mask(coords: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return local


def _is_rectangular_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 8 or w < 8:
        return False
    local = _frame_local_mask(coords, (r0, c0, r1, c1))
    border = np.zeros_like(local, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(local, border))


def _outside_wall_cells(
    box: tuple[int, int, int, int], shape: tuple[int, int]
) -> dict[str, list[tuple[int, int]]]:
    h, w = shape
    r0, c0, r1, c1 = box
    cells = {side: [] for side in SIDE_ORDER}
    if c0 > 0:
        for r in range(r0, r1 + 1):
            cells["left"].append((r, c0 - 1))
    if c1 + 1 < w:
        for r in range(r0, r1 + 1):
            cells["right"].append((r, c1 + 1))
    if r0 > 0:
        for c in range(c0, c1 + 1):
            cells["top"].append((r0 - 1, c))
    if r1 + 1 < h:
        for c in range(c0, c1 + 1):
            cells["bottom"].append((r1 + 1, c))
    return cells


def _direction_candidates(
    sprite_box: tuple[int, int, int, int], frame_box: tuple[int, int, int, int]
) -> list[str]:
    sr0, sc0, sr1, sc1 = sprite_box
    fr0, fc0, fr1, fc1 = frame_box
    out: list[str] = []
    if sc1 < fc0 and _interval_overlap(sr0, sr1, fr0, fr1):
        out.append("left")
    if sc0 > fc1 and _interval_overlap(sr0, sr1, fr0, fr1):
        out.append("right")
    if sr1 < fr0 and _interval_overlap(sc0, sc1, fc0, fc1):
        out.append("top")
    if sr0 > fr1 and _interval_overlap(sc0, sc1, fc0, fc1):
        out.append("bottom")
    return out


def _slide_vector(
    sprite_box: tuple[int, int, int, int], frame_box: tuple[int, int, int, int]
) -> tuple[str, int, int, int]:
    sr0, sc0, sr1, sc1 = sprite_box
    fr0, fc0, fr1, fc1 = frame_box
    candidates = _direction_candidates(sprite_box, frame_box)
    assert len(candidates) == 1, "sprite/frame relation must determine one slide direction"
    side = candidates[0]
    if side == "left":
        dist = fc0 - sc1 - 1
        return side, 0, 1, dist
    if side == "right":
        dist = sc0 - fc1 - 1
        return side, 0, -1, dist
    if side == "top":
        dist = fr0 - sr1 - 1
        return side, 1, 0, dist
    dist = sr0 - fr1 - 1
    return side, -1, 0, dist


def _translate(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return coords + np.array([int(dr), int(dc)], dtype=int)


def _mirror_into_frame(
    touch_coords: np.ndarray, frame_box: tuple[int, int, int, int], side: str
) -> np.ndarray:
    r0, c0, r1, c1 = frame_box
    out = touch_coords.copy()
    if side == "left":
        out[:, 1] = 2 * c0 - out[:, 1]
    elif side == "right":
        out[:, 1] = 2 * c1 - out[:, 1]
    elif side == "top":
        out[:, 0] = 2 * r0 - out[:, 0]
    else:
        out[:, 0] = 2 * r1 - out[:, 0]
    assert np.all((out[:, 0] > r0) & (out[:, 0] < r1))
    assert np.all((out[:, 1] > c0) & (out[:, 1] < c1))
    return _sorted_coords(out)


def _path_clear(
    grid: np.ndarray,
    sprite_coords: np.ndarray,
    antenna_cells: set[tuple[int, int]],
    dr: int,
    dc: int,
    dist: int,
) -> bool:
    h, w = grid.shape
    blocked = grid != BG
    for r, c in antenna_cells:
        blocked[r, c] = False
    blocked[sprite_coords[:, 0], sprite_coords[:, 1]] = False
    step_r = int(np.sign(dr))
    step_c = int(np.sign(dc))
    for k in range(1, int(dist) + 1):
        moved = _translate(sprite_coords, step_r * k, step_c * k)
        if (
            np.any(moved[:, 0] < 0)
            or np.any(moved[:, 0] >= h)
            or np.any(moved[:, 1] < 0)
            or np.any(moved[:, 1] >= w)
        ):
            return False
        if np.any(blocked[moved[:, 0], moved[:, 1]]):
            return False
    return True


def _parse_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    frame_comps = _components(g == FRAME)
    frames = []
    for coords in frame_comps:
        if _is_rectangular_frame(coords):
            frames.append({"bbox": _bbox(coords), "coords": _sorted_coords(coords)})
    assert 2 <= len(frames) <= 3, "expected 2-3 gray rectangular frames"

    used_colors: set[int] = set()
    antenna_cells: list[tuple[int, int]] = []
    color_to_frame: dict[int, dict] = {}
    for frame in frames:
        candidates: list[tuple[tuple[int, int], int]] = []
        for cells in _outside_wall_cells(frame["bbox"], (h, w)).values():
            for rr, cc in cells:
                color = int(g[rr, cc])
                if color not in (BG, FRAME):
                    candidates.append(((rr, cc), color))
        assert len(candidates) == 1, "each frame must have exactly one antenna pixel"
        (rr, cc), color = candidates[0]
        assert color in SPRITE_COLORS
        assert color not in used_colors, "antenna colors must be unique"
        used_colors.add(color)
        frame["antenna"] = (rr, cc)
        frame["color"] = color
        antenna_cells.append((rr, cc))
        color_to_frame[color] = frame

    work = g.copy()
    for rr, cc in antenna_cells:
        work[rr, cc] = BG

    sprites: dict[int, dict] = {}
    for color in sorted(color_to_frame):
        comps = _components(work == color)
        assert len(comps) == 1, "each antenna color must match exactly one sprite"
        coords = _sorted_coords(comps[0])
        assert 3 <= len(coords) <= 7, "sprites must have 3-7 pixels"
        sprites[color] = {"coords": coords, "bbox": _bbox(coords), "color": color}

    return g, frames, color_to_frame, sprites, set(antenna_cells)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored sprite to its same-color antenna frame, then mirror it through the touched wall into the frame interior.

    Input structure:
    - The grid has a black background, 2-3 gray one-pixel-thick rectangular frames, and exactly one colored
      antenna pixel on the outside of each frame.
    - Each antenna color is unique among blue, green, and yellow, and there is exactly one 4-connected sprite of
      the same color elsewhere on the canvas.
    - Every sprite lies strictly to the left, right, above, or below its matching frame with row or column overlap,
      so a straight unobstructed slide can reach one outer wall.

    Rule:
    - Match each sprite to the frame whose antenna has the same color, then erase all antenna pixels.
    - For each color in the order blue, green, yellow, slide the whole sprite straight toward its matched frame
      until it is flush against the contacted outer wall without overlapping the frame.
    - Use that contacted wall line as the mirror axis and reflect the slid sprite through the wall so the copy lands
      strictly inside the frame; erase the exterior sprite position and keep the gray frame.

    Key insight:
    - Color chooses the destination frame, the slide determines which wall is touched, and that touched wall is the
      only valid mirror axis for placing the sprite inside.
    """
    g, frames, color_to_frame, sprites, antenna_cells = _parse_scene(grid)
    out = g.copy()
    for rr, cc in antenna_cells:
        out[rr, cc] = BG
    for sprite in sprites.values():
        out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = BG

    for color in sorted(color_to_frame):
        frame = color_to_frame[color]
        sprite = sprites[color]
        side, dr, dc, dist = _slide_vector(sprite["bbox"], frame["bbox"])
        assert dist > 0, "sprite must start away from the frame"
        assert _path_clear(g, sprite["coords"], antenna_cells, dr, dc, dist), "slide path must be unobstructed"
        touch = _translate(sprite["coords"], dr * dist, dc * dist)
        mirrored = _mirror_into_frame(touch, frame["bbox"], side)
        _paint_coords(out, mirrored, color)
    return out


def _box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = box
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _local_lr_symmetric(coords: np.ndarray) -> bool:
    norm = _normalize(coords)
    h = int(norm[:, 0].max()) + 1
    w = int(norm[:, 1].max()) + 1
    mirrored = {(int(r), w - 1 - int(c)) for r, c in norm}
    return {(int(r), int(c)) for r, c in norm} == mirrored


def _local_ud_symmetric(coords: np.ndarray) -> bool:
    norm = _normalize(coords)
    h = int(norm[:, 0].max()) + 1
    mirrored = {(h - 1 - int(r), int(c)) for r, c in norm}
    return {(int(r), int(c)) for r, c in norm} == mirrored


def _random_connected_shape(max_h: int, max_w: int, side: str) -> np.ndarray:
    axis = "lr" if side in ("left", "right") else "ud"
    for _ in range(200):
        area = random.randint(3, 7)
        h = random.randint(2, min(4, max_h))
        w = random.randint(2, min(4, max_w))
        if h * w < area:
            continue
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c, h, w):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue
        coords = np.array(sorted(cells), dtype=int)
        coords = _normalize(coords)
        if axis == "lr" and _local_lr_symmetric(coords):
            continue
        if axis == "ud" and _local_ud_symmetric(coords):
            continue
        return coords
    raise RuntimeError("failed to generate asymmetric connected shape")


def _orient_local_shape(coords: np.ndarray, side: str) -> np.ndarray:
    norm = _normalize(coords)
    h = int(norm[:, 0].max()) + 1
    w = int(norm[:, 1].max()) + 1
    out = norm.copy()
    if side == "right":
        out[:, 1] = w - 1 - out[:, 1]
    elif side == "bottom":
        out[:, 0] = h - 1 - out[:, 0]
    return _sorted_coords(out)


def _place_internal_shape(
    frame_box: tuple[int, int, int, int], side: str
) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = frame_box
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    for _ in range(200):
        local = _orient_local_shape(_random_connected_shape(inner_h, inner_w, side), side)
        lh = int(local[:, 0].max()) + 1
        lw = int(local[:, 1].max()) + 1
        if side in ("left", "right"):
            row0 = random.randint(r0 + 1, r1 - lh)
            col0 = c0 + 1 if side == "left" else c1 - lw
        else:
            row0 = r0 + 1 if side == "top" else r1 - lh
            col0 = random.randint(c0 + 1, c1 - lw)
        absolute = local + np.array([row0, col0], dtype=int)
        if np.all((absolute[:, 0] > r0) & (absolute[:, 0] < r1)) and np.all(
            (absolute[:, 1] > c0) & (absolute[:, 1] < c1)
        ):
            return absolute, local
    raise RuntimeError("failed to place interior shape")


def _touch_from_inside(
    inside_coords: np.ndarray, frame_box: tuple[int, int, int, int], side: str
) -> np.ndarray:
    r0, c0, r1, c1 = frame_box
    touch = inside_coords.copy()
    if side == "left":
        touch[:, 1] = 2 * c0 - touch[:, 1]
    elif side == "right":
        touch[:, 1] = 2 * c1 - touch[:, 1]
    elif side == "top":
        touch[:, 0] = 2 * r0 - touch[:, 0]
    else:
        touch[:, 0] = 2 * r1 - touch[:, 0]
    return _sorted_coords(touch)


def _inward_step(side: str) -> tuple[int, int]:
    if side == "left":
        return 0, 1
    if side == "right":
        return 0, -1
    if side == "top":
        return 1, 0
    return -1, 0


def _outward_step(side: str) -> tuple[int, int]:
    dr, dc = _inward_step(side)
    return -dr, -dc


def _max_outward_distance(coords: np.ndarray, side: str, shape: tuple[int, int]) -> int:
    h, w = shape
    if side == "left":
        return int(coords[:, 1].min())
    if side == "right":
        return int((w - 1) - coords[:, 1].max())
    if side == "top":
        return int(coords[:, 0].min())
    return int((h - 1) - coords[:, 0].max())


def _path_positions(start_coords: np.ndarray, side: str, dist: int) -> np.ndarray:
    dr, dc = _inward_step(side)
    parts = [_translate(start_coords, dr * k, dc * k) for k in range(dist + 1)]
    return np.vstack(parts)


def _boxes_too_close(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int
) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_frames(h: int, w: int, count: int) -> list[tuple[int, int, int, int]] | None:
    for _ in range(300):
        boxes: list[tuple[int, int, int, int]] = []
        ok = True
        for _idx in range(count):
            placed = False
            for _ in range(300):
                fh = random.randint(8, 10)
                fw = random.randint(8, 10)
                if fh + 2 > h or fw + 2 > w:
                    continue
                r0 = random.randint(1, h - fh - 1)
                c0 = random.randint(1, w - fw - 1)
                box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
                if all(not _boxes_too_close(box, other, gap=2) for other in boxes):
                    boxes.append(box)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return boxes
    return None


def _place_antennas(
    grid: np.ndarray,
    specs: list[dict],
    blocked: np.ndarray,
) -> bool:
    h, w = grid.shape
    for spec in specs:
        box = spec["frame_box"]
        sprite_side = spec["side"]
        choices = list(SIDE_ORDER)
        random.shuffle(choices)
        if choices[0] == sprite_side and len(choices) > 1:
            choices[0], choices[1] = choices[1], choices[0]
        placed = False
        wall_cells = _outside_wall_cells(box, (h, w))
        for side in choices:
            cells = wall_cells[side][:]
            random.shuffle(cells)
            for r, c in cells:
                if not blocked[r, c] and grid[r, c] == BG:
                    grid[r, c] = spec["color"]
                    spec["antenna"] = (r, c)
                    blocked[r, c] = True
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return False
    return True


def _build_input_and_expected(specs: list[dict], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["frame_box"]
        inp[r0, c0 : c1 + 1] = FRAME
        inp[r1, c0 : c1 + 1] = FRAME
        inp[r0 : r1 + 1, c0] = FRAME
        inp[r0 : r1 + 1, c1] = FRAME
        out[r0, c0 : c1 + 1] = FRAME
        out[r1, c0 : c1 + 1] = FRAME
        out[r0 : r1 + 1, c0] = FRAME
        out[r0 : r1 + 1, c1] = FRAME
        _paint_coords(inp, spec["start_coords"], spec["color"])
        _paint_coords(out, spec["inside_coords"], spec["color"])
    blocked = inp != BG
    if not _place_antennas(inp, specs, blocked):
        raise RuntimeError("failed to place antennas")
    return inp, out


def _nearest_frame_color(
    sprite_box: tuple[int, int, int, int], specs: list[dict]
) -> int:
    sr, sc = _box_center(sprite_box)
    best_color = -1
    best_dist = None
    for spec in specs:
        fr, fc = _box_center(spec["frame_box"])
        dist = (sr - fr) ** 2 + (sc - fc) ** 2
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_color = spec["color"]
    return best_color


def _has_geometry_decoy(specs: list[dict]) -> bool:
    for spec in specs:
        sprite_box = _bbox(spec["start_coords"])
        candidate_count = 0
        for other in specs:
            if _direction_candidates(sprite_box, other["frame_box"]):
                candidate_count += 1
        if candidate_count > 1:
            return True
        if _nearest_frame_color(sprite_box, specs) != spec["color"]:
            return True
    return False


def _check_input_semantics(inp: np.ndarray) -> bool:
    try:
        g, frames, color_to_frame, sprites, antennas = _parse_scene(inp)
    except AssertionError:
        return False

    if len(frames) != len(color_to_frame) or len(color_to_frame) != len(sprites):
        return False

    for color in sorted(color_to_frame):
        frame = color_to_frame[color]
        sprite = sprites[color]
        try:
            side, dr, dc, dist = _slide_vector(sprite["bbox"], frame["bbox"])
        except AssertionError:
            return False
        if dist < 2:
            return False
        if not _path_clear(g, sprite["coords"], antennas, dr, dc, dist):
            return False
        mirrored = _mirror_into_frame(_translate(sprite["coords"], dr * dist, dc * dist), frame["bbox"], side)
        local = _normalize(mirrored)
        if side in ("left", "right"):
            if _local_lr_symmetric(local):
                return False
        else:
            if _local_ud_symmetric(local):
                return False
    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(20, 27)
    w = random.randint(20, 27)
    count = random.choice((2, 3))
    boxes = _place_frames(h, w, count)
    if boxes is None:
        return None

    colors = random.sample(list(SPRITE_COLORS), count)
    specs: list[dict] = []
    path_blocked = np.zeros((h, w), dtype=bool)
    for box in boxes:
        r0, c0, r1, c1 = box
        path_blocked[r0, c0 : c1 + 1] = True
        path_blocked[r1, c0 : c1 + 1] = True
        path_blocked[r0 : r1 + 1, c0] = True
        path_blocked[r0 : r1 + 1, c1] = True

    order = list(range(count))
    random.shuffle(order)
    for idx in order:
        color = colors[idx]
        box = boxes[idx]
        success = False
        side_choices = list(SIDE_ORDER)
        random.shuffle(side_choices)
        for side in side_choices:
            for _ in range(120):
                inside_coords, local = _place_internal_shape(box, side)
                touch_coords = _touch_from_inside(inside_coords, box, side)
                max_dist = _max_outward_distance(touch_coords, side, (h, w))
                if max_dist < 2:
                    continue
                upper = min(5, max_dist)
                distances = list(range(2, upper + 1))
                random.shuffle(distances)
                if not distances:
                    continue
                odr, odc = _outward_step(side)
                for dist in distances:
                    start_coords = _translate(touch_coords, odr * dist, odc * dist)
                    if (
                        np.any(start_coords[:, 0] < 0)
                        or np.any(start_coords[:, 0] >= h)
                        or np.any(start_coords[:, 1] < 0)
                        or np.any(start_coords[:, 1] >= w)
                    ):
                        continue
                    path_coords = _path_positions(start_coords, side, dist)
                    if np.any(path_blocked[path_coords[:, 0], path_coords[:, 1]]):
                        continue
                    spec = {
                        "color": color,
                        "frame_box": box,
                        "side": side,
                        "inside_coords": inside_coords,
                        "local_inside": local,
                        "start_coords": start_coords,
                        "touch_coords": touch_coords,
                    }
                    specs.append(spec)
                    path_blocked[path_coords[:, 0], path_coords[:, 1]] = True
                    success = True
                    break
                if success:
                    break
            if success:
                break
        if not success:
            return None

    try:
        inp, expected = _build_input_and_expected(specs, (h, w))
    except RuntimeError:
        return None

    if not _has_geometry_decoy(specs):
        return None
    if not _check_input_semantics(inp):
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
