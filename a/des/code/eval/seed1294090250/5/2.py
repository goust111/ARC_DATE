from __future__ import annotations

import random

import numpy as np


MARKER_COLOR = 2
SIDES = ("top", "right", "bottom", "left")
SIDE_TO_INDEX = {side: idx for idx, side in enumerate(SIDES)}
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def connected_components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        neigh = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        neigh = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )

    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in neigh:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(pts)
    return parts


def bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def points_to_mask(points: list[tuple[int, int]], box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    if box is None:
        box = bbox(points)
    r0, c0, r1, c1 = box
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        out[r - r0, c - c0] = True
    return out


def rotate_side(side: str, k_ccw: int) -> str:
    return SIDES[(SIDE_TO_INDEX[side] - k_ccw) % 4]


def rotation_to_align(src_side: str, dst_side: str) -> int:
    return (SIDE_TO_INDEX[src_side] - SIDE_TO_INDEX[dst_side]) % 4


def side_marker_for_bbox(
    marker: tuple[int, int],
    box: tuple[int, int, int, int],
    occupied: set[tuple[int, int]],
) -> str | None:
    mr, mc = marker
    r0, c0, r1, c1 = box
    if mr == r0 - 1 and c0 <= mc <= c1 and (r0, mc) in occupied:
        return "top"
    if mr == r1 + 1 and c0 <= mc <= c1 and (r1, mc) in occupied:
        return "bottom"
    if mc == c0 - 1 and r0 <= mr <= r1 and (mr, c0) in occupied:
        return "left"
    if mc == c1 + 1 and r0 <= mr <= r1 and (mr, c1) in occupied:
        return "right"
    return None


def is_rectangular_frame(
    grid: np.ndarray,
    color: int,
    points: list[tuple[int, int]],
    bg: int,
) -> bool:
    r0, c0, r1, c1 = bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * (h + w) - 4
    if len(points) != expected:
        return False
    point_set = set(points)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in point_set):
                return False
            if not on_border and int(grid[r, c]) != bg:
                return False
    return True


def extract_frames(grid: np.ndarray, bg: int) -> tuple[dict[int, dict], set[tuple[int, int]], set[tuple[int, int]]]:
    red_points = [tuple(map(int, pt)) for pt in np.argwhere(grid == MARKER_COLOR)]
    frames: dict[int, dict] = {}
    frame_cells: set[tuple[int, int]] = set()
    frame_markers: set[tuple[int, int]] = set()
    colors = sorted(set(map(int, grid.flat)) - {bg, MARKER_COLOR})
    for color in colors:
        for pts in connected_components(grid == color, connectivity=4):
            if not is_rectangular_frame(grid, color, pts, bg):
                continue
            box = bbox(pts)
            point_set = set(pts)
            marker_matches: list[tuple[tuple[int, int], str]] = []
            for marker in red_points:
                side = side_marker_for_bbox(marker, box, point_set)
                if side is not None:
                    marker_matches.append((marker, side))
            if len(marker_matches) != 1:
                continue
            marker_pos, side = marker_matches[0]
            frames[color] = {
                "color": color,
                "bbox": box,
                "side": side,
            }
            frame_cells.update(point_set)
            frame_markers.add(marker_pos)
    return frames, frame_cells, frame_markers


def extract_fragments(
    grid: np.ndarray,
    bg: int,
    frame_cells: set[tuple[int, int]],
    frame_markers: set[tuple[int, int]],
) -> dict[int, dict]:
    residual = grid != bg
    for r, c in frame_cells:
        residual[r, c] = False
    for r, c in frame_markers:
        residual[r, c] = False

    fragments: dict[int, dict] = {}
    for pts in connected_components(residual, connectivity=4):
        colors = {int(grid[r, c]) for r, c in pts}
        if MARKER_COLOR not in colors:
            continue
        body_colors = colors - {MARKER_COLOR}
        if len(body_colors) != 1:
            continue
        color = next(iter(body_colors))
        body = [(r, c) for r, c in pts if int(grid[r, c]) == color]
        markers = [(r, c) for r, c in pts if int(grid[r, c]) == MARKER_COLOR]
        if not body or len(markers) != 1:
            continue
        box = bbox(body)
        body_set = set(body)
        side = side_marker_for_bbox(markers[0], box, body_set)
        if side is None:
            continue
        fragments[color] = {
            "color": color,
            "bbox": box,
            "side": side,
            "mask": points_to_mask(body, box),
            "body": body,
        }
    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored fragment into the matching frame cavity after rotating it so that the
    fragment's red marker side matches the frame's red marker side.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frames, frame_cells, frame_markers = extract_frames(g, bg)
    fragments = extract_fragments(g, bg, frame_cells, frame_markers)

    out = g.copy()
    out[out == MARKER_COLOR] = bg
    for frag in fragments.values():
        for r, c in frag["body"]:
            out[r, c] = bg

    for color, frame in frames.items():
        frag = fragments.get(color)
        if frag is None:
            continue
        r0, c0, r1, c1 = frame["bbox"]
        cavity_shape = (r1 - r0 - 1, c1 - c0 - 1)
        k = rotation_to_align(frag["side"], frame["side"])
        rotated = np.rot90(frag["mask"], k)
        if rotated.shape != cavity_shape:
            continue
        patch = out[r0 + 1 : r1, c0 + 1 : c1]
        patch[rotated] = color
    return out


def mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    seen = np.zeros_like(mask, dtype=bool)
    start_r, start_c = map(int, pts[0])
    stack = [(start_r, start_c)]
    seen[start_r, start_c] = True
    total = 0
    while stack:
        r, c = stack.pop()
        total += 1
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return total == int(mask.sum())


def touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def generate_irregular_mask(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    full_area = h * w
    if full_area <= 1:
        return None
    min_area = max(3, h + w - 1)
    max_area = full_area - 1
    if min_area > max_area:
        return None

    for _ in range(80):
        target_area = rng.randint(min_area, max_area)
        mask = np.ones((h, w), dtype=bool)
        steps = full_area - target_area
        ok = True
        for _step in range(steps):
            removable: list[tuple[int, int]] = []
            for r in range(h):
                for c in range(w):
                    if not mask[r, c]:
                        continue
                    trial = mask.copy()
                    trial[r, c] = False
                    if not touches_all_sides(trial):
                        continue
                    if not mask_connected(trial):
                        continue
                    removable.append((r, c))
            if not removable:
                ok = False
                break
            rr, cc = rng.choice(removable)
            mask[rr, cc] = False
        if not ok:
            continue
        if not touches_all_sides(mask) or not mask_connected(mask):
            continue
        if len({mask_signature(np.rot90(mask, k)) for k in range(4)}) < 4:
            continue
        return mask
    return None


def side_cells(mask: np.ndarray, side: str) -> list[tuple[int, int]]:
    h, w = mask.shape
    if side == "top":
        return [(0, c) for c in range(w) if mask[0, c]]
    if side == "bottom":
        return [(h - 1, c) for c in range(w) if mask[h - 1, c]]
    if side == "left":
        return [(r, 0) for r in range(h) if mask[r, 0]]
    return [(r, w - 1) for r in range(h) if mask[r, w - 1]]


def frame_marker_pos(top: int, left: int, fh: int, fw: int, side: str, offset: int) -> tuple[int, int]:
    if side == "top":
        return top - 1, left + offset
    if side == "bottom":
        return top + fh, left + offset
    if side == "left":
        return top + offset, left - 1
    return top + offset, left + fw


def fragment_marker_pos(top: int, left: int, h: int, w: int, side: str, anchor: tuple[int, int]) -> tuple[int, int]:
    ar, ac = anchor
    if side == "top":
        return top - 1, left + ac
    if side == "bottom":
        return top + h, left + ac
    if side == "left":
        return top + ar, left - 1
    return top + ar, left + w


def reserve_rectangle_with_marker(
    top: int,
    left: int,
    h: int,
    w: int,
    marker: tuple[int, int],
) -> set[tuple[int, int]]:
    cells = {(r, c) for r in range(top, top + h) for c in range(left, left + w)}
    cells.add(marker)
    return cells


def dilate_cells(cells: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def can_place(reserved: set[tuple[int, int]], blocked: set[tuple[int, int]]) -> bool:
    return all(cell not in blocked for cell in reserved)


def choose_frame_offset(length: int, rng: random.Random) -> int:
    if length > 2:
        return rng.randint(1, length - 2)
    return rng.randint(0, length - 1)


def place_frame(
    obj: dict,
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
    rng: random.Random,
) -> bool:
    fh = obj["cavity_h"] + 2
    fw = obj["cavity_w"] + 2
    side = obj["frame_side"]
    r_min = 1 + (1 if side == "top" else 0)
    c_min = 1 + (1 if side == "left" else 0)
    r_max = h - fh - 1 - (1 if side == "bottom" else 0)
    c_max = w - fw - 1 - (1 if side == "right" else 0)
    if r_min > r_max or c_min > c_max:
        return False

    positions = [(r, c) for r in range(r_min, r_max + 1) for c in range(c_min, c_max + 1)]
    rng.shuffle(positions)
    marker_offset = choose_frame_offset(fw if side in ("top", "bottom") else fh, rng)
    for top, left in positions:
        marker = frame_marker_pos(top, left, fh, fw, side, marker_offset)
        reserved = reserve_rectangle_with_marker(top, left, fh, fw, marker)
        if not can_place(reserved, blocked):
            continue
        obj["frame_top"] = top
        obj["frame_left"] = left
        obj["frame_marker"] = marker
        blocked.update(dilate_cells(reserved, h, w, pad=1))
        return True
    return False


def place_fragment(
    obj: dict,
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
    rng: random.Random,
) -> bool:
    mask = obj["input_mask"]
    frag_h, frag_w = map(int, mask.shape)
    side = obj["fragment_side"]
    anchor = rng.choice(side_cells(mask, side))
    r_min = 1 + (1 if side == "top" else 0)
    c_min = 1 + (1 if side == "left" else 0)
    r_max = h - frag_h - 1 - (1 if side == "bottom" else 0)
    c_max = w - frag_w - 1 - (1 if side == "right" else 0)
    if r_min > r_max or c_min > c_max:
        return False

    positions = [(r, c) for r in range(r_min, r_max + 1) for c in range(c_min, c_max + 1)]
    rng.shuffle(positions)
    for top, left in positions:
        marker = fragment_marker_pos(top, left, frag_h, frag_w, side, anchor)
        reserved = reserve_rectangle_with_marker(top, left, frag_h, frag_w, marker)
        if not can_place(reserved, blocked):
            continue
        obj["fragment_top"] = top
        obj["fragment_left"] = left
        obj["fragment_marker"] = marker
        obj["fragment_anchor"] = anchor
        blocked.update(dilate_cells(reserved, h, w, pad=1))
        return True
    return False


def draw_frame(grid: np.ndarray, top: int, left: int, cavity_h: int, cavity_w: int, color: int) -> None:
    bottom = top + cavity_h + 1
    right = left + cavity_w + 1
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color


def build_expected_output(h: int, w: int, bg: int, objects: list[dict]) -> np.ndarray:
    out = np.full((h, w), bg, dtype=int)
    for obj in objects:
        draw_frame(out, obj["frame_top"], obj["frame_left"], obj["cavity_h"], obj["cavity_w"], obj["color"])
        patch = out[
            obj["frame_top"] + 1 : obj["frame_top"] + 1 + obj["cavity_h"],
            obj["frame_left"] + 1 : obj["frame_left"] + 1 + obj["cavity_w"],
        ]
        patch[obj["solved_mask"]] = obj["color"]
    return out


def build_input_grid(h: int, w: int, bg: int, objects: list[dict]) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for obj in objects:
        draw_frame(grid, obj["frame_top"], obj["frame_left"], obj["cavity_h"], obj["cavity_w"], obj["color"])
        fr, fc = obj["frame_marker"]
        grid[fr, fc] = MARKER_COLOR

        frag_top = obj["fragment_top"]
        frag_left = obj["fragment_left"]
        frag_h, frag_w = obj["input_mask"].shape
        patch = grid[frag_top : frag_top + frag_h, frag_left : frag_left + frag_w]
        patch[obj["input_mask"]] = obj["color"]
        mr, mc = obj["fragment_marker"]
        grid[mr, mc] = MARKER_COLOR
    return grid


def sample_dimensions(count: int, rng: random.Random) -> tuple[int, int]:
    base_h = rng.randint(18, 28)
    base_w = rng.randint(18, 28)
    if count >= 4:
        base_h = max(base_h, 24)
        base_w = max(base_w, 24)
    elif count == 3:
        base_h = max(base_h, 21)
        base_w = max(base_w, 21)
    return base_h, base_w


def generate_object(color: int, rng: random.Random) -> dict | None:
    cavity_h = rng.randint(2, 5)
    cavity_w = rng.randint(2, 5)
    solved_mask = generate_irregular_mask(cavity_h, cavity_w, rng)
    if solved_mask is None:
        return None
    frame_side = rng.choice(SIDES)
    k = rng.randrange(4)
    input_mask = np.rot90(solved_mask, k)
    return {
        "color": color,
        "cavity_h": cavity_h,
        "cavity_w": cavity_w,
        "solved_mask": solved_mask,
        "frame_side": frame_side,
        "fragment_side": rotate_side(frame_side, k),
        "input_mask": input_mask,
    }


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random()
        bg_choices = [0, 1, 3, 4, 5, 6, 7, 8, 9]
        bg = rng.choice(bg_choices)
        colors = [c for c in range(10) if c not in {bg, MARKER_COLOR}]
        rng.shuffle(colors)
        count = rng.randint(2, 4)
        object_colors = colors[:count]

        objects: list[dict] = []
        ok = True
        for color in object_colors:
            obj = generate_object(color, rng)
            if obj is None:
                ok = False
                break
            objects.append(obj)
        if not ok:
            continue

        h, w = sample_dimensions(count, rng)
        blocked: set[tuple[int, int]] = set()

        frame_order = sorted(objects, key=lambda obj: (obj["cavity_h"] + 2) * (obj["cavity_w"] + 2), reverse=True)
        for obj in frame_order:
            if not place_frame(obj, h, w, blocked, rng):
                ok = False
                break
        if not ok:
            continue

        frag_order = sorted(objects, key=lambda obj: obj["input_mask"].shape[0] * obj["input_mask"].shape[1], reverse=True)
        for obj in frag_order:
            if not place_fragment(obj, h, w, blocked, rng):
                ok = False
                break
        if not ok:
            continue

        inp = build_input_grid(h, w, bg, objects)
        expected = build_expected_output(h, w, bg, objects)
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp

    raise RuntimeError("failed to generate a valid puzzle instance")
