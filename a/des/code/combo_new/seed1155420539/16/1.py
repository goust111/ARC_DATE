import random
from typing import Optional

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5

ROTATION_TURNS = {
    RED: 0,
    GREEN: 1,
    YELLOW: 2,
    BLUE: 3,
}

SPRITE_COLOR_POOL = [BLUE, RED, GREEN, YELLOW, 6, 7, 8, 9]
ROTATION_COLORS = [RED, GREEN, YELLOW, BLUE]
CANVAS_LIMIT = 30
GEN_RETRIES = 300

SHAPE_LIBRARY = [
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 1], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_component(points: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(points)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask


def _rotate_cw(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(mask, -turns)


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for pts in _components(grid == GRAY):
        r0, c0, r1, c1 = _bbox(pts)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 4 or width < 4:
            continue
        perimeter = 2 * (height + width) - 4
        if len(pts) != perimeter:
            continue
        if any(r not in (r0, r1) and c not in (c0, c1) for r, c in pts):
            continue
        if not (
            np.all(grid[r0, c0 : c1 + 1] == GRAY)
            and np.all(grid[r1, c0 : c1 + 1] == GRAY)
            and np.all(grid[r0 : r1 + 1, c0] == GRAY)
            and np.all(grid[r0 : r1 + 1, c1] == GRAY)
        ):
            continue
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "interior": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
            }
        )
    frames.sort(key=lambda frame: (frame["bbox"][0], frame["bbox"][1]))
    return frames


def _extract_outer_sprites(grid: np.ndarray, frames: list[dict]) -> dict[int, np.ndarray]:
    inside = np.zeros_like(grid, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["interior"]
        inside[r0 : r1 + 1, c0 : c1 + 1] = True
    mask = (grid != BG) & (grid != GRAY) & ~inside
    sprites: dict[int, np.ndarray] = {}
    ambiguous: set[int] = set()
    for pts in _components(mask):
        colors = {int(grid[r, c]) for r, c in pts}
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        if color in sprites:
            ambiguous.add(color)
            continue
        sprites[color] = _normalize_component(pts)
    for color in ambiguous:
        sprites.pop(color, None)
    return sprites


def _ordered_anchors(grid: np.ndarray, frame: dict) -> Optional[tuple[int, int]]:
    r0, c0, r1, c1 = frame["interior"]
    pts = [(int(r), int(c)) for r, c in np.argwhere(grid[r0 : r1 + 1, c0 : c1 + 1] != BG)]
    abs_pts = [(r + r0, c + c0) for r, c in pts]
    if len(abs_pts) != 2:
        return None
    (ar, ac), (br, bc) = abs_pts
    if abs(ar - br) + abs(ac - bc) != 1:
        return None
    if ar == br:
        first, second = sorted(abs_pts, key=lambda p: (p[1], p[0]))
    elif ac == bc:
        first, second = sorted(abs_pts, key=lambda p: (p[0], p[1]))
    else:
        return None
    a = int(grid[first])
    b = int(grid[second])
    if a == b:
        return None
    return a, b


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gray rectangular room with the matching outside sprite chosen by an ordered anchor pair.

    Input structure:
    - The background is black. Several one-pixel-thick gray rectangular frames enclose black rooms.
    - Each room interior contains exactly two adjacent colored anchor pixels with different colors.
    - Outside the rooms, every remaining non-black non-gray 4-connected monochrome component is a small sprite, and sprite colors are unique.

    Rule:
    - Read each anchor pair in its visible direction: left-to-right for a horizontal pair, top-to-bottom for a vertical pair. The first color A selects the outside sprite of the same color.
    - The second color B encodes a clockwise rotation: red = 0 degrees, green = 90 degrees, yellow = 180 degrees, blue = 270 degrees.
    - Rotate the matched sprite, recolor the placed copy to A, and center it inside the room interior.
    - If the rotated sprite does not fit fully inside the room, leave that room unchanged.

    Key insight:
    - The anchor pair is an ordered two-channel instruction: position order separates the sprite-selection color from the rotation-command color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _find_frames(g)
    sprites = _extract_outer_sprites(g, frames)

    for frame in frames:
        anchor_info = _ordered_anchors(g, frame)
        if anchor_info is None:
            continue
        sprite_color, rotation_color = anchor_info
        if sprite_color not in sprites or rotation_color not in ROTATION_TURNS:
            continue
        sprite = sprites[sprite_color]
        placed = _rotate_cw(sprite, ROTATION_TURNS[rotation_color])
        ih0, iw0, ih1, iw1 = frame["interior"]
        room_h = ih1 - ih0 + 1
        room_w = iw1 - iw0 + 1
        sh, sw = placed.shape
        if sh > room_h or sw > room_w:
            continue
        off_r = ih0 + (room_h - sh) // 2
        off_c = iw0 + (room_w - sw) // 2
        out[ih0 : ih1 + 1, iw0 : iw1 + 1] = BG
        out[off_r : off_r + sh, off_c : off_c + sw][placed] = sprite_color
    return out


def _random_shape(rng: random.Random) -> np.ndarray:
    shape = SHAPE_LIBRARY[rng.randrange(len(SHAPE_LIBRARY))].copy()
    if rng.random() < 0.5:
        shape = np.fliplr(shape)
    shape = _rotate_cw(shape, rng.randrange(4))
    return shape


def _available_anchor_pairs(mask: np.ndarray) -> dict[str, list[tuple[tuple[int, int], tuple[int, int]]]]:
    pairs = {"h": [], "v": []}
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if c + 1 < w and mask[r, c + 1]:
                pairs["h"].append(((r, c), (r, c + 1)))
            if r + 1 < h and mask[r + 1, c]:
                pairs["v"].append(((r, c), (r + 1, c)))
    return pairs


def _assign_sprite_colors(commands: list[int], rng: random.Random) -> Optional[list[int]]:
    for _ in range(50):
        remaining = SPRITE_COLOR_POOL[:]
        rng.shuffle(remaining)
        chosen: list[int] = []
        ok = True
        for command in commands:
            options = [color for color in remaining if color != command]
            if not options:
                ok = False
                break
            color = rng.choice(options)
            remaining.remove(color)
            chosen.append(color)
        if ok:
            return chosen
    return None


def _choose_room_size(
    sprite_shape: np.ndarray,
    should_fit: bool,
    rng: random.Random,
) -> Optional[tuple[int, int]]:
    sh, sw = sprite_shape.shape
    if should_fit:
        room_h = sh + rng.randint(0, 3)
        room_w = sw + rng.randint(0, 3)
        return room_h, room_w
    fail_axes = []
    if sh > 2:
        fail_axes.append("h")
    if sw > 2:
        fail_axes.append("w")
    if not fail_axes:
        return None
    axis = rng.choice(fail_axes)
    room_h = sh + rng.randint(0, 3)
    room_w = sw + rng.randint(0, 3)
    if axis == "h":
        room_h = rng.randint(2, sh - 1)
    else:
        room_w = rng.randint(2, sw - 1)
    return room_h, room_w


def _centered_cells(
    room_h: int,
    room_w: int,
    sprite_shape: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]]]:
    sh, sw = sprite_shape.shape
    off_r = (room_h - sh) // 2
    off_c = (room_w - sw) // 2
    cells = [(off_r + r, off_c + c) for r, c in np.argwhere(sprite_shape)]
    return off_r, off_c, cells


def _pick_anchor_positions(
    room_h: int,
    room_w: int,
    final_shape: np.ndarray,
    should_fit: bool,
    rng: random.Random,
) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
    if should_fit:
        pair_options = _available_anchor_pairs(final_shape)
        usable = [key for key, pairs in pair_options.items() if pairs]
        if not usable:
            return None
        orient = rng.choice(usable)
        pair = rng.choice(pair_options[orient])
        off_r, off_c, _ = _centered_cells(room_h, room_w, final_shape)
        return (
            (off_r + pair[0][0], off_c + pair[0][1]),
            (off_r + pair[1][0], off_c + pair[1][1]),
        )
    if rng.random() < 0.5:
        r = rng.randrange(room_h)
        c = rng.randrange(room_w - 1)
        return (r, c), (r, c + 1)
    r = rng.randrange(room_h - 1)
    c = rng.randrange(room_w)
    return (r, c), (r + 1, c)


def _reserve(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    hh, ww = occupied.shape
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if r0 < 0 or c0 < 0 or r1 >= hh or c1 >= ww:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r1 + pad + 1)
    cc1 = min(ww, c1 + pad + 1)
    if occupied[rr0:rr1, cc0:cc1].any():
        return False
    occupied[rr0:rr1, cc0:cc1] = True
    return True


def _place_rect(occupied: np.ndarray, h: int, w: int, rng: random.Random) -> Optional[tuple[int, int]]:
    hh, ww = occupied.shape
    order_r = list(range(1, hh - h))
    order_c = list(range(1, ww - w))
    rng.shuffle(order_r)
    rng.shuffle(order_c)
    for r0 in order_r:
        for c0 in order_c:
            snapshot = occupied.copy()
            if _reserve(snapshot, r0, c0, h, w, pad=1):
                occupied[:, :] = snapshot
                return r0, c0
    return None


def _paint_shape(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    grid[top : top + h, left : left + w][mask] = color


def _crop_with_border(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    pts = np.argwhere(grid != BG)
    if pts.size == 0:
        return grid.copy()
    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    r1 = int(pts[:, 0].max())
    c1 = int(pts[:, 1].max())
    top_pad = min(r0, rng.randint(1, 2))
    left_pad = min(c0, rng.randint(1, 2))
    bottom_pad = min(grid.shape[0] - 1 - r1, rng.randint(1, 2))
    right_pad = min(grid.shape[1] - 1 - c1, rng.randint(1, 2))
    return grid[r0 - top_pad : r1 + bottom_pad + 1, c0 - left_pad : c1 + right_pad + 1].copy()


def _build_candidate(rng: random.Random) -> Optional[np.ndarray]:
    grid = np.zeros((CANVAS_LIMIT, CANVAS_LIMIT), dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)
    room_count = rng.randint(1, 3)
    commands = [rng.choice(ROTATION_COLORS) for _ in range(room_count)]
    sprite_colors = _assign_sprite_colors(commands, rng)
    if sprite_colors is None:
        return None

    specs: list[dict] = []
    for idx in range(room_count):
        sprite_shape = _random_shape(rng)
        should_fit = bool(rng.randrange(2))
        final_shape = _rotate_cw(sprite_shape, ROTATION_TURNS[commands[idx]])
        room_size = _choose_room_size(final_shape, should_fit, rng)
        if room_size is None:
            return None
        room_h, room_w = room_size
        if room_h > 9 or room_w > 9:
            return None
        anchors = _pick_anchor_positions(room_h, room_w, final_shape, should_fit, rng)
        if anchors is None:
            return None
        specs.append(
            {
                "sprite_color": sprite_colors[idx],
                "command": commands[idx],
                "sprite_shape": sprite_shape,
                "final_shape": final_shape,
                "room_h": room_h,
                "room_w": room_w,
                "anchors": anchors,
                "should_fit": should_fit,
            }
        )

    for spec in sorted(specs, key=lambda item: (item["room_h"] + 2) * (item["room_w"] + 2), reverse=True):
        placed = _place_rect(occupied, spec["room_h"] + 2, spec["room_w"] + 2, rng)
        if placed is None:
            return None
        top, left = placed
        spec["frame_top"] = top
        spec["frame_left"] = left
        grid[top, left : left + spec["room_w"] + 2] = GRAY
        grid[top + spec["room_h"] + 1, left : left + spec["room_w"] + 2] = GRAY
        grid[top : top + spec["room_h"] + 2, left] = GRAY
        grid[top : top + spec["room_h"] + 2, left + spec["room_w"] + 1] = GRAY
        (a_r, a_c), (b_r, b_c) = spec["anchors"]
        grid[top + 1 + a_r, left + 1 + a_c] = spec["sprite_color"]
        grid[top + 1 + b_r, left + 1 + b_c] = spec["command"]

    for spec in sorted(specs, key=lambda item: item["sprite_shape"].shape[0] * item["sprite_shape"].shape[1], reverse=True):
        sh, sw = spec["sprite_shape"].shape
        placed = _place_rect(occupied, sh, sw, rng)
        if placed is None:
            return None
        top, left = placed
        _paint_shape(grid, top, left, spec["sprite_shape"], spec["sprite_color"])

    cropped = _crop_with_border(grid, rng)
    if max(cropped.shape) > CANVAS_LIMIT:
        return None

    frames = _find_frames(cropped)
    sprites = _extract_outer_sprites(cropped, frames)
    if len(frames) != room_count or len(sprites) != room_count:
        return None
    for spec in specs:
        if spec["sprite_color"] not in sprites:
            return None
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _build_candidate(rng)
        if grid is not None:
            _ = main(grid)
            return grid
    raise RuntimeError("failed to generate a valid input")
