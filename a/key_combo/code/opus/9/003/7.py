from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox, bbox_h, bbox_w, crop_bbox, find_components, get_bg, in_bounds, paste


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ORDERED_DIRS = ("up", "down", "left", "right")


def _arm_length(mask: np.ndarray, r: int, c: int, dr: int, dc: int) -> int:
    length = 0
    rr = r + dr
    cc = c + dc
    while in_bounds(mask.shape, rr, cc) and mask[rr, cc]:
        length += 1
        rr += dr
        cc += dc
    return length


def _find_cross(grid: np.ndarray, color: int) -> dict | None:
    mask = grid == color
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None

    junctions = []
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        degree = sum(
            in_bounds(mask.shape, r + dr, c + dc) and mask[r + dr, c + dc]
            for dr, dc in DIRS.values()
        )
        if degree == 4:
            junctions.append((r, c))
    if len(junctions) != 1:
        return None

    jr, jc = junctions[0]
    if any(int(r) != jr and int(c) != jc for r, c in coords):
        return None

    lengths = {name: _arm_length(mask, jr, jc, dr, dc) for name, (dr, dc) in DIRS.items()}
    if any(lengths[name] <= 0 for name in ORDERED_DIRS):
        return None
    longest = max(lengths.values())
    longest_dirs = [name for name, length in lengths.items() if length == longest]
    if len(longest_dirs) != 1:
        return None
    if any(longest - lengths[name] < 2 for name in ORDERED_DIRS if name != longest_dirs[0]):
        return None
    if int(mask.sum()) != 1 + sum(lengths.values()):
        return None

    return {
        "center": (jr, jc),
        "direction": longest_dirs[0],
        "length": int(longest),
        "mask": mask,
        "coords": coords,
    }


def _locate_roles(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = sorted(set(map(int, g.ravel())) - {bg})
    assert Color.RED in colors, "expected red anchor pixels"

    marker_candidates = []
    for color in colors:
        if color == Color.RED:
            continue
        cross = _find_cross(g, color)
        if cross is not None:
            marker_candidates.append((color, cross))
    assert len(marker_candidates) == 1, "expected exactly one asymmetric cross marker"
    marker_color, cross = marker_candidates[0]

    sprite_colors = [color for color in colors if color not in (Color.RED, marker_color)]
    assert len(sprite_colors) == 1, "expected exactly one sprite color"
    sprite_color = sprite_colors[0]

    comps = find_components(g, bg=bg, colors=[sprite_color], connectivity=4, monochromatic=True)
    assert len(comps) == 1, "expected a single sprite component"
    sprite = comps[0]
    sprite_bbox = sprite["bbox"]
    sprite_patch = crop_bbox((g == sprite_color).astype(int), sprite_bbox).astype(bool)

    return {
        "grid": g,
        "bg": int(bg),
        "marker_color": int(marker_color),
        "cross": cross,
        "sprite_color": int(sprite_color),
        "sprite_bbox": sprite_bbox,
        "sprite_patch": sprite_patch,
    }


def _frame_bbox(sprite_bbox: tuple[int, int, int, int], direction: str, depth: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = sprite_bbox
    depth = int(depth)
    if direction == "up":
        return r0 - depth - 1, c0, r0 - 2, c1
    if direction == "down":
        return r1 + 2, c0, r1 + depth + 1, c1
    if direction == "left":
        return r0, c0 - depth - 1, r1, c0 - 2
    if direction == "right":
        return r0, c1 + 2, r1, c1 + depth + 1
    raise ValueError(f"unknown direction: {direction}")


def _bbox_in_bounds(shape: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return 0 <= r0 <= r1 < shape[0] and 0 <= c0 <= c1 < shape[1]


def _draw_border(out: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color


def _strict_interior_coords(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _border_coords(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    coords = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    return sorted(set(coords))


def _thumbnail(sprite_patch: np.ndarray, color: int) -> np.ndarray:
    h, w = sprite_patch.shape
    thumb = np.zeros((3, 3), dtype=int)
    for rr, cc in np.argwhere(sprite_patch):
        br = (3 * int(rr)) // h
        bc = (3 * int(cc)) // w
        thumb[br, bc] = color
    return thumb


def _transform_from_roles(
    roles: dict,
    *,
    direction: str | None = None,
    depth: int | None = None,
    anchor_mode: str = "strict",
) -> np.ndarray:
    g = roles["grid"]
    bg = roles["bg"]
    sprite_color = roles["sprite_color"]
    cross = roles["cross"]
    direction = direction or cross["direction"]
    depth = int(depth if depth is not None else cross["length"])

    frame_box = _frame_bbox(roles["sprite_bbox"], direction, depth)
    assert _bbox_in_bounds(g.shape, frame_box), "frame must stay on the grid"

    out = g.copy()
    _draw_border(out, frame_box, sprite_color)

    thumb = _thumbnail(roles["sprite_patch"], sprite_color)
    if anchor_mode == "strict":
        anchor_coords = _strict_interior_coords(frame_box)
    elif anchor_mode == "inclusive":
        r0, c0, r1, c1 = frame_box
        anchor_coords = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    else:
        raise ValueError(f"unknown anchor mode: {anchor_mode}")

    anchors = [(r, c) for r, c in anchor_coords if int(g[r, c]) == Color.RED]
    for r, c in anchors:
        paste(out, thumb, r - 1, c - 1, clip=True)

    out[cross["mask"]] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a side frame from the cross-arm code and stamp a 3x3 summary of the sprite at interior anchors.

    Input structure:
    - The grid has a black background, one 4-connected irregular sprite in color A, one one-pixel-thick cross marker in color B, and one or two red anchor pixels.
    - The cross has a unique junction and four cardinal arms, with exactly one arm at least two pixels longer than every other arm.
    - The sprite and cross are separated, and the anchors lie in the region that the decoded side frame will cover.

    Rule:
    - Read the marker's uniquely longest arm to get both a direction D and a length L.
    - Take the sprite's bounding box, then build a single-pixel rectangular frame in color A on side D of that box, leaving a one-cell gap from the sprite; the frame depth is L and its side parallel to the sprite matches the sprite bbox span on that side.
    - Inside the frame's strict interior, find each red anchor and stamp a 3x3 thumbnail of the sprite there, where each thumbnail cell is filled iff the corresponding third of the sprite bbox contains any sprite pixel.
    - The stamp overwrites existing pixels, then the cross marker is erased to background.

    Key insight:
    - The asymmetric cross does not mark a location to copy toward; its longest arm simultaneously chooses which side of the sprite receives the frame and how deep that frame extends, while the sprite itself is reduced into the stamp pattern.
    """
    roles = _locate_roles(grid)
    return _transform_from_roles(roles)


def _grow_connected_shape(rng: random.Random, h: int, w: int, area: int) -> np.ndarray:
    cells = {(rng.randrange(h), rng.randrange(w))}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIRS.values():
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        cells.add(frontier[rng.randrange(len(frontier))])
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _sprite_candidates() -> list[tuple[int, int, int]]:
    out = []
    for h in range(3, 6):
        for w in range(3, 6):
            for area in range(5, min(8, h * w) + 1):
                out.append((h, w, area))
    return out


SPRITE_CANDIDATES = _sprite_candidates()


def _random_sprite_patch(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h, w, area = SPRITE_CANDIDATES[rng.randrange(len(SPRITE_CANDIDATES))]
        mask = _grow_connected_shape(rng, h, w, area)
        b = bbox(mask)
        if b != (0, 0, h - 1, w - 1):
            continue
        if int(mask.sum()) == h * w:
            continue
        thumb = _thumbnail(mask, 1)
        filled = int(np.count_nonzero(thumb))
        if not (3 <= filled <= 7):
            continue
        if _find_cross(mask.astype(int), 1) is not None:
            continue
        return mask
    raise RuntimeError("failed to generate a suitable sprite patch")


def _footprint(anchor: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = anchor
    return {(rr, cc) for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)}


def _pick_anchor_plan(
    rng: random.Random, frame_box: tuple[int, int, int, int]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    interior = _strict_interior_coords(frame_box)
    assert interior, "frame interior must be non-empty"
    border = _border_coords(frame_box)

    modes = ["one_interior", "two_interior", "interior_plus_border"]
    rng.shuffle(modes)

    for mode in modes:
        first = rng.choice(interior)
        if mode == "one_interior":
            return [first], []

        if mode == "two_interior":
            second_opts = [
                cell
                for cell in interior
                if cell != first and (abs(cell[0] - first[0]) > 2 or abs(cell[1] - first[1]) > 2)
            ]
            if second_opts:
                return [first, rng.choice(second_opts)], []

        if mode == "interior_plus_border":
            blocked = _footprint(first)
            border_opts = [cell for cell in border if cell not in blocked]
            if border_opts:
                return [first], [rng.choice(border_opts)]

    return [rng.choice(interior)], []


def _place_cross(
    grid: np.ndarray,
    rng: random.Random,
    color: int,
    direction: str,
    length: int,
    forbidden: set[tuple[int, int]],
) -> bool:
    others = {}
    for name in ORDERED_DIRS:
        if name == direction:
            continue
        others[name] = rng.randint(1, max(1, length - 2))

    top = length if direction == "up" else others.get("up", 0)
    bottom = length if direction == "down" else others.get("down", 0)
    left = length if direction == "left" else others.get("left", 0)
    right = length if direction == "right" else others.get("right", 0)

    H, W = grid.shape
    candidates = []
    for r in range(top, H - bottom):
        for c in range(left, W - right):
            pixels = {(r, c)}
            pixels |= {(r - k, c) for k in range(1, top + 1)}
            pixels |= {(r + k, c) for k in range(1, bottom + 1)}
            pixels |= {(r, c - k) for k in range(1, left + 1)}
            pixels |= {(r, c + k) for k in range(1, right + 1)}
            if any(p in forbidden for p in pixels):
                continue
            candidates.append((r, c, pixels))
    if not candidates:
        return False

    r, c, pixels = candidates[rng.randrange(len(candidates))]
    for rr, cc in pixels:
        grid[rr, cc] = color
    return True


def _forbidden_cross_cells(
    shape: tuple[int, int],
    sprite_box: tuple[int, int, int, int],
    frame_box: tuple[int, int, int, int],
    anchors: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    H, W = shape
    forbidden = set()
    for box, pad in ((sprite_box, 2), (frame_box, 1)):
        r0, c0, r1, c1 = box
        for r in range(max(0, r0 - pad), min(H, r1 + pad + 1)):
            for c in range(max(0, c0 - pad), min(W, c1 + pad + 1)):
                forbidden.add((r, c))
    for r, c in anchors:
        for rr in range(max(0, r - 1), min(H, r + 2)):
            for cc in range(max(0, c - 1), min(W, c + 2)):
                forbidden.add((rr, cc))
    return forbidden


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        roles = _locate_roles(grid)
        out = _transform_from_roles(roles)
    except AssertionError:
        return False

    direction = roles["cross"]["direction"]
    length = roles["cross"]["length"]
    thumb = _thumbnail(roles["sprite_patch"], roles["sprite_color"])
    if not (3 <= int(np.count_nonzero(thumb)) <= 7):
        return False

    frame_box = _frame_bbox(roles["sprite_bbox"], direction, length)
    anchors = [cell for cell in _strict_interior_coords(frame_box) if int(grid[cell]) == Color.RED]
    if not anchors:
        return False
    if np.array_equal(grid, out):
        return False

    alt_length = length + 1 if length < 6 else length - 1
    try:
        alt_out = _transform_from_roles(roles, depth=alt_length)
    except AssertionError:
        return False
    if np.array_equal(out, alt_out):
        return False

    border_anchors = [cell for cell in _border_coords(frame_box) if int(grid[cell]) == Color.RED]
    if border_anchors:
        inclusive_out = _transform_from_roles(roles, anchor_mode="inclusive")
        if np.array_equal(out, inclusive_out):
            return False

    for alt_direction in ORDERED_DIRS:
        if alt_direction == direction:
            continue
        try:
            alt_side = _transform_from_roles(roles, direction=alt_direction, depth=length)
        except AssertionError:
            continue
        if np.array_equal(out, alt_side):
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    H = rng.randint(18, 25)
    W = rng.randint(18, 25)
    grid = np.zeros((H, W), dtype=int)

    sprite_color = rng.choice([Color.BLUE, Color.GREEN, Color.YELLOW, Color.PINK, Color.TEAL, Color.MAROON])
    marker_choices = [Color.ORANGE, Color.GREY, Color.YELLOW, Color.PINK, Color.TEAL, Color.MAROON]
    marker_choices = [c for c in marker_choices if c not in (sprite_color, Color.RED)]
    marker_color = rng.choice(marker_choices)

    direction = rng.choice(list(ORDERED_DIRS))
    length = rng.randint(3, 6)
    alt_length = length + 1 if length < 6 else length - 1
    need_depth = max(length, alt_length)

    sprite_patch = _random_sprite_patch(rng)
    h, w = sprite_patch.shape

    min_r0 = 1
    max_r0 = H - h - 1
    min_c0 = 1
    max_c0 = W - w - 1
    if direction in ("up", "down"):
        min_c0 = 2
        max_c0 = W - w - 2
    else:
        min_r0 = 2
        max_r0 = H - h - 2

    if direction == "up":
        min_r0 = max(min_r0, need_depth + 1)
    elif direction == "down":
        max_r0 = min(max_r0, H - h - need_depth - 1)
    elif direction == "left":
        min_c0 = max(min_c0, need_depth + 1)
    else:
        max_c0 = min(max_c0, W - w - need_depth - 1)

    if min_r0 > max_r0 or min_c0 > max_c0:
        return None
    r0 = rng.randint(min_r0, max_r0)
    c0 = rng.randint(min_c0, max_c0)

    sprite_cells = np.argwhere(sprite_patch)
    for rr, cc in sprite_cells:
        grid[r0 + int(rr), c0 + int(cc)] = sprite_color
    sprite_box = (r0, c0, r0 + h - 1, c0 + w - 1)
    frame_box = _frame_bbox(sprite_box, direction, length)
    alt_box = _frame_bbox(sprite_box, direction, alt_length)
    if not _bbox_in_bounds(grid.shape, frame_box) or not _bbox_in_bounds(grid.shape, alt_box):
        return None

    interior_anchors, border_anchors = _pick_anchor_plan(rng, frame_box)
    all_anchors = interior_anchors + border_anchors
    for rr, cc in all_anchors:
        if grid[rr, cc] != Color.BLACK:
            return None
        grid[rr, cc] = Color.RED

    forbidden = _forbidden_cross_cells(grid.shape, sprite_box, frame_box, all_anchors)
    if not _place_cross(grid, rng, marker_color, direction, length, forbidden):
        return None

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
