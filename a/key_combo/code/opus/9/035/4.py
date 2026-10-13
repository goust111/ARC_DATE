from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MARKER_COLORS = (BLUE, RED, YELLOW)
QUADRANTS = ("TL", "TR", "BR", "BL")
ROTATION_K = {"TL": 0, "TR": -1, "BR": 2, "BL": 1}

GRID_SIDES = (17, 19, 21, 23)
GEN_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _same_mask(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b)


def _rotation_distinct(mask: np.ndarray) -> bool:
    return all(not _same_mask(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _rotate_for_quadrant(sprite: np.ndarray, quadrant: str) -> np.ndarray:
    return np.rot90(sprite, ROTATION_K[quadrant])


def _quadrant(shape: tuple[int, int], r: int, c: int) -> str:
    h, w = map(int, shape)
    mid_r = h // 2
    mid_c = w // 2
    if h % 2 == 1:
        assert r != mid_r, "anchor may not lie on the horizontal center line"
    if w % 2 == 1:
        assert c != mid_c, "anchor may not lie on the vertical center line"
    top = r < mid_r
    left = c < mid_c
    if top and left:
        return "TL"
    if top and not left:
        return "TR"
    if not top and not left:
        return "BR"
    return "BL"


def _unique_least_marker_color(grid: np.ndarray) -> int:
    counts = {color: int(np.count_nonzero(grid == color)) for color in MARKER_COLORS}
    assert all(counts[color] > 0 for color in MARKER_COLORS), "all three marker colors must be present"
    least = min(counts.values())
    colors = [color for color, count in counts.items() if count == least]
    assert len(colors) == 1, "marker color with the fewest pixels must be unique"
    return colors[0]


def _stamp_transparent(
    out: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    protected: np.ndarray,
) -> None:
    ph, pw = patch.shape
    for pr in range(ph):
        for pc in range(pw):
            value = int(patch[pr, pc])
            if value == BLACK:
                continue
            rr = r0 + pr
            cc = c0 + pc
            if not protected[rr, cc]:
                out[rr, cc] = value


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the green sprite, pick the uniquely rare marker color, and stamp rotated copies at its quadrant-coded anchors.

    Input structure:
    - The grid has black background, exactly one 4-connected green sprite near the upper-left, and scattered single-pixel markers in blue, red, and yellow.
    - The sprite is defined by all green pixels; its minimal bounding box may contain black hole cells that remain part of the cropped template.
    - One marker color occurs strictly fewer times than the other two, and each pixel of that rare color is an anchor located away from the sprite.

    Rule:
    - Crop the green sprite to its minimal bounding box, preserving black holes inside that box.
    - Count blue, red, and yellow marker pixels; the unique least frequent color is the anchor color.
    - Erase all non-anchor marker pixels to black.
    - For each anchor, rotate the cropped sprite by the anchor's grid quadrant relative to the whole-grid center: top-left 0 degrees, top-right 90 degrees clockwise, bottom-right 180 degrees, bottom-left 270 degrees clockwise.
    - Paste each rotated sprite with its top-left corner at the anchor, treating black template cells as transparent and never overwriting the original green sprite pixels.

    Key insight:
    - Three independent cues must be combined: green pixels define the stamp template, marker-frequency selects which scattered pixels are anchors, and each anchor's global quadrant determines which rotation of that template is pasted there.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    green_mask = g == GREEN
    assert np.any(green_mask), "expected a green sprite"
    assert len(_components4(green_mask)) == 1, "expected exactly one green 4-connected sprite"

    box = _bbox(green_mask)
    assert box is not None
    sprite = _crop_bbox(g, box)
    selected_color = _unique_least_marker_color(g)

    out = g.copy()
    for color in MARKER_COLORS:
        if color != selected_color:
            out[out == color] = BLACK

    anchors = sorted((int(r), int(c)) for r, c in np.argwhere(g == selected_color))
    h, w = g.shape
    for r, c in anchors:
        assert not green_mask[r, c], "anchor markers may not coincide with the source sprite"
        quadrant = _quadrant((h, w), r, c)
        patch = _rotate_for_quadrant(sprite, quadrant)
        ph, pw = patch.shape
        assert r + ph <= h and c + pw <= w, "rotated stamp must fit inside the grid"
        _stamp_transparent(out, patch, r, c, green_mask)

    return out


def _grown_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    frontier = set()
    frontier.update(
        (start[0] + dr, start[1] + dc)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        if 0 <= start[0] + dr < height and 0 <= start[1] + dc < width
    )
    while int(mask.sum()) < area:
        if not frontier:
            break
        rr, cc = random.choice(tuple(frontier))
        frontier.remove((rr, cc))
        if mask[rr, cc]:
            continue
        mask[rr, cc] = True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.add((nr, nc))
    return mask


def _make_sprite_mask() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        base_h = random.randint(4, 6)
        base_w = random.randint(4, 6)
        area = random.randint(8, min(14, base_h * base_w))
        mask = _grown_mask(base_h, base_w, area)
        box = _bbox(mask)
        if box is None:
            continue
        mask = _crop_bbox(mask.astype(int), box).astype(bool)
        h, w = mask.shape
        corners = [bool(mask[0, 0]), bool(mask[0, -1]), bool(mask[-1, 0]), bool(mask[-1, -1])]
        if h < 3 or w < 3:
            continue
        if not (8 <= int(mask.sum()) <= 14):
            continue
        if len(_components4(mask)) != 1:
            continue
        if not _rotation_distinct(mask):
            continue
        if len(set(corners)) < 2:
            continue
        if max(np.count_nonzero(mask, axis=0)) < 2 or max(np.count_nonzero(mask, axis=1)) < 2:
            continue
        return mask
    raise RuntimeError("failed to generate a non-degenerate sprite")


def _place_sprite(grid: np.ndarray, sprite_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = grid.copy()
    h, w = sprite_mask.shape
    max_r0 = min(3, out.shape[0] - h)
    max_c0 = min(3, out.shape[1] - w)
    r0 = random.randint(0, max_r0)
    c0 = random.randint(0, max_c0)
    out[r0 : r0 + h, c0 : c0 + w][sprite_mask] = GREEN
    placed = np.zeros_like(out, dtype=bool)
    placed[r0 : r0 + h, c0 : c0 + w][sprite_mask] = True
    return out, placed


def _quadrant_ranges(length: int, top_or_left: bool, stamp_len: int) -> range:
    mid = length // 2
    if top_or_left:
        return range(0, max(0, min(mid, length - stamp_len + 1)))
    start = mid + 1
    return range(start, max(start, length - stamp_len + 1))


def _stamp_hits_protected(patch: np.ndarray, protected: np.ndarray, r0: int, c0: int) -> bool:
    for pr, pc in np.argwhere(patch != BLACK):
        if protected[r0 + int(pr), c0 + int(pc)]:
            return True
    return False


def _candidate_anchors(
    shape: tuple[int, int],
    sprite: np.ndarray,
    protected: np.ndarray,
    sprite_box: tuple[int, int, int, int],
) -> dict[str, list[tuple[int, int]]]:
    h, w = shape
    r0_box, c0_box, r1_box, c1_box = sprite_box
    out: dict[str, list[tuple[int, int]]] = {}
    for quadrant in QUADRANTS:
        patch = _rotate_for_quadrant(sprite, quadrant)
        ph, pw = patch.shape
        rows = _quadrant_ranges(h, quadrant in ("TL", "TR"), ph)
        cols = _quadrant_ranges(w, quadrant in ("TL", "BL"), pw)
        cells: list[tuple[int, int]] = []
        for r in rows:
            for c in cols:
                if r0_box <= r <= r1_box and c0_box <= c <= c1_box:
                    continue
                if _stamp_hits_protected(patch, protected, r, c):
                    continue
                cells.append((r, c))
        out[quadrant] = cells
    return out


def _stamp_bbox(anchor: tuple[int, int], patch: np.ndarray) -> tuple[int, int, int, int]:
    r, c = anchor
    ph, pw = patch.shape
    return r, c, r + ph - 1, c + pw - 1


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _choose_anchor_map(
    candidates: dict[str, list[tuple[int, int]]],
    sprite: np.ndarray,
) -> dict[str, tuple[int, int]] | None:
    count = 4 if random.random() < 0.65 else 3
    for _ in range(GEN_ATTEMPTS):
        quadrants = random.sample(list(QUADRANTS), count)
        chosen: dict[str, tuple[int, int]] = {}
        boxes: list[tuple[int, int, int, int]] = []
        ok = True
        for quadrant in quadrants:
            if not candidates[quadrant]:
                ok = False
                break
            patch = _rotate_for_quadrant(sprite, quadrant)
            random.shuffle(candidates[quadrant])
            placed = None
            for anchor in candidates[quadrant]:
                box = _stamp_bbox(anchor, patch)
                if any(_bbox_overlap(box, other) > 4 for other in boxes):
                    continue
                placed = anchor
                boxes.append(box)
                break
            if placed is None:
                ok = False
                break
            chosen[quadrant] = placed
        if ok:
            return chosen
    return None


def _reserve_stamp_boxes(
    reserved: np.ndarray,
    sprite: np.ndarray,
    anchor_map: dict[str, tuple[int, int]],
) -> None:
    for quadrant, anchor in anchor_map.items():
        patch = _rotate_for_quadrant(sprite, quadrant)
        r0, c0, r1, c1 = _stamp_bbox(anchor, patch)
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True


def _free_cells_by_quadrant(mask: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    pools = {quadrant: [] for quadrant in QUADRANTS}
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            if h % 2 == 1 and r == h // 2:
                continue
            if w % 2 == 1 and c == w // 2:
                continue
            pools[_quadrant((h, w), r, c)].append((r, c))
    return pools


def _place_singletons(
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
    count: int,
) -> bool:
    pools = _free_cells_by_quadrant(reserved)
    choices: list[tuple[int, int]] = []
    spread = min(3, count)
    quadrants = random.sample(list(QUADRANTS), spread)
    for quadrant in quadrants:
        if not pools[quadrant]:
            return False
        cell = random.choice(pools[quadrant])
        choices.append(cell)
        reserved[cell] = True
        pools = _free_cells_by_quadrant(reserved)
    remaining = count - len(choices)
    all_free = [cell for cell in np.argwhere(~reserved)]
    if remaining > len(all_free):
        return False
    for rr, cc in random.sample([(int(r), int(c)) for r, c in all_free], remaining):
        choices.append((rr, cc))
        reserved[rr, cc] = True
    for r, c in choices:
        grid[r, c] = color
    return True


def _validate_grid(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        if g.ndim != 2 or max(g.shape) > 30:
            return False
        green_mask = g == GREEN
        comps = _components4(green_mask)
        if len(comps) != 1:
            return False
        sprite_box = _bbox(green_mask)
        if sprite_box is None:
            return False
        sprite_mask = _crop_bbox(green_mask.astype(int), sprite_box).astype(bool)
        if not (8 <= int(sprite_mask.sum()) <= 14):
            return False
        if not _rotation_distinct(sprite_mask):
            return False
        counts = {color: int(np.count_nonzero(g == color)) for color in MARKER_COLORS}
        if not all(counts[color] > 0 for color in MARKER_COLORS):
            return False
        least = min(counts.values())
        selected = [color for color, count in counts.items() if count == least]
        if len(selected) != 1 or least not in (3, 4):
            return False
        selected_color = selected[0]
        anchors = [(int(r), int(c)) for r, c in np.argwhere(g == selected_color)]
        quadrants = [_quadrant(g.shape, r, c) for r, c in anchors]
        if len(set(quadrants)) != len(quadrants):
            return False
        for (r, c), quadrant in zip(anchors, quadrants):
            patch = _rotate_for_quadrant(_crop_bbox(g, sprite_box), quadrant)
            ph, pw = patch.shape
            if r + ph > g.shape[0] or c + pw > g.shape[1]:
                return False
            if _stamp_hits_protected(patch, green_mask, r, c):
                return False
        out = main(g)
        if out.shape != g.shape:
            return False
        if np.array_equal(out, g):
            return False
        if np.count_nonzero(out == GREEN) <= np.count_nonzero(green_mask):
            return False
        for color in MARKER_COLORS:
            if color != selected_color and np.any(out == color):
                return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        height = random.choice(GRID_SIDES)
        width = random.choice(GRID_SIDES)
        grid = np.zeros((height, width), dtype=int)

        sprite_mask = _make_sprite_mask()
        sprite = sprite_mask.astype(int) * GREEN
        grid, protected = _place_sprite(grid, sprite_mask)
        sprite_box = _bbox(protected)
        if sprite_box is None:
            continue

        candidates = _candidate_anchors(grid.shape, sprite, protected, sprite_box)
        anchor_map = _choose_anchor_map(candidates, sprite)
        if anchor_map is None:
            continue

        selected_color = random.choice(MARKER_COLORS)
        for quadrant, anchor in anchor_map.items():
            grid[anchor] = selected_color

        least_count = len(anchor_map)
        other_colors = [color for color in MARKER_COLORS if color != selected_color]
        other_counts = [random.randint(least_count + 1, least_count + 4) for _ in other_colors]
        reserved = protected.copy()
        _reserve_stamp_boxes(reserved, sprite, anchor_map)

        ok = True
        for color, count in zip(other_colors, other_counts):
            if not _place_singletons(grid, reserved, color, count):
                ok = False
                break
        if not ok:
            continue
        if not _validate_grid(grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input grid")
