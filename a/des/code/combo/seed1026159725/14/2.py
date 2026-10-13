import random
from collections import deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _corner_name(local_r: int, local_c: int, h: int, w: int) -> str:
    if (local_r, local_c) == (0, 0):
        return "tl"
    if (local_r, local_c) == (0, w - 1):
        return "tr"
    if (local_r, local_c) == (h - 1, 0):
        return "bl"
    if (local_r, local_c) == (h - 1, w - 1):
        return "br"
    raise AssertionError("marker must be at a bbox corner")


def _quadrant_offset(name: str, h: int, w: int) -> tuple[int, int]:
    offsets = {
        "tl": (0, 0),
        "tr": (0, w),
        "bl": (h, 0),
        "br": (h, w),
    }
    return offsets[name]


def _sprite_patch(grid: np.ndarray, coords: np.ndarray, bg: int) -> tuple[str, np.ndarray]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    h, w = r1 - r0 + 1, c1 - c0 + 1

    colors, counts = np.unique(grid[coords[:, 0], coords[:, 1]], return_counts=True)
    assert len(colors) == 2, "each sprite must have one body color and one marker color"
    order = np.argsort(counts)
    marker_color = int(colors[order[0]])
    body_color = int(colors[order[1]])
    assert int(counts[order[0]]) == 1, "marker must be unique inside each sprite"

    marker = np.argwhere(grid[r0:r1 + 1, c0:c1 + 1] == marker_color)
    assert marker.shape[0] == 1
    local_r, local_c = map(int, marker[0])
    corner = _corner_name(local_r, local_c, h, w)

    patch = np.full((h, w), bg, dtype=int)
    rel = coords - np.array([r0, c0], dtype=int)
    patch[rel[:, 0], rel[:, 1]] = body_color
    return corner, patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble same-color sprites into a 2x2 layout using the corner position of
    each unique marker pixel as the placement instruction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    parts = _components(g != bg)
    assert parts, "expected at least one sprite"

    patches: dict[str, np.ndarray] = {}
    base_shape: tuple[int, int] | None = None
    for coords in parts:
        corner, patch = _sprite_patch(g, coords, bg)
        if base_shape is None:
            base_shape = patch.shape
        assert patch.shape == base_shape, "all sprites must share the same bbox size"
        assert corner not in patches, "at most one sprite per quadrant"
        patches[corner] = patch

    assert base_shape is not None
    h, w = base_shape
    out = np.full((2 * h, 2 * w), bg, dtype=int)
    for corner, patch in patches.items():
        r0, c0 = _quadrant_offset(corner, h, w)
        out[r0:r0 + h, c0:c0 + w] = patch
    return out


_RETRY_LIMIT = 300
_PALETTE = tuple(range(10))
_GRID_MIN = 12
_GRID_MAX = 24
_BBOX_H_RANGE = (3, 6)
_BBOX_W_RANGE = (3, 7)
_AREA_FRACTION_RANGE = (0.4, 0.72)
_PLACEMENT_GAP = 1
_CORNERS = ("tl", "tr", "bl", "br")


def _random_path_to_side(
    start: tuple[int, int],
    h: int,
    w: int,
    target: str,
) -> set[tuple[int, int]]:
    r, c = start
    cells = {(r, c)}
    while True:
        if target == "top" and r == 0:
            return cells
        if target == "bottom" and r == h - 1:
            return cells
        if target == "left" and c == 0:
            return cells
        if target == "right" and c == w - 1:
            return cells

        moves: list[tuple[int, int]] = []
        if target == "top":
            moves.append((r - 1, c))
        elif target == "bottom":
            moves.append((r + 1, c))
        elif target == "left":
            moves.append((r, c - 1))
        else:
            moves.append((r, c + 1))

        if random.random() < 0.45:
            for nr, nc in ((r, c - 1), (r, c + 1), (r - 1, c), (r + 1, c)):
                if 0 <= nr < h and 0 <= nc < w:
                    if target in ("top", "bottom") and nr == r:
                        moves.append((nr, nc))
                    if target in ("left", "right") and nc == c:
                        moves.append((nr, nc))

        valid = []
        for nr, nc in moves:
            if 0 <= nr < h and 0 <= nc < w:
                valid.append((nr, nc))
        r, c = random.choice(valid)
        cells.add((r, c))


def _required_sides(corner: str) -> tuple[str, str]:
    if corner == "tl":
        return "bottom", "right"
    if corner == "tr":
        return "bottom", "left"
    if corner == "bl":
        return "top", "right"
    return "top", "left"


def _corner_cell(corner: str, h: int, w: int) -> tuple[int, int]:
    corners = {
        "tl": (0, 0),
        "tr": (0, w - 1),
        "bl": (h - 1, 0),
        "br": (h - 1, w - 1),
    }
    return corners[corner]


def _touches_all_sides(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    return (
        any(r == 0 for r, _ in cells)
        and any(r == h - 1 for r, _ in cells)
        and any(c == 0 for _, c in cells)
        and any(c == w - 1 for _, c in cells)
    )


def _random_sprite_mask(h: int, w: int, corner: str) -> np.ndarray:
    start = _corner_cell(corner, h, w)
    target_area = random.randint(
        max(5, int(np.ceil(h * w * _AREA_FRACTION_RANGE[0]))),
        max(6, int(np.floor(h * w * _AREA_FRACTION_RANGE[1]))),
    )
    cells = {start}
    for side in _required_sides(corner):
        cells |= _random_path_to_side(start, h, w, side)

    frontier = set()
    for r, c in list(cells):
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    while frontier and len(cells) < target_area:
        choice = random.choice(tuple(frontier))
        frontier.remove(choice)
        cells.add(choice)
        r, c = choice
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    assert _touches_all_sides(cells, h, w)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _bbox_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _paint_sprite(
    grid: np.ndarray,
    top: int,
    left: int,
    mask: np.ndarray,
    body_color: int,
    marker_color: int,
    corner: str,
) -> None:
    h, w = mask.shape
    grid[top:top + h, left:left + w][mask] = body_color
    mr, mc = _corner_cell(corner, h, w)
    grid[top + mr, left + mc] = marker_color


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_PALETTE)
    remaining = [c for c in _PALETTE if c != bg]
    body_color = random.choice(remaining)
    marker_pool = [c for c in remaining if c != body_color]
    marker_count = random.randint(1, min(4, len(marker_pool)))
    marker_colors = random.sample(marker_pool, marker_count)

    h = random.randint(*_BBOX_H_RANGE)
    w = random.randint(*_BBOX_W_RANGE)
    min_h = max(_GRID_MIN, 2 * h + 3)
    min_w = max(_GRID_MIN, 2 * w + 3)
    if min_h > _GRID_MAX or min_w > _GRID_MAX:
        return None
    grid_h = random.randint(min_h, _GRID_MAX)
    grid_w = random.randint(min_w, _GRID_MAX)
    grid = np.full((grid_h, grid_w), bg, dtype=int)

    placements: list[tuple[int, int, int, int]] = []
    sprite_specs: list[tuple[int, int, np.ndarray, int, str]] = []
    random_corners = list(_CORNERS)
    random.shuffle(random_corners)
    for idx, corner in enumerate(random_corners):
        mask = _random_sprite_mask(h, w, corner)
        placed = False
        for _ in range(120):
            top = random.randint(0, grid_h - h)
            left = random.randint(0, grid_w - w)
            bbox = (top, left, top + h - 1, left + w - 1)
            if any(_bbox_conflict(bbox, other, _PLACEMENT_GAP) for other in placements):
                continue
            marker_color = marker_colors[idx % marker_count]
            sprite_specs.append((top, left, mask, marker_color, corner))
            placements.append(bbox)
            placed = True
            break
        if not placed:
            return None

    for top, left, mask, marker_color, corner in sprite_specs:
        _paint_sprite(grid, top, left, mask, body_color, marker_color, corner)

    if _get_bg(grid) != bg:
        return None

    parts = _components(grid != bg)
    if len(parts) != 4:
        return None

    quadrant_names = set()
    for coords in parts:
        try:
            corner, patch = _sprite_patch(grid, coords, bg)
        except AssertionError:
            return None
        if patch.shape != (h, w):
            return None
        quadrant_names.add(corner)
    if quadrant_names != set(_CORNERS):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
