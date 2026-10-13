from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
FRAME = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("top", "bottom", "left", "right")
CORNERS = ("tl", "tr", "bl", "br")
GEN_ATTEMPTS = 400


def _normalize(cells: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _rotate(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    max_r = max(r for r, _c in shape)
    return _normalize(tuple((c, max_r - r) for r, c in shape))


def _flip(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    max_c = max(c for _r, c in shape)
    return _normalize(tuple((r, max_c - c) for r, c in shape))


def _variants(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], ...]:
    out: set[tuple[tuple[int, int], ...]] = set()
    cur = _normalize(shape)
    for _ in range(4):
        out.add(cur)
        out.add(_flip(cur))
        cur = _rotate(cur)
    return tuple(sorted(out))


def _shape_hw(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _c in shape) + 1, max(c for _r, c in shape) + 1


BASE_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
)

ALL_SHAPES = tuple(
    sorted(
        {variant for base in BASE_SHAPES for variant in _variants(base)},
        key=lambda shape: (len(shape), _shape_hw(shape), shape),
    )
)
SQUARE_2 = _normalize(((0, 0), (0, 1), (1, 0), (1, 1)))


def _outer_frame_color(grid: np.ndarray) -> int:
    outer = np.concatenate([grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]])
    colors = np.unique(outer)
    assert len(colors) == 1, "expected a uniform 1-cell border"
    return int(colors[0])


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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _contact_count(coords: np.ndarray, h: int, w: int) -> int:
    sides = 0
    if np.any(coords[:, 0] == 1):
        sides += 1
    if np.any(coords[:, 0] == h - 2):
        sides += 1
    if np.any(coords[:, 1] == 1):
        sides += 1
    if np.any(coords[:, 1] == w - 2):
        sides += 1
    return sides


def _scaled_cells_from_coords(coords: np.ndarray, scale: int, h: int, w: int) -> list[tuple[int, int, int]]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    bh = r1 - r0 + 1
    bw = c1 - c0 + 1
    start_r = (r0 + r1 - bh * scale + 1) // 2
    start_c = (c0 + c1 - bw * scale + 1) // 2

    cells: list[tuple[int, int, int]] = []
    for r, c in coords:
        local_r = int(r - r0) * scale
        local_c = int(c - c0) * scale
        for dr in range(scale):
            rr = int(start_r + local_r + dr)
            if not (1 <= rr < h - 1):
                continue
            for dc in range(scale):
                cc = int(start_c + local_c + dc)
                if 1 <= cc < w - 1:
                    cells.append((rr, cc, int(r * w + c)))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """Scale each small interior object in place according to how many frame sides it touches.

    Input structure:
    - The background is black.
    - The whole outer ring is a uniform 1-pixel grey frame.
    - Inside the frame are several separated small 4-connected colored objects, each 2 to 4 cells.
    - An object may touch zero, one, or two frame sides by sitting 4-neighbor adjacent to the inner edge of the frame.

    Rule:
    - Count how many distinct frame sides each object touches.
    - Leave 0-side objects unchanged, scale 1-side objects by 2, and scale 2-side objects by 3.
    - Draw the scaled replacement about the original bbox center and clip any part that leaves the frame interior.
    - Remove the original object cells, keep the grey frame unchanged, and resolve overlaps by letting larger scale factors overwrite smaller ones.

    Key insight:
    - The transformation parameter is encoded by spatial contact with the frame: border-side count directly determines the object's scale factor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 3 and w >= 3

    frame_color = _outer_frame_color(g)
    mask = (g != BG) & (g != frame_color)
    comps = _components(mask)
    assert comps, "expected at least one interior object"

    out = np.full_like(g, BG)
    out[0, :] = frame_color
    out[-1, :] = frame_color
    out[:, 0] = frame_color
    out[:, -1] = frame_color

    ordered: list[tuple[int, int, np.ndarray]] = []
    for idx, coords in enumerate(comps):
        touches = _contact_count(coords, h, w)
        assert 0 <= touches <= 2, "objects should touch at most two frame sides"
        ordered.append((touches + 1, idx, coords))

    for scale, _idx, coords in sorted(ordered, key=lambda item: (item[0], item[1])):
        for rr, cc, key in _scaled_cells_from_coords(coords, scale, h, w):
            src_r = key // w
            src_c = key % w
            out[rr, cc] = int(g[src_r, src_c])
    return out


def _offset(shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((r0 + r, c0 + c) for r, c in shape))


def _can_place(cells: tuple[tuple[int, int], ...], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            return False
        if occupied[r, c]:
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w and occupied[rr, cc]:
                    return False
    return True


def _place_object(
    grid: np.ndarray,
    occupied: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    r0: int,
    c0: int,
    color: int,
) -> dict[str, object] | None:
    cells = _offset(shape, r0, c0)
    if not _can_place(cells, occupied):
        return None
    for r, c in cells:
        grid[r, c] = color
        occupied[r, c] = True
    return {"cells": cells, "color": int(color)}


def _candidate_offsets(
    shape: tuple[tuple[int, int], ...],
    h: int,
    w: int,
    category: int,
    label: str | None,
) -> list[tuple[int, int]]:
    bh, bw = _shape_hw(shape)
    positions: list[tuple[int, int]] = []
    if category == 0:
        for r0 in range(2, h - bh - 1):
            for c0 in range(2, w - bw - 1):
                positions.append((r0, c0))
    elif category == 1:
        assert label in SIDES
        if label == "top":
            for c0 in range(2, w - bw - 1):
                positions.append((1, c0))
        elif label == "bottom":
            for c0 in range(2, w - bw - 1):
                positions.append((h - bh - 1, c0))
        elif label == "left":
            for r0 in range(2, h - bh - 1):
                positions.append((r0, 1))
        else:
            for r0 in range(2, h - bh - 1):
                positions.append((r0, w - bw - 1))
    else:
        assert label in CORNERS
        if label == "tl":
            positions.append((1, 1))
        elif label == "tr":
            positions.append((1, w - bw - 1))
        elif label == "bl":
            positions.append((h - bh - 1, 1))
        else:
            positions.append((h - bh - 1, w - bw - 1))
    random.shuffle(positions)
    return positions


def _random_shape() -> tuple[tuple[int, int], ...]:
    return random.choice(ALL_SHAPES)


def _overlap_anchor(corner: str, h: int, w: int) -> tuple[int, int]:
    if corner == "tl":
        return 4, 4
    if corner == "tr":
        return 4, w - 6
    if corner == "bl":
        return h - 6, 4
    return h - 6, w - 6


def _scene_visible_cells(cells: tuple[tuple[int, int], ...], h: int, w: int) -> tuple[int, set[tuple[int, int]]]:
    coords = np.array(cells, dtype=int)
    scale = _contact_count(coords, h, w) + 1
    visible = {(rr, cc) for rr, cc, _key in _scaled_cells_from_coords(coords, scale, h, w)}
    return scale, visible


def _validate_scene(grid: np.ndarray, objects: list[dict[str, object]]) -> bool:
    h, w = grid.shape
    counts = [_contact_count(np.array(obj["cells"], dtype=int), h, w) for obj in objects]
    if set(counts) != {0, 1, 2}:
        return False

    infos = []
    for idx, obj in enumerate(objects):
        scale, visible = _scene_visible_cells(obj["cells"], h, w)
        if not visible:
            return False
        infos.append((idx, scale, visible))

    diff_scale_overlap = False
    for i in range(len(infos)):
        for j in range(i + 1, len(infos)):
            _idx_i, scale_i, vis_i = infos[i]
            _idx_j, scale_j, vis_j = infos[j]
            overlap = vis_i & vis_j
            if not overlap:
                continue
            if scale_i == scale_j:
                return False
            diff_scale_overlap = True
    if not diff_scale_overlap:
        return False

    owner = np.full((h, w), -1, dtype=int)
    priority = np.full((h, w), -1, dtype=int)
    for idx, scale, visible in sorted(infos, key=lambda item: (item[1], item[0])):
        for r, c in visible:
            owner[r, c] = idx
            priority[r, c] = scale
    if any(idx not in owner for idx, _scale, _visible in infos):
        return False

    try:
        out = main(grid)
    except Exception:
        return False
    if out.shape != grid.shape:
        return False
    if not np.all(out[0, :] == FRAME) or not np.all(out[-1, :] == FRAME):
        return False
    if not np.all(out[:, 0] == FRAME) or not np.all(out[:, -1] == FRAME):
        return False
    return True


def _attempt_place_random(
    grid: np.ndarray,
    occupied: np.ndarray,
    color: int,
    category: int,
    label: str | None,
) -> dict[str, object] | None:
    shapes = list(ALL_SHAPES)
    random.shuffle(shapes)
    for shape in shapes:
        positions = _candidate_offsets(shape, grid.shape[0], grid.shape[1], category, label)
        for r0, c0 in positions[:60]:
            placed = _place_object(grid, occupied, shape, r0, c0, color)
            if placed is not None:
                return placed
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 22)
    w = random.randint(12, 22)
    object_count = random.randint(3, 5)
    colors = random.sample(PALETTE, object_count)

    grid = np.full((h, w), BG, dtype=int)
    grid[0, :] = FRAME
    grid[-1, :] = FRAME
    grid[:, 0] = FRAME
    grid[:, -1] = FRAME
    occupied = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []

    primary_corner = random.choice(CORNERS)
    primary = _place_object(
        grid,
        occupied,
        SQUARE_2,
        *_candidate_offsets(SQUARE_2, h, w, 2, primary_corner)[0],
        colors[0],
    )
    if primary is None:
        return None
    objects.append(primary)

    overlap_r0, overlap_c0 = _overlap_anchor(primary_corner, h, w)
    overlap_obj = _place_object(grid, occupied, SQUARE_2, overlap_r0, overlap_c0, colors[1])
    if overlap_obj is None:
        return None
    objects.append(overlap_obj)

    side = random.choice(SIDES)
    edge_obj = _attempt_place_random(grid, occupied, colors[2], 1, side)
    if edge_obj is None:
        return None
    objects.append(edge_obj)

    for color in colors[3:]:
        category = random.choice((0, 1, 2))
        if category == 0:
            label = None
        elif category == 1:
            label = random.choice(SIDES)
        else:
            label = random.choice(CORNERS)
        placed = _attempt_place_random(grid, occupied, color, category, label)
        if placed is None:
            return None
        objects.append(placed)

    if not _validate_scene(grid, objects):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid border-touch scaling task")
