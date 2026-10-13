import random
from collections import deque

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 3
PIECE_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)

GRID_MIN = 14
GRID_MAX = 30
SHAPE_RETRIES = 120
GEN_RETRIES = 400
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _coords_to_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "mask": _coords_to_mask(coords),
                }
            )
    return comps


def _mask_key(mask: np.ndarray) -> tuple:
    coords = tuple(map(int, np.argwhere(mask).flatten()))
    return mask.shape, coords


def _is_hollow_rectangle(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(mask, border)


def _orientation_data(piece_masks: list[np.ndarray]) -> tuple[np.ndarray, set[int]]:
    rotated = []
    for mask in piece_masks:
        for k in range(4):
            rmask = np.rot90(mask, k)
            rotated.append((_mask_key(rmask), rmask.copy()))
    canon_key, canon_mask = min(rotated, key=lambda item: item[0])

    present: set[int] = set()
    for mask in piece_masks:
        matches = [k for k in range(4) if _mask_key(np.rot90(mask, k)) == canon_key]
        assert len(matches) == 1
        k_to_canon = matches[0]
        present.add((-k_to_canon) % 4)
    return canon_mask, present


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the four-way rotation set by drawing the missing rotated sprite inside the green marker frame.

    Input structure:
    - The background is black.
    - One disconnected monochrome component is a hollow one-cell-thick green rectangle whose empty interior is the target slot.
    - Three other disconnected monochrome components share one non-black, non-green color and are the same asymmetric sprite shown in three different 90-degree rotations.

    Rule:
    - Identify the three sprite components and treat them as three members of one four-rotation cycle.
    - Determine which of the rotations {0, 90, 180, 270} is missing.
    - Keep the green frame unchanged and paint that missing rotation inside the frame, aligned to the frame's inner top-left corner so it fills the frame interior.

    Key insight:
    - The scattered colored components must be normalized as rotated copies of one asymmetric prototype; once the absent orientation is known, the frame is just a placeholder whose interior has the exact bounding-box size of that missing rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    comps = _find_components(g, bg)
    frame_candidates = [comp for comp in comps if _is_hollow_rectangle(comp["mask"])]
    assert len(frame_candidates) == 1
    frame = frame_candidates[0]

    pieces = [comp for comp in comps if comp is not frame]
    assert len(pieces) == 3
    piece_colors = {comp["color"] for comp in pieces}
    assert len(piece_colors) == 1
    piece_color = next(iter(piece_colors))

    canon_mask, present = _orientation_data([comp["mask"] for comp in pieces])
    assert len(present) == 3
    missing = next(iter({0, 1, 2, 3} - present))
    missing_mask = np.rot90(canon_mask, missing)

    r0, c0, r1, c1 = frame["bbox"]
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    assert inner.shape == missing_mask.shape

    out = g.copy()
    top, left = r0 + 1, c0 + 1
    rows, cols = np.nonzero(missing_mask)
    out[top + rows, left + cols] = piece_color
    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _random_shape() -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        h = random.randint(2, 5)
        w = random.randint(2, 6)
        target_hi = min(h * w - 1, 8)
        if target_hi < 4:
            continue
        target = random.randint(4, target_hi)
        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        mask[seed] = True

        while int(mask.sum()) < target:
            frontier = set()
            for r, c in np.argwhere(mask):
                for dr, dc in DIRS4:
                    nr, nc = int(r) + dr, int(c) + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            r, c = random.choice(tuple(frontier))
            mask[r, c] = True

        if int(mask.sum()) != target:
            continue
        mask = _crop_mask(mask)
        if min(mask.shape) < 2:
            continue
        if _is_hollow_rectangle(mask):
            continue
        if len({_mask_key(np.rot90(mask, k)) for k in range(4)}) != 4:
            continue
        return mask
    raise RuntimeError("failed to generate an asymmetric prototype")


def _border_patch(inner_shape: tuple[int, int]) -> np.ndarray:
    h, w = inner_shape
    patch = np.zeros((h + 2, w + 2), dtype=bool)
    patch[0, :] = True
    patch[-1, :] = True
    patch[:, 0] = True
    patch[:, -1] = True
    return patch


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr, dc in DIRS8:
        rs = slice(max(0, dr), min(h, h + dr))
        rt = slice(max(0, -dr), min(h, h - dr))
        cs = slice(max(0, dc), min(w, w + dc))
        ct = slice(max(0, -dc), min(w, w - dc))
        out[rt, ct] |= mask[rs, cs]
    return out


def _place_patch(
    grid: np.ndarray,
    occupied: np.ndarray,
    patch_mask: np.ndarray,
    color: int,
) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = patch_mask.shape
    blocked = _dilate8(occupied)
    coords = [(int(r), int(c)) for r, c in np.argwhere(patch_mask)]
    candidates = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            if any(blocked[r0 + r, c0 + c] for r, c in coords):
                continue
            candidates.append((r0, c0))
    if not candidates:
        return None

    r0, c0 = random.choice(candidates)
    rows, cols = np.nonzero(patch_mask)
    grid[r0 + rows, c0 + cols] = color
    occupied[r0 + rows, c0 + cols] = True
    return r0, c0


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), BG_COLOR, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    piece_color = random.choice(PIECE_COLORS)
    prototype = _random_shape()
    missing = random.randrange(4)
    rotations = [np.rot90(prototype, k) for k in range(4)]
    frame_inner = rotations[missing].shape
    frame_patch = _border_patch(frame_inner)

    objects = [("frame", frame_patch, FRAME_COLOR)]
    visible = [k for k in range(4) if k != missing]
    random.shuffle(visible)
    for k in visible:
        objects.append((f"piece_{k}", rotations[k], piece_color))
    objects.sort(key=lambda item: int(item[1].shape[0] * item[1].shape[1]), reverse=True)

    frame_origin = None
    for name, mask, color in objects:
        placed = _place_patch(grid, occupied, mask, color)
        if placed is None:
            return None
        if name == "frame":
            frame_origin = placed

    assert frame_origin is not None
    fr, fc = frame_origin
    expected = grid.copy()
    missing_mask = rotations[missing]
    rows, cols = np.nonzero(missing_mask)
    expected[fr + 1 + rows, fc + 1 + cols] = piece_color

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
