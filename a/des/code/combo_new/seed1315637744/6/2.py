from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
PIECE_COLORS = [3, 4, 5, 6, 7, 8, 9]

MAX_GENERATION_ATTEMPTS = 400
GRID_SIDE_MIN = 18
GRID_SIDE_MAX = 30
FRAME_COUNT_CHOICES = [2, 3, 4]
INNER_H_RANGE = (4, 8)
INNER_W_RANGE = (4, 8)
NOTCH_COUNT_RANGE = (2, 5)
PADDING = 1
PLACE_TRIES = 300


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, (r0, c0, r1, c1)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.tobytes()


def _canon180_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    a = _mask_key(mask)
    b = _mask_key(np.rot90(mask, 2))
    return min(a, b)


def _enclosed_background_masks(frame_mask: np.ndarray) -> list[np.ndarray]:
    h, w = frame_mask.shape
    bg_mask = ~frame_mask
    enclosed: list[np.ndarray] = []
    for comp in _components(bg_mask):
        r0, c0, r1, c1 = _bbox(comp)
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        local = np.zeros((h, w), dtype=bool)
        local[comp[:, 0], comp[:, 1]] = True
        enclosed.append(local)
    return enclosed


def _paste_mask(canvas: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    patch = canvas[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = int(color)


def _place_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    canvas[r0 : r0 + h, c0 : c0 + w] = patch


def _patch_from_mask(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _frame_candidates(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    frames: list[dict] = []
    pieces: list[dict] = []
    for comp in _components(grid != int(bg)):
        mask, bbox = _tight_mask(comp)
        r0, c0, r1, c1 = bbox
        colors = np.unique(grid[comp[:, 0], comp[:, 1]])
        assert len(colors) == 1, "each object should be monochromatic"
        color = int(colors[0])
        enclosed = _enclosed_background_masks(mask)
        if enclosed:
            cavity = np.zeros_like(mask)
            for part in enclosed:
                cavity |= part
            cavity_coords = np.argwhere(cavity)
            cav_mask, cav_bbox_local = _tight_mask(cavity_coords)
            frames.append(
                {
                    "color": color,
                    "bbox": bbox,
                    "mask": mask,
                    "cavity_mask": cav_mask,
                    "cavity_bbox": (
                        int(r0 + cav_bbox_local[0]),
                        int(c0 + cav_bbox_local[1]),
                        int(r0 + cav_bbox_local[2]),
                        int(c0 + cav_bbox_local[3]),
                    ),
                }
            )
        else:
            pieces.append(
                {
                    "color": color,
                    "bbox": bbox,
                    "mask": mask,
                }
            )
    return frames, pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit scattered solid pieces back into matching framed cavities, rotating only for blue frames.

    Input structure:
    - The background is black.
    - Several disconnected monochromatic frame objects are hollow rectangles whose inner boundary is reshaped by inward border notches.
    - Outside the frames are equally many monochromatic solid piece components whose shapes match the frame cavities up to a possible 180-degree turn.

    Rule:
    - Identify each frame's enclosed cavity as the background region fully surrounded by its border.
    - Pair every outside piece with the unique frame cavity that has the same shape, using the piece as-is for red frames and its 180-degree rotation for blue frames.
    - Erase the pieces from their original positions and place each matched piece inside its frame cavity.

    Key insight:
    - The geometry of the enclosed cavity determines the piece-to-frame match, while the frame color encodes whether the piece must keep its orientation or be turned upside down.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    frames, pieces = _frame_candidates(g, bg)

    assert frames, "expected at least one frame"
    assert len(frames) == len(pieces), "frame and piece counts must match"

    frame_infos = []
    for frame in frames:
        frame_infos.append(
            {
                **frame,
                "need_blue_rotation": int(frame["color"]) == BLUE,
            }
        )

    candidates: list[list[int]] = []
    for fi, frame in enumerate(frame_infos):
        matches: list[int] = []
        for pi, piece in enumerate(pieces):
            placed_mask = np.rot90(piece["mask"], 2) if frame["need_blue_rotation"] else piece["mask"]
            if placed_mask.shape == frame["cavity_mask"].shape and np.array_equal(placed_mask, frame["cavity_mask"]):
                matches.append(int(pi))
        assert matches, "every frame must have at least one matching piece"
        candidates.append(matches)

    order = sorted(range(len(frame_infos)), key=lambda idx: len(candidates[idx]))
    assignment = [-1] * len(frame_infos)
    used_pieces: set[int] = set()

    def search(k: int) -> bool:
        if k == len(order):
            return True
        fi = order[k]
        for pi in candidates[fi]:
            if pi in used_pieces:
                continue
            used_pieces.add(pi)
            assignment[fi] = int(pi)
            if search(k + 1):
                return True
            used_pieces.remove(pi)
            assignment[fi] = -1
        return False

    assert search(0), "piece-to-frame matching must be solvable"

    out = g.copy()
    for piece in pieces:
        r0, c0, r1, c1 = piece["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1][piece["mask"]] = int(bg)

    for fi, pi in enumerate(assignment):
        frame = frame_infos[fi]
        piece = pieces[pi]
        placed_mask = np.rot90(piece["mask"], 2) if frame["need_blue_rotation"] else piece["mask"]
        cr0, cc0, _cr1, _cc1 = frame["cavity_bbox"]
        _paste_mask(out, placed_mask, int(cr0), int(cc0), int(piece["color"]))

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    if not bool(mask.any()):
        return False
    comps = _components(mask)
    return len(comps) == 1


def _random_notch(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    notch = np.zeros((h, w), dtype=bool)
    side = random.choice(["top", "bottom", "left", "right"])
    max_depth = max(1, (h - 1) if side in {"top", "bottom"} else (w - 1))
    depth = int(rng.integers(1, max_depth))
    width = 1
    if side in {"top", "bottom"}:
        col = int(rng.integers(0, w))
        row_start = 0 if side == "top" else h - depth
        notch[row_start : row_start + depth, col : col + width] = True
    else:
        row = int(rng.integers(0, h))
        col_start = 0 if side == "left" else w - depth
        notch[row : row + width, col_start : col_start + depth] = True
    return notch


def _random_cavity_mask(rng: np.random.Generator) -> np.ndarray | None:
    inner_h = int(rng.integers(INNER_H_RANGE[0], INNER_H_RANGE[1] + 1))
    inner_w = int(rng.integers(INNER_W_RANGE[0], INNER_W_RANGE[1] + 1))
    cavity = np.ones((inner_h, inner_w), dtype=bool)
    notch_count = int(rng.integers(NOTCH_COUNT_RANGE[0], NOTCH_COUNT_RANGE[1] + 1))
    for _ in range(notch_count):
        candidate = cavity & ~_random_notch(rng, inner_h, inner_w)
        if bool(candidate.any()) and _mask_connected(candidate):
            cavity = candidate
    if cavity.shape[0] < 3 or cavity.shape[1] < 3:
        return None
    if not _mask_connected(cavity):
        return None
    if np.array_equal(cavity, np.rot90(cavity, 2)):
        return None
    area = int(cavity.sum())
    if area < max(6, (inner_h * inner_w) // 2):
        return None
    return cavity


def _frame_patch_from_cavity(cavity: np.ndarray, frame_color: int, bg: int) -> np.ndarray:
    inner_h, inner_w = cavity.shape
    patch = np.full((inner_h + 2, inner_w + 2), int(bg), dtype=int)
    patch[0, :] = int(frame_color)
    patch[-1, :] = int(frame_color)
    patch[:, 0] = int(frame_color)
    patch[:, -1] = int(frame_color)
    patch[1:-1, 1:-1][~cavity] = int(frame_color)
    return patch


def _can_place(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int) -> bool:
    h, w = patch.shape
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(H, r0 + h + padding)
    cc1 = min(W, c0 + w + padding)
    if np.any(occupied[rr0:rr1, cc0:cc1]):
        return False
    return True


def _mark_occupied(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    occupied[r0 : r0 + h, c0 : c0 + w] |= patch != BG


def _random_place(rng: np.random.Generator, occupied: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    H, W = occupied.shape
    h, w = patch.shape
    if h > H or w > W:
        return None
    positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    rng.shuffle(positions)
    for r0, c0 in positions[:PLACE_TRIES]:
        if _can_place(occupied, patch, int(r0), int(c0), PADDING):
            return int(r0), int(c0)
    return None


def _branch_colors(rng: np.random.Generator, count: int) -> list[int]:
    if count == 2:
        colors = [RED, BLUE]
    elif count == 3:
        colors = [RED, RED, BLUE] if bool(rng.integers(0, 2)) else [RED, BLUE, BLUE]
    else:
        colors = [RED, RED, BLUE, BLUE]
    rng.shuffle(colors)
    return [int(c) for c in colors]


def _expected_output(shape_specs: list[dict], height: int, width: int) -> np.ndarray:
    out = np.full((height, width), BG, dtype=int)
    for spec in shape_specs:
        _place_patch(out, spec["frame_patch"], spec["frame_r"], spec["frame_c"])
        cr0 = int(spec["frame_r"] + 1)
        cc0 = int(spec["frame_c"] + 1)
        _paste_mask(out, spec["cavity"], cr0, cc0, int(spec["piece_color"]))
    return out


def _input_from_specs(shape_specs: list[dict], height: int, width: int) -> np.ndarray:
    inp = np.full((height, width), BG, dtype=int)
    for spec in shape_specs:
        _place_patch(inp, spec["frame_patch"], spec["frame_r"], spec["frame_c"])
        _place_patch(inp, spec["piece_patch"], spec["piece_r"], spec["piece_c"])
    return inp


def _generate_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    frame_count = int(rng.choice(np.array(FRAME_COUNT_CHOICES, dtype=int)))
    frame_colors = _branch_colors(rng, frame_count)
    piece_colors = list(map(int, rng.choice(np.array(PIECE_COLORS, dtype=int), size=frame_count, replace=False)))

    cavities: list[np.ndarray] = []
    keys: set[tuple[tuple[int, int], bytes]] = set()
    for _ in range(frame_count):
        cavity = None
        for _ in range(60):
            candidate = _random_cavity_mask(rng)
            if candidate is None:
                continue
            key = _canon180_key(candidate)
            if key in keys:
                continue
            cavity = candidate
            keys.add(key)
            break
        if cavity is None:
            return None
        cavities.append(cavity)

    shape_specs: list[dict] = []
    total_area = 0
    for cavity, frame_color, piece_color in zip(cavities, frame_colors, piece_colors):
        frame_patch = _frame_patch_from_cavity(cavity, int(frame_color), BG)
        piece_mask = cavity if int(frame_color) == RED else np.rot90(cavity, 2)
        piece_patch = _patch_from_mask(piece_mask, int(piece_color), BG)
        total_area += int((frame_patch != BG).sum() + piece_mask.sum())
        shape_specs.append(
            {
                "cavity": cavity,
                "frame_patch": frame_patch,
                "piece_patch": piece_patch,
                "piece_color": int(piece_color),
            }
        )

    min_side = max(GRID_SIDE_MIN, int(np.ceil(np.sqrt(total_area * 2.3))))
    if min_side > GRID_SIDE_MAX:
        return None
    height = int(rng.integers(min_side, GRID_SIDE_MAX + 1))
    width = int(rng.integers(min_side, GRID_SIDE_MAX + 1))

    occupied = np.zeros((height, width), dtype=bool)
    order = list(range(frame_count))
    rng.shuffle(order)
    order.sort(key=lambda idx: int(shape_specs[idx]["frame_patch"].shape[0] * shape_specs[idx]["frame_patch"].shape[1]), reverse=True)
    for idx in order:
        spec = shape_specs[idx]
        pos = _random_place(rng, occupied, spec["frame_patch"])
        if pos is None:
            return None
        spec["frame_r"], spec["frame_c"] = pos
        _mark_occupied(occupied, spec["frame_patch"], int(pos[0]), int(pos[1]))

    rng.shuffle(order)
    order.sort(key=lambda idx: int(shape_specs[idx]["piece_patch"].shape[0] * shape_specs[idx]["piece_patch"].shape[1]), reverse=True)
    for idx in order:
        spec = shape_specs[idx]
        pos = _random_place(rng, occupied, spec["piece_patch"])
        if pos is None:
            return None
        spec["piece_r"], spec["piece_c"] = pos
        _mark_occupied(occupied, spec["piece_patch"], int(pos[0]), int(pos[1]))

    inp = _input_from_specs(shape_specs, height, width)
    out = _expected_output(shape_specs, height, width)

    if Counter(inp.flatten()).most_common(1)[0][0] != BG:
        return None
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
