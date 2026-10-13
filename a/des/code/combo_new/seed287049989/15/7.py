from __future__ import annotations

import random
from typing import Iterable

import numpy as np


Coord = tuple[int, int]

BG = 0
COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 500
OUTER_PAD = 1
OBJECT_MARGIN = 1


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: Iterable[Coord]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: Iterable[Coord]) -> tuple[Coord, ...]:
    pts = list(coords)
    if not pts:
        return ()
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((r - min_r, c - min_c) for r, c in pts))


def _coords_to_mask(coords: Iterable[Coord]) -> np.ndarray:
    pts = _normalize(coords)
    if not pts:
        return np.zeros((0, 0), dtype=bool)
    h = max(r for r, _ in pts) + 1
    w = max(c for _, c in pts) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in pts:
        mask[r, c] = True
    return mask


def _rotate_shape(coords: Iterable[Coord], k: int) -> tuple[Coord, ...]:
    mask = _coords_to_mask(coords)
    rot = np.rot90(mask, int(k))
    return tuple(sorted(map(tuple, np.argwhere(rot))))


def _is_connected(coords: Iterable[Coord]) -> bool:
    pts = set(coords)
    if not pts:
        return False
    stack = [next(iter(pts))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if (nr, nc) in pts and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(pts)


def _border_cells(h: int, w: int) -> set[Coord]:
    cells = {(0, c) for c in range(w)}
    cells |= {(h - 1, c) for c in range(w)}
    cells |= {(r, 0) for r in range(h)}
    cells |= {(r, w - 1) for r in range(h)}
    return cells


def _frame_signature(coords: list[Coord]) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None
    local = {(r - r0, c - c0) for r, c in coords}
    border = _border_cells(h, w)
    if not local.issubset(border):
        return None
    missing = border - local
    if not (3 <= len(missing) <= 5):
        return None
    if not _is_connected(missing):
        return None
    return {
        "bbox": (r0, c0, r1, c1),
        "missing": _normalize(missing),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit each colored key into the matching missing border segment of its same-color frame, then fill the closed rectangle.

    Input structure:
    - The background is black.
    - Each non-black color forms one larger hollow rectangular frame with a small connected notch missing from its border, plus one smaller separated key sprite of the same color.
    - Frames are one-pixel-thick axis-aligned rectangles with black interiors; each notch is a 3-5 pixel connected border polyomino.

    Rule:
    - Pair every key with the frame of the same color.
    - Try the four 90-degree rotations of the key and find the unique rotation whose shape matches the frame's missing border polyomino, then translate the key into that notch.
    - Clear the key from its old position, close the frame, and fill the entire interior of the now-closed rectangle with that color.

    Key insight:
    - The decisive feature is the exact set of border cells missing from each rectangular frame: once the same-color key matches that notch under one rotation, the whole frame bbox becomes solid.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = g.copy()

    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        comps = _components(g == color)
        comp_info = []
        for comp in comps:
            r0, c0, r1, c1 = _bbox(comp)
            local = _normalize((r - r0, c - c0) for r, c in comp)
            comp_info.append(
                {
                    "coords": comp,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(comp),
                    "local": local,
                }
            )

        pair: tuple[tuple[int, int, int, int], list[Coord]] | None = None
        for i, frame in enumerate(comp_info):
            signature = _frame_signature(frame["coords"])
            if signature is None:
                continue
            missing = signature["missing"]
            assert isinstance(missing, tuple)
            for j, key in enumerate(comp_info):
                if i == j or int(key["area"]) != len(missing):
                    continue
                matches = [k for k in range(4) if _rotate_shape(key["local"], k) == missing]
                if len(matches) != 1:
                    continue
                candidate = (signature["bbox"], key["coords"])
                if pair is not None:
                    raise AssertionError(f"ambiguous frame-key pairing for color {color}")
                pair = candidate

        assert pair is not None, f"no valid frame-key pairing found for color {color}"
        (fr0, fc0, fr1, fc1), key_coords = pair
        for r, c in key_coords:
            out[r, c] = bg
        out[fr0 : fr1 + 1, fc0 : fc1 + 1] = color

    return out


def _corner_notch(h: int, w: int) -> tuple[tuple[Coord, ...], int]:
    size = random.randint(3, 5)
    splits = []
    for arm_h in range(2, size):
        arm_w = size + 1 - arm_h
        if arm_h >= 2 and arm_w >= 2 and arm_h <= h and arm_w <= w:
            splits.append((arm_h, arm_w))
    arm_h, arm_w = random.choice(splits)
    corner = random.randrange(4)

    if corner == 0:  # top-left
        coords = {(r, 0) for r in range(arm_h)} | {(0, c) for c in range(arm_w)}
    elif corner == 1:  # top-right
        coords = {(r, w - 1) for r in range(arm_h)} | {(0, w - 1 - c) for c in range(arm_w)}
    elif corner == 2:  # bottom-right
        coords = {(h - 1 - r, w - 1) for r in range(arm_h)} | {
            (h - 1, w - 1 - c) for c in range(arm_w)
        }
    else:  # bottom-left
        coords = {(h - 1 - r, 0) for r in range(arm_h)} | {(h - 1, c) for c in range(arm_w)}
    return tuple(sorted(coords)), corner


def _frame_patch(h: int, w: int, color: int, missing: Iterable[Coord]) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = color
    patch[h - 1, :] = color
    patch[:, 0] = color
    patch[:, w - 1] = color
    for r, c in missing:
        patch[r, c] = BG
    return patch


def _key_patch(color: int, missing: tuple[Coord, ...], rotation: int) -> np.ndarray:
    mask = _coords_to_mask(missing).astype(int) * color
    return np.rot90(mask, int(rotation))


def _sample_pair(color: int, num_pairs: int) -> dict[str, object]:
    if num_pairs == 1:
        h = random.randint(6, 10)
        w = random.randint(6, 10)
    elif num_pairs == 2:
        h = random.randint(5, 9)
        w = random.randint(5, 9)
    else:
        h = random.randint(5, 8)
        w = random.randint(5, 8)

    raw_missing, corner = _corner_notch(h, w)
    missing = _normalize(raw_missing)
    rotation = random.randrange(4)
    frame = _frame_patch(h, w, color, raw_missing)
    key = _key_patch(color, missing, rotation)

    matches = [k for k in range(4) if _rotate_shape(_normalize(np.argwhere(key == color)), k) == missing]
    assert len(matches) == 1

    return {
        "color": color,
        "corner": corner,
        "rotation": rotation,
        "missing": missing,
        "frame_patch": frame,
        "key_patch": key,
        "frame_shape": frame.shape,
        "key_shape": key.shape,
    }


def _sample_canvas(shapes: list[tuple[int, int]]) -> tuple[int, int]:
    need = sum((h + 2 * (OBJECT_MARGIN + OUTER_PAD)) * (w + 2 * (OBJECT_MARGIN + OUTER_PAD)) for h, w in shapes)
    max_h = max(h for h, _ in shapes) + 2 * OUTER_PAD
    max_w = max(w for _, w in shapes) + 2 * OUTER_PAD
    side = max(max_h, max_w, int(np.ceil(np.sqrt(need))))
    min_h = max(12, side - 1)
    min_w = max(12, side - 1)
    max_h_allowed = min(30, side + 5)
    max_w_allowed = min(30, side + 6)
    return random.randint(min_h, max_h_allowed), random.randint(min_w, max_w_allowed)


def _find_position(
    occupied: np.ndarray, patch_shape: tuple[int, int], margin: int = OBJECT_MARGIN, pad: int = OUTER_PAD
) -> tuple[int, int] | None:
    h, w = occupied.shape
    ph, pw = patch_shape
    positions: list[Coord] = []
    for r in range(pad, h - ph - pad + 1):
        for c in range(pad, w - pw - pad + 1):
            r0 = max(0, r - margin)
            c0 = max(0, c - margin)
            r1 = min(h, r + ph + margin)
            c1 = min(w, c + pw + margin)
            if not occupied[r0:r1, c0:c1].any():
                positions.append((r, c))
    if not positions:
        return None
    return random.choice(positions)


def _reserve(occupied: np.ndarray, top: int, left: int, patch_shape: tuple[int, int], margin: int = OBJECT_MARGIN) -> None:
    ph, pw = patch_shape
    r0 = max(0, top - margin)
    c0 = max(0, left - margin)
    r1 = min(occupied.shape[0], top + ph + margin)
    c1 = min(occupied.shape[1], left + pw + margin)
    occupied[r0:r1, c0:c1] = True


def _paste(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    ph, pw = patch.shape
    view = grid[top : top + ph, left : left + pw]
    nz = patch != BG
    view[nz] = patch[nz]


def _expected_output(inp: np.ndarray, records: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for rec in records:
        key_r, key_c = rec["key_pos"]
        key_patch = rec["key_patch"]
        key_view = out[key_r : key_r + key_patch.shape[0], key_c : key_c + key_patch.shape[1]]
        key_view[key_patch != BG] = BG

        frame_r, frame_c = rec["frame_pos"]
        fh, fw = rec["frame_shape"]
        out[frame_r : frame_r + fh, frame_c : frame_c + fw] = int(rec["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    num_pairs = random.randint(1, 4)
    colors = random.sample(list(COLORS), num_pairs)
    records = [_sample_pair(color, num_pairs) for color in colors]
    shapes = [rec["frame_shape"] for rec in records] + [rec["key_shape"] for rec in records]
    h, w = _sample_canvas(shapes)

    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    frame_order = records[:]
    key_order = records[:]
    random.shuffle(frame_order)
    random.shuffle(key_order)
    frame_order.sort(key=lambda rec: rec["frame_shape"][0] * rec["frame_shape"][1], reverse=True)
    key_order.sort(key=lambda rec: rec["key_shape"][0] * rec["key_shape"][1], reverse=True)

    for rec in frame_order:
        pos = _find_position(occupied, rec["frame_shape"])
        if pos is None:
            return None
        rec["frame_pos"] = pos
        _paste(grid, rec["frame_patch"], pos[0], pos[1])
        _reserve(occupied, pos[0], pos[1], rec["frame_shape"])

    for rec in key_order:
        pos = _find_position(occupied, rec["key_shape"])
        if pos is None:
            return None
        rec["key_pos"] = pos
        _paste(grid, rec["key_patch"], pos[0], pos[1])
        _reserve(occupied, pos[0], pos[1], rec["key_shape"])

    expected = _expected_output(grid, records)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
