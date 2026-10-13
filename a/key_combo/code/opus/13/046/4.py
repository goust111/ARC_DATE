from __future__ import annotations

import random
from typing import Any

import numpy as np


BG = 0
SPRITE_COLOR = 3
FRAME_COLOR = 5
KEY_PATTERN = np.array([[2, 1], [4, 7]], dtype=int)
ANGLE_TO_KEY_INDEX = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _extract_components(mask: np.ndarray) -> list[dict[str, Any]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, Any]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            sub[coords[:, 0] - r0, coords[:, 1] - c0] = True
            comps.append(
                {
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": sub,
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _rotate_cw(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(mask, -int(turns) % 4)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_mask_key(_rotate_cw(mask, k)) for k in range(4))


def _all_rotations_distinct(mask: np.ndarray) -> bool:
    return len({_mask_key(_rotate_cw(mask, k)) for k in range(4)}) == 4


def _solve_instance(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[dict[str, Any]], list[dict[str, Any]], dict[int, tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    frame_comps = _extract_components(g == FRAME_COLOR)
    assert frame_comps, "expected a grey frame"
    max_area = max(comp["area"] for comp in frame_comps)
    frame_candidates = [comp for comp in frame_comps if comp["area"] == max_area]
    assert len(frame_candidates) == 1, "expected one largest grey frame"
    frame = frame_candidates[0]
    fr0, fc0, fr1, fc1 = frame["bbox"]

    frame_box = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    assert np.all(frame_box[0, :] == FRAME_COLOR)
    assert np.all(frame_box[-1, :] == FRAME_COLOR)
    assert np.all(frame_box[:, 0] == FRAME_COLOR)
    assert np.all(frame_box[:, -1] == FRAME_COLOR)

    holes = _extract_components(frame_box == BG)
    assert len(holes) == 4, "expected four holes inside the frame"
    for hole in holes:
        hr0, hc0, hr1, hc1 = hole["bbox"]
        assert 0 < hr0 and 0 < hc0 and hr1 < frame_box.shape[0] - 1 and hc1 < frame_box.shape[1] - 1

    sprites = _extract_components(g == SPRITE_COLOR)
    assert len(sprites) == 4, "expected four green sprites"
    for sprite in sprites:
        sr0, sc0, sr1, sc1 = sprite["bbox"]
        assert sr1 < fr0 or sr0 > fr1 or sc1 < fc0 or sc0 > fc1, "sprites must be outside the frame"
        assert _all_rotations_distinct(sprite["mask"]), "sprite must have no rotational symmetry"

    hole_candidates: list[list[tuple[int, int]]] = []
    for hole in holes:
        candidates: list[tuple[int, int]] = []
        for si, sprite in enumerate(sprites):
            for angle in range(4):
                if np.array_equal(_rotate_cw(sprite["mask"], angle), hole["mask"]):
                    candidates.append((si, angle))
        assert candidates, "each hole must match at least one sprite"
        hole_candidates.append(candidates)

    order = sorted(range(len(holes)), key=lambda idx: len(hole_candidates[idx]))
    solutions: list[dict[int, tuple[int, int]]] = []

    def backtrack(pos: int, used_sprites: set[int], cur: dict[int, tuple[int, int]]) -> None:
        if len(solutions) > 1:
            return
        if pos == len(order):
            solutions.append(cur.copy())
            return
        hi = order[pos]
        for si, angle in hole_candidates[hi]:
            if si in used_sprites:
                continue
            used_sprites.add(si)
            cur[hi] = (si, angle)
            backtrack(pos + 1, used_sprites, cur)
            used_sprites.remove(si)
            cur.pop(hi)

    backtrack(0, set(), {})
    assert len(solutions) == 1, "expected a unique sprite-hole matching"
    assignment = solutions[0]
    assert {angle for _si, angle in assignment.values()} == {0, 1, 2, 3}, "all four rotations must be used exactly once"
    return (fr0, fc0, fr1, fc1), holes, sprites, assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each frame hole with the color encoded by the rotation needed to match a green outside sprite.

    Input structure:
    - Black background with a fixed 2x2 color key in the upper-left corner.
    - One large grey rectangular frame contains exactly four black holes.
    - Outside the frame are exactly four green 4-connected sprites, all the same color and all rotation-asymmetric.

    Rule:
    - For each hole, find the unique green sprite that matches it under rotation only (0, 90, 180, 270 degrees clockwise; no flips).
    - Read the 2x2 key as rotation-to-color: upper-left for 0 degrees, upper-right for 90 degrees, lower-left for 180 degrees, lower-right for 270 degrees.
    - Fill every pixel of that hole with the corresponding key color, erase the outside green sprites to black, and leave the grey frame plus the key unchanged.

    Key insight:
    - The sprites' visible color carries no information; the information is the rotation amount required for exact shape matching, and that geometric amount must be decoded through the key block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 2 and g.shape[1] >= 2, "grid must contain the 2x2 key"

    key = g[:2, :2]
    frame_bbox, holes, _sprites, assignment = _solve_instance(g)
    fr0, fc0, _fr1, _fc1 = frame_bbox
    out = g.copy()
    out[out == SPRITE_COLOR] = BG
    for hi, hole in enumerate(holes):
        _si, angle = assignment[hi]
        kr, kc = ANGLE_TO_KEY_INDEX[angle]
        fill = int(key[kr, kc])
        abs_coords = hole["coords"] + np.array([fr0, fc0])
        out[abs_coords[:, 0], abs_coords[:, 1]] = fill
    return out


def _connected(mask: np.ndarray) -> bool:
    comps = _extract_components(mask)
    return len(comps) == 1


_VALID_SHAPES_CACHE: dict[tuple[int, int], list[np.ndarray]] = {}


def _valid_shapes_for_dims(h: int, w: int) -> list[np.ndarray]:
    key = (h, w)
    if key in _VALID_SHAPES_CACHE:
        return _VALID_SHAPES_CACHE[key]

    valid: list[np.ndarray] = []
    for bits in range(1 << (h * w)):
        area = bits.bit_count()
        if area < 4 or area > 6:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for idx in range(h * w):
            if bits & (1 << idx):
                mask[idx // w, idx % w] = True
        coords = np.argwhere(mask)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if (int(r0), int(c0), int(r1), int(c1)) != (0, 0, h - 1, w - 1):
            continue
        if not _connected(mask):
            continue
        if not _all_rotations_distinct(mask):
            continue
        valid.append(mask)
    _VALID_SHAPES_CACHE[key] = valid
    return valid


def _sample_shape(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    valid = _valid_shapes_for_dims(h, w)
    if not valid:
        raise RuntimeError(f"no valid shapes for {h}x{w}")
    return valid[int(rng.integers(len(valid)))].copy()


def _choose_shapes(rng: np.random.Generator) -> list[dict[str, Any]]:
    dims = [(3, 4)] * 4
    shapes: list[dict[str, Any]] = []
    seen = set()
    for h, w in dims:
        while True:
            mask = _sample_shape(rng, h, w)
            key = _canonical_key(mask)
            if key in seen:
                continue
            seen.add(key)
            shapes.append({"base_dims": (h, w), "mask": mask, "canon": key})
            break
    return shapes


def _assign_rotations(rng: np.random.Generator, shapes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    angles = [0, 1, 2, 3]
    rng.shuffle(angles)

    out: list[dict[str, Any]] = []
    hole_keys = set()
    for item, angle in zip(shapes, angles):
        hole = _rotate_cw(item["mask"], angle)
        hole_keys.add(_mask_key(hole))
        out.append(
            {
                "sprite_mask": item["mask"],
                "hole_mask": hole,
                "angle": angle,
                "canon": item["canon"],
            }
        )
    assert len(hole_keys) == 4
    return out


def _build_frame(rng: np.random.Generator, records: list[dict[str, Any]]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    slots = [0, 1, 2, 3]
    rng.shuffle(slots)
    placed = [records[i] for i in slots]
    tl, tr, bl, br = placed

    tl_h, tl_w = tl["hole_mask"].shape
    tr_h, tr_w = tr["hole_mask"].shape
    bl_h, bl_w = bl["hole_mask"].shape
    br_h, br_w = br["hole_mask"].shape

    cell_h_top = max(tl_h, tr_h) + int(rng.integers(0, 3))
    cell_h_bottom = max(bl_h, br_h) + int(rng.integers(0, 3))
    cell_w_left = max(tl_w, bl_w) + int(rng.integers(0, 3))
    cell_w_right = max(tr_w, br_w) + int(rng.integers(0, 3))

    pad_top = int(rng.integers(1, 3))
    pad_bottom = int(rng.integers(1, 3))
    pad_left = int(rng.integers(1, 3))
    pad_right = int(rng.integers(1, 3))
    gap_h = int(rng.integers(1, 3))
    gap_w = int(rng.integers(1, 3))

    frame_h = pad_top + cell_h_top + gap_h + cell_h_bottom + pad_bottom
    frame_w = pad_left + cell_w_left + gap_w + cell_w_right + pad_right
    frame = np.full((frame_h, frame_w), FRAME_COLOR, dtype=int)

    quadrant_origins = [
        (pad_top, pad_left, cell_h_top, cell_w_left, tl),
        (pad_top, pad_left + cell_w_left + gap_w, cell_h_top, cell_w_right, tr),
        (pad_top + cell_h_top + gap_h, pad_left, cell_h_bottom, cell_w_left, bl),
        (pad_top + cell_h_top + gap_h, pad_left + cell_w_left + gap_w, cell_h_bottom, cell_w_right, br),
    ]

    top_lefts: list[tuple[int, int]] = []
    for r0, c0, cell_h, cell_w, rec in quadrant_origins:
        hole = rec["hole_mask"]
        hh, hw = hole.shape
        off_r = int(rng.integers(0, cell_h - hh + 1))
        off_c = int(rng.integers(0, cell_w - hw + 1))
        pr = r0 + off_r
        pc = c0 + off_c
        frame[pr : pr + hh, pc : pc + hw][hole] = BG
        rec["hole_top_left"] = (pr, pc)
        top_lefts.append((pr, pc))

    holes = _extract_components(frame == BG)
    assert len(holes) == 4
    return frame, top_lefts


def _place_sprites(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], records: list[dict[str, Any]]) -> None:
    fr0, fc0, fr1, fc1 = frame_bbox
    order = sorted(range(4), key=lambda idx: records[idx]["canon"])
    zones = ["top", "left", "right", "bottom"]
    for zone, idx in zip(zones, order):
        mask = records[idx]["sprite_mask"]
        h, w = mask.shape
        if zone == "top":
            r0 = 0
            c0 = grid.shape[1] - w
        elif zone == "left":
            r0 = fr0 + (fr1 - fr0 + 1 - h) // 2
            c0 = 0
        elif zone == "right":
            r0 = fr0 + (fr1 - fr0 + 1 - h) // 2
            c0 = grid.shape[1] - w
        else:
            r0 = grid.shape[0] - h
            c0 = (grid.shape[1] - w) // 2
        patch = np.zeros((h, w), dtype=int)
        patch[mask] = SPRITE_COLOR
        region = grid[r0 : r0 + h, c0 : c0 + w]
        assert region.shape == patch.shape
        assert np.all(region == BG)
        region[mask] = SPRITE_COLOR
        records[idx]["sprite_top_left"] = (r0, c0)


def _expected_output(inp: np.ndarray, frame_bbox: tuple[int, int, int, int], records: list[dict[str, Any]]) -> np.ndarray:
    fr0, fc0, _fr1, _fc1 = frame_bbox
    out = inp.copy()
    out[out == SPRITE_COLOR] = BG
    for rec in records:
        pr, pc = rec["hole_top_left"]
        hr, hc = ANGLE_TO_KEY_INDEX[rec["angle"]]
        color = int(KEY_PATTERN[hr, hc])
        abs_r = fr0 + pr
        abs_c = fc0 + pc
        hole = rec["hole_mask"]
        out[abs_r : abs_r + hole.shape[0], abs_c : abs_c + hole.shape[1]][hole] = color
    return out


def _semantic_checks(inp: np.ndarray, records: list[dict[str, Any]], expected: np.ndarray) -> bool:
    try:
        solved = main(inp)
        if not np.array_equal(solved, expected):
            return False
        frame_bbox, holes, sprites, assignment = _solve_instance(inp)
    except Exception:
        return False

    if len(holes) != 4 or len(sprites) != 4:
        return False
    if {angle for _si, angle in assignment.values()} != {0, 1, 2, 3}:
        return False

    hole_keys = {_mask_key(hole["mask"]) for hole in holes}
    if len(hole_keys) != 4:
        return False

    sprite_canon = {_canonical_key(sprite["mask"]) for sprite in sprites}
    if len(sprite_canon) != 4:
        return False

    fr0, fc0, fr1, fc1 = frame_bbox
    green_coords = np.argwhere(inp == SPRITE_COLOR)
    for r, c in green_coords:
        if fr0 <= r <= fr1 and fc0 <= c <= fc1:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    records = _assign_rotations(rng, _choose_shapes(rng))
    frame, _hole_positions = _build_frame(rng, records)

    top_margin = int(rng.integers(5, 7))
    left_margin = int(rng.integers(5, 7))
    right_margin = int(rng.integers(5, 7))
    bottom_margin = int(rng.integers(5, 7))

    grid_h = top_margin + frame.shape[0] + bottom_margin
    grid_w = left_margin + frame.shape[1] + right_margin
    if max(grid_h, grid_w) > 30:
        return None

    fr0, fc0 = top_margin, left_margin
    fr1, fc1 = fr0 + frame.shape[0] - 1, fc0 + frame.shape[1] - 1
    inp = np.zeros((grid_h, grid_w), dtype=int)
    inp[:2, :2] = KEY_PATTERN
    inp[fr0 : fr1 + 1, fc0 : fc1 + 1] = frame

    try:
        _place_sprites(inp, (fr0, fc0, fr1, fc1), records)
    except AssertionError:
        return None

    expected = _expected_output(inp, (fr0, fc0, fr1, fc1), records)
    if not _semantic_checks(inp, records, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotation-coded inlay task")
