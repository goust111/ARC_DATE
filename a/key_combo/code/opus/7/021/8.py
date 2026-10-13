from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
ANCHOR = 4
COLOR_POOL = [1, 2, 3, 6, 7, 8, 9]
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GRID_SIDE_MIN = 22
GRID_SIDE_MAX = 30
PLACEMENT_GAP = 1
MAX_GENERATION_ATTEMPTS = 400


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int, *, transparent: int = BG) -> None:
    h, w = patch.shape
    dst[top : top + h, left : left + w][patch != transparent] = patch[patch != transparent]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS8:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _scale_nn(arr: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(arr, np.ones((factor, factor), dtype=int))


def _rot_cw(arr: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(arr, -int(turns) % 4)


def _anchor_bbox(arr: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(arr == ANCHOR)
    if coords.size == 0:
        raise ValueError("missing anchor color")
    return _bbox_from_coords(coords)


def _anchor_top_left(arr: np.ndarray) -> tuple[int, int]:
    r0, c0, _r1, _c1 = _anchor_bbox(arr)
    return int(r0), int(c0)


def _extract_records(grid: np.ndarray) -> list[dict]:
    records = []
    for coords in _components(grid != BG):
        bbox = _bbox_from_coords(coords)
        patch = _crop(grid, bbox)
        records.append(
            {
                "coords": coords,
                "bbox": bbox,
                "patch": patch,
                "shape": patch.shape,
            }
        )
    records.sort(key=lambda rec: (rec["shape"][0] * rec["shape"][1], rec["bbox"][0], rec["bbox"][1]))
    return records


def _infer_prototype(records: list[dict]) -> tuple[np.ndarray, int]:
    if not records:
        raise ValueError("expected at least one sprite")
    min_side = min(int(rec["shape"][0]) for rec in records)
    if min_side not in (2, 3):
        raise ValueError("prototype side must be 2 or 3")
    candidates = [rec for rec in records if rec["shape"] == (min_side, min_side)]
    if not candidates:
        raise ValueError("missing unit-scale prototype")
    proto = candidates[0]["patch"]
    for rec in candidates[1:]:
        if not np.array_equal(rec["patch"], proto):
            raise ValueError("smallest sprites must share one prototype")
    if int(np.count_nonzero(proto == ANCHOR)) != 1:
        raise ValueError("prototype must contain exactly one anchor cell")
    return proto, min_side


def _analyze_grid(grid: np.ndarray) -> tuple[np.ndarray, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")
    records = _extract_records(g)
    proto, proto_side = _infer_prototype(records)
    for rec in records:
        h, w = rec["shape"]
        if h != w or h % proto_side != 0:
            raise ValueError("every sprite must be a square integer scale of the prototype")
        scale = h // proto_side
        if scale not in (1, 2, 3, 4):
            raise ValueError("scale factor must be in 1..4")
        expected = _scale_nn(proto, scale)
        if not np.array_equal(rec["patch"], expected):
            raise ValueError("sprite does not match the inferred scaled prototype")
        ar0, ac0, ar1, ac1 = _anchor_bbox(rec["patch"])
        if (ar1 - ar0 + 1, ac1 - ac0 + 1) != (scale, scale):
            raise ValueError("anchor region must scale to an SxS block")
        rec["scale"] = int(scale)
        rec["anchor_tl"] = (int(ar0), int(ac0))
    return proto, proto_side, records


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    proto, _proto_side, records = _analyze_grid(g)
    out = np.full_like(g, BG)
    for rec in records:
        turns = rec["scale"] % 4
        rotated = _rot_cw(rec["patch"], turns)
        orig_anchor = rec["anchor_tl"]
        rot_anchor = _anchor_top_left(rotated)
        top = rec["bbox"][0] + orig_anchor[0] - rot_anchor[0]
        left = rec["bbox"][1] + orig_anchor[1] - rot_anchor[1]
        if top < 0 or left < 0 or top + rotated.shape[0] > g.shape[0] or left + rotated.shape[1] > g.shape[1]:
            raise ValueError("transformed sprite would leave the canvas")
        target = out[top : top + rotated.shape[0], left : left + rotated.shape[1]]
        if np.any((target != BG) & (rotated != BG)):
            raise ValueError("transformed sprites must not overlap")
        _paste(out, rotated, top, left)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each scaled prototype copy by a scale-derived amount and then re-anchor its yellow block.

    Input structure:
    - The background is black, and the foreground consists of 3-5 non-overlapping square sprites.
    - One sprite is the 1x prototype: a 2x2 or 3x3 multicolor pattern with exactly one yellow anchor pixel.
    - Every other sprite is an integer nearest-neighbor scale copy of that same prototype, so its yellow anchor becomes an SxS block.

    Rule:
    - Infer the prototype from the smallest sprite and read each sprite's scale factor S from its side length.
    - Rotate each sprite clockwise by (S mod 4) quarter-turns.
    - After rotation, translate the whole sprite so the top-left corner of its yellow anchor block returns to the original anchor-block top-left coordinate, then redraw all transformed sprites on a black canvas of the same size.

    Key insight:
    - The scale factor is not preserved as geometry in the output; instead it becomes the hidden control signal for rotation, while the yellow anchor block fixes the final placement after that rotation.
    """
    return _solve(np.asarray(grid, dtype=int))


def _make_prototype() -> np.ndarray:
    while True:
        side = random.choice([2, 3])
        other_colors = random.sample(COLOR_POOL, random.choice([2, 3]))
        proto = np.empty((side, side), dtype=int)
        for r in range(side):
            for c in range(side):
                proto[r, c] = random.choice(other_colors)

        anchor_positions = [(r, c) for r in range(side) for c in range(side)]
        if side == 3:
            anchor_positions.remove((1, 1))
        ar, ac = random.choice(anchor_positions)
        proto[ar, ac] = ANCHOR

        visible_colors = set(map(int, proto.flatten())) - {ANCHOR}
        if len(visible_colors) < 2:
            continue
        if any(np.array_equal(proto, _rot_cw(proto, turns)) for turns in (1, 2, 3)):
            continue
        if any(_anchor_top_left(_rot_cw(proto, turns)) == (ar, ac) for turns in (1, 2, 3)):
            continue
        return proto


def _rect_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _candidate_positions(
    grid_shape: tuple[int, int],
    side: int,
    delta: tuple[int, int],
    placed_input: list[tuple[int, int, int, int]],
    placed_output: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    h, w = grid_shape
    dr, dc = delta
    candidates = []
    for top in range(h - side + 1):
        out_top = top + dr
        if out_top < 0 or out_top + side > h:
            continue
        for left in range(w - side + 1):
            out_left = left + dc
            if out_left < 0 or out_left + side > w:
                continue
            inbox = (top, left, top + side - 1, left + side - 1)
            outbox = (out_top, out_left, out_top + side - 1, out_left + side - 1)
            if any(_rect_overlaps(inbox, other, PLACEMENT_GAP) for other in placed_input):
                continue
            if any(_rect_overlaps(outbox, other, PLACEMENT_GAP) for other in placed_output):
                continue
            candidates.append((top, left))
    random.shuffle(candidates)
    return candidates


def _build_case() -> np.ndarray | None:
    proto = _make_prototype()
    proto_side = proto.shape[0]
    count = random.choice([4, 5])
    scales = [1, 2, 3, 4]
    if count == 5:
        scales.append(random.choice([1, 2, 3, 4]))
    random.shuffle(scales)

    side_budget = proto_side * max(scales)
    min_side = max(GRID_SIDE_MIN, side_budget + 8)
    if min_side > GRID_SIDE_MAX:
        return None
    h = random.randint(min_side, GRID_SIDE_MAX)
    w = random.randint(min_side, GRID_SIDE_MAX)

    specs = []
    for scale in scales:
        patch = _scale_nn(proto, scale)
        turns = scale % 4
        rotated = _rot_cw(patch, turns)
        orig_anchor = _anchor_top_left(patch)
        rot_anchor = _anchor_top_left(rotated)
        delta = (orig_anchor[0] - rot_anchor[0], orig_anchor[1] - rot_anchor[1])
        specs.append(
            {
                "scale": scale,
                "patch": patch,
                "rotated": rotated,
                "side": patch.shape[0],
                "delta": delta,
            }
        )

    specs.sort(key=lambda spec: (-spec["side"], spec["scale"]))
    placed_input: list[tuple[int, int, int, int]] = []
    placed_output: list[tuple[int, int, int, int]] = []
    placements = []

    for spec in specs:
        candidates = _candidate_positions((h, w), spec["side"], spec["delta"], placed_input, placed_output)
        if not candidates:
            return None
        top, left = candidates[0]
        dr, dc = spec["delta"]
        placements.append((spec, top, left))
        placed_input.append((top, left, top + spec["side"] - 1, left + spec["side"] - 1))
        placed_output.append(
            (
                top + dr,
                left + dc,
                top + dr + spec["side"] - 1,
                left + dc + spec["side"] - 1,
            )
        )

    grid = np.full((h, w), BG, dtype=int)
    for spec, top, left in placements:
        _paste(grid, spec["patch"], top, left)

    if not _semantic_check(grid, expected_scales=scales):
        return None
    return grid


def _uniform_rotation_output(grid: np.ndarray, turns: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _proto, _proto_side, records = _analyze_grid(g)
    out = np.full_like(g, BG)
    for rec in records:
        rotated = _rot_cw(rec["patch"], turns)
        orig_anchor = rec["anchor_tl"]
        rot_anchor = _anchor_top_left(rotated)
        top = rec["bbox"][0] + orig_anchor[0] - rot_anchor[0]
        left = rec["bbox"][1] + orig_anchor[1] - rot_anchor[1]
        if top < 0 or left < 0 or top + rotated.shape[0] > g.shape[0] or left + rotated.shape[1] > g.shape[1]:
            raise ValueError("uniform-rotation check went out of bounds")
        target = out[top : top + rotated.shape[0], left : left + rotated.shape[1]]
        if np.any((target != BG) & (rotated != BG)):
            raise ValueError("uniform-rotation check overlapped")
        _paste(out, rotated, top, left)
    return out


def _naive_in_place_rotation_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _proto, _proto_side, records = _analyze_grid(g)
    out = np.full_like(g, BG)
    for rec in records:
        rotated = _rot_cw(rec["patch"], rec["scale"] % 4)
        top, left, _r1, _c1 = rec["bbox"]
        target = out[top : top + rotated.shape[0], left : left + rotated.shape[1]]
        if np.any((target != BG) & (rotated != BG)):
            return out
        _paste(out, rotated, top, left)
    return out


def _semantic_check(grid: np.ndarray, *, expected_scales: Iterable[int] | None = None) -> bool:
    try:
        proto, proto_side, records = _analyze_grid(grid)
        if len(records) not in (4, 5):
            return False
        scales = [int(rec["scale"]) for rec in records]
        if expected_scales is not None and sorted(scales) != sorted(map(int, expected_scales)):
            return False
        if sorted(set(scales)) != [1, 2, 3, 4]:
            return False

        for turns in (1, 2, 3):
            if np.array_equal(proto, _rot_cw(proto, turns)):
                return False

        rotated_with_motion = 0
        for rec in records:
            turns = rec["scale"] % 4
            rotated = _rot_cw(rec["patch"], turns)
            if not np.array_equal(rotated, rec["patch"]):
                rotated_with_motion += 1
            if turns != 0 and _anchor_top_left(rotated) == rec["anchor_tl"]:
                return False

        if rotated_with_motion < 2:
            return False

        solved = main(grid)
        if np.array_equal(solved, grid):
            return False
        if np.array_equal(solved, _naive_in_place_rotation_output(grid)):
            return False
        for turns in range(4):
            if np.array_equal(solved, _uniform_rotation_output(grid, turns)):
                return False

        solved_records = _extract_records(solved)
        if len(solved_records) != len(records):
            return False
        if proto_side * 4 > max(grid.shape):
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
