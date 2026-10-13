from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDE_TO_TURNS_CW = {"top": 0, "right": 1, "bottom": 2, "left": 3}
ALL_SIDES = tuple(SIDE_TO_TURNS_CW)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    box = _bbox(coords)
    r0, c0, r1, c1 = box
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out, box


def _rotate_cw(arr: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(arr, -int(turns) % 4)


def _paste_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    grid[int(r0) : int(r0) + h, int(c0) : int(c0) + w][mask] = int(color)


def _contiguous_false_run(line: np.ndarray) -> int | None:
    miss = np.flatnonzero(~line)
    if miss.size == 0:
        return 0
    if not np.array_equal(miss, np.arange(int(miss[0]), int(miss[-1]) + 1)):
        return None
    return int(miss.size)


def _frame_gap_side(mask: np.ndarray) -> str | None:
    h, w = mask.shape
    if h < 3 or w < 3:
        return None
    if np.any(mask[1:-1, 1:-1]):
        return None
    if not (bool(mask[0, 0]) and bool(mask[0, -1]) and bool(mask[-1, 0]) and bool(mask[-1, -1])):
        return None

    side_lines = {
        "top": mask[0, 1:-1],
        "right": mask[1:-1, -1],
        "bottom": mask[-1, 1:-1],
        "left": mask[1:-1, 0],
    }
    gap_side = None
    gap_len = 0
    for side, line in side_lines.items():
        run = _contiguous_false_run(line)
        if run is None:
            return None
        if run == 0:
            continue
        if gap_side is not None:
            return None
        gap_side = side
        gap_len = int(run)

    if gap_side is None:
        return None

    expected = 2 * (h + w) - 4 - gap_len
    if int(mask.sum()) != int(expected):
        return None
    return gap_side


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each scattered fragment to the frame of the same color, rotate it using the
    frame-gap direction, and place it into the frame's interior with top-left alignment.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    for color in colors:
        comps = _components(g == int(color))
        if not comps:
            continue

        frame_idx = None
        frame_side = None
        frame_box = None
        for idx, coords in enumerate(comps):
            mask, box = _tight_mask(coords)
            side = _frame_gap_side(mask)
            if side is None:
                continue
            assert frame_idx is None, "each color should have exactly one frame"
            frame_idx = int(idx)
            frame_side = side
            frame_box = box

        assert frame_idx is not None and frame_side is not None and frame_box is not None
        piece_indices = [idx for idx in range(len(comps)) if idx != frame_idx]
        assert len(piece_indices) == 1, "each frame should have exactly one matching fragment"

        piece_coords = comps[piece_indices[0]]
        piece_mask, _piece_box = _tight_mask(piece_coords)
        turns = int(SIDE_TO_TURNS_CW[frame_side])
        rotated = _rotate_cw(piece_mask, turns)

        out[piece_coords[:, 0], piece_coords[:, 1]] = BG
        inner_r = int(frame_box[0]) + 1
        inner_c = int(frame_box[1]) + 1
        inner_h = int(frame_box[2] - frame_box[0] - 1)
        inner_w = int(frame_box[3] - frame_box[1] - 1)
        rh, rw = rotated.shape
        assert rh <= inner_h and rw <= inner_w, "rotated fragment must fit inside the frame"
        _paste_mask(out, rotated, inner_r, inner_c, color)

    return out


# --------------- generator ---------------

COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
OBJECT_COUNT_RANGE = (2, 4)
INTERIOR_SIDE_RANGE = (4, 6)
CANVAS_SIDE_RANGE = (20, 30)
PLACEMENT_PADDING = 1
SHAPE_TRIES = 200
CASE_TRIES = 300


def _random_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    cells = [(int(r), int(c))]
    while (r, c) != b:
        options = []
        if r < b[0]:
            options.append((r + 1, c))
        elif r > b[0]:
            options.append((r - 1, c))
        if c < b[1]:
            options.append((r, c + 1))
        elif c > b[1]:
            options.append((r, c - 1))
        r, c = random.choice(options)
        cells.append((int(r), int(c)))
    return cells


def _rotation_count(mask: np.ndarray) -> int:
    seen = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        seen.add((tuple(map(int, rot.shape)), rot.tobytes()))
    return len(seen)


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _grow_shape(h: int, w: int) -> np.ndarray | None:
    for _ in range(SHAPE_TRIES):
        anchor = (random.randrange(h), random.randrange(w))
        endpoints = [
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        ]

        cells = {anchor}
        for endpoint in endpoints:
            cells.update(_random_path(anchor, endpoint))

        min_area = len(cells)
        max_area = min(int(np.floor(0.72 * h * w)), h * w - 1)
        min_target = max(min_area, int(np.ceil(0.38 * h * w)))
        if min_target > max_area:
            continue

        target_area = random.randint(min_target, max_area)
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[int(r), int(c)] = True

        frontier = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                    frontier.add((int(nr), int(nc)))

        while int(mask.sum()) < target_area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            r, c = cell
            mask[int(r), int(c)] = True
            for dr, dc in DIRS4:
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                    frontier.add((int(nr), int(nc)))

        if int(mask.sum()) != target_area:
            continue
        if not _touches_all_sides(mask):
            continue
        if _rotation_count(mask) != 4:
            continue
        if _frame_gap_side(mask) is not None:
            continue
        return mask
    return None


def _frame_sprite(color: int, inner_h: int, inner_w: int, side: str) -> np.ndarray:
    sprite = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    sprite[0, :] = int(color)
    sprite[-1, :] = int(color)
    sprite[:, 0] = int(color)
    sprite[:, -1] = int(color)

    if side in {"top", "bottom"}:
        gap_len = random.randint(1, inner_w - 1)
        start = random.randint(1, inner_w - gap_len + 1)
        row = 0 if side == "top" else inner_h + 1
        sprite[row, start : start + gap_len] = BG
    else:
        gap_len = random.randint(1, inner_h - 1)
        start = random.randint(1, inner_h - gap_len + 1)
        col = 0 if side == "left" else inner_w + 1
        sprite[start : start + gap_len, col] = BG

    return sprite


def _boxes_separate(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int
) -> bool:
    return (
        int(a[2]) + padding < int(b[0])
        or int(b[2]) + padding < int(a[0])
        or int(a[3]) + padding < int(b[1])
        or int(b[3]) + padding < int(a[1])
    )


def _find_position(
    h: int,
    w: int,
    sprite_h: int,
    sprite_w: int,
    boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    positions = [(r, c) for r in range(h - sprite_h + 1) for c in range(w - sprite_w + 1)]
    random.shuffle(positions)
    for r0, c0 in positions:
        box = (int(r0), int(c0), int(r0 + sprite_h - 1), int(c0 + sprite_w - 1))
        if all(_boxes_separate(box, other, PLACEMENT_PADDING) for other in boxes):
            return int(r0), int(c0)
    return None


def _paste_sprite(grid: np.ndarray, sprite: np.ndarray, r0: int, c0: int) -> None:
    h, w = sprite.shape
    patch = grid[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    mask = sprite != BG
    patch[mask] = sprite[mask]


def _make_case() -> np.ndarray | None:
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(COLOR_POOL, n_objects)
    sides = random.sample(list(ALL_SIDES), n_objects)

    specs: list[dict[str, object]] = []
    for color, side in zip(colors, sides, strict=False):
        inner_h = random.randint(*INTERIOR_SIDE_RANGE)
        inner_w = random.randint(*INTERIOR_SIDE_RANGE)
        base_mask = _grow_shape(inner_h, inner_w)
        if base_mask is None:
            return None

        turns = int(SIDE_TO_TURNS_CW[str(side)])
        scattered_mask = _rotate_cw(base_mask, (-turns) % 4)
        frame = _frame_sprite(int(color), inner_h, inner_w, str(side))
        piece = np.where(scattered_mask, int(color), BG)
        specs.append(
            {
                "color": int(color),
                "side": str(side),
                "base_mask": base_mask,
                "frame": frame,
                "piece": piece,
            }
        )

    total_box_area = sum(
        int(spec["frame"].shape[0] * spec["frame"].shape[1] + spec["piece"].shape[0] * spec["piece"].shape[1])
        for spec in specs
    )
    min_side = max(CANVAS_SIDE_RANGE[0], int(np.ceil(np.sqrt(total_box_area * 1.7))))
    if min_side > CANVAS_SIDE_RANGE[1]:
        return None

    h = random.randint(min_side, CANVAS_SIDE_RANGE[1])
    w = random.randint(min_side, CANVAS_SIDE_RANGE[1])
    items = []
    for idx, spec in enumerate(specs):
        items.append(("frame", idx, spec["frame"]))
        items.append(("piece", idx, spec["piece"]))
    items.sort(key=lambda item: int(item[2].shape[0] * item[2].shape[1]), reverse=True)

    boxes: list[tuple[int, int, int, int]] = []
    positions: dict[tuple[str, int], tuple[int, int]] = {}
    for kind, idx, sprite in items:
        pos = _find_position(h, w, int(sprite.shape[0]), int(sprite.shape[1]), boxes)
        if pos is None:
            return None
        r0, c0 = pos
        positions[(str(kind), int(idx))] = (int(r0), int(c0))
        boxes.append((int(r0), int(c0), int(r0 + sprite.shape[0] - 1), int(c0 + sprite.shape[1] - 1)))

    grid = np.zeros((h, w), dtype=int)
    for kind, idx, sprite in items:
        r0, c0 = positions[(str(kind), int(idx))]
        _paste_sprite(grid, sprite, r0, c0)

    expected = grid.copy()
    for idx, spec in enumerate(specs):
        piece = spec["piece"]
        pr, pc = positions[("piece", int(idx))]
        ph, pw = piece.shape
        piece_mask = piece != BG
        expected[pr : pr + ph, pc : pc + pw][piece_mask] = BG

        frame = spec["frame"]
        fr, fc = positions[("frame", int(idx))]
        inner_r = int(fr) + 1
        inner_c = int(fc) + 1
        _paste_mask(expected, spec["base_mask"], inner_r, inner_c, int(spec["color"]))

        frame_coords = _components(frame == int(spec["color"]))
        assert len(frame_coords) == 1
        frame_mask, _ = _tight_mask(frame_coords[0])
        assert _frame_gap_side(frame_mask) == spec["side"]

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(CASE_TRIES):
        grid = _make_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
