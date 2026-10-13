from __future__ import annotations

import numpy as np


UP, RIGHT, DOWN, LEFT = range(4)
DIR_TO_DELTA = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR_TO_DELTA.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_patch(grid: np.ndarray, mask: np.ndarray, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _mask_bbox(mask)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    submask = mask[r0 : r1 + 1, c0 : c1 + 1]
    subgrid = grid[r0 : r1 + 1, c0 : c1 + 1]
    patch[submask] = subgrid[submask]
    return patch


def _find_fence(grid: np.ndarray, bg: int) -> dict:
    candidates: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        if len(coords) < 12:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if min(h, w) < 5:
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        color_mask = sub == color
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        if np.any(color_mask & ~border):
            continue

        missing = border & ~color_mask
        if not missing.any():
            continue

        side = None
        start = end = None
        if missing[0, :].any() and not missing[1:, :].any():
            cols = np.where(missing[0, :])[0]
            side = UP
            start, end = int(cols[0]), int(cols[-1])
        elif missing[-1, :].any() and not missing[:-1, :].any():
            cols = np.where(missing[-1, :])[0]
            side = DOWN
            start, end = int(cols[0]), int(cols[-1])
        elif missing[:, 0].any() and not missing[:, 1:].any():
            rows = np.where(missing[:, 0])[0]
            side = LEFT
            start, end = int(rows[0]), int(rows[-1])
        elif missing[:, -1].any() and not missing[:, :-1].any():
            rows = np.where(missing[:, -1])[0]
            side = RIGHT
            start, end = int(rows[0]), int(rows[-1])
        if side is None:
            continue
        if end < start:
            continue
        if side in (UP, DOWN):
            if start == 0 or end == w - 1:
                continue
            gap_len = end - start + 1
            gap = {"side": side, "start": int(c0 + start), "end": int(c0 + end)}
        else:
            if start == 0 or end == h - 1:
                continue
            gap_len = end - start + 1
            gap = {"side": side, "start": int(r0 + start), "end": int(r0 + end)}
        if gap_len not in (2, 3):
            continue

        candidates.append(
            {
                "color": color,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "gap": gap,
                "score": h * w,
            }
        )

    assert candidates, "expected one rectangular fence with one gap"
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[0]


def _extract_object_and_push(grid: np.ndarray, bg: int, fence_color: int) -> dict:
    mask = (grid != bg) & (grid != fence_color)
    coords = np.argwhere(mask)
    assert len(coords) >= 4, "expected one object plus one push pixel"

    candidates: list[dict] = []
    for pr, pc in coords:
        reduced = mask.copy()
        reduced[int(pr), int(pc)] = False
        reduced_coords = np.argwhere(reduced)
        if reduced_coords.size == 0:
            continue
        r0, c0 = reduced_coords.min(axis=0)
        r1, c1 = reduced_coords.max(axis=0)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h != w or h not in (2, 3):
            continue
        submask = reduced[r0 : r1 + 1, c0 : c1 + 1]
        if not _is_connected(submask):
            continue

        direction = None
        pr = int(pr)
        pc = int(pc)
        if pr == int(r0) - 1 and int(c0) <= pc <= int(c1):
            direction = DOWN
        elif pr == int(r1) + 1 and int(c0) <= pc <= int(c1):
            direction = UP
        elif pc == int(c0) - 1 and int(r0) <= pr <= int(r1):
            direction = RIGHT
        elif pc == int(c1) + 1 and int(r0) <= pr <= int(r1):
            direction = LEFT
        if direction is None:
            continue

        if not any(
            0 <= pr + dr < grid.shape[0]
            and 0 <= pc + dc < grid.shape[1]
            and reduced[pr + dr, pc + dc]
            for dr, dc in DIR_TO_DELTA.values()
        ):
            continue

        object_mask = np.zeros_like(mask, dtype=bool)
        object_mask[r0 : r1 + 1, c0 : c1 + 1] = submask
        patch = _crop_patch(grid, object_mask, bg)
        candidates.append(
            {
                "patch": patch,
                "top": int(r0),
                "left": int(c0),
                "push": (pr, pc),
                "direction": direction,
            }
        )

    assert len(candidates) == 1, "push pixel should be uniquely identifiable"
    return candidates[0]


def _matches_gap(gap: dict, direction: int, top: int, left: int, h: int, w: int) -> bool:
    if gap["side"] != direction:
        return False
    if direction in (UP, DOWN):
        return gap["start"] == left and gap["end"] == left + w - 1
    return gap["start"] == top and gap["end"] == top + h - 1


def _wall_contact(
    fence_bbox: tuple[int, int, int, int],
    direction: int,
    top: int,
    left: int,
    h: int,
    w: int,
) -> tuple[int, int]:
    r0, c0, r1, c1 = fence_bbox
    if direction == UP:
        return int(r0 + 1), left
    if direction == DOWN:
        return int(r1 - h), left
    if direction == LEFT:
        return top, int(c0 + 1)
    return top, int(c1 - w)


def _outside_position(
    fence_bbox: tuple[int, int, int, int],
    direction: int,
    top: int,
    left: int,
    h: int,
    w: int,
) -> tuple[int, int]:
    r0, c0, r1, c1 = fence_bbox
    if direction == UP:
        return int(r0 - h), left
    if direction == DOWN:
        return int(r1 + 1), left
    if direction == LEFT:
        return top, int(c0 - w)
    return top, int(c1 + 1)


def _simulate_exit(
    fence_bbox: tuple[int, int, int, int],
    gap: dict,
    patch: np.ndarray,
    top: int,
    left: int,
    direction: int,
    bg: int,
    max_steps: int = 20,
) -> tuple[np.ndarray, int, int, int]:
    cur_patch = np.array(patch, copy=True)
    cur_top = int(top)
    cur_left = int(left)
    cur_dir = int(direction)
    seen = set()

    for stage in range(max_steps):
        h, w = map(int, cur_patch.shape)
        cur_top, cur_left = _wall_contact(fence_bbox, cur_dir, cur_top, cur_left, h, w)
        if _matches_gap(gap, cur_dir, cur_top, cur_left, h, w):
            out_top, out_left = _outside_position(fence_bbox, cur_dir, cur_top, cur_left, h, w)
            return cur_patch, int(out_top), int(out_left), stage

        state = (cur_dir, cur_top, cur_left, cur_patch.shape, cur_patch.tobytes())
        if state in seen:
            break
        seen.add(state)
        cur_patch = np.rot90(cur_patch, -1)
        cur_dir = (cur_dir + 1) % 4

    raise AssertionError("object never reaches the exit")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a pushed inner sprite around a fenced rectangle until it rotates out through the only gap.

    Input structure:
    - The background is the most common color.
    - One large monochrome rectangular fence has a one-cell-thick border and a single contiguous gap on one side.
    - Inside the fence there is one small 2x2 or 3x3 non-symmetric sprite plus one adjacent single-pixel push marker.

    Rule:
    - The push marker sets the initial motion direction: left marker pushes right, right pushes left, top pushes down, bottom pushes up.
    - Move the sprite straight until it reaches the inner wall position for that direction.
    - If that wall position aligns the sprite with the fence gap, place the sprite just outside the gap and keep its current rotation.
    - Otherwise rotate the sprite 90 degrees clockwise, turn the motion direction clockwise, and repeat; erase the push marker and leave the fence unchanged.

    Key insight:
    - The task is a deterministic wall-contact simulation: each failed wall contact adds one clockwise rotation, so the exit orientation is determined by how many contacts happen before the gap lines up.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    fence = _find_fence(g, bg)
    moving = _extract_object_and_push(g, bg, fence["color"])

    exited_patch, out_top, out_left, _stage = _simulate_exit(
        fence["bbox"],
        fence["gap"],
        moving["patch"],
        moving["top"],
        moving["left"],
        moving["direction"],
        bg,
    )

    out = g.copy()
    out[(g != bg) & (g != fence["color"])] = bg
    h, w = exited_patch.shape
    sub = out[out_top : out_top + h, out_left : out_left + w]
    mask = exited_patch != bg
    sub[mask] = exited_patch[mask]
    return out


def _shape_library(size: int) -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    total = size * size
    for bits in range(1 << total):
        mask = np.zeros((size, size), dtype=bool)
        for idx in range(total):
            if bits & (1 << idx):
                mask[idx // size, idx % size] = True
        area = int(mask.sum())
        if area < 3 or area >= total:
            continue
        if not _is_connected(mask):
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        rotations = [np.rot90(mask, k) for k in range(4)]
        if any(np.array_equal(mask, rot) for rot in rotations[1:]):
            continue
        shapes.append(mask)
    assert shapes
    return shapes


_SHAPES = {2: _shape_library(2), 3: _shape_library(3)}
_RNG = np.random.default_rng()
_PALETTE = list(range(10))
_MAX_ATTEMPTS = 500


def _paint_object(mask: np.ndarray, bg: int, colors: list[int]) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    _RNG.shuffle(cells)
    n_colors = int(_RNG.integers(1, min(3, len(cells), len(colors)) + 1))
    chosen = [int(c) for c in _RNG.choice(colors, size=n_colors, replace=False)]
    splits = np.array_split(np.arange(len(cells)), n_colors)
    for color, idxs in zip(chosen, splits):
        for idx in idxs:
            r, c = cells[int(idx)]
            patch[r, c] = color
    return patch


def _side_cells(mask: np.ndarray, side: int) -> list[tuple[int, int]]:
    if side == UP:
        return [(0, int(c)) for c in np.where(mask[0, :])[0]]
    if side == DOWN:
        row = mask.shape[0] - 1
        return [(row, int(c)) for c in np.where(mask[row, :])[0]]
    if side == LEFT:
        return [(int(r), 0) for r in np.where(mask[:, 0])[0]]
    col = mask.shape[1] - 1
    return [(int(r), col) for r in np.where(mask[:, col])[0]]


def _contact_states(
    fence_bbox: tuple[int, int, int, int],
    patch: np.ndarray,
    top: int,
    left: int,
    direction: int,
    steps: int = 8,
) -> list[dict]:
    states: list[dict] = []
    cur_patch = np.array(patch, copy=True)
    cur_top = int(top)
    cur_left = int(left)
    cur_dir = int(direction)
    for _ in range(steps):
        h, w = map(int, cur_patch.shape)
        cur_top, cur_left = _wall_contact(fence_bbox, cur_dir, cur_top, cur_left, h, w)
        states.append(
            {
                "direction": cur_dir,
                "top": int(cur_top),
                "left": int(cur_left),
                "height": h,
                "width": w,
                "patch": np.array(cur_patch, copy=True),
            }
        )
        cur_patch = np.rot90(cur_patch, -1)
        cur_dir = (cur_dir + 1) % 4
    return states


def _gap_from_state(state: dict) -> dict:
    direction = int(state["direction"])
    if direction in (UP, DOWN):
        start = int(state["left"])
        end = start + int(state["width"]) - 1
    else:
        start = int(state["top"])
        end = start + int(state["height"]) - 1
    return {"side": direction, "start": start, "end": end}


def _render_expected(
    grid: np.ndarray,
    bg: int,
    fence_color: int,
    patch: np.ndarray,
    fence_bbox: tuple[int, int, int, int],
    gap: dict,
    top: int,
    left: int,
    direction: int,
) -> tuple[np.ndarray, int]:
    exited_patch, out_top, out_left, stage = _simulate_exit(
        fence_bbox, gap, patch, top, left, direction, bg
    )
    out = grid.copy()
    out[(out != bg) & (out != fence_color)] = bg
    h, w = map(int, exited_patch.shape)
    sub = out[out_top : out_top + h, out_left : out_left + w]
    sub[exited_patch != bg] = exited_patch[exited_patch != bg]
    return out, int(stage)


def _gen_attempt() -> np.ndarray | None:
    size = int(_RNG.choice([2, 3]))
    base_mask = np.array(_RNG.choice(_SHAPES[size]), copy=True)
    bg, fence_color, push_color = map(
        int, _RNG.choice(_PALETTE, size=3, replace=False)
    )
    object_palette = [c for c in _PALETTE if c not in {bg, fence_color, push_color}]
    base_patch = _paint_object(base_mask, bg, object_palette)
    patch = np.rot90(base_patch, -int(_RNG.integers(0, 4)))

    direction = int(_RNG.integers(0, 4))
    push_side = (direction + 2) % 4

    outer_h = int(_RNG.integers(size + 6, 17))
    outer_w = int(_RNG.integers(size + 6, 17))
    if outer_h > 16 or outer_w > 16:
        return None

    margins = {
        UP: int(_RNG.integers(size, size + 4)),
        DOWN: int(_RNG.integers(size, size + 4)),
        LEFT: int(_RNG.integers(size, size + 4)),
        RIGHT: int(_RNG.integers(size, size + 4)),
    }

    grid_h = margins[UP] + outer_h + margins[DOWN]
    grid_w = margins[LEFT] + outer_w + margins[RIGHT]
    if max(grid_h, grid_w) > 30:
        return None

    fr0 = margins[UP]
    fc0 = margins[LEFT]
    fr1 = fr0 + outer_h - 1
    fc1 = fc0 + outer_w - 1
    fence_bbox = (fr0, fc0, fr1, fc1)

    top_lo = fr0 + 1 + (1 if push_side == UP else 0)
    top_hi = fr1 - size - (1 if push_side == DOWN else 0)
    left_lo = fc0 + 1 + (1 if push_side == LEFT else 0)
    left_hi = fc1 - size - (1 if push_side == RIGHT else 0)
    if top_lo > top_hi or left_lo > left_hi:
        return None

    top = int(_RNG.integers(top_lo, top_hi + 1))
    left = int(_RNG.integers(left_lo, left_hi + 1))

    mask = patch != bg
    side_candidates = _side_cells(mask, push_side)
    if not side_candidates:
        return None
    rr, cc = side_candidates[int(_RNG.integers(0, len(side_candidates)))]
    dr, dc = DIR_TO_DELTA[push_side]
    push_r = top + rr + dr
    push_c = left + cc + dc

    states = _contact_states(fence_bbox, patch, top, left, direction, steps=8)
    valid_stages = []
    for stage, state in enumerate(states[:5]):
        gap = _gap_from_state(state)
        if gap["side"] in (UP, DOWN):
            if gap["start"] <= fc0 or gap["end"] >= fc1:
                continue
        else:
            if gap["start"] <= fr0 or gap["end"] >= fr1:
                continue
        valid_stages.append((stage, gap))
    if not valid_stages:
        return None

    desired_stage = int(_RNG.integers(0, 5))
    choices = [item for item in valid_stages if item[0] == desired_stage]
    if not choices:
        return None
    chosen_stage, gap = choices[0]

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    grid[fr0, fc0 : fc1 + 1] = fence_color
    grid[fr1, fc0 : fc1 + 1] = fence_color
    grid[fr0 : fr1 + 1, fc0] = fence_color
    grid[fr0 : fr1 + 1, fc1] = fence_color
    if gap["side"] == UP:
        grid[fr0, gap["start"] : gap["end"] + 1] = bg
    elif gap["side"] == DOWN:
        grid[fr1, gap["start"] : gap["end"] + 1] = bg
    elif gap["side"] == LEFT:
        grid[gap["start"] : gap["end"] + 1, fc0] = bg
    else:
        grid[gap["start"] : gap["end"] + 1, fc1] = bg

    obj_sub = grid[top : top + size, left : left + size]
    obj_sub[mask] = patch[mask]
    grid[push_r, push_c] = push_color

    expected, actual_stage = _render_expected(
        grid, bg, fence_color, patch, fence_bbox, gap, top, left, direction
    )
    if actual_stage != chosen_stage:
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if _get_bg(grid) != bg:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
