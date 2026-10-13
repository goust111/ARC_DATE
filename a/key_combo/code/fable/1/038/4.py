from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRY_LIMIT = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    height, width = map(int, shape)
    return 0 <= int(r) < height and 0 <= int(c) < width


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _ring_border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((int(height), int(width)), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    candidates: list[tuple[int, int, int, int]] = []

    for color in sorted(int(v) for v in np.unique(g) if int(v) != BLACK):
        for coords in _components_mask(g == int(color)):
            area = int(coords.shape[0])
            r0, c0, r1, c1 = _bbox_coords(coords)
            height = int(r1 - r0 + 1)
            width = int(c1 - c0 + 1)
            if height < 5 or width < 5:
                continue
            if area != 2 * (height + width) - 4:
                continue

            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            border = _ring_border_mask(height, width)
            if not np.all(sub[border] == int(color)):
                continue
            if np.any(sub[~border] == int(color)):
                continue

            inner = sub[1:-1, 1:-1]
            if inner.shape[0] != inner.shape[1] or inner.shape[0] % 2 == 0:
                continue
            if np.count_nonzero(inner == GREY) != 1:
                continue
            if np.count_nonzero(inner != BLACK) < 3:
                continue
            candidates.append((r0, c0, r1, c1))

    assert len(candidates) == 1, "expected a unique legend frame"
    return candidates[0]


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int], tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_bbox = _find_frame_bbox(g)
    r0, c0, r1, c1 = frame_bbox
    legend = g[r0 + 1 : r1, c0 + 1 : c1].copy()
    side = int(legend.shape[0])
    assert legend.shape == (side, side) and side >= 3 and side % 2 == 1, "expected an odd square legend"

    inner_pointer = np.argwhere(legend == GREY)
    assert len(inner_pointer) == 1, "expected one grey pointer cell inside the legend"

    outside_grey: list[tuple[int, int]] = []
    for row, col in np.argwhere(g == GREY):
        rr = int(row)
        cc = int(col)
        if r0 < rr < r1 and c0 < cc < c1:
            continue
        outside_grey.append((rr, cc))

    assert len(outside_grey) == 1, "expected one grey anchor outside the frame"
    return legend, outside_grey[0], frame_bbox


def _stamp_once(
    out: np.ndarray,
    legend: np.ndarray,
    anchor: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = out.shape
    side = int(legend.shape[0])
    center = side // 2

    coords = np.argwhere(legend != BLACK)
    rows = coords[:, 0] + int(anchor[0]) - center
    cols = coords[:, 1] + int(anchor[1]) - center
    keep = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)

    stamp_mask = np.zeros((height, width), dtype=bool)
    if np.any(keep):
        out[rows[keep], cols[keep]] = legend[coords[keep, 0], coords[keep, 1]]
        stamp_mask[rows[keep], cols[keep]] = True

    pointer = np.argwhere(legend == GREY)
    assert len(pointer) == 1, "expected one grey pointer cell in every rotation"
    pr, pc = map(int, pointer[0])
    next_anchor = (int(anchor[0]) - center + pr, int(anchor[1]) - center + pc)
    return stamp_mask, next_anchor


def _run_chain(
    base_grid: np.ndarray,
    legend: np.ndarray,
    start_anchor: tuple[int, int],
) -> tuple[np.ndarray, dict]:
    out = np.array(base_grid, dtype=int, copy=True)
    historical = np.zeros(out.shape, dtype=bool)
    visible = np.zeros(out.shape, dtype=bool)
    anchor = (int(start_anchor[0]), int(start_anchor[1]))

    reason: str | None = None
    steps = 0
    for turn in range(out.shape[0] * out.shape[1] + 5):
        rotated = np.rot90(legend, -(turn % 4))
        stamp_mask, next_anchor = _stamp_once(out, rotated, anchor)
        visible |= stamp_mask
        steps += 1

        nr, nc = next_anchor
        if not _in_bounds(out.shape, nr, nc):
            reason = "out_of_bounds"
            break
        if historical[nr, nc]:
            reason = "overlap"
            break

        historical |= stamp_mask
        anchor = next_anchor

    assert reason is not None, "chain should terminate by leaving the grid or revisiting an older stamp"
    return out, {"steps": int(steps), "reason": reason, "visible_mask": visible}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeatedly stamp the framed legend from one grey anchor to the next, rotating 90 degrees more each time.

    Input structure:
    - Black is the background.
    - One one-cell-thick closed frame encloses an odd square legend crop.
    - Inside that crop, black cells are empty and the non-black cells form an asymmetric multicolor motif with exactly one grey pointer cell.
    - Outside the frame there is exactly one isolated grey anchor cell.

    Rule:
    - Keep the framed legend unchanged.
    - Stamp the frame interior itself, centered on the current anchor, treating black as transparent and clipping any off-grid cells.
    - After stamp k, the landed grey pointer cell defines the next anchor, and stamp k+1 uses the legend rotated clockwise by 90 degrees times k.
    - Continue until the spawned anchor is out of bounds or lies on a non-black cell painted by an earlier stamp; later stamps overwrite earlier stamp colors where they overlap.

    Key insight:
    - The unique grey legend cell is both part of the motif and the movement vector, so every stamp rotates and reuses its own pointer to create the next link in the chain.
    """
    g = np.asarray(grid, dtype=int)
    legend, anchor, _frame_bbox = _parse_input(g)
    out, _meta = _run_chain(g, legend, anchor)
    return out


def _is_rotationally_asymmetric(sprite: np.ndarray) -> bool:
    return all(not np.array_equal(sprite, np.rot90(sprite, k)) for k in (1, 2, 3))


def _grow_mask(side: int, pointer: tuple[int, int], target: int) -> np.ndarray:
    center = side // 2
    mask = np.zeros((side, side), dtype=bool)
    mask[center, center] = True
    mask[pointer] = True

    while int(mask.sum()) < int(target):
        frontier: set[tuple[int, int]] = set()
        for row, col in np.argwhere(mask):
            rr = int(row)
            cc = int(col)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < side and 0 <= nc < side and not mask[nr, nc]:
                        frontier.add((nr, nc))
        if not frontier:
            break
        row, col = random.choice(list(frontier))
        mask[row, col] = True
    return mask


def _generate_legend() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        side = random.choice((3, 5))
        center = side // 2
        pointer_choices = [
            (r, c)
            for r in range(side)
            for c in range(side)
            if (r, c) != (center, center) and max(abs(r - center), abs(c - center)) == center
        ]
        pointer = random.choice(pointer_choices)
        target = random.randint(max(5, side + 1), min(side * side - 1, side + 5))
        mask = _grow_mask(side, pointer, target)
        if int(mask.sum()) < 5 or int(mask.sum()) == side * side:
            continue

        colors = random.sample(PALETTE, random.randint(2, 4))
        legend = np.zeros((side, side), dtype=int)
        legend[center, center] = colors[0]

        extras = [tuple(map(int, p)) for p in np.argwhere(mask) if tuple(map(int, p)) not in {pointer, (center, center)}]
        if len(extras) < 2:
            continue
        random.shuffle(extras)
        legend[extras[0]] = colors[1]
        for row, col in extras[1:]:
            legend[row, col] = random.choice(colors)
        legend[pointer] = GREY

        non_grey_colors = {int(v) for v in legend.flatten() if int(v) not in {BLACK, GREY}}
        if len(non_grey_colors) < 2:
            continue
        if not _is_rotationally_asymmetric(legend):
            continue
        return legend

    raise RuntimeError("failed to generate a valid legend motif")


def _find_anchor_for_mode(
    legend: np.ndarray,
    shape: tuple[int, int],
    stop_mode: str,
) -> tuple[tuple[int, int], dict] | None:
    blank = np.zeros(shape, dtype=int)
    anchors = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
    random.shuffle(anchors)

    target_steps = 4 if stop_mode == "overlap" else random.choice((2, 3))
    fallback: tuple[tuple[int, int], dict] | None = None
    for anchor in anchors:
        _out, meta = _run_chain(blank, legend, anchor)
        if meta["reason"] != stop_mode:
            continue
        if stop_mode == "overlap":
            if meta["steps"] == 4:
                return anchor, meta
            continue
        if meta["steps"] == target_steps:
            return anchor, meta
        if meta["steps"] in {2, 3} and fallback is None:
            fallback = (anchor, meta)
    return fallback


def _frame_positions(
    shape: tuple[int, int],
    frame_side: int,
    stamp_mask: np.ndarray,
    anchor: tuple[int, int],
) -> list[tuple[int, int]]:
    height, width = shape
    positions: list[tuple[int, int, int]] = []
    for r0 in range(height - frame_side + 1):
        for c0 in range(width - frame_side + 1):
            r1 = r0 + frame_side - 1
            c1 = c0 + frame_side - 1
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(height - 1, r1 + 1)
            cc1 = min(width - 1, c1 + 1)
            if np.any(stamp_mask[rr0 : rr1 + 1, cc0 : cc1 + 1]):
                continue
            if r0 <= anchor[0] <= r1 and c0 <= anchor[1] <= c1:
                continue

            dr = 0 if r0 <= anchor[0] <= r1 else min(abs(anchor[0] - r0), abs(anchor[0] - r1))
            dc = 0 if c0 <= anchor[1] <= c1 else min(abs(anchor[1] - c0), abs(anchor[1] - c1))
            if max(dr, dc) < 2:
                continue

            center_dist = abs(anchor[0] - (r0 + r1) // 2) + abs(anchor[1] - (c0 + c1) // 2)
            positions.append((center_dist, r0, c0))

    if not positions:
        return []

    preferred = [pos for pos in positions if 4 <= pos[0] <= 14]
    chosen = preferred if preferred else positions
    return [(r0, c0) for _dist, r0, c0 in sorted(chosen)]


def _build_input(
    shape: tuple[int, int],
    legend: np.ndarray,
    anchor: tuple[int, int],
    frame_pos: tuple[int, int],
    frame_color: int,
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    side = int(legend.shape[0])
    r0, c0 = frame_pos
    frame_side = side + 2
    r1 = r0 + frame_side - 1
    c1 = c0 + frame_side - 1

    grid[r0, c0 : c1 + 1] = int(frame_color)
    grid[r1, c0 : c1 + 1] = int(frame_color)
    grid[r0 : r1 + 1, c0] = int(frame_color)
    grid[r0 : r1 + 1, c1] = int(frame_color)
    grid[r0 + 1 : r1, c0 + 1 : c1] = legend
    grid[int(anchor[0]), int(anchor[1])] = GREY
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        legend = _generate_legend()
        stop_mode = random.choice(("out_of_bounds", "overlap"))
        height = random.randint(max(16, legend.shape[0] + 8), 30)
        width = random.randint(max(16, legend.shape[0] + 8), 30)
        shape = (height, width)

        anchor_info = _find_anchor_for_mode(legend, shape, stop_mode)
        if anchor_info is None:
            continue
        anchor, meta = anchor_info

        frame_positions = _frame_positions(shape, int(legend.shape[0]) + 2, meta["visible_mask"], anchor)
        if not frame_positions:
            continue

        used_colors = {int(v) for v in np.unique(legend) if int(v) != BLACK}
        frame_choices = [color for color in PALETTE if color not in used_colors]
        if not frame_choices:
            continue

        top_left_options = frame_positions[: min(20, len(frame_positions))]
        frame_pos = random.choice(top_left_options)
        frame_color = random.choice(frame_choices)
        grid = _build_input(shape, legend, anchor, frame_pos, frame_color)

        try:
            parsed_legend, parsed_anchor, _bbox = _parse_input(grid)
            expected, expected_meta = _run_chain(grid, legend, anchor)
            actual = main(grid)
        except AssertionError:
            continue

        if not np.array_equal(parsed_legend, legend):
            continue
        if tuple(map(int, parsed_anchor)) != tuple(map(int, anchor)):
            continue
        if expected_meta["reason"] != stop_mode:
            continue
        if not np.array_equal(actual, expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid pointer-chain stamping puzzle")
