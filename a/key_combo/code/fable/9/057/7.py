from __future__ import annotations

from collections import Counter
from itertools import permutations
import random

import numpy as np


BACKGROUND = 0
BODY_COLORS = (1, 2, 3)
CONNECTOR_COLORS = (4, 5, 6, 7, 8, 9)

GROUP_COUNT_RANGE = (3, 5)
CANVAS_HEIGHT_RANGE = (6, 10)
GROUP_WIDTH_RANGE = (2, 4)
GROUP_RAW_HEIGHT_RANGE = (1, 4)
SEPARATOR_CHOICES = (1, 2)
MAX_GENERATION_ATTEMPTS = 400
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
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
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _column_groups(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    nonempty = np.any(grid != bg, axis=0)
    groups: list[np.ndarray] = []
    c = 0
    while c < grid.shape[1]:
        if not nonempty[c]:
            c += 1
            continue
        c0 = c
        while c + 1 < grid.shape[1] and nonempty[c + 1]:
            c += 1
        groups.append(np.asarray(grid[:, c0 : c + 1], dtype=int).copy())
        c += 1
    assert groups, "expected at least one non-background column group"
    return groups


def _cell_degrees(mask: np.ndarray) -> dict[tuple[int, int], int]:
    coords = [tuple(map(int, x)) for x in np.argwhere(mask)]
    coord_set = set(coords)
    degrees: dict[tuple[int, int], int] = {}
    for r, c in coords:
        degrees[(r, c)] = sum((r + dr, c + dc) in coord_set for dr, dc in NEIGHBORS4)
    return degrees


def _group_info(group: np.ndarray, bg: int) -> dict[str, int]:
    mask = group != bg
    coords = np.argwhere(mask)
    assert coords.size > 0, "group must contain foreground cells"
    assert len(_components(mask)) == 1, "group must be connected"

    degrees = _cell_degrees(mask)
    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    assert len(endpoints) == 2, "path group must have exactly two degree-1 endpoints"
    assert all(degree <= 2 for degree in degrees.values()), "path group cannot branch"

    width = int(group.shape[1])
    left_candidates = [cell for cell in endpoints if cell[1] == 0]
    right_candidates = [cell for cell in endpoints if cell[1] == width - 1]
    assert len(left_candidates) == 1 and len(right_candidates) == 1, "expected one endpoint on each outer edge"

    left_r, _ = left_candidates[0]
    right_r, _ = right_candidates[0]
    return {
        "left_row": int(left_r),
        "right_row": int(right_r),
        "left_color": int(group[left_r, 0]),
        "right_color": int(group[right_r, width - 1]),
        "width": width,
    }


def _shift_group(group: np.ndarray, bg: int, dr: int) -> np.ndarray:
    h, w = map(int, group.shape)
    out = np.full((h, w), bg, dtype=int)
    coords = np.argwhere(group != bg)
    assert coords.size > 0
    shifted = coords.copy()
    shifted[:, 0] += int(dr)
    assert int(shifted[:, 0].min()) >= 0 and int(shifted[:, 0].max()) < h, "vertical shift leaves canvas"
    out[shifted[:, 0], shifted[:, 1]] = group[coords[:, 0], coords[:, 1]]
    return out


def _valid_orders(groups: list[np.ndarray], bg: int) -> list[tuple[int, ...]]:
    infos = [_group_info(group, bg) for group in groups]
    rights = [info["right_color"] for info in infos]
    valid: list[tuple[int, ...]] = []
    for order in permutations(range(len(groups))):
        start = infos[order[0]]
        if start["left_color"] in rights:
            continue
        if any(infos[order[i]]["right_color"] != infos[order[i + 1]]["left_color"] for i in range(len(order) - 1)):
            continue
        valid.append(order)
    return valid


def main(grid: np.ndarray) -> np.ndarray:
    """Reorder the separated path segments by connector colors, then pack them into one touching chain.

    Input structure:
    - A horizontal strip contains several path-like foreground groups separated by one or more all-background columns.
    - Inside each group, all non-endpoint cells share one body color, while the unique degree-1 endpoint on the left outer column and the unique degree-1 endpoint on the right outer column use connector colors.
    - The drawing order on the strip is arbitrary, but there is exactly one chain order whose adjacent groups satisfy right-connector color equals next left-connector color.

    Rule:
    - Remove every all-background separator column and treat each remaining contiguous column run as one rigid group.
    - Find the unique domino-like ordering: the first group is the only one whose left connector color does not appear as any group's right connector color.
    - Place that first group unchanged, then place each later group immediately to its right after a vertical slide, without rotation or recoloring, so the later group's left endpoint is on the same row as the previous group's right endpoint.

    Key insight:
    - The true order is encoded by the left/right connector-color pairing, not by the groups' visible left-to-right positions in the input strip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    groups = _column_groups(g, bg)

    orders = _valid_orders(groups, bg)
    assert len(orders) == 1, "expected a unique connector chain"
    order = orders[0]
    infos = [_group_info(group, bg) for group in groups]

    h = int(g.shape[0])
    total_width = sum(int(groups[idx].shape[1]) for idx in order)
    out = np.full((h, total_width), bg, dtype=int)

    c = 0
    first_idx = order[0]
    first = groups[first_idx]
    out[:, : first.shape[1]] = first
    prev_right_row = infos[first_idx]["right_row"]
    c += int(first.shape[1])

    for idx in order[1:]:
        dr = prev_right_row - infos[idx]["left_row"]
        shifted = _shift_group(groups[idx], bg, dr)
        out[:, c : c + shifted.shape[1]] = shifted
        prev_right_row = infos[idx]["right_row"] + dr
        c += int(shifted.shape[1])

    assert len(_components(out != bg)) == 1, "packed output must be one connected chain"
    return out


def _build_path_cells(rows: list[int]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = [(int(rows[0]), 0)]
    cur_row = int(rows[0])
    for c in range(1, len(rows)):
        cells.append((cur_row, c))
        target = int(rows[c])
        step = 0 if target == cur_row else (1 if target > cur_row else -1)
        while cur_row != target:
            cur_row += step
            cells.append((cur_row, c))
    min_row = min(r for r, _ in cells)
    return [(r - min_row, c) for r, c in cells]


def _sample_group_shape(rng: random.Random) -> tuple[np.ndarray, int, int]:
    while True:
        width = rng.randint(*GROUP_WIDTH_RANGE)
        raw_h = rng.randint(*GROUP_RAW_HEIGHT_RANGE)
        rows = [rng.randrange(raw_h)]
        for _ in range(1, width):
            rows.append(rng.randrange(raw_h))

        cells = _build_path_cells(rows)
        max_row = max(r for r, _ in cells)
        arr = np.zeros((max_row + 1, width), dtype=int)
        for r, c in cells:
            arr[r, c] = 1

        try:
            info = _group_info(arr, 0)
        except AssertionError:
            continue
        if arr.shape[0] > 4:
            continue
        if width == 2 and arr.shape[0] == 1:
            continue
        if len(np.unique(np.array(rows))) == 1 and rng.random() < 0.6:
            continue
        return arr, info["left_row"], info["right_row"]


def _colorize_group(mask_arr: np.ndarray, left_row: int, right_row: int, body: int, left_color: int, right_color: int) -> np.ndarray:
    out = np.zeros_like(mask_arr, dtype=int)
    out[mask_arr != 0] = int(body)
    out[int(left_row), 0] = int(left_color)
    out[int(right_row), out.shape[1] - 1] = int(right_color)
    return out


def _embed_group(local: np.ndarray, height: int, top: int, bg: int) -> np.ndarray:
    top = int(top)
    assert 0 <= top <= height - local.shape[0]
    out = np.full((height, local.shape[1]), bg, dtype=int)
    out[top : top + local.shape[0], :] = local
    return out


def _assemble_input(full_groups: list[np.ndarray], separators: list[int], order: list[int], bg: int) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for pos, idx in enumerate(order):
        pieces.append(full_groups[idx])
        if pos < len(order) - 1:
            pieces.append(np.full((full_groups[idx].shape[0], separators[pos]), bg, dtype=int))
    return np.concatenate(pieces, axis=1)


def _plan_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(*GROUP_COUNT_RANGE)
    height = rng.randint(*CANVAS_HEIGHT_RANGE)
    chain_colors = list(rng.sample(CONNECTOR_COLORS, count + 1))
    body_colors = [rng.choice(BODY_COLORS) for _ in range(count)]

    locals_: list[np.ndarray] = []
    left_rows: list[int] = []
    right_rows: list[int] = []
    output_tops: list[int] = []

    start_top: int | None = None
    prev_right_global: int | None = None
    shift_steps: list[int] = []
    for i in range(count):
        for _ in range(80):
            mask_arr, left_row, right_row = _sample_group_shape(rng)
            if i == 0:
                top = rng.randint(0, height - mask_arr.shape[0])
            else:
                assert prev_right_global is not None
                top = prev_right_global - left_row
                if not (0 <= top <= height - mask_arr.shape[0]):
                    continue
            locals_.append(
                _colorize_group(
                    mask_arr,
                    left_row,
                    right_row,
                    body_colors[i],
                    chain_colors[i],
                    chain_colors[i + 1],
                )
            )
            left_rows.append(left_row)
            right_rows.append(right_row)
            output_tops.append(int(top))
            if i == 0:
                start_top = int(top)
            prev_right_global = int(top + right_row)
            if i > 0:
                shift_steps.append(int(top - output_tops[i - 1]))
            break
        else:
            return None

    assert start_top is not None

    widths = [int(arr.shape[1]) for arr in locals_]
    if sum(widths) + (count - 1) * max(SEPARATOR_CHOICES) > 30:
        return None

    if not any(arr.shape[0] > 1 for arr in locals_):
        return None
    if not any(abs(delta) > 0 for delta in shift_steps):
        return None

    input_tops = [start_top]
    nonzero_input_slide = False
    for i in range(1, count):
        local_h = locals_[i].shape[0]
        choices = list(range(0, height - local_h + 1))
        rng.shuffle(choices)
        chosen = None
        for top in choices:
            if top != output_tops[i]:
                chosen = int(top)
                break
        if chosen is None:
            chosen = int(output_tops[i])
        if chosen != output_tops[i]:
            nonzero_input_slide = True
        input_tops.append(chosen)
    if not nonzero_input_slide:
        return None

    full_groups_output_order = [_embed_group(locals_[i], height, output_tops[i], BACKGROUND) for i in range(count)]
    full_groups_input_order = [_embed_group(locals_[i], height, input_tops[i], BACKGROUND) for i in range(count)]

    perm = list(range(count))
    rng.shuffle(perm)
    if perm == list(range(count)):
        return None

    separators = [rng.choice(SEPARATOR_CHOICES) for _ in range(count - 1)]
    inp = _assemble_input(full_groups_input_order, separators, perm, BACKGROUND)
    if max(inp.shape) > 30:
        return None

    expected = np.concatenate(full_groups_output_order, axis=1)
    if max(expected.shape) > 30:
        return None

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    groups = _column_groups(inp, BACKGROUND)
    orders = _valid_orders(groups, BACKGROUND)
    if len(orders) != 1:
        return None
    if orders[0] == tuple(range(len(groups))):
        return None

    info_by_input = [_group_info(group, BACKGROUND) for group in groups]
    rights = [info["right_color"] for info in info_by_input]
    starts = [i for i, info in enumerate(info_by_input) if info["left_color"] not in rights]
    if len(starts) != 1:
        return None

    if len(_components(actual != BACKGROUND)) != 1:
        return None

    return inp


def generate_input() -> np.ndarray:
    for seed in range(MAX_GENERATION_ATTEMPTS):
        rng = random.Random(random.randrange(1 << 30) ^ (seed << 16))
        grid = _plan_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid domino-chain input")
