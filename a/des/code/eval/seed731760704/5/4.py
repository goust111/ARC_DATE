from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

LEGEND_COLORS = (RED, BLUE, GREEN, YELLOW)
TRANSFORM_ORDER = ("hflip", "vflip", "rot_cw", "rot_ccw")

PATTERN_SIZE_RANGE = (3, 5)
TARGET_PAD_RANGE = (0, 1)
LAYOUT_MARGIN_RANGE = (1, 3)
LAYOUT_GAP_RANGE = (1, 3)
PATTERN_ATTEMPTS = 200
GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _apply_transform(arr: np.ndarray, op: str) -> np.ndarray:
    if op == "hflip":
        return np.fliplr(arr)
    if op == "vflip":
        return np.flipud(arr)
    if op == "rot_cw":
        return np.rot90(arr, -1)
    if op == "rot_ccw":
        return np.rot90(arr, 1)
    raise AssertionError(f"unknown transform: {op}")


def _array_signature(arr: np.ndarray) -> tuple[tuple[int, int], bytes]:
    a = np.asarray(arr)
    return tuple(map(int, a.shape)), a.tobytes()


def _find_divider_col(g: np.ndarray) -> int:
    solid = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert solid, "expected a solid grey divider column"
    runs: list[list[int]] = []
    run = [solid[0]]
    for c in solid[1:]:
        if c == run[-1] + 1:
            run.append(c)
        else:
            runs.append(run)
            run = [c]
    runs.append(run)
    best = max(runs, key=len)
    assert len(best) == 1, "divider must be one column wide"
    return int(best[0])


def _find_legend_blocks(g: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = g.shape
    blocks: list[tuple[int, int, int]] = []
    forbidden = {BLACK, GREY, ORANGE}
    for r in range(h - 1):
        for c in range(w - 1):
            color = int(g[r, c])
            if color in forbidden:
                continue
            if not np.all(g[r : r + 2, c : c + 2] == color):
                continue
            rr0 = max(0, r - 1)
            rr1 = min(h, r + 3)
            cc0 = max(0, c - 1)
            cc1 = min(w, c + 3)
            hood = g[rr0:rr1, cc0:cc1]
            mask = np.zeros_like(hood, dtype=bool)
            mask[r - rr0 : r - rr0 + 2, c - cc0 : c - cc0 + 2] = True
            if np.any((hood == color) & (~mask)):
                continue
            blocks.append((int(r), int(c), color))
    blocks.sort(key=lambda item: (item[0], item[1]))
    assert len(blocks) == 4, "expected exactly four legend blocks"
    rows = {r for r, _c, _color in blocks}
    assert len(rows) == 1, "legend blocks must share one row"
    colors = [color for _r, _c, color in blocks]
    assert len(colors) == len(set(colors)), "legend colors must be unique"
    return sorted(blocks, key=lambda item: item[1])


def _ring_bbox_for_component(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if np.array_equal(mask, expected):
        return int(r0), int(c0), int(r1), int(c1)
    return None


def _ring_boxes_for_color(g: np.ndarray, color: int) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for comp in _components(g == color):
        box = _ring_bbox_for_component(comp)
        if box is not None:
            boxes.append(box)
    return boxes


def _source_target_boxes(
    g: np.ndarray, color: int, divider_col: int
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    boxes = _ring_boxes_for_color(g, color)
    left = [box for box in boxes if box[3] < divider_col]
    right = [box for box in boxes if box[1] > divider_col]
    assert len(left) == 1 and len(right) == 1, "each legend color needs one left and one right frame"
    return left[0], right[0]


def _paste_centered(inner: np.ndarray, patch: np.ndarray) -> None:
    ih, iw = inner.shape
    ph, pw = patch.shape
    assert ph <= ih and pw <= iw
    inner[:] = BLACK
    off_r = (ih - ph) // 2
    off_c = (iw - pw) // 2
    inner[off_r : off_r + ph, off_c : off_c + pw] = patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the bottom legend order to decode border-color transforms, then copy each
    left black/orange pattern into the same-colored right frame after applying the
    assigned flip or rotation.
    """
    g = np.asarray(grid, dtype=int)
    divider_col = _find_divider_col(g)
    legend = _find_legend_blocks(g)
    color_to_op = {
        int(color): TRANSFORM_ORDER[i]
        for i, (_r, _c, color) in enumerate(legend)
    }

    out = g.copy()
    for color, op in color_to_op.items():
        src_box, dst_box = _source_target_boxes(g, color, divider_col)
        sr0, sc0, sr1, sc1 = src_box
        src_inner = g[sr0 + 1 : sr1, sc0 + 1 : sc1]
        assert src_inner.size > 0
        assert set(map(int, np.unique(src_inner))).issubset({BLACK, ORANGE})
        assert np.any(src_inner == ORANGE), "left frame must contain an orange pattern"

        transformed = _apply_transform(src_inner, op)

        dr0, dc0, dr1, dc1 = dst_box
        dst_inner = out[dr0 + 1 : dr1, dc0 + 1 : dc1]
        assert np.all(g[dr0 + 1 : dr1, dc0 + 1 : dc1] == BLACK), "right frame must start empty"
        _paste_centered(dst_inner, transformed)

    return out


def _random_pattern(h: int, w: int) -> np.ndarray | None:
    for _ in range(PATTERN_ATTEMPTS):
        density = random.uniform(0.28, 0.62)
        mask = np.random.random((h, w)) < density
        if not mask.any() or mask.all():
            continue
        if not (2 <= int(mask.sum()) <= h * w - 2):
            continue
        if len(set(np.where(mask)[0].tolist())) < 2:
            continue
        if len(set(np.where(mask)[1].tolist())) < 2:
            continue
        variants = [_apply_transform(mask.astype(int), op) for op in TRANSFORM_ORDER]
        sigs = {_array_signature(mask.astype(int))}
        sigs.update(_array_signature(v) for v in variants)
        if len(sigs) != 5:
            continue
        return np.where(mask, ORANGE, BLACK).astype(int)
    return None


def _draw_frame(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int, color: int) -> None:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color


def _frame_outer_size(inner_shape: tuple[int, int]) -> tuple[int, int]:
    return inner_shape[0] + 2, inner_shape[1] + 2


def _legend_positions(width: int, divider_col: int) -> list[int] | None:
    gaps = [random.randint(1, 3) for _ in range(3)]
    total = 8 + sum(gaps)
    candidates: list[list[int]] = []
    for start in range(width - total + 1):
        cols = []
        c = start
        ok = True
        for i in range(4):
            if c <= divider_col <= c + 1:
                ok = False
                break
            cols.append(c)
            if i < 3:
                c += 2 + gaps[i]
        if ok:
            candidates.append(cols)
    if not candidates:
        return None
    return random.choice(candidates)


def _expected_output(grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        pattern = np.asarray(spec["pattern"], dtype=int)
        op = str(spec["op"])
        transformed = _apply_transform(pattern, op)
        top, left = spec["target_pos"]
        inner_h, inner_w = spec["target_inner_shape"]
        patch = out[top + 1 : top + 1 + inner_h, left + 1 : left + 1 + inner_w]
        _paste_centered(patch, transformed)
    return out


def _build_attempt() -> np.ndarray | None:
    specs: list[dict[str, object]] = []
    has_extra_padding = False

    for color, op in zip(LEGEND_COLORS, TRANSFORM_ORDER, strict=True):
        src_h = random.randint(*PATTERN_SIZE_RANGE)
        src_w = random.randint(*PATTERN_SIZE_RANGE)
        pattern = _random_pattern(src_h, src_w)
        if pattern is None:
            return None

        transformed_shape = _apply_transform(pattern, op).shape
        pad_r = random.randint(*TARGET_PAD_RANGE)
        pad_c = random.randint(*TARGET_PAD_RANGE)
        has_extra_padding |= pad_r > 0 or pad_c > 0
        target_inner_shape = (
            int(transformed_shape[0] + 2 * pad_r),
            int(transformed_shape[1] + 2 * pad_c),
        )
        specs.append(
            {
                "color": int(color),
                "op": op,
                "pattern": pattern,
                "source_inner_shape": tuple(map(int, pattern.shape)),
                "target_inner_shape": target_inner_shape,
            }
        )

    if not has_extra_padding:
        pick = random.randrange(len(specs))
        op = str(specs[pick]["op"])
        pattern = np.asarray(specs[pick]["pattern"], dtype=int)
        transformed_shape = _apply_transform(pattern, op).shape
        axis = random.choice((0, 1))
        target_inner_shape = list(specs[pick]["target_inner_shape"])
        target_inner_shape[axis] = int(transformed_shape[axis] + 2)
        specs[pick]["target_inner_shape"] = tuple(target_inner_shape)

    src_outer_sizes = [_frame_outer_size(spec["source_inner_shape"]) for spec in specs]
    dst_outer_sizes = [_frame_outer_size(spec["target_inner_shape"]) for spec in specs]

    slot_h = (
        max(max(h for h, _w in src_outer_sizes), max(h for h, _w in dst_outer_sizes))
        + random.randint(1, 3)
    )
    slot_w = (
        max(max(w for _h, w in src_outer_sizes), max(w for _h, w in dst_outer_sizes))
        + random.randint(1, 3)
    )
    margin = random.randint(*LAYOUT_MARGIN_RANGE)
    gap_h = random.randint(*LAYOUT_GAP_RANGE)
    gap_w = random.randint(*LAYOUT_GAP_RANGE)
    legend_gap = random.randint(*LAYOUT_GAP_RANGE)

    half_w = 2 * slot_w + gap_w + 2 * margin
    divider_col = half_w
    width = half_w * 2 + 1
    content_h = 2 * slot_h + gap_h + 2 * margin
    height = content_h + legend_gap + 2

    grid = np.full((height, width), BLACK, dtype=int)
    grid[:, divider_col] = GREY

    slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
    left_perm = random.sample(range(4), 4)
    right_perm = random.sample(range(4), 4)

    def slot_origin(side: str, row: int, col: int) -> tuple[int, int]:
        top = margin + row * (slot_h + gap_h)
        left = margin + col * (slot_w + gap_w)
        if side == "right":
            left += divider_col + 1
        return top, left

    for slot_index, spec_index in enumerate(left_perm):
        row, col = slots[slot_index]
        slot_top, slot_left = slot_origin("left", row, col)
        spec = specs[spec_index]
        outer_h, outer_w = _frame_outer_size(spec["source_inner_shape"])
        top = slot_top + random.randint(0, slot_h - outer_h)
        left = slot_left + random.randint(0, slot_w - outer_w)
        spec["source_pos"] = (int(top), int(left))
        _draw_frame(
            grid,
            int(top),
            int(left),
            int(spec["source_inner_shape"][0]),
            int(spec["source_inner_shape"][1]),
            int(spec["color"]),
        )
        pattern = np.asarray(spec["pattern"], dtype=int)
        grid[top + 1 : top + 1 + pattern.shape[0], left + 1 : left + 1 + pattern.shape[1]] = pattern

    for slot_index, spec_index in enumerate(right_perm):
        row, col = slots[slot_index]
        slot_top, slot_left = slot_origin("right", row, col)
        spec = specs[spec_index]
        outer_h, outer_w = _frame_outer_size(spec["target_inner_shape"])
        top = slot_top + random.randint(0, slot_h - outer_h)
        left = slot_left + random.randint(0, slot_w - outer_w)
        spec["target_pos"] = (int(top), int(left))
        _draw_frame(
            grid,
            int(top),
            int(left),
            int(spec["target_inner_shape"][0]),
            int(spec["target_inner_shape"][1]),
            int(spec["color"]),
        )

    legend_cols = _legend_positions(width, divider_col)
    if legend_cols is None:
        return None
    legend_row = height - 2
    for color, left in zip(LEGEND_COLORS, legend_cols, strict=True):
        grid[legend_row : legend_row + 2, left : left + 2] = int(color)

    expected = _expected_output(grid, specs)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
