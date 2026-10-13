from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
TRAIN_FRAME = 5
TEST_FRAME = 7
FRAME_SIDE_RANGE = (5, 7)
GRID_HEIGHT_RANGE = (14, 22)
FRAME_GAP_RANGE = (2, 3)
VERTICAL_MARGIN_RANGE = (4, 8)
HORIZONTAL_MARGIN_RANGE = (1, 2)
PATTERN_SIZE_RANGE = (3, 4)
COLOR_POOL = np.array([1, 2, 3, 4, 6, 8, 9], dtype=int)
MAX_GEN_ATTEMPTS = 400
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    cols = {int(v) for v in np.unique(grid)}
    if exclude is not None:
        cols -= {int(v) for v in exclude}
    return sorted(cols)


def _swap_two_colors(grid: np.ndarray, colors: tuple[int, int]) -> np.ndarray:
    a, b = map(int, colors)
    out = np.array(grid, dtype=int, copy=True)
    a_mask = out == a
    b_mask = out == b
    out[a_mask] = b
    out[b_mask] = a
    return out


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _frame_components(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in _colors_present(grid, exclude={BG}):
        if color not in {TRAIN_FRAME, TEST_FRAME}:
            continue
        for coords in _component_coords(grid == color):
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            side_h = r1 - r0 + 1
            side_w = c1 - c0 + 1
            if side_h != side_w or not (FRAME_SIDE_RANGE[0] <= side_h <= FRAME_SIDE_RANGE[1]):
                continue
            side = side_h
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros_like(patch, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(patch[border] == color):
                continue
            if np.any(patch[~border] == color):
                continue
            if int(coords.shape[0]) != 4 * side - 4:
                continue
            frames.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "inner": patch[1:-1, 1:-1].copy(),
                }
            )
    frames.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], item["color"]))
    return frames


def _find_training_pair(frames: list[dict]) -> tuple[int, int, int]:
    signatures = [tuple(_colors_present(frame["inner"], exclude={BG})) for frame in frames]
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            if signatures[i] == signatures[j] and len(signatures[i]) == 2:
                rest = [k for k in range(len(frames)) if k not in (i, j)]
                if len(rest) == 1:
                    return i, j, rest[0]
    raise AssertionError("expected exactly one training pair and one test frame")


def _infer_swap_rule(src: np.ndarray, dst: np.ndarray) -> tuple[int, int]:
    colors = _colors_present(src, exclude={BG})
    assert colors == _colors_present(dst, exclude={BG}), "training pair must share the same palette"
    assert len(colors) == 2, "training pair must use exactly two colors"
    pair = (int(colors[0]), int(colors[1]))
    assert np.array_equal(_swap_two_colors(src, pair), dst) or np.array_equal(_swap_two_colors(dst, pair), src), (
        "training pair must differ by a two-color swap"
    )
    return pair


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a two-color swap from a framed training pair, then mirror-concatenate the test sprite and swap its colors.

    Input structure:
    - The grid has a black background with exactly three hollow square frames laid out left-to-right.
    - Two grey frames contain a training pair: same interior shape, same two non-black colors, but with those colors exchanged.
    - One orange frame contains a different two-color sprite on black, with no left-right symmetry.

    Rule:
    - Detect the three hollow square frames and extract their interiors.
    - Identify the training pair as the two interiors sharing the same two-color palette, and confirm they differ only by swapping those colors.
    - Take the remaining test interior, concatenate it with its horizontal mirror on the right, then swap the test sprite's two colors throughout that doubled pattern.
    - Return only the minimal bounding box of the non-black pixels in the transformed test pattern.

    Key insight:
    - The grey pair teaches an abstract "swap the two foreground colors" rule, while the orange test frame adds an extra geometric preprocessing step: mirror first, then recolor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _frame_components(g)
    assert len(frames) == 3, "expected exactly three hollow square frames"

    train_a, train_b, test_idx = _find_training_pair(frames)
    _infer_swap_rule(frames[train_a]["inner"], frames[train_b]["inner"])

    test_inner = frames[test_idx]["inner"]
    test_colors = _colors_present(test_inner, exclude={BG})
    assert len(test_colors) == 2, "test sprite must use exactly two colors"

    doubled = np.concatenate([test_inner, np.fliplr(test_inner)], axis=1)
    swapped = _swap_two_colors(doubled, (test_colors[0], test_colors[1]))
    out_box = _bbox(swapped != BG)
    assert out_box is not None, "transformed test sprite must stay non-empty"
    return _crop_bbox(swapped, out_box)


def _random_mask(rng: np.random.Generator, h: int, w: int, *, require_lr_asym: bool) -> np.ndarray | None:
    area_min = max(4, (h * w) // 2)
    area_max = h * w - 1
    if area_min > area_max:
        area_min = area_max
    for _ in range(80):
        target = int(rng.integers(area_min, area_max + 1))
        mask = np.zeros((h, w), dtype=bool)
        start = (int(rng.integers(h)), int(rng.integers(w)))
        mask[start] = True
        frontier = {start}
        while int(mask.sum()) < target:
            expand = set()
            for r, c in frontier:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        expand.add((nr, nc))
            if not expand:
                break
            cell = list(expand)[int(rng.integers(len(expand)))]
            mask[cell] = True
            frontier.add(cell)
        if int(mask.sum()) != target:
            continue
        if not require_lr_asym or not np.array_equal(mask, np.fliplr(mask)):
            return mask
    return None


def _paint_pattern(
    rng: np.random.Generator,
    inner_side: int,
    colors: tuple[int, int],
    *,
    require_lr_asym: bool,
) -> np.ndarray | None:
    max_size = min(PATTERN_SIZE_RANGE[1], inner_side)
    min_size = min(PATTERN_SIZE_RANGE[0], max_size)
    for _ in range(80):
        h = int(rng.integers(min_size, max_size + 1))
        w = int(rng.integers(min_size, max_size + 1))
        mask = _random_mask(rng, h, w, require_lr_asym=require_lr_asym)
        if mask is None:
            continue
        vals = rng.choice(np.array(colors, dtype=int), size=int(mask.sum()), replace=True)
        vals[0] = int(colors[0])
        vals[-1] = int(colors[1])
        patch = np.zeros((h, w), dtype=int)
        patch[mask] = vals
        inner = np.zeros((inner_side, inner_side), dtype=int)
        top = int(rng.integers(0, inner_side - h + 1))
        left = int(rng.integers(0, inner_side - w + 1))
        inner[top : top + h, left : left + w] = patch
        if require_lr_asym and np.array_equal(inner, np.fliplr(inner)):
            continue
        return inner
    return None


def _draw_frame(canvas: np.ndarray, top: int, left: int, side: int, frame_color: int, inner: np.ndarray) -> None:
    canvas[top, left : left + side] = frame_color
    canvas[top + side - 1, left : left + side] = frame_color
    canvas[top : top + side, left] = frame_color
    canvas[top : top + side, left + side - 1] = frame_color
    canvas[top + 1 : top + side - 1, left + 1 : left + side - 1] = BG
    canvas[top + 1 : top + side - 1, left + 1 : left + side - 1] = inner


def _manual_output(test_inner: np.ndarray) -> np.ndarray:
    colors = _colors_present(test_inner, exclude={BG})
    doubled = np.concatenate([test_inner, np.fliplr(test_inner)], axis=1)
    swapped = _swap_two_colors(doubled, (colors[0], colors[1]))
    return _crop_bbox(swapped, _bbox(swapped != BG))


def _generate_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    side = int(rng.integers(FRAME_SIDE_RANGE[0], FRAME_SIDE_RANGE[1] + 1))
    inner_side = side - 2
    top_margin = int(rng.integers(VERTICAL_MARGIN_RANGE[0], VERTICAL_MARGIN_RANGE[1] + 1))
    bottom_margin = int(rng.integers(VERTICAL_MARGIN_RANGE[0], VERTICAL_MARGIN_RANGE[1] + 1))
    gap1 = int(rng.integers(FRAME_GAP_RANGE[0], FRAME_GAP_RANGE[1] + 1))
    gap2 = int(rng.integers(FRAME_GAP_RANGE[0], FRAME_GAP_RANGE[1] + 1))
    left_margin = int(rng.integers(HORIZONTAL_MARGIN_RANGE[0], HORIZONTAL_MARGIN_RANGE[1] + 1))
    right_margin = int(rng.integers(HORIZONTAL_MARGIN_RANGE[0], HORIZONTAL_MARGIN_RANGE[1] + 1))

    H = side + top_margin + bottom_margin
    W = left_margin + side + gap1 + side + gap2 + side + right_margin
    if H > 30 or W > 30 or H < GRID_HEIGHT_RANGE[0] or H > GRID_HEIGHT_RANGE[1]:
        return None

    palette = rng.choice(COLOR_POOL, size=4, replace=False)
    train_colors = (int(palette[0]), int(palette[1]))
    test_colors = (int(palette[2]), int(palette[3]))

    train_inner = _paint_pattern(rng, inner_side, train_colors, require_lr_asym=False)
    test_inner = _paint_pattern(rng, inner_side, test_colors, require_lr_asym=True)
    if train_inner is None or test_inner is None:
        return None
    if np.array_equal(train_inner != BG, test_inner != BG):
        return None

    swapped_train = _swap_two_colors(train_inner, train_colors)
    if np.array_equal(train_inner, swapped_train):
        return None

    tops = [top_margin, top_margin, top_margin]
    if H - side - top_margin > 0:
        for idx in range(3):
            jitter_low = max(0, top_margin - 1)
            jitter_high = min(H - side, top_margin + 1)
            tops[idx] = int(rng.integers(jitter_low, jitter_high + 1))
    lefts = [
        left_margin,
        left_margin + side + gap1,
        left_margin + side + gap1 + side + gap2,
    ]

    canvas = np.zeros((H, W), dtype=int)
    train_pair = [train_inner, swapped_train]
    if bool(rng.integers(0, 2)):
        train_pair.reverse()
    _draw_frame(canvas, tops[0], lefts[0], side, TRAIN_FRAME, train_pair[0])
    _draw_frame(canvas, tops[1], lefts[1], side, TRAIN_FRAME, train_pair[1])
    _draw_frame(canvas, tops[2], lefts[2], side, TEST_FRAME, test_inner)

    expected = _manual_output(test_inner)
    if expected.shape[0] > 30 or expected.shape[1] > 30:
        return None
    if not np.array_equal(main(canvas), expected):
        return None
    return canvas, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        built = _generate_case(rng)
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input for puzzle 7")
