from __future__ import annotations

import numpy as np


GAP_TO_ROTATION = {"up": 0, "right": 3, "down": 2, "left": 1}
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _gap_side(border_mask: np.ndarray) -> str | None:
    h, w = border_mask.shape
    if h < 3 or w < 3:
        return None
    if np.any(border_mask[1:-1, 1:-1]):
        return None
    if not (
        border_mask[0, 0]
        and border_mask[0, -1]
        and border_mask[-1, 0]
        and border_mask[-1, -1]
    ):
        return None

    sides = {
        "up": border_mask[0, 1:-1],
        "down": border_mask[-1, 1:-1],
        "left": border_mask[1:-1, 0],
        "right": border_mask[1:-1, -1],
    }
    missing_side: str | None = None
    for name, arr in sides.items():
        missing = np.flatnonzero(~arr)
        if missing.size == 0:
            continue
        if missing.size not in (2, 3):
            return None
        if int(missing[-1] - missing[0] + 1) != int(missing.size):
            return None
        if missing_side is not None:
            return None
        missing_side = name
    if missing_side is None:
        return None
    for name, arr in sides.items():
        if name == missing_side:
            continue
        if not np.all(arr):
            return None
    return missing_side


def _find_window(grid: np.ndarray, bg: int) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            submask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            submask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            side = _gap_side(submask)
            if side is None:
                continue
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            if interior.size == 0:
                continue
            if np.any(interior == bg) or np.any(interior == color):
                continue
            candidates.append(
                {
                    "color": int(color),
                    "gap_side": side,
                    "bbox": (r0, c0, r1, c1),
                    "patch": interior.copy(),
                }
            )
    if len(candidates) != 1:
        raise AssertionError(f"expected exactly one window frame, found {len(candidates)}")
    return candidates[0]


def _find_board(grid: np.ndarray, bg: int, frame_color: int, patch_shape: tuple[int, int]) -> np.ndarray:
    mask = (grid != bg) & (grid != frame_color)
    rects: list[dict[str, object]] = []
    for coords in _components(mask):
        r0, c0, r1, c1 = _bbox(coords)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if int(coords.shape[0]) != h * w:
            continue
        rects.append({"area": h * w, "patch": grid[r0 : r1 + 1, c0 : c1 + 1].copy()})
    if not rects:
        raise AssertionError("no solid multicolor rectangle found")
    need_area = int(patch_shape[0] * patch_shape[1])
    bigger = [item for item in rects if int(item["area"]) > need_area]
    if not bigger:
        raise AssertionError("no collage board larger than the window interior")
    bigger.sort(key=lambda item: int(item["area"]), reverse=True)
    return np.asarray(bigger[0]["patch"], dtype=int)


def _all_matches(board: np.ndarray, patch: np.ndarray) -> list[tuple[int, int, int]]:
    ph, pw = map(int, patch.shape)
    matches: list[tuple[int, int, int]] = []
    for k in range(4):
        rotated = np.rot90(board, k)
        bh, bw = map(int, rotated.shape)
        if ph > bh or pw > bw:
            continue
        for r in range(bh - ph + 1):
            for c in range(bw - pw + 1):
                if np.array_equal(rotated[r : r + ph, c : c + pw], patch):
                    matches.append((k, r, c))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Locate the window's visible fragment inside the full collage, then rotate the crop so the frame gap faces upward.

    Input structure:
    - The background is the most common color.
    - One object is a one-pixel rectangular frame with a contiguous 2- or 3-cell gap on exactly one side.
    - The frame encloses a fully filled multicolor rectangular fragment with no background pixels.
    - Elsewhere there is a larger filled rectangular collage made of non-background, non-frame colors.

    Rule:
    - Identify the frame, its gap direction, and the multicolor fragment inside it.
    - Find the unique rotation of the larger collage in which that fragment appears as an exact rectangular subregion.
    - Crop that matched subregion from the rotated collage.
    - Rotate the crop again so that the frame gap direction becomes upward, then output the crop without the frame.

    Key insight:
    The window interior is only a locator clue; the frame gap encodes a second independent rotation that determines the final orientation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    window = _find_window(g, bg)
    patch = np.asarray(window["patch"], dtype=int)
    frame_color = int(window["color"])
    board = _find_board(g, bg, frame_color, patch.shape)

    matches = _all_matches(board, patch)
    if len(matches) != 1:
        raise AssertionError(f"expected unique collage match, found {len(matches)}")
    rot_k, row, col = matches[0]
    rotated_board = np.rot90(board, rot_k)
    crop = rotated_board[row : row + patch.shape[0], col : col + patch.shape[1]]
    return np.rot90(crop, GAP_TO_ROTATION[str(window["gap_side"])])


def _random_board(rng: np.random.Generator, palette: list[int], h: int, w: int) -> np.ndarray:
    for _ in range(40):
        board = rng.choice(palette, size=(h, w))
        if len(set(map(int, board.ravel()))) >= min(3, len(palette)):
            return board.astype(int)
    raise RuntimeError("failed to sample a sufficiently varied board")


def _choose_crop(
    rng: np.random.Generator, rotated_board: np.ndarray, gap_side: str
) -> tuple[np.ndarray, int, int, int]:
    rh, rw = map(int, rotated_board.shape)
    for _ in range(80):
        crop_h = int(rng.integers(4, rh + 1))
        crop_w = int(rng.integers(4, rw + 1))
        if crop_h == rh and crop_w == rw:
            continue
        side_len = crop_w if gap_side in ("up", "down") else crop_h
        gap_choices = [gap for gap in (2, 3) if side_len >= gap + 2]
        if not gap_choices:
            continue
        top = int(rng.integers(0, rh - crop_h + 1))
        left = int(rng.integers(0, rw - crop_w + 1))
        patch = rotated_board[top : top + crop_h, left : left + crop_w]
        if len(set(map(int, patch.ravel()))) < 2:
            continue
        gap_len = int(rng.choice(gap_choices))
        return patch.copy(), top, left, gap_len
    raise RuntimeError("failed to choose a valid crop")


def _make_window(
    patch: np.ndarray,
    bg: int,
    frame_color: int,
    gap_side: str,
    gap_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    ph, pw = map(int, patch.shape)
    window = np.full((ph + 2, pw + 2), bg, dtype=int)
    window[0, :] = frame_color
    window[-1, :] = frame_color
    window[:, 0] = frame_color
    window[:, -1] = frame_color
    window[1:-1, 1:-1] = patch

    span = pw if gap_side in ("up", "down") else ph
    start = int(rng.integers(1, span - gap_len))
    if gap_side == "up":
        window[0, 1 + start : 1 + start + gap_len] = bg
    elif gap_side == "down":
        window[-1, 1 + start : 1 + start + gap_len] = bg
    elif gap_side == "left":
        window[1 + start : 1 + start + gap_len, 0] = bg
    else:
        window[1 + start : 1 + start + gap_len, -1] = bg
    return window


def _place_objects(
    rng: np.random.Generator, board: np.ndarray, window: np.ndarray, bg: int
) -> np.ndarray | None:
    bh, bw = map(int, board.shape)
    wh, ww = map(int, window.shape)
    gap = int(rng.integers(2, 5))
    margin_a = int(rng.integers(1, 4))
    margin_b = int(rng.integers(1, 4))
    margin_c = int(rng.integers(1, 4))
    margin_d = int(rng.integers(1, 4))
    extra = int(rng.integers(0, 4))
    arrangement = str(rng.choice(["horizontal", "vertical"]))
    swapped = bool(rng.integers(0, 2))

    if arrangement == "horizontal":
        canvas_h = max(bh, wh) + margin_a + margin_b + extra
        canvas_w = bw + ww + gap + margin_c + margin_d
        if canvas_h > 30 or canvas_w > 30:
            return None
        board_x = margin_c if not swapped else margin_c + ww + gap
        window_x = margin_c + bw + gap if not swapped else margin_c
        board_y = int(rng.integers(margin_a, canvas_h - margin_b - bh + 1))
        window_y = int(rng.integers(margin_a, canvas_h - margin_b - wh + 1))
    else:
        canvas_h = bh + wh + gap + margin_a + margin_b
        canvas_w = max(bw, ww) + margin_c + margin_d + extra
        if canvas_h > 30 or canvas_w > 30:
            return None
        board_y = margin_a if not swapped else margin_a + wh + gap
        window_y = margin_a + bh + gap if not swapped else margin_a
        board_x = int(rng.integers(margin_c, canvas_w - margin_d - bw + 1))
        window_x = int(rng.integers(margin_c, canvas_w - margin_d - ww + 1))

    grid = np.full((canvas_h, canvas_w), bg, dtype=int)
    grid[board_y : board_y + bh, board_x : board_x + bw] = board
    grid[window_y : window_y + wh, window_x : window_x + ww] = window
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    colors = list(range(10))
    rng.shuffle(colors)
    bg = int(colors[0])
    frame_color = int(colors[1])
    palette = [int(c) for c in colors[2 : 2 + int(rng.integers(3, 6))]]

    board_h = int(rng.integers(6, 11))
    board_w = int(rng.integers(6, 11))
    board = _random_board(rng, palette, board_h, board_w)

    gap_side = str(rng.choice(["up", "right", "down", "left"]))
    match_k = int(rng.integers(0, 4))
    rotated_board = np.rot90(board, match_k)
    try:
        patch, top, left, gap_len = _choose_crop(rng, rotated_board, gap_side)
    except RuntimeError:
        return None
    matches = _all_matches(board, patch)
    if matches != [(match_k, top, left)]:
        return None

    window = _make_window(patch, bg, frame_color, gap_side, gap_len, rng)
    grid = _place_objects(rng, board, window, bg)
    if grid is None:
        return None

    values, counts = np.unique(grid, return_counts=True)
    if int(values[np.argmax(counts)]) != bg:
        return None

    expected = np.rot90(patch, GAP_TO_ROTATION[gap_side])
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to produce a valid example")
