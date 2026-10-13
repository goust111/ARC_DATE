from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MIN_WALL_RUN = 5
MAX_GENERATION_ATTEMPTS = 300
CHANNEL_PLACEMENT_ATTEMPTS = 80


def _most_common(values: np.ndarray) -> int:
    vals, counts = np.unique(values.astype(int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _background_color(g: np.ndarray) -> int:
    if np.any(g == BG):
        return BG
    return _most_common(g)


def _wall_color(g: np.ndarray, bg: int) -> int:
    non_bg = g[g != bg]
    assert non_bg.size > 0, "expected grey channel walls"
    if np.any(non_bg == GREY):
        return GREY
    return _most_common(non_bg)


def _runs_1d(mask: np.ndarray, min_len: int = MIN_WALL_RUN) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    n = int(mask.size)
    while i < n:
        if not bool(mask[i]):
            i += 1
            continue
        start = i
        while i < n and bool(mask[i]):
            i += 1
        end = i - 1
        if end - start + 1 >= min_len:
            runs.append((int(start), int(end)))
    return runs


def _horizontal_wall_runs(g: np.ndarray, wall: int) -> dict[int, list[tuple[int, int]]]:
    by_row: dict[int, list[tuple[int, int]]] = {}
    for r in range(g.shape[0]):
        runs = _runs_1d(g[r, :] == wall)
        if runs:
            by_row[int(r)] = runs
    return by_row


def _vertical_wall_runs(g: np.ndarray, wall: int) -> dict[int, list[tuple[int, int]]]:
    by_col: dict[int, list[tuple[int, int]]] = {}
    for c in range(g.shape[1]):
        runs = _runs_1d(g[:, c] == wall)
        if runs:
            by_col[int(c)] = runs
    return by_col


def _channel_seeds(g: np.ndarray, bg: int, wall: int, r0: int, r1: int, c0: int, c1: int) -> list[tuple[int, int, int]]:
    region = g[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(region == wall):
        return []
    coords = np.argwhere((region != bg) & (region != wall))
    seeds = [(int(r0 + r), int(c0 + c), int(region[r, c])) for r, c in coords]
    if len(seeds) != 2:
        return []
    if seeds[0][2] == seeds[1][2]:
        return []
    return seeds


def _find_channels(g: np.ndarray, bg: int, wall: int) -> list[dict]:
    channels: list[dict] = []

    horizontal = _horizontal_wall_runs(g, wall)
    for top_r, top_runs in horizontal.items():
        for width in (2, 3):
            bottom_r = top_r + width + 1
            if bottom_r not in horizontal:
                continue
            for tc0, tc1 in top_runs:
                for bc0, bc1 in horizontal[bottom_r]:
                    c0 = max(tc0, bc0)
                    c1 = min(tc1, bc1)
                    if c1 - c0 + 1 < MIN_WALL_RUN:
                        continue
                    seeds = _channel_seeds(g, bg, wall, top_r + 1, bottom_r - 1, c0, c1)
                    if len(seeds) != 2 or seeds[0][1] == seeds[1][1]:
                        continue
                    channels.append(
                        {
                            "orientation": "h",
                            "width": int(width),
                            "top": int(top_r),
                            "bottom": int(bottom_r),
                            "left": int(c0),
                            "right": int(c1),
                            "seeds": seeds,
                        }
                    )

    vertical = _vertical_wall_runs(g, wall)
    for left_c, left_runs in vertical.items():
        for width in (2, 3):
            right_c = left_c + width + 1
            if right_c not in vertical:
                continue
            for lr0, lr1 in left_runs:
                for rr0, rr1 in vertical[right_c]:
                    r0 = max(lr0, rr0)
                    r1 = min(lr1, rr1)
                    if r1 - r0 + 1 < MIN_WALL_RUN:
                        continue
                    seeds = _channel_seeds(g, bg, wall, r0, r1, left_c + 1, right_c - 1)
                    if len(seeds) != 2 or seeds[0][0] == seeds[1][0]:
                        continue
                    channels.append(
                        {
                            "orientation": "v",
                            "width": int(width),
                            "top": int(r0),
                            "bottom": int(r1),
                            "left": int(left_c),
                            "right": int(right_c),
                            "seeds": seeds,
                        }
                    )

    deduped: list[dict] = []
    used_seed_sets: set[frozenset[tuple[int, int]]] = set()
    for channel in sorted(channels, key=lambda ch: (ch["orientation"], ch["top"], ch["left"], ch["bottom"], ch["right"])):
        key = frozenset((r, c) for r, c, _ in channel["seeds"])
        if key in used_seed_sets:
            continue
        used_seed_sets.add(key)
        deduped.append(channel)
    return deduped


def _offset_sign(width: int, local_offset: int) -> int:
    doubled = 2 * int(local_offset)
    center = int(width) - 1
    if doubled < center:
        return -1
    if doubled > center:
        return 1
    return 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow full-width colored bands from seeds inside grey-walled straight channels.

    Input structure:
    - The background is black.
    - Grey pixels form several parallel straight horizontal or vertical channels, each bounded by two parallel grey wall lines.
    - Every channel has interior width two or three pixels and contains exactly two one-pixel seeds of different colors.
    - A seed may sit near the upper/left wall, near the lower/right wall, or in the center of a width-three channel.

    Rule:
    - For each seed, compare its cross-channel offset with the channel midline.
    - A seed near the upper or left wall grows in the negative channel direction; a seed near the lower or right wall grows in the positive channel direction.
    - A centered seed does not grow.
    - The growth length is the along-channel distance to the other seed in the same channel, and the painted band uses the seed color and the full channel width.
    - Grey walls and the black background remain unchanged, and growth never paints past the paired grey walls.

    Key insight:
    - Direction comes from cross-channel offset, length from the two seeds' separation, and thickness from the channel width.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)
    wall = _wall_color(g, bg)
    channels = _find_channels(g, bg, wall)
    assert channels, "expected at least one two-seed grey channel"

    out = g.copy()
    for channel in channels:
        seeds = channel["seeds"]
        seed_positions = {(int(r), int(c)) for r, c, _ in seeds}

        for idx, (sr, sc, color) in enumerate(seeds):
            other = seeds[1 - idx]
            if channel["orientation"] == "h":
                local = int(sr - channel["top"] - 1)
                sign = _offset_sign(int(channel["width"]), local)
                if sign == 0:
                    continue
                dist = abs(int(sc) - int(other[1]))
                for step in range(dist):
                    c = int(sc + sign * step)
                    if c < channel["left"] or c > channel["right"]:
                        break
                    for r in range(int(channel["top"]) + 1, int(channel["bottom"])):
                        if (r, c) in seed_positions and (r, c) != (sr, sc):
                            continue
                        out[r, c] = int(color)
            else:
                local = int(sc - channel["left"] - 1)
                sign = _offset_sign(int(channel["width"]), local)
                if sign == 0:
                    continue
                dist = abs(int(sr) - int(other[0]))
                for step in range(dist):
                    r = int(sr + sign * step)
                    if r < channel["top"] or r > channel["bottom"]:
                        break
                    for c in range(int(channel["left"]) + 1, int(channel["right"])):
                        if (r, c) in seed_positions and (r, c) != (sr, sc):
                            continue
                        out[r, c] = int(color)

    return out


def _dir_pairs(width: int) -> list[tuple[int, int]]:
    choices = (-1, 1) if width == 2 else (-1, 0, 1)
    pairs = []
    for left_dir in choices:
        for right_dir in choices:
            if left_dir == 0 and right_dir == 0:
                continue
            if left_dir == 1 and right_dir == -1:
                continue
            pairs.append((int(left_dir), int(right_dir)))
    return pairs


def _local_from_dir(width: int, direction: int) -> int:
    if direction < 0:
        return 0
    if direction > 0:
        return int(width) - 1
    assert width == 3
    return 1


def _fits_growth(axis: int, direction: int, distance: int, length: int) -> bool:
    if direction < 0:
        return axis - distance + 1 >= 0
    if direction > 0:
        return axis + distance - 1 < length
    return True


def _choose_seed_axes(length: int, left_dir: int, right_dir: int) -> tuple[int, int] | None:
    candidates = []
    for a in range(length):
        for b in range(a + 3, length):
            distance = b - a
            if not _fits_growth(a, left_dir, distance, length):
                continue
            if not _fits_growth(b, right_dir, distance, length):
                continue
            candidates.append((int(a), int(b)))
    if not candidates:
        return None
    return random.choice(candidates)


def _clear_with_padding(reserved: np.ndarray, r0: int, r1: int, c0: int, c1: int, pad: int = 1) -> bool:
    H, W = reserved.shape
    rr0 = max(0, r0 - pad)
    rr1 = min(H - 1, r1 + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(W - 1, c1 + pad)
    return not bool(np.any(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1]))


def _reserve(reserved: np.ndarray, r0: int, r1: int, c0: int, c1: int, pad: int = 1) -> None:
    H, W = reserved.shape
    rr0 = max(0, r0 - pad)
    rr1 = min(H - 1, r1 + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(W - 1, c1 + pad)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_horizontal_channel(grid: np.ndarray, reserved: np.ndarray, colors: tuple[int, int]) -> bool:
    H, W = grid.shape
    for _ in range(CHANNEL_PLACEMENT_ATTEMPTS):
        width = random.choice((2, 3))
        block_h = width + 2
        if H < block_h or W < MIN_WALL_RUN + 3:
            return False
        length = random.randint(8, min(W, 24))
        if length > W:
            continue

        left_dir, right_dir = random.choice(_dir_pairs(width))
        axes = _choose_seed_axes(length, left_dir, right_dir)
        if axes is None:
            continue

        r0 = random.randint(0, H - block_h)
        c0 = random.randint(0, W - length)
        r1 = r0 + block_h - 1
        c1 = c0 + length - 1
        if not _clear_with_padding(reserved, r0, r1, c0, c1):
            continue

        bottom = r0 + width + 1
        grid[r0, c0 : c1 + 1] = GREY
        grid[bottom, c0 : c1 + 1] = GREY

        a, b = axes
        local_a = _local_from_dir(width, left_dir)
        local_b = _local_from_dir(width, right_dir)
        grid[r0 + 1 + local_a, c0 + a] = int(colors[0])
        grid[r0 + 1 + local_b, c0 + b] = int(colors[1])
        _reserve(reserved, r0, r1, c0, c1)
        return True
    return False


def _place_vertical_channel(grid: np.ndarray, reserved: np.ndarray, colors: tuple[int, int]) -> bool:
    H, W = grid.shape
    for _ in range(CHANNEL_PLACEMENT_ATTEMPTS):
        width = random.choice((2, 3))
        block_w = width + 2
        if W < block_w or H < MIN_WALL_RUN + 3:
            return False
        length = random.randint(8, min(H, 24))
        if length > H:
            continue

        top_dir, bottom_dir = random.choice(_dir_pairs(width))
        axes = _choose_seed_axes(length, top_dir, bottom_dir)
        if axes is None:
            continue

        r0 = random.randint(0, H - length)
        c0 = random.randint(0, W - block_w)
        r1 = r0 + length - 1
        c1 = c0 + block_w - 1
        if not _clear_with_padding(reserved, r0, r1, c0, c1):
            continue

        right = c0 + width + 1
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, right] = GREY

        a, b = axes
        local_a = _local_from_dir(width, top_dir)
        local_b = _local_from_dir(width, bottom_dir)
        grid[r0 + a, c0 + 1 + local_a] = int(colors[0])
        grid[r0 + b, c0 + 1 + local_b] = int(colors[1])
        _reserve(reserved, r0, r1, c0, c1)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(10, 30)
    W = random.randint(10, 30)
    orientation = random.choice(("h", "v"))
    channel_count = random.randint(1, 4)
    colors = random.sample(SEED_COLORS, 2 * channel_count)

    grid = np.full((H, W), BG, dtype=int)
    reserved = np.zeros((H, W), dtype=bool)
    for idx in range(channel_count):
        pair = (int(colors[2 * idx]), int(colors[2 * idx + 1]))
        if orientation == "h":
            placed = _place_horizontal_channel(grid, reserved, pair)
        else:
            placed = _place_vertical_channel(grid, reserved, pair)
        if not placed:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid channel input after {MAX_GENERATION_ATTEMPTS} attempts")
