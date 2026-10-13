from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
AZURE = 8

LEGEND_COLORS = (RED, GREEN, YELLOW, MAGENTA)
SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_frames(grid: np.ndarray) -> list[dict[str, int | tuple[int, int, int, int]]]:
    frames: list[dict[str, int | tuple[int, int, int, int]]] = []
    for coords in _components(grid == GRAY):
        r0, c0, r1, c1 = _bbox(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width < 3:
            continue
        border = set()
        for c in range(c0, c1 + 1):
            border.add((r0, c))
            border.add((r1, c))
        for r in range(r0 + 1, r1):
            border.add((r, c0))
            border.add((r, c1))
        actual = {(int(r), int(c)) for r, c in coords}
        if actual != border:
            continue
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0 or not np.all(interior == BLACK):
            continue
        frames.append({"bbox": (r0, c0, r1, c1), "r0": r0, "c0": c0, "r1": r1, "c1": c1})
    frames.sort(key=lambda item: (int(item["r0"]), int(item["c0"])))
    assert frames, "expected at least one hollow gray frame"
    return frames


def _read_legend(grid: np.ndarray) -> tuple[dict[int, int], set[int]]:
    h, _w = grid.shape
    legend_entries: list[tuple[int, int]] = []
    for r in range(h):
        value = int(grid[r, 0])
        if value not in (BLACK, AZURE, GRAY):
            legend_entries.append((r, value))
    assert len(legend_entries) == 4, "expected four legend pixels in the leftmost column"

    legend_rows = {r for r, _value in legend_entries}
    for r, value in legend_entries:
        assert all(int(grid[rr, 0]) == BLACK for rr in (r - 1, r + 1) if 0 <= rr < h and rr not in legend_rows), (
            "legend pixels must be isolated by black gaps"
        )
        assert int(value) in LEGEND_COLORS
    ordered = [value for _r, value in sorted(legend_entries)]
    return {idx + 1: int(color) for idx, color in enumerate(ordered)}, legend_rows


def _edge_sources(grid: np.ndarray) -> dict[str, list[int]]:
    h, w = grid.shape
    return {
        "top": [c for c in range(w) if int(grid[0, c]) == AZURE],
        "bottom": [c for c in range(w) if int(grid[h - 1, c]) == AZURE],
        "left": [r for r in range(h) if int(grid[r, 0]) == AZURE],
        "right": [r for r in range(h) if int(grid[r, w - 1]) == AZURE],
    }


def _accessible_side_coords(
    frames: list[dict[str, int | tuple[int, int, int, int]]],
    h: int,
    w: int,
    blocked_left_rows: set[int] | None = None,
) -> list[dict[str, list[int]]]:
    blocked_left_rows = blocked_left_rows or set()
    side_coords = [{side: [] for side in SIDES} for _ in frames]

    for c in range(1, w - 1):
        candidates = [idx for idx, frame in enumerate(frames) if int(frame["c0"]) <= c <= int(frame["c1"])]
        if candidates:
            top_idx = min(candidates, key=lambda idx: int(frames[idx]["r0"]))
            bottom_idx = max(candidates, key=lambda idx: int(frames[idx]["r1"]))
            side_coords[top_idx]["top"].append(c)
            side_coords[bottom_idx]["bottom"].append(c)

    for r in range(1, h - 1):
        candidates = [idx for idx, frame in enumerate(frames) if int(frame["r0"]) <= r <= int(frame["r1"])]
        if candidates:
            left_idx = min(candidates, key=lambda idx: int(frames[idx]["c0"]))
            right_idx = max(candidates, key=lambda idx: int(frames[idx]["c1"]))
            if r not in blocked_left_rows:
                side_coords[left_idx]["left"].append(r)
            side_coords[right_idx]["right"].append(r)

    return side_coords


def _illuminated_sides(
    frames: list[dict[str, int | tuple[int, int, int, int]]],
    h: int,
    w: int,
    sources: dict[str, list[int]],
) -> list[set[str]]:
    illuminated = [set() for _ in frames]

    for c in sources["top"]:
        candidates = [idx for idx, frame in enumerate(frames) if int(frame["c0"]) <= c <= int(frame["c1"])]
        if candidates:
            hit = min(candidates, key=lambda idx: int(frames[idx]["r0"]))
            illuminated[hit].add("top")

    for c in sources["bottom"]:
        candidates = [idx for idx, frame in enumerate(frames) if int(frame["c0"]) <= c <= int(frame["c1"])]
        if candidates:
            hit = max(candidates, key=lambda idx: int(frames[idx]["r1"]))
            illuminated[hit].add("bottom")

    for r in sources["left"]:
        candidates = [idx for idx, frame in enumerate(frames) if int(frame["r0"]) <= r <= int(frame["r1"])]
        if candidates:
            hit = min(candidates, key=lambda idx: int(frames[idx]["c0"]))
            illuminated[hit].add("left")

    for r in sources["right"]:
        candidates = [idx for idx, frame in enumerate(frames) if int(frame["r0"]) <= r <= int(frame["r1"])]
        if candidates:
            hit = max(candidates, key=lambda idx: int(frames[idx]["c1"]))
            illuminated[hit].add("right")

    return illuminated


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _find_frames(g)
    legend, _legend_rows = _read_legend(g)
    sources = _edge_sources(g)
    illuminated = _illuminated_sides(frames, *g.shape, sources)

    out = g.copy()
    for frame, sides in zip(frames, illuminated):
        k = len(sides)
        if k == 0:
            continue
        color = legend[k]
        r0, c0, r1, c1 = map(int, frame["bbox"])
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == BLACK] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow gray frame according to how many of its sides are hit first by inward edge rays.
    The leftmost-column legend maps side count k=1..4 to a fill color. Unlit frames stay black inside.
    """
    return _solve(grid)


def _can_place(rect: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]], h: int, w: int) -> bool:
    r0, c0, r1, c1 = rect
    if r0 < 2 or c0 < 2 or r1 > h - 3 or c1 > w - 3:
        return False
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return False
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + 1 < pr0 or pr1 + 1 < r0 or c1 + 1 < pc0 or pc1 + 1 < c0):
            return False
    return True


def _place_independent(h: int, w: int, rng: random.Random) -> list[tuple[int, int, int, int]] | None:
    frames: list[tuple[int, int, int, int]] = []
    mid_r = h // 2
    mid_c = w // 2
    zones = [
        (2, mid_r - 3, 3, mid_c - 3),
        (2, mid_r - 3, mid_c + 1, w - 4),
        (mid_r + 1, h - 4, 3, mid_c - 3),
        (mid_r + 1, h - 4, mid_c + 1, w - 4),
    ]
    for zr0, zr1, zc0, zc1 in zones:
        if zr1 - zr0 + 1 < 3 or zc1 - zc0 + 1 < 3:
            return None
        for _ in range(60):
            height = rng.randint(3, min(6, zr1 - zr0 + 1))
            width = rng.randint(3, min(7, zc1 - zc0 + 1))
            r0 = rng.randint(zr0, zr1 - height + 1)
            c0 = rng.randint(zc0, zc1 - width + 1)
            rect = (r0, c0, r0 + height - 1, c0 + width - 1)
            if _can_place(rect, frames, h, w):
                frames.append(rect)
                break
        else:
            return None
    return frames


def _place_vertical_stack(h: int, w: int, rng: random.Random) -> list[tuple[int, int, int, int]] | None:
    heights = [rng.randint(3, 5) for _ in range(3)]
    gap1 = rng.randint(2, 4)
    gap2 = rng.randint(2, 4)
    total_h = sum(heights) + gap1 + gap2
    if total_h > h - 4:
        return None

    stack_w = rng.randint(4, min(7, w - 8))
    c0 = rng.randint(max(3, w // 2 - 3), min(w - stack_w - 4, w // 2 + 1))
    top_r0 = rng.randint(2, h - total_h - 2)
    mid_r0 = top_r0 + heights[0] + gap1
    bot_r0 = mid_r0 + heights[1] + gap2
    frames = [
        (top_r0, c0, top_r0 + heights[0] - 1, c0 + stack_w - 1),
        (mid_r0, c0, mid_r0 + heights[1] - 1, c0 + stack_w - 1),
        (bot_r0, c0, bot_r0 + heights[2] - 1, c0 + stack_w - 1),
    ]

    side_h = rng.randint(3, 6)
    side_w = rng.randint(3, 6)
    overlap = rng.random() < 0.5
    for _ in range(80):
        if overlap:
            sr0 = rng.randint(mid_r0, max(mid_r0, min(bot_r0 + heights[2] - side_h, h - side_h - 3)))
        else:
            if rng.random() < 0.5:
                upper_limit = top_r0 - side_h - 2
                if upper_limit < 2:
                    continue
                sr0 = rng.randint(2, upper_limit)
            else:
                lower_limit = bot_r0 + heights[2] + 2
                if lower_limit > h - side_h - 3:
                    continue
                sr0 = rng.randint(lower_limit, h - side_h - 3)
        if rng.random() < 0.5:
            left_limit = c0 - side_w - 2
            if left_limit < 3:
                continue
            sc0 = rng.randint(3, left_limit)
        else:
            right_limit = c0 + stack_w + 2
            if right_limit > w - side_w - 3:
                continue
            sc0 = rng.randint(right_limit, w - side_w - 3)
        rect = (sr0, sc0, sr0 + side_h - 1, sc0 + side_w - 1)
        if _can_place(rect, frames, h, w):
            frames.append(rect)
            return frames
    return None


def _place_horizontal_stack(h: int, w: int, rng: random.Random) -> list[tuple[int, int, int, int]] | None:
    widths = [rng.randint(3, 6) for _ in range(3)]
    gap1 = rng.randint(2, 4)
    gap2 = rng.randint(2, 4)
    total_w = sum(widths) + gap1 + gap2
    if total_w > w - 6:
        return None

    stack_h = rng.randint(4, min(7, h - 8))
    r0 = rng.randint(max(2, h // 2 - 3), min(h - stack_h - 3, h // 2 + 1))
    left_c0 = rng.randint(3, w - total_w - 3)
    mid_c0 = left_c0 + widths[0] + gap1
    right_c0 = mid_c0 + widths[1] + gap2
    frames = [
        (r0, left_c0, r0 + stack_h - 1, left_c0 + widths[0] - 1),
        (r0, mid_c0, r0 + stack_h - 1, mid_c0 + widths[1] - 1),
        (r0, right_c0, r0 + stack_h - 1, right_c0 + widths[2] - 1),
    ]

    side_h = rng.randint(3, 6)
    side_w = rng.randint(3, 6)
    overlap = rng.random() < 0.5
    for _ in range(80):
        if overlap:
            sc0 = rng.randint(mid_c0, max(mid_c0, min(right_c0 + widths[2] - side_w, w - side_w - 3)))
            if rng.random() < 0.5:
                upper_limit = r0 - side_h - 2
                if upper_limit < 2:
                    continue
                sr0 = rng.randint(2, upper_limit)
            else:
                lower_limit = r0 + stack_h + 2
                if lower_limit > h - side_h - 3:
                    continue
                sr0 = rng.randint(lower_limit, h - side_h - 3)
        else:
            if rng.random() < 0.5:
                left_limit = left_c0 - side_w - 2
                if left_limit < 3:
                    continue
                sc0 = rng.randint(3, left_limit)
            else:
                right_limit = right_c0 + widths[2] + 2
                if right_limit > w - side_w - 3:
                    continue
                sc0 = rng.randint(right_limit, w - side_w - 3)
            sr0 = rng.randint(2, h - side_h - 3)
        rect = (sr0, sc0, sr0 + side_h - 1, sc0 + side_w - 1)
        if _can_place(rect, frames, h, w):
            frames.append(rect)
            return frames
    return None


def _place_cross(h: int, w: int, rng: random.Random) -> list[tuple[int, int, int, int]] | None:
    ch = rng.randint(3, 5)
    cw = rng.randint(3, 6)
    th = rng.randint(3, 5)
    tw = rng.randint(max(3, cw - 1), min(7, cw + 2))
    bh = rng.randint(3, 5)
    bw = rng.randint(max(3, cw - 1), min(7, cw + 2))
    lh = rng.randint(max(3, ch - 1), min(7, ch + 2))
    lw = rng.randint(3, 5)
    rh = rng.randint(max(3, ch - 1), min(7, ch + 2))
    rw = rng.randint(3, 5)
    gap_t = rng.randint(2, 4)
    gap_b = rng.randint(2, 4)
    gap_l = rng.randint(2, 4)
    gap_r = rng.randint(2, 4)

    min_center_r0 = th + gap_t + 2
    max_center_r0 = h - (ch + bh + gap_b) - 3
    min_center_c0 = lw + gap_l + 3
    max_center_c0 = w - (cw + rw + gap_r) - 4
    if min_center_r0 > max_center_r0 or min_center_c0 > max_center_c0:
        return None

    cr0 = rng.randint(min_center_r0, max_center_r0)
    cc0 = rng.randint(min_center_c0, max_center_c0)
    center = (cr0, cc0, cr0 + ch - 1, cc0 + cw - 1)

    top_r0 = cr0 - gap_t - th
    bottom_r0 = cr0 + ch + gap_b
    left_c0 = cc0 - gap_l - lw
    right_c0 = cc0 + cw + gap_r

    top_c0_min = max(3, cc0 - tw + 1)
    top_c0_max = min(w - tw - 3, cc0 + cw - 1)
    bottom_c0_min = max(3, cc0 - bw + 1)
    bottom_c0_max = min(w - bw - 3, cc0 + cw - 1)
    left_r0_min = max(2, cr0 - lh + 1)
    left_r0_max = min(h - lh - 3, cr0 + ch - 1)
    right_r0_min = max(2, cr0 - rh + 1)
    right_r0_max = min(h - rh - 3, cr0 + ch - 1)
    if (
        top_c0_min > top_c0_max
        or bottom_c0_min > bottom_c0_max
        or left_r0_min > left_r0_max
        or right_r0_min > right_r0_max
    ):
        return None

    top = (top_r0, rng.randint(top_c0_min, top_c0_max), top_r0 + th - 1, 0)
    top = (top[0], top[1], top[2], top[1] + tw - 1)
    bottom = (bottom_r0, rng.randint(bottom_c0_min, bottom_c0_max), bottom_r0 + bh - 1, 0)
    bottom = (bottom[0], bottom[1], bottom[2], bottom[1] + bw - 1)
    left = (rng.randint(left_r0_min, left_r0_max), left_c0, 0, left_c0 + lw - 1)
    left = (left[0], left[1], left[0] + lh - 1, left[3])
    right = (rng.randint(right_r0_min, right_r0_max), right_c0, 0, right_c0 + rw - 1)
    right = (right[0], right[1], right[0] + rh - 1, right[3])

    frames = [center]
    for rect in (top, bottom, left, right):
        if not _can_place(rect, frames, h, w):
            return None
        frames.append(rect)
    return frames


def _choose_targets(side_coords: list[dict[str, list[int]]], rng: random.Random) -> list[int] | None:
    capacities = [sum(1 for side in SIDES if coords[side]) for coords in side_coords]
    frame_count = len(capacities)

    if frame_count >= 5:
        profiles = [
            [4, 3, 2, 1, 0],
            [3, 2, 1, 0, 0],
            [4, 2, 1, 0, 0],
        ]
    elif frame_count == 4:
        profiles = [
            [4, 3, 2, 1],
            [3, 2, 1, 0],
            [4, 2, 1, 0],
        ]
    else:
        profiles = [
            [3, 2, 1],
            [2, 1, 0],
        ]
    rng.shuffle(profiles)

    def fit_profile(profile: list[int]) -> list[int] | None:
        targets: list[int | None] = [None] * frame_count
        free = set(range(frame_count))
        for k in sorted(profile, reverse=True):
            candidates = [idx for idx in free if capacities[idx] >= k]
            if not candidates:
                return None
            min_cap = min(capacities[idx] for idx in candidates)
            pick_pool = [idx for idx in candidates if capacities[idx] == min_cap]
            pick = rng.choice(pick_pool)
            targets[pick] = k
            free.remove(pick)
        for idx in free:
            targets[idx] = rng.randint(0, capacities[idx])
        return [int(value) for value in targets]

    for profile in profiles:
        fitted = fit_profile(profile)
        if fitted is not None and len(set(fitted)) >= min(3, frame_count):
            return fitted

    if max(capacities) == 0:
        return None
    return [rng.randint(0, cap) for cap in capacities]


def _draw_frame(grid: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = rect
    grid[r0, c0 : c1 + 1] = GRAY
    grid[r1, c0 : c1 + 1] = GRAY
    grid[r0 : r1 + 1, c0] = GRAY
    grid[r0 : r1 + 1, c1] = GRAY


def _stamp_sources(
    grid: np.ndarray,
    side_coords: list[dict[str, list[int]]],
    targets: list[int],
    rng: random.Random,
) -> list[set[str]] | None:
    chosen_sides: list[set[str]] = []
    for idx, target in enumerate(targets):
        available = [side for side in SIDES if side_coords[idx][side]]
        if target > len(available):
            return None
        picked = set(rng.sample(available, target))
        chosen_sides.append(picked)

        for side in picked:
            pos = rng.choice(side_coords[idx][side])
            if side == "top":
                grid[0, pos] = AZURE
            elif side == "bottom":
                grid[grid.shape[0] - 1, pos] = AZURE
            elif side == "left":
                grid[pos, 0] = AZURE
            else:
                grid[pos, grid.shape[1] - 1] = AZURE
    return chosen_sides


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    template = rng.choice(("independent", "vertical", "horizontal", "cross"))
    if template == "cross":
        h = rng.randint(18, 24)
        w = rng.randint(19, 26)
    else:
        h = rng.randint(16, 23)
        w = rng.randint(17, 25)

    grid = np.full((h, w), BLACK, dtype=int)
    legend_start = rng.randint(1, h - 8)
    legend_rows = [legend_start + 2 * idx for idx in range(4)]
    blocked_left_rows = set(legend_rows)
    for r in legend_rows:
        if r - 1 >= 0:
            blocked_left_rows.add(r - 1)
        if r + 1 < h:
            blocked_left_rows.add(r + 1)
    legend_colors = list(LEGEND_COLORS)
    rng.shuffle(legend_colors)
    for r, color in zip(legend_rows, legend_colors):
        grid[r, 0] = color

    if template == "independent":
        frames = _place_independent(h, w, rng)
    elif template == "vertical":
        frames = _place_vertical_stack(h, w, rng)
    elif template == "horizontal":
        frames = _place_horizontal_stack(h, w, rng)
    else:
        frames = _place_cross(h, w, rng)
    if frames is None:
        return None

    for rect in frames:
        _draw_frame(grid, rect)

    frame_dicts = [
        {"bbox": rect, "r0": rect[0], "c0": rect[1], "r1": rect[2], "c1": rect[3]}
        for rect in sorted(frames)
    ]
    side_coords = _accessible_side_coords(frame_dicts, h, w, blocked_left_rows)
    targets = _choose_targets(side_coords, rng)
    if targets is None:
        return None

    chosen_sides = _stamp_sources(grid, side_coords, targets, rng)
    if chosen_sides is None:
        return None

    actual_frames = _find_frames(grid)
    sources = _edge_sources(grid)
    actual_sides = _illuminated_sides(actual_frames, h, w, sources)
    actual_counts = [len(sides) for sides in actual_sides]
    if sorted(actual_counts) != sorted(targets):
        return None
    if len(set(actual_counts)) < 3:
        return None
    if not any(count == 0 for count in actual_counts) and rng.random() < 0.2:
        return None
    if not any(count == 4 for count in actual_counts) and template == "independent":
        return None

    out = _solve(grid)
    if not np.any(out != grid):
        return None
    if not any(chosen_sides):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
