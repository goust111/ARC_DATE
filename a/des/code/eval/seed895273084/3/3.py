from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


MARKER_COLOR = 9
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


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
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(np.array(pts, dtype=int))
    return comps


def _frame_from_component(grid: np.ndarray, coords: np.ndarray, color: int, bg: int) -> dict | None:
    if int(coords.shape[0]) < 8:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    outer_h = int(r1 - r0 + 1)
    outer_w = int(c1 - c0 + 1)
    if outer_h < 3 or outer_w < 3:
        return None
    expected = 2 * (outer_h + outer_w) - 4
    if int(coords.shape[0]) != expected:
        return None
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.ones((outer_h, outer_w), dtype=bool)
    border[1:-1, 1:-1] = False
    if not np.all(patch[border] == color):
        return None
    if not np.all(patch[1:-1, 1:-1] == bg):
        return None
    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "inner_h": int(outer_h - 2),
        "inner_w": int(outer_w - 2),
    }


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg), int(MARKER_COLOR)}):
        for comp in _components(grid == int(color)):
            frame = _frame_from_component(grid, comp, int(color), int(bg))
            if frame is not None:
                frames.append(frame)
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _find_marker_side(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    sides: list[str] = []
    if r0 - 1 >= 0 and np.any(grid[r0 - 1, c0 : c1 + 1] == MARKER_COLOR):
        sides.append("top")
    if r1 + 1 < h and np.any(grid[r1 + 1, c0 : c1 + 1] == MARKER_COLOR):
        sides.append("bottom")
    if c0 - 1 >= 0 and np.any(grid[r0 : r1 + 1, c0 - 1] == MARKER_COLOR):
        sides.append("left")
    if c1 + 1 < w and np.any(grid[r0 : r1 + 1, c1 + 1] == MARKER_COLOR):
        sides.append("right")
    assert len(sides) == 1
    return sides[0]


def _ruler_length(grid: np.ndarray, color: int) -> int:
    edge_mask = np.zeros_like(grid, dtype=bool)
    edge_mask[0, :] = grid[0, :] == color
    edge_mask[:, 0] |= grid[:, 0] == color
    return int(edge_mask.sum())


def _stripe_sizes(length: int, n_parts: int) -> list[int]:
    base, extra = divmod(int(length), int(n_parts))
    return [int(base + (1 if idx < extra else 0)) for idx in range(int(n_parts))]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use edge rulers to decide how many stripes to draw inside same-colored frames.

    Input structure:
    - Most cells are background.
    - Several hollow rectangular frames are drawn in distinct colors.
    - A top-edge or left-edge ruler segment shares each frame's color.
    - One maroon pixel sits just outside one side of each frame.

    Rule:
    - For each frame, the matching ruler length is the subdivision count n.
    - A top/bottom marker means split the interior into n vertical stripes.
    - A left/right marker means split the interior into n horizontal stripes.
    - Stripe sizes are as even as possible; earlier stripes get the remainder.
    - Fill odd-numbered stripes with the frame color, leave even ones background.
    - Erase all rulers and maroon markers.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    frames = _find_frames(g, int(bg))
    assert frames

    out[g == MARKER_COLOR] = int(bg)
    out[0, g[0, :] != bg] = int(bg)
    out[g[:, 0] != bg, 0] = int(bg)

    for frame in frames:
        color = int(frame["color"])
        r0, c0, r1, c1 = map(int, frame["bbox"])
        n = _ruler_length(g, int(color))
        assert n >= 1
        side = _find_marker_side(g, frame["bbox"])

        if side in ("top", "bottom"):
            sizes = _stripe_sizes(int(frame["inner_w"]), int(n))
            start = int(c0 + 1)
            for idx, size in enumerate(sizes):
                if idx % 2 == 0 and size > 0:
                    out[r0 + 1 : r1, start : start + size] = int(color)
                start += int(size)
        else:
            sizes = _stripe_sizes(int(frame["inner_h"]), int(n))
            start = int(r0 + 1)
            for idx, size in enumerate(sizes):
                if idx % 2 == 0 and size > 0:
                    out[start : start + size, c0 + 1 : c1] = int(color)
                start += int(size)

    return out


FRAME_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)
BG_COLORS = (0, 1, 2, 3, 4, 5)
MARKER_SIDES = ("top", "bottom", "left", "right")
RULER_SIDES = ("top", "left")


def _side_choices(n_frames: int, rng: random.Random) -> list[str]:
    sides = [rng.choice(("top", "bottom")), rng.choice(("left", "right"))]
    while len(sides) < n_frames:
        sides.append(rng.choice(MARKER_SIDES))
    rng.shuffle(sides)
    return sides


def _ruler_choices(n_frames: int, rng: random.Random) -> list[str]:
    sides = ["top", "left"]
    while len(sides) < n_frames:
        sides.append(rng.choice(RULER_SIDES))
    rng.shuffle(sides)
    return sides


def _remainder_choices(n_frames: int, rng: random.Random) -> list[bool]:
    flags = [False, True]
    while len(flags) < n_frames:
        flags.append(bool(rng.randint(0, 1)))
    rng.shuffle(flags)
    return flags


def _axis_length(n: int, needs_remainder: bool, rng: random.Random) -> int | None:
    candidates = [length for length in range(int(n), 11) if (length % int(n) != 0) == bool(needs_remainder)]
    if not candidates:
        return None
    return int(rng.choice(candidates))


def _make_specs(colors: list[int], rng: random.Random) -> list[dict] | None:
    n_frames = len(colors)
    marker_sides = _side_choices(n_frames, rng)
    ruler_sides = _ruler_choices(n_frames, rng)
    remainder_flags = _remainder_choices(n_frames, rng)

    specs: list[dict] = []
    for idx, color in enumerate(colors):
        n = int(rng.randint(2, 6))
        axis_len = _axis_length(int(n), bool(remainder_flags[idx]), rng)
        if axis_len is None:
            return None
        other_len = int(rng.randint(2, 8))
        marker_side = marker_sides[idx]
        if marker_side in ("top", "bottom"):
            inner_h, inner_w = other_len, axis_len
        else:
            inner_h, inner_w = axis_len, other_len
        specs.append(
            {
                "color": int(color),
                "n": int(n),
                "marker_side": marker_side,
                "ruler_side": ruler_sides[idx],
                "inner_h": int(inner_h),
                "inner_w": int(inner_w),
                "outer_h": int(inner_h + 2),
                "outer_w": int(inner_w + 2),
            }
        )
    return specs


def _place_frames(height: int, width: int, specs: list[dict], rng: random.Random) -> dict[int, tuple[int, int, int, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placements: dict[int, tuple[int, int, int, int]] = {}
    order = sorted(
        range(len(specs)),
        key=lambda idx: (specs[idx]["outer_h"] + 2) * (specs[idx]["outer_w"] + 2),
        reverse=True,
    )
    for idx in order:
        spec = specs[idx]
        max_r0 = int(height - spec["outer_h"] - 2)
        max_c0 = int(width - spec["outer_w"] - 2)
        if max_r0 < 2 or max_c0 < 2:
            return None
        candidates = [(r0, c0) for r0 in range(2, max_r0 + 1) for c0 in range(2, max_c0 + 1)]
        rng.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            r1 = int(r0 + spec["outer_h"] - 1)
            c1 = int(c0 + spec["outer_w"] - 1)
            if occupied[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
                continue
            occupied[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True
            placements[idx] = (int(r0), int(c0), int(r1), int(c1))
            placed = True
            break
        if not placed:
            return None
    return placements


def _place_intervals(limit: int, lengths: list[int], rng: random.Random) -> list[int] | None:
    if not lengths:
        return []
    blocked = np.zeros(limit, dtype=bool)
    starts = [0] * len(lengths)
    order = sorted(range(len(lengths)), key=lambda idx: lengths[idx], reverse=True)
    for idx in order:
        length = int(lengths[idx])
        choices: list[int] = []
        for start in range(1, limit - length + 1):
            lo = max(1, start - 1)
            hi = min(limit, start + length + 1)
            if not blocked[lo:hi].any():
                choices.append(int(start))
        if not choices:
            return None
        start = int(rng.choice(choices))
        starts[idx] = start
        lo = max(1, start - 1)
        hi = min(limit, start + length + 1)
        blocked[lo:hi] = True
    return starts


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _draw_marker(grid: np.ndarray, bbox: tuple[int, int, int, int], side: str, rng: random.Random) -> None:
    r0, c0, r1, c1 = bbox
    if side == "top":
        grid[r0 - 1, rng.randint(c0, c1)] = MARKER_COLOR
    elif side == "bottom":
        grid[r1 + 1, rng.randint(c0, c1)] = MARKER_COLOR
    elif side == "left":
        grid[rng.randint(r0, r1), c0 - 1] = MARKER_COLOR
    else:
        grid[rng.randint(r0, r1), c1 + 1] = MARKER_COLOR


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = int(rng.randint(18, 30))
    width = int(rng.randint(18, 30))
    bg = int(rng.choice(BG_COLORS))
    colors = [color for color in FRAME_COLORS if color != bg and color != MARKER_COLOR]
    n_frames = int(rng.randint(2, min(4, len(colors))))
    chosen_colors = rng.sample(colors, n_frames)

    specs = _make_specs(chosen_colors, rng)
    if specs is None:
        return None

    placements = _place_frames(height, width, specs, rng)
    if placements is None:
        return None

    top_specs = [idx for idx, spec in enumerate(specs) if spec["ruler_side"] == "top"]
    left_specs = [idx for idx, spec in enumerate(specs) if spec["ruler_side"] == "left"]
    top_starts = _place_intervals(width, [specs[idx]["n"] for idx in top_specs], rng)
    left_starts = _place_intervals(height, [specs[idx]["n"] for idx in left_specs], rng)
    if top_starts is None or left_starts is None:
        return None

    grid = np.full((height, width), bg, dtype=int)

    for idx, spec in enumerate(specs):
        _draw_frame(grid, placements[idx], int(spec["color"]))
        _draw_marker(grid, placements[idx], str(spec["marker_side"]), rng)

    for start, idx in zip(top_starts, top_specs):
        grid[0, start : start + specs[idx]["n"]] = int(specs[idx]["color"])
    for start, idx in zip(left_starts, left_specs):
        grid[start : start + specs[idx]["n"], 0] = int(specs[idx]["color"])

    counts = Counter(map(int, grid.flatten()))
    if counts[int(bg)] <= max(count for color, count in counts.items() if color != int(bg)):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
