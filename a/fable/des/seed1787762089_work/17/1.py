from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
SEPARATOR_COLOR = 5
VISIBLE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_RETRIES = 400
FRAME_RETRIES = 120
PLACEMENT_RETRIES = 240


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(comp)
    return parts


def _interior_bg_components(subgrid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = subgrid.shape
    rooms: list[list[tuple[int, int]]] = []
    for comp in _components(subgrid == BG):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        rooms.append(comp)
    return rooms


def _find_separator_row(grid: np.ndarray, bg: int) -> int:
    h = grid.shape[0]
    for r in range(1, h):
        row = grid[r]
        if int(row[0]) != bg and np.all(row == row[0]):
            return r
    raise AssertionError("separator row not found")


def _read_legend(grid: np.ndarray, legend_row: int, bg: int) -> list[int]:
    cols = [c for c in range(grid.shape[1]) if int(grid[legend_row, c]) != bg]
    assert 4 <= len(cols) <= 5, "legend must contain 4 or 5 entries"
    assert all(cols[i + 1] - cols[i] >= 2 for i in range(len(cols) - 1)), "legend cells must be separated"
    return [int(grid[legend_row, c]) for c in cols]


def _extract_frames(grid: np.ndarray, separator_row: int, bg: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    sub = grid[separator_row + 1 :]
    for comp in _components(sub != bg):
        coords = [(r + separator_row + 1, c) for r, c in comp]
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "each frame wall must be monochrome"
        wall_color = colors.pop()
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        frame = grid[r0 : r1 + 1, c0 : c1 + 1]
        rooms = _interior_bg_components(frame)
        frames.append(
            {
                "bbox": (r0, r1, c0, c1),
                "wall_color": wall_color,
                "room_count": len(rooms),
                "rooms": rooms,
            }
        )
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected 2D grid"

    bg = int(np.bincount(g.reshape(-1)).argmax())
    separator_row = _find_separator_row(g, bg)
    legend_colors = _read_legend(g, separator_row - 1, bg)
    n = len(legend_colors)

    frames = _extract_frames(g, separator_row, bg)
    used_counts = {int(frame["room_count"]) for frame in frames}
    assert len(frames) == n - 1, "frame count must be one fewer than legend count"
    assert used_counts <= set(range(1, n + 1)), "room counts must index the legend"
    assert len(used_counts) == len(frames), "room counts must be unique"

    missing = [k for k in range(1, n + 1) if k not in used_counts]
    assert len(missing) == 1, "exactly one legend index must be unused"
    unused_color = legend_colors[missing[0] - 1]

    out = g.copy()
    distractors = 0
    for frame in frames:
        r0, r1, c0, c1 = frame["bbox"]  # type: ignore[misc]
        wall_color = int(frame["wall_color"])
        room_count = int(frame["room_count"])
        if wall_color == unused_color:
            out[r0 : r1 + 1, c0 : c1 + 1] = bg
            distractors += 1
            continue
        fill_color = legend_colors[room_count % n]
        for room in frame["rooms"]:  # type: ignore[assignment]
            for lr, lc in room:
                out[r0 + lr, c0 + lc] = fill_color

    assert distractors == 1, "there must be exactly one distractor frame"
    return out


def _room_count_from_walls(wall_mask: np.ndarray) -> int:
    return len(_interior_bg_components(np.where(wall_mask, 1, 0)))


def _build_frame(room_count: int, max_h: int, max_w: int, rng: random.Random) -> np.ndarray | None:
    dims: list[tuple[int, int]] = []
    for fh in range(4, max_h + 1):
        for fw in range(4, max_w + 1):
            inner_h = fh - 2
            inner_w = fw - 2
            if room_count == 1:
                if inner_h * inner_w >= 4:
                    dims.append((fh, fw))
                continue
            if inner_h * inner_w < 2 * room_count - 1:
                continue
            if inner_h < 2 or inner_w < 2:
                continue
            if inner_h < 3 and inner_w < 3:
                continue
            dims.append((fh, fw))
    rng.shuffle(dims)

    for fh, fw in dims:
        for _ in range(FRAME_RETRIES):
            wall = np.zeros((fh, fw), dtype=bool)
            wall[0, :] = True
            wall[-1, :] = True
            wall[:, 0] = True
            wall[:, -1] = True
            rooms = [(1, fh - 2, 1, fw - 2)]

            for _ in range(room_count - 1):
                options: list[tuple[int, str]] = []
                for idx, (r0, r1, c0, c1) in enumerate(rooms):
                    if r1 - r0 + 1 >= 3:
                        options.append((idx, "h"))
                    if c1 - c0 + 1 >= 3:
                        options.append((idx, "v"))
                if not options:
                    break
                idx, orient = rng.choice(options)
                r0, r1, c0, c1 = rooms.pop(idx)
                if orient == "h":
                    wall_r = rng.randint(r0 + 1, r1 - 1)
                    wall[wall_r, c0 : c1 + 1] = True
                    rooms.append((r0, wall_r - 1, c0, c1))
                    rooms.append((wall_r + 1, r1, c0, c1))
                else:
                    wall_c = rng.randint(c0 + 1, c1 - 1)
                    wall[r0 : r1 + 1, wall_c] = True
                    rooms.append((r0, r1, c0, wall_c - 1))
                    rooms.append((r0, r1, wall_c + 1, c1))
            else:
                rot = rng.randrange(4)
                if rot:
                    wall = np.rot90(wall, rot)
                if wall.shape[0] > max_h or wall.shape[1] > max_w:
                    continue
                if _room_count_from_walls(wall) == room_count:
                    return wall
    return None


def _place_frames(
    h: int, w: int, frames: list[dict[str, object]], rng: random.Random
) -> list[dict[str, object]] | None:
    blocked = np.zeros((h, w), dtype=bool)
    blocked[:3, :] = True
    placed: list[dict[str, object]] = []

    ordered = sorted(frames, key=lambda frame: int(np.prod(frame["wall_mask"].shape)), reverse=True)
    for frame in ordered:
        wall_mask = frame["wall_mask"]  # type: ignore[assignment]
        fh, fw = wall_mask.shape
        if h - fh - 2 < 3 or w - fw - 2 < 1:
            return None
        for _ in range(PLACEMENT_RETRIES):
            top = rng.randint(3, h - fh - 2)
            left = rng.randint(1, w - fw - 2)
            bottom = top + fh - 1
            right = left + fw - 1
            rr0 = max(0, top - 1)
            rr1 = min(h, bottom + 2)
            cc0 = max(0, left - 1)
            cc1 = min(w, right + 2)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            placed_frame = dict(frame)
            placed_frame["top"] = top
            placed_frame["left"] = left
            placed.append(placed_frame)
            blocked[rr0:rr1, cc0:cc1] = True
            break
        else:
            return None
    return placed


def _render_case(
    h: int,
    w: int,
    legend_colors: list[int],
    legend_start: int,
    missing_index: int,
    frames: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros((h, w), dtype=int)
    grid[1, :] = SEPARATOR_COLOR
    for i, color in enumerate(legend_colors):
        grid[0, legend_start + 2 * i] = color

    for frame in frames:
        top = int(frame["top"])
        left = int(frame["left"])
        wall_color = int(frame["wall_color"])
        wall_mask = frame["wall_mask"]  # type: ignore[assignment]
        fh, fw = wall_mask.shape
        patch = grid[top : top + fh, left : left + fw]
        patch[wall_mask] = wall_color

    out = grid.copy()
    n = len(legend_colors)
    unused_color = legend_colors[missing_index - 1]
    for frame in frames:
        top = int(frame["top"])
        left = int(frame["left"])
        wall_color = int(frame["wall_color"])
        room_count = int(frame["room_count"])
        wall_mask = frame["wall_mask"]  # type: ignore[assignment]
        fh, fw = wall_mask.shape
        patch = out[top : top + fh, left : left + fw]
        if wall_color == unused_color:
            patch[:] = BG
        else:
            patch[~wall_mask] = legend_colors[room_count % n]
    return grid, out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.choice((4, 5))
    h_min = 18 if n == 4 else 20
    w_min = 18 if n == 4 else 20
    h = rng.randint(h_min, 24)
    w = rng.randint(w_min, 24)

    legend_colors = rng.sample(VISIBLE_COLORS, n)
    if rng.random() < 0.5:
        missing_index = n
    else:
        missing_index = rng.randint(1, n - 1)

    room_counts = [k for k in range(1, n + 1) if k != missing_index]
    rng.shuffle(room_counts)

    wall_pool = [color for color in VISIBLE_COLORS if color not in legend_colors]
    frames: list[dict[str, object]] = []

    max_h = min(8 if n == 5 else 10, h - 5)
    max_w = min(8 if n == 5 else 10, w - 4)
    if max_h < 4 or max_w < 4:
        return None

    for room_count in room_counts:
        wall_mask = _build_frame(room_count, max_h, max_w, rng)
        if wall_mask is None:
            return None
        frames.append({"room_count": room_count, "wall_mask": wall_mask})

    distractor_idx = rng.randrange(len(frames))
    other_colors = rng.sample(wall_pool, len(frames) - 1)
    other_iter = iter(other_colors)
    for idx, frame in enumerate(frames):
        if idx == distractor_idx:
            frame["wall_color"] = legend_colors[missing_index - 1]
        else:
            frame["wall_color"] = next(other_iter)

    placed = _place_frames(h, w, frames, rng)
    if placed is None:
        return None

    legend_width = 2 * n - 1
    legend_start = rng.randint(0, w - legend_width)
    grid, expected = _render_case(h, w, legend_colors, legend_start, missing_index, placed)

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
