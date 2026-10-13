import random
from itertools import combinations

import numpy as np


BG = 0
BAR = 5
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
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
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _detect_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, BAR)):
        for coords in _components_of_color(grid, color):
            r0, c0, r1, c1 = _bbox(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 4 or w < 4:
                continue
            if len(coords) != 2 * (h + w) - 4:
                continue
            local = np.zeros((h, w), dtype=bool)
            local[coords[:, 0] - r0, coords[:, 1] - c0] = True
            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.array_equal(local, border):
                continue
            frames.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                }
            )
    frames.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], item["color"]))
    return frames


def _detect_bars(grid: np.ndarray) -> list[dict]:
    bars: list[dict] = []
    for coords in _components_of_color(grid, BAR):
        r0, c0, r1, c1 = _bbox(coords)
        if c0 != c1:
            continue
        if len(coords) != r1 - r0 + 1:
            continue
        bars.append(
            {
                "col": c0,
                "top": r0,
                "height": r1 - r0 + 1,
            }
        )
    bars.sort(key=lambda item: (item["col"], item["top"]))
    return bars


def _docked_bar_counts(grid: np.ndarray, frames: list[dict], bars: list[dict]) -> list[int]:
    lookup: dict[tuple[int, int], int] = {}
    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        for c in range(c0, c1 + 1):
            lookup[(r1, c)] = idx

    counts = [0] * len(frames)
    for bar in bars:
        col = int(bar["col"])
        hit_row = None
        for r in range(int(bar["top"]) - 1, -1, -1):
            value = int(grid[r, col])
            if value not in (BG, BAR):
                hit_row = r
                break
        if hit_row is None:
            continue
        idx = lookup.get((int(hit_row), col))
        if idx is not None:
            counts[idx] += 1
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide grey vertical bars upward to the hollow frames and fill each frame interior from the docking-count parity.

    Input structure:
    - Black background contains 2-3 separated hollow rectangular frames of different non-grey colors in the upper area.
    - Lower rows contain several grey vertical bars, each occupying one column and having height 1-3.
    - Frame horizontal spans do not overlap, so a bar column can be matched to at most one frame above it.

    Rule:
    - Treat every grey connected component as one vertical bar and scan upward in its column.
    - If the first non-black non-grey pixel above the bar is a frame bottom-edge pixel, that bar docks under that frame; otherwise erase the bar.
    - Count docked bars per frame. Odd counts fill the frame interior with a top-left-anchored checkerboard in the frame color; even positive counts fill the entire interior solidly; zero leaves the interior black.
    - Keep the frame outlines unchanged and remove all grey bars from the final output.

    Key insight:
    - The visible fill pattern is controlled by a quantity produced by motion: sliding determines which frame each bar reaches, and the parity of that docking count selects checkerboard versus solid fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _detect_frames(g)
    bars = _detect_bars(g)
    counts = _docked_bar_counts(g, frames, bars)

    out = g.copy()
    out[out == BAR] = BG

    for frame, count in zip(frames, counts):
        if count == 0:
            continue
        color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]
        if count % 2 == 0:
            out[r0 + 1 : r1, c0 + 1 : c1] = color
            continue
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if ((r - (r0 + 1)) + (c - (c0 + 1))) % 2 == 0:
                    out[r, c] = color
                else:
                    out[r, c] = BG

    return out


def _sample_counts(frame_count: int) -> list[int]:
    counts = [random.choice((1, 3)), random.choice((2, 4))]
    if frame_count == 3:
        counts.append(random.randint(1, 4))
    random.shuffle(counts)
    return counts


def _sample_bar_offsets(width: int, count: int) -> list[int]:
    options = [combo for combo in combinations(range(width), count) if all(b - a >= 2 for a, b in zip(combo, combo[1:]))]
    return list(random.choice(options))


def _expected_output(scene: dict) -> np.ndarray:
    out = scene["input"].copy()
    out[out == BAR] = BG
    for frame in scene["frames"]:
        color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]
        count = int(frame["count"])
        if count == 0:
            continue
        if count % 2 == 0:
            out[r0 + 1 : r1, c0 + 1 : c1] = color
        else:
            for r in range(r0 + 1, r1):
                for c in range(c0 + 1, c1):
                    out[r, c] = color if ((r - (r0 + 1)) + (c - (c0 + 1))) % 2 == 0 else BG
    return out


def _scene_semantics_ok(scene: dict) -> bool:
    grid = scene["input"]
    frames = _detect_frames(grid)
    bars = _detect_bars(grid)
    expected_frames = sorted(
        [(tuple(frame["bbox"]), int(frame["color"]), int(frame["count"])) for frame in scene["frames"]],
        key=lambda item: (item[0][1], item[0][0], item[1]),
    )
    actual_frames = sorted(
        [(tuple(frame["bbox"]), int(frame["color"]), 0) for frame in frames],
        key=lambda item: (item[0][1], item[0][0], item[1]),
    )
    if [(bbox, color) for bbox, color, _ in expected_frames] != [(bbox, color) for bbox, color, _ in actual_frames]:
        return False

    counts = _docked_bar_counts(grid, frames, bars)
    if counts != [int(frame["count"]) for frame in scene["frames"]]:
        return False

    if not any(count % 2 == 0 for count in counts):
        return False
    if not any(count % 2 == 1 for count in counts):
        return False

    heights = {int(bar["height"]) for bar in bars}
    if len(heights) < 2:
        return False

    for frame in scene["frames"]:
        r0, c0, r1, c1 = frame["bbox"]
        color = int(frame["color"])
        if not np.all(grid[r0, c0 : c1 + 1] == color):
            return False
        if not np.all(grid[r1, c0 : c1 + 1] == color):
            return False
        if not np.all(grid[r0 : r1 + 1, c0] == color):
            return False
        if not np.all(grid[r0 : r1 + 1, c1] == color):
            return False
        if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != BG):
            return False

    for bar in scene["bars"]:
        col = int(bar["col"])
        top = int(bar["top"])
        height = int(bar["height"])
        owner = int(bar["frame_index"])
        frame = scene["frames"][owner]
        if not (frame["bbox"][1] <= col <= frame["bbox"][3]):
            return False
        if top <= max(grid.shape[0] // 2, frame["bbox"][2] + 1):
            return False
        if not np.all(grid[top : top + height, col] == BAR):
            return False
        above = grid[frame["bbox"][2] + 1 : top, col]
        if np.any(above != BG):
            return False
        for r in range(top - 1, -1, -1):
            if grid[r, col] not in (BG, BAR):
                if r != frame["bbox"][2] or int(grid[r, col]) != int(frame["color"]):
                    return False
                break

    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, _expected_output(scene))


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choice((2, 3))
    counts = _sample_counts(frame_count)
    colors = random.sample(FRAME_COLORS, frame_count)

    widths: list[int] = []
    heights: list[int] = []
    for count in counts:
        min_width = max(4, 2 * count - 1)
        widths.append(random.randint(min_width, min(9, min_width + 2)))
        heights.append(random.randint(4, 7))

    base_gaps = [random.randint(2, 4) for _ in range(frame_count - 1)]
    side_margins = [random.randint(1, 3), random.randint(1, 3)]
    min_width_total = side_margins[0] + sum(widths) + sum(base_gaps) + side_margins[1]
    if min_width_total > 30:
        return None
    width = random.randint(max(18, min_width_total), min(30, min_width_total + 4))

    height = random.randint(max(18, max(heights) + 11), 24)
    if height > 30:
        return None

    extra = width - min_width_total
    slack = [0] * (frame_count + 1)
    for _ in range(extra):
        slack[random.randrange(len(slack))] += 1

    frames: list[dict] = []
    c = side_margins[0] + slack[0]
    for idx in range(frame_count):
        w = widths[idx]
        h = heights[idx]
        top_max = min(6, height // 2 - h - 1)
        if top_max < 1:
            return None
        r0 = random.randint(1, top_max)
        frame = {
            "bbox": (r0, c, r0 + h - 1, c + w - 1),
            "color": colors[idx],
            "count": counts[idx],
        }
        frames.append(frame)
        c += w
        if idx < frame_count - 1:
            c += base_gaps[idx] + slack[idx + 1]

    grid = np.zeros((height, width), dtype=int)
    bars: list[dict] = []
    sampled_heights: list[int] = []

    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        color = int(frame["color"])
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color

        offsets = _sample_bar_offsets(c1 - c0 + 1, int(frame["count"]))
        for offset in offsets:
            bar_height = random.randint(1, 3)
            min_top = max(height // 2 + 1, r1 + 4)
            max_top = height - bar_height
            if min_top > max_top:
                return None
            top = random.randint(min_top, max_top)
            col = c0 + offset
            grid[top : top + bar_height, col] = BAR
            bars.append(
                {
                    "frame_index": idx,
                    "col": col,
                    "top": top,
                    "height": bar_height,
                }
            )
            sampled_heights.append(bar_height)

    if len(set(sampled_heights)) < 2:
        return None

    scene = {"input": grid, "frames": frames, "bars": bars}
    if not _scene_semantics_ok(scene):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
