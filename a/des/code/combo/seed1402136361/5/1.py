from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False

    pts = set(coords)
    if len(pts) != 2 * (height + width) - 4:
        return False

    for c in range(c0, c1 + 1):
        if (r0, c) not in pts or (r1, c) not in pts:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in pts or (r, c1) not in pts:
            return False
    return True


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        for comp in _components(grid == color):
            if _is_rect_frame(comp):
                frames.append({"color": int(color), "bbox": _bbox(comp)})
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _mode_color(values: np.ndarray) -> int:
    flat = [int(v) for v in np.ravel(values)]
    counts = Counter(flat)
    first_pos: dict[int, int] = {}
    for idx, value in enumerate(flat):
        first_pos.setdefault(value, idx)
    return max(counts, key=lambda color: (counts[color], -first_pos[color]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each container's interior pixel count into an exterior bar.

    - Find every 1-pixel-thick rectangular frame.
    - Count non-background interior pixels.
    - Use the most common interior color for the bar.
    - Clear the interior, keep the frame, and draw the bar outside the frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _get_bg(g)

    frames = _find_frames(g, bg)
    assert frames, "expected at least one rectangular container"

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        interior_mask = interior != bg
        count = int(interior_mask.sum())
        assert count > 0, "each container must contain at least one non-background pixel"
        draw_color = _mode_color(interior[interior_mask])

        out[r0 + 1 : r1, c0 + 1 : c1] = bg
        if width >= height:
            start_r = r1 + 1
            assert start_r + count <= h, "vertical bar would exceed grid bounds"
            out[start_r : start_r + count, c0] = draw_color
        else:
            start_c = c1 + 1
            assert start_c + count <= w, "horizontal bar would exceed grid bounds"
            out[r0, start_c : start_c + count] = draw_color

    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (15, 27)
GRID_WIDTH_RANGE = (15, 27)
GEN_ATTEMPTS = 400

CONTAINER_COUNT_CHOICES = (1, 2, 3, 4)
MAX_BAR_LEN = 8
PALETTE = tuple(range(10))

DOWN_DIMS = [(h, w) for h in range(3, 8) for w in range(h, 9)]
RIGHT_DIMS = [(h, w) for h in range(4, 9) for w in range(3, h)]


def _bg_is_unique_mode(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for color, n in counts.items() if color != int(bg)), default=0)


def _sample_branch_sequence(count: int, rng: random.Random) -> list[str]:
    if count == 1:
        return [rng.choice(["down", "right"])]
    branches = ["down", "right"] * (count // 2)
    if count % 2:
        branches.append(rng.choice(["down", "right"]))
    rng.shuffle(branches)
    return branches


def _unique_mode_counts(total: int, parts: int, rng: random.Random) -> list[int]:
    if parts == 1:
        return [total]
    counts = [2] + [1] * (parts - 1)
    remaining = total - (parts + 1)
    assert remaining >= 0
    while remaining > 0:
        choices = [0]
        choices.extend(i for i in range(1, parts) if counts[i] + 1 < counts[0])
        idx = rng.choice(choices)
        counts[idx] += 1
        remaining -= 1
    return counts


def _sample_color_plan(total: int, bg: int, wall: int, rng: random.Random) -> tuple[list[int], list[int]]:
    pool = [color for color in PALETTE if color not in {int(bg), int(wall)}]
    if total == 1:
        color = rng.choice(pool)
        return [int(color)], [1]

    max_colors = min(3, len(pool), total - 1)
    num_colors = rng.choice([1] + list(range(2, max_colors + 1)))
    if num_colors == 1:
        color = rng.choice(pool)
        return [int(color)], [int(total)]

    mode_color = int(rng.choice(pool))
    others = rng.sample([color for color in pool if color != mode_color], num_colors - 1)
    colors = [mode_color] + [int(color) for color in others]
    counts = _unique_mode_counts(total, num_colors, rng)
    return colors, counts


def _reserve_bbox(r0: int, c0: int, branch: str, height: int, width: int, count: int) -> tuple[int, int, int, int]:
    if branch == "down":
        return r0, c0, r0 + height + count - 1, c0 + width - 1
    return r0, c0, r0 + height - 1, c0 + width + count - 1


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _sample_profile(branch: str, bg: int, wall: int, rng: random.Random) -> dict:
    dims = rng.choice(DOWN_DIMS if branch == "down" else RIGHT_DIMS)
    height, width = map(int, dims)
    interior_area = (height - 2) * (width - 2)
    count = rng.randint(1, min(MAX_BAR_LEN, interior_area))
    colors, counts = _sample_color_plan(count, bg, wall, rng)
    return {
        "branch": branch,
        "height": height,
        "width": width,
        "count": count,
        "colors": colors,
        "counts": counts,
        "mode_color": colors[0],
    }


def _placement_area(profile: dict) -> int:
    if profile["branch"] == "down":
        return (profile["height"] + profile["count"]) * profile["width"]
    return profile["height"] * (profile["width"] + profile["count"])


def _place_profiles(height: int, width: int, profiles: list[dict], rng: random.Random) -> list[dict] | None:
    order = sorted(range(len(profiles)), key=lambda i: _placement_area(profiles[i]), reverse=True)
    placed: list[dict | None] = [None] * len(profiles)
    reserved: list[tuple[int, int, int, int]] = []

    for idx in order:
        profile = profiles[idx]
        branch = profile["branch"]
        box_h = profile["height"] + (profile["count"] if branch == "down" else 0)
        box_w = profile["width"] + (profile["count"] if branch == "right" else 0)
        candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
        for r0 in range(1, height - box_h):
            for c0 in range(1, width - box_w):
                bbox = _reserve_bbox(r0, c0, branch, profile["height"], profile["width"], profile["count"])
                if all(_boxes_separated(bbox, other) for other in reserved):
                    candidates.append((r0, c0, bbox))
        if not candidates:
            return None
        r0, c0, bbox = rng.choice(candidates)
        placed[idx] = {**profile, "r0": r0, "c0": c0, "reserved_bbox": bbox}
        reserved.append(bbox)

    return [item for item in placed if item is not None]


def _draw_frame(grid: np.ndarray, r0: int, c0: int, height: int, width: int, color: int) -> None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _paint_interior(grid: np.ndarray, profile: dict, rng: random.Random) -> None:
    r0 = profile["r0"]
    c0 = profile["c0"]
    r1 = r0 + profile["height"] - 1
    c1 = c0 + profile["width"] - 1
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    chosen = rng.sample(cells, profile["count"])

    colors: list[int] = []
    for color, count in zip(profile["colors"], profile["counts"]):
        colors.extend([int(color)] * int(count))
    rng.shuffle(colors)

    for (r, c), color in zip(chosen, colors):
        grid[r, c] = color


def _build_expected(inp: np.ndarray, bg: int, profiles: list[dict]) -> np.ndarray:
    out = inp.copy()
    for profile in profiles:
        r0 = profile["r0"]
        c0 = profile["c0"]
        r1 = r0 + profile["height"] - 1
        c1 = c0 + profile["width"] - 1
        out[r0 + 1 : r1, c0 + 1 : c1] = bg
        if profile["branch"] == "down":
            out[r1 + 1 : r1 + 1 + profile["count"], c0] = profile["mode_color"]
        else:
            out[r0, c1 + 1 : c1 + 1 + profile["count"]] = profile["mode_color"]
    return out


def _frame_bboxes(frames: list[dict]) -> list[tuple[int, int, int, int]]:
    return sorted(frame["bbox"] for frame in frames)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_HEIGHT_RANGE)
    width = rng.randint(*GRID_WIDTH_RANGE)
    bg = int(rng.choice(PALETTE))
    wall = int(rng.choice([color for color in PALETTE if color != bg]))

    max_containers = 4
    if height * width < 260:
        max_containers = 3
    if height * width < 210:
        max_containers = 2
    count = int(rng.choice([n for n in CONTAINER_COUNT_CHOICES if n <= max_containers]))

    branches = _sample_branch_sequence(count, rng)
    profiles = [_sample_profile(branch, bg, wall, rng) for branch in branches]
    placed = _place_profiles(height, width, profiles, rng)
    if placed is None:
        return None

    inp = np.full((height, width), bg, dtype=int)
    for profile in placed:
        _draw_frame(inp, profile["r0"], profile["c0"], profile["height"], profile["width"], wall)
        _paint_interior(inp, profile, rng)

    if not _bg_is_unique_mode(inp, bg):
        return None

    detected = _find_frames(inp, bg)
    expected_bboxes = sorted(
        (
            profile["r0"],
            profile["c0"],
            profile["r0"] + profile["height"] - 1,
            profile["c0"] + profile["width"] - 1,
        )
        for profile in placed
    )
    if _frame_bboxes(detected) != expected_bboxes:
        return None

    out = _build_expected(inp, bg, placed)
    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
