from __future__ import annotations

import random

import numpy as np


BG = 0
STAGING_SIZE = 30
GEN_RETRIES = 400
PLACEMENT_RETRIES = 300
ALL_COLORS = tuple(range(10))


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(cnts.argmax())])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rectangular_ring(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False
    expected = 2 * (height + width) - 4
    if len(points) != expected:
        return False
    point_set = set(points)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in point_set):
                return False
    return True


def _find_frames(grid: np.ndarray) -> tuple[int, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frames: list[dict[str, int]] = []

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for comp in _components4(g == color):
            if not _is_rectangular_ring(comp):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            seed_coords = np.argwhere(inner != bg)
            if seed_coords.shape != (1, 2):
                continue
            sr, sc = map(int, seed_coords[0])
            seed_color = int(inner[sr, sc])
            if seed_color == color:
                continue
            frames.append(
                {
                    "frame_color": int(color),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "seed_r": int(r0 + 1 + sr),
                    "seed_c": int(c0 + 1 + sc),
                    "seed_color": int(seed_color),
                }
            )

    frames.sort(key=lambda item: (item["r0"], item["c0"], item["r1"], item["c1"]))
    return int(bg), frames


def _paint_inner_ring(
    out: np.ndarray,
    *,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    fill_color: int,
    bg: int,
) -> None:
    top = r0 + 1
    bottom = r1 - 1
    left = c0 + 1
    right = c1 - 1
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            if r in (top, bottom) or c in (left, right):
                if int(out[r, c]) == bg:
                    out[r, c] = fill_color


def _paint_small_interior(
    out: np.ndarray,
    *,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    seed_r: int,
    seed_c: int,
    fill_color: int,
    bg: int,
) -> None:
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if (r, c) != (seed_r, seed_c) and int(out[r, c]) == bg:
                out[r, c] = fill_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint each rectangular frame's inner wall with its seed color, then swap the seed into the frame color.

    Input structure:
    - The background is black.
    - One or more non-touching one-pixel-thick rectangular frame rings appear on the grid.
    - Each frame encloses exactly one single-pixel seed of a distinct color, and the rest of its interior is black.

    Rule:
    - Keep every frame pixel unchanged.
    - For each frame, read the enclosed seed color C.
    - If the frame interior is at least 3x3, recolor the black pixels on the inner border of that interior with C.
    - Otherwise, recolor every non-seed black pixel inside the frame with C.
    - In every case, replace the seed pixel itself with the frame color.

    Key insight:
    - The seed color does not fill the whole cavity; it traces only the frame's inner wall, while the seed pixel is converted back into the enclosing frame color.
    """
    g = np.asarray(grid, dtype=int)
    bg, frames = _find_frames(g)
    out = g.copy()

    for frame in frames:
        r0 = frame["r0"]
        c0 = frame["c0"]
        r1 = frame["r1"]
        c1 = frame["c1"]
        seed_r = frame["seed_r"]
        seed_c = frame["seed_c"]
        frame_color = frame["frame_color"]
        seed_color = frame["seed_color"]

        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        if inner_h > 2 and inner_w > 2:
            _paint_inner_ring(out, r0=r0, c0=c0, r1=r1, c1=c1, fill_color=seed_color, bg=bg)
        else:
            _paint_small_interior(
                out,
                r0=r0,
                c0=c0,
                r1=r1,
                c1=c1,
                seed_r=seed_r,
                seed_c=seed_c,
                fill_color=seed_color,
                bg=bg,
            )
        out[seed_r, seed_c] = frame_color

    return out


def _sample_branch_types(frame_count: int) -> list[str]:
    if frame_count == 1:
        return [random.choice(["spacious", "narrow"])]
    narrow_count = frame_count // 2
    if frame_count % 2 == 1 and random.random() < 0.5:
        narrow_count += 1
    branch_types = ["narrow"] * narrow_count + ["spacious"] * (frame_count - narrow_count)
    random.shuffle(branch_types)
    return branch_types


def _sample_inner_dims(branch_type: str) -> tuple[int, int]:
    if branch_type == "spacious":
        return random.randint(3, 8), random.randint(3, 8)

    mode = random.choice(["short", "thin", "tiny"])
    if mode == "short":
        return random.randint(1, 2), random.randint(1, 8)
    if mode == "thin":
        return random.randint(1, 8), random.randint(1, 2)
    return random.randint(1, 2), random.randint(1, 2)


def _choose_seed(inner_h: int, inner_w: int, branch_type: str) -> tuple[int, int]:
    ring_cells = [(r, c) for r in range(inner_h) for c in range(inner_w) if r in (0, inner_h - 1) or c in (0, inner_w - 1)]
    ring_set = set(ring_cells)
    core_cells = [(r, c) for r in range(inner_h) for c in range(inner_w) if (r, c) not in ring_set]

    if branch_type == "spacious" and ring_cells and core_cells:
        pool = ring_cells if random.random() < 0.5 else core_cells
        return random.choice(pool)
    return random.choice(ring_cells if ring_cells else core_cells)


def _choose_crop_span(min_idx: int, max_idx: int) -> tuple[int, int]:
    base = max_idx - min_idx + 1
    left_cap = min_idx
    right_cap = STAGING_SIZE - 1 - max_idx

    if left_cap >= 1 and right_cap >= 1 and base <= STAGING_SIZE - 2:
        min_target = max(8, base + 2)
        max_target = min(STAGING_SIZE, base + 10)
        target = random.randint(min_target, max_target)
        extra = target - base
        low = max(1, extra - right_cap)
        high = min(left_cap, extra - 1)
        if low <= high:
            left_extra = random.randint(low, high)
            right_extra = extra - left_extra
            return min_idx - left_extra, max_idx + right_extra

    extra = min(STAGING_SIZE - base, random.randint(0, min(8, STAGING_SIZE - base)))
    low = max(0, extra - right_cap)
    high = min(left_cap, extra)
    left_extra = random.randint(low, high)
    right_extra = extra - left_extra
    return min_idx - left_extra, max_idx + right_extra


def _render_scene(placements: list[dict[str, int]], frame_color: int) -> np.ndarray:
    min_r = min(item["r0"] for item in placements)
    min_c = min(item["c0"] for item in placements)
    max_r = max(item["r0"] + item["outer_h"] - 1 for item in placements)
    max_c = max(item["c0"] + item["outer_w"] - 1 for item in placements)
    crop_r0, crop_r1 = _choose_crop_span(min_r, max_r)
    crop_c0, crop_c1 = _choose_crop_span(min_c, max_c)

    grid = np.full((crop_r1 - crop_r0 + 1, crop_c1 - crop_c0 + 1), BG, dtype=int)
    for item in placements:
        r0 = item["r0"] - crop_r0
        c0 = item["c0"] - crop_c0
        outer_h = item["outer_h"]
        outer_w = item["outer_w"]
        seed_r = r0 + 1 + item["seed_dr"]
        seed_c = c0 + 1 + item["seed_dc"]

        grid[r0, c0 : c0 + outer_w] = frame_color
        grid[r0 + outer_h - 1, c0 : c0 + outer_w] = frame_color
        grid[r0 : r0 + outer_h, c0] = frame_color
        grid[r0 : r0 + outer_h, c0 + outer_w - 1] = frame_color
        grid[seed_r, seed_c] = item["seed_color"]

    return grid


def _validate_scene(grid: np.ndarray, expected_count: int) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30

    bg, frames = _find_frames(grid)
    assert bg == BG
    assert len(frames) == expected_count

    used = np.zeros_like(grid, dtype=bool)
    seed_colors: set[int] = set()
    branch_hits = {"spacious": 0, "narrow": 0}

    for frame in frames:
        r0 = frame["r0"]
        c0 = frame["c0"]
        r1 = frame["r1"]
        c1 = frame["c1"]
        used[r0 : r1 + 1, c0 : c1 + 1] = True

        seed_color = frame["seed_color"]
        assert seed_color not in seed_colors
        seed_colors.add(seed_color)

        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        if inner_h > 2 and inner_w > 2:
            branch_hits["spacious"] += 1
        else:
            branch_hits["narrow"] += 1

        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        assert int(np.count_nonzero(inner != bg)) == 1

    for r, c in np.argwhere(grid != bg):
        assert used[int(r), int(c)]

    if expected_count > 1:
        assert branch_hits["spacious"] >= 1
        assert branch_hits["narrow"] >= 1


def _generate_attempt() -> np.ndarray | None:
    frame_count = random.choice([1, 2, 2, 3, 3, 4, 4, 5])
    branch_types = _sample_branch_types(frame_count)
    frame_color = random.choice([c for c in ALL_COLORS if c != BG])
    seed_palette = [c for c in ALL_COLORS if c not in (BG, frame_color)]
    seed_colors = random.sample(seed_palette, frame_count)

    specs: list[dict[str, int | str]] = []
    for branch_type, seed_color in zip(branch_types, seed_colors, strict=False):
        inner_h, inner_w = _sample_inner_dims(branch_type)
        seed_dr, seed_dc = _choose_seed(inner_h, inner_w, branch_type)
        specs.append(
            {
                "branch_type": branch_type,
                "seed_color": int(seed_color),
                "inner_h": int(inner_h),
                "inner_w": int(inner_w),
                "outer_h": int(inner_h + 2),
                "outer_w": int(inner_w + 2),
                "seed_dr": int(seed_dr),
                "seed_dc": int(seed_dc),
            }
        )

    specs.sort(key=lambda item: (-(int(item["outer_h"]) * int(item["outer_w"])), random.random()))
    occupancy = np.zeros((STAGING_SIZE, STAGING_SIZE), dtype=bool)
    placements: list[dict[str, int]] = []

    for spec in specs:
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(1, STAGING_SIZE - outer_h - 1)
            c0 = random.randint(1, STAGING_SIZE - outer_w - 1)
            rr0 = r0 - 1
            rr1 = r0 + outer_h + 1
            cc0 = c0 - 1
            cc1 = c0 + outer_w + 1
            if occupancy[rr0:rr1, cc0:cc1].any():
                continue
            occupancy[rr0:rr1, cc0:cc1] = True
            placements.append(
                {
                    "r0": int(r0),
                    "c0": int(c0),
                    "outer_h": outer_h,
                    "outer_w": outer_w,
                    "seed_dr": int(spec["seed_dr"]),
                    "seed_dc": int(spec["seed_dc"]),
                    "seed_color": int(spec["seed_color"]),
                }
            )
            placed = True
            break
        if not placed:
            return None

    grid = _render_scene(placements, frame_color)
    _validate_scene(grid, frame_count)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
