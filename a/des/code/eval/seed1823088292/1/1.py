from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400

SHAPES = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
)


@dataclass
class Frame:
    r0: int
    c0: int
    r1: int
    c1: int
    cells: list[tuple[int, int]]
    parent: int | None = None
    depth: int = 0

    @property
    def h(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def w(self) -> int:
        return self.c1 - self.c0 + 1

    @property
    def area(self) -> int:
        return self.h * self.w


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_legend(grid: np.ndarray) -> tuple[int, list[int]]:
    colored = (grid != BLACK) & (grid != GREY)
    coords = np.argwhere(colored)
    assert coords.size > 0
    legend_col = int(coords[:, 1].min())
    rows = np.where(colored[:, legend_col])[0]
    assert rows.size > 0

    colors: list[int] = []
    r = int(rows.min())
    r_end = int(rows.max())
    while r <= r_end:
        if not colored[r, legend_col]:
            r += 1
            continue
        color = int(grid[r, legend_col])
        colors.append(color)
        while r <= r_end and colored[r, legend_col] and int(grid[r, legend_col]) == color:
            r += 1
    assert colors
    return legend_col, colors


def _find_frames(grid: np.ndarray) -> list[Frame]:
    comps = _components(grid == GREY)
    assert comps
    frames: list[Frame] = []
    for cells in comps:
        r0, c0, r1, c1 = _bbox(cells)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 3 and w >= 3
        assert len(cells) == 2 * (h + w) - 4
        cell_set = set(cells)
        for c in range(c0, c1 + 1):
            assert (r0, c) in cell_set
            assert (r1, c) in cell_set
        for r in range(r0, r1 + 1):
            assert (r, c0) in cell_set
            assert (r, c1) in cell_set
        frames.append(Frame(r0, c0, r1, c1, cells))

    for i, frame in enumerate(frames):
        containers = [
            j
            for j, outer in enumerate(frames)
            if outer.r0 < frame.r0 and outer.c0 < frame.c0 and outer.r1 > frame.r1 and outer.c1 > frame.c1
        ]
        frame.depth = len(containers) + 1
        if containers:
            frame.parent = min(containers, key=lambda j: frames[j].area)
    return frames


def _leaf_fill(grid: np.ndarray, frame: Frame) -> tuple[int, list[tuple[int, int]]]:
    sub = grid[frame.r0 + 1 : frame.r1, frame.c0 + 1 : frame.c1]
    mask = (sub != BLACK) & (sub != GREY)
    coords = np.argwhere(mask)
    assert coords.size > 0
    colors = np.unique(sub[mask])
    assert colors.size == 1
    offsets = [(int(r), int(c)) for r, c in coords]
    return int(colors[0]), offsets


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _legend_col, legend_colors = _extract_legend(g)
    frames = _find_frames(g)

    children = [0] * len(frames)
    for frame in frames:
        if frame.parent is not None:
            children[frame.parent] += 1

    out = g.copy()
    for frame in frames:
        assert 1 <= frame.depth <= len(legend_colors)
        color = int(legend_colors[frame.depth - 1])
        for r, c in frame.cells:
            out[r, c] = color

    for i, frame in enumerate(frames):
        if children[i] != 0:
            continue
        fill_color, offsets = _leaf_fill(g, frame)
        cur = i
        while cur is not None:
            target = frames[cur]
            for dr, dc in offsets:
                rr = target.r0 + 1 + dr
                cc = target.c0 + 1 + dc
                if out[rr, cc] == BLACK:
                    out[rr, cc] = fill_color
            cur = target.parent

    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _pack_groups(groups: list[dict], rng: random.Random) -> tuple[dict[int, tuple[int, int]], int, int]:
    mode = rng.choice(("columns", "rows"))
    order = list(range(len(groups)))
    rng.shuffle(order)
    placements: dict[int, tuple[int, int]] = {}

    if mode == "columns":
        bucket_count = 1 if len(groups) == 1 else rng.randint(1, min(2, len(groups)))
        buckets: list[list[int]] = [[] for _ in range(bucket_count)]
        heights = [0] * bucket_count
        for idx in order:
            bucket = min(range(bucket_count), key=lambda b: heights[b])
            buckets[bucket].append(idx)
            heights[bucket] += groups[idx]["top_h"] + 3

        x = 0
        content_h = 0
        content_w = 0
        for bucket in buckets:
            col_w = max(groups[idx]["top_w"] for idx in bucket)
            y = rng.randint(0, 2)
            for idx in bucket:
                dx = rng.randint(0, col_w - groups[idx]["top_w"])
                placements[idx] = (y, x + dx)
                y += groups[idx]["top_h"] + rng.randint(2, 4)
                content_w = max(content_w, x + dx + groups[idx]["top_w"])
            content_h = max(content_h, y)
            x += col_w + rng.randint(3, 6)
        return placements, content_h, content_w

    bucket_count = 1 if len(groups) == 1 else rng.randint(1, min(2, len(groups)))
    buckets = [[] for _ in range(bucket_count)]
    widths = [0] * bucket_count
    for idx in order:
        bucket = min(range(bucket_count), key=lambda b: widths[b])
        buckets[bucket].append(idx)
        widths[bucket] += groups[idx]["top_w"] + 3

    y = 0
    content_h = 0
    content_w = 0
    for bucket in buckets:
        row_h = max(groups[idx]["top_h"] for idx in bucket)
        x = rng.randint(0, 2)
        for idx in bucket:
            dy = rng.randint(0, row_h - groups[idx]["top_h"])
            placements[idx] = (y + dy, x)
            x += groups[idx]["top_w"] + rng.randint(3, 6)
            content_h = max(content_h, y + dy + groups[idx]["top_h"])
        content_w = max(content_w, x)
        y += row_h + rng.randint(3, 6)
    return placements, content_h, content_w


def _build_group(depth: int, fill_color: int, rng: random.Random) -> dict:
    shape = rng.choice(SHAPES)
    block_h, block_w = shape.shape

    overlap_mode = depth > 1 and rng.random() < 0.5
    if overlap_mode:
        inner_h = block_h + rng.randint(1, 2)
        inner_w = block_w + rng.randint(1, 2)
        dr = rng.randint(1, inner_h - block_h)
        dc = rng.randint(1, inner_w - block_w)
    else:
        inner_h = block_h + rng.randint(0, 2)
        inner_w = block_w + rng.randint(0, 2)
        dr = rng.randint(0, inner_h - block_h)
        dc = rng.randint(0, inner_w - block_w)

    sizes = [(inner_h + 2, inner_w + 2)]
    child_offsets: list[tuple[int, int]] = []

    for level in range(1, depth):
        child_h, child_w = sizes[-1]
        if level == 1 and overlap_mode:
            top_gap = dr
            left_gap = dc
            bottom_gap = rng.randint(1, 2)
            right_gap = rng.randint(1, 2)
        elif level == 1 and rng.random() < 0.5:
            top_gap = dr + block_h
            left_gap = dc + block_w
            bottom_gap = rng.randint(1, 2)
            right_gap = rng.randint(1, 2)
        else:
            top_gap = rng.randint(1, 2)
            left_gap = rng.randint(1, 2)
            bottom_gap = rng.randint(1, 2)
            right_gap = rng.randint(1, 2)

        parent_h = child_h + top_gap + bottom_gap + 2
        parent_w = child_w + left_gap + right_gap + 2
        child_offsets.append((top_gap + 1, left_gap + 1))
        sizes.append((parent_h, parent_w))

    return {
        "depth": depth,
        "shape": shape,
        "fill_color": fill_color,
        "fill_offset": (dr, dc),
        "sizes": sizes,
        "child_offsets": child_offsets,
        "top_h": sizes[-1][0],
        "top_w": sizes[-1][1],
    }


def _instantiate_group(group: dict, top_r: int, top_c: int) -> dict:
    depth = int(group["depth"])
    sizes: list[tuple[int, int]] = group["sizes"]
    child_offsets: list[tuple[int, int]] = group["child_offsets"]

    positions: list[tuple[int, int] | None] = [None] * depth
    positions[-1] = (top_r, top_c)
    for idx in range(depth - 2, -1, -1):
        pr, pc = positions[idx + 1]
        off_r, off_c = child_offsets[idx]
        positions[idx] = (pr + off_r, pc + off_c)

    frames = []
    for idx in range(depth - 1, -1, -1):
        r0, c0 = positions[idx]
        h, w = sizes[idx]
        frames.append(
            {
                "r0": r0,
                "c0": c0,
                "h": h,
                "w": w,
                "depth": depth - idx,
            }
        )

    shape = group["shape"]
    fill_offsets = [(int(r), int(c)) for r, c in np.argwhere(shape)]
    leaf_r, leaf_c = positions[0]
    dr, dc = group["fill_offset"]
    return {
        "frames": frames,
        "fill_color": int(group["fill_color"]),
        "fill_offsets": fill_offsets,
        "fill_anchor": (leaf_r + 1 + dr, leaf_c + 1 + dc),
        "shape": shape,
    }


def _expected_from_spec(grid: np.ndarray, legend_colors: list[int], groups: list[dict]) -> np.ndarray:
    out = grid.copy()
    for group in groups:
        for frame in group["frames"]:
            _draw_frame(out, frame["r0"], frame["c0"], frame["h"], frame["w"], legend_colors[frame["depth"] - 1])

    for group in groups:
        fill_color = int(group["fill_color"])
        for frame in group["frames"]:
            for dr, dc in group["fill_offsets"]:
                rr = frame["r0"] + 1 + dr
                cc = frame["c0"] + 1 + dc
                if out[rr, cc] == BLACK:
                    out[rr, cc] = fill_color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    max_depth = rng.choices([1, 2, 3, 4], weights=[1, 3, 3, 2])[0]
    max_groups = 3 if max_depth == 4 else 4
    group_count = rng.randint(1, max_groups)

    depths = [rng.randint(1, max_depth) for _ in range(group_count)]
    depths[rng.randrange(group_count)] = max_depth

    palette = rng.sample(COLOR_POOL, max_depth + group_count)
    legend_colors = palette[:max_depth]
    fill_colors = palette[max_depth:]
    groups = [_build_group(depths[i], fill_colors[i], rng) for i in range(group_count)]

    placements, content_h, content_w = _pack_groups(groups, rng)
    legend_heights = [rng.randint(1, 3) for _ in range(max_depth)]
    legend_total_h = sum(legend_heights)

    legend_col = rng.choice((0, 1))
    main_left = legend_col + 1 + rng.randint(2, 4)
    main_top = rng.randint(0, 3)
    height = max(legend_total_h, main_top + content_h) + rng.randint(1, 4)
    width = main_left + content_w + rng.randint(1, 4)
    if height > 36 or width > 36:
        return None

    legend_top = rng.randint(0, height - legend_total_h)
    grid = np.zeros((height, width), dtype=int)

    rr = legend_top
    for seg_h, color in zip(legend_heights, legend_colors):
        grid[rr : rr + seg_h, legend_col] = int(color)
        rr += seg_h

    placed_groups = []
    for idx, group in enumerate(groups):
        rel_r, rel_c = placements[idx]
        placed = _instantiate_group(group, main_top + rel_r, main_left + rel_c)
        placed_groups.append(placed)
        for frame in placed["frames"]:
            _draw_frame(grid, frame["r0"], frame["c0"], frame["h"], frame["w"], GREY)
        fr, fc = placed["fill_anchor"]
        for dr, dc in placed["fill_offsets"]:
            grid[fr + dr, fc + dc] = placed["fill_color"]

    expected = _expected_from_spec(grid, legend_colors, placed_groups)
    actual = main(grid)
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
