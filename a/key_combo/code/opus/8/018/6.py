import random
from collections import Counter

import numpy as np


BACKGROUND = 0
FRAME_COLOR = 7
RANK_COLORS = (2, 1, 3, 4, 6)
INPUT_FILL_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)
GENERATION_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude=()) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    colors -= set(map(int, exclude))
    return sorted(colors)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_h(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    colors: list[int] | tuple[int, ...] | None = None,
    connectivity: int = 4,
) -> list[dict]:
    if bg is None:
        bg = get_bg(grid)
    if connectivity != 4:
        raise ValueError("this task only needs 4-connected components")

    if colors is None:
        color_list = colors_present(grid, exclude={bg})
    else:
        color_list = [int(color) for color in colors if int(color) != int(bg)]

    out: list[dict] = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for color in color_list:
        mask = grid == color
        visited = np.zeros_like(mask, dtype=bool)
        coords_all = np.argwhere(mask)
        for sr, sc in coords_all:
            sr = int(sr)
            sc = int(sc)
            if visited[sr, sc]:
                continue
            stack = [(sr, sc)]
            visited[sr, sc] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in neighbors:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                        if mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _frame_records(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frames = find_components(g, bg=bg, colors=[FRAME_COLOR], connectivity=4)
    assert 3 <= len(frames) <= 5, "expected 3 to 5 orange frames"

    records = []
    outer_sizes = set()
    fill_colors = set()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        h = bbox_h(frame["bbox"])
        w = bbox_w(frame["bbox"])
        assert h >= 4 and w >= 3, "frame too small"
        outer_sizes.add((h, w))

        rel = np.zeros((h, w), dtype=bool)
        coords = frame["coords"]
        rel[coords[:, 0] - r0, coords[:, 1] - c0] = True
        expected = np.zeros((h, w), dtype=bool)
        expected[:, 0] = True
        expected[:, -1] = True
        expected[-1, :] = True
        assert np.array_equal(rel, expected), "orange component must be a U frame"

        interior = g[r0:r1, c0 + 1 : c1]
        interior_h, interior_w = interior.shape
        assert interior_h >= 3 and interior_w >= 1

        interior_colors = colors_present(interior, exclude={bg})
        assert FRAME_COLOR not in interior_colors, "frame color cannot appear inside the basin"
        assert len(interior_colors) == 1, "each basin must contain exactly one fill color"
        fill_color = int(interior_colors[0])
        fill_colors.add(fill_color)

        fill_mask = interior == fill_color
        row_filled = np.all(fill_mask, axis=1)
        expected_fill = np.repeat(row_filled[:, None], interior_w, axis=1)
        assert np.array_equal(fill_mask, expected_fill), "fill must be a solid rectangle"
        fill_height = int(row_filled.sum())
        assert 1 <= fill_height <= interior_h
        assert np.all(row_filled[-fill_height:]), "fill must touch the bottom"
        assert not np.any(row_filled[:-fill_height]), "empty rows must stay above the fill"

        empty_rows = int(interior_h - fill_height)
        fill_coords = np.argwhere(g == fill_color)
        fill_coords = fill_coords[
            (fill_coords[:, 0] >= r0)
            & (fill_coords[:, 0] < r1)
            & (fill_coords[:, 1] > c0)
            & (fill_coords[:, 1] < c1)
        ]
        assert fill_coords.shape[0] == fill_height * interior_w

        records.append(
            {
                "bbox": frame["bbox"],
                "top": int(r0),
                "left": int(c0),
                "empty_rows": empty_rows,
                "fill_color": fill_color,
                "fill_coords": fill_coords,
            }
        )

    assert len(outer_sizes) == 1, "all frames must have the same outer size"
    assert len(fill_colors) == len(records), "input fill colors must be distinct"
    empty_rows = [record["empty_rows"] for record in records]
    assert len(set(empty_rows)) == len(empty_rows), "empty-row counts must be distinct"
    return sorted(records, key=lambda record: record["left"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each U-frame's liquid by its global liquid-level rank.

    Input structure:
    - The grid has a black background and 3 to 5 separated orange U-shaped frames.
    - Every frame has the same outer size, with a bottom-aligned solid liquid rectangle inside it.
    - Each liquid uses one non-orange color, and the empty space above it stays black.

    Rule:
    - For each frame, count the number of empty interior rows above the liquid.
    - Sort the frames by that count in ascending order, so fewer empty rows means a higher liquid level.
    - Recolor only the liquid cells by rank: rank 0 -> red, rank 1 -> blue, rank 2 -> green, rank 3 -> yellow, rank 4 -> pink.

    Key insight:
    - A frame's output color depends on its liquid level relative to the other frames, not on its original color or absolute screen height.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    records = _frame_records(g)
    out = g.copy()
    for rank, record in enumerate(sorted(records, key=lambda record: record["empty_rows"])):
        target_color = int(RANK_COLORS[rank])
        coords = record["fill_coords"]
        out[coords[:, 0], coords[:, 1]] = target_color
    return out


def _weighted_frame_count(rng: random.Random) -> int:
    return rng.choices([3, 4, 5], weights=[2, 3, 5], k=1)[0]


def _sample_layout(rng: random.Random):
    frame_count = _weighted_frame_count(rng)
    outer_h = rng.randint(8, 10)
    outer_w = rng.randint(5, 7)
    interior_h = outer_h - 1
    if interior_h < frame_count:
        return None

    gaps = [rng.randint(2, 4) for _ in range(frame_count - 1)]
    left_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)
    width = left_margin + right_margin + frame_count * outer_w + sum(gaps)
    if width > 30:
        return None

    min_height = max(12, outer_h + 2)
    max_height = min(18, outer_h + 8)
    if min_height > max_height:
        return None
    height = rng.randint(min_height, max_height)

    max_top = height - outer_h - 1
    if max_top < 1:
        return None

    empty_rows = rng.sample(list(range(interior_h)), frame_count)

    lefts = []
    cursor = left_margin
    for idx in range(frame_count):
        lefts.append(cursor)
        cursor += outer_w
        if idx < frame_count - 1:
            cursor += gaps[idx]

    for _ in range(120):
        tops = [rng.randint(1, max_top) for _ in range(frame_count)]
        if len(set(tops)) < 2:
            continue
        rank_order = sorted(range(frame_count), key=lambda idx: empty_rows[idx])
        abs_fill_tops = [tops[idx] + empty_rows[idx] for idx in range(frame_count)]
        if len(set(abs_fill_tops)) < frame_count:
            continue
        if sorted(range(frame_count), key=lambda idx: abs_fill_tops[idx]) == rank_order:
            continue
        if rank_order == list(range(frame_count)):
            continue
        fill_colors = rng.sample(list(INPUT_FILL_COLORS), frame_count)
        if all(fill_colors[idx] == RANK_COLORS[rank] for rank, idx in enumerate(rank_order)):
            continue
        return {
            "frame_count": frame_count,
            "height": height,
            "width": width,
            "outer_h": outer_h,
            "outer_w": outer_w,
            "interior_h": interior_h,
            "tops": tops,
            "lefts": lefts,
            "empty_rows": empty_rows,
            "fill_colors": fill_colors,
        }
    return None


def _render_layout(meta: dict) -> np.ndarray:
    grid = np.full((meta["height"], meta["width"]), BACKGROUND, dtype=int)
    outer_h = int(meta["outer_h"])
    outer_w = int(meta["outer_w"])
    interior_h = int(meta["interior_h"])

    for idx in range(meta["frame_count"]):
        top = int(meta["tops"][idx])
        left = int(meta["lefts"][idx])
        fill_color = int(meta["fill_colors"][idx])
        empty_rows = int(meta["empty_rows"][idx])

        grid[top : top + outer_h, left] = FRAME_COLOR
        grid[top : top + outer_h, left + outer_w - 1] = FRAME_COLOR
        grid[top + outer_h - 1, left : left + outer_w] = FRAME_COLOR

        fill_start = top + empty_rows
        fill_end = top + interior_h
        grid[fill_start:fill_end, left + 1 : left + outer_w - 1] = fill_color

    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    records = _frame_records(grid)
    if len(records) < 3:
        return False

    rank_order = sorted(range(len(records)), key=lambda idx: records[idx]["empty_rows"])
    if rank_order == list(range(len(records))):
        return False

    abs_fill_tops = [records[idx]["top"] + records[idx]["empty_rows"] for idx in range(len(records))]
    if len(set(abs_fill_tops)) != len(records):
        return False
    if sorted(range(len(records)), key=lambda idx: abs_fill_tops[idx]) == rank_order:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    changed_frames = 0
    for rank, idx in enumerate(rank_order):
        coords = records[idx]["fill_coords"]
        new_colors = set(map(int, out[coords[:, 0], coords[:, 1]]))
        if new_colors != {int(RANK_COLORS[rank])}:
            return False
        old_colors = set(map(int, grid[coords[:, 0], coords[:, 1]]))
        if old_colors != new_colors:
            changed_frames += 1
    if changed_frames < 2:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        meta = _sample_layout(rng)
        if meta is None:
            continue
        grid = _render_layout(meta)
        if _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid ranked U-frame liquid puzzle")
