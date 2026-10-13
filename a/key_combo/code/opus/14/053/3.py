from __future__ import annotations
from collections import deque

import numpy as np


BG = 0
FRAME_PALETTE = (1, 2, 3, 4, 5, 6, 7, 8)
MAX_ATTEMPTS = 500


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _classify_rect_component(grid: np.ndarray, color: int, comp: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox_from_coords(comp)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    area = len(comp)
    if h < 3 or w < 3:
        return None

    border = _border_mask((h, w))
    if np.all(patch == color):
        return {"kind": "solid", "color": int(color), "bbox": (r0, c0, r1, c1)}

    if area != int(border.sum()):
        return None
    if not np.all(patch[border] == int(color)):
        return None
    interior = patch[1:-1, 1:-1]
    if interior.size == 0 or np.any(interior == int(color)) or not np.any(interior == BG):
        return None

    return {
        "kind": "frame",
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "interior": interior.copy(),
        "count": int(np.count_nonzero(interior != BG)),
    }


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        for comp in _components_of_color(grid, color):
            info = _classify_rect_component(grid, color, comp)
            if info is not None and info["kind"] == "frame":
                frames.append(info)
    return frames


def _stack_output(frames: list[dict]) -> np.ndarray:
    ordered = sorted(frames, key=lambda item: (-item["count"], item["bbox"][0], item["bbox"][1]))
    pieces = []
    for frame in ordered:
        piece = frame["interior"].copy()
        piece[piece == BG] = int(frame["color"])
        pieces.append(piece)
    return np.vstack(pieces)


def main(grid: np.ndarray) -> np.ndarray:
    """Stack each hollow frame's interior by descending content count, recoloring its black background to the frame color.

    Input structure:
    - The grid has black background plus several 1-pixel-thick monochrome hollow rectangles, each with scattered non-black pixels inside.
    - It also contains similarly sized solid monochrome rectangles that act as distractors and must be ignored.

    Rule:
    - Detect every hollow rectangular frame whose strict interior still contains at least one black cell.
    - For each such frame, count the non-black cells in its strict interior, sort the frames by that count from large to small, crop each strict interior, and stack those crops vertically.
    - In each stacked crop, recolor black cells to that source frame's border color while keeping the non-black interior pixels unchanged.

    Key insight:
    - The ordering signal is hidden inside the frames: only by looking into each frame and counting its interior foreground cells can you determine the stack order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _extract_frames(g)
    assert frames, "expected at least one hollow frame"
    counts = [frame["count"] for frame in frames]
    widths = {frame["interior"].shape[1] for frame in frames}
    assert len(widths) == 1, "all frame interiors must have the same width"
    assert len(set(counts)) == len(counts), "content counts must be unique"

    return _stack_output(frames)


def _empty_grid(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=int)


def _fits(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, gap: int = 1) -> bool:
    hh, ww = occupied.shape
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if r0 < 0 or c0 < 0 or r1 >= hh or c1 >= ww:
        return False
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(hh - 1, r1 + gap)
    cc1 = min(ww - 1, c1 + gap)
    return not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _mark(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _choose_positions(rng: np.random.Generator, shape: tuple[int, int], sizes: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    occupied = np.zeros(shape, dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)
    for idx in order:
        h, w = sizes[idx]
        candidates = [(r, c) for r in range(shape[0] - h + 1) for c in range(shape[1] - w + 1) if _fits(occupied, r, c, h, w)]
        if not candidates:
            return None
        r0, c0 = candidates[int(rng.integers(len(candidates)))]
        positions[idx] = (r0, c0)
        _mark(occupied, r0, c0, h, w)
    return [pos for pos in positions if pos is not None]


def _make_frame_patch(
    rng: np.random.Generator,
    border_color: int,
    inner_h: int,
    inner_w: int,
    count: int,
) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), border_color, dtype=int)
    interior = np.zeros((inner_h, inner_w), dtype=int)
    cells = rng.choice(inner_h * inner_w, size=count, replace=False)
    palette = [color for color in FRAME_PALETTE if color != border_color]
    colors = rng.choice(np.array(palette, dtype=int), size=count, replace=True)
    if count >= 2 and len(set(int(v) for v in colors)) == 1:
        colors[0] = palette[0]
        colors[-1] = palette[-1]
    for flat_idx, color in zip(cells, colors):
        r, c = divmod(int(flat_idx), inner_w)
        interior[r, c] = int(color)
    patch[1:-1, 1:-1] = interior
    return patch


def _make_solid_patch(color: int, h: int, w: int) -> np.ndarray:
    return np.full((h, w), int(color), dtype=int)


def _expected_from_meta(frames: list[dict]) -> np.ndarray:
    ordered = sorted(frames, key=lambda item: (-item["count"], item["row"], item["col"]))
    pieces = []
    for frame in ordered:
        piece = frame["interior"].copy()
        piece[piece == BG] = int(frame["color"])
        pieces.append(piece)
    return np.vstack(pieces)


def _rank(values: list[tuple]) -> list[int]:
    return [idx for idx, _ in sorted(enumerate(values), key=lambda item: item[1])]


def _anti_shortcut_ok(frames: list[dict]) -> bool:
    count_rank = _rank([(-frame["count"], frame["row"], frame["col"]) for frame in frames])
    row_rank = _rank([(frame["row"], frame["col"]) for frame in frames])
    col_rank = _rank([(frame["col"], frame["row"]) for frame in frames])
    area_rank = _rank([(-frame["interior"].size, frame["row"], frame["col"]) for frame in frames])
    color_up = _rank([(frame["color"],) for frame in frames])
    color_down = _rank([(-frame["color"],) for frame in frames])
    return (
        count_rank != row_rank
        and count_rank != col_rank
        and count_rank != area_rank
        and count_rank != color_up
        and count_rank != color_down
    )


def _matches_metadata(grid: np.ndarray, frames: list[dict], expected: np.ndarray) -> bool:
    try:
        detected = _extract_frames(grid)
    except Exception:
        return False
    if len(detected) != len(frames):
        return False

    want = {
        (frame["color"], frame["row"], frame["col"], frame["bbox"][2], frame["bbox"][3]): frame
        for frame in frames
    }
    for det in detected:
        key = (det["color"], det["bbox"][0], det["bbox"][1], det["bbox"][2], det["bbox"][3])
        if key not in want:
            return False
        src = want[key]
        if det["count"] != src["count"]:
            return False
        if det["interior"].shape != src["interior"].shape:
            return False
        if not np.array_equal(det["interior"], src["interior"]):
            return False

    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if not _anti_shortcut_ok(frames):
        return False
    return True


def _sample_counts(rng: np.random.Generator, heights: list[int], inner_w: int) -> list[int] | None:
    areas = [h * inner_w for h in heights]
    max_choices = sum(max(area - 1, 0) for area in areas)
    if min(areas) < 2 or max_choices < len(areas):
        return None

    for _ in range(120):
        counts = [int(rng.integers(1, area)) for area in areas]
        if len(set(counts)) != len(counts):
            continue
        area_rank = _rank([(-area, idx) for idx, area in enumerate(areas)])
        count_rank = _rank([(-count, idx) for idx, count in enumerate(counts)])
        if count_rank == area_rank:
            continue
        return counts
    return None


def _attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_frames = int(rng.integers(3, 6))
    n_solids = int(rng.integers(1, 3))
    inner_w = int(rng.integers(3, 7))
    heights = [int(rng.integers(3, 7)) for _ in range(n_frames)]
    counts = _sample_counts(rng, heights, inner_w)
    if counts is None:
        return None

    frame_colors = list(map(int, rng.choice(np.array(FRAME_PALETTE, dtype=int), size=n_frames, replace=False)))
    solid_colors_pool = [color for color in FRAME_PALETTE if color not in frame_colors]
    if len(solid_colors_pool) < n_solids:
        return None
    solid_colors = list(map(int, rng.choice(np.array(solid_colors_pool, dtype=int), size=n_solids, replace=False)))

    frame_sizes = [(h + 2, inner_w + 2) for h in heights]
    solid_sizes = []
    for _ in range(n_solids):
        solid_h = int(rng.integers(max(4, min(heights) + 1), min(9, max(heights) + 3)))
        solid_w = int(rng.integers(max(4, inner_w), min(9, inner_w + 3)))
        solid_sizes.append((solid_h, solid_w))

    grid_h = int(rng.integers(18, 25))
    grid_w = int(rng.integers(18, 25))
    all_sizes = frame_sizes + solid_sizes
    positions = _choose_positions(rng, (grid_h, grid_w), all_sizes)
    if positions is None:
        return None

    grid = _empty_grid((grid_h, grid_w))
    frames: list[dict] = []
    for idx, ((outer_h, outer_w), (r0, c0), color, count) in enumerate(zip(frame_sizes, positions[:n_frames], frame_colors, counts)):
        patch = _make_frame_patch(rng, color, outer_h - 2, outer_w - 2, count)
        grid[r0 : r0 + outer_h, c0 : c0 + outer_w] = patch
        frames.append(
            {
                "color": int(color),
                "row": int(r0),
                "col": int(c0),
                "bbox": (int(r0), int(c0), int(r0 + outer_h - 1), int(c0 + outer_w - 1)),
                "interior": patch[1:-1, 1:-1].copy(),
                "count": int(count),
            }
        )

    for (solid_h, solid_w), (r0, c0), color in zip(solid_sizes, positions[n_frames:], solid_colors):
        grid[r0 : r0 + solid_h, c0 : c0 + solid_w] = _make_solid_patch(color, solid_h, solid_w)

    expected = _expected_from_meta(frames)
    if max(expected.shape) > 30:
        return None
    if not _matches_metadata(grid, frames, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
