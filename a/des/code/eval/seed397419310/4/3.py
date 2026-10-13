from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _Frame:
    coords: np.ndarray
    bbox: tuple[int, int, int, int]
    depth: int = 0


GRID_H_MIN = 14
GRID_H_MAX = 30
GRID_W_MIN = 14
GRID_W_MAX = 30
MAIN_DEPTH_MIN = 1
MAIN_DEPTH_MAX = 4
GEN_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _frame_size(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _frame_area(bbox: tuple[int, int, int, int]) -> int:
    h, w = _frame_size(bbox)
    return 2 * (h + w) - 4


def _frame_coords_from_bbox(bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    pts = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        if r1 != r0:
            pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        if c1 != c0:
            pts.append((r, c1))
    return np.asarray(pts, dtype=int)


def _is_rect_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * (h + w) - 4:
        return False
    coord_set = {(int(r), int(c)) for r, c in coords}
    for r, c in _frame_coords_from_bbox((r0, c0, r1, c1)):
        if (int(r), int(c)) not in coord_set:
            return False
    return True


def _is_palette_run(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if int(rows.min()) != int(rows.max()):
        return False
    width = int(cols.max()) - int(cols.min()) + 1
    if len(coords) != width:
        return False
    row = int(rows[0])
    return min(row, shape[0] - 1 - row) <= 2


def _extract_palette_and_frames(grid: np.ndarray, bg: int) -> tuple[np.ndarray, list[int], list[_Frame]]:
    comps = _components(grid != bg)
    palette_candidates: list[np.ndarray] = []
    frames: list[_Frame] = []
    for coords in comps:
        colors = {int(grid[r, c]) for r, c in coords}
        if _is_palette_run(coords, grid.shape):
            palette_candidates.append(coords)
            continue
        if _is_rect_frame(coords):
            assert len(colors) == 1
            frames.append(_Frame(coords=coords, bbox=_bbox(coords)))
            continue
        raise AssertionError("unexpected foreground component")

    assert len(palette_candidates) == 1
    palette_coords = palette_candidates[0]
    order = np.argsort(palette_coords[:, 1])
    palette_colors = [int(grid[r, c]) for r, c in palette_coords[order]]
    assert frames
    return palette_coords, palette_colors, frames


def _assign_depths(frames: list[_Frame]) -> int:
    max_depth = 0
    for frame in frames:
        r0, c0, r1, c1 = frame.bbox
        depth = 1
        for other in frames:
            or0, oc0, or1, oc1 = other.bbox
            if or0 < r0 and oc0 < c0 and or1 > r1 and oc1 > c1:
                depth += 1
        frame.depth = depth
        max_depth = max(max_depth, depth)
    return max_depth


def _paint_center_square(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    side: int,
    color: int,
) -> None:
    r0, c0, r1, c1 = bbox
    inner_top = r0 + 1
    inner_left = c0 + 1
    inner_h = max(0, r1 - r0 - 1)
    inner_w = max(0, c1 - c0 - 1)
    start_r = inner_top + (inner_h - side) // 2
    start_c = inner_left + (inner_w - side) // 2
    end_r = start_r + side
    end_c = start_c + side
    rr0 = max(inner_top, start_r)
    cc0 = max(inner_left, start_c)
    rr1 = min(r1, end_r)
    cc1 = min(c1, end_c)
    if rr0 < rr1 and cc0 < cc1:
        grid[rr0:rr1, cc0:cc1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested rectangular frames by containment depth and add a centered square to the deepest layer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _most_common_color(g)

    palette_coords, palette_colors, frames = _extract_palette_and_frames(g, bg)
    max_depth = _assign_depths(frames)
    assert len(palette_colors) >= max_depth

    out = g.copy()
    out[tuple(palette_coords.T)] = bg

    for frame in frames:
        new_color = palette_colors[frame.depth - 1]
        out[tuple(frame.coords.T)] = int(new_color)

    deepest_color = palette_colors[max_depth - 1]
    for frame in frames:
        if frame.depth == max_depth:
            _paint_center_square(out, frame.bbox, max_depth, deepest_color)

    return out


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)


def _make_chain(
    rng: np.random.Generator,
    depth: int,
    clip_mode: str,
) -> list[tuple[int, int, int, int]]:
    if clip_mode == "none":
        inner_h = int(rng.integers(depth, depth + 4))
        inner_w = int(rng.integers(depth, depth + 4))
    elif clip_mode == "row":
        inner_h = int(rng.integers(1, depth))
        inner_w = int(rng.integers(depth, depth + 4))
    elif clip_mode == "col":
        inner_h = int(rng.integers(depth, depth + 4))
        inner_w = int(rng.integers(1, depth))
    elif clip_mode == "both":
        inner_h = int(rng.integers(1, depth))
        inner_w = int(rng.integers(1, depth))
    else:
        inner_h = int(rng.integers(1, 7))
        inner_w = int(rng.integers(1, 7))

    boxes = [(0, 0, inner_h + 1, inner_w + 1)]
    for _ in range(depth - 1):
        r0, c0, r1, c1 = boxes[-1]
        top = int(rng.integers(2, 4))
        bottom = int(rng.integers(2, 4))
        left = int(rng.integers(2, 4))
        right = int(rng.integers(2, 4))
        boxes.append((r0 - top, c0 - left, r1 + bottom, c1 + right))

    min_r = min(b[0] for b in boxes)
    min_c = min(b[1] for b in boxes)
    shifted = [(r0 - min_r, c0 - min_c, r1 - min_r, c1 - min_c) for r0, c0, r1, c1 in boxes]
    shifted.reverse()
    return shifted


def _choose_extra_depths(rng: np.random.Generator, main_depth: int) -> list[int]:
    if main_depth <= 1:
        return []
    if main_depth == 4:
        max_extra_chains = 1
    else:
        max_extra_chains = 2
    count = int(rng.integers(0, max_extra_chains + 1))
    depths: list[int] = []
    budget = 8 - main_depth
    for _ in range(count):
        max_depth = min(main_depth - 1, budget)
        if max_depth < 1:
            break
        depth = int(rng.integers(1, max_depth + 1))
        depths.append(depth)
        budget -= depth
    return depths


def _build_layout(
    rng: np.random.Generator,
    blocks: list[dict],
    palette_len: int,
) -> tuple[int, int, int, int, list[tuple[int, int]]] | None:
    widths = [int(block["width"]) for block in blocks]
    heights = [int(block["height"]) for block in blocks]
    order = list(rng.permutation(len(blocks)))
    gaps = [int(rng.integers(2, 6)) for _ in range(max(0, len(blocks) - 1))]
    band_w = sum(widths[i] for i in order) + sum(gaps)
    max_h = max(heights)
    band_h = max_h + int(rng.integers(0, 4))

    min_w = max(GRID_W_MIN, band_w, palette_len)
    if min_w > GRID_W_MAX:
        return None
    max_w = min(GRID_W_MAX, min_w + 6)
    grid_w = int(rng.integers(min_w, max_w + 1))
    band_x = int(rng.integers(0, grid_w - band_w + 1))

    top_palette = bool(rng.integers(0, 2))
    if top_palette:
        palette_inset = int(rng.integers(0, 2))
        bottom_pad = int(rng.integers(0, 3))
        grid_h = palette_inset + 2 + band_h + bottom_pad
        palette_row = palette_inset
        content_top = palette_row + 2
    else:
        top_pad = int(rng.integers(0, 3))
        palette_inset = int(rng.integers(0, 2))
        grid_h = top_pad + band_h + 2 + palette_inset
        content_top = top_pad
        palette_row = content_top + band_h + 1

    if not (GRID_H_MIN <= grid_h <= GRID_H_MAX):
        return None

    positions = [(-1, -1)] * len(blocks)
    current_x = band_x
    for idx, block_idx in enumerate(order):
        block_h = heights[block_idx]
        top_in_band = int(rng.integers(0, band_h - block_h + 1))
        positions[block_idx] = (content_top + top_in_band, current_x)
        current_x += widths[block_idx]
        if idx < len(gaps):
            current_x += gaps[idx]

    return grid_h, grid_w, palette_row, content_top, positions


def _render_expected(
    grid: np.ndarray,
    palette_row: int,
    palette_col: int,
    palette_colors: list[int],
    frames: list[dict],
    max_depth: int,
) -> np.ndarray:
    out = grid.copy()
    out[palette_row, palette_col : palette_col + len(palette_colors)] = _most_common_color(grid)
    for frame in frames:
        coords = _frame_coords_from_bbox(frame["bbox"])
        out[tuple(coords.T)] = int(palette_colors[frame["depth"] - 1])
    deepest_color = palette_colors[max_depth - 1]
    for frame in frames:
        if frame["depth"] == max_depth:
            _paint_center_square(out, frame["bbox"], max_depth, deepest_color)
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    if rng.random() < 0.5:
        main_depth = int(rng.integers(2, MAIN_DEPTH_MAX + 1))
        clip_mode = ["row", "col", "both"][int(rng.integers(0, 3))]
    else:
        main_depth = int(rng.integers(MAIN_DEPTH_MIN, MAIN_DEPTH_MAX + 1))
        clip_mode = "none"

    bg = int(rng.integers(0, 10))
    all_colors = [c for c in range(10) if c != bg]

    extra_depths = _choose_extra_depths(rng, main_depth)
    main_chain = _make_chain(rng, main_depth, clip_mode)
    chains = [{"depths": main_depth, "boxes": main_chain, "is_main": True}]
    for depth in extra_depths:
        chains.append({"depths": depth, "boxes": _make_chain(rng, depth, "free"), "is_main": False})

    total_frames = sum(chain["depths"] for chain in chains)
    if total_frames > len(all_colors):
        return None

    blocks = []
    for chain in chains:
        outer = chain["boxes"][0]
        h, w = _frame_size(outer)
        blocks.append(
            {
                "frames": chain["boxes"],
                "height": h,
                "width": w,
                "is_main": chain["is_main"],
                "depths": chain["depths"],
            }
        )

    layout = _build_layout(rng, blocks, main_depth)
    if layout is None:
        return None
    grid_h, grid_w, palette_row, _content_top, positions = layout

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    palette_colors = list(rng.choice(np.asarray(all_colors, dtype=int), size=main_depth, replace=False))
    palette_col = int(rng.integers(0, grid_w - main_depth + 1))
    grid[palette_row, palette_col : palette_col + main_depth] = np.asarray(palette_colors, dtype=int)

    frame_colors = list(rng.choice(np.asarray(all_colors, dtype=int), size=total_frames, replace=False))
    color_idx = 0
    frame_records: list[dict] = []
    for block, (top, left) in zip(blocks, positions):
        for depth, rel_bbox in enumerate(block["frames"], start=1):
            r0, c0, r1, c1 = rel_bbox
            abs_bbox = (r0 + top, c0 + left, r1 + top, c1 + left)
            _paint_frame(grid, abs_bbox, frame_colors[color_idx])
            frame_records.append(
                {
                    "bbox": abs_bbox,
                    "depth": depth,
                    "is_main": block["is_main"] and depth == block["depths"],
                }
            )
            color_idx += 1

    if _most_common_color(grid) != bg:
        return None

    depths = [frame["depth"] for frame in frame_records]
    if depths.count(max(depths)) != 1:
        return None

    expected = _render_expected(grid, palette_row, palette_col, palette_colors, frame_records, main_depth)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
