import random
from typing import Iterable

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
GEN_RETRIES = 400


def _bbox_of(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_rect_frame(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    bbox = _bbox_of(mask)
    if bbox is None:
        return None
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    expected = np.zeros_like(mask, dtype=bool)
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    if np.array_equal(mask, expected):
        return bbox
    return None


def _corner_palette_candidates(grid: np.ndarray) -> list[tuple[dict[int, int], np.ndarray]]:
    h, w = grid.shape
    out: list[tuple[dict[int, int], np.ndarray]] = []
    sizes = (2, 3)
    for n in sizes:
        corner_boxes = [
            (0, 0, n, 2),
            (0, w - 2, n, w),
            (h - n, 0, h, 2),
            (h - n, w - 2, h, w),
        ]
        for r0, c0, r1, c1 in corner_boxes:
            if r0 < 0 or c0 < 0 or r1 > h or c1 > w:
                continue
            sub = grid[r0:r1, c0:c1]
            if sub.shape != (n, 2):
                continue
            if np.any(sub == BG):
                continue
            mapping: dict[int, int] = {}
            ok = True
            for row in sub:
                left, right = int(row[0]), int(row[1])
                if left == right or left in mapping:
                    ok = False
                    break
                mapping[left] = right
            if not ok:
                continue
            mask = np.zeros_like(grid, dtype=bool)
            mask[r0:r1, c0:c1] = True
            if np.any(grid[mask] == BG):
                continue
            out.append((mapping, mask))
    return out


def _extract_palette(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    candidates = _corner_palette_candidates(grid)
    assert candidates, "expected a corner legend"
    frame_like_counts = []
    for mapping, mask in candidates:
        g2 = grid.copy()
        g2[mask] = BG
        count = 0
        for color in np.unique(g2):
            color = int(color)
            if color == BG:
                continue
            if _is_rect_frame(g2 == color) is not None:
                count += 1
        frame_like_counts.append((count, mapping, mask))
    best_count, best_mapping, best_mask = max(frame_like_counts, key=lambda x: x[0])
    assert best_count >= 2, "legend removal should reveal nested frames"
    return best_mapping, best_mask


def _find_frames(grid: np.ndarray, palette_mask: np.ndarray) -> list[dict]:
    g = grid.copy()
    g[palette_mask] = BG
    frames = []
    for color in np.unique(g):
        color = int(color)
        if color == BG:
            continue
        bbox = _is_rect_frame(g == color)
        if bbox is None:
            continue
        r0, c0, r1, c1 = bbox
        frames.append(
            {
                "color": color,
                "bbox": bbox,
                "area": (r1 - r0 + 1) * (c1 - c0 + 1),
            }
        )
    frames.sort(key=lambda item: item["area"], reverse=True)
    assert 2 <= len(frames) <= 3, "expected 2 or 3 nested frames"
    for outer, inner in zip(frames, frames[1:]):
        or0, oc0, or1, oc1 = outer["bbox"]
        ir0, ic0, ir1, ic1 = inner["bbox"]
        assert or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1, "frames must be strictly nested"
    return frames


def _region_mask(shape: tuple[int, int], outer_bbox: tuple[int, int, int, int], inner_bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = outer_bbox
    mask[r0 + 1 : r1, c0 + 1 : c1] = True
    if inner_bbox is not None:
        ir0, ic0, ir1, ic1 = inner_bbox
        mask[ir0 : ir1 + 1, ic0 : ic1 + 1] = False
    return mask


def _apply_rule(grid: np.ndarray, mapping: dict[int, int], palette_mask: np.ndarray, frames: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    out[palette_mask] = BG
    frame_colors = {frame["color"] for frame in frames}
    protected = palette_mask.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        protected[r0, c0 : c1 + 1] = True
        protected[r1, c0 : c1 + 1] = True
        protected[r0 : r1 + 1, c0] = True
        protected[r0 : r1 + 1, c1] = True

    for idx, frame in enumerate(frames):
        outer_bbox = frame["bbox"]
        inner_bbox = frames[idx + 1]["bbox"] if idx + 1 < len(frames) else None
        keep_color = mapping[frame["color"]]
        region = _region_mask(grid.shape, outer_bbox, inner_bbox)
        coords = np.argwhere(region)
        for r, c in coords:
            val = int(out[r, c])
            if val == BG or val == keep_color:
                continue
            if val in frame_colors or protected[r, c]:
                continue
            out[r, c] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter content inside nested rectangular frames according to a corner color legend.

    Each legend row is a two-pixel pair: frame color on the left, allowed content
    color on the right. For every frame, only pixels of its mapped allowed color
    survive in the region governed by that frame. Frames themselves stay unchanged
    and the legend is erased.
    """
    g = np.asarray(grid, dtype=int)
    mapping, palette_mask = _extract_palette(g)
    frames = _find_frames(g, palette_mask)
    assert set(mapping) == {frame["color"] for frame in frames}, "legend must map every frame color exactly once"
    return _apply_rule(g, mapping, palette_mask, frames)


def _sample_frame_layout(rng: random.Random, h: int, w: int, n_frames: int) -> list[tuple[int, int, int, int]] | None:
    min_outer_h = 5 + 4 * (n_frames - 1)
    min_outer_w = 5 + 4 * (n_frames - 1)
    if h < min_outer_h + 2 or w < min_outer_w + 4:
        return None

    outer_h = rng.randint(min_outer_h, h - 2)
    outer_w = rng.randint(min_outer_w, w - 4)
    top = rng.randint(1, h - outer_h - 1)
    left = rng.randint(3, w - outer_w - 1)
    bottom = top + outer_h - 1
    right = left + outer_w - 1

    boxes = [(top, left, bottom, right)]
    for _ in range(1, n_frames):
        pr0, pc0, pr1, pc1 = boxes[-1]
        max_inset_r = (pr1 - pr0 + 1 - 5) // 2
        max_inset_c = (pc1 - pc0 + 1 - 5) // 2
        if max_inset_r < 2 or max_inset_c < 2:
            return None
        inset_top = rng.randint(2, min(3, max_inset_r))
        inset_left = rng.randint(2, min(3, max_inset_c))
        inset_bottom = rng.randint(2, min(3, max_inset_r))
        inset_right = rng.randint(2, min(3, max_inset_c))
        nr0 = pr0 + inset_top
        nc0 = pc0 + inset_left
        nr1 = pr1 - inset_bottom
        nc1 = pc1 - inset_right
        if nr1 - nr0 + 1 < 5 or nc1 - nc0 + 1 < 5:
            return None
        boxes.append((nr0, nc0, nr1, nc1))
    return boxes


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _place_palette(grid: np.ndarray, mapping: dict[int, int], corner: str) -> np.ndarray:
    n = len(mapping)
    rows = list(mapping.items())
    block = np.array(rows, dtype=int)
    h, w = grid.shape
    mask = np.zeros_like(grid, dtype=bool)
    if corner == "tl":
        r0, c0 = 0, 0
        sub = block
    elif corner == "tr":
        r0, c0 = 0, w - 2
        sub = block
    elif corner == "bl":
        r0, c0 = h - n, 0
        sub = block
    else:
        r0, c0 = h - n, w - 2
        sub = block
    grid[r0 : r0 + n, c0 : c0 + 2] = sub
    mask[r0 : r0 + n, c0 : c0 + 2] = True
    return mask


def _random_cells(rng: random.Random, cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    picks = cells[:]
    rng.shuffle(picks)
    return picks[:count]


def _populate_region(
    rng: random.Random,
    grid: np.ndarray,
    region: np.ndarray,
    allowed_color: int,
    other_colors: list[int],
) -> None:
    cells = [(int(r), int(c)) for r, c in np.argwhere(region)]
    assert cells, "region must have usable interior cells"
    total = rng.randint(max(3, len(cells) // 8), max(4, len(cells) // 3))
    total = min(total, len(cells))
    allowed_count = rng.randint(1, max(1, total - 1))
    chosen = _random_cells(rng, cells, total)
    allowed_cells = chosen[:allowed_count]
    other_cells = chosen[allowed_count:]
    for r, c in allowed_cells:
        grid[r, c] = allowed_color
    if not other_colors:
        return
    for idx, (r, c) in enumerate(other_cells):
        grid[r, c] = other_colors[idx % len(other_colors)]
    if len(cells) >= 6 and rng.random() < 0.55:
        seed_r, seed_c = rng.choice(cells)
        obj_color = allowed_color if rng.random() < 0.5 else rng.choice(other_colors)
        for dr, dc in rng.choice(
            [
                [(0, 0), (1, 0)],
                [(0, 0), (0, 1)],
                [(0, 0), (1, 0), (0, 1)],
                [(0, 0), (1, 0), (1, 1)],
            ]
        ):
            rr, cc = seed_r + dr, seed_c + dc
            if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and region[rr, cc]:
                grid[rr, cc] = obj_color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(16, 24)
    w = rng.randint(16, 24)
    n_frames = rng.choice([2, 3])
    grid = np.zeros((h, w), dtype=int)

    layout = _sample_frame_layout(rng, h, w, n_frames)
    if layout is None:
        return None

    possible_corners = []
    if layout[0][1] > 2:
        possible_corners.extend(["tl", "bl"])
    if layout[0][3] < w - 3:
        possible_corners.extend(["tr", "br"])
    if not possible_corners:
        return None
    corner = rng.choice(possible_corners)
    legend_rows = n_frames

    colors = list(COLOR_POOL)
    rng.shuffle(colors)
    frame_colors = colors[:n_frames]
    allowed_colors = colors[n_frames : 2 * n_frames]
    distractor_colors = colors[2 * n_frames :]
    mapping = dict(zip(frame_colors, allowed_colors))

    for bbox, color in zip(layout, frame_colors):
        _draw_frame(grid, bbox, color)

    for idx, bbox in enumerate(layout):
        inner = layout[idx + 1] if idx + 1 < len(layout) else None
        region = _region_mask(grid.shape, bbox, inner)
        others = [c for c in distractor_colors if c != mapping[frame_colors[idx]]]
        others.extend(ac for ac in allowed_colors if ac != mapping[frame_colors[idx]])
        rng.shuffle(others)
        if not others:
            return None
        _populate_region(rng, grid, region, mapping[frame_colors[idx]], others[: max(2, len(others))])

    palette_mask = _place_palette(grid, mapping, corner)

    try:
        out = main(grid)
    except Exception:
        return None

    # Guarantee the transformation is non-trivial in every governed region.
    changed = np.argwhere(grid != out)
    if len(changed) < legend_rows * 2 + 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle instance")
