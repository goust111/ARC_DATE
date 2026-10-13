from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
BLUE = 1
FRAME_COLORS = tuple(range(2, 10))
GEN_RETRY_LIMIT = 500

GRID_HEIGHT_RANGE = (18, 26)
GRID_WIDTH_RANGE = (18, 26)
WIDE_OUTER_HEIGHT_RANGE = (6, 8)
WIDE_OUTER_WIDTH_RANGE = (8, 11)
TALL_OUTER_HEIGHT_RANGE = (8, 11)
TALL_OUTER_WIDTH_RANGE = (6, 8)


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        directions = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in directions:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, clip: bool = False) -> None:
    h, w = patch.shape
    dh, dw = dst.shape
    if not clip:
        dst[r0 : r0 + h, c0 : c0 + w] = patch
        return
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(dh, int(r0) + h)
    cc1 = min(dw, int(c0) + w)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    pr0 = rr0 - int(r0)
    pc0 = cc0 - int(c0)
    pr1 = pr0 + (rr1 - rr0)
    pc1 = pc0 + (cc1 - cc0)
    dst[rr0:rr1, cc0:cc1] = patch[pr0:pr1, pc0:pc1]


def _frame_border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _interval_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1 - 1
    if b1 < a0:
        return a0 - b1 - 1
    return 0


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    row_gap = _interval_gap(a[0], a[2], b[0], b[2])
    col_gap = _interval_gap(a[1], a[3], b[1], b[3])
    return bool(row_gap >= gap or col_gap >= gap)


def _half_indices(length: int) -> tuple[list[int], list[int]]:
    left_or_top = [idx for idx in range(length) if 2 * idx < length - 1]
    right_or_bottom = [idx for idx in range(length) if 2 * idx > length - 1]
    return left_or_top, right_or_bottom


def _mirror_axis(inner_height: int, inner_width: int) -> str:
    if inner_width > inner_height:
        return "vertical"
    if inner_height > inner_width:
        return "horizontal"
    raise AssertionError("square interiors are not allowed")


def _build_sprite_from_seed(seed_mask: np.ndarray, axis: str) -> np.ndarray:
    full = seed_mask.copy()
    if axis == "vertical":
        full |= np.fliplr(seed_mask)
    else:
        full |= np.flipud(seed_mask)
    return full.astype(int)


def _build_sprite_variants(grid: np.ndarray, frame: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = frame["bbox"]
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    seed = (inner == BLUE).astype(int)
    axis = _mirror_axis(*seed.shape)
    correct = _build_sprite_from_seed(seed, axis)
    wrong_axis = "horizontal" if axis == "vertical" else "vertical"
    wrong = _build_sprite_from_seed(seed, wrong_axis)
    return seed, correct, wrong


def _detect_frames(grid: np.ndarray) -> list[dict]:
    frame_mask = (grid != BACKGROUND) & (grid != BLUE)
    comps = _components(frame_mask, connectivity=4)
    frames = []
    for coords in comps:
        vals = grid[coords[:, 0], coords[:, 1]]
        if np.unique(vals).size != 1:
            continue
        color = int(vals[0])
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width < 3:
            continue
        expected = _frame_border_mask(height, width)
        observed = grid[r0 : r1 + 1, c0 : c1 + 1] == color
        if not np.array_equal(expected, observed):
            continue
        frames.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "outer_h": height,
                "outer_w": width,
                "inner_h": height - 2,
                "inner_w": width - 2,
                "area": height * width,
            }
        )
    frames.sort(key=lambda frame: (frame["bbox"][0], frame["bbox"][1]))
    return frames


def _priority_key(frame: dict) -> tuple[int, int, int, int]:
    r0, c0, _r1, _c1 = frame["bbox"]
    return (int(frame["area"]), int(r0 * r0 + c0 * c0), int(r0), int(c0))


def _placement_origins(source: dict, shape: tuple[int, int], *, stride: tuple[int, int] | None = None, only_original: bool = False) -> list[tuple[int, int]]:
    grid_h, grid_w = shape
    step_h, step_w = stride if stride is not None else (source["outer_h"], source["outer_w"])
    start_r = source["bbox"][0] + 1
    start_c = source["bbox"][1] + 1
    sprite_h, sprite_w = source["sprite"].shape
    origins = [(start_r, start_c)]
    if only_original:
        return origins
    for sign in (-1, 1):
        k = 1
        while True:
            rr = start_r + k * step_h
            cc = start_c + sign * k * step_w
            if rr >= grid_h or cc >= grid_w or cc + sprite_w <= 0:
                break
            origins.append((rr, cc))
            k += 1
    return origins


def _render_output(
    shape: tuple[int, int],
    sources: list[dict],
    *,
    sprite_overrides: dict[int, np.ndarray] | None = None,
    stride_overrides: dict[int, tuple[int, int]] | None = None,
    reverse_priority: bool = False,
    only_original: bool = False,
) -> np.ndarray:
    sprite_overrides = sprite_overrides or {}
    stride_overrides = stride_overrides or {}
    ranked = []
    for idx, source in enumerate(sources):
        ranked.append((idx, source, _priority_key(source)))
    ranked.sort(key=lambda item: item[2], reverse=not reverse_priority)

    out = np.zeros(shape, dtype=int)
    for idx, source, _priority in ranked:
        sprite = sprite_overrides.get(idx, source["sprite"])
        stride = stride_overrides.get(idx)
        tmp_source = dict(source)
        tmp_source["sprite"] = sprite
        for rr, cc in _placement_origins(tmp_source, shape, stride=stride, only_original=only_original):
            _paste(out, sprite, rr, cc, clip=True)
    return out


def _parse_sources(grid: np.ndarray) -> list[dict]:
    frames = _detect_frames(grid)
    assert len(frames) == 2, "expected exactly two rectangular frames"
    sources = []
    for frame in frames:
        seed, sprite, wrong_sprite = _build_sprite_variants(grid, frame)
        blue_cells = int(seed.sum())
        assert 3 <= blue_cells <= 5, "expected 3-5 blue seed cells"
        comps = _components(seed.astype(bool), connectivity=4)
        assert len(comps) == 1, "seed must be a single 4-connected component"
        axis = _mirror_axis(frame["inner_h"], frame["inner_w"])
        frame["seed"] = seed
        frame["sprite"] = sprite
        frame["wrong_sprite"] = wrong_sprite
        frame["axis"] = axis
        sources.append(frame)

    areas = {source["area"] for source in sources}
    assert len(areas) == 2, "frame areas must be distinct"
    return sources


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each frame's inner blue seed by aspect ratio, then stamp the resulting sprite along both downward diagonals with smaller-frame area taking overlap priority.

    Input structure:
    - A black background contains exactly two non-overlapping 1-pixel-thick rectangular frames in non-blue colors.
    - Each frame encloses a 3-5 pixel 4-connected blue seed shape inside its interior, confined to one quadrant and away from the mirror center line.
    - One frame is wide and the other is tall, so their interior aspect ratios unambiguously choose different mirror axes.

    Rule:
    - For each frame, inspect its interior size: if the interior is wider than tall, mirror the blue seed left-right across the interior vertical center line; if taller than wide, mirror it top-bottom across the interior horizontal center line.
    - Crop the frame interior only, keeping blue cells and black interior background but discarding the frame wall; this interior crop is the sprite.
    - Place that sprite at the frame's original interior location and also at offsets (kH, kW) and (kH, -kW) for k=1,2,..., where H and W are the outer frame height and width; clip partial stamps at the grid border and stop each diagonal once the whole sprite is outside.
    - When stamps from different source frames overlap, every covered pixel takes the value from the source whose outer-frame area is smaller; if areas tie, prefer the source whose frame top-left corner is nearer the grid top-left.

    Key insight:
    - The same frame contributes three independent clues: its interior aspect ratio chooses the mirror axis, its outer size chooses the diagonal stride, and its outer area chooses who wins blue-vs-black conflicts in overlaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sources = _parse_sources(g)
    return _render_output(g.shape, sources)


def _random_polyomino(height: int, width: int, area: int) -> set[tuple[int, int]] | None:
    if area > height * width:
        return None
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    return cells


def _quadrant_seed(inner_h: int, inner_w: int) -> np.ndarray | None:
    row_top, row_bottom = _half_indices(inner_h)
    col_left, col_right = _half_indices(inner_w)
    row_halves = [row_top, row_bottom]
    col_halves = [col_left, col_right]
    choices = [(rows, cols) for rows in row_halves for cols in col_halves if rows and cols]
    random.shuffle(choices)
    for rows, cols in choices:
        max_cells = min(5, len(rows) * len(cols))
        if max_cells < 3:
            continue
        target = random.randint(3, max_cells)
        shape = _random_polyomino(len(rows), len(cols), target)
        if shape is None:
            continue
        seed = np.zeros((inner_h, inner_w), dtype=int)
        for rr, cc in shape:
            seed[rows[rr], cols[cc]] = 1
        if len(_components(seed.astype(bool), connectivity=4)) != 1:
            continue
        return seed
    return None


def _candidate_frame(is_wide: bool, grid_h: int, grid_w: int) -> dict:
    for _ in range(100):
        if is_wide:
            outer_h = random.randint(*WIDE_OUTER_HEIGHT_RANGE)
            outer_w = random.randint(*WIDE_OUTER_WIDTH_RANGE)
            if outer_w <= outer_h:
                continue
        else:
            outer_h = random.randint(*TALL_OUTER_HEIGHT_RANGE)
            outer_w = random.randint(*TALL_OUTER_WIDTH_RANGE)
            if outer_h <= outer_w:
                continue
        break
    else:
        raise AssertionError("failed to sample a non-square frame")
    max_top = max(1, grid_h - outer_h - 2)
    top = random.randint(1, max_top)
    left = random.randint(1, max(1, grid_w - outer_w - 2))
    return {
        "bbox": (top, left, top + outer_h - 1, left + outer_w - 1),
        "outer_h": outer_h,
        "outer_w": outer_w,
        "inner_h": outer_h - 2,
        "inner_w": outer_w - 2,
        "area": outer_h * outer_w,
    }


def _place_two_frames(grid_h: int, grid_w: int) -> tuple[dict, dict] | None:
    for _ in range(200):
        wide = _candidate_frame(True, grid_h, grid_w)
        tall = _candidate_frame(False, grid_h, grid_w)
        if wide["area"] == tall["area"]:
            continue
        if not _boxes_separated(wide["bbox"], tall["bbox"], gap=1):
            continue
        ok = True
        for frame in (wide, tall):
            top, left, _bottom, _right = frame["bbox"]
            start_r = top + 1
            start_c = left + 1
            sprite_h = frame["inner_h"]
            sprite_w = frame["inner_w"]
            if start_r + frame["outer_h"] >= grid_h:
                ok = False
                break
            if start_c - frame["outer_w"] + sprite_w <= 0:
                ok = False
                break
            if start_c + frame["outer_w"] >= grid_w:
                ok = False
                break
        if ok:
            return wide, tall
    return None


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _coverage_map(shape: tuple[int, int], source: dict) -> dict[tuple[int, int], int]:
    sprite = source["sprite"]
    h, w = sprite.shape
    out: dict[tuple[int, int], int] = {}
    for r0, c0 in _placement_origins(source, shape):
        for rr in range(h):
            for cc in range(w):
                ar = r0 + rr
                ac = c0 + cc
                if 0 <= ar < shape[0] and 0 <= ac < shape[1]:
                    out[(ar, ac)] = int(sprite[rr, cc])
    return out


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        sources = _parse_sources(grid)
    except AssertionError:
        return False

    output = _render_output(grid.shape, sources)
    if np.array_equal(output, _render_output(grid.shape, sources, only_original=True)):
        return False

    for idx, source in enumerate(sources):
        if np.array_equal(source["sprite"], source["wrong_sprite"]):
            return False
        wrong_axis_output = _render_output(grid.shape, sources, sprite_overrides={idx: source["wrong_sprite"]})
        if np.array_equal(output, wrong_axis_output):
            return False
        seed_only_output = _render_output(grid.shape, sources, sprite_overrides={idx: source["seed"]})
        if np.array_equal(output, seed_only_output):
            return False
        wrong_stride_output = _render_output(
            grid.shape,
            sources,
            stride_overrides={idx: (source["inner_h"], source["inner_w"])},
        )
        if np.array_equal(output, wrong_stride_output):
            return False

    cover_a = _coverage_map(grid.shape, sources[0])
    cover_b = _coverage_map(grid.shape, sources[1])
    disagreements = [pos for pos in set(cover_a) & set(cover_b) if cover_a[pos] != cover_b[pos]]
    if not disagreements:
        return False

    reversed_output = _render_output(grid.shape, sources, reverse_priority=True)
    if np.array_equal(output, reversed_output):
        return False

    observed = main(grid)
    return bool(np.array_equal(observed, output))


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_HEIGHT_RANGE)
    grid_w = random.randint(*GRID_WIDTH_RANGE)
    placed = _place_two_frames(grid_h, grid_w)
    if placed is None:
        return None
    wide, tall = placed
    colors = random.sample(FRAME_COLORS, 2)
    wide["color"] = colors[0]
    tall["color"] = colors[1]

    wide_seed = _quadrant_seed(wide["inner_h"], wide["inner_w"])
    tall_seed = _quadrant_seed(tall["inner_h"], tall["inner_w"])
    if wide_seed is None or tall_seed is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for frame, seed in ((wide, wide_seed), (tall, tall_seed)):
        _draw_frame(grid, frame["bbox"], frame["color"])
        r0, c0, _r1, _c1 = frame["bbox"]
        inner = grid[r0 + 1 : r0 + 1 + frame["inner_h"], c0 + 1 : c0 + 1 + frame["inner_w"]]
        inner[seed.astype(bool)] = BLUE

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
