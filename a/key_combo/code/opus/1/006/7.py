from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
NON_BLACK = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _frame_border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _detect_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == BACKGROUND:
            continue
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        if height < 6 or width < 6:
            continue

        view = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = _frame_border_mask(height, width)
        if not np.all(view[border] == color):
            continue
        if np.any(view[~border] == color):
            continue
        if int(coords.shape[0]) != int(border.sum()):
            continue

        frames.append(
            {
                "color": color,
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "height": height,
                "width": width,
                "area": int((height - 2) * (width - 2)),
            }
        )

    assert 2 <= len(frames) <= 3, "expected 2-3 nested frames"
    frames.sort(key=lambda frame: frame["area"])
    areas = [frame["area"] for frame in frames]
    assert len(areas) == len(set(areas)), "frame areas must define a unique nesting order"
    for inner, outer in zip(frames, frames[1:]):
        assert outer["r0"] < inner["r0"] < inner["r1"] < outer["r1"], "frames must be strictly nested by rows"
        assert outer["c0"] < inner["c0"] < inner["c1"] < outer["c1"], "frames must be strictly nested by cols"
    return frames


def _extract_seed(grid: np.ndarray, frame: dict) -> np.ndarray:
    top = int(frame["r0"]) + 1
    left = int(frame["c0"]) + 1
    bottom = int(frame["r1"])
    right = int(frame["c1"])
    interior = grid[top:bottom, left:right]
    assert interior.size > 0, "frame must have an interior"
    assert int(interior[0, 0]) != BACKGROUND, "seed must start at the inner top-left corner"

    seed_h = 0
    while seed_h < interior.shape[0] and int(interior[seed_h, 0]) != BACKGROUND:
        seed_h += 1
    seed_w = 0
    while seed_w < interior.shape[1] and int(interior[0, seed_w]) != BACKGROUND:
        seed_w += 1

    seed = interior[:seed_h, :seed_w]
    assert seed_h >= 2 and seed_w >= 2, "seed must be at least 2x2"
    assert np.all(seed != BACKGROUND), "seed rectangle must be fully non-black"
    return seed.copy()


def _mirror_tile(seed: np.ndarray) -> np.ndarray:
    top = np.concatenate([seed, np.fliplr(seed)], axis=1)
    return np.concatenate([top, np.flipud(top)], axis=0)


def _plain_tile(seed: np.ndarray) -> np.ndarray:
    return seed


def _write_tiled_pattern(out: np.ndarray, frame: dict, tile: np.ndarray, *, preserve_existing: bool) -> None:
    top = int(frame["r0"]) + 1
    left = int(frame["c0"]) + 1
    bottom = int(frame["r1"])
    right = int(frame["c1"])
    target = out[top:bottom, left:right]
    reps_r = (target.shape[0] + tile.shape[0] - 1) // tile.shape[0]
    reps_c = (target.shape[1] + tile.shape[1] - 1) // tile.shape[1]
    tiled = np.tile(tile, (reps_r, reps_c))[: target.shape[0], : target.shape[1]]

    if preserve_existing:
        mask = target == BACKGROUND
        target[mask] = tiled[mask]
    else:
        target[:, :] = tiled


def _solve_with_order(grid: np.ndarray, frames: list[dict], *, inner_to_outer: bool, mirror: bool, preserve_existing: bool) -> np.ndarray:
    out = np.array(grid, copy=True)
    ordered = frames if inner_to_outer else list(reversed(frames))
    seeds = {frame["area"]: _extract_seed(grid, frame) for frame in frames}
    for frame in ordered:
        seed = seeds[frame["area"]]
        tile = _mirror_tile(seed) if mirror else _plain_tile(seed)
        _write_tiled_pattern(out, frame, tile, preserve_existing=preserve_existing)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each nested frame's corner seed into a repeating four-quadrant tile that fills that frame's interior from the inside out.

    Input structure:
    - The grid is mostly black background and contains 2 or 3 differently colored one-pixel rectangular frames.
    - The frames are strictly nested, and each frame has a small non-black seed block touching its own inner top-left corner.
    - Seed blocks sit away from any deeper frame, so the frame chain and each seed are uniquely identifiable.

    Rule:
    - Detect all rectangular frames and sort them by enclosed area from smallest to largest.
    - For each frame, extract the largest non-black rectangle anchored at the frame's inner top-left corner as that frame's seed.
    - Build a symmetric tile by placing the seed beside its left-right mirror, then stacking a top-bottom mirror of that strip underneath.
    - Repeat that tile from the frame's inner top-left corner to cover the frame interior, crop at the frame boundary, and never overwrite existing non-black cells.
    - Process frames from inner to outer so inner borders and earlier fills stay visible.

    Key insight:
    - The nested-frame order is part of the semantics: only an inner-to-outer pass preserves the structures already created inside deeper frames while the mirrored tiling spreads outward layer by layer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    frames = _detect_frames(g)
    return _solve_with_order(g, frames, inner_to_outer=True, mirror=True, preserve_existing=True)


def _sample_seed(size: int, colors: tuple[int, int]) -> np.ndarray | None:
    for _ in range(200):
        seed = np.array([random.choice(colors) for _ in range(size * size)], dtype=int).reshape(size, size)
        if len(set(map(int, seed.flat))) < 2:
            continue
        if np.array_equal(seed, np.fliplr(seed)):
            continue
        if np.array_equal(seed, np.flipud(seed)):
            continue
        return seed
    return None


def _sample_seed_sizes(count: int) -> list[int]:
    if count == 2:
        sizes = [2, 3]
    else:
        sizes = [2, 3, random.choice((2, 3))]
    random.shuffle(sizes)
    return sizes


def _sample_innermost_bbox(seed_size: int) -> tuple[int, int]:
    interior_h = random.randint(seed_size * 2, seed_size * 2 + 4)
    interior_w = random.randint(seed_size * 2, seed_size * 2 + 5)
    if interior_h == interior_w and random.random() < 0.6:
        interior_w += random.choice((-1, 1))
        interior_w = max(seed_size * 2, interior_w)
    return interior_h + 2, interior_w + 2


def _build_spec() -> dict | None:
    frame_count = 3 if random.random() < 0.75 else 2
    seed_sizes = _sample_seed_sizes(frame_count)
    palette = random.sample(list(NON_BLACK), frame_count * 3)
    frame_colors = palette[:frame_count]
    seed_colors = palette[frame_count:]

    frames: list[dict] = []
    inner_h, inner_w = _sample_innermost_bbox(seed_sizes[0])
    frames.append(
        {
            "bbox_h": inner_h,
            "bbox_w": inner_w,
            "seed_size": seed_sizes[0],
            "frame_color": frame_colors[0],
            "seed_palette": (seed_colors[0], seed_colors[1]),
        }
    )

    for idx in range(1, frame_count):
        seed_size = seed_sizes[idx]
        child = frames[-1]
        top_gap = random.randint(seed_size + 1, seed_size + 3)
        left_gap = random.randint(seed_size + 1, seed_size + 3)
        bottom_gap = random.randint(1, 3)
        right_gap = random.randint(1, 3)
        bbox_h = int(child["bbox_h"]) + top_gap + bottom_gap + 2
        bbox_w = int(child["bbox_w"]) + left_gap + right_gap + 2
        frames.append(
            {
                "bbox_h": bbox_h,
                "bbox_w": bbox_w,
                "seed_size": seed_size,
                "frame_color": frame_colors[idx],
                "seed_palette": (seed_colors[idx * 2], seed_colors[idx * 2 + 1]),
                "child_top_gap": top_gap,
                "child_left_gap": left_gap,
                "child_bottom_gap": bottom_gap,
                "child_right_gap": right_gap,
            }
        )

    outer = frames[-1]
    top_margin = random.randint(1, 3)
    left_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    grid_h = int(outer["bbox_h"]) + top_margin + bottom_margin
    grid_w = int(outer["bbox_w"]) + left_margin + right_margin
    if grid_h > 30 or grid_w > 30:
        return None

    frames[-1]["r0"] = top_margin
    frames[-1]["c0"] = left_margin
    for idx in range(frame_count - 2, -1, -1):
        parent = frames[idx + 1]
        frames[idx]["r0"] = int(parent["r0"]) + 1 + int(parent["child_top_gap"])
        frames[idx]["c0"] = int(parent["c0"]) + 1 + int(parent["child_left_gap"])

    for frame in frames:
        frame["r1"] = int(frame["r0"]) + int(frame["bbox_h"]) - 1
        frame["c1"] = int(frame["c0"]) + int(frame["bbox_w"]) - 1
        frame["area"] = int((int(frame["bbox_h"]) - 2) * (int(frame["bbox_w"]) - 2))
        seed = _sample_seed(int(frame["seed_size"]), tuple(map(int, frame["seed_palette"])))
        if seed is None:
            return None
        frame["seed"] = seed

    return {"grid_shape": (grid_h, grid_w), "frames_inner_to_outer": frames}


def _draw_frame(grid: np.ndarray, frame: dict) -> None:
    color = int(frame["frame_color"])
    r0, c0, r1, c1 = int(frame["r0"]), int(frame["c0"]), int(frame["r1"]), int(frame["c1"])
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _render_input(spec: dict) -> np.ndarray:
    grid = np.zeros(spec["grid_shape"], dtype=int)
    for frame in reversed(spec["frames_inner_to_outer"]):
        _draw_frame(grid, frame)
    for frame in spec["frames_inner_to_outer"]:
        r0 = int(frame["r0"]) + 1
        c0 = int(frame["c0"]) + 1
        seed = frame["seed"]
        grid[r0 : r0 + seed.shape[0], c0 : c0 + seed.shape[1]] = seed
    return grid


def _oracle_output(spec: dict, grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    for frame in spec["frames_inner_to_outer"]:
        tile = _mirror_tile(frame["seed"])
        oracle_frame = {
            "r0": frame["r0"],
            "c0": frame["c0"],
            "r1": frame["r1"],
            "c1": frame["c1"],
        }
        _write_tiled_pattern(out, oracle_frame, tile, preserve_existing=True)
    return out


def _no_mirror_output(spec: dict, grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    for frame in spec["frames_inner_to_outer"]:
        _write_tiled_pattern(
            out,
            {"r0": frame["r0"], "c0": frame["c0"], "r1": frame["r1"], "c1": frame["c1"]},
            frame["seed"],
            preserve_existing=True,
        )
    return out


def _outer_first_output(spec: dict, grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    for frame in reversed(spec["frames_inner_to_outer"]):
        _write_tiled_pattern(
            out,
            {"r0": frame["r0"], "c0": frame["c0"], "r1": frame["r1"], "c1": frame["c1"]},
            _mirror_tile(frame["seed"]),
            preserve_existing=True,
        )
    return out


def _overwrite_output(spec: dict, grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    for frame in spec["frames_inner_to_outer"]:
        _write_tiled_pattern(
            out,
            {"r0": frame["r0"], "c0": frame["c0"], "r1": frame["r1"], "c1": frame["c1"]},
            _mirror_tile(frame["seed"]),
            preserve_existing=False,
        )
    return out


def _spec_semantics_ok(spec: dict, grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        detected = _detect_frames(grid)
        if len(detected) != len(spec["frames_inner_to_outer"]):
            return False
        detected_boxes = [(f["r0"], f["c0"], f["r1"], f["c1"]) for f in detected]
        spec_boxes = [(int(f["r0"]), int(f["c0"]), int(f["r1"]), int(f["c1"])) for f in spec["frames_inner_to_outer"]]
        if detected_boxes != spec_boxes:
            return False

        for frame in detected:
            seed = _extract_seed(grid, frame)
            matching = next(item for item in spec["frames_inner_to_outer"] if int(item["area"]) == int(frame["area"]))
            if not np.array_equal(seed, matching["seed"]):
                return False

        if np.array_equal(expected, _no_mirror_output(spec, grid)):
            return False
        if np.array_equal(expected, _outer_first_output(spec, grid)):
            return False
        if np.array_equal(expected, _overwrite_output(spec, grid)):
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        spec = _build_spec()
        if spec is None:
            continue
        grid = _render_input(spec)
        expected = _oracle_output(spec, grid)
        if not _spec_semantics_ok(spec, grid, expected):
            continue
        try:
            actual = main(grid)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue
        return grid
    raise RuntimeError("failed to generate a valid nested mirrored tiling input")
