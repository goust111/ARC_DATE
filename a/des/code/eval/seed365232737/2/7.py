from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
GREY = 5
ALL_FRAME_COLORS = tuple(c for c in range(10) if c not in {BG, GREY})

PAIR_MODES = ("both", "width_only", "height_only", "none")
PAIR_COUNT_CHOICES = (1, 2, 3)
EXTRA_FRAME_MAX = 2

GRID_SIDE_MAX = 30
FRAME_H_MIN = 4
FRAME_H_MAX = 9
FRAME_W_MIN = 4
FRAME_W_MAX = 10
PLACEMENT_PADDING = 1
GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _label_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc), h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _ring_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((int(height), int(width)), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _frame_from_component(coords: np.ndarray, color: int) -> dict | None:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None
    local = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        local[int(r) - r0, int(c) - c0] = True
    if not np.array_equal(local, _ring_mask(h, w)):
        return None
    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "h": int(h),
        "w": int(w),
    }


def _find_frames(grid: np.ndarray) -> dict[int, dict]:
    frames: dict[int, dict] = {}
    for color in sorted(int(v) for v in np.unique(grid)):
        if color in {BG, GREY}:
            continue
        for coords in _label_components(grid == int(color)):
            frame = _frame_from_component(coords, int(color))
            if frame is None:
                continue
            assert int(color) not in frames, "frame colors must be unique"
            frames[int(color)] = frame
    assert frames, "expected at least one frame"
    return frames


def _read_indicator_pairs(grid: np.ndarray) -> list[tuple[int, int]]:
    grey_coords = np.argwhere(grid == GREY)
    assert grey_coords.size > 0, "expected a grey pairing indicator"
    r0, c0, r1, c1 = _bbox_from_coords(grey_coords)

    pairs: list[tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        row_colors = [
            int(grid[r, c])
            for c in range(c0, c1 + 1)
            if int(grid[r, c]) not in {BG, GREY}
        ]
        if not row_colors:
            continue
        assert len(row_colors) == 2, "each indicator row must show exactly two colors"
        assert row_colors[0] != row_colors[1], "indicator pairs must reference two different frames"
        pairs.append((int(row_colors[0]), int(row_colors[1])))
    assert pairs, "expected at least one pair row in the indicator"
    return pairs


def _edge_plan(frames: dict[int, dict], pairs: list[tuple[int, int]]) -> dict[int, dict[str, int]]:
    edge_colors = {
        int(color): {
            "top": int(color),
            "bottom": int(color),
            "left": int(color),
            "right": int(color),
        }
        for color in frames
    }
    for color_a, color_b in pairs:
        assert int(color_a) in frames and int(color_b) in frames, "indicator colors must match frames"
        frame_a = frames[int(color_a)]
        frame_b = frames[int(color_b)]
        if int(frame_a["w"]) == int(frame_b["w"]):
            edge_colors[int(color_a)]["top"] = int(color_b)
            edge_colors[int(color_a)]["bottom"] = int(color_b)
            edge_colors[int(color_b)]["top"] = int(color_a)
            edge_colors[int(color_b)]["bottom"] = int(color_a)
        if int(frame_a["h"]) == int(frame_b["h"]):
            edge_colors[int(color_a)]["left"] = int(color_b)
            edge_colors[int(color_a)]["right"] = int(color_b)
            edge_colors[int(color_b)]["left"] = int(color_a)
            edge_colors[int(color_b)]["right"] = int(color_a)
    return edge_colors


def _paint_frame(out: np.ndarray, frame: dict, colors: dict[str, int]) -> None:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    out[r0, c0 : c1 + 1] = int(colors["top"])
    out[r1, c0 : c1 + 1] = int(colors["bottom"])
    if r1 - r0 >= 2:
        out[r0 + 1 : r1, c0] = int(colors["left"])
        out[r0 + 1 : r1, c1] = int(colors["right"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split paired monochrome frames into edge segments and swap only equal-length segments.
    """
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    pairs = _read_indicator_pairs(g)
    edge_colors = _edge_plan(frames, pairs)

    out = g.copy()
    for color, frame in frames.items():
        _paint_frame(out, frame, edge_colors[int(color)])
    return out


def _sample_pair_modes(rng: np.random.Generator, pair_count: int) -> list[str]:
    modes = [PAIR_MODES[int(rng.integers(0, len(PAIR_MODES)))] for _ in range(int(pair_count))]
    if all(mode == "none" for mode in modes):
        idx = int(rng.integers(0, int(pair_count)))
        modes[idx] = PAIR_MODES[int(rng.integers(0, len(PAIR_MODES) - 1))]
    return modes


def _sample_dims_for_mode(rng: np.random.Generator, mode: str) -> tuple[tuple[int, int], tuple[int, int]]:
    heights = list(range(FRAME_H_MIN, FRAME_H_MAX + 1))
    widths = list(range(FRAME_W_MIN, FRAME_W_MAX + 1))
    while True:
        h1 = int(rng.choice(heights))
        h2 = int(rng.choice(heights))
        w1 = int(rng.choice(widths))
        w2 = int(rng.choice(widths))
        if mode == "both" and h1 == h2 and w1 == w2:
            return (h1, w1), (h2, w2)
        if mode == "width_only" and h1 != h2 and w1 == w2:
            return (h1, w1), (h2, w2)
        if mode == "height_only" and h1 == h2 and w1 != w2:
            return (h1, w1), (h2, w2)
        if mode == "none" and h1 != h2 and w1 != w2:
            return (h1, w1), (h2, w2)


def _sample_unpaired_dims(rng: np.random.Generator) -> tuple[int, int]:
    h = int(rng.integers(FRAME_H_MIN, FRAME_H_MAX + 1))
    w = int(rng.integers(FRAME_W_MIN, FRAME_W_MAX + 1))
    return h, w


def _can_place_box(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int) -> bool:
    rr0 = max(0, int(r0) - int(padding))
    cc0 = max(0, int(c0) - int(padding))
    rr1 = min(int(occupied.shape[0]), int(r0) + int(h) + int(padding))
    cc1 = min(int(occupied.shape[1]), int(c0) + int(w) + int(padding))
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _all_positions(occupied: np.ndarray, h: int, w: int, padding: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    H, W = map(int, occupied.shape)
    for r0 in range(H - int(h) + 1):
        for c0 in range(W - int(w) + 1):
            if _can_place_box(occupied, int(r0), int(c0), int(h), int(w), int(padding)):
                positions.append((int(r0), int(c0)))
    return positions


def _reserve_box(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[int(r0) : int(r0) + int(h), int(c0) : int(c0) + int(w)] = True


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[int(r0), int(c0) : int(c0) + int(w)] = int(color)
    grid[int(r0) + int(h) - 1, int(c0) : int(c0) + int(w)] = int(color)
    grid[int(r0) : int(r0) + int(h), int(c0)] = int(color)
    grid[int(r0) : int(r0) + int(h), int(c0) + int(w) - 1] = int(color)


def _build_indicator_patch(
    rng: np.random.Generator, pairs: list[tuple[int, int]]
) -> np.ndarray:
    use_gap = bool(rng.integers(0, 2))
    height = int(len(pairs)) + 2
    width = 5 if use_gap else 4
    patch = np.full((height, width), GREY, dtype=int)
    for i, (left_color, right_color) in enumerate(pairs):
        rr = 1 + int(i)
        patch[rr, 1] = int(left_color)
        if use_gap:
            patch[rr, 2] = BG
            patch[rr, 3] = int(right_color)
        else:
            patch[rr, 2] = int(right_color)
    return patch


def _corner_position(grid_shape: tuple[int, int], patch_shape: tuple[int, int], corner: str) -> tuple[int, int]:
    H, W = map(int, grid_shape)
    h, w = map(int, patch_shape)
    if corner == "top_left":
        return 0, 0
    if corner == "top_right":
        return 0, W - w
    if corner == "bottom_left":
        return H - h, 0
    if corner == "bottom_right":
        return H - h, W - w
    raise AssertionError("unknown corner")


def _expected_output(grid: np.ndarray, frame_specs: list[dict], pairs: list[tuple[int, int]]) -> np.ndarray:
    frames = {
        int(spec["color"]): {
            "color": int(spec["color"]),
            "bbox": tuple(int(v) for v in spec["bbox"]),
            "h": int(spec["h"]),
            "w": int(spec["w"]),
        }
        for spec in frame_specs
    }
    edge_colors = _edge_plan(frames, pairs)
    out = np.array(grid, copy=True)
    for color, frame in frames.items():
        _paint_frame(out, frame, edge_colors[int(color)])
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    pair_count = int(rng.choice(PAIR_COUNT_CHOICES))
    extra_cap = min(EXTRA_FRAME_MAX, len(ALL_FRAME_COLORS) - 2 * pair_count)
    extra_count = int(rng.integers(0, extra_cap + 1))
    frame_count = 2 * pair_count + extra_count

    min_side = 16 + 2 * frame_count
    if min_side > GRID_SIDE_MAX:
        return None
    H = int(rng.integers(min_side, GRID_SIDE_MAX + 1))
    W = int(rng.integers(min_side, GRID_SIDE_MAX + 1))

    colors = [int(c) for c in rng.choice(np.array(ALL_FRAME_COLORS, dtype=int), size=frame_count, replace=False)]
    modes = _sample_pair_modes(rng, pair_count)

    pair_rows: list[tuple[int, int]] = []
    frame_specs: list[dict] = []
    color_idx = 0
    for mode in modes:
        (h1, w1), (h2, w2) = _sample_dims_for_mode(rng, mode)
        color_a = colors[color_idx]
        color_b = colors[color_idx + 1]
        color_idx += 2
        if bool(rng.integers(0, 2)):
            pair_rows.append((int(color_a), int(color_b)))
        else:
            pair_rows.append((int(color_b), int(color_a)))
        frame_specs.append({"color": int(color_a), "h": int(h1), "w": int(w1)})
        frame_specs.append({"color": int(color_b), "h": int(h2), "w": int(w2)})
    while color_idx < frame_count:
        h, w = _sample_unpaired_dims(rng)
        frame_specs.append({"color": int(colors[color_idx]), "h": int(h), "w": int(w)})
        color_idx += 1

    rng.shuffle(pair_rows)
    indicator = _build_indicator_patch(rng, pair_rows)
    corner = str(rng.choice(np.array(["top_left", "top_right", "bottom_left", "bottom_right"], dtype=object)))
    ind_r0, ind_c0 = _corner_position((H, W), indicator.shape, corner)

    occupied = np.zeros((H, W), dtype=bool)
    if not _can_place_box(occupied, ind_r0, ind_c0, indicator.shape[0], indicator.shape[1], 0):
        return None
    _reserve_box(occupied, ind_r0, ind_c0, indicator.shape[0], indicator.shape[1])

    order = sorted(range(len(frame_specs)), key=lambda i: int(frame_specs[i]["h"]) * int(frame_specs[i]["w"]), reverse=True)
    for idx in order:
        spec = frame_specs[idx]
        positions = _all_positions(occupied, int(spec["h"]), int(spec["w"]), PLACEMENT_PADDING)
        if not positions:
            return None
        r0, c0 = positions[int(rng.integers(0, len(positions)))]
        spec["bbox"] = (int(r0), int(c0), int(r0) + int(spec["h"]) - 1, int(c0) + int(spec["w"]) - 1)
        _reserve_box(occupied, int(r0), int(c0), int(spec["h"]), int(spec["w"]))

    grid = np.full((H, W), BG, dtype=int)
    grid[ind_r0 : ind_r0 + indicator.shape[0], ind_c0 : ind_c0 + indicator.shape[1]] = indicator
    for spec in frame_specs:
        r0, c0, _r1, _c1 = map(int, spec["bbox"])
        _draw_frame(grid, int(r0), int(c0), int(spec["h"]), int(spec["w"]), int(spec["color"]))

    expected = _expected_output(grid, frame_specs, pair_rows)
    if np.array_equal(grid, expected):
        return None
    if len(_find_frames(grid)) != len(frame_specs):
        return None
    if _read_indicator_pairs(grid) != pair_rows:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
