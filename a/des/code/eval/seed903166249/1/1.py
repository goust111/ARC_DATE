from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

_RNG = random.Random()

FRAME_COUNT_OPTIONS = (2, 3, 4)
FRAME_H_RANGE = (5, 7)
FRAME_W_RANGE = (5, 7)
TOTAL_SEEDS_MIN = 3
TOTAL_SEEDS_MAX = 5
ROW_SHIFT_RANGE = (0, 3)
EDGE_SLACK_RANGE = (1, 3)
GENERATION_RETRIES = 400


def _randint(lo: int, hi: int) -> int:
    return _RNG.randint(int(lo), int(hi))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((int(height), int(width)), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _paint_border(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    h, w = out.shape
    if r1 < 0 or c1 < 0 or r0 >= h or c0 >= w:
        return

    top = max(int(r0), 0)
    left = max(int(c0), 0)
    bottom = min(int(r1), int(h - 1))
    right = min(int(c1), int(w - 1))
    if top > bottom or left > right:
        return

    if 0 <= r0 < h:
        out[int(r0), left : right + 1] = int(color)
    if r1 != r0 and 0 <= r1 < h:
        out[int(r1), left : right + 1] = int(color)
    if 0 <= c0 < w:
        out[top : bottom + 1, int(c0)] = int(color)
    if c1 != c0 and 0 <= c1 < w:
        out[top : bottom + 1, int(c1)] = int(color)


def _extract_frames(grid: np.ndarray) -> list[dict]:
    grey_comps = _components(grid == GREY)
    frames: list[dict] = []
    for comp in grey_comps:
        r0, c0 = map(int, comp.min(axis=0))
        r1, c1 = map(int, comp.max(axis=0))
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        assert height >= 3 and width >= 3, "frames must have an interior"

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = _border_mask(height, width)
        assert int(comp.shape[0]) == int(border.sum()), "grey object must be a one-cell rectangular border"
        assert bool(np.all(sub[border] == GREY)), "every border cell must be grey"
        assert not bool(np.any(sub[~border] == GREY)), "grey must not appear inside a frame"

        interior = sub[1:-1, 1:-1]
        seeds = [int(v) for v in interior.reshape(-1) if int(v) not in (BLACK, GREY)]
        seed_counts = Counter(seeds)
        assert len(seed_counts) == 2, "each frame interior must contain exactly two seed colors"
        ranked = sorted(seed_counts.items(), key=lambda kv: -int(kv[1]))
        assert int(ranked[0][1]) != int(ranked[1][1]), "seed counts must determine a unique color order"

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "colors": (int(ranked[0][0]), int(ranked[1][0])),
                "layers": int(sum(seed_counts.values())),
            }
        )

    assert frames, "expected at least one grey frame"
    return frames


def _apply_rule(grid: np.ndarray, frames: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True, dtype=int)
    ordered = sorted(frames, key=lambda item: (int(item["bbox"][1]), int(item["bbox"][0])))

    for frame in ordered:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        first, second = map(int, frame["colors"])
        for layer in range(1, int(frame["layers"]) + 1):
            color = first if layer % 2 == 1 else second
            _paint_border(out, r0 - layer, c0 - layer, r1 + layer, c1 + layer, color)

    for frame in frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        out[r0 : r1 + 1, c0 : c1 + 1] = grid[r0 : r1 + 1, c0 : c1 + 1]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow alternating concentric outer borders from each grey frame.

    For each frame, the more frequent seed color becomes the first outer ring color,
    the second color alternates with it, and the total number of outer layers equals
    the total number of seed pixels in that frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    frames = _extract_frames(g)
    return _apply_rule(g, frames)


def _sample_row_partition(frame_count: int) -> list[int]:
    if frame_count == 2:
        return [2]
    if frame_count == 3:
        return [3] if _RNG.random() < 0.4 else [2, 1]
    assert frame_count == 4
    options = ([2, 2], [3, 1], [1, 3])
    return list(options[_randint(0, len(options) - 1)])


def _sample_seed_map(frame_h: int, frame_w: int, major_color: int, minor_color: int, total: int) -> dict[tuple[int, int], int]:
    inner_cells = [(r, c) for r in range(1, frame_h - 1) for c in range(1, frame_w - 1)]
    chosen = _RNG.sample(inner_cells, int(total))
    major_count = int(total - _randint(1, int((total - 1) // 2)))
    if _RNG.random() < 0.5:
        major_color, minor_color = minor_color, major_color

    major_cells = set(_RNG.sample(chosen, major_count))
    return {
        (r, c): (major_color if (r, c) in major_cells else minor_color)
        for r, c in chosen
    }


def _sample_specs(frame_count: int, frame_h: int, frame_w: int) -> list[dict]:
    palette = list(_RNG.sample(SEED_COLORS, 2 * int(frame_count)))
    _RNG.shuffle(palette)

    inner_area = int((frame_h - 2) * (frame_w - 2))
    total_max = min(TOTAL_SEEDS_MAX, inner_area - 1)
    assert total_max >= TOTAL_SEEDS_MIN

    specs: list[dict] = []
    for i in range(frame_count):
        c_a = int(palette[2 * i])
        c_b = int(palette[2 * i + 1])
        total = _randint(TOTAL_SEEDS_MIN, total_max)
        seed_map = _sample_seed_map(frame_h, frame_w, c_a, c_b, total)
        counts = Counter(seed_map.values())
        ranked = sorted(counts.items(), key=lambda kv: -int(kv[1]))
        specs.append(
            {
                "seed_map": seed_map,
                "colors": (int(ranked[0][0]), int(ranked[1][0])),
                "layers": int(total),
            }
        )

    _RNG.shuffle(specs)
    return specs


def _place_specs(specs: list[dict], frame_h: int, frame_w: int, overlap_mode: bool) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    frame_count = len(specs)
    row_counts = _sample_row_partition(frame_count)
    rows: list[list[int]] = []
    cursor = 0
    for count in row_counts:
        rows.append(list(range(cursor, cursor + int(count))))
        cursor += int(count)

    overlap_row = -1
    overlap_slot = -1
    if overlap_mode:
        candidate_rows = [i for i, row in enumerate(rows) if len(row) >= 2]
        overlap_row = candidate_rows[_randint(0, len(candidate_rows) - 1)]
        overlap_slot = _randint(0, len(rows[overlap_row]) - 2)

    row_layouts: list[dict] = []
    for row_index, row in enumerate(rows):
        local_cols: list[int] = []
        col = 0
        for j, idx in enumerate(row):
            local_cols.append(int(col))
            if j == len(row) - 1:
                continue
            next_idx = row[j + 1]
            layers_a = int(specs[idx]["layers"])
            layers_b = int(specs[next_idx]["layers"])
            if row_index == overlap_row and j == overlap_slot:
                gap = _randint(max(layers_a, layers_b), layers_a + layers_b - 1)
            else:
                gap = layers_a + layers_b + _randint(0, 2)
            col += int(frame_w + gap)
        row_width = int(local_cols[-1] + frame_w)
        row_layouts.append(
            {
                "indices": row,
                "local_cols": local_cols,
                "width": row_width,
                "max_layers": max(int(specs[idx]["layers"]) for idx in row),
                "shift": _randint(*ROW_SHIFT_RANGE),
            }
        )

    top_margin = int(row_layouts[0]["max_layers"] + _randint(*EDGE_SLACK_RANGE))
    row_tops: list[int] = []
    top = top_margin
    for i, row_info in enumerate(row_layouts):
        row_tops.append(int(top))
        if i == len(row_layouts) - 1:
            continue
        gap = int(row_info["max_layers"] + row_layouts[i + 1]["max_layers"] + _randint(1, 3))
        top += int(frame_h + gap)
    bottom_margin = int(row_layouts[-1]["max_layers"] + _randint(*EDGE_SLACK_RANGE))

    max_layers = max(int(spec["layers"]) for spec in specs)
    left_margin = int(max_layers + _randint(*EDGE_SLACK_RANGE))
    right_margin = int(max_layers + _randint(*EDGE_SLACK_RANGE))

    positions = [(-1, -1)] * frame_count
    width = 0
    for row_info, row_top in zip(row_layouts, row_tops):
        for idx, local_col in zip(row_info["indices"], row_info["local_cols"]):
            col = int(left_margin + row_info["shift"] + local_col)
            positions[idx] = (int(row_top), col)
            width = max(width, int(col + frame_w + right_margin))
    height = int(row_tops[-1] + frame_h + bottom_margin)
    return positions, (height, width)


def _render_input(specs: list[dict], positions: list[tuple[int, int]], shape: tuple[int, int], frame_h: int, frame_w: int) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for spec, (r0, c0) in zip(specs, positions):
        r1 = int(r0 + frame_h - 1)
        c1 = int(c0 + frame_w - 1)
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY
        for (rr, cc), color in spec["seed_map"].items():
            grid[int(r0 + rr), int(c0 + cc)] = int(color)
    return grid


def _expected_frames(specs: list[dict], positions: list[tuple[int, int]], frame_h: int, frame_w: int) -> list[dict]:
    frames: list[dict] = []
    for spec, (r0, c0) in zip(specs, positions):
        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r0 + frame_h - 1), int(c0 + frame_w - 1)),
                "colors": tuple(map(int, spec["colors"])),
                "layers": int(spec["layers"]),
            }
        )
    return frames


def _generate_one() -> np.ndarray | None:
    frame_count = FRAME_COUNT_OPTIONS[_randint(0, len(FRAME_COUNT_OPTIONS) - 1)]
    frame_h = _randint(*FRAME_H_RANGE)
    frame_w = _randint(*FRAME_W_RANGE)
    overlap_mode = bool(_RNG.random() < 0.5)

    specs = _sample_specs(frame_count, frame_h, frame_w)
    positions, shape = _place_specs(specs, frame_h, frame_w, overlap_mode)
    inp = _render_input(specs, positions, shape, frame_h, frame_w)
    frames = _expected_frames(specs, positions, frame_h, frame_w)
    expected = _apply_rule(inp, frames)

    try:
        extracted = _extract_frames(inp)
        want = sorted(
            [(tuple(frame["bbox"]), tuple(frame["colors"]), int(frame["layers"])) for frame in frames],
            key=lambda item: (item[0][0], item[0][1]),
        )
        got = sorted(
            [(tuple(frame["bbox"]), tuple(frame["colors"]), int(frame["layers"])) for frame in extracted],
            key=lambda item: (item[0][0], item[0][1]),
        )
        if got != want:
            return None
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-frame input")
