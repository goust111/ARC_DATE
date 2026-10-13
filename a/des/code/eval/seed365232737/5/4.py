from __future__ import annotations

from collections import Counter
import random

import numpy as np


GRID_W_MAX = 34
GRID_H_MAX = 30
GEN_RETRIES = 400


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        box = _bbox(grid == int(color))
        assert box is not None
        r0, c0, r1, c1 = box
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert bool(np.all((patch == int(bg)) | (patch == int(color))))
        hole_count = int(np.sum(patch == int(bg)))
        objects.append(
            {
                "color": int(color),
                "bbox": box,
                "height": int(r1 - r0 + 1),
                "width": int(c1 - c0 + 1),
                "patch": patch.copy(),
                "holes": hole_count,
            }
        )
    return objects


def _classify_objects(objects: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    stamps: list[dict[str, object]] = []
    canvases: list[dict[str, object]] = []
    for obj in objects:
        h = int(obj["height"])
        w = int(obj["width"])
        holes = int(obj["holes"])
        if 3 <= h <= 5 and 3 <= w <= 5 and 1 <= holes <= 3:
            stamps.append(obj)
        elif holes == 0:
            canvases.append(obj)
        else:
            raise AssertionError("unexpected object shape")
    assert len(stamps) == len(canvases) >= 2
    return stamps, canvases


def _pair_objects(
    stamps: list[dict[str, object]],
    canvases: list[dict[str, object]],
) -> list[tuple[dict[str, object], dict[str, object]]]:
    unused = set(range(len(canvases)))
    pairs: list[tuple[dict[str, object], dict[str, object]]] = []

    for stamp in sorted(stamps, key=lambda obj: (obj["bbox"][0], obj["bbox"][1])):  # type: ignore[index]
        sr0, sc0, sr1, sc1 = stamp["bbox"]  # type: ignore[misc]
        candidates: list[tuple[int, int, int, int]] = []
        for idx in sorted(unused):
            canvas = canvases[idx]
            cr0, cc0, _cr1, cc1 = canvas["bbox"]  # type: ignore[misc]
            if int(cr0) <= int(sr1):
                continue
            contains = int(cc0) <= int(sc0) and int(sc1) <= int(cc1)
            if not contains:
                continue
            if int(canvas["width"]) % int(stamp["width"]) != 0:
                continue
            slack = (int(sc0) - int(cc0)) + (int(cc1) - int(sc1))
            candidates.append((int(cr0) - int(sr1), slack, int(cc0), idx))
        assert candidates, "each stamp must have one canvas directly below it"
        candidates.sort()
        chosen_idx = candidates[0][-1]
        unused.remove(chosen_idx)
        pairs.append((stamp, canvases[chosen_idx]))

    assert not unused
    return pairs


def _tiled_holes(stamp_patch: np.ndarray, bg: int, out_h: int, out_w: int) -> np.ndarray:
    hole_mask = stamp_patch == int(bg)
    rr = np.arange(int(out_h))[:, None] % hole_mask.shape[0]
    cc = np.arange(int(out_w))[None, :] % hole_mask.shape[1]
    return hole_mask[rr, cc]


def _render_from_pairs(grid: np.ndarray, bg: int, pairs: list[tuple[dict[str, object], dict[str, object]]]) -> np.ndarray:
    plans: list[tuple[dict[str, object], np.ndarray, np.ndarray]] = []
    out_h = int(grid.shape[0])
    out_w = int(grid.shape[1])

    for stamp, canvas in pairs:
        cr0, cc0, cr1, cc1 = canvas["bbox"]  # type: ignore[misc]
        canvas_h = int(cr1 - cr0 + 1)
        canvas_w = int(cc1 - cc0 + 1)
        holes = _tiled_holes(np.asarray(stamp["patch"], dtype=int), bg, canvas_h, canvas_w)
        drop_counts = holes.sum(axis=0).astype(int)
        out_h = max(out_h, int(cr1) + 1 + int(drop_counts.max(initial=0)))
        plans.append((canvas, holes, drop_counts))

    out = np.full((out_h, out_w), int(bg), dtype=int)
    out[: grid.shape[0], : grid.shape[1]] = grid

    for canvas, holes, drop_counts in plans:
        cr0, cc0, cr1, cc1 = canvas["bbox"]  # type: ignore[misc]
        color = int(canvas["color"])
        view = out[int(cr0) : int(cr1) + 1, int(cc0) : int(cc1) + 1]
        view[holes] = int(bg)
        for dc, count in enumerate(drop_counts):
            if int(count) == 0:
                continue
            out[int(cr1) + 1 : int(cr1) + 1 + int(count), int(cc0) + int(dc)] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify several holed "stamps" above solid "canvases".
    Tile each stamp over its paired canvas, punch out the stamp's background cells,
    and stack the removed canvas pixels below the canvas in the same columns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _mode_color(g)
    objects = _extract_objects(g, bg)
    stamps, canvases = _classify_objects(objects)
    pairs = _pair_objects(stamps, canvases)
    return _render_from_pairs(g, bg, pairs)


def _sample_holes(height: int, width: int, hole_count: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(int(height)) for c in range(int(width))]
    for _ in range(120):
        holes = random.sample(cells, int(hole_count))
        rows = Counter(r for r, _c in holes)
        cols = Counter(c for _r, c in holes)
        if any(int(rows[r]) == int(width) for r in range(int(height))):
            continue
        if any(int(cols[c]) == int(height) for c in range(int(width))):
            continue
        return sorted((int(r), int(c)) for r, c in holes)
    raise RuntimeError("failed to sample valid holes")


def _sample_canvas_height(stamp_h: int) -> int:
    if random.random() < 0.5:
        return int(stamp_h * random.choice([1, 2]))
    candidates = [
        h
        for h in range(max(4, stamp_h - 1), stamp_h * 2 + 3)
        if h % int(stamp_h) != 0
    ]
    return int(random.choice(candidates or [stamp_h + 1]))


def _render_input(bg: int, height: int, width: int, specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((int(height), int(width)), int(bg), dtype=int)
    for spec in specs:
        sr0, sc0, sr1, sc1 = spec["stamp_bbox"]  # type: ignore[misc]
        stamp = np.full((int(sr1) - int(sr0) + 1, int(sc1) - int(sc0) + 1), int(spec["stamp_color"]), dtype=int)
        for hr, hc in spec["holes"]:  # type: ignore[assignment]
            stamp[int(hr), int(hc)] = int(bg)
        grid[int(sr0) : int(sr1) + 1, int(sc0) : int(sc1) + 1] = stamp

        cr0, cc0, cr1, cc1 = spec["canvas_bbox"]  # type: ignore[misc]
        grid[int(cr0) : int(cr1) + 1, int(cc0) : int(cc1) + 1] = int(spec["canvas_color"])
    return grid


def _render_expected(bg: int, grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    for spec in specs:
        sh = int(spec["stamp_h"])
        sw = int(spec["stamp_w"])
        stamp_patch = np.full((sh, sw), int(spec["stamp_color"]), dtype=int)
        for hr, hc in spec["holes"]:  # type: ignore[assignment]
            stamp_patch[int(hr), int(hc)] = int(bg)
        pairs.append(
            (
                {"patch": stamp_patch, "bbox": spec["stamp_bbox"], "width": sw},
                {
                    "bbox": spec["canvas_bbox"],
                    "width": int(spec["canvas_w"]),
                    "color": int(spec["canvas_color"]),
                },
            )
        )
    return _render_from_pairs(grid, bg, pairs)


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    count = random.choice([2, 2, 3])
    bg = random.randint(0, 9)
    palette = [color for color in range(10) if color != int(bg)]
    random.shuffle(palette)
    stamp_colors = palette[:count]
    canvas_colors = palette[count : 2 * count]

    specs: list[dict[str, object]] = []
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    gaps = [random.randint(2, 4) for _ in range(count - 1)]

    canvas_widths: list[int] = []
    canvas_heights: list[int] = []
    stamp_tops: list[int] = []

    for i in range(count):
        stamp_h = random.randint(3, 5)
        stamp_w_hi = 5 if count == 2 or random.random() < 0.35 else 4
        stamp_w = random.randint(3, stamp_w_hi)
        hole_count = random.randint(1, 3)
        holes = _sample_holes(stamp_h, stamp_w, hole_count)

        multipliers = [2, 3, 4] if count == 2 else [2, 3]
        canvas_w = int(stamp_w * random.choice(multipliers))
        canvas_h = _sample_canvas_height(stamp_h)
        x_offset = random.randint(0, canvas_w - stamp_w)

        specs.append(
            {
                "stamp_color": int(stamp_colors[i]),
                "canvas_color": int(canvas_colors[i]),
                "stamp_h": int(stamp_h),
                "stamp_w": int(stamp_w),
                "canvas_h": int(canvas_h),
                "canvas_w": int(canvas_w),
                "x_offset": int(x_offset),
                "holes": holes,
            }
        )
        canvas_widths.append(canvas_w)
        canvas_heights.append(canvas_h)
        stamp_tops.append(random.randint(1, 3))

    width = int(left_margin + sum(canvas_widths) + sum(gaps) + right_margin)
    if width > GRID_W_MAX:
        return None

    x_positions: list[int] = []
    cursor = int(left_margin)
    for i, canvas_w in enumerate(canvas_widths):
        x_positions.append(cursor)
        cursor += int(canvas_w)
        if i + 1 < count:
            cursor += int(gaps[i])

    stamp_bottom = max(
        int(stamp_tops[i]) + int(specs[i]["stamp_h"]) - 1 for i in range(count)
    )
    bottom_margin = random.randint(0, 3)
    max_canvas_h = max(canvas_heights)
    canvas_top = max(
        stamp_bottom + random.randint(2, 5),
        max_canvas_h + bottom_margin + random.randint(1, 3),
    )
    height = int(canvas_top + max_canvas_h + bottom_margin)

    non_bg = sum(
        int(spec["stamp_h"]) * int(spec["stamp_w"]) - len(spec["holes"]) + int(spec["canvas_h"]) * int(spec["canvas_w"])
        for spec in specs
    )
    max_single = max(
        max(int(spec["stamp_h"]) * int(spec["stamp_w"]) - len(spec["holes"]) for spec in specs),
        max(int(spec["canvas_h"]) * int(spec["canvas_w"]) for spec in specs),
    )
    bg_count = int(width * height - non_bg)
    if bg_count <= max_single:
        extra_rows = (int(max_single) + 1 - int(bg_count) + int(width) - 1) // int(width)
        canvas_top += extra_rows
        height += extra_rows

    if height > GRID_H_MAX:
        return None

    for i, spec in enumerate(specs):
        canvas_x = int(x_positions[i])
        canvas_y = int(canvas_top)
        stamp_x = canvas_x + int(spec["x_offset"])
        stamp_y = int(stamp_tops[i])

        spec["stamp_bbox"] = (
            stamp_y,
            stamp_x,
            stamp_y + int(spec["stamp_h"]) - 1,
            stamp_x + int(spec["stamp_w"]) - 1,
        )
        spec["canvas_bbox"] = (
            canvas_y,
            canvas_x,
            canvas_y + int(spec["canvas_h"]) - 1,
            canvas_x + int(spec["canvas_w"]) - 1,
        )

    grid = _render_input(bg, height, width, specs)
    expected = _render_expected(bg, grid, specs)

    if not np.array_equal(main(grid), expected):
        return None

    bg_counts = Counter(map(int, grid.reshape(-1)))
    if bg_counts[int(bg)] <= max(count for color, count in bg_counts.items() if color != int(bg)):
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_case()
        if built is None:
            continue
        grid, _expected = built
        return grid
    raise RuntimeError("failed to generate valid input")
