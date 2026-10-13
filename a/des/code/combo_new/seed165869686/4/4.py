from __future__ import annotations

import numpy as np


BG = 0
PALETTE = np.arange(1, 10, dtype=int)
MAX_GENERATION_ATTEMPTS = 400
INNER_HEIGHT_RANGE = (2, 8)
INNER_WIDTH_RANGE = (2, 8)
RECT_COUNT_RANGE = (2, 5)
ITEM_GAP_RANGE = (2, 5)
ROW_GAP_RANGE = (2, 5)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_full_rectangle(cells: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _read_frame(
    grid: np.ndarray, inner_bbox: tuple[int, int, int, int]
) -> dict[str, int] | None:
    h, w = grid.shape
    r0, c0, r1, c1 = inner_bbox
    if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
        return None

    top = grid[r0 - 1, c0 - 1 : c1 + 2]
    bottom = grid[r1 + 1, c0 - 1 : c1 + 2]
    left = grid[r0 : r1 + 1, c0 - 1]
    right = grid[r0 : r1 + 1, c1 + 1]

    if (
        np.any(top == BG)
        or np.any(bottom == BG)
        or np.any(left == BG)
        or np.any(right == BG)
    ):
        return None
    if not (
        np.all(top == top[0])
        and np.all(bottom == bottom[0])
        and np.all(left == left[0])
        and np.all(right == right[0])
    ):
        return None

    top_color = int(top[0])
    bottom_color = int(bottom[0])
    left_color = int(left[0])
    right_color = int(right[0])

    horizontal_match = top_color == bottom_color
    vertical_match = left_color == right_color
    if horizontal_match == vertical_match:
        return None

    return {
        "inner_r0": int(r0),
        "inner_c0": int(c0),
        "inner_r1": int(r1),
        "inner_c1": int(c1),
        "top": top_color,
        "bottom": bottom_color,
        "left": left_color,
        "right": right_color,
    }


def _find_frames(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict[str, int]] = []
    for comp in _components(g == BG):
        if any(r == 0 or c == 0 or r == g.shape[0] - 1 or c == g.shape[1] - 1 for r, c in comp):
            continue
        box = _bbox(comp)
        if not _is_full_rectangle(comp, box):
            continue
        frame = _read_frame(g, box)
        if frame is not None:
            frames.append(frame)
    frames.sort(key=lambda frame: (frame["inner_r0"], frame["inner_c0"]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow multicolor rectangle with stripes encoded by its matching opposite edges.

    Input structure:
    - The background is black.
    - The grid contains one or more 1-pixel-thick rectangular frames whose interiors are black.
    - Each frame side is a solid color; the four sides may differ, and the corner pixels belong to the horizontal sides.

    Rule:
    - Detect each enclosed black rectangular cavity and read the colors of the four surrounding sides.
    - If the top and bottom sides match, fill the interior with horizontal stripes alternating the shared top/bottom color and the left-side color, starting with the shared color on the first interior row.
    - Otherwise the left and right sides match, so fill the interior with vertical stripes alternating the shared left/right color and the top-side color, starting with the shared color on the first interior column.
    - Leave every border pixel unchanged.

    Key insight:
    - The stripe direction is determined by which pair of opposite sides shares a color, and the secondary stripe color comes from a specific side of the non-matching pair.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for frame in _find_frames(g):
        r0 = frame["inner_r0"]
        c0 = frame["inner_c0"]
        r1 = frame["inner_r1"]
        c1 = frame["inner_c1"]

        if frame["top"] == frame["bottom"]:
            main_color = frame["top"]
            alt_color = frame["left"]
            for r in range(r0, r1 + 1):
                fill_color = main_color if (r - r0) % 2 == 0 else alt_color
                out[r, c0 : c1 + 1] = fill_color
        else:
            main_color = frame["left"]
            alt_color = frame["top"]
            for c in range(c0, c1 + 1):
                fill_color = main_color if (c - c0) % 2 == 0 else alt_color
                out[r0 : r1 + 1, c] = fill_color

    return out


def _positive_composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(int(x) for x in rng.choice(np.arange(1, total), size=parts - 1, replace=False))
    out: list[int] = []
    start = 0
    for cut in cuts + [total]:
        out.append(int(cut - start))
        start = int(cut)
    return out


def _balanced_orientations(count: int, rng: np.random.Generator) -> list[str]:
    horizontal_count = count // 2
    if count % 2 == 1 and bool(rng.integers(0, 2)):
        horizontal_count += 1
    orientations = ["horizontal"] * horizontal_count + ["vertical"] * (count - horizontal_count)
    rng.shuffle(orientations)
    return orientations


def _sample_spec(orientation: str, rng: np.random.Generator) -> dict[str, int | str]:
    inner_h = int(rng.integers(INNER_HEIGHT_RANGE[0], INNER_HEIGHT_RANGE[1] + 1))
    inner_w = int(rng.integers(INNER_WIDTH_RANGE[0], INNER_WIDTH_RANGE[1] + 1))
    colors = [int(x) for x in rng.choice(PALETTE, size=3, replace=False)]
    main_color, aux_a, aux_b = colors

    if orientation == "horizontal":
        top = bottom = main_color
        left = aux_a
        right = aux_b
    else:
        left = right = main_color
        top = aux_a
        bottom = aux_b

    return {
        "orientation": orientation,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": inner_h + 2,
        "outer_w": inner_w + 2,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
    }


def _layout_specs(
    specs: list[dict[str, int | str]], rng: np.random.Generator
) -> tuple[int, int, list[tuple[dict[str, int | str], int, int]]] | None:
    count = len(specs)
    for _ in range(120):
        order = list(specs)
        rng.shuffle(order)
        row_count = int(rng.integers(1, min(3, count) + 1))
        row_sizes = _positive_composition(count, row_count, rng)

        groups: list[list[dict[str, int | str]]] = []
        idx = 0
        for size in row_sizes:
            groups.append(order[idx : idx + size])
            idx += size

        row_data: list[dict[str, object]] = []
        total_w = 0
        total_h = 0
        between_rows: list[int] = []

        for row_idx, group in enumerate(groups):
            row_h = max(int(spec["outer_h"]) for spec in group)
            item_gaps = [
                int(rng.integers(ITEM_GAP_RANGE[0], ITEM_GAP_RANGE[1]))
                for _ in range(max(0, len(group) - 1))
            ]
            row_w = sum(int(spec["outer_w"]) for spec in group) + sum(item_gaps)
            row_data.append(
                {
                    "specs": group,
                    "row_h": row_h,
                    "row_w": row_w,
                    "item_gaps": item_gaps,
                }
            )
            total_w = max(total_w, row_w)
            total_h += row_h
            if row_idx + 1 < len(groups):
                gap = int(rng.integers(ROW_GAP_RANGE[0], ROW_GAP_RANGE[1]))
                between_rows.append(gap)
                total_h += gap

        if total_h + 2 > 30 or total_w + 2 > 30:
            continue

        extra_h = int(rng.integers(2, 31 - total_h))
        extra_w = int(rng.integers(2, 31 - total_w))
        top_margin = int(rng.integers(1, extra_h))
        left_margin = int(rng.integers(1, extra_w))

        placements: list[tuple[dict[str, int | str], int, int]] = []
        y = top_margin
        for row_idx, row in enumerate(row_data):
            row_h = int(row["row_h"])
            row_w = int(row["row_w"])
            row_specs = list(row["specs"])
            item_gaps = list(row["item_gaps"])
            row_left = left_margin + int(rng.integers(0, total_w - row_w + 1))
            x = row_left
            for spec_idx, spec in enumerate(row_specs):
                outer_h = int(spec["outer_h"])
                top = y + int(rng.integers(0, row_h - outer_h + 1))
                placements.append((spec, int(top), int(x)))
                x += int(spec["outer_w"])
                if spec_idx < len(item_gaps):
                    x += int(item_gaps[spec_idx])
            y += row_h
            if row_idx < len(between_rows):
                y += between_rows[row_idx]

        final_h = total_h + extra_h
        final_w = total_w + extra_w
        if final_h <= 30 and final_w <= 30:
            return final_h, final_w, placements
    return None


def _draw_frame(grid: np.ndarray, spec: dict[str, int | str], r0: int, c0: int) -> None:
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    top = int(spec["top"])
    bottom = int(spec["bottom"])
    left = int(spec["left"])
    right = int(spec["right"])

    grid[r0, c0 : c0 + outer_w] = top
    grid[r0 + outer_h - 1, c0 : c0 + outer_w] = bottom
    grid[r0 + 1 : r0 + outer_h - 1, c0] = left
    grid[r0 + 1 : r0 + outer_h - 1, c0 + outer_w - 1] = right


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    rect_count = int(rng.integers(RECT_COUNT_RANGE[0], RECT_COUNT_RANGE[1] + 1))
    specs = [_sample_spec(orientation, rng) for orientation in _balanced_orientations(rect_count, rng)]
    layout = _layout_specs(specs, rng)
    if layout is None:
        return None

    height, width, placements = layout
    grid = np.full((height, width), BG, dtype=int)
    for spec, r0, c0 in placements:
        _draw_frame(grid, spec, r0, c0)

    if len(_find_frames(grid)) != rect_count:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
