from __future__ import annotations

import numpy as np
from scipy.ndimage import label


BG = 0
BAR_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    lab, n = label(grid == int(color), structure=STRUCT4)
    comps = []
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == k)
        if coords.size == 0:
            continue
        comps.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": _bbox(coords),
                "area": int(coords.shape[0]),
            }
        )
    return comps


def _is_solid_rectangle(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    color = int(comp["color"])
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    return bool(np.all(patch == color))


def _validate_panel(panel: np.ndarray, bg: int, bar_color: int) -> list[int] | None:
    if panel.size == 0:
        return None
    values = set(map(int, np.unique(panel)))
    if values - {bg, bar_color}:
        return None
    comps = _components_of_color(panel, bar_color)
    if len(comps) < 2:
        return None
    bottom = panel.shape[0] - 1
    heights = []
    for comp in comps:
        if not _is_solid_rectangle(panel, comp):
            return None
        r0, _c0, r1, _c1 = comp["bbox"]
        if r1 != bottom:
            return None
        heights.append(bottom - r0 + 1)
    if len(set(heights)) != len(heights):
        return None
    return sorted(heights)


def _validate_main(main: np.ndarray, bg: int, bar_color: int, expected_colors: int) -> dict[int, list[dict]] | None:
    values = sorted(set(map(int, np.unique(main))) - {bg, bar_color})
    if len(values) != expected_colors or not values:
        return None
    comps_by_color: dict[int, list[dict]] = {}
    counts = []
    for color in values:
        comps = _components_of_color(main, color)
        if not comps:
            return None
        if any(not _is_solid_rectangle(main, comp) for comp in comps):
            return None
        comps_by_color[color] = comps
        counts.append(len(comps))
    if len(set(counts)) != len(counts):
        return None
    return comps_by_color


def _candidate_layouts(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    H, W = g.shape
    layouts = []

    for c in range(W):
        if not np.all(g[:, c] == g[0, c]):
            continue
        divider_color = int(g[0, c])
        if divider_color == bg:
            continue
        left = g[:, :c]
        right = g[:, c + 1 :]
        for panel_side, panel, main in (("left", left, right), ("right", right, left)):
            bar_heights = _validate_panel(panel, bg, divider_color)
            if bar_heights is None:
                continue
            comps_by_color = _validate_main(main, bg, divider_color, len(bar_heights))
            if comps_by_color is None:
                continue
            layouts.append(
                {
                    "orientation": "vertical",
                    "panel_side": panel_side,
                    "divider_index": c,
                    "bg": bg,
                    "bar_color": divider_color,
                    "bar_heights": bar_heights,
                    "main": main.copy(),
                    "components_by_color": comps_by_color,
                }
            )

    for r in range(H):
        if not np.all(g[r, :] == g[r, 0]):
            continue
        divider_color = int(g[r, 0])
        if divider_color == bg:
            continue
        top = g[:r, :]
        bottom = g[r + 1 :, :]
        for panel_side, panel, main in (("top", top, bottom), ("bottom", bottom, top)):
            bar_heights = _validate_panel(panel, bg, divider_color)
            if bar_heights is None:
                continue
            comps_by_color = _validate_main(main, bg, divider_color, len(bar_heights))
            if comps_by_color is None:
                continue
            layouts.append(
                {
                    "orientation": "horizontal",
                    "panel_side": panel_side,
                    "divider_index": r,
                    "bg": bg,
                    "bar_color": divider_color,
                    "bar_heights": bar_heights,
                    "main": main.copy(),
                    "components_by_color": comps_by_color,
                }
            )

    return layouts


def _analyze_layout(grid: np.ndarray) -> dict:
    layouts = _candidate_layouts(grid)
    assert len(layouts) == 1, "expected exactly one valid divider/panel layout"
    return layouts[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resize each colored rectangle family to the histogram-command height indicated by its block-count rank.

    Input structure:
    - A full divider line splits the grid into a main field and a side panel.
    - The side panel contains gray solid bars with distinct heights; every bar is bottom-aligned and disconnected from the others.
    - The main field contains solid colored rectangles; each color appears in a unique number of disconnected blocks.

    Rule:
    - Count how many rectangles belong to each color in the main field.
    - Sort colors by increasing block count and sort the gray panel bars by increasing height; match ranks one-to-one.
    - Remove the panel, then redraw every rectangle of each color with unchanged width and bottom edge but with the matched target height.

    Key insight:
    - The gray histogram is not a summary of the main field; it is an instruction panel whose height ranking controls every color via count-rank correspondence.
    """
    info = _analyze_layout(np.asarray(grid, dtype=int))
    main_grid = info["main"]
    bg = int(info["bg"])
    bar_heights = list(info["bar_heights"])
    comps_by_color = info["components_by_color"]

    colors_by_count = sorted(comps_by_color, key=lambda color: len(comps_by_color[color]))
    target_height_by_color = {
        color: int(bar_heights[idx]) for idx, color in enumerate(colors_by_count)
    }

    out = np.full(main_grid.shape, bg, dtype=int)
    for color, comps in comps_by_color.items():
        target_h = target_height_by_color[color]
        for comp in comps:
            _r0, c0, r1, c1 = comp["bbox"]
            new_r0 = r1 - target_h + 1
            assert new_r0 >= 0, "target height must fit above the rectangle bottom"
            assert np.all(out[new_r0 : r1 + 1, c0 : c1 + 1] == bg), "resized rectangles must stay disjoint"
            out[new_r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def _sample_counts(rng: np.random.Generator, n_colors: int) -> list[int]:
    options = np.arange(1, 7, dtype=int)
    for _ in range(100):
        counts = sorted(map(int, rng.choice(options, size=n_colors, replace=False)))
        if sum(counts) <= 11:
            return counts
    raise RuntimeError("failed to sample rectangle counts")


def _sample_heights(rng: np.random.Generator, n_colors: int, H: int) -> list[int]:
    options = np.arange(2, min(H - 1, 10), dtype=int)
    if options.size < n_colors:
        raise RuntimeError("not enough distinct height options")
    return sorted(map(int, rng.choice(options, size=n_colors, replace=False)))


def _choose_initial_height(rng: np.random.Generator, target_h: int, H: int) -> tuple[int, str]:
    smaller = list(range(1, target_h))
    larger = list(range(target_h + 1, min(H - 1, target_h + 4) + 1))
    modes = ["same"]
    probs = [0.3]
    if smaller:
        modes.append("grow")
        probs.append(0.35)
    if larger:
        modes.append("shrink")
        probs.append(0.35)
    probs = np.array(probs, dtype=float)
    probs /= probs.sum()
    mode = str(rng.choice(modes, p=probs))
    if mode == "grow":
        return int(rng.choice(smaller)), mode
    if mode == "shrink":
        return int(rng.choice(larger)), mode
    return int(target_h), mode


def _build_main_field(
    rng: np.random.Generator,
    H: int,
    W: int,
    color_specs: dict[int, dict[str, int]],
) -> np.ndarray | None:
    grid = np.full((H, W), BG, dtype=int)
    reserved = np.zeros((H, W), dtype=bool)
    items = []
    seen_grow = False
    seen_shrink = False

    for color, spec in color_specs.items():
        for _ in range(spec["count"]):
            width = int(rng.integers(1, min(4, W - 1) + 1))
            init_h, mode = _choose_initial_height(rng, spec["target_h"], H)
            reserve_h = max(init_h, spec["target_h"])
            items.append(
                {
                    "color": int(color),
                    "width": width,
                    "init_h": init_h,
                    "target_h": int(spec["target_h"]),
                    "reserve_h": reserve_h,
                    "priority": reserve_h * width,
                    "mode": mode,
                }
            )
            seen_grow |= mode == "grow"
            seen_shrink |= mode == "shrink"

    if items and not seen_grow:
        for item in items:
            if item["target_h"] > 1:
                item["init_h"] = item["target_h"] - 1
                item["reserve_h"] = item["target_h"]
                item["priority"] = item["reserve_h"] * item["width"]
                item["mode"] = "grow"
                seen_grow = True
                break
    if items and not seen_shrink:
        for item in items:
            if item["target_h"] + 1 < H:
                item["init_h"] = item["target_h"] + 1
                item["reserve_h"] = item["init_h"]
                item["priority"] = item["reserve_h"] * item["width"]
                item["mode"] = "shrink"
                seen_shrink = True
                break

    items.sort(key=lambda item: (-item["priority"], -item["reserve_h"], -item["width"]))
    for item in items:
        width = item["width"]
        reserve_h = item["reserve_h"]
        if reserve_h >= H or width >= W:
            return None
        placed = False
        for _ in range(250):
            bottom = int(rng.integers(reserve_h - 1, H))
            top = bottom - reserve_h + 1
            left = int(rng.integers(0, W - width + 1))
            rr0 = max(0, top - 1)
            rr1 = min(H, bottom + 2)
            cc0 = max(0, left - 1)
            cc1 = min(W, left + width + 1)
            if reserved[rr0:rr1, cc0:cc1].any():
                continue
            reserved[rr0:rr1, cc0:cc1] = True
            init_h = item["init_h"]
            color = item["color"]
            grid[bottom - init_h + 1 : bottom + 1, left : left + width] = color
            placed = True
            break
        if not placed:
            return None
    return grid


def _build_panel(
    H: int,
    heights: list[int],
    widths: list[int],
) -> np.ndarray:
    panel_w = 2 + sum(widths) + (len(widths) - 1)
    panel = np.full((H, panel_w), BG, dtype=int)
    col = 1
    for height, width in zip(heights, widths):
        panel[H - height : H, col : col + width] = BAR_COLOR
        col += width + 1
    return panel


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_colors = int(rng.integers(2, 5))
    H = int(rng.integers(max(12, n_colors + 8), 23))

    try:
        counts = _sample_counts(rng, n_colors)
        heights = _sample_heights(rng, n_colors, H)
    except RuntimeError:
        return None

    widths = [int(rng.integers(1, 3)) for _ in range(n_colors)]
    panel = _build_panel(H, heights, widths)
    panel_w = panel.shape[1]
    max_main_w = 30 - panel_w - 1
    if max_main_w < 10:
        return None
    main_w = int(rng.integers(10, min(21, max_main_w) + 1))

    colors = list(map(int, rng.choice(PALETTE, size=n_colors, replace=False)))
    color_specs = {
        color: {"count": count, "target_h": height}
        for color, count, height in zip(colors, counts, heights)
    }
    main_field = _build_main_field(rng, H, main_w, color_specs)
    if main_field is None:
        return None

    divider = np.full((H, 1), BAR_COLOR, dtype=int)
    if bool(rng.integers(0, 2)):
        grid = np.concatenate([panel, divider, main_field], axis=1)
    else:
        grid = np.concatenate([main_field, divider, panel], axis=1)

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != main_field.shape:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
