from __future__ import annotations

import random

import numpy as np


BG = 0
SIDE_NAMES = ("top", "right", "bottom", "left")

LAYER_COUNT_CHOICES = (2, 3)
LEGEND_FACTOR_CHOICES = (2, 3, 4)
TEMPLATE_SIDE_RANGE = (5, 11)
OUTER_OUTPUT_SIDE_RANGE = (9, 24)
TOP_PADDING_RANGE = (0, 3)
TEMPLATE_GAP_RANGE = (1, 3)
CANVAS_MARGIN_RANGE = (2, 6)
CANVAS_EXTRA_RANGE = (0, 6)
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in pts]
    cols = [c for _r, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _border_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _window_side_and_offset(row: int, col: int, size: int) -> tuple[str, int] | None:
    if row == 0 and 0 < col < size - 1:
        return "top", int(col)
    if col == size - 1 and 0 < row < size - 1:
        return "right", int(row)
    if row == size - 1 and 0 < col < size - 1:
        return "bottom", int(col)
    if col == 0 and 0 < row < size - 1:
        return "left", int(row)
    return None


def _parse_template(grid: np.ndarray, bg: int, pts: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(pts)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h != w or h < 3:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = _border_mask(h)
    if np.any(sub[~border] != bg):
        return None
    if np.any(sub[border] == bg):
        return None
    if int(np.sum(sub != bg)) != int(np.sum(border)):
        return None

    border_vals = sub[border]
    colors = [int(v) for v in np.unique(border_vals)]
    if len(colors) != 2:
        return None

    counts = {color: int(np.sum(border_vals == color)) for color in colors}
    border_color = max(counts, key=counts.get)
    window_color = next(color for color in colors if color != border_color)
    if counts[window_color] != 1:
        return None

    wr, wc = map(int, np.argwhere(sub == window_color)[0])
    side_offset = _window_side_and_offset(wr, wc, h)
    if side_offset is None:
        return None
    side, offset = side_offset

    return {
        "bbox": (r0, c0, r1, c1),
        "side": h,
        "border_color": int(border_color),
        "window_color": int(window_color),
        "window_side": side,
        "window_offset": int(offset),
    }


def _parse_legend_row(grid: np.ndarray, bg: int, pts: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(pts)
    if r0 != r1:
        return None
    width = int(c1 - c0 + 1)
    if width < 3:
        return None
    row = grid[r0, c0 : c1 + 1]
    if np.any(row == bg):
        return None
    if len(np.unique(row[1:])) != 1:
        return None
    if int(row[0]) == int(row[1]):
        return None
    return {
        "row": int(r0),
        "gray_color": int(row[1]),
        "key_color": int(row[0]),
        "factor": int(width - 1),
        "bbox": (r0, c0, r1, c1),
    }


def _scene_parts(grid: np.ndarray) -> tuple[int, list[dict], dict[int, int], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    template_infos: list[dict] = []
    legend_candidates: list[dict] = []
    for pts in _components(g != bg):
        template = _parse_template(g, bg, pts)
        if template is not None:
            template_infos.append(template)
            continue
        legend = _parse_legend_row(g, bg, pts)
        if legend is not None:
            legend_candidates.append(legend)

    assert template_infos, "expected at least one template frame"
    assert legend_candidates, "expected a legend in the lower-right area"

    by_gray: dict[int, list[dict]] = {}
    for row in legend_candidates:
        by_gray.setdefault(int(row["gray_color"]), []).append(row)
    legend_rows = max(
        by_gray.values(),
        key=lambda rows: (len(rows), min(int(row["row"]) for row in rows)),
    )
    ratio_map = {int(row["key_color"]): int(row["factor"]) for row in legend_rows}
    legend_top = min(int(row["row"]) for row in legend_rows)

    templates = sorted(template_infos, key=lambda info: (int(info["bbox"][0]), int(info["bbox"][1])))
    for template in templates[:-1]:
        assert int(template["window_color"]) in ratio_map, "every non-final window color must appear in the legend"

    canvas_left = max(int(info["bbox"][3]) for info in templates) + 1
    canvas_h = int(legend_top)
    canvas_w = int(g.shape[1] - canvas_left)
    assert canvas_h > 0 and canvas_w > 0, "canvas area must be a non-empty right-side rectangle"
    return bg, templates, ratio_map, (canvas_h, canvas_w)


def _scaled_window_offset(src_size: int, dst_size: int, offset: int) -> int:
    if dst_size <= 3 or src_size <= 3:
        return 1
    mapped = 1 + int(round((offset - 1) * (dst_size - 3) / (src_size - 3)))
    return max(1, min(dst_size - 2, mapped))


def _window_cell(side: str, size: int, offset: int) -> tuple[int, int]:
    if side == "top":
        return 0, offset
    if side == "right":
        return offset, size - 1
    if side == "bottom":
        return size - 1, offset
    if side == "left":
        return offset, 0
    raise ValueError(side)


def _make_template_patch(size: int, border_color: int, window_color: int, side: str, offset: int) -> np.ndarray:
    patch = np.full((size, size), BG, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color
    wr, wc = _window_cell(side, size, offset)
    patch[wr, wc] = int(window_color)
    return patch


def _draw_scaled_template(
    out: np.ndarray,
    top: int,
    left: int,
    dst_size: int,
    template: dict,
) -> None:
    patch = _make_template_patch(
        dst_size,
        int(template["border_color"]),
        int(template["window_color"]),
        str(template["window_side"]),
        _scaled_window_offset(int(template["side"]), dst_size, int(template["window_offset"])),
    )
    out[top : top + dst_size, left : left + dst_size] = patch


def _render_nested_frames(
    templates: list[dict],
    ratio_map: dict[int, int],
    canvas_shape: tuple[int, int],
    bg: int,
) -> np.ndarray:
    canvas_h, canvas_w = canvas_shape
    out = np.full((canvas_h, canvas_w), int(bg), dtype=int)

    prev_top = prev_left = prev_size = None
    for idx, template in enumerate(templates):
        if idx == 0:
            size = int(template["side"])
            top = (canvas_h - size) // 2
            left = (canvas_w - size) // 2
        else:
            assert prev_top is not None and prev_left is not None and prev_size is not None
            factor = int(ratio_map[int(templates[idx - 1]["window_color"])])
            inner = int(prev_size - 2)
            size = max(3, inner // factor)
            assert size <= inner, "next layer must fit inside the previous frame interior"
            top = int(prev_top + 1 + (inner - size) // 2)
            left = int(prev_left + 1 + (inner - size) // 2)

        assert 0 <= top and 0 <= left
        assert top + size <= canvas_h and left + size <= canvas_w
        _draw_scaled_template(out, top, left, size, template)
        prev_top, prev_left, prev_size = top, left, size

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw nested scaled frames on the right-side blank canvas.

    The left-side template order determines the nesting order.  Each template
    carries a one-pixel window whose color indexes the bottom-right legend;
    the legend's gray-run length gives the divisor used to scale the next
    template before centering it inside the current frame.
    """
    g = np.asarray(grid, dtype=int)
    bg, templates, ratio_map, canvas_shape = _scene_parts(g)
    return _render_nested_frames(templates, ratio_map, canvas_shape, bg)


def _sample_outer_size(window_factors: list[int]) -> int | None:
    for _ in range(80):
        size0 = random.randint(*OUTER_OUTPUT_SIDE_RANGE)
        current = size0
        ok = True
        for idx, factor in enumerate(window_factors):
            next_size = max(3, (current - 2) // factor)
            if idx < len(window_factors) - 1 and next_size < 5:
                ok = False
                break
            current = int(next_size)
        if ok:
            return int(size0)
    return None


def _sample_templates(
    num_layers: int,
    legend_colors: list[int],
    border_colors: list[int],
) -> tuple[list[dict], dict[int, int]] | None:
    window_colors = [random.choice(legend_colors) for _ in range(num_layers)]
    ratio_map = {int(color): random.choice(LEGEND_FACTOR_CHOICES) for color in legend_colors}
    outer_size = _sample_outer_size([ratio_map[int(color)] for color in window_colors[:-1]])
    if outer_size is None:
        return None

    templates: list[dict] = []
    for idx in range(num_layers):
        side = outer_size if idx == 0 else random.randint(*TEMPLATE_SIDE_RANGE)
        offset = random.randint(1, side - 2)
        templates.append(
            {
                "side": int(side),
                "border_color": int(border_colors[idx]),
                "window_color": int(window_colors[idx]),
                "window_side": random.choice(SIDE_NAMES),
                "window_offset": int(offset),
            }
        )
    return templates, ratio_map


def _build_input_and_expected() -> tuple[np.ndarray, np.ndarray] | None:
    num_layers = random.choice(LAYER_COUNT_CHOICES)
    colors = list(range(1, 10))
    random.shuffle(colors)

    gray_color = colors.pop()
    max_legend_entries = min(3, len(colors) - num_layers)
    if max_legend_entries < 1:
        return None
    legend_count = random.randint(1, max_legend_entries)
    legend_colors = colors[:legend_count]
    border_colors = colors[legend_count : legend_count + num_layers]

    sampled = _sample_templates(num_layers, legend_colors, border_colors)
    if sampled is None:
        return None
    templates, ratio_map = sampled
    legend_rows = list(ratio_map.items())
    random.shuffle(legend_rows)

    template_patches = [
        _make_template_patch(
            int(template["side"]),
            int(template["border_color"]),
            int(template["window_color"]),
            str(template["window_side"]),
            int(template["window_offset"]),
        )
        for template in templates
    ]
    left_width = max(int(patch.shape[1]) for patch in template_patches)

    top_padding = random.randint(*TOP_PADDING_RANGE)
    gaps = [random.randint(*TEMPLATE_GAP_RANGE) for _ in range(num_layers - 1)]
    template_column_height = top_padding
    for idx, patch in enumerate(template_patches):
        template_column_height += int(patch.shape[0])
        if idx < len(gaps):
            template_column_height += int(gaps[idx])

    margin_y = random.randint(*CANVAS_MARGIN_RANGE)
    margin_x = random.randint(*CANVAS_MARGIN_RANGE)
    outer_size = int(templates[0]["side"])
    canvas_h = max(template_column_height + random.randint(*CANVAS_EXTRA_RANGE), outer_size + 2 * margin_y)
    canvas_w = outer_size + 2 * margin_x + random.randint(*CANVAS_EXTRA_RANGE)

    legend_height = 2 * len(legend_rows) - 1
    grid = np.full((canvas_h + legend_height, left_width + canvas_w), BG, dtype=int)

    row = top_padding
    for idx, patch in enumerate(template_patches):
        ph, pw = patch.shape
        grid[row : row + ph, 0:pw] = patch
        row += ph
        if idx < len(gaps):
            row += int(gaps[idx])

    for idx, (color, factor) in enumerate(legend_rows):
        rr = canvas_h + 2 * idx
        cc = grid.shape[1] - (factor + 1)
        grid[rr, cc] = int(color)
        grid[rr, cc + 1 : cc + 1 + factor] = int(gray_color)

    expected = _render_nested_frames(templates, ratio_map, (canvas_h, canvas_w), BG)
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_input_and_expected()
        if built is None:
            continue
        grid, _expected = built
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
