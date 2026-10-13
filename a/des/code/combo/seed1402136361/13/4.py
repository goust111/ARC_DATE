from __future__ import annotations

import random

import numpy as np


MAX_GEN_ATTEMPTS = 400
LAYER_COUNT_RANGE = (2, 5)
INNER_HEIGHT_RANGE = (3, 10)
INNER_WIDTH_RANGE = (3, 10)
EXPANSION_RANGE = (1, 3)
MARGIN_RANGE = (1, 5)
MAX_OUTER_SIZE = 22
MIN_CANVAS_SIZE = 10
MAX_CANVAS_SIZE = 30


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _shape_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(cnts.argmax())])


def _frame_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_frame(grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]) -> bool:
    height, width = _shape_from_bbox(bbox)
    if height < 3 or width < 3:
        return False
    r0, c0, r1, c1 = bbox
    submask = grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)
    return np.array_equal(submask, _frame_mask(height, width))


def _is_solid_rect(grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)))


def _strictly_contains(
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _draw_frame(canvas: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    canvas[top, left : left + width] = int(color)
    canvas[top + height - 1, left : left + width] = int(color)
    canvas[top : top + height, left] = int(color)
    canvas[top : top + height, left + width - 1] = int(color)


def _render_objects(bg: int, objects: list[dict[str, int | str]]) -> np.ndarray:
    heights = [int(obj["height"]) for obj in objects]
    widths = [int(obj["width"]) for obj in objects]
    out_h = max(heights)
    out_w = sum(widths) + max(0, len(objects) - 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    cursor = 0
    for obj in objects:
        height = int(obj["height"])
        width = int(obj["width"])
        row = (out_h - height) // 2
        color = int(obj["color"])
        if obj["kind"] == "solid":
            out[row : row + height, cursor : cursor + width] = color
        else:
            _draw_frame(out, row, cursor, height, width, color)
        cursor += width + 1

    return out


def _detect_objects(grid: np.ndarray) -> tuple[int, list[dict[str, int | str]]]:
    bg = _most_common_color(grid)
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    assert colors, "expected at least one foreground color"

    frames: list[dict[str, int | str]] = []
    solids: list[dict[str, int | str]] = []
    for color in colors:
        coords = np.argwhere(grid == int(color))
        bbox = _bbox(coords)
        height, width = _shape_from_bbox(bbox)
        if _is_frame(grid, color, bbox):
            frames.append(
                {
                    "kind": "frame",
                    "color": int(color),
                    "bbox": bbox,
                    "height": height,
                    "width": width,
                }
            )
        elif _is_solid_rect(grid, color, bbox):
            solids.append(
                {
                    "kind": "solid",
                    "color": int(color),
                    "bbox": bbox,
                    "height": height,
                    "width": width,
                }
            )
        else:
            raise AssertionError(f"color {color} is neither a frame nor a solid fill")

    assert len(frames) >= 2, "expected multiple nested frames"
    frames.sort(key=lambda obj: (int(obj["height"]) * int(obj["width"]), int(obj["height"]), int(obj["width"])))

    centers = {(bbox[0] + bbox[2], bbox[1] + bbox[3]) for bbox in (obj["bbox"] for obj in frames)}
    assert len(centers) == 1, "frames must be concentric"
    for inner, outer in zip(frames, frames[1:], strict=False):
        assert _strictly_contains(outer["bbox"], inner["bbox"]), "frame bboxes must be strictly nested"

    objects: list[dict[str, int | str]] = []
    if solids:
        assert len(solids) == 1, "expected at most one solid fill"
        solid = solids[0]
        assert _strictly_contains(frames[0]["bbox"], solid["bbox"]), "fill must sit inside the innermost frame"
        objects.append(solid)

    objects.extend(frames)
    return bg, objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split nested rectangular frames apart and lay them out from inner to outer.

    Input structure:
    - Background is the dominant color.
    - A single concentric stack of one-pixel-thick rectangular frames sits in the grid.
    - Each frame has a distinct color; the innermost frame may contain a solid fill.

    Rule:
    - Identify all frame layers and sort them from innermost to outermost.
    - Draw them as independent objects from left to right with one background column
      between neighbors.
    - If the innermost frame contains a solid rectangle, draw that solid first.

    Key insight:
    - The input is organized by geometric containment, while the output order is the
      reverse nesting order expressed as a horizontal sequence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, objects = _detect_objects(g)
    return _render_objects(bg, objects)


def _sample_nested_spec() -> dict[str, object] | None:
    layer_count = random.randint(*LAYER_COUNT_RANGE)
    expansions = [random.randint(*EXPANSION_RANGE) for _ in range(layer_count - 1)]
    total_expansion = sum(expansions)

    max_inner_h = min(INNER_HEIGHT_RANGE[1], MAX_OUTER_SIZE - 2 * total_expansion)
    max_inner_w = min(INNER_WIDTH_RANGE[1], MAX_OUTER_SIZE - 2 * total_expansion)
    if max_inner_h < INNER_HEIGHT_RANGE[0] or max_inner_w < INNER_WIDTH_RANGE[0]:
        return None

    inner_h = random.randint(INNER_HEIGHT_RANGE[0], max_inner_h)
    inner_w = random.randint(INNER_WIDTH_RANGE[0], max_inner_w)
    outer_h = inner_h + 2 * total_expansion
    outer_w = inner_w + 2 * total_expansion

    top = random.randint(*MARGIN_RANGE)
    bottom_lo = max(MARGIN_RANGE[0], top - 2)
    bottom_hi = min(MARGIN_RANGE[1], top + 2)
    bottom = random.randint(bottom_lo, bottom_hi)
    left = random.randint(*MARGIN_RANGE)
    right_lo = max(MARGIN_RANGE[0], left - 2)
    right_hi = min(MARGIN_RANGE[1], left + 2)
    right = random.randint(right_lo, right_hi)

    height = outer_h + top + bottom
    width = outer_w + left + right
    if not (MIN_CANVAS_SIZE <= height <= MAX_CANVAS_SIZE and MIN_CANVAS_SIZE <= width <= MAX_CANVAS_SIZE):
        return None

    fill_present = bool(random.randint(0, 1))
    needed_colors = 1 + layer_count + (1 if fill_present else 0)
    if needed_colors > 10:
        return None

    palette = list(range(10))
    bg = random.choice(palette)
    fg_palette = [c for c in palette if c != bg]
    random.shuffle(fg_palette)
    frame_colors = fg_palette[:layer_count]
    fill_color = fg_palette[layer_count] if fill_present else None

    return {
        "bg": bg,
        "frame_colors": frame_colors,
        "fill_color": fill_color,
        "fill_present": fill_present,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "expansions": expansions,
        "top": top,
        "left": left,
        "height": height,
        "width": width,
    }


def _frame_sizes(inner_h: int, inner_w: int, expansions: list[int]) -> list[tuple[int, int]]:
    sizes = [(int(inner_h), int(inner_w))]
    h = int(inner_h)
    w = int(inner_w)
    for step in expansions:
        h += 2 * int(step)
        w += 2 * int(step)
        sizes.append((h, w))
    return sizes


def _build_input(spec: dict[str, object]) -> np.ndarray:
    bg = int(spec["bg"])
    height = int(spec["height"])
    width = int(spec["width"])
    top = int(spec["top"])
    left = int(spec["left"])
    inner_h = int(spec["inner_h"])
    inner_w = int(spec["inner_w"])
    expansions = [int(x) for x in spec["expansions"]]
    frame_colors = [int(x) for x in spec["frame_colors"]]

    grid = np.full((height, width), bg, dtype=int)
    sizes = _frame_sizes(inner_h, inner_w, expansions)

    for i, (frame_h, frame_w) in enumerate(sizes):
        inset = sum(expansions[i:])
        color = frame_colors[i]
        _draw_frame(grid, top + inset, left + inset, frame_h, frame_w, color)

    if bool(spec["fill_present"]):
        fill_color = int(spec["fill_color"])
        inner_top = top + sum(expansions)
        inner_left = left + sum(expansions)
        grid[inner_top + 1 : inner_top + inner_h - 1, inner_left + 1 : inner_left + inner_w - 1] = fill_color

    return grid


def _build_expected(spec: dict[str, object]) -> np.ndarray:
    bg = int(spec["bg"])
    inner_h = int(spec["inner_h"])
    inner_w = int(spec["inner_w"])
    expansions = [int(x) for x in spec["expansions"]]
    frame_colors = [int(x) for x in spec["frame_colors"]]

    objects: list[dict[str, int | str]] = []
    if bool(spec["fill_present"]):
        objects.append(
            {
                "kind": "solid",
                "color": int(spec["fill_color"]),
                "height": inner_h - 2,
                "width": inner_w - 2,
            }
        )

    for color, (frame_h, frame_w) in zip(frame_colors, _frame_sizes(inner_h, inner_w, expansions), strict=False):
        objects.append({"kind": "frame", "color": int(color), "height": frame_h, "width": frame_w})

    return _render_objects(bg, objects)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        spec = _sample_nested_spec()
        if spec is None:
            continue

        inp = _build_input(spec)
        out = _build_expected(spec)

        bg = int(spec["bg"])
        counts = {int(color): int(np.count_nonzero(inp == int(color))) for color in np.unique(inp)}
        if counts[bg] <= max(count for color, count in counts.items() if color != bg):
            continue

        if not np.array_equal(main(inp), out):
            continue
        return inp

    raise RuntimeError("failed to generate a valid nested-frame input")
