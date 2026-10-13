from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
BRACKET_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
OPENINGS = ("left", "right", "up", "down")
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
GEN_RETRIES = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _components_mask(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    dirs = NEIGHBORS4 if int(connectivity) == 4 else NEIGHBORS8
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    height, width = mask.shape
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_components(
    grid: np.ndarray,
    *,
    colors: Iterable[int] | None = None,
    connectivity: int = 4,
) -> list[dict]:
    if colors is None:
        colors = sorted(set(map(int, grid.flat)) - {int(_get_bg(grid))})
    out: list[dict] = []
    for color in colors:
        mask = grid == int(color)
        for coords in _components_mask(mask, connectivity):
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _mask_from_component(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in component["coords"]:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _flip_mask(mask: np.ndarray, axis: str) -> np.ndarray:
    if axis == "horizontal":
        return np.fliplr(mask)
    if axis == "vertical":
        return np.flipud(mask)
    raise ValueError(f"unknown flip axis: {axis}")


def _bracket_mask(height: int, width: int, opening: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if opening in ("left", "right"):
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, -1 if opening == "left" else 0] = True
    elif opening in ("up", "down"):
        mask[:, 0] = True
        mask[:, -1] = True
        mask[-1 if opening == "up" else 0, :] = True
    else:
        raise ValueError(f"unknown opening: {opening}")
    return mask


def _opening_axis(opening: str) -> str:
    return "horizontal" if opening in ("left", "right") else "vertical"


def _bracket_opening(component: dict) -> str | None:
    mask = _mask_from_component(component)
    if mask.shape[0] < 4 or mask.shape[1] < 4:
        return None
    for opening in OPENINGS:
        if np.array_equal(mask, _bracket_mask(*mask.shape, opening)):
            return opening
    return None


def _inside_marker(grid: np.ndarray, component: dict) -> tuple[tuple[int, int], int]:
    r0, c0, r1, c1 = component["bbox"]
    bracket_rel = _mask_from_component(component)
    bg = _get_bg(grid)
    markers: list[tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if bracket_rel[r - r0, c - c0]:
                continue
            if int(grid[r, c]) != bg:
                markers.append((r, c))
    assert len(markers) == 1, "each bracket must contain exactly one marker"
    marker = markers[0]
    return marker, int(grid[marker])


def _sprite_copies_for_color(
    grid: np.ndarray,
    color: int,
    marker_positions: set[tuple[int, int]],
) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    copies: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for component in _find_components(grid, colors=[color], connectivity=8):
        coords = {tuple(map(int, p)) for p in component["coords"]}
        if coords <= marker_positions:
            continue
        copies.append((component["bbox"], _mask_from_component(component)))
    assert copies, f"expected sprite copies for color {color}"
    return copies


def _template_from_copies(copies: list[tuple[tuple[int, int, int, int], np.ndarray]]) -> np.ndarray:
    template = copies[0][1]
    for _bbox, mask in copies:
        assert mask.shape == template.shape, "same-color copies must share a bbox size"
        assert np.array_equal(mask, template), "same-color copies must share one template"
    return template


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the two marker-selected sprite templates, mirroring each pasted replacement by the destination bracket's opening axis.

    Input structure:
    - The background is the most common color, with several colored sprite copies on it.
    - Two grey three-sided brackets each contain exactly one non-background marker pixel.
    - The two marker colors name the only sprite colors that participate, and those two sprite templates share one bbox size.

    Rule:
    - Read the marker colors A and B from the two brackets and keep every non-selected color, both brackets, and both markers unchanged.
    - Clear every sprite bbox of colors A and B simultaneously.
    - Redraw B's template inside every former A bbox and A's template inside every former B bbox.
    - Before each redraw, flip the incoming template according to the opening of the destination bracket: left/right means horizontal flip, up/down means vertical flip.

    Key insight:
    - The brackets do two jobs at once: their markers select the two swap participants, and their openings decide which mirror is applied to the template pasted into that participant's old boxes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    components = _find_components(g, colors=[BRACKET_COLOR], connectivity=4)
    brackets = []
    for component in components:
        opening = _bracket_opening(component)
        if opening is None:
            continue
        marker_pos, marker_color = _inside_marker(g, component)
        brackets.append(
            {
                "component": component,
                "opening": opening,
                "axis": _opening_axis(opening),
                "marker_pos": marker_pos,
                "marker_color": marker_color,
            }
        )
    assert len(brackets) == 2, "expected exactly two grey brackets"

    selected_colors = [int(item["marker_color"]) for item in brackets]
    assert selected_colors[0] != selected_colors[1], "marker colors must differ"
    marker_positions = {tuple(item["marker_pos"]) for item in brackets}
    axis_for_destination = {int(item["marker_color"]): item["axis"] for item in brackets}

    copies = {color: _sprite_copies_for_color(g, color, marker_positions) for color in selected_colors}
    templates = {color: _template_from_copies(copies[color]) for color in selected_colors}

    color_a, color_b = selected_colors
    assert templates[color_a].shape == templates[color_b].shape, "selected templates must share one bbox size"

    other = {color_a: color_b, color_b: color_a}
    out = g.copy()

    for color in selected_colors:
        for (r0, c0, r1, c1), _mask in copies[color]:
            out[r0 : r1 + 1, c0 : c1 + 1] = bg

    for destination_color in selected_colors:
        source_color = other[destination_color]
        source_mask = templates[source_color]
        transformed = _flip_mask(source_mask, axis_for_destination[destination_color])
        patch = np.full(transformed.shape, bg, dtype=int)
        patch[transformed] = int(source_color)
        for (r0, c0, _r1, _c1), _mask in copies[destination_color]:
            _paste(out, patch, r0, c0)

    return out


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _path_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    out: list[tuple[int, int]] = []
    while (r, c) != b:
        dr = _sign(b[0] - r)
        dc = _sign(b[1] - c)
        choices = []
        if dr:
            choices.append((dr, 0))
        if dc:
            choices.append((0, dc))
        if dr and dc:
            choices.append((dr, dc))
        step_r, step_c = random.choice(choices)
        r += step_r
        c += step_c
        out.append((r, c))
    return out


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _random_sprite_mask(
    height: int,
    width: int,
    *,
    require_axis_distinction: bool,
) -> np.ndarray:
    total = height * width
    min_area = max(3, (total + 3) // 4)
    max_area = min(total - 1, (3 * total) // 4 + 1)
    if min_area > max_area:
        min_area = max_area
    for _ in range(500):
        anchors = [
            (0, random.randrange(width)),
            (height - 1, random.randrange(width)),
            (random.randrange(height), 0),
            (random.randrange(height), width - 1),
        ]
        random.shuffle(anchors)
        target = random.randint(min_area, max_area)
        cells = {anchors[0]}
        for anchor in anchors[1:]:
            start = random.choice(tuple(cells))
            cells.update(_path_between(start, anchor))
            cells.add(anchor)
        if len(cells) > target:
            continue
        frontier = {
            (r + dr, c + dc)
            for r, c in cells
            for dr, dc in NEIGHBORS8
            if 0 <= r + dr < height and 0 <= c + dc < width and (r + dr, c + dc) not in cells
        }
        while len(cells) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            r, c = cell
            for dr, dc in NEIGHBORS8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not _touches_all_sides(mask):
            continue
        hflip = np.fliplr(mask)
        vflip = np.flipud(mask)
        if np.array_equal(mask, hflip) or np.array_equal(mask, vflip):
            continue
        if require_axis_distinction and np.array_equal(hflip, vflip):
            continue
        return mask
    raise RuntimeError("failed to build a non-degenerate sprite mask")


def _render_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _rect_conflict(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return not (
        ar + ah + gap <= br
        or br + bh + gap <= ar
        or ac + aw + gap <= bc
        or bc + bw + gap <= ac
    )


def _place_layout(height: int, width: int, items: list[dict]) -> bool:
    placed_rects: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(items)), key=lambda idx: items[idx]["height"] * items[idx]["width"], reverse=True)
    for idx in order:
        item = items[idx]
        h = int(item["height"])
        w = int(item["width"])
        r_min = c_min = 1
        r_max = height - h - 1
        c_max = width - w - 1
        if r_min > r_max or c_min > c_max:
            return False
        candidates = [(r, c) for r in range(r_min, r_max + 1) for c in range(c_min, c_max + 1)]
        random.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            rect = (r0, c0, h, w)
            if any(_rect_conflict(rect, prev, gap=1) for prev in placed_rects):
                continue
            item["r0"] = r0
            item["c0"] = c0
            placed_rects.append(rect)
            placed = True
            break
        if not placed:
            return False
    return True


def _marker_cell(height: int, width: int, opening: str) -> tuple[int, int]:
    bracket = _bracket_mask(height, width, opening)
    candidates = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width - 1)
        if not bracket[r, c]
    ]
    assert candidates, "bracket must have an interior marker cell"
    return random.choice(candidates)


def _expected_output(layout: dict) -> np.ndarray:
    out = layout["input"].copy()
    selected_colors = layout["selected_colors"]
    other = {selected_colors[0]: selected_colors[1], selected_colors[1]: selected_colors[0]}

    for item in layout["items"]:
        if item["kind"] == "sprite" and item["color"] in selected_colors:
            r0, c0 = item["r0"], item["c0"]
            h, w = item["height"], item["width"]
            out[r0 : r0 + h, c0 : c0 + w] = BG

    for item in layout["items"]:
        if item["kind"] != "sprite" or item["color"] not in selected_colors:
            continue
        destination_color = int(item["color"])
        source_color = other[destination_color]
        axis = layout["axis_for_destination"][destination_color]
        transformed = _flip_mask(layout["templates"][source_color], axis)
        _paste(out, _render_patch(transformed, source_color), item["r0"], item["c0"])
    return out


def _semantic_checks(layout: dict, output: np.ndarray) -> bool:
    selected = set(layout["selected_colors"])
    expected = layout["expected"]
    if not np.array_equal(output, expected):
        return False
    if np.array_equal(layout["input"], output):
        return False

    changed = np.zeros(layout["input"].shape, dtype=bool)
    for item in layout["items"]:
        if item["kind"] == "sprite" and item["color"] in selected:
            r0, c0 = item["r0"], item["c0"]
            h, w = item["height"], item["width"]
            changed[r0 : r0 + h, c0 : c0 + w] = True
            destination_color = int(item["color"])
            source_color = next(color for color in selected if color != destination_color)
            axis = layout["axis_for_destination"][destination_color]
            transformed = _flip_mask(layout["templates"][source_color], axis)
            source_patch = _render_patch(layout["templates"][source_color], source_color)
            dest_patch = _render_patch(layout["templates"][destination_color], destination_color)
            observed = _crop_bbox(output, (r0, c0, r0 + h - 1, c0 + w - 1))
            if not np.array_equal(observed, _render_patch(transformed, source_color)):
                return False
            if np.array_equal(observed, source_patch):
                return False
            if np.array_equal(observed, dest_patch):
                return False

    if not np.array_equal(layout["input"][~changed], output[~changed]):
        return False
    for item in layout["items"]:
        if item["kind"] == "bracket":
            r0, c0 = item["r0"], item["c0"]
            h, w = item["height"], item["width"]
            inp = _crop_bbox(layout["input"], (r0, c0, r0 + h - 1, c0 + w - 1))
            out = _crop_bbox(output, (r0, c0, r0 + h - 1, c0 + w - 1))
            if not np.array_equal(inp, out):
                return False
    return True


def _build_attempt() -> dict | None:
    height = random.randint(20, 29)
    width = random.randint(20, 29)
    grid = np.full((height, width), BG, dtype=int)

    colors = random.sample(PALETTE, k=random.randint(3, 4))
    selected_colors = colors[:2]
    distractor_colors = colors[2:]

    sprite_shape = random.choice([(3, 4), (4, 4), (4, 5), (5, 5)])
    template_a = _random_sprite_mask(*sprite_shape, require_axis_distinction=True)
    template_b = _random_sprite_mask(*sprite_shape, require_axis_distinction=True)
    if np.array_equal(template_a, template_b):
        return None

    axis_assignment = random.choice([(0, 1), (1, 0)])
    openings_for_color = {
        selected_colors[axis_assignment[0]]: random.choice(["left", "right"]),
        selected_colors[axis_assignment[1]]: random.choice(["up", "down"]),
    }
    axis_for_destination = {
        color: _opening_axis(opening)
        for color, opening in openings_for_color.items()
    }

    if np.array_equal(_flip_mask(template_b, axis_for_destination[selected_colors[0]]), template_a):
        return None
    if np.array_equal(_flip_mask(template_a, axis_for_destination[selected_colors[1]]), template_b):
        return None

    items: list[dict] = []
    templates = {selected_colors[0]: template_a, selected_colors[1]: template_b}

    for color in selected_colors:
        for _ in range(random.randint(2, 3)):
            items.append(
                {
                    "kind": "sprite",
                    "color": int(color),
                    "mask": templates[color],
                    "height": sprite_shape[0],
                    "width": sprite_shape[1],
                }
            )

    for color in distractor_colors:
        shape = random.choice([shape for shape in [(2, 3), (3, 3), (3, 5), (4, 3)] if shape != sprite_shape])
        mask = _random_sprite_mask(*shape, require_axis_distinction=False)
        for _ in range(random.randint(1, 2)):
            items.append(
                {
                    "kind": "sprite",
                    "color": int(color),
                    "mask": mask,
                    "height": shape[0],
                    "width": shape[1],
                }
            )

    for color in selected_colors:
        bracket_h = random.randint(4, 6)
        bracket_w = random.randint(4, 6)
        items.append(
            {
                "kind": "bracket",
                "marker_color": int(color),
                "opening": openings_for_color[color],
                "height": bracket_h,
                "width": bracket_w,
            }
        )

    if not _place_layout(height, width, items):
        return None

    for item in items:
        r0, c0 = item["r0"], item["c0"]
        if item["kind"] == "sprite":
            _paste(grid, _render_patch(item["mask"], item["color"]), r0, c0)
        else:
            mask = _bracket_mask(item["height"], item["width"], item["opening"])
            patch = _render_patch(mask, BRACKET_COLOR)
            mr, mc = _marker_cell(item["height"], item["width"], item["opening"])
            patch[mr, mc] = int(item["marker_color"])
            item["marker_rel"] = (mr, mc)
            _paste(grid, patch, r0, c0)

    layout = {
        "input": grid,
        "items": items,
        "selected_colors": tuple(selected_colors),
        "templates": templates,
        "axis_for_destination": axis_for_destination,
    }
    layout["expected"] = _expected_output(layout)
    return layout


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        layout = _build_attempt()
        if layout is None:
            continue
        output = main(layout["input"])
        if _semantic_checks(layout, output):
            return layout["input"]
    raise RuntimeError("failed to generate a valid input for puzzle 3")
