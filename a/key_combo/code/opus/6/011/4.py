from __future__ import annotations

import random

import numpy as np


PALETTE_COLORS = [1, 2, 3, 4, 7]
PALETTE_ROW_PREFIX = np.array([1, 1, 2, 2, 3, 3, 4, 4, 7, 7], dtype=int)
OBJECT_COLORS = [5, 6, 8, 9]
ORIENTATIONS = ("TL", "TR", "BL", "BR")
INDEX_TO_DIMS = {
    1: [(4, 4)],
    2: [(4, 5), (5, 4)],
    3: [(4, 6), (5, 5), (6, 4)],
    4: [(4, 7), (5, 6), (6, 5), (7, 4)],
    5: [(5, 7), (6, 6), (7, 5)],
}
MIN_OBJECTS = 3
MAX_OBJECTS = 4
MAX_ATTEMPTS = 400


def _components8(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_to_tl(sprite: np.ndarray, flip_rows: bool, flip_cols: bool) -> np.ndarray:
    out = sprite
    if flip_rows:
        out = np.flipud(out)
    if flip_cols:
        out = np.fliplr(out)
    return out


def _restore_from_tl(sprite: np.ndarray, flip_rows: bool, flip_cols: bool) -> np.ndarray:
    out = sprite
    if flip_cols:
        out = np.fliplr(out)
    if flip_rows:
        out = np.flipud(out)
    return out


def _parse_corner_object(sprite: np.ndarray) -> dict:
    colors = sorted(int(v) for v in np.unique(sprite))
    assert len(colors) == 2, "each object must use exactly two colors"
    candidates: list[tuple[int, int, bool, bool, str]] = []
    for border_color in colors:
        fill_color = next(color for color in colors if color != border_color)
        for flip_rows, flip_cols, orientation in (
            (False, False, "TL"),
            (False, True, "TR"),
            (True, False, "BL"),
            (True, True, "BR"),
        ):
            canon = _normalize_to_tl(sprite, flip_rows, flip_cols)
            if (
                np.all(canon[0, :] == border_color)
                and np.all(canon[:, 0] == border_color)
                and np.all(canon[1:, 1:] == fill_color)
            ):
                candidates.append((border_color, fill_color, flip_rows, flip_cols, orientation))

    assert len(candidates) == 1, "object must be one corner-L rectangle in a unique orientation"
    border_color, fill_color, flip_rows, flip_cols, orientation = candidates[0]
    h, w = sprite.shape
    border_pixels = int(np.count_nonzero(sprite == border_color))
    assert border_pixels == h + w - 1
    assert 7 <= border_pixels <= 11, "border-pixel count must map into the 5-entry palette"
    return {
        "border_color": int(border_color),
        "fill_color": int(fill_color),
        "flip_rows": bool(flip_rows),
        "flip_cols": bool(flip_cols),
        "orientation": orientation,
        "height": int(h),
        "width": int(w),
        "border_pixels": border_pixels,
        "palette_index": border_pixels - 6,
        "palette_color": PALETTE_COLORS[border_pixels - 7],
    }


def _canonical_nested_transform(h: int, w: int, border_color: int, fill_color: int, inner_color: int) -> np.ndarray:
    out = np.full((h, w), inner_color, dtype=int)
    out[0, 1:] = fill_color
    out[1:, 0] = fill_color
    out[1, 2:] = border_color
    out[2:, 1] = border_color
    diag = np.arange(min(h - 2, w - 2))
    out[2 + diag, 2 + diag] = fill_color
    return out


def _extract_objects(grid: np.ndarray) -> list[dict]:
    mask = np.zeros_like(grid, dtype=bool)
    mask[1:, :] = grid[1:, :] != 0
    objects: list[dict] = []
    for coords in _components8(mask):
        r0, c0, r1, c1 = _bbox(coords)
        sprite = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert np.all(sprite != 0), "each object must fill its entire bounding rectangle"
        spec = _parse_corner_object(sprite)
        spec["bbox"] = (r0, c0, r1, c1)
        spec["sprite"] = sprite
        objects.append(spec)
    objects.sort(key=lambda item: item["bbox"])
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each corner border, use that count to choose a palette color, and repaint the object's nested inner layer after orientation normalization.

    Input structure:
    - The background is black, and the top row starts with five adjacent palette pairs:
      blue, red, green, yellow, and orange.
    - Below the palette row are 3-4 separated rectangular two-color objects.
    - In each object, one color forms a one-cell-thick L along two adjacent outer
      edges, while the other color fills the rest of the rectangle.

    Rule:
    - Keep the palette row unchanged.
    - For each object, count the original L-border pixels P and compute palette
      index I = P - 6.
    - Flip the object to the top-left orientation, repaint the outer L with the
      fill color, repaint the next inner L with the border color, paint the main
      diagonal of the remaining inner rectangle with the fill color, and paint all
      remaining cells with palette color I.
    - Flip the transformed object back to its original orientation and place it
      back into the grid.

    Key insight:
    - The chosen interior color depends on the pre-transform border size, but that
      border disappears during the nested repaint, so the count must be extracted
      before canonicalization and recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 8 and w >= 10
    assert np.array_equal(g[0, :10], PALETTE_ROW_PREFIX), "top-row palette pairs must match the task palette"
    assert np.all(g[0, 10:] == 0), "only the first ten top-row cells belong to the palette"
    assert set(int(v) for v in np.unique(g[1:, :])).issubset({0, *OBJECT_COLORS}), "lower rows must contain only object colors"

    objects = _extract_objects(g)
    assert MIN_OBJECTS <= len(objects) <= MAX_OBJECTS, "expected 3-4 separated corner objects"

    out = np.zeros_like(g)
    out[0, :] = g[0, :]
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        canon = _canonical_nested_transform(
            obj["height"],
            obj["width"],
            obj["border_color"],
            obj["fill_color"],
            obj["palette_color"],
        )
        restored = _restore_from_tl(canon, obj["flip_rows"], obj["flip_cols"])
        out[r0 : r1 + 1, c0 : c1 + 1] = restored
    return out


def _draw_input_object(h: int, w: int, border_color: int, fill_color: int, orientation: str) -> np.ndarray:
    sprite = np.full((h, w), fill_color, dtype=int)
    if orientation == "TL":
        sprite[0, :] = border_color
        sprite[:, 0] = border_color
    elif orientation == "TR":
        sprite[0, :] = border_color
        sprite[:, -1] = border_color
    elif orientation == "BL":
        sprite[-1, :] = border_color
        sprite[:, 0] = border_color
    else:
        sprite[-1, :] = border_color
        sprite[:, -1] = border_color
    return sprite


def _draw_expected_object(spec: dict) -> np.ndarray:
    canon = _canonical_nested_transform(
        spec["height"],
        spec["width"],
        spec["border_color"],
        spec["fill_color"],
        PALETTE_COLORS[spec["palette_index"] - 1],
    )
    flip_rows = spec["orientation"] in {"BL", "BR"}
    flip_cols = spec["orientation"] in {"TR", "BR"}
    return _restore_from_tl(canon, flip_rows, flip_cols)


def _choose_dims(index: int) -> tuple[int, int]:
    options = INDEX_TO_DIMS[index]
    rectangular = [dims for dims in options if dims[0] != dims[1]]
    if rectangular and random.random() < 0.75:
        return random.choice(rectangular)
    return random.choice(options)


def _sample_object_specs(num_objects: int) -> list[dict] | None:
    indices = random.sample([1, 2, 3, 4, 5], k=num_objects)
    orientations = random.sample(list(ORIENTATIONS), k=num_objects)
    specs = []
    for index, orientation in zip(indices, orientations):
        border_color = random.choice(OBJECT_COLORS)
        fill_choices = [color for color in OBJECT_COLORS if color != border_color]
        fill_color = random.choice(fill_choices)
        height, width = _choose_dims(index)
        specs.append(
            {
                "palette_index": index,
                "orientation": orientation,
                "border_color": border_color,
                "fill_color": fill_color,
                "height": height,
                "width": width,
                "border_pixels": index + 6,
            }
        )

    if len({spec["orientation"] for spec in specs}) != num_objects:
        return None
    if len({spec["palette_index"] for spec in specs}) < min(3, num_objects):
        return None
    if not any(spec["height"] != spec["width"] for spec in specs):
        return None
    border_counts = [sum(spec["border_color"] == other["border_color"] for other in specs) for spec in specs]
    fill_counts = [sum(spec["fill_color"] == other["fill_color"] for other in specs) for spec in specs]
    if max(border_counts) < 2 and max(fill_counts) < 2:
        return None
    return specs


def _layout_specs(specs: list[dict]) -> tuple[int, int, list[dict]] | None:
    ordered = specs[:]
    random.shuffle(ordered)
    row_counts = random.choice([(2, 1), (1, 2)]) if len(ordered) == 3 else (2, 2)
    groups = []
    cursor = 0
    for count in row_counts:
        groups.append(ordered[cursor : cursor + count])
        cursor += count

    gap_xs = [[random.randint(2, 4) for _ in range(max(0, len(group) - 1))] for group in groups]
    row_widths = [
        sum(spec["width"] for spec in group) + sum(gap_xs[row_id])
        for row_id, group in enumerate(groups)
    ]
    row_heights = [max(spec["height"] for spec in group) for group in groups]

    base_w = max(row_widths) + 2
    base_h = 1 + 1 + sum(row_heights) + 2 + 1
    if base_w > 30 or base_h > 30:
        return None

    w = min(30, base_w + random.randint(0, min(5, 30 - base_w)))
    h = min(30, base_h + random.randint(0, min(4, 30 - base_h)))
    extra_w = w - base_w
    extra_h = h - base_h

    top_slack = random.randint(0, extra_h)
    middle_slack = random.randint(0, extra_h - top_slack)
    first_row_y = 2 + top_slack
    second_row_y = first_row_y + row_heights[0] + 2 + middle_slack
    row_ys = [first_row_y, second_row_y]

    placed: list[dict] = []
    for row_id, group in enumerate(groups):
        row_width = row_widths[row_id]
        left = 1 + random.randint(0, extra_w + (max(row_widths) - row_width))
        x = left
        for idx, spec in enumerate(group):
            y = row_ys[row_id] + random.randint(0, row_heights[row_id] - spec["height"])
            placed.append({**spec, "top": y, "left": x})
            x += spec["width"]
            if idx < len(group) - 1:
                x += gap_xs[row_id][idx]

    # Strong non-adjacency check on expanded bounding boxes.
    boxes = []
    for spec in placed:
        r0 = spec["top"]
        c0 = spec["left"]
        r1 = r0 + spec["height"] - 1
        c1 = c0 + spec["width"] - 1
        boxes.append((r0, c0, r1, c1))
    for i, (r0a, c0a, r1a, c1a) in enumerate(boxes):
        for j, (r0b, c0b, r1b, c1b) in enumerate(boxes):
            if i >= j:
                continue
            if not (r1a + 1 < r0b or r1b + 1 < r0a or c1a + 1 < c0b or c1b + 1 < c0a):
                return None
    return h, w, placed


def _render_pair(h: int, w: int, specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    inp[0, :10] = PALETTE_ROW_PREFIX
    out[0, :10] = PALETTE_ROW_PREFIX
    for spec in specs:
        r0 = spec["top"]
        c0 = spec["left"]
        input_sprite = _draw_input_object(
            spec["height"],
            spec["width"],
            spec["border_color"],
            spec["fill_color"],
            spec["orientation"],
        )
        output_sprite = _draw_expected_object(spec)
        inp[r0 : r0 + spec["height"], c0 : c0 + spec["width"]] = input_sprite
        out[r0 : r0 + spec["height"], c0 : c0 + spec["width"]] = output_sprite
    return inp, out


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, specs: list[dict]) -> bool:
    parsed = _extract_objects(inp)
    if len(parsed) != len(specs):
        return False
    if len({item["orientation"] for item in parsed}) != len(parsed):
        return False
    if len({item["palette_index"] for item in parsed}) < min(3, len(parsed)):
        return False
    if not any(item["height"] != item["width"] for item in parsed):
        return False
    if len({item["palette_color"] for item in parsed}) < min(3, len(parsed)):
        return False
    if not np.array_equal(main(inp), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    num_objects = 4 if random.random() < 0.65 else 3
    specs = _sample_object_specs(num_objects)
    if specs is None:
        return None
    laid_out = _layout_specs(specs)
    if laid_out is None:
        return None
    h, w, placed_specs = laid_out
    inp, expected = _render_pair(h, w, placed_specs)
    if not _semantic_checks(inp, expected, placed_specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested corner-count puzzle")
