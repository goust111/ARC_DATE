from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
MAROON = 9
KEY_COLORS = [1, 2, 3, 6, 7, 8]

MIN_OBJECTS = 2
MAX_OBJECTS = 4
MAX_BBOX = 5
GEN_RETRIES = 400
MASK_RETRIES = 150


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(_components(mask)) == 1


def _full_bbox(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    h, w = mask.shape
    return r0 == 0 and c0 == 0 and r1 == h - 1 and c1 == w - 1


def _extract_masks(top: np.ndarray) -> dict[int, np.ndarray]:
    colors = sorted(set(map(int, np.unique(top))) - {BLACK, GREY})
    out: dict[int, np.ndarray] = {}
    for color in colors:
        comps = _components(top == color)
        assert len(comps) == 1
        coords = comps[0]
        r0, c0, r1, c1 = _bbox(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        out[int(color)] = mask
    assert out
    return out


def _extract_templates(bottom: np.ndarray) -> dict[int, dict[str, object]]:
    colors = sorted(set(map(int, np.unique(bottom))) - {BLACK, GREY})
    out: dict[int, dict[str, object]] = {}
    for color in colors:
        for coords in _components(bottom == color):
            r0, c0, r1, c1 = _bbox(coords)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < 3 or w < 3:
                continue
            comp_mask = np.zeros((h, w), dtype=bool)
            comp_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.array_equal(comp_mask, border):
                continue
            out[int(color)] = {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "inner_top": int(r0 + 1),
                "inner_left": int(c0 + 1),
                "interior": bottom[r0 + 1 : r1, c0 + 1 : c1].copy(),
            }
    assert out
    return out


def _assemble_output(tiles: list[dict[str, object]]) -> np.ndarray:
    ordered = sorted(tiles, key=lambda item: (int(item["left"]), int(item["top"])))
    min_top = min(int(item["top"]) for item in ordered)
    max_bottom = max(int(item["top"]) + int(item["crop"].shape[0]) - 1 for item in ordered)
    height = int(max_bottom - min_top + 1)
    width = int(sum(int(item["crop"].shape[1]) for item in ordered) + max(0, len(ordered) - 1))
    out = np.full((height, width), BLACK, dtype=int)
    col = 0
    for i, item in enumerate(ordered):
        crop = np.asarray(item["crop"], dtype=int)
        row = int(item["top"]) - min_top
        h, w = map(int, crop.shape)
        out[row : row + h, col : col + w] = crop
        col += w
        if i + 1 < len(ordered):
            out[:, col] = GREY
            col += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each upper colored region as a crop mask for the same-colored lower template.

    The input is split by one horizontal grey line. Upper monochrome regions define
    boolean masks inside their own bounding boxes. Lower same-colored rectangular
    frames contain maroon/yellow template patterns whose interior size matches the
    upper mask bbox. Keep template pixels where the mask is true, black out the rest,
    then concatenate the cropped interiors in lower-panel left-to-right order with a
    grey separator column between neighbors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    separator_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    assert len(separator_rows) == 1
    split = int(separator_rows[0])
    top = g[:split, :]
    bottom = g[split + 1 :, :]
    assert top.size and bottom.size

    masks = _extract_masks(top)
    templates = _extract_templates(bottom)
    assert set(masks) == set(templates)

    tiles = []
    for color in sorted(masks):
        mask = masks[color]
        template = templates[color]
        interior = np.asarray(template["interior"], dtype=int)
        assert interior.shape == mask.shape
        crop = interior.copy()
        crop[~mask] = BLACK
        tiles.append(
            {
                "top": int(template["inner_top"]),
                "left": int(template["inner_left"]),
                "crop": crop,
            }
        )

    return _assemble_output(tiles)


def _random_mask(h: int, w: int) -> np.ndarray:
    area = h * w
    low = max(3, (area + 1) // 2)
    for _ in range(MASK_RETRIES):
        target = random.randint(low, area - 1)
        mask = np.ones((h, w), dtype=bool)
        while int(mask.sum()) > target:
            cells = [(r, c) for r in range(h) for c in range(w) if mask[r, c]]
            random.shuffle(cells)
            removed = False
            for r, c in cells:
                mask[r, c] = False
                if _is_connected(mask) and _full_bbox(mask):
                    removed = True
                    break
                mask[r, c] = True
            if not removed:
                break
        if int(mask.sum()) == target and _full_bbox(mask):
            return mask
    raise RuntimeError("failed to generate connected full-bbox mask")


def _make_pattern(h: int, w: int) -> np.ndarray:
    rr, cc = np.indices((h, w))
    while True:
        kind = random.choice(("checker", "vstripe", "hstripe", "diag"))
        phase = random.randint(0, 1)
        if kind == "checker":
            selector = (rr + cc + phase) % 2 == 0
        elif kind == "vstripe":
            selector = (cc + phase) % 2 == 0
        elif kind == "hstripe":
            selector = (rr + phase) % 2 == 0
        else:
            selector = ((rr + cc) // 2 + phase) % 2 == 0
        a, b = (YELLOW, MAROON) if random.random() < 0.5 else (MAROON, YELLOW)
        out = np.where(selector, a, b).astype(int)
        if len(set(map(int, np.unique(out)))) == 2:
            return out


def _paint_mask(canvas: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    patch = canvas[top : top + h, left : left + w]
    patch[mask] = int(color)


def _paint_template(canvas: np.ndarray, top: int, left: int, interior: np.ndarray, color: int) -> None:
    h, w = map(int, interior.shape)
    patch = canvas[top : top + h + 2, left : left + w + 2]
    patch[0, :] = int(color)
    patch[-1, :] = int(color)
    patch[:, 0] = int(color)
    patch[:, -1] = int(color)
    patch[1:-1, 1:-1] = interior


def _layout_top(objects: list[dict[str, object]]) -> tuple[np.ndarray, list[dict[str, object]]]:
    order = objects[:]
    random.shuffle(order)
    jitter = random.randint(0, 3)
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 2)
    x = left_margin
    placements = []
    max_bottom = 0
    for i, obj in enumerate(order):
        mask = np.asarray(obj["mask"], dtype=bool)
        h, w = map(int, mask.shape)
        top = random.randint(0, jitter)
        placements.append({"obj": obj, "top": top, "left": x})
        max_bottom = max(max_bottom, top + h)
        x += w
        if i + 1 < len(order):
            x += random.randint(2, 4)
    width = x + right_margin
    height = max_bottom + bottom_margin
    canvas = np.full((height, width), BLACK, dtype=int)
    for item in placements:
        obj = item["obj"]
        _paint_mask(canvas, int(item["top"]), int(item["left"]), np.asarray(obj["mask"], dtype=bool), int(obj["color"]))
    return canvas, placements


def _layout_bottom(objects: list[dict[str, object]]) -> tuple[np.ndarray, list[dict[str, object]]]:
    order = objects[:]
    random.shuffle(order)
    jitter = random.randint(0, 2)
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 2)
    x = left_margin
    placements = []
    max_bottom = 0
    for i, obj in enumerate(order):
        interior = np.asarray(obj["pattern"], dtype=int)
        h, w = map(int, interior.shape)
        top = random.randint(0, jitter)
        placements.append({"obj": obj, "top": top, "left": x})
        max_bottom = max(max_bottom, top + h + 2)
        x += w + 2
        if i + 1 < len(order):
            x += random.randint(2, 4)
    width = x + right_margin
    height = max_bottom + bottom_margin
    canvas = np.full((height, width), BLACK, dtype=int)
    for item in placements:
        obj = item["obj"]
        _paint_template(canvas, int(item["top"]), int(item["left"]), np.asarray(obj["pattern"], dtype=int), int(obj["color"]))
    return canvas, placements


def _build_objects() -> list[dict[str, object]]:
    count = random.randint(MIN_OBJECTS, MAX_OBJECTS)
    colors = random.sample(KEY_COLORS, count)
    objects = []
    for color in colors:
        h = random.randint(2, MAX_BBOX)
        w = random.randint(2, MAX_BBOX)
        mask = _random_mask(h, w)
        pattern = _make_pattern(h, w)
        objects.append({"color": int(color), "mask": mask, "pattern": pattern})
    return objects


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    objects = _build_objects()
    top_canvas, _top_placements = _layout_top(objects)
    bottom_canvas, bottom_placements = _layout_bottom(objects)

    width = max(top_canvas.shape[1], bottom_canvas.shape[1]) + random.randint(0, 4)
    top_x = random.randint(0, width - top_canvas.shape[1])
    bottom_x = random.randint(0, width - bottom_canvas.shape[1])

    grid = np.full((top_canvas.shape[0] + 1 + bottom_canvas.shape[0], width), BLACK, dtype=int)
    grid[: top_canvas.shape[0], top_x : top_x + top_canvas.shape[1]] = top_canvas
    grid[top_canvas.shape[0], :] = GREY
    grid[top_canvas.shape[0] + 1 :, bottom_x : bottom_x + bottom_canvas.shape[1]] = bottom_canvas

    tiles = []
    for item in bottom_placements:
        obj = item["obj"]
        crop = np.asarray(obj["pattern"], dtype=int).copy()
        crop[~np.asarray(obj["mask"], dtype=bool)] = BLACK
        tiles.append(
            {
                "top": int(item["top"]) + 1,
                "left": int(item["left"]) + 1,
                "crop": crop,
            }
        )
    expected = _assemble_output(tiles)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid, expected = _generate_case()
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
