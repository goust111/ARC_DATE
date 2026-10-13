from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
MAROON = 9

FRAME_COLORS = (RED, BLUE, GREEN, YELLOW)
SIDES = ("top", "right", "bottom", "left")
CATALOG_SIZES = (3, 4)
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _component_mask(points: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(points)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return int(mask.shape[0]), int(mask.shape[1]), mask.astype(np.uint8).tobytes()


def _uniform_color(line: np.ndarray) -> int:
    vals = np.unique(line)
    assert len(vals) == 1
    return int(vals[0])


def _find_frame(g: np.ndarray) -> tuple[tuple[int, int, int, int], dict[str, int]]:
    frame_mask = np.isin(g, FRAME_COLORS)
    pts = [(int(r), int(c)) for r, c in np.argwhere(frame_mask)]
    assert pts, "expected a colored frame"
    r0, c0, r1, c1 = _bbox(pts)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h >= 5 and w >= 5, "expected a non-degenerate frame"

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    assert np.array_equal(frame_mask[r0 : r1 + 1, c0 : c1 + 1], border), "frame must be a 1-cell rectangle"

    side_color = {
        "top": _uniform_color(g[r0, c0 + 1 : c1]),
        "right": _uniform_color(g[r0 + 1 : r1, c1]),
        "bottom": _uniform_color(g[r1, c0 + 1 : c1]),
        "left": _uniform_color(g[r0 + 1 : r1, c0]),
    }
    assert len(set(side_color.values())) == 4, "frame sides must have four distinct colors"

    assert int(g[r0, c0]) in {side_color["top"], side_color["left"]}
    assert int(g[r0, c1]) in {side_color["top"], side_color["right"]}
    assert int(g[r1, c0]) in {side_color["bottom"], side_color["left"]}
    assert int(g[r1, c1]) in {side_color["bottom"], side_color["right"]}
    return (r0, c0, r1, c1), side_color


def _notch_side(mask: np.ndarray) -> str:
    holes = _components(~mask)
    candidates: list[str] = []
    h, w = mask.shape
    for comp in holes:
        touched = set()
        for r, c in comp:
            if r == 0:
                touched.add("top")
            if r == h - 1:
                touched.add("bottom")
            if c == 0:
                touched.add("left")
            if c == w - 1:
                touched.add("right")
        if len(touched) == 1:
            candidates.append(next(iter(touched)))
    assert len(candidates) == 1, "each inner shape must have exactly one directional notch"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each outer template with the frame-side color encoded by the matching inner notch.

    Input structure:
    - The background is black.
    - A central one-cell-thick rectangular frame has four differently colored sides.
    - Inside the frame are several orange connected shapes, each with one one-cell-wide notch that opens toward exactly one frame side.
    - Outside the frame are maroon template components whose shapes exactly match the inner orange shapes.

    Rule:
    - Find the central frame and read the color of each side.
    - For every inner orange shape, determine which frame side its notch opens toward by examining the missing cells inside its tight bounding box.
    - Find the outer maroon template with the exact same bounding-box mask.
    - Recolor that whole maroon template to the color of the corresponding frame side, leaving the frame and inner shapes unchanged.

    Key insight:
    - The notch direction chooses the color, while exact shape matching chooses which outer template receives that color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()

    (r0, c0, r1, c1), side_color = _find_frame(g)
    inside_window = np.zeros_like(g, dtype=bool)
    inside_window[r0 + 1 : r1, c0 + 1 : c1] = True

    inner_comps = _components((g == ORANGE) & inside_window)
    outer_comps = _components(g == MAROON)
    assert inner_comps and outer_comps, "expected both inner objects and outer templates"
    assert len(inner_comps) == len(outer_comps), "expected one outer template per inner object"

    outer_by_shape: dict[tuple[int, int, bytes], list[tuple[int, int]]] = {}
    for comp in outer_comps:
        mask, _ = _component_mask(comp)
        key = _shape_key(mask)
        assert key not in outer_by_shape, "outer templates must be shape-unique"
        outer_by_shape[key] = comp

    for comp in inner_comps:
        mask, _ = _component_mask(comp)
        key = _shape_key(mask)
        assert key in outer_by_shape, "every inner shape must have a matching outer template"
        color = side_color[_notch_side(mask)]
        for r, c in outer_by_shape.pop(key):
            out[r, c] = color

    assert not outer_by_shape, "every outer template must be matched exactly once"
    return out


def _make_u_mask(height: int, width: int, side: str, offset: int, depth: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    if side == "top":
        mask[:depth, offset] = False
    elif side == "bottom":
        mask[height - depth :, offset] = False
    elif side == "left":
        mask[offset, :depth] = False
    else:
        mask[offset, width - depth :] = False
    return mask


def _build_catalog() -> dict[str, list[np.ndarray]]:
    catalog: dict[str, list[np.ndarray]] = {side: [] for side in SIDES}
    seen: set[tuple[int, int, bytes]] = set()
    for height in CATALOG_SIZES:
        for width in CATALOG_SIZES:
            for side in SIDES:
                if side in ("top", "bottom"):
                    offsets = range(1, width - 1)
                    depths = range(1, height - 1)
                else:
                    offsets = range(1, height - 1)
                    depths = range(1, width - 1)
                for offset in offsets:
                    for depth in depths:
                        mask = _make_u_mask(height, width, side, offset, depth)
                        key = _shape_key(mask)
                        if key in seen:
                            continue
                        seen.add(key)
                        catalog[side].append(mask)
    for side in SIDES:
        assert catalog[side], f"missing generator masks for {side}"
    return catalog


MASK_CATALOG = _build_catalog()


def _random_template_set() -> list[tuple[str, np.ndarray]]:
    count = random.randint(2, 4)
    chosen_sides = random.sample(SIDES, count)
    specs: list[tuple[str, np.ndarray]] = []
    used: set[tuple[int, int, bytes]] = set()
    for side in chosen_sides:
        options = MASK_CATALOG[side][:]
        random.shuffle(options)
        pick = None
        for mask in options:
            key = _shape_key(mask)
            if key not in used:
                used.add(key)
                pick = mask.copy()
                break
        assert pick is not None
        specs.append((side, pick))
    random.shuffle(specs)
    return specs


def _placement_options(
    occupied: np.ndarray,
    shape: tuple[int, int],
    margin: int,
    padding: int,
) -> list[tuple[int, int]]:
    H, W = occupied.shape
    h, w = shape
    options: list[tuple[int, int]] = []
    max_r = H - h - margin
    max_c = W - w - margin
    if max_r < margin or max_c < margin:
        return options
    for r in range(margin, max_r + 1):
        for c in range(margin, max_c + 1):
            rr0 = max(0, r - padding)
            rr1 = min(H, r + h + padding)
            cc0 = max(0, c - padding)
            cc1 = min(W, c + w + padding)
            if not occupied[rr0:rr1, cc0:cc1].any():
                options.append((r, c))
    return options


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = color


def _paint_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> dict[str, int]:
    side_color = {"top": RED, "right": BLUE, "bottom": GREEN, "left": YELLOW}
    grid[r0, c0 + 1 : c1] = side_color["top"]
    grid[r0 + 1 : r1, c1] = side_color["right"]
    grid[r1, c0 + 1 : c1] = side_color["bottom"]
    grid[r0 + 1 : r1, c0] = side_color["left"]
    grid[r0, c0] = random.choice((side_color["top"], side_color["left"]))
    grid[r0, c1] = random.choice((side_color["top"], side_color["right"]))
    grid[r1, c0] = random.choice((side_color["bottom"], side_color["left"]))
    grid[r1, c1] = random.choice((side_color["bottom"], side_color["right"]))
    return side_color


def _generate_attempt() -> np.ndarray | None:
    specs = _random_template_set()
    interior_h = random.randint(10, 16)
    interior_w = random.randint(10, 16)

    v_center = random.randint(3, 7)
    h_center = random.randint(3, 7)
    top_margin = max(3, v_center + random.randint(-1, 1))
    bottom_margin = max(3, v_center + random.randint(-1, 1))
    left_margin = max(3, h_center + random.randint(-1, 1))
    right_margin = max(3, h_center + random.randint(-1, 1))

    H = top_margin + interior_h + 2 + bottom_margin
    W = left_margin + interior_w + 2 + right_margin
    if H > 30 or W > 30:
        return None

    grid = np.full((H, W), BLACK, dtype=int)

    frame_r0 = top_margin
    frame_c0 = left_margin
    frame_r1 = frame_r0 + interior_h + 1
    frame_c1 = frame_c0 + interior_w + 1
    side_color = _paint_frame(grid, frame_r0, frame_c0, frame_r1, frame_c1)
    expected = grid.copy()

    inner_occupied = np.zeros((interior_h, interior_w), dtype=bool)
    inner_specs = sorted(
        specs,
        key=lambda item: item[1].shape[0] * item[1].shape[1],
        reverse=True,
    )
    template_records: list[tuple[str, np.ndarray, int, int]] = []
    for side, mask in inner_specs:
        options = _placement_options(inner_occupied, mask.shape, margin=1, padding=1)
        if not options:
            return None
        top, left = random.choice(options)
        h, w = mask.shape
        inner_occupied[top : top + h, left : left + w] = True
        _paint_mask(grid, mask, frame_r0 + 1 + top, frame_c0 + 1 + left, ORANGE)
        _paint_mask(expected, mask, frame_r0 + 1 + top, frame_c0 + 1 + left, ORANGE)
        template_records.append((side, mask, 0, 0))

    outer_occupied = np.zeros((H, W), dtype=bool)
    outer_occupied[frame_r0 : frame_r1 + 1, frame_c0 : frame_c1 + 1] = True
    placed_templates: list[tuple[str, np.ndarray, int, int]] = []
    for side, mask, _unused_r, _unused_c in template_records:
        options = _placement_options(outer_occupied, mask.shape, margin=1, padding=1)
        filtered = []
        for top, left in options:
            h, w = mask.shape
            if top + h - 1 < frame_r0 or top > frame_r1 or left + w - 1 < frame_c0 or left > frame_c1:
                filtered.append((top, left))
        if not filtered:
            return None
        top, left = random.choice(filtered)
        h, w = mask.shape
        outer_occupied[top : top + h, left : left + w] = True
        _paint_mask(grid, mask, top, left, MAROON)
        _paint_mask(expected, mask, top, left, side_color[side])
        placed_templates.append((side, mask, top, left))

    assert placed_templates
    produced = main(grid)
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
