from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

COLOR_TO_SIDE = {
    RED: "top",
    BLUE: "bottom",
    GREEN: "left",
    YELLOW: "right",
}
FRAME_COLORS = tuple(COLOR_TO_SIDE)

GRID_MIN = 20
GRID_MAX = 25
EDGE_MARGIN = 3
FRAME_PAD = 1
MAX_GENERATION_ATTEMPTS = 400
SHAPE_ATTEMPTS = 80


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    assert rows.size > 0, "expected a nonempty pattern"
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].copy()


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    c0 = int(cols.min())
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _rect_frame_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None

    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    expected = np.zeros_like(mask, dtype=bool)
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    if not np.array_equal(mask, expected):
        return None
    return r0, c0, r1, c1


def _extract_templates(grid: np.ndarray) -> list[tuple[int, np.ndarray]]:
    templates: list[tuple[int, np.ndarray]] = []
    for color in FRAME_COLORS:
        mask = grid == color
        if not bool(mask.any()):
            continue

        bbox = _rect_frame_bbox(mask)
        assert bbox is not None, f"color {color} must form a single hollow rectangle"
        r0, c0, r1, c1 = bbox

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        grey_mask = interior == GREY
        comps = _components(grey_mask)
        assert len(comps) == 1, "each frame must contain exactly one grey pattern"
        template = _crop_mask(grey_mask)
        templates.append((color, template))

    assert templates, "expected at least one frame"
    return templates


def _stamp_horizontal(out: np.ndarray, patch: np.ndarray, align_top: bool) -> None:
    ph, pw = patch.shape
    row = 0 if align_top else out.shape[0] - ph
    col = 0
    while col + pw <= out.shape[1]:
        window = out[row : row + ph, col : col + pw]
        window[patch] = GREY
        col += pw + 1


def _stamp_vertical(out: np.ndarray, patch: np.ndarray, align_left: bool) -> None:
    ph, pw = patch.shape
    col = 0 if align_left else out.shape[1] - pw
    row = 0
    while row + ph <= out.shape[0]:
        window = out[row : row + ph, col : col + pw]
        window[patch] = GREY
        row += ph + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat each grey template extracted from a colored hollow frame along the edge assigned to that frame color.

    Input structure:
    - The background is black.
    - The interior of the grid contains several small hollow rectangular frames, each drawn
      in one solid color chosen from red, blue, green, and yellow, with at most one frame
      per color.
    - Each frame encloses exactly one grey 4-connected pattern whose tight bounding box is at
      most 3x3, and the frames do not overlap.

    Rule:
    - For every colored frame, detect its rectangular ring and extract the enclosed grey
      pattern by cropping to the pattern's own bounding box.
    - Use the fixed color-to-edge mapping red -> top, blue -> bottom, green -> left,
      yellow -> right.
    - Repeat the extracted grey pattern flush against that edge, starting from the first row
      or column, with one black separator cell between consecutive copies, stopping before a
      copy would cross the opposite boundary.
    - Clear the original frames and their source patterns; the output contains only the edge
      repetitions on a black background.

    Key insight:
    - The colored rectangle is a container, not part of the output: its color chooses one
      destination edge, while the grey shape inside it is the reusable stamp that gets tiled
      along that edge.
    """
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BLACK)

    for color, patch in _extract_templates(g):
        side = COLOR_TO_SIDE[color]
        if side == "top":
            _stamp_horizontal(out, patch, align_top=True)
        elif side == "bottom":
            _stamp_horizontal(out, patch, align_top=False)
        elif side == "left":
            _stamp_vertical(out, patch, align_left=True)
        else:
            _stamp_vertical(out, patch, align_left=False)

    return out


def _random_template() -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        box_h = random.randint(1, 3)
        box_w = random.randint(1, 3)
        max_area = min(box_h * box_w, 5)
        area = random.randint(1, max_area)

        seed = (random.randrange(box_h), random.randrange(box_w))
        cells = {seed}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0 = min(rows)
        c0 = min(cols)
        norm = {(r - r0, c - c0) for r, c in cells}
        h = 1 + max(r for r, _ in norm)
        w = 1 + max(c for _, c in norm)
        patch = np.zeros((h, w), dtype=bool)
        for r, c in norm:
            patch[r, c] = True
        if len(_components(patch)) == 1:
            return patch

    raise RuntimeError("failed to sample a connected template")


def _choose_colors(count: int) -> list[int]:
    if count == 2:
        horizontal = random.choice([RED, BLUE])
        vertical = random.choice([GREEN, YELLOW])
        colors = [horizontal, vertical]
        random.shuffle(colors)
        return colors

    colors = list(FRAME_COLORS)
    random.shuffle(colors)
    return colors[:count]


def _sample_specs(count: int) -> list[dict]:
    specs: list[dict] = []
    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    for color in _choose_colors(count):
        for _ in range(SHAPE_ATTEMPTS):
            template = _random_template()
            key = _shape_key(template)
            if key in used_shapes:
                continue
            used_shapes.add(key)
            break
        else:
            return []

        th, tw = template.shape
        interior_h = th + random.randint(0, 2)
        interior_w = tw + random.randint(0, 2)
        offset_r = random.randint(0, interior_h - th)
        offset_c = random.randint(0, interior_w - tw)
        specs.append(
            {
                "color": color,
                "template": template,
                "outer_h": interior_h + 2,
                "outer_w": interior_w + 2,
                "offset_r": offset_r,
                "offset_c": offset_c,
            }
        )
    return specs


def _can_place(occupied: np.ndarray, r0: int, c0: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    if r0 < EDGE_MARGIN or c0 < EDGE_MARGIN:
        return False
    if r0 + height > h - EDGE_MARGIN or c0 + width > w - EDGE_MARGIN:
        return False

    rr0 = max(0, r0 - FRAME_PAD)
    cc0 = max(0, c0 - FRAME_PAD)
    rr1 = min(h, r0 + height + FRAME_PAD)
    cc1 = min(w, c0 + width + FRAME_PAD)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_specs(shape: tuple[int, int], specs: list[dict]) -> list[dict] | None:
    h, w = shape
    occupied = np.zeros((h, w), dtype=bool)
    order = sorted(range(len(specs)), key=lambda idx: specs[idx]["outer_h"] * specs[idx]["outer_w"], reverse=True)
    placed: list[dict | None] = [None] * len(specs)

    for idx in order:
        spec = specs[idx]
        positions: list[tuple[int, int]] = []
        for r0 in range(EDGE_MARGIN, h - EDGE_MARGIN - spec["outer_h"] + 1):
            for c0 in range(EDGE_MARGIN, w - EDGE_MARGIN - spec["outer_w"] + 1):
                if _can_place(occupied, r0, c0, spec["outer_h"], spec["outer_w"]):
                    positions.append((r0, c0))
        if not positions:
            return None

        r0, c0 = random.choice(positions)
        rr0 = max(0, r0 - FRAME_PAD)
        cc0 = max(0, c0 - FRAME_PAD)
        rr1 = min(h, r0 + spec["outer_h"] + FRAME_PAD)
        cc1 = min(w, c0 + spec["outer_w"] + FRAME_PAD)
        occupied[rr0:rr1, cc0:cc1] = True

        placed[idx] = {
            **spec,
            "r0": r0,
            "c0": c0,
        }

    return [spec for spec in placed if spec is not None]


def _draw_frame(grid: np.ndarray, spec: dict) -> None:
    r0 = spec["r0"]
    c0 = spec["c0"]
    r1 = r0 + spec["outer_h"] - 1
    c1 = c0 + spec["outer_w"] - 1
    color = spec["color"]
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color

    patch = spec["template"]
    ph, pw = patch.shape
    pr0 = r0 + 1 + spec["offset_r"]
    pc0 = c0 + 1 + spec["offset_c"]
    grid[pr0 : pr0 + ph, pc0 : pc0 + pw][patch] = GREY


def _render_expected(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for spec in specs:
        patch = spec["template"]
        side = COLOR_TO_SIDE[spec["color"]]
        if side == "top":
            _stamp_horizontal(out, patch, align_top=True)
        elif side == "bottom":
            _stamp_horizontal(out, patch, align_top=False)
        elif side == "left":
            _stamp_vertical(out, patch, align_left=True)
        else:
            _stamp_vertical(out, patch, align_left=False)
    return out


def _sample_shape() -> tuple[int, int]:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    if random.random() < 0.4:
        width = max(GRID_MIN, min(GRID_MAX, height + random.randint(-2, 2)))
    return height, width


def _gen_attempt() -> np.ndarray | None:
    shape = _sample_shape()
    count = random.randint(2, 4)
    specs = _sample_specs(count)
    if len(specs) != count:
        return None

    placed = _place_specs(shape, specs)
    if placed is None or len(placed) != count:
        return None

    grid = np.full(shape, BLACK, dtype=int)
    for spec in placed:
        _draw_frame(grid, spec)

    expected = _render_expected(shape, placed)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    if not np.any(expected == GREY):
        return None
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_GENERATION_ATTEMPTS} attempts")
