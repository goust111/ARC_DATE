from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
CORNERS = ("tl", "tr", "bl", "br")
WORK_CANVAS = 30
MIN_OUTER_MARGIN = 1
MAX_BORDER_PADDING = 4
PAIR_COUNT_OPTIONS = (1, 2, 3, 4)
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 300


def _connected_components(grid: np.ndarray, bg: int = BACKGROUND) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append({"color": color, "coords": np.array(coords, dtype=int)})

    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _analyze_component(grid: np.ndarray, component: dict) -> dict:
    color = int(component["color"])
    coords = component["coords"]
    r0, c0, r1, c1 = _bbox(coords)
    window = grid[r0 : r1 + 1, c0 : c1 + 1]
    local_mask = window == color

    assert np.all((window == BACKGROUND) | local_mask), "components must be isolated single-color objects"

    missing = ~local_mask
    assert missing.any(), "each object must omit a corner rectangle"

    miss_coords = np.argwhere(missing)
    mr0, mc0 = miss_coords.min(axis=0)
    mr1, mc1 = miss_coords.max(axis=0)
    mr0 = int(mr0)
    mc0 = int(mc0)
    mr1 = int(mr1)
    mc1 = int(mc1)

    miss_h = mr1 - mr0 + 1
    miss_w = mc1 - mc0 + 1
    assert int(missing.sum()) == miss_h * miss_w, "missing region must be a solid rectangle"
    assert missing[mr0 : mr1 + 1, mc0 : mc1 + 1].all(), "missing rectangle must be contiguous"

    height, width = local_mask.shape
    touches_top = mr0 == 0
    touches_bottom = mr1 == height - 1
    touches_left = mc0 == 0
    touches_right = mc1 == width - 1
    assert (touches_top + touches_bottom) == 1, "missing rectangle must touch exactly one vertical side"
    assert (touches_left + touches_right) == 1, "missing rectangle must touch exactly one horizontal side"

    if touches_top and touches_left:
        corner = "tl"
    elif touches_top and touches_right:
        corner = "tr"
    elif touches_bottom and touches_left:
        corner = "bl"
    else:
        corner = "br"

    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "missing_bbox": (r0 + mr0, c0 + mc0, r0 + mr1, c0 + mc1),
        "signature": (miss_h, miss_w),
        "norm_signature": tuple(sorted((miss_h, miss_w))),
        "corner": corner,
    }


def _collect_l_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    objects = [_analyze_component(g, component) for component in _connected_components(g, BACKGROUND)]
    assert objects and len(objects) % 2 == 0, "expected an even number of L-shaped objects"
    return objects


def _pair_groups(objects: list[dict]) -> list[list[dict]]:
    exact: dict[tuple[int, int], list[dict]] = {}
    for obj in objects:
        exact.setdefault(obj["signature"], []).append(obj)
    if all(len(group) == 2 for group in exact.values()):
        return list(exact.values())

    flexible: dict[tuple[int, int], list[dict]] = {}
    for obj in objects:
        flexible.setdefault(obj["norm_signature"], []).append(obj)
    assert all(len(group) == 2 for group in flexible.values()), "missing-corner pairing must be unique"
    return list(flexible.values())


def _paint_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each L-shape to its full rectangle using the paired L-shape's color.

    Input structure:
    - The background is black.
    - Several disconnected monochrome L-shapes are scattered across the grid.
    - Each L-shape is an axis-aligned rectangle with one rectangular corner removed.
    - The number of L-shapes is even, and their missing corner rectangles can be uniquely paired by shape.

    Rule:
    - For each colored component, find its bounding box and infer the missing corner rectangle as the cells inside that box that are still black.
    - Pair L-shapes whose missing corner rectangles have the same shape.
    - Fill each missing corner rectangle in place, but use the color of its paired L-shape rather than its own color.
    - Leave every original colored cell unchanged.

    Key insight:
    - The task is not just rectangle completion; the missing corner geometry identifies object pairs, and the partner's color determines the fill.
    """
    g = np.asarray(grid, dtype=int)
    objects = _collect_l_objects(g)
    groups = _pair_groups(objects)

    out = g.copy()
    for first, second in groups:
        _paint_rect(out, first["missing_bbox"], second["color"])
        _paint_rect(out, second["missing_bbox"], first["color"])
    return out


def _sample_signature(pair_count: int, used: set[tuple[int, int]]) -> tuple[int, int]:
    sig_cap = 5 if pair_count <= 2 else 4
    candidates = [(h, w) for h in range(1, sig_cap + 1) for w in range(1, sig_cap + 1) if (h, w) not in used]
    assert candidates, "signature pool exhausted"
    return random.choice(candidates)


def _sample_object_spec(signature: tuple[int, int], color: int, pair_count: int) -> dict:
    miss_h, miss_w = signature
    extra_cap = 4 if pair_count <= 2 else 3
    outer_h = miss_h + random.randint(1, extra_cap)
    outer_w = miss_w + random.randint(1, extra_cap)
    return {
        "color": int(color),
        "missing": signature,
        "outer": (outer_h, outer_w),
        "corner": random.choice(CORNERS),
    }


def _missing_slice(top: int, left: int, outer_h: int, outer_w: int, miss_h: int, miss_w: int, corner: str) -> tuple[slice, slice]:
    if corner == "tl":
        return slice(top, top + miss_h), slice(left, left + miss_w)
    if corner == "tr":
        return slice(top, top + miss_h), slice(left + outer_w - miss_w, left + outer_w)
    if corner == "bl":
        return slice(top + outer_h - miss_h, top + outer_h), slice(left, left + miss_w)
    return slice(top + outer_h - miss_h, top + outer_h), slice(left + outer_w - miss_w, left + outer_w)


def _draw_object(grid: np.ndarray, spec: dict, top: int, left: int) -> None:
    outer_h, outer_w = spec["outer"]
    miss_h, miss_w = spec["missing"]
    color = spec["color"]

    grid[top : top + outer_h, left : left + outer_w] = color
    rs, cs = _missing_slice(top, left, outer_h, outer_w, miss_h, miss_w, spec["corner"])
    grid[rs, cs] = BACKGROUND


def _place_objects(specs: list[dict]) -> np.ndarray | None:
    grid = np.full((WORK_CANVAS, WORK_CANVAS), BACKGROUND, dtype=int)
    blocked = np.zeros_like(grid, dtype=bool)
    order = list(range(len(specs)))
    random.shuffle(order)

    for idx in order:
        spec = specs[idx]
        outer_h, outer_w = spec["outer"]
        row_hi = WORK_CANVAS - outer_h - MIN_OUTER_MARGIN
        col_hi = WORK_CANVAS - outer_w - MIN_OUTER_MARGIN
        if row_hi < MIN_OUTER_MARGIN or col_hi < MIN_OUTER_MARGIN:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(MIN_OUTER_MARGIN, row_hi)
            left = random.randint(MIN_OUTER_MARGIN, col_hi)
            rr0 = top - 1
            rr1 = top + outer_h + 1
            cc0 = left - 1
            cc1 = left + outer_w + 1
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            blocked[rr0:rr1, cc0:cc1] = True
            spec["top"] = top
            spec["left"] = left
            _draw_object(grid, spec, top, left)
            placed = True
            break
        if not placed:
            return None

    return grid


def _crop_grid(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BACKGROUND)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)

    top_keep = random.randint(1, min(MAX_BORDER_PADDING, r0))
    left_keep = random.randint(1, min(MAX_BORDER_PADDING, c0))
    bottom_keep = random.randint(1, min(MAX_BORDER_PADDING, WORK_CANVAS - 1 - r1))
    right_keep = random.randint(1, min(MAX_BORDER_PADDING, WORK_CANVAS - 1 - c1))

    return grid[r0 - top_keep : r1 + bottom_keep + 1, c0 - left_keep : c1 + right_keep + 1]


def _build_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_OPTIONS)
    n_objects = 2 * pair_count
    colors = random.sample(range(1, 10), n_objects)
    used_signatures: set[tuple[int, int]] = set()
    specs: list[dict] = []

    for pair_idx in range(pair_count):
        signature = _sample_signature(pair_count, used_signatures)
        used_signatures.add(signature)
        color_a = colors[2 * pair_idx]
        color_b = colors[2 * pair_idx + 1]
        specs.append(_sample_object_spec(signature, color_a, pair_count))
        specs.append(_sample_object_spec(signature, color_b, pair_count))

    padded_area = sum((spec["outer"][0] + 2) * (spec["outer"][1] + 2) for spec in specs)
    if padded_area > WORK_CANVAS * WORK_CANVAS * 0.72:
        return None

    grid = _place_objects(specs)
    if grid is None:
        return None
    cropped = _crop_grid(grid)

    objects = _collect_l_objects(cropped)
    groups = _pair_groups(objects)
    if len(objects) != n_objects or len(groups) != pair_count:
        return None

    out = main(cropped)
    if out.shape != cropped.shape:
        return None
    if np.array_equal(out, cropped):
        return None
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
