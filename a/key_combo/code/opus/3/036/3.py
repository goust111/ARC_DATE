from __future__ import annotations

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4
INPUT_COLORS = (3, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def bbox_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel())
    return int(counts.argmax())


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    if connectivity != 4:
        raise ValueError("this task only needs 4-connectivity")
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for nr, nc in neighbors4(cr, cc):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def find_components(
    grid: np.ndarray,
    *,
    bg: int = BG,
    connectivity: int = 4,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    out: list[dict] = []
    for color in sorted(set(map(int, g.ravel())) - {int(bg)}):
        for coords in components_mask(g == color, connectivity=connectivity):
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _is_solid_rectangle(comp: dict) -> bool:
    box = comp["bbox"]
    return comp["area"] == bbox_h(box) * bbox_w(box) and bbox_h(box) >= 2 and bbox_w(box) >= 2


def _detect_pairs(grid: np.ndarray) -> list[dict]:
    comps = [comp for comp in find_components(grid, bg=BG, connectivity=4) if _is_solid_rectangle(comp)]
    assert comps, "expected at least one solid rectangle pair"

    by_color: dict[int, list[dict]] = {}
    for comp in comps:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    pairs: list[dict] = []
    for color, group in sorted(by_color.items()):
        assert len(group) == 2, "each foreground color must define exactly one rectangle pair"
        a, b = group
        ah, aw = bbox_h(a["bbox"]), bbox_w(a["bbox"])
        bh, bw = bbox_h(b["bbox"]), bbox_w(b["bbox"])
        assert (ah, aw) == (bh, bw), "paired rectangles must share the same size"

        same_rows = (a["bbox"][0], a["bbox"][2]) == (b["bbox"][0], b["bbox"][2])
        same_cols = (a["bbox"][1], a["bbox"][3]) == (b["bbox"][1], b["bbox"][3])
        assert same_rows != same_cols, "paired rectangles must align horizontally or vertically"

        if same_rows:
            left, right = sorted(group, key=lambda comp: comp["bbox"][1])
            gap = int(right["bbox"][1] - left["bbox"][3] - 1)
            assert gap >= 1, "horizontal pair must have a gap"
            pairs.append(
                {
                    "color": int(color),
                    "orientation": "H",
                    "gap": gap,
                    "first": left,
                    "second": right,
                }
            )
        else:
            top, bottom = sorted(group, key=lambda comp: comp["bbox"][0])
            gap = int(bottom["bbox"][0] - top["bbox"][2] - 1)
            assert gap >= 1, "vertical pair must have a gap"
            pairs.append(
                {
                    "color": int(color),
                    "orientation": "V",
                    "gap": gap,
                    "first": top,
                    "second": bottom,
                }
            )

    return pairs


def _paint_classified_block(
    out: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    *,
    odd_gap: bool,
) -> None:
    if r0 > r1 or c0 > c1:
        return

    if odd_gap:
        corner_color, edge_color, inner_color = RED, BLUE, YELLOW
    else:
        corner_color, edge_color, inner_color = BLUE, YELLOW, RED

    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h == 1 or w == 1:
        out[r0 : r1 + 1, c0 : c1 + 1] = edge_color
        return

    out[r0 : r1 + 1, c0 : c1 + 1] = inner_color
    out[r0, c0 : c1 + 1] = edge_color
    out[r1, c0 : c1 + 1] = edge_color
    out[r0 : r1 + 1, c0] = edge_color
    out[r0 : r1 + 1, c1] = edge_color
    out[r0, c0] = corner_color
    out[r0, c1] = corner_color
    out[r1, c0] = corner_color
    out[r1, c1] = corner_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gap between aligned rectangle pairs with two half-blocks, then recolor each half-block by local position class and gap parity.

    Input structure:
    - The grid has a black background and 3-4 colors, each color forming exactly two solid same-sized rectangles.
    - Each same-color rectangle pair shares either the same row span or the same column span.
    - Between each pair there is a black gap of width 3-8 and different pairs can use different orientations and gap widths.

    Rule:
    - Detect every same-color rectangle pair and measure the gap between their facing sides.
    - Split the gap in half along the alignment axis: the side next to the first rectangle is one new rectangular half-block and the side next to the second rectangle is another; if the gap is odd, keep the middle row or column black.
    - Recolor each half-block by local position within that block: for even gaps use blue corners, yellow non-corner edges, and red interior; for odd gaps use red corners, blue non-corner edges, and yellow interior.
    - If a half-block has height 1 or width 1, treat all of its cells as edge cells.

    Key insight:
    - The rectangles only locate a corridor to split; the actual output colors come from classifying the newly created half-blocks, and the corner/edge/interior palette flips with the gap parity.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for pair in _detect_pairs(g):
        gap = int(pair["gap"])
        span = gap // 2
        if span == 0:
            continue
        odd_gap = (gap % 2) == 1

        first_box = pair["first"]["bbox"]
        second_box = pair["second"]["bbox"]
        if pair["orientation"] == "H":
            r0, r1 = int(first_box[0]), int(first_box[2])
            left_c0 = int(first_box[3] + 1)
            right_c1 = int(second_box[1] - 1)
            _paint_classified_block(out, r0, left_c0, r1, left_c0 + span - 1, odd_gap=odd_gap)
            _paint_classified_block(out, r0, right_c1 - span + 1, r1, right_c1, odd_gap=odd_gap)
        else:
            c0, c1 = int(first_box[1]), int(first_box[3])
            top_r0 = int(first_box[2] + 1)
            bottom_r1 = int(second_box[0] - 1)
            _paint_classified_block(out, top_r0, c0, top_r0 + span - 1, c1, odd_gap=odd_gap)
            _paint_classified_block(out, bottom_r1 - span + 1, c0, bottom_r1, c1, odd_gap=odd_gap)

    return out


def _footprint_shape(spec: dict) -> tuple[int, int]:
    if spec["orientation"] == "H":
        return int(spec["rect_h"]), int(2 * spec["rect_w"] + spec["gap"])
    return int(2 * spec["rect_h"] + spec["gap"]), int(spec["rect_w"])


def _sample_gaps(rng: np.random.Generator, pair_count: int) -> list[int]:
    gaps = [7, int(rng.choice([6, 8]))]
    remaining = [gap for gap in range(3, 9) if gap not in gaps]
    weights = {3: 4.0, 4: 3.0, 5: 3.0, 6: 2.0, 8: 2.0}
    while len(gaps) < pair_count:
        probs = np.array([weights[gap] for gap in remaining], dtype=float)
        probs /= probs.sum()
        idx = int(rng.choice(len(remaining), p=probs))
        gaps.append(int(remaining.pop(idx)))
    return gaps


def _sample_orientations(rng: np.random.Generator, pair_count: int) -> list[str]:
    orientations = rng.choice(np.array(["H", "V"]), size=pair_count, replace=True).tolist()
    if pair_count >= 3 and len(set(orientations)) == 1 and rng.random() < 0.75:
        flip_idx = int(rng.integers(0, pair_count))
        orientations[flip_idx] = "V" if orientations[flip_idx] == "H" else "H"
    return orientations


def _make_pair_spec(
    rng: np.random.Generator,
    *,
    color: int,
    orientation: str,
    gap: int,
    require_interior: bool,
) -> dict:
    if orientation == "H":
        rect_h = int(rng.integers(3 if require_interior else 2, 6))
        rect_w = int(rng.integers(2, 5))
    else:
        rect_h = int(rng.integers(2, 5))
        rect_w = int(rng.integers(3 if require_interior else 2, 6))
    spec = {
        "color": int(color),
        "orientation": orientation,
        "gap": int(gap),
        "rect_h": rect_h,
        "rect_w": rect_w,
    }
    fh, fw = _footprint_shape(spec)
    spec["footprint_h"] = fh
    spec["footprint_w"] = fw
    return spec


def _can_place(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    return not blocked[r0 : r0 + h, c0 : c0 + w].any()


def _reserve(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(blocked.shape[0], r0 + h + 1)
    cc1 = min(blocked.shape[1], c0 + w + 1)
    blocked[rr0:rr1, cc0:cc1] = True


def _draw_pair(grid: np.ndarray, spec: dict, r0: int, c0: int) -> None:
    color = int(spec["color"])
    rect_h = int(spec["rect_h"])
    rect_w = int(spec["rect_w"])
    gap = int(spec["gap"])
    if spec["orientation"] == "H":
        grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = color
        grid[r0 : r0 + rect_h, c0 + rect_w + gap : c0 + 2 * rect_w + gap] = color
    else:
        grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = color
        grid[r0 + rect_h + gap : r0 + 2 * rect_h + gap, c0 : c0 + rect_w] = color


def _validate_generated(grid: np.ndarray, expected_pairs: int) -> bool:
    try:
        pairs = _detect_pairs(grid)
        if len(pairs) != expected_pairs:
            return False
        gaps = [int(pair["gap"]) for pair in pairs]
        if len(set(gaps)) != expected_pairs:
            return False
        if not any(gap % 2 == 0 for gap in gaps):
            return False
        if not any(gap % 2 == 1 for gap in gaps):
            return False

        out = main(grid)
        if out.shape != grid.shape or np.array_equal(out, grid):
            return False
        if max(grid.shape) > 30 or max(out.shape) > 30:
            return False
        if get_bg(grid) != BG:
            return False
        if not all(np.any(out == color) for color in (BLUE, RED, YELLOW)):
            return False
    except (AssertionError, ValueError):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    pair_count = int(rng.choice([3, 4]))
    gaps = _sample_gaps(rng, pair_count)
    orientations = _sample_orientations(rng, pair_count)
    colors = list(map(int, rng.choice(np.array(INPUT_COLORS), size=pair_count, replace=False)))

    specs = [
        _make_pair_spec(
            rng,
            color=colors[idx],
            orientation=orientations[idx],
            gap=gaps[idx],
            require_interior=(idx < 2),
        )
        for idx in range(pair_count)
    ]
    specs.sort(key=lambda spec: spec["footprint_h"] * spec["footprint_w"], reverse=True)

    max_h = max(spec["footprint_h"] for spec in specs)
    max_w = max(spec["footprint_w"] for spec in specs)
    side_floor = 20 + 2 * max(0, pair_count - 3)
    grid_h = int(rng.integers(max(side_floor, max_h + 2), 29))
    grid_w = int(rng.integers(max(side_floor, max_w + 2), 29))

    grid = np.zeros((grid_h, grid_w), dtype=int)
    blocked = np.zeros_like(grid, dtype=bool)

    for spec in specs:
        fh = int(spec["footprint_h"])
        fw = int(spec["footprint_w"])
        placed = False
        for _ in range(200):
            r0 = int(rng.integers(0, grid_h - fh + 1))
            c0 = int(rng.integers(0, grid_w - fw + 1))
            if not _can_place(blocked, r0, c0, fh, fw):
                continue
            _draw_pair(grid, spec, r0, c0)
            _reserve(blocked, r0, c0, fh, fw)
            placed = True
            break
        if not placed:
            return None

    if not _validate_generated(grid, pair_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
