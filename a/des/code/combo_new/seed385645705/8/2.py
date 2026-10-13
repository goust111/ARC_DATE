from __future__ import annotations

import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")

NOTCH_COUNT_RANGE = (2, 6)
NOTCH_DIM_VALUES = (1, 2, 3, 4)
MIN_TEMPLATE_SIDE = 8
MAX_TEMPLATE_EXTRA = 2
PIECE_DISTANCE_RANGE = (1, 3)
MARGIN_EXTRA_RANGE = (0, 1)
GEN_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == int(bg) or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max()) + 1
            c1 = int(arr[:, 1].max()) + 1
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(len(coords)),
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return comps


def _rect_shape(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0), int(c1 - c0)


def _extract_notches(
    template_patch: np.ndarray, bg: int, abs_r0: int, abs_c0: int
) -> list[dict]:
    mask = template_patch == int(bg)
    comps = _components_mask(mask)
    assert comps, "expected at least one edge notch"

    th, tw = template_patch.shape
    notches: list[dict] = []
    for comp in comps:
        rows = comp[:, 0]
        cols = comp[:, 1]
        touches = {
            "top": bool(np.any(rows == 0)),
            "bottom": bool(np.any(rows == th - 1)),
            "left": bool(np.any(cols == 0)),
            "right": bool(np.any(cols == tw - 1)),
        }
        sides = [side for side, hit in touches.items() if hit]
        assert len(sides) == 1, "each missing region must be a single edge notch"

        r0 = int(rows.min())
        c0 = int(cols.min())
        r1 = int(rows.max()) + 1
        c1 = int(cols.max()) + 1
        submask = mask[r0:r1, c0:c1]
        assert bool(np.all(submask)), "each notch must be a solid rectangle"
        notches.append(
            {
                "side": sides[0],
                "shape": (int(r1 - r0), int(c1 - c0)),
                "bbox": (abs_r0 + r0, abs_c0 + c0, abs_r0 + r1, abs_c0 + c1),
            }
        )

    return notches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each outer rectangle fragment to the unique edge notch of the same shape and slide it into the template.

    Input structure:
    - The background is black-like empty space.
    - One large monochrome connected object is a rectangular template whose boundary has several rectangular edge notches cut out as background.
    - Outside the template are smaller monochrome solid rectangles, one per notch, each with a unique height-by-width shape and a color different from the template.

    Rule:
    - Identify the largest monochrome component as the template and inspect the background cells inside its bounding box to recover all edge notches.
    - Match every outer solid rectangle to the unique notch with the same `(height, width)` shape.
    - Produce a same-size output with only the template kept: preserve the template color everywhere else and fill each matched notch with the moved piece's original color.

    Key insight:
    Shape determines which notch a piece belongs to, while the notch's edge side determines the slide direction; the final placement itself is recovered from the notch geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    comps = _monochrome_components(g, bg)
    assert comps, "expected a template and some pieces"

    template = max(comps, key=lambda comp: int(comp["area"]))
    r0, c0, r1, c1 = template["bbox"]
    template_patch = g[r0:r1, c0:c1]

    notches = _extract_notches(template_patch, bg, r0, c0)
    notch_by_shape: dict[tuple[int, int], dict] = {}
    for notch in notches:
        shape = tuple(map(int, notch["shape"]))
        assert shape not in notch_by_shape, "notch shapes must be unique"
        notch_by_shape[shape] = notch

    out = np.full_like(g, int(bg))
    out[r0:r1, c0:c1] = template_patch

    used_shapes: set[tuple[int, int]] = set()
    for comp in comps:
        if comp is template:
            continue
        pr0, pc0, pr1, pc1 = comp["bbox"]
        ph, pw = _rect_shape(comp["bbox"])
        patch = g[pr0:pr1, pc0:pc1]
        assert int(comp["area"]) == ph * pw, "each movable piece must be a solid rectangle"
        assert bool(np.all(patch == int(comp["color"]))), "piece bbox must be monochrome"

        shape = (ph, pw)
        assert shape in notch_by_shape, "every piece must match a notch shape"
        assert shape not in used_shapes, "piece shapes must be unique"
        nr0, nc0, nr1, nc1 = notch_by_shape[shape]["bbox"]
        out[nr0:nr1, nc0:nc1] = int(comp["color"])
        used_shapes.add(shape)

    assert used_shapes == set(notch_by_shape), "every notch must receive exactly one piece"
    return out


def _required_side_length(lengths: list[int]) -> int:
    if not lengths:
        return MIN_TEMPLATE_SIDE
    return max(MIN_TEMPLATE_SIDE, int(sum(lengths) + max(0, len(lengths) - 1) + 2))


def _scatter_slack(slack: int, slots: int) -> list[int]:
    extras = [0] * slots
    for _ in range(int(slack)):
        extras[random.randrange(slots)] += 1
    return extras


def _place_segments(lengths: list[int], total: int) -> list[int] | None:
    if not lengths:
        return []
    base = int(sum(lengths) + max(0, len(lengths) - 1) + 2)
    if base > total:
        return None
    extras = _scatter_slack(total - base, len(lengths) + 1)
    starts: list[int] = []
    pos = 1 + extras[0]
    for i, length in enumerate(lengths):
        starts.append(int(pos))
        pos += int(length)
        if i < len(lengths) - 1:
            pos += 1 + extras[i + 1]
    return starts


def _shape_pool() -> list[tuple[int, int]]:
    return [(h, w) for h in NOTCH_DIM_VALUES for w in NOTCH_DIM_VALUES]


def _side_cycle(count: int) -> list[str]:
    seq: list[str] = []
    while len(seq) < count:
        seq.extend(random.sample(SIDES, len(SIDES)))
    return seq[:count]


def _notch_mask_and_layout(
    template_h: int, template_w: int, notches: list[dict]
) -> tuple[np.ndarray, list[dict]] | None:
    by_side = {side: [] for side in SIDES}
    for idx, notch in enumerate(notches):
        by_side[notch["side"]].append((idx, notch))

    starts_by_idx: dict[int, int] = {}
    for side in ("top", "bottom"):
        group = by_side[side]
        starts = _place_segments([n["shape"][1] for _, n in group], template_w)
        if starts is None:
            return None
        for (idx, _), start in zip(group, starts):
            starts_by_idx[idx] = int(start)

    for side in ("left", "right"):
        group = by_side[side]
        starts = _place_segments([n["shape"][0] for _, n in group], template_h)
        if starts is None:
            return None
        for (idx, _), start in zip(group, starts):
            starts_by_idx[idx] = int(start)

    mask = np.zeros((template_h, template_w), dtype=bool)
    layout: list[dict] = []
    for idx, notch in enumerate(notches):
        h, w = map(int, notch["shape"])
        start = int(starts_by_idx[idx])
        side = notch["side"]
        if side == "top":
            r0, c0 = 0, start
        elif side == "bottom":
            r0, c0 = template_h - h, start
        elif side == "left":
            r0, c0 = start, 0
        else:
            r0, c0 = start, template_w - w
        r1 = r0 + h
        c1 = c0 + w
        if bool(mask[r0:r1, c0:c1].any()):
            return None
        mask[r0:r1, c0:c1] = True
        placed = dict(notch)
        placed["rel_bbox"] = (r0, c0, r1, c1)
        layout.append(placed)

    if len(_components_mask(mask)) != len(notches):
        return None
    template_mask = ~mask
    if len(_components_mask(template_mask)) != 1:
        return None
    if not (template_mask[0].any() and template_mask[-1].any() and template_mask[:, 0].any() and template_mask[:, -1].any()):
        return None
    return mask, layout


def _margin_requirement(notches: list[dict], side: str) -> int:
    if side == "top" or side == "bottom":
        values = [int(n["shape"][0]) + int(n["distance"]) for n in notches if n["side"] == side]
    else:
        values = [int(n["shape"][1]) + int(n["distance"]) for n in notches if n["side"] == side]
    return max(values, default=2)


def _expected_output(
    shape: tuple[int, int],
    bg: int,
    template_color: int,
    template_bbox: tuple[int, int, int, int],
    notches: list[dict],
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    r0, c0, r1, c1 = template_bbox
    out[r0:r1, c0:c1] = int(template_color)
    for notch in notches:
        nr0, nc0, nr1, nc1 = notch["bbox"]
        out[nr0:nr1, nc0:nc1] = int(notch["piece_color"])
    return out


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    notch_count = random.randint(*NOTCH_COUNT_RANGE)
    shapes = random.sample(_shape_pool(), notch_count)
    sides = _side_cycle(notch_count)
    piece_distances = [random.randint(*PIECE_DISTANCE_RANGE) for _ in range(notch_count)]

    notches = [
        {"side": side, "shape": shape, "distance": dist}
        for side, shape, dist in zip(sides, shapes, piece_distances)
    ]

    width_need = _required_side_length(
        [shape[1] for side, shape in zip(sides, shapes) if side in ("top", "bottom")]
    )
    height_need = _required_side_length(
        [shape[0] for side, shape in zip(sides, shapes) if side in ("left", "right")]
    )

    template_h = height_need + random.randint(0, MAX_TEMPLATE_EXTRA)
    template_w = width_need + random.randint(0, MAX_TEMPLATE_EXTRA)

    layout_data = _notch_mask_and_layout(template_h, template_w, notches)
    if layout_data is None:
        return None
    _mask, placed_notches = layout_data

    top_margin = _margin_requirement(placed_notches, "top") + random.randint(*MARGIN_EXTRA_RANGE)
    bottom_margin = _margin_requirement(placed_notches, "bottom") + random.randint(*MARGIN_EXTRA_RANGE)
    left_margin = _margin_requirement(placed_notches, "left") + random.randint(*MARGIN_EXTRA_RANGE)
    right_margin = _margin_requirement(placed_notches, "right") + random.randint(*MARGIN_EXTRA_RANGE)

    grid_h = top_margin + template_h + bottom_margin
    grid_w = left_margin + template_w + right_margin
    if grid_h > 30 or grid_w > 30:
        return None
    if grid_h * grid_w <= 2 * template_h * template_w:
        return None

    bg = 0
    colors = random.sample(range(1, 10), notch_count + 1)
    template_color = int(colors[0])
    piece_colors = [int(c) for c in colors[1:]]

    template_r0 = top_margin
    template_c0 = left_margin
    template_bbox = (template_r0, template_c0, template_r0 + template_h, template_c0 + template_w)

    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    grid[template_r0 : template_r0 + template_h, template_c0 : template_c0 + template_w] = int(template_color)

    for notch, piece_color in zip(placed_notches, piece_colors):
        r0, c0, r1, c1 = notch["rel_bbox"]
        abs_bbox = (
            template_r0 + r0,
            template_c0 + c0,
            template_r0 + r1,
            template_c0 + c1,
        )
        notch["bbox"] = abs_bbox
        notch["piece_color"] = int(piece_color)
        ar0, ac0, ar1, ac1 = abs_bbox
        grid[ar0:ar1, ac0:ac1] = int(bg)

    for notch in placed_notches:
        h, w = map(int, notch["shape"])
        nr0, nc0, _nr1, _nc1 = notch["bbox"]
        dist = int(notch["distance"])
        if notch["side"] == "top":
            pr0, pc0 = nr0 - dist - h, nc0
        elif notch["side"] == "bottom":
            pr0, pc0 = template_r0 + template_h + dist, nc0
        elif notch["side"] == "left":
            pr0, pc0 = nr0, nc0 - dist - w
        else:
            pr0, pc0 = nr0, template_c0 + template_w + dist
        if pr0 < 0 or pc0 < 0 or pr0 + h > grid_h or pc0 + w > grid_w:
            return None
        if bool(np.any(grid[pr0 : pr0 + h, pc0 : pc0 + w] != int(bg))):
            return None
        grid[pr0 : pr0 + h, pc0 : pc0 + w] = int(notch["piece_color"])

    expected = _expected_output(grid.shape, bg, template_color, template_bbox, placed_notches)
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_instance()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
