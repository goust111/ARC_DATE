from __future__ import annotations

import numpy as np


BG = 0
YELLOW = 4
ORANGE = 7
MAROON = 9
FENCE_COLORS = (1, 2, 3, 5, 6, 8)

GRID_H_RANGE = (20, 29)
GRID_W_RANGE = (20, 29)
FRAME_COUNT_BY_MODE = {
    "none": (2, 5),
    "pair": (2, 5),
    "triple": (3, 5),
}
FRAME_SIZE_RANGE = (4, 7)
COMPACT_FRAME_SIZE_RANGE = (4, 6)
MAX_GENERATION_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 200


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _is_rectangular_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    expected_area = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if int(coords.shape[0]) != expected_area:
        return False
    for r, c in coords:
        if int(r) not in (r0, r1) and int(c) not in (c0, c1):
            return False
    return True


def _expand_bbox(
    bbox: tuple[int, int, int, int], amount: int, height: int, width: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return (
        max(0, int(r0) - int(amount)),
        max(0, int(c0) - int(amount)),
        min(int(height) - 1, int(r1) + int(amount)),
        min(int(width) - 1, int(c1) + int(amount)),
    )


def _border_mask(
    shape: tuple[int, int], bbox: tuple[int, int, int, int]
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    if r1 - r0 > 1:
        mask[r0 + 1 : r1, c0] = True
        mask[r0 + 1 : r1, c1] = True
    else:
        mask[r0 : r1 + 1, c0] = True
        mask[r0 : r1 + 1, c1] = True
    return mask


def _coverage_map(specs: list[dict], shape: tuple[int, int]) -> np.ndarray:
    cover = np.zeros(shape, dtype=int)
    h, w = shape
    for spec in specs:
        bbox = _expand_bbox(spec["bbox"], spec["count"], h, w)
        cover += _border_mask(shape, bbox).astype(int)
    return cover


def _render_output_from_specs(specs: list[dict], shape: tuple[int, int]) -> np.ndarray:
    cover = _coverage_map(specs, shape)
    out = np.full(shape, BG, dtype=int)
    for spec in specs:
        bbox = _expand_bbox(spec["bbox"], spec["count"], shape[0], shape[1])
        mask = _border_mask(shape, bbox)
        out[(cover == 1) & mask] = int(spec["color"])
    out[cover == 2] = ORANGE
    out[cover >= 3] = MAROON
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each rectangular fence outward by the number of yellow dots it encloses,
    then recolor overlaps: 2-way -> orange, 3+-way -> maroon.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _background_color(g)

    specs: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color in (bg, YELLOW):
            continue
        for coords in _components(g == color):
            if not _is_rectangular_frame(coords):
                continue
            r0, c0, r1, c1 = _bbox(coords)
            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            count = int(np.sum(interior == YELLOW))
            if count <= 0:
                continue
            specs.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "count": count,
                    "color": color,
                }
            )

    if not specs:
        return np.full_like(g, bg)

    cover = np.zeros((h, w), dtype=int)
    masks: list[np.ndarray] = []
    for spec in specs:
        mask = _border_mask((h, w), _expand_bbox(spec["bbox"], spec["count"], h, w))
        masks.append(mask)
        cover += mask.astype(int)

    out = np.full_like(g, bg)
    for spec, mask in zip(specs, masks, strict=False):
        out[(cover == 1) & mask] = int(spec["color"])
    out[cover == 2] = ORANGE
    out[cover >= 3] = MAROON
    return out


def _sample_frame_size(
    rng: np.random.Generator, count: int, compact: bool
) -> tuple[int, int]:
    lo, hi = COMPACT_FRAME_SIZE_RANGE if compact else FRAME_SIZE_RANGE
    choices = [
        (h, w)
        for h in range(lo, hi + 1)
        for w in range(lo, hi + 1)
        if (h - 2) * (w - 2) >= int(count)
    ]
    idx = int(rng.integers(0, len(choices)))
    return choices[idx]


def _original_bbox_from_corner(
    point: tuple[int, int], count: int, height: int, width: int, orient: str
) -> tuple[int, int, int, int]:
    pr, pc = point
    if orient == "tl":
        r0 = pr + count
        c0 = pc + count
    elif orient == "tr":
        r0 = pr + count
        c0 = pc - count - width + 1
    elif orient == "bl":
        r0 = pr - count - height + 1
        c0 = pc + count
    elif orient == "br":
        r0 = pr - count - height + 1
        c0 = pc - count - width + 1
    else:
        raise AssertionError("unknown corner orientation")
    return int(r0), int(c0), int(r0 + height - 1), int(c0 + width - 1)


def _corner_extents(height: int, width: int, count: int, orient: str) -> tuple[int, int, int, int]:
    total_h = int(height - 1 + 2 * count)
    total_w = int(width - 1 + 2 * count)
    up = total_h if orient in ("bl", "br") else 0
    down = total_h if orient in ("tl", "tr") else 0
    left = total_w if orient in ("tr", "br") else 0
    right = total_w if orient in ("tl", "bl") else 0
    return up, down, left, right


def _boxes_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _rectangles_intersect(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _mode_matches(specs: list[dict], shape: tuple[int, int], mode: str) -> bool:
    cover = _coverage_map(specs, shape)
    has_two = bool(np.any(cover == 2))
    has_three = bool(np.any(cover >= 3))
    if mode == "none":
        return not has_two and not has_three
    if mode == "pair":
        return has_two and not has_three
    if mode == "triple":
        return has_three
    raise AssertionError("unknown mode")


def _cluster_orientations(rng: np.random.Generator, mode: str) -> tuple[str, ...]:
    if mode == "pair":
        choices = (("tl", "br"), ("tr", "bl"))
    else:
        choices = (
            ("tl", "tr", "br"),
            ("tl", "tr", "bl"),
            ("tl", "bl", "br"),
            ("tr", "bl", "br"),
        )
    return choices[int(rng.integers(0, len(choices)))]


def _build_cluster_specs(
    rng: np.random.Generator,
    shape: tuple[int, int],
    mode: str,
    colors: list[int],
) -> list[dict] | None:
    assert mode in ("pair", "triple")
    h_grid, w_grid = shape
    orientations = _cluster_orientations(rng, mode)
    specs_meta = []
    for color, orient in zip(colors, orientations, strict=False):
        count = int(rng.integers(1, 5))
        height, width = _sample_frame_size(rng, count, compact=True)
        specs_meta.append(
            {
                "color": int(color),
                "count": count,
                "height": height,
                "width": width,
                "orient": orient,
            }
        )

    max_up = max(_corner_extents(m["height"], m["width"], m["count"], m["orient"])[0] for m in specs_meta)
    max_down = max(_corner_extents(m["height"], m["width"], m["count"], m["orient"])[1] for m in specs_meta)
    max_left = max(_corner_extents(m["height"], m["width"], m["count"], m["orient"])[2] for m in specs_meta)
    max_right = max(_corner_extents(m["height"], m["width"], m["count"], m["orient"])[3] for m in specs_meta)
    if max_up + max_down >= h_grid or max_left + max_right >= w_grid:
        return None

    pr = int(rng.integers(max_up, h_grid - max_down))
    pc = int(rng.integers(max_left, w_grid - max_right))
    point = (pr, pc)

    specs: list[dict] = []
    for meta in specs_meta:
        bbox = _original_bbox_from_corner(
            point, meta["count"], meta["height"], meta["width"], meta["orient"]
        )
        r0, c0, r1, c1 = bbox
        if not (0 <= r0 <= r1 < h_grid and 0 <= c0 <= c1 < w_grid):
            return None
        specs.append(
            {
                "bbox": bbox,
                "count": int(meta["count"]),
                "color": int(meta["color"]),
            }
        )

    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            if not _boxes_separated(specs[i]["bbox"], specs[j]["bbox"], pad=1):
                return None
    return specs


def _candidate_positions(
    shape: tuple[int, int], height: int, width: int, count: int, must_clip: bool
) -> list[tuple[int, int]]:
    h_grid, w_grid = shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h_grid - height + 1):
        for c0 in range(w_grid - width + 1):
            r1 = r0 + height - 1
            c1 = c0 + width - 1
            clipped = r0 - count < 0 or c0 - count < 0 or r1 + count >= h_grid or c1 + count >= w_grid
            if clipped == must_clip:
                positions.append((int(r0), int(c0)))
    return positions


def _place_isolated_spec(
    rng: np.random.Generator,
    shape: tuple[int, int],
    color: int,
    existing_specs: list[dict],
    must_clip: bool,
) -> dict | None:
    h_grid, w_grid = shape
    for _ in range(PLACEMENT_ATTEMPTS):
        count = int(rng.integers(1, 5))
        height, width = _sample_frame_size(rng, count, compact=False)
        positions = _candidate_positions(shape, height, width, count, must_clip=must_clip)
        if not positions:
            continue
        order = rng.permutation(len(positions))
        for idx in order:
            r0, c0 = positions[int(idx)]
            bbox = (r0, c0, r0 + height - 1, c0 + width - 1)
            if any(not _boxes_separated(bbox, spec["bbox"], pad=1) for spec in existing_specs):
                continue
            expanded = _expand_bbox(bbox, count, h_grid, w_grid)
            if any(
                _rectangles_intersect(
                    expanded,
                    _expand_bbox(spec["bbox"], spec["count"], h_grid, w_grid),
                )
                for spec in existing_specs
            ):
                continue
            return {"bbox": bbox, "count": count, "color": int(color)}
    return None


def _draw_original_frame(
    grid: np.ndarray, bbox: tuple[int, int, int, int], color: int
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _place_yellow_dots(
    rng: np.random.Generator, grid: np.ndarray, bbox: tuple[int, int, int, int], count: int
) -> None:
    r0, c0, r1, c1 = bbox
    cells = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
    ]
    picks = rng.choice(len(cells), size=count, replace=False)
    for idx in np.atleast_1d(picks):
        rr, cc = cells[int(idx)]
        grid[rr, cc] = YELLOW


def _render_input(rng: np.random.Generator, specs: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        _draw_original_frame(grid, spec["bbox"], int(spec["color"]))
        _place_yellow_dots(rng, grid, spec["bbox"], int(spec["count"]))
    return grid


def _attempt_case(rng: np.random.Generator) -> np.ndarray | None:
    mode = ("none", "pair", "triple")[int(rng.integers(0, 3))]
    shape = (
        int(rng.integers(*GRID_H_RANGE)),
        int(rng.integers(*GRID_W_RANGE)),
    )

    min_frames, max_frames = FRAME_COUNT_BY_MODE[mode]
    need_extra_for_clip = bool(rng.integers(0, 2))
    cluster_size = 0 if mode == "none" else (2 if mode == "pair" else 3)
    if need_extra_for_clip and cluster_size == max_frames:
        need_extra_for_clip = False

    frame_count = int(rng.integers(min_frames, max_frames + 1))
    if need_extra_for_clip and frame_count == cluster_size:
        frame_count += 1

    colors = [int(v) for v in rng.choice(FENCE_COLORS, size=frame_count, replace=False)]
    specs: list[dict] = []

    if mode != "none":
        cluster_specs = _build_cluster_specs(rng, shape, mode, colors[:cluster_size])
        if cluster_specs is None:
            return None
        specs.extend(cluster_specs)

    extra_colors = colors[cluster_size:]
    for i, color in enumerate(extra_colors):
        must_clip = bool(need_extra_for_clip and i == 0)
        spec = _place_isolated_spec(rng, shape, color, specs, must_clip=must_clip)
        if spec is None:
            return None
        specs.append(spec)

    if not _mode_matches(specs, shape, mode):
        return None

    inp = _render_input(rng, specs, shape)
    expected = _render_output_from_specs(specs, shape)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
