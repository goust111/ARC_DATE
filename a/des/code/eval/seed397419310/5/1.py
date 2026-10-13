from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
MAX_GEN_ATTEMPTS = 500
GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 28)
PAIR_COUNT_CHOICES = (2, 3)
SHAPE_DIM_RANGE = (2, 4)
SHAPE_AREA_RANGE = (3, 6)
INNER_EXTRA_RANGE = (0, 3)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert len(pts) > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _transform_variants(mask: np.ndarray) -> list[np.ndarray]:
    base = mask.astype(bool)
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for flip in (False, True):
        flipped = np.fliplr(base) if flip else base
        for k in range(4):
            transformed = np.rot90(flipped, k)
            key = (transformed.shape, transformed.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(transformed.copy())
    return variants


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((variant.shape, variant.tobytes()) for variant in _transform_variants(mask))


def _same_mask(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _is_rect_outline(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(mask, expected))


def _strictly_contains(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar0 < br0 and ac0 < bc0 and ar1 > br1 and ac1 > bc1


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    colors = [int(color) for color in np.unique(grid) if int(color) not in (bg, BLACK)]
    candidates: list[dict] = []
    for color in colors:
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
                continue
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in coords:
                mask[r - r0, c - c0] = True
            if not _is_rect_outline(mask):
                continue
            if not np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == BLACK):
                continue
            candidates.append({"color": color, "bbox": (r0, c0, r1, c1)})

    frames = []
    for cand in candidates:
        if any(
            _strictly_contains(other["bbox"], cand["bbox"])
            for other in candidates
            if other is not cand
        ):
            continue
        frames.append(cand)
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _pattern_key_from_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[tuple[tuple[int, int], bytes], np.ndarray]:
    r0, c0, r1, c1 = bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    pts = np.argwhere(inner == BLACK)
    assert len(pts) > 0, "each frame must contain a black pattern"
    pr0, pc0 = pts.min(axis=0)
    pr1, pc1 = pts.max(axis=0)
    mask = np.zeros((pr1 - pr0 + 1, pc1 - pc0 + 1), dtype=bool)
    rel = pts - np.array([pr0, pc0])
    mask[rel[:, 0], rel[:, 1]] = True
    return _canonical_key(mask), inner


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn a pattern-shape -> fill-color mapping from demonstration frames,
    then apply it to unfilled frames with shape-matched black patterns.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames = _find_frames(g, bg)
    assert frames, "no candidate frames found"

    mapping: dict[tuple[tuple[int, int], bytes], int] = {}
    pending: list[tuple[tuple[int, int, int, int], tuple[tuple[int, int], bytes]]] = []

    for frame in frames:
        key, inner = _pattern_key_from_frame(g, frame["bbox"])
        fill_colors = set(map(int, np.unique(inner))) - {bg, BLACK}
        if fill_colors:
            assert len(fill_colors) == 1, "demonstration frame must use a single fill color"
            fill_color = next(iter(fill_colors))
            if key in mapping:
                assert mapping[key] == fill_color, "conflicting fill colors for the same pattern"
            else:
                mapping[key] = fill_color
        else:
            pending.append((frame["bbox"], key))

    assert mapping, "at least one demonstration frame is required"

    out = g.copy()
    for bbox, key in pending:
        assert key in mapping, "pending frame pattern must match some demonstration pattern"
        fill_color = mapping[key]
        r0, c0, r1, c1 = bbox
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[inner == bg] = fill_color
    return out


def _random_shape(rng: random.Random) -> np.ndarray | None:
    for _ in range(120):
        h = rng.randint(*SHAPE_DIM_RANGE)
        w = rng.randint(*SHAPE_DIM_RANGE)
        max_area = min(SHAPE_AREA_RANGE[1], h * w - 1)
        if max_area < SHAPE_AREA_RANGE[0]:
            continue
        target_area = rng.randint(SHAPE_AREA_RANGE[0], max_area)

        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        frontier = set()
        for nr, nc in _neighbors4(*start):
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))

        while len(cells) < target_area and frontier:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nr, nc in _neighbors4(*cell):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        if len(cells) != target_area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if np.all(mask):
            continue
        if len(_transform_variants(mask)) < 2:
            continue
        return mask
    return None


def _sample_shapes(rng: random.Random, count: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for _ in range(400):
        if len(shapes) == count:
            return shapes
        shape = _random_shape(rng)
        if shape is None:
            continue
        key = _canonical_key(shape)
        if key in seen:
            continue
        seen.add(key)
        shapes.append(shape)
    return None


def _choose_fill_colors(
    rng: random.Random,
    colors: list[int],
    demo_borders: list[int],
    pending_borders: list[int],
) -> list[int] | None:
    options = []
    for fills in itertools.permutations(colors, len(demo_borders)):
        if all(fill not in (demo_borders[i], pending_borders[i]) for i, fill in enumerate(fills)):
            options.append(list(fills))
    if not options:
        return None
    return rng.choice(options)


def _frame_spec(
    rng: random.Random,
    mask: np.ndarray,
    border_color: int,
    fill_color: int,
    is_demo: bool,
) -> dict:
    mh, mw = mask.shape
    inner_h = mh + 2 + rng.randint(*INNER_EXTRA_RANGE)
    inner_w = mw + 2 + rng.randint(*INNER_EXTRA_RANGE)
    row = rng.randint(1, inner_h - mh - 1)
    col = rng.randint(1, inner_w - mw - 1)
    return {
        "pattern": mask.copy(),
        "border_color": int(border_color),
        "fill_color": int(fill_color),
        "is_demo": bool(is_demo),
        "inner_h": int(inner_h),
        "inner_w": int(inner_w),
        "pattern_row": int(row),
        "pattern_col": int(col),
        "outer_h": int(inner_h + 2),
        "outer_w": int(inner_w + 2),
    }


def _render_frame(target: np.ndarray, top: int, left: int, spec: dict, bg: int, fill_empty: bool) -> None:
    outer_h = spec["outer_h"]
    outer_w = spec["outer_w"]
    border_color = spec["border_color"]
    target[top, left : left + outer_w] = border_color
    target[top + outer_h - 1, left : left + outer_w] = border_color
    target[top : top + outer_h, left] = border_color
    target[top : top + outer_h, left + outer_w - 1] = border_color

    ir0 = top + 1
    ic0 = left + 1
    inner = target[ir0 : ir0 + spec["inner_h"], ic0 : ic0 + spec["inner_w"]]
    if fill_empty:
        inner[:, :] = spec["fill_color"]
    else:
        inner[:, :] = bg

    pr = spec["pattern_row"]
    pc = spec["pattern_col"]
    patch = inner[pr : pr + spec["pattern"].shape[0], pc : pc + spec["pattern"].shape[1]]
    patch[spec["pattern"]] = BLACK


def _place_specs(
    rng: random.Random,
    height: int,
    width: int,
    specs: list[dict],
) -> list[tuple[int, int]] | None:
    order = list(range(len(specs)))
    rng.shuffle(order)
    order.sort(key=lambda idx: specs[idx]["outer_h"] * specs[idx]["outer_w"], reverse=True)

    occupied = np.zeros((height, width), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(specs)
    for idx in order:
        oh = specs[idx]["outer_h"]
        ow = specs[idx]["outer_w"]
        choices = []
        for top in range(1, height - oh):
            for left in range(1, width - ow):
                if occupied[top - 1 : top + oh + 1, left - 1 : left + ow + 1].any():
                    continue
                choices.append((top, left))
        if not choices:
            return None
        top, left = rng.choice(choices)
        positions[idx] = (top, left)
        occupied[top - 1 : top + oh + 1, left - 1 : left + ow + 1] = True
    return [pos for pos in positions if pos is not None]


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    num_pairs = rng.choice(PAIR_COUNT_CHOICES)
    bg = rng.randint(1, 9)
    usable_colors = [color for color in range(1, 10) if color != bg]

    border_colors = rng.sample(usable_colors, 2 * num_pairs)
    demo_borders = border_colors[:num_pairs]
    pending_borders = border_colors[num_pairs:]
    fill_colors = _choose_fill_colors(rng, usable_colors, demo_borders, pending_borders)
    if fill_colors is None:
        return None

    shapes = _sample_shapes(rng, num_pairs)
    if shapes is None:
        return None

    specs: list[dict] = []
    has_nontrivial_match = False
    for idx, shape in enumerate(shapes):
        variants = _transform_variants(shape)
        demo_mask = rng.choice(variants)
        if len(variants) > 1 and rng.random() < 0.8:
            different = [variant for variant in variants if not _same_mask(variant, demo_mask)]
            pending_mask = rng.choice(different if different else variants)
        else:
            pending_mask = rng.choice(variants)
        if not _same_mask(demo_mask, pending_mask):
            has_nontrivial_match = True

        specs.append(_frame_spec(rng, demo_mask, demo_borders[idx], fill_colors[idx], True))
        specs.append(_frame_spec(rng, pending_mask, pending_borders[idx], fill_colors[idx], False))

    if not has_nontrivial_match:
        return None

    max_h = max(spec["outer_h"] for spec in specs) + 2
    max_w = max(spec["outer_w"] for spec in specs) + 2
    total_area = sum(spec["outer_h"] * spec["outer_w"] for spec in specs)

    dims = [
        (h, w)
        for h in range(max(GRID_H_RANGE[0], max_h), GRID_H_RANGE[1] + 1)
        for w in range(max(GRID_W_RANGE[0], max_w), GRID_W_RANGE[1] + 1)
        if h * w >= total_area * 2
    ]
    if not dims:
        return None

    height, width = rng.choice(dims)
    positions = _place_specs(rng, height, width, specs)
    if positions is None:
        return None

    inp = np.full((height, width), bg, dtype=int)
    out = np.full((height, width), bg, dtype=int)
    for spec, (top, left) in zip(specs, positions):
        _render_frame(inp, top, left, spec, bg, fill_empty=spec["is_demo"])
        _render_frame(out, top, left, spec, bg, fill_empty=True)

    if _get_bg(inp) != bg:
        return None
    return inp, out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    built = _build_case(rng)
    if built is None:
        return None
    inp, out = built
    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
