from __future__ import annotations

from collections import deque
import random

import numpy as np


BASELINE_COLOR = 5
ALL_COLORS = tuple(range(10))
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GENERATION_ATTEMPTS = 400
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (20, 30)
TOP_BAND_HEIGHT_RANGE = (2, 5)
REGION_GAP_RANGE = (3, 7)
RULER_COUNT_RANGE = (3, 5)
BASE_RULER_LENGTH_RANGE = (2, 4)
SHAPE_AREA_RANGE = (3, 6)
SCALED_PAD = 1
RULER_PAD = 1
RULER_ROW_MARGIN = 1
MAX_SHAPE_BBOX = 4


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _connected_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )

    return comps


def _is_horizontal_segment(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return r0 == r1 and int(comp["area"]) == c1 - c0 + 1 and c1 > c0


def _infer_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    comps = _connected_components(g, bg)
    assert comps, "expected foreground objects"

    h = int(g.shape[0])
    candidates: list[tuple[int, int, list[dict], list[dict]]] = []
    for split in range(h - 1):
        top: list[dict] = []
        bottom: list[dict] = []
        valid = True
        for comp in comps:
            r0, _c0, r1, _c1 = comp["bbox"]
            if r1 <= split:
                top.append(comp)
            elif r0 > split:
                bottom.append(comp)
            else:
                valid = False
                break

        if not valid or not top or not bottom:
            continue
        if not all(_is_horizontal_segment(comp) for comp in top):
            continue

        top_colors = [int(comp["color"]) for comp in top]
        if len(set(top_colors)) != len(top_colors):
            continue
        if BASELINE_COLOR not in top_colors:
            continue

        bottom_colors = {int(comp["color"]) for comp in bottom}
        if BASELINE_COLOR in bottom_colors or not bottom_colors:
            continue
        if not bottom_colors.issubset(set(top_colors) - {BASELINE_COLOR}):
            continue

        gap = min(comp["bbox"][0] for comp in bottom) - max(comp["bbox"][2] for comp in top)
        candidates.append((gap, split, top, bottom))

    assert candidates, "failed to infer ruler/shape split"
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _gap, _split, top, bottom = candidates[0]
    return bg, top, bottom


def _ruler_scales(rulers: list[dict]) -> dict[int, int]:
    lengths = {}
    for comp in rulers:
        color = int(comp["color"])
        _r0, c0, _r1, c1 = comp["bbox"]
        lengths[color] = int(c1 - c0 + 1)

    assert BASELINE_COLOR in lengths, "missing baseline ruler"
    base = int(lengths[BASELINE_COLOR])
    assert base >= 1, "baseline ruler must be nonempty"

    scales: dict[int, int] = {}
    for color, length in lengths.items():
        if color == BASELINE_COLOR:
            continue
        assert length % base == 0, "ruler length must be an integer multiple of the baseline"
        scale = int(length // base)
        assert scale in (1, 2, 3), "expected scaling factors 1..3"
        scales[int(color)] = scale
    return scales


def _mask_from_component(comp: dict) -> tuple[int, int, np.ndarray]:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for rr, cc in comp["coords"]:
        mask[int(rr) - r0, int(cc) - c0] = True
    return int(r0), int(c0), mask


def _scale_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    h, w = map(int, mask.shape)
    out = np.zeros((h * scale, w * scale), dtype=bool)
    for r, c in np.argwhere(mask):
        rr = int(r) * scale
        cc = int(c) * scale
        out[rr : rr + scale, cc : cc + scale] = True
    return out


def _apply_scaling(grid: np.ndarray, bg: int, shapes: list[dict], scales: dict[int, int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for comp in shapes:
        rows = comp["coords"][:, 0]
        cols = comp["coords"][:, 1]
        out[rows, cols] = bg

    h, w = map(int, out.shape)
    for comp in shapes:
        color = int(comp["color"])
        assert color in scales, "every shape color must have a matching ruler"
        r0, c0, mask = _mask_from_component(comp)
        scaled = _scale_mask(mask, int(scales[color]))
        sh, sw = map(int, scaled.shape)
        assert r0 + sh <= h and c0 + sw <= w, "scaled shape must stay inside the grid"
        out[r0 : r0 + sh, c0 : c0 + sw][scaled] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale each lower shape according to the ratio between its same-colored ruler
    and the grey baseline ruler.
    """
    g = np.asarray(grid, dtype=int)
    bg, rulers, shapes = _infer_scene(g)
    scales = _ruler_scales(rulers)
    return _apply_scaling(g, bg, shapes, scales)


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(int(r - min_r), int(c - min_c)) for r, c in cells}


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, copy=True)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    rot_k = random.randrange(4)
    if rot_k:
        out = np.rot90(out, k=rot_k)
    return out.copy()


def _random_shape_mask() -> np.ndarray:
    for _ in range(200):
        area = random.randint(*SHAPE_AREA_RANGE)
        cells: set[tuple[int, int]] = {(0, 0)}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nxt = (int(r + dr), int(c + dc))
                    if nxt not in cells:
                        frontier.add(nxt)
            cells.add(random.choice(tuple(frontier)))

        norm = _normalize_cells(cells)
        mask = _transform_mask(_cells_to_mask(norm))
        h, w = map(int, mask.shape)
        if 2 <= h <= MAX_SHAPE_BBOX and 2 <= w <= MAX_SHAPE_BBOX:
            return mask
    raise RuntimeError("failed to sample a non-degenerate shape")


def _sample_factors(count: int) -> list[int]:
    factors = [1, 2, 3]
    while len(factors) < count:
        factors.append(random.choice((1, 2, 3)))
    random.shuffle(factors)
    return factors


def _mark_padded(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(int(occupied.shape[0]), r0 + h + pad)
    cc1 = min(int(occupied.shape[1]), c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _place_rulers(
    width: int,
    top_height: int,
    lengths_by_color: list[tuple[int, int]],
) -> list[tuple[int, int, int, int]] | None:
    occupied = np.zeros((top_height, width), dtype=bool)
    placements: list[tuple[int, int, int, int]] = []

    order = lengths_by_color[:]
    random.shuffle(order)
    for color, length in sorted(order, key=lambda item: item[1], reverse=True):
        candidates: list[tuple[int, int]] = []
        for r in range(top_height):
            for c in range(width - length + 1):
                rr0 = max(0, r - RULER_ROW_MARGIN)
                rr1 = min(top_height, r + RULER_ROW_MARGIN + 1)
                cc0 = max(0, c - RULER_PAD)
                cc1 = min(width, c + length + RULER_PAD)
                if occupied[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((int(r), int(c)))
        if not candidates:
            return None
        r, c = random.choice(candidates)
        placements.append((int(color), int(r), int(c), int(length)))
        _mark_padded(occupied, r, c, 1, length, RULER_PAD)
    return placements


def _place_shapes(
    grid_shape: tuple[int, int],
    bottom_start: int,
    shapes: list[dict],
) -> list[dict] | None:
    h, w = map(int, grid_shape)
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[dict] = []

    order = sorted(shapes, key=lambda item: item["scaled_shape"][0] * item["scaled_shape"][1], reverse=True)
    for item in order:
        sh, sw = item["scaled_shape"]
        candidates: list[tuple[int, int]] = []
        for r0 in range(bottom_start, h - sh + 1):
            for c0 in range(0, w - sw + 1):
                rr0 = max(0, r0 - SCALED_PAD)
                cc0 = max(0, c0 - SCALED_PAD)
                rr1 = min(h, r0 + sh + SCALED_PAD)
                cc1 = min(w, c0 + sw + SCALED_PAD)
                if occupied[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((int(r0), int(c0)))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        _mark_padded(occupied, r0, c0, sh, sw, SCALED_PAD)
        placed = dict(item)
        placed["anchor"] = (int(r0), int(c0))
        placements.append(placed)

    random.shuffle(placements)
    return placements


def _render_input(
    grid_shape: tuple[int, int],
    bg: int,
    rulers: list[tuple[int, int, int, int]],
    shapes: list[dict],
) -> np.ndarray:
    grid = np.full(grid_shape, bg, dtype=int)
    for color, r, c, length in rulers:
        grid[r, c : c + length] = int(color)
    for item in shapes:
        r0, c0 = item["anchor"]
        color = int(item["color"])
        mask = item["mask"]
        for rr, cc in np.argwhere(mask):
            grid[r0 + int(rr), c0 + int(cc)] = color
    return grid


def _render_expected(inp: np.ndarray, bg: int, shapes: list[dict]) -> np.ndarray:
    out = np.array(inp, copy=True)
    for item in shapes:
        r0, c0 = item["anchor"]
        color = int(item["color"])
        mask = item["mask"]
        scale = int(item["factor"])
        for rr, cc in np.argwhere(mask):
            out[r0 + int(rr), c0 + int(cc)] = bg
        scaled = _scale_mask(mask, scale)
        sh, sw = map(int, scaled.shape)
        out[r0 : r0 + sh, c0 : c0 + sw][scaled] = color
    return out


def _sample_scene() -> tuple[np.ndarray, np.ndarray] | None:
    bg_choices = [color for color in ALL_COLORS if color != BASELINE_COLOR]
    bg = random.choice(bg_choices)

    ruler_count = random.randint(*RULER_COUNT_RANGE)
    shape_colors = random.sample([color for color in ALL_COLORS if color not in (bg, BASELINE_COLOR)], ruler_count)
    factors = _sample_factors(ruler_count)
    base_length = random.randint(*BASE_RULER_LENGTH_RANGE)

    shapes: list[dict] = []
    for color, factor in zip(shape_colors, factors):
        mask = _random_shape_mask()
        mh, mw = map(int, mask.shape)
        shapes.append(
            {
                "color": int(color),
                "factor": int(factor),
                "mask": mask,
                "scaled_shape": (mh * int(factor), mw * int(factor)),
            }
        )

    top_height = random.randint(*TOP_BAND_HEIGHT_RANGE)
    gap = random.randint(*REGION_GAP_RANGE)

    min_h = top_height + gap + max(item["scaled_shape"][0] for item in shapes)
    min_w = max(
        max(base_length * int(factor) for factor in factors),
        max(item["scaled_shape"][1] for item in shapes),
    )
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    height = random.randint(max(GRID_H_RANGE[0], min_h), GRID_H_RANGE[1])
    width = random.randint(max(GRID_W_RANGE[0], min_w), GRID_W_RANGE[1])

    lengths_by_color = [(BASELINE_COLOR, base_length)] + [
        (int(color), int(base_length * factor)) for color, factor in zip(shape_colors, factors)
    ]
    rulers = _place_rulers(width, top_height, lengths_by_color)
    if rulers is None:
        return None

    bottom_start = top_height + gap
    placed_shapes = _place_shapes((height, width), bottom_start, shapes)
    if placed_shapes is None:
        return None

    inp = _render_input((height, width), bg, rulers, placed_shapes)
    expected = _render_expected(inp, bg, placed_shapes)
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        sampled = _sample_scene()
        if sampled is None:
            continue
        inp, expected = sampled
        try:
            out = main(inp)
        except Exception:
            continue
        if np.array_equal(out, expected):
            return inp
    raise RuntimeError("failed to generate a valid ruler-scaling scene")
