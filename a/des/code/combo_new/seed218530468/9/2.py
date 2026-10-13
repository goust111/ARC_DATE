from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
FRAME_COLOR = 5
CANVAS_SIDE = 30

FRAME_COUNT_CHOICES = (1, 2, 3)
INNER_H_RANGE = (4, 8)
INNER_W_RANGE = (4, 8)
HOLE_AREA_MIN = 2
HOLE_AREA_MAX = 8

MAX_GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 100
REUSE_SHAPE_PROB = 0.4
TOUCH_BORDER_PROB = 0.5
MARGIN_MAX = 3

FILL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
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
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    if not shape:
        return 0, 0
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _shift_shape(shape: tuple[tuple[int, int], ...], dr: int, dc: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((r + dr, c + dc) for r, c in shape))


def _transform_shape(shape: tuple[tuple[int, int], ...], kind: int) -> tuple[tuple[int, int], ...]:
    h, w = _shape_dims(shape)
    pts = list(shape)
    if kind == 0:
        transformed = pts
    elif kind == 1:
        transformed = [(c, h - 1 - r) for r, c in pts]
    elif kind == 2:
        transformed = [(h - 1 - r, w - 1 - c) for r, c in pts]
    elif kind == 3:
        transformed = [(w - 1 - c, r) for r, c in pts]
    elif kind == 4:
        transformed = [(r, w - 1 - c) for r, c in pts]
    elif kind == 5:
        transformed = [(w - 1 - c, h - 1 - r) for r, c in pts]
    elif kind == 6:
        transformed = [(h - 1 - r, c) for r, c in pts]
    else:
        transformed = [(c, r) for r, c in pts]
    return _normalize(transformed)


def _all_transforms(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    seen = set()
    out = []
    for kind in range(8):
        transformed = _transform_shape(shape, kind)
        if transformed not in seen:
            seen.add(transformed)
            out.append(transformed)
    return out


def _is_rectangular_border(shape: tuple[tuple[int, int], ...]) -> bool:
    h, w = _shape_dims(shape)
    if h < 3 or w < 3:
        return False
    expected = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if r in (0, h - 1) or c in (0, w - 1)
    }
    return set(shape) == expected


def _is_frame_component(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * (h + w) - 4:
        return False
    coord_set = set(coords)
    for r in range(r0, r1 + 1):
        if (r, c0) not in coord_set or (r, c1) not in coord_set:
            return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in coord_set or (r1, c) not in coord_set:
            return False
    return True


def _frame_records(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    h, w = grid.shape
    frame_mask = np.zeros((h, w), dtype=bool)
    frames: list[dict] = []
    for color in [int(v) for v in np.unique(grid) if int(v) != bg]:
        for comp in _components(grid == color):
            if not _is_frame_component(comp):
                continue
            for r, c in comp:
                frame_mask[r, c] = True
            r0, c0, r1, c1 = _bbox(comp)
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            inner_values = [int(v) for v in inner.flatten() if int(v) not in (bg, color)]
            assert inner_values, "frame interior must contain a fill color"
            fill_color = Counter(inner_values).most_common(1)[0][0]
            hole_components = _components(inner == bg)
            assert hole_components, "frame interior must contain a black hole"
            hole = max(hole_components, key=len)
            hole_shape = _normalize(hole)
            hole_abs = [(r0 + 1 + r, c0 + 1 + c) for r, c in hole]
            frames.append(
                {
                    "frame_color": color,
                    "fill_color": int(fill_color),
                    "bbox": (r0, c0, r1, c1),
                    "hole_shape": hole_shape,
                    "hole_abs": hole_abs,
                }
            )
    assert frames, "expected at least one hollow frame"
    return frames, frame_mask


def _inside_frame_interior(coords: list[tuple[int, int]], frames: list[dict]) -> bool:
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        if all(r0 < r < r1 and c0 < c < c1 for r, c in coords):
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Put each scattered colored fragment back into the matching framed hole.

    Input structure:
    - The background is black.
    - Several grey hollow rectangular frames appear on the grid.
    - Inside each frame, one non-black non-grey color fills the interior except for one 4-connected black hole.
    - Outside the frames, one monochrome 4-connected fragment of that same color is scattered on the background.

    Rule:
    - Detect every hollow rectangular frame and read the color already filling its interior.
    - Extract the frame's black hole shape as normalized relative coordinates.
    - Find the outside fragment with the same color and the same normalized shape.
    - Clear that fragment from its original position and translate it into the hole without rotating or flipping it.

    Key insight:
    - Color identifies which fragment belongs to which frame, while the black cavity itself is the exact translation target.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    frames, frame_mask = _frame_records(g, bg)

    fragments_by_color: dict[int, list[dict]] = {}
    for color in [int(v) for v in np.unique(g) if int(v) != bg]:
        for comp in _components(g == color):
            if all(frame_mask[r, c] for r, c in comp):
                continue
            if _inside_frame_interior(comp, frames):
                continue
            fragments_by_color.setdefault(color, []).append(
                {
                    "coords": comp,
                    "shape": _normalize(comp),
                }
            )

    out = g.copy()
    used_fragments: set[tuple[int, int]] = set()
    for frame in frames:
        color = frame["fill_color"]
        matches = []
        for idx, fragment in enumerate(fragments_by_color.get(color, [])):
            key = (color, idx)
            if key in used_fragments:
                continue
            if fragment["shape"] == frame["hole_shape"]:
                matches.append((key, fragment))
        assert len(matches) == 1, "each frame must match exactly one outside fragment"
        key, fragment = matches[0]
        used_fragments.add(key)
        for r, c in fragment["coords"]:
            out[r, c] = bg
        for r, c in frame["hole_abs"]:
            out[r, c] = color
    return out


def _random_polyomino(area: int, max_h: int, max_w: int, rng: random.Random) -> tuple[tuple[int, int], ...] | None:
    for _ in range(SHAPE_ATTEMPTS):
        start = (rng.randrange(max_h), rng.randrange(max_w))
        cells = {start}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < max_h and 0 <= nc < max_w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))
        if len(cells) != area:
            continue
        shape = _normalize(tuple(cells))
        if _is_rectangular_border(shape):
            continue
        return shape
    return None


def _choose_shape(
    inner_h: int,
    inner_w: int,
    shape_pool: list[tuple[tuple[int, int], ...]],
    rng: random.Random,
) -> tuple[tuple[int, int], ...] | None:
    if shape_pool and rng.random() < REUSE_SHAPE_PROB:
        candidates = []
        for old_shape in shape_pool:
            for transformed in _all_transforms(old_shape):
                h, w = _shape_dims(transformed)
                if h <= inner_h and w <= inner_w and len(transformed) < inner_h * inner_w:
                    candidates.append(transformed)
        if candidates:
            return rng.choice(candidates)

    area_max = min(HOLE_AREA_MAX, inner_h * inner_w - 1)
    if area_max < HOLE_AREA_MIN:
        return None
    area = rng.randint(HOLE_AREA_MIN, area_max)
    return _random_polyomino(area, inner_h, inner_w, rng)


def _place_shape_interior(
    shape: tuple[tuple[int, int], ...],
    inner_h: int,
    inner_w: int,
    rng: random.Random,
) -> tuple[tuple[int, int], ...]:
    h, w = _shape_dims(shape)
    max_r = inner_h - h
    max_c = inner_w - w
    all_offsets = [(r, c) for r in range(max_r + 1) for c in range(max_c + 1)]
    border_offsets = [(r, c) for r, c in all_offsets if r in (0, max_r) or c in (0, max_c)]
    if border_offsets and rng.random() < TOUCH_BORDER_PROB:
        dr, dc = rng.choice(border_offsets)
    else:
        dr, dc = rng.choice(all_offsets)
    return _shift_shape(shape, dr, dc)


def _render_frame(inner_h: int, inner_w: int, fill_color: int, hole: tuple[tuple[int, int], ...]) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), fill_color, dtype=int)
    patch[0, :] = FRAME_COLOR
    patch[-1, :] = FRAME_COLOR
    patch[:, 0] = FRAME_COLOR
    patch[:, -1] = FRAME_COLOR
    for r, c in hole:
        patch[1 + r, 1 + c] = BG
    return patch


def _render_fragment(shape: tuple[tuple[int, int], ...], fill_color: int) -> np.ndarray:
    h, w = _shape_dims(shape)
    patch = np.full((h, w), BG, dtype=int)
    for r, c in shape:
        patch[r, c] = fill_color
    return patch


def _can_place(reserved: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(reserved.shape[0], top + height + pad)
    c1 = min(reserved.shape[1], left + width + pad)
    return not reserved[r0:r1, c0:c1].any()


def _mark_reserved(reserved: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(reserved.shape[0], top + height + pad)
    c1 = min(reserved.shape[1], left + width + pad)
    reserved[r0:r1, c0:c1] = True


def _place_patch(
    canvas: np.ndarray,
    reserved: np.ndarray,
    patch: np.ndarray,
    rng: random.Random,
) -> tuple[int, int] | None:
    h, w = patch.shape
    max_top = CANVAS_SIDE - h - 1
    max_left = CANVAS_SIDE - w - 1
    if max_top < 1 or max_left < 1:
        return None
    for _ in range(PLACEMENT_ATTEMPTS):
        top = rng.randint(1, max_top)
        left = rng.randint(1, max_left)
        if not _can_place(reserved, top, left, h, w, pad=1):
            continue
        canvas[top : top + h, left : left + w] = np.maximum(canvas[top : top + h, left : left + w], patch)
        _mark_reserved(reserved, top, left, h, w, pad=1)
        return top, left
    return None


def _crop_pair(inp: np.ndarray, out: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top_margin = rng.randint(1, min(MARGIN_MAX, int(r0)))
    left_margin = rng.randint(1, min(MARGIN_MAX, int(c0)))
    bottom_margin = rng.randint(1, min(MARGIN_MAX, int(inp.shape[0] - 1 - r1)))
    right_margin = rng.randint(1, min(MARGIN_MAX, int(inp.shape[1] - 1 - c1)))
    rs = slice(int(r0 - top_margin), int(r1 + bottom_margin + 1))
    cs = slice(int(c0 - left_margin), int(c1 + right_margin + 1))
    return inp[rs, cs].copy(), out[rs, cs].copy()


def _build_expected(canvas: np.ndarray, records: list[dict]) -> np.ndarray:
    out = canvas.copy()
    for record in records:
        for r, c in record["fragment_abs"]:
            out[r, c] = BG
        for r, c in record["hole_abs"]:
            out[r, c] = record["fill_color"]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    canvas = np.full((CANVAS_SIDE, CANVAS_SIDE), BG, dtype=int)
    reserved = np.zeros((CANVAS_SIDE, CANVAS_SIDE), dtype=bool)
    records: list[dict] = []
    shape_pool: list[tuple[tuple[int, int], ...]] = []

    frame_count = rng.choice(FRAME_COUNT_CHOICES)
    fill_colors = rng.sample(FILL_COLORS, frame_count)

    specs = []
    for fill_color in fill_colors:
        inner_h = rng.randint(*INNER_H_RANGE)
        inner_w = rng.randint(*INNER_W_RANGE)
        shape = _choose_shape(inner_h, inner_w, shape_pool, rng)
        if shape is None:
            return None
        hole = _place_shape_interior(shape, inner_h, inner_w, rng)
        frame_patch = _render_frame(inner_h, inner_w, fill_color, hole)
        fragment_patch = _render_fragment(shape, fill_color)
        specs.append(
            {
                "fill_color": fill_color,
                "shape": shape,
                "hole": hole,
                "frame_patch": frame_patch,
                "fragment_patch": fragment_patch,
            }
        )
        shape_pool.append(shape)

    specs.sort(key=lambda spec: spec["frame_patch"].size, reverse=True)

    for spec in specs:
        frame_pos = _place_patch(canvas, reserved, spec["frame_patch"], rng)
        if frame_pos is None:
            return None
        fragment_pos = _place_patch(canvas, reserved, spec["fragment_patch"], rng)
        if fragment_pos is None:
            return None

        frame_top, frame_left = frame_pos
        frag_top, frag_left = fragment_pos
        hole_abs = [(frame_top + 1 + r, frame_left + 1 + c) for r, c in spec["hole"]]
        fragment_abs = [(frag_top + r, frag_left + c) for r, c in spec["shape"]]
        records.append(
            {
                "fill_color": spec["fill_color"],
                "hole_abs": hole_abs,
                "fragment_abs": fragment_abs,
            }
        )

    expected = _build_expected(canvas, records)
    inp, out = _crop_pair(canvas, expected, rng)
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
