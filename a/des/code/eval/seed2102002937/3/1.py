import random
from collections import Counter, deque

import numpy as np


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0, "shape mask must contain at least one cell"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    norm = _normalize_mask(mask)
    return tuple(tuple(int(v) for v in row) for row in norm.astype(int))


def _is_rectangular_border(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3 or len(coords) != 2 * (h + w) - 4:
        return False
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(sub == color, border)


def _find_templates(grid: np.ndarray, bg: int) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    comps = _components_mask(grid != bg)
    templates = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if sub.shape != (3, 3):
            continue
        if len(comp) != 9:
            continue
        colors = set(map(int, np.unique(sub)))
        if bg in colors or len(colors) != 2:
            continue
        templates.append(((r0, c0, r1, c1), sub.copy()))
    return templates


def _find_frames(grid: np.ndarray, bg: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    frames: list[tuple[int, tuple[int, int, int, int]]] = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color == bg:
            continue
        for comp in _components_mask(grid == color):
            if _is_rectangular_border(grid, comp, color):
                frames.append((color, _bbox(comp)))
    return frames


def _frame_shape(grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> tuple[tuple[int, ...], ...]:
    r0, c0, r1, c1 = bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    seed_mask = interior != bg
    assert np.any(seed_mask), "each frame must contain a seed pattern"
    return _shape_signature(seed_mask)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each frame's interior seed shape to a 3x3 two-color template and recolor the frame border
    using the template's non-shape color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    templates = _find_templates(g, bg)
    frames = _find_frames(g, bg)
    assert templates, "expected at least one 3x3 template"
    assert frames, "expected at least one rectangular frame"
    assert len(templates) == len(frames), "template count must equal frame count"

    frame_shapes: dict[tuple[tuple[int, ...], ...], tuple[int, int, int, int]] = {}
    for frame_color, bbox in frames:
        sig = _frame_shape(g, bbox, bg)
        assert sig not in frame_shapes, "frame seed shapes must be unique"
        frame_shapes[sig] = bbox
        assert frame_color == frames[0][0], "all frame borders must share one color"

    shape_to_fill: dict[tuple[tuple[int, ...], ...], int] = {}
    for _bbox0, sub in templates:
        colors = [int(v) for v in np.unique(sub)]
        matches: list[tuple[tuple[tuple[int, ...], ...], int]] = []
        for color in colors:
            sig = _shape_signature(sub == color)
            if sig in frame_shapes:
                matches.append((sig, color))
        assert len(matches) == 1, "each template must match exactly one frame shape"
        sig, shape_color = matches[0]
        fill_color = next(color for color in colors if color != shape_color)
        assert sig not in shape_to_fill, "template mapping must be unique"
        shape_to_fill[sig] = fill_color

    assert set(shape_to_fill) == set(frame_shapes), "every frame shape must have a matching template"

    out = g.copy()
    for sig, bbox in frame_shapes.items():
        fill_color = shape_to_fill[sig]
        r0, c0, r1, c1 = bbox
        out[r0, c0 : c1 + 1] = fill_color
        out[r1, c0 : c1 + 1] = fill_color
        out[r0 : r1 + 1, c0] = fill_color
        out[r0 : r1 + 1, c1] = fill_color
    return out


_BASE_SHAPES = [
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
]

_GRID_H_RANGE = (20, 29)
_GRID_W_RANGE = (20, 29)
_OBJECT_COUNT_CHOICES = (2, 3)
_PALETTE = tuple(range(10))
_GAP = 1
_FRAME_MARGIN = 1
_FRAME_OUTER_MIN = 5
_FRAME_OUTER_MAX = 10
_PLACEMENT_RETRIES = 150
_GEN_RETRIES = 300


def _transforms(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for candidate in (rot, np.fliplr(rot)):
            sig = _shape_signature(candidate)
            if sig not in seen:
                seen.add(sig)
                variants.append(_normalize_mask(candidate))
    return variants


_SHAPE_LIBRARY = []
for _base in _BASE_SHAPES:
    for _variant in _transforms(_base):
        sig = _shape_signature(_variant)
        if sig not in { _shape_signature(x) for x in _SHAPE_LIBRARY }:
            _SHAPE_LIBRARY.append(_variant)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def _shape_fits(mask: np.ndarray, canvas_h: int, canvas_w: int) -> bool:
    return mask.shape[0] <= canvas_h and mask.shape[1] <= canvas_w


def _embed_mask(mask: np.ndarray, canvas_shape: tuple[int, int], rng: random.Random) -> tuple[np.ndarray, int, int]:
    ch, cw = canvas_shape
    mh, mw = mask.shape
    r0 = rng.randint(0, ch - mh)
    c0 = rng.randint(0, cw - mw)
    placed = np.zeros((ch, cw), dtype=bool)
    placed[r0 : r0 + mh, c0 : c0 + mw] = mask
    return placed, r0, c0


def _template_positions(
    h: int,
    w: int,
    count: int,
    corner: str,
    vertical: bool,
) -> list[tuple[int, int]]:
    if vertical:
        total_h = count * 3 + (count - 1) * _GAP
        total_w = 3
        r0 = 0 if "top" in corner else h - total_h
        c0 = 0 if "left" in corner else w - total_w
        return [(r0 + i * (3 + _GAP), c0) for i in range(count)]
    total_h = 3
    total_w = count * 3 + (count - 1) * _GAP
    r0 = 0 if "top" in corner else h - total_h
    c0 = 0 if "left" in corner else w - total_w
    return [(r0, c0 + i * (3 + _GAP)) for i in range(count)]


def _expanded_box(box: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    return (
        max(0, box[0] - pad),
        max(0, box[1] - pad),
        min(h - 1, box[2] + pad),
        min(w - 1, box[3] + pad),
    )


def _choose_shapes(count: int, rng: random.Random) -> list[np.ndarray]:
    indices = list(range(len(_SHAPE_LIBRARY)))
    rng.shuffle(indices)
    chosen: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for idx in indices:
        mask = _SHAPE_LIBRARY[idx]
        sig = _shape_signature(mask)
        if sig in seen:
            continue
        chosen.append(mask.copy())
        seen.add(sig)
        if len(chosen) == count:
            return chosen
    raise RuntimeError("failed to choose distinct shapes")


def _direct_output(
    inp: np.ndarray,
    frames: list[tuple[tuple[int, int, int, int], int]],
) -> np.ndarray:
    out = inp.copy()
    for (r0, c0, r1, c1), fill_color in frames:
        out[r0, c0 : c1 + 1] = fill_color
        out[r1, c0 : c1 + 1] = fill_color
        out[r0 : r1 + 1, c0] = fill_color
        out[r0 : r1 + 1, c1] = fill_color
    return out


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    h = rng.randint(*_GRID_H_RANGE)
    w = rng.randint(*_GRID_W_RANGE)
    count = rng.choice(_OBJECT_COUNT_CHOICES)

    colors = list(_PALETTE)
    rng.shuffle(colors)
    needed = 3 + 2 * count
    if len(colors) < needed:
        return None
    bg = colors[0]
    frame_color = colors[1]
    template_shape_color = colors[2]
    seed_colors = colors[3 : 3 + count]
    fill_colors = colors[3 + count : 3 + 2 * count]

    grid = np.full((h, w), bg, dtype=int)

    corner = rng.choice(("top-left", "top-right", "bottom-left", "bottom-right"))
    vertical = bool(rng.randint(0, 1))
    template_origins = _template_positions(h, w, count, corner, vertical)
    template_boxes = [(r, c, r + 2, c + 2) for r, c in template_origins]
    for box in template_boxes:
        if not (0 <= box[0] and 0 <= box[1] and box[2] < h and box[3] < w):
            return None

    shapes = _choose_shapes(count, rng)
    rng.shuffle(shapes)

    reserved = [_expanded_box(box, _FRAME_MARGIN, h, w) for box in template_boxes]
    frame_jobs: list[tuple[tuple[int, int, int, int], np.ndarray, int, int]] = []

    for idx in range(count):
        seed_shape = shapes[idx]
        sh, sw = seed_shape.shape
        min_oh = max(_FRAME_OUTER_MIN, sh + 4)
        min_ow = max(_FRAME_OUTER_MIN, sw + 4)
        if min_oh > _FRAME_OUTER_MAX or min_ow > _FRAME_OUTER_MAX:
            return None

        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            oh = rng.randint(min_oh, _FRAME_OUTER_MAX)
            ow = rng.randint(min_ow, _FRAME_OUTER_MAX)
            r0 = rng.randint(0, h - oh)
            c0 = rng.randint(0, w - ow)
            box = (r0, c0, r0 + oh - 1, c0 + ow - 1)
            padded = _expanded_box(box, _FRAME_MARGIN, h, w)
            if any(_boxes_overlap(padded, other) for other in reserved):
                continue
            reserved.append(padded)
            frame_jobs.append((box, seed_shape, seed_colors[idx], fill_colors[idx]))
            placed = True
            break
        if not placed:
            return None

    template_order = list(range(count))
    rng.shuffle(template_order)
    for template_slot, shape_idx in enumerate(template_order):
        origin_r, origin_c = template_origins[template_slot]
        embedded, _, _ = _embed_mask(shapes[shape_idx], (3, 3), rng)
        patch = np.full((3, 3), fill_colors[shape_idx], dtype=int)
        patch[embedded] = template_shape_color
        grid[origin_r : origin_r + 3, origin_c : origin_c + 3] = patch

    expected_frames: list[tuple[tuple[int, int, int, int], int]] = []
    for box, seed_shape, seed_color, fill_color in frame_jobs:
        r0, c0, r1, c1 = box
        grid[r0, c0 : c1 + 1] = frame_color
        grid[r1, c0 : c1 + 1] = frame_color
        grid[r0 : r1 + 1, c0] = frame_color
        grid[r0 : r1 + 1, c1] = frame_color

        ih = r1 - r0 - 1
        iw = c1 - c0 - 1
        if not _shape_fits(seed_shape, ih - 2, iw - 2):
            return None
        seed_canvas, sr0, sc0 = _embed_mask(seed_shape, (ih - 2, iw - 2), rng)
        rr = r0 + 2 + sr0
        cc = c0 + 2 + sc0
        sh, sw = seed_shape.shape
        grid[rr : rr + sh, cc : cc + sw][seed_shape] = seed_color
        expected_frames.append((box, fill_color))

    values, counts = np.unique(grid, return_counts=True)
    color_counts = dict(zip(map(int, values), map(int, counts)))
    if color_counts[bg] != max(color_counts.values()) or list(color_counts.values()).count(color_counts[bg]) != 1:
        return None

    expected = _direct_output(grid, expected_frames)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        rng = random.Random()
        built = _build_case(rng)
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate valid input")
