from __future__ import annotations

import random
from importlib.util import module_from_spec, spec_from_file_location

import numpy as np


_ARC_UTILS_PATH = "/home/lixinyu/snap/new/solvers/arc_utils.py"
_ARC_UTILS_SPEC = spec_from_file_location("_arc_utils_044_6", _ARC_UTILS_PATH)
if _ARC_UTILS_SPEC is None or _ARC_UTILS_SPEC.loader is None:
    raise ImportError(f"failed to load arc_utils from {_ARC_UTILS_PATH}")
_ARC_UTILS = module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
bbox_h = _ARC_UTILS.bbox_h
bbox_w = _ARC_UTILS.bbox_w
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
in_bounds = _ARC_UTILS.in_bounds
paste = _ARC_UTILS.paste


DIVIDER_COLOR = Color.GREY
FRAME_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
GRID_SIZES = (25, 27, 29)
MAX_GENERATION_ATTEMPTS = 400

SPRITE_SHAPES_BY_AREA: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {
    2: (
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
    ),
    3: (
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 1)),
    ),
}

AREA_PROFILES = (
    {1: (3,), 2: (3, 3), 3: (2, 2, 2), 4: (2, 2, 2, 2)},
    {1: (2,), 2: (3, 3), 3: (3, 3, 3), 4: (2, 2, 2, 2)},
    {1: (3,), 2: (2, 3), 3: (3, 2, 2), 4: (2, 2, 2, 3)},
)

QUADRANTS = ((0, 0), (0, 1), (1, 0), (1, 1))


def _find_dividers(grid: np.ndarray) -> tuple[int, int, int]:
    rows = [(r, int(grid[r, 0])) for r in range(grid.shape[0]) if np.all(grid[r, :] == grid[r, 0])]
    cols = [(c, int(grid[0, c])) for c in range(grid.shape[1]) if np.all(grid[:, c] == grid[0, c])]
    matches = [(r, c, row_color) for r, row_color in rows for c, col_color in cols if row_color == col_color]
    assert len(matches) == 1, "expected exactly one full-row/full-column divider cross"
    return matches[0]


def _quadrant_slice(shape: tuple[int, int], mid_r: int, mid_c: int, quad: tuple[int, int]) -> tuple[slice, slice]:
    h, w = shape
    qb, qr = quad
    rs = slice(0, mid_r) if qb == 0 else slice(mid_r + 1, h)
    cs = slice(0, mid_c) if qr == 0 else slice(mid_c + 1, w)
    return rs, cs


def _frame_border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_hollow_frame(quad_grid: np.ndarray, comp: dict) -> bool:
    color = int(comp["color"])
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = bbox_h(comp["bbox"])
    w = bbox_w(comp["bbox"])
    if h < 4 or w < 5:
        return False
    if int(comp["area"]) != 2 * (h + w) - 4:
        return False
    patch = quad_grid[r0 : r1 + 1, c0 : c1 + 1]
    border = _frame_border_mask(h, w)
    if not np.all(patch[border] == color):
        return False
    if h > 2 and w > 2 and np.any(patch[1:-1, 1:-1] == color):
        return False
    return True


def _analyze_quadrant(grid: np.ndarray, mid_r: int, mid_c: int, quad: tuple[int, int], bg: int) -> dict:
    rs, cs = _quadrant_slice(grid.shape, mid_r, mid_c, quad)
    quad_grid = grid[rs, cs]
    colors = sorted(int(x) for x in np.unique(quad_grid) if int(x) not in {bg, DIVIDER_COLOR})
    assert colors, "quadrant must contain a colored frame"
    comps = find_components(
        quad_grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=colors,
        return_format="dict",
    )
    frame_candidates = [comp for comp in comps if _is_hollow_frame(quad_grid, comp)]
    assert len(frame_candidates) == 1, "quadrant must contain exactly one hollow frame"
    frame = frame_candidates[0]
    color = int(frame["color"])
    assert all(int(comp["color"]) == color for comp in comps), "each quadrant should use one foreground color"
    sprites = [comp for comp in comps if comp is not frame]
    frame_r0, frame_c0, frame_r1, frame_c1 = map(int, frame["bbox"])
    for sprite in sprites:
        sr0, sc0, sr1, sc1 = map(int, sprite["bbox"])
        assert not (frame_r0 <= sr0 <= frame_r1 and frame_c0 <= sc0 <= frame_c1), "sprite cannot lie inside frame bbox"
        for r, c in sprite["coords"]:
            rr = int(r)
            cc = int(c)
            assert 0 < rr < quad_grid.shape[0] - 1 and 0 < cc < quad_grid.shape[1] - 1, "sprite must stay away from quadrant boundary"
    gr0 = rs.start + frame_r0
    gc0 = cs.start + frame_c0
    gr1 = rs.start + frame_r1
    gc1 = cs.start + frame_c1
    return {
        "quad": quad,
        "color": color,
        "count": len(sprites),
        "frame_bbox": (gr0, gc0, gr1, gc1),
        "frame_bbox_local": (frame_r0, frame_c0, frame_r1, frame_c1),
        "sprites": sprites,
        "quad_slice": (rs, cs),
        "quad_grid": quad_grid,
    }


def _analyze_grid(grid: np.ndarray) -> tuple[int, int, int, list[dict]]:
    bg = int(get_bg(grid))
    mid_r, mid_c, divider_color = _find_dividers(grid)
    infos = [_analyze_quadrant(grid, mid_r, mid_c, quad, bg) for quad in QUADRANTS]
    return bg, divider_color, mid_r, mid_c, infos


def _pyramid_patch(height: int, color: int, flip_h: bool, flip_v: bool) -> np.ndarray:
    patch = np.zeros((height, 2 * height - 1), dtype=int)
    center = height - 1
    for r in range(height):
        half = r
        patch[r, center - half : center + half + 1] = color
    if flip_h:
        patch = np.fliplr(patch)
    if flip_v:
        patch = np.flipud(patch)
    return patch


def _synthesize_output(grid: np.ndarray, infos: list[dict], bg: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for info in infos:
        rs, cs = info["quad_slice"]
        for sprite in info["sprites"]:
            coords = sprite["coords"]
            out[rs.start + coords[:, 0], cs.start + coords[:, 1]] = bg

        r0, c0, r1, c1 = info["frame_bbox"]
        inner_r0, inner_c0 = r0 + 1, c0 + 1
        inner_r1, inner_c1 = r1 - 1, c1 - 1
        inner_h = inner_r1 - inner_r0 + 1
        inner_w = inner_c1 - inner_c0 + 1
        n = int(info["count"])
        assert n >= 1, "every quadrant must contribute at least one sprite"
        assert inner_h >= n and inner_w >= 2 * n - 1, "frame interior must fit pyramid"

        qb, qr = info["quad"]
        patch = _pyramid_patch(n, int(info["color"]), flip_h=(qr == 1), flip_v=(qb == 1))
        place_r = inner_r1 - patch.shape[0] + 1 if qb == 0 else inner_r0
        place_c = inner_c0 + (inner_w - patch.shape[1]) // 2
        paste(out, patch, place_r, place_c, transparent=Color.BLACK)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-color sprites per quadrant frame, turn each count into a centered pyramid, and flip it by quadrant.

    Input structure:
    - A gray full row and full column split the grid into four quadrants.
    - Each quadrant contains one single-color hollow rectangular frame and several separate same-color sprite components outside that frame.
    - A quadrant uses exactly one foreground color, while the divider and background use different colors.

    Rule:
    - In each quadrant, find the hollow frame and count the 4-connected same-color sprite components outside it.
    - Erase those sprites from the grid.
    - Generate a solid pyramid of that height with row widths 1, 3, 5, ..., 2N-1 in the frame color.
    - Place the pyramid centered inside the frame: top quadrants keep the apex upward and align the wide base to the inner bottom edge; bottom quadrants flip vertically so the wide base aligns to the inner top edge. Right-side quadrants additionally apply a horizontal reflection.

    Key insight:
    - The scattered sprites are not copied; only their component count matters, and quadrant position decides whether the generated pyramid is upright or upside-down.
    """
    g = np.asarray(grid, dtype=int)
    bg, divider_color, mid_r, mid_c, infos = _analyze_grid(g)
    assert divider_color != bg, "divider color must differ from background"
    return _synthesize_output(g, infos, bg)


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        out[rr, cc] = True
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _draw_frame(canvas: np.ndarray, r0: int, c0: int, height: int, width: int, color: int) -> None:
    canvas[r0, c0 : c0 + width] = color
    canvas[r0 + height - 1, c0 : c0 + width] = color
    canvas[r0 : r0 + height, c0] = color
    canvas[r0 : r0 + height, c0 + width - 1] = color


def _normalized_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    return tuple(sorted((r - min_r, c - min_c) for r, c in shape))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    return h, w


def _place_sprite_set(
    quad: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    color: int,
    areas: tuple[int, ...],
) -> bool:
    qsize = quad.shape[0]
    blocked = np.zeros_like(quad, dtype=bool)
    fr0, fc0, fr1, fc1 = frame_bbox
    blocked[fr0 : fr1 + 1, fc0 : fc1 + 1] = True
    blocked = _dilate4(blocked)

    for area in areas:
        shape_bank = list(SPRITE_SHAPES_BY_AREA[int(area)])
        random.shuffle(shape_bank)
        candidates: list[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = []
        for shape in shape_bank:
            norm = _normalized_shape(shape)
            sh, sw = _shape_dims(norm)
            for r0 in range(1, qsize - sh):
                for c0 in range(1, qsize - sw):
                    cells = [(r0 + dr, c0 + dc) for dr, dc in norm]
                    if any(blocked[r, c] for r, c in cells):
                        continue
                    candidates.append(((r0, c0), norm))
        if not candidates:
            return False
        (_, _), chosen = random.choice(candidates)
        anchor = random.choice([cand for cand in candidates if cand[1] == chosen])[0]
        new_mask = np.zeros_like(quad, dtype=bool)
        for dr, dc in chosen:
            rr = anchor[0] + dr
            cc = anchor[1] + dc
            quad[rr, cc] = color
            new_mask[rr, cc] = True
        blocked |= _dilate4(new_mask)
    return True


def _frame_dimensions(qsize: int) -> tuple[int, int]:
    height_choices = [6]
    width_choices = [9]
    if qsize >= 13:
        height_choices.append(7)
        width_choices.append(11)
    return random.choice(height_choices), random.choice(width_choices)


def _make_quadrant(qsize: int, color: int, count: int, area_profile: dict[int, tuple[int, ...]]) -> np.ndarray | None:
    quad = np.zeros((qsize, qsize), dtype=int)
    frame_h, frame_w = _frame_dimensions(qsize)
    r0 = (qsize - frame_h) // 2
    c0 = (qsize - frame_w) // 2
    _draw_frame(quad, r0, c0, frame_h, frame_w, color)
    frame_bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
    if not _place_sprite_set(quad, frame_bbox, color, area_profile[count]):
        return None
    return quad


def _pixel_totals_are_nontrivial(grid: np.ndarray, infos: list[dict]) -> bool:
    by_count = {}
    for info in infos:
        total_pixels = int(sum(sprite["area"] for sprite in info["sprites"]))
        by_count[int(info["count"])] = total_pixels
    counts = sorted(by_count)
    return any(by_count[counts[i]] >= by_count[counts[i + 1]] for i in range(len(counts) - 1))


def _frame_is_centered(quad_grid: np.ndarray, info: dict) -> bool:
    r0, c0, r1, c1 = info["frame_bbox_local"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    expected_r0 = (quad_grid.shape[0] - h) // 2
    expected_c0 = (quad_grid.shape[1] - w) // 2
    return r0 == expected_r0 and c0 == expected_c0


def _validate_generated_grid(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    try:
        bg, divider_color, mid_r, mid_c, infos = _analyze_grid(grid)
    except AssertionError:
        return False
    if divider_color != DIVIDER_COLOR or bg != Color.BLACK:
        return False
    if sorted(int(info["count"]) for info in infos) != [1, 2, 3, 4]:
        return False
    if len({int(info["color"]) for info in infos}) != 4:
        return False
    if not all(_frame_is_centered(info["quad_grid"], info) for info in infos):
        return False
    if not _pixel_totals_are_nontrivial(grid, infos):
        return False

    out = main(grid)
    expected = _synthesize_output(grid, infos, bg)
    if not np.array_equal(out, expected):
        return False

    # Semantic output check: all non-frame foreground outside frame interiors must be gone.
    for info in infos:
        rs, cs = info["quad_slice"]
        quad_out = out[rs, cs]
        color = int(info["color"])
        fr0, fc0, fr1, fc1 = info["frame_bbox_local"]
        outside = np.ones_like(quad_out, dtype=bool)
        outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
        if np.any(quad_out[outside] == color):
            return False
        inner = quad_out[fr0 + 1 : fr1, fc0 + 1 : fc1]
        n = int(info["count"])
        expected_patch = _pyramid_patch(n, color, flip_h=(info["quad"][1] == 1), flip_v=(info["quad"][0] == 1))
        inner_expected = np.zeros_like(inner)
        place_r = inner.shape[0] - expected_patch.shape[0] if info["quad"][0] == 0 else 0
        place_c = (inner.shape[1] - expected_patch.shape[1]) // 2
        paste(inner_expected, expected_patch, place_r, place_c, transparent=Color.BLACK)
        if not np.array_equal(inner, inner_expected):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    size = random.choice(GRID_SIZES)
    mid = size // 2
    qsize = mid
    area_profile = random.choice(AREA_PROFILES)
    counts = random.sample([1, 2, 3, 4], k=4)
    colors = random.sample(list(FRAME_COLORS), k=4)

    grid = np.zeros((size, size), dtype=int)
    grid[mid, :] = DIVIDER_COLOR
    grid[:, mid] = DIVIDER_COLOR

    for quad, color, count in zip(QUADRANTS, colors, counts, strict=True):
        quad_grid = _make_quadrant(qsize, int(color), int(count), area_profile)
        if quad_grid is None:
            return None
        rs, cs = _quadrant_slice(grid.shape, mid, mid, quad)
        grid[rs, cs] = quad_grid

    return grid if _validate_generated_grid(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
