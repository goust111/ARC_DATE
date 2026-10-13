import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

DIRS = ("up", "down", "left", "right")

SHAPE_LIBRARY = {
    "domino": np.array([[1, 1]], dtype=bool),
    "line3": np.array([[1, 1, 1]], dtype=bool),
    "l3": np.array([[1, 0], [1, 1]], dtype=bool),
    "line4": np.array([[1, 1, 1, 1]], dtype=bool),
    "square4": np.array([[1, 1], [1, 1]], dtype=bool),
    "l4": np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    "t4": np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    "s4": np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
}

MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    mask = grid == color
    seen = np.zeros_like(mask, dtype=bool)
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


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _tight_mask_from_coords(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return _trim_mask(mask)


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if v else "0" for v in row) for row in mask.astype(int))


def _canonical_signature(mask: np.ndarray) -> tuple[str, ...]:
    variants = []
    base = _trim_mask(mask.astype(bool))
    for k in range(4):
        rot = np.rot90(base, k)
        variants.append(_mask_key(_trim_mask(rot)))
        variants.append(_mask_key(_trim_mask(np.fliplr(rot))))
    return min(variants)


def _is_rect_frame(comp: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(comp)
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
        return False
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0 + 1, r1):
        expected.add((r, c0))
        expected.add((r, c1))
    return set(comp) == expected


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames = []
    for comp in _components_of_color(grid, GREY):
        if _is_rect_frame(comp):
            frames.append(_bbox(comp))
    frames.sort()
    assert frames, "expected at least one grey rectangular frame"
    return frames


def _classify_red_components(
    grid: np.ndarray, frames: list[tuple[int, int, int, int]]
) -> tuple[dict[int, list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    inside: dict[int, list[tuple[int, int]]] = {}
    outside: list[list[tuple[int, int]]] = []
    for comp in _components_of_color(grid, RED):
        owners = []
        for idx, (r0, c0, r1, c1) in enumerate(frames):
            if all(r0 < r < r1 and c0 < c < c1 for r, c in comp):
                owners.append(idx)
        if not owners:
            outside.append(comp)
            continue
        assert len(owners) == 1, "a red fragment cannot belong to multiple frames"
        owner = owners[0]
        assert owner not in inside, "each frame must contain exactly one red fragment"
        inside[owner] = comp
    assert len(inside) == len(frames), "every frame must contain one red fragment"
    return inside, outside


def _template_blue_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    comps = _components_of_color(grid, BLUE)
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        assert len(comp) == 3, "template arrows must use exactly three blue pixels"
        assert r1 - r0 + 1 <= 2 and c1 - c0 + 1 <= 2, "template arrows must be compact right-angle triominoes"
    return comps


def _component_distance(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    best = 10**9
    for ar, ac in a:
        for br, bc in b:
            best = min(best, abs(ar - br) + abs(ac - bc))
    return best


def _component_center(comp: list[tuple[int, int]]) -> tuple[float, float]:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return float(sum(rs)) / len(rs), float(sum(cs)) / len(cs)


def _pair_templates(
    red_templates: list[list[tuple[int, int]]], blue_templates: list[list[tuple[int, int]]]
) -> list[tuple[list[tuple[int, int]], list[tuple[int, int]]]]:
    assert len(red_templates) == len(blue_templates), "template reds and blues must come in pairs"
    scored = []
    for i, red_comp in enumerate(red_templates):
        for j, blue_comp in enumerate(blue_templates):
            scored.append((_component_distance(red_comp, blue_comp), i, j))
    scored.sort()

    assigned_red = set()
    assigned_blue = set()
    pairs = []
    for dist, i, j in scored:
        if i in assigned_red or j in assigned_blue:
            continue
        pairs.append((red_templates[i], blue_templates[j]))
        assigned_red.add(i)
        assigned_blue.add(j)
        if len(pairs) == len(red_templates):
            break

    assert len(pairs) == len(red_templates), "failed to recover all template pairs"
    assert scored[0][0] == 1, "paired templates should be immediately adjacent"
    return pairs


def _direction_from_template_pair(
    red_comp: list[tuple[int, int]], blue_comp: list[tuple[int, int]]
) -> str:
    rr, rc = _component_center(red_comp)
    br, bc = _component_center(blue_comp)
    if abs(br - rr) > abs(bc - rc):
        return "down" if br > rr else "up"
    return "right" if bc > rc else "left"


def _build_template_map(
    grid: np.ndarray, frames: list[tuple[int, int, int, int]]
) -> tuple[dict[int, list[tuple[int, int]]], dict[tuple[str, ...], str], set[tuple[int, int]]]:
    frame_fragments, red_templates = _classify_red_components(grid, frames)
    blue_templates = _template_blue_components(grid)
    paired = _pair_templates(red_templates, blue_templates)

    template_map: dict[tuple[str, ...], str] = {}
    template_cells: set[tuple[int, int]] = set()
    for red_comp, blue_comp in paired:
        sig = _canonical_signature(_tight_mask_from_coords(red_comp))
        assert sig not in template_map, "template shapes must be unique"
        template_map[sig] = _direction_from_template_pair(red_comp, blue_comp)
        template_cells.update(red_comp)
        template_cells.update(blue_comp)

    return frame_fragments, template_map, template_cells


def _fill_from_fragment(
    out: np.ndarray,
    frame: tuple[int, int, int, int],
    fragment: list[tuple[int, int]],
    direction: str,
) -> None:
    r0, c0, r1, c1 = frame
    frag_set = set(fragment)
    if direction == "right":
        for r in sorted({r for r, _ in frag_set}):
            edge = max(c for rr, c in frag_set if rr == r)
            for c in range(edge + 1, c1):
                if out[r, c] == BLACK:
                    out[r, c] = GREEN
    elif direction == "left":
        for r in sorted({r for r, _ in frag_set}):
            edge = min(c for rr, c in frag_set if rr == r)
            for c in range(edge - 1, c0, -1):
                if out[r, c] == BLACK:
                    out[r, c] = GREEN
    elif direction == "down":
        for c in sorted({c for _, c in frag_set}):
            edge = max(r for r, cc in frag_set if cc == c)
            for r in range(edge + 1, r1):
                if out[r, c] == BLACK:
                    out[r, c] = GREEN
    else:
        for c in sorted({c for _, c in frag_set}):
            edge = min(r for r, cc in frag_set if cc == c)
            for r in range(edge - 1, r0, -1):
                if out[r, c] == BLACK:
                    out[r, c] = GREEN


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each frame's inner red fragment to an external template, then fill the frame interior in that template's direction.

    Input structure:
    - The background is black.
    - Several grey one-pixel-thick hollow rectangular frames appear in the main area, and each frame contains one small red fragment inside its interior.
    - Near a grid edge there are template pairs made of a red fragment shape plus an adjacent blue three-pixel right-angle arrow mark.

    Rule:
    - Extract each red fragment shape as a tight binary mask and compare shapes up to rotation and reflection.
    - For every frame, find the template pair whose red fragment has the same normalized shape.
    - Read the matched template's direction from the adjacent blue arrow placement, then extend the frame fragment row-wise or column-wise in that direction: repaint interior black cells green until the grey frame boundary, while keeping the fragment itself red.
    - Clear every external template fragment and blue arrow marker to black in the output.

    Key insight:
    - The red fragment inside a frame does not directly encode its fill direction; it first indexes an external shape dictionary, and only the matched template provides the direction to use.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = _find_frames(g)
    frame_fragments, template_map, template_cells = _build_template_map(g, frames)

    for idx, frame in enumerate(frames):
        fragment = frame_fragments[idx]
        sig = _canonical_signature(_tight_mask_from_coords(fragment))
        assert sig in template_map, "every frame fragment must match a template shape"
        _fill_from_fragment(out, frame, fragment, template_map[sig])

    for r, c in template_cells:
        out[r, c] = BLACK
    return out


def _random_shape_variant(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = mask.copy()
    if rng.random() < 0.5:
        out = np.fliplr(out)
    out = np.rot90(out, rng.randrange(4))
    return out.copy()


def _arrow_mask(direction: str) -> np.ndarray:
    base = np.array([[1, 1], [1, 0]], dtype=bool)
    rotations = {"right": 0, "down": 1, "left": 2, "up": 3}
    return np.rot90(base, rotations[direction])


def _paint_mask(canvas: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    view = canvas[top : top + h, left : left + w]
    view[mask] = color


def _make_template_patch(shape_mask: np.ndarray, direction: str) -> np.ndarray:
    arrow = _arrow_mask(direction)
    sh, sw = shape_mask.shape
    if direction in ("left", "right"):
        ph = max(sh, 2)
        pw = sw + 2
        patch = np.zeros((ph, pw), dtype=int)
        sr = (ph - sh) // 2
        ar = (ph - 2) // 2
        if direction == "right":
            _paint_mask(patch, sr, 0, shape_mask, RED)
            _paint_mask(patch, ar, sw, arrow, BLUE)
        else:
            _paint_mask(patch, ar, 0, arrow, BLUE)
            _paint_mask(patch, sr, 2, shape_mask, RED)
    else:
        ph = sh + 2
        pw = max(sw, 2)
        patch = np.zeros((ph, pw), dtype=int)
        sc = (pw - sw) // 2
        ac = (pw - 2) // 2
        if direction == "down":
            _paint_mask(patch, 0, sc, shape_mask, RED)
            _paint_mask(patch, sh, ac, arrow, BLUE)
        else:
            _paint_mask(patch, 0, ac, arrow, BLUE)
            _paint_mask(patch, 2, sc, shape_mask, RED)
    return patch


def _make_frame_patch(shape_mask: np.ndarray, direction: str, rng: random.Random) -> np.ndarray:
    sh, sw = shape_mask.shape
    extra_h_min = 1 if direction in ("up", "down") else 0
    extra_w_min = 1 if direction in ("left", "right") else 0
    inner_h = max(3, sh + rng.randint(extra_h_min, 3))
    inner_w = max(3, sw + rng.randint(extra_w_min, 3))
    inner_h = min(inner_h, 7)
    inner_w = min(inner_w, 7)

    while direction in ("up", "down") and inner_h - sh < 1:
        inner_h += 1
    while direction in ("left", "right") and inner_w - sw < 1:
        inner_w += 1

    row_slack = inner_h - sh
    col_slack = inner_w - sw

    if direction == "up":
        top = rng.randint(1, row_slack)
        left = rng.randint(0, col_slack)
    elif direction == "down":
        top = rng.randint(0, row_slack - 1)
        left = rng.randint(0, col_slack)
    elif direction == "left":
        top = rng.randint(0, row_slack)
        left = rng.randint(1, col_slack)
    else:
        top = rng.randint(0, row_slack)
        left = rng.randint(0, col_slack - 1)

    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    _paint_mask(patch, 1 + top, 1 + left, shape_mask, RED)
    return patch


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _can_place(
    bbox: tuple[int, int, int, int], occupied: list[tuple[int, int, int, int]], pad: int = 1
) -> bool:
    expanded = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    for other in occupied:
        other_expanded = (other[0] - pad, other[1] - pad, other[2] + pad, other[3] + pad)
        if _intersects(expanded, other_expanded):
            return False
    return True


def _blit_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    view = grid[top : top + h, left : left + w]
    mask = patch != BLACK
    view[mask] = patch[mask]


def _place_in_zone(
    grid: np.ndarray,
    patch: np.ndarray,
    zone: tuple[int, int, int, int],
    occupied: list[tuple[int, int, int, int]],
    rng: random.Random,
    pad: int = 1,
) -> bool:
    zh, zw = zone[2] - zone[0] + 1, zone[3] - zone[1] + 1
    ph, pw = patch.shape
    if ph > zh or pw > zw:
        return False
    for _ in range(250):
        top = rng.randint(zone[0], zone[2] - ph + 1)
        left = rng.randint(zone[1], zone[3] - pw + 1)
        bbox = (top, left, top + ph - 1, left + pw - 1)
        if not _can_place(bbox, occupied, pad=pad):
            continue
        _blit_patch(grid, patch, top, left)
        occupied.append(bbox)
        return True
    return False


def _shape_signature_from_library(name: str) -> tuple[str, ...]:
    return _canonical_signature(SHAPE_LIBRARY[name])


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(24, 30)
    w = rng.randint(24, 30)
    grid = np.zeros((h, w), dtype=int)

    num_frames = rng.randint(2, 4)
    shape_names = list(SHAPE_LIBRARY)
    unique_used_count = rng.randint(2, min(4, num_frames, len(shape_names)))
    used_shapes = rng.sample(shape_names, unique_used_count)
    frame_shape_names = used_shapes.copy()
    while len(frame_shape_names) < num_frames:
        frame_shape_names.append(rng.choice(used_shapes))
    rng.shuffle(frame_shape_names)

    extra_templates = 0
    if len(shape_names) > len(used_shapes) and rng.random() < 0.6:
        extra_templates = 1
    remaining_shapes = [name for name in shape_names if name not in used_shapes]
    template_shape_names = used_shapes + rng.sample(remaining_shapes, extra_templates)

    direction_pool = list(DIRS)
    rng.shuffle(direction_pool)
    template_directions = {}
    for idx, shape_name in enumerate(template_shape_names):
        if idx < len(direction_pool):
            template_directions[shape_name] = direction_pool[idx]
        else:
            template_directions[shape_name] = rng.choice(DIRS)

    template_side = "top" if rng.random() < 0.5 else "bottom"
    template_zone_h = 8
    separator = 2
    if template_side == "top":
        template_zone = (0, 0, template_zone_h - 1, w - 1)
        frame_zone = (template_zone_h + separator, 0, h - 1, w - 1)
    else:
        template_zone = (h - template_zone_h, 0, h - 1, w - 1)
        frame_zone = (0, 0, h - template_zone_h - separator - 1, w - 1)
    if frame_zone[2] - frame_zone[0] + 1 < 10:
        return None

    occupied_templates: list[tuple[int, int, int, int]] = []
    for shape_name in template_shape_names:
        patch = _make_template_patch(
            _random_shape_variant(SHAPE_LIBRARY[shape_name], rng),
            template_directions[shape_name],
        )
        if not _place_in_zone(grid, patch, template_zone, occupied_templates, rng, pad=1):
            return None

    occupied_frames: list[tuple[int, int, int, int]] = []
    for shape_name in frame_shape_names:
        direction = template_directions[shape_name]
        patch = _make_frame_patch(_random_shape_variant(SHAPE_LIBRARY[shape_name], rng), direction, rng)
        if not _place_in_zone(grid, patch, frame_zone, occupied_frames, rng, pad=1):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.any(out == GREEN):
        return None
    if np.any(out == BLUE):
        return None

    frame_sigs = set()
    template_sigs = set()
    frames = _find_frames(grid)
    frame_fragments, red_templates = _classify_red_components(grid, frames)
    for comp in frame_fragments.values():
        frame_sigs.add(_canonical_signature(_tight_mask_from_coords(comp)))
    for comp in red_templates:
        template_sigs.add(_canonical_signature(_tight_mask_from_coords(comp)))
    if not frame_sigs <= template_sigs:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the frame-template direction task")
