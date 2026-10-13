from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None, "expected a non-empty mask"
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _crop_array(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None, "expected a non-empty mask"
    r0, c0, r1, c1 = box
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors(r: int, c: int, shape: tuple[int, int], connectivity: int) -> list[tuple[int, int]]:
    h, w = shape
    out: list[tuple[int, int]] = []
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dr, dc in deltas:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for nr, nc in _neighbors(cr, cc, mask.shape, connectivity):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        parts.append(comp)
    return parts


def _reflect_coords(coords: np.ndarray, center: tuple[int, int]) -> np.ndarray:
    cr, cc = center
    reflected = np.empty_like(coords)
    reflected[:, 0] = 2 * cr - coords[:, 0]
    reflected[:, 1] = 2 * cc - coords[:, 1]
    return reflected


def _score_candidate(observed: np.ndarray, center: tuple[int, int]) -> tuple[int, int, int] | None:
    coords = np.argwhere(observed)
    if coords.size == 0:
        return None
    reflected = _reflect_coords(coords, center)
    h, w = observed.shape
    if np.any(reflected[:, 0] < 0) or np.any(reflected[:, 0] >= h):
        return None
    if np.any(reflected[:, 1] < 0) or np.any(reflected[:, 1] >= w):
        return None

    full = observed.copy()
    full[reflected[:, 0], reflected[:, 1]] = True
    missing = full & ~observed
    if int(missing.sum()) < 2:
        return None
    if len(_components(full, connectivity=8)) != 1:
        return None
    if len(_components(missing, connectivity=8)) < 2:
        return None

    full_box = _bbox(full)
    missing_box = _bbox(missing)
    assert full_box is not None and missing_box is not None
    fr0, fc0, fr1, fc1 = full_box
    mr0, mc0, mr1, mc1 = missing_box
    full_area = (fr1 - fr0 + 1) * (fc1 - fc0 + 1)
    missing_area = (mr1 - mr0 + 1) * (mc1 - mc0 + 1)
    return int(missing.sum()), int(full_area), int(missing_area)


def _recover_center_and_missing(observed: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    h, w = observed.shape
    candidates: list[tuple[tuple[int, int, int], tuple[int, int], np.ndarray]] = []
    coords = np.argwhere(observed)
    assert coords.size > 0, "expected a non-empty large sprite"

    for cr in range(h):
        for cc in range(w):
            score = _score_candidate(observed, (cr, cc))
            if score is None:
                continue
            reflected = _reflect_coords(coords, (cr, cc))
            full = observed.copy()
            full[reflected[:, 0], reflected[:, 1]] = True
            missing = full & ~observed
            candidates.append((score, (cr, cc), missing))

    assert candidates, "no valid point-symmetry center found"
    best_score = min(score for score, _center, _missing in candidates)
    best = [(center, missing) for score, center, missing in candidates if score == best_score]
    assert len(best) == 1, "point-symmetry center must be uniquely recoverable"
    return best[0]


def _fragment_crops(grid: np.ndarray, bg: int, sprite_color: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
    fragments: list[tuple[int, np.ndarray, np.ndarray]] = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) not in {bg, sprite_color}):
        mask = grid == color
        crop = _crop_array(grid, mask)
        shape = crop == color
        fragments.append((color, crop, shape))
    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the missing pixels of a 180-degree symmetric sprite, aggregate them, and return the uniquely matching fragment.

    Input structure:
    - The grid has a dominant background, one large monochrome sprite, and several smaller monochrome fragments in other colors.
    - The large sprite is incomplete but comes from a shape that is 180-degree point-symmetric around one cell.
    - The missing pixels are split into several disconnected groups, while one fragment matches their total cropped layout and another is its mirror decoy.

    Rule:
    - Identify the large sprite and find the unique point-symmetry center that completes it by reflecting visible pixels.
    - Collect every missing pixel implied by that completion, then tight-crop the whole missing set as one pattern.
    - Compare that cropped pattern to each fragment using shape equality up to translation only, and return the uniquely matching fragment crop with its own color preserved.

    Key insight:
    - The target pattern is not visible directly: it must be reconstructed from 180-degree symmetry and combined across multiple missing regions before any fragment can be matched.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    color_counts = Counter(int(v) for v in g.flatten() if int(v) != bg)
    assert color_counts, "expected foreground content"
    sprite_color, _ = max(color_counts.items(), key=lambda item: item[1])

    observed = g == sprite_color
    _center, missing = _recover_center_and_missing(observed)
    target_shape = _crop_mask(missing)

    matches: list[np.ndarray] = []
    for _color, crop, shape in _fragment_crops(g, bg, sprite_color):
        if shape.shape == target_shape.shape and np.array_equal(shape, target_shape):
            matches.append(crop)

    assert len(matches) == 1, "expected exactly one matching fragment"
    return matches[0]


_MIN_SPRITE_SIZE = 11
_MAX_SPRITE_SIZE = 17
_MAX_ATTEMPTS = 600


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _normalize_shape(coords: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(
        (
            max(r for r, _c in coords) - min(r for r, _c in coords) + 1,
            max(c for _r, c in coords) - min(c for _r, c in coords) + 1,
        ),
        dtype=bool,
    )
    r0 = min(r for r, _c in coords)
    c0 = min(c for _r, c in coords)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _random_component_shape() -> list[tuple[int, int]]:
    options = [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1)],
    ]
    return random.choice(options)


def _generate_missing_pattern() -> np.ndarray | None:
    h = random.randint(3, 6)
    w = random.randint(4, 7)
    canvas = np.zeros((h, w), dtype=bool)
    placed: list[set[tuple[int, int]]] = []
    target_components = random.randint(2, 4)

    for _ in range(target_components):
        shape = _random_component_shape()
        sh = max(r for r, _c in shape) + 1
        sw = max(c for _r, c in shape) + 1
        anchors = [(r0, c0) for r0 in range(h - sh + 1) for c0 in range(w - sw + 1)]
        random.shuffle(anchors)
        for r0, c0 in anchors:
            cells = {(r0 + r, c0 + c) for r, c in shape}
            ok = True
            for pr, pc in cells:
                if canvas[pr, pc]:
                    ok = False
                    break
                for rr in range(pr - 1, pr + 2):
                    for cc in range(pc - 1, pc + 2):
                        if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells and canvas[rr, cc]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                for pr, pc in cells:
                    canvas[pr, pc] = True
                placed.append(cells)
                break
        else:
            return None

    cropped = _crop_mask(canvas)
    if len(_components(cropped, connectivity=8)) < 2:
        return None
    if int(cropped.sum()) < 4:
        return None
    if np.array_equal(cropped, np.fliplr(cropped)):
        return None
    if np.array_equal(cropped, np.flipud(cropped)):
        return None
    return cropped


def _embed_pattern(
    pattern: np.ndarray, sprite_h: int, sprite_w: int, quadrant: str, touch_boundary: bool
) -> set[tuple[int, int]] | None:
    ph, pw = pattern.shape
    cr = sprite_h // 2
    cc = sprite_w // 2

    if quadrant == "tl":
        row_values = [0] if touch_boundary else list(range(0, cr - ph + 1))
        col_values = [0] if touch_boundary else list(range(0, cc - pw + 1))
    elif quadrant == "tr":
        row_values = [0] if touch_boundary else list(range(0, cr - ph + 1))
        base = cc + 1
        col_values = [sprite_w - pw] if touch_boundary else list(range(base, sprite_w - pw + 1))
    elif quadrant == "bl":
        base = cr + 1
        row_values = [sprite_h - ph] if touch_boundary else list(range(base, sprite_h - ph + 1))
        col_values = [0] if touch_boundary else list(range(0, cc - pw + 1))
    else:
        row_values = [sprite_h - ph] if touch_boundary else list(range(cr + 1, sprite_h - ph + 1))
        col_values = [sprite_w - pw] if touch_boundary else list(range(cc + 1, sprite_w - pw + 1))

    if not row_values or not col_values:
        return None
    anchors = [(r0, c0) for r0 in row_values for c0 in col_values]
    random.shuffle(anchors)
    for r0, c0 in anchors:
        coords = {(r0 + r, c0 + c) for r, c in np.argwhere(pattern)}
        if any(r == cr or c == cc for r, c in coords):
            continue
        return coords
    return None


def _reflect_set(coords: set[tuple[int, int]], center: tuple[int, int]) -> set[tuple[int, int]]:
    cr, cc = center
    return {(2 * cr - r, 2 * cc - c) for r, c in coords}


def _path_cells(start: tuple[int, int], end: tuple[int, int], row_first: bool) -> list[tuple[int, int]]:
    r, c = start
    er, ec = end
    path = [(r, c)]
    if row_first:
        while r != er:
            r += _sign(er - r)
            path.append((r, c))
        while c != ec:
            c += _sign(ec - c)
            path.append((r, c))
    else:
        while c != ec:
            c += _sign(ec - c)
            path.append((r, c))
        while r != er:
            r += _sign(er - r)
            path.append((r, c))
    return path


def _choose_attachment(
    comp: list[tuple[int, int]],
    center: tuple[int, int],
    forbidden: set[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[int, int] | None:
    cr, cc = center
    ranked = sorted(comp, key=lambda p: abs(p[0] - cr) + abs(p[1] - cc))
    for r, c in ranked:
        candidates = []
        for nr, nc in _neighbors(r, c, shape, connectivity=4):
            if (nr, nc) in forbidden:
                continue
            if abs(nr - cr) + abs(nc - cc) < abs(r - cr) + abs(c - cc):
                candidates.append((nr, nc))
        if candidates:
            candidates.sort(key=lambda p: abs(p[0] - cr) + abs(p[1] - cc))
            return candidates[0]
    return None


def _build_full_sprite(
    missing_coords: set[tuple[int, int]], sprite_h: int, sprite_w: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
    center = (sprite_h // 2, sprite_w // 2)
    mirror_coords = _reflect_set(missing_coords, center)
    if missing_coords & mirror_coords:
        return None

    full = np.zeros((sprite_h, sprite_w), dtype=bool)
    cr, cc = center
    core = {(cr, cc), (cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)}
    for r, c in core:
        if not (0 <= r < sprite_h and 0 <= c < sprite_w):
            return None
        full[r, c] = True

    forbidden = set(missing_coords) | mirror_coords
    for comp in _components(_mask_from_coords(missing_coords, (sprite_h, sprite_w)), connectivity=8):
        attach = _choose_attachment(comp, center, forbidden, (sprite_h, sprite_w))
        if attach is None:
            return None
        valid_path = None
        for row_first in [random.choice([True, False]), True, False]:
            path = _path_cells(center, attach, row_first=row_first)
            mirrored = _reflect_set(set(path), center)
            if set(path) & forbidden:
                continue
            if mirrored & forbidden:
                continue
            valid_path = path
            break
        if valid_path is None:
            return None
        for r, c in valid_path:
            full[r, c] = True
            full[2 * cr - r, 2 * cc - c] = True

    growth_steps = random.randint(3, 8)
    for _ in range(growth_steps):
        frontier: list[tuple[int, int]] = []
        for r, c in np.argwhere(full):
            rr = int(r)
            cc2 = int(c)
            for nr, nc in _neighbors(rr, cc2, full.shape, connectivity=4):
                mr, mc = 2 * cr - nr, 2 * cc - nc
                if not (0 <= mr < sprite_h and 0 <= mc < sprite_w):
                    continue
                if (nr, nc) in forbidden or (mr, mc) in forbidden:
                    continue
                if full[nr, nc] and full[mr, mc]:
                    continue
                frontier.append((nr, nc))
        if not frontier:
            break
        nr, nc = random.choice(frontier)
        mr, mc = 2 * cr - nr, 2 * cc - nc
        full[nr, nc] = True
        full[mr, mc] = True

    for r, c in missing_coords | mirror_coords:
        full[r, c] = True

    observed = full.copy()
    for r, c in missing_coords:
        observed[r, c] = False

    if len(_components(full, connectivity=8)) != 1:
        return None
    if len(_components(observed, connectivity=8)) != 1:
        return None
    if np.array_equal(full, np.fliplr(full)) or np.array_equal(full, np.flipud(full)):
        return None
    return full, observed, center


def _mask_from_coords(coords: set[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _random_distractor(base_shape: np.ndarray, forbidden: list[np.ndarray]) -> np.ndarray | None:
    h = random.randint(max(2, base_shape.shape[0] - 1), min(7, base_shape.shape[0] + 1))
    w = random.randint(max(2, base_shape.shape[1] - 1), min(8, base_shape.shape[1] + 1))
    canvas = np.zeros((h, w), dtype=bool)
    pieces = random.randint(2, 4)
    for _ in range(pieces):
        shape = _random_component_shape()
        sh = max(r for r, _c in shape) + 1
        sw = max(c for _r, c in shape) + 1
        anchors = [(r0, c0) for r0 in range(h - sh + 1) for c0 in range(w - sw + 1)]
        random.shuffle(anchors)
        for r0, c0 in anchors:
            cells = {(r0 + r, c0 + c) for r, c in shape}
            ok = True
            for pr, pc in cells:
                if canvas[pr, pc]:
                    ok = False
                    break
            if ok:
                for pr, pc in cells:
                    canvas[pr, pc] = True
                break
        else:
            return None
    cropped = _crop_mask(canvas)
    if int(cropped.sum()) < 3:
        return None
    if len(_components(cropped, connectivity=8)) < 2:
        return None
    for bad in forbidden:
        if cropped.shape == bad.shape and np.array_equal(cropped, bad):
            return None
    return cropped


def _place_pattern(
    grid: np.ndarray, pattern: np.ndarray, color: int, used_boxes: list[tuple[int, int, int, int]], padding: int = 1
) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = pattern.shape
    anchors = [(r0, c0) for r0 in range(h - ph + 1) for c0 in range(w - pw + 1)]
    random.shuffle(anchors)
    for r0, c0 in anchors:
        box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        separated = True
        for ur0, uc0, ur1, uc1 in used_boxes:
            if not (
                box[2] + padding < ur0
                or ur1 + padding < box[0]
                or box[3] + padding < uc0
                or uc1 + padding < box[1]
            ):
                separated = False
                break
        if not separated:
            continue
        region = grid[r0 : r0 + ph, c0 : c0 + pw]
        if np.any(region[pattern] != 0):
            continue
        region[pattern] = color
        used_boxes.append(box)
        return r0, c0
    return None


def _render_scene(
    observed: np.ndarray,
    sprite_color: int,
    fragment_specs: list[tuple[int, np.ndarray]],
) -> tuple[np.ndarray, tuple[int, int]] | None:
    oh, ow = observed.shape
    min_h = max(oh + 4, 18)
    min_w = max(ow + 8, 18)
    max_h = min(30, max(min_h, oh + 12))
    max_w = min(30, max(min_w, ow + 12))
    if min_h > max_h or min_w > max_w:
        return None
    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)
    grid = np.zeros((h, w), dtype=int)

    sprite_r = random.randint(1, h - oh - 1)
    sprite_c = random.randint(1, w - ow - 1)
    grid[sprite_r : sprite_r + oh, sprite_c : sprite_c + ow][observed] = sprite_color
    used_boxes = [(sprite_r, sprite_c, sprite_r + oh - 1, sprite_c + ow - 1)]

    for color, pattern in fragment_specs:
        if _place_pattern(grid, pattern, color, used_boxes, padding=1) is None:
            return None
    return grid, (sprite_r, sprite_c)


def _verify_scene(
    grid: np.ndarray,
    sprite_color: int,
    local_center: tuple[int, int],
    sprite_offset: tuple[int, int],
    local_missing: np.ndarray,
    target_color: int,
    mirror_shape: np.ndarray,
) -> bool:
    sprite_r, sprite_c = sprite_offset
    center = (local_center[0] + sprite_r, local_center[1] + sprite_c)
    observed = grid == sprite_color
    recovered_center, recovered_missing = _recover_center_and_missing(observed)
    if recovered_center != center:
        return False

    expected_missing = np.zeros_like(grid, dtype=bool)
    mh, mw = local_missing.shape
    expected_missing[sprite_r : sprite_r + mh, sprite_c : sprite_c + mw] = local_missing
    if not np.array_equal(recovered_missing, expected_missing):
        return False

    target_shape = _crop_mask(expected_missing)
    if np.array_equal(target_shape, mirror_shape):
        return False

    bg = _get_bg(grid)
    fragments = _fragment_crops(grid, bg, sprite_color)
    matches = [color for color, _crop, shape in fragments if shape.shape == target_shape.shape and np.array_equal(shape, target_shape)]
    mirrors = [color for color, _crop, shape in fragments if shape.shape == mirror_shape.shape and np.array_equal(shape, mirror_shape)]
    if matches != [target_color]:
        return False
    if len(mirrors) != 1 or mirrors[0] == target_color:
        return False

    if len(_components(target_shape, connectivity=8)) < 2:
        return False

    result = main(grid)
    target_crop = next(crop for color, crop, _shape in fragments if color == target_color)
    return np.array_equal(result, target_crop)


def _gen_attempt() -> np.ndarray | None:
    pattern = _generate_missing_pattern()
    if pattern is None:
        return None

    mirror_axis = random.choice(["lr", "ud"])
    mirror_shape = np.fliplr(pattern) if mirror_axis == "lr" else np.flipud(pattern)
    if np.array_equal(pattern, mirror_shape):
        return None

    sprite_h = random.randrange(_MIN_SPRITE_SIZE, _MAX_SPRITE_SIZE + 1, 2)
    sprite_w = random.randrange(_MIN_SPRITE_SIZE, _MAX_SPRITE_SIZE + 1, 2)
    quadrant = random.choice(["tl", "tr", "bl", "br"])
    touch_boundary = random.random() < 0.55
    missing_coords = _embed_pattern(pattern, sprite_h, sprite_w, quadrant, touch_boundary)
    if missing_coords is None:
        return None

    built = _build_full_sprite(missing_coords, sprite_h, sprite_w)
    if built is None:
        return None
    _full, observed, center = built

    observed_box = _bbox(observed)
    if observed_box is None:
        return None
    if touch_boundary:
        r0, c0, r1, c1 = observed_box
        bbox_center = ((r0 + r1) // 2, (c0 + c1) // 2)
        if bbox_center == center:
            return None

    sprite_color = random.choice(list(range(1, 10)))
    remaining_colors = [color for color in range(1, 10) if color != sprite_color]
    random.shuffle(remaining_colors)

    target_color = remaining_colors.pop()
    mirror_color = remaining_colors.pop()

    forbidden = [pattern, mirror_shape]
    distractor_count = random.randint(1, min(3, len(remaining_colors)))
    fragments: list[tuple[int, np.ndarray]] = [(target_color, pattern.copy()), (mirror_color, mirror_shape.copy())]
    for _ in range(distractor_count):
        distractor = _random_distractor(pattern, forbidden)
        if distractor is None:
            return None
        forbidden.append(distractor)
        fragments.append((remaining_colors.pop(), distractor))
    random.shuffle(fragments)

    rendered = _render_scene(observed, sprite_color, fragments)
    if rendered is None:
        return None
    grid, sprite_offset = rendered

    expected_missing = np.zeros(observed.shape, dtype=bool)
    for r, c in missing_coords:
        expected_missing[r, c] = True

    if not _verify_scene(grid, sprite_color, center, sprite_offset, expected_missing, target_color, mirror_shape):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 054/2")
