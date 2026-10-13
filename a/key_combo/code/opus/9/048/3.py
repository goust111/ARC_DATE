from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

PALETTE = (BLUE, RED, YELLOW, PINK, ORANGE, TEAL, MAROON)
TEMPLATE_PALETTE = (BLUE, RED, YELLOW, PINK, ORANGE, TEAL, MAROON)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _component_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _transforms(mask: np.ndarray) -> list[np.ndarray]:
    arr = np.asarray(mask, dtype=bool)
    out: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(arr, k)
        out.append(rot)
        out.append(np.fliplr(rot))
    unique: list[np.ndarray] = []
    seen: set[tuple[bytes, tuple[int, int]]] = set()
    for item in out:
        key = (item.tobytes(), tuple(map(int, item.shape)))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    best: tuple[tuple[int, int], bytes] | None = None
    for item in _transforms(mask):
        key = (tuple(map(int, item.shape)), item.tobytes())
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def _reflect_cell(r: int, c: int, bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {
        (r, c),
        (r, c0 + c1 - c),
        (r0 + r1 - r, c),
        (r0 + r1 - r, c0 + c1 - c),
    }


def _orbit_family(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> frozenset[tuple[int, int]]:
    family: set[tuple[int, int]] = set()
    for r, c in coords:
        family |= _reflect_cell(int(r), int(c), bbox)
    return frozenset(family)


def _is_gray_outline_box(grid: np.ndarray, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    patch = _crop(grid, (r0, c0, r1, c1))
    if patch.shape[0] < 3 or patch.shape[1] < 3:
        return False
    gray_mask = patch == GRAY
    border = np.zeros_like(gray_mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(gray_mask, border))


def _extract_templates(grid: np.ndarray) -> list[dict]:
    gray_comps = _components(grid == GRAY)
    boxes = [coords for coords in gray_comps if _is_gray_outline_box(grid, coords)]
    assert len(boxes) == 4, "expected four gray reference boxes"

    templates: list[dict] = []
    for coords in boxes:
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        inner_colors = {int(v) for v in inner.flatten() if int(v) not in (BLACK, GRAY)}
        assert len(inner_colors) == 1, "each box should contain one monochrome sprite"
        color = inner_colors.pop()
        sprite_comps = _components(inner == color)
        assert len(sprite_comps) == 1, "each box interior should contain one connected sprite"
        mask = _component_mask_from_coords(sprite_comps[0])
        templates.append(
            {
                "bbox": (r0, c0, r1, c1),
                "color": color,
                "mask": mask,
                "shape_key": _canonical_key(mask),
            }
        )

    keys = [item["shape_key"] for item in templates]
    assert len(set(keys)) == len(keys), "reference sprite shapes must be unique up to rotation/flip"
    return templates


def _pattern_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != BLACK)
    assert coords.size > 0, "expected a central pattern"
    return _bbox_from_coords(coords)


def _check_lr_ud_symmetry(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    return bool(np.array_equal(patch, np.fliplr(patch)) and np.array_equal(patch, np.flipud(patch)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair green symmetric placeholder families by recoloring them from same-shaped boxed template sprites.

    Input structure:
    - Black background surrounds a central colored pattern and four gray hollow reference boxes in the corners.
    - The central pattern is left-right and top-bottom reflection-symmetric once green placeholder pixels are repaired.
    - Each gray box contains one monochrome sprite; green pixels in the central pattern appear as 4-connected placeholder clusters.

    Rule:
    - Detect the four gray outline boxes, extract each interior sprite's color and shape mask, and ignore those corner boxes in the final canvas.
    - Find each green cluster in the central pattern, group it with its horizontal/vertical reflection partners inside the pattern bbox, and compute the cluster's shape up to rotation/flip.
    - Replace every cell in that full reflection family with the color of the uniquely matching boxed sprite, then turn all gray boxes and boxed sprites into black background.

    Key insight:
    - Symmetry determines which damaged positions belong to one repair family, but the replacement color comes from external shape matching against the boxed templates rather than from any visible color inside the family itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    templates = _extract_templates(g)
    shape_to_color = {item["shape_key"]: int(item["color"]) for item in templates}

    work = g.copy()
    for item in templates:
        r0, c0, r1, c1 = item["bbox"]
        work[r0 : r1 + 1, c0 : c1 + 1] = BLACK

    pb = _pattern_bbox(work)
    green_comps = _components(work == GREEN)
    assert green_comps, "expected green placeholder clusters"

    out = work.copy()
    seen_families: set[frozenset[tuple[int, int]]] = set()
    filled = 0
    for comp in green_comps:
        family = _orbit_family(comp, pb)
        if family in seen_families:
            continue
        seen_families.add(family)

        shape = _component_mask_from_coords(comp)
        shape_key = _canonical_key(shape)
        assert shape_key in shape_to_color, "every green cluster must match one boxed template"
        color = shape_to_color[shape_key]

        for r, c in family:
            assert out[r, c] == GREEN, "entire reflection family should be green placeholders"
            out[r, c] = color
            filled += 1

    assert filled == int(np.count_nonzero(work == GREEN)), "all green cells must be repaired exactly once"
    assert not np.any(out == GREEN), "repair should remove all green placeholders"
    assert _check_lr_ud_symmetry(out, pb), "repaired pattern should satisfy two-axis reflection symmetry"
    return out


def _shape(mask_rows: list[str]) -> np.ndarray:
    arr = np.array([[ch == "1" for ch in row] for row in mask_rows], dtype=bool)
    coords = np.argwhere(arr)
    return _component_mask_from_coords(coords)


SHAPE_LIBRARY = [
    _shape(["11"]),
    _shape(["111"]),
    _shape(["10", "11"]),
    _shape(["11", "11"]),
    _shape(["111", "010"]),
    _shape(["10", "10", "11"]),
    _shape(["110", "011"]),
    _shape(["111", "101"]),
    _shape(["11", "11", "10"]),
]


def _shape_area(mask: np.ndarray) -> int:
    return int(mask.sum())


def _shape_is_reflection_symmetric(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)))


def _random_transform(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.rot90(mask, rng.randrange(4))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    return np.asarray(out, dtype=bool)


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = int(color)


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _local_family_coords(
    mask: np.ndarray,
    top: int,
    left: int,
    pattern_h: int,
    pattern_w: int,
) -> set[tuple[int, int]]:
    coords = set()
    for r, c in np.argwhere(mask):
        rr = top + int(r)
        cc = left + int(c)
        coords.add((rr, cc))
        coords.add((rr, pattern_w - 1 - cc))
        coords.add((pattern_h - 1 - rr, cc))
        coords.add((pattern_h - 1 - rr, pattern_w - 1 - cc))
    return coords


def _can_place_family(
    occupied: np.ndarray,
    family_coords: set[tuple[int, int]],
    pattern_h: int,
    pattern_w: int,
) -> bool:
    for r, c in family_coords:
        if not (0 <= r < pattern_h and 0 <= c < pattern_w):
            return False
        if r == 0 or r == pattern_h - 1 or c == 0 or c == pattern_w - 1:
            return False
    return all(not occupied[r, c] for r, c in family_coords)


def _place_family(
    grid: np.ndarray,
    occupied: np.ndarray,
    pattern_bbox: tuple[int, int, int, int],
    mask: np.ndarray,
    color: int,
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    r0, c0, r1, c1 = pattern_bbox
    pattern_h = r1 - r0 + 1
    pattern_w = c1 - c0 + 1
    h, w = mask.shape
    max_top = (pattern_h - 2 * h - 1) // 2
    max_left = (pattern_w - 2 * w - 1) // 2
    if max_top < 1 or max_left < 1:
        return None

    candidates = [(top, left) for top in range(1, max_top + 1) for left in range(1, max_left + 1)]
    rng.shuffle(candidates)
    for top, left in candidates:
        family_local = _local_family_coords(mask, top, left, pattern_h, pattern_w)
        if not _can_place_family(occupied, family_local, pattern_h, pattern_w):
            continue
        family_global = {(r0 + r, c0 + c) for r, c in family_local}
        for gr, gc in family_global:
            grid[gr, gc] = int(color)
        for lr, lc in family_local:
            occupied[lr, lc] = True
        return family_global
    return None


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _corner_positions(h: int, w: int, box_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    bh, bw = box_shape
    return 0, 0, h - bh, w - bw


def _draw_template_box(
    grid: np.ndarray,
    corner: str,
    mask: np.ndarray,
    color: int,
) -> tuple[int, int, int, int]:
    sh, sw = mask.shape
    bh, bw = sh + 2, sw + 2
    tl_r, tl_c, br_r, br_c = _corner_positions(grid.shape[0], grid.shape[1], (bh, bw))
    if corner == "tl":
        r0, c0 = tl_r, tl_c
    elif corner == "tr":
        r0, c0 = tl_r, br_c
    elif corner == "bl":
        r0, c0 = br_r, tl_c
    else:
        r0, c0 = br_r, br_c

    grid[r0 : r0 + bh, c0 : c0 + bw] = GRAY
    grid[r0 + 1 : r0 + bh - 1, c0 + 1 : c0 + bw - 1] = BLACK
    _paint_mask(grid, mask, r0 + 1, c0 + 1, color)
    return r0, c0, r0 + bh - 1, c0 + bw - 1


def _choose_template_shapes(rng: random.Random) -> list[np.ndarray]:
    indices = list(range(len(SHAPE_LIBRARY)))
    rng.shuffle(indices)
    chosen = [SHAPE_LIBRARY[i] for i in indices[:4]]
    keys = [_canonical_key(mask) for mask in chosen]
    if len(set(keys)) != 4:
        return []
    areas = [_shape_area(mask) for mask in chosen]
    if len(set(areas)) == 4:
        return []
    if not any(_shape_is_reflection_symmetric(mask) for mask in chosen):
        return []
    if not any(not _shape_is_reflection_symmetric(mask) for mask in chosen):
        return []
    return chosen


def _collect_green_families_after_box_removal(grid: np.ndarray) -> list[tuple[frozenset[tuple[int, int]], tuple[tuple[int, int], bytes]]]:
    templates = _extract_templates(grid)
    work = grid.copy()
    for item in templates:
        r0, c0, r1, c1 = item["bbox"]
        work[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    pb = _pattern_bbox(work)
    result = []
    seen: set[frozenset[tuple[int, int]]] = set()
    for comp in _components(work == GREEN):
        family = _orbit_family(comp, pb)
        if family in seen:
            continue
        seen.add(family)
        result.append((family, _canonical_key(_component_mask_from_coords(comp))))
    return result


def _validate_instance(inp: np.ndarray, out: np.ndarray) -> None:
    solved = main(inp)
    assert np.array_equal(solved, out), "solver output mismatch"

    templates = _extract_templates(inp)
    template_keys = {item["shape_key"]: item["color"] for item in templates}
    assert len(template_keys) == 4

    work = inp.copy()
    for item in templates:
        r0, c0, r1, c1 = item["bbox"]
        work[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    pb = _pattern_bbox(work)
    assert _check_lr_ud_symmetry(out, pb), "output must be two-axis symmetric"
    assert np.count_nonzero(inp == GREEN) > 0
    assert np.count_nonzero(out == GREEN) == 0
    assert np.count_nonzero(out == GRAY) == 0

    families = _collect_green_families_after_box_removal(inp)
    assert len(families) == 4, "expected four damaged symmetry families"
    seen_colors: set[int] = set()
    for family, shape_key in families:
        assert shape_key in template_keys, "each damaged family must match a template"
        color = template_keys[shape_key]
        seen_colors.add(int(color))
        for r, c in family:
            assert inp[r, c] == GREEN, "every family cell must be green in the input"
            assert out[r, c] == color, "every family cell must be recolored from its template"
    assert len(seen_colors) == 4


def _generate_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    template_shapes = _choose_template_shapes(rng)
    if not template_shapes:
        return None

    template_colors = list(rng.sample(TEMPLATE_PALETTE, 4))
    frame_color_candidates = [c for c in PALETTE if c not in template_colors]
    if not frame_color_candidates:
        return None
    frame_color = rng.choice(frame_color_candidates)

    grid_h = rng.randint(26, 30)
    grid_w = rng.randint(26, 30)
    pattern_h = rng.randint(15, 16)
    pattern_w = rng.randint(15, 16)
    r0 = (grid_h - pattern_h) // 2
    c0 = (grid_w - pattern_w) // 2
    pattern_bbox = (r0, c0, r0 + pattern_h - 1, c0 + pattern_w - 1)

    out = np.zeros((grid_h, grid_w), dtype=int)
    _draw_frame(out, pattern_bbox, frame_color)

    occupied = np.zeros((pattern_h, pattern_w), dtype=bool)
    occupied[0, :] = True
    occupied[-1, :] = True
    occupied[:, 0] = True
    occupied[:, -1] = True

    families: list[dict] = []
    order = list(range(4))
    order.sort(key=lambda i: int(template_shapes[i].sum()), reverse=True)
    for idx in order:
        displayed_shape = _random_transform(template_shapes[idx], rng)
        family_coords = _place_family(out, occupied, pattern_bbox, displayed_shape, template_colors[idx], rng)
        if family_coords is None:
            return None
        families.append(
            {
                "shape": template_shapes[idx],
                "displayed_shape": displayed_shape,
                "color": template_colors[idx],
                "coords": family_coords,
            }
        )

    inp = out.copy()
    for family in families:
        for r, c in family["coords"]:
            inp[r, c] = GREEN

    corners = ["tl", "tr", "bl", "br"]
    rng.shuffle(corners)
    for family, corner in zip(families, corners):
        box_mask = _random_transform(family["shape"], rng)
        _draw_template_box(inp, corner, box_mask, family["color"])

    if np.any(inp == GRAY) and np.any(inp == GREEN):
        try:
            _validate_instance(inp, out)
        except AssertionError:
            return None
        return inp, out
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        attempt = _generate_attempt(rng)
        if attempt is not None:
            return attempt[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
