from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
COLORS = list(range(1, 10))
MAX_GEN_ATTEMPTS = 700


def _background(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(counts.most_common(1)[0][0])


def _bbox(coords: list[Coord] | tuple[Coord, ...]) -> BBox:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normal_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return out


def _lattice_components(grid: np.ndarray, bg: int, template_color: int) -> list[dict]:
    out: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg), int(template_color)}):
        pts = {(int(r), int(c)) for r, c in np.argwhere(grid == color)}
        while pts:
            seed = next(iter(pts))
            pts.remove(seed)
            coords = [seed]
            q = deque([seed])
            while q:
                r, c = q.popleft()
                for nr, nc in ((r - 2, c), (r + 2, c), (r, c - 2), (r, c + 2)):
                    if (nr, nc) in pts:
                        pts.remove((nr, nc))
                        coords.append((nr, nc))
                        q.append((nr, nc))
            out.append({"color": int(color), "coords": coords, "bbox": _bbox(coords)})
    return out


def _dense_shape_key(coords: list[Coord] | tuple[Coord, ...]) -> tuple[Coord, ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _sparse_contracted_key(coords: list[Coord] | tuple[Coord, ...]) -> tuple[Coord, ...]:
    r0, c0, _, _ = _bbox(coords)
    compact: list[Coord] = []
    for r, c in coords:
        dr = r - r0
        dc = c - c0
        assert dr % 2 == 0 and dc % 2 == 0, "lattice pixels must share a 2-step grid"
        compact.append((dr // 2, dc // 2))
    return tuple(sorted(compact))


def _shape_size(shape: tuple[Coord, ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the dense template with the color of the sparse lattice figure that contracts to the same shape.

    Input structure:
    - The background is the dominant empty color.
    - Exactly one non-background object is a small dense 4-connected template in a single color.
    - Several other monochrome figures are sparse lattice components whose same-color pixels touch only by two-cell cardinal steps.
    - Distractors can have the same point count as the template or a similar lattice bounding box.

    Rule:
    - Find the dense template as the unique ordinary 4-connected foreground component with more than one pixel.
    - For each sparse same-color component, divide its row and column offsets by two and normalize away translation.
    - Exactly one contracted sparse shape equals the normalized template shape without rotation.
    - Return the original grid with only the template pixels recolored to that matching sparse figure's color.

    Key insight:
    - The scattered points are not noise or paths; they are a two-times enlarged trace of a dense shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _background(g)

    dense_components = _normal_components(g, bg)
    template_candidates = [comp for comp in dense_components if len(comp["coords"]) > 1]
    assert len(template_candidates) == 1, "expected exactly one dense template component"

    template = template_candidates[0]
    template_color = int(template["color"])
    assert int(np.count_nonzero(g == template_color)) == len(template["coords"]), (
        "template color must be reserved for the dense template"
    )
    template_key = _dense_shape_key(template["coords"])

    matches = []
    for comp in _lattice_components(g, bg, template_color):
        contracted = _sparse_contracted_key(comp["coords"])
        if contracted == template_key:
            matches.append(comp)

    assert len(matches) == 1, "expected a unique sparse lattice match"
    match_color = int(matches[0]["color"])
    assert match_color != template_color

    out = g.copy()
    for r, c in template["coords"]:
        out[r, c] = match_color
    return out


def _normalize_shape(cells: set[Coord] | list[Coord] | tuple[Coord, ...]) -> tuple[Coord, ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _shape_connected(shape: tuple[Coord, ...]) -> bool:
    cells = set(shape)
    seen = {shape[0]}
    q = deque([shape[0]])
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _shape_has_hole(shape: tuple[Coord, ...]) -> bool:
    h, w = _shape_size(shape)
    cells = set(shape)
    empty = {(r, c) for r in range(h) for c in range(w) if (r, c) not in cells}
    if not empty:
        return False

    outside: set[Coord] = set()
    q: deque[Coord] = deque()
    for r, c in empty:
        if r in (0, h - 1) or c in (0, w - 1):
            outside.add((r, c))
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in empty and (nr, nc) not in outside:
                outside.add((nr, nc))
                q.append((nr, nc))
    return len(outside) != len(empty)


def _shape_to_mask(shape: tuple[Coord, ...]) -> np.ndarray:
    h, w = _shape_size(shape)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _mask_to_shape(mask: np.ndarray) -> tuple[Coord, ...]:
    return _normalize_shape([(int(r), int(c)) for r, c in np.argwhere(mask)])


def _rotate_shape(shape: tuple[Coord, ...], k: int) -> tuple[Coord, ...]:
    return _mask_to_shape(np.rot90(_shape_to_mask(shape), k))


def _valid_shape(shape: tuple[Coord, ...], *, min_h: int = 1, min_w: int = 1) -> bool:
    h, w = _shape_size(shape)
    return (
        h >= min_h
        and w >= min_w
        and _shape_connected(shape)
        and not _shape_has_hole(shape)
    )


def _random_connected_shape(
    *,
    area: int | None = None,
    area_range: tuple[int, int] = (4, 8),
    max_h: int = 5,
    max_w: int = 5,
    min_h: int = 2,
    min_w: int = 2,
) -> tuple[Coord, ...] | None:
    for _ in range(300):
        target = int(area if area is not None else random.randint(*area_range))
        cells: set[Coord] = {(0, 0)}
        while len(cells) < target:
            boundary = []
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (nr, nc) not in cells:
                        boundary.append((nr, nc))
            cells.add(random.choice(boundary))
        shape = _normalize_shape(cells)
        h, w = _shape_size(shape)
        if h <= max_h and w <= max_w and _valid_shape(shape, min_h=min_h, min_w=min_w):
            return shape
    return None


def _spanning_shape(height: int, width: int, area: int) -> tuple[Coord, ...] | None:
    min_area = height + width - 1
    if area < min_area or area > height * width:
        return None

    for _ in range(200):
        cells: set[Coord] = {(0, 0)}
        r = c = 0
        steps = [(1, 0)] * (height - 1) + [(0, 1)] * (width - 1)
        random.shuffle(steps)
        for dr, dc in steps:
            r += dr
            c += dc
            cells.add((r, c))

        while len(cells) < area:
            candidates = []
            for rr, cc in cells:
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        candidates.append((nr, nc))
            if not candidates:
                break
            cells.add(random.choice(candidates))

        if len(cells) != area:
            continue
        shape = _normalize_shape(cells)
        if _shape_size(shape) == (height, width) and _valid_shape(shape):
            return shape
    return None


def _same_count_decoy(target: tuple[Coord, ...]) -> tuple[Coord, ...] | None:
    rotations = [_rotate_shape(target, k) for k in (1, 2, 3)]
    rotations = [shape for shape in rotations if shape != target and _valid_shape(shape, min_h=2, min_w=2)]
    if rotations and random.random() < 0.75:
        return random.choice(rotations)

    area = len(target)
    for _ in range(250):
        shape = _random_connected_shape(area=area, max_h=5, max_w=5, min_h=2, min_w=2)
        if shape is not None and shape != target:
            return shape
    return None


def _bbox_decoy(target: tuple[Coord, ...]) -> tuple[Coord, ...] | None:
    h, w = _shape_size(target)
    min_area = h + w - 1
    possible = [area for area in range(min_area, h * w + 1) if area != len(target)]
    random.shuffle(possible)
    for area in possible:
        for _ in range(80):
            shape = _spanning_shape(h, w, area)
            if shape is not None and shape != target:
                return shape
    return None


def _dense_patch(shape: tuple[Coord, ...], color: int) -> np.ndarray:
    h, w = _shape_size(shape)
    patch = np.full((h, w), BG, dtype=int)
    for r, c in shape:
        patch[r, c] = int(color)
    return patch


def _sparse_patch(shape: tuple[Coord, ...], color: int) -> np.ndarray:
    h, w = _shape_size(shape)
    patch = np.full((2 * h - 1, 2 * w - 1), BG, dtype=int)
    for r, c in shape:
        patch[2 * r, 2 * c] = int(color)
    return patch


def _rects_overlap(a: BBox, b: BBox) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _expanded_rect(r: int, c: int, h: int, w: int, pad: int, shape: tuple[int, int]) -> BBox:
    rows, cols = shape
    return (
        max(0, r - pad),
        max(0, c - pad),
        min(rows - 1, r + h - 1 + pad),
        min(cols - 1, c + w - 1 + pad),
    )


def _place_patch(
    grid: np.ndarray,
    reserved: list[BBox],
    patch: np.ndarray,
    color: int,
    *,
    pad: int = 1,
) -> tuple[int, int] | None:
    h, w = patch.shape
    rows, cols = grid.shape
    candidates = [
        (r, c)
        for r in range(1, rows - h)
        for c in range(1, cols - w)
    ]
    random.shuffle(candidates)
    for r, c in candidates:
        box = _expanded_rect(r, c, h, w, pad, grid.shape)
        if any(_rects_overlap(box, old) for old in reserved):
            continue
        mask = patch != BG
        if np.any(grid[r : r + h, c : c + w][mask] != BG):
            continue
        grid[r : r + h, c : c + w][mask] = int(color)
        reserved.append(box)
        return int(r), int(c)
    return None


def _build_objects(target_shape: tuple[Coord, ...]) -> list[dict] | None:
    same_count = _same_count_decoy(target_shape)
    bbox_like = _bbox_decoy(target_shape)
    if same_count is None or bbox_like is None:
        return None

    shapes = [target_shape, same_count, bbox_like]
    extra_count = random.randint(0, 2)
    for _ in range(extra_count):
        for _attempt in range(120):
            shape = _random_connected_shape(area_range=(3, 8), max_h=5, max_w=5, min_h=1, min_w=1)
            if shape is not None and shape != target_shape:
                shapes.append(shape)
                break

    colors = random.sample(COLORS, len(shapes) + 1)
    template_color = colors[0]
    lattice_colors = colors[1:]

    objects = [{"kind": "template", "shape": target_shape, "color": template_color}]
    lattice = [
        {"kind": "target", "shape": target_shape, "color": lattice_colors[0]},
        {"kind": "same_count_decoy", "shape": same_count, "color": lattice_colors[1]},
        {"kind": "bbox_decoy", "shape": bbox_like, "color": lattice_colors[2]},
    ]
    for shape, color in zip(shapes[3:], lattice_colors[3:]):
        lattice.append({"kind": "extra_decoy", "shape": shape, "color": color})
    random.shuffle(lattice)
    return objects + lattice


def _semantic_check(inp: np.ndarray, expected: np.ndarray) -> bool:
    bg = _background(inp)
    dense = _normal_components(inp, bg)
    templates = [comp for comp in dense if len(comp["coords"]) > 1]
    if len(templates) != 1:
        return False
    template = templates[0]
    template_color = int(template["color"])
    template_key = _dense_shape_key(template["coords"])
    template_h, template_w = _shape_size(template_key)

    groups = _lattice_components(inp, bg, template_color)
    if len(groups) < 3:
        return False

    contracted = [(comp, _sparse_contracted_key(comp["coords"])) for comp in groups]
    matches = [comp for comp, key in contracted if key == template_key]
    if len(matches) != 1:
        return False

    same_count_decoys = [
        key for comp, key in contracted if len(key) == len(template_key) and key != template_key
    ]
    if not same_count_decoys:
        return False

    bbox_decoys = [
        key
        for comp, key in contracted
        if key != template_key and _shape_size(key) == (template_h, template_w)
    ]
    if not bbox_decoys:
        return False

    changed = np.argwhere(inp != expected)
    if changed.shape[0] != len(template["coords"]):
        return False
    template_cells = set(template["coords"])
    if {(int(r), int(c)) for r, c in changed} != template_cells:
        return False
    if len({int(expected[r, c]) for r, c in template_cells}) != 1:
        return False
    if int(expected[template["coords"][0][0], template["coords"][0][1]]) != int(matches[0]["color"]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    target = _random_connected_shape(area_range=(4, 8), max_h=5, max_w=5, min_h=2, min_w=2)
    if target is None:
        return None
    if all(_rotate_shape(target, k) == target for k in (1, 2, 3)):
        return None

    objects = _build_objects(target)
    if objects is None:
        return None

    rows = random.randint(20, 30)
    cols = random.randint(20, 30)
    grid = np.full((rows, cols), BG, dtype=int)
    reserved: list[BBox] = []
    placements: dict[str, tuple[int, int]] = {}

    draw_order = objects[:]
    random.shuffle(draw_order)
    for obj in draw_order:
        patch = (
            _dense_patch(obj["shape"], obj["color"])
            if obj["kind"] == "template"
            else _sparse_patch(obj["shape"], obj["color"])
        )
        pos = _place_patch(grid, reserved, patch, obj["color"], pad=1)
        if pos is None:
            return None
        placements[obj["kind"]] = pos

    if "template" not in placements:
        return None

    target_color = next(int(obj["color"]) for obj in objects if obj["kind"] == "target")
    expected = grid.copy()
    template_obj = next(obj for obj in objects if obj["kind"] == "template")
    template_top, template_left = placements["template"]
    for r, c in template_obj["shape"]:
        expected[template_top + r, template_left + c] = target_color

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _semantic_check(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid sparse-lattice template input")
