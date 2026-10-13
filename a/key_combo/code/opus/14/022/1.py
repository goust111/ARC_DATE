from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
BOX_SIZE = 5
INTERIOR_RANGE = range(1, 4)
CORNERS = ("TL", "TR", "BR", "BL")
CORNER_INDEX = {corner: i for i, corner in enumerate(CORNERS)}
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NONZERO_COLORS = tuple(range(1, 10))
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            objects.append(
                {
                    "color": color,
                    "coords": tuple(sorted(coords)),
                    "bbox": _bbox_from_coords(coords),
                    "area": len(coords),
                }
            )
    return objects


def _bbox_corner(bbox: tuple[int, int, int, int], height: int, width: int) -> str | None:
    r0, c0, r1, c1 = bbox
    corners = []
    if r0 == 0 and c0 == 0:
        corners.append("TL")
    if r0 == 0 and c1 == width - 1:
        corners.append("TR")
    if r1 == height - 1 and c1 == width - 1:
        corners.append("BR")
    if r1 == height - 1 and c0 == 0:
        corners.append("BL")
    assert len(corners) <= 1, "a component bbox must not satisfy multiple corners"
    return corners[0] if corners else None


def _corner_anchor(corner: str, height: int, width: int, obj_h: int, obj_w: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, width - obj_w
    if corner == "BR":
        return height - obj_h, width - obj_w
    if corner == "BL":
        return height - obj_h, 0
    raise AssertionError(corner)


def _shift_corner(corner: str, steps: int) -> str:
    return CORNERS[(CORNER_INDEX[corner] + int(steps)) % 4]


def _corner_objects(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = _most_common_color(grid)
    height, width = grid.shape
    objects = _find_components(grid, bg=bg)
    corner_objects = []
    for obj in objects:
        corner = _bbox_corner(obj["bbox"], height, width)
        if corner is None:
            continue
        obj = dict(obj)
        obj["corner"] = corner
        corner_objects.append(obj)
    return bg, corner_objects


def main(grid: np.ndarray) -> np.ndarray:
    """Move each true corner object to the clockwise corner encoded by its bbox hole count.

    Input structure:
    - The grid has black background and several monochromatic 4-connected foreground components.
    - Relevant components are exactly those whose bounding boxes touch two adjacent grid borders at one true corner.
    - Corner components use their own bounding boxes as reference frames; all other components are distractors and may touch at most one border.

    Rule:
    - Find every corner component and count the number of background pixels inside its bounding box.
    - Treat that count h as a clockwise corner offset: h=1 means the next corner, h=2 the opposite corner, h=3 the previous corner.
    - Clear each source component from its original location, then stamp the same pixel pattern and color into the destination corner by aligning the bounding box to that corner.
    - Leave all non-corner distractors and untouched background cells unchanged.

    Key insight:
    - The corner object is not chosen by color or position alone: first isolate true corner-anchored components, then use the black pixels inside each component's own bbox as the code for how far it must travel clockwise.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    bg, corner_objects = _corner_objects(g)
    assert corner_objects, "expected at least one corner object"

    source_cells = {cell for obj in corner_objects for cell in obj["coords"]}
    out = g.copy()
    for r, c in source_cells:
        out[r, c] = bg

    planned: dict[tuple[int, int], int] = {}
    for obj in corner_objects:
        r0, c0, r1, c1 = obj["bbox"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        holes = int(np.sum(patch == bg))
        assert 1 <= holes <= 3, "corner hole count must encode one of three clockwise offsets"
        dst_corner = _shift_corner(obj["corner"], holes)
        dst_r0, dst_c0 = _corner_anchor(dst_corner, height, width, patch.shape[0], patch.shape[1])

        for r, c in obj["coords"]:
            nr = dst_r0 + (r - r0)
            nc = dst_c0 + (c - c0)
            assert 0 <= nr < height and 0 <= nc < width
            if int(g[nr, nc]) != bg and (nr, nc) not in source_cells:
                raise AssertionError("corner move would overwrite a distractor")
            previous = planned.get((nr, nc))
            color = int(obj["color"])
            assert previous is None, "destination corner moves must remain disjoint"
            planned[(nr, nc)] = color

    for (r, c), color in planned.items():
        out[r, c] = color
    return out


_HOLE_CELLS = tuple((r, c) for r in INTERIOR_RANGE for c in INTERIOR_RANGE)
_HOLE_PATTERNS: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {}
for hole_count in (1, 2, 3):
    patterns = []
    for combo in itertools.combinations(_HOLE_CELLS, hole_count):
        if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in itertools.combinations(combo, 2)):
            continue
        patterns.append(combo)
    _HOLE_PATTERNS[hole_count] = tuple(patterns)


def _make_corner_patch(color: int, holes: int) -> np.ndarray:
    patch = np.full((BOX_SIZE, BOX_SIZE), int(color), dtype=int)
    for r, c in random.choice(_HOLE_PATTERNS[int(holes)]):
        patch[r, c] = BG
    return patch


def _stamp_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    mask = patch != BG
    window = grid[r0 : r0 + h, c0 : c0 + w]
    window[mask] = patch[mask]


def _box_cells(r0: int, c0: int, h: int, w: int, *, pad: int, height: int, width: int) -> set[tuple[int, int]]:
    cells = set()
    for r in range(max(0, r0 - pad), min(height, r0 + h + pad)):
        for c in range(max(0, c0 - pad), min(width, c0 + w + pad)):
            cells.add((r, c))
    return cells


def _grow_blob(h: int, w: int, cells: int) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    seed = (random.randrange(h), random.randrange(w))
    mask[seed] = True
    frontier = {seed}
    while int(mask.sum()) < cells:
        expand = set()
        for r, c in frontier:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    expand.add((nr, nc))
        if not expand:
            return None
        pick = random.choice(tuple(expand))
        mask[pick] = True
        frontier.add(pick)
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _make_distractor_patch(color: int) -> np.ndarray:
    for _ in range(80):
        h = random.randint(1, 4)
        w = random.randint(1, 4)
        max_area = min(6, h * w)
        area = 1 if max_area == 1 else random.randint(2, max_area)
        mask = _grow_blob(h, w, area)
        if mask is None:
            continue
        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = int(color)
        return patch
    raise RuntimeError("failed to build distractor patch")


def _touch_count(r0: int, c0: int, h: int, w: int, height: int, width: int) -> int:
    touches = 0
    touches += int(r0 == 0)
    touches += int(c0 == 0)
    touches += int(r0 + h == height)
    touches += int(c0 + w == width)
    return touches


def _can_place_distractor(
    patch: np.ndarray,
    r0: int,
    c0: int,
    height: int,
    width: int,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    h, w = patch.shape
    for rr, cc in np.argwhere(patch != BG):
        r = r0 + int(rr)
        c = c0 + int(cc)
        if not (0 <= r < height and 0 <= c < width):
            return False
        if (r, c) in occupied or (r, c) in reserved:
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return False
    return True


def _place_distractor(
    grid: np.ndarray,
    patch: np.ndarray,
    *,
    side_touch: bool,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    height, width = grid.shape
    h, w = patch.shape
    candidates = []
    for r0 in range(height - h + 1):
        for c0 in range(width - w + 1):
            touches = _touch_count(r0, c0, h, w, height, width)
            if side_touch and touches != 1:
                continue
            if not side_touch and touches != 0:
                continue
            if _can_place_distractor(patch, r0, c0, height, width, occupied, reserved):
                candidates.append((r0, c0))
    if not candidates:
        return False
    r0, c0 = random.choice(candidates)
    _stamp_patch(grid, patch, r0, c0)
    for rr, cc in np.argwhere(patch != BG):
        occupied.add((r0 + int(rr), c0 + int(cc)))
    return True


def _plan_corner_assignment(corners: tuple[str, ...]) -> dict[str, int] | None:
    counts = (1, 2, 3) if len(corners) == 3 else tuple(sorted(random.sample((1, 2, 3), len(corners))))
    perms = list(itertools.permutations(counts))
    random.shuffle(perms)
    for perm in perms:
        mapping = {corner: perm[i] for i, corner in enumerate(corners)}
        destinations = [_shift_corner(corner, mapping[corner]) for corner in corners]
        if len(set(destinations)) == len(destinations):
            return mapping
    return None


def _expected_output(
    grid: np.ndarray,
    corner_specs: list[dict],
    distractor_cells: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.zeros_like(grid)
    for r, c, color in distractor_cells:
        out[r, c] = color
    for spec in corner_specs:
        dst_r0, dst_c0 = _corner_anchor(spec["dst_corner"], grid.shape[0], grid.shape[1], BOX_SIZE, BOX_SIZE)
        _stamp_patch(out, spec["patch"], dst_r0, dst_c0)
    return out


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(15, 20)
    width = random.randint(15, 20)
    grid = np.zeros((height, width), dtype=int)

    # Hole counts are restricted to {1,2,3}, so unique-count samples can use at most three corner objects.
    corner_count = 3 if random.random() < 0.72 else 2
    corners = tuple(random.sample(CORNERS, corner_count))
    count_by_corner = _plan_corner_assignment(corners)
    if count_by_corner is None:
        return None

    corner_colors = random.sample(NONZERO_COLORS, corner_count)
    corner_specs: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    for corner, color in zip(corners, corner_colors):
        patch = _make_corner_patch(color, count_by_corner[corner])
        r0, c0 = _corner_anchor(corner, height, width, BOX_SIZE, BOX_SIZE)
        _stamp_patch(grid, patch, r0, c0)
        dst_corner = _shift_corner(corner, count_by_corner[corner])
        corner_specs.append(
            {
                "corner": corner,
                "dst_corner": dst_corner,
                "holes": count_by_corner[corner],
                "color": color,
                "patch": patch,
            }
        )
        for rr, cc in np.argwhere(patch != BG):
            occupied.add((r0 + int(rr), c0 + int(cc)))
        reserved |= _box_cells(r0, c0, BOX_SIZE, BOX_SIZE, pad=1, height=height, width=width)
        dst_r0, dst_c0 = _corner_anchor(dst_corner, height, width, BOX_SIZE, BOX_SIZE)
        reserved |= _box_cells(dst_r0, dst_c0, BOX_SIZE, BOX_SIZE, pad=1, height=height, width=width)

    distractor_target = random.choices([0, 1, 2], weights=[2, 4, 3], k=1)[0]
    need_side_touch = distractor_target > 0 and random.random() < 0.65
    placed_side_touch = False
    for i in range(distractor_target):
        if random.random() < 0.55:
            color = random.choice(corner_colors)
        else:
            color = random.choice(NONZERO_COLORS)
        patch = _make_distractor_patch(color)
        side_touch = need_side_touch and not placed_side_touch and (i == distractor_target - 1 or random.random() < 0.7)
        if not _place_distractor(grid, patch, side_touch=side_touch, occupied=occupied, reserved=reserved):
            if side_touch and _place_distractor(grid, patch, side_touch=False, occupied=occupied, reserved=reserved):
                side_touch = False
            else:
                return None
        if side_touch:
            placed_side_touch = True

    # Recover absolute distractor cells from the rendered grid so validation uses visible data only.
    abs_distractors: list[tuple[int, int, int]] = []
    bg, corner_objects = _corner_objects(grid)
    corner_source_cells = {cell for obj in corner_objects for cell in obj["coords"]}
    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color != bg and (r, c) not in corner_source_cells:
                abs_distractors.append((r, c, color))

    expected = _expected_output(grid, corner_specs, abs_distractors)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    analysis_bg, analyzed_corners = _corner_objects(grid)
    if analysis_bg != BG or len(analyzed_corners) != corner_count:
        return None
    analyzed_counts = sorted(int(np.sum(grid[r0 : r1 + 1, c0 : c1 + 1] == BG)) for r0, c0, r1, c1 in (obj["bbox"] for obj in analyzed_corners))
    if analyzed_counts != sorted(count_by_corner.values()):
        return None
    if len(set(analyzed_counts)) != corner_count:
        return None

    if need_side_touch:
        one_edge_distractors = 0
        for obj in _find_components(grid, bg=BG):
            if _bbox_corner(obj["bbox"], height, width) is not None:
                continue
            r0, c0, r1, c1 = obj["bbox"]
            touches = _touch_count(r0, c0, r1 - r0 + 1, c1 - c0 + 1, height, width)
            one_edge_distractors += int(touches == 1)
        if one_edge_distractors == 0:
            return None

    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corner-hole ARC input")
