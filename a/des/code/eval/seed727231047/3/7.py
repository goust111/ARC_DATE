import random
from collections import defaultdict

import numpy as np


BG = 0
GREY = 5
MAROON = 9

FRAME_SIZE = 5
INNER_SIZE = FRAME_SIZE - 2
SHAPE_SIZE = 3
MAX_ATTEMPTS = 400

GRID_H_RANGE = (26, 32)
GRID_W_RANGE = (26, 32)
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8]

DIRS = ("up", "right", "down", "left")
DIR_TO_K = {name: k for k, name in enumerate(DIRS)}
OPPOSITE = {"up": "down", "right": "left", "down": "up", "left": "right"}

SAFE_ASSIGNMENTS = (
    {"tl": "up", "tr": "right", "bl": "left", "br": "down"},
    {"tl": "left", "tr": "up", "bl": "down", "br": "right"},
)

SHAPE_FAMILIES = (
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
)


def _rotate_cw(arr: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(arr, -k)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


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
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _crop_component(points: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _is_hollow_frame(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(mask, border)


def _identify_shape(mask: np.ndarray) -> tuple[int, str]:
    matches: list[tuple[int, str]] = []
    for family_idx, base in enumerate(SHAPE_FAMILIES):
        if mask.shape != base.shape:
            continue
        for k, direction in enumerate(DIRS):
            if np.array_equal(mask, _rotate_cw(base, k)):
                matches.append((family_idx, direction))
    assert len(matches) == 1, f"ambiguous sprite shape: {matches}"
    return matches[0]


def _extract_frames(grid: np.ndarray) -> dict[str, dict]:
    grey_components = _components(grid == GREY)
    frames = []
    h, w = grid.shape
    for pts in grey_components:
        mask, bbox = _crop_component(pts)
        if not _is_hollow_frame(mask):
            continue
        r0, c0, r1, c1 = bbox
        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2
        candidates = {
            "up": (r0 - 1, center_c),
            "down": (r1 + 1, center_c),
            "left": (center_r, c0 - 1),
            "right": (center_r, c1 + 1),
        }
        marker_dirs = [
            direction
            for direction, (rr, cc) in candidates.items()
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == MAROON
        ]
        assert len(marker_dirs) == 1, f"frame {bbox} must have exactly one marker"
        direction = marker_dirs[0]
        frames.append(
            {
                "bbox": bbox,
                "dir": direction,
                "marker": candidates[direction],
            }
        )
    assert len(frames) == 4, f"expected 4 collection frames, got {len(frames)}"
    by_dir = {frame["dir"]: frame for frame in frames}
    assert len(by_dir) == 4, "frame marker directions must be unique"
    return by_dir


def _extract_sprites(grid: np.ndarray) -> tuple[int, list[dict]]:
    sprite_mask = (grid != BG) & (grid != GREY) & (grid != MAROON)
    components = _components(sprite_mask)
    assert components, "no sprites detected"
    family_idx: int | None = None
    sprites = []
    for pts in components:
        colors = {int(grid[r, c]) for r, c in pts}
        assert len(colors) == 1, "sprite must be monochrome"
        mask, bbox = _crop_component(pts)
        found_family, direction = _identify_shape(mask)
        if family_idx is None:
            family_idx = found_family
        assert found_family == family_idx, "all sprites must share one shape family"
        sprites.append(
            {
                "color": colors.pop(),
                "bbox": bbox,
                "dir": direction,
            }
        )
    assert family_idx is not None
    sprites.sort(key=lambda sprite: (sprite["bbox"][0], sprite["bbox"][1]))
    return family_idx, sprites


def _expanded_bbox(bbox: tuple[int, int, int, int], marker_dir: str, count: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    stack_len = max(INNER_SIZE, count * SHAPE_SIZE)
    outer_len = stack_len + 2
    if marker_dir == "up":
        return (r0, c0, r0 + outer_len - 1, c1)
    if marker_dir == "down":
        return (r1 - outer_len + 1, c0, r1, c1)
    if marker_dir == "left":
        return (r0, c0, r1, c0 + outer_len - 1)
    if marker_dir == "right":
        return (r0, c1 - outer_len + 1, r1, c1)
    raise AssertionError(f"unknown direction: {marker_dir}")


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _draw_frame(out: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    out[r0 : r1 + 1, c0] = GREY
    out[r0 : r1 + 1, c1] = GREY
    out[r0, c0 : c1 + 1] = GREY
    out[r1, c0 : c1 + 1] = GREY


def _paint_mask(out: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = mask.shape
    region = out[top : top + h, left : left + w]
    region[mask] = color


def _plan_output(
    grid_shape: tuple[int, int],
    frames: dict[str, dict],
    groups: dict[str, list[dict]],
    family_idx: int,
) -> tuple[np.ndarray, int, int]:
    h, w = grid_shape
    base_shape = SHAPE_FAMILIES[family_idx]
    plans = []
    min_r, min_c = 0, 0
    max_r, max_c = h - 1, w - 1

    for direction in DIRS:
        frame = frames[direction]
        sprites = groups.get(direction, [])
        bbox = _expanded_bbox(frame["bbox"], direction, len(sprites))
        marker_r, marker_c = frame["marker"]
        min_r = min(min_r, bbox[0], marker_r)
        min_c = min(min_c, bbox[1], marker_c)
        max_r = max(max_r, bbox[2], marker_r)
        max_c = max(max_c, bbox[3], marker_c)
        plans.append((direction, bbox, frame["marker"], sprites))

    for i in range(len(plans)):
        for j in range(i + 1, len(plans)):
            assert not _boxes_overlap(plans[i][1], plans[j][1]), "expanded frames overlap"

    row_shift = -min_r
    col_shift = -min_c
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), BG, dtype=int)

    for direction, bbox, marker, sprites in plans:
        shifted_bbox = (
            bbox[0] + row_shift,
            bbox[1] + col_shift,
            bbox[2] + row_shift,
            bbox[3] + col_shift,
        )
        shifted_marker = (marker[0] + row_shift, marker[1] + col_shift)
        out[shifted_marker] = MAROON
        _draw_frame(out, shifted_bbox)

        target_mask = _rotate_cw(base_shape, DIR_TO_K[OPPOSITE[direction]])
        r0, c0, r1, c1 = shifted_bbox
        for idx, sprite in enumerate(sprites):
            if direction == "up":
                top, left = r0 + 1 + idx * SHAPE_SIZE, c0 + 1
            elif direction == "down":
                top, left = r1 - SHAPE_SIZE * (idx + 1), c0 + 1
            elif direction == "left":
                top, left = r0 + 1, c0 + 1 + idx * SHAPE_SIZE
            else:
                top, left = r0 + 1, c1 - SHAPE_SIZE * (idx + 1)
            _paint_mask(out, target_mask, sprite["color"], top, left)

    return out, row_shift, col_shift


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group monochrome sprites by orientation and stack them into marker-labelled corner frames.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _extract_frames(g)
    family_idx, sprites = _extract_sprites(g)

    groups: dict[str, list[dict]] = defaultdict(list)
    for sprite in sprites:
        groups[sprite["dir"]].append(sprite)

    out, _, _ = _plan_output(g.shape, frames, groups, family_idx)
    return out


def _frame_positions(h: int, w: int) -> dict[str, tuple[int, int, int, int]]:
    return {
        "tl": (1, 1, 1 + FRAME_SIZE - 1, 1 + FRAME_SIZE - 1),
        "tr": (1, w - FRAME_SIZE - 1, 1 + FRAME_SIZE - 1, w - 2),
        "bl": (h - FRAME_SIZE - 1, 1, h - 2, 1 + FRAME_SIZE - 1),
        "br": (h - FRAME_SIZE - 1, w - FRAME_SIZE - 1, h - 2, w - 2),
    }


def _place_marker(grid: np.ndarray, bbox: tuple[int, int, int, int], direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    if direction == "up":
        pos = (r0 - 1, center_c)
    elif direction == "down":
        pos = (r1 + 1, center_c)
    elif direction == "left":
        pos = (center_r, c0 - 1)
    else:
        pos = (center_r, c1 + 1)
    grid[pos] = MAROON
    return pos


def _choose_sprite_directions(rng: random.Random) -> list[str]:
    active_count = rng.randint(2, 4)
    active = rng.sample(list(DIRS), active_count)
    expanding = rng.random() < 0.5
    directions = list(active)
    if expanding:
        total = rng.randint(active_count + 1, len(COLOR_POOL))
        while len(directions) < total:
            directions.append(rng.choice(active))
    rng.shuffle(directions)
    return directions


def _choose_positions(h: int, w: int, count: int, rng: random.Random) -> list[tuple[int, int]] | None:
    candidates = [
        (r, c)
        for r in range(7, h - 9)
        for c in range(7, w - 9)
    ]
    rng.shuffle(candidates)
    blocked = np.zeros((h, w), dtype=bool)
    picked: list[tuple[int, int]] = []
    for r, c in candidates:
        if blocked[r - 1 : r + SHAPE_SIZE + 1, c - 1 : c + SHAPE_SIZE + 1].any():
            continue
        picked.append((r, c))
        blocked[r - 1 : r + SHAPE_SIZE + 1, c - 1 : c + SHAPE_SIZE + 1] = True
        if len(picked) == count:
            return picked
    return None


def _build_input(
    h: int,
    w: int,
    family_idx: int,
    assignment: dict[str, str],
    directions: list[str],
    colors: list[int],
    positions: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    frames = _frame_positions(h, w)
    for corner, bbox in frames.items():
        _draw_frame(grid, bbox)
        _place_marker(grid, bbox, assignment[corner])

    for direction, color, (r, c) in zip(directions, colors, positions):
        mask = _rotate_cw(SHAPE_FAMILIES[family_idx], DIR_TO_K[direction])
        _paint_mask(grid, mask, color, r, c)
    return grid


def _expanded_boxes_fit(h: int, w: int, assignment: dict[str, str], counts: dict[str, int]) -> bool:
    frame_boxes = _frame_positions(h, w)
    planned = []
    for corner, bbox in frame_boxes.items():
        direction = assignment[corner]
        planned.append(_expanded_bbox(bbox, direction, counts.get(direction, 0)))
    for bbox in planned:
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] >= h or bbox[3] >= w:
            return False
    for i in range(len(planned)):
        for j in range(i + 1, len(planned)):
            if _boxes_overlap(planned[i], planned[j]):
                return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    family_idx = rng.randrange(len(SHAPE_FAMILIES))
    assignment = dict(rng.choice(SAFE_ASSIGNMENTS))
    directions = _choose_sprite_directions(rng)
    colors = rng.sample(COLOR_POOL, len(directions))
    counts = {direction: directions.count(direction) for direction in DIRS}
    if not _expanded_boxes_fit(h, w, assignment, counts):
        return None
    positions = _choose_positions(h, w, len(directions), rng)
    if positions is None:
        return None
    grid = _build_input(h, w, family_idx, assignment, directions, colors, positions)
    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape[0] > h or out.shape[1] > w:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
