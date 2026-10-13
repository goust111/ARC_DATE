import random
from collections import deque

import numpy as np


BG = 0
SPRITE_COLOR = 1
MARKER_ORDER = (2, 3, 4)
MARKER_RANK = {color: idx for idx, color in enumerate(MARKER_ORDER)}
DIAGONAL_TO_ROT_K = {
    (-1, 1): 0,   # NE -> 0 deg
    (1, 1): 3,    # SE -> 90 deg clockwise
    (1, -1): 2,   # SW -> 180 deg
    (-1, -1): 1,  # NW -> 270 deg clockwise
}
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _shape_library() -> tuple[np.ndarray, ...]:
    reps: dict[tuple[int, ...], np.ndarray] = {}
    for bits in range(1 << 9):
        area = bits.bit_count()
        if not 3 <= area <= 6:
            continue
        mask = np.zeros((3, 3), dtype=bool)
        for idx in range(9):
            if bits & (1 << idx):
                mask[idx // 3, idx % 3] = True
        comps = _components(mask)
        if len(comps) != 1:
            continue
        if _bbox(comps[0]) != (0, 0, 2, 2):
            continue
        rotations = [np.rot90(mask, k=k) for k in range(4)]
        if any(np.array_equal(mask, rotations[k]) for k in (1, 2, 3)):
            continue
        key = min(tuple(rot.astype(int).ravel()) for rot in rotations)
        reps.setdefault(key, mask.astype(int))
    return tuple(reps.values())


SHAPE_LIBRARY = _shape_library()


def _concat_with_gaps(parts: list[np.ndarray]) -> np.ndarray:
    height = parts[0].shape[0]
    assert all(part.shape[0] == height for part in parts)
    gap = np.zeros((height, 1), dtype=int)
    strips = []
    for idx, part in enumerate(parts):
        if idx:
            strips.append(gap)
        strips.append(part)
    return np.concatenate(strips, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each sprite by the angle encoded by its diagonal marker, then sort by marker color and concatenate.

    Input structure:
    - A black background contains exactly three disconnected monochrome sprites plus three singleton marker pixels.
    - Each sprite occupies a square odd-sized bounding box, and one marker lies exactly one diagonal step outside one bbox corner.
    - Marker colors are red, green, and yellow, one of each, and markers are not part of the sprite components.

    Rule:
    - For each sprite, find the unique marker at one of the four diagonal-outside corner cells of its bounding box.
    - Convert marker position into a rotation: northeast -> 0 degrees, southeast -> 90 degrees clockwise, southwest -> 180 degrees, northwest -> 270 degrees clockwise.
    - Rotate the cropped sprite inside its bbox, discard the marker pixels, sort the rotated crops by marker color in red/green/yellow order, and concatenate them left-to-right with one black column between neighbors.

    Key insight:
    - Each marker carries two independent signals: its diagonal position encodes rotation, while its color encodes output order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    marker_positions: dict[int, tuple[int, int]] = {}
    for color in MARKER_ORDER:
        coords = np.argwhere(g == color)
        assert len(coords) == 1
        marker_positions[color] = tuple(map(int, coords[0]))

    sprite_mask = (g != BG) & ~np.isin(g, MARKER_ORDER)
    components = _components(sprite_mask)
    assert len(components) == 3

    pieces = []
    used_markers: set[int] = set()
    for coords in components:
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        side_h = r1 - r0 + 1
        side_w = c1 - c0 + 1
        assert side_h == side_w and side_h % 2 == 1

        sprite = _crop_bbox(g, box)
        candidates = {
            (-1, -1): (r0 - 1, c0 - 1),
            (-1, 1): (r0 - 1, c1 + 1),
            (1, 1): (r1 + 1, c1 + 1),
            (1, -1): (r1 + 1, c0 - 1),
        }
        matches = []
        for diag, pos in candidates.items():
            marker_color = int(g[pos]) if 0 <= pos[0] < g.shape[0] and 0 <= pos[1] < g.shape[1] else BG
            if marker_color in MARKER_RANK:
                matches.append((marker_color, diag))
        assert len(matches) == 1
        marker_color, diag = matches[0]
        assert marker_color not in used_markers
        used_markers.add(marker_color)
        pieces.append((marker_color, np.rot90(sprite, k=DIAGONAL_TO_ROT_K[diag])))

    assert used_markers == set(MARKER_ORDER)
    ordered = [part for _color, part in sorted(pieces, key=lambda item: MARKER_RANK[item[0]])]
    return _concat_with_gaps(ordered)


def _can_place(new_box: tuple[int, int, int, int], boxes: list[tuple[int, int, int, int]]) -> bool:
    r0, c0, r1, c1 = new_box
    for er0, ec0, er1, ec1 in boxes:
        if not (r1 + 2 < er0 or er1 + 2 < r0 or c1 + 2 < ec0 or ec1 + 2 < c0):
            return False
    return True


def _manual_expected(specs):
    parts = []
    for marker_color in MARKER_ORDER:
        spec = next(item for item in specs if item["marker_color"] == marker_color)
        parts.append(np.rot90(spec["sprite"], k=DIAGONAL_TO_ROT_K[spec["diag"]]))
    return _concat_with_gaps(parts)


def _gen_attempt() -> np.ndarray | None:
    if len(SHAPE_LIBRARY) < 3:
        raise RuntimeError("shape library is unexpectedly small")

    h = random.randint(13, 20)
    w = random.randint(13, 20)
    shapes = random.sample(list(SHAPE_LIBRARY), 3)
    marker_colors = list(MARKER_ORDER)
    random.shuffle(marker_colors)
    diags = random.sample(list(DIAGONAL_TO_ROT_K), 3)

    placed_boxes: list[tuple[int, int, int, int]] = []
    specs = []
    for shape, marker_color, diag in zip(shapes, marker_colors, diags):
        sprite = np.rot90(shape, k=random.randrange(4)).astype(int) * SPRITE_COLOR
        side = sprite.shape[0]
        success = False
        for _ in range(120):
            r0 = random.randint(1, h - side - 1)
            c0 = random.randint(1, w - side - 1)
            box = (r0, c0, r0 + side - 1, c0 + side - 1)
            if not _can_place(box, placed_boxes):
                continue
            marker = {
                (-1, -1): (r0 - 1, c0 - 1),
                (-1, 1): (r0 - 1, c0 + side),
                (1, 1): (r0 + side, c0 + side),
                (1, -1): (r0 + side, c0 - 1),
            }[diag]
            if not (0 <= marker[0] < h and 0 <= marker[1] < w):
                continue
            placed_boxes.append(box)
            specs.append(
                {
                    "sprite": sprite,
                    "bbox": box,
                    "marker": marker,
                    "marker_color": marker_color,
                    "diag": diag,
                }
            )
            success = True
            break
        if not success:
            return None

    xs = [spec["bbox"][1] for spec in specs]
    left_to_right = [spec["marker_color"] for spec in sorted(specs, key=lambda item: item["bbox"][1])]
    if left_to_right == list(MARKER_ORDER):
        return None
    if all(diag == (-1, 1) for diag in diags):
        return None

    grid = np.zeros((h, w), dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        sprite = spec["sprite"]
        grid[r0 : r1 + 1, c0 : c1 + 1][sprite != BG] = SPRITE_COLOR
        mr, mc = spec["marker"]
        grid[mr, mc] = spec["marker_color"]

    if len(_components((grid != BG) & ~np.isin(grid, MARKER_ORDER))) != 3:
        return None

    expected = _manual_expected(specs)
    if max(expected.shape) > 30:
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
