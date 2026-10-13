from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
ARC_MAX = 30

PATTERN_COLORS = (3, 4, 6, 7, 8, 9)
GEN_RETRIES = 600


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
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _separator_col(g: np.ndarray) -> int:
    cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert len(cols) == 1, "expected one full-height grey separator column"
    return int(cols[0])


def _is_blue_frame(g: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    ring = np.zeros((h, w), dtype=bool)
    ring[0, :] = True
    ring[-1, :] = True
    ring[:, 0] = True
    ring[:, -1] = True
    return bool(np.array_equal(sub == BLUE, ring))


def _blue_frames(g: np.ndarray, sep: int, side: str) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for pts in _components(g == BLUE):
        rows = [r for r, _c in pts]
        cols = [c for _r, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if side == "left" and c1 >= sep:
            continue
        if side == "right" and c0 <= sep:
            continue
        bbox = (int(r0), int(c0), int(r1), int(c1))
        if _is_blue_frame(g, bbox):
            frames.append(bbox)
    frames.sort()
    return frames


def _read_dictionary(g: np.ndarray, sep: int) -> dict[int, np.ndarray]:
    templates: dict[int, np.ndarray] = {}
    for r0, c0, r1, c1 in _blue_frames(g, sep, "left"):
        if (r1 - r0 + 1, c1 - c0 + 1) != (5, 5):
            continue
        marker_col = c0 - 1
        assert marker_col >= 0, "dictionary frames must have a marker column"
        marker_count = int(np.count_nonzero(g[r0 : r1 + 1, marker_col] == RED))
        assert marker_count > 0, "each dictionary template must have red count markers"
        assert marker_count not in templates, "dictionary marker counts must be unique"
        template = g[r0 + 1 : r1, c0 + 1 : c1].copy()
        assert template.shape == (3, 3)
        assert bool(np.any(template != BLACK)), "dictionary templates must be nonempty"
        templates[marker_count] = template
    assert templates, "expected at least one dictionary template"
    return templates


def _tile_template(template: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    ih, iw = map(int, shape)
    th, tw = map(int, template.shape)
    rr, cc = np.indices((ih, iw))
    return template[rr % th, cc % tw]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed room with the left dictionary pattern indexed by its count of red markers.

    Input structure:
    - Black background is split by one full-height grey vertical separator.
    - The left side is a pattern dictionary: 5x5 blue-framed entries with 3x3 interiors, each tagged by a vertical stack of red pixels immediately to its left.
    - The right side has blue rectangular closed rooms of varied sizes; their interiors are black except for scattered red marker pixels.

    Rule:
    - Count the red pixels beside each left dictionary frame and store that count as the key for its 3x3 interior template.
    - For every right-side room, count its internal red pixels, erase the whole interior, and look up the template with the same count.
    - Periodically tile the matched 3x3 template from the room interior's top-left corner, clipping at the room's bottom and right edges.
    - Keep the left dictionary, the grey separator, and all blue room borders unchanged.

    Key insight:
    The red pixels are count keys, not decoration or position cues; the chosen fixed-size template repeats periodically instead of scaling to the room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    sep = _separator_col(g)
    templates = _read_dictionary(g, sep)

    rooms = _blue_frames(g, sep, "right")
    assert rooms, "expected at least one right-side room"

    out = g.copy()
    for r0, c0, r1, c1 in rooms:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_count = int(np.count_nonzero(interior == RED))
        assert marker_count in templates, "room marker count must match a dictionary key"
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_template(templates[marker_count], interior.shape)
    return out


# --------------- generator ---------------


def _base_masks() -> list[np.ndarray]:
    return [
        np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    ]


def _mask_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    seen: set[bytes] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for var in (rot, np.fliplr(rot), np.flipud(rot)):
            key = var.tobytes()
            if key not in seen:
                seen.add(key)
                variants.append(var.copy())
    return variants


def _sample_templates(rng: random.Random, count: int) -> dict[int, np.ndarray]:
    pool: list[np.ndarray] = []
    seen: set[bytes] = set()
    for mask in _base_masks():
        for var in _mask_variants(mask):
            key = var.tobytes()
            if key not in seen:
                seen.add(key)
                pool.append(var)
    masks = rng.sample(pool, count)
    colors = rng.sample(PATTERN_COLORS, count)

    templates: dict[int, np.ndarray] = {}
    for idx, (mask, color) in enumerate(zip(masks, colors), start=1):
        template = np.full((3, 3), BLACK, dtype=int)
        template[mask] = int(color)
        templates[int(idx)] = template
    return templates


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    grid[r0, c0 : c0 + w] = BLUE
    grid[r0 + h - 1, c0 : c0 + w] = BLUE
    grid[r0 : r0 + h, c0] = BLUE
    grid[r0 : r0 + h, c0 + w - 1] = BLUE


def _draw_dictionary(
    grid: np.ndarray,
    sep: int,
    templates: dict[int, np.ndarray],
    rng: random.Random,
) -> bool:
    h, _w = grid.shape
    count = len(templates)
    gap_options = [gap for gap in (1, 2) if count * 5 + (count - 1) * gap <= h - 2]
    if not gap_options:
        return False
    gap = rng.choice(gap_options)
    total_h = count * 5 + (count - 1) * gap
    top = rng.randint(1, h - total_h - 1)
    frame_c0 = rng.randint(2, max(2, sep - 5))
    marker_col = frame_c0 - 1

    keys = list(templates)
    rng.shuffle(keys)
    for idx, key in enumerate(keys):
        r0 = top + idx * (5 + gap)
        c0 = frame_c0
        _draw_frame(grid, r0, c0, 5, 5)
        grid[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4] = templates[key]

        start = rng.randint(r0, r0 + 5 - int(key))
        for rr in range(start, start + int(key)):
            grid[rr, marker_col] = RED
    return True


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _place_one_room(
    grid: np.ndarray,
    sep: int,
    key: int,
    boxes: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    right_c0 = sep + 2
    right_c1 = w - 2
    if right_c1 - right_c0 + 1 < 5:
        return False

    for _ in range(160):
        max_h = min(10, h - 2)
        max_w = min(10, right_c1 - right_c0 + 1)
        outer_h = rng.randint(5, max_h)
        outer_w = rng.randint(5, max_w)
        r0 = rng.randint(1, h - outer_h - 1)
        c0 = rng.randint(right_c0, right_c1 - outer_w + 1)
        bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
        if any(_overlaps(bbox, other) for other in boxes):
            continue

        _draw_frame(grid, r0, c0, outer_h, outer_w)
        cells = [
            (r, c)
            for r in range(r0 + 1, r0 + outer_h - 1)
            for c in range(c0 + 1, c0 + outer_w - 1)
        ]
        if len(cells) < key:
            return False
        for r, c in rng.sample(cells, int(key)):
            grid[r, c] = RED
        boxes.append(bbox)
        return True
    return False


def _draw_rooms(
    grid: np.ndarray,
    sep: int,
    templates: dict[int, np.ndarray],
    rng: random.Random,
) -> bool:
    keys = list(templates)
    room_keys = keys[:]
    if len(keys) < 4 and rng.random() < 0.5:
        room_keys.append(rng.choice(keys))
    rng.shuffle(room_keys)

    boxes: list[tuple[int, int, int, int]] = []
    for key in room_keys:
        if not _place_one_room(grid, sep, int(key), boxes, rng):
            return False

    if len(boxes) > 1:
        shapes = {(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in boxes}
        if len(shapes) == 1:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    template_count = rng.randint(2, 4)
    min_h = template_count * 5 + (template_count - 1) + 2
    height = rng.randint(max(22, min_h), ARC_MAX)
    width = rng.randint(25, ARC_MAX)
    sep_max = min(11, width - 14)
    if sep_max < 8:
        return None
    sep = rng.randint(8, sep_max)

    grid = np.full((height, width), BLACK, dtype=int)
    grid[:, sep] = GREY

    templates = _sample_templates(rng, template_count)
    if not _draw_dictionary(grid, sep, templates, rng):
        return None
    if not _draw_rooms(grid, sep, templates, rng):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or max(out.shape) > ARC_MAX:
        return None

    # Ensure the transformation is visible in every generated task.
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-template dictionary input")
