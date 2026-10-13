"""ARC task: route each monochrome sticker to frames addressed by its color."""

from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BACKGROUND = 0
GRAY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as arrays of (row, column) coordinates."""
    h, w = mask.shape
    unseen = mask.copy()
    result: list[np.ndarray] = []
    for sr, sc in np.argwhere(unseen):
        sr, sc = int(sr), int(sc)
        if not unseen[sr, sc]:
            continue
        unseen[sr, sc] = False
        queue = deque([(sr, sc)])
        cells = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and unseen[rr, cc]:
                    unseen[rr, cc] = False
                    queue.append((rr, cc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _is_frame(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    height, width = r1 - r0 + 1, c1 - c0 + 1
    if height < 5 or width < 5 or len(coords) != 2 * height + 2 * width - 4:
        return None
    expected = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    actual = {tuple(map(int, cell)) for cell in coords}
    return (r0, c0, r1, c1) if actual == expected else None


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames = []
    for color in np.unique(grid):
        color = int(color)
        if color in (bg, GRAY):
            continue
        for coords in _components(grid == color):
            box = _is_frame(coords)
            if box is not None:
                frames.append({"color": color, "bbox": box, "coords": coords})
    frames.sort(key=lambda frame: frame["bbox"])
    colors = [frame["color"] for frame in frames]
    assert len(frames) >= 2, "expected at least two hollow rectangular frames"
    assert len(colors) == len(set(colors)), "frame colors must be unique"
    return frames


def _read_contents(grid: np.ndarray, frames: list[dict], bg: int) -> None:
    frame_colors = {frame["color"] for frame in frames}
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        anchors_local = np.argwhere(interior == GRAY)
        assert len(anchors_local) >= 1, "each frame must contain gray anchors"
        frame["anchors"] = [
            (r0 + 1 + int(r), c0 + 1 + int(c)) for r, c in anchors_local
        ]

        sticker_mask = (interior != bg) & (interior != GRAY)
        sticker_cells = np.argwhere(sticker_mask)
        assert len(sticker_cells) >= 2, "each frame must contain one multi-cell sticker"
        sticker_colors = set(map(int, interior[sticker_mask]))
        assert len(sticker_colors) == 1, "a sticker must be monochrome"
        sticker_color = sticker_colors.pop()
        assert sticker_color in frame_colors and sticker_color != frame["color"], (
            "sticker color must address a different frame"
        )
        comps = _components(sticker_mask)
        assert len(comps) == 1, "the sticker must be 4-connected"

        sr0, sc0 = sticker_cells.min(axis=0)
        sr1, sc1 = sticker_cells.max(axis=0)
        sh, sw = int(sr1 - sr0 + 1), int(sc1 - sc0 + 1)
        assert sh % 2 == 1 and sw % 2 == 1, "sticker bbox sides must be odd"
        center_local = (int((sr0 + sr1) // 2), int((sc0 + sc1) // 2))
        assert sticker_mask[center_local], "sticker bbox center must belong to sticker"
        center = (r0 + 1 + center_local[0], c0 + 1 + center_local[1])
        absolute = [(r0 + 1 + int(r), c0 + 1 + int(c)) for r, c in sticker_cells]
        frame["sticker_color"] = sticker_color
        frame["sticker_cells"] = absolute
        frame["offsets"] = [(r - center[0], c - center[1]) for r, c in absolute]


def main(grid: np.ndarray) -> np.ndarray:
    """Route each sticker to its same-colored frame and stamp it on every gray anchor.

    Input structure:
    - The dominant background contains at least two disjoint hollow rectangular frames,
      each with a unique non-gray color.
    - Inside every frame are one connected monochrome multi-cell sticker with an odd
      bounding box and occupied center, plus one or more isolated gray anchor pixels.
    - Every sticker has the color of a different frame.

    Rule:
    - Use a sticker's color to select the frame whose border has that color, erase the
      original sticker, and translate a copy of its exact mask to every gray anchor in
      the selected frame, aligning the sticker's bounding-box center with the anchor.
    - The stamps cover their anchors; frame borders and all other background stay fixed.

    Key insight:
    - Sticker color is a destination address and gray pixels are stamp centers, so the
      shapes travel between containers rather than expanding inside their source frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a two-dimensional grid"
    values, counts = np.unique(g, return_counts=True)
    bg = int(values[int(np.argmax(counts))])
    assert bg != GRAY, "gray is reserved for anchors"
    frames = _find_frames(g, bg)
    _read_contents(g, frames, bg)
    by_color = {frame["color"]: frame for frame in frames}

    frame_cells = {tuple(map(int, p)) for frame in frames for p in frame["coords"]}
    all_anchors = {p for frame in frames for p in frame["anchors"]}
    assert set(map(tuple, np.argwhere(g == GRAY))) == all_anchors, (
        "gray pixels must be anchors inside frames"
    )

    out = g.copy()
    for frame in frames:
        for r, c in frame["sticker_cells"]:
            out[r, c] = bg

    claims: dict[tuple[int, int], int] = {}
    for source in frames:
        target = by_color[source["sticker_color"]]
        tr0, tc0, tr1, tc1 = target["bbox"]
        for ar, ac in target["anchors"]:
            for dr, dc in source["offsets"]:
                r, c = ar + dr, ac + dc
                assert tr0 < r < tr1 and tc0 < c < tc1, "stamp must fit inside target"
                assert (r, c) not in frame_cells, "stamp must not overwrite a frame"
                previous = claims.get((r, c))
                assert previous is None or previous == source["sticker_color"], (
                    "different stamps must not conflict"
                )
                claims[(r, c)] = source["sticker_color"]

    for (r, c), color in claims.items():
        out[r, c] = color
    return out


def _random_sprite(rng: random.Random) -> np.ndarray:
    """Make a varied connected mask with an odd bbox and occupied center."""
    h, w = rng.choice(((3, 3), (3, 5), (5, 3)))
    mask = np.zeros((h, w), dtype=bool)
    center = (h // 2, w // 2)
    mask[center] = True
    required = {(0, rng.randrange(w)), (h - 1, rng.randrange(w)),
                (rng.randrange(h), 0), (rng.randrange(h), w - 1)}
    target_size = rng.randint(max(5, len(required) + 1), min(h * w - 1, 9))
    targets = list(required)
    rng.shuffle(targets)
    for target in targets:
        r, c = center
        axes = [0, 1]
        rng.shuffle(axes)
        for axis in axes:
            goal = target[axis]
            while (r, c)[axis] != goal:
                if axis == 0:
                    r += 1 if goal > r else -1
                else:
                    c += 1 if goal > c else -1
                mask[r, c] = True
    while int(mask.sum()) < target_size:
        candidates = []
        for r, c in np.argwhere(mask):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                    candidates.append((rr, cc))
        if not candidates:
            break
        mask[rng.choice(candidates)] = True
    return mask


def _rect_cells(center: tuple[int, int], shape: tuple[int, int], pad: int = 0) -> set[tuple[int, int]]:
    r, c = center
    hh, hw = shape[0] // 2 + pad, shape[1] // 2 + pad
    return {(rr, cc) for rr in range(r - hh, r + hh + 1)
            for cc in range(c - hw, c + hw + 1)}


def _choose_layout(rng: random.Random, n: int) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    h, w = rng.randint(27, 30), rng.randint(27, 30)
    slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
    rng.shuffle(slots)
    boxes = []
    for sr, sc in slots[:n]:
        ra, rb = (1, h // 2 - 1) if sr == 0 else (h // 2 + 1, h - 2)
        ca, cb = (1, w // 2 - 1) if sc == 0 else (w // 2 + 1, w - 2)
        fh = rng.randint(11, min(13, rb - ra + 1))
        fw = rng.randint(11, min(13, cb - ca + 1))
        r0 = rng.randint(ra, rb - fh + 1)
        c0 = rng.randint(ca, cb - fw + 1)
        boxes.append((r0, c0, r0 + fh - 1, c0 + fw - 1))
    return h, w, boxes


def _attempt_generate(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    n = rng.randint(2, 4)
    h, w, boxes = _choose_layout(rng, n)
    colors = rng.sample(PALETTE, n)
    order = list(range(n))
    rng.shuffle(order)
    destination = {}
    for i, source in enumerate(order):
        destination[source] = order[(i + 1) % n]  # one visible, non-self cycle
    sprites = [_random_sprite(rng) for _ in range(n)]

    grid = np.full((h, w), BACKGROUND, dtype=int)
    source_centers: list[tuple[int, int]] = []
    anchors: list[list[tuple[int, int]]] = []
    for i, (r0, c0, r1, c1) in enumerate(boxes):
        color = colors[i]
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color

        incoming = next(src for src, dst in destination.items() if dst == i)
        incoming_shape = sprites[incoming].shape
        own_shape = sprites[i].shape
        usable = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        rng.shuffle(usable)
        own_center = None
        for center in usable:
            if _rect_cells(center, own_shape) <= set(usable):
                own_center = center
                break
        if own_center is None:
            return None
        occupied = _rect_cells(own_center, own_shape, pad=1)
        want = 2 if rng.random() < 0.60 else 1
        chosen = []
        for center in usable:
            footprint = _rect_cells(center, incoming_shape)
            clearance = _rect_cells(center, incoming_shape, pad=1)
            if footprint <= set(usable) and not (clearance & occupied):
                chosen.append(center)
                occupied |= clearance
                if len(chosen) == want:
                    break
        if len(chosen) != want:
            return None
        source_centers.append(own_center)
        anchors.append(chosen)

    for i, mask in enumerate(sprites):
        cr, cc = source_centers[i]
        mh, mw = mask.shape
        color = colors[destination[i]]
        for rr, dc in np.argwhere(mask):
            grid[cr + int(rr) - mh // 2, cc + int(dc) - mw // 2] = color
    for points in anchors:
        for r, c in points:
            grid[r, c] = GRAY

    expected = grid.copy()
    for i, mask in enumerate(sprites):
        cr, cc = source_centers[i]
        mh, mw = mask.shape
        for rr, dc in np.argwhere(mask):
            expected[cr + int(rr) - mh // 2, cc + int(dc) - mw // 2] = BACKGROUND
    for src, dst in destination.items():
        mask = sprites[src]
        mh, mw = mask.shape
        color = colors[dst]
        for ar, ac in anchors[dst]:
            for rr, dc in np.argwhere(mask):
                expected[ar + int(rr) - mh // 2, ac + int(dc) - mw // 2] = color

    try:
        actual = main(grid)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(actual, expected) or np.array_equal(grid, expected):
        return None

    # Semantic anti-shortcut checks: erasure, cross-frame addressing, full masks,
    # and anchor multiplicity must each affect the independently rendered result.
    no_erase = expected.copy()
    for i, mask in enumerate(sprites):
        cr, cc = source_centers[i]
        mh, mw = mask.shape
        for rr, dc in np.argwhere(mask):
            no_erase[cr + int(rr) - mh // 2, cc + int(dc) - mw // 2] = colors[destination[i]]
    if np.array_equal(no_erase, expected) or any(int(mask.sum()) < 2 for mask in sprites):
        return None

    wrong_self = grid.copy()
    for i, mask in enumerate(sprites):
        cr, cc = source_centers[i]
        mh, mw = mask.shape
        for rr, dc in np.argwhere(mask):
            wrong_self[cr + int(rr) - mh // 2, cc + int(dc) - mw // 2] = BACKGROUND
        for ar, ac in anchors[i]:
            for rr, dc in np.argwhere(mask):
                r = ar + int(rr) - mh // 2
                c = ac + int(dc) - mw // 2
                if 0 <= r < h and 0 <= c < w:
                    wrong_self[r, c] = colors[destination[i]]
    if np.array_equal(wrong_self, expected):
        return None
    if not any(len(points) > 1 for points in anchors) and rng.random() < 0.70:
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    """Generate a diverse, validated instance with 2--4 mutually addressed frames."""
    rng = random.Random()
    for _ in range(400):
        made = _attempt_generate(rng)
        if made is not None:
            return made[0]
    raise RuntimeError("failed to generate a valid color-addressed stamping instance")
