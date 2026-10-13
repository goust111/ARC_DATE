from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    winners = values[counts == counts.max()]
    assert len(winners) == 1, "the background must be uniquely dominant"
    return int(winners[0])


def _mono_components(grid: np.ndarray, color: int) -> list[set[Coord]]:
    """Return monochromatic 4-connected components in (row, column) coordinates."""
    unseen = {tuple(map(int, rc)) for rc in np.argwhere(grid == color)}
    components: list[set[Coord]] = []
    while unseen:
        start = unseen.pop()
        component = {start}
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            for cell in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if cell in unseen:
                    unseen.remove(cell)
                    component.add(cell)
                    queue.append(cell)
        components.append(component)
    return components


def _rect_border(box: BBox) -> set[Coord]:
    r0, c0, r1, c1 = box
    return (
        {(r0, c) for c in range(c0, c1 + 1)}
        | {(r1, c) for c in range(c0, c1 + 1)}
        | {(r, c0) for r in range(r0, r1 + 1)}
        | {(r, c1) for r in range(r0, r1 + 1)}
    )


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for component in _mono_components(grid, color):
            rows = [p[0] for p in component]
            cols = [p[1] for p in component]
            box = (min(rows), min(cols), max(rows), max(cols))
            r0, c0, r1, c1 = box
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            if component != _rect_border(box):
                continue

            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            motif_colors = set(map(int, np.unique(interior))) - {bg}
            assert len(motif_colors) == 1, "each cavity must contain one monochromatic motif"
            motif_color = motif_colors.pop()
            assert motif_color != color, "the motif must differ from its enclosing frame"
            mask = interior != bg
            assert bool(mask.any()), "each cavity must contain a nonempty motif"
            frames.append(
                {
                    "box": box,
                    "color": color,
                    "motif_color": motif_color,
                    "mask": mask.copy(),
                    "key": tuple(map(int, interior.shape)),
                }
            )
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap motifs between hollow frames with equal cavity dimensions and recolor them with their source frame colors.

    Input structure:
    - A uniquely dominant background contains an even number of separated, one-pixel-thick
      monochromatic rectangular frames, each with one nonempty monochromatic motif in its cavity.
    - Every cavity height-by-width occurs exactly twice; frame colors may repeat across other pairs,
      and motif colors carry no matching information.

    Rule:
    - Pair frames whose cavity height and width are equal. Clear both original cavity motifs to
      background, then copy each motif mask to the same cavity-relative coordinates in its partner.
    - Color a copied mask with the source motif's enclosing frame color, while preserving every frame.

    Key insight:
    - Equality of the derived cavity dimensions uniquely determines the pairs; only the motif mask
      and its source frame color cross the pair, while proximity and all original motif colors are distractors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be two-dimensional"
    bg = _background(g)
    frames = _find_frames(g, bg)
    assert len(frames) >= 4 and len(frames) % 2 == 0, "expected an even set of frames"

    by_size: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for frame in frames:
        by_size[frame["key"]].append(frame)
    assert all(len(pair) == 2 for pair in by_size.values()), "each cavity size must occur exactly twice"
    assert len(by_size) == len(frames) // 2, "different pairs must have different cavity sizes"

    out = g.copy()
    for pair in by_size.values():
        first, second = pair
        for destination, source in ((first, second), (second, first)):
            r0, c0, r1, c1 = destination["box"]
            cavity = out[r0 + 1 : r1, c0 + 1 : c1]
            cavity[:] = bg
            cavity[source["mask"]] = int(source["color"])
    return out


# Generator geometry.  Sizes are cavity sizes, not outer-frame sizes.
_SIZE_POOL = ((3, 4), (3, 5), (4, 3), (4, 5), (5, 3), (5, 4), (5, 6), (6, 4))
_RETRIES = 500


def _grow_motif(h: int, w: int) -> np.ndarray:
    target = random.randint(3, min(8, max(3, h * w // 2)))
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < target:
        frontier: set[Coord] = set()
        for r, c in cells:
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.add((rr, cc))
        cells.add(random.choice(tuple(frontier)))
    mask = np.zeros((h, w), dtype=bool)
    for cell in cells:
        mask[cell] = True
    return mask


def _separated(a: BBox, b: BBox, gap: int = 1) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _place_boxes(H: int, W: int, sizes: list[tuple[int, int]]) -> list[BBox] | None:
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    boxes: list[BBox | None] = [None] * len(sizes)

    def search(k: int) -> bool:
        if k == len(order):
            return True
        i = order[k]
        ih, iw = sizes[i]
        fh, fw = ih + 2, iw + 2
        candidates = [(r, c) for r in range(1, H - fh) for c in range(1, W - fw)]
        random.shuffle(candidates)
        for r, c in candidates[:180]:
            box = (r, c, r + fh - 1, c + fw - 1)
            if all(other is None or _separated(box, other) for other in boxes):
                boxes[i] = box
                if search(k + 1):
                    return True
                boxes[i] = None
        return False

    if not search(0):
        return None
    return [box for box in boxes if box is not None]


def _centers(boxes: list[BBox]) -> list[tuple[float, float]]:
    return [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]


def _has_proximity_counterexample(boxes: list[BBox], pair_ids: list[int]) -> bool:
    centers = _centers(boxes)
    for i, (r, c) in enumerate(centers):
        distances = [((r - rr) ** 2 + (c - cc) ** 2, j) for j, (rr, cc) in enumerate(centers) if j != i]
        nearest = min(distances)[1]
        if pair_ids[nearest] != pair_ids[i]:
            return True
    return False


def _expected(grid: np.ndarray, specs: list[dict], bg: int) -> np.ndarray:
    """Independent construction oracle using generator metadata, not frame discovery."""
    out = grid.copy()
    groups: dict[int, list[int]] = defaultdict(list)
    for i, spec in enumerate(specs):
        groups[spec["pair"]].append(i)
    for indices in groups.values():
        a, b = indices
        for dst_i, src_i in ((a, b), (b, a)):
            r0, c0, r1, c1 = specs[dst_i]["box"]
            patch = out[r0 + 1 : r1, c0 + 1 : c1]
            patch[:] = bg
            patch[specs[src_i]["mask"]] = specs[src_i]["frame_color"]
    return out


def _validate_semantics(grid: np.ndarray, specs: list[dict], bg: int) -> bool:
    expected = _expected(grid, specs, bg)
    try:
        actual = main(grid)
        found = _find_frames(grid, bg)
    except (AssertionError, ValueError):
        return False
    if len(found) != len(specs) or not np.array_equal(actual, expected) or np.array_equal(actual, grid):
        return False

    keys = Counter(spec["size"] for spec in specs)
    if not keys or set(keys.values()) != {2}:
        return False
    if any(np.array_equal(specs[2 * p]["mask"], specs[2 * p + 1]["mask"]) for p in range(len(specs) // 2)):
        return False
    if any(specs[2 * p]["frame_color"] == specs[2 * p + 1]["frame_color"] for p in range(len(specs) // 2)):
        return False

    # Motif colors must be semantically inert: replacing all of them leaves the answer unchanged.
    used = {bg} | {spec["frame_color"] for spec in specs}
    replacement = next((c for c in range(10) if c not in used), None)
    if replacement is None:
        return False
    recolored = grid.copy()
    for spec in specs:
        r0, c0, r1, c1 = spec["box"]
        cavity = recolored[r0 + 1 : r1, c0 + 1 : c1]
        cavity[spec["mask"]] = replacement
    if not np.array_equal(main(recolored), expected):
        return False

    # A visible proximity shortcut must disagree with the true size-keyed operation.
    boxes = [spec["box"] for spec in specs]
    pair_ids = [spec["pair"] for spec in specs]
    if not _has_proximity_counterexample(boxes, pair_ids):
        return False
    centers = _centers(boxes)
    wrong = grid.copy()
    changed_wrong_pair = False
    for i, (r, c) in enumerate(centers):
        distances = [((r - rr) ** 2 + (c - cc) ** 2, j) for j, (rr, cc) in enumerate(centers) if j != i]
        j = min(distances)[1]
        if pair_ids[j] != pair_ids[i] and specs[i]["size"] == specs[j]["size"]:
            return False
        r0, c0, r1, c1 = specs[i]["box"]
        cavity = wrong[r0 + 1 : r1, c0 + 1 : c1]
        cavity[:] = bg
        source_mask = specs[j]["mask"]
        hh, ww = cavity.shape
        clipped = np.zeros((hh, ww), dtype=bool)
        sh, sw = source_mask.shape
        clipped[: min(hh, sh), : min(ww, sw)] = source_mask[: min(hh, sh), : min(ww, sw)]
        cavity[clipped] = specs[j]["frame_color"]
        changed_wrong_pair |= pair_ids[j] != pair_ids[i]
    return changed_wrong_pair and not np.array_equal(wrong, expected)


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice((2, 2, 3))
    # Include a same-area/different-dimensions decoy so the actual key must be (height, width).
    equal_area_choices = [
        (a, b) for i, a in enumerate(_SIZE_POOL) for b in _SIZE_POOL[i + 1 :]
        if a[0] * a[1] == b[0] * b[1]
    ]
    first, second = random.choice(equal_area_choices)
    pair_sizes = [first, second]
    if pair_count == 3:
        pair_sizes.append(random.choice([size for size in _SIZE_POOL if size not in pair_sizes]))
    random.shuffle(pair_sizes)
    sizes = [size for size in pair_sizes for _ in range(2)]
    pair_ids = [p for p in range(pair_count) for _ in range(2)]

    H = random.randint(22 if pair_count == 3 else 18, 30)
    W = random.randint(22 if pair_count == 3 else 18, 30)
    boxes = _place_boxes(H, W, sizes)
    if boxes is None or not _has_proximity_counterexample(boxes, pair_ids):
        return None

    bg = random.randrange(10)
    palette = [c for c in range(10) if c != bg]
    # Partners differ, while color 0 of every pair repeats across unequal cavity sizes.
    repeated = random.choice(palette)
    frame_colors: list[int] = []
    for _ in range(pair_count):
        frame_colors.extend((repeated, random.choice([c for c in palette if c != repeated])))

    specs: list[dict] = []
    grid = np.full((H, W), bg, dtype=int)
    for i, (box, size) in enumerate(zip(boxes, sizes)):
        r0, c0, r1, c1 = box
        frame_color = frame_colors[i]
        for cell in _rect_border(box):
            grid[cell] = frame_color
        mask = _grow_motif(*size)
        if i % 2 == 1 and np.array_equal(mask, specs[-1]["mask"]):
            return None
        motif_choices = [c for c in palette if c != frame_color]
        motif_color = random.choice(motif_choices)
        cavity = grid[r0 + 1 : r1, c0 + 1 : c1]
        cavity[mask] = motif_color
        specs.append(
            {
                "pair": pair_ids[i],
                "size": size,
                "box": box,
                "mask": mask,
                "frame_color": frame_color,
                "motif_color": motif_color,
            }
        )

    if _background(grid) != bg or not _validate_semantics(grid, specs, bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid equal-cavity motif-swap puzzle")
