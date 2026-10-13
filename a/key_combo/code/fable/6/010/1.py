from __future__ import annotations

from collections import Counter
import random

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    """Background color = most common value in the grid."""
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    """Return sorted colors present in the grid."""
    cols = set(map(int, grid.flat))
    if exclude is not None:
        cols -= set(map(int, exclude))
    return sorted(cols)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the inclusive bbox of a boolean mask."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def frame_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    """Return the one-cell-thick border mask of a rectangle bbox."""
    r0, c0, r1, c1 = map(int, box)
    assert r0 < r1 and c0 < c1
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def frame_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    """Return the border cells of a rectangle bbox."""
    r0, c0, r1, c1 = map(int, box)
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def on_frame_border(rc: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    """Return whether a coord lies on the rectangle border."""
    r, c = map(int, rc)
    r0, c0, r1, c1 = map(int, box)
    return ((r == r0 or r == r1) and c0 <= c <= c1) or ((c == c0 or c == c1) and r0 <= r <= r1)


def neighbors8(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    """Return the in-bounds 8-neighborhood of a cell."""
    out: list[tuple[int, int]] = []
    h, w = map(int, shape)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
    return out


def _candidate_score(coords: list[tuple[int, int]], box: tuple[int, int, int, int]) -> int:
    """Count visible same-color pixels that lie on the candidate frame border."""
    return sum(1 for rc in coords if on_frame_border(rc, box))


def best_rectangle(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    """Choose the unique rectangle whose border hits the most same-color pixels."""
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    best_boxes: list[tuple[int, int, int, int]] = []
    best_score = -1
    for i, r0 in enumerate(rows):
        for r1 in rows[i + 1 :]:
            if r1 - r0 < 2:
                continue
            for j, c0 in enumerate(cols):
                for c1 in cols[j + 1 :]:
                    if c1 - c0 < 2:
                        continue
                    box = (int(r0), int(c0), int(r1), int(c1))
                    score = _candidate_score(coords, box)
                    if score > best_score:
                        best_score = score
                        best_boxes = [box]
                    elif score == best_score:
                        best_boxes.append(box)
    assert best_score > 0, "each frame color must support at least one rectangle"
    unique = sorted(set(best_boxes))
    assert len(unique) == 1, "best border-supporting rectangle must be unique"
    return unique[0]


def unique_topo_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    """Return the unique bottom-to-top topological order implied by edges."""
    above: dict[int, set[int]] = {color: set() for color in colors}
    indegree: dict[int, int] = {color: 0 for color in colors}
    for lower, upper in edges:
        assert lower != upper
        if upper not in above[lower]:
            above[lower].add(upper)
            indegree[upper] += 1

    order: list[int] = []
    placed: set[int] = set()
    while len(order) < len(colors):
        available = [color for color in colors if color not in placed and indegree[color] == 0]
        assert len(available) == 1, "overlaps must determine a unique layer order"
        color = available[0]
        order.append(color)
        placed.add(color)
        for upper in above[color]:
            indegree[upper] -= 1
    return order


def reconstruct_frames(grid: np.ndarray) -> tuple[int, list[int], dict[int, tuple[int, int, int, int]], dict[int, np.ndarray]]:
    """Reconstruct the true rectangle bbox for each foreground color."""
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    colors = colors_present(g, exclude={bg})
    boxes: dict[int, tuple[int, int, int, int]] = {}
    masks: dict[int, np.ndarray] = {}
    for color in colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        box = best_rectangle(coords)
        mask = frame_mask(g.shape, box)
        assert any(mask[r, c] for r, c in coords), "reconstructed frame must explain visible pixels"
        boxes[color] = box
        masks[color] = mask
    return bg, colors, boxes, masks


def recover_order(grid: np.ndarray, colors: list[int], masks: dict[int, np.ndarray]) -> list[int]:
    """Recover the unique bottom-to-top order from reconstructed frame overlaps."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    edges: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            covering = [color for color in colors if masks[color][r, c]]
            if len(covering) < 2:
                continue
            visible = int(g[r, c])
            assert visible in covering, "noise must not occupy a true overlap cell"
            for color in covering:
                if color != visible:
                    edges.add((color, visible))
    assert edges, "at least one overlap is required to infer the order"
    return unique_topo_order(colors, edges)


def render_frames(shape: tuple[int, int], bg: int, order: list[int], masks: dict[int, np.ndarray]) -> np.ndarray:
    """Render full frames in the given bottom-to-top order."""
    out = np.full(shape, int(bg), dtype=int)
    for color in order:
        out[masks[color]] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct all partially occluded hollow rectangular frames, then redraw them with the stacking order reversed.

    Input structure:
    - The background contains several one-cell-thick hollow rectangular frames in distinct foreground colors.
    - Frames overlap, and each overlap cell shows only the color of the frame that is higher in the original stack.
    - Each frame color also has a few isolated same-color noise pixels that do not lie on that frame's true border.

    Rule:
    - For each foreground color, choose the unique rectangle whose border contains the maximum number of that color's visible pixels; pixels of that color outside the chosen border are noise and are discarded.
    - Use the reconstructed full frame borders to infer the unique original bottom-to-top layer order: at every cell covered by multiple reconstructed frames, the visible color is above every other covering frame.
    - Redraw the complete frames on a same-size blank background using the reversed layer order.

    Key insight:
    - The input is only one projection of a hidden full stack: first denoise each color to recover its true frame, then use overlap visibility constraints to recover the original order, and only then can the inverted-order rendering be produced.
    """
    g = np.asarray(grid, dtype=int)
    bg, colors, _boxes, masks = reconstruct_frames(g)
    order = recover_order(g, colors, masks)
    return render_frames(g.shape, bg, list(reversed(order)), masks)


GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (16, 24)
FRAME_COUNT_CHOICES = (3, 4)
FRAME_H_RANGE = (4, 10)
FRAME_W_RANGE = (4, 10)
REQUIRED_PREV_OVERLAPS = (2, 4)
ALLOWED_OLDER_OVERLAPS = (0, 2, 4)
NOISE_COUNT_CHOICES = (1, 2)
GEN_TRIES = 500
PLACE_TRIES = 250
ALL_COLORS = list(range(10))


def random_frame_size(h: int, w: int) -> tuple[int, int]:
    """Sample a frame height and width that fit the canvas."""
    max_h = min(FRAME_H_RANGE[1], h - 2)
    max_w = min(FRAME_W_RANGE[1], w - 2)
    fh = random.randint(FRAME_H_RANGE[0], max_h)
    fw = random.randint(FRAME_W_RANGE[0], max_w)
    return fh, fw


def place_first_frame(h: int, w: int) -> tuple[int, int, int, int]:
    """Sample the first frame anywhere on the canvas."""
    fh, fw = random_frame_size(h, w)
    r0 = random.randint(0, h - fh)
    c0 = random.randint(0, w - fw)
    return r0, c0, r0 + fh - 1, c0 + fw - 1


def place_next_frame(
    h: int,
    w: int,
    prev_box: tuple[int, int, int, int],
    prior_cell_sets: list[set[tuple[int, int]]],
) -> tuple[tuple[int, int, int, int], set[tuple[int, int]]] | None:
    """Sample one frame that crosses the previous frame and stays compatible with older frames."""
    pr0, pc0, pr1, pc1 = map(int, prev_box)
    for _ in range(PLACE_TRIES):
        fh, fw = random_frame_size(h, w)
        r_min = max(0, pr0 - fh + 1)
        r_max = min(h - fh, pr1)
        c_min = max(0, pc0 - fw + 1)
        c_max = min(w - fw, pc1)
        if r_min > r_max or c_min > c_max:
            continue
        r0 = random.randint(r_min, r_max)
        c0 = random.randint(c_min, c_max)
        box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        cells = frame_cells(box)
        if len(cells & prior_cell_sets[-1]) not in REQUIRED_PREV_OVERLAPS:
            continue
        if any(len(cells & older) not in ALLOWED_OLDER_OVERLAPS for older in prior_cell_sets[:-1]):
            continue
        return box, cells
    return None


def place_frame_stack(h: int, w: int, n: int) -> tuple[list[tuple[int, int, int, int]], list[set[tuple[int, int]]]] | None:
    """Build a chain of overlapping frames with a unique order."""
    first = place_first_frame(h, w)
    boxes = [first]
    cell_sets = [frame_cells(first)]
    for _ in range(1, n):
        placed = place_next_frame(h, w, boxes[-1], cell_sets)
        if placed is None:
            return None
        box, cells = placed
        boxes.append(box)
        cell_sets.append(cells)
    return boxes, cell_sets


def visible_bboxes_match(grid: np.ndarray, colors: list[int], boxes: list[tuple[int, int, int, int]]) -> bool:
    """Check that each frame still exposes its true bbox before noise is added."""
    for color, box in zip(colors, boxes):
        if bbox(grid == color) != box:
            return False
    return True


def random_layer_order(bg: int, n: int) -> list[int]:
    """Sample distinct frame colors."""
    palette = [color for color in ALL_COLORS if color != bg]
    colors = random.sample(palette, n)
    random.shuffle(colors)
    return colors


def raw_bbox_of_color(grid: np.ndarray, color: int) -> tuple[int, int, int, int] | None:
    """Return the visible bbox of one color, including noise."""
    return bbox(grid == color)


def can_place_noise(
    grid: np.ndarray,
    r: int,
    c: int,
    bg: int,
    all_boxes: list[tuple[int, int, int, int]],
) -> bool:
    """Require a completely isolated background cell outside all frame bboxes."""
    if int(grid[r, c]) != bg:
        return False
    for box in all_boxes:
        r0, c0, r1, c1 = map(int, box)
        if r0 <= r <= r1 and c0 <= c <= c1:
            return False
    if any(int(grid[rr, cc]) != bg for rr, cc in neighbors8(grid.shape, r, c)):
        return False
    return True


def add_noise(
    base: np.ndarray,
    colors: list[int],
    boxes: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]] | None:
    """Add isolated same-color noise pixels that sit outside each true frame bbox."""
    g = base.copy()
    bg = int(get_bg(base))
    placed: dict[int, list[tuple[int, int]]] = {color: [] for color in colors}
    h, w = g.shape

    for color, box in zip(colors, boxes):
        r0, c0, r1, c1 = map(int, box)
        need = random.choice(NOISE_COUNT_CHOICES)
        candidates = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if can_place_noise(g, r, c, bg, boxes)
            and not (r0 <= r <= r1 and c0 <= c <= c1)
            and r not in (r0, r1)
            and c not in (c0, c1)
        ]
        random.shuffle(candidates)
        if len(candidates) < need:
            return None
        for r, c in candidates[:need]:
            g[r, c] = int(color)
            placed[color].append((int(r), int(c)))
    return g, placed


def overlap_edges(colors: list[int], cell_sets: list[set[tuple[int, int]]]) -> set[tuple[int, int]]:
    """Return true layer constraints induced by the planted frame order."""
    edges: set[tuple[int, int]] = set()
    for i, lower in enumerate(colors):
        for j in range(i + 1, len(colors)):
            upper = colors[j]
            if cell_sets[i] & cell_sets[j]:
                edges.add((int(lower), int(upper)))
    return edges


def validate_semantics(
    inp: np.ndarray,
    expected: np.ndarray,
    true_order: list[int],
    true_boxes: list[tuple[int, int, int, int]],
    noises: dict[int, list[tuple[int, int]]],
) -> bool:
    """Reject samples that allow common shortcuts or violate the task semantics."""
    try:
        bg, colors, boxes, masks = reconstruct_frames(inp)
        if colors != sorted(true_order):
            return False
        true_by_color = {color: box for color, box in zip(true_order, true_boxes)}
        for color in colors:
            if boxes[color] != true_by_color[color]:
                return False
            if raw_bbox_of_color(inp, color) == true_by_color[color]:
                return False
            if not noises[color]:
                return False
            if not all(not on_frame_border(rc, true_by_color[color]) for rc in noises[color]):
                return False
        order = recover_order(inp, colors, masks)
        if order != true_order:
            return False
        if unique_topo_order(true_order, overlap_edges(true_order, [frame_cells(true_by_color[c]) for c in true_order])) != true_order:
            return False
        out = main(inp)
        if not np.array_equal(out, expected):
            return False
        if np.array_equal(out, inp):
            return False
        return True
    except Exception:
        return False


def gen_attempt() -> np.ndarray | None:
    """Generate one valid puzzle instance."""
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n = random.choice(FRAME_COUNT_CHOICES)
    bg = random.choice(ALL_COLORS)
    true_order = random_layer_order(bg, n)

    placed = place_frame_stack(h, w, n)
    if placed is None:
        return None
    boxes, cell_sets = placed

    base = np.full((h, w), int(bg), dtype=int)
    for color, cells in zip(true_order, cell_sets):
        rows, cols = zip(*cells)
        base[rows, cols] = int(color)

    if not visible_bboxes_match(base, true_order, boxes):
        return None

    if unique_topo_order(true_order, overlap_edges(true_order, cell_sets)) != true_order:
        return None

    added = add_noise(base, true_order, boxes)
    if added is None:
        return None
    inp, noises = added

    masks = {color: frame_mask(base.shape, box) for color, box in zip(true_order, boxes)}
    expected = render_frames(base.shape, bg, list(reversed(true_order)), masks)
    if not validate_semantics(inp, expected, true_order, boxes, noises):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a random valid input grid for the task."""
    for _ in range(GEN_TRIES):
        grid = gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
