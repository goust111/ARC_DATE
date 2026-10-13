from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


GREY = 5
BG = 0
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_perimeter(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells.update((r1, c) for c in range(c0, c1 + 1))
    cells.update((r, c0) for r in range(r0 + 1, r1))
    cells.update((r, c1) for r in range(r0 + 1, r1))
    return cells


def _visible_frame_boxes(g: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    bg = _get_bg(g)
    colors = [int(c) for c in np.unique(g) if int(c) not in {bg, GREY}]
    boxes: dict[int, tuple[int, int, int, int]] = {}

    for color in colors:
        coords = np.argwhere(g == int(color))
        if coords.size == 0:
            continue
        box = _bbox_coords(coords)
        r0, c0, r1, c1 = box
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame colors must form rectangular borders"
        perim = _rect_perimeter(box)
        assert all((int(r), int(c)) in perim for r, c in coords), "visible frame pixels must lie on one rectangle"

        sides = [
            any(int(g[r0, c]) == color for c in range(c0, c1 + 1)),
            any(int(g[r1, c]) == color for c in range(c0, c1 + 1)),
            any(int(g[r, c0]) == color for r in range(r0, r1 + 1)),
            any(int(g[r, c1]) == color for r in range(r0, r1 + 1)),
        ]
        assert all(sides), "each frame side must remain visible enough to recover its bbox"
        boxes[color] = box

    assert len(boxes) >= 2, "expected several colored rectangle frames"
    return boxes


def _occlusion_chain(g: np.ndarray, boxes: dict[int, tuple[int, int, int, int]]) -> list[int]:
    colors = list(boxes)
    perims = {color: _rect_perimeter(box) for color, box in boxes.items()}
    edges: set[tuple[int, int]] = set()

    for i, a in enumerate(colors):
        for b in colors[i + 1:]:
            relation: set[tuple[int, int]] = set()
            for r, c in perims[a] & perims[b]:
                visible = int(g[r, c])
                if visible == a:
                    relation.add((b, a))
                elif visible == b:
                    relation.add((a, b))
                else:
                    assert False, "frame crossings must show one of the two crossing frame colors"
            assert len(relation) <= 1, "one pair of frames cannot imply conflicting occlusion directions"
            edges.update(relation)

    assert edges, "expected at least one visible frame crossing"

    children: dict[int, set[int]] = {color: set() for color in colors}
    indeg: dict[int, int] = {color: 0 for color in colors}
    for src, dst in edges:
        if dst not in children[src]:
            children[src].add(dst)
            indeg[dst] += 1

    zeros = deque([color for color in colors if indeg[color] == 0])
    chain: list[int] = []
    while zeros:
        assert len(zeros) == 1, "occlusion relations must determine one color chain"
        color = zeros.popleft()
        chain.append(color)
        for nxt in children[color]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                zeros.append(nxt)

    assert len(chain) == len(colors), "occlusion chain must visit every frame color exactly once"
    return chain


def _marker_order(g: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(g == GREY)
    assert coords.size > 0, "expected grey marker pixels"
    r0, c0, r1, c1 = _bbox_coords(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    pts = [(int(r), int(c)) for r, c in coords]
    if height >= width:
        return sorted(pts, key=lambda p: (p[0], p[1]))
    return sorted(pts, key=lambda p: (p[1], p[0]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the occlusion order of overlapping rectangular frames and cycle those colors over grey markers.

    Input structure:
    - A dominant background contains several one-pixel-thick, differently
      colored rectangular frames whose borders cross.
    - At each frame crossing, only one frame color is visible; that visible
      cell records the directed relation from the hidden frame to the visible frame.
    - Away from all frame borders, isolated grey single-pixel markers have a
      marker bounding box with a definite long axis.

    Rule:
    - Recover every frame rectangle from the visible pixels of its color, then
      inspect shared border cells to build hidden-to-visible occlusion edges.
    - The edges must have a unique topological chain that uses each frame color once.
    - Leave all rectangular frame pixels exactly as they appear in the input.
    - If the grey-marker bounding box is at least as tall as it is wide, sort
      markers in row-major order; otherwise sort them in column-major order.
      Recolor the sorted markers by cycling through the chain colors.

    Key insight:
    - The overlapped frames are not redrawn; their crossings are a visible
      ordering table, and the grey markers read that table in the long-axis scan order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert _get_bg(g) != GREY, "grey is reserved for markers"

    boxes = _visible_frame_boxes(g)
    chain = _occlusion_chain(g, boxes)
    markers = _marker_order(g)

    out = g.copy()
    for i, (r, c) in enumerate(markers):
        out[r, c] = int(chain[i % len(chain)])
    return out


# ---------------- generator ----------------

GRID_SIZE_RANGE = (23, 30)
FRAME_COUNT_RANGE = (3, 5)
FRAME_SIZE_RANGE = (7, 14)
GEN_ATTEMPTS = 500
RECT_ATTEMPTS = 800

_CALL_COUNT = 0


def _intersection_cells(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    return _rect_perimeter(a) & _rect_perimeter(b)


def _has_collinear_side(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar0 in (br0, br1) or ar1 in (br0, br1) or ac0 in (bc0, bc1) or ac1 in (bc0, bc1)


def _random_rect(zone: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    zr0, zc0, zr1, zc1 = zone
    zh = zr1 - zr0 + 1
    zw = zc1 - zc0 + 1
    min_size, max_size = FRAME_SIZE_RANGE
    if zh < min_size or zw < min_size:
        return None
    h = random.randint(min_size, min(max_size, zh))
    w = random.randint(min_size, min(max_size, zw))
    r0 = random.randint(zr0, zr1 - h + 1)
    c0 = random.randint(zc0, zc1 - w + 1)
    return r0, c0, r0 + h - 1, c0 + w - 1


def _no_triple_crossing(boxes: list[tuple[int, int, int, int]]) -> bool:
    counts: Counter[tuple[int, int]] = Counter()
    for box in boxes:
        counts.update(_rect_perimeter(box))
    return all(n <= 2 for n in counts.values())


def _chain_boxes(zone: tuple[int, int, int, int], count: int) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    first = _random_rect(zone)
    if first is None:
        return None
    boxes.append(first)

    for _ in range(1, count):
        prev = boxes[-1]
        placed = None
        for _attempt in range(RECT_ATTEMPTS):
            cand = _random_rect(zone)
            if cand is None:
                return None
            if any(_has_collinear_side(cand, old) for old in boxes):
                continue
            if not _intersection_cells(cand, prev):
                continue
            trial = boxes + [cand]
            if not _no_triple_crossing(trial):
                continue
            placed = cand
            break
        if placed is None:
            return None
        boxes.append(placed)

    return boxes


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0:c1 + 1] = color
    grid[r1, c0:c1 + 1] = color
    grid[r0:r1 + 1, c0] = color
    grid[r0:r1 + 1, c1] = color


def _render_frames(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]], colors: list[int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for box, color in zip(boxes, colors):
        _draw_frame(grid, box, color)
    return grid


def _frame_visibility_ok(grid: np.ndarray, boxes: list[tuple[int, int, int, int]], colors: list[int]) -> bool:
    for box, color in zip(boxes, colors):
        coords = np.argwhere(grid == int(color))
        if coords.size == 0 or _bbox_coords(coords) != box:
            return False
        r0, c0, r1, c1 = box
        if not all((int(r), int(c)) in _rect_perimeter(box) for r, c in coords):
            return False
        side_counts = [
            sum(int(grid[r0, c]) == color for c in range(c0, c1 + 1)),
            sum(int(grid[r1, c]) == color for c in range(c0, c1 + 1)),
            sum(int(grid[r, c0]) == color for r in range(r0, r1 + 1)),
            sum(int(grid[r, c1]) == color for r in range(r0, r1 + 1)),
        ]
        if min(side_counts) < 2:
            return False
    return True


def _scan_order(coords: list[tuple[int, int]], row_major: bool) -> list[tuple[int, int]]:
    if row_major:
        return sorted(coords, key=lambda p: (p[0], p[1]))
    return sorted(coords, key=lambda p: (p[1], p[0]))


def _marker_coords(shape: tuple[int, int], branch: str, count: int) -> list[tuple[int, int]] | None:
    H, W = shape
    if branch == "row":
        height = random.choice([9, 11, 13])
        width = random.choice([3, 5])
        if height > H - 2 or width > 5:
            return None
        r0 = random.randint(1, H - height - 1)
        c0 = random.randint(1, max(1, 5 - width + 1))
    else:
        height = random.choice([3, 5])
        width = random.choice([9, 11, 13])
        if height > 5 or width > W - 2:
            return None
        r0 = random.randint(1, max(1, 5 - height + 1))
        c0 = random.randint(1, W - width - 1)

    r1 = r0 + height - 1
    c1 = c0 + width - 1
    lattice = [(r0 + dr, c0 + dc) for dr in range(0, height, 2) for dc in range(0, width, 2)]
    forced = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    if any(p not in lattice for p in forced):
        return None
    if count > len(lattice):
        return None

    remaining = [p for p in lattice if p not in forced]
    random.shuffle(remaining)
    coords = forced + remaining[:count - len(forced)]
    random.shuffle(coords)

    row_order = _scan_order(coords, True)
    col_order = _scan_order(coords, False)
    if row_order == col_order:
        return None
    return coords


def _place_markers(grid: np.ndarray, coords: list[tuple[int, int]]) -> None:
    H, W = grid.shape
    for r, c in coords:
        if int(grid[r, c]) != BG:
            raise ValueError("marker collides with frame")
        for dr, dc in NEIGHBORS8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) != BG:
                raise ValueError("marker is not isolated")
        grid[r, c] = GREY


def _expected_output(inp: np.ndarray, colors: list[int]) -> np.ndarray:
    out = inp.copy()
    for i, (r, c) in enumerate(_marker_order(inp)):
        out[r, c] = colors[i % len(colors)]
    return out


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    colors: list[int],
    branch: str,
) -> bool:
    try:
        if not np.array_equal(main(inp), expected):
            return False
    except Exception:
        return False

    try:
        recovered_boxes = _visible_frame_boxes(inp)
        if _occlusion_chain(inp, recovered_boxes) != colors:
            return False
    except Exception:
        return False

    marker_pts = [(int(r), int(c)) for r, c in np.argwhere(inp == GREY)]
    r0, c0, r1, c1 = _bbox_coords(np.array(marker_pts, dtype=int))
    row_branch = (r1 - r0 + 1) >= (c1 - c0 + 1)
    if (branch == "row") != row_branch:
        return False

    wrong_order = _scan_order(marker_pts, row_major=not row_branch)
    wrong = inp.copy()
    for i, (r, c) in enumerate(wrong_order):
        wrong[r, c] = colors[i % len(colors)]
    if np.array_equal(wrong, expected):
        return False

    bbox_scan_colors = [color for _box, color in sorted(zip(boxes, colors), key=lambda bc: (bc[0][0], bc[0][1]))]
    if bbox_scan_colors == colors:
        return False
    first_seen = []
    seen = set()
    for value in inp.reshape(-1):
        v = int(value)
        if v in colors and v not in seen:
            seen.add(v)
            first_seen.append(v)
    if first_seen == colors:
        return False

    return True


def _gen_attempt(branch: str) -> np.ndarray | None:
    H = random.randint(*GRID_SIZE_RANGE)
    W = random.randint(*GRID_SIZE_RANGE)
    count = random.randint(*FRAME_COUNT_RANGE)

    if branch == "row":
        zone = (1, 8, H - 2, W - 2)
    else:
        zone = (8, 1, H - 2, W - 2)

    boxes = _chain_boxes(zone, count)
    if boxes is None:
        return None
    colors = random.sample(FRAME_COLORS, count)
    if colors == sorted(colors):
        return None

    grid = _render_frames((H, W), boxes, colors)
    if not _frame_visibility_ok(grid, boxes, colors):
        return None

    max_markers = 10 if count >= 4 else 9
    marker_count = random.randint(count + 2, min(max_markers, count + 5))
    coords = _marker_coords((H, W), branch, marker_count)
    if coords is None:
        return None
    try:
        _place_markers(grid, coords)
    except ValueError:
        return None

    expected = _expected_output(grid, colors)
    if not _semantic_checks(grid, expected, boxes, colors, branch):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _CALL_COUNT
    branch = "row" if _CALL_COUNT % 3 != 1 else "col"
    _CALL_COUNT += 1

    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(branch)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid occlusion-chain recolor input")
