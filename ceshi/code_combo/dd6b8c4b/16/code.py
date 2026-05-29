# Combo variant of dd6b8c4b (index 16)
# combo_spec: D11/0+D14/0
# Name: 裁剪到方块区域+框架与方块有间隔通道

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4

# Design analysis for D11/0 + D14/0
# a) Concrete behavior trace:
#    Original: a 3x3 destination block sits in the full 11x11 canvas and any nearby wall cells
#    are directly adjacent to that block. Reachable maroon cells are found by BFS from the block's
#    perimeter, moved into the block, and the output keeps the full canvas.
#    Variant: a closed magenta frame sits 1 or 2 orange cells away from the 3x3 block, creating
#    a visible three-layer layout: block -> orange channel -> frame. Example with gap=1:
#      input full grid contains a 7x7 framed region plus extra outside maroons;
#      BFS still starts from passable cells adjacent to the block, but now it spreads through the
#      inner channel rather than through exterior openings;
#      after moving the closest reachable maroons into the block, the result is cropped to the
#      frame's bounding box, so outside space and outside distractors disappear.
# b) Complexity calibration:
#    The key control variable is the channel width g in {1, 2}. Keeping it to two values makes the
#    visual rule stable enough for ARC while still yielding two output sizes (7x7 and 9x9). Maroons
#    are split into inner-channel reachable tokens and exterior distractors, so both reachability
#    and cropping remain visible without introducing extra object types.
# c) Broken assumptions from the original solver:
#    - Output shape is no longer equal to input shape.
#    - Wall cells are no longer adjacent to the destination; immediate neighbors of the block are
#      passable channel cells.
#    - Unreachable maroons can survive in the full grid but vanish from the final answer purely
#      because cropping removes the outer area.
#    The new solver therefore detects the enclosing rectangular frame explicitly and crops only at
#    the end, after applying the same distance-ordered movement logic inside the full grid.
# d) Placement legality:
#    The generator chooses only center/gap combinations whose frame stays strictly inside 11x11.
#    The frame is a 1-pixel rectangular border, the channel is the frame interior minus the 3x3
#    block, and outside distractors are sampled only from cells outside the frame.
# e) Visual observability:
#    Across examples, humans can see (i) the fixed 3x3 block, (ii) a 1-2 cell orange moat, and
#    (iii) an enclosing magenta frame. Output size changes with the moat width, so cropping is not
#    hidden. Reachable maroons are visibly the ones inside the frame; outside maroons disappear
#    because they fall outside the crop.
# f) Visual cleanliness:
#    The three layers do not overlap: the block uses green/red, the channel uses orange/maroon, and
#    the frame uses magenta. This keeps the transformed region legible after cropping.
# g) New solver assumptions audited against the generator:
#    - exactly one red center exists;
#    - the 3x3 block is fully in-bounds;
#    - all magenta cells belong to one rectangular border enclosing the block;
#    - immediate neighbors of the block contain at least one passable cell;
#    - floor is the unique color besides {2,3,6,9}.
#    The generator constructs every sample to satisfy these assumptions.

# ---- Fixed palette ----
GRID_H, GRID_W = 11, 11
FLOOR_COLOR, FRAME_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
MAROON = 9

# ---- Generation controls ----
CHANNEL_WIDTHS = (1, 2)
CENTER_OPTIONS = {
    1: [(4, 4), (4, 5), (4, 6), (5, 4), (5, 5), (5, 6), (6, 4), (6, 5), (6, 6)],
    2: [(5, 5)],
}
REACHABLE_TOTAL_RANGE = (2, 12)
OUTSIDE_DISTRACTOR_RANGE = (1, 6)
GEN_RETRIES = 400


def _dest_cells(cr: int, cc: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(cr - 1, cr + 2) for c in range(cc - 1, cc + 2)]


def _frame_bbox(cr: int, cc: int, gap: int) -> tuple[int, int, int, int]:
    pad = gap + 2
    return cr - pad, cc - pad, cr + pad, cc + pad


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _layout_from_center(
    cr: int, cc: int, gap: int
) -> tuple[list[tuple[int, int]], set[tuple[int, int]], tuple[int, int, int, int], list[tuple[int, int]], list[tuple[int, int]]]:
    dest_cells = _dest_cells(cr, cc)
    dest_set = set(dest_cells)
    r0, c0, r1, c1 = _frame_bbox(cr, cc, gap)
    frame_cells = _border_cells(r0, c0, r1, c1)
    channel_cells = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if (r, c) not in dest_set
    ]
    return dest_cells, dest_set, (r0, c0, r1, c1), frame_cells, channel_cells


def _bfs_distances(
    grid: np.ndarray,
    dest_cells: list[tuple[int, int]],
    passable: set[int],
) -> np.ndarray:
    h, w = grid.shape
    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    q = deque()
    for r, c in dest_cells:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) in passable and dist[rr, cc] == inf:
                dist[rr, cc] = 0
                q.append((rr, cc))
    assert q, "destination block must touch at least one passable channel cell"

    while q:
        r, c = q.popleft()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) in passable and dist[rr, cc] == inf:
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    return dist


def _find_frame_bbox(grid: np.ndarray, cr: int, cc: int) -> tuple[int, int, int, int]:
    frame_pts = np.argwhere(grid == FRAME_COLOR)
    assert len(frame_pts) >= 16, "expected a visible rectangular frame"
    r0, c0 = frame_pts.min(axis=0)
    r1, c1 = frame_pts.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert r0 < cr < r1 and c0 < cc < c1, "frame must enclose the red center"
    assert np.all(grid[r0, c0 : c1 + 1] == FRAME_COLOR)
    assert np.all(grid[r1, c0 : c1 + 1] == FRAME_COLOR)
    assert np.all(grid[r0 : r1 + 1, c0] == FRAME_COLOR)
    assert np.all(grid[r0 : r1 + 1, c1] == FRAME_COLOR)
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert FRAME_COLOR not in set(map(int, inner.flatten()))
    return r0, c0, r1, c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the closest reachable maroon pixels into the central 3x3 block, then crop to the enclosing frame.

    Input structure:
    - The canvas contains orange floor, scattered maroon tokens, a unique red center inside a 3x3 green block,
      and one magenta rectangular frame around that block.
    - The frame is not adjacent to the block: a 1-2 cell orange channel separates them, and maroons may appear
      inside that channel or outside the frame.

    Rule:
    - Flood-fill from passable cells adjacent to the 3x3 block through orange/maroon cells; maroons reached by
      this BFS are candidates.
    - Sort reachable maroons by distance to the block, breaking ties by row then column, and place the closest
      ones into the block in row-major order while erasing their source cells back to floor.
    - After the movement, discard the distant outer canvas and return only the frame's bounding rectangle, which
      therefore contains the block, the channel, and the frame.

    Key insight: The magenta border now defines both connectivity and the crop boundary; only maroons inside the
    frame's reachable interior can move, and everything outside that framed region is removed by the final crop.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    assert (h, w) == (GRID_H, GRID_W)

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must contain exactly one red center"
    cr, cc = map(int, centers[0])

    dest_cells = _dest_cells(cr, cc)
    assert all(0 <= r < h and 0 <= c < w for r, c in dest_cells), "3x3 destination block must be in-bounds"

    r0, c0, r1, c1 = _find_frame_bbox(grid, cr, cc)
    other_colors = sorted(set(map(int, grid.flatten())) - {RED_CENTER, DEST_COLOR, FRAME_COLOR, MAROON})
    assert other_colors == [FLOOR_COLOR], "expected a single floor color"
    floor = FLOOR_COLOR

    dist = _bfs_distances(grid, dest_cells, {floor, MAROON})
    reachable = [
        (int(dist[r, c]), int(r), int(c))
        for r, c in np.argwhere(grid == MAROON)
        if int(dist[r, c]) < 10**9
    ]
    reachable.sort()

    out = grid.copy()
    k = min(len(dest_cells), len(reachable))
    chosen = {(r, c) for _, r, c in reachable[:k]}
    for r, c in chosen:
        out[r, c] = floor
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON

    return out[r0 : r1 + 1, c0 : c1 + 1]


def _expected_output(
    inp: np.ndarray,
    dest_cells: list[tuple[int, int]],
    frame_box: tuple[int, int, int, int],
) -> np.ndarray:
    dist = _bfs_distances(inp, dest_cells, {FLOOR_COLOR, MAROON})
    reachable = [
        (int(dist[r, c]), int(r), int(c))
        for r, c in np.argwhere(inp == MAROON)
        if int(dist[r, c]) < 10**9
    ]
    reachable.sort()

    out = inp.copy()
    k = min(len(dest_cells), len(reachable))
    for _, r, c in reachable[:k]:
        out[r, c] = FLOOR_COLOR
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON

    r0, c0, r1, c1 = frame_box
    return out[r0 : r1 + 1, c0 : c1 + 1]


def _gen_attempt() -> np.ndarray | None:
    gap = random.choice(CHANNEL_WIDTHS)
    cr, cc = random.choice(CENTER_OPTIONS[gap])
    dest_cells, dest_set, frame_box, frame_cells, channel_cells = _layout_from_center(cr, cc, gap)
    r0, c0, r1, c1 = frame_box

    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in frame_cells:
        grid[r, c] = FRAME_COLOR
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[cr, cc] = RED_CENTER

    outside_cells = [
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if not (r0 <= r <= r1 and c0 <= c <= c1) and (r, c) not in dest_set
    ]

    max_reachable = min(len(channel_cells), REACHABLE_TOTAL_RANGE[1])
    if max_reachable < REACHABLE_TOTAL_RANGE[0]:
        return None
    reachable_total = random.randint(REACHABLE_TOTAL_RANGE[0], max_reachable)
    reachable_cells = random.sample(channel_cells, reachable_total)

    max_outside = min(len(outside_cells), OUTSIDE_DISTRACTOR_RANGE[1])
    min_outside = min(OUTSIDE_DISTRACTOR_RANGE[0], max_outside)
    if max_outside == 0:
        return None
    outside_total = random.randint(min_outside, max_outside)
    distractor_cells = random.sample(outside_cells, outside_total)

    for r, c in reachable_cells + distractor_cells:
        grid[r, c] = MAROON

    # Independent expectation: only maroons in the inner channel are reachable because the frame is closed.
    expected = _expected_output(grid, dest_cells, frame_box)
    try:
        result = main(grid)
    except Exception:
        return None
    if not np.array_equal(result, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")


if __name__ == "__main__":
    random.seed(0)
    np.set_printoptions(linewidth=200)
    sample = generate_input()
    print(sample)
    print(main(sample))
