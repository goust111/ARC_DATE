import numpy as np
from collections import deque


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
RNG = np.random.default_rng()

DIR_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_legend(grid: np.ndarray) -> tuple[int, list[int]]:
    h, _ = grid.shape
    for r in range(h - 1, -1, -1):
        vals = [int(v) for v in grid[r] if int(v) not in (BG, GREY)]
        if len(vals) >= 4 and len(set(vals)) >= 4:
            legend: list[int] = []
            prev = None
            for v in vals:
                if v != prev:
                    legend.append(v)
                    prev = v
            return r, legend
    raise AssertionError("legend row not found")


def _extract_pieces(grid: np.ndarray, legend_row: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    pieces: list[dict[str, object]] = []
    for r in range(h):
        if r == legend_row:
            continue
        for c in range(w):
            color = int(grid[r, c])
            if color in (BG, GREY) or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and nr != legend_row
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            pieces.append({"color": color, "coords": tuple(sorted(comp))})
    return pieces


def _attach_arrows(
    grid: np.ndarray,
    legend_row: int,
    pieces: list[dict[str, object]],
) -> None:
    coord_to_piece: dict[tuple[int, int], int] = {}
    for idx, piece in enumerate(pieces):
        for cell in piece["coords"]:
            coord_to_piece[cell] = idx

    arrow_mask = (grid == GREY).copy()
    arrow_mask[legend_row, :] = False
    piece_to_dir: dict[int, tuple[int, int]] = {}

    for comp in _mask_components(arrow_mask):
        if len(comp) != 2:
            continue
        a, b = sorted(comp)
        dr = b[0] - a[0]
        dc = b[1] - a[1]
        if abs(dr) + abs(dc) != 1:
            continue
        candidates = []
        for near, far in ((a, b), (b, a)):
            d = (far[0] - near[0], far[1] - near[1])
            anchor = (near[0] - d[0], near[1] - d[1])
            if anchor in coord_to_piece:
                candidates.append((coord_to_piece[anchor], d))
        if len(candidates) != 1:
            continue
        pid, direction = candidates[0]
        piece_to_dir[pid] = direction

    assert len(piece_to_dir) == len(pieces), "each piece must have exactly one arrow"
    for idx, piece in enumerate(pieces):
        piece["direction"] = piece_to_dir[idx]


def _simulate_slide(
    pieces: list[dict[str, object]],
    play_h: int,
    play_w: int,
) -> list[set[tuple[int, int]]]:
    current = [set(piece["coords"]) for piece in pieces]
    directions = [piece["direction"] for piece in pieces]
    active = set(range(len(pieces)))
    stopped_cells: set[tuple[int, int]] = set()

    for _ in range(128):
        if not active:
            break

        proposals: dict[int, set[tuple[int, int]]] = {}
        moving = set()
        for idx in active:
            dr, dc = directions[idx]
            shifted = {(r + dr, c + dc) for r, c in current[idx]}
            proposals[idx] = shifted
            if all(0 <= r < play_h and 0 <= c < play_w for r, c in shifted) and not (shifted & stopped_cells):
                moving.add(idx)

        changed = True
        while changed:
            changed = False
            resting = active - moving
            resting_cells: set[tuple[int, int]] = set()
            for idx in resting:
                resting_cells.update(current[idx])

            blocked = {idx for idx in moving if proposals[idx] & resting_cells}
            if blocked:
                moving -= blocked
                changed = True
                continue

            seen: dict[tuple[int, int], int] = {}
            colliding: set[int] = set()
            for idx in moving:
                for cell in proposals[idx]:
                    other = seen.get(cell)
                    if other is None:
                        seen[cell] = idx
                    else:
                        colliding.add(idx)
                        colliding.add(other)
            if colliding:
                moving -= colliding
                changed = True

        resting = active - moving
        for idx in resting:
            stopped_cells.update(current[idx])

        if not moving:
            break

        for idx in moving:
            current[idx] = proposals[idx]
        active = moving
    else:
        raise AssertionError("sliding did not converge")

    return current


def _touching(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    if a & b:
        return True
    for r, c in a:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in b:
                return True
    return False


def _merge_color(group_colors: list[int], legend: list[int]) -> int:
    if len(group_colors) <= 1:
        return group_colors[0]

    index = {color: i for i, color in enumerate(legend)}
    positions = sorted({index[color] for color in group_colors if color in index})
    if len(positions) <= 1:
        return group_colors[0]

    n = len(legend)
    gaps = []
    for i, pos in enumerate(positions):
        nxt = positions[(i + 1) % len(positions)]
        gap = (nxt - pos) % n
        gaps.append((gap, i))

    largest_gap, gap_idx = max(gaps, key=lambda item: (item[0], -item[1]))
    start = positions[(gap_idx + 1) % len(positions)]
    arc_len = n - largest_gap
    mid = (start + (arc_len + 1) // 2) % n
    return legend[mid]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored fragment along its grey arrow and recolor touching end-groups by the bottom color ring.

    Input structure:
    - Background is black.
    - The play area contains several disconnected monochrome fragments, each with a grey length-2 arrow segment attached to one exposed side.
    - The bottom row contains a left-to-right color legend whose colors define a circular order.

    Rule:
    - Read each grey 2-cell segment as an orthogonal arrow: the fragment cell immediately behind the nearer grey cell is the owner, and the arrow direction points away from that fragment.
    - Remove the arrows, then update all fragments in synchronous 1-cell steps. A fragment moves only if its next translated shape stays inside the play area and avoids every fragment that is not moving in that same step; fragments that cannot move become fixed obstacles.
    - After motion ends, merge every 4-connected touching group of fragments. Keep the union of their original pixel positions, but recolor the whole group to the midpoint color of the shortest arc covering the member colors on the bottom legend ring; a singleton keeps its own color, and an exact halfway case rounds clockwise.

    Key insight:
    - The puzzle is a two-stage pipeline: first infer motion from detachable arrow markers and synchronous blocking, then use the bottom legend as a cyclic lookup table for the post-collision recolor.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    legend_row, legend = _find_legend(g)
    pieces = _extract_pieces(g, legend_row)
    _attach_arrows(g, legend_row, pieces)
    final_coords = _simulate_slide(pieces, legend_row, w)

    adjacency = [[] for _ in pieces]
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            if _touching(final_coords[i], final_coords[j]):
                adjacency[i].append(j)
                adjacency[j].append(i)

    groups: list[list[int]] = []
    seen = [False] * len(pieces)
    for i in range(len(pieces)):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        group: list[int] = []
        while stack:
            cur = stack.pop()
            group.append(cur)
            for nxt in adjacency[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        groups.append(group)

    out = np.full((h, w), BG, dtype=int)
    out[legend_row:, :] = g[legend_row:, :]
    for group in groups:
        color = _merge_color([int(pieces[idx]["color"]) for idx in group], legend)
        for idx in group:
            for r, c in final_coords[idx]:
                out[r, c] = color
    return out


def _sample_lane_counts(num_lanes: int) -> list[int]:
    for _ in range(64):
        counts = RNG.integers(1, 4, size=num_lanes).tolist()
        if any(c == 1 for c in counts) and any(c > 1 for c in counts):
            return [int(c) for c in counts]
    counts = [1] * num_lanes
    counts[0] = 2
    return counts


def _lane_colors(legend: list[int], count: int) -> list[int]:
    if count == 1:
        return [int(RNG.choice(legend))]

    n = len(legend)
    max_span = (n - 1) // 2
    span = int(RNG.integers(count - 1, max_span + 1))
    start = int(RNG.integers(0, n))
    offsets = sorted(int(v) for v in RNG.choice(np.arange(span + 1), size=count, replace=False))
    return [legend[(start + offset) % n] for offset in offsets]


def _paint_rect(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    color: int,
) -> bool:
    h, w = grid.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    if np.any(grid[top : top + height, left : left + width] != BG):
        return False
    grid[top : top + height, left : left + width] = color
    return True


def _paint_arrow(
    grid: np.ndarray,
    anchor: tuple[int, int],
    direction: tuple[int, int],
) -> bool:
    h, w = grid.shape
    for step in (1, 2):
        r = anchor[0] + direction[0] * step
        c = anchor[1] + direction[1] * step
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BG:
            return False
    for step in (1, 2):
        r = anchor[0] + direction[0] * step
        c = anchor[1] + direction[1] * step
        grid[r, c] = GREY
    return True


def _build_horizontal_scene(legend: list[int]) -> np.ndarray | None:
    num_lanes = int(RNG.integers(2, 5))
    counts = _sample_lane_counts(num_lanes)
    dirs = ["left", "right"]
    while len(dirs) < num_lanes:
        dirs.append("left" if RNG.random() < 0.5 else "right")
    RNG.shuffle(dirs)

    lanes = []
    width_need = len(legend)
    for lane_dir, count in zip(dirs, counts):
        heights = [int(RNG.integers(1, 3)) for _ in range(count)]
        widths = [int(RNG.integers(1, 4)) for _ in range(count)]
        runs = [int(RNG.integers(3, 6))]
        for _ in range(1, count):
            runs.append(runs[-1] + int(RNG.integers(3, 7)))
        lanes.append(
            {
                "dir": lane_dir,
                "count": count,
                "heights": heights,
                "widths": widths,
                "runs": runs,
                "colors": _lane_colors(legend, count),
            }
        )
        width_need = max(width_need, sum(widths) + runs[-1])

    gap = int(RNG.integers(1, 3))
    top_margin = int(RNG.integers(0, 3))
    play_h = top_margin + num_lanes * 2 + (num_lanes - 1) * gap + int(RNG.integers(0, 3))
    h = play_h + 1
    w = width_need + int(RNG.integers(1, 5))
    if h > 30 or w > 30:
        return None

    grid = np.zeros((h, w), dtype=int)
    legend_start = int(RNG.integers(0, w - len(legend) + 1))
    grid[h - 1, legend_start : legend_start + len(legend)] = legend

    lane_top = top_margin
    for lane in lanes:
        widths = lane["widths"]
        heights = lane["heights"]
        runs = lane["runs"]
        colors = lane["colors"]

        final_xs: list[int] = []
        if lane["dir"] == "right":
            cursor = w
            for width in widths:
                cursor -= width
                final_xs.append(cursor)
            initial_xs = [fx - run for fx, run in zip(final_xs, runs)]
            arrow_dir = DIR_TO_DELTA["right"]
            anchors = [(lane_top, x + width - 1) for x, width in zip(initial_xs, widths)]
        else:
            cursor = 0
            for width in widths:
                final_xs.append(cursor)
                cursor += width
            initial_xs = [fx + run for fx, run in zip(final_xs, runs)]
            arrow_dir = DIR_TO_DELTA["left"]
            anchors = [(lane_top, x) for x in initial_xs]

        for x, height, width, color, anchor in zip(initial_xs, heights, widths, colors, anchors):
            if not _paint_rect(grid, lane_top, x, height, width, color):
                return None
            if not _paint_arrow(grid, anchor, arrow_dir):
                return None
        lane_top += 2 + gap

    return grid


def _build_vertical_scene(legend: list[int]) -> np.ndarray | None:
    num_lanes = int(RNG.integers(2, 5))
    counts = _sample_lane_counts(num_lanes)
    dirs = ["up", "down"]
    while len(dirs) < num_lanes:
        dirs.append("up" if RNG.random() < 0.5 else "down")
    RNG.shuffle(dirs)

    lanes = []
    height_need = 4
    for lane_dir, count in zip(dirs, counts):
        heights = [int(RNG.integers(1, 4)) for _ in range(count)]
        widths = [int(RNG.integers(1, 3)) for _ in range(count)]
        runs = [int(RNG.integers(3, 6))]
        for _ in range(1, count):
            runs.append(runs[-1] + int(RNG.integers(3, 7)))
        lanes.append(
            {
                "dir": lane_dir,
                "count": count,
                "heights": heights,
                "widths": widths,
                "runs": runs,
                "colors": _lane_colors(legend, count),
            }
        )
        height_need = max(height_need, sum(heights) + runs[-1])

    gap = int(RNG.integers(1, 3))
    left_margin = int(RNG.integers(0, 3))
    w = left_margin + num_lanes * 2 + (num_lanes - 1) * gap + int(RNG.integers(1, 4))
    h = height_need + int(RNG.integers(1, 5)) + 1
    if h > 30 or w > 30 or w < len(legend):
        return None

    grid = np.zeros((h, w), dtype=int)
    legend_start = int(RNG.integers(0, w - len(legend) + 1))
    grid[h - 1, legend_start : legend_start + len(legend)] = legend
    play_h = h - 1

    lane_left = left_margin
    for lane in lanes:
        heights = lane["heights"]
        widths = lane["widths"]
        runs = lane["runs"]
        colors = lane["colors"]

        final_ys: list[int] = []
        if lane["dir"] == "down":
            cursor = play_h
            for height in heights:
                cursor -= height
                final_ys.append(cursor)
            initial_ys = [fy - run for fy, run in zip(final_ys, runs)]
            arrow_dir = DIR_TO_DELTA["down"]
            anchors = [(y + height - 1, lane_left) for y, height in zip(initial_ys, heights)]
        else:
            cursor = 0
            for height in heights:
                final_ys.append(cursor)
                cursor += height
            initial_ys = [fy + run for fy, run in zip(final_ys, runs)]
            arrow_dir = DIR_TO_DELTA["up"]
            anchors = [(y, lane_left) for y in initial_ys]

        for y, height, width, color, anchor in zip(initial_ys, heights, widths, colors, anchors):
            if not _paint_rect(grid, y, lane_left, height, width, color):
                return None
            if not _paint_arrow(grid, anchor, arrow_dir):
                return None
        lane_left += 2 + gap

    return grid


def _gen_attempt() -> np.ndarray | None:
    legend_len = int(RNG.integers(5, 8))
    legend = [int(v) for v in RNG.choice(PALETTE, size=legend_len, replace=False)]
    grid = _build_horizontal_scene(legend) if RNG.random() < 0.5 else _build_vertical_scene(legend)
    if grid is None:
        return None
    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after 400 attempts")
