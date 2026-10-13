import numpy as np


BLACK = 0
GRAY = 5
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)


def _inside(shape, pos):
    r, c = pos
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _add(pos, direction, k=1):
    return pos[0] + direction[0] * k, pos[1] + direction[1] * k


def _opposite(direction):
    return -direction[0], -direction[1]


def _perpendiculars(direction):
    if direction[0] != 0:
        return ((0, -1), (0, 1))
    return ((-1, 0), (1, 0))


def _find_terminal_frames(grid):
    h, w = grid.shape
    terminals = []
    border = set()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != BLACK:
                continue
            cells = []
            ok = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if int(grid[rr, cc]) != GRAY:
                        ok = False
                        break
                    cells.append((rr, cc))
                if not ok:
                    break
            if ok:
                terminals.append((r, c, tuple(cells)))
                border.update(cells)
    return terminals, border


def _find_source_blocks(grid):
    h, w = grid.shape
    sources = []
    source_pixels = set()
    for r in range(h - 2):
        for c in range(w - 2):
            color = int(grid[r + 1, c + 1])
            if color in (BLACK, GRAY):
                continue
            if np.all(grid[r : r + 3, c : c + 3] == color):
                cells = {
                    (rr, cc)
                    for rr in range(r, r + 3)
                    for cc in range(c, c + 3)
                }
                sources.append(((r + 1, c + 1), color, cells))
                source_pixels.update(cells)
    return sources, source_pixels


def _pipe_degree(pos, pipe_set):
    return sum(_add(pos, direction) in pipe_set for direction in DIRS)


def _marker_direction(grid, pos, source_pixels):
    for direction in DIRS:
        q = _add(pos, direction)
        if not _inside(grid.shape, q) or q in source_pixels:
            continue
        value = int(grid[q])
        if value not in (BLACK, GRAY):
            return direction
    return None


def _marked_branch_step(grid, pos, direction, pipe_set, source_pixels):
    q = _add(pos, direction)
    if q in pipe_set:
        return q
    q2 = _add(pos, direction, 2)
    if (
        _inside(grid.shape, q)
        and _inside(grid.shape, q2)
        and q not in source_pixels
        and int(grid[q]) not in (BLACK, GRAY)
        and q2 in pipe_set
    ):
        return q2
    return None


def _straight_or_bend_step(pos, direction, pipe_set):
    q = _add(pos, direction)
    if q in pipe_set:
        return q, direction

    back = _add(pos, direction, -1)
    choices = []
    for nd in DIRS:
        nq = _add(pos, nd)
        if nq in pipe_set and nq != back:
            choices.append((nq, nd))
    if len(choices) == 1:
        return choices[0]
    return None, direction


def main(grid: np.ndarray) -> np.ndarray:
    """Route colored source signals through gray pipes and fill reached terminal containers.

    Input structure:
    A black background contains one-pixel-wide gray horizontal and vertical pipe networks.
    Some pipe dead ends touch solid 3x3 colored source blocks, while other dead ends touch
    3x3 gray terminal frames with a black center. Gray pipe crossings have a nearby
    non-black, non-gray single-pixel direction marker.

    Rule:
    Each source begins at its adjacent pipe end and moves away from the source. The signal
    continues along the gray pipe; at a crossing, the marker direction is tried first, and
    the signal turns that way when the indicated branch exists. If that branch is absent,
    the signal keeps moving straight. When the signal reaches a pipe end attached to a
    terminal frame, only the black center of that terminal is filled with the source color.
    Pipes, markers, source blocks, and terminal borders remain unchanged.

    Key insight:
    The visible pipe topology supplies the possible routes, while each crossing marker
    locally overrides straight travel only when it points to an available branch.
    """
    out = grid.copy()
    terminals, terminal_border = _find_terminal_frames(grid)
    sources, source_pixels = _find_source_blocks(grid)

    pipe_set = {
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if int(grid[r, c]) == GRAY and (r, c) not in terminal_border
    }

    terminal_connectors = {}
    for r, c, _ in terminals:
        for direction in DIRS:
            p = _add((r, c), direction, 2)
            if p in pipe_set:
                terminal_connectors[p] = (r, c)

    starts = []
    for center, color, _ in sources:
        for direction in DIRS:
            p = _add(center, direction, 2)
            if p in pipe_set:
                starts.append((p, direction, color))

    for start, direction, color in starts:
        pos = start
        seen = set()
        for _ in range(grid.size * 4 + 20):
            state = (pos, direction)
            if state in seen or pos not in pipe_set:
                break
            seen.add(state)

            if pos in terminal_connectors:
                tr, tc = terminal_connectors[pos]
                out[tr, tc] = color
                break

            if _pipe_degree(pos, pipe_set) >= 3:
                marked = _marker_direction(grid, pos, source_pixels)
                if marked is not None:
                    marked_next = _marked_branch_step(
                        grid, pos, marked, pipe_set, source_pixels
                    )
                    if marked_next is not None:
                        pos = marked_next
                        direction = marked
                        continue

            pos, direction = _straight_or_bend_step(pos, direction, pipe_set)
            if pos is None:
                break

    return out


def _paint_candidate(grid, cells, kinds, reserved, pos, color, kind):
    if not _inside(grid.shape, pos) or pos in reserved:
        return False
    if int(grid[pos]) != BLACK:
        return False
    if pos in cells:
        return cells[pos] == color and kinds[pos] == "pipe" and kind == "pipe"
    cells[pos] = color
    kinds[pos] = kind
    return True


def _reserve_black(grid, cells, reserved, pos):
    if not _inside(grid.shape, pos):
        return False
    if int(grid[pos]) != BLACK or pos in cells:
        return False
    reserved.add(pos)
    return True


def _add_pipe_line(grid, cells, kinds, reserved, junction, direction, start_i, end_i):
    for i in range(start_i, end_i + 1):
        if not _paint_candidate(
            grid, cells, kinds, reserved, _add(junction, direction, i), GRAY, "pipe"
        ):
            return False
    return True


def _add_source(grid, cells, kinds, reserved, endpoint, direction, color):
    center = _add(endpoint, direction, 2)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if not _paint_candidate(
                grid,
                cells,
                kinds,
                reserved,
                (center[0] + dr, center[1] + dc),
                color,
                "source",
            ):
                return False
    return True


def _add_terminal(grid, cells, kinds, reserved, endpoint, direction):
    center = _add(endpoint, direction, 2)
    if not _reserve_black(grid, cells, reserved, center):
        return False
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if not _paint_candidate(
                grid,
                cells,
                kinds,
                reserved,
                (center[0] + dr, center[1] + dc),
                GRAY,
                "frame",
            ):
                return False
    return True


def _place_motif(grid, rng, source_color):
    for _ in range(120):
        cells = {}
        kinds = {}
        reserved = set()

        junction = (
            int(rng.integers(4, grid.shape[0] - 4)),
            int(rng.integers(4, grid.shape[1] - 4)),
        )
        arrive = DIRS[int(rng.integers(0, len(DIRS)))]
        source_dir = _opposite(arrive)
        perps = _perpendiculars(arrive)
        has_marked_branch = bool(rng.integers(0, 2))

        if has_marked_branch:
            target_dir = perps[int(rng.integers(0, 2))]
            marker_dir = target_dir
            side_dir = _opposite(target_dir)
            branch_specs = [
                (source_dir, "source", int(rng.integers(3, 6)), 1),
                (arrive, "terminal", int(rng.integers(3, 6)), 1),
                (side_dir, "terminal", int(rng.integers(3, 6)), 1),
                (target_dir, "target", int(rng.integers(4, 7)), 2),
            ]
        else:
            marker_dir = perps[int(rng.integers(0, 2))]
            side_dir = _opposite(marker_dir)
            branch_specs = [
                (source_dir, "source", int(rng.integers(3, 6)), 1),
                (arrive, "target", int(rng.integers(3, 6)), 1),
                (side_dir, "terminal", int(rng.integers(3, 6)), 1),
            ]

        ok = _paint_candidate(grid, cells, kinds, reserved, junction, GRAY, "pipe")
        if not ok:
            continue

        marker = _add(junction, marker_dir)
        marker_color = PALETTE[int(rng.integers(0, len(PALETTE)))]
        ok = _paint_candidate(
            grid, cells, kinds, reserved, marker, marker_color, "marker"
        )
        if not ok:
            continue

        for direction, role, length, start_i in branch_specs:
            if not _add_pipe_line(
                grid, cells, kinds, reserved, junction, direction, start_i, length
            ):
                ok = False
                break
            endpoint = _add(junction, direction, length)
            if role == "source":
                ok = _add_source(
                    grid, cells, kinds, reserved, endpoint, direction, source_color
                )
            else:
                ok = _add_terminal(grid, cells, kinds, reserved, endpoint, direction)
            if not ok:
                break

        if not ok:
            continue

        for pos, color in cells.items():
            grid[pos] = color
        return True
    return False


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(80):
        h = int(rng.integers(20, 31))
        w = int(rng.integers(20, 31))
        grid = np.zeros((h, w), dtype=int)
        colors = list(rng.permutation(PALETTE))
        motif_count = 1 + int(min(h, w) >= 24 and rng.random() < 0.35)
        placed = 0
        for _ in range(motif_count):
            if _place_motif(grid, rng, int(colors.pop())):
                placed += 1
        if placed:
            return grid

    grid = np.zeros((24, 24), dtype=int)
    _place_motif(grid, np.random.default_rng(), 2)
    return grid
