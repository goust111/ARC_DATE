from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, find_components, get_bg


DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0))


def _square_bbox(square: dict) -> tuple[int, int, int, int]:
    return square["bbox"]


def _portal_squares(grid: np.ndarray, bg: int) -> list[dict]:
    portals: list[dict] = []
    for comp in find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    ):
        r0, c0, r1, c1 = comp["bbox"]
        color = int(comp["color"])
        if (
            int(comp["area"]) == 9
            and bbox_h(comp["bbox"]) == 3
            and bbox_w(comp["bbox"]) == 3
            and np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)
        ):
            portals.append(
                {
                    "color": color,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    assert portals, "expected at least one portal pair"
    by_color: dict[int, list[dict]] = {}
    for square in portals:
        by_color.setdefault(int(square["color"]), []).append(square)
    assert all(len(items) == 2 for items in by_color.values()), "every portal color must appear exactly twice"
    return portals


def _marker_color(grid: np.ndarray, bg: int, squares: list[dict]) -> int:
    portal_colors = {int(square["color"]) for square in squares}
    colors = set(map(int, np.unique(grid))) - {int(bg)}
    marker_colors = colors - portal_colors
    assert len(marker_colors) == 1, "expected exactly one marker color"
    return marker_colors.pop()


def _adjacent_square_direction(r: int, c: int, square: dict) -> tuple[int, int] | None:
    r0, c0, r1, c1 = _square_bbox(square)
    if r0 <= r <= r1:
        if c == c0 - 1:
            return (0, 1)
        if c == c1 + 1:
            return (0, -1)
    if c0 <= c <= c1:
        if r == r0 - 1:
            return (1, 0)
        if r == r1 + 1:
            return (-1, 0)
    return None


def _adjacent_hits(r: int, c: int, squares: list[dict]) -> list[tuple[dict, tuple[int, int]]]:
    hits: list[tuple[dict, tuple[int, int]]] = []
    for square in squares:
        direction = _adjacent_square_direction(int(r), int(c), square)
        if direction is not None:
            hits.append((square, direction))
    return hits


def _paired_square(square: dict, by_color: dict[int, list[dict]]) -> dict:
    mates = by_color[int(square["color"])]
    assert len(mates) == 2
    return mates[1] if mates[0] is square else mates[0]


def _teleport_exit(
    cell: tuple[int, int],
    entry_square: dict,
    exit_square: dict,
    direction: tuple[int, int],
) -> tuple[int, int]:
    r, c = map(int, cell)
    er0, ec0, er1, ec1 = _square_bbox(entry_square)
    xr0, xc0, xr1, xc1 = _square_bbox(exit_square)

    if direction == (0, 1):
        offset = r - er0
        return xr0 + offset, xc1 + 1
    if direction == (0, -1):
        offset = r - er0
        return xr0 + offset, xc0 - 1
    if direction == (1, 0):
        offset = c - ec0
        return xr1 + 1, xc0 + offset
    assert direction == (-1, 0)
    offset = c - ec0
    return xr0 - 1, xc0 + offset


def _trace_marker(
    marker: tuple[int, int],
    shape: tuple[int, int],
    squares: list[dict],
    by_color: dict[int, list[dict]],
) -> dict:
    hits = _adjacent_hits(marker[0], marker[1], squares)
    assert len(hits) == 1, "every marker must touch exactly one portal side"
    entry_square, direction = hits[0]
    current = _teleport_exit(marker, entry_square, _paired_square(entry_square, by_color), direction)

    H, W = map(int, shape)
    path: list[tuple[int, int]] = []
    hit_colors = [int(entry_square["color"])]
    seen_states: set[tuple[int, int, int, int]] = set()
    dr, dc = direction

    while 0 <= current[0] < H and 0 <= current[1] < W:
        state = (int(current[0]), int(current[1]), int(dr), int(dc))
        assert state not in seen_states, "portal layout must not create an infinite teleport cycle"
        seen_states.add(state)
        path.append(current)

        front_hits = [
            square
            for square in squares
            if _adjacent_square_direction(int(current[0]), int(current[1]), square) == direction
        ]
        if front_hits:
            assert len(front_hits) == 1, "ray must face at most one portal"
            front_square = front_hits[0]
            hit_colors.append(int(front_square["color"]))
            current = _teleport_exit(
                current,
                front_square,
                _paired_square(front_square, by_color),
                direction,
            )
        else:
            current = (int(current[0]) + dr, int(current[1]) + dc)

    return {
        "marker": (int(marker[0]), int(marker[1])),
        "direction": direction,
        "path": path,
        "path_set": set(path),
        "teleports": len(hit_colors),
        "hit_colors": tuple(hit_colors),
    }


def _analyze_grid(grid: np.ndarray) -> tuple[int, int, list[dict], dict[int, list[dict]], list[tuple[int, int]], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    squares = _portal_squares(g, bg)
    by_color: dict[int, list[dict]] = {}
    for square in squares:
        by_color.setdefault(int(square["color"]), []).append(square)

    marker_color = _marker_color(g, bg, squares)
    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == marker_color)]
    assert markers, "expected at least one marker"
    traces = [_trace_marker(marker, g.shape, squares, by_color) for marker in markers]
    return bg, marker_color, squares, by_color, markers, traces


def main(grid: np.ndarray) -> np.ndarray:
    """Teleport each side-attached marker through same-color 3x3 portal pairs and paint the chained ray trace.

    Input structure
    - The grid has a dominant background, several solid 3x3 portal squares, and exactly one distinct marker color used only by isolated side-adjacent pixels.
    - Every portal color appears on exactly two 3x3 squares, forming a same-color pair.
    - Each marker pixel orthogonally touches exactly one portal side and therefore points into that portal.

    Rule
    - Remove every original marker pixel, keep all 3x3 portal squares unchanged, and process each marker independently.
    - A marker enters the touched square in the inward normal direction, reappears outside the paired square on the opposite side, and preserves its offset along the contacted side.
    - From that exit cell, extend a straight ray in the same direction, painting the marker color on every visited cell until the ray leaves the grid; whenever a visited cell directly faces another portal side head-on, teleport again by the same same-color pairing rule and continue the chain.

    Key insight
    - Portal color selects the partner square, while the incoming side and side-offset are conserved across each jump, so the output is the full chained trajectory rather than a local stamp near one square.
    """
    g = np.asarray(grid, dtype=int)
    bg, marker_color, _squares, _by_color, markers, traces = _analyze_grid(g)

    out = g.copy()
    for r, c in markers:
        out[r, c] = bg
    for trace in traces:
        for r, c in trace["path"]:
            out[r, c] = marker_color
    return out


def _gap_ok(r0: int, c0: int, placed: list[dict]) -> bool:
    r1 = r0 + 2
    c1 = c0 + 2
    for square in placed:
        sr0, sc0, sr1, sc1 = _square_bbox(square)
        if not (r1 < sr0 - 1 or sr1 < r0 - 1 or c1 < sc0 - 1 or sc1 < c0 - 1):
            return False
    return True


def _place_square(
    color: int,
    r0: int,
    c0: int,
    H: int,
    W: int,
    placed: list[dict],
) -> dict | None:
    if not (1 <= r0 <= H - 4 and 1 <= c0 <= W - 4):
        return None
    if not _gap_ok(r0, c0, placed):
        return None
    return {"color": int(color), "bbox": (int(r0), int(c0), int(r0 + 2), int(c0 + 2))}


def _available_positions(H: int, W: int, placed: list[dict]) -> list[tuple[int, int]]:
    positions = []
    for r0 in range(1, H - 3):
        for c0 in range(1, W - 3):
            if _gap_ok(r0, c0, placed):
                positions.append((r0, c0))
    random.shuffle(positions)
    return positions


def _exit_cell_for_offset(square: dict, offset: int, direction: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = _square_bbox(square)
    if direction == (0, 1):
        return r0 + offset, c1 + 1
    if direction == (0, -1):
        return r0 + offset, c0 - 1
    if direction == (1, 0):
        return r1 + 1, c0 + offset
    return r0 - 1, c0 + offset


def _initial_marker_cell(square: dict, offset: int, direction: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = _square_bbox(square)
    if direction == (0, 1):
        return r0 + offset, c0 - 1
    if direction == (0, -1):
        return r0 + offset, c1 + 1
    if direction == (1, 0):
        return r0 - 1, c0 + offset
    return r1 + 1, c0 + offset


def _next_hit_candidates(
    current: tuple[int, int],
    direction: tuple[int, int],
    H: int,
    W: int,
    placed: list[dict],
) -> list[tuple[int, int, int]]:
    r, c = map(int, current)
    candidates: list[tuple[int, int, int]] = []
    if direction == (0, 1):
        for offset in range(3):
            r0 = r - offset
            for c0 in range(c + 1, W - 3):
                if _place_square(1, r0, c0, H, W, placed) is not None:
                    candidates.append((r0, c0, offset))
    elif direction == (0, -1):
        for offset in range(3):
            r0 = r - offset
            for c0 in range(1, c - 2):
                if _place_square(1, r0, c0, H, W, placed) is not None:
                    candidates.append((r0, c0, offset))
    elif direction == (1, 0):
        for offset in range(3):
            c0 = c - offset
            for r0 in range(r + 1, H - 3):
                if _place_square(1, r0, c0, H, W, placed) is not None:
                    candidates.append((r0, c0, offset))
    else:
        for offset in range(3):
            c0 = c - offset
            for r0 in range(1, r - 2):
                if _place_square(1, r0, c0, H, W, placed) is not None:
                    candidates.append((r0, c0, offset))
    random.shuffle(candidates)
    return candidates


def _build_required_route(
    route_colors: list[int],
    direction: tuple[int, int],
    H: int,
    W: int,
    placed: list[dict],
) -> tuple[list[dict], tuple[int, int]] | None:
    positions = _available_positions(H, W, placed)
    if not positions:
        return None

    for _ in range(200):
        local = list(placed)
        built: list[dict] = []
        marker_cell: tuple[int, int] | None = None

        color0 = route_colors[0]
        a1_r0, a1_c0 = random.choice(positions)
        a1 = _place_square(color0, a1_r0, a1_c0, H, W, local)
        if a1 is None:
            continue
        offset = random.randrange(3)
        marker_cell = _initial_marker_cell(a1, offset, direction)
        if not (0 <= marker_cell[0] < H and 0 <= marker_cell[1] < W):
            continue
        local.append(a1)
        built.append(a1)

        b_choices = _available_positions(H, W, local)
        random.shuffle(b_choices)
        ok = False
        current = None
        for b_r0, b_c0 in b_choices:
            b1 = _place_square(color0, b_r0, b_c0, H, W, local)
            if b1 is None:
                continue
            current = _exit_cell_for_offset(b1, offset, direction)
            if not (0 <= current[0] < H and 0 <= current[1] < W):
                continue
            if len(route_colors) > 1 and not _next_hit_candidates(current, direction, H, W, local + [b1]):
                continue
            local.append(b1)
            built.append(b1)
            ok = True
            break
        if not ok or current is None:
            continue

        success = True
        for color in route_colors[1:]:
            candidates = _next_hit_candidates(current, direction, H, W, local)
            if not candidates:
                success = False
                break
            hit_r0, hit_c0, offset = random.choice(candidates)
            hit_square = _place_square(color, hit_r0, hit_c0, H, W, local)
            if hit_square is None:
                success = False
                break
            local.append(hit_square)
            built.append(hit_square)

            mate_choices = _available_positions(H, W, local)
            random.shuffle(mate_choices)
            mate_ok = False
            for mate_r0, mate_c0 in mate_choices:
                mate = _place_square(color, mate_r0, mate_c0, H, W, local)
                if mate is None:
                    continue
                next_current = _exit_cell_for_offset(mate, offset, direction)
                if not (0 <= next_current[0] < H and 0 <= next_current[1] < W):
                    continue
                remaining = route_colors[route_colors.index(color) + 1 :]
                if remaining and not _next_hit_candidates(next_current, direction, H, W, local + [mate]):
                    continue
                local.append(mate)
                built.append(mate)
                current = next_current
                mate_ok = True
                break
            if not mate_ok:
                success = False
                break

        if success and marker_cell is not None:
            return built, marker_cell
    return None


def _render_base_grid(H: int, W: int, bg: int, squares: list[dict]) -> np.ndarray:
    grid = np.full((H, W), int(bg), dtype=int)
    for square in squares:
        r0, c0, r1, c1 = _square_bbox(square)
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(square["color"])
    return grid


def _candidate_markers(base: np.ndarray, squares: list[dict]) -> list[tuple[int, int]]:
    H, W = base.shape
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for square in squares:
        r0, c0, r1, c1 = _square_bbox(square)
        cells = (
            [(r, c0 - 1) for r in range(r0, r1 + 1)]
            + [(r, c1 + 1) for r in range(r0, r1 + 1)]
            + [(r0 - 1, c) for c in range(c0, c1 + 1)]
            + [(r1 + 1, c) for c in range(c0, c1 + 1)]
        )
        for r, c in cells:
            if not (0 <= r < H and 0 <= c < W):
                continue
            if int(base[r, c]) != int(get_bg(base)):
                continue
            hits = _adjacent_hits(r, c, squares)
            if len(hits) == 1 and (r, c) not in seen:
                seen.add((r, c))
                out.append((int(r), int(c)))
    random.shuffle(out)
    return out


def _choose_markers(
    base: np.ndarray,
    marker_color: int,
    required_marker: tuple[int, int],
    target_marker_count: int,
    required_mode: str,
) -> list[tuple[int, int]] | None:
    try:
        bg, _mc, squares, by_color, _markers, _traces = _analyze_grid(
            np.where(base == marker_color, int(get_bg(base)), base)
        )
    except Exception:
        bg = get_bg(base)
        squares = _portal_squares(base, bg)
        by_color = {}
        for square in squares:
            by_color.setdefault(int(square["color"]), []).append(square)

    candidates = _candidate_markers(base, squares)
    if required_marker not in candidates:
        return None

    traces: dict[tuple[int, int], dict] = {}
    for marker in candidates:
        try:
            traces[marker] = _trace_marker(marker, base.shape, squares, by_color)
        except Exception:
            continue
    if required_marker not in traces:
        return None

    required_trace = traces[required_marker]
    if required_mode == "simple":
        if required_trace["teleports"] != 1:
            return None
    else:
        if required_trace["teleports"] < 2:
            return None

    chosen = [required_marker]
    used_paths = [required_trace["path_set"]]

    extras = [marker for marker in candidates if marker != required_marker and marker in traces]
    extras.sort(
        key=lambda marker: (
            traces[marker]["teleports"],
            len(traces[marker]["path"]),
            marker,
        ),
        reverse=True,
    )

    for marker in extras:
        if len(chosen) >= target_marker_count:
            break
        path_set = traces[marker]["path_set"]
        if any(path_set == other for other in used_paths):
            continue
        if sum(len(path_set & other) for other in used_paths) > max(2, len(path_set) // 2):
            continue
        chosen.append(marker)
        used_paths.append(path_set)

    if len(chosen) != target_marker_count:
        return None

    total_path = sum(len(traces[marker]["path"]) for marker in chosen)
    union_path = set().union(*(traces[marker]["path_set"] for marker in chosen))
    if len(union_path) < max(4, total_path // 2):
        return None

    return chosen


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(15, 24)
    W = random.randint(15, 24)
    n_pairs = random.randint(2, 4)
    target_marker_count = random.choices([1, 2, 3], weights=[2, 5, 3], k=1)[0]
    required_mode = random.choices(["simple", "chain"], weights=[2, 3], k=1)[0]
    route_len = 1 if required_mode == "simple" else random.randint(2, min(3, n_pairs))
    direction = random.choice(DIRS)

    palette = random.sample(list(range(10)), n_pairs + 2)
    bg = palette[0]
    marker_color = palette[1]
    pair_colors = palette[2:]
    route_colors = pair_colors[:route_len]
    filler_colors = pair_colors[route_len:]

    placed: list[dict] = []
    built = _build_required_route(route_colors, direction, H, W, placed)
    if built is None:
        return None
    route_squares, required_marker = built
    placed.extend(route_squares)

    for color in filler_colors:
        choices = _available_positions(H, W, placed)
        if len(choices) < 2:
            return None
        first = _place_square(color, choices[0][0], choices[0][1], H, W, placed)
        if first is None:
            return None
        placed.append(first)
        choices = _available_positions(H, W, placed)
        if not choices:
            return None
        second = _place_square(color, choices[0][0], choices[0][1], H, W, placed)
        if second is None:
            return None
        placed.append(second)

    base = _render_base_grid(H, W, bg, placed)
    markers = _choose_markers(base, marker_color, required_marker, target_marker_count, required_mode)
    if markers is None:
        return None

    inp = base.copy()
    for r, c in markers:
        inp[r, c] = marker_color

    try:
        out = main(inp)
    except Exception:
        return None

    if out.shape != inp.shape:
        return None
    if np.array_equal(inp, out):
        return None

    try:
        bg2, marker2, _squares, _by_color, _marker_cells, traces = _analyze_grid(inp)
    except Exception:
        return None
    assert bg2 == bg and marker2 == marker_color

    if len(traces) != len(markers):
        return None
    if required_mode == "simple":
        if not any(trace["marker"] == required_marker and trace["teleports"] == 1 for trace in traces):
            return None
    else:
        if not any(trace["marker"] == required_marker and trace["teleports"] >= 2 for trace in traces):
            return None

    total_path = sum(len(trace["path"]) for trace in traces)
    unique_path = set().union(*(trace["path_set"] for trace in traces))
    if total_path < 4 or len(unique_path) < max(4, total_path // 2):
        return None
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid same-color teleport portals input")
