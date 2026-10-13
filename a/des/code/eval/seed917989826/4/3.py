from __future__ import annotations

import random
from collections import Counter

import numpy as np


RED = 2
BLUE = 1

NW = (-1, -1)
NE = (-1, 1)
SE = (1, 1)
SW = (1, -1)
DIAGS = (NW, NE, SE, SW)

CORNERS = ("tl", "tr", "bl", "br")
CORNER_TO_DIR = {
    "tl": SE,
    "tr": SW,
    "bl": NE,
    "br": NW,
}
DIR_TO_NAME = {
    NW: "nw",
    NE: "ne",
    SE: "se",
    SW: "sw",
}

GEN_RETRIES = 400


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _rotate_cw(d: tuple[int, int]) -> tuple[int, int]:
    dr, dc = d
    return (int(dc), int(-dr))


def _rotate_ccw(d: tuple[int, int]) -> tuple[int, int]:
    dr, dc = d
    return (int(-dc), int(dr))


def _parse_legend(grid: np.ndarray, bg: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            rule = int(grid[r, c])
            if rule not in (RED, BLUE):
                continue
            partners: list[int] = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = int(r + dr)
                nc = int(c + dc)
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                val = int(grid[nr, nc])
                if val == bg or val in (RED, BLUE):
                    continue
                partners.append(val)
            if len(set(partners)) != 1:
                continue
            mapping[int(partners[0])] = 1 if rule == RED else -1
    assert mapping, "expected at least one mirror rule in the legend"
    return mapping


def _arrow_direction(points: list[tuple[int, int]]) -> tuple[int, int] | None:
    if len(points) != 3:
        return None
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 != 1 or c1 - c0 != 1:
        return None
    cells = {(int(r), int(c)) for r, c in points}
    square = {
        (int(r0), int(c0)),
        (int(r0), int(c1)),
        (int(r1), int(c0)),
        (int(r1), int(c1)),
    }
    if len(cells) != 3 or not cells.issubset(square):
        return None
    missing = list(square - cells)
    if len(missing) != 1:
        return None
    mr, mc = missing[0]
    if (mr, mc) == (r0, c0):
        return NW
    if (mr, mc) == (r0, c1):
        return NE
    if (mr, mc) == (r1, c0):
        return SW
    return SE


def _find_arrow(
    grid: np.ndarray,
    bg: int,
    mirror_colors: set[int],
) -> tuple[int, tuple[int, int], tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int], tuple[int, int, int, int], int]] = []
    colors = [int(v) for v in np.unique(grid)]
    for color in colors:
        if color in mirror_colors or color in (bg, RED, BLUE):
            continue
        for comp in _components(grid == color):
            direction = _arrow_direction(comp)
            if direction is None:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            corner_dist = min(r0 + c0, r0 + (grid.shape[1] - 1 - c1), (grid.shape[0] - 1 - r1) + c0, (grid.shape[0] - 1 - r1) + (grid.shape[1] - 1 - c1))
            candidates.append((int(color), direction, (int(r0), int(c0), int(r1), int(c1)), int(corner_dist)))
    assert len(candidates) == 1, "expected a unique arrow marker"
    color, direction, bbox, _dist = candidates[0]
    return color, direction, bbox


def _find_source(
    grid: np.ndarray,
    bg: int,
    mirror_colors: set[int],
    arrow_color: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    excluded = {int(bg), RED, BLUE, int(arrow_color), *mirror_colors}
    mask = np.array([[int(v) not in excluded for v in row] for row in grid], dtype=bool)
    candidates: list[tuple[int, int, int, int]] = []
    for comp in _components(mask):
        r0, c0, r1, c1 = _bbox(comp)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if not (3 <= h <= 5 and 3 <= w <= 5):
            continue
        if len(comp) != h * w:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if len(set(int(v) for v in patch.flatten())) < 2:
            continue
        candidates.append((int(r0), int(c0), int(r1), int(c1)))
    assert len(candidates) == 1, "expected a unique multicolor source pattern"
    r0, c0, r1, c1 = candidates[0]
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy(), (int(r0), int(c0), int(r1), int(c1))


def _emit_corner(bbox: tuple[int, int, int, int], direction: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if direction == NW:
        return int(r0), int(c0)
    if direction == NE:
        return int(r0), int(c1)
    if direction == SW:
        return int(r1), int(c0)
    return int(r1), int(c1)


def _anchor_for_endpoint(
    endpoint: tuple[int, int],
    shape: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[int, int]:
    r, c = endpoint
    h, w = shape
    if direction == NW:
        return int(r), int(c)
    if direction == NE:
        return int(r), int(c - w + 1)
    if direction == SW:
        return int(r - h + 1), int(c)
    return int(r - h + 1), int(c - w + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Colored line segments behave as mirrors for a diagonal light ray.

    The legend maps mirror color -> quarter-turn direction (red = clockwise,
    blue = counterclockwise). The arrow L-shape gives the initial diagonal
    direction. The ray paints its path in the arrow color, turns on mirrors,
    and drops a rotated copy of the source pattern at the terminal cell.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    mirror_turn = _parse_legend(g, bg)
    mirror_colors = set(int(c) for c in mirror_turn)
    arrow_color, direction, _arrow_bbox = _find_arrow(g, bg, mirror_colors)
    source, source_bbox = _find_source(g, bg, mirror_colors, arrow_color)

    out = g.copy()
    r, c = _emit_corner(source_bbox, direction)
    total_turns = 0
    seen: set[tuple[int, int, tuple[int, int]]] = set()
    h, w = g.shape

    while True:
        state = (int(r), int(c), direction)
        assert state not in seen, "ray path must not cycle"
        seen.add(state)

        nr = int(r + direction[0])
        nc = int(c + direction[1])
        if not (0 <= nr < h and 0 <= nc < w):
            endpoint = (int(r), int(c))
            break

        cell = int(g[nr, nc])
        if cell == bg:
            out[nr, nc] = int(arrow_color)
            r, c = int(nr), int(nc)
            continue

        if cell in mirror_colors:
            r, c = int(nr), int(nc)
            turn = int(mirror_turn[cell])
            total_turns += turn
            direction = _rotate_cw(direction) if turn == 1 else _rotate_ccw(direction)
            continue

        endpoint = (int(r), int(c))
        break

    rot = total_turns % 4
    rotated = np.rot90(source, -rot)
    ar, ac = _anchor_for_endpoint(endpoint, rotated.shape, direction)
    rr = int(ar + rotated.shape[0])
    cc = int(ac + rotated.shape[1])
    assert 0 <= ar and 0 <= ac and rr <= h and cc <= w, "terminal copy must fit"
    out[ar:rr, ac:cc] = rotated
    return out


def _pattern_has_arrow_like_component(patch: np.ndarray) -> bool:
    for color in set(int(v) for v in patch.flatten()):
        for comp in _components(patch == color):
            if _arrow_direction(comp) is not None:
                return True
    return False


def _make_pattern(ph: int, pw: int, colors: list[int]) -> np.ndarray:
    for _ in range(200):
        patch = np.empty((ph, pw), dtype=int)
        for r in range(ph):
            for c in range(pw):
                patch[r, c] = int(random.choice(colors))
        if len(set(int(v) for v in patch.flatten())) < 2:
            continue
        if ph == pw:
            if any(np.array_equal(patch, np.rot90(patch, k)) for k in (1, 2, 3)):
                continue
        if _pattern_has_arrow_like_component(patch):
            continue
        return patch
    raise AssertionError("failed to build a non-degenerate source pattern")


def _arrow_patch(color: int, bg: int, direction: tuple[int, int]) -> np.ndarray:
    patch = np.full((2, 2), int(bg), dtype=int)
    if direction == NW:
        missing = (0, 0)
    elif direction == NE:
        missing = (0, 1)
    elif direction == SW:
        missing = (1, 0)
    else:
        missing = (1, 1)
    for r in range(2):
        for c in range(2):
            if (r, c) != missing:
                patch[r, c] = int(color)
    return patch


def _legend_cells(
    corner: str,
    orientation: str,
    mirror_colors: list[int],
    turns: list[int],
    shape: tuple[int, int],
) -> dict[tuple[int, int], int]:
    h, w = shape
    cells: dict[tuple[int, int], int] = {}
    if orientation == "horizontal":
        height = 2 * len(mirror_colors) - 1
        width = 2
        if corner in ("tl", "tr"):
            r0 = 0
        else:
            r0 = h - height
        if corner in ("tl", "bl"):
            c0 = 0
        else:
            c0 = w - width
        for i, (mirror, turn) in enumerate(zip(mirror_colors, turns, strict=True)):
            rr = int(r0 + 2 * i)
            cells[(rr, int(c0))] = int(mirror)
            cells[(rr, int(c0 + 1))] = RED if int(turn) == 1 else BLUE
    else:
        height = 2
        width = 2 * len(mirror_colors) - 1
        if corner in ("tl", "tr"):
            r0 = 0
        else:
            r0 = h - height
        if corner in ("tl", "bl"):
            c0 = 0
        else:
            c0 = w - width
        for i, (mirror, turn) in enumerate(zip(mirror_colors, turns, strict=True)):
            cc = int(c0 + 2 * i)
            cells[(int(r0), cc)] = int(mirror)
            cells[(int(r0 + 1), cc)] = RED if int(turn) == 1 else BLUE
    return cells


def _corner_positions(
    corner: str,
    ph: int,
    pw: int,
    shape: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = shape
    margin_r = random.randint(3, 5)
    margin_c = random.randint(3, 5)
    if corner == "tl":
        src = (int(margin_r), int(margin_c))
        arrow = (int(src[0] - 3), int(src[1] - 3))
    elif corner == "tr":
        src = (int(margin_r), int(w - pw - margin_c))
        arrow = (int(src[0] - 3), int(src[1] + pw + 1))
    elif corner == "bl":
        src = (int(h - ph - margin_r), int(margin_c))
        arrow = (int(src[0] + ph + 1), int(src[1] - 3))
    else:
        src = (int(h - ph - margin_r), int(w - pw - margin_c))
        arrow = (int(src[0] + ph + 1), int(src[1] + pw + 1))
    return src, arrow


def _cells_of_patch(top_left: tuple[int, int], patch: np.ndarray) -> set[tuple[int, int]]:
    r0, c0 = top_left
    cells: set[tuple[int, int]] = set()
    for r in range(patch.shape[0]):
        for c in range(patch.shape[1]):
            if int(patch[r, c]) >= 0:
                cells.add((int(r0 + r), int(c0 + c)))
    return cells


def _fits(top_left: tuple[int, int], patch_shape: tuple[int, int], shape: tuple[int, int]) -> bool:
    r0, c0 = top_left
    h, w = patch_shape
    return 0 <= r0 and 0 <= c0 and r0 + h <= shape[0] and c0 + w <= shape[1]


def _ray_cells_until_stop(
    pos: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], str]:
    h, w = shape
    r, c = pos
    cells: list[tuple[int, int]] = []
    while True:
        r += direction[0]
        c += direction[1]
        if not (0 <= r < h and 0 <= c < w):
            return cells, "boundary"
        if (int(r), int(c)) in blocked:
            return cells, "blocked"
        if (int(r), int(c)) in forbidden:
            return cells, "forbidden"
        cells.append((int(r), int(c)))


def _plan_path(
    start: tuple[int, int],
    direction: tuple[int, int],
    turns_needed: int,
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> tuple[list[dict[str, object]], set[tuple[int, int]], tuple[int, int], tuple[int, int]] | None:
    min_leg = 2

    def rec(
        pos: tuple[int, int],
        cur_dir: tuple[int, int],
        left: int,
        used: set[tuple[int, int]],
    ) -> tuple[list[dict[str, object]], set[tuple[int, int]], tuple[int, int], tuple[int, int]] | None:
        cells, reason = _ray_cells_until_stop(pos, cur_dir, shape, blocked, used)
        if left == 0:
            if reason != "boundary" or len(cells) < min_leg:
                return None
            new_used = set(used)
            new_used.update(cells)
            return [], new_used, cur_dir, cells[-1]

        if len(cells) < min_leg + 1:
            return None

        indices = list(range(min_leg - 1, len(cells)))
        random.shuffle(indices)
        for idx in indices:
            hit = cells[idx]
            if hit[0] in (0, shape[0] - 1) or hit[1] in (0, shape[1] - 1):
                continue
            prefix = cells[: idx + 1]
            next_used = set(used)
            next_used.update(prefix)
            turn_options = [1, -1]
            random.shuffle(turn_options)
            for turn in turn_options:
                next_dir = _rotate_cw(cur_dir) if turn == 1 else _rotate_ccw(cur_dir)
                result = rec(hit, next_dir, left - 1, next_used)
                if result is None:
                    continue
                events, final_used, final_dir, endpoint = result
                event = {"hit": hit, "turn": int(turn)}
                return [event] + events, final_used, final_dir, endpoint
        return None

    return rec(start, direction, turns_needed, set())


def _place_segments(
    hits: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    path_cells: set[tuple[int, int]],
    shape: tuple[int, int],
) -> list[list[tuple[int, int]]] | None:
    h, w = shape
    placed: list[list[tuple[int, int]]] = []
    occ = set(occupied)

    for hit in hits:
        hr, hc = hit
        choices: list[list[tuple[int, int]]] = []
        for orient in ("h", "v"):
            for length in (2, 3, 4):
                if orient == "h":
                    for c0 in range(hc - length + 1, hc + 1):
                        c1 = c0 + length - 1
                        if c0 < 0 or c1 >= w:
                            continue
                        cells = [(int(hr), int(cc)) for cc in range(c0, c1 + 1)]
                        if hit not in cells:
                            continue
                        bad = False
                        for cell in cells:
                            if cell != hit and (cell in occ or cell in path_cells):
                                bad = True
                                break
                        if not bad:
                            choices.append(cells)
                else:
                    for r0 in range(hr - length + 1, hr + 1):
                        r1 = r0 + length - 1
                        if r0 < 0 or r1 >= h:
                            continue
                        cells = [(int(rr), int(hc)) for rr in range(r0, r1 + 1)]
                        if hit not in cells:
                            continue
                        bad = False
                        for cell in cells:
                            if cell != hit and (cell in occ or cell in path_cells):
                                bad = True
                                break
                        if not bad:
                            choices.append(cells)
        if not choices:
            return None
        cells = random.choice(choices)
        placed.append(cells)
        occ.update(cells)
    return placed


def _build_attempt() -> np.ndarray | None:
    h = random.randint(19, 28)
    w = random.randint(19, 28)
    grid_shape = (int(h), int(w))

    corner = random.choice(CORNERS)
    direction = CORNER_TO_DIR[corner]
    ph = random.randint(3, 5)
    pw = random.randint(3, 5)
    turns_needed = random.randint(2, 4)

    bg = random.choice([0, 3, 4, 5, 6, 7, 8, 9])
    free_colors = [c for c in range(10) if c not in (bg, RED, BLUE)]
    random.shuffle(free_colors)
    mirror_colors = free_colors[:turns_needed]
    arrow_color = free_colors[turns_needed]
    pattern_colors = free_colors[turns_needed + 1 :]
    if len(pattern_colors) < 2:
        return None
    pattern_color_count = 2 if len(pattern_colors) == 2 else random.randint(2, min(4, len(pattern_colors)))
    pattern = _make_pattern(ph, pw, pattern_colors[:pattern_color_count])

    src_tl, arrow_tl = _corner_positions(corner, ph, pw, grid_shape)
    if not _fits(src_tl, pattern.shape, grid_shape):
        return None
    arrow_patch = _arrow_patch(arrow_color, bg, direction)
    if not _fits(arrow_tl, arrow_patch.shape, grid_shape):
        return None

    legend_corner_choices = [c for c in CORNERS if c != corner]
    legend_corner = random.choice(legend_corner_choices)
    legend_orientation = random.choice(["horizontal", "vertical"])
    legend = _legend_cells(legend_corner, legend_orientation, mirror_colors, [1] * turns_needed, grid_shape)

    occupied: set[tuple[int, int]] = set()
    source_cells = _cells_of_patch(src_tl, pattern)
    arrow_cells = {(r, c) for (r, c), val in [((rr + arrow_tl[0], cc + arrow_tl[1]), int(arrow_patch[rr, cc])) for rr in range(2) for cc in range(2)] if val != bg}
    legend_cells = set(legend)
    occupied.update(source_cells)
    occupied.update(arrow_cells)
    occupied.update(legend_cells)
    if len(occupied) != len(source_cells) + len(arrow_cells) + len(legend_cells):
        return None

    source_bbox = (src_tl[0], src_tl[1], src_tl[0] + ph - 1, src_tl[1] + pw - 1)
    start = _emit_corner(source_bbox, direction)
    planned = _plan_path(start, direction, turns_needed, grid_shape, occupied)
    if planned is None:
        return None
    events, path_cells, final_dir, endpoint = planned
    hit_cells = [tuple(event["hit"]) for event in events]
    turn_values = [int(event["turn"]) for event in events]
    legend = _legend_cells(legend_corner, legend_orientation, mirror_colors, turn_values, grid_shape)

    total_turns = sum(int(event["turn"]) for event in events)
    rotated = np.rot90(pattern, -(total_turns % 4))
    anchor = _anchor_for_endpoint(endpoint, rotated.shape, final_dir)
    if not _fits(anchor, rotated.shape, grid_shape):
        return None

    segment_cells = _place_segments(hit_cells, occupied, path_cells, grid_shape)
    if segment_cells is None:
        return None

    mirror_object_cells = set()
    for cells in segment_cells:
        mirror_object_cells.update(cells)
    final_box_cells = {(anchor[0] + r, anchor[1] + c) for r in range(rotated.shape[0]) for c in range(rotated.shape[1])}
    if final_box_cells & (occupied | mirror_object_cells):
        return None

    grid = np.full(grid_shape, int(bg), dtype=int)
    grid[src_tl[0] : src_tl[0] + ph, src_tl[1] : src_tl[1] + pw] = pattern
    for rr in range(2):
        for cc in range(2):
            if int(arrow_patch[rr, cc]) != bg:
                grid[arrow_tl[0] + rr, arrow_tl[1] + cc] = int(arrow_color)
    for (r, c), val in legend.items():
        grid[r, c] = int(val)
    for cells, color in zip(segment_cells, mirror_colors, strict=True):
        for r, c in cells:
            grid[r, c] = int(color)

    out = main(grid)
    assert isinstance(out, np.ndarray)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid puzzle instance")
