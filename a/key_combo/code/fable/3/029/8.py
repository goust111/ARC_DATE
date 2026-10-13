from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BACKGROUND = 0
SIDES = ("top", "bottom", "left", "right")
OUTWARD = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_TO_SIDE = {v: k for k, v in OUTWARD.items()}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0)
        ]

    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[Coord] = deque([(sr, sc)])
        seen[sr, sc] = True
        pts: list[Coord] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for dr, dc in steps:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.asarray(pts, dtype=int))
    return comps


def _is_border_cell(r: int, c: int, box: BBox) -> bool:
    r0, c0, r1, c1 = box
    return r in (r0, r1) or c in (c0, c1)


def _is_corner(r: int, c: int, box: BBox) -> bool:
    r0, c0, r1, c1 = box
    return r in (r0, r1) and c in (c0, c1)


def _border_cells(box: BBox) -> list[Coord]:
    r0, c0, r1, c1 = box
    cells: list[Coord] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _marker_side(r: int, c: int, box: BBox) -> str:
    r0, c0, r1, c1 = box
    assert not _is_corner(r, c, box)
    if r == r0:
        return "top"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    assert c == c1
    return "right"


def _inner_cell(marker: Coord, side: str) -> Coord:
    r, c = marker
    dr, dc = OUTWARD[side]
    return r - dr, c - dc


def _rot_vec(v: Coord, turns: int) -> Coord:
    r, c = v
    turns %= 4
    if turns == 0:
        return int(r), int(c)
    if turns == 1:
        return int(c), int(-r)
    if turns == 2:
        return int(-r), int(-c)
    return int(-c), int(r)


def _turns_between(src: Coord, dst: Coord) -> int:
    matches = [k for k in range(4) if _rot_vec(src, k) == dst]
    assert len(matches) == 1
    return matches[0]


def _find_frame(grid: np.ndarray, bg: int) -> tuple[BBox, int, dict[int, list[tuple[Coord, str]]]]:
    candidates: list[tuple[int, int, BBox, int, dict[int, list[tuple[Coord, str]]]]] = []
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        box = _bbox_coords(coords)
        r0, c0, r1, c1 = box
        height, width = r1 - r0 + 1, c1 - c0 + 1
        if height < 7 or width < 7:
            continue
        if not all(_is_border_cell(int(r), int(c), box) for r, c in coords):
            continue
        corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
        if any(int(grid[r, c]) != color for r, c in corners):
            continue
        if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != bg):
            continue

        markers: dict[int, list[tuple[Coord, str]]] = defaultdict(list)
        complete_border = True
        for r, c in _border_cells(box):
            val = int(grid[r, c])
            if val == bg:
                complete_border = False
                break
            if val != color:
                if _is_corner(r, c, box):
                    complete_border = False
                    break
                markers[val].append(((r, c), _marker_side(r, c, box)))
        if not complete_border or not markers:
            continue
        score = height * width
        candidates.append((score, height + width, box, color, dict(markers)))

    assert candidates, "expected one rectangular frame"
    candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))
    assert len(candidates) == 1 or (candidates[0][0], candidates[0][1]) != (
        candidates[1][0],
        candidates[1][1],
    )
    _score, _span, box, color, markers = candidates[0]
    return box, int(color), markers


def _outside_frame_mask(shape: tuple[int, int], box: BBox) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = box
    outside = np.ones((h, w), dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    return outside


def _sprite_records(grid: np.ndarray, bg: int, frame_box: BBox, marker_colors: set[int]):
    outside = _outside_frame_mask(grid.shape, frame_box)
    mask = (grid != bg) & outside
    comps = _components(mask, connectivity=4)
    records: dict[int, dict] = {}

    for pts in comps:
        vals = [int(grid[r, c]) for r, c in pts]
        counts = Counter(vals)
        port_candidates = [
            color
            for color, count in counts.items()
            if color in marker_colors and int(count) == 1
        ]
        assert len(port_candidates) == 1, "each external sprite must have one marker-colored port pixel"
        port_color = int(port_candidates[0])
        body_colors = [color for color in counts if color != port_color]
        assert len(body_colors) == 1, "sprite body must be monochrome"
        body_color = int(body_colors[0])
        port_hits = [(int(r), int(c)) for r, c in pts if int(grid[r, c]) == port_color]
        assert len(port_hits) == 1
        port = port_hits[0]

        body_neighbors = [
            (nr, nc)
            for nr, nc in _neighbors4(*port)
            if 0 <= nr < grid.shape[0]
            and 0 <= nc < grid.shape[1]
            and int(grid[nr, nc]) == body_color
        ]
        assert len(body_neighbors) == 1, "port pixel must protrude from one body edge"
        nbr = body_neighbors[0]
        port_dir = (port[0] - nbr[0], port[1] - nbr[1])
        assert port_dir in DIR_TO_SIDE

        rel_cells = []
        for r, c in pts:
            rr, cc = int(r), int(c)
            rel_cells.append((rr - port[0], cc - port[1], int(grid[rr, cc])))

        assert port_color not in records, "one external sprite per marker color is required"
        records[port_color] = {
            "port_color": port_color,
            "body_color": body_color,
            "port_dir": port_dir,
            "rel_cells": rel_cells,
            "coords": [(int(r), int(c)) for r, c in pts],
        }

    return records


def _place_cells_for_marker(sprite: dict, marker: Coord, side: str) -> list[tuple[int, int, int]]:
    target_dir = OUTWARD[side]
    turns = _turns_between(sprite["port_dir"], target_dir)
    ar, ac = _inner_cell(marker, side)
    cells = []
    for dr, dc, color in sprite["rel_cells"]:
        rr, cc = _rot_vec((dr, dc), turns)
        cells.append((ar + rr, ac + cc, int(color)))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each external ported sprite and dock copies against same-colored frame markers.

    Input structure:
    - The background contains one single-pixel-thick rectangular frame whose corners and ordinary border cells share one frame color.
    - Non-corner border cells may be replaced by colored marker pixels, and each marker color appears as the one-pixel port of exactly one external sprite.
    - Every external sprite is a 4-connected component with a monochrome body and one marker-colored port pixel protruding from exactly one body edge.

    Rule:
    - Find the frame, the colored edge markers, and the external sprite paired to each marker color.
    - For every marker occurrence, rotate the paired sprite by a quarter-turn so its protruding port direction points toward that marker's frame side.
    - Erase the original external sprites, then stamp a copy inside the frame with the port pixel on the cell immediately inside the marker and the body extending inward.
    - Preserve the frame, markers, and all other background cells.

    Key insight:
    - The same-colored port and marker select the sprite, but the port's protruding direction is the only cue for the required rotation before docking.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    frame_box, _frame_color, markers = _find_frame(g, bg)
    marker_colors = set(markers)
    sprites = _sprite_records(g, bg, frame_box, marker_colors)
    assert marker_colors <= set(sprites), "each marker color must have a matching external sprite"

    r0, c0, r1, c1 = frame_box
    out = g.copy()
    for sprite in sprites.values():
        for r, c in sprite["coords"]:
            out[r, c] = bg

    for color, placements in markers.items():
        sprite = sprites[int(color)]
        for marker, side in sorted(placements):
            cells = _place_cells_for_marker(sprite, marker, side)
            for rr, cc, _color in cells:
                assert r0 < rr < r1 and c0 < cc < c1, "docked sprite must fit inside frame"
                assert int(out[rr, cc]) == bg, "docked copies must not overlap"
            for rr, cc, cell_color in cells:
                out[rr, cc] = int(cell_color)
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

GRID_SIZE_RANGE = (24, 30)
INTERIOR_H_RANGE = (10, 16)
INTERIOR_W_RANGE = (10, 16)
SPRITE_COUNT_CHOICES = (2, 3)
MAX_ATTEMPTS = 500


def _body_connected(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    return len(_components(mask, connectivity=4)) == 1


def _make_sprite(body_color: int, port_color: int, port_dir: Coord) -> dict:
    for _ in range(100):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        if random.random() < 0.7 and h == w:
            if random.random() < 0.5 and h < 4:
                h += 1
            elif w < 4:
                w += 1
        body = np.ones((h, w), dtype=bool)
        for _drop in range(random.randint(0, 2)):
            filled = list(map(tuple, np.argwhere(body)))
            random.shuffle(filled)
            for r, c in filled:
                trial = body.copy()
                trial[int(r), int(c)] = False
                if int(trial.sum()) >= 4 and _body_connected(trial):
                    body = trial
                    break

        dr, dc = port_dir
        if dr == -1:
            attach = [(0, c) for c in range(w) if body[0, c]]
        elif dr == 1:
            attach = [(h - 1, c) for c in range(w) if body[h - 1, c]]
        elif dc == -1:
            attach = [(r, 0) for r in range(h) if body[r, 0]]
        else:
            attach = [(r, w - 1) for r in range(h) if body[r, w - 1]]
        if not attach:
            continue
        ar, ac = random.choice(attach)
        port = (ar + dr, ac + dc)
        rel_cells = [(port[0] - port[0], port[1] - port[1], int(port_color))]
        for r, c in np.argwhere(body):
            rel_cells.append((int(r) - port[0], int(c) - port[1], int(body_color)))

        rows = [r for r, _c, _color in rel_cells]
        cols = [c for _r, c, _color in rel_cells]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        patch = np.full((max_r - min_r + 1, max_c - min_c + 1), BACKGROUND, dtype=int)
        port_local = (-min_r, -min_c)
        for rr, cc, color in rel_cells:
            patch[rr - min_r, cc - min_c] = int(color)
        return {
            "body_color": int(body_color),
            "port_color": int(port_color),
            "port_dir": port_dir,
            "rel_cells": rel_cells,
            "patch": patch,
            "port_local": port_local,
        }
    raise RuntimeError("failed to make sprite")


def _marker_tasks(sprites: list[dict]) -> list[tuple[int, str]]:
    n = len(sprites)
    counts = [1] * n
    extra = random.randint(2, 4)
    counts[random.randrange(n)] += 1
    extra -= 1
    while extra > 0:
        idx = random.randrange(n)
        if counts[idx] < 3:
            counts[idx] += 1
            extra -= 1

    tasks: list[tuple[int, str]] = []
    cycle = list(SIDES)
    random.shuffle(cycle)
    cursor = 0
    for idx, count in enumerate(counts):
        chosen: list[str] = []
        for _ in range(count):
            if cursor < len(cycle):
                side = cycle[cursor]
                cursor += 1
            else:
                side = random.choice(SIDES)
            chosen.append(side)
        original_side = DIR_TO_SIDE[sprites[idx]["port_dir"]]
        if all(side == original_side for side in chosen):
            chosen[0] = random.choice([side for side in SIDES if side != original_side])
        if count > 1 and len(set(chosen)) == 1:
            chosen[-1] = random.choice([side for side in SIDES if side != chosen[0]])
        tasks.extend((idx, side) for side in chosen)

    if len({side for _idx, side in tasks}) < 3:
        present = {side for _idx, side in tasks}
        missing = [side for side in SIDES if side not in present]
        for replace_i, side in zip(random.sample(range(len(tasks)), len(missing)), missing):
            idx, _old = tasks[replace_i]
            tasks[replace_i] = (idx, side)
    random.shuffle(tasks)
    return tasks


def _frame_border_positions(box: BBox, side: str) -> list[Coord]:
    r0, c0, r1, c1 = box
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    return [(r, c1) for r in range(r0 + 1, r1)]


def _footprint_cells(sprite: dict, marker: Coord, side: str) -> list[tuple[int, int, int]]:
    return _place_cells_for_marker(sprite, marker, side)


def _clear_with_padding(cells: list[tuple[int, int, int]], occupied: set[Coord], pad: int = 1) -> bool:
    coords = [(r, c) for r, c, _color in cells]
    for r, c in coords:
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if (rr, cc) in occupied:
                    return False
    return True


def _choose_markers(
    frame_box: BBox,
    sprites: list[dict],
    tasks: list[tuple[int, str]],
) -> tuple[dict[int, list[tuple[Coord, str]]], dict[Coord, int], set[Coord]] | None:
    r0, c0, r1, c1 = frame_box
    occupied: set[Coord] = set()
    used_markers: set[Coord] = set()
    markers: dict[int, list[tuple[Coord, str]]] = defaultdict(list)
    marker_at: dict[Coord, int] = {}

    for idx, side in tasks:
        sprite = sprites[idx]
        positions = _frame_border_positions(frame_box, side)
        random.shuffle(positions)
        found: tuple[Coord, list[tuple[int, int, int]]] | None = None
        for marker in positions:
            if marker in used_markers:
                continue
            cells = _footprint_cells(sprite, marker, side)
            if any(not (r0 < rr < r1 and c0 < cc < c1) for rr, cc, _color in cells):
                continue
            if not _clear_with_padding(cells, occupied, pad=1):
                continue
            found = (marker, cells)
            break
        if found is None:
            return None
        marker, cells = found
        used_markers.add(marker)
        marker_at[marker] = int(sprite["port_color"])
        markers[int(sprite["port_color"])].append((marker, side))
        for rr, cc, _color in cells:
            occupied.add((rr, cc))

    return dict(markers), marker_at, occupied


def _paint_frame(grid: np.ndarray, box: BBox, frame_color: int, marker_at: dict[Coord, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    for (r, c), color in marker_at.items():
        grid[r, c] = int(color)


def _fits_external(
    grid: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
    frame_box: BBox,
    occupied: set[Coord],
) -> bool:
    h, w = patch.shape
    rows, cols = grid.shape
    if top < 0 or left < 0 or top + h > rows or left + w > cols:
        return False
    fr0, fc0, fr1, fc1 = frame_box
    for pr, pc in np.argwhere(patch != BACKGROUND):
        r, c = top + int(pr), left + int(pc)
        if fr0 - 1 <= r <= fr1 + 1 and fc0 - 1 <= c <= fc1 + 1:
            return False
        if int(grid[r, c]) != BACKGROUND:
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied:
                    return False
    return True


def _place_external_sprites(grid: np.ndarray, frame_box: BBox, sprites: list[dict]) -> bool:
    occupied: set[Coord] = {
        (int(r), int(c))
        for r, c in np.argwhere(grid != BACKGROUND)
    }
    order = list(range(len(sprites)))
    random.shuffle(order)
    for idx in order:
        patch = sprites[idx]["patch"]
        candidates: list[Coord] = []
        for r in range(grid.shape[0] - patch.shape[0] + 1):
            for c in range(grid.shape[1] - patch.shape[1] + 1):
                if _fits_external(grid, patch, r, c, frame_box, occupied):
                    candidates.append((r, c))
        if not candidates:
            return False
        top, left = random.choice(candidates)
        coords = []
        for pr, pc in np.argwhere(patch != BACKGROUND):
            rr, cc = top + int(pr), left + int(pc)
            grid[rr, cc] = int(patch[int(pr), int(pc)])
            coords.append((rr, cc))
        sprites[idx]["external_coords"] = coords
        occupied.update(coords)
    return True


def _expected_output(
    clean: np.ndarray,
    sprites: list[dict],
    markers: dict[int, list[tuple[Coord, str]]],
) -> np.ndarray:
    out = clean.copy()
    by_port = {int(s["port_color"]): s for s in sprites}
    for port_color, items in markers.items():
        sprite = by_port[int(port_color)]
        for marker, side in items:
            for r, c, color in _footprint_cells(sprite, marker, side):
                out[r, c] = int(color)
    return out


def _semantically_valid(
    inp: np.ndarray,
    expected: np.ndarray,
    sprites: list[dict],
    markers: dict[int, list[tuple[Coord, str]]],
) -> bool:
    if inp.ndim != 2 or expected.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.array_equal(inp, expected):
        return False
    marker_counts = [len(v) for v in markers.values()]
    if max(marker_counts) < 2:
        return False
    marker_sides = [side for items in markers.values() for _marker, side in items]
    if len(set(marker_sides)) < 3:
        return False
    if not any(len({side for _marker, side in items}) > 1 for items in markers.values()):
        return False
    by_port = {int(s["port_color"]): s for s in sprites}
    if not any(
        OUTWARD[side] != by_port[int(color)]["port_dir"]
        for color, items in markers.items()
        for _marker, side in items
    ):
        return False
    try:
        solved = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(solved, expected))


def _gen_attempt() -> np.ndarray | None:
    sprite_count = random.choice(SPRITE_COUNT_CHOICES)
    colors = list(range(1, 10))
    random.shuffle(colors)
    frame_color = colors.pop()
    marker_colors = [colors.pop() for _ in range(sprite_count)]
    body_colors = [colors.pop() for _ in range(sprite_count)]

    port_dirs = random.sample(list(OUTWARD.values()), k=sprite_count)
    sprites = [
        _make_sprite(body_color, port_color, port_dir)
        for body_color, port_color, port_dir in zip(body_colors, marker_colors, port_dirs)
    ]

    interior_h = random.randint(*INTERIOR_H_RANGE)
    interior_w = random.randint(*INTERIOR_W_RANGE)
    frame_h, frame_w = interior_h + 2, interior_w + 2
    rows = random.randint(max(GRID_SIZE_RANGE[0], frame_h + 8), GRID_SIZE_RANGE[1])
    cols = random.randint(max(GRID_SIZE_RANGE[0], frame_w + 8), GRID_SIZE_RANGE[1])
    if rows > 30 or cols > 30:
        return None

    fr0 = random.randint(3, rows - frame_h - 4)
    fc0 = random.randint(3, cols - frame_w - 4)
    frame_box = (fr0, fc0, fr0 + frame_h - 1, fc0 + frame_w - 1)

    tasks = _marker_tasks(sprites)
    chosen = _choose_markers(frame_box, sprites, tasks)
    if chosen is None:
        return None
    markers, marker_at, _dock_occupied = chosen

    clean = np.full((rows, cols), BACKGROUND, dtype=int)
    _paint_frame(clean, frame_box, frame_color, marker_at)
    inp = clean.copy()
    if not _place_external_sprites(inp, frame_box, sprites):
        return None

    expected = _expected_output(clean, sprites, markers)
    if not _semantically_valid(inp, expected, sprites, markers):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotating dock input")
