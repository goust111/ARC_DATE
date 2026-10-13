from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
FRAME_COLOR = 1
SOURCE_COLORS = [2, 3, 4, 5, 6, 7, 8, 9]
SIDES = ("top", "bottom", "left", "right")
OUTWARD = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
INWARD = {side: (-dr, -dc) for side, (dr, dc) in OUTWARD.items()}


@dataclass(frozen=True)
class FrameSpec:
    bbox: tuple[int, int, int, int]
    gaps: tuple[tuple[int, int], tuple[int, int]]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    return int(prefix[r1 + 1, c1 + 1] - prefix[r0, c1 + 1] - prefix[r1 + 1, c0] + prefix[r0, c0])


def _prefix(mask: np.ndarray) -> np.ndarray:
    return np.pad(mask.astype(int), ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)


def _border_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _interior_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _is_corner(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r in (r0, r1) and c in (c0, c1)


def _gap_side(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = gap
    r0, c0, r1, c1 = bbox
    assert int(r in (r0, r1)) + int(c in (c0, c1)) == 1
    if r == r0:
        return "top"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    return "right"


def _entry_side_for_dir(direction: tuple[int, int]) -> str:
    if direction == (1, 0):
        return "top"
    if direction == (-1, 0):
        return "bottom"
    if direction == (0, 1):
        return "left"
    assert direction == (0, -1)
    return "right"


def _source_direction(source: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r, c = source
    h, w = shape
    sides = []
    if r == 0:
        sides.append((1, 0))
    if r == h - 1:
        sides.append((-1, 0))
    if c == 0:
        sides.append((0, 1))
    if c == w - 1:
        sides.append((0, -1))
    assert len(sides) == 1, "source must be on exactly one non-corner boundary side"
    return sides[0]


def _frame_pixels(spec: FrameSpec) -> set[tuple[int, int]]:
    return {cell for cell in _border_cells(spec.bbox) if cell not in spec.gaps}


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _expanded_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0


def _find_source(g: np.ndarray) -> tuple[int, int]:
    h, w = g.shape
    boundary: list[tuple[int, int]] = []
    for c in range(w):
        boundary.append((0, c))
        boundary.append((h - 1, c))
    for r in range(1, h - 1):
        boundary.append((r, 0))
        boundary.append((r, w - 1))
    sources = [(r, c) for r, c in boundary if int(g[r, c]) != BG]
    assert len(sources) == 1, "expected exactly one boundary source pixel"
    _source_direction(sources[0], g.shape)
    return sources[0]


def _extract_frames(g: np.ndarray, frame_color: int) -> list[FrameSpec]:
    h, w = g.shape
    frame_mask = g == int(frame_color)
    fg_mask = g != BG
    frame_prefix = _prefix(frame_mask)
    fg_prefix = _prefix(fg_mask)

    rows = [int(r) for r in np.where(frame_mask.any(axis=1))[0]]
    cols = [int(c) for c in np.where(frame_mask.any(axis=0))[0]]
    candidates: list[FrameSpec] = []

    for i, r0 in enumerate(rows):
        if r0 == 0:
            continue
        for r1 in rows[i + 1 :]:
            if r1 == h - 1 or r1 - r0 + 1 < 3:
                continue
            for j, c0 in enumerate(cols):
                if c0 == 0:
                    continue
                for c1 in cols[j + 1 :]:
                    if c1 == w - 1 or c1 - c0 + 1 < 3:
                        continue
                    if not (frame_mask[r0, c0] and frame_mask[r0, c1] and frame_mask[r1, c0] and frame_mask[r1, c1]):
                        continue

                    interior_fg = _rect_sum(fg_prefix, r0 + 1, c0 + 1, r1 - 1, c1 - 1)
                    if interior_fg != 0:
                        continue

                    total_frame = _rect_sum(frame_prefix, r0, c0, r1, c1)
                    border_frame = total_frame
                    total_fg = _rect_sum(fg_prefix, r0, c0, r1, c1)
                    border_fg = total_fg
                    perim = 2 * (r1 - r0 + 1) + 2 * (c1 - c0 + 1) - 4
                    if border_frame != perim - 2 or border_fg != border_frame:
                        continue

                    bbox = (r0, c0, r1, c1)
                    gaps = [cell for cell in _border_cells(bbox) if not frame_mask[cell]]
                    if len(gaps) != 2:
                        continue
                    if any(_is_corner(gap, bbox) for gap in gaps):
                        continue
                    candidates.append(FrameSpec(bbox, (gaps[0], gaps[1])))

    all_pixels = {tuple(map(int, cell)) for cell in np.argwhere(frame_mask)}
    candidates.sort(key=lambda spec: (_bbox_area(spec.bbox), len(_frame_pixels(spec))), reverse=True)
    selected: list[FrameSpec] = []
    covered: set[tuple[int, int]] = set()
    for spec in candidates:
        pixels = _frame_pixels(spec)
        if pixels & covered:
            continue
        selected.append(spec)
        covered.update(pixels)
        if covered == all_pixels:
            break

    assert selected and covered == all_pixels, "frame pixels must decompose into two-gap rectangular frames"
    for i, spec in enumerate(selected):
        for other in selected[i + 1 :]:
            assert _expanded_separated(spec.bbox, other.bbox), "frames must not touch"
    return sorted(selected, key=lambda spec: spec.bbox)


def _trace(
    g: np.ndarray,
    source: tuple[int, int],
    frames: list[FrameSpec],
) -> tuple[np.ndarray, set[int], str]:
    source_color = int(g[source])
    out = g.copy()

    gap_info: dict[tuple[int, int], tuple[int, str]] = {}
    wall_owner: dict[tuple[int, int], int] = {}
    interior_owner: dict[tuple[int, int], int] = {}
    for idx, spec in enumerate(frames):
        for gap in spec.gaps:
            side = _gap_side(gap, spec.bbox)
            assert gap not in gap_info
            gap_info[gap] = (idx, side)
        for cell in _frame_pixels(spec):
            assert cell not in wall_owner
            wall_owner[cell] = idx
        for cell in _interior_cells(spec.bbox):
            assert cell not in interior_owner
            interior_owner[cell] = idx

    dr, dc = _source_direction(source, g.shape)
    r, c = source[0] + dr, source[1] + dc
    visited_frames: set[int] = set()
    seen_states: set[tuple[int, int, int, int]] = set()
    terminal = "boundary"

    while _in_bounds(g.shape, r, c):
        state = (r, c, dr, dc)
        assert state not in seen_states, "beam chain must not loop"
        seen_states.add(state)

        cell = (r, c)
        if cell in wall_owner:
            terminal = "wall"
            break

        if cell in gap_info:
            frame_idx, side = gap_info[cell]
            if (dr, dc) == INWARD[side]:
                assert frame_idx not in visited_frames, "a frame may be entered at most once"
                spec = frames[frame_idx]
                entry_gap = cell
                exit_gap = spec.gaps[1] if spec.gaps[0] == entry_gap else spec.gaps[0]
                r0, c0, r1, c1 = spec.bbox

                out[entry_gap] = source_color
                out[r0 + 1 : r1, c0 + 1 : c1] = source_color
                out[exit_gap] = source_color
                visited_frames.add(frame_idx)

                exit_side = _gap_side(exit_gap, spec.bbox)
                dr, dc = OUTWARD[exit_side]
                r, c = exit_gap[0] + dr, exit_gap[1] + dc
                continue

        assert cell not in interior_owner, "beam may enter a frame only through a facing gap"
        assert int(g[cell]) == BG, "beam path must cross only black background before a wall or gap"
        out[cell] = source_color
        r += dr
        c += dc

    return out, visited_frames, terminal


def main(grid: np.ndarray) -> np.ndarray:
    """
    Ferry a boundary beam through paired gaps of rectangular frames and light the visited interiors.

    Input structure:
    - The grid has a black background with one non-corner source pixel on the outer boundary.
    - A single frame color forms several internal one-cell-thick rectangular frames.
    - Every frame has exactly two black, non-corner border gaps, and frames do not touch.

    Rule:
    - The source emits a straight beam perpendicular to its boundary side, coloring crossed black cells with the source color.
    - When the beam meets a facing gap, that frame interior is filled with the source color and the beam exits through the other gap, perpendicular to that exit side and outward.
    - A non-gap frame wall absorbs the beam without recoloring the wall; otherwise the beam stops after reaching the grid boundary.

    Key insight:
    - Each pair of gaps acts as a one-in, one-out portal: the unique chain of entered frames, not the frame shapes by themselves, determines which interiors are lit.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert np.all((0 <= g) & (g <= 9))

    source = _find_source(g)
    source_color = int(g[source])
    colors = set(map(int, g.ravel())) - {BG, source_color}
    assert len(colors) == 1, "expected one shared frame color distinct from the source"
    frame_color = next(iter(colors))

    frames = _extract_frames(g, frame_color)
    out, _visited, _terminal = _trace(g, source, frames)
    return out


# ----------------------------- generator -----------------------------

_MAX_ATTEMPTS = 1500
_FRAME_SIZE_RANGE = (4, 8)
_CHAIN_LENGTHS = (2, 3, 4)


def _add(a: tuple[int, int], b: tuple[int, int], k: int = 1) -> tuple[int, int]:
    return a[0] + b[0] * k, a[1] + b[1] * k


def _ray_cells_until(start: tuple[int, int], direction: tuple[int, int], stop: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    while (r, c) != stop:
        cells.append((r, c))
        r += dr
        c += dc
    cells.append(stop)
    return cells


def _ray_to_boundary(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    while _in_bounds(shape, r, c):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _cell_in_bbox(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _path_clear(cells: list[tuple[int, int]], frames: list[FrameSpec]) -> bool:
    return all(not _cell_in_bbox(cell, spec.bbox) for cell in cells for spec in frames)


def _bbox_clear(bbox: tuple[int, int, int, int], frames: list[FrameSpec]) -> bool:
    return all(_expanded_separated(bbox, spec.bbox) for spec in frames)


def _side_gap_cells(bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    return [(r, c1) for r in range(r0 + 1, r1)]


def _bbox_from_entry(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    dist: int,
    height: int,
    width: int,
    offset: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int]] | None:
    entry = _add(start, direction, dist)
    er, ec = entry
    side = _entry_side_for_dir(direction)
    if side == "top":
        bbox = (er, ec - offset, er + height - 1, ec - offset + width - 1)
    elif side == "bottom":
        bbox = (er - height + 1, ec - offset, er, ec - offset + width - 1)
    elif side == "left":
        bbox = (er - offset, ec, er - offset + height - 1, ec + width - 1)
    else:
        bbox = (er - offset, ec - width + 1, er - offset + height - 1, ec)
    r0, c0, r1, c1 = bbox
    h, w = shape
    if not (1 <= r0 < r1 < h - 1 and 1 <= c0 < c1 < w - 1):
        return None
    return bbox, entry


def _choose_chain_frame(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    frames: list[FrameSpec],
    require_turn: bool,
) -> tuple[FrameSpec, tuple[int, int], tuple[int, int], bool] | None:
    entry_side = _entry_side_for_dir(direction)
    straight_exit = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}[entry_side]
    choices: list[tuple[FrameSpec, tuple[int, int], tuple[int, int], bool]] = []
    for _ in range(260):
        height = random.randint(*_FRAME_SIZE_RANGE)
        width = random.randint(*_FRAME_SIZE_RANGE)
        dist = random.randint(2, 9)
        offset_limit = (width - 2) if direction[0] else (height - 2)
        offset = random.randint(1, offset_limit)
        placed = _bbox_from_entry(shape, start, direction, dist, height, width, offset)
        if placed is None:
            continue
        bbox, entry_gap = placed
        if not _bbox_clear(bbox, frames):
            continue
        if not _path_clear(_ray_cells_until(start, direction, entry_gap)[:-1], frames):
            continue

        exit_sides = [side for side in SIDES if side != entry_side]
        if require_turn:
            exit_sides = [side for side in exit_sides if side != straight_exit]
        random.shuffle(exit_sides)
        for exit_side in exit_sides:
            exit_cells = _side_gap_cells(bbox, exit_side)
            random.shuffle(exit_cells)
            for exit_gap in exit_cells:
                if exit_gap == entry_gap:
                    continue
                out_dir = OUTWARD[exit_side]
                next_start = _add(exit_gap, out_dir)
                if not _in_bounds(shape, *next_start):
                    continue
                if any(_cell_in_bbox(next_start, spec.bbox) for spec in frames):
                    continue
                spec = FrameSpec(bbox, (entry_gap, exit_gap))
                turned = out_dir != direction
                choices.append((spec, next_start, out_dir, turned))
                break
            if choices and choices[-1][0].bbox == bbox:
                break
    if not choices:
        return None
    return random.choice(choices)


def _choose_blocker(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    frames: list[FrameSpec],
) -> FrameSpec | None:
    choices: list[FrameSpec] = []
    for _ in range(260):
        height = random.randint(*_FRAME_SIZE_RANGE)
        width = random.randint(*_FRAME_SIZE_RANGE)
        dist = random.randint(2, 10)
        offset_limit = (width - 2) if direction[0] else (height - 2)
        offset = random.randint(1, offset_limit)
        placed = _bbox_from_entry(shape, start, direction, dist, height, width, offset)
        if placed is None:
            continue
        bbox, impact = placed
        if not _bbox_clear(bbox, frames):
            continue
        if not _path_clear(_ray_cells_until(start, direction, impact)[:-1], frames):
            continue
        possible = [cell for side in SIDES for cell in _side_gap_cells(bbox, side) if cell != impact]
        if len(possible) < 2:
            continue
        gaps = tuple(random.sample(possible, 2))
        choices.append(FrameSpec(bbox, (gaps[0], gaps[1])))
    if not choices:
        return None
    return random.choice(choices)


def _random_frame(
    shape: tuple[int, int],
    frames: list[FrameSpec],
    forbidden: set[tuple[int, int]],
) -> FrameSpec | None:
    h_grid, w_grid = shape
    for _ in range(400):
        height = random.randint(*_FRAME_SIZE_RANGE)
        width = random.randint(*_FRAME_SIZE_RANGE)
        r0 = random.randint(1, h_grid - height - 1)
        c0 = random.randint(1, w_grid - width - 1)
        bbox = (r0, c0, r0 + height - 1, c0 + width - 1)
        if not _bbox_clear(bbox, frames):
            continue
        if any(_cell_in_bbox(cell, bbox) for cell in forbidden):
            continue
        possible = [cell for side in SIDES for cell in _side_gap_cells(bbox, side)]
        gaps = tuple(random.sample(possible, 2))
        return FrameSpec(bbox, (gaps[0], gaps[1]))
    return None


def _render_input(shape: tuple[int, int], source: tuple[int, int], source_color: int, frames: list[FrameSpec]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    grid[source] = int(source_color)
    for spec in frames:
        for cell in _frame_pixels(spec):
            assert grid[cell] == BG
            grid[cell] = FRAME_COLOR
        for gap in spec.gaps:
            assert grid[gap] == BG
    return grid


def _random_source(shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = shape
    side = random.choice(SIDES)
    if side == "top":
        source = (0, random.randint(2, w - 3))
    elif side == "bottom":
        source = (h - 1, random.randint(2, w - 3))
    elif side == "left":
        source = (random.randint(2, h - 3), 0)
    else:
        source = (random.randint(2, h - 3), w - 1)
    return source, _source_direction(source, shape)


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(20, 30), random.randint(20, 30))
    source, direction = _random_source(shape)
    source_color = random.choice(SOURCE_COLORS)
    start = _add(source, direction)

    frames: list[FrameSpec] = []
    chain_len = random.choice(_CHAIN_LENGTHS)
    force_turn_at = random.randrange(chain_len)
    has_turn = False
    for idx in range(chain_len):
        choice = _choose_chain_frame(shape, start, direction, frames, idx == force_turn_at)
        if choice is None:
            return None
        spec, start, direction, turned = choice
        frames.append(spec)
        has_turn = has_turn or turned
    if not has_turn:
        return None

    prefer_wall = random.random() < 0.55
    if prefer_wall:
        blocker = _choose_blocker(shape, start, direction, frames)
        if blocker is not None:
            frames.append(blocker)

    provisional = _render_input(shape, source, source_color, frames)
    try:
        expected, visited, terminal = _trace(provisional, source, frames)
    except AssertionError:
        return None
    if prefer_wall and terminal != "wall":
        return None
    if terminal == "boundary":
        ray = _ray_to_boundary(shape, start, direction)
        if not _path_clear(ray, frames):
            return None

    forbidden = set(map(tuple, np.argwhere(expected == source_color)))
    distractor_target = random.randint(1, 2)
    for _ in range(distractor_target):
        distractor = _random_frame(shape, frames, forbidden)
        if distractor is None:
            return None
        frames.append(distractor)

    inp = _render_input(shape, source, source_color, frames)
    try:
        expected, visited, terminal = _trace(inp, source, frames)
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    if len(visited) < chain_len:
        return None
    if len(frames) == len(visited):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gap-ferry beam input")
