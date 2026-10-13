from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

SLOT_COUNT = 4
SLOT_SIZE = 5
STRIDE = SLOT_SIZE + 1
GRID_SIZE = SLOT_COUNT * SLOT_SIZE + (SLOT_COUNT - 1)
FRAME_SIZE = 4
FRAME_OFFSET = 1
FRAME_COLOR = GREEN
SOURCE_COLORS = (ORANGE, MAROON)
INNER_COLORS = (BLUE, RED, YELLOW, PINK, TEAL)

DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
DIR_TO_INDEX = {d: i for i, d in enumerate(DIRS8)}

BORDER_CELLS = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 3), (2, 3), (3, 3), (3, 2),
    (3, 1), (3, 0), (2, 0), (1, 0),
)

E_TOP = (1, 3)
E_BOTTOM = (2, 3)
S_LEFT = (3, 1)
S_RIGHT = (3, 2)
NE = (0, 3)
SW = (3, 0)
W_TOP = (1, 0)
W_BOTTOM = (2, 0)

SOURCE_PATTERNS = (
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 0), (0, 2)),
    ((0, 0), (0, 1), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0), (0, 2), (2, 0)),
    ((0, 0), (1, 0), (2, 0), (0, 1)),
)


@dataclass
class Frame:
    top: int
    left: int
    color: int
    gaps: list[tuple[int, int]]


@dataclass
class Beam:
    bid: int
    color: int
    pos: tuple[int, int]
    pattern: tuple[tuple[int, int], ...]
    phase: int
    path: list[tuple[int, int]]
    enter_dirs: list[tuple[int, int]]
    active: bool = True

    def next_dir(self) -> tuple[int, int]:
        return self.pattern[self.phase]

    def advance_phase(self) -> None:
        self.phase = (self.phase + 1) % len(self.pattern)


def _full_separator_indices(grid: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]


def _bands_from_separators(seps: list[int], size: int) -> list[tuple[int, int]]:
    starts = [0] + [s + 1 for s in seps]
    ends = seps + [size]
    bands = list(zip(starts, ends))
    assert len(bands) == SLOT_COUNT
    band_sizes = {hi - lo for lo, hi in bands}
    assert len(band_sizes) == 1
    return bands


def _slot_origins() -> list[int]:
    return [i * STRIDE for i in range(SLOT_COUNT)]


def _slot_origin(br: int, bc: int) -> tuple[int, int]:
    return br * STRIDE, bc * STRIDE


def _frame_top_left(br: int, bc: int) -> tuple[int, int]:
    r0, c0 = _slot_origin(br, bc)
    return r0 + FRAME_OFFSET, c0 + FRAME_OFFSET


def _component_is_4_connected(coords: list[tuple[int, int]]) -> bool:
    if not coords:
        return False
    cells = set(coords)
    stack = [coords[0]]
    seen = {coords[0]}
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _extract_source_tile(
    grid: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    r0, r1 = row_bands[0]
    c0, c1 = col_bands[0]
    coords: list[tuple[int, int, int]] = []
    for r in range(r0, r1):
        for c in range(c0, c1):
            v = int(grid[r, c])
            if v in SOURCE_COLORS:
                coords.append((r - r0, c - c0, v))
    assert 3 <= len(coords) <= 5, "expected a 3-5 pixel source tile in the top-left slot"
    assert {color for _r, _c, color in coords} == set(SOURCE_COLORS), "source tile must use both colors"
    assert _component_is_4_connected([(r, c) for r, c, _v in coords]), "source tile must be 4-connected"
    return coords


def _find_frames(grid: np.ndarray) -> list[Frame]:
    h, w = grid.shape
    frames: list[Frame] = []
    used_centers: set[tuple[int, int]] = set()
    for r0 in range(h - FRAME_SIZE + 1):
        for c0 in range(w - FRAME_SIZE + 1):
            inner = grid[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3]
            inner_vals = {int(v) for v in inner.ravel()}
            if len(inner_vals) != 1:
                continue
            color = inner_vals.pop()
            if color in {BLACK, GREY, FRAME_COLOR, ORANGE, MAROON}:
                continue

            border_vals = [int(grid[r0 + dr, c0 + dc]) for dr, dc in BORDER_CELLS]
            if any(v not in {BLACK, FRAME_COLOR} for v in border_vals):
                continue
            gaps = [(r0 + dr, c0 + dc) for dr, dc in BORDER_CELLS if int(grid[r0 + dr, c0 + dc]) == BLACK]
            if not (1 <= len(gaps) <= 2):
                continue

            center = (r0 + 1, c0 + 1)
            if center in used_centers:
                continue
            used_centers.add(center)
            frames.append(Frame(top=r0, left=c0, color=color, gaps=gaps))

    assert len(frames) == 3, "expected exactly three framed 2x2 blocks"
    return sorted(frames, key=lambda fr: (fr.top, fr.left))


def _normalize_dir(dr: int, dc: int) -> tuple[int, int]:
    if dr == 0 and dc == 0:
        return (0, 0)
    if dr != 0:
        dr = 1 if dr > 0 else -1
    if dc != 0:
        dc = 1 if dc > 0 else -1
    return dr, dc


def _gap_direction(frame: Frame, gap: tuple[int, int]) -> tuple[int, int]:
    gr = gap[0] - frame.top
    gc = gap[1] - frame.left
    dr = -1 if gr == 0 else (1 if gr == FRAME_SIZE - 1 else 0)
    dc = -1 if gc == 0 else (1 if gc == FRAME_SIZE - 1 else 0)
    direction = _normalize_dir(dr, dc)
    assert direction != (0, 0), "gap direction must be outward"
    return direction


def _dir_diff(a: tuple[int, int], b: tuple[int, int]) -> int:
    ia = DIR_TO_INDEX[a]
    ib = DIR_TO_INDEX[b]
    d = (ia - ib) % 8
    return min(d, 8 - d)


def _is_cardinal(d: tuple[int, int]) -> bool:
    return abs(d[0]) + abs(d[1]) == 1


def _is_diagonal(d: tuple[int, int]) -> bool:
    return abs(d[0]) + abs(d[1]) == 2


def _simulate_rays(grid: np.ndarray, frames: list[Frame]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    blocked: set[tuple[int, int]] = set()
    beams: dict[int, Beam] = {}
    next_bid = 0
    cell_owner: dict[tuple[int, int], int] = {}
    cell_enter: dict[tuple[int, int], tuple[int, int]] = {}
    cell_index: dict[tuple[int, int], int] = {}
    stats = {"collision_45": 0, "collision_90": 0}

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def passable(cell: tuple[int, int]) -> bool:
        r, c = cell
        return int(g[r, c]) in {BLACK, GREY} and cell not in blocked

    def paint_cell(bid: int, cell: tuple[int, int], enter_dir: tuple[int, int]) -> None:
        r, c = cell
        out[r, c] = beams[bid].color
        cell_owner[cell] = bid
        cell_enter[cell] = enter_dir
        cell_index[cell] = len(beams[bid].path) - 1

    def unpaint_cell(cell: tuple[int, int]) -> None:
        if cell not in cell_owner:
            return
        r, c = cell
        out[r, c] = g[r, c]
        cell_owner.pop(cell, None)
        cell_enter.pop(cell, None)
        cell_index.pop(cell, None)

    def truncate_beam(bid: int, keep_len: int) -> None:
        beam = beams[bid]
        if keep_len >= len(beam.path):
            return
        assert keep_len > 0
        for cell in beam.path[keep_len:]:
            if cell_owner.get(cell) == bid:
                unpaint_cell(cell)
        beam.path = beam.path[:keep_len]
        beam.enter_dirs = beam.enter_dirs[:keep_len]
        beam.pos = beam.path[-1]
        beam.active = True

    def spawn_beam(color: int, pos: tuple[int, int], direction: tuple[int, int]) -> None:
        nonlocal next_bid
        bid = next_bid
        next_bid += 1
        beam = Beam(
            bid=bid,
            color=color,
            pos=pos,
            pattern=(direction,),
            phase=0,
            path=[pos],
            enter_dirs=[direction],
        )
        beams[bid] = beam
        paint_cell(bid, pos, direction)

    def find_partner(bid: int) -> int | None:
        beam = beams[bid]
        if not beam.active:
            return None
        direction = beam.next_dir()
        partner: int | None = None
        for other_id, other in beams.items():
            if other_id == bid or not other.active:
                continue
            if other.pattern != beam.pattern or other.phase != beam.phase:
                continue
            if other.next_dir() != direction:
                continue
            if abs(other.pos[0] - beam.pos[0]) + abs(other.pos[1] - beam.pos[1]) != 1:
                continue
            if (beam.pos[0] + direction[0], beam.pos[1] + direction[1]) == other.pos:
                continue
            if (other.pos[0] + direction[0], other.pos[1] + direction[1]) == beam.pos:
                continue
            assert partner is None
            partner = other_id
        return partner

    def stop_group(ids: set[int]) -> None:
        group = set(ids)
        queue = list(ids)
        while queue:
            current = queue.pop()
            partner = find_partner(current)
            if partner is not None and partner not in group:
                group.add(partner)
                queue.append(partner)
        for gid in group:
            beams[gid].active = False

    def resolve_90_collision(cell: tuple[int, int], a: int, b: int) -> None:
        stats["collision_90"] += 1
        owner = cell_owner.get(cell)
        if owner == a:
            truncate_beam(a, cell_index[cell])
        elif owner == b:
            truncate_beam(b, cell_index[cell])
        blocked.add(cell)
        unpaint_cell(cell)
        stop_group({a, b})

    def resolve_45_collision(cell: tuple[int, int], a: int, b: int, prop: dict[int, tuple[tuple[int, int], tuple[int, int]]]) -> None:
        stats["collision_45"] += 1
        da = prop[a][1]
        db = prop[b][1] if b in prop and prop[b][0] == cell else cell_enter[cell]

        if _is_cardinal(da):
            u, v = da, db
            bid_u, bid_v = a, b
        else:
            u, v = db, da
            bid_u, bid_v = b, a

        if cell_owner.get(cell) != bid_u:
            if cell_owner.get(cell) == bid_v:
                truncate_beam(bid_v, cell_index[cell])
                unpaint_cell(cell)
            beam_u = beams[bid_u]
            beam_u.path.append(cell)
            beam_u.enter_dirs.append(u)
            beam_u.pos = cell
            paint_cell(bid_u, cell, u)
        else:
            idx = cell_index[cell]
            if idx + 1 < len(beams[bid_u].path):
                truncate_beam(bid_u, idx + 1)

        beam_v = beams[bid_v]
        adj = (beam_v.pos[0] + u[0], beam_v.pos[1] + u[1])
        assert in_bounds(adj) and passable(adj) and cell_owner.get(adj) is None
        beam_v.path.append(adj)
        beam_v.enter_dirs.append(u)
        beam_v.pos = adj
        paint_cell(bid_v, adj, u)

        zig = (v, u)
        for gid in (bid_u, bid_v):
            beams[gid].pattern = zig
            beams[gid].phase = 0
            beams[gid].active = True

    def resolve_cell_collision(cell: tuple[int, int], a: int, b: int, prop: dict[int, tuple[tuple[int, int], tuple[int, int]]]) -> None:
        da = prop[a][1]
        if b in prop and prop[b][0] == cell:
            db = prop[b][1]
        else:
            db = cell_enter[cell]
        diff = _dir_diff(da, db)
        if diff in (1, 3) and (_is_cardinal(da) != _is_cardinal(db)):
            resolve_45_collision(cell, a, b, prop)
            return
        resolve_90_collision(cell, a, b)

    def resolve_cross_collision(a: int, b: int) -> None:
        stats["collision_90"] += 1
        stop_group({a, b})

    for frame in frames:
        for gap in frame.gaps:
            direction = _gap_direction(frame, gap)
            spawn_beam(frame.color, gap, direction)

    max_iters = h * w * 200
    for _ in range(max_iters):
        active = [bid for bid, beam in beams.items() if beam.active]
        if not active:
            break

        prop: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
        for bid in active:
            beam = beams[bid]
            direction = beam.next_dir()
            nxt = (beam.pos[0] + direction[0], beam.pos[1] + direction[1])
            if not in_bounds(nxt) or not passable(nxt):
                beam.active = False
                continue
            prop[bid] = (nxt, direction)

        if not prop:
            break

        move_map: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
        cross_pairs: set[tuple[int, int]] = set()
        for bid, (nxt, d) in prop.items():
            if not _is_diagonal(d):
                continue
            move_map[(beams[bid].pos, nxt)] = bid
        for (p, n), bid in list(move_map.items()):
            pr, pc = p
            nr, nc = n
            dr = nr - pr
            dc = nc - pc
            p2 = (pr, pc + dc)
            n2 = (pr + dr, pc)
            other = move_map.get((p2, n2))
            if other is None or other == bid:
                continue
            cross_pairs.add(tuple(sorted((bid, other))))

        cell_to_bids: dict[tuple[int, int], list[int]] = {}
        for bid, (nxt, _d) in prop.items():
            cell_to_bids.setdefault(nxt, []).append(bid)

        events: list[tuple[int, int, int, int, int, str]] = []
        for cell, bidders in cell_to_bids.items():
            owner = cell_owner.get(cell)
            bidders = sorted(bidders)
            a = bidders[0]
            if owner is not None and owner != a:
                b = owner
            elif len(bidders) > 1:
                b = bidders[1]
            else:
                continue
            events.append((0, cell[0], cell[1], a, b, "cell"))
        for a, b in cross_pairs:
            ca = prop[a][0]
            cb = prop[b][0]
            events.append((1, min(ca[0], cb[0]), min(ca[1], cb[1]), a, b, "cross"))

        if events:
            events.sort()
            _, r, c, a, b, kind = events[0]
            if kind == "cell":
                resolve_cell_collision((r, c), a, b, prop)
            else:
                resolve_cross_collision(a, b)
            continue

        for bid in sorted(prop):
            nxt, direction = prop[bid]
            assert cell_owner.get(nxt) is None
            beam = beams[bid]
            beam.path.append(nxt)
            beam.enter_dirs.append(direction)
            beam.pos = nxt
            paint_cell(bid, nxt, direction)
            beam.advance_phase()
    else:
        raise AssertionError("ray simulation did not converge")

    owner_grid = np.full(g.shape, -1, dtype=int)
    for cell, owner in cell_owner.items():
        owner_grid[cell] = owner
    return out, owner_grid, stats


def _analyze(grid: np.ndarray) -> dict[str, Any]:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 23x23 grid"
    row_seps = _full_separator_indices(g, axis=0)
    col_seps = _full_separator_indices(g, axis=1)
    assert row_seps == [5, 11, 17] and col_seps == [5, 11, 17], "expected a 4x4 grey slot lattice"
    row_bands = _bands_from_separators(row_seps, g.shape[0])
    col_bands = _bands_from_separators(col_seps, g.shape[1])

    source_tile = _extract_source_tile(g, row_bands, col_bands)
    frames = _find_frames(g)
    ray_grid, owner_grid, stats = _simulate_rays(g, frames)

    beam_colors = {frame.color for frame in frames}
    slot_colors: list[list[set[int]]] = [[set() for _ in range(SLOT_COUNT)] for _ in range(SLOT_COUNT)]
    ray_pixels_per_color = {color: 0 for color in beam_colors}

    for br, (r0, r1) in enumerate(row_bands):
        for bc, (c0, c1) in enumerate(col_bands):
            for r in range(r0, r1):
                for c in range(c0, c1):
                    if owner_grid[r, c] >= 0:
                        color = int(ray_grid[r, c])
                        if color in beam_colors:
                            slot_colors[br][bc].add(color)
                            ray_pixels_per_color[color] += 1

    single_slots = [(br, bc) for br in range(SLOT_COUNT) for bc in range(SLOT_COUNT) if len(slot_colors[br][bc]) == 1]
    multicolor_slots = [(br, bc) for br in range(SLOT_COUNT) for bc in range(SLOT_COUNT) if len(slot_colors[br][bc]) >= 2]
    zero_slots = [(br, bc) for br in range(SLOT_COUNT) for bc in range(SLOT_COUNT) if len(slot_colors[br][bc]) == 0]

    return {
        "row_bands": row_bands,
        "col_bands": col_bands,
        "source_tile": source_tile,
        "frames": frames,
        "ray_grid": ray_grid,
        "owner_grid": owner_grid,
        "slot_colors": slot_colors,
        "single_slots": single_slots,
        "multicolor_slots": multicolor_slots,
        "zero_slots": zero_slots,
        "ray_pixels_per_color": ray_pixels_per_color,
        **stats,
    }


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    info = _analyze(grid)
    out = info["ray_grid"].copy()
    row_bands = info["row_bands"]
    col_bands = info["col_bands"]
    source_tile = info["source_tile"]
    slot_colors = info["slot_colors"]

    for br in range(SLOT_COUNT):
        for bc in range(SLOT_COUNT):
            if len(slot_colors[br][bc]) != 1:
                continue
            r0, _r1 = row_bands[br]
            c0, _c1 = col_bands[bc]
            for lr, lc, color in source_tile:
                out[r0 + lr, c0 + lc] = color

    src_r0, _src_r1 = row_bands[0]
    src_c0, _src_c1 = col_bands[0]
    for lr, lc, _color in source_tile:
        out[src_r0 + lr, src_c0 + lc] = BLACK

    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire rays from framed blocks, classify slots by how many ray colors reach them, and copy the source tile only into single-color slots.

    Input structure:
    - A black grid is partitioned by three full grey separator rows and columns into a 4x4 array of equal slots.
    - The top-left slot contains one small orange-and-maroon 4-connected source tile.
    - Three green 4x4 frames in other slots surround 2x2 monochrome inner blocks, and each frame has one or two border gaps.

    Rule:
    - Each gap emits an outward ray in the direction given by the gap's position relative to the framed 2x2 block.
    - Rays may paint black cells and grey separator cells, but never overwrite frames, the source tile, or already-painted ray cells; 90-degree collisions stop both rays with a black gap, while 45-degree collisions merge into a two-strand zigzag.
    - After all rays finish, every slot is classified by how many distinct ray colors appear inside its non-separator cells; the source tile is copied, aligned to the slot's top-left corner, only into slots with exactly one ray color, and the original source tile is then erased to black.

    Key insight:
    The copied tiles are not chosen directly from frame colors or positions; they emerge from the ray physics, because collisions change the paths and therefore change which slots are single-color, multi-color, or untouched.
    """
    out, _info = _solve(np.asarray(grid, dtype=int))
    return out


def _paint_source_tile(grid: np.ndarray) -> None:
    pattern = random.choice(SOURCE_PATTERNS)
    colors = [ORANGE if i % 2 == 0 else MAROON for i in range(len(pattern))]
    if random.random() < 0.5:
        colors = [MAROON if color == ORANGE else ORANGE for color in colors]
    for (r, c), color in zip(pattern, colors):
        grid[r, c] = color


def _draw_frame(grid: np.ndarray, br: int, bc: int, color: int, gaps: list[tuple[int, int]]) -> None:
    top, left = _frame_top_left(br, bc)
    patch = np.full((FRAME_SIZE, FRAME_SIZE), FRAME_COLOR, dtype=int)
    patch[1:3, 1:3] = color
    for gr, gc in gaps:
        patch[gr, gc] = BLACK
    grid[top : top + FRAME_SIZE, left : left + FRAME_SIZE] = patch


def _empty_grid() -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for idx in (5, 11, 17):
        grid[idx, :] = GREY
        grid[:, idx] = GREY
    return grid


def _template_specs() -> list[dict[str, Any]]:
    return [
        {
            "required": "45",
            "primary": [
                {"slot": (1, 0), "gap_choices": [[E_TOP], [E_BOTTOM]], "extra_choices": [None]},
                {"slot": (2, 0), "gap_choices": [[NE]], "extra_choices": [None]},
                {"slot": (0, 3), "gap_choices": [[S_LEFT], [S_RIGHT]], "extra_choices": [None, [W_TOP], [W_BOTTOM]]},
            ],
        },
        {
            "required": "90",
            "primary": [
                {"slot": (1, 0), "gap_choices": [[E_TOP]], "extra_choices": [None]},
                {"slot": (0, 1), "gap_choices": [[S_LEFT]], "extra_choices": [None]},
                {"slot": (0, 3), "gap_choices": [[S_LEFT], [S_RIGHT]], "extra_choices": [None, [W_TOP], [W_BOTTOM]]},
            ],
        },
        {
            "required": "90",
            "primary": [
                {"slot": (1, 0), "gap_choices": [[E_BOTTOM]], "extra_choices": [None]},
                {"slot": (0, 1), "gap_choices": [[S_RIGHT]], "extra_choices": [None]},
                {"slot": (0, 3), "gap_choices": [[S_LEFT], [S_RIGHT]], "extra_choices": [None, [W_TOP], [W_BOTTOM]]},
            ],
        },
        {
            "required": "45",
            "primary": [
                {"slot": (0, 1), "gap_choices": [[S_LEFT], [S_RIGHT]], "extra_choices": [None]},
                {"slot": (0, 2), "gap_choices": [[SW]], "extra_choices": [None]},
                {"slot": (3, 0), "gap_choices": [[E_TOP], [E_BOTTOM]], "extra_choices": [None, [S_LEFT], [S_RIGHT]]},
            ],
        },
    ]


def _build_candidate_input() -> tuple[np.ndarray, str]:
    grid = _empty_grid()
    _paint_source_tile(grid)

    spec = random.choice(_template_specs())
    colors = random.sample(list(INNER_COLORS), 3)
    for color, block_spec in zip(colors, spec["primary"]):
        gaps = list(random.choice(block_spec["gap_choices"]))
        extra = random.choice(block_spec["extra_choices"])
        if extra is not None:
            for gap in extra:
                if gap not in gaps:
                    gaps.append(gap)
        assert 1 <= len(gaps) <= 2
        br, bc = block_spec["slot"]
        _draw_frame(grid, br, bc, color, gaps)
    return grid, spec["required"]


def _validate_generated_case(grid: np.ndarray, required_collision: str) -> bool:
    try:
        out, info = _solve(grid)
    except AssertionError:
        return False

    single_slots = [slot for slot in info["single_slots"] if slot != (0, 0)]
    multicolor_slots = [slot for slot in info["multicolor_slots"] if slot != (0, 0)]
    zero_slots = [slot for slot in info["zero_slots"] if slot != (0, 0)]

    if len(single_slots) < 4:
        return False
    if len(multicolor_slots) < 1:
        return False
    if len(zero_slots) < 1:
        return False
    if any(count == 0 for count in info["ray_pixels_per_color"].values()):
        return False
    if required_collision == "45" and info["collision_45"] < 1:
        return False
    if required_collision == "90" and info["collision_90"] < 1:
        return False
    if np.array_equal(out, grid):
        return False

    source_tile = info["source_tile"]
    for lr, lc, _color in source_tile:
        if int(out[lr, lc]) in SOURCE_COLORS:
            return False

    copied_slots = 0
    for br, bc in single_slots:
        r0, _r1 = info["row_bands"][br]
        c0, _c1 = info["col_bands"][bc]
        if any(int(out[r0 + lr, c0 + lc]) in SOURCE_COLORS for lr, lc, _color in source_tile):
            copied_slots += 1
    if copied_slots < 4:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid, required_collision = _build_candidate_input()
        if _validate_generated_case(grid, required_collision):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
