from __future__ import annotations

from dataclasses import dataclass
import random
from collections import Counter, deque

import numpy as np


BG = 0

SIDES = ("top", "bottom", "left", "right")
STEP_BY_SIDE = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

FRAME_RING = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)
MISSING_BY_SIDE = {
    "top": ((2, 0), (2, 1), (2, 2)),
    "bottom": ((0, 0), (0, 1), (0, 2)),
    "left": ((0, 2), (1, 2), (2, 2)),
    "right": ((0, 0), (1, 0), (2, 0)),
}

SHAPES = (
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (1, 2)),
)


@dataclass
class Obj:
    kind: str
    anchor: tuple[int, int]
    rels: tuple[tuple[int, int], ...]
    colors: tuple[int, ...]
    obj_id: str
    ring_color: int | None = None
    center_color: int | None = None
    missing_rel: tuple[int, int] | None = None


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _coords(obj: Obj) -> np.ndarray:
    base = np.array(obj.anchor, dtype=int)
    rels = np.array(obj.rels, dtype=int)
    return rels + base


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _copy_obj(obj: Obj, *, anchor: tuple[int, int] | None = None, kind: str | None = None) -> Obj:
    return Obj(
        kind=obj.kind if kind is None else kind,
        anchor=obj.anchor if anchor is None else anchor,
        rels=obj.rels,
        colors=obj.colors,
        obj_id=obj.obj_id,
        ring_color=obj.ring_color,
        center_color=obj.center_color,
        missing_rel=obj.missing_rel,
    )


def _render(shape: tuple[int, int], markers: list[Obj], fixed_frames: list[Obj], moving: list[Obj]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for obj in markers + fixed_frames + moving:
        for (r, c), color in zip(_coords(obj), obj.colors):
            assert 0 <= r < shape[0] and 0 <= c < shape[1]
            assert grid[r, c] == BG
            grid[r, c] = int(color)
    return grid


def _side_of_marker(r: int, c: int, h: int, w: int) -> str:
    assert not ((r in (0, h - 1)) and (c in (0, w - 1))), "markers must not sit on corners"
    if r == 0:
        return "top"
    if r == h - 1:
        return "bottom"
    if c == 0:
        return "left"
    if c == w - 1:
        return "right"
    raise AssertionError("marker must lie on a border")


def _find_markers(grid: np.ndarray, bg: int) -> tuple[list[Obj], dict[str, int], set[tuple[int, int]]]:
    h, w = grid.shape
    border_cells = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == bg:
                continue
            if r in (0, h - 1) or c in (0, w - 1):
                border_cells.append((r, c))
    assert len(border_cells) == 4, "expected exactly four edge markers"

    side_to_color: dict[str, int] = {}
    markers: list[Obj] = []
    used = set()
    for idx, (r, c) in enumerate(border_cells):
        side = _side_of_marker(r, c, h, w)
        color = int(grid[r, c])
        assert side not in side_to_color, "expected one marker per side"
        assert color not in side_to_color.values(), "expected distinct marker colors"
        side_to_color[side] = color
        markers.append(Obj("marker", (r, c), ((0, 0),), (color,), f"marker_{idx}"))
        used.add((r, c))
    assert set(side_to_color) == set(SIDES)
    return markers, side_to_color, used


def _scan_frames(grid: np.ndarray, bg: int, reserved: set[tuple[int, int]]) -> tuple[list[Obj], list[Obj], set[tuple[int, int]]]:
    h, w = grid.shape
    full_frames: list[Obj] = []
    pseudo_frames: list[Obj] = []
    used = set(reserved)
    frame_idx = 0

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            center = int(grid[r0 + 1, c0 + 1])
            if center == bg:
                continue
            ring_values = [int(grid[r0 + dr, c0 + dc]) for dr, dc in FRAME_RING]
            ring_non_bg = [v for v in ring_values if v != bg]
            if len(set(ring_non_bg)) != 1 or not ring_non_bg:
                continue
            ring_color = ring_non_bg[0]
            if ring_color == center:
                continue

            coords = []
            colors = []
            missing_rel = None
            for (dr, dc), value in zip(FRAME_RING, ring_values):
                if value == bg:
                    if missing_rel is not None:
                        missing_rel = None
                        break
                    missing_rel = (dr, dc)
                    continue
                if value != ring_color:
                    missing_rel = None
                    break
                coords.append((dr, dc))
                colors.append(ring_color)
            if missing_rel is None and len(coords) != 8:
                continue
            if missing_rel is not None and len(coords) != 7:
                continue

            coords.append((1, 1))
            colors.append(center)
            abs_cells = {(r0 + dr, c0 + dc) for dr, dc in coords}
            if abs_cells & used:
                continue

            if missing_rel is None:
                full_frames.append(
                    Obj(
                        "full_frame",
                        (r0, c0),
                        tuple(coords),
                        tuple(colors),
                        f"full_{frame_idx}",
                        ring_color=ring_color,
                        center_color=center,
                    )
                )
            else:
                pseudo_frames.append(
                    Obj(
                        "pseudo_frame",
                        (r0, c0),
                        tuple(coords),
                        tuple(colors),
                        f"pseudo_{frame_idx}",
                        ring_color=ring_color,
                        center_color=center,
                        missing_rel=missing_rel,
                    )
                )
            used |= abs_cells
            frame_idx += 1

    return full_frames, pseudo_frames, used


def _remaining_pieces(grid: np.ndarray, bg: int, reserved: set[tuple[int, int]]) -> list[Obj]:
    h, w = grid.shape
    seen = set()
    pieces: list[Obj] = []
    idx = 0
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or (r, c) in reserved or (r, c) in seen:
                continue
            q = deque([(r, c)])
            seen.add((r, c))
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and (nr, nc) not in reserved and int(grid[nr, nc]) == color:
                        seen.add((nr, nc))
                        q.append((nr, nc))
            r0 = min(rr for rr, _ in comp)
            c0 = min(cc for _, cc in comp)
            rels = tuple((rr - r0, cc - c0) for rr, cc in sorted(comp))
            pieces.append(Obj("piece", (r0, c0), rels, tuple([color] * len(rels)), f"piece_{idx}"))
            idx += 1
    return pieces


def _parse_grid(grid: np.ndarray) -> tuple[list[Obj], list[Obj], list[Obj], dict[str, int], str]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    bg = _get_bg(g)
    assert bg == BG, "generator uses black background"

    markers, side_to_color, marker_cells = _find_markers(g, bg)
    full_frames, pseudo_frames, used = _scan_frames(g, bg, marker_cells)
    assert len(full_frames) == 1, "expected exactly one complete starting frame"
    initial_frame = full_frames[0]
    assert initial_frame.center_color in side_to_color.values(), "full-frame center must match a marker color"
    active_side = next(side for side, color in side_to_color.items() if color == initial_frame.center_color)

    moving = pseudo_frames + _remaining_pieces(g, bg, used)
    return markers, full_frames, moving, side_to_color, active_side


def _occupied(shape: tuple[int, int], objs: list[Obj]) -> np.ndarray:
    occ = np.zeros(shape, dtype=bool)
    for obj in objs:
        cells = _coords(obj)
        occ[cells[:, 0], cells[:, 1]] = True
    return occ


def _settle_key(obj: Obj, side: str) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _bbox(_coords(obj))
    if side == "top":
        return (r0, c0, r1, c1)
    if side == "bottom":
        return (-r1, c0, r0, c1)
    if side == "left":
        return (c0, r0, r1, c1)
    return (-c1, r0, r1, c0)


def _can_place(coords: np.ndarray, shape: tuple[int, int], occupied: np.ndarray) -> bool:
    h, w = shape
    if np.any(coords[:, 0] < 0) or np.any(coords[:, 0] >= h) or np.any(coords[:, 1] < 0) or np.any(coords[:, 1] >= w):
        return False
    return not np.any(occupied[coords[:, 0], coords[:, 1]])


def _settle_round(shape: tuple[int, int], side: str, fixed: list[Obj], moving: list[Obj]) -> list[Obj]:
    dr, dc = STEP_BY_SIDE[side]
    step = np.array([dr, dc], dtype=int)
    occupied = _occupied(shape, fixed)
    settled: list[Obj] = []

    for obj in sorted(moving, key=lambda item: _settle_key(item, side)):
        current = np.array(obj.anchor, dtype=int)
        rels = np.array(obj.rels, dtype=int)
        while True:
            nxt = current + step
            coords = rels + nxt
            if not _can_place(coords, shape, occupied):
                break
            current = nxt
        settled_obj = _copy_obj(obj, anchor=(int(current[0]), int(current[1])))
        coords = _coords(settled_obj)
        assert not np.any(occupied[coords[:, 0], coords[:, 1]])
        occupied[coords[:, 0], coords[:, 1]] = True
        settled.append(settled_obj)
    return settled


def _activation_candidate(settled: list[Obj], side_to_color: dict[str, int], current_side: str) -> tuple[Obj, Obj, str] | None:
    color_to_side = {color: side for side, color in side_to_color.items()}
    singletons = {}
    for obj in settled:
        if obj.kind == "piece" and len(obj.rels) == 1:
            cell = tuple(map(int, _coords(obj)[0]))
            singletons[(cell, int(obj.colors[0]))] = obj

    candidates = []
    for obj in settled:
        if obj.kind != "pseudo_frame":
            continue
        assert obj.center_color in color_to_side, "pseudo-frame center must match some marker color"
        new_side = color_to_side[obj.center_color]
        if new_side == current_side:
            continue
        mr, mc = obj.missing_rel
        gap = (obj.anchor[0] + mr, obj.anchor[1] + mc)
        filler = singletons.get((gap, int(obj.ring_color)))
        if filler is not None:
            candidates.append((obj, filler, new_side))
    assert len(candidates) <= 1, "expected at most one newly assembled compass per round"
    return candidates[0] if candidates else None


def _promote_to_fixed(obj: Obj) -> Obj:
    rels = list(FRAME_RING) + [(1, 1)]
    colors = [int(obj.ring_color)] * 8 + [int(obj.center_color)]
    return Obj(
        "full_frame",
        obj.anchor,
        tuple(rels),
        tuple(colors),
        f"fixed_{obj.obj_id}",
        ring_color=obj.ring_color,
        center_color=obj.center_color,
    )


def _simulate(grid: np.ndarray, *, trace: bool = False):
    markers, fixed_frames, moving, side_to_color, active_side = _parse_grid(grid)
    rounds = []

    while True:
        settled = _settle_round(grid.shape, active_side, markers + fixed_frames, moving)
        activation = _activation_candidate(settled, side_to_color, active_side)
        round_info = {
            "side": active_side,
            "settled": settled,
            "activation": activation,
        }
        rounds.append(round_info)
        if activation is None:
            final_grid = _render(grid.shape, markers, fixed_frames, settled)
            return (final_grid, rounds) if trace else final_grid

        pseudo_obj, filler_obj, active_side = activation
        fixed_frames = fixed_frames + [_promote_to_fixed(pseudo_obj)]
        moving = [obj for obj in settled if obj.obj_id not in {pseudo_obj.obj_id, filler_obj.obj_id}]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop every loose piece toward the edge named by the unique compass frame, then keep redirecting whenever a settled filler completes a new frame.

    Input structure:
    - Four distinct one-pixel edge markers sit on the four borders.
    - Exactly one complete 3x3 frame is present: eight border pixels share one color and the center has a different color matching one edge marker.
    - Other pieces are loose rigid parts, including incomplete 3x3 pseudo-frames with one missing border cell and a colored center.

    Rule:
    - Keep the complete frame and the four edge markers fixed.
    - Move every other piece rigidly toward the edge whose marker color matches the current frame center; settle nearer pieces first.
    - After a round finishes, if a single loose pixel exactly fills a pseudo-frame gap and creates a new complete 3x3 frame whose center points to another marker edge, freeze that new frame as an added obstacle and drop all remaining loose pieces again in the new direction.
    - Repeat until no new frame is assembled, then return the final settled grid.

    Key insight:
    - The first fall is only an assembly phase: the decisive step is spotting which pseudo-frame becomes a real compass, because that newly fixed frame redirects the next gravity round.
    """
    return _simulate(np.asarray(grid, dtype=int), trace=False)


def _make_marker(side: str, pos: int, color: int, shape: tuple[int, int], obj_id: str) -> Obj:
    h, w = shape
    if side == "top":
        return Obj("marker", (0, pos), ((0, 0),), (color,), obj_id)
    if side == "bottom":
        return Obj("marker", (h - 1, pos), ((0, 0),), (color,), obj_id)
    if side == "left":
        return Obj("marker", (pos, 0), ((0, 0),), (color,), obj_id)
    return Obj("marker", (pos, w - 1), ((0, 0),), (color,), obj_id)


def _make_full(anchor: tuple[int, int], ring_color: int, center_color: int, obj_id: str) -> Obj:
    rels = list(FRAME_RING) + [(1, 1)]
    colors = [ring_color] * 8 + [center_color]
    return Obj("full_frame", anchor, tuple(rels), tuple(colors), obj_id, ring_color=ring_color, center_color=center_color)


def _make_pseudo(anchor: tuple[int, int], ring_color: int, center_color: int, missing_rel: tuple[int, int], obj_id: str) -> Obj:
    rels = [rel for rel in FRAME_RING if rel != missing_rel] + [(1, 1)]
    colors = [ring_color] * 7 + [center_color]
    return Obj(
        "pseudo_frame",
        anchor,
        tuple(rels),
        tuple(colors),
        obj_id,
        ring_color=ring_color,
        center_color=center_color,
        missing_rel=missing_rel,
    )


def _make_piece(anchor: tuple[int, int], rels: tuple[tuple[int, int], ...], color: int, obj_id: str) -> Obj:
    return Obj("piece", anchor, rels, tuple([color] * len(rels)), obj_id)


def _shift_up(obj: Obj, amount: int) -> Obj:
    return _copy_obj(obj, anchor=(obj.anchor[0] - amount, obj.anchor[1]))


def _touches_any(obj: Obj, occupied: np.ndarray, margin: int = 0) -> bool:
    h, w = occupied.shape
    for r, c in _coords(obj):
        for dr in range(-margin, margin + 1):
            for dc in range(-margin, margin + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and occupied[rr, cc]:
                    return True
    return False


def _random_transform(grid: np.ndarray) -> np.ndarray:
    variants = [
        grid,
        np.rot90(grid, 1),
        np.rot90(grid, 2),
        np.rot90(grid, 3),
        np.fliplr(grid),
        np.flipud(grid),
        np.transpose(grid),
        np.fliplr(np.transpose(grid)),
    ]
    return np.asarray(random.choice(variants), dtype=int).copy()


def _build_canonical_case() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    shape = (h, w)

    palette = list(range(1, 10))
    random.shuffle(palette)
    marker_colors = {
        "top": palette.pop(),
        "bottom": palette.pop(),
        "left": palette.pop(),
        "right": palette.pop(),
    }
    frame_a_ring = palette.pop()
    frame_b_ring = palette.pop()
    piece_colors = [palette.pop() for _ in range(3)]

    cb = random.randint(3, w - 11)
    rb = h - 3
    missing_rel = random.choice(MISSING_BY_SIDE["bottom"])

    ra = random.randint(3, h - 12)
    ca = random.randint(w - 7, w - 5)

    full_a = _make_full((ra, ca), frame_a_ring, marker_colors["bottom"], "initial_full")
    pseudo_b = _make_pseudo((rb, cb), frame_b_ring, marker_colors["left"], missing_rel, "trigger_pseudo")
    filler_b = _make_piece((rb + missing_rel[0], cb + missing_rel[1]), ((0, 0),), frame_b_ring, "trigger_fill")

    col_c = cb + 4
    col_d = cb + 7
    if col_d >= w - 1:
        return None
    piece_c = _make_piece((h - 2, col_c), ((0, 0), (1, 0)), piece_colors[0], "stack_c")
    piece_d = _make_piece((h - 2, col_d), ((0, 0), (1, 0)), piece_colors[1], "stack_d")
    piece_e = _make_piece((ra - 2, ca + 1), ((0, 0), (1, 0)), piece_colors[2], "frame_rest")

    moving_round1 = [pseudo_b, filler_b, piece_c, piece_d, piece_e]
    fixed = [full_a]

    occ = _occupied(shape, fixed + moving_round1)
    if np.any(occ.sum(axis=0) > h) or np.any(occ.sum(axis=1) > w):
        return None

    forbidden_bottom = set(range(cb, cb + 3)) | {col_c, col_d}
    forbidden_left = {rb, rb + 1, rb + 2, ra - 2, ra - 1}
    forbidden_right = forbidden_left
    bottom_pos_choices = [c for c in range(1, w - 1) if c not in forbidden_bottom]
    top_pos_choices = [c for c in range(1, w - 1) if c not in {ca, ca + 1, ca + 2}]
    side_row_choices = [r for r in range(1, h - 1) if r not in forbidden_left]
    if not bottom_pos_choices or not top_pos_choices or len(side_row_choices) < 2:
        return None

    markers = [
        _make_marker("top", random.choice(top_pos_choices), marker_colors["top"], shape, "marker_top"),
        _make_marker("bottom", random.choice(bottom_pos_choices), marker_colors["bottom"], shape, "marker_bottom"),
        _make_marker("left", random.choice(side_row_choices), marker_colors["left"], shape, "marker_left"),
        _make_marker("right", random.choice(side_row_choices), marker_colors["right"], shape, "marker_right"),
    ]
    if len({obj.anchor for obj in markers}) != 4:
        return None

    fixed_with_markers = markers + fixed
    occupied_fixed = _occupied(shape, fixed_with_markers)

    shifts = {}
    max_shifts = {}
    for obj in moving_round1:
        max_shift = min(r for r, _ in _coords(obj))
        while max_shift > 0:
            shifted = _shift_up(obj, max_shift)
            if _touches_any(shifted, occupied_fixed, margin=0):
                max_shift -= 1
            else:
                break
        max_shifts[obj.obj_id] = max_shift
    if min(max_shifts.values()) < 1:
        return None

    for _ in range(200):
        input_objects = []
        occ0 = _occupied(shape, fixed_with_markers)

        pseudo_shift = random.randint(1, max_shifts["trigger_pseudo"])
        filler_min = pseudo_shift + 1
        if filler_min > max_shifts["trigger_fill"]:
            continue
        chosen = {
            "trigger_pseudo": pseudo_shift,
            "trigger_fill": random.randint(filler_min, max_shifts["trigger_fill"]),
        }
        for name in ("stack_c", "stack_d", "frame_rest"):
            chosen[name] = random.randint(1, max_shifts[name])

        ok = True
        for obj in moving_round1:
            shifted = _shift_up(obj, chosen[obj.obj_id])
            cells = _coords(shifted)
            if not _can_place(cells, shape, occ0):
                ok = False
                break
            if _touches_any(shifted, occ0, margin=0):
                ok = False
                break
            occ0[cells[:, 0], cells[:, 1]] = True
            input_objects.append(shifted)
        if not ok:
            continue

        grid = _render(shape, markers, fixed, input_objects)
        transformed = _random_transform(grid)
        if _validate_generated(transformed):
            return transformed
    return None


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        out, rounds = _simulate(grid, trace=True)
    except AssertionError:
        return False

    if len(rounds) < 2:
        return False
    if sum(1 for info in rounds if info["activation"] is not None) != 1:
        return False

    first = rounds[0]
    second = rounds[1]
    if first["activation"] is None:
        return False
    pseudo_obj, filler_obj, _new_side = first["activation"]
    if pseudo_obj.kind != "pseudo_frame":
        return False
    if filler_obj.kind != "piece" or len(filler_obj.rels) != 1:
        return False

    first_positions = {obj.obj_id: obj.anchor for obj in first["settled"]}
    second_positions = {obj.obj_id: obj.anchor for obj in second["settled"]}
    common = set(first_positions) & set(second_positions)
    moved_again = [obj_id for obj_id in common if first_positions[obj_id] != second_positions[obj_id]]
    if not moved_again:
        return False

    if np.array_equal(out, grid):
        return False

    round1_grid = _render(grid.shape, *_parse_grid(grid)[:2], first["settled"])  # type: ignore[arg-type]
    if np.array_equal(out, round1_grid):
        return False

    parsed = _parse_grid(grid)
    if len(parsed[1]) != 1:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _build_canonical_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chain-compass puzzle")
