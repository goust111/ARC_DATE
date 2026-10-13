from __future__ import annotations

from collections import Counter
import random

import numpy as np


SIDE_DIRS = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
}
OPPOSITE_SIDE = {"N": "S", "S": "N", "W": "E", "E": "W"}
OPPOSITE_DIR = {v: (-v[0], -v[1]) for v in SIDE_DIRS.values()}
ALL_SIDES = tuple(SIDE_DIRS)
SLOTS = (1, 2)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _border_coords(r0: int, c0: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for dr in range(4):
        for dc in range(4):
            if dr in (0, 3) or dc in (0, 3):
                coords.append((r0 + dr, c0 + dc))
    return coords


def _gap_pos(r0: int, c0: int, side: str, slot: int) -> tuple[int, int]:
    slot = int(slot)
    if side == "N":
        return r0, c0 + slot
    if side == "S":
        return r0 + 3, c0 + slot
    if side == "W":
        return r0 + slot, c0
    if side == "E":
        return r0 + slot, c0 + 3
    raise ValueError("bad side")


def _gap_side_slot(r0: int, c0: int, gap: tuple[int, int]) -> tuple[str, int] | None:
    gr, gc = gap
    if gr == r0 and c0 < gc < c0 + 3:
        return "N", gc - c0
    if gr == r0 + 3 and c0 < gc < c0 + 3:
        return "S", gc - c0
    if gc == c0 and r0 < gr < r0 + 3:
        return "W", gr - r0
    if gc == c0 + 3 and r0 < gr < r0 + 3:
        return "E", gr - r0
    return None


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    frames: list[dict] = []
    for r0 in range(h - 3):
        for c0 in range(w - 3):
            inner = grid[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3]
            inner_colors = np.unique(inner)
            if len(inner_colors) != 1:
                continue
            inner_color = int(inner_colors[0])
            if inner_color == bg:
                continue

            border = _border_coords(r0, c0)
            gap_cells = [(r, c) for r, c in border if int(grid[r, c]) == bg]
            if len(gap_cells) != 1:
                continue
            side_slot = _gap_side_slot(r0, c0, gap_cells[0])
            if side_slot is None:
                continue

            wall_values = [int(grid[r, c]) for r, c in border if int(grid[r, c]) != bg]
            if len(wall_values) != 11 or len(set(wall_values)) != 1:
                continue
            wall_color = wall_values[0]
            if wall_color == inner_color:
                continue

            side, slot = side_slot
            frames.append(
                {
                    "r": int(r0),
                    "c": int(c0),
                    "gap": gap_cells[0],
                    "side": side,
                    "slot": int(slot),
                    "dir": SIDE_DIRS[side],
                    "wall": int(wall_color),
                    "color": int(inner_color),
                }
            )

    assert frames, "expected at least one 4x4 framed block"
    assert len({frame["wall"] for frame in frames}) == 1, "all frames must use one wall color"

    occupied: set[tuple[int, int]] = set()
    for frame in sorted(frames, key=lambda f: (f["r"], f["c"])):
        cells = {(frame["r"] + dr, frame["c"] + dc) for dr in range(4) for dc in range(4)}
        assert not (occupied & cells), "4x4 frame boxes must not overlap"
        occupied |= cells
    return sorted(frames, key=lambda f: (f["r"], f["c"]))


def _trace_signal(
    grid: np.ndarray, frames: list[dict], gap_to_idx: dict[tuple[int, int], int], sender: int, bg: int
) -> tuple[int | None, str]:
    h, w = grid.shape
    frame = frames[sender]
    dr, dc = frame["dir"]
    r, c = frame["gap"][0] + dr, frame["gap"][1] + dc
    while 0 <= r < h and 0 <= c < w:
        pos = (int(r), int(c))
        if pos in gap_to_idx:
            receiver = gap_to_idx[pos]
            if receiver != sender and frames[receiver]["dir"] == OPPOSITE_DIR[(dr, dc)]:
                return receiver, "receiver"
            return None, "wrong_gap"
        if int(grid[r, c]) != bg:
            return None, "blocked"
        r += dr
        c += dc
    return None, "edge"


def _transfers(grid: np.ndarray, frames: list[dict], bg: int) -> tuple[dict[int, int], list[str]]:
    gap_to_idx = {frame["gap"]: i for i, frame in enumerate(frames)}
    assert len(gap_to_idx) == len(frames), "gap positions must be unique"

    incoming: dict[int, int] = {}
    reasons: list[str] = []
    for sender in range(len(frames)):
        receiver, reason = _trace_signal(grid, frames, gap_to_idx, sender, bg)
        reasons.append(reason)
        if receiver is not None:
            assert receiver not in incoming, "each block may receive at most one signal"
            incoming[receiver] = sender
    return incoming, reasons


def _paint_inner(grid: np.ndarray, frame: dict, color: int) -> None:
    r0, c0 = frame["r"], frame["c"]
    grid[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each framed 2x2 block from the facing gap signal that reaches it, and erase unfed blocks.

    Input structure:
    The background is the most common color.  A single wall color forms several
    separated 4x4 frames, each with exactly one non-corner background gap in its
    border and one solid 2x2 colored block inside the frame.

    Rule:
    Each gap sends an invisible straight signal outward along that side's normal.
    The signal may pass only through background cells; if it first meets another
    frame's gap from the outside face, that receiver's 2x2 block becomes the
    sender's original block color.  Signals are simultaneous.  Blocks with no
    received signal are erased to the background while walls and gaps stay fixed.

    Key insight:
    The visible line is missing; the color transfer is determined only by which
    framed gaps geometrically face one another through unobstructed background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    frames = _find_frames(g, bg)
    incoming, _reasons = _transfers(g, frames, bg)

    out = g.copy()
    for frame in frames:
        _paint_inner(out, frame, bg)
    for receiver, sender in incoming.items():
        _paint_inner(out, frames[receiver], frames[sender]["color"])
    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 600
PALETTE = tuple(range(10))


def _sample_lattice() -> tuple[int, int, list[tuple[int, int, int, int]]]:
    row_count = random.randint(3, 4)
    col_count = random.randint(3, 4)
    row_step = random.randint(5, 7)
    col_step = random.randint(5, 7)
    min_h = (row_count - 1) * row_step + 4
    min_w = (col_count - 1) * col_step + 4

    top_margin = random.randint(1, 30 - min_h - 1)
    left_margin = random.randint(1, 30 - min_w - 1)
    bottom_margin = random.randint(1, 30 - min_h - top_margin)
    right_margin = random.randint(1, 30 - min_w - left_margin)
    height = top_margin + min_h + bottom_margin
    width = left_margin + min_w + right_margin

    positions = []
    for i in range(row_count):
        for j in range(col_count):
            positions.append((i, j, top_margin + i * row_step, left_margin + j * col_step))
    return height, width, positions


def _lattice_edges(positions: list[tuple[int, int, int, int]]) -> list[tuple[tuple[int, int], tuple[int, int], str]]:
    by_ij = {(i, j): (r, c) for i, j, r, c in positions}
    edges = []
    for i, j in by_ij:
        if (i, j + 1) in by_ij:
            edges.append(((i, j), (i, j + 1), "h"))
        if (i + 1, j) in by_ij:
            edges.append(((i, j), (i + 1, j), "v"))
    random.shuffle(edges)
    return edges


def _choose_pairs(positions: list[tuple[int, int, int, int]]) -> list[tuple[tuple[int, int], tuple[int, int], str]]:
    pair_count = random.choice([1, 1, 2])
    chosen = []
    used: set[tuple[int, int]] = set()
    for a, b, orient in _lattice_edges(positions):
        if a in used or b in used:
            continue
        chosen.append((a, b, orient))
        used.add(a)
        used.add(b)
        if len(chosen) == pair_count:
            break
    return chosen


def _nearest_occupied_in_direction(
    pos: tuple[int, int], occupied: dict[tuple[int, int], dict], side: str
) -> tuple[int, int] | None:
    i, j = pos
    candidates = []
    for oi, oj in occupied:
        if side == "N" and oj == j and oi < i:
            candidates.append((i - oi, (oi, oj)))
        elif side == "S" and oj == j and oi > i:
            candidates.append((oi - i, (oi, oj)))
        elif side == "W" and oi == i and oj < j:
            candidates.append((j - oj, (oi, oj)))
        elif side == "E" and oi == i and oj > j:
            candidates.append((oj - j, (oi, oj)))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _blocked_gap_option(pos: tuple[int, int], occupied: dict[tuple[int, int], dict]) -> tuple[str, int] | None:
    sides = list(ALL_SIDES)
    random.shuffle(sides)
    for side in sides:
        target = _nearest_occupied_in_direction(pos, occupied, side)
        if target is None:
            continue
        target_side = OPPOSITE_SIDE[side]
        target_info = occupied[target]
        possible = list(SLOTS)
        if target_info["side"] == target_side:
            possible = [slot for slot in possible if slot != target_info["slot"]]
        if possible:
            return side, random.choice(possible)
    return None


def _edge_sides(pos: tuple[int, int], positions: list[tuple[int, int, int, int]]) -> list[str]:
    rows = [i for i, _j, _r, _c in positions]
    cols = [j for _i, j, _r, _c in positions]
    i, j = pos
    sides = []
    if i == min(rows):
        sides.append("N")
    if i == max(rows):
        sides.append("S")
    if j == min(cols):
        sides.append("W")
    if j == max(cols):
        sides.append("E")
    return sides


def _paint_frame(grid: np.ndarray, r0: int, c0: int, side: str, slot: int, wall: int, color: int, bg: int) -> None:
    assert np.all(grid[r0 : r0 + 4, c0 : c0 + 4] == bg)
    for r, c in _border_coords(r0, c0):
        grid[r, c] = wall
    grid[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3] = color
    gr, gc = _gap_pos(r0, c0, side, slot)
    grid[gr, gc] = bg


def _render_input(
    height: int,
    width: int,
    positions: list[tuple[int, int, int, int]],
    placement: dict[tuple[int, int], dict],
    bg: int,
    wall: int,
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int]]]:
    grid = np.full((height, width), bg, dtype=int)
    by_ij = {(i, j): (r, c) for i, j, r, c in positions}
    top_left_by_pos: dict[tuple[int, int], tuple[int, int]] = {}
    for pos, info in placement.items():
        r0, c0 = by_ij[pos]
        _paint_frame(grid, r0, c0, info["side"], info["slot"], wall, info["color"], bg)
        top_left_by_pos[pos] = (r0, c0)
    return grid, top_left_by_pos


def _expected_from_pairs(
    grid: np.ndarray,
    detected: list[dict],
    top_left_to_pos: dict[tuple[int, int], tuple[int, int]],
    partner: dict[tuple[int, int], tuple[int, int]],
    bg: int,
) -> np.ndarray:
    frame_by_pos = {(frame["r"], frame["c"]): idx for idx, frame in enumerate(detected)}
    pos_to_idx = {pos: frame_by_pos[top_left] for top_left, pos in top_left_to_pos.items()}
    out = grid.copy()
    for frame in detected:
        _paint_inner(out, frame, bg)
    for pos, other in partner.items():
        receiver = pos_to_idx[pos]
        sender = pos_to_idx[other]
        _paint_inner(out, detected[receiver], detected[sender]["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    height, width, positions = _sample_lattice()
    pairs = _choose_pairs(positions)
    if not pairs:
        return None

    placement: dict[tuple[int, int], dict] = {}
    partner: dict[tuple[int, int], tuple[int, int]] = {}
    for a, b, orient in pairs:
        slot = random.choice(SLOTS)
        if orient == "h":
            placement[a] = {"side": "E", "slot": slot}
            placement[b] = {"side": "W", "slot": slot}
        else:
            placement[a] = {"side": "S", "slot": slot}
            placement[b] = {"side": "N", "slot": slot}
        partner[a] = b
        partner[b] = a

    remaining = [(i, j) for i, j, _r, _c in positions if (i, j) not in placement]
    random.shuffle(remaining)

    blocker_candidates = [(pos, _blocked_gap_option(pos, placement)) for pos in remaining]
    blocker_candidates = [(pos, option) for pos, option in blocker_candidates if option is not None]
    if not blocker_candidates:
        return None
    blocker_pos, blocker_option = random.choice(blocker_candidates)
    assert blocker_option is not None
    side, slot = blocker_option
    placement[blocker_pos] = {"side": side, "slot": slot, "blocked": True}
    remaining = [pos for pos in remaining if pos != blocker_pos]

    extra_count = random.randint(0, min(2, len(remaining)))
    edge_first = sorted(remaining, key=lambda pos: 0 if _edge_sides(pos, positions) else 1)
    random.shuffle(edge_first)
    for pos in edge_first[:extra_count]:
        edges = _edge_sides(pos, positions)
        if edges:
            side = random.choice(edges)
        else:
            side = random.choice(ALL_SIDES)
        placement[pos] = {"side": side, "slot": random.choice(SLOTS)}

    frame_count = len(placement)
    if frame_count > 8:
        return None

    bg = random.choice(PALETTE)
    wall = random.choice([c for c in PALETTE if c != bg])
    colors = random.sample([c for c in PALETTE if c not in (bg, wall)], frame_count)
    for info, color in zip(placement.values(), colors):
        info["color"] = int(color)

    grid, top_left_by_pos = _render_input(height, width, positions, placement, bg, wall)
    top_left_to_pos = {top_left: pos for pos, top_left in top_left_by_pos.items()}

    try:
        detected = _find_frames(grid, bg)
        if len(detected) != frame_count:
            return None
        incoming, reasons = _transfers(grid, detected, bg)
        frame_by_top_left = {(frame["r"], frame["c"]): idx for idx, frame in enumerate(detected)}
        pos_to_idx = {pos: frame_by_top_left[top_left] for pos, top_left in top_left_by_pos.items()}
        expected_incoming = {pos_to_idx[pos]: pos_to_idx[other] for pos, other in partner.items()}
        if incoming != expected_incoming:
            return None
        if "blocked" not in reasons:
            return None
        if len(incoming) < 2 or len(incoming) >= frame_count:
            return None

        expected = _expected_from_pairs(grid, detected, top_left_to_pos, partner, bg)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return None
        if np.array_equal(actual, grid):
            return None

        erased = 0
        changed_receivers = 0
        for idx, frame in enumerate(detected):
            patch_in = grid[frame["r"] + 1 : frame["r"] + 3, frame["c"] + 1 : frame["c"] + 3]
            patch_out = actual[frame["r"] + 1 : frame["r"] + 3, frame["c"] + 1 : frame["c"] + 3]
            if idx not in incoming:
                if not np.all(patch_out == bg):
                    return None
                erased += 1
            elif not np.array_equal(patch_in, patch_out):
                changed_receivers += 1
        if erased < 1 or changed_receivers < len(incoming):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid framed-gap transfer input")
