import random
from collections import Counter

import numpy as np


BACKGROUND = 0
DIRS = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
}
REL_GAPS = {
    "N": (0, 1),
    "S": (2, 1),
    "W": (1, 0),
    "E": (1, 2),
}
BORDER_RELS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)
NON_CORNER_RELS = tuple(REL_GAPS.values())
REL_TO_SIDE = {rel: side for side, rel in REL_GAPS.items()}
FRAME_COLOR_POOL = tuple(range(1, 10))
TRANSFORMS = (
    "identity",
    "flipud",
    "fliplr",
    "rot90",
    "rot180",
    "rot270",
    "transpose",
    "anti_transpose",
)
MAX_GENERATION_ATTEMPTS = 400


def _bg_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).ravel()))
    return int(counts.most_common(1)[0][0])


def _offset(coord: tuple[int, int], side: str) -> tuple[int, int]:
    dr, dc = DIRS[side]
    return coord[0] + dr, coord[1] + dc


def _frame_role(center_color: int, gap_count: int, bg: int) -> str:
    if gap_count == 1 and center_color != bg:
        return "source"
    if gap_count == 1 and center_color == bg:
        return "receiver"
    if gap_count == 2 and center_color == bg:
        return "tunnel"
    raise AssertionError("invalid frame role")


def _detect_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    frames = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            patch = g[r0 : r0 + 3, c0 : c0 + 3]
            frame_color = int(patch[0, 0])
            if frame_color == bg:
                continue
            if (
                int(patch[0, 2]) != frame_color
                or int(patch[2, 0]) != frame_color
                or int(patch[2, 2]) != frame_color
            ):
                continue

            gap_sides = []
            valid = True
            for rel in NON_CORNER_RELS:
                value = int(patch[rel])
                if value == bg:
                    gap_sides.append(REL_TO_SIDE[rel])
                elif value != frame_color:
                    valid = False
                    break
            if not valid or len(gap_sides) not in (1, 2):
                continue

            center_color = int(patch[1, 1])
            if center_color == frame_color:
                continue
            try:
                role = _frame_role(center_color, len(gap_sides), bg)
            except AssertionError:
                continue

            center = (r0 + 1, c0 + 1)
            gap_cells = {side: _offset(center, side) for side in gap_sides}
            solid_border = {
                (r0 + rr, c0 + cc)
                for rr, cc in BORDER_RELS
                if REL_TO_SIDE.get((rr, cc)) not in gap_sides
            }
            frames.append(
                {
                    "bbox": (r0, c0, r0 + 2, c0 + 2),
                    "center": center,
                    "frame_color": frame_color,
                    "center_color": center_color,
                    "gap_sides": tuple(gap_sides),
                    "gap_cells": gap_cells,
                    "solid_border": solid_border,
                    "role": role,
                }
            )

    centers = [frame["center"] for frame in frames]
    assert len(centers) == len(set(centers)), "detected overlapping 3x3 frames"
    return frames


def _build_indexes(frames: list[dict]) -> tuple[dict, dict]:
    gap_lookup = {}
    solid_lookup = {}
    for idx, frame in enumerate(frames):
        for side, cell in frame["gap_cells"].items():
            assert cell not in gap_lookup, "gap cells must be unique"
            gap_lookup[cell] = (idx, side)
        for cell in frame["solid_border"]:
            assert cell not in gap_lookup, "solid border cannot overlap a gap"
            assert cell not in solid_lookup, "solid borders must be disjoint"
            solid_lookup[cell] = idx
    return gap_lookup, solid_lookup


def _trace_projectile(frames: list[dict], start_idx: int, shape: tuple[int, int]) -> dict:
    h, w = shape
    gap_lookup, solid_lookup = _build_indexes(frames)
    start = frames[start_idx]
    assert start["role"] == "source"
    side = start["gap_sides"][0]
    pos = start["gap_cells"][side]
    direction = DIRS[side]
    visited_states = set()
    visited_cells = [start["center"], pos]
    hit_roles = []

    while True:
        state = (pos, direction)
        assert state not in visited_states, "trajectory must not cycle"
        visited_states.add(state)

        nr, nc = pos[0] + direction[0], pos[1] + direction[1]
        if not (0 <= nr < h and 0 <= nc < w):
            return {"kind": "vanish", "cells": tuple(visited_cells), "hits": tuple(hit_roles)}

        next_cell = (nr, nc)
        if next_cell in gap_lookup:
            frame_idx, entry_side = gap_lookup[next_cell]
            frame = frames[frame_idx]
            hit_roles.append(frame["role"])
            visited_cells.append(next_cell)
            if frame["role"] == "receiver":
                return {
                    "kind": "receiver",
                    "cells": tuple(visited_cells),
                    "hits": tuple(hit_roles),
                    "target": frame["center"],
                }
            assert frame["role"] == "tunnel", "only tunnels or receivers may be entered"
            exit_sides = [side_name for side_name in frame["gap_sides"] if side_name != entry_side]
            assert len(exit_sides) == 1, "tunnel must have exactly one exit gap"
            exit_side = exit_sides[0]
            pos = frame["gap_cells"][exit_side]
            direction = DIRS[exit_side]
            if pos != next_cell:
                visited_cells.append(pos)
            continue

        if next_cell in solid_lookup:
            return {
                "kind": "stop",
                "cells": tuple(visited_cells),
                "hits": tuple(hit_roles),
                "target": pos,
            }

        pos = next_cell
        visited_cells.append(pos)


def main(grid: np.ndarray) -> np.ndarray:
    """Launch each center pellet through gap-directed tunnel chains until it reaches a receiver, a wall-adjacent stop, or the edge.

    Input structure:
    - A black background contains scattered monochrome 3x3 rectangular frames.
    - Each frame has all four corners present, while one or two non-corner border cells are missing as visible gaps.
    - A one-gap frame with a non-background center pixel is a launcher, a one-gap frame with an empty center is a receiver, and a two-gap frame with an empty center is a tunnel.

    Rule:
    - Clear every launcher center to background, then move its pellet from the frame center along the direction indicated by that frame's gap.
    - A pellet flies cell by cell in a straight line; entering a receiver gap places it at that receiver center, while entering a tunnel gap immediately relays it out through the tunnel's other gap and continues with the new direction.
    - If the next step would enter any solid frame-border cell, the pellet stops on the current outside cell; if it leaves the grid, it disappears.

    Key insight:
    - Gap count determines frame role, and only exact gap hits permit passage; aligned solid borders still block the pellet outside the frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    frames = _detect_frames(g)
    assert frames, "expected at least one frame"

    sources = [idx for idx, frame in enumerate(frames) if frame["role"] == "source"]
    assert sources, "expected at least one launcher"

    out = g.copy()
    bg = _bg_color(g)
    used_targets = set()
    for idx in sources:
        frame = frames[idx]
        out[frame["center"]] = bg

    for idx in sources:
        frame = frames[idx]
        result = _trace_projectile(frames, idx, g.shape)
        if result["kind"] == "vanish":
            continue
        target = result["target"]
        assert target not in used_targets, "pellet destinations must be unique"
        used_targets.add(target)
        assert out[target] == bg, "pellets may only land on background or empty receiver centers"
        out[target] = frame["center_color"]

    return out


def _make_frame(center: tuple[int, int], gap_sides: tuple[str, ...], frame_color: int, center_color: int) -> dict:
    role = _frame_role(center_color, len(gap_sides), BACKGROUND)
    return {
        "center": center,
        "gap_sides": tuple(gap_sides),
        "frame_color": int(frame_color),
        "center_color": int(center_color),
        "role": role,
    }


def _paint_frame(grid: np.ndarray, frame: dict) -> None:
    r, c = frame["center"]
    color = frame["frame_color"]
    grid[r - 1 : r + 2, c - 1 : c + 2] = color
    grid[r, c] = frame["center_color"]
    for side in frame["gap_sides"]:
        gr, gc = _offset((r, c), side)
        grid[gr, gc] = BACKGROUND


def _spec_indexes(frames: list[dict]) -> tuple[dict, dict]:
    gap_lookup = {}
    solid_lookup = {}
    for idx, frame in enumerate(frames):
        r, c = frame["center"]
        gap_cells = {side: _offset((r, c), side) for side in frame["gap_sides"]}
        for side, cell in gap_cells.items():
            assert cell not in gap_lookup
            gap_lookup[cell] = (idx, side)
        for rr, cc in BORDER_RELS:
            cell = (r - 1 + rr, c - 1 + cc)
            side = REL_TO_SIDE.get((rr, cc))
            if side in frame["gap_sides"]:
                continue
            assert cell not in gap_lookup
            assert cell not in solid_lookup
            solid_lookup[cell] = idx
    return gap_lookup, solid_lookup


def _trace_from_specs(frames: list[dict], start_idx: int, shape: tuple[int, int]) -> dict:
    h, w = shape
    gap_lookup, solid_lookup = _spec_indexes(frames)
    frame = frames[start_idx]
    side = frame["gap_sides"][0]
    pos = _offset(frame["center"], side)
    direction = DIRS[side]
    visited_states = set()
    visited_cells = [frame["center"], pos]
    tunnel_hits = 0

    while True:
        state = (pos, direction)
        assert state not in visited_states
        visited_states.add(state)

        nr, nc = pos[0] + direction[0], pos[1] + direction[1]
        if not (0 <= nr < h and 0 <= nc < w):
            return {
                "kind": "vanish",
                "cells": tuple(visited_cells),
                "tunnel_hits": tunnel_hits,
            }

        next_cell = (nr, nc)
        if next_cell in gap_lookup:
            frame_idx, entry_side = gap_lookup[next_cell]
            target = frames[frame_idx]
            visited_cells.append(next_cell)
            if target["role"] == "receiver":
                return {
                    "kind": "receiver",
                    "cells": tuple(visited_cells),
                    "tunnel_hits": tunnel_hits,
                    "target": target["center"],
                }
            assert target["role"] == "tunnel"
            tunnel_hits += 1
            exit_sides = [side_name for side_name in target["gap_sides"] if side_name != entry_side]
            assert len(exit_sides) == 1
            exit_side = exit_sides[0]
            pos = _offset(target["center"], exit_side)
            direction = DIRS[exit_side]
            if pos != next_cell:
                visited_cells.append(pos)
            continue

        if next_cell in solid_lookup:
            return {
                "kind": "stop",
                "cells": tuple(visited_cells),
                "tunnel_hits": tunnel_hits,
                "target": pos,
                "blocked_by": frames[solid_lookup[next_cell]]["center"],
            }

        pos = next_cell
        visited_cells.append(pos)


def _expected_output(inp: np.ndarray, frames: list[dict], traces: dict[int, dict]) -> np.ndarray:
    out = inp.copy()
    for idx, frame in enumerate(frames):
        if frame["role"] == "source":
            out[frame["center"]] = BACKGROUND
            result = traces[idx]
            if result["kind"] != "vanish":
                out[result["target"]] = frame["center_color"]
    return out


def _apply_transform(grid: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return grid.copy()
    if name == "flipud":
        return np.flipud(grid)
    if name == "fliplr":
        return np.fliplr(grid)
    if name == "rot90":
        return np.rot90(grid, 1)
    if name == "rot180":
        return np.rot90(grid, 2)
    if name == "rot270":
        return np.rot90(grid, 3)
    if name == "transpose":
        return grid.T.copy()
    if name == "anti_transpose":
        return np.flipud(np.rot90(grid, 1))
    raise ValueError(f"unknown transform {name}")


def _random_color(exclude: set[int]) -> int:
    choices = [color for color in FRAME_COLOR_POOL if color not in exclude]
    return int(random.choice(choices))


def _sample_rows_cols(height: int, width: int) -> tuple[list[int], list[int]] | None:
    row_candidates = list(range(2, height - 1, 4))
    col_candidates = list(range(2, width - 1, 4))
    if len(row_candidates) < 5 or len(col_candidates) < 5:
        return None
    rows = sorted(random.sample(row_candidates, 5))
    cols = sorted(random.sample(col_candidates, 5))
    return rows, cols


def _build_layout(height: int, width: int) -> tuple[list[dict], dict[int, str]] | None:
    sampled = _sample_rows_cols(height, width)
    if sampled is None:
        return None
    rows, cols = sampled
    r0, r1, r2, r3, r4 = rows
    c0, c1, c2, c3, c4 = cols

    a_turn_col = random.choice((c1, c2))
    a_end_col = random.choice(tuple(col for col in (c3, c4) if col > a_turn_col))
    b_turn_col = random.choice((c1, c2, c3))
    c_source_col = random.choice((c2, c3, c4))

    used_projectile_colors = set()

    def source(center: tuple[int, int], gap_side: str) -> dict:
        frame_color = _random_color(set())
        pellet_color = _random_color({frame_color} | used_projectile_colors)
        used_projectile_colors.add(pellet_color)
        return _make_frame(center, (gap_side,), frame_color, pellet_color)

    def empty_frame(center: tuple[int, int], gap_sides: tuple[str, ...]) -> dict:
        frame_color = _random_color(set())
        return _make_frame(center, gap_sides, frame_color, BACKGROUND)

    frames = [
        source((r0, c0), "E"),
        empty_frame((r0, a_turn_col), ("W", "S")),
        empty_frame((r1, a_turn_col), ("N", "E")),
        empty_frame((r1, a_end_col), ("W",)),
        source((r2, c0), "E"),
        empty_frame((r2, b_turn_col), ("W", "S")),
        None,
        source((r4, c_source_col), "E"),
    ]

    blocker_role = random.choice(("receiver", "tunnel"))
    if blocker_role == "receiver":
        blocker_gaps = (random.choice(("S", "E", "W")),)
    else:
        blocker_gaps = tuple(sorted(random.sample(("S", "E", "W"), 2)))
    frames[6] = empty_frame((r3, b_turn_col), blocker_gaps)

    expected_kinds = {
        0: "receiver",
        4: "stop",
        7: "vanish",
    }
    return frames, expected_kinds


def _validate_generated(inp: np.ndarray, frames: list[dict], expected_kinds: dict[int, str]) -> np.ndarray | None:
    traces = {}
    used_path_cells = set()
    used_targets = set()

    for idx, frame in enumerate(frames):
        if frame["role"] != "source":
            continue
        trace = _trace_from_specs(frames, idx, inp.shape)
        traces[idx] = trace
        if trace["kind"] != expected_kinds[idx]:
            return None

        path_cells = set(trace["cells"])
        if used_path_cells & path_cells:
            return None
        used_path_cells |= path_cells

        if trace["kind"] == "receiver":
            if idx == 0 and trace["tunnel_hits"] < 2:
                return None
        if trace["kind"] == "stop":
            if idx == 4 and trace["tunnel_hits"] < 1:
                return None
            target = trace["target"]
            if target in used_targets:
                return None
            used_targets.add(target)
        if trace["kind"] == "receiver":
            target = trace["target"]
            if target in used_targets:
                return None
            used_targets.add(target)

    expected = _expected_output(inp, frames, traces)
    detected = _detect_frames(inp)
    if len(detected) != len(frames):
        return None
    if sum(frame["role"] == "source" for frame in detected) != 3:
        return None

    solved = main(inp)
    if not np.array_equal(expected, solved):
        return None
    return expected


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(20, 30)
    width = random.randint(20, 30)
    built = _build_layout(height, width)
    if built is None:
        return None
    frames, expected_kinds = built

    inp = np.zeros((height, width), dtype=int)
    for frame in frames:
        _paint_frame(inp, frame)

    expected = _validate_generated(inp, frames, expected_kinds)
    if expected is None:
        return None

    transform = random.choice(TRANSFORMS)
    inp = _apply_transform(inp, transform)
    expected = _apply_transform(expected, transform)
    detected = _detect_frames(inp)
    if len(detected) != len(frames):
        return None
    if sum(frame["role"] == "source" for frame in detected) != 3:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ballistic tunnel puzzle")
