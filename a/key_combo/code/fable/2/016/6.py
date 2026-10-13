from __future__ import annotations

from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _sign(n: int) -> int:
    return (n > 0) - (n < 0)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    height, width = grid.shape
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        stack = [(rr, cc)]
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_info(
    grid: np.ndarray,
    coords: np.ndarray,
    *,
    bg: int,
    frame_color: int,
) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert height >= 3 and width >= 3, "frame bbox must be at least 3x3"

    coord_set = {(int(r), int(c)) for r, c in coords}
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))

    assert all((r in (r0, r1) or c in (c0, c1)) for r, c in coord_set), "frame pixels must stay on the border"
    missing = border - coord_set
    assert len(missing) == 1, "each frame must have exactly one doorway gap"
    door = next(iter(missing))
    dr, dc = door
    if dr == r0:
        assert c0 < dc < c1, "top doorway cannot be on a corner"
        outward = (-1, 0)
    elif dr == r1:
        assert c0 < dc < c1, "bottom doorway cannot be on a corner"
        outward = (1, 0)
    elif dc == c0:
        assert r0 < dr < r1, "left doorway cannot be on a corner"
        outward = (0, -1)
    else:
        assert dc == c1 and r0 < dr < r1, "right doorway must lie on one edge"
        outward = (0, 1)

    assert len(coord_set) == len(border) - 1, "frame must be a one-cell-thick hollow rectangle"
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    inner_fg = np.argwhere(interior != bg)
    assert inner_fg.shape[0] == 1, "frame interior must contain exactly one core pixel"
    core_r, core_c = map(int, inner_fg[0])
    core = (r0 + 1 + core_r, c0 + 1 + core_c)
    core_color = int(grid[core])
    assert core_color != frame_color, "core color must differ from frame color"

    return {
        "bbox": (r0, c0, r1, c1),
        "door": door,
        "dir": outward,
        "core": core,
        "core_color": core_color,
    }


def _extract_scene(grid: np.ndarray) -> dict:
    bg = _get_bg(grid)
    counts = Counter(map(int, grid.ravel()))
    fg_counts = {color: count for color, count in counts.items() if color != bg}
    assert fg_counts, "expected some foreground"

    frame_color = max(fg_counts, key=fg_counts.get)
    frames = []
    blockers = []
    for comp in _components_of_color(grid, frame_color):
        if comp.shape[0] == 1:
            blockers.append(tuple(map(int, comp[0])))
        else:
            frames.append(_frame_info(grid, comp, bg=bg, frame_color=frame_color))

    assert len(frames) >= 2, "expected at least two frames"
    core_colors = [frame["core_color"] for frame in frames]
    assert len(set(core_colors)) == len(core_colors), "core colors must be unique"
    return {
        "bg": int(bg),
        "frame_color": int(frame_color),
        "frames": frames,
        "blockers": blockers,
    }


def _doors_face_each_other(frame_a: dict, frame_b: dict, grid: np.ndarray, bg: int) -> bool:
    door_a = frame_a["door"]
    door_b = frame_b["door"]
    if door_a[0] == door_b[0]:
        step = (0, _sign(door_b[1] - door_a[1]))
        if step[1] == 0:
            return False
        if frame_a["dir"] != step or frame_b["dir"] != (0, -step[1]):
            return False
        row = door_a[0]
        for col in range(door_a[1] + step[1], door_b[1], step[1]):
            if int(grid[row, col]) != bg:
                return False
        return True

    if door_a[1] == door_b[1]:
        step = (_sign(door_b[0] - door_a[0]), 0)
        if step[0] == 0:
            return False
        if frame_a["dir"] != step or frame_b["dir"] != (-step[0], 0):
            return False
        col = door_a[1]
        for row in range(door_a[0] + step[0], door_b[0], step[0]):
            if int(grid[row, col]) != bg:
                return False
        return True

    return False


def _find_pairs(scene: dict, grid: np.ndarray) -> list[tuple[int, int]]:
    frames = scene["frames"]
    bg = scene["bg"]
    partners = [None] * len(frames)
    pairs: list[tuple[int, int]] = []
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            if not _doors_face_each_other(frames[i], frames[j], grid, bg):
                continue
            assert partners[i] is None and partners[j] is None, "each doorway can face at most one partner"
            partners[i] = j
            partners[j] = i
            pairs.append((i, j))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """Swap core colors only between doorway frames that face each other through a clear corridor.

    Input structure
    A background grid contains several hollow rectangular frames in one shared frame color, each with
    exactly one one-cell doorway gap on one edge. Each frame interior contains a single colored core pixel,
    and the same frame color may also appear as isolated one-cell blockers elsewhere on the canvas.

    Rule
    Detect every frame's doorway and cast a ray straight outward, perpendicular to the side containing
    that gap. If two doorways lie on the same row or column, the two doorway rays point directly into one
    another, and every cell strictly between the doorway gaps is background, then those two frames are a
    mutual visible pair and their core pixel colors are exchanged. All non-paired frames, blockers, and
    doorway gaps remain unchanged.

    Key insight
    Pairing depends on doorway orientation and an unobstructed line of sight between the gaps, not on the
    positions of the core pixels themselves.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    scene = _extract_scene(g)
    out = g.copy()
    for i, j in _find_pairs(scene, g):
        frame_i = scene["frames"][i]
        frame_j = scene["frames"][j]
        out[frame_i["core"]] = frame_j["core_color"]
        out[frame_j["core"]] = frame_i["core_color"]
    return out


# --------------- generator (o5x) ---------------

GEN_BG = 0
GEN_FRAME = 5
GEN_CORE_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
GEN_ODD_SIDES = (5, 7, 9)
GEN_SCENARIOS = (
    ("clear_pair", "blocked_pair"),
    ("clear_pair", "blocked_pair", "single"),
    ("clear_pair", "single", "single"),
    ("clear_pair", "clear_pair"),
    ("clear_pair", "clear_pair", "single"),
)
GEN_SCENARIO_WEIGHTS = np.array([0.23, 0.27, 0.14, 0.19, 0.17], dtype=float)
GEN_MIN_SIDE_BY_FRAMES = {4: 22, 5: 25}
GEN_MAX_SIDE = 30
GEN_ATTEMPTS = 500
GEN_RNG = np.random.default_rng()


def _rand_odd_side(rng: np.random.Generator) -> int:
    return int(rng.choice(GEN_ODD_SIDES))


def _rand_offset(length: int, rng: np.random.Generator) -> int:
    return int(rng.integers(1, length - 1))


def _frame_patch(height: int, width: int, side: str, offset: int, core_color: int) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], tuple[int, int]]:
    patch = np.full((height, width), GEN_BG, dtype=int)
    patch[0, :] = GEN_FRAME
    patch[-1, :] = GEN_FRAME
    patch[:, 0] = GEN_FRAME
    patch[:, -1] = GEN_FRAME

    if side == "U":
        door = (0, offset)
        outward = (-1, 0)
    elif side == "D":
        door = (height - 1, offset)
        outward = (1, 0)
    elif side == "L":
        door = (offset, 0)
        outward = (0, -1)
    else:
        assert side == "R"
        door = (offset, width - 1)
        outward = (0, 1)
    patch[door] = GEN_BG

    core = (height // 2, width // 2)
    patch[core] = int(core_color)
    return patch, door, outward, core


def _placed_frame(r0: int, c0: int, patch: np.ndarray, door_local: tuple[int, int], outward: tuple[int, int], core_local: tuple[int, int], core_color: int) -> dict:
    height, width = patch.shape
    return {
        "r0": int(r0),
        "c0": int(c0),
        "patch": patch,
        "bbox": (int(r0), int(c0), int(r0 + height - 1), int(c0 + width - 1)),
        "door": (int(r0 + door_local[0]), int(c0 + door_local[1])),
        "dir": outward,
        "core": (int(r0 + core_local[0]), int(c0 + core_local[1])),
        "core_color": int(core_color),
    }


def _frame_cells(frame: dict) -> dict[tuple[int, int], int]:
    coords = np.argwhere(frame["patch"] != GEN_BG)
    cells: dict[tuple[int, int], int] = {}
    for rr, cc in coords:
        r = frame["r0"] + int(rr)
        c = frame["c0"] + int(cc)
        cells[(r, c)] = int(frame["patch"][int(rr), int(cc)])
    return cells


def _reserve_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0], r1 + pad + 1)
    cc1 = min(mask.shape[1], c1 + pad + 1)
    mask[rr0:rr1, cc0:cc1] = True


def _reserve_cell(mask: np.ndarray, cell: tuple[int, int], pad: int = 1) -> None:
    r, c = cell
    rr0 = max(0, r - pad)
    cc0 = max(0, c - pad)
    rr1 = min(mask.shape[0], r + pad + 2)
    cc1 = min(mask.shape[1], c + pad + 2)
    mask[rr0:rr1, cc0:cc1] = True


def _structure_from_parts(
    shape: tuple[int, int],
    frames: list[dict],
    blockers: list[tuple[int, int]],
    clear_pairs: list[tuple[int, int]],
) -> dict:
    cells: dict[tuple[int, int], int] = {}
    reserve = np.zeros(shape, dtype=bool)
    for frame in frames:
        for coord, color in _frame_cells(frame).items():
            assert coord not in cells, "structure frames must not overlap"
            cells[coord] = color
        _reserve_bbox(reserve, frame["bbox"], pad=1)
    for blocker in blockers:
        assert blocker not in cells, "blocker must not overlap a frame"
        cells[blocker] = GEN_FRAME
        _reserve_cell(reserve, blocker, pad=1)
    for i, j in clear_pairs:
        door_a = frames[i]["door"]
        door_b = frames[j]["door"]
        if door_a[0] == door_b[0]:
            step = _sign(door_b[1] - door_a[1])
            for col in range(door_a[1] + step, door_b[1], step):
                reserve[door_a[0], col] = True
        else:
            step = _sign(door_b[0] - door_a[0])
            for row in range(door_a[0] + step, door_b[0], step):
                reserve[row, door_a[1]] = True
    return {"frames": frames, "blockers": blockers, "clear_pairs": clear_pairs, "cells": cells, "reserve": reserve}


def _build_pair_structure(
    shape: tuple[int, int],
    colors: tuple[int, int],
    rng: np.random.Generator,
    *,
    blocked: bool,
) -> dict | None:
    height, width = shape
    for _ in range(120):
        axis = str(rng.choice(["h", "v"]))
        if axis == "h":
            h1 = _rand_odd_side(rng)
            w1 = _rand_odd_side(rng)
            h2 = _rand_odd_side(rng)
            w2 = _rand_odd_side(rng)
            max_gap = min(6, width - (w1 + w2))
            if max_gap < 1:
                continue
            gap = int(rng.integers(1, max_gap + 1))
            off1 = _rand_offset(h1, rng)
            off2 = _rand_offset(h2, rng)
            door_row_lo = max(off1, off2)
            door_row_hi = min(height - h1 + off1, height - h2 + off2)
            if door_row_lo > door_row_hi:
                continue
            door_row = int(rng.integers(door_row_lo, door_row_hi + 1))
            max_c0 = width - (w1 + gap + w2)
            if max_c0 < 0:
                continue
            c0_a = int(rng.integers(0, max_c0 + 1))
            c0_b = c0_a + w1 + gap
            r0_a = door_row - off1
            r0_b = door_row - off2
            patch_a, door_a, dir_a, core_a = _frame_patch(h1, w1, "R", off1, int(colors[0]))
            patch_b, door_b, dir_b, core_b = _frame_patch(h2, w2, "L", off2, int(colors[1]))
            frame_a = _placed_frame(r0_a, c0_a, patch_a, door_a, dir_a, core_a, int(colors[0]))
            frame_b = _placed_frame(r0_b, c0_b, patch_b, door_b, dir_b, core_b, int(colors[1]))
            blocker_choices = [(door_row, col) for col in range(frame_a["door"][1] + 1, frame_b["door"][1])]
        else:
            h1 = _rand_odd_side(rng)
            w1 = _rand_odd_side(rng)
            h2 = _rand_odd_side(rng)
            w2 = _rand_odd_side(rng)
            max_gap = min(6, height - (h1 + h2))
            if max_gap < 1:
                continue
            gap = int(rng.integers(1, max_gap + 1))
            off1 = _rand_offset(w1, rng)
            off2 = _rand_offset(w2, rng)
            door_col_lo = max(off1, off2)
            door_col_hi = min(width - w1 + off1, width - w2 + off2)
            if door_col_lo > door_col_hi:
                continue
            door_col = int(rng.integers(door_col_lo, door_col_hi + 1))
            max_r0 = height - (h1 + gap + h2)
            if max_r0 < 0:
                continue
            r0_a = int(rng.integers(0, max_r0 + 1))
            r0_b = r0_a + h1 + gap
            c0_a = door_col - off1
            c0_b = door_col - off2
            patch_a, door_a, dir_a, core_a = _frame_patch(h1, w1, "D", off1, int(colors[0]))
            patch_b, door_b, dir_b, core_b = _frame_patch(h2, w2, "U", off2, int(colors[1]))
            frame_a = _placed_frame(r0_a, c0_a, patch_a, door_a, dir_a, core_a, int(colors[0]))
            frame_b = _placed_frame(r0_b, c0_b, patch_b, door_b, dir_b, core_b, int(colors[1]))
            blocker_choices = [(row, door_col) for row in range(frame_a["door"][0] + 1, frame_b["door"][0])]

        blockers = []
        if blocked:
            if not blocker_choices:
                continue
            blockers = [tuple(map(int, blocker_choices[int(rng.integers(len(blocker_choices)))]))]
            clear_pairs = []
        else:
            clear_pairs = [(0, 1)]
        structure = _structure_from_parts(shape, [frame_a, frame_b], blockers, clear_pairs)
        door_a = frame_a["door"]
        door_b = frame_b["door"]
        if door_a[0] == door_b[0]:
            step = _sign(door_b[1] - door_a[1])
            for col in range(door_a[1] + step, door_b[1], step):
                structure["reserve"][door_a[0], col] = True
        else:
            step = _sign(door_b[0] - door_a[0])
            for row in range(door_a[0] + step, door_b[0], step):
                structure["reserve"][row, door_a[1]] = True
        return structure
    return None


def _build_single_structure(shape: tuple[int, int], color: int, rng: np.random.Generator) -> dict | None:
    height, width = shape
    for _ in range(120):
        h = _rand_odd_side(rng)
        w = _rand_odd_side(rng)
        if h > height or w > width:
            continue
        side = str(rng.choice(["U", "D", "L", "R"]))
        offset = _rand_offset(h if side in ("L", "R") else w, rng)
        r0 = int(rng.integers(0, height - h + 1))
        c0 = int(rng.integers(0, width - w + 1))
        patch, door, outward, core = _frame_patch(h, w, side, offset, int(color))
        frame = _placed_frame(r0, c0, patch, door, outward, core, int(color))
        return _structure_from_parts(shape, [frame], [], [])
    return None


def _build_expected(grid: np.ndarray, frames: list[dict], clear_pairs: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    for i, j in clear_pairs:
        frame_i = frames[i]
        frame_j = frames[j]
        out[frame_i["core"]] = frame_j["core_color"]
        out[frame_j["core"]] = frame_i["core_color"]
    return out


def _scenario_frame_count(scenario: tuple[str, ...]) -> int:
    return sum(2 if item.endswith("pair") else 1 for item in scenario)


def _gen_attempt() -> np.ndarray | None:
    rng = GEN_RNG
    scenario_idx = int(rng.choice(len(GEN_SCENARIOS), p=GEN_SCENARIO_WEIGHTS))
    scenario = GEN_SCENARIOS[scenario_idx]
    frame_count = _scenario_frame_count(scenario)
    min_side = GEN_MIN_SIDE_BY_FRAMES[frame_count]
    height = int(rng.integers(min_side, GEN_MAX_SIDE + 1))
    width = int(rng.integers(min_side, GEN_MAX_SIDE + 1))
    shape = (height, width)

    colors = rng.choice(GEN_CORE_COLORS, size=frame_count, replace=False).tolist()
    grid = np.full(shape, GEN_BG, dtype=int)
    blocked_mask = np.zeros(shape, dtype=bool)
    frames: list[dict] = []
    clear_pairs: list[tuple[int, int]] = []
    color_idx = 0

    for kind in scenario:
        if kind == "clear_pair":
            structure = _build_pair_structure(shape, (int(colors[color_idx]), int(colors[color_idx + 1])), rng, blocked=False)
            color_idx += 2
        elif kind == "blocked_pair":
            structure = _build_pair_structure(shape, (int(colors[color_idx]), int(colors[color_idx + 1])), rng, blocked=True)
            color_idx += 2
        else:
            structure = _build_single_structure(shape, int(colors[color_idx]), rng)
            color_idx += 1
        if structure is None:
            return None
        if any(blocked_mask[r, c] for r, c in structure["cells"]):
            return None

        frame_base = len(frames)
        for (r, c), color in structure["cells"].items():
            if grid[r, c] != GEN_BG:
                return None
            grid[r, c] = int(color)
        blocked_mask |= structure["reserve"]
        frames.extend(structure["frames"])
        clear_pairs.extend((frame_base + i, frame_base + j) for i, j in structure["clear_pairs"])

    expected = _build_expected(grid, frames, clear_pairs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid doorway-swap input")
