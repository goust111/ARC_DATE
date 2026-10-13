from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, crop_bbox, paste


_DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
_OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
_EDGE_MID = {"up": (0, 1), "down": (2, 1), "left": (1, 0), "right": (1, 2)}
_MARKER_PATTERNS = {
    "up": [[(0, 1)], [(0, 1), (0, 0)], [(0, 1), (0, 2)]],
    "down": [[(2, 1)], [(2, 1), (2, 0)], [(2, 1), (2, 2)]],
    "left": [[(1, 0)], [(1, 0), (0, 0)], [(1, 0), (2, 0)]],
    "right": [[(1, 2)], [(1, 2), (0, 2)], [(1, 2), (2, 2)]],
}
_MARKER_COLORS = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]
_RETRY_LIMIT = 500


def _sign(n: int) -> int:
    if n > 0:
        return 1
    if n < 0:
        return -1
    return 0


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _top_left_box(top_left: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = top_left
    return int(r), int(c), int(r) + 2, int(c) + 2


def _classify_block(patch: np.ndarray) -> dict[str, object]:
    black_coords = np.argwhere(patch == Color.BLACK)
    color_coords = np.argwhere((patch != Color.BLACK) & (patch != Color.GREY))
    info: dict[str, object] = {
        "patch": patch,
        "color_coords": color_coords,
    }
    if color_coords.size == 0:
        assert np.all(patch == Color.GREY)
        info["kind"] = "blank"
        return info

    candidates = []
    if np.all(color_coords[:, 0] == 0):
        candidates.append("up")
    if np.all(color_coords[:, 0] == 2):
        candidates.append("down")
    if np.all(color_coords[:, 1] == 0):
        candidates.append("left")
    if np.all(color_coords[:, 1] == 2):
        candidates.append("right")
    assert len(candidates) == 1, "colored direction markers must lie on exactly one edge"
    info["direction"] = candidates[0]

    if black_coords.shape[0] == 1:
        gap = tuple(map(int, black_coords[0]))
        assert gap in {(0, 1), (2, 1), (1, 0), (1, 2)}
        info["kind"] = "door"
        info["gap_local"] = gap
    else:
        assert black_coords.shape[0] == 0
        info["kind"] = "normal"
    return info


def _detect_blocks(grid: np.ndarray) -> list[dict[str, object]]:
    blocks = []
    for coords in components_mask(grid != Color.BLACK, connectivity=4):
        b = bbox_coords(coords)
        assert b[2] - b[0] == 2 and b[3] - b[1] == 2, "every block must be 3x3"
        patch = crop_bbox(grid, b)
        info = _classify_block(patch)
        info["bbox"] = b
        blocks.append(info)
    blocks.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return blocks


def _slide_top_left(direction: str, top_left: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    r, c = map(int, top_left)
    if direction == "up":
        return 0, c
    if direction == "down":
        return h - 3, c
    if direction == "left":
        return r, 0
    assert direction == "right"
    return r, w - 3


def _reachable(
    grid: np.ndarray,
    seed: tuple[int, int],
    moves: list[tuple[int, int]],
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    r0, c0, r1, c1 = bounds
    assert grid[seed] == Color.BLACK
    seen = np.zeros(grid.shape, dtype=bool)
    seen[seed] = True
    q = deque([seed])
    while q:
        r, c = q.popleft()
        for dr, dc in moves:
            rr, cc = r + dr, c + dc
            if r0 <= rr <= r1 and c0 <= cc <= c1 and grid[rr, cc] == Color.BLACK and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen


def _solve_layout(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    blocks = _detect_blocks(g)
    doors: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for block in blocks:
        kind = block["kind"]
        r0, c0, r1, c1 = block["bbox"]
        if kind == "blank":
            continue

        patch = block["patch"]
        direction = block["direction"]
        out[r0 : r1 + 1, c0 : c1 + 1] = Color.BLACK
        dst_r, dst_c = _slide_top_left(direction, (r0, c0), g.shape)
        paste(out, patch, dst_r, dst_c)

        if kind == "door":
            gap_r, gap_c = block["gap_local"]
            global_gap = (dst_r + int(gap_r), dst_c + int(gap_c))
            gap_dir = _DIRS[_OPPOSITE[direction]]
            outer = (global_gap[0] + gap_dir[0], global_gap[1] + gap_dir[1])
            assert 0 <= outer[0] < out.shape[0] and 0 <= outer[1] < out.shape[1]
            assert out[global_gap] == Color.BLACK
            assert out[outer] == Color.BLACK
            doors.append((global_gap, outer))

    assert len(doors) == 2, "expected exactly two doorway blocks"
    start = doors[0][1]
    end = doors[1][1]

    moves = []
    row_step = _sign(end[0] - start[0])
    col_step = _sign(end[1] - start[1])
    if row_step:
        moves.append((row_step, 0))
    if col_step:
        moves.append((0, col_step))
    assert moves

    bounds = (
        min(start[0], end[0]),
        min(start[1], end[1]),
        max(start[0], end[0]),
        max(start[1], end[1]),
    )
    forward = _reachable(out, start, moves, bounds)
    backward = _reachable(out, end, [(-dr, -dc) for dr, dc in moves], bounds)
    path = forward & backward
    assert path[start] and path[end], "door exteriors must admit a monotone shortest path"
    return out, [door[0] for door in doors], path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide direction-coded 3x3 blocks to the boundary, then paint every monotone shortest corridor between the two doorway blocks.

    Input structure:
    - The grid is a black background containing 5-7 separate gray 3x3 blocks.
    - One block is all gray, two are doorway blocks with one black edge-center gap, and the remaining blocks contain colored edge markers that encode a slide direction.
    - Doorway and ordinary blocks use colored marker pixels on exactly one edge; the blank gray block has no colored marker and does not move.

    Rule:
    - Detect each 3x3 block and ignore the all-gray blank block.
    - Decode each remaining block's slide direction from the edge containing its colored marker pixels, then slide the entire 3x3 patch until that edge of the patch touches the grid boundary.
    - Treat the slid gray pixels as walls, find the two doorway gaps and the background cell immediately outside each gap on the interior-facing side, and compute the union of all wall-avoiding shortest monotone paths between those two exterior cells.
    - Paint that path union yellow, also paint both doorway gap cells yellow, and keep all gray walls and colored direction markers unchanged.

    Key insight:
    - The colored edge markers first determine where the wall blocks land, and that new wall geometry then determines which monotone shortest corridor cells can be reached between the two doorway openings.
    """
    moved, gap_cells, path = _solve_layout(grid)
    out = moved.copy()
    out[path] = Color.YELLOW
    for gap in gap_cells:
        out[gap] = Color.YELLOW
    return out


def _make_door_patch(direction: str, color: int) -> np.ndarray:
    patch = np.full((3, 3), Color.GREY, dtype=int)
    gap_r, gap_c = _EDGE_MID[_OPPOSITE[direction]]
    patch[gap_r, gap_c] = Color.BLACK
    mr, mc = _EDGE_MID[direction]
    patch[mr, mc] = int(color)
    return patch


def _make_normal_patch(direction: str, color: int, rng: np.random.Generator) -> np.ndarray:
    patch = np.full((3, 3), Color.GREY, dtype=int)
    pattern = _MARKER_PATTERNS[direction][int(rng.integers(0, len(_MARKER_PATTERNS[direction])))]
    if len(pattern) == 2 and rng.random() < 0.45:
        pattern = [pattern[0]]
    for r, c in pattern:
        patch[r, c] = int(color)
    return patch


def _choose_slots(
    candidates: list[tuple[int, int]],
    existing: list[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    shuffled = candidates.copy()
    rng.shuffle(shuffled)
    existing_boxes = [_top_left_box(pos) for pos in existing]
    for pos in shuffled:
        box = _top_left_box(pos)
        if all(not _boxes_too_close(box, other, pad=1) for other in existing_boxes):
            return pos
    return None


def _edge_candidates(height: int, width: int, direction: str) -> list[tuple[int, int]]:
    if direction == "up":
        return [(0, c) for c in range(width - 2)]
    if direction == "down":
        return [(height - 3, c) for c in range(width - 2)]
    if direction == "left":
        return [(r, 0) for r in range(height - 2)]
    return [(r, width - 3) for r in range(height - 2)]


def _initial_candidates(
    height: int,
    width: int,
    direction: str,
    final_pos: tuple[int, int],
) -> list[tuple[int, int]]:
    fr, fc = final_pos
    if direction in {"up", "down"}:
        return [(r, fc) for r in range(1, height - 3) if r != fr]
    return [(fr, c) for c in range(1, width - 3) if c != fc]


def _assign_initial_positions(
    moved_specs: list[dict[str, object]],
    blank_pos: tuple[int, int],
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    candidates = []
    for idx, spec in enumerate(moved_specs):
        choices = _initial_candidates(spec["height"], spec["width"], spec["direction"], spec["final_pos"])
        rng.shuffle(choices)
        candidates.append((idx, choices))
    candidates.sort(key=lambda item: len(item[1]))

    chosen: dict[int, tuple[int, int]] = {}
    blank_box = _top_left_box(blank_pos)

    def rec(pos: int) -> bool:
        if pos == len(candidates):
            return True
        idx, choices = candidates[pos]
        for top_left in choices:
            box = _top_left_box(top_left)
            if _boxes_too_close(box, blank_box, pad=1):
                continue
            ok = True
            for other in chosen.values():
                if _boxes_too_close(box, _top_left_box(other), pad=1):
                    ok = False
                    break
            if not ok:
                continue
            chosen[idx] = top_left
            if rec(pos + 1):
                return True
            del chosen[idx]
        return False

    if not rec(0):
        return None
    return [chosen[i] for i in range(len(moved_specs))]


def _blank_candidates(
    height: int,
    width: int,
    moved_final: list[tuple[int, int]],
    endpoints: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    moved_boxes = [_top_left_box(pos) for pos in moved_final]
    r0 = min(p[0] for p in endpoints)
    r1 = max(p[0] for p in endpoints)
    c0 = min(p[1] for p in endpoints)
    c1 = max(p[1] for p in endpoints)
    inside = []
    fallback = []
    for r in range(1, height - 3):
        for c in range(1, width - 3):
            box = _top_left_box((r, c))
            if any(_boxes_too_close(box, other, pad=1) for other in moved_boxes):
                continue
            if r <= r1 and r + 2 >= r0 and c <= c1 and c + 2 >= c0:
                inside.append((r, c))
            fallback.append((r, c))
    return inside or fallback


def _build_input(
    height: int,
    width: int,
    blank_pos: tuple[int, int],
    blank_patch: np.ndarray,
    moved_specs: list[dict[str, object]],
    initial_positions: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((height, width), Color.BLACK, dtype=int)
    paste(grid, blank_patch, blank_pos[0], blank_pos[1])
    for spec, top_left in zip(moved_specs, initial_positions, strict=True):
        paste(grid, spec["patch"], top_left[0], top_left[1])
    return grid


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(15, 19))
    width = int(rng.integers(15, 19))
    ordinary_count = int(rng.integers(2, 5))

    door_directions = [("up", "left"), ("up", "right"), ("down", "left"), ("down", "right")]
    door_dir_a, door_dir_b = door_directions[int(rng.integers(0, len(door_directions)))]
    moved_dirs = [door_dir_a, door_dir_b] + [str(rng.choice(list(_DIRS.keys()))) for _ in range(ordinary_count)]

    moved_final: list[tuple[int, int]] = []
    moved_specs: list[dict[str, object]] = []

    for idx, direction in enumerate(moved_dirs):
        pos = _choose_slots(_edge_candidates(height, width, direction), moved_final, rng)
        if pos is None:
            return None
        moved_final.append(pos)
        color = int(rng.choice(_MARKER_COLORS))
        if idx < 2:
            patch = _make_door_patch(direction, color)
            kind = "door"
        else:
            patch = _make_normal_patch(direction, color, rng)
            kind = "normal"
        moved_specs.append(
            {
                "kind": kind,
                "direction": direction,
                "patch": patch,
                "final_pos": pos,
                "height": height,
                "width": width,
            }
        )

    endpoints = []
    for spec in moved_specs[:2]:
        fr, fc = spec["final_pos"]
        gap_r, gap_c = _EDGE_MID[_OPPOSITE[spec["direction"]]]
        dr, dc = _DIRS[_OPPOSITE[spec["direction"]]]
        endpoints.append((fr + gap_r + dr, fc + gap_c + dc))
    if endpoints[0][0] == endpoints[1][0] or endpoints[0][1] == endpoints[1][1]:
        return None

    blank_choices = _blank_candidates(height, width, moved_final, endpoints)
    rng.shuffle(blank_choices)
    blank_patch = np.full((3, 3), Color.GREY, dtype=int)

    for blank_pos in blank_choices[:40]:
        initial_positions = _assign_initial_positions(moved_specs, blank_pos, rng)
        if initial_positions is None:
            continue
        inp = _build_input(height, width, blank_pos, blank_patch, moved_specs, initial_positions)
        try:
            moved, gap_cells, path = _solve_layout(inp)
        except Exception:
            continue
        yellow_count = int(path.sum()) + len(gap_cells)
        if yellow_count < 8 or yellow_count > max(height, width) * 3:
            continue
        out = main(inp)
        if out.shape != inp.shape:
            continue
        if np.count_nonzero(inp != out) < 8:
            continue
        if not np.any(moved == Color.GREY):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_RETRY_LIMIT):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid direction-maze input")
