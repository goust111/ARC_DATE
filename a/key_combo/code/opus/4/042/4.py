from __future__ import annotations

import random
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np


_ROOT = Path(__file__).resolve().parents[6]
_ARC_UTILS_DIR = _ROOT / "new" / "solvers"
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import find_components, get_bg
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        counts = Counter(map(int, grid.ravel()))
        return int(counts.most_common(1)[0][0])


    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors: list[int] | None = None,
        return_format: str = "dict",
    ) -> list[dict]:
        assert return_format == "dict"
        assert monochromatic
        if bg is None:
            bg = get_bg(grid)
        h, w = grid.shape
        seen = np.zeros((h, w), dtype=bool)
        out: list[dict] = []
        palette = set(map(int, np.unique(grid))) - {int(bg)}
        if colors is not None:
            palette &= set(map(int, colors))
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)] if connectivity == 4 else [
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
        ]
        for r in range(h):
            for c in range(w):
                color = int(grid[r, c])
                if seen[r, c] or color not in palette:
                    continue
                seen[r, c] = True
                q: deque[tuple[int, int]] = deque([(r, c)])
                coords: list[tuple[int, int]] = []
                while q:
                    rr, cc = q.popleft()
                    coords.append((rr, cc))
                    for dr, dc in deltas:
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                arr = np.array(coords, dtype=int)
                r0, c0 = arr.min(axis=0)
                r1, c1 = arr.max(axis=0)
                out.append(
                    {
                        "color": color,
                        "coords": arr,
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "area": int(arr.shape[0]),
                    }
                )
        return out


BLOCK = 1
BG = 0
CORNER_ORDER = ("NW", "NE", "SE", "SW")
CORNER_OFFSETS = {
    "NW": (-1, -1),
    "NE": (-1, 2),
    "SE": (2, 2),
    "SW": (2, -1),
}
ROTATED_SOURCE = {
    "NW": "SW",
    "NE": "NW",
    "SE": "NE",
    "SW": "SE",
}
BRIDGE_CORNER = {"R": "NE", "L": "SW", "D": "SE", "U": "NW"}

PALETTE = (2, 3, 4, 5, 6, 7, 8)
LATTICE_STEP = 6
GEN_TRIES = 500
TURN_PATTERNS = (
    ("R", "U"),
    ("R", "D"),
    ("L", "U"),
    ("L", "D"),
    ("U", "L"),
    ("U", "R"),
    ("D", "L"),
    ("D", "R"),
)


def _rotate_corners(corners: dict[str, int]) -> dict[str, int]:
    return {corner: int(corners[ROTATED_SOURCE[corner]]) for corner in CORNER_ORDER}


def _parse_blocks(grid: np.ndarray, bg: int) -> list[dict]:
    counts = Counter(map(int, grid.ravel()))
    components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[BLOCK],
        return_format="dict",
    )
    blocks: list[dict] = []
    h, w = grid.shape
    for comp in components:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if int(comp["area"]) != 4 or (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
            continue
        corners: dict[str, int] = {}
        for name in CORNER_ORDER:
            dr, dc = CORNER_OFFSETS[name]
            rr = r0 + dr
            cc = c0 + dc
            assert 0 <= rr < h and 0 <= cc < w, "all corner ports must be in bounds"
            color = int(grid[rr, cc])
            assert color not in (bg, BLOCK), "each block needs four colored diagonal ports"
            corners[name] = color
        assert len(set(corners.values())) == 4, "each block must use four distinct corner colors"
        freqs = {color: int(counts[color]) for color in corners.values()}
        low = min(freqs.values())
        identities = [color for color in corners.values() if freqs[color] == low]
        assert len(identities) == 1, "every block must have a unique identity color"
        blocks.append(
            {
                "top_left": (r0, c0),
                "corners": corners,
                "rotated": _rotate_corners(corners),
                "identity": int(identities[0]),
            }
        )
    assert 3 <= len(blocks) <= 4, "expected 3-4 blue 2x2 blocks"
    ids = [block["identity"] for block in blocks]
    assert len(ids) == len(set(ids)), "identity colors must uniquely identify blocks"
    return sorted(blocks, key=lambda block: block["top_left"])


def _control_sequence(grid: np.ndarray, bg: int) -> list[int]:
    rows = [r for r in range(grid.shape[0]) if np.any(grid[r] != bg)]
    assert rows, "expected a non-empty grid"
    ctrl_r = max(rows)
    row = [int(v) for v in grid[ctrl_r]]
    assert row.count(BLOCK) >= 4, "control row must contain blue separators"
    seq = [color for color in row if color not in (bg, BLOCK)]
    assert len(seq) == 3, "generator uses a three-step control sequence"
    return seq


def _bridge_direction(src: tuple[int, int], dst: tuple[int, int]) -> str:
    dr = int(dst[0] - src[0])
    dc = int(dst[1] - src[1])
    if abs(dc) >= abs(dr):
        return "R" if dc > 0 else "L"
    return "D" if dr > 0 else "U"


def _bridge_cells(src: tuple[int, int], dst: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = src
    tr, tc = dst
    direction = _bridge_direction(src, dst)
    cells: list[tuple[int, int]] = []
    if direction == "R":
        for c in range(sc + 2, tc):
            cells.append((sr, c))
    elif direction == "L":
        for c in range(tc + 2, sc):
            cells.append((sr, c))
    elif direction == "D":
        for r in range(sr + 2, tr):
            cells.append((r, sc))
    else:
        for r in range(tr + 2, sr):
            cells.append((r, sc))
    assert cells, "each bridge must span at least one background cell"
    return cells


def _solve(
    grid: np.ndarray,
    *,
    rotate_corners: bool = True,
    bridge_uses_rotated: bool = True,
    sequence_override: list[int] | None = None,
    forced_corner: str | None = None,
) -> tuple[np.ndarray, list[dict], list[int], dict[int, dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    assert bg == BG, "generated tasks use black background"
    blocks = _parse_blocks(g, bg)
    by_identity = {int(block["identity"]): block for block in blocks}
    sequence = list(_control_sequence(g, bg) if sequence_override is None else sequence_override)
    assert len(sequence) == len(set(sequence)), "control sequence should visit distinct blocks"
    assert set(sequence).issubset(by_identity), "control sequence must reference known blocks"

    out = g.copy()
    if rotate_corners:
        for block in blocks:
            r0, c0 = block["top_left"]
            for name, color in block["rotated"].items():
                dr, dc = CORNER_OFFSETS[name]
                out[r0 + dr, c0 + dc] = int(color)

    bridge_meta: list[dict] = []
    for src_id, dst_id in zip(sequence, sequence[1:]):
        src = by_identity[int(src_id)]
        dst = by_identity[int(dst_id)]
        direction = _bridge_direction(src["top_left"], dst["top_left"])
        corner = forced_corner or BRIDGE_CORNER[direction]
        corner_map = src["rotated"] if bridge_uses_rotated else src["corners"]
        color = int(corner_map[corner])
        cells = _bridge_cells(src["top_left"], dst["top_left"])
        for r, c in cells:
            out[r, c] = color
        bridge_meta.append(
            {
                "src": int(src_id),
                "dst": int(dst_id),
                "direction": direction,
                "color": color,
                "corner": corner,
                "cells": cells,
            }
        )

    return out, bridge_meta, sequence, by_identity


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each 2x2 block's diagonal corner colors, then connect control-selected blocks with direction-dependent bridges.

    Input structure:
    - The background is black, with 3-4 solid blue 2x2 blocks in the main area.
    - Each block has four distinct colored pixels at its diagonal outside corners, one per NW/NE/SE/SW port.
    - The bottom row is a control bar made of blue separator pixels with three identity colors between them.

    Rule:
    - For every 2x2 block, rotate the four diagonal corner colors clockwise by one step while keeping the blue block unchanged.
    - Read the bottom control colors left-to-right, ignoring blue separators, to get the visitation order of blocks identified by their uniquely least frequent corner color.
    - For each consecutive source-target pair, choose the dominant axis from the block-center displacement, use the source-side corner facing that axis, take its rotated color, and draw a one-pixel bridge from the source edge to the target edge along that axis without overwriting block cells.

    Key insight:
    - The control bar only chooses which blocks connect; the actual bridge color is hidden in the source block's post-rotation corner that faces the target direction.
    """
    return _solve(grid)[0]


def _top_left_candidates(limit: int, offset: int) -> list[int]:
    vals: list[int] = []
    cur = offset
    while cur <= limit:
        vals.append(cur)
        cur += LATTICE_STEP
    return vals


def _move_index(row_i: int, col_i: int, direction: str, steps: int) -> tuple[int, int]:
    if direction == "R":
        return row_i, col_i + steps
    if direction == "L":
        return row_i, col_i - steps
    if direction == "D":
        return row_i + steps, col_i
    return row_i - steps, col_i


def _expanded_box(top_left: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = top_left
    return (r - 1, c - 1, r + 2, c + 2)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _choose_path(rows: list[int], cols: list[int]) -> tuple[list[tuple[int, int]], tuple[str, str]] | None:
    for _ in range(200):
        d1, d2 = random.choice(TURN_PATTERNS)
        r0_i = random.randrange(len(rows))
        c0_i = random.randrange(len(cols))
        max1 = {
            "R": len(cols) - 1 - c0_i,
            "L": c0_i,
            "D": len(rows) - 1 - r0_i,
            "U": r0_i,
        }[d1]
        if max1 < 1:
            continue
        step1 = random.randint(1, max1)
        r1_i, c1_i = _move_index(r0_i, c0_i, d1, step1)
        max2 = {
            "R": len(cols) - 1 - c1_i,
            "L": c1_i,
            "D": len(rows) - 1 - r1_i,
            "U": r1_i,
        }[d2]
        if max2 < 1:
            continue
        step2 = random.randint(1, max2)
        r2_i, c2_i = _move_index(r1_i, c1_i, d2, step2)
        coords = [(rows[r0_i], cols[c0_i]), (rows[r1_i], cols[c1_i]), (rows[r2_i], cols[c2_i])]
        if len(set(coords)) != 3:
            continue
        return coords, (d1, d2)
    return None


def _control_row(grid: np.ndarray, sequence: list[int]) -> None:
    width = grid.shape[1]
    pattern_len = 2 * len(sequence) + 1
    start = random.randint(0, width - pattern_len)
    row = grid.shape[0] - 1
    for idx, color in enumerate(sequence):
        grid[row, start + 2 * idx] = BLOCK
        grid[row, start + 2 * idx + 1] = int(color)
    grid[row, start + pattern_len - 1] = BLOCK


def _place_block(grid: np.ndarray, top_left: tuple[int, int], corners: dict[str, int]) -> None:
    r0, c0 = top_left
    grid[r0 : r0 + 2, c0 : c0 + 2] = BLOCK
    for name, color in corners.items():
        dr, dc = CORNER_OFFSETS[name]
        grid[r0 + dr, c0 + dc] = int(color)


def _alternate_sequences(by_identity: dict[int, dict], sequence: list[int]) -> list[list[int]]:
    pos = {identity: block["top_left"] for identity, block in by_identity.items()}
    visited = list(sequence)
    row_major = [identity for identity, _ in sorted(((v, pos[v]) for v in visited), key=lambda item: item[1])]
    col_major = [identity for identity, _ in sorted(((v, pos[v]) for v in visited), key=lambda item: (item[1][1], item[1][0]))]
    options = [row_major, col_major, list(reversed(sequence))]
    out: list[list[int]] = []
    for option in options:
        if option != sequence and option not in out:
            out.append(option)
    return out


def _generator_checks(grid: np.ndarray) -> bool:
    try:
        out, bridges, sequence, by_identity = _solve(grid)
    except AssertionError:
        return False

    if len(by_identity) != 4 or len(sequence) != 3 or len(bridges) != 2:
        return False
    if len({bridge["direction"] for bridge in bridges}) != 2:
        return False
    if len({bridge["color"] for bridge in bridges}) != 2:
        return False

    no_rot_bridge = _solve(grid, rotate_corners=True, bridge_uses_rotated=False)[0]
    if np.array_equal(out, no_rot_bridge):
        return False

    fixed_corner = _solve(grid, forced_corner="NE")[0]
    if np.array_equal(out, fixed_corner):
        return False

    for alt in _alternate_sequences(by_identity, sequence):
        if np.array_equal(out, _solve(grid, sequence_override=alt)[0]):
            return False

    counts = Counter(map(int, grid.ravel()))
    for identity, block in by_identity.items():
        if int(counts[identity]) not in (1, 2):
            return False
        others = [color for color in block["corners"].values() if color != identity]
        if not all(int(counts[color]) >= 4 for color in others):
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(20, 26)
    width = random.randint(20, 26)
    row_offset = random.randint(1, 3)
    col_offset = random.randint(1, 3)
    rows = _top_left_candidates(height - 4, row_offset)
    cols = _top_left_candidates(width - 4, col_offset)
    if len(rows) < 3 or len(cols) < 3:
        return None

    path = _choose_path(rows, cols)
    if path is None:
        return None
    visited_positions, dirs = path
    visited_set = set(visited_positions)

    bridge_cells = set()
    for src, dst in zip(visited_positions, visited_positions[1:]):
        bridge_cells.update(_bridge_cells(src, dst))

    candidates = [(r, c) for r in rows for c in cols if (r, c) not in visited_set]
    random.shuffle(candidates)
    distractor = None
    visited_boxes = [_expanded_box(pos) for pos in visited_positions]
    for pos in candidates:
        box = _expanded_box(pos)
        if any(_boxes_overlap(box, other) for other in visited_boxes):
            continue
        if any((rr, cc) in bridge_cells for rr in range(box[0], box[2] + 1) for cc in range(box[1], box[3] + 1)):
            continue
        distractor = pos
        break
    if distractor is None:
        return None

    common = random.sample(list(PALETTE), 3)
    identity_pool = [color for color in PALETTE if color not in common]
    identities = random.sample(identity_pool, 4)

    blocks = []
    for idx, pos in enumerate(visited_positions + [distractor]):
        colors = [identities[idx], *common]
        random.shuffle(colors)
        corners = {name: color for name, color in zip(CORNER_ORDER, colors)}
        blocks.append({"top_left": pos, "identity": identities[idx], "corners": corners})

    sequence = [int(block["identity"]) for block in blocks[:3]]
    if dirs[0] == dirs[1]:
        return None

    grid = np.zeros((height, width), dtype=int)
    for block in blocks:
        _place_block(grid, block["top_left"], block["corners"])
    _control_row(grid, sequence)

    return grid if _generator_checks(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_TRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
