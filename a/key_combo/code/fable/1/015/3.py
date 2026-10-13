from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    arc_utils_dir = str(_ARC_UTILS_DIR)
    if arc_utils_dir not in sys.path:
        sys.path.append(arc_utils_dir)

try:
    from arc_utils import get_bg, paste
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        values, counts = np.unique(np.asarray(grid), return_counts=True)
        return int(values[np.argmax(counts)])

    def paste(
        dst: np.ndarray,
        patch: np.ndarray,
        r0: int,
        c0: int,
        *,
        transparent: int | None = None,
        clip: bool = False,
    ) -> np.ndarray:
        rr0 = int(r0)
        cc0 = int(c0)
        h, w = map(int, patch.shape)
        H, W = dst.shape
        if clip:
            dr0 = max(0, rr0)
            dc0 = max(0, cc0)
            dr1 = min(H, rr0 + h)
            dc1 = min(W, cc0 + w)
            if dr0 >= dr1 or dc0 >= dc1:
                return dst
            pr0 = dr0 - rr0
            pc0 = dc0 - cc0
            pr1 = pr0 + (dr1 - dr0)
            pc1 = pc0 + (dc1 - dc0)
            view = patch[pr0:pr1, pc0:pc1]
            if transparent is None:
                dst[dr0:dr1, dc0:dc1] = view
            else:
                mask = view != int(transparent)
                dst_view = dst[dr0:dr1, dc0:dc1]
                dst_view[mask] = view[mask]
            return dst
        if rr0 < 0 or cc0 < 0 or rr0 + h > H or cc0 + w > W:
            raise ValueError("paste out of bounds")
        if transparent is None:
            dst[rr0 : rr0 + h, cc0 : cc0 + w] = patch
        else:
            view = dst[rr0 : rr0 + h, cc0 : cc0 + w]
            mask = patch != int(transparent)
            view[mask] = patch[mask]
        return dst


_PALETTE = tuple(range(10))
_SIDES = ("north", "south", "west", "east")
_GEN_ATTEMPTS = 400


def _panel_layout(panel_w: int) -> tuple[list[int], list[int]]:
    starts = [0, panel_w + 1, 2 * panel_w + 2, 3 * panel_w + 3]
    separators = [panel_w, 2 * panel_w + 1, 3 * panel_w + 2]
    return starts, separators


def _extract_panels(grid: np.ndarray, bg: int) -> tuple[int, int, list[int], list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    candidates: list[tuple[int, int, list[int]]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        cols = [c for c in range(W) if bool(np.all(g[:, c] == color))]
        if len(cols) != 3:
            continue
        step1 = cols[1] - cols[0]
        step2 = cols[2] - cols[1]
        if step1 != step2 or step1 <= 1:
            continue
        panel_w = step1 - 1
        starts, separators = _panel_layout(panel_w)
        if separators != cols:
            continue
        if W != 4 * panel_w + 3:
            continue
        candidates.append((color, panel_w, starts))
    assert len(candidates) == 1, "expected one separator-color panel layout"
    divider, panel_w, starts = candidates[0]
    panels = [g[:, c0 : c0 + panel_w].copy() for c0 in starts]
    assert all(panel.shape == (H, panel_w) for panel in panels)
    return int(divider), int(panel_w), starts, panels


def _rotate_coords(coords: np.ndarray, pivot: tuple[int, int], clockwise: bool) -> np.ndarray:
    pr, pc = map(int, pivot)
    dr = coords[:, 0] - pr
    dc = coords[:, 1] - pc
    if clockwise:
        rr = pr + dc
        cc = pc - dr
    else:
        rr = pr - dc
        cc = pc + dr
    return np.stack([rr, cc], axis=1).astype(int)


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, coord)) for coord in np.asarray(coords, dtype=int)}


def _infer_color_mapping(
    src: np.ndarray,
    dst: np.ndarray,
    src_coords: np.ndarray,
    pivot: tuple[int, int],
    clockwise: bool,
    bg: int,
) -> dict[int, int] | None:
    rotated = _rotate_coords(src_coords, pivot, clockwise)
    H, W = dst.shape
    if np.any(rotated[:, 0] < 0) or np.any(rotated[:, 0] >= H):
        return None
    if np.any(rotated[:, 1] < 0) or np.any(rotated[:, 1] >= W):
        return None

    mapping: dict[int, int] = {}
    reverse: dict[int, int] = {}
    for (r, c), (rr, cc) in zip(src_coords, rotated):
        src_color = int(src[int(r), int(c)])
        dst_color = int(dst[int(rr), int(cc)])
        if dst_color == bg:
            return None
        prev = mapping.setdefault(src_color, dst_color)
        if prev != dst_color:
            return None
        back = reverse.setdefault(dst_color, src_color)
        if back != src_color:
            return None
    return mapping


def _advance_panel(
    panel: np.ndarray,
    pivot: tuple[int, int],
    clockwise: bool,
    mapping: dict[int, int],
    bg: int,
) -> np.ndarray:
    coords = np.argwhere(panel != bg)
    rotated = _rotate_coords(coords, pivot, clockwise)
    H, W = panel.shape
    assert not np.any(rotated[:, 0] < 0) and not np.any(rotated[:, 0] >= H)
    assert not np.any(rotated[:, 1] < 0) and not np.any(rotated[:, 1] >= W)

    out = np.full_like(panel, int(bg))
    for (r, c), (rr, cc) in zip(coords, rotated):
        out[int(rr), int(cc)] = int(mapping[int(panel[int(r), int(c)])])
    return out


def _solve_sequence(panels: list[np.ndarray], bg: int) -> tuple[tuple[int, int], bool, dict[int, int]]:
    frames = [np.asarray(panel, dtype=int) for panel in panels]
    H, W = frames[0].shape
    coords = [np.argwhere(frame != bg) for frame in frames]
    assert all(len(cc) > 0 for cc in coords), "each observed panel must contain a sprite"
    assert len({len(cc) for cc in coords}) == 1, "rigid rotation must preserve occupied area"
    color_sets = [{int(v) for v in np.unique(frame) if int(v) != bg} for frame in frames]

    candidates: dict[tuple[tuple[int, int], bool, tuple[tuple[int, int], ...]], tuple[tuple[int, int], bool, dict[int, int]]] = {}
    target_sets = [_coord_set(cc) for cc in coords]

    for clockwise in (True, False):
        for pr in range(H):
            for pc in range(W):
                if any(int(frame[pr, pc]) != bg for frame in frames):
                    continue
                pivot = (pr, pc)
                step_maps: list[dict[int, int]] = []
                valid = True
                for idx in range(2):
                    rotated = _rotate_coords(coords[idx], pivot, clockwise)
                    if _coord_set(rotated) != target_sets[idx + 1]:
                        valid = False
                        break
                    mapping = _infer_color_mapping(frames[idx], frames[idx + 1], coords[idx], pivot, clockwise, bg)
                    if mapping is None:
                        valid = False
                        break
                    if set(mapping.keys()) != color_sets[idx]:
                        valid = False
                        break
                    if set(mapping.values()) != color_sets[idx + 1]:
                        valid = False
                        break
                    step_maps.append(mapping)
                if not valid or step_maps[0] != step_maps[1]:
                    continue

                try:
                    _advance_panel(frames[2], pivot, clockwise, step_maps[0], bg)
                except AssertionError:
                    continue
                key = (pivot, clockwise, tuple(sorted(step_maps[0].items())))
                candidates[key] = (pivot, clockwise, step_maps[0])

    assert len(candidates) == 1, "expected a unique hidden pivot, rotation direction, and color cycle"
    return next(iter(candidates.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Predict the missing fourth panel by extending a hidden-pivot rotation-and-recolor sequence.

    Input structure:
    - A one-cell-wide separator color splits the grid into four equal panels in one row.
    - Panels 1-3 each show one frame of the same multicolor sprite on a shared background, while panel 4 is blank.
    - From frame to frame the sprite rotates 90 degrees around one never-drawn pivot cell inside the panel, so its position drifts.

    Rule:
    - Find the unique rotation direction, pivot cell, and color permutation that make P1->P2 and P2->P3 the same rigid step.
    - Apply that same 90-degree rotation and the same one-step color permutation to panel 3.
    - Leave the divider columns and the first three panels unchanged, and draw the predicted frame into panel 4.

    Key insight:
    - The sprite is not rotating around itself: the changing pose and the panel-to-panel drift come from the same invisible external pivot, and the colors advance by a fixed cycle each step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    bg = int(get_bg(g))
    _divider, panel_w, starts, panels = _extract_panels(g, bg)
    assert np.all(panels[3] == bg), "the fourth panel must be blank in the input"

    pivot, clockwise, mapping = _solve_sequence(panels[:3], bg)
    predicted = _advance_panel(panels[2], pivot, clockwise, mapping, bg)

    out = g.copy()
    out[:, starts[3] : starts[3] + panel_w] = predicted
    return out


def _neighbors4(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _allowed_relative_cells(radius: int, side: str) -> set[tuple[int, int]]:
    allowed: set[tuple[int, int]] = set()
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr == 0 and dc == 0:
                continue
            if side == "north" and dr < 0:
                allowed.add((dr, dc))
            elif side == "south" and dr > 0:
                allowed.add((dr, dc))
            elif side == "west" and dc < 0:
                allowed.add((dr, dc))
            elif side == "east" and dc > 0:
                allowed.add((dr, dc))
    return allowed


def _random_connected_shape(radius: int, side: str, area: int) -> set[tuple[int, int]] | None:
    allowed = _allowed_relative_cells(radius, side)
    if len(allowed) < area:
        return None

    for _ in range(80):
        start = random.choice(list(allowed))
        cells = {start}
        while len(cells) < area:
            frontier = {
                nb
                for cell in cells
                for nb in _neighbors4(cell)
                if nb in allowed and nb not in cells
            }
            if not frontier:
                break
            cells.add(random.choice(list(frontier)))
        if len(cells) != area:
            continue
        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
        if abs(sum(rows)) + abs(sum(cols)) < area:
            continue
        return cells
    return None


def _relative_to_absolute(
    rel_cells: set[tuple[int, int]],
    pivot: tuple[int, int],
    coloring: dict[tuple[int, int], int],
) -> dict[tuple[int, int], int]:
    pr, pc = pivot
    return {
        (pr + dr, pc + dc): int(coloring[(dr, dc)])
        for dr, dc in rel_cells
    }


def _advance_cells(
    cells: dict[tuple[int, int], int],
    pivot: tuple[int, int],
    clockwise: bool,
    mapping: dict[int, int],
) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    pr, pc = pivot
    for (r, c), color in cells.items():
        dr = r - pr
        dc = c - pc
        if clockwise:
            rr = pr + dc
            cc = pc - dr
        else:
            rr = pr - dc
            cc = pc + dr
        out[(int(rr), int(cc))] = int(mapping[int(color)])
    return out


def _render_panel(panel_shape: tuple[int, int], cells: dict[tuple[int, int], int], bg: int) -> np.ndarray:
    panel = np.full(panel_shape, int(bg), dtype=int)
    for (r, c), color in cells.items():
        panel[int(r), int(c)] = int(color)
    return panel


def _build_input_and_output(
    panel_shape: tuple[int, int],
    divider: int,
    bg: int,
    frames: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    panel_h, panel_w = panel_shape
    starts, separators = _panel_layout(panel_w)
    width = 4 * panel_w + 3

    inp = np.full((panel_h, width), int(bg), dtype=int)
    out = np.full((panel_h, width), int(bg), dtype=int)
    for col in separators:
        inp[:, int(col)] = int(divider)
        out[:, int(col)] = int(divider)

    for idx in range(3):
        paste(inp, frames[idx], 0, starts[idx])
        paste(out, frames[idx], 0, starts[idx])
    paste(out, frames[3], 0, starts[3])
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(6, 9)
    panel_w = random.choice((5, 6))
    bg = random.choice(_PALETTE)
    remaining = [color for color in _PALETTE if color != bg]
    divider = random.choice(remaining)
    remaining = [color for color in remaining if color != divider]

    pivot = (random.randint(2, panel_h - 3), random.randint(2, panel_w - 3))
    radius = min(pivot[0], panel_h - 1 - pivot[0], pivot[1], panel_w - 1 - pivot[1])
    if radius < 2:
        return None

    side = random.choice(_SIDES)
    max_cycle = min(4, len(remaining))
    cycle_len = random.choice([k for k in (2, 3, 4) if k <= max_cycle])
    cycle_colors = random.sample(remaining, cycle_len)

    area = random.randint(cycle_len, 7)
    rel_cells = _random_connected_shape(radius, side, area)
    if rel_cells is None:
        return None

    assignments = cycle_colors[:]
    assignments.extend(random.choice(cycle_colors) for _ in range(area - cycle_len))
    random.shuffle(assignments)
    rel_list = list(rel_cells)
    random.shuffle(rel_list)
    coloring = {cell: int(color) for cell, color in zip(rel_list, assignments)}
    color_map = {
        int(cycle_colors[idx]): int(cycle_colors[(idx + 1) % cycle_len])
        for idx in range(cycle_len)
    }
    clockwise = random.choice((True, False))

    frame_cells = [_relative_to_absolute(rel_cells, pivot, coloring)]
    for _ in range(3):
        frame_cells.append(_advance_cells(frame_cells[-1], pivot, clockwise, color_map))
    if any(pivot in cells for cells in frame_cells):
        return None

    frames = [_render_panel((panel_h, panel_w), cells, bg) for cells in frame_cells]
    inp, expected = _build_input_and_output((panel_h, panel_w), divider, bg, frames)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-pivot panel sequence")
