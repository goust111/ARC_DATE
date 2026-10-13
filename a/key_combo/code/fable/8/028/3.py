from __future__ import annotations

import importlib.util
import itertools
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("_arc_utils_fable_8_028_3", _ARC_UTILS_PATH)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
is_contiguous = _ARC_UTILS.is_contiguous


DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

FRAME_ORDER = ("right", "bottom", "left", "top")
PALETTE = tuple(range(10))

FRAME_H_RANGE = (4, 6)
FRAME_W_RANGE = (4, 6)
CORRIDOR_MIN = 1
CORRIDOR_MAX = 3
PAD_MIN = 1
PAD_MAX = 2
HCOUNT_CHOICES = (1, 2)
VCOUNT_CHOICES = (1, 2)
GEN_ATTEMPTS = 500


@dataclass(frozen=True)
class ParsedFrame:
    bbox: tuple[int, int, int, int]
    coords: np.ndarray
    gap: tuple[int, int]
    side: str
    interior_color: int
    interior_mask: np.ndarray


@dataclass(frozen=True)
class FrameSpec:
    bbox: tuple[int, int, int, int]
    side: str
    gap_local: tuple[int, int]
    interior_color: int
    interior_mask: np.ndarray


def _border_mask(height: int, width: int) -> np.ndarray:
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _side_from_gap(local_r: int, local_c: int, height: int, width: int) -> str | None:
    on_top = local_r == 0
    on_bottom = local_r == height - 1
    on_left = local_c == 0
    on_right = local_c == width - 1
    if (on_top or on_bottom) and (on_left or on_right):
        return None
    if on_top:
        return "top"
    if on_bottom:
        return "bottom"
    if on_left:
        return "left"
    if on_right:
        return "right"
    return None


def _parse_frame_component(grid: np.ndarray, obj: dict, bg: int) -> ParsedFrame | None:
    color = int(obj["color"])
    bbox = tuple(map(int, obj["bbox"]))
    r0, c0, r1, c1 = bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    patch = crop_bbox(grid, bbox)
    border = _border_mask(height, width)
    frame_mask = patch == color
    if np.any(frame_mask & ~border):
        return None

    missing = np.argwhere(border & ~frame_mask)
    if missing.shape[0] != 1:
        return None
    gap_r, gap_c = map(int, missing[0])
    side = _side_from_gap(gap_r, gap_c, height, width)
    if side is None:
        return None
    if int(patch[gap_r, gap_c]) != int(bg):
        return None
    if int(obj["area"]) != int(border.sum()) - 1:
        return None

    interior = patch[1:-1, 1:-1]
    interior_colors = set(map(int, np.unique(interior))) - {int(bg), color}
    if len(interior_colors) != 1:
        return None
    interior_color = int(next(iter(interior_colors)))
    interior_mask = interior == interior_color
    if not interior_mask.any():
        return None
    return ParsedFrame(
        bbox=bbox,
        coords=np.asarray(obj["coords"], dtype=int).copy(),
        gap=(r0 + gap_r, c0 + gap_c),
        side=side,
        interior_color=interior_color,
        interior_mask=interior_mask,
    )


def _extract_frames(grid: np.ndarray) -> tuple[int, int, list[ParsedFrame]]:
    bg = int(get_bg(grid))
    colors = sorted(set(map(int, grid.ravel())) - {bg})
    candidates: list[tuple[int, list[ParsedFrame]]] = []

    for color in colors:
        comps = find_components(
            grid,
            bg=bg,
            connectivity=4,
            monochromatic=True,
            colors=[color],
            return_format="dict",
        )
        if len(comps) < 3:
            continue
        parsed: list[ParsedFrame] = []
        for comp in comps:
            frame = _parse_frame_component(grid, comp, bg)
            if frame is None:
                parsed = []
                break
            parsed.append(frame)
        if parsed:
            candidates.append((color, parsed))

    assert len(candidates) == 1, "expected a unique frame color"
    frame_color, frames = candidates[0]
    interior_colors = [frame.interior_color for frame in frames]
    assert len(set(interior_colors)) == len(interior_colors), "expected distinct interior colors"
    return bg, int(frame_color), frames


def _trace_successors(grid: np.ndarray, frame_color: int, frames: list[ParsedFrame]) -> list[int]:
    height, width = map(int, grid.shape)
    cell_to_frame: dict[tuple[int, int], int] = {}
    for idx, frame in enumerate(frames):
        for r, c in frame.coords:
            cell_to_frame[(int(r), int(c))] = idx

    successors: list[int] = []
    for idx, frame in enumerate(frames):
        dr, dc = DIRS[frame.side]
        r = int(frame.gap[0] + dr)
        c = int(frame.gap[1] + dc)
        hit = None
        while 0 <= r < height and 0 <= c < width:
            hit = cell_to_frame.get((r, c))
            if hit is not None:
                break
            r += dr
            c += dc
        assert hit is not None, "each gap ray must hit a successor frame"
        assert hit != idx, "a frame may not point to itself"
        successors.append(int(hit))

    indegrees = Counter(successors)
    assert all(indegrees[i] == 1 for i in range(len(frames))), "expected one predecessor per frame"

    seen: set[int] = set()
    cur = 0
    while cur not in seen:
        seen.add(cur)
        cur = successors[cur]
    assert cur == 0 and len(seen) == len(frames), "successor graph must be one directed cycle"
    return successors


def _parse_scene(grid: np.ndarray) -> tuple[int, int, list[ParsedFrame], list[int], list[int]]:
    bg, frame_color, frames = _extract_frames(grid)
    successors = _trace_successors(grid, frame_color, frames)
    predecessors = [-1] * len(frames)
    for src, dst in enumerate(successors):
        predecessors[dst] = src
    assert all(pred >= 0 for pred in predecessors)
    return bg, frame_color, frames, successors, predecessors


def _recolor_by_source(
    grid: np.ndarray,
    frames: list[ParsedFrame],
    source_for_frame: list[int],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for frame_idx, src_idx in enumerate(source_for_frame):
        frame = frames[frame_idx]
        r0, c0, r1, c1 = frame.bbox
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[frame.interior_mask] = frames[src_idx].interior_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each framed interior mask with the color received from the frame whose gap points to it.

    Input structure:
    - A dominant background contains several same-colored hollow rectangular frames.
    - Each frame has exactly one single-cell gap on one wall, never on a corner.
    - Each frame interior contains background plus one monochrome mask, and these mask colors are all distinct.

    Rule:
    - For every frame, follow the outward ray starting from its gap cell; the first frame hit on that ray is its successor.
    - The successors form one directed cycle across all frames.
    - Keep every frame border, gap, background cell, and interior mask shape fixed, but recolor each frame's mask with the mask color of its predecessor in that cycle.

    Key insight:
    - The missing wall cell is an arrowhead rather than damage: the hidden cycle is encoded by gap directions, and only the interior colors rotate along that cycle.
    """
    grid = np.asarray(grid, dtype=int)
    assert grid.ndim == 2 and min(grid.shape) > 0
    _bg, _frame_color, frames, _successors, predecessors = _parse_scene(grid)
    return _recolor_by_source(grid, frames, predecessors)


def _neighbors4(r: int, c: int, height: int, width: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            out.append((nr, nc))
    return out


def _random_mask(height: int, width: int) -> np.ndarray:
    area = height * width
    min_cells = max(2, (area + 3) // 4)
    max_cells = max(min_cells, area - 1)
    target = random.randint(min_cells, max_cells)

    mask = np.zeros((height, width), dtype=int)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = 1
    frontier = set(_neighbors4(start[0], start[1], height, width))
    while int(mask.sum()) < target and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        if mask[cell] == 1:
            continue
        mask[cell] = 1
        for nxt in _neighbors4(cell[0], cell[1], height, width):
            if mask[nxt] == 0:
                frontier.add(nxt)

    assert is_contiguous(mask, background=0, connectivity=4)
    return mask.astype(bool)


def _sample_parts(length: int, total: int | None = None) -> list[int] | None:
    values = range(CORRIDOR_MIN, CORRIDOR_MAX + 1)
    options = [
        list(parts)
        for parts in itertools.product(values, repeat=length)
        if total is None or sum(parts) == total
    ]
    if not options:
        return None
    return random.choice(options)


def _make_frame_patch(
    height: int,
    width: int,
    bg: int,
    frame_color: int,
    gap_local: tuple[int, int],
    interior_color: int,
    interior_mask: np.ndarray,
) -> np.ndarray:
    patch = np.full((height, width), int(bg), dtype=int)
    patch[_border_mask(height, width)] = int(frame_color)
    patch[gap_local] = int(bg)
    patch[1:-1, 1:-1][interior_mask] = int(interior_color)
    return patch


def _render_specs(
    height: int,
    width: int,
    bg: int,
    frame_color: int,
    specs: list[FrameSpec],
) -> np.ndarray:
    grid = np.full((height, width), int(bg), dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec.bbox
        patch = _make_frame_patch(
            r1 - r0 + 1,
            c1 - c0 + 1,
            bg,
            frame_color,
            spec.gap_local,
            spec.interior_color,
            spec.interior_mask,
        )
        grid[r0 : r1 + 1, c0 : c1 + 1] = patch
    return grid


def _expected_output_from_specs(
    height: int,
    width: int,
    bg: int,
    frame_color: int,
    specs: list[FrameSpec],
) -> np.ndarray:
    out = _render_specs(height, width, bg, frame_color, specs)
    for idx, spec in enumerate(specs):
        src = specs[idx - 1]
        r0, c0, r1, c1 = spec.bbox
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[spec.interior_mask] = src.interior_color
    return out


def _row_major_wrong_output(grid: np.ndarray) -> np.ndarray:
    _bg, _frame_color, frames, _successors, _predecessors = _parse_scene(grid)
    order = sorted(range(len(frames)), key=lambda idx: (frames[idx].bbox[0], frames[idx].bbox[1]))
    predecessors = [-1] * len(frames)
    for pos, idx in enumerate(order):
        predecessors[idx] = order[(pos - 1) % len(order)]
    return _recolor_by_source(grid, frames, predecessors)


def _forward_wrong_output(grid: np.ndarray) -> np.ndarray:
    _bg, _frame_color, frames, successors, _predecessors = _parse_scene(grid)
    return _recolor_by_source(grid, frames, successors)


def _scene_ok(grid: np.ndarray, expected: np.ndarray) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False

    try:
        bg, frame_color, frames, successors, predecessors = _parse_scene(grid)
        solved = main(grid)
    except AssertionError:
        return False

    if bg == frame_color or len(frames) < 4:
        return False
    if not np.array_equal(solved, expected):
        return False
    if np.array_equal(grid, expected):
        return False
    if np.array_equal(_forward_wrong_output(grid), expected):
        return False
    if np.array_equal(_row_major_wrong_output(grid), expected):
        return False
    if len(set(map(int, grid.ravel()))) < len(frames) + 2:
        return False

    sides = {frame.side for frame in frames}
    if sides != set(FRAME_ORDER):
        return False
    if len(set(successors)) != len(frames) or len(set(predecessors)) != len(frames):
        return False

    for frame in frames:
        r0, c0, r1, c1 = frame.bbox
        before = grid[r0 + 1 : r1, c0 + 1 : c1]
        after = expected[r0 + 1 : r1, c0 + 1 : c1]
        if not np.array_equal(before[~frame.interior_mask], after[~frame.interior_mask]):
            return False
    return True


def _build_specs() -> tuple[int, int, int, int, list[FrameSpec]] | None:
    frame_h = random.randint(*FRAME_H_RANGE)
    frame_w = random.randint(*FRAME_W_RANGE)
    hcount = random.choice(HCOUNT_CHOICES)
    vcount = random.choice(VCOUNT_CHOICES)

    top_corridors = _sample_parts(hcount)
    if top_corridors is None:
        return None
    bottom_corridors = _sample_parts(hcount, total=sum(top_corridors))
    right_corridors = _sample_parts(vcount)
    if right_corridors is None:
        return None
    left_corridors = _sample_parts(vcount, total=sum(right_corridors))
    if bottom_corridors is None or left_corridors is None:
        return None

    pad_top = random.randint(PAD_MIN, PAD_MAX)
    pad_left = random.randint(PAD_MIN, PAD_MAX)
    pad_bottom = random.randint(PAD_MIN, PAD_MAX)
    pad_right = random.randint(PAD_MIN, PAD_MAX)

    top_cols: list[int] = []
    cur_c = pad_left
    for span in top_corridors:
        top_cols.append(cur_c)
        cur_c += frame_w + span
    right_col = cur_c

    right_rows: list[int] = []
    cur_r = pad_top
    for span in right_corridors:
        right_rows.append(cur_r)
        cur_r += frame_h + span
    bottom_row = cur_r

    bottom_cols: list[int] = []
    cur_c = right_col
    for span in bottom_corridors:
        bottom_cols.append(cur_c)
        cur_c -= frame_w + span
    if cur_c != pad_left:
        return None

    left_rows: list[int] = []
    cur_r = bottom_row
    for span in left_corridors:
        left_rows.append(cur_r)
        cur_r -= frame_h + span
    if cur_r != pad_top:
        return None

    height = bottom_row + frame_h + pad_bottom
    width = right_col + frame_w + pad_right
    if max(height, width) > 30:
        return None

    palette = list(PALETTE)
    random.shuffle(palette)
    frame_count = 2 * (hcount + vcount)
    if frame_count + 2 > len(palette):
        return None
    bg = palette.pop()
    frame_color = palette.pop()
    interior_colors = palette[:frame_count]

    specs: list[FrameSpec] = []
    color_iter = iter(interior_colors)

    for col in top_cols:
        gap_local = (random.randint(1, frame_h - 2), frame_w - 1)
        specs.append(
            FrameSpec(
                bbox=(pad_top, col, pad_top + frame_h - 1, col + frame_w - 1),
                side="right",
                gap_local=gap_local,
                interior_color=next(color_iter),
                interior_mask=_random_mask(frame_h - 2, frame_w - 2),
            )
        )

    for row in right_rows:
        gap_local = (frame_h - 1, random.randint(1, frame_w - 2))
        specs.append(
            FrameSpec(
                bbox=(row, right_col, row + frame_h - 1, right_col + frame_w - 1),
                side="bottom",
                gap_local=gap_local,
                interior_color=next(color_iter),
                interior_mask=_random_mask(frame_h - 2, frame_w - 2),
            )
        )

    for col in bottom_cols:
        gap_local = (random.randint(1, frame_h - 2), 0)
        specs.append(
            FrameSpec(
                bbox=(bottom_row, col, bottom_row + frame_h - 1, col + frame_w - 1),
                side="left",
                gap_local=gap_local,
                interior_color=next(color_iter),
                interior_mask=_random_mask(frame_h - 2, frame_w - 2),
            )
        )

    for row in left_rows:
        gap_local = (0, random.randint(1, frame_w - 2))
        specs.append(
            FrameSpec(
                bbox=(row, pad_left, row + frame_h - 1, pad_left + frame_w - 1),
                side="top",
                gap_local=gap_local,
                interior_color=next(color_iter),
                interior_mask=_random_mask(frame_h - 2, frame_w - 2),
            )
        )

    return height, width, bg, frame_color, specs


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_specs()
        if built is None:
            continue
        height, width, bg, frame_color, specs = built
        grid = _render_specs(height, width, bg, frame_color, specs)
        expected = _expected_output_from_specs(height, width, bg, frame_color, specs)
        if _scene_ok(grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
