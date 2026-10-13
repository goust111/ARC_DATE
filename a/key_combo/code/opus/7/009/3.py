from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


_ARC_ROOT = Path("/home/lixinyu/snap/new")
if str(_ARC_ROOT) not in sys.path:
    sys.path.append(str(_ARC_ROOT))

from solvers.arc_utils import Color, find_components


BG = Color.BLACK
FRAME = Color.GREY
MARKER_COLORS = (Color.BLUE, Color.RED, Color.GREEN)
GEN_RETRIES = 400
SHAPES = ((3, 3), (3, 5), (5, 3), (5, 5))


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _checker_mask(h: int, w: int) -> np.ndarray:
    rr, cc = np.indices((h, w))
    return (rr + cc) % 2 == 0


def _cross_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    starts = (
        (0, 0, 1, 1),
        (0, w - 1, 1, -1),
        (h - 1, 0, -1, 1),
        (h - 1, w - 1, -1, -1),
    )
    for r, c, dr, dc in starts:
        while 0 <= r < h and 0 <= c < w:
            mask[r, c] = True
            r += dr
            c += dc
    return mask


def _is_hollow_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    return np.array_equal(patch == FRAME, _border_mask(h, w))


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    grey_components = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[FRAME],
        return_format="dict",
    )
    for comp in grey_components:
        bbox = tuple(map(int, comp["bbox"]))
        if not _is_hollow_frame(grid, bbox):
            continue
        r0, c0, r1, c1 = bbox
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        markers = np.argwhere((interior != BG) & (interior != FRAME))
        if markers.shape[0] != 1:
            continue
        mr, mc = map(int, markers[0])
        frames.append(
            {
                "bbox": bbox,
                "inner_top_left": (r0 + 1, c0 + 1),
                "inner_shape": tuple(map(int, interior.shape)),
                "marker_pos": (r0 + 1 + mr, c0 + 1 + mc),
                "color": int(interior[mr, mc]),
            }
        )
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _paint_inner(out: np.ndarray, frame: dict, use_checker: bool) -> None:
    r0, c0 = frame["inner_top_left"]
    h, w = frame["inner_shape"]
    color = int(frame["color"])
    mask = _checker_mask(h, w) if use_checker else _cross_mask(h, w)
    inner = out[r0 : r0 + h, c0 : c0 + w]
    inner[:] = BG
    inner[mask] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each frame's outside color count to choose checkerboard or truncated X filling inside the frame.

    Input structure:
    - A black background contains 2-3 grey hollow rectangular frames that do not overlap.
    - Each frame contains exactly one centered colored marker pixel inside its black interior.
    - Outside the frames, isolated pixels reuse those marker colors as count signals.

    Rule:
    - Detect every grey hollow frame and read its unique interior marker color C.
    - Count outside-frame pixels of color C, and compare that count with the number of checkerboard cells in the frame interior, starting from the interior top-left cell.
    - If the counts match, fill the interior with a checkerboard using color C on the selected cells; otherwise draw the truncated diagonal X from the four interior corners with color C.
    - Erase all outside colored pixels to black and keep the grey frame borders unchanged.

    Key insight:
    - The scattered outside pixels are not moved into the frame; their count is the control signal that selects between two different interior fill algorithms.
    """
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    out = np.zeros_like(g)
    out[g == FRAME] = FRAME

    inside_any_frame = np.zeros(g.shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inside_any_frame[r0 : r1 + 1, c0 : c1 + 1] = True

    for frame in frames:
        h, w = frame["inner_shape"]
        checker_cells = (h * w + 1) // 2
        color = int(frame["color"])
        outside_count = int(np.count_nonzero((g == color) & (~inside_any_frame)))
        _paint_inner(out, frame, outside_count == checker_cells)

    return out


def _expanded_bbox(bbox: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(h - 1, r1 + pad),
        min(w - 1, c1 + pad),
    )


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 2) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _place_frames(rng: np.random.Generator, shape: tuple[int, int], specs: list[dict]) -> list[dict] | None:
    h, w = shape
    placed = []
    for spec in specs:
        ih, iw = spec["inner_shape"]
        fh = ih + 2
        fw = iw + 2
        candidates = [
            (r0, c0)
            for r0 in range(h - fh + 1)
            for c0 in range(w - fw + 1)
        ]
        rng.shuffle(candidates)
        chosen = None
        for r0, c0 in candidates:
            bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if any(_boxes_too_close(bbox, other["bbox"]) for other in placed):
                continue
            chosen = bbox
            break
        if chosen is None:
            return None
        placed.append({**spec, "bbox": chosen})
    return placed


def _mark_blocked(blocked: np.ndarray, r: int, c: int) -> None:
    h, w = blocked.shape
    r0 = max(0, r - 1)
    r1 = min(h, r + 2)
    c0 = max(0, c - 1)
    c1 = min(w, c + 2)
    blocked[r0:r1, c0:c1] = True


def _place_outside_pixels(
    rng: np.random.Generator,
    grid: np.ndarray,
    frames: list[dict],
    counts: dict[int, int],
) -> bool:
    blocked = np.zeros(grid.shape, dtype=bool)
    for frame in frames:
        halo = _expanded_bbox(frame["bbox"], 1, grid.shape)
        r0, c0, r1, c1 = halo
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    coords = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1])]
    for color, count in counts.items():
        for _ in range(count):
            rng.shuffle(coords)
            placed = False
            for r, c in coords:
                if blocked[r, c] or grid[r, c] != BG:
                    continue
                grid[r, c] = color
                _mark_blocked(blocked, r, c)
                placed = True
                break
            if not placed:
                return False
    return True


def _make_frame_specs(rng: np.random.Generator, frame_count: int) -> list[dict] | None:
    shape_indices = rng.choice(len(SHAPES), size=frame_count, replace=False)
    inner_shapes = [SHAPES[int(i)] for i in shape_indices]
    if not any(h != w for h, w in inner_shapes):
        return None

    colors = list(rng.choice(MARKER_COLORS, size=frame_count, replace=False))
    rng.shuffle(colors)

    checker_flags = [False] * frame_count
    checker_idx = int(rng.integers(frame_count))
    checker_flags[checker_idx] = True
    other_indices = [idx for idx in range(frame_count) if idx != checker_idx]
    if other_indices and rng.random() < 0.45:
        checker_flags[int(rng.choice(other_indices))] = True
    if all(checker_flags):
        checker_flags[int(rng.choice(other_indices))] = False

    specs = []
    for idx, (inner_shape, color, use_checker) in enumerate(zip(inner_shapes, colors, checker_flags)):
        h, w = inner_shape
        checker_cells = (h * w + 1) // 2
        if use_checker:
            if checker_cells > 10:
                return None
            count = checker_cells
        else:
            choices = [n for n in range(2, 11) if n != checker_cells]
            count = int(rng.choice(choices))
        specs.append(
            {
                "inner_shape": inner_shape,
                "color": int(color),
                "count": int(count),
                "use_checker": bool(use_checker),
            }
        )

    if not any(spec["use_checker"] for spec in specs):
        return None
    if not any(not spec["use_checker"] for spec in specs):
        return None
    return specs


def _draw_input(grid: np.ndarray, frames: list[dict]) -> None:
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        grid[r0 : r1 + 1, c0] = FRAME
        grid[r0 : r1 + 1, c1] = FRAME
        grid[r0, c0 : c1 + 1] = FRAME
        grid[r1, c0 : c1 + 1] = FRAME
        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        grid[cr, cc] = int(frame["color"])
        frame["marker_pos"] = (cr, cc)


def _generate_case(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(16, 24))
    w = int(rng.integers(16, 24))
    frame_count = int(rng.integers(2, 4))

    specs = _make_frame_specs(rng, frame_count)
    if specs is None:
        return None

    frames = _place_frames(rng, (h, w), specs)
    if frames is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    _draw_input(grid, frames)

    counts = {int(frame["color"]): int(frame["count"]) for frame in frames}
    if not _place_outside_pixels(rng, grid, frames, counts):
        return None

    parsed = _find_frames(grid)
    if len(parsed) != frame_count:
        return None

    branch_set = set()
    inside_any_frame = np.zeros(grid.shape, dtype=bool)
    for frame in parsed:
        r0, c0, r1, c1 = frame["bbox"]
        inside_any_frame[r0 : r1 + 1, c0 : c1 + 1] = True
    for frame in parsed:
        h0, w0 = frame["inner_shape"]
        checker_cells = (h0 * w0 + 1) // 2
        outside_count = int(np.count_nonzero((grid == frame["color"]) & (~inside_any_frame)))
        branch_set.add(outside_count == checker_cells)
    if branch_set != {False, True}:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = np.random.default_rng()
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
