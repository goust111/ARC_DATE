import random
from typing import List, Sequence, Tuple

import numpy as np


GRID_MIN = 12
GRID_MAX = 30
MAX_GENERATION_ATTEMPTS = 500
COLOR_IDS = tuple(range(10))
ROW_MODES = ("top", "bottom", "tie")
COL_MODES = ("left", "right", "tie")


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> List[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                if rr > 0 and mask[rr - 1, cc] and not seen[rr - 1, cc]:
                    seen[rr - 1, cc] = True
                    stack.append((rr - 1, cc))
                if rr + 1 < h and mask[rr + 1, cc] and not seen[rr + 1, cc]:
                    seen[rr + 1, cc] = True
                    stack.append((rr + 1, cc))
                if cc > 0 and mask[rr, cc - 1] and not seen[rr, cc - 1]:
                    seen[rr, cc - 1] = True
                    stack.append((rr, cc - 1))
                if cc + 1 < w and mask[rr, cc + 1] and not seen[rr, cc + 1]:
                    seen[rr, cc + 1] = True
                    stack.append((rr, cc + 1))
            comps.append(np.asarray(cells, dtype=int))

    return comps


def _frame_bbox(coords: np.ndarray) -> Tuple[int, int, int, int] | None:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())
    height = r1 - r0 + 1
    width = c1 - c0 + 1

    if height < 3 or width < 3:
        return None
    if int(coords.shape[0]) != 2 * (height + width) - 4:
        return None
    on_border = (rows == r0) | (rows == r1) | (cols == c0) | (cols == c1)
    if not bool(np.all(on_border)):
        return None
    return r0, c0, r1, c1


def _detect_frames(grid: np.ndarray, bg: int) -> Tuple[int, List[Tuple[Tuple[int, int, int, int], np.ndarray]]]:
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        frames = []
        for comp in _components(grid == color):
            bbox = _frame_bbox(comp)
            if bbox is not None:
                frames.append((bbox, comp))
        if frames:
            area = sum(int(comp.shape[0]) for _, comp in frames)
            candidates.append((len(frames), area, color, frames))

    if not candidates:
        return -1, []

    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    _, _, frame_color, frames = candidates[0]
    frames.sort(key=lambda item: item[0])
    return frame_color, frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each rectangular frame edge according to where its interior anchor projects.

    Input structure:
    - The grid has a dominant background color.
    - One or more one-pixel-thick closed rectangular frames share a single frame color and do not overlap.
    - Each frame contains exactly one interior anchor pixel whose color differs from both the background and the frame.

    Rule:
    - For each frame, find its unique interior anchor pixel.
    - Compare the anchor row against the top and bottom frame edges, and repaint the nearer horizontal edge with the anchor color; if the distances are equal, repaint neither horizontal edge.
    - Compare the anchor column against the left and right frame edges, and repaint the nearer vertical edge with the anchor color; if the distances are equal, repaint neither vertical edge.
    - Erase the anchor pixel itself to background, while leaving every unrepainted frame cell unchanged.

    Key insight:
    - The anchor independently chooses one horizontal side and one vertical side by relative offset inside its enclosing rectangle, so the two axes must be solved separately.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    frame_color, frames = _detect_frames(g, bg)
    if not frames:
        return g.copy()

    out = g.copy()
    for (r0, c0, r1, c1), _ in frames:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        anchor_local = np.argwhere((interior != bg) & (interior != frame_color))
        assert anchor_local.shape[0] == 1, "each frame must contain exactly one anchor"

        ar = int(anchor_local[0, 0] + r0 + 1)
        ac = int(anchor_local[0, 1] + c0 + 1)
        anchor_color = int(g[ar, ac])

        out[ar, ac] = bg

        dist_top = ar - r0
        dist_bottom = r1 - ar
        if dist_top < dist_bottom:
            out[r0, c0 : c1 + 1] = anchor_color
        elif dist_bottom < dist_top:
            out[r1, c0 : c1 + 1] = anchor_color

        dist_left = ac - c0
        dist_right = c1 - ac
        if dist_left < dist_right:
            out[r0 : r1 + 1, c0] = anchor_color
        elif dist_right < dist_left:
            out[r0 : r1 + 1, c1] = anchor_color

    return out


def _sample_box_size(mode: str, limit: int, rng: random.Random) -> int | None:
    min_size = 3 if mode == "tie" else 4
    choices = [size for size in range(min_size, limit + 1) if mode != "tie" or size % 2 == 1]
    if not choices:
        return None
    return int(rng.choice(choices))


def _sample_anchor_index(size: int, mode: str, rng: random.Random) -> int:
    if mode == "tie":
        return (size - 1) // 2
    if mode in ("top", "left"):
        return rng.randint(1, (size - 2) // 2)
    return rng.randint((size + 1) // 2, size - 2)


def _candidate_positions(blocked: np.ndarray, box_h: int, box_w: int) -> List[Tuple[int, int]]:
    h, w = blocked.shape
    if h - box_h < 2 or w - box_w < 2:
        return []

    positions = []
    for r0 in range(1, h - box_h):
        for c0 in range(1, w - box_w):
            if not bool(blocked[r0 : r0 + box_h, c0 : c0 + box_w].any()):
                positions.append((r0, c0))
    return positions


def _paint_frame(grid: np.ndarray, r0: int, c0: int, box_h: int, box_w: int, frame_color: int) -> None:
    r1 = r0 + box_h - 1
    c1 = c0 + box_w - 1
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color


def _spec_counts_valid(specs: Sequence[dict]) -> bool:
    return any(spec["row_mode"] != "tie" or spec["col_mode"] != "tie" for spec in specs)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    area = h * w
    if area < 220:
        max_frames = 2
    elif area < 480:
        max_frames = 3
    else:
        max_frames = 4
    num_frames = rng.randint(1, max_frames)

    colors = rng.sample(COLOR_IDS, 2 + num_frames)
    bg = colors[0]
    frame_color = colors[1]
    anchor_colors = colors[2:]

    size_cap = {1: 13, 2: 11, 3: 9, 4: 8}[num_frames]
    specs = []
    for anchor_color in anchor_colors:
        row_mode = rng.choice(ROW_MODES)
        col_mode = rng.choice(COL_MODES)
        box_h_limit = min(size_cap, h - 2)
        box_w_limit = min(size_cap, w - 2)
        box_h = _sample_box_size(row_mode, box_h_limit, rng)
        box_w = _sample_box_size(col_mode, box_w_limit, rng)
        if box_h is None or box_w is None:
            return None
        specs.append(
            {
                "row_mode": row_mode,
                "col_mode": col_mode,
                "box_h": box_h,
                "box_w": box_w,
                "anchor_color": anchor_color,
            }
        )

    if not _spec_counts_valid(specs):
        return None

    blocked = np.zeros((h, w), dtype=bool)
    placements = []
    for spec in sorted(specs, key=lambda item: item["box_h"] * item["box_w"], reverse=True):
        positions = _candidate_positions(blocked, spec["box_h"], spec["box_w"])
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        r1 = r0 + spec["box_h"] - 1
        c1 = c0 + spec["box_w"] - 1
        blocked[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True
        placements.append(
            {
                **spec,
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
            }
        )

    grid = np.full((h, w), bg, dtype=int)
    for placement in placements:
        _paint_frame(
            grid,
            placement["r0"],
            placement["c0"],
            placement["box_h"],
            placement["box_w"],
            frame_color,
        )
        anchor_r = placement["r0"] + _sample_anchor_index(placement["box_h"], placement["row_mode"], rng)
        anchor_c = placement["c0"] + _sample_anchor_index(placement["box_w"], placement["col_mode"], rng)
        grid[anchor_r, anchor_c] = placement["anchor_color"]

    if _mode_color(grid) != bg:
        return None

    detected_frame_color, frames = _detect_frames(grid, bg)
    if detected_frame_color != frame_color or len(frames) != num_frames:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
