from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(int(v) for v in np.unique(grid) if int(v) != int(bg))


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _frame_mask(height: int, width: int, notch_side: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    if notch_side in {"up", "down"}:
        mid = width // 2
        row = 0 if notch_side == "up" else height - 1
        mask[row, mid] = False
    else:
        mid = height // 2
        col = 0 if notch_side == "left" else width - 1
        mask[mid, col] = False
    return mask


def _detect_notch_side(mask: np.ndarray) -> str:
    matches = [side for side in DIRS if np.array_equal(mask, _frame_mask(mask.shape[0], mask.shape[1], side))]
    assert len(matches) == 1, "expected a single-gap rectangular frame"
    return matches[0]


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _extract_frames(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _get_bg(g)
    colors = _colors_present(g, bg)
    assert len(colors) in {3, 4}, "expected 3-4 nested frame colors"

    frames: list[dict] = []
    for color in colors:
        comps = _components(g == color)
        assert len(comps) == 1, "each frame color must form one component"
        coords = comps[0]
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        notch_side = _detect_notch_side(local)
        frames.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": bbox,
                "mask": local,
                "notch_side": notch_side,
            }
        )

    for frame in frames:
        depth = sum(_strictly_contains(other["bbox"], frame["bbox"]) for other in frames)
        frame["depth"] = int(depth)

    depths = sorted(frame["depth"] for frame in frames)
    assert depths == list(range(len(frames))), "frames must form one strict nesting chain"
    ordered = sorted(frames, key=lambda frame: frame["depth"])
    for outer, inner in zip(ordered, ordered[1:]):
        assert _strictly_contains(outer["bbox"], inner["bbox"])
    return bg, ordered


def _render_frames(
    shape: tuple[int, int],
    bg: int,
    frames: list[dict],
    *,
    use_depth_distance: bool = True,
    deep_priority: bool = True,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    ordered = sorted(frames, key=lambda frame: frame["depth"], reverse=not deep_priority)
    for frame in ordered:
        dist = frame["depth"] + 1 if use_depth_distance else 1
        dr, dc = DIRS[frame["notch_side"]]
        rr = frame["coords"][:, 0] + dr * dist
        cc = frame["coords"][:, 1] + dc * dist
        assert np.all((0 <= rr) & (rr < shape[0]) & (0 <= cc) & (cc < shape[1]))
        out[rr, cc] = int(frame["color"])
    return out


def _overlap_count(shape: tuple[int, int], frames: list[dict]) -> int:
    counts = np.zeros(shape, dtype=int)
    for frame in frames:
        dist = frame["depth"] + 1
        dr, dc = DIRS[frame["notch_side"]]
        rr = frame["coords"][:, 0] + dr * dist
        cc = frame["coords"][:, 1] + dc * dist
        counts[rr, cc] += 1
    return int(np.count_nonzero(counts > 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each nested rectangular frame along its notch direction, with deeper frames moving farther and painting on top.

    Input structure:
    - The grid has a black background and 3-4 distinct non-background colors.
    - Each color forms one one-pixel-thick rectangular frame with exactly one one-pixel gap at the midpoint of one side.
    - The frames are strictly nested, so every inner frame is fully contained inside the next outer frame.

    Rule:
    - Compute each frame's nesting depth: the outermost frame has depth 0, the next has depth 1, and so on.
    - The missing midpoint determines slide direction: top gap means up, bottom means down, left means left, right means right.
    - Translate every frame by depth+1 cells in that direction, then compose all translated frames on one blank canvas.
    - When translated frames overlap, the deeper frame overwrites the shallower frame.

    Key insight:
    - The same nesting depth controls both the motion magnitude and the conflict priority, so solving the frame hierarchy is necessary before either sliding or overlaying.
    """
    bg, frames = _extract_frames(grid)
    return _render_frames(np.asarray(grid).shape, bg, frames, use_depth_distance=True, deep_priority=True)


def _choose_directions(count: int) -> list[str]:
    pair_idx = random.randrange(count - 1)
    axis = random.choice(["vertical", "horizontal"])
    directions: list[str | None] = [None] * count
    if axis == "vertical":
        directions[pair_idx] = "down"
        directions[pair_idx + 1] = "up"
        leftovers = ["left", "right"]
    else:
        directions[pair_idx] = "right"
        directions[pair_idx + 1] = "left"
        leftovers = ["up", "down"]
    random.shuffle(leftovers)
    for i in range(count):
        if directions[i] is None:
            directions[i] = leftovers.pop()
    return [str(v) for v in directions]


def _sample_geometry() -> tuple[int, int, list[tuple[int, int, int, int]], list[int]]:
    count = random.choice([3, 4])
    gaps = [random.choice([3, 4]) for _ in range(count - 1)]
    gaps[random.randrange(count - 1)] = 3
    inset_total = sum(gaps)

    max_outer = 25
    inner_h_choices = [v for v in (3, 5, 7, 9) if v + 2 * inset_total <= max_outer]
    inner_w_choices = [v for v in (3, 5, 7, 9, 11) if v + 2 * inset_total <= max_outer]
    inner_h = random.choice(inner_h_choices)
    inner_w = random.choice(inner_w_choices)
    outer_h = inner_h + 2 * inset_total
    outer_w = inner_w + 2 * inset_total

    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    grid_h = outer_h + top_margin + bottom_margin
    grid_w = outer_w + left_margin + right_margin
    if grid_h > 30 or grid_w > 30:
        raise AssertionError("sampled grid exceeds ARC size bound")

    outer_r0 = top_margin
    outer_c0 = left_margin
    bboxes = []
    inset = 0
    for depth in range(count):
        bboxes.append(
            (
                outer_r0 + inset,
                outer_c0 + inset,
                outer_r0 + outer_h - inset - 1,
                outer_c0 + outer_w - inset - 1,
            )
        )
        if depth + 1 < count:
            inset += gaps[depth]
    return grid_h, grid_w, bboxes, gaps


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, notch_side: str) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = _frame_mask(r1 - r0 + 1, c1 - c0 + 1, notch_side)
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    grid[r0 : r1 + 1, c0 : c1 + 1] = np.where(patch != 0, patch, grid[r0 : r1 + 1, c0 : c1 + 1])
    return mask


def _validate_input(grid: np.ndarray, frames: list[dict], gaps: list[int]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if any(gap < 3 for gap in gaps):
        return False
    parsed_bg, parsed = _extract_frames(grid)
    if parsed_bg != 0 or len(parsed) != len(frames):
        return False
    for want, got in zip(frames, parsed):
        if want["color"] != got["color"]:
            return False
        if want["bbox"] != got["bbox"]:
            return False
        if want["notch_side"] != got["notch_side"]:
            return False
        if want["depth"] != got["depth"]:
            return False
    return True


def _try_build_case() -> tuple[np.ndarray, np.ndarray]:
    grid_h, grid_w, bboxes, gaps = _sample_geometry()
    count = len(bboxes)
    colors = random.sample(list(range(1, 10)), count)
    directions = _choose_directions(count)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    frames: list[dict] = []
    for depth, (bbox, color, direction) in enumerate(zip(bboxes, colors, directions)):
        mask = _paint_frame(grid, bbox, color, direction)
        r0, c0, _r1, _c1 = bbox
        local_coords = np.argwhere(mask)
        coords = local_coords + np.array([r0, c0], dtype=int)
        frames.append(
            {
                "color": int(color),
                "bbox": bbox,
                "coords": coords.astype(int),
                "mask": mask,
                "notch_side": direction,
                "depth": depth,
            }
        )

    expected = _render_frames(grid.shape, 0, frames, use_depth_distance=True, deep_priority=True)
    if not _validate_input(grid, frames, gaps):
        raise AssertionError("constructed input failed validation")
    if _overlap_count(grid.shape, frames) <= 0:
        raise AssertionError("priority must matter on at least one cell")
    if np.array_equal(expected, _render_frames(grid.shape, 0, frames, use_depth_distance=True, deep_priority=False)):
        raise AssertionError("reversed layer priority should change the output")
    if np.array_equal(expected, _render_frames(grid.shape, 0, frames, use_depth_distance=False, deep_priority=True)):
        raise AssertionError("fixed slide distance should change the output")
    if not np.array_equal(main(grid), expected):
        raise AssertionError("solver disagrees with construction oracle")
    return grid, expected


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(200):
        try:
            return _try_build_case()
        except AssertionError:
            continue
    raise AssertionError("failed to build a semantically valid case")


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            grid, _ = _build_case()
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid nested-frame sliding puzzle")
