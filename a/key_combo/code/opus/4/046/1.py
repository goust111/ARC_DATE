from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, find_components, get_bg, in_bounds


FRAME_COLOR = Color.GREY
CONTROL_COLORS = (Color.BLUE, Color.GREEN, Color.YELLOW, Color.ORANGE)
BG_COLOR = Color.BLACK
GEN_RETRIES = 400


@dataclass(frozen=True)
class Frame:
    bbox: tuple[int, int, int, int]
    parent: int | None


@dataclass(frozen=True)
class ControlInfo:
    row: int
    col: int
    color: int
    depth: int
    direction_name: str
    direction: tuple[int, int]
    ray: tuple[tuple[int, int], ...]
    nearest_distance: int
    crosses_frame: bool
    differs_from_outer: bool


def _contains_bbox(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _contains_point(frame: Frame, row: int, col: int) -> bool:
    r0, c0, r1, c1 = frame.bbox
    return r0 < row < r1 and c0 < col < c1


def _frame_area(frame: Frame) -> int:
    return bbox_h(frame.bbox) * bbox_w(frame.bbox)


def _rect_frame_mask(bbox: tuple[int, int, int, int]) -> np.ndarray:
    h = bbox_h(bbox)
    w = bbox_w(bbox)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _extract_frames(grid: np.ndarray) -> list[Frame]:
    comps = find_components(
        grid,
        bg=BG_COLOR,
        colors=[FRAME_COLOR],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert 2 <= len(comps) <= 3, "expected 2 or 3 grey frame components"

    bboxes: list[tuple[int, int, int, int]] = []
    for comp in comps:
        bbox = tuple(map(int, comp["bbox"]))
        h = bbox_h(bbox)
        w = bbox_w(bbox)
        assert h >= 3 and w >= 3, "frame bbox must have an interior"
        patch = np.zeros((h, w), dtype=bool)
        r0, c0, _r1, _c1 = bbox
        coords = np.asarray(comp["coords"], dtype=int)
        patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
        expected = _rect_frame_mask(bbox)
        assert np.array_equal(patch, expected), "grey component must be a hollow rectangle"
        bboxes.append(bbox)

    for i, box_a in enumerate(bboxes):
        for j in range(i + 1, len(bboxes)):
            box_b = bboxes[j]
            assert (
                _contains_bbox(box_a, box_b)
                or _contains_bbox(box_b, box_a)
                or box_a[2] < box_b[0]
                or box_b[2] < box_a[0]
                or box_a[3] < box_b[1]
                or box_b[3] < box_a[1]
            ), "frames may only be nested or disjoint"

    bboxes.sort(key=lambda bbox: bbox_h(bbox) * bbox_w(bbox), reverse=True)
    frames: list[Frame] = []
    for i, bbox in enumerate(bboxes):
        containers = [j for j in range(i) if _contains_bbox(bboxes[j], bbox)]
        parent = min(containers, key=lambda j: bbox_h(bboxes[j]) * bbox_w(bboxes[j])) if containers else None
        frames.append(Frame(bbox=bbox, parent=parent))
    return frames


def _extract_controls(grid: np.ndarray) -> list[tuple[int, int, int]]:
    comps = find_components(
        grid,
        bg=BG_COLOR,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    controls: list[tuple[int, int, int]] = []
    for comp in comps:
        color = int(comp["color"])
        if color in (BG_COLOR, FRAME_COLOR):
            continue
        assert color in CONTROL_COLORS, "unexpected control color"
        assert int(comp["area"]) == 1, "each control must be an isolated single pixel"
        row, col = map(int, comp["coords"][0])
        controls.append((row, col, color))
    assert controls, "expected at least one control pixel"
    controls.sort()
    return controls


def _distances_to_edges(bbox: tuple[int, int, int, int], row: int, col: int) -> dict[str, int]:
    r0, c0, r1, c1 = bbox
    return {
        "up": row - r0,
        "down": r1 - row,
        "left": col - c0,
        "right": c1 - col,
    }


def _direction_step(name: str) -> tuple[int, int]:
    return {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[name]


def _control_info(frames: list[Frame], row: int, col: int, color: int, shape: tuple[int, int]) -> ControlInfo:
    containing = [idx for idx, frame in enumerate(frames) if _contains_point(frame, row, col)]
    assert containing, "control must be inside at least one frame"

    direct_idx = min(containing, key=lambda idx: _frame_area(frames[idx]))
    direct_bbox = frames[direct_idx].bbox
    depth = len(containing)

    distances = _distances_to_edges(direct_bbox, row, col)
    assert len(set(distances.values())) == 4, "nearest direct-frame edge must be unique"
    direction_name = min(distances, key=distances.get)
    direction = _direction_step(direction_name)

    ray: list[tuple[int, int]] = []
    for step in range(1, depth + 1):
        rr = row + direction[0] * step
        cc = col + direction[1] * step
        if not in_bounds(shape, rr, cc):
            break
        ray.append((rr, cc))

    outer_bbox = frames[max(containing, key=lambda idx: _frame_area(frames[idx]))].bbox
    outer_distances = _distances_to_edges(outer_bbox, row, col)
    outer_unique = len(set(outer_distances.values())) == 4
    outer_direction = min(outer_distances, key=outer_distances.get) if outer_unique else None

    return ControlInfo(
        row=row,
        col=col,
        color=color,
        depth=depth,
        direction_name=direction_name,
        direction=direction,
        ray=tuple(ray),
        nearest_distance=int(distances[direction_name]),
        crosses_frame=depth >= int(distances[direction_name]),
        differs_from_outer=bool(outer_unique and outer_direction != direction_name),
    )


def _predict_output(grid: np.ndarray) -> np.ndarray:
    frames = _extract_frames(grid)
    controls = _extract_controls(grid)
    infos = [_control_info(frames, row, col, color, grid.shape) for row, col, color in controls]
    control_cells = {(info.row, info.col) for info in infos}

    out = np.array(grid, copy=True)
    for info in infos:
        for rr, cc in info.ray:
            if (rr, cc) in control_cells:
                raise AssertionError("ray may not pass through another control")
            current = int(out[rr, cc])
            assert current in (BG_COLOR, FRAME_COLOR, info.color), "rays may only cover background or frame cells"
            out[rr, cc] = info.color
        out[info.row, info.col] = Color.RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each nested-frame control pixel red and emit a depth-length ray toward its nearest enclosing frame edge.

    Input structure:
    - The background is black.
    - Grey pixels form 2 or 3 hollow rectangular frames that are pairwise nested or disjoint, with the task instances using one strict nesting chain.
    - Blue, green, yellow, or orange single-pixel controls lie strictly inside at least one frame and never on a grey border.

    Rule:
    - Detect every grey rectangular frame and recover the containment relation between them.
    - For each control pixel, let its depth be the number of frames whose interior strictly contains it, and let its direct container be the smallest such frame.
    - Use the unique nearest edge of that direct container to choose one cardinal ray direction, paint the next `depth` cells in the control's original color, and finally recolor the control pixel itself red.

    Key insight:
    - The enclosing-frame topology provides a numeric depth, while the geometry of the direct enclosing frame provides the ray direction; both must be recovered from the same nested-box structure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert get_bg(g) == BG_COLOR, "background must be black"
    return _predict_output(g)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR


def _sample_chain(rng: np.random.Generator, height: int, width: int, frame_count: int) -> list[tuple[int, int, int, int]] | None:
    min_outer_h = 5 + 4 * (frame_count - 1)
    min_outer_w = 5 + 4 * (frame_count - 1)
    if height < min_outer_h + 2 or width < min_outer_w + 2:
        return None

    outer_h = int(rng.integers(min_outer_h, height - 1))
    outer_w = int(rng.integers(min_outer_w, width - 1))
    r0 = int(rng.integers(1, height - outer_h))
    c0 = int(rng.integers(1, width - outer_w))
    current = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
    chain = [current]

    for depth in range(1, frame_count):
        remaining = frame_count - depth - 1
        min_child_h = 5 + 4 * remaining
        min_child_w = 5 + 4 * remaining
        cur_h = bbox_h(current)
        cur_w = bbox_w(current)
        if cur_h < min_child_h + 4 or cur_w < min_child_w + 4:
            return None

        for _ in range(80):
            child_h = int(rng.integers(min_child_h, cur_h - 3))
            child_w = int(rng.integers(min_child_w, cur_w - 3))
            gap_h = cur_h - child_h
            gap_w = cur_w - child_w
            if gap_h < 4 or gap_w < 4:
                continue
            top_gap = int(rng.integers(2, gap_h - 1))
            left_gap = int(rng.integers(2, gap_w - 1))
            bottom_gap = gap_h - top_gap
            right_gap = gap_w - left_gap
            if min(bottom_gap, right_gap) < 2:
                continue

            r0, c0, r1, c1 = current
            child = (r0 + top_gap, c0 + left_gap, r1 - bottom_gap, c1 - right_gap)
            if bbox_h(child) < min_child_h or bbox_w(child) < min_child_w:
                continue
            chain.append(child)
            current = child
            break
        else:
            return None

    return chain


def _zone_depth(chain: list[tuple[int, int, int, int]], row: int, col: int) -> int:
    depth = 0
    for bbox in chain:
        if bbox[0] < row < bbox[2] and bbox[1] < col < bbox[3]:
            depth += 1
    return depth


def _on_frame_border(chain: list[tuple[int, int, int, int]], row: int, col: int) -> bool:
    for r0, c0, r1, c1 in chain:
        if (row == r0 or row == r1) and c0 <= col <= c1:
            return True
        if (col == c0 or col == c1) and r0 <= row <= r1:
            return True
    return False


def _candidate_controls(chain: list[tuple[int, int, int, int]], shape: tuple[int, int]) -> dict[int, list[ControlInfo]]:
    height, width = shape
    frames = [Frame(bbox=bbox, parent=(idx - 1 if idx > 0 else None)) for idx, bbox in enumerate(chain)]
    candidates = {depth: [] for depth in range(1, len(chain) + 1)}

    for row in range(height):
        for col in range(width):
            if _on_frame_border(chain, row, col):
                continue
            depth = _zone_depth(chain, row, col)
            if depth == 0:
                continue
            try:
                info = _control_info(frames, row, col, CONTROL_COLORS[0], shape)
            except AssertionError:
                continue
            if len(info.ray) != depth:
                continue
            candidates[depth].append(info)

    return candidates


def _touching(p1: tuple[int, int], p2: tuple[int, int]) -> bool:
    return abs(p1[0] - p2[0]) <= 1 and abs(p1[1] - p2[1]) <= 1


def _pick_controls(rng: np.random.Generator, candidates: dict[int, list[ControlInfo]], frame_count: int) -> list[ControlInfo] | None:
    required_depths = list(range(1, frame_count + 1))
    extra_count = 1
    extra_depths = rng.integers(1, frame_count + 1, size=extra_count).tolist()
    wanted_depths = required_depths + extra_depths
    rng.shuffle(wanted_depths)

    pools = {depth: list(candidates[depth]) for depth in candidates}
    for depth in pools:
        rng.shuffle(pools[depth])
        pools[depth] = pools[depth][:24]

    chosen: list[ControlInfo] = []
    used_ray: set[tuple[int, int]] = set()
    used_controls: set[tuple[int, int]] = set()

    def backtrack(index: int) -> bool:
        if index == len(wanted_depths):
            directions = {info.direction_name for info in chosen}
            depths = {info.depth for info in chosen}
            if depths != set(range(1, frame_count + 1)):
                return False
            if len(directions) < 2:
                return False
            if not any(info.crosses_frame for info in chosen):
                return False
            if not any(not info.crosses_frame for info in chosen):
                return False
            if frame_count >= 2 and not any(info.differs_from_outer for info in chosen):
                return False
            return True

        depth = wanted_depths[index]
        for candidate in pools[depth]:
            pos = (candidate.row, candidate.col)
            if pos in used_controls or pos in used_ray:
                continue
            if any(_touching(pos, other) for other in used_controls):
                continue
            if any(cell in used_ray or cell in used_controls for cell in candidate.ray):
                continue

            chosen.append(candidate)
            used_controls.add(pos)
            used_ray.update(candidate.ray)
            if backtrack(index + 1):
                return True
            chosen.pop()
            used_controls.remove(pos)
            for cell in candidate.ray:
                if all(cell not in other.ray for other in chosen):
                    used_ray.discard(cell)
        return False

    return chosen if backtrack(0) else None


def _assign_colors(rng: np.random.Generator, count: int) -> list[int]:
    if count <= len(CONTROL_COLORS) and rng.random() < 0.7:
        colors = rng.choice(np.array(CONTROL_COLORS, dtype=int), size=count, replace=False).tolist()
    else:
        colors = rng.choice(np.array(CONTROL_COLORS, dtype=int), size=count, replace=True).tolist()
        if len(set(colors)) == 1:
            colors[-1] = int(rng.choice([color for color in CONTROL_COLORS if color != colors[0]]))
    return [int(color) for color in colors]


def _build_grid(
    shape: tuple[int, int],
    chain: list[tuple[int, int, int, int]],
    controls: list[ControlInfo],
    colors: list[int],
) -> np.ndarray:
    grid = np.full(shape, BG_COLOR, dtype=int)
    for bbox in chain:
        _draw_frame(grid, bbox)
    for info, color in zip(controls, colors):
        grid[info.row, info.col] = color
    return grid


def _validate_instance(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert get_bg(grid) == BG_COLOR

    frames = _extract_frames(grid)
    controls = _extract_controls(grid)
    infos = [_control_info(frames, row, col, color, grid.shape) for row, col, color in controls]

    assert len(frames) in (2, 3)
    assert {info.depth for info in infos} == set(range(1, len(frames) + 1))
    assert len({info.direction_name for info in infos}) >= 2
    assert any(info.crosses_frame for info in infos)
    assert any(not info.crosses_frame for info in infos)
    if len(frames) >= 2:
        assert any(info.differs_from_outer for info in infos)

    chain = [frame.bbox for frame in frames]
    ray_cells: set[tuple[int, int]] = set()
    control_cells = {(info.row, info.col) for info in infos}
    for info in infos:
        assert not _on_frame_border(chain, info.row, info.col)
        assert len(info.ray) == info.depth
        assert all(in_bounds(grid.shape, rr, cc) for rr, cc in info.ray)
        assert (info.row, info.col) not in ray_cells
        for cell in info.ray:
            assert cell not in ray_cells
            assert cell not in control_cells
            ray_cells.add(cell)

    out = main(grid)
    expected = _predict_output(grid)
    assert np.array_equal(out, expected)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(15, 23))
    width = int(rng.integers(15, 23))
    frame_count = int(rng.choice(np.array([2, 3], dtype=int), p=np.array([0.45, 0.55])))

    chain = _sample_chain(rng, height, width, frame_count)
    if chain is None:
        return None

    candidates = _candidate_controls(chain, (height, width))
    if any(not candidates[depth] for depth in range(1, frame_count + 1)):
        return None

    controls = _pick_controls(rng, candidates, frame_count)
    if controls is None:
        return None

    colors = _assign_colors(rng, len(controls))
    grid = _build_grid((height, width), chain, controls, colors)
    _validate_instance(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid nested-depth ray puzzle after {GEN_RETRIES} attempts")
