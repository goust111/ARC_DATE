from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("_arc_utils_shared", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load arc_utils from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC = _load_arc_utils()
Color = _ARC.Color
find_components = _ARC.find_components
bbox_h = _ARC.bbox_h
bbox_w = _ARC.bbox_w

BLACK = Color.BLACK
YELLOW = Color.YELLOW
CONTENT_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return bbox_h(bbox) * bbox_w(bbox)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int = YELLOW) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _is_hollow_rectangle(comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = bbox_h(comp["bbox"])
    w = bbox_w(comp["bbox"])
    if h < 3 or w < 3:
        return False
    expected_area = 2 * (h + w) - 4
    if int(comp["area"]) != expected_area:
        return False

    mask = np.zeros((h, w), dtype=bool)
    coords = np.asarray(comp["coords"], dtype=int)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(mask, border)


def _strictly_inside(bbox: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = map(int, bbox)
    return r0 < r < r1 and c0 < c < c1


def _bbox_disjoint(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def _bbox_strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and inner[2] < outer[2] and inner[3] < outer[3]


def _extract_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    comps = find_components(
        grid,
        bg=BLACK,
        connectivity=4,
        monochromatic=True,
        colors=[YELLOW],
        return_format="dict",
    )
    frames = []
    for comp in comps:
        assert int(comp["color"]) == YELLOW, "only yellow should define frames"
        assert _is_hollow_rectangle(comp), "yellow components must be 1-cell-thick hollow rectangles"
        frames.append(tuple(map(int, comp["bbox"])))

    assert frames, "expected at least one yellow hollow rectangle"
    for i, box_a in enumerate(frames):
        for box_b in frames[i + 1 :]:
            assert _bbox_disjoint(box_a, box_b) or _bbox_strictly_contains(box_a, box_b) or _bbox_strictly_contains(box_b, box_a), (
                "frames must be disjoint or strictly nested"
            )
    return frames


def _depth_maps(grid: np.ndarray, frames: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    h, w = grid.shape
    depth = np.zeros((h, w), dtype=int)
    innermost = np.full((h, w), -1, dtype=int)
    frame_areas = [_bbox_area(box) for box in frames]

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) == YELLOW:
                continue
            containing = [idx for idx, box in enumerate(frames) if _strictly_inside(box, r, c)]
            depth[r, c] = len(containing)
            if containing:
                innermost[r, c] = min(containing, key=lambda idx: frame_areas[idx])
    return depth, innermost


def _move_target(bbox: tuple[int, int, int, int], r: int, c: int) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    return r0 + r1 - r, c0 + c1 - c


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Relocate odd-depth ring content pixels by 180-degree point symmetry around their innermost containing hollow rectangle.

    Input structure:
    - The grid has a black background, several yellow 1-cell-thick hollow rectangles, and scattered non-yellow content pixels.
    - Rectangles are pairwise disjoint or strictly nested, so every non-frame cell has a well-defined containment depth.
    - Content may appear outside all rectangles, in odd-depth rings, and in even-depth nested regions.

    Rule:
    - Keep every yellow rectangle unchanged and compute, for each non-frame cell, how many rectangle interiors strictly contain it.
    - If a content pixel lies at odd depth, move it to the 180-degree point-symmetric cell with respect to the bbox center of its innermost containing rectangle.
    - If a content pixel lies at even depth, including depth 0 outside all rectangles, leave it in place; moved pixels vacate their original cells to black.

    Key insight:
    - The real switch is containment-depth parity: only odd-depth ring content moves, and the reference center is the innermost containing frame rather than the whole scene.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"

    frames = _extract_frames(grid)
    depth, innermost = _depth_maps(grid, frames)
    out = grid.copy()
    moving: list[tuple[int, int, int, int, int]] = []

    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BLACK, YELLOW):
                continue
            cell_depth = int(depth[r, c])
            if cell_depth % 2 == 0:
                continue
            frame_idx = int(innermost[r, c])
            assert frame_idx >= 0, "odd-depth content must lie inside some frame"
            rr, cc = _move_target(frames[frame_idx], r, c)
            assert 0 <= rr < h and 0 <= cc < w
            assert int(depth[rr, cc]) == cell_depth, "generator guarantees odd-depth targets stay in the same ring"
            assert int(innermost[rr, cc]) == frame_idx, "movement reference must be the same innermost frame"
            target_color = int(grid[rr, cc])
            assert target_color != YELLOW, "odd-depth movement cannot land on a frame"
            if target_color != BLACK:
                assert int(depth[rr, cc]) % 2 == 1, "odd-depth movement cannot overwrite static even-depth content"
            moving.append((r, c, rr, cc, color))

    for r, c, _rr, _cc, _color in moving:
        out[r, c] = BLACK
    for _r, _c, rr, cc, color in moving:
        out[rr, cc] = color
    return out


def _expand_bbox(bbox: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    return bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not _bbox_disjoint(a, b)


def _sample_system(depth: int, h: int, w: int, blocked: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]] | None:
    for _ in range(250):
        inner_h = random.randint(3, 6)
        inner_w = random.randint(3, 6)
        row_steps = [random.randint(2, 3) for _ in range(depth - 1)]
        col_steps = [random.randint(2, 3) for _ in range(depth - 1)]
        outer_h = inner_h + 2 * sum(row_steps)
        outer_w = inner_w + 2 * sum(col_steps)
        if outer_h > h - 2 or outer_w > w - 2:
            continue

        r0 = random.randint(1, h - outer_h - 1)
        c0 = random.randint(1, w - outer_w - 1)
        outer = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
        if any(_overlap(_expand_bbox(outer, 1), _expand_bbox(box, 1)) for box in blocked):
            continue

        frames = []
        acc_r = 0
        acc_c = 0
        for level in range(depth):
            frames.append((r0 + acc_r, c0 + acc_c, r0 + outer_h - 1 - acc_r, c0 + outer_w - 1 - acc_c))
            if level < depth - 1:
                acc_r += row_steps[level]
                acc_c += col_steps[level]
        return frames
    return None


def _orbit_pairs_for_group(
    grid: np.ndarray,
    depth: np.ndarray,
    innermost: np.ndarray,
    bbox: tuple[int, int, int, int],
    frame_idx: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pairs: dict[tuple[tuple[int, int], tuple[int, int]], None] = {}
    coords = np.argwhere((grid == BLACK) & (innermost == frame_idx) & (depth % 2 == 1))
    for r, c in coords:
        rr, cc = _move_target(bbox, int(r), int(c))
        if (rr, cc) == (int(r), int(c)):
            continue
        if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and int(innermost[rr, cc]) == frame_idx and int(depth[rr, cc]) == int(depth[r, c]):
            key = tuple(sorted(((int(r), int(c)), (int(rr), int(cc)))))
            pairs[key] = None
    return list(pairs)


def _paint_random_points(grid: np.ndarray, coords: list[tuple[int, int]], count: int) -> int:
    if not coords or count <= 0:
        return 0
    random.shuffle(coords)
    painted = 0
    for r, c in coords:
        if int(grid[r, c]) != BLACK:
            continue
        grid[r, c] = random.choice(CONTENT_COLORS)
        painted += 1
        if painted == count:
            break
    return painted


def generate_input() -> np.ndarray:
    for _ in range(400):
        h = random.randint(18, 30)
        w = random.randint(18, 30)
        grid = np.full((h, w), BLACK, dtype=int)

        num_systems = random.choice([1, 2, 2, 3])
        desired_depths = [random.choice([2, 3])]
        while len(desired_depths) < num_systems:
            desired_depths.append(random.choice([1, 1, 2, 3]))

        systems: list[list[tuple[int, int, int, int]]] = []
        top_boxes: list[tuple[int, int, int, int]] = []
        ok = True
        for depth_value in desired_depths:
            system = _sample_system(depth_value, h, w, top_boxes)
            if system is None:
                ok = False
                break
            systems.append(system)
            top_boxes.append(system[0])

        if not ok:
            continue

        frames: list[tuple[int, int, int, int]] = []
        system_frame_indices: list[list[int]] = []
        for system in systems:
            idxs = []
            for box in system:
                idxs.append(len(frames))
                frames.append(box)
                _draw_frame(grid, box, YELLOW)
            system_frame_indices.append(idxs)

        depth, innermost = _depth_maps(grid, frames)
        moved_pixels = 0
        even_pixels = 0
        depth3_pixels = 0

        for frame_indices in system_frame_indices:
            for local_level, frame_idx in enumerate(frame_indices):
                group_depth = local_level + 1
                if group_depth % 2 == 1:
                    pairs = _orbit_pairs_for_group(grid, depth, innermost, frames[frame_idx], frame_idx)
                    if not pairs:
                        ok = False
                        break
                    random.shuffle(pairs)
                    take = random.randint(1, min(len(pairs), 4))
                    for pair in pairs[:take]:
                        color_a = random.choice(CONTENT_COLORS)
                        if random.random() < 0.72:
                            src = random.choice(pair)
                            grid[src] = color_a
                            moved_pixels += 1
                            if group_depth == 3:
                                depth3_pixels += 1
                        else:
                            other_choices = [c for c in CONTENT_COLORS if c != color_a]
                            color_b = random.choice(other_choices) if other_choices else color_a
                            (r1, c1), (r2, c2) = pair
                            grid[r1, c1] = color_a
                            grid[r2, c2] = color_b
                            moved_pixels += 2
                            if group_depth == 3:
                                depth3_pixels += 2
                else:
                    candidates = [(int(r), int(c)) for r, c in np.argwhere((grid == BLACK) & (innermost == frame_idx) & (depth == group_depth))]
                    target_count = random.randint(1, min(4, len(candidates))) if candidates else 0
                    even_pixels += _paint_random_points(grid, candidates, target_count)
            if not ok:
                break
        if not ok:
            continue

        outside_candidates = [(int(r), int(c)) for r, c in np.argwhere((grid == BLACK) & (depth == 0))]
        outside_count = random.randint(4, min(12, len(outside_candidates))) if outside_candidates else 0
        outside_pixels = _paint_random_points(grid, outside_candidates, outside_count)

        if moved_pixels == 0 or outside_pixels == 0 or even_pixels == 0:
            continue
        if desired_depths[0] == 3 and depth3_pixels == 0:
            continue

        out = main(grid)
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        if np.array_equal(grid, out):
            continue
        return grid

    raise RuntimeError("generate_input failed to build a valid nondegenerate sample")
