from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("arc_utils_for_arc_026_4", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load arc_utils from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_arc_utils = _load_arc_utils()
Color = _arc_utils.Color
bbox = _arc_utils.bbox
find_components = _arc_utils.find_components
in_bounds = _arc_utils.in_bounds

BG = Color.BLACK
FRAME = Color.GREY
RAY = Color.ORANGE
BLUE = Color.BLUE
RED = Color.RED
GREEN = Color.GREEN

NODE_ORDER = ("A", "B", "C", "D", "E")
BASE_POSITIONS = {
    "A": (0, 0),
    "B": (0, 1),
    "C": (0, 2),
    "D": (1, 0),
    "E": (1, 1),
}
GEN_ATTEMPTS = 300


def _frame_border(box):
    r0, c0, r1, c1 = map(int, box)
    return {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }


def _corner_cells(box):
    r0, c0, r1, c1 = map(int, box)
    return {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}


def _gap_direction(box, gap):
    r0, c0, r1, c1 = map(int, box)
    gap_r, gap_c = map(int, gap)
    if gap_r == r0:
        return (-1, 0), "top"
    if gap_r == r1:
        return (1, 0), "bottom"
    if gap_c == c0:
        return (0, -1), "left"
    if gap_c == c1:
        return (0, 1), "right"
    raise AssertionError("gap must lie on the frame border")


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    components = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[FRAME],
        return_format="dict",
    )
    for idx, comp in enumerate(components):
        box = tuple(map(int, comp["bbox"]))
        r0, c0, r1, c1 = box
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        assert height == width and height in (4, 5), "expected 4x4 or 5x5 square frames"

        coords = {tuple(map(int, rc)) for rc in comp["coords"]}
        comp_mask = np.zeros_like(grid, dtype=bool)
        for r, c in coords:
            comp_mask[r, c] = True
        assert bbox(comp_mask) == box
        border = _frame_border(box)
        assert coords <= border, "frame pixels must lie on the perimeter"

        gaps = list(border - coords)
        assert len(gaps) == 1, "each frame must have exactly one missing border pixel"
        gap = tuple(map(int, gaps[0]))
        assert gap not in _corner_cells(box), "gap must be a non-corner border pixel"

        interior = {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}
        for r, c in interior:
            assert int(grid[r, c]) == BG, "frame interior must be background in the input"

        direction, side = _gap_direction(box, gap)
        frames.append(
            {
                "idx": idx,
                "bbox": box,
                "coords": coords,
                "gap": gap,
                "direction": direction,
                "gap_side": side,
                "size": height,
            }
        )

    frames.sort(key=lambda frame: frame["bbox"])
    for idx, frame in enumerate(frames):
        frame["idx"] = idx
    return frames


def _trace_rays(grid: np.ndarray, frames: list[dict]) -> list[dict]:
    border_owner = {}
    for frame in frames:
        for coord in frame["coords"]:
            assert coord not in border_owner, "frame borders must not overlap"
            border_owner[coord] = frame["idx"]

    rays = []
    for frame in frames:
        gap_r, gap_c = frame["gap"]
        dr, dc = frame["direction"]
        path = []
        r, c = gap_r, gap_c

        while True:
            assert in_bounds(grid.shape, r, c), "ray path left the grid"
            assert int(grid[r, c]) == BG, "ray path must traverse background cells only"
            path.append((int(r), int(c)))

            nr, nc = int(r + dr), int(c + dc)
            assert in_bounds(grid.shape, nr, nc), "each ray must hit another frame before the edge"
            target = border_owner.get((nr, nc))
            if target is not None:
                assert target != frame["idx"], "a frame cannot target itself"
                rays.append({"src": frame["idx"], "dst": target, "path": tuple(path)})
                break

            assert int(grid[nr, nc]) == BG, "ray corridor must remain background until it hits a frame"
            r, c = nr, nc

    used = {}
    for ray_idx, ray in enumerate(rays):
        for cell in ray["path"]:
            assert cell not in used, "rays may not cross or overlap"
            used[cell] = ray_idx

    return rays


def _analyze_grid(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.size > 0
    assert int(np.max(g.shape)) <= 30, "ARC grids must stay within 30x30"
    assert set(map(int, np.unique(g))) <= {BG, FRAME}, "input should contain only black background and grey frames"

    frames = _extract_frames(g)
    assert 3 <= len(frames) <= 5, "expected between 3 and 5 frames"
    rays = _trace_rays(g, frames)
    assert len(rays) == len(frames), "each frame must emit exactly one ray"

    indegrees = [0] * len(frames)
    for ray in rays:
        indegrees[ray["dst"]] += 1
    degrees = [1 + indegrees[idx] for idx in range(len(frames))]
    return frames, rays, indegrees, degrees


def _degree_color(degree: int) -> int:
    if degree == 1:
        return BLUE
    if degree == 2:
        return RED
    return GREEN


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a ray from each one-gap frame, then fill every frame interior by its total ray connectivity.

    Input structure:
    - The grid is mostly black background.
    - Foreground objects are 3-5 grey hollow square frames, each 4x4 or 5x5.
    - Every frame is missing exactly one non-corner border pixel, which defines a unique outward-facing gap.

    Rule:
    - For each frame, locate the missing border pixel and send an orange ray outward perpendicular to that side.
    - Paint the gap cell and every background cell on the ray until the next step would enter another frame border.
    - Count each frame's total connectivity as 1 outgoing ray plus however many other rays hit its border.
    - Fill the frame interior blue for degree 1, red for degree 2, and green for degree 3 or more; keep grey borders and orange rays unchanged.

    Key insight:
    - The missing pixel is both the ray source and the direction clue, so the solver must reconstruct the entire non-crossing ray network before the frame fill colors are determined.
    """
    g = np.asarray(grid, dtype=int)
    frames, rays, _indegrees, degrees = _analyze_grid(g)

    out = g.copy()
    for ray in rays:
        for r, c in ray["path"]:
            out[r, c] = RAY

    for frame, degree in zip(frames, degrees):
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = _degree_color(degree)

    return out


def _shared_offsets(size_a: int, size_b: int) -> list[int]:
    return list(range(1, min(size_a, size_b) - 1))


def _paint_frame(grid: np.ndarray, top: int, left: int, size: int, gap: tuple[int, int]) -> None:
    bottom = int(top + size - 1)
    right = int(left + size - 1)
    for r, c in _frame_border((top, left, bottom, right)):
        if (r, c) != gap:
            grid[r, c] = FRAME


def _transform_grid(grid: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(grid, k)
    if flip:
        out = np.fliplr(out)
    return out.copy()


def _pad_with_budget(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = map(int, grid.shape)
    extra_h = int(30 - h)
    extra_w = int(30 - w)
    assert extra_h >= 0 and extra_w >= 0

    def sample_pair(extra: int) -> tuple[int, int]:
        if extra <= 0:
            return 0, 0
        if extra == 1:
            split = rng.randrange(2)
            return split, 1 - split

        upper = min(3, extra - 1)
        first = rng.randint(1, upper)
        second_upper = min(3, extra - first)
        if second_upper <= 0:
            return first, 0
        second = rng.randint(0, second_upper)
        return first, second

    top, bottom = sample_pair(extra_h)
    left, right = sample_pair(extra_w)
    return np.pad(grid, ((top, bottom), (left, right)), constant_values=BG)


def _build_base_input(rng: random.Random) -> np.ndarray:
    sizes = {node: rng.choice((4, 5)) for node in NODE_ORDER}
    gap_ab = rng.randint(2, 5)
    gap_bc = rng.randint(2, 5)
    gap_vertical = rng.randint(2, 5)

    top_row = 0
    col0 = 0
    col1 = col0 + sizes["A"] + gap_ab
    col2 = col1 + sizes["B"] + gap_bc
    bottom_row = top_row + max(sizes["A"], sizes["B"], sizes["C"]) + gap_vertical
    row_bands = {0: top_row, 1: bottom_row}
    col_bands = {0: col0, 1: col1, 2: col2}

    positions = {
        node: (row_bands[row_idx], col_bands[col_idx])
        for node, (row_idx, col_idx) in BASE_POSITIONS.items()
    }

    gap_specs = {
        "A": ("right", rng.choice(_shared_offsets(sizes["A"], sizes["B"]))),
        "B": ("bottom", rng.choice(_shared_offsets(sizes["B"], sizes["E"]))),
        "C": ("left", rng.choice(_shared_offsets(sizes["C"], sizes["B"]))),
        "D": ("top", rng.choice(_shared_offsets(sizes["D"], sizes["A"]))),
        "E": ("left", rng.choice(_shared_offsets(sizes["E"], sizes["D"]))),
    }

    gaps = {}
    for node, (side, offset) in gap_specs.items():
        top, left = positions[node]
        size = sizes[node]
        bottom = int(top + size - 1)
        right = int(left + size - 1)
        if side == "right":
            gaps[node] = (int(top + offset), right)
        elif side == "left":
            gaps[node] = (int(top + offset), left)
        elif side == "top":
            gaps[node] = (top, int(left + offset))
        else:
            gaps[node] = (bottom, int(left + offset))

    height = bottom_row + max(sizes["D"], sizes["E"])
    width = col2 + sizes["C"]
    grid = np.full((height, width), BG, dtype=int)
    for node in NODE_ORDER:
        top, left = positions[node]
        _paint_frame(grid, top, left, sizes[node], gaps[node])
    return grid


def _validate_generated(grid: np.ndarray) -> None:
    frames, rays, indegrees, degrees = _analyze_grid(grid)
    assert len(frames) == 5
    assert sorted(indegrees) == [0, 1, 1, 1, 2], "generator should realize the intended 5-node degree signature"
    assert sorted(degrees) == [1, 2, 2, 2, 3]
    assert len(set(degrees)) == 3
    assert all(len(ray["path"]) >= 2 for ray in rays)
    assert {abs(ray["path"][0][0] - ray["path"][-1][0]) == 0 for ray in rays} == {False, True}

    direction_to_colors = {}
    size_to_colors = {}
    for frame, degree in zip(frames, degrees):
        direction_to_colors.setdefault(frame["gap_side"], set()).add(_degree_color(degree))
        size_to_colors.setdefault(frame["size"], set()).add(_degree_color(degree))
    assert any(len(colors) >= 2 for colors in direction_to_colors.values()), "gap direction should not determine fill color"
    assert any(len(colors) >= 2 for colors in size_to_colors.values()), "frame size should not determine fill color"

    out = main(grid)
    assert set(map(int, np.unique(out))) >= {BG, FRAME, RAY, BLUE, RED, GREEN}
    assert not np.array_equal(out, grid)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            rng = random.Random()
            grid = _build_base_input(rng)
            grid = _transform_grid(grid, rng.randrange(4), rng.random() < 0.5)
            grid = _pad_with_budget(grid, rng)
            _validate_generated(grid)
            return grid
        except AssertionError:
            continue
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
