from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

FRAME_COUNT_CHOICES = (3, 4)
GRID_RANGE_BY_COUNT = {
    3: ((15, 23), (15, 23)),
    4: ((18, 27), (18, 27)),
}
OUTER_SIZE_OPTIONS = tuple(
    (h, w)
    for h in range(5, 10)
    for w in range(5, 10)
    if h >= 5 and w >= 5
)
EDGE_MARGIN = 1
FRAME_GAP = 1
MAX_GENERATION_ATTEMPTS = 500


def _connected_components_same_color(grid: np.ndarray, bg: int = BG) -> list[dict]:
    height, width = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    components: list[dict] = []

    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color == bg or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            rows = arr[:, 0]
            cols = arr[:, 1]
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (
                        int(rows.min()),
                        int(cols.min()),
                        int(rows.max()),
                        int(cols.max()),
                    ),
                }
            )
    return components


def _is_hollow_rectangle_component(grid: np.ndarray, comp: dict) -> bool:
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 5 or width < 5:
        return False

    expected_area = 2 * (height + width) - 4
    if int(comp["area"]) != expected_area:
        return False

    coords = comp["coords"]
    if not np.all(
        (coords[:, 0] == r0)
        | (coords[:, 0] == r1)
        | (coords[:, 1] == c0)
        | (coords[:, 1] == c1)
    ):
        return False

    if not np.all(grid[r0, c0 : c1 + 1] == color):
        return False
    if not np.all(grid[r1, c0 : c1 + 1] == color):
        return False
    if not np.all(grid[r0 : r1 + 1, c0] == color):
        return False
    if not np.all(grid[r0 : r1 + 1, c1] == color):
        return False
    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == color):
        return False
    return True


def _classify_components(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    frames: list[dict] = []
    markers: list[dict] = []
    for comp in _connected_components_same_color(grid, bg=BG):
        if comp["area"] == 1:
            r, c = map(int, comp["coords"][0])
            markers.append({"color": int(comp["color"]), "row": r, "col": c})
        else:
            assert _is_hollow_rectangle_component(grid, comp), "unexpected foreground component"
            frames.append(comp)
    return frames, markers


def _interior_distances(frame_bbox: tuple[int, int, int, int], row: int, col: int) -> dict:
    r0, c0, r1, c1 = frame_bbox
    top = row - (r0 + 1)
    bottom = (r1 - 1) - row
    left = col - (c0 + 1)
    right = (c1 - 1) - col
    return {"top": top, "bottom": bottom, "left": left, "right": right}


def _stripe_axis(frame_bbox: tuple[int, int, int, int], row: int, col: int) -> str:
    distances = _interior_distances(frame_bbox, row, col)
    nearest_horizontal = min(distances["top"], distances["bottom"])
    nearest_vertical = min(distances["left"], distances["right"])
    assert nearest_horizontal != nearest_vertical, "marker direction must be unambiguous"
    return "vertical" if nearest_horizontal < nearest_vertical else "horizontal"


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each hollow frame interior with marker/grey stripes whose axis is inferred from the marker's eccentricity.

    Input structure:
    - A black background contains 3-4 non-overlapping 1-pixel-thick hollow rectangular frames.
    - Each frame is monochromatic, and its interior contains exactly one single-pixel marker of a different color.
    - The marker sits strictly inside its frame and is closer to a horizontal inner wall or to a vertical inner wall, but not equally close to both orientations.

    Rule:
    - For each frame, compare the marker's nearest horizontal inner-wall distance against its nearest vertical inner-wall distance.
    - If the marker is closer to a top/bottom inner wall, fill the interior by vertical stripes; if it is closer to a left/right inner wall, fill by horizontal stripes.
    - Use the marker's column or row as stripe offset 0: even offsets are the marker color, odd offsets are grey (5), and only interior black cells are painted.

    Key insight:
    - The marker's position does not indicate a cell to copy; its eccentricity inside the frame encodes the stripe axis, while its color supplies the even stripes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames, markers = _classify_components(g)
    assert 3 <= len(frames) <= 4, "expected 3-4 hollow frames"
    assert len(markers) == len(frames), "expected one marker per frame"

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inside = [
            marker
            for marker in markers
            if r0 < marker["row"] < r1 and c0 < marker["col"] < c1
        ]
        assert len(inside) == 1, "each frame must contain exactly one marker"
        marker = inside[0]

        axis = _stripe_axis(frame["bbox"], marker["row"], marker["col"])
        ir0, ic0 = r0 + 1, c0 + 1
        ir1, ic1 = r1 - 1, c1 - 1

        if axis == "vertical":
            for c in range(ic0, ic1 + 1):
                fill_color = int(marker["color"]) if abs(c - marker["col"]) % 2 == 0 else GREY
                mask = g[ir0 : ir1 + 1, c] == BG
                out[ir0 : ir1 + 1, c][mask] = fill_color
        else:
            for r in range(ir0, ir1 + 1):
                fill_color = int(marker["color"]) if abs(r - marker["row"]) % 2 == 0 else GREY
                mask = g[r, ic0 : ic1 + 1] == BG
                out[r, ic0 : ic1 + 1][mask] = fill_color

    return out


def _marker_candidates(interior_h: int, interior_w: int, side: str) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for ir in range(interior_h):
        for ic in range(interior_w):
            distances = {
                "top": ir,
                "bottom": interior_h - 1 - ir,
                "left": ic,
                "right": interior_w - 1 - ic,
            }
            nearest = min(distances.values())
            if distances[side] != nearest:
                continue
            if sum(value == nearest for value in distances.values()) != 1:
                continue
            candidates.append((ir, ic))
    return candidates


def _stripe_axis_from_side(side: str) -> str:
    return "vertical" if side in {"top", "bottom"} else "horizontal"


def _render_input(
    shape: tuple[int, int],
    placements: list[dict],
    frame_colors: list[int],
    marker_colors: list[int],
) -> tuple[np.ndarray, list[dict]]:
    grid = np.zeros(shape, dtype=int)
    specs: list[dict] = []
    for placement, frame_color, marker_color in zip(placements, frame_colors, marker_colors):
        r0 = int(placement["r0"])
        c0 = int(placement["c0"])
        outer_h = int(placement["outer_h"])
        outer_w = int(placement["outer_w"])
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1

        grid[r0, c0 : c1 + 1] = frame_color
        grid[r1, c0 : c1 + 1] = frame_color
        grid[r0 : r1 + 1, c0] = frame_color
        grid[r0 : r1 + 1, c1] = frame_color

        ir, ic = placement["marker_pos"]
        mr = r0 + 1 + ir
        mc = c0 + 1 + ic
        grid[mr, mc] = marker_color

        specs.append(
            {
                "frame_bbox": (r0, c0, r1, c1),
                "frame_color": int(frame_color),
                "marker_color": int(marker_color),
                "marker_pos": (mr, mc),
                "side": str(placement["side"]),
            }
        )
    return grid, specs


def _expected_from_specs(shape: tuple[int, int], specs: list[dict], input_grid: np.ndarray) -> np.ndarray:
    out = input_grid.copy()
    for spec in specs:
        r0, c0, r1, c1 = spec["frame_bbox"]
        mr, mc = spec["marker_pos"]
        axis = _stripe_axis_from_side(spec["side"])

        if axis == "vertical":
            for c in range(c0 + 1, c1):
                fill_color = spec["marker_color"] if abs(c - mc) % 2 == 0 else GREY
                mask = input_grid[r0 + 1 : r1, c] == BG
                out[r0 + 1 : r1, c][mask] = fill_color
        else:
            for r in range(r0 + 1, r1):
                fill_color = spec["marker_color"] if abs(r - mr) % 2 == 0 else GREY
                mask = input_grid[r, c0 + 1 : c1] == BG
                out[r, c0 + 1 : c1][mask] = fill_color
    return out


def _wrong_parallel_output(shape: tuple[int, int], specs: list[dict], input_grid: np.ndarray) -> np.ndarray:
    out = input_grid.copy()
    for spec in specs:
        r0, c0, r1, c1 = spec["frame_bbox"]
        mr, mc = spec["marker_pos"]
        axis = "horizontal" if _stripe_axis_from_side(spec["side"]) == "vertical" else "vertical"

        if axis == "vertical":
            for c in range(c0 + 1, c1):
                fill_color = spec["marker_color"] if abs(c - mc) % 2 == 0 else GREY
                mask = input_grid[r0 + 1 : r1, c] == BG
                out[r0 + 1 : r1, c][mask] = fill_color
        else:
            for r in range(r0 + 1, r1):
                fill_color = spec["marker_color"] if abs(r - mr) % 2 == 0 else GREY
                mask = input_grid[r, c0 + 1 : c1] == BG
                out[r, c0 + 1 : c1][mask] = fill_color
    return out


def _wrong_edge_anchored_output(shape: tuple[int, int], specs: list[dict], input_grid: np.ndarray) -> np.ndarray:
    out = input_grid.copy()
    for spec in specs:
        r0, c0, r1, c1 = spec["frame_bbox"]
        axis = _stripe_axis_from_side(spec["side"])

        if axis == "vertical":
            for c in range(c0 + 1, c1):
                fill_color = spec["marker_color"] if (c - (c0 + 1)) % 2 == 0 else GREY
                mask = input_grid[r0 + 1 : r1, c] == BG
                out[r0 + 1 : r1, c][mask] = fill_color
        else:
            for r in range(r0 + 1, r1):
                fill_color = spec["marker_color"] if (r - (r0 + 1)) % 2 == 0 else GREY
                mask = input_grid[r, c0 + 1 : c1] == BG
                out[r, c0 + 1 : c1][mask] = fill_color
    return out


def _side_plan(n_frames: int, rng: np.random.Generator) -> list[str]:
    sides = [
        str(rng.choice(("top", "bottom"))),
        str(rng.choice(("left", "right"))),
    ]
    all_sides = ("top", "bottom", "left", "right")
    while len(sides) < n_frames:
        sides.append(str(rng.choice(all_sides)))
    rng.shuffle(sides)
    return sides


def _sample_placements(shape: tuple[int, int], sides: list[str], rng: np.random.Generator) -> list[dict] | None:
    height, width = shape
    occupancy = np.zeros(shape, dtype=bool)
    placements: list[dict] = []

    sizes = list(OUTER_SIZE_OPTIONS)
    rng.shuffle(sizes)
    selected_sizes = sizes[: len(sides)]
    order = sorted(
        range(len(selected_sizes)),
        key=lambda idx: selected_sizes[idx][0] * selected_sizes[idx][1],
        reverse=True,
    )

    slots: list[dict | None] = [None] * len(sides)
    for idx in order:
        outer_h, outer_w = selected_sizes[idx]
        placed = False
        for _ in range(150):
            max_r0 = height - outer_h - EDGE_MARGIN
            max_c0 = width - outer_w - EDGE_MARGIN
            if max_r0 < EDGE_MARGIN or max_c0 < EDGE_MARGIN:
                return None
            r0 = int(rng.integers(EDGE_MARGIN, max_r0 + 1))
            c0 = int(rng.integers(EDGE_MARGIN, max_c0 + 1))
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1

            pad_r0 = max(0, r0 - FRAME_GAP)
            pad_c0 = max(0, c0 - FRAME_GAP)
            pad_r1 = min(height - 1, r1 + FRAME_GAP)
            pad_c1 = min(width - 1, c1 + FRAME_GAP)
            if np.any(occupancy[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1]):
                continue

            candidates = _marker_candidates(outer_h - 2, outer_w - 2, sides[idx])
            if not candidates:
                continue
            marker_pos = candidates[int(rng.integers(len(candidates)))]
            occupancy[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1] = True
            slots[idx] = {
                "r0": r0,
                "c0": c0,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "marker_pos": marker_pos,
                "side": sides[idx],
            }
            placed = True
            break
        if not placed:
            return None

    placements = [slot for slot in slots if slot is not None]
    return placements if len(placements) == len(sides) else None


def _spec_semantics_ok(specs: list[dict]) -> bool:
    axes = [_stripe_axis_from_side(spec["side"]) for spec in specs]
    if "vertical" not in axes or "horizontal" not in axes:
        return False

    has_odd_vertical_anchor = False
    has_odd_horizontal_anchor = False
    has_nonadjacent_marker = False

    for spec in specs:
        r0, c0, r1, c1 = spec["frame_bbox"]
        mr, mc = spec["marker_pos"]
        distances = _interior_distances(spec["frame_bbox"], mr, mc)
        nearest = min(distances.values())
        if sum(value == nearest for value in distances.values()) != 1:
            return False
        if min(distances["top"], distances["bottom"]) == min(distances["left"], distances["right"]):
            return False
        if nearest > 0:
            has_nonadjacent_marker = True

        axis = _stripe_axis_from_side(spec["side"])
        if axis == "vertical" and (mc - (c0 + 1)) % 2 == 1:
            has_odd_vertical_anchor = True
        if axis == "horizontal" and (mr - (r0 + 1)) % 2 == 1:
            has_odd_horizontal_anchor = True

    return has_nonadjacent_marker and has_odd_vertical_anchor and has_odd_horizontal_anchor


def _input_structure_ok(grid: np.ndarray, specs: list[dict]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    colors = set(map(int, np.unique(grid)))
    if GREY in colors:
        return False

    frame_colors = {spec["frame_color"] for spec in specs}
    marker_colors = {spec["marker_color"] for spec in specs}
    if len(frame_colors) != len(specs) or len(marker_colors) != len(specs):
        return False
    if frame_colors & marker_colors:
        return False

    for spec in specs:
        r0, c0, r1, c1 = spec["frame_bbox"]
        if not np.all(grid[r0, c0 : c1 + 1] == spec["frame_color"]):
            return False
        if not np.all(grid[r1, c0 : c1 + 1] == spec["frame_color"]):
            return False
        if not np.all(grid[r0 : r1 + 1, c0] == spec["frame_color"]):
            return False
        if not np.all(grid[r0 : r1 + 1, c1] == spec["frame_color"]):
            return False
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.shape[0] < 3 or interior.shape[1] < 3:
            return False
        mr, mc = spec["marker_pos"]
        if int(grid[mr, mc]) != spec["marker_color"]:
            return False
        interior_bg = interior.copy()
        interior_bg[mr - (r0 + 1), mc - (c0 + 1)] = BG
        if np.any(interior_bg != BG):
            return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_frames = int(rng.choice(FRAME_COUNT_CHOICES))
    h_range, w_range = GRID_RANGE_BY_COUNT[n_frames]
    shape = (
        int(rng.integers(h_range[0], h_range[1])),
        int(rng.integers(w_range[0], w_range[1])),
    )

    sides = _side_plan(n_frames, rng)
    placements = _sample_placements(shape, sides, rng)
    if placements is None:
        return None

    colors = list(rng.permutation(PALETTE))
    frame_colors = [int(color) for color in colors[:n_frames]]
    marker_colors = [int(color) for color in colors[n_frames : 2 * n_frames]]

    grid, specs = _render_input(shape, placements, frame_colors, marker_colors)
    if not _spec_semantics_ok(specs):
        return None
    if not _input_structure_ok(grid, specs):
        return None

    expected = _expected_from_specs(shape, specs, grid)
    wrong_parallel = _wrong_parallel_output(shape, specs, grid)
    wrong_edge_anchored = _wrong_edge_anchored_output(shape, specs, grid)
    if np.array_equal(expected, wrong_parallel):
        return None
    if np.array_equal(expected, wrong_edge_anchored):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid eccentric-frame stripe input")
