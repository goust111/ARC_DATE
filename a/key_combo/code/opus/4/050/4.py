from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("arc_utils_shared_for_4_050_4", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load arc_utils from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC = _load_arc_utils()
Color = _ARC.Color
bbox_coords = _ARC.bbox_coords
components_mask = _ARC.components_mask
get_bg = _ARC.get_bg


GREY = Color.GREY
BG = Color.BLACK
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
FRAME_SIZES = [(3, 5), (5, 5)]
MIN_MARKER_GAP = 5
MAX_ATTEMPTS = 400


def _rect_border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames = []
    for coords in components_mask(grid == GREY, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = patch.shape
        assert h >= 3 and w >= 3, "grey objects must be hollow frames"
        border = _rect_border_mask(h, w)
        assert np.all(patch[border] == GREY), "frame border must be fully grey"
        assert not np.any(patch[~border] == GREY), "frame interior must not contain grey"

        seed_coords = np.argwhere((patch != bg) & (patch != GREY))
        assert seed_coords.shape[0] <= 1, "each frame may contain at most one seed"

        frame = {"bbox": (int(r0), int(c0), int(r1), int(c1))}
        if seed_coords.size == 0:
            frame["seed"] = None
        else:
            sr, sc = map(int, seed_coords[0])
            frame["seed"] = (int(r0 + sr), int(c0 + sc), int(patch[sr, sc]))
        frames.append(frame)
    assert frames, "expected at least one grey frame"
    return frames


def _find_bottom_markers(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    row = grid.shape[0] - 1
    markers = []
    for c in range(grid.shape[1]):
        color = int(grid[row, c])
        if color == bg:
            continue
        assert color != GREY, "bottom-row markers cannot be grey"
        markers.append((c, color))
    assert 2 <= len(markers) <= 3, "expected 2-3 bottom-row markers"
    cols = [c for c, _color in markers]
    colors = [color for _c, color in markers]
    assert len(set(cols)) == len(cols), "marker columns must be unique"
    assert len(set(colors)) == len(colors), "marker colors must be distinct"
    return markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each bottom-row marker upward through grey frames, switching color at seeds and stopping inside empty frames.

    Input structure:
    - The grid has a black background, 2-3 colored singleton markers in the bottom row, and several one-pixel-thick grey hollow rectangular frames.
    - Some frames contain one non-grey seed pixel inside; other frames are empty.
    - Each active marker column passes through at least one frame, while unmarked columns contain no dynamic propagation.

    Rule:
    - For each bottom-row marker, scan upward along its column starting from the row above the marker with the marker color as the current color.
    - Paint background cells with the current color, leave grey frame borders unchanged, and whenever the scan hits a non-grey seed pixel keep that pixel and update the current color to the seed color.
    - If the scan reaches the bottom border of an empty frame, fill that frame's interior cells in the same column with the current color and then stop; do not continue above that frame's top border.

    Key insight:
    - The grey frames are not mere containers: seeded frames are color-changing relay stations, while empty frames are blockers that terminate the vertical band.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    frames = _find_frames(g, bg)
    markers = _find_bottom_markers(g, bg)

    empty_frame_bottom = {}
    for frame in frames:
        if frame["seed"] is not None:
            continue
        r0, c0, r1, c1 = frame["bbox"]
        for c in range(c0, c1 + 1):
            empty_frame_bottom[(r1, c)] = frame

    out = g.copy()
    h, _w = g.shape
    for c, marker_color in markers:
        current = int(marker_color)
        for r in range(h - 2, -1, -1):
            empty_frame = empty_frame_bottom.get((r, c))
            if empty_frame is not None:
                r0, _c0, r1, _c1 = empty_frame["bbox"]
                out[r, c] = GREY
                for rr in range(r1 - 1, r0, -1):
                    if g[rr, c] == bg:
                        out[rr, c] = current
                break

            cell = int(g[r, c])
            if cell == bg:
                out[r, c] = current
            elif cell == GREY:
                continue
            else:
                current = cell

    return out


def _pick_marker_columns(width: int, count: int, margin: int) -> list[int] | None:
    candidates = list(range(margin, width - margin))
    if len(candidates) < count:
        return None
    for _ in range(120):
        cols = sorted(random.sample(candidates, count))
        if all(cols[i + 1] - cols[i] >= MIN_MARKER_GAP for i in range(len(cols) - 1)):
            return cols
    return None


def _plan_frames(marker_count: int) -> tuple[list[dict], int, int]:
    multi_idx = random.randrange(marker_count)
    block_choices = [idx for idx in range(marker_count) if idx != multi_idx]
    block_idx = random.choice(block_choices)
    other_indices = [idx for idx in range(marker_count) if idx not in {multi_idx, block_idx}]

    plan = [
        {"marker_idx": multi_idx, "kind": "seeded"},
        {"marker_idx": block_idx, "kind": "empty"},
        {"marker_idx": multi_idx, "kind": "seeded"},
    ]
    for idx in other_indices:
        plan.append({"marker_idx": idx, "kind": random.choice(["seeded", "seeded", "empty"])})
    if len(plan) < 5 and random.random() < 0.55:
        plan.append({"marker_idx": random.choice([multi_idx, *other_indices] or [multi_idx]), "kind": "seeded"})
    if len(plan) < 5 and random.random() < 0.2:
        plan.append({"marker_idx": multi_idx, "kind": "seeded"})

    random.shuffle(plan)
    return plan, multi_idx, block_idx


def _assign_vertical_positions(plan: list[dict]) -> tuple[int, list[tuple[int, int]], list[int]] | None:
    sizes = [random.choice(FRAME_SIZES) for _ in plan]
    heights = [h for h, _w in sizes]
    gaps = [random.randint(1, 3) for _ in range(max(0, len(plan) - 1))]
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)

    total_height = top_margin + bottom_margin + 1 + sum(heights) + sum(gaps)
    if total_height > 30:
        return None

    positions = []
    current_bottom = total_height - 2 - bottom_margin
    for idx, (h, _w) in enumerate(sizes):
        r1 = current_bottom
        r0 = r1 - h + 1
        positions.append((r0, r1))
        if idx < len(gaps):
            current_bottom = r0 - 1 - gaps[idx]

    assert positions[-1][0] >= top_margin
    return total_height, positions, [w for _h, w in sizes]


def _build_attempt() -> tuple[np.ndarray, dict] | None:
    marker_count = random.choice([2, 3])
    plan, multi_idx, block_idx = _plan_frames(marker_count)
    vertical = _assign_vertical_positions(plan)
    if vertical is None:
        return None
    height, positions, widths = vertical
    margin = max(widths) // 2

    min_width = max(15, 2 * margin + 1 + (marker_count - 1) * MIN_MARKER_GAP + 2)
    max_width = min(20, min_width + 4)
    if min_width > max_width:
        return None
    width = random.randint(min_width, max_width)

    marker_cols = _pick_marker_columns(width, marker_count, margin)
    if marker_cols is None:
        return None

    seeded_count = sum(spec["kind"] == "seeded" for spec in plan)
    marker_colors = random.sample(PALETTE, marker_count)
    remaining = [color for color in PALETTE if color not in marker_colors]
    if len(remaining) < seeded_count:
        return None
    seed_colors = iter(random.sample(remaining, seeded_count))

    grid = np.zeros((height, width), dtype=int)
    markers = []
    for col, color in zip(marker_cols, marker_colors):
        grid[height - 1, col] = color
        markers.append({"col": int(col), "color": int(color)})

    frames = []
    for spec, (r0, r1), frame_w in zip(plan, positions, widths):
        marker_col = marker_cols[spec["marker_idx"]]
        c0 = marker_col - frame_w // 2
        c1 = marker_col + frame_w // 2
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY

        frame = {
            "kind": spec["kind"],
            "marker_idx": int(spec["marker_idx"]),
            "marker_col": int(marker_col),
            "r0": int(r0),
            "c0": int(c0),
            "r1": int(r1),
            "c1": int(c1),
        }
        if spec["kind"] == "seeded":
            seed_color = int(next(seed_colors))
            seed_r = (r0 + r1) // 2
            grid[seed_r, marker_col] = seed_color
            frame["seed_row"] = int(seed_r)
            frame["seed_col"] = int(marker_col)
            frame["seed_color"] = seed_color
        frames.append(frame)

    meta = {
        "grid": grid,
        "markers": markers,
        "frames": frames,
        "multi_idx": int(multi_idx),
        "block_idx": int(block_idx),
    }
    return grid, meta


def _simulate_from_meta(meta: dict) -> np.ndarray:
    grid = np.asarray(meta["grid"], dtype=int)
    out = grid.copy()
    h, _w = grid.shape

    for marker in meta["markers"]:
        col = int(marker["col"])
        current = int(marker["color"])
        frames = sorted(
            [frame for frame in meta["frames"] if frame["marker_col"] == col],
            key=lambda frame: frame["r1"],
            reverse=True,
        )
        cursor = h - 2
        for frame in frames:
            for r in range(cursor, frame["r1"], -1):
                if out[r, col] == BG:
                    out[r, col] = current
            if frame["kind"] == "empty":
                for r in range(frame["r1"] - 1, frame["r0"], -1):
                    out[r, col] = current
                cursor = frame["r0"] - 1
                break

            seed_row = int(frame["seed_row"])
            for r in range(frame["r1"] - 1, seed_row, -1):
                out[r, col] = current
            current = int(frame["seed_color"])
            for r in range(seed_row - 1, frame["r0"], -1):
                out[r, col] = current
            cursor = frame["r0"] - 1
        else:
            for r in range(cursor, -1, -1):
                if out[r, col] == BG:
                    out[r, col] = current
            continue

    return out


def _validate_attempt(grid: np.ndarray, meta: dict) -> bool:
    assert np.array_equal(grid, meta["grid"])
    assert grid.ndim == 2 and max(grid.shape) <= 30

    markers = meta["markers"]
    frames = meta["frames"]
    marker_cols = {marker["col"] for marker in markers}
    marker_colors = {marker["color"] for marker in markers}
    seeded_frames = [frame for frame in frames if frame["kind"] == "seeded"]
    empty_frames = [frame for frame in frames if frame["kind"] == "empty"]

    assert 2 <= len(markers) <= 3
    assert 3 <= len(frames) <= 5
    assert len(seeded_frames) >= 2
    assert len(empty_frames) >= 1
    assert len(marker_cols) == len(markers)
    assert len(marker_colors) == len(markers)

    for frame in frames:
        assert frame["marker_col"] in marker_cols
        patch = grid[frame["r0"] : frame["r1"] + 1, frame["c0"] : frame["c1"] + 1]
        border = _rect_border_mask(patch.shape[0], patch.shape[1])
        assert np.all(patch[border] == GREY)
        covered_cols = sorted(col for col in marker_cols if frame["c0"] <= col <= frame["c1"])
        assert covered_cols == [frame["marker_col"]], "each frame should visibly belong to one marker column"
        if frame["kind"] == "empty":
            assert np.all(patch[~border] == BG)
        else:
            seed_cells = np.argwhere((patch != BG) & (patch != GREY))
            assert seed_cells.shape == (1, 2)

    for marker in markers:
        touched = [frame for frame in frames if frame["marker_col"] == marker["col"]]
        assert touched, "each marker column must pass through at least one frame"

    multi_col = markers[meta["multi_idx"]]["col"]
    block_col = markers[meta["block_idx"]]["col"]
    assert sum(frame["kind"] == "seeded" and frame["marker_col"] == multi_col for frame in frames) >= 2
    assert any(frame["kind"] == "empty" and frame["marker_col"] == block_col for frame in frames)

    output = main(grid)
    expected = _simulate_from_meta(meta)
    assert np.array_equal(output, expected), "main() disagrees with metadata simulator"

    untouched_cols = [c for c in range(grid.shape[1]) if c not in marker_cols]
    if untouched_cols:
        assert np.array_equal(output[:, untouched_cols], grid[:, untouched_cols]), "unmarked columns must stay unchanged"

    multi_frames = sorted(
        [frame for frame in seeded_frames if frame["marker_col"] == multi_col],
        key=lambda frame: frame["r1"],
        reverse=True,
    )
    top_seed = multi_frames[-1]
    assert top_seed["r0"] > 0, "need visible space above a recolor station"
    assert output[top_seed["r0"] - 1, multi_col] == top_seed["seed_color"], "top seed must affect the band above it"

    empty_frame = next(frame for frame in empty_frames if frame["marker_col"] == block_col)
    if empty_frame["r0"] > 0:
        assert np.all(output[: empty_frame["r0"], block_col] == grid[: empty_frame["r0"], block_col]), (
            "empty frame must block all propagation above its top border"
        )

    assert np.count_nonzero(output != grid) >= len(markers) * 3, "output must visibly differ from input"
    changed_columns = {c for _r, c in np.argwhere(output != grid)}
    assert marker_cols <= changed_columns, "every marker column should change"
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _build_attempt()
        if built is None:
            continue
        grid, meta = built
        if _validate_attempt(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
