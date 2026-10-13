from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, paste, random_free_location


BG = Color.BLACK
PATTERN = Color.GREY
SMALL_BORDER = Color.GREY
LARGE_BORDER = Color.ORANGE
FRAME_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK)
INNER_SIZES = (3, 5, 7)
VISIBLE_AREAS = (3, 4, 5, 6, 7)
MAX_GEN_ATTEMPTS = 400


def _frame_border_mask(shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_frames(grid: np.ndarray) -> list[dict]:
    colors = [int(c) for c in sorted(set(map(int, grid.ravel()))) if int(c) not in {BG, PATTERN}]
    frames: list[dict] = []
    for comp in find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=colors,
    ):
        color = int(comp["color"])
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
            continue
        sprite = crop_bbox(grid, (r0, c0, r1, c1))
        border_mask = _frame_border_mask(sprite.shape)
        if np.array_equal(sprite == color, border_mask):
            frames.append({"color": color, "bbox": (r0, c0, r1, c1)})
    return sorted(frames, key=lambda item: item["bbox"])


def _axis_candidates(interior: np.ndarray) -> list[tuple[str, str]]:
    coords = np.argwhere(interior == PATTERN)
    assert coords.size > 0, "expected a visible half-pattern"
    h, w = map(int, interior.shape)
    assert h % 2 == 1 and w % 2 == 1, "embedded mirror axes require odd interior sizes"

    mid_r = h // 2
    mid_c = w // 2
    rows = coords[:, 0]
    cols = coords[:, 1]
    candidates: list[tuple[str, str]] = []

    if np.all(rows < mid_r) or np.all(rows > mid_r):
        if not (np.all(cols < mid_c) or np.all(cols > mid_c)):
            side = "top" if np.all(rows < mid_r) else "bottom"
            candidates.append(("horizontal", side))
    if np.all(cols < mid_c) or np.all(cols > mid_c):
        if not (np.all(rows < mid_r) or np.all(rows > mid_r)):
            side = "left" if np.all(cols < mid_c) else "right"
            candidates.append(("vertical", side))
    return candidates


def _complete_pattern(interior: np.ndarray) -> tuple[str, set[tuple[int, int]]]:
    candidates = _axis_candidates(interior)
    assert len(candidates) == 1, "mirror-axis direction must be uniquely inferable per frame"
    axis, _side = candidates[0]
    h, w = map(int, interior.shape)
    mid_r = h // 2
    mid_c = w // 2

    completed = {tuple(map(int, coord)) for coord in np.argwhere(interior == PATTERN)}
    for r, c in list(completed):
        if axis == "horizontal":
            completed.add((2 * mid_r - r, c))
        else:
            completed.add((r, 2 * mid_c - c))
    return axis, completed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror-complete each framed half-pattern, then recolor the frame from the completed interior area.

    Input structure:
    - The background is black.
    - The grid contains 3-5 non-overlapping one-pixel-thick rectangular frames in distinct non-grey colors.
    - Inside each frame, grey pixels form one 4-connected irregular pattern confined to exactly one side of either the horizontal or vertical interior midline.
    - The matching opposite half of that frame interior is empty background.

    Rule:
    - For each frame independently, infer whether the interior mirror axis is the horizontal or vertical midline.
    - Mirror the visible grey half-pattern across that local axis to fill the empty opposite half.
    - Let A be the total number of grey interior cells after completion.
    - Recolor the whole frame grey when A <= 6, otherwise recolor the whole frame orange.
    - Keep the completed grey interior pattern in the output.

    Key insight:
    - The visible half-area is not the quantity that controls the border color; the pattern must be mirrored first, and the local side confinement is what reveals each frame's own axis.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = _find_frames(g)
    assert frames, "expected at least one rectangular frame"
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        axis, completed = _complete_pattern(interior)
        assert axis in {"horizontal", "vertical"}

        out[r0 + 1 : r1, c0 + 1 : c1] = np.where(interior == PATTERN, PATTERN, out[r0 + 1 : r1, c0 + 1 : c1])
        for rr, cc in completed:
            out[r0 + 1 + rr, c0 + 1 + cc] = PATTERN

        border_color = SMALL_BORDER if len(completed) <= 6 else LARGE_BORDER
        out[r0, c0 : c1 + 1] = border_color
        out[r1, c0 : c1 + 1] = border_color
        out[r0 : r1 + 1, c0] = border_color
        out[r0 : r1 + 1, c1] = border_color

    return out


def _crosses_other_midline(coords: set[tuple[int, int]], axis: str, mid_r: int, mid_c: int) -> bool:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    if axis == "horizontal":
        return not (all(c < mid_c for c in cols) or all(c > mid_c for c in cols))
    return not (all(r < mid_r for r in rows) or all(r > mid_r for r in rows))


def _candidate_inner_sizes(axis: str, visible_area: int) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    for inner_h in INNER_SIZES:
        for inner_w in INNER_SIZES:
            half_capacity = (inner_h // 2) * inner_w if axis == "horizontal" else inner_h * (inner_w // 2)
            if half_capacity >= visible_area:
                options.append((inner_h, inner_w))
    random.shuffle(options)
    return options


def _grow_connected_pattern(allowed: set[tuple[int, int]], n_cells: int) -> set[tuple[int, int]] | None:
    allowed_list = list(allowed)
    for _ in range(120):
        seed = random.choice(allowed_list)
        coords = {seed}
        while len(coords) < n_cells:
            frontier = []
            for r, c in coords:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in allowed and (nr, nc) not in coords:
                        frontier.append((nr, nc))
            if not frontier:
                break
            coords.add(random.choice(frontier))
        if len(coords) == n_cells:
            return coords
    return None


def _build_visible_pattern(
    axis: str,
    side: str,
    inner_h: int,
    inner_w: int,
    visible_area: int,
) -> set[tuple[int, int]]:
    mid_r = inner_h // 2
    mid_c = inner_w // 2

    if axis == "horizontal":
        rows = range(0, mid_r) if side == "top" else range(mid_r + 1, inner_h)
        allowed = {(r, c) for r in rows for c in range(inner_w)}
    else:
        cols = range(0, mid_c) if side == "left" else range(mid_c + 1, inner_w)
        allowed = {(r, c) for r in range(inner_h) for c in cols}

    for _ in range(120):
        coords = _grow_connected_pattern(allowed, visible_area)
        if coords is None:
            continue
        if _crosses_other_midline(coords, axis, mid_r, mid_c):
            return coords
    raise RuntimeError("failed to build an unambiguous visible half-pattern")


def _mirror_coords(coords: set[tuple[int, int]], axis: str, inner_h: int, inner_w: int) -> set[tuple[int, int]]:
    mid_r = inner_h // 2
    mid_c = inner_w // 2
    mirrored = set(coords)
    for r, c in coords:
        if axis == "horizontal":
            mirrored.add((2 * mid_r - r, c))
        else:
            mirrored.add((r, 2 * mid_c - c))
    return mirrored


def _render_frame(
    border_color: int,
    axis: str,
    side: str,
    visible_area: int,
) -> dict | None:
    for inner_h, inner_w in _candidate_inner_sizes(axis, visible_area):
        try:
            visible = _build_visible_pattern(axis, side, inner_h, inner_w, visible_area)
        except RuntimeError:
            continue

        completed = _mirror_coords(visible, axis, inner_h, inner_w)
        patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
        patch[0, :] = border_color
        patch[-1, :] = border_color
        patch[:, 0] = border_color
        patch[:, -1] = border_color
        for r, c in visible:
            patch[r + 1, c + 1] = PATTERN

        out_patch = np.full_like(patch, BG)
        border_after = SMALL_BORDER if len(completed) <= 6 else LARGE_BORDER
        out_patch[0, :] = border_after
        out_patch[-1, :] = border_after
        out_patch[:, 0] = border_after
        out_patch[:, -1] = border_after
        for r, c in completed:
            out_patch[r + 1, c + 1] = PATTERN

        return {
            "axis": axis,
            "side": side,
            "visible_area": visible_area,
            "completed_area": len(completed),
            "inner_shape": (inner_h, inner_w),
            "visible": visible,
            "completed": completed,
            "patch": patch,
            "out_patch": out_patch,
        }
    return None


def _frame_specs(n_frames: int) -> list[tuple[int, str, str, int]]:
    visible_areas = [3, 4]
    if n_frames > 2:
        visible_areas.extend(random.sample(list(VISIBLE_AREAS[2:]), n_frames - 2))
    random.shuffle(visible_areas)

    axes = ["horizontal", "vertical"]
    if n_frames > 2:
        axes.extend(random.choice(("horizontal", "vertical")) for _ in range(n_frames - 2))
    random.shuffle(axes)

    colors = random.sample(FRAME_COLORS, n_frames)
    specs: list[tuple[int, str, str, int]] = []
    for color, axis, area in zip(colors, axes, visible_areas):
        if axis == "horizontal":
            side = random.choice(("top", "bottom"))
        else:
            side = random.choice(("left", "right"))
        specs.append((color, axis, side, area))
    return specs


def _expected_from_metas(shape: tuple[int, int], metas: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for meta in metas:
        r0, c0 = meta["loc"]
        paste(out, meta["out_patch"], r0, c0)
    return out


def _validate_semantics(grid: np.ndarray, metas: list[dict]) -> None:
    assert 3 <= len(metas) <= 5
    assert {meta["axis"] for meta in metas} == {"horizontal", "vertical"}

    completed_areas = [int(meta["completed_area"]) for meta in metas]
    assert len(set(completed_areas)) == len(completed_areas)
    assert 6 in completed_areas
    assert any(area > 6 for area in completed_areas)
    assert any(meta["visible_area"] <= 6 < meta["completed_area"] for meta in metas)

    frames = _find_frames(grid)
    assert len(frames) == len(metas)
    for meta in metas:
        r0, c0 = meta["loc"]
        patch = meta["patch"]
        h, w = patch.shape
        local = grid[r0 : r0 + h, c0 : c0 + w]
        assert np.array_equal(local, patch)
        interior = local[1:-1, 1:-1]
        candidates = _axis_candidates(interior)
        assert candidates == [(meta["axis"], meta["side"])]


def _attempt_generate() -> tuple[np.ndarray, np.ndarray] | None:
    n_frames = random.randint(3, 5)
    specs = _frame_specs(n_frames)
    metas: list[dict] = []
    for color, axis, side, visible_area in specs:
        meta = _render_frame(color, axis, side, visible_area)
        if meta is None:
            return None
        metas.append(meta)

    height = random.randint(18, 26)
    width = random.randint(18, 26)
    grid = np.full((height, width), BG, dtype=int)
    occupancy = np.full((height, width), BG, dtype=int)

    random.shuffle(metas)
    for meta in metas:
        patch = meta["patch"]
        footprint = np.ones(patch.shape, dtype=int)
        try:
            r0, c0 = random_free_location(
                occupancy,
                footprint,
                transparent=BG,
                border=1,
                padding=2,
            )
        except ValueError:
            return None
        paste(occupancy, footprint, r0, c0)
        paste(grid, patch, r0, c0)
        meta["loc"] = (r0, c0)

    expected = _expected_from_metas(grid.shape, metas)
    _validate_semantics(grid, metas)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        pair = _attempt_generate()
        if pair is not None:
            grid, _expected = pair
            return grid
    raise RuntimeError("failed to generate a valid mirror-completion frame puzzle")
