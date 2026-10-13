from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, in_bounds, neighbors8


LEFT_CORNERS = ("TL", "BL")
RIGHT_CORNERS = ("TR", "BR")
CORNER_NAMES = ("TL", "TR", "BL", "BR")
MOTIF_COLORS = (Color.BLUE, Color.GREEN)
CONTROL_COLORS = (Color.RED, Color.YELLOW, Color.GREY)

GEN_ATTEMPTS = 600
GRID_H_RANGE = (13, 19)
GRID_W_RANGE = (13, 19)
MOTIF_AREA_RANGE = (10, 25)
TIE_BRANCH_PROB = 0.4


def _corner_coord(shape: tuple[int, int], name: str) -> tuple[int, int]:
    h, w = shape
    if name == "TL":
        return 0, 0
    if name == "TR":
        return 0, w - 1
    if name == "BL":
        return h - 1, 0
    if name == "BR":
        return h - 1, w - 1
    raise ValueError(f"unknown corner name: {name}")


def _stripe_columns(width: int, start_col: int, period: int) -> list[int]:
    if start_col == 0:
        return list(range(0, width, period))
    return list(range(width - 1, -1, -period))


def _render_output(
    shape: tuple[int, int],
    motif_coords: set[tuple[int, int]],
    motif_color: int,
    stripe_cols: list[int],
    stripe_color: int,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    out[:, stripe_cols] = int(stripe_color)
    for r, c in motif_coords:
        out[r, c] = int(motif_color)
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for nr, nc in neighbors8(r, c):
            if in_bounds(mask.shape, nr, nc):
                out[nr, nc] = True
    return out


def _choose_stripe_color(points: list[tuple[int, int, int]]) -> int:
    counts = Counter(color for _r, _c, color in points)
    max_count = max(counts.values())
    tied = [color for color, count in counts.items() if count == max_count]
    if len(tied) == 1:
        return int(tied[0])

    first_pos = {}
    for color in tied:
        positions = [(r, c) for r, c, col in points if col == color]
        first_pos[color] = min(positions)
    return int(min(tied, key=lambda color: first_pos[color]))


def _make_connected_patch() -> np.ndarray | None:
    for _ in range(200):
        patch_h = random.randint(4, 7)
        patch_w = random.randint(4, 7)
        max_area = min(MOTIF_AREA_RANGE[1], patch_h * patch_w - 1)
        if max_area < MOTIF_AREA_RANGE[0]:
            continue
        target_area = random.randint(MOTIF_AREA_RANGE[0], max_area)

        cells = {(random.randrange(patch_h), random.randrange(patch_w))}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for nr, nc in neighbors8(r, c):
                    if 0 <= nr < patch_h and 0 <= nc < patch_w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != target_area:
            continue

        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        bbox_h = r1 - r0 + 1
        bbox_w = c1 - c0 + 1
        bbox_area = bbox_h * bbox_w
        density = target_area / bbox_area
        if bbox_h < 3 or bbox_w < 3:
            continue
        if not (0.45 <= density <= 0.85):
            continue

        patch = np.zeros((bbox_h, bbox_w), dtype=bool)
        for r, c in cells:
            patch[r - r0, c - c0] = True
        if np.all(patch.sum(axis=0) >= 1) and np.all(patch.sum(axis=1) >= 1):
            return patch
    return None


def _place_motif(
    shape: tuple[int, int],
    patch: np.ndarray,
    stripe_cols: list[int],
) -> set[tuple[int, int]] | None:
    h, w = shape
    patch_h, patch_w = patch.shape
    cells = np.argwhere(patch)
    placements = []
    for top in range(2, h - patch_h - 1):
        for left in range(2, w - patch_w - 1):
            placed = {(int(top + r), int(left + c)) for r, c in cells}
            overlap = sum(1 for _r, c in placed if c in stripe_cols)
            if overlap == 0:
                continue
            if overlap == len(placed):
                continue
            placements.append(placed)
    if not placements:
        return None
    return random.choice(placements)


def _sample_control_spec() -> dict:
    if random.random() < TIE_BRANCH_PROB:
        tie_colors = random.sample(CONTROL_COLORS, 2)
        singleton = random.choice([color for color in CONTROL_COLORS if color not in tie_colors])
        return {
            "mode": "tie",
            "counts": {tie_colors[0]: 2, tie_colors[1]: 2, singleton: 1},
            "winner": random.choice(tie_colors),
        }

    major = random.choice(CONTROL_COLORS)
    distractors = random.sample(
        [color for color in CONTROL_COLORS if color != major],
        random.choice((1, 2)),
    )
    major_count = random.randint(2, 5 - len(distractors))
    counts = {major: major_count}
    for color in distractors:
        counts[color] = 1
    return {"mode": "majority", "counts": counts, "major": major}


def _sample_control_positions(
    shape: tuple[int, int],
    count: int,
    blocked: np.ndarray,
    corner_name: str,
    stripe_cols: list[int],
) -> list[tuple[int, int]] | None:
    h, w = shape
    blocked = blocked.copy()
    corner_r, corner_c = _corner_coord(shape, corner_name)
    blocked[:, stripe_cols] = True
    for name in CORNER_NAMES:
        r, c = _corner_coord(shape, name)
        blocked[r, c] = True
    for nr, nc in neighbors8(corner_r, corner_c):
        if in_bounds(shape, nr, nc):
            blocked[nr, nc] = True

    points = []
    for _ in range(count):
        pool = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if not blocked[r, c]
        ]
        if not pool:
            return None
        pos = random.choice(pool)
        points.append(pos)
        pr, pc = pos
        blocked[pr, pc] = True
        for nr, nc in neighbors8(pr, pc):
            if in_bounds(shape, nr, nc):
                blocked[nr, nc] = True
    return points


def _assign_control_colors(
    positions: list[tuple[int, int]],
    spec: dict,
) -> dict[tuple[int, int], int]:
    ordered = sorted(positions)
    remaining = dict(spec["counts"])
    assigned: dict[tuple[int, int], int] = {}

    if spec["mode"] == "tie":
        first_color = int(spec["winner"])
    else:
        major = int(spec["major"])
        distractors = [color for color in remaining if color != major and remaining[color] > 0]
        if distractors and random.random() < 0.65:
            first_color = int(random.choice(distractors))
        else:
            first_color = major

    assigned[ordered[0]] = first_color
    remaining[first_color] -= 1
    color_bag = [color for color, count in remaining.items() for _ in range(count)]
    random.shuffle(color_bag)
    for pos, color in zip(ordered[1:], color_bag):
        assigned[pos] = int(color)
    return assigned


def _scene_is_valid(
    inp: np.ndarray,
    expected: np.ndarray,
    motif_coords: set[tuple[int, int]],
    motif_color: int,
    stripe_cols: list[int],
    stripe_color: int,
    period: int,
    corner_name: str,
    controls: dict[tuple[int, int], int],
) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    overlap = sum(1 for _r, c in motif_coords if c in stripe_cols)
    visible_stripes = len(stripe_cols) * inp.shape[0] - overlap
    if len(stripe_cols) < 2 or overlap == 0 or visible_stripes < inp.shape[0]:
        return False

    points = [(r, c, color) for (r, c), color in controls.items()]
    counts = Counter(color for _r, _c, color in points)
    if len(counts) < 2:
        return False
    if _choose_stripe_color(points) != stripe_color:
        return False

    alt_side = "TR" if corner_name == "TL" else "TL"
    if corner_name == "BL":
        alt_side = "BR"
    elif corner_name == "BR":
        alt_side = "BL"
    wrong_start = _corner_coord(inp.shape, alt_side)[1]
    wrong_cols = _stripe_columns(inp.shape[1], wrong_start, period)
    wrong_side_output = _render_output(inp.shape, motif_coords, motif_color, wrong_cols, stripe_color)
    if np.array_equal(wrong_side_output, expected):
        return False

    max_count = max(counts.values())
    wrong_period_output = _render_output(
        inp.shape,
        motif_coords,
        motif_color,
        _stripe_columns(inp.shape[1], _corner_coord(inp.shape, corner_name)[1], max_count),
        stripe_color,
    )
    if np.array_equal(wrong_period_output, expected):
        return False

    alt_color = next(color for color in CONTROL_COLORS if color != stripe_color)
    wrong_color_output = _render_output(inp.shape, motif_coords, motif_color, stripe_cols, alt_color)
    if np.array_equal(wrong_color_output, expected):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the unique largest 8-connected motif and repaint the rest as periodic corner-anchored vertical stripes.

    Input structure:
    - The background is black, with one irregular blue or green motif that is the only foreground component larger than one cell.
    - Exactly one orange corner pixel marks the side from which the stripe phase starts.
    - All other non-black cells are isolated control singletons colored red, yellow, or grey.
    - At least two control colors are present, and the motif is separate from every control pixel.

    Rule:
    - Find the unique largest monochrome 8-connected component and keep it unchanged.
    - Count all non-corner control singletons to get the stripe period N.
    - Choose the stripe color as the most frequent control color; if several colors tie, use the tied color whose control pixel appears earliest in row-major order.
    - Erase every control singleton and the corner pixel, then draw full-height vertical stripes every N columns starting from the marked corner side.
    - Whenever a stripe would overlap the motif, keep the motif pixel instead.

    Key insight:
    - Three different clues must be combined: the largest component is the only shape that survives, the number of scattered controls sets the stripe spacing, and the control-color frequency selects the stripe color.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape

    marked_corners = [
        (name, _corner_coord((h, w), name))
        for name in CORNER_NAMES
        if int(g[_corner_coord((h, w), name)]) != bg
    ]
    assert len(marked_corners) == 1, "expected exactly one foreground corner marker"
    corner_name, corner_rc = marked_corners[0]

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert comps, "expected foreground components"
    max_area = max(int(comp["area"]) for comp in comps)
    motif_candidates = [comp for comp in comps if int(comp["area"]) == max_area]
    assert max_area > 1 and len(motif_candidates) == 1, "expected a unique largest motif"
    motif = motif_candidates[0]
    motif_color = int(motif["color"])
    motif_coords = {tuple(map(int, coord)) for coord in motif["coords"]}

    controls = []
    for comp in comps:
        if comp is motif:
            continue
        assert int(comp["area"]) == 1, "all non-motif components must be singleton controls"
        r, c = map(int, comp["coords"][0])
        if (r, c) != corner_rc:
            controls.append((r, c, int(comp["color"])))

    assert controls, "expected at least one non-corner control"
    period = len(controls)
    stripe_color = _choose_stripe_color(controls)
    stripe_cols = _stripe_columns(w, corner_rc[1], period)

    out = _render_output(g.shape, motif_coords, motif_color, stripe_cols, stripe_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    spec = _sample_control_spec()
    control_count = sum(int(count) for count in spec["counts"].values())

    valid_widths = [
        width
        for width in range(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1)
        if width - 1 > control_count and (width - 1) % control_count != 0
    ]
    if not valid_widths:
        return None

    h = random.randint(*GRID_H_RANGE)
    w = random.choice(valid_widths)
    corner_name = random.choice(CORNER_NAMES)
    start_col = _corner_coord((h, w), corner_name)[1]
    stripe_cols = _stripe_columns(w, start_col, control_count)

    patch = _make_connected_patch()
    if patch is None:
        return None
    motif_coords = _place_motif((h, w), patch, stripe_cols)
    if motif_coords is None:
        return None

    motif_mask = np.zeros((h, w), dtype=bool)
    for r, c in motif_coords:
        motif_mask[r, c] = True

    controls_blocked = _dilate8(motif_mask)
    positions = _sample_control_positions((h, w), control_count, controls_blocked, corner_name, stripe_cols)
    if positions is None:
        return None
    control_colors = _assign_control_colors(positions, spec)
    points = [(r, c, color) for (r, c), color in control_colors.items()]
    stripe_color = _choose_stripe_color(points)

    motif_color = random.choice(MOTIF_COLORS)
    inp = np.zeros((h, w), dtype=int)
    for r, c in motif_coords:
        inp[r, c] = motif_color
    inp[_corner_coord((h, w), corner_name)] = Color.ORANGE
    for (r, c), color in control_colors.items():
        inp[r, c] = int(color)

    expected = _render_output((h, w), motif_coords, motif_color, stripe_cols, stripe_color)
    if not _scene_is_valid(
        inp,
        expected,
        motif_coords,
        motif_color,
        stripe_cols,
        stripe_color,
        control_count,
        corner_name,
        control_colors,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for periodic stripe motif task")
