from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg


BG = Color.BLACK
CROSS = Color.GREY
QUADRANT_COLORS = {"tl": Color.BLUE, "tr": Color.RED, "bl": Color.GREEN, "br": Color.YELLOW}
PATTERN_COLORS = (Color.ORANGE, Color.TEAL, Color.MAROON)
GEN_ATTEMPTS = 400


def _rot_cw(arr: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(arr, -int(quarter_turns) % 4).copy()


def _largest_panel_crop(grid: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "expected a foreground panel"
    max_area = max(int(comp["area"]) for comp in comps)
    panels = [comp for comp in comps if int(comp["area"]) == max_area]
    assert len(panels) == 1, "expected a unique largest panel component"
    panel_bbox = tuple(map(int, panels[0]["bbox"]))
    return crop_bbox(grid, panel_bbox), panel_bbox


def _parse_cross(panel: np.ndarray) -> tuple[int, int]:
    grey_rows = [r for r in range(panel.shape[0]) if bool(np.all(panel[r, :] == CROSS))]
    grey_cols = [c for c in range(panel.shape[1]) if bool(np.all(panel[:, c] == CROSS))]
    assert len(grey_rows) == 1 and len(grey_cols) == 1, "expected one full grey cross"
    mid_r = int(grey_rows[0])
    mid_c = int(grey_cols[0])
    assert mid_r == mid_c, "quadrants must be square for rotation"
    assert panel.shape[0] == panel.shape[1] == 2 * mid_r + 1, "expected a square odd-sized panel"
    return mid_r, mid_c


def _segment_marker(panel: np.ndarray, coords: list[tuple[int, int]]) -> tuple[tuple[int, int], int]:
    hits = [((r, c), int(panel[r, c])) for r, c in coords if int(panel[r, c]) not in (BG, CROSS)]
    assert len(hits) == 1, "each marker segment must contain one colored marker"
    return hits[0]


def _marker_info(panel: np.ndarray, q: int) -> tuple[dict[str, int], dict[str, list[tuple[int, int, int]]]]:
    top_left = _segment_marker(panel, [(0, c) for c in range(q)])
    top_right = _segment_marker(panel, [(0, c) for c in range(q + 1, 2 * q + 1)])
    left_top = _segment_marker(panel, [(r, 0) for r in range(q)])
    left_bottom = _segment_marker(panel, [(r, 0) for r in range(q + 1, 2 * q + 1)])
    right_top = _segment_marker(panel, [(r, 2 * q) for r in range(q)])
    right_bottom = _segment_marker(panel, [(r, 2 * q) for r in range(q + 1, 2 * q + 1)])
    bottom_left = _segment_marker(panel, [(2 * q, c) for c in range(q)])
    bottom_right = _segment_marker(panel, [(2 * q, c) for c in range(q + 1, 2 * q + 1)])

    colors = {
        "tl": int(top_left[1]),
        "tr": int(top_right[1]),
        "bl": int(bottom_left[1]),
        "br": int(bottom_right[1]),
    }
    assert colors["tl"] == int(left_top[1]) and colors["tr"] == int(right_top[1])
    assert colors["bl"] == int(left_bottom[1]) and colors["br"] == int(right_bottom[1])
    assert colors == QUADRANT_COLORS, "marker colors must uniquely label the four quadrants"

    positions = {
        "tl": [(top_left[0][0], top_left[0][1], colors["tl"]), (left_top[0][0], left_top[0][1], colors["tl"])],
        "tr": [(top_right[0][0], top_right[0][1], colors["tr"]), (right_top[0][0], right_top[0][1], colors["tr"])],
        "bl": [(bottom_left[0][0], bottom_left[0][1], colors["bl"]), (left_bottom[0][0], left_bottom[0][1], colors["bl"])],
        "br": [(bottom_right[0][0], bottom_right[0][1], colors["br"]), (right_bottom[0][0], right_bottom[0][1], colors["br"])],
    }
    return colors, positions


def _source_pattern(panel: np.ndarray, q: int, markers: dict[str, list[tuple[int, int, int]]]) -> np.ndarray:
    source = panel[:q, :q].copy()
    for r, c, _color in markers["tl"]:
        if 0 <= r < q and 0 <= c < q:
            source[r, c] = BG
    return source


def _cue_rotations(grid: np.ndarray, panel_bbox: tuple[int, int, int, int], bg: int) -> dict[int, int]:
    outside = np.array(grid, dtype=int, copy=True)
    r0, c0, r1, c1 = panel_bbox
    outside[r0 : r1 + 1, c0 : c1 + 1] = bg
    rotations = {int(color): 0 for color in QUADRANT_COLORS.values()}
    for color in rotations:
        comps = find_components(
            outside,
            bg=bg,
            connectivity=4,
            monochromatic=True,
            colors=[color],
            return_format="dict",
        )
        assert len(comps) <= 1, f"expected at most one cue sprite for color {color}"
        if comps:
            rotations[color] = int(comps[0]["area"]) % 4
    return rotations


def _build_output(panel: np.ndarray, source: np.ndarray, q: int, markers: dict[str, list[tuple[int, int, int]]], rotations: dict[int, int]) -> np.ndarray:
    out = np.full_like(panel, BG)
    out[q, :] = CROSS
    out[:, q] = CROSS

    quadrants = {
        "tl": (slice(0, q), slice(0, q)),
        "tr": (slice(0, q), slice(q + 1, 2 * q + 1)),
        "bl": (slice(q + 1, 2 * q + 1), slice(0, q)),
        "br": (slice(q + 1, 2 * q + 1), slice(q + 1, 2 * q + 1)),
    }
    for name, (rs, cs) in quadrants.items():
        quad = _rot_cw(source, rotations[QUADRANT_COLORS[name]])
        out[rs, cs] = quad
        for r, c, color in markers[name]:
            out[r, c] = color
    return out


def _expected_from_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    panel, panel_bbox = _largest_panel_crop(g, bg)
    q, _ = _parse_cross(panel)
    _colors, markers = _marker_info(panel, q)
    source = _source_pattern(panel, q, markers)
    rotations = _cue_rotations(g, panel_bbox, bg)
    return _build_output(panel, source, q, markers, rotations)


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the cropped panel by rotating the top-left source quadrant according to outside cue-sprite sizes.

    Input structure:
    - The background is black and the largest foreground component is a square panel with one full grey row and one full grey column forming a cross.
    - The panel boundary carries blue, red, green, and yellow edge markers that label the top-left, top-right, bottom-left, and bottom-right quadrants.
    - Outside the panel, optional monochrome cue sprites of those four marker colors appear on the black background.

    Rule:
    - Tight-crop the largest panel component and split it with the grey cross into four equal square quadrants.
    - Use the top-left quadrant, excluding its fixed edge-marker cells, as the source pattern.
    - For each quadrant color, rotate the source clockwise by `(cue_size mod 4) * 90` if a same-color outside cue exists, otherwise keep it unrotated.
    - Fill every quadrant with its rotated source copy while preserving the grey cross and all edge markers, and return only the cropped panel.

    Key insight:
    - The inside quadrant contents are distractors; only the top-left source pattern and the outside same-color cue sizes determine the rebuilt panel.
    """
    return _expected_from_grid(input_grid)


def _distinct_rotations(pattern: np.ndarray) -> bool:
    rots = {_rot_cw(pattern, k).tobytes() for k in range(4)}
    return len(rots) == 4


def _grow_connected_cells(cells: list[tuple[int, int]], size: int) -> list[tuple[int, int]]:
    shape = {random.choice(cells)}
    while len(shape) < size:
        frontier = []
        for r, c in shape:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                cand = (r + dr, c + dc)
                if cand in cells and cand not in shape:
                    frontier.append(cand)
        if not frontier:
            return []
        shape.add(random.choice(frontier))
    return sorted(shape)


def _make_source(q: int) -> np.ndarray:
    allowed = [(r, c) for r in range(1, q) for c in range(1, q)]
    min_size = max(4, q)
    max_size = min(len(allowed), q + 2)
    for _ in range(200):
        coords = _grow_connected_cells(allowed, random.randint(min_size, max_size))
        if not coords:
            continue
        pattern = np.zeros((q, q), dtype=int)
        colors = random.sample(PATTERN_COLORS, k=random.randint(2, len(PATTERN_COLORS)))
        for r, c in coords:
            pattern[r, c] = random.choice(colors)
        if len(set(map(int, pattern[pattern != BG].tolist()))) < 2:
            continue
        if not any(r == q - 1 or c == q - 1 for r, c in coords):
            continue
        if _distinct_rotations(pattern):
            return pattern
    raise RuntimeError("failed to build a rotation-asymmetric source pattern")


def _random_shape(q: int) -> np.ndarray:
    allowed = [(r, c) for r in range(1, q) for c in range(1, q)]
    for _ in range(120):
        size = random.randint(3, min(len(allowed), q + 2))
        coords = _grow_connected_cells(allowed, size)
        if not coords:
            continue
        pattern = np.zeros((q, q), dtype=int)
        colors = random.sample(PATTERN_COLORS, k=random.randint(1, len(PATTERN_COLORS)))
        for r, c in coords:
            pattern[r, c] = random.choice(colors)
        if np.count_nonzero(pattern) >= 3:
            return pattern
    raise RuntimeError("failed to build distractor pattern")


def _make_distractor(source: np.ndarray, expected_rot: int) -> np.ndarray:
    wrong_rots = [_rot_cw(source, k) for k in range(4) if k != expected_rot]
    if random.random() < 0.7:
        return random.choice(wrong_rots).copy()
    forbidden = {arr.tobytes() for arr in wrong_rots + [_rot_cw(source, expected_rot)]}
    for _ in range(120):
        candidate = _random_shape(source.shape[0])
        if candidate.tobytes() not in forbidden:
            return candidate
    return random.choice(wrong_rots).copy()


def _cue_patch(size: int, color: int) -> np.ndarray:
    cells = [(r, c) for r in range(4) for c in range(4)]
    for _ in range(120):
        coords = _grow_connected_cells(cells, size)
        if not coords:
            continue
        arr = np.zeros((4, 4), dtype=int)
        for r, c in coords:
            arr[r, c] = color
        nz = np.argwhere(arr != BG)
        r0, c0 = nz.min(axis=0)
        r1, c1 = nz.max(axis=0)
        return arr[r0 : r1 + 1, c0 : c1 + 1].copy()
    raise RuntimeError("failed to build cue patch")


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    view = grid[top : top + h, left : left + w]
    view[patch != BG] = patch[patch != BG]


def _free_positions(grid: np.ndarray, patch: np.ndarray, pad: int) -> list[tuple[int, int]]:
    h, w = patch.shape
    H, W = grid.shape
    out = []
    occ = (grid != BG)
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            rr0 = max(0, r0 - pad)
            cc0 = max(0, c0 - pad)
            rr1 = min(H, r0 + h + pad)
            cc1 = min(W, c0 + w + pad)
            if bool(occ[rr0:rr1, cc0:cc1].any()):
                continue
            out.append((r0, c0))
    return out


def _marker_template(q: int) -> dict[str, list[tuple[int, int, int]]]:
    inner = q // 2
    return {
        "tl": [(0, inner, QUADRANT_COLORS["tl"]), (inner, 0, QUADRANT_COLORS["tl"])],
        "tr": [(0, q + 1 + inner, QUADRANT_COLORS["tr"]), (inner, 2 * q, QUADRANT_COLORS["tr"])],
        "bl": [(2 * q, inner, QUADRANT_COLORS["bl"]), (q + 1 + inner, 0, QUADRANT_COLORS["bl"])],
        "br": [(2 * q, q + 1 + inner, QUADRANT_COLORS["br"]), (q + 1 + inner, 2 * q, QUADRANT_COLORS["br"])],
    }


def _panel_from_parts(source: np.ndarray, rotations: dict[int, int], distractors: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, list[tuple[int, int, int]]]]:
    q = source.shape[0]
    panel = np.zeros((2 * q + 1, 2 * q + 1), dtype=int)
    panel[q, :] = CROSS
    panel[:, q] = CROSS
    markers = _marker_template(q)
    quadrants = {
        "tl": (slice(0, q), slice(0, q)),
        "tr": (slice(0, q), slice(q + 1, 2 * q + 1)),
        "bl": (slice(q + 1, 2 * q + 1), slice(0, q)),
        "br": (slice(q + 1, 2 * q + 1), slice(q + 1, 2 * q + 1)),
    }
    panel[quadrants["tl"]] = source
    for name in ("tr", "bl", "br"):
        panel[quadrants[name]] = distractors[name]
    for name, triples in markers.items():
        for r, c, color in triples:
            panel[r, c] = color
    expected = _build_output(panel, source, q, markers, rotations)
    return panel, markers


def _sample_rotations() -> dict[int, int]:
    cue_count = random.randint(1, 3)
    chosen = random.sample(list(QUADRANT_COLORS.values()), k=cue_count)
    rotations = {int(color): 0 for color in QUADRANT_COLORS.values()}
    nonzero_seen = False
    for i, color in enumerate(chosen):
        options = [1, 2, 3] if i == cue_count - 1 and not nonzero_seen else [0, 1, 2, 3]
        rot = random.choice(options)
        rotations[int(color)] = rot
        nonzero_seen |= rot != 0
    assert nonzero_seen and any(rot == 0 for rot in rotations.values())
    return rotations


def _make_instance() -> tuple[np.ndarray, np.ndarray]:
    q = random.randint(4, 6)
    source = _make_source(q)
    rotations = _sample_rotations()
    distractors = {
        "tr": _make_distractor(source, rotations[QUADRANT_COLORS["tr"]]),
        "bl": _make_distractor(source, rotations[QUADRANT_COLORS["bl"]]),
        "br": _make_distractor(source, rotations[QUADRANT_COLORS["br"]]),
    }

    panel, markers = _panel_from_parts(source, rotations, distractors)
    expected = _build_output(panel, source, q, markers, rotations)
    assert not np.array_equal(panel[q + 1 :, q + 1 :], expected[q + 1 :, q + 1 :]) or not np.array_equal(
        panel[:q, q + 1 :], expected[:q, q + 1 :]
    ) or not np.array_equal(panel[q + 1 :, :q], expected[q + 1 :, :q])

    H = random.randint(max(panel.shape[0] + 4, 17), min(30, panel.shape[0] + 12))
    W = random.randint(max(panel.shape[1] + 4, 17), min(30, panel.shape[1] + 12))
    grid = np.zeros((H, W), dtype=int)

    pr = random.randint(2, H - panel.shape[0] - 2)
    pc = random.randint(2, W - panel.shape[1] - 2)
    _paint_patch(grid, panel, pr, pc)

    chosen_colors = [color for color, rot in rotations.items() if rot != 0 or random.random() < 0.45]
    chosen_colors = [color for color in chosen_colors if rotations[color] != 0 or color in rotations]
    present_colors = [color for color, rot in rotations.items() if rot != 0 or (color in chosen_colors and rot == 0)]
    if len(present_colors) == 0:
        present_colors = [next(color for color, rot in rotations.items() if rot != 0)]
    if len(present_colors) > 3:
        random.shuffle(present_colors)
        keep = [c for c in present_colors if rotations[c] != 0][:1]
        keep += [c for c in present_colors if c not in keep][: 3 - len(keep)]
        present_colors = keep

    panel_bbox = (pr, pc, pr + panel.shape[0] - 1, pc + panel.shape[1] - 1)
    for color in present_colors:
        rot = rotations[color]
        size_options = [n for n in range(2, 8) if n % 4 == rot]
        size = random.choice(size_options)
        patch = _cue_patch(size, color)
        candidates = _free_positions(grid, patch, pad=1)
        candidates = [
            (r0, c0)
            for r0, c0 in candidates
            if r0 + patch.shape[0] - 1 < panel_bbox[0] - 1
            or r0 > panel_bbox[2] + 1
            or c0 + patch.shape[1] - 1 < panel_bbox[1] - 1
            or c0 > panel_bbox[3] + 1
        ]
        assert candidates, "expected free cue placement"
        r0, c0 = random.choice(candidates)
        _paint_patch(grid, patch, r0, c0)

    return grid, expected


def _validate_generated(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        if grid.ndim != 2 or max(grid.shape) > 30:
            return False
        if expected.ndim != 2 or max(expected.shape) > 30:
            return False

        solved = main(grid)
        if not np.array_equal(solved, expected):
            return False

        bg = int(get_bg(grid))
        panel, panel_bbox = _largest_panel_crop(grid, bg)
        q, _ = _parse_cross(panel)
        _colors, markers = _marker_info(panel, q)
        source = _source_pattern(panel, q, markers)
        if not _distinct_rotations(source):
            return False

        rotations = _cue_rotations(grid, panel_bbox, bg)
        if not any(rot != 0 for rot in rotations.values()):
            return False
        if not any(rot == 0 for rot in rotations.values()):
            return False

        quadrants = {
            "tl": panel[:q, :q],
            "tr": panel[:q, q + 1 :],
            "bl": panel[q + 1 :, :q],
            "br": panel[q + 1 :, q + 1 :],
        }
        expected_quadrants = {
            "tl": solved[:q, :q],
            "tr": solved[:q, q + 1 :],
            "bl": solved[q + 1 :, :q],
            "br": solved[q + 1 :, q + 1 :],
        }
        changed_non_source = sum(not np.array_equal(quadrants[name], expected_quadrants[name]) for name in ("tr", "bl", "br"))
        if changed_non_source < 2:
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid, expected = _make_instance()
        except Exception:
            continue
        if _validate_generated(grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
