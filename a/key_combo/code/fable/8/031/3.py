from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np


ORTH4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PLUS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RECT_COLORS = tuple(range(1, 10))

GRID_MIN = 24
GRID_MAX = 28
CORE_SIZE_RANGE = (3, 4)
LONG_EXT_RANGE = (4, 8)
SHORT_EXT_RANGE = (1, 4)
GEN_ATTEMPTS = 400


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = sorted({int(v) for v in np.asarray(grid).ravel()})
    if exclude is None:
        return colors
    banned = {int(v) for v in exclude}
    return [color for color in colors if color not in banned]


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ORTH4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _paint_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _render_rectangles(
    shape: tuple[int, int],
    rects: list[dict],
    color_order: list[int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    by_color = {int(rect["color"]): rect for rect in rects}
    for color in color_order:
        _paint_rect(out, tuple(by_color[int(color)]["bbox"]), int(color))
    return out


def _visible_bbox_matches(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    if _bbox_coords(coords) != tuple(map(int, bbox)):
        return False
    r0, c0, r1, c1 = map(int, bbox)
    return bool(mask[r0, c0 : c1 + 1].any()) and bool(mask[r1, c0 : c1 + 1].any()) and bool(
        mask[r0 : r1 + 1, c0].any()
    ) and bool(mask[r0 : r1 + 1, c1].any())


def _marker_signature(grid: np.ndarray, coords: np.ndarray) -> dict | None:
    coords = np.asarray(coords, dtype=int)
    if not 3 <= len(coords) <= 5:
        return None

    cell_set = {(int(r), int(c)) for r, c in coords}
    centers: list[tuple[int, int]] = []
    for r, c in cell_set:
        others = cell_set - {(r, c)}
        if all(abs(rr - r) + abs(cc - c) == 1 for rr, cc in others):
            centers.append((r, c))
    if len(centers) != 1:
        return None

    center = centers[0]
    r, c = center
    h, w = grid.shape
    if not (2 <= r < h - 2 and 2 <= c < w - 2):
        return None

    expected_cells = {center}
    present_arms: list[tuple[int, int]] = []
    for dr, dc in PLUS_DIRS:
        arm = (r + dr, c + dc)
        expected_cells.add(arm)
        if arm in cell_set:
            present_arms.append(arm)
    if cell_set - expected_cells:
        return None
    if len(present_arms) not in (2, 3, 4):
        return None

    ring = grid[r - 2 : r + 3, c - 2 : c + 3]
    for rr in range(r - 2, r + 3):
        for cc in range(c - 2, c + 3):
            if int(grid[rr, cc]) != 0 and (rr, cc) not in cell_set:
                return None

    arm_colors = [int(grid[rr, cc]) for rr, cc in present_arms]
    if len(set(arm_colors)) != 1:
        return None

    return {
        "center": center,
        "center_color": int(grid[r, c]),
        "arm_color": arm_colors[0],
        "complete": len(present_arms) == 4,
        "coords": coords,
    }


def _extract_markers(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    markers: list[dict] = []
    removed = g.copy()
    for coords in _component_coords(g != 0):
        sig = _marker_signature(g, coords)
        if sig is None:
            continue
        markers.append(sig)
        for r, c in coords:
            removed[int(r), int(c)] = 0
    return markers, removed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the occluding rectangles by the number of complete center-colored plus votes they receive.

    Input structure:
    - The background is black.
    - Several different-colored solid rectangles overlap, so lower rectangles may appear only as visible fragments.
    - Empty background cells may contain isolated complete or incomplete plus markers whose center color and arm color are independent.

    Rule:
    - Treat an isolated complete plus as a vote only for the rectangle whose color matches the plus center pixel.
    - Remove every complete plus and every incomplete plus from the scene.
    - For each remaining rectangle color, recover the full solid rectangle from the bbox of its visible fragment, then redraw all rectangles from fewest votes to most votes so higher-vote rectangles end up on top.

    Key insight:
    - The arm color is a distractor, the visible z-order is a decoy, and the real answer comes from stripping the isolated plus markers, amodally completing each rectangle, then repainting by center-color vote counts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    markers, stripped = _extract_markers(g)
    vote_counts: Counter[int] = Counter()
    for marker in markers:
        if marker["complete"]:
            vote_counts[int(marker["center_color"])] += 1

    rect_colors = _colors_present(stripped, exclude={0})
    assert rect_colors, "expected at least one rectangle color"

    rects: list[tuple[int, tuple[int, int, int, int], int]] = []
    for color in rect_colors:
        coords = np.argwhere(stripped == int(color))
        assert coords.size > 0, "rectangle color must remain visible after marker removal"
        rects.append((int(color), _bbox_coords(coords), int(vote_counts.get(int(color), 0))))

    votes = [votes for _color, _bbox, votes in rects]
    assert len(set(votes)) == len(votes), "true vote counts must be pairwise distinct"

    out = np.zeros_like(stripped)
    for color, bbox, _votes in sorted(rects, key=lambda item: item[2]):
        _paint_rect(out, bbox, color)
    return out


def _sample_extent(lo: int, hi: int, available: int) -> int | None:
    if available < lo:
        return None
    return random.randint(lo, min(hi, available))


def _role_bbox(
    role: str,
    core_bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    h, w = shape
    cr0, cc0, cr1, cc1 = core_bbox
    top_avail = cr0
    left_avail = cc0
    bottom_avail = h - 1 - cr1
    right_avail = w - 1 - cc1

    long_top = _sample_extent(LONG_EXT_RANGE[0], LONG_EXT_RANGE[1], top_avail)
    long_left = _sample_extent(LONG_EXT_RANGE[0], LONG_EXT_RANGE[1], left_avail)
    long_bottom = _sample_extent(LONG_EXT_RANGE[0], LONG_EXT_RANGE[1], bottom_avail)
    long_right = _sample_extent(LONG_EXT_RANGE[0], LONG_EXT_RANGE[1], right_avail)
    short_top = _sample_extent(SHORT_EXT_RANGE[0], SHORT_EXT_RANGE[1], top_avail)
    short_left = _sample_extent(SHORT_EXT_RANGE[0], SHORT_EXT_RANGE[1], left_avail)
    short_bottom = _sample_extent(SHORT_EXT_RANGE[0], SHORT_EXT_RANGE[1], bottom_avail)
    short_right = _sample_extent(SHORT_EXT_RANGE[0], SHORT_EXT_RANGE[1], right_avail)

    needed = [
        long_top,
        long_left,
        long_bottom,
        long_right,
        short_top,
        short_left,
        short_bottom,
        short_right,
    ]
    if any(value is None for value in needed):
        return None

    if role == "UL":
        return cr0 - long_top, cc0 - long_left, cr1 + short_bottom, cc1 + short_right
    if role == "UR":
        return cr0 - long_top, cc0 - short_left, cr1 + short_bottom, cc1 + long_right
    if role == "DL":
        return cr0 - short_top, cc0 - long_left, cr1 + long_bottom, cc1 + short_right
    if role == "DR":
        return cr0 - short_top, cc0 - short_left, cr1 + long_bottom, cc1 + long_right
    raise ValueError(f"unknown role {role}")


def _choose_votes(n_rect: int) -> list[int]:
    if n_rect == 4:
        base = [1, 2, 3, 4]
    else:
        options = ([1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4])
        base = list(random.choice(options))
    random.shuffle(base)
    return base


def _marker_cells(center: tuple[int, int], dirs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r, c = center
    cells = [(r, c)]
    for dr, dc in dirs:
        cells.append((r + dr, c + dc))
    return cells


def _available_marker_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(2, h - 2):
        for c in range(2, w - 2):
            if np.all(grid[r - 2 : r + 3, c - 2 : c + 3] == 0):
                centers.append((r, c))
    return centers


def _place_marker(
    grid: np.ndarray,
    center_color: int,
    arm_color: int,
    dirs: list[tuple[int, int]],
) -> dict | None:
    centers = _available_marker_centers(grid)
    if not centers:
        return None
    center = random.choice(centers)
    cells = _marker_cells(center, dirs)
    r, c = center
    grid[r, c] = int(center_color)
    for rr, cc in cells[1:]:
        grid[rr, cc] = int(arm_color)
    return {
        "center": center,
        "center_color": int(center_color),
        "arm_color": int(arm_color),
        "dirs": list(dirs),
        "complete": len(dirs) == 4,
    }


def _expected_output(shape: tuple[int, int], rects: list[dict]) -> np.ndarray:
    order = [int(rect["color"]) for rect in sorted(rects, key=lambda rect: int(rect["votes"]))]
    return _render_rectangles(shape, rects, order)


def _valid_input_orders(rects: list[dict], shape: tuple[int, int], output_order: list[int]) -> list[list[int]]:
    candidates: list[list[int]] = []
    colors = [int(rect["color"]) for rect in rects]
    expected = _render_rectangles(shape, rects, output_order)
    for order in itertools.permutations(colors):
        if list(order) == list(output_order):
            continue
        rect_input = _render_rectangles(shape, rects, list(order))
        if np.array_equal(rect_input, expected):
            continue
        occluded = False
        ok = True
        for rect in rects:
            color = int(rect["color"])
            bbox = tuple(rect["bbox"])
            full_area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
            visible_mask = rect_input == color
            if not _visible_bbox_matches(visible_mask, bbox):
                ok = False
                break
            if int(visible_mask.sum()) < full_area:
                occluded = True
        if ok and occluded and not any(
            _marker_signature(rect_input, coords) is not None for coords in _component_coords(rect_input != 0)
        ):
            candidates.append(list(order))
    return candidates


def _build_scene() -> dict | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    n_rect = random.choice([3, 4])

    core_h = random.randint(*CORE_SIZE_RANGE)
    core_w = random.randint(*CORE_SIZE_RANGE)
    core_r0 = random.randint(8, h - core_h - 9)
    core_c0 = random.randint(8, w - core_w - 9)
    core_bbox = (core_r0, core_c0, core_r0 + core_h - 1, core_c0 + core_w - 1)

    roles = random.sample(["UL", "UR", "DL", "DR"], n_rect)
    rect_colors = random.sample(list(RECT_COLORS), n_rect)
    votes = _choose_votes(n_rect)
    rects: list[dict] = []
    seen_bboxes: set[tuple[int, int, int, int]] = set()
    for role, color, vote in zip(roles, rect_colors, votes):
        bbox = _role_bbox(role, core_bbox, (h, w))
        if bbox is None or bbox in seen_bboxes:
            return None
        seen_bboxes.add(bbox)
        rects.append(
            {
                "color": int(color),
                "bbox": tuple(map(int, bbox)),
                "votes": int(vote),
                "role": role,
            }
        )

    output_order = [int(rect["color"]) for rect in sorted(rects, key=lambda rect: int(rect["votes"]))]
    expected = _expected_output((h, w), rects)
    valid_orders = _valid_input_orders(rects, (h, w), output_order)
    if not valid_orders:
        return None
    input_order = random.choice(valid_orders)
    rect_input = _render_rectangles((h, w), rects, input_order)

    grid = rect_input.copy()
    markers: list[dict] = []
    spare_colors = [color for color in RECT_COLORS if color not in rect_colors]
    if len(spare_colors) < 2:
        return None

    for rect in rects:
        center_color = int(rect["color"])
        arm_palette = list(RECT_COLORS)
        if random.random() < 0.7:
            arm_color = random.choice(rect_colors)
        else:
            arm_color = random.choice(arm_palette)
        placed = _place_marker(grid, center_color, arm_color, PLUS_DIRS)
        if placed is None:
            return None
        markers.append(placed)
        for _ in range(int(rect["votes"]) - 1):
            if random.random() < 0.45:
                arm_color = center_color
            elif random.random() < 0.65:
                arm_color = random.choice(rect_colors)
            else:
                arm_color = random.choice(arm_palette)
            placed = _place_marker(grid, center_color, arm_color, PLUS_DIRS)
            if placed is None:
                return None
            markers.append(placed)

    false_complete_n = random.randint(1, 2)
    for _ in range(false_complete_n):
        center_color = random.choice(spare_colors)
        arm_color = random.choice(rect_colors)
        placed = _place_marker(grid, center_color, arm_color, PLUS_DIRS)
        if placed is None:
            return None
        markers.append(placed)

    partial_n = random.randint(1, 2)
    for i in range(partial_n):
        dirs = random.sample(PLUS_DIRS, random.choice([2, 3]))
        if i == 0 or random.random() < 0.55:
            center_color = random.choice(rect_colors)
        else:
            center_color = random.choice(spare_colors)
        if random.random() < 0.7:
            arm_color = random.choice(rect_colors)
        else:
            arm_color = random.choice(spare_colors)
        placed = _place_marker(grid, center_color, arm_color, dirs)
        if placed is None:
            return None
        markers.append(placed)

    return {
        "input": grid,
        "rect_input": rect_input,
        "expected": expected,
        "rects": rects,
        "markers": markers,
        "input_order": input_order,
        "output_order": output_order,
    }


def _validate_scene(scene: dict) -> bool:
    inp = np.asarray(scene["input"], dtype=int)
    expected = np.asarray(scene["expected"], dtype=int)
    rect_input = np.asarray(scene["rect_input"], dtype=int)
    rects = list(scene["rects"])
    markers = list(scene["markers"])

    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.array_equal(inp, rect_input):
        return False
    if np.array_equal(rect_input, expected):
        return False
    if scene["input_order"] == scene["output_order"]:
        return False
    if len({int(rect["votes"]) for rect in rects}) != len(rects):
        return False
    if not any(marker["complete"] for marker in markers):
        return False
    if not any((not marker["complete"]) for marker in markers):
        return False

    stripped_expected = rect_input
    extracted, stripped = _extract_markers(inp)
    if not np.array_equal(stripped, stripped_expected):
        return False

    true_counts = Counter()
    for marker in markers:
        if marker["complete"]:
            true_counts[int(marker["center_color"])] += 1
    for rect in rects:
        if true_counts[int(rect["color"])] != int(rect["votes"]):
            return False

    for rect in rects:
        color = int(rect["color"])
        bbox = tuple(rect["bbox"])
        mask = rect_input == color
        if not _visible_bbox_matches(mask, bbox):
            return False

    if not any(int((rect_input == int(rect["color"])).sum()) < (rect["bbox"][2] - rect["bbox"][0] + 1) * (rect["bbox"][3] - rect["bbox"][1] + 1) for rect in rects):
        return False

    if any(_marker_signature(rect_input, coords) is not None for coords in _component_coords(rect_input != 0)):
        return False

    return np.array_equal(main(inp), expected) and len(extracted) == len(markers)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        scene = _build_scene()
        if scene is None:
            continue
        if _validate_scene(scene):
            return np.asarray(scene["input"], dtype=int)
    raise RuntimeError("failed to generate a valid non-degenerate scene")
