from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, find_components, get_bg, in_bounds


PATTERN_COLORS = [1, 3, 4, 6, 7, 8, 9]
MARKER_COLOR = Color.GREY
BLOCKER_CHOICES_EVEN = [0, 2, 4]
BLOCKER_CHOICES_ODD = [1, 3]
GRID_HEIGHTS = list(range(18, 25))
GRID_WIDTHS = list(range(18, 25))
MARKER_DIMS = [(2, 3), (3, 2), (3, 4), (4, 3)]
MAX_GENERATION_ATTEMPTS = 500


def _has_complete_square_border(g: np.ndarray, obj: dict) -> bool:
    r0, c0, r1, c1 = map(int, obj["bbox"])
    h = bbox_h(obj["bbox"])
    w = bbox_w(obj["bbox"])
    if h != w or h < 2:
        return False
    color = int(obj["color"])
    return bool(
        np.all(g[r0, c0 : c1 + 1] == color)
        and np.all(g[r1, c0 : c1 + 1] == color)
        and np.all(g[r0 : r1 + 1, c0] == color)
        and np.all(g[r0 : r1 + 1, c1] == color)
    )


def _corner_marker_coords(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    return [
        (r0 - 1, c0),
        (r0, c0 - 1),
        (r0 - 1, c1),
        (r0, c1 + 1),
        (r1 + 1, c0),
        (r1, c0 - 1),
        (r1 + 1, c1),
        (r1, c1 + 1),
    ]


def _solid_marker_bbox(g: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for obj in find_components(
        g,
        bg=int(get_bg(g)),
        connectivity=4,
        monochromatic=True,
        colors=[MARKER_COLOR],
    ):
        r0, c0, r1, c1 = map(int, obj["bbox"])
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if int(obj["area"]) == area and bool(np.all(g[r0 : r1 + 1, c0 : c1 + 1] == MARKER_COLOR)):
            candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, "expected one solid grey marker rectangle"
    return candidates[0]


def _pattern_bbox(g: np.ndarray, marker_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = marker_bbox
    marker_mask = np.zeros_like(g, dtype=bool)
    marker_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    matches = []
    for obj in find_components(g, bg=int(get_bg(g)), connectivity=4, monochromatic=False):
        coords = obj["coords"]
        if bool(marker_mask[coords[:, 0], coords[:, 1]].any()):
            matches.append(tuple(map(int, obj["bbox"])))
    assert len(matches) == 1, "expected one foreground component containing the marker"
    return matches[0]


def _reflection_value(
    g: np.ndarray,
    bbox: tuple[int, int, int, int],
    r: int,
    c: int,
    axis: str,
) -> int:
    r0, c0, r1, c1 = map(int, bbox)
    if axis == "horizontal":
        rr, cc = r0 + r1 - int(r), int(c)
    else:
        rr, cc = int(r), c0 + c1 - int(c)
    assert int(g[rr, cc]) != MARKER_COLOR
    return int(g[rr, cc])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count corner marks from complete squares, then use that parity to choose the symmetry axis that restores the hidden patch.

    Input structure:
    - A black background contains several small monochrome objects, of which some are complete filled squares and some are non-square distractors.
    - A larger central multicolor pattern is partially covered by one solid grey rectangle.
    - The grey rectangle lies inside the same connected foreground pattern component whose bbox defines the reflection frame.

    Rule:
    - Detect each monochrome 4-connected component whose bbox is a square of side at least 2 and whose bbox border is fully that component color.
    - For every such square, paint red on each in-bounds corner-outside orthogonal cell that is still background, and count how many new red marks are placed.
    - If that count is even, refill every grey cell from its horizontal mirror across the pattern bbox; if odd, refill from the vertical mirror.

    Key insight:
    - The scattered squares do not affect the central pattern directly; only the parity of their successful corner marks chooses which reflection axis reveals the occluded patch.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    mark_count = 0
    for obj in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        if _has_complete_square_border(g, obj):
            for r, c in _corner_marker_coords(obj["bbox"]):
                if in_bounds(g.shape, r, c) and int(g[r, c]) == bg:
                    out[r, c] = Color.RED
                    mark_count += 1

    marker_bbox = _solid_marker_bbox(g)
    pattern_bbox = _pattern_bbox(g, marker_bbox)
    axis = "horizontal" if mark_count % 2 == 0 else "vertical"

    r0, c0, r1, c1 = marker_bbox
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            out[r, c] = _reflection_value(g, pattern_bbox, r, c, axis)

    return out


def _mirror_patch(patch: np.ndarray, axis: str) -> np.ndarray:
    h, w = map(int, patch.shape)
    out = patch.copy()
    if axis == "horizontal":
        for r in range(h):
            out[h - 1 - r, :] = out[r, :]
    else:
        for c in range(w):
            out[:, w - 1 - c] = out[:, c]
    return out


def _sample_pattern_patch(h: int, w: int, axis: str, palette: list[int]) -> np.ndarray:
    for _ in range(200):
        if axis == "horizontal":
            seed = np.array(
                [[random.choice(palette) for _ in range(w)] for _ in range((h + 1) // 2)],
                dtype=int,
            )
            patch = np.empty((h, w), dtype=int)
            patch[: seed.shape[0], :] = seed
            for r in range(seed.shape[0], h):
                patch[r, :] = patch[h - 1 - r, :]
        else:
            seed = np.array(
                [[random.choice(palette) for _ in range((w + 1) // 2)] for _ in range(h)],
                dtype=int,
            )
            patch = np.empty((h, w), dtype=int)
            patch[:, : seed.shape[1]] = seed
            for c in range(seed.shape[1], w):
                patch[:, c] = patch[:, w - 1 - c]
        used = sorted(set(map(int, patch.flatten())))
        if len(used) < 3:
            continue
        if axis == "horizontal" and np.array_equal(patch, patch[:, ::-1]):
            continue
        if axis == "vertical" and np.array_equal(patch, patch[::-1, :]):
            continue
        return patch
    raise AssertionError("failed to sample a non-degenerate pattern patch")


def _marker_options(pattern_h: int, pattern_w: int, axis: str) -> list[tuple[int, int, int, int]]:
    options = []
    for mh, mw in MARKER_DIMS:
        max_r = pattern_h // 2 - mh
        max_c = pattern_w // 2 - mw
        if max_r < 1 or max_c < 1:
            continue
        for rr in range(1, max_r + 1):
            for cc in range(1, max_c + 1):
                cells = [(rr + dr, cc + dc) for dr in range(mh) for dc in range(mw)]
                ok = True
                for r, c in cells:
                    hr = pattern_h - 1 - r
                    vc = pattern_w - 1 - c
                    if rr <= hr < rr + mh and cc <= c < cc + mw:
                        ok = False
                    if rr <= r < rr + mh and cc <= vc < cc + mw:
                        ok = False
                if ok:
                    options.append((rr, cc, mh, mw))
    return options


def _choose_pattern_and_marker(axis: str) -> tuple[np.ndarray, tuple[int, int, int, int], np.ndarray]:
    for _ in range(200):
        pattern_h = random.choice([7, 9])
        pattern_w = random.choice([7, 9])
        palette = random.sample(PATTERN_COLORS, k=random.choice([3, 4]))
        patch = _sample_pattern_patch(pattern_h, pattern_w, axis, palette)
        options = _marker_options(pattern_h, pattern_w, axis)
        random.shuffle(options)
        for rr, cc, mh, mw in options:
            hidden = patch[rr : rr + mh, cc : cc + mw]
            if axis == "horizontal":
                wrong = patch[rr : rr + mh, pattern_w - cc - mw : pattern_w - cc]
            else:
                wrong = patch[pattern_h - rr - mh : pattern_h - rr, cc : cc + mw]
            if np.array_equal(hidden, wrong):
                continue
            clean = patch.copy()
            clean[rr : rr + mh, cc : cc + mw] = MARKER_COLOR
            return patch, (rr, cc, rr + mh - 1, cc + mw - 1), clean
    raise AssertionError("failed to place a discriminative marker rectangle")


def _square_cells(r0: int, c0: int, size: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(r0, r0 + size) for c in range(c0, c0 + size)]


def _place_square(
    grid: np.ndarray,
    occupied: np.ndarray,
    color: int,
    size: int,
    pattern_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    h, w = map(int, grid.shape)
    pr0, pc0, pr1, pc1 = pattern_bbox
    candidates = []
    for r0 in range(2, h - size - 1):
        for c0 in range(2, w - size - 1):
            r1 = r0 + size - 1
            c1 = c0 + size - 1
            if not (r1 < pr0 - 2 or r0 > pr1 + 2 or c1 < pc0 - 2 or c0 > pc1 + 2):
                continue
            rr0 = r0 - 1
            cc0 = c0 - 1
            rr1 = r1 + 1
            cc1 = c1 + 1
            if np.any(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1]):
                continue
            candidates.append((r0, c0, r1, c1))
    if not candidates:
        return None
    r0, c0, r1, c1 = random.choice(candidates)
    grid[r0 : r1 + 1, c0 : c1 + 1] = color
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True
    return r0, c0, r1, c1


def _blocker_templates(square_bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> list[dict]:
    r0, c0, r1, c1 = map(int, square_bbox)
    templates = [
        {"cells": [(r0 - 1, c) for c in range(c0, c1 + 1)], "count": 2},
        {"cells": [(r1 + 1, c) for c in range(c0, c1 + 1)], "count": 2},
        {"cells": [(r, c0 - 1) for r in range(r0, r1 + 1)], "count": 2},
        {"cells": [(r, c1 + 1) for r in range(r0, r1 + 1)], "count": 2},
        {"cells": [(r0 - 1, c0), (r0 - 2, c0)], "count": 1},
        {"cells": [(r0, c0 - 1), (r0, c0 - 2)], "count": 1},
        {"cells": [(r0 - 1, c1), (r0 - 2, c1)], "count": 1},
        {"cells": [(r0, c1 + 1), (r0, c1 + 2)], "count": 1},
        {"cells": [(r1 + 1, c0), (r1 + 2, c0)], "count": 1},
        {"cells": [(r1, c0 - 1), (r1, c0 - 2)], "count": 1},
        {"cells": [(r1 + 1, c1), (r1 + 2, c1)], "count": 1},
        {"cells": [(r1, c1 + 1), (r1, c1 + 2)], "count": 1},
    ]
    valid = []
    for template in templates:
        cells = [(int(r), int(c)) for r, c in template["cells"]]
        if all(in_bounds(shape, r, c) for r, c in cells):
            valid.append({"cells": cells, "count": int(template["count"])})
    return valid


def _select_blockers(
    grid: np.ndarray,
    occupied: np.ndarray,
    square_bboxes: list[tuple[int, int, int, int]],
    target_blocked: int,
    available_colors: list[int],
) -> list[dict] | None:
    all_templates = []
    for bbox in square_bboxes:
        for template in _blocker_templates(bbox, grid.shape):
            if any(occupied[r, c] for r, c in template["cells"]):
                continue
            all_templates.append(template)

    combos: list[list[dict]] = []
    if target_blocked == 0:
        combos.append([])
    for template in all_templates:
        if template["count"] == target_blocked:
            combos.append([template])
    for i, first in enumerate(all_templates):
        for second in all_templates[i + 1 :]:
            cells1 = set(first["cells"])
            cells2 = set(second["cells"])
            if cells1 & cells2:
                continue
            if first["count"] + second["count"] == target_blocked:
                combos.append([first, second])
    random.shuffle(combos)
    for combo in combos:
        if len(combo) > len(available_colors):
            continue
        return combo
    return None


def _paint_blockers(
    grid: np.ndarray,
    occupied: np.ndarray,
    blockers: list[dict],
    color_pool: list[int],
) -> list[int]:
    used = []
    for blocker, color in zip(blockers, color_pool):
        for r, c in blocker["cells"]:
            assert int(grid[r, c]) == Color.BLACK
            grid[r, c] = int(color)
            occupied[r, c] = True
        used.append(int(color))
    return used


def _remote_distractor_shapes() -> list[np.ndarray]:
    return [
        np.array([[1, 1, 1]], dtype=int),
        np.array([[1], [1], [1]], dtype=int),
        np.array([[1, 0], [1, 1]], dtype=int),
        np.array([[0, 1], [1, 1]], dtype=int),
    ]


def _place_remote_distractor(
    grid: np.ndarray,
    occupied: np.ndarray,
    color: int,
    pattern_bbox: tuple[int, int, int, int],
) -> bool:
    pr0, pc0, pr1, pc1 = pattern_bbox
    shapes = _remote_distractor_shapes()
    random.shuffle(shapes)
    for mask in shapes:
        h, w = map(int, mask.shape)
        for _ in range(80):
            r0 = random.randint(1, grid.shape[0] - h - 1)
            c0 = random.randint(1, grid.shape[1] - w - 1)
            r1 = r0 + h - 1
            c1 = c0 + w - 1
            if not (r1 < pr0 - 2 or r0 > pr1 + 2 or c1 < pc0 - 2 or c0 > pc1 + 2):
                continue
            rr0 = r0 - 1
            cc0 = c0 - 1
            rr1 = r1 + 1
            cc1 = c1 + 1
            region = occupied[rr0 : rr1 + 1, cc0 : cc1 + 1]
            mask_h, mask_w = region.shape
            if mask_h != h + 2 or mask_w != w + 2:
                continue
            if np.any(region):
                continue
            for dr in range(h):
                for dc in range(w):
                    if int(mask[dr, dc]):
                        grid[r0 + dr, c0 + dc] = int(color)
                        occupied[r0 + dr, c0 + dc] = True
            return True
    return False


def _count_marks(g: np.ndarray) -> int:
    bg = int(get_bg(g))
    total = 0
    for obj in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        if _has_complete_square_border(g, obj):
            for r, c in _corner_marker_coords(obj["bbox"]):
                if in_bounds(g.shape, r, c) and int(g[r, c]) == bg:
                    total += 1
    return total


def _expected_output(g: np.ndarray, clean_pattern: np.ndarray, pattern_bbox: tuple[int, int, int, int]) -> np.ndarray:
    out = g.copy()
    bg = int(get_bg(g))
    mark_count = 0
    for obj in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        if _has_complete_square_border(g, obj):
            for r, c in _corner_marker_coords(obj["bbox"]):
                if in_bounds(g.shape, r, c) and int(g[r, c]) == bg:
                    out[r, c] = Color.RED
                    mark_count += 1
    pr0, pc0, pr1, pc1 = pattern_bbox
    marker_bbox = _solid_marker_bbox(g)
    mr0, mc0, mr1, mc1 = marker_bbox
    out[mr0 : mr1 + 1, mc0 : mc1 + 1] = clean_pattern[mr0 - pr0 : mr1 - pr0 + 1, mc0 - pc0 : mc1 - pc0 + 1]
    assert (mark_count % 2 == 0) == np.array_equal(
        clean_pattern[mr0 - pr0 : mr1 - pr0 + 1, mc0 - pc0 : mc1 - pc0 + 1],
        _axis_fill(g, pattern_bbox, marker_bbox, "horizontal"),
    )
    return out


def _axis_fill(
    g: np.ndarray,
    pattern_bbox: tuple[int, int, int, int],
    marker_bbox: tuple[int, int, int, int],
    axis: str,
) -> np.ndarray:
    r0, c0, r1, c1 = marker_bbox
    out = np.empty((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for rr, r in enumerate(range(r0, r1 + 1)):
        for cc, c in enumerate(range(c0, c1 + 1)):
            out[rr, cc] = _reflection_value(g, pattern_bbox, r, c, axis)
    return out


def _generate_single() -> np.ndarray | None:
    grid = np.zeros((random.choice(GRID_HEIGHTS), random.choice(GRID_WIDTHS)), dtype=int)

    axis = random.choice(["horizontal", "vertical"])
    target_blocked = random.choice(BLOCKER_CHOICES_EVEN if axis == "horizontal" else BLOCKER_CHOICES_ODD)
    clean_patch, marker_rel_bbox, visible_patch = _choose_pattern_and_marker(axis)

    ph, pw = map(int, clean_patch.shape)
    pr0 = grid.shape[0] // 2 - ph // 2 + random.choice([-1, 0, 1])
    pc0 = grid.shape[1] // 2 - pw // 2 + random.choice([-1, 0, 1])
    pr0 = max(2, min(pr0, grid.shape[0] - ph - 2))
    pc0 = max(2, min(pc0, grid.shape[1] - pw - 2))
    pr1 = pr0 + ph - 1
    pc1 = pc0 + pw - 1
    grid[pr0 : pr1 + 1, pc0 : pc1 + 1] = visible_patch

    occupied = grid != Color.BLACK
    square_count = random.randint(2, 4)
    palette = random.sample(PATTERN_COLORS, k=len(PATTERN_COLORS))
    square_colors = palette[:square_count]
    spare_colors = palette[square_count:]

    square_bboxes = []
    for color in square_colors:
        placed = None
        for _ in range(120):
            size = random.randint(2, 4)
            placed = _place_square(grid, occupied, int(color), size, (pr0, pc0, pr1, pc1))
            if placed is not None:
                square_bboxes.append(placed)
                break
        if placed is None:
            return None

    blockers = _select_blockers(grid, occupied, square_bboxes, target_blocked, spare_colors)
    if blockers is None:
        return None
    blocker_colors = spare_colors[: max(1, len(blockers))]
    _paint_blockers(grid, occupied, blockers, blocker_colors)

    if len(blockers) < 2:
        extra_color_candidates = [c for c in PATTERN_COLORS if c not in square_colors and c not in blocker_colors]
        random.shuffle(extra_color_candidates)
        if extra_color_candidates and random.random() < 0.85:
            _place_remote_distractor(grid, occupied, extra_color_candidates[0], (pr0, pc0, pr1, pc1))

    marker_bbox = (
        pr0 + marker_rel_bbox[0],
        pc0 + marker_rel_bbox[1],
        pr0 + marker_rel_bbox[2],
        pc0 + marker_rel_bbox[3],
    )
    try:
        correct_fill = clean_patch[
            marker_rel_bbox[0] : marker_rel_bbox[2] + 1,
            marker_rel_bbox[1] : marker_rel_bbox[3] + 1,
        ]
        wrong_axis = "vertical" if axis == "horizontal" else "horizontal"
        wrong_fill = _axis_fill(grid, (pr0, pc0, pr1, pc1), marker_bbox, wrong_axis)

        if np.array_equal(correct_fill, wrong_fill):
            return None

        mark_count = _count_marks(grid)
        if axis == "horizontal" and mark_count % 2 != 0:
            return None
        if axis == "vertical" and mark_count % 2 != 1:
            return None

        square_hits = [
            obj
            for obj in find_components(grid, bg=Color.BLACK, connectivity=4, monochromatic=True)
            if _has_complete_square_border(grid, obj)
        ]
        if len(square_hits) != square_count:
            return None

        expected = grid.copy()
        for obj in square_hits:
            for r, c in _corner_marker_coords(obj["bbox"]):
                if in_bounds(grid.shape, r, c) and int(grid[r, c]) == Color.BLACK:
                    expected[r, c] = Color.RED
        mr0, mc0, mr1, mc1 = marker_bbox
        expected[mr0 : mr1 + 1, mc0 : mc1 + 1] = correct_fill

        if not np.array_equal(main(grid), expected):
            return None

        if np.array_equal(expected, grid):
            return None
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_single()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
