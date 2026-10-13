from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, bbox_h, bbox_w, components_mask, crop_bbox, paste


BACKGROUND = Color.BLACK
STAMP_COLOR = Color.GREY
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (18, 26)
TEMPLATE_SHAPES = [(3, 3), (3, 4), (4, 3), (4, 4)]
BOX_GAP = 2
GEN_ATTEMPTS = 400

BRANCHES = [
    (1, "same"),
    (1, "swapped"),
    (2, "swapped"),
    (3, "same"),
    (3, "swapped"),
]


def _pattern_key(arr: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(x) for x in row) for row in np.asarray(arr, dtype=int))


def _swap_two_colors(arr: np.ndarray, a: int, b: int) -> np.ndarray:
    out = np.asarray(arr, dtype=int).copy()
    mask_a = out == int(a)
    mask_b = out == int(b)
    out[mask_a] = int(b)
    out[mask_b] = int(a)
    return out


def _center2(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, box)
    return r0 + r1, c0 + c1


def _avg_center_distance(
    box: tuple[int, int, int, int], others: list[tuple[int, int, int, int]]
) -> float:
    cr, cc = _center2(box)
    return sum(abs(cr - orr) + abs(cc - occ) for orr, occ in map(_center2, others)) / len(others)


def _boxes_clear(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = BOX_GAP
) -> bool:
    return (
        a[2] + int(gap) < b[0]
        or b[2] + int(gap) < a[0]
        or a[3] + int(gap) < b[1]
        or b[3] + int(gap) < a[1]
    )


def _box_from_pos(r: int, c: int, h: int, w: int) -> tuple[int, int, int, int]:
    return int(r), int(c), int(r) + int(h) - 1, int(c) + int(w) - 1


def _frontier_cuts(count: int, limit: int) -> list[int]:
    cut = random.randint(1, limit - 1)
    cuts = []
    for _ in range(int(count)):
        cut = max(1, min(limit - 1, cut + random.choice((-1, 0, 1))))
        cuts.append(cut)
    if random.random() < 0.5:
        cuts.reverse()
    return cuts


def _row_frontier_template(th: int, tw: int, a: int, b: int) -> np.ndarray:
    cuts = np.array(_frontier_cuts(th, tw))[:, None]
    return np.where(np.arange(tw) < cuts, a, b).astype(int)


def _col_frontier_template(th: int, tw: int, a: int, b: int) -> np.ndarray:
    cuts = np.array(_frontier_cuts(tw, th))[None, :]
    return np.where(np.arange(th)[:, None] < cuts, a, b).astype(int)


def _corner_region_template(th: int, tw: int, a: int, b: int) -> np.ndarray:
    rr, cc = np.indices((th, tw))
    corners = [(0, 0), (0, tw - 1), (th - 1, 0), (th - 1, tw - 1)]
    first, second = random.sample(corners, 2)
    jitter = np.random.uniform(-0.45, 0.45, size=(th, tw))
    d_first = np.abs(rr - first[0]) + np.abs(cc - first[1]) + jitter
    d_second = np.abs(rr - second[0]) + np.abs(cc - second[1]) - jitter
    return np.where(d_first <= d_second, a, b).astype(int)


def _diag_band_template(th: int, tw: int, a: int, b: int) -> np.ndarray:
    rr, cc = np.indices((th, tw))
    offset = random.choice((-1, 0, 1))
    return np.where((rr - cc + offset) >= 0, a, b).astype(int)


def _make_template(th: int, tw: int, colors: tuple[int, int]) -> np.ndarray:
    a, b = map(int, colors)
    maker = random.choice(
        [_row_frontier_template, _col_frontier_template, _corner_region_template, _diag_band_template]
    )
    patch = maker(int(th), int(tw), a, b)
    if random.random() < 0.5:
        patch = np.flipud(patch)
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    if patch.shape[0] == patch.shape[1] and random.random() < 0.5:
        patch = np.rot90(patch, random.randint(0, 3))
    if len(set(map(int, patch.reshape(-1)))) != 2:
        patch[0, 0] = a
        patch[-1, -1] = b
    return patch.astype(int)


def _rect_objects(grid: np.ndarray) -> list[dict]:
    comps = components_mask(np.asarray(grid) != int(BACKGROUND), connectivity=4)
    objects: list[dict] = []
    for coords in comps:
        box = bbox_coords(coords)
        h = bbox_h(box)
        w = bbox_w(box)
        if int(coords.shape[0]) != int(h * w):
            continue
        patch = crop_bbox(grid, box)
        colors = sorted(set(map(int, patch.reshape(-1))))
        objects.append(
            {
                "coords": coords,
                "bbox": box,
                "shape": (int(h), int(w)),
                "patch": patch,
                "colors": colors,
            }
        )
    return objects


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    objects = _rect_objects(g)
    assert objects, "expected rectangular foreground objects"

    grouped: dict[tuple[int, int], list[dict]] = {}
    for obj in objects:
        grouped.setdefault(obj["shape"], []).append(obj)

    matches = []
    for shape, members in grouped.items():
        grey_targets = [obj for obj in members if obj["colors"] == [int(STAMP_COLOR)]]
        colored = [
            obj
            for obj in members
            if int(STAMP_COLOR) not in obj["colors"]
            and int(BACKGROUND) not in obj["colors"]
            and len(obj["colors"]) == 2
        ]
        if grey_targets and len(colored) >= 2:
            matches.append((shape, grey_targets, colored))

    assert len(matches) == 1, "expected exactly one shape family with stamp slots and colored rectangles"
    shape, grey_targets, colored = matches[0]
    target_boxes = [obj["bbox"] for obj in grey_targets]
    scores = [_avg_center_distance(obj["bbox"], target_boxes) for obj in colored]
    best = min(scores)
    assert sum(score == best for score in scores) == 1, "template must be uniquely closest to grey targets"
    template = colored[scores.index(best)]
    distractors = [obj for obj in colored if obj is not template]
    assert distractors, "expected at least one distractor rectangle"

    return {
        "shape": shape,
        "grey_targets": grey_targets,
        "template": template,
        "distractors": distractors,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the source rectangle into grey slots, swap colors in the stamped copies, then extract the minority pattern.

    Input structure:
    - Black background surrounds several fully occupied same-size rectangles.
    - One or more rectangles are solid grey stamp targets.
    - At least two same-size colored rectangles use exactly two non-black, non-grey colors; the source template is the one closest to the grey target group.

    Rule:
    - Identify the colored source template by its smaller average center distance to the grey rectangles.
    - Copy that template into every grey rectangle, then swap the template's two colors inside each stamped copy only.
    - Compare the resulting pattern multiset: original template, all swapped copies, and the pre-colored distractor rectangles already present.
    - Return the single pattern whose frequency is strictly smaller than the other pattern's frequency.

    Key insight:
    - The source template is determined spatially, but the output is determined only after combining three coupled steps: stamping creates a camp of copies, color swapping flips that camp, and the minority-vs-majority comparison decides which pattern to crop out.
    """
    g = np.asarray(grid, dtype=int)
    scene = _parse_scene(g)

    template = scene["template"]["patch"]
    a, b = map(int, scene["template"]["colors"])
    swapped = _swap_two_colors(template, a, b)

    patterns = [template] + [swapped] * len(scene["grey_targets"])
    patterns.extend(obj["patch"] for obj in scene["distractors"])

    counts = Counter(_pattern_key(pattern) for pattern in patterns)
    min_count = min(counts.values())
    winners = [key for key, count in counts.items() if count == min_count]
    assert len(winners) == 1, "expected exactly one minority pattern"
    return np.array(winners[0], dtype=int)


def _choose_targets(
    candidates: list[tuple[int, int]],
    template_box: tuple[int, int, int, int],
    th: int,
    tw: int,
    n_targets: int,
) -> list[tuple[int, int]] | None:
    near = []
    for r, c in candidates:
        box = _box_from_pos(r, c, th, tw)
        if not _boxes_clear(box, template_box):
            continue
        dist = _avg_center_distance(box, [template_box])
        if 6 <= dist <= 22:
            near.append((r, c))

    for _ in range(80):
        random.shuffle(near)
        chosen: list[tuple[int, int]] = []
        chosen_boxes: list[tuple[int, int, int, int]] = []
        for r, c in near:
            box = _box_from_pos(r, c, th, tw)
            if all(_boxes_clear(box, other) for other in chosen_boxes):
                chosen.append((r, c))
                chosen_boxes.append(box)
            if len(chosen) == int(n_targets):
                return chosen
    return None


def _choose_layout(
    height: int, width: int, th: int, tw: int, n_targets: int
) -> tuple[tuple[int, int], list[tuple[int, int]], tuple[int, int]] | None:
    candidates = [(r, c) for r in range(1, height - th) for c in range(1, width - tw)]
    template_candidates = [
        (r, c)
        for r, c in candidates
        if 2 <= r <= height - th - 3 and 2 <= c <= width - tw - 3
    ]
    random.shuffle(template_candidates)

    for tr, tc in template_candidates[:120]:
        template_box = _box_from_pos(tr, tc, th, tw)
        targets = _choose_targets(candidates, template_box, th, tw, n_targets)
        if targets is None:
            continue

        target_boxes = [_box_from_pos(r, c, th, tw) for r, c in targets]
        template_score = _avg_center_distance(template_box, target_boxes)
        far = []
        for dr, dc in candidates:
            distractor_box = _box_from_pos(dr, dc, th, tw)
            if not _boxes_clear(distractor_box, template_box):
                continue
            if not all(_boxes_clear(distractor_box, box) for box in target_boxes):
                continue
            distractor_score = _avg_center_distance(distractor_box, target_boxes)
            if distractor_score >= template_score + 8:
                far.append((dr, dc, distractor_score))

        if not far:
            continue
        far.sort(key=lambda item: (-item[2], item[0], item[1]))
        dr, dc, _score = random.choice(far[: max(1, min(10, len(far)))])
        return (tr, tc), targets, (dr, dc)
    return None


def _distractor_relation(
    patch: np.ndarray, template: np.ndarray, swapped: np.ndarray
) -> str | None:
    if np.array_equal(patch, template):
        return "same"
    if np.array_equal(patch, swapped):
        return "swapped"
    return None


def _branch_expected(template: np.ndarray, relation: str, n_targets: int) -> np.ndarray:
    a, b = sorted(set(map(int, template.reshape(-1))))
    swapped = _swap_two_colors(template, a, b)
    patterns = [template] + [swapped] * int(n_targets)
    patterns.append(template if relation == "same" else swapped)
    counts = Counter(_pattern_key(pattern) for pattern in patterns)
    min_count = min(counts.values())
    winners = [key for key, count in counts.items() if count == min_count]
    assert len(winners) == 1
    return np.array(winners[0], dtype=int)


def _gen_attempt() -> np.ndarray | None:
    n_targets, relation = random.choice(BRANCHES)
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    th, tw = random.choice(TEMPLATE_SHAPES)
    layout = _choose_layout(height, width, th, tw, n_targets)
    if layout is None:
        return None

    template_pos, target_positions, distractor_pos = layout
    a, b = random.sample(PALETTE, 2)
    template = _make_template(th, tw, (a, b))
    swapped = _swap_two_colors(template, a, b)
    distractor = template if relation == "same" else swapped

    grid = np.full((height, width), int(BACKGROUND), dtype=int)
    paste(grid, template, template_pos[0], template_pos[1])
    grey_patch = np.full((th, tw), int(STAMP_COLOR), dtype=int)
    for r, c in target_positions:
        paste(grid, grey_patch, r, c)
    paste(grid, distractor, distractor_pos[0], distractor_pos[1])

    try:
        scene = _parse_scene(grid)
    except AssertionError:
        return None

    parsed_template = scene["template"]["patch"]
    parsed_targets = scene["grey_targets"]
    parsed_distractors = scene["distractors"]
    if len(parsed_targets) != int(n_targets) or len(parsed_distractors) != 1:
        return None

    parsed_a, parsed_b = scene["template"]["colors"]
    parsed_swapped = _swap_two_colors(parsed_template, parsed_a, parsed_b)
    parsed_relation = _distractor_relation(parsed_distractors[0]["patch"], parsed_template, parsed_swapped)
    if parsed_relation != relation:
        return None

    if np.array_equal(parsed_template, parsed_swapped):
        return None

    expected = _branch_expected(parsed_template, relation, n_targets)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    score_template = _avg_center_distance(scene["template"]["bbox"], [t["bbox"] for t in parsed_targets])
    score_distractor = _avg_center_distance(parsed_distractors[0]["bbox"], [t["bbox"] for t in parsed_targets])
    if not (score_template + 8 <= score_distractor):
        return None

    all_patterns = [parsed_template] + [parsed_swapped] * int(n_targets) + [parsed_distractors[0]["patch"]]
    counts = Counter(_pattern_key(pattern) for pattern in all_patterns)
    if len(counts) != 2:
        return None
    ranked = sorted(counts.values())
    if ranked[0] == ranked[1]:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
