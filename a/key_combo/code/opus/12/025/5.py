from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np


_TOOLBOX_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX_DIR) not in sys.path:
    sys.path.append(str(_TOOLBOX_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


BG = 0
GREY = 5
MARKER = 9
TEMPLATE_COLORS = (1, 2, 3, 4, 7, 8)
TEMPLATE_SHAPES = tuple((h, w) for h in range(3, 6) for w in range(3, 5))
GRID_H_RANGE = (21, 27)
GRID_W_RANGE = (23, 30)
MAX_TEMPLATE_ATTEMPTS = 120
MAX_INSTANCE_ATTEMPTS = 400
MAX_PLACEMENT_ATTEMPTS = 40


def _classify_objects(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    bg = int(get_bg(grid))
    components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    templates: list[dict] = []
    targets: list[dict] = []
    for comp in components:
        patch = crop_bbox(grid, comp["bbox"])
        shape = tuple(map(int, patch.shape))
        entry = {
            "bbox": tuple(map(int, comp["bbox"])),
            "patch": patch,
            "shape": shape,
        }
        if np.all(patch == GREY):
            assert int(comp["area"]) == int(patch.size), "grey objects must be solid rectangles"
            targets.append(entry)
            continue
        assert np.all(patch != bg), "templates must fully occupy their bounding boxes"
        assert np.all(patch != GREY), "templates cannot contain grey cells"
        marker_count = int(np.count_nonzero(patch == MARKER))
        assert marker_count >= 1, "every template must contain at least one marker"
        entry["marker_count"] = marker_count
        templates.append(entry)
    return templates, targets


def _dominant_neighbor_color(patch: np.ndarray, r: int, c: int) -> int:
    counts: Counter[int] = Counter()
    h, w = map(int, patch.shape)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = int(r + dr)
        nc = int(c + dc)
        if 0 <= nr < h and 0 <= nc < w:
            color = int(patch[nr, nc])
            if color != MARKER:
                counts[color] += 1
    assert counts, "marker cell must see at least one non-marker neighbor"
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    assert len(winners) == 1, "marker cleanup must be unambiguous"
    return winners[0]


def _clean_template(patch: np.ndarray) -> np.ndarray:
    out = np.array(patch, copy=True)
    for r, c in np.argwhere(patch == MARKER):
        out[int(r), int(c)] = _dominant_neighbor_color(patch, int(r), int(c))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the least-marked template and stamp its cleaned pattern into same-sized grey rectangles.

    Input structure:
    - A black background contains 2-3 separated multicolor templates and 3-6 separated solid grey rectangles.
    - Every template is a fully occupied rectangle using visible colors plus 1-4 maroon marker pixels.
    - Every grey rectangle has the same size as exactly one template, and every template size appears among the grey rectangles.

    Rule:
    - Count the maroon marker pixels inside each template and choose the unique template with the fewest markers.
    - Clean that active template by replacing each marker with the unique most frequent non-marker color in its 4-neighborhood.
    - Copy the cleaned active template into every grey rectangle of the same size.
    - Erase every grey rectangle whose size matches a non-active template, while leaving all original templates unchanged.

    Key insight:
    - Marker counts do not delete the templates themselves; they select which template earns the right to be reused as a stamp.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    templates, targets = _classify_objects(g)
    assert 2 <= len(templates) <= 3, "expected 2-3 templates"
    assert 3 <= len(targets) <= 6, "expected 3-6 grey rectangles"

    marker_counts = [int(t["marker_count"]) for t in templates]
    active_count = min(marker_counts)
    assert marker_counts.count(active_count) == 1, "active template must be unique"
    active = templates[marker_counts.index(active_count)]
    clean = _clean_template(active["patch"])

    template_shapes = {tuple(t["shape"]) for t in templates}
    assert all(tuple(t["shape"]) in template_shapes for t in targets), "every grey rectangle must match a template size"

    out = g.copy()
    for target in targets:
        r0, c0, r1, c1 = target["bbox"]
        if tuple(target["shape"]) == tuple(active["shape"]):
            out[r0 : r1 + 1, c0 : c1 + 1] = clean
        else:
            out[r0 : r1 + 1, c0 : c1 + 1] = bg

    return out


def _random_partition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(1, int(total)), int(parts) - 1))
    values = [cuts[0], *[b - a for a, b in zip(cuts, cuts[1:])], int(total) - cuts[-1]]
    return [int(v) for v in values]


def _seeded_patch(shape: tuple[int, int], colors: list[int]) -> np.ndarray:
    h, w = shape
    style = random.choice(("row_split", "col_split", "quad", "voronoi"))
    patch = np.zeros((h, w), dtype=int)
    if style == "row_split":
        cut = random.randint(1, h - 1)
        patch[:cut, :] = colors[0]
        patch[cut:, :] = colors[1]
        if len(colors) == 3:
            col = random.randint(0, w - 1)
            patch[:, col] = colors[2]
    elif style == "col_split":
        cut = random.randint(1, w - 1)
        patch[:, :cut] = colors[0]
        patch[:, cut:] = colors[1]
        if len(colors) == 3:
            row = random.randint(0, h - 1)
            patch[row, :] = colors[2]
    elif style == "quad":
        rcut = random.randint(1, h - 1)
        ccut = random.randint(1, w - 1)
        ordered = (colors * 2)[:4]
        patch[:rcut, :ccut] = ordered[0]
        patch[:rcut, ccut:] = ordered[1]
        patch[rcut:, :ccut] = ordered[2]
        patch[rcut:, ccut:] = ordered[3]
    else:
        candidate_seeds = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1), (h // 2, w // 2)]
        seed_count = 2 if len(colors) == 2 else 3
        seeds = random.sample(candidate_seeds, seed_count)
        for r in range(h):
            for c in range(w):
                distances = [(abs(r - sr) + abs(c - sc), idx) for idx, (sr, sc) in enumerate(seeds)]
                patch[r, c] = colors[min(distances)[1]]
    return patch


def _marker_candidates(clean: np.ndarray) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    h, w = map(int, clean.shape)
    for r in range(h):
        for c in range(w):
            counts: Counter[int] = Counter()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    counts[int(clean[nr, nc])] += 1
            if not counts:
                continue
            best = max(counts.values())
            winners = [color for color, count in counts.items() if count == best]
            if len(winners) == 1 and winners[0] == int(clean[r, c]):
                candidates.append((int(r), int(c)))
    random.shuffle(candidates)
    return candidates


def _select_marker_cells(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in chosen):
            chosen.append((int(r), int(c)))
            if len(chosen) == int(count):
                return chosen
    return None


def _build_template(shape: tuple[int, int], marker_count: int) -> dict | None:
    h, w = shape
    for _ in range(MAX_TEMPLATE_ATTEMPTS):
        color_count = random.choice((2, 3))
        colors = random.sample(TEMPLATE_COLORS, color_count)
        clean = _seeded_patch((h, w), colors)
        if len(set(map(int, clean.flatten()))) < 2:
            continue
        candidates = _marker_candidates(clean)
        cells = _select_marker_cells(candidates, marker_count)
        if cells is None:
            continue
        dirty = np.array(clean, copy=True)
        for r, c in cells:
            dirty[r, c] = MARKER
        if not np.array_equal(_clean_template(dirty), clean):
            continue
        return {
            "shape": (int(h), int(w)),
            "clean": clean,
            "dirty": dirty,
            "marker_count": int(marker_count),
            "marker_cells": tuple(cells),
        }
    return None


def _place_objects(objects: list[dict], grid_shape: tuple[int, int]) -> list[tuple[int, int]] | None:
    h, w = grid_shape
    occupancy = np.zeros((h, w), dtype=int)
    positions: list[tuple[int, int] | None] = [None] * len(objects)
    order = sorted(range(len(objects)), key=lambda i: objects[i]["shape"][0] * objects[i]["shape"][1], reverse=True)
    for idx in order:
        oh, ow = objects[idx]["shape"]
        patch = np.ones((oh, ow), dtype=int)
        try:
            r0, c0 = random_free_location(
                occupancy,
                patch,
                transparent=0,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(occupancy, patch, r0, c0)
        positions[idx] = (int(r0), int(c0))
    return [pos for pos in positions if pos is not None]


def _render_instance(
    templates: list[dict],
    target_counts: list[int],
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    objects: list[dict] = []
    for idx, template in enumerate(templates):
        objects.append(
            {
                "kind": "template",
                "template_index": int(idx),
                "shape": tuple(template["shape"]),
                "patch": template["dirty"],
            }
        )
    for idx, target_count in enumerate(target_counts):
        for _ in range(int(target_count)):
            objects.append(
                {
                    "kind": "target",
                    "template_index": int(idx),
                    "shape": tuple(templates[idx]["shape"]),
                    "patch": np.full(templates[idx]["shape"], GREY, dtype=int),
                }
            )

    positions = _place_objects(objects, grid_shape)
    if positions is None or len(positions) != len(objects):
        return None

    grid = np.zeros(grid_shape, dtype=int)
    expected = np.zeros_like(grid)
    active_index = min(range(len(templates)), key=lambda i: int(templates[i]["marker_count"]))

    placed_templates: list[dict] = []
    placed_targets: list[dict] = []
    for obj, (r0, c0) in zip(objects, positions):
        patch = obj["patch"]
        h, w = map(int, patch.shape)
        paste(grid, patch, r0, c0)
        if obj["kind"] == "template":
            paste(expected, patch, r0, c0)
            placed_templates.append(
                {
                    "template_index": int(obj["template_index"]),
                    "shape": (h, w),
                    "pos": (int(r0), int(c0)),
                    "dirty": patch,
                    "clean": templates[obj["template_index"]]["clean"],
                    "marker_count": int(templates[obj["template_index"]]["marker_count"]),
                }
            )
        elif int(obj["template_index"]) == int(active_index):
            paste(expected, templates[active_index]["clean"], r0, c0)
            placed_targets.append(
                {
                    "template_index": int(obj["template_index"]),
                    "shape": (h, w),
                    "pos": (int(r0), int(c0)),
                }
            )
        else:
            placed_targets.append(
                {
                    "template_index": int(obj["template_index"]),
                    "shape": (h, w),
                    "pos": (int(r0), int(c0)),
                }
            )

    meta = {
        "templates": placed_templates,
        "targets": placed_targets,
        "active_index": int(active_index),
        "target_counts": [int(v) for v in target_counts],
    }
    return grid, expected, meta


def _validate_instance(grid: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    if int(get_bg(grid)) != BG:
        return False
    if len(meta["templates"]) not in (2, 3):
        return False
    if not (3 <= len(meta["targets"]) <= 6):
        return False

    marker_counts = [int(t["marker_count"]) for t in meta["templates"]]
    if marker_counts.count(min(marker_counts)) != 1:
        return False

    shapes = [tuple(t["shape"]) for t in meta["templates"]]
    if len(set(shapes)) != len(shapes):
        return False
    if sum(meta["target_counts"]) != len(meta["targets"]):
        return False
    if any(count < 1 for count in meta["target_counts"]):
        return False

    active_index = int(meta["active_index"])
    if meta["target_counts"][active_index] < 1:
        return False
    if not any(idx != active_index and count >= 1 for idx, count in enumerate(meta["target_counts"])):
        return False

    for template in meta["templates"]:
        if not np.array_equal(_clean_template(template["dirty"]), template["clean"]):
            return False

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if not np.any(actual != grid):
        return False

    active_clean = meta["templates"][active_index]["clean"]
    saw_stamp = False
    saw_erase = False
    for target in meta["targets"]:
        r0, c0 = target["pos"]
        h, w = target["shape"]
        patch = actual[r0 : r0 + h, c0 : c0 + w]
        if int(target["template_index"]) == active_index:
            if not np.array_equal(patch, active_clean):
                return False
            saw_stamp = True
        else:
            if np.any(patch != BG):
                return False
            saw_erase = True
    return saw_stamp and saw_erase


def _generate_instance() -> np.ndarray | None:
    template_count = random.choice((2, 3))
    shapes = random.sample(TEMPLATE_SHAPES, template_count)
    marker_counts = random.sample((1, 2, 3, 4), template_count)
    target_total = random.randint(max(3, template_count), 6)
    target_counts = _random_partition(target_total, template_count)

    templates: list[dict] = []
    for shape, marker_count in zip(shapes, marker_counts):
        template = _build_template(shape, marker_count)
        if template is None:
            return None
        templates.append(template)

    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        grid_shape = (
            random.randint(*GRID_H_RANGE),
            random.randint(*GRID_W_RANGE),
        )
        rendered = _render_instance(templates, target_counts, grid_shape)
        if rendered is None:
            continue
        grid, expected, meta = rendered
        if _validate_instance(grid, expected, meta):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_INSTANCE_ATTEMPTS):
        grid = _generate_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
