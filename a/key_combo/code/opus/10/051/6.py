from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

LEGEND_PALETTE = (BLUE, GREEN, YELLOW, PINK, ORANGE, TEAL, MAROON)

HOLLOW_SHAPES = (
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]], dtype=bool),
)

SOLID_SHAPES = (
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1, 1], [0, 1, 0, 0], [0, 1, 0, 0]], dtype=bool),
)


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(comp, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _shape_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    pts = sorted((int(r - r0), int(c - c0)) for r, c in coords)
    return tuple(pts)


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _hole_cells(coords: np.ndarray) -> list[tuple[int, int]]:
    mask = _mask_from_coords(coords)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if not mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        rr, cc = q.popleft()
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    r0, c0, _, _ = _bbox(coords)
    holes: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] and not seen[r, c]:
                holes.append((r0 + r, c0 + c))
    return holes


def _gray_objects(grid: np.ndarray) -> list[dict]:
    objs = []
    for coords in _components(grid == GREY):
        objs.append(
            {
                "coords": coords,
                "bbox": _bbox(coords),
                "shape_key": _shape_key(coords),
                "holes": _hole_cells(coords),
            }
        )
    return objs


def _legend_components(grid: np.ndarray) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) not in {BG, RED, GREY}):
        for coords in _components(grid == color):
            comps.append({"color": color, "bbox": _bbox(coords), "coords": coords})
    comps.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return comps


def _has_corner_mark(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    corners = ((r0 - 1, c0 - 1), (r0 - 1, c1 + 1), (r1 + 1, c0 - 1), (r1 + 1, c1 + 1))
    return all(0 <= r < h and 0 <= c < w and int(grid[r, c]) == RED for r, c in corners)


def _analyze(grid: np.ndarray) -> dict:
    objects = _gray_objects(grid)
    assert 6 <= len(objects) <= 8, "expected 6-8 gray objects"
    grouped: dict[tuple[tuple[int, int], ...], list[dict]] = {}
    for obj in objects:
        grouped.setdefault(obj["shape_key"], []).append(obj)
    groups: list[dict] = []
    for shape_key, members in grouped.items():
        anchor = min((m["bbox"][0], m["bbox"][1]) for m in members)
        groups.append(
            {
                "shape_key": shape_key,
                "objects": members,
                "anchor": anchor,
                "marked": any(_has_corner_mark(grid, m["bbox"]) for m in members),
            }
        )
    groups.sort(key=lambda g: (-len(g["objects"]), g["anchor"][0], g["anchor"][1]))

    legends = _legend_components(grid)
    assert len(legends) == len(groups), "legend row count must equal group count"
    for group, legend in zip(groups, legends):
        group["color"] = legend["color"]
        group["legend_bbox"] = legend["bbox"]
    return {"objects": objects, "groups": groups, "legends": legends}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor gray objects by translation-normalized shape groups, and fill holes only for groups activated by corner marks.

    Input structure:
    - A black background contains 6-8 gray 4-connected objects, partitioned into 2-3 translation-normalized shape groups.
    - Some groups are hollow, and at most one instance per group may have four red diagonal corner marks just outside its bbox corners.
    - A small legend in the lower-right uses one colored block per row; from top to bottom these rows correspond to the ranked shape groups.

    Rule:
    - Group all gray objects by translation-normalized shape, then rank the groups by instance count descending, breaking ties by the earliest bbox top-left position in row-major order.
    - Recolor every gray pixel in each group with the legend color from the matching rank row.
    - If any instance in a group has the four red corner marks, fill every enclosed black hole inside every instance of that group with the same legend color; otherwise leave holes black.
    - Keep the red corner marks and the legend pixels unchanged.

    Key insight:
    - The red corner marks are not local instructions for one object; they are a group-level switch that propagates through shape matching and decides whether all same-shape instances also receive interior filling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    info = _analyze(g)
    out = g.copy()
    for group in info["groups"]:
        color = int(group["color"])
        for obj in group["objects"]:
            coords = obj["coords"]
            out[coords[:, 0], coords[:, 1]] = color
            if group["marked"]:
                for r, c in obj["holes"]:
                    out[r, c] = color
    return out


def _instance_patch(mask: np.ndarray, marked: bool) -> np.ndarray:
    h, w = mask.shape
    patch = np.zeros((h + 2, w + 2), dtype=int)
    patch[1 : h + 1, 1 : w + 1][mask] = GREY
    if marked:
        patch[0, 0] = RED
        patch[0, w + 1] = RED
        patch[h + 1, 0] = RED
        patch[h + 1, w + 1] = RED
    return patch


def _choose_counts(rng: np.random.Generator, group_count: int) -> list[int]:
    if group_count == 2:
        return [3, 3]
    choices = ([2, 2, 2], [3, 2, 2], [3, 3, 2])
    return list(choices[int(rng.integers(len(choices)))])


def _choose_group_specs(rng: np.random.Generator) -> list[dict]:
    group_count = int(rng.integers(2, 4))
    counts = _choose_counts(rng, group_count)
    rng.shuffle(counts)

    hollow_indices = [0, 1]
    if group_count == 3 and rng.random() < 0.45:
        hollow_indices.append(2)
    marked_count = 1 if len(hollow_indices) == 2 else int(rng.integers(1, 3))
    marked_indices = set(rng.choice(hollow_indices, size=marked_count, replace=False).tolist())
    if len(marked_indices) == len(hollow_indices):
        marked_indices.remove(int(rng.choice(tuple(marked_indices))))

    hollow_order = rng.choice(len(HOLLOW_SHAPES), size=len(hollow_indices), replace=False)
    solid_order = rng.choice(len(SOLID_SHAPES), size=group_count - len(hollow_indices), replace=False)
    specs: list[dict] = []
    solid_ptr = 0
    hollow_ptr = 0
    for idx in range(group_count):
        if idx in hollow_indices:
            mask = HOLLOW_SHAPES[int(hollow_order[hollow_ptr])]
            hollow_ptr += 1
            kind = "hollow"
        else:
            mask = SOLID_SHAPES[int(solid_order[solid_ptr])]
            solid_ptr += 1
            kind = "solid"
        specs.append(
            {
                "mask": mask.copy(),
                "shape_key": _shape_key(np.argwhere(mask)),
                "count": counts[idx],
                "marked": idx in marked_indices,
                "marked_instance_index": int(rng.integers(counts[idx])) if idx in marked_indices else -1,
                "kind": kind,
                "instances": [],
            }
        )
    return specs


def _place_patch(
    rng: np.random.Generator,
    grid: np.ndarray,
    reserved: np.ndarray,
    patch: np.ndarray,
) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    for _ in range(400):
        r0 = int(rng.integers(0, h - ph + 1))
        c0 = int(rng.integers(0, w - pw + 1))
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h, r0 + ph + 1)
        cc1 = min(w, c0 + pw + 1)
        if reserved[rr0:rr1, cc0:cc1].any():
            continue
        view = grid[r0 : r0 + ph, c0 : c0 + pw]
        if np.any((patch != BG) & (view != BG)):
            continue
        view[patch != BG] = patch[patch != BG]
        reserved[rr0:rr1, cc0:cc1] = True
        return r0, c0
    return None


def _rank_groups(specs: list[dict]) -> list[dict]:
    ranked = []
    for spec in specs:
        anchor = min((inst["bbox"][0], inst["bbox"][1]) for inst in spec["instances"])
        ranked.append({"spec": spec, "anchor": anchor})
    ranked.sort(key=lambda item: (-item["spec"]["count"], item["anchor"][0], item["anchor"][1]))
    return [item["spec"] for item in ranked]


def _validate(grid: np.ndarray, specs: list[dict]) -> bool:
    try:
        info = _analyze(grid)
        out = main(grid)
    except AssertionError:
        return False

    if out.shape != grid.shape or np.any(out[grid == RED] != RED):
        return False
    if np.any(out == GREY):
        return False
    for legend in info["legends"]:
        coords = legend["coords"]
        color = int(legend["color"])
        if not np.all(out[coords[:, 0], coords[:, 1]] == color):
            return False

    expected_by_key = {spec["shape_key"]: spec for spec in specs}
    actual_by_key = {group["shape_key"]: group for group in info["groups"]}
    if expected_by_key.keys() != actual_by_key.keys():
        return False

    saw_marked_hole = False
    saw_unmarked_hole = False
    for shape_key, spec in expected_by_key.items():
        actual = actual_by_key[shape_key]
        if len(actual["objects"]) != spec["count"]:
            return False
        if bool(actual["marked"]) != bool(spec["marked"]):
            return False
        if int(actual["color"]) != int(spec["color"]):
            return False

        color = int(spec["color"])
        for inst in spec["instances"]:
            coords = inst["coords"]
            if not np.all(out[coords[:, 0], coords[:, 1]] == color):
                return False
            if inst["holes"]:
                if spec["marked"]:
                    saw_marked_hole = True
                    if any(int(out[r, c]) != color for r, c in inst["holes"]):
                        return False
                else:
                    saw_unmarked_hole = True
                    if any(int(out[r, c]) != BG for r, c in inst["holes"]):
                        return False

    return saw_marked_hole and saw_unmarked_hole


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    specs = _choose_group_specs(rng)
    group_count = len(specs)
    legend_h = group_count * 2 + (group_count - 1)
    h = int(rng.integers(max(18, legend_h + 8), 27))
    w = int(rng.integers(18, 27))
    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    legend_row0 = h - legend_h
    legend_col0 = w - 2
    reserved[max(0, legend_row0 - 1) :, max(0, legend_col0 - 2) :] = True

    order = sorted(
        (
            (spec_idx, inst_idx, specs[spec_idx]["mask"].shape[0] * specs[spec_idx]["mask"].shape[1])
            for spec_idx, spec in enumerate(specs)
            for inst_idx in range(spec["count"])
        ),
        key=lambda item: -item[2],
    )
    for spec_idx, inst_idx, _ in order:
        spec = specs[spec_idx]
        marked_instance = spec["marked"] and inst_idx == spec["marked_instance_index"]
        patch = _instance_patch(spec["mask"], marked_instance)
        pos = _place_patch(rng, grid, reserved, patch)
        if pos is None:
            return None
        pr, pc = pos
        mask = spec["mask"]
        coords = np.argwhere(mask) + np.array([pr + 1, pc + 1])
        spec["instances"].append(
            {
                "coords": coords,
                "bbox": (pr + 1, pc + 1, pr + mask.shape[0], pc + mask.shape[1]),
                "holes": _hole_cells(coords),
                "marked": marked_instance,
            }
        )

    ranked = _rank_groups(specs)
    colors = rng.choice(LEGEND_PALETTE, size=group_count, replace=False).tolist()
    for idx, spec in enumerate(ranked):
        spec["color"] = int(colors[idx])
        r0 = legend_row0 + idx * 3
        grid[r0 : r0 + 2, legend_col0 : legend_col0 + 2] = int(colors[idx])

    if not _validate(grid, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
